"""SFT 模板渲染测试（day050）：ChatML / Llama3 / Plain 三种模板的监督切分.

守的核心不变式只有一条：**字符偏移必须精确标出"从哪里开始是答案"**。
它的每一个推论（前缀里不含答案、监督区间非空、偏移之和等于总长）都
在这里被逐条断言。
"""

from __future__ import annotations

import pytest

from smart_research_agent.finetune.schema import TrainingExample
from smart_research_agent.sft.template import (
    CHATML,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPLATE,
    LLAMA3,
    PLAIN,
    SUPPORTED_TEMPLATES,
    RenderedSample,
    SFTTemplateError,
    load_training_examples,
    render_supervised,
    render_supervised_list,
)

EX = TrainingExample(
    instruction="什么是 RAG？",
    input="请用一句话回答",
    output="检索增强生成：先检索片段再生成答案。",
    source="test",
    tags=("rag",),
)

PLAIN_EX = TrainingExample(instruction="什么是 RAG？", output="检索增强生成。")


@pytest.fixture
def repo_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """把数据目录指向仓库绝对路径，使测试不依赖进程工作目录（与 day048 同例）."""
    from pathlib import Path

    from smart_research_agent.config import settings

    finetune_dir = Path(__file__).resolve().parent.parent / "data" / "finetune"
    monkeypatch.setattr(settings, "finetune_data_dir", str(finetune_dir))
    monkeypatch.setattr(settings, "finetune_seed_path", str(finetune_dir / "seed_examples.jsonl"))


class TestModuleConstants:
    """模块常量：模板名与缺省值必须稳定（它们会出现在日志与 API 响应里）."""

    def test_supported_templates(self):
        assert SUPPORTED_TEMPLATES == (CHATML, LLAMA3, PLAIN)

    def test_default_template_is_chatml(self):
        assert DEFAULT_TEMPLATE == CHATML

    def test_default_system_prompt_mentions_role(self):
        assert "智研 AI 助手" in DEFAULT_SYSTEM_PROMPT


class TestRenderedSampleInvariants:
    """RenderedSample 的构造期校验：偏移不合法必须当场失败."""

    def test_offsets_must_sum_to_length(self):
        with pytest.raises(SFTTemplateError, match="偏移之和与文本长度不一致"):
            RenderedSample(text="abc", prompt_chars=1, supervised_chars=1, template=CHATML)

    def test_prompt_chars_cannot_be_negative(self):
        with pytest.raises(SFTTemplateError, match="偏移不合法"):
            RenderedSample(text="abc", prompt_chars=-1, supervised_chars=4, template=CHATML)

    def test_supervised_chars_must_be_positive(self):
        with pytest.raises(SFTTemplateError, match="偏移不合法"):
            RenderedSample(text="abc", prompt_chars=3, supervised_chars=0, template=CHATML)

    def test_properties_split_text_exactly(self):
        sample = render_supervised(PLAIN_EX)
        assert sample.prompt_text + sample.supervised_text == sample.text
        assert sample.total_chars == len(sample.text)
        assert (
            sample.text[sample.prompt_chars : sample.prompt_chars + sample.supervised_chars]
            == sample.supervised_text
        )


class TestRenderSupervised:
    """三种模板的渲染结果."""

    @pytest.mark.parametrize("template", SUPPORTED_TEMPLATES)
    def test_offset_invariant_holds_for_every_template(self, template):
        sample = render_supervised(EX, template=template)
        assert sample.prompt_chars + sample.supervised_chars == sample.total_chars
        assert sample.template == template

    @pytest.mark.parametrize("template", SUPPORTED_TEMPLATES)
    def test_answer_never_appears_in_prompt(self, template):
        """前缀里不能出现答案——否则"只监督答案"这条纪律一开始就被破坏了."""
        sample = render_supervised(EX, template=template)
        assert EX.output not in sample.prompt_text
        assert sample.supervised_text.startswith(EX.output)

    @pytest.mark.parametrize("template", SUPPORTED_TEMPLATES)
    def test_input_is_joined_with_blank_line(self, template):
        """instruction 与 input 之间用一个空行分隔（与 prompt_text 口径一致）."""
        sample = render_supervised(EX, template=template)
        assert "什么是 RAG？\n\n请用一句话回答" in sample.prompt_text

    def test_chatml_uses_im_start_markers(self):
        sample = render_supervised(EX, template=CHATML)
        assert sample.prompt_text.startswith("<|im_start|>system\n")
        assert "<|im_start|>user\n" in sample.prompt_text
        assert sample.prompt_text.endswith("<|im_start|>assistant\n")

    def test_chatml_supervisor_span_includes_turn_terminator(self):
        """监督区间包含轮次终止符：模型要学会"在哪里停下"."""
        sample = render_supervised(EX, template=CHATML)
        assert sample.supervised_text == EX.output + "<|im_end|>\n"

    def test_llama3_uses_header_markers_and_double_newline(self):
        sample = render_supervised(EX, template=LLAMA3)
        assert sample.prompt_text.startswith("<|begin_of_text|>")
        assert "<|start_header_id|>assistant<|end_header_id|>\n\n" in sample.prompt_text
        assert sample.supervised_text == EX.output + "<|eot_id|>"

    def test_plain_template_has_no_special_markers(self):
        sample = render_supervised(EX, template=PLAIN)
        assert sample.prompt_text.endswith("### 回答\n")
        assert "<|" not in sample.text
        assert sample.supervised_text == EX.output + "\n"

    def test_turn_terminator_can_be_excluded(self):
        """关闭终止符监督后，监督区间恰好等于答案本身."""
        with_term = render_supervised(EX, template=CHATML)
        without = render_supervised(EX, template=CHATML, include_turn_terminator=False)
        assert without.supervised_text == EX.output
        assert with_term.supervised_chars > without.supervised_chars

    def test_unknown_template_rejected(self):
        with pytest.raises(SFTTemplateError, match="不支持的模板"):
            render_supervised(EX, template="not-a-template")

    def test_empty_prompt_rejected(self):
        blank = TrainingExample(instruction="   ", output="有答案")
        with pytest.raises(SFTTemplateError, match="prompt_text 为空"):
            render_supervised(blank)

    def test_empty_output_rejected(self):
        blank = TrainingExample(instruction="有问题", output="   ")
        with pytest.raises(SFTTemplateError, match="output 为空"):
            render_supervised(blank)


class TestSystemPromptResolution:
    """system 提示词的三态语义：缺省 / None / 自定义，样本级优先."""

    def test_default_when_not_passed(self):
        sample = render_supervised(EX)
        assert DEFAULT_SYSTEM_PROMPT in sample.prompt_text

    def test_none_removes_system_turn(self):
        sample = render_supervised(EX, system_prompt=None)
        assert "system" not in sample.prompt_text
        assert DEFAULT_SYSTEM_PROMPT not in sample.prompt_text
        assert sample.prompt_text.startswith("<|im_start|>user\n")

    def test_custom_system_prompt_used(self):
        sample = render_supervised(EX, system_prompt="你是严谨的审稿人。")
        assert "你是严谨的审稿人。" in sample.prompt_text
        assert DEFAULT_SYSTEM_PROMPT not in sample.prompt_text

    def test_example_system_overrides_global_default(self):
        """样本级设定优先级最高：它不该被全局缺省值盖掉."""
        custom = TrainingExample(
            instruction="问题", output="答案", system="你是出题人。"
        )
        sample = render_supervised(custom, system_prompt="你是审稿人。")
        assert "你是出题人。" in sample.prompt_text
        assert "你是审稿人。" not in sample.prompt_text

    def test_none_system_removes_turn_in_llama3_and_plain(self):
        for template in (LLAMA3, PLAIN):
            sample = render_supervised(EX, template=template, system_prompt=None)
            assert DEFAULT_SYSTEM_PROMPT not in sample.prompt_text


class TestRenderSupervisedList:
    """批量渲染：顺序与长度都必须与输入一致."""

    def test_order_preserved(self):
        examples = [
            TrainingExample(instruction=f"问题{i}", output=f"答案{i}") for i in range(5)
        ]
        rendered = render_supervised_list(examples)
        assert len(rendered) == 5
        for index, item in enumerate(rendered):
            assert item.supervised_text.startswith(f"答案{index}")

    def test_empty_input_returns_empty(self):
        assert render_supervised_list([]) == []

    def test_one_bad_sample_raises(self):
        examples = [
            TrainingExample(instruction="好的", output="有答案"),
            TrainingExample(instruction="坏的", output="  "),
        ]
        with pytest.raises(SFTTemplateError):
            render_supervised_list(examples)


class TestLoadTrainingExamples:
    """读取 day048 落盘的 train.jsonl / eval.jsonl."""

    def test_reads_real_repo_file(self, repo_data):
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        examples = load_training_examples(root / "data" / "finetune" / "seed_examples.jsonl")
        assert len(examples) == 18
        assert all(item.instruction for item in examples)
        assert all(item.output for item in examples)

    def test_roundtrip_through_dump_jsonl(self, tmp_path):
        from smart_research_agent.finetune.schema import dump_jsonl

        target = tmp_path / "train.jsonl"
        dump_jsonl([EX, PLAIN_EX], target)
        loaded = load_training_examples(target)
        assert [item.instruction for item in loaded] == [EX.instruction, PLAIN_EX.instruction]
        assert loaded[0].input == EX.input

    def test_invalid_record_raises(self, tmp_path):
        from smart_research_agent.finetune.schema import DatasetFormatError

        target = tmp_path / "bad.jsonl"
        target.write_text('{"instruction": "只有问题"}\n', encoding="utf-8")
        with pytest.raises(DatasetFormatError):
            load_training_examples(target)
