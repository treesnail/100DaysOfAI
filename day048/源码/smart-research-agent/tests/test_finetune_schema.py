"""微调数据格式测试（day048）：三种格式的解析、投影与 JSONL 往返.

全部离线：只碰纯函数与临时文件，不读仓库里的真实语料。
"""

from __future__ import annotations

import json

import pytest

from smart_research_agent.finetune.schema import (
    ALPACA,
    CHAT,
    DEFAULT_FORMAT,
    PROMPT_COMPLETION,
    SUPPORTED_FORMATS,
    DatasetFormatError,
    TrainingExample,
    dump_jsonl,
    load_jsonl,
    parse_example,
)


class TestFormatConstants:
    """格式常量：值本身是公开契约，教程与 API 都按字面量引用."""

    def test_format_values(self):
        assert ALPACA == "alpaca"
        assert CHAT == "chat"
        assert PROMPT_COMPLETION == "prompt-completion"

    def test_supported_formats_and_default(self):
        assert SUPPORTED_FORMATS == (ALPACA, CHAT, PROMPT_COMPLETION)
        assert DEFAULT_FORMAT == ALPACA


class TestTrainingExampleProjection:
    """三种目标格式的投影规则."""

    def test_prompt_text_plain(self):
        example = TrainingExample(instruction="问题", output="答案")
        assert example.prompt_text == "问题"

    def test_prompt_text_joins_input_with_blank_line(self):
        example = TrainingExample(instruction="总结", input="正文", output="答案")
        assert example.prompt_text == "总结\n\n正文"

    def test_prompt_text_ignores_input_when_instruction_empty(self):
        """instruction 为空时不拼接 input（否则会返回以空行开头的脏文本）."""
        example = TrainingExample(instruction="", input="正文", output="答案")
        assert example.prompt_text == ""

    def test_to_alpaca_keeps_empty_input_key(self):
        payload = TrainingExample(instruction="问题", output="答案").to_alpaca()
        assert payload == {"instruction": "问题", "input": "", "output": "答案"}
        assert "input" in payload

    def test_to_chat_without_system(self):
        payload = TrainingExample(instruction="问题", input="材料", output="答案").to_chat()
        assert payload == {
            "messages": [
                {"role": "user", "content": "问题\n\n材料"},
                {"role": "assistant", "content": "答案"},
            ]
        }

    def test_to_chat_puts_system_first(self):
        payload = TrainingExample(
            instruction="问题", output="答案", system="你是研究助手"
        ).to_chat()
        assert payload["messages"][0] == {"role": "system", "content": "你是研究助手"}
        assert [m["role"] for m in payload["messages"]] == ["system", "user", "assistant"]

    def test_to_chat_skips_blank_system(self):
        payload = TrainingExample(instruction="问题", output="答案", system="").to_chat()
        assert [m["role"] for m in payload["messages"]] == ["user", "assistant"]

    def test_to_prompt_completion(self):
        payload = TrainingExample(instruction="问题", input="材料", output="答案").to_dict(
            PROMPT_COMPLETION
        )
        assert payload == {"prompt": "问题\n\n材料", "completion": "答案"}

    def test_to_dict_default_is_alpaca(self):
        example = TrainingExample(instruction="问题", output="答案")
        assert example.to_dict() == example.to_alpaca()

    def test_to_dict_unknown_format_raises(self):
        with pytest.raises(DatasetFormatError, match="不支持的数据集格式"):
            TrainingExample(instruction="问题", output="答案").to_dict("sharegpt")

    def test_is_safety_example(self):
        safety = TrainingExample(instruction="攻击", output="拒绝", tags=("safety", "jailbreak"))
        normal = TrainingExample(instruction="问题", output="答案", tags=("rag",))
        assert safety.is_safety_example() is True
        assert normal.is_safety_example() is False


class TestParseExample:
    """解析：必填字段缺失必须报错，元信息必须带出来."""

    def test_parse_alpaca_minimal(self):
        example = parse_example({"instruction": "问题", "output": "答案"})
        assert example.instruction == "问题"
        assert example.output == "答案"
        assert example.input == ""

    def test_parse_alpaca_carries_metadata(self):
        example = parse_example(
            {
                "instruction": "问题",
                "output": "答案",
                "input": "材料",
                "system": "你是助手",
                "source": "seed",
                "tags": ["rag", "eval"],
                "license": "CC-BY-4.0",
            }
        )
        assert example.input == "材料"
        assert example.system == "你是助手"
        assert example.source == "seed"
        assert example.tags == ("rag", "eval")
        assert example.license == "CC-BY-4.0"

    def test_parse_alpaca_tags_accepts_single_string(self):
        example = parse_example({"instruction": "问题", "output": "答案", "tags": "safety"})
        assert example.tags == ("safety",)

    def test_parse_alpaca_tags_scalar_fallback(self):
        """tags 写成非字符串标量（int 等）时规整为单元素元组，而不是崩掉."""
        example = parse_example({"instruction": "问题", "output": "答案", "tags": 42})
        assert example.tags == ("42",)

    def test_parse_alpaca_missing_output_raises(self):
        with pytest.raises(DatasetFormatError, match="output/completion 不能为空"):
            parse_example({"instruction": "问题"})

    def test_parse_alpaca_blank_instruction_raises(self):
        with pytest.raises(DatasetFormatError, match="instruction/prompt 不能为空"):
            parse_example({"instruction": "   ", "output": "答案"})

    def test_parse_normalizes_non_string_numbers(self):
        """字段类型漂移（数字写成 int）不该让整条样本报废."""
        example = parse_example({"instruction": 42, "output": 7})
        assert example.instruction == "42"
        assert example.output == "7"

    def test_parse_null_system_becomes_none(self):
        example = parse_example({"instruction": "问题", "output": "答案", "system": None})
        assert example.system is None

    def test_parse_unknown_format_raises(self):
        with pytest.raises(DatasetFormatError, match="不支持的数据集格式"):
            parse_example({"instruction": "问题", "output": "答案"}, "dpo")

    def test_parse_non_dict_raises(self):
        with pytest.raises(DatasetFormatError, match="必须是 JSON 对象"):
            parse_example(["not", "a", "dict"])  # type: ignore[arg-type]

    def test_parse_chat_ok(self):
        example = parse_example(
            {
                "messages": [
                    {"role": "system", "content": "你是助手"},
                    {"role": "user", "content": "问题"},
                    {"role": "assistant", "content": "答案"},
                ]
            },
            CHAT,
        )
        assert example.prompt_text == "问题"
        assert example.output == "答案"
        assert example.system == "你是助手"

    def test_parse_chat_missing_assistant_raises(self):
        with pytest.raises(DatasetFormatError, match="user 与 assistant"):
            parse_example({"messages": [{"role": "user", "content": "问题"}]}, CHAT)

    def test_parse_chat_blank_content_raises(self):
        with pytest.raises(DatasetFormatError, match="user 与 assistant"):
            parse_example(
                {
                    "messages": [
                        {"role": "user", "content": "问题"},
                        {"role": "assistant", "content": "   "},
                    ]
                },
                CHAT,
            )

    def test_parse_chat_requires_message_list(self):
        with pytest.raises(DatasetFormatError, match="messages 为非空列表"):
            parse_example({"messages": "问题"}, CHAT)

    def test_parse_prompt_completion(self):
        example = parse_example({"prompt": "续写", "completion": "结果"}, PROMPT_COMPLETION)
        assert example.instruction == "续写"
        assert example.output == "结果"

    def test_parse_prompt_completion_missing_completion_raises(self):
        with pytest.raises(DatasetFormatError, match="output/completion 不能为空"):
            parse_example({"prompt": "续写"}, PROMPT_COMPLETION)


class TestJsonlRoundTrip:
    """JSONL 读写往返：字段不丢、父目录自动创建、空行被跳过.

    注意往返的**边界**：三种目标格式都只承载"指令 / 输入 / 输出（+ chat 的
    system）"，``source``/``tags``/``license`` 这些治理元信息不在其中——
    它们是数据集级别的元信息，落盘后由数据卡（data card）另行记录，
    而不是塞进每条训练样本。因此这里只断言格式本身承载的字段。
    """

    def test_round_trip_alpaca(self, tmp_path):
        original = TrainingExample(instruction="问题", output="答案", input="材料")
        path = tmp_path / "nested" / "alpaca.jsonl"
        assert dump_jsonl([original], path, ALPACA) == 1
        assert path.exists()

        restored = parse_example(load_jsonl(path)[0], ALPACA)
        assert restored.instruction == "问题"
        assert restored.input == "材料"
        assert restored.output == "答案"

    def test_round_trip_chat_keeps_system(self, tmp_path):
        original = TrainingExample(
            instruction="问题", output="答案", input="材料", system="你是研究助手"
        )
        path = tmp_path / "chat.jsonl"
        assert dump_jsonl([original], path, CHAT) == 1

        restored = parse_example(load_jsonl(path)[0], CHAT)
        assert restored.system == "你是研究助手"
        assert restored.instruction == "问题\n\n材料"
        assert restored.output == "答案"

    def test_round_trip_prompt_completion(self, tmp_path):
        original = TrainingExample(instruction="续写", output="结果")
        path = tmp_path / "pc.jsonl"
        assert dump_jsonl([original], path, PROMPT_COMPLETION) == 1

        restored = parse_example(load_jsonl(path)[0], PROMPT_COMPLETION)
        assert restored.instruction == "续写"
        assert restored.output == "结果"

    def test_dump_jsonl_returns_count_and_creates_parent(self, tmp_path):
        path = tmp_path / "a" / "b" / "out.jsonl"
        examples = [TrainingExample(instruction=f"q{i}", output="答案") for i in range(3)]
        assert dump_jsonl(examples, path) == 3
        assert path.parent.is_dir()
        assert len(load_jsonl(path)) == 3

    def test_dump_jsonl_writes_utf8_readable(self, tmp_path):
        path = tmp_path / "zh.jsonl"
        dump_jsonl([TrainingExample(instruction="中文", output="答案")], path)
        content = path.read_text(encoding="utf-8")
        assert "中文" in content
        assert json.loads(content.strip())["instruction"] == "中文"

    def test_load_jsonl_skips_blank_lines(self, tmp_path):
        path = tmp_path / "blank.jsonl"
        path.write_text(
            '{"instruction": "a", "output": "b"}\n\n   \n{"instruction": "c", "output": "d"}\n',
            encoding="utf-8",
        )
        assert len(load_jsonl(path)) == 2

    def test_load_jsonl_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        assert load_jsonl(path) == []
