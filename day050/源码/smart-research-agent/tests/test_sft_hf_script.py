"""训练脚本生成测试（day050）：生成物必须是**可执行的 Python**，且参数同源.

这里守的是一条容易被忽略的纪律：**教程里的参数与脚本里的参数不能有
两个来源**。所以测试会从生成的脚本文本里把内嵌的 JSON 抠出来，再与
``SFTTrainingArgs.to_hf_dict()`` 逐键比对。
"""

from __future__ import annotations

import json
import re

import pytest

from smart_research_agent.sft.args import SFTConfigError, SFTTrainingArgs
from smart_research_agent.sft.hf_script import (
    DEFAULT_BASE_MODEL,
    HF_DEPENDENCIES,
    TRL_DEPENDENCIES,
    dependency_commands,
    generated_script_summary,
    render_hf_sft_script,
    render_trl_sft_script,
)


def extract_json_payload(script: str, marker: str) -> dict:
    """从生成脚本里抠出 ``marker = json.loads(\"\"\"...\"\"\")`` 里的 JSON."""
    pattern = re.compile(
        re.escape(marker) + r'\s*=\s*json\.loads\(\s*"""(.+?)"""\s*\)', re.DOTALL
    )
    match = pattern.search(script)
    assert match is not None, f"脚本里找不到 {marker} 的 JSON 负载"
    return json.loads(match.group(1))


class TestDependencyConstants:
    """依赖常量必须带版本下限（本课核对于 2026-09）."""

    def test_hf_dependencies_pin_versions(self):
        joined = " ".join(HF_DEPENDENCIES)
        assert "transformers>=5.16" in joined
        assert "datasets>=3.0" in joined
        assert "accelerate>=1.0" in joined
        assert "torch>=2.5" in joined

    def test_trl_dependencies_extend_hf(self):
        assert set(HF_DEPENDENCIES) <= set(TRL_DEPENDENCIES)
        assert any("trl>=1.5" in item for item in TRL_DEPENDENCIES)

    def test_default_base_model(self):
        assert DEFAULT_BASE_MODEL == "Qwen/Qwen3-0.6B"


class TestRenderHfScript:
    """``transformers`` + ``Trainer`` 路径的生成脚本."""

    def test_is_valid_python(self):
        compile(render_hf_sft_script(SFTTrainingArgs()), "sft_train_transformers.py", "exec")

    def test_custom_model_and_system_prompt_are_substituted(self):
        script = render_hf_sft_script(
            SFTTrainingArgs(), base_model="Qwen/Qwen2.5-1.5B", system_prompt="你是审稿人。"
        )
        assert "Qwen/Qwen2.5-1.5B" in script
        assert "你是审稿人。" in script

    def test_embedded_args_match_to_hf_dict(self):
        """内嵌 JSON 必须与 ``to_hf_dict()`` 完全一致——参数只有一个来源."""
        args = SFTTrainingArgs(learning_rate=1e-4, num_train_epochs=2.0, max_length=128)
        script = render_hf_sft_script(args)
        payload = extract_json_payload(script, "TRAINING_ARGUMENTS")
        assert payload == args.to_hf_dict()

    def test_max_length_is_injected(self):
        script = render_hf_sft_script(SFTTrainingArgs(max_length=768))
        assert "MAX_LENGTH = 768" in script

    def test_ignore_index_convention(self):
        script = render_hf_sft_script(SFTTrainingArgs())
        assert "IGNORE_INDEX = -100" in script

    def test_label_masking_lines_present(self):
        script = render_hf_sft_script(SFTTrainingArgs())
        assert '"labels": [IGNORE_INDEX] * len(kept_prompt) + answer_ids,' in script
        assert "kept_prompt = prompt_ids[-budget:]" in script

    def test_padding_trio_in_collator(self):
        script = render_hf_sft_script(SFTTrainingArgs())
        assert '[pad_token_id] * pad' in script
        assert "[0] * pad" in script
        assert "[IGNORE_INDEX] * pad" in script

    def test_drops_samples_whose_answer_does_not_fit(self):
        script = render_hf_sft_script(SFTTrainingArgs())
        assert "if budget < 1:" in script
        assert "return None" in script

    def test_tokenizes_prefix_and_answer_separately(self):
        """必须分别 tokenize：合并不认字符边界（教程实现要点第 1 条）."""
        script = render_hf_sft_script(SFTTrainingArgs())
        assert 'tokenizer(prefix, add_special_tokens=False)["input_ids"]' in script
        assert 'tokenizer(answer, add_special_tokens=False)["input_ids"]' in script

    def test_prints_supervision_ratio(self):
        script = render_hf_sft_script(SFTTrainingArgs())
        assert "监督" in script

    def test_compute_metrics_uses_mean_token_accuracy(self):
        script = render_hf_sft_script(SFTTrainingArgs())
        assert "mean_token_accuracy" in script
        assert "preprocess_logits_for_metrics" in script

    def test_uses_eval_strategy_not_deprecated_name(self):
        script = render_hf_sft_script(SFTTrainingArgs())
        assert "evaluation_strategy" not in script

    def test_invalid_args_rejected(self):
        with pytest.raises(SFTConfigError):
            render_hf_sft_script(SFTTrainingArgs(bf16=True, fp16=True))


class TestRenderTrlScript:
    """TRL ``SFTTrainer`` 路径的生成脚本."""

    def test_is_valid_python(self):
        compile(render_trl_sft_script(SFTTrainingArgs()), "sft_train_trl.py", "exec")

    def test_imports_json(self):
        """脚本里用了 ``json.loads``，必须自带 import（这里曾被漏掉过）."""
        script = render_trl_sft_script(SFTTrainingArgs())
        assert "\nimport json\n" in script

    def test_embedded_config_matches_to_trl_dict(self):
        args = SFTTrainingArgs(max_length=192)
        payload = extract_json_payload(render_trl_sft_script(args), "SFT_CONFIG")
        assert payload == args.to_trl_dict()

    def test_sft_specific_switches_present(self):
        script = render_trl_sft_script(SFTTrainingArgs())
        assert '"max_length": 384' in script
        assert '"packing": false' in script
        assert '"assistant_only_loss": true' in script
        assert '"completion_only_loss": true' in script

    def test_converts_to_prompt_completion(self):
        script = render_trl_sft_script(SFTTrainingArgs())
        assert "to_prompt_completion" in script
        assert '"role": "assistant"' in script

    def test_invalid_args_rejected(self):
        with pytest.raises(SFTConfigError):
            render_trl_sft_script(SFTTrainingArgs(lr_scheduler_type="nope"))


class TestDependencyCommands:
    """安装命令：可直接复制粘贴，而不是散在注释里."""

    def test_hf_only(self):
        payload = dependency_commands()
        assert payload["pip"].startswith("pip install ")
        assert payload["uv"].startswith("uv pip install ")
        assert "trl" not in payload["packages"]

    def test_with_trl(self):
        payload = dependency_commands(use_trl=True)
        assert "trl>=1.5" in payload["packages"]
        assert payload["packages"] == ", ".join(TRL_DEPENDENCIES)


class TestGeneratedScriptSummary:
    """脚本自述：不读脚本文本也能知道"这份脚本用什么超参、要什么依赖"."""

    def test_summary_fields(self):
        args = SFTTrainingArgs(max_length=256)
        summary = generated_script_summary(args)
        assert summary["base_model"] == DEFAULT_BASE_MODEL
        assert summary["max_length"] == 256
        assert summary["packing"] is False
        assert summary["assistant_only_loss"] is True
        assert summary["hf_dependencies"] == list(HF_DEPENDENCIES)
        assert summary["training_arguments"] == args.to_hf_dict()
        assert summary["sft_specific"]["ignore_index"] == -100
        assert summary["sft_specific"]["max_length"] == 256

    def test_summary_is_json_ready(self):
        payload = generated_script_summary(SFTTrainingArgs())
        assert json.loads(json.dumps(payload, ensure_ascii=False))["max_length"] == 384
