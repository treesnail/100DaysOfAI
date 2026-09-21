"""LoRA / QLoRA 脚本生成测试（day051）：生成物必须是**可执行的 Python**，且参数同源.

day050 已经立下这条纪律：**教程里的参数与脚本里的参数不能有两个来源**。本文件
把它扩展到 LoRA / QLoRA：

1. :meth:`TestRenderLoraScript.test_embedded_json_matches_sources` —— 从脚本文本里
   把 ``TRAINING_ARGUMENTS`` / ``LORA_CONFIG`` / ``QUANT_CONFIG`` 三块内嵌 JSON
   抠出来 ``json.loads``，再与 ``SFTTrainingArgs.to_hf_dict()`` /
   ``LoRAConfig.to_peft_dict()`` / ``QLoRAConfig.to_bnb_dict()`` **逐键比对**；
2. :meth:`TestRenderLoraScript.test_lora_script_has_no_qlora_machinery` —— LoRA
   脚本里**不允许**出现 ``prepare_model_for_kbit_training`` 与
   ``BitsAndBytesConfig``。混进 4-bit 的机制不会报错，只会让"你以为在用 LoRA"
   与"实际在用 QLoRA"之间的差别消失。
"""

from __future__ import annotations

import json
import re

import pytest

from smart_research_agent.peft.config import LoRAConfig, PEFTConfigError, QLoRAConfig
from smart_research_agent.peft.hf_script import (
    DTYPE_MAPPING_SNIPPET,
    PEFT_DEPENDENCIES,
    QLORA_DEPENDENCIES,
    peft_dependency_commands,
    render_lora_script,
    render_qlora_script,
    rendered_peft_summary,
)
from smart_research_agent.peft.targets import MODEL_SPECS
from smart_research_agent.sft.args import SFTConfigError, SFTTrainingArgs
from smart_research_agent.sft.hf_script import (
    DEFAULT_BASE_MODEL,
    DEFAULT_SYSTEM_IN_SCRIPT,
    HF_DEPENDENCIES,
)


def extract_json_payload(script: str, marker: str) -> dict:
    """从生成脚本里抠出 ``marker = json.loads(\"\"\"...\"\"\")`` 里的 JSON."""
    pattern = re.compile(
        re.escape(marker) + r'\s*=\s*json\.loads\(\s*"""(.+?)"""\s*\)', re.DOTALL
    )
    match = pattern.search(script)
    assert match is not None, f"脚本里找不到 {marker} 的 JSON 负载"
    return json.loads(match.group(1))


class TestPeftDependencies:
    """依赖清单：在 day050 的全参依赖上只多 ``peft``（QLoRA 再多 ``bitsandbytes``）."""

    def test_peft_dependencies_extend_hf(self):
        assert set(HF_DEPENDENCIES) <= set(PEFT_DEPENDENCIES)
        assert PEFT_DEPENDENCIES == (*HF_DEPENDENCIES, "peft>=0.20")

    def test_qlora_dependencies_extend_peft(self):
        assert set(PEFT_DEPENDENCIES) <= set(QLORA_DEPENDENCIES)
        assert QLORA_DEPENDENCIES == (*PEFT_DEPENDENCIES, "bitsandbytes>=0.50")

    def test_day050_packages_are_still_pinned(self):
        joined = " ".join(PEFT_DEPENDENCIES)
        for package in (
            "transformers>=5.16",
            "datasets>=3.0",
            "accelerate>=1.0",
            "torch>=2.5",
        ):
            assert package in joined


class TestRenderLoraScript:
    """``transformers`` + ``peft`` 路径的生成脚本."""

    def test_is_valid_python_and_long_enough(self):
        script = render_lora_script(SFTTrainingArgs(), LoRAConfig())
        assert len(script.splitlines()) > 100
        compile(script, "lora_train.py", "exec")

    def test_embedded_json_matches_sources(self):
        """内嵌 JSON 必须与三个来源逐键一致——参数只有一个来源."""
        args = SFTTrainingArgs(learning_rate=1e-4, num_train_epochs=2.0, max_length=128)
        lora = LoRAConfig(r=16, lora_alpha=32, target_modules="mlp")
        script = render_lora_script(args, lora)
        assert extract_json_payload(script, "TRAINING_ARGUMENTS") == args.to_hf_dict()
        assert extract_json_payload(script, "LORA_CONFIG") == lora.to_peft_dict()
        assert "QUANT_CONFIG" not in script

    def test_preset_is_expanded_in_embedded_json(self):
        """脚本里没有预设表：JSON 必须已经展开成具体模块名."""
        lora = LoRAConfig(target_modules="attention")
        script = render_lora_script(SFTTrainingArgs(), lora)
        payload = extract_json_payload(script, "LORA_CONFIG")
        assert payload["target_modules"] == ["q_proj", "v_proj"]
        assert payload["target_modules"] == list(lora.resolved_targets)

    def test_expected_trainable_is_injected(self):
        script = render_lora_script(SFTTrainingArgs(), LoRAConfig(), expected_trainable=4_194_304)
        assert "EXPECTED_TRAINABLE = 4194304" in script

    def test_expected_trainable_defaults_to_zero(self):
        script = render_lora_script(SFTTrainingArgs(), LoRAConfig())
        assert "EXPECTED_TRAINABLE = 0" in script

    def test_max_length_and_model_and_prompt_are_substituted(self):
        script = render_lora_script(
            SFTTrainingArgs(max_length=768),
            LoRAConfig(),
            base_model="Qwen/Qwen2.5-7B",
            system_prompt="你是审稿人。",
        )
        assert "MAX_LENGTH = 768" in script
        assert "Qwen/Qwen2.5-7B" in script
        assert "你是审稿人。" in script

    def test_defaults_come_from_day050(self):
        script = render_lora_script(SFTTrainingArgs(), LoRAConfig())
        assert DEFAULT_BASE_MODEL in script
        assert DEFAULT_SYSTEM_IN_SCRIPT in script

    def test_key_fragments_present(self):
        script = render_lora_script(SFTTrainingArgs(), LoRAConfig())
        for fragment in (
            "from peft import LoraConfig, TaskType, get_peft_model",
            "model = get_peft_model(model, peft_config)",
            "IGNORE_INDEX = -100",
            "model.save_pretrained(args.output_dir)",
            DTYPE_MAPPING_SNIPPET,
        ):
            assert fragment in script

    def test_peft_config_is_built_from_json(self):
        script = render_lora_script(SFTTrainingArgs(), LoRAConfig())
        assert 'task_type = TaskType[LORA_CONFIG["task_type"]]' in script
        assert 'if key != "task_type"' in script

    def test_lora_script_has_no_qlora_machinery(self):
        script = render_lora_script(SFTTrainingArgs(), LoRAConfig())
        assert "prepare_model_for_kbit_training" not in script
        assert "BitsAndBytesConfig" not in script

    def test_invalid_args_rejected(self):
        with pytest.raises(SFTConfigError):
            render_lora_script(SFTTrainingArgs(bf16=True, fp16=True), LoRAConfig())

    def test_invalid_lora_config_rejected(self):
        with pytest.raises(PEFTConfigError, match="r 必须为正整数"):
            render_lora_script(SFTTrainingArgs(), LoRAConfig(r=0))


class TestRenderQloraScript:
    """4-bit 基座 + ``peft`` 路径的生成脚本."""

    def test_is_valid_python_and_long_enough(self):
        script = render_qlora_script(SFTTrainingArgs(), LoRAConfig(), QLoRAConfig())
        assert len(script.splitlines()) > 100
        compile(script, "qlora_train.py", "exec")

    def test_embedded_json_matches_sources(self):
        args = SFTTrainingArgs(learning_rate=1e-4, max_length=192)
        lora = LoRAConfig(r=16, lora_alpha=32, target_modules="attention_all")
        qlora = QLoRAConfig(block_size=128)
        script = render_qlora_script(args, lora, qlora)
        assert extract_json_payload(script, "TRAINING_ARGUMENTS") == args.to_hf_dict()
        assert extract_json_payload(script, "LORA_CONFIG") == lora.to_peft_dict()
        assert extract_json_payload(script, "QUANT_CONFIG") == qlora.to_bnb_dict()

    def test_quant_config_defaults_when_omitted(self):
        """不传 ``QLoRAConfig`` 时用缺省值，但**仍然要写进脚本**."""
        script = render_qlora_script(SFTTrainingArgs(), LoRAConfig())
        assert extract_json_payload(script, "QUANT_CONFIG") == QLoRAConfig().to_bnb_dict()

    def test_expected_trainable_is_injected(self):
        script = render_qlora_script(
            SFTTrainingArgs(), LoRAConfig(), expected_trainable=123456
        )
        assert "EXPECTED_TRAINABLE = 123456" in script

    def test_key_fragments_present(self):
        script = render_qlora_script(SFTTrainingArgs(), LoRAConfig())
        for fragment in (
            "get_peft_model",
            "prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)",
            "BitsAndBytesConfig(**payload)",
            "build_quantization_config",
            "IGNORE_INDEX = -100",
            "model.save_pretrained(args.output_dir)",
            DTYPE_MAPPING_SNIPPET,
        ):
            assert fragment in script

    def test_compute_dtype_is_mapped_from_json(self):
        script = render_qlora_script(SFTTrainingArgs(), LoRAConfig())
        assert 'COMPUTE_DTYPES[QUANT_CONFIG["bnb_4bit_compute_dtype"]]' in script

    def test_bytes_per_parameter_is_written(self):
        qlora = QLoRAConfig()
        script = render_qlora_script(SFTTrainingArgs(), LoRAConfig(), qlora)
        assert f"{qlora.bytes_per_parameter:.6f}" in script

    def test_int4_baseline_rejected(self):
        with pytest.raises(PEFTConfigError, match="对照基线"):
            render_qlora_script(
                SFTTrainingArgs(), LoRAConfig(), QLoRAConfig(bnb_4bit_quant_type="int4")
            )

    def test_invalid_lora_config_rejected(self):
        with pytest.raises(PEFTConfigError, match="lora_alpha"):
            render_qlora_script(SFTTrainingArgs(), LoRAConfig(lora_alpha=0))

    def test_invalid_args_rejected(self):
        with pytest.raises(SFTConfigError):
            render_qlora_script(SFTTrainingArgs(truncation="nope"), LoRAConfig())


class TestPeftDependencyCommands:
    """安装命令：可直接复制粘贴，而不是散在注释里."""

    def test_lora_only(self):
        payload = peft_dependency_commands()
        assert payload["pip"].startswith("pip install ")
        assert payload["uv"].startswith("uv pip install ")
        assert payload["packages"] == ", ".join(PEFT_DEPENDENCIES)
        assert "bitsandbytes" not in payload["packages"]

    def test_qlora(self):
        payload = peft_dependency_commands(use_qlora=True)
        assert payload["packages"] == ", ".join(QLORA_DEPENDENCIES)
        assert "bitsandbytes>=0.50" in payload["packages"]

    def test_packages_are_quoted(self):
        payload = peft_dependency_commands()
        assert '"peft>=0.20"' in payload["pip"]
        assert '"peft>=0.20"' in payload["uv"]


class TestRenderedPeftSummary:
    """脚本自述：不读脚本文本也能知道"这次训多少参数、要装什么依赖"."""

    def test_lora_summary_fields(self):
        args = SFTTrainingArgs(max_length=256)
        lora = LoRAConfig(r=16, lora_alpha=32, target_modules="mlp")
        summary = rendered_peft_summary(args, lora)
        assert summary["base_model"] == DEFAULT_BASE_MODEL
        assert summary["lora"] == lora.to_peft_dict()
        assert summary["scaling"] == pytest.approx(2.0)
        assert summary["scaling_formula"] == lora.scaling_formula
        assert summary["quantization"] is None
        assert summary["quantization_bytes_per_parameter"] is None
        assert summary["training_arguments"] == args.to_hf_dict()
        assert summary["max_length"] == 256
        assert summary["peft_dependencies"] == list(PEFT_DEPENDENCIES)
        assert summary["qlora_dependencies"] == list(QLORA_DEPENDENCIES)
        assert summary["install"] == peft_dependency_commands()
        assert "plan" not in summary

    def test_qlora_summary_carries_quantization(self):
        qlora = QLoRAConfig(block_size=128)
        summary = rendered_peft_summary(SFTTrainingArgs(), LoRAConfig(), qlora)
        assert summary["quantization"] == qlora.to_bnb_dict()
        assert summary["quantization_bytes_per_parameter"] == round(
            qlora.bytes_per_parameter, 6
        )
        assert summary["install"] == peft_dependency_commands(use_qlora=True)

    def test_plan_is_attached_when_spec_given(self):
        """``spec`` 非空时附带参数量计划——脚本里 ``EXPECTED_TRAINABLE`` 的来源."""
        lora = LoRAConfig(r=8, lora_alpha=16, target_modules="attention")
        summary = rendered_peft_summary(SFTTrainingArgs(), lora, spec=MODEL_SPECS["llama-2-7b"])
        plan = summary["plan"]
        assert plan["model"] == "llama-2-7b"
        assert plan["targets"] == ["q_proj", "v_proj"]
        assert plan["adapter_parameters"] == 4_194_304
        assert plan["trainable_parameters"] == plan["adapter_parameters"]
        total = plan["frozen_parameters"] + plan["trainable_parameters"]
        assert plan["trainable_ratio"] == pytest.approx(plan["trainable_parameters"] / total)

    def test_summary_is_json_ready(self):
        summary = rendered_peft_summary(SFTTrainingArgs(), LoRAConfig())
        payload = json.loads(json.dumps(summary, ensure_ascii=False))
        assert payload["max_length"] == 384
        assert payload["lora"]["r"] == 8
