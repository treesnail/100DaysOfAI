"""多卡 LoRA 脚本生成测试（day052）：生成物必须是**可执行的 Python**，且参数同源.

本文件验证三件事，每一件都对应一个真实的坑：

1. :meth:`TestRenderAccelerateLoraScript.test_embedded_json_matches_sources` ——
   从脚本文本里把 ``TRAINING_ARGUMENTS`` / ``LORA_CONFIG`` 抠出来 ``json.loads``，
   与 ``SFTTrainingArgs.to_hf_dict()`` / ``LoRAConfig.to_peft_dict()`` **逐键比对**：
   教程里的参数与脚本里的参数不可能漂移（day050 立下、day051 继承的纪律）；
2. :meth:`TestRenderAccelerateLoraScript.test_gradient_accumulation_is_left_to_accelerator`
   —— 梯度累积交给 ``Accelerator``，``TrainingArguments`` 里**留 1**：两处各累积
   一次会让有效批变成 ``accum²`` 倍，而训练照常跑、loss 照常降；
3. :meth:`TestRenderMergeScriptForwarding.test_forwards_to_deploy_verbatim` ——
   ``hf_script.render_merge_script`` 只是转发到 ``peft.deploy.render_merge_script``，
   输出必须**逐字相同**：合并脚本只有一个来源。
"""

from __future__ import annotations

import json
import re

import pytest

from smart_research_agent.peft import deploy
from smart_research_agent.peft.config import LoRAConfig, PEFTConfigError
from smart_research_agent.peft.hf_script import (
    COMMON_DATA_PREP,
    render_accelerate_lora_script,
    render_merge_script,
)
from smart_research_agent.sft.args import SFTTrainingArgs

#: 生成脚本的行数下限（数据准备那一段本身就是一百来行）
MIN_SCRIPT_LINES = 150


def extract_json_payload(script: str, marker: str) -> dict:
    """从生成脚本里抠出 ``marker = json.loads(\"\"\"...\"\"\")`` 里的 JSON."""
    pattern = re.compile(
        re.escape(marker) + r'\s*=\s*json\.loads\(\s*"""(.+?)"""\s*\)', re.DOTALL
    )
    match = pattern.search(script)
    assert match is not None, f"脚本里找不到 {marker} 的 JSON 负载"
    return json.loads(match.group(1))


def call_block(script: str, callee: str) -> str:
    """抠出 ``callee`` 这一处调用的完整括号块（按括号配平，够用且不依赖格式）."""
    start = script.index(callee)
    depth = 0
    for index in range(start, len(script)):
        char = script[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return script[start : index + 1]
    raise AssertionError(f"脚本里找不到完整的 {callee} 调用")


@pytest.fixture
def args() -> SFTTrainingArgs:
    """一份与脚本无关的参考超参（累积步数显式取 4，便于核对两处开关）."""
    return SFTTrainingArgs(learning_rate=1e-4, max_length=192, gradient_accumulation_steps=4)


@pytest.fixture
def lora() -> LoRAConfig:
    """一份普通的 attention 预设 LoRA 配置."""
    return LoRAConfig(r=16, lora_alpha=32, target_modules="attention")


@pytest.fixture
def script(args: SFTTrainingArgs, lora: LoRAConfig) -> str:
    """4 进程 / ZeRO-2 的多卡训练脚本."""
    return render_accelerate_lora_script(args, lora, devices=4, strategy="zero2")


class TestRenderAccelerateLoraScript:
    """``render_accelerate_lora_script``：多卡脚本的生成物."""

    def test_is_valid_python_and_long_enough(self, script: str):
        assert len(script.splitlines()) > MIN_SCRIPT_LINES
        compile(script, "train_lora_zero2.py", "exec")

    def test_key_fragments_present(self, script: str):
        for fragment in (
            "Accelerator(",
            "accelerator.prepare(",
            "is_main_process",
            "get_peft_model(",
            "accelerate launch",
            "--num_processes 4",
        ):
            assert fragment in script

    def test_embedded_json_matches_sources(
        self, script: str, args: SFTTrainingArgs, lora: LoRAConfig
    ):
        """内嵌 JSON 与来源同源，**只少一个键**：``gradient_accumulation_steps``.

        那个键必须被删掉（脚本里显式写 ``=1``，累积交给 Accelerator），
        其余每个键都要逐字相等——**参数只有一个来源**这条性质不能因为
        "多卡脚本多了一层"就打折。
        """
        expected = args.to_hf_dict()
        expected.pop("gradient_accumulation_steps")
        assert extract_json_payload(script, "TRAINING_ARGUMENTS") == expected
        assert extract_json_payload(script, "LORA_CONFIG") == lora.to_peft_dict()

    def test_gradient_accumulation_is_left_to_accelerator(
        self, script: str, args: SFTTrainingArgs
    ):
        """累积交给 ``Accelerator``：``TrainingArguments`` 里留 1，**内嵌 JSON 里必须删掉**.

        这两件事必须一起做。只写 ``gradient_accumulation_steps=1`` 而让内嵌 JSON
        也带这个键，展开时会抛
        ``TypeError: got multiple values for keyword argument``——**脚本语法合法、
        ``compile()`` 也通过，只有真正运行时才崩**。本课实现时踩过这一次，
        所以这条用例专门钉住"JSON 里没有它"。
        """
        training_arguments = call_block(script, "TrainingArguments(")
        assert "gradient_accumulation_steps=1" in training_arguments
        accelerator_call = call_block(script, "Accelerator(")
        assert (
            f"Accelerator(gradient_accumulation_steps={args.gradient_accumulation_steps})"
            in accelerator_call
        )
        # 内嵌 JSON 里**必须没有**这个键（否则上面那行会 TypeError）
        embedded = extract_json_payload(script, "TRAINING_ARGUMENTS")
        assert "gradient_accumulation_steps" not in embedded
        assert embedded["per_device_train_batch_size"] == 2

    def test_single_device_omits_num_processes(self, args: SFTTrainingArgs, lora: LoRAConfig):
        single = render_accelerate_lora_script(args, lora, devices=1)
        assert "accelerate launch" in single
        assert "--num_processes" not in single

    def test_common_data_prep_is_injected(self, script: str):
        assert COMMON_DATA_PREP in script
        assert "def tokenize_example(" in script
        assert "IGNORE_INDEX = -100" in script

    def test_invalid_lora_config_rejected(self, args: SFTTrainingArgs):
        with pytest.raises(PEFTConfigError, match="r 必须为正整数"):
            render_accelerate_lora_script(args, LoRAConfig(r=0), devices=4)


class TestRenderMergeScriptForwarding:
    """``hf_script.render_merge_script`` 只是 ``deploy.render_merge_script`` 的转发."""

    def test_forwards_to_deploy_verbatim(self, lora: LoRAConfig):
        assert render_merge_script(lora) == deploy.render_merge_script(lora)

    def test_kwargs_are_forwarded(self, lora: LoRAConfig):
        assert render_merge_script(
            lora, base_model="Qwen/Qwen2.5-7B", output_dir="outputs/merged"
        ) == deploy.render_merge_script(
            lora, base_model="Qwen/Qwen2.5-7B", output_dir="outputs/merged"
        )

    def test_is_valid_python(self, lora: LoRAConfig):
        compile(render_merge_script(lora), "merge_adapter.py", "exec")
