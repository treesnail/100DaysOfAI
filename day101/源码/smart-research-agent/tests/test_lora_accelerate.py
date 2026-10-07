"""多卡与混合精度算术测试（M5-D4）：把"D 个设备呢？"这个问题钉成数字.

本文件守的是 ``peft/accelerate.py`` 里那几组**只靠算术就能决定对错**的数字：

- 有效批 = ``per_device × accum × devices``，以及随之而来的学习率缩放
  （批 8 → 32 时 linear 给 4 倍、sqrt 给 2 倍、``none`` 给 1 倍）；
- 分片策略对**每设备显存**的影响：``ddp`` 与 day051 的 ``plan_memory`` 逐位
  相同；``zero2`` 只切优化器状态与梯度（**参数不分片**，所以 LoRA 场景下
  7B 每设备显存只从 12.6450 GiB 降到 12.5864 GiB，几乎不降）；``zero3``
  连参数一起切，7B / 4 卡时**恰好**是 ddp 的 1/4（12.6450 → 3.1613 GiB）；
- ``communication_bytes_per_step`` 里 ZeRO-3 多出的那一项是**两倍参数规模**
  的 all-gather（``2 × parameter_parameters × bytes``）；
- ``plan_distributed`` 的五类告警各自的触发条件——它们每一条都对应一个
  真实会踩的坑，而不是"防御性编程"；
- 生成物 ``accelerate config`` / DeepSpeed ZeRO 配置 / ``accelerate launch``
  命令行的字段，以及 ``render_config_yaml`` 与 ``yaml.safe_load`` 的往返。

所有断言都只依赖标准库与 PyYAML，不需要 GPU、不需要 ``accelerate``。
"""

from __future__ import annotations

import json

import pytest
import yaml

from smart_research_agent.peft.accelerate import (
    MIXED_PRECISION_PROFILES,
    MIXED_PRECISION_VALUES,
    SHARDING_STRATEGIES,
    accelerate_config,
    communication_bytes_per_step,
    deepspeed_zero_config,
    device_fit_summary,
    launch_command,
    mixed_precision_profile,
    parse_gib,
    per_device_memory,
    plan_distributed,
    render_config_yaml,
    scale_learning_rate,
)
from smart_research_agent.peft.config import LoRAConfig, PEFTConfigError
from smart_research_agent.peft.memory import plan_memory
from smart_research_agent.peft.targets import MODEL_SPECS
from smart_research_agent.sft.args import SFTConfigError, SFTTrainingArgs

#: 本文件统一用 7B 规格：它是"多卡才跑得动"的那个量级（全参 75.31 GiB / 卡）
SPEC = MODEL_SPECS["llama-2-7b"]

#: 7B 上的 LoRA 配置（适配全部 7 个投影，可训练 8 388 608 个参数）
LORA = LoRAConfig(r=8, lora_alpha=16, target_modules="attention_all")

#: 缺省设备数（4 卡：strategy 之间的差别在这个倍数上看得最清楚）
DEVICES = 4

#: 缺省训练集规模（1000 条，足够让 steps_per_epoch 不为 0）
TRAIN_SIZE = 1000


def make_plan(
    *,
    args: SFTTrainingArgs,
    devices: int = DEVICES,
    strategy: str = "ddp",
    train_size: int = TRAIN_SIZE,
    lora_config: LoRAConfig | None = None,
    **extra,
):
    """按本文件统一的 7B 规格构造一份多卡计划（测试里只写要变的那几项）."""
    return plan_distributed(
        SPEC,
        args,
        devices=devices,
        strategy=strategy,
        train_size=train_size,
        lora_config=lora_config,
        **extra,
    )


class TestMixedPrecisionProfiles:
    """混合精度画像表：字节数一样、指数位不同，后果是"要不要 loss scaling"."""

    def test_three_profiles_are_registered(self):
        assert set(MIXED_PRECISION_PROFILES) == {"bf16", "fp16", "fp32"}

    def test_bf16_profile_fields(self):
        profile = MIXED_PRECISION_PROFILES["bf16"]
        assert profile.name == "bf16"
        assert profile.torch_dtype == "torch.bfloat16"
        assert profile.bytes_per_parameter == 2
        assert profile.needs_loss_scaling is False
        assert len(profile.notes) == 2

    def test_fp16_needs_loss_scaling(self):
        """fp16 的 5 位指数动态范围窄——这就是它必须配梯度缩放的全部理由."""
        profile = MIXED_PRECISION_PROFILES["fp16"]
        assert profile.torch_dtype == "torch.float16"
        assert profile.bytes_per_parameter == 2
        assert profile.needs_loss_scaling is True

    def test_fp32_profile_is_the_reference_point(self):
        profile = MIXED_PRECISION_PROFILES["fp32"]
        assert profile.torch_dtype == "torch.float32"
        assert profile.bytes_per_parameter == 4
        assert profile.needs_loss_scaling is False

    def test_fp16_and_bf16_have_identical_footprint(self):
        """两者都是 2 字节：选 bf16 不是省显存，而是省掉 loss scaling 的麻烦."""
        assert (
            MIXED_PRECISION_PROFILES["fp16"].bytes_per_parameter
            == MIXED_PRECISION_PROFILES["bf16"].bytes_per_parameter
            == 2
        )

    def test_lookup_returns_the_table_entry(self):
        assert mixed_precision_profile("bf16") is MIXED_PRECISION_PROFILES["bf16"]
        assert mixed_precision_profile("fp32").bytes_per_parameter == 4

    def test_unknown_name_rejected(self):
        with pytest.raises(PEFTConfigError, match="未知的混合精度"):
            mixed_precision_profile("fp8")

    def test_supported_value_tuple(self):
        assert MIXED_PRECISION_VALUES == ("no", "fp16", "bf16")

    def test_sharding_strategies_order_is_by_memory(self):
        """顺序即"每设备显存从大到小"——报告与告警都依赖这个顺序."""
        assert SHARDING_STRATEGIES == ("ddp", "zero2", "zero3")


class TestScaleLearningRate:
    """学习率缩放：linear / sqrt / none 三条法则，以及三处边界."""

    def test_linear_scales_with_batch_ratio(self):
        """批 8 → 32 是 4 倍，线性法则给 4 倍."""
        assert scale_learning_rate(
            1e-4, base_global_batch_size=8, new_global_batch_size=32, mode="linear"
        ) == pytest.approx(4e-4)

    def test_sqrt_takes_the_square_root(self):
        assert scale_learning_rate(
            1e-4, base_global_batch_size=8, new_global_batch_size=32, mode="sqrt"
        ) == pytest.approx(2e-4)

    def test_none_is_identity(self):
        """``none`` 不是"错"，但它会命中 plan_distributed 的一条告警."""
        assert scale_learning_rate(
            1e-4, base_global_batch_size=8, new_global_batch_size=32, mode="none"
        ) == pytest.approx(1e-4)

    def test_default_mode_is_linear(self):
        assert scale_learning_rate(
            2e-4, base_global_batch_size=8, new_global_batch_size=32
        ) == pytest.approx(8e-4)

    def test_sqrt_equals_linear_for_quadrupled_batch(self):
        linear = scale_learning_rate(
            1.0, base_global_batch_size=8, new_global_batch_size=32, mode="linear"
        )
        sqrt = scale_learning_rate(
            1.0, base_global_batch_size=8, new_global_batch_size=32, mode="sqrt"
        )
        assert sqrt == pytest.approx(linear**0.5)
        assert sqrt == pytest.approx(2.0)

    def test_unknown_mode_rejected(self):
        with pytest.raises(PEFTConfigError, match="未知的缩放法则"):
            scale_learning_rate(
                1e-4, base_global_batch_size=8, new_global_batch_size=32, mode="cosine"
            )

    @pytest.mark.parametrize(
        "changes",
        [
            {"base_global_batch_size": 0},
            {"base_global_batch_size": -8},
            {"new_global_batch_size": 0},
        ],
    )
    def test_non_positive_batch_rejected(self, changes):
        kwargs = {"base_global_batch_size": 8, "new_global_batch_size": 32}
        kwargs.update(changes)
        with pytest.raises(PEFTConfigError, match="批大小必须为正整数"):
            scale_learning_rate(1e-4, **kwargs)

    @pytest.mark.parametrize("learning_rate", [0.0, -1e-4])
    def test_non_positive_learning_rate_rejected(self, learning_rate):
        with pytest.raises(PEFTConfigError, match="学习率必须为正数"):
            scale_learning_rate(
                learning_rate, base_global_batch_size=8, new_global_batch_size=32
            )


class TestPerDeviceMemory:
    """分片策略对**每设备**显存的影响（本文件的核心数字）."""

    def _lora(self, strategy: str, *, devices: int = DEVICES):
        return per_device_memory(
            SPEC, strategy=strategy, devices=devices, lora_config=LORA
        )

    def test_ddp_reuses_plan_memory_verbatim(self):
        """DDP 在每个设备上放一份完整副本——单设备预算与单卡**逐位相同**."""
        baseline = plan_memory(SPEC, strategy="lora", lora_config=LORA)
        per_device = self._lora("ddp")
        assert per_device.total_bytes == baseline.total_bytes
        assert per_device.as_gib == baseline.as_gib
        assert per_device.items == baseline.items
        assert per_device.strategy == baseline.strategy
        assert "每个设备一份完整副本" in " ".join(per_device.notes)

    def test_ddp_lora_7b_is_12_6450_gib(self):
        assert self._lora("ddp").as_gib == pytest.approx(12.6450, abs=1e-3)

    def test_zero2_does_not_shard_parameters(self):
        """ZeRO-2 只切优化器状态与梯度：**参数仍是一份完整的**."""
        ddp = self._lora("ddp")
        zero2 = self._lora("zero2")
        assert zero2.base_weight_bytes == ddp.base_weight_bytes
        assert zero2.adapter_weight_bytes == ddp.adapter_weight_bytes
        assert zero2.base_gradient_bytes == ddp.base_gradient_bytes == 0

    def test_zero2_shards_gradients_and_optimizer_states(self):
        ddp = self._lora("ddp")
        zero2 = self._lora("zero2")
        assert zero2.adapter_gradient_bytes == ddp.adapter_gradient_bytes // DEVICES
        assert zero2.adapter_optimizer_bytes == ddp.adapter_optimizer_bytes // DEVICES
        assert zero2.strategy == "lora+zero2"

    def test_zero2_barely_saves_on_lora(self):
        """LoRA 上最大的一项是**冻结的基座权重**，切它才能省——zero2 只省 0.5%."""
        ddp = self._lora("ddp")
        zero2 = self._lora("zero2")
        assert zero2.as_gib == pytest.approx(12.5864, abs=1e-3)
        assert zero2.as_gib < ddp.as_gib
        assert ddp.as_gib / zero2.as_gib < 1.01

    def test_zero3_shards_parameters_as_well(self):
        ddp = self._lora("ddp")
        zero3 = self._lora("zero3")
        assert zero3.base_weight_bytes == ddp.base_weight_bytes // DEVICES
        assert zero3.adapter_weight_bytes == ddp.adapter_weight_bytes // DEVICES
        assert zero3.strategy == "lora+zero3"

    def test_zero3_lora_7b_is_a_quarter_of_ddp(self):
        """7B LoRA / 4 卡：zero3 恰好是 ddp 的 1/4（12.6450 → 3.1613 GiB）."""
        ddp = self._lora("ddp")
        zero3 = self._lora("zero3")
        assert zero3.total_bytes == ddp.total_bytes // DEVICES
        assert zero3.as_gib == pytest.approx(3.1613, abs=1e-3)
        assert ddp.as_gib / zero3.as_gib == pytest.approx(4.0, abs=1e-3)

    def test_full_finetune_zero2_is_28_2404_gib(self):
        """全参微调里最大的一项是优化器状态（8 字节/参数），所以 zero2 一下子省掉 62%."""
        zero2 = per_device_memory(
            SPEC, strategy="zero2", devices=DEVICES, full_finetune=True
        )
        assert zero2.as_gib == pytest.approx(28.2404, abs=1e-3)

    def test_full_finetune_zero3_is_18_8269_gib(self):
        ddp = per_device_memory(
            SPEC, strategy="ddp", devices=DEVICES, full_finetune=True
        )
        zero3 = per_device_memory(
            SPEC, strategy="zero3", devices=DEVICES, full_finetune=True
        )
        assert ddp.as_gib == pytest.approx(75.3077, abs=1e-3)
        assert zero3.total_bytes == ddp.total_bytes // DEVICES
        assert zero3.as_gib == pytest.approx(18.8269, abs=1e-3)

    def test_single_device_is_always_a_full_copy(self):
        """``devices=1`` 时分片无从谈起：任何策略都退化为完整副本."""
        for strategy in SHARDING_STRATEGIES:
            per_device = self._lora(strategy, devices=1)
            assert per_device.total_bytes == self._lora("ddp").total_bytes
            assert "每个设备一份完整副本" in " ".join(per_device.notes)

    def test_unknown_strategy_rejected(self):
        with pytest.raises(PEFTConfigError, match="未知的分片策略"):
            self._lora("zero1")

    @pytest.mark.parametrize("devices", [0, -1])
    def test_non_positive_devices_rejected(self, devices):
        with pytest.raises(PEFTConfigError, match="设备数必须为正整数"):
            self._lora("ddp", devices=devices)

    def test_missing_lora_config_rejected(self):
        with pytest.raises(PEFTConfigError, match="必须提供 lora_config"):
            per_device_memory(SPEC, strategy="ddp", devices=DEVICES)


class TestCommunicationBytesPerStep:
    """每步通信量：ZeRO-2 只有梯度 all-reduce，ZeRO-3 额外搬两遍参数."""

    def test_zero2_is_the_gradient_all_reduce(self):
        """ring all-reduce 的每卡流量约 ``2(N-1)/N × 参数字节``."""
        value = communication_bytes_per_step(
            gradient_parameters=1000,
            parameter_parameters=100_000,
            devices=4,
            bytes_per_parameter=2,
            stage="zero2",
        )
        assert value == int(2 * (4 - 1) / 4 * 1000 * 2) == 3000

    def test_zero3_adds_two_parameter_all_gathers(self):
        gradient = communication_bytes_per_step(
            gradient_parameters=1000,
            parameter_parameters=100_000,
            devices=4,
            bytes_per_parameter=2,
            stage="zero2",
        )
        zero3 = communication_bytes_per_step(
            gradient_parameters=1000,
            parameter_parameters=100_000,
            devices=4,
            bytes_per_parameter=2,
            stage="zero3",
        )
        assert zero3 - gradient == 2 * 100_000 * 2 == 400_000
        assert zero3 == 403_000

    def test_parameter_defaults_to_gradient_parameters(self):
        """不给参数规模时按梯度参数算——于是差值是两倍的梯度规模."""
        zero2 = communication_bytes_per_step(
            gradient_parameters=1000, devices=4, bytes_per_parameter=2, stage="zero2"
        )
        zero3 = communication_bytes_per_step(
            gradient_parameters=1000, devices=4, bytes_per_parameter=2, stage="zero3"
        )
        assert zero3 - zero2 == 2 * 1000 * 2 == 4000

    def test_communication_grows_with_devices(self):
        small = communication_bytes_per_step(
            gradient_parameters=1000, devices=2, bytes_per_parameter=2, stage="zero2"
        )
        large = communication_bytes_per_step(
            gradient_parameters=1000, devices=8, bytes_per_parameter=2, stage="zero2"
        )
        assert small == 2000
        assert large == 3500
        assert large > small

    def test_invalid_stage_rejected(self):
        with pytest.raises(PEFTConfigError, match="通信量只对 zero2 / zero3 建模"):
            communication_bytes_per_step(gradient_parameters=1000, devices=4, stage="ddp")

    @pytest.mark.parametrize(
        "changes",
        [
            {"gradient_parameters": 0},
            {"devices": 0},
            {"bytes_per_parameter": 0},
        ],
    )
    def test_non_positive_basics_rejected(self, changes):
        kwargs = {
            "gradient_parameters": 1000,
            "devices": 4,
            "bytes_per_parameter": 2,
            "stage": "zero2",
        }
        kwargs.update(changes)
        with pytest.raises(PEFTConfigError, match="必须为正整数"):
            communication_bytes_per_step(**kwargs)

    def test_non_positive_parameter_parameters_rejected(self):
        with pytest.raises(PEFTConfigError, match="parameter_parameters 必须为正整数"):
            communication_bytes_per_step(
                gradient_parameters=1000,
                parameter_parameters=0,
                devices=4,
                stage="zero3",
            )


class TestPlanDistributedArithmetic:
    """``plan_distributed`` 的算术：批、步、学习率、显存、通信、落盘."""

    def test_global_batch_is_per_device_times_accum_times_devices(self):
        """``2 × 4 × 4 = 32``——单卡上的有效批 8 在多卡上被无声放大 4 倍."""
        plan = make_plan(args=SFTTrainingArgs(), lora_config=LORA)
        assert plan.global_batch_size == 32
        assert plan.global_micro_batch_size == 8
        assert plan.per_device_train_batch_size == 2
        assert plan.gradient_accumulation_steps == 4

    def test_step_arithmetic_matches_sft_args(self):
        plan = make_plan(args=SFTTrainingArgs(), lora_config=LORA)
        assert plan.micro_batches_per_epoch == TRAIN_SIZE // plan.global_micro_batch_size
        assert plan.steps_per_epoch == plan.micro_batches_per_epoch // 4
        assert (plan.micro_batches_per_epoch, plan.steps_per_epoch) == (125, 31)
        assert plan.total_steps == 93
        assert plan.total_steps == max(1, int(plan.steps_per_epoch * 3.0))
        assert plan.warmup_steps == 2

    def test_total_steps_lower_bound_is_one(self):
        """数据极少时也至少更新一次（下界 1，与 day050 的口径一致）."""
        plan = make_plan(
            args=SFTTrainingArgs(num_train_epochs=1.0, per_device_train_batch_size=1),
            train_size=100,
            lora_config=LORA,
        )
        assert plan.total_steps >= 1

    def test_learning_rate_is_scaled_against_effective_batch(self):
        """分母是**单卡有效批**（8），不是 ``per_device``——这条口径容易写错."""
        plan = make_plan(args=SFTTrainingArgs(), lora_config=LORA)
        assert plan.base_learning_rate == pytest.approx(2e-4)
        assert plan.scaling_mode == "linear"
        assert plan.scaled_learning_rate == pytest.approx(8e-4)

    def test_sqrt_scaling(self):
        plan = make_plan(args=SFTTrainingArgs(), lora_config=LORA, lr_scaling_mode="sqrt")
        assert plan.scaled_learning_rate == pytest.approx(4e-4)

    def test_per_device_memory_matches_helper(self):
        plan = make_plan(args=SFTTrainingArgs(), strategy="zero3", lora_config=LORA)
        helper = per_device_memory(SPEC, strategy="zero3", devices=DEVICES, lora_config=LORA)
        assert plan.per_device_gib == helper.as_gib
        assert plan.single_device_gib == pytest.approx(12.6450, abs=1e-3)
        assert plan.per_device_gib == pytest.approx(3.1613, abs=1e-3)
        assert plan.memory_saving_factor == pytest.approx(4.0, abs=1e-3)

    def test_communication_matches_helper(self):
        """LoRA 的梯度项很小、参数项很大：ZeRO-3 的通信几乎全花在搬基座上."""
        plan = make_plan(args=SFTTrainingArgs(), strategy="zero3", lora_config=LORA)
        expected = communication_bytes_per_step(
            gradient_parameters=8_388_608,
            parameter_parameters=SPEC.total_parameters(),
            devices=DEVICES,
            stage="zero3",
        )
        assert plan.communication_bytes_per_step == expected
        assert plan.communication_bytes_total == expected * plan.total_steps

    def test_zero3_communication_dwarfs_zero2_on_lora(self):
        zero2 = make_plan(args=SFTTrainingArgs(), strategy="zero2", lora_config=LORA)
        zero3 = make_plan(args=SFTTrainingArgs(), strategy="zero3", lora_config=LORA)
        assert zero3.communication_bytes_per_step > zero2.communication_bytes_per_step

    def test_checkpoint_bytes_only_covers_trainable_parameters(self):
        """LoRA 只需落盘适配器（8 388 608 × 2 B = 16 MB），比全参小三个数量级."""
        plan = make_plan(args=SFTTrainingArgs(), lora_config=LORA)
        assert plan.checkpoint_bytes == 8_388_608 * 2 == 16_777_216

    def test_full_finetune_communication_uses_all_parameters(self):
        plan = make_plan(args=SFTTrainingArgs(), full_finetune=True)
        assert plan.checkpoint_bytes == SPEC.total_parameters() * 2

    def test_summary_line_reports_the_derived_numbers(self):
        plan = make_plan(args=SFTTrainingArgs(), strategy="zero3", lora_config=LORA)
        line = plan.summary_line()
        assert line.startswith("4× zero3 | ")
        assert "全局批 32" in line
        assert "总步数 93" in line
        assert "lr 0.0002 → 0.0008" in line
        assert f"每设备 {plan.per_device_gib:.2f} GiB" in line

    def test_to_dict_is_json_serializable_and_complete(self):
        plan = make_plan(args=SFTTrainingArgs(), lora_config=LORA)
        payload = plan.to_dict()
        assert json.loads(json.dumps(payload)) == payload
        assert {
            "model",
            "devices",
            "strategy",
            "global_batch_size",
            "steps_per_epoch",
            "total_steps",
            "scaled_learning_rate",
            "per_device_gib",
            "memory_saving_factor",
            "communication_bytes_per_step",
            "checkpoint_bytes",
            "warnings",
        } <= set(payload)
        assert payload["model"] == "llama-2-7b"
        assert payload["devices"] == DEVICES

    def test_clean_config_has_no_warnings(self):
        """基线（bf16 + ddp + linear + 足够数据）不该报任何警."""
        plan = make_plan(args=SFTTrainingArgs(), lora_config=LORA)
        assert plan.warnings == []

    def test_unknown_strategy_rejected(self):
        with pytest.raises(PEFTConfigError, match="未知的分片策略"):
            make_plan(args=SFTTrainingArgs(), strategy="zero4", lora_config=LORA)

    @pytest.mark.parametrize("devices", [0, -2])
    def test_non_positive_devices_rejected(self, devices):
        with pytest.raises(PEFTConfigError, match="设备数必须为正整数"):
            make_plan(args=SFTTrainingArgs(), devices=devices, lora_config=LORA)

    def test_unknown_mixed_precision_rejected(self):
        with pytest.raises(PEFTConfigError, match="未知的混合精度"):
            make_plan(args=SFTTrainingArgs(), lora_config=LORA, mixed_precision="fp8")

    def test_invalid_sft_args_are_validated_first(self):
        with pytest.raises(SFTConfigError, match="per_device_train_batch_size"):
            make_plan(args=SFTTrainingArgs(per_device_train_batch_size=0), lora_config=LORA)


class TestPlanDistributedWarnings:
    """五类告警：每一条都对应一个真实会踩的坑（各自一条用例）."""

    def test_none_scaling_warns_about_smaller_gradients(self):
        plan = make_plan(args=SFTTrainingArgs(), lora_config=LORA, lr_scaling_mode="none")
        warnings = "\n".join(plan.warnings)
        assert "lr_scaling_mode=none" in warnings
        assert "全局批从 8 放大到 32" in warnings

    def test_single_device_does_not_warn_about_scaling(self):
        """只有一个设备时全局批没有被放大，这条告警不该出现."""
        plan = make_plan(
            args=SFTTrainingArgs(), devices=1, lora_config=LORA, lr_scaling_mode="none"
        )
        assert not any("lr_scaling_mode=none" in item for item in plan.warnings)

    def test_zero_warmup_warns(self):
        plan = make_plan(args=SFTTrainingArgs(warmup_ratio=0.0), lora_config=LORA)
        assert any("warmup_steps 仍是 0" in item for item in plan.warnings)

    def test_global_batch_larger_than_train_set_warns(self):
        """32 条的全局批配 10 条训练集：一个 epoch 连一步都走不完."""
        plan = make_plan(args=SFTTrainingArgs(), train_size=10, lora_config=LORA)
        assert plan.steps_per_epoch == 0
        assert any("超过训练集" in item for item in plan.warnings)

    def test_zero3_communication_tradeoff_warns(self):
        """LoRA + zero3：显存省 4 倍，通信量却放大三个数量级——必须告警.

        **这条告警的口径是"通信量 vs 显存收益"，不是"显存没省多少"。**
        本课实现时先写成后者，结果它几乎不可达：``zero3`` 把六项全部按设备数
        分片，节省倍数恒为设备数，永远不会"省不到 1.2 倍"（单卡时更是恒为 1）。
        换成通信量的比值之后，它每一次都会命中，而且指出的是真实的代价。
        """
        plan = make_plan(args=SFTTrainingArgs(), devices=4, strategy="zero3", lora_config=LORA)
        assert plan.memory_saving_factor == pytest.approx(4.0)
        assert any("每步通信量是 zero2 的" in item for item in plan.warnings)

    def test_zero3_with_enough_saving_does_not_warn(self):
        plan = make_plan(args=SFTTrainingArgs(), strategy="zero3", lora_config=LORA)
        assert plan.memory_saving_factor == pytest.approx(4.0, abs=1e-3)
        assert not any("分片收益小于通信代价" in item for item in plan.warnings)

    def test_fp16_warns_about_dynamic_range(self):
        plan = make_plan(args=SFTTrainingArgs(), lora_config=LORA, mixed_precision="fp16")
        assert any("fp16 的动态范围窄" in item for item in plan.warnings)

    def test_bf16_has_no_precision_warning(self):
        plan = make_plan(args=SFTTrainingArgs(), lora_config=LORA, mixed_precision="bf16")
        assert not any("fp16" in item for item in plan.warnings)


class TestAccelerateConfig:
    """``accelerate config`` 的字段：ddp → MULTI_GPU，ZeRO → DEEPSPEED + 配置文件."""

    def test_ddp_uses_multi_gpu(self):
        payload = accelerate_config(devices=DEVICES, strategy="ddp")
        assert payload["distributed_type"] == "MULTI_GPU"
        assert "deepspeed_config" not in payload

    def test_common_fields_match_official_defaults(self):
        payload = accelerate_config(devices=DEVICES, strategy="ddp")
        assert payload["compute_environment"] == "LOCAL_MACHINE"
        assert payload["num_processes"] == DEVICES
        assert payload["machine_rank"] == 0
        assert payload["main_training_function"] == "main"
        assert payload["mixed_precision"] == "bf16"
        assert payload["use_cpu"] is False
        assert payload["gpu_ids"] == "all"

    def test_zero2_switches_to_deepspeed(self):
        payload = accelerate_config(
            devices=DEVICES, strategy="zero2", deepspeed_config_file="ds.json"
        )
        assert payload["distributed_type"] == "DEEPSPEED"
        assert payload["deepspeed_config"] == {
            "deepspeed_config_file": "ds.json",
            "zero3_init_flag": False,
        }

    def test_zero3_sets_init_flag(self):
        payload = accelerate_config(
            devices=DEVICES, strategy="zero3", deepspeed_config_file="ds.json"
        )
        assert payload["deepspeed_config"]["zero3_init_flag"] is True

    @pytest.mark.parametrize("strategy", ["zero2", "zero3"])
    def test_zero_stages_require_a_config_file(self, strategy):
        with pytest.raises(PEFTConfigError, match="需要 deepspeed_config_file"):
            accelerate_config(devices=DEVICES, strategy=strategy)

    def test_ddp_must_not_carry_a_deepspeed_file(self):
        with pytest.raises(PEFTConfigError, match="ddp 不该带 deepspeed_config_file"):
            accelerate_config(devices=DEVICES, strategy="ddp", deepspeed_config_file="ds.json")

    def test_unknown_strategy_rejected(self):
        with pytest.raises(PEFTConfigError, match="未知的分片策略"):
            accelerate_config(devices=DEVICES, strategy="zero1")

    @pytest.mark.parametrize("devices", [0, -1])
    def test_non_positive_devices_rejected(self, devices):
        with pytest.raises(PEFTConfigError, match="设备数必须为正整数"):
            accelerate_config(devices=devices, strategy="ddp")

    def test_negative_machine_rank_rejected(self):
        with pytest.raises(PEFTConfigError, match="machine_rank 不能为负数"):
            accelerate_config(devices=DEVICES, strategy="ddp", machine_rank=-1)

    def test_fp32_is_written_as_no(self):
        """``accelerate`` 的取值是 ``no`` 而不是 ``fp32``——写错它会被静默忽略."""
        payload = accelerate_config(devices=DEVICES, strategy="ddp", mixed_precision="fp32")
        assert payload["mixed_precision"] == "no"

    def test_unknown_mixed_precision_rejected(self):
        with pytest.raises(PEFTConfigError, match="未知的混合精度"):
            accelerate_config(devices=DEVICES, strategy="ddp", mixed_precision="int8")


class TestDeepSpeedZeroConfig:
    """DeepSpeed ZeRO 配置：stage 2 / 3 的差别与 ``train_batch_size="auto"``."""

    def test_stage2_has_no_parameter_offload(self):
        payload = deepspeed_zero_config(stage=2)
        zero = payload["zero_optimization"]
        assert zero["stage"] == 2
        assert zero["offload_optimizer"] == {"device": "none"}
        assert "offload_param" not in zero
        assert "stage3_gather_16bit_weights_on_model_save" not in zero

    def test_stage3_gathers_weights_and_offloads_parameters(self):
        """不开 ``stage3_gather_16bit_weights_on_model_save``，落盘的模型是"半份"的."""
        payload = deepspeed_zero_config(stage=3)
        zero = payload["zero_optimization"]
        assert zero["stage"] == 3
        assert zero["stage3_gather_16bit_weights_on_model_save"] is True
        assert zero["offload_param"] == {"device": "none"}
        assert "offload_param" in zero

    @pytest.mark.parametrize("stage", [0, 1, 4, 5])
    def test_invalid_stage_rejected(self, stage):
        with pytest.raises(PEFTConfigError, match="ZeRO 只支持 stage 2 / 3"):
            deepspeed_zero_config(stage=stage)

    def test_offload_flags_switch_devices_to_cpu(self):
        payload = deepspeed_zero_config(
            stage=3, offload_optimizer=True, offload_parameters=True
        )
        zero = payload["zero_optimization"]
        assert zero["offload_optimizer"] == {"device": "cpu"}
        assert zero["offload_param"] == {"device": "cpu"}

    def test_batch_size_fields_are_auto(self):
        """写死批大小会让"配置里的批大小"与"训练日志里的批大小"对不上."""
        payload = deepspeed_zero_config(stage=2)
        assert payload["train_batch_size"] == "auto"
        assert payload["train_micro_batch_size_per_gpu"] == "auto"
        assert payload["gradient_accumulation_steps"] == "auto"

    def test_precision_block_follows_mixed_precision(self):
        assert deepspeed_zero_config(stage=2)["bf16"] == {"enabled": True}
        assert deepspeed_zero_config(stage=2, mixed_precision="fp32")["fp32"] == {"enabled": True}

    def test_unknown_mixed_precision_rejected(self):
        with pytest.raises(PEFTConfigError, match="未知的混合精度"):
            deepspeed_zero_config(stage=2, mixed_precision="tf32")


class TestRenderConfigYaml:
    """``render_config_yaml``：手写渲染 + ``yaml.safe_load`` 往返逐键相等."""

    @pytest.mark.parametrize("strategy", ["ddp", "zero2", "zero3"])
    def test_roundtrip_is_key_for_key_equal(self, strategy):
        payload = accelerate_config(
            devices=DEVICES,
            strategy=strategy,
            mixed_precision="bf16",
            deepspeed_config_file=None if strategy == "ddp" else "ds.json",
        )
        assert yaml.safe_load(render_config_yaml(payload)) == payload

    def test_roundtrip_keeps_the_nested_deepspeed_block(self):
        payload = accelerate_config(
            devices=8, strategy="zero3", mixed_precision="fp16", deepspeed_config_file="zero3.json"
        )
        parsed = yaml.safe_load(render_config_yaml(payload))
        assert parsed["deepspeed_config"] == {
            "deepspeed_config_file": "zero3.json",
            "zero3_init_flag": True,
        }
        assert parsed["num_processes"] == 8
        assert parsed["mixed_precision"] == "fp16"

    def test_booleans_and_scalars_are_rendered_unquoted(self):
        text = render_config_yaml(accelerate_config(devices=DEVICES, strategy="ddp"))
        assert "use_cpu: false" in text
        assert "num_processes: 4" in text
        assert "gpu_ids: all" in text
        assert text.endswith("\n")

    def test_nested_block_is_indented_by_two_spaces(self):
        payload = accelerate_config(devices=2, strategy="zero2", deepspeed_config_file="ds.json")
        lines = render_config_yaml(payload).splitlines()
        assert "deepspeed_config:" in lines
        assert "  deepspeed_config_file: ds.json" in lines
        assert "  zero3_init_flag: false" in lines


class TestLaunchCommand:
    """``accelerate launch``：``--num_processes`` 只在多卡时出现."""

    def test_single_device_omits_num_processes(self):
        command = launch_command("train.py", devices=1)
        assert command == "accelerate launch --config_file accelerate_config.yaml train.py"
        assert "--num_processes" not in command

    def test_multi_device_adds_num_processes(self):
        command = launch_command("train.py", devices=DEVICES)
        assert command == (
            "accelerate launch --config_file accelerate_config.yaml "
            "--num_processes 4 train.py"
        )
        assert "--num_processes 4" in command

    def test_strategy_does_not_change_the_flags(self):
        """三种策略的命令行差别只在配置内容——启动方式是一样的."""
        zero3 = launch_command(
            "train.py", devices=DEVICES, strategy="zero3", config_file="zero3.yaml"
        )
        assert zero3 == "accelerate launch --config_file zero3.yaml --num_processes 4 train.py"

    def test_script_is_last_and_config_file_first(self):
        parts = launch_command("scripts/train.py", devices=2).split()
        assert parts[0:3] == ["accelerate", "launch", "--config_file"]
        assert parts[-1] == "scripts/train.py"

    def test_empty_script_rejected(self):
        with pytest.raises(PEFTConfigError, match="script 不能为空"):
            launch_command("   ", devices=2)

    @pytest.mark.parametrize("devices", [0, -3])
    def test_non_positive_devices_rejected(self, devices):
        with pytest.raises(PEFTConfigError, match="设备数必须为正整数"):
            launch_command("train.py", devices=devices)

    def test_unknown_strategy_rejected(self):
        with pytest.raises(PEFTConfigError, match="未知的分片策略"):
            launch_command("train.py", devices=DEVICES, strategy="zero9")

    def test_empty_config_file_rejected(self):
        with pytest.raises(PEFTConfigError, match="config_file 不能为空"):
            launch_command("train.py", devices=DEVICES, config_file=" ")


class TestBudgetHelpers:
    """``parse_gib`` / ``device_fit_summary``：把"能不能放下"翻译成一句话."""

    def test_parse_gib_uses_binary_units(self):
        assert parse_gib(1.0) == 1024**3
        assert parse_gib(2.5) == int(2.5 * 1024**3)

    @pytest.mark.parametrize("budget", [0.0, -1.0])
    def test_parse_gib_rejects_non_positive(self, budget):
        with pytest.raises(PEFTConfigError, match="预算必须为正数"):
            parse_gib(budget)

    def test_device_fit_summary_when_it_fits(self):
        plan = make_plan(args=SFTTrainingArgs(), strategy="zero3", lora_config=LORA)
        summary = device_fit_summary(plan, budget_gib=80.0)
        assert summary["fits"] is True
        assert summary["devices"] == DEVICES
        assert summary["strategy"] == "zero3"
        assert summary["per_device_gib"] == pytest.approx(3.1613, abs=1e-3)
        assert summary["headroom_gib"] == pytest.approx(80.0 - 3.1613, abs=1e-3)
        assert "激活" in summary["note"]

    def test_device_fit_summary_when_it_does_not_fit(self):
        plan = make_plan(args=SFTTrainingArgs(), strategy="ddp", lora_config=LORA)
        summary = device_fit_summary(plan, budget_gib=1.0)
        assert summary["fits"] is False
        assert summary["headroom_gib"] < 0.0

    def test_device_fit_summary_rejects_non_positive_budget(self):
        plan = make_plan(args=SFTTrainingArgs(), lora_config=LORA)
        with pytest.raises(PEFTConfigError, match="预算必须为正数"):
            device_fit_summary(plan, budget_gib=0.0)
