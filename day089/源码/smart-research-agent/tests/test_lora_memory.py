"""显存算术测试（M5-D3）：全参 / LoRA / QLoRA 的六项预算与工具函数.

本文件守的是"为什么需要 LoRA"的量化版本：优化器状态 **8 字节/参数**
（AdamW 的两个 fp32 矩）、LoRA 把"基座梯度 + 基座优化器状态"整块删掉、
QLoRA 只把**基座权重**从 2 字节压到 0.515869 字节。7B 上的三个数字
（``6.738e9 × 12 = 80.86 GB`` 全参、``13.48 GB`` LoRA 基座、``3.48 GB``
QLoRA 基座）全部由公式手算后写成断言。
"""

from __future__ import annotations

import pytest

from smart_research_agent.peft.config import LoRAConfig, PEFTConfigError, QLoRAConfig
from smart_research_agent.peft.memory import (
    COMPUTE_DTYPE_BYTES,
    DEVICE_BUDGETS,
    GIB,
    OPTIMIZER_STATES,
    STRATEGIES,
    MemoryBreakdown,
    adamw_state_bytes,
    bytes_per_parameter_of_16bit,
    compare_strategies,
    device_fit_table,
    format_bytes,
    logistic_estimate_hours,
    minimal_device,
    parameter_scale,
    parse_size,
    plan_memory,
    quantization_memory_note,
    savings_table,
)
from smart_research_agent.peft.targets import MODEL_SPECS

#: 两个公开规格：参数量可在对应模型的 config.json 里逐位核对
LLAMA = MODEL_SPECS["llama-2-7b"]
QWEN = MODEL_SPECS["qwen3-0.6b"]

#: 公开参数量（规格写错任一字段，这两个总数立刻对不上）
LLAMA_PARAMETERS = 6_738_415_616
QWEN_PARAMETERS = 596_049_920

#: 默认 LoRA（``attention`` / r=8）在 llama-2-7b 上的可训练参数
LLAMA_ADAPTER_PARAMETERS = 4_194_304


def make_breakdown(**overrides) -> MemoryBreakdown:
    """构造一份显式取值的预算（不经过 ``plan_memory``，便于逐项核对加法）."""
    payload = {
        "model": "demo",
        "strategy": "lora",
        "optimizer": "adamw_torch",
        "compute_dtype": "bfloat16",
        "base_parameters": 1000,
        "trainable_parameters": 10,
        "base_weight_bytes": 2000,
        "base_gradient_bytes": 0,
        "base_optimizer_bytes": 0,
        "adapter_weight_bytes": 20,
        "adapter_gradient_bytes": 20,
        "adapter_optimizer_bytes": 80,
        "base_bits_per_parameter": 16.0,
    }
    payload.update(overrides)
    return MemoryBreakdown(**payload)


class TestConstants:
    """常量表：显存口径的单一事实来源."""

    def test_gib_is_1024_cubed(self):
        """用 GiB 而不是 GB：24 GB 的卡实际是 24 GiB（25.77 GB）."""
        assert GIB == 1024**3

    def test_strategy_order_is_report_order(self):
        assert STRATEGIES == ("full", "lora", "qlora")

    def test_optimizer_states(self):
        """AdamW 的两个 fp32 矩是 8 字节；8-bit Adam 降到 2；SGD 不占."""
        assert OPTIMIZER_STATES["adamw_torch"] == 8
        assert OPTIMIZER_STATES["adamw_bnb_8bit"] == 2
        assert OPTIMIZER_STATES["sgd"] == 0

    def test_compute_dtype_bytes(self):
        assert COMPUTE_DTYPE_BYTES == {"bfloat16": 2, "float16": 2, "float32": 4}

    def test_device_budgets_have_24gib_card(self):
        assert DEVICE_BUDGETS["RTX 4090 (24 GiB)"] == 24.0
        assert DEVICE_BUDGETS["A100/H100 (80 GiB)"] == 80.0


class TestParseSize:
    """``parse_size``：``"24GiB"`` / ``"80GB"`` 必须解释到同一个坐标系."""

    def test_gib_is_identity_and_case_insensitive(self):
        assert parse_size("24GiB") == pytest.approx(24.0)
        assert parse_size("24gib") == pytest.approx(24.0)

    def test_bare_g_is_treated_as_gib(self):
        assert parse_size("80G") == pytest.approx(80.0)

    def test_gb_is_decimal(self):
        assert parse_size("80GB") == pytest.approx(80 * 1000**3 / GIB)

    def test_gb_and_gib_differ_for_same_number(self):
        """单位混用会给出错误答案：同样写 80，GB 比 GiB 小 7%."""
        assert parse_size("80GB") < parse_size("80GiB")

    def test_mib_and_m(self):
        assert parse_size("512MiB") == pytest.approx(0.5)
        assert parse_size("512M") == pytest.approx(0.5)

    def test_surrounding_whitespace_is_tolerated(self):
        assert parse_size("  16 GiB ") == pytest.approx(16.0)

    def test_missing_unit_rejected(self):
        with pytest.raises(PEFTConfigError, match="请带单位"):
            parse_size("24")

    def test_unknown_suffix_rejected(self):
        with pytest.raises(PEFTConfigError, match="请带单位"):
            parse_size("24TiB")

    def test_unit_only_rejected(self):
        with pytest.raises(PEFTConfigError, match="请带单位"):
            parse_size("GiB")

    def test_non_numeric_body_rejected(self):
        with pytest.raises(PEFTConfigError, match="无法解析容量"):
            parse_size("abcGiB")


class TestFormatBytes:
    """``format_bytes``：GiB 与 GB 必须一起给出（两者相差 7.4%）."""

    def test_one_gib_prints_both_units(self):
        assert format_bytes(GIB) == "1.00 GiB（1.07 GB）"

    def test_zero(self):
        assert format_bytes(0) == "0.00 GiB（0.00 GB）"

    def test_lora_base_prints_13_gib(self):
        text = format_bytes(13.48 * GIB)
        assert "13.48 GiB" in text
        assert "14.47 GB" in text

    def test_negative_rejected(self):
        with pytest.raises(PEFTConfigError, match="字节数不能为负"):
            format_bytes(-1)


class TestMemoryBreakdownArithmetic:
    """``MemoryBreakdown`` 的六项加法、判据与格式化."""

    def test_total_bytes_sums_six_items(self):
        plan = make_breakdown()
        assert plan.total_bytes == 2120
        assert plan.as_gib == pytest.approx(2120 / GIB)

    def test_items_keys_match_module_table(self):
        plan = make_breakdown()
        assert plan.items == {
            "base_weight_bytes": 2000,
            "base_gradient_bytes": 0,
            "base_optimizer_bytes": 0,
            "adapter_weight_bytes": 20,
            "adapter_gradient_bytes": 20,
            "adapter_optimizer_bytes": 80,
        }
        assert sum(plan.items.values()) == plan.total_bytes

    def test_fits_compares_as_gib_against_budget(self):
        plan = make_breakdown()
        assert plan.fits(1.0) is True
        assert plan.fits(2120 / GIB) is True

    def test_fits_rejects_non_positive_budget(self):
        with pytest.raises(PEFTConfigError, match="预算必须为正数"):
            make_breakdown().fits(0)

    def test_summary_line_format(self):
        assert make_breakdown().summary_line() == (
            "lora  | 可训练 10 | 总计 0.00 GiB（0.00 GB） | 基座权重 16.000 bit/参数"
        )

    def test_to_dict_adds_derived_values_and_is_json_ready(self):
        import json

        plan = make_breakdown(notes=["不含激活显存"])
        payload = plan.to_dict()
        assert payload["total_bytes"] == plan.total_bytes
        assert payload["as_gib"] == round(plan.as_gib, 4)
        assert payload["as_gb"] == round(plan.total_bytes / 1e9, 4)
        assert payload["items"] == plan.items
        assert json.loads(json.dumps(payload, ensure_ascii=False))["total_bytes"] == 2120


class TestModelSpecsSelfCheck:
    """规格自检：参数量与公开数字逐位一致（规格写错的唯一发现手段）."""

    def test_llama_2_7b_parameters(self):
        assert LLAMA.total_parameters() == LLAMA_PARAMETERS

    def test_qwen3_0_6b_parameters(self):
        assert QWEN.total_parameters() == QWEN_PARAMETERS

    def test_llama_layer_and_embedding_add_up(self):
        assert LLAMA.num_layers * LLAMA.layer_parameters() + LLAMA.embedding_parameters() == (
            LLAMA_PARAMETERS - LLAMA.hidden_size
        )


class TestPlanMemoryErrors:
    """非法 strategy / optimizer / compute_dtype 与缺参数必须被拒绝."""

    def test_invalid_strategy(self):
        with pytest.raises(PEFTConfigError, match="未知的策略"):
            plan_memory(LLAMA, strategy="bf16")

    def test_invalid_optimizer(self):
        with pytest.raises(PEFTConfigError, match="未知的优化器"):
            plan_memory(LLAMA, optimizer="lamb")

    def test_invalid_compute_dtype(self):
        with pytest.raises(PEFTConfigError, match="未知的计算精度"):
            plan_memory(LLAMA, compute_dtype="int8")

    def test_lora_requires_lora_config(self):
        with pytest.raises(PEFTConfigError, match="必须提供 lora_config"):
            plan_memory(LLAMA, strategy="lora")

    def test_qlora_requires_lora_config(self):
        with pytest.raises(PEFTConfigError, match="必须提供 lora_config"):
            plan_memory(LLAMA, strategy="qlora")

    def test_qlora_validates_its_quantization_config(self):
        with pytest.raises(PEFTConfigError, match="未知的 block_size"):
            plan_memory(
                LLAMA,
                strategy="qlora",
                lora_config=LoRAConfig(),
                qlora_config=QLoRAConfig(block_size=100),
            )


class TestPlanMemoryFull:
    """``full``：两个口径（bf16 直接训练 / fp32 主权重）相差 33%."""

    def test_bf16_full_uses_12_bytes_per_parameter(self):
        plan = plan_memory(LLAMA, strategy="full")
        assert plan.model == "llama-2-7b"
        assert plan.base_parameters == LLAMA_PARAMETERS
        assert plan.trainable_parameters == LLAMA_PARAMETERS
        assert plan.base_weight_bytes == LLAMA_PARAMETERS * 2
        assert plan.base_gradient_bytes == LLAMA_PARAMETERS * 2
        assert plan.base_optimizer_bytes == LLAMA_PARAMETERS * 8
        assert plan.adapter_weight_bytes == 0
        assert plan.adapter_gradient_bytes == 0
        assert plan.adapter_optimizer_bytes == 0
        assert plan.total_bytes == LLAMA_PARAMETERS * 12
        assert plan.as_gib == pytest.approx(LLAMA_PARAMETERS * 12 / GIB)
        assert plan.base_bits_per_parameter == pytest.approx(16.0)

    def test_fp32_master_weights_uses_16_bytes_per_parameter(self):
        plan = plan_memory(LLAMA, strategy="full", full_finetune_master_weights_fp32=True)
        assert plan.base_weight_bytes == LLAMA_PARAMETERS * 4
        assert plan.base_gradient_bytes == LLAMA_PARAMETERS * 4
        assert plan.total_bytes == LLAMA_PARAMETERS * 16
        assert plan.base_bits_per_parameter == pytest.approx(32.0)

    def test_8bit_adam_shrinks_optimizer_state(self):
        plan = plan_memory(LLAMA, strategy="full", optimizer="adamw_bnb_8bit")
        assert plan.base_optimizer_bytes == LLAMA_PARAMETERS * 2
        assert plan.total_bytes == LLAMA_PARAMETERS * 6

    def test_sgd_has_no_optimizer_state(self):
        plan = plan_memory(LLAMA, strategy="full", optimizer="sgd")
        assert plan.base_optimizer_bytes == 0
        assert plan.total_bytes == LLAMA_PARAMETERS * 4

    def test_notes_exclude_activations_and_explain_adamw(self):
        plan = plan_memory(LLAMA, strategy="full")
        joined = " ".join(plan.notes)
        assert "不含激活显存" in joined
        assert "AdamW 的一阶/二阶矩占 8 字节/参数" in joined


class TestPlanMemoryLora:
    """``lora``：基座 bf16 冻结，梯度与优化器状态只按适配器规模计费."""

    def test_lora_freezes_base_gradient_and_optimizer(self):
        plan = plan_memory(LLAMA, strategy="lora", lora_config=LoRAConfig())
        assert plan.base_weight_bytes == LLAMA_PARAMETERS * 2
        assert plan.base_gradient_bytes == 0
        assert plan.base_optimizer_bytes == 0
        assert plan.base_bits_per_parameter == pytest.approx(16.0)

    def test_lora_trainable_parameters_match_plan_lora(self):
        plan = plan_memory(LLAMA, strategy="lora", lora_config=LoRAConfig())
        assert plan.trainable_parameters == LLAMA_ADAPTER_PARAMETERS
        assert plan.adapter_weight_bytes == LLAMA_ADAPTER_PARAMETERS * 2
        assert plan.adapter_gradient_bytes == LLAMA_ADAPTER_PARAMETERS * 2
        assert plan.adapter_optimizer_bytes == LLAMA_ADAPTER_PARAMETERS * 8

    def test_lora_total_is_base_plus_adapter(self):
        plan = plan_memory(LLAMA, strategy="lora", lora_config=LoRAConfig())
        expected = LLAMA_PARAMETERS * 2 + LLAMA_ADAPTER_PARAMETERS * 12
        assert plan.total_bytes == expected
        assert plan.as_gib == pytest.approx(expected / GIB)

    def test_lora_fits_24gib_although_full_does_not(self):
        """这就是 24 GB 卡能跑 7B 微调的原因."""
        lora = plan_memory(LLAMA, strategy="lora", lora_config=LoRAConfig())
        full = plan_memory(LLAMA, strategy="full")
        assert lora.fits(24.0) is True
        assert full.fits(24.0) is False

    def test_lora_notes_mention_frozen_base_and_trainable_ratio(self):
        plan = plan_memory(LLAMA, strategy="lora", lora_config=LoRAConfig())
        joined = " ".join(plan.notes)
        assert "bfloat16 冻结存储" in joined
        assert "可训练参数 4,194,304" in joined


class TestPlanMemoryQLora:
    """``qlora``：只把基座权重从 2 字节压到 0.515869 字节，适配器仍是 16-bit."""

    def test_qlora_compresses_base_weight_only(self):
        plan = plan_memory(LLAMA, strategy="qlora", lora_config=LoRAConfig())
        assert plan.base_weight_bytes == 3_476_140_673
        assert plan.base_weight_bytes == int(round(LLAMA_PARAMETERS * 0.515869140625))
        assert plan.base_bits_per_parameter == pytest.approx(4.126953125)
        assert plan.adapter_weight_bytes == LLAMA_ADAPTER_PARAMETERS * 2
        assert plan.adapter_optimizer_bytes == LLAMA_ADAPTER_PARAMETERS * 8

    def test_qlora_total_matches_hand_arithmetic(self):
        plan = plan_memory(LLAMA, strategy="qlora", lora_config=LoRAConfig())
        expected = 3_476_140_673 + LLAMA_ADAPTER_PARAMETERS * 12
        assert plan.total_bytes == expected
        assert plan.as_gib == pytest.approx(expected / GIB)

    def test_qlora_notes_explain_quantization_and_16bit_adapter(self):
        plan = plan_memory(LLAMA, strategy="qlora", lora_config=LoRAConfig())
        joined = " ".join(plan.notes)
        assert "nf4" in joined and "block=64" in joined and "double_quant=True" in joined
        assert "适配器仍是 16-bit" in joined

    def test_single_quant_config_changes_base_weight(self):
        single = QLoRAConfig(bnb_4bit_use_double_quant=False)
        plan = plan_memory(
            LLAMA,
            strategy="qlora",
            lora_config=LoRAConfig(),
            qlora_config=single,
        )
        # 0.5625 = 9/16 字节/参数，仍是可手算的精确值
        assert plan.base_weight_bytes == LLAMA_PARAMETERS * 9 // 16
        assert plan.base_bits_per_parameter == pytest.approx(4.5)

    def test_qwen_qlora_is_under_one_gib(self):
        """0.6B 基座 4-bit 量化后不到 1 GiB：这是 day050 "单卡即可 SFT"的依据."""
        plan = plan_memory(QWEN, strategy="qlora", lora_config=LoRAConfig())
        assert plan.base_parameters == QWEN_PARAMETERS
        assert plan.fits(1.0) is True


class TestCompareStrategies:
    """``compare_strategies``：三种策略的对照（顺序即报告顺序）."""

    def test_returns_three_plans_in_order(self):
        plans = compare_strategies(LLAMA, lora_config=LoRAConfig())
        assert [plan.strategy for plan in plans] == list(STRATEGIES)

    def test_totals_are_ordered_full_gt_lora_gt_qlora(self):
        plans = compare_strategies(LLAMA, lora_config=LoRAConfig())
        totals = {plan.strategy: plan.total_bytes for plan in plans}
        assert totals["full"] > totals["lora"] > totals["qlora"]

    def test_optimizer_and_dtype_are_forwarded(self):
        plans = compare_strategies(
            LLAMA, lora_config=LoRAConfig(), optimizer="sgd", compute_dtype="float32"
        )
        assert all(plan.optimizer == "sgd" for plan in plans)
        assert all(plan.compute_dtype == "float32" for plan in plans)


class TestSavingsTable:
    """``savings_table``：相对全参微调的节省倍数（分母是同一份预算）."""

    def test_empty_input_rejected(self):
        with pytest.raises(PEFTConfigError, match="至少一份预算"):
            savings_table([])

    def test_full_baseline_is_one_x(self):
        rows = savings_table(compare_strategies(LLAMA, lora_config=LoRAConfig()))
        assert rows[0]["strategy"] == "full"
        assert rows[0]["savings_factor"] == 1.0

    def test_lora_savings_factor_matches_ratio(self):
        plans = compare_strategies(LLAMA, lora_config=LoRAConfig())
        rows = savings_table(plans)
        expected = round(plans[0].total_bytes / plans[1].total_bytes, 4)
        assert rows[1]["strategy"] == "lora"
        assert rows[1]["savings_factor"] == expected

    def test_rows_without_full_baseline_omit_savings_factor(self):
        lora_only = plan_memory(LLAMA, strategy="lora", lora_config=LoRAConfig())
        rows = savings_table([lora_only])
        assert rows[0]["strategy"] == "lora"
        assert "savings_factor" not in rows[0]

    def test_rows_carry_derived_values(self):
        plans = compare_strategies(LLAMA, lora_config=LoRAConfig())
        rows = savings_table(plans)
        for plan, row in zip(plans, rows):
            assert row["total_bytes"] == plan.total_bytes
            assert row["as_gib"] == round(plan.as_gib, 4)
            assert row["as_gb"] == round(plan.total_bytes / 1e9, 4)
            assert row["base_bits_per_parameter"] == round(plan.base_bits_per_parameter, 6)


class TestDeviceFitTable:
    """``device_fit_table``：每种策略在每个单卡预算下能否放下（不含激活）."""

    def test_default_budgets_follow_device_table_order(self):
        plans = compare_strategies(LLAMA, lora_config=LoRAConfig())
        rows = device_fit_table(plans)
        assert [row["device"] for row in rows] == list(DEVICE_BUDGETS)
        assert all(row["budget_gib"] == DEVICE_BUDGETS[row["device"]] for row in rows)
        assert all("不含激活显存" in row["note"] for row in rows)

    def test_24gib_card_fits_lora_and_qlora_only(self):
        plans = compare_strategies(LLAMA, lora_config=LoRAConfig())
        row = device_fit_table(plans, budgets={"RTX 4090 (24 GiB)": 24.0})[0]
        assert row["fits"] == {"full": False, "lora": True, "qlora": True}

    def test_custom_tight_budget_only_fits_qlora(self):
        plans = compare_strategies(LLAMA, lora_config=LoRAConfig())
        rows = device_fit_table(plans, budgets={"4 GiB": 4.0})
        assert rows[0]["fits"] == {"full": False, "lora": False, "qlora": True}

    def test_big_budget_fits_everything(self):
        plans = compare_strategies(LLAMA, lora_config=LoRAConfig())
        rows = device_fit_table(plans, budgets={"huge": 100.0})
        assert all(rows[0]["fits"].values())


class TestAdamwStateBytes:
    """``adamw_state_bytes``：公式里的 2（一阶矩 + 二阶矩）最容易漏."""

    def test_two_fp32_moments(self):
        assert adamw_state_bytes(1000) == 2 * 1000 * 4 == 8000

    def test_16_bit_states(self):
        assert adamw_state_bytes(1000, bits=16) == 2 * 1000 * 2 == 4000

    def test_rejects_non_positive_parameters(self):
        with pytest.raises(PEFTConfigError, match="parameters 必须为正整数"):
            adamw_state_bytes(0)

    @pytest.mark.parametrize("bits", [0, -8, 12])
    def test_rejects_bits_not_multiple_of_eight(self, bits):
        with pytest.raises(PEFTConfigError, match="bits 必须为 8 的正整数倍"):
            adamw_state_bytes(10, bits=bits)


class TestMinimalDevice:
    """``minimal_device``：把"13.48 GiB"翻译成"一张 24 GiB 的卡就够了"."""

    def test_smallest_card_wins(self):
        assert minimal_device(1.0) == "RTX 4090 (24 GiB)"

    def test_exact_budget_reuses_the_card(self):
        assert minimal_device(24.0) == "RTX 4090 (24 GiB)"

    def test_rounds_up_to_next_size(self):
        assert minimal_device(30.0) == "A100 (40 GiB)"
        assert minimal_device(48.0) == "RTX 6000 Ada (48 GiB)"

    def test_beyond_every_card_is_informational(self):
        """这是一条信息性查询：没有型号满足时返回文字，而不是抛异常."""
        assert minimal_device(1000.0) == "需要多卡或更小的模型"

    def test_rejects_non_positive_budget(self):
        with pytest.raises(PEFTConfigError, match="预算必须为正数"):
            minimal_device(0)


class TestParameterScale:
    """``parameter_scale``：参数量 → 十亿（B）显示值."""

    def test_billions(self):
        assert parameter_scale(LLAMA_PARAMETERS) == pytest.approx(6.738415616)
        assert parameter_scale(QWEN_PARAMETERS) == pytest.approx(0.59604992)

    def test_rejects_non_positive(self):
        with pytest.raises(PEFTConfigError, match="参数量必须为正整数"):
            parameter_scale(0)


class TestLogisticEstimateHours:
    """``logistic_estimate_hours``：一行的线性外推，不做非线性假设."""

    def test_single_device(self):
        assert logistic_estimate_hours(
            total_steps=100, seconds_per_step=1.0
        ) == pytest.approx(100 / 3600)

    def test_devices_divide_the_total(self):
        assert logistic_estimate_hours(
            total_steps=100, seconds_per_step=2.0, devices=4
        ) == pytest.approx(100 * 2 / 4 / 3600)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"total_steps": 0, "seconds_per_step": 1.0},
            {"total_steps": 1, "seconds_per_step": 0.0},
            {"total_steps": 1, "seconds_per_step": 1.0, "devices": 0},
        ],
    )
    def test_rejects_non_positive_inputs(self, kwargs):
        with pytest.raises(PEFTConfigError, match="必须为正数"):
            logistic_estimate_hours(**kwargs)


class TestQuantizationMemoryNote:
    """``quantization_memory_note``：把"4-bit 并不等于 0.5 字节"写成一句话."""

    def test_default_mentions_double_quant_savings(self):
        note = quantization_memory_note(QLoRAConfig())
        assert "码点 4 bit" in note
        assert "常数 0.126953 bit/参数" in note
        assert "合计 4.126953 bit = 0.515869 字节/参数" in note
        assert "二级量化比单重量化省 0.046631 字节/参数" in note

    def test_single_quant_mentions_fp32_constants(self):
        note = quantization_memory_note(QLoRAConfig(bnb_4bit_use_double_quant=False))
        assert "常数 0.500000 bit/参数" in note
        assert "合计 4.500000 bit = 0.562500 字节/参数" in note
        assert "未启用二级量化：常数仍是 fp32" in note


class TestBytesPerParameterOf16Bit:
    """``bytes_per_parameter_of_16bit``：QLoRA 的对照基线."""

    def test_default_is_bfloat16(self):
        assert bytes_per_parameter_of_16bit() == 2
        assert bytes_per_parameter_of_16bit("float16") == 2
        assert bytes_per_parameter_of_16bit("float32") == 4

    def test_rejects_unknown_dtype(self):
        with pytest.raises(PEFTConfigError, match="未知的精度"):
            bytes_per_parameter_of_16bit("int8")
