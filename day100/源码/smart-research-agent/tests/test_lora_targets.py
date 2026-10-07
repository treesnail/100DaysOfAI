"""LoRA 架构算术测试（M5-D3）：参数量、占比、告警与规格自检.

本文件把 day049 预先算出的四个"单矩阵比例"与两个"模型总参数量"变成
可执行的断言：

- ``4096/8 -> 0.00390625``、``4096/16 -> 0.0078125``、``2048/8 -> 0.0078125``、
  ``11008/16 -> 0.0029069767441860465``；
- ``llama-2-7b -> 6 738 415 616``、``qwen3-0.6b -> 596 049 920``——
  规格里写错任何一个字段，这两个总数立刻对不上。

``plan_lora`` 的四类告警（秩超过最小维度、占比超过 1%、只适配 q/v 之一、
带偏置却放开了偏置训练）在下面各自有"触发"与"不触发"两条用例守着。
"""

from __future__ import annotations

import pytest

from smart_research_agent.peft.config import LoRAConfig, PEFTConfigError
from smart_research_agent.peft.targets import (
    MODEL_SPECS,
    DecoderSpec,
    LoRAPlan,
    ModuleShape,
    adapter_footprint,
    describe_model,
    parameter_check,
    plan_lora,
    rank_comparison,
    ratio_curve,
    reference_matrix_ratios,
    single_matrix_ratio,
    square_ratio_shortcut,
    target_preset_table,
    theoretical_adapter_parameter_cap,
)

#: 公开配置里两个模型规格的键名
LLAMA = "llama-2-7b"
QWEN = "qwen3-0.6b"


def small_spec(*, qk_norm: bool = True, linear_bias: bool = True) -> DecoderSpec:
    """一个"小到可以手算"的规格，用来隔离 ``plan_lora`` 的各类告警.

    ``hidden=64 / intermediate=128 / layers=2 / heads=4 / kv_heads=2 /
    head_dim=16``：attention 宽度 64、kv 宽度 32，七类投影的形状都能口算。
    """
    return DecoderSpec(
        name="small",
        hidden_size=64,
        intermediate_size=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        vocab_size=256,
        tie_word_embeddings=True,
        qk_norm=qk_norm,
        linear_bias=linear_bias,
    )


class TestModuleShape:
    """``ModuleShape``：参数量、适配器参数量、单层比例与方阵简化式."""

    def test_parameters(self):
        assert ModuleShape("q_proj", 4096, 4096).parameters == 16_777_216

    def test_adapter_parameters(self):
        """新增参数 = ``r × (in + out)``，把平方级降成线性级."""
        assert ModuleShape("q_proj", 4096, 4096).adapter_parameters(8) == 65_536

    def test_ratio_matches_day049_number(self):
        """4096 方阵 + ``r=8`` 的单层比例 = ``2r/d`` = 0.390625%."""
        assert ModuleShape("q_proj", 4096, 4096).ratio(8) == pytest.approx(0.00390625)

    def test_square_shortcut_returns_value_for_square(self):
        shape = ModuleShape("q_proj", 4096, 4096)
        assert shape.square_shortcut_ratio(8) == pytest.approx(2 * 8 / 4096)

    def test_square_shortcut_is_none_for_rectangular(self):
        """非方阵刻意不给近似值：``2r/d`` 会明显低估 MLP 投影的比例."""
        assert ModuleShape("gate_proj", 4096, 11008).square_shortcut_ratio(8) is None

    def test_non_positive_rank_rejected(self):
        with pytest.raises(PEFTConfigError, match="r 必须为正整数"):
            ModuleShape("q_proj", 4096, 4096).adapter_parameters(0)

    def test_non_positive_shape_rejected(self):
        with pytest.raises(PEFTConfigError, match="形状必须为正整数"):
            ModuleShape("bad", 0, 4)


class TestDecoderSpec:
    """``DecoderSpec``：投影宽度、逐层参数量、导数派生量与 ``to_dict``."""

    def test_attention_and_kv_width(self):
        spec = MODEL_SPECS[QWEN]
        assert spec.attention_width == 16 * 128
        assert spec.kv_width == 8 * 128

    def test_linear_modules_names_and_order(self):
        names = [shape.name for shape in MODEL_SPECS[LLAMA].linear_modules()]
        assert names == [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    def test_linear_module_shapes(self):
        shapes = {shape.name: shape for shape in MODEL_SPECS[LLAMA].linear_modules()}
        assert shapes["q_proj"] == ModuleShape("q_proj", 4096, 4096)
        assert shapes["down_proj"] == ModuleShape("down_proj", 11008, 4096)

    def test_layer_parameters_includes_two_norms(self):
        """单层 = 七类线性投影 + 2 × hidden（LayerNorm）."""
        assert MODEL_SPECS[LLAMA].layer_parameters() == 202_383_360

    def test_layer_parameters_adds_qk_norm(self):
        delta = (
            small_spec(qk_norm=True).layer_parameters()
            - small_spec(qk_norm=False).layer_parameters()
        )
        assert delta == 2 * 16  # 2 × head_dim

    def test_layer_parameters_adds_bias_when_present(self):
        delta = (
            small_spec(linear_bias=True).layer_parameters()
            - small_spec(linear_bias=False).layer_parameters()
        )
        expected = sum(shape.out_features for shape in small_spec().linear_modules())
        assert delta == expected

    def test_embedding_parameters_respects_tie(self):
        assert MODEL_SPECS[LLAMA].embedding_parameters() == 2 * 32000 * 4096
        assert MODEL_SPECS[QWEN].embedding_parameters() == 151936 * 1024

    def test_linear_parameters_excludes_embeddings(self):
        spec = MODEL_SPECS[LLAMA]
        assert spec.linear_parameters() == 6_476_005_376
        assert spec.linear_parameters() < spec.total_parameters()

    def test_to_dict_contains_derived_values(self):
        payload = MODEL_SPECS[QWEN].to_dict()
        for key in (
            "attention_width",
            "kv_width",
            "layer_parameters",
            "embedding_parameters",
            "linear_parameters",
            "total_parameters",
            "linear_modules",
        ):
            assert key in payload
        expected = [shape.name for shape in MODEL_SPECS[QWEN].linear_modules()]
        assert payload["linear_modules"] == expected
        assert payload["total_parameters"] == 596_049_920

    def test_num_heads_must_be_divisible_by_kv_heads(self):
        """GQA 的分组要求：num_heads 不能被 num_kv_heads 整除时直接拒绝."""
        with pytest.raises(PEFTConfigError, match="必须能被"):
            DecoderSpec(
                name="bad",
                hidden_size=64,
                intermediate_size=128,
                num_layers=2,
                num_heads=5,
                num_kv_heads=2,
                head_dim=16,
                vocab_size=256,
                tie_word_embeddings=True,
            )


class TestModelSpecs:
    """``MODEL_SPECS``：总参数量必须与公开数字逐位一致（最重要的一道自检）."""

    def test_llama_2_7b_total_parameters(self):
        assert MODEL_SPECS[LLAMA].total_parameters() == 6_738_415_616

    def test_qwen3_0_6b_total_parameters(self):
        assert MODEL_SPECS[QWEN].total_parameters() == 596_049_920

    def test_specs_are_keyed_by_their_name(self):
        for key, spec in MODEL_SPECS.items():
            assert key == spec.name


class TestRatioArithmetic:
    """单矩阵比例：精确式、简化式，以及 day049 的四个数字."""

    def test_square_shortcut_equals_single_matrix_ratio(self):
        """简化式与精确式在方阵上必须等价（一条断言守住公式）."""
        exact = single_matrix_ratio(4096, 4096, 8)
        assert square_ratio_shortcut(4096, 8) == pytest.approx(exact)

    def test_rectangular_exact_ratio_beats_shortcut(self):
        """4096×11008 上精确值约 0.536%，而 ``2r/11008`` 只有 0.2907%."""
        exact = single_matrix_ratio(4096, 11008, 16)
        shortcut = square_ratio_shortcut(11008, 16)
        assert exact == pytest.approx(16 * (4096 + 11008) / (4096 * 11008))
        assert exact > shortcut

    def test_reference_matrix_ratios_values(self):
        rows = reference_matrix_ratios()
        assert [row["ratio"] for row in rows] == pytest.approx(
            [0.00390625, 0.0078125, 0.0078125, 0.0029069767441860465]
        )

    def test_reference_matrix_ratio_cases(self):
        rows = reference_matrix_ratios()
        assert [(row["dimension"], row["r"]) for row in rows] == [
            (4096.0, 8.0),
            (4096.0, 16.0),
            (2048.0, 8.0),
            (11008.0, 16.0),
        ]

    def test_ratio_percent_is_hundred_times_ratio(self):
        for row in reference_matrix_ratios():
            assert row["ratio_percent"] == pytest.approx(row["ratio"] * 100)

    def test_single_matrix_ratio_rejects_non_positive(self):
        with pytest.raises(PEFTConfigError, match="必须为正整数"):
            single_matrix_ratio(0, 4096, 8)

    def test_square_shortcut_rejects_non_positive(self):
        with pytest.raises(PEFTConfigError, match="必须为正整数"):
            square_ratio_shortcut(0, 8)


class TestPlanLora:
    """``plan_lora``：模型级参数量、占比、每模块比例，以及四类告警."""

    def test_adapter_parameters_and_ratio(self):
        plan = plan_lora(MODEL_SPECS[LLAMA], LoRAConfig(target_modules="all_linear"))
        assert isinstance(plan, LoRAPlan)
        assert plan.adapter_parameters == 19_988_480
        assert plan.adapter_parameters_per_layer == 624_640
        assert plan.trainable_parameters == plan.adapter_parameters
        assert plan.base_parameters == 6_738_415_616
        denominator = plan.base_parameters + plan.trainable_parameters
        assert plan.trainable_ratio == pytest.approx(plan.trainable_parameters / denominator)

    def test_attention_preset_matches_day049_number(self):
        plan = plan_lora(MODEL_SPECS[LLAMA], LoRAConfig(target_modules="attention"))
        assert plan.adapter_parameters == 4_194_304
        assert plan.per_module_ratio["q_proj"] == pytest.approx(0.00390625)
        assert plan.per_module_ratio["v_proj"] == pytest.approx(0.00390625)

    def test_per_module_ratio_keys_match_matched_modules(self):
        plan = plan_lora(MODEL_SPECS[LLAMA], LoRAConfig(target_modules="mlp"))
        assert set(plan.per_module_ratio) == set(plan.matched_modules_per_layer)
        assert plan.matched_modules_per_layer == ("gate_proj", "up_proj", "down_proj")

    def test_rank_upper_bound(self):
        plan = plan_lora(MODEL_SPECS[LLAMA], LoRAConfig(r=8, target_modules="attention"))
        assert plan.rank_upper_bound == 8

    def test_frozen_parameters_equal_base(self):
        plan = plan_lora(MODEL_SPECS[LLAMA], LoRAConfig(target_modules="attention"))
        assert plan.frozen_parameters == plan.base_parameters

    def test_to_dict_is_json_ready(self):
        import json

        plan = plan_lora(MODEL_SPECS[LLAMA], LoRAConfig(target_modules="attention"))
        payload = plan.to_dict()
        assert isinstance(payload["targets"], list)
        assert isinstance(payload["matched_modules_per_layer"], list)
        assert json.loads(json.dumps(payload, ensure_ascii=False))["model"] == LLAMA

    def test_summary_line_mentions_model_and_rank(self):
        plan = plan_lora(MODEL_SPECS[LLAMA], LoRAConfig(target_modules="attention"))
        assert LLAMA in plan.summary_line()
        assert "r=8" in plan.summary_line()

    def test_unmatched_target_modules_rejected(self):
        """目标预设没有命中任何模块：配置看着正常，实际一个适配器都没插上."""
        with pytest.raises(PEFTConfigError, match="没有匹配"):
            plan_lora(MODEL_SPECS[LLAMA], LoRAConfig(target_modules=("weight",)))

    def test_healthy_config_produces_no_warnings(self):
        plan = plan_lora(small_spec(), LoRAConfig(r=1, target_modules="attention"))
        assert plan.warnings == []

    def test_rank_above_min_dimension_warns(self):
        """``r`` 超过被适配层的最小维度：秩上界由维度决定，加大 r 只花显存."""
        plan = plan_lora(small_spec(), LoRAConfig(r=128, target_modules="attention"))
        assert any("超过被适配层的最小维度" in message for message in plan.warnings)

    def test_trainable_ratio_above_one_percent_warns(self):
        plan = plan_lora(small_spec(), LoRAConfig(r=8, target_modules="attention"))
        assert plan.trainable_ratio > 0.01
        assert any("超过 1%" in message for message in plan.warnings)

    def test_single_sided_attention_warns(self):
        """只适配 q/v 之一：不是原论文消融实验比较的那件事."""
        plan = plan_lora(small_spec(), LoRAConfig(r=2, target_modules=("q_proj",)))
        assert any("只适配了 q_proj / v_proj 中的一个" in message for message in plan.warnings)

    def test_two_sided_attention_does_not_warn(self):
        plan = plan_lora(small_spec(), LoRAConfig(r=1, target_modules=("q_proj", "v_proj")))
        assert not any("只适配了" in message for message in plan.warnings)

    def test_linear_bias_with_bias_enabled_warns(self):
        plan = plan_lora(
            small_spec(), LoRAConfig(r=1, target_modules="attention", bias="lora_only")
        )
        assert any("线性层带偏置" in message for message in plan.warnings)

    def test_linear_bias_with_bias_none_does_not_warn(self):
        plan = plan_lora(
            small_spec(), LoRAConfig(r=1, target_modules="attention", bias="none")
        )
        assert not any("带偏置" in message for message in plan.warnings)

    def test_bias_all_counts_every_linear_bias(self):
        """``bias=all`` 会把未被适配层的偏置也算进可训练参数."""
        plan = plan_lora(
            small_spec(), LoRAConfig(r=1, target_modules="attention", bias="all")
        )
        assert plan.trainable_parameters > plan.adapter_parameters


class TestRankComparison:
    """``rank_comparison``：参数量对 ``r`` 严格线性，容量收益却递减."""

    def test_adapter_parameters_are_linear_in_rank(self):
        rows = rank_comparison(MODEL_SPECS[LLAMA], ranks=(1, 2, 4, 8))
        per_rank = {row["r"]: row["adapter_parameters"] for row in rows}
        assert per_rank[2] == 2 * per_rank[1]
        assert per_rank[8] == 8 * per_rank[1]

    def test_parameters_per_layer_are_linear_in_rank(self):
        rows = rank_comparison(MODEL_SPECS[LLAMA], ranks=(1, 2, 4, 8))
        per_rank = {row["r"]: row["parameters_per_layer"] for row in rows}
        assert per_rank[4] == 4 * per_rank[1]
        assert per_rank[8] == 2 * per_rank[4]

    def test_alpha_doubles_with_rank_keeping_scaling(self):
        """``lora_alpha = 2r`` 让缩放恒为 2：加大秩不改变有效步长."""
        rows = rank_comparison(MODEL_SPECS[LLAMA], ranks=(1, 2, 4))
        assert all(row["scaling"] == pytest.approx(2.0) for row in rows)

    def test_default_ranks_are_powers_of_two(self):
        rows = rank_comparison(MODEL_SPECS[LLAMA])
        assert [row["r"] for row in rows] == [1, 2, 4, 8, 16, 32, 64, 128, 256]


class TestTargetPresetTable:
    """``target_preset_table``：四种预设在同一模型、同一 ``r`` 上的对照."""

    def test_four_presets_in_order(self):
        rows = target_preset_table(MODEL_SPECS[LLAMA], r=8)
        assert [row["preset"] for row in rows] == [
            "attention",
            "attention_all",
            "mlp",
            "all_linear",
        ]

    def test_adapter_parameters_per_preset(self):
        rows = {row["preset"]: row for row in target_preset_table(MODEL_SPECS[LLAMA], r=8)}
        assert rows["attention"]["adapter_parameters"] == 4_194_304
        assert rows["attention_all"]["adapter_parameters"] == 8_388_608
        assert rows["mlp"]["adapter_parameters"] == 11_599_872
        assert rows["all_linear"]["adapter_parameters"] == 19_988_480

    def test_all_linear_is_about_five_times_attention(self):
        """同一份数据、同一个 ``r``，只因目标模块不同就差 4.8 倍."""
        rows = {row["preset"]: row for row in target_preset_table(MODEL_SPECS[LLAMA], r=8)}
        ratio = rows["all_linear"]["adapter_parameters"] / rows["attention"]["adapter_parameters"]
        assert ratio == pytest.approx(19_988_480 / 4_194_304)
        assert ratio > 4.7

    def test_ratio_percent_matches_ratio(self):
        for row in target_preset_table(MODEL_SPECS[QWEN], r=4):
            expected = row["trainable_ratio"] * 100
            assert row["trainable_ratio_percent"] == pytest.approx(expected)


class TestAdapterFootprint:
    """``adapter_footprint``：单个适配器的落盘体积（A + B 两个矩阵）."""

    def test_values(self):
        footprint = adapter_footprint(
            r=8, in_features=4096, out_features=4096, dtype_bytes=2
        )
        assert footprint["parameters"] == 65_536
        assert footprint["bytes"] == 131_072
        assert footprint["mebibytes"] == pytest.approx(0.125)
        assert footprint["equivalent_full_layer_parameters"] == 16_777_216
        assert footprint["compression_ratio"] == pytest.approx(256.0)

    def test_non_positive_arguments_rejected(self):
        with pytest.raises(PEFTConfigError, match="必须为正整数"):
            adapter_footprint(r=0, in_features=4096, out_features=4096)


class TestModelHelpers:
    """``parameter_check`` / ``describe_model`` / 秩上限 / ``ratio_curve``."""

    def test_parameter_check(self):
        total, billions = parameter_check(MODEL_SPECS[LLAMA])
        assert total == 6_738_415_616
        assert billions == pytest.approx(6.738415616)

    def test_describe_model_mentions_name_and_size(self):
        text = describe_model(MODEL_SPECS[QWEN])
        assert "qwen3-0.6b" in text
        assert "596,049,920" in text

    def test_theoretical_cap_exceeds_base_parameters(self):
        """全秩 LoRA 的新增参数量超过基座本身：``r`` 不能乱加的量化边界."""
        cap = theoretical_adapter_parameter_cap(MODEL_SPECS[LLAMA])
        assert cap == 10_234_101_760
        assert cap > MODEL_SPECS[LLAMA].total_parameters()

    def test_ratio_curve_ranks_are_powers_of_two(self):
        curve = ratio_curve(MODEL_SPECS[LLAMA], r_max=128)
        assert [r for r, _ in curve] == [1, 2, 4, 8, 16, 32, 64, 128]

    def test_ratio_curve_is_strictly_increasing(self):
        ratios = [ratio for _, ratio in ratio_curve(MODEL_SPECS[LLAMA], r_max=128)]
        assert all(later > earlier for earlier, later in zip(ratios, ratios[1:]))

    def test_ratio_curve_rejects_non_positive_r_max(self):
        with pytest.raises(PEFTConfigError, match="r_max 必须为正整数"):
            ratio_curve(MODEL_SPECS[LLAMA], r_max=0)


class TestSpecValidationGaps:
    """规格校验里剩下的一条：``hidden_size`` / ``head_dim`` 必须为正."""

    def test_rejects_non_positive_head_dim(self):
        base = MODEL_SPECS[LLAMA]
        with pytest.raises(PEFTConfigError, match="hidden_size / head_dim 必须为正整数"):
            DecoderSpec(
                name="broken",
                hidden_size=base.hidden_size,
                intermediate_size=base.intermediate_size,
                num_layers=base.num_layers,
                num_heads=base.num_heads,
                num_kv_heads=base.num_kv_heads,
                head_dim=0,
                vocab_size=base.vocab_size,
                tie_word_embeddings=base.tie_word_embeddings,
            )
