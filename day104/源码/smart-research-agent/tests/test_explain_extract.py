"""``explainability`` 的读数与提取测试：类型校验 + 两条读法（day083）."""

from __future__ import annotations

import math

import pytest

from smart_research_agent.arch_variants.types import (
    VARIANTS,
    VARIANT_DECODER_ONLY,
    VARIANT_ENCODER_DECODER,
    VARIANT_ENCODER_ONLY,
)
from smart_research_agent.explainability import extract, types
from smart_research_agent.explainability.errors import (
    FAMILY_OUTCOMES,
    AssemblyError,
    ExplainError,
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.explainability.types import (
    LEVELS,
    STREAM_CROSS,
    STREAM_SELF,
    AttentionRecord,
    Attribution,
    HeadProfile,
)
from tests import explain_samples as samples


class TestErrors:
    """失败族（本课是**四族**：``GradientError`` 缺席，理由写在模块说明里）."""

    def test_family_table_is_closed(self) -> None:
        assert set(FAMILY_OUTCOMES) == {
            "ShapeError",
            "ParameterError",
            "AssemblyError",
            "NumericError",
        }

    def test_every_family_says_who_should_fix_it(self) -> None:
        for name, outcome in FAMILY_OUTCOMES.items():
            assert outcome.startswith("改调用") or outcome.startswith("改数据")
            assert name in ("ShapeError", "ParameterError", "AssemblyError", "NumericError")

    def test_the_five_error_classes_are_siblings_of_the_other_packages(self) -> None:
        from smart_research_agent.arch_variants.errors import ShapeError as VariantShapeError
        from smart_research_agent.transformer_core.errors import ShapeError as CoreShapeError

        assert issubclass(ShapeError, ExplainError)
        assert issubclass(ShapeError, CoreShapeError)
        assert not issubclass(ShapeError, VariantShapeError)

    def test_assembly_error_is_a_parameter_error(self) -> None:
        assert issubclass(AssemblyError, ParameterError)


class TestRowReadings:
    """逐行读数的**手算样本**（每一个数都能被纸笔复核）."""

    def test_uniform_entropy_is_ln_n(self) -> None:
        assert types.row_entropy((0.25, 0.25, 0.25, 0.25)) == pytest.approx(math.log(4))

    def test_one_hot_entropy_is_zero(self) -> None:
        assert types.row_entropy((1.0, 0.0, 0.0)) == 0.0

    def test_entropy_ignores_zero_terms(self) -> None:
        """``0·ln 0`` 取 0（而不是 nan）——这是熵的定义里的那一条约定."""
        assert types.row_entropy((1.0, 0.0)) == 0.0

    def test_entropy_rejects_empty_row(self) -> None:
        with pytest.raises(ParameterError):
            types.row_entropy(())

    def test_entropy_rejects_negative(self) -> None:
        with pytest.raises(NumericError):
            types.row_entropy((1.2, -0.2))

    def test_entropy_rejects_non_finite(self) -> None:
        with pytest.raises(NumericError):
            types.row_entropy((1.0, float("nan")))

    def test_ceiling_of_one_position_is_zero(self) -> None:
        assert types.entropy_ceiling(1) == 0.0

    def test_ceiling_rejects_zero(self) -> None:
        with pytest.raises(ParameterError):
            types.entropy_ceiling(0)

    def test_allowed_counts_of_a_causal_mask(self) -> None:
        assert types.allowed_counts(types.causal_mask(4)) == (1, 2, 3, 4)

    def test_allowed_counts_rejects_an_empty_row(self) -> None:
        with pytest.raises(NumericError):
            types.allowed_counts(((True, False), (False, False)))

    def test_allowed_counts_rejects_empty(self) -> None:
        with pytest.raises(ParameterError):
            types.allowed_counts(())

    def test_allowed_counts_rejects_ragged(self) -> None:
        with pytest.raises(ShapeError):
            types.allowed_counts(((True, True), (True,)))

    def test_mean_of_rejects_empty(self) -> None:
        with pytest.raises(ParameterError):
            types.mean_of(())

    def test_mean_uses_fsum(self) -> None:
        assert types.mean_of((1.0, 2.0, 3.0)) == 2.0

    def test_forwarded_masks_match_the_core_ones(self) -> None:
        from smart_research_agent.math_foundations.attention import causal_mask
        from smart_research_agent.transformer_core.layers import full_mask

        assert types.causal_mask(3) == causal_mask(3)
        assert types.full_mask(3) == full_mask(3)

    def test_mask_constants_are_ten_levels(self) -> None:
        assert len(LEVELS) == 10
        assert types.LEVEL_COUNT == 10


class TestAttentionRecord:
    """记录的校验与读数（**每一行是一条分布**这一条转发 core 的实现）."""

    def test_uniform_record_readings_are_hand_computable(self) -> None:
        record = samples.uniform_record(4)
        assert record.mean_entropy == pytest.approx(math.log(4))
        assert record.mean_ceiling == pytest.approx(math.log(4))
        assert record.normalized_entropy == pytest.approx(1.0)
        assert record.rows == 4 and record.columns == 4
        assert record.masked_zeroes() == 0
        assert record.allowed_counts == (4, 4, 4, 4)

    def test_one_hot_record_has_zero_entropy(self) -> None:
        record = samples.one_hot_record(3)
        assert record.mean_entropy == 0.0
        assert record.normalized_entropy == 0.0
        assert record.entropies == (0.0, 0.0, 0.0)

    def test_causal_uniform_record_sits_at_the_ceiling(self) -> None:
        """因果均匀表的熵**正好等于**天花板：``(1/n)Σln(i+1)``."""
        record = samples.causal_uniform_record(4)
        expected = sum(math.log(index + 1) for index in range(4)) / 4
        assert record.mean_entropy == pytest.approx(expected)
        assert record.mean_ceiling == pytest.approx(expected)
        assert record.normalized_entropy == pytest.approx(1.0)

    def test_cross_record_is_rectangular(self) -> None:
        record = samples.cross_record(3, 5)
        assert (record.rows, record.columns) == (3, 5)
        assert record.stream == STREAM_CROSS
        assert record.mean_entropy == pytest.approx(math.log(5))

    def test_square_cross_record_is_allowed(self) -> None:
        """``n_tgt == n_src`` 时形状与自注意力完全一样——区分它们的是 ``stream``."""
        record = samples.cross_record(4, 4)
        assert record.rows == record.columns == 4
        assert record.stream == STREAM_CROSS

    def test_unknown_stream_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            AttentionRecord(label="x", weights=((1.0,),), mask=((True,),), stream="sideways")

    def test_empty_label_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            AttentionRecord(label="", weights=((1.0,),), mask=((True,),))

    def test_self_record_must_be_square(self) -> None:
        with pytest.raises(AssemblyError):
            AttentionRecord(
                label="x",
                weights=((0.5, 0.5, 0.0),),
                mask=((True, True, True),),
                stream=STREAM_SELF,
            )

    def test_mask_shape_must_match(self) -> None:
        with pytest.raises(ShapeError):
            AttentionRecord(label="x", weights=((1.0,),), mask=((True, True),))

    def test_mask_row_without_any_allowed_position_is_rejected(self) -> None:
        with pytest.raises(NumericError):
            AttentionRecord(
                label="x",
                weights=((1.0, 0.0), (0.0, 1.0)),
                mask=((True, False), (False, False)),
            )

    def test_rows_must_be_distributions(self) -> None:
        """行和不为 1 时抛的是 **core** 的那一族（转发而不是重写）."""
        from smart_research_agent.transformer_core.errors import NumericError as CoreNumeric

        with pytest.raises(CoreNumeric):
            AttentionRecord(
                label="x",
                weights=((0.5, 0.4), (0.5, 0.5)),
                mask=((True, True), (True, True)),
            )

    def test_negative_weight_is_rejected(self) -> None:
        from smart_research_agent.transformer_core.errors import NumericError as CoreNumeric

        with pytest.raises(CoreNumeric):
            AttentionRecord(
                label="x",
                weights=((1.2, -0.2), (0.5, 0.5)),
                mask=((True, True), (True, True)),
            )

    def test_token_labels_must_match_the_rows(self) -> None:
        with pytest.raises(ShapeError):
            AttentionRecord(
                label="x",
                weights=((1.0,),),
                mask=((True,),),
                tokens=("a", "b"),
            )

    def test_source_labels_must_match_the_columns(self) -> None:
        with pytest.raises(ShapeError):
            AttentionRecord(
                label="x",
                weights=((0.2, 0.2, 0.2, 0.2, 0.2),),
                mask=((True, True, True, True, True),),
                stream=STREAM_CROSS,
                source_tokens=("a",),
            )

    def test_weight_at_checks_bounds(self) -> None:
        record = samples.uniform_record(3)
        assert record.weight_at(0, 0) == pytest.approx(1 / 3)
        with pytest.raises(ParameterError):
            record.weight_at(3, 0)
        with pytest.raises(ParameterError):
            record.weight_at(0, 9)
        with pytest.raises(ParameterError):
            record.weight_at(True, 0)

    def test_masked_zeroes_counts_the_blocked_cells(self) -> None:
        record = samples.causal_uniform_record(4)
        assert record.masked_zeroes() == 6

    def test_to_dict_and_summary(self) -> None:
        record = samples.uniform_record(3)
        payload = record.to_dict()
        assert payload["rows"] == 3 and payload["masked_zeroes"] == 0
        assert "归一化" in record.summary_line()

    def test_labels_are_coerced_to_strings(self) -> None:
        record = AttentionRecord(
            label="x",
            weights=((1.0,),),
            mask=((True,),),
            tokens=(1,),
            source_tokens=(2,),
        )
        assert record.tokens == ("1",) and record.source_tokens == ("2",)


class TestHeadProfile:
    """profile 的校验与读数."""

    def _profile(self, **overrides: object) -> HeadProfile:
        payload: dict[str, object] = {
            "label": "层 0 · 头 0",
            "entropy": 1.0,
            "ceiling": 1.4,
            "peak_weight": 0.5,
            "peak_index": 0,
            "support": 4,
            "frobenius": 1.5,
            "diagonal_mass": 0.25,
            "rows": 2,
            "columns": 2,
        }
        payload.update(overrides)
        return HeadProfile(**payload)  # type: ignore[arg-type]

    def test_normalized_entropy_and_sparsity(self) -> None:
        profile = self._profile()
        assert profile.normalized_entropy == pytest.approx(1.0 / 1.4)
        assert profile.sparsity == pytest.approx(0.0)

    def test_ceiling_zero_gives_zero(self) -> None:
        assert self._profile(entropy=0.0, ceiling=0.0).normalized_entropy == 0.0

    def test_entropy_above_ceiling_is_rejected(self) -> None:
        with pytest.raises(NumericError):
            self._profile(entropy=1.5, ceiling=1.4)

    def test_negative_entropy_is_rejected(self) -> None:
        with pytest.raises(NumericError):
            self._profile(entropy=-0.1)

    def test_non_finite_is_rejected(self) -> None:
        with pytest.raises(NumericError):
            self._profile(frobenius=float("inf"))

    def test_zero_support_is_rejected(self) -> None:
        with pytest.raises(NumericError):
            self._profile(support=0)

    def test_bad_rows_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            self._profile(rows=0)

    def test_bad_columns_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            self._profile(columns=0)

    def test_to_dict_and_summary(self) -> None:
        profile = self._profile()
        assert profile.to_dict()["columns"] == 2
        assert "归一化" in profile.summary_line()


class TestAttribution:
    """列质量（"谁在被看"）."""

    def test_uniform_attribution(self) -> None:
        from smart_research_agent.explainability.analyze import attribution_of

        record = samples.uniform_record(4)
        attribution = attribution_of(record, top=2)
        assert attribution.mass == pytest.approx((0.25, 0.25, 0.25, 0.25))
        assert attribution.top_positions == (0, 1)
        assert attribution.mass_at(2) == pytest.approx(0.25)

    def test_attribution_of_a_rectangular_table(self) -> None:
        from smart_research_agent.explainability.analyze import attribution_of

        record = samples.cross_record(3, 5)
        attribution = attribution_of(record, top=1)
        assert attribution.columns == 5
        assert sum(attribution.mass) == pytest.approx(1.0)

    def test_ticks_break_ties_by_index(self) -> None:
        from smart_research_agent.explainability.analyze import attribution_of

        record = samples.uniform_record(3)
        attribution = attribution_of(record, top=3)
        assert attribution.top_positions == (0, 1, 2)
    def test_top_must_be_positive_and_fit(self) -> None:
        from smart_research_agent.explainability.analyze import attribution_of

        record = samples.uniform_record(3)
        with pytest.raises(ParameterError):
            attribution_of(record, top=0)
        with pytest.raises(ParameterError):
            attribution_of(record, top=4)

    def test_mass_must_be_normalized(self) -> None:
        with pytest.raises(NumericError):
            Attribution(label="x", mass=(0.5, 0.4), top_positions=(0,), columns=2)

    def test_mass_length_must_match(self) -> None:
        with pytest.raises(ShapeError):
            Attribution(label="x", mass=(0.5, 0.5), top_positions=(0,), columns=3)

    def test_top_positions_must_be_in_range(self) -> None:
        with pytest.raises(ParameterError):
            Attribution(label="x", mass=(1.0,), top_positions=(3,), columns=1)

    def test_mass_at_checks_bounds(self) -> None:
        attribution = Attribution(label="x", mass=(1.0,), top_positions=(0,), columns=1)
        assert attribution.mass_at(0) == 1.0
        with pytest.raises(ParameterError):
            attribution.mass_at(1)
        with pytest.raises(ParameterError):
            attribution.mass_at(True)
        assert "列质量" in attribution.summary_line()
        assert attribution.to_dict()["columns"] == 1


class TestCoverageEdges:
    """几个只在"边界输入"上才走到的分支（**每一个都对应一条真实的用法**）."""

    def test_ceiling_zero_gives_zero_normalized_entropy(self) -> None:
        """``n = 1`` 的因果表：第 0 行只能看到自己（天花板 ``ln 1 = 0``）."""
        record = AttentionRecord(label="单点", weights=((1.0,),), mask=((True,),))
        assert record.mean_ceiling == 0.0
        assert record.normalized_entropy == 0.0

    def test_weight_at_rejects_a_bool_column(self) -> None:
        record = samples.uniform_record(3)
        with pytest.raises(ParameterError):
            record.weight_at(0, True)

    def test_profile_rows_must_be_an_integer(self) -> None:
        with pytest.raises(ParameterError):
            HeadProfile(
                label="x",
                entropy=0.0,
                ceiling=0.0,
                peak_weight=1.0,
                peak_index=0,
                support=1,
                frobenius=1.0,
                diagonal_mass=1.0,
                rows=1.5,  # type: ignore[arg-type]
                columns=1,
            )

    def test_profile_thresholds_are_validated(self) -> None:
        from smart_research_agent.explainability.analyze import profile_of

        record = samples.uniform_record(3)
        with pytest.raises(ParameterError):
            profile_of(record, threshold="0.1")  # type: ignore[arg-type]
        with pytest.raises(ParameterError):
            profile_of(record, threshold=-0.1)
        with pytest.raises(ParameterError):
            profile_of(record, threshold=float("nan"))


class TestSelfRecords:
    """模型**真实**权重（每一层一条；``encoder_decoder`` 三条流各一份）."""

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_records_are_read_from_the_forward(self, variant: str) -> None:
        records = samples.sample_self_records(variant)
        assert len(records) == (9 if variant == VARIANT_ENCODER_DECODER else 3)
        assert all(record.stream in (STREAM_SELF, STREAM_CROSS) for record in records)

    def test_the_records_are_the_forward_weights_bit_for_bit(self) -> None:
        """**这条最关键**：读出来的东西与模型实际算的东西逐位相同."""
        from smart_research_agent.arch_variants.stacks import variant_forward

        params = samples.sample_params(VARIANT_DECODER_ONLY)
        inputs = samples.sample_inputs()
        forward = variant_forward(params, inputs)
        records = extract.self_records(params, inputs)
        assert tuple(record.weights for record in records) == forward.block_weights

    def test_encoder_decoder_has_three_streams(self) -> None:
        records = samples.sample_self_records(VARIANT_ENCODER_DECODER)
        labels = [record.label for record in records]
        assert sum(label.startswith("编码器") for label in labels) == 3
        assert sum(label.startswith("解码器") for label in labels) == 3
        assert sum(label.startswith("交叉") for label in labels) == 3

    def test_cross_records_are_rectangular(self) -> None:
        records = extract.cross_records(
            samples.sample_params(VARIANT_ENCODER_DECODER),
            samples.sample_inputs(),
            samples.sample_source(),
        )
        assert len(records) == 3
        assert all((record.rows, record.columns) == (4, 5) for record in records)
        assert all(record.stream == STREAM_CROSS for record in records)

    def test_cross_records_need_two_streams(self) -> None:
        with pytest.raises(AssemblyError):
            extract.cross_records(
                samples.sample_params(VARIANT_ENCODER_ONLY),
                samples.sample_inputs(),
                samples.sample_source(),
            )

    def test_self_stream_records_filter_out_cross(self) -> None:
        records = samples.sample_self_records(VARIANT_ENCODER_DECODER)
        filtered = extract.self_stream_records(
            samples.sample_params(VARIANT_ENCODER_DECODER),
            samples.sample_inputs(),
            source=samples.sample_source(),
        )
        assert len(filtered) == len(records) - 3

    def test_custom_token_labels(self) -> None:
        records = extract.self_records(
            samples.sample_params(VARIANT_ENCODER_ONLY),
            samples.sample_inputs(),
            tokens=("甲", "乙", "丙", "丁"),
        )
        assert records[0].tokens == ("甲", "乙", "丙", "丁")

    def test_wrong_label_count_is_rejected(self) -> None:
        with pytest.raises(ShapeError):
            extract.self_records(
                samples.sample_params(VARIANT_ENCODER_ONLY),
                samples.sample_inputs(),
                tokens=("甲",),
            )

    def test_params_type_is_checked(self) -> None:
        with pytest.raises(ParameterError):
            extract.self_records("params", samples.sample_inputs())

    def test_records_are_deterministic(self) -> None:
        first = samples.sample_self_records()
        second = samples.sample_self_records()
        assert first[0].weights == second[0].weights


class TestHeadRecords:
    """多头读法（**同一组投影**在多头划分下）."""

    def test_two_heads_give_two_records(self) -> None:
        records = samples.sample_head_records(heads=2)
        assert [record.label for record in records] == ["层 0 · 头 0", "层 0 · 头 1"]
        assert all(record.rows == 4 for record in records)

    def test_heads_one_matches_the_model_bit_for_bit(self) -> None:
        """**跨口径的那条接缝**：``heads=1`` 必须逐位等于模型真实权重."""
        params = samples.sample_params(VARIANT_DECODER_ONLY)
        assert extract.single_head_matches(params, samples.sample_inputs())
        assert extract.single_head_matches(params, samples.sample_inputs(), layer=2)

    def test_single_head_matches_is_defined_for_single_streams_only(self) -> None:
        with pytest.raises(AssemblyError):
            extract.single_head_matches(
                samples.sample_params(VARIANT_ENCODER_DECODER),
                samples.sample_inputs(),
                source=samples.sample_source(),
            )

    def test_single_head_matches_checks_the_layer(self) -> None:
        with pytest.raises(ParameterError):
            extract.single_head_matches(
                samples.sample_params(VARIANT_DECODER_ONLY), samples.sample_inputs(), layer=9
            )

    def test_heads_must_divide_the_hidden_dimension(self) -> None:
        with pytest.raises(ParameterError):
            samples.sample_head_records(heads=4)

    def test_bad_heads_are_rejected(self) -> None:
        params = samples.sample_params(VARIANT_DECODER_ONLY)
        inputs = samples.sample_inputs()
        for heads in (0, -1, True, 1.5):
            with pytest.raises(ParameterError):
                extract.head_records(params, inputs, heads=heads)  # type: ignore[arg-type]

    def test_layer_must_be_in_range(self) -> None:
        params = samples.sample_params(VARIANT_DECODER_ONLY)
        with pytest.raises(ParameterError):
            extract.head_records(params, samples.sample_inputs(), layer=3)
        with pytest.raises(ParameterError):
            extract.head_records(params, samples.sample_inputs(), layer=True)  # type: ignore[arg-type]

    def test_encoder_decoder_needs_an_explicit_source(self) -> None:
        """不给源序列时**不接受默认样本**：那会量出一段与调用方无关的表."""
        with pytest.raises(AssemblyError):
            extract.head_records(
                samples.sample_params(VARIANT_ENCODER_DECODER), samples.sample_inputs()
            )

    def test_source_shape_is_checked(self) -> None:
        with pytest.raises(ShapeError):
            extract.head_records(
                samples.sample_params(VARIANT_ENCODER_DECODER),
                samples.sample_inputs(),
                source=samples.sample_source()[:2],
            )

    def test_encoder_decoder_uses_the_source_side(self) -> None:
        records = samples.sample_head_records(VARIANT_ENCODER_DECODER, heads=2)
        assert records[0].rows == 5  # 编码器吃的是源序列（5 个位置）

    def test_post_placement_skips_the_norm(self) -> None:
        """post 摆放时第一个子层吃的是**原始输入**；而这条路径在第 0 层之外才真正走到
        （前几层要先重放过去，因此它同时覆盖了 `_stream_input_at_layer` 的 post 分支）."""
        params = samples.sample_params(VARIANT_DECODER_ONLY)
        inputs = samples.sample_inputs()
        pre = extract.head_records(params, inputs, heads=2, placement="pre", layer=2)
        post = extract.head_records(params, inputs, heads=2, placement="post", layer=2)
        assert pre[0].weights != post[0].weights

    def test_bad_placement_is_rejected(self) -> None:
        from smart_research_agent.encoder_decoder.errors import ParameterError as EdError

        with pytest.raises(EdError):
            extract.head_records(
                samples.sample_params(VARIANT_DECODER_ONLY),
                samples.sample_inputs(),
                placement="middle",
            )

    def test_notes_explain_the_two_readings(self) -> None:
        record = samples.sample_head_records(heads=2)[0]
        joined = " ".join(record.notes)
        assert "不是" in joined and "缩放" in joined


class TestAggregation:
    """逐格平均与按层分组（滚动的前置步骤）."""

    def test_average_keeps_rows_distributions(self) -> None:
        records = samples.sample_head_records(heads=2)
        merged = extract.aggregate_heads(records)
        for row in merged.weights:
            assert sum(row) == pytest.approx(1.0, abs=1e-12)

    def test_average_of_one_record_is_itself(self) -> None:
        record = samples.uniform_record(3)
        assert extract.aggregate_heads((record,)).weights == record.weights

    def test_empty_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            extract.aggregate_heads(())

    def test_shape_mismatch_is_rejected(self) -> None:
        with pytest.raises(ShapeError):
            extract.aggregate_heads((samples.uniform_record(3), samples.uniform_record(4)))

    def test_mask_mismatch_is_rejected(self) -> None:
        with pytest.raises(AssemblyError):
            extract.aggregate_heads(
                (samples.uniform_record(4), samples.causal_uniform_record(4))
            )

    def test_record_of_and_records_of_layer(self) -> None:
        records = samples.sample_layer_records(heads=2)
        assert extract.record_of(records, "层 1 · 头 1").label == "层 1 · 头 1"
        assert len(extract.records_of_layer(records, 1)) == 2
        with pytest.raises(ParameterError):
            extract.record_of(records, "层 9 · 头 9")
        with pytest.raises(ParameterError):
            extract.records_of_layer(records, 9)
        with pytest.raises(ParameterError):
            extract.records_of_layer(records, True)  # type: ignore[arg-type]

    def test_layer_inputs_returns_one_matrix_per_layer(self) -> None:
        layers = extract.layer_inputs(
            samples.sample_params(VARIANT_DECODER_ONLY), samples.sample_inputs()
        )
        assert len(layers) == 3
        assert len(layers[0]) == 4

    def test_layer_inputs_need_a_source_for_two_streams(self) -> None:
        with pytest.raises(AssemblyError):
            extract.layer_inputs(
                samples.sample_params(VARIANT_ENCODER_DECODER), samples.sample_inputs()
            )

    def test_layer_count(self) -> None:
        assert extract.layer_count(samples.sample_layer_records(heads=2)) == 3

    def test_validate_heads(self) -> None:
        assert extract.validate_heads(6, 2) == 2
        with pytest.raises(ParameterError):
            extract.validate_heads(6, 4)
        with pytest.raises(ParameterError):
            extract.validate_heads(0, 1)

    def test_variant_of(self) -> None:
        assert extract.variant_of(samples.sample_params(VARIANT_DECODER_ONLY)) == (
            VARIANT_DECODER_ONLY
        )

    def test_summary_lines(self) -> None:
        assert len(extract.summary_lines(samples.sample_self_records())) == 3
