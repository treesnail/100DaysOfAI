"""``explainability.rollout`` 的测试：把逐层权重乘起来（day083）."""

from __future__ import annotations

import math

import pytest

from smart_research_agent.explainability import rollout
from smart_research_agent.explainability.errors import (
    AssemblyError,
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.explainability.types import AttentionRecord
from tests import explain_samples as samples


class TestAggregateByLayer:
    """按层分组、组内逐格平均（滚动的前置步骤）."""

    def test_groups_by_layer_and_sorts(self) -> None:
        layers = rollout.aggregate_by_layer(samples.sample_layer_records(heads=2))
        assert [record.label for record in layers] == [
            "层 0 · 平均 2 头",
            "层 1 · 平均 2 头",
            "层 2 · 平均 2 头",
        ]

    def test_cross_stream_records_are_rejected(self) -> None:
        """长方形的交叉权重与方阵的自注意力**不能相乘**."""
        with pytest.raises(AssemblyError):
            rollout.aggregate_by_layer(samples.sample_self_records("encoder_decoder"))

    def test_a_cross_stream_record_is_rejected_even_when_it_is_square(self) -> None:
        """等长的交叉表形状与自注意力一样，因此它只能靠 ``stream`` 被认出来."""
        square_cross = samples.cross_record(3, 3, label="层 0 · 头 0")
        with pytest.raises(AssemblyError):
            rollout.aggregate_by_layer((square_cross,))

    def test_layer_numbers_must_be_contiguous(self) -> None:
        record = samples.uniform_record(3)
        second = AttentionRecord(
            label="层 5 · 头 0", weights=record.weights, mask=record.mask
        )
        with pytest.raises(AssemblyError):
            rollout.aggregate_by_layer((record, second))

    def test_bad_label_is_rejected(self) -> None:
        record = samples.uniform_record(3)
        odd = AttentionRecord(label="第零层", weights=record.weights, mask=record.mask)
        with pytest.raises(AssemblyError):
            rollout.aggregate_by_layer((odd,))

    def test_mask_mismatch_is_rejected(self) -> None:
        """掩码不同的两层不能相乘：滚动的三个性质都要求同一张掩码."""
        full = samples.uniform_record(4)
        causal = samples.causal_uniform_record(4)
        first = AttentionRecord(label="层 0 · 头 0", weights=full.weights, mask=full.mask)
        second = AttentionRecord(
            label="层 1 · 头 0", weights=causal.weights, mask=causal.mask
        )
        with pytest.raises(AssemblyError):
            rollout.aggregate_by_layer((first, second))

    def test_shape_mismatch_is_rejected(self) -> None:
        small = samples.uniform_record(3)
        big = samples.uniform_record(4)
        first = AttentionRecord(label="层 0 · 头 0", weights=small.weights, mask=small.mask)
        second = AttentionRecord(label="层 1 · 头 0", weights=big.weights, mask=big.mask)
        with pytest.raises(ShapeError):
            rollout.aggregate_by_layer((first, second))

    def test_empty_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            rollout.aggregate_by_layer(())


class TestRolloutWeights:
    """``Â = α·A + (1−α)·I`` 与 ``R = Â_L ⋯ Â_1``."""

    def test_alpha_zero_gives_the_identity(self) -> None:
        """一条把公式说清的判据：α = 0 时信息一跳都没走."""
        assert rollout.identity_when_alpha_zero(samples.sample_layer_records(heads=2))

    def test_uniform_layers_stay_uniform_when_alpha_is_one(self) -> None:
        """**手算**：``(1/n)J`` 是幂等的（``J·J = n·J``），因此乘积还是它自己."""
        layers = tuple(
            AttentionRecord(
                label=f"层 {index} · 头 0",
                weights=samples.uniform_record(3).weights,
                mask=samples.uniform_record(3).mask,
            )
            for index in range(2)
        )
        weights = rollout.rollout_weights(layers, alpha=1.0)
        expected = samples.uniform_record(3).weights
        assert all(
            abs(a - b) < 1e-12
            for left, right in zip(weights, expected, strict=True)
            for a, b in zip(left, right, strict=True)
        )

    def test_rows_stay_stochastic(self) -> None:
        weights = rollout.rollout_weights(samples.sample_layer_records(heads=2), alpha=0.5)
        for row in weights:
            assert sum(row) == pytest.approx(1.0, abs=1e-12)

    def test_causal_zeroes_are_exact(self) -> None:
        """下三角 × 下三角 = 下三角，而且那些 0 是**精确的**."""
        records = samples.sample_layer_records("decoder_only", heads=2)
        assert rollout.preserves_causal_zeroes(records, alpha=0.5)

    def test_full_masks_have_no_upper_zeroes_to_preserve(self) -> None:
        """全开掩码的表在上三角**没有** 0 可保（因此这条判据在这类表上为假）."""
        layers = tuple(
            AttentionRecord(
                label=f"层 {index} · 头 0",
                weights=samples.uniform_record(3).weights,
                mask=samples.uniform_record(3).mask,
            )
            for index in range(2)
        )
        assert not rollout.preserves_causal_zeroes(layers)

    @pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
    def test_alpha_range(self, alpha: float) -> None:
        assert rollout.rollout_weights(samples.sample_layer_records(heads=2), alpha=alpha)

    def test_alpha_must_be_a_number_in_range(self) -> None:
        records = samples.sample_layer_records(heads=2)
        for alpha in (-0.1, 1.5, "0.5", True):
            with pytest.raises(ParameterError):
                rollout.rollout_weights(records, alpha=alpha)  # type: ignore[arg-type]

    def test_alpha_must_be_finite(self) -> None:
        with pytest.raises(NumericError):
            rollout.rollout_weights(
                samples.sample_layer_records(heads=2), alpha=float("inf")
            )


class TestRolloutRecord:
    """滚动的结果作为一条记录（标签里写明 α 与层数）."""

    def test_label_mentions_alpha_and_layers(self) -> None:
        record = rollout.rollout_record(samples.sample_layer_records(heads=2), alpha=0.25)
        assert "0.25" in record.label and "3 层" in record.label
        assert record.stream == "self"

    def test_mask_is_inherited_from_the_first_layer(self) -> None:
        records = samples.sample_layer_records("decoder_only", heads=2)
        rolled = rollout.rollout_record(records, alpha=0.5)
        layers = rollout.aggregate_by_layer(records)
        assert rolled.mask == layers[0].mask

    def test_notes_explain_the_residual(self) -> None:
        rolled = rollout.rollout_record(samples.sample_layer_records(heads=2))
        assert any("残差" in note for note in rolled.notes)


class TestRolloutFocus:
    """滚动的集中度（三个数一起给，**可能为负**）."""

    def test_three_numbers_are_returned(self) -> None:
        last, rolled, gain = rollout.rollout_focus(samples.sample_layer_records(heads=2))
        assert gain == pytest.approx(last - rolled)

    def test_alpha_zero_leaves_everything_at_the_diagonal(self) -> None:
        """α = 0 时滚动就是单位阵：归一化熵为 0（**全部质量在对角线**）."""
        _last, rolled, gain = rollout.rollout_focus(
            samples.sample_layer_records(heads=2), alpha=0.0
        )
        assert rolled == 0.0
        assert gain > 0.0

    def test_uniform_records_can_gain_focus(self) -> None:
        layers = tuple(
            AttentionRecord(
                label=f"层 {index} · 头 0",
                weights=samples.uniform_record(4).weights,
                mask=samples.uniform_record(4).mask,
            )
            for index in range(3)
        )
        _last, _rolled, gain = rollout.rollout_focus(layers, alpha=0.5)
        assert math.isfinite(gain)


class TestRowSums:
    """逐行和（滚动之后它们应当仍然是 1）."""

    def test_a_uniform_record_passes(self) -> None:
        sums = rollout.row_sums(samples.uniform_record(3).weights)
        assert sums == pytest.approx((1.0, 1.0, 1.0))

    def test_a_broken_table_is_rejected(self) -> None:
        with pytest.raises(NumericError):
            rollout.row_sums(((0.5, 0.4),), tolerance=1e-12)

    def test_tolerance_is_validated(self) -> None:
        with pytest.raises(ParameterError):
            rollout.row_sums(((1.0,),), tolerance=-1.0)
        with pytest.raises(ParameterError):
            rollout.row_sums(((1.0,),), tolerance="1e-12")  # type: ignore[arg-type]

    def test_loose_tolerance_accepts_a_small_drift(self) -> None:
        assert rollout.row_sums(((1.0 - 1e-9, 1e-9),), tolerance=1e-6)
