"""``explainability.verify`` 的测试：六条性质 + 它们的反证（day083）."""

from __future__ import annotations

import pytest

from smart_research_agent.arch_variants.types import (
    VARIANTS,
    VARIANT_DECODER_ONLY,
    VARIANT_ENCODER_DECODER,
)
from smart_research_agent.explainability import verify
from smart_research_agent.explainability.errors import (
    ParameterError,
    ShapeError,
)
from smart_research_agent.explainability.types import (
    EXPLAIN_PROPERTIES,
    PROPERTY_DETERMINISTIC,
    PROPERTY_ENTROPY_WITHIN_CEILING,
    PROPERTY_HEATMAP_ROUND_TRIP,
    PROPERTY_MASKED_ENTRIES_ARE_EXACT_ZERO,
    PROPERTY_ROLLOUT_IS_STOCHASTIC,
    PROPERTY_ROWS_ARE_DISTRIBUTIONS,
    AttentionRecord,
)
from tests import explain_samples as samples


def _full_mask_layers(count: int = 2, size: int = 3):
    """两层**全开掩码**的记录（上三角没有 0 可保的那一类）."""
    base = samples.uniform_record(size)
    return tuple(
        AttentionRecord(
            label=f"层 {index} · 头 0", weights=base.weights, mask=base.mask
        )
        for index in range(count)
    )


class TestRecords:
    """读数记录与报告的校验."""

    def test_outcome_of_each_state(self) -> None:
        passing = verify.PropertyOutcome(name=PROPERTY_ROWS_ARE_DISTRIBUTIONS, passed=True, evidence="x")
        failing = verify.PropertyOutcome(
            name=PROPERTY_ROWS_ARE_DISTRIBUTIONS, passed=False, evidence="x"
        )
        assert passing.state == "通过"
        assert failing.state == "未通过"
        assert passing.summary_line().startswith("[通过]")
        assert passing.to_dict()["passed"] is True

    def test_outcome_describes_itself(self) -> None:
        outcome = verify.PropertyOutcome(
            name=PROPERTY_ENTROPY_WITHIN_CEILING, passed=True, evidence="x"
        )
        assert "位置数" in outcome.description

    def test_unknown_property_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            verify.PropertyOutcome(name="nope", passed=True, evidence="x")

    def test_report_needs_every_property(self) -> None:
        outcome = verify.PropertyOutcome(
            name=PROPERTY_ROWS_ARE_DISTRIBUTIONS, passed=True, evidence="x"
        )
        with pytest.raises(ShapeError):
            verify.PropertyReport(subject="x", outcomes=(outcome,))

    def test_report_needs_a_subject(self) -> None:
        outcomes = tuple(
            verify.PropertyOutcome(name=name, passed=True, evidence="x")
            for name in EXPLAIN_PROPERTIES
        )
        with pytest.raises(ParameterError):
            verify.PropertyReport(subject="", outcomes=outcomes)

    def test_empty_report_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            verify.PropertyReport(subject="x", outcomes=())

    def test_report_readings(self) -> None:
        report = verify.check_properties(samples.sample_layer_records(heads=2))
        assert report.ok
        assert len(report.summary_lines()) == 6
        assert report.to_dict()["ok"] is True
        assert report.outcome_of(PROPERTY_HEATMAP_ROUND_TRIP).passed

    def test_report_rejects_an_unknown_query(self) -> None:
        report = verify.check_properties(samples.sample_layer_records(heads=2))
        with pytest.raises(ParameterError):
            report.outcome_of("nope")

    def test_report_subject_counts_records(self) -> None:
        report = verify.check_properties(samples.sample_layer_records(heads=2))
        assert report.subject == "records=6 heads=6"

    def test_notes_explain_the_discipline(self) -> None:
        notes = " ".join(verify.check_properties(samples.sample_layer_records(heads=2)).notes)
        assert "天花板" in notes and "同一次前向" in notes


class TestIndividualChecks:
    """六条判据各自的读数与**它们的反证**."""

    def test_rows_are_distributions_passes_and_cannot_be_broken(self) -> None:
        """这条判据**不可能**在合法记录上失败：构造期就挡住了行和不为 1 的表.

        （它不是"没有分辨力"，而是"它的分辨力在构造期"——
        下面那条断言把这句话钉住：`AttentionRecord` 直接拒绝坏行和。）
        """
        outcome = verify.check_rows_are_distributions(samples.sample_layer_records(heads=2))
        assert outcome.passed
        assert "行和误差" in outcome.evidence

        from smart_research_agent.transformer_core.errors import NumericError as CoreNumeric

        with pytest.raises(CoreNumeric):
            AttentionRecord(
                label="x",
                weights=((0.5, 0.4), (0.5, 0.5)),
                mask=((True, True), (True, True)),
            )

    def test_masked_zeroes_check_counts_the_blocked_cells(self) -> None:
        outcome = verify.check_masked_entries_are_exact_zero(
            samples.sample_layer_records("decoder_only", heads=2)
        )
        assert outcome.passed
        assert "掩码外共 36 个格子" in outcome.evidence

    def test_masked_zeroes_check_turns_red(self) -> None:
        """**反证**：把一个被挡住的格子填上 1e-3，这条立刻亮红（逐位判据）."""
        broken = AttentionRecord(
            label="层 0 · 头 0",
            weights=((0.999, 0.001), (0.5, 0.5)),
            mask=((True, False), (True, True)),
        )
        outcome = verify.check_masked_entries_are_exact_zero((broken,))
        assert not outcome.passed
        assert "不是逐位为 0" in outcome.evidence

    def test_entropy_check_is_tight_for_a_causal_uniform_record(self) -> None:
        outcome = verify.check_entropy_within_ceiling((samples.causal_uniform_record(4),))
        assert outcome.passed
        assert "4 行" in outcome.evidence

    def test_entropy_check_turns_red(self) -> None:
        """**反证**：掩码只允许一个位置、而权重却散在两个位置上 ⇒ 熵超过天花板."""
        broken = AttentionRecord(
            label="层 0 · 头 0",
            weights=((0.5, 0.5), (0.5, 0.5)),
            mask=((True, False), (True, False)),
        )
        outcome = verify.check_entropy_within_ceiling((broken,))
        assert not outcome.passed

    def test_heatmap_round_trip_is_exact(self) -> None:
        outcome = verify.check_heatmap_round_trip(samples.sample_layer_records(heads=2))
        assert outcome.passed
        assert "逐级相同" in outcome.evidence

    def test_heatmap_round_trip_is_guarded_by_the_parser(self) -> None:
        """这条判据的"会亮红"那一面落在**解析侧**：不认识的字符当场拒绝."""
        from smart_research_agent.explainability.render import parse_heatmap

        with pytest.raises(ParameterError):
            parse_heatmap("0 |?#?| 峰值\n")

    def test_rollout_check_on_causal_records(self) -> None:
        outcome = verify.check_rollout_is_stochastic(
            samples.sample_layer_records("decoder_only", heads=2)
        )
        assert outcome.passed
        assert "上三角全 0：True" in outcome.evidence
        assert "掩码外 6 个格子" in outcome.evidence

    def test_rollout_check_turns_red_when_a_blocked_cell_is_dirty(self) -> None:
        """**反证**：某一层的被挡格子不是 0 ⇒ 滚动之后它仍然不是 0（而且被放大）."""
        broken = AttentionRecord(
            label="层 0 · 头 0",
            weights=((0.999, 0.001), (0.5, 0.5)),
            mask=((True, False), (True, True)),
        )
        second = AttentionRecord(
            label="层 1 · 头 0", weights=broken.weights, mask=broken.mask
        )
        outcome = verify.check_rollout_is_stochastic((broken, second))
        assert not outcome.passed

    def test_rollout_check_on_full_masks_has_nothing_blocked(self) -> None:
        """全开掩码的表**没有**被挡住的格子 ⇒ 这一条上的读数是"0 个格子".

        这不是 bug，而是口径：这条判据问的是"挡住的仍然挡住"，
        而全开掩码什么都没挡——**那个 0 要如实印出来**（否则"通过"会显得比实际更强）。
        """
        outcome = verify.check_rollout_is_stochastic(_full_mask_layers())
        assert outcome.passed
        assert "掩码外 0 个格子" in outcome.evidence
        assert "上三角全 0：False" in outcome.evidence

    def test_rollout_check_needs_at_least_two_layers(self) -> None:
        """同一个层的两个头只是**同一个因子**：它们不构成"两跳"."""
        with pytest.raises(ParameterError):
            verify.check_rollout_is_stochastic(samples.sample_layer_records(heads=2)[:2])

    def test_deterministic_passes(self) -> None:
        outcome = verify.check_deterministic(samples.sample_layer_records(heads=2))
        assert outcome.passed
        assert "逐字符相同：True" in outcome.evidence


class TestCheckProperties:
    """六条一起跑（三个变体各一次）."""

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_all_six_pass(self, variant: str) -> None:
        records = samples.sample_layer_records(variant, heads=2)
        report = verify.check_properties(records)
        assert report.ok
        assert {outcome.name for outcome in report.outcomes} == set(EXPLAIN_PROPERTIES)

    def test_alpha_flows_into_the_rollout_check(self) -> None:
        records = samples.sample_layer_records(heads=2)
        report = verify.check_properties(records, alpha=0.2)
        assert "α=0.2" in report.outcome_of(PROPERTY_ROLLOUT_IS_STOCHASTIC).evidence

    def test_empty_records_are_rejected(self) -> None:
        with pytest.raises(ParameterError):
            verify.check_properties(())

    def test_non_record_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            verify.check_properties(("record",))  # type: ignore[arg-type]

    def test_cross_only_records_are_rejected(self) -> None:
        """只有交叉那一路时，滚动这条判据没有自变量（它的权重是长方形的）."""
        records = samples.sample_self_records("encoder_decoder")
        cross_only = tuple(record for record in records if record.stream == "cross")
        assert len(cross_only) == 3
        with pytest.raises(ParameterError):
            verify.check_properties(cross_only)

    def test_check_all_is_an_alias(self) -> None:
        records = samples.sample_layer_records(heads=2)
        assert verify.check_all(records).ok == verify.check_properties(records).ok

    def test_encoder_decoder_records_include_both_streams(self) -> None:
        records = samples.sample_layer_records(VARIANT_ENCODER_DECODER, heads=2)
        assert all(record.stream == "self" for record in records)
        assert len(records) == 6


class TestTolerances:
    """两个容差常量（它们定义了"多小的漂移算 0"）."""

    def test_rollout_tolerance_is_tight(self) -> None:
        assert 0.0 < verify.ROLLOUT_TOLERANCE <= 1e-9

    def test_entropy_tolerance_is_tight(self) -> None:
        assert 0.0 < verify.ENTROPY_TOLERANCE <= 1e-9

    def test_causal_uniform_record_is_within_the_entropy_tolerance(self) -> None:
        """因果均匀表的熵**正好等于**天花板，而浮点求和会有 1e-16 的漂移."""
        record = samples.causal_uniform_record(4)
        gap = max(
            record.entropies[index] - record.ceiling[index] for index in range(record.rows)
        )
        assert abs(gap) <= verify.ENTROPY_TOLERANCE

    def test_variant_constant_used_by_the_samples(self) -> None:
        assert samples.sample_shape().layers == 3
        assert VARIANT_DECODER_ONLY != VARIANT_ENCODER_DECODER
