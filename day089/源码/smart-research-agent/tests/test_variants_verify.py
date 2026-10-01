"""``arch_variants.verify`` 的测试：六条性质 + 整条变体的梯度校验（day082）."""

from __future__ import annotations

import pytest

from smart_research_agent.arch_variants import masks, probe, stacks, verify
from smart_research_agent.arch_variants.errors import (
    AssemblyError,
    GradientError,
    NumericError,
    ParameterError,
)
from smart_research_agent.arch_variants.types import (
    ARCH_PROPERTIES,
    VARIANTS,
    VARIANT_DECODER_ONLY,
    VARIANT_ENCODER_DECODER,
    VARIANT_ENCODER_ONLY,
    PROPERTY_CROSS_SPANS_ALL_SOURCES,
    PROPERTY_DECODER_READS_ONLY_PAST,
    PROPERTY_ENCODER_READS_EVERY_POSITION,
    PROPERTY_MASK_ZEROES_ARE_EXACT,
)
from tests import arch_variants_samples as samples


def _source_kwargs(variant: str) -> dict[str, object]:
    if variant == VARIANT_ENCODER_DECODER:
        return {"source": samples.sample_source()}
    return {}


class TestPropertyRecords:
    """性质读数的三态（通过 / 未通过 / 不适用）."""

    def test_unknown_name_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            verify.PropertyOutcome(name="nope", passed=True, evidence="")

    def test_not_applicable_cannot_also_pass(self) -> None:
        """**这条纪律**:“不适用”与“通过”必须分开（否则删掉检查与它通过长得一样）."""
        with pytest.raises(NumericError):
            verify.PropertyOutcome(
                name=PROPERTY_MASK_ZEROES_ARE_EXACT,
                passed=True,
                evidence="",
                applicable=False,
            )

    def test_state_of_each_kind(self) -> None:
        passing = verify.PropertyOutcome(
            name=PROPERTY_MASK_ZEROES_ARE_EXACT, passed=True, evidence=""
        )
        failing = verify.PropertyOutcome(
            name=PROPERTY_MASK_ZEROES_ARE_EXACT, passed=False, evidence=""
        )
        skipped = verify.PropertyOutcome(
            name=PROPERTY_MASK_ZEROES_ARE_EXACT,
            passed=False,
            evidence="",
            applicable=False,
        )
        assert passing.state == "通过"
        assert failing.state == "未通过"
        assert skipped.state == "不适用"

    def test_description_and_summary(self) -> None:
        outcome = verify.PropertyOutcome(
            name=PROPERTY_MASK_ZEROES_ARE_EXACT, passed=True, evidence="0 个"
        )
        assert "掩码" in outcome.description
        assert outcome.summary_line().startswith("[通过]")
        assert outcome.to_dict()["state"] == "通过"


class TestPropertyReport:
    """六条性质的报告：只对适用的那些下结论."""

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_report_is_ok(self, variant: str) -> None:
        params = samples.sample_parameters(variant)
        report = verify.check_properties(
            params, samples.sample_inputs(), **_source_kwargs(variant)
        )
        assert report.ok
        assert {item.name for item in report.outcomes} == set(ARCH_PROPERTIES)

    def test_encoder_only_skips_two(self) -> None:
        params = samples.sample_parameters(VARIANT_ENCODER_ONLY)
        report = verify.check_properties(params, samples.sample_inputs())
        assert set(report.skipped) == {
            PROPERTY_DECODER_READS_ONLY_PAST,
            PROPERTY_CROSS_SPANS_ALL_SOURCES,
        }
        assert all(item.passed for item in report.applicable_outcomes)

    def test_decoder_only_skips_two(self) -> None:
        params = samples.sample_parameters(VARIANT_DECODER_ONLY)
        report = verify.check_properties(params, samples.sample_inputs())
        assert set(report.skipped) == {
            PROPERTY_ENCODER_READS_EVERY_POSITION,
            PROPERTY_CROSS_SPANS_ALL_SOURCES,
        }

    def test_encoder_decoder_skips_nothing(self) -> None:
        params = samples.sample_parameters(VARIANT_ENCODER_DECODER)
        report = verify.check_properties(
            params, samples.sample_inputs(), source=samples.sample_source()
        )
        assert report.skipped == ()

    def test_empty_report_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            verify.PropertyReport(variant=VARIANT_ENCODER_ONLY, outcomes=())

    def test_incomplete_report_is_rejected(self) -> None:
        """缺一条与“它不适用”在报告里必须长得不一样."""
        outcome = verify.PropertyOutcome(
            name=PROPERTY_MASK_ZEROES_ARE_EXACT, passed=True, evidence=""
        )
        with pytest.raises(NumericError):
            verify.PropertyReport(variant=VARIANT_ENCODER_ONLY, outcomes=(outcome,))

    def test_outcome_lookup(self) -> None:
        params = samples.sample_parameters(VARIANT_DECODER_ONLY)
        report = verify.check_properties(params, samples.sample_inputs())
        assert report.outcome_of(PROPERTY_DECODER_READS_ONLY_PAST).passed
        with pytest.raises(ParameterError):
            report.outcome_of("nope")

    def test_to_dict_and_summary_lines(self) -> None:
        params = samples.sample_parameters(VARIANT_ENCODER_ONLY)
        report = verify.check_properties(params, samples.sample_inputs())
        payload = report.to_dict()
        assert payload["ok"] is True
        assert len(report.summary_lines()) == 6
        assert any("实测" in line or "权" in line for line in report.notes)

    def test_params_type_is_checked(self) -> None:
        with pytest.raises(ParameterError):
            verify.check_properties("params", samples.sample_inputs())  # type: ignore[arg-type]

    def test_source_is_rejected_for_single_stream(self) -> None:
        params = samples.sample_parameters(VARIANT_ENCODER_ONLY)
        with pytest.raises(AssemblyError):
            verify.check_properties(
                params, samples.sample_inputs(), source=samples.sample_source()
            )


class TestIndividualChecks:
    """六条性质各自的判据（含**它们真的会亮红**的那一面）."""

    def test_zeroes_check_counts_the_masked_cells(self) -> None:
        forward = samples.sample_forward(VARIANT_DECODER_ONLY)
        outcome = verify.check_weights_are_zero_where_masked(forward)
        assert outcome.passed
        assert "18 个权重" in outcome.evidence

    def test_zeroes_check_on_a_full_mask_has_nothing_to_check(self) -> None:
        """全开掩码下“掩码外”是 0 个格子——这条读数必须如实说出来."""
        forward = samples.sample_forward(VARIANT_ENCODER_ONLY)
        outcome = verify.check_weights_are_zero_where_masked(forward)
        assert outcome.passed and "0 个权重" in outcome.evidence

    def test_reads_only_past_fails_on_a_full_mask(self) -> None:
        """**判据真的有分辨力**：把因果掩码换成全开，这一条立刻亮红."""
        params = samples.sample_parameters(VARIANT_DECODER_ONLY)
        report = probe.dependency_matrix(
            params, samples.sample_inputs(), mask_kind=masks.MASK_FULL
        )
        outcome = verify.check_reads_only_past(params, report)
        assert not outcome.passed

    def test_reads_only_past_is_skipped_for_the_encoder(self) -> None:
        params = samples.sample_parameters(VARIANT_ENCODER_ONLY)
        report = probe.dependency_matrix(params, samples.sample_inputs())
        outcome = verify.check_reads_only_past(params, report)
        assert not outcome.applicable and not outcome.passed

    def test_reads_every_position_uses_the_encoder_stream(self) -> None:
        params = samples.sample_parameters(VARIANT_ENCODER_DECODER)
        report = probe.encoder_dependency(
            params, samples.sample_inputs(), source=samples.sample_source()
        )
        outcome = verify.check_reads_every_position(params, report)
        assert outcome.passed

    def test_reads_every_position_fails_on_a_causal_mask(self) -> None:
        params = samples.sample_parameters(VARIANT_ENCODER_ONLY)
        report = probe.dependency_matrix(
            params, samples.sample_inputs(), mask_kind=masks.MASK_CAUSAL
        )
        outcome = verify.check_reads_every_position(params, report)
        assert not outcome.passed

    def test_reads_every_position_is_skipped_for_the_decoder(self) -> None:
        params = samples.sample_parameters(VARIANT_DECODER_ONLY)
        report = probe.dependency_matrix(params, samples.sample_inputs())
        assert not verify.check_reads_every_position(params, report).applicable

    def test_cross_spans_all_sources(self) -> None:
        params = samples.sample_parameters(VARIANT_ENCODER_DECODER)
        report = probe.cross_dependency(
            params, samples.sample_inputs(), samples.sample_source()
        )
        outcome = verify.check_cross_spans_all_sources(params, report)
        assert outcome.passed
        assert "长方形" in outcome.evidence

    def test_cross_check_is_skipped_without_a_cross_stream(self) -> None:
        params = samples.sample_parameters(VARIANT_ENCODER_ONLY)
        assert not verify.check_cross_spans_all_sources(params, None).applicable

    def test_shape_preserved(self) -> None:
        assert verify.check_shape_preserved(
            samples.sample_forward(VARIANT_ENCODER_DECODER)
        ).passed

    def test_deterministic(self) -> None:
        params = samples.sample_parameters(VARIANT_ENCODER_DECODER)
        outcome = verify.check_deterministic(
            params, samples.sample_inputs(), source=samples.sample_source()
        )
        assert outcome.passed

    def test_deterministic_reads_the_cross_weights_too(self) -> None:
        params = samples.sample_parameters(VARIANT_ENCODER_DECODER)
        outcome = verify.check_deterministic(
            params, samples.sample_inputs(), source=samples.sample_source()
        )
        assert "交叉权重相同：True" in outcome.evidence


class TestGradientReport:
    """梯度报告的记录与“不过就抛”."""

    def _passing(self):
        return verify.check_variant_gradients(
            samples.sample_parameters(VARIANT_ENCODER_ONLY),
            samples.sample_inputs(),
            samples.sample_target(),
        )

    def test_report_is_ok_and_has_one_stream(self) -> None:
        report = self._passing()
        assert report.ok
        assert len(report.outcomes) == 1
        assert report.outcomes[0].stream == probe.STREAM_MAIN
        assert report.require_ok() is None

    def test_two_stream_report(self) -> None:
        report = verify.check_variant_gradients(
            samples.sample_parameters(VARIANT_ENCODER_DECODER),
            samples.sample_inputs(),
            samples.sample_target(),
            source=samples.sample_source(),
        )
        assert report.ok
        assert len(report.outcomes) == 2
        assert report.outcome_of(probe.STREAM_SOURCE).samples == 30

    def test_gap_is_at_the_round_off_level(self) -> None:
        """**解析与数值的差在 1e-9 量级**（不是“差不多”——这四位数就是判据）."""
        report = self._passing()
        assert report.outcomes[0].relative_gap < 1e-7

    def test_big_step_breaks_the_check(self) -> None:
        """把步长放大到 1.0，中心差分不再近似导数 ⇒ 报告立刻亮红.

        这条测试证明“梯度校验有分辨力”，而不是一句“它通过了”。
        """
        report = verify.check_variant_gradients(
            samples.sample_parameters(VARIANT_ENCODER_ONLY),
            samples.sample_inputs(),
            samples.sample_target(),
            step=1.0,
        )
        assert not report.ok
        with pytest.raises(GradientError):
            report.require_ok()

    def test_empty_report_is_rejected(self) -> None:
        analytic = self._passing().analytic
        with pytest.raises(ParameterError):
            verify.GradientReport(
                variant=VARIANT_ENCODER_ONLY, tolerance=1e-6, step=1e-6, outcomes=(), analytic=analytic
            )

    def test_bad_tolerance_is_rejected(self) -> None:
        analytic = self._passing().analytic
        outcome = self._passing().outcomes[0]
        with pytest.raises(ParameterError):
            verify.GradientReport(
                variant=VARIANT_ENCODER_ONLY,
                tolerance=0.0,
                step=1e-6,
                outcomes=(outcome,),
                analytic=analytic,
            )

    def test_outcome_lookup_rejects_an_unknown_stream(self) -> None:
        with pytest.raises(ParameterError):
            self._passing().outcome_of("sideways")

    def test_to_dict_and_summary_lines(self) -> None:
        report = self._passing()
        assert report.to_dict()["ok"] is True
        assert "解析范数" in report.summary_lines()[0]
        assert len(report.notes) == 3

    def test_outcome_to_dict(self) -> None:
        outcome = self._passing().outcomes[0]
        payload = outcome.to_dict()
        assert payload["stream"] == probe.STREAM_MAIN
        assert payload["passed"] is True


class TestGradientArguments:
    """数值梯度的入口校验."""

    def test_unknown_stream_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            verify.numerical_stream_gradient(
                samples.sample_parameters(VARIANT_ENCODER_ONLY),
                samples.sample_inputs(),
                samples.sample_target(),
                stream="sideways",
            )

    def test_source_stream_needs_a_source(self) -> None:
        with pytest.raises(AssemblyError):
            verify.numerical_stream_gradient(
                samples.sample_parameters(VARIANT_ENCODER_DECODER),
                samples.sample_inputs(),
                samples.sample_target(),
                stream=probe.STREAM_SOURCE,
            )

    def test_source_stream_needs_two_streams(self) -> None:
        with pytest.raises(AssemblyError):
            verify.numerical_stream_gradient(
                samples.sample_parameters(VARIANT_ENCODER_ONLY),
                samples.sample_inputs(),
                samples.sample_target(),
                stream=probe.STREAM_SOURCE,
                source=samples.sample_source(),
            )

    @pytest.mark.parametrize("step", [0.0, -1e-6, float("inf")])
    def test_bad_step_is_rejected(self, step: float) -> None:
        with pytest.raises(ParameterError):
            verify.numerical_stream_gradient(
                samples.sample_parameters(VARIANT_ENCODER_ONLY),
                samples.sample_inputs(),
                samples.sample_target(),
                step=step,
            )

    def test_non_number_step_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            verify.numerical_stream_gradient(
                samples.sample_parameters(VARIANT_ENCODER_ONLY),
                samples.sample_inputs(),
                samples.sample_target(),
                step="1e-6",  # type: ignore[arg-type]
            )

    def test_params_type_is_checked(self) -> None:
        with pytest.raises(ParameterError):
            verify.check_variant_gradients(
                "params", samples.sample_inputs(), samples.sample_target()  # type: ignore[arg-type]
            )

    def test_check_all_returns_both_reports(self) -> None:
        properties, gradients = verify.check_all(
            samples.sample_parameters(VARIANT_DECODER_ONLY),
            samples.sample_inputs(),
            samples.sample_target(),
        )
        assert properties.ok and gradients.ok

    def test_check_all_for_two_streams(self) -> None:
        properties, gradients = verify.check_all(
            samples.sample_parameters(VARIANT_ENCODER_DECODER),
            samples.sample_inputs(),
            samples.sample_target(),
            source=samples.sample_source(),
        )
        assert properties.ok and gradients.ok


class TestGradientWiring:
    """两处接线失败会被梯度校验抓住（**这是把校验放在整条变体上的理由**）."""

    def test_gradients_match_for_every_variant(self) -> None:
        for variant in VARIANTS:
            report = verify.check_variant_gradients(
                samples.sample_parameters(variant),
                samples.sample_inputs(),
                samples.sample_target(),
                **_source_kwargs(variant),
            )
            assert report.ok, variant

    def test_gradients_match_under_post_placement(self) -> None:
        """post-LN 的链走另一条分支（``encoder_block_backward`` 的另一半）."""
        report = verify.check_variant_gradients(
            samples.sample_parameters(VARIANT_ENCODER_ONLY),
            samples.sample_inputs(),
            samples.sample_target(),
            placement="post",
        )
        assert report.ok

    def test_gradients_match_without_residual(self) -> None:
        report = verify.check_variant_gradients(
            samples.sample_parameters(VARIANT_ENCODER_ONLY),
            samples.sample_inputs(),
            samples.sample_target(),
            use_residual=False,
        )
        assert report.ok

    def test_gradient_of_the_encoder_output_flows_to_the_source(self) -> None:
        """``grad_source_inputs`` 必须穿过整条编码器链——**形状对而数值错**是最坏的那种."""
        forward = samples.sample_forward(VARIANT_ENCODER_DECODER)
        params = samples.sample_parameters(VARIANT_ENCODER_DECODER)
        grads = stacks.variant_backward(
            forward, params, stacks.loss_gradient(forward.output, samples.sample_target())
        )
        assert grads.grad_source_inputs is not None
        assert any(
            value != 0.0 for row in grads.grad_source_inputs for value in row
        )

    def test_source_gradient_matches_numerical(self) -> None:
        report = verify.check_variant_gradients(
            samples.sample_parameters(VARIANT_ENCODER_DECODER),
            samples.sample_inputs(),
            samples.sample_target(),
            source=samples.sample_source(),
        )
        assert report.outcome_of(probe.STREAM_SOURCE).relative_gap < 1e-7
