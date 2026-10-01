"""``arch_variants.probe`` 的测试：**用扰动把“因果”量出来**（day082 的核心判据）."""

from __future__ import annotations

import pytest

from smart_research_agent.arch_variants import masks, probe, stacks
from smart_research_agent.arch_variants.errors import (
    AssemblyError,
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.arch_variants.types import (
    VARIANTS,
    VARIANT_DECODER_ONLY,
    VARIANT_ENCODER_DECODER,
    VARIANT_ENCODER_ONLY,
    VariantShape,
)
from tests import arch_variants_samples as samples


class TestPerturbRow:
    """扰动一个位置：只改那一个位置，其余**逐位**不变."""

    def test_only_the_target_row_changes(self) -> None:
        matrix = ((1.0, 2.0), (3.0, 4.0))
        moved = probe.perturb_row(matrix, 0, factor=2.0)
        assert moved == ((2.0, 4.0), (3.0, 4.0))
        assert moved[1] == matrix[1]

    def test_index_out_of_range(self) -> None:
        with pytest.raises(ParameterError):
            probe.perturb_row(((1.0,),), 1)

    def test_bool_index_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            probe.perturb_row(((1.0,),), True)  # type: ignore[arg-type]

    def test_non_integer_index_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            probe.perturb_row(((1.0,),), 0.5)  # type: ignore[arg-type]

    def test_factor_one_is_rejected(self) -> None:
        """扰动为 0 时“所有格子都不依赖”——一张什么都没量的表必须当场失败."""
        with pytest.raises(ParameterError):
            probe.perturb_row(((1.0,),), 0, factor=1.0)

    def test_non_number_factor_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            probe.perturb_row(((1.0,),), 0, factor="2")  # type: ignore[arg-type]

    def test_non_finite_factor_is_rejected(self) -> None:
        with pytest.raises(NumericError):
            probe.perturb_row(((1.0,),), 0, factor=float("inf"))


class TestDependencyReportRecord:
    """依赖表的校验与各种读数."""

    def _report(self, **overrides: object):
        matrix = ((1.0, 0.0), (1.0, 1.0))
        mask = ((True, False), (True, True))
        payload: dict[str, object] = {
            "variant": VARIANT_DECODER_ONLY,
            "stream": probe.STREAM_MAIN,
            "shape": samples.sample_shape(tokens=2, hidden=6, layers=2, sources=2),
            "matrix": matrix,
            "mask": mask,
        }
        payload.update(overrides)
        return probe.DependencyReport(**payload)  # type: ignore[arg-type]

    def test_readings(self) -> None:
        report = self._report()
        assert report.rows == 2 and report.columns == 2
        assert report.blocked_gap == 0.0
        assert report.allowed_floor == 1.0
        assert report.exact is True
        assert report.matches_mask is True
        assert report.leaks() == ()
        assert report.reach(0) == 1
        assert report.reach_profile() == (1, 2)
        assert probe.contract_gap(report) == 0.0
        assert probe.largest_change(report) == 1.0

    def test_leaks_are_visible(self) -> None:
        report = self._report(matrix=((1.0, 0.5), (1.0, 1.0)))
        assert report.leaks() == ((0, 1),)
        assert report.matches_mask is False
        # 那个 0.5 落在**被挡住**的格子上 ⇒ blocked_gap 不再是 0，exact 为假
        assert report.exact is False
        assert report.blocked_gap == pytest.approx(0.5)
        assert probe.contract_gap(report) == pytest.approx(0.5)

    def test_dead_cells_are_visible(self) -> None:
        report = self._report(matrix=((1.0, 0.0), (0.0, 1.0)))
        assert report.matches_mask is False
        assert report.allowed_floor == 0.0
        assert probe.contract_gap(report) == 1.0

    def test_unknown_variant_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            self._report(variant="bert")

    def test_unknown_stream_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            self._report(stream="sideways")

    def test_shape_mismatch_is_rejected(self) -> None:
        with pytest.raises(ShapeError):
            self._report(mask=((True,), (True,)))

    def test_negative_tolerance_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            self._report(tolerance=-1.0)

    def test_reach_out_of_range(self) -> None:
        with pytest.raises(ParameterError):
            self._report().reach(5)

    def test_reach_bool_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            self._report().reach(True)  # type: ignore[arg-type]

    def test_dependency_of_checks_bounds(self) -> None:
        report = self._report()
        assert probe.dependency_of(report, 0, 0) == 1.0
        with pytest.raises(ParameterError):
            probe.dependency_of(report, 2, 0)
        with pytest.raises(ParameterError):
            probe.dependency_of(report, 0, 5)
        with pytest.raises(ParameterError):
            probe.dependency_of(report, True, 0)  # type: ignore[arg-type]

    def test_is_rectangular(self) -> None:
        assert not probe.is_rectangular(self._report())
        rectangular = self._report(matrix=((1.0, 1.0, 1.0),), mask=((True, True, True),))
        assert probe.is_rectangular(rectangular)

    def test_causal_verdict_reads_the_table(self) -> None:
        assert probe.causal_verdict(self._report())
        full = self._report(matrix=((1.0, 1.0), (1.0, 1.0)), mask=((True, True), (True, True)))
        assert not probe.causal_verdict(full)

    def test_summary_line_and_dict(self) -> None:
        report = self._report()
        line = report.summary_line()
        assert "decoder_only" in line and "可达 (1, 2)" in line
        payload = report.to_dict()
        assert payload["exact"] is True
        assert payload["matrix"] == [[1.0, 0.0], [1.0, 1.0]]

    def test_report_matches_free_function(self) -> None:
        assert probe.report_matches(self._report()) is True


class TestDependencyMatrix:
    """逐列扰动：三个变体的实测表."""

    def test_encoder_only_sees_everything(self) -> None:
        params = samples.sample_parameters(VARIANT_ENCODER_ONLY)
        report = probe.dependency_matrix(params, samples.sample_inputs())
        assert report.reach_profile() == (4, 4, 4, 4)
        assert report.matches_mask
        assert report.allowed_floor > 0.0

    def test_decoder_only_sees_only_the_past(self) -> None:
        """**本课的主判据**：第 i 行只依赖 j <= i，而且挡掉的那些格子**逐位是 0**."""
        params = samples.sample_parameters(VARIANT_DECODER_ONLY)
        report = probe.dependency_matrix(params, samples.sample_inputs())
        assert report.reach_profile() == (1, 2, 3, 4)
        assert report.exact
        assert report.matches_mask
        for row in range(4):
            for column in range(4):
                if column > row:
                    assert report.matrix[row][column] == 0.0

    def test_encoder_decoder_main_stream_is_causal(self) -> None:
        params = samples.sample_parameters(VARIANT_ENCODER_DECODER)
        report = probe.dependency_matrix(
            params, samples.sample_inputs(), source=samples.sample_source()
        )
        assert report.reach_profile() == (1, 2, 3, 4)
        assert report.matches_mask and report.exact

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_source_argument_is_only_for_two_streams(self, variant: str) -> None:
        params = samples.sample_parameters(variant)
        if variant == VARIANT_ENCODER_DECODER:
            assert probe.dependency_matrix(
                params, samples.sample_inputs(), source=samples.sample_source()
            ).matches_mask
        else:
            with pytest.raises(AssemblyError):
                probe.dependency_matrix(
                    params, samples.sample_inputs(), source=samples.sample_source()
                )

    def test_wrong_mask_is_detected(self) -> None:
        """反证：把因果掩码换成全开，同一张表上立刻出现越界."""
        params = samples.sample_parameters(VARIANT_DECODER_ONLY)
        report = probe.dependency_matrix(
            params, samples.sample_inputs(), mask_kind=masks.MASK_FULL
        )
        assert not probe.causal_verdict(report)
        assert report.matches_mask  # 与**它自己那张**掩码一致（全开）
        assert report.reach_profile() == (4, 4, 4, 4)

    def test_tolerance_above_zero_degrades_the_table(self) -> None:
        """把阈值调大之后，“这一点点读数”不再被算作依赖——表会退化."""
        params = samples.sample_parameters(VARIANT_DECODER_ONLY)
        report = probe.dependency_matrix(params, samples.sample_inputs(), perturbation=1.0 + 1e-3)
        strict = probe.DependencyReport(
            variant=report.variant,
            stream=report.stream,
            shape=report.shape,
            matrix=report.matrix,
            mask=report.mask,
            tolerance=0.0,
        )
        loose = probe.DependencyReport(
            variant=report.variant,
            stream=report.stream,
            shape=report.shape,
            matrix=report.matrix,
            mask=report.mask,
            tolerance=1.0,
        )
        assert loose.matches_mask is False or strict.matches_mask is False
        assert sum(sum(row) for row in loose.measured_mask) <= sum(
            sum(row) for row in strict.measured_mask
        )

    def test_params_type_is_checked(self) -> None:
        with pytest.raises(ParameterError):
            probe.dependency_matrix("params", samples.sample_inputs())  # type: ignore[arg-type]

    def test_input_shape_is_checked(self) -> None:
        params = samples.sample_parameters(VARIANT_ENCODER_ONLY)
        with pytest.raises(ShapeError):
            probe.dependency_matrix(params, samples.sample_inputs()[:2])

    def test_notes_mention_the_perturbation(self) -> None:
        params = samples.sample_parameters(VARIANT_ENCODER_ONLY)
        report = probe.dependency_matrix(params, samples.sample_inputs())
        assert any("逐列扰动" in note for note in report.notes)


class TestCrossAndEncoderDependency:
    """另外两条流的实测表（长方形的那一路与编码器那一侧）."""

    def test_cross_dependency_is_rectangular_and_full(self) -> None:
        params = samples.sample_parameters(VARIANT_ENCODER_DECODER)
        report = probe.cross_dependency(
            params, samples.sample_inputs(), samples.sample_source()
        )
        assert (report.rows, report.columns) == (4, 5)
        assert report.reach_profile() == (5, 5, 5, 5)
        assert report.matches_mask and report.exact
        assert probe.is_rectangular(report)

    def test_cross_dependency_requires_two_streams(self) -> None:
        params = samples.sample_parameters(VARIANT_ENCODER_ONLY)
        with pytest.raises(AssemblyError):
            probe.cross_dependency(params, samples.sample_inputs(), samples.sample_source())

    def test_cross_dependency_checks_the_source_shape(self) -> None:
        params = samples.sample_parameters(VARIANT_ENCODER_DECODER)
        with pytest.raises(ShapeError):
            probe.cross_dependency(params, samples.sample_inputs(), samples.sample_source()[:2])

    def test_encoder_dependency_for_a_single_stream_matches_the_main_table(self) -> None:
        params = samples.sample_parameters(VARIANT_ENCODER_ONLY)
        main = probe.dependency_matrix(params, samples.sample_inputs())
        encoder = probe.encoder_dependency(params, samples.sample_inputs())
        assert encoder.matrix == main.matrix
        assert encoder.stream == probe.STREAM_ENCODER

    def test_encoder_dependency_for_two_streams_is_bidirectional(self) -> None:
        params = samples.sample_parameters(VARIANT_ENCODER_DECODER)
        report = probe.encoder_dependency(
            params, samples.sample_inputs(), source=samples.sample_source()
        )
        assert report.reach_profile() == (5, 5, 5, 5, 5)
        assert report.matches_mask

    def test_encoder_dependency_needs_an_explicit_source(self) -> None:
        params = samples.sample_parameters(VARIANT_ENCODER_DECODER)
        with pytest.raises(AssemblyError):
            probe.encoder_dependency(params, samples.sample_inputs())

    def test_encoder_dependency_rejects_source_for_single_stream(self) -> None:
        params = samples.sample_parameters(VARIANT_ENCODER_ONLY)
        with pytest.raises(AssemblyError):
            probe.encoder_dependency(
                params, samples.sample_inputs(), source=samples.sample_source()
            )

    def test_encoder_dependency_checks_the_source_shape(self) -> None:
        params = samples.sample_parameters(VARIANT_ENCODER_DECODER)
        with pytest.raises(ShapeError):
            probe.encoder_dependency(
                params, samples.sample_inputs(), source=samples.sample_source()[:2]
            )

    def test_encoder_dependency_checks_the_params_type(self) -> None:
        with pytest.raises(ParameterError):
            probe.encoder_dependency("params", samples.sample_inputs())  # type: ignore[arg-type]


class TestRowChangeHelpers:
    """前缀稳定性用到的两个辅助函数."""

    def test_row_change_profile(self) -> None:
        before = ((1.0, 1.0), (1.0, 1.0))
        after = ((1.0, 2.0), (1.0, 1.0))
        assert probe.row_change_profile(before, after) == ((1.0,), (0.0,))

    def test_row_change_profile_checks_shapes(self) -> None:
        with pytest.raises(ShapeError):
            probe.row_change_profile(((1.0,),), ((1.0, 2.0),))

    def test_changed_rows(self) -> None:
        before = ((1.0, 1.0), (1.0, 1.0))
        after = ((1.0, 1.0), (1.0, 3.0))
        assert probe.changed_rows(before, after) == (1,)

    def test_changed_rows_with_tolerance(self) -> None:
        before = ((1.0,),)
        after = ((1.0 + 1e-9,),)
        assert probe.changed_rows(before, after) == (0,)
        assert probe.changed_rows(before, after, tolerance=1e-6) == ()


class TestBitExactness:
    """**这一课最硬的一条**：被掩码挡掉的位置读数是 ``0.0``，不是 ``1e-16``."""

    def test_every_blocked_cell_is_exactly_zero(self) -> None:
        params = samples.sample_parameters(VARIANT_DECODER_ONLY)
        report = probe.dependency_matrix(params, samples.sample_inputs())
        blocked = [
            report.matrix[row][column]
            for row in range(4)
            for column in range(4)
            if not report.mask[row][column]
        ]
        assert blocked
        assert all(value == 0.0 for value in blocked)

    def test_the_masked_scores_are_never_read(self) -> None:
        """根因在实现里：``masked_softmax_rows`` 只在允许的位置上做 softmax."""
        params = samples.sample_parameters(VARIANT_DECODER_ONLY)
        inputs = samples.sample_inputs()
        forward = stacks.variant_forward(params, inputs)
        moved = stacks.variant_forward(
            params, probe.perturb_row(inputs, 3, factor=3.0)
        )
        # 第 0 行看不到位置 3，因此它的输出**逐位**不变
        assert forward.output[0] == moved.output[0]
        assert forward.output[3] != moved.output[3]

    def test_perturbing_a_visible_position_changes_the_row(self) -> None:
        params = samples.sample_parameters(VARIANT_ENCODER_ONLY)
        inputs = samples.sample_inputs()
        forward = stacks.variant_forward(params, inputs)
        moved = stacks.variant_forward(params, probe.perturb_row(inputs, 0, factor=3.0))
        assert forward.output[3] != moved.output[3]

    def test_shape_type_is_checked_in_the_report(self) -> None:
        with pytest.raises(ParameterError):
            probe.DependencyReport(
                variant=VARIANT_ENCODER_ONLY,
                stream=probe.STREAM_MAIN,
                shape="shape",  # type: ignore[arg-type]
                matrix=((1.0,),),
                mask=((True,),),
            )
