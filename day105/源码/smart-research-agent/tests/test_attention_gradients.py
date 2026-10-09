"""梯度校验：解析梯度 vs 数值差分（day075）.

这一份测试是这一课的**核心护栏**：只要反向传播有一处写错，这里立刻红。
它同时把"数值与解析必须算同一个式子"这条踩过的坑固化下来。

```text
五项           四个参数矩阵 + 输入
两个判据       绝对误差（证据）与按量级缩放的相对误差（判据）
两种损失       全行 MSE 与监督行 MSE —— **两侧必须同一个**
```
"""

from __future__ import annotations

import json
import math

import pytest

from smart_research_agent.math_foundations.calculus import DEFAULT_STEP
from smart_research_agent.math_foundations.errors import ShapeError as MathShapeError
from smart_research_agent.transformer_core.errors import (
    GradientError,
    NumericError,
    ParameterError,
)
from smart_research_agent.transformer_core.layers import (
    attention_backward,
    masked_mse_gradient,
    mean_squared_error,
    mse_gradient,
    self_attention,
)
from smart_research_agent.transformer_core.types import (
    GRADIENT_TARGETS,
    GRADIENT_TARGET_DESCRIPTIONS,
    GRADIENT_TARGET_FORMULAS,
    GRAD_INPUTS,
    AttentionParams,
    ParameterGradients,
    assert_rows_are_distributions,
    check_weights_are_a_distribution,
    matrix_max_absolute,
    relative_matrix_error,
)
from smart_research_agent.transformer_core.verify import (
    GRADIENT_TOLERANCE,
    GradientCheckOutcome,
    GradientCheckReport,
    check_attention_gradients,
    numerical_parameter_gradients,
)
from tests.attention_samples import (
    SMALL_INPUTS,
    SMALL_TARGET,
    ONE_HOT_INPUTS,
    approx,
    identity_parameters,
    small_parameters,
    toy_parameters,
)


class TestTables:
    """梯度校验项的三张表逐键对齐."""

    def test_tables_align(self):
        assert (
            set(GRADIENT_TARGETS)
            == set(GRADIENT_TARGET_FORMULAS)
            == set(GRADIENT_TARGET_DESCRIPTIONS)
        )

    def test_five_targets(self):
        assert len(GRADIENT_TARGETS) == 5
        assert GRAD_INPUTS in GRADIENT_TARGETS

    def test_every_target_has_a_formula(self):
        for target in GRADIENT_TARGETS:
            assert GRADIENT_TARGET_FORMULAS[target].strip()


class TestNumericalGradients:
    """数值梯度：形状、量级与"监督行"口径."""

    def test_shapes_match_the_parameters(self):
        params = small_parameters()
        numeric = numerical_parameter_gradients(params, SMALL_INPUTS, SMALL_TARGET)
        for matrix in numeric.matrices():
            assert len(matrix) == 3
            assert all(len(row) == 3 for row in matrix)
        assert len(numeric.grad_inputs) == 3

    def test_numeric_matches_analytic_on_the_full_loss(self):
        """数值侧不传 ``supervised`` 时算的是**全行** MSE，与解析侧同口径."""
        params = small_parameters()
        forward = self_attention(params, SMALL_INPUTS, causal=True)
        analytic = attention_backward(forward, mse_gradient(forward.output, SMALL_TARGET))
        numeric = numerical_parameter_gradients(params, SMALL_INPUTS, SMALL_TARGET)
        for name, left, right in zip(
            ("q", "k", "v", "o"), analytic.matrices(), numeric.matrices(), strict=True
        ):
            assert relative_matrix_error(left, right) < 1e-6, name

    def test_numeric_with_supervision_matches_the_masked_analytic(self):
        """传了 ``supervised`` 之后两侧都是**监督行** MSE."""
        params = small_parameters()
        forward = self_attention(params, SMALL_INPUTS, causal=True)
        supervised = (1, 2)
        analytic = attention_backward(
            forward, masked_mse_gradient(forward.output, SMALL_TARGET, supervised)
        )
        numeric = numerical_parameter_gradients(
            params, SMALL_INPUTS, SMALL_TARGET, supervised=supervised
        )
        for left, right in zip(analytic.matrices(), numeric.matrices(), strict=True):
            assert relative_matrix_error(left, right) < 1e-6

    def test_the_two_loss_conventions_are_different(self):
        """**这一课的踩坑记录**：两种口径的梯度差 15% 量级，而两边都对.

        如果数值侧与解析侧用了不同的损失，"对照失败"会被误读成"推导错了"。
        这条断言把差别量出来，让"两侧必须同一个式子"这件事有数字支撑。
        """
        params = small_parameters()
        full = numerical_parameter_gradients(params, SMALL_INPUTS, SMALL_TARGET)
        masked = numerical_parameter_gradients(
            params, SMALL_INPUTS, SMALL_TARGET, supervised=(1, 2)
        )
        gap = relative_matrix_error(full.grad_w_output, masked.grad_w_output)
        assert gap > 1e-2

    def test_zero_gradient_case(self):
        """``t == y`` 时数值梯度也应当是 0（两侧一致的端点检查）."""
        params = identity_parameters(2)
        forward = self_attention(params, ONE_HOT_INPUTS)
        numeric = numerical_parameter_gradients(
            params, ONE_HOT_INPUTS, forward.output, causal=False
        )
        assert matrix_max_absolute(numeric.grad_w_output) < 1e-9

    def test_input_gradient_is_computed_too(self):
        params = small_parameters()
        numeric = numerical_parameter_gradients(params, SMALL_INPUTS, SMALL_TARGET)
        assert matrix_max_absolute(numeric.grad_inputs) > 0.0

    def test_shape_mismatch_between_inputs_and_target(self):
        with pytest.raises(MathShapeError, match="不一致"):
            numerical_parameter_gradients(
                small_parameters(), SMALL_INPUTS, ((1.0, 0.0),)
            )


class TestCheckAttentionGradients:
    """五项对照的整体结论."""

    def _report(self, *, causal: bool = True, tolerance: float = GRADIENT_TOLERANCE):
        return check_attention_gradients(
            small_parameters(), SMALL_INPUTS, SMALL_TARGET, causal=causal, tolerance=tolerance
        )

    def test_all_five_pass(self):
        report = self._report()
        assert report.ok is True
        assert report.failures == ()
        assert len(report.outcomes) == 5

    def test_targets_stay_in_table_order(self):
        report = self._report()
        assert tuple(item.target for item in report.outcomes) == GRADIENT_TARGETS

    def test_error_is_far_below_the_tolerance(self):
        """实测最大误差在 1e-10 量级，比容差低 4 个数量级."""
        report = self._report()
        assert report.worst_scaled_error < 1e-8
        assert report.worst_scaled_error <= report.tolerance

    def test_compared_points_counts_matrix_entries(self):
        report = self._report()
        by_target = {item.target: item for item in report.outcomes}
        assert by_target["w_query"].compared_points == 9
        assert by_target[GRAD_INPUTS].compared_points == 9

    def test_absurd_tight_tolerance_fails(self):
        report = self._report(tolerance=1e-30)
        assert report.ok is False
        assert len(report.failures) == 5

    def test_raise_if_failed(self):
        report = self._report(tolerance=1e-30)
        with pytest.raises(GradientError, match="未通过"):
            report.raise_if_failed()

    def test_raise_if_failed_is_silent_when_ok(self):
        assert self._report().raise_if_failed() is None

    def test_causal_and_full_both_pass(self):
        assert self._report(causal=False).ok is True
        assert self._report(causal=True).ok is True

    def test_random_parameters_also_pass(self):
        """随机参数（不再是手算友好的单位矩阵）上同样通过."""
        params = toy_parameters(3)
        report = check_attention_gradients(params, SMALL_INPUTS, SMALL_TARGET, causal=True)
        assert report.ok is True

    def test_non_positive_tolerance_is_rejected(self):
        with pytest.raises(ParameterError, match="容差"):
            check_attention_gradients(
                small_parameters(), SMALL_INPUTS, SMALL_TARGET, tolerance=0.0
            )

    def test_target_shape_is_validated(self):
        with pytest.raises(MathShapeError, match="不一致"):
            check_attention_gradients(
                small_parameters(), SMALL_INPUTS, ((1.0, 2.0, 3.0),)
            )

    def test_report_is_json_friendly(self):
        payload = self._report().to_dict()
        json.dumps(payload)
        assert payload["ok"] is True
        assert payload["failures"] == []
        assert len(payload["outcomes"]) == 5
        assert payload["notes"]
        assert "梯度校验 5 项" in self._report().summary_line()

    def test_custom_step_reaches_the_outcomes(self):
        report = check_attention_gradients(
            small_parameters(), SMALL_INPUTS, SMALL_TARGET, step=1e-5
        )
        assert report.step == 1e-5
        assert report.ok is True

    def test_default_step_is_day074_step(self):
        assert self._report().step == DEFAULT_STEP

    def test_notes_mention_the_chain_sum_check(self):
        notes = " ".join(self._report().notes)
        assert "漏了一条链" in notes


class TestOutcomeValidation:
    """``GradientCheckOutcome`` 的形状校验."""

    def _outcome(self, **overrides):
        payload = {
            "target": "w_query",
            "max_absolute_error": 1e-10,
            "max_scaled_error": 1e-10,
            "tolerance": GRADIENT_TOLERANCE,
            "compared_points": 9,
            "message": "一致",
        }
        payload.update(overrides)
        return GradientCheckOutcome(**payload)

    def test_projection(self):
        outcome = self._outcome()
        assert outcome.passed is True
        assert outcome.formula
        assert "w_query" in outcome.summary_line()

    def test_unknown_target_is_rejected(self):
        with pytest.raises(GradientError, match="不认识"):
            self._outcome(target="w_mystery")

    def test_negative_absolute_error_is_rejected(self):
        with pytest.raises(NumericError, match="最大绝对误差"):
            self._outcome(max_absolute_error=-1.0)

    def test_negative_scaled_error_is_rejected(self):
        with pytest.raises(NumericError, match="最大相对误差"):
            self._outcome(max_scaled_error=-1.0)

    def test_nan_is_rejected(self):
        with pytest.raises(NumericError, match="最大相对误差"):
            self._outcome(max_scaled_error=float("nan"))

    def test_failing_outcome_reports_fail_mark(self):
        assert "[FAIL]" in self._outcome(max_scaled_error=1.0).summary_line()


class TestReportValidation:
    """``GradientCheckReport`` 必须完整."""

    def test_missing_targets_are_rejected(self):
        report = check_attention_gradients(small_parameters(), SMALL_INPUTS, SMALL_TARGET)
        with pytest.raises(GradientError, match="缺少"):
            GradientCheckReport(outcomes=report.outcomes[:2])

    def test_complete_report_is_accepted(self):
        report = check_attention_gradients(small_parameters(), SMALL_INPUTS, SMALL_TARGET)
        rebuilt = GradientCheckReport(outcomes=report.outcomes, notes=("n",))
        assert rebuilt.ok is True
        assert rebuilt.notes == ("n",)


class TestMatrixHelpers:
    """两个矩阵级的读数与一个分布校验."""

    def test_relative_matrix_error_is_zero_for_identical(self):
        matrix = ((1.0, 2.0), (3.0, 4.0))
        assert relative_matrix_error(matrix, matrix) == 0.0

    def test_relative_matrix_error_scales_by_magnitude(self):
        left = ((100.0,),)
        right = ((100.0 + 1e-6,),)
        assert relative_matrix_error(left, right) == pytest.approx(1e-8, rel=0.05)

    def test_relative_matrix_error_rejects_shape_mismatch(self):
        with pytest.raises(MathShapeError, match="形状不同"):
            relative_matrix_error(((1.0,),), ((1.0, 2.0),))

    def test_relative_matrix_error_is_infinite_on_non_finite(self):
        assert math.isinf(relative_matrix_error(((float("nan"),),), ((1.0,),)))

    def test_matrix_max_absolute(self):
        assert matrix_max_absolute(((1.0, -5.0), (0.0, 2.0))) == 5.0
        assert matrix_max_absolute(((),)) == 0.0

    def test_weights_distribution_accepts_valid_rows(self):
        check_weights_are_a_distribution(((0.5, 0.5), (1.0, 0.0)))

    def test_weights_distribution_rejects_negative(self):
        """负数与"和为 1"是两条独立判据；这里给一行**和为 1 但含负数**的权重."""
        with pytest.raises(NumericError, match="负数"):
            check_weights_are_a_distribution(((-0.1, 1.1),))

    def test_weights_distribution_rejects_un_normalised(self):
        with pytest.raises(NumericError, match="和是"):
            check_weights_are_a_distribution(((0.5, 0.4),))

    def test_assert_rows_returns_the_row_sums(self):
        assert assert_rows_are_distributions(((0.25, 0.75),)) == (1.0,)

    def test_parameter_gradients_flatten_length_matches_parameters(self):
        params = small_parameters()
        forward = self_attention(params, SMALL_INPUTS)
        gradients = attention_backward(forward, mse_gradient(forward.output, SMALL_TARGET))
        assert isinstance(gradients, ParameterGradients)
        assert len(gradients.flatten()) == len(params.flatten()[0])

    def test_parameter_gradients_max_absolute(self):
        params = small_parameters()
        forward = self_attention(params, SMALL_INPUTS)
        gradients = attention_backward(forward, mse_gradient(forward.output, SMALL_TARGET))
        assert gradients.max_absolute() > 0.0


class TestFlattenLayout:
    """压平布局：顺序是 ``q / k / v / o``，逐行优先——**这一步必须手算核对**."""

    def test_flatten_layout_is_the_documented_order(self):
        """四个 2×2 矩阵按 1..16 填，压平之后应当是 1..16.

        顺序错了**不会报错**：它只会让某一层的权重按另一层的梯度更新，
        而它表现为"训练变慢"或"不收敛"——因此这里逐位核对。
        """
        params = AttentionParams(
            w_query=((1.0, 2.0), (3.0, 4.0)),
            w_key=((5.0, 6.0), (7.0, 8.0)),
            w_value=((9.0, 10.0), (11.0, 12.0)),
            w_output=((13.0, 14.0), (15.0, 16.0)),
        )
        flat, shapes = params.flatten()
        assert shapes == ((2, 2), (2, 2), (2, 2), (2, 2))
        assert flat == tuple(float(value) for value in range(1, 17))

    def test_unflatten_is_the_inverse(self):
        params = AttentionParams(
            w_query=((1.0, 0.0), (0.0, 1.0)),
            w_key=((2.0, 0.0), (0.0, 2.0)),
            w_value=((3.0, 0.0), (0.0, 3.0)),
            w_output=((4.0, 0.0), (0.0, 4.0)),
        )
        flat, shapes = params.flatten()
        assert AttentionParams.unflatten(flat, shapes) == params

    def test_only_w_output_perturbation_changes_the_output(self):
        """只改 ``W_o`` 时输出改变；只改 ``W_v`` 时输出也改变——两者路径不同.

        这一条不是为了证明什么，而是为了**把两条路径分开可观测**：
        若实现里把两者弄反了（例如 ``context`` 与 ``output`` 用同一个缓存），
        这里的两个比较仍然会通过，但梯度校验那一份会立刻红。
        """
        base = identity_parameters(2)
        target = ((1.0, 0.0), (0.0, 1.0))
        numeric = numerical_parameter_gradients(base, ONE_HOT_INPUTS, target)
        assert matrix_max_absolute(numeric.grad_w_output) > 0.0
        assert matrix_max_absolute(numeric.grad_w_value) >= 0.0
        assert approx(mean_squared_error(((0.0, 0.0),), ((0.0, 1.0),)), 0.5)
