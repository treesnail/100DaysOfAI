"""``math_foundations.gradcheck``：解析梯度与数值差分的六项对照（day074）.

这一份测试的重点与 ``test_math_bridge.py`` 一致——**结论必须能被独立复核**：

```text
六项全部 agrees        由逐点比较得出（不是"跑一遍看差不多"）
两种结论而不是三种      没有 rejected，理由写在模块说明里（两侧都是我们自己算的）
容差由分辨率决定        difference_resolution 把"为什么是 1e-8"变成一个可验算的数字
判据按量级缩放          max_scaled_gap：导数为 100 的量不该被绝对阈值卡死
```

以及一条**这一课最想讲清的性质**（``TestSoftmaxJacobianCancellation``）：

```text
softmax 雅可比写错时，交叉熵的梯度**仍然是 p − onehot(y)**——
因此"雅可比单独比一次"是发现它写错的唯一办法。
```
"""

from __future__ import annotations

import json
import math

import pytest

from smart_research_agent.math_foundations.calculus import DEFAULT_STEP
from smart_research_agent.math_foundations.errors import MathError, NumericError
from smart_research_agent.math_foundations.gradcheck import (
    AUTOGRAD_CASES,
    FLOAT_EPSILON,
    GRADIENT_CHECKS,
    GRADIENT_TOLERANCE,
    LOGIT_VECTORS,
    PERPLEXITY_INPUTS,
    SCALAR_POINTS,
    TARGET_INDICES,
    GradientOutcome,
    GradientReport,
    check_autograd_chain,
    check_cross_entropy_gradient,
    check_gradients_all,
    check_log_sigmoid_gradient,
    check_perplexity_gradient,
    check_sigmoid_gradient,
    check_softmax_jacobian,
    difference_resolution,
    max_absolute_gap,
    max_scaled_gap,
)
from smart_research_agent.math_foundations.types import (
    GRADIENT_STATUSES,
    GRADIENT_TARGETS,
    GRADIENT_TARGET_DESCRIPTIONS,
    GRADIENT_TARGET_FORMULAS,
    GRADIENT_TARGET_SOURCES,
)
from tests.math_samples import approx


class TestTables:
    """四张表逐键对齐（少一个键的对照项会静默地不被执行）."""

    def test_four_tables_align(self):
        assert (
            set(GRADIENT_TARGETS)
            == set(GRADIENT_TARGET_DESCRIPTIONS)
            == set(GRADIENT_TARGET_FORMULAS)
            == set(GRADIENT_TARGET_SOURCES)
        )

    def test_six_targets(self):
        assert len(GRADIENT_TARGETS) == 6

    def test_two_statuses_only(self):
        """比函数值对照少一种状态：两侧都是我们自己算的，不存在"一边拒绝一边放行"."""
        assert GRADIENT_STATUSES == ("agrees", "differs")

    def test_every_target_has_a_formula(self):
        for target in GRADIENT_TARGETS:
            assert GRADIENT_TARGET_FORMULAS[target].strip()

    def test_sources_point_at_real_modules(self):
        for target in GRADIENT_TARGETS:
            assert GRADIENT_TARGET_SOURCES[target].startswith("math_foundations.")


class TestSchedulingTable:
    """调度表与目标表逐键对齐."""

    def test_keys_match(self):
        assert set(GRADIENT_CHECKS) == set(GRADIENT_TARGETS)
        assert tuple(GRADIENT_CHECKS) == GRADIENT_TARGETS

    def test_every_target_returns_a_verdict(self):
        for name, function in GRADIENT_CHECKS.items():
            outcome = function()
            assert outcome.target == name
            assert outcome.status in GRADIENT_STATUSES


class TestIndividualChecks:
    """六项对照各自都要能单独跑、单独读."""

    def test_softmax_jacobian_agrees(self):
        outcome = check_softmax_jacobian()
        assert outcome.status == "agrees"
        assert outcome.compared_points == sum(len(item) ** 2 for item in LOGIT_VECTORS)

    def test_cross_entropy_gradient_agrees(self):
        outcome = check_cross_entropy_gradient()
        assert outcome.status == "agrees"
        assert outcome.compared_points == sum(len(item) for item in LOGIT_VECTORS)
        assert "logits" in outcome.formula or "p − onehot" in outcome.formula

    def test_sigmoid_gradient_agrees(self):
        outcome = check_sigmoid_gradient()
        assert outcome.status == "agrees"
        assert outcome.compared_points == len(SCALAR_POINTS)

    def test_log_sigmoid_gradient_agrees(self):
        assert check_log_sigmoid_gradient().status == "agrees"

    def test_perplexity_gradient_agrees(self):
        outcome = check_perplexity_gradient()
        assert outcome.status == "agrees"
        assert outcome.compared_points == len(PERPLEXITY_INPUTS)

    def test_autograd_chain_agrees(self):
        outcome = check_autograd_chain()
        assert outcome.status == "agrees"
        assert outcome.compared_points == sum(len(point) for _n, _e, point in AUTOGRAD_CASES)

    def test_absurdly_tight_tolerance_flags_a_difference(self):
        """容差低于分辨率时对照会报 differs——这解释了"容差为什么不能拍得太小"."""
        outcome = check_softmax_jacobian(tolerance=1e-30)
        assert outcome.status == "differs"
        assert outcome.ok is False

    def test_step_that_is_too_small_degrades_the_comparison(self):
        """步长太小 → 分辨率变差 → 对照报 ``differs``（而两边都没错）.

        这正是本模块容差取 ``1e-8`` 而不是更小的理由：
        ``h = 1e-8`` 时的分辨率下限是 ``eps·100/(2·1e-8) ≈ 1.1e-6``
        （打分的量级是 10），它把"正确的实现"逼到了容差之外。
        把容差放大到分辨率之上，结论立刻回到 ``agrees``——
        **这一步证明那条"不一致"是测量手段的问题，不是代码的问题。**
        """
        assert check_softmax_jacobian(step=1e-8).status == "differs"
        assert check_softmax_jacobian(step=1e-8, tolerance=1e-5).status == "agrees"
        assert check_softmax_jacobian(step=1e-4).status == "agrees"


class TestSoftmaxJacobianCancellation:
    """这一课最想讲清的性质：**雅可比在交叉熵的梯度里被抵消掉了**."""

    def test_cross_entropy_gradient_equals_minus_jacobian_column_over_p(self):
        """``∂(−log p_y)/∂z = −J[:, y] / p_y``，而它恰好化简成 ``p − onehot(y)``.

        等式的左边**用到了完整雅可比**，右边只用到概率。两者相等，
        说明雅可比里那一堆"δ_ij − p_j"在乘以 ``−1/p_y`` 并按行求和之后
        全部抵消。因此：

        ```text
        雅可比写错 → 交叉熵的梯度**看起来仍然正确**（如果实现没有走雅可比这条路）
                   → 只有单独比一次雅可比才能发现
        ```
        """
        from smart_research_agent.math_foundations.calculus import jacobian
        from smart_research_agent.math_foundations.linalg import softmax

        for logits, target in zip(LOGIT_VECTORS, TARGET_INDICES):
            probabilities = softmax(logits)
            matrix = jacobian(lambda vector: softmax(vector), logits)
            for index in range(len(logits)):
                left = -matrix[index][target] / probabilities[target]
                right = probabilities[index] - (1.0 if index == target else 0.0)
                assert approx(left, right, tolerance=1e-7)

    def test_jacobian_rows_sum_to_zero(self):
        """雅可比每一行之和为 0（输出恒在概率单纯形上）."""
        from smart_research_agent.math_foundations.calculus import jacobian
        from smart_research_agent.math_foundations.linalg import softmax

        for logits in LOGIT_VECTORS:
            for row in jacobian(lambda vector: softmax(vector), logits):
                assert approx(math.fsum(row), 0.0, tolerance=1e-8)

    def test_logits_of_length_one_give_a_zero_jacobian(self):
        """只有一个事件时 softmax 恒为 1，雅可比退化成一个 0.

        ``(1 − 1) = 0``：这一条断言的是"退化情形没有被特殊处理掉"——
        长度为 1 的打分在真实模型里不会出现，但它出现在**掩码之后**
        （一行里只剩一个允许的位置），因此它必须是合法的输入而不是边界条件。
        """
        from smart_research_agent.math_foundations.calculus import jacobian
        from smart_research_agent.math_foundations.linalg import softmax

        single = LOGIT_VECTORS[0]
        assert len(single) == 1
        assert jacobian(lambda vector: softmax(vector), single) == ((pytest.approx(0.0),),)
        assert check_softmax_jacobian().compared_points == sum(
            len(item) ** 2 for item in LOGIT_VECTORS
        )


class TestFullReport:
    """汇总报告：六项齐全、顺序一致、JSON 友好."""

    def test_report_covers_all_targets_in_order(self):
        report = check_gradients_all()
        assert tuple(item.target for item in report.outcomes) == GRADIENT_TARGETS

    def test_report_has_no_disagreement(self):
        report = check_gradients_all()
        assert report.ok is True
        assert report.disagreements == ()

    def test_report_summary_counts(self):
        summary = check_gradients_all().summary_line()
        assert "梯度对照 6 项" in summary
        assert "一致 6" in summary
        assert "分歧 0" in summary

    def test_worst_error_is_the_largest_of_the_six(self):
        report = check_gradients_all()
        assert report.worst_error == max(item.max_absolute_error for item in report.outcomes)
        assert report.worst_error < 1e-6

    def test_report_is_json_friendly(self):
        payload = check_gradients_all().to_dict()
        json.dumps(payload)
        assert payload["ok"] is True
        assert payload["disagreements"] == []
        assert len(payload["outcomes"]) == 6
        assert payload["notes"]

    def test_missing_target_is_rejected(self):
        with pytest.raises(MathError, match="缺少"):
            GradientReport(outcomes=(check_sigmoid_gradient(),))

    def test_custom_tolerance_reaches_the_outcomes(self):
        report = check_gradients_all(tolerance=1e-6)
        assert report.tolerance == 1e-6
        assert all(item.tolerance == 1e-6 for item in report.outcomes)

    def test_non_positive_tolerance_is_rejected(self):
        with pytest.raises(NumericError, match="容差必须为正"):
            check_gradients_all(tolerance=0.0)

    def test_notes_explain_the_cancellation_and_the_resolution(self):
        notes = " ".join(check_gradients_all().notes)
        assert "抵消" in notes
        assert "分辨率" in notes


class TestOutcomeShape:
    """结论形状：只允许两种状态、误差不能为负、JSON 可读."""

    def _outcome(self, **overrides) -> GradientOutcome:
        payload = {
            "target": "softmax",
            "status": "agrees",
            "max_absolute_error": 1e-12,
            "compared_points": 3,
            "message": "逐点一致",
            "source": "math_foundations.x",
        }
        payload.update(overrides)
        return GradientOutcome(**payload)

    def test_projection_carries_the_formula(self):
        outcome = self._outcome()
        assert outcome.formula
        payload = outcome.to_dict()
        assert payload["ok"] is True
        assert payload["formula"]
        assert "softmax" in outcome.summary_line()

    def test_unknown_target_is_rejected(self):
        with pytest.raises(MathError, match="不认识的梯度对照项"):
            self._outcome(target="mystery")

    def test_unknown_status_is_rejected(self):
        with pytest.raises(MathError, match="不认识的梯度对照结论"):
            self._outcome(status="rejected")

    def test_negative_error_is_rejected(self):
        with pytest.raises(NumericError, match="最大绝对误差"):
            self._outcome(max_absolute_error=-1.0)

    def test_negative_scaled_error_is_rejected(self):
        with pytest.raises(NumericError, match="最大相对误差"):
            self._outcome(max_scaled_error=-0.5)

    def test_nan_error_is_rejected(self):
        with pytest.raises(NumericError, match="最大绝对误差"):
            self._outcome(max_absolute_error=float("nan"))

    def test_differing_outcome_is_not_ok(self):
        assert self._outcome(status="differs").ok is False


class TestComparisonHelpers:
    """两个比较器本身：长度不一致、非有限数、缩放判据."""

    def test_identical_vectors_have_zero_gap(self):
        assert max_absolute_gap((1.0, 2.0), (1.0, 2.0)) == 0.0
        assert max_scaled_gap((1.0, 2.0), (1.0, 2.0)) == 0.0

    def test_length_mismatch_is_infinite(self):
        assert math.isinf(max_absolute_gap((1.0,), (1.0, 2.0)))
        assert math.isinf(max_scaled_gap((1.0,), (1.0, 2.0)))

    def test_non_finite_values_are_infinite(self):
        assert math.isinf(max_absolute_gap((float("nan"),), (1.0,)))
        assert math.isinf(max_scaled_gap((1.0,), (float("inf"),)))

    def test_scaling_favours_large_magnitudes(self):
        """同样的绝对误差，在量级为 100 的分量上给出的相对误差小 100 倍."""
        wobbled = (100.0 + 1e-8,)
        assert max_absolute_gap((100.0,), wobbled) == pytest.approx(1e-8, rel=0.05)
        assert max_scaled_gap((100.0,), wobbled) == pytest.approx(1e-10, rel=0.05)

    def test_small_values_fall_back_to_absolute(self):
        """分母取 ``max(1, |r|)``：0 附近自动退化成绝对判据."""
        assert max_scaled_gap((0.0,), (1e-9,)) == pytest.approx(1e-9)


class TestDifferenceResolution:
    """分辨率下限：把"容差该取多少"变成一次可验算的计算."""

    def test_hand_computed_values(self):
        """``eps·|f|/(2h)``：|f| = 1 时约 1.1e-10、|f| = 100 时约 1.1e-8."""
        assert difference_resolution(magnitude=1.0) == pytest.approx(
            FLOAT_EPSILON / (2.0 * DEFAULT_STEP)
        )
        assert difference_resolution(magnitude=1.0) == pytest.approx(1.11e-10, rel=0.02)
        assert difference_resolution(magnitude=100.0) == pytest.approx(1.11e-8, rel=0.02)

    def test_scales_linearly_with_magnitude(self):
        assert difference_resolution(magnitude=30.0) == pytest.approx(
            30.0 * difference_resolution(magnitude=1.0)
        )

    def test_larger_step_lowers_the_floor(self):
        coarse = difference_resolution(magnitude=1.0, step=1e-4)
        fine = difference_resolution(magnitude=1.0, step=1e-8)
        assert coarse < fine

    def test_the_chosen_tolerance_is_above_the_floor(self):
        """这一课的关键自洽性：容差必须**大于**最坏那个案例的分辨率.

        否则报告里会出现一条"不一致"，而两边都没错——
        那种假警报比"漏报"更坏，因为它会让人去改一份本来正确的代码。
        """
        worst_case_scale = max(max(item) for item in LOGIT_VECTORS)
        assert GRADIENT_TOLERANCE > difference_resolution(magnitude=worst_case_scale)

    def test_bad_parameters_are_rejected(self):
        with pytest.raises(NumericError, match="量级"):
            difference_resolution(magnitude=float("nan"))
        with pytest.raises(NumericError, match="步长"):
            difference_resolution(magnitude=1.0, step=0.0)


class TestAutogradCasesAreMeaningful:
    """自动微分对照用的表达式集合：必须包含"同一个输入走两条路径"的写法."""

    def test_cases_are_named_and_unique(self):
        names = [name for name, _expression, _point in AUTOGRAD_CASES]
        assert len(names) == len(set(names))

    def test_at_least_one_case_uses_an_input_twice(self):
        """``tanh(x)·relu(x+1)`` 里 ``x`` 走了两条边——漏了累加会在这里少算一半."""
        _name, expression, point = AUTOGRAD_CASES[2]
        from smart_research_agent.math_foundations.autograd import value_and_grad

        _value, grads = value_and_grad(expression, point)
        assert grads[0] != 0.0
        # 手算：d/dx[tanh(x)·(x+1)] = (1−tanh²x)(x+1) + tanh(x)
        x = point[0]
        expected = (1.0 - math.tanh(x) ** 2) * (x + 1.0) + math.tanh(x)
        assert approx(grads[0], expected, tolerance=1e-9)

    def test_points_have_the_declared_dimension(self):
        for _name, expression, point in AUTOGRAD_CASES:
            assert point
            from smart_research_agent.math_foundations.autograd import value_and_grad

            _value, grads = value_and_grad(expression, point)
            assert len(grads) == len(point)


class TestSamplesAreReproducible:
    """样本写死（随机输入会让"这次对上了"不可复现）."""

    def test_logit_vectors_and_targets_align(self):
        assert len(LOGIT_VECTORS) == len(TARGET_INDICES)
        for logits, target in zip(LOGIT_VECTORS, TARGET_INDICES):
            assert 0 <= target < len(logits)

    def test_perplexity_inputs_avoid_the_domain_boundary(self):
        """``0`` 被排除：中心差分需要左邻点，而 ``perplexity`` 的定义域是 ``[0, ∞)``."""
        assert all(value > 0.0 for value in PERPLEXITY_INPUTS)
        assert min(PERPLEXITY_INPUTS) > DEFAULT_STEP

    def test_scalar_points_include_a_saturated_region(self):
        assert 30.0 in SCALAR_POINTS and -30.0 in SCALAR_POINTS
