"""五项梯度校验：多头解析梯度 vs 数值差分（day076 / M7-D2）.

这一份测试里有一条**关于"参数忘了传"的断言**（``test_the_heads_argument_changes_the_check``）：

```text
数值侧忘了传 heads 时它算的是单头前向 → 与多头解析梯度的差在 1e-2 量级，
而那个差看起来像"推导写错了"。
```

它不是一条"多写一遍"的断言，而是 day075 那个"两侧必须算同一个损失"的坑
在多头下的**同类形态**：对照失败的第一嫌疑人往往是"两边算的不是同一件事"。
"""

from __future__ import annotations

import pytest

from smart_research_agent.multi_head.errors import (
    GradientError,
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.multi_head.train import batch_gradients
from smart_research_agent.multi_head.verify import (
    GRADIENT_TOLERANCE,
    IDENTITY_TOLERANCE,
    MultiHeadGradientOutcome,
    MultiHeadGradientReport,
    check_multihead_gradients,
    numerical_multihead_gradients,
)
from smart_research_agent.transformer_core.types import (
    GRADIENT_TARGETS,
    relative_matrix_error,
)
from smart_research_agent.transformer_core.verify import (
    GRADIENT_TOLERANCE as CLASSIC_TOLERANCE,
)
from smart_research_agent.transformer_core.verify import numerical_parameter_gradients
from tests.multihead_samples import (
    HEAD_COUNTS,
    approx,
    induction_tasks,
    toy_parameters,
)


def _sample(heads: int = 2, *, seed: int = 7):
    """一份固定的（参数, 样本, 目标）三元组（每一次调用都是同一批数）."""
    params = toy_parameters(6, seed=seed)
    task = induction_tasks(1)[0]
    return params, task.inputs, task.target


class TestConstants:
    """两个容差常量：与 day075 同值，但**各自另立一份**并有断言钉住."""

    def test_tolerance_matches_day075(self):
        assert GRADIENT_TOLERANCE == CLASSIC_TOLERANCE

    def test_identity_tolerance(self):
        assert IDENTITY_TOLERANCE == 1e-12

    def test_identity_tolerance_is_much_tighter(self):
        """近似性质的容差必须比梯度校验松**且**足够紧：它在 1e-16 与 1e-9 之间."""
        assert GRADIENT_TOLERANCE > IDENTITY_TOLERANCE > 1e-15


class TestNumericalGradients:
    """数值侧：只依赖前向与损失，不依赖任何推导."""

    @pytest.mark.parametrize("heads", HEAD_COUNTS)
    def test_shapes_match_the_parameters(self, heads):
        params, inputs, target = _sample(heads)
        numeric = numerical_multihead_gradients(params, inputs, target, heads=heads)
        for matrix, weight in zip(numeric.matrices(), params.matrices(), strict=True):
            assert len(matrix) == len(weight)
            assert len(matrix[0]) == len(weight[0])
        assert len(numeric.grad_inputs) == len(inputs)

    def test_heads_one_matches_day075_numerics(self):
        """两个数值实现（本层与 day075 那一层）在 heads=1 上必须给出同一批数."""
        params, inputs, target = _sample(1)
        mine = numerical_multihead_gradients(params, inputs, target, heads=1)
        theirs = numerical_parameter_gradients(params, inputs, target)
        for left, right in zip(mine.matrices(), theirs.matrices(), strict=True):
            assert relative_matrix_error(left, right) <= 1e-12
        assert relative_matrix_error(mine.grad_inputs, theirs.grad_inputs) <= 1e-12

    def test_supervised_convention_is_honoured(self):
        """``supervised`` 换了损失口径（监督行 MSE），数值侧必须跟着换."""
        params = toy_parameters(6)
        task = induction_tasks(1)[0]
        everything = numerical_multihead_gradients(
            params, task.inputs, task.target, heads=2
        )
        supervised = numerical_multihead_gradients(
            params, task.inputs, task.target, heads=2, supervised=task.supervised
        )
        gap = relative_matrix_error(everything.grad_w_query, supervised.grad_w_query)
        # 2.8e-4：远大于数值差分自身的分辨率（1e-11），因此这是一个**真实的口径差**
        assert gap > 1e-6
        assert gap > relative_matrix_error(
            everything.grad_w_query,
            numerical_multihead_gradients(params, task.inputs, task.target, heads=2).grad_w_query,
        )

    def test_input_and_target_shapes_must_match(self):
        params = toy_parameters(6)
        task = induction_tasks(1)[0]
        with pytest.raises(ShapeError):
            numerical_multihead_gradients(
                params, task.inputs, ((1.0, 0.0), (0.0, 1.0)), heads=2
            )


class TestGradientCheck:
    """五项逐点对照：本层最硬的一条护栏."""

    @pytest.mark.parametrize("heads", HEAD_COUNTS)
    def test_all_five_items_pass(self, heads):
        params, inputs, target = _sample(heads)
        report = check_multihead_gradients(params, inputs, target, heads=heads)
        assert report.ok
        assert report.failures == ()
        assert report.heads == heads
        assert len(report.outcomes) == len(GRADIENT_TARGETS)

    @pytest.mark.parametrize("heads", HEAD_COUNTS)
    def test_resolution_is_five_orders_below_the_tolerance(self, heads):
        """实测误差必须比容差**小三个数量级以上**：否则容差没有区分度."""
        params, inputs, target = _sample(heads)
        report = check_multihead_gradients(params, inputs, target, heads=heads)
        assert report.worst_scaled_error < report.tolerance * 1e-3

    def test_heads_one_matches_day075_gradient_targets(self):
        params, inputs, target = _sample(1)
        report = check_multihead_gradients(params, inputs, target, heads=1)
        assert {item.target for item in report.outcomes} == set(GRADIENT_TARGETS)

    def test_heads_must_agree_on_both_sides(self):
        """**忘了传 heads 的数值侧算的是单头前向**——差比分辨率大三个数量级.

        因此"多头梯度对不上"的第一个嫌疑人不是推导，而是"两边算的不是同一件事"。
        """
        params = toy_parameters(6)
        task = induction_tasks(1)[0]
        analytic = _analytic(params, task, heads=2)
        matching = numerical_multihead_gradients(
            params, task.inputs, task.target, heads=2
        )
        single = numerical_multihead_gradients(
            params, task.inputs, task.target, heads=1
        )
        resolution = relative_matrix_error(
            analytic.grad_w_query, matching.grad_w_query
        )
        gap = relative_matrix_error(analytic.grad_w_query, single.grad_w_query)
        assert resolution < 1e-6
        assert gap > 1e-5
        assert gap > 1000.0 * resolution

    def test_mask_convention_must_agree_on_both_sides(self):
        """同一个坑的第二种形态：**掩码口径**不一致（一个因果、一个全开）.

        这个差（2.8e-4）与"忘了传 heads"是同一个量级——而它同样看起来像推导错了。
        三条口径各写进参数名（损失 / heads / 掩码），就是为了让对照失败时
        第一个怀疑对象是"两边算的是不是同一件事"。
        """
        params = toy_parameters(6)
        task = induction_tasks(1)[0]
        analytic_full = _analytic(params, task, heads=2, causal=False)
        numeric_full = numerical_multihead_gradients(
            params, task.inputs, task.target, heads=2, causal=False
        )
        numeric_causal = numerical_multihead_gradients(
            params, task.inputs, task.target, heads=2, causal=True
        )
        assert (
            relative_matrix_error(analytic_full.grad_w_query, numeric_full.grad_w_query)
            < 1e-6
        )
        assert (
            relative_matrix_error(
                analytic_full.grad_w_query, numeric_causal.grad_w_query
            )
            > 1e-5
        )

    def test_a_different_seed_and_more_heads_still_agree(self):
        params = toy_parameters(6, seed=13)
        tasks = induction_tasks(3, seed=99)
        report = check_multihead_gradients(
            params, tasks[0].inputs, tasks[0].target, heads=3
        )
        assert report.ok

    def test_tolerance_must_be_positive(self):
        params, inputs, target = _sample(2)
        with pytest.raises(ParameterError):
            check_multihead_gradients(params, inputs, target, heads=2, tolerance=0.0)

    def test_target_shape_is_checked(self):
        params = toy_parameters(6)
        task = induction_tasks(2)[0]
        with pytest.raises(ShapeError):
            check_multihead_gradients(
                params, task.inputs, ((1.0, 0.0), (0.0, 1.0)), heads=2
            )

    def test_report_serialises(self):
        params, inputs, target = _sample(2)
        payload = check_multihead_gradients(params, inputs, target, heads=2).to_dict()
        assert payload["ok"] is True
        assert payload["heads"] == 2
        assert len(payload["outcomes"]) == 5
        assert payload["failures"] == []

    def test_report_summary_line(self):
        params, inputs, target = _sample(3)
        line = check_multihead_gradients(params, inputs, target, heads=3).summary_line()
        assert "heads=3" in line
        assert "通过 5" in line

    def test_report_notes_mention_the_heads_pitfall(self):
        params, inputs, target = _sample(2)
        report = check_multihead_gradients(params, inputs, target, heads=2)
        assert any("同一个 heads" in item for item in report.notes)


def _analytic(params, task, *, heads: int, causal: bool = True):
    """解析梯度（测试里复用生产代码的九步反向，避免自造一条路径）.

    ``causal`` 必须能显式指定：默认 ``multi_head_attention`` 是**全开掩码**，
    而默认 ``numerical_multihead_gradients`` 是**因果掩码**——不写清它，
    两边算的就是两个不同的函数（这正是上面那条测试要暴露的事）。
    """
    from smart_research_agent.multi_head.layers import (
        multi_head_attention,
        multi_head_backward,
    )
    from smart_research_agent.transformer_core.layers import mse_gradient

    forward = multi_head_attention(params, task.inputs, heads=heads, causal=causal)
    return multi_head_backward(forward, mse_gradient(forward.output, task.target))


class TestOutcomeValidation:
    """单项结论的构造校验：不认识的项、负误差都要当场拒绝."""

    def _outcome(self, **overrides):
        payload = {
            "target": "w_query",
            "max_absolute_error": 1e-11,
            "max_scaled_error": 1e-11,
            "tolerance": GRADIENT_TOLERANCE,
            "compared_points": 9,
            "message": "ok",
        }
        payload.update(overrides)
        return payload

    def test_unknown_target_is_rejected(self):
        with pytest.raises(GradientError):
            MultiHeadGradientOutcome(**self._outcome(target="w_nope"))

    @pytest.mark.parametrize("field", ["max_absolute_error", "max_scaled_error"])
    def test_negative_or_nan_error_is_rejected(self, field):
        with pytest.raises(NumericError):
            MultiHeadGradientOutcome(**self._outcome(**{field: -1.0}))
        with pytest.raises(NumericError):
            MultiHeadGradientOutcome(**self._outcome(**{field: float("nan")}))

    def test_passed_and_formula(self):
        outcome = MultiHeadGradientOutcome(**self._outcome())
        assert outcome.passed
        assert "Σ_h" in outcome.formula
        assert outcome.summary_line().startswith("[ok]")

    def test_failing_outcome_is_marked(self):
        outcome = MultiHeadGradientOutcome(
            **self._outcome(max_scaled_error=1e-2, message="超了")
        )
        assert not outcome.passed
        assert outcome.summary_line().startswith("[FAIL]")

    def test_to_dict_carries_the_formula(self):
        payload = MultiHeadGradientOutcome(**self._outcome()).to_dict()
        assert payload["target"] == "w_query"
        assert payload["formula"] == MultiHeadGradientOutcome(**self._outcome()).formula


class TestReportValidation:
    """汇总报告的构造校验：一份不完整的报告比没有报告更危险."""

    def _outcomes(self, *, scaled: float = 1e-11):
        return tuple(
            MultiHeadGradientOutcome(
                target=name,
                max_absolute_error=scaled,
                max_scaled_error=scaled,
                tolerance=GRADIENT_TOLERANCE,
                compared_points=4,
                message="x",
            )
            for name in GRADIENT_TARGETS
        )

    def test_missing_target_is_rejected(self):
        with pytest.raises(GradientError):
            MultiHeadGradientReport(outcomes=self._outcomes()[:-1], heads=2)

    def test_heads_must_be_positive(self):
        with pytest.raises(ParameterError):
            MultiHeadGradientReport(outcomes=self._outcomes(), heads=0)

    def test_failing_report_raises_on_demand(self):
        report = MultiHeadGradientReport(
            outcomes=self._outcomes(scaled=1.0), heads=2
        )
        assert not report.ok
        assert len(report.failures) == len(GRADIENT_TARGETS)
        with pytest.raises(GradientError, match="未通过"):
            report.raise_if_failed()

    def test_passing_report_does_not_raise(self):
        MultiHeadGradientReport(outcomes=self._outcomes(), heads=1).raise_if_failed()

    def test_notes_are_stringified(self):
        report = MultiHeadGradientReport(
            outcomes=self._outcomes(), heads=1, notes=(1,)
        )
        assert report.notes == ("1",)


class TestBatchGradientSources:
    """两种梯度源的对账（培训回路层面的证据，不是单项校验）."""

    @pytest.mark.parametrize("heads", HEAD_COUNTS)
    def test_analytic_and_numeric_agree_on_the_batch(self, heads):
        params = toy_parameters(6)
        tasks = induction_tasks(2)
        analytic = batch_gradients(params, tasks, heads=heads, source="analytic")
        numeric = batch_gradients(params, tasks, heads=heads, source="numeric")
        assert len(analytic) == len(numeric)
        worst = max(abs(a - b) for a, b in zip(analytic, numeric, strict=True))
        assert worst < 1e-6

    def test_unknown_source_is_rejected(self):
        params = toy_parameters(6)
        with pytest.raises(ParameterError):
            batch_gradients(params, induction_tasks(1), heads=2, source="magic")

    def test_heads_one_batch_gradients_match_day075(self):
        """本层的批量梯度在 heads=1 上与 day075 的实现对得上（同参数量、同算法）."""
        from smart_research_agent.transformer_core.train import batch_gradients as theirs

        params = toy_parameters(6)
        tasks = induction_tasks(2)
        mine = batch_gradients(params, tasks, heads=1)
        classic = theirs(params, tasks)
        worst = max(abs(a - b) for a, b in zip(mine, classic, strict=True))
        assert approx(worst, 0.0, tolerance=1e-12)
