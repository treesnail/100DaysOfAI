"""六项梯度校验：解析反向 vs 中心差分（day078 / M7-D3）.

这一份测试守一件本课新增的事：

```text
day075 五项    w_output / w_value / w_query / w_key / inputs
day078 六项    上面五项 + **table**（位置表）——它是唯一"按位置累加"的一项
```

``table`` 那一项有一个字面上不报错的错法（把"按位置累加"写成"按行覆盖"）：
位置互不相同时它**恰好**给出同样的表梯度，位置重复时才露出差额。
因此这里专门用一条**位置重复**的样本跑同一套校验——它是那一处坑的唯一判据。
"""

from __future__ import annotations

import pytest

from smart_research_agent.positional_encoding import (
    GRAD_INPUTS,
    GRAD_TABLE,
    GRADIENT_TOLERANCE,
    OFFSET_LAW_TOLERANCE,
    OFFSET_TOLERANCE,
    POSITIONAL_GRADIENT_DESCRIPTIONS,
    POSITIONAL_GRADIENT_FORMULAS,
    POSITIONAL_GRADIENT_TARGETS,
    PositionalGradientOutcome,
    PositionalGradientReport,
    check_positional_gradients,
    learnable_table,
    loss_gradient,
    numerical_positional_gradients,
    positional_backward,
    positional_forward,
    scatter_add_rows,
)
from smart_research_agent.positional_encoding.errors import (
    GradientError,
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.positional_encoding.verify import (
    ROTATION_TOLERANCE,
    _compare,
    _scaled_error,
    numeric_params_resolve,
)
from smart_research_agent.transformer_core.types import (
    GRADIENT_TARGETS as CORE_GRADIENT_TARGETS,
)
from smart_research_agent.transformer_core.types import relative_matrix_error
from smart_research_agent.transformer_core.verify import (
    GRADIENT_TOLERANCE as CORE_TOLERANCE,
)
from tests.position_samples import (
    LENGTH,
    VOCABULARY,
    approx,
    position_parameters,
    readout_task,
    sinusoidal,
)

#: 一份固定的位置序列：**位置重复**——表那一项在这个样本上才真正被区分出来.
REPEATED_POSITIONS: tuple[int, ...] = (0, 1, 0, 1)


def _sample(*, table=None):
    """一份固定的（参数, 表, 输入, 目标）四元组（每一次调用都是同一批数）."""
    task = readout_task()
    resolved = sinusoidal(LENGTH, VOCABULARY) if table is None else table
    return position_parameters(VOCABULARY), resolved, task.inputs, task.target


class TestConstants:
    """两个容差常量与名字清单：与 day075 同值，但**各自另立一份**并有断言钉住."""

    def test_tolerance_matches_day075(self):
        assert GRADIENT_TOLERANCE == CORE_TOLERANCE

    def test_offset_and_rotation_tolerances(self):
        assert OFFSET_LAW_TOLERANCE == OFFSET_TOLERANCE == 1e-12
        assert ROTATION_TOLERANCE == 1e-12

    def test_tolerances_are_ordered(self):
        """梯度容差必须比"两条求和路径"的容差松**且**足够紧：它在 1e-12 与 1e-3 之间."""
        assert GRADIENT_TOLERANCE > ROTATION_TOLERANCE
        assert GRADIENT_TOLERANCE < 1e-3

    def test_six_targets_extend_the_core_five(self):
        """去掉 ``table`` 之后必须**逐字**等于 day075 的五项，且 ``table`` 排在 ``inputs`` 前."""
        without_table = tuple(name for name in POSITIONAL_GRADIENT_TARGETS if name != GRAD_TABLE)
        assert without_table == CORE_GRADIENT_TARGETS
        assert POSITIONAL_GRADIENT_TARGETS.index(GRAD_TABLE) == POSITIONAL_GRADIENT_TARGETS.index(
            GRAD_INPUTS
        ) - 1
        assert GRAD_INPUTS == CORE_GRADIENT_TARGETS[-1]

    def test_formulas_describe_the_two_new_steps(self):
        assert "Σ" in POSITIONAL_GRADIENT_FORMULAS[GRAD_TABLE]
        assert "恒等映射" in POSITIONAL_GRADIENT_FORMULAS[GRAD_INPUTS]
        assert "本课唯一新增" in POSITIONAL_GRADIENT_DESCRIPTIONS[GRAD_TABLE]


class TestNumericalGradients:
    """数值侧：只依赖前向与损失，不依赖任何推导."""

    def test_shapes_match_the_parameters(self):
        params, table, inputs, target = _sample()
        numeric = numerical_positional_gradients(params, table, inputs, target=target)
        for matrix, weight in zip(numeric.matrices()[:4], params.matrices(), strict=True):
            assert len(matrix) == len(weight)
            assert len(matrix[0]) == len(weight[0])
        assert len(numeric.grad_table) == LENGTH
        assert len(numeric.grad_inputs) == LENGTH

    def test_is_deterministic(self):
        params, table, inputs, target = _sample()
        first = numerical_positional_gradients(params, table, inputs, target=target)
        second = numerical_positional_gradients(params, table, inputs, target=target)
        assert first.flatten() == second.flatten()
        assert first.grad_inputs == second.grad_inputs

    def test_supervised_convention_is_honoured(self):
        """``supervised`` 换了损失口径（监督行 MSE），数值侧必须跟着换."""
        params, table, inputs, target = _sample()
        everything = numerical_positional_gradients(params, table, inputs, target=target)
        partial = numerical_positional_gradients(
            params, table, inputs, target=target, supervised=(0, 1)
        )
        gap = relative_matrix_error(everything.grad_w_query, partial.grad_w_query)
        # 远大于数值差分自身的分辨率（~1e-11），因此这是一个**真实的口径差**
        assert gap > 1e-6

    def test_positions_convention_is_honoured(self):
        """显式位置序列必须一路传到数值侧——否则两边算的是"不同的注入"."""
        params, table = position_parameters(VOCABULARY), sinusoidal(LENGTH, VOCABULARY)
        task = readout_task()
        repeated = numerical_positional_gradients(
            params, table, task.inputs, target=task.target, positions=REPEATED_POSITIONS
        )
        plain = numerical_positional_gradients(params, table, task.inputs, target=task.target)
        assert relative_matrix_error(repeated.grad_inputs, plain.grad_inputs) > 1e-6

    def test_target_shape_is_checked(self):
        params, table, inputs, _target = _sample()
        with pytest.raises(ShapeError):
            numerical_positional_gradients(params, table, inputs, target=((1.0, 0.0),))

    def test_resolve_rejects_a_wrong_block_count(self):
        """压平得对，但块数不是 5：这一步必须当场拒绝（否则"表落在哪一段"会错位）."""
        with pytest.raises(ShapeError, match="5 个"):
            numeric_params_resolve((1.0,), ((1, 1),))


class TestGradientCheck:
    """六项逐点对照：本层最硬的一条护栏."""

    def test_all_six_items_pass(self):
        params, table, inputs, target = _sample()
        report = check_positional_gradients(params, table, inputs, target=target)
        assert report.ok
        assert report.failures == ()
        assert len(report.outcomes) == len(POSITIONAL_GRADIENT_TARGETS) == 6

    def test_all_six_items_pass_with_repeated_positions(self):
        """**表的那一项**在位置重复时才被真正区分——按行覆盖的写法会在这里露出来."""
        params, table, inputs, target = _sample()
        report = check_positional_gradients(
            params, table, inputs, target=target, positions=REPEATED_POSITIONS
        )
        assert report.ok
        assert any("重复" in note for note in report.notes)

    def test_all_six_items_pass_on_a_learnable_table(self):
        params, table, inputs, target = _sample(
            table=learnable_table(LENGTH, VOCABULARY, initializer="random", seed=7)
        )
        assert check_positional_gradients(params, table, inputs, target=target).ok

    def test_compared_points_cover_every_entry(self):
        params, table, inputs, target = _sample()
        report = check_positional_gradients(params, table, inputs, target=target)
        points = {item.target: item.compared_points for item in report.outcomes}
        assert points[GRAD_TABLE] == LENGTH * VOCABULARY
        assert points[GRAD_INPUTS] == LENGTH * VOCABULARY
        assert points["w_query"] == VOCABULARY * VOCABULARY

    def test_resolution_is_far_below_the_tolerance(self):
        params, table, inputs, target = _sample()
        report = check_positional_gradients(params, table, inputs, target=target)
        assert report.worst_scaled_error < report.tolerance * 1e-3

    def test_tolerance_must_be_positive(self):
        params, table, inputs, target = _sample()
        with pytest.raises(ParameterError):
            check_positional_gradients(params, table, inputs, target=target, tolerance=0.0)

    def test_target_shape_is_checked(self):
        params, table, inputs, _target = _sample()
        with pytest.raises(ShapeError):
            check_positional_gradients(params, table, inputs, target=((1.0, 0.0),))

    def test_an_impossibly_tight_tolerance_fails_every_item(self):
        """把容差压到 1e-30：六项全部不通过，而**消息**必须指向"同一个函数"."""
        params, table, inputs, target = _sample()
        report = check_positional_gradients(
            params, table, inputs, target=target, tolerance=1e-30
        )
        assert not report.ok
        assert len(report.failures) == 6
        assert all("同一个函数" in item.message for item in report.failures)
        with pytest.raises(GradientError, match="未通过"):
            report.raise_if_failed()

    def test_a_passing_report_does_not_raise(self):
        params, table, inputs, target = _sample()
        check_positional_gradients(params, table, inputs, target=target).raise_if_failed()

    def test_report_serialises(self):
        params, table, inputs, target = _sample()
        payload = check_positional_gradients(params, table, inputs, target=target).to_dict()
        assert payload["ok"] is True
        assert payload["step"] > 0.0
        assert len(payload["outcomes"]) == 6
        assert payload["worst_scaled_error"] < GRADIENT_TOLERANCE

    def test_report_summary_line(self):
        params, table, inputs, target = _sample()
        line = check_positional_gradients(params, table, inputs, target=target).summary_line()
        assert "通过 6" in line
        assert "失败 0" in line


class TestOutcomeValidation:
    """单项结论的构造校验：不认识的项、负误差都要当场拒绝."""

    def _outcome(self, **overrides):
        payload = {
            "target": GRAD_TABLE,
            "max_absolute_error": 1e-11,
            "max_scaled_error": 1e-11,
            "tolerance": GRADIENT_TOLERANCE,
            "compared_points": 24,
            "message": "ok",
        }
        payload.update(overrides)
        return payload

    def test_unknown_target_is_rejected(self):
        with pytest.raises(ParameterError, match="未知的梯度校验项"):
            PositionalGradientOutcome(**self._outcome(target="w_nope"))

    @pytest.mark.parametrize("field", ["max_absolute_error", "max_scaled_error"])
    def test_negative_or_nan_error_is_rejected(self, field):
        with pytest.raises(NumericError):
            PositionalGradientOutcome(**self._outcome(**{field: -1.0}))
        with pytest.raises(NumericError):
            PositionalGradientOutcome(**self._outcome(**{field: float("nan")}))

    def test_compared_points_must_be_a_non_negative_integer(self):
        with pytest.raises(NumericError, match="compared_points"):
            PositionalGradientOutcome(**self._outcome(compared_points=-1))

    def test_tolerance_must_be_positive(self):
        with pytest.raises(ParameterError):
            PositionalGradientOutcome(**self._outcome(tolerance=0.0))

    def test_passed_formula_and_description(self):
        outcome = PositionalGradientOutcome(**self._outcome())
        assert outcome.passed
        assert "Σ" in outcome.formula
        assert "位置表" in outcome.description
        assert outcome.summary_line().startswith("[ok]")

    def test_failing_outcome_is_marked(self):
        outcome = PositionalGradientOutcome(
            **self._outcome(max_scaled_error=1e-2, message="超了")
        )
        assert not outcome.passed
        assert outcome.summary_line().startswith("[!!]")
        assert outcome.to_dict()["message"] == "超了"

    def test_to_dict_carries_the_formula(self):
        payload = PositionalGradientOutcome(**self._outcome()).to_dict()
        assert payload["target"] == GRAD_TABLE
        assert payload["formula"]
        assert payload["passed"] is True


class TestReportValidation:
    """汇总报告的构造校验：一份不完整的报告比没有报告更危险."""

    def _outcomes(self, *, scaled: float = 1e-11):
        return tuple(
            PositionalGradientOutcome(
                target=name,
                max_absolute_error=scaled,
                max_scaled_error=scaled,
                tolerance=GRADIENT_TOLERANCE,
                compared_points=4,
                message="x",
            )
            for name in POSITIONAL_GRADIENT_TARGETS
        )

    def test_empty_report_is_rejected(self):
        with pytest.raises(ParameterError, match="不能为空"):
            PositionalGradientReport(outcomes=())

    def test_missing_target_is_rejected(self):
        with pytest.raises(NumericError, match="不一致"):
            PositionalGradientReport(outcomes=self._outcomes()[:-1])

    def test_tolerance_must_be_positive(self):
        with pytest.raises(ParameterError):
            PositionalGradientReport(outcomes=self._outcomes(), tolerance=0.0)

    def test_notes_are_stringified(self):
        report = PositionalGradientReport(outcomes=self._outcomes(), notes=(1,))
        assert report.notes == ("1",)

    def test_failing_report_lists_the_failures(self):
        report = PositionalGradientReport(outcomes=self._outcomes(scaled=1.0))
        assert not report.ok
        assert len(report.failures) == 6
        assert approx(report.worst_scaled_error, 1.0)
        assert "通过 0、失败 6" in report.summary_line()


class TestComparisonHelpers:
    """对照用的小工具：逐点相对误差与两块梯度的比较."""

    def test_scaled_error_hand_computed(self):
        assert _scaled_error(1.0, 1.0) == 0.0
        assert approx(_scaled_error(1.0, 2.0), 0.5)

    @pytest.mark.parametrize("value", [float("inf"), float("nan")])
    def test_scaled_error_of_a_non_finite_value_is_infinite(self, value):
        """非有限数**不是**"差得很小"：它必须被标成无穷，否则它会静默通过."""
        assert _scaled_error(value, 1.0) == float("inf")

    def test_compare_counts_every_point(self):
        worst_absolute, worst_scaled, points = _compare(
            GRAD_TABLE, ((1.0, 2.0), (3.0, 4.0)), ((1.0, 2.0), (3.0, 4.0))
        )
        assert (worst_absolute, worst_scaled, points) == (0.0, 0.0, 4)

    def test_compare_checks_the_shape(self):
        with pytest.raises(ShapeError, match="形状不一致"):
            _compare(GRAD_TABLE, ((1.0,),), ((1.0, 2.0),))

    def test_backward_is_the_scatter_add_of_the_input_gradient(self):
        """结构守卫：``dTable`` 必须恰好是 ``scatter_add(dInjected)``（不是按行覆盖）."""
        params, table, inputs, target = _sample()
        forward = positional_forward(
            params, table, inputs, target=target, positions=REPEATED_POSITIONS
        )
        gradients = positional_backward(forward, loss_gradient(forward))
        assert gradients.grad_table == scatter_add_rows(
            gradients.grad_inputs,
            REPEATED_POSITIONS,
            table_positions=table.positions,
        )
