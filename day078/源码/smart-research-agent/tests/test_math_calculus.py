"""``math_foundations.calculus``：差分、梯度、链式法则与步长实验（day074）.

断言里的期望值全部是**手算**的（或来自数学上可推导的界）：

```text
d(exp)/dx @1 = e                  d(sin)/dx @1 = cos 1
d(x³)/dx @2 = 12                  d(1/x)/dx @4 = −1/16
∂f/∂x = 2(x−1)                    ∂f/∂y = 8(y+2)
```

步长实验里的两条断言值得单独看：**前向差分的实测阶数 ≈ 1、中心差分 ≈ 2**，
以及"h 小到 1e-14 之后误差反而上升"——它们是这一课"量出来"的两个结论。
"""

from __future__ import annotations

import math

import pytest

from smart_research_agent.math_foundations.calculus import (
    DEFAULT_STEP,
    DEFAULT_STEP_SIZES,
    backward_difference,
    best_step_for_central,
    central_difference,
    chain_rule,
    compose,
    derivative,
    directional_derivative,
    forward_difference,
    gradient,
    is_close,
    jacobian,
    linear_approximation,
    second_difference,
    step_size_study,
    taylor_accuracy,
)
from smart_research_agent.math_foundations.errors import (
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.math_foundations.types import BACKWARD, CENTRAL, FLOAT_EPSILON, FORWARD
from tests.math_samples import CALCULUS_CASES, approx


class TestSingleVariableDifferences:
    """三种差分方法在五个手算案例上的表现."""

    @pytest.mark.parametrize(("name", "function", "point", "exact"), CALCULUS_CASES)
    def test_central_difference_hits_hand_computed_derivative(self, name, function, point, exact):
        # 中心差分在 h = 1e-6 上误差约 1e-9（截断 O(h²) + 舍入），因此用 1e-6 的容差
        assert approx(central_difference(function, point), exact, tolerance=1e-6)

    @pytest.mark.parametrize(("name", "function", "point", "exact"), CALCULUS_CASES)
    def test_forward_difference_is_less_accurate_than_central(
        self, name, function, point, exact
    ):
        forward_error = abs(forward_difference(function, point) - exact)
        central_error = abs(central_difference(function, point) - exact)
        # 前向是 O(h)、中心是 O(h²)：在同一个 h 上中心的误差应当小几个数量级
        assert central_error <= forward_error + 1e-12

    @pytest.mark.parametrize(("name", "function", "point", "exact"), CALCULUS_CASES)
    def test_backward_difference_matches_too(self, name, function, point, exact):
        assert approx(backward_difference(function, point), exact, tolerance=1e-5)

    def test_forward_and_backward_bracket_the_derivative(self):
        """前向、后向差分分别在真值两侧（对凸函数成立）.

        ``x²`` 的导数在 ``0`` 处是 0；在 ``0`` 的两侧，前向差分给出 ``h``、
        后向差分给出 ``−h``——两者一正一负，真值在中间。
        """
        assert forward_difference(lambda value: value**2, 0.0) == pytest.approx(DEFAULT_STEP)
        assert backward_difference(lambda value: value**2, 0.0) == pytest.approx(-DEFAULT_STEP)

    def test_backward_difference_works_at_the_right_edge(self):
        """``√(1−x)`` 在 ``x = 1`` 是定义域右端点：前向差分越界，后向差分可用.

        这是"单侧导数"的实际含义：端点处只有一侧的函数值存在，
        而"哪一侧存在"决定了该用哪一种差分。中心差分在这里也不行
        （它要同时取 ``f(1+h)`` 与 ``f(1−h)``）。
        """
        root = lambda value: math.sqrt(1.0 - value)  # noqa: E731
        with pytest.raises(ValueError):
            forward_difference(root, 1.0, step=1e-3)
        with pytest.raises(ValueError):
            central_difference(root, 1.0, step=1e-3)
        value = backward_difference(root, 1.0, step=1e-3)
        # 单侧导数在端点处趋于 −∞，因此只断言"有限且为负"（不硬编码一个数）
        assert math.isfinite(value)
        assert value < 0.0

    def test_second_difference_of_cubic(self):
        """``x³`` 的二阶导数是 ``6x``，在 2 处是 12.

        步长取 ``1e-4`` 而不是缺省的 ``1e-6``：二阶差分的分母是 ``h²``，
        舍入误差被放大 ``1/h²`` 倍——``h = 1e-6`` 时噪声地板是 ``7e-3``，
        比真值 12 的千分之一还大。这是"误差量级知识直接用在了选参数上"。
        """
        assert approx(
            second_difference(lambda value: value**3, 2.0, step=1e-4), 12.0, tolerance=1e-4
        )

    def test_second_difference_of_quadratic_is_constant(self):
        """二次函数的二阶导数是常数（与取值点无关）——这也是"曲率"的定义."""
        for point in (-3.0, 0.0, 2.5):
            assert approx(
                second_difference(
                    lambda value: 5.0 * value**2 + 3.0 * value, point, step=1e-4
                ),
                10.0,
                tolerance=1e-4,
            )


class TestDerivativeDispatch:
    """``derivative`` 的分发与参数校验."""

    def test_default_method_is_central(self):
        assert derivative(math.exp, 1.0) == central_difference(math.exp, 1.0)

    def test_dispatch_by_name(self):
        assert derivative(math.exp, 1.0, method=FORWARD) == forward_difference(math.exp, 1.0)
        assert derivative(math.exp, 1.0, method=BACKWARD) == backward_difference(math.exp, 1.0)

    def test_unknown_method_is_rejected(self):
        with pytest.raises(ParameterError, match="不认识的差分方法"):
            derivative(math.exp, 1.0, method="spectral")

    @pytest.mark.parametrize("bad_step", [0.0, -1e-6, float("inf"), float("nan")])
    def test_step_must_be_positive_and_finite(self, bad_step):
        with pytest.raises(ParameterError, match="步长"):
            central_difference(math.exp, 1.0, step=bad_step)

    @pytest.mark.parametrize("bad_point", [float("nan"), float("inf"), float("-inf")])
    def test_point_must_be_finite(self, bad_point):
        with pytest.raises(NumericError, match="有限实数"):
            central_difference(math.exp, bad_point)

    def test_non_finite_function_value_is_rejected(self):
        """函数在扰动点上返回非有限数时当场报错，而不是得到一个"极大的导数".

        用乘法溢出构造 ``inf``（``1e308 * 10`` 在 Python 里**不抛异常**，
        只是安静地变成 ``inf``）——这类"安静的非有限数"正是要拦的东西：
        ``(inf − inf) / h`` 会给出 ``nan``，而 ``nan`` 看起来像"这一步算坏了"。
        """
        with pytest.raises(NumericError, match="取值不是有限实数"):
            central_difference(lambda value: 1e308 * value, 10.0)


class TestGradient:
    """向量函数的梯度：与手算的偏导数逐一对照."""

    def test_gradient_of_bowl(self):
        """``f = (x−1)² + 4(y+2)²`` 在 (3, 3) 处的梯度是 (4, 40)."""
        result = gradient(
            lambda params: (params[0] - 1.0) ** 2 + 4.0 * (params[1] + 2.0) ** 2,
            (3.0, 3.0),
        )
        assert approx(result[0], 4.0, tolerance=1e-6)
        assert approx(result[1], 40.0, tolerance=1e-6)

    def test_gradient_has_the_same_shape_as_the_point(self):
        result = gradient(lambda params: sum(value * value for value in params), (1.0, 2.0, 3.0))
        assert len(result) == 3

    def test_gradient_at_the_minimum_is_zero(self):
        result = gradient(
            lambda params: (params[0] - 1.0) ** 2 + 4.0 * (params[1] + 2.0) ** 2,
            (1.0, -2.0),
        )
        assert approx(result[0], 0.0, tolerance=1e-6)
        assert approx(result[1], 0.0, tolerance=1e-6)

    def test_gradient_accepts_other_methods(self):
        forward = gradient(lambda params: params[0] ** 2, (3.0,), method=FORWARD)
        backward = gradient(lambda params: params[0] ** 2, (3.0,), method=BACKWARD)
        assert approx(forward[0], 6.0, tolerance=1e-5)
        assert approx(backward[0], 6.0, tolerance=1e-5)

    def test_gradient_rejects_unknown_method_and_empty_point(self):
        with pytest.raises(ParameterError, match="不认识的差分方法"):
            gradient(lambda params: params[0], (1.0,), method="magic")
        with pytest.raises(ShapeError, match="不能为空"):
            gradient(lambda params: 1.0, ())

    def test_gradient_step_is_validated(self):
        with pytest.raises(ParameterError, match="步长"):
            gradient(lambda params: params[0], (1.0,), step=0.0)


class TestDirectionalDerivative:
    """方向导数与"梯度是最陡方向"这条可验证的结论."""

    def test_along_an_axis(self):
        """沿 x 轴的方向导数就是 ∂f/∂x（在 (3,3) 处是 4）."""
        function = lambda params: (params[0] - 1.0) ** 2 + 4.0 * (params[1] + 2.0) ** 2  # noqa: E731
        assert approx(
            directional_derivative(function, (3.0, 3.0), (1.0, 0.0)), 4.0, tolerance=1e-6
        )

    def test_direction_is_normalized(self):
        """``(2, 0)`` 与 ``(1, 0)`` 是同一个方向，方向导数必须相同."""
        function = lambda params: params[0] ** 2 + params[1] ** 2  # noqa: E731
        long = directional_derivative(function, (1.0, 0.0), (2.0, 0.0))
        short = directional_derivative(function, (1.0, 0.0), (1.0, 0.0))
        assert approx(long, short, tolerance=1e-9)
        assert approx(long, 2.0, tolerance=1e-6)

    def test_steepest_direction_reaches_the_gradient_norm(self):
        """在单位方向里扫一圈，最大值恰好在 ``∇f/‖∇f‖`` 处、且等于 ``‖∇f‖``.

        这是柯西–施瓦茨不等式的一个可测推论，也是"梯度下降为什么沿梯度走"
        这句话的**全部数学内容**。扫描用的是确定性的角度序列（可复现）。
        """
        function = lambda params: 3.0 * params[0] ** 2 + params[1] ** 2 + params[0] * params[1]  # noqa: E731
        point = (0.7, -0.4)
        grads = gradient(function, point)
        norm = math.sqrt(math.fsum(value * value for value in grads))
        best = grads[0] / norm, grads[1] / norm
        best_value = directional_derivative(function, point, best)
        assert approx(best_value, norm, tolerance=1e-5)
        for index in range(36):
            angle = index * math.pi / 18.0
            candidate = (math.cos(angle), math.sin(angle))
            assert directional_derivative(function, point, candidate) <= best_value + 1e-6

    def test_zero_direction_is_rejected(self):
        with pytest.raises(NumericError, match="零向量"):
            directional_derivative(lambda params: params[0], (0.0, 0.0), (0.0, 0.0))

    def test_dimension_mismatch_is_rejected(self):
        with pytest.raises(ShapeError, match="维"):
            directional_derivative(lambda params: params[0], (0.0,), (1.0, 0.0))


class TestJacobian:
    """雅可比矩阵：形状与两个可手算的例子."""

    def test_linear_map_gives_the_matrix_back(self):
        """线性映射 ``A x`` 的雅可比就是 ``A``（差分在这里必然精确）."""
        matrix = ((2.0, 0.0), (0.0, 3.0))
        result = jacobian(
            lambda params: (
                matrix[0][0] * params[0] + matrix[0][1] * params[1],
                matrix[1][0] * params[0] + matrix[1][1] * params[1],
            ),
            (1.0, 1.0),
        )
        assert result[0][0] == pytest.approx(2.0, abs=1e-6)
        assert result[0][1] == pytest.approx(0.0, abs=1e-6)
        assert result[1][0] == pytest.approx(0.0, abs=1e-6)
        assert result[1][1] == pytest.approx(3.0, abs=1e-6)

    def test_shape_is_outputs_by_inputs(self):
        result = jacobian(
            lambda params: (params[0] + params[1], params[0] - params[1], params[0] * params[1]),
            (1.0, 2.0),
        )
        assert len(result) == 3
        assert all(len(row) == 2 for row in result)

    def test_softmax_jacobian_rows_sum_to_zero(self):
        """softmax 的雅可比每一行之和为 0（因为输出之和恒为 1）.

        这条性质是"softmax 的输出永远在同一张概率单纯形上"的微分形式：
        任何方向的扰动都不能改变"和为 1"，因此每一行的导数和必然为 0。
        """
        from smart_research_agent.math_foundations.linalg import softmax

        result = jacobian(lambda params: softmax(params), (0.5, -0.25, 2.0))
        for row in result:
            assert approx(math.fsum(row), 0.0, tolerance=1e-8)

    def test_jacobian_of_scalar_function_is_a_single_row(self):
        result = jacobian(lambda params: (params[0] * params[0] + params[1],), (2.0, 0.0))
        assert len(result) == 1
        assert approx(result[0][0], 4.0, tolerance=1e-6)


class TestStepSizeStudy:
    """步长实验：两个实测结论（误差阶数与误差触底）."""

    def _study(self, function=math.exp, point=1.0, exact=math.e):
        return step_size_study(function, point, exact)

    def test_forward_order_is_about_one(self):
        assert self._study().observed_order(FORWARD) == pytest.approx(1.0, abs=0.15)

    def test_backward_order_is_about_one(self):
        assert self._study().observed_order(BACKWARD) == pytest.approx(1.0, abs=0.15)

    def test_central_order_is_about_two(self):
        assert self._study().observed_order(CENTRAL) == pytest.approx(2.0, abs=0.15)

    def test_observed_order_rejects_out_of_range_index(self):
        study = self._study()
        with pytest.raises(ParameterError, match="相邻两条"):
            study.observed_order(CENTRAL, index=len(study.records) - 1)
        with pytest.raises(ParameterError, match="相邻两条"):
            study.observed_order(CENTRAL, index=-1)

    def test_observed_order_rejects_unknown_method(self):
        with pytest.raises(ParameterError, match="不认识的差分方法"):
            self._study().observed_order("taylor")

    def test_zero_error_record_breaks_the_order(self):
        """误差恰好为 0 时阶数没有定义（比值取对数）.

        用 ``f(x) = x``（导数恒为 1）与 ``0.5 / 0.25`` 两个步长：
        这两步的减法与除法在浮点里**精确**，因此误差恰好是 ``0.0``。
        （用 ``0.1`` 这类步长会得到 ``1.0000000000000009``——
        误差不是 0，而是 9e-16，于是这条断言会"看起来没生效"。）
        """
        study = step_size_study(lambda value: value, 1.0, 1.0, steps=(0.5, 0.25))
        with pytest.raises(NumericError, match="没有定义"):
            study.observed_order(FORWARD)

    def test_best_step_is_in_the_micron_range(self):
        """中心差分的实测最优步长落在 1e-5 ~ 1e-7 这个区间（理论 ``eps^{1/3} ≈ 6e-6``）."""
        best = self._study().best(CENTRAL)
        assert 1e-7 <= best.step <= 1e-5

    def test_error_rises_again_for_absurdly_small_step(self):
        """``h = 1e-14`` 时误差比 ``h = 1e-5`` 大几个数量级（舍入误差主导）."""
        study = self._study()
        records = {record.step: record for record in study.records}
        assert records[1e-14].central_error > 100.0 * records[1e-5].central_error

    def test_scaled_columns_are_constant_in_the_truncation_region(self):
        """``误差/h``（前向）与 ``误差/h²``（中心）在截断误差主导区大致是常数."""
        records = {record.step: record for record in self._study().records}
        first = records[1e-1].forward_scaled
        second = records[1e-2].forward_scaled
        assert first == pytest.approx(second, rel=0.2)
        central_first = records[1e-1].central_scaled
        central_second = records[1e-2].central_scaled
        assert central_first == pytest.approx(central_second, rel=0.2)

    def test_records_cover_every_requested_step(self):
        study = self._study()
        assert len(study.records) == len(DEFAULT_STEP_SIZES)
        assert [record.step for record in study.records] == list(DEFAULT_STEP_SIZES)

    def test_empty_steps_is_rejected(self):
        with pytest.raises(ParameterError, match="步长序列不能为空"):
            step_size_study(math.exp, 1.0, math.e, steps=())

    def test_step_size_study_rejects_non_finite_exact_value(self):
        with pytest.raises(NumericError, match="精确导数值"):
            step_size_study(math.exp, 1.0, float("nan"))

    def test_report_shapes_are_json_friendly(self):
        import json

        study = self._study()
        payload = study.to_dict()
        json.dumps(payload)
        assert payload["observed_order_central"] == pytest.approx(2.0, abs=0.15)
        assert payload["best_central_step"] > 0
        assert len(payload["records"]) == len(DEFAULT_STEP_SIZES)
        assert payload["records"][0]["forward_scaled"] > 0
        assert study.notes
        assert "实测阶数" in study.summary_line()
        assert study.summary_lines()[0].startswith("h=1e-01")

    def test_notes_explain_both_error_sources(self):
        notes = " ".join(self._study().notes)
        assert "截断误差" in notes
        assert "舍入误差" in notes


class TestBestStep:
    """理论最优步长：``(eps/|f''|)^{1/3}`` 与"直线函数"的兜底."""

    def test_curved_function_gives_the_eps_cuberoot_scale(self):
        """``exp`` 在 1 处曲率为 e，因此最优步长是 ``(eps/e)^{1/3} ≈ 4.34e-6``.

        期望值是**手算**的（不抄实现的返回值）：``(2.22e-16 / 2.718)^{1/3}``。
        容差 1e-5 而不是 1e-9：曲率是**量出来**的（``second_difference``），
        它自身带 1e-6 量级的相对误差，而开三次方把它压缩到 1e-7 量级——
        这正是"实测值与理论值差多少"的合理期望。
        """
        expected = (FLOAT_EPSILON / math.e) ** (1.0 / 3.0)
        assert best_step_for_central(math.exp, 1.0) == pytest.approx(expected, rel=1e-5)

    def test_flat_function_falls_back_to_eps_cuberoot(self):
        """直线函数的真曲率为 0：返回 ``eps^{1/3} ≈ 6.06e-6``.

        这一条同时是"噪声地板"那个设计的证明：量出来的 ``|f''|`` 是 1.8e-3
        （全是舍入噪声），如果拿它去算，会得到一个约 5e-5 的步长——
        比正确值大 10 倍，而且完全由浮点噪声决定。
        """
        assert best_step_for_central(lambda value: 3.0 * value + 1.0, 2.0) == pytest.approx(
            FLOAT_EPSILON ** (1.0 / 3.0), rel=1e-9
        )

    def test_measured_curvature_of_a_line_is_pure_noise(self):
        """把"噪声"这件事直接断言下来：直线的二阶差分给出 1e-3 量级的假曲率."""
        measured = abs(second_difference(lambda value: 3.0 * value + 1.0, 2.0))
        assert measured > 1e-6  # 真值是 0，量出来却不是
        assert measured < 1e-2  # 但它仍然落在噪声地板的量级内

    def test_non_finite_point_is_rejected(self):
        with pytest.raises(NumericError, match="有限实数"):
            best_step_for_central(math.exp, float("inf"))


class TestComposeAndChainRule:
    """复合的方向（右到左）与链式法则的两条路."""

    def test_compose_applies_from_the_right(self):
        """``compose(f, g)(x) == f(g(x))``——与数学上的 ``f ∘ g`` 同一个方向."""
        assert compose(lambda value: value + 1.0, lambda value: value * 2.0)(3.0) == 7.0

    def test_compose_with_one_function_is_identity(self):
        function = lambda value: value * 5.0  # noqa: E731
        assert compose(function) is function

    def test_empty_compose_is_rejected(self):
        with pytest.raises(ParameterError, match="至少要有一个函数"):
            compose()

    def test_chain_rule_matches_the_product_of_locals(self):
        """``(x+1)²`` 在 3 处：整体导数 8，两段局部导数 2 与 4 之积也是 8."""
        overall, product = chain_rule([lambda value: value**2, lambda value: value + 1.0], 3.0)
        assert approx(overall, 8.0, tolerance=1e-6)
        assert approx(product, 8.0, tolerance=1e-6)

    def test_three_layer_chain(self):
        """``sin(exp(2x))`` 在 0 处：导数 ``2·cos(1)·e⁰ = 2cos 1``."""
        functions = [math.sin, math.exp, lambda value: 2.0 * value]
        overall, product = chain_rule(functions, 0.0)
        expected = 2.0 * math.cos(1.0)
        assert approx(overall, expected, tolerance=1e-5)
        assert approx(product, expected, tolerance=1e-5)

    def test_empty_chain_is_rejected(self):
        with pytest.raises(ParameterError, match="至少要有一个函数"):
            chain_rule([], 1.0)


class TestLinearisation:
    """一阶泰勒近似与它的误差（二阶项的量级）."""

    def test_at_the_expansion_point_it_is_exact(self):
        assert approx(linear_approximation(math.exp, 1.0, 1.0), math.e, tolerance=1e-9)

    def test_approximation_error_grows_with_distance(self):
        near = abs(linear_approximation(math.exp, 1.0, 1.01) - math.exp(1.01))
        far = abs(linear_approximation(math.exp, 1.0, 1.5) - math.exp(1.5))
        assert far > near

    def test_measured_error_is_of_the_same_order_as_the_second_order_term(self):
        """实测误差与 ``|f''(x₀)|·Δ²/2`` 同量级（这是"小步走"的数学理由）."""
        measured, bound = taylor_accuracy(math.exp, 1.0, 1.2)
        assert measured <= bound * 1.5
        assert measured >= bound * 0.2

    def test_non_finite_inputs_are_rejected(self):
        with pytest.raises(NumericError, match="展开点"):
            linear_approximation(math.exp, float("nan"), 1.0)
        with pytest.raises(NumericError, match="目标点"):
            taylor_accuracy(math.exp, 1.0, float("nan"))


class TestCloseForwarding:
    """``is_close`` 与 ``types.close`` 同口径（两处判据不能分家）."""

    def test_matches_types_close(self):
        from smart_research_agent.math_foundations.types import close

        for left, right in ((1.0, 1.0 + 1e-12), (1.0, 1.0 + 1e-3), (0.0, 1e-12)):
            assert is_close(left, right) == close(left, right)

    def test_relative_tolerance_at_large_scale(self):
        assert is_close(1e6, 1e6 + 1e-3) is True
