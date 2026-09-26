"""``math_foundations.autograd``：计算图、局部导数与梯度累积（day074）.

这一份测试的期望值全部是**手算**的：

```text
y = x·x        @3   → dy/dx = 6       （两条边各贡献 3）
y = x + x      @3   → dy/dx = 2
y = exp(x)     @1   → dy/dx = e
y = log(x)     @2   → dy/dx = 0.5
y = tanh(x)    @0   → dy/dx = 1
y = σ(x)       @0   → dy/dx = 0.25
y = x³         @2   → dy/dx = 12
```

另外三条断言对应三个"不报错但结果错"的陷阱：

```text
① 梯度必须**累加**（x·x 的两条边）——写成 = 只会给出一个偏小的梯度
② 上游梯度取的是**本节点**的梯度，不是左操作数的（这一课真的踩到了）
③ 复用一张图前必须 zero_grad()，否则第二次 backward 会把梯度叠上去
```
"""

from __future__ import annotations

import math

import pytest

from smart_research_agent.math_foundations.autograd import (
    Scalar,
    TraceRow,
    ensure_scalar,
    gradients_of,
    topological_order,
    value_and_grad,
)
from smart_research_agent.math_foundations.errors import NumericError, ParameterError
from tests.math_samples import approx


def grad_of(expression, point: float) -> float:
    """一元表达式的梯度（``value_and_grad`` 的一元简写）."""
    _value, grads = value_and_grad(expression, (point,))
    return grads[0]


class TestScalarConstruction:
    """节点的构造与校验."""

    def test_leaf_has_zero_gradient(self):
        node = Scalar(3.0, label="x")
        assert node.value == 3.0
        assert node.grad == 0.0
        assert node.label == "x"
        assert node.operation == "leaf"
        assert node.requires_grad is True

    def test_auto_label_uses_the_operation_name(self):
        assert Scalar(1.0).label == "nodeleaf"

    def test_int_is_accepted_and_converted(self):
        assert Scalar(2).value == 2.0
        assert isinstance(Scalar(2).value, float)

    @pytest.mark.parametrize("bad", ["3.0", None, [1.0]])
    def test_non_numeric_value_is_rejected(self, bad):
        with pytest.raises(NumericError, match="必须是实数"):
            Scalar(bad)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_value_is_rejected(self, bad):
        with pytest.raises(NumericError, match="有限实数"):
            Scalar(bad)

    def test_bool_is_rejected(self):
        """``True`` 是 ``int`` 的子类，但它显然不是"一个数"."""
        with pytest.raises(NumericError, match="必须是实数"):
            Scalar(True)


class TestArithmeticGradients:
    """四则运算的局部导数（手算值）."""

    def test_add(self):
        assert grad_of(lambda xs: xs[0] + 2.0, 3.0) == 1.0
        assert grad_of(lambda xs: 2.0 + xs[0], 3.0) == 1.0

    def test_sub(self):
        assert grad_of(lambda xs: xs[0] - 2.0, 3.0) == 1.0
        assert grad_of(lambda xs: 2.0 - xs[0], 3.0) == -1.0

    def test_mul(self):
        """``2x`` 的导数是 2；``x·x`` 的导数是 2x = 6."""
        assert grad_of(lambda xs: xs[0] * 2.0, 3.0) == 2.0
        assert grad_of(lambda xs: 2.0 * xs[0], 3.0) == 2.0
        assert grad_of(lambda xs: xs[0] * xs[0], 3.0) == 6.0

    def test_div(self):
        """``x/2`` 的导数是 0.5；``2/x`` 的导数是 −2/x² = −0.5（x=2）."""
        assert grad_of(lambda xs: xs[0] / 2.0, 3.0) == 0.5
        assert grad_of(lambda xs: 2.0 / xs[0], 2.0) == pytest.approx(-0.5)

    def test_pow(self):
        """``x³`` 在 2 处的导数是 12；``x^0.5`` 在 4 处的导数是 0.25."""
        assert grad_of(lambda xs: xs[0] ** 3, 2.0) == pytest.approx(12.0)
        assert grad_of(lambda xs: xs[0] ** 0.5, 4.0) == pytest.approx(0.25)

    def test_neg(self):
        assert grad_of(lambda xs: -xs[0], 3.0) == -1.0

    def test_combined_expression(self):
        """``3x² − 2x + 1`` 在 2 处的导数是 ``6x − 2 = 10``."""
        assert grad_of(lambda xs: 3.0 * xs[0] * xs[0] - 2.0 * xs[0] + 1.0, 2.0) == pytest.approx(
            10.0
        )


class TestUnaryFunctionGradients:
    """初等函数的局部导数（手算值）."""

    def test_exp(self):
        assert grad_of(lambda xs: xs[0].exp(), 1.0) == pytest.approx(math.e)

    def test_log(self):
        assert grad_of(lambda xs: xs[0].log(), 2.0) == pytest.approx(0.5)

    def test_sqrt(self):
        """``√x`` 在 9 处的导数是 ``1/(2·3) = 1/6``."""
        assert grad_of(lambda xs: xs[0].sqrt(), 9.0) == pytest.approx(1.0 / 6.0)

    def test_sigmoid_at_zero(self):
        """``σ'(0) = σ(0)(1−σ(0)) = 0.5·0.5 = 0.25``."""
        assert grad_of(lambda xs: xs[0].sigmoid(), 0.0) == pytest.approx(0.25)

    def test_tanh_at_zero(self):
        assert grad_of(lambda xs: xs[0].tanh(), 0.0) == pytest.approx(1.0)

    def test_sin_and_cos(self):
        """``sin`` 的导数是 ``cos``、``cos`` 的导数是 ``−sin``（符号最易漏）."""
        assert grad_of(lambda xs: xs[0].sin(), 0.0) == pytest.approx(1.0)
        assert grad_of(lambda xs: xs[0].cos(), 0.0) == pytest.approx(0.0)
        assert grad_of(lambda xs: xs[0].cos(), math.pi / 2.0) == pytest.approx(-1.0)

    def test_relu_positive_and_negative(self):
        """``ReLU`` 的正侧导数是 1、负侧是 0."""
        assert grad_of(lambda xs: xs[0].relu(), 2.0) == 1.0
        assert grad_of(lambda xs: xs[0].relu(), -2.0) == 0.0

    def test_relu_at_zero_takes_the_subgradient_zero(self):
        """``x = 0`` 处取次梯度 0（这是一个**约定**，因此必须被断言下来）."""
        assert grad_of(lambda xs: xs[0].relu(), 0.0) == 0.0

    def test_softplus_expression(self):
        """``log(1 + e^x)`` 的导数是 ``σ(x)``，在 0 处是 0.5."""
        assert grad_of(lambda xs: (xs[0].exp() + 1.0).log(), 0.0) == pytest.approx(0.5)


class TestDomainGuards:
    """定义域越界必须当场报错（不是安静地给出 inf/nan）."""

    def test_log_of_non_positive_is_rejected(self):
        with pytest.raises(NumericError, match="必须为正"):
            Scalar(0.0).log()
        with pytest.raises(NumericError, match="必须为正"):
            Scalar(-1.0).log()

    def test_sqrt_of_non_positive_is_rejected(self):
        with pytest.raises(NumericError, match="必须为正"):
            Scalar(0.0).sqrt()
        with pytest.raises(NumericError, match="必须为正"):
            Scalar(-4.0).sqrt()

    def test_division_by_zero_is_rejected(self):
        with pytest.raises(NumericError, match="除数是 0"):
            Scalar(1.0) / Scalar(0.0)
        with pytest.raises(NumericError, match="除数是 0"):
            Scalar(1.0) / 0.0

    def test_zero_to_a_fractional_power_is_rejected(self):
        with pytest.raises(NumericError, match="没有定义"):
            Scalar(0.0) ** 0.5

    def test_negative_base_with_fractional_power_is_rejected(self):
        with pytest.raises(NumericError, match="实数范围内没有定义"):
            Scalar(-2.0) ** 0.5

    def test_negative_base_with_integer_power_is_fine(self):
        assert (Scalar(-2.0) ** 2).value == 4.0

    def test_non_finite_exponent_is_rejected(self):
        with pytest.raises(ParameterError, match="有限实数"):
            Scalar(2.0) ** float("inf")

    def test_non_numeric_exponent_is_rejected(self):
        with pytest.raises(ParameterError, match="必须是实数"):
            Scalar(2.0) ** Scalar(2.0)  # type: ignore[operator]


class TestAccumulation:
    """梯度累积：这一课最值钱的断言（写成 ``=`` 不会报错，只会给出偏小的梯度）."""

    def test_same_node_used_twice_sums_the_paths(self):
        """``x·x`` 在 3 处的梯度是 6（3 + 3），而不是 3."""
        x = Scalar(3.0, label="x")
        (x * x).backward()
        assert x.grad == pytest.approx(6.0)

    def test_x_plus_x_sums_to_two(self):
        x = Scalar(3.0, label="x")
        (x + x).backward()
        assert x.grad == pytest.approx(2.0)

    def test_shared_subexpression_is_counted_once_per_path(self):
        """``y = x·x；z = y·x``：``dz/dx = ∂(x³)/∂x = 3x²``，在 2 处是 12."""
        x = Scalar(2.0, label="x")
        y = x * x
        z = y * x
        z.backward()
        assert x.grad == pytest.approx(12.0)

    def test_three_inputs_get_independent_gradients(self):
        """``x·y + z²`` 在 (2, 3, 4) 处的梯度是 (3, 2, 8)."""
        x, y, z = Scalar(2.0, label="x"), Scalar(3.0, label="y"), Scalar(4.0, label="z")
        (x * y + z * z).backward()
        assert (x.grad, y.grad, z.grad) == (pytest.approx(3.0), pytest.approx(2.0), pytest.approx(8.0))

    def test_repeated_backward_stacks_gradients(self):
        """同一张图跑两次 ``backward()`` 会叠加（因此存在 ``zero_grad``）."""
        x = Scalar(3.0, label="x")
        y = x * x
        y.backward()
        y.backward()
        assert x.grad == pytest.approx(12.0)

    def test_zero_grad_resets_the_whole_graph(self):
        x = Scalar(3.0, label="x")
        y = x * x
        y.backward()
        y.zero_grad()
        assert x.grad == 0.0
        y.backward()
        assert x.grad == pytest.approx(6.0)


class TestBackwardDetails:
    """``backward`` 的 seed、requires_grad 与叶子节点."""

    def test_seed_scales_the_gradient(self):
        x = Scalar(3.0, label="x")
        (x * x).backward(seed=2.0)
        assert x.grad == pytest.approx(12.0)

    def test_non_finite_seed_is_rejected(self):
        with pytest.raises(ParameterError, match="seed"):
            Scalar(1.0).backward(seed=float("inf"))

    def test_leaf_backward_sets_its_own_gradient(self):
        x = Scalar(3.0, label="x")
        x.backward()
        assert x.grad == 1.0

    def test_constant_node_does_not_receive_gradient(self):
        """``requires_grad=False`` 的节点是常量：它参与建图，但不接收梯度."""
        constant = Scalar(2.0, label="c", requires_grad=False)
        result = constant * 3.0
        assert result.requires_grad is False
        result.backward()
        assert constant.grad == 0.0

    def test_all_constant_expression_has_no_graph_above_it(self):
        product = ensure_scalar(2.0) * ensure_scalar(3.0)
        assert product.requires_grad is False
        assert product.value == 6.0

    def test_gradient_flows_through_constants_to_variables(self):
        """常量在链上，但它不是终点：``x * 2`` 的梯度仍然是 2."""
        x = Scalar(5.0, label="x")
        (x * 2.0).backward()
        assert x.grad == 2.0


class TestTopologicalOrder:
    """拓扑序：从根到叶，正是反向传播要的顺序."""

    def test_order_is_root_first(self):
        x = Scalar(1.0, label="x")
        y = x * x
        z = y + x
        order = topological_order(z)
        assert order[0] is z
        assert order[-1] is x

    def test_every_node_appears_exactly_once(self):
        x = Scalar(1.0, label="x")
        y = (x + 1.0) * (x - 1.0)
        order = topological_order(y)
        assert len(order) == len({id(node) for node in order})

    def test_parents_come_after_their_children(self):
        x = Scalar(1.0, label="x")
        y = x.exp()
        z = y.log()
        order = topological_order(z)
        assert order.index(z) < order.index(y) < order.index(x)

    def test_deep_chain_does_not_hit_the_recursion_limit(self):
        """两千层加法链：迭代式实现不会撞上 Python 的递归上限（缺省 1000）."""
        x = Scalar(0.0, label="x")
        node = x
        for _ in range(2000):
            node = node + 1.0
        node.backward()
        assert x.grad == 1.0

    def test_diamond_graph_visits_the_shared_node_once(self):
        """菱形图（两条路都经过 ``x``）里 ``x`` 只出现一次，且在中间节点之后.

        注意"最后一个是 ``x``"**不成立**：两个常量（``3.0`` 与 ``4.0``）
        也在图上，它们的拓扑位置与 ``x`` 不可比。断言必须写成
        "``x`` 在两条路径的中间节点之后"，而不是"它是最后一个"——
        后者在图上多一个常量时就会失败，而失败的原因与这一条想说的话无关。
        """
        x = Scalar(2.0, label="x")
        left = x * 3.0
        right = x + 4.0
        order = topological_order(left * right)
        assert len(order) == len({id(node) for node in order})
        assert sum(1 for node in order if node is x) == 1
        assert order.index(x) > order.index(left)
        assert order.index(x) > order.index(right)


class TestValueAndGrad:
    """``value_and_grad``：一串普通数进、一串普通数出."""

    def test_single_input(self):
        value, grads = value_and_grad(lambda xs: (xs[0] ** 2).exp(), (2.0,))
        assert value == pytest.approx(math.exp(4.0))
        assert grads[0] == pytest.approx(4.0 * math.exp(4.0))

    def test_two_inputs(self):
        """``x·y + e^y`` 在 (0.6, −0.8) 处：(∂x, ∂y) = (y, x + e^y)."""
        value, grads = value_and_grad(lambda xs: xs[0] * xs[1] + xs[1].exp(), (0.6, -0.8))
        assert value == pytest.approx(0.6 * -0.8 + math.exp(-0.8))
        assert grads[0] == pytest.approx(-0.8)
        assert grads[1] == pytest.approx(0.6 + math.exp(-0.8))

    def test_empty_inputs_are_rejected(self):
        with pytest.raises(ParameterError, match="至少需要一个输入"):
            value_and_grad(lambda xs: Scalar(1.0), ())

    def test_expression_must_return_a_scalar(self):
        with pytest.raises(ParameterError, match="必须返回一个 Scalar"):
            value_and_grad(lambda xs: xs[0].value, (1.0,))  # type: ignore[return-value]

    def test_gradient_is_zero_at_a_minimum(self):
        """``(x−3)²`` 在 3 处的梯度是 0（这一点也是"收敛"的定义）."""
        _value, grads = value_and_grad(lambda xs: (xs[0] - 3.0) * (xs[0] - 3.0), (3.0,))
        assert grads[0] == pytest.approx(0.0)


class TestTracing:
    """计算图的可读形式（``gradients_of`` / ``trace`` / ``TraceRow``）."""

    def test_gradients_of_returns_labels(self):
        x = Scalar(3.0, label="x")
        y = x * x
        y.backward()
        table = gradients_of(y)
        assert table["x"] == pytest.approx(6.0)

    def test_trace_rows_are_leaf_first(self):
        x = Scalar(2.0, label="x")
        y = (x * x).exp()
        y.backward()
        rows = y.trace()
        assert rows[0].label == "x"
        assert rows[-1].label == y.label
        assert rows[-1].grad == pytest.approx(1.0)

    def test_trace_row_is_json_friendly(self):
        import json

        row = TraceRow(label="x", operation="leaf", value=2.0, grad=4.0)
        payload = row.to_dict()
        json.dumps(payload)
        assert payload["operation"] == "leaf"
        assert "value=+2.000000" in row.summary_line()

    def test_operation_names_are_recorded(self):
        x = Scalar(2.0, label="x")
        y = x * x + x.sigmoid()
        operations = {node.operation for node in topological_order(y)}
        assert {"mul", "add", "sigmoid"} <= operations


class TestEnsureScalar:
    """``ensure_scalar``：把普通数包成常量节点."""

    def test_wraps_a_number(self):
        node = ensure_scalar(2.5)
        assert node.value == 2.5
        assert node.requires_grad is False

    def test_returns_the_same_node_for_a_scalar(self):
        existing = Scalar(1.0)
        assert ensure_scalar(existing) is existing

    def test_rejects_strings(self):
        with pytest.raises(NumericError, match="必须是实数或 Scalar"):
            ensure_scalar("2.0")  # type: ignore[arg-type]

    def test_rejects_bool(self):
        with pytest.raises(NumericError, match="必须是实数或 Scalar"):
            ensure_scalar(True)

    def test_error_message_uses_the_given_name(self):
        with pytest.raises(NumericError, match="除数"):
            Scalar(1.0) / None  # type: ignore[operator]


class TestMatchesNumericDerivative:
    """自动微分与中心差分逐点一致（两条独立路径的对照，比单点断言强得多）."""

    CASES = (
        (lambda xs: (xs[0] * 3.0 + 1.0).sigmoid(), (0.7,)),
        (lambda xs: (xs[0].exp() + 1.0).log() + xs[0] * xs[0], (0.4,)),
        (lambda xs: xs[0].tanh() * (xs[0] + 1.0).relu(), (0.25,)),
        (lambda xs: (xs[0] * xs[1] + xs[1].exp()) / (1.0 + xs[0] * xs[0]), (0.6, -0.8)),
    )

    @pytest.mark.parametrize("index", range(4))
    def test_all_partials_match_central_difference(self, index):
        from smart_research_agent.math_foundations.calculus import gradient

        expression, point = self.CASES[index]
        _value, analytic = value_and_grad(expression, point)
        numeric = gradient(
            lambda vector: expression([Scalar(value) for value in vector]).value, point
        )
        for expected, observed in zip(analytic, numeric):
            assert approx(expected, observed, tolerance=1e-7)
