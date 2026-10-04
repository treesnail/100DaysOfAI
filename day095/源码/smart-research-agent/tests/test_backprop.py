"""``backprop``：一次反向传播的测试（day090 / M8-D2）.

本文件覆盖新包的九个功能模块与包入口：口径表、八个失败族、六个激活的局部导数、
两个损失的梯度、一张张量级计算图、Dense 的三块梯度、整条 MLP 的回传、
一次解析反向训练、七条性质与五张表。

样本全部来自本包自己的确定性构造（LCG + 写死输入）；**跨天对账真的调用既有包**
（``math_foundations`` 的 ``calculus`` / ``gradcheck`` / ``optim``、``sft`` 的生产交叉熵、
``encoder_decoder`` 的前馈反向），因此这一课考的就是"这条反向被真的接上了没有"。
"""

from __future__ import annotations

import math

import pytest

from smart_research_agent.backprop import (
    errors,
    gradients,
    graph,
    layers,
    network,
    study,
    train,
    types,
    verify,
)
from smart_research_agent.neural_basics.activations import softmax as reference_softmax

# --------------------------------------------------------------------------- 口径表


def test_six_derivative_formulas_are_closed() -> None:
    """六个激活的导数公式逐键对齐（名单与 day089 的 ACTIVATIONS 相同）。"""
    from smart_research_agent.neural_basics.types import ACTIVATIONS

    assert set(types.ACTIVATION_DERIVATIVE_FORMULAS) == set(ACTIVATIONS)
    assert len(types.ACTIVATION_DERIVATIVE_FORMULAS) == 6
    assert all(types.ACTIVATION_DERIVATIVE_FORMULAS.values())


def test_elementwise_activations_plus_softmax() -> None:
    """逐元素激活五个 + softmax 一个 == 六个（少一个就会在反向里静默走不到分支）。"""
    assert len(types.ELEMENTWISE_ACTIVATIONS) == 5
    assert "softmax" not in types.ELEMENTWISE_ACTIVATIONS
    assert set(types.ELEMENTWISE_ACTIVATIONS) | {"softmax"} == set(types.ACTIVATION_DERIVATIVE_FORMULAS)


def test_two_losses_are_closed() -> None:
    """两个损失的梯度公式逐键对齐。"""
    assert types.LOSSES == ("mse", "cross_entropy")
    assert set(types.LOSS_GRADIENT_FORMULAS) == set(types.LOSSES)
    assert all(types.LOSS_GRADIENT_FORMULAS.values())


def test_backward_rules_cover_every_operator() -> None:
    """逐算子规则表覆盖这一课用到的全部算子。"""
    required = {
        "add",
        "mul",
        "matmul",
        "add_bias",
        "relu",
        "sigmoid",
        "tanh",
        "gelu",
        "softmax",
        "mse",
        "cross_entropy",
    }
    assert required <= set(types.BACKWARD_RULES)
    assert all(types.BACKWARD_RULES.values())


def test_seven_properties_have_descriptions_and_failures() -> None:
    """七条性质：名单、说明、"失败意味着什么"三张表逐键对齐。"""
    assert len(types.BACKPROP_PROPERTIES) == 7
    assert set(types.PROPERTY_DESCRIPTIONS) == set(types.BACKPROP_PROPERTIES)
    assert set(types.PROPERTY_FAILURE) == set(types.BACKPROP_PROPERTIES)
    assert all(types.PROPERTY_DESCRIPTIONS.values())
    assert all(types.PROPERTY_FAILURE.values())


def test_notes_are_ten_and_ordered() -> None:
    """十条笔记，且顺序表与键集合一致。"""
    assert len(types.BACKPROP_NOTES) == 10
    assert types.BACKPROP_NOTES_ORDER == tuple(types.BACKPROP_NOTES)
    assert all(types.BACKPROP_NOTES.values())


def test_boundaries_are_five() -> None:
    """五条边界是一份非空清单（**这一课明确不承诺的事**）。"""
    assert len(types.BACKPROP_BOUNDARIES) == 5
    assert all(types.BACKPROP_BOUNDARIES)


def test_torch_counterparts_cover_every_operator() -> None:
    """PyTorch 对照表覆盖反向的每一个算子，且**不出现**版本号。"""
    required = {
        "backward",
        "zero_grad",
        "graph",
        "dense_backward",
        "relu_backward",
        "sigmoid_backward",
        "tanh_backward",
        "gelu_backward",
        "softmax_backward",
        "mse_grad",
        "cross_entropy_grad",
        "grad_check",
    }
    assert required <= set(types.TORCH_COUNTERPARTS)
    assert all(("torch" in value or "nn." in value) for value in types.TORCH_COUNTERPARTS.values())
    assert "torch==" not in " ".join(types.TORCH_COUNTERPARTS.values())


# --------------------------------------------------------------------------- 记录


def test_dense_gradients_record() -> None:
    """DenseGradients：形状、摊平字段与一行说明。"""
    grads = types.DenseGradients(
        grad_weight=((1.0, 2.0), (3.0, 4.0), (5.0, 6.0)),
        grad_bias=(0.5, -0.5, 1.5),
        grad_inputs=((1.0, 1.0),),
    )
    assert grads.weight_shape == (3, 2)
    assert grads.to_dict()["grad_bias_len"] == 3
    assert grads.to_dict()["sum_grad_bias"] == pytest.approx(1.5)
    assert grads.line().startswith("dW (3, 2)")


def test_ffn_gradients_record() -> None:
    """FFNGradients：摊平字段与一行说明（字段名与 encoder_decoder 逐一相同）。"""
    grads = types.FFNGradients(
        grad_w_in=((1.0, 2.0), (3.0, 4.0)),
        grad_b_in=(0.5, 0.5),
        grad_w_out=((1.0, 2.0), (3.0, 4.0)),
        grad_b_out=(0.5, 0.5),
        grad_inputs=((1.0, 2.0),),
    )
    assert grads.to_dict()["w_in_shape"] == [2, 2]
    assert grads.to_dict()["b_out_len"] == 2
    assert grads.line().startswith("dw_in (2, 2)")


def test_gradient_reading_validates_fields() -> None:
    """GradientReading：非有限读数 / 空形状 / 空公式都被拒。"""
    row = types.GradientReading(name="relu", value=1.0, shape="逐元素", formula="1 if x > 0 else 0")
    assert row.line().startswith("relu")
    assert row.to_dict()["shape"] == "逐元素"
    with pytest.raises(errors.NumericError, match="非有限"):
        types.GradientReading(name="x", value=math.inf, shape="s", formula="f")
    with pytest.raises(errors.ParameterError, match="形状或公式"):
        types.GradientReading(name="x", value=1.0, shape="", formula="f")


def test_layer_gradients_validates_norms() -> None:
    """LayerGradients：范数必须非负有限。"""
    row = types.LayerGradients(
        index=1, weight_shape=(3, 2), weight_norm=1.0, bias_norm=2.0, input_norm=3.0
    )
    assert row.line().startswith("第 1 层")
    assert row.to_dict()["weight_shape"] == [3, 2]
    with pytest.raises(errors.NumericError, match="非负有限"):
        types.LayerGradients(
            index=1, weight_shape=(1, 1), weight_norm=-1.0, bias_norm=0.0, input_norm=0.0
        )


def test_require_positive_float_guard() -> None:
    """正有限数护栏拒绝 bool / 字符串 / 非正数。"""
    assert types._require_positive_float(2, name="x") == 2.0
    for bad in (True, "2"):
        with pytest.raises(errors.ParameterError, match="实数"):
            types._require_positive_float(bad, name="x")
    for bad in (0.0, -1.0, math.inf):
        with pytest.raises(errors.ParameterError, match="正的有限数"):
            types._require_positive_float(bad, name="x")


# --------------------------------------------------------------------------- 失败族


def test_every_family_has_a_class_and_an_outcome() -> None:
    """七个失败族：``FAMILY_OUTCOMES`` 与异常类逐键对齐（闭合检查）。"""
    assert set(errors.FAMILY_OUTCOMES) == {
        "ShapeError",
        "ParameterError",
        "NumericError",
        "GradientError",
        "BackwardError",
        "ChainError",
        "StepError",
    }
    for name in errors.FAMILY_OUTCOMES:
        assert isinstance(getattr(errors, name), type)
        assert issubclass(getattr(errors, name), errors.BackpropError)
        assert errors.FAMILY_OUTCOMES[name].startswith("改")


def test_families_inherit_from_transformer_core() -> None:
    """四族多继承 day075 的族（因此 ``except GradientError`` 能兜住本层）。"""
    from smart_research_agent.transformer_core import errors as core

    assert issubclass(errors.ShapeError, core.ShapeError)
    assert issubclass(errors.ParameterError, core.ParameterError)
    assert issubclass(errors.NumericError, core.NumericError)
    assert issubclass(errors.GradientError, core.GradientError)
    assert issubclass(errors.BackpropError, ValueError)


def test_gradient_error_has_returned() -> None:
    """**GradientError 回来了**：连续缺席八天之后第一次被真的抛出。"""
    assert errors.RETURNED_FAMILY == "GradientError"
    assert errors.ABSENT_FAMILY is None
    assert "day075" in errors.RETURNED_FAMILY_REASON
    assert "day090" in errors.RETURNED_FAMILY_REASON
    assert "八天" in errors.RETURNED_FAMILY_REASON
    assert "不缺席任何一族" in errors.ABSENT_FAMILY_REASON


# --------------------------------------------------------------------------- 局部导数


def test_relu_derivative_is_a_switch() -> None:
    """relu 的导数是一个开关：正数 1、负数 0、**0 处取 0**。"""
    assert gradients.relu_derivative(2.0) == 1.0
    assert gradients.relu_derivative(-2.0) == 0.0
    assert gradients.relu_derivative(0.0) == 0.0
    assert gradients.relu_derivative(-1e6) == 0.0


def test_leaky_relu_derivative_keeps_the_slope() -> None:
    """leaky_relu 的导数在负半轴是那条坡度本身。"""
    assert gradients.leaky_relu_derivative(2.0) == 1.0
    assert gradients.leaky_relu_derivative(-2.0) == pytest.approx(0.01)
    assert gradients.leaky_relu_derivative(-2.0, slope=0.3) == pytest.approx(0.3)
    with pytest.raises(errors.NumericError, match="斜率"):
        gradients.leaky_relu_derivative(-1.0, slope=-0.5)


def test_sigmoid_and_tanh_derivatives() -> None:
    """σ' = σ(1−σ)（上界 0.25）、tanh' = 1 − tanh²（上界 1），且可用输出直接算。"""
    assert gradients.sigmoid_derivative(0.0) == pytest.approx(0.25)
    assert gradients.sigmoid_derivative(1.0) == pytest.approx(0.19661193324148185, abs=1e-15)
    assert gradients.sigmoid_derivative_from_output(0.5) == pytest.approx(0.25)
    assert gradients.tanh_derivative(0.0) == pytest.approx(1.0)
    assert gradients.tanh_derivative_from_output(0.0) == pytest.approx(1.0)
    assert gradients.SIGMOID_DERIVATIVE_CEILING == 0.25


def test_derivative_from_output_rejects_non_finite() -> None:
    """用输出直接算的两条导数拒绝非有限输出。"""
    with pytest.raises(errors.NumericError, match="有限数"):
        gradients.sigmoid_derivative_from_output(math.inf)
    with pytest.raises(errors.NumericError, match="有限数"):
        gradients.tanh_derivative_from_output(math.nan)


def test_gelu_derivative_matches_the_exact_formula() -> None:
    """gelu 的导数在 0 处恰好 0.5，且与 ``encoder_decoder`` 的激活反向逐位一致。"""
    from smart_research_agent.encoder_decoder.layers import activation_backward as day079

    assert gradients.gelu_derivative(0.0) == pytest.approx(0.5, abs=1e-15)
    assert gradients.gelu_derivative(1.0) == pytest.approx(1.0833154705876863, abs=1e-15)
    points = tuple((-3.0, -1.0, 0.0, 0.5, 2.0))
    mine = gradients.elementwise_backward("gelu", (points,), (tuple(1.0 for _ in points),))
    theirs = day079(tuple((points,)), tuple((tuple(1.0 for _ in points),)), "gelu")
    assert mine == theirs


def test_gelu_tanh_derivative_is_a_different_formula() -> None:
    """gelu_new 的 tanh 近似：0 处与精确式同值，两侧有可量出的偏差。"""
    assert gradients.gelu_tanh_derivative(0.0) == pytest.approx(0.5, abs=1e-15)
    worst = max(
        abs(gradients.gelu_derivative(x) - gradients.gelu_tanh_derivative(x))
        for x in (-3.0, -1.5, 1.5, 3.0)
    )
    assert worst > 1e-4


def test_activation_derivative_dispatch_and_guards() -> None:
    """activation_derivative：未知激活与 softmax 都被拒（后者不是逐元素）。"""
    assert gradients.activation_derivative("relu", 2.0) == 1.0
    assert gradients.activation_derivative("sigmoid", 0.0) == pytest.approx(0.25)
    with pytest.raises(errors.ParameterError, match="未知的激活"):
        gradients.activation_derivative("rellu", 1.0)
    with pytest.raises(errors.ParameterError, match="不是逐元素"):
        gradients.activation_derivative("softmax", 1.0)


# --------------------------------------------------------------------------- softmax


def test_softmax_jacobian_rows_sum_to_zero() -> None:
    """softmax 的雅可比每一行和为 0（这是"平移不变"在导数上的样子）。"""
    probabilities = reference_softmax((1.0, 2.0, 3.0))
    jacobian = gradients.softmax_jacobian(probabilities)
    assert len(jacobian) == 3
    for row in jacobian:
        assert math.fsum(row) == pytest.approx(0.0, abs=1e-16)
    assert jacobian[0][0] == pytest.approx(probabilities[0] * (1.0 - probabilities[0]), abs=1e-18)


def test_softmax_jacobian_rejects_empty() -> None:
    """空分布没有雅可比。"""
    with pytest.raises(errors.ShapeError, match="不能为空"):
        gradients.softmax_jacobian(())


def test_softmax_jvp_matches_explicit_matrix() -> None:
    """``(Jᵀv)_i = p_i(v_i − ⟨p,v⟩)`` 与显式矩阵乘逐点一致。"""
    for probabilities, vector in verify.JVP_CASES:
        jacobian = gradients.softmax_jacobian(probabilities)
        explicit = tuple(
            math.fsum(jacobian[row][column] * vector[row] for row in range(len(vector)))
            for column in range(len(vector))
        )
        assert gradients.softmax_jacobian_vector_product(probabilities, vector) == pytest.approx(
            explicit, abs=1e-15
        )


def test_softmax_jvp_guards() -> None:
    """JVP 拒绝空输入与长度不一致。"""
    with pytest.raises(errors.ShapeError, match="不接受空输入"):
        gradients.softmax_jacobian_vector_product((), (1.0,))
    with pytest.raises(errors.ShapeError, match="一一对应"):
        gradients.softmax_jacobian_vector_product((0.5, 0.5), (1.0,))


def test_softmax_backward_rows() -> None:
    """逐行 softmax 的反向：形状不符抛 ShapeError；恒等梯度给出 0（因为行内积为 1）。"""
    probabilities = ((0.2, 0.3, 0.5),)
    grad = gradients.softmax_backward_rows(probabilities, ((1.0, 1.0, 1.0),))
    assert grad[0] == pytest.approx((0.0, 0.0, 0.0), abs=1e-18)
    with pytest.raises(errors.ShapeError, match="形状必须一致"):
        gradients.softmax_backward_rows(probabilities, ((1.0, 1.0),))


# --------------------------------------------------------------------------- 逐元素反向


def test_elementwise_backward_relu_is_exactly_zero() -> None:
    """relu 的反向是逐元素的开关，且负半轴与 0 处都是**字面量 0.0**。"""
    result = gradients.elementwise_backward("relu", ((0.0, -1.0, 2.0),), ((5.0, 5.0, 5.0),))
    assert result == ((0.0, 0.0, 5.0),)
    assert result[0][0] == 0.0  # x = 0：用 >= 的实现会在这里把梯度放过去
    assert result[0][1] == 0.0


def test_elementwise_backward_guards() -> None:
    """未知/非逐元素激活、形状不符、非有限数都被拒。"""
    with pytest.raises(errors.ParameterError, match="只支持逐元素"):
        gradients.elementwise_backward("softmax", ((1.0,),), ((1.0,),))
    with pytest.raises(errors.ShapeError, match="形状必须一致"):
        gradients.elementwise_backward("relu", ((1.0, 2.0),), ((1.0,),))
    with pytest.raises(errors.NumericError, match="非有限数"):
        gradients.elementwise_backward("relu", ((math.inf,),), ((1.0,),))
    with pytest.raises(errors.ShapeError, match="非空的二维序列"):
        gradients.elementwise_backward("relu", (), ())


def test_elementwise_backward_gelu_uses_the_input() -> None:
    """gelu 的反向需要**激活前**的值（同一输出配不同输入会给出不同梯度）。"""
    first = gradients.elementwise_backward("gelu", ((0.0,),), ((1.0,),))
    second = gradients.elementwise_backward("gelu", ((2.0,),), ((1.0,),))
    assert first[0][0] == pytest.approx(0.5)
    assert second[0][0] == pytest.approx(gradients.gelu_derivative(2.0))


# --------------------------------------------------------------------------- 损失梯度


def test_mse_grad_matches_hand() -> None:
    """mse 的梯度 = 2(pred − target)/N（N 是**全部元素**个数）。"""
    pred = ((1.0, 2.0), (3.0, 4.0))
    target = ((1.5, 1.5), (4.0, 3.0))
    grads = gradients.mse_grad(pred, target)
    assert grads == ((-0.25, 0.25), (-0.5, 0.5))


def test_mse_grad_guards() -> None:
    """mse 的梯度：行数 / 宽度 / 有限性三道护栏。"""
    with pytest.raises(errors.ShapeError, match="同形非空"):
        gradients.mse_grad((), ())
    with pytest.raises(errors.ShapeError, match="同形非空"):
        gradients.mse_grad(((1.0,),), ())
    with pytest.raises(errors.ShapeError, match="宽度"):
        gradients.mse_grad(((1.0, 2.0),), ((1.0,),))
    with pytest.raises(errors.NumericError, match="非有限数"):
        gradients.mse_grad(((math.inf,),), ((1.0,),))


def test_cross_entropy_grad_is_p_minus_onehot() -> None:
    """交叉熵的梯度 = p − onehot。"""
    grads = gradients.cross_entropy_grad((1.0, 2.0, 3.0), 0)
    probabilities = reference_softmax((1.0, 2.0, 3.0))
    assert grads[0] == pytest.approx(probabilities[0] - 1.0)
    assert grads[1] == pytest.approx(probabilities[1])
    assert grads[2] == pytest.approx(probabilities[2])


def test_cross_entropy_grad_guards() -> None:
    """交叉熵的梯度：空打分、非有限、越界标签、bool 标签都被拒。"""
    with pytest.raises(errors.ShapeError, match="不能为空"):
        gradients.cross_entropy_grad((), 0)
    with pytest.raises(errors.NumericError, match="有限数"):
        gradients.cross_entropy_grad((math.inf, 0.0), 0)
    with pytest.raises(errors.ShapeError, match="之外"):
        gradients.cross_entropy_grad((1.0, 2.0), 5)
    with pytest.raises(errors.ParameterError, match="整数"):
        gradients.cross_entropy_grad((1.0, 2.0), True)


def test_cross_entropy_grad_rows_divides_by_rows() -> None:
    """逐行交叉熵的梯度对**行取平均**（忘记分母会让梯度大 rows 倍）。"""
    logits = ((1.0, 2.0), (2.0, 1.0))
    rows = gradients.cross_entropy_grad_rows(logits, (1, 0))
    single = gradients.cross_entropy_grad((1.0, 2.0), 1)
    assert rows[0] == tuple(value / 2 for value in single)
    with pytest.raises(errors.ShapeError, match="一一对应"):
        gradients.cross_entropy_grad_rows(logits, (1,))
    with pytest.raises(errors.ShapeError, match="样本为空"):
        gradients.cross_entropy_grad_rows((), ())


def test_loss_gradient_names() -> None:
    """损失梯度名单与 types.LOSSES 一致。"""
    assert gradients.LOSS_GRADIENT_NAMES == types.LOSSES


# --------------------------------------------------------------------------- 计算图


def test_tensor_shape_helpers() -> None:
    """标量 / 向量 / 矩阵的识别与形状（bool 不是标量）。"""
    assert graph.is_scalar(1.0) is True
    assert graph.is_scalar(True) is False
    assert graph.is_vector((1.0, 2.0)) is True
    assert graph.is_vector(()) is False
    assert graph.is_matrix(((1.0, 2.0), (3.0, 4.0))) is True
    assert graph.is_matrix(((1.0, 2.0), (3.0,))) is False
    assert graph.tensor_shape(1.0) == ()
    assert graph.tensor_shape((1.0, 2.0)) == (2,)
    assert graph.tensor_shape(((1.0, 2.0),)) == (1, 2)
    with pytest.raises(errors.ShapeError, match="无法识别"):
        graph.tensor_shape("abc")


def test_zeros_and_ones_and_flatten() -> None:
    """同形状的全零 / 全一张量，以及按行优先压平。"""
    assert graph.zeros_like(((1.0, 2.0),)) == ((0.0, 0.0),)
    assert graph.ones_like(3.0) == 1.0
    assert graph.ones_like((1.0, 2.0)) == (1.0, 1.0)
    assert graph.flatten_value(((1.0, 2.0), (3.0, 4.0))) == (1.0, 2.0, 3.0, 4.0)
    assert graph.flatten_value(2.0) == (2.0,)
    with pytest.raises(errors.ShapeError, match="无法识别"):
        graph.flatten_value("abc")
    with pytest.raises(errors.ShapeError, match="无法识别"):
        graph.zeros_like("abc")


def test_add_values_accumulates_and_guards() -> None:
    """梯度累加：同形相加；形状不一致当场拒绝（而不是静默覆盖）。"""
    assert graph.add_values(1.0, 2.0) == 3.0
    assert graph.add_values((1.0, 2.0), (3.0, 4.0)) == (4.0, 6.0)
    assert graph.add_values(((1.0,),), ((2.0,),)) == ((3.0,),)
    with pytest.raises(errors.ShapeError, match="无法累加"):
        graph.add_values((1.0, 2.0), (1.0,))


def test_node_validates_value() -> None:
    """节点护栏：结构不明 / 含非有限数 / 从非标量反向 都被拒。"""
    with pytest.raises(errors.ShapeError, match="标量 / 向量 / 矩阵"):
        graph.Node("abc")
    with pytest.raises(errors.ShapeError, match="有限数"):
        graph.Node((1.0, math.inf))
    node = graph.variable((1.0, 2.0), label="v")
    assert node.label == "v"
    assert "sum_all" in repr(node) or "variable" in repr(node)
    with pytest.raises(errors.ShapeError, match="只能从"):
        node.backward()
    with pytest.raises(errors.ParameterError, match="seed"):
        graph.sum_all(node).backward(seed=math.inf)


def test_variable_and_constant_requires_grad() -> None:
    """变量的 requires_grad 为真、常量为假；常量参与前向但不接收梯度。"""
    inputs = graph.constant(((1.0, 1.0),), label="X")
    weight = graph.variable(((1.0, 2.0),), label="W")
    bias = graph.variable((0.5,), label="b")
    assert weight.requires_grad is True
    assert inputs.requires_grad is False
    total = graph.sum_all(graph.dense(inputs, weight, bias))
    total.backward()
    assert inputs.grad == ((0.0, 0.0),)  # 常量不接收梯度
    assert weight.grad == ((1.0, 1.0),)
    assert bias.grad == (1.0,)
    assert graph.grad_of(total) == 1.0


def test_ensure_node_wraps_tensors() -> None:
    """裸张量被包成常量节点；已经是节点时原样返回。"""
    wrapped = graph.ensure_node(((1.0, 2.0),), name="X")
    assert wrapped.operation == "constant"
    assert wrapped.requires_grad is False
    same = graph.variable((1.0, 2.0))
    assert graph.ensure_node(same) is same


def test_matmul_forward_and_backward() -> None:
    """matmul：``C = A·B`` ⇒ ``dA = dC·Bᵀ``、``dB = Aᵀ·dC``（对手算值）。"""
    left = graph.variable(((1.0, 2.0), (3.0, 4.0)))
    right = graph.variable(((5.0, 6.0), (7.0, 8.0)))
    product = graph.matmul(left, right)
    assert product.value == ((19.0, 22.0), (43.0, 50.0))
    graph.sum_all(product).backward()
    assert left.grad == ((11.0, 15.0), (11.0, 15.0))
    assert right.grad == ((4.0, 4.0), (6.0, 6.0))


def test_matmul_guards() -> None:
    """matmul 只接受矩阵，且内维必须一致。"""
    with pytest.raises(errors.ShapeError, match="两个矩阵"):
        graph.matmul(graph.constant((1.0, 2.0)), graph.constant(((1.0,), (2.0,))))
    with pytest.raises(errors.ShapeError, match="内维不一致"):
        graph.matmul(graph.constant(((1.0, 2.0),)), graph.constant(((1.0, 2.0),)))


def test_add_bias_forward_and_backward() -> None:
    """add_bias：偏置逐行广播；反向里偏置拿到**列求和**。"""
    values = graph.variable(((1.0, 2.0), (3.0, 4.0)))
    bias = graph.variable((0.5, -0.5))
    total = graph.sum_all(graph.add_bias(values, bias))
    assert total.value == pytest.approx(10.0)
    total.backward()
    assert values.grad == ((1.0, 1.0), (1.0, 1.0))
    assert bias.grad == (2.0, 2.0)


def test_add_bias_guards() -> None:
    """add_bias 需要一个矩阵与一个向量，且长度要对齐。"""
    with pytest.raises(errors.ShapeError, match="矩阵与一个向量"):
        graph.add_bias(graph.constant((1.0, 2.0)), graph.constant((1.0, 2.0)))
    with pytest.raises(errors.ShapeError, match="偏置长度"):
        graph.add_bias(graph.constant(((1.0, 2.0),)), graph.constant((1.0,)))


def test_dense_node_matches_handwritten_backward() -> None:
    """**两条独立路径**：图的 dense 与 layers.dense_backward 给出同一组梯度。"""
    weight = ((1.0, 2.0), (3.0, 4.0))
    bias = (0.5, -0.5)
    inputs = ((1.0, 0.0),)
    weight_node = graph.variable(weight)
    bias_node = graph.variable(bias)
    output = graph.dense(graph.constant(inputs), weight_node, bias_node)
    assert output.value == ((1.5, 2.5),)
    grad_output = ((1.0, 1.0),)
    graph.sum_all(output).backward()
    handwritten = layers.dense_backward(weight, bias, inputs, grad_output)
    assert weight_node.grad == handwritten.grad_weight
    assert bias_node.grad == handwritten.grad_bias


def test_dense_guards() -> None:
    """dense 需要 矩阵 / 矩阵 / 向量，且三个宽度都要对齐。"""
    with pytest.raises(errors.ShapeError, match="输入矩阵"):
        graph.dense(graph.constant((1.0, 2.0)), graph.constant(((1.0, 2.0),)), graph.constant((1.0,)))
    with pytest.raises(errors.ShapeError, match="列数"):
        graph.dense(graph.constant(((1.0, 2.0),)), graph.constant(((1.0, 2.0, 3.0),)), graph.constant((1.0,)))
    with pytest.raises(errors.ShapeError, match="偏置长度"):
        graph.dense(graph.constant(((1.0, 2.0),)), graph.constant(((1.0, 2.0),)), graph.constant((1.0, 1.0)))


def test_relu_and_gelu_nodes_on_scalar() -> None:
    """relu / gelu 的节点在**标量**上也成立（逐元素算子不关心形状）。"""
    positive = graph.relu(graph.variable(2.0))
    assert positive.value == 2.0
    positive.backward()
    assert positive.grad == 1.0

    negative = graph.relu(graph.variable(-2.0))
    assert negative.value == 0.0
    negative.backward()
    assert negative.grad == 1.0  # 标量节点的梯度是 seed；它的父节点拿到 0

    smooth = graph.gelu(graph.variable(2.0))
    assert smooth.value == pytest.approx(1.9544997361036416, abs=1e-15)
    smooth.backward()


def test_gelu_node_gradient_matches_derivative() -> None:
    """gelu 节点的梯度 = gelu'(x)（对向量输入也成立）。"""
    inner = graph.variable((0.5, -1.0, 2.0))
    total = graph.sum_all(graph.gelu(inner))
    total.backward()
    expected = tuple(gradients.gelu_derivative(value) for value in (0.5, -1.0, 2.0))
    assert inner.grad == pytest.approx(expected, abs=1e-15)


def test_softmax_node_and_cross_entropy_node() -> None:
    """softmax_rows + cross_entropy 的节点：dZ = (p − onehot)/rows。"""
    logits = graph.variable(((1.0, 2.0),))
    probabilities = graph.softmax_rows(logits)
    assert probabilities.value[0] == pytest.approx(reference_softmax((1.0, 2.0)))
    loss = graph.cross_entropy(logits, (1,))
    expected = gradients.cross_entropy_grad((1.0, 2.0), 1)
    loss.backward()
    assert logits.grad[0] == pytest.approx(expected)
    assert loss.value == pytest.approx(0.3132616875182228, abs=1e-15)


def test_softmax_and_mse_node_guards() -> None:
    """softmax_rows 需要矩阵；mse / cross_entropy 的形状与标签都有护栏。"""
    with pytest.raises(errors.ShapeError, match="一行以上"):
        graph.softmax_rows(graph.constant((1.0, 2.0)))
    with pytest.raises(errors.ShapeError, match="矩阵作为预测"):
        graph.mse(graph.constant((1.0, 2.0)), ((1.0, 2.0),))
    with pytest.raises(errors.ShapeError, match="不一致"):
        graph.mse(graph.constant(((1.0, 2.0),)), ((1.0, 2.0, 3.0),))
    with pytest.raises(errors.ShapeError, match="一行以上"):
        graph.cross_entropy(graph.constant((1.0, 2.0)), (0,))
    with pytest.raises(errors.ShapeError, match="一一对应"):
        graph.cross_entropy(graph.constant(((1.0, 2.0),)), (0, 1))


def test_mse_node_gradient() -> None:
    """mse 节点：dP = 2(P − T)/N。"""
    prediction = graph.variable(((1.0, 2.0), (3.0, 4.0)))
    loss = graph.mse(prediction, ((1.5, 1.5), (4.0, 3.0)))
    loss.backward()
    assert prediction.grad == ((-0.25, 0.25), (-0.5, 0.5))
    assert loss.value == pytest.approx(0.625, abs=1e-15)


def test_zero_grad_and_grad_of() -> None:
    """zero_grad 清空整张图；不清零时第二次反向会**叠加**。"""
    node = graph.variable(3.0)
    total = graph.sum_all(node)
    total.backward()
    assert graph.grad_of(node) == 1.0
    total.backward()
    assert graph.grad_of(node) == 2.0  # 叠加：这就是"忘记清零"的后果
    total.zero_grad()
    assert graph.grad_of(node) == 0.0
    assert node.requires_grad is True


# --------------------------------------------------------------------------- 一层反向


def test_dense_backward_matches_hand() -> None:
    """一层反向的三块梯度都对手算值。"""
    grads = layers.dense_backward(
        ((1.0, 2.0), (3.0, 4.0)), (0.5, -0.5), ((1.0, 0.0),), ((1.0, 2.0),)
    )
    assert grads.grad_weight == ((1.0, 0.0), (2.0, 0.0))
    assert grads.grad_bias == (1.0, 2.0)
    assert grads.grad_inputs == ((7.0, 10.0),)


def test_affine_backward_is_dense_backward() -> None:
    """仿射反向就是一层反向（无激活的那一半）。"""
    args = (((1.0, 2.0),), (0.5,), ((1.0, 2.0),), ((3.0,),))
    assert layers.affine_backward(*args) == layers.dense_backward(*args)


def test_dense_backward_guards() -> None:
    """一层反向的形状护栏（行数 / 输出宽度 / 偏置长度 / 输入宽度）。"""
    with pytest.raises(errors.ShapeError, match="一一对应"):
        layers.dense_backward(((1.0, 2.0),), (0.0,), ((1.0, 2.0),), ((1.0,), (2.0,)))
    with pytest.raises(errors.ShapeError, match="输出宽度"):
        layers.dense_backward(((1.0, 2.0),), (0.0,), ((1.0, 2.0),), ((1.0, 2.0),))
    with pytest.raises(errors.ShapeError, match="bias"):
        layers.dense_backward(((1.0, 2.0),), (0.0, 0.0), ((1.0, 2.0),), ((1.0,),))
    with pytest.raises(errors.ShapeError, match="输入的列数"):
        layers.dense_backward(((1.0, 2.0),), (0.0,), ((1.0, 2.0, 3.0),), ((1.0,),))
    with pytest.raises(errors.ShapeError, match="非空"):
        layers.dense_backward((), (), ((1.0,),), ())
    with pytest.raises(errors.NumericError, match="非有限数"):
        layers.dense_backward(((1.0, math.nan),), (0.0,), ((1.0, 2.0),), ((1.0,),))


def test_gradient_summary_norms() -> None:
    """一层的三个范数：‖dW‖=3、‖db‖=2、‖dx‖=5。"""
    grads = types.DenseGradients(grad_weight=((1.0, 2.0),), grad_bias=(2.0,), grad_inputs=((3.0, 4.0),))
    summary = layers.gradient_summary(grads, index=2)
    assert summary.weight_norm == pytest.approx(math.sqrt(5.0))
    assert summary.bias_norm == pytest.approx(2.0)
    assert summary.input_norm == pytest.approx(5.0)
    assert summary.index == 2


def test_flatten_and_unflatten_round_trip() -> None:
    """参数压平与还原是一个往返；形状表与长度必须一致。"""
    params = (((1.0, 2.0), (3.0, 4.0)), (0.5, -0.5))
    flat, shapes = layers.flatten_parameters((params,))
    assert flat == (1.0, 2.0, 3.0, 4.0, 0.5, -0.5)
    assert shapes == ((2, 2, 2),)
    assert layers.parameter_count(shapes) == 6
    assert layers.unflatten_parameters(flat, shapes) == (params,)


def test_flatten_parameters_guards() -> None:
    """空参数表 / bias 长度不符都被拒。"""
    with pytest.raises(errors.ShapeError, match="至少要有一层"):
        layers.flatten_parameters(())
    with pytest.raises(errors.ShapeError, match="bias 长度"):
        layers.flatten_parameters(((((1.0, 2.0),), (0.5, 0.5)),))


def test_unflatten_parameters_guards() -> None:
    """长度不符 / 空形状表 / 非有限数都被拒。"""
    with pytest.raises(errors.ShapeError, match="形状表需要"):
        layers.unflatten_parameters((1.0, 2.0), ((2, 2, 2),))
    with pytest.raises(errors.ShapeError, match="形状表不能为空"):
        layers.unflatten_parameters((1.0,), ())
    with pytest.raises(errors.NumericError, match="非有限数"):
        layers.unflatten_parameters((math.inf,), ((1, 1, 1),))


def test_layer_shape_record() -> None:
    """ParameterPair / LayerShape 是类型别名（形状表长度为三）。"""
    shapes: tuple[layers.LayerShape, ...] = ((3, 2, 3),)
    assert layers.parameter_count(shapes) == 3 * 2 + 3


# --------------------------------------------------------------------------- 网络


def test_chain_forward_matches_day089_bitwise() -> None:
    """**跨天对账**：chain_forward 与 day089 的 ``mlp_forward`` **逐位**相同。"""
    from smart_research_agent.neural_basics.network import mlp_forward

    chain = network.build_chain(verify.MLP_SPEC)
    assert network.chain_forward(chain, verify.MLP_INPUTS) == mlp_forward(
        verify.MLP_SPEC, verify.MLP_INPUTS
    )


def test_chain_from_parameters_builds_layers() -> None:
    """用显式权重造层链；未知激活 / 长度不符 / 空链都被拒。"""
    params = network.build_parameters(verify.MLP_SPEC)
    activations = tuple(layer.activation for layer in verify.MLP_SPEC.layers)
    chain = network.chain_from_parameters(params, activations)
    assert network.chain_forward(chain, verify.MLP_INPUTS) == network.chain_forward(
        network.build_chain(verify.MLP_SPEC), verify.MLP_INPUTS
    )
    with pytest.raises(errors.ShapeError, match="一一对应"):
        network.chain_from_parameters(params, activations[:1])
    with pytest.raises(errors.ShapeError, match="至少要有一层"):
        network.chain_from_parameters((), ())
    with pytest.raises(errors.ParameterError, match="未知的激活"):
        network.chain_from_parameters(params, ("relu", "magic"))


def test_forward_cache_validates_depth() -> None:
    """ForwardCache：零层或三个序列长度不一致都抛 BackwardError。"""
    with pytest.raises(errors.BackwardError, match="至少要有一层"):
        network.ForwardCache(layers=(), inputs=(), pre_activations=(), outputs=())
    chain = network.build_chain(verify.MLP_SPEC)
    with pytest.raises(errors.BackwardError, match="长度对不上"):
        network.ForwardCache(
            layers=chain,
            inputs=(verify.MLP_INPUTS,),
            pre_activations=(None, None),
            outputs=(verify.MLP_INPUTS, verify.MLP_INPUTS),
        )


def test_chain_forward_guards() -> None:
    """空层链与空输入都被拒。"""
    with pytest.raises(errors.ShapeError, match="层链不能为空"):
        network.chain_forward((), verify.MLP_INPUTS)
    with pytest.raises(errors.ShapeError, match="输入不能为空"):
        network.chain_forward(network.build_chain(verify.MLP_SPEC), ())
    with pytest.raises(errors.ShapeError, match="层链不能为空"):
        network.chain_forward_with_cache((), verify.MLP_INPUTS)
    with pytest.raises(errors.ShapeError, match="输入不能为空"):
        network.chain_forward_with_cache(network.build_chain(verify.MLP_SPEC), ())


def test_activation_backward_branches() -> None:
    """激活反向的三个分支：恒等 / softmax（读输出）/ 逐元素（读输入）。"""
    grad = ((1.0, 2.0),)
    assert network.activation_backward(None, None, ((0.5, 0.5),), grad) == grad
    softmax_grad = network.activation_backward("softmax", None, ((0.5, 0.5),), grad)
    assert softmax_grad == gradients.softmax_backward_rows(((0.5, 0.5),), grad)
    elementwise = network.activation_backward("relu", ((-1.0, 2.0),), ((0.0, 2.0),), grad)
    assert elementwise == ((0.0, 2.0),)
    with pytest.raises(errors.BackwardError, match="缓存里没有它"):
        network.activation_backward("gelu", None, ((0.0,),), ((1.0,),))


def test_mlp_backward_shape_guard() -> None:
    """回传梯度与网络输出形状不符抛 ShapeError。"""
    cache = network.mlp_forward_with_cache(verify.MLP_SPEC, verify.MLP_INPUTS)
    with pytest.raises(errors.ShapeError, match="形状不一致"):
        network.mlp_backward(cache, ((1.0,),))


def test_mlp_loss_gradients_and_trace() -> None:
    """MLP 的 MSE 反向：逐层账、参数梯度、压平梯度都自洽。"""
    cache = network.mlp_forward_with_cache(verify.MLP_SPEC, verify.MLP_INPUTS)
    trace = network.mlp_loss_gradients(cache, verify.MLP_TARGETS)
    summaries = trace.layer_summaries()
    assert len(summaries) == 2
    assert summaries[0].weight_shape == (5, 3)
    pairs = trace.parameter_gradients()
    assert len(pairs) == 2 and len(pairs[0]) == 2
    flat = trace.flat_parameter_gradients()
    flat_again, shapes = trace.flatten()
    assert flat == flat_again
    assert layers.parameter_count(shapes) == len(flat) == 32
    assert network.gradient_norm((3.0, 4.0)) == pytest.approx(5.0)


def test_backward_trace_rejects_empty() -> None:
    """BackwardTrace 至少要有一层的梯度。"""
    with pytest.raises(errors.BackwardError, match="至少要有一层"):
        network.BackwardTrace(layer_gradients=(), grad_inputs=((1.0,),))


def test_softmax_loss_gradients() -> None:
    """以交叉熵为损失的反向：外层 softmax + 逐行 CE。"""
    spec = _mlp_with_softmax()
    cache = network.mlp_forward_with_cache(spec, verify.MLP_INPUTS)
    trace = network.softmax_loss_gradients(cache, (0, 1))
    assert len(trace.layer_gradients) == 2
    assert trace.grad_inputs


def _mlp_with_softmax() -> object:
    """构造一个以 softmax 结尾的两层网络（测试用）。"""
    from smart_research_agent.neural_basics.types import DenseSpec, MLPSpec

    return MLPSpec(
        layers=(
            DenseSpec(3, 4, activation="gelu", init="xavier", seed=51),
            DenseSpec(4, 3, activation="softmax", init="xavier", seed=52),
        )
    )


def test_ffn_forward_matches_encoder_decoder_bitwise() -> None:
    """**跨天对账**：本包前馈的前向与 ``feed_forward`` 逐位相同（relu 与 gelu）。"""
    from smart_research_agent.encoder_decoder.layers import feed_forward
    from smart_research_agent.encoder_decoder.types import FFNWeights
    from smart_research_agent.neural_basics.network import build_ffn_params

    params = build_ffn_params(hidden=verify.FFN_HIDDEN, seed=verify.FFN_SEED)
    real = FFNWeights(w_in=params.w_in, b_in=params.b_in, w_out=params.w_out, b_out=params.b_out)
    for activation in ("relu", "gelu"):
        output, pre, hidden = network.ffn_forward(verify.FFN_INPUTS, real, activation=activation)
        theirs, _cache = feed_forward(verify.FFN_INPUTS, real, activation=activation)
        assert output == theirs
        assert len(pre) == len(hidden) == len(verify.FFN_INPUTS)


def test_ffn_backward_matches_encoder_decoder_bitwise() -> None:
    """**跨天对账**：五块梯度与 ``feed_forward_backward`` **逐位**相同。"""
    from smart_research_agent.encoder_decoder.layers import feed_forward, feed_forward_backward
    from smart_research_agent.encoder_decoder.types import FFNWeights
    from smart_research_agent.neural_basics.network import build_ffn_params

    params = build_ffn_params(hidden=verify.FFN_HIDDEN, seed=verify.FFN_SEED)
    real = FFNWeights(w_in=params.w_in, b_in=params.b_in, w_out=params.w_out, b_out=params.b_out)
    for activation in ("relu", "gelu"):
        ours = network.ffn_backward(
            verify.FFN_INPUTS, real, activation=activation, grad_output=verify.FFN_GRAD_OUTPUT
        )
        _output, cache = feed_forward(verify.FFN_INPUTS, real, activation=activation)
        theirs = feed_forward_backward(cache, verify.FFN_GRAD_OUTPUT)
        assert ours.grad_w_in == theirs.grad_w_in
        assert ours.grad_b_in == theirs.grad_b_in
        assert ours.grad_w_out == theirs.grad_w_out
        assert ours.grad_b_out == theirs.grad_b_out
        assert ours.grad_inputs == theirs.grad_inputs


def test_ffn_guards() -> None:
    """前馈的两道护栏：未知激活、缺字段、形状不符。"""
    from smart_research_agent.encoder_decoder.types import FFNWeights
    from smart_research_agent.neural_basics.network import build_ffn_params

    params = build_ffn_params(hidden=verify.FFN_HIDDEN, seed=verify.FFN_SEED)
    real = FFNWeights(w_in=params.w_in, b_in=params.b_in, w_out=params.w_out, b_out=params.b_out)
    with pytest.raises(errors.ParameterError, match="只能是"):
        network.ffn_forward(verify.FFN_INPUTS, real, activation="softmax")
    with pytest.raises(errors.ParameterError, match="四个属性"):
        network.ffn_forward(verify.FFN_INPUTS, object(), activation="relu")
    with pytest.raises(errors.ShapeError, match="不能为空"):
        network.ffn_forward((), real, activation="relu")
    with pytest.raises(errors.ShapeError, match="列数"):
        network.ffn_forward(((1.0, 2.0),), real, activation="relu")
    with pytest.raises(errors.ShapeError, match="形状不一致"):
        network.ffn_backward(verify.FFN_INPUTS, real, activation="relu", grad_output=((1.0,),))


def test_spec_shapes_and_build_parameters() -> None:
    """形状表与参数：层数、形状、可复现。"""
    shapes = network.spec_shapes(verify.MLP_SPEC)
    assert shapes == ((5, 3, 5), (2, 5, 2))
    first = network.build_parameters(verify.MLP_SPEC)
    second = network.build_parameters(verify.MLP_SPEC)
    assert first == second


def test_parameter_objective_losses_and_guard() -> None:
    """参数目标函数：MSE 与交叉熵两条分支；未知损失被拒。"""
    objective = network.parameter_objective(verify.MLP_SPEC, verify.MLP_INPUTS, verify.MLP_TARGETS)
    params, _ = layers.flatten_parameters(network.build_parameters(verify.MLP_SPEC))
    assert objective(params) == pytest.approx(train.evaluate_loss(verify.MLP_SPEC, verify.MLP_INPUTS, verify.MLP_TARGETS))
    cross = network.parameter_objective(
        verify.MLP_SPEC, verify.MLP_INPUTS, ((0.0, 1.0), (1.0, 0.0)), loss=types.LOSS_CROSS_ENTROPY
    )
    assert math.isfinite(cross(params))
    with pytest.raises(errors.ParameterError, match="未知的损失"):
        network.parameter_objective(verify.MLP_SPEC, verify.MLP_INPUTS, verify.MLP_TARGETS, loss="hinge")


# --------------------------------------------------------------------------- 训练


def test_train_mlp_reduces_loss() -> None:
    """一次解析反向训练真的让损失下降，且每一步都记录在案。"""
    report = train.train_mlp(verify.MLP_SPEC, verify.MLP_INPUTS, verify.MLP_TARGETS, steps=30, learning_rate=0.05)
    assert report.steps == 30
    assert len(report.losses) == 31
    assert report.final_loss < report.initial_loss
    assert report.improvement > 0.5
    assert report.monotone() is True
    assert report.optimizer_name == train.DEFAULT_OPTIMIZER == "sgd"
    assert all(norm >= 0.0 for norm in report.gradient_norms)
    assert report.step_line(0).startswith("step   0")
    assert "lr" in report.step_line(1)
    assert report.to_dict()["steps"] == 30


def test_train_mlp_accepts_other_optimizers() -> None:
    """优化器可以换（day074 的 adam），且轨迹仍单调下降。"""
    report = train.train_mlp(
        verify.MLP_SPEC, verify.MLP_INPUTS, verify.MLP_TARGETS, steps=20, learning_rate=0.05,
        optimizer_name="adam",
    )
    assert report.final_loss < report.initial_loss
    assert report.optimizer_name == "adam"


def test_train_mlp_guards() -> None:
    """步数非法 / 训练集为空都被拒。"""
    with pytest.raises(errors.ParameterError, match="steps"):
        train.train_mlp(verify.MLP_SPEC, verify.MLP_INPUTS, verify.MLP_TARGETS, steps=0)
    with pytest.raises(errors.ParameterError, match="不能为空"):
        train.train_mlp(verify.MLP_SPEC, (), verify.MLP_TARGETS, steps=1)


def test_train_mlp_step_error_on_non_finite_parameters() -> None:
    """注入一个"把参数推成 inf"的优化器 ⇒ StepError（而不是让它静默变成 nan）。"""
    from smart_research_agent.math_foundations.optim import Optimizer

    class Overflowing(Optimizer):
        name = "sgd"

        def step(self, params: object, grads: object) -> tuple[float, ...]:
            self.step_count += 1
            return tuple(math.inf for _ in params)  # type: ignore[call-overload]

    with pytest.raises(errors.StepError, match="非有限数"):
        train.train_mlp(
            verify.MLP_SPEC,
            verify.MLP_INPUTS,
            verify.MLP_TARGETS,
            steps=1,
            optimizer=Overflowing(0.1),
        )


def test_train_report_validates_lengths() -> None:
    """TrainReport：损失比学习率多一个、梯度范数与学习率同长。"""
    good = train.TrainReport(
        losses=(1.0, 0.5), learning_rates=(0.1,), gradient_norms=(0.2,)
    )
    assert good.summary_line().startswith("1 步")
    assert good.improvement == pytest.approx(0.5)
    with pytest.raises(errors.NumericError, match="少一个"):
        train.TrainReport(losses=(1.0, 0.5), learning_rates=(), gradient_norms=())
    with pytest.raises(errors.NumericError, match="每一步一个"):
        train.TrainReport(losses=(1.0, 0.5), learning_rates=(0.1,), gradient_norms=())
    with pytest.raises(errors.ParameterError, match="超出范围"):
        good.step_line(5)


def test_train_report_zero_initial_loss() -> None:
    """初始损失为 0 时下降比例记 0.0（不是除零）。"""
    report = train.TrainReport(losses=(0.0, 0.0), learning_rates=(0.1,), gradient_norms=(0.0,))
    assert report.improvement == 0.0
    assert report.monotone() is True


def test_evaluate_loss_and_train_loss() -> None:
    """起点读数与损失名。"""
    value = train.evaluate_loss(verify.MLP_SPEC, verify.MLP_INPUTS, verify.MLP_TARGETS)
    assert value > 0.0
    assert train.TRAIN_LOSS == types.LOSS_MSE


# --------------------------------------------------------------------------- 性质


def test_gradient_check_semantics() -> None:
    """GradientCheck：有 upper_bound 走"≤"、没有时走"=="；负界 / 非有限读数被拒。"""
    assert verify.GradientCheck("a", "l", "r", reading=0.5, expected=0.0, upper_bound=1.0).passed
    assert not verify.GradientCheck("a", "l", "r", reading=2.0, expected=0.0, upper_bound=1.0).passed
    assert verify.GradientCheck("a", "l", "r", reading=1.0, expected=1.0, exact=True).passed
    assert not verify.GradientCheck("a", "l", "r", reading=1.0, expected=1.0 + 1e-9, exact=True).passed
    assert verify.GradientCheck("a", "l", "r", reading=1.0 + 1e-13, expected=1.0, exact=False).passed
    with pytest.raises(errors.NumericError, match="有限"):
        verify.GradientCheck("a", "l", "r", reading=math.inf, expected=0.0)
    with pytest.raises(errors.NumericError, match="上界"):
        verify.GradientCheck("a", "l", "r", reading=0.0, expected=0.0, upper_bound=-1.0)


def test_property_outcome_and_report() -> None:
    """PropertyOutcome：不适用 + 通过 是矛盾；PropertyReport 只看适用项。"""
    outcome = verify.PropertyOutcome("x", applicable=False, passed=False, evidence=("跳过",))
    assert outcome.line().startswith("[不适用]")
    with pytest.raises(errors.NumericError, match="不适用"):
        verify.PropertyOutcome("x", applicable=False, passed=True)
    skipped = verify.PropertyOutcome("x", applicable=False, passed=False)
    passing = verify.PropertyOutcome("y", applicable=True, passed=True)
    report = verify.PropertyReport(outcomes=(skipped, passing))
    assert report.ok is True
    assert len(report.applicable) == 1
    report.require_ok()  # 不抛


def test_property_report_require_ok_raises_gradient_error() -> None:
    """**第一次有人真的抛 GradientError**：有适用项失败时 require_ok 抛它。"""
    failing = verify.PropertyOutcome("z", applicable=True, passed=False, evidence=("坏",))
    report = verify.PropertyReport(outcomes=(failing,))
    assert report.ok is False
    with pytest.raises(errors.GradientError, match="未全部通过"):
        report.require_ok()


def test_check_all_matches_property_names() -> None:
    """七条性质的名字与 ``types.BACKPROP_PROPERTIES`` 逐一对应。"""
    report = verify.check_all()
    assert report.ok is True
    assert {outcome.name for outcome in report.outcomes} == set(types.BACKPROP_PROPERTIES)
    assert report.to_dict()["counts"]["total"] == 7
    assert len(report.lines()) == 7


def test_individual_property_checks_pass() -> None:
    """七条性质各自单独跑一遍也都通过（返回值不是一个常量）。"""
    outcomes = (
        verify.evaluate_derivatives_vs_numerical(),
        verify.check_relu_subgradient_is_zero(),
        verify.check_softmax_jacobian_matches_numerical(),
        verify.check_softmax_jvp_avoids_matrix(),
        verify.check_cross_entropy_gradient_is_p_minus_onehot(),
        verify.check_mlp_backward_matches_numerical(),
        verify.check_ffn_backward_matches_encoder_decoder(),
    )
    assert all(outcome.applicable for outcome in outcomes)
    assert all(outcome.passed for outcome in outcomes)


def test_property_readings_are_the_expected_zeros() -> None:
    """逐位 / 整数判据的三条读数必须恰好是 0。"""
    assert verify.check_relu_subgradient_is_zero().cross_check.reading == 0.0
    assert verify.check_ffn_backward_matches_encoder_decoder().cross_check.reading == 0.0
    bounded = verify.evaluate_derivatives_vs_numerical().cross_check
    assert bounded is not None and bounded.reading <= verify.DERIVATIVE_TOLERANCE
    jvp = verify.check_softmax_jvp_avoids_matrix().cross_check
    assert jvp is not None and jvp.reading <= verify.IDENTITY_TOLERANCE
    mlp = verify.check_mlp_backward_matches_numerical().cross_check
    assert mlp is not None and mlp.reading <= verify.NETWORK_TOLERANCE


def test_difference_resolution_and_scaled_gap() -> None:
    """分辨率下限与相对误差判据都由公式给出（不是拍一个数）。"""
    assert verify.difference_resolution(magnitude=1.0) == pytest.approx(1.1102230246251565e-10)
    assert verify.difference_resolution(magnitude=100.0) > verify.difference_resolution(magnitude=1.0)
    with pytest.raises(errors.NumericError, match="量级"):
        verify.difference_resolution(magnitude=math.inf)
    with pytest.raises(errors.ParameterError, match="步长"):
        verify.difference_resolution(magnitude=1.0, step=0.0)
    assert verify.max_scaled_gap((1.0,), (1.0, 2.0)) == math.inf
    assert verify.max_scaled_gap((math.inf,), (1.0,)) == math.inf
    assert verify.max_scaled_gap((2.0,), (1.0,)) == pytest.approx(1.0)
    assert verify.numerical_gradient(lambda point: point[0] * point[0], (3.0,))[0] == pytest.approx(6.0, abs=1e-6)


def test_verify_constants_are_sane() -> None:
    """性质用的样本与常量自洽（长度、上界、种子）。"""
    assert 0.0 not in verify.DERIVATIVE_POINTS  # 避开 relu / leaky_relu 的拐点
    assert len(verify.CE_CASES) == 4
    assert len(verify.JVP_CASES) == 3
    assert verify.MLP_SPEC.widths == (3, 5, 2)
    assert len(verify.FFN_GRAD_OUTPUT) == 2 and len(verify.FFN_GRAD_OUTPUT[0]) == 4
    assert verify.IDENTITY_TOLERANCE < verify.DERIVATIVE_TOLERANCE < verify.NETWORK_TOLERANCE


# --------------------------------------------------------------------------- 表


def test_derivative_rows_cover_all() -> None:
    """导数表：六个激活，每一行都带公式与读数。"""
    rows = study.derivative_rows()
    assert len(rows) == 6
    assert all(row.formula and row.reading for row in rows)
    assert rows[0].line().startswith("relu")
    softmax_row = rows[-1]
    assert softmax_row.name == "softmax"
    assert "每行和" in softmax_row.reading


def test_dense_layer_gradients_row() -> None:
    """一层反向表：三块范数都非负，权重形状 (4, 3)。"""
    row = study.dense_layer_gradients()
    assert row.weight_shape == (4, 3)
    assert row.weight_norm > 0.0 and row.bias_norm > 0.0 and row.input_norm > 0.0


def test_network_layer_rows_cover_depth() -> None:
    """网络反向表：3→5→2 的两层，逐层都有一行。"""
    rows = study.network_layer_rows()
    assert len(rows) == 2
    assert rows[0].weight_shape == (5, 3)
    assert rows[1].weight_shape == (2, 5)


def test_check_rows_and_property_rows() -> None:
    """校验表与性质表：各七行，且与 check_all 的结论一致。"""
    checks = study.check_rows()
    assert len(checks) == 7
    assert all(row.left and row.right for row in checks)
    rows = study.property_rows()
    report = verify.check_all()
    assert len(rows) == 7
    assert [row.passed for row in rows] == [outcome.passed for outcome in report.outcomes]


def test_study_lines_and_notes() -> None:
    """study_lines：五节；note_lines：十条（顺序即写入顺序）。"""
    lines = study.study_lines()
    assert sum(1 for line in lines if line.startswith("== ")) == 5
    notes = study.note_lines()
    assert len(notes) == 10
    assert notes[0].startswith(" 1.")
    assert len(study.note_lines(limit=3)) == 3
    assert study.PROPERTY_NAMES == types.BACKPROP_PROPERTIES


# --------------------------------------------------------------------------- 包入口


def test_package_all_is_sorted_and_resolvable() -> None:
    """``__all__`` 是字母序，且每一个名字都能解析（不会被同名子模块遮掉）。"""
    import smart_research_agent.backprop as package

    assert package.__all__ == sorted(package.__all__)
    assert all(hasattr(package, name) for name in package.__all__)
    submodules = {
        "errors",
        "types",
        "gradients",
        "graph",
        "layers",
        "network",
        "train",
        "verify",
        "study",
    }
    assert not (submodules & set(package.__all__))


def test_package_all_is_the_union_of_modules() -> None:
    """``__all__`` 就是九个模块各自公开名单的并集（"一个量只写一遍"）。"""
    import smart_research_agent.backprop as package

    union = set()
    for name in ("errors", "types", "gradients", "graph", "layers", "network", "train", "verify", "study"):
        union |= set(getattr(package, name).__all__)
    assert set(package.__all__) == union


def test_key_symbols_are_exported() -> None:
    """几个最该被 `from backprop import X` 拿到的名字确实在名单里。"""
    import smart_research_agent.backprop as package

    for name in (
        "dense_backward",
        "mlp_backward",
        "ffn_backward",
        "train_mlp",
        "check_all",
        "GradientError",
        "Node",
        "dense",
        "TrainReport",
    ):
        assert name in package.__all__
        assert getattr(package, name) is not None
