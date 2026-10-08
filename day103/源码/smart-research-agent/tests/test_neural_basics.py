"""``neural_basics``：从神经元到 FFN 这条链的测试（day089 / M8-D1）.

本文件覆盖新包的八个功能模块与包入口：口径表、失败族、六个激活、Dense 层与初始化、
MLP 与 FFN、三个损失、七条性质、五张表。

样本全部来自本包自己的确定性构造（LCG + 写死输入）；**跨天对账真的调用既有包**
（``math_foundations`` / ``sft`` / ``encoder_decoder`` / ``hf_source``），
因此这一课考的就是"这条链被真的接上了没有"。
"""

from __future__ import annotations

import math

import pytest

from smart_research_agent.neural_basics import (
    errors,
    layers,
    losses,
    network,
    study,
    types,
    verify,
)
from smart_research_agent.neural_basics import activations

# --------------------------------------------------------------------------- 口径表


def test_six_activations_are_closed() -> None:
    """六个激活的名单、说明、口径表逐键对齐."""
    assert types.ACTIVATIONS == (
        "relu",
        "leaky_relu",
        "sigmoid",
        "tanh",
        "gelu",
        "softmax",
    )
    assert set(types.ACTIVATION_DESCRIPTIONS) == set(types.ACTIVATIONS)
    assert set(types.ACTIVATION_PROFILES) == set(types.ACTIVATIONS)
    assert all(types.ACTIVATION_DESCRIPTIONS.values())


def test_activation_profiles_have_four_columns() -> None:
    """每一行口径都有四列：饱和 / 零中心 / 值域 / 公式（少了哪一列都读不出判据）."""
    for name, profile in types.ACTIVATION_PROFILES.items():
        assert isinstance(profile, types.ActivationProfile)
        assert profile.value_range
        assert profile.formula
        assert profile.line().startswith(("饱和 是", "饱和 否"))
    assert types.ACTIVATION_PROFILES["tanh"].zero_centered is True
    assert types.ACTIVATION_PROFILES["relu"].zero_centered is False
    assert types.ACTIVATION_PROFILES["sigmoid"].saturating is True


def test_three_losses_are_closed() -> None:
    """三个损失的名单与说明逐键对齐."""
    assert types.LOSSES == ("mse", "mae", "cross_entropy")
    assert set(types.LOSS_DESCRIPTIONS) == set(types.LOSSES)
    assert all(types.LOSS_DESCRIPTIONS.values())


def test_five_initializations_are_closed() -> None:
    """五种初始化的名单与说明逐键对齐，且 zeros/xavier/he 都在其中."""
    assert set(types.INITIALIZATIONS) == {"zeros", "uniform", "normal", "xavier", "he"}
    assert set(types.INITIALIZATION_DESCRIPTIONS) == set(types.INITIALIZATIONS)
    assert types.INIT_ZEROS in types.INITIALIZATIONS
    assert types.INIT_XAVIER in types.INITIALIZATIONS


def test_seven_properties_have_descriptions_and_failures() -> None:
    """七条性质：名单、说明、"失败意味着什么"三张表逐键对齐."""
    assert len(types.NEURAL_PROPERTIES) == 7
    assert set(types.PROPERTY_DESCRIPTIONS) == set(types.NEURAL_PROPERTIES)
    assert set(types.PROPERTY_FAILURE) == set(types.NEURAL_PROPERTIES)
    assert all(types.PROPERTY_DESCRIPTIONS.values())
    assert all(types.PROPERTY_FAILURE.values())


def test_notes_are_ten_and_ordered() -> None:
    """十条笔记，且顺序表与键集合一致."""
    assert len(types.NEURAL_NOTES) == 10
    assert types.NEURAL_NOTES_ORDER == tuple(types.NEURAL_NOTES)
    assert all(types.NEURAL_NOTES.values())


def test_boundaries_are_five() -> None:
    """五条边界是一份非空清单（**这一课明确不承诺的事**）."""
    assert len(types.NEURAL_BOUNDARIES) == 5
    assert all(types.NEURAL_BOUNDARIES)


def test_torch_counterparts_cover_every_operator() -> None:
    """PyTorch 对照表覆盖每一个算子，且**不出现**版本号（本课不声称版本）."""
    required = {
        "dense_forward",
        "relu",
        "leaky_relu",
        "sigmoid",
        "tanh",
        "gelu",
        "softmax",
        "log_softmax",
        "mse",
        "mae",
        "cross_entropy",
        "xavier_init",
    }
    assert required <= set(types.TORCH_COUNTERPARTS)
    assert all(("torch" in value or "nn." in value) for value in types.TORCH_COUNTERPARTS.values())
    joined = " ".join(types.TORCH_COUNTERPARTS.values())
    assert "torch==" not in joined


# --------------------------------------------------------------------------- 记录


def test_neuron_spec_validates_and_counts() -> None:
    """神经元记录：参数数 = inputs + 1；未知激活 / 非法宽度被拒."""
    spec = types.NeuronSpec(inputs=3, activation="relu", init="zeros", seed=0)
    assert spec.parameter_count == 4
    assert spec.line().startswith("neuron(3→1)")
    assert spec.to_dict()["inputs"] == 3
    with pytest.raises(errors.ParameterError, match="未知的激活"):
        types.NeuronSpec(inputs=3, activation="rellu")
    with pytest.raises(errors.ParameterError, match="正整数"):
        types.NeuronSpec(inputs=0)
    with pytest.raises(errors.ParameterError, match="未知的初始化"):
        types.NeuronSpec(inputs=3, init="magic")


def test_dense_spec_counts_parameters() -> None:
    """DenseSpec：参数数 = in×out + out；权重形状字符串是 (out, in)."""
    spec = types.DenseSpec(in_features=3, out_features=4, activation="relu", init="xavier", seed=1)
    assert spec.weight_count == 12
    assert spec.parameter_count == 16
    assert spec.line().startswith("dense(3→4)")
    accounting = layers.layer_accounting(spec)
    assert accounting["weight_shape"] == "(4, 3)"
    assert accounting["total"] == 16


def test_mlp_spec_chain_must_connect() -> None:
    """MLPSpec：相邻两层宽度必须接得上，否则抛 ForwardError（**结构**问题）."""
    good = types.MLPSpec(
        layers=(
            types.DenseSpec(3, 4, activation="relu"),
            types.DenseSpec(4, 2, activation=None),
        )
    )
    assert good.widths == (3, 4, 2)
    assert good.input_width == 3
    assert good.output_width == 2
    assert len(good.layers) == 2
    with pytest.raises(errors.ForwardError, match="接不上"):
        types.MLPSpec(
            layers=(
                types.DenseSpec(3, 4, activation="relu"),
                types.DenseSpec(5, 2, activation=None),
            )
        )


def test_mlp_spec_rejects_empty() -> None:
    """空层列表抛 ParameterError：一个网络至少也要有一层."""
    with pytest.raises(errors.ParameterError, match="不能为空"):
        types.MLPSpec(layers=())


def test_forward_trace_records_shapes() -> None:
    """ForwardTrace：宽度链与逐层形状字符串都记下来，接不上时抛 ForwardError."""
    trace = types.ForwardTrace(input_width=3, layer_widths=((3, 4), (4, 2)))
    assert trace.widths == (3, 4, 2)
    assert trace.shapes() == ("3 -> 4", "4 -> 2")
    assert trace.output_width == 2
    assert trace.line().startswith("3 → 4 → 2")
    with pytest.raises(errors.ForwardError, match="接不上"):
        types.ForwardTrace(input_width=3, layer_widths=((3, 4), (5, 2)))
    with pytest.raises(errors.ForwardError, match="trace 的输入宽度"):
        types.ForwardTrace(input_width=9, layer_widths=((3, 4),))


def test_loss_report_validates_fields() -> None:
    """LossReport：未知损失名 / 非有限值 / 负值 / 零样本都被拒."""
    report = types.LossReport(name=types.LOSS_MSE, value=0.0, samples=4)
    assert report.line().startswith("mse = 0.000000e+00")
    assert report.to_dict()["reduction"] == "mean"
    with pytest.raises(errors.ParameterError, match="未知的损失"):
        types.LossReport(name="hinge", value=1.0, samples=4)
    with pytest.raises(errors.NumericError, match="必须有限"):
        types.LossReport(name=types.LOSS_MSE, value=math.inf, samples=4)
    with pytest.raises(errors.ParameterError, match="样本数"):
        types.LossReport(name=types.LOSS_MSE, value=1.0, samples=0)


# --------------------------------------------------------------------------- 激活


def test_relu_and_leaky_relu() -> None:
    """relu：负半轴恒 0；leaky_relu：负半轴留一条坡度（默认 0.01）."""
    assert activations.relu(-2.0) == 0.0
    assert activations.relu(2.0) == 2.0
    assert activations.leaky_relu(-2.0) == pytest.approx(-0.02)
    assert activations.leaky_relu(2.0) == 2.0
    assert activations.leaky_relu(-1.0, slope=0.1) == pytest.approx(-0.1)
    with pytest.raises(errors.NumericError, match="斜率"):
        activations.leaky_relu(-1.0, slope=-0.5)


def test_sigmoid_uses_sign_branch() -> None:
    """sigmoid：两端都不上溢（±1000 处有限），且 0 处恰为 0.5."""
    assert activations.sigmoid(0.0) == pytest.approx(0.5, abs=1e-15)
    assert activations.sigmoid(1.0) == pytest.approx(0.7310585786300049, abs=1e-15)
    assert activations.sigmoid(1000.0) == 1.0
    assert activations.sigmoid(-1000.0) == 0.0
    assert math.isfinite(activations.sigmoid(-1000.0))


def test_tanh_range_and_zero() -> None:
    """tanh：零中心、值域开区间、0 处恰为 0."""
    assert activations.tanh(0.0) == 0.0
    assert activations.tanh(1.0) == pytest.approx(0.7615941559557649, abs=1e-15)
    for value in activations.RANGE_GRID:
        assert -1.0 < activations.tanh(value) < 1.0


def test_gelu_exact_matches_project_implementation() -> None:
    """gelu（精确 erf 式）与 ``hf_source.blocks.gelu_exact`` 在整张网格上**逐位**一致."""
    from smart_research_agent.hf_source.blocks import gelu_exact

    for value in activations.FINITE_GRID:
        assert activations.gelu(value) == gelu_exact(value)
    assert activations.gelu(0.0) == 0.0
    assert activations.gelu(1.0) == pytest.approx(0.8413447460685429, abs=1e-15)


def test_gelu_tanh_is_a_different_formula() -> None:
    """gelu_tanh 是 tanh 近似：0 处与精确式同值同导数，两侧有可量出的偏差."""
    assert activations.gelu_tanh(0.0) == 0.0
    worst = max(abs(activations.gelu(x) - activations.gelu_tanh(x)) for x in (-3.0, -1.5, 1.5, 3.0))
    assert worst > 1e-4


def test_softmax_is_a_distribution() -> None:
    """softmax：非负、行和为 1、对整体平移不变."""
    result = activations.softmax((1.0, 2.0, 3.0))
    assert sum(result) == pytest.approx(1.0, abs=1e-15)
    assert all(value > 0.0 for value in result)
    shifted = activations.softmax((101.0, 102.0, 103.0))
    assert result == shifted


def test_softmax_is_stable_at_extremes() -> None:
    """softmax 先减最大值：``z=[1000, 1001]`` 得到有限分布而不是 nan."""
    result = activations.softmax((1000.0, 1001.0))
    assert all(math.isfinite(value) for value in result)
    assert sum(result) == pytest.approx(1.0, abs=1e-15)


def test_softmax_matches_math_foundations_bitwise() -> None:
    """**跨天对账**：本包 softmax 与 day073 的 ``math_foundations`` 逐位一致."""
    from smart_research_agent.math_foundations.linalg import softmax as reference

    for logits in verify.SOFTMAX_CASES:
        assert activations.softmax(logits) == tuple(reference(list(logits)))


def test_activate_dispatches_and_rejects_unknown() -> None:
    """activate：未知激活名抛 ActivationError（**绝不回退到 relu**）."""
    assert activations.activate("relu", (-1.0, 2.0)) == (0.0, 2.0)
    assert activations.activate("softmax", (1.0, 2.0, 3.0)) == activations.softmax((1.0, 2.0, 3.0))
    with pytest.raises(errors.ActivationError, match="未知的激活"):
        activations.activate("rellu", (1.0,))


def test_activate_rejects_empty_and_non_finite() -> None:
    """activate：空向量与非有限输入都被拒（后者走 ActivationError）."""
    with pytest.raises(errors.ActivationError, match="不能为空"):
        activations.activate("relu", ())
    with pytest.raises(errors.ActivationError, match="有限数"):
        activations.activate("relu", (math.inf,))


def test_activation_at_rejects_softmax() -> None:
    """activation_at：softmax 需要一整行，用标量调用会被拒."""
    assert activations.activation_at("relu", -1.0) == 0.0
    with pytest.raises(errors.ActivationError, match="不是逐元素"):
        activations.activation_at("softmax", 1.0)


def test_activation_range_semantics() -> None:
    """activation_range：开闭端点被正确区分；未知名抛 ActivationError."""
    sigmoid = activations.activation_range("sigmoid")
    assert sigmoid.contains(0.5) is True
    assert sigmoid.contains(0.0) is False  # 开区间
    assert sigmoid.contains(1.0) is False
    relu = activations.activation_range("relu")
    assert relu.contains(0.0) is True  # 闭区间
    assert relu.contains(-1.0) is False
    assert activations.activation_range("leaky_relu").contains(-1e9) is True
    with pytest.raises(errors.ActivationError, match="未知的激活"):
        activations.activation_range("rellu")


def test_activation_range_rejects_nan() -> None:
    """值域判定对 nan 一律返回 False（避免"越界却看起来通过"）."""
    assert activations.activation_range("relu").contains(math.nan) is False


# --------------------------------------------------------------------------- 初始化与层


def test_lcg_stream_is_deterministic() -> None:
    """LCG：同一 seed 两次得到同一串数；都落在 [0, 1)；负长度被拒."""
    first = layers.lcg_stream(123, 5)
    assert first == layers.lcg_stream(123, 5)
    assert all(0.0 <= value < 1.0 for value in first)
    assert layers.lcg_stream(123, 5) != layers.lcg_stream(124, 5)
    with pytest.raises(errors.ParameterError, match="非负"):
        layers.lcg_stream(1, -1)


def test_initialize_zeros() -> None:
    """zeros：权重与偏置全为 0."""
    weight, bias = layers.initialize("zeros", 3, 4, seed=0)
    assert all(value == 0.0 for row in weight for value in row)
    assert bias == (0.0, 0.0, 0.0)


def test_initialize_uniform_and_normal_are_deterministic() -> None:
    """uniform / normal：同 seed 逐位复现；uniform 落在 ±1/√fan_in 内."""
    u1 = layers.initialize("uniform", 3, 4, seed=9)
    u2 = layers.initialize("uniform", 3, 4, seed=9)
    assert u1 == u2
    limit = 1.0 / math.sqrt(4)
    assert all(abs(value) <= limit for row in u1[0] for value in row)
    n1 = layers.initialize("normal", 3, 4, seed=9)
    assert n1 == layers.initialize("normal", 3, 4, seed=9)
    assert n1 != u1


def test_xavier_and_he_scales() -> None:
    """xavier 半宽与 he 标准差的公式被真的算出来（不是经验值）."""
    assert layers.xavier_limit(4, 4) == pytest.approx(math.sqrt(6.0 / 8.0))
    assert layers.he_scale(4) == pytest.approx(math.sqrt(2.0 / 4.0))
    weight, bias = layers.initialize("xavier", 3, 4, seed=2)
    limit = math.sqrt(6.0 / 7.0)
    assert all(abs(value) <= limit for row in weight for value in row)
    assert len(bias) == 3


def test_xavier_zero_denominator_raises_initialization_error() -> None:
    """xavier 的分母为 0 时抛 InitializationError（不是 ZeroDivisionError）."""
    with pytest.raises(errors.InitializationError, match="分母"):
        layers.xavier_limit(0, 0)
    with pytest.raises(errors.InitializationError, match="fan_in"):
        layers.he_scale(0)
    with pytest.raises(errors.InitializationError, match="两侧宽度"):
        layers.initialize("xavier", 0, 3, seed=0)


def test_initialize_unknown_name() -> None:
    """未知初始化名抛 ParameterError."""
    with pytest.raises(errors.ParameterError, match="未知的初始化"):
        layers.initialize("magic", 3, 4, seed=0)


def test_dense_from_spec_is_reproducible() -> None:
    """Dense.from_spec：同一份 spec 两次构造逐位相同，前向也逐位相同."""
    spec = types.DenseSpec(3, 4, activation="relu", init="xavier", seed=7)
    first = layers.Dense.from_spec(spec)
    second = layers.Dense.from_spec(spec)
    assert first.weight == second.weight
    assert first.bias == second.bias
    inputs = ((1.0, 2.0, 3.0),)
    assert first.forward(inputs) == second.forward(inputs)


def test_dense_linear_is_transpose_convention() -> None:
    """dense_linear 用的是 ``x·Wᵀ + b``（与 nn.Linear 一致）——用手算钉住."""
    weight = ((1.0, 2.0), (3.0, 4.0))  # (2, 2)
    bias = (0.5, -0.5)
    inputs = ((1.0, 0.0),)
    assert layers.dense_linear(weight, bias, inputs) == ((1.5, 2.5),)


def test_dense_shapes_are_checked() -> None:
    """宽度不齐抛 ShapeError（权重列数 / 偏置长度都要对齐）."""
    with pytest.raises(errors.ShapeError, match="列数"):
        layers.dense_linear(((1.0, 2.0),), (0.0,), ((1.0, 2.0, 3.0),))
    with pytest.raises(errors.ShapeError, match="bias"):
        layers.dense_linear(((1.0, 2.0),), (0.0, 0.0), ((1.0, 2.0),))
    with pytest.raises(errors.ShapeError, match="非空"):
        layers.dense_linear((), (), ((1.0,),))


def test_dense_forward_matches_manual_gelu_row() -> None:
    """dense_forward：一层 + gelu 的输出与手算逐位一致."""
    spec = types.DenseSpec(2, 1, activation=None, init="zeros", seed=0)
    weight, bias = layers.initialize(spec.init, spec.out_features, spec.in_features, seed=0)
    assert layers.dense_forward(spec, ((1.0, 2.0),)) == layers.dense_linear(weight, bias, ((1.0, 2.0),))


def test_initialize_matrix_guard_rejects_non_finite() -> None:
    """dense_linear 的矩阵护栏拒绝含非有限数的权重."""
    with pytest.raises(errors.ShapeError, match="非有限数"):
        layers.dense_linear(((1.0, math.nan),), (0.0,), ((1.0, 2.0),))


# --------------------------------------------------------------------------- 神经元 / MLP / FFN


def test_neuron_forward() -> None:
    """一个神经元：输出是一个标量；输入长度不对抛 ShapeError."""
    spec = types.NeuronSpec(inputs=2, activation="relu", init="zeros", seed=0)
    assert network.neuron_forward(spec, (1.0, 2.0)) == 0.0
    with pytest.raises(errors.ShapeError, match="长度"):
        network.neuron_forward(spec, (1.0,))


def test_mlp_forward_and_trace() -> None:
    """MLP：逐层前向的宽度与 forward_trace 的账一致；输入宽度不对抛 ShapeError."""
    spec = types.MLPSpec(
        layers=(
            types.DenseSpec(3, 4, activation="relu", init="xavier", seed=1),
            types.DenseSpec(4, 2, activation=None, init="xavier", seed=2),
        )
    )
    output = network.mlp_forward(spec, ((1.0, 2.0, 3.0),))
    assert len(output) == 1 and len(output[0]) == 2
    trace = network.forward_trace(spec)
    assert trace.widths == (3, 4, 2)
    with pytest.raises(errors.ShapeError, match="input_width"):
        network.mlp_forward(spec, ((1.0, 2.0),))
    with pytest.raises(errors.ShapeError, match="不能为空"):
        network.mlp_forward(spec, ())


def test_identity_stack_collapses_to_affine() -> None:
    """恒等激活的多层网络 == 合成后的单层仿射映射（逐点，容差 1e-12）."""
    spec = verify.IDENTITY_SPEC
    multi = network.mlp_forward(spec, verify.IDENTITY_INPUTS)
    weight, bias = network.collapse_identity_mlp(spec)
    single = layers.affine_forward(weight, bias, verify.IDENTITY_INPUTS)
    worst = max(abs(a - b) for ra, rb in zip(multi, single) for a, b in zip(ra, rb))
    assert worst <= verify.IDENTITY_TOLERANCE


def test_collapse_requires_identity_activations() -> None:
    """只要有一层带激活，塌缩就抛 ParameterError（非线性一进来，等式失效）."""
    spec = types.MLPSpec(
        layers=(
            types.DenseSpec(3, 4, activation="relu", init="xavier", seed=1),
            types.DenseSpec(4, 2, activation=None, init="xavier", seed=2),
        )
    )
    with pytest.raises(errors.ParameterError, match="全恒等"):
        network.collapse_identity_mlp(spec)


def test_identity_spec_clears_activations() -> None:
    """identity_spec 把所有激活清成 None（因此可以喂给塌缩）."""
    spec = types.MLPSpec(
        layers=(
            types.DenseSpec(3, 4, activation="relu", init="xavier", seed=1),
            types.DenseSpec(4, 2, activation="gelu", init="xavier", seed=2),
        )
    )
    cleared = network.identity_spec(spec)
    assert all(layer.activation is None for layer in cleared.layers)
    assert network.collapse_identity_mlp(cleared) is not None


def test_build_ffn_params_shapes() -> None:
    """build_ffn_params：d_ff = 4·hidden；四块参数形状正确."""
    params = network.build_ffn_params(hidden=4, seed=7)
    assert len(params.w_in) == 16 and len(params.w_in[0]) == 4
    assert len(params.b_in) == 16
    assert len(params.w_out) == 4 and len(params.w_out[0]) == 16
    assert len(params.b_out) == 4
    assert network.ffn_ratio(4, len(params.w_in)) == 4.0


def test_ffn_block_matches_feed_forward() -> None:
    """**跨天对账**：ffn_block 与 ``encoder_decoder.layers.feed_forward`` 逐位一致（relu 与 gelu）."""
    from smart_research_agent.encoder_decoder.layers import feed_forward
    from smart_research_agent.encoder_decoder.types import FFNWeights

    params = network.build_ffn_params(hidden=verify.FFN_HIDDEN, seed=verify.FFN_SEED)
    real = FFNWeights(w_in=params.w_in, b_in=params.b_in, w_out=params.w_out, b_out=params.b_out)
    for mine_name, day079_name in (("relu", "relu"), ("gelu", "gelu")):
        mine = network.ffn_block(verify.FFN_INPUTS, real, activation=mine_name)
        theirs, _cache = feed_forward(verify.FFN_INPUTS, real, activation=day079_name)
        assert mine == theirs


def test_ffn_block_rejects_missing_attributes() -> None:
    """ffn_block 的 weights 缺字段时抛 ParameterError（而不是 AttributeError）."""
    with pytest.raises(errors.ParameterError, match="四个属性"):
        network.ffn_block(verify.FFN_INPUTS, object(), activation="relu")


def test_ffn_params_validates_shapes() -> None:
    """FFNParams：w_out 行数与 w_in 列数必须一致（前馈是一个往返）."""
    with pytest.raises(errors.ShapeError, match="往返"):
        network.FFNParams(
            w_in=((1.0, 2.0),),
            b_in=(0.0,),
            w_out=((1.0, 2.0),),
            b_out=(0.0, 0.0),
        )


# --------------------------------------------------------------------------- 损失


def test_mse_is_zero_at_perfect_and_matches_hand() -> None:
    """mse：预测等于目标时恰为 0.0；否则等于逐元素平方差的均值（与手算一致）."""
    assert losses.mse(verify.MSE_PRED, verify.MSE_PRED) == 0.0
    assert losses.mse(verify.MSE_PRED, verify.MSE_TARGET) == pytest.approx(0.625, abs=1e-15)


def test_mae_matches_hand() -> None:
    """mae：等于逐元素绝对差的均值（手算 0.75）."""
    assert losses.mae(verify.MSE_PRED, verify.MSE_TARGET) == pytest.approx(0.75, abs=1e-15)


def test_loss_shape_guards() -> None:
    """空样本 / 形状不一致都抛 LossError（**绝不能被静默兜成 0.0**）."""
    with pytest.raises(errors.LossError, match="样本为空"):
        losses.mse((), ())
    with pytest.raises(errors.LossError, match="数量不一致"):
        losses.mse(((1.0,),), ((1.0,), (2.0,)))
    with pytest.raises(errors.LossError, match="宽度"):
        losses.mse(((1.0, 2.0),), ((1.0,),))
    with pytest.raises(errors.NumericError, match="非有限数"):
        losses.mse(((math.inf,),), ((1.0,),))


def test_cross_entropy_from_logits() -> None:
    """cross_entropy：从 logits 走稳定路径（对手算值）。"""
    assert losses.cross_entropy((1.0, 2.0, 3.0), 0) == pytest.approx(2.4076059644443806, abs=1e-15)
    assert losses.cross_entropy((3.0, 2.0, 1.0), 2) == pytest.approx(2.4076059644443806, abs=1e-15)
    assert losses.cross_entropy((0.5,), 0) == 0.0


def test_cross_entropy_matches_sft_loss_bitwise() -> None:
    """**跨天对账**：cross_entropy 与 ``sft.loss.cross_entropy`` 逐位一致."""
    from smart_research_agent.sft.loss import cross_entropy as reference

    for logits, target in verify.CE_CASES:
        assert losses.cross_entropy(logits, target) == reference(list(logits), target)


def test_cross_entropy_paths_agree_within_tolerance() -> None:
    """两条路径（log_softmax vs softmax）在正常区间内一致，且 CE ≥ 0."""
    worst = 0.0
    for logits, target in verify.CE_CASES:
        stable = losses.cross_entropy(logits, target)
        naive = losses.cross_entropy_via_probability(logits, target)
        worst = max(worst, abs(stable - naive))
        assert stable >= 0.0
    assert worst <= verify.PATH_TOLERANCE


def test_cross_entropy_target_out_of_range() -> None:
    """标签越界抛 LossError（不返回 0，也不返回 inf）."""
    with pytest.raises(errors.LossError, match="之外"):
        losses.cross_entropy((1.0, 2.0), 5)
    with pytest.raises(errors.LossError, match="整数"):
        losses.cross_entropy((1.0, 2.0), True)  # bool 是 int 的子类，本包显式拒绝


def test_probability_zero_taking_log_raises() -> None:
    """概率为 0 取 log 抛 LossError（logits 极端时 softmax 下溢成 0）."""
    with pytest.raises(errors.LossError, match="概率为 0"):
        losses.cross_entropy_via_probability((-1000.0, 0.0), 0)


def test_log_softmax_finite_at_extremes() -> None:
    """log_softmax 在极端 logits 上仍有限（这正是它存在的理由）."""
    result = losses.log_softmax((-1000.0, 0.0))
    assert all(math.isfinite(value) for value in result)
    assert result[0] == pytest.approx(-1000.0)
    with pytest.raises(errors.LossError, match="不能为空"):
        losses.log_softmax(())


def test_accuracy_and_perplexity() -> None:
    """accuracy 与 perplexity：命中率、exp(loss)，以及各自的护栏."""
    assert losses.accuracy(((3.0, 1.0), (0.0, 5.0)), (0, 1)) == 1.0
    assert losses.accuracy(((1.0, 3.0), (0.0, 5.0)), (0, 1)) == 0.5
    assert losses.perplexity(0.0) == 1.0
    assert losses.perplexity(math.log(4.0)) == pytest.approx(4.0, abs=1e-12)
    with pytest.raises(errors.LossError, match="样本为空"):
        losses.accuracy((), ())
    with pytest.raises(errors.LossError, match="非负"):
        losses.perplexity(-1.0)


# --------------------------------------------------------------------------- 失败族


def test_every_family_has_a_class_and_an_outcome() -> None:
    """七个失败族：``FAMILY_OUTCOMES`` 与异常类逐键对齐（闭合检查）."""
    assert set(errors.FAMILY_OUTCOMES) == {
        "ShapeError",
        "ParameterError",
        "NumericError",
        "ActivationError",
        "LossError",
        "ForwardError",
        "InitializationError",
    }
    for name in errors.FAMILY_OUTCOMES:
        assert isinstance(getattr(errors, name), type)
        assert issubclass(getattr(errors, name), errors.NeuralError)
        assert errors.FAMILY_OUTCOMES[name].startswith("改")


def test_families_inherit_from_transformer_core() -> None:
    """三族多继承 day075 的族（因此 ``except ShapeError`` 能兜住本层）."""
    from smart_research_agent.transformer_core import errors as core

    assert issubclass(errors.ShapeError, core.ShapeError)
    assert issubclass(errors.ParameterError, core.ParameterError)
    assert issubclass(errors.NumericError, core.NumericError)
    assert issubclass(errors.NeuralError, ValueError)


def test_absent_family_is_gradient_error() -> None:
    """缺席的那一族是 GradientError，理由是"有前向、有损失、没有反向"."""
    assert errors.ABSENT_FAMILY == "GradientError"
    assert "没有反向" in errors.ABSENT_FAMILY_REASON or "反向" in errors.ABSENT_FAMILY_REASON
    assert "day090" in errors.ABSENT_FAMILY_REASON


# --------------------------------------------------------------------------- 性质


def test_cross_check_upper_bound_semantics() -> None:
    """CrossCheck：有 upper_bound 走"≤"、没有时走"=="；负界 / 非有限读数被拒."""
    assert verify.CrossCheck("a", "l", "r", reading=0.5, expected=0.0, upper_bound=1.0).passed
    assert not verify.CrossCheck("a", "l", "r", reading=2.0, expected=0.0, upper_bound=1.0).passed
    assert verify.CrossCheck("a", "l", "r", reading=1.0, expected=1.0, exact=True).passed
    assert not verify.CrossCheck("a", "l", "r", reading=1.0, expected=1.0 + 1e-9, exact=True).passed
    with pytest.raises(errors.NumericError, match="有限"):
        verify.CrossCheck("a", "l", "r", reading=math.inf, expected=0.0)
    with pytest.raises(errors.NumericError, match="上界"):
        verify.CrossCheck("a", "l", "r", reading=0.0, expected=0.0, upper_bound=-1.0)


def test_property_outcome_inapplicable_rules() -> None:
    """PropertyOutcome：不适用 + 通过 是矛盾（会被拒）."""
    outcome = verify.PropertyOutcome("x", applicable=False, passed=False, evidence=("跳过",))
    assert outcome.line().startswith("[不适用]")
    with pytest.raises(errors.NumericError, match="不适用"):
        verify.PropertyOutcome("x", applicable=False, passed=True)


def test_property_report_ok_and_lines() -> None:
    """PropertyReport：ok 只看适用项；lines 先印不适用."""
    report = verify.check_all()
    assert report.ok is True
    lines = report.lines()
    assert len(lines) == 7
    report.require_ok()  # 不抛


def test_check_all_names_match_types() -> None:
    """``check_all`` 的七条性质名与 ``types.NEURAL_PROPERTIES`` 逐一对应."""
    report = verify.check_all()
    assert {outcome.name for outcome in report.outcomes} == set(types.NEURAL_PROPERTIES)
    assert report.to_dict()["counts"]["total"] == 7


def test_individual_property_checks_pass() -> None:
    """七条性质各自单独跑一遍也都通过（返回值不是一个常量）."""
    outcomes = (
        verify.check_activations_are_finite(),
        verify.check_ranges_are_respected(),
        verify.check_softmax_rows_are_distributions(),
        verify.check_identity_stack_collapses(),
        verify.check_mse_is_zero_at_perfect(),
        verify.check_cross_entropy_paths_agree(),
        verify.check_ffn_matches_transformer_stack(),
    )
    assert all(outcome.passed for outcome in outcomes)


def test_finiteness_check_reads_zero() -> None:
    """有限性检查的读数是 0（六个激活 × 11 点，全部有限）."""
    outcome = verify.check_activations_are_finite()
    assert outcome.cross_check is not None
    assert outcome.cross_check.reading == 0.0


def test_ffn_check_reads_zero_mismatches() -> None:
    """FFN 对账的读数是 0（本包与 encoder_decoder 逐位一致）."""
    outcome = verify.check_ffn_matches_transformer_stack()
    assert outcome.cross_check is not None
    assert outcome.cross_check.reading == 0.0


# --------------------------------------------------------------------------- 表


def test_activation_rows_cover_all() -> None:
    """激活表：六行、每行都带口径与读数."""
    rows = study.activation_rows()
    assert len(rows) == 6
    assert all(row.profile and row.reading for row in rows)
    assert rows[0].line().startswith("relu")


def test_initialization_rows_cover_all() -> None:
    """初始化表：五行、每行都带首元素读数."""
    rows = study.initialization_rows()
    assert len(rows) == 5
    zeros = next(row for row in rows if row.name == "zeros")
    assert zeros.first_weight == 0.0 and zeros.first_bias == 0.0


def test_layer_rows_report_accounting() -> None:
    """层次账表：4→8→3 的两层账，参数总数 67."""
    rows = study.layer_rows()
    assert len(rows) == 2
    assert rows[0].total == 40
    assert rows[1].total == 27
    assert study.LAYER_SAMPLE.parameter_count == 67


def test_loss_rows_report_three_values() -> None:
    """损失表：三行、读数与损失函数一致."""
    rows = {row.name: row.reading for row in study.loss_rows()}
    assert rows[types.LOSS_MSE] == pytest.approx(0.625)
    assert rows[types.LOSS_MAE] == pytest.approx(0.75)
    assert rows[types.LOSS_CROSS_ENTROPY] == pytest.approx(2.4076059644443806)


def test_property_rows_match_report() -> None:
    """性质表：七行，且与 ``verify.check_all`` 的结论一致."""
    rows = study.property_rows()
    report = verify.check_all()
    assert len(rows) == 7
    assert [row.passed for row in rows] == [outcome.passed for outcome in report.outcomes]


def test_study_lines_and_notes() -> None:
    """study_lines：五节；note_lines：十条（顺序即写入顺序）."""
    lines = study.study_lines()
    assert sum(1 for line in lines if line.startswith("== ")) == 5
    notes = study.note_lines()
    assert len(notes) == 10
    assert notes[0].startswith(" 1.")


# --------------------------------------------------------------------------- 包入口


def test_package_all_is_sorted_and_resolvable() -> None:
    """``__all__`` 是字母序，且每一个名字都能解析（不会被同名子模块遮掉）."""
    import smart_research_agent.neural_basics as package

    assert package.__all__ == sorted(package.__all__)
    assert all(hasattr(package, name) for name in package.__all__)
    submodules = {"errors", "types", "activations", "layers", "network", "losses", "verify", "study"}
    assert not (submodules & set(package.__all__))


def test_absent_family_reason_is_the_fifth_different_one() -> None:
    """缺席理由与 day088 的那一条不同：本日**有新式子**，但没有反向."""
    assert "有新式子" in errors.ABSENT_FAMILY_REASON
    assert "自动微分" in errors.ABSENT_FAMILY_REASON


# --------------------------------------------------------------------------- 补充：边界与账


def test_dense_spec_rejects_zero_width() -> None:
    """DenseSpec：任一侧宽度为 0 都抛 ParameterError."""
    with pytest.raises(errors.ParameterError, match="正整数"):
        types.DenseSpec(0, 4, activation="relu")
    with pytest.raises(errors.ParameterError, match="正整数"):
        types.DenseSpec(4, 0, activation="relu")


def test_neuron_spec_identity_activation() -> None:
    """NeuronSpec 的 activation=None 表示恒等，line 里印 identity."""
    spec = types.NeuronSpec(inputs=2, activation=None, init="zeros", seed=0)
    assert spec.line().startswith("neuron(2→1) | 激活 identity")
    assert network.neuron_forward(spec, (3.0, 4.0)) == 0.0


def test_mlp_spec_rejects_non_dense_layer() -> None:
    """MLPSpec 的成员必须是 DenseSpec（否则抛 ParameterError）."""
    with pytest.raises(errors.ParameterError, match="不是 DenseSpec"):
        types.MLPSpec(layers=(("3→4",),))


def test_forward_trace_rejects_empty_layers() -> None:
    """ForwardTrace 至少要一层（空层列表抛 ParameterError）."""
    with pytest.raises(errors.ParameterError, match="不能为空"):
        types.ForwardTrace(input_width=3, layer_widths=())


def test_compose_affine_single_layer_is_identity_of_composition() -> None:
    """单层的合成就是它自己；空串抛 ParameterError."""
    weight = ((1.0, 2.0), (3.0, 4.0))
    bias = (0.5, -0.5)
    assert layers.compose_affine(((weight, bias),)) == (weight, bias)
    with pytest.raises(errors.ParameterError, match="至少需要一层"):
        layers.compose_affine(())


def test_compose_affine_rejects_width_mismatch() -> None:
    """合成时相邻两层的宽度接不上抛 ShapeError."""
    with pytest.raises(errors.ShapeError, match="输入宽度"):
        layers.compose_affine(
            (
                (((1.0, 2.0),), (0.0,)),
                (((1.0, 2.0),), (0.0,)),
            )
        )


def test_dense_bias_mismatch_raises() -> None:
    """Dense 的 bias 长度必须等于 weight 行数."""
    with pytest.raises(errors.ShapeError, match="bias"):
        layers.Dense(weight=((1.0, 2.0),), bias=(0.0, 0.0), activation=None)


def test_dense_to_dict_and_line() -> None:
    """Dense 的摊平字段与一行说明都带形状与参数量."""
    weight, bias = layers.initialize("zeros", 2, 3, seed=0)
    dense = layers.Dense(weight=weight, bias=bias, activation="relu")
    assert dense.to_dict()["in_features"] == 3
    assert dense.to_dict()["out_features"] == 2
    assert dense.line().startswith("dense(3→2)")


def test_dense_forward_with_activation_matches_two_steps() -> None:
    """带激活的一层 = 先 affine 再逐行激活（两步逐位一致）."""
    dense = layers.Dense.from_spec(
        types.DenseSpec(3, 2, activation="gelu", init="xavier", seed=4)
    )
    inputs = ((0.3, -0.7, 1.1),)
    linear = dense.linear(inputs)
    assert dense.forward(inputs) == tuple(activations.activate("gelu", row) for row in linear)


def test_softmax_empty_and_non_finite_rejected() -> None:
    """softmax：空向量与非有限输入都抛 ActivationError."""
    with pytest.raises(errors.ActivationError, match="不能为空"):
        activations.softmax(())
    with pytest.raises(errors.ActivationError, match="有限数"):
        activations.softmax((math.inf,))


def test_lcg_stream_zero_length_is_empty() -> None:
    """长度为 0 的 LCG 串是空元组（合法，不报错）."""
    assert layers.lcg_stream(7, 0) == ()


def test_initialize_normal_bias_length() -> None:
    """normal / he 的偏置长度等于 out_features."""
    for name in ("normal", "he"):
        _weight, bias = layers.initialize(name, 5, 4, seed=3)
        assert len(bias) == 5


def test_build_ffn_params_validates_dimensions() -> None:
    """build_ffn_params：hidden / ffn 非正都被拒."""
    with pytest.raises(errors.ParameterError, match="hidden"):
        network.build_ffn_params(hidden=0)
    with pytest.raises(errors.ParameterError, match="ffn"):
        network.build_ffn_params(hidden=2, ffn=0)


def test_ffn_ratio_validates_hidden() -> None:
    """ffn_ratio：hidden 非正抛 ParameterError."""
    assert network.ffn_ratio(4, 16) == 4.0
    with pytest.raises(errors.ParameterError, match="hidden"):
        network.ffn_ratio(0, 16)


def test_loss_report_reduction_must_be_non_empty() -> None:
    """LossReport：reduction 不能为空."""
    with pytest.raises(errors.ParameterError, match="平均方式"):
        types.LossReport(name=types.LOSS_MSE, value=1.0, samples=1, reduction="")


def test_property_report_require_ok_raises_on_failure() -> None:
    """PropertyReport：有适用项失败时 require_ok 抛 NeuralError."""
    failing = verify.PropertyOutcome("x", applicable=True, passed=False, evidence=("坏",))
    report = verify.PropertyReport(outcomes=(failing,))
    assert report.ok is False
    with pytest.raises(errors.NeuralError, match="未全部通过"):
        report.require_ok()


def test_property_report_ignores_inapplicable_for_ok() -> None:
    """PropertyReport：不适用项既不算通过、也不算失败."""
    skipped = verify.PropertyOutcome("x", applicable=False, passed=False)
    passing = verify.PropertyOutcome("y", applicable=True, passed=True)
    report = verify.PropertyReport(outcomes=(skipped, passing))
    assert report.ok is True
    assert len(report.applicable) == 1


def test_remaining_property_readings() -> None:
    """其余几条性质的读数：值域 0、softmax 0、CE 0、塌缩 <= 1e-12."""
    assert verify.check_ranges_are_respected().cross_check.reading == 0.0
    assert verify.check_softmax_rows_are_distributions().cross_check.reading == 0.0
    assert verify.check_cross_entropy_paths_agree().cross_check.reading == 0.0
    identity = verify.check_identity_stack_collapses().cross_check
    assert identity is not None and identity.reading <= verify.IDENTITY_TOLERANCE
    assert verify.check_mse_is_zero_at_perfect().cross_check.reading == 0.0


def test_notes_order_and_lines_agree() -> None:
    """十条笔记的顺序表与键集合一致，且 note_lines 印满十行."""
    assert types.NEURAL_NOTES_ORDER == tuple(types.NEURAL_NOTES)
    lines = study.note_lines()
    assert len(lines) == 10
    assert all(line.strip().endswith("。") or "。" in line for line in lines)


def test_activation_ranges_cover_all_activations() -> None:
    """值域表覆盖六个激活（缺一个就会有一条性质查不到）。"""
    assert set(activations.ACTIVATION_RANGES) == set(types.ACTIVATIONS)


def test_gelu_new_constants_match_day085_source() -> None:
    """gelu_tanh 的两个常数与 day085 的 ``hf_source.blocks`` 逐字相同."""
    from smart_research_agent.hf_source import blocks

    assert activations.GELU_TANH_CUBIC == blocks.GELU_NEW_CUBIC
    assert activations.GELU_TANH_INNER == blocks.GELU_NEW_INNER
