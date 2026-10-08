"""``conv_net``：卷积神经网络前向 / 反向 / 训练的测试（day093 / M8-D4）.

本文件覆盖新包的九个功能模块与包入口：口径表、五个失败族、算子层
（尺寸公式 / 填充 / 卷积 / 池化 / 感受野）、层（可复现初始化 / 多通道前向）、
反向（dK / dB / dX / 池化）、小 CNN 的前向反向与参数压平、合成数据集上的训练、
七条性质与七张表。

样本全部来自本包自己的确定性构造；**跨天对账真的调用既有包**：
数值差分用 ``math_foundations.calculus``（day074），激活导数用 ``backprop.gradients``（day090），
交叉熵用 ``neural_basics.losses``（day089），优化器用 ``optimizers``（day092）。
"""

from __future__ import annotations

import math

import pytest

from smart_research_agent.conv_net import (
    errors,
    gradients,
    layers,
    network,
    ops,
    study,
    train,
    types,
    verify,
)

# --------------------------------------------------------------------------- 口径表


def test_padding_and_pool_tables_are_closed() -> None:
    """两种填充 / 两种池化的三张表逐键对齐。"""
    assert types.PADDING_MODES == ("valid", "same")
    assert set(types.PADDING_MODES) == set(types.PADDING_DESCRIPTIONS) == set(types.PADDING_FORMULAS)
    assert types.POOL_MODES == ("max", "avg")
    assert set(types.POOL_MODES) == set(types.POOL_DESCRIPTIONS) == set(types.POOL_FORMULAS)


def test_conv_activations_are_a_subset_of_day089() -> None:
    """卷积可用的激活是 day089 ACTIVATIONS 的子集，且不含 softmax。"""
    from smart_research_agent.neural_basics.types import ACTIVATIONS

    assert set(types.CONV_ACTIVATIONS) <= set(ACTIVATIONS)
    assert "softmax" not in types.CONV_ACTIVATIONS


def test_seven_properties_have_descriptions_and_failures() -> None:
    """七条性质：名单 / 说明 / "失败意味着什么"三张表逐键对齐。"""
    assert len(types.CONV_PROPERTIES) == 7
    assert set(types.PROPERTY_DESCRIPTIONS) == set(types.CONV_PROPERTIES)
    assert set(types.PROPERTY_FAILURE) == set(types.CONV_PROPERTIES)
    assert all(types.PROPERTY_DESCRIPTIONS.values())
    assert all(types.PROPERTY_FAILURE.values())


def test_notes_are_ten_and_ordered() -> None:
    """十条笔记，且顺序表与键集合一致。"""
    assert len(types.CONV_NOTES) == 10
    assert types.CONV_NOTES_ORDER == tuple(types.CONV_NOTES)
    assert all(types.CONV_NOTES.values())


def test_boundaries_are_five() -> None:
    """五条边界是一份非空清单（**这一课明确不承诺的事**）。"""
    assert len(types.CONV_BOUNDARIES) == 5
    assert all(types.CONV_BOUNDARIES)


def test_torch_counterparts_cover_core_ops_and_no_version() -> None:
    """PyTorch 对照表覆盖卷积 / 池化 / 反向 / 优化器，且**不出现**版本号。"""
    required = {"conv2d", "max_pool2d", "avg_pool2d", "padding", "conv_backward", "optimizer"}
    assert required <= set(types.TORCH_COUNTERPARTS)
    assert all(types.TORCH_COUNTERPARTS.values())
    assert "torch==" not in " ".join(types.TORCH_COUNTERPARTS.values())


def test_three_formulas_are_non_empty() -> None:
    """三张公式都写下了一行可读的式子。"""
    assert "⌊" in types.OUTPUT_SIZE_FORMULA
    assert "r" in types.RECEPTIVE_FIELD_FORMULA
    assert "C_out" in types.CONV_PARAM_FORMULA


# --------------------------------------------------------------------------- 失败族


def test_family_tables_are_aligned() -> None:
    """五个族的"该怎么办"表与类表逐键对齐。"""
    assert len(errors.FAMILY_OUTCOMES) == 5
    assert set(errors.FAMILY_OUTCOMES) == {
        "ShapeError",
        "ParameterError",
        "NumericError",
        "WindowError",
        "BackwardError",
    }


def test_returned_and_absent_families_are_constants() -> None:
    """本课没有"回来"的族，且 ``GradientError`` 继续缺席（可断言的事实）。"""
    assert errors.RETURNED_FAMILY is None
    assert errors.RETURNED_FAMILY_REASON
    assert errors.ABSENT_FAMILY == "GradientError"
    assert errors.ABSENT_FAMILY_REASON


def test_errors_inherit_both_families() -> None:
    """本族错误既是 ``ConvError``、也是 day075 的对应族。"""
    from smart_research_agent.transformer_core import errors as core_errors

    assert issubclass(errors.ShapeError, errors.ConvError)
    assert issubclass(errors.ShapeError, core_errors.ShapeError)
    assert issubclass(errors.ParameterError, core_errors.ParameterError)
    assert issubclass(errors.NumericError, core_errors.NumericError)
    assert issubclass(errors.WindowError, errors.ConvError)
    assert issubclass(errors.BackwardError, errors.ConvError)


# --------------------------------------------------------------------------- 算子：校验


def test_checked_positive_int() -> None:
    """整数校验：非整数 / 过小都拒绝。"""
    assert ops.checked_positive_int(3, name="k") == 3
    with pytest.raises(errors.ParameterError):
        ops.checked_positive_int(3.0, name="k")  # type: ignore[arg-type]
    with pytest.raises(errors.ParameterError):
        ops.checked_positive_int(0, name="k")
    with pytest.raises(errors.ParameterError):
        ops.checked_positive_int(True, name="k")


def test_as_matrix_rejects_bad_inputs() -> None:
    """非二维 / 空 / 行宽不齐 / 非有限数都拒绝。"""
    with pytest.raises(errors.ShapeError):
        ops.as_matrix((), name="x")
    with pytest.raises(errors.ShapeError):
        ops.as_matrix(((1.0, 2.0), (3.0,)), name="x")
    with pytest.raises(errors.ShapeError):
        ops.as_matrix((("a", "b"),), name="x")  # type: ignore[arg-type]
    with pytest.raises(errors.NumericError):
        ops.as_matrix(((1.0, float("inf")),), name="x")


def test_as_matrix_accepts_rectangular() -> None:
    """正常的矩形输入被收敛成 ``tuple[tuple[float, ...], ...]``。"""
    assert ops.as_matrix([[1, 2], [3, 4]], name="x") == ((1.0, 2.0), (3.0, 4.0))


# --------------------------------------------------------------------------- 算子：尺寸与填充


def test_output_size_valid_cases() -> None:
    """尺寸公式在几组参数下的读数。"""
    assert ops.output_size(6, 3, stride=1, padding=0) == 4
    assert ops.output_size(6, 3, stride=1, padding=1) == 6
    assert ops.output_size(6, 3, stride=2, padding=1) == 3
    assert ops.output_size(6, 3, stride=1, padding=1, dilation=2) == 4


def test_output_size_rejects_bad_padding() -> None:
    """padding 必须是整数（字符串请走 resolve_padding / feature_size）。"""
    with pytest.raises(errors.ParameterError):
        ops.output_size(6, 3, padding="same")  # type: ignore[arg-type]


def test_output_size_window_too_small() -> None:
    """输入装不下有效核宽时抛 ``WindowError``。"""
    with pytest.raises(errors.WindowError):
        ops.output_size(2, 5, stride=1, padding=0)


def test_resolve_padding_valid_and_int() -> None:
    """valid 解析成 0；整数原样返回。"""
    assert ops.resolve_padding(6, 3, padding="valid") == 0
    assert ops.resolve_padding(6, 3, padding=2) == 2
    with pytest.raises(errors.ParameterError):
        ops.resolve_padding(6, 3, padding=-1)
    with pytest.raises(errors.ParameterError):
        ops.resolve_padding(6, 3, padding="full")


def test_resolve_padding_same_requires_stride_one() -> None:
    """same 与步长 > 1 冲突。"""
    with pytest.raises(errors.WindowError):
        ops.resolve_padding(6, 3, stride=2, padding="same")


def test_resolve_padding_same_even_kernel_has_no_solution() -> None:
    """偶数核 + 步长 1 得不到同尺寸。"""
    with pytest.raises(errors.WindowError):
        ops.resolve_padding(6, 2, padding="same")
    assert ops.resolve_padding(6, 3, padding="same") == 1


def test_feature_size_resolves_padding() -> None:
    """``feature_size`` 一步到位：先解析填充再算尺寸。"""
    assert ops.feature_size(6, 3, padding="same") == 6
    assert ops.feature_size(6, 3, padding="valid") == 4


def test_pad2d_symmetric_zero_padding() -> None:
    """对称补零：四边各补 padding。"""
    image = ((1.0, 2.0), (3.0, 4.0))
    padded = ops.pad2d(image, 1)
    assert padded == (
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 2.0, 0.0),
        (0.0, 3.0, 4.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
    )
    assert ops.pad2d(image, 0) == image


# --------------------------------------------------------------------------- 算子：卷积与池化


def test_conv2d_known_value() -> None:
    """2×2 的图配 2×2 的核：只得到一个数。"""
    assert ops.conv2d(((1.0, 2.0), (3.0, 4.0)), ((1.0, 0.0), (0.0, 1.0))) == ((5.0,),)


def test_conv2d_is_cross_correlation_not_flipped() -> None:
    """互相关约定：核不翻转（翻转会让这个不对称的核给出不同的数）。"""
    image = ((0.0, 1.0), (2.0, 3.0))
    kernel = ((1.0, 0.0), (0.0, 0.0))
    # 互相关取左上角元素；若翻转核，就会取右下角的 3.0
    assert ops.conv2d(image, kernel) == ((0.0,),)


def test_conv2d_stride_and_dilation() -> None:
    """步长与膨胀都改变输出形状。"""
    image = tuple(tuple(float(i * 5 + j) for j in range(5)) for i in range(5))
    kernel = ((1.0, 1.0), (1.0, 1.0))
    assert len(ops.conv2d(image, kernel, stride=2)) == 2
    assert len(ops.conv2d(image, kernel, dilation=2)) == 3


def test_conv2d_same_padding_shape() -> None:
    """same 填充保住空间尺寸。"""
    kernel = ((1.0, 0.0, -1.0),) * 3
    produced = ops.conv2d(verify.SLIDING_IMAGE, kernel, padding="same")
    assert len(produced) == len(verify.SLIDING_IMAGE)
    assert len(produced[0]) == len(verify.SLIDING_IMAGE[0])


def test_pool2d_max_and_avg() -> None:
    """池化：max 取窗口极值、avg 取窗口均值。"""
    image = ((1.0, 3.0), (2.0, 4.0))
    assert ops.max_pool2d(image) == ((4.0,),)
    assert ops.avg_pool2d(image) == ((2.5,),)


def test_pool2d_rejects_unknown_mode() -> None:
    """未知池化模式当场拒绝。"""
    with pytest.raises(errors.ParameterError):
        ops.pool2d(((1.0, 2.0), (3.0, 4.0)), mode="median")


def test_pool2d_window_must_cover_input() -> None:
    """窗口盖不满输入时抛 ``WindowError``（不做"丢掉最后一块"的兜底）。"""
    image = tuple(tuple(float(i * 5 + j) for j in range(5)) for i in range(5))
    with pytest.raises(errors.WindowError):
        ops.pool2d(image, window=2, stride=2)


def test_pool2d_stride_two_on_six() -> None:
    """6×6 按窗口 2、步长 2 下采样成 3×3。"""
    image = tuple(tuple(float(i * 6 + j) for j in range(6)) for i in range(6))
    assert len(ops.pool2d(image, window=2, stride=2)) == 3


def test_receptive_field_empty_and_stacked() -> None:
    """感受野：空序列为 1；堆两层 3×3 得到 5×5。"""
    assert ops.receptive_field(()) == 1
    assert ops.receptive_field(((3, 1),)) == 3
    assert ops.receptive_field(((3, 1), (3, 1))) == 5
    assert ops.receptive_field(((3, 1), (3, 1), (2, 2))) == 6


def test_flatten_map_row_major() -> None:
    """压平是行优先。"""
    assert ops.flatten_map(((1.0, 2.0), (3.0, 4.0))) == (1.0, 2.0, 3.0, 4.0)


def test_feature_stats() -> None:
    """特征图画像：均值 / 最大值 / 非零比例。"""
    stats = ops.feature_stats(((0.0, 1.0), (2.0, 3.0)))
    assert stats["mean"] == pytest.approx(1.5)
    assert stats["max"] == 3.0
    assert stats["nonzero_ratio"] == pytest.approx(0.75)


# --------------------------------------------------------------------------- 层


def test_conv_spec_validation() -> None:
    """ConvSpec 的每个字段都有护栏。"""
    assert layers.ConvSpec(1, 2, 3).fan_in == 9
    assert layers.ConvSpec(1, 2, 3).parameter_count == 20
    with pytest.raises(errors.ParameterError):
        layers.ConvSpec(0, 2, 3)
    with pytest.raises(errors.ParameterError):
        layers.ConvSpec(1, 2, 3, stride=0)
    with pytest.raises(errors.ParameterError):
        layers.ConvSpec(1, 2, 3, activation="softmax")
    with pytest.raises(errors.ParameterError):
        layers.ConvSpec(1, 2, 3, seed=-1)


def test_conv_spec_line() -> None:
    """一行说明含通道 / 核 / 参数数。"""
    text = layers.ConvSpec(1, 2, 3).line()
    assert "conv(1→2" in text
    assert "参数 20" in text


def test_initialize_conv_is_reproducible() -> None:
    """同一份 spec 两次初始化逐位相同；不同种子不同。"""
    spec = layers.ConvSpec(1, 2, 3, seed=7)
    first = layers.initialize_conv(spec)
    second = layers.initialize_conv(spec)
    assert first.kernels == second.kernels
    assert first.bias == second.bias
    other = layers.initialize_conv(layers.ConvSpec(1, 2, 3, seed=8))
    assert other.kernels != first.kernels


def test_conv_params_properties() -> None:
    """ConvParams 的形状属性与参数量。"""
    spec = layers.ConvSpec(2, 3, 3, seed=1)
    params = layers.initialize_conv(spec)
    assert params.out_channels == 3
    assert params.in_channels == 2
    assert params.kernel_size == (3, 3)
    assert params.parameter_count == spec.parameter_count


def test_conv_params_validation() -> None:
    """ConvParams 拒绝空通道、通道数不齐、bias 长度不符。"""
    kernel = ((1.0, 0.0), (0.0, 1.0))
    with pytest.raises(errors.ShapeError):
        layers.ConvParams(kernels=(), bias=())
    with pytest.raises(errors.ShapeError):
        layers.ConvParams(kernels=(((kernel,), (kernel,)), ((kernel,),)), bias=(0.0, 0.0))
    with pytest.raises(errors.ShapeError):
        layers.ConvParams(kernels=(((kernel,),),), bias=(0.0, 0.0))


def test_conv_block_forward_channel_mismatch() -> None:
    """输入通道数与核数不符时抛 ``ShapeError``。"""
    spec = layers.ConvSpec(2, 1, 3)
    params = layers.initialize_conv(spec)
    with pytest.raises(errors.ShapeError):
        layers.conv_block_forward(params, (verify.SLIDING_IMAGE,), spec)


def test_conv_layer_forward_is_reproducible() -> None:
    """按同一份 spec 前向两次逐位相同。"""
    spec = layers.ConvSpec(1, 2, 3, padding="same", seed=3)
    first = layers.conv_layer_forward(spec, (study.SAMPLE_IMAGE,))
    second = layers.conv_layer_forward(spec, (study.SAMPLE_IMAGE,))
    assert first == second
    assert len(first) == 2


def test_activate_map_none_is_identity() -> None:
    """``activation=None`` 时不改变特征图。"""
    image = ((1.0, -2.0),)
    assert layers.activate_map(None, image) == image
    assert layers.activate_map("relu", image) == ((1.0, 0.0),)


def test_pool_block_forward_rejects_empty() -> None:
    """空通道的池化没有定义。"""
    with pytest.raises(errors.WindowError):
        layers.pool_block_forward(())


# --------------------------------------------------------------------------- 反向


def test_conv2d_backward_known_value() -> None:
    """2×2 图 + 2×2 核：d_kernel 与 d_image 的形状与取值。"""
    d_kernel, d_image = gradients.conv2d_backward(
        ((1.0, 2.0), (3.0, 4.0)), ((1.0, 0.0), (0.0, 1.0)), ((1.0,),)
    )
    assert d_kernel == ((1.0, 2.0), (3.0, 4.0))
    assert d_image == ((1.0, 0.0), (0.0, 1.0))


def test_conv2d_backward_shapes_and_padding() -> None:
    """带填充时 d_image 回到**原图**尺寸（梯度形状必须与前向输出配对）。"""
    grad = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    d_kernel, d_image = gradients.conv2d_backward(
        verify.SLIDING_IMAGE, verify.SLIDING_KERNEL, grad, padding=1
    )
    assert len(d_kernel) == 3 and len(d_kernel[0]) == 3
    assert len(d_image) == len(verify.SLIDING_IMAGE)
    assert len(d_image[0]) == len(verify.SLIDING_IMAGE[0])


def test_conv2d_backward_rejects_wrong_grad_shape() -> None:
    """grad_output 与前向输出不配对时抛 ``BackwardError``。"""
    with pytest.raises(errors.BackwardError):
        gradients.conv2d_backward(verify.SLIDING_IMAGE, verify.SLIDING_KERNEL, ((1.0,),))


def test_conv2d_backward_matches_numerical() -> None:
    """解析梯度与数值差分一致（本课命门那条）。"""
    outcome = verify.check_conv_backward_matches_numerical()
    assert outcome.passed is True
    assert outcome.check.reading <= verify.GRAD_TOLERANCE


def test_conv_block_backward_bias_grad_is_sum() -> None:
    """d_bias 是这一输出通道 grad_output 的总和。"""
    spec = layers.ConvSpec(1, 1, 3, activation=None, seed=1)
    params = layers.initialize_conv(spec)
    grads = gradients.conv_block_backward(spec, params, (verify.SLIDING_IMAGE,), (verify.GRAD_OUTPUT,))
    expected = math.fsum(value for row in verify.GRAD_OUTPUT for value in row)
    assert grads.bias_grad[0] == pytest.approx(expected)


def test_conv_block_backward_requires_pre_activation() -> None:
    """激活非恒等而缺少 pre_activations 时抛 ``BackwardError``。"""
    spec = layers.ConvSpec(1, 1, 3, activation="relu", seed=1)
    params = layers.initialize_conv(spec)
    with pytest.raises(errors.BackwardError):
        gradients.conv_block_backward(spec, params, (verify.SLIDING_IMAGE,), (verify.GRAD_OUTPUT,))


def test_conv_block_backward_grad_output_channel_mismatch() -> None:
    """grad_outputs 的通道数不符时抛 ``BackwardError``。"""
    spec = layers.ConvSpec(1, 2, 3, activation=None, seed=1)
    params = layers.initialize_conv(spec)
    with pytest.raises(errors.BackwardError):
        gradients.conv_block_backward(spec, params, (verify.SLIDING_IMAGE,), (verify.GRAD_OUTPUT,))


def test_conv_forward_with_cache_matches_layers() -> None:
    """带缓存的前向与 ``layers.conv_block_forward`` 逐位一致（前向只写一遍）。"""
    spec = layers.ConvSpec(1, 2, 3, padding="same", activation="relu", seed=3)
    params = layers.initialize_conv(spec)
    outputs, pre = gradients.conv_forward_with_cache(spec, params, (study.SAMPLE_IMAGE,))
    assert outputs == layers.conv_block_forward(params, (study.SAMPLE_IMAGE,), spec)
    assert len(pre) == 2


def test_pool2d_backward_max_routes_to_maximum() -> None:
    """max 池化把梯度给到窗口最大值那一格，其余为 0。"""
    image = ((1.0, 3.0), (2.0, 4.0))
    grad = gradients.max_pool_backward(image, ((1.0,),))
    assert grad == ((0.0, 0.0), (0.0, 1.0))


def test_pool2d_backward_avg_shares_evenly() -> None:
    """avg 池化把梯度均分给窗口内每一格（总和不放大）。"""
    image = ((1.0, 3.0), (2.0, 4.0))
    grad = gradients.avg_pool_backward(image, ((4.0,),))
    assert grad == ((1.0, 1.0), (1.0, 1.0))


def test_pool2d_backward_rejects_wrong_grad_shape() -> None:
    """池化反向的梯度形状与输出不符时抛 ``BackwardError``。"""
    with pytest.raises(errors.BackwardError):
        gradients.pool2d_backward(((1.0, 3.0), (2.0, 4.0)), ((1.0, 1.0),))


def test_pool2d_backward_rejects_unknown_mode() -> None:
    """未知池化模式当场拒绝。"""
    with pytest.raises(errors.ParameterError):
        gradients.pool2d_backward(((1.0, 3.0), (2.0, 4.0)), ((1.0,),), mode="median")


# --------------------------------------------------------------------------- 网络


def test_build_cnn_is_reproducible_and_shaped() -> None:
    """同一 spec 两次建网逐位相同，且形状与预期一致。"""
    spec = study.TRAIN_SPEC
    first = network.build_cnn(spec, (6, 6), seed=100)
    second = network.build_cnn(spec, (6, 6), seed=100)
    assert first.conv.kernels == second.conv.kernels
    assert first.head_weight == second.head_weight
    assert first.classes == 2
    assert first.flatten_features == 18  # 2 通道 × 3×3 池化输出
    assert first.parameter_count == spec.parameter_count + 2 * 18 + 2


def test_build_cnn_rejects_single_class() -> None:
    """类别数必须 >= 2。"""
    with pytest.raises(errors.ParameterError):
        network.build_cnn(study.TRAIN_SPEC, (6, 6), classes=1)


def test_pooled_size_matches_formula() -> None:
    """池化后的尺寸与逐层公式一致。"""
    assert network.pooled_size((6, 6), study.TRAIN_SPEC, 2) == (3, 3)


def test_cnn_forward_and_predict() -> None:
    """前向给出两个 logits，predict 取最大值下标。"""
    spec = study.TRAIN_SPEC
    params = network.build_cnn(spec, (6, 6), seed=100)
    logits = network.cnn_forward(params, spec, (study.SAMPLE_IMAGE,))
    assert len(logits) == 2
    assert network.predict(params, spec, (study.SAMPLE_IMAGE,)) in (0, 1)


def test_cnn_gradients_shapes_match_parameters() -> None:
    """梯度的压平长度与参数的压平长度一致（两者必须能逐位相加）。"""
    spec = study.TRAIN_SPEC
    params = network.build_cnn(spec, (6, 6), seed=100)
    loss, grads = network.loss_and_grad(params, spec, (study.SAMPLE_IMAGE,), 0)
    assert loss > 0
    assert len(network.flatten_gradients(grads)) == len(network.flatten_params(params))


def test_cnn_backward_rejects_wrong_grad_logits() -> None:
    """grad_logits 长度与类别数不符时抛 ``BackwardError``。"""
    spec = study.TRAIN_SPEC
    params = network.build_cnn(spec, (6, 6), seed=100)
    with pytest.raises(errors.BackwardError):
        network.cnn_backward(params, spec, (study.SAMPLE_IMAGE,), (1.0,))


def test_loss_gradient_is_probability_minus_onehot() -> None:
    """交叉熵对 logits 的梯度就是 ``p − onehot``（与 day090 那条一致）。"""
    spec = study.TRAIN_SPEC
    params = network.build_cnn(spec, (6, 6), seed=100)
    logits, _cache = network.cnn_forward_cached(params, spec, (study.SAMPLE_IMAGE,))
    probabilities = network._softmax(logits)
    loss, grads = network.loss_and_grad(params, spec, (study.SAMPLE_IMAGE,), 0)
    # 用 logits 的差分数值核对 loss 的梯度（前两个分量即 dloss/dlogits）
    h = 1e-6
    shifted = tuple(value + (h if i == 0 else 0.0) for i, value in enumerate(logits))
    from smart_research_agent.neural_basics.losses import cross_entropy

    numeric = (cross_entropy(shifted, 0) - loss) / h
    assert numeric == pytest.approx(probabilities[0] - 1.0, abs=1e-4)


def test_flatten_unflatten_roundtrip() -> None:
    """压平再还原逐位相同。"""
    spec = study.TRAIN_SPEC
    params = network.build_cnn(spec, (6, 6), seed=100)
    flat = network.flatten_params(params)
    restored = network.unflatten_params(flat, template=params)
    assert restored.conv.kernels == params.conv.kernels
    assert restored.head_weight == params.head_weight
    assert restored.head_bias == params.head_bias


def test_unflatten_rejects_wrong_length() -> None:
    """长度对不上当场抛 ``ShapeError``。"""
    params = network.build_cnn(study.TRAIN_SPEC, (6, 6), seed=100)
    with pytest.raises(errors.ShapeError):
        network.unflatten_params((1.0, 2.0), template=params)


def test_unflatten_rejects_non_finite() -> None:
    """非有限数当场抛 ``ShapeError``。"""
    params = network.build_cnn(study.TRAIN_SPEC, (6, 6), seed=100)
    flat = list(network.flatten_params(params))
    flat[0] = float("inf")
    with pytest.raises(errors.ShapeError):
        network.unflatten_params(tuple(flat), template=params)


def test_cnn_params_validation() -> None:
    """CNNParams 的 bias 长度与 head_weight 行数必须一致。"""
    spec = study.TRAIN_SPEC
    params = network.build_cnn(spec, (6, 6), seed=100)
    with pytest.raises(errors.ShapeError):
        network.CNNParams(conv=params.conv, head_weight=params.head_weight, head_bias=(0.0,))


# --------------------------------------------------------------------------- 数据集与训练


def test_make_stripe_sample_shapes_and_labels() -> None:
    """样本的形状与标签正确；竖线的非零元素排成一列。"""
    image, label = train.make_stripe_sample(6, 0)
    assert label == 0
    assert len(image) == 6 and len(image[0]) == 6
    marked = {column for row in image for column, value in enumerate(row) if value == 1.0}
    assert marked == {3}


def test_make_stripe_sample_validation() -> None:
    """标签 / 尺寸 / 噪声都有护栏。"""
    with pytest.raises(errors.ParameterError):
        train.make_stripe_sample(6, 2)
    with pytest.raises(errors.ParameterError):
        train.make_stripe_sample(2, 0)
    with pytest.raises(errors.ParameterError):
        train.make_stripe_sample(6, 0, noise=-1.0)


def test_make_stripe_dataset_balanced() -> None:
    """数据集两类各占一半。"""
    dataset = train.make_stripe_dataset(6, per_class=3)
    assert len(dataset) == 6
    assert sum(1 for _image, label in dataset if label == 0) == 3
    with pytest.raises(errors.ParameterError):
        train.make_stripe_dataset(6, per_class=0)


def test_accuracy_and_mean_loss_reject_empty() -> None:
    """空数据集没有准确率 / 损失。"""
    params = network.build_cnn(study.TRAIN_SPEC, (6, 6), seed=100)
    with pytest.raises(errors.ParameterError):
        train.accuracy(params, study.TRAIN_SPEC, ())
    with pytest.raises(errors.ParameterError):
        train.mean_loss(params, study.TRAIN_SPEC, ())


def test_train_cnn_learns_the_stripe_task() -> None:
    """在合成数据上真训练一次：损失显著下降、准确率达到 100%。"""
    dataset = train.make_stripe_dataset(noise=0.05, seed=5)
    _params, report = train.train_cnn(
        study.TRAIN_SPEC, dataset, steps=200, learning_rate=0.05, optimizer_name="adam", seed=100
    )
    assert report.final_loss < report.initial_loss
    assert report.accuracy == 1.0
    assert report.improvement > 0.9
    assert "adam" in report.summary_line()


def test_train_cnn_is_reproducible() -> None:
    """同一配置两次训练给出同一条损失曲线（LCG 决定一切）。"""
    dataset = train.make_stripe_dataset(noise=0.05, seed=5)
    first = train.train_cnn(study.TRAIN_SPEC, dataset, steps=20, seed=100)[1]
    second = train.train_cnn(study.TRAIN_SPEC, dataset, steps=20, seed=100)[1]
    assert first.losses == second.losses


def test_train_cnn_rejects_bad_steps_and_empty_dataset() -> None:
    """步数与数据集都有护栏。"""
    dataset = train.make_stripe_dataset(noise=0.05, seed=5)
    with pytest.raises(errors.ParameterError):
        train.train_cnn(study.TRAIN_SPEC, dataset, steps=0)
    with pytest.raises(errors.ParameterError):
        train.train_cnn(study.TRAIN_SPEC, ())


def test_train_report_improvement_zero_when_initial_zero() -> None:
    """初始损失为 0 时改善率记 0.0（不除零）。"""
    report = train.TrainReport(
        steps=1, losses=(0.0,), initial_loss=0.0, final_loss=0.0, best_loss=0.0, accuracy=1.0, optimizer="adam"
    )
    assert report.improvement == 0.0


# --------------------------------------------------------------------------- 性质校验


def test_check_semantics() -> None:
    """``Check`` 的相等 / 上界两类判据。"""
    assert verify.Check(reading=0.0, upper_bound=0.0).passed() is True
    assert verify.Check(reading=1e-13, upper_bound=0.0).passed() is False
    assert verify.Check(reading=0.5, upper_bound=1.0).passed() is True
    assert verify.Check(reading=0.0).bound_text() == "== 逐位"
    assert verify.Check(reading=0.0, upper_bound=0.0).bound_text() == "== 0"


def test_check_all_is_seven_and_all_pass() -> None:
    """七条性质全部通过（这就是"这条链被真的接上了"）。"""
    report = verify.check_all()
    assert report.total == 7
    assert report.all_passed()
    assert len(report.lines()) == 7


def test_each_check_returns_passing_outcome() -> None:
    """逐条跑：每条都返回通过的 ``PropertyOutcome`` 且读数有限。"""
    for name, function in verify.CHECKS.items():
        outcome = function()
        assert outcome.name == name
        assert outcome.passed is True
        assert math.isfinite(outcome.check.reading)
        assert outcome.line().startswith("通过")


def test_direct_receptive_field_matches_known_values() -> None:
    """区间传播在已知序列上的读数。"""
    assert verify.direct_receptive_field(((3, 1), (3, 1))) == 5
    assert verify.direct_receptive_field(((3, 1), (3, 1), (2, 2))) == 6
    assert verify.direct_receptive_field(()) == 1


# --------------------------------------------------------------------------- 七张表


def test_layer_rows() -> None:
    """层表一行，且公式里的参数数与读数一致。"""
    rows = study.layer_rows()
    assert len(rows) == 1
    assert "参数 20" in rows[0].line()


def test_size_rows_cover_cases() -> None:
    """尺寸表覆盖全部写死的用例，且读数与公式一致。"""
    rows = study.size_rows()
    assert len(rows) == len(study.SIZE_CASES)
    assert all(row.size >= 1 for row in rows)
    assert any(row.kernel == 3 and row.stride == 1 and row.padding == 1 and row.size == 6 for row in rows)


def test_field_rows_formula_equals_direct() -> None:
    """感受野表：公式与区间传播逐行相等。"""
    rows = study.field_rows()
    assert all(row.field == row.direct for row in rows)
    assert len(rows) == len(study.FIELD_CASES)


def test_feature_rows_cover_channels() -> None:
    """特征图表覆盖两个输出通道，且形状与层表一致。"""
    rows = study.feature_rows()
    assert len(rows) == study.SAMPLE_SPEC.out_channels
    assert all(row.height == 6 and row.width == 6 for row in rows)


def test_pool_rows_two_modes() -> None:
    """池化表两行（max / avg），且形状都降到 3×3。"""
    rows = study.pool_rows()
    assert [row.mode for row in rows] == ["max", "avg"]
    assert all(row.height == 3 and row.width == 3 for row in rows)


def test_train_rows_have_summary_and_points() -> None:
    """训练表：一行总结 + 若干个损失点。"""
    report = study.train_report()
    lines = study.train_rows(report)
    assert "准确率" in lines[0]
    assert len(lines) >= 3


def test_property_rows_are_seven_and_pass() -> None:
    """性质表七行，全部通过。"""
    rows = study.property_rows()
    assert len(rows) == 7
    assert all(row.passed for row in rows)
    assert all("vs" in row.cross_check for row in rows)


def test_study_lines_has_seven_sections() -> None:
    """一次跑完七张表：七个小节标题都在。"""
    lines = study.study_lines()
    text = "\n".join(lines)
    for index in range(1, 8):
        assert f"== {index}." in text


def test_property_names_alias() -> None:
    """``PROPERTY_NAMES`` 就是性质名单。"""
    assert study.PROPERTY_NAMES == types.CONV_PROPERTIES


# --------------------------------------------------------------------------- 包入口


def test_package_all_is_non_empty_and_has_no_submodules() -> None:
    """包入口的 ``__all__`` 非空、且不含子模块名。"""
    import smart_research_agent.conv_net as package

    assert len(package.__all__) > 60
    assert not (
        {"errors", "types", "ops", "layers", "gradients", "network", "train", "verify", "study"}
        & set(package.__all__)
    )


def test_package_star_import_resolves_every_name() -> None:
    """每个 ``__all__`` 里的名字都能在包命名空间里取到。"""
    import smart_research_agent.conv_net as package

    missing = [name for name in package.__all__ if not hasattr(package, name)]
    assert missing == []


def test_package_exposes_key_objects() -> None:
    """几个关键对象从包入口就能取到。"""
    import smart_research_agent.conv_net as package

    assert package.conv2d is ops.conv2d
    assert package.ConvSpec is layers.ConvSpec
    assert package.train_cnn is train.train_cnn
    assert package.Check is verify.Check
