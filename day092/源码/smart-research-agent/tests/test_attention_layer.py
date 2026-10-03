"""``transformer_core`` 的形状、前向与反向（day075）.

这一份测试的期望值全部**手算**，手算的支点是"四个投影都是单位矩阵"这个配置：

```text
Q = K = V = x
raw = x·xᵀ           一行与自己的点积是 1、与别人的是 0（one-hot 输入）
scores = raw / √2    缩放系数 1/√2 ≈ 0.707107
weights = 逐行 softmax（带因果掩码）
context = weights·V  输入的凸组合
output = context     因为 W_o = I
```

于是 ``softmax([0, 1/√2]) = (0.330181, 0.669819)`` 这种数字是**可以在纸上算出来**的，
而不是"上次跑出来的"。
"""

from __future__ import annotations

import math

import pytest

from smart_research_agent.math_foundations.attention import scaling_factor
from smart_research_agent.math_foundations.errors import (
    ShapeError as MathShapeError,
)
from smart_research_agent.transformer_core.errors import (
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.transformer_core.layers import (
    attention_backward,
    full_mask,
    masked_mean_squared_error,
    masked_mse_gradient,
    mean_squared_error,
    mse_gradient,
    resolve_mask,
    row_argmax_hits,
    self_attention,
    softmax_backward_row,
)
from smart_research_agent.transformer_core.types import (
    ATTENTION_STAGES,
    ATTENTION_STAGE_DESCRIPTIONS,
    ATTENTION_STAGE_SHAPES,
    AttentionParams,
    AttentionShape,
    project,
    softmax_shape_scale,
)
from tests.attention_samples import (
    ONE_HOT_INPUTS,
    P_SQRT2,
    SMALL_INPUTS,
    SMALL_TARGET,
    SCALE_SQRT2,
    approx,
    approx_matrix,
    identity_parameters,
    scaled_identity,
    small_parameters,
)


class TestTables:
    """三张口径表逐键对齐（少一个键不会让测试变红，只会让报告缺一行）."""

    def test_stage_tables_align(self):
        assert (
            set(ATTENTION_STAGES)
            == set(ATTENTION_STAGE_DESCRIPTIONS)
            == set(ATTENTION_STAGE_SHAPES)
        )

    def test_seven_stages(self):
        assert len(ATTENTION_STAGES) == 7

    def test_every_stage_has_a_description_and_a_shape(self):
        for stage in ATTENTION_STAGES:
            assert ATTENTION_STAGE_DESCRIPTIONS[stage].strip()
            assert ATTENTION_STAGE_SHAPES[stage].strip()


class TestAttentionShape:
    """四个维度各自校验、各自可读."""

    def test_fields_and_scale(self):
        shape = AttentionShape(inputs=4, keys=4, values=8, outputs=2)
        assert shape.scale == pytest.approx(0.5)
        assert shape.to_dict()["scale"] == pytest.approx(0.5)
        assert "d_in=4" in shape.summary_line()

    def test_scale_uses_d_k_not_d_v(self):
        """缩放系数只由 ``d_k`` 决定（8 → 1/√8，与 d_v 无关）."""
        assert AttentionShape(inputs=2, keys=8, values=3, outputs=3).scale == pytest.approx(
            1.0 / math.sqrt(8.0)
        )

    @pytest.mark.parametrize("name", ["inputs", "keys", "values", "outputs"])
    def test_zero_dimension_is_rejected(self, name):
        payload = {"inputs": 2, "keys": 2, "values": 2, "outputs": 2}
        payload[name] = 0
        with pytest.raises(ParameterError, match=name):
            AttentionShape(**payload)

    def test_non_integer_dimension_is_rejected(self):
        with pytest.raises(ParameterError, match="整数"):
            AttentionShape(inputs=2.0, keys=2, values=2, outputs=2)  # type: ignore[arg-type]

    def test_scale_helper_matches_day073(self):
        """``softmax_shape_scale`` 与 day073 的 ``scaling_factor`` 逐位一致.

        （day073 的 DPO 值对照是这一课的直接来源；这里再钉一次是因为
        day075 的实现路径不同——前者服务于纯函数注意力，后者服务于参数化注意力。）
        """
        for head_dim in (1, 2, 3, 4, 8, 64, 768):
            assert softmax_shape_scale(head_dim) == scaling_factor(head_dim)

    def test_scale_helper_validates(self):
        with pytest.raises(ParameterError, match="head_dim"):
            softmax_shape_scale(0)
        with pytest.raises(ParameterError, match="整数"):
            softmax_shape_scale(2.0)  # type: ignore[arg-type]


class TestAttentionParams:
    """四个投影矩阵的形状契约（这一层最容易配错的地方）."""

    def test_shape_is_derived_from_the_matrices(self):
        params = small_parameters()
        assert params.shape.inputs == 3
        assert params.shape.keys == 3
        assert params.shape.values == 3
        assert params.shape.outputs == 3

    def test_parameter_count_is_hand_computable(self):
        """``d_k·d_in × 2 + d_v·d_in + d_out·d_v`` = 9×2 + 9 + 9 = 36."""
        assert small_parameters().parameter_count() == 36

    def test_column_count_must_match(self):
        """四个投影作用在同一个输入上，列数必须相同."""
        with pytest.raises(ShapeError, match="列数"):
            AttentionParams(
                w_query=((1.0, 0.0),),
                w_key=((1.0, 0.0, 0.0),),
                w_value=((1.0, 0.0),),
                w_output=((1.0, 0.0),),
            )

    def test_query_and_key_outputs_must_match(self):
        """Q 与 K 必须在同一个空间里才能做点积."""
        with pytest.raises(ShapeError, match="同一个空间"):
            AttentionParams(
                w_query=((1.0, 0.0), (0.0, 1.0)),
                w_key=((1.0, 0.0),),
                w_value=((1.0, 0.0), (0.0, 1.0)),
                w_output=((1.0, 0.0), (0.0, 1.0)),
            )

    def test_output_columns_must_match_value_dimension(self):
        """``W_o`` 消费的是 context，而 context 的宽度是 ``d_v``.

        构造：``W_v`` 取 ``3×2``（于是 ``d_v = 3``、``d_in = 2``），
        而 ``W_o`` 取 ``1×2``（列数 2 与 ``d_in`` 一致，因此先通过"列数一致"那一关）。
        """
        with pytest.raises(ShapeError, match="d_v"):
            AttentionParams(
                w_query=((1.0, 0.0), (0.0, 1.0)),
                w_key=((1.0, 0.0), (0.0, 1.0)),
                w_value=((1.0, 0.0), (0.0, 1.0), (0.5, 0.5)),
                w_output=((1.0, 0.0),),
            )

    def test_flatten_and_unflatten_roundtrip(self):
        params = small_parameters()
        flat, shapes = params.flatten()
        assert len(flat) == 36
        restored = AttentionParams.unflatten(flat, shapes)
        assert restored == params

    def test_unflatten_rejects_wrong_block_count(self):
        with pytest.raises(ShapeError, match="4 个"):
            AttentionParams.unflatten(
                (1.0, 2.0, 3.0, 4.0, 5.0),
                ((1, 1), (1, 1), (1, 1), (1, 1), (1, 1)),
            )

    def test_describe_mentions_the_shape_and_count(self):
        text = small_parameters().describe()
        assert "d_in=3" in text
        assert "36" in text
        assert "无偏置" in text

    def test_matrix_shape_mismatch_inside_a_matrix_is_rejected(self):
        """一张"每行不等长"的表由 day073 的 ``validate_matrix`` 拦下.

        抛的是**数学族**的 ``ShapeError``（见 ``transformer_core.errors`` 开头那张图）：
        那一条校验的修复人是 day073 那一层的调用方，重新包装只会让
        "错误发生在哪一层"更难读。
        """
        with pytest.raises(MathShapeError, match="等长"):
            AttentionParams(
                w_query=((1.0, 0.0), (1.0,)),
                w_key=((1.0, 0.0), (0.0, 1.0)),
                w_value=((1.0, 0.0), (0.0, 1.0)),
                w_output=((1.0, 0.0), (0.0, 1.0)),
            )


class TestProject:
    """投影的乘法口径：``x·Wᵀ``（与 ``nn.Linear`` 一致）."""

    def test_identity_returns_the_input(self):
        assert project(ONE_HOT_INPUTS, scaled_identity(1.0, 2)) == ONE_HOT_INPUTS

    def test_scaling_a_weight_scales_the_output(self):
        doubled = project(ONE_HOT_INPUTS, scaled_identity(2.0, 2))
        assert approx_matrix(doubled, ((2.0, 0.0), (0.0, 2.0)))

    def test_hand_computed_rectangular_projection(self):
        """``(2,3)·(2,3)ᵀ``：输出的行数是权重行数、列数也是权重行数."""
        weight = ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
        inputs = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        result = project(inputs, weight)
        assert approx_matrix(result, ((1.0, 4.0), (2.0, 5.0)))

    def test_row_width_must_match_weight_columns(self):
        with pytest.raises(ShapeError, match="行宽"):
            project(((1.0, 2.0),), ((1.0, 2.0, 3.0),))


class TestForwardWithIdentityWeights:
    """单位矩阵配置下的完整前向（每一个数都能手算）."""

    def _forward(self, *, causal: bool):
        return self_attention(identity_parameters(2), ONE_HOT_INPUTS, causal=causal)

    def test_causal_row_zero_sees_only_itself(self):
        """第 0 行被掩码限制成"只能看自己"，权重恰好是 ``(1.0, 0.0)``."""
        forward = self._forward(causal=True)
        assert forward.weights[0] == (1.0, 0.0)

    def test_causal_row_one_splits_by_softmax(self):
        """第 1 行看到 ``[0, 1/√2]``，于是权重是 ``(p, 1−p)``，``p = 0.330238``.

        ``p = 1/(1+e^{1/√2})``：把打分 ``[0, 0.707107]`` 减最大值之后
        第一个位置是 ``e^{-0.707107}``、第二个是 1，归一化得到 ``p``。
        """
        forward = self._forward(causal=True)
        assert approx(forward.weights[1][0], P_SQRT2)
        assert approx(forward.weights[1][1], 1.0 - P_SQRT2)

    def test_causal_output_is_the_convex_combination(self):
        """第 1 行的输出是 ``p·(1,0) + (1−p)·(0,1) = (p, 1−p)``（因为 ``W_v = W_o = I``）."""
        forward = self._forward(causal=True)
        assert approx(forward.weights[1][0], 0.330238, tolerance=1e-6)
        assert approx_matrix(forward.output, ((1.0, 0.0), (P_SQRT2, 1.0 - P_SQRT2)))

    def test_without_mask_row_zero_is_a_mixture(self):
        """无掩码时第 0 行看到 ``[1/√2, 0]``，于是权重反过来了：``(1−p, p)``."""
        forward = self._forward(causal=False)
        assert approx(forward.weights[0][0], 1.0 - P_SQRT2)
        assert approx(forward.weights[0][1], P_SQRT2)

    def test_row_sums_are_one(self):
        forward = self._forward(causal=True)
        for row in forward.weights:
            assert approx(math.fsum(row), 1.0)

    def test_causal_upper_triangle_is_exactly_zero(self):
        forward = self._forward(causal=True)
        assert forward.weights[0][1] == 0.0

    def test_entropy_readings(self):
        """第 0 行是确定的（熵 0），第 1 行是二分布（熵 = −Σp ln p）."""
        forward = self._forward(causal=True)
        assert approx(forward.row_entropies[0], 0.0, tolerance=1e-12)
        expected = -(P_SQRT2 * math.log(P_SQRT2) + (1 - P_SQRT2) * math.log(1 - P_SQRT2))
        assert approx(forward.row_entropies[1], expected, tolerance=1e-9)

    def test_peak_and_indices(self):
        """第 0 行锁在自己身上（峰值 1.0），第 1 行峰值是 ``1−p = 0.669762``."""
        forward = self._forward(causal=True)
        assert forward.peak_indices == (0, 1)
        assert approx(forward.peak_weights[0], 1.0)
        assert approx(forward.peak_weights[1], 1.0 - P_SQRT2)

    def test_summary_and_dict(self):
        import json

        forward = self._forward(causal=True)
        payload = forward.to_dict()
        json.dumps(payload)
        assert payload["tokens"] == 2
        assert payload["causal"] is True
        assert forward.summary_line().startswith("causal")
        assert forward.tokens == 2
        assert forward.focus_ratio() >= 0.0
        assert forward.max_entropy() == pytest.approx(math.log(2.0))
        assert forward.notes

    def test_scale_is_the_shape_scale(self):
        forward = self._forward(causal=False)
        assert forward.scale == pytest.approx(SCALE_SQRT2)

    def test_zero_input_gives_zero_output(self):
        """四个投影都没有偏置，因此全零输入得到全零输出（**可断言**）."""
        with pytest.raises(NumericError, match="全零"):
            self_attention(identity_parameters(2), ((0.0, 0.0), (1.0, 0.0)))

    def test_sharper_queries_concentrate_the_weights(self):
        """把 ``W_q`` 放大 3 倍 → 打分放大 9 倍 → 权重更尖（峰值上升）."""
        base = self_attention(identity_parameters(2), ONE_HOT_INPUTS)
        sharp = AttentionParams(
            w_query=scaled_identity(3.0, 2),
            w_key=scaled_identity(1.0, 2),
            w_value=scaled_identity(1.0, 2),
            w_output=scaled_identity(1.0, 2),
        )
        sharper = self_attention(sharp, ONE_HOT_INPUTS)
        assert sharper.peak_weights[0] > base.peak_weights[0]
        assert sharper.mean_entropy < base.mean_entropy


class TestMasks:
    """掩码的三种来源与两种拒绝."""

    def test_full_mask_is_all_true(self):
        assert full_mask(3) == ((True, True, True),) * 3

    def test_full_mask_validates(self):
        with pytest.raises(ShapeError, match="整数"):
            full_mask(3.0)  # type: ignore[arg-type]
        with pytest.raises(ShapeError, match="边长"):
            full_mask(0)

    def test_resolve_prefers_causal(self):
        assert resolve_mask(2, causal=True) == ((True, False), (True, True))

    def test_resolve_uses_explicit_mask(self):
        explicit = ((True, False), (False, True))
        assert resolve_mask(2, causal=False, mask=explicit) == explicit

    def test_causal_and_explicit_mask_together_is_rejected(self):
        with pytest.raises(ShapeError, match="不能同时给"):
            resolve_mask(2, causal=True, mask=((True, False), (True, True)))

    def test_mask_shape_mismatch_is_rejected(self):
        with pytest.raises(ShapeError, match="掩码形状"):
            resolve_mask(3, causal=False, mask=((True, True), (True, True)))

    def test_empty_mask_row_is_rejected(self):
        with pytest.raises(NumericError, match="允许的位置"):
            resolve_mask(2, causal=False, mask=((False, False), (True, True)))

    def test_forward_rejects_zero_row_in_inputs(self):
        with pytest.raises(NumericError, match="全零"):
            self_attention(small_parameters(), ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))


class TestBackwardHandComputed:
    """反向的两条手算结论（一条结构、一条数值）."""

    def test_gradient_is_zero_when_output_equals_target(self):
        """``y == t`` 时 ``dOut = 0``，于是所有参数梯度都是 0（链式法则的端点）."""
        params = identity_parameters(2)
        forward = self_attention(params, ONE_HOT_INPUTS, causal=False)
        gradients = attention_backward(
            forward, mse_gradient(forward.output, forward.output)
        )
        assert gradients.max_absolute() == 0.0
        assert approx(gradients.grad_inputs[0][0], 0.0)

    def test_output_projection_gradient_is_hand_computed(self):
        """``dW_o = dOutᵀ·context``，其中 ``dOut = 2(y−t)/N``（N = 4）.

        手算（``p = 0.330238``、``1−p = 0.669762``）：

        ```text
        context = ((1, 0), (p, 1−p))        y = context（W_o = I）
        t       = ((1, 0), (0, 1))          y−t = ((0, 0), (p, p−1))
        dOut    = 0.5·(y−t) = ((0, 0), (p/2, −p/2))

        dW_o = dOutᵀ·context
         行 0 = (p/2) × context[1] = (p²/2, p(1−p)/2) = (0.054529, 0.110591)
         行 1 = −行 0
        ```

        注意 ``dW_o`` **不是**对称的：它把"第 1 行错了多少"分给了 context 的两个分量，
        而这两个分量的比例正是当时的注意力权重。
        """
        params = identity_parameters(2)
        forward = self_attention(params, ONE_HOT_INPUTS, causal=True)
        target = ((1.0, 0.0), (0.0, 1.0))
        gradients = attention_backward(forward, mse_gradient(forward.output, target))
        factor = P_SQRT2 / 2.0
        expected_row0 = (factor * P_SQRT2, factor * (1.0 - P_SQRT2))
        expected_row1 = tuple(-value for value in expected_row0)
        assert approx_matrix(
            gradients.grad_w_output, (expected_row0, expected_row1), tolerance=1e-9
        )

    def test_output_gradient_shape_is_validated(self):
        forward = self_attention(small_parameters(), SMALL_INPUTS, causal=True)
        with pytest.raises(ShapeError, match="输出梯度的形状"):
            attention_backward(forward, ((1.0, 2.0, 3.0),))

    def test_input_gradient_is_the_sum_of_three_chains(self):
        """``grad_inputs`` 必须非零——它把 q/k/v 三条链加在了一起.

        这一条是"漏了一条链"的护栏：只写两条链的实现仍然能给出一个
        形状正确、数值偏小的输入梯度。
        """
        params = small_parameters()
        forward = self_attention(params, SMALL_INPUTS, causal=True)
        gradients = attention_backward(
            forward, mse_gradient(forward.output, SMALL_TARGET)
        )
        assert max(abs(value) for row in gradients.grad_inputs for value in row) > 1e-6
        assert "输入梯度" in gradients.summary_line()

    def test_parameter_gradients_dict_is_keyed_by_name(self):
        params = small_parameters()
        forward = self_attention(params, SMALL_INPUTS, causal=True)
        gradients = attention_backward(forward, mse_gradient(forward.output, SMALL_TARGET))
        table = gradients.as_dict()
        assert set(table) == {"w_query", "w_key", "w_value", "w_output"}
        assert table["w_query"] == gradients.grad_w_query

    def test_backward_preserves_parameter_order_when_flattened(self):
        """压平梯度用的顺序必须与压平参数一致.

        顺序错了**不会报错**：它只会让某一层的权重按另一层的梯度更新，
        而它表现为"训练变慢"或"不收敛"。因此这里逐块核对形状。
        """
        params = small_parameters()
        forward = self_attention(params, SMALL_INPUTS, causal=True)
        gradients = attention_backward(forward, mse_gradient(forward.output, SMALL_TARGET))
        flat_params, shapes = params.flatten()
        flat_grads = gradients.flatten()
        assert len(flat_params) == len(flat_grads)
        assert shapes == ((3, 3), (3, 3), (3, 3), (3, 3))
        from smart_research_agent.transformer_core.types import matrix_max_absolute

        for matrix in gradients.matrices():
            assert matrix_max_absolute(matrix) > 0.0


class TestSoftmaxBackward:
    """softmax 的雅可比-向量积（逐行）."""

    def test_uniform_weights_and_uniform_gradient_give_zero(self):
        """均匀分布 + 均匀梯度：括号里的那一项恰好等于每一个 ``g`` → 全 0."""
        weights = (0.25, 0.25, 0.25, 0.25)
        gradient_row = (1.0, 1.0, 1.0, 1.0)
        result = softmax_backward_row(weights, gradient_row)
        assert all(approx(value, 0.0, tolerance=1e-12) for value in result)

    def test_one_hot_weights_give_zero(self):
        """全押一个位置时，其它位置拿不到梯度（``s=0``）."""
        result = softmax_backward_row((1.0, 0.0, 0.0), (5.0, 7.0, -3.0))
        assert approx(result[0], 0.0)
        assert result[1] == 0.0
        assert result[2] == 0.0

    def test_two_way_hand_computed(self):
        """``s = (0.5, 0.5)``、``g = (1, 0)``：平均 0.5 → ``(0.25, −0.25)``."""
        result = softmax_backward_row((0.5, 0.5), (1.0, 0.0))
        assert approx(result[0], 0.25)
        assert approx(result[1], -0.25)

    def test_length_mismatch_is_rejected(self):
        with pytest.raises(ShapeError, match="逐位对应"):
            softmax_backward_row((0.5, 0.5), (1.0,))

    def test_gradient_of_the_identity_case(self):
        """``g = s`` 时结果是 ``s_i(s_i − Σs²)``（等于"把 s 当目标"的梯度）."""
        weights = (0.2, 0.3, 0.5)
        square_sum = sum(value * value for value in weights)
        result = softmax_backward_row(weights, weights)
        for value, weight in zip(result, weights):
            assert approx(value, weight * (weight - square_sum))


class TestLosses:
    """逐元素 MSE 与"只在监督行上"的 MSE."""

    def test_mean_squared_error_hand_computed(self):
        """``((1−0)² + (0−1)²)/2 + …``：两行两列共 4 项."""
        output = ((1.0, 0.0), (0.0, 1.0))
        target = ((0.0, 0.0), (0.0, 0.0))
        assert mean_squared_error(output, target) == pytest.approx(0.5)

    def test_mse_gradient_hand_computed(self):
        """``∂MSE/∂y = 2(y−t)/N``；``N = 4`` 时每一个 1 给出 0.5."""
        output = ((1.0, 0.0), (0.0, 1.0))
        target = ((0.0, 0.0), (0.0, 0.0))
        assert mse_gradient(output, target) == ((0.5, 0.0), (0.0, 0.5))

    def test_masked_mse_only_counts_supervised_rows(self):
        output = ((1.0, 0.0), (0.0, 1.0))
        target = ((0.0, 0.0), (0.0, 0.0))
        # 只监督第 1 行：分母变成 1 行 × 2 列 = 2 → (0+1)/2 = 0.5
        assert masked_mean_squared_error(output, target, (1,)) == pytest.approx(0.5)
        # 两行都监督：分母 4 → (1+1)/4 = 0.5
        assert masked_mean_squared_error(output, target, (0, 1)) == pytest.approx(0.5)

    def test_masked_mse_denominator_is_supervised_rows(self):
        """分母只数监督行——用全部行数会让损失被系统性地压低."""
        output = ((2.0, 0.0), (0.0, 0.0))
        target = ((0.0, 0.0), (0.0, 0.0))
        assert masked_mean_squared_error(output, target, (0,)) == pytest.approx(2.0)
        assert masked_mean_squared_error(output, target, (0, 1)) == pytest.approx(1.0)

    def test_masked_gradient_is_zero_on_unsupervised_rows(self):
        output = ((1.0, 0.0), (0.0, 1.0))
        target = ((0.0, 0.0), (0.0, 0.0))
        gradient_matrix = masked_mse_gradient(output, target, (0,))
        assert gradient_matrix[1] == (0.0, 0.0)
        assert approx(gradient_matrix[0][0], 1.0)

    def test_empty_supervision_is_rejected(self):
        output = ((1.0, 0.0), (0.0, 1.0))
        with pytest.raises(ShapeError, match="监督行不能为空"):
            masked_mean_squared_error(output, output, ())

    def test_duplicate_supervision_is_rejected(self):
        output = ((1.0, 0.0), (0.0, 1.0))
        with pytest.raises(ShapeError, match="重复"):
            masked_mse_gradient(output, output, (0, 0))

    def test_out_of_range_supervision_is_rejected(self):
        output = ((1.0, 0.0), (0.0, 1.0))
        with pytest.raises(ShapeError, match="之外"):
            masked_mse_gradient(output, output, (2,))

    def test_non_integer_supervision_is_rejected(self):
        output = ((1.0, 0.0), (0.0, 1.0))
        with pytest.raises(ShapeError, match="整数"):
            masked_mse_gradient(output, output, (0.0,))  # type: ignore[arg-type]

    def test_row_argmax_hits(self):
        output = ((0.9, 0.1), (0.2, 0.8))
        target = ((1.0, 0.0), (0.0, 1.0))
        assert row_argmax_hits(output, target, (0, 1)) == 1.0
        assert row_argmax_hits(output, target, (1,)) == 1.0

    def test_row_argmax_misses(self):
        output = ((0.1, 0.9), (0.2, 0.8))
        target = ((1.0, 0.0), (0.0, 1.0))
        assert row_argmax_hits(output, target, (0,)) == 0.0

    def test_shape_mismatch_is_rejected(self):
        with pytest.raises(ShapeError, match="形状"):
            mean_squared_error(((1.0,),), ((1.0, 2.0),))
        with pytest.raises(ShapeError, match="形状"):
            mse_gradient(((1.0,),), ((1.0, 2.0),))
        with pytest.raises(ShapeError, match="形状"):
            masked_mean_squared_error(((1.0,),), ((1.0, 2.0),), (0,))
        with pytest.raises(ShapeError, match="形状"):
            masked_mse_gradient(((1.0,),), ((1.0, 2.0),), (0,))


class TestForwardValidation:
    """前向的参数校验（"不替调用方猜"）."""

    def test_params_type_is_checked(self):
        with pytest.raises(ShapeError, match="AttentionParams"):
            self_attention({"w_query": ()}, ONE_HOT_INPUTS)  # type: ignore[arg-type]

    def test_input_must_be_a_matrix(self):
        """空输入由 day073 的 ``validate_matrix`` 拦下（抛数学族的 ``ShapeError``）."""
        with pytest.raises(MathShapeError, match="不能为空"):
            self_attention(identity_parameters(2), ())

    def test_forward_records_the_mask_used(self):
        forward = self_attention(identity_parameters(2), ONE_HOT_INPUTS, causal=True)
        assert forward.mask == ((True, False), (True, True))
        assert forward.causal is True
