"""LoRA 矩阵算术与参考层的单元测试（day051）.

本文件里最重要的两条用例：

1. ``test_gradients_match_numerical_differences``——用**中心差分**核对
   ``∂L/∂A`` 与 ``∂L/∂B`` 的解析式。LoRA 的增量本身很小（``B`` 初始为 0），
   数值差分的误差量级与增量接近，因此这条用例是"梯度算得对不对"唯一
   可信的判据；
2. ``test_forward_row_differs_from_column_convention``——把"行约定"与
   "列约定"的差别钉死。实现时真的踩到过这个坑：把按列算的增量叠加到
   按行取用的基座上，loss 曲线完全正常，但模型与合并结果对不上。
"""

from __future__ import annotations

import random

import pytest

from smart_research_agent.peft.config import LoRAConfig, PEFTConfigError
from smart_research_agent.peft.layers import (
    ZERO_TOLERANCE,
    LoRALinear,
    adapter_state_size,
    add_matrices,
    context_delta,
    describe_delta,
    frobenius_norm,
    init_lora_weights,
    is_rank_one,
    lora_delta,
    matmul,
    matrix_rank,
    max_abs,
    merge_lora_weight,
)

#: 一个小而形状不规则的基座矩阵（3×4），用于逐元素核对
BASE = [[0.3, -0.2, 0.5, 0.1], [-0.4, 0.2, 0.0, 0.25], [0.1, 0.15, -0.3, 0.05]]


def small_config(**overrides) -> LoRAConfig:
    """小矩阵上可用的 LoRA 配置（关闭 dropout，保证确定性）."""
    base = LoRAConfig(r=2, lora_alpha=4, lora_dropout=0.0, target_modules=("weight",))
    return base.with_overrides(**overrides) if overrides else base


def make_layer(base=BASE, config=None, *, seed: int = 7) -> LoRALinear:
    """构造一个测试用层（seed 固定 → A 可复现）."""
    return LoRALinear(base, config or small_config(), seed=seed)


def probe_layer(base, config, a_matrix, b_matrix, *, seed: int = 7) -> LoRALinear:
    """构造一个载入了指定 ``A`` / ``B`` 的层（用于数值差分）."""
    layer = LoRALinear(base, config, seed=seed)
    layer.load_adapter_state(
        {"a": a_matrix, "b": b_matrix, "updates": 0, "scaling": layer.scaling}
    )
    return layer


class TestMatrixHelpers:
    """矩阵工具：矩形校验、形状校验、算术正确性."""

    def test_matmul_identity(self):
        result = matmul([[1.0, 0.0], [0.0, 1.0]], [[3.0], [4.0]])
        assert result == [[3.0], [4.0]]

    def test_matmul_shape(self):
        result = matmul(BASE, [[1.0], [1.0], [1.0], [1.0]])
        assert len(result) == 3 and len(result[0]) == 1
        assert result[0][0] == pytest.approx(sum(BASE[0]))

    def test_matmul_rejects_dimension_mismatch(self):
        with pytest.raises(PEFTConfigError, match="维度不匹配"):
            matmul([[1.0, 2.0]], [[1.0, 2.0]])

    def test_matmul_rejects_empty(self):
        with pytest.raises(PEFTConfigError, match="不能为空矩阵"):
            matmul([], [[1.0]])

    def test_matmul_rejects_zero_columns(self):
        with pytest.raises(PEFTConfigError, match="列数不能为 0"):
            matmul([[]], [[1.0]])

    def test_matmul_rejects_ragged(self):
        with pytest.raises(PEFTConfigError, match="不是矩形"):
            matmul([[1.0, 2.0], [3.0]], [[1.0], [2.0]])

    def test_add_matrices(self):
        result = add_matrices([[1.0, 2.0]], [[0.5, -0.5]])
        assert result == [[1.5, 1.5]]

    def test_add_matrices_rejects_shape_mismatch(self):
        with pytest.raises(PEFTConfigError, match="形状不一致"):
            add_matrices([[1.0]], [[1.0, 2.0]])


class TestDeltaAndMerge:
    """增量矩阵、合并权重、范数：与手算逐项对照."""

    def test_delta_scales_product(self):
        a_matrix = [[1.0, 2.0], [3.0, 4.0]]
        b_matrix = [[1.0, 0.0], [0.0, 1.0]]
        delta = lora_delta(a_matrix, b_matrix, 0.5)
        # B 是单位阵 → B @ A == A，再乘缩放 0.5
        assert delta == [[0.5, 1.0], [1.5, 2.0]]

    def test_delta_shape_is_out_by_in(self):
        a_matrix, b_matrix = init_lora_weights(6, 5, 3, seed=1, init_mode="random")
        assert len(lora_delta(a_matrix, b_matrix, 2.0)) == 5
        assert len(lora_delta(a_matrix, b_matrix, 2.0)[0]) == 6

    def test_merge_equals_base_plus_delta(self):
        a_matrix, b_matrix = init_lora_weights(4, 3, 2, seed=2, init_mode="random")
        merged = merge_lora_weight(BASE, a_matrix, b_matrix, 1.5)
        delta = lora_delta(a_matrix, b_matrix, 1.5)
        assert merged == add_matrices(BASE, delta)

    def test_merge_with_zero_delta_returns_base(self):
        a_matrix, b_matrix = init_lora_weights(4, 3, 2, seed=2)
        assert merge_lora_weight(BASE, a_matrix, b_matrix, 2.0) == BASE

    def test_context_delta_matches_delta_matrix_column(self):
        """``context_delta`` 取的是 ``ΔW`` 的一**列**（不是一行）."""
        a_matrix, b_matrix = init_lora_weights(4, 3, 2, seed=5, init_mode="random")
        delta = lora_delta(a_matrix, b_matrix, 3.0)
        for index in range(4):
            column = [delta[row][index] for row in range(3)]
            assert context_delta(a_matrix, b_matrix, index, 3.0) == pytest.approx(column)

    def test_context_delta_index_out_of_range(self):
        a_matrix, b_matrix = init_lora_weights(4, 3, 2, seed=5)
        with pytest.raises(PEFTConfigError, match="越界"):
            context_delta(a_matrix, b_matrix, 4, 1.0)

    def test_frobenius_and_max_abs(self):
        matrix = [[3.0, 4.0], [0.0, 0.0]]
        assert frobenius_norm(matrix) == pytest.approx(5.0)
        assert max_abs(matrix) == 4.0

    def test_frobenius_rejects_empty(self):
        with pytest.raises(PEFTConfigError, match="不能为空矩阵"):
            frobenius_norm([])

    def test_adapter_state_size(self):
        a_matrix, b_matrix = init_lora_weights(11, 7, 3, seed=1)
        assert adapter_state_size(a_matrix, b_matrix) == 3 * (11 + 7)

    def test_adapter_state_size_rejects_rank_mismatch(self):
        with pytest.raises(PEFTConfigError, match="必须等于 A 的行数"):
            adapter_state_size([[0.0, 0.0]], [[0.0, 0.0]])


class TestRankBound:
    """秩上界：``rank(ΔW) ≤ min(r, in, out)`` —— LoRA 唯一能被钉死的结构性质."""

    @pytest.mark.parametrize("r", [1, 2, 3])
    def test_rank_never_exceeds_r(self, r: int):
        a_matrix, b_matrix = init_lora_weights(6, 6, r, seed=3, init_mode="random")
        assert matrix_rank(lora_delta(a_matrix, b_matrix, 2.0)) <= r

    def test_rank_one_when_r_is_one(self):
        a_matrix, b_matrix = init_lora_weights(8, 6, 1, seed=3, init_mode="random")
        delta = lora_delta(a_matrix, b_matrix, 1.0)
        assert matrix_rank(delta) == 1
        assert is_rank_one(delta) is True

    def test_rank_capped_by_smaller_dimension(self):
        # r=6 但矩阵只有 5×6 → 秩最多 5
        a_matrix, b_matrix = init_lora_weights(6, 5, 6, seed=3, init_mode="random")
        assert matrix_rank(lora_delta(a_matrix, b_matrix, 1.0)) == 5

    def test_zero_matrix_has_rank_zero(self):
        a_matrix, b_matrix = init_lora_weights(4, 4, 2, seed=3)
        assert matrix_rank(lora_delta(a_matrix, b_matrix, 1.0)) == 0
        assert is_rank_one(lora_delta(a_matrix, b_matrix, 1.0)) is True

    def test_rank_tolerance_drops_small_pivots(self):
        """缺省容差 1e-9 会把 ``1e-12`` 的主元判为零；放宽容差后秩回升."""
        matrix = [[1.0, 0.0], [0.0, 1e-12]]
        assert matrix_rank(matrix) == 1
        assert matrix_rank(matrix, tolerance=1e-15) == 2

    def test_rank_requires_rectangular(self):
        with pytest.raises(PEFTConfigError, match="不能为空矩阵"):
            matrix_rank([])


class TestInitWeights:
    """初始化：``B`` 全零保证 ``ΔW = 0``；两种模式的区别可被测出."""

    def test_gaussian_mode_zeros_b(self):
        a_matrix, b_matrix = init_lora_weights(5, 4, 2, seed=11)
        assert all(value == 0.0 for row in b_matrix for value in row)
        assert any(value != 0.0 for row in a_matrix for value in row)

    def test_random_mode_fills_b(self):
        _, b_matrix = init_lora_weights(5, 4, 2, seed=11, init_mode="random")
        assert any(value != 0.0 for row in b_matrix for value in row)

    def test_reproducible_with_same_seed(self):
        first = init_lora_weights(5, 4, 2, seed=11, init_mode="random")
        second = init_lora_weights(5, 4, 2, seed=11, init_mode="random")
        assert first == second

    def test_different_seed_changes_values(self):
        first = init_lora_weights(5, 4, 2, seed=11, init_mode="random")
        second = init_lora_weights(5, 4, 2, seed=12, init_mode="random")
        assert first != second

    def test_std_scales_with_fan_in(self):
        """``A`` 的标准差按 ``1/sqrt(in)`` 缩放：输入维度变大时量级变小."""
        wide, _ = init_lora_weights(400, 4, 4, seed=1)
        narrow, _ = init_lora_weights(4, 4, 4, seed=1)
        wide_rms = (sum(v * v for row in wide for v in row) / (4 * 400)) ** 0.5
        narrow_rms = (sum(v * v for row in narrow for v in row) / (4 * 4)) ** 0.5
        assert wide_rms < narrow_rms

    @pytest.mark.parametrize(
        ("in_features", "out_features", "r", "mode"),
        [(0, 4, 2, "gaussian"), (4, 0, 2, "gaussian"), (4, 4, 0, "gaussian"),
         (4, 4, 2, "loftq")],
    )
    def test_invalid_arguments(self, in_features, out_features, r, mode):
        with pytest.raises(PEFTConfigError):
            init_lora_weights(
                in_features, out_features, r, init_mode=mode
            )

    def test_invalid_std_scale(self):
        with pytest.raises(PEFTConfigError, match="std_scale"):
            init_lora_weights(4, 4, 2, std_scale=0.0)


class TestLoRALinear:
    """参考层：前向、反向、更新、序列化的完整行为."""

    def test_shapes_and_counts(self):
        layer = make_layer()
        assert (layer.in_features, layer.out_features, layer.r) == (4, 3, 2)
        assert layer.frozen_parameters == 12
        assert layer.trainable_parameters == 2 * (4 + 3)
        assert layer.trainable_ratio == pytest.approx(14 / 26)

    def test_scaling_follows_config(self):
        layer = make_layer(config=small_config(lora_alpha=8))
        assert layer.scaling == pytest.approx(4.0)

    def test_delta_is_zero_before_first_update(self):
        layer = make_layer()
        assert layer.delta_is_zero() is True
        assert layer.delta() == [[0.0] * 4 for _ in range(3)]

    def test_forward_equals_base_at_init(self):
        layer = make_layer()
        output = layer.forward([1.0, 2.0, 3.0, 4.0])
        expected = [sum(row[i] * v for i, v in enumerate([1.0, 2.0, 3.0, 4.0])) for row in BASE]
        assert output == pytest.approx(expected)

    def test_forward_equals_merged_weight_product_after_training(self):
        layer = make_layer()
        for _ in range(3):
            layer.accumulate([0.5, -1.0, 2.0, 0.25], [1.0, -1.0, 0.5])
        layer.apply_update(0.05)
        x = [0.3, 0.7, -0.2, 0.9]
        merged = layer.merged_weight()
        expected = [sum(row[i] * x[i] for i in range(4)) for row in merged]
        assert layer.forward(x) == pytest.approx(expected)

    def test_forward_rejects_wrong_input_dimension(self):
        with pytest.raises(PEFTConfigError, match="输入维度不匹配"):
            make_layer().forward([1.0, 2.0])

    def test_base_weight_is_a_copy(self):
        layer = make_layer()
        copy = layer.base_weight
        copy[0][0] = 999.0
        assert layer.base_weight[0][0] == BASE[0][0]

    def test_accumulate_and_apply_update_counts(self):
        layer = make_layer()
        layer.accumulate([1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        assert layer.pending_positions == 1
        layer.accumulate([0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        assert layer.pending_positions == 2
        layer.apply_update(0.1)
        assert layer.updates == 1
        assert layer.pending_positions == 0

    def test_apply_update_requires_gradients(self):
        with pytest.raises(PEFTConfigError, match="没有待应用的梯度"):
            make_layer().apply_update(0.1)

    @pytest.mark.parametrize("learning_rate", [0.0, -1.0])
    def test_apply_update_rejects_non_positive_lr(self, learning_rate: float):
        layer = make_layer()
        layer.accumulate([1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        with pytest.raises(PEFTConfigError, match="learning_rate"):
            layer.apply_update(learning_rate)

    def test_zero_grad_discards_pending(self):
        layer = make_layer()
        layer.accumulate([1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        layer.zero_grad()
        assert layer.pending_positions == 0
        with pytest.raises(PEFTConfigError, match="没有待应用的梯度"):
            layer.apply_update(1.0)

    def test_backward_requires_forward(self):
        with pytest.raises(PEFTConfigError, match="必须先调用 forward"):
            make_layer().backward([1.0, 0.0, 0.0])

    def test_backward_rejects_wrong_gradient_dimension(self):
        layer = make_layer()
        layer.forward([1.0, 0.0, 0.0, 0.0])
        with pytest.raises(PEFTConfigError, match="输出梯度维度不匹配"):
            layer.backward([1.0, 0.0])

    def test_gradients_match_numerical_differences(self):
        """解析梯度 vs 中心差分：``∂L/∂A`` 与 ``∂L/∂B`` 逐元素核对.

        ``L = Σ_i g_i · out_i``，因此上游梯度就是 ``g``。差分用同一份
        ``A`` / ``B`` 重建一个层（``load_adapter_state`` 是公开入口），
        步长 ``h = 1e-6``，容差 ``1e-4``。

        **必须先做一次更新再核对**：``B`` 初始全零时 ``∂L/∂A`` 恒等于 0
        （``∂L/∂a_j ∝ Σ_i g_i B[i][j] = 0``），此时"A 的梯度算得对不对"
        这条断言会自动通过——**用退化输入验证出来的正确性是假的**。
        """
        config = small_config()
        layer = make_layer(config=config)
        # 先更新一次，让 B 离开全零；否则 ∂L/∂A ≡ 0，核对毫无信息量
        layer.accumulate([1.0, 0.0, 0.0, 0.0], [0.5, -0.25, 0.75])
        layer.apply_update(0.4)
        layer.zero_grad()

        a_matrix, b_matrix = layer.snapshot_matrices()
        assert any(value != 0.0 for row in b_matrix for value in row)
        x = [0.4, -0.9, 0.7, 0.2]
        g = [0.8, -1.2, 0.35]

        layer.forward(x)
        layer.backward(g)
        grad_a, grad_b, pending = layer.gradient_snapshot()
        assert pending == 1

        def loss(a_values, b_values) -> float:
            probe = probe_layer(BASE, config, a_values, b_values)
            return sum(gi * oi for gi, oi in zip(g, probe.forward(x)))

        step = 1e-6
        for j in range(len(a_matrix)):
            for k in range(len(a_matrix[j])):
                plus = [row[:] for row in a_matrix]
                minus = [row[:] for row in a_matrix]
                plus[j][k] += step
                minus[j][k] -= step
                numeric = (loss(plus, b_matrix) - loss(minus, b_matrix)) / (2 * step)
                assert grad_a[j][k] == pytest.approx(numeric, abs=1e-4)

        for i in range(len(b_matrix)):
            for j in range(len(b_matrix[i])):
                plus = [row[:] for row in b_matrix]
                minus = [row[:] for row in b_matrix]
                plus[i][j] += step
                minus[i][j] -= step
                numeric = (loss(a_matrix, plus) - loss(a_matrix, minus)) / (2 * step)
                assert grad_b[i][j] == pytest.approx(numeric, abs=1e-4)

    def test_first_update_only_moves_b(self):
        """``B`` 初始全零时 ``∂L/∂A`` 恒为 0：第一次更新只动 ``B``.

        这是 LoRA 初始化设计带来的一个直接推论，也让"训练的前几步在学什么"
        变得可解释：适配器先学"输出方向"（``B``），再学"输入投影"（``A``）。
        """
        layer = make_layer()
        before_a, before_b = layer.snapshot_matrices()
        layer.accumulate([1.0, 0.0, 0.0, 0.0], [0.5, -0.25, 0.75])
        layer.apply_update(0.5)
        after_a, after_b = layer.snapshot_matrices()
        assert after_a == before_a
        assert after_b != before_b

    def test_one_hot_forward_equals_dense_forward(self):
        layer = make_layer()
        for index in range(4):
            one_hot = [0.0] * 4
            one_hot[index] = 1.0
            assert layer.forward_onehot(index) == layer.forward(one_hot)

    def test_one_hot_backward_matches_dense_backward(self):
        dense = make_layer()
        sparse = make_layer()
        grad = [0.3, -0.8, 0.5]
        for index in (0, 3):
            one_hot = [0.0] * 4
            one_hot[index] = 1.0
            dense.forward(one_hot)
            dense.backward(grad)
            sparse.forward_onehot(index)
            sparse.backward_onehot(grad)
        assert sparse.gradient_snapshot() == dense.gradient_snapshot()

    def test_forward_row_differs_from_column_convention(self):
        """行约定取 ``W[index]``、列约定取 ``W[:, index]``：同一行/列下标结果不同.

        这条用例守的是本课踩过的真实 bug——方阵上两种约定都能跑通、loss 也
        正常下降，但"基座 + 增量"的合并结果对不上。
        """
        square = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        layer = LoRALinear(square, small_config(r=2, lora_alpha=4, lora_dropout=0.0), seed=5)
        row = layer.forward_context(1)
        column = layer.forward_onehot(1)
        assert row == pytest.approx(square[1])
        assert column == pytest.approx([square[i][1] for i in range(3)])
        assert row != pytest.approx(column)

    def test_forward_context_matches_merged_by_context(self):
        """``forward_context(i)`` 必须等于按上下文合并后的第 ``i`` 行.

        **不能用 ``merged_weight()``（密集约定）来核对**：两条约定用的增量
        互为转置（``merged_by_context = W + ΔWᵀ``，``merged_weight = W + ΔW``），
        在非对称基座上逐行比较必然不等——这正是本课踩过的那个坑。
        """
        square = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        layer = LoRALinear(square, small_config(r=2, lora_alpha=4, lora_dropout=0.0), seed=5)
        for _ in range(2):
            layer.accumulate([1.0, 0.0, 0.0], [0.5, -0.5, 1.0])
        layer.apply_update(0.3)
        merged = layer.merged_by_context()
        dense = layer.merged_weight()
        for index in range(3):
            assert layer.forward_context(index) == pytest.approx(merged[index])
        # 两条约定用的增量互为转置：merged_by_context = W + ΔWᵀ，merged_weight = W + ΔW
        observed = [
            [square[i][j] + layer.delta_for_context(i)[j] for j in range(3)] for i in range(3)
        ]
        delta_transposed = [
            [dense[j][i] - square[j][i] for j in range(3)] for i in range(3)
        ]
        for i in range(3):
            for j in range(3):
                assert observed[i][j] == pytest.approx(merged[i][j])
                assert merged[i][j] == pytest.approx(square[i][j] + delta_transposed[i][j])

    def test_forward_context_requires_square(self):
        with pytest.raises(PEFTConfigError, match="只适用于方阵"):
            make_layer().forward_context(0)

    def test_merged_by_context_requires_square(self):
        with pytest.raises(PEFTConfigError, match="只适用于方阵"):
            make_layer().merged_by_context()

    def test_forward_context_rejects_out_of_range(self):
        square = [[1.0, 0.0], [0.0, 1.0]]
        layer = LoRALinear(square, small_config(r=1, lora_alpha=1, lora_dropout=0.0), seed=1)
        with pytest.raises(PEFTConfigError, match="越界"):
            layer.forward_context(2)

    def test_row_and_column_backward_share_gradients(self):
        """两种约定的反向完全一样：``A`` 只更新一列，``B`` 更新一行."""
        square = [[0.2, -0.1, 0.4], [0.3, 0.5, -0.2], [-0.4, 0.1, 0.25]]
        row_layer = LoRALinear(square, small_config(r=2, lora_alpha=4, lora_dropout=0.0), seed=9)
        col_layer = LoRALinear(square, small_config(r=2, lora_alpha=4, lora_dropout=0.0), seed=9)
        grad = [1.0, -0.5, 0.25]
        row_layer.forward_context(2)
        row_layer.backward_context(grad)
        col_layer.forward_onehot(2)
        col_layer.backward_onehot(grad)
        assert row_layer.gradient_snapshot() == col_layer.gradient_snapshot()

    def test_dropout_is_inactive_in_eval_mode(self):
        config = small_config(lora_dropout=0.5)
        layer = make_layer(config=config)
        assert layer.forward([1.0, 0.0, 0.0, 0.0]) == layer.forward([1.0, 0.0, 0.0, 0.0])

    def test_dropout_rescales_lora_branch_only(self):
        """dropout 只作用在适配器分支：基座那部分输出永远不变.

        ``x = [1, 0]`` 时基座输出就是权重矩阵的第 0 列；而适配器分支要么
        被整体置零（回到基座输出），要么被放大 ``1/(1-p) = 2`` 倍。
        两种结果都必须出现，且**不允许出现第三种**。
        """
        square = [[0.1, 0.2], [0.3, 0.4]]
        layer = LoRALinear(
            square, small_config(r=1, lora_alpha=1, lora_dropout=0.5), seed=2
        )
        layer.load_adapter_state(
            {
                "a": [[1.0, 1.0]],
                "b": [[1.0], [1.0]],
                "updates": 0,
                "scaling": layer.scaling,
            }
        )
        base_only = [square[index][0] for index in range(2)]
        seen = set()
        for _ in range(40):
            out = layer.forward([1.0, 0.0], training=True)
            assert out == pytest.approx(base_only) or len(out) == 2
            seen.add("dropped" if out == pytest.approx(base_only) else "kept")
        assert seen == {"dropped", "kept"}

    def test_one_hot_dropout_gates_whole_branch(self):
        """one-hot 输入下 dropout 等价于"以概率 p 整体丢弃适配器分支".

        训练模式下 mask 的取值只有两种：``0``（丢弃整条适配器分支，输出
        回到基座）或 ``1/(1-p) = 2``（适配器增量被放大 2 倍）。**不允许
        出现第三种结果**——出现第三种就说明 dropout 被施加到了基座路径上。
        """
        square = [[0.5, -0.25], [0.75, 0.125]]
        layer = LoRALinear(
            square, small_config(r=1, lora_alpha=1, lora_dropout=0.5), seed=4
        )
        layer.load_adapter_state(
            {
                "a": [[1.0, 1.0]],
                "b": [[1.0], [1.0]],
                "updates": 0,
                "scaling": layer.scaling,
            }
        )
        base_row = square[0]
        eval_out = layer.forward_context(0)
        delta = [value - base for value, base in zip(eval_out, base_row)]
        kept = [base + 2 * d for base, d in zip(base_row, delta)]
        seen = set()
        for _ in range(40):
            out = layer.forward_context(0, training=True)
            if out == pytest.approx(kept):
                seen.add("kept")
            elif out == pytest.approx(base_row):
                seen.add("dropped")
            else:
                seen.add("other")
        assert seen == {"kept", "dropped"}

    def test_adapter_state_roundtrip(self):
        layer = make_layer()
        for _ in range(2):
            layer.accumulate([1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        layer.apply_update(0.2)
        state = layer.adapter_state()
        restored = make_layer()
        restored.load_adapter_state(state)
        assert restored.updates == 1
        assert restored.snapshot_matrices() == layer.snapshot_matrices()
        assert restored.pending_positions == 0

    def test_load_adapter_state_requires_matrices(self):
        with pytest.raises(PEFTConfigError, match="缺少 a / b 矩阵"):
            make_layer().load_adapter_state({"updates": 0})

    def test_load_adapter_state_rejects_wrong_a_shape(self):
        with pytest.raises(PEFTConfigError, match="A 的形状"):
            make_layer().load_adapter_state({"a": [[1.0] * 4], "b": [[0.0], [0.0], [0.0]]})

    def test_load_adapter_state_rejects_wrong_b_shape(self):
        with pytest.raises(PEFTConfigError, match="B 的形状"):
            make_layer().load_adapter_state({"a": [[1.0] * 4, [1.0] * 4], "b": [[0.0]]})

    def test_load_adapter_state_rejects_scaling_drift(self):
        layer = make_layer()
        state = layer.adapter_state()
        state["scaling"] = layer.scaling * 2
        with pytest.raises(PEFTConfigError, match="scaling"):
            layer.load_adapter_state(state)

    def test_describe_reports_rank_and_norms(self):
        layer = make_layer()
        for _ in range(2):
            layer.accumulate([1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        layer.apply_update(1.0)
        info = layer.describe()
        assert info["rank_upper_bound"] == 2
        assert info["parameters"] == 14
        assert info["max_abs"] > 0.0
        assert info["scaling"] == pytest.approx(layer.scaling)

    def test_zero_tolerance_constant(self):
        assert ZERO_TOLERANCE == 1e-12

    def test_training_actually_moves_delta(self):
        layer = make_layer()
        before = layer.snapshot_matrices()
        rng = random.Random(0)
        for _ in range(5):
            x = [rng.uniform(-1, 1) for _ in range(4)]
            layer.accumulate(x, [1.0, 0.0, -1.0])
        layer.apply_update(0.5)
        assert layer.snapshot_matrices() != before
        assert layer.delta_is_zero() is False
        assert max_abs(layer.delta()) > 0.0

    def test_describe_delta_reports_shapes(self):
        a_matrix, b_matrix = init_lora_weights(4, 3, 2, seed=1, init_mode="random")
        info = describe_delta(a_matrix, b_matrix, 2.0)
        assert info["out_features"] == 3
        assert info["in_features"] == 4
        assert info["parameters"] == 14
        assert info["rank_upper_bound"] == 2


class TestDenseBackwardAndEdgeCases:
    """稠密反向的 ``∂L/∂x``、以及三条"没走前向就反向"的错误路径.

    稠密反向除了要更新 ``A`` / ``B``，还要把 ``∂L/∂x`` 交出来（供前一层使用）。
    它由两部分组成：基座路径 ``Wᵀ·g`` 与适配器路径 ``scaling·Aᵀ·Bᵀ·g``。
    **两部分都必须在**：少了基座那一项，串联使用的层会收到错误的梯度，
    而单看 LoRA 自己的参数更新完全正常——又是一个"loss 看起来没问题"的缺陷。
    """

    def test_backward_returns_full_gradient_for_input(self):
        config = small_config()
        layer = make_layer(config=config)
        a_matrix, b_matrix = layer.snapshot_matrices()
        x = [0.4, -0.9, 0.7, 0.2]
        grad_out = [0.8, -1.2, 0.35]
        grad_input = layer.accumulate(x, grad_out)

        scaling = layer.scaling
        # 适配器路径：scaling · Aᵀ·Bᵀ·g
        adapter = [0.0] * 4
        for index in range(4):
            total = 0.0
            for j in range(2):
                for i in range(3):
                    total += scaling * a_matrix[j][index] * b_matrix[i][j] * grad_out[i]
            adapter[index] = total
        # 基座路径：Wᵀ·g
        base = [sum(BASE[i][index] * grad_out[i] for i in range(3)) for index in range(4)]
        for index in range(4):
            assert grad_input[index] == pytest.approx(base[index] + adapter[index])

    def test_forward_onehot_rejects_out_of_range(self):
        with pytest.raises(PEFTConfigError, match="one-hot 下标"):
            make_layer().forward_onehot(4)

    def test_backward_onehot_requires_forward(self):
        """没走前向就反向：先被缓存检查拦下（取不到行/列下标）."""
        with pytest.raises(PEFTConfigError, match="缺少前向缓存"):
            make_layer().backward_onehot([1.0, 0.0, 0.0])

    def test_backward_context_requires_forward(self):
        with pytest.raises(PEFTConfigError, match="缺少前向缓存"):
            make_layer().backward_context([1.0, 0.0, 0.0])

    def test_backward_context_rejects_wrong_gradient_dimension(self):
        """定点反向的维度校验：``grad_out`` 长度必须等于 ``out_features``."""
        layer = LoRALinear(
            [[0.1, 0.2], [0.3, 0.4]], small_config(r=1, lora_alpha=1, lora_dropout=0.0), seed=1
        )
        layer.forward_context(0)
        with pytest.raises(PEFTConfigError, match="输出梯度维度不匹配"):
            layer.backward_context([1.0, 0.0, 0.0])

    def test_cached_index_requires_forward(self):
        with pytest.raises(PEFTConfigError, match="缺少前向缓存"):
            make_layer()._cached_index()

    def test_config_property(self):
        config = small_config(lora_alpha=8)
        assert make_layer(config=config).config is config
