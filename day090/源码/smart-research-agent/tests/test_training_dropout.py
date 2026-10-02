"""``training_optim.dropout``：一个掩码、两个相、一行反向（day081）."""

from __future__ import annotations

import pytest

from smart_research_agent.math_foundations.types import matrix_shape
from smart_research_agent.training_optim import (
    DEFAULT_DROPOUT_RATE,
    PHASES,
    PHASE_DESCRIPTIONS,
    PHASE_EVAL,
    PHASE_TRAIN,
    ParameterError,
    ShapeError,
    dropout_backward,
    dropout_forward,
    dropout_mask,
    expected_keep,
    kept_fraction,
    scale_of,
)
from smart_research_agent.training_optim.errors import NumericError
from tests.training_samples import matrix_shape_of

SQUARE = ((1.0, 2.0, 3.0, 4.0), (5.0, 6.0, 7.0, 8.0))


class TestConfigTables:
    """两个相与默认值（表与键必须闭合）."""

    def test_two_phases_with_descriptions(self):
        """两个相各有一条说明，且默认丢弃概率是 0（基线）。"""
        assert PHASES == (PHASE_TRAIN, PHASE_EVAL)
        assert set(PHASE_DESCRIPTIONS) == set(PHASES)
        assert DEFAULT_DROPOUT_RATE == 0.0

    def test_expected_keep_is_one_minus_rate(self):
        """期望保留比例 ``1 − p``（它是实测值的参照物）。"""
        assert expected_keep(0.25) == pytest.approx(0.75)
        assert expected_keep(0.0) == 1.0


class TestMask:
    """掩码：确定性、0/1、形状正确."""

    def test_mask_is_deterministic(self):
        """同一颗种子给出同一张掩码（**整个实验可复现的前提**）。"""
        first = dropout_mask(3, 4, rate=0.3, seed=11)
        second = dropout_mask(3, 4, rate=0.3, seed=11)
        assert first == second

    def test_different_seeds_give_different_masks(self):
        """换一颗种子会换一张掩码（否则那不是 dropout）。"""
        assert dropout_mask(6, 6, rate=0.3, seed=11) != dropout_mask(6, 6, rate=0.3, seed=12)

    def test_mask_is_binary(self):
        """掩码里只有 0 与 1（没有别的数）。"""
        mask = dropout_mask(4, 5, rate=0.4, seed=3)
        assert {value for row in mask for value in row} <= {0.0, 1.0}

    def test_rate_zero_keeps_everything(self):
        """``rate = 0`` 时掩码全是 1（于是它退化成恒等）。"""
        mask = dropout_mask(3, 3, rate=0.0, seed=5)
        assert all(value == 1.0 for row in mask for value in row)
        assert kept_fraction(mask) == 1.0

    def test_kept_fraction_is_in_the_unit_interval(self):
        """保留比例落在 ``[0, 1]``，而且它随概率下降（本样本上）。"""
        low = kept_fraction(dropout_mask(20, 20, rate=0.05, seed=9))
        high = kept_fraction(dropout_mask(20, 20, rate=0.5, seed=9))
        assert 0.0 <= high <= low <= 1.0

    def test_a_big_mask_approaches_one_minus_p(self):
        """``200 × 200`` 上实测保留比例已经贴近 ``1 − p``（小样本上会摆动）。"""
        measured = kept_fraction(dropout_mask(200, 200, rate=0.25, seed=13))
        assert abs(measured - expected_keep(0.25)) < 0.01

    @pytest.mark.parametrize("rate", [1.0, 1.5, -0.1, True, "0.2"])
    def test_rate_must_live_in_the_half_open_interval(self, rate: object):
        """``rate`` 必须落在 ``[0, 1)``——``1.0`` 会让 ``1/(1−p)`` 除零。"""
        with pytest.raises(ParameterError):
            dropout_mask(2, 2, rate=rate, seed=1)  # type: ignore[arg-type]

    @pytest.mark.parametrize("rows,columns", [(0, 2), (2, 0), (1.5, 2)])
    def test_shape_must_be_positive_integers(self, rows: object, columns: object):
        """掩码的形状必须是正整数的两个维度。"""
        with pytest.raises(ParameterError):
            dropout_mask(rows, columns, rate=0.2, seed=1)  # type: ignore[arg-type]

    def test_seed_must_be_an_integer(self):
        """种子必须是整数。"""
        with pytest.raises(ParameterError):
            dropout_mask(2, 2, rate=0.2, seed=1.5)  # type: ignore[arg-type]

    def test_kept_fraction_rejects_broken_masks(self):
        """形状不对的掩码会被 ``validate_matrix`` 挡住。"""
        with pytest.raises(ValueError):
            kept_fraction(((1.0, 1.0), (1.0,)))  # type: ignore[arg-type]


class TestForward:
    """前向：两个相、缩放、三个返回值."""

    def test_train_phase_drops_and_scales(self):
        """训练相：被丢的位置是 0，活下来的乘上 ``1/(1−p)``."""
        output, mask, scale = dropout_forward(SQUARE, rate=0.5, seed=4, phase=PHASE_TRAIN)
        assert scale == pytest.approx(2.0)
        for row_input, row_mask, row_output in zip(SQUARE, mask, output, strict=True):
            for value, flag, result in zip(row_input, row_mask, row_output, strict=True):
                expected = 0.0 if flag == 0.0 else value * 2.0
                assert result == pytest.approx(expected)

    def test_eval_phase_is_the_identity(self):
        """推理相与输入**逐位**相同（inverted dropout 的全部好处）。"""
        output, mask, scale = dropout_forward(SQUARE, rate=0.5, seed=4, phase=PHASE_EVAL)
        assert output == SQUARE
        assert scale == 1.0
        assert all(value == 1.0 for row in mask for value in row)

    def test_rate_zero_is_the_identity_in_both_phases(self):
        """``rate = 0`` 时两个相都退化成恒等（因此"关掉 dropout"没有第二条代码路径）。"""
        for phase in PHASES:
            output, _mask, scale = dropout_forward(SQUARE, rate=0.0, seed=4, phase=phase)
            assert output == SQUARE
            assert scale == 1.0

    def test_shape_is_preserved(self):
        """三个返回值的形状与输入一致（掩码是逐元素的）。"""
        output, mask, _scale = dropout_forward(SQUARE, rate=0.3, seed=2)
        assert matrix_shape_of(output) == matrix_shape_of(mask) == matrix_shape(SQUARE)

    def test_scaling_keeps_the_expectation(self):
        """``E[掩码 × 缩放] = 1``：在**一张大掩码**上取平均.

        一个学到的边界：``uniforms`` 的 LCG 在**相邻的小种子**上前几个值彼此接近
        （第一个值是 ``(M·s + I) mod MOD``，随 s 近似线性增长）。
        因此"取 40 颗小种子求平均"会得到一个**有偏**的答案，而它有偏的方向是固定的
        ——第一版就是这么写的，它稳定地给出 0.0。

        正确做法是把**一张大掩码**当成一个样本：``60 × 40 = 2400`` 个元素的均值
        已经足够贴近 1（``±0.05`` 以内）。
        """
        mask = dropout_mask(60, 40, rate=0.3, seed=7)
        scale = 1.0 / (1.0 - 0.3)
        total = sum(value * scale for row in mask for value in row)
        assert total / (60 * 40) == pytest.approx(1.0, abs=0.05)

    def test_adjacent_seeds_are_correlated_at_the_first_element(self):
        """把上面那条边界**写成一条断言**：相邻种子的第一个元素彼此接近.

        这不是 bug，而是 LCG 的性质；它值得被写下来，
        因为"换一颗种子重跑一遍"是实验里最常见的动作——
        而在这个 LCG 上，"小种子的前几个值"几乎不随种子变化。
        """
        firsts = [dropout_mask(1, 1, rate=0.5, seed=seed)[0][0] for seed in range(6)]
        assert len(set(firsts)) <= 2

    @pytest.mark.parametrize("phase", ["trainn", "", None, "TRAIN"])
    def test_unknown_phase_is_rejected(self, phase: object):
        """不认识的相当场报错（**不给它挑一个默认值**）。"""
        with pytest.raises(ParameterError):
            dropout_forward(SQUARE, rate=0.2, seed=1, phase=phase)  # type: ignore[arg-type]

    def test_scale_of_matches_the_forward(self):
        """``scale_of`` 与 ``dropout_forward`` 给出的缩放一致（口径只有一处）。"""
        _output, _mask, scale = dropout_forward(SQUARE, rate=0.25, seed=8)
        assert scale_of(0.25) == pytest.approx(scale)
        assert scale_of(0.25, phase=PHASE_EVAL) == 1.0


class TestBackward:
    """反向：一行、可校验."""

    def test_backward_is_elementwise(self):
        """``dx = dy ⊙ 掩码 × 缩放``（逐元素，三个因子一一对应）。"""
        mask = ((1.0, 0.0), (0.0, 1.0))
        grad = ((2.0, 4.0), (6.0, 8.0))
        result = dropout_backward(grad, mask, 1.5)
        assert result == ((3.0, 0.0), (0.0, 12.0))

    def test_backward_with_ones_is_the_identity(self):
        """全 1 掩码 + 缩放 1 ⇒ 逐位恒等（这就是 ``rate = 0`` 时那条路径）。"""
        ones = tuple(tuple(1.0 for _ in row) for row in SQUARE)
        assert dropout_backward(SQUARE, ones, 1.0) == SQUARE

    def test_backward_matches_a_numerical_difference(self):
        """**核心断言**：掩码固定之后，这个算子的梯度与中心差分逐项对上.

        ``f(x) = Σ (x ⊙ m · s)`` 是一个线性函数，因此它的梯度就是 ``m · s``——
        数值差分给出的答案必须与 :func:`dropout_backward` 逐位一致到 ``1e-9``。
        """
        mask = dropout_mask(1, 4, rate=0.5, seed=17)
        scale = 1.0 / (1.0 - 0.5)
        point = (0.3, -0.7, 1.1, -0.2)
        step = 1e-6
        numeric: list[float] = []
        for index in range(4):
            bumped = list(point)
            bumped[index] += step
            forward_plus = sum(
                value * mask[0][column] * scale for column, value in enumerate(bumped)
            )
            bumped[index] -= 2 * step
            forward_minus = sum(
                value * mask[0][column] * scale for column, value in enumerate(bumped)
            )
            numeric.append((forward_plus - forward_minus) / (2 * step))
        analytic = dropout_backward((tuple(1.0 for _ in point),), mask, scale)[0]
        for index in range(4):
            assert analytic[index] == pytest.approx(numeric[index], abs=1e-9)

    def test_shape_mismatch_is_rejected(self):
        """梯度与掩码不同形时当场报错（掩码是逐元素的）。"""
        with pytest.raises(ShapeError):
            dropout_backward(SQUARE, ((1.0, 1.0),), 1.0)

    @pytest.mark.parametrize("scale", [float("inf"), float("nan"), "1.0"])
    def test_scale_must_be_a_finite_number(self, scale: object):
        """缩放必须是有限数。"""
        with pytest.raises(NumericError):
            dropout_backward(SQUARE, SQUARE, scale)  # type: ignore[arg-type]
