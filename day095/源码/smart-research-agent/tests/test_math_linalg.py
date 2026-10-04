"""``math_foundations.linalg``：十个算子（day073）.

这一课的**手算性质**在本文件里体现得最充分：几乎每一条断言都能在一张纸上推出来。

```text
cos(e1, e1) = 1          cos(e1, e2) = 0        cos(a, 2a) = 1
‖(3,4)‖ = 5              归一化后模长恒为 1       投影到 e1 上 = 第一个分量
对角矩阵的奇异值 = 对角线  秩 k 近似对秩 ≤ k 的矩阵误差为 0
softmax([0,0]) = [0.5,0.5] softmax 在 [1000,1001] 上仍然稳定
```
"""

from __future__ import annotations

import math

import pytest

from smart_research_agent.math_foundations.errors import (
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.math_foundations.linalg import (
    add,
    argmax,
    cosine,
    dot,
    frobenius_norm,
    identity,
    log_softmax,
    low_rank_approximation,
    matmul,
    matrix_add,
    matrix_scale,
    matvec,
    norm,
    normalize,
    outer,
    power_iteration,
    projection,
    reconstruction_error,
    row_normalize,
    scale,
    softmax,
    subtract,
    top_k_indices,
    transpose,
    zeros,
)
from tests.math_samples import (
    A3,
    A3_ORTHOGONAL,
    A3_SCALED,
    DIAGONAL_2X2,
    DIAGONAL_3X3,
    E1,
    E2,
    E3,
    RECT_2X3,
    SQRT_3,
    approx,
    approx_vector,
)


class TestVectorOperations:
    """点积、模长、归一化、余弦、投影——都能手算."""

    def test_dot_of_basis_vectors(self) -> None:
        assert dot(E1, E1) == 1.0
        assert dot(E1, E2) == 0.0
        assert dot((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)) == 32.0

    def test_dot_requires_same_dimension(self) -> None:
        with pytest.raises(ShapeError):
            dot((1.0, 2.0), (1.0, 2.0, 3.0))

    def test_norm_matches_pythagoras(self) -> None:
        assert norm((3.0, 4.0)) == 5.0
        assert approx(norm(A3), math.sqrt(14.0))
        assert norm((0.0,)) == 0.0

    def test_normalize_gives_unit_length(self) -> None:
        unit = normalize(A3)
        assert approx(norm(unit), 1.0)
        assert approx(unit[0], 1.0 / math.sqrt(14.0))

    def test_normalize_rejects_zero_vector(self) -> None:
        with pytest.raises(NumericError) as excinfo:
            normalize((0.0, 0.0))
        assert "零向量" in str(excinfo.value)

    def test_scale_add_subtract(self) -> None:
        assert scale((1.0, 2.0), 3.0) == (3.0, 6.0)
        assert add((1.0, 2.0), (3.0, 4.0)) == (4.0, 6.0)
        assert subtract((1.0, 2.0), (3.0, 4.0)) == (-2.0, -2.0)
        with pytest.raises(ShapeError):
            add((1.0,), (1.0, 2.0))

    def test_scale_rejects_non_finite_factor(self) -> None:
        with pytest.raises(NumericError):
            scale((1.0,), float("inf"))

    def test_cosine_hand_computed(self) -> None:
        assert cosine(E1, E1) == 1.0
        assert cosine(E1, E2) == 0.0
        assert approx(cosine(A3, A3_SCALED), 1.0)  # 同向：与长度无关
        assert approx(cosine(A3, A3_ORTHOGONAL), 0.0)  # 正交
        assert approx(cosine((1.0, 0.0), (-1.0, 0.0)), -1.0)  # 相反

    def test_cosine_of_zero_vector_is_zero(self) -> None:
        assert cosine((0.0, 0.0), (1.0, 2.0)) == 0.0
        assert cosine((1.0, 2.0), (0.0, 0.0)) == 0.0

    def test_projection_is_the_coordinate_along_the_direction(self) -> None:
        assert approx(projection((3.0, 4.0), (1.0, 0.0)), 3.0)
        assert approx(projection((3.0, 4.0), (0.0, 5.0)), 4.0)
        # 投影到"45 度"方向上：(3,4)·(1,1)/√2 = 7/√2
        assert approx(projection((3.0, 4.0), (1.0, 1.0)), 7.0 / math.sqrt(2.0))
        with pytest.raises(NumericError):
            projection((1.0, 2.0), (0.0, 0.0))


class TestMatrixOperations:
    """矩阵乘法、转置、外积、单位矩阵——形状错误一律当场拒绝."""

    def test_transpose_swaps_shape(self) -> None:
        assert transpose(RECT_2X3) == ((1.0, 4.0), (2.0, 5.0), (3.0, 6.0))

    def test_matmul_hand_computed(self) -> None:
        left = ((1.0, 2.0), (3.0, 4.0))
        right = ((5.0, 6.0), (7.0, 8.0))
        assert matmul(left, right) == ((19.0, 22.0), (43.0, 50.0))

    def test_matmul_identity_is_neutral(self) -> None:
        assert matmul(DIAGONAL_2X2, identity(2)) == DIAGONAL_2X2

    def test_matmul_inner_dimension_must_match(self) -> None:
        with pytest.raises(ShapeError):
            matmul(RECT_2X3, RECT_2X3)

    def test_matvec(self) -> None:
        assert matvec(RECT_2X3, (1.0, 0.0, 0.0)) == (1.0, 4.0)
        with pytest.raises(ShapeError):
            matvec(RECT_2X3, (1.0, 0.0))

    def test_outer_product(self) -> None:
        assert outer((1.0, 2.0), (3.0, 4.0)) == ((3.0, 4.0), (6.0, 8.0))

    def test_matrix_add_and_scale(self) -> None:
        assert matrix_add(DIAGONAL_2X2, DIAGONAL_2X2) == ((6.0, 0.0), (0.0, 4.0))
        assert matrix_scale(DIAGONAL_2X2, 0.5) == ((1.5, 0.0), (0.0, 1.0))
        with pytest.raises(ShapeError):
            matrix_add(DIAGONAL_2X2, RECT_2X3)

    def test_identity_and_zeros(self) -> None:
        assert identity(2) == ((1.0, 0.0), (0.0, 1.0))
        assert zeros(2, 3) == ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        with pytest.raises(ParameterError):
            identity(0)
        with pytest.raises(ParameterError):
            zeros(0, 1)

    def test_frobenius_norm(self) -> None:
        assert frobenius_norm(((3.0, 0.0), (0.0, 4.0))) == 5.0


class TestSoftmax:
    """softmax：稳定、和为 1、温度的作用方向正确、极端输入不炸."""

    def test_uniform_logits_give_uniform_probabilities(self) -> None:
        assert approx_vector(softmax((0.0, 0.0)), (0.5, 0.5))
        assert approx_vector(softmax((7.0, 7.0, 7.0)), (1 / 3, 1 / 3, 1 / 3))

    def test_softmax_sums_to_one(self) -> None:
        probabilities = softmax((1.0, 2.0, 3.0))
        assert approx(sum(probabilities), 1.0)
        assert probabilities[2] > probabilities[1] > probabilities[0]

    def test_softmax_is_stable_on_huge_logits(self) -> None:
        probabilities = softmax((1000.0, 1001.0))
        assert all(math.isfinite(value) for value in probabilities)
        assert approx(sum(probabilities), 1.0)
        # 手算：(0, 1) 的 softmax = [1/(1+e), e/(1+e)]
        expected_low = 1.0 / (1.0 + math.e)
        assert approx(probabilities[0], expected_low)

    def test_low_temperature_concentrates(self) -> None:
        concentrated = softmax((1.0, 2.0), temperature=0.01)
        assert concentrated[1] > 0.99
        # 高温时趋于均匀：softmax([0.01, 0.02]) 的第一项 = 1/(1+e^{0.01})
        flattened = softmax((1.0, 2.0), temperature=100.0)
        assert approx(flattened[0], 1.0 / (1.0 + math.exp(0.01)))

    @pytest.mark.parametrize("temperature", [0.0, -1.0, float("inf")])
    def test_bad_temperature_is_rejected(self, temperature: float) -> None:
        with pytest.raises(ParameterError):
            softmax((1.0, 2.0), temperature=temperature)

    def test_log_softmax_matches_log_of_softmax(self) -> None:
        logits = (1.0, 2.0, 3.0)
        for expected, actual in zip(
            (math.log(value) for value in softmax(logits)), log_softmax(logits)
        ):
            assert approx(expected, actual)

    def test_naive_log_softmax_crashes_while_ours_stays_finite(self) -> None:
        # 朴素写法 log(softmax(x))：softmax 先下溢到 0（这一项的概率小到无法表示），
        # 而 math.log(0.0) 在 Python 里**直接抛 ValueError**（不是返回 -inf）。
        # 本包的 log_softmax 不经由 softmax，因此在同样的输入上仍然给出有限值。
        probabilities = softmax((0.0, -1000.0))
        assert probabilities[1] == 0.0
        with pytest.raises(ValueError):
            math.log(probabilities[1])
        assert approx(log_softmax((0.0, -1000.0))[1], -1000.0)

    def test_argmax_and_top_k(self) -> None:
        assert argmax((1.0, 3.0, 3.0)) == 1  # 并列取最小下标
        assert top_k_indices((1.0, 3.0, 3.0), 2) == (1, 2)
        with pytest.raises(ParameterError):
            top_k_indices((1.0, 2.0), 0)
        with pytest.raises(ParameterError):
            top_k_indices((1.0, 2.0), 3)

    def test_row_normalize(self) -> None:
        normalized = row_normalize(((1.0, 3.0), (2.0, 2.0)))
        assert approx_vector(normalized[0], (0.25, 0.75))
        assert approx_vector(normalized[1], (0.5, 0.5))

    def test_row_normalize_rejects_zero_row(self) -> None:
        with pytest.raises(NumericError):
            row_normalize(((1.0, 1.0), (0.0, 0.0)))


class TestPowerIteration:
    """幂迭代：对角矩阵的奇异值就是对角线（可手算验证）."""

    def test_diagonal_matrix_gives_the_largest_singular_value(self) -> None:
        sigma, u, v = power_iteration(DIAGONAL_2X2)
        assert approx(sigma, 3.0)
        # 方向判据是**二次**的：余弦差 ~1e-12 对应分量误差 ~1e-6
        # （σ₂/σ₁ = 2/3，每轮把次方向的比重压到 0.444 倍，因此 20 轮就到 1e-7）
        assert approx_vector(u, (1.0, 0.0), tolerance=1e-5)
        assert approx_vector(v, (1.0, 0.0), tolerance=1e-5)

    def test_three_by_three_diagonal(self) -> None:
        sigma, _, _ = power_iteration(DIAGONAL_3X3)
        assert approx(sigma, 2.0)

    def test_singular_value_of_a_rank_one_matrix(self) -> None:
        matrix = outer((3.0, 4.0), (0.6, 0.8))
        sigma, u, v = power_iteration(matrix)
        assert approx(sigma, 5.0)
        assert approx_vector(u, normalize((3.0, 4.0)))
        assert approx_vector(v, (0.6, 0.8))

    def test_parameters_are_validated(self) -> None:
        with pytest.raises(ParameterError):
            power_iteration(DIAGONAL_2X2, max_iterations=0)
        with pytest.raises(ParameterError):
            power_iteration(DIAGONAL_2X2, tolerance=0.0)

    def test_zero_matrix_is_rejected(self) -> None:
        with pytest.raises(NumericError):
            power_iteration(zeros(2, 2))

    def test_convergence_is_reported_by_rerunning(self) -> None:
        # "收敛了吗"这个问题靠**换一个容差再跑一遍**回答：两次结果一致才算收敛了。
        # 注意 σ 的精度也受同一个收敛判据限制（方向差 → σ 的二次误差），
        # 因此这里比的是"两次一致到 1e-6"，而不是"精确相等"。
        loose = power_iteration(DIAGONAL_3X3, tolerance=1e-6)[0]
        tight = power_iteration(DIAGONAL_3X3, tolerance=1e-14)[0]
        assert approx(loose, tight, tolerance=1e-6)


class TestLowRankApproximation:
    """低秩近似：秩 k 对秩 ≤ k 的矩阵误差为 0（这条性质能精确断言）."""

    def test_diagonal_matrix_needs_two_factors(self) -> None:
        values, _, _, approximation = low_rank_approximation(DIAGONAL_3X3, 2)
        assert approx_vector(values, (2.0, 1.0))
        # A_2 对 A（奇异值 2、1、0.5）不是精确的：丢掉了 0.5 那一层
        error = reconstruction_error(DIAGONAL_3X3, approximation)
        assert approx(error, 0.5 / math.sqrt(2**2 + 1.0 + 0.5**2))

    def test_full_rank_reconstruction_is_exact(self) -> None:
        _, _, _, approximation = low_rank_approximation(DIAGONAL_3X3, 3)
        assert approx(reconstruction_error(DIAGONAL_3X3, approximation), 0.0, tolerance=1e-9)

    def test_rank_one_matrix_is_reconstructed_by_rank_one(self) -> None:
        matrix = outer((1.0, 2.0, 3.0), (0.0, 1.0))
        _, _, _, approximation = low_rank_approximation(matrix, 1)
        assert approx(reconstruction_error(matrix, approximation), 0.0, tolerance=1e-9)

    def test_rank_is_validated(self) -> None:
        with pytest.raises(ParameterError):
            low_rank_approximation(DIAGONAL_2X2, 0)
        with pytest.raises(ParameterError):
            low_rank_approximation(DIAGONAL_2X2, 3)

    def test_reconstruction_error_requires_same_shape(self) -> None:
        with pytest.raises(ShapeError):
            reconstruction_error(DIAGONAL_2X2, RECT_2X3)

    def test_reconstruction_error_rejects_zero_matrix(self) -> None:
        with pytest.raises(NumericError):
            reconstruction_error(zeros(2, 2), zeros(2, 2))


class TestDeterminism:
    """同一份输入永远同一份输出（这一层是可复现实验的地基）."""

    def test_repeated_calls_are_bitwise_identical(self) -> None:
        assert softmax((0.1, 0.2, 0.3)) == softmax((0.1, 0.2, 0.3))
        assert low_rank_approximation(DIAGONAL_3X3, 2)[3] == low_rank_approximation(
            DIAGONAL_3X3, 2
        )[3]

    def test_epsilon_sum_matches_expected(self) -> None:
        assert approx(norm(E3), 1.0)
        assert approx(norm(scale(E1, SQRT_3)), SQRT_3)
