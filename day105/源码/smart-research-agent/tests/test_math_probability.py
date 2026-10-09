"""``math_foundations.probability``：分布、联合表与随机性（day073）.

手算性质（每一条都能在一张纸上验证）：

```text
H(均匀 n) = ln n          H(独热) = 0          H(0.5, 0.5) = ln 2
H(p, p) = H(p)            KL(p, p) = 0         KL(p‖q) ≠ KL(q‖p)
E[Bernoulli(0.5)] = 0.5   Var = 0.25           困惑度(均匀 4) = 4
P(A|B) = P(A,B)/P(B)      独立时互信息 = 0      贝叶斯后验 = 列条件分布
```
"""

from __future__ import annotations

import math

import pytest

from smart_research_agent.math_foundations.errors import (
    NumericError,
    ParameterError,
    TableError,
)
from smart_research_agent.math_foundations.probability import (
    LCG_MODULUS,
    Distribution,
    JointTable,
    cross_entropy,
    entropy,
    expectation,
    kl_divergence,
    max_entropy,
    perplexity_from_entropy,
    sample_index,
    uniforms,
    variance,
)
from tests.math_samples import (
    LN_3,
    approx,
    approx_vector,
    distribution,
    independent_joint,
    joint,
    one_hot,
    uniform,
)


class TestEntropy:
    """熵：均匀最大、独热为 0、p=0 的那一项贡献 0（不是 nan）."""

    def test_uniform_reaches_the_ceiling(self) -> None:
        assert approx(entropy(uniform(3)), LN_3)
        assert approx(entropy(uniform(4)), math.log(4.0))
        assert approx(max_entropy(3), LN_3)

    def test_one_hot_has_zero_entropy(self) -> None:
        assert entropy(one_hot(1, 3)) == 0.0

    def test_half_half_is_ln_two(self) -> None:
        assert approx(entropy((0.5, 0.5)), math.log(2.0))

    def test_zero_probability_contributes_zero_not_nan(self) -> None:
        # 朴素写法 -p*log(p) 在 p=0 时得到 nan；本课的口径是"跳过 p=0 的项"
        value = entropy((1.0, 0.0, 0.0))
        assert math.isfinite(value) and value == 0.0

    def test_max_entropy_validates_size(self) -> None:
        with pytest.raises(ParameterError):
            max_entropy(0)


class TestCrossEntropyAndKL:
    """交叉熵与 KL：非对称、≥ 熵、在 p == q 处为 0（KL）."""

    def test_cross_entropy_with_itself_equals_entropy(self) -> None:
        probabilities = (0.2, 0.3, 0.5)
        assert approx(cross_entropy(probabilities, probabilities), entropy(probabilities))

    def test_cross_entropy_is_not_symmetric(self) -> None:
        p = (0.9, 0.1)
        q = (0.5, 0.5)
        assert not approx(cross_entropy(p, q), cross_entropy(q, p))

    def test_cross_entropy_is_never_below_entropy(self) -> None:
        p = (0.6, 0.4)
        q = (0.3, 0.7)
        assert cross_entropy(p, q) >= entropy(p) - 1e-12

    def test_hand_computed_cross_entropy(self) -> None:
        # 独热真值 → 交叉熵退化成 -ln(预测概率)
        assert approx(cross_entropy((1.0, 0.0), (0.8, 0.2)), -math.log(0.8))

    def test_zero_probability_is_rejected_instead_of_inf(self) -> None:
        # 真值给 (0.5, 0.5)、预测给 (1.0, 0.0)：第二个事件"真会发生"而预测说不可能
        with pytest.raises(NumericError) as excinfo:
            cross_entropy((0.5, 0.5), (1.0, 0.0))
        assert "+inf" in str(excinfo.value)

    def test_zero_probability_in_the_truth_is_fine(self) -> None:
        # 反过来是可以的：真值里概率为 0 的事件不贡献任何代价
        assert approx(cross_entropy((1.0, 0.0), (0.9, 0.1)), -math.log(0.9))

    def test_kl_is_zero_for_identical_distributions(self) -> None:
        assert approx(kl_divergence((0.2, 0.8), (0.2, 0.8)), 0.0)

    def test_kl_is_asymmetric_and_matches_definition(self) -> None:
        p = (0.75, 0.25)
        q = (0.5, 0.5)
        assert approx(kl_divergence(p, q), cross_entropy(p, q) - entropy(p))
        assert kl_divergence(p, q) > 0
        assert not approx(kl_divergence(p, q), kl_divergence(q, p))

    def test_length_mismatch_is_rejected(self) -> None:
        with pytest.raises(NumericError):
            cross_entropy((0.5, 0.5), (1.0,))


class TestExpectationVariance:
    """期望与方差：手算的伯努利、以及"取值与概率个数必须一致"."""

    def test_bernoulli(self) -> None:
        probabilities = (0.5, 0.5)
        assert approx(expectation((0.0, 1.0), probabilities), 0.5)
        assert approx(variance((0.0, 1.0), probabilities), 0.25)

    def test_variance_of_a_constant_distribution_is_zero(self) -> None:
        assert approx(variance((7.0, 7.0), (0.5, 0.5)), 0.0)

    def test_biased_coin_hand_computed(self) -> None:
        probabilities = (0.25, 0.75)
        assert approx(expectation((1.0, 2.0), probabilities), 1.75)
        # Var = 0.25·(1−1.75)² + 0.75·(2−1.75)² = 0.140625 + 0.046875
        assert approx(variance((1.0, 2.0), probabilities), 0.1875)

    def test_length_mismatch_is_rejected(self) -> None:
        with pytest.raises(NumericError):
            expectation((1.0, 2.0, 3.0), (0.5, 0.5))


class TestSampling:
    """采样：累积区间、边界约定、可复现的随机数."""

    def test_cumulative_boundaries(self) -> None:
        probabilities = (0.5, 0.3, 0.2)
        assert sample_index(probabilities, u=0.0) == 0
        assert sample_index(probabilities, u=0.49) == 0
        assert sample_index(probabilities, u=0.5) == 1  # 边界取后一段
        assert sample_index(probabilities, u=0.79) == 1
        assert sample_index(probabilities, u=0.99) == 2

    @pytest.mark.parametrize("u", [-0.1, 1.0, 1.5, float("nan")])
    def test_bad_u_is_rejected(self, u: float) -> None:
        with pytest.raises(ParameterError):
            sample_index((0.5, 0.5), u=u)

    def test_uniforms_are_deterministic_and_in_range(self) -> None:
        first = uniforms(5, seed=7)
        second = uniforms(5, seed=7)
        assert first == second
        assert all(0.0 <= value < 1.0 for value in first)
        assert uniforms(0) == ()
        # 换成另一个种子就是另一串数（同一个种子永远同一串）
        assert uniforms(5, seed=8) != first
        # 前缀性质：同一个种子的前 3 个数就是那 5 个数里的前 3 个
        assert uniforms(3, seed=7) == first[:3]

    def test_uniforms_validate_count(self) -> None:
        with pytest.raises(ParameterError):
            uniforms(-1)

    def test_sampling_matches_the_distribution_roughly(self) -> None:
        # 用确定性随机数采 2000 次：频率应当接近概率（±0.08 的宽松界）
        probabilities = (0.7, 0.2, 0.1)
        draws = [sample_index(probabilities, u=value) for value in uniforms(2000, seed=1)]
        for index, expected in enumerate(probabilities):
            frequency = draws.count(index) / len(draws)
            assert abs(frequency - expected) < 0.08, index
        assert LCG_MODULUS == 2**32


class TestDistributionShape:
    """分布对象：构造路径、标签、派生量."""

    def test_uniform_construction(self) -> None:
        dist = Distribution.uniform(4)
        assert dist.size == 4
        assert dist.labels == ("0", "1", "2", "3")
        assert approx(dist.entropy(), math.log(4.0))
        assert approx(dist.perplexity(), 4.0)

    def test_from_logits_matches_softmax(self) -> None:
        dist = Distribution.from_logits((0.0, 0.0), name="coin")
        assert approx_vector(dist.probabilities, (0.5, 0.5))
        assert "coin" in dist.summary_line()

    def test_from_counts_normalizes(self) -> None:
        dist = Distribution.from_counts((2.0, 6.0), labels=("yes", "no"))
        assert approx_vector(dist.probabilities, (0.25, 0.75))
        assert dist.labels == ("yes", "no")

    def test_from_counts_rejects_all_zero(self) -> None:
        with pytest.raises(NumericError):
            Distribution.from_counts((0.0, 0.0))

    def test_from_counts_rejects_negative(self) -> None:
        with pytest.raises(NumericError):
            Distribution.from_counts((1.0, -1.0))

    def test_uniform_size_is_validated(self) -> None:
        with pytest.raises(ParameterError):
            Distribution.uniform(0)

    def test_top_k_is_sorted_and_deterministic(self) -> None:
        dist = Distribution(probabilities=(0.2, 0.3, 0.5), labels=("a", "b", "c"))
        assert dist.top_k(2) == (("c", 0.5), ("b", 0.3))
        with pytest.raises(ParameterError):
            dist.top_k(4)

    def test_expectation_variance_and_cross_entropy(self) -> None:
        dist = Distribution(probabilities=(0.25, 0.75))
        assert approx(dist.expectation((0.0, 1.0)), 0.75)
        assert approx(dist.variance((0.0, 1.0)), 0.1875)
        other = Distribution.uniform(2)
        assert approx(dist.cross_entropy(other), -0.25 * math.log(0.5) - 0.75 * math.log(0.5))
        assert dist.kl(other) > 0

    def test_sample_uses_the_cumulative_rule(self) -> None:
        dist = Distribution(probabilities=(0.5, 0.5), labels=("heads", "tails"))
        assert dist.sample(u=0.1) == "heads"
        assert dist.sample(u=0.9) == "tails"

    def test_label_count_must_match(self) -> None:
        with pytest.raises(NumericError):
            Distribution(probabilities=(0.5, 0.5), labels=("only_one",))

    def test_dict_is_json_friendly(self) -> None:
        import json

        payload = distribution().to_dict()
        json.dumps(payload)
        assert payload["size"] == 3
        assert payload["top3"][0][1] >= payload["top3"][1][1]


class TestJointTable:
    """联合分布表：边缘、条件、贝叶斯后验、互信息."""

    def test_marginals_are_hand_computed(self) -> None:
        table = joint()
        assert approx_vector(table.marginal_a().probabilities, (0.4, 0.6))
        assert approx_vector(table.marginal_b().probabilities, (0.5, 0.5))

    def test_marginals_sum_to_one(self) -> None:
        table = joint()
        assert approx(sum(table.marginal_a().probabilities), 1.0)
        assert approx(sum(table.marginal_b().probabilities), 1.0)

    def test_conditional_distributions(self) -> None:
        table = joint()
        assert approx_vector(table.conditional_b_given_a(0).probabilities, (0.75, 0.25))
        assert approx_vector(table.conditional_b_given_a(1).probabilities, (1 / 3, 2 / 3))
        assert approx_vector(table.conditional_a_given_b(0).probabilities, (0.6, 0.4))

    def test_bayes_posterior_equals_the_column_conditional(self) -> None:
        # 先验取表的边缘时，后验必然等于列条件分布——这是一条可断言的自洽性
        table = joint()
        for index in range(2):
            posterior = table.posterior_a_given_b(index)
            assert approx_vector(
                posterior.probabilities, table.conditional_a_given_b(index).probabilities
            )

    def test_custom_prior_changes_the_posterior(self) -> None:
        table = joint()
        sneaky_prior = Distribution(probabilities=(0.9, 0.1), labels=table.names_a)
        posterior = table.posterior_a_given_b(0, prior=sneaky_prior)
        # 先验大幅偏向"下雨"，后验也应当更高
        assert posterior.probabilities[0] > table.conditional_a_given_b(0).probabilities[0]

    def test_posterior_prior_size_is_validated(self) -> None:
        table = joint()
        with pytest.raises(TableError):
            table.posterior_a_given_b(0, prior=Distribution.uniform(3))

    def test_zero_likelihood_with_positive_prior_is_rejected(self) -> None:
        table = JointTable(
            cells=((0.5, 0.0), (0.0, 0.5)),
            names_a=("A1", "A2"),
            names_b=("B1", "B2"),
        )
        with pytest.raises(NumericError):
            table.posterior_a_given_b(0, prior=Distribution.uniform(2))

    def test_independent_table_has_zero_mutual_information(self) -> None:
        table = independent_joint()
        # 互信息在数学上是 0；浮点上是一个 1e-17 级的残差（乘除法的舍入）
        assert approx(table.mutual_information(), 0.0)
        assert approx(table.independence_gap(), 0.0, tolerance=1e-12)
        assert table.is_independent() is True

    def test_dependent_table_has_positive_mutual_information(self) -> None:
        table = joint()
        assert table.mutual_information() > 0
        assert table.independence_gap() > 0
        assert table.is_independent() is False

    def test_zero_row_cannot_be_conditioned_on(self) -> None:
        # 第 2 行全零 → P(A = A2) = 0 → "在这个条件下"没有定义
        table = JointTable(
            cells=((1.0, 0.0), (0.0, 0.0)),
            names_a=("A1", "A2"),
            names_b=("B1", "B2"),
        )
        with pytest.raises(TableError):
            table.conditional_b_given_a(1)

    def test_zero_column_cannot_be_conditioned_on(self) -> None:
        table = JointTable(
            cells=((1.0, 0.0), (0.0, 0.0)),
            names_a=("A1", "A2"),
            names_b=("B1", "B2"),
        )
        with pytest.raises(TableError):
            table.conditional_a_given_b(1)

    def test_index_out_of_range(self) -> None:
        with pytest.raises(ParameterError):
            joint().conditional_b_given_a(5)
        with pytest.raises(ParameterError):
            joint().conditional_a_given_b(-1)

    def test_name_count_must_match(self) -> None:
        with pytest.raises(TableError):
            JointTable(cells=((0.5, 0.5),), names_a=("a", "b"), names_b=("x", "y"))
        with pytest.raises(TableError):
            JointTable(cells=((0.5, 0.5),), names_a=("a",), names_b=("x",))

    def test_duplicate_names_are_rejected(self) -> None:
        with pytest.raises(TableError):
            JointTable(cells=((0.5, 0.5),), names_a=("dup", "dup"), names_b=("x", "y"))

    def test_negative_and_unnormalized_cells_are_rejected(self) -> None:
        with pytest.raises(TableError):
            JointTable(cells=((1.2, -0.2),), names_a=("a",), names_b=("x", "y"))
        with pytest.raises(NumericError):
            JointTable(cells=((0.2, 0.2),), names_a=("a",), names_b=("x", "y"))

    def test_tolerance_is_honored(self) -> None:
        # 容差内的"没归一"是舍入误差（1e-12 级），一律放行
        table = JointTable(
            cells=((0.5, 0.5 + 1e-12),), names_a=("a",), names_b=("x", "y")
        )
        assert math.isfinite(table.mutual_information())

    def test_unnormalized_table_is_rejected(self) -> None:
        # 容差外的"没归一"是真错：一张和为 0.6 的表算出来的条件概率全是错的
        with pytest.raises(NumericError):
            JointTable(cells=((0.3, 0.3),), names_a=("a",), names_b=("x", "y"))

    def test_dict_carries_the_two_summaries(self) -> None:
        import json

        payload = joint().to_dict()
        json.dumps(payload)
        assert payload["shape"] == [2, 2]
        assert payload["mutual_information"] > 0
