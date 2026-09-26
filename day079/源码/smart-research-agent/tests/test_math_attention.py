"""``math_foundations.attention``：打分、掩码、多头、位置编码与方差实验（day073）.

本文件钉住这一课最核心的两条性质：

```text
① 缩放的作用可以被**量出来**：打分的方差随 d 线性增长、除以 √d 之后回到常数
② 因果掩码是"权重恰好为 0"：位置 i 看不到 i 之后的位置（不是"很少看到"）
```

第二条尤其重要：用 ``-inf`` 的实现数值上等价（``exp(−∞) = 0``），
但本包用**显式掩码**——因为 softmax 拒绝非有限数（见 ``masked_softmax_rows``）。
"""

from __future__ import annotations

import math

import pytest

from smart_research_agent.math_foundations.attention import (
    POSITIONAL_BASE,
    DotProductStudy,
    attention_scores,
    causal_mask,
    masked_softmax_rows,
    merge_heads,
    multi_head_attention,
    positional_encoding,
    sampled_dot_product_variance,
    scaled_dot_product_attention,
    scaling_factor,
    split_heads,
)
from smart_research_agent.math_foundations.errors import NumericError, ParameterError
from tests.math_samples import KEYS, QUERIES, VALUES, approx, approx_vector


class TestScaling:
    """缩放系数与打分矩阵（手算可验证的 1/√d_k）."""

    def test_scaling_factor_is_one_over_sqrt_d(self) -> None:
        assert scaling_factor(4) == 0.5
        assert approx(scaling_factor(1), 1.0)
        assert approx(scaling_factor(2), 1.0 / math.sqrt(2.0))
        with pytest.raises(ParameterError):
            scaling_factor(0)

    def test_scores_are_scaled_dot_products(self) -> None:
        queries = ((1.0, 0.0, 0.0, 0.0),)
        keys = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0))
        scores = attention_scores(queries, keys)
        # 点积 = (1, 0)，乘 scale = 1/√4 = 0.5
        assert approx_vector(scores[0], (0.5, 0.0))

    def test_explicit_scale_is_used_verbatim(self) -> None:
        queries = ((1.0, 0.0, 0.0, 0.0),)
        keys = ((1.0, 0.0, 0.0, 0.0),)
        assert approx_vector(attention_scores(queries, keys, scale=1.0)[0], (1.0,))

    def test_head_dimension_must_match(self) -> None:
        with pytest.raises(NumericError):
            attention_scores(((1.0, 0.0, 0.0),), ((1.0, 0.0),))

    def test_scale_must_be_positive(self) -> None:
        with pytest.raises(ParameterError):
            attention_scores(QUERIES, KEYS, scale=0.0)
        with pytest.raises(ParameterError):
            attention_scores(QUERIES, KEYS, scale=-1.0)


class TestCausalMask:
    """因果掩码：下三角（含对角线）允许、上三角禁止."""

    def test_mask_shape_and_values(self) -> None:
        mask = causal_mask(3)
        assert mask == (
            (True, False, False),
            (True, True, False),
            (True, True, True),
        )

    def test_mask_validates_size(self) -> None:
        with pytest.raises(ParameterError):
            causal_mask(0)

    def test_masked_softmax_puts_exact_zeros(self) -> None:
        scores = ((1.0, 2.0, 3.0), (1.0, 2.0, 3.0), (1.0, 2.0, 3.0))
        weights = masked_softmax_rows(scores, causal_mask(3))
        assert weights[0][1] == 0.0 and weights[0][2] == 0.0
        assert weights[1][2] == 0.0
        # 第 0 行只能看自己 → 权重恰好是 1（这就是"看不到未来"的可断言形式）
        assert weights[0][0] == 1.0
        for row in weights:
            assert approx(sum(row), 1.0)

    def test_masked_softmax_matches_a_manual_computation(self) -> None:
        scores = ((1.0, 3.0), (2.0, 0.0))
        weights = masked_softmax_rows(scores, ((True, False), (True, True)))
        # 第 0 行只有一列允许 → 1.0；第 1 行两列都允许 → softmax([2, 0])
        assert weights[0] == (1.0, 0.0)
        expected_second = math.e**2 / (math.e**2 + 1.0)
        assert approx(weights[1][0], expected_second)

    def test_row_without_allowed_position_is_rejected(self) -> None:
        with pytest.raises(NumericError):
            masked_softmax_rows(((1.0, 2.0),), ((False, False),))

    def test_mask_shape_must_match_scores(self) -> None:
        with pytest.raises(NumericError):
            masked_softmax_rows(((1.0, 2.0),), causal_mask(3))


class TestScaledDotProductAttention:
    """一次完整的注意力：权重、输出、熵、峰值与变体判定."""

    def test_report_shape_and_row_sums(self) -> None:
        report = scaled_dot_product_attention(QUERIES, KEYS, VALUES)
        assert (report.queries, report.keys) == (3, 3)
        assert report.head_dim == 2
        assert report.scale == 0.5  # 1/√4
        assert len(report.weights) == 3
        for row in report.weights:
            assert approx(sum(row), 1.0)

    def test_output_is_a_weighted_average_of_values(self) -> None:
        # 第 1 行（等价于只有一列允许）应当输出对应的那一行 value
        single_value = ((0.0, 1.0),)
        single_key = ((1.0, 1.0, 1.0, 1.0),)
        query = ((1.0, 1.0, 1.0, 1.0),)
        report = scaled_dot_product_attention(query, single_key, single_value)
        assert approx_vector(report.weights[0], (1.0,))
        assert approx_vector(report.output[0], (0.0, 1.0))

    def test_entropy_and_peaks_are_reported_per_row(self) -> None:
        report = scaled_dot_product_attention(QUERIES, KEYS, VALUES)
        assert len(report.entropies) == len(report.peak_weights) == len(report.peak_indices)
        for row, entropy_value, peak, index in zip(
            report.weights, report.entropies, report.peak_weights, report.peak_indices
        ):
            assert approx(entropy_value, -sum(
                value * math.log(value) for value in row if value > 0
            ))
            assert peak == max(row)
            assert row[index] == peak

    def test_uniform_scores_give_maximal_entropy(self) -> None:
        # 所有 query 与 key 相同 → 打分相同 → 权重均匀 → 熵 = ln(3)
        identical = ((1.0, 0.0), (1.0, 0.0), (1.0, 0.0))
        values = ((1.0,), (0.0,), (0.0,))
        report = scaled_dot_product_attention(identical, identical, values)
        assert approx(report.mean_entropy, math.log(3.0))
        assert approx(report.focus_ratio(), 0.0)

    def test_causal_variant_blocks_the_future(self) -> None:
        report = scaled_dot_product_attention(QUERIES, KEYS, VALUES, causal=True)
        assert report.variant == "causal" and report.causal is True
        assert report.weights[0] == (1.0, 0.0, 0.0)
        assert report.weights[1][2] == 0.0
        # 最后一行的可见集合是全集，因此它与"不掩码"的那一行相同
        plain = scaled_dot_product_attention(QUERIES, KEYS, VALUES)
        assert approx_vector(report.weights[2], plain.weights[2])
        assert any("因果掩码" in note for note in report.notes)

    def test_temperature_sharpens_and_flattens(self) -> None:
        cold = scaled_dot_product_attention(QUERIES, KEYS, VALUES, temperature=0.05)
        warm = scaled_dot_product_attention(QUERIES, KEYS, VALUES, temperature=50.0)
        assert cold.focus_ratio() > warm.focus_ratio()
        assert cold.temperature == 0.05
        assert any("温度" in note for note in cold.notes)

    @pytest.mark.parametrize("temperature", [0.0, -1.0, float("inf")])
    def test_bad_temperature_is_rejected(self, temperature: float) -> None:
        with pytest.raises(ParameterError):
            scaled_dot_product_attention(QUERIES, KEYS, VALUES, temperature=temperature)

    def test_keys_and_values_must_align(self) -> None:
        with pytest.raises(NumericError):
            scaled_dot_product_attention(QUERIES, KEYS, ((1.0, 0.0),))

    def test_causal_requires_square_scores(self) -> None:
        with pytest.raises(NumericError):
            scaled_dot_product_attention(QUERIES[:2], KEYS, VALUES, causal=True)

    def test_explicit_unit_scale_is_reported_as_dot(self) -> None:
        report = scaled_dot_product_attention(QUERIES, KEYS, VALUES, scale=1.0)
        assert report.variant == "dot"
        assert report.scale == 1.0

    def test_dict_is_json_friendly(self) -> None:
        import json

        payload = scaled_dot_product_attention(QUERIES, KEYS, VALUES).to_dict()
        json.dumps(payload)
        assert payload["variant"] == "scaled"
        assert 0.0 <= payload["focus_ratio"] <= 1.0


class TestMultiHead:
    """多头：拆分/合并的往返、每个头一个分布、拼回来的输出."""

    def test_split_and_merge_round_trip(self) -> None:
        parts = split_heads(QUERIES, 2)
        assert len(parts) == 2
        assert all(len(row) == 2 for part in parts for row in part)
        assert merge_heads(parts) == QUERIES

    def test_split_requires_divisible_width(self) -> None:
        with pytest.raises(NumericError):
            split_heads(QUERIES, 3)
        with pytest.raises(ParameterError):
            split_heads(QUERIES, 0)

    def test_merge_requires_same_row_count(self) -> None:
        with pytest.raises(NumericError):
            merge_heads((((1.0,),), ((1.0,), (2.0,))))
        with pytest.raises(ParameterError):
            merge_heads(())

    def test_multi_head_report(self) -> None:
        report = multi_head_attention(QUERIES, KEYS, VALUES, heads=2)
        assert report.heads == 2 and len(report.reports) == 2
        assert len(report.focus_by_head()) == 2
        assert len(report.output) == 3
        assert report.mean_entropy > 0
        assert "多头 2×3" in report.summary_line()
        assert any("d_k = 2" in note for note in report.notes)

    def test_head_count_must_match_reports(self) -> None:
        from smart_research_agent.math_foundations.attention import MultiHeadReport

        with pytest.raises(NumericError):
            MultiHeadReport(heads=3, reports=(), output=())
        with pytest.raises(ParameterError):
            MultiHeadReport(heads=0, reports=(), output=())

    def test_heads_with_different_scale(self) -> None:
        # 每个头的 d_k 是 d/heads，因此缩放系数是 1/√(d/heads) 而不是 1/√d
        report = multi_head_attention(QUERIES, KEYS, VALUES, heads=2)
        for part in report.reports:
            assert part.scale == scaling_factor(2)

    def test_dict_is_json_friendly(self) -> None:
        import json

        payload = multi_head_attention(QUERIES, KEYS, VALUES, heads=2).to_dict()
        json.dumps(payload)
        assert payload["heads"] == 2


class TestPositionalEncoding:
    """位置编码：pos = 0 时是 (0, 1, 0, 1, …)，且每一维是一条不同频率的波."""

    def test_position_zero_is_sin0_cos0(self) -> None:
        encoding = positional_encoding(1, 4)
        assert approx_vector(encoding[0], (0.0, 1.0, 0.0, 1.0))

    def test_shape_and_bounds(self) -> None:
        encoding = positional_encoding(5, 6)
        assert len(encoding) == 5
        assert all(len(row) == 6 for row in encoding)
        assert all(-1.0 <= value <= 1.0 for row in encoding for value in row)

    def test_each_dimension_has_its_own_frequency(self) -> None:
        # 低频那一维（第 0 维）变化快，高频那一维（index 大）变化慢
        encoding = positional_encoding(4, 4)
        low_freq_span = abs(encoding[3][2] - encoding[0][2])
        high_freq_span = abs(encoding[3][0] - encoding[0][0])
        assert low_freq_span < high_freq_span

    def test_odd_dimensions_are_supported(self) -> None:
        encoding = positional_encoding(2, 3)
        assert len(encoding[0]) == 3
        assert encoding[0][0] == 0.0

    @pytest.mark.parametrize(
        "positions,dimension",
        [(0, 4), (2, 1), (2, 0)],
    )
    def test_bad_sizes_are_rejected(self, positions: int, dimension: int) -> None:
        with pytest.raises(ParameterError):
            positional_encoding(positions, dimension)

    def test_bad_base_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            positional_encoding(2, 4, base=1.0)
        assert POSITIONAL_BASE == 10000.0


class TestDotProductVarianceStudy:
    """数值实验：方差随维度线性增长，除以 √d 之后回到常数."""

    def test_variance_grows_with_dimension(self) -> None:
        small = sampled_dot_product_variance(4, trials=800, seed=42)
        large = sampled_dot_product_variance(64, trials=800, seed=42)
        assert large.measured_variance > small.measured_variance * 10

    def test_scaled_std_is_constant_across_dimensions(self) -> None:
        # 除以 √d 之后，打分标准差回到 1/3（= 每个分量的方差 σ² = bound²/3）
        # 这才是"缩放点积注意力"里那个缩放的**全部意义**：
        # 不缩放时标准差随 √d 增长（d=256 时约 5.34），缩放后与 d 无关。
        for dimension in (4, 16, 64, 256):
            study = sampled_dot_product_variance(dimension, trials=2000, seed=42)
            assert abs(study.scaled_std - 1.0 / 3.0) < 0.03, dimension

    def test_measured_variance_is_close_to_the_theory(self) -> None:
        study = sampled_dot_product_variance(64, trials=2000, seed=42)
        assert abs(study.measured_variance - study.theoretical_variance) / (
            study.theoretical_variance
        ) < 0.15

    def test_study_is_deterministic(self) -> None:
        first = sampled_dot_product_variance(16, trials=200, seed=3)
        second = sampled_dot_product_variance(16, trials=200, seed=3)
        assert first.to_dict() == second.to_dict()

    def test_parameters_are_validated(self) -> None:
        with pytest.raises(ParameterError):
            sampled_dot_product_variance(0)
        with pytest.raises(ParameterError):
            sampled_dot_product_variance(4, trials=0)
        with pytest.raises(ParameterError):
            sampled_dot_product_variance(4, component_bound=0.0)

    def test_study_dict_and_line(self) -> None:
        import json

        study = DotProductStudy(
            dimension=8,
            trials=100,
            component_bound=1.0,
            measured_variance=0.9,
            measured_mean=0.0,
            theoretical_variance=8.0 / 9.0,
        )
        json.dumps(study.to_dict())
        assert approx(study.std, math.sqrt(0.9))
        assert "d=8" in study.summary_line()
