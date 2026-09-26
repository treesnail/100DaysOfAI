"""性质检查与"注意力 vs 检索"的类比（day075）.

四条性质各自回答一个不同的问题，而它们的**结论可以互相独立地失败**：

```text
行和为 1        权重是一个条件分布
非负            加权平均没有变成减法
因果无泄漏      上三角恰好 0.0 —— "看不到未来"在数字上的样子
置换等变        注意力本身不知道顺序（**因果掩码会破坏它**）
```

最后一条是 day078（位置编码）的伏笔：它不是"通过/不通过"，
而是**在两种配置下应当得到两个不同的答案**——因此这一份测试里
既断言"无掩码时偏差为 0"，也断言"有掩码时偏差显著不为 0"。
"""

from __future__ import annotations

import json
import math

import pytest

from smart_research_agent.transformer_core.errors import (
    NumericError,
    ParameterError,
)
from smart_research_agent.transformer_core.layers import self_attention
from smart_research_agent.transformer_core.types import (
    PROPERTY_CHECKS,
    PROPERTY_CHECK_DESCRIPTIONS,
)
from smart_research_agent.transformer_core.verify import (
    PropertyOutcome,
    PropertyReport,
    RetrievalComparison,
    check_causal_no_leak,
    check_non_negative,
    check_permutation_equivariance,
    check_properties,
    check_row_stochastic,
    compare_with_retrieval,
    default_permutation,
    permutation_gap,
    rank_correlation,
)
from tests.attention_samples import (
    ONE_HOT_INPUTS,
    SMALL_INPUTS,
    approx,
    identity_parameters,
    small_parameters,
)


class TestPropertyTables:
    """性质表逐键对齐."""

    def test_tables_align(self):
        assert set(PROPERTY_CHECKS) == set(PROPERTY_CHECK_DESCRIPTIONS)

    def test_four_properties(self):
        assert len(PROPERTY_CHECKS) == 4


class TestDefaultPermutation:
    """缺省置换 = 倒序（对任何长度都合法）."""

    def test_reverses_the_order(self):
        assert default_permutation(4) == (3, 2, 1, 0)

    def test_single_row(self):
        assert default_permutation(1) == (0,)

    def test_validates(self):
        with pytest.raises(ParameterError, match="整数"):
            default_permutation(2.0)  # type: ignore[arg-type]
        with pytest.raises(ParameterError, match="行数"):
            default_permutation(0)


class TestRowStochastic:
    """行和为 1：前向一定是满足的（因为走了 softmax），这条检查守的是"以后"."""

    def test_identity_case(self):
        forward = self_attention(identity_parameters(2), ONE_HOT_INPUTS, causal=True)
        outcome = check_row_stochastic(forward)
        assert outcome.passed is True
        assert "行和最大偏差" in outcome.evidence
        assert outcome.description

    def test_random_case(self):
        forward = self_attention(small_parameters(), SMALL_INPUTS, causal=True)
        assert check_row_stochastic(forward).passed is True

    def test_tolerance_can_be_tightened(self):
        forward = self_attention(small_parameters(), SMALL_INPUTS, causal=True)
        assert check_row_stochastic(forward, tolerance=0.0).passed is False


class TestNonNegative:
    """非负：softmax 的输出天然非负."""

    def test_passes_on_a_real_forward(self):
        forward = self_attention(small_parameters(), SMALL_INPUTS, causal=True)
        outcome = check_non_negative(forward)
        assert outcome.passed is True
        assert "最小权重" in outcome.evidence

    def test_outcome_is_json_friendly(self):
        forward = self_attention(small_parameters(), SMALL_INPUTS)
        payload = check_non_negative(forward).to_dict()
        json.dumps(payload)
        assert payload["name"] == "non_negative"


class TestCausalNoLeak:
    """因果无泄漏：上三角恰好是 0.0."""

    def test_causal_forward_has_exact_zeros(self):
        forward = self_attention(small_parameters(), SMALL_INPUTS, causal=True)
        outcome = check_causal_no_leak(forward)
        assert outcome.passed is True
        assert outcome.evidence.endswith("0.00e+00")

    def test_full_forward_is_reported_as_not_applicable(self):
        """没开掩码时不是"通过"，而是"没有可检查的对象"——两者必须区分开."""
        forward = self_attention(small_parameters(), SMALL_INPUTS, causal=False)
        outcome = check_causal_no_leak(forward)
        assert outcome.passed is True
        assert "未启用因果掩码" in outcome.evidence
        assert "跳过" in outcome.evidence

    def test_upper_triangle_is_exactly_zero_not_approximately(self):
        forward = self_attention(small_parameters(), SMALL_INPUTS, causal=True)
        for row in range(3):
            for column in range(row + 1, 3):
                assert forward.weights[row][column] == 0.0


class TestPermutationEquivariance:
    """置换等变：无掩码时成立，因果掩码下**被破坏**."""

    def test_gap_is_zero_without_mask(self):
        params = small_parameters()
        assert permutation_gap(params, SMALL_INPUTS, causal=False) < 1e-12

    def test_check_passes_without_mask(self):
        outcome = check_permutation_equivariance(
            small_parameters(), SMALL_INPUTS, causal=False
        )
        assert outcome.passed is True
        assert "最大偏差" in outcome.evidence

    def test_causal_mask_breaks_equivariance(self):
        """**这不是 bug**：掩码把顺序信息写进了那张表里.

        实测偏差对 3 行的样本约 0.1~0.3（与输出本身的量级相当），
        而它正是"顺序必须由输入带进来"的直接证据。
        """
        gap = permutation_gap(small_parameters(), SMALL_INPUTS, causal=True)
        assert gap > 1e-3

    def test_check_fails_with_causal_mask(self):
        outcome = check_permutation_equivariance(
            small_parameters(), SMALL_INPUTS, causal=True
        )
        assert outcome.passed is False

    def test_explicit_permutation_is_used(self):
        """给一个"只交换前两行"的置换，偏差应当与倒序不同（但都非零）."""
        params = small_parameters()
        swap = permutation_gap(params, SMALL_INPUTS, causal=True, permutation=(1, 0, 2))
        reverse = permutation_gap(params, SMALL_INPUTS, causal=True)
        assert swap > 0.0
        assert reverse > 0.0
        assert swap != pytest.approx(reverse)

    def test_invalid_permutation_is_rejected(self):
        with pytest.raises(ParameterError, match="排列"):
            permutation_gap(small_parameters(), SMALL_INPUTS, permutation=(0, 1))

    def test_identity_parameters_are_equivariant_without_mask(self):
        """单位矩阵 + 无掩码：输出就是输入的某种混合，换序结论显然成立."""
        assert permutation_gap(identity_parameters(2), ONE_HOT_INPUTS, causal=False) < 1e-12


class TestPropertyReport:
    """四条性质的汇总."""

    def test_report_is_complete_and_ok(self):
        report = check_properties(small_parameters(), SMALL_INPUTS, causal=True)
        assert report.ok is True
        assert tuple(item.name for item in report.outcomes) == PROPERTY_CHECKS

    def test_report_summary_and_notes(self):
        report = check_properties(small_parameters(), SMALL_INPUTS, causal=True)
        assert "性质检查 4 项" in report.summary_line()
        assert report.notes
        assert "位置编码" in " ".join(report.notes)

    def test_report_is_json_friendly(self):
        payload = check_properties(small_parameters(), SMALL_INPUTS, causal=True).to_dict()
        json.dumps(payload)
        assert payload["ok"] is True
        assert len(payload["outcomes"]) == 4

    def test_missing_property_is_rejected(self):
        report = check_properties(small_parameters(), SMALL_INPUTS, causal=True)
        with pytest.raises(NumericError, match="缺少"):
            PropertyReport(outcomes=report.outcomes[:2])

    def test_full_mask_report_also_ok(self):
        assert check_properties(small_parameters(), SMALL_INPUTS, causal=False).ok is True


class TestPropertyOutcomeValidation:
    """``PropertyOutcome`` 只接受表里的名字."""

    def test_unknown_name_is_rejected(self):
        with pytest.raises(NumericError, match="不认识的性质"):
            PropertyOutcome(name="mystery", passed=True, evidence="", detail="")

    def test_projection(self):
        outcome = PropertyOutcome(
            name="row_stochastic", passed=False, evidence="偏差 0.1", detail="说明"
        )
        assert outcome.passed is False
        assert outcome.description
        assert "[FAIL]" in outcome.summary_line()
        assert outcome.to_dict()["name"] == "row_stochastic"


class TestRankCorrelation:
    """秩相关：三个手算案例 + 两个必须写下来的约定."""

    def test_perfect_agreement(self):
        assert rank_correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)

    def test_perfect_reversal(self):
        assert rank_correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)

    def test_ties_use_average_ranks(self):
        """``(1,1,2)`` 的平均秩是 ``(1.5, 1.5, 3)``，与 ``(1,2,3)`` 的相关系数是 ``√3/2``.

        手算：中心化后 ``(−0.5, −0.5, 1)`` 与 ``(−1, 0, 1)``，
        协方差 ``1.5``、两边平方和 ``1.5`` 与 ``2``，于是 ``1.5/√3 = 0.866025``。
        如果并列用"字典序名次"，这个数会变成 1.0——**两种约定会给出不同的答案**，
        因此它必须写在文档里。
        """
        assert rank_correlation([1.0, 1.0, 2.0], [1.0, 2.0, 3.0]) == pytest.approx(math.sqrt(3) / 2)

    def test_zero_variance_gives_zero(self):
        assert rank_correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0

    def test_single_point_gives_zero(self):
        assert rank_correlation([1.0], [2.0]) == 0.0

    def test_empty_lists_give_zero(self):
        assert rank_correlation([], []) == 0.0

    def test_length_mismatch_is_rejected(self):
        from smart_research_agent.math_foundations.errors import ShapeError as MathShapeError

        with pytest.raises(MathShapeError, match="长度不同"):
            rank_correlation([1.0, 2.0], [1.0])


class TestRetrievalComparison:
    """与检索的类比：三个读数各自回答一件事."""

    def _comparison(self, *, top_k: int = 2):
        forward = self_attention(identity_parameters(2), ONE_HOT_INPUTS, causal=True)
        return compare_with_retrieval(forward, top_k=top_k)

    def test_identity_case_is_hand_computable(self):
        """两行 one-hot + 因果掩码：

        ```text
        第 0 行只有 1 个候选（权重 1.0）      → 跳过排序，记 1 次一致 + overlap 1
        第 1 行看到 [0, 1]，注意力峰值在 1     → 余弦峰值也在 1（自己与自己），一致
        top-2 在第 1 行覆盖全部两个候选       → overlap 再 +2
        合计：峰值一致 2/2、overlap 3/4 = 75%、秩相关 1.0
        ```
        """
        comparison = self._comparison()
        assert comparison.rows == 2
        assert comparison.top_k == 2
        assert comparison.peak_agreement == 2
        assert comparison.peak_ratio == pytest.approx(1.0)
        assert comparison.overlap == 3
        assert comparison.overlap_ratio == pytest.approx(0.75)
        assert comparison.rank_correlation == pytest.approx(1.0)

    def test_top_k_one(self):
        comparison = self._comparison(top_k=1)
        assert comparison.top_k == 1
        assert comparison.overlap <= comparison.rows

    def test_general_case_readings(self):
        forward = self_attention(small_parameters(), SMALL_INPUTS, causal=True)
        comparison = compare_with_retrieval(forward)
        assert comparison.rows == 3
        assert 0 <= comparison.peak_agreement <= 3
        assert 0.0 <= comparison.overlap_ratio <= 1.0
        assert -1.0 <= comparison.rank_correlation <= 1.0

    def test_notes_explain_the_masking_caveat(self):
        notes = " ".join(self._comparison().notes)
        assert "允许的位置" in notes
        assert "可微" in notes

    def test_summary_and_json(self):
        comparison = self._comparison()
        assert "峰值一致" in comparison.summary_line()
        payload = comparison.to_dict()
        json.dumps(payload)
        assert payload["top_k"] == 2
        assert payload["notes"]

    def test_top_k_validation(self):
        forward = self_attention(identity_parameters(2), ONE_HOT_INPUTS)
        with pytest.raises(ParameterError, match="top_k"):
            compare_with_retrieval(forward, top_k=0)
        with pytest.raises(ParameterError, match="超过行数"):
            compare_with_retrieval(forward, top_k=5)

    def test_comparison_validation(self):
        with pytest.raises(ParameterError, match="top_k"):
            RetrievalComparison(top_k=0, rows=2, peak_agreement=1, overlap=1, rank_correlation=0.0)
        with pytest.raises(ParameterError, match="行数"):
            RetrievalComparison(top_k=1, rows=0, peak_agreement=0, overlap=0, rank_correlation=0.0)
        with pytest.raises(NumericError, match="峰值一致"):
            RetrievalComparison(top_k=1, rows=2, peak_agreement=5, overlap=1, rank_correlation=0.0)
        with pytest.raises(NumericError, match="重合数"):
            RetrievalComparison(top_k=1, rows=2, peak_agreement=1, overlap=9, rank_correlation=0.0)
