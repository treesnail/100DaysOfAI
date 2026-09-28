"""六条性质、头间差异与"头序只是记账"（day076 / M7-D2）.

这一份测试守两件"多头特有"的事：

```text
① heads = 1 必须**逐位**等于 day075（前向与 context）
   —— 它是"多头是单头的严格超集"这条工程结论的回归护栏
② "头序只是一个约定"必须能被**双向**验证：
   一致地重排 → 输出不变；只重排一个矩阵 → 输出必须变
```

第 ② 条的后半句才是这条性质的内容。只测"一致重排不变"会漏掉一种实现：
**把 W_o 的列也按行块切**——那样"一致重排"仍然不变，而这一层的语义已经错了。
"""

from __future__ import annotations

import math

import pytest

from smart_research_agent.multi_head.errors import (
    NumericError,
    ParameterError,
    PartitionError,
    ShapeError,
)
from smart_research_agent.multi_head.layers import multi_head_attention
from smart_research_agent.multi_head.types import (
    MULTIHEAD_PROPERTIES,
    HeadPartition,
    MultiHeadShape,
)
from smart_research_agent.multi_head.verify import (
    HeadDisagreement,
    MultiHeadPropertyOutcome,
    MultiHeadPropertyReport,
    check_causal_no_leak_per_head,
    check_head_non_negative,
    check_head_order_is_bookkeeping,
    check_merge_inverts_split,
    check_per_head_row_stochastic,
    check_properties,
    check_single_head_matches_classic,
    checked_head_order,
    default_head_order,
    head_disagreement,
    head_weight_table,
    permute_head_blocks,
    total_variation,
)
from smart_research_agent.transformer_core.types import AttentionParams, relative_matrix_error
from tests.multihead_samples import (
    HEAD_COUNTS,
    IDENTITY_INPUTS,
    IDENTITY_ROW_TOTAL_VARIATIONS,
    P_SQRT2,
    P_THREE_SQRT2_HALF,
    approx,
    identity_parameters,
    induction_tasks,
    repeated_block_parameters,
    toy_parameters,
    witness_sample,
)


def _forward(heads: int = 2, *, causal: bool = True, params=None, inputs=IDENTITY_INPUTS):
    return multi_head_attention(
        identity_parameters(4) if params is None else params,
        inputs,
        heads=heads,
        causal=causal,
    )


class TestTotalVariation:
    """全变差距离：为什么这一课用它而不是 KL（对称、有界、没有无穷）."""

    def test_hand_computed(self):
        assert approx(total_variation((0.7, 0.3), (0.4, 0.6)), 0.3)

    def test_identical_distributions_are_zero(self):
        assert total_variation((0.25,) * 4, (0.25,) * 4) == 0.0

    def test_disjoint_support_is_one(self):
        assert total_variation((1.0, 0.0), (0.0, 1.0)) == 1.0

    def test_is_symmetric(self):
        left = (0.9, 0.05, 0.05)
        right = (0.1, 0.8, 0.1)
        assert total_variation(left, right) == total_variation(right, left)

    def test_length_mismatch_is_rejected(self):
        with pytest.raises(ShapeError):
            total_variation((0.5, 0.5), (1.0,))

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_non_finite_is_rejected(self, value):
        with pytest.raises(NumericError):
            total_variation((0.5, 0.5), (value, 0.5))


class TestHeadDisagreement:
    """各头"看到的东西"有多不一样——多头最大的失败模式要能被看见."""

    def test_hand_computed_mean_and_max(self):
        """单位配置下每一行的 TV 都能手算（第 1 行 = |p − q|）."""
        report = head_disagreement(_forward(2))
        expected_row_one = IDENTITY_ROW_TOTAL_VARIATIONS[1]
        assert approx(report.mean_total_variation, expected_row_one / 2.0)
        assert approx(report.max_total_variation, expected_row_one)
        assert approx(report.min_total_variation, 0.0)

    def test_hand_computed_components(self):
        report = head_disagreement(_forward(2))
        assert approx(report.mean_total_variation, (P_SQRT2 - P_THREE_SQRT2_HALF) / 2.0)
        assert report.pairs == 1
        assert report.comparisons == 2
        assert report.rows == 2

    def test_counts(self):
        report = head_disagreement(
            multi_head_attention(toy_parameters(6), induction_tasks(1)[0].inputs, heads=3)
        )
        assert report.pairs == 3
        assert report.heads == 3
        assert report.comparisons == 3 * report.rows

    def test_single_head_has_no_pairs(self):
        report = head_disagreement(_forward(1))
        assert report.pairs == 0
        assert report.peak_agreement == 0
        assert report.peak_agreement_ratio == 0.0
        # 单头**不是**"退化"：没有可以退化的对象
        assert report.degenerate is False
        assert "没有可比的头对" in report.summary_line()

    def test_identical_heads_are_degenerate(self):
        params = repeated_block_parameters(6, 2)
        forward = multi_head_attention(params, induction_tasks(1)[0].inputs, heads=2)
        report = head_disagreement(forward)
        assert report.mean_total_variation == 0.0
        assert report.max_total_variation == 0.0
        assert report.degenerate is True

    def test_peak_agreement_ratio_bounds(self):
        report = head_disagreement(
            multi_head_attention(toy_parameters(6), induction_tasks(2)[0].inputs, heads=3)
        )
        assert 0.0 <= report.peak_agreement_ratio <= 1.0

    def test_to_dict(self):
        payload = head_disagreement(_forward(2)).to_dict()
        assert payload["pairs"] == 1
        assert payload["comparisons"] == 2
        assert isinstance(payload["degenerate"], bool)

    def test_summary_line(self):
        line = head_disagreement(_forward(2)).summary_line()
        assert "2 头" in line
        assert "平均 TV" in line


class TestHeadDisagreementValidation:
    """读数对象的构造校验（一个越界的统计量会让整张对照表失真）."""

    def _kwargs(self, **overrides):
        payload = {
            "heads": 2,
            "rows": 3,
            "pairs": 1,
            "mean_total_variation": 0.5,
            "max_total_variation": 0.6,
            "min_total_variation": 0.4,
            "peak_agreement": 1,
        }
        payload.update(overrides)
        return payload

    def test_heads_must_be_positive(self):
        with pytest.raises(ParameterError):
            HeadDisagreement(**self._kwargs(heads=0))

    def test_pairs_must_match_the_head_count(self):
        with pytest.raises(NumericError):
            HeadDisagreement(**self._kwargs(pairs=3))

    def test_rows_must_be_positive(self):
        with pytest.raises(ParameterError):
            HeadDisagreement(**self._kwargs(rows=0))

    def test_tv_must_be_a_distance(self):
        with pytest.raises(NumericError):
            HeadDisagreement(**self._kwargs(max_total_variation=1.5))
        with pytest.raises(NumericError):
            HeadDisagreement(
                **self._kwargs(
                    min_total_variation=0.7, max_total_variation=0.6
                )
            )

    def test_mean_must_be_between_bounds(self):
        with pytest.raises(NumericError):
            HeadDisagreement(**self._kwargs(mean_total_variation=0.9))

    def test_peak_agreement_must_be_bounded(self):
        with pytest.raises(NumericError):
            HeadDisagreement(**self._kwargs(peak_agreement=99))

    def test_notes_are_stringified(self):
        report = HeadDisagreement(**self._kwargs(notes=(1, 2)))
        assert report.notes == ("1", "2")


class TestPropertyChecks:
    """六条性质：每一项都能独立失败，而且证据里要带上被检查的规模."""

    @pytest.mark.parametrize("heads", HEAD_COUNTS)
    def test_all_six_pass(self, heads):
        params = toy_parameters(6)
        report = check_properties(params, induction_tasks(1)[0].inputs, heads=heads)
        assert report.ok
        assert len(report.outcomes) == len(MULTIHEAD_PROPERTIES)

    def test_row_stochastic_evidence_carries_the_row_count(self):
        forward = _forward(2)
        outcome = check_per_head_row_stochastic(forward)
        assert outcome.passed
        assert "4 行" in outcome.evidence

    def test_non_negative(self):
        outcome = check_head_non_negative(_forward(2))
        assert outcome.passed
        assert "最小权重" in outcome.evidence

    def test_causal_leak_is_checked_per_head(self):
        params = toy_parameters(6)
        forward = multi_head_attention(
            params, induction_tasks(1)[0].inputs, heads=3, causal=True
        )
        outcome = check_causal_no_leak_per_head(forward)
        assert outcome.passed
        assert "3 头" in outcome.evidence

    def test_causal_leak_is_skipped_when_not_causal(self):
        """"没开掩码"不是"掩码正确"——报告里必须把两者分开."""
        outcome = check_causal_no_leak_per_head(_forward(2, causal=False))
        assert outcome.passed
        assert "跳过" in outcome.evidence
        assert "没有可检查的对象" in outcome.detail

    def test_merge_inverts_split_is_exact(self):
        outcome = check_merge_inverts_split(_forward(2))
        assert outcome.passed
        assert outcome.evidence.endswith("0.00e+00（9 组矩阵）")

    def test_single_head_matches_classic(self):
        outcome = check_single_head_matches_classic(
            identity_parameters(4), IDENTITY_INPUTS
        )
        assert outcome.passed
        assert "0.00e+00" in outcome.evidence

    def test_head_order_default_uses_reversal(self):
        outcome = check_head_order_is_bookkeeping(
            toy_parameters(6), induction_tasks(1)[0].inputs, heads=3
        )
        assert outcome.passed
        assert "头序 (2, 1, 0)" in outcome.evidence

    def test_report_notes_flag_the_approximate_property(self):
        report = check_properties(
            toy_parameters(6), induction_tasks(1)[0].inputs, heads=2
        )
        assert any("1e-12" in item for item in report.notes)

    def test_report_serialises(self):
        report = check_properties(
            toy_parameters(6), induction_tasks(1)[0].inputs, heads=2
        )
        payload = report.to_dict()
        assert payload["ok"] is True
        assert len(payload["outcomes"]) == 6
        assert report.summary_line() == "性质检查 6 项：通过 6、失败 0"


class TestHeadOrder:
    """"头序只是一个约定"：一致重排不变、只重排一个必须变."""

    @pytest.mark.parametrize("heads", (2, 3))
    def test_consistent_permutation_keeps_the_output(self, heads):
        params = toy_parameters(6)
        inputs = induction_tasks(1)[0].inputs
        base = multi_head_attention(params, inputs, heads=heads)
        shuffled = multi_head_attention(
            permute_head_blocks(params, heads), inputs, heads=heads
        )
        assert relative_matrix_error(shuffled.output, base.output) <= 1e-12

    def test_explicit_order_is_honoured(self):
        params = toy_parameters(6)
        identity_order = permute_head_blocks(params, 3, (0, 1, 2))
        assert identity_order.w_query == params.w_query
        assert identity_order.w_output == params.w_output

    def test_permuting_only_one_matrix_changes_the_output(self):
        """**只换 W_q 而不换 W_o** 会得到一组完全不同的参数——这就是判据的另一半."""
        params = toy_parameters(6)
        inputs = induction_tasks(1)[0].inputs
        partition = HeadPartition(6, 2)
        blocks = partition.split_rows(params.w_query)
        broken = AttentionParams(
            w_query=partition.merge_rows([blocks[1], blocks[0]]),
            w_key=params.w_key,
            w_value=params.w_value,
            w_output=params.w_output,
        )
        base = multi_head_attention(params, inputs, heads=2)
        wrong = multi_head_attention(broken, inputs, heads=2)
        assert relative_matrix_error(wrong.output, base.output) > 1e-6

    def test_single_head_permutation_is_the_identity(self):
        params = toy_parameters(6)
        assert permute_head_blocks(params, 1) == params

    def test_indivisible_heads_is_rejected(self):
        with pytest.raises(PartitionError):
            permute_head_blocks(toy_parameters(6), 4)

    def test_default_head_order(self):
        assert default_head_order(3) == (2, 1, 0)
        assert default_head_order(1) == (0,)

    def test_default_head_order_validation(self):
        with pytest.raises(ParameterError):
            default_head_order(0)
        with pytest.raises(ParameterError):
            default_head_order(2.0)  # type: ignore[arg-type]

    def test_checked_head_order_validation(self):
        assert checked_head_order((1, 0), 2) == (1, 0)
        with pytest.raises(ParameterError):
            checked_head_order((0, 0), 2)

    def test_head_order_with_the_witness_sample(self):
        """见证样本上也要成立（那个样本的参数结构很特殊：两行全零）."""
        witness = witness_sample()
        outcome = check_head_order_is_bookkeeping(
            witness.params, witness.inputs, heads=witness.heads
        )
        assert outcome.passed

    def test_witness_heads_really_differ(self):
        witness = witness_sample()
        forward = multi_head_attention(
            witness.params, witness.inputs, heads=witness.heads, causal=True
        )
        report = head_disagreement(forward)
        assert report.mean_total_variation > 1e-3
        assert report.degenerate is False


class TestPropertyRecords:
    """性质记录的构造校验：不认识的项、缺项的汇总都要拒绝."""

    def test_unknown_name_is_rejected(self):
        with pytest.raises(NumericError):
            MultiHeadPropertyOutcome(
                name="nope", passed=True, evidence="x", detail="y"
            )

    def test_description_comes_from_the_table(self):
        outcome = MultiHeadPropertyOutcome(
            name="merge_inverts_split", passed=True, evidence="x", detail="y"
        )
        assert "往返恒等" in outcome.description

    def test_report_rejects_incomplete_outcomes(self):
        outcome = MultiHeadPropertyOutcome(
            name="merge_inverts_split", passed=True, evidence="x", detail="y"
        )
        with pytest.raises(NumericError):
            MultiHeadPropertyReport(outcomes=(outcome,))

    def test_report_serialises(self):
        report = check_properties(
            toy_parameters(6), induction_tasks(1)[0].inputs, heads=2
        )
        payload = report.to_dict()
        assert payload["outcomes"][0]["name"] == MULTIHEAD_PROPERTIES[0]
        assert payload["outcomes"][0]["description"]

    def test_failing_outcome_marks_the_report(self):
        outcomes = tuple(
            MultiHeadPropertyOutcome(
                name=name, passed=name != "head_non_negative", evidence="e", detail="d"
            )
            for name in MULTIHEAD_PROPERTIES
        )
        report = MultiHeadPropertyReport(outcomes=outcomes)
        assert not report.ok
        assert report.summary_line().endswith("通过 5、失败 1")
        # 失败项自己的那一行必须带 FAIL 标记（报告里的红要能被单独读到）
        failing = [item for item in report.outcomes if not item.passed]
        assert len(failing) == 1
        assert failing[0].summary_line().startswith("[FAIL]")

    def test_notes_are_stringified(self):
        outcomes = tuple(
            MultiHeadPropertyOutcome(name=name, passed=True, evidence="e", detail="d")
            for name in MULTIHEAD_PROPERTIES
        )
        assert MultiHeadPropertyReport(outcomes=outcomes, notes=(1,)).notes == ("1",)


class TestHeadWeightTable:
    """"同一行的 heads 份分布"必须能被并排打出来——否则那句话不可核对."""

    def test_structure(self):
        table = head_weight_table(_forward(2))
        assert table["heads"] == 2
        assert table["tokens"] == 2
        assert table["distributions"] == 4
        assert len(table["rows"]) == 2

    def test_rows_carry_allowed_positions_and_argmax(self):
        table = head_weight_table(_forward(2))
        assert table["rows"][0]["allowed"] == [0]
        assert table["rows"][1]["allowed"] == [0, 1]
        assert len(table["rows"][1]["argmax"]) == 2
        assert len(table["rows"][1]["distributions"]) == 2

    def test_values_are_rounded_to_six_places(self):
        table = head_weight_table(_forward(2))
        # 第 0 行每一头都只看自己
        assert table["rows"][0]["distributions"][0] == [1.0, 0.0]
        assert table["rows"][0]["distributions"][1] == [1.0, 0.0]
        # 第 1 行两个头各给一个**手算锚点**的四舍五入（不是另算一遍）
        assert table["rows"][1]["distributions"][0][0] == round(P_SQRT2, 6)
        assert table["rows"][1]["distributions"][1][0] == round(P_THREE_SQRT2_HALF, 6)


class TestShapeHelpers:
    """形状记录的辅助构造（报告里"这一层是什么形状"的入口）."""

    def test_shape_from_parameters(self):
        shape = head_shape_of(toy_parameters(6), 3)
        assert shape.head_dim == 2
        assert isinstance(shape, MultiHeadShape)

    def test_indivisible_heads_is_rejected(self):
        with pytest.raises(PartitionError):
            head_shape_of(toy_parameters(6), 5)

    def test_entropy_ceiling_of_a_single_token(self):
        """``n = 1`` 时"熵的上界"是 0——因此集中度必须退回 0 而不是除零."""
        forward = _forward_with_one_token()
        assert forward.max_entropy() == 0.0
        assert forward.focus_ratio() == 0.0


def head_shape_of(params: AttentionParams, heads: int) -> MultiHeadShape:
    """从 ``reachability`` 里拿形状（放在文件末尾避免与测试类抢名字）."""
    from smart_research_agent.multi_head.reachability import multi_head_shape_of

    return multi_head_shape_of(params, heads)


def _forward_with_one_token():
    """只有一个 token 的前向（``n = 1`` 是熵与集中度的边界情形）."""
    from smart_research_agent.multi_head.layers import multi_head_attention as build

    return build(identity_parameters(4), ((1.0, 0.0, 0.0, 0.0),), heads=2, causal=True)


def test_one_token_forward_is_reproducible():
    """``n = 1`` 时每一头都只看自己：输出应当等于输入经过 W_o（这里是单位矩阵）."""
    forward = _forward_with_one_token()
    assert forward.distributions == 2
    assert forward.output == ((1.0, 0.0, 0.0, 0.0),)
    assert math.fsum(forward.head_weights[0][0]) == 1.0
