"""day068 三条提升指标与 ``measure_lift`` 的单元测试.

三条指标是"重排到底有没有用"这个问题唯一的答案来源，因此它们必须能被**手算**：
期望值一律写成手算常量（小数点后六位），而不是"跑一遍抄下来"。这里用两种独立手段钉住：

```text
1. 手算常量      十几组输入的名次、位置与分数都逐项写死在用例里（注释里有算术过程）
2. 独立实现      另写一份**直白的公式实现**（不用 seen 集合、不用滑窗，
                 而是"摊开前 k 条 + 逐项乘 1/log2(i+2)"），
                 在 900 多组枚举输入上与被测实现逐位比对
```

``measure_lift`` 那一段还要证明一件容易做错的事：**before 与 after 来自同一批命中**。
用一个会计数的检索器把它变成断言——``calls == 探针数``，而不是"探针数的两倍"。

全部离线、确定性、零网络。
"""

from __future__ import annotations

import math
from itertools import product
from typing import Any

import pytest

from smart_research_agent.retrieval.errors import RerankError
from smart_research_agent.retrieval.hybrid import HybridRetriever
from smart_research_agent.retrieval.rerank import (
    CrossEncoderReranker,
    LiftProbe,
    measure_lift,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from tests.hybrid_samples import (
    HYBRID_DEFAULT,
    QUERY_EXACT,
    hybrid_retriever,
)
from tests.rerank_samples import (
    HAND_LIFT,
    HAND_LIFT_AFTER,
    HAND_LIFT_BEFORE,
    HAND_PROBE_LIFT,
    HAND_PROBE_METRICS,
    HAND_RERANK_ORDER,
    HAND_STAGE1_ORDER,
    LIFT_K,
    METRIC_NAMES,
    PROBE_SPECS,
    RERANK_QUERY,
    CountingRetriever,
    lift_probes,
    probe_of,
)

#: ``1/log2(3)`` 与 ``1/log2(2) + 1/log2(3)``——nDCG 的两块手算砖.
ONE_OVER_LOG2_3 = 0.6309297535714574
IDCG_TWO = 1.6309297535714574


# --------------------------------------------------------------------------- #
# 独立实现（与 rerank 无关的第二份算术）
# --------------------------------------------------------------------------- #


def reference_recall_at_k(ids: list[str], relevant: list[str], k: int) -> float:
    """直白版 recall@k：先给金标准去重（保序），再数前 k 条里命中了几条.

    分母是**去重后**的金标准（标了两次同一条不会把分母抬高）；
    分子数的是**金标准里的成员**在前 k 条里出现了几个（同一条重复出现也只算一次）。
    """
    pool = list(dict.fromkeys(relevant))
    if not pool:
        return 0.0
    head = list(ids)[:k]
    hits = sum(1 for record_id in pool if record_id in head)
    return hits / len(pool)


def reference_reciprocal_rank(ids: list[str], relevant: list[str]) -> float:
    """直白版 RR：从头扫一遍名单，第一个命中给 ``1/(位置+1)``。"""
    pool = list(dict.fromkeys(relevant))
    ordered = list(ids)
    for position in range(len(ordered)):
        if ordered[position] in pool:
            return 1.0 / (position + 1)
    return 0.0


def reference_ndcg_at_k(ids: list[str], relevant: list[str], k: int) -> float:
    """直白版 nDCG@k：**不用 seen 集合**，而是用切片判"这条之前出现过没有".

    ```text
    gain_i = 1  若第 i 条相关、且它在前 i 条里没出现过（重复出现按 0 增益）
    DCG    = Σ gain_i / log2(i + 2)               i 从 0 起
    IDCG   = Σ 1 / log2(i + 2)，项数 = min(|金标准|, k)
    ```
    """
    pool = list(dict.fromkeys(relevant))
    head = list(ids)[:k]
    gains = [
        1.0 if (record_id in pool and record_id not in head[:index]) else 0.0
        for index, record_id in enumerate(head)
    ]
    dcg = sum(gain / math.log2(index + 2.0) for index, gain in enumerate(gains))
    ideal_gains = [1.0] * min(len(pool), k)
    idcg = sum(gain / math.log2(index + 2.0) for index, gain in enumerate(ideal_gains))
    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def enumerated_cases() -> list[tuple[tuple[str, ...], tuple[str, ...], int]]:
    """900 多组枚举输入：名单长度 0~3、金标准 8 种（含重复与全不相关）、k ∈ {1,2,4}."""
    alphabet = ("a", "b", "c")
    id_lists: list[tuple[str, ...]] = [()]
    id_lists += [tuple(combo) for size in (1, 2, 3) for combo in product(alphabet, repeat=size)]
    golds: list[tuple[str, ...]] = [
        ("a",),
        ("b",),
        ("c",),
        ("a", "b"),
        ("a", "a"),
        ("b", "c"),
        ("a", "b", "c"),
        ("d",),
    ]
    return [(ids, gold, k) for ids in id_lists for gold in golds for k in (1, 2, 4)]


# --------------------------------------------------------------------------- #
# recall@k
# --------------------------------------------------------------------------- #


class TestRecallAtK:
    """``|前 k 条 ∩ 金标准| / |金标准|``——它只看"捞回来没有"，对名次完全不敏感."""

    @pytest.mark.parametrize(
        ("ids", "relevant", "k", "expected"),
        [
            (["a", "b", "c"], ["a"], 3, 1.0),
            (["a", "b", "c"], ["b"], 1, 0.0),
            (["a", "b", "c"], ["b"], 2, 1.0),
            (["a", "b", "c"], ["b"], 3, 1.0),
            (["a", "b", "c"], ["a", "b", "d"], 3, 2 / 3),
            (["a", "b", "c"], ["d"], 3, 0.0),
            (["a", "b"], ["a", "b"], 2, 1.0),
            (["a", "b", "c", "d", "e"], ["e"], 5, 1.0),
            (["a", "b", "c", "d", "e"], ["e"], 4, 0.0),
            ([], ["a"], 3, 0.0),
        ],
    )
    def test_hand_computed_values(
        self, ids: list[str], relevant: list[str], k: int, expected: float
    ) -> None:
        """十组常规输入：分子分母都小到可以口算."""
        assert recall_at_k(ids, relevant, k) == pytest.approx(expected, rel=1e-12)

    def test_k_larger_than_the_list_uses_the_whole_list(self) -> None:
        """k 大于长度时取整份名单（不是报错，也不是补零）."""
        assert recall_at_k(["a", "b"], ["b"], 100) == 1.0

    def test_an_empty_gold_standard_gives_zero(self) -> None:
        """空金标准返回 0.0 而不是 0/0 崩溃——但它算出来的 0.0 看起来像"一条都没召回到"."""
        assert recall_at_k(["a", "b"], [], 2) == 0.0

    def test_duplicate_gold_ids_do_not_inflate_the_denominator(self) -> None:
        """标了两次同一条不会把分母抬到 2（否则一次命中只算 0.5）."""
        assert recall_at_k(["a"], ["a", "a"], 1) == 1.0
        assert recall_at_k(["a"], ["a", "a", "a"], 1) == 1.0

    def test_duplicate_ids_in_the_list_count_once(self) -> None:
        """名单里同一条出现两次只算一次命中（分子是"金标准里的成员出现几个"）."""
        assert recall_at_k(["a", "a", "b"], ["a"], 3) == 1.0
        assert recall_at_k(["a", "a"], ["a", "b"], 2) == 0.5

    def test_it_does_not_react_to_ranking_within_the_window(self) -> None:
        """**这条指标对名次不敏感**：只要那条相关命中在前 k 条里，排第几名都一样.
        """
        tail = ["x", "x", "x", "x", "a"]
        head = ["a", "x", "x", "x", "x"]

        assert recall_at_k(tail, ["a"], 5) == recall_at_k(head, ["a"], 5) == 1.0

    def test_boundary_k_equals_one(self) -> None:
        """``k=1``：只看第一名（它把"头部质量"单独切出来）."""
        assert recall_at_k(["a", "b"], ["a"], 1) == 1.0
        assert recall_at_k(["a", "b"], ["b"], 1) == 0.0


# --------------------------------------------------------------------------- #
# reciprocal_rank
# --------------------------------------------------------------------------- #


class TestReciprocalRank:
    """``1 / (第一个相关命中的位置 + 1)``——位置从 0 起，因此第一条就相关给 1.0."""

    @pytest.mark.parametrize(
        ("ids", "relevant", "expected"),
        [
            (["a"], ["a"], 1.0),
            (["b", "a"], ["a"], 1 / 2),
            (["x", "y", "z"], ["z"], 1 / 3),
            (["x"] * 4 + ["a"], ["a"], 1 / 5),
            (["x"] * 9 + ["a"], ["a"], 1 / 10),
            (["x", "y", "z"], ["w"], 0.0),
            ([], ["a"], 0.0),
            (["a", "a"], ["a"], 1.0),
            (["x", "c"], ["b", "c"], 1 / 2),
            (["x", "a"], ["a", "a"], 1 / 2),
        ],
    )
    def test_hand_computed_values(
        self, ids: list[str], relevant: list[str], expected: float
    ) -> None:
        """位置 i 对应 ``1/(i+1)``：与 ``fusion`` 里 RRF 的 ``1/(k+rank+1)`` 同一个形状."""
        assert reciprocal_rank(ids, relevant) == pytest.approx(expected, rel=1e-12)

    def test_the_first_relevant_hit_wins_even_if_a_later_one_is_stronger(self) -> None:
        """只看**第一个**相关命中（后面的相关条再多也不加分）."""
        assert reciprocal_rank(["x", "a", "a", "a"], ["a"]) == pytest.approx(0.5, rel=1e-12)

    def test_an_empty_gold_standard_gives_zero(self) -> None:
        """空金标准返回 0.0（没有"第一个相关命中"可言）."""
        assert reciprocal_rank(["a", "b"], []) == 0.0

    def test_a_perfect_run_gives_exactly_one(self) -> None:
        """前十名全相关时也只给 1.0（它衡量的是"第一名有多相关"）."""
        assert reciprocal_rank([f"g{index}" for index in range(10)],
                               [f"g{index}" for index in range(10)]) == 1.0

    def test_it_does_not_react_to_how_many_were_recalled(self) -> None:
        """**这条指标看不见"命中了几条"**：两条名单给出同一个 RR."""
        assert reciprocal_rank(["a", "x"], ["a", "b"]) == reciprocal_rank(["a", "b"], ["a", "b"])

    @pytest.mark.parametrize("position", list(range(10)))
    def test_every_position_matches_one_over_position_plus_one(self, position: int) -> None:
        """逐个位置核对 ``1/(位置+1)``（"从 0 起"这件事必须逐位对上）."""
        ids = ["x"] * position + ["a"]

        assert reciprocal_rank(ids, ["a"]) == pytest.approx(1.0 / (position + 1), rel=1e-12)


# --------------------------------------------------------------------------- #
# nDCG@k
# --------------------------------------------------------------------------- #


class TestNdcgAtK:
    """``DCG@k / IDCG@k``——唯一同时看"命中几条"与"排得多靠前"的那条."""

    @pytest.mark.parametrize(
        ("ids", "relevant", "k", "expected"),
        [
            (["a"], ["a"], 1, 1.0),
            (["b", "a"], ["a"], 2, ONE_OVER_LOG2_3),
            (["b", "a"], ["a"], 1, 0.0),
            (["a", "x"], ["a", "d"], 2, 1 / IDCG_TWO),
            (["a", "b"], ["a", "b"], 2, 1.0),
            (["x", "a"], ["a", "b"], 2, ONE_OVER_LOG2_3 / IDCG_TWO),
            (["a", "a", "b"], ["a", "b"], 3, 1.5 / IDCG_TWO),
            (["a", "a", "a"], ["a"], 3, 1.0),
            (["x", "y"], ["a"], 2, 0.0),
            (["x"], ["a", "b"], 1, 0.0),
            ([], ["a"], 3, 0.0),
            (["a", "b", "c"], ["a"], 5, 1.0),
            (["x", "a"], ["a"], 5, ONE_OVER_LOG2_3),
        ],
    )
    def test_hand_computed_values(
        self, ids: list[str], relevant: list[str], k: int, expected: float
    ) -> None:
        """十三组输入逐项对照手算值（折损用 ``log2(i+2)``，理想名单用 ``min(|金标准|, k)``）."""
        assert ndcg_at_k(ids, relevant, k) == pytest.approx(expected, rel=1e-12)

    def test_the_first_slot_has_no_discount(self) -> None:
        """第一条的折损是 ``log2(2) = 1``：命中它等于满分增益."""
        assert ndcg_at_k(["a"], ["a"], 1) == 1.0
        assert 1.0 / math.log2(2.0) == 1.0

    def test_a_repeated_id_never_exceeds_one(self) -> None:
        """重复 id 按 0 增益：一份"把同一条刷满"的名单不能拿到大于 1 的 nDCG."""
        assert ndcg_at_k(["a"] * 10, ["a"], 10) == 1.0

    def test_the_ideal_list_stops_at_k(self) -> None:
        """金标准比 k 多时，理想 DCG 也只数前 k 条（否则分母会凭空变大）."""
        assert ndcg_at_k(["a", "b", "c"], ["a", "b", "c", "d", "e"], 2) == 1.0

    def test_a_longer_ideal_list_needs_a_longer_perfect_run(self) -> None:
        """金标准 2 条时，只命中第一条拿不到 1.0（``1/log2(2) / (1+1/log2(3)) = 0.613147``）."""
        assert ndcg_at_k(["a", "x"], ["a", "b"], 2) == pytest.approx(0.613147, rel=1e-6)
        assert ndcg_at_k(["a", "b"], ["a", "b"], 2) == 1.0

    def test_it_sees_both_hits_and_positions(self) -> None:
        """同一个"命中 2 条"，排得靠前的那份 nDCG 更高（recall 看不见这个差别）.

        ```text
        前 3 条 = a, b, x    DCG = 1 + 1/log2(3) = 1.63093，IDCG = 1.63093 → 1.0
        前 3 条 = x, a, b    DCG = 1/log2(3) + 1/log2(4) = 1.13093 → 1.13093/1.63093 = 0.693426
        ```
        """
        front = ndcg_at_k(["a", "b", "x"], ["a", "b"], 3)
        back = ndcg_at_k(["x", "a", "b"], ["a", "b"], 3)

        assert recall_at_k(["a", "b", "x"], ["a", "b"], 3) == recall_at_k(
            ["x", "a", "b"], ["a", "b"], 3
        )
        assert front == 1.0
        assert back == pytest.approx(0.693426, rel=1e-6)
        assert front > back

    def test_an_empty_gold_standard_gives_zero(self) -> None:
        """空金标准返回 0.0（IDCG 的分母不存在）."""
        assert ndcg_at_k(["a", "b"], [], 2) == 0.0

    def test_more_matches_beat_fewer_matches_at_the_same_position(self) -> None:
        """第一条都命中时，命中两条的那份更高（说明它确实看"命中几条"）."""
        one = ndcg_at_k(["a", "x"], ["a", "b"], 2)
        both = ndcg_at_k(["a", "b"], ["a", "b"], 2)

        assert both > one


# --------------------------------------------------------------------------- #
# 交叉核对：独立实现
# --------------------------------------------------------------------------- #


class TestIndependentFormula:
    """在与被测实现无关的第二份算术上比对 900 多组枚举输入（照 ``test_hybrid_bm25`` 的做法）."""

    def test_recall_matches_the_reference_over_the_enumeration(self) -> None:
        """每一组输入上两边逐位相同（``rel=1e-12``）——一个口径写错就必然露出来."""
        for ids, gold, k in enumerated_cases():
            assert recall_at_k(list(ids), list(gold), k) == pytest.approx(
                reference_recall_at_k(list(ids), list(gold), k), rel=1e-12
            ), (ids, gold, k)

    def test_reciprocal_rank_matches_the_reference_over_the_enumeration(self) -> None:
        """RR 的"从 0 起"这件事最容易差一格，因此逐个位置都比一遍."""
        for ids, gold, _k in enumerated_cases():
            assert reciprocal_rank(list(ids), list(gold)) == pytest.approx(
                reference_reciprocal_rank(list(ids), list(gold)), rel=1e-12
            ), (ids, gold)

    def test_ndcg_matches_the_reference_over_the_enumeration(self) -> None:
        """nDCG 的重复增益与理想名单两条规则都要与直白版一致."""
        for ids, gold, k in enumerated_cases():
            assert ndcg_at_k(list(ids), list(gold), k) == pytest.approx(
                reference_ndcg_at_k(list(ids), list(gold), k), rel=1e-12
            ), (ids, gold, k)

    def test_every_metric_stays_within_the_unit_interval(self) -> None:
        """三条指标都落在 [0, 1]（均值"逐指标取平均"这件事因此是安全的）."""
        for ids, gold, k in enumerated_cases():
            for value in (
                recall_at_k(list(ids), list(gold), k),
                reciprocal_rank(list(ids), list(gold)),
                ndcg_at_k(list(ids), list(gold), k),
            ):

                assert 0.0 <= value <= 1.0, (ids, gold, k, value)

    def test_ndcg_is_order_sensitive_while_recall_is_not(self) -> None:
        """两份实现的共同结论：换次序时 recall 不动、nDCG 会动."""
        forward = ("a", "b", "x", "y")
        backward = ("x", "y", "b", "a")
        gold = ("a", "b")

        assert recall_at_k(list(forward), list(gold), 4) == recall_at_k(
            list(backward), list(gold), 4
        )
        assert ndcg_at_k(list(forward), list(gold), 4) > ndcg_at_k(
            list(backward), list(gold), 4
        )


class TestMetricInputValidation:
    """三条指标的入参校验（``k`` / 金标准 / 名单三处各一条出路）."""

    @pytest.mark.parametrize("k", [0, -1, 1.5, "3", None, True])
    def test_recall_rejects_bad_k(self, k: Any) -> None:
        """``k=0`` 不是一个"只看零条"的请求，而是让两条指标恒为 0."""
        with pytest.raises(RerankError) as excinfo:
            recall_at_k(["a"], ["a"], k)

        message = str(excinfo.value)
        assert "k 必须是 >= 1 的整数" in message
        assert "k=0 不是一个'只看零条'的请求" in message

    @pytest.mark.parametrize("k", [0, -1, 2.5])
    def test_ndcg_rejects_bad_k(self, k: Any) -> None:
        """nDCG 与 recall 共用同一个 ``k`` 校验（一份实现）."""
        with pytest.raises(RerankError) as excinfo:
            ndcg_at_k(["a"], ["a"], k)

        assert "recall@k 与 nDCG@k 的名字里带着它" in str(excinfo.value)

    def test_reciprocal_rank_takes_no_k(self) -> None:
        """RR 没有 ``k``（它的名字里就没有）——它只看第一个相关命中的位置."""
        assert reciprocal_rank(["x", "a"], ["a"]) == 0.5

    @pytest.mark.parametrize("relevant", ["a", b"a", 3, None])
    def test_rejects_a_non_sequence_gold_standard(self, relevant: Any) -> None:
        """一个字符串会被按字符拆开——因此必须写成 ``('c-a-01',)``（注意那个逗号）."""
        with pytest.raises(RerankError) as excinfo:
            recall_at_k(["a"], relevant, 1)

        message = str(excinfo.value)
        assert "的 relevant 必须是序列（列表/元组）" in message
        assert "一条金标准请写成 ('c-a-01',)（注意那个逗号）" in message

    @pytest.mark.parametrize("gold", [[""], ["  "], [None], ["a", 7]])
    def test_rejects_empty_ids_inside_the_gold_standard(self, gold: list) -> None:
        """空 id 永远匹配不上任何命中，而它会让分母变大、指标变小."""
        with pytest.raises(RerankError) as excinfo:
            reciprocal_rank(["a"], gold)

        assert "而它会让分母变大、指标变小——看起来像'检索变差了'" in str(excinfo.value)

    @pytest.mark.parametrize("ids", ["abc", b"abc", 3, None])
    def test_rejects_a_non_sequence_id_list(self, ids: Any) -> None:
        """声明方式：传 ``result.ids()`` 或 ``[hit.record_id for hit in reranked.hits]``."""
        with pytest.raises(RerankError) as excinfo:
            ndcg_at_k(ids, ["a"], 1)

        assert "出路：传 result.ids()（RetrievalResult 已经按名次排好了）" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# measure_lift
# --------------------------------------------------------------------------- #


class TestMeasureLift:
    """逐条探针量"重排前后"——before/after 只差一个变量（有没有重排）."""

    def test_before_and_after_come_from_one_retrieval_per_probe(self) -> None:
        """**只检索一次、不检索两次**：会计数的检索器上 ``calls`` 恰好等于探针数."""
        counter = CountingRetriever()
        report = measure_lift(lift_probes(), counter.retrieve, CrossEncoderReranker(), k=LIFT_K)

        assert counter.calls == len(PROBE_SPECS) == 5
        assert counter.queries == [RERANK_QUERY] * 5
        assert report.queries == 5

    def test_before_ids_are_the_stage_one_order(self) -> None:
        """before 用的是**重排之前**的 id 序列（它等于第一阶段的完整名单）."""
        report = self.run_report()

        for row in report.per_query:

            assert row["before_ids"] == list(HAND_STAGE1_ORDER)

    def test_after_ids_are_the_same_hits_reranked(self) -> None:
        """after 用的是**同一批 hits** 重排后的序列（因此它是一份手算常量）."""
        report = self.run_report()

        for row in report.per_query:

            assert row["after_ids"] == list(HAND_RERANK_ORDER)

    def test_per_probe_metrics_match_the_hand_computed_table(self) -> None:
        """逐条探针的六个数（before 三条 + after 三条）逐项对照手算表."""
        report = self.run_report()

        for (name, _gold), row in zip(PROBE_SPECS, report.per_query):
            before, after = HAND_PROBE_METRICS[name]

            assert [row["before"][metric] for metric in METRIC_NAMES] == pytest.approx(
                list(before), rel=1e-12
            )
            assert [row["after"][metric] for metric in METRIC_NAMES] == pytest.approx(
                list(after), rel=1e-12
            )

    def test_per_probe_lift_is_after_minus_before(self) -> None:
        """逐条的 ``lift`` 恰好等于 ``after - before``（逐位相减后 round 6）."""
        report = self.run_report()

        for (name, _gold), row in zip(PROBE_SPECS, report.per_query):
            expected = {
                metric: round(row["after"][metric] - row["before"][metric], 6)
                for metric in METRIC_NAMES
            }

            assert row["lift"] == expected
            assert [row["lift"][metric] for metric in METRIC_NAMES] == pytest.approx(
                list(HAND_PROBE_LIFT[name]), rel=1e-12
            )

    def test_the_report_means_are_hand_computed(self) -> None:
        """三份均值表逐键对照手算常量（五条探针 / k=5）."""
        report = self.run_report()

        assert report.before == HAND_LIFT_BEFORE
        assert report.after == HAND_LIFT_AFTER
        assert report.lift == HAND_LIFT

    def test_the_aggregate_lift_is_after_minus_before(self) -> None:
        """``lift`` 是**差值**而不是比值（比值在 before == 0 时是无穷）."""
        report = self.run_report()

        for metric in METRIC_NAMES:

            assert report.lift[metric] == round(
                report.after[metric] - report.before[metric], 6
            )

    def test_the_report_carries_k_and_the_metric_names(self) -> None:
        """``k`` 只写一次（三条指标共用它），而三张表的键就是那三个指标名."""
        report = self.run_report()

        assert report.k == LIFT_K
        assert set(report.before) == set(METRIC_NAMES)
        assert report.metrics == ("ndcg", "recall", "reciprocal_rank")

    def test_the_row_reports_the_window_account(self) -> None:
        """逐条明细里带着窗口的账（``candidates`` / ``scored`` / ``top_n`` / 移动了几条）."""
        report = self.run_report()

        for row in report.per_query:

            assert row["candidates"] == 10
            assert row["scored"] == 10
            assert row["top_n"] == 20
            assert row["mode"] == "replace"
            assert row["weight"] == 0.0
            assert row["moved"] == list(HAND_RERANK_ORDER)
            assert row["k"] == LIFT_K

    def test_the_row_key_set_is_exact(self) -> None:
        """逐条明细的键集合全等（下游按这些键取数）."""
        row = self.run_report().per_query[0]

        assert set(row) == {
            "query",
            "relevant",
            "k",
            "before_ids",
            "after_ids",
            "before",
            "after",
            "moved",
            "candidates",
            "scored",
            "top_n",
            "mode",
            "weight",
            "lift",
        }

    def test_the_report_projection_matches_the_constants(self) -> None:
        """``to_dict()`` 的三张表与 ``per_query`` 长度都要对得上."""
        report = self.run_report()
        payload = report.to_dict()

        assert payload["queries"] == 5
        assert payload["k"] == LIFT_K
        assert payload["metrics"] == ["ndcg", "recall", "reciprocal_rank"]
        assert payload["lift"] == HAND_LIFT
        assert len(payload["per_query"]) == 5
        assert payload["per_query"][0]["relevant"] == ["r-t-01"]

    def test_the_probes_are_not_mutated(self) -> None:
        """探针是**只读**输入：跑完之后它仍然是原样（否则第二次评估会不一样）."""
        probes = lift_probes()
        measure_lift(probes, CountingRetriever().retrieve, CrossEncoderReranker(), k=LIFT_K)

        assert [probe.relevant for probe in probes] == [
            relevant for _name, relevant in PROBE_SPECS
        ]
        assert {probe.query for probe in probes} == {RERANK_QUERY}

    def test_a_different_k_changes_the_two_k_dependent_metrics_only(self) -> None:
        """``k=2`` 的探针 p-b：recall@2 从 0 变 1，而 RR 与 k 无关.

        ```text
        before（r-t-08 在第 7 位）  recall@2 0.0   RR 1/8 = 0.125   nDCG@2 0.0
        after （它在第 0 位）        recall@2 1.0   RR 1.0           nDCG@2 1.0
        ```
        """
        probe = probe_of("p-b")
        report = measure_lift([probe], CountingRetriever().retrieve, CrossEncoderReranker(), k=2)

        assert report.k == 2
        assert report.before == {"recall": 0.0, "reciprocal_rank": 0.125, "ndcg": 0.0}
        assert report.after == {"recall": 1.0, "reciprocal_rank": 1.0, "ndcg": 1.0}
        assert report.lift == {"recall": 1.0, "reciprocal_rank": 0.875, "ndcg": 1.0}

    def test_no_probes_gives_a_zero_report_and_no_retrieval(self) -> None:
        """空探针列表返回一份"三条指标全 0"的报告（但 ``queries=0`` 要如实写在那里）."""
        counter = CountingRetriever()
        report = measure_lift([], counter.retrieve, CrossEncoderReranker(), k=LIFT_K)

        assert report.queries == 0
        assert report.before == {"recall": 0.0, "reciprocal_rank": 0.0, "ndcg": 0.0}
        assert report.after == {"recall": 0.0, "reciprocal_rank": 0.0, "ndcg": 0.0}
        assert report.lift == {"recall": 0.0, "reciprocal_rank": 0.0, "ndcg": 0.0}
        assert report.per_query == ()
        assert counter.calls == 0

    def test_a_top_n_overrides_the_window(self) -> None:
        """``top_n`` 透传进 ``rerank_hits``：``top_n=3`` 时逐条明细里的账就是 3."""
        report = measure_lift(
            lift_probes(), CountingRetriever().retrieve, CrossEncoderReranker(), k=LIFT_K, top_n=3
        )

        assert report.per_query[0]["scored"] == 3
        assert report.per_query[0]["top_n"] == 3

    def run_report(self) -> Any:
        """五条探针 / k=5 的标准报告（大多数用例的起点）."""
        return measure_lift(
            lift_probes(), CountingRetriever().retrieve, CrossEncoderReranker(), k=LIFT_K
        )


class TestMeasureLiftValidation:
    """``measure_lift`` 的四条入参校验（每条都要给出一个可照做的写法）."""

    @pytest.mark.parametrize("probes", ["probe", b"probe", 3, None, {"p": 1}])
    def test_rejects_non_sequence_probes(self, probes: Any) -> None:
        """探针列表必须是序列（一个 ``LiftProbe`` 写成单元素列表要注意那个逗号）."""
        with pytest.raises(RerankError) as excinfo:
            measure_lift(probes, CountingRetriever().retrieve)

        message = str(excinfo.value)
        assert "measure_lift 需要一批 LiftProbe（序列）" in message
        assert (
            "写法：measure_lift([LiftProbe(query='…', relevant=('c-a-01',))], retriever)"
            in message
        )

    @pytest.mark.parametrize("retrieve", [None, 3, "retriever", ["retrieve"]])
    def test_rejects_a_non_callable_retrieve(self, retrieve: Any) -> None:
        """要的是"一句话 → 一份结果"的可调用对象，不是检索器本身."""
        with pytest.raises(RerankError) as excinfo:
            measure_lift(lift_probes(), retrieve)

        assert "出路：传 retriever.retrieve 或 hybrid.retrieve（不要传检索器本身）" in str(
            excinfo.value
        )

    def test_rejects_a_non_probe_item(self) -> None:
        """第几条不是 ``LiftProbe`` 要点名（一个字典不行：它没有 ``.query`` 的形状约定）."""
        with pytest.raises(RerankError) as excinfo:
            measure_lift([probe_of("p-a"), {"query": "x"}], CountingRetriever().retrieve)

        message = str(excinfo.value)
        assert "第 1 条探针不是 LiftProbe，收到 dict" in message
        assert "写法：LiftProbe(query='阈值怎么设', relevant=('c-a-01',))" in message

    @pytest.mark.parametrize("k", [0, -1, 2.5, "5"])
    def test_rejects_bad_k(self, k: Any) -> None:
        """``k`` 在三条指标与报告之间共用一份校验."""
        with pytest.raises(RerankError) as excinfo:
            measure_lift(lift_probes(), CountingRetriever().retrieve, CrossEncoderReranker(), k=k)

        assert "k 必须是 >= 1 的整数" in str(excinfo.value)

    def test_rejects_a_retrieve_that_returns_a_plain_dict(self) -> None:
        """``retrieve`` 必须返回 ``RetrievalResult``（要有 ``.hits`` 与 ``.ids()``）."""
        with pytest.raises(RerankError) as excinfo:
            measure_lift([probe_of("p-a")], lambda query: {"hits": [], "ids": []})

        message = str(excinfo.value)
        assert "而本层要的是一份 RetrievalResult（要有 .hits 与 .ids()）" in message
        assert "出路：传 retriever.retrieve / hybrid.retrieve" in message

    def test_an_invalid_reranker_is_rejected(self) -> None:
        """``reranker`` 那一侧的校验与 ``rerank_hits`` 共用一份实现."""
        with pytest.raises(RerankError) as excinfo:
            measure_lift(lift_probes(), CountingRetriever().retrieve, "bge-reranker")

        assert "reranker 必须是 BaseReranker" in str(excinfo.value)


class TestLiftContrast:
    """一组"重排确实提升"与一组"重排没有提升甚至变差"的**对照**（符号与数值都钉住）."""

    def test_a_probe_that_gets_pushed_up_improves_on_all_three_metrics(self) -> None:
        """p-b（金标准 = 被顶到第 0 名的那条）：三条指标全为正，且数值手算得出.

        ```text
        recall@5  0.0 → 1.0（从"前 5 条里没有"变成"第 1 条就是它"）
        RR        1/8 → 1.0
        nDCG@5    0.0 → 1.0
        ```
        """
        report = measure_lift([probe_of("p-b")], CountingRetriever().retrieve,
                              CrossEncoderReranker(), k=LIFT_K)

        assert report.lift == {"recall": 1.0, "reciprocal_rank": 0.875, "ndcg": 1.0}
        assert all(value > 0.0 for value in report.lift.values())

    def test_a_probe_that_was_already_first_gets_worse(self) -> None:
        """p-a（金标准 = 第一阶段第一名）：重排把它压到第 5 位，三条指标全为负.

        ```text
        recall@5  1.0 → 0.0    RR 1.0 → 1/6 = 0.166667    nDCG@5 1.0 → 0.0
        ```
        """
        report = measure_lift([probe_of("p-a")], CountingRetriever().retrieve,
                              CrossEncoderReranker(), k=LIFT_K)

        assert report.lift == {"recall": -1.0, "reciprocal_rank": -0.833333, "ndcg": -1.0}
        assert all(value < 0.0 for value in report.lift.values())

    def test_a_probe_that_stays_deep_gets_slightly_worse(self) -> None:
        """p-d（金标准 = 长文档）：两个名单里都在第 9 名之后，只有 RR 掉了 1/90."""
        report = measure_lift([probe_of("p-d")], CountingRetriever().retrieve,
                              CrossEncoderReranker(), k=LIFT_K)

        assert report.lift == {"recall": 0.0, "reciprocal_rank": -0.011111, "ndcg": 0.0}
        assert report.before["reciprocal_rank"] == pytest.approx(1 / 9, abs=1e-6)
        assert report.after["reciprocal_rank"] == 0.1

    def test_two_gold_ids_show_recall_and_ranking_together(self) -> None:
        """p-e（两条金标准）：一条升、一条降，于是 recall 与 nDCG 的涨幅不同.

        ```text
        before  recall 1/2 = 0.5   RR 1/2 = 0.5        nDCG 0.63093/1.63093 = 0.386853
        after   recall 2/2 = 1.0   RR 1/1 = 1.0        nDCG 1.5/1.63093     = 0.919721
        ```
        """
        report = measure_lift([probe_of("p-e")], CountingRetriever().retrieve,
                              CrossEncoderReranker(), k=LIFT_K)

        assert report.before == {"recall": 0.5, "reciprocal_rank": 0.5, "ndcg": 0.386853}
        assert report.after == {"recall": 1.0, "reciprocal_rank": 1.0, "ndcg": 0.919721}
        assert report.lift == {"recall": 0.5, "reciprocal_rank": 0.5, "ndcg": 0.532868}
        assert report.lift["ndcg"] > report.lift["recall"]

    def test_the_mixed_report_averages_the_wins_and_the_losses(self) -> None:
        """五条探针合起来：recall 净提升 0.3，而 RR 只提升 0.172777（有一条在拖后腿）."""
        report = measure_lift(lift_probes(), CountingRetriever().retrieve,
                              CrossEncoderReranker(), k=LIFT_K)

        assert report.lift["recall"] == 0.3
        assert report.lift["reciprocal_rank"] == 0.172777
        assert report.lift["ndcg"] == 0.232759
        assert report.lift["recall"] > report.lift["ndcg"] > 0.0
        # 五条里有两条的 recall 提升不为正 → 净提升 0.3 而不是 5/5。
        non_positive = [row for row in report.per_query if row["lift"]["recall"] <= 0.0]

        assert [row["query"] for row in non_positive] == [RERANK_QUERY, RERANK_QUERY]
        assert len(non_positive) == 2

    def test_a_query_no_record_contains_leaves_only_the_length_penalty(self) -> None:
        """换一句语料里一个词都没有的查询时，四个特征只剩长度罚项还在起作用.

        ```text
        覆盖率 0、整句 0、邻近度 0      → 短文档（九条）的分数都是 0.125
        长度罚项（`240/300 = 0.8`）     → 长文档 r-t-09 的分数是 0.1
        ```
        于是**唯一**的次序变化是把那条长文档压到最后，而金标准（第一名 r-t-01）
        纹丝不动 —— 三条指标的 lift 因此都是 0（重排没能改动前五条里的任何一条）。
        这同时说明"重排分是对 (查询, 正文) 打的"：换了查询文本，重排就换了看法。
        """
        probe = LiftProbe(query="cache", relevant=("r-t-01",))
        report = measure_lift(
            [probe], CountingRetriever().retrieve, CrossEncoderReranker(), k=LIFT_K
        )
        row = report.per_query[0]

        assert row["before_ids"] == list(HAND_STAGE1_ORDER)
        assert row["after_ids"][-2:] == ["r-t-10", "r-t-09"]
        assert row["after_ids"][:5] == row["before_ids"][:5]
        assert report.lift == {"recall": 0.0, "reciprocal_rank": 0.0, "ndcg": 0.0}
        assert report.before == report.after == {
            "recall": 1.0,
            "reciprocal_rank": 1.0,
            "ndcg": 1.0,
        }

    def test_a_custom_reranker_can_make_things_much_worse(self) -> None:
        """换一个"把最不相关的那条顶上来"的重排器：提升指标必须如实变成负数.

        用不上任何模型：只要一个 `score_pairs` 把次序倒过来。这是"这三条指标真的
        在衡量重排的效果"这条主张最直接的反证。
        """
        from smart_research_agent.retrieval.rerank import BaseReranker

        class Reversed(BaseReranker):
            @property
            def name(self) -> str:
                return "reversed"

            @property
            def dimension(self) -> int:
                return 1

            def score_pairs(self, query: str, texts: Any) -> list[float]:
                return [float(index + 1) for index in range(len(list(texts)))]

            def describe(self) -> dict[str, Any]:
                return {"name": "reversed"}

        report = measure_lift(
            [probe_of("p-a")], CountingRetriever().retrieve, Reversed(), k=LIFT_K
        )

        assert report.per_query[0]["after_ids"] == list(reversed(HAND_STAGE1_ORDER))
        assert report.lift["recall"] == -1.0
        assert report.lift["reciprocal_rank"] == -0.9
        assert report.lift["ndcg"] == -1.0


class TestMeasureLiftOnHybridSamples:
    """跨样本模块的一条：``hybrid_samples`` 的混合检索器 + 手算过的那份名次."""

    def test_the_before_list_is_the_hand_computed_hybrid_default(self) -> None:
        """``before_ids`` 逐项等于 ``HYBRID_DEFAULT`` 里手算出来的那份名单."""
        probe = LiftProbe(query=QUERY_EXACT, relevant=("h-t-01",))
        report = measure_lift([probe], hybrid_retriever().retrieve, CrossEncoderReranker(), k=5)

        assert report.per_query[0]["before_ids"] == list(HYBRID_DEFAULT[QUERY_EXACT])
        assert report.queries == 1

    def test_a_probe_whose_gold_is_already_first_has_nothing_to_gain(self) -> None:
        """金标准就是第一名时：三条指标前后都是满分，lift 全 0（重排无从提升）."""
        probe = LiftProbe(query=QUERY_EXACT, relevant=("h-t-01",))
        report = measure_lift([probe], hybrid_retriever().retrieve, CrossEncoderReranker(), k=5)

        assert report.before == {"recall": 1.0, "reciprocal_rank": 1.0, "ndcg": 1.0}
        assert report.after == report.before
        assert report.lift == {"recall": 0.0, "reciprocal_rank": 0.0, "ndcg": 0.0}

    def test_the_same_hits_are_reranked_not_re_retrieved(self) -> None:
        """``after_ids`` 是 before 那份名单的**一份排列**（两次检索会混进随机性）."""
        probe = LiftProbe(query=QUERY_EXACT, relevant=("h-t-01",))
        retriever = hybrid_retriever()
        report = measure_lift([probe], retriever.retrieve, CrossEncoderReranker(), k=5)
        row = report.per_query[0]

        assert sorted(row["after_ids"]) == sorted(row["before_ids"])
        assert len(row["after_ids"]) == len(row["before_ids"]) == 5
        assert row["after_ids"] == row["before_ids"]

    def test_the_hybrid_retriever_is_read_only_for_the_report(self) -> None:
        """报告不改写检索器的配置：两次跑给出同一份结果（同一条查询两处可比）."""
        retriever = hybrid_retriever()
        probe = LiftProbe(query=QUERY_EXACT, relevant=("h-c-01",))
        first = measure_lift([probe], retriever.retrieve, CrossEncoderReranker(), k=5)
        second = measure_lift([probe], retriever.retrieve, CrossEncoderReranker(), k=5)

        assert first.before == second.before
        assert first.after == second.after
        assert isinstance(retriever, HybridRetriever)
        assert first.per_query[0]["candidates"] == second.per_query[0]["candidates"]
