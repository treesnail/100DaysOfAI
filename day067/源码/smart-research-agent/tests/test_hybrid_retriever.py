"""day067 ``retrieval.hybrid`` 的单元测试：十三步流水线与它的六处刻意取舍.

混合检索的价值不在"多了一路"，而在**两路的对错不是同一种对错**，且这件事
在报告里可核对。因此这里按流水线的关键分叉分组，每一组都同时断言
"结果是什么"与"证据记在哪"：

```text
规范化与深度   两路共用同一个 fetch_k（深度只有一份实现）
过滤           同一份 where 在**两路都落刀**（否则会把已排除的东西抬回来，且不报错）
阈值           只作用于向量通道，而且**无条件进 notes**（被忽略的参数必须被说出来）
融合           每条命中的 channel / channels / contributions 都要对得上
多样性与截断   在融合之后做（次序是融合的权威次序，只重编号）
诊断            channel_candidates / fusion / candidates / empty_reason / dropped_*
```

期望名次用的是 ``tests/hybrid_samples.py`` 里手算出来的常量
（两路互补的主案例见那份样本的模块 docstring）。

全部离线、确定性、零网络。
"""

from __future__ import annotations

from typing import Any

import pytest

from smart_research_agent.config import settings
from smart_research_agent.retrieval.errors import (
    FusionError,
    IndexStateError,
    QueryError,
)
from smart_research_agent.retrieval.hybrid import (
    DEFAULT_HYBRID_NAME,
    HybridRetriever,
    build_hybrid_retriever,
    hybrid_enabled,
)
from smart_research_agent.retrieval.lexical import BM25Params, LexicalIndex
from smart_research_agent.retrieval.retriever import Retriever
from smart_research_agent.retrieval.types import (
    CHANNEL_BM25,
    CHANNEL_VECTOR,
    RetrievalQuery,
)
from tests.hybrid_samples import (
    HYBRID_DEFAULT,
    HYBRID_SHALLOW,
    QUERY_BOTH,
    QUERY_CODE,
    QUERY_EXACT,
    QUERY_SEMANTIC,
    RECORD_IDS,
    VECTOR_ONLY_SHALLOW,
    documents,
    empty_store,
    hybrid_embedding,
    hybrid_lexical,
    hybrid_retriever,
    hybrid_store,
    hybrid_vector_retriever,
    make_record,
)
from tests.retrieval_samples import sample_manifest

#: 全部四个查询（逐个断言名次的用例用它做参数化）.
ALL_QUERIES: tuple[str, ...] = (QUERY_EXACT, QUERY_CODE, QUERY_SEMANTIC, QUERY_BOTH)


def evidence_of(result: Any) -> list[dict[str, Any]]:
    """命中里的"多路证据"三件套（逐条核对时用）."""
    return [
        {
            "record_id": hit.record_id,
            "channel": hit.channel,
            "channels": hit.channels,
        }
        for hit in result.hits
    ]


# --------------------------------------------------------------------------- #
# 两路互补（本课的主案例）
# --------------------------------------------------------------------------- #


class TestComplementaryChannels:
    """**主案例**：词面精确的那条在向量路里排得很后，在混合结果里排第一. """

    @pytest.mark.parametrize("query", ALL_QUERIES)
    def test_default_depth_matches_the_hand_computed_order(self, query: str) -> None:
        result = hybrid_retriever().retrieve(query)

        assert result.ids() == list(HYBRID_DEFAULT[query])

    @pytest.mark.parametrize("query", ALL_QUERIES)
    def test_shallow_depth_matches_the_hand_computed_order(self, query: str) -> None:
        """``top_k=3, fetch_k=3``：两路各自只取 3 条，互补关系因此最清楚. """
        result = hybrid_retriever().retrieve(RetrievalQuery(text=query, top_k=3, fetch_k=3))

        assert result.ids() == list(HYBRID_SHALLOW[query])

    def test_exact_match_record_comes_first_only_in_hybrid(self) -> None:
        """``ERR-2043``：单路向量检索的第一名**不含这个编号**，混合检索的第一名含.

        向量路的 top-3 是 ``h-c-01/h-c-02/h-c-03``（三条分数都是 0.0，
        次序完全由 id 决定）——**一条都不含 "ERR-2043"**；而含它的 ``h-t-01``
        在向量路里排第 8（0 分并列里 id 靠后）。关键词路把唯一含它的一条
        排到了名次 0，融合之后它成了第一名。
        """
        store = hybrid_store()
        embedding = hybrid_embedding()
        vector_only = hybrid_vector_retriever(store, embedding=embedding).retrieve(
            RetrievalQuery(text=QUERY_EXACT, top_k=3, fetch_k=3)
        )
        hybrid = HybridRetriever(
            hybrid_vector_retriever(store, embedding=embedding), hybrid_lexical(store)
        ).retrieve(QUERY_EXACT)

        assert "h-t-01" not in vector_only.ids()
        assert any("ERR-2043" in hit.text for hit in vector_only.hits) is False
        assert hybrid.ids()[0] == "h-t-01"
        assert "ERR-2043" in hybrid.hits[0].text

    def test_error_code_record_comes_first_only_in_hybrid(self) -> None:
        """``401``：同一条故事线（关键词路擅长编号，向量路对它没有方向）. """
        store = hybrid_store()
        embedding = hybrid_embedding()
        vector_only = hybrid_vector_retriever(store, embedding=embedding).retrieve(
            RetrievalQuery(text=QUERY_CODE, top_k=3, fetch_k=3)
        )
        hybrid = HybridRetriever(
            hybrid_vector_retriever(store, embedding=embedding), hybrid_lexical(store)
        ).retrieve(QUERY_CODE)

        assert "h-t-03" not in vector_only.ids()
        assert hybrid.ids()[0] == "h-t-03"
        assert "401" in hybrid.hits[0].text

    def test_semantic_rewrite_is_served_by_the_vector_channel_alone(self) -> None:
        """反方向：纯语义改写时关键词路一条都没有，向量路独立支撑整个结果. """
        result = hybrid_retriever().retrieve(QUERY_SEMANTIC)

        assert result.channel_candidates[CHANNEL_BM25] == 0
        assert result.channel_candidates[CHANNEL_VECTOR] == len(RECORD_IDS)
        assert all(hit.channels == (CHANNEL_VECTOR,) for hit in result.hits)
        assert result.candidates == len(RECORD_IDS)

    @pytest.mark.parametrize("query", [QUERY_EXACT, QUERY_CODE])
    def test_word_matching_queries_have_a_keyword_contribution(self, query: str) -> None:
        result = hybrid_retriever().retrieve(query)

        assert result.channel_candidates[CHANNEL_BM25] == 1
        assert result.fusion["contributions"][CHANNEL_BM25] == 1

    def test_shared_record_holds_both_channels(self) -> None:
        """两路都砸中 ``h-t-02``：它必须同时留下两条通道证据（最强的证据形状）. """
        result = hybrid_retriever().retrieve(QUERY_BOTH)

        assert result.hits[0].record_id == "h-t-02"
        assert result.hits[0].channels == (CHANNEL_VECTOR, CHANNEL_BM25)
        assert result.fusion["deduped"] == 1

    def test_shallow_tie_is_broken_by_record_id(self) -> None:
        """``fetch_k=3`` 时 ``h-c-01`` 与 ``h-t-01`` **精确同分**（都是 1/61）.

        于是次序由第三键 ``record_id`` 决定——这不是"向量路赢了"，
        而是"分数并列时必须有确定的第三键"（与 day066 的 ``(-score, record_id)`` 同源）。
        """
        result = hybrid_retriever().retrieve(
            RetrievalQuery(text=QUERY_EXACT, top_k=3, fetch_k=3)
        )
        first, second = result.hits[0], result.hits[1]

        assert first.score == second.score == pytest.approx(1.0 / 61.0, rel=1e-12)
        assert first.record_id < second.record_id
        assert first.channels == (CHANNEL_VECTOR,)
        assert second.channels == (CHANNEL_BM25,)

    def test_vector_only_hits_are_the_same_as_a_single_path_retriever(self) -> None:
        """混合检索的**向量那一半**与 day066 的单路结果一致（同一份流水线、同一份深度）."""
        store = hybrid_store()
        embedding = hybrid_embedding()
        single = hybrid_vector_retriever(store, embedding=embedding).retrieve(
            RetrievalQuery(text=QUERY_EXACT, top_k=3, fetch_k=3)
        )
        hybrid = HybridRetriever(
            hybrid_vector_retriever(store, embedding=embedding), hybrid_lexical(store)
        ).retrieve(RetrievalQuery(text=QUERY_EXACT, top_k=3, fetch_k=3))

        assert hybrid.channel_candidates[CHANNEL_VECTOR] == single.count


# --------------------------------------------------------------------------- #
# 深度、过滤与两路的公共口径
# --------------------------------------------------------------------------- #


class TestSharedDepthAndFilter:
    """两路共用同一个深度与同一份 ``where``：这两件事是"融合才有意义"的前提. """

    def test_fetch_k_is_the_same_number_for_both_channels(self) -> None:
        """深度由向量路那**一份**实现算出（过取倍率与封顶都算过），两路都用它. """
        result = hybrid_retriever().retrieve(QUERY_EXACT)

        assert result.fetch_k == 15
        assert result.channel_candidates[CHANNEL_VECTOR] == len(RECORD_IDS)
        assert result.channel_candidates[CHANNEL_BM25] == 1

    def test_short_depth_limits_both_channels(self) -> None:
        result = hybrid_retriever().retrieve(
            RetrievalQuery(text=QUERY_EXACT, top_k=3, fetch_k=3)
        )

        assert result.fetch_k == 3
        assert result.channel_candidates[CHANNEL_VECTOR] == 3
        assert result.channel_candidates[CHANNEL_BM25] == 1

    def test_the_where_clause_is_applied_to_both_channels(self) -> None:
        """把过滤条件收紧到 ``doc-concept``：**两路**都被筛过.

        如果只有向量路被筛，关键词路会把 ``doc-trouble`` 里的 ``h-t-01``
        重新抬回结果里——而报告里 ``filter_applied=True``、没有任何异常。
        """
        result = hybrid_retriever().retrieve(
            RetrievalQuery(text=QUERY_EXACT, where={"parent_doc_id": "doc-concept"})
        )

        assert result.filter_applied is True
        assert result.channel_candidates[CHANNEL_BM25] == 0
        assert result.channel_candidates[CHANNEL_VECTOR] == 4
        assert "h-t-01" not in result.ids()
        for hit in result.hits:
            assert hit.metadata["parent_doc_id"] == "doc-concept"

    def test_a_filter_that_kills_both_channels_reports_filtered_out(self) -> None:
        result = hybrid_retriever().retrieve(
            RetrievalQuery(text=QUERY_EXACT, where={"strategy": "nope"})
        )

        assert result.hits == ()
        assert result.empty_reason == "filtered_out"
        assert result.channel_candidates == {CHANNEL_VECTOR: 0, CHANNEL_BM25: 0}
        assert result.filter_applied is True

    def test_the_union_is_computed_across_the_two_channels(self) -> None:
        """``candidates`` 是**并集**：两路之和减去被两路同时命中的条数. """
        result = hybrid_retriever().retrieve(QUERY_BOTH)

        assert result.candidates == len(RECORD_IDS)
        assert result.fusion["recall"] == {
            CHANNEL_BM25: 1,
            CHANNEL_VECTOR: len(RECORD_IDS),
        }
        assert result.fusion["deduped"] == 1

    def test_a_time_range_excludes_the_record_without_a_timestamp_in_both_channels(self) -> None:
        result = hybrid_retriever().retrieve(
            RetrievalQuery(
                text=QUERY_CODE, time_range=None, where={"created_at": {"$gte": "2026-01-01"}}
            )
        )

        assert result.channel_candidates[CHANNEL_BM25] == 0
        assert "h-t-03" not in result.ids()
        assert result.candidates == len(RECORD_IDS) - 1

    @pytest.mark.parametrize("query", ALL_QUERIES)
    def test_filter_applied_is_reported_honestly(self, query: str) -> None:
        unfiltered = hybrid_retriever().retrieve(query)
        filtered = hybrid_retriever().retrieve(
            RetrievalQuery(text=query, where={"topic": {"$in": ["errors", "cache"]}})
        )

        assert unfiltered.filter_applied is False
        assert filtered.filter_applied is True

    def test_where_and_time_range_conflict_is_reported(self) -> None:
        """``where`` 与 ``time_range`` 撞在同一个字段上时，**两路都不会开始工作**. """
        from smart_research_agent.retrieval.types import TimeRange

        with pytest.raises(QueryError) as excinfo:
            hybrid_retriever().retrieve(
                RetrievalQuery(
                    text=QUERY_EXACT,
                    where={"created_at": {"$gte": "2026-01-01"}},
                    time_range=TimeRange(start="2026-06-01"),
                )
            )

        assert "过滤条件冲突" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# 阈值：只作用于向量通道，而且必须进 notes
# --------------------------------------------------------------------------- #


class TestThresholdScope:
    """``min_score`` 只对向量通道落刀，且被忽略这件事**无条件进 notes**. """

    def test_threshold_only_trims_the_vector_channel(self) -> None:
        """``min_score=1.0`` 把向量路切到只剩 1 条，关键词路一条没被切. """
        result = hybrid_retriever().retrieve(
            RetrievalQuery(text=QUERY_EXACT, min_score=1.0)
        )

        assert result.dropped_below_threshold == 9
        assert result.channel_candidates[CHANNEL_VECTOR] == 1
        # h-t-01 只有关键词路召回它（向量路那 9 条被切掉了）——它照样进结果。
        assert result.ids() == ["h-c-01", "h-t-01"]

    def test_the_keyword_channel_survives_a_threshold_that_empties_the_vector_channel(self) -> None:
        """阈值把向量路清空时，关键词路仍然交出它那一条（这正是本节要展示的行为）. """
        result = hybrid_retriever().retrieve(
            RetrievalQuery(text=QUERY_EXACT, min_score=1.5)
        )

        assert result.dropped_below_threshold == len(RECORD_IDS)
        assert result.channel_candidates == {CHANNEL_VECTOR: 0, CHANNEL_BM25: 1}
        assert result.ids() == ["h-t-01"]
        assert result.hits[0].channels == (CHANNEL_BM25,)

    def test_the_note_is_written_even_when_nothing_was_dropped(self) -> None:
        """**没有切掉任何一条时也要说**：一个被忽略的参数必须被说出来. """
        result = hybrid_retriever().retrieve(
            RetrievalQuery(text=QUERY_SEMANTIC, min_score=-1.0)
        )

        assert result.dropped_below_threshold == 0
        note = " ".join(result.notes)
        assert "只作用于向量通道" in note
        assert "假的安全感" in note
        assert "关键词通道不设阈值" in note

    def test_no_note_when_no_threshold_was_given(self) -> None:
        result = hybrid_retriever().retrieve(QUERY_EXACT)

        assert all("min_score" not in note for note in result.notes)

    def test_threshold_at_the_boundary_keeps_the_hit(self) -> None:
        """用 ``>=`` 而不是 ``>``：阈值**恰好等于**某条命中的分数时它必须留下.

        ``h-c-01`` 的余弦分数恰好是 1.0（查询向量与它同向），因此
        ``min_score=1.0`` 时它必须留下——一个 ``>`` 会让它消失，
        而那种差别在"阈值恰好等于分数"时**必然出现**（确定性编码器）。
        """
        result = hybrid_retriever().retrieve(
            RetrievalQuery(text=QUERY_EXACT, min_score=1.0)
        )

        assert "h-c-01" in result.ids()
        assert result.dropped_below_threshold == 9

    def test_threshold_is_applied_before_fusion_not_after(self) -> None:
        """融合后的分数**没有绝对标度**，因此阈值不可能在融合之后落刀.

        证据：融合分数都远小于 1.0，而 ``min_score=1.0`` 仍然切掉了 9 条——
        被切的只能是向量路的**原始**分数（余弦），不可能是融合之后的 1/61。
        """
        result = hybrid_retriever().retrieve(
            RetrievalQuery(text=QUERY_EXACT, min_score=1.0)
        )

        assert result.dropped_below_threshold == 9
        assert all(hit.score < 1.0 for hit in result.hits)


# --------------------------------------------------------------------------- #
# 融合证据的归属
# --------------------------------------------------------------------------- #


class TestChannelAttribution:
    """``channel`` / ``channels`` / ``contributions`` 三者的关系必须自洽. """

    @pytest.mark.parametrize("query", ALL_QUERIES)
    def test_channel_is_the_first_of_channels(self, query: str) -> None:
        """``channels[0] == channel`` 必须永远成立（它们是同一个问题的两个问法）. """
        result = hybrid_retriever().retrieve(query)

        for hit in result.hits:
            assert hit.channels
            assert hit.channels[0] == hit.channel

    @pytest.mark.parametrize("query", ALL_QUERIES)
    def test_channels_have_no_duplicates(self, query: str) -> None:
        for hit in hybrid_retriever().retrieve(query).hits:
            assert len(set(hit.channels)) == len(hit.channels)

    def test_exact_match_hit_is_attributed_to_the_keyword_channel(self) -> None:
        """``h-t-01`` 的词面证据更强（关键词 rank 0 vs 向量 rank 7）→ 归属关键词路. """
        result = hybrid_retriever().retrieve(QUERY_EXACT)
        hit = next(item for item in result.hits if item.record_id == "h-t-01")

        assert hit.channel == CHANNEL_BM25
        assert hit.channels == (CHANNEL_BM25, CHANNEL_VECTOR)

    def test_shared_hit_is_attributed_to_the_vector_channel_by_tie_rule(self) -> None:
        """两路贡献并列时**向量优先**（与 ``fusion._ordered_channels`` 同一条规则）. """
        result = hybrid_retriever().retrieve(QUERY_BOTH)
        hit = result.hits[0]

        assert hit.record_id == "h-t-02"
        assert hit.channel == CHANNEL_VECTOR
        assert hit.channels == (CHANNEL_VECTOR, CHANNEL_BM25)

    @pytest.mark.parametrize("query", ALL_QUERIES)
    def test_rrf_evidence_is_aggregated_in_the_summary(self, query: str) -> None:
        """RRF 下每路的贡献条数必须等于"召回条数"（它恒正，因此不会 0 贡献）. """
        result = hybrid_retriever(strategy="rrf").retrieve(query)

        assert result.hits
        assert all(hit.score > 0 for hit in result.hits)
        assert all(hit.channels for hit in result.hits)
        for channel, recalled in result.fusion["recall"].items():
            assert result.fusion["contributions"].get(channel, 0) == recalled

    @pytest.mark.parametrize("query", ALL_QUERIES)
    def test_weighted_scores_are_non_negative(self, query: str) -> None:
        """weighted 的分数落在 ``[0, Σweight]``：非负，但**不保证恒正**（见下一条）. """
        result = hybrid_retriever(strategy="weighted").retrieve(query)

        assert result.hits
        assert result.hits[0].score > 0
        assert all(hit.score >= 0 for hit in result.hits)

    def test_weighted_gives_the_worst_hit_of_a_channel_a_zero_score(self) -> None:
        """min-max 的必然结果：一路里**最差的那条**归一化成 0.0，它在名单里得 0 分.

        它仍然留在名单里（它确实被这一路召回了），但会排到最后——**这是要求
        "别静默丢召回"的结果**：把 0 分的条目直接从名单里删掉，会让 weighted
        的召回比 rrf 少几条，而少了的那几条不会出现在任何数字里。
        """
        result = hybrid_retriever(strategy="weighted").retrieve(QUERY_EXACT)

        assert result.hits[-1].score == pytest.approx(0.0, abs=1e-15)
        assert result.hits[-1].channels == (CHANNEL_VECTOR,)

    def test_the_final_hit_keeps_channels_but_not_the_numeric_evidence(self) -> None:
        """边界：逐条的**数值**证据（原始分数与贡献）留在融合层，不进 ``RetrievalHit``.

        ``RetrievalHit`` 是"交给打包与生成"的形状（不带向量、不带 distance），
        它只保留"这一条是哪几路召回的"这个**可读结论**（``channels``）；
        逐条的 ``channel_scores`` / ``contributions`` 是融合层的产物
        （``fusion.fuse`` 的 ``FusedHit``），因为把它们塞进每一条命中
        会让一份报告里出现三张小表 × N 条。
        """
        result = hybrid_retriever().retrieve(QUERY_EXACT)

        for hit in result.hits:
            assert not hasattr(hit, "contributions")
            assert not hasattr(hit, "channel_scores")
            assert hit.channels
        assert "contributions" in result.fusion

    @pytest.mark.parametrize("strategy", ["rrf", "weighted"])
    def test_channels_are_sorted_by_rank_quality(self, strategy: str) -> None:
        """``channels`` 的顺序 = 名次好坏（不是通道名的字典序）. """
        result = hybrid_retriever(strategy=strategy).retrieve(QUERY_EXACT)
        by_id = {hit.record_id: hit for hit in result.hits}

        assert by_id["h-t-01"].channels == (CHANNEL_BM25, CHANNEL_VECTOR)
        assert by_id["h-c-01"].channels == (CHANNEL_VECTOR,)

    def test_single_channel_hits_carry_one_channel(self) -> None:
        """只被一路召回时不伪造第二条通道（``channels`` 的长度就是证据条数）. """
        result = hybrid_retriever().retrieve(QUERY_SEMANTIC)

        assert all(len(hit.channels) == 1 for hit in result.hits)


# --------------------------------------------------------------------------- #
# 融合参数：ctor 默认 ← RetrievalQuery.extra
# --------------------------------------------------------------------------- #


class TestFusionParameters:
    """融合参数的两处来源，以及"extra 里写错键名"的报错. """

    def test_strategy_defaults_to_rrf(self) -> None:
        assert hybrid_retriever().strategy == "rrf"
        assert hybrid_retriever().retrieve(QUERY_EXACT).fusion["strategy"] == "rrf"

    def test_weighted_strategy_is_reported_in_the_summary(self) -> None:
        result = hybrid_retriever(strategy="weighted").retrieve(QUERY_EXACT)

        assert result.fusion["strategy"] == "weighted"
        assert result.fusion["params"] == {"alpha": 0.5}

    def test_rrf_summary_reports_k_rrf(self) -> None:
        result = hybrid_retriever(k_rrf=10).retrieve(QUERY_EXACT)

        assert result.fusion["params"] == {"k_rrf": 10}

    def test_extra_overrides_the_strategy_for_one_call(self) -> None:
        query = RetrievalQuery(text=QUERY_EXACT, extra={"strategy": "weighted"})
        result = hybrid_retriever().retrieve(query)

        assert result.fusion["strategy"] == "weighted"
        assert any("extra" in note for note in result.notes)

    def test_extra_overrides_alpha_and_k_rrf(self) -> None:
        weighted = hybrid_retriever().retrieve(
            RetrievalQuery(text=QUERY_EXACT, extra={"strategy": "weighted", "alpha": 0.1})
        )
        rrf = hybrid_retriever().retrieve(
            RetrievalQuery(text=QUERY_EXACT, extra={"k_rrf": 5})
        )

        assert weighted.fusion["params"]["alpha"] == 0.1
        assert rrf.fusion["params"]["k_rrf"] == 5

    def test_extra_override_changes_the_outcome_when_alpha_differs(self) -> None:
        """alpha 顶到 0.9 时关键词路的贡献被压到 0.1——名次必须跟着变.

        ``QUERY_EXACT`` 的两条"第一名"（向量的 ``h-c-01`` 与关键词的 ``h-t-01``）
        分数是 ``alpha`` 与 ``1 - alpha``，因此 alpha=0.9 时向量那条翻上来。
        """
        low = hybrid_retriever().retrieve(
            RetrievalQuery(
                text=QUERY_EXACT, top_k=3, fetch_k=3,
                extra={"strategy": "weighted", "alpha": 0.1},
            )
        )
        high = hybrid_retriever().retrieve(
            RetrievalQuery(
                text=QUERY_EXACT, top_k=3, fetch_k=3,
                extra={"strategy": "weighted", "alpha": 0.9},
            )
        )

        assert low.ids()[0] == "h-t-01"
        assert high.ids()[0] == "h-c-01"

    @pytest.mark.parametrize("key", ["alhpa", "fusion", "weight", "strategy_", ""])
    def test_unknown_extra_keys_are_rejected(self, key: str) -> None:
        """静默忽略一个覆盖参数会让调用方以为它生效了（与 RagPipeline 同一条纪律）. """
        with pytest.raises(FusionError) as excinfo:
            hybrid_retriever().retrieve(RetrievalQuery(text=QUERY_EXACT, extra={key: 1}))

        assert "extra" in str(excinfo.value)
        assert "strategy" in str(excinfo.value)

    def test_bad_strategy_in_extra_is_rejected_before_any_work(self) -> None:
        with pytest.raises(FusionError):
            hybrid_retriever().retrieve(
                RetrievalQuery(text=QUERY_EXACT, extra={"strategy": "nope"})
            )

    def test_bad_parameters_are_rejected_even_when_the_store_is_empty(self) -> None:
        """空库早退**排在参数校验之后**：一个写错的 strategy 不该被"库恰好为空"掩盖. """
        hybrid = HybridRetriever(
            hybrid_vector_retriever(empty_store()), LexicalIndex()
        )

        with pytest.raises(FusionError):
            hybrid.retrieve(RetrievalQuery(text=QUERY_EXACT, extra={"strategy": "nope"}))

    def test_weights_in_extra_reach_the_fusion(self) -> None:
        result = hybrid_retriever().retrieve(
            RetrievalQuery(
                text=QUERY_EXACT,
                extra={
                    "strategy": "weighted",
                    "weights": {CHANNEL_VECTOR: 0.8, CHANNEL_BM25: 0.2},
                },
            )
        )

        assert result.fusion["params"]["weights"] == {
            CHANNEL_VECTOR: 0.8,
            CHANNEL_BM25: 0.2,
        }


# --------------------------------------------------------------------------- #
# 多样性、截断与诊断
# --------------------------------------------------------------------------- #


class TestDiversityAndTruncation:
    """多样性在**融合之后**裁剪，次序是融合的权威次序（只重编号）. """

    def test_max_per_doc_trims_after_fusion(self) -> None:
        result = hybrid_retriever().retrieve(
            RetrievalQuery(text=QUERY_EXACT, max_per_doc=1)
        )

        assert result.count == 3
        assert result.dropped_by_diversity == 7
        assert result.doc_ids == ["doc-trouble", "doc-concept", "doc-manual"]

    def test_max_per_doc_from_the_constructor_is_used(self) -> None:
        result = hybrid_retriever(max_per_doc=1).retrieve(QUERY_EXACT)

        assert result.count == 3
        assert result.dropped_by_diversity == len(RECORD_IDS) - 3

    def test_query_level_max_per_doc_wins_over_the_constructor(self) -> None:
        """``max_per_doc=2``：三份文档各留两条 → 融合 10 条里留下 6 条、再被 top_k 截到 5.

        三个数字一起读："阈值没切（0）、多样性挤掉 4 条、top_k 再丢掉 1 条"——
        条数少于期望时该调哪个参数，就是靠这三行分开回答的。
        """
        hybrid = hybrid_retriever(max_per_doc=1)
        resolved = hybrid.retrieve(RetrievalQuery(text=QUERY_EXACT, max_per_doc=2))

        assert resolved.count == 5
        assert resolved.dropped_by_diversity == 4
        assert resolved.dropped_by_top_k == 1
        assert resolved.dropped_below_threshold == 0

    def test_diversity_keeps_the_fusion_order(self) -> None:
        """裁剪不重排：留下的三条必须仍是融合次序的子序列（否则"次序是权威"就没了）. """
        full = hybrid_retriever().retrieve(RetrievalQuery(text=QUERY_EXACT, top_k=10))
        trimmed = hybrid_retriever().retrieve(RetrievalQuery(text=QUERY_EXACT, max_per_doc=1))
        full_order = [hit.record_id for hit in full.hits]
        kept = [hit.record_id for hit in trimmed.hits]

        assert kept == [record_id for record_id in full_order if record_id in set(kept)]

    def test_top_k_truncation_is_counted(self) -> None:
        """``top_k=2`` 时深度跟着变成 ``ceil(2 × 3) = 6``：融合 7 条、截断丢掉 5 条.

        这条用例同时钉住两件事：截断要留数字，以及**深度是 top_k 的函数**
        （两路都只取 6 条，因此关键词路的候选也一起变浅）。
        """
        result = hybrid_retriever().retrieve(RetrievalQuery(text=QUERY_EXACT, top_k=2))

        assert result.count == 2
        assert result.fetch_k == 6
        assert result.fusion["fused"] == 7
        assert result.dropped_by_top_k == 5

    @pytest.mark.parametrize("query", ALL_QUERIES)
    def test_ranks_are_contiguous_from_zero(self, query: str) -> None:
        result = hybrid_retriever().retrieve(query)

        assert [hit.rank for hit in result.hits] == list(range(result.count))

    def test_scores_are_descending(self) -> None:
        scores = [hit.score for hit in hybrid_retriever().retrieve(QUERY_EXACT).hits]

        assert scores == sorted(scores, reverse=True)

    def test_doc_ids_dedupe_in_first_seen_order(self) -> None:
        """``top_k=5`` 的名单里只出现两份文档（``h-t-01`` 一条 + 四条概念文档）."""
        result = hybrid_retriever().retrieve(QUERY_EXACT)

        assert result.doc_ids == ["doc-trouble", "doc-concept"]
        assert len(set(result.doc_ids)) == len(result.doc_ids)

    def test_empty_result_reports_one_reason(self) -> None:
        result = hybrid_retriever().retrieve(
            RetrievalQuery(text=QUERY_EXACT, where={"strategy": "nope"})
        )

        assert result.empty_reason == "filtered_out"
        assert result.count == 0

    def test_no_data_on_an_empty_store(self) -> None:
        result = HybridRetriever(
            hybrid_vector_retriever(empty_store()), LexicalIndex()
        ).retrieve(QUERY_EXACT)

        assert result.hits == ()
        assert result.empty_reason == "no_data"
        assert result.fetch_k == 15
        assert result.channel_candidates == {CHANNEL_VECTOR: 0, CHANNEL_BM25: 0}
        assert any("合法状态" in note for note in result.notes)


# --------------------------------------------------------------------------- #
# 诊断通道：channel_candidates / fusion / notes / explain
# --------------------------------------------------------------------------- #


class TestDiagnostics:
    """"为什么某一路空着"与"为什么这条排第一"都必须能读出来. """

    def test_keyword_channel_emptiness_is_named(self) -> None:
        """纯语义改写时点名关键词路空着的原因（day066 留的那个钩子）. """
        result = hybrid_retriever().retrieve(QUERY_SEMANTIC)
        note = " ".join(result.notes)

        assert "关键词路一条都没召回" in note
        assert "交集为空" in note
        assert "纯语义改写" in note

    def test_keyword_channel_filtered_out_is_named_differently(self) -> None:
        """被 ``where`` 筛没时说的是"筛成了 0 条"（与"词元不认识"是两种成因）. """
        result = hybrid_retriever().retrieve(
            RetrievalQuery(text=QUERY_EXACT, where={"topic": "nope"})
        )
        note = " ".join(result.notes)

        assert "关键词路一条都没召回" in note
        assert "筛成了 0 条" in note
        assert "交集为空" not in note

    def test_known_terms_without_a_matching_document_is_named_differently(self) -> None:
        """词元都认识、过滤之后却没有一篇同时含它们：第三种成因（语料分布）. """
        result = hybrid_retriever().retrieve(
            RetrievalQuery(text=QUERY_EXACT, where={"strategy": "structural"})
        )
        note = " ".join(result.notes)

        assert "关键词路一条都没召回" in note
        assert "没有任何一篇同时包含它们" in note
        assert "筛成了 0 条" not in note

    def test_vector_channel_emptiness_is_named(self) -> None:
        result = hybrid_retriever().retrieve(
            RetrievalQuery(text=QUERY_EXACT, min_score=1.5)
        )

        assert "向量通道一条都没活下来" in " ".join(result.notes)

    def test_fusion_summary_is_complete(self) -> None:
        summary = hybrid_retriever().retrieve(QUERY_BOTH).fusion

        assert summary["strategy"] == "rrf"
        assert summary["params"] == {"k_rrf": 60}
        assert set(summary["recall"]) == {CHANNEL_BM25, CHANNEL_VECTOR}
        assert summary["fused"] == 10
        assert summary["deduped"] == 1

    def test_channel_candidates_are_both_reported(self) -> None:
        result = hybrid_retriever().retrieve(QUERY_EXACT)

        assert result.channel_candidates == {
            CHANNEL_VECTOR: len(RECORD_IDS),
            CHANNEL_BM25: 1,
        }

    def test_lexical_failure_evidence_is_forwarded_into_notes(self) -> None:
        """关键词路的 0 分排除与未登录词两类证据都要进 notes（一路的失败要能解释）. """
        result = hybrid_retriever().retrieve(QUERY_EXACT)
        note = " ".join(result.notes)

        assert "关键词路：" in note
        assert "0 分" in note

    def test_latency_is_reported(self) -> None:
        assert hybrid_retriever().retrieve(QUERY_EXACT).latency_ms >= 0.0

    def test_metric_and_index_state_come_from_the_vector_side(self) -> None:
        result = hybrid_retriever().retrieve(QUERY_EXACT)

        assert result.metric == "cosine"
        assert result.index_state.count_store == len(RECORD_IDS)

    def test_explain_adds_the_hybrid_lines(self) -> None:
        hybrid = hybrid_retriever()
        lines = hybrid.explain(hybrid.retrieve(QUERY_BOTH))
        joined = "\n".join(lines)

        assert "两路明细" in joined
        assert "融合：" in joined
        assert "阈值口径" in joined
        assert "关键词索引" in joined
        assert "分组口径" in joined

    def test_explain_keeps_the_four_original_answers(self) -> None:
        """混合模式不重写 day066 那四问（它们仍然由 ``RetrievalResult.explain`` 给）. """
        hybrid = hybrid_retriever()
        result = hybrid.retrieve(QUERY_EXACT)
        joined = "\n".join(hybrid.explain(result))

        assert "索引版本" in joined
        assert "过滤条件" in joined
        assert "条数" in joined
        assert "空结果原因" in joined

    def test_result_to_dict_carries_the_new_evidence(self) -> None:
        payload = hybrid_retriever().retrieve(QUERY_BOTH).to_dict()

        assert payload["channel_candidates"] == {
            CHANNEL_VECTOR: len(RECORD_IDS),
            CHANNEL_BM25: 1,
        }
        assert payload["fusion"]["strategy"] == "rrf"
        assert payload["hits"][0]["channels"] == [CHANNEL_VECTOR, CHANNEL_BM25]

    def test_describe_reports_both_sides(self) -> None:
        described = hybrid_retriever().describe()

        assert described["name"] == DEFAULT_HYBRID_NAME
        assert described["strategy"] == "rrf"
        assert described["k_rrf"] == 60
        assert described["alpha"] == 0.5
        assert described["weights"] == {}
        assert described["channels"] == [CHANNEL_VECTOR, CHANNEL_BM25]
        assert described["lexical"]["count"] == len(RECORD_IDS)
        assert described["vector"]["top_k"] == 5
        assert described["vector"]["index"]["count_store"] == len(RECORD_IDS)

    def test_describe_shows_the_effective_weights_for_weighted(self) -> None:
        described = hybrid_retriever(strategy="weighted").describe()

        assert described["weights"] == {CHANNEL_VECTOR: 0.5, CHANNEL_BM25: 0.5}

    def test_index_state_is_observable_without_raising(self) -> None:
        """观测入口（``index_state``）在漂移时照常返回状态（与 day066 同一条取舍）. """
        store = hybrid_store()
        manifest = sample_manifest(store)
        store.delete(["h-c-04"])
        hybrid = HybridRetriever(
            hybrid_vector_retriever(store, manifest=manifest), hybrid_lexical(store)
        )

        assert hybrid.index_state.has_drift is True
        assert hybrid.retrieve(QUERY_EXACT).count > 0

    def test_strict_index_refuses_to_answer_on_drift(self) -> None:
        store = hybrid_store()
        manifest = sample_manifest(store)
        store.delete(["h-c-04"])
        hybrid = HybridRetriever(
            hybrid_vector_retriever(store, manifest=manifest),
            hybrid_lexical(store),
            strict_index=True,
        )

        with pytest.raises(IndexStateError) as excinfo:
            hybrid.retrieve(QUERY_EXACT)

        assert "strict_index=True" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# 确定性与装配
# --------------------------------------------------------------------------- #


class TestDeterminism:
    """同一份输入 → 同一份输出（包括库的插入顺序不影响结果）. """

    @pytest.mark.parametrize("query", ALL_QUERIES)
    def test_repeated_retrieval_is_identical(self, query: str) -> None:
        hybrid = hybrid_retriever()
        first = hybrid.retrieve(query)
        second = hybrid.retrieve(query)

        assert first.ids() == second.ids()
        assert first.to_dict()["hits"] == second.to_dict()["hits"]

    def test_reversed_insertion_order_gives_the_same_answer(self) -> None:
        forward = hybrid_retriever(hybrid_store())
        backward = hybrid_retriever(hybrid_store(reverse=True))

        assert forward.retrieve(QUERY_EXACT).ids() == backward.retrieve(QUERY_EXACT).ids()
        assert forward.retrieve(QUERY_CODE).ids() == backward.retrieve(QUERY_CODE).ids()

    def test_string_and_query_objects_agree(self) -> None:
        hybrid = hybrid_retriever()

        assert hybrid.retrieve(QUERY_EXACT).ids() == hybrid.retrieve(
            RetrievalQuery(text=QUERY_EXACT)
        ).ids()

    def test_retrieve_many_is_ordered(self) -> None:
        results = hybrid_retriever().retrieve_many(list(ALL_QUERIES))

        assert [result.ids() for result in results] == [
            list(HYBRID_DEFAULT[query]) for query in ALL_QUERIES
        ]

    def test_blank_query_is_rejected(self) -> None:
        with pytest.raises(QueryError):
            hybrid_retriever().retrieve("   ")

    def test_fetch_k_smaller_than_top_k_is_rejected(self) -> None:
        with pytest.raises(QueryError):
            hybrid_retriever().retrieve(RetrievalQuery(text=QUERY_EXACT, top_k=5, fetch_k=2))


class TestConstruction:
    """构造期的三组校验（每个错误都属于一个明确的族）. """

    def test_requires_a_retriever(self) -> None:
        with pytest.raises(QueryError) as excinfo:
            HybridRetriever(hybrid_store(), hybrid_lexical())  # type: ignore[arg-type]

        assert "Retriever" in str(excinfo.value)

    def test_requires_a_lexical_index(self) -> None:
        with pytest.raises(QueryError) as excinfo:
            HybridRetriever(hybrid_vector_retriever(), hybrid_store())  # type: ignore[arg-type]

        assert "LexicalIndex" in str(excinfo.value)

    def test_rrf_with_weights_is_rejected_at_construction(self) -> None:
        """这是**装配**错误而不是请求错误——因此在构造期就响. """
        with pytest.raises(FusionError) as excinfo:
            hybrid_retriever(weights={CHANNEL_VECTOR: 1.0, CHANNEL_BM25: 0.0})

        assert "不接受 weights" in str(excinfo.value)

    def test_weighted_with_weights_is_accepted(self) -> None:
        hybrid = hybrid_retriever(
            strategy="weighted", weights={CHANNEL_VECTOR: 0.7, CHANNEL_BM25: 0.3}
        )

        assert hybrid.describe()["weights"] == {CHANNEL_VECTOR: 0.7, CHANNEL_BM25: 0.3}

    @pytest.mark.parametrize("max_per_doc", [-1, 1.5, "1", True])
    def test_bad_max_per_doc_is_rejected(self, max_per_doc: Any) -> None:
        with pytest.raises(QueryError):
            hybrid_retriever(max_per_doc=max_per_doc)

    @pytest.mark.parametrize("name", ["", "   ", None, 7])
    def test_bad_name_is_rejected(self, name: Any) -> None:
        with pytest.raises(QueryError):
            hybrid_retriever(name=name)

    @pytest.mark.parametrize("strategy", [1, "", "RRF"])
    def test_bad_strategy_is_rejected(self, strategy: Any) -> None:
        with pytest.raises(FusionError):
            hybrid_retriever(strategy=strategy)

    @pytest.mark.parametrize("alpha", [1.5, -0.1, "0.5", [], {}])
    def test_bad_alpha_is_rejected(self, alpha: Any) -> None:
        """``None`` **不在**这个清单里：它表示"没指定，去读 settings"（见构造契约）. """
        with pytest.raises(FusionError):
            hybrid_retriever(alpha=alpha)

    def test_alpha_none_means_read_the_settings(self) -> None:
        hybrid = hybrid_retriever(alpha=None)

        assert hybrid.describe()["alpha"] == settings.retrieval_hybrid_alpha

    @pytest.mark.parametrize("k_rrf", [0, -3, 1.5, "60"])
    def test_bad_k_rrf_is_rejected(self, k_rrf: Any) -> None:
        with pytest.raises(FusionError):
            hybrid_retriever(k_rrf=k_rrf)

    @pytest.mark.parametrize(
        "weights", [["vector"], {CHANNEL_VECTOR: -1.0}, {CHANNEL_VECTOR: "0.5"},
                    {CHANNEL_VECTOR: 0.5, "bm25x": 0.5}]
    )
    def test_bad_weights_are_rejected(self, weights: Any) -> None:
        with pytest.raises(FusionError):
            hybrid_retriever(strategy="weighted", weights=weights)

    def test_lexical_and_vector_accessors(self) -> None:
        hybrid = hybrid_retriever()

        assert isinstance(hybrid.vector, Retriever)
        assert isinstance(hybrid.lexical, LexicalIndex)
        assert hybrid.lexical.count == len(RECORD_IDS)

    def test_constructor_does_no_index_work(self) -> None:
        """构造期零副作用：没有清单、没有编码、没有检索（与 routes 的导入期纪律同源）. """
        embedding = hybrid_embedding()
        store = hybrid_store()
        hybrid = HybridRetriever(
            hybrid_vector_retriever(store, embedding=embedding), hybrid_lexical(store)
        )

        assert embedding.calls == []
        assert hybrid.name == DEFAULT_HYBRID_NAME


class TestBuildHybridRetriever:
    """``build_hybrid_retriever``：三个数字从 settings 来，关键词索引现建一份. """

    def test_reads_the_settings_defaults(self) -> None:
        hybrid = build_hybrid_retriever(hybrid_store(), hybrid_embedding())
        described = hybrid.describe()

        assert described["strategy"] == settings.retrieval_hybrid_strategy
        assert described["k_rrf"] == settings.retrieval_hybrid_rrf_k
        assert described["alpha"] == settings.retrieval_hybrid_alpha
        assert described["lexical"]["k1"] == settings.retrieval_bm25_k1
        assert described["lexical"]["b"] == settings.retrieval_bm25_b

    def test_builds_the_lexical_index_from_the_backend(self) -> None:
        hybrid = build_hybrid_retriever(hybrid_store(), hybrid_embedding())

        assert hybrid.lexical.count == len(RECORD_IDS)
        assert hybrid.lexical.ids() == sorted(RECORD_IDS)

    def test_accepts_an_injected_lexical_index(self) -> None:
        index = LexicalIndex.build(documents(), params=BM25Params(k1=0.0, b=0.0))
        hybrid = build_hybrid_retriever(hybrid_store(), hybrid_embedding(), lexical=index)

        assert hybrid.lexical is index
        assert hybrid.describe()["lexical"]["k1"] == 0.0

    def test_explicit_arguments_win_over_settings(self) -> None:
        hybrid = build_hybrid_retriever(
            hybrid_store(),
            hybrid_embedding(),
            strategy="weighted",
            alpha=0.25,
            k_rrf=7,
            name="custom",
        )
        described = hybrid.describe()

        assert described["strategy"] == "weighted"
        assert described["alpha"] == 0.25
        assert described["k_rrf"] == 7
        assert described["name"] == "custom"

    def test_passes_the_manifest_through(self) -> None:
        store = hybrid_store()
        hybrid = build_hybrid_retriever(
            store, hybrid_embedding(), manifest=sample_manifest(store)
        )

        assert hybrid.retrieve(QUERY_EXACT).index_state.version_id != ""

    def test_end_to_end_answer_matches_the_manual_retriever(self) -> None:
        manual = hybrid_retriever()
        assembled = build_hybrid_retriever(hybrid_store(), hybrid_embedding())

        assert assembled.retrieve(QUERY_EXACT).ids() == manual.retrieve(QUERY_EXACT).ids()

    def test_lexical_index_is_a_snapshot_not_a_live_view(self) -> None:
        """装配函数**调用一次**只建一次索引；库之后变了，关键词索引不会自动跟上.

        这条边界必须被钉住：它决定了"库里新增了文档之后关键词路为什么看不到"
        这个问题的答案（重建关键词索引，或注入一个自己管生命周期的 LexicalIndex）。
        """
        store = hybrid_store()
        hybrid = build_hybrid_retriever(store, hybrid_embedding())
        before = hybrid.retrieve(QUERY_EXACT).channel_candidates[CHANNEL_VECTOR]

        store.upsert(
            [
                make_record(
                    record_id="h-new",
                    vector=(0.0,) * 7 + (1.0,),
                    text="新加的一条记录",
                    metadata={"parent_doc_id": "doc-new"},
                )
            ]
        )
        after = hybrid.retrieve(QUERY_EXACT)

        assert after.channel_candidates[CHANNEL_VECTOR] == before + 1
        assert hybrid.lexical.count == len(RECORD_IDS)


class TestHybridEnabled:
    """``hybrid_enabled`` 读的是 ``settings.retrieval_hybrid_enabled``（默认关闭）. """

    def test_default_is_disabled(self) -> None:
        assert settings.retrieval_hybrid_enabled is False
        assert hybrid_enabled() is False

    def test_reflects_the_setting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "retrieval_hybrid_enabled", True)

        assert hybrid_enabled() is True


class TestVectorOnlyBaseline:
    """单路向量路的名次也钉住：互补的说法必须有对照物. """

    @pytest.mark.parametrize("query", sorted(VECTOR_ONLY_SHALLOW))
    def test_shallow_vector_hits(self, query: str) -> None:
        result = hybrid_vector_retriever().retrieve(
            RetrievalQuery(text=query, top_k=3, fetch_k=3)
        )

        assert result.ids() == list(VECTOR_ONLY_SHALLOW[query])

    def test_vector_scores_are_all_zero_besides_the_first(self) -> None:
        """样本的向量只有 1.0 与 0.0 两种取值——"0 分并列"是本课要用的事实. """
        result = hybrid_vector_retriever().retrieve(
            RetrievalQuery(text=QUERY_EXACT, top_k=3, fetch_k=3)
        )

        assert result.hits[0].score == 1.0
        assert all(hit.score == 0.0 for hit in result.hits[1:])
