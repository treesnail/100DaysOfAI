"""day066 ``retrieval.retriever`` 的单元测试：九步流水线与三道减法.

这一层的全部价值在于"条数少于期望"必须能被回答，因此断言按九步里的四步
（深度 / 过滤 / 阈值 / 多样性）分组，每一步都同时核对**条数**与**记下的数字**：

```text
深度       fetch_k = max(top_k, ceil(top_k × multiplier))，封顶 MAX_FETCH_K 并留注记
过滤       candidates 是"过滤后还剩多少条可选"，filter_applied 是"这次到底有没有过滤"
阈值       在本层落刀：切掉几条必须能报出来（day067 的融合要用这个数字）
多样性     按 doc_id 每组留前 N 条、被挤掉的计数、名次重编成 0 起连续
诊断       empty_reason 只有一个取值；四因各自有触发用例
索引状态   清单漂移只观测不阻断（strict_index=True 时才拒绝服务）
```

**"diversity_trimmed" 那一支按规格说明用单元级构造触发**：端到端路径上
每一组至少会留下一条，因此 ``dropped_by_diversity > 0`` 时命中必然非空
（见 ``retriever._diagnose`` 的可达性说明）。用例里直接调用 ``_diagnose``
并写清了这一点，免得下一个人拿着"四种空结果"的清单在端到端路径上找一个
不存在的情形。

全部离线、确定性：库是 ``FlatVectorStore``，编码器是查表的 ``TableEmbedding``，
期望分数是手算出来的常量（见 ``tests/retrieval_samples.py``）。
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from smart_research_agent.config import settings
from smart_research_agent.indexing.manifest import build_manifest
from smart_research_agent.retrieval.errors import IndexStateError, QueryError
from smart_research_agent.retrieval.retriever import (
    Retriever,
    _apply_diversity,
    _apply_threshold,
    _diagnose,
    _rerank,
    build_retriever,
)
from smart_research_agent.retrieval.types import (
    DEFAULT_FETCH_MULTIPLIER,
    EMPTY_REASON_BELOW_THRESHOLD,
    EMPTY_REASON_DIVERSITY,
    EMPTY_REASON_FILTERED_OUT,
    EMPTY_REASON_NO_DATA,
    EMPTY_REASON_NONE,
    MAX_FETCH_K,
    RetrievalHit,
    RetrievalQuery,
    TimeRange,
)
from smart_research_agent.vectorstore.flat import FlatVectorStore
from smart_research_agent.vectorstore.types import make_record
from tests.retrieval_samples import (
    EXPECTED_AXIS_ORDER,
    EXPECTED_AXIS_SCORES,
    EXPECTED_ORTHO_ORDER,
    EXPECTED_TILT_ORDER,
    QUERY_TEXTS,
    QUERY_VECTORS,
    RECORD_IDS,
    RECORD_TEXTS,
    FixedVectorEmbedding,
    NotASequenceEmbedding,
    TableEmbedding,
    doc_ids_by_parent,
    empty_store,
    encoded_store,
    flat_store,
    lexical_embedding,
    plain_records,
    sample_manifest,
    sample_store,
)

#: 三个查询（文本本身没有语义，向量由 ``TableEmbedding`` 给定）.
AXIS = QUERY_TEXTS["axis"]
TILT = QUERY_TEXTS["tilt"]
ORTHO = QUERY_TEXTS["ortho"]


def sample_retriever(
    store: FlatVectorStore | None = None,
    embedding: Any | None = None,
    **overrides: Any,
) -> Retriever:
    """样本库 + ``TableEmbedding`` 的检索器（构造参数逐个覆盖）."""
    return Retriever(
        store if store is not None else sample_store(),
        embedding if embedding is not None else TableEmbedding(),
        **overrides,
    )


# --------------------------------------------------------------------------- #
# 第一步与第八步：Top-K 正确性与同分排序
# --------------------------------------------------------------------------- #


class TestTopK:
    """Top-K 的正确答案是"分数降序 + 同分按 id 升序"，两条都要断言."""

    def test_returns_the_best_five_in_order(self) -> None:
        result = sample_retriever().retrieve(AXIS)

        assert result.count == 5
        assert result.top_k == 5
        assert result.ids() == list(EXPECTED_AXIS_ORDER[:5])
        assert [hit.score for hit in result.hits] == list(EXPECTED_AXIS_SCORES[:5])

    def test_ranks_are_contiguous_from_zero(self) -> None:
        """名次必须从 0 连续（带空洞的名次列表不能用来索引"第 3 条"）."""
        result = sample_retriever().retrieve(AXIS)

        assert [hit.rank for hit in result.hits] == [0, 1, 2, 3, 4]

    def test_scores_are_descending(self) -> None:
        scores = [hit.score for hit in sample_retriever().retrieve(AXIS).hits]

        assert scores == sorted(scores, reverse=True)

    def test_tie_is_broken_by_record_id(self) -> None:
        """``c-a-01`` 与 ``c-b-01`` 的文本与向量逐位相同 → 分数**精确相等**.

        没有第二排序键时，顺序取决于字典迭代顺序（也就是插入顺序），
        于是"同一次查询两次运行给出不同的 top-1"。
        """
        result = sample_retriever().retrieve(AXIS)
        first, second = result.hits[0], result.hits[1]

        assert first.score == second.score == 1.0
        assert [first.record_id, second.record_id] == ["c-a-01", "c-b-01"]
        assert first.record_id < second.record_id

    def test_reversed_insertion_order_gives_the_same_answer(self) -> None:
        """逆序写入的库给出同一份结果（排序规则不依赖后端实现）."""
        forward = sample_retriever(sample_store()).retrieve(AXIS)
        backward = sample_retriever(sample_store(reverse=True)).retrieve(AXIS)

        assert backward.ids() == forward.ids()
        assert [hit.score for hit in backward.hits] == [hit.score for hit in forward.hits]

    def test_orthogonal_query_orders_by_id_only(self) -> None:
        """八条分数全是 0.0 时，次序完全由 id 决定（分数不能提供任何信息）."""
        result = sample_retriever().retrieve(ORTHO)

        assert result.ids() == list(EXPECTED_ORTHO_ORDER[:5])
        assert all(hit.score == 0.0 for hit in result.hits)

    def test_tilt_query_prefers_the_tilted_records(self) -> None:
        result = sample_retriever().retrieve(TILT)

        assert result.ids() == list(EXPECTED_TILT_ORDER[:5])
        assert result.hits[0].record_id == "c-c-02"
        assert result.hits[0].score == pytest.approx(1.0)

    def test_top_k_one_keeps_only_the_first(self) -> None:
        """深度足够时（8 条全取回）截断到 1 条：丢掉的是 7 条."""
        result = sample_retriever(default_top_k=1, fetch_multiplier=8).retrieve(AXIS)

        assert result.ids() == ["c-a-01"]
        assert result.fetch_k == 8
        assert result.dropped_by_top_k == 7

    def test_shallow_default_depth_limits_the_candidate_pool(self) -> None:
        """深度只有 3 时，被截断丢掉的只有 2 条（剩下的 5 条根本没被取回来）."""
        result = sample_retriever(default_top_k=1).retrieve(AXIS)

        assert result.fetch_k == 3
        assert result.ids() == ["c-a-01"]
        assert result.dropped_by_top_k == 2

    def test_hits_carry_text_metadata_and_channel(self) -> None:
        """命中要自带正文与元数据（展示侧不必再回库取一次）."""
        hit = sample_retriever().retrieve(AXIS).hits[0]

        assert hit.text == RECORD_TEXTS["c-a-01"]
        assert hit.doc_id == "doc-alpha"
        assert hit.heading_path == "检索手册 > 分块"
        assert hit.channel == "vector"
        assert hit.metadata["strategy"] == "structural"

    def test_hit_text_is_the_record_text_not_the_query(self) -> None:
        """命中里的正文来自库（把查询文本当正文返回是一个很容易犯的错）."""
        result = sample_retriever().retrieve(AXIS)

        assert all(hit.text != AXIS for hit in result.hits)

    def test_metric_comes_from_the_store(self) -> None:
        assert sample_retriever().retrieve(AXIS).metric == "cosine"

    def test_channels_is_the_single_vector_channel(self) -> None:
        assert sample_retriever().retrieve(AXIS).channels == ("vector",)

    def test_query_text_is_stripped_before_encoding(self) -> None:
        """同一句话不该因为首尾空格而被当成两次检索（缓存与评估都会算错）."""
        embedding = TableEmbedding()

        result = sample_retriever(embedding=embedding).retrieve(f"  {AXIS}  ")

        assert embedding.calls == [AXIS]
        assert result.query.text == AXIS

    def test_query_is_encoded_exactly_once(self) -> None:
        """一次检索只编码一次（阈值与多样性都在结果上落刀，不重新编码）."""
        embedding = TableEmbedding()

        sample_retriever(embedding=embedding).retrieve(AXIS)

        assert len(embedding.calls) == 1


# --------------------------------------------------------------------------- #
# 第三步：召回深度
# --------------------------------------------------------------------------- #


class TestFetchDepth:
    """``fetch_k`` 要能被复述：它可能来自倍率、可能被显式指定、可能被封顶."""

    def test_default_depth_is_three_times_top_k(self) -> None:
        result = sample_retriever().retrieve(AXIS)

        assert result.query.top_k == 5
        assert result.fetch_k == 5 * DEFAULT_FETCH_MULTIPLIER == 15
        assert result.notes == ()

    @pytest.mark.parametrize(("multiplier", "expected"), [(1, 5), (2, 10), (4, 20)])
    def test_multiplier_scales_the_depth(self, multiplier: int, expected: int) -> None:
        result = sample_retriever(fetch_multiplier=multiplier).retrieve(AXIS)

        assert result.fetch_k == expected

    def test_explicit_fetch_k_is_used_as_is(self) -> None:
        query = RetrievalQuery(text=AXIS, top_k=4, fetch_k=4)

        result = sample_retriever().retrieve(query)

        assert result.fetch_k == 4
        assert result.count == 4
        assert result.dropped_by_top_k == 0

    def test_explicit_fetch_k_can_be_deeper_than_needed(self) -> None:
        """深度给得更大时只影响"取回多少条"，不影响最终条数."""
        result = sample_retriever().retrieve(RetrievalQuery(text=AXIS, top_k=5, fetch_k=40))

        assert result.fetch_k == 40
        assert result.count == 5
        assert result.dropped_by_top_k == 3

    def test_depth_is_capped_with_a_note(self) -> None:
        """封顶必须在 ``notes`` 里说明：否则"设置里写 400、实际取 1000"要靠读代码发现."""
        result = sample_retriever(default_top_k=400).retrieve(AXIS)

        assert result.fetch_k == MAX_FETCH_K
        assert any(
            "已封顶" in note and f"MAX_FETCH_K={MAX_FETCH_K}" in note for note in result.notes
        )

    def test_explicit_fetch_k_is_capped_too(self) -> None:
        result = sample_retriever().retrieve(
            RetrievalQuery(text=AXIS, top_k=5, fetch_k=MAX_FETCH_K + 500)
        )

        assert result.fetch_k == MAX_FETCH_K
        assert any("已封顶" in note for note in result.notes)

    @pytest.mark.parametrize(
        ("top_k", "expected", "capped"),
        [(333, 999, False), (334, 1000, True)],
    )
    def test_cap_boundary(self, top_k: int, expected: int, capped: bool) -> None:
        """恰好不超过上限时不封顶（333×3=999 → 原样；334×3=1002 → 封顶）."""
        result = sample_retriever(default_top_k=top_k).retrieve(AXIS)

        assert result.fetch_k == expected
        assert any("已封顶" in note for note in result.notes) is capped

    def test_explicit_fetch_k_at_the_cap_is_not_capped(self) -> None:
        result = sample_retriever().retrieve(
            RetrievalQuery(text=AXIS, top_k=5, fetch_k=MAX_FETCH_K)
        )

        assert result.fetch_k == MAX_FETCH_K
        assert result.notes == ()

    def test_small_fetch_k_converges_top_k_and_leaves_a_note(self) -> None:
        """显式 ``fetch_k`` 小于默认 ``top_k`` 时收敛 ``top_k`` 而不是报错.

        "深度 >= 条数"这条不变量必须成立，而调用方显然想要那个更小的深度；
        这次改写必须可见——静默改写会让"我明明要 5 条"无从解释。
        """
        result = sample_retriever().retrieve(RetrievalQuery(text=AXIS, fetch_k=2))

        assert result.query.top_k == 2
        assert result.fetch_k == 2
        assert result.count == 2
        assert any("已把本次 top_k 收敛为 2" in note for note in result.notes)

    def test_fetch_k_equal_to_default_top_k_leaves_no_note(self) -> None:
        result = sample_retriever().retrieve(RetrievalQuery(text=AXIS, fetch_k=5))

        assert result.query.top_k == 5
        assert result.notes == ()


# --------------------------------------------------------------------------- #
# 第五步：库侧过滤
# --------------------------------------------------------------------------- #


class TestFiltering:
    """``where`` 与时间范围都在库里生效，检索器负责把"过滤后剩多少"报出来."""

    def test_no_filter_reports_the_whole_store_as_candidates(self) -> None:
        result = sample_retriever().retrieve(AXIS)

        assert result.candidates == 8
        assert result.filter_applied is False

    def test_where_narrows_candidates(self) -> None:
        query = RetrievalQuery(text=AXIS, where={"strategy": "structural"})

        result = sample_retriever().retrieve(query)

        assert result.filter_applied is True
        assert result.candidates == 4
        assert result.ids() == ["c-a-01", "c-b-01", "c-a-02", "c-c-02"]

    def test_where_on_an_array_field(self) -> None:
        """``$contains`` 让"按标签筛"变成一个可核对的数量（guide 出现在 5 条里）."""
        result = sample_retriever().retrieve(
            RetrievalQuery(text=AXIS, where={"tags": {"$contains": "guide"}})
        )

        assert result.candidates == 5
        assert result.ids() == ["c-a-01", "c-b-01", "c-a-02", "c-c-02", "c-a-03"]

    def test_where_on_heading_path(self) -> None:
        """day062 的 ``heading_path`` 在这里第一次被当成过滤条件用上."""
        result = sample_retriever().retrieve(
            RetrievalQuery(text=AXIS, where={"heading_path": "检索手册 > 阈值"})
        )

        assert result.candidates == 1
        assert result.ids() == ["c-c-02"]

    def test_where_matching_nothing_is_not_an_error(self) -> None:
        """过滤器把候选筛没了：返回空结果 + 一条注记，而不是抛异常."""
        result = sample_retriever().retrieve(
            RetrievalQuery(text=AXIS, where={"strategy": "不存在的策略"})
        )

        assert result.count == 0
        assert result.candidates == 0
        assert result.filter_applied is True
        assert result.empty_reason == EMPTY_REASON_FILTERED_OUT
        assert any("筛成了 0 条" in note for note in result.notes)

    def test_time_range_is_applied_by_the_store(self) -> None:
        """时间范围走 ``filters.combine_where`` → 全 September 有 6 条."""
        result = sample_retriever().retrieve(
            RetrievalQuery(
                text=AXIS,
                time_range=TimeRange(start="2026-09-01", end="2026-09-30"),
            )
        )

        assert result.candidates == 6
        assert result.filter_applied is True
        assert result.count == 5

    def test_records_missing_created_at_are_excluded(self) -> None:
        """**缺 ``created_at`` 的那条被排除**（8 条里只有 6 条落在区间内）.

        库里有 8 条：``c-c-02`` 是 8 月（区间外）、``c-c-01`` 没有时间字段
        （``$gte`` 对缺失键返回 False）——两者都不在候选里，这不是 bug。
        """
        result = sample_retriever().retrieve(
            RetrievalQuery(
                text=AXIS,
                time_range=TimeRange(start="2026-09-01", end="2026-09-30"),
            )
        )

        assert result.candidates == 6
        assert "c-c-01" not in result.ids()
        assert "c-c-02" not in result.ids()

    def test_whole_year_range_still_excludes_the_missing_field(self) -> None:
        """区间放大到整年时，被排除的只剩"没有这个字段"的那一条（7 vs 8）."""
        result = sample_retriever().retrieve(
            RetrievalQuery(
                text=AXIS,
                time_range=TimeRange(start="2026-01-01", end="2026-12-31"),
            )
        )

        assert result.candidates == 7
        assert "c-c-01" not in result.ids()

    def test_closed_interval_includes_both_ends(self) -> None:
        """``start == end`` 时闭区间命中那一天的那一条（不是空集）."""
        result = sample_retriever(default_top_k=8).retrieve(
            RetrievalQuery(text=AXIS, time_range=TimeRange(start="2026-09-15", end="2026-09-15"))
        )

        assert result.ids() == ["c-b-01"]

    def test_single_end_range(self) -> None:
        """单端区间：上界 2026-09-05 时落在区间内的是 3 条（含 8 月那条，不含 9 月 15 日那条）."""
        result = sample_retriever().retrieve(
            RetrievalQuery(text=AXIS, time_range=TimeRange(end="2026-09-05"))
        )

        assert result.candidates == 3
        assert result.ids() == ["c-a-01", "c-a-02", "c-c-02"]

    def test_where_and_time_range_are_and_ed(self) -> None:
        """元数据条件与时间范围同时生效（candidates 是两者取交集的结果）."""
        result = sample_retriever().retrieve(
            RetrievalQuery(
                text=AXIS,
                where={"strategy": "structural"},
                time_range=TimeRange(start="2026-09-01", end="2026-09-30"),
            )
        )

        assert result.candidates == 3
        assert result.ids() == ["c-a-01", "c-b-01", "c-a-02"]
        assert result.filter_applied is True

    def test_time_field_of_the_range_is_configurable(self) -> None:
        """换一个时间字段名：样本里没有这个键 → 候选为 0（而不是静默忽略条件）."""
        range_over_another_field = TimeRange(field="published_at", start="2026-01-01")

        result = sample_retriever().retrieve(
            RetrievalQuery(text=AXIS, time_range=range_over_another_field)
        )

        assert result.candidates == 0
        assert result.empty_reason == EMPTY_REASON_FILTERED_OUT

    def test_field_conflict_surfaces_through_retrieve(self) -> None:
        """``where`` 与 ``time_range`` 撞在同一个字段 → ``QueryError``（不许猜）."""
        with pytest.raises(QueryError) as excinfo:
            sample_retriever().retrieve(
                RetrievalQuery(
                    text=AXIS,
                    where={"created_at": {"$gt": "2026-01-01"}},
                    time_range=TimeRange(start="2026-06-01"),
                )
            )

        assert "过滤条件冲突" in str(excinfo.value)

    def test_filter_conflict_is_reported_even_on_an_empty_store(self) -> None:
        """同一个字段撞在两处，**空库上也要报** —— 错误的库状态改变不了错误的查询.

        ``retrieve`` 里"库是空的"这条早退排在 ``combine_where`` **之后**，
        因此"过滤条件冲突"这个**调用方错误**在任何库状态下都抛 ``QueryError``。

        这条顺序不是随手定的：如果早退在前，同一种调用方错误会在库非空时
        抛 ``QueryError``、在库为空时被静默降级成 ``empty_reason=no_data``——
        后者会让调用方以为"条件是合并生效的，只是恰好没数据"，
        于是把"我写错了条件"读成"这批数据里没有"。
        """
        conflict_query = RetrievalQuery(
            text=AXIS,
            where={"created_at": "2026-01-01"},
            time_range=TimeRange(start="2026-06-01"),
        )
        with pytest.raises(QueryError):
            sample_retriever().retrieve(conflict_query)

        with pytest.raises(QueryError):
            sample_retriever(empty_store()).retrieve(conflict_query)


# --------------------------------------------------------------------------- #
# 第六步：阈值在本层落刀
# --------------------------------------------------------------------------- #


class TestThreshold:
    """阈值必须在本层落刀：交给库过滤就拿不到"切掉了几条"这个数字."""

    def test_constructor_threshold_drops_five(self) -> None:
        result = sample_retriever(min_score=0.8).retrieve(AXIS)

        assert result.ids() == ["c-a-01", "c-b-01", "c-a-02"]
        assert result.dropped_below_threshold == 5
        assert result.dropped_by_top_k == 0
        assert result.candidates == 8

    def test_hit_equal_to_the_threshold_is_kept(self) -> None:
        """``>=`` 而不是 ``>``：``c-a-02`` 的分数精确等于 0.8，必须留下.

        一个 ``>`` 与 ``>=`` 的差别在"阈值恰好等于某条命中分数"时体现出来，
        而那种情形**必然出现**——同一个确定性编码器对同文本给出逐位相同的向量。
        """
        result = sample_retriever(min_score=0.8).retrieve(AXIS)

        assert result.hits[-1].record_id == "c-a-02"
        assert result.hits[-1].score == 0.8

    def test_query_threshold_overrides_the_constructor(self) -> None:
        result = sample_retriever(min_score=0.99).retrieve(
            RetrievalQuery(text=AXIS, min_score=0.5)
        )

        assert result.count == 5
        assert result.dropped_below_threshold == 3

    def test_query_without_threshold_uses_the_constructor_one(self) -> None:
        result = sample_retriever(min_score=0.6).retrieve(AXIS)

        assert result.ids() == ["c-a-01", "c-b-01", "c-a-02", "c-c-02"]
        assert result.dropped_below_threshold == 4

    def test_zero_threshold_drops_nothing(self) -> None:
        """阈值 0.0 与"不设阈值"在余弦下等价（分数不会小于 0 的那几条之外还有更小）."""
        result = sample_retriever(min_score=0.0).retrieve(RetrievalQuery(text=ORTHO, top_k=8))

        assert result.dropped_below_threshold == 0
        assert result.count == 8

    def test_all_hits_can_be_cut_away(self) -> None:
        """阈值把命中全切了 → 空结果 + ``below_threshold``（原因只有一个）."""
        result = sample_retriever().retrieve(RetrievalQuery(text=ORTHO, min_score=0.5))

        assert result.count == 0
        assert result.candidates == 8
        assert result.dropped_below_threshold == 8
        assert result.empty_reason == EMPTY_REASON_BELOW_THRESHOLD

    def test_threshold_matches_the_backend_semantics(self) -> None:
        """与 ``VectorBackend.query(min_score=...)`` 的结果集合逐条相同（同一个口径）.

        差别只在"切掉了几条"这个数字有没有被记下来——而那个数字正是 day067
        判断"这一路是不是一条都没活下来"的依据。
        """
        store = sample_store()
        backend_hits = store.query(
            TableEmbedding().embed(AXIS), 15, min_score=0.6
        ).ids()

        result = sample_retriever(store, min_score=0.6).retrieve(AXIS)

        assert result.ids() == backend_hits

    @pytest.mark.parametrize(
        "overrides",
        [{}, {"min_score": 0.5}, {"max_per_doc": 1}, {"default_top_k": 3}],
    )
    def test_three_dropped_numbers_account_for_every_candidate(
        self, overrides: dict[str, Any]
    ) -> None:
        """**三道减法相加必须等于深度取回的条数**（不该有"凭空少掉"的部分）.

        深度 15 ≥ 候选 8，因此深度取回的就是全部候选（无过滤时是 8 条）。
        """
        result = sample_retriever(**overrides).retrieve(AXIS)

        accounted = (
            result.count
            + result.dropped_below_threshold
            + result.dropped_by_diversity
            + result.dropped_by_top_k
        )
        assert accounted == result.candidates == 8

    def test_accounting_also_holds_for_a_filtered_query(self) -> None:
        """有过滤时账要按"过滤后的候选数"算（4 条），而不是库的总条数."""
        query = RetrievalQuery(text=AXIS, where={"strategy": "structural"})

        result = sample_retriever().retrieve(query)

        accounted = (
            result.count
            + result.dropped_below_threshold
            + result.dropped_by_diversity
            + result.dropped_by_top_k
        )
        assert accounted == result.candidates == 4

    def test_accounting_holds_when_the_depth_is_shallow(self) -> None:
        """深度小于候选数时，取回的是 ``fetch_k`` 条（账要按它算）."""
        result = sample_retriever(default_top_k=2, fetch_multiplier=1).retrieve(AXIS)

        accounted = (
            result.count
            + result.dropped_below_threshold
            + result.dropped_by_diversity
            + result.dropped_by_top_k
        )
        assert accounted == result.fetch_k == 2


# --------------------------------------------------------------------------- #
# 第七步：多样性裁剪
# --------------------------------------------------------------------------- #


class TestDiversity:
    """``max_per_doc`` 是"把名额让给别的文档"这个取舍的开关."""

    def test_max_per_doc_one_keeps_the_best_of_each_doc(self) -> None:
        result = sample_retriever(max_per_doc=1).retrieve(AXIS)

        assert result.ids() == ["c-a-01", "c-b-01", "c-c-02"]
        assert result.doc_ids == ["doc-alpha", "doc-beta", "doc-gamma"]
        assert result.dropped_by_diversity == 5
        assert result.dropped_by_top_k == 0

    def test_dropped_count_matches_the_per_doc_excess(self) -> None:
        """被挤掉的条数 = 各文档条数减 1 之和（doc-alpha 3 条、doc-beta 2 条、doc-gamma 3 条）."""
        grouped = doc_ids_by_parent()
        expected = sum(len(ids) - 1 for ids in grouped.values())

        result = sample_retriever(max_per_doc=1).retrieve(AXIS)

        assert grouped == {
            "doc-alpha": ["c-a-01", "c-a-02", "c-a-03"],
            "doc-beta": ["c-b-01", "c-b-02"],
            "doc-gamma": ["c-c-01", "c-c-02", "c-c-03"],
        }
        assert result.dropped_by_diversity == expected == 5

    def test_max_per_doc_two_still_trims(self) -> None:
        result = sample_retriever(max_per_doc=2).retrieve(AXIS)

        assert result.ids() == ["c-a-01", "c-b-01", "c-a-02", "c-c-02", "c-c-01"]
        assert result.dropped_by_diversity == 2
        assert result.dropped_by_top_k == 1

    def test_ranks_are_renumbered_after_trimming(self) -> None:
        """名次在三道减法之后重编（阈值/多样性留下的空洞不能带进结果）."""
        result = sample_retriever(max_per_doc=1, min_score=0.6).retrieve(AXIS)

        assert [hit.rank for hit in result.hits] == list(range(result.count))
        assert result.ids() == ["c-a-01", "c-b-01", "c-c-02"]

    def test_query_level_max_per_doc_overrides_the_constructor(self) -> None:
        result = sample_retriever(max_per_doc=2).retrieve(
            RetrievalQuery(text=AXIS, max_per_doc=1)
        )

        assert result.count == 3
        assert result.dropped_by_diversity == 5

    def test_zero_means_unlimited(self) -> None:
        """构造参数里的 0 = 不限（与 settings 的口径一致）；日志里也照实写 0."""
        retriever = sample_retriever(max_per_doc=0, default_top_k=8)

        result = retriever.retrieve(AXIS)

        assert result.count == 8
        assert result.dropped_by_diversity == 0
        assert retriever.describe()["max_per_doc"] == 0

    def test_grouping_field_is_configurable(self) -> None:
        """按 ``topic`` 分组时剪法完全不同（3 个 cache/glossary 之间互相挤）."""
        result = sample_retriever(doc_id_field="topic", max_per_doc=1).retrieve(AXIS)

        assert result.ids() == ["c-a-01", "c-b-01", "c-a-02", "c-c-01", "c-a-03"]
        assert result.dropped_by_diversity == 3
        # 分组口径（doc_id_field）与展示口径（hit.doc_id）是两个东西：后者固定取
        # metadata["parent_doc_id"]（见 RetrievalHit 的 docstring），因此不会跟着
        # doc_id_field 走。这条断言把现状写下来，免得"doc_ids 怎么还是 3 个文档"
        # 变成一个需要读源码才能回答的问题。
        assert set(result.doc_ids) == {"doc-alpha", "doc-beta", "doc-gamma"}

    def test_records_without_parent_share_one_group(self) -> None:
        """没有 ``parent_doc_id`` 的记录自成一类（键为 ``""``），因此会互相挤.

        "它们不属于任何文档"与"它们各自是一个文档"是两件不同的事——
        后者会让一批无 parent 的记录全部通过上限（见 ``hit.doc_id`` 的取舍）。
        """
        store = flat_store(plain_records(count=2), metric="cosine")

        result = sample_retriever(store, max_per_doc=1).retrieve(AXIS)

        assert result.ids() == ["plain-00"]
        assert result.dropped_by_diversity == 1
        assert result.doc_ids == [""]

    def test_tie_within_one_doc_keeps_the_smaller_id(self) -> None:
        """同一文档内的同分记录：按 id 升序先遇到的留下（裁剪顺序与排序纪律一致）."""
        axes = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        store = flat_store(
            [
                make_record("z-dup", axes, "同文", {"parent_doc_id": "d"}),
                make_record("a-dup", axes, "同文", {"parent_doc_id": "d"}),
            ],
            metric="cosine",
        )

        result = sample_retriever(store, max_per_doc=1).retrieve(AXIS)

        assert result.ids() == ["a-dup"]
        assert result.dropped_by_diversity == 1


# --------------------------------------------------------------------------- #
# 第九步：空结果诊断
# --------------------------------------------------------------------------- #


class TestEmptyReasons:
    """四个失败成因各自有触发用例；``hits`` 是"没有失败"的哨兵."""

    def test_none_for_a_normal_result(self) -> None:
        assert sample_retriever().retrieve(AXIS).empty_reason == EMPTY_REASON_NONE

    def test_no_data_for_an_empty_store(self) -> None:
        assert sample_retriever(empty_store()).retrieve(AXIS).empty_reason == EMPTY_REASON_NO_DATA

    def test_filtered_out_when_the_filter_removes_everything(self) -> None:
        result = sample_retriever().retrieve(
            RetrievalQuery(text=AXIS, where={"topic": "不存在的话题"})
        )

        assert result.empty_reason == EMPTY_REASON_FILTERED_OUT

    def test_below_threshold_when_the_threshold_cuts_everything(self) -> None:
        result = sample_retriever().retrieve(RetrievalQuery(text=ORTHO, min_score=0.5))

        assert result.empty_reason == EMPTY_REASON_BELOW_THRESHOLD

    def test_diversity_trimmed_is_a_unit_level_branch(self) -> None:
        """**这一支端到端不可达**，因此用单元级构造触发（规格第 3.4 节写明）.

        ``_apply_diversity`` 每组至少留一条（``used >= limit`` 只对第二、三条
        之后的命中成立），所以 ``dropped_by_diversity > 0`` 时 ``kept`` 必然非空。
        保留这一支的理由是"诊断规则不该依赖某一步的内部细节"：一旦第 7 步
        改成"每组至少两条"，它就会立刻变成活路径，而那时它必须在正确的位置
        （阈值之后、兜底之前）。因此这里直接调用 ``_diagnose``。
        """
        reason = _diagnose(
            hits=(),
            count_store=8,
            candidates=8,
            filter_applied=False,
            dropped_below_threshold=0,
            dropped_by_diversity=2,
        )

        assert reason == EMPTY_REASON_DIVERSITY

    def test_diversity_is_checked_after_the_threshold(self) -> None:
        """阈值切掉了东西时，原因记在阈值头上（更上游的成因先说话）."""
        reason = _diagnose(
            hits=(),
            count_store=8,
            candidates=8,
            filter_applied=False,
            dropped_below_threshold=3,
            dropped_by_diversity=2,
        )

        assert reason == EMPTY_REASON_BELOW_THRESHOLD

    def test_filter_is_checked_before_the_threshold(self) -> None:
        """过滤把候选筛成 0 时，原因记在过滤头上（根本没东西可切）."""
        reason = _diagnose(
            hits=(),
            count_store=8,
            candidates=0,
            filter_applied=True,
            dropped_below_threshold=0,
            dropped_by_diversity=0,
        )

        assert reason == EMPTY_REASON_FILTERED_OUT

    def test_no_data_wins_over_everything(self) -> None:
        """库是空的 → 不管其它数字是多少，原因只能是 ``no_data``."""
        reason = _diagnose(
            hits=(),
            count_store=0,
            candidates=0,
            filter_applied=True,
            dropped_below_threshold=9,
            dropped_by_diversity=9,
        )

        assert reason == EMPTY_REASON_NO_DATA

    def test_filter_applied_without_drops_is_still_a_filter_problem(self) -> None:
        """开着过滤、候选非空、什么也没切掉，却一条都没排出来 → 仍是过滤器这一边."""
        reason = _diagnose(
            hits=(),
            count_store=8,
            candidates=5,
            filter_applied=True,
            dropped_below_threshold=0,
            dropped_by_diversity=0,
        )

        assert reason == EMPTY_REASON_FILTERED_OUT

    def test_defensive_fallback_reports_no_data(self) -> None:
        """库非空、无过滤、无减法却排不出结果 → 兜底归到 ``no_data``（不是任何一种"被切掉"）."""
        reason = _diagnose(
            hits=(),
            count_store=8,
            candidates=8,
            filter_applied=False,
            dropped_below_threshold=0,
            dropped_by_diversity=0,
        )

        assert reason == EMPTY_REASON_NO_DATA

    def test_hits_win_over_every_counter(self) -> None:
        """有命中就是有命中（哪怕同时记着切掉的条数）."""
        reason = _diagnose(
            hits=(make_hit_stub(),),
            count_store=8,
            candidates=8,
            filter_applied=True,
            dropped_below_threshold=1,
            dropped_by_diversity=1,
        )

        assert reason == EMPTY_REASON_NONE


def make_hit_stub() -> RetrievalHit:
    """一条最小可用的命中（只用于诊断分支的单元级构造）."""
    return RetrievalHit(record_id="c-a-01", score=0.5, rank=0)


# --------------------------------------------------------------------------- #
# 第 2 步：索引体检（空库 / 漂移 / 严格模式）
# --------------------------------------------------------------------------- #


class TestIndexChecks:
    """空库是合法状态；漂移只观测不阻断（除严格模式）."""

    def test_empty_store_returns_no_data_without_raising(self) -> None:
        result = sample_retriever(empty_store()).retrieve(AXIS)

        assert result.count == 0
        assert result.candidates == 0
        assert result.empty_reason == EMPTY_REASON_NO_DATA
        assert result.metric == "cosine"
        assert any("库是空的" in note for note in result.notes)

    def test_empty_store_still_reports_the_planned_depth(self) -> None:
        """空结果里的 ``fetch_k`` 不能是 0：否则会被误读成"这次没设深度"."""
        result = sample_retriever(empty_store()).retrieve(AXIS)

        assert result.fetch_k == 15
        assert result.query.top_k == 5

    def test_without_a_manifest_only_the_store_is_reported(self) -> None:
        result = sample_retriever().retrieve(AXIS)
        state = result.index_state

        assert state.version_id == ""
        assert state.count_store == 8
        assert state.count_manifest is None
        assert state.has_drift is False
        assert result.notes == ()

    def test_manifest_matching_the_store_has_no_drift(self) -> None:
        store = sample_store()
        manifest = sample_manifest(store)

        result = sample_retriever(store, manifest=manifest).retrieve(AXIS)

        assert result.index_state.version_id == manifest.version_id
        assert result.index_state.count_manifest == 8
        assert result.index_state.has_drift is False
        assert result.notes == ()

    def test_deleted_record_becomes_an_observable_drift(self) -> None:
        """手工删一条 → 漂移非空 + 一条注记（不是静默的质量下降）."""
        store = sample_store()
        manifest = sample_manifest(store)
        store.delete(ids=[RECORD_IDS[-1]])

        result = sample_retriever(store, manifest=manifest).retrieve(AXIS)

        assert result.index_state.has_drift is True
        assert result.index_state.count_store == 7
        assert result.index_state.count_manifest == 8
        assert RECORD_IDS[-1] in result.index_state.drift[0]
        assert "请先重建清单" in result.index_state.drift[0]
        assert result.notes[0].startswith("索引漂移 1 项：")
        assert result.count > 0

    def test_drift_is_logged_as_a_warning(self, caplog: Any) -> None:
        """漂移必须在一个不依赖调用方配合的地方留下痕迹——日志就是那个地方."""
        store = sample_store()
        manifest = sample_manifest(store)
        store.delete(ids=[RECORD_IDS[-1]])

        with caplog.at_level(logging.WARNING, logger="smart_research_agent.retrieval.retriever"):
            sample_retriever(store, manifest=manifest).retrieve(AXIS)

        warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
        assert any("检测到索引漂移 1 项" in record.getMessage() for record in warnings)
        assert any(record.name == "smart_research_agent.retrieval.retriever" for record in warnings)

    def test_strict_index_refuses_to_serve(self) -> None:
        """严格模式的选择是**拒绝服务**，而不是给出一个可能少召回的结果."""
        store = sample_store()
        manifest = sample_manifest(store)
        store.delete(ids=[RECORD_IDS[-1]])

        with pytest.raises(IndexStateError) as excinfo:
            sample_retriever(store, manifest=manifest, strict_index=True).retrieve(AXIS)

        message = str(excinfo.value)
        assert "strict_index=True 且索引漂移 1 项" in message
        assert "拒绝服务" in message
        assert "关掉 strict_index" in message

    def test_strict_index_on_a_healthy_index_still_serves(self) -> None:
        store = sample_store()

        result = sample_retriever(
            store, manifest=sample_manifest(store), strict_index=True
        ).retrieve(AXIS)

        assert result.count == 5

    def test_settings_can_turn_on_strict_mode(self, monkeypatch: Any) -> None:
        """严格模式是**只增不减**的运维开关：构造参数关不掉 settings 里打开的 True.

        它是"这台机器一律不接受漂移"的声明，而这个声明不该被某一处调用的
        默认参数悄悄撤销。
        """
        store = sample_store()
        manifest = sample_manifest(store)
        store.delete(ids=[RECORD_IDS[-1]])
        monkeypatch.setattr(settings, "retrieval_strict_index", True)

        direct = sample_retriever(store, manifest=manifest, strict_index=False)
        with pytest.raises(IndexStateError):
            direct.retrieve(AXIS)

        assembled = build_retriever(store, TableEmbedding(), manifest=manifest)
        with pytest.raises(IndexStateError):
            assembled.retrieve(AXIS)

    def test_index_state_property_never_raises(self) -> None:
        """``index_state`` 是**观测**入口：它在漂移时照常返回状态（运维要看现状）."""
        store = sample_store()
        manifest = sample_manifest(store)
        store.delete(ids=[RECORD_IDS[-1]])

        state = sample_retriever(store, manifest=manifest, strict_index=True).index_state

        assert state.has_drift is True
        assert state.count_store == 7

    def test_dimension_drift_is_reported_and_annotated(self) -> None:
        """清单记 16 维、库是 8 维 → 既是漂移，也是一条"编码器身份不符"的注记."""
        store = sample_store()
        manifest = sample_manifest(store, dimension=16)

        result = sample_retriever(store, manifest=manifest).retrieve(AXIS)

        assert "维度不一致" in result.index_state.drift[0]
        assert any("编码器维度 8 与清单记的 16 不一致" in note for note in result.notes)

    def test_metric_drift_is_reported(self) -> None:
        """换度量 → 同一批向量在两种度量下的最近邻不是同一批."""
        store = sample_store()
        manifest = sample_manifest(store, metric="ip")

        result = sample_retriever(store, manifest=manifest).retrieve(AXIS)

        assert "度量不一致" in result.index_state.drift[0]

    def test_orphan_record_in_the_store_is_reported(self) -> None:
        """库里有、清单里没有的记录也要报出来（它们不会被增量更新覆盖）."""
        store = sample_store()
        manifest = sample_manifest(store)
        store.upsert(
            [
                make_record(
                    "c-d-01",
                    (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    "清单之后新加的一条记录。",
                    {"parent_doc_id": "doc-delta", "strategy": "fixed"},
                )
            ]
        )

        result = sample_retriever(store, manifest=manifest).retrieve(AXIS)

        assert result.index_state.count_store == 9
        assert "库里有 1 条记录不在清单里" in result.index_state.drift[0]
        assert "c-d-01" in result.index_state.drift[0]

    def test_backend_drift_is_reported(self) -> None:
        """清单说它描述的是另一个后端 → 换后端可以重建，但不能与旧版共用一份清单."""
        store = sample_store()
        self_consistent = sample_manifest(store)
        manifest = build_manifest(
            self_consistent.entries,
            identity=self_consistent.identity,
            metric=self_consistent.metric,
            backend="chroma",
        )

        result = sample_retriever(store, manifest=manifest).retrieve(AXIS)

        assert "后端不一致" in result.index_state.drift[0]
        assert "'chroma'" in result.index_state.drift[0]

    def test_provider_mismatch_is_only_a_note(self) -> None:
        """编码器实现与清单记的不同只是注记：结论仍然可用.

        把它升级成异常会让"清单里 provider 名字写得不完全一样"这种无害情形
        变成一次服务不可用。
        """
        store = sample_store()
        manifest = sample_manifest(store, provider="AnotherEmbedding")

        result = sample_retriever(store, manifest=manifest).retrieve(AXIS)

        assert result.index_state.has_drift is False
        assert any("编码器实现 TableEmbedding 与清单记的 AnotherEmbedding 不同" in note
                   for note in result.notes)

    def test_fake_embedding_with_matching_identity_leaves_no_notes(self) -> None:
        """身份对得上时不该出现任何注记（否则"注记"会变成噪声）."""
        embedding = FixedVectorEmbedding(list(QUERY_VECTORS["axis"]))
        store = encoded_store(embedding, metric="cosine")
        manifest = sample_manifest(store, provider="FixedVectorEmbedding", model="fixed-v1")

        result = sample_retriever(store, embedding, manifest=manifest).retrieve(AXIS)

        assert result.notes == ()
        assert result.index_state.has_drift is False


class TestEncoderValidation:
    """第 4 步的三项校验都属于 ``IndexStateError``：问题在装配，不在那句话."""

    def test_dimension_mismatch_points_at_the_encoder(self) -> None:
        """库是 64 维（字符 n-gram）而查询编码器是 8 维 → 必须在编码后就拦下."""
        store = encoded_store(lexical_embedding())

        with pytest.raises(IndexStateError) as excinfo:
            sample_retriever(store, TableEmbedding()).retrieve(AXIS)

        message = str(excinfo.value)
        assert "查询向量 8 维、本库 64 维" in message
        assert "请用同一个提供方重建索引" in message

    def test_matching_dimension_runs_end_to_end(self) -> None:
        """同一个 64 维编码器建库 + 检索 → 跑通（校验不是"一律拒绝"）."""
        embedding = lexical_embedding()
        store = encoded_store(embedding)

        result = sample_retriever(store, embedding).retrieve(AXIS)

        assert result.index_state.dimension == 64
        assert result.count == 5

    def test_zero_vector_is_rejected(self) -> None:
        """余弦对零向量是 0/0：它会"平等地命中或不命中一切"."""
        with pytest.raises(IndexStateError) as excinfo:
            sample_retriever(embedding=FixedVectorEmbedding([0.0] * 8)).retrieve(AXIS)

        message = str(excinfo.value)
        assert "零向量" in message
        assert "余弦相似度对它是 0/0" in message

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_non_finite_components_are_rejected_with_position(self, bad: float) -> None:
        """报错要指出是第几个分量（否则 nan 是从哪来的无法定位）."""
        with pytest.raises(IndexStateError) as excinfo:
            sample_retriever(embedding=FixedVectorEmbedding([bad] + [1.0] * 7)).retrieve(AXIS)

        message = str(excinfo.value)
        assert "第 0 个分量不是有限数" in message
        assert "排序结果不可复现" in message

    def test_empty_vector_is_rejected(self) -> None:
        """空向量无法与任何向量比距离，让它进检索会得到"永远返回空集"."""
        with pytest.raises(IndexStateError) as excinfo:
            sample_retriever(embedding=FixedVectorEmbedding([], dimension=8)).retrieve(AXIS)

        assert "编码器返回了空向量" in str(excinfo.value)

    def test_non_sequence_is_rejected(self) -> None:
        """提供方把 ndarray 或字符串塞进结果时，报错要指向 ``embed`` 的实现."""
        with pytest.raises(IndexStateError) as excinfo:
            sample_retriever(embedding=NotASequenceEmbedding()).retrieve(AXIS)

        message = str(excinfo.value)
        assert "编码器返回了 str" in message
        assert "必须返回定长浮点序列" in message

    def test_non_numeric_components_are_rejected(self) -> None:
        """分量转不成浮点数时同样属于编码器的问题（不是调用方写错了查询）."""

        class StringComponents(FixedVectorEmbedding):
            def embed(self, text: str) -> list[float]:
                return ["一"] * 8  # type: ignore[list-item]

        with pytest.raises(IndexStateError) as excinfo:
            sample_retriever(embedding=StringComponents([0.0] * 8)).retrieve(AXIS)

        assert "无法转成浮点数" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# 三道减法的纯函数（直接核对"留下哪些、切掉几条"）
# --------------------------------------------------------------------------- #


class TestPureReductions:
    """这三个函数是纯函数：给定一批命中，回答"留下哪些、切掉几条"."""

    @staticmethod
    def hits() -> list[RetrievalHit]:
        """五条命中：分数 0.9 / 0.8 / 0.8 / 0.5 / 0.5，分属两个文档."""
        return [
            RetrievalHit("c-a-01", 0.9, 0, metadata={"parent_doc_id": "doc-alpha"}),
            RetrievalHit("c-a-02", 0.8, 1, metadata={"parent_doc_id": "doc-alpha"}),
            RetrievalHit("c-b-01", 0.8, 2, metadata={"parent_doc_id": "doc-beta"}),
            RetrievalHit("c-b-02", 0.5, 3, metadata={"parent_doc_id": "doc-beta"}),
            RetrievalHit("plain-01", 0.5, 4, metadata={}),
        ]

    def test_threshold_keeps_the_boundary(self) -> None:
        kept, dropped = _apply_threshold(self.hits(), 0.8)

        assert [hit.record_id for hit in kept] == ["c-a-01", "c-a-02", "c-b-01"]
        assert dropped == 2

    def test_threshold_none_keeps_everything(self) -> None:
        kept, dropped = _apply_threshold(self.hits(), None)

        assert len(kept) == 5
        assert dropped == 0

    def test_diversity_keeps_the_first_per_group(self) -> None:
        kept, dropped = _apply_diversity(self.hits(), 1, "parent_doc_id")

        assert [hit.record_id for hit in kept] == ["c-a-01", "c-b-01", "plain-01"]
        assert dropped == 2

    @pytest.mark.parametrize("limit", [None, 0])
    def test_diversity_without_a_limit_is_a_noop(self, limit: int | None) -> None:
        kept, dropped = _apply_diversity(self.hits(), limit, "parent_doc_id")

        assert len(kept) == 5
        assert dropped == 0

    def test_diversity_uses_the_configured_field(self) -> None:
        """分组键取的是 ``metadata[doc_id_field]``：换个字段名就换一种剪法."""
        kept, dropped = _apply_diversity(self.hits(), 1, "missing_field")

        assert [hit.record_id for hit in kept] == ["c-a-01"]
        assert dropped == 4

    def test_rerank_sorts_and_renumbers(self) -> None:
        """分数降序、同分按 id 升序，名次重编成 0 起连续（原 rank 被丢掉）."""
        shuffled = list(reversed(self.hits()))

        ordered = _rerank(shuffled)

        assert [hit.record_id for hit in ordered] == [
            "c-a-01",
            "c-a-02",
            "c-b-01",
            "c-b-02",
            "plain-01",
        ]
        assert [hit.rank for hit in ordered] == [0, 1, 2, 3, 4]

    def test_rerank_is_not_in_place(self) -> None:
        """纯函数：不改入参（三道减法共享同一批命中对象，就地改会互相污染）."""
        original = self.hits()
        snapshot = [hit.rank for hit in original]

        _rerank(original)

        assert [hit.rank for hit in original] == snapshot


# --------------------------------------------------------------------------- #
# 只读视图、装配与 explain
# --------------------------------------------------------------------------- #


class TestRetrieverConfiguration:
    """构造参数决定默认值；``describe()`` 是端点直接返回的东西."""

    def test_describe_key_set(self) -> None:
        payload = sample_retriever().describe()

        assert set(payload) == {
            "name",
            "top_k",
            "fetch_multiplier",
            "min_score",
            "max_per_doc",
            "doc_id_field",
            "time_field",
            "index",
        }
        assert set(payload["index"]) == {
            "version_id",
            "count_store",
            "count_manifest",
            "dimension",
            "metric",
            "has_drift",
            "drift",
        }

    def test_describe_reports_the_configured_values(self) -> None:
        payload = sample_retriever(
            default_top_k=3, fetch_multiplier=2, min_score=0.4, max_per_doc=2
        ).describe()

        assert payload["top_k"] == 3
        assert payload["fetch_multiplier"] == 2
        assert payload["min_score"] == 0.4
        assert payload["max_per_doc"] == 2
        assert payload["doc_id_field"] == "parent_doc_id"
        assert payload["index"]["count_store"] == 8

    def test_name_and_default_top_k_properties(self) -> None:
        retriever = sample_retriever(name="  手册库  ", default_top_k=7)

        assert retriever.name == "手册库"
        assert retriever.default_top_k == 7

    def test_defaults_come_from_settings(self) -> None:
        """``None`` 在这些参数上意味着"没指定，去读 settings"（默认值只有一处定义）."""
        retriever = sample_retriever()

        assert retriever.default_top_k == settings.retrieval_top_k == 5
        assert retriever.describe()["fetch_multiplier"] == settings.retrieval_fetch_multiplier
        assert retriever.describe()["min_score"] == settings.retrieval_min_score
        assert retriever.describe()["max_per_doc"] == settings.retrieval_max_per_doc

    def test_time_field_follows_settings_when_the_signature_default_is_used(
        self, monkeypatch: Any
    ) -> None:
        """签名默认值是字面量 ``created_at``，因此"等于默认值"时按"没指定"处理去读 settings.

        这一条写在 ``__init__`` 里而不是藏着：不然"我明明传了 created_at"
        这种疑问没法解释。
        """
        monkeypatch.setattr(settings, "retrieval_time_field", "published_at")

        assert sample_retriever().describe()["time_field"] == "published_at"
        assert sample_retriever(time_field="indexed_at").describe()["time_field"] == "indexed_at"

    @pytest.mark.parametrize(
        ("overrides", "fragment"),
        [
            ({"default_top_k": 0}, "default_top_k 必须是 >= 1 的整数"),
            ({"default_top_k": True}, "default_top_k 必须是 >= 1 的整数"),
            ({"default_top_k": MAX_FETCH_K + 1}, "超过上限"),
            ({"fetch_multiplier": 0}, "fetch_multiplier 必须是 >= 1 的整数"),
            ({"fetch_multiplier": True}, "fetch_multiplier 必须是 >= 1 的整数"),
            ({"min_score": float("nan")}, "min_score 必须是有限数"),
            ({"min_score": "0.5"}, "min_score 必须是数字或 None"),
            ({"max_per_doc": -1}, "max_per_doc 必须是 >= 0 的整数"),
            ({"max_per_doc": True}, "max_per_doc 必须是 >= 0 的整数"),
            ({"doc_id_field": "  "}, "doc_id_field 必须是非空字符串"),
            ({"name": ""}, "name 必须是非空字符串"),
        ],
    )
    def test_constructor_validates_every_number(
        self, overrides: dict[str, Any], fragment: str
    ) -> None:
        """构造期的错误也属于"调用方改一处就能走通"，因此是 ``QueryError``."""
        with pytest.raises(QueryError) as excinfo:
            sample_retriever(**overrides)

        assert fragment in str(excinfo.value)

    def test_build_retriever_reads_settings(self) -> None:
        """装配入口从 settings 取缺省值，并显式指过去 ``time_field`` / ``strict_index``."""
        payload = build_retriever(sample_store(), TableEmbedding()).describe()

        assert payload["top_k"] == settings.retrieval_top_k
        assert payload["fetch_multiplier"] == settings.retrieval_fetch_multiplier
        assert payload["time_field"] == settings.retrieval_time_field

    def test_build_retriever_overrides_win(self) -> None:
        """显式传参永远赢过 settings（否则演示脚本没法试不同的问法）."""
        payload = build_retriever(
            sample_store(), TableEmbedding(), default_top_k=3, min_score=0.6
        ).describe()

        assert payload["top_k"] == 3
        assert payload["min_score"] == 0.6


class TestRetrieveManyAndExplain:
    """批量入口与人类可读诊断."""

    def test_retrieve_many_keeps_the_input_order(self) -> None:
        results = sample_retriever().retrieve_many([AXIS, TILT, ORTHO])

        assert [result.query.text for result in results] == [AXIS, TILT, ORTHO]
        assert [result.count for result in results] == [5, 5, 5]

    def test_retrieve_many_accepts_query_objects(self) -> None:
        results = sample_retriever().retrieve_many(
            [RetrievalQuery(text=AXIS, where={"strategy": "structural"})]
        )

        assert results[0].candidates == 4

    def test_retrieve_many_on_empty_input(self) -> None:
        assert sample_retriever().retrieve_many([]) == []

    def test_explain_starts_with_the_four_questions(self) -> None:
        retriever = sample_retriever()

        lines = retriever.explain(retriever.retrieve(AXIS))

        assert lines[0].startswith("索引版本：")
        assert lines[1].startswith("过滤条件：")
        assert lines[2].startswith("条数：")
        assert lines[3].startswith("空结果原因：")

    def test_explain_adds_a_spelling_check_when_a_field_is_unknown(self) -> None:
        """字段拼写检查需要"库里出现过哪些字段"这个全集，因此只能由检索器给出."""
        retriever = sample_retriever()
        result = retriever.retrieve(RetrievalQuery(text=AXIS, where={"stratgey": "structural"}))

        lines = retriever.explain(result)

        assert any("字段拼写检查：['stratgey'] 在本次库里从未出现过" in line for line in lines)
        assert any("发现不了「值写错」" in line for line in lines)

    def test_explain_confirms_known_fields(self) -> None:
        retriever = sample_retriever()
        result = retriever.retrieve(RetrievalQuery(text=AXIS, where={"strategy": "structural"}))

        lines = retriever.explain(result)

        assert any("过滤用到的字段在库里都出现过" in line for line in lines)

    def test_explain_skips_the_spelling_check_without_where(self) -> None:
        retriever = sample_retriever()

        lines = retriever.explain(retriever.retrieve(AXIS))

        assert not any("字段拼写检查" in line for line in lines)

    def test_explain_adds_drift_advice(self) -> None:
        """漂移处置建议需要知道"这次到底漂在哪"，因此也只能由检索器给出."""
        store = sample_store()
        manifest = sample_manifest(store)
        store.delete(ids=[RECORD_IDS[-1]])
        retriever = sample_retriever(store, manifest=manifest)

        lines = retriever.explain(retriever.retrieve(AXIS))

        assert any("漂移处置建议" in line for line in lines)
        assert any("manifest_from_store" in line for line in lines)

    def test_explain_always_names_the_grouping_contract(self) -> None:
        """分组口径必须逐次回显：它决定"为什么这条被挤掉了"."""
        retriever = sample_retriever(doc_id_field="topic", time_field="indexed_at")

        line = retriever.explain(retriever.retrieve(AXIS))[-1]

        assert "metadata['topic']" in line
        assert "'indexed_at'" in line
        assert "默认 top_k=5" in line

    def test_explain_does_not_mutate_the_result(self) -> None:
        retriever = sample_retriever()
        result = retriever.retrieve(AXIS)
        before = result.notes

        retriever.explain(result)

        assert result.notes == before
