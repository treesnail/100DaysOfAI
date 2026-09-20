"""day067 混合检索的**过滤一致性**测试：同一份 ``where`` 必须在两路都落刀.

这一组用例的立足点是一个很安静的失败模式（本课最值得讲的坑之一）：

```text
只过滤向量路    用户 where={"strategy": "structural"}
                向量路只取 structural 的；关键词路不过滤 → 它把 fixed/semantic
                的记录也捞了回来；融合之后**结果里出现了明确被排除掉的记录**
                而报告里 filter_applied=True（"过滤生效了"），没有任何异常
```

"多而不报"比"少而不报"更难发现：少了几条还能靠对比两次运行看出来，
多了几条在报告里看起来完全正常（没人会逐条核对每条命中的 ``strategy``）。
因此这里的每一条用例都**独立重算**一遍期望的候选集合（用样本元数据手写谓词，
不引用实现里的任何过滤器），然后断言：

```text
1. 混合结果的每一条命中都满足条件（没有任何一条漏网）
2. 两路各自取回的候选都满足条件（不是"只有融合那一步筛了一下"）
3. 两路的候选数与期望集合的规模对得上（同一份语义 → 同一个候选集）
```

全部离线、确定性、零网络。
"""

from __future__ import annotations

from typing import Any

import pytest

from smart_research_agent.retrieval.errors import QueryError
from smart_research_agent.retrieval.filters import (
    combine_where,
    describe_conditions,
    time_range_clause,
    unknown_filter_fields,
)
from smart_research_agent.retrieval.types import (
    CHANNEL_BM25,
    CHANNEL_VECTOR,
    RetrievalQuery,
    TimeRange,
)
from smart_research_agent.vectorstore.base import MAX_TOP_K
from tests.hybrid_samples import (
    MISSING_CREATED_AT_ID,
    QUERY_BOTH,
    QUERY_CODE,
    QUERY_EXACT,
    RECORD_IDS,
    hybrid_lexical,
    hybrid_retriever,
    hybrid_vector_retriever,
    metadata_field_names,
    sample_metadata,
)

#: 过滤用例：``(子句, 独立写出的谓词)``.
#:
#: 谓词是**测试侧**的期望值来源（手写在样本的元数据上），与实现里的
#: ``vectorstore.compile_filter`` 是两条独立路径——两边一致才说明"同一份语义"。
FILTER_CASES: tuple[tuple[dict[str, Any], Any], ...] = (
    ({"strategy": "semantic"}, lambda meta: meta["strategy"] == "semantic"),
    ({"strategy": "structural"}, lambda meta: meta["strategy"] == "structural"),
    ({"parent_doc_id": "doc-trouble"}, lambda meta: meta["parent_doc_id"] == "doc-trouble"),
    (
        {"topic": {"$in": ["errors", "glossary"]}},
        lambda meta: meta["topic"] in ("errors", "glossary"),
    ),
    ({"topic": {"$ne": "errors"}}, lambda meta: meta["topic"] != "errors"),
    ({"index": {"$lt": 2}}, lambda meta: meta["index"] < 2),
    (
        {"$and": [{"topic": "errors"}, {"strategy": "semantic"}]},
        lambda meta: meta["topic"] == "errors" and meta["strategy"] == "semantic",
    ),
    (
        {"$or": [{"parent_doc_id": "doc-manual"}, {"strategy": "semantic"}]},
        lambda meta: meta["parent_doc_id"] == "doc-manual" or meta["strategy"] == "semantic",
    ),
    (
        {"created_at": {"$gte": "2026-09-15"}},
        lambda meta: meta.get("created_at", "") >= "2026-09-15",
    ),
    (
        {"created_at": {"$lte": "2026-09-10"}},
        lambda meta: "created_at" in meta and meta["created_at"] <= "2026-09-10",
    ),
    (
        {"$and": [{"strategy": "semantic"}, {"created_at": {"$lte": "2026-09-15"}}]},
        lambda meta: meta["strategy"] == "semantic"
        and meta.get("created_at", "") <= "2026-09-15"
        and "created_at" in meta,
    ),
)


def matching_ids(predicate: Any, *, require_time: bool = False) -> set[str]:
    """独立重算的期望命中 id 集合（谓词写在测试侧，不引用实现）."""
    metadata = sample_metadata()
    return {record_id for record_id, meta in metadata.items() if predicate(meta)}


class TestFilterConsistencyAcrossChannels:
    """同一份子句在两路取到**同一个候选集**，融合之后没有任何漏网之鱼. """

    @pytest.mark.parametrize(("where", "predicate"), FILTER_CASES)
    def test_no_hybrid_hit_violates_the_filter(self, where: dict, predicate: Any) -> None:
        """最要紧的一条：混合结果的每一条都必须满足条件（否则过滤没在两路都落刀）. """
        result = hybrid_retriever().retrieve(RetrievalQuery(text=QUERY_EXACT, where=where))
        expected = matching_ids(predicate)

        assert set(result.ids()) <= expected
        for hit in result.hits:
            assert predicate(hit.metadata)
            assert hit.record_id in expected

    @pytest.mark.parametrize(("where", "predicate"), FILTER_CASES)
    def test_both_channels_return_only_matching_records(self, where: dict, predicate: Any) -> None:
        """两路**各自**都只返回满足条件的记录（不是"最后融合时补一刀"）.

        向量路用 ``fetch_k=MAX_TOP_K`` 取全部候选；关键词路同样取全部。
        """
        expected = matching_ids(predicate)
        vector_hits = hybrid_vector_retriever().retrieve(
            RetrievalQuery(text=QUERY_EXACT, top_k=MAX_TOP_K, fetch_k=MAX_TOP_K, where=where)
        )
        lexical = hybrid_lexical().search(QUERY_BOTH, top_k=MAX_TOP_K, where=where)

        assert set(vector_hits.ids()) <= expected
        assert set(lexical.ids()) <= expected

    @pytest.mark.parametrize(("where", "predicate"), FILTER_CASES)
    def test_the_two_channels_agree_on_the_candidate_count(
        self, where: dict, predicate: Any
    ) -> None:
        """两路的候选数必须等于**同一个**期望集合的规模（同一份语义的两个投影）.

        向量路：取满深度时候选数 = 期望集合规模（库里有几条就返几条）。
        关键词路：``candidates`` 是"过滤后待打分的篇数"，也就是同一个集合的规模。
        """
        expected = matching_ids(predicate)
        vector_hits = hybrid_vector_retriever().retrieve(
            RetrievalQuery(text=QUERY_EXACT, top_k=MAX_TOP_K, fetch_k=MAX_TOP_K, where=where)
        )
        lexical = hybrid_lexical().search(QUERY_BOTH, top_k=MAX_TOP_K, where=where)

        assert len(vector_hits.ids()) == len(expected)
        assert lexical.candidates == len(expected)

    @pytest.mark.parametrize(("where", "predicate"), FILTER_CASES)
    def test_hybrid_channel_candidates_stay_within_the_expected_set(
        self, where: dict, predicate: Any
    ) -> None:
        result = hybrid_retriever().retrieve(RetrievalQuery(text=QUERY_EXACT, where=where))
        expected = matching_ids(predicate)

        assert result.channel_candidates[CHANNEL_VECTOR] == min(len(expected), result.fetch_k)
        assert result.channel_candidates[CHANNEL_BM25] <= len(expected)
        assert result.candidates <= len(expected)
        assert result.filter_applied is True

    def test_an_excluded_record_is_never_pulled_back_by_the_other_channel(self) -> None:
        """**反面对照**：``h-t-01`` 只被关键词路召回，因此"过滤在两路都落刀"这件事
        对它尤其危险——只过滤向量路时它一定会被抬回来（向量路压根没召回它）。"""
        excluded = {"strategy": "structural"}
        result = hybrid_retriever().retrieve(
            RetrievalQuery(text=QUERY_EXACT, where=excluded)
        )

        assert "h-t-01" not in result.ids()
        assert result.channel_candidates[CHANNEL_BM25] == 0
        assert result.channel_candidates[CHANNEL_VECTOR] == 4

    def test_the_keyword_channel_is_actually_trimmed(self) -> None:
        """收紧到 ``doc-trouble`` 时关键词路照旧有命中（说明它确实在工作）. """
        result = hybrid_retriever().retrieve(
            RetrievalQuery(text=QUERY_EXACT, where={"parent_doc_id": "doc-trouble"})
        )

        assert result.channel_candidates[CHANNEL_BM25] == 1
        assert result.ids()[0] == "h-t-01"

    def test_the_filter_applies_to_the_keyword_channel_too(self) -> None:
        """同一句话、只加一个过滤条件：关键词路的候选从 1 条变成 0 条. """
        unfiltered = hybrid_retriever().retrieve(QUERY_EXACT)
        filtered = hybrid_retriever().retrieve(
            RetrievalQuery(text=QUERY_EXACT, where={"parent_doc_id": "doc-concept"})
        )

        assert unfiltered.channel_candidates[CHANNEL_BM25] == 1
        assert filtered.channel_candidates[CHANNEL_BM25] == 0


class TestFilterAccounting:
    """``filter_applied`` 与 ``candidates`` 的口径（``{}`` / ``None`` / 非空三态）. """

    def test_none_means_no_filter(self) -> None:
        result = hybrid_retriever().retrieve(RetrievalQuery(text=QUERY_EXACT, where=None))

        assert result.filter_applied is False
        assert result.candidates == len(RECORD_IDS)

    def test_empty_dict_means_no_filter(self) -> None:
        result = hybrid_retriever().retrieve(RetrievalQuery(text=QUERY_EXACT, where={}))

        assert result.filter_applied is False
        assert result.candidates == len(RECORD_IDS)

    def test_a_narrowing_filter_reduces_the_union(self) -> None:
        everything = hybrid_retriever().retrieve(QUERY_EXACT)
        narrowed = hybrid_retriever().retrieve(
            RetrievalQuery(text=QUERY_EXACT, where={"parent_doc_id": "doc-concept"})
        )

        assert narrowed.candidates < everything.candidates
        assert narrowed.candidates == 4

    def test_a_filter_that_matches_nothing_reports_zero_candidates(self) -> None:
        result = hybrid_retriever().retrieve(
            RetrievalQuery(text=QUERY_EXACT, where={"topic": "nope"})
        )

        assert result.candidates == 0
        assert result.channel_candidates == {CHANNEL_VECTOR: 0, CHANNEL_BM25: 0}
        assert result.empty_reason == "filtered_out"

    def test_the_union_never_exceeds_the_two_channel_sum(self) -> None:
        for where, _predicate in FILTER_CASES:
            result = hybrid_retriever().retrieve(
                RetrievalQuery(text=QUERY_BOTH, where=where)
            )
            total = sum(result.channel_candidates.values())
            assert result.candidates <= total
            assert result.fusion["deduped"] == total - result.candidates


class TestTimeRangeFiltering:
    """时间范围：闭区间、按 ISO 文本比较、字段缺失被排除——三条约定在两路都成立. """

    def test_a_closed_interval_excludes_the_record_without_a_timestamp(self) -> None:
        """``h-t-03`` 没有 ``created_at``：两路都会把它排除（这不是 bug，是约定三）.

        闭区间必须写成 ``$and`` 包两条：``vectorstore`` 明确拒绝"同一字段多个运算符"，
        因此 ``{"created_at": {"$gte": …, "$lte": …}}`` **本身**就是非法子句
        （见 ``test_two_operators_on_one_field_is_invalid``）。
        """
        result = hybrid_retriever().retrieve(
            RetrievalQuery(
                text=QUERY_CODE,
                where={
                    "$and": [
                        {"created_at": {"$gte": "2026-01-01"}},
                        {"created_at": {"$lte": "2026-12-31"}},
                    ]
                },
            )
        )

        assert MISSING_CREATED_AT_ID not in result.ids()
        assert result.channel_candidates[CHANNEL_BM25] == 0

    def test_two_operators_on_one_field_is_invalid(self) -> None:
        """同一字段的多运算符条件必须用 ``$and`` 表达（这一条也是时间范围的写法约束）. """
        with pytest.raises(QueryError) as excinfo:
            hybrid_retriever().retrieve(
                RetrievalQuery(
                    text=QUERY_CODE,
                    where={"created_at": {"$gte": "2026-01-01", "$lte": "2026-12-31"}},
                )
            )

        assert "$and" in str(excinfo.value)

    def test_time_range_via_the_type_still_excludes_it(self) -> None:
        """用 ``time_range`` 参数（而不是 ``where``）走的仍是同一份约定. """
        result = hybrid_retriever().retrieve(
            RetrievalQuery(
                text=QUERY_BOTH,
                time_range=TimeRange(field="created_at", start="2026-01-01", end="2026-12-31"),
            )
        )

        assert MISSING_CREATED_AT_ID not in result.ids()
        assert result.filter_applied is True

    def test_half_open_interval_boundaries(self) -> None:
        """按 ISO 文本比较：``>= 2026-09-15`` 包含 09-15 当天的那条. """
        included = hybrid_retriever().retrieve(
            RetrievalQuery(text=QUERY_BOTH, where={"created_at": {"$gte": "2026-09-15"}})
        )
        excluded = hybrid_retriever().retrieve(
            RetrievalQuery(text=QUERY_BOTH, where={"created_at": {"$gte": "2026-09-16"}})
        )

        assert "h-t-02" in included.ids()
        assert "h-t-02" not in excluded.ids()

    def test_time_clause_shape(self) -> None:
        """双端必须用 ``$and`` 包一层（``vectorstore`` 拒绝同字段多运算符）. """
        clause = time_range_clause(
            TimeRange(field="created_at", start="2026-09-01", end="2026-09-30")
        )

        assert clause == {
            "$and": [
                {"created_at": {"$gte": "2026-09-01"}},
                {"created_at": {"$lte": "2026-09-30"}},
            ]
        }

    def test_combine_where_returns_the_same_object_shape_for_both_channels(self) -> None:
        """两路拿到的是**同一份**子句（模块 docstring 里那个坑的代码形状）. """
        combined = combine_where(
            {"topic": "errors"},
            TimeRange(field="created_at", start="2026-09-01"),
        )

        assert combined == {
            "$and": [{"topic": "errors"}, {"created_at": {"$gte": "2026-09-01"}}]
        }

    def test_describe_conditions_mentions_both_parts(self) -> None:
        rendered = describe_conditions(
            {"topic": "errors"}, TimeRange(field="created_at", start="2026-09-01")
        )

        assert "topic" in rendered
        assert "created_at" in rendered
        assert " 且 " in rendered


class TestFilterConflicts:
    """调用方写错条件时的四种报错（都属于 ``QueryError`` 那一族）. """

    def test_where_and_time_range_on_the_same_field_is_rejected(self) -> None:
        with pytest.raises(QueryError) as excinfo:
            hybrid_retriever().retrieve(
                RetrievalQuery(
                    text=QUERY_EXACT,
                    where={"created_at": {"$gte": "2026-01-01"}},
                    time_range=TimeRange(field="created_at", start="2026-06-01"),
                )
            )

        assert "过滤条件冲突" in str(excinfo.value)

    def test_the_conflict_is_reported_even_on_an_empty_store(self) -> None:
        """库是空的时候也要报（参数错误不该被"恰好没数据"掩盖）——day066 的同一条纪律. """
        from smart_research_agent.retrieval.hybrid import HybridRetriever
        from smart_research_agent.retrieval.lexical import LexicalIndex
        from tests.hybrid_samples import empty_store

        hybrid = HybridRetriever(hybrid_vector_retriever(empty_store()), LexicalIndex())

        with pytest.raises(QueryError):
            hybrid.retrieve(
                RetrievalQuery(
                    text=QUERY_EXACT,
                    where={"created_at": {"$gte": "2026-01-01"}},
                    time_range=TimeRange(field="created_at", start="2026-06-01"),
                )
            )

    @pytest.mark.parametrize(
        "where",
        [
            {"topic": {"$bad": "errors"}},
            {"$and": []},
            {"topic": {"$gt": 1, "$lt": 2}},
            {"topic": {"$in": []}},
            {"$xor": [{"topic": "errors"}]},
        ],
    )
    def test_invalid_where_syntax_is_rejected(self, where: dict) -> None:
        with pytest.raises(QueryError):
            hybrid_retriever().retrieve(RetrievalQuery(text=QUERY_EXACT, where=where))

    def test_non_dict_where_is_rejected(self) -> None:
        with pytest.raises(QueryError):
            hybrid_retriever().retrieve(RetrievalQuery(text=QUERY_EXACT, where=["topic"]))

    def test_inverted_time_range_is_rejected(self) -> None:
        with pytest.raises(QueryError) as excinfo:
            TimeRange(field="created_at", start="2026-09-30", end="2026-09-01")

        assert "倒置" in str(excinfo.value)


class TestFilterDiagnostics:
    """拼错的字段名要让两路一起空掉，而报告必须指出"该改的是字段名". """

    def test_an_unknown_field_empties_both_channels(self) -> None:
        result = hybrid_retriever().retrieve(
            RetrievalQuery(text=QUERY_EXACT, where={"stratgey": "structural"})
        )

        assert result.channel_candidates == {CHANNEL_VECTOR: 0, CHANNEL_BM25: 0}
        assert result.empty_reason == "filtered_out"

    def test_the_explain_output_names_the_unknown_field(self) -> None:
        hybrid = hybrid_retriever()
        result = hybrid.retrieve(
            RetrievalQuery(text=QUERY_EXACT, where={"stratgey": "structural"})
        )
        joined = "\n".join(hybrid.explain(result))

        assert "字段拼写检查" in joined
        assert "stratgey" in joined
        assert "两路会同时空掉" in joined

    def test_a_known_field_passes_the_spelling_check(self) -> None:
        hybrid = hybrid_retriever()
        result = hybrid.retrieve(
            RetrievalQuery(text=QUERY_EXACT, where={"strategy": "semantic"})
        )
        joined = "\n".join(hybrid.explain(result))

        assert "字段拼写检查：过滤用到的字段在库里都出现过" in joined

    def test_no_spelling_check_without_a_where_clause(self) -> None:
        hybrid = hybrid_retriever()
        joined = "\n".join(hybrid.explain(hybrid.retrieve(QUERY_EXACT)))

        assert "字段拼写检查" not in joined

    def test_unknown_filter_fields_helper_agrees_with_the_sample(self) -> None:
        """测试侧的字段全集与实现用的那一份必须一致（否则拼写检查会误报）."""
        known = metadata_field_names()

        assert "strategy" in known
        assert "created_at" in known
        assert unknown_filter_fields({"stratgey": 1}, known) == ["stratgey"]
        assert unknown_filter_fields({"strategy": 1}, known) == []

    def test_the_check_cannot_detect_a_wrong_value(self) -> None:
        """它能发现"整个库里都没有这个键"，但**发现不了**"键对、值错". """
        hybrid = hybrid_retriever()
        result = hybrid.retrieve(
            RetrievalQuery(text=QUERY_EXACT, where={"strategy": "structual"})
        )
        joined = "\n".join(hybrid.explain(result))

        assert "过滤用到的字段在库里都出现过" in joined
        assert result.channel_candidates == {CHANNEL_VECTOR: 0, CHANNEL_BM25: 0}
