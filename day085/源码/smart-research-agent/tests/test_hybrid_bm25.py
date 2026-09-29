"""day067 ``retrieval.lexical`` 的单元测试：BM25 的算术、边界与两个参数的方向.

这一层最容易被写成"跑一遍看看"——因为 BM25 的分数看起来就是一串浮点数。
因此这里用**两种**独立手段钉住它：

```text
1. 独立实现     brute_force_bm25() 只用 math.log 与手写的 tf/df/avgdl 重算一遍
               （不引用 lexical 里的任何函数），两边必须逐位相同
2. 结构性不变量  词越稀有权重越大（idf 关于 df 单调降）、k1 越大词频收益越大、
               b 越大长文档被压得越狠——三条方向都不依赖具体数字
```

期望值里只有"名次"与"有没有命中"是用手算常量写死的（它们由公式直接推出），
分数一律用 ``rel=1e-9`` 与独立实现比较——**这样实现改错时用例不会跟着一起错**。

参数扫描（k1 / b）的用例刻意用一份**受控小语料**（同一条词在不同文档里的
词频与长度都不一样），因为本课样本里没有这种 tf/长度梯度。

全部离线、确定性、零网络。
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from smart_research_agent.retrieval.errors import LexicalError, QueryError
from smart_research_agent.retrieval.lexical import (
    DEFAULT_B,
    DEFAULT_K1,
    BM25Params,
    LexicalDocument,
    LexicalHit,
    LexicalIndex,
    LexicalSearchResult,
    _idf,
    tokenize,
)
from smart_research_agent.vectorstore.base import MAX_TOP_K
from tests.hybrid_samples import (
    EXPECTED_AVGDL,
    EXPECTED_LEXICAL_IDS,
    EXPECTED_LEXICAL_SCORES,
    EXPECTED_TOTAL_TERMS,
    EXPECTED_VOCABULARY_SIZE,
    QUERY_BOTH,
    QUERY_CODE,
    QUERY_EXACT,
    QUERY_SEMANTIC,
    QUERY_SEMANTIC_TERM_COUNT,
    RECORD_IDS,
    documents,
    empty_store,
    hybrid_lexical,
    hybrid_store,
)

# --------------------------------------------------------------------------- #
# 独立实现（与 lexical 无关的第二份算术）
# --------------------------------------------------------------------------- #


def brute_force_bm25(
    index: LexicalIndex,
    text: str,
    *,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
) -> list[tuple[str, float]]:
    """用**手写的公式**重算一遍 BM25（只用 ``math.log``、``list.count`` 与字典）.

    ```text
    idf   = log(1 + (N - df + 0.5) / (df + 0.5))
    score = Σ_t idf · tf · (k1 + 1) / (tf + k1 · (1 - b + b · dl / avgdl))
    ```

    ``df`` 与 ``avgdl`` 都由**全库**统计（不受 where 影响）——这一条也是被测的性质：
    如果实现改成"在过滤后的子集上重算 df"，这里的数就会对不上。
    """
    terms = sorted(set(tokenize(text)))
    corpus = {record_id: tokenize(index.document(record_id).text) for record_id in index.ids()}
    total_docs = len(corpus)
    avgdl = sum(len(tokens) for tokens in corpus.values()) / total_docs
    scored: list[tuple[str, float]] = []
    for record_id, tokens in corpus.items():
        length = len(tokens)
        score = 0.0
        for term in terms:
            tf = tokens.count(term)
            if not tf:
                continue
            df = sum(1 for other in corpus.values() if term in other)
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / avgdl))
        if score > 0.0:
            scored.append((record_id, score))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored


def graded_index(*, params: BM25Params | None = None) -> LexicalIndex:
    """一份**受控梯度**语料：同一个词元的词频与文档长度都可控.

    ```text
    heavy   同一个词元出现 4 次、文档最长   → 词频收益与长度惩罚的"重"一侧
    light   同一个词元出现 1 次、文档最短   → 另一侧
    middle  出现 2 次、长度居中
    ```
    """
    return LexicalIndex.build(
        [
            LexicalDocument(record_id="heavy", text="缓存缓存缓存缓存"),
            LexicalDocument(record_id="middle", text="缓存缓存检索"),
            LexicalDocument(record_id="light", text="缓存"),
        ],
        params=params,
    )


def score_of(index: LexicalIndex, query: str, record_id: str) -> float:
    """一次检索里某条记录的分数（没命中时 ``-1.0``，便于断言"没被召回"）."""
    for hit in index.search(query, top_k=MAX_TOP_K).hits:
        if hit.record_id == record_id:
            return hit.score
    return -1.0


def ratio(index: LexicalIndex, query: str, heavy: str, light: str) -> float:
    """两条记录分数的比值（参数扫描看的就是它的方向）."""
    return score_of(index, query, heavy) / score_of(index, query, light)


# --------------------------------------------------------------------------- #
# BM25Params
# --------------------------------------------------------------------------- #


class TestBM25Params:
    """两个参数的构造期校验：非法取值当场拒，合法取值收敛成 float. """

    @pytest.mark.parametrize(
        ("k1", "b"),
        [(1.5, 0.75), (0.0, 0.0), (0.0, 1.0), (2, 0.5), (10.0, 1.0), (0.5, 0)],
    )
    def test_accepts_the_documented_range(self, k1: float, b: float) -> None:
        params = BM25Params(k1=k1, b=b)

        assert isinstance(params.k1, float)
        assert isinstance(params.b, float)
        assert params.k1 == float(k1)
        assert params.b == float(b)

    def test_defaults_are_the_classic_robertson_values(self) -> None:
        """缺省 1.5 / 0.75 是 Robertson & Zaragoza 的经典取值（不是本课调的）. """
        params = BM25Params()

        assert params.k1 == 1.5
        assert params.b == 0.75
        assert DEFAULT_K1 == 1.5
        assert DEFAULT_B == 0.75

    @pytest.mark.parametrize(
        "k1",
        [-1, -0.001, "1.5", None, True, False, float("nan"), float("inf"), [1.5], {"k1": 1.5}],
    )
    def test_rejects_bad_k1(self, k1: Any) -> None:
        with pytest.raises(QueryError) as excinfo:
            BM25Params(k1=k1)

        assert "k1" in str(excinfo.value)

    @pytest.mark.parametrize(
        "b", [-0.1, 1.0001, 1.5, -1, "0.5", None, True, float("nan"), float("-inf")]
    )
    def test_rejects_bad_b(self, b: Any) -> None:
        with pytest.raises(QueryError) as excinfo:
            BM25Params(b=b)

        assert "b" in str(excinfo.value)

    def test_b_out_of_range_message_explains_the_denominator(self) -> None:
        """b 越界的报错必须说清后果（分母变负 → 长文档反而分高）. """
        with pytest.raises(QueryError) as excinfo:
            BM25Params(b=1.5)

        message = str(excinfo.value)
        assert "分母" in message
        assert "[0, 1]" in message

    def test_to_dict_and_summary(self) -> None:
        params = BM25Params(k1=1.2, b=0.6)

        assert params.to_dict() == {"k1": 1.2, "b": 0.6}
        assert params.summary_line() == "BM25(k1=1.2, b=0.6)"


# --------------------------------------------------------------------------- #
# IDF：恒正、单调、可核对
# --------------------------------------------------------------------------- #


class TestIdf:
    """``log(1 + …)`` 的那个 1 换来了"恒正"，这条性质支撑着"0 分"的唯一含义. """

    @pytest.mark.parametrize(
        ("total_docs", "doc_freq"),
        [
            (1, 1),
            (2, 1),
            (2, 2),
            (5, 1),
            (5, 5),
            (10, 1),
            (10, 9),
            (10, 10),
            (100, 1),
            (100, 50),
            (100, 100),
            (1000, 999),
        ],
    )
    def test_idf_is_always_positive(self, total_docs: int, doc_freq: int) -> None:
        """**恒正**是这套写法的全部理由：经典写法在 df > N/2 时给负数. """
        assert _idf(total_docs, doc_freq) > 0.0

    @pytest.mark.parametrize("total_docs", [3, 10, 100])
    def test_idf_decreases_as_doc_freq_grows(self, total_docs: int) -> None:
        """词越常见权重越小（单调，不只是"大致"）. """
        values = [_idf(total_docs, df) for df in range(1, total_docs + 1)]

        assert values == sorted(values, reverse=True)
        assert values[0] > values[-1]

    @pytest.mark.parametrize("doc_freq", [1, 2, 5])
    def test_idf_grows_with_the_corpus(self, doc_freq: int) -> None:
        """同一份 df 在更大的库里更"稀有"（N 单调增）. """
        values = [_idf(total_docs, doc_freq) for total_docs in (doc_freq, 10, 100, 1000)]

        assert values == sorted(values)

    def test_idf_matches_the_closed_form(self) -> None:
        """与手写公式逐位相同（``log(1 + (N-df+0.5)/(df+0.5))``）. """
        for total_docs, doc_freq in ((10, 1), (10, 5), (1, 1), (100, 99)):
            expected = math.log(1.0 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            assert _idf(total_docs, doc_freq) == pytest.approx(expected, rel=1e-15)

    def test_idf_of_a_unique_term_has_a_hand_computable_value(self) -> None:
        """``df=1``、``N=10`` 时是 ``log(1 + 9.5/1.5)``——手算 == 实现. """
        assert _idf(10, 1) == pytest.approx(math.log(1 + 9.5 / 1.5), rel=1e-15)
        assert _idf(10, 1) == pytest.approx(1.992430, rel=1e-6)


# --------------------------------------------------------------------------- #
# 两副小形状的校验
# --------------------------------------------------------------------------- #


class TestLexicalDocument:
    """一篇文档的三项校验与它的投影. """

    def test_accepts_a_normal_document(self) -> None:
        document = LexicalDocument(record_id="c-1", text="缓存", metadata={"topic": "cache"})

        assert document.record_id == "c-1"
        assert document.text == "缓存"
        assert document.metadata == {"topic": "cache"}

    def test_empty_text_is_allowed(self) -> None:
        """空串是**一个事实**（"这条没有可索引的正文"），不是错误. """
        assert LexicalDocument(record_id="c-1", text="").text == ""

    @pytest.mark.parametrize("record_id", ["", "   ", None, 123, ["c-1"]])
    def test_rejects_bad_record_id(self, record_id: Any) -> None:
        with pytest.raises(QueryError) as excinfo:
            LexicalDocument(record_id=record_id, text="缓存")

        assert "record_id" in str(excinfo.value)

    @pytest.mark.parametrize("text", [None, 123, ["缓存"], b"cache"])
    def test_rejects_non_string_text(self, text: Any) -> None:
        with pytest.raises(QueryError) as excinfo:
            LexicalDocument(record_id="c-1", text=text)

        assert "text" in str(excinfo.value)

    @pytest.mark.parametrize("text", [" ", "\n", "\t  "])
    def test_rejects_whitespace_only_text(self, text: str) -> None:
        with pytest.raises(QueryError) as excinfo:
            LexicalDocument(record_id="c-1", text=text)

        assert "只有空白" in str(excinfo.value)

    def test_rejects_non_dict_metadata(self) -> None:
        with pytest.raises(QueryError) as excinfo:
            LexicalDocument(record_id="c-1", text="缓存", metadata=["topic"])

        assert "metadata" in str(excinfo.value)

    def test_to_dict_omits_the_body(self) -> None:
        payload = LexicalDocument(record_id="c-1", text="缓存", metadata={"a": 1}).to_dict()

        assert payload == {"record_id": "c-1", "char_count": 2, "metadata": {"a": 1}}


class TestLexicalHit:
    """一条命中的校验、通道属性与投影. """

    def test_channel_is_always_bm25(self) -> None:
        """``channel`` 是属性而不是字段：它在这条路上恒为 ``bm25``. """
        hit = LexicalHit(record_id="c-1", score=1.5, rank=0)

        assert hit.channel == "bm25"

    @pytest.mark.parametrize("record_id", ["", "  ", None, 7])
    def test_rejects_bad_record_id(self, record_id: Any) -> None:
        with pytest.raises(QueryError):
            LexicalHit(record_id=record_id, score=1.0, rank=0)

    @pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf"), "1.0", None])
    def test_rejects_bad_score(self, score: Any) -> None:
        with pytest.raises(QueryError):
            LexicalHit(record_id="c-1", score=score, rank=0)

    @pytest.mark.parametrize("rank", [-1, 1.5, "0", None, True])
    def test_rejects_bad_rank(self, rank: Any) -> None:
        with pytest.raises(QueryError):
            LexicalHit(record_id="c-1", score=1.0, rank=rank)

    def test_to_dict_includes_text_on_demand(self) -> None:
        hit = LexicalHit(record_id="c-1", score=1.2345678, rank=2, text="缓存", metadata={"a": 1})

        assert set(hit.to_dict()) == {"rank", "record_id", "score", "channel", "metadata", "text"}
        assert set(hit.to_dict(include_text=False)) == {
            "rank",
            "record_id",
            "score",
            "channel",
            "metadata",
        }
        assert hit.to_dict()["score"] == 1.234568
        assert "bm25" in hit.summary_line()

    def test_search_result_shape_is_validated(self) -> None:
        with pytest.raises(QueryError):
            LexicalSearchResult(candidates=-1)


# --------------------------------------------------------------------------- #
# 索引的只读视图
# --------------------------------------------------------------------------- #


class TestIndexViews:
    """篇数、词表、平均长度与投影：这些数字是所有诊断的分母. """

    def test_count_and_vocabulary_size(self) -> None:
        index = hybrid_lexical()

        assert index.count == len(RECORD_IDS) == 10
        assert len(index) == 10
        assert index.vocabulary_size == EXPECTED_VOCABULARY_SIZE

    def test_avgdl_is_the_mean_document_length(self) -> None:
        index = hybrid_lexical()

        assert index.avgdl == pytest.approx(EXPECTED_AVGDL, rel=1e-9)
        assert index.describe()["total_terms"] == EXPECTED_TOTAL_TERMS
        assert index.describe()["avgdl"] == pytest.approx(EXPECTED_AVGDL, rel=1e-9)

    def test_ids_are_sorted(self) -> None:
        """升序而不是插入序：报告要能被逐行 diff（与 ``VectorBackend.ids`` 同一纪律）. """
        assert hybrid_lexical().ids() == sorted(RECORD_IDS)

    def test_document_lookup(self) -> None:
        index = hybrid_lexical()

        assert index.document("h-t-01") is not None
        assert index.document("h-t-01").text.startswith("ERR-2043")
        assert index.document("nope") is None

    def test_vocabulary_is_sorted_by_term(self) -> None:
        vocabulary = hybrid_lexical().vocabulary()

        assert list(vocabulary) == sorted(vocabulary)
        assert vocabulary["2043"] == 1
        assert all(value >= 1 for value in vocabulary.values())

    def test_describe_reports_the_bm25_params(self) -> None:
        index = hybrid_lexical(params=BM25Params(k1=1.2, b=0.6))
        described = index.describe()

        assert described["k1"] == 1.2
        assert described["b"] == 0.6
        assert described["params"] == {"k1": 1.2, "b": 0.6}
        assert described["channel"] == "bm25"
        assert described["name"] == "lexical"

    def test_summary_line(self) -> None:
        summary = hybrid_lexical().summary_line()

        assert "10 篇" in summary
        assert "BM25(k1=1.5, b=0.75)" in summary
        assert str(EXPECTED_VOCABULARY_SIZE) in summary

    def test_params_property(self) -> None:
        assert hybrid_lexical().params.k1 == DEFAULT_K1
        assert hybrid_lexical(params=BM25Params(k1=0.0, b=0.0)).params.k1 == 0.0

    def test_name_is_reported(self) -> None:
        assert hybrid_lexical(name="手册").name == "手册"
        assert hybrid_lexical().name == "lexical"

    def test_empty_index_views(self) -> None:
        """空索引的四个视图都必须有确定值，``avgdl`` 是 0.0（不是除零）. """
        index = LexicalIndex()

        assert index.count == 0
        assert len(index) == 0
        assert index.vocabulary_size == 0
        assert index.avgdl == 0.0
        assert index.ids() == []
        assert index.vocabulary() == {}
        assert index.document("c-1") is None

    def test_bad_params_object_is_rejected(self) -> None:
        with pytest.raises(QueryError) as excinfo:
            LexicalIndex(params={"k1": 1.5})

        assert "BM25Params" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# 检索：名次用常量，分数与独立实现比
# --------------------------------------------------------------------------- #


class TestSearchGolden:
    """四个查询的**名次**是手算常量（它们由"只有唯一一条含这些词元"直接推出）. """

    @pytest.mark.parametrize("query", sorted(EXPECTED_LEXICAL_IDS))
    def test_hits_match_the_expected_ids(self, query: str) -> None:
        result = hybrid_lexical().search(query)

        assert result.ids() == list(EXPECTED_LEXICAL_IDS[query])

    @pytest.mark.parametrize("query", sorted(EXPECTED_LEXICAL_SCORES))
    def test_scores_match_the_constant(self, query: str) -> None:
        result = hybrid_lexical().search(query)

        assert [hit.score for hit in result.hits] == pytest.approx(
            [EXPECTED_LEXICAL_SCORES[query]], rel=1e-6
        )

    @pytest.mark.parametrize(
        "query",
        [QUERY_EXACT, QUERY_CODE, QUERY_BOTH, "预算", QUERY_SEMANTIC, "缓存", "检索 阈值"],
    )
    def test_scores_match_an_independent_reimplementation(self, query: str) -> None:
        """**最要紧的一条**：与手写公式的实现逐位相同（含名次与全部分数）. """
        index = hybrid_lexical()
        result = index.search(query, top_k=MAX_TOP_K)

        assert result.ids() == [record_id for record_id, _score in brute_force_bm25(index, query)]
        assert [hit.score for hit in result.hits] == pytest.approx(
            [score for _record_id, score in brute_force_bm25(index, query)], rel=1e-9
        )

    def test_candidates_is_the_whole_store_without_a_filter(self) -> None:
        result = hybrid_lexical().search(QUERY_EXACT)

        assert result.candidates == len(RECORD_IDS)
        assert result.filter_applied is False

    def test_ranks_are_contiguous_from_zero(self) -> None:
        result = hybrid_lexical().search("缓存", top_k=5)

        assert [hit.rank for hit in result.hits] == list(range(result.count))

    def test_query_terms_keep_duplicates_and_order(self) -> None:
        """``query_terms`` 是"这一路看到了什么"，因此保留重复与原序.

        ``"缓存缓存"`` 切出 7 个词元：4 个单字 + 3 个 2-gram（``缓存`` 出现两次、
        ``存缓`` 一次）——重复保留是刻意的，词频是 BM25 的输入之一。
        """
        result = hybrid_lexical().search("缓存缓存")

        assert result.query_terms == tuple(tokenize("缓存缓存"))
        assert len(result.query_terms) == 7
        assert result.query_terms.count("缓存") == 2


class TestZeroScoreExclusion:
    """0 分的文档**不进名次**（否则 RRF 会把"毫无关系"当成"第 N 名证据"）. """

    @pytest.mark.parametrize("query", [QUERY_EXACT, QUERY_CODE, QUERY_BOTH, "预算", "缓存"])
    def test_every_hit_has_a_positive_score(self, query: str) -> None:
        result = hybrid_lexical().search(query, top_k=MAX_TOP_K)

        assert result.hits
        assert all(hit.score > 0 for hit in result.hits)

    @pytest.mark.parametrize("query", [QUERY_EXACT, QUERY_CODE, QUERY_BOTH])
    def test_notes_report_how_many_zero_score_documents_were_dropped(self, query: str) -> None:
        """被排除的篇数必须写出来：它解释了"为什么只返回这么几条". """
        result = hybrid_lexical().search(query)

        assert any("0 分" in note for note in result.notes)
        assert any("RRF" in note for note in result.notes)

    def test_note_counts_the_excluded_documents_exactly(self) -> None:
        result = hybrid_lexical().search(QUERY_EXACT, top_k=MAX_TOP_K)
        excluded = result.candidates - result.count

        assert excluded == 9
        assert any(f"{excluded} 篇候选文档" in note for note in result.notes)

    def test_truncation_is_reported(self) -> None:
        """真命中多于 top_k 时的截断也要留数字（与向量路的 dropped 同一纪律）.

        用 ``"报错码"``：它命中 ``h-t-03`` 与 ``h-c-04`` 两条（两条都写到了
        "报错码"），因此 ``top_k=1`` 一定发生截断。
        """
        result = hybrid_lexical().search("报错码", top_k=1)

        assert result.count == 1
        assert result.candidates == len(RECORD_IDS)
        assert any("截断到 top_k=1" in note for note in result.notes)
        assert any("真命中" in note for note in result.notes)


class TestMissingTerms:
    """``matched_terms`` / ``missing_terms``：关键词路"为什么空着"的唯一证据. """

    def test_all_terms_missing_is_reported_verbatim(self) -> None:
        result = hybrid_lexical().search(QUERY_SEMANTIC)

        assert result.hits == ()
        assert result.matched_terms == ()
        assert len(result.missing_terms) == QUERY_SEMANTIC_TERM_COUNT
        assert len(result.query_terms) == QUERY_SEMANTIC_TERM_COUNT
        assert result.candidates == len(RECORD_IDS)
        assert any("交集为空" in note for note in result.notes)
        assert any("词表里有 274 个词元" in note for note in result.notes)

    def test_missing_terms_are_sorted_and_unique(self) -> None:
        result = hybrid_lexical().search(QUERY_SEMANTIC)

        assert list(result.missing_terms) == sorted(set(result.missing_terms))

    @pytest.mark.parametrize(
        "query", [QUERY_EXACT, QUERY_CODE, QUERY_BOTH, "预算", "缓存", "阈值"]
    )
    def test_fully_known_queries_have_no_missing_terms(self, query: str) -> None:
        result = hybrid_lexical().search(query)

        assert result.missing_terms == ()
        assert result.matched_terms == tuple(sorted(set(tokenize(query))))

    def test_partially_known_query_reports_both_sides(self) -> None:
        """一半认识一半不认识时，两边都要列出来（这正是"没召回全"的原因）. """
        result = hybrid_lexical().search("err 午饭")

        assert "err" in result.matched_terms
        assert "午饭" in result.missing_terms
        assert result.hits and result.hits[0].record_id == "h-t-01"
        assert any("不在词表里" in note for note in result.notes)

    @pytest.mark.parametrize("query", [QUERY_EXACT, QUERY_CODE, QUERY_SEMANTIC, "预算"])
    def test_term_accounting_is_closed(self, query: str) -> None:
        """``query_terms`` 去重后必须恰好被 ``matched + missing`` 张成（对账闭合）. """
        result = hybrid_lexical().search(query)

        assert set(result.matched_terms) | set(result.missing_terms) == set(result.query_terms)
        assert not set(result.matched_terms) & set(result.missing_terms)

    def test_vocabulary_contains_every_matched_term(self) -> None:
        index = hybrid_lexical()
        result = index.search("err 2043 预算")

        # ``matched_terms`` 是**排序去重**后的（报告要能逐行 diff），因此是字典序：
        # 数字 < 拉丁 < 汉字。
        assert result.matched_terms == ("2043", "err", "算", "预", "预算")
        for term in result.matched_terms:
            assert term in index.vocabulary()


class TestEmptyIndex:
    """空索引是**合法状态**：返回空结果 + notes，不抛异常（与向量路 count=0 同一口径）. """

    def test_empty_index_returns_an_empty_result_not_an_error(self) -> None:
        result = LexicalIndex().search(QUERY_EXACT)

        assert result.hits == ()
        assert result.candidates == 0
        assert any("关键词索引是空的（0 篇文档）" in note for note in result.notes)
        assert any("合法状态" in note for note in result.notes)

    def test_empty_index_still_accounts_for_the_query_terms(self) -> None:
        """空索引也要给出词元对账（否则"为什么空"在两条不同成因下长得一样）. """
        result = LexicalIndex().search(QUERY_EXACT)

        assert result.query_terms == tuple(tokenize(QUERY_EXACT))
        assert result.matched_terms == ()
        assert result.missing_terms == ("2043", "err")

    def test_empty_index_does_not_claim_a_filter_when_none_was_given(self) -> None:
        assert LexicalIndex().search(QUERY_EXACT).filter_applied is False

    def test_empty_index_keeps_the_filter_flag(self) -> None:
        result = LexicalIndex().search(QUERY_EXACT, where={"topic": "errors"})

        assert result.filter_applied is True
        assert result.candidates == 0

    def test_search_on_an_empty_index_is_idempotent(self) -> None:
        index = LexicalIndex()

        assert index.search("缓存") == index.search("缓存")


class TestWhereFiltering:
    """``where`` 复用 ``vectorstore.compile_filter``（两路过滤语义必须同一份）. """

    @pytest.mark.parametrize(
        ("where", "expected_ids", "expected_candidates"),
        [
            ({"strategy": "semantic"}, ("h-t-01",), 4),
            ({"strategy": "structural"}, (), 4),
            ({"parent_doc_id": "doc-trouble"}, ("h-t-01",), 3),
            ({"parent_doc_id": "doc-concept"}, (), 4),
            ({"topic": {"$in": ["errors"]}}, ("h-t-01",), 3),
            ({"topic": {"$ne": "errors"}}, (), 7),
            ({"$and": [{"topic": "errors"}, {"strategy": "semantic"}]}, ("h-t-01",), 2),
            ({"$or": [{"topic": "errors"}, {"topic": "glossary"}]}, ("h-t-01",), 4),
            ({"index": {"$lt": 1}}, ("h-t-01",), 3),
            ({"created_at": {"$gte": "2026-09-15"}}, (), 5),
            ({"created_at": {"$lte": "2026-09-10"}}, (), 3),
            ({"topic": "nope"}, (), 0),
            ({"token_count": {"$gt": 100}}, (), 0),
        ],
    )
    def test_filter_narrows_the_candidate_set(
        self, where: dict, expected_ids: tuple[str, ...], expected_candidates: int
    ) -> None:
        result = hybrid_lexical().search(QUERY_EXACT, where=where)

        assert result.ids() == list(expected_ids)
        assert result.candidates == expected_candidates
        assert result.filter_applied is True

    def test_missing_time_field_is_excluded_by_the_range(self) -> None:
        """``h-t-03`` 没有 ``created_at``：任何时间范围都会把它排除（day066 的约定）. """
        index = hybrid_lexical()
        unfiltered = index.search(QUERY_CODE)
        filtered = index.search(QUERY_CODE, where={"created_at": {"$gte": "2026-01-01"}})

        assert unfiltered.ids() == ["h-t-03"]
        assert filtered.ids() == []
        assert filtered.candidates == len(RECORD_IDS) - 1

    def test_zero_candidates_is_reported(self) -> None:
        result = hybrid_lexical().search(QUERY_EXACT, where={"strategy": "nope"})

        assert result.candidates == 0
        assert any("筛成了 0 条" in note for note in result.notes)

    def test_empty_dict_counts_as_no_filter(self) -> None:
        """``{}`` 与 ``None`` 等价于"没有过滤"，而 ``{}`` 与"条件是空的"是两件事. """
        index = hybrid_lexical()

        assert index.search(QUERY_EXACT, where={}).filter_applied is False
        assert index.search(QUERY_EXACT, where={}).candidates == len(RECORD_IDS)

    def test_idf_is_a_corpus_statistic_not_a_subset_statistic(self) -> None:
        """过滤不会改变 idf（同一份语料的性质）——分数因此可以跨查询比较. """
        index = hybrid_lexical()
        unfiltered = index.search(QUERY_EXACT)
        filtered = index.search(QUERY_EXACT, where={"topic": "errors"})

        assert filtered.hits[0].score == pytest.approx(unfiltered.hits[0].score, rel=1e-12)

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
    def test_invalid_where_syntax_raises_query_error(self, where: dict) -> None:
        with pytest.raises(QueryError) as excinfo:
            hybrid_lexical().search(QUERY_EXACT, where=where)

        assert "where" in str(excinfo.value)

    def test_non_dict_where_raises(self) -> None:
        with pytest.raises(QueryError):
            hybrid_lexical().search(QUERY_EXACT, where=["topic"])


class TestParameterDirections:
    """k1 与 b 的**方向**：不依赖具体数字的三条不变量（受控梯度语料）. """

    @pytest.mark.parametrize("k1", [0.0, 0.5, 1.5, 3.0, 10.0])
    def test_bigger_k1_grows_the_frequency_advantage(self, k1: float) -> None:
        """k1 越大，"同一个词出现更多次"的收益越大（词频饱和被推后）. """
        index = graded_index(params=BM25Params(k1=k1, b=0.75))
        heavier = ratio(index, "缓存", "heavy", "light")

        assert heavier >= 1.0

    def test_k1_sweep_is_monotone(self) -> None:
        """整条扫描链单调不减：k1=0 时两条同分，k1 增大时重的那条越来越占优. """
        ratios = [
            ratio(graded_index(params=BM25Params(k1=k1, b=0.75)), "缓存", "heavy", "light")
            for k1 in (0.0, 0.5, 1.5, 3.0, 10.0, 50.0)
        ]

        assert ratios == sorted(ratios)
        assert ratios[0] == pytest.approx(1.0, rel=1e-12)
        assert ratios[-1] > ratios[0]

    def test_k1_zero_ignores_the_term_frequency_entirely(self) -> None:
        """``k1=0`` 退化成"只看出现过没有"：两条含同一批词元的文档分数相同. """
        index = graded_index(params=BM25Params(k1=0.0, b=0.0))

        assert score_of(index, "缓存", "heavy") == pytest.approx(
            score_of(index, "缓存", "light"), rel=1e-12
        )

    @pytest.mark.parametrize("b", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_big_b_penalises_the_longer_document(self, b: float) -> None:
        """b 越大，长文档里的同一个词越不值钱（长度归一化强度）. """
        index = graded_index(params=BM25Params(k1=1.5, b=b))
        heavier = ratio(index, "缓存", "heavy", "light")

        assert heavier > 0.0

    def test_b_sweep_is_monotone_decreasing(self) -> None:
        """b 从 0 增加到 1 时，"重而长"那条的相对优势单调下降. """
        ratios = [
            ratio(graded_index(params=BM25Params(k1=1.5, b=b)), "缓存", "heavy", "light")
            for b in (0.0, 0.25, 0.5, 0.75, 1.0)
        ]

        assert ratios == sorted(ratios, reverse=True)
        assert ratios[0] > ratios[-1]

    def test_b_zero_makes_length_irrelevant(self) -> None:
        """``b=0``：长度完全不参与打分，同词频的两条文档分数相同. """
        index = LexicalIndex.build(
            [
                LexicalDocument(record_id="short", text="缓存"),
                LexicalDocument(record_id="long", text="缓存检索手册说明"),
            ],
            params=BM25Params(k1=1.5, b=0.0),
        )

        assert score_of(index, "缓存", "short") == pytest.approx(
            score_of(index, "缓存", "long"), rel=1e-12
        )

    def test_b_one_makes_length_decisive(self) -> None:
        """``b=1``：长度完全归一，两条的分数比恰好由 ``(1 + k1·dl/avgdl)`` 给出.

        两条文档含**同一批词元、同样的 tf（都是 1）**，因此 idf 那一项在比值里
        整体约掉，剩下的只有分母：``(1 + k1·dl_long/avgdl) / (1 + k1·dl_short/avgdl)``。
        这个比值是手推出来的（不是抄实现的），因此它能抓住"长度归一化系数写错"这类回归。
        """
        k1 = 1.5
        index = LexicalIndex.build(
            [
                LexicalDocument(record_id="short", text="缓存"),
                LexicalDocument(record_id="long", text="缓存检索手册说明"),
            ],
            params=BM25Params(k1=k1, b=1.0),
        )
        short_tokens = len(tokenize("缓存"))
        long_tokens = len(tokenize("缓存检索手册说明"))
        avgdl = (short_tokens + long_tokens) / 2
        expected = (1 + k1 * long_tokens / avgdl) / (1 + k1 * short_tokens / avgdl)

        assert score_of(index, "缓存", "short") == pytest.approx(
            score_of(index, "缓存", "long") * expected, rel=1e-9
        )
        assert expected > 1.0

    def test_tf_and_length_are_both_used(self) -> None:
        """默认参数下"重"那条既因词频占优、又因更长被压——两个方向同时在场. """
        heavy = score_of(graded_index(), "缓存", "heavy")
        middle = score_of(graded_index(), "缓存", "middle")
        light = score_of(graded_index(), "缓存", "light")

        assert heavy > light > 0
        assert middle > 0


class TestOrderingDeterminism:
    """排序的确定性：分数降序 + 同分按 ``record_id`` 升序，两次运行逐位相同. """

    def test_search_is_reproducible(self) -> None:
        index = hybrid_lexical()

        first = index.search("缓存")
        second = index.search("缓存")

        assert first.ids() == second.ids()
        assert [hit.score for hit in first.hits] == [hit.score for hit in second.hits]

    def test_reversed_document_order_gives_the_same_answer(self) -> None:
        """插入顺序不影响结果（"排序规则不依赖插入顺序"这件事要被证明）. """
        forward = LexicalIndex.build(documents())
        backward = LexicalIndex.build(list(reversed(documents())))

        assert forward.search("缓存").ids() == backward.search("缓存").ids()
        assert forward.ids() == backward.ids()

    def test_tied_scores_are_ordered_by_record_id(self) -> None:
        """文本逐字相同的两条 → 分数精确相等 → 由 id 升序决定（否则两次运行会不同）. """
        index = LexicalIndex.build(
            [
                LexicalDocument(record_id="b-2", text="缓存检索"),
                LexicalDocument(record_id="a-1", text="缓存检索"),
            ]
        )
        result = index.search("缓存")

        assert result.hits[0].score == result.hits[1].score
        assert result.ids() == ["a-1", "b-2"]

    def test_tie_ordering_does_not_depend_on_insertion_order(self) -> None:
        forward = LexicalIndex.build(
            [
                LexicalDocument(record_id="b-2", text="缓存检索"),
                LexicalDocument(record_id="a-1", text="缓存检索"),
            ]
        )
        backward = LexicalIndex.build(
            [
                LexicalDocument(record_id="a-1", text="缓存检索"),
                LexicalDocument(record_id="b-2", text="缓存检索"),
            ]
        )

        assert forward.search("缓存").ids() == backward.search("缓存").ids() == ["a-1", "b-2"]

    def test_query_term_order_does_not_change_the_scores(self) -> None:
        """词元顺序不影响分数（求和是交换的），但 ``query_terms`` 会保留原序. """
        index = hybrid_lexical()

        assert score_of(index, "err 2043", "h-t-01") == pytest.approx(
            score_of(index, "2043 err", "h-t-01"), rel=1e-12
        )

    def test_repeated_query_terms_do_not_double_count(self) -> None:
        """查询里写两遍同一个词不会让分数翻倍（打分对查询词元去重）. """
        index = hybrid_lexical()

        assert score_of(index, "err", "h-t-01") == pytest.approx(
            score_of(index, "err err err", "h-t-01"), rel=1e-12
        )


# --------------------------------------------------------------------------- #
# 装配、增删与批量
# --------------------------------------------------------------------------- #


class NotImplementedBackend:
    """既没有 ``ids()`` 也没有 ``get_many()``：``from_backend`` 要给出出路. """


class IdsOnlyBackend:
    """只有 ``ids()``：``from_backend`` 必须点名缺的是哪一个方法. """

    def __init__(self, ids: list[str]) -> None:
        self._ids = ids

    def ids(self) -> list[str]:
        return list(self._ids)


class DroppingBackend:
    """``get_many`` 少返回几条：模拟"记录表与向量索引不同步"（day064 的经典故障）. """

    def __init__(self, store: Any, *, drop: int = 1) -> None:
        self._store = store
        self._drop = drop

    def ids(self) -> list[str]:
        return self._store.ids()

    def get_many(self, ids: list[str]) -> list[Any]:
        return list(self._store.get_many(ids))[: -self._drop]


class TestFromBackend:
    """``from_backend`` 是这一层的推荐入口（关键词索引从向量库现建）. """

    def test_builds_from_the_store(self) -> None:
        index = LexicalIndex.from_backend(hybrid_store())

        assert index.count == len(RECORD_IDS)
        assert index.ids() == sorted(RECORD_IDS)

    def test_matches_build_from_documents(self) -> None:
        """两条装配路径给出同一份索引（同一批文档、同一个打分口径）. """
        from_backend = LexicalIndex.from_backend(hybrid_store())
        from_documents = LexicalIndex.build(documents())

        assert from_backend.vocabulary() == from_documents.vocabulary()
        assert from_backend.avgdl == pytest.approx(from_documents.avgdl, rel=1e-12)
        assert from_backend.search(QUERY_EXACT).ids() == from_documents.search(QUERY_EXACT).ids()

    def test_carries_the_metadata_through(self) -> None:
        """元数据必须跟着过来（``where`` 过滤要用它）. """
        index = LexicalIndex.from_backend(hybrid_store())

        assert index.document("h-t-01").metadata["strategy"] == "semantic"
        assert index.document("h-t-03").metadata.get("created_at") is None

    def test_empty_store_builds_an_empty_index(self) -> None:
        index = LexicalIndex.from_backend(empty_store())

        assert index.count == 0
        assert index.avgdl == 0.0

    def test_params_are_forwarded(self) -> None:
        index = LexicalIndex.from_backend(hybrid_store(), params=BM25Params(k1=0.0, b=1.0))

        assert index.params.k1 == 0.0
        assert index.params.b == 1.0

    def test_rejects_an_object_without_the_primitives(self) -> None:
        with pytest.raises(QueryError) as excinfo:
            LexicalIndex.from_backend(NotImplementedBackend())

        assert "ids" in str(excinfo.value)
        assert "LexicalIndex.build" in str(excinfo.value)

    def test_rejects_an_object_missing_only_get_many(self) -> None:
        with pytest.raises(QueryError) as excinfo:
            LexicalIndex.from_backend(IdsOnlyBackend(["a-1"]))

        assert "get_many" in str(excinfo.value)

    def test_backend_reporting_more_ids_than_records_is_an_index_error(self) -> None:
        """库里报 10 条、只取回 9 条 → ``LexicalError``（**不许静默少召回**）. """
        with pytest.raises(LexicalError) as excinfo:
            LexicalIndex.from_backend(DroppingBackend(hybrid_store(), drop=1))

        message = str(excinfo.value)
        assert "10 条" in message
        assert "9 条" in message
        assert "from_backend" in message

    def test_the_error_is_not_a_query_error(self) -> None:
        """它属于"索引运维"那一族：改调用点走不通（见 errors 的分族依据）. """
        with pytest.raises(LexicalError) as excinfo:
            LexicalIndex.from_backend(DroppingBackend(hybrid_store(), drop=3))

        assert not isinstance(excinfo.value, QueryError)


class TestAddAndReplace:
    """``add`` 对已存在的 id 是**替换**（这份索引是"现在长什么样"的投影）. """

    def test_add_grows_the_index(self) -> None:
        index = LexicalIndex.build(documents())
        index.add(LexicalDocument(record_id="new-1", text="新加的片段"))

        assert index.count == len(RECORD_IDS) + 1
        assert "new-1" in index.ids()
        # "新加" 的单字 "新" 在 h-m-01 里也出现过（"重新标定"），因此命中不止一条——
        # 单字粒度带来的召回正是它的用途；要断言的是"新加的那条排在前面"。
        assert index.search("新加").ids()[0] == "new-1"

    def test_replacing_an_id_keeps_the_count(self) -> None:
        index = LexicalIndex.build(documents())
        index.add(LexicalDocument(record_id="h-t-01", text="换成了完全不同的内容"))

        assert index.count == len(RECORD_IDS)
        assert index.document("h-t-01").text == "换成了完全不同的内容"

    def test_replacing_drops_the_old_terms(self) -> None:
        """替换之后旧词元必须从词表里消失（否则会命中一条已经没有它的记录）. """
        index = LexicalIndex.build(documents())
        index.add(LexicalDocument(record_id="h-t-01", text="换成了完全不同的内容"))

        assert "2043" not in index.vocabulary()
        assert index.search(QUERY_EXACT).ids() == []

    def test_replacing_updates_the_document_frequency(self) -> None:
        """``df`` 是增量维护的：替换后 2043 的 df 归零（旧词元被摘掉）. """
        index = LexicalIndex.from_backend(hybrid_store())
        before = index.vocabulary()["2043"]
        index.add(LexicalDocument(record_id="h-t-01", text="换掉了"))

        assert before == 1
        assert "2043" not in index.vocabulary()
        assert index.count == len(RECORD_IDS)

    def test_replacing_updates_avgdl(self) -> None:
        index = LexicalIndex.from_backend(hybrid_store())
        before = index.avgdl
        index.add(LexicalDocument(record_id="h-t-01", text="短"))

        assert index.avgdl != pytest.approx(before, rel=1e-12)

    def test_repeated_replace_is_stable(self) -> None:
        """同一份文档替换两次 == 替换一次（幂等；否则 df 会被算两遍）. """
        once = LexicalIndex.from_backend(hybrid_store())
        twice = LexicalIndex.from_backend(hybrid_store())
        replacement = LexicalDocument(record_id="h-t-01", text="替换后的正文")
        once.add(replacement)
        twice.add(replacement)
        twice.add(replacement)

        assert once.vocabulary() == twice.vocabulary()
        assert once.avgdl == pytest.approx(twice.avgdl, rel=1e-12)

    def test_add_rejects_non_documents(self) -> None:
        with pytest.raises(QueryError) as excinfo:
            LexicalIndex().add({"record_id": "c-1", "text": "缓存"})

        assert "LexicalDocument" in str(excinfo.value)


class TestSearchValidation:
    """入参校验：与 ``RetrievalQuery`` 同一套口径（两条路的入口规则必须一致）. """

    @pytest.mark.parametrize("query", ["", "   ", "\n", "\t"])
    def test_blank_query_is_rejected(self, query: str) -> None:
        with pytest.raises(QueryError) as excinfo:
            hybrid_lexical().search(query)

        assert "空查询" in str(excinfo.value)
        assert "0 个词元" in str(excinfo.value)

    @pytest.mark.parametrize("query", [None, 123, ["缓存"], b"cache"])
    def test_non_string_query_is_rejected(self, query: Any) -> None:
        with pytest.raises(QueryError):
            hybrid_lexical().search(query)

    @pytest.mark.parametrize("top_k", [0, -1, 1.5, "5", None, True])
    def test_bad_top_k_is_rejected(self, top_k: Any) -> None:
        with pytest.raises(QueryError):
            hybrid_lexical().search(QUERY_EXACT, top_k=top_k)

    def test_top_k_beyond_the_ceiling_is_rejected(self) -> None:
        """上限**复用** ``vectorstore.MAX_TOP_K``（两路的深度上限必须是同一个数）. """
        with pytest.raises(QueryError) as excinfo:
            hybrid_lexical().search(QUERY_EXACT, top_k=MAX_TOP_K + 1)

        assert str(MAX_TOP_K) in str(excinfo.value)

    def test_top_k_equal_to_the_ceiling_is_accepted(self) -> None:
        index = hybrid_lexical()

        assert index.search(QUERY_EXACT, top_k=MAX_TOP_K).ids() == ["h-t-01"]


class TestPoisonedIndex:
    """有文档、但一个词元都没有：这是**索引侧**的事（去修入库口径），不是"没有命中". """

    def poisoned_index(self) -> LexicalIndex:
        return LexicalIndex.build(
            [
                LexicalDocument(record_id="p-1", text="，。！"),
                LexicalDocument(record_id="p-2", text="——…"),
            ]
        )

    def test_search_raises_a_lexical_error(self) -> None:
        with pytest.raises(LexicalError) as excinfo:
            self.poisoned_index().search("缓存")

        message = str(excinfo.value)
        assert "词元总数为 0" in message
        assert "入库口径" in message

    def test_the_poisoned_index_still_reports_its_size(self) -> None:
        index = self.poisoned_index()

        assert index.count == 2
        assert index.vocabulary_size == 0
        assert index.avgdl == 0.0

    def test_it_is_not_a_query_error(self) -> None:
        with pytest.raises(LexicalError) as excinfo:
            self.poisoned_index().search("缓存")

        assert not isinstance(excinfo.value, QueryError)


class TestSearchMany:
    """批量检索刻意逐条（每次的词元对账是独立证据）. """

    def test_returns_one_result_per_query_in_order(self) -> None:
        results = hybrid_lexical().search_many([QUERY_EXACT, QUERY_SEMANTIC, QUERY_CODE])

        assert [result.ids() for result in results] == [
            ["h-t-01"],
            [],
            ["h-t-03"],
        ]

    def test_forward_searches_are_reproducible(self) -> None:
        index = hybrid_lexical()

        assert index.search_many([QUERY_EXACT, "缓存"]) == index.search_many([QUERY_EXACT, "缓存"])
