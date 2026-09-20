"""day066 ``retrieval.types`` 的单元测试：四副形状 + 常量顺序 + 诊断四问.

全部离线、确定性。这一层的断言分四类，每一类都盯着"这个字段被改动之后，
报告里的哪句话会变成假的"：

```text
常量与顺序    EMPTY_REASONS 的顺序就是诊断优先级；描述表必须覆盖全部取值
TimeRange     三条约定各自可核对（闭区间 / 按 ISO 文本比较 / 缺失字段被排除）
Query/Hit     校验分支逐条有正反例（校验在构造期，不在调用点）
Result        三道减法、explain() 的四问、aggregate_results 的空批次
```

**刻意不写"软断言"**（``is not None`` / 只判真假）：一条软断言在实现被改坏
之后仍然会通过，它证明不了任何事。所有断言都指向具体数值、具体次序或
具体错误消息片段。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from smart_research_agent.retrieval import types as types_module
from smart_research_agent.retrieval.errors import QueryError
from smart_research_agent.retrieval.types import (
    CHANNEL_VECTOR,
    DEFAULT_DOC_ID_FIELD,
    DEFAULT_FETCH_MULTIPLIER,
    DEFAULT_MIN_HIT_CHARS,
    EMPTY_REASON_BELOW_THRESHOLD,
    EMPTY_REASON_DESCRIPTIONS,
    EMPTY_REASON_DIVERSITY,
    EMPTY_REASON_FILTERED_OUT,
    EMPTY_REASON_NO_DATA,
    EMPTY_REASON_NONE,
    EMPTY_REASONS,
    MAX_FETCH_K,
    RETRIEVAL_LIMITATIONS,
    RETRIEVAL_OUT_OF_SCOPE,
    IndexState,
    RetrievalHit,
    RetrievalQuery,
    RetrievalResult,
    TimeRange,
    aggregate_results,
)
from smart_research_agent.vectorstore.base import MAX_TOP_K
from smart_research_agent.vectorstore.filters import SUPPORTED_OPERATORS

# --------------------------------------------------------------------------- #
# 造样本的小工具（每个都只改一处，让"哪一支坏了"能一眼定位）
# --------------------------------------------------------------------------- #


def make_query(**overrides: Any) -> RetrievalQuery:
    """造一个合法的查询（覆盖时才用给定值）."""
    payload: dict[str, Any] = {"text": "语义缓存怎么配置阈值"}
    payload.update(overrides)
    return RetrievalQuery(**payload)


def make_hit(record_id: str = "c-a-01", **overrides: Any) -> RetrievalHit:
    """造一条合法命中（元数据默认带 parent_doc_id / heading_path / source）."""
    payload: dict[str, Any] = {
        "score": 0.9,
        "rank": 0,
        "text": "相似度阈值用分位数标定。",
        "metadata": {
            "parent_doc_id": "doc-alpha",
            "heading_path": "检索手册 > 阈值",
            "source": "docs/检索手册.md",
        },
    }
    payload.update(overrides)
    return RetrievalHit(record_id=record_id, **payload)


def make_result(*hits: RetrievalHit, **overrides: Any) -> RetrievalResult:
    """造一个检索结果（默认：query/top_k=5、无过滤、有命中时 ``empty_reason=hits``）."""
    payload: dict[str, Any] = {
        "query": make_query(top_k=5),
        "hits": hits,
        "fetch_k": 15,
        "candidates": len(hits),
        "filter_applied": False,
        "empty_reason": EMPTY_REASON_NONE if hits else EMPTY_REASON_BELOW_THRESHOLD,
        "metric": "cosine",
    }
    payload.update(overrides)
    return RetrievalResult(**payload)


# --------------------------------------------------------------------------- #
# 常量与顺序
# --------------------------------------------------------------------------- #


class TestConstants:
    """常量不是装饰：它们每一个都被别处引用，改动会静默改变行为."""

    def test_fetch_multiplier_and_cap(self) -> None:
        """过取倍率取 3；深度上限**复用** vectorstore 的 MAX_TOP_K（不是另写一个 1000）."""
        assert DEFAULT_FETCH_MULTIPLIER == 3
        assert MAX_FETCH_K == MAX_TOP_K
        assert MAX_FETCH_K == 1000

    def test_min_hit_chars_and_doc_id_field(self) -> None:
        """单条命中的最小长度与"文档身份字段"各只有一个定义."""
        assert DEFAULT_MIN_HIT_CHARS == 32
        assert DEFAULT_DOC_ID_FIELD == "parent_doc_id"
        assert CHANNEL_VECTOR == "vector"

    def test_empty_reason_values(self) -> None:
        """五个取值都是稳定字符串（它们会出现在报告与端点响应里）."""
        assert EMPTY_REASON_NONE == "hits"
        assert EMPTY_REASON_NO_DATA == "no_data"
        assert EMPTY_REASON_FILTERED_OUT == "filtered_out"
        assert EMPTY_REASON_BELOW_THRESHOLD == "below_threshold"
        assert EMPTY_REASON_DIVERSITY == "diversity_trimmed"

    def test_empty_reasons_order_is_the_priority_order(self) -> None:
        """``EMPTY_REASONS`` 的顺序 = 诊断优先级（上游成因在前）.

        顺序本身是接口的一部分：``_diagnose`` 按它判定，报告按它解释。
        把 "no_data" 与 "below_threshold" 调换位置，会让"库是空的"这件事
        被解释成"阈值定得太高"——一条指向错误动作的诊断。
        """
        assert EMPTY_REASONS == (
            EMPTY_REASON_NONE,
            EMPTY_REASON_NO_DATA,
            EMPTY_REASON_FILTERED_OUT,
            EMPTY_REASON_BELOW_THRESHOLD,
            EMPTY_REASON_DIVERSITY,
        )

    def test_descriptions_cover_every_reason(self) -> None:
        """人话解释必须覆盖全部取值，否则 ``explain()`` 会出现"未知原因"."""
        assert set(EMPTY_REASON_DESCRIPTIONS) == set(EMPTY_REASONS)
        assert all(text.strip() for text in EMPTY_REASON_DESCRIPTIONS.values())

    def test_limitations_and_out_of_scope_are_written_down(self) -> None:
        """两份"边界声明"必须写明与 day067/day068 的分工（写下来才不算 bug）."""
        assert len(RETRIEVAL_LIMITATIONS) >= 3
        assert all(isinstance(item, str) and item.strip() for item in RETRIEVAL_LIMITATIONS)
        assert "BM25" in " ".join(RETRIEVAL_LIMITATIONS)
        assert "day067" in " ".join(RETRIEVAL_LIMITATIONS)

        assert len(RETRIEVAL_OUT_OF_SCOPE) >= 3
        assert "查询改写" in " ".join(RETRIEVAL_OUT_OF_SCOPE)

    def test_package_reexports_the_same_objects(self) -> None:
        """``retrieval/__init__`` 导出的是**同一批对象**（不是各复制一份）."""
        import smart_research_agent.retrieval as package

        assert package.EMPTY_REASONS is types_module.EMPTY_REASONS
        assert package.MAX_FETCH_K is types_module.MAX_FETCH_K
        assert package.DEFAULT_FETCH_MULTIPLIER == DEFAULT_FETCH_MULTIPLIER
        for name in ("TimeRange", "RetrievalQuery", "RetrievalHit", "IndexState",
                     "RetrievalResult", "aggregate_results"):
            assert name in package.__all__


# --------------------------------------------------------------------------- #
# TimeRange：三条约定
# --------------------------------------------------------------------------- #


class TestTimeRangeSemantics:
    """闭区间 / 单端 / 日期按 0 点 / 描述与投影."""

    def test_defaults_are_empty(self) -> None:
        """两端都没给 = "这次不涉及时间条件"（≠ 区间宽度为 0）."""
        span = TimeRange()

        assert span.field == "created_at"
        assert span.start is None
        assert span.end is None
        assert span.is_empty is True
        assert span.clause() == {}
        assert span.describe() == "created_at（未设区间）"

    def test_single_bound_is_not_empty(self) -> None:
        """只给一端就已经是一次时间过滤（最常见的问法是"这个月之后"）."""
        assert TimeRange(start="2026-09-01").is_empty is False
        assert TimeRange(end="2026-09-30").is_empty is False

    def test_clause_shapes_for_three_cases(self) -> None:
        """双端 → ``$and`` 包两条；单端 → 一条；空 → 空字典.

        双端必须用 ``$and``：``vectorstore`` 拒绝同一字段的多个运算符，
        包一层是那层留下的唯一合法写法（见 ``filters.time_range_clause``）。
        """
        both = TimeRange(start="2026-09-01", end="2026-09-30")
        assert both.clause() == {
            "$and": [
                {"created_at": {"$gte": "2026-09-01"}},
                {"created_at": {"$lte": "2026-09-30"}},
            ]
        }
        assert TimeRange(start="2026-09-01").clause() == {"created_at": {"$gte": "2026-09-01"}}
        assert TimeRange(end="2026-09-30").clause() == {"created_at": {"$lte": "2026-09-30"}}
        assert TimeRange().clause() == {}

    def test_date_only_is_interpreted_as_midnight(self) -> None:
        """只写日期时按当日 00:00:00 解释（"闭区间包含当天起点"的来源）."""
        low, high = TimeRange(start="2026-09-20", end="2026-09-21").bounds()

        assert low == datetime(2026, 9, 20, 0, 0, 0)
        assert low.hour == 0 and low.minute == 0 and low.second == 0
        assert high == datetime(2026, 9, 21, 0, 0, 0)

    def test_bounds_are_inclusive_and_ordered(self) -> None:
        """闭区间的两端都在区间内：``start == end`` 是合法区间（不是倒置）."""
        same = TimeRange(start="2026-09-15", end="2026-09-15")

        low, high = same.bounds()

        assert low == high == datetime(2026, 9, 15)

    def test_bounds_handle_zulu_suffix(self) -> None:
        """``Z`` 结尾在 Python 3.10 的 fromisoformat 里不认，必须在这里被换掉."""
        low, _ = TimeRange(start="2026-09-01T08:30:00Z").bounds()

        assert low == datetime(2026, 9, 1, 8, 30, tzinfo=timezone.utc)

    def test_bounds_returns_none_for_missing_endpoints(self) -> None:
        """省略的那一端是 ``None``（不是"现在"、不是 0）."""
        assert TimeRange().bounds() == (None, None)
        assert TimeRange(start="2026-09-01").bounds()[1] is None
        assert TimeRange(end="2026-09-01").bounds()[0] is None

    def test_field_is_stripped(self) -> None:
        """字段名收敛成 strip 之后的形式（`' created_at '` 与 `'created_at'` 是同一个）."""
        span = TimeRange(field="  created_at  ", start="2026-09-01")

        assert span.field == "created_at"
        assert span.clause() == {"created_at": {"$gte": "2026-09-01"}}

    def test_describe_uses_infinities_for_open_ends(self) -> None:
        """单端区间的描述里，省略的那一端写成 ±∞（"没有边界"要读得出来）."""
        assert TimeRange(start="2026-09-01").describe() == "created_at ∈ [2026-09-01, +∞]"
        assert TimeRange(end="2026-09-30").describe() == "created_at ∈ [-∞, 2026-09-30]"
        both = TimeRange(start="2026-09-01", end="2026-09-30")
        assert both.describe() == "created_at ∈ [2026-09-01, 2026-09-30]"
        assert both.summary_line() == both.describe()

    def test_to_dict_key_set(self) -> None:
        """投影必须可直接 ``json.dumps``（端点会原样返回它）."""
        payload = TimeRange(start="2026-09-01").to_dict()

        assert set(payload) == {"field", "start", "end", "is_empty", "describe"}
        assert payload["is_empty"] is False
        assert payload["describe"] == "created_at ∈ [2026-09-01, +∞]"


class TestTimeRangeValidation:
    """五类写法错误各自当场报 ``QueryError``（不是让它去库里筛出一个空集）."""

    def test_empty_field_name_is_rejected(self) -> None:
        """空字段名会让每次时间过滤都返回空——一个必然为空的条件不该被接受."""
        with pytest.raises(QueryError) as excinfo:
            TimeRange(field="   ")

        assert "TimeRange.field 必须是非空字符串" in str(excinfo.value)

    def test_non_string_field_is_rejected(self) -> None:
        with pytest.raises(QueryError) as excinfo:
            TimeRange(field=123)  # type: ignore[arg-type]

        assert "TimeRange.field 必须是非空字符串" in str(excinfo.value)

    @pytest.mark.parametrize("label", ["start", "end"])
    def test_empty_string_bound_is_rejected(self, label: str) -> None:
        """空串不是"没有边界"（没有边界请写 ``None``）."""
        with pytest.raises(QueryError) as excinfo:
            TimeRange(**{label: "  "})  # type: ignore[arg-type]

        assert "空串不是'没有边界'" in str(excinfo.value)

    @pytest.mark.parametrize("label", ["start", "end"])
    def test_non_string_bound_is_rejected(self, label: str) -> None:
        with pytest.raises(QueryError) as excinfo:
            TimeRange(**{label: 20260901})  # type: ignore[arg-type]

        assert "必须是非空字符串或 None" in str(excinfo.value)

    @pytest.mark.parametrize("bad", ["2026/09/01", "2026-13-01", "上周二", "2026-09-01T25:00"])
    def test_invalid_iso_is_rejected_with_examples(self, bad: str) -> None:
        """报错里要给出可照抄的正确写法（"现象 + 出路"两句）."""
        with pytest.raises(QueryError) as excinfo:
            TimeRange(start=bad)

        message = str(excinfo.value)
        assert "不是合法的 ISO-8601 时间" in message
        assert "2026-09-20" in message

    def test_inverted_range_is_rejected(self) -> None:
        """倒置的区间必然返回空结果，因此在这里当场拒掉."""
        with pytest.raises(QueryError) as excinfo:
            TimeRange(start="2026-09-30", end="2026-09-01")

        message = str(excinfo.value)
        assert "时间范围倒置" in message
        assert "必然返回空结果" in message

    def test_mixed_timezones_cannot_be_compared(self) -> None:
        """一个带时区、一个不带 → ``TypeError`` → 包成可照做的 ``QueryError``."""
        with pytest.raises(QueryError) as excinfo:
            TimeRange(start="2026-09-01", end="2026-09-02T00:00:00+08:00")

        assert "两个端点无法比较" in str(excinfo.value)

    def test_label_points_at_the_bad_endpoint(self) -> None:
        """报错要指出坏的是哪一端（``TimeRange.end`` 而不是笼统的"时间范围"）."""
        with pytest.raises(QueryError) as excinfo:
            TimeRange(start="2026-09-01", end="昨天")

        assert "TimeRange.end" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# RetrievalQuery：全部校验分支
# --------------------------------------------------------------------------- #


class TestRetrievalQueryValidation:
    """查询是入口，因此它把所有"要 0 条""空查询"这类请求挡在门外."""

    def test_text_is_stripped(self) -> None:
        """首尾空格被收敛掉：同一句话不该因为空格而变成两次不同的检索（缓存会算错）."""
        assert make_query(text="  语义缓存  ").text == "语义缓存"

    @pytest.mark.parametrize("text", ["", "   ", "\n\t "])
    def test_empty_text_is_rejected(self, text: str) -> None:
        """空查询在向量检索里没有定义（它会编码成一个固定向量，返回一批"最像空白的记录"）."""
        with pytest.raises(QueryError) as excinfo:
            RetrievalQuery(text=text)

        message = str(excinfo.value)
        assert "空查询在向量检索里没有定义" in message
        assert "请给一段非空文本" in message

    def test_non_string_text_is_rejected(self) -> None:
        with pytest.raises(QueryError) as excinfo:
            RetrievalQuery(text=42)  # type: ignore[arg-type]

        assert "query.text 必须是字符串" in str(excinfo.value)

    @pytest.mark.parametrize("value", [True, 1.5, "5"])
    def test_top_k_must_be_an_integer(self, value: Any) -> None:
        """``bool`` 也是 int，因此必须显式排除（``True`` 不是一个条数）."""
        with pytest.raises(QueryError) as excinfo:
            make_query(top_k=value)

        assert "top_k 必须是整数" in str(excinfo.value)

    def test_top_k_below_one_is_rejected(self) -> None:
        """'要 0 条'不是一个请求；要探测'库里有没有数据'请用 backend.count()."""
        with pytest.raises(QueryError) as excinfo:
            make_query(top_k=0)

        message = str(excinfo.value)
        assert "top_k 必须 >= 1" in message
        assert "backend.count()" in message

    def test_top_k_above_the_cap_is_rejected(self) -> None:
        with pytest.raises(QueryError) as excinfo:
            make_query(top_k=MAX_FETCH_K + 1)

        message = str(excinfo.value)
        assert f"超过上限 {MAX_FETCH_K}" in message
        assert "MAX_TOP_K" in message

    def test_top_k_at_the_cap_is_accepted(self) -> None:
        """边界值本身合法（上限是"超过才拒"，不是"达到就拒"）."""
        assert make_query(top_k=MAX_FETCH_K).top_k == MAX_FETCH_K

    def test_fetch_k_below_top_k_is_rejected(self) -> None:
        """深度小于条数时，那些"本来能进 top_k"的候选根本不会被取回来."""
        with pytest.raises(QueryError) as excinfo:
            make_query(top_k=5, fetch_k=2)

        message = str(excinfo.value)
        assert "fetch_k=2 小于 top_k=5" in message
        assert "召回深度必须 >= 要返回的条数" in message

    def test_fetch_k_equal_to_top_k_is_accepted(self) -> None:
        query = make_query(top_k=3, fetch_k=3)

        assert query.fetch_k == 3

    def test_fetch_k_zero_is_rejected(self) -> None:
        with pytest.raises(QueryError) as excinfo:
            make_query(fetch_k=0)

        assert "fetch_k 必须 >= 1" in str(excinfo.value)

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_min_score_is_rejected(self, value: float) -> None:
        """nan 与任何分数比较都是 False → 一条都不会留下，而报告里它只是一个阈值."""
        with pytest.raises(QueryError) as excinfo:
            make_query(min_score=value)

        assert "min_score 必须是有限数" in str(excinfo.value)

    @pytest.mark.parametrize("value", ["0.5", [0.5], {"min": 0.5}])
    def test_min_score_must_be_a_number(self, value: Any) -> None:
        with pytest.raises(QueryError) as excinfo:
            make_query(min_score=value)

        assert "min_score 必须是数字或 None" in str(excinfo.value)

    def test_min_score_accepts_int_and_float(self) -> None:
        """整数也是数字（``1`` 与 ``1.0`` 都是合法阈值）."""
        assert make_query(min_score=1).min_score == 1
        assert make_query(min_score=0.75).min_score == 0.75

    @pytest.mark.parametrize("value", [0, -1])
    def test_max_per_doc_must_be_at_least_one(self, value: Any) -> None:
        """查询里不允许写 0：''最多 0 条''不是一个请求，而是一个空结果."""
        with pytest.raises(QueryError) as excinfo:
            make_query(max_per_doc=value)

        message = str(excinfo.value)
        assert "max_per_doc" in message
        assert "必须 >= 1" in message

    @pytest.mark.parametrize("name", ["top_k", "fetch_k", "max_per_doc"])
    def test_three_integer_fields_reject_bool(self, name: str) -> None:
        """``True`` 也是 int（``isinstance(True, int)`` 为真），必须显式排除."""
        with pytest.raises(QueryError) as excinfo:
            make_query(**{name: True})

        assert f"{name} 必须是整数" in str(excinfo.value)

    def test_where_must_be_a_dict(self) -> None:
        with pytest.raises(QueryError) as excinfo:
            make_query(where=["strategy"])  # type: ignore[arg-type]

        message = str(excinfo.value)
        assert "where 必须是字典或 None" in message
        assert "{'strategy': 'structural'}" in message

    def test_where_syntax_is_validated_at_construction(self) -> None:
        """非法语法当场报错，并附上**全部**支持的运算符（写错的人能自己改对）."""
        with pytest.raises(QueryError) as excinfo:
            make_query(where={"strategy": {"$like": "struct%"}})

        message = str(excinfo.value)
        assert "where 子句不合法" in message
        assert "$like" in message
        assert [operator for operator in SUPPORTED_OPERATORS if operator not in message] == []

    def test_extra_must_be_a_dict(self) -> None:
        """``extra`` 是 day067 融合参数的预留槽，但也要求形状正确."""
        with pytest.raises(QueryError) as excinfo:
            make_query(extra=["alpha"])  # type: ignore[arg-type]

        assert "extra 必须是字典" in str(excinfo.value)

    def test_has_filter_covers_both_kinds(self) -> None:
        """元数据与时间范围都算过滤；空区间与空 where 都不算."""
        assert make_query().has_filter is False
        assert make_query(where={}).has_filter is False
        assert make_query(time_range=TimeRange()).has_filter is False
        assert make_query(where={"strategy": "structural"}).has_filter is True
        assert make_query(time_range=TimeRange(start="2026-09-01")).has_filter is True

    def test_to_dict_key_set_and_extra_is_optional(self) -> None:
        """``extra`` 为空时不进投影（响应体不该多出一个恒为空的键）."""
        plain = make_query(top_k=5, min_score=0.5).to_dict()

        assert set(plain) == {
            "text",
            "top_k",
            "fetch_k",
            "where",
            "time_range",
            "min_score",
            "max_per_doc",
            "route",
        }
        assert plain["time_range"] is None
        assert "extra" in make_query(extra={"alpha": 0.3}).to_dict()

    def test_to_dict_renders_nested_time_range(self) -> None:
        payload = make_query(time_range=TimeRange(start="2026-09-01")).to_dict()

        assert payload["time_range"]["describe"] == "created_at ∈ [2026-09-01, +∞]"

    def test_summary_line_mentions_filters(self) -> None:
        """摘要要么不提过滤，要么把过滤条件写出来（不能含糊）."""
        assert make_query().summary_line() == "查询 '语义缓存怎么配置阈值'"

        described = make_query(
            top_k=3,
            fetch_k=9,
            where={"strategy": "structural"},
            time_range=TimeRange(start="2026-09-01"),
        ).summary_line()

        assert "查询 '语义缓存怎么配置阈值'" in described
        assert "top_k=3" in described
        assert "fetch_k=9" in described
        assert "strategy" in described
        assert "created_at ∈ [2026-09-01, +∞]" in described


# --------------------------------------------------------------------------- #
# RetrievalHit：三个便捷属性与引用渲染
# --------------------------------------------------------------------------- #


class TestRetrievalHit:
    """命中的每个属性都对应下游一件事：分组、展示、引用."""

    def test_convenience_properties(self) -> None:
        hit = make_hit()

        assert hit.doc_id == "doc-alpha"
        assert hit.heading_path == "检索手册 > 阈值"
        assert hit.char_count == len("相似度阈值用分位数标定。")
        assert hit.channel == CHANNEL_VECTOR

    def test_missing_doc_id_falls_back_to_empty_string(self) -> None:
        """缺失时回落 ``""`` 而**不是** ``record_id``.

        回落到 record_id 会把"这条记录没有文档归属"伪装成"它自成一个文档"，
        于是多样性裁剪会把"一批没有 parent 的记录"当成不同文档全留下。
        """
        hit = make_hit(metadata={})

        assert hit.doc_id == ""
        assert hit.doc_id != hit.record_id

    def test_missing_heading_path_falls_back_to_empty_string(self) -> None:
        hit = make_hit(metadata={"parent_doc_id": "doc-beta"})

        assert hit.heading_path == ""

    def test_citation_prefers_source_then_doc_id_then_record_id(self) -> None:
        """引用标记的三级回落：``source`` → ``doc_id`` → ``record_id``."""
        assert make_hit().citation(1) == "[1] docs/检索手册.md › 检索手册 > 阈值"
        assert make_hit(metadata={"doc_id": "doc-x"}).citation(2) == "[2] doc-x"
        assert make_hit(metadata={}).citation(3) == "[3] c-a-01"

    def test_citation_omits_empty_heading(self) -> None:
        """标题路径为空时省略 ``›`` 那一段（不留一个悬空的分隔符）."""
        rendered = make_hit(metadata={"source": "docs/x.md"}).citation(1)

        assert rendered == "[1] docs/x.md"
        assert "›" not in rendered

    @pytest.mark.parametrize("index", [0, -1, 1.5, "1"])
    def test_citation_rejects_bad_index(self, index: Any) -> None:
        """编号从 1 起（编号是"在这次提示词里的位置"，由调用方给）."""
        with pytest.raises(QueryError) as excinfo:
            make_hit().citation(index)

        assert "引用编号从 1 起" in str(excinfo.value)

    def test_to_dict_key_set(self) -> None:
        payload = make_hit(score=0.123456789, rank=2).to_dict()

        assert set(payload) == {
            "rank",
            "record_id",
            "score",
            "doc_id",
            "heading_path",
            "char_count",
            "channel",
            "metadata",
            "text",
        }
        assert payload["rank"] == 2
        assert payload["score"] == 0.123457
        assert payload["doc_id"] == "doc-alpha"

    def test_to_dict_can_hide_text(self) -> None:
        """``include_text=False`` 用于"只看排序不变量"的场景（diff 不被正文淹没）."""
        lean = make_hit().to_dict(include_text=False)

        assert "text" not in lean
        assert lean["char_count"] > 0

    def test_to_dict_copies_metadata(self) -> None:
        """投影里的元数据是副本：改它不能影响那条命中本身."""
        hit = make_hit()

        hit.to_dict()["metadata"]["injected"] = True

        assert "injected" not in hit.metadata

    def test_summary_line_carries_rank_score_and_parent(self) -> None:
        line = make_hit(score=0.9, rank=3).summary_line()

        assert line.startswith("#3 c-a-01 [vector] score=+0.900000")
        assert "doc-alpha" in line

    def test_summary_line_marks_records_without_parent(self) -> None:
        """没有 parent 的命中要显式写成"（无 parent）"（不能让那一栏空着）."""
        line = make_hit(metadata={}).summary_line()

        assert "（无 parent）" in line

    @pytest.mark.parametrize("record_id", ["", "   "])
    def test_record_id_must_be_non_empty(self, record_id: str) -> None:
        """命中没有 id 就无法被引用、无法被去重、无法回库取原文."""
        with pytest.raises(QueryError) as excinfo:
            RetrievalHit(record_id=record_id, score=0.5, rank=0)

        assert "record_id 必须是非空字符串" in str(excinfo.value)

    @pytest.mark.parametrize("score", [float("nan"), float("inf")])
    def test_score_must_be_finite(self, score: float) -> None:
        """排序依赖它；nan 会让这条命中落到任意位置（且不报错）."""
        with pytest.raises(QueryError) as excinfo:
            RetrievalHit(record_id="c-a-01", score=score, rank=0)

        assert "score 必须是有限数" in str(excinfo.value)

    @pytest.mark.parametrize("rank", [-1, 1.5, "0"])
    def test_rank_must_be_a_non_negative_int(self, rank: Any) -> None:
        with pytest.raises(QueryError) as excinfo:
            RetrievalHit(record_id="c-a-01", score=0.5, rank=rank)

        assert "rank 必须是非负整数" in str(excinfo.value)

    def test_metadata_must_be_a_dict(self) -> None:
        with pytest.raises(QueryError) as excinfo:
            RetrievalHit(
                record_id="c-a-01",
                score=0.5,
                rank=0,
                metadata=["parent_doc_id"],  # type: ignore[arg-type]
            )

        assert "metadata 必须是字典" in str(excinfo.value)

    def test_channel_must_be_non_empty(self) -> None:
        """day067 的融合要靠它区分"这一条是哪一路召回"（空通道名做不到）."""
        with pytest.raises(QueryError) as excinfo:
            RetrievalHit(record_id="c-a-01", score=0.5, rank=0, channel="")

        assert "channel 必须是非空字符串" in str(excinfo.value)

    def test_hit_is_frozen(self) -> None:
        """frozen：一次检索的交代不该在传递过程中被就地改写."""
        hit = make_hit()

        with pytest.raises(Exception):
            hit.rank = 9  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# IndexState：版本与漂移
# --------------------------------------------------------------------------- #


class TestIndexState:
    """漂移是"清单与库相互矛盾"的统称；本课只观测、不阻断（除严格模式）."""

    def test_defaults_report_no_manifest(self) -> None:
        state = IndexState()

        assert state.version_id == ""
        assert state.count_store == 0
        assert state.count_manifest is None
        assert state.dimension == 0
        assert state.metric == ""
        assert state.drift == ()
        assert state.has_drift is False

    @pytest.mark.parametrize("drift", [(), ("一条不一致",)])
    def test_has_drift_follows_the_tuple(self, drift: tuple[str, ...]) -> None:
        assert IndexState(drift=drift).has_drift is bool(drift)

    def test_to_dict_key_set(self) -> None:
        payload = IndexState(
            version_id="0f2c1d3e4a5b6c7d",
            count_store=7,
            count_manifest=8,
            dimension=8,
            metric="cosine",
            drift=("清单里有 1 条记录在库里找不到",),
        ).to_dict()

        assert set(payload) == {
            "version_id",
            "count_store",
            "count_manifest",
            "dimension",
            "metric",
            "has_drift",
            "drift",
        }
        assert payload["count_manifest"] == 8
        assert payload["has_drift"] is True
        assert payload["drift"] == ["清单里有 1 条记录在库里找不到"]

    def test_summary_line_without_manifest(self) -> None:
        """没有清单时要写"（未提供清单）"与"（无）"，而不是留空或编一个版本号."""
        line = IndexState(count_store=6, dimension=8, metric="cosine").summary_line()

        assert "（未提供清单）" in line
        assert "清单 （无） 条" in line
        assert "无漂移" in line

    def test_summary_line_with_drift_counts_items(self) -> None:
        state = IndexState(
            version_id="0f2c1d3e4a5b6c7d",
            count_store=6,
            count_manifest=8,
            dimension=8,
            metric="ip",
            drift=("甲", "乙"),
        )

        line = state.summary_line()

        assert "0f2c1d3e4a5b6c7d" in line
        assert "漂移 2 项" in line
        assert "（未知度量）" not in line

    def test_summary_line_marks_unknown_metric(self) -> None:
        assert "（未知度量）" in IndexState(count_store=1).summary_line()


# --------------------------------------------------------------------------- #
# RetrievalResult：三道减法 + 四问诊断
# --------------------------------------------------------------------------- #


class TestRetrievalResultContract:
    """两个互相矛盾的字段必须当场拒绝（否则诊断失去意义）."""

    def test_default_empty_result(self) -> None:
        """手工构造的空结果不会自动改 ``empty_reason``——它由检索器负责填.

        这条写下来是为了让"``empty_reason`` 永远有值"这件事与"它由谁填"
        分开：默认值是哨兵 ``hits``，而空结果的原因只有检索器知道。
        """
        result = RetrievalResult(query=make_query())

        assert result.count == 0
        assert result.is_empty is True
        assert result.empty_reason == EMPTY_REASON_NONE
        assert result.top() is None

    def test_unknown_empty_reason_is_rejected(self) -> None:
        """诊断结论必须来自一份**封闭**的清单（否则它会慢慢长成自由文本）."""
        with pytest.raises(QueryError) as excinfo:
            make_result(empty_reason="大概是被过滤了吧")

        message = str(excinfo.value)
        assert "empty_reason 只能是" in message
        for reason in EMPTY_REASONS:
            assert reason in message

    @pytest.mark.parametrize(
        "reason",
        [EMPTY_REASON_NO_DATA, EMPTY_REASON_FILTERED_OUT, EMPTY_REASON_BELOW_THRESHOLD],
    )
    def test_hits_conflict_with_a_failure_reason(self, reason: str) -> None:
        """有命中却报告"空结果的原因"——这两个字段互相矛盾，不能并存."""
        with pytest.raises(QueryError) as excinfo:
            make_result(make_hit(), empty_reason=reason)

        assert "却报告 empty_reason" in str(excinfo.value)

    @pytest.mark.parametrize(
        "name",
        [
            "fetch_k",
            "candidates",
            "dropped_below_threshold",
            "dropped_by_diversity",
            "dropped_by_top_k",
        ],
    )
    def test_counters_must_be_non_negative(self, name: str) -> None:
        """负数意味着某处把"减出来的差值"当成了计数."""
        with pytest.raises(QueryError) as excinfo:
            make_result(**{name: -1})

        assert f"RetrievalResult.{name} 必须非负" in str(excinfo.value)

    def test_empty_reason_none_is_allowed_without_hits(self) -> None:
        """``hits`` 是哨兵值：空结果 + 它不矛盾（那是"还没来得及诊断"）. 见上一个用例."""
        assert make_result(empty_reason=EMPTY_REASON_NONE).is_empty is True

    def test_accessors(self) -> None:
        first = make_hit("c-a-01", score=1.0, rank=0)
        second = make_hit("c-c-02", score=0.6, rank=1, metadata={"parent_doc_id": "doc-gamma"})
        result = make_result(first, second, candidates=8, dropped_by_top_k=3)

        assert result.count == 2
        assert result.ids() == ["c-a-01", "c-c-02"]
        assert result.top() is first
        assert result.top_k == 5
        assert result.channels == ("vector",)
        assert result.doc_ids == ["doc-alpha", "doc-gamma"]
        assert result.dropped_by_top_k == 3

    def test_top_k_is_none_when_query_leaves_it_open(self) -> None:
        result = RetrievalResult(query=make_query())

        assert result.top_k is None

    def test_channels_dedupes_in_first_seen_order(self) -> None:
        """单路检索时永远是 ``("vector",)``；形状要先留出来给 day067 的 BM25."""
        result = make_result(
            make_hit("c-a-01", rank=0, channel="vector"),
            make_hit("c-a-02", rank=1, channel="bm25"),
            make_hit("c-a-03", rank=2, channel="vector"),
        )

        assert result.channels == ("vector", "bm25")

    def test_doc_ids_dedupes_in_first_seen_order(self) -> None:
        """``doc_ids`` 是多样性裁剪效果的直接证据（同一文档只出现一次）."""
        result = make_result(
            make_hit("c-a-01", rank=0),
            make_hit("c-a-02", rank=1),
            make_hit("c-b-01", rank=2, metadata={"parent_doc_id": "doc-beta"}),
        )

        assert result.doc_ids == ["doc-alpha", "doc-beta"]

    def test_to_dict_key_set(self) -> None:
        result = make_result(make_hit(), candidates=8, dropped_by_top_k=3)
        payload = result.to_dict()

        assert set(payload) == {
            "query",
            "count",
            "fetch_k",
            "candidates",
            "filter_applied",
            "dropped_below_threshold",
            "dropped_by_diversity",
            "dropped_by_top_k",
            "empty_reason",
            "metric",
            "index",
            "latency_ms",
            "channels",
            "doc_ids",
            "notes",
            "hits",
        }
        assert payload["count"] == 1
        assert payload["channels"] == ["vector"]
        assert payload["doc_ids"] == ["doc-alpha"]

    def test_to_dict_hides_text_by_default(self) -> None:
        """默认**不**带正文（与 ``VectorRecord.to_dict`` 相反）：检索报告要看的不是正文."""
        result = make_result(make_hit())

        assert "text" not in result.to_dict()["hits"][0]
        assert "text" in result.to_dict(include_text=True)["hits"][0]

    def test_summary_line_carries_the_three_numbers(self) -> None:
        line = make_result(make_hit(score=0.9), candidates=8, fetch_k=15).summary_line()

        assert line.startswith("cosine | 深度 15 | 候选 8 | 命中 1 | hits")
        assert "c-a-01:+0.9000" in line

    def test_summary_line_marks_empty(self) -> None:
        line = make_result(empty_reason=EMPTY_REASON_NO_DATA).summary_line()

        assert "（无）" in line
        assert EMPTY_REASON_NO_DATA in line


class TestExplainAnswersFourQuestions:
    """``explain()`` 的四行与四问一一对应，顺序固定（读它的人找的就是这四句）."""

    def test_minimal_result_has_exactly_four_lines(self) -> None:
        result = make_result(
            empty_reason=EMPTY_REASON_BELOW_THRESHOLD,
            candidates=8,
            dropped_below_threshold=8,
            fetch_k=15,
        )

        lines = result.explain()

        assert len(lines) == 4

    def test_line_one_answers_which_index_version(self) -> None:
        result = make_result(
            index_state=IndexState(
                version_id="0f2c1d3e4a5b6c7d",
                count_store=7,
                count_manifest=8,
                dimension=8,
                metric="cosine",
            )
        )

        line = result.explain()[0]

        assert "索引版本：0f2c1d3e4a5b6c7d" in line
        assert "库 7 条 / 清单 8 条" in line
        assert "8d cosine" in line
        assert "漂移 0 项" in line

    def test_line_one_says_no_manifest_when_absent(self) -> None:
        line = make_result(index_state=IndexState(count_store=6)).explain()[0]

        assert "（未提供清单，只报告库的现状）" in line
        assert "（无清单）" in line

    def test_line_two_answers_what_the_filter_was(self) -> None:
        """第二问：过滤条件是什么（含候选数——"过滤之后还剩几条可选"）."""
        result = make_result(
            query=make_query(
                top_k=5,
                where={"strategy": "structural"},
                time_range=TimeRange(start="2026-09-01"),
            ),
            filter_applied=True,
            candidates=4,
        )

        line = result.explain()[1]

        assert line.startswith("过滤条件：")
        assert "strategy" in line
        assert "created_at ∈ [2026-09-01, +∞]" in line
        assert "filter_applied=True" in line
        assert "过滤后候选 4 条" in line

    def test_line_two_says_no_filter(self) -> None:
        line = make_result(candidates=8).explain()[1]

        assert "（无过滤）" in line
        assert "filter_applied=False" in line

    def test_line_three_answers_why_fewer_than_top_k(self) -> None:
        """第三问：三道减法各自的数字（它们是"该调哪个参数"的唯一判据）."""
        result = make_result(
            make_hit(score=0.9, rank=0),
            query=make_query(top_k=5),
            fetch_k=15,
            candidates=8,
            dropped_below_threshold=2,
            dropped_by_diversity=1,
            dropped_by_top_k=4,
        )

        line = result.explain()[2]

        assert line.startswith("条数：1 / top_k=5（召回深度 15）")
        assert "阈值切掉 2 条" in line
        assert "每文档上限挤掉 1 条" in line
        assert "top_k 截断丢掉 4 条" in line

    def test_line_three_marks_default_top_k(self) -> None:
        """``top_k=None`` 时写"（默认）"而不是一个假的数字."""
        line = make_result(query=make_query()).explain()[2]

        assert "top_k=（默认）" in line

    def test_line_four_gives_the_single_reason_with_plain_words(self) -> None:
        """第四问：空结果的**唯一**原因 + 该去调什么."""
        result = make_result(
            empty_reason=EMPTY_REASON_FILTERED_OUT,
            filter_applied=True,
            candidates=0,
        )

        line = result.explain()[3]

        assert f"空结果原因：{EMPTY_REASON_FILTERED_OUT}" in line
        assert EMPTY_REASON_DESCRIPTIONS[EMPTY_REASON_FILTERED_OUT] in line

    def test_drift_and_notes_are_appended_after_the_four_questions(self) -> None:
        """漂移明细与注记追加在后面：它们不属于四问，但"结果可能不完整"必须被看见."""
        result = make_result(
            make_hit(),
            index_state=IndexState(version_id="0f2c1d3e4a5b6c7d", drift=("清单里有 1 条少了",)),
            notes=("索引漂移 1 项：清单里有 1 条少了",),
        )

        lines = result.explain()

        assert len(lines) == 7
        assert lines[4] == "漂移明细：清单里有 1 条少了"
        assert lines[5].startswith("注记：")
        assert lines[6].startswith("前 1 条：c-a-01")

    def test_preview_lists_at_most_three_ids(self) -> None:
        result = make_result(
            *[make_hit(f"c-a-0{index}", rank=index) for index in range(1, 6)]
        )

        preview = result.explain()[-1]

        assert preview == "前 3 条：c-a-01、c-a-02、c-a-03"


class TestAggregateResults:
    """批量汇总：空批次给全 0 一行，而不是 ``ZeroDivisionError``."""

    def test_empty_batch_returns_zero_row(self) -> None:
        summary = aggregate_results([])

        assert summary == {
            "count": 0,
            "hits": 0,
            "avg_hits": 0.0,
            "empty": 0,
            "empty_rate": 0.0,
            "empty_reasons": {},
            "avg_fetch_k": 0.0,
            "depth_hit_rate": 0.0,
            "avg_latency_ms": 0.0,
            "with_drift": 0,
            "channels": [],
        }

    def test_mixed_batch_counts_and_rates(self) -> None:
        """三条结果：一条两条命中、一条被阈值切空、一条带漂移."""
        summary = aggregate_results(
            [
                make_result(
                    make_hit("c-a-01", rank=0),
                    make_hit("c-a-02", rank=1),
                    fetch_k=4,
                    candidates=8,
                    latency_ms=2.0,
                ),
                make_result(
                    empty_reason=EMPTY_REASON_BELOW_THRESHOLD,
                    fetch_k=15,
                    candidates=8,
                    dropped_below_threshold=8,
                    latency_ms=4.0,
                ),
                make_result(
                    make_hit("c-b-01", rank=0),
                    fetch_k=15,
                    candidates=8,
                    latency_ms=6.0,
                    index_state=IndexState(version_id="0f2c1d3e4a5b6c7d", drift=("有一个",)),
                ),
            ]
        )

        assert summary["count"] == 3
        assert summary["hits"] == 3
        assert summary["avg_hits"] == 1.0
        assert summary["empty"] == 1
        assert summary["empty_rate"] == 0.3333
        assert summary["empty_reasons"] == {EMPTY_REASON_BELOW_THRESHOLD: 1}
        assert summary["avg_fetch_k"] == 11.3333
        # 三次的 命中/深度 分别是 2/4、0/15、1/15，平均后是 0.1889
        assert summary["depth_hit_rate"] == 0.1889
        assert summary["avg_latency_ms"] == 4.0
        assert summary["with_drift"] == 1
        assert summary["channels"] == ["vector"]

    def test_depth_hit_rate_measures_depth_not_recall(self) -> None:
        """``depth_hit_rate`` = 平均(命中数 / 召回深度)：接近 1 说明深度几乎没被浪费."""
        wider = aggregate_results([make_result(make_hit(), fetch_k=10, candidates=10)])
        tighter = aggregate_results([make_result(make_hit(), fetch_k=2, candidates=10)])

        assert wider["depth_hit_rate"] == 0.1
        assert tighter["depth_hit_rate"] == 0.5

    def test_zero_depth_results_are_skipped_in_the_ratio(self) -> None:
        """``fetch_k == 0`` 的结果进不了比值（否则会除零）；``avg_fetch_k`` 仍然统计它."""
        summary = aggregate_results(
            [
                make_result(fetch_k=0, candidates=0, empty_reason=EMPTY_REASON_NO_DATA),
                make_result(make_hit(), fetch_k=4, candidates=4),
            ]
        )

        assert summary["avg_fetch_k"] == 2.0
        assert summary["depth_hit_rate"] == 0.25

    def test_channels_are_unioned_across_results(self) -> None:
        """这批结果里出现过的通道要去重并按首次出现顺序排列（融合前要分配权重）."""
        summary = aggregate_results(
            [
                make_result(make_hit("c-a-01", rank=0, channel="vector")),
                make_result(make_hit("c-b-01", rank=0, channel="bm25")),
                make_result(make_hit("c-c-01", rank=0, channel="vector")),
            ]
        )

        assert summary["channels"] == ["vector", "bm25"]

    def test_empty_reasons_are_sorted(self) -> None:
        """原因分布按 key 排序（报告要能被逐行 diff）."""
        summary = aggregate_results(
            [
                make_result(empty_reason=EMPTY_REASON_NO_DATA, fetch_k=0),
                make_result(empty_reason=EMPTY_REASON_FILTERED_OUT, filter_applied=True),
                make_result(empty_reason=EMPTY_REASON_NO_DATA, fetch_k=0),
            ]
        )

        assert list(summary["empty_reasons"]) == [
            EMPTY_REASON_FILTERED_OUT,
            EMPTY_REASON_NO_DATA,
        ]
        assert summary["empty_reasons"][EMPTY_REASON_NO_DATA] == 2
        assert summary["empty_rate"] == 1.0

    def test_all_drift_batch_counts_with_drift(self) -> None:
        drifted = IndexState(version_id="0f2c1d3e4a5b6c7d", drift=("有一条不一致",))
        summary = aggregate_results(
            [
                make_result(make_hit(), index_state=drifted),
                make_result(make_hit(), index_state=drifted),
            ]
        )

        assert summary["with_drift"] == 2
