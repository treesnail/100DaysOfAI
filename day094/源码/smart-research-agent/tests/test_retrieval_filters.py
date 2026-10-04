"""day066 ``retrieval.filters`` 的单元测试：子句形状、冲突检测、字段拼写检查.

``vectorstore.filters`` 已经把 ``where`` 编译成谓词了，这一层存在的理由是它
回答了那层**故意不回答**的三个问题，因此本文件的断言按这三个问题分组：

```text
1. 时间范围怎么变成 where？          → 双端 / 单端 / 空 三种形状，逐个精确比对
2. where 与 time_range 撞车怎么办？   → 当场 QueryError（不许猜，也不许静默取交集）
3. 字段名是不是写错了？               → unknown_filter_fields 与"能发现什么/不能发现什么"
```

全部离线、确定性。时间比较一律走 ``$gte`` / ``$lte`` 的**字典序**，
因此断言里用的都是同一批 ISO 文本（见 ``types.TimeRange`` 的三条约定）。
"""

from __future__ import annotations

from typing import Any

import pytest

from smart_research_agent.retrieval.errors import QueryError
from smart_research_agent.retrieval.filters import (
    DEFAULT_TIME_FIELD,
    TIME_FIELD_SUFFIXES,
    combine_where,
    describe_conditions,
    time_field_of,
    time_range_clause,
    unknown_filter_fields,
    validate_where,
)
from smart_research_agent.retrieval.types import TimeRange
from smart_research_agent.vectorstore.filters import (
    SUPPORTED_OPERATORS,
    compile_filter,
    match_metadata,
)
from tests.retrieval_samples import MISSING_CREATED_AT_ID, metadata_field_names, sample_metadata

# --------------------------------------------------------------------------- #
# 时间范围 → 子句
# --------------------------------------------------------------------------- #


class TestTimeRangeClause:
    """三种形状与 ``TimeRange`` 的三条约定一一对应."""

    def test_empty_range_becomes_empty_dict(self) -> None:
        """两端都没给 → ``{}``（"这次不涉及时间条件"而不是"条件恒真"）."""
        assert time_range_clause(None) == {}
        assert time_range_clause(TimeRange()) == {}

    def test_both_bounds_use_one_and_layer(self) -> None:
        """双端必须用 ``$and`` 包两根条件：``vectorstore`` 拒绝同一字段的多个运算符."""
        clause = time_range_clause(TimeRange(start="2026-09-01", end="2026-09-30"))

        assert clause == {
            "$and": [
                {"created_at": {"$gte": "2026-09-01"}},
                {"created_at": {"$lte": "2026-09-30"}},
            ]
        }

    def test_single_bounds_are_plain_conditions(self) -> None:
        """单端不加壳（加一层 ``$and`` 只会给报告增加噪声）."""
        assert time_range_clause(TimeRange(start="2026-09-01")) == {
            "created_at": {"$gte": "2026-09-01"}
        }
        assert time_range_clause(TimeRange(end="2026-09-30")) == {
            "created_at": {"$lte": "2026-09-30"}
        }

    def test_field_name_follows_the_range(self) -> None:
        clause = time_range_clause(TimeRange(field="indexed_at", start="2026-09-01"))

        assert clause == {"indexed_at": {"$gte": "2026-09-01"}}

    @pytest.mark.parametrize(
        "span",
        [
            TimeRange(start="2026-09-01"),
            TimeRange(end="2026-09-30"),
            TimeRange(start="2026-09-01", end="2026-09-30"),
        ],
    )
    def test_clause_is_legal_where(self, span: TimeRange) -> None:
        """生成的子句必须能直接被 ``vectorstore.compile_filter`` 编译.

        这条断言的意义：形状对不对不该靠人眼看——形状写错时
        ``compile_filter`` 会抛 ``FilterError``，而那个错误出现在**检索**路径上，
        排查时看起来像是"过滤条件写错了"（其实是检索器自己拼错了）。
        """
        compile_filter(time_range_clause(span))

    def test_clause_semantics_are_inclusive_on_both_ends(self) -> None:
        """闭区间：两端当日都算命中（字典序比较，``$gte`` + ``$lte``）."""
        span = TimeRange(start="2026-09-01", end="2026-09-30")
        predicate = compile_filter(time_range_clause(span))

        assert predicate({"created_at": "2026-09-01"}) is True
        assert predicate({"created_at": "2026-09-30"}) is True
        assert predicate({"created_at": "2026-08-31"}) is False
        assert predicate({"created_at": "2026-10-01"}) is False

    def test_records_without_the_field_are_excluded(self) -> None:
        """**字段缺失的记录被排除**（``$gte`` 对缺失键返回 False），这不是 bug.

        替代方案（把缺失当 0 或当"现在"）会给"这条什么时候建的"编一个答案，
        而错误的时间过滤**不报错**，只表现为"过滤之后少了几条"。
        """
        predicate = compile_filter(time_range_clause(TimeRange(start="2026-09-01")))

        assert predicate({"strategy": "fixed"}) is False

    def test_sample_record_without_created_at_is_excluded(self) -> None:
        """用样本库里那条唯一的"缺 ``created_at``"记录把上一条落到实处."""
        metadata = sample_metadata()[MISSING_CREATED_AT_ID]
        assert "created_at" not in metadata

        predicate = compile_filter(
            time_range_clause(TimeRange(start="2026-01-01", end="2026-12-31"))
        )

        assert predicate(metadata) is False


# --------------------------------------------------------------------------- #
# where 与时间范围的合成
# --------------------------------------------------------------------------- #


class TestCombineWhere:
    """四种输入组合四种输出；冲突当场报错."""

    def test_nothing_to_combine_returns_none(self) -> None:
        """两者都空 → ``None``（"没有过滤"与"空字典过滤"在报告里必须能区分）."""
        assert combine_where(None, None) is None
        assert combine_where(None, TimeRange()) is None
        assert combine_where({}, None) is None
        assert combine_where({}, TimeRange()) is None

    def test_only_where_is_returned_as_is(self) -> None:
        """只有 ``where`` 时**不做任何包装**（包装只会给报告加一层噪声）."""
        where = {"strategy": "structural"}

        assert combine_where(where, None) == where
        assert combine_where(where, TimeRange()) == where

    def test_only_time_range_returns_the_clause(self) -> None:
        combined = combine_where(None, TimeRange(start="2026-09-01"))

        assert combined == {"created_at": {"$gte": "2026-09-01"}}
        assert combine_where({}, TimeRange(end="2026-09-30")) == {
            "created_at": {"$lte": "2026-09-30"}
        }

    def test_both_are_joined_with_and(self) -> None:
        """都有 → ``$and`` 一层，顺序是"先 where、后时间"（与报告的可读顺序一致）."""
        where = {"strategy": "structural"}

        combined = combine_where(where, TimeRange(start="2026-09-01", end="2026-09-30"))

        assert combined == {
            "$and": [
                where,
                {
                    "$and": [
                        {"created_at": {"$gte": "2026-09-01"}},
                        {"created_at": {"$lte": "2026-09-30"}},
                    ]
                },
            ]
        }

    def test_combined_clause_is_still_legal(self) -> None:
        """合成结果同样要能被编译（合成是这一层最容易拼出非法形状的地方）."""
        combined = combine_where(
            {"strategy": {"$in": ["fixed", "structural"]}},
            TimeRange(start="2026-09-01", end="2026-09-30"),
        )

        compile_filter(combined)

    def test_combined_clause_semantics(self) -> None:
        combined = combine_where(
            {"strategy": "structural"}, TimeRange(start="2026-09-01", end="2026-09-30")
        )

        assert match_metadata(
            {"strategy": "structural", "created_at": "2026-09-15"}, combined
        ) is True
        assert match_metadata(
            {"strategy": "fixed", "created_at": "2026-09-15"}, combined
        ) is False
        assert match_metadata(
            {"strategy": "structural", "created_at": "2026-01-01"}, combined
        ) is False

    def test_time_range_inside_where_conflicts(self) -> None:
        """同一个字段的两组条件 → ``QueryError``，消息里给出"请自己合并成一个显式区间"."""
        where = {"created_at": {"$gt": "2026-01-01"}}

        with pytest.raises(QueryError) as excinfo:
            combine_where(where, TimeRange(start="2026-06-01"))

        message = str(excinfo.value)
        assert "过滤条件冲突" in message
        assert "'created_at'" in message
        assert "请自己合并成一个显式区间" in message

    def test_conflict_is_found_inside_nested_clauses(self) -> None:
        """嵌套子句里的同名字段也要被发现（``filter_fields`` 会递归、并跳过逻辑键）."""
        where = {
            "$and": [
                {"created_at": {"$gte": "2026-01-01"}},
                {"strategy": "structural"},
            ]
        }

        with pytest.raises(QueryError) as excinfo:
            combine_where(where, TimeRange(start="2026-06-01"))

        assert "过滤条件冲突" in str(excinfo.value)

    def test_conflict_uses_the_time_ranges_own_field(self) -> None:
        """判定用的是 ``time_range.field`` 精确比较，而不是"看起来像时间的名字"."""
        where = {"published_at": {"$gt": "2026-01-01"}}

        combined = combine_where(where, TimeRange(start="2026-06-01"))

        assert combined == {
            "$and": [where, {"created_at": {"$gte": "2026-06-01"}}]
        }

    def test_empty_time_range_means_no_conflict(self) -> None:
        """``TimeRange`` 两端都没给时它没有任何条件，此时同名字段只是一次普通过滤."""
        where = {"created_at": "2026-09-01"}

        assert combine_where(where, TimeRange()) == where

    def test_where_inside_and_layer_conflicts_with_nested_time_range(self) -> None:
        """时间字段出现在 ``$and`` 的第二层也要被检出（递归深度不设限）."""
        where = {"$and": [{"strategy": "fixed"}, {"$and": [{"created_at": "2026-01-01"}]}]}

        with pytest.raises(QueryError):
            combine_where(where, TimeRange(end="2026-06-01"))


# --------------------------------------------------------------------------- #
# 语法校验：把 FilterError 包成 QueryError
# --------------------------------------------------------------------------- #


class TestValidateWhere:
    """写错过滤条件的是**调用方**，因此它该收到 ``QueryError`` 而不是 ``FilterError``."""

    def test_none_and_empty_are_legal(self) -> None:
        validate_where(None)
        validate_where({})

    def test_scalar_shorthand_is_legal(self) -> None:
        validate_where({"strategy": "structural"})
        validate_where({"token_count": {"$gt": 100}})
        validate_where({"$and": [{"a": 1}, {"b": 2}]})

    def test_non_dict_is_rejected_with_examples(self) -> None:
        with pytest.raises(QueryError) as excinfo:
            validate_where(["strategy"])  # type: ignore[arg-type]

        message = str(excinfo.value)
        assert "where 必须是字典或 None" in message
        assert "token_count" in message

    def test_unknown_operator_lists_every_supported_one(self) -> None:
        """消息里要带上**全部**支持的运算符：读它的人在调检索器，不是在调库."""
        with pytest.raises(QueryError) as excinfo:
            validate_where({"strategy": {"$like": "struct%"}})

        message = str(excinfo.value)
        assert "where 子句不合法" in message
        assert "$like" in message
        assert [operator for operator in SUPPORTED_OPERATORS if operator not in message] == []

    @pytest.mark.parametrize(
        "where",
        [
            {"index": {"$gt": 1, "$lt": 9}},  # 同一字段多个运算符
            {"$and": [{"a": 1}], "index": 2},  # 逻辑运算符与字段条件混写
            {"index": {}},  # 空条件字典
            {"$or": []},  # 空子句列表
            {"$and": [{"a": 1}, 5]},  # 子句元素不是字典
            {"tags": {"$in": []}},  # $in 的空列表
        ],
    )
    def test_common_typos_are_wrapped_into_query_error(self, where: dict[str, Any]) -> None:
        """六种典型写法错误都必须变成 ``QueryError``（调用方那一边的问题）."""
        with pytest.raises(QueryError) as excinfo:
            validate_where(where)

        assert "where 子句不合法" in str(excinfo.value)

    def test_error_keeps_the_original_reason(self) -> None:
        """包一层不能丢掉原始原因（否则"到底哪里不合法"就查不到了）."""
        with pytest.raises(QueryError) as excinfo:
            validate_where({"index": {"$gt": 1, "$lt": 9}})

        assert "多个运算符" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# 一行人类可读描述
# --------------------------------------------------------------------------- #


class TestDescribeConditions:
    """报告里的连接词必须与执行语义一致：两段用"且"连接，不用逗号."""

    def test_no_filter(self) -> None:
        assert describe_conditions(None) == "（无过滤）"
        assert describe_conditions(None, None) == "（无过滤）"
        assert describe_conditions({}, None) == "（无过滤）"
        assert describe_conditions({}, TimeRange()) == "（无过滤）"

    def test_where_only_falls_back_to_json(self) -> None:
        """``where`` 部分复用 ``describe_filter``（``json.dumps``，保真嵌套结构）."""
        assert describe_conditions({"strategy": "structural"}) == '{"strategy": "structural"}'
        assert (
            describe_conditions({"token_count": {"$gt": 100}})
            == '{"token_count": {"$gt": 100}}'
        )

    def test_time_only(self) -> None:
        assert (
            describe_conditions(None, TimeRange(start="2026-09-01"))
            == "created_at ∈ [2026-09-01, +∞]"
        )

    def test_empty_time_range_is_ignored(self) -> None:
        """空区间不产生任何文字（写出来会让人以为"这次有时间条件"）."""
        assert describe_conditions({"strategy": "fixed"}, TimeRange()) == '{"strategy": "fixed"}'

    def test_both_are_joined_by_the_semantic_word(self) -> None:
        described = describe_conditions(
            {"strategy": "structural"}, TimeRange(start="2026-09-01", end="2026-09-30")
        )

        assert described == (
            '{"strategy": "structural"} 且 created_at ∈ [2026-09-01, 2026-09-30]'
        )
        assert "，" not in described

    def test_combined_clause_and_description_agree(self) -> None:
        """描述里的两个条件确实同时生效（描述与语义必须一致）."""
        combined = combine_where(
            {"strategy": "structural"}, TimeRange(start="2026-09-01")
        )
        described = describe_conditions(
            {"strategy": "structural"}, TimeRange(start="2026-09-01")
        )

        assert match_metadata({"strategy": "structural", "created_at": "2026-09-05"}, combined)
        assert "strategy" in described and "created_at" in described


# --------------------------------------------------------------------------- #
# 字段拼写检查
# --------------------------------------------------------------------------- #


class TestUnknownFilterFields:
    """它只能发现"整个库里都没有这个字段"，绝不能当成万能的检查器."""

    def test_empty_where_has_no_unknown_fields(self) -> None:
        assert unknown_filter_fields(None, ["strategy"]) == []
        assert unknown_filter_fields({}, ["strategy"]) == []

    def test_known_fields_are_not_reported(self) -> None:
        """样本库的字段全集里都有 → 一个都不报（"没出现过"才是判据）."""
        known = metadata_field_names()

        assert unknown_filter_fields({"strategy": "structural", "topic": "cache"}, known) == []

    def test_typo_is_reported(self) -> None:
        """``stratgey`` 在整个库里都没有出现过 → 报出来（这就是它的用途）."""
        unknown = unknown_filter_fields({"stratgey": "structural"}, metadata_field_names())

        assert unknown == ["stratgey"]

    def test_nested_typos_are_reported_sorted(self) -> None:
        where = {
            "$and": [
                {"stratgey": "structural"},
                {"$or": [{"heading": "x"}, {"topic": "cache"}]},
            ]
        }

        assert unknown_filter_fields(where, metadata_field_names()) == ["heading", "stratgey"]

    def test_value_typos_are_invisible(self) -> None:
        """**值写错发现不了**：键是对的，而值域是业务知识，不是元数据事实."""
        assert unknown_filter_fields({"strategy": "structual"}, metadata_field_names()) == []

    def test_bad_iso_date_is_invisible_too(self) -> None:
        """键对、格式错也发现不了——那是 ``TimeRange`` 校验器的事."""
        unknown = unknown_filter_fields(
            {"created_at": {"$gte": "2026-99-01"}}, metadata_field_names()
        )

        assert unknown == []

    def test_field_present_in_only_some_records_is_known(self) -> None:
        """只要有一条记录带这个键，它就算"出现过"（``created_at`` 在样本里只出现 7 次）."""
        known = metadata_field_names()

        assert "created_at" in known
        assert "created_at" not in sample_metadata()[MISSING_CREATED_AT_ID]
        assert unknown_filter_fields({"created_at": "2026-09-01"}, known) == []

    def test_field_names_argument_can_be_any_iterable(self) -> None:
        """它接受任意可迭代对象（调用方常常只有一个 ``set``）."""
        assert unknown_filter_fields({"topic": "x"}, {"topic"}) == []
        assert unknown_filter_fields({"topic": "x"}, {"parent_doc_id"}) == ["topic"]


# --------------------------------------------------------------------------- #
# 时间字段名的猜测
# --------------------------------------------------------------------------- #


class TestTimeFieldOf:
    """它**不参与任何判定**，只用于报告与提示（命名约定永远有例外）."""

    def test_default_field_is_created_at(self) -> None:
        assert DEFAULT_TIME_FIELD == "created_at"
        assert time_field_of(None) == DEFAULT_TIME_FIELD
        assert time_field_of({}) == DEFAULT_TIME_FIELD

    def test_default_name_present_wins(self) -> None:
        """``where`` 里直接出现默认字段名时就是它（不看后缀）."""
        where = {"created_at": "2026-09-01", "published_at": "2026-01-01"}

        assert time_field_of(where) == "created_at"

    @pytest.mark.parametrize("suffix", TIME_FIELD_SUFFIXES)
    def test_suffix_candidates_are_recognised(self, suffix: str) -> None:
        name = f"published{suffix}"

        assert time_field_of({name: "2026-09-01"}) == name

    def test_two_candidates_pick_the_first_in_sorted_order(self) -> None:
        """两个都像时间字段时取字典序第一个（确定性，不依赖插入顺序）."""
        where = {"updated_at": "2026-09-01", "created_ts": "2026-09-01"}

        assert time_field_of(where) == "created_ts"

    def test_non_time_fields_fall_back_to_default(self) -> None:
        assert time_field_of({"strategy": "fixed", "topic": "cache"}) == DEFAULT_TIME_FIELD

    def test_custom_default_is_used(self) -> None:
        assert time_field_of({"strategy": "fixed"}, "indexed_at") == "indexed_at"
        assert time_field_of({"indexed_at": "2026-09-01"}, "created_at") == "indexed_at"

    def test_it_never_raises_on_odd_shapes(self) -> None:
        """它是提示，不是判定：畸形的 ``where`` 不该让它抛异常."""
        assert time_field_of({"$and": [{"created_at": "2026-09-01"}]}) == "created_at"
        assert time_field_of(True) == DEFAULT_TIME_FIELD  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 模块导出
# --------------------------------------------------------------------------- #


class TestFilterModuleExports:
    """工具函数都在 ``__all__`` 里（端点与检索器都从这一层取）."""

    def test_all_names_are_importable(self) -> None:
        from smart_research_agent.retrieval import filters as module

        for name in ("combine_where", "describe_conditions", "time_field_of",
                     "time_range_clause", "unknown_filter_fields", "validate_where"):
            assert name in module.__all__
            assert callable(getattr(module, name))

    def test_default_time_field_matches_the_range_default(self) -> None:
        """三处默认值必须是同一个常量（``filters`` / ``types`` / ``config``）."""
        assert TimeRange().field == DEFAULT_TIME_FIELD
