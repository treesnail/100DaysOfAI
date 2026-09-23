"""day064 ``vectorstore.filters`` 的单元测试：12 个运算符与 6 种写法错误.

全部离线、确定性。这一层的写法有一个刻意的取舍：
**用 Chroma 的语法作为本包的规范语法**（``$eq``/``$gt``/``$in``/``$and``…），
因为那已经是一份被真实实现的规格，自己发明一套只会多一个翻译层。

两类断言各自回答一个问题：

```text
运算符语义    这条记录该不该被这条条件选中（逐条 True/False）
写法错误      条件本身不合法时，报错里有没有"怎么改"（列出全部支持项）
```

还有两条"不报错"是刻意的（见 ``filters`` 模块 docstring）：
类型不可比的比较判为不命中、``$nin`` 对缺失键算命中。
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from smart_research_agent.vectorstore.errors import FilterError
from smart_research_agent.vectorstore.filters import (
    SUPPORTED_OPERATORS,
    compile_filter,
    describe_filter,
    filter_fields,
    match_metadata,
)

#: 一份类型齐全的元数据（str / int / bool / 同类型数组）.
METADATA: dict[str, Any] = {
    "parent_doc_id": "b41ad29c8f70e613",
    "strategy": "structural",
    "index": 2,
    "token_count": 120,
    "oversized": False,
    "tags": ["guide", "cost"],
    "heading_path": "语义缓存 > 阈值",
}

#: 12 个运算符 × 正反例。**顺序就是 ``SUPPORTED_OPERATORS`` 的顺序往下的展开**，
#: 每一条都只改一个东西，好让"哪个运算符坏了"一眼能定位。
OPERATOR_TABLE: tuple[tuple[dict[str, Any], bool], ...] = (
    # $eq
    ({"strategy": {"$eq": "structural"}}, True),
    ({"strategy": {"$eq": "fixed"}}, False),
    # $ne
    ({"strategy": {"$ne": "fixed"}}, True),
    ({"strategy": {"$ne": "structural"}}, False),
    # $gt
    ({"index": {"$gt": 1}}, True),
    ({"index": {"$gt": 2}}, False),
    # $gte
    ({"index": {"$gte": 2}}, True),
    ({"index": {"$gte": 3}}, False),
    # $lt
    ({"index": {"$lt": 3}}, True),
    ({"index": {"$lt": 2}}, False),
    # $lte
    ({"index": {"$lte": 2}}, True),
    ({"index": {"$lte": 1}}, False),
    # $in
    ({"strategy": {"$in": ["fixed", "structural"]}}, True),
    ({"strategy": {"$in": ["fixed"]}}, False),
    # $nin
    ({"strategy": {"$nin": ["fixed"]}}, True),
    ({"strategy": {"$nin": ["structural"]}}, False),
    # $contains（字段必须是数组）
    ({"tags": {"$contains": "cost"}}, True),
    ({"tags": {"$contains": "search"}}, False),
    # $not_contains
    ({"tags": {"$not_contains": "search"}}, True),
    ({"tags": {"$not_contains": "cost"}}, False),
    # $and
    ({"$and": [{"strategy": "structural"}, {"index": {"$gt": 1}}]}, True),
    ({"$and": [{"strategy": "structural"}, {"index": {"$gt": 2}}]}, False),
    # $or
    ({"$or": [{"strategy": "fixed"}, {"index": {"$gte": 2}}]}, True),
    ({"$or": [{"strategy": "fixed"}, {"index": {"$lt": 2}}]}, False),
)


@pytest.mark.parametrize(("where", "expected"), OPERATOR_TABLE)
def test_every_operator_has_a_positive_and_negative_case(
    where: dict[str, Any], expected: bool
) -> None:
    """12 个运算符逐条验证（``is True`` / ``is False``，不接受真值等价物）."""
    assert match_metadata(METADATA, where) is expected


def test_operator_table_is_complete() -> None:
    """表里必须出现**全部** 12 个运算符——漏一个就等于那一行没被覆盖."""
    rendered = "".join(repr(where) for where, _ in OPERATOR_TABLE)

    assert len(SUPPORTED_OPERATORS) == 12
    assert [operator for operator in SUPPORTED_OPERATORS if operator not in rendered] == []


def test_scalar_value_is_shorthand_for_eq() -> None:
    """``{"strategy": "structural"}`` 与 ``{"strategy": {"$eq": ...}}`` 等价."""
    assert match_metadata(METADATA, {"strategy": "structural"}) is True
    assert match_metadata(METADATA, {"strategy": {"$eq": "structural"}}) is True
    assert match_metadata(METADATA, {"strategy": "fixed"}) is False


def test_nested_and_or() -> None:
    """``$and`` 里套 ``$or``：内层任一为真、外层两项都真才算命中."""
    hits = {
        "$and": [
            {"$or": [{"strategy": "fixed"}, {"strategy": "structural"}]},
            {"index": {"$lt": 5}},
        ]
    }
    misses = {
        "$and": [
            {"$or": [{"strategy": "fixed"}, {"strategy": "semantic"}]},
            {"index": {"$lt": 5}},
        ]
    }

    assert match_metadata(METADATA, hits) is True
    assert match_metadata(METADATA, misses) is False


def test_compile_none_and_empty_dict_are_always_true() -> None:
    """``None`` 与 ``{}`` 都恒真，但报告里的含义不同（见 ``describe_filter``）."""
    assert compile_filter(None)({}) is True
    assert compile_filter(None)({"anything": 1}) is True
    assert compile_filter({})({}) is True
    assert match_metadata(METADATA, None) is True


# --------------------------------------------------------------------------- #
# 写法错误：报错必须能指导下一步
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("operator", ["$in", "$nin"])
def test_in_and_nin_reject_empty_list(operator: str) -> None:
    """空列表会让命中集恒为空，因此直接拒绝而不是静默返回空结果."""
    with pytest.raises(FilterError) as excinfo:
        compile_filter({"tags": {operator: []}})

    message = str(excinfo.value)
    assert operator in message
    assert "非空列表" in message


def test_unknown_operator_lists_all_supported() -> None:
    """未知运算符的报错要列出全部支持项（写错的查询能自己改对）."""
    with pytest.raises(FilterError) as excinfo:
        compile_filter({"strategy": {"$like": "str%"}})

    message = str(excinfo.value)
    assert "$like" in message
    for operator in SUPPORTED_OPERATORS:
        assert operator in message


def test_unknown_dollar_key_at_top_level() -> None:
    """顶层以 ``$`` 开头但不在支持表里 → 未知运算符（而不是"字段名"）."""
    with pytest.raises(FilterError) as excinfo:
        compile_filter({"$nor": [{"index": 1}]})

    message = str(excinfo.value)
    assert "$nor" in message
    assert "$and" in message


def test_logical_operator_cannot_mix_with_field_conditions() -> None:
    """``$and`` 与字段条件混写会让语义不确定，编译期直接拒掉."""
    with pytest.raises(FilterError) as excinfo:
        compile_filter({"$and": [{"strategy": "structural"}], "index": 2})

    assert "必须单独占一层" in str(excinfo.value)


def test_field_with_multiple_operators_is_rejected() -> None:
    """同一字段的多个条件必须用 ``$and`` 组合，不能塞进同一个字典."""
    with pytest.raises(FilterError) as excinfo:
        compile_filter({"index": {"$gt": 1, "$lt": 9}})

    assert "多个运算符" in str(excinfo.value)


def test_field_with_empty_condition_is_rejected() -> None:
    """``{"index": {}}`` 通常是写漏了运算符，不能当成"字段存在"."""
    with pytest.raises(FilterError) as excinfo:
        compile_filter({"index": {}})

    assert "空字典" in str(excinfo.value)


def test_logical_operator_inside_field_is_rejected() -> None:
    """``$and`` 组合的是条件（不是字段值），写在字段内部一定是写错了."""
    with pytest.raises(FilterError) as excinfo:
        compile_filter({"tags": {"$and": [{"a": 1}]}})

    assert "不能写在字段内部" in str(excinfo.value)


@pytest.mark.parametrize("operator", ["$and", "$or"])
def test_logical_operator_needs_non_empty_list(operator: str) -> None:
    """``$and`` / ``$or`` 的值必须是非空子句列表（``[]`` 与标量都拒绝）."""
    with pytest.raises(FilterError) as excinfo:
        compile_filter({operator: []})
    assert "非空列表" in str(excinfo.value)

    with pytest.raises(FilterError):
        compile_filter({operator: {"index": 1}})


def test_logical_operator_items_must_be_dicts() -> None:
    """子句列表里的元素必须本身是子句（字典）."""
    with pytest.raises(FilterError) as excinfo:
        compile_filter({"$and": [{"index": 1}, 5]})

    assert "必须是字典" in str(excinfo.value)


def test_where_must_be_a_dict() -> None:
    """``where`` 不是字典时给出写法示例（而不是让它去深处炸）."""
    with pytest.raises(FilterError) as excinfo:
        compile_filter(["strategy"])  # type: ignore[arg-type]

    assert "where 必须是字典" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# 两个刻意的"不报错"
# --------------------------------------------------------------------------- #


def test_contains_on_non_array_field_is_not_a_match() -> None:
    """``$contains`` 只对数组有意义；字段是标量（或缺失）一律判不命中.

    ``$not_contains`` 在非数组字段上也是不命中，而不是"不包含所以命中"：
    两者都要求"这个字段是数组"这个前提成立。
    """
    assert compile_filter({"strategy": {"$contains": "structural"}})(METADATA) is False
    assert compile_filter({"strategy": {"$not_contains": "structural"}})(METADATA) is False
    assert compile_filter({"absent": {"$contains": "x"}})(METADATA) is False


def test_nin_and_ne_treat_missing_key_as_hit() -> None:
    """``$nin`` / ``$ne`` 对**缺失键**算命中（"不在集合内"对不存在的值也成立）."""
    assert match_metadata(METADATA, {"absent": {"$nin": ["x"]}}) is True
    assert match_metadata(METADATA, {"absent": {"$ne": "x"}}) is True
    # 而 $eq / $in 要求字段真的存在
    assert match_metadata(METADATA, {"absent": {"$eq": "x"}}) is False
    assert match_metadata(METADATA, {"absent": {"$in": ["x"]}}) is False


def test_incomparable_types_are_not_a_match_and_do_not_raise() -> None:
    """``str`` 与 ``int`` 相比触发 ``TypeError``；判为不命中而不是抛错.

    取舍写在明面上：**查询的可用性优先于对单条脏数据的严格性**
    （一份库里 ``page`` 既可能是 int 又可能是 "12"）。
    """
    assert match_metadata(METADATA, {"index": {"$gt": "a"}}) is False
    assert match_metadata(METADATA, {"strategy": {"$gt": 1}}) is False
    assert match_metadata(METADATA, {"index": {"$lte": "a"}}) is False
    # 缺失字段的比较同样不命中（没有值就没有可比性）
    assert match_metadata(METADATA, {"absent": {"$gt": 0}}) is False


def test_bool_and_int_are_equal_in_python() -> None:
    """``True == 1`` 是 Python 的语义，因此不要用同一个键混存布尔与整数."""
    assert match_metadata({"flag": True}, {"flag": {"$eq": 1}}) is True
    assert match_metadata(METADATA, {"oversized": {"$eq": 0}}) is True


# --------------------------------------------------------------------------- #
# describe_filter 与 filter_fields
# --------------------------------------------------------------------------- #


def test_describe_filter_distinguishes_none_and_empty() -> None:
    """``None`` 是"没提过滤"，``{}`` 是"提了但条件是空的"——两者必须可区分."""
    assert describe_filter(None) == "（无过滤）"
    assert describe_filter({}) == "（空子句：等价于无过滤）"


def test_describe_filter_keeps_nested_structure() -> None:
    """嵌套结构要保真（自己拼字符串很容易丢掉括号，报告就与语义不符）."""
    where = {"$and": [{"b": 2}, {"$or": [{"a": 1}, {"c": 3}]}]}

    rendered = describe_filter(where)

    assert json.loads(rendered) == where
    # 三层子句 = 五个字典：{"$and"} / {"b"} / {"$or"} / {"a"} / {"c"}
    assert rendered.count("{") == 5
    assert rendered.count("{") == rendered.count("}")


def test_filter_fields_collects_nested_fields_sorted_and_deduped() -> None:
    """字段名要**递归收集、去重、字典序**；逻辑运算符本身不算字段."""
    nested = {
        "$and": [
            {"strategy": "structural"},
            {"$or": [{"index": 1}, {"token_count": 2}]},
        ]
    }
    deduped = {"$and": [{"strategy": "a"}, {"strategy": "b"}]}

    assert filter_fields(nested) == ["index", "strategy", "token_count"]
    assert filter_fields(deduped) == ["strategy"]
    assert "$and" not in filter_fields(nested)


def test_filter_fields_on_empty_clauses() -> None:
    assert filter_fields(None) == []
    assert filter_fields({}) == []
    assert filter_fields({"strategy": "structural"}) == ["strategy"]
