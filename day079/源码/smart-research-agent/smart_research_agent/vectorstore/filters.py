"""``where`` 过滤子句：把"元数据筛选"写成一份**可移植的声明**（M6-D3）.

向量库的检索从来不是"只要最像的 K 条"，而是"**在满足条件的那些里**
最像的 K 条"。条件的表达方式每个库都不一样：

```text
Chroma   collection.query(..., where={"strategy": "structural"})
FAISS    没有过滤能力——只能自己先筛出 id，再用 IDSelector 把它交给索引
本包     where 子句由这一层编译成谓词，三个后端共用同一份语义
```

因此这里的第一个决定是：**采用 Chroma 的语法作为本包的规范语法**。
理由不是它更好，而是它**已经是一份写下来的、被真实实现的规格**
（运算符集合、嵌套规则、类型限制都有明文），而自己发明一套 DSL
只会多出一个"我们的语法与库的语法不一致"的翻译层。

## 支持的运算符与它们的语义

| 运算符 | 含义 | 值的要求 |
|--------|------|---------|
| ``$eq`` | 等于 | 标量（**直接写值等价于 ``$eq``**） |
| ``$ne`` | 不等于 | 标量 |
| ``$gt`` / ``$gte`` | 大于 / 大于等于 | 标量 |
| ``$lt`` / ``$lte`` | 小于 / 小于等于 | 标量 |
| ``$in`` / ``$nin`` | 在 / 不在集合内 | **非空**列表 |
| ``$contains`` / ``$not_contains`` | 数组元数据是否包含某元素 | 标量（字段必须是数组） |
| ``$and`` / ``$or`` | 逻辑组合 | **非空**的子句列表 |

``$nin`` 有一条容易意外但不该改的语义：**字段根本不存在的记录也算命中**。
理由与它的定义一致——"不在集合内"对一个不存在的值也成立；
而如果想要"存在且不在内"，写成 ``$and`` 组合。

## 两个刻意留下的"不报错"

1. **类型不可比的比较判为不命中，而不是抛错。** 一份库里
   ``page`` 既可能是 ``int`` 又可能是 ``"12"``（不同批次入库的结果），
   此时 ``{"page": {"$gt": 10}}`` 遇到字符串会触发 ``TypeError``。
   抛错会让**一次查询因为一条脏记录而整体失败**；判为不命中则把它
   降级成一个"这一条没被选中"。取舍写在明面上：**查询的可用性
   优先于对单条脏数据的严格性**，而脏数据的清理是入库侧的事
   （``types.assert_metadata`` 已经在拦类型越界）。
2. **``True == 1``。** 这是 Python 的语义，不是本模块的选择：
   ``{"flag": {"$eq": 1}}`` 会命中 ``flag=True`` 的记录。
   因此**不要用同一个键混存布尔值与整数**——这一条没有代码层面的解法，
   只能靠约定，所以写在这里。

## 为什么要 ``describe_filter``

端点与报告需要把"这次是在什么条件下搜的"写下来。只回显原始字典是不够的：
``{}`` 与 ``None`` 都表示"没有过滤"，而 ``{"a": {"$in": []}}`` 会让
**整次查询必然为空**——后者是一个典型的"过滤器把自己筛没了"的错误，
必须在报告里一眼看出（``SearchResult.filter_applied`` 与这里的
``describe_filter`` 是配套的两个字段）。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from smart_research_agent.vectorstore.errors import FilterError

#: 支持的全部运算符。**顺序就是错误信息里的顺序**，也是手册里的顺序。
SUPPORTED_OPERATORS: tuple[str, ...] = (
    "$eq",
    "$ne",
    "$gt",
    "$gte",
    "$lt",
    "$lte",
    "$in",
    "$nin",
    "$contains",
    "$not_contains",
    "$and",
    "$or",
)

#: 逻辑组合运算符（值必须是子句列表）。
LOGICAL_OPERATORS: tuple[str, ...] = ("$and", "$or")

#: 需要"非空列表"作为值的运算符。
LIST_VALUE_OPERATORS: tuple[str, ...] = ("$in", "$nin")

#: 只对**数组型**元数据有意义的运算符。
ARRAY_FIELD_OPERATORS: tuple[str, ...] = ("$contains", "$not_contains")

#: 元数据谓词：给一个 metadata 字典，回答"这条记录是否满足条件"。
MetadataPredicate = Callable[[dict[str, Any]], bool]


def compile_filter(where: dict[str, Any] | None) -> MetadataPredicate:
    """把 ``where`` 子句编译成谓词.

    ``None`` 与 ``{}`` 都编译成"恒真"，但两者的**报告含义不同**：
    前者是"调用方没提过滤"，后者是"调用方提了但条件是空的"。
    真正要紧的是第三种——**条件非空却把所有记录筛掉**
    （``{"$in": []}`` 那种）——它会在 ``SearchResult.candidates == 0``
    且 ``filter_applied is True`` 时体现出来。
    """
    if where is None or where == {}:
        return _always_true
    if not isinstance(where, dict):
        raise FilterError(
            f"where 必须是字典，收到 {type(where).__name__}。"
            "写法：{'字段名': 值} 或 {'字段名': {'$gt': 10}} 或 {'$and': [..., ...]}"
        )
    return _compile_clause(where)


def match_metadata(metadata: dict[str, Any], where: dict[str, Any] | None) -> bool:
    """便捷入口：直接判断一份元数据是否满足子句（不做编译缓存）."""
    return compile_filter(where)(metadata)


def describe_filter(where: dict[str, Any] | None) -> str:
    """把子句渲染成一行人类可读的说明（报告与端点回显用）.

    用 ``json.dumps`` 而不是自己拼字符串，是为了让**嵌套结构保持原样**：
    ``$and`` 里套 ``$or`` 时，自己拼出来的字符串很容易丢掉括号，
    于是一份报告读起来与它实际执行的语义不一致。
    """
    if where is None:
        return "（无过滤）"
    if where == {}:
        return "（空子句：等价于无过滤）"
    try:
        rendered = json.dumps(where, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        rendered = repr(where)
    return rendered


def filter_fields(where: dict[str, Any] | None) -> list[str]:
    """子句里出现过的**元数据字段名**（去重、字典序；逻辑运算符不算）.

    用途是"解释一次空结果"：如果 ``filter_fields`` 里的某个字段
    在整个库里从来没出现过，那么空结果的原因就是字段名写错了
    ——这一条在 day066 的检索调试里会被反复用到。
    """
    if not where:
        return []
    names: set[str] = set()
    _collect_fields(where, names)
    return sorted(names)


def _collect_fields(clause: Any, names: set[str]) -> None:
    if not isinstance(clause, dict):
        return
    for key, value in clause.items():
        if key in LOGICAL_OPERATORS:
            if isinstance(value, list):
                for item in value:
                    _collect_fields(item, names)
            continue
        names.add(key)


def _always_true(_metadata: dict[str, Any]) -> bool:
    return True


def _compile_clause(clause: dict[str, Any]) -> MetadataPredicate:
    """编译一层子句：既可能是一组"字段条件"，也可能是一个逻辑组合."""
    if not clause:
        return _always_true

    if len(clause) > 1 and any(key in LOGICAL_OPERATORS for key in clause):
        # 混写会让语义变得不确定（"A 且 ($or ...)" 到底是与还是或），
        # 因此在编译期就拒掉，而不是给它挑一个"看起来合理"的解释。
        raise FilterError(
            "$and / $or 必须单独占一层，不能与其他字段条件混写在同一个字典里；"
            f"收到 {sorted(clause)}。请写成 {{'$and': [{{'a': 1}}, {{'b': 2}}]}}。"
        )

    key, value = next(iter(clause.items()))
    if key in LOGICAL_OPERATORS:
        return _compile_logical(key, value)
    if key.startswith("$"):
        raise FilterError(
            f"未知运算符 {key!r}，支持的运算符：{', '.join(SUPPORTED_OPERATORS)}。"
            "（字段名不可以以 $ 开头。）"
        )
    return _compile_field(key, value)


def _compile_logical(operator: str, value: Any) -> MetadataPredicate:
    if not isinstance(value, list) or not value:
        raise FilterError(
            f"{operator} 的值必须是非空列表，收到 {value!r}。"
            f"写法：{{'{operator}': [{{'a': 1}}, {{'b': 2}}]}}"
        )
    sub_predicates = []
    for item in value:
        if not isinstance(item, dict):
            raise FilterError(f"{operator} 的每个元素都必须是字典，收到 {item!r}")
        sub_predicates.append(_compile_clause(item))
    if operator == "$and":
        return lambda metadata: all(predicate(metadata) for predicate in sub_predicates)
    return lambda metadata: any(predicate(metadata) for predicate in sub_predicates)


def _compile_field(field: str, condition: Any) -> MetadataPredicate:
    if not isinstance(condition, dict):
        # 直接写标量等价于 $eq（与 Chroma 的语法一致）
        return _equality_predicate(field, condition)

    if not condition:
        raise FilterError(
            f"字段 {field!r} 的条件是空字典 —— 这通常意味着"
            f"{{'{field}': {{}}}} 是写漏了运算符。若想表达'字段存在'，"
            "请用 {'$and': [{field: {'$ne': …}}]} 这类显式条件。"
        )
    if len(condition) > 1:
        raise FilterError(
            f"字段 {field!r} 的条件里有多个运算符 {sorted(condition)}："
            "同一个字段的多个条件请用 $and 组合，"
            f"例如 {{'$and': [{{'{field}': {{'$gt': 1}}}}, "
            f"{{'{field}': {{'$lt': 9}}}}]}}。"
        )

    operator, expected = next(iter(condition.items()))
    if operator not in SUPPORTED_OPERATORS:
        raise FilterError(
            f"未知运算符 {operator!r}，支持的运算符：{', '.join(SUPPORTED_OPERATORS)}"
        )
    if operator in LOGICAL_OPERATORS:
        raise FilterError(
            f"{operator} 不能写在字段内部（它组合的是**条件**，不是字段值）："
            f"正确写法是把整个字典替换成 {{'{operator}': [...]}}。"
        )
    return _operator_predicate(field, operator, expected)


def _operator_predicate(field: str, operator: str, expected: Any) -> MetadataPredicate:
    if operator in LIST_VALUE_OPERATORS:
        if not isinstance(expected, (list, tuple)) or not expected:
            raise FilterError(
                f"{operator} 的值必须是非空列表，收到 {expected!r}"
                f"（空列表会让条件的命中集恒为空，因此直接拒绝而不是静默返回空结果）"
            )
        members = list(expected)
        if operator == "$in":
            return lambda metadata: metadata.get(field) in members
        return lambda metadata: metadata.get(field) not in members

    if operator in ARRAY_FIELD_OPERATORS:
        def array_predicate(metadata: dict[str, Any]) -> bool:
            value = metadata.get(field)
            if not isinstance(value, (list, tuple)):
                # 字段不是数组时 $contains 无意义：判为不命中（见模块 docstring 取舍一）
                return False
            contained = expected in value
            return contained if operator == "$contains" else not contained

        return array_predicate

    if operator == "$eq":
        return _equality_predicate(field, expected)
    if operator == "$ne":
        return _inequality_predicate(field, expected)

    def comparison(metadata: dict[str, Any]) -> bool:
        value = metadata.get(field)
        if value is None:
            return False
        try:
            if operator == "$gt":
                return bool(value > expected)
            if operator == "$gte":
                return bool(value >= expected)
            if operator == "$lt":
                return bool(value < expected)
            return bool(value <= expected)
        except TypeError:
            # 类型不可比 → 判为不命中（见模块 docstring 取舍一）
            return False

    return comparison


def _equality_predicate(field: str, expected: Any) -> MetadataPredicate:
    return lambda metadata: field in metadata and metadata[field] == expected


def _inequality_predicate(field: str, expected: Any) -> MetadataPredicate:
    # 字段不存在也算"不等于"——与 $nin 的语义保持一致（见模块 docstring）
    return lambda metadata: metadata.get(field) != expected


__all__ = [
    "ARRAY_FIELD_OPERATORS",
    "LIST_VALUE_OPERATORS",
    "LOGICAL_OPERATORS",
    "MetadataPredicate",
    "SUPPORTED_OPERATORS",
    "compile_filter",
    "describe_filter",
    "filter_fields",
    "match_metadata",
]
