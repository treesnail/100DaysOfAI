"""检索过滤：把"元数据条件 + 时间范围"合成**一份可校验的 ``where``**（M6-D5）.

``vectorstore.filters`` 已经把 ``where`` 编译成谓词了，这一层为什么还要存在？
因为它回答的是那层**故意不回答**的三个问题：

```text
1. 时间范围怎么变成 where？         → 闭区间 → $gte + $lte（两条 $and 包起来）
2. where 与 time_range 撞车怎么办？ → 同一个字段两组条件 → 必须报错，不许猜
3. 字段名是不是写错了？             → 与库里的元数据字段全集对账
```

## 时间范围的三条约定（与 ``types.TimeRange`` 的 docstring 是同一份）

| 约定 | 具体含义 | 为什么 |
|------|---------|--------|
| 闭区间 | ``[start, end]``，两端可单独省略 | 单端区间（"这个月之后"）是最常见的问法 |
| 按 ISO 文本比较 | 比较在 ``$gte/$lte`` 里做 | 同一种 ISO 文本的字典序 = 时间序 |
| 字段缺失 → 排除 | ``$gte`` 对缺失字段返回 False | 宁可少召回，也不给时间编一个答案 |

## 为什么要在这里拦"字段冲突"

``{"created_at": {"$gt": "2026-01-01"}}`` 配上
``TimeRange(field="created_at", start="2026-06-01")`` 时，合成结果会是
``{"$and": [{"created_at": {...}}, {"$and": [{"created_at": {...}}]}]}``——
**这在语义上是可解释的（两个条件都满足）**，但它是调用方几乎不可能想要的东西：
"我要 1 月之后"与"我要 6 月之后"放在一起，真实意图通常是"我复制粘贴时忘了删一个"。

而它**查得出结果、也不报错**：结果只是比预期少。这类"少而不报"正是本课
要消灭的东西，因此这里选择当场报 ``QueryError``，并在消息里给出出路——
"请自己合并成一个显式区间"。**把两个可能都成立的意图交给调用方决定**，
比替它挑一个更安全。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from smart_research_agent.retrieval.errors import QueryError
from smart_research_agent.vectorstore.errors import FilterError
from smart_research_agent.vectorstore.filters import (
    SUPPORTED_OPERATORS,
    compile_filter,
    describe_filter,
    filter_fields,
)

if TYPE_CHECKING:  # pragma: no cover - 只为类型检查器与 IDE 提供跳转
    from smart_research_agent.retrieval.types import TimeRange

#: 默认的时间字段名。与 ``types.TimeRange.field`` 的默认值、
#: ``config.retrieval_time_field`` 的默认值三处必须一致——
#: 它们描述的是同一个约定："本项目的记录用这个键记创建时间"。
DEFAULT_TIME_FIELD = "created_at"

#: "看起来像时间字段"的名字后缀（只用于 ``time_field_of`` 的提示，不参与任何判定）。
TIME_FIELD_SUFFIXES: tuple[str, ...] = ("_at", "_date", "_time", "_ts")


# --------------------------------------------------------------------------- #
# 时间范围 → 子句
# --------------------------------------------------------------------------- #


def time_range_clause(time_range: TimeRange | None) -> dict[str, Any]:
    """把时间范围翻译成 ``where`` 子句（空区间 → 空字典）.

    三种形状，与 ``TimeRange`` 的三条约定一一对应：

    ```text
    两端都给   {"$and": [{field: {"$gte": start}}, {field: {"$lte": end}}]}
    只给下界   {field: {"$gte": start}}
    只给上界   {field: {"$lte": end}}
    两端都没给 {}（等价于"这次不涉及时间条件"）
    ```

    为什么双端要用 ``$and`` 包一层而不是写成一个字段的两个运算符：
    ``vectorstore.filters`` 明确拒绝"同一字段多个运算符"的条件
    （见其 ``_compile_field``），因此**必须**用 ``$and`` 表达"两个条件同时成立"。
    这不是绕开校验，而是那层留下的唯一合法写法。
    """
    if time_range is None or time_range.is_empty:
        return {}
    field = time_range.field
    start = time_range.start
    end = time_range.end
    if start is not None and end is not None:
        return {"$and": [{field: {"$gte": start}}, {field: {"$lte": end}}]}
    if start is not None:
        return {field: {"$gte": start}}
    return {field: {"$lte": end}}


def combine_where(
    where: dict[str, Any] | None,
    time_range: TimeRange | None,
) -> dict[str, Any] | None:
    """把 ``where`` 与时间范围合成一份子句（两者都空 → ``None``）.

    ```text
    两者都空       → None（"没有过滤"与"空字典过滤"在报告里必须能区分）
    只有时间范围   → 时间子句
    只有 where     → where 原样（不做任何包装：包装会给报告加一层噪声）
    都有           → {"$and": [where, 时间子句]}
    ```

    **冲突检测**：``where`` 里出现了 ``time_range.field`` 时抛 ``QueryError``。
    用 ``vectorstore.filter_fields`` 递归取字段名（它会正确跳过 ``$and`` / ``$or``
    这些逻辑键，因此嵌套子句里的 ``created_at`` 也能被发现）。

    冲突只在**时间子句非空**时判：``TimeRange`` 两端都没给时它没有任何条件，
    此时 ``where`` 里出现同名字段只是一次普通的元数据过滤。
    """
    clause = time_range_clause(time_range)
    if clause and where and time_range is not None:
        duplicated = sorted({name for name in filter_fields(where) if name == time_range.field})
        if duplicated:
            raise QueryError(
                f"过滤条件冲突：字段 {time_range.field!r} 同时出现在 where 与 time_range 里。"
                "同一个字段的两组条件请自己合并成一个显式区间——"
                "例如把 where 里的条件删掉、改用 TimeRange(start=..., end=...)；"
                "不合并的话，'我到底想要哪一段'会由两份条件取交集的副作用决定，"
                "而结果只会变少、不会报错。"
            )
    if not where and not clause:
        return None
    if not where:
        return clause
    if not clause:
        return where
    return {"$and": [where, clause]}


# --------------------------------------------------------------------------- #
# 校验与描述
# --------------------------------------------------------------------------- #


def validate_where(where: dict[str, Any] | None) -> None:
    """用 ``vectorstore.compile_filter`` 校验语法，把 ``FilterError`` 包成 ``QueryError``.

    **为什么不直接把 ``FilterError`` 抛上去**：这一层的错误族按"谁去修"划分
    （见 ``errors``），而写错过滤条件的是**调用方**——它该收到 ``QueryError``。
    同时消息里附上支持的运算符清单：``vectorstore`` 的原文里已经带了这份清单，
    但它的读者是"在调库的人"，而本层的读者是"在调检索器的人"，
    多一次转述能让报错在端点层不必再包一层。

    ``where`` 为 ``None`` / ``{}`` 都是合法的（编译成恒真谓词）。
    """
    if where is not None and not isinstance(where, dict):
        raise QueryError(
            f"where 必须是字典或 None，收到 {type(where).__name__}。"
            "写法：{'strategy': 'structural'} 或 {'token_count': {'$gt': 100}}"
        )
    try:
        compile_filter(where)
    except FilterError as exc:
        raise QueryError(
            f"where 子句不合法：{exc}\n"
            f"支持的运算符：{', '.join(SUPPORTED_OPERATORS)}"
        ) from exc


def describe_conditions(
    where: dict[str, Any] | None,
    time_range: TimeRange | None = None,
) -> str:
    """一行人类可读描述（无过滤时返回 ``"（无过滤）"``）.

    两段用 ``" 且 "`` 连接，而不是逗号：合成后的语义就是"两个条件同时成立"
    （见 ``combine_where``），报告里的连接词与执行语义必须一致——
    写成逗号会让人以为它们是两个独立的可选项。

    ``where`` 部分复用 ``vectorstore.describe_filter``（``json.dumps``，
    嵌套结构保持原样）；时间部分用 ``TimeRange.describe()``。
    """
    parts: list[str] = []
    if where:
        parts.append(describe_filter(where))
    if time_range is not None and not time_range.is_empty:
        parts.append(time_range.describe())
    if not parts:
        return "（无过滤）"
    return " 且 ".join(parts)


def unknown_filter_fields(
    where: dict[str, Any] | None,
    field_names: Iterable[str],
) -> list[str]:
    """过滤条件里**在库的元数据中从未出现过**的字段名（拼写错误的探测器）.

    它能发现什么、不能发现什么必须写清楚，否则会被当成一个万能的检查器：

    ```text
    能发现     {"stratgey": "structural"}  —— 整个库里都没有 stratgey 这个键
    不能发现   {"strategy": "structual"}   —— 键对、值错（值域是业务知识，不是元数据事实）
    不能发现   {"created_at": {"$gte": "2026-99-01"}} —— 键对、格式错（那是校验器的事）
    ```

    因此它只用在 ``Retriever.explain()`` 的"字段拼写检查"一行里，
    作为**提示**而不是判定：一个字段在库里没出现过，完全可能是因为
    "这一批数据恰好都不带它"，所以它绝不参与 ``empty_reason`` 的判定。
    """
    known = {str(name) for name in field_names}
    return [name for name in filter_fields(where) if name not in known]


def time_field_of(where: dict[str, Any] | None, default: str = DEFAULT_TIME_FIELD) -> str:
    """从 ``where`` 里猜出"这次做时间筛选的是哪个字段"（只用于报告与提示）.

    判定顺序：

    ```text
    1. where 里直接出现 default 这个名字 → 就是它
    2. 出现名字以 _at / _date / _time / _ts 结尾的字段 → 取字典序第一个（确定性）
    3. 都没有 → 返回 default
    ```

    它**不参与任何判定**（冲突检测用的是 ``time_range.field`` 精确比较）：
    猜时间字段名必须靠命名约定，而命名约定永远会有例外，
    因此把它的输出只当"报告里的一句话"用，是唯一安全的用法。
    """
    names = filter_fields(where)
    if default in names:
        return default
    for name in names:
        if name.endswith(TIME_FIELD_SUFFIXES):
            return name
    return default


__all__ = [
    "DEFAULT_TIME_FIELD",
    "TIME_FIELD_SUFFIXES",
    "combine_where",
    "describe_conditions",
    "time_field_of",
    "time_range_clause",
    "unknown_filter_fields",
    "validate_where",
]
