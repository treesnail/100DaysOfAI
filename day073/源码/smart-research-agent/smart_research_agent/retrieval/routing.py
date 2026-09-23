"""多索引路由：一句查询该去哪个库，是一层**显式的业务决策**（M6-D5）.

一个真实的知识库从来不是一个库：

```text
手册库      产品手册，char-ngram 编码，索引版本 a1b2…
源码库      代码片段，另一套编码器，索引版本 0f2c…
历史归档    旧版本手册，清单已冻结（只读），索引版本 77de…
```

"用户问的这句话该去哪几个库"**不是技术细节，是业务决策**：
问"怎么配置阈值"要去手册库，问"这个函数签名"要去源码库，
而两台库都可能命中同一句话（那时该以谁为准）——这些判断只可能由
懂业务的人给出，代码能做的是**把这个判断写成一层可读、可测、可回显的结构**，
而不是散在各个调用点上的 ``if doc_type == "handbook": ...``。

散在调用点上的版本有三个必然结局：一处漏改、两处不一致、没人知道一共有几条路。

## 三条规则（都在 ``resolve`` 里）

```text
route 为空 + 设了 default      → 用 default（并说明"用的是默认路由"）
route 为空 + 没设 default      → 只有一个索引就用它；多个则报错要求显式指定
route 非空                     → 必须存在；不存在时报错并**列出全部可用名**
```

第二条是这一层最重要的取舍：**只有一个索引时不该强迫调用方写路由名**
（那是纯粹的样板），但**有多个时必须报错**——"随便挑一个"会让
"问源码库的问题去查了手册库"变成一个静默的错误答案，而它看起来
与正确答案一模一样（都是一段通顺的引用文本）。宁可要求显式指定。

报错格式沿用 ``vectorstore.registry`` 的三段式（现象 → 可用清单 → 出路），
因为读报的人在同一次排查里要同时看这两层：

```text
未知路由 'handbok'：本路由器没有这个名字的索引。
可用路由：handbook、source、archive
或者改用：不传 route（当前默认路由是 'handbook'）
```
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from smart_research_agent.retrieval.errors import IndexStateError, QueryError
from smart_research_agent.retrieval.retriever import Retriever
from smart_research_agent.retrieval.types import RetrievalQuery, RetrievalResult


@dataclass(frozen=True)
class RouteDecision:
    """一次选路的结果：**选了谁 + 为什么选它**（M6-D5）.

    ``reason`` 是本结构存在的理由。路由如果只返回一个 ``Retriever``，
    调用方就无法回答"这次到底是我指定的，还是它替我挑的"——
    而这个差别在排查时是决定性的：

    ```text
    matched=True    调用方显式写的路由名到了 → 结果不对就是那个库的问题
    matched=False   路由器替它选了（默认 / 只有一个） → 先确认"该不该是它"
    ```

    因此 ``retrieve`` 把这两个字段一路带到报告里，端点回显的也是它们。
    """

    name: str
    reason: str
    matched: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise QueryError(
                f"RouteDecision.name 必须是非空字符串，收到 {self.name!r}："
                "选路的结论不能是'选了某个没名字的东西'。"
            )
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise QueryError(
                f"RouteDecision.reason 必须是非空字符串，收到 {self.reason!r}："
                "没有理由的选路等于不可排查。"
            )

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return {"name": self.name, "reason": self.reason, "matched": self.matched}

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        mark = "显式" if self.matched else "路由器选定"
        return f"路由 {self.name}（{mark}）：{self.reason}"


class StoreRouter:
    """多个检索器的命名路由表（注册 → 解析 → 检索 → 报告）.

    ``default`` 在构造期只记名字、**不校验存在性**：路由表的典型用法是
    "先建路由器、再逐个注册"，如果构造期就要求 default 已存在，
    调用方只能把注册顺序写死（而那个顺序在动态装配里根本不固定）。
    校验落在 ``resolve`` 里——那里既知道调用的意图，也知道当下有什么。
    """

    def __init__(self, *, default: str | None = None) -> None:
        self._retrievers: dict[str, Retriever] = {}
        self._descriptions: dict[str, str] = {}
        self._default = _clean_optional_name("default", default)

    # ------------------------------------------------------------------ 注册

    def register(self, name: str, retriever: Retriever, *, description: str = "") -> None:
        """注册一个索引（重名 → ``QueryError``）.

        **重名直接报错而不是覆盖**：路由名是"这句话该去哪"的唯一依据，
        而覆盖会让一次装配顺序的偶然变化变成"某个路由悄悄指向了另一个库"
        ——那时所有通过名字的调用都还在工作，只是答案换了一个语料。
        这与 ``vectorstore.registry`` 对未知后端"宁可报错、不要挑一个差不多的"
        是同一条纪律。
        """
        key = _clean_name("name", name)
        if key in self._retrievers:
            raise QueryError(
                f"路由名 {key!r} 已经被注册过：注册是**幂等的拒绝**而不是覆盖。"
                "重名的后果是'某个路由悄悄指向了另一个库'，而按名字调用的一方"
                "不会有任何察觉。请换一个名字，或先确认装配顺序。"
            )
        if not isinstance(retriever, Retriever):
            raise QueryError(
                f"路由 {key!r} 的值必须是 Retriever，收到 {type(retriever).__name__}。"
                "路由表管的是'去哪查'，查询逻辑本身在检索器里。"
            )
        self._retrievers[key] = retriever
        self._descriptions[key] = str(description or "").strip()

    # ------------------------------------------------------------------ 视图

    def names(self) -> list[str]:
        """全部路由名（**升序**）.

        升序而不是注册序：报告要能被逐行 diff，而注册序会随装配顺序变化
        （与 ``FlatVectorStore.ids()`` 同一条理由）。
        """
        return sorted(self._retrievers)

    @property
    def default(self) -> str:
        """默认路由名（未设置时为空串）."""
        return self._default or ""

    def __len__(self) -> int:
        """已注册的索引个数（``len(router)`` 比 ``len(router.names())`` 直白）."""
        return len(self._retrievers)

    def __contains__(self, name: object) -> bool:
        """``"handbook" in router``（按名字判断是否注册过）."""
        return isinstance(name, str) and name in self._retrievers

    # ------------------------------------------------------------------ 选路

    def resolve(self, route: str | None) -> tuple[Retriever, RouteDecision]:
        """按名字取检索器，并给出"为什么是它"（三条规则见模块 docstring）.

        空路由器 → ``IndexStateError``：那不是调用方写错了名字，
        而是**索引还没建/没注册**，属于库侧的事（见 ``errors`` 的分族依据）。
        """
        if not self._retrievers:
            raise IndexStateError(
                "路由器里没有注册任何索引：请先用 register(name, retriever) 注册。"
                "空路由器不是'调用方写错了名字'，而是'索引还没装配好'——"
                "因此它属于库侧的问题，而不是参数问题。"
            )
        requested = _clean_optional_name("route", route)
        if requested is None:
            return self._resolve_default()
        retriever = self._retrievers.get(requested)
        if retriever is None:
            raise QueryError(self._unknown_route_message(requested))
        return retriever, RouteDecision(
            name=requested,
            reason="调用方显式指定的路由",
            matched=True,
        )

    def _resolve_default(self) -> tuple[Retriever, RouteDecision]:
        """没指定 route 时的两条出路（默认路由 / 唯一的那个索引）."""
        if self._default is not None:
            retriever = self._retrievers.get(self._default)
            if retriever is None:
                raise QueryError(
                    f"默认路由 {self._default!r} 不在已注册的路由里。\n"
                    f"可用路由：{'、'.join(self.names())}\n"
                    f"或者改用：显式传 route=（例如 route={self.names()[0]!r}），"
                    "或把 default 改成一个已注册的名字。"
                )
            return retriever, RouteDecision(
                name=self._default,
                reason="调用方没有指定 route，按路由器的 default 选择",
                matched=False,
            )
        if len(self._retrievers) == 1:
            only = self.names()[0]
            return self._retrievers[only], RouteDecision(
                name=only,
                reason="调用方没有指定 route，且路由器里只有一个索引，直接用它",
                matched=False,
            )
        raise QueryError(
            f"有 {len(self._retrievers)} 个索引但没有设置默认路由，必须显式指定 route。\n"
            f"可用路由：{'、'.join(self.names())}\n"
            "或者改用：在 StoreRouter(default=...) 里指定一个默认路由。"
            "替调用方随便挑一个的后果是'问源码库的问题去查了手册库'——"
            "而它给出的答案看起来与正确答案一模一样。"
        )

    def _unknown_route_message(self, route: str) -> str:
        """三段式：现象 → 可用清单 → 出路（见模块 docstring）."""
        if self._default is not None:
            fallback = f"不传 route（当前默认路由是 {self._default!r}）"
        else:
            fallback = (
                f"不传 route（当前有 {len(self._retrievers)} 个索引且没有默认路由，"
                "那时必须显式指定）"
            )
        return (
            f"未知路由 {route!r}：本路由器没有这个名字的索引。\n"
            f"可用路由：{'、'.join(self.names())}\n"
            f"或者改用：{fallback}"
        )

    # ------------------------------------------------------------------ 检索与报告

    def retrieve(
        self,
        query: str | RetrievalQuery,
        *,
        route: str | None = None,
    ) -> RetrievalResult:
        """选路 + 检索（**优先用 ``query.route``**，再考虑 ``route`` 参数）.

        为什么查询里那个 ``route`` 优先：一份 ``RetrievalQuery`` 是可以被
        序列化、存盘、再回放的（端点就是这么收的），那时"该去哪"是**请求的一部分**；
        而 ``route`` 参数更像是"调用方这次想覆盖一下"。**显式传入的请求内容
        优先于调用点的默认**——反过来会让"回放同一份请求去了不同的库"。
        """
        requested = route
        if isinstance(query, RetrievalQuery) and query.route:
            requested = query.route
        retriever, decision = self.resolve(requested)
        result = retriever.retrieve(query)
        return _with_route_note(result, decision)

    def report(self) -> dict[str, Any]:
        """路由清单（``/retrieval/routes`` 直接返回它）.

        每个路由带三样东西：**名字**（怎么问）、**说明**（为什么有它）、
        **检索器的现状**（``describe()``：深度/阈值/索引版本/漂移）。
        三者缺一都会让"这一路为什么没结果"变得无法回答——
        例如漂移在 ``describe().index`` 里，而它恰恰是最常见的原因。
        """
        return {
            "count": len(self._retrievers),
            "default": self._default or "",
            "names": self.names(),
            "routes": [
                {
                    "name": name,
                    "description": self._descriptions.get(name, ""),
                    "retriever": self._retrievers[name].describe(),
                }
                for name in self.names()
            ],
        }


def _with_route_note(result: RetrievalResult, decision: RouteDecision) -> RetrievalResult:
    """把选路结论追加进 ``notes``（结果自己交代"这份结果来自哪个路由"）.

    用 ``replace`` 而不是改字段：``RetrievalResult`` 是 frozen 的，
    而"结果可以被就地修改"会让一次检索的交代在传递过程中被悄悄改写。
    """
    return replace(result, notes=(*result.notes, decision.summary_line()))


def _clean_name(label: str, value: str) -> str:
    """路由名/默认名必须是非空字符串（只是收敛，不做命名风格限制）."""
    if not isinstance(value, str) or not value.strip():
        raise QueryError(f"{label} 必须是非空字符串，收到 {value!r}")
    return value.strip()


def _clean_optional_name(label: str, value: str | None) -> str | None:
    """可省略的名字：``None`` 与空串都归一成 ``None``（"没指定"只有一种表示）."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise QueryError(f"{label} 必须是字符串或 None，收到 {type(value).__name__}")
    stripped = value.strip()
    return stripped or None


__all__ = [
    "RouteDecision",
    "StoreRouter",
]
