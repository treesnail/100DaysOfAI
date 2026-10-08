"""``query``：在倒排索引上做一次**确定性**检索（day101）.

检索只有一条公式：

```text
score = 该文档命中的查询词数 / 查询词总数        ⇒  数学上落在 [0, 1]
```

排序规则也只有两条，而且第二条是"为了确定性"才存在的：

```text
① 分数高的在前
② 分数并列时，**文档名**字典序在前     ⇒ 同一个问题两次得到同一份名单
```

## 一、今天最值钱的一句话

> **"查了什么"与"查出什么"必须都能被复算：查询词、命中词、分数、名次，
> 四样都要原样落在结果里——只报一个排名，读的人无法核对它。**

因此 :class:`Hit` 里同时有 `score` 与 `matched`（命中的那几个词），
而 :class:`SearchResult` 同时有 `query_tokens`（切出来的词）与 `hits`（名单）。
一份结果里"分数是 0.5"与"命中 1/2 个词"必须能互相印证。

## 二、``QueryError`` 的两个场景

```text
空查询        ""          ⇒  没有可检索的词
无词查询      "、、、"     ⇒  分词之后一个 token 都没有
```

两者都抛 :class:`QueryError`，而不是安静地返回空名单：
**"没查"与"查了没命中"是两件事**——后者是语料的问题，前者是调用点的问题。

## 三、与既有包的接缝

- **上游**：:mod:`course_index.index`（倒排索引）、:mod:`course_index.types`（`TOP_K_DEFAULT`）；
- **下游**：:mod:`course_index.verify` 用它检查"检索是确定的"与"分数有界"，
  :mod:`course_index.study` 打印检索表。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smart_research_agent.course_index.errors import NumericError, ParameterError, QueryError, ScoreError
from smart_research_agent.course_index.index import InvertedIndex, build_index, tokenize
from smart_research_agent.course_index.types import TOP_K_DEFAULT, require_positive_int

#: 复算口径里保留的小数位（浮点在两次检索间逐位相同，这里只是去掉尾零噪声）.
SCORE_DIGITS = 12


@dataclass(frozen=True)
class Hit:
    """一条命中：文档名 + 分数 + 命中的查询词.

    ``matched`` 是"这条分数是怎么来的"的唯一证据：分数是 ``len(matched) / 查询词数``，
    少了它，一个 0.5 分就只能被相信、不能被核对。
    """

    document: str
    score: float
    matched: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.document:
            raise ParameterError("命中的文档名不能为空。")
        if self.score != self.score or self.score in (float("inf"), float("-inf")):
            raise NumericError(f"命中分数必须有限，收到 {self.score!r}。")
        if not 0.0 <= self.score <= 1.0:
            raise ScoreError(
                f"命中分数 {self.score!r} 落在 [0, 1] 之外："
                "分数是'命中的查询词数 / 查询词总数'，越界说明实现里多减了一次或归一化错了。"
            )
        if not self.matched:
            raise ScoreError(f"命中 {self.document!r} 的分数非零，却一个命中词都没有。")

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {"document": self.document, "score": round(self.score, 6), "matched": list(self.matched)}

    def line(self) -> str:
        """一行读数：``subpackage:course_index   | 分数 1.0000 | 命中 1/1 [course_index]``."""
        return (
            f"{self.document:<34} | 分数 {self.score:.4f} | 命中 {len(self.matched)} 词"
            f" [{'、'.join(self.matched)}]"
        )


@dataclass(frozen=True)
class SearchResult:
    """一次检索的结果：查询 + 切出来的词 + 名单（按分数、再按名字排序）."""

    query: str
    query_tokens: tuple[str, ...]
    hits: tuple[Hit, ...]

    def __post_init__(self) -> None:
        if not self.query or not self.query.strip():
            raise QueryError("查询不能为空。")
        if not self.query_tokens:
            raise QueryError(
                f"查询 {self.query!r} 切出来一个 token 都没有——"
                "'无词查询'与'查了没命中'是两件事，前者该在调用点被看见。"
            )
        order = [(-round(hit.score, SCORE_DIGITS), hit.document) for hit in self.hits]
        if order != sorted(order):
            raise ParameterError(
                "命中的排序不是'分数降序、名字升序'："
                "并列的排序不稳定会让同一个问题两次得到两份名单。"
            )

    # ------------------------------------------------------------------ 只读视图

    @property
    def documents(self) -> tuple[str, ...]:
        """命中的文档名（顺序即名次）."""
        return tuple(hit.document for hit in self.hits)

    @property
    def top_score(self) -> float:
        """最高的一条分数（没有命中时为 0）."""
        return self.hits[0].score if self.hits else 0.0

    def rank_of(self, document: str) -> int:
        """某份文档的名次（1 起；没命中时为 0）."""
        for index, hit in enumerate(self.hits, start=1):
            if hit.document == document:
                return index
        return 0

    # ------------------------------------------------------------------ 复算口径

    def comparable(self) -> tuple[Any, ...]:
        """**只含确定性字段**的元组（两次检索比较它）."""
        return (
            ("query", self.query),
            ("tokens", self.query_tokens),
            ("hits", tuple((hit.document, round(hit.score, SCORE_DIGITS)) for hit in self.hits)),
        )

    def diff_count(self, other: SearchResult) -> int:
        """与另一次检索在复算口径上**差了几项**（0 = 逐位相同）."""
        left = self.comparable()
        right = other.comparable()
        return sum(1 for a, b in zip(left, right, strict=True) if a != b)

    def is_identical_to(self, other: SearchResult) -> bool:
        """是否与另一次检索逐位相同."""
        return self.comparable() == other.comparable()

    def to_dict(self) -> dict[str, Any]:
        """摊平成可 JSON 化的字段."""
        return {
            "query": self.query,
            "tokens": list(self.query_tokens),
            "hits": [hit.to_dict() for hit in self.hits],
        }

    def line(self) -> str:
        """一行读数：``查询 "course_index" | 词 1 个 | 命中 3 条 | 首条 subpackage:course_index 1.0000``."""
        if not self.hits:
            return f"查询 {self.query!r} | 词 {len(self.query_tokens)} 个 | 命中 0 条"
        top = self.hits[0]
        return (
            f"查询 {self.query!r} | 词 {len(self.query_tokens)} 个 | 命中 {len(self.hits)} 条"
            f" | 首条 {top.document} {top.score:.4f}"
        )


def unique_tokens(query: str) -> tuple[str, ...]:
    """把查询切成 token 并**去重**（保留首次出现的顺序）.

    去重是为了让分母是"查询里有几个**不同的**词"：
    否则 ``"向量 向量 向量"`` 会被算成三个词，分数因此虚低。
    """
    seen: list[str] = []
    for token in tokenize(query):
        if token not in seen:
            seen.append(token)
    return tuple(seen)


def search(
    query: str,
    *,
    index: InvertedIndex | None = None,
    top_k: int = TOP_K_DEFAULT,
) -> SearchResult:
    """在倒排索引上检索，返回一份 :class:`SearchResult`（**确定性**）.

    ``index`` 缺省时现场建一份；``top_k`` 非法（< 1）当场拒绝。
    """
    require_positive_int("top_k", top_k)
    resolved = build_index() if index is None else index
    tokens = unique_tokens(query)
    if not tokens:
        raise QueryError(
            f"查询 {query!r} 切出来一个 token 都没有（空查询或全是不可索引的字符）。"
        )
    hits: list[Hit] = []
    for document in resolved.documents:
        matched = tuple(
            token for token in tokens if document in resolved.documents_containing(token)
        )
        if not matched:
            continue
        hits.append(
            Hit(document=document, score=len(matched) / len(tokens), matched=matched)
        )
    ordered = tuple(sorted(hits, key=lambda hit: (-round(hit.score, SCORE_DIGITS), hit.document)))
    return SearchResult(query=query, query_tokens=tokens, hits=ordered[:top_k])


def require_hits(result: SearchResult | None = None, *, query: str | None = None) -> SearchResult:
    """没有命中时抛 :class:`QueryError`（"拒绝交付"的那条路）.

    它与 :func:`search` 的分工是"给名单 vs 拒绝空名单"：
    空的检索结果在报告里只是一行"命中 0 条"，而金标准场景下它是一次真失败。
    """
    resolved = search(query or "") if result is None else result
    if resolved.hits:
        return resolved
    raise QueryError(
        f"查询 {resolved.query!r} 一条命中都没有——"
        "要么语料漏了该有的那一份，要么索引错了，要么这个词本来就不在语料里。"
    )


def search_matches(expected: SearchResult, actual: SearchResult) -> int:
    """两次检索在复算口径上差了几项（供性质检查用）."""
    return expected.diff_count(actual)


def search_lines(result: SearchResult | None = None, *, query: str | None = None) -> tuple[str, ...]:
    """把一次检索逐行印出来."""
    resolved = search(query, index=None) if result is None else result
    lines = ["检索表（查询 → 名单）：", "  " + resolved.line()]
    lines.extend("  " + hit.line() for hit in resolved.hits)
    return tuple(lines)


__all__ = [
    "SCORE_DIGITS",
    "Hit",
    "SearchResult",
    "require_hits",
    "search",
    "search_lines",
    "search_matches",
    "unique_tokens",
]
