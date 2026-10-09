"""``study``：四张表（day101）.

**一个数只有被印出来才可能被反驳**。今天四张表各回答一个"这份索引成不成立"的问题：

```text
① 语料表     全部材料：名字（带种类）/ 种类 / 字符数
② 索引表     词典里最常见的若干 token：出现它的文档数 / 前几个文档与次数
③ 检索表     一次检索的名次 / 文档 / 分数 / 命中的词
④ 性质表     7 条性质：判据类别 / 读数 / 结论 / 一行证据
```

## 一条纪律：每一行都要带**两个数**

与前面各天的表同源——一行只写"通过"的表是没法反驳的。
因此每一行的"读数"旁边都跟着一个"参照"：字符数 / 文档数 / 分数与命中词 / 判据与期望。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smart_research_agent.course_index import corpus as corpus_module
from smart_research_agent.course_index import query as query_module
from smart_research_agent.course_index import verify
from smart_research_agent.course_index.corpus import Corpus, build_corpus
from smart_research_agent.course_index.index import InvertedIndex, build_index
from smart_research_agent.course_index.query import SearchResult, search
from smart_research_agent.course_index.types import (
    COURSE_INDEX_BOUNDARIES,
    COURSE_INDEX_NOTES,
    COURSE_INDEX_NOTES_ORDER,
    GOLD_QUERY,
    TOP_K_DEFAULT,
)


@dataclass(frozen=True)
class CorpusRow:
    """语料表的一行：一份材料."""

    index: int
    name: str
    kind: str
    chars: int

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {"index": self.index, "name": self.name, "kind": self.kind, "chars": self.chars}

    def line(self) -> str:
        """`` 1. doc:capstone                     | doc        |  3751 字``."""
        return f"{self.index:>2}. {self.name:<32} | {self.kind:<10} | {self.chars:>6} 字"


@dataclass(frozen=True)
class IndexRow:
    """索引表的一行：一个 token 的倒排表."""

    index: int
    token: str
    documents: int
    preview: str

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {"index": self.index, "token": self.token, "documents": self.documents}

    def line(self) -> str:
        """`` 1. 反向                       | 文档  12 | doc:backprop=9、subpackage:backprop=5…``."""
        return f"{self.index:>2}. {self.token:<24} | 文档 {self.documents:>3} | {self.preview}"


@dataclass(frozen=True)
class HitRow:
    """检索表的一行：一条命中."""

    rank: int
    document: str
    score: float
    matched: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {
            "rank": self.rank,
            "document": self.document,
            "score": round(self.score, 6),
            "matched": list(self.matched),
        }

    def line(self) -> str:
        """`` 1. subpackage:course_index         | 分数 1.0000 | 命中 [course_index]``."""
        return (
            f"{self.rank:>2}. {self.document:<34} | 分数 {self.score:.4f}"
            f" | 命中 [{('、'.join(self.matched))}]"
        )


@dataclass(frozen=True)
class PropertyRow:
    """性质表的一行：一条性质 + 它的判据与读数."""

    index: int
    name: str
    criterion: str
    reading: float
    passed: bool
    evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {
            "index": self.index,
            "name": self.name,
            "criterion": self.criterion,
            "reading": self.reading,
            "passed": self.passed,
            "evidence": list(self.evidence),
        }

    def line(self) -> str:
        """`` 1. corpus_covers_all_docs            | [equality   ] | 读数 0 ✓ | ……``."""
        mark = "✓" if self.passed else "✗"
        detail = "；".join(self.evidence)
        return (
            f"{self.index:>2}. {self.name:<36} | [{self.criterion:<11}] | "
            f"读数 {self.reading:>10.6g} {mark} | {detail}"
        )


def corpus_rows(corpus: Corpus | None = None) -> tuple[CorpusRow, ...]:
    """① 语料表：全部材料逐行."""
    resolved = build_corpus() if corpus is None else corpus
    return tuple(
        CorpusRow(index=index, name=document.name, kind=document.kind, chars=document.chars)
        for index, document in enumerate(resolved.documents, start=1)
    )


def index_rows(index: InvertedIndex | None = None, *, limit: int = 12) -> tuple[IndexRow, ...]:
    """② 索引表：按"出现文档数"降序取前 ``limit`` 个 token（并列时按名字）."""
    resolved = build_index() if index is None else index
    ranked = sorted(resolved.postings, key=lambda item: (-len(item[1]), item[0]))[:limit]
    rows: list[IndexRow] = []
    for position, (token, pairs) in enumerate(ranked, start=1):
        preview = "、".join(f"{name}={count}" for name, count in pairs[:4])
        more = "" if len(pairs) <= 4 else f"（+{len(pairs) - 4}）"
        rows.append(
            IndexRow(index=position, token=token, documents=len(pairs), preview=preview + more)
        )
    return tuple(rows)


def hit_rows(result: SearchResult | None = None, *, query: str = GOLD_QUERY) -> tuple[HitRow, ...]:
    """③ 检索表：一次检索的名次逐行."""
    resolved = search(query) if result is None else result
    return tuple(
        HitRow(rank=rank, document=hit.document, score=hit.score, matched=hit.matched)
        for rank, hit in enumerate(resolved.hits, start=1)
    )


def property_rows(report: verify.PropertyReport | None = None) -> tuple[PropertyRow, ...]:
    """④ 性质表：7 条性质逐行（判据类别与读数一起印）."""
    resolved = verify.check_all() if report is None else report
    return tuple(
        PropertyRow(
            index=index,
            name=outcome.name,
            criterion=outcome.criterion,
            reading=outcome.cross_check.reading if outcome.cross_check else 0.0,
            passed=outcome.passed,
            evidence=outcome.evidence,
        )
        for index, outcome in enumerate(resolved.outcomes, start=1)
    )


def note_lines(limit: int | None = None) -> tuple[str, ...]:
    """十条笔记逐行印出（顺序即写入顺序）."""
    keys = COURSE_INDEX_NOTES_ORDER if limit is None else COURSE_INDEX_NOTES_ORDER[:limit]
    return tuple(f"{index:>2}. {COURSE_INDEX_NOTES[key]}" for index, key in enumerate(keys, start=1))


def boundary_lines() -> tuple[str, ...]:
    """五条边界逐行印出."""
    return tuple(f"- {item}" for item in COURSE_INDEX_BOUNDARIES)


def study_lines(
    *,
    corpus: Corpus | None = None,
    index: InvertedIndex | None = None,
    result: SearchResult | None = None,
    report: verify.PropertyReport | None = None,
    query: str = GOLD_QUERY,
    top_k: int = TOP_K_DEFAULT,
) -> tuple[str, ...]:
    """一次跑完四张表（演示脚本与教程引用的是同一批读数）.

    四张表共用**同一份语料与索引**，因此它们读的是同一时刻的状态。
    """
    resolved_corpus = build_corpus() if corpus is None else corpus
    resolved_index = build_index(resolved_corpus) if index is None else index
    resolved_result = (
        search(query, index=resolved_index, top_k=top_k) if result is None else result
    )
    resolved_report = (
        verify.check_all(corpus=resolved_corpus, index=resolved_index, result=resolved_result, query=query)
        if report is None
        else report
    )
    lines: list[str] = []
    lines.append("== 1. 语料表（两种材料：名字 / 种类 / 字符数）")
    for row in corpus_rows(resolved_corpus):
        lines.append("  " + row.line())
    lines.append("  " + resolved_corpus.line())
    lines.append("== 2. 索引表（最常见的 token：出现它的文档数 / 前几个文档）")
    for row in index_rows(resolved_index):
        lines.append("  " + row.line())
    lines.append("  " + resolved_index.line())
    lines.append(f"== 3. 检索表（金标准查询 {query!r}：名次 / 文档 / 分数 / 命中词）")
    for row in hit_rows(resolved_result):
        lines.append("  " + row.line())
    lines.append("  " + resolved_result.line())
    lines.append("== 4. 性质表（7 条性质：判据类别 / 读数 / 结论）")
    for row in property_rows(resolved_report):
        lines.append("  " + row.line())
    lines.append("  " + corpus_module.corpus_lines(resolved_corpus)[-1])
    lines.append("  " + query_module.search_lines(resolved_result)[1])
    return tuple(lines)


def to_dict_lines(rows: tuple[Any, ...]) -> tuple[dict[str, Any], ...]:
    """把一串带 ``to_dict`` 的行折成可 JSON 化的字段（报告导出的便利函数）."""
    return tuple(row.to_dict() for row in rows)


__all__ = [
    "CorpusRow",
    "HitRow",
    "IndexRow",
    "PropertyRow",
    "boundary_lines",
    "corpus_rows",
    "hit_rows",
    "index_rows",
    "note_lines",
    "property_rows",
    "study_lines",
    "to_dict_lines",
]
