"""七条性质与三类判据（day101）.

本模块是"这份索引成不成立"的判据所在。七条性质按三类判据分：

```text
相等（==）   ① corpus_covers_all_docs            手册缺失数 == 0
             ② corpus_covers_all_subpackages     子包缺失数 == 0
             ③ index_is_reproducible             两次构建的差异项数 == 0
             ④ search_is_deterministic           两次检索的差异项数 == 0
上界（<=）   ⑤ scores_within_bounds               最高分数 <= 1.0
下界（>=）   ⑥ every_document_is_indexed          最小词数 >= 1
             ⑦ gold_query_finds_gold_document     金标准材料被命中数 >= 1
```

## 一、为什么 ⑥ ⑦ 必须是下界

```text
"每份文档至少被索引一个词"   越多越好（词多不是坏事）⇒ 只能从下面兜住
"金标准材料必须被命中"       命中 >= 1 才是成立         ⇒ 只能从下面兜住
```

把它写成相等（``== 1``）会在"一份文档切出更多词"或"金标准被命中两次（前排与后排）"时
误报失败。下界才是它们真正的方向。

## 二、为什么 ⑤ 是上界

"分数不超过 1"是一条**越少越好**的量。它当然也写作 ``== 1`` 也说得通（数学上界就是 1），
但写成上界能让"为什么不是下界"这件事在判据名上说清楚：
分数是**越界就不可用**的量，因此兜的是上限。

## 三、一条纪律：`CrossCheck.passed` 必须与 `PropertyOutcome.passed` 一致

一份报告里，"那一行说通过"与"它挂的对账说没通过"是最糟的状态——
读的人只会相信看起来更合理的那一个。因此 :class:`PropertyOutcome` 在
``__post_init__`` 里当场拒绝两者不一致的构造。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.course_index.corpus import (
    NAME_SEPARATOR,
    Corpus,
    build_corpus,
    expected_names,
)
from smart_research_agent.course_index.errors import NumericError, ParameterError
from smart_research_agent.course_index.index import InvertedIndex, build_index
from smart_research_agent.course_index.query import SearchResult, search
from smart_research_agent.course_index.types import (
    COURSE_INDEX_PROPERTIES,
    CRITERION_EQUALITY,
    CRITERION_LOWER_BOUND,
    CRITERION_UPPER_BOUND,
    DOC_KIND_DOC,
    DOC_KIND_SUBPACKAGE,
    GOLD_DOC,
    GOLD_QUERY,
    MIN_WORD_LEN,
    PROPERTY_CORPUS_COVERS_ALL_DOCS,
    PROPERTY_CORPUS_COVERS_ALL_SUBPACKAGES,
    PROPERTY_EVERY_DOCUMENT_IS_INDEXED,
    PROPERTY_GOLD_QUERY_FINDS_GOLD_DOCUMENT,
    PROPERTY_INDEX_IS_REPRODUCIBLE,
    PROPERTY_SCORES_WITHIN_BOUNDS,
    PROPERTY_SEARCH_IS_DETERMINISTIC,
)

#: 相等判据的浮点容差（逐位 / 整数相等时不走它）.
EQUALITY_TOLERANCE = 1e-12

#: 分数上界（``score = 命中词数 / 查询词数`` 的数学上界）.
SCORE_CEILING = 1.0

#: 每份文档至少要索引多少个词（**下界**）.
TOKEN_FLOOR = 1.0


@dataclass(frozen=True)
class CrossCheck:
    """一次对账：来源、读数、判据（**含方向**：相等 / 上界 / 下界）."""

    name: str
    left: str
    right: str
    reading: float
    expected: float
    exact: bool = True
    upper_bound: float | None = None
    lower_bound: float | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.left or not self.right:
            raise ParameterError("对账的来源与名字都不能为空。")
        if not math.isfinite(self.reading) or not math.isfinite(self.expected):
            raise NumericError(
                f"对账读数必须有限：reading={self.reading!r}、expected={self.expected!r}。"
            )
        if self.upper_bound is not None and self.lower_bound is not None:
            raise ParameterError(
                f"对账 {self.name!r} 同时给了上界与下界：请只给一个方向。"
            )
        if self.upper_bound is not None and (
            not math.isfinite(self.upper_bound) or self.upper_bound < 0
        ):
            raise NumericError(f"上界必须是有限非负数，收到 {self.upper_bound!r}。")
        if self.lower_bound is not None and not math.isfinite(self.lower_bound):
            raise NumericError(f"下界必须是有限数，收到 {self.lower_bound!r}。")

    @property
    def criterion(self) -> str:
        """这次用的是哪一类判据（相等 / 上界 / 下界）."""
        if self.lower_bound is not None:
            return CRITERION_LOWER_BOUND
        if self.upper_bound is not None:
            return CRITERION_UPPER_BOUND
        return CRITERION_EQUALITY

    @property
    def passed(self) -> bool:
        """三类判据各自的判定."""
        if self.lower_bound is not None:
            return float(self.reading) >= float(self.lower_bound) - EQUALITY_TOLERANCE
        if self.upper_bound is not None:
            return float(self.reading) <= float(self.upper_bound) + EQUALITY_TOLERANCE
        if self.exact:
            return self.reading == self.expected
        return abs(float(self.reading) - float(self.expected)) <= EQUALITY_TOLERANCE

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {
            "name": self.name,
            "criterion": self.criterion,
            "reading": self.reading,
            "expected": self.expected,
            "upper_bound": self.upper_bound,
            "lower_bound": self.lower_bound,
            "passed": self.passed,
        }

    def line(self) -> str:
        """一行可读读数（带上方向与来源）."""
        verdict = "满足" if self.passed else "不满足"
        if self.lower_bound is not None:
            return (
                f"[{verdict}] {self.name}: 读数 {self.reading:.6e} ≥ 下界 "
                f"{self.lower_bound:.6e}（{self.left} vs {self.right}）"
            )
        if self.upper_bound is not None:
            return (
                f"[{verdict}] {self.name}: 读数 {self.reading:.6e} ≤ 上界 "
                f"{self.upper_bound:.6e}（{self.left} vs {self.right}）"
            )
        return (
            f"[{verdict}] {self.name}: {self.left} vs {self.right} | "
            f"读数 {self.reading} / 期望 {self.expected}"
        )


@dataclass(frozen=True)
class PropertyOutcome:
    """一条性质的结论：适用性 + 结果 + 证据行 + 一次对账."""

    name: str
    applicable: bool
    passed: bool
    evidence: tuple[str, ...] = field(default_factory=tuple)
    cross_check: CrossCheck | None = None

    def __post_init__(self) -> None:
        if not self.applicable and self.passed:
            raise NumericError(
                f"性质 {self.name!r} 标了'不适用'却又标了'通过'：两者必须分开。"
            )
        if self.cross_check is not None and self.cross_check.passed != self.passed:
            raise NumericError(
                f"性质 {self.name!r} 的结论与它挂的对账不一致："
                f"outcome.passed={self.passed}、cross_check.passed={self.cross_check.passed}。"
            )

    @property
    def criterion(self) -> str:
        """这条性质用的是哪一类判据（没有对账时按相等记）."""
        return CRITERION_EQUALITY if self.cross_check is None else self.cross_check.criterion

    def line(self) -> str:
        """一行可读结论（不适用也要印出来）."""
        if not self.applicable:
            return f"[不适用] {self.name} | {'；'.join(self.evidence)}"
        verdict = "通过" if self.passed else "失败"
        detail = "；".join(self.evidence)
        suffix = f" | {detail}" if detail else ""
        return f"[{verdict}] {self.name}{suffix}"

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {
            "name": self.name,
            "criterion": self.criterion,
            "applicable": self.applicable,
            "passed": self.passed,
            "evidence": list(self.evidence),
            "cross_check": None if self.cross_check is None else self.cross_check.to_dict(),
        }


@dataclass(frozen=True)
class PropertyReport:
    """一组性质的报告（``ok`` 要求**所有适用**的都通过）."""

    outcomes: tuple[PropertyOutcome, ...]

    @property
    def applicable(self) -> tuple[PropertyOutcome, ...]:
        """适用（``applicable=True``）的那些性质."""
        return tuple(outcome for outcome in self.outcomes if outcome.applicable)

    @property
    def ok(self) -> bool:
        """是否全部通过——不适用不算通过、也不算失败."""
        return all(outcome.passed for outcome in self.applicable)

    def require_ok(self) -> None:
        """不通过时抛 :class:`errors.CorpusError`（"拒绝交付"的那条路）."""
        if self.ok:
            return
        from smart_research_agent.course_index.errors import CorpusError

        failures = [outcome.line() for outcome in self.applicable if not outcome.passed]
        raise CorpusError("性质检查未全部通过：" + "；".join(failures))

    def lines(self) -> tuple[str, ...]:
        """逐行文本（**先印不适用**）."""
        return tuple(
            outcome.line()
            for outcome in sorted(self.outcomes, key=lambda outcome: outcome.applicable)
        )

    def to_dict(self) -> dict[str, Any]:
        """摊平成可 JSON 化的字段."""
        return {
            "ok": self.ok,
            "counts": {"total": len(self.outcomes), "applicable": len(self.applicable)},
            "lines": list(self.lines()),
        }


# --------------------------------------------------------------------------- #
# 七条性质
# --------------------------------------------------------------------------- #


def _expected_of_kind(expected: tuple[str, ...] | None, kind: str) -> tuple[str, ...]:
    """从"期望名单"里挑出某一种语料的名字（``None`` 时现场读一遍）."""
    resolved = expected_names() if expected is None else expected
    prefix = f"{kind}{NAME_SEPARATOR}"
    return tuple(name for name in resolved if name.startswith(prefix))


def check_corpus_covers_all_docs(
    corpus: Corpus | None = None,
    *,
    expected: tuple[str, ...] | None = None,
) -> PropertyOutcome:
    """① ``docs/`` 下的每一份 ``*.md`` 都进了语料."""
    resolved = build_corpus() if corpus is None else corpus
    doc_names = _expected_of_kind(expected, DOC_KIND_DOC)
    missing = resolved.missing(doc_names)
    check = CrossCheck(
        name="手册缺失数",
        left="Corpus 里 doc: 开头的名字",
        right="docs/*.md 的文件名",
        reading=float(len(missing)),
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_CORPUS_COVERS_ALL_DOCS,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"手册 {len(resolved.docs)} 份 / 期望 {len(doc_names)} 份 | 缺失 {len(missing)}",
            "缺失的：" + ("、".join(missing) if missing else "无"),
        ),
        cross_check=check,
    )


def check_corpus_covers_all_subpackages(
    corpus: Corpus | None = None,
    *,
    expected: tuple[str, ...] | None = None,
) -> PropertyOutcome:
    """② 每一个子包都有一条语料条目."""
    resolved = build_corpus() if corpus is None else corpus
    sub_names = _expected_of_kind(expected, DOC_KIND_SUBPACKAGE)
    missing = resolved.missing(sub_names)
    check = CrossCheck(
        name="子包缺失数",
        left="Corpus 里 subpackage: 开头的名字",
        right="discover_subpackages()",
        reading=float(len(missing)),
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_CORPUS_COVERS_ALL_SUBPACKAGES,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"子包 {len(resolved.subpackages)} 条 / 期望 {len(sub_names)} 条 | 缺失 {len(missing)}",
            "缺失的：" + ("、".join(missing) if missing else "无"),
        ),
        cross_check=check,
    )


def check_index_is_reproducible(
    first: InvertedIndex | None = None,
    second: InvertedIndex | None = None,
    *,
    corpus: Corpus | None = None,
) -> PropertyOutcome:
    """③ 同一份语料两次构建索引**逐位相同**."""
    resolved_corpus = build_corpus() if corpus is None else corpus
    left = build_index(resolved_corpus) if first is None else first
    right = build_index(resolved_corpus) if second is None else second
    diff = left.diff_count(right)
    check = CrossCheck(
        name="两次构建的差异项数",
        left="build_index 第一次",
        right="build_index 第二次",
        reading=float(diff),
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_INDEX_IS_REPRODUCIBLE,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"差异项数 {diff}（0 = 逐位相同）",
            f"摘要 {left.digest()} / {right.digest()}",
        ),
        cross_check=check,
    )


def check_search_is_deterministic(
    first: SearchResult | None = None,
    second: SearchResult | None = None,
    *,
    index: InvertedIndex | None = None,
    query: str = GOLD_QUERY,
) -> PropertyOutcome:
    """④ 同一个查询两次检索给出同一份命中（顺序也相同）."""
    resolved_index = build_index() if index is None else index
    left = search(query, index=resolved_index) if first is None else first
    right = search(query, index=resolved_index) if second is None else second
    diff = left.diff_count(right)
    check = CrossCheck(
        name="两次检索的差异项数",
        left="search 第一次",
        right="search 第二次",
        reading=float(diff),
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_SEARCH_IS_DETERMINISTIC,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"差异项数 {diff}（0 = 逐位相同）",
            f"查询 {query!r} 命中 {len(left.hits)} 条",
        ),
        cross_check=check,
    )


def check_scores_within_bounds(
    result: SearchResult | None = None,
    *,
    index: InvertedIndex | None = None,
    query: str = GOLD_QUERY,
) -> PropertyOutcome:
    """⑤ **上界**：每一个命中分数都落在 ``[0, 1]``."""
    resolved = (
        search(query, index=build_index() if index is None else index) if result is None else result
    )
    highest = float(resolved.top_score)
    check = CrossCheck(
        name="最高命中分数",
        left="SearchResult.top_score",
        right=f"上界 {SCORE_CEILING:.1f}",
        reading=highest,
        expected=SCORE_CEILING,
        exact=False,
        upper_bound=SCORE_CEILING,
    )
    return PropertyOutcome(
        name=PROPERTY_SCORES_WITHIN_BOUNDS,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"最高分数 {highest:.4f} ≤ 上界 {SCORE_CEILING:.1f}",
            resolved.line(),
        ),
        cross_check=check,
    )


def check_every_document_is_indexed(
    index: InvertedIndex | None = None,
) -> PropertyOutcome:
    """⑥ **下界**：每一份语料至少被索引了一个词."""
    resolved = build_index() if index is None else index
    fewest = float(min(count for _, count in resolved.tokens_per_document))
    check = CrossCheck(
        name="最小的每份文档词数",
        left="InvertedIndex.tokens_per_document",
        right=f"下界 {TOKEN_FLOOR:.0f}",
        reading=fewest,
        expected=TOKEN_FLOOR,
        exact=False,
        lower_bound=TOKEN_FLOOR,
    )
    return PropertyOutcome(
        name=PROPERTY_EVERY_DOCUMENT_IS_INDEXED,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"最小词数 {fewest:.0f} ≥ 下界 {TOKEN_FLOOR:.0f}"
            f"（分词最小长度 {MIN_WORD_LEN}）",
            resolved.line(),
        ),
        cross_check=check,
    )


def check_gold_query_finds_gold_document(
    result: SearchResult | None = None,
    *,
    index: InvertedIndex | None = None,
) -> PropertyOutcome:
    """⑦ **下界**：金标准查询命中它该命中的那一份材料."""
    resolved = (
        search(GOLD_QUERY, index=build_index() if index is None else index)
        if result is None
        else result
    )
    matched = sum(1 for hit in resolved.hits if hit.document == GOLD_DOC)
    check = CrossCheck(
        name="金标准材料的命中数",
        left=f"search({GOLD_QUERY!r})",
        right=f"必须命中 {GOLD_DOC}",
        reading=float(matched),
        expected=1.0,
        exact=False,
        lower_bound=1.0,
    )
    return PropertyOutcome(
        name=PROPERTY_GOLD_QUERY_FINDS_GOLD_DOCUMENT,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"命中 {matched} 次 ≥ 下界 1 | 名次 {resolved.rank_of(GOLD_DOC)}",
            resolved.line(),
        ),
        cross_check=check,
    )


def check_all(
    *,
    corpus: Corpus | None = None,
    index: InvertedIndex | None = None,
    result: SearchResult | None = None,
    expected: tuple[str, ...] | None = None,
    query: str = GOLD_QUERY,
) -> PropertyReport:
    """一次跑完七条性质（顺序与 :data:`types.COURSE_INDEX_PROPERTIES` 一致）.

    七条性质**共用同一份语料与索引**，因此报告读的是同一时刻的状态。
    """
    from smart_research_agent.course_index.errors import CourseIndexError

    resolved_corpus = build_corpus() if corpus is None else corpus
    resolved_index = build_index(resolved_corpus) if index is None else index
    resolved_result = (
        search(query, index=resolved_index) if result is None else result
    )
    outcomes = (
        check_corpus_covers_all_docs(resolved_corpus, expected=expected),
        check_corpus_covers_all_subpackages(resolved_corpus, expected=expected),
        check_index_is_reproducible(corpus=resolved_corpus),
        check_search_is_deterministic(index=resolved_index, query=query),
        check_scores_within_bounds(resolved_result),
        check_every_document_is_indexed(resolved_index),
        check_gold_query_finds_gold_document(resolved_result),
    )
    names = {outcome.name for outcome in outcomes}
    extra = names - set(COURSE_INDEX_PROPERTIES)
    missing = set(COURSE_INDEX_PROPERTIES) - names
    if extra or missing:  # pragma: no cover - 只在有人改性质名单时触发
        raise CourseIndexError(
            "性质名单与 types.COURSE_INDEX_PROPERTIES 不一致："
            f"多 {sorted(extra)}、缺 {sorted(missing)}。"
        )
    return PropertyReport(outcomes=outcomes)


def require_ok(report: PropertyReport | None = None) -> PropertyReport:
    """七条性质没全部通过时抛 :class:`CorpusError`（"拒绝交付"的那条路）."""
    resolved = check_all() if report is None else report
    if not resolved.ok:
        resolved.require_ok()
    return resolved


__all__ = [
    "EQUALITY_TOLERANCE",
    "SCORE_CEILING",
    "TOKEN_FLOOR",
    "CrossCheck",
    "PropertyOutcome",
    "PropertyReport",
    "check_all",
    "check_corpus_covers_all_docs",
    "check_corpus_covers_all_subpackages",
    "check_every_document_is_indexed",
    "check_gold_query_finds_gold_document",
    "check_index_is_reproducible",
    "check_scores_within_bounds",
    "check_search_is_deterministic",
    "require_ok",
]
