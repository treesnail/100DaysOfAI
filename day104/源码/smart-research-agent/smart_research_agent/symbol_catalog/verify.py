"""七条性质与三类判据（day104）.

本模块是"这份符号清单成不成立"的判据所在。七条性质按三类判据分：

```text
相等（==）   ① catalog_covers_all_modules          缺失数 == 0
             ② catalog_is_reproducible             两次构建的差异项数 == 0
             ③ phantom_exports_are_sound           不成立的幽灵数 == 0
             ④ shared_names_are_sound              不成立的重名条目数 == 0
上界（<=）   ⑤ duplicate_declarations_within_module 重复声明的模块数 <= 0
下界（>=）   ⑥ every_declaring_module_promises       最小承诺数 >= 1
             ⑦ export_names_are_reused              跨模块重名数 >= 1
```

## 一、为什么 ⑥ ⑦ 必须是下界

```text
"每个写了 __all__ 的模块至少承诺一个名字"   空承诺是**从下面**兜住的（写成 == 1 会漏掉更长的名单）
"至少有一个名字被两个以上模块导出"           "有重名"这件事只能从下面兜住
```

## 二、为什么 ⑤ 是上界

"重复声明的模块数"是一条**越少越好**的量：0 才是健康的。
一个名字在同一个 ``__all__`` 里出现两次，读的人就不知道这一条到底算几条——
这与 day101 的"分数不超过 1"、day102 的"未解析基类不超过 0"、day103 的"未解析目标不超过 0"同源。

## 三、一条纪律：`CrossCheck.passed` 必须与 `PropertyOutcome.passed` 一致

一份报告里，"那一行说通过"与"它挂的对账说没通过"是最糟的状态。
因此 :class:`PropertyOutcome` 在 ``__post_init__`` 里当场拒绝两者不一致的构造。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from smart_research_agent.symbol_catalog.catalog import SymbolCatalog, build_catalog
from smart_research_agent.symbol_catalog.errors import NumericError, ParameterError, SymbolError
from smart_research_agent.symbol_catalog.parse import (
    ModuleBindings,
    expected_modules,
    outside_all_names,
)
from smart_research_agent.symbol_catalog.types import (
    CRITERION_EQUALITY,
    CRITERION_LOWER_BOUND,
    CRITERION_UPPER_BOUND,
    PROPERTY_CATALOG_COVERS_ALL_MODULES,
    PROPERTY_CATALOG_IS_REPRODUCIBLE,
    PROPERTY_DUPLICATE_DECLARATIONS_WITHIN_MODULE,
    PROPERTY_EVERY_DECLARING_MODULE_PROMISES,
    PROPERTY_EXPORT_NAMES_ARE_REUSED,
    PROPERTY_PHANTOMS_ARE_SOUND,
    PROPERTY_QUALIFIED_NAMES_ARE_UNIQUE,
    SOURCE_DECLARED,
    SYMBOL_CATALOG_PROPERTIES,
)

#: 相等判据的浮点容差（逐位 / 整数相等时不走它）.
EQUALITY_TOLERANCE = 1e-12

#: 重复声明的模块数的上界（一个都不许有）.
DUPLICATE_CEILING = 0.0

#: 每个"写了 __all__"的模块至少要承诺几个名字（**下界**：至少一个）.
PROMISE_FLOOR = 1.0

#: 跨模块重名的下界（**下界**：至少有一个名字被两个以上模块导出）.
REUSE_FLOOR = 1.0


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
            raise ParameterError(f"对账 {self.name!r} 同时给了上界与下界：请只给一个方向。")
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
            raise NumericError(f"性质 {self.name!r} 标了'不适用'却又标了'通过'：两者必须分开。")
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
        """不通过时抛 :class:`errors.ParseError`（"拒绝交付"的那条路）."""
        if self.ok:
            return
        from smart_research_agent.symbol_catalog.errors import ParseError

        failures = [outcome.line() for outcome in self.applicable if not outcome.passed]
        raise ParseError("性质检查未全部通过：" + "；".join(failures))

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


def _resolve(catalog: SymbolCatalog | None) -> SymbolCatalog:
    """缺省时现场编一份清单."""
    return build_catalog() if catalog is None else catalog


def check_catalog_covers_all_modules(
    catalog: SymbolCatalog | None = None,
    *,
    expected: tuple[str, ...] | None = None,
) -> PropertyOutcome:
    """① 每一个 ``.py`` 都是清单里的一行."""
    resolved = _resolve(catalog)
    expected_names = expected_modules() if expected is None else expected
    missing = tuple(name for name in expected_names if name not in resolved.module_set)
    check = CrossCheck(
        name="模块缺失数",
        left="SymbolCatalog.modules",
        right="扫描到的 .py 模块名",
        reading=float(len(missing)),
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_CATALOG_COVERS_ALL_MODULES,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"模块 {resolved.module_count} 个 / 期望 {len(expected_names)} 个 | 缺失 {len(missing)}",
            "缺失的：" + ("、".join(missing) if missing else "无"),
        ),
        cross_check=check,
    )


def check_catalog_is_reproducible(
    first: SymbolCatalog | None = None,
    second: SymbolCatalog | None = None,
    *,
    modules: tuple[ModuleBindings, ...] | None = None,
) -> PropertyOutcome:
    """② 同一批文件两次编出的清单**逐位相同**."""
    left = build_catalog(modules) if first is None else first
    right = build_catalog(modules) if second is None else second
    diff = left.diff_count(right)
    check = CrossCheck(
        name="两次构建的差异项数",
        left="build_catalog 第一次",
        right="build_catalog 第二次",
        reading=float(diff),
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_CATALOG_IS_REPRODUCIBLE,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"差异项数 {diff}（0 = 逐位相同）",
            f"摘要 {left.digest()} / {right.digest()}",
        ),
        cross_check=check,
    )


def check_phantom_exports_are_sound(catalog: SymbolCatalog | None = None) -> PropertyOutcome:
    """③ 报出来的每一个幽灵导出都是**真的**.

    独立复核走的是**纯文本**这条路：把该模块里每一段 ``__all__`` 的源码挖掉之后，
    这个名字应当**一次都不出现**（见 :func:`parse.outside_all_names`）。
    这与 :meth:`ModuleBindings.binding_of` 是两条独立的路，因此它能抓住"解析漏了一类绑定"。
    """
    resolved = _resolve(catalog)
    unsound: list[str] = []
    for module, name in resolved.phantom_pairs():
        outside = outside_all_names(Path(resolved.entry_of(module).path))
        if name in outside:
            unsound.append(f"{module}.{name}")
    check = CrossCheck(
        name="不成立的幽灵数",
        left="SymbolCatalog.phantom_pairs()",
        right="名字在 __all__ 之外一次都不出现",
        reading=float(len(unsound)),
        expected=0.0,
        exact=True,
    )
    preview = "、".join(unsound[:3])
    return PropertyOutcome(
        name=PROPERTY_PHANTOMS_ARE_SOUND,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"幽灵 {len(resolved.phantom_pairs())} 个 / 不成立 {len(unsound)} 个",
            "不成立的：" + (preview if preview else "无"),
        ),
        cross_check=check,
    )


def check_qualified_names_are_unique(catalog: SymbolCatalog | None = None) -> PropertyOutcome:
    """④ 报出来的每一个"跨模块重名"都是**真的**（独立再数一遍它的导出模块）.

    这条与 :meth:`SymbolCatalog.shared_names` 的算法是两条独立的路：
    这里从 ``export_pairs()`` 逐条重数，因此它能抓住"重名表把某个模块算漏了 / 算重了"。
    """
    resolved = _resolve(catalog)
    by_name: dict[str, set[str]] = {}
    for module, name in resolved.export_pairs():
        by_name.setdefault(name, set()).add(module)
    unsound: list[str] = []
    for name, modules in resolved.shared_names():
        if len(modules) < 2 or set(modules) != by_name.get(name, set()):
            unsound.append(name)
    check = CrossCheck(
        name="不成立的重名条目数",
        left="SymbolCatalog.shared_names()",
        right="export_pairs() 逐条重数",
        reading=float(len(unsound)),
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_QUALIFIED_NAMES_ARE_UNIQUE,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"重名条目 {len(resolved.shared_names())} 个 / 不成立 {len(unsound)} 个",
            "不成立的：" + ("、".join(unsound[:3]) if unsound else "无"),
        ),
        cross_check=check,
    )


def check_duplicate_declarations_within_module(
    catalog: SymbolCatalog | None = None,
) -> PropertyOutcome:
    """⑤ **上界**：同一个 ``__all__`` 里不重复声明同一个名字."""
    resolved = _resolve(catalog)
    duplicated = resolved.duplicate_declarations()
    check = CrossCheck(
        name="重复声明的模块数",
        left="SymbolCatalog.duplicate_declarations()",
        right=f"上界 {DUPLICATE_CEILING:.0f}",
        reading=float(len(duplicated)),
        expected=DUPLICATE_CEILING,
        exact=False,
        upper_bound=DUPLICATE_CEILING,
    )
    return PropertyOutcome(
        name=PROPERTY_DUPLICATE_DECLARATIONS_WITHIN_MODULE,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"重复 {len(duplicated)} 个 ≤ 上界 {DUPLICATE_CEILING:.0f}",
            "重复的：" + ("、".join(duplicated[:3]) if duplicated else "无"),
        ),
        cross_check=check,
    )


def check_every_declaring_module_promises_something(
    catalog: SymbolCatalog | None = None,
) -> PropertyOutcome:
    """⑥ **下界**：每一个写了字面量 ``__all__`` 的模块至少承诺一个名字."""
    resolved = _resolve(catalog)
    declared_lengths = [
        len(entry.declared) for entry in resolved.entries if entry.source == SOURCE_DECLARED
    ]
    fewest = float(min(declared_lengths, default=0))
    check = CrossCheck(
        name="最小的字面量承诺数",
        left="SymbolCatalog.entries[*].declared",
        right=f"下界 {PROMISE_FLOOR:.0f}",
        reading=fewest,
        expected=PROMISE_FLOOR,
        exact=False,
        lower_bound=PROMISE_FLOOR,
    )
    return PropertyOutcome(
        name=PROPERTY_EVERY_DECLARING_MODULE_PROMISES,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"最小承诺 {fewest:.0f} ≥ 下界 {PROMISE_FLOOR:.0f}",
            f"字面量承诺的模块 {len(declared_lengths)} 个",
        ),
        cross_check=check,
    )


def check_export_names_are_reused(catalog: SymbolCatalog | None = None) -> PropertyOutcome:
    """⑦ **下界**：至少有一个名字被两个以上的模块导出."""
    resolved = _resolve(catalog)
    shared = len(resolved.shared_names())
    check = CrossCheck(
        name="被多个模块导出的名字数",
        left="SymbolCatalog.shared_names()",
        right=f"下界 {REUSE_FLOOR:.0f}",
        reading=float(shared),
        expected=REUSE_FLOOR,
        exact=False,
        lower_bound=REUSE_FLOOR,
    )
    return PropertyOutcome(
        name=PROPERTY_EXPORT_NAMES_ARE_REUSED,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"跨模块重名 {shared} 个 ≥ 下界 {REUSE_FLOOR:.0f}",
            f"承诺总数 {resolved.promise_count()} | 模块 {resolved.module_count}",
        ),
        cross_check=check,
    )


def check_all(
    *,
    catalog: SymbolCatalog | None = None,
    modules: tuple[ModuleBindings, ...] | None = None,
    expected: tuple[str, ...] | None = None,
) -> PropertyReport:
    """一次跑完七条性质（顺序与 :data:`types.SYMBOL_CATALOG_PROPERTIES` 一致）.

    七条性质**共用同一份清单**，因此报告读的是同一时刻的状态。
    """
    resolved = build_catalog(modules) if catalog is None else catalog
    outcomes = (
        check_catalog_covers_all_modules(resolved, expected=expected),
        check_catalog_is_reproducible(resolved, resolved),
        check_phantom_exports_are_sound(resolved),
        check_qualified_names_are_unique(resolved),
        check_duplicate_declarations_within_module(resolved),
        check_every_declaring_module_promises_something(resolved),
        check_export_names_are_reused(resolved),
    )
    names = {outcome.name for outcome in outcomes}
    extra = names - set(SYMBOL_CATALOG_PROPERTIES)
    missing = set(SYMBOL_CATALOG_PROPERTIES) - names
    if extra or missing:  # pragma: no cover - 只在有人改性质名单时触发
        raise SymbolError(
            "性质名单与 types.SYMBOL_CATALOG_PROPERTIES 不一致："
            f"多 {sorted(extra)}、缺 {sorted(missing)}。"
        )
    return PropertyReport(outcomes=outcomes)


def require_ok(report: PropertyReport | None = None) -> PropertyReport:
    """七条性质没全部通过时抛 :class:`ParseError`（"拒绝交付"的那条路）."""
    resolved = check_all() if report is None else report
    if not resolved.ok:
        resolved.require_ok()
    return resolved


__all__ = [
    "DUPLICATE_CEILING",
    "EQUALITY_TOLERANCE",
    "PROMISE_FLOOR",
    "REUSE_FLOOR",
    "CrossCheck",
    "PropertyOutcome",
    "PropertyReport",
    "check_all",
    "check_catalog_covers_all_modules",
    "check_catalog_is_reproducible",
    "check_duplicate_declarations_within_module",
    "check_every_declaring_module_promises_something",
    "check_export_names_are_reused",
    "check_phantom_exports_are_sound",
    "check_qualified_names_are_unique",
    "require_ok",
]
