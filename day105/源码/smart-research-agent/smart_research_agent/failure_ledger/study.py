"""``study``：四张表（day102）.

**一个数只有被印出来才可能被反驳**。今天四张表各回答一个"这份台账成不成立"的问题：

```text
① 模块表   每份 errors.py：包名 / 族数 / 它的根族
② 族表     每个族：包.名 / 身份 / 它的基类
③ 归属表   基类按四类归属的计数（local / builtin / imported / unresolved）
④ 性质表   7 条性质：判据类别 / 读数 / 结论 / 一行证据
```

## 一条纪律：每一行都要带**两个数**

与前面各天的表同源——一行只写"通过"的表是没法反驳的。
因此每一行的"读数"旁边都跟着一个"参照"：族数 / 身份 / 归属类别 / 判据与期望。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smart_research_agent.failure_ledger import ledger as ledger_module
from smart_research_agent.failure_ledger import verify
from smart_research_agent.failure_ledger.ledger import Family, Ledger, build_ledger
from smart_research_agent.failure_ledger.types import (
    BASE_KINDS,
    BASE_KIND_DESCRIPTIONS,
    FAILURE_LEDGER_BOUNDARIES,
    FAILURE_LEDGER_NOTES,
    FAILURE_LEDGER_NOTES_ORDER,
)


@dataclass(frozen=True)
class ModuleRow:
    """模块表的一行：一份 ``errors.py``."""

    index: int
    package: str
    families: int
    root: str

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {
            "index": self.index,
            "package": self.package,
            "families": self.families,
            "root": self.root,
        }

    def line(self) -> str:
        """`` 1. arch_variants            |  6 族 | 根 VariantError``."""
        return f"{self.index:>2}. {self.package:<24} | {self.families:>3} 族 | 根 {self.root}"


@dataclass(frozen=True)
class FamilyRow:
    """族表的一行：一个族."""

    index: int
    qualified: str
    kind: str
    bases: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {
            "index": self.index,
            "qualified": self.qualified,
            "kind": self.kind,
            "bases": list(self.bases),
        }

    def line(self) -> str:
        """`` 1. transformer_stack.ShapeError | sub  | CoreShapeError、StackError``."""
        rendered = "、".join(self.bases) or "（无基类）"
        return f"{self.index:>2}. {self.qualified:<34} | {self.kind:<4} | {rendered}"


@dataclass(frozen=True)
class BaseKindRow:
    """归属表的一行：一类基类归属."""

    kind: str
    description: str
    count: int

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {"kind": self.kind, "count": self.count}

    def line(self) -> str:
        """``local     |   152 | local：基类名就在同一个 errors.py 里定义……``."""
        return f"{self.kind:<12} | {self.count:>5} | {self.description}"


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
        """`` 1. ledger_covers_all_error_modules | [equality   ] | 读数 0 ✓ | ……``."""
        mark = "✓" if self.passed else "✗"
        detail = "；".join(self.evidence)
        return (
            f"{self.index:>2}. {self.name:<38} | [{self.criterion:<11}] | "
            f"读数 {self.reading:>8.4g} {mark} | {detail}"
        )


def module_rows(ledger: Ledger | None = None) -> tuple[ModuleRow, ...]:
    """① 模块表：每份 ``errors.py`` 一行（根族名一起印）."""
    resolved = build_ledger() if ledger is None else ledger
    rows: list[ModuleRow] = []
    for index, package in enumerate(resolved.modules, start=1):
        families = resolved.by_package(package)
        roots = [family.name for family in families if family.is_root]
        rows.append(
            ModuleRow(
                index=index,
                package=package,
                families=len(families),
                root=roots[0] if len(roots) == 1 else "、".join(roots) or "（无）",
            )
        )
    return tuple(rows)


def family_rows(ledger: Ledger | None = None, *, limit: int | None = None) -> tuple[FamilyRow, ...]:
    """② 族表：每个族一行（可选截断）."""
    resolved = build_ledger() if ledger is None else ledger
    chosen: tuple[Family, ...] = resolved.families if limit is None else resolved.families[:limit]
    return tuple(
        FamilyRow(
            index=index,
            qualified=family.qualified,
            kind=family.kind,
            bases=tuple(ref.raw for ref in family.bases),
        )
        for index, family in enumerate(chosen, start=1)
    )


def base_kind_rows(ledger: Ledger | None = None) -> tuple[BaseKindRow, ...]:
    """③ 归属表：四类归属各有多少条基类引用（**顺序与 BASE_KINDS 一致**）."""
    resolved = build_ledger() if ledger is None else ledger
    counts = {kind: 0 for kind in BASE_KINDS}
    for family in resolved.families:
        for ref in family.bases:
            counts[ref.kind] = counts.get(ref.kind, 0) + 1
    return tuple(
        BaseKindRow(kind=kind, description=BASE_KIND_DESCRIPTIONS[kind], count=counts[kind])
        for kind in BASE_KINDS
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
    keys = FAILURE_LEDGER_NOTES_ORDER if limit is None else FAILURE_LEDGER_NOTES_ORDER[:limit]
    return tuple(f"{index:>2}. {FAILURE_LEDGER_NOTES[key]}" for index, key in enumerate(keys, start=1))


def boundary_lines() -> tuple[str, ...]:
    """五条边界逐行印出."""
    return tuple(f"- {item}" for item in FAILURE_LEDGER_BOUNDARIES)


def study_lines(
    *,
    ledger: Ledger | None = None,
    report: verify.PropertyReport | None = None,
    family_limit: int = 12,
) -> tuple[str, ...]:
    """一次跑完四张表（演示脚本与教程引用的是同一批读数）.

    四张表共用**同一份台账**，因此它们读的是同一时刻的状态。
    """
    resolved = build_ledger() if ledger is None else ledger
    resolved_report = verify.check_all(ledger=resolved) if report is None else report
    lines: list[str] = []
    lines.append("== 1. 模块表（每份 errors.py：包名 / 族数 / 根族）")
    for row in module_rows(resolved):
        lines.append("  " + row.line())
    lines.append("  " + resolved.line())
    lines.append("== 2. 族表（包.名 / 身份 / 基类，前 %d 行）" % family_limit)
    for row in family_rows(resolved, limit=family_limit):
        lines.append("  " + row.line())
    lines.append("== 3. 归属表（四类归属的基类引用计数）")
    for row in base_kind_rows(resolved):
        lines.append("  " + row.line())
    lines.append("== 4. 性质表（7 条性质：判据类别 / 读数 / 结论）")
    for row in property_rows(resolved_report):
        lines.append("  " + row.line())
    lines.append("  " + ledger_module.ledger_lines(resolved)[-1])
    return tuple(lines)


def to_dict_lines(rows: tuple[Any, ...]) -> tuple[dict[str, Any], ...]:
    """把一串带 ``to_dict`` 的行折成可 JSON 化的字段（报告导出的便利函数）."""
    return tuple(row.to_dict() for row in rows)


__all__ = [
    "BaseKindRow",
    "FamilyRow",
    "ModuleRow",
    "PropertyRow",
    "base_kind_rows",
    "boundary_lines",
    "family_rows",
    "module_rows",
    "note_lines",
    "property_rows",
    "study_lines",
    "to_dict_lines",
]
