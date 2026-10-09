"""``study``：四张表（day104）.

**一个数只有被印出来才可能被反驳**。今天四张表各回答一个"这份符号清单成不成立"的问题：

```text
① 模块表   每个模块：属于哪个子包 / 承诺来源 / 承诺数 / 落点数
② 幽灵表   承诺了却找不到落点的名字（本仓库实测 1 个）
③ 重名表   被两个以上模块导出的名字（跨包契约的入口）
④ 性质表   7 条性质：判据类别 / 读数 / 结论 / 一行证据
```

## 一条纪律：每一行都要带**两个数**

与前面各天的表同源——一行只写"通过"的表是没法反驳的。
因此每一行的"读数"旁边都跟着一个"参照"：承诺来源 / 落点数 / 出现模块数 / 判据与期望。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smart_research_agent.symbol_catalog import verify
from smart_research_agent.symbol_catalog.catalog import SymbolCatalog, build_catalog
from smart_research_agent.symbol_catalog.types import (
    SYMBOL_CATALOG_BOUNDARIES,
    SYMBOL_CATALOG_NOTES,
    SYMBOL_CATALOG_NOTES_ORDER,
)


@dataclass(frozen=True)
class ModuleRow:
    """模块表的一行：一个模块."""

    index: int
    module: str
    package: str
    source: str
    promises: int
    bound: int
    phantoms: int

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {
            "index": self.index,
            "module": self.module,
            "package": self.package,
            "source": self.source,
            "promises": self.promises,
            "bound": self.bound,
            "phantoms": self.phantoms,
        }

    def line(self) -> str:
        """`` 1. smart_research_agent.symbol_catalog.catalog | symbol_catalog | declared | 承诺 10 落点 41``."""
        return (
            f"{self.index:>2}. {self.module:<58} | {self.package:<16} | {self.source:<8} | "
            f"承诺 {self.promises:>3} 落点 {self.bound:>3} 幽灵 {self.phantoms:>2}"
        )


@dataclass(frozen=True)
class PhantomRow:
    """幽灵表的一行：一个"承诺了却没有落点"的名字."""

    index: int
    module: str
    name: str

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {"index": self.index, "module": self.module, "name": self.name}

    def line(self) -> str:
        """`` 1. transformer_stack.verify | STAGE_ITEMS``."""
        return f"{self.index:>2}. {self.module} | {self.name}"


@dataclass(frozen=True)
class SharedRow:
    """重名表的一行：一个被多个模块导出的名字."""

    index: int
    name: str
    count: int
    modules: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {
            "index": self.index,
            "name": self.name,
            "count": self.count,
            "modules": list(self.modules),
        }

    def line(self) -> str:
        """`` 1. NumericError | 39 个模块 | alignment.types、attention.types、……``."""
        preview = "、".join(self.modules[:3])
        suffix = "、…" if len(self.modules) > 3 else ""
        return f"{self.index:>2}. {self.name:<30} | {self.count:>3} 个模块 | {preview}{suffix}"


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
        """`` 1. catalog_covers_all_modules | [equality   ] | 读数 0 ✓ | ……``."""
        mark = "✓" if self.passed else "✗"
        detail = "；".join(self.evidence)
        return (
            f"{self.index:>2}. {self.name:<40} | [{self.criterion:<11}] | "
            f"读数 {self.reading:>8.4g} {mark} | {detail}"
        )


def module_rows(catalog: SymbolCatalog | None = None, *, limit: int | None = None) -> tuple[ModuleRow, ...]:
    """① 模块表：每个模块一行（可选截断）."""
    resolved = build_catalog() if catalog is None else catalog
    chosen = resolved.entries if limit is None else resolved.entries[:limit]
    return tuple(
        ModuleRow(
            index=index,
            module=entry.module,
            package=resolved.package_of(entry.module),
            source=entry.source,
            promises=len(entry.exports()),
            bound=len(entry.bound_names),
            phantoms=len(entry.phantoms()),
        )
        for index, entry in enumerate(chosen, start=1)
    )


def phantom_rows(catalog: SymbolCatalog | None = None) -> tuple[PhantomRow, ...]:
    """② 幽灵表：承诺了却没有落点的名字（本仓库实测 1 个）."""
    resolved = build_catalog() if catalog is None else catalog
    return tuple(
        PhantomRow(index=index, module=module, name=name)
        for index, (module, name) in enumerate(resolved.phantom_pairs(), start=1)
    )


def shared_rows(catalog: SymbolCatalog | None = None, *, limit: int | None = None) -> tuple[SharedRow, ...]:
    """③ 重名表：被两个以上模块导出的名字（按出现模块数降序，可截断）."""
    resolved = build_catalog() if catalog is None else catalog
    shared = resolved.shared_names()
    chosen = shared if limit is None else shared[:limit]
    return tuple(
        SharedRow(index=index, name=name, count=len(modules), modules=modules)
        for index, (name, modules) in enumerate(chosen, start=1)
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
    keys = SYMBOL_CATALOG_NOTES_ORDER if limit is None else SYMBOL_CATALOG_NOTES_ORDER[:limit]
    return tuple(f"{index:>2}. {SYMBOL_CATALOG_NOTES[key]}" for index, key in enumerate(keys, start=1))


def boundary_lines() -> tuple[str, ...]:
    """五条边界逐行印出."""
    return tuple(f"- {item}" for item in SYMBOL_CATALOG_BOUNDARIES)


def study_lines(
    *,
    catalog: SymbolCatalog | None = None,
    report: verify.PropertyReport | None = None,
    module_limit: int = 12,
    shared_limit: int = 12,
) -> tuple[str, ...]:
    """一次跑完四张表（演示脚本与教程引用的是同一批读数）.

    四张表共用**同一份清单**，因此它们读的是同一时刻的状态。
    """
    resolved = build_catalog() if catalog is None else catalog
    resolved_report = verify.check_all(catalog=resolved) if report is None else report
    lines: list[str] = []
    lines.append("== 1. 模块表（模块 / 子包 / 来源 / 承诺 / 落点 / 幽灵，前 %d 行）" % module_limit)
    for row in module_rows(resolved, limit=module_limit):
        lines.append("  " + row.line())
    lines.append("  " + resolved.line())
    lines.append("== 2. 幽灵表（承诺了却找不到落点的名字）")
    for row in phantom_rows(resolved):
        lines.append("  " + row.line())
    lines.append(f"  幽灵 {len(resolved.phantom_pairs())} 个 / 字面量承诺的模块 {resolved.declared_count()} 个")
    lines.append("== 3. 重名表（被两个以上模块导出的名字，前 %d 行）" % shared_limit)
    for row in shared_rows(resolved, limit=shared_limit):
        lines.append("  " + row.line())
    lines.append(f"  跨模块重名 {len(resolved.shared_names())} 个 / 承诺总数 {resolved.promise_count()}")
    lines.append("== 4. 性质表（7 条性质：判据类别 / 读数 / 结论）")
    for row in property_rows(resolved_report):
        lines.append("  " + row.line())
    return tuple(lines)


def source_lines(catalog: SymbolCatalog | None = None) -> tuple[str, ...]:
    """三种承诺来源各几个模块（一行一个来源）."""
    resolved = build_catalog() if catalog is None else catalog
    return tuple(f"{source}：{count} 个模块" for source, count in resolved.source_counts())


def to_dict_lines(rows: tuple[Any, ...]) -> tuple[dict[str, Any], ...]:
    """把一串带 ``to_dict`` 的行折成可 JSON 化的字段（报告导出的便利函数）."""
    return tuple(row.to_dict() for row in rows)


__all__ = [
    "ModuleRow",
    "PhantomRow",
    "PropertyRow",
    "SharedRow",
    "boundary_lines",
    "module_rows",
    "note_lines",
    "phantom_rows",
    "property_rows",
    "shared_rows",
    "source_lines",
    "study_lines",
    "to_dict_lines",
]
