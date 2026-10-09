"""``study``：四张表（day105）.

**一个数只有被印出来才可能被反驳**。今天四张表各回答一个"这份常量台账成不成立"的问题：

```text
① 模块表   每个模块：属于哪个子包 / 常量数 / 可取值数 / 读不出数 / 重名数
② 同名表   出现在两个以上模块里的常量名：关系 / 成员数 / 取值
③ 冲突表   同名不同值的那几组（本课最值钱的读数）：逐行点名
④ 性质表   7 条性质：判据类别 / 读数 / 结论 / 一行证据
```

## 一条纪律：每一行都要带**两个数**

与前面各天的表同源——一行只写"通过"的表是没法反驳的。
因此每一行的"读数"旁边都跟着一个"参照"：常量数 / 可取值数 / 出现模块数 / 判据与期望。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smart_research_agent.constant_ledger import verify
from smart_research_agent.constant_ledger.ledger import ConstantLedger, build_ledger
from smart_research_agent.constant_ledger.types import (
    CONSTANT_LEDGER_BOUNDARIES,
    CONSTANT_LEDGER_NOTES,
    CONSTANT_LEDGER_NOTES_ORDER,
    RELATION_CONFLICT,
    RELATION_CONSISTENT,
    RELATION_INCOMPARABLE,
    RELATION_UNIQUE,
    VALUE_LITERAL,
)


@dataclass(frozen=True)
class ModuleRow:
    """模块表的一行：一个模块."""

    index: int
    module: str
    package: str
    constants: int
    literal: int
    opaque: int
    duplicates: int

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {
            "index": self.index,
            "module": self.module,
            "package": self.package,
            "constants": self.constants,
            "literal": self.literal,
            "opaque": self.opaque,
            "duplicates": self.duplicates,
        }

    def line(self) -> str:
        """`` 1. smart_research_agent.symbol_catalog.types | symbol_catalog | 常量 41 | 可取值 40 | 读不出 1``."""
        return (
            f"{self.index:>2}. {self.module:<60} | {self.package:<16} | 常量 {self.constants:>3}"
            f" | 可取值 {self.literal:>3} | 读不出 {self.opaque:>3} | 重名 {self.duplicates:>2}"
        )


@dataclass(frozen=True)
class NameRow:
    """同名表的一行：一个跨模块同名的常量名."""

    index: int
    name: str
    relation: str
    count: int
    values: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {
            "index": self.index,
            "name": self.name,
            "relation": self.relation,
            "count": self.count,
            "values": list(self.values),
        }

    def line(self) -> str:
        """`` 1. DEFAULT_EPSILON | conflict | 3 个模块 | 取值 ['1e-05', '1e-08']``."""
        shown = "、".join(self.values) or "（读不出来）"
        return (
            f"{self.index:>2}. {self.name:<34} | {self.relation:<12} | {self.count:>3} 个模块"
            f" | 取值 [{shown}]"
        )


@dataclass(frozen=True)
class ConflictRow:
    """冲突表的一行：一组"同名不同值"，逐成员点名「模块 = 取值」."""

    index: int
    name: str
    pairs: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {"index": self.index, "name": self.name, "pairs": [list(pair) for pair in self.pairs]}

    def line(self) -> str:
        """`` 1. DEFAULT_EPSILON | a.types = 1e-05、b.types = 1e-08``."""
        shown = "、".join(f"{module} = {value}" for module, value in self.pairs)
        return f"{self.index:>2}. {self.name:<34} | {shown}"


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
        """`` 1. ledger_covers_all_modules | [equality   ] | 读数 0 ✓ | ……``."""
        mark = "✓" if self.passed else "✗"
        detail = "；".join(self.evidence)
        return (
            f"{self.index:>2}. {self.name:<40} | [{self.criterion:<11}] | "
            f"读数 {self.reading:>8.4g} {mark} | {detail}"
        )


def module_rows(
    ledger: ConstantLedger | None = None, *, limit: int | None = None
) -> tuple[ModuleRow, ...]:
    """① 模块表：每个模块一行（可选截断）."""
    resolved = build_ledger() if ledger is None else ledger
    chosen = resolved.entries if limit is None else resolved.entries[:limit]
    rows: list[ModuleRow] = []
    for index, entry in enumerate(chosen, start=1):
        primary = entry.primary_defs()
        literal = sum(1 for item in primary if item.value_kind == VALUE_LITERAL)
        rows.append(
            ModuleRow(
                index=index,
                module=entry.module,
                package=resolved.package_of(entry.module),
                constants=len(primary),
                literal=literal,
                opaque=len(primary) - literal,
                duplicates=len(entry.duplicate_names()),
            )
        )
    return tuple(rows)


def shared_rows(
    ledger: ConstantLedger | None = None, *, limit: int | None = None
) -> tuple[NameRow, ...]:
    """② 同名表：跨模块同名的常量名（按出现模块数降序，可截断）."""
    resolved = build_ledger() if ledger is None else ledger
    groups = resolved.shared_groups()
    chosen = groups if limit is None else groups[:limit]
    return tuple(
        NameRow(
            index=index,
            name=group.name,
            relation=group.relation,
            count=len(group.module_names),
            values=group.literals(),
        )
        for index, group in enumerate(chosen, start=1)
    )


def conflict_rows(ledger: ConstantLedger | None = None) -> tuple[ConflictRow, ...]:
    """③ 冲突表：同名不同值的那几组（本课最值钱的读数）."""
    resolved = build_ledger() if ledger is None else ledger
    return tuple(
        ConflictRow(
            index=index,
            name=group.name,
            pairs=tuple(
                (item.module, item.literal if item.literal is not None else "（读不出来）")
                for item in group.members
            ),
        )
        for index, group in enumerate(resolved.conflicts(), start=1)
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


def relation_lines(ledger: ConstantLedger | None = None) -> tuple[str, ...]:
    """四类同名关系各几组（一行一个关系）."""
    resolved = build_ledger() if ledger is None else ledger
    every = resolved.name_groups()
    table = {
        RELATION_UNIQUE: sum(1 for group in every if group.relation == RELATION_UNIQUE),
        RELATION_CONSISTENT: sum(1 for group in every if group.relation == RELATION_CONSISTENT),
        RELATION_CONFLICT: sum(1 for group in every if group.relation == RELATION_CONFLICT),
        RELATION_INCOMPARABLE: sum(
            1 for group in every if group.relation == RELATION_INCOMPARABLE
        ),
    }
    return tuple(f"{relation}：{table[relation]} 组" for relation in table)


def value_kind_lines(ledger: ConstantLedger | None = None) -> tuple[str, ...]:
    """两类取值形态各有几条（一行一个形态）."""
    resolved = build_ledger() if ledger is None else ledger
    return tuple(f"{kind}：{count} 条" for kind, count in resolved.value_kind_counts())


def note_lines(limit: int | None = None) -> tuple[str, ...]:
    """十条笔记逐行印出（顺序即写入顺序）."""
    keys = CONSTANT_LEDGER_NOTES_ORDER if limit is None else CONSTANT_LEDGER_NOTES_ORDER[:limit]
    return tuple(f"{index:>2}. {CONSTANT_LEDGER_NOTES[key]}" for index, key in enumerate(keys, start=1))


def boundary_lines() -> tuple[str, ...]:
    """五条边界逐行印出."""
    return tuple(f"- {item}" for item in CONSTANT_LEDGER_BOUNDARIES)


def study_lines(
    *,
    ledger: ConstantLedger | None = None,
    report: verify.PropertyReport | None = None,
    module_limit: int = 12,
    shared_limit: int = 12,
    conflict_limit: int = 12,
) -> tuple[str, ...]:
    """一次跑完四张表（演示脚本与教程引用的是同一批读数）.

    四张表共用**同一份台账**，因此它们读的是同一时刻的状态。
    """
    resolved = build_ledger() if ledger is None else ledger
    resolved_report = verify.check_all(ledger=resolved) if report is None else report
    lines: list[str] = []
    lines.append("== 1. 模块表（模块 / 子包 / 常量 / 可取值 / 读不出 / 重名，前 %d 行）" % module_limit)
    for row in module_rows(resolved, limit=module_limit):
        lines.append("  " + row.line())
    lines.append("  " + resolved.line())
    lines.append("== 2. 同名表（跨模块同名的常量名，前 %d 行）" % shared_limit)
    for row in shared_rows(resolved, limit=shared_limit):
        lines.append("  " + row.line())
    lines.append(f"  跨模块同名 {len(resolved.shared_groups())} 组 / 常量总数 {resolved.constant_count()}")
    lines.append("== 3. 冲突表（同名不同值，前 %d 行）" % conflict_limit)
    for row in conflict_rows(resolved)[:conflict_limit]:
        lines.append("  " + row.line())
    lines.append(f"  冲突 {len(resolved.conflicts())} 组 / 无法比较 {len(resolved.incomparables())} 组")
    lines.append("== 4. 性质表（7 条性质：判据类别 / 读数 / 结论）")
    for row in property_rows(resolved_report):
        lines.append("  " + row.line())
    return tuple(lines)


def to_dict_lines(rows: tuple[Any, ...]) -> tuple[dict[str, Any], ...]:
    """把一串带 ``to_dict`` 的行折成可 JSON 化的字段（报告导出的便利函数）."""
    return tuple(row.to_dict() for row in rows)


__all__ = [
    "ConflictRow",
    "ModuleRow",
    "NameRow",
    "PropertyRow",
    "boundary_lines",
    "conflict_rows",
    "module_rows",
    "note_lines",
    "property_rows",
    "relation_lines",
    "shared_rows",
    "study_lines",
    "to_dict_lines",
    "value_kind_lines",
]
