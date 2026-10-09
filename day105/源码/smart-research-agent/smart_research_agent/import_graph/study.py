"""``study``：四张表（day103）.

**一个数只有被印出来才可能被反驳**。今天四张表各回答一个"这张依赖图成不成立"的问题：

```text
① 模块表   每个模块：它属于哪个子包 / 出度 / 入度
② 环表     每个强连通分量（>1 的那些才是环）：成员 / 规模
③ 边表     若干条边：源 → 目标 | 种类
④ 性质表   7 条性质：判据类别 / 读数 / 结论 / 一行证据
```

## 一条纪律：每一行都要带**两个数**

与前面各天的表同源——一行只写"通过"的表是没法反驳的。
因此每一行的"读数"旁边都跟着一个"参照"：出度 / 入度 / 规模 / 边种类 / 判据与期望。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smart_research_agent.import_graph import graph as graph_module
from smart_research_agent.import_graph import verify
from smart_research_agent.import_graph.graph import Edge, ModuleGraph, build_graph
from smart_research_agent.import_graph.types import (
    IMPORT_GRAPH_BOUNDARIES,
    IMPORT_GRAPH_NOTES,
    IMPORT_GRAPH_NOTES_ORDER,
)


@dataclass(frozen=True)
class ModuleRow:
    """模块表的一行：一个模块."""

    index: int
    module: str
    package: str
    out_degree: int
    in_degree: int

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {
            "index": self.index,
            "module": self.module,
            "package": self.package,
            "out_degree": self.out_degree,
            "in_degree": self.in_degree,
        }

    def line(self) -> str:
        """`` 1. smart_research_agent.import_graph.graph | graph | 出 2 入 1``."""
        return (
            f"{self.index:>2}. {self.module:<56} | {self.package:<20} | "
            f"出 {self.out_degree:>3} 入 {self.in_degree:>3}"
        )


@dataclass(frozen=True)
class CycleRow:
    """环表的一行：一个强连通分量."""

    index: int
    key: str
    size: int
    members: tuple[str, ...]

    @property
    def is_cycle(self) -> bool:
        """成员数 > 1 的才是真正的环."""
        return self.size > 1

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {"index": self.index, "key": self.key, "size": self.size, "members": list(self.members)}

    def line(self) -> str:
        """`` 1. [环 ] api.app | 2 个模块 | api.app、tools.image_analysis``."""
        mark = "环 " if self.is_cycle else "单点"
        return f"{self.index:>2}. [{mark}] {self.key:<50} | {self.size:>2} 个模块 | {'、'.join(self.members)}"


@dataclass(frozen=True)
class EdgeRow:
    """边表的一行：一条依赖边."""

    index: int
    source: str
    target: str
    kind: str

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {"index": self.index, "source": self.source, "target": self.target, "kind": self.kind}

    def line(self) -> str:
        """`` 1. api.app → tools.image_analysis | absolute``."""
        return f"{self.index:>2}. {self.source} → {self.target} | {self.kind}"


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
        """`` 1. graph_covers_all_modules | [equality   ] | 读数 0 ✓ | ……``."""
        mark = "✓" if self.passed else "✗"
        detail = "；".join(self.evidence)
        return (
            f"{self.index:>2}. {self.name:<34} | [{self.criterion:<11}] | "
            f"读数 {self.reading:>8.4g} {mark} | {detail}"
        )


def module_rows(graph: ModuleGraph | None = None, *, limit: int | None = None) -> tuple[ModuleRow, ...]:
    """① 模块表：每个模块一行（可选截断）."""
    resolved = build_graph() if graph is None else graph
    chosen = resolved.nodes if limit is None else resolved.nodes[:limit]
    return tuple(
        ModuleRow(
            index=index,
            module=node,
            package=resolved.package_of(node),
            out_degree=len(resolved.successors(node)),
            in_degree=len(resolved.predecessors(node)),
        )
        for index, node in enumerate(chosen, start=1)
    )


def cycle_rows(graph: ModuleGraph | None = None, *, cycles_only: bool = False) -> tuple[CycleRow, ...]:
    """② 环表：每个强连通分量一行（``cycles_only`` 只留 >1 的那些）."""
    resolved = build_graph() if graph is None else graph
    components = resolved.cycles() if cycles_only else resolved.sccs()
    return tuple(
        CycleRow(index=index, key=component[0], size=len(component), members=component)
        for index, component in enumerate(components, start=1)
    )


def edge_rows(graph: ModuleGraph | None = None, *, limit: int | None = None) -> tuple[EdgeRow, ...]:
    """③ 边表：若干条边（默认全部；可截断）."""
    resolved = build_graph() if graph is None else graph
    chosen: tuple[Edge, ...] = resolved.edges if limit is None else resolved.edges[:limit]
    return tuple(
        EdgeRow(index=index, source=edge.source, target=edge.target, kind=edge.kind)
        for index, edge in enumerate(chosen, start=1)
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
    keys = IMPORT_GRAPH_NOTES_ORDER if limit is None else IMPORT_GRAPH_NOTES_ORDER[:limit]
    return tuple(f"{index:>2}. {IMPORT_GRAPH_NOTES[key]}" for index, key in enumerate(keys, start=1))


def boundary_lines() -> tuple[str, ...]:
    """五条边界逐行印出."""
    return tuple(f"- {item}" for item in IMPORT_GRAPH_BOUNDARIES)


def study_lines(
    *,
    graph: ModuleGraph | None = None,
    report: verify.PropertyReport | None = None,
    module_limit: int = 12,
    edge_limit: int = 12,
) -> tuple[str, ...]:
    """一次跑完四张表（演示脚本与教程引用的是同一批读数）.

    四张表共用**同一张图**，因此它们读的是同一时刻的状态。
    """
    resolved = build_graph() if graph is None else graph
    resolved_report = verify.check_all(graph=resolved) if report is None else report
    lines: list[str] = []
    lines.append("== 1. 模块表（模块 / 子包 / 出度 / 入度，前 %d 行）" % module_limit)
    for row in module_rows(resolved, limit=module_limit):
        lines.append("  " + row.line())
    lines.append("  " + resolved.line())
    lines.append("== 2. 环表（强连通分量：成员 / 规模）")
    for row in cycle_rows(resolved, cycles_only=True):
        lines.append("  " + row.line())
    lines.append(f"  真正的环 {len(resolved.cycles())} 个 / 分量 {len(resolved.sccs())} 个")
    lines.append("== 3. 边表（源 → 目标 | 种类，前 %d 行）" % edge_limit)
    for row in edge_rows(resolved, limit=edge_limit):
        lines.append("  " + row.line())
    lines.append("== 4. 性质表（7 条性质：判据类别 / 读数 / 结论）")
    for row in property_rows(resolved_report):
        lines.append("  " + row.line())
    lines.append("  " + graph_module.graph_lines(resolved, limit=1)[-1])
    return tuple(lines)


def to_dict_lines(rows: tuple[Any, ...]) -> tuple[dict[str, Any], ...]:
    """把一串带 ``to_dict`` 的行折成可 JSON 化的字段（报告导出的便利函数）."""
    return tuple(row.to_dict() for row in rows)


__all__ = [
    "CycleRow",
    "EdgeRow",
    "ModuleRow",
    "PropertyRow",
    "boundary_lines",
    "cycle_rows",
    "edge_rows",
    "module_rows",
    "note_lines",
    "property_rows",
    "study_lines",
    "to_dict_lines",
]
