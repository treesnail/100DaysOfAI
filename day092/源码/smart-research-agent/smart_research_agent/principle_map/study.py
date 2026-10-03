"""``study``：五张表（day088 / M7-D12）.

前十六天每天最后都留下几张表，因为**一个数只有被印出来才可能被反驳**。
今天五张表各回答一个"这张图拼对了吗"的问题：

```text
原理表      十二块拼图：id / 层 / 支撑应用 / 实现落点 / 来源天 / 现场读数
分层覆盖表  四层各几条、分别是哪几条（哪一层空了看得见）
应用覆盖表  六个应用各被几条原理支撑（哪座是孤岛看得见）
证据表      十二条读数 / 期望 / 上界 / 是否满足（两类判据并排印）
提纲表      四节的标题 / demo / 时长（总时长由它加出来）
```

## 一条纪律：每一行都要带**两个数**

与 day082 的探针同源——一行只写"通过"的表是没法反驳的。
因此每张表的每一行都至少有两列：**读数**与**参照**
（落点 / 计数 / 期望 / 上界 / 时长）。
"""

from __future__ import annotations

from dataclasses import dataclass

from smart_research_agent.principle_map import claims, graph as graph_module, outline as outline_module
from smart_research_agent.principle_map.types import (
    APPLICATIONS,
    LAYERS,
    PRINCIPLE_NOTES,
    PRINCIPLE_NOTES_ORDER,
    TalkSection,
)


@dataclass(frozen=True)
class PrincipleRow:
    """原理表的一行：一块拼图 + 它的现场读数."""

    index: int
    principle_id: str
    layer: str
    application: str
    artifact: str
    source_day: str
    reading: float
    passed: bool

    def line(self) -> str:
        """`` 1. [math] attention_is_differentiable_retrieval -> agent_reasoning | 落点 ... | 读数 0.0 ✓``."""
        mark = "✓" if self.passed else "✗"
        return (
            f"{self.index:>2}. [{self.layer:<14}] {self.principle_id:<48} → "
            f"{self.application:<16} | {self.source_day:<10} | 读数 {self.reading:g} {mark}"
        )


@dataclass(frozen=True)
class LayerRow:
    """分层覆盖表的一行：一个层几条原理."""

    layer: str
    count: int
    members: tuple[str, ...]

    def line(self) -> str:
        """``[math]          3 条：a、b、c``."""
        return f"[{self.layer:<14}] {self.count} 条：{'、'.join(self.members) if self.members else '（空）'}"


@dataclass(frozen=True)
class ApplicationRow:
    """应用覆盖表的一行：一个应用被几条原理支撑."""

    application: str
    count: int
    members: tuple[str, ...]

    @property
    def supported(self) -> bool:
        """是否至少被一条原理支撑."""
        return self.count > 0

    def line(self) -> str:
        """``serving          4 条 | a、b、c、d``."""
        mark = "✓" if self.supported else "✗"
        return (
            f"{self.application:<16} {self.count} 条 {mark} | "
            f"{'、'.join(self.members) if self.members else '（没有入边）'}"
        )


@dataclass(frozen=True)
class EvidenceRow:
    """证据表的一行：一条读数 + 判据."""

    principle_id: str
    reading: float
    expected: float
    upper_bound: float | None
    passed: bool
    note: str

    def line(self) -> str:
        """``attention_rows_are_distributions | 读数 3.3e-16 ≤ 上界 1e-09 ✓``."""
        mark = "✓" if self.passed else "✗"
        if self.upper_bound is not None:
            criterion = f"读数 {self.reading:.3e} ≤ 上界 {self.upper_bound:.3e}"
        else:
            criterion = f"读数 {self.reading:g} == 期望 {self.expected:g}"
        return f"{self.principle_id:<48} | {criterion} {mark}"


@dataclass(frozen=True)
class OutlineRow:
    """提纲表的一行：一节分享."""

    section: TalkSection

    def line(self) -> str:
        """``第 1 节 [math] 标题...（8 分钟）| demo：python scripts/...``."""
        return f"{self.section.line()} | demo：{self.section.demo}"


def principle_rows(graph: graph_module.PrincipleGraph | None = None) -> tuple[PrincipleRow, ...]:
    """原理表：十二块拼图逐行（读数来自探针）."""
    resolved = graph or graph_module._default_graph()
    rows: list[PrincipleRow] = []
    for index, item in enumerate(resolved.principles, start=1):
        evidence = resolved.evidence_of(item.id)
        rows.append(
            PrincipleRow(
                index=index,
                principle_id=item.id,
                layer=item.layer,
                application=item.application,
                artifact=item.artifact,
                source_day=item.source_day,
                reading=evidence.reading,
                passed=evidence.passed,
            )
        )
    return tuple(rows)


def layer_rows(graph: graph_module.PrincipleGraph | None = None) -> tuple[LayerRow, ...]:
    """分层覆盖表：四个层各一行（层空了也印）."""
    resolved = graph or graph_module._default_graph()
    return tuple(
        LayerRow(
            layer=layer,
            count=len(claims.by_layer(layer)),
            members=tuple(item.id for item in claims.by_layer(layer)),
        )
        for layer in LAYERS
    )


def application_rows(graph: graph_module.PrincipleGraph | None = None) -> tuple[ApplicationRow, ...]:
    """应用覆盖表：六个应用各一行（没有入边也印）."""
    resolved = graph or graph_module._default_graph()
    return tuple(
        ApplicationRow(
            application=app,
            count=len(resolved.application(app).principles),
            members=resolved.application(app).principles,
        )
        for app in APPLICATIONS
    )


def evidence_rows(graph: graph_module.PrincipleGraph | None = None) -> tuple[EvidenceRow, ...]:
    """证据表：十二条读数逐行（含上界那一列）."""
    resolved = graph or graph_module._default_graph()
    return tuple(
        EvidenceRow(
            principle_id=item.principle,
            reading=item.reading,
            expected=item.expected,
            upper_bound=item.upper_bound,
            passed=item.passed,
            note=item.note,
        )
        for item in resolved.evidence
    )


def outline_rows(sections: tuple[TalkSection, ...] | None = None) -> tuple[OutlineRow, ...]:
    """提纲表：四节逐行."""
    resolved = outline_module.build_outline() if sections is None else sections
    return tuple(OutlineRow(section=section) for section in resolved)


def note_lines(limit: int | None = None) -> tuple[str, ...]:
    """十条笔记逐行印出（顺序即写入顺序）."""
    keys = PRINCIPLE_NOTES_ORDER if limit is None else PRINCIPLE_NOTES_ORDER[:limit]
    return tuple(f"{index:>2}. {PRINCIPLE_NOTES[key]}" for index, key in enumerate(keys, start=1))


def study_lines(graph: graph_module.PrincipleGraph | None = None) -> tuple[str, ...]:
    """一次跑完五张表（演示脚本与教程引用的是同一批读数）."""
    resolved = graph or graph_module._default_graph()
    lines: list[str] = []
    lines.append("== 1. 原理表（十二块拼图：层 / 应用 / 落点 / 来源 / 读数）")
    for row in principle_rows(resolved):
        lines.append("  " + row.line())
    lines.append("== 2. 分层覆盖表（四层各几条）")
    for row in layer_rows(resolved):
        lines.append("  " + row.line())
    lines.append("== 3. 应用覆盖表（六个应用各被几条原理支撑）")
    for row in application_rows(resolved):
        lines.append("  " + row.line())
    lines.append("== 4. 证据表（读数 / 期望 / 上界）")
    for row in evidence_rows(resolved):
        lines.append("  " + row.line())
    lines.append("== 5. 提纲表（四节：标题 / demo / 时长）")
    for row in outline_rows():
        lines.append("  " + row.line())
    lines.append(f"  总时长 {outline_module.outline_minutes(outline_module.build_outline())} 分钟")
    return tuple(lines)


__all__ = [
    "ApplicationRow",
    "EvidenceRow",
    "LayerRow",
    "OutlineRow",
    "PrincipleRow",
    "application_rows",
    "evidence_rows",
    "layer_rows",
    "note_lines",
    "outline_rows",
    "principle_rows",
    "study_lines",
]
