"""``study``：五张表（day100 / G2-D1）.

前面各天每天最后都留下几张表，因为**一个数只有被印出来才可能被反驳**。
今天五张表各回答一个"这次交付成不成立"的问题：

```text
① 交付物表   4 份交付物：id / 标题 / 它回答的问题 / 正文行数
② 清单表     全部一级子包：在场 / 公开名字 / 文件 / __all__
③ 自评表     8 项能力：评级 / 承担子包 / 名字数 / 链上阶段
④ 规划表     每个真实缺口一条：类型 / 理由 / 下一步
⑤ 性质表     7 条性质：判据类别 / 读数 / 结论 / 一行证据
```

## 一条纪律：每一行都要带**两个数**

与前面各天的表同源——一行只写"通过"的表是没法反驳的。
因此每张表的每一行都至少有两列：一个**读数**与一个**参照**
（行数 / 名字数 / 评级 / 判据 / 结论）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smart_research_agent.capstone.manifest import Manifest, build_manifest
from smart_research_agent.graduation import verify
from smart_research_agent.graduation.assessment import LEVEL_NAMES, Assessment, assess
from smart_research_agent.graduation.demo import ReplayReport, build_transcript, replay
from smart_research_agent.graduation.inventory import InventoryReport, build_inventory
from smart_research_agent.graduation.roadmap import Roadmap, plan
from smart_research_agent.graduation.summary import CourseSummary, build_summary
from smart_research_agent.graduation.types import (
    GRADUATION_BOUNDARIES,
    GRADUATION_NOTES,
    GRADUATION_NOTES_ORDER,
    LEVEL_DESCRIPTIONS,
    deliverables,
)


@dataclass(frozen=True)
class DeliverableRow:
    """交付物表的一行：一份交付物 + 它的正文规模."""

    index: int
    deliverable: str
    title: str
    line_count: int
    question: str

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {
            "index": self.index,
            "deliverable": self.deliverable,
            "title": self.title,
            "line_count": self.line_count,
            "question": self.question,
        }

    def line(self) -> str:
        """`` 1. demo        | 演示剧本 | 正文  13 行 | 它回答：……``."""
        return (
            f"{self.index:>2}. {self.deliverable:<11} | {self.title:<6} | 正文 {self.line_count:>3} 行"
            f" | 它回答：{self.question}"
        )


@dataclass(frozen=True)
class SubpackageRow:
    """清单表的一行：一个一级子包 + 它的解析结论."""

    index: int
    name: str
    exists: bool
    symbol_count: int
    module_count: int
    has_all: bool

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {
            "index": self.index,
            "name": self.name,
            "exists": self.exists,
            "symbol_count": self.symbol_count,
            "module_count": self.module_count,
            "has_all": self.has_all,
        }

    def line(self) -> str:
        """`` 1. agent            | 在场 | 公开  11 个名字 | 文件  12 | __all__ 无``."""
        state = "在场" if self.exists else "缺失"
        exported = "有" if self.has_all else "无"
        return (
            f"{self.index:>2}. {self.name:<20} | {state} | 公开 {self.symbol_count:>4} 个名字"
            f" | 文件 {self.module_count:>3} | __all__ {exported}"
        )


@dataclass(frozen=True)
class ScoreRow:
    """自评表的一行：一项能力 + 它的评级与证据."""

    index: int
    capability: str
    level: int
    owners_present: int
    owners_total: int
    symbols: int
    stage: str
    stage_ok: bool

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {
            "index": self.index,
            "capability": self.capability,
            "level": self.level,
            "owners_present": self.owners_present,
            "owners_total": self.owners_total,
            "symbols": self.symbols,
            "stage": self.stage,
            "stage_ok": self.stage_ok,
        }

    def line(self) -> str:
        """`` 1. input_guard          | L3 已交付 ✓ | 承担 2/2 | 名字  13 | 阶段 guard ok=True``."""
        mark = "✓" if self.stage_ok else "·"
        return (
            f"{self.index:>2}. {self.capability:<22} | L{self.level} {LEVEL_NAMES[self.level]:<4} {mark}"
            f" | 承担 {self.owners_present}/{self.owners_total} | 名字 {self.symbols:>4}"
            f" | 阶段 {self.stage} ok={self.stage_ok}"
        )


@dataclass(frozen=True)
class RoadmapRow:
    """规划表的一行：一个真实缺口 + 它的下一步."""

    index: int
    key: str
    kind: str
    reason: str
    action: str

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {
            "index": self.index,
            "key": self.key,
            "kind": self.kind,
            "reason": self.reason,
            "action": self.action,
        }

    def line(self) -> str:
        """`` 1. [unclaimed_package ] api      | 理由：…… | 下一步：……``."""
        return (
            f"{self.index:>2}. [{self.kind:<18}] {self.key:<8}"
            f" | 理由：{self.reason} | 下一步：{self.action}"
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
        """`` 1. demo_replays_identically          | [equality   ] | 读数 0 ✓ | ……``."""
        mark = "✓" if self.passed else "✗"
        detail = "；".join(self.evidence)
        return (
            f"{self.index:>2}. {self.name:<36} | [{self.criterion:<11}] | "
            f"读数 {self.reading:>10.6g} {mark} | {detail}"
        )


def deliverable_rows(
    bodies: dict[str, tuple[str, ...]] | None = None,
) -> tuple[DeliverableRow, ...]:
    """① 交付物表：4 份交付物逐行（正文行数来自真实渲染）."""
    resolved = verify.render_deliverables() if bodies is None else bodies
    rows: list[DeliverableRow] = []
    for index, spec in enumerate(deliverables(), start=1):
        lines = resolved.get(spec.id, ())
        rows.append(
            DeliverableRow(
                index=index,
                deliverable=spec.id,
                title=spec.title,
                line_count=len(lines),
                question=spec.question,
            )
        )
    return tuple(rows)


def subpackage_rows(report: InventoryReport | None = None) -> tuple[SubpackageRow, ...]:
    """② 清单表：全部一级子包逐行."""
    resolved = build_inventory() if report is None else report
    return tuple(
        SubpackageRow(
            index=index,
            name=row.name,
            exists=row.exists,
            symbol_count=row.symbol_count,
            module_count=row.module_count,
            has_all=row.has_all,
        )
        for index, row in enumerate(resolved.rows, start=1)
    )


def score_rows(assessment: Assessment | None = None) -> tuple[ScoreRow, ...]:
    """③ 自评表：8 项能力逐行."""
    resolved = assess() if assessment is None else assessment
    return tuple(
        ScoreRow(
            index=index,
            capability=row.capability,
            level=row.level,
            owners_present=row.owners_present,
            owners_total=row.owners_total,
            symbols=row.symbols,
            stage=row.stage,
            stage_ok=row.stage_ok,
        )
        for index, row in enumerate(resolved.rows, start=1)
    )


def roadmap_rows(roadmap: Roadmap | None = None) -> tuple[RoadmapRow, ...]:
    """④ 规划表：每个真实缺口逐行."""
    resolved = plan() if roadmap is None else roadmap
    return tuple(
        RoadmapRow(
            index=index,
            key=item.key,
            kind=item.kind,
            reason=item.reason,
            action=item.action,
        )
        for index, item in enumerate(resolved.items, start=1)
    )


def property_rows(report: verify.PropertyReport | None = None) -> tuple[PropertyRow, ...]:
    """⑤ 性质表：7 条性质逐行（判据类别与读数一起印）."""
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
    keys = GRADUATION_NOTES_ORDER if limit is None else GRADUATION_NOTES_ORDER[:limit]
    return tuple(f"{index:>2}. {GRADUATION_NOTES[key]}" for index, key in enumerate(keys, start=1))


def boundary_lines() -> tuple[str, ...]:
    """五条边界逐行印出."""
    return tuple(f"- {item}" for item in GRADUATION_BOUNDARIES)


def level_lines() -> tuple[str, ...]:
    """4 格量程逐行印出（评级 + 它对应的一条证据）."""
    return tuple(
        f"{level} | {LEVEL_DESCRIPTIONS[level]}" for level in sorted(LEVEL_DESCRIPTIONS)
    )


def study_lines(
    *,
    replay_report: ReplayReport | None = None,
    inventory: InventoryReport | None = None,
    manifest: Manifest | None = None,
    assessment: Assessment | None = None,
    roadmap: Roadmap | None = None,
    summary: CourseSummary | None = None,
    bodies: dict[str, tuple[str, ...]] | None = None,
    property_report: verify.PropertyReport | None = None,
) -> tuple[str, ...]:
    """一次跑完五张表（演示脚本与教程引用的是同一批读数）.

    五张表共用**同一份清单 / 自评 / 规划 / 总结**，因此它们读的是同一时刻的状态。
    """
    resolved_replay = replay() if replay_report is None else replay_report
    resolved_inventory = build_inventory() if inventory is None else inventory
    resolved_manifest = build_manifest() if manifest is None else manifest
    resolved_assessment = assess(resolved_manifest) if assessment is None else assessment
    resolved_roadmap = plan(resolved_manifest, resolved_assessment) if roadmap is None else roadmap
    resolved_summary = build_summary(resolved_inventory) if summary is None else summary
    resolved_bodies = (
        verify.render_deliverables(
            transcript=build_transcript(),
            inventory=resolved_inventory,
            summary=resolved_summary,
            assessment=resolved_assessment,
            roadmap=resolved_roadmap,
        )
        if bodies is None
        else bodies
    )
    report = (
        verify.check_all(
            replay_report=resolved_replay,
            bodies=resolved_bodies,
            inventory=resolved_inventory,
            manifest=resolved_manifest,
            assessment=resolved_assessment,
            roadmap=resolved_roadmap,
            summary=resolved_summary,
        )
        if property_report is None
        else property_report
    )
    lines: list[str] = []
    lines.append("== 1. 交付物表（4 份交付物：正文行数 / 它回答的问题）")
    for row in deliverable_rows(resolved_bodies):
        lines.append("  " + row.line())
    lines.append("== 2. 清单表（全部一级子包：在场 / 公开名字 / 文件）")
    for row in subpackage_rows(resolved_inventory):
        lines.append("  " + row.line())
    lines.append("== 3. 自评表（8 项能力：评级 / 承担子包 / 名字数 / 链上阶段）")
    for row in score_rows(resolved_assessment):
        lines.append("  " + row.line())
    lines.append("== 4. 规划表（每个真实缺口一条：类型 / 理由 / 下一步）")
    for row in roadmap_rows(resolved_roadmap):
        lines.append("  " + row.line())
    lines.append("== 5. 性质表（7 条性质：判据类别 / 读数 / 结论）")
    for row in property_rows(report):
        lines.append("  " + row.line())
    lines.append("  " + resolved_replay.line())
    lines.append("  " + resolved_summary.line())
    return tuple(lines)


def to_dict_lines(rows: tuple[Any, ...]) -> tuple[dict[str, Any], ...]:
    """把一串带 ``to_dict`` 的行折成可 JSON 化的字段（报告导出的便利函数）."""
    return tuple(row.to_dict() for row in rows)


__all__ = [
    "DeliverableRow",
    "PropertyRow",
    "RoadmapRow",
    "ScoreRow",
    "SubpackageRow",
    "boundary_lines",
    "deliverable_rows",
    "level_lines",
    "note_lines",
    "property_rows",
    "roadmap_rows",
    "score_rows",
    "study_lines",
    "subpackage_rows",
    "to_dict_lines",
]
