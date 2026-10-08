"""``study``：五张表（day099 / G1-D1）.

前面各天每天最后都留下几张表，因为**一个数只有被印出来才可能被反驳**。
今天五张表各回答一个"这条链装对了吗"的问题：

```text
① 能力表    8 项能力：id / 标题 / 最近可见阶段 / 承担子包 / 覆没覆上 / 子包符号数
② 清单表    12 个候选子包：在场 / 公开符号数 / 被谁认领（或无人认领）
③ 阶段表    9 段端到端链：阶段 / 承担能力 / 通过 / 读数 / 一行证据
④ 性质表    7 条性质：判据类别 / 读数 / 结论 / 一行证据
⑤ 文档表    2 份渲染文档：字符数 / 覆盖了多少项能力 / 缺了什么
```

## 一条纪律：每一行都要带**两个数**

与前面各天的表同源——一行只写"通过"的表是没法反驳的。
因此每张表的每一行都至少有两列：一个**读数**与一个**参照**
（覆盖与否 / 符号数 / 判据 / 期望 / 字符数）。
"""

from __future__ import annotations

from dataclasses import dataclass

from smart_research_agent.capstone import (
    adapters,
    assembly,
    document,
    manifest as manifest_module,
    verify,
)
from smart_research_agent.capstone.manifest import Manifest, build_manifest
from smart_research_agent.capstone.types import (
    CAPABILITY_ORDER,
    CANDIDATE_SUBPACKAGES,
    capabilities,
)


@dataclass(frozen=True)
class CapabilityRow:
    """能力表的一行：一项能力 + 它的覆盖结论."""

    index: int
    capability: str
    title: str
    stage: str
    owners: tuple[str, ...]
    covered: bool
    symbol_counts: tuple[int, ...]

    def line(self) -> str:
        """`` 1. input_guard       | 输入侧护栏 | [guard] | security、llm | ✓ | 8/5``."""
        mark = "✓" if self.covered else "✗"
        counts = "/".join(str(count) for count in self.symbol_counts)
        return (
            f"{self.index:>2}. {self.capability:<22} | {self.title:<6} | "
            f"[{self.stage:<8}] | {'、'.join(self.owners):<22} | {mark} | {counts}"
        )


@dataclass(frozen=True)
class ManifestRow:
    """清单表的一行：一个候选子包 + 它的解析结论."""

    name: str
    exists: bool
    symbol_count: int
    claimed_by: tuple[str, ...]

    @property
    def claimed(self) -> bool:
        """是否被至少一项能力认领."""
        return bool(self.claimed_by)

    def line(self) -> str:
        """``security     | 在场 | 公开  8 个符号 | 认领：input_guard``."""
        state = "在场" if self.exists else "缺失"
        role = "、".join(self.claimed_by) if self.claimed_by else "（无人认领）"
        return f"{self.name:<12} | {state} | 公开 {self.symbol_count:>3} 个符号 | 认领：{role}"


@dataclass(frozen=True)
class StageRow:
    """阶段表的一行：一段端到端链."""

    index: int
    stage: str
    capability: str
    ok: bool
    reading: float
    detail: str

    def line(self) -> str:
        """`` 1. guard    | input_guard        | ✓ | 读数 0 | 护栏：放行……``."""
        mark = "✓" if self.ok else "✗"
        return (
            f"{self.index:>2}. {self.stage:<8} | {self.capability:<22} | {mark} | "
            f"读数 {self.reading:>10.6g} | {self.detail}"
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

    def line(self) -> str:
        """`` 1. capabilities_are_covered           | [equality]    | 读数 8 == 8 | ✓``."""
        mark = "✓" if self.passed else "✗"
        detail = "；".join(self.evidence)
        return (
            f"{self.index:>2}. {self.name:<40} | [{self.criterion:<11}] | "
            f"读数 {self.reading:>10.6g} {mark} | {detail}"
        )


@dataclass(frozen=True)
class DocumentRow:
    """文档表的一行：一份渲染文档."""

    name: str
    chars: int
    covered: int
    missing: tuple[str, ...]

    def line(self) -> str:
        """``readme         | 1234 字符 | 覆盖 8/8 项能力 | 缺：（无）``."""
        missing = "、".join(self.missing) if self.missing else "（无）"
        return (
            f"{self.name:<14} | {self.chars:>5} 字符 | 覆盖 {self.covered}/"
            f"{len(CAPABILITY_ORDER)} 项能力 | 缺：{missing}"
        )


def capability_rows(manifest: Manifest | None = None) -> tuple[CapabilityRow, ...]:
    """① 能力表：8 项能力逐行（覆盖结论来自清单）."""
    resolved = build_manifest() if manifest is None else manifest
    rows: list[CapabilityRow] = []
    for index, cap in enumerate(capabilities(), start=1):
        coverage = resolved.coverage_of(cap.id)
        rows.append(
            CapabilityRow(
                index=index,
                capability=cap.id,
                title=cap.title,
                stage=cap.stage,
                owners=cap.owners,
                covered=coverage.covered,
                symbol_counts=tuple(module.symbol_count for module in coverage.modules),
            )
        )
    return tuple(rows)


def manifest_rows(manifest: Manifest | None = None) -> tuple[ManifestRow, ...]:
    """② 清单表：12 个候选子包逐行（含"被谁认领"）."""
    resolved = build_manifest() if manifest is None else manifest
    claimants: dict[str, list[str]] = {name: [] for name in CANDIDATE_SUBPACKAGES}
    for coverage in resolved.coverages:
        for name in coverage.owners:
            claimants[name].append(coverage.capability)
    rows: list[ManifestRow] = []
    for name in CANDIDATE_SUBPACKAGES:
        info = manifest_module.resolve_subpackage(name)
        rows.append(
            ManifestRow(
                name=name,
                exists=info.exists,
                symbol_count=info.symbol_count,
                claimed_by=tuple(claimants[name]),
            )
        )
    return tuple(rows)


def stage_rows(run_result: assembly.SystemRun | None = None) -> tuple[StageRow, ...]:
    """③ 阶段表：9 段端到端链逐行."""
    resolved = assembly.run() if run_result is None else run_result
    return tuple(
        StageRow(
            index=record.index,
            stage=record.stage,
            capability=record.capability,
            ok=record.ok,
            reading=record.reading,
            detail=record.detail,
        )
        for record in resolved.records
    )


def property_rows(report: verify.PropertyReport | None = None) -> tuple[PropertyRow, ...]:
    """④ 性质表：7 条性质逐行（判据类别与读数一起印）."""
    resolved = verify.check_all() if report is None else report
    rows: list[PropertyRow] = []
    for index, outcome in enumerate(resolved.outcomes, start=1):
        rows.append(
            PropertyRow(
                index=index,
                name=outcome.name,
                criterion=outcome.criterion,
                reading=outcome.cross_check.reading if outcome.cross_check else 0.0,
                passed=outcome.passed,
                evidence=outcome.evidence,
            )
        )
    return tuple(rows)


def document_rows(documents: dict[str, str] | None = None) -> tuple[DocumentRow, ...]:
    """⑤ 文档表：两份渲染文档逐行."""
    resolved = document.render_documents() if documents is None else documents
    rows: list[DocumentRow] = []
    for name in document.DOCUMENT_NAMES:
        text = resolved.get(name, "")
        covered = sum(1 for key in CAPABILITY_ORDER if key in text)
        rows.append(
            DocumentRow(
                name=name,
                chars=len(text),
                covered=covered,
                missing=tuple(key for key in CAPABILITY_ORDER if key not in text),
            )
        )
    return tuple(rows)


def note_lines(limit: int | None = None) -> tuple[str, ...]:
    """十条笔记逐行印出（顺序即写入顺序）."""
    from smart_research_agent.capstone.types import CAPSTONE_NOTES, CAPSTONE_NOTES_ORDER

    keys = CAPSTONE_NOTES_ORDER if limit is None else CAPSTONE_NOTES_ORDER[:limit]
    return tuple(f"{index:>2}. {CAPSTONE_NOTES[key]}" for index, key in enumerate(keys, start=1))


def boundary_lines() -> tuple[str, ...]:
    """五条边界逐行印出."""
    from smart_research_agent.capstone.types import CAPSTONE_BOUNDARIES

    return tuple(f"- {item}" for item in CAPSTONE_BOUNDARIES)


def study_lines(
    run_result: assembly.SystemRun | None = None,
    manifest: Manifest | None = None,
    documents: dict[str, str] | None = None,
) -> tuple[str, ...]:
    """一次跑完五张表（演示脚本与教程引用的是同一批读数）.

    第一、三、四张表共用**同一份 run 与同一份清单**，因此五张表读的是同一时刻的状态。
    """
    resolved_run = assembly.run() if run_result is None else run_result
    resolved_manifest = build_manifest() if manifest is None else manifest
    resolved_documents = (
        document.render_documents(resolved_manifest) if documents is None else documents
    )
    report = verify.check_all(resolved_run, resolved_manifest, resolved_documents)
    lines: list[str] = []
    lines.append("== 1. 能力表（8 项能力：最近阶段 / 承担子包 / 覆盖 / 符号数）")
    for row in capability_rows(resolved_manifest):
        lines.append("  " + row.line())
    lines.append("== 2. 清单表（12 个候选子包：在场 / 公开符号数 / 认领）")
    for row in manifest_rows(resolved_manifest):
        lines.append("  " + row.line())
    lines.append("== 3. 阶段表（9 段端到端链：阶段 / 承担能力 / 读数）")
    for row in stage_rows(resolved_run):
        lines.append("  " + row.line())
    lines.append("== 4. 性质表（7 条性质：判据类别 / 读数 / 结论）")
    for row in property_rows(report):
        lines.append("  " + row.line())
    lines.append("== 5. 文档表（2 份渲染文档：字符数 / 覆盖能力数）")
    for row in document_rows(resolved_documents):
        lines.append("  " + row.line())
    lines.append("  " + adapters.ADAPTER_BOUNDARY)
    lines.append("  " + resolved_manifest.line())
    return tuple(lines)


__all__ = [
    "CapabilityRow",
    "DocumentRow",
    "ManifestRow",
    "PropertyRow",
    "StageRow",
    "boundary_lines",
    "capability_rows",
    "document_rows",
    "manifest_rows",
    "note_lines",
    "property_rows",
    "stage_rows",
    "study_lines",
]
