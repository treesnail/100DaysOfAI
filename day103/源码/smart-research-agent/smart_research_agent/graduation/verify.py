"""七条性质与三类判据（day100 / G2-D1）.

本模块是"这次交付成不成立"的判据所在。七条性质按三类判据分：

```text
相等（==）   ① demo_replays_identically       两次重放的差异项数 == 0
             ② deliverables_are_complete      缺失交付物数 == 0
             ③ inventory_accounts_for_all     重复+漏项 == 0
             ④ roadmap_covers_every_gap       未覆盖的缺口数 == 0
             ⑦ summary_matches_inventory      总结与清单的差异项数 == 0
上界（<=）   ⑤ assessment_within_scale        最高评级 <= 3（TOP_LEVEL）
下界（>=）   ⑥ assessment_has_evidence        最少证据条数 >= 1
```

## 一、为什么第 ⑥ 条必须是下界

"每一项能力都至少挂一条证据"**只能从下面兜住**：证据越多越好，
到 3 条、4 条都不会"超过"某个上限。写成相等判据会在报告补了一条证据时误报失败。
下界才是它真正的方向，而这条判据在报告里的读数是"越低越危险"。

## 二、为什么第 ⑤ 条是上界

"最高评级不超过量程"是一条**越少越好**的量（越界 = 一个分数被读成它不属于的档）。
它不是相等：8 项能力不可能都恰好是 3 分才是合格的——
某些能力只到 2 分是合法的，只要它有对应的规划项（第 ④ 条会兜住它）。

## 三、一条纪律：`CrossCheck.passed` 必须与 `PropertyOutcome.passed` 一致

一份报告里，"那一行说通过"与"它挂的对账说没通过"是最糟的状态——
读的人只会相信看起来更合理的那一个。因此 :class:`PropertyOutcome` 在
``__post_init__`` 里当场拒绝两者不一致的构造。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.capstone.manifest import Manifest, build_manifest
from smart_research_agent.graduation.assessment import (
    Assessment,
    assess,
)
from smart_research_agent.graduation.demo import DemoTranscript, ReplayReport, build_transcript, replay
from smart_research_agent.graduation.errors import (
    AssessmentError,
    NumericError,
    ParameterError,
)
from smart_research_agent.graduation.inventory import InventoryReport, build_inventory
from smart_research_agent.graduation.roadmap import Roadmap, gaps, plan
from smart_research_agent.graduation.summary import (
    CourseSummary,
    build_summary,
    summary_matches_inventory,
)
from smart_research_agent.graduation.types import (
    CRITERION_EQUALITY,
    CRITERION_LOWER_BOUND,
    CRITERION_UPPER_BOUND,
    DELIVERABLE_ORDER,
    GRADUATION_PROPERTIES,
    PROPERTY_ASSESSMENT_HAS_EVIDENCE,
    PROPERTY_ASSESSMENT_WITHIN_SCALE,
    PROPERTY_DELIVERABLES_ARE_COMPLETE,
    PROPERTY_DEMO_REPLAYS_IDENTICALLY,
    PROPERTY_INVENTORY_ACCOUNTS_FOR_ALL,
    PROPERTY_ROADMAP_COVERS_EVERY_GAP,
    PROPERTY_SUMMARY_MATCHES_INVENTORY,
    TOP_LEVEL,
)

#: 相等判据的浮点容差（逐位 / 整数相等时不走它）.
EQUALITY_TOLERANCE = 1e-12

#: 每项能力至少要挂几条证据（**下界**：这是一条"只能从下面兜住"的性质）.
EVIDENCE_FLOOR = 1

#: 自评评级的上界（越过它 = 一个分数被读成它不属于的档）.
SCALE_CEILING = float(TOP_LEVEL)


@dataclass(frozen=True)
class CrossCheck:
    """一次对账：来源、读数、判据（**含方向**：相等 / 上界 / 下界）.

    三个字段里**最多只能给一个方向**：给 ``lower_bound`` 时判据是"读数 >= 下界"，
    给 ``upper_bound`` 时是"读数 <= 上界"，都不给时才是"读数 == 期望"。
    """

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
                f"对账 {self.name!r} 同时给了上界与下界："
                "一个数不可能既是上界又是下界——请只给一个方向。"
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
                f"性质 {self.name!r} 标了'不适用'却又标了'通过'：两者必须分开——"
                "否则'把这条检查删掉'与'它通过了'在报告里长得一模一样。"
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
        """不通过时抛 :class:`errors.AssessmentError`（"拒绝交付"的那条路）."""
        if self.ok:
            return
        failures = [outcome.line() for outcome in self.applicable if not outcome.passed]
        raise AssessmentError("性质检查未全部通过：" + "；".join(failures))

    def lines(self) -> tuple[str, ...]:
        """逐行文本（**先印不适用**，让"这一次没查它"一眼可见）."""
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
# 四份交付物的正文（第 ② 条性质读它）
# --------------------------------------------------------------------------- #


def render_deliverables(
    *,
    transcript: DemoTranscript | None = None,
    inventory: InventoryReport | None = None,
    summary: CourseSummary | None = None,
    assessment: Assessment | None = None,
    roadmap: Roadmap | None = None,
) -> dict[str, tuple[str, ...]]:
    """把四份交付物渲染成正文行（``{交付物 id: 行}``，键与 :data:`DELIVERABLE_ORDER` 对齐）.

    每份交付物的正文都**来自一次真实读数**，没有一行是手写的：
    剧本来自九段链、清单来自子包扫描、自评来自证据、规划来自缺口。
    """
    resolved_inventory = build_inventory() if inventory is None else inventory
    resolved_transcript = build_transcript() if transcript is None else transcript
    resolved_summary = build_summary(resolved_inventory) if summary is None else summary
    resolved_assessment = assess() if assessment is None else assessment
    resolved_roadmap = plan() if roadmap is None else roadmap
    return {
        "demo": resolved_transcript.lines(),
        "summary": (resolved_summary.line(), *inventory_lines_of(resolved_inventory)),
        "assessment": tuple(row.line() for row in resolved_assessment.rows),
        "roadmap": tuple(item.line() for item in resolved_roadmap.items),
    }


def inventory_lines_of(report: InventoryReport) -> tuple[str, ...]:
    """把清单折成"摘要 + 名册"两段（供交付物正文复用）."""
    return (report.line(), *((row.line() for row in report.rows)))


def missing_deliverables(bodies: dict[str, tuple[str, ...]] | None = None) -> tuple[str, ...]:
    """返回**空白**的交付物 id（空元组表示四份都非空）.

    判据只有一条：正文里有没有**非空的行**。一份没有下一步的规划、
    一份没有读数的剧本，都会在这里被点名——而不是被"看起来有标题"蒙混过去。
    """
    resolved = render_deliverables() if bodies is None else bodies
    missing: list[str] = []
    for name in DELIVERABLE_ORDER:
        lines = resolved.get(name, ())
        if not any(line.strip() for line in lines):
            missing.append(name)
    return tuple(missing)


# --------------------------------------------------------------------------- #
# 七条性质
# --------------------------------------------------------------------------- #


def check_demo_replays_identically(report: ReplayReport | None = None) -> PropertyOutcome:
    """① 剧本重放两次**逐位相同**."""
    resolved = replay() if report is None else report
    diff = resolved.diff_count
    check = CrossCheck(
        name="两次重放的差异项数",
        left="replay().first",
        right="replay().second",
        reading=float(diff),
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_DEMO_REPLAYS_IDENTICALLY,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"差异项数 {diff}（0 = 逐位相同）",
            f"摘要 {resolved.first.digest} / {resolved.second.digest}",
        ),
        cross_check=check,
    )


def check_deliverables_are_complete(
    bodies: dict[str, tuple[str, ...]] | None = None,
) -> PropertyOutcome:
    """② 四份交付物全部非空."""
    resolved = render_deliverables() if bodies is None else bodies
    missing = missing_deliverables(resolved)
    check = CrossCheck(
        name="空白交付物的份数",
        left="render_deliverables() 的正文",
        right="DELIVERABLE_ORDER 的 4 份",
        reading=float(len(missing)),
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_DELIVERABLES_ARE_COMPLETE,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"交付物 {len(DELIVERABLE_ORDER) - len(missing)}/{len(DELIVERABLE_ORDER)} 份非空",
            "空白的：" + ("、".join(missing) if missing else "无"),
        ),
        cross_check=check,
    )


def check_inventory_accounts_for_all(report: InventoryReport | None = None) -> PropertyOutcome:
    """③ 清单既不重复也不漏项（发现集合 == 记录集合）."""
    resolved = build_inventory() if report is None else report
    names = [row.name for row in resolved.rows]
    duplicated = len(names) - len(set(names))
    missing = len(resolved.missing)
    check = CrossCheck(
        name="重复项 + 漏项",
        left="InventoryReport.rows",
        right="discover_subpackages()",
        reading=float(duplicated + missing),
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_INVENTORY_ACCOUNTS_FOR_ALL,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"子包 {len(names)} 个 / 发现 {len(resolved.discovered)} 个 | 重复 {duplicated} / 漏项 {missing}",
            resolved.line(),
        ),
        cross_check=check,
    )


def check_roadmap_covers_every_gap(
    roadmap: Roadmap | None = None,
    *,
    manifest: Manifest | None = None,
    assessment: Assessment | None = None,
) -> PropertyOutcome:
    """④ 每一个真实缺口都有对应规划项（且规划项不指向不存在的缺口）."""
    resolved_manifest = build_manifest() if manifest is None else manifest
    resolved_assessment = assess(manifest=resolved_manifest) if assessment is None else assessment
    resolved_roadmap = plan(resolved_manifest, resolved_assessment) if roadmap is None else roadmap
    gap_keys = tuple(key for key, _ in gaps(resolved_manifest, resolved_assessment))
    missing = resolved_roadmap.missing(gap_keys)
    ignored = resolved_roadmap.ignored(gap_keys)
    check = CrossCheck(
        name="未覆盖的缺口数 + 多余的规划项数",
        left="Roadmap vs gaps()",
        right="缺口集合 == 规划项集合",
        reading=float(len(missing) + len(ignored)),
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_ROADMAP_COVERS_EVERY_GAP,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"缺口 {len(gap_keys)} 个 / 规划 {resolved_roadmap.size} 项 | 未覆盖 {len(missing)} / 多余 {len(ignored)}",
            resolved_roadmap.line(),
        ),
        cross_check=check,
    )


def check_assessment_within_scale(assessment: Assessment | None = None) -> PropertyOutcome:
    """⑤ **上界**：最高评级不超过量程（读数 <= TOP_LEVEL）."""
    resolved = assess() if assessment is None else assessment
    highest = float(resolved.max_level)
    check = CrossCheck(
        name="最高评级",
        left="Assessment.max_level",
        right=f"上界 {SCALE_CEILING:.0f}",
        reading=highest,
        expected=SCALE_CEILING,
        exact=False,
        upper_bound=SCALE_CEILING,
    )
    return PropertyOutcome(
        name=PROPERTY_ASSESSMENT_WITHIN_SCALE,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"最高 L{resolved.max_level} ≤ 上界 L{TOP_LEVEL}",
            resolved.line(),
        ),
        cross_check=check,
    )


def check_assessment_has_evidence(assessment: Assessment | None = None) -> PropertyOutcome:
    """⑥ **下界**：每一项能力都至少挂一条证据（读数 >= 1）."""
    resolved = assess() if assessment is None else assessment
    fewest = float(resolved.min_evidence)
    check = CrossCheck(
        name="最少证据条数",
        left="Assessment.min_evidence",
        right=f"下界 {EVIDENCE_FLOOR}",
        reading=fewest,
        expected=float(EVIDENCE_FLOOR),
        exact=False,
        lower_bound=float(EVIDENCE_FLOOR),
    )
    return PropertyOutcome(
        name=PROPERTY_ASSESSMENT_HAS_EVIDENCE,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"最少证据 {resolved.min_evidence} 条 ≥ 下界 {EVIDENCE_FLOOR}",
            resolved.line(),
        ),
        cross_check=check,
    )


def check_summary_matches_inventory(
    summary: CourseSummary | None = None,
    inventory: InventoryReport | None = None,
) -> PropertyOutcome:
    """⑦ 总结与清单给出同一个计数（差异项数 == 0）."""
    resolved_inventory = build_inventory() if inventory is None else inventory
    resolved_summary = build_summary(resolved_inventory) if summary is None else summary
    diff = summary_matches_inventory(resolved_summary, resolved_inventory)
    check = CrossCheck(
        name="总结与清单的差异项数",
        left="CourseSummary 的三个计数",
        right="InventoryReport 的三个计数",
        reading=float(diff),
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_SUMMARY_MATCHES_INVENTORY,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"差异项数 {diff}（0 = 一致）",
            resolved_summary.line(),
        ),
        cross_check=check,
    )


def check_all(
    *,
    replay_report: ReplayReport | None = None,
    bodies: dict[str, tuple[str, ...]] | None = None,
    inventory: InventoryReport | None = None,
    manifest: Manifest | None = None,
    assessment: Assessment | None = None,
    roadmap: Roadmap | None = None,
    summary: CourseSummary | None = None,
) -> PropertyReport:
    """一次跑完七条性质（顺序与 :data:`types.GRADUATION_PROPERTIES` 一致）.

    七条性质**共用同一批输入**（同一份清单 / 自评 / 规划 / 总结），
    因此报告读的是同一时刻的状态，而不是七次各跑一遍的七份快照。
    """
    resolved_inventory = build_inventory() if inventory is None else inventory
    resolved_manifest = build_manifest() if manifest is None else manifest
    resolved_assessment = assess(resolved_manifest) if assessment is None else assessment
    resolved_roadmap = plan(resolved_manifest, resolved_assessment) if roadmap is None else roadmap
    resolved_summary = build_summary(resolved_inventory) if summary is None else summary
    resolved_bodies = (
        render_deliverables(
            inventory=resolved_inventory,
            summary=resolved_summary,
            assessment=resolved_assessment,
            roadmap=resolved_roadmap,
        )
        if bodies is None
        else bodies
    )
    outcomes = (
        check_demo_replays_identically(replay_report),
        check_deliverables_are_complete(resolved_bodies),
        check_inventory_accounts_for_all(resolved_inventory),
        check_roadmap_covers_every_gap(resolved_roadmap, manifest=resolved_manifest, assessment=resolved_assessment),
        check_assessment_within_scale(resolved_assessment),
        check_assessment_has_evidence(resolved_assessment),
        check_summary_matches_inventory(resolved_summary, resolved_inventory),
    )
    names = {outcome.name for outcome in outcomes}
    extra = names - set(GRADUATION_PROPERTIES)
    missing = set(GRADUATION_PROPERTIES) - names
    if extra or missing:  # pragma: no cover - 只在有人改性质名单时触发
        raise AssessmentError(
            "性质名单与 types.GRADUATION_PROPERTIES 不一致："
            f"多 {sorted(extra)}、缺 {sorted(missing)}。"
        )
    return PropertyReport(outcomes=outcomes)


def require_ok(report: PropertyReport | None = None) -> PropertyReport:
    """七条性质没全部通过时抛 :class:`AssessmentError`（"拒绝交付"的那条路）.

    它只检查**七条性质**，不检查"是否八项能力都到 L3"——
    后者是 :func:`assessment.require_delivered` 那条更严的路。
    两条路刻意分开：本课的真实状态是"七条性质全过、但有一项能力停在 L1"，
    把两者绑在一起会让"有一条能力在 L1 且已被规划"这种**合法**状态被当成失败，
    从而逼着人把自评的分数往上写。
    """
    resolved = check_all() if report is None else report
    if not resolved.ok:
        resolved.require_ok()
    return resolved


__all__ = [
    "EQUALITY_TOLERANCE",
    "EVIDENCE_FLOOR",
    "SCALE_CEILING",
    "CrossCheck",
    "PropertyOutcome",
    "PropertyReport",
    "check_all",
    "check_assessment_has_evidence",
    "check_assessment_within_scale",
    "check_deliverables_are_complete",
    "check_demo_replays_identically",
    "check_inventory_accounts_for_all",
    "check_roadmap_covers_every_gap",
    "check_summary_matches_inventory",
    "inventory_lines_of",
    "missing_deliverables",
    "render_deliverables",
    "require_ok",
]
