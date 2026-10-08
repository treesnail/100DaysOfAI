"""``assessment``：八项能力逐项**按证据**评级（day100 / G2-D1）.

自评不是"我们做完了"这一句，而是 8 行**带证据**的评级。每一行的评级由三条证据推出：

```text
证据① 承担子包在场数 / 承担数     security、llm …… 全部解析得到吗
证据② 公开名字数（对门槛）         这个包对外面公开了多少个名字
证据③ 端到端链上那一段 ok          这一项能力在 day099 的九段链里真的跑通了吗
```

三级门槛把评级切成四档（:data:`graduation.types.LEVELS`）：

```text
0 未建     有承担子包不在场
1 已建     全部在场，但公开名字数 < 门槛
2 可用     前两项成立，但链上那一段 ok=False
3 已交付   三项全过
```

## 一、今天最值钱的一句话

> **一个没有证据的满分，与"我猜的满分"在报告里长得一模一样。**

因此 :class:`CapabilityScore` 在 ``__post_init__`` 里做两条闭合检查：
① 证据条数至少一条；② 评级必须**等于**由证据重算出来的那一档——
一个手写的 3 分会被第二条检查当场拦下。

## 二、为什么证据里要同时有"符号数"和"阶段 ok"

```text
只看符号数   →  "包很大"被读成"这项能力很强"（大 ≠ 用得上）
只看阶段 ok  →  "这一步跑通"被读成"整项能力交付"（一段 ≠ 一项）
两者一起     →  "接口在、而且被真的接过" 才是本课愿意给 3 分的理由
```

## 三、与既有包的接缝

- **上游**：``capstone.manifest.build_manifest``（12 个候选子包的解析结论）、
  ``capstone.assembly.run``（九段读数）、``capstone.types``（8 项能力与阶段映射）；
- **下游**：:mod:`graduation.roadmap` 从"没到最高级的能力"里取缺口，
  :mod:`graduation.verify` 检查"评级在量程内"与"每一项都有证据"。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smart_research_agent.capstone import assembly as capstone_assembly
from smart_research_agent.capstone.manifest import Manifest, build_manifest
from smart_research_agent.capstone.types import (
    CAPABILITIES,
    CAPABILITY_ORDER,
    Capability,
    capabilities,
)
from smart_research_agent.graduation.errors import AssessmentError, ParameterError
from smart_research_agent.graduation.types import (
    LEVEL_BUILT,
    LEVEL_DELIVERED,
    LEVEL_NOT_BUILT,
    LEVEL_USABLE,
    LEVELS,
    SYMBOL_FLOOR,
    TOP_LEVEL,
    require_level,
    require_positive_threshold,
)


def level_for(
    *,
    owners_present: int,
    owners_total: int,
    symbols: int,
    stage_ok: bool,
    symbol_floor: int = SYMBOL_FLOOR,
) -> int:
    """由三条证据推出评级（**一处定义**，供 :class:`CapabilityScore` 复核）.

    四档的判定顺序就是 :data:`graduation.types.LEVELS` 的顺序：
    先看"建没建起来"，再看"够不够厚"，最后看"接没接上"。
    """
    require_positive_threshold("symbol_floor", symbol_floor)
    if owners_total < 1:
        raise ParameterError("owners_total 至少要有一个承担子包——否则这一项没有任何落点。")
    if owners_present < 0 or owners_present > owners_total:
        raise ParameterError(
            f"owners_present={owners_present} 必须落在 [0, {owners_total}]："
            "在场数超过承担数说明两份数据不是同一次读的。"
        )
    if symbols < 0:
        raise ParameterError(f"symbols 不能为负，收到 {symbols}。")
    if owners_present < owners_total:
        return LEVEL_NOT_BUILT
    if symbols < symbol_floor:
        return LEVEL_BUILT
    if not stage_ok:
        return LEVEL_USABLE
    return LEVEL_DELIVERED


@dataclass(frozen=True)
class CapabilityScore:
    """一项能力的一次自评：评级 + 三项证据 + 一行可读结论.

    ``level`` 必须等于 :func:`level_for` 用同一批证据重算出来的那一档——
    这条闭合检查让"手写一个 3 分"不可能悄悄通过。
    """

    capability: str
    level: int
    owners: tuple[str, ...]
    owners_present: int
    owners_total: int
    symbols: int
    stage: str
    stage_ok: bool
    evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.capability not in CAPABILITIES:
            raise AssessmentError(
                f"未知的能力 {self.capability!r}：可选 {list(CAPABILITY_ORDER)}。"
            )
        require_level(self.level)
        if not self.evidence:
            raise AssessmentError(
                f"能力 {self.capability!r} 的自评没有任何证据："
                "一个没有证据的分数与'我猜的分数'在报告里长得一模一样。"
            )
        expected = level_for(
            owners_present=self.owners_present,
            owners_total=self.owners_total,
            symbols=self.symbols,
            stage_ok=self.stage_ok,
        )
        if self.level != expected:
            raise AssessmentError(
                f"能力 {self.capability!r} 的评级 {self.level} 与证据重算出的 {expected} 不一致："
                "评级只能由证据推出，不能由人手写——否则'我检查过'与'我猜的'分不开。"
            )

    @property
    def delivered(self) -> bool:
        """是否达到最高级（已交付）."""
        return self.level == TOP_LEVEL

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {
            "capability": self.capability,
            "level": self.level,
            "level_name": LEVEL_NAMES[self.level],
            "owners": list(self.owners),
            "owners_present": self.owners_present,
            "owners_total": self.owners_total,
            "symbols": self.symbols,
            "stage": self.stage,
            "stage_ok": self.stage_ok,
            "evidence": list(self.evidence),
        }

    def line(self) -> str:
        """一行读数：``L3 已交付 | input_guard | 承担 2/2 | 名字 13 | 阶段 guard ok=True``."""
        mark = "✓" if self.delivered else "·"
        return (
            f"L{self.level} {LEVEL_NAMES[self.level]} {mark} | {self.capability:<22}"
            f" | 承担 {self.owners_present}/{self.owners_total} | 名字 {self.symbols}"
            f" | 阶段 {self.stage} ok={self.stage_ok}"
        )


#: 评级 → 名字（供 :meth:`CapabilityScore.line` 与报告复用）.
LEVEL_NAMES: dict[int, str] = {
    LEVEL_NOT_BUILT: "未建",
    LEVEL_BUILT: "已建",
    LEVEL_USABLE: "可用",
    LEVEL_DELIVERED: "已交付",
}

if set(LEVEL_NAMES) != set(LEVELS):  # pragma: no cover - 导入期不变式
    raise AssessmentError("评级名字表与量程不一致：LEVEL_NAMES 必须与 LEVELS 逐键对齐。")


@dataclass(frozen=True)
class Assessment:
    """一份自评：8 项能力逐行（``rows`` 与 :data:`CAPABILITY_ORDER` 逐项对齐）.

    ``min_evidence`` 是"最少的一项能力挂了几条证据"——它是性质
    ``assessment_has_evidence`` 的读数（一条下界）。
    ``max_level`` 是"最高的一项评级"——它是性质
    ``assessment_within_scale`` 的读数（一条上界）。
    """

    rows: tuple[CapabilityScore, ...]

    def __post_init__(self) -> None:
        keys = [row.capability for row in self.rows]
        if len(keys) != len(set(keys)):
            raise AssessmentError(f"自评里有重复的能力：{sorted(keys)}——同一项能力不能被评两次。")
        if set(keys) != set(CAPABILITY_ORDER):
            raise AssessmentError(
                f"自评的能力集合与 CAPABILITY_ORDER 不一致："
                f"多 {sorted(set(keys) - set(CAPABILITY_ORDER))}、"
                f"缺 {sorted(set(CAPABILITY_ORDER) - set(keys))}。"
            )

    @property
    def max_level(self) -> int:
        """最高的一项评级."""
        return max(row.level for row in self.rows)

    @property
    def min_level(self) -> int:
        """最低的一项评级."""
        return min(row.level for row in self.rows)

    @property
    def min_evidence(self) -> int:
        """最少的一项能力挂了几条证据（下界判据的读数）."""
        return min(len(row.evidence) for row in self.rows)

    @property
    def delivered(self) -> tuple[str, ...]:
        """达到最高级的能力 id（应当有 8 个）."""
        return tuple(row.capability for row in self.rows if row.delivered)

    @property
    def below(self) -> tuple[str, ...]:
        """没到最高级的能力 id（应当为空，否则它们会进后续规划）."""
        return tuple(row.capability for row in self.rows if not row.delivered)

    def score_of(self, capability: str) -> CapabilityScore:
        """取一项能力的自评（未知能力当场拒绝）."""
        for row in self.rows:
            if row.capability == capability:
                return row
        raise ParameterError(f"自评里没有能力 {capability!r}。")

    def to_dict(self) -> dict[str, Any]:
        """摊平成可 JSON 化的字段."""
        return {
            "max_level": self.max_level,
            "min_level": self.min_level,
            "min_evidence": self.min_evidence,
            "delivered": list(self.delivered),
            "below": list(self.below),
            "rows": [row.to_dict() for row in self.rows],
        }

    def line(self) -> str:
        """一行读数：``自评：最高 L3 / 最低 L3 | 已交付 8/8 项 | 最少证据 3 条``."""
        return (
            f"自评：最高 L{self.max_level} / 最低 L{self.min_level}"
            f" | 已交付 {len(self.delivered)}/{len(CAPABILITY_ORDER)} 项"
            f" | 最少证据 {self.min_evidence} 条"
        )


def _evidence_for(
    *,
    capability: Capability,
    owners_present: int,
    owners_total: int,
    symbols: int,
    stage_ok: bool,
    symbol_floor: int,
    missing: tuple[str, ...],
) -> tuple[str, ...]:
    """把三条证据折成可读的行（缺包时**点名**缺的是谁）."""
    items = [
        f"承担子包 {owners_present}/{owners_total} 在场",
        f"公开名字 {symbols}（门槛 {symbol_floor}）",
        f"链上阶段 {capability.stage} ok={stage_ok}",
    ]
    if missing:
        items.append("缺失的承担子包：" + "、".join(missing))
    return tuple(items)


def assess(
    manifest: Manifest | None = None,
    run: capstone_assembly.SystemRun | None = None,
    *,
    symbol_floor: int = SYMBOL_FLOOR,
) -> Assessment:
    """对 8 项能力逐项评级（证据来自清单与端到端链，**不读任何写死的分数**）.

    ``manifest`` / ``run`` / ``symbol_floor`` 都可注入（测试用它构造反例）。
    """
    require_positive_threshold("symbol_floor", symbol_floor)
    resolved_manifest = build_manifest() if manifest is None else manifest
    resolved_run = capstone_assembly.run() if run is None else run
    rows: list[CapabilityScore] = []
    for cap in capabilities():
        coverage = resolved_manifest.coverage_of(cap.id)
        owners = cap.owners
        owners_total = len(owners)
        owners_present = sum(1 for module in coverage.modules if module.exists)
        symbols = sum(module.symbol_count for module in coverage.modules)
        stage_ok = resolved_run.record_of(cap.stage).ok
        rows.append(
            CapabilityScore(
                capability=cap.id,
                level=level_for(
                    owners_present=owners_present,
                    owners_total=owners_total,
                    symbols=symbols,
                    stage_ok=stage_ok,
                    symbol_floor=symbol_floor,
                ),
                owners=owners,
                owners_present=owners_present,
                owners_total=owners_total,
                symbols=symbols,
                stage=cap.stage,
                stage_ok=stage_ok,
                evidence=_evidence_for(
                    capability=cap,
                    owners_present=owners_present,
                    owners_total=owners_total,
                    symbols=symbols,
                    stage_ok=stage_ok,
                    symbol_floor=symbol_floor,
                    missing=coverage.missing,
                ),
            )
        )
    return Assessment(rows=tuple(rows))


def require_delivered(assessment: Assessment | None = None) -> Assessment:
    """有任一能力没到最高级时抛 :class:`AssessmentError`（"拒绝交付"的那条路）.

    它与 :func:`assess` 的分工是"记录 vs 拒绝"：
    前者给自评表用（不抛），后者给"拒绝交付"那条路用（抛）。
    """
    resolved = assess() if assessment is None else assessment
    if not resolved.below:
        return resolved
    failures = [row.line() for row in resolved.rows if not row.delivered]
    raise AssessmentError("有能力的自评没有到最高级：" + "；".join(failures))


__all__ = [
    "LEVEL_NAMES",
    "Assessment",
    "CapabilityScore",
    "assess",
    "level_for",
    "require_delivered",
]
