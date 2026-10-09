"""``roadmap``：从**真实缺口**推出后续规划（day100 / G2-D1）.

后续规划最容易写成许愿：一串"下一步要做 X、Y、Z"，读起来很完整，
却没有任何一处能对上"我们到底缺什么"。本模块把这件事反过来做：

```text
先算缺口（gaps）        →  再让每一条缺口对应一个规划项（RoadmapItem）
缺口只有两种，两种都来自一次真实读数，而不是来自一份愿望清单
```

```text
unclaimed_package    无人认领的子包     已交付、但不属于这 8 项能力清单（来自清单）
capability_below     没到最高级的能力   在场 / 符号 / 阶段三项里有一项没过（来自自评）
```

## 一、今天最值钱的一句话

> **规划要能从缺口推出：一条"没有对上的下一步"，读起来比"没有下一步"更像交付。**

因此 :class:`Roadmap` 与 :func:`gaps` 之间有一条可断言的对应关系：
**缺口集合 == 规划项集合**。少一条（有缺口没规划）与多一条（规划指向不存在的缺口）
都会被 :meth:`Roadmap.missing` / :meth:`Roadmap.ignored` 点名。

## 二、为什么"无人认领"也算一种缺口

```text
indexing / mcp_server / finetune / api 都是真的写出来了的包，
但它们不在这 8 项能力的清单里——它们不是"坏了的包"，而是"还没被纳进主线的能力"。
"要不要纳进来"是一个真实的待决问题；把它列进规划，比假装看不见更诚实。
```

## 三、与既有包的接缝

- **上游**：``capstone.manifest.build_manifest``（无人认领的子包）、
  :mod:`graduation.assessment`（没到最高级的能力）；
- **下游**：:mod:`graduation.verify` 检查"每一个缺口都有规划"，
  :mod:`graduation.study` 用它打印规划表。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smart_research_agent.capstone.manifest import Manifest, build_manifest
from smart_research_agent.capstone.types import (
    CAPABILITY_ORDER,
    CANDIDATE_SUBPACKAGES,
    SUBPACKAGE_DESCRIPTIONS,
)
from smart_research_agent.graduation.assessment import Assessment, assess
from smart_research_agent.graduation.errors import ParameterError, RoadmapError
from smart_research_agent.graduation.types import TOP_LEVEL

#: 缺口类型一：无人认领的子包（已交付，但不在这 8 项能力清单里）.
GAP_UNCLAIMED = "unclaimed_package"

#: 缺口类型二：没有到最高级的能力（自评里没拿到"已交付"）.
GAP_CAPABILITY = "capability_below"

#: 两种缺口（顺序 = 规划项的输出顺序）.
GAP_KINDS: tuple[str, ...] = (GAP_UNCLAIMED, GAP_CAPABILITY)

#: 每种缺口的一句话解释（报告里印它）.
GAP_DESCRIPTIONS: dict[str, str] = {
    GAP_UNCLAIMED: "无人认领的子包：已交付、但不属于这 8 项能力清单（要不要纳进主线？）",
    GAP_CAPABILITY: "没到最高级的能力：在场 / 符号 / 阶段三项里有一项没过",
}

if set(GAP_KINDS) != set(GAP_DESCRIPTIONS):  # pragma: no cover - 导入期不变式
    raise ValueError(
        "两种缺口的两张表不一致：GAP_KINDS 与 GAP_DESCRIPTIONS 必须逐键对齐——"
        "少一个键的缺口在报告里只有名字、没有它为什么算缺口。"
    )


def require_gap_kind(kind: str) -> str:
    """校验一种缺口类型（未知类型当场拒绝）."""
    if kind not in GAP_DESCRIPTIONS:
        raise ParameterError(
            f"未知的缺口类型 {kind!r}：可选 {list(GAP_KINDS)}——"
            "报告按这两种类型归拢规划项；类型外的名字既不会被计数、也不会被检查。"
        )
    return kind


@dataclass(frozen=True)
class RoadmapItem:
    """一个规划项：缺口键 + 缺口类型 + 为什么算缺口 + 下一步做什么.

    ``key`` 与 ``kind`` 合起来唯一确定一条缺口：键是"哪个包 / 哪一项能力"，
    类型是"它属于哪一类缺口"。两者分开，是因为同一个名字**不可能**同时是
    "无人认领的子包"与"没到最高级的能力"——一个包与一项能力从来不是同一个集合。
    """

    key: str
    kind: str
    reason: str
    action: str

    def __post_init__(self) -> None:
        require_gap_kind(self.kind)
        if not self.key:
            raise ParameterError("规划项的缺口键不能为空。")
        if not self.reason or not self.action:
            raise RoadmapError(
                f"规划项 {self.key!r} 的'为什么算缺口'与'下一步做什么'都不能为空："
                "只写要做什么的规划项无法回答'我为什么要现在做它'。"
            )

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {
            "key": self.key,
            "kind": self.kind,
            "reason": self.reason,
            "action": self.action,
        }

    def line(self) -> str:
        """一行读数：``[unclaimed_package] api | 理由：…… | 下一步：……``."""
        return f"[{self.kind:<18}] {self.key:<8} | 理由：{self.reason} | 下一步：{self.action}"


@dataclass(frozen=True)
class Roadmap:
    """一份后续规划：规划项逐行（应当与 :func:`gaps` 逐键对上）."""

    items: tuple[RoadmapItem, ...]

    def __post_init__(self) -> None:
        keys = [item.key for item in self.items]
        if len(keys) != len(set(keys)):
            raise RoadmapError(
                f"规划里有重复的缺口键：{sorted(keys)}——"
                "同一个缺口被规划两次会让'覆盖了几个缺口'这个计数虚高。"
            )

    @property
    def keys(self) -> tuple[str, ...]:
        """规划项覆盖的缺口键（顺序即输出顺序）."""
        return tuple(item.key for item in self.items)

    @property
    def kinds(self) -> tuple[str, ...]:
        """规划项里出现过的缺口类型（去重，按 :data:`GAP_KINDS` 顺序）."""
        return tuple(kind for kind in GAP_KINDS if any(item.kind == kind for item in self.items))

    def missing(self, gap_keys: tuple[str, ...]) -> tuple[str, ...]:
        """**有缺口却没有规划项**的键（应当为空）."""
        planned = set(self.keys)
        return tuple(key for key in gap_keys if key not in planned)

    def ignored(self, gap_keys: tuple[str, ...]) -> tuple[str, ...]:
        """**有规划项却没有对应缺口**的键（应当为空）."""
        known = set(gap_keys)
        return tuple(key for key in self.keys if key not in known)

    @property
    def size(self) -> int:
        """规划项条数（= 缺口条数，若两者对得上）."""
        return len(self.items)

    def to_dict(self) -> dict[str, Any]:
        """摊平成可 JSON 化的字段."""
        return {
            "size": self.size,
            "keys": list(self.keys),
            "kinds": list(self.kinds),
            "items": [item.to_dict() for item in self.items],
        }

    def line(self) -> str:
        """一行读数：``规划：4 项（unclaimed_package 4 / capability_below 0）``."""
        counts = " / ".join(
            f"{kind} {sum(1 for item in self.items if item.kind == kind)}" for kind in GAP_KINDS
        )
        return f"规划：{self.size} 项（{counts}）"


def gaps(
    manifest: Manifest | None = None,
    assessment: Assessment | None = None,
) -> tuple[tuple[str, str], ...]:
    """算出全部真实缺口：``((键, 类型), ...)``（顺序确定：先包、后能力）."""
    resolved_manifest = build_manifest() if manifest is None else manifest
    resolved_assessment = assess(manifest=resolved_manifest) if assessment is None else assessment
    found: list[tuple[str, str]] = [
        (name, GAP_UNCLAIMED) for name in resolved_manifest.unclaimed
    ]
    found.extend(
        (capability, GAP_CAPABILITY)
        for capability in CAPABILITY_ORDER
        if capability in set(resolved_assessment.below)
    )
    return tuple(found)


def plan(
    manifest: Manifest | None = None,
    assessment: Assessment | None = None,
) -> Roadmap:
    """从真实缺口推出一份规划（每个缺口恰好一个规划项，**不重不漏**）.

    ``manifest`` / ``assessment`` 可注入（测试用它构造"某一项能力没到最高级"的反例）。
    """
    resolved_manifest = build_manifest() if manifest is None else manifest
    resolved_assessment = assess(manifest=resolved_manifest) if assessment is None else assessment
    items: list[RoadmapItem] = []
    for name in resolved_manifest.unclaimed:
        items.append(
            RoadmapItem(
                key=name,
                kind=GAP_UNCLAIMED,
                reason=SUBPACKAGE_DESCRIPTIONS[name],
                action="评估是否纳入主线能力清单；若纳入，则为它补一条端到端阶段与一组读数",
            )
        )
    below = set(resolved_assessment.below)
    for capability in CAPABILITY_ORDER:
        if capability in below:
            score = resolved_assessment.score_of(capability)
            items.append(
                RoadmapItem(
                    key=capability,
                    kind=GAP_CAPABILITY,
                    reason=score.line(),
                    action="把这一项能力的证据补齐到 L3（在场 / 符号 / 阶段三项都要过）",
                )
            )
    return Roadmap(items=tuple(items))


def require_complete(
    roadmap: Roadmap | None = None,
    *,
    manifest: Manifest | None = None,
    assessment: Assessment | None = None,
) -> Roadmap:
    """规划与缺口对不上时抛 :class:`RoadmapError`（"拒绝交付"的那条路）.

    它同时检查两个方向：有缺口没规划（``missing``）与规划指向不存在的缺口（``ignored``）。
    """
    resolved_manifest = build_manifest() if manifest is None else manifest
    resolved_assessment = assess(manifest=resolved_manifest) if assessment is None else assessment
    resolved = plan(resolved_manifest, resolved_assessment) if roadmap is None else roadmap
    gap_keys = tuple(key for key, _ in gaps(resolved_manifest, resolved_assessment))
    problems: list[str] = []
    missing = resolved.missing(gap_keys)
    ignored = resolved.ignored(gap_keys)
    if missing:
        problems.append("有缺口没有规划：" + "、".join(missing))
    if ignored:
        problems.append("有规划没有对应缺口：" + "、".join(ignored))
    if problems:
        raise RoadmapError("；".join(problems) + "——一份对不上的规划读起来更完整，但它不可反驳。")
    return resolved


__all__ = [
    "GAP_CAPABILITY",
    "GAP_DESCRIPTIONS",
    "GAP_KINDS",
    "GAP_UNCLAIMED",
    "Roadmap",
    "RoadmapItem",
    "gaps",
    "plan",
    "require_complete",
    "require_gap_kind",
]
