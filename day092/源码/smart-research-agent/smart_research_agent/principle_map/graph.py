"""``graph``：把"原理 → 实现 → 应用"拼成一张**可校验**的图（day088 / M7-D12）.

图只有两种节点与一种边：

```text
节点     原理（12 块拼图）、应用（6 种能力）
边       principle → application（由 Principle.application 给出）
标注     each principle 的 artifact（实现落点）与 evidence（现场读数）
```

## 一条纪律：**单次计算、可重复**

图的构建只做一次（默认那份用 ``lru_cache`` 缓存），而它的每一个读数都由
:mod:`reconcile` 的探针现场算出。因此"这张图"不是一个可以被手改的常量，
而是一次计算的结果——重复构建两次，得到的边与读数**逐位相同**。

## 缺支撑时要抛 ``CoverageError``

两种"缺一块"都会被检查出来，而且消息里必须指出**缺的是哪一个**：

```text
某个应用没有任何原理支撑  ⇒ "应用 serving 在图上没有任何入边"
某条原理没有实现落点      ⇒ "原理 cosine_is_normalized_dot 没有对应证据"
```

"覆盖不全"这句话没有任何下一步动作；"缺的是 serving"有。
"""

from __future__ import annotations

import functools
import importlib
from dataclasses import dataclass, field

from smart_research_agent.principle_map import claims, reconcile
from smart_research_agent.principle_map.errors import CoverageError
from smart_research_agent.principle_map.types import (
    APPLICATIONS,
    APPLICATION_DESCRIPTIONS,
    LAYERS,
    Application,
    CoverageReport,
    Evidence,
    Principle,
)


@dataclass(frozen=True)
class PrincipleGraph:
    """一张图：十二块拼图 + 六个应用 + 十二条现场读数."""

    principles: tuple[Principle, ...]
    applications: tuple[Application, ...]
    evidence: tuple[Evidence, ...] = field(default=())

    @property
    def edges(self) -> tuple[tuple[str, str], ...]:
        """全部边：``(principle_id, application_id)``（顺序与 ``principles`` 一致）."""
        return tuple((item.id, item.application) for item in self.principles)

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        """有读数的原理 id（顺序与 ``evidence`` 一致）."""
        return tuple(item.principle for item in self.evidence)

    def evidence_of(self, principle_id: str) -> Evidence:
        """某条原理的读数（没有则抛 ``CoverageError``）."""
        for item in self.evidence:
            if item.principle == principle_id:
                return item
        raise CoverageError(
            f"原理 {principle_id!r} 没有对应证据：它在这张图上只有箭头、没有读数。"
        )

    def application(self, application_id: str) -> Application:
        """某个应用的记录（没有则抛 ``CoverageError``）."""
        for item in self.applications:
            if item.id == application_id:
                return item
        raise CoverageError(f"图里没有应用 {application_id!r}。")

    @property
    def supported(self) -> tuple[str, ...]:
        """有原理支撑的应用（顺序与 ``APPLICATIONS`` 一致）."""
        return tuple(item.id for item in self.applications if item.supported)

    @property
    def unsupported(self) -> tuple[str, ...]:
        """一条原理都没有的应用（应当为空）."""
        return tuple(item.id for item in self.applications if not item.supported)

    @property
    def orphans(self) -> tuple[str, ...]:
        """没有被任何应用引用的原理（应当为空）."""
        referenced = {pid for item in self.applications for pid in item.principles}
        return tuple(item.id for item in self.principles if item.id not in referenced)

    @property
    def missing_evidence(self) -> tuple[str, ...]:
        """没有读数的原理（应当为空）."""
        present = set(self.evidence_ids)
        return tuple(item.id for item in self.principles if item.id not in present)

    @property
    def complete(self) -> bool:
        """两个条件同时成立：每个应用都有原理、每条原理都有读数."""
        return not self.unsupported and not self.orphans and not self.missing_evidence

    def layer_counts(self) -> dict[str, int]:
        """按层计数（四个层全部出现，计数可以为 0）."""
        counts = {layer: 0 for layer in LAYERS}
        for item in self.principles:
            counts[item.layer] = counts.get(item.layer, 0) + 1
        return counts

    def application_counts(self) -> dict[str, int]:
        """按应用计数（六个应用全部出现，计数可以为 0）."""
        return {item.id: len(item.principles) for item in self.applications}

    def coverage(self) -> CoverageReport:
        """把覆盖情况摊成一份报告（**不抛错**，供报告与测试读明细）."""
        return CoverageReport(
            per_layer=self.layer_counts(),
            per_application=self.application_counts(),
            complete=self.complete,
        )

    def require_complete(self) -> None:
        """缺支撑时抛 :class:`errors.CoverageError`（消息里指出缺的是哪一个）."""
        if self.unsupported:
            raise CoverageError(
                "以下应用在图上没有任何入边（没有得到任何原理支撑）："
                + "、".join(self.unsupported)
                + "。一条没有依据的能力就是一句口号。"
            )
        if self.orphans:
            raise CoverageError(
                "以下原理没有被任何应用引用（它们悬在图上）："
                + "、".join(self.orphans)
                + "。"
            )
        if self.missing_evidence:
            raise CoverageError(
                "以下原理没有实现落点 / 没有读数："
                + "、".join(self.missing_evidence)
                + "。"
            )

    def lines(self) -> tuple[str, ...]:
        """逐行读数：每个应用一行 + 一行覆盖汇总."""
        lines = [item.line() for item in self.applications]
        lines.append(self.coverage().line())
        return tuple(lines)

    def to_dict(self) -> dict[str, object]:
        """摊平成可 JSON 化的字段."""
        return {
            "principles": [item.to_dict() for item in self.principles],
            "applications": [item.to_dict() for item in self.applications],
            "evidence": [item.to_dict() for item in self.evidence],
            "edges": [list(edge) for edge in self.edges],
            "coverage": self.coverage().to_dict(),
        }


def resolve_artifact(artifact: str) -> object:
    """把 ``"模块.函数"`` 解析成一个真实存在的对象（失败抛 ``ReferenceError``）.

    这是"每一条原理都有实现落点"这句断言的**唯一实现处**：
    解析成功意味着那个包真的存在、那个名字真的在那儿。
    """
    from smart_research_agent.principle_map.errors import ReferenceError

    if "." not in artifact:
        raise ReferenceError(
            f"artifact 必须是'模块.函数'的形式，收到 {artifact!r}：没有点号的引用无法被解析。"
        )
    module_name, _, attribute = artifact.rpartition(".")
    full_module = f"smart_research_agent.{module_name}"
    try:
        module = importlib.import_module(full_module)
    except ImportError as error:
        raise ReferenceError(
            f"artifact 指向的模块 {full_module!r} 不存在（{error}）："
            "要么名字写错了，要么那个包根本没有实现过这件事。"
        ) from error
    if not hasattr(module, attribute):
        raise ReferenceError(
            f"模块 {full_module!r} 里没有 {attribute!r}："
            "名字对不上时'引用'与'不成立'读起来一样，因此这里必须当场拒绝。"
        )
    return getattr(module, attribute)


@functools.lru_cache(maxsize=1)
def _default_graph() -> PrincipleGraph:
    """默认那张图（**缓存一次**：探针只跑一遍，"单次计算、可重复"由此成立）."""
    return build_graph()


def build_graph(
    principles: tuple[Principle, ...] | None = None,
    evidence: tuple[Evidence, ...] | None = None,
) -> PrincipleGraph:
    """构建一张图（默认用 :mod:`claims` 与 :mod:`reconcile` 的真实现场读数）.

    参数只在测试"故意造一张破图"时传入——生产路径永远用默认那份。
    """
    items = claims.principles() if principles is None else tuple(principles)
    evidences = reconcile.probe_all() if evidence is None else tuple(evidence)
    applications = tuple(
        Application(
            id=app,
            description=APPLICATION_DESCRIPTIONS[app],
            principles=tuple(item.id for item in items if item.application == app),
        )
        for app in APPLICATIONS
    )
    return PrincipleGraph(principles=items, applications=applications, evidence=evidences)


def supported_applications(graph: PrincipleGraph | None = None) -> tuple[str, ...]:
    """有原理支撑的应用（默认那份图）."""
    return (graph or _default_graph()).supported


def unsupported_applications(graph: PrincipleGraph | None = None) -> tuple[str, ...]:
    """没有任何原理支撑的应用（应当为空）."""
    return (graph or _default_graph()).unsupported


def orphan_principles(graph: PrincipleGraph | None = None) -> tuple[str, ...]:
    """没有被任何应用引用的原理（应当为空）."""
    return (graph or _default_graph()).orphans


def layer_coverage(graph: PrincipleGraph | None = None) -> dict[str, int]:
    """按层的覆盖计数."""
    return (graph or _default_graph()).layer_counts()


def application_coverage(graph: PrincipleGraph | None = None) -> dict[str, int]:
    """按应用的覆盖计数."""
    return (graph or _default_graph()).application_counts()


def coverage_report(graph: PrincipleGraph | None = None) -> CoverageReport:
    """覆盖报告（**缺支撑时抛 ``CoverageError``**）."""
    resolved = graph or _default_graph()
    resolved.require_complete()
    return resolved.coverage()


def graph_lines(graph: PrincipleGraph | None = None) -> tuple[str, ...]:
    """逐行读数（报告里读它）."""
    return (graph or _default_graph()).lines()


__all__ = [
    "PrincipleGraph",
    "application_coverage",
    "build_graph",
    "coverage_report",
    "graph_lines",
    "layer_coverage",
    "orphan_principles",
    "resolve_artifact",
    "supported_applications",
    "unsupported_applications",
]
