"""七条性质与三类判据（day103）.

本模块是"这张依赖图成不成立"的判据所在。七条性质按三类判据分：

```text
相等（==）   ① graph_covers_all_modules        缺失数 == 0
             ② graph_is_reproducible            两次构建的差异项数 == 0
             ③ topological_order_is_valid       违反线性序的跨分量边数 == 0
             ④ cycles_are_sound                 不成立的环数 == 0
上界（<=）   ⑤ import_targets_resolve           未解析的包内目标数 <= 0
下界（>=）   ⑥ closures_include_self            最小下游闭包 >= 1
             ⑦ dependency_depth_is_positive      最长下游闭包 >= 2
```

## 一、为什么 ⑥ ⑦ 必须是下界

```text
"每个模块的闭包至少含它自己"   闭包是集合 ⇒ 只能从下面兜住
"至少存在一条依赖链"           "有边"这件事只能从下面兜住（写成 == 2 会漏掉更长的链）
```

## 二、为什么 ⑤ 是上界

"未解析的包内目标数"是一条**越少越好**的量：0 才是健康的。
相对导入少减一层点，边就指到隔壁——图照样画得出来、照样能排序，
**只有这条上界会被它顶破**。这与 day101 的"分数不超过 1"、day102 的"未解析基类不超过 0"同源。

## 三、一条纪律：`CrossCheck.passed` 必须与 `PropertyOutcome.passed` 一致

一份报告里，"那一行说通过"与"它挂的对账说没通过"是最糟的状态。
因此 :class:`PropertyOutcome` 在 ``__post_init__`` 里当场拒绝两者不一致的构造。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.import_graph.errors import GraphError, NumericError, ParameterError
from smart_research_agent.import_graph.graph import ModuleGraph, build_graph
from smart_research_agent.import_graph.parse import (
    ScannedModule,
    expected_modules,
    unresolved_of,
)
from smart_research_agent.import_graph.types import (
    DIRECTION_DOWN,
    DIRECTION_UP,
    IMPORT_GRAPH_PROPERTIES,
    CRITERION_EQUALITY,
    CRITERION_LOWER_BOUND,
    CRITERION_UPPER_BOUND,
    PROPERTY_CLOSURES_INCLUDE_SELF,
    PROPERTY_CYCLES_ARE_SOUND,
    PROPERTY_DEPENDENCY_DEPTH_IS_POSITIVE,
    PROPERTY_GRAPH_COVERS_ALL_MODULES,
    PROPERTY_GRAPH_IS_REPRODUCIBLE,
    PROPERTY_IMPORT_TARGETS_RESOLVE,
    PROPERTY_TOPOLOGICAL_ORDER_IS_VALID,
)

#: 相等判据的浮点容差（逐位 / 整数相等时不走它）.
EQUALITY_TOLERANCE = 1e-12

#: 未解析的包内目标数的上界（一个都不许有）.
UNRESOLVED_CEILING = 0.0

#: 每个模块的下游闭包至少要含几个节点（**下界**：至少含自己）.
CLOSURE_FLOOR = 1.0

#: 最长下游闭包的下界（**下界**：至少存在一条边）.
DEPTH_FLOOR = 2.0


@dataclass(frozen=True)
class CrossCheck:
    """一次对账：来源、读数、判据（**含方向**：相等 / 上界 / 下界）."""

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
            raise ParameterError(f"对账 {self.name!r} 同时给了上界与下界：请只给一个方向。")
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
            raise NumericError(f"性质 {self.name!r} 标了'不适用'却又标了'通过'：两者必须分开。")
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
        """不通过时抛 :class:`errors.ParseError`（"拒绝交付"的那条路）."""
        if self.ok:
            return
        from smart_research_agent.import_graph.errors import ParseError

        failures = [outcome.line() for outcome in self.applicable if not outcome.passed]
        raise ParseError("性质检查未全部通过：" + "；".join(failures))

    def lines(self) -> tuple[str, ...]:
        """逐行文本（**先印不适用**）."""
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
# 七条性质
# --------------------------------------------------------------------------- #


def _resolve(graph: ModuleGraph | None) -> ModuleGraph:
    """缺省时现场建一张图."""
    return build_graph() if graph is None else graph


def check_graph_covers_all_modules(
    graph: ModuleGraph | None = None,
    *,
    expected: tuple[str, ...] | None = None,
) -> PropertyOutcome:
    """① 每一个 ``.py`` 都是图里的一个节点."""
    resolved = _resolve(graph)
    expected_names = expected_modules() if expected is None else expected
    missing = tuple(name for name in expected_names if name not in resolved.node_set)
    check = CrossCheck(
        name="模块缺失数",
        left="ModuleGraph.nodes",
        right="扫描到的 .py 模块名",
        reading=float(len(missing)),
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_GRAPH_COVERS_ALL_MODULES,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"节点 {resolved.node_count} 个 / 期望 {len(expected_names)} 个 | 缺失 {len(missing)}",
            "缺失的：" + ("、".join(missing) if missing else "无"),
        ),
        cross_check=check,
    )


def check_graph_is_reproducible(
    first: ModuleGraph | None = None,
    second: ModuleGraph | None = None,
    *,
    modules: tuple[ScannedModule, ...] | None = None,
) -> PropertyOutcome:
    """② 同一批文件两次构建图**逐位相同**."""
    left = build_graph(modules) if first is None else first
    right = build_graph(modules) if second is None else second
    diff = left.diff_count(right)
    check = CrossCheck(
        name="两次构建的差异项数",
        left="build_graph 第一次",
        right="build_graph 第二次",
        reading=float(diff),
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_GRAPH_IS_REPRODUCIBLE,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"差异项数 {diff}（0 = 逐位相同）",
            f"摘要 {left.digest()} / {right.digest()}",
        ),
        cross_check=check,
    )


def check_topological_order_is_valid(graph: ModuleGraph | None = None) -> PropertyOutcome:
    """③ 凝缩后的线性序合法（跨分量的边一律指向后面）."""
    resolved = _resolve(graph)
    positions = {key: index for index, key in enumerate(resolved.topological_order())}
    violations = sum(
        1
        for source, target in resolved.condensation_edges()
        if positions[source] >= positions[target]
    )
    check = CrossCheck(
        name="违反线性序的跨分量边数",
        left="ModuleGraph.condensation_edges()",
        right="ModuleGraph.topological_order()",
        reading=float(violations),
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_TOPOLOGICAL_ORDER_IS_VALID,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"凝缩边 {len(resolved.condensation_edges())} 条 | 违反 {violations} 条",
            f"线性序长度 {len(resolved.topological_order())}（= 分量数 {len(resolved.sccs())}）",
        ),
        cross_check=check,
    )


def _reach_within(
    graph: ModuleGraph, start: str, members: frozenset[str], direction: str
) -> frozenset[str]:
    """只沿"两端都在分量内"的边做可达（环的真伪要用**分量内部**的可达性判）."""
    seen: set[str] = {start}
    stack = [start]
    while stack:
        current = stack.pop()
        neighbours = graph.successors(current) if direction == DIRECTION_DOWN else graph.predecessors(current)
        for nxt in neighbours:
            if nxt in members and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return frozenset(seen)


def check_cycles_are_sound(graph: ModuleGraph | None = None) -> PropertyOutcome:
    """④ 报出来的每一个环都是真环（分量内任意两点互达）.

    "互达"只在**分量内部**看：一个环上的节点当然还能走到环外面去，
    但那不改变"它们是环"这件事——因此这里把可达性限制在成员集内。
    """
    resolved = _resolve(graph)
    cycles = resolved.cycles()
    unsound: list[str] = []
    for component in cycles:
        members = frozenset(component)
        base = component[0]
        if _reach_within(resolved, base, members, DIRECTION_DOWN) != members:
            unsound.append(base)
        elif _reach_within(resolved, base, members, DIRECTION_UP) != members:
            unsound.append(base)
    check = CrossCheck(
        name="不成立的环数",
        left="ModuleGraph.cycles()",
        right="分量内双向可达",
        reading=float(len(unsound)),
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_CYCLES_ARE_SOUND,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"环 {len(cycles)} 个 / 不成立 {len(unsound)} 个",
            "不成立的：" + ("、".join(unsound) if unsound else "无"),
        ),
        cross_check=check,
    )


def check_import_targets_resolve(
    modules: tuple[ScannedModule, ...] | None = None,
) -> PropertyOutcome:
    """⑤ **上界**：每一条包内 import 的目标都能匹配到模块或包."""
    bad = unresolved_of(modules)
    check = CrossCheck(
        name="未解析的包内目标数",
        left="ImportRef.unresolved",
        right=f"上界 {UNRESOLVED_CEILING:.0f}",
        reading=float(len(bad)),
        expected=UNRESOLVED_CEILING,
        exact=False,
        upper_bound=UNRESOLVED_CEILING,
    )
    preview = "、".join(f"{module} 里的 {raw!r}" for module, raw in bad[:3])
    return PropertyOutcome(
        name=PROPERTY_IMPORT_TARGETS_RESOLVE,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"未解析 {len(bad)} 条 ≤ 上界 {UNRESOLVED_CEILING:.0f}",
            "未解析的：" + (preview if preview else "无"),
        ),
        cross_check=check,
    )


def check_closures_include_self(graph: ModuleGraph | None = None) -> PropertyOutcome:
    """⑥ **下界**：每个模块的下游闭包至少含它自己."""
    resolved = _resolve(graph)
    fewest = float(min(len(resolved.closure(node, DIRECTION_DOWN)) for node in resolved.nodes))
    check = CrossCheck(
        name="最小的下游闭包大小",
        left="ModuleGraph.closure(node, 'down')",
        right=f"下界 {CLOSURE_FLOOR:.0f}",
        reading=fewest,
        expected=CLOSURE_FLOOR,
        exact=False,
        lower_bound=CLOSURE_FLOOR,
    )
    return PropertyOutcome(
        name=PROPERTY_CLOSURES_INCLUDE_SELF,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"最小闭包 {fewest:.0f} ≥ 下界 {CLOSURE_FLOOR:.0f}",
            resolved.line(),
        ),
        cross_check=check,
    )


def check_dependency_depth_is_positive(graph: ModuleGraph | None = None) -> PropertyOutcome:
    """⑦ **下界**：图里真的存在一条依赖链（最长下游闭包 >= 2）."""
    resolved = _resolve(graph)
    deepest = float(max(len(resolved.closure(node, DIRECTION_DOWN)) for node in resolved.nodes))
    check = CrossCheck(
        name="最长的下游闭包大小",
        left="ModuleGraph.closure(node, 'down')",
        right=f"下界 {DEPTH_FLOOR:.0f}",
        reading=deepest,
        expected=DEPTH_FLOOR,
        exact=False,
        lower_bound=DEPTH_FLOOR,
    )
    return PropertyOutcome(
        name=PROPERTY_DEPENDENCY_DEPTH_IS_POSITIVE,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"最长闭包 {deepest:.0f} ≥ 下界 {DEPTH_FLOOR:.0f}",
            f"边 {resolved.edge_count} 条 / 跨包 {len(resolved.cross_package_edges())} 条",
        ),
        cross_check=check,
    )


def check_all(
    *,
    graph: ModuleGraph | None = None,
    modules: tuple[ScannedModule, ...] | None = None,
    expected: tuple[str, ...] | None = None,
) -> PropertyReport:
    """一次跑完七条性质（顺序与 :data:`types.IMPORT_GRAPH_PROPERTIES` 一致）.

    七条性质**共用同一张图**，因此报告读的是同一时刻的状态。
    """
    resolved = build_graph(modules) if graph is None else graph
    outcomes = (
        check_graph_covers_all_modules(resolved, expected=expected),
        check_graph_is_reproducible(resolved, resolved),
        check_topological_order_is_valid(resolved),
        check_cycles_are_sound(resolved),
        check_import_targets_resolve(modules),
        check_closures_include_self(resolved),
        check_dependency_depth_is_positive(resolved),
    )
    names = {outcome.name for outcome in outcomes}
    extra = names - set(IMPORT_GRAPH_PROPERTIES)
    missing = set(IMPORT_GRAPH_PROPERTIES) - names
    if extra or missing:  # pragma: no cover - 只在有人改性质名单时触发
        raise GraphError(
            "性质名单与 types.IMPORT_GRAPH_PROPERTIES 不一致："
            f"多 {sorted(extra)}、缺 {sorted(missing)}。"
        )
    return PropertyReport(outcomes=outcomes)


def require_ok(report: PropertyReport | None = None) -> PropertyReport:
    """七条性质没全部通过时抛 :class:`ParseError`（"拒绝交付"的那条路）."""
    resolved = check_all() if report is None else report
    if not resolved.ok:
        resolved.require_ok()
    return resolved


__all__ = [
    "CLOSURE_FLOOR",
    "DEPTH_FLOOR",
    "EQUALITY_TOLERANCE",
    "UNRESOLVED_CEILING",
    "CrossCheck",
    "PropertyOutcome",
    "PropertyReport",
    "check_all",
    "check_closures_include_self",
    "check_cycles_are_sound",
    "check_dependency_depth_is_positive",
    "check_graph_covers_all_modules",
    "check_graph_is_reproducible",
    "check_import_targets_resolve",
    "check_topological_order_is_valid",
    "require_ok",
]
