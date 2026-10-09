"""``graph``：把"谁 import 了谁"编成一张**有向图**（day103）.

图只有三件事：

```text
建图    每个模块一个节点，每条 import 一条边（自环不算边）
找环    强连通分量（SCC）——**依赖图天生可能不是 DAG**
排序    凝缩（把每个环缩成一个点）之后，才存在一个确定的线性序
```

## 一、今天最值钱的一句话

> **依赖图天生可能不是 DAG：450 个模块里就有 4 个环。
> 谁都不说清"这个线性序对谁而言"，报告里那两个数字就会互相矛盾。**

因此本课把"环"当成一个**读数**印出来（而不是让一条"图必须无环"的性质把它删掉），
把"线性序"定义在**凝缩图**上（顺序的长度是"分量数"，不是"模块数"）。

## 二、一条纪律：**并列**要有一个确定的顺序

```text
SCC 的成员排序、SCC 之间的排序、Kahn 排队的取点顺序——三处都显式排序
comparable() 里没有任何集合 / 字典：因此 repr 稳定、摘要才有意义
```

与 day101 的倒排表、day102 的台账同源：把"集合 → 序列"的地方显式排序，
"两次构建逐位相同"就从"希望如此"变成"不可能不如此"。

## 三、一个真实的环（本课最值钱的一处读数）

```text
api/app.py            from smart_research_agent.tools.image_analysis import ImageAnalysisTool   ← 顶层导入
tools/image_analysis.py   （函数体内）from smart_research_agent.api.app import default_llm       ← 延迟导入
```

两条边合起来是一个环。它被一个**函数体内的延迟 import** 挡在了运行期之外，
但静态依赖图看得见它——这正是本课"读 import **语句**、不执行 import"的价值。

## 四、与既有包的接缝

- **上游**：:mod:`import_graph.parse`（模块索引与 import 目标）、:mod:`import_graph.types`（两类边）；
- **下游**：:mod:`import_graph.verify` 在它上面检查七条性质，
  :mod:`import_graph.study` 打印图的表。
"""

from __future__ import annotations

import hashlib
import heapq
from dataclasses import dataclass
from typing import Any

from smart_research_agent.import_graph.errors import CycleError, GraphBuildError, ParameterError
from smart_research_agent.import_graph.parse import ScannedModule, module_index, scan_all
from smart_research_agent.import_graph.types import (
    DIRECTION_DOWN,
    DIRECTION_UP,
    EDGE_KIND_ABSOLUTE,
    PACKAGE_NAME,
    require_direction,
    require_edge_kind,
)

#: 图摘要取前多少位十六进制（与 day099~day102 同一口径）.
GRAPH_DIGEST_LENGTH = 16


@dataclass(frozen=True)
class Edge:
    """一条依赖边：源模块 + 目标模块 + 边种类."""

    source: str
    target: str
    kind: str = EDGE_KIND_ABSOLUTE

    def __post_init__(self) -> None:
        if not self.source or not self.target:
            raise ParameterError("边的两个端点都不能为空。")
        require_edge_kind(self.kind)

    @property
    def key(self) -> tuple[str, str, str]:
        """排序键（源、目标、种类）."""
        return (self.source, self.target, self.kind)

    @property
    def is_cross_package(self) -> bool:
        """是不是一条跨子包的边."""
        return _package_of(self.source) != _package_of(self.target)

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {"source": self.source, "target": self.target, "kind": self.kind}

    def line(self) -> str:
        """一行读数：``api.app → tools.image_analysis | absolute``."""
        return f"{self.source} → {self.target} | {self.kind}"


def _package_of(node: str) -> str:
    """一个节点的顶层子包名（``smart_research_agent.api.app`` → ``api``）.

    根包 ``smart_research_agent``（自身只有一个点）返回它自己。
    """
    parts = node.split(".")
    return parts[1] if len(parts) > 1 else node


@dataclass(frozen=True)
class ModuleGraph:
    """一张依赖图：节点（模块，排序唯一）+ 包节点 + 边（排序唯一）.

    ``package_of`` 把任何节点折到它的顶层子包上——这正是 :meth:`rollup_to_packages`
    能给出"一张更粗的图"的原因。
    """

    nodes: tuple[str, ...]
    packages: tuple[str, ...]
    edges: tuple[Edge, ...]

    def __post_init__(self) -> None:
        if not self.nodes:
            raise GraphBuildError("图不能没有节点。")
        if list(self.nodes) != sorted(self.nodes) or len(set(self.nodes)) != len(self.nodes):
            raise GraphBuildError(f"节点必须是一份排序且不重复的名单：{len(self.nodes)} 个。")
        unknown_packages = set(self.packages) - set(self.nodes)
        if unknown_packages:
            raise GraphBuildError(f"包节点不在节点集里：{sorted(unknown_packages)}。")
        keys = [edge.key for edge in self.edges]
        if keys != sorted(keys):
            raise GraphBuildError(
                "边没有排序：并列的不确定性会让'两次构建逐位相同'永远失败。"
            )
        if len(set(keys)) != len(keys):
            raise GraphBuildError("边里有重复项。")
        unknown = {node for edge in self.edges for node in (edge.source, edge.target)} - set(self.nodes)
        if unknown:
            raise GraphBuildError(
                f"边的端点在节点集里找不到：{sorted(unknown)}——"
                "一条指向不存在的模块的边会让'覆盖完整'这条性质永远说不清。"
            )

    # ------------------------------------------------------------------ 只读视图

    @property
    def node_set(self) -> frozenset[str]:
        """节点集合（查表用）."""
        return frozenset(self.nodes)

    @property
    def node_count(self) -> int:
        """节点数."""
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        """边数."""
        return len(self.edges)

    @property
    def package_count(self) -> int:
        """包节点数（``__init__.py`` 的那些）."""
        return len(self.packages)

    def package_of(self, node: str) -> str:
        """一个节点的顶层子包名（不在图里当场拒绝）."""
        if node not in self.node_set:
            raise ParameterError(f"图里没有节点 {node!r}。")
        return _package_of(node)

    def package_groups(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """按顶层子包分组（组名排序，组内成员排序）."""
        groups: dict[str, list[str]] = {}
        for node in self.nodes:
            groups.setdefault(_package_of(node), []).append(node)
        return tuple((name, tuple(sorted(groups[name]))) for name in sorted(groups))

    def successors(self, node: str) -> tuple[str, ...]:
        """下游邻居（我依赖谁，排序）."""
        if node not in self.node_set:
            raise ParameterError(f"图里没有节点 {node!r}。")
        return tuple(sorted(edge.target for edge in self.edges if edge.source == node))

    def predecessors(self, node: str) -> tuple[str, ...]:
        """上游邻居（谁依赖我，排序）."""
        if node not in self.node_set:
            raise ParameterError(f"图里没有节点 {node!r}。")
        return tuple(sorted(edge.source for edge in self.edges if edge.target == node))

    def edges_of(self, node: str) -> tuple[Edge, ...]:
        """与一个节点相连的边（排序）."""
        if node not in self.node_set:
            raise ParameterError(f"图里没有节点 {node!r}。")
        return tuple(edge for edge in self.edges if node in (edge.source, edge.target))

    def cross_package_edges(self) -> tuple[Edge, ...]:
        """跨子包的边（排序）."""
        return tuple(edge for edge in self.edges if edge.is_cross_package)

    # ------------------------------------------------------------------ 强连通分量

    def _adjacency(self) -> dict[str, tuple[str, ...]]:
        """邻接表（排序，确定性）."""
        table: dict[str, list[str]] = {node: [] for node in self.nodes}
        for edge in self.edges:
            table[edge.source].append(edge.target)
        return {node: tuple(sorted(set(targets))) for node, targets in table.items()}

    def _reverse_adjacency(self) -> dict[str, tuple[str, ...]]:
        """反图邻接表（排序，确定性）."""
        table: dict[str, list[str]] = {node: [] for node in self.nodes}
        for edge in self.edges:
            table[edge.target].append(edge.source)
        return {node: tuple(sorted(set(sources))) for node, sources in table.items()}

    @staticmethod
    def _finish_order(adjacency: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
        """Kosaraju 第一遍：迭代 DFS 求完成时间序（节点名升序作起点顺序）."""
        visited: set[str] = set()
        order: list[str] = []
        for start in sorted(adjacency):
            if start in visited:
                continue
            stack: list[tuple[str, int]] = [(start, 0)]
            visited.add(start)
            while stack:
                node, position = stack.pop()
                targets = adjacency.get(node, ())
                if position < len(targets):
                    stack.append((node, position + 1))
                    nxt = targets[position]
                    if nxt not in visited:
                        visited.add(nxt)
                        stack.append((nxt, 0))
                else:
                    order.append(node)
        return tuple(order)

    def sccs(self) -> tuple[tuple[str, ...], ...]:
        """强连通分量（**确定性**：分量内成员排序，分量之间按首成员排序）.

        只包含成员数 >= 1 的全部分量；成员数 > 1 的那些才是"环"。
        """
        adjacency = self._adjacency()
        reverse = self._reverse_adjacency()
        visited: set[str] = set()
        components: list[tuple[str, ...]] = []
        for start in reversed(self._finish_order(adjacency)):
            if start in visited:
                continue
            members: list[str] = []
            stack = [start]
            visited.add(start)
            while stack:
                node = stack.pop()
                members.append(node)
                for nxt in reverse.get(node, ()):
                    if nxt not in visited:
                        visited.add(nxt)
                        stack.append(nxt)
            components.append(tuple(sorted(members)))
        return tuple(sorted(components, key=lambda member: member[0]))

    def cycles(self) -> tuple[tuple[str, ...], ...]:
        """真正的环（成员数 > 1 的强连通分量）."""
        return tuple(component for component in self.sccs() if len(component) > 1)

    def component_key(self, node: str) -> str:
        """一个节点所属分量的小写键（分量里的最小成员名）."""
        for component in self.sccs():
            if node in component:
                return component[0]
        raise ParameterError(f"图里没有节点 {node!r}。")

    def condensation_edges(self) -> tuple[tuple[str, str], ...]:
        """凝缩图的边（分量键 → 分量键，去掉自环，排序去重）."""
        keys = {node: self.component_key(node) for node in self.nodes}
        pairs = {
            (keys[edge.source], keys[edge.target])
            for edge in self.edges
            if keys[edge.source] != keys[edge.target]
        }
        return tuple(sorted(pairs))

    def topological_order(self) -> tuple[str, ...]:
        """凝缩图的线性序（分量键的序列，**确定性**）.

        Kahn 算法 + 最小堆取点：并列时总是先取字典序最小的分量。
        """
        components = self.sccs()
        keys = sorted(component[0] for component in components)
        incoming: dict[str, int] = {key: 0 for key in keys}
        outgoing: dict[str, list[str]] = {key: [] for key in keys}
        for source, target in self.condensation_edges():
            outgoing[source].append(target)
            incoming[target] += 1
        ready = [key for key in keys if incoming[key] == 0]
        heapq.heapify(ready)
        order: list[str] = []
        while ready:
            key = heapq.heappop(ready)
            order.append(key)
            for nxt in sorted(outgoing[key]):
                incoming[nxt] -= 1
                if incoming[nxt] == 0:
                    heapq.heappush(ready, nxt)
        if len(order) != len(keys):  # pragma: no cover - 凝缩图按构造一定是 DAG
            raise CycleError(
                f"凝缩图里还剩 {len(keys) - len(order)} 个分量排不进线性序——"
                "凝缩图按构造应当是 DAG，出现这种情况说明分量算错了。"
            )
        return tuple(order)

    def order_index(self) -> tuple[tuple[str, int], ...]:
        """每个节点的分量在线性序里的位置（键 → 位置）."""
        positions = {key: index for index, key in enumerate(self.topological_order())}
        keys = {node: self.component_key(node) for node in self.nodes}
        return tuple((node, positions[keys[node]]) for node in self.nodes)

    # ------------------------------------------------------------------ 闭包

    def closure(self, node: str, direction: str = DIRECTION_DOWN) -> tuple[str, ...]:
        """一个模块的闭包（含它自己，排序）.

        ``down`` = 我依赖谁（沿边）；``up`` = 谁依赖我（逆边）。
        """
        require_direction(direction)
        if node not in self.node_set:
            raise ParameterError(f"图里没有节点 {node!r}。")
        adjacency = self._adjacency() if direction == DIRECTION_DOWN else self._reverse_adjacency()
        seen: set[str] = {node}
        stack = [node]
        while stack:
            current = stack.pop()
            for nxt in adjacency.get(current, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return tuple(sorted(seen))

    # ------------------------------------------------------------------ 折成包级

    def rollup_to_packages(self) -> ModuleGraph:
        """把模块折成顶层子包，得到一张更粗的图（节点是子包名）.

        它更容易看出"哪两个包互相咬住"，但会丢掉"到底哪两个文件在互相咬"。
        """
        nodes = tuple(sorted({_package_of(node) for node in self.nodes}))
        pairs: set[tuple[str, str, str]] = set()
        for edge in self.edges:
            source = _package_of(edge.source)
            target = _package_of(edge.target)
            if source != target:
                pairs.add((source, target, edge.kind))
        edges = tuple(Edge(source, target, kind) for source, target, kind in sorted(pairs))
        return ModuleGraph(nodes=nodes, packages=nodes, edges=edges)

    # ------------------------------------------------------------------ 复算口径

    def comparable(self) -> tuple[Any, ...]:
        """**只含确定性字段**的元组（两张图逐位比较它）."""
        return (self.nodes, self.packages, tuple(edge.key for edge in self.edges))

    def digest(self) -> str:
        """图摘要（两次构建摘要相同 ⇔ 逐位相同）."""
        return hashlib.sha256(repr(self.comparable()).encode("utf-8")).hexdigest()[
            :GRAPH_DIGEST_LENGTH
        ]

    def diff_count(self, other: ModuleGraph) -> int:
        """与另一张图在复算口径上**差了几项**（0 = 逐位相同）."""
        left = self.comparable()
        right = other.comparable()
        return sum(1 for a, b in zip(left, right, strict=True) if a != b)

    def to_dict(self) -> dict[str, Any]:
        """摊平成可 JSON 化的字段."""
        return {
            "nodes": self.node_count,
            "packages": self.package_count,
            "edges": self.edge_count,
            "cycles": len(self.cycles()),
            "digest": self.digest(),
        }

    def line(self) -> str:
        """一行读数：``图：450 个节点 | 55 个包 | 1983 条边 | 4 个环 | 摘要 xxxx``."""
        return (
            f"图：{self.node_count} 个节点 | {self.package_count} 个包 | {self.edge_count} 条边"
            f" | {len(self.cycles())} 个环 | 摘要 {self.digest()}"
        )


def build_graph(modules: tuple[ScannedModule, ...] | None = None) -> ModuleGraph:
    """把解析结果编成一张 :class:`ModuleGraph`（**确定性**）.

    ``modules`` 缺省时现场扫一遍；测试可以注入一组小模块。
    一个模块 import 它自己**不算一条边**（自环是退化情形，不进图）。
    """
    resolved = scan_all() if modules is None else modules
    nodes = tuple(sorted(module.module for module in resolved))
    packages = tuple(sorted(module.module for module in resolved if module.is_package))
    edges: set[Edge] = set()
    for module in resolved:
        for ref in module.refs:
            if ref.target is None or ref.target == module.module:
                continue
            edges.add(Edge(source=module.module, target=ref.target, kind=ref.kind))
    return ModuleGraph(
        nodes=nodes,
        packages=packages,
        edges=tuple(sorted(edges, key=lambda edge: edge.key)),
    )


def graph_digest(graph: ModuleGraph | None = None) -> str:
    """图摘要（缺省时现场建一张图）."""
    resolved = build_graph() if graph is None else graph
    return resolved.digest()


def require_reproducible(first: ModuleGraph, second: ModuleGraph) -> ModuleGraph:
    """两张图不一致时抛 :class:`GraphBuildError`（"拒绝交付"的那条路）."""
    diff = first.diff_count(second)
    if diff == 0:
        return first
    raise GraphBuildError(
        f"同一批文件编出的两张图差了 {diff} 项——"
        "图里混进了未固定的量（集合序 / 字典序），它还不是一份可以被别人重算的中间物。"
    )


def require_acyclic(graph: ModuleGraph | None = None) -> ModuleGraph:
    """图里有环时抛 :class:`CycleError`（"你要线性序、但图里有环"的那条路）."""
    resolved = build_graph() if graph is None else graph
    cycles = resolved.cycles()
    if cycles:
        rendered = "、".join("{" + "、".join(component) + "}" for component in cycles[:3])
        raise CycleError(
            f"图里有 {len(cycles)} 个环：{rendered}——"
            "有环的图没有线性序，只有它的**凝缩图**有；"
            "要么先凝缩（用 topological_order），要么改层去真正拆掉这个环。"
        )
    return resolved


def expected_packages(graph: ModuleGraph | None = None) -> tuple[str, ...]:
    """读取一份图的顶层子包名单（排序）."""
    resolved = build_graph() if graph is None else graph
    return tuple(name for name, _members in resolved.package_groups())


def graph_lines(graph: ModuleGraph | None = None, *, limit: int | None = None) -> tuple[str, ...]:
    """把图逐行印出来（可选截断）."""
    from smart_research_agent.import_graph.types import require_positive_int

    resolved = build_graph() if graph is None else graph
    chosen = resolved.edges if limit is None else resolved.edges[: require_positive_int("limit", limit)]
    lines = ["依赖图（源 → 目标 | 种类）："]
    lines.extend("  " + edge.line() for edge in chosen)
    lines.append("  " + resolved.line())
    return tuple(lines)


__all__ = [
    "GRAPH_DIGEST_LENGTH",
    "Edge",
    "ModuleGraph",
    "build_graph",
    "expected_packages",
    "graph_digest",
    "graph_lines",
    "require_acyclic",
    "require_reproducible",
]
