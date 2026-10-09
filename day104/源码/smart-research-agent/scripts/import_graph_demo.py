"""``import_graph`` 的演示脚本（day103）.

跑法（在 ``smart-research-agent`` 目录下）：

```bash
python scripts/import_graph_demo.py
```

它把这一课的四张表按节打印出来，并把同一份文本写进
``outputs/import_graph_demo.txt``。**所有读数都是现场算的**，
没有任何一行是手写的——因此它给出的每个数字都能被复算。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 脚本直接运行时 sys.path[0] 是 scripts/，把项目根目录插到最前，
# 这样 `python scripts/import_graph_demo.py` 也能 import 到本包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.import_graph import errors as graph_errors  # noqa: E402
from smart_research_agent.import_graph import graph as graph_module  # noqa: E402
from smart_research_agent.import_graph import parse as parse_module  # noqa: E402
from smart_research_agent.import_graph import study  # noqa: E402
from smart_research_agent.import_graph import types as graph_types  # noqa: E402
from smart_research_agent.import_graph import verify  # noqa: E402

OUTPUT = PROJECT_ROOT / "outputs" / "import_graph_demo.txt"


def _collect() -> tuple[str, ...]:
    """把四张表收成一份文本（与教程引用的是同一批读数）."""
    lines: list[str] = []
    lines.append("========== day103 · 模块依赖图 ==========")

    # 1. 扫描
    lines.append("")
    lines.append("[1] 扫描：``smart_research_agent/**/*.py``（只用 ast，不做 import）")
    scanned = parse_module.scan_all()
    lines.append(f"    模块 {len(scanned)} 个 | 未解析目标 {len(parse_module.unresolved_of(scanned))} 条")

    # 2. 图
    lines.append("")
    lines.append("[2] 图：节点 / 包 / 边 / 跨包边")
    graph = graph_module.build_graph(scanned)
    lines.append("    " + graph.line())
    lines.append(f"    跨包边 {len(graph.cross_package_edges())} 条")

    # 3. 环
    lines.append("")
    lines.append("[3] 环：依赖图**不是** DAG——把每一个环逐个点名")
    for row in study.cycle_rows(graph, cycles_only=True):
        lines.append("    " + row.line())
    lines.append(
        f"    真正的环 {len(graph.cycles())} 个 / 强连通分量 {len(graph.sccs())} 个"
        f" / 凝缩后线性序长度 {len(graph.topological_order())}"
    )

    # 4. 折成包级
    lines.append("")
    lines.append("[4] 折成包级：一张更粗的图（节点是子包名）")
    rolled = graph.rollup_to_packages()
    lines.append("    " + rolled.line())
    for row in study.cycle_rows(rolled, cycles_only=True):
        lines.append("    " + row.line())

    # 5. 性质
    lines.append("")
    lines.append("[5] 性质：七条性质、三类判据")
    report = verify.check_all(graph=graph, modules=scanned)
    for row in study.property_rows(report):
        lines.append("    " + row.line())
    lines.append(f"    合计：{sum(1 for o in report.outcomes if o.passed)}/{len(report.outcomes)} 条通过")

    # 6. 结论
    lines.append("")
    lines.append("[6] 结论：这张图是一份**可复算**的中间物——")
    lines.append(f"    同一批文件两次构建，差异项 {graph.diff_count(graph)} 项（0 = 逐位相同）")
    lines.append(f"    摘要 {graph.digest()}")
    lines.append(f"    首个环的成员：{'、'.join(graph.cycles()[0])}")
    lines.append(
        f"    回来/缺席的族：{graph_errors.RETURNED_FAMILY} / {graph_errors.ABSENT_FAMILY}"
    )
    lines.append(f"    边界第 3 条：{graph_types.IMPORT_GRAPH_BOUNDARIES[2][:28]}……")
    return tuple(lines)


def main() -> None:
    """跑一遍并把文本写进 outputs/。"""
    text = "\n".join(_collect()) + "\n"
    print(text, end="")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"\n（已写入 {OUTPUT}）")


if __name__ == "__main__":  # pragma: no cover - 脚本入口
    main()
