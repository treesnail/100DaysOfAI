"""day088 离线演示：项目底层原理串联 —— 十二块拼图.

十一节，全部离线、全部确定性（不需要 API Key，也不装 transformers / torch / numpy）：

```text
1   十二块拼图总览：id / 层 / 支撑应用 / 实现落点 / 来源天 / 现场读数
2   第一层：数学地基（可微检索 / 余弦 = 归一化点积）
3   第二层：注意力与结构（行分布 / 因果掩码 / 分头 / 位置）
4   第三层：表征与检索（方向相似 / 排序对照 / 缓存复用）
5   第四层：推理与部署（缓存公式 / 量化上界 / 预算拆分）
6   覆盖：四层 × 六应用（有没有孤岛）
7   七条性质与两类判据（含两条跨天对账）
8   失败族与缺席的 GradientError（理由第四次换了一条）
9   十条笔记
10  五条边界与与后续的接缝
11  第十一节附：分享提纲（四节、总时长、demo 存在性）
```

运行方式::

    cd day088/源码/smart-research-agent
    python scripts/principle_map_demo.py

产出 ``outputs/principle_map_demo.txt``（在 .gitignore 里）。
"""

from __future__ import annotations

import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.principle_map import (  # noqa: E402
    ABSENT_FAMILY,
    ABSENT_FAMILY_REASON,
    FAMILY_OUTCOMES,
    PRINCIPLE_BOUNDARIES,
    PRINCIPLE_NOTES,
    PRINCIPLE_NOTES_ORDER,
    claims,
    errors,
    graph,
    outline,
    study,
    types,
    verify,
)

OUTPUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "principle_map_demo.txt"


class Report:
    """攒行 + 落盘（**不做任何计算**）."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def section(self, index: int, title: str) -> None:
        self.lines.append("")
        self.lines.append(f"== {index}. {title}")

    def add(self, *texts: str) -> None:
        self.lines.extend(texts)

    def flush(self) -> None:
        text = "\n".join(self.lines)
        print(text)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_FILE.write_text(text + "\n", encoding="utf-8")


_CASE: dict[str, object] = {}


def ensure_graph() -> graph.PrincipleGraph:
    """构建一次那张图（**单次计算**：探针只跑一遍）."""
    if "graph" not in _CASE:
        _CASE["graph"] = graph.build_graph()
    return _CASE["graph"]  # type: ignore[return-value]


def section_overview(report: Report) -> None:
    """第 1 节：十二块拼图总览."""
    report.section(1, "十二块拼图总览（id / 层 / 支撑应用 / 实现落点 / 来源天 / 读数）")
    for row in study.principle_rows(ensure_graph()):
        report.add("  " + row.line())
    report.add(f"  {ensure_graph().coverage().line()}")


def section_layer(report: Report, index: int, layer: str, title: str) -> None:
    """第 2~5 节：逐层把原理与读数印出来."""
    report.section(index, title)
    for item in claims.by_layer(layer):
        evidence = ensure_graph().evidence_of(item.id)
        report.add(f"  [{item.source_day}] {item.id}")
        report.add(f"      落点：{item.artifact}")
        report.add(f"      {evidence.line()}")
        report.add(f"      {evidence.note}")


def section_coverage(report: Report) -> None:
    """第 6 节：覆盖（四层 × 六应用）."""
    report.section(6, "覆盖：四层 × 六应用（有没有孤岛）")
    report.add("  按层：")
    for row in study.layer_rows(ensure_graph()):
        report.add("    " + row.line())
    report.add("  按应用：")
    for row in study.application_rows(ensure_graph()):
        report.add("    " + row.line())
    report.add(f"  有支撑的应用：{list(ensure_graph().supported)}")
    report.add(f"  没有入边的应用：{list(ensure_graph().unsupported)}")
    report.add(f"  孤儿原理：{list(ensure_graph().orphans)}")
    report.add(f"  {graph.coverage_report(ensure_graph()).line()}")


def section_properties(report: Report) -> None:
    """第 7 节：七条性质与两类判据."""
    report.section(7, "七条性质与两类判据（含两条跨天对账）")
    report_result = verify.check_all(ensure_graph())
    for line in report_result.lines():
        report.add("  " + line)
    report.add(f"  全部通过：{report_result.ok}")


def section_families(report: Report) -> None:
    """第 8 节：失败族与缺席的那一族."""
    report.section(8, "七个失败族与缺席的 GradientError（理由第四次换了一条）")
    for name, outcome in FAMILY_OUTCOMES.items():
        report.add(f"  {name:<15} {outcome}")
    report.add("")
    report.add(f"  连续缺席的那一族：{ABSENT_FAMILY}")
    report.add(f"  理由：{ABSENT_FAMILY_REASON}")
    report.add(f"  本族基类：{errors.BridgeError.__name__}（继承 {errors.BridgeError.__mro__[1].__name__}）")


def section_notes(report: Report) -> None:
    """第 9 节：十条笔记."""
    report.section(9, "十条笔记")
    for line in study.note_lines():
        report.add("  " + line)
    report.add("")
    report.add(f"  十条笔记的键：{list(PRINCIPLE_NOTES)}")
    report.add(f"  顺序表一致：{PRINCIPLE_NOTES_ORDER == tuple(PRINCIPLE_NOTES)}")


def section_boundaries(report: Report) -> None:
    """第 10 节：五条边界与接缝."""
    report.section(10, "五条边界（**这一课明确不承诺的事**）与后续接缝")
    for index, boundary in enumerate(PRINCIPLE_BOUNDARIES, start=1):
        report.add(f"  {index}. {boundary}")
    report.add("")
    report.add(
        "  接缝：day073（数学地基）→ day075~079（注意力与结构）→ day041/064（表征与检索）→ "
        "day087（高效推理）；本课把它们的读数对齐成一张图。"
    )
    report.add(
        "  下游：day099（结业项目）会把这张图当成部署前的第一张检查单——"
        "任何一条没有落点的能力，都会在那一天以'跑不起来'的形式出现。"
    )


def section_outline(report: Report) -> None:
    """第 11 节附：分享提纲."""
    report.section(11, "第十一节附：分享提纲（原理 → 应用）")
    sections = outline.build_outline(ensure_graph())
    for line in outline.outline_lines(sections):
        report.add("  " + line)
    report.add("")
    report.add(f"  总时长 {outline.outline_minutes(sections)} 分钟")
    report.add(f"  缺失 demo：{list(outline.missing_demos(sections))}")
    report.add(f"  demo 脚本：{list(outline.demo_scripts())}")
    document = outline.render_document(ensure_graph())
    report.add(f"  原理文档：{len(document)} 字符、缺失命题 {list(outline.missing_principles(document))}")
    report.add(f"  层次的枚举：{list(types.LAYERS)}")
    report.add(f"  应用的枚举：{list(types.APPLICATIONS)}")


def main() -> None:
    """跑完十一节并落盘."""
    report = Report()
    section_overview(report)
    section_layer(report, 2, types.LAYER_MATH, "第一层：数学地基 —— 为什么『看哪里』能被学出来")
    section_layer(report, 3, types.LAYER_ATTENTION, "第二层：注意力与结构 —— 分布、掩码、分头与位置")
    section_layer(report, 4, types.LAYER_REPRESENTATION, "第三层：表征与检索 —— 方向、排序与缓存复用")
    section_layer(report, 5, types.LAYER_INFERENCE, "第四层：推理与部署 —— 缓存、量化与预算三笔账")
    section_coverage(report)
    section_properties(report)
    section_families(report)
    section_notes(report)
    section_boundaries(report)
    section_outline(report)
    report.flush()


if __name__ == "__main__":
    main()
