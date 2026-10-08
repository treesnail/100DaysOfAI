#!/usr/bin/env python
"""day099 离线演示：结业项目整合（一）——把八项能力装成一条可复算的链.

十一节，全部离线、全部确定性（不需要 API Key，也不联网）：

```text
1   固定底座：6 条语料 / 64 维字符 n-gram 编码器 / MockLLM 脚本
2   端到端九段：guard → plan → retrieve → pack → generate → ground → evaluate → account → trace
3   复算：同一输入两次运行逐位相同（摘要与差异项数）
4   八项能力的清单与覆盖：能力 → 承担子包（importlib 解析）
5   十二个候选子包：被认领 8 个 / 无人认领 4 个
6   七条性质与三类判据（相等 / 上界 / 下界）
7   五张表：能力 / 清单 / 阶段 / 性质 / 文档
8   失败族：七个族各自"该谁去修" + 回来的族与缺席的族
9   文档渲染：从清单渲染「最终版 README + 架构文档」并检查覆盖
10  十条笔记
11  五条边界与与既有包的接缝
```

运行方式::

    cd day099/源码/smart-research-agent
    python scripts/capstone_demo.py

产出：

```text
outputs/capstone_demo.txt                 本脚本的完整输出（在 .gitignore 里）
outputs/capstone/README.capstone.md       渲染出的「最终版 README」
outputs/capstone/architecture.capstone.md 渲染出的「架构文档」
```

**不覆盖仓库根的那份 ``README.md``**：本课只新增文件，既有文件一行未改。
"""

from __future__ import annotations

import logging
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.capstone import (  # noqa: E402
    ABSENT_FAMILY,
    ABSENT_FAMILY_REASON,
    FAMILY_OUTCOMES,
    RETURNED_FAMILY,
    RETURNED_FAMILY_REASON,
    adapters,
    assembly,
    document,
    manifest,
    study,
    types,
    verify,
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "capstone_demo.txt"
DOC_OUTPUT_DIR = OUTPUT_DIR / "capstone"


class Report:
    """攒行 + 落盘（**不做任何计算**）."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def section(self, index: int, title: str) -> None:
        """开一节并打印标题."""
        self.lines.append("")
        self.lines.append(f"== {index}. {title}")

    def add(self, *texts: str) -> None:
        """往这一节里追加若干行."""
        self.lines.extend(texts)

    def flush(self) -> None:
        """把攒好的文本同时打印到终端并写入 ``outputs/capstone_demo.txt``."""
        text = "\n".join(self.lines)
        print(text)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_FILE.write_text(text + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# 第 1 节：固定底座
# --------------------------------------------------------------------------- #


def section_1_substrate(report: Report) -> None:
    """第 1 节：打印固定底座的现状（语料 / 编码器 / 模型 / 金标准）."""
    report.section(1, "固定底座（离线、确定性：语料写死 / 编码器确定性 / 模型按脚本）")
    substrate = adapters.build_substrate()
    described = substrate.describe()
    report.add(f"  {adapters.ADAPTER_BOUNDARY}")
    report.add(f"  问题：{described['question']}")
    report.add(
        f"  语料：{described['corpus']} 条记录 / 维度 {described['dimension']} / "
        f"金标准 {described['gold']}"
    )
    report.add(f"  编码器：{type(substrate.embedding).__name__}（确定性字符 n-gram）")
    report.add(f"  模型：{described['model']}（脚本：规划 JSON + 一段带引用的答案）")
    report.add(f"  {described['lexical']}")
    report.add("  六条语料：")
    for record_id, text, meta in adapters.CORPUS:
        report.add(f"    {record_id:<4} {meta['source']:<14} {text}")


# --------------------------------------------------------------------------- #
# 第 2 节：端到端九段
# --------------------------------------------------------------------------- #


def section_2_stages(report: Report, run_result: assembly.SystemRun) -> None:
    """第 2 节：把一次运行的九段逐行印出来."""
    report.section(2, "端到端九段（guard → plan → retrieve → pack → generate → ground → evaluate → account → trace）")
    for spec in types.stage_specs():
        report.add(f"  {spec.index}. [{spec.key}] {spec.description}")
    report.add("")
    for line in run_result.lines():
        report.add("  " + line)


# --------------------------------------------------------------------------- #
# 第 3 节：复算
# --------------------------------------------------------------------------- #


def section_3_reproducible(
    report: Report, first: assembly.SystemRun, second: assembly.SystemRun
) -> None:
    """第 3 节：同一输入两次运行逐位相同（本课最硬的一条证据）."""
    report.section(3, "复算：同一输入两次运行**逐位相同**（复算口径里没有时间与 uuid）")
    report.add(f"  第一次摘要：{first.digest()}")
    report.add(f"  第二次摘要：{second.digest()}")
    report.add(
        f"  逐位相同：{first.is_identical_to(second)} | 差异项数 {first.diff_count(second)}"
        "（0 = 逐位相同）"
    )
    report.add(
        "  口径（comparable 的字段）：问题 / 九段记录 / 答案 / 四条指标 / 费用 / token / span 条数"
    )
    report.add(
        "  **不含**：时间戳、uuid、耗时、日志文本——把它们放进读数会让第二条性质永远失败"
    )


# --------------------------------------------------------------------------- #
# 第 4 节：能力清单与覆盖
# --------------------------------------------------------------------------- #


def section_4_capabilities(report: Report, built: manifest.Manifest) -> None:
    """第 4 节：八项能力 → 承担子包（importlib 解析出来的）."""
    report.section(4, "八项能力的清单与覆盖（能力 → 承担子包，importlib 说了算）")
    for row in study.capability_rows(built):
        report.add("  " + row.line())
    report.add("")
    report.add("  " + built.line())


# --------------------------------------------------------------------------- #
# 第 5 节：十二个候选子包
# --------------------------------------------------------------------------- #


def section_5_subpackages(report: Report, built: manifest.Manifest) -> None:
    """第 5 节：十二个候选子包（被认领 / 无人认领）."""
    report.section(5, "十二个候选子包（在场 / 公开符号数 / 被谁认领）")
    for row in study.manifest_rows(built):
        report.add("  " + row.line())
    report.add("")
    report.add(f"  被认领（{len(built.claimed)}）：{'、'.join(built.claimed)}")
    report.add(
        f"  无人认领（{len(built.unclaimed)}）：{'、'.join(built.unclaimed)}"
        "（已交付，但不属于这 8 项能力清单）"
    )


# --------------------------------------------------------------------------- #
# 第 6 节：七条性质与三类判据
# --------------------------------------------------------------------------- #


def section_6_properties(report: Report, outcome: verify.PropertyReport) -> None:
    """第 6 节：七条性质逐条（判据类别与读数一起印）."""
    report.section(6, "七条性质与三类判据（相等 / 上界 / 下界）")
    for row in study.property_rows(outcome):
        report.add("  " + row.line())
    report.add("")
    report.add(f"  全部通过：{outcome.ok}")
    for criterion, description in types.CRITERION_DESCRIPTIONS.items():
        count = sum(1 for spec in types.property_specs() if spec.criterion == criterion)
        report.add(f"  [{criterion}] {count} 条：{description}")


# --------------------------------------------------------------------------- #
# 第 7 节：五张表
# --------------------------------------------------------------------------- #


def section_7_tables(
    report: Report,
    run_result: assembly.SystemRun,
    built: manifest.Manifest,
    documents: dict[str, str],
) -> None:
    """第 7 节：一次跑完五张表."""
    report.section(7, "五张表（能力 / 清单 / 阶段 / 性质 / 文档）")
    for line in study.study_lines(run_result, built, documents):
        report.add("  " + line)


# --------------------------------------------------------------------------- #
# 第 8 节：失败族
# --------------------------------------------------------------------------- #


def section_8_families(report: Report) -> None:
    """第 8 节：七个失败族 + 回来的族与缺席的族."""
    report.section(8, "失败族（按「该谁去修」分）")
    for name, outcome in FAMILY_OUTCOMES.items():
        report.add(f"  {name:<16} → {outcome}")
    report.add("")
    report.add(f"  回来的族：{RETURNED_FAMILY}")
    report.add(f"    理由：{RETURNED_FAMILY_REASON}")
    report.add(f"  缺席的族：{ABSENT_FAMILY}")
    report.add(f"    理由：{ABSENT_FAMILY_REASON}")


# --------------------------------------------------------------------------- #
# 第 9 节：文档渲染
# --------------------------------------------------------------------------- #


def section_9_documents(report: Report, documents: dict[str, str]) -> None:
    """第 9 节：渲染两份文档、检查覆盖、并落盘到 outputs/capstone/."""
    report.section(9, "文档渲染（从清单渲染「最终版 README + 架构文档」，并检查覆盖 8 项能力）")
    for row in study.document_rows(documents):
        report.add("  " + row.line())
    readme_path, arch_path = document.write_documents(DOC_OUTPUT_DIR)
    report.add("")
    report.add(f"  已落盘：{readme_path}")
    report.add(f"           {arch_path}")
    report.add(
        "  注：**不改动仓库根的 README.md**——本课只新增文件；渲染出来的是一份新文档"
        "（'新渲染一份'与'改掉旧的那份'是两件事）。"
    )
    report.add("")
    report.add("  「最终版 README」前 12 行：")
    for line in documents[document.DOC_README].splitlines()[:12]:
        report.add("    " + line)


# --------------------------------------------------------------------------- #
# 第 10 节：十条笔记
# --------------------------------------------------------------------------- #


def section_10_notes(report: Report) -> None:
    """第 10 节：十条笔记逐行."""
    report.section(10, "十条笔记")
    for line in study.note_lines():
        report.add("  " + line)


# --------------------------------------------------------------------------- #
# 第 11 节：边界与接缝
# --------------------------------------------------------------------------- #


def section_11_boundaries(report: Report) -> None:
    """第 11 节：五条边界 + 与既有包的接缝 + 收尾."""
    report.section(11, "五条边界与与既有包的接缝")
    report.add("  五条边界（本课明确不承诺的事）：")
    for line in study.boundary_lines():
        report.add("    " + line)
    report.add("")
    report.add("  与既有包的接缝（本课是它们的使用者，一行未改）：")
    report.add("    guard     security.injection_detector / security.content_moderator")
    report.add("    plan      agent.planner.Planner（+ tools.calculator.CalculatorTool）")
    report.add("    retrieve  retrieval.hybrid.HybridRetriever（build_retriever + LexicalIndex）")
    report.add("    pack      retrieval.context.pack_context")
    report.add("    generate  retrieval.generation.RAGGenerator（配 llm.mock.MockLLM）")
    report.add("    ground    retrieval.generation.GroundingReport")
    report.add("    evaluate  evaluation.rag_metrics（recall / precision / mrr / ndcg）")
    report.add("    account   observability.cost_tracker.CostTracker")
    report.add("    trace     observability.tracing.Tracer")
    report.add("    library   vectorstore.FlatVectorStore / llm.embedding.CharNgramEmbedding")
    report.add("")
    report.add(f"  性质名单（{len(types.CAPSTONE_PROPERTIES)} 条）：{'、'.join(types.CAPSTONE_PROPERTIES)}")
    report.add(f"  笔记条数：{len(types.CAPSTONE_NOTES)} | 边界条数：{len(types.CAPSTONE_BOUNDARIES)}")
    report.add(f"  能力条数：{len(types.CAPABILITIES)} | 阶段条数：{len(types.ASSEMBLY_STAGES)}")


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #


def main() -> int:
    """按节运行演示，并把完整输出写入 ``outputs/capstone_demo.txt``."""
    # 让演示输出**逐字节可复算**：Planner 的 INFO 日志带时间戳，它会让"跑两次两份文件不同"，
    # 而本课要证明的正是"同一输入两次运行逐位相同"。日志降到 WARNING 只影响本脚本的终端输出。
    # （get_logger 会给每个模块级 logger 单独设 level，因此这里要按名字点名，而不是只设父 logger。）
    logging.getLogger("smart_research_agent").setLevel(logging.WARNING)
    logging.getLogger("smart_research_agent.agent.planner").setLevel(logging.WARNING)
    # 单次计算：run / 清单 / 文档各算一次，七条性质与五张表共用它们。
    first = assembly.run()
    second = assembly.run()
    built = manifest.build_manifest()
    documents = document.render_documents(built)
    outcome = verify.check_all(first, built, documents)

    report = Report()
    report.add("#" * 78)
    report.add("# day099 结业项目整合（一）：把八项能力装成一条可复算的链")
    report.add("# 全部离线、全部确定性：不联网、不读任何环境变量密钥")
    report.add("#" * 78)
    section_1_substrate(report)
    section_2_stages(report, first)
    section_3_reproducible(report, first, second)
    section_4_capabilities(report, built)
    section_5_subpackages(report, built)
    section_6_properties(report, outcome)
    section_7_tables(report, first, built, documents)
    section_8_families(report)
    section_9_documents(report, documents)
    section_10_notes(report)
    section_11_boundaries(report)
    report.add("")
    report.add(f"演示完成（工作目录 {PROJECT_ROOT}）")
    report.add(f"本节输出已同时写入 {OUTPUT_FILE}")
    report.add("十一节全部离线：零网络、零新增依赖；读数、复算与清单都是真算出来的。")
    report.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
