#!/usr/bin/env python
"""day100 离线演示：结业项目整合（二）——把一次能跑的运行交付成四份产物.

十一节，全部离线、全部确定性（不需要 API Key，也不联网）：

```text
1   里程碑与四份交付物（口径：demo / summary / assessment / roadmap）
2   演示剧本：九段读数 + 汇总 + 归档行 + 摘要
3   重放：两次重放逐位相同（本课最硬的一条证据）
4   课程清单：用 pkgutil + importlib 数遍全部一级子包
5   课程总结：八个计数（前四个数出来，后四个折出来）
6   能力自评：8 项能力逐项评级 + 三项证据
7   后续规划：从真实缺口推出下一步（缺口集合 == 规划项集合）
8   七条性质与三类判据（相等 / 上界 / 下界）
9   五张表：交付物 / 清单 / 自评 / 规划 / 性质
10  失败族：七个族各自"该谁去修" + 回来的族与缺席的族
11  四格量程 / 十条笔记 / 五条边界与与既有包的接缝
```

运行方式::

    cd day100/源码/smart-research-agent
    python scripts/graduation_demo.py

产出：

```text
outputs/graduation_demo.txt            本脚本的完整输出（在 .gitignore 里）
outputs/graduation/demo.transcript.txt 落盘的演示剧本（新的产物，不覆盖任何既有文件）
```

**不覆盖仓库根的任何既有文件**：本课只新增文件，既有文件一行未改。
"""

from __future__ import annotations

import logging
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.graduation import (  # noqa: E402
    ABSENT_FAMILY,
    ABSENT_FAMILY_REASON,
    FAMILY_OUTCOMES,
    RETURNED_FAMILY,
    RETURNED_FAMILY_REASON,
    assessment,
    demo,
    inventory,
    roadmap,
    study,
    summary,
    types,
    verify,
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "graduation_demo.txt"
ARTIFACT_DIR = OUTPUT_DIR / "graduation"


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
        """把攒好的文本同时打印到终端并写入 ``outputs/graduation_demo.txt``."""
        text = "\n".join(self.lines)
        print(text)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_FILE.write_text(text + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# 第 1 节：里程碑与四份交付物
# --------------------------------------------------------------------------- #


def section_1_deliverables(report: Report) -> None:
    """第 1 节：里程碑常量 + 四份交付物各回答什么问题."""
    report.section(1, "里程碑与四份交付物（交付不是贴结果，而是把结果变成别人能复算的产物）")
    report.add(f"  里程碑：{types.MILESTONE_TITLE} | 第 {types.MILESTONE_TOTAL_DAYS} 天")
    report.add(f"  说明：{types.MILESTONE_DESCRIPTION}")
    report.add("")
    for spec in types.deliverables():
        report.add("  " + spec.line())
    report.add("")
    report.add("  四格量程（自评的证据口径）：")
    for line in study.level_lines():
        report.add("    " + line)


# --------------------------------------------------------------------------- #
# 第 2 节：演示剧本
# --------------------------------------------------------------------------- #


def section_2_transcript(report: Report, transcript: demo.DemoTranscript) -> None:
    """第 2 节：把剧本逐行印出来（每一行都是真算出来的读数）."""
    report.section(2, "演示剧本（九段读数 + 汇总 + 归档行 + 摘要）")
    for line in transcript.lines():
        report.add("  " + line)


# --------------------------------------------------------------------------- #
# 第 3 节：重放
# --------------------------------------------------------------------------- #


def section_3_replay(report: Report, outcome: demo.ReplayReport) -> None:
    """第 3 节：剧本重放两次逐位相同（本课最硬的一条证据）."""
    report.section(3, "重放：剧本跑两次**逐位相同**（重放口径里没有时间与 uuid）")
    report.add(f"  {outcome.line()}")
    report.add(
        "  口径（comparable 的字段）：问题 / 九段读数 / 汇总行 / 归档行——"
        "不含时间戳、uuid、耗时与日志文本"
    )
    report.add(f"  require_identical 通过：{outcome.require_identical() is None}")


# --------------------------------------------------------------------------- #
# 第 4 节：课程清单
# --------------------------------------------------------------------------- #


def section_4_inventory(report: Report, built: inventory.InventoryReport) -> None:
    """第 4 节：全部一级子包逐行（在场 / 公开名字 / 文件 / __all__）."""
    report.section(4, f"课程清单（pkgutil + importlib 数遍全部一级子包，共 {len(built.rows)} 个）")
    for row in built.rows:
        report.add("  " + row.line())
    report.add("")
    report.add("  " + built.line())


# --------------------------------------------------------------------------- #
# 第 5 节：课程总结
# --------------------------------------------------------------------------- #


def section_5_summary(
    report: Report,
    built_summary: summary.CourseSummary,
    built: inventory.InventoryReport,
) -> None:
    """第 5 节：八个计数 + 与清单的一致性对账."""
    report.section(5, "课程总结（前四个计数来自清单，后四个来自各包的常量）")
    report.add("  " + built_summary.line())
    diff = summary.summary_matches_inventory(built_summary, built)
    report.add(
        f"  与清单的三个可数计数对账：差异项数 {diff}（0 = 一致）——"
        "总结给人读、清单给程序读，两条独立路径给出同一个数才叫核对过"
    )


# --------------------------------------------------------------------------- #
# 第 6 节：能力自评
# --------------------------------------------------------------------------- #


def section_6_assessment(report: Report, result: assessment.Assessment) -> None:
    """第 6 节：8 项能力逐项评级 + 三项证据."""
    report.section(6, "能力自评（8 项能力：评级 / 承担子包 / 名字数 / 链上阶段，每一项都挂证据）")
    for row in result.rows:
        report.add("  " + row.line())
        report.add("      证据：" + "；".join(row.evidence))
    report.add("")
    report.add("  " + result.line())
    report.add(
        "  读法：评级只由证据推出（在场 / 符号 / 阶段），因此'我检查过'与'我猜的'分得开。"
    )


# --------------------------------------------------------------------------- #
# 第 7 节：后续规划
# --------------------------------------------------------------------------- #


def section_7_roadmap(
    report: Report,
    built_roadmap: roadmap.Roadmap,
    scored: assessment.Assessment,
) -> None:
    """第 7 节：从真实缺口推出的规划."""
    report.section(7, "后续规划（从真实缺口推出：缺口集合 == 规划项集合）")
    for row in study.roadmap_rows(built_roadmap):
        report.add("  " + row.line())
    report.add("")
    report.add("  " + built_roadmap.line())
    report.add(
        "  两类缺口：无人认领的子包（已交付、但不在 8 项能力清单里）"
        " 与 没到最高级的能力（自评 < L3）"
    )
    report.add(f"  没到最高级的能力：{list(scored.below) or '（无）'}")


# --------------------------------------------------------------------------- #
# 第 8 节：七条性质与三类判据
# --------------------------------------------------------------------------- #


def section_8_properties(report: Report, outcome: verify.PropertyReport) -> None:
    """第 8 节：七条性质逐条（判据类别与读数一起印）."""
    report.section(8, "七条性质与三类判据（相等 / 上界 / 下界）")
    for row in study.property_rows(outcome):
        report.add("  " + row.line())
    report.add("")
    report.add(f"  全部通过：{outcome.ok}")
    for criterion, description in types.CRITERION_DESCRIPTIONS.items():
        count = sum(1 for spec in types.property_specs() if spec.criterion == criterion)
        report.add(f"  [{criterion}] {count} 条：{description}")


# --------------------------------------------------------------------------- #
# 第 9 节：五张表
# --------------------------------------------------------------------------- #


def section_9_tables(report: Report) -> None:
    """第 9 节：一次跑完五张表."""
    report.section(9, "五张表（交付物 / 清单 / 自评 / 规划 / 性质）")
    for line in study.study_lines():
        report.add("  " + line)


# --------------------------------------------------------------------------- #
# 第 10 节：失败族
# --------------------------------------------------------------------------- #


def section_10_families(report: Report) -> None:
    """第 10 节：七个失败族 + 回来的族与缺席的族."""
    report.section(10, "失败族（按「该谁去修」分）")
    for name, outcome in FAMILY_OUTCOMES.items():
        report.add(f"  {name:<16} → {outcome}")
    report.add("")
    report.add(f"  回来的族：{RETURNED_FAMILY}")
    report.add(f"    理由：{RETURNED_FAMILY_REASON}")
    report.add(f"  缺席的族：{ABSENT_FAMILY}")
    report.add(f"    理由：{ABSENT_FAMILY_REASON}")


# --------------------------------------------------------------------------- #
# 第 11 节：笔记、边界与接缝
# --------------------------------------------------------------------------- #


def section_11_notes(report: Report, transcript: demo.DemoTranscript) -> None:
    """第 11 节：十条笔记 + 五条边界 + 与既有包的接缝 + 剧本落盘."""
    report.section(11, "十条笔记 / 五条边界与与既有包的接缝")
    report.add("  十条笔记：")
    for line in study.note_lines():
        report.add("    " + line)
    report.add("")
    report.add("  五条边界（本课明确不承诺的事）：")
    for line in study.boundary_lines():
        report.add("    " + line)
    report.add("")
    report.add("  与既有包的接缝（本课是它们的使用者，一行未改）：")
    report.add("    chain     capstone.assembly.run（九段端到端链，复用 day099）")
    report.add("    manifest  capstone.manifest.build_manifest（12 个候选子包的解析）")
    report.add("    scan      pkgutil + importlib（全部一级子包，本课新增）")
    report.add("    reproduce capstone.SystemRun.comparable / digest（复算口径）")
    report.add("")
    path = demo.write_transcript(ARTIFACT_DIR, transcript)
    report.add(f"  剧本已落盘：{path}")
    report.add(
        "  注：**不覆盖仓库里既有的任何文件**——本课只新增文件；"
        "落盘的是一份新的产物（'新生成一份'与'改掉旧的那份'是两件事）。"
    )
    report.add("")
    report.add(f"  性质名单（{len(types.GRADUATION_PROPERTIES)} 条）：{'、'.join(types.GRADUATION_PROPERTIES)}")
    report.add(f"  笔记条数：{len(types.GRADUATION_NOTES)} | 边界条数：{len(types.GRADUATION_BOUNDARIES)}")
    report.add(f"  交付物件数：{len(types.DELIVERABLE_ORDER)} | 量程格数：{len(types.LEVELS)}")


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #


def main() -> int:
    """按节运行演示，并把完整输出写入 ``outputs/graduation_demo.txt``."""
    # 让演示输出**逐字节可复算**：Planner 的 INFO 日志带时间戳，
    # 它会让"跑两次两份文件不同"，而本课要证明的正是"同一输入两次运行逐位相同"。
    logging.getLogger("smart_research_agent").setLevel(logging.WARNING)
    logging.getLogger("smart_research_agent.agent.planner").setLevel(logging.WARNING)
    # 单次计算：清单 / 自评 / 规划 / 总结 / 剧本各算一次，七条性质与五张表共用它们。
    built = inventory.build_inventory()
    scored = assessment.assess()
    planned = roadmap.plan()
    built_summary = summary.build_summary(built)
    outcome = demo.replay()
    bodies = verify.render_deliverables(
        inventory=built,
        summary=built_summary,
        assessment=scored,
        roadmap=planned,
        transcript=outcome.first,
    )
    report_object = verify.check_all(
        replay_report=outcome,
        bodies=bodies,
        inventory=built,
        assessment=scored,
        roadmap=planned,
        summary=built_summary,
    )

    report = Report()
    report.add("#" * 78)
    report.add("# day100 结业项目整合（二）：把一次能跑的运行交付成四份能被复算的产物")
    report.add("# 全部离线、全部确定性：不联网、不读任何环境变量密钥")
    report.add("#" * 78)
    section_1_deliverables(report)
    section_2_transcript(report, outcome.first)
    section_3_replay(report, outcome)
    section_4_inventory(report, built)
    section_5_summary(report, built_summary, built)
    section_6_assessment(report, scored)
    section_7_roadmap(report, planned, scored)
    section_8_properties(report, report_object)
    section_9_tables(report)
    section_10_families(report)
    section_11_notes(report, outcome.first)
    report.add("")
    report.add(f"演示完成（工作目录 {PROJECT_ROOT}）")
    report.add(f"本节输出已同时写入 {OUTPUT_FILE}")
    report.add("十一节全部离线：零网络、零新增依赖；四份产物与七条判据都是真算出来的。")
    report.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
