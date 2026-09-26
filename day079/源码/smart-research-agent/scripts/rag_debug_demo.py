"""RAG 评估与调试演示（day071）：端到端打分 → 坏例归因 → 质量基线 → A/B 对照.

用法::

    python scripts/rag_debug_demo.py

九节，每节回答一个问题（顺序 = 读一份评估报告的顺序）：

```text
1  套件现在是长什么样       评测集 / 语料 / k / 判据 / 四个变体
2  这一批答案整体好不好     11 个质量指标（含两段召回的那个漏斗）
3  不好在哪一层             12 类坏例标签的分布 + 前几条逐条归因
4  下一步动哪个旋钮         去重后的动作清单
5  这一切在什么范围内成立    报告的 limitations（它必须自述边界）
6  这次比上次差吗           折出基线 → 存盘 → 与已提交基线对比
7  换一版提示词之后变了吗    A/B 对照（同一索引、同一批用例，只差提示词版本）
8  预算收紧会丢掉什么        tight-budget 变体：packed 召回掉了多少（那一门独有的失败）
9  指标口径与坏例标签表     方向表 + 处置动作表（读数字前先读它）

离线说明：本脚本用项目自带的 ``default_embedding()``（字符 n-gram，确定性、
不需要 API Key）与一个按脚本回复的 ``DemoLLM`` 跑通整条评测流水线。
真实评估请注入真模型与真编码器——报告会如实记下 ``llm_called`` 与回退分布，
因此"用替身跑通"与"用替身得出结论"是两件事（后者要不得）。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smart_research_agent.config import settings
from smart_research_agent.llm.base import BaseLLM, Message
from smart_research_agent.llm.embedding import default_embedding
from smart_research_agent.rag_debug import (
    RagBaseline,
    RagBaselineGuard,
    build_suite,
    compare_variants,
    described_tags,
    metric_table,
)
from smart_research_agent.vectorstore.flat import FlatVectorStore

#: 演示用的兜底答复（提示词里一个编号都没有时用到）。
DEMO_ANSWER = "资料中没有相关内容，因此不作答。"

#: 提示词里**片段块的头部**编号（形如行首的 ``[1] 来源 › 标题``）.
#: 必须锚在行首：提示词模板里也会出现 ``[1]`` 这种举例写法，而把那些也算进来
#: 会让模型"引用"一个并不存在的编号——那是一次**假的幻觉引用**，
#: 由替身的解析错误造出来，而不是系统的问题（下面第 8 节会看到这个区别有多要紧）。
MARKER_PATTERN = re.compile(r"^\[\s*(\d+)\s*\]", re.MULTILINE)


class DemoLLM(BaseLLM):
    """按提示词里出现的编号**逐条引用**的模型（确定性，且答案可核对）.

    它模拟的是一个"会把用上的片段都标出来"的正常模型：从提示词里读出编号，
    再在答案里逐个引用。这样跑出来的报告才有信息量——覆盖率不再是替身的
    副作用（一个只会写 ``[1]`` 的假模型会让每条用例都变成"低覆盖"，
    而那与系统质量无关，只是替身太懒）。
    """

    def __init__(self) -> None:
        self.calls = 0

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """读出提示词里的编号并逐条引用（不联网、不读盘、逐位可复现）."""
        self.calls += 1
        prompt = messages[0].content if messages else ""
        markers = sorted({int(item) for item in MARKER_PATTERN.findall(prompt)})
        if not markers:
            return DEMO_ANSWER
        cites = "、".join(f"[{number}]" for number in markers)
        return (
            f"依据 {cites}：检索增强生成先把相关片段召回，再交给模型生成答案，"
            "因此每一句都可以回到具体片段上核对。"
        )


class HallucinatingLLM(BaseLLM):
    """引用了**不存在的编号**的模型（``[9]`` 不在任何一次提示词里）.

    它是 day069 那节演示的复现：``[9]`` 不是"多写了一个数字"，而是
    **指到了一份并不存在的依据**——因此在报告里它是 ``invalid``，
    并且会让那一条用例的 ``grounded`` 直接变成 False。
    """

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """返回一段「编号写多了」的答案（用来演示幻觉引用的归因）."""
        return "依据 [1] 与 [9]，检索增强生成的结论成立。"


def section(title: str) -> None:
    """打印一节的分隔标题."""
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    """九节演示：跑变体 → 归因 → 基线 → A/B。"""
    suite = build_suite()
    embedding = default_embedding()
    store = FlatVectorStore()
    write = suite.index_into(store, embedding)

    section("1. 套件现状")
    print(json.dumps(suite.describe(), ensure_ascii=False, indent=2))
    print(
        f"语料入库：新增 {write.added} / 覆盖 {write.updated} / "
        f"未变 {write.unchanged} / 跳过 {write.skipped}"
    )

    llm = DemoLLM()
    reports = suite.run_all(store, embedding, llm)

    section("2. 每个变体的 11 个质量指标")
    for report in reports:
        print(f"\n--- {report.label} ---")
        print(report.summary_line())
        for name in report.to_dict()["metrics"]:
            value = report.value(name)
            print(f"    {name:<24} {value:.4f}")

    section("3. 坏例归因（前一条变体）")
    baseline_report = reports[0]
    for line in baseline_report.explain(limit=6):
        print(line)

    section("4. 下一步动作（去重后）")
    for index, action in enumerate(baseline_report.actions, start=1):
        print(f"{index}. {action}")
    print(f"\n坏例标签分布：{baseline_report.to_dict()['tag_counts']}")
    print(f"回退原因分布：{baseline_report.to_dict()['fallback_counts']}")

    section("5. 本次结论的适用范围")
    for item in baseline_report.limitations:
        print(f"- {item}")

    section("6. 折出质量基线并对比")
    baseline = baseline_report.to_baseline()
    target = Path(settings.rag_eval_baseline_path)
    baseline.save(target)
    print(baseline.summary_line())
    print(f"基线已写入：{target}")
    report_path = baseline_report.save(Path(settings.rag_eval_report_path))
    print(f"完整报告已写入：{report_path}")

    committed = RagBaseline.load(target)
    comparison = RagBaselineGuard(committed).compare(baseline)
    print(f"\n与已提交基线对比：{comparison.summary_line()}")

    section("7. A/B：只换提示词版本")
    for name in ("prompt-v1", "cite-gate"):
        current = next(item for item in reports if item.label.endswith(name))
        ab = compare_variants(baseline, current.to_baseline())
        print(ab.summary_line())
        for item in ab.deltas:
            if item.verdict != "unchanged":
                print(f"    {item.summary_line()}")

    section("8. 两个「把系统弄坏」的实验，与它们的归因")
    tight = next(item for item in reports if item.label.endswith("tight-budget"))
    ab = compare_variants(baseline, tight.to_baseline())
    print(ab.summary_line())
    print(
        "两段召回的差：检索 "
        f"{tight.value('retrieval_recall'):.4f} → 进提示词 "
        f"{tight.value('packed_recall'):.4f}"
    )
    print(
        "读法：这次收紧预算**没有**伤到召回——因为金标准总是排在最前面，"
        "而预算从尾部丢。这正是 packed_recall 这个新指标的意义：\n"
        "      它把「预算有没有伤到依据」变成一个有数字可查的问题，"
        "而不是靠「答案看起来变短了」来判断。"
    )

    empty = FlatVectorStore()
    empty_report = suite.run_variant(empty, embedding, DemoLLM(), "baseline")
    print(f"\n空库那一次：{empty_report.summary_line()}")
    print(f"  坏例标签分布：{empty_report.to_dict()['tag_counts']}")
    print(
        "  注意 llm_call_rate = "
        f"{empty_report.value('llm_call_rate'):.4f} 与回退分布 "
        f"{empty_report.to_dict()['fallback_counts']}："
        "检索为空时一次模型都没调（护栏生效），\n"
        "  因此那一段兜底答复不是模型编的——这类用例的处置动作是"
        "「先灌语料」，不是「换个模型」。"
    )

    hallucinating = suite.run_variant(
        store, embedding, HallucinatingLLM(), "baseline"
    )
    print(f"\n幻觉引用那一次：{hallucinating.summary_line()}")
    print(f"  坏例标签分布：{hallucinating.to_dict()['tag_counts']}")
    if hallucinating.bad_cases:
        first = hallucinating.bad_cases[0]
        print(f"  第一条：{first.to_dict()['description']}")
        print(f"  动作：{first.action}")
    print(
        "  注意 grounded_rate = "
        f"{hallucinating.value('grounded_rate'):.4f}：一个引用了不存在编号的答案"
        "即使「其余部分看起来都对」，接地判定也只能是否定的——\n"
        "  因为读的人无法区分哪一句有依据（定义见 day069 的 GroundingReport）。"
    )

    section("9. 指标口径与坏例标签表")
    print("指标（名字 / 方向 / 说明）：")
    for row in metric_table():
        print(f"    {row['metric']:<24} {row['direction']:<14} {row['description']}")
    print("\n坏例标签（标签 / 是什么问题 / 下一步做什么）：")
    for tag, item in described_tags().items():
        print(f"    [{tag}]")
        print(f"        问题：{item['description']}")
        print(f"        动作：{item['action']}")


if __name__ == "__main__":
    main()
