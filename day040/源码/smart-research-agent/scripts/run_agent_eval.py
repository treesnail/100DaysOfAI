"""Agent 评估演示脚本：MockLLM 驱动，全程离线、确定性输出.

运行方式（在项目根目录）：

    python scripts/run_agent_eval.py

产出：在 data/eval/ 下生成 report.md（Markdown 评估报告）。

day032 起支持门禁阈值：--min-completion-rate / --min-step-efficiency，
任一指标低于阈值时报告追加红色告警段并以退出码 1 结束（供 CI/流水线判定）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 脚本直接运行时 sys.path[0] 是 scripts/，把项目根目录插到最前，
# 保证 import 到的是本目录快照内的 smart_research_agent，而非环境里已装的版本
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.agent.react_agent import ReactAgent
from smart_research_agent.evaluation.agent_eval import AgentEvaluator
from smart_research_agent.evaluation.report import write_report
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.observability.alerting import check_thresholds, render_alerts
from smart_research_agent.tools.calculator import CalculatorTool
from smart_research_agent.tools.registry import ToolRegistry

TASKS_PATH = PROJECT_ROOT / "data" / "eval" / "agent_tasks.jsonl"
REPORT_PATH = PROJECT_ROOT / "data" / "eval" / "report.md"

SEARCH_REPLY = "模拟搜索结果：这是离线评估使用的固定回复"


class FakeSearchTool(CalculatorTool):
    """离线演示用的假搜索工具：不联网，返回固定文本."""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "联网搜索，返回与查询相关的资料摘要"

    def execute(self, expression: str = "", **kwargs) -> str:  # noqa: ARG002
        return SEARCH_REPLY


def make_agent(responses: list[str]) -> ReactAgent:
    """装配被测 Agent：MockLLM + calculator/web_search 两个工具."""
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(FakeSearchTool())
    return ReactAgent(llm=MockLLM(responses=responses), registry=registry, max_steps=6)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-completion-rate", type=float, default=0.0, help="任务完成率门禁阈值"
    )
    parser.add_argument(
        "--min-step-efficiency", type=float, default=0.0, help="步数效率门禁阈值"
    )
    args = parser.parse_args(argv)

    evaluator = AgentEvaluator(agent_factory=make_agent)
    evaluation = evaluator.evaluate(TASKS_PATH)
    metrics = evaluation["metrics"]
    report_path = write_report(metrics, evaluation["results"], REPORT_PATH)

    # 门禁判定：阈值与指标计算解耦（observability.alerting）
    thresholds = {
        "completion_rate": args.min_completion_rate,
        "step_efficiency": args.min_step_efficiency,
    }
    thresholds = {k: v for k, v in thresholds.items() if v > 0.0}
    alerts = check_thresholds(
        {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        thresholds,
    )
    if alerts:
        with report_path.open("a", encoding="utf-8") as f:
            f.write(render_alerts(alerts))
        for alert in alerts:
            print(f"[告警] {alert.message}")

    print(f"评估任务数: {metrics['total']}")
    print(f"任务完成率: {metrics['completion_rate']:.2f}")
    print(f"步数效率:   {metrics['step_efficiency']:.2f}")
    tool = metrics["tool_accuracy"]
    print(f"工具正确性: precision={tool['precision']:.2f} recall={tool['recall']:.2f} "
          f"f1={tool['f1']:.2f} order={tool['order']:.2f}")
    print(f"报告已写入: {report_path}")
    return 1 if alerts else 0


if __name__ == "__main__":
    raise SystemExit(main())
