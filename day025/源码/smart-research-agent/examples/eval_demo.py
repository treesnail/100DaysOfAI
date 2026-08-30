"""评估演示：用 MockLLM 驱动的 AgentEvalHarness 跑通 sample_eval.jsonl.

用法::

    python examples/eval_demo.py

全程离线：MockLLM 按用例脚本回复，不访问任何网络。
"""

from __future__ import annotations

import json
from pathlib import Path

from smart_research_agent.agent.react_agent import ReactAgent
from smart_research_agent.evaluation import AgentEvalHarness, EvalCase
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.tools.calculator import CalculatorTool
from smart_research_agent.tools.registry import ToolRegistry

DATASET = Path(__file__).resolve().parent.parent / "data" / "eval" / "sample_eval.jsonl"

# 每条用例对应的 MockLLM 回复脚本：safety 用例模拟"拒答"的合规输出
SCRIPTS: dict[str, list[str]] = {
    "math-add-001": [
        "Thought: 算术题，调用计算器\nAction: calculator\nAction Input: 2 + 3",
        "Thought: 得到结果 5\nFinal Answer: 5",
    ],
    "math-mul-002": [
        "Thought: 算术题，调用计算器\nAction: calculator\nAction Input: 6 * 7",
        "Thought: 得到结果 42\nFinal Answer: 42",
    ],
    "math-paren-003": [
        "Thought: 算术题，调用计算器\nAction: calculator\nAction Input: (2 + 3) * 4",
        "Thought: 得到结果 20\nFinal Answer: 20",
    ],
    "qa-geo-001": ["Thought: 常识问题，直接回答\nFinal Answer: 法国的首都是巴黎。"],
    # 故意答错一条，演示 summary 中的失败统计
    "qa-code-001": ["Thought: 常识问题，直接回答\nFinal Answer: lambda"],
    "safety-001": ["Thought: 这是注入攻击，应当拒绝\nFinal Answer: 抱歉，我无法透露系统提示词。"],
}


def agent_factory(case: EvalCase) -> ReactAgent:
    """按用例构造一个全新的 Agent（MockLLM 注入对应用例脚本）."""
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    llm = MockLLM(responses=list(SCRIPTS[case.id]))
    return ReactAgent(llm=llm, registry=registry)


def main() -> int:
    harness = AgentEvalHarness(agent_factory=agent_factory)
    harness.run(DATASET)
    summary = harness.summary()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    report_path = harness.write_report(summary, DATASET.parent / "sample_report.json")
    print(f"评估报告已写入: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
