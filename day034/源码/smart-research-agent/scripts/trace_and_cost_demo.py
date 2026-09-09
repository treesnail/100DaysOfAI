"""day032 习题 10 参考答案：带 trace 跑一次 Agent 并算一笔账.

运行方式（在项目根目录）：

    python scripts/trace_and_cost_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.agent.react_agent import ReactAgent
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.observability.cost_tracker import CostTracker
from smart_research_agent.observability.tracing import Tracer, run_with_trace
from smart_research_agent.tools.calculator import CalculatorTool
from smart_research_agent.tools.registry import ToolRegistry

TRACE_PATH = PROJECT_ROOT / "data" / "eval" / "traces.jsonl"


class FakeSearchTool(CalculatorTool):
    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "联网搜索，返回与查询相关的资料摘要"

    def execute(self, expression: str = "", **kwargs) -> str:  # noqa: ARG002
        return "模拟搜索结果：这是离线评估使用的固定回复"


def make_agent(responses: list[str]) -> ReactAgent:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(FakeSearchTool())
    return ReactAgent(llm=MockLLM(responses=responses), registry=registry, max_steps=6)


def main() -> int:
    agent = make_agent(
        [
            "Thought: 需要先搜索一下\nAction: web_search\nAction Input: RAG 检索增强",
            "Thought: 资料够了\nFinal Answer: RAG 是检索增强生成技术。",
        ]
    )
    llm = agent.llm

    # 1) 带 trace 运行；结束后 agent.llm 已换回，可直接记账
    tracer = Tracer(sink_path=TRACE_PATH)
    answer = run_with_trace(agent, "什么是 RAG？", tracer)

    tracker = CostTracker()
    tracker.record_from_llm(llm, model="gpt-4o-mini")

    # 2) 打印 span 明细与费用
    trace_id = tracer.spans[-1].trace_id
    print(f"trace_id: {trace_id}")
    for span in tracer.get_trace(trace_id):
        print(f"  {span.name:<12} {span.duration_ms:>8.3f}ms  {span.status}")
    print(f"prompt tokens:     {llm.total_prompt_tokens}")
    print(f"completion tokens: {llm.total_completion_tokens}")
    print(f"费用: ${tracker.total_cost:.6f}")

    # 3) 落盘与回放验证
    reloaded = Tracer.load(TRACE_PATH)
    assert len(reloaded) == len(tracer.spans), "回放记录数与内存不一致"
    print(f"回放: 从 {TRACE_PATH.name} 读回 {len(reloaded)} 条 span，与内存一致")

    # 4) 断言自测
    root = tracer.query(name="agent.run")[0]
    assert root.status == "ok"
    assert root.attributes["answer"] == answer
    assert len(tracer.query(name="llm.chat")) == len(llm.calls)
    print("全部断言通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
