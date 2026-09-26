"""ReAct 循环端到端测试（MockLLM 驱动，完全离线）."""

from __future__ import annotations

from smart_research_agent.agent.react_agent import ReactAgent
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.tools.calculator import CalculatorTool
from smart_research_agent.tools.registry import ToolRegistry


def _make_agent(responses: list[str], max_steps: int = 10) -> ReactAgent:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    return ReactAgent(llm=MockLLM(responses=responses), registry=registry, max_steps=max_steps)


class TestReactLoop:
    def test_full_loop_action_then_answer(self):
        agent = _make_agent(
            [
                "Thought: 需要先计算 2+3\nAction: calculator\nAction Input: 2+3",
                "Thought: 已得到结果\nFinal Answer: 2+3 等于 5",
            ]
        )
        answer = agent.run("2+3 等于几？")
        assert "5" in answer
        assert len(agent.history) == 2

    def test_unknown_tool_feeds_error_back(self):
        agent = _make_agent(
            [
                "Thought: 试试不存在的工具\nAction: time_machine\nAction Input: now",
                "Thought: 工具不存在，直接回答\nFinal Answer: 无法使用工具",
            ]
        )
        answer = agent.run("现在几点？")
        assert answer == "无法使用工具"
        # 第二次调用 LLM 时，消息里应包含错误 Observation
        second_call = agent.llm.calls[1]
        assert any("不存在" in m.content for m in second_call)

    def test_max_steps_guard(self):
        agent = _make_agent(
            ["Thought: 一直算\nAction: calculator\nAction Input: 1+1"] * 3,
            max_steps=3,
        )
        answer = agent.run("1+1？")
        assert "最大步数" in answer
        assert len(agent.history) == 3
