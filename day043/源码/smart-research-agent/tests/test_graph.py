"""状态图测试：用 MockLLM 驱动 LangGraph 跑通 ReAct 流程，全程离线."""

from __future__ import annotations

import pytest

from smart_research_agent.graph import build_react_graph, initial_state, run_graph
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.tools.calculator import CalculatorTool
from smart_research_agent.tools.registry import ToolRegistry

TASK = "计算 2+3 等于几"

THINK_CALC = "Thought: 需要用计算器\nAction: calculator\nAction Input: 2+3"
FINAL = "Thought: 已得到计算结果\nFinal Answer: 2+3 等于 5"


@pytest.fixture
def registry() -> ToolRegistry:
    """提供注册了计算器的工具注册表."""
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    return reg


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


class TestReActGraph:
    """图的端到端行为."""

    def test_full_react_flow(self, registry: ToolRegistry):
        llm = MockLLM(responses=[THINK_CALC, FINAL])
        graph = build_react_graph(llm, registry)

        answer = run_graph(graph, TASK, thread_id="t-full")

        assert "5" in answer
        state = graph.get_state(_config("t-full"))
        assert state.values["done"] is True
        assert state.values["steps"] == 1
        assert state.values["observation"] == "5"
        assert len(state.values["history"]) == 1

    def test_max_steps_guard(self, registry: ToolRegistry):
        """模型始终不给 Final Answer 时，图必须在 max_steps 后终止."""
        llm = MockLLM(default=THINK_CALC)  # 永远要求调计算器
        graph = build_react_graph(llm, registry, max_steps=3)

        answer = run_graph(graph, TASK, thread_id="t-max")

        assert "最大步数" in answer
        state = graph.get_state(_config("t-max"))
        assert state.values["steps"] == 3
        assert state.values["done"] is False

    def test_unknown_tool_observation(self, registry: ToolRegistry):
        """调用不存在的工具时，act 节点把错误写回 observation 而不是崩溃."""
        llm = MockLLM(
            responses=["Thought: 试试不存在的工具\nAction: nope\nAction Input: x", FINAL]
        )
        graph = build_react_graph(llm, registry)

        run_graph(graph, TASK, thread_id="t-unknown")

        state = graph.get_state(_config("t-unknown"))
        assert "不存在" in state.values["observation"]

    def test_direct_answer_without_tool(self, registry: ToolRegistry):
        """模型第一步就给 Final Answer 时，不经过 act 节点."""
        llm = MockLLM(responses=[FINAL])
        graph = build_react_graph(llm, registry)

        answer = run_graph(graph, TASK, thread_id="t-direct")

        assert "5" in answer
        state = graph.get_state(_config("t-direct"))
        assert state.values["steps"] == 0
        assert state.values["done"] is True


class TestCheckpoint:
    """检查点持久化：同一 thread_id 可取回状态，不同 thread 相互隔离."""

    def test_get_state_after_run(self, registry: ToolRegistry):
        llm = MockLLM(responses=[THINK_CALC, FINAL])
        graph = build_react_graph(llm, registry)
        run_graph(graph, TASK, thread_id="t-ckpt")

        snapshot = graph.get_state(_config("t-ckpt"))

        assert snapshot.values["task"] == TASK
        assert snapshot.values["final_answer"] == "2+3 等于 5"
        assert snapshot.next == ()  # 已运行到 END，没有待执行节点

    def test_threads_are_isolated(self, registry: ToolRegistry):
        llm = MockLLM(responses=[FINAL])
        graph = build_react_graph(llm, registry)
        run_graph(graph, TASK, thread_id="thread-a")

        snapshot_b = graph.get_state(_config("thread-b"))

        assert snapshot_b.values == {}  # 另一个线程没有任何状态


class TestInterrupt:
    """断点调试：interrupt_before 暂停，invoke(None, config) 恢复."""

    def test_interrupt_before_act_and_resume(self, registry: ToolRegistry):
        llm = MockLLM(responses=[THINK_CALC, FINAL])
        graph = build_react_graph(llm, registry, interrupt_before=["act"])
        config = _config("t-bp")

        paused = graph.invoke(initial_state(TASK), config=config)

        # 图停在 act 之前：think 已写入 action，但 observation 还是空
        assert paused["action"] == "calculator"
        assert paused["observation"] == ""
        assert graph.get_state(config).next == ("act",)

        # 恢复执行：传入 None 表示从检查点继续
        final = graph.invoke(None, config=config)

        assert final["done"] is True
        assert final["steps"] == 1
        assert "5" in final["final_answer"]

    def test_edit_state_at_breakpoint(self, registry: ToolRegistry):
        """断点处人工修正状态（人在回路），恢复后按修正值执行."""
        llm = MockLLM(responses=[THINK_CALC, FINAL])
        graph = build_react_graph(llm, registry, interrupt_before=["act"])
        config = _config("t-edit")

        graph.invoke(initial_state(TASK), config=config)
        # 人工介入：把 2+3 改成 10*10
        graph.update_state(config, {"action_input": "10*10"})

        final = graph.invoke(None, config=config)

        assert final["observation"] == "100"
