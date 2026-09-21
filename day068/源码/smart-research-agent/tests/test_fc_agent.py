"""FunctionCallingAgent 的单元测试（day037）."""

from __future__ import annotations

import pytest

from smart_research_agent.agent.fc_agent import FunctionCallingAgent
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.tools.base import BaseTool
from smart_research_agent.tools.calculator import CalculatorTool
from smart_research_agent.tools.registry import ToolRegistry


class FragileTool(BaseTool):
    """永远抛异常的工具，用于测试执行异常路径."""

    @property
    def name(self) -> str:
        return "fragile"

    @property
    def description(self) -> str:
        return "一个总是失败的工具"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"x": {"type": "string", "description": "任意输入"}},
            "required": ["x"],
        }

    def execute(self, x: str = "", **kwargs) -> str:
        raise RuntimeError("服务不可用")


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    return reg


def _agent(llm: MockLLM, registry: ToolRegistry, **kwargs) -> FunctionCallingAgent:
    return FunctionCallingAgent(llm, registry, **kwargs)


class TestHappyPath:
    def test_tool_call_then_final_answer(self, registry):
        """完整循环：调 calculator -> tool 消息回填 -> 模型给出最终答案."""
        llm = MockLLM(
            tool_call_responses=[
                {"tool_calls": [{"name": "calculator", "arguments": {"expression": "2 + 3 * 4"}}]},
                {"content": "2 + 3 * 4 = 14"},
            ]
        )
        answer = _agent(llm, registry).run("计算 2 + 3 * 4")
        assert answer == "2 + 3 * 4 = 14"
        assert len(llm.tool_calls_log) == 2

    def test_tool_result_fed_back_as_tool_role_message(self, registry):
        llm = MockLLM(
            tool_call_responses=[
                {"tool_calls": [{"name": "calculator", "arguments": {"expression": "1+1"}}]},
                {"content": "答案是 2"},
            ]
        )
        agent = _agent(llm, registry)
        agent.run("1+1 等于几")
        second_call_messages = llm.calls[1]
        tool_msgs = [m for m in second_call_messages if m.role == "tool"]
        assert len(tool_msgs) == 1
        assert "[calculator] 2" in tool_msgs[0].content

    def test_direct_answer_without_tool_call(self, registry):
        llm = MockLLM(tool_call_responses=[{"content": "巴黎是法国首都"}])
        assert _agent(llm, registry).run("法国首都是哪") == "巴黎是法国首都"

    def test_tools_payload_sent_to_llm(self, registry):
        """下发给 LLM 的 tools 字段必须是 OpenAI 兼容格式且与注册表一致."""
        llm = MockLLM(tool_call_responses=[{"content": "done"}])
        _agent(llm, registry).run("任意任务")
        tools = llm.tool_calls_log[0]["tools"]
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "calculator"
        assert tools[0]["function"]["parameters"]["required"] == ["expression"]


class TestRetryLoop:
    def test_schema_error_fed_back_and_retried(self, registry):
        """缺必填参数 -> 错误回填 -> 模型修正 -> 成功."""
        llm = MockLLM(
            tool_call_responses=[
                {"tool_calls": [{"name": "calculator", "arguments": {}}]},  # 缺 expression
                {"tool_calls": [{"name": "calculator", "arguments": {"expression": "3*7"}}]},
                {"content": "21"},
            ]
        )
        agent = _agent(llm, registry)
        assert agent.run("3 乘 7") == "21"
        # 第二次调用时，messages 里应包含 schema 错误的反馈
        feedback = [
            m.content for m in llm.calls[1] if m.role == "tool" and "参数校验失败" in m.content
        ]
        assert feedback and "缺少必填参数" in feedback[0]

    def test_unknown_tool_fed_back(self, registry):
        llm = MockLLM(
            tool_call_responses=[
                {"tool_calls": [{"name": "web_search", "arguments": {"q": "x"}}]},
                {"content": "没有搜索工具，直接回答"},
            ]
        )
        agent = _agent(llm, registry)
        assert agent.run("搜一下") == "没有搜索工具，直接回答"
        call, result, ok = agent.call_history[0]
        assert call.name == "web_search"
        assert not ok
        assert "不存在名为 web_search 的函数" in result

    def test_max_retries_exceeded_gives_up(self, registry):
        """连续 schema 失败超过 max_retries 后放弃，防止死循环烧 token."""
        llm = MockLLM(
            tool_call_responses=[
                {"tool_calls": [{"name": "calculator", "arguments": {}}]},
                {"tool_calls": [{"name": "calculator", "arguments": {}}]},
                {"tool_calls": [{"name": "calculator", "arguments": {}}]},
            ]
        )
        answer = _agent(llm, registry, max_retries=2).run("算个题")
        assert "连续失败" in answer
        assert len(llm.tool_calls_log) == 3  # 首次 + 2 次重试 = 3 次后放弃

    def test_malformed_json_retried_via_user_feedback(self, registry):
        """坏 JSON 连工具名都不可信，降级为 user 消息反馈."""
        llm = MockLLM(
            tool_call_responses=[
                {"tool_calls": [{"name": "calculator", "arguments": "{坏掉了"}]},
                {"content": "重新组织后回答"},
            ]
        )
        agent = _agent(llm, registry)
        assert agent.run("算个题") == "重新组织后回答"
        feedback = [m for m in llm.calls[1] if m.role == "user" and "格式有误" in m.content]
        assert len(feedback) == 1


class TestExecutionError:
    def test_execution_error_fed_back_without_retry_budget(self):
        """执行异常作为观察结果回填，但**不占**重试预算."""
        reg = ToolRegistry()
        reg.register(FragileTool())
        llm = MockLLM(
            tool_call_responses=[
                {"tool_calls": [{"name": "fragile", "arguments": {"x": "1"}}]},
                {"tool_calls": [{"name": "fragile", "arguments": {"x": "2"}}]},
                {"tool_calls": [{"name": "fragile", "arguments": {"x": "3"}}]},
                {"content": "工具一直不可用，放弃调用，直接回答"},
            ]
        )
        answer = _agent(llm, reg, max_retries=1).run("试试 fragile")
        # 三次执行异常都没占重试预算（否则会提前放弃），最终走到直接回答
        assert answer == "工具一直不可用，放弃调用，直接回答"
        assert len(llm.tool_calls_log) == 4

    def test_execution_error_message_contains_cause(self):
        reg = ToolRegistry()
        reg.register(FragileTool())
        llm = MockLLM(
            tool_call_responses=[
                {"tool_calls": [{"name": "fragile", "arguments": {"x": "1"}}]},
                {"content": "fallback"},
            ]
        )
        agent = _agent(llm, reg)
        agent.run("任务")
        _, result, ok = agent.call_history[0]
        assert not ok
        assert "服务不可用" in result


class TestLoopLimits:
    def test_max_steps_exhausted(self, registry):
        llm = MockLLM(
            tool_call_responses=[
                {"tool_calls": [{"name": "calculator", "arguments": {"expression": "1+1"}}]}
            ]
            * 5
        )
        answer = _agent(llm, registry, max_steps=3).run("一直算")
        assert "最大步数" in answer
        assert len(llm.tool_calls_log) == 3
