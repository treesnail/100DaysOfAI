"""day037 编程题参考：function calling 全链路离线演示.

运行方式（项目根目录）：
    python scripts/function_calling_demo.py
全程离线：LLM 由 MockLLM 的 tool_call_responses 脚本扮演。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smart_research_agent.agent.fc_agent import FunctionCallingAgent
from smart_research_agent.llm.function_calling import (
    SchemaValidationError,
    parse_tool_calls,
    specs_from_tools,
    validate_arguments,
)
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.tools.calculator import CalculatorTool
from smart_research_agent.tools.registry import ToolRegistry


def demo_spec_and_parse() -> None:
    """FunctionSpec 自动生成 + OpenAI 兼容响应解析 + schema 校验."""
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    spec = specs_from_tools(registry.list_tools())[0]

    payload = spec.to_openai_tool()
    print("[spec] 下发给 LLM 的 tools 字段：")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    assert payload["type"] == "function"
    assert payload["function"]["parameters"]["required"] == ["expression"]

    # 模拟一段 OpenAI 线上格式响应（arguments 是 JSON 字符串）
    raw_response = {
        "tool_calls": [
            {
                "id": "call_abc",
                "type": "function",
                "function": {
                    "name": "calculator",
                    "arguments": json.dumps({"expression": "(1 + 2) * 3"}),
                },
            }
        ]
    }
    calls = parse_tool_calls(raw_response)
    assert calls[0].id == "call_abc"
    assert calls[0].arguments == {"expression": "(1 + 2) * 3"}
    print(f"[parse] 解析结果: {calls[0]}")

    validate_arguments(spec, calls[0].arguments)  # 合法参数通过
    try:
        validate_arguments(spec, {"expression": 123})
        raise AssertionError("类型错误应被拦截")
    except SchemaValidationError as exc:
        print(f"[validate] 类型错误被拦截: {exc}")


def demo_agent_retry() -> None:
    """FunctionCallingAgent：schema 失败 -> 反馈重试 -> 成功 -> 最终答案."""
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    llm = MockLLM(
        tool_call_responses=[
            # 第 1 轮：模型漏了必填参数 expression
            {"tool_calls": [{"name": "calculator", "arguments": {}}]},
            # 第 2 轮：看到错误反馈后修正参数
            {"tool_calls": [{"name": "calculator", "arguments": {"expression": "(1 + 2) * 3"}}]},
            # 第 3 轮：拿到结果，给出最终答案
            {"content": "(1 + 2) * 3 = 9"},
        ]
    )
    agent = FunctionCallingAgent(llm, registry, max_retries=2)
    answer = agent.run("计算 (1 + 2) * 3")
    print(f"[agent] 最终答案: {answer}")
    assert answer == "(1 + 2) * 3 = 9"

    # 轨迹断言：第一次失败（缺参）、第二次成功（结果为 9）
    (call1, result1, ok1), (call2, result2, ok2) = agent.call_history
    assert not ok1 and "缺少必填参数" in result1
    assert ok2 and result2 == "9"
    print(f"[agent] 第 1 次调用失败并反馈: {result1}")
    print(f"[agent] 第 2 次调用成功: {call2.arguments} -> {result2}")

    # 错误反馈确实进入了第 2 轮调用的上下文
    feedback = [
        m for m in llm.calls[1] if m.role == "tool" and "参数校验失败" in m.content
    ]
    assert feedback, "schema 错误必须以 tool 消息回填给 LLM"
    # 工具结果以 tool role 消息回填
    tool_results = [m for m in llm.calls[2] if m.role == "tool" and "[calculator] 9" in m.content]
    assert tool_results, "工具执行结果必须以 tool 消息回填"
    print("[agent] 失败反馈与工具结果均以 tool 角色消息正确回填")


if __name__ == "__main__":
    demo_spec_and_parse()
    demo_agent_retry()
    print("全部断言通过")
