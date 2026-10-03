"""function_calling 模块的单元测试（day037）."""

from __future__ import annotations

import json

import pytest

from smart_research_agent.llm.base import Message
from smart_research_agent.llm.function_calling import (
    FunctionSpec,
    SchemaValidationError,
    ToolCallError,
    parse_tool_calls,
    specs_from_tools,
    validate_arguments,
)
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.tools.calculator import CalculatorTool


@pytest.fixture
def calc_spec() -> FunctionSpec:
    return FunctionSpec.from_tool(CalculatorTool())


class TestFunctionSpec:
    def test_from_tool_copies_metadata(self, calc_spec):
        assert calc_spec.name == "calculator"
        assert "计算数学表达式" in calc_spec.description
        assert calc_spec.parameters["required"] == ["expression"]

    def test_to_openai_tool_format(self, calc_spec):
        payload = calc_spec.to_openai_tool()
        assert payload["type"] == "function"
        assert payload["function"]["name"] == "calculator"
        assert payload["function"]["parameters"]["properties"]["expression"]["type"] == "string"

    def test_specs_from_tools_batch(self):
        specs = specs_from_tools([CalculatorTool(), CalculatorTool()])
        assert [s.name for s in specs] == ["calculator", "calculator"]

    def test_schema_and_tool_share_source(self, calc_spec):
        """schema 直接引用工具的 parameters 元数据——同一份事实，永不漂移."""
        assert calc_spec.parameters is CalculatorTool().parameters or (
            calc_spec.parameters == CalculatorTool().parameters
        )


class TestParseToolCalls:
    def test_parse_json_string_arguments(self):
        response = {
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": json.dumps({"expression": "1+1"}),
                    },
                }
            ]
        }
        calls = parse_tool_calls(response)
        assert len(calls) == 1
        assert calls[0].id == "call_1"
        assert calls[0].name == "calculator"
        assert calls[0].arguments == {"expression": "1+1"}

    def test_parse_dict_arguments(self):
        """部分 SDK 会代理解析 arguments，两种形态都要兼容."""
        response = {
            "tool_calls": [{"function": {"name": "calculator", "arguments": {"expression": "2*3"}}}]
        }
        calls = parse_tool_calls(response)
        assert calls[0].arguments == {"expression": "2*3"}
        assert calls[0].id == "call_0"  # 缺 id 时自动补

    def test_empty_or_missing_tool_calls(self):
        assert parse_tool_calls({}) == []
        assert parse_tool_calls({"content": "直接回答"}) == []
        assert parse_tool_calls({"tool_calls": []}) == []

    def test_malformed_json_raises_schema_error(self):
        response = {
            "tool_calls": [{"function": {"name": "calculator", "arguments": "{bad json"}}]
        }
        with pytest.raises(SchemaValidationError, match="不是合法 JSON"):
            parse_tool_calls(response)

    def test_non_object_arguments_raises(self):
        response = {
            "tool_calls": [{"function": {"name": "calculator", "arguments": "[1, 2]"}}]
        }
        with pytest.raises(SchemaValidationError, match="必须是 JSON 对象"):
            parse_tool_calls(response)


class TestValidateArguments:
    def test_valid_arguments_pass(self, calc_spec):
        validate_arguments(calc_spec, {"expression": "1+1"})  # 不抛异常即通过

    def test_missing_required_raises(self, calc_spec):
        with pytest.raises(SchemaValidationError, match="缺少必填参数"):
            validate_arguments(calc_spec, {})

    def test_wrong_type_raises(self, calc_spec):
        with pytest.raises(SchemaValidationError, match="应为 string"):
            validate_arguments(calc_spec, {"expression": 42})

    def test_bool_rejected_for_number(self):
        """Python 里 bool 是 int 子类，但 JSON Schema 里 boolean 与 number 互不相容."""
        spec = FunctionSpec(
            name="t",
            description="t",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "number"}},
                "required": ["x"],
            },
        )
        with pytest.raises(SchemaValidationError, match="boolean"):
            validate_arguments(spec, {"x": True})

    def test_undeclared_fields_allowed(self, calc_spec):
        """additionalProperties 默认 True：schema 未声明的字段不做限制."""
        validate_arguments(calc_spec, {"expression": "1+1", "extra": "ok"})

    def test_error_kinds_are_distinct(self):
        from smart_research_agent.llm.function_calling import (
            ToolExecutionError,
            UnknownToolError,
        )

        assert SchemaValidationError.kind == "schema"
        assert UnknownToolError.kind == "unknown_tool"
        assert ToolExecutionError.kind == "execution"
        for cls in (SchemaValidationError, UnknownToolError, ToolExecutionError):
            assert issubclass(cls, ToolCallError)


class TestMockLLMFunctionCalling:
    def test_scripted_tool_call_normalized_to_openai_format(self):
        llm = MockLLM(
            tool_call_responses=[
                {"tool_calls": [{"name": "calculator", "arguments": {"expression": "1+1"}}]}
            ]
        )
        response = llm.chat_with_tools([Message(role="user", content="算一下")], tools=[])
        call = response["tool_calls"][0]
        assert call["type"] == "function"
        assert call["id"] == "call_0"
        assert call["function"]["name"] == "calculator"
        # arguments 被序列化为 JSON 字符串（线上协议形态），parse_tool_calls 能解回
        assert isinstance(call["function"]["arguments"], str)
        assert parse_tool_calls(response)[0].arguments == {"expression": "1+1"}

    def test_scripted_content_response(self):
        llm = MockLLM(tool_call_responses=[{"content": "最终答案"}])
        response = llm.chat_with_tools([Message(role="user", content="hi")], tools=[])
        assert response == {"content": "最终答案"}
        assert "tool_calls" not in response

    def test_script_exhaustion_falls_back_to_content(self):
        llm = MockLLM(tool_call_responses=[], default="兜底回答")
        response = llm.chat_with_tools([Message(role="user", content="hi")], tools=[])
        assert response == {"content": "兜底回答"}

    def test_usage_logged(self):
        llm = MockLLM(tool_call_responses=[{"content": "ok"}])
        llm.chat_with_tools([Message(role="user", content="统计一下用量")], tools=[])
        assert len(llm.usage_log) == 1
        assert llm.total_prompt_tokens > 0
        assert len(llm.tool_calls_log) == 1

    def test_base_llm_default_raises(self):
        """不支持 function calling 的 LLM 应明确报错，引导走 ReAct 文本协议."""
        from smart_research_agent.llm.base import BaseLLM

        class TextOnlyLLM(BaseLLM):
            def chat(self, messages, temperature=0.7, max_tokens=1024):
                return "text"

        with pytest.raises(NotImplementedError, match="不支持 function calling"):
            TextOnlyLLM().chat_with_tools([], tools=[])
