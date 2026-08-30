"""MCP 协议模型单元测试（测试金字塔最底层：纯内存、无网络、毫秒级）."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from smart_research_agent.mcp_server.protocol import (
    JsonRpcError,
    JsonRpcRequest,
    JsonRpcResponse,
    PromptDescriptor,
    ResourceDescriptor,
    ToolDescriptor,
)


class TestJsonRpcRequest:
    def test_default_jsonrpc_and_params(self):
        req = JsonRpcRequest(id=1, method="tools/call")
        assert req.jsonrpc == "2.0"
        assert req.params == {}

    def test_round_trip(self):
        req = JsonRpcRequest(id="abc", method="resources/read", params={"uri": "research://knowledge/rag-intro"})
        restored = JsonRpcRequest.model_validate(req.model_dump())
        assert restored == req

    def test_wrong_version_rejected(self):
        with pytest.raises(ValidationError):
            JsonRpcRequest(id=1, method="tools/call", jsonrpc="1.0")

    def test_missing_required_field_rejected(self):
        with pytest.raises(ValidationError):
            JsonRpcRequest(id=1)  # 缺 method


class TestJsonRpcResponse:
    def test_success_response(self):
        resp = JsonRpcResponse(id=1, result={"content": []})
        assert resp.jsonrpc == "2.0"
        assert resp.result == {"content": []}

    def test_error_response(self):
        err = JsonRpcError(id=1, error={"code": -32601, "message": "Method not found"})
        assert err.error.code == -32601
        assert "not found" in err.error.message

    def test_error_body_requires_code_and_message(self):
        with pytest.raises(ValidationError):
            JsonRpcError(id=1, error={"code": -32601})  # 缺 message


class TestCapabilityDescriptors:
    def test_resource_descriptor(self):
        res = ResourceDescriptor(uri="research://knowledge/{doc_id}", name="knowledge")
        assert res.mime_type == "text/plain"
        assert res.description == ""

    def test_tool_descriptor_input_schema(self):
        tool = ToolDescriptor(
            name="calculator",
            input_schema={"type": "object", "properties": {"expression": {"type": "string"}}},
        )
        assert tool.input_schema["properties"]["expression"]["type"] == "string"

    def test_prompt_descriptor_arguments(self):
        prompt = PromptDescriptor(
            name="research_report",
            arguments=[{"name": "topic", "required": True}, {"name": "style", "required": False}],
        )
        assert prompt.arguments[0].required is True
        assert prompt.arguments[1].name == "style"
