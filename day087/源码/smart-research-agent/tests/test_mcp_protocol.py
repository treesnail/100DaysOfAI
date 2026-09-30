"""MCP 协议建模层测试."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from smart_research_agent.mcp_server.protocol import (
    JSONRPC_VERSION,
    MCP_PROTOCOL_VERSION,
    METHOD_NOT_FOUND,
    ErrorObject,
    JsonRpcError,
    JsonRpcRequest,
    JsonRpcResponse,
    PromptDescriptor,
    ResourceDescriptor,
    ToolDescriptor,
    build_initialize_request,
    build_initialize_response,
)


class TestJsonRpcRoundTrip:
    """消息序列化 / 反序列化往返."""

    def test_request_round_trip(self):
        req = JsonRpcRequest(id=1, method="tools/call", params={"name": "calculator"})
        payload = req.model_dump_json()
        restored = JsonRpcRequest.model_validate_json(payload)
        assert restored == req
        assert restored.jsonrpc == JSONRPC_VERSION

    def test_response_round_trip(self):
        resp = JsonRpcResponse(id="req-42", result={"content": "ok"})
        restored = JsonRpcResponse.model_validate_json(resp.model_dump_json())
        assert restored == resp
        assert restored.id == "req-42"

    def test_error_round_trip(self):
        err = JsonRpcError(id=None, error=ErrorObject(code=METHOD_NOT_FOUND, message="方法不存在"))
        restored = JsonRpcError.model_validate_json(err.model_dump_json())
        assert restored == err
        assert restored.error.code == METHOD_NOT_FOUND

    def test_request_default_params_and_jsonrpc(self):
        req = JsonRpcRequest(id=1, method="ping")
        assert req.params == {}
        assert req.jsonrpc == "2.0"


class TestJsonRpcValidation:
    """非法消息必须被拒绝."""

    def test_invalid_jsonrpc_version_raises(self):
        with pytest.raises(ValidationError):
            JsonRpcRequest(id=1, method="ping", jsonrpc="1.0")

    def test_invalid_jsonrpc_version_on_deserialize(self):
        with pytest.raises(ValidationError):
            JsonRpcResponse.model_validate({"jsonrpc": "3.0", "id": 1, "result": {}})

    def test_empty_method_raises(self):
        with pytest.raises(ValidationError):
            JsonRpcRequest(id=1, method="")

    def test_missing_method_rejected(self):
        """day023 补充：method 字段整体缺失同样必须被拒绝."""
        with pytest.raises(ValidationError):
            JsonRpcRequest(id=1)  # type: ignore[call-arg]  # 缺 method

    def test_error_requires_code_and_message(self):
        with pytest.raises(ValidationError):
            ErrorObject(message="缺少 code")
        with pytest.raises(ValidationError):
            JsonRpcError(id=1, error={"code": -32600})


class TestCapabilityDescriptors:
    """Resources / Tools / Prompts 三类能力描述的校验."""

    def test_resource_descriptor_defaults(self):
        res = ResourceDescriptor(uri="memory://notes/today", name="今日笔记")
        assert res.mime_type == "text/plain"
        assert res.description == ""

    def test_resource_uri_and_name_required(self):
        with pytest.raises(ValidationError):
            ResourceDescriptor(uri="", name="x")
        with pytest.raises(ValidationError):
            ResourceDescriptor(uri="memory://a", name="")

    def test_tool_descriptor_input_schema_must_be_dict(self):
        tool = ToolDescriptor(
            name="web_search",
            description="搜索互联网",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
        assert tool.input_schema["required"] == ["query"]
        with pytest.raises(ValidationError):
            ToolDescriptor(name="bad", input_schema="not-a-dict")

    def test_prompt_descriptor_arguments(self):
        prompt = PromptDescriptor(
            name="research_report",
            description="调研报告模板",
            arguments=[
                {"name": "topic", "description": "调研主题", "required": True},
                {"name": "depth"},
            ],
        )
        assert [a.name for a in prompt.arguments] == ["topic", "depth"]
        assert prompt.arguments[0].required is True
        assert prompt.arguments[1].required is False

    def test_descriptor_round_trip(self):
        tool = ToolDescriptor(name="calculator", input_schema={"type": "object"})
        restored = ToolDescriptor.model_validate_json(tool.model_dump_json())
        assert restored == tool


class TestInitializeHandshake:
    """initialize 握手消息的建模."""

    def test_build_initialize_request(self):
        req = build_initialize_request()
        assert req.method == "initialize"
        assert req.params["protocolVersion"] == MCP_PROTOCOL_VERSION
        assert req.params["clientInfo"]["name"] == "smart-research-agent"
        assert "capabilities" in req.params

    def test_build_initialize_response(self):
        resp = build_initialize_response()
        assert resp.result["protocolVersion"] == MCP_PROTOCOL_VERSION
        assert resp.result["serverInfo"]["name"] == "smart-research-agent"
        assert set(resp.result["capabilities"]) == {"resources", "tools", "prompts"}

    def test_handshake_id_must_match(self):
        req = build_initialize_request(request_id="hs-1")
        resp = build_initialize_response(request_id=req.id)
        assert resp.id == req.id

    def test_handshake_round_trip(self):
        req = build_initialize_request(client_name="test-client", request_id=7)
        restored_req = JsonRpcRequest.model_validate_json(req.model_dump_json())
        assert restored_req.params["clientInfo"]["name"] == "test-client"

        resp = build_initialize_response(request_id=restored_req.id)
        restored_resp = JsonRpcResponse.model_validate_json(resp.model_dump_json())
        assert restored_resp.id == 7
        assert restored_resp.result["protocolVersion"] == MCP_PROTOCOL_VERSION
