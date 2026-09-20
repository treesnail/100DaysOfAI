"""MCP Tools 与 Prompts 测试.

两组验证路径：
  - 内存内会话（create_connected_server_and_client_session）：同进程直连，
    毫秒级、确定性，覆盖 list_tools / call_tool / get_prompt 主路径；
  - stdio 子进程（McpClient）：真实拉起 server 进程，验证端到端集成。
全部离线：没有任何网络访问，也不依赖 LLM。
"""

from __future__ import annotations

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from smart_research_agent.mcp_server.client import McpClient
from smart_research_agent.mcp_server.protocol import (
    JsonRpcError,
    JsonRpcErrorBody,
    JsonRpcRequest,
    JsonRpcResponse,
    PromptDescriptor,
    ResourceDescriptor,
    ToolDescriptor,
)
from smart_research_agent.mcp_server.server import app


class TestProtocolModels:
    """protocol.py：JSON-RPC 2.0 消息与能力描述符的建模."""

    def test_request_defaults_to_jsonrpc_2(self):
        req = JsonRpcRequest(id=1, method="tools/call", params={"name": "calculator"})
        assert req.jsonrpc == "2.0"
        assert req.params == {"name": "calculator"}

    def test_response_carries_result(self):
        resp = JsonRpcResponse(id="abc", result={"content": []})
        assert resp.id == "abc"
        assert resp.jsonrpc == "2.0"

    def test_error_body_has_code_and_message(self):
        err = JsonRpcError(id=None, error=JsonRpcErrorBody(code=-32601, message="Method not found"))
        assert err.error.code == -32601
        assert "not found" in err.error.message

    def test_roundtrip_serialization(self):
        req = JsonRpcRequest(id=7, method="prompts/get", params={"name": "research_report"})
        restored = JsonRpcRequest.model_validate(req.model_dump())
        assert restored == req

    def test_descriptors(self):
        tool = ToolDescriptor(
            name="calculator",
            description="计算",
            input_schema={"type": "object", "properties": {"expression": {"type": "string"}}},
        )
        assert tool.input_schema["properties"]["expression"]["type"] == "string"
        res = ResourceDescriptor(uri="kb://docs/intro", name="intro")
        assert res.mime_type == "text/plain"
        prompt = PromptDescriptor(name="research_report", arguments=["topic", "depth"])
        assert prompt.arguments == ["topic", "depth"]


class TestMcpTools:
    """内存内会话：Tool 原语的注册、发现与调用."""

    async def test_list_tools_contains_calculator_and_search(self):
        async with create_connected_server_and_client_session(app) as session:
            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            assert "calculator" in names
            assert "knowledge_search" in names

    async def test_calculator_schema_advertises_expression_param(self):
        async with create_connected_server_and_client_session(app) as session:
            tools = await session.list_tools()
            calc = next(t for t in tools.tools if t.name == "calculator")
            assert "expression" in calc.inputSchema["properties"]
            assert calc.inputSchema["required"] == ["expression"]

    async def test_call_calculator_returns_correct_result(self):
        async with create_connected_server_and_client_session(app) as session:
            result = await session.call_tool("calculator", {"expression": "2 + 3 * 4"})
            assert result.isError is False
            assert result.content[0].text == "14"

    async def test_call_calculator_with_invalid_expression(self):
        async with create_connected_server_and_client_session(app) as session:
            result = await session.call_tool("calculator", {"expression": "import os"})
            # 业务错误以文本形式返回（"计算失败: ..."），而非协议级错误
            assert result.isError is False
            assert "计算失败" in result.content[0].text

    async def test_knowledge_search_retrieves_relevant_record(self):
        async with create_connected_server_and_client_session(app) as session:
            result = await session.call_tool("knowledge_search", {"query": "ReAct 推理"})
            text = result.content[0].text
            assert "ReAct" in text
            assert "相似度" in text


class TestMcpPrompts:
    """内存内会话：Prompt 原语的发现与动态参数渲染."""

    async def test_list_prompts_contains_research_report(self):
        async with create_connected_server_and_client_session(app) as session:
            prompts = await session.list_prompts()
            report = next(p for p in prompts.prompts if p.name == "research_report")
            arg_names = [a.name for a in report.arguments or []]
            assert "topic" in arg_names
            assert "depth" in arg_names

    async def test_get_prompt_renders_topic(self):
        async with create_connected_server_and_client_session(app) as session:
            result = await session.get_prompt("research_report", {"topic": "RAG 技术选型"})
            assert len(result.messages) == 1
            message = result.messages[0]
            assert message.role == "user"
            assert "RAG 技术选型" in message.content.text
            assert "调研报告" in message.content.text

    async def test_get_prompt_uses_default_and_custom_depth(self):
        async with create_connected_server_and_client_session(app) as session:
            default = await session.get_prompt("research_report", {"topic": "MCP"})
            custom = await session.get_prompt(
                "research_report", {"topic": "MCP", "depth": "简要"}
            )
            assert "深度：详细" in default.messages[0].content.text
            assert "深度：简要" in custom.messages[0].content.text


class TestMcpClient:
    """stdio 子进程：McpClient 的真实端到端集成."""

    async def test_call_before_connect_raises(self):
        client = McpClient()
        with pytest.raises(RuntimeError, match="尚未连接"):
            await client.list_tools()

    async def test_stdio_end_to_end(self):
        async with McpClient() as client:
            tools = await client.list_tools()
            names = [t.name for t in tools]
            assert "calculator" in names
            assert "knowledge_search" in names

            answer = await client.call_tool("calculator", {"expression": "6 * 7"})
            assert "42" in answer

            hits = await client.call_tool("knowledge_search", {"query": "向量检索"})
            assert "向量" in hits

            messages = await client.get_prompt("research_report", {"topic": "MCP 协议"})
            assert any("MCP 协议" in m for m in messages)

    async def test_session_closed_after_context_exit(self):
        client = McpClient()
        await client.connect()
        await client.close()
        with pytest.raises(RuntimeError, match="尚未连接"):
            await client.list_tools()
