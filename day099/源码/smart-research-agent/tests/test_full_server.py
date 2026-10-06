"""full_server 的内存会话集成测试.

使用 ``create_connected_server_and_client_session`` 在同一进程内建立
客户端-服务器会话：不经过 stdio/SSE 传输、不访问网络、完全离线。
"""

from __future__ import annotations

import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import AnyUrl

from smart_research_agent.mcp_server.full_server import (
    SERVER_INSTRUCTIONS,
    SERVER_NAME,
    capability_descriptors,
    create_server,
)
from smart_research_agent.mcp_server.protocol import (
    JsonRpcError,
    JsonRpcErrorDetail,
    JsonRpcRequest,
    JsonRpcResponse,
    PromptDescriptor,
    ResourceDescriptor,
    ToolDescriptor,
)


class TestInitializeHandshake:
    """initialize 握手：客户端拿到 Server 的元数据与能力声明."""

    async def test_server_metadata(self):
        server = create_server()
        async with create_connected_server_and_client_session(server) as session:
            result = await session.initialize()
            assert result.serverInfo.name == SERVER_NAME
            assert result.instructions == SERVER_INSTRUCTIONS

    async def test_capabilities_declared(self):
        server = create_server()
        async with create_connected_server_and_client_session(server) as session:
            result = await session.initialize()
            # 三类能力都应在握手中声明，客户端据此决定可以调用哪些方法
            assert result.capabilities.tools is not None
            assert result.capabilities.resources is not None
            assert result.capabilities.prompts is not None


class TestCapabilityListing:
    """三类能力均可列出，且内容与服务端注册的一致."""

    async def test_list_tools(self):
        server = create_server()
        async with create_connected_server_and_client_session(server) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            assert names == ["calculate"]
            assert "expression" in tools.tools[0].inputSchema["properties"]

    async def test_list_resource_templates(self):
        server = create_server()
        async with create_connected_server_and_client_session(server) as session:
            await session.initialize()
            # 带 {doc_id} 参数的是资源模板，出现在 templates 列表而非 resources
            templates = await session.list_resource_templates()
            uris = [t.uriTemplate for t in templates.resourceTemplates]
            assert uris == ["research://knowledge/{doc_id}"]

    async def test_list_prompts(self):
        server = create_server()
        async with create_connected_server_and_client_session(server) as session:
            await session.initialize()
            prompts = await session.list_prompts()
            names = [p.name for p in prompts.prompts]
            assert names == ["research_report"]
            assert prompts.prompts[0].arguments[0].name == "topic"


class TestToolCalls:
    """工具调用：正常路径返回值，错误路径返回结构化错误而非崩溃."""

    async def test_calculate_success(self):
        server = create_server()
        async with create_connected_server_and_client_session(server) as session:
            await session.initialize()
            result = await session.call_tool("calculate", {"expression": "2 + 3 * 4"})
            assert result.isError is False
            assert result.content[0].text == "14"

    async def test_calculate_error_returns_iserror(self):
        server = create_server()
        async with create_connected_server_and_client_session(server) as session:
            await session.initialize()
            result = await session.call_tool("calculate", {"expression": "1 / 0"})
            # 业务失败必须转为 isError=true 的错误结果，连接保持可用
            assert result.isError is True
            assert "计算失败" in result.content[0].text
            # 会话未被错误破坏，后续调用依然可用
            ok = await session.call_tool("calculate", {"expression": "1 + 1"})
            assert ok.isError is False

    async def test_calculate_rejects_malicious_expression(self):
        server = create_server()
        async with create_connected_server_and_client_session(server) as session:
            await session.initialize()
            result = await session.call_tool("calculate", {"expression": "__import__('os')"})
            assert result.isError is True


class TestResourcesAndPrompts:
    """资源读取与提示词获取的端到端验证."""

    async def test_read_knowledge_resource(self):
        server = create_server()
        async with create_connected_server_and_client_session(server) as session:
            await session.initialize()
            result = await session.read_resource(AnyUrl("research://knowledge/rag_overview"))
            assert "RAG" in result.contents[0].text

    async def test_read_unknown_doc_fails_cleanly(self):
        server = create_server()
        async with create_connected_server_and_client_session(server) as session:
            await session.initialize()
            with pytest.raises(Exception, match="不存在"):
                await session.read_resource(AnyUrl("research://knowledge/no_such_doc"))

    async def test_read_rejects_unsafe_doc_id(self):
        server = create_server()
        async with create_connected_server_and_client_session(server) as session:
            await session.initialize()
            with pytest.raises(Exception, match="非法"):
                await session.read_resource(AnyUrl("research://knowledge/..%2Fsecret"))

    async def test_get_prompt(self):
        server = create_server()
        async with create_connected_server_and_client_session(server) as session:
            await session.initialize()
            result = await session.get_prompt("research_report", {"topic": "RAG 框架"})
            assert result.messages[0].role == "user"
            assert "RAG 框架" in result.messages[0].content.text


class TestProtocolModels:
    """protocol.py 的 pydantic 模型：JSON-RPC 2.0 消息与能力描述符."""

    def test_request_defaults_jsonrpc(self):
        req = JsonRpcRequest(id=1, method="tools/list")
        assert req.jsonrpc == "2.0"
        assert req.params == {}

    def test_response_round_trip(self):
        resp = JsonRpcResponse(id="abc", result={"tools": []})
        assert resp.model_dump() == {"id": "abc", "result": {"tools": []}, "jsonrpc": "2.0"}

    def test_error_model(self):
        err = JsonRpcError(
            id=7,
            error=JsonRpcErrorDetail(code=-32601, message="Method not found"),
        )
        assert err.error.code == -32601
        assert err.jsonrpc == "2.0"

    def test_descriptors(self):
        res = ResourceDescriptor(uri="research://knowledge/{doc_id}", name="read_knowledge")
        assert res.mime_type == "text/plain"
        tool = ToolDescriptor(name="calculate", input_schema={"type": "object"})
        assert tool.description == ""
        prompt = PromptDescriptor(name="research_report", arguments=[{"name": "topic"}])
        # 统一规范版 protocol 会把 dict 参数解析为 PromptArgument 对象
        assert prompt.arguments[0].name == "topic"

    def test_capability_descriptors_match_server(self):
        """静态能力清单与 Server 注册的三类能力一一对应."""
        caps = capability_descriptors()
        assert caps["resources"][0].uri == "research://knowledge/{doc_id}"
        assert caps["tools"][0].name == "calculate"
        assert caps["prompts"][0].name == "research_report"
