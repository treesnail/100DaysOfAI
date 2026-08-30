"""MCP Server 测试：协议模型、内存会话端到端、健康检查路由、入口参数.

全部离线运行：MCP 端到端测试使用 ``create_connected_server_and_client_session``
在进程内建立客户端-服务器会话，不经过任何网络；``/health`` 路由用 Starlette
TestClient 直接驱动 ASGI 应用。
"""

from __future__ import annotations

import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from starlette.testclient import TestClient

from smart_research_agent.mcp_server.protocol import (
    JsonRpcError,
    JsonRpcRequest,
    JsonRpcResponse,
    PromptDescriptor,
    ResourceDescriptor,
    ToolDescriptor,
)
from smart_research_agent.mcp_server.server import (
    KNOWLEDGE_BASE,
    create_server,
    describe_capabilities,
    main,
)


class TestProtocolModels:
    """protocol.py 的 JSON-RPC 2.0 与能力描述模型."""

    def test_request_defaults(self):
        req = JsonRpcRequest(method="tools/list")
        assert req.jsonrpc == "2.0"
        assert req.id is None
        assert req.params == {}

    def test_request_round_trip(self):
        req = JsonRpcRequest(id=1, method="tools/call", params={"name": "calculator"})
        restored = JsonRpcRequest.model_validate_json(req.model_dump_json())
        assert restored == req

    def test_response_defaults(self):
        resp = JsonRpcResponse(id="abc", result={"content": []})
        assert resp.jsonrpc == "2.0"
        assert resp.result == {"content": []}

    def test_error_model(self):
        err = JsonRpcError(id=2, error={"code": -32601, "message": "Method not found"})
        assert err.error.code == -32601
        assert "not found" in err.error.message

    def test_invalid_jsonrpc_version_rejected(self):
        with pytest.raises(Exception, match="jsonrpc"):
            JsonRpcRequest(method="ping", jsonrpc="1.0")  # type: ignore[arg-type]

    def test_descriptors(self):
        res = ResourceDescriptor(uri="research://knowledge/{doc_id}", name="knowledge")
        assert res.mime_type == "text/plain"
        tool = ToolDescriptor(name="calculator", input_schema={"type": "object"})
        assert tool.input_schema["type"] == "object"
        prompt = PromptDescriptor(name="research-report", arguments=[{"name": "topic"}])
        assert prompt.arguments[0]["name"] == "topic"


class TestServerInMemory:
    """通过内存会话对 Server 做端到端测试（无网络、无子进程）."""

    async def test_list_tools_contains_calculator(self):
        server = create_server()
        async with create_connected_server_and_client_session(server) as session:
            tools = await session.list_tools()
        names = [t.name for t in tools.tools]
        assert "calculator" in names

    async def test_call_calculator_tool(self):
        server = create_server()
        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool("calculator", {"expression": "2 + 3 * 4"})
        assert result.content[0].text == "14"

    async def test_call_calculator_invalid_expression(self):
        server = create_server()
        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool("calculator", {"expression": "import os"})
        assert "计算失败" in result.content[0].text

    async def test_read_knowledge_resource(self):
        server = create_server()
        doc_id = next(iter(KNOWLEDGE_BASE))
        async with create_connected_server_and_client_session(server) as session:
            result = await session.read_resource(f"research://knowledge/{doc_id}")
        assert result.contents[0].text == KNOWLEDGE_BASE[doc_id]

    async def test_read_unknown_resource_raises(self):
        server = create_server()
        async with create_connected_server_and_client_session(server) as session:
            with pytest.raises(Exception, match="未知文档"):
                await session.read_resource("research://knowledge/not-exist")

    async def test_server_name(self):
        server = create_server()
        assert server.name == "smart-research-agent"


class TestHealthRoute:
    """/health 自定义路由：TestClient 直接驱动 SSE 应用的 ASGI 接口."""

    def test_health_returns_ok(self):
        app = create_server().sse_app()
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "server": "smart-research-agent"}

    def test_health_rejects_post(self):
        app = create_server().sse_app()
        client = TestClient(app)
        resp = client.post("/health")
        assert resp.status_code == 405


class TestConfigAndEntrypoint:
    """环境变量注入与命令行入口."""

    def test_host_port_from_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MCP_HOST", "127.0.0.1")
        monkeypatch.setenv("MCP_PORT", "9999")
        server = create_server()
        assert server.settings.host == "127.0.0.1"
        assert server.settings.port == 9999

    def test_default_port_is_8000(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("MCP_PORT", raising=False)
        assert create_server().settings.port == 8000

    def test_describe_capabilities(self):
        caps = describe_capabilities()
        assert caps["tools"][0]["name"] == "calculator"
        assert caps["resources"][0]["uri"] == "research://knowledge/{doc_id}"
        assert caps["prompts"] == []

    @pytest.mark.parametrize("transport", ["stdio", "sse"])
    def test_main_dispatches_transport(
        self, monkeypatch: pytest.MonkeyPatch, transport: str
    ):
        calls: list[str] = []
        monkeypatch.setattr(
            "smart_research_agent.mcp_server.server.FastMCP.run",
            lambda self, **kw: calls.append(kw["transport"]),
        )
        main(["--transport", transport])
        assert calls == [transport]

    def test_main_rejects_unknown_transport(self):
        with pytest.raises(SystemExit):
            main(["--transport", "websocket"])
