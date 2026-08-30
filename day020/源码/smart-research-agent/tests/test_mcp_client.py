"""MCP Client 集成测试：内存 Server + MockLLM，全程离线."""

from __future__ import annotations

import asyncio
import sys

import pytest
from mcp import StdioServerParameters
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session

from smart_research_agent.agent.react_agent import ReactAgent
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.mcp_server.client import (
    McpClient,
    McpClientError,
    McpToolCallError,
)
from smart_research_agent.mcp_server.demo_server import build_demo_server
from smart_research_agent.mcp_server.protocol import (
    JsonRpcError,
    JsonRpcErrorDetail,
    JsonRpcRequest,
    JsonRpcResponse,
    ResourceDescriptor,
    ToolDescriptor,
)
from smart_research_agent.tools.mcp_tool import McpToolAdapter, discover_mcp_tools
from smart_research_agent.tools.registry import ToolRegistry


@pytest.fixture
def demo_server() -> FastMCP:
    """现场构建一个内存 MCP Server（含正常工具、报错工具、资源与提示词）."""
    return build_demo_server()


@pytest.fixture
def mcp_client(demo_server: FastMCP):
    """已握手的 McpClient：内存会话的生命周期托管在常驻 loop 的同一任务里."""
    import threading

    client = McpClient()
    holder: dict = {}
    ready = threading.Event()
    stop = threading.Event()

    async def _session_lifecycle():
        # 进入与退出必须在同一个任务内（anyio cancel scope 约束）
        async with create_connected_server_and_client_session(demo_server) as session:
            holder["session"] = session
            ready.set()
            await asyncio.get_running_loop().run_in_executor(None, stop.wait)

    future = client.submit(_session_lifecycle())
    assert ready.wait(timeout=10), "内存会话建立超时"
    client.attach_session(holder["session"])
    yield client
    stop.set()
    future.result(timeout=10)
    client.shutdown()


class TestProtocolModels:
    """共享协议模型（JSON-RPC 2.0 + 能力描述符）."""

    def test_request_defaults(self):
        req = JsonRpcRequest(id=1, method="tools/list")
        assert req.jsonrpc == "2.0"
        assert req.params == {}

    def test_response_round_trip(self):
        resp = JsonRpcResponse(id="abc", result={"tools": []})
        loaded = JsonRpcResponse.model_validate_json(resp.model_dump_json())
        assert loaded.result == {"tools": []}
        assert loaded.jsonrpc == "2.0"

    def test_error_response(self):
        err = JsonRpcError(
            id=1, error=JsonRpcErrorDetail(code=-32601, message="Method not found")
        )
        assert err.error.code == -32601
        assert "Method not found" in err.error.message

    def test_tool_descriptor_default_schema(self):
        desc = ToolDescriptor(name="t")
        assert desc.input_schema == {"type": "object", "properties": {}}
        res = ResourceDescriptor(uri="docs://a", name="a")
        assert res.mime_type == "text/plain"


class TestMcpClientDiscovery:
    """连接握手与能力发现（异步测试，pytest-asyncio auto 模式）."""

    async def test_list_tools(self, mcp_client: McpClient):
        tools = await mcp_client.list_tools()
        names = {t.name for t in tools}
        assert {"search_docs", "word_count", "unstable_tool"} <= names
        search = next(t for t in tools if t.name == "search_docs")
        assert "检索" in search.description
        assert search.input_schema["required"] == ["query"]

    async def test_list_resources(self, mcp_client: McpClient):
        resources = await mcp_client.list_resources()
        assert any(r.uri == "docs://readme" for r in resources)

    async def test_list_prompts(self, mcp_client: McpClient):
        prompts = await mcp_client.list_prompts()
        report = next(p for p in prompts if p.name == "research_report")
        assert report.arguments[0]["name"] == "topic"
        assert report.arguments[0]["required"] is True

    async def test_call_tool_returns_text(self, mcp_client: McpClient):
        text = await mcp_client.call_tool("search_docs", {"query": "MCP"})
        assert "检索结果" in text
        assert "MCP" in text

    async def test_call_tool_server_error_raises(self, mcp_client: McpClient):
        with pytest.raises(McpToolCallError, match="unstable_tool"):
            await mcp_client.call_tool("unstable_tool", {"trigger": "x"})

    async def test_not_connected_raises(self):
        client = McpClient()
        try:
            assert client.connected is False
            with pytest.raises(McpClientError, match="尚未连接"):
                await client.list_tools()
        finally:
            client.shutdown()


class TestStdioConnect:
    """真实 stdio 子进程连接：覆盖 connect/initialize/disconnect 全链路."""

    async def test_connect_handshake_and_call(self):
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "smart_research_agent.mcp_server.demo_server"],
        )
        client = McpClient(params)
        try:
            async with client:
                assert client.connected is True
                tools = await client.list_tools()
                assert "search_docs" in {t.name for t in tools}
                text = await client.call_tool("search_docs", {"query": "stdio"})
                assert "stdio" in text
            assert client.connected is False
        finally:
            client.shutdown()


class TestMcpToolAdapter:
    """远程工具本地化：适配器属性、执行与回退."""

    def _get_adapter(self, client: McpClient, name: str) -> McpToolAdapter:
        descriptor = next(t for t in client.run(client.list_tools()) if t.name == name)
        return McpToolAdapter(client=client, descriptor=descriptor)

    def test_properties_from_descriptor(self, mcp_client: McpClient):
        adapter = self._get_adapter(mcp_client, "search_docs")
        assert adapter.name == "search_docs"
        assert "检索" in adapter.description
        assert adapter.parameters["required"] == ["query"]
        schema = adapter.to_schema()
        assert schema["name"] == "search_docs"

    def test_execute_returns_remote_result(self, mcp_client: McpClient):
        adapter = self._get_adapter(mcp_client, "search_docs")
        result = adapter.execute(query="RAG")
        assert "检索结果" in result

    def test_execute_fallback_on_server_error(self, mcp_client: McpClient):
        adapter = self._get_adapter(mcp_client, "unstable_tool")
        result = adapter.execute(trigger="boom")
        # 回退策略：返回错误说明字符串而不是抛异常
        assert "调用失败" in result
        assert "unstable_tool" in result

    def test_discover_and_register(self, mcp_client: McpClient):
        adapters = mcp_client.run(discover_mcp_tools(mcp_client))
        assert len(adapters) >= 3
        registry = ToolRegistry()
        for adapter in adapters:
            registry.register(adapter)
        assert isinstance(registry.get("search_docs"), McpToolAdapter)
        assert "search_docs" in registry.describe()


class TestReActIntegration:
    """MCP 工具注册进 ToolRegistry 后，可被 ReAct 流程（MockLLM）调用."""

    def _build_agent(self, client: McpClient, responses: list[str]) -> ReactAgent:
        registry = ToolRegistry()
        for adapter in client.run(discover_mcp_tools(client)):
            registry.register(adapter)
        return ReactAgent(llm=MockLLM(responses=responses), registry=registry)

    def test_react_calls_mcp_tool(self, mcp_client: McpClient):
        agent = self._build_agent(
            mcp_client,
            [
                "Thought: 需要先检索资料\nAction: search_docs\nAction Input: RAG 技术",
                "Thought: 资料足够\nFinal Answer: RAG 是检索增强生成",
            ],
        )
        answer = agent.run("调研 RAG 技术")
        assert answer == "RAG 是检索增强生成"
        # 第二次 LLM 调用的上下文中应包含远程工具的 Observation
        second_call = agent.llm.calls[1]
        observation_msg = second_call[-1].content
        assert "检索结果" in observation_msg

    def test_react_fallback_not_crash(self, mcp_client: McpClient):
        agent = self._build_agent(
            mcp_client,
            [
                "Thought: 试试这个工具\nAction: unstable_tool\nAction Input: 任意输入",
                "Thought: 工具挂了，直接回答\nFinal Answer: 工具不可用，改用已有知识回答",
            ],
        )
        answer = agent.run("任意任务")
        assert "改用已有知识" in answer
        second_call = agent.llm.calls[1]
        assert "调用失败" in second_call[-1].content
