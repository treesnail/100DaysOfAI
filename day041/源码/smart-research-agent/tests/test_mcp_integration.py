"""MCP 端到端集成测试（真实 stdio：子进程起 Server，走完整握手与调用）.

每个测试通过 ``McpClient.stdio()`` 以
``StdioServerParameters(command=sys.executable, args=["-m", "smart_research_agent.mcp_server.server"])``
启动真实子进程，验证进程间 JSON-RPC 通信的完整链路。
"""

from __future__ import annotations

from smart_research_agent.mcp_server.client import McpClient


class TestStdioHandshake:
    async def test_capability_discovery_over_stdio(self):
        """完整握手：initialize + 三类能力发现."""
        async with McpClient.stdio() as client:
            tool_names = [t.name for t in client.tools]
            resource_uris = [r.uri for r in client.resources]
            prompt_names = [p.name for p in client.prompts]

        assert "calculator" in tool_names
        assert "research://knowledge/{doc_id}" in resource_uris
        assert "research_report" in prompt_names

        calc = next(t for t in client.tools if t.name == "calculator")
        assert calc.input_schema["properties"]["expression"]["type"] == "string"
        prompt = next(p for p in client.prompts if p.name == "research_report")
        assert prompt.arguments[0].name == "topic"


class TestStdioEndToEnd:
    async def test_call_tool_over_stdio(self):
        async with McpClient.stdio() as client:
            text = await client.call_tool("calculator", {"expression": "(2 + 3) * 4"})
        assert text == "20"

    async def test_read_resource_over_stdio(self):
        async with McpClient.stdio() as client:
            text = await client.read_resource("research://knowledge/mcp-intro")
        assert "MCP" in text
        assert "Resources" in text

    async def test_get_prompt_over_stdio(self):
        async with McpClient.stdio() as client:
            text = await client.get_prompt("research_report", {"topic": "向量数据库"})
        assert "向量数据库" in text
        assert "正式" in text
