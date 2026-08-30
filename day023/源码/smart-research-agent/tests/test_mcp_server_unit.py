"""MCP Server 单元测试（内存传输：真实协议握手，但不起子进程、不走管道）.

使用 ``create_connected_server_and_client_session`` 在内存中连接真实的
Server 实例，验证 calculator 工具、knowledge 资源与 research_report 提示词
每一个能力的行为。asyncio_mode = "auto"，异步测试函数无需额外装饰器。
"""

from __future__ import annotations

import pytest

from mcp.shared.memory import create_connected_server_and_client_session

from smart_research_agent.mcp_server.server import KNOWLEDGE_BASE, app


def _connect():
    """返回内存会话的异步上下文管理器（测试内用 async with 消费）."""
    return create_connected_server_and_client_session(app._mcp_server)


class TestToolCapability:
    async def test_list_tools_contains_calculator(self):
        async with _connect() as session:
            tools = (await session.list_tools()).tools
        names = [t.name for t in tools]
        assert "calculator" in names
        calc = next(t for t in tools if t.name == "calculator")
        assert "expression" in calc.inputSchema["properties"]

    async def test_call_calculator_success(self):
        async with _connect() as session:
            result = await session.call_tool("calculator", {"expression": "2 + 3 * (4 - 1)"})
        assert result.isError is False
        assert result.content[0].text == "11"

    async def test_call_calculator_invalid_expression(self):
        async with _connect() as session:
            result = await session.call_tool("calculator", {"expression": "1 +"})
        # 工具内部容错：返回"计算失败"文本而不是协议级错误
        assert result.isError is False
        assert result.content[0].text.startswith("计算失败")


class TestResourceCapability:
    async def test_list_resource_templates(self):
        async with _connect() as session:
            templates = (await session.list_resource_templates()).resourceTemplates
        uris = [str(t.uriTemplate) for t in templates]
        assert "research://knowledge/{doc_id}" in uris

    async def test_read_existing_document(self):
        async with _connect() as session:
            result = await session.read_resource("research://knowledge/rag-intro")
        assert result.contents[0].text == KNOWLEDGE_BASE["rag-intro"]
        assert result.contents[0].mimeType == "text/plain"

    async def test_read_unknown_document_raises(self):
        async with _connect() as session:
            with pytest.raises(Exception, match="rag-intro|不存在|Error"):
                await session.read_resource("research://knowledge/not-exist")


class TestPromptCapability:
    async def test_list_prompts(self):
        async with _connect() as session:
            prompts = (await session.list_prompts()).prompts
        names = [p.name for p in prompts]
        assert "research_report" in names
        report = next(p for p in prompts if p.name == "research_report")
        required = [a.name for a in report.arguments if a.required]
        assert required == ["topic"]

    async def test_get_prompt_fills_arguments(self):
        async with _connect() as session:
            result = await session.get_prompt(
                "research_report", {"topic": "RAG 选型", "style": "学术"}
            )
        text = result.messages[0].content.text
        assert "RAG 选型" in text
        assert "学术" in text

    async def test_get_prompt_uses_default_style(self):
        async with _connect() as session:
            result = await session.get_prompt("research_report", {"topic": "Agent 架构"})
        assert "正式" in result.messages[0].content.text
