"""MCP Client：通过 stdio 传输连接 MCP Server，做能力发现与调用.

用法::

    async with McpClient.stdio() as client:
        print(client.tools)          # 能力发现的结果
        text = await client.call_tool("calculator", {"expression": "1+2"})
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from smart_research_agent.mcp_server.protocol import (
    PromptDescriptor,
    ResourceDescriptor,
    ToolDescriptor,
)

# 项目根目录：子进程从这里启动，保证能 import 到 smart_research_agent 包
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class McpClient:
    """stdio 传输的 MCP 客户端：连接、能力发现、调用三类能力."""

    def __init__(self, session: ClientSession):
        self._session = session
        self.tools: list[ToolDescriptor] = []
        self.resources: list[ResourceDescriptor] = []
        self.prompts: list[PromptDescriptor] = []

    @classmethod
    def stdio(
        cls,
        command: str | None = None,
        args: list[str] | None = None,
        cwd: Path | None = None,
    ) -> "McpClientConnector":
        """构造一个指向本项目 MCP Server 的 stdio 连接器.

        默认以 ``python -m smart_research_agent.mcp_server.server`` 起子进程。
        """
        params = StdioServerParameters(
            command=command or sys.executable,
            args=args or ["-m", "smart_research_agent.mcp_server.server"],
            cwd=str(cwd or PROJECT_ROOT),
        )
        return McpClientConnector(params)

    async def discover_capabilities(self) -> None:
        """初始化握手后，拉取 Server 声明的全部能力并转为描述符."""
        await self._session.initialize()

        tools = await self._session.list_tools()
        self.tools = [
            ToolDescriptor(
                name=t.name,
                description=t.description or "",
                input_schema=t.inputSchema,
            )
            for t in tools.tools
        ]

        templates = await self._session.list_resource_templates()
        self.resources = [
            ResourceDescriptor(
                uri=str(t.uriTemplate),
                name=t.name,
                description=t.description or "",
                mime_type=t.mimeType or "text/plain",
            )
            for t in templates.resourceTemplates
        ]

        prompts = await self._session.list_prompts()
        self.prompts = [
            PromptDescriptor(
                name=p.name,
                description=p.description or "",
                arguments=[
                    {"name": a.name, "description": a.description or "", "required": bool(a.required)}
                    for a in (p.arguments or [])
                ],
            )
            for p in prompts.prompts
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """调用一个 MCP Tool，返回其文本结果."""
        result = await self._session.call_tool(name, arguments)
        if result.isError:
            text = result.content[0].text if result.content else "未知错误"
            raise RuntimeError(f"工具调用失败 [{name}]: {text}")
        return "".join(c.text for c in result.content if getattr(c, "type", None) == "text")

    async def read_resource(self, uri: str) -> str:
        """读取一个 MCP Resource，返回其文本内容."""
        result = await self._session.read_resource(uri)
        return "".join(c.text for c in result.contents if getattr(c, "text", None) is not None)

    async def get_prompt(self, name: str, arguments: dict[str, str]) -> str:
        """获取一个填充参数后的 MCP Prompt，返回其全部消息文本."""
        result = await self._session.get_prompt(name, arguments)
        return "\n".join(
            m.content.text for m in result.messages if getattr(m.content, "text", None) is not None
        )


class McpClientConnector:
    """异步上下文管理器：进入时建立 stdio 连接并完成能力发现，退出时关闭."""

    def __init__(self, params: StdioServerParameters):
        self._params = params
        self._client: McpClient | None = None
        self._stdio_cm = None
        self._session_cm = None

    async def __aenter__(self) -> McpClient:
        self._stdio_cm = stdio_client(self._params)
        read, write = await self._stdio_cm.__aenter__()
        self._session_cm = ClientSession(read, write)
        session = await self._session_cm.__aenter__()
        self._client = McpClient(session)
        await self._client.discover_capabilities()
        return self._client

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._session_cm is not None:
            await self._session_cm.__aexit__(exc_type, exc_val, exc_tb)
        if self._stdio_cm is not None:
            await self._stdio_cm.__aexit__(exc_type, exc_val, exc_tb)
