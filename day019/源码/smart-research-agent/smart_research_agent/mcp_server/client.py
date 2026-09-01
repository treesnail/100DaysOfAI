"""MCP Client：以 stdio 传输启动 server 子进程并与之会话.

进程模型：Client 通过 ``StdioServerParameters + stdio_client`` 把 MCP Server
拉成一个子进程，父子进程之间用 stdin/stdout 双向管道传 JSON-RPC 消息——
Server 的 stdout 被协议独占，因此 Server 侧的日志必须走 stderr/文件。

用法::

    async with McpClient() as client:
        tools = await client.list_tools()
        answer = await client.call_tool("calculator", {"expression": "6 * 7"})
"""

from __future__ import annotations

import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from smart_research_agent.mcp_server.protocol import ToolDescriptor
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

# client.py → mcp_server/ → smart_research_agent/ → 项目根
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class McpClient:
    """MCP 客户端：托管 server 子进程的生命周期与 JSON-RPC 会话."""

    def __init__(self, command: str | None = None, args: list[str] | None = None) -> None:
        # 把项目根注入子进程 PYTHONPATH：venv 里可能装着旧版本的同名包，
        # 显式注入保证子进程加载的是"本项目这份代码"而不是 site-packages 里的。
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in [str(_PROJECT_ROOT), env.get("PYTHONPATH", "")] if part
        )
        self._params = StdioServerParameters(
            command=command or sys.executable,
            args=args or ["-m", "smart_research_agent.mcp_server.server"],
            env=env,
            cwd=str(_PROJECT_ROOT),
        )
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def connect(self) -> "McpClient":
        """启动 server 子进程并完成 MCP 初始化握手."""
        self._stack = AsyncExitStack()
        read, write = await self._stack.enter_async_context(stdio_client(self._params))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        logger.info("MCP 会话已建立: %s %s", self._params.command, self._params.args)
        return self

    async def close(self) -> None:
        """关闭会话并终止子进程."""
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
            self._session = None

    async def __aenter__(self) -> "McpClient":
        return await self.connect()

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError("McpClient 尚未连接，请先 connect() 或使用 async with")
        return self._session

    async def list_tools(self) -> list[ToolDescriptor]:
        """列出 server 注册的全部工具，转换为本项目的 ToolDescriptor."""
        result = await self._require_session().list_tools()
        return [
            ToolDescriptor(
                name=t.name,
                description=t.description or "",
                input_schema=dict(t.inputSchema),
            )
            for t in result.tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """调用 server 侧工具，返回拼接后的文本结果；协议级错误抛 RuntimeError."""
        result = await self._require_session().call_tool(name, arguments)
        text = "".join(c.text for c in result.content if c.type == "text")
        if result.isError:
            raise RuntimeError(f"工具调用失败: {text}")
        return text

    async def get_prompt(self, name: str, arguments: dict[str, str] | None = None) -> list[str]:
        """获取渲染后的提示词，返回各消息的文本内容."""
        result = await self._require_session().get_prompt(name, arguments or {})
        return [m.content.text for m in result.messages if m.content.type == "text"]
