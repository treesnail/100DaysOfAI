"""MCP Client：stdio 连接、initialize 握手与能力发现.

提供两种用法：

1. ``McpClient``（day019 引入，day020 兼容层）：常驻事件循环托管连接生命周期，
   方法级能力发现（``list_tools()`` / ``call_tool()`` 等），支持异步 await
   与同步 asyncio.run 桥接两种调用方式；
2. ``McpClient.stdio()`` + ``McpClientConnector``（day023 引入）：
   一个异步上下文管理器，进入时建立 stdio 连接并完成能力发现，
   返回带 ``tools`` / ``resources`` / ``prompts`` 属性的能力视图。

设计要点（McpClient 常驻 loop 方案）：
  - 客户端持有**常驻事件循环**（独立守护线程），连接与会话在其上长期存活；
  - 连接的整个生命周期（建连 → 握手 → 保活 → 断开）由常驻 loop 上的
    **同一个协程任务**托管——anyio 的 cancel scope 要求进出在同一任务，
    拆成多个任务会在断开时抛 "exit cancel scope in a different task"；
  - 对外暴露的 connect / list_tools / call_tool 等方法均为 async，
    内部把真正的 I/O 协程桥接到常驻循环执行，因此：
      * 异步调用方（async 测试、未来的异步 Agent）可以直接 await；
      * 同步调用方（当前 ReAct 循环）可以用 asyncio.run(...) 逐次调用，
        不会踩到"跨事件循环使用同一条流"的坑。

用法（day023 能力视图）::

    async with McpClient.stdio() as client:
        print(client.tools)          # 能力发现的结果
        text = await client.call_tool("calculator", {"expression": "1+2"})
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from collections.abc import Coroutine
from concurrent.futures import Future as ConcurrentFuture
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, TypeVar

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from smart_research_agent.mcp_server.protocol import (
    PromptDescriptor,
    ResourceDescriptor,
    ToolDescriptor,
)
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

_CONNECT_TIMEOUT = 10.0

# client.py → mcp_server/ → smart_research_agent/ → 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 兼容旧代码里的私有名引用
_PROJECT_ROOT = PROJECT_ROOT


def _stdio_env() -> dict[str, str]:
    """子进程环境：注入项目根到 PYTHONPATH.

    venv 里可能装着旧版本的同名包，显式注入保证子进程加载的是
    "本项目这份代码"而不是 site-packages 里的。
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in [str(PROJECT_ROOT), env.get("PYTHONPATH", "")] if part
    )
    return env


def _default_server_params() -> StdioServerParameters:
    """默认连接参数：拉起本项目自带的 MCP server 子进程（day018 行为）."""
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "smart_research_agent.mcp_server.server"],
        env=_stdio_env(),
        cwd=str(PROJECT_ROOT),
    )


class McpClientError(RuntimeError):
    """MCP 客户端错误（未连接、连接失败等）."""


class McpToolCallError(RuntimeError):
    """远程工具返回 isError=True 的业务错误."""


class McpClient:
    """封装 MCP stdio 连接的客户端.

    用法（异步）::

        async with McpClient(StdioServerParameters(command="python", args=["server.py"])) as c:
            tools = await c.list_tools()

    用法（同步，经 asyncio.run 桥接）::

        client = McpClient(params)
        asyncio.run(client.connect())
        text = asyncio.run(client.call_tool("search_docs", {"query": "RAG"}))
        client.shutdown()
    """

    def __init__(self, server_params: StdioServerParameters | None = None):
        # server_params 缺省时回退到本项目自带 server（兼容 day018 的 McpClient() 无参用法）
        self._params = server_params if server_params is not None else _default_server_params()
        self._session: ClientSession | None = None
        # 连接生命周期任务的同步原语
        self._conn_future: ConcurrentFuture[None] | None = None
        self._ready_signal: threading.Event | None = None
        self._stop_signal: threading.Event | None = None
        self._conn_error: BaseException | None = None
        # 常驻事件循环：连接一旦建立就常驻其上，直到 shutdown()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="mcp-client-loop", daemon=True
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # day023：能力视图式连接的工厂入口
    # ------------------------------------------------------------------
    @classmethod
    def stdio(
        cls,
        command: str | None = None,
        args: list[str] | None = None,
        cwd: Path | None = None,
    ) -> McpClientConnector:
        """构造一个指向本项目 MCP Server 的 stdio 连接器（day023 用法）.

        默认以 ``python -m smart_research_agent.mcp_server.server`` 起子进程。
        """
        params = StdioServerParameters(
            command=command or sys.executable,
            args=args or ["-m", "smart_research_agent.mcp_server.server"],
            env=_stdio_env(),
            cwd=str(cwd or PROJECT_ROOT),
        )
        return McpClientConnector(params)

    # ------------------------------------------------------------------
    # 桥接层
    # ------------------------------------------------------------------
    def submit(self, coro: Coroutine[Any, Any, T]) -> ConcurrentFuture[T]:
        """把协程提交到常驻 loop 后台执行，立即返回 Future（不等待）."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _submit(self, coro: Coroutine[Any, Any, T]) -> asyncio.Future[T]:
        """提交协程到常驻 loop，包装成可在调用方 loop 上 await 的 Future."""
        return asyncio.wrap_future(self.submit(coro))

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        """同步桥接：在常驻 loop 上执行协程并阻塞等待结果（测试/脚本用）."""
        return self.submit(coro).result()

    # ------------------------------------------------------------------
    # 连接管理：整个生命周期由常驻 loop 上的同一个任务托管
    # ------------------------------------------------------------------
    @property
    def connected(self) -> bool:
        return self._session is not None

    def attach_session(self, session: ClientSession) -> None:
        """接管一个已在外部完成 initialize 的会话（如内存测试会话）.

        注意：该会话必须创建在本 client 的常驻 loop 上，
        测试里通过 ``client.submit(...)`` / ``client.run(...)`` 进入常驻 loop 创建。
        """
        self._session = session

    async def connect(self) -> None:
        """建立 stdio 连接并完成 initialize 握手."""
        if self._session is not None:
            return
        if self._params is None:
            raise McpClientError("未提供 StdioServerParameters，无法建立 stdio 连接")
        self._ready_signal = threading.Event()
        self._stop_signal = threading.Event()
        self._conn_error = None
        self._conn_future = self.submit(self._connection_lifecycle(self._params))
        # 在调用方线程等待握手完成（不阻塞调用方的 loop）
        ok = await asyncio.to_thread(self._ready_signal.wait, _CONNECT_TIMEOUT)
        if not ok:
            raise McpClientError("连接超时：握手未在限定时间内完成")
        if self._conn_error is not None:
            raise McpClientError(f"连接失败: {self._conn_error}") from self._conn_error

    async def _connection_lifecycle(self, params: StdioServerParameters) -> None:
        """建连 → 握手 → 保活 → 断开，全部在同一个任务内完成."""
        assert self._ready_signal is not None and self._stop_signal is not None
        try:
            async with AsyncExitStack() as stack:
                read, write = await stack.enter_async_context(stdio_client(params))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()  # 握手：交换协议版本与双方能力
                self._session = session
                self._ready_signal.set()
                logger.info("MCP 连接已建立并完成 initialize 握手")
                # 保活：阻塞等待 disconnect 信号，期间连接一直可用
                await asyncio.get_running_loop().run_in_executor(None, self._stop_signal.wait)
                logger.info("MCP 连接即将断开")
        except BaseException as exc:
            if not self._ready_signal.is_set():
                # 握手阶段失败：记录错误并唤醒 connect()，由它抛出
                self._conn_error = exc
                self._ready_signal.set()
                return
            raise
        finally:
            self._session = None

    async def disconnect(self) -> None:
        """断开连接（常驻 loop 保留，可重新 connect）."""
        if self._conn_future is not None:
            assert self._stop_signal is not None
            self._stop_signal.set()
            await asyncio.wrap_future(self._conn_future)
            self._conn_future = None
        self._session = None

    async def close(self) -> None:
        """兼容 day018 的关闭语义：断开当前连接（常驻 loop 保留）."""
        await self.disconnect()

    def shutdown(self) -> None:
        """同步收尾：断开连接并停掉常驻 loop（进程/测试结束时调用一次）."""
        if self._conn_future is not None:
            assert self._stop_signal is not None
            self._stop_signal.set()
            self._conn_future.result(timeout=_CONNECT_TIMEOUT)
            self._conn_future = None
        self._session = None
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        self._loop.close()

    async def __aenter__(self) -> McpClient:
        await self.connect()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.disconnect()

    # ------------------------------------------------------------------
    # 能力发现
    # ------------------------------------------------------------------
    def _require_session(self) -> None:
        if self._session is None:
            raise McpClientError("尚未连接，请先 connect()")

    async def list_tools(self) -> list[ToolDescriptor]:
        """发现 Server 暴露的全部工具."""
        self._require_session()
        return await self._submit(self._list_tools_impl())

    async def _list_tools_impl(self) -> list[ToolDescriptor]:
        assert self._session is not None
        result = await self._session.list_tools()
        return [
            ToolDescriptor(
                name=t.name,
                description=t.description or "",
                input_schema=dict(t.inputSchema),
            )
            for t in result.tools
        ]

    async def list_resources(self) -> list[ResourceDescriptor]:
        """发现 Server 暴露的全部资源."""
        self._require_session()
        return await self._submit(self._list_resources_impl())

    async def _list_resources_impl(self) -> list[ResourceDescriptor]:
        assert self._session is not None
        result = await self._session.list_resources()
        return [
            ResourceDescriptor(
                uri=str(r.uri),
                name=r.name,
                description=r.description or "",
                mime_type=r.mimeType or "text/plain",
            )
            for r in result.resources
        ]

    async def list_prompts(self) -> list[PromptDescriptor]:
        """发现 Server 暴露的全部提示词模板."""
        self._require_session()
        return await self._submit(self._list_prompts_impl())

    async def _list_prompts_impl(self) -> list[PromptDescriptor]:
        assert self._session is not None
        result = await self._session.list_prompts()
        return [
            PromptDescriptor(
                name=p.name,
                description=p.description or "",
                arguments=[
                    {
                        "name": a.name,
                        "description": a.description or "",
                        "required": a.required or False,
                    }
                    for a in (p.arguments or [])
                ],
            )
            for p in result.prompts
        ]

    async def get_prompt(self, name: str, arguments: dict[str, str] | None = None) -> list[str]:
        """获取渲染后的提示词，返回各消息的文本内容（day018 起的能力）."""
        self._require_session()
        return await self._submit(self._get_prompt_impl(name, arguments or {}))

    async def _get_prompt_impl(self, name: str, arguments: dict[str, str]) -> list[str]:
        assert self._session is not None
        result = await self._session.get_prompt(name, arguments)
        return [m.content.text for m in result.messages if m.content.type == "text"]

    # ------------------------------------------------------------------
    # 工具调用
    # ------------------------------------------------------------------
    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """调用远程工具，返回拼接后的文本结果.

        Server 端业务错误（isError=True）会抛出 McpToolCallError，
        由上层（如 McpToolAdapter）决定回退策略。
        """
        self._require_session()
        return await self._submit(self._call_tool_impl(name, arguments or {}))

    async def _call_tool_impl(self, name: str, arguments: dict[str, Any]) -> str:
        assert self._session is not None
        result = await self._session.call_tool(name, arguments)
        text = "\n".join(c.text for c in result.content if getattr(c, "type", None) == "text")
        if result.isError:
            raise McpToolCallError(f"远程工具 {name} 返回错误: {text}")
        return text


class DiscoveredMcpClient:
    """day023 能力视图：握手后把 Server 能力缓存为属性，直接在调用方 loop 上调用.

    由 ``McpClientConnector`` 建立并返回；与常驻 loop 的 ``McpClient`` 不同，
    本类的会话完全属于调用方所在的协程任务，因此只能在 ``async with`` 块内使用。
    """

    def __init__(self, session: ClientSession):
        self._session = session
        self.tools: list[ToolDescriptor] = []
        self.resources: list[ResourceDescriptor] = []
        self.prompts: list[PromptDescriptor] = []

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
        self._client: DiscoveredMcpClient | None = None
        self._stdio_cm = None
        self._session_cm = None

    async def __aenter__(self) -> DiscoveredMcpClient:
        self._stdio_cm = stdio_client(self._params)
        read, write = await self._stdio_cm.__aenter__()
        self._session_cm = ClientSession(read, write)
        session = await self._session_cm.__aenter__()
        self._client = DiscoveredMcpClient(session)
        await self._client.discover_capabilities()
        return self._client

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._session_cm is not None:
            await self._session_cm.__aexit__(exc_type, exc_val, exc_tb)
        if self._stdio_cm is not None:
            await self._stdio_cm.__aexit__(exc_type, exc_val, exc_tb)
