"""MCP Client：stdio 连接、initialize 握手与能力发现.

设计要点：
  - 客户端持有**常驻事件循环**（独立守护线程），连接与会话在其上长期存活；
  - 连接的整个生命周期（建连 → 握手 → 保活 → 断开）由常驻 loop 上的
    **同一个协程任务**托管——anyio 的 cancel scope 要求进出在同一任务，
    拆成多个任务会在断开时抛 "exit cancel scope in a different task"；
  - 对外暴露的 connect / list_tools / call_tool 等方法均为 async，
    内部把真正的 I/O 协程桥接到常驻循环执行，因此：
      * 异步调用方（async 测试、未来的异步 Agent）可以直接 await；
      * 同步调用方（当前 ReAct 循环）可以用 asyncio.run(...) 逐次调用，
        不会踩到"跨事件循环使用同一条流"的坑。
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from concurrent.futures import Future as ConcurrentFuture
from contextlib import AsyncExitStack
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
        self._params = server_params
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
