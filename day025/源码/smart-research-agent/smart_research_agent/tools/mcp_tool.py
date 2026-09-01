"""远程工具本地化：把 MCP Server 上的工具包装成本地 BaseTool."""

from __future__ import annotations

import asyncio
from typing import Any

from smart_research_agent.mcp_server.client import McpClient
from smart_research_agent.mcp_server.protocol import ToolDescriptor
from smart_research_agent.tools.base import BaseTool
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)


class McpToolAdapter(BaseTool):
    """把一个远程 MCP tool 适配成本地 BaseTool.

    name / description / parameters 直接来自能力发现得到的 ToolDescriptor，
    execute 通过常驻连接的 McpClient 发起远程调用；
    任何异常（网络错误、Server 业务错误）都按回退策略转成错误说明字符串，
    保证 ReAct 循环不被远程故障打崩。
    """

    def __init__(self, client: McpClient, descriptor: ToolDescriptor):
        self._client = client
        self._descriptor = descriptor

    @property
    def name(self) -> str:
        return self._descriptor.name

    @property
    def description(self) -> str:
        return self._descriptor.description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._descriptor.input_schema

    def execute(self, **kwargs: Any) -> str:
        try:
            # ReAct 循环是同步代码，这里用 asyncio.run 逐次桥接异步调用；
            # client 内部会把协程调度到常驻 loop，跨 loop 安全。
            return asyncio.run(self._client.call_tool(self.name, kwargs))
        except Exception as exc:  # noqa: BLE001 —— 回退策略：吞掉一切远程异常
            logger.warning("远程工具 %s 调用失败: %s", self.name, exc)
            return f"远程工具 {self.name} 调用失败: {exc}"


async def discover_mcp_tools(client: McpClient) -> list[McpToolAdapter]:
    """发现 Server 上的全部工具并包装成适配器列表.

    调用方把返回值逐个注册进 ToolRegistry 即可::

        for tool in await discover_mcp_tools(client):
            registry.register(tool)
    """
    descriptors = await client.list_tools()
    adapters = [McpToolAdapter(client=client, descriptor=d) for d in descriptors]
    logger.info("发现远程工具 %d 个: %s", len(adapters), [a.name for a in adapters])
    return adapters
