"""MCP Server 包：把 SmartResearch Agent 的能力以 MCP 原语暴露.

注意：包名是 ``mcp_server`` 而不是 ``mcp``——后者会遮蔽官方 ``mcp`` 包，
导致 ``from mcp.server.fastmcp import FastMCP`` 在子进程/新解释器中解析失败。

``app`` 与 ``McpClient`` 采用惰性加载（``__getattr__``）：避免
``python -m smart_research_agent.mcp_server.server`` 因子模块被包
``__init__`` 提前 import 而触发 runpy 的重复加载警告。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from smart_research_agent.mcp_server.protocol import (
    JsonRpcError,
    JsonRpcRequest,
    JsonRpcResponse,
    PromptDescriptor,
    ResourceDescriptor,
    ToolDescriptor,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from smart_research_agent.mcp_server.client import McpClient

__all__ = [
    "McpClient",
    "JsonRpcError",
    "JsonRpcRequest",
    "JsonRpcResponse",
    "PromptDescriptor",
    "ResourceDescriptor",
    "ToolDescriptor",
    "app",
]


def __getattr__(name: str) -> Any:
    if name == "app":
        from smart_research_agent.mcp_server.server import app

        return app
    if name == "McpClient":
        from smart_research_agent.mcp_server.client import McpClient

        return McpClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
