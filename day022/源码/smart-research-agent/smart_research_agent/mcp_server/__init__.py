"""MCP Server 子包：SmartResearch Agent 的能力对外暴露层.

注意：包名必须是 ``smart_research_agent.mcp_server``，绝不能命名为 ``mcp``，
否则会遮蔽（shadow）官方 ``mcp`` 包，导致 ``from mcp.server.fastmcp import ...``
导入到本项目自己的模块。
"""

from __future__ import annotations

from typing import Any

from smart_research_agent.mcp_server.protocol import (
    JsonRpcError,
    JsonRpcRequest,
    JsonRpcResponse,
    PromptDescriptor,
    ResourceDescriptor,
    ToolDescriptor,
)

__all__ = [
    "JsonRpcError",
    "JsonRpcRequest",
    "JsonRpcResponse",
    "PromptDescriptor",
    "ResourceDescriptor",
    "ToolDescriptor",
    "create_server",
    "describe_capabilities",
]


def __getattr__(name: str) -> Any:
    """惰性导出 server 模块的工厂函数.

    避免 ``python -m smart_research_agent.mcp_server.server`` 时
    server 模块先经包导入进入 sys.modules 而产生 RuntimeWarning。
    """
    if name in ("create_server", "describe_capabilities"):
        from smart_research_agent.mcp_server import server

        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
