"""SmartResearch 自建 MCP 服务包.

包名固定为 ``mcp_server`` 而非 ``mcp``：Python 的 import 机制会优先
命中本项目内的同名包，若叫 ``mcp`` 会遮蔽官方 ``mcp`` 包（FastMCP 所在包）。
"""

from smart_research_agent.mcp_server.protocol import (
    JsonRpcError,
    JsonRpcErrorBody,
    JsonRpcRequest,
    JsonRpcResponse,
    PromptDescriptor,
    ResourceDescriptor,
    ToolDescriptor,
)

__all__ = [
    "JsonRpcError",
    "JsonRpcErrorBody",
    "JsonRpcRequest",
    "JsonRpcResponse",
    "PromptDescriptor",
    "ResourceDescriptor",
    "ToolDescriptor",
]
