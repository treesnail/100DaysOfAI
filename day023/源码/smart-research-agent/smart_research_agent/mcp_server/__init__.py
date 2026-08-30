"""MCP Server/Client 子包：协议模型、FastMCP 服务与 stdio 客户端.

注意：包名固定为 ``smart_research_agent.mcp_server``，绝不能命名为 ``mcp``，
否则会遮蔽官方 ``mcp`` 包导致 import 冲突。
"""

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
]
