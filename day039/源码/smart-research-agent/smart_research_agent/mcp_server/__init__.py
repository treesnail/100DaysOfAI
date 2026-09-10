"""smart_research_agent 自建的 MCP 相关包.

注意：包名绝不能叫 ``mcp``，否则会遮蔽官方 ``mcp`` 包。
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
