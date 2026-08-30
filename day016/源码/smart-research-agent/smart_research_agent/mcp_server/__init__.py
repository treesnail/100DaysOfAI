"""MCP Server 模块：SmartResearch Agent 的 Model Context Protocol 服务端.

注意：包名固定为 ``mcp_server``，绝不能命名为 ``mcp``，否则会遮蔽
已安装的官方 ``mcp`` 包，导致 ``import mcp`` 解析到本包。
"""

from smart_research_agent.mcp_server.protocol import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    JSONRPC_VERSION,
    MCP_PROTOCOL_VERSION,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    ErrorObject,
    JsonRpcError,
    JsonRpcRequest,
    JsonRpcResponse,
    PromptArgument,
    PromptDescriptor,
    ResourceDescriptor,
    ToolDescriptor,
    build_initialize_request,
    build_initialize_response,
)

__all__ = [
    "INTERNAL_ERROR",
    "INVALID_PARAMS",
    "INVALID_REQUEST",
    "JSONRPC_VERSION",
    "MCP_PROTOCOL_VERSION",
    "METHOD_NOT_FOUND",
    "PARSE_ERROR",
    "ErrorObject",
    "JsonRpcError",
    "JsonRpcRequest",
    "JsonRpcResponse",
    "PromptArgument",
    "PromptDescriptor",
    "ResourceDescriptor",
    "ToolDescriptor",
    "build_initialize_request",
    "build_initialize_response",
]
