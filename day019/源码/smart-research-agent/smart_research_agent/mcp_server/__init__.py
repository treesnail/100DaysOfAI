"""MCP Server 子包：智研 AI 助手的模型上下文协议服务端.

注意：包名固定为 ``mcp_server``，绝不能命名为 ``mcp``——后者会遮蔽
官方 ``mcp`` 第三方包，导致 ``from mcp.server.fastmcp import FastMCP``
之类的导入解析到本项目内部模块而失败。
"""

from smart_research_agent.mcp_server.protocol import (
    JsonRpcError,
    JsonRpcErrorDetail,
    JsonRpcRequest,
    JsonRpcResponse,
    PromptDescriptor,
    ResourceDescriptor,
    ToolDescriptor,
)

__all__ = [
    "JsonRpcError",
    "JsonRpcErrorDetail",
    "JsonRpcRequest",
    "JsonRpcResponse",
    "PromptDescriptor",
    "ResourceDescriptor",
    "ToolDescriptor",
]
