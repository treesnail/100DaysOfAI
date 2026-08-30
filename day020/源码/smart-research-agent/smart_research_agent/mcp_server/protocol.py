"""MCP 协议的 pydantic 数据模型.

按项目共享规格建模 JSON-RPC 2.0 消息与 MCP 能力描述符，
供 Client / Server 两侧共用，保证双方对同一份契约编程。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

JSONRPC_VERSION = "2.0"


class JsonRpcRequest(BaseModel):
    """JSON-RPC 2.0 请求."""

    id: int | str
    method: str
    params: dict[str, Any] = Field(default_factory=dict)
    jsonrpc: str = JSONRPC_VERSION


class JsonRpcResponse(BaseModel):
    """JSON-RPC 2.0 成功响应."""

    id: int | str
    result: Any = None
    jsonrpc: str = JSONRPC_VERSION


class JsonRpcErrorDetail(BaseModel):
    """JSON-RPC 2.0 错误对象."""

    code: int
    message: str
    data: Any = None


class JsonRpcError(BaseModel):
    """JSON-RPC 2.0 错误响应."""

    id: int | str | None
    error: JsonRpcErrorDetail
    jsonrpc: str = JSONRPC_VERSION


class ResourceDescriptor(BaseModel):
    """MCP 资源能力描述."""

    uri: str
    name: str
    description: str = ""
    mime_type: str = "text/plain"


class ToolDescriptor(BaseModel):
    """MCP 工具能力描述."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )


class PromptDescriptor(BaseModel):
    """MCP 提示词能力描述."""

    name: str
    description: str = ""
    arguments: list[dict[str, Any]] = Field(default_factory=list)
