"""MCP 协议基础模型：JSON-RPC 2.0 消息与能力描述符.

MCP 在传输层之上跑的是 JSON-RPC 2.0：每条消息都是一个 JSON 对象，
用 ``jsonrpc`` 字段锁定协议版本，用 ``id`` 关联请求与响应。
本模块用 pydantic 对这些消息与三类能力描述符（Resource / Tool /
Prompt）建模，为文档生成、能力注册表和协议学习提供权威的数据契约。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

JSONRPC_VERSION = "2.0"


class JsonRpcRequest(BaseModel):
    """JSON-RPC 2.0 请求消息."""

    id: int | str
    method: str
    params: dict[str, Any] = Field(default_factory=dict)
    jsonrpc: str = JSONRPC_VERSION


class JsonRpcResponse(BaseModel):
    """JSON-RPC 2.0 成功响应消息."""

    id: int | str
    result: Any = None
    jsonrpc: str = JSONRPC_VERSION


class JsonRpcErrorDetail(BaseModel):
    """JSON-RPC 2.0 错误对象：``error`` 字段的内部结构."""

    code: int
    message: str
    data: Any = None


class JsonRpcError(BaseModel):
    """JSON-RPC 2.0 错误响应消息."""

    id: int | str | None
    error: JsonRpcErrorDetail
    jsonrpc: str = JSONRPC_VERSION


class ResourceDescriptor(BaseModel):
    """资源能力描述符：对应 MCP ``resources/list`` 返回的条目."""

    uri: str
    name: str
    description: str = ""
    mime_type: str = "text/plain"


class ToolDescriptor(BaseModel):
    """工具能力描述符：对应 MCP ``tools/list`` 返回的条目."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)


class PromptDescriptor(BaseModel):
    """提示词能力描述符：对应 MCP ``prompts/list`` 返回的条目."""

    name: str
    description: str = ""
    arguments: list[dict[str, Any]] = Field(default_factory=list)
