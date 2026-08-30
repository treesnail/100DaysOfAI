"""MCP 线协议（wire protocol）与能力描述的 pydantic 模型.

MCP 的底层传输格式是 JSON-RPC 2.0：无论是 stdio 还是 SSE 传输，
线上跑的都是 ``{"jsonrpc": "2.0", "id": ..., "method": ...}`` 这样的消息。
本模块用 pydantic 对这些消息与三类能力描述（Resource / Tool / Prompt）建模，
供文档生成、能力清单导出与协议级测试使用。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class JsonRpcRequest(BaseModel):
    """JSON-RPC 2.0 请求：Client → Server 的调用消息."""

    id: str | int | None = None
    method: str
    params: dict[str, Any] = Field(default_factory=dict)
    jsonrpc: Literal["2.0"] = "2.0"


class JsonRpcResponse(BaseModel):
    """JSON-RPC 2.0 成功响应：携带与请求对应的 id 和 result."""

    id: str | int | None = None
    result: Any = None
    jsonrpc: Literal["2.0"] = "2.0"


class JsonRpcErrorDetail(BaseModel):
    """JSON-RPC 2.0 错误对象：code 为整数错误码，message 为人类可读描述."""

    code: int
    message: str


class JsonRpcError(BaseModel):
    """JSON-RPC 2.0 错误响应."""

    id: str | int | None = None
    error: JsonRpcErrorDetail
    jsonrpc: Literal["2.0"] = "2.0"


class ResourceDescriptor(BaseModel):
    """Resource 能力描述：一份可被 Client 读取的内容（文档、配置等）."""

    uri: str
    name: str
    description: str = ""
    mime_type: str = "text/plain"


class ToolDescriptor(BaseModel):
    """Tool 能力描述：一个可被 Client 调用的函数，参数用 JSON Schema 描述."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)


class PromptDescriptor(BaseModel):
    """Prompt 能力描述：一个可参数化的提示词模板."""

    name: str
    description: str = ""
    arguments: list[dict[str, Any]] = Field(default_factory=list)
