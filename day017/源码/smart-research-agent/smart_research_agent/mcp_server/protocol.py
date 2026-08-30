"""MCP 协议消息与能力描述的 pydantic 模型.

MCP 的传输层承载 JSON-RPC 2.0 消息。这里用 pydantic 对三类消息
（请求 / 成功响应 / 错误响应）与三类能力描述（Resource / Tool / Prompt）
建模，作为项目内 MCP 相关代码的共享词汇表。

注意：本包名固定为 ``smart_research_agent.mcp_server``，绝不能叫 ``mcp``，
否则会遮蔽官方 ``mcp`` 包（FastMCP 所在包）导致 import 冲突。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# JSON-RPC 2.0 规范预定义的错误码，供抛错时引用
PARSE_ERROR = -32700  # 收到非法 JSON
INVALID_REQUEST = -32600  # 不是合法的 Request 对象
METHOD_NOT_FOUND = -32601  # 方法不存在
INVALID_PARAMS = -32602  # 参数非法
INTERNAL_ERROR = -32603  # 服务器内部错误


class JsonRpcRequest(BaseModel):
    """JSON-RPC 2.0 请求消息.

    ``id`` 由客户端生成，服务器必须在响应中原样带回，
    客户端借此把响应与请求配对（一对多并发时尤其重要）。
    """

    jsonrpc: Literal["2.0"] = "2.0"
    id: int | str
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


class JsonRpcResponse(BaseModel):
    """JSON-RPC 2.0 成功响应消息."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: int | str
    result: Any = None


class JsonRpcErrorBody(BaseModel):
    """JSON-RPC 2.0 错误对象（error 字段的内层结构）."""

    code: int
    message: str
    data: Any = None


class JsonRpcError(BaseModel):
    """JSON-RPC 2.0 错误响应消息.

    与 :class:`JsonRpcResponse` 互斥：一条响应要么带 ``result``，
    要么带 ``error``，绝不同时出现。
    """

    jsonrpc: Literal["2.0"] = "2.0"
    id: int | str | None
    error: JsonRpcErrorBody


class ResourceDescriptor(BaseModel):
    """资源（Resource）能力描述：服务器对外暴露的一份可读取的数据."""

    uri: str
    name: str
    description: str = ""
    mime_type: str = "text/plain"


class ToolDescriptor(BaseModel):
    """工具（Tool）能力描述：服务器对外暴露的一个可执行行为.

    ``input_schema`` 是 JSON Schema 对象，描述调用参数的形状，
    客户端（尤其是 LLM）依据它构造合法调用。
    """

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)


class PromptDescriptor(BaseModel):
    """提示词模板（Prompt）能力描述：可参数化的提示词骨架."""

    name: str
    description: str = ""
    arguments: list[str] = Field(default_factory=list)
