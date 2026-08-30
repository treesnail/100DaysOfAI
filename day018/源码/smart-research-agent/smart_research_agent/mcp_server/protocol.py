"""MCP 线协议的 pydantic 模型：JSON-RPC 2.0 消息与能力描述符.

MCP 的所有通信都是 JSON-RPC 2.0 消息。这里的模型是协议的"教学镜像"：
官方 ``mcp`` SDK 内部有自己的类型，但我们用 pydantic 显式建模一遍，
既能在 Client/Server 边界做校验，也让协议结构在代码里可读。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# JSON-RPC 2.0 消息
# ---------------------------------------------------------------------------


class JsonRpcRequest(BaseModel):
    """JSON-RPC 2.0 请求：Client → Server 的调用."""

    id: int | str
    method: str
    params: dict[str, Any] = Field(default_factory=dict)
    jsonrpc: str = "2.0"


class JsonRpcResponse(BaseModel):
    """JSON-RPC 2.0 成功响应：携带与请求相同的 id."""

    id: int | str
    result: Any = None
    jsonrpc: str = "2.0"


class JsonRpcErrorBody(BaseModel):
    """错误对象：code 是协议级错误码，message 是人类可读说明."""

    code: int
    message: str


class JsonRpcError(BaseModel):
    """JSON-RPC 2.0 错误响应：id 可能为 None（无法解析请求 id 时）."""

    id: int | str | None
    error: JsonRpcErrorBody
    jsonrpc: str = "2.0"


# ---------------------------------------------------------------------------
# 能力描述符：Resources / Tools / Prompts 三类原语的元数据
# ---------------------------------------------------------------------------


class ResourceDescriptor(BaseModel):
    """Resource 能力描述：一份可被读取的静态/动态内容."""

    uri: str
    name: str
    description: str = ""
    mime_type: str = "text/plain"


class ToolDescriptor(BaseModel):
    """Tool 能力描述：一个可被模型调用的函数.

    ``input_schema`` 是 JSON Schema——它是 LLM 选择工具、构造参数的
    唯一依据，schema 的质量直接决定工具被用对还是被用错。
    """

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)


class PromptDescriptor(BaseModel):
    """Prompt 能力描述：一个参数化的提示词模板."""

    name: str
    description: str = ""
    arguments: list[str] = Field(default_factory=list)
