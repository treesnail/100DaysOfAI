"""MCP 协议模型：用 pydantic 对 JSON-RPC 2.0 与能力描述符建模.

MCP 的消息层是 JSON-RPC 2.0，本模块定义了三个消息模型
（Request / Response / Error）和三个能力描述符
（Resource / Tool / Prompt）。它们既是教学用的协议抽象，
也是 docs/mcp_capabilities.md 能力文档的数据来源。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class JsonRpcRequest(BaseModel):
    """JSON-RPC 2.0 请求：调用方发出的方法调用."""

    id: int | str = Field(description="请求 ID，响应按 ID 与请求配对")
    method: str = Field(description="方法名，如 tools/call、resources/read")
    params: dict[str, Any] = Field(default_factory=dict, description="方法参数")
    jsonrpc: Literal["2.0"] = Field(default="2.0", description="协议版本，固定为 2.0")


class JsonRpcResponse(BaseModel):
    """JSON-RPC 2.0 成功响应."""

    id: int | str = Field(description="与请求相同的 ID")
    result: dict[str, Any] = Field(default_factory=dict, description="方法执行结果")
    jsonrpc: Literal["2.0"] = Field(default="2.0", description="协议版本，固定为 2.0")


class JsonRpcErrorBody(BaseModel):
    """JSON-RPC 2.0 错误体：code 标识类别，message 面向人."""

    code: int = Field(description="错误码，如 -32601 表示方法不存在")
    message: str = Field(description="人类可读的错误描述")


class JsonRpcError(BaseModel):
    """JSON-RPC 2.0 错误响应."""

    id: int | str = Field(description="与请求相同的 ID")
    error: JsonRpcErrorBody = Field(description="错误体")
    jsonrpc: Literal["2.0"] = Field(default="2.0", description="协议版本，固定为 2.0")


class ResourceDescriptor(BaseModel):
    """资源能力描述符：一个可被读取的 MCP Resource."""

    uri: str = Field(description="资源 URI 或 URI 模板，如 research://knowledge/{doc_id}")
    name: str = Field(description="资源名称")
    description: str = Field(default="", description="资源用途说明")
    mime_type: str = Field(default="text/plain", description="资源内容的 MIME 类型")


class ToolDescriptor(BaseModel):
    """工具能力描述符：一个可被调用的 MCP Tool."""

    name: str = Field(description="工具名称")
    description: str = Field(default="", description="工具用途说明")
    input_schema: dict[str, Any] = Field(
        default_factory=dict, description="参数的 JSON Schema 描述"
    )


class PromptArgumentDescriptor(BaseModel):
    """Prompt 模板的一个参数."""

    name: str = Field(description="参数名")
    description: str = Field(default="", description="参数说明")
    required: bool = Field(default=False, description="是否必填")


class PromptDescriptor(BaseModel):
    """提示词能力描述符：一个可被获取的 MCP Prompt 模板."""

    name: str = Field(description="Prompt 名称")
    description: str = Field(default="", description="Prompt 用途说明")
    arguments: list[PromptArgumentDescriptor] = Field(
        default_factory=list, description="模板参数列表"
    )
