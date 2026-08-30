"""MCP 协议建模：JSON-RPC 2.0 消息、能力描述与 initialize 握手.

本模块用 pydantic 手工建模 MCP（Model Context Protocol）的核心消息结构，
目的是在引入官方 SDK 之前先吃透协议本身。

对照说明：官方 ``mcp`` 包（已安装在 venv 中）在 ``mcp.types`` 模块里提供了
同一套消息的 pydantic 模型（如 ``mcp.types.JSONRPCRequest``、
``mcp.types.InitializeRequest``）。本模块是它的"教学简化版"：字段更少、
校验更显式，便于逐字段理解协议。day017 起接入 FastMCP 后，线上代码将改用
官方类型，本模块保留为协议学习的参照与单元测试对象。

注意：本包必须命名为 ``mcp_server`` 而非 ``mcp``，否则会遮蔽官方 ``mcp`` 包。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# JSON-RPC 2.0 协议版本标识，所有消息的 jsonrpc 字段都必须等于它
JSONRPC_VERSION = "2.0"

# 本 Server 声明支持的 MCP 协议版本（对应官方 2024-11-05 版规范）
MCP_PROTOCOL_VERSION = "2024-11-05"

# JSON-RPC 2.0 规范保留的标准错误码
PARSE_ERROR = -32700  # 报文不是合法 JSON
INVALID_REQUEST = -32600  # 不是合法的 JSON-RPC 请求
METHOD_NOT_FOUND = -32601  # 方法不存在
INVALID_PARAMS = -32602  # 参数不合法
INTERNAL_ERROR = -32603  # 服务器内部错误


class JsonRpcRequest(BaseModel):
    """JSON-RPC 2.0 请求消息：Client → Server（或反向）调用一个方法."""

    jsonrpc: Literal["2.0"] = JSONRPC_VERSION
    id: int | str
    method: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class JsonRpcResponse(BaseModel):
    """JSON-RPC 2.0 成功响应消息：id 必须与对应请求一致."""

    jsonrpc: Literal["2.0"] = JSONRPC_VERSION
    id: int | str
    result: Any = None


class ErrorObject(BaseModel):
    """JSON-RPC 2.0 错误对象：code 为整数错误码，message 为人类可读描述."""

    code: int
    message: str = Field(min_length=1)
    data: Any = None


class JsonRpcError(BaseModel):
    """JSON-RPC 2.0 错误响应消息：解析失败等场景下 id 可能为 None."""

    jsonrpc: Literal["2.0"] = JSONRPC_VERSION
    id: int | str | None
    error: ErrorObject


class ResourceDescriptor(BaseModel):
    """MCP Resource 能力描述：一份可用 URI 寻址的只读数据."""

    uri: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = ""
    mime_type: str = Field(default="text/plain", min_length=1)


class ToolDescriptor(BaseModel):
    """MCP Tool 能力描述：一个可被模型调用、有副作用的函数."""

    name: str = Field(min_length=1)
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)


class PromptArgument(BaseModel):
    """MCP Prompt 模板的单个参数."""

    name: str = Field(min_length=1)
    description: str = ""
    required: bool = False


class PromptDescriptor(BaseModel):
    """MCP Prompt 能力描述：一个可参数化填充的提示词模板."""

    name: str = Field(min_length=1)
    description: str = ""
    arguments: list[PromptArgument] = Field(default_factory=list)


def build_initialize_request(
    client_name: str = "smart-research-agent",
    client_version: str = "0.1.0",
    *,
    request_id: int | str = 1,
    protocol_version: str = MCP_PROTOCOL_VERSION,
) -> JsonRpcRequest:
    """构造 initialize 握手请求（Client → Server）.

    握手是 MCP 生命周期的第一条消息：Client 声明自己支持的协议版本与能力，
    并附上客户端身份信息。
    """
    return JsonRpcRequest(
        id=request_id,
        method="initialize",
        params={
            "protocolVersion": protocol_version,
            "capabilities": {},
            "clientInfo": {"name": client_name, "version": client_version},
        },
    )


def build_initialize_response(
    server_name: str = "smart-research-agent",
    server_version: str = "0.1.0",
    *,
    request_id: int | str = 1,
    protocol_version: str = MCP_PROTOCOL_VERSION,
    capabilities: dict[str, Any] | None = None,
) -> JsonRpcResponse:
    """构造 initialize 握手响应（Server → Client）.

    Server 回告协商后的协议版本、自己暴露的能力（resources/tools/prompts）
    以及服务器身份信息；响应的 id 必须与握手请求一致。
    """
    if capabilities is None:
        capabilities = {"resources": {}, "tools": {}, "prompts": {}}
    return JsonRpcResponse(
        id=request_id,
        result={
            "protocolVersion": protocol_version,
            "capabilities": capabilities,
            "serverInfo": {"name": server_name, "version": server_version},
        },
    )
