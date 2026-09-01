"""MCP 协议建模（合并基准版）：JSON-RPC 2.0 消息、能力描述与 initialize 握手.

本模块是 day016~day023 六个 protocol.py 快照的并集：
- 以 day016 的完整 API 为基础（常量、build_initialize_*、PromptArgument、严格校验）；
- 错误对象统一命名为 ``ErrorObject``（day016 命名），``JsonRpcErrorBody``（day017/018/023）
  与 ``JsonRpcErrorDetail``（day019/022）保留为兼容别名；
- ``PromptArgumentDescriptor``（day023）保留为 ``PromptArgument`` 的兼容别名；
- ``JsonRpcRequest.id`` / ``JsonRpcResponse.id`` 允许缺省为 None（day022 的通知消息语义）；
- ``PromptDescriptor.arguments`` 为 ``list[PromptArgument] | list[str]`` 联合类型，
  同时兼容 day016/019/022/023 的 dict 列表写法与 day017/018 的字符串列表写法。

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
    """JSON-RPC 2.0 请求消息：Client → Server（或反向）调用一个方法.

    ``id`` 由客户端生成，服务器必须在响应中原样带回；允许为 None，
    以兼容 day022 中"无 id 的通知型消息"写法。
    """

    jsonrpc: Literal["2.0"] = JSONRPC_VERSION
    id: int | str | None = None
    method: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class JsonRpcResponse(BaseModel):
    """JSON-RPC 2.0 成功响应消息：id 必须与对应请求一致."""

    jsonrpc: Literal["2.0"] = JSONRPC_VERSION
    id: int | str | None = None
    result: Any = None


class ErrorObject(BaseModel):
    """JSON-RPC 2.0 错误对象：code 为整数错误码，message 为人类可读描述.

    合并基准名（day016）；day018/022/023 版本没有 ``data`` 字段，
    这里保留 ``data: Any = None`` 作为可选扩展（day016/017/019 均有），
    只传 code/message 的构造方式全部兼容。
    """

    code: int
    message: str = Field(min_length=1)
    data: Any = None


#: 兼容别名：day017/day018/day023 的错误对象命名
JsonRpcErrorBody = ErrorObject
#: 兼容别名：day019/day022 的错误对象命名
JsonRpcErrorDetail = ErrorObject


class JsonRpcError(BaseModel):
    """JSON-RPC 2.0 错误响应消息：解析失败等场景下 id 可能为 None."""

    jsonrpc: Literal["2.0"] = JSONRPC_VERSION
    id: int | str | None = None
    error: ErrorObject


class ResourceDescriptor(BaseModel):
    """MCP Resource 能力描述：一份可用 URI 寻址的只读数据."""

    uri: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = ""
    mime_type: str = Field(default="text/plain", min_length=1)


class ToolDescriptor(BaseModel):
    """MCP Tool 能力描述：一个可被模型调用、有副作用的函数.

    ``input_schema`` 是 JSON Schema 对象，描述调用参数的形状，
    客户端（尤其是 LLM）依据它构造合法调用。
    """

    name: str = Field(min_length=1)
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)


class PromptArgument(BaseModel):
    """MCP Prompt 模板的单个参数."""

    name: str = Field(min_length=1)
    description: str = ""
    required: bool = False


#: 兼容别名：day023 的 Prompt 参数命名
PromptArgumentDescriptor = PromptArgument


class PromptDescriptor(BaseModel):
    """MCP Prompt 能力描述：一个可参数化填充的提示词模板.

    ``arguments`` 取各天版本的并集：day016/019/022/023 传 dict 列表
    （解析为 PromptArgument），day017/018 传字符串列表。
    """

    name: str = Field(min_length=1)
    description: str = ""
    arguments: list[PromptArgument] | list[str] = Field(default_factory=list)


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
