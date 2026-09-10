"""API 请求/响应模型：服务的对外契约（day038）.

每个模型同时承担两个角色：
  - 运行时契约：FastAPI 在请求进入路由函数之前按这些模型解析并校验请求体，
    不合法的请求根本不会碰到业务代码，而是返回 422；
  - 文档源：这些模型被反射生成 JSON Schema，构成 /openapi.json 与
    /docs（Swagger UI）中接口文档的全部内容。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """POST /chat 与 /chat/stream 的请求体：一轮简单对话.

    day039 起新增 ``model`` 字段：显式指定模型名。缺省（None）时走应用
    注入的默认 LLM（可以是单模型或 ModelRouter）；显式给出时在注册表
    ``app.state.models`` 中按名查找，未命中返回 404（见 routes.resolve_model）。
    """

    message: str = Field(min_length=1, description="用户消息，不能为空")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="采样温度")
    model: str | None = Field(
        default=None, description="显式指定模型名；缺省使用默认模型/路由"
    )


class ChatResponse(BaseModel):
    """POST /chat 的响应体."""

    reply: str = Field(description="模型回复")


class VisionDescribeResponse(BaseModel):
    """POST /vision/describe 的响应体（day040）.

    注意：请求侧不走这个模型——图片是二进制，必须走 multipart/form-data，
    由 ``routes.vision_describe`` 的 ``UploadFile``/``Form`` 参数接收并校验；
    Pydantic 只负责响应侧序列化，让客户端拿到稳定的 JSON 契约。
    """

    description: str = Field(description="模型基于图片内容给出的回答")


class AgentRunRequest(BaseModel):
    """POST /agent/run 的请求体：跑一次完整 Agent 任务."""

    task: str = Field(min_length=1, description="交给 Agent 的任务描述，不能为空")
    max_steps: int = Field(default=10, ge=1, le=50, description="最大 LLM 调用轮数")


class ToolCallRecord(BaseModel):
    """Agent 执行轨迹中的一次工具调用."""

    name: str = Field(description="被调用的工具名")
    arguments: dict = Field(description="模型给出的工具参数")
    result: str = Field(description="工具返回（或校验/执行错误）文本")
    success: bool = Field(description="本次调用是否成功")


class AgentRunResponse(BaseModel):
    """POST /agent/run 的响应体：最终答案 + 可审计的执行轨迹."""

    answer: str = Field(description="Agent 给出的最终答案")
    steps: int = Field(description="本次任务消耗的 LLM 调用轮数")
    tool_calls: list[ToolCallRecord] = Field(
        default_factory=list, description="执行过程中的全部工具调用记录"
    )


class ErrorResponse(BaseModel):
    """统一错误响应：未捕获异常经全局异常处理器转为此结构（HTTP 500）."""

    detail: str = Field(description="错误详情")
    error_type: str = Field(description="异常类名，便于客户端按类型分支处理")
