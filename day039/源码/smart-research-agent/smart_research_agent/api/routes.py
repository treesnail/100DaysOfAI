"""路由层：/health、/chat、/chat/stream、/models、/agent/run（day039）.

依赖注入的读法：``request.app.state`` 持有 create_app 注入的 LLM、模型
注册表与 Agent 工厂，``Depends`` 把它们"投喂"给路由函数——路由不自己
import 全局单例，因此测试只需替换 app.state 上的注入对象即可整体换血。

day039 起新增两条能力（详见 day039 教程）：
  - ``/chat/stream``：SSE 流式对话，逐片段下发增量，实现"首字秒出"；
  - ``model`` 参数 + ``/models``：调用方显式点名模型（按名切换）。

除 ``/chat/stream`` 外端点均声明为同步 ``def``：LLM 调用与 Agent 循环是
阻塞式同步代码，由 FastAPI 放入线程池执行，避免阻塞事件循环。``/chat/stream``
的增量来自 ``BaseLLM.stream`` 返回的生成器，交给 ``StreamingResponse``
消费；生成器内部仍是同步阻塞调用，Starlette 会在线程池里迭代它（详见
day038 教程第三章的线程池机制）。
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from smart_research_agent.agent.fc_agent import FunctionCallingAgent
from smart_research_agent.api.schemas import (
    AgentRunRequest,
    AgentRunResponse,
    ChatRequest,
    ChatResponse,
    ToolCallRecord,
)
from smart_research_agent.llm.base import BaseLLM, Message

router = APIRouter()


def get_llm(request: Request) -> BaseLLM:
    """从 app.state 取注入的默认 LLM 客户端（缺省 target）."""
    return request.app.state.llm


def get_models(request: Request) -> dict[str, BaseLLM]:
    """从 app.state 取模型注册表（``model`` 参数的按名查找表）."""
    return request.app.state.models


def get_agent(request: Request) -> FunctionCallingAgent:
    """每次请求新建一个 Agent（/agent/run 的依赖）.

    Agent 实例携带单次任务的执行轨迹，必须按请求隔离；
    工厂本身由 create_app 注入，测试可整体替换。
    """
    return request.app.state.agent_factory()


def resolve_model(model: str | None, llm: BaseLLM, models: dict[str, BaseLLM]) -> BaseLLM:
    """把 ``model`` 参数解析为具体 LLM 客户端（day039）.

    语义：None（未点名）→ 走默认 llm（单模型或 ModelRouter）；
    显式点名 → 在注册表按名查找，未命中抛 404（错误也是契约，见 day038）。
    注意 404 而非 422：模型名是「合法字符串、但不在候选集里」——这是
    资源不存在，不是请求语义不合法。
    """
    if model is None:
        return llm
    target = models.get(model)
    if target is None:
        available = ", ".join(sorted(models)) or "（无）"
        raise HTTPException(
            status_code=404,
            detail=f"未知模型: {model}，可用模型: {available}",
        )
    return target


def _sse_event(payload: str) -> str:
    """把一个纯文本 payload 编成一条 SSE data 帧（以空行结尾）."""
    return f"data: {payload}\n\n"


@router.get("/health")
def health(request: Request) -> dict:
    """健康检查：供负载均衡与容器探活使用（对应 day022 的容器健康检查）."""
    return {"status": "ok", "version": request.app.version}


@router.get("/models")
def list_models(models: dict[str, BaseLLM] = Depends(get_models)) -> dict:
    """列出可显式切换的模型名（``model`` 参数的合法取值）."""
    return {"models": sorted(models)}


@router.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    llm: BaseLLM = Depends(get_llm),
    models: dict[str, BaseLLM] = Depends(get_models),
) -> ChatResponse:
    """简单对话：单轮问答，不走工具，可按 ``model`` 显式切换."""
    target = resolve_model(req.model, llm, models)
    reply = target.chat(
        [Message(role="user", content=req.message)], temperature=req.temperature
    )
    return ChatResponse(reply=reply)


@router.post("/chat/stream")
def chat_stream(
    req: ChatRequest,
    llm: BaseLLM = Depends(get_llm),
    models: dict[str, BaseLLM] = Depends(get_models),
) -> StreamingResponse:
    """SSE 流式对话：逐片段下发增量文本，直到 ``data: [DONE]`` 收尾.

    每条数据帧形如 ``data: {"delta": "..."}``；``delta`` 是 BaseLLM.stream
    yield 的一个片段，客户端按出现顺序拼接即得完整回复。``[DONE]`` 是
    约定哨兵，与 OpenAI 流式协议对齐，客户端据此判定流结束。
    """
    target = resolve_model(req.model, llm, models)

    def event_stream():
        for chunk in target.stream(
            [Message(role="user", content=req.message)], temperature=req.temperature
        ):
            yield _sse_event(json.dumps({"delta": chunk}, ensure_ascii=False))
        yield _sse_event("[DONE]")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/agent/run", response_model=AgentRunResponse)
def agent_run(
    req: AgentRunRequest, agent: FunctionCallingAgent = Depends(get_agent)
) -> AgentRunResponse:
    """跑完整 Agent 任务：function calling 循环，返回答案与执行轨迹."""
    agent.max_steps = req.max_steps
    # steps = 本次任务消耗的 LLM 调用轮数。LLM 实例可能跨请求共享（calls 会累积），
    # 因此取 run 前后的差值；生产 LLM 不记录调用日志时退化为下限估计
    calls_log = getattr(agent.llm, "calls", None)
    calls_before = len(calls_log) if calls_log is not None else None
    answer = agent.run(req.task)
    if calls_before is not None:
        steps = len(calls_log) - calls_before
    else:
        steps = len(agent.call_history) + 1  # 每轮至多一次工具调用批 + 最终回答一轮
    tool_calls = [
        ToolCallRecord(
            name=call.name, arguments=call.arguments, result=result, success=success
        )
        for call, result, success in agent.call_history
    ]
    return AgentRunResponse(answer=answer, steps=steps, tool_calls=tool_calls)
