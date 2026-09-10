"""路由层：/health、/chat、/agent/run 三个端点（day038）.

依赖注入的读法：``request.app.state`` 持有 create_app 注入的 LLM 与
Agent 工厂，``Depends`` 把它们"投喂"给路由函数——路由不自己 import
全局单例，因此测试只需替换 app.state 上的注入对象即可整体换血。

三个端点均声明为同步 ``def``：LLM 调用与 Agent 循环是阻塞式同步代码，
由 FastAPI 放入线程池执行，避免阻塞事件循环（详见 day038 教程第三章）。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

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
    """从 app.state 取注入的 LLM 客户端（/chat 的依赖）."""
    return request.app.state.llm


def get_agent(request: Request) -> FunctionCallingAgent:
    """每次请求新建一个 Agent（/agent/run 的依赖）.

    Agent 实例携带单次任务的执行轨迹，必须按请求隔离；
    工厂本身由 create_app 注入，测试可整体替换。
    """
    return request.app.state.agent_factory()


@router.get("/health")
def health(request: Request) -> dict:
    """健康检查：供负载均衡与容器探活使用（对应 day022 的容器健康检查）."""
    return {"status": "ok", "version": request.app.version}


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, llm: BaseLLM = Depends(get_llm)) -> ChatResponse:
    """简单对话：单轮问答，不走工具."""
    reply = llm.chat(
        [Message(role="user", content=req.message)], temperature=req.temperature
    )
    return ChatResponse(reply=reply)


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
