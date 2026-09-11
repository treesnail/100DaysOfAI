"""路由层：/health、/chat、/chat/stream、/models、/agent/run、/vision/describe、/embeddings 系列.

依赖注入的读法：``request.app.state`` 持有 create_app 注入的 LLM、模型
注册表与 Agent 工厂，``Depends`` 把它们"投喂"给路由函数——路由不自己
import 全局单例，因此测试只需替换 app.state 上的注入对象即可整体换血。

day039 起新增两条能力（详见 day039 教程）：
  - ``/chat/stream``：SSE 流式对话，逐片段下发增量，实现"首字秒出"；
  - ``model`` 参数 + ``/models``：调用方显式点名模型（按名切换）。

day040 起新增多模态能力：
  - ``/vision/describe``：multipart 上传图片 + 问题，由支持视觉的模型回答。
    图片走 multipart 而非 JSON（二进制不适合塞进 JSON 字符串），校验与
    编码逻辑收敛在 ``multimodal.image``，路由层只负责协议编解码。

day041 起新增 embedding 能力：
  - ``/embeddings``：批量文本编码，返回向量与提供方信息；
  - ``/embeddings/similarity``：两段文本的余弦相似度。
    两者共用 ``app.state.embedding`` 注入的提供方（create_app 可替换），
    路由层不关心向量是哈希的还是神经的——这正是提供方抽象的回报。

除 ``/chat/stream`` 外端点均声明为同步 ``def``：LLM 调用与 Agent 循环是
阻塞式同步代码，由 FastAPI 放入线程池执行，避免阻塞事件循环。``/chat/stream``
的增量来自 ``BaseLLM.stream`` 返回的生成器，交给 ``StreamingResponse``
消费；生成器内部仍是同步阻塞调用，Starlette 会在线程池里迭代它（详见
day038 教程第三章的线程池机制）。
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from smart_research_agent.agent.fc_agent import FunctionCallingAgent
from smart_research_agent.api.schemas import (
    AgentRunRequest,
    AgentRunResponse,
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    SimilarityRequest,
    SimilarityResponse,
    ToolCallRecord,
    VisionDescribeResponse,
)
from smart_research_agent.llm.base import BaseLLM, Message
from smart_research_agent.llm.embedding import EmbeddingProvider
from smart_research_agent.memory.vector_store import cosine_similarity
from smart_research_agent.multimodal.image import (
    EmptyImageError,
    ImageTooLargeError,
    UnsupportedImageTypeError,
    build_data_url,
    validate_image,
)

router = APIRouter()


def get_llm(request: Request) -> BaseLLM:
    """从 app.state 取注入的默认 LLM 客户端（缺省 target）."""
    return request.app.state.llm


def get_models(request: Request) -> dict[str, BaseLLM]:
    """从 app.state 取模型注册表（``model`` 参数的按名查找表）."""
    return request.app.state.models


def get_embedding(request: Request) -> EmbeddingProvider:
    """从 app.state 取注入的 embedding 提供方（day041，/embeddings 系列端点的依赖）."""
    return request.app.state.embedding


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


@router.post("/embeddings", response_model=EmbeddingResponse)
def embeddings(
    req: EmbeddingRequest, embedding: EmbeddingProvider = Depends(get_embedding)
) -> EmbeddingResponse:
    """批量文本编码（day041）：一次请求编码整批文本.

    响应里的 ``vectors`` 与请求 ``texts`` 顺序一一对应；``provider``/
    ``dimension`` 让客户端无需猜测"向量是谁编的、多长"——向量只有
    配合维度与提供方信息才是自描述的。
    """
    vectors = embedding.embed_batch(req.texts)
    return EmbeddingResponse(
        provider=type(embedding).__name__, dimension=embedding.dimension, vectors=vectors
    )


@router.post("/embeddings/similarity", response_model=SimilarityResponse)
def embeddings_similarity(
    req: SimilarityRequest, embedding: EmbeddingProvider = Depends(get_embedding)
) -> SimilarityResponse:
    """两段文本的余弦相似度（day041）：embedding 的最小可用演示.

    两段文本走**一次** ``embed_batch``（而非两次 embed）——对云端提供方
    就是一次 HTTP。相似度用向量库的 ``cosine_similarity`` 计算，与检索
    链路同一口径，保证"接口演示"与"生产检索"不会出现两套度量。
    """
    vec_a, vec_b = embedding.embed_batch([req.text_a, req.text_b])
    return SimilarityResponse(
        similarity=cosine_similarity(vec_a, vec_b), provider=type(embedding).__name__
    )


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


@router.post("/vision/describe", response_model=VisionDescribeResponse)
def vision_describe(
    file: UploadFile = File(...),
    question: str = Form("描述这张图片的内容"),
    model: str | None = Form(None),
    llm: BaseLLM = Depends(get_llm),
    models: dict[str, BaseLLM] = Depends(get_models),
) -> VisionDescribeResponse:
    """多模态问答：上传一张图片，让支持视觉的模型回答关于它的问题（day040）.

    请求侧是 multipart/form-data：``file`` 是图片文件，``question`` 是
    可选问题（缺省"描述这张图片的内容"），``model`` 沿用 day039 的按名
    切换语义。图片经 ``multimodal.image`` 校验（空/超限/不支持格式 →
    422）并编码为 data URL 后，交给 ``chat_vision`` 走视觉协议。

    目标模型没有视觉能力时返回 400：这是调用方点名了不合适的模型，
    属于「请求语义与能力不匹配」；若默认 LLM 是 ModelRouter，视觉链
    会自行跳过文本模型、找到视觉候选，不会走到这里。
    """
    data = file.file.read()
    try:
        mime = validate_image(data)
    except (EmptyImageError, ImageTooLargeError, UnsupportedImageTypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    data_url = build_data_url(data, mime)
    target = resolve_model(model, llm, models)
    if not getattr(target, "supports_vision", False):
        raise HTTPException(
            status_code=400,
            detail=f"模型 {model or '默认'} 不支持图像输入（缺少视觉能力）",
        )
    reply = target.chat_vision(
        [Message(role="user", content=question)], image_data_url=data_url
    )
    return VisionDescribeResponse(description=reply)


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
