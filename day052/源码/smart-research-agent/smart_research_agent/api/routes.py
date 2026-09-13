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

day044 起新增安全合规能力（详见 day044 教程）：
  - 输出侧审核：``/chat`` 与 ``/agent/run`` 的回复在返回前过
    ``ContentModerator``，PII 脱敏、敏感词拦截，响应携带 ``moderation``
    报告；审核器由 ``app.state.moderator`` 注入，测试可替换为自定义规则；
  - 访问日志：由 app.py 挂载的 ``AccessLogMiddleware`` 在路由外层统一
    记录，路由层不感知——中间件与路由解耦，是 AOP 式的横切能力。

day046 起新增一体化流水线能力（详见 day046 教程）：
  - ``/pipeline/run``：一次请求走完「输入侧护栏 → 语义缓存 → 路由与生成 →
    成本归因 → 输出审核 → 缓存回写」六个阶段，响应携带逐阶段的耗时账
    （``stages``）与单次请求的费用归因（``cost_usd``）；装配好的流水线由
    ``app.state.pipeline`` 注入，测试可整体替换；
  - ``/pipeline/baseline``：返回已存档的性能基线（供 CI 与看板比对），
    文件缺失时返回 404——"还没有基线"必须被显式暴露，而不是返回一个
    全零的假基线（那会让所有回归判定静默失效）。

day048 起新增微调数据工程能力（详见 day048 教程）：
  - ``/finetune/methods``：五种微调方法的画像（供前端/文档直接渲染）；
  - ``/finetune/dataset/validate``：校验一批原始样本能否用于训练，
    返回 valid/invalid/逐条 issue 与合格样本的画像；未知 format 返回 400；
  - ``/finetune/dataset/stats``：基于默认采集器跑一遍
    ``data/finetune`` + ``data/eval`` 的数据集画像。
    三个端点全部只读、只消费 ``finetune`` 包，不引入任何新的外部依赖。

day050 起新增 SFT 监督微调的三个只读端点（详见 day050 教程）：
  - ``/finetune/sft/defaults``：与 ``TrainingArguments`` 同名的默认超参、
    SFT 专属项（``max_length`` / ``truncation`` / ``ignore_index``）与依赖清单；
  - ``/finetune/sft/plan``：算一次训练的派生量（步数 / warmup / 有效批）
    并给出风险告警，同时返回**渲染后的长度分布**与 ``max_length`` 建议值
    ——``max_length`` 必须由数据的长度分位数决定，这个端点把该判断暴露出来；
  - ``/finetune/sft/preview``：把一条原始样本渲染 + 编码，返回前缀/监督区间
    的字符与 token 切分，并用 ``prompt_fully_masked`` 直接回答"label mask
    真的生效了吗"。
    三者同样只读：**训练本身**由 ``scripts/sft_demo.py`` 与生成的
    ``transformers`` 脚本承担，HTTP 端点不适合承载一个需要 GPU 的长任务。

day051 起新增 LoRA / QLoRA 参数高效微调的三个只读端点（详见 day051 教程）：
  - ``/finetune/lora/defaults``：与 ``peft.LoraConfig`` /
    ``BitsAndBytesConfig`` 同名的默认配置、量化的每参数存储表（块大小对照）、
    目标模块预设，以及**参考模型上实测的 LoRA 计划**；
  - ``/finetune/lora/plan``：算一次 LoRA 训多少参数（含目标预设与秩的对照表）
    并给出风险告警——"r 写大了反而比全参更贵""预设只命中了一半"这类问题
    必须在提交任务之前暴露；
  - ``/finetune/lora/memory``：算全参 / LoRA / QLoRA 三种策略的显存预算
    （权重 + 梯度 + 优化器状态六项），并**显式标注不含激活显存**。
    与 day050 一样，训练本身不进 HTTP 端点。


除 ``/chat/stream`` 外端点均声明为同步 ``def``：LLM 调用与 Agent 循环是
阻塞式同步代码，由 FastAPI 放入线程池执行，避免阻塞事件循环。``/chat/stream``
的增量来自 ``BaseLLM.stream`` 返回的生成器，交给 ``StreamingResponse``
消费；生成器内部仍是同步阻塞调用，Starlette 会在线程池里迭代它（详见
day038 教程第三章的线程池机制）。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from smart_research_agent.agent.fc_agent import FunctionCallingAgent
from smart_research_agent.api.schemas import (
    AgentRunRequest,
    AgentRunResponse,
    ChatRequest,
    ChatResponse,
    DatasetIssue,
    DatasetStatsResponse,
    DatasetValidateRequest,
    DatasetValidateResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    FinetuneMethodInfo,
    FinetuneMethodsResponse,
    InjectionInfo,
    LoRADefaultsResponse,
    LoRADeployRequest,
    LoRADeployResponse,
    LoRADistributedRequest,
    LoRADistributedResponse,
    LoRAMemoryRequest,
    LoRAMemoryResponse,
    LoRAPlanRequest,
    LoRAPlanResponse,
    ModerationInfo,
    PerfBaselineResponse,
    PipelineRunRequest,
    PipelineRunResponse,
    SFTDefaultsResponse,
    SFTPlanRequest,
    SFTPlanResponse,
    SFTPreviewRequest,
    SFTPreviewResponse,
    SimilarityRequest,
    SimilarityResponse,
    StageInfo,
    ToolCallRecord,
    VisionDescribeResponse,
)
from smart_research_agent.config import settings
from smart_research_agent.evaluation.perf_baseline import PerfBaseline
from smart_research_agent.finetune.dataset import build_dataset, compute_stats, split_dataset
from smart_research_agent.finetune.overview import METHODS, MethodProfile
from smart_research_agent.finetune.schema import (
    SUPPORTED_FORMATS,
    DatasetFormatError,
    parse_example,
)
from smart_research_agent.integration.pipeline import IntegratedPipeline, PipelineResult
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
from smart_research_agent.peft.accelerate import (
    accelerate_config,
    deepspeed_zero_config,
    device_fit_summary,
    launch_command,
    plan_distributed,
    render_config_yaml,
)
from smart_research_agent.peft.config import (
    BASELINE_ONLY_QUANT_TYPES,
    LORA_TARGET_PRESETS,
    REFERENCE_MODULE_NAME,
    LoRAConfig,
    PEFTConfigError,
    QLoRAConfig,
    quantization_table,
)
from smart_research_agent.peft.deploy import (
    adapter_registry,
    build_manifest,
    deployment_report,
    render_inference_script,
    render_merge_script,
    verify_merge,
)
from smart_research_agent.peft.hf_script import (
    PEFT_DEPENDENCIES,
    QLORA_DEPENDENCIES,
    peft_dependency_commands,
)
from smart_research_agent.peft.memory import (
    compare_strategies,
    device_fit_table,
    quantization_memory_note,
    savings_table,
)
from smart_research_agent.peft.models import (
    LoRAReferenceModel,
    default_reference_lora_config,
    reference_lora_accounting,
)
from smart_research_agent.peft.targets import (
    MODEL_SPECS,
    DecoderSpec,
    plan_lora,
    rank_comparison,
    target_preset_table,
)
from smart_research_agent.peft.trainer import ADAPTER_FILES, save_adapter
from smart_research_agent.security.content_moderator import ContentModerator
from smart_research_agent.sft.args import (
    LR_SOFT_RANGE_REFERENCE,
    SFTTrainingArgs,
    plan_training,
)
from smart_research_agent.sft.encoding import (
    IGNORE_INDEX,
    CharTokenizer,
    encode_supervised,
    iter_batches,
    length_summary,
    suggest_max_length,
)
from smart_research_agent.sft.hf_script import (
    DEFAULT_BASE_MODEL,
    HF_DEPENDENCIES,
    TRL_DEPENDENCIES,
    dependency_commands,
)
from smart_research_agent.sft.reference_model import (
    REFERENCE_LEARNING_RATE,
    ReferenceSFTModel,
)
from smart_research_agent.sft.template import (
    render_supervised,
    render_supervised_list,
)

router = APIRouter()

#: /finetune/dataset/validate 最多返回多少条 issue（day048）.
#: 一批脏数据可能有上万条问题样本，全量返回只会撑爆响应体；首条错误通常
#: 已经暴露了问题模式，够用且响应体大小可控。
MAX_DATASET_ISSUES = 20


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


def get_moderator(request: Request) -> ContentModerator:
    """从 app.state 取注入的内容审核器（day044，输出侧审核的依赖）."""
    return request.app.state.moderator


def get_pipeline(request: Request) -> IntegratedPipeline:
    """从 app.state 取注入的一体化流水线（day046，/pipeline/run 的依赖）.

    流水线是有状态的（缓存、成本追踪器、计数器），因此它与 LLM 一样在
    ``create_app`` 时构造一次、跨请求共享；缓存的效果正来自"跨请求复用
    同一份历史"。测试注入自建流水线即可整体替换装配方案。
    """
    return request.app.state.pipeline


def to_moderation_info(result) -> ModerationInfo:
    """把 ContentModerator 的审核结果投影为 HTTP 契约的 ModerationInfo."""
    return ModerationInfo(
        is_safe=result.is_safe,
        flagged_words=result.flagged_words,
        pii_types=result.pii_types,
    )


def to_pipeline_response(result: PipelineResult) -> PipelineRunResponse:
    """把 PipelineResult 投影为 HTTP 契约（阶段账与护栏报告一并透出）."""
    return PipelineRunResponse(
        reply=result.reply,
        model=result.model,
        cached=result.cached,
        blocked=result.blocked,
        cost_usd=result.cost_usd,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_ms=result.total_ms,
        stages=[
            StageInfo(name=s.name, duration_ms=s.duration_ms, detail=s.detail)
            for s in result.stages
        ],
        moderation=to_moderation_info(result.moderation),
        injection=(
            None
            if result.injection is None
            else InjectionInfo(
                is_injection=result.injection.is_injection,
                matched_patterns=result.injection.matched_patterns,
            )
        ),
    )


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
    moderator: ContentModerator = Depends(get_moderator),
) -> ChatResponse:
    """简单对话：单轮问答，不走工具，可按 ``model`` 显式切换.

    day044 起：模型回复在返回前过输出侧审核——PII 脱敏、敏感词拦截，
    ``reply`` 是脱敏后的文本，``moderation`` 报告审核结论。审核发生在
    响应构造前（而非中间件里），因为审核结果要随响应体返回给客户端，
    中间件只能"拦"不能"改语义"。
    """
    target = resolve_model(req.model, llm, models)
    reply = target.chat(
        [Message(role="user", content=req.message)], temperature=req.temperature
    )
    result = moderator.moderate(reply)
    return ChatResponse(reply=result.sanitized_text, moderation=to_moderation_info(result))


@router.post("/pipeline/run", response_model=PipelineRunResponse)
def pipeline_run(
    req: PipelineRunRequest,
    pipeline: IntegratedPipeline = Depends(get_pipeline),
) -> PipelineRunResponse:
    """一体化流水线（day046）：一次请求走完六个阶段，返回答复与逐阶段的账.

    与 ``/chat`` 的分工：``/chat`` 是"只要答案"的轻接口（审核后返回文本），
    本接口是"要答案 + 要账本"的重接口——阶段耗时、单次费用归因、是否命中
    缓存、输入侧护栏结论全部随响应返回。两者共用同一套底层能力（审核器、
    LLM/路由），差别只在**是否把过程暴露给调用方**。

    注意流水线内部的异常（如 LLM 连接失败）不在这里捕获：交由全局异常
    处理器统一转成 ``ErrorResponse``（HTTP 500），保持全服务的错误契约一致。
    """
    return to_pipeline_response(
        pipeline.run(
            req.task, temperature=req.temperature, system_prompt=req.system_prompt
        )
    )


@router.get("/pipeline/baseline", response_model=PerfBaselineResponse)
def pipeline_baseline(request: Request) -> PerfBaselineResponse:
    """返回已存档的性能基线（day046）：供 CI 与看板做回归比对.

    基线路径来自 ``app.state.perf_baseline_path``（create_app 按 settings
    注入），测试可指向临时文件。文件缺失返回 404 而非零值基线——"还没有
    基线"是一个必须被看见的状态：假基线会让所有回归判定静默通过。
    """
    try:
        baseline = PerfBaseline.load(request.app.state.perf_baseline_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return PerfBaselineResponse(**baseline.to_dict())


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
    req: AgentRunRequest,
    agent: FunctionCallingAgent = Depends(get_agent),
    moderator: ContentModerator = Depends(get_moderator),
) -> AgentRunResponse:
    """跑完整 Agent 任务：function calling 循环，返回答案与执行轨迹.

    day044 起：最终答案同样过输出侧审核（PII 脱敏 + 敏感词拦截），
    ``answer`` 是脱敏后的文本，``moderation`` 报告审核结论。工具调用
    轨迹（tool_calls）保留原始记录不做脱敏——它是审计证据，脱敏会破坏
    可追溯性；需要对外展示时再由消费方决定如何脱敏。
    """
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
    result = moderator.moderate(answer)
    return AgentRunResponse(
        answer=result.sanitized_text,
        steps=steps,
        tool_calls=tool_calls,
        moderation=to_moderation_info(result),
    )


def to_method_info(profile: MethodProfile) -> FinetuneMethodInfo:
    """把 ``MethodProfile`` 投影为 HTTP 契约（tuple → list，字段一一对应）."""
    return FinetuneMethodInfo(
        key=profile.key,
        full_name=profile.full_name,
        family=profile.family,
        data_need=profile.data_need,
        trainable_ratio_hint=profile.trainable_ratio_hint,
        stability=profile.stability,
        best_for=profile.best_for,
        artifacts=list(profile.artifacts),
        notes=profile.notes,
    )


@router.get("/finetune/methods", response_model=FinetuneMethodsResponse)
def finetune_methods() -> FinetuneMethodsResponse:
    """列出五种微调方法的结构化画像（day048）：SFT/LoRA/QLoRA/DPO/RLHF-PPO.

    响应直接投影 ``finetune.overview.METHODS`` 这份知识常量，顺序即字典
    插入顺序（稳定可断言）——"有哪些方法、各自适合什么"是产品文档与
    选型工具的公共数据源，不该由前端各写一份硬编码。
    """
    return FinetuneMethodsResponse(
        methods=[to_method_info(profile) for profile in METHODS.values()]
    )


@router.post("/finetune/dataset/validate", response_model=DatasetValidateResponse)
def finetune_dataset_validate(req: DatasetValidateRequest) -> DatasetValidateResponse:
    """校验一批原始训练样本能否用于微调（day048）.

    逐条走 ``parse_example``：能解析的进入合格集合（并用 ``compute_stats``
    给出画像），不能解析的记录下标与原因。错误码分工沿用既有契约：

    - 未知 ``format`` → **400**：格式名是合法字符串但不在候选集里，
      属于"请求语义与能力不匹配"（与 /vision/describe 的选型错误同构）；
    - ``examples`` 为空 → **422**：由 Pydantic 的 ``min_length=1`` 在进路由
      之前拦下，不需要业务代码再判一次。

    校验**不抛第一条错误就返回**：批量数据的价值在于"坏了几条、坏在哪"，
    只报第一条会让用户来回试错。``issues`` 截断到前 ``MAX_DATASET_ISSUES`` 条。
    """
    if req.format not in SUPPORTED_FORMATS:
        available = ", ".join(SUPPORTED_FORMATS)
        raise HTTPException(
            status_code=400,
            detail=f"未知数据集格式: {req.format}，可用格式: {available}",
        )

    valid_examples = []
    issues: list[DatasetIssue] = []
    for index, raw in enumerate(req.examples):
        try:
            valid_examples.append(parse_example(raw, req.format))
        except DatasetFormatError as exc:
            if len(issues) < MAX_DATASET_ISSUES:
                issues.append(DatasetIssue(index=index, reason=str(exc)))

    return DatasetValidateResponse(
        total=len(req.examples),
        valid=len(valid_examples),
        invalid=len(req.examples) - len(valid_examples),
        issues=issues,
        stats=compute_stats(valid_examples).to_dict(),
    )


@router.get("/finetune/dataset/stats", response_model=DatasetStatsResponse)
def finetune_dataset_stats() -> DatasetStatsResponse:
    """当前数据集画像（day048）：跑一遍默认采集 + 清洗后的统计数字.

    数据目录取自 ``settings.finetune_data_dir``（``data/finetune``）与其
    同级 ``data/eval``：种子样本、评估轨迹、红队安全样本三个源一起进流水线。
    ``source_distribution`` 让"这批数据主要来自哪个源"一目了然——它是判断
    数据是否失衡的第一手依据。
    """
    bundle = build_dataset()
    return DatasetStatsResponse(stats=compute_stats(bundle.examples).to_dict())


def sft_args_from_settings(overrides: dict | None = None) -> SFTTrainingArgs:
    """按配置装配一份 SFT 超参，并应用 ``overrides``（day050）.

    为什么要经过配置而不是直接用 ``SFTTrainingArgs()`` 的字段缺省值：
    两者的**标定对象不同**——dataclass 的缺省值是"7B 全参 + AdamW"的
    现实取值（``learning_rate = 2e-4``），而 ``settings.sft_*`` 是"参考
    模型 + 本课程数据集"的取值（``learning_rate = 8.0``）。让端点走配置，
    意味着"部署层面可调"这件事对 SFT 也成立（与 day048 的七个
    ``finetune_*`` 配置项同一种思路）。

    ``overrides`` 走 ``from_dict`` 合并，因此**未知键会被忽略**（向前兼容），
    非法取值则会在随后的 ``validate()`` 里被拒绝。
    """
    base = SFTTrainingArgs(
        output_dir=settings.sft_output_dir,
        max_length=settings.sft_max_length,
        num_train_epochs=settings.sft_num_train_epochs,
        learning_rate=settings.sft_learning_rate,
        per_device_train_batch_size=settings.sft_per_device_train_batch_size,
        gradient_accumulation_steps=settings.sft_gradient_accumulation_steps,
        warmup_ratio=settings.sft_warmup_ratio,
        lr_scheduler_type=settings.sft_lr_scheduler_type,
        seed=settings.sft_seed,
    )
    if not overrides:
        return base
    merged = {**base.to_dict(), **overrides}
    return SFTTrainingArgs.from_dict(merged)


#: ``/finetune/sft/plan`` 的 ``max_length`` 建议值所用分位（day050）.
#: 取 p95 而不是最大值：长尾样本往往是"答案啰嗦"的脏数据，为它们抬高整批
#: 数据的 ``max_length`` 代价过大；被切掉的样本会在编码统计里如实记录。
SFT_SUGGEST_QUANTILE = 0.95


@router.get("/finetune/sft/defaults", response_model=SFTDefaultsResponse)
def finetune_sft_defaults() -> SFTDefaultsResponse:
    """SFT 默认超参、专属项与依赖清单（day050）.

    响应里的 ``training_arguments`` 与 ``transformers.TrainingArguments``
    **逐字同名**，因此前端可以拿它直接拼出训练任务，不需要再维护一张
    "我们的字段 → HF 字段"的映射表——而那张表就是漂移的来源。
    """
    args = sft_args_from_settings()
    return SFTDefaultsResponse(
        training_arguments=args.to_hf_dict(),
        sft_specific={
            "max_length": args.max_length,
            "truncation": args.truncation,
            "ignore_index": IGNORE_INDEX,
        },
        base_model=DEFAULT_BASE_MODEL,
        hf_dependencies=list(HF_DEPENDENCIES),
        trl_dependencies=list(TRL_DEPENDENCIES),
        install=dependency_commands(use_trl=True),
    )


@router.post("/finetune/sft/plan", response_model=SFTPlanResponse)
def finetune_sft_plan(req: SFTPlanRequest) -> SFTPlanResponse:
    """算一次 SFT 的派生量并给出风险告警（day050）.

    这个端点回答的是"**先算再跑**"：在提交一个可能跑几小时的任务之前，
    先知道它会跑多少步、warmup 会不会退化成 0、``max_length`` 定得对不对。
    day049 已经证明这些数字**完全可以在没有训练框架的情况下算准**。

    错误码延续既有契约：超参非法（含 ``overrides`` 类型不对）→ **400**；
    ``train_size`` / ``eval_size`` 的取值由 Pydantic 在进路由前拦下 → **422**。
    """
    try:
        args = sft_args_from_settings(req.overrides)
        args.validate()
    except ValueError as exc:
        # SFTConfigError 继承 ValueError，此处一次覆盖"字段非法"与"类型不对"
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    bundle = build_dataset()
    train, eval_set = split_dataset(
        bundle.examples,
        eval_ratio=settings.finetune_eval_ratio,
        seed=settings.finetune_split_seed,
    )
    train_size = req.train_size if req.train_size is not None else len(train)
    eval_size = req.eval_size if req.eval_size is not None else len(eval_set)

    rendered = render_supervised_list(bundle.examples)
    tokenizer = CharTokenizer.from_texts([item.text for item in rendered])
    summary = length_summary(rendered, tokenizer)
    suggested = suggest_max_length(rendered, tokenizer, quantile=SFT_SUGGEST_QUANTILE)

    try:
        plan = plan_training(
            args,
            train_size=train_size,
            eval_size=eval_size,
            dataset_avg_tokens=summary.mean,
            dataset_max_tokens=summary.maximum,
            lr_range=LR_SOFT_RANGE_REFERENCE,
        )
    except ValueError as exc:  # pragma: no cover - 上面的 validate 已覆盖主要分支
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return SFTPlanResponse(
        training_arguments=args.to_hf_dict(),
        plan=plan.to_dict(),
        warnings=list(plan.warnings),
        length=summary.to_dict(),
        suggested_max_length=suggested,
        suggested_quantile=SFT_SUGGEST_QUANTILE,
    )


@router.post("/finetune/sft/preview", response_model=SFTPreviewResponse)
def finetune_sft_preview(req: SFTPreviewRequest) -> SFTPreviewResponse:
    """预览一条样本的监督切分与 label mask（day050）.

    这是把"prompt 段置 -100"这条机制**变成可观测事实**的端点：返回体里
    同时给出字符级切分（``prompt_chars`` / ``supervised_chars``）、token 级
    切分（``prompt_tokens`` / ``supervised_tokens``）与一个直判字段
    ``prompt_fully_masked``。**只要它不是 true，这条样本就不该进训练集。**

    模板与长度问题（未知模板、答案放不下 ``max_length``）都返回 **400**：
    它们都是"请求语义与能力不匹配"，与 day048 未知 ``format`` 同构。
    """
    try:
        example = parse_example(req.example)
        rendered = render_supervised(example, template=req.template)
        # 词表从这一条样本构建即可：预览只需自洽（同一段文本进、同一段出），
        # 不需要与训练时的全局词表一致。
        tokenizer = CharTokenizer.from_texts([rendered.text])
        encoded = encode_supervised(rendered, tokenizer, max_length=req.max_length)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    prompt_labels = encoded.labels[: encoded.prompt_tokens]
    return SFTPreviewResponse(
        template=rendered.template,
        total_chars=rendered.total_chars,
        prompt_chars=rendered.prompt_chars,
        supervised_chars=rendered.supervised_chars,
        total_tokens=encoded.total_tokens,
        prompt_tokens=encoded.prompt_tokens,
        supervised_tokens=encoded.supervised_tokens,
        masked_tokens=encoded.masked_tokens,
        supervised_ratio=encoded.supervised_ratio,
        truncated=encoded.truncated,
        prompt_fully_masked=bool(prompt_labels)
        and all(value == IGNORE_INDEX for value in prompt_labels),
        text_preview=rendered.text[:200],
        supervised_preview=rendered.supervised_text[:120],
    )


#: ``/finetune/lora/*`` 用到的模型规格名（day051）：规格里的每个数字都取自
#: 公开 config.json，并附带参数量自检（见 ``peft/targets.py``）。
LORA_DEFAULT_MODEL_SPEC = "llama-2-7b"


def lora_config_from_settings(overrides: dict | None = None) -> LoRAConfig:
    """按 ``lora_*`` 配置项构造 ``LoRAConfig``，并应用 ``overrides``.

    与 day050 ``sft_args_from_settings`` 同一种做法：配置是**默认值的来源**，
    请求体里的 ``overrides`` 是**本次调用的差异**，两者合并后统一走
    ``from_dict``（未知键忽略，向前兼容）。
    """
    base = LoRAConfig(
        r=settings.lora_r,
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout,
        target_modules=settings.lora_target_preset,
        bias=settings.lora_bias,
        use_rslora=settings.lora_use_rslora,
    )
    if not overrides:
        return base
    return LoRAConfig.from_dict({**base.to_dict(), **overrides})


def qlora_config_from_settings(overrides: dict | None = None) -> QLoRAConfig:
    """按 ``qlora_*`` 配置项构造 ``QLoRAConfig``，并应用 ``overrides``."""
    base = QLoRAConfig(
        bnb_4bit_quant_type=settings.qlora_quant_type,
        bnb_4bit_compute_dtype=settings.qlora_compute_dtype,
        bnb_4bit_use_double_quant=settings.qlora_use_double_quant,
        block_size=settings.qlora_block_size,
    )
    if not overrides:
        return base
    return QLoRAConfig.from_dict({**base.to_dict(), **overrides})


def resolve_model_spec(name: str) -> DecoderSpec:
    """按名取模型规格；未知名字抛 **400**（与 day048 未知 ``format`` 同构）."""
    spec = MODEL_SPECS.get(name)
    if spec is None:
        raise HTTPException(
            status_code=400,
            detail=f"未知的模型规格 {name!r}，可选：{', '.join(sorted(MODEL_SPECS))}",
        )
    return spec


@router.get("/finetune/lora/defaults", response_model=LoRADefaultsResponse)
def finetune_lora_defaults() -> LoRADefaultsResponse:
    """LoRA / QLoRA 的默认配置、量化派生量与依赖清单（day051）.

    ``reference_model`` 里的数字来自**真实数据集**（day048 落盘的 30 + 7 条
    样本渲染后的字符级词表），因此"可训练参数占 2.7875%"这句话是可复核的，
    而不是一个引用的经验值。

    参考模型的账用 ``reference_lora_accounting`` 单独算，**不能**走
    ``plan_lora``：后者面向解码器规格（``q_proj`` 等七个投影），而参考模型
    只有一个叫 ``weight`` 的 bigram 矩阵——把两者硬接上会命中"目标模块未
    匹配"的校验并让端点返回 500（本课实现时真的踩到过：单元测试全绿，
    因为测试只覆盖了 helper，没有覆盖端点的组合方式）。
    """
    lora = lora_config_from_settings()
    qlora = qlora_config_from_settings()
    bundle = build_dataset()
    rendered = render_supervised_list(bundle.examples)
    tokenizer = CharTokenizer.from_texts([item.text for item in rendered])
    reference = reference_lora_accounting(
        tokenizer.vocab_size,
        default_reference_lora_config(
            r=lora.r, lora_alpha=lora.lora_alpha, lora_dropout=0.0
        ),
    )
    return LoRADefaultsResponse(
        lora=lora.to_peft_dict(),
        scaling=lora.scaling,
        scaling_formula=lora.scaling_formula,
        quantization_config=qlora.to_bnb_dict(),
        quantization=qlora.to_dict(),
        quantization_block_table=quantization_table(),
        target_presets={name: list(targets) for name, targets in LORA_TARGET_PRESETS.items()},
        reference_model=reference,
        base_model=DEFAULT_BASE_MODEL,
        peft_dependencies=list(PEFT_DEPENDENCIES),
        qlora_dependencies=list(QLORA_DEPENDENCIES),
        install=peft_dependency_commands(use_qlora=True),
    )


@router.post("/finetune/lora/plan", response_model=LoRAPlanResponse)
def finetune_lora_plan(req: LoRAPlanRequest) -> LoRAPlanResponse:
    """算一次 LoRA 的参数量，并给出预设/秩对照与风险告警（day051）.

    这个端点回答"**这次微调到底训了多少参数**"——在提交任务之前。它不需要
    权重、不需要 GPU，只需要模型规格（公开 config.json 的字段）与 LoRA 配置。

    错误码延续既有契约：未知模型规格 / 配置非法（含未命中目标模块）→ **400**。
    """
    spec = resolve_model_spec(req.model)
    try:
        config = lora_config_from_settings(req.overrides)
        config.validate()
        plan = plan_lora(spec, config)
    except ValueError as exc:
        # PEFTConfigError 继承 ValueError，此处一次覆盖"配置非法"与"未命中模块"
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return LoRAPlanResponse(
        model=spec.to_dict(),
        plan=plan.to_dict(),
        warnings=list(plan.warnings),
        presets=target_preset_table(spec, r=config.r),
        rank_table=rank_comparison(spec),
    )


@router.post("/finetune/lora/memory", response_model=LoRAMemoryResponse)
def finetune_lora_memory(req: LoRAMemoryRequest) -> LoRAMemoryResponse:
    """算全参 / LoRA / QLoRA 三种策略的显存预算（day051）.

    "能不能放进这张卡"是一个会被反复问到的问题，而回答它只需要
    "参数量 × 每参数字节数"这六个分项。这个端点把六项都列出来，并明确
    标注**不含激活显存**——预算表最危险的误读就是把"放得下"当成"跑得起来"。

    错误码：未知模型规格、未知优化器/计算精度、配置非法 → **400**。
    ``qlora_overrides`` 里传本课的对照基线口径（``int4``）同样返回 **400**：
    显存预算只关心位宽，会静默接受它并给出与 ``nf4`` 相同的结果——而
    "对照基线能进生产配置"这件事本身就是个陷阱（详见 ``peft/config.py``
    的 ``BASELINE_ONLY_QUANT_TYPES``）。
    """
    spec = resolve_model_spec(req.model)
    try:
        lora = lora_config_from_settings(req.overrides)
        lora.validate()
        qlora = qlora_config_from_settings(req.qlora_overrides)
        qlora.validate()
        if qlora.bnb_4bit_quant_type in BASELINE_ONLY_QUANT_TYPES:
            raise PEFTConfigError(
                f"{qlora.bnb_4bit_quant_type!r} 是本课的量化对照基线，"
                "不能用于显存/训练配置（bitsandbytes 只接受 nf4 / fp4）"
            )
        plans = compare_strategies(
            spec,
            lora_config=lora,
            qlora_config=qlora,
            optimizer=req.optimizer,
            compute_dtype=req.compute_dtype,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return LoRAMemoryResponse(
        model=spec.to_dict(),
        strategies=savings_table(plans),
        details=[plan.to_dict() for plan in plans],
        devices=device_fit_table(plans),
        quantization_note=quantization_memory_note(qlora),
        optimizer=req.optimizer,
        compute_dtype=req.compute_dtype,
    )


#: ``/finetune/lora/distributed`` 的 ``device_fit`` 所用的单卡预算（GiB）.
#: 取 24 是最常见的"单卡起点"；要换预算直接改这里，而不是在响应里留一个
#: 看起来像真实设备规格的默认值——**配置项与响应必须能一一对应**。
LORA_DISTRIBUTED_BUDGET_GIB = 24.0


@router.post("/finetune/lora/distributed", response_model=LoRADistributedResponse)
def finetune_lora_distributed(req: LoRADistributedRequest) -> LoRADistributedResponse:
    """算多卡/混合精度训练的步数、每设备显存与通信量，并给出可用的配置（day052）.

    这个端点回答三个"多卡之后才会出现的问题"：

    1. **全局批被放大了多少倍**（``per_device × accum × devices``）——学习率、
       warmup、总步数都要跟着改；
    2. **每设备显存降了多少**——取决于策略：``zero2`` 只切优化器状态与梯度，
       ``zero3`` 才切参数。**LoRA 场景下 ``zero2`` 几乎不省**，因为它的优化器
       状态本来就只占适配器那一百来 MB；
    3. **通信量涨了多少**——``zero3`` 每层前向都要 all-gather 参数，代价可能
       比省下的显存更贵。

    返回的 ``config_yaml`` / ``deepspeed_config`` 可以被 accelerate 直接消费；
    ``launch_command`` 与它们同源（同一份 ``devices`` / ``strategy``）。

    错误码：未知模型规格、未知策略/精度/优化器、配置非法 → **400**；
    ``devices`` / ``train_size`` 的取值越界由 Pydantic 在进路由前拦下 → **422**
    （与 day050、day051 的错误码语言一致）。
    """
    spec = resolve_model_spec(req.model)
    mixed_precision = "fp32" if req.mixed_precision == "no" else req.mixed_precision
    try:
        lora = lora_config_from_settings(req.overrides)
        lora.validate()
        qlora = qlora_config_from_settings(req.qlora_overrides) if req.qlora_overrides else None
        if qlora is not None:
            qlora.validate()
        bundle = build_dataset()
        train, eval_set = split_dataset(
            bundle.examples,
            eval_ratio=settings.finetune_eval_ratio,
            seed=settings.finetune_split_seed,
        )
        args = sft_args_from_settings()
        plan = plan_distributed(
            spec,
            args,
            devices=req.devices,
            strategy=req.strategy,
            train_size=req.train_size if req.train_size is not None else len(train),
            eval_size=len(eval_set),
            lora_config=lora,
            qlora_config=qlora,
            full_finetune=req.full_finetune,
            mixed_precision=mixed_precision,
            lr_scaling_mode=req.lr_scaling_mode,
            optimizer=req.optimizer,
        )
        deepspeed_file = (
            None
            if req.strategy == "ddp"
            else f"deepspeed_zero{req.strategy[-1]}.json"
        )
        config = accelerate_config(
            devices=req.devices,
            strategy=req.strategy,
            mixed_precision=mixed_precision,
            deepspeed_config_file=deepspeed_file,
        )
        zero_config = (
            None
            if req.strategy == "ddp"
            else deepspeed_zero_config(
                stage=int(req.strategy[-1]), mixed_precision=mixed_precision
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return LoRADistributedResponse(
        model=spec.to_dict(),
        plan=plan.to_dict(),
        config_yaml=render_config_yaml(config),
        accelerate_config=config,
        deepspeed_config=zero_config,
        launch_command=launch_command(
            f"train_lora_{req.strategy}.py", devices=req.devices, strategy=req.strategy
        ),
        device_fit=device_fit_summary(plan, budget_gib=LORA_DISTRIBUTED_BUDGET_GIB),
        warnings=list(plan.warnings),
    )


@router.post("/finetune/lora/deploy", response_model=LoRADeployResponse)
def finetune_lora_deploy(req: LoRADeployRequest) -> LoRADeployResponse:
    """走一遍"落盘适配器 → 生成清单 → 验证合并 → 体积对照"的部署流程（day052）.

    这是本课唯一一个**会写磁盘**的端点，而且写的是临时目录（请求结束即删除）：
    它要演示的正是"一个适配器 = 三个文件 + 一份清单"这件事，而产物的形状
    必须是真的。``train_steps=0`` 时适配器未训练（``ΔW = 0``），合并验证必然
    通过——响应里的 ``adapter_trained`` 会如实标注这一点。

    错误码：未知模型规格 / 配置非法 / 目标模块不是 ``bigram``（本端点用参考
    模型演示）→ **400**。
    """
    spec = resolve_model_spec(req.model)
    try:
        # 本端点用参考模型的 bigram 适配器演示部署流程，因此**预设缺省就是
        # bigram**：否则空请求体会因为"缺省预设是 attention"而必然 400，
        # 与"不传参数就能走完整流程"的意图自相矛盾。
        overrides = dict(req.overrides)
        overrides.setdefault("target_modules", "bigram")
        lora = lora_config_from_settings(overrides)
        lora.validate()
        if lora.resolved_targets != (REFERENCE_MODULE_NAME,):
            raise PEFTConfigError(
                "本端点用参考模型的 bigram 适配器演示部署流程，"
                "请把 target_modules 设为 'bigram'（真实模型的合并脚本由 "
                "render_merge_script 生成）"
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    bundle = build_dataset()
    train, eval_set = split_dataset(
        bundle.examples,
        eval_ratio=settings.finetune_eval_ratio,
        seed=settings.finetune_split_seed,
    )
    rendered = render_supervised_list(bundle.examples)
    tokenizer = CharTokenizer.from_texts([item.text for item in rendered])
    reference_config = default_reference_lora_config(
        r=lora.r, lora_alpha=lora.lora_alpha, lora_dropout=0.0
    )
    model = LoRAReferenceModel(
        ReferenceSFTModel(tokenizer.vocab_size, seed=42), reference_config, seed=42
    )
    if req.train_steps:
        train_batches = iter_batches(
            [
                encode_supervised(item, tokenizer, max_length=320)
                for item in render_supervised_list(list(train))
            ],
            batch_size=2,
            drop_last=True,
        )
        for step in range(req.train_steps):
            batch = train_batches[step % len(train_batches)]
            model.accumulate(batch)
            model.apply_update(REFERENCE_LEARNING_RATE)
    eval_batches = iter_batches(
        [
            encode_supervised(item, tokenizer, max_length=320)
            for item in render_supervised_list(list(eval_set))
        ],
        batch_size=2,
        drop_last=False,
    )
    verification = verify_merge(model, eval_batches)

    with tempfile.TemporaryDirectory() as tmp:
        args = SFTTrainingArgs(
            output_dir=tmp,
            learning_rate=REFERENCE_LEARNING_RATE,
            num_train_epochs=2.0,
            max_length=320,
            save_strategy="no",
            eval_strategy="no",
        )
        save_adapter(
            Path(tmp) / "adapter-final",
            model=model,
            args=args,
            step=req.train_steps,
            boundary="adapter-final",
        )
        manifest = build_manifest(
            Path(tmp) / "adapter-final",
            base_model=DEFAULT_BASE_MODEL,
            tags=req.tags,
        )
        deployment = deployment_report(
            spec_parameters=spec.total_parameters(), manifest=manifest
        )
        registry = adapter_registry([manifest])
        merge_lines = len(
            render_merge_script(reference_config).splitlines()
        )
        inference_lines = len(render_inference_script().splitlines())

    return LoRADeployResponse(
        reference=reference_lora_accounting(tokenizer.vocab_size, reference_config),
        manifest=manifest.to_dict(),
        verification=verification.to_dict(),
        deployment=deployment,
        registry=registry,
        adapter_files=list(ADAPTER_FILES),
        merge_script_lines=merge_lines,
        inference_script_lines=inference_lines,
        adapter_trained=req.train_steps > 0,
    )
