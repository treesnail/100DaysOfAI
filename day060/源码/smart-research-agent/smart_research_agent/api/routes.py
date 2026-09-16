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

import hashlib
import json
import tempfile
from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from smart_research_agent.agent.fc_agent import FunctionCallingAgent
from smart_research_agent.alignment import (
    ALIGNMENT_RISKS,
    SEED_PAIRS,
    dimension_table,
    objective_requirements,
    pairs_for_margin,
    preference_stats,
    run_alignment,
    strategy_notes,
)
from smart_research_agent.api.schemas import (
    AgentRunRequest,
    AgentRunResponse,
    AlignmentDimensionsResponse,
    AlignmentRunRequest,
    AlignmentRunResponse,
    ChatRequest,
    ChatResponse,
    DatasetIssue,
    DatasetStatsResponse,
    DatasetValidateRequest,
    DatasetValidateResponse,
    DomainAugmentRequest,
    DomainAugmentResponse,
    DomainDimensionsResponse,
    DomainRunRequest,
    DomainRunResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    FinetuneEvalRunRequest,
    FinetuneEvalRunResponse,
    FinetuneEvalSuiteResponse,
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
    MLOpsDryRunRequest,
    MLOpsDryRunResponse,
    MLOpsGateRequest,
    MLOpsGateResponse,
    MLOpsStagesResponse,
    MLOpsWorkflowResponse,
    ModerationInfo,
    PerfBaselineResponse,
    PipelineRunRequest,
    PipelineRunResponse,
    RegistryCandidateRequest,
    RegistryCandidateResponse,
    RegistryRollbackRequest,
    RegistryRollbackResponse,
    RegistryTriggersRequest,
    RegistryTriggersResponse,
    RegistryVersionInput,
    RegistryVersionsRequest,
    RegistryVersionsResponse,
    SFTDefaultsResponse,
    SFTPlanRequest,
    SFTPlanResponse,
    SFTPreviewRequest,
    SFTPreviewResponse,
    ServingBindingRequest,
    ServingBindingResponse,
    ServingCostRequest,
    ServingCostResponse,
    ServingProbeInput,
    ServingTargetsResponse,
    ServingVerifyRequest,
    ServingVerifyResponse,
    SimilarityRequest,
    SimilarityResponse,
    StageInfo,
    ToolCallRecord,
    VisionDescribeResponse,
)
from smart_research_agent.config import settings

# day057 领域数据准备与增强（M5-D8）：/data/domain/* 三个端点消费的公开接口。
# 注意 ``domain_data.dimension_table`` 与上面 ``alignment.dimension_table``
# 同名（一个是数据质量五维、一个是偏好对齐五维），这里显式改名——
# 两张表都叫"维度表"，但一个回答"质量分怎么算"、另一个回答"偏好买哪几种"，
# 混用会让接口返回一张看起来合理、实际毫不相干的表。
# 显式别名按 isort 的口径单独成块（`import y as z` 与普通导入不同组）。
from smart_research_agent.domain_data import (
    DEFAULT_OPS,
    STAGE_ORDER,
    DomainDataError,
    DomainDataPipeline,
    augment_dataset,
    augment_ops_table,
    default_weights,
)
from smart_research_agent.domain_data import (
    dimension_table as domain_dimension_table,
)
from smart_research_agent.evaluation.perf_baseline import PerfBaseline
from smart_research_agent.finetune.dataset import build_dataset, compute_stats, split_dataset
from smart_research_agent.finetune.overview import METHODS, MethodProfile
from smart_research_agent.finetune.schema import (
    SUPPORTED_FORMATS,
    DatasetFormatError,
    parse_example,
)
from smart_research_agent.finetune_eval import (
    BUCKET_DESCRIPTIONS,
    BUCKETS,
    COMPONENT_NAMES,
    audit_suite,
    build_suite,
    detect_leakage,
    encode_probe_batches,
    run_finetune_evaluation,
    split_suite,
    suite_stats,
    weights_snapshot,
)
from smart_research_agent.integration.pipeline import IntegratedPipeline, PipelineResult
from smart_research_agent.llm.base import BaseLLM, Message
from smart_research_agent.llm.embedding import EmbeddingProvider
from smart_research_agent.memory.vector_store import cosine_similarity

# day059 MLOps 微调流水线（M5-D10）：/mlops/* 四个端点消费的公开接口。
# 四个端点全部只读或纯计算：不写磁盘、不联网、不做真实训练——真正的
# "训练 + 发布"由 scripts/mlops_demo.py 与 GitHub Actions 承担。
from smart_research_agent.mlops import (
    INTENDED_USE,
    LIMITATIONS,
    OUT_OF_SCOPE,
    WORKFLOW_DIRECTORY,
    WORKFLOW_FILENAME,
    CIConfig,
    ExperimentTracker,
    FinetunePipeline,
    MLOpsError,
    ReleaseGates,
    critical_path,
    evaluate_gates,
    gate_table,
    render_github_actions,
    stage_table,
    workflow_commands,
    workflow_summary,
)
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
    train_lora_reference,
)
from smart_research_agent.peft.targets import (
    MODEL_SPECS,
    DecoderSpec,
    plan_lora,
    rank_comparison,
    target_preset_table,
)
from smart_research_agent.peft.trainer import ADAPTER_FILES, save_adapter

# day058 模型版本管理与持续微调（M5-D9）：/registry/* 五个端点消费的公开接口。
# 五个端点全部只读或纯计算，因此路由层只做三件事：把请求体折叠成注册表、
# 调一次判定、把结果连同 markdown 渲染一起返回——**判定逻辑一行都不复制**。
from smart_research_agent.registry import (
    ALLOWED_TRANSITIONS,
    ARTIFACT_NAMES,
    BUMP_KINDS,
    BUMP_MAJOR,
    BUMP_MINOR,
    BUMP_PATCH,
    DEPLOY_REQUIRED_ARTIFACTS,
    PLAN_STEPS,
    STAGES,
    STAGE_STABLE,
    STEP_MEANINGS,
    VERSION_KEY_LENGTH,
    ModelRegistry,
    ModelVersion,
    PromotionPolicy,
    RegistryError,
    TriggerPolicy,
    TriggerState,
    VersionTriple,
    evaluate_candidate,
    evaluate_triggers,
    plan_rollback,
    semver_sort_key,
    trigger_table,
    utc_now_iso,
)
from smart_research_agent.security.content_moderator import ContentModerator

# day060 专属模型部署与切换（M5-D11）：/serving/* 四个端点消费的公开接口。
# 四个端点全部只读或纯计算：不写磁盘、不联网、**不加载任何模型**。
# 真正加载模型、压测延迟、切流量由 scripts/serving_demo.py 与部署脚本承担
# ——一个需要 GPU 的流程不该挂在一个 HTTP 请求上（与 day059 的取舍同源）。
from smart_research_agent.serving import (
    ARCH_QWEN3_8B,
    DEFAULT_NUM_PARALLEL,
    HOURS_PER_MONTH,
    SERVING_LIMITATIONS,
    SERVING_OUT_OF_SCOPE,
    BindingPolicy,
    ProbeCase,
    ServedEndpoint,
    ServingError,
    ServingSpec,
    TrafficPolicy,
    VerifyPolicy,
    binding_table,
    bind_version,
    cloud_pricing,
    compare_costs,
    gpu_pricing,
    memory_breakdown,
    price_book,
    recommend_kind,
    route_table,
    run_verification,
    spec_table,
    verify_binding,
    verify_table,
)
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


#: ``/finetune/eval/suite`` 最多返回多少条用例明细（day053）.
#: 与 ``MAX_DATASET_ISSUES`` 同一条纪律：评估集是要被人 review 的，
#: 而"review 18 条"与"review 一万条"是两件事。本课套件只有 18 条，
#: 这个上限是给将来扩充套件时留的闸门——统计画像永远是全量的。
MAX_EVAL_SUITE_ITEMS = 200


@router.get("/finetune/eval/suite", response_model=FinetuneEvalSuiteResponse)
def finetune_eval_suite() -> FinetuneEvalSuiteResponse:
    """给出领域评估集的画像：桶/难度分布、难度体检、切分与泄漏体检（day053）.

    这个端点**不跑模型**，它回答的是"要评什么"：

    1. **评哪六类能力**——``citation`` / ``tool_use`` / ``format`` /
       ``factuality`` / ``refusal`` / ``conciseness``，每个桶 3 条；
    2. **难度是结构负载推出来的**，不是主观标的——``audit`` 列出所有
       "声明难度 ≠ 推导难度"的用例，这是评估集最容易被悄悄污染的地方；
    3. **训练集与评估集的重叠**——``leakage`` 用规范化字符 3-gram 的
       Jaccard 找可疑对；实测本课套件的最大相似度是 0.054348（阈值 0.5），
       而把同一条用例复制一份进评估集时它立刻报到 1.0（有测试守着）。

    切分用的是 ``finetune_eval_suite_ratio``，**注意它与
    ``finetune_eval_ratio`` 不是一回事**：后者切的是微调数据（day048），
    前者切的是评估套件。
    """
    items = build_suite()
    train, evaluation = split_suite(
        items,
        eval_ratio=settings.finetune_eval_suite_ratio,
        seed=settings.finetune_eval_seed,
    )
    difficulty = {
        level: sum(1 for item in evaluation if item.difficulty == level)
        for level in ("easy", "normal", "hard")
    }
    return FinetuneEvalSuiteResponse(
        suite=suite_stats(items),
        audit=audit_suite(items),
        buckets=[
            {"name": bucket, "description": BUCKET_DESCRIPTIONS[bucket]}
            for bucket in BUCKETS
        ],
        component_names=list(COMPONENT_NAMES),
        weights=weights_snapshot(),
        train_ids=[item.id for item in train],
        eval_ids=[item.id for item in evaluation],
        eval_difficulty=difficulty,
        leakage=detect_leakage(train, evaluation),
        items=[item.to_dict() for item in items[:MAX_EVAL_SUITE_ITEMS]],
    )


@router.post("/finetune/eval/run", response_model=FinetuneEvalRunResponse)
def finetune_eval_run(req: FinetuneEvalRunRequest) -> FinetuneEvalRunResponse:
    """跑一次完整的微调评估：两臂 → 配对比较 → 白盒探针 → 报告（day053）.

    流水线的每一步都有它自己的"为什么"：

    1. **两臂必须跑在同一份评估集上**——配对比较（McNemar、配对自助法）
       的前提，跑在不同子集上的比较在统计上无意义；
    2. **合格是合取**——事实点全覆盖 + 无禁项 + 格式合规 + 拒答行为与预期
       一致。加权总分单独报告，回答"好多少"，不参与合格判定；
    3. **白盒探针是唯一花算力的环节**（``probe=true`` 时在参考模型上真的
       训练一次 LoRA）。它给出"模型对领域答案有多熟"这个内部信号，
       与文本侧指标是两件事——实测里两者甚至会背离（本课第六章）。

    错误码：``eval_ratio`` / ``min_pass_rate`` / ``alpha`` 等取值越界由
    Pydantic 拦下 → **422**；未知对照臂、权重不合法等配置问题 → **400**
    （与 day050、day051、day052 的错误码语言一致）。
    """
    items = build_suite()
    probe = None
    if req.probe:
        tokenizer = CharTokenizer.from_texts(
            [item.instruction + item.reference for item in items]
        )
        base_model = ReferenceSFTModel(tokenizer.vocab_size, seed=req.seed)
        train_items, _ = split_suite(items, eval_ratio=req.eval_ratio, seed=req.seed)
        try:
            model, _, _ = train_lora_reference(
                base_model,
                encode_probe_batches(train_items, tokenizer),
                config=default_reference_lora_config(),
                learning_rate=req.probe_learning_rate,
                epochs=req.probe_epochs,
                seed=req.seed,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        probe = (base_model, model, tokenizer)
    try:
        outcome = run_finetune_evaluation(
            items=items,
            eval_ratio=req.eval_ratio,
            seed=req.seed,
            min_pass_rate=req.min_pass_rate,
            max_regression=req.max_regression,
            bootstrap_samples=req.bootstrap_samples,
            alpha=req.alpha,
            probe=probe,
            notes=(
                "两臂为脚本化对照臂（确定性函数），不是真实模型输出",
                "白盒探针在参考模型的 bigram 适配器上真实训练得到",
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FinetuneEvalRunResponse(**outcome.to_dict())


#: ``/finetune/alignment/dimensions`` 里样本量表的目标准确率档位（day054）.
#:
#: 取这四个档位不是为了"多给几个数"，而是因为**它们之间的差距本身就是结论**：
#: 0.55 需要 778 条/维度，0.70 只需要 42 条——相差 18 倍。
ALIGNMENT_TARGET_TABLE = (0.55, 0.6, 0.65, 0.7)


@router.get("/finetune/alignment/dimensions", response_model=AlignmentDimensionsResponse)
def finetune_alignment_dimensions() -> AlignmentDimensionsResponse:
    """给出对齐的"要买什么"：五个维度、样本量算术、目标对照与风险（day054）.

    这个端点**不跑模型、也不需要偏好数据**——它回答的是开工前的三个问题：

    1. **偏好哪几个维度**（``citation`` / ``format`` / ``conciseness`` /
       ``refusal`` / ``honesty``）。维度不明确时标注者之间的一致度会低到
       数据不可用（见 ``strategy.annotator_agreement`` 的 κ）；
    2. **要多少条样本**——按"二元比例 vs 0.5"的正态近似算：
       ``n = ⌈(z_{α/2}+z_β)²·p(1−p)/(p−0.5)²⌉``，实测 60% 需要 189 条/维度、
       55% 需要 778 条/维度。**它回答的是"这个目标值不值"**；
    3. **这一步有哪些已知风险**（长度偏置 / 过优化 / 维度混淆 / 参考模型漂移），
       每条都带症状、缓解手段与**用哪个函数检测**。
    """
    return AlignmentDimensionsResponse(
        dimensions=dimension_table(),
        sample_budget=[
            {"target_accuracy": target, "pairs_per_dimension": pairs_for_margin(
                target_accuracy=target
            )}
            for target in ALIGNMENT_TARGET_TABLE
        ],
        objectives=objective_requirements(),
        risks=[dict(risk) for risk in ALIGNMENT_RISKS],
        notes=strategy_notes(),
        seed_stats=preference_stats(list(SEED_PAIRS)),
    )


@router.post("/finetune/alignment/run", response_model=AlignmentRunResponse)
def finetune_alignment_run(req: AlignmentRunRequest) -> AlignmentRunResponse:
    """跑一次最小 DPO 对齐：起点自检 → 真训练 → 留出评估 → 过优化体检（day054）.

    三处刻意的设计：

    1. **起点自检**：未训练时策略与参考模型逐位相同，DPO loss 必须等于
       ``ln 2 = 0.693147``（``initial_check_passed``）。这条检查能抓住
       "ref 或 β 接错了"——**它们在训练日志里只表现为"loss 有点怪"**；
    2. **留出评估**：偏好准确率在按维度分层留出的偏好对上算。用训练集
       margin 当早停判据等价于"训练 loss 降到 0 就停"；
    3. **过优化体检**：三条判据（留出 margin 回落 / KL 超预算 / 准确率饱和）
       的结论**都交出来**——空的告警清单有歧义，它既可能是"没有过优化"，
       也可能是"历史太短，什么都没测出来"。

    错误码：``beta`` / ``learning_rate`` / ``epochs`` / ``valid_ratio`` 等取值
    越界由 Pydantic 拦下 → **422**；偏好数据为空、切分后验证集为空等配置问题
    → **400**（与 day050~day053 的错误码语言一致）。
    """
    try:
        outcome = run_alignment(
            valid_ratio=req.valid_ratio,
            beta=req.beta,
            learning_rate=req.learning_rate,
            epochs=req.epochs,
            seed=req.seed,
            kl_budget=req.kl_budget,
            target_accuracy=req.target_accuracy,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    payload = outcome.to_dict()
    return AlignmentRunResponse(
        stats=payload["stats"],
        train_ids=payload["train_ids"],
        valid_ids=payload["valid_ids"],
        plan=payload["plan"],
        reference_hash=payload["reference_hash"],
        initial_loss=payload["initial_loss"],
        zero_margin_loss=payload["zero_margin_loss"],
        initial_check_passed=payload["initial_check_passed"],
        before=payload["before"],
        after=payload["after"],
        accuracy_gain=payload["accuracy_gain"],
        report=payload["report"] or {},
        optimization=payload["optimization"],
    )


# --------------------------------------------------------------------------- #
# day057 领域数据准备与增强（M5-D8）：/data/domain/* 三个端点
#
# 三个端点全部**只读或纯计算**：不写磁盘、不联网、不用随机数（增强算子的
# 轮转与配比削减都是确定性的），因此同一个请求永远得到同一个响应。
# 唯二会读文件系统的是 ``/data/domain/run``（它读的是仓库内置的三个数据源，
# 与 day048 ``/finetune/dataset/stats`` 同一口径）。
# --------------------------------------------------------------------------- #


def domain_pipeline_from_settings(
    *,
    augment: bool | None = None,
    mixing: bool | None = None,
    group_by: str | None = None,
    max_group_ratio: float | None = None,
    quality_threshold: float | None = None,
    near_dup_threshold: float | None = None,
) -> DomainDataPipeline:
    """按配置装配领域数据流水线，并应用请求里的覆盖项（day057）.

    为什么默认值走 ``settings.domain_*`` 而不是 dataclass 的字段缺省值：
    两者的**标定对象不同**——``DomainDataPipeline`` 的缺省是"通用经验值"
    （近重复阈值 0.7、配比上限 0.5），而 ``settings.domain_*`` 是"本课程
    语料标定出来的值"（质量门槛 0.6 对应清洗后 37 条的实测分布）。让端点
    走配置，意味着"部署层面可调"这件事对领域数据流水线也成立（与 day050
    的 ``sft_args_from_settings``、day051 的 ``lora_config_from_settings``
    同一种思路）。

    ``mixing`` 没有对应的配置项：配比削峰是"先削峰、后填谷"这条顺序里的
    前一环，缺省必须开着（关掉它就等于让 augment 在一个倾斜的分布上填谷）。
    要关只能在请求里显式关。

    **这里不做任何数值校验**：``max_group_ratio`` / ``group_by`` /
    ``near_dup_threshold`` 的合法区间由 ``domain_data`` 包在构造期与
    进流水线之前判定（该包 ``errors.py`` 立的纪律），越界时抛出
    ``DomainDataError`` / ``ValueError``；路由只负责把它转成 400。
    把校验复制一份到这里，等于维护两份会各自漂移的规则。
    """
    return DomainDataPipeline(
        quality_threshold=(
            settings.domain_quality_threshold if quality_threshold is None else quality_threshold
        ),
        near_dup_threshold=(
            settings.domain_near_dup_threshold
            if near_dup_threshold is None
            else near_dup_threshold
        ),
        shingle_k=settings.domain_shingle_k,
        num_perm=settings.domain_num_perm,
        group_by=settings.domain_group_by if group_by is None else group_by,
        max_group_ratio=(
            settings.domain_max_group_ratio if max_group_ratio is None else max_group_ratio
        ),
        mixing=True if mixing is None else mixing,
        augment=settings.domain_augment_enabled if augment is None else augment,
        max_augment_per_example=settings.domain_max_augment_per_example,
    )


@router.get("/data/domain/dimensions", response_model=DomainDimensionsResponse)
def data_domain_dimensions() -> DomainDimensionsResponse:
    """给出领域数据流水线的"怎么算、能做什么、按什么顺序"（day057）.

    这个端点**不读数据、不跑流水线、不花钱**，它回答的是开工前的三个问题：

    1. **质量分怎么算**（``dimensions`` + ``weights`` + ``threshold``）：
       五个维度各自的含义、计算方式、挡住的故障与缺省权重。表里的权重列
       由 ``QualityWeights`` 现场读出，门槛来自 ``settings``——"文档说
       0.25、代码是 0.2"这类漂移会在接口上立刻可见；
    2. **增强能做哪几种**（``augment_ops`` + ``default_ops``）：四个算子
       各自改善什么维度、有没有风险、默认是否启用。``noise`` 在表里但
       **不在** ``default_ops`` 里——它造成的质量下降恰好不在质量打分的
       五个维度内，属于"打包分器看不见的风险"，只能靠显式标签与默认关闭
       治理（见 ``domain_data/augment.py`` 的纪律三）；
    3. **流水线按什么顺序走**（``stage_order``）：六阶段顺序即策略，
       换来换去账就对不上（先配比后增强会白花一次增强）。

    返回的每一个数字都来自代码常量而不是手写文档，因此它可以被当作
    "这份实现的自我描述"来消费。
    """
    return DomainDimensionsResponse(
        dimensions=domain_dimension_table(),
        weights=default_weights().as_dict(),
        threshold=settings.domain_quality_threshold,
        augment_ops=augment_ops_table(),
        default_ops=list(DEFAULT_OPS),
        stage_order=list(STAGE_ORDER),
    )


@router.post("/data/domain/augment", response_model=DomainAugmentResponse)
def data_domain_augment(req: DomainAugmentRequest) -> DomainAugmentResponse:
    """对一批原始样本做增强，返回新增样本与报告（day057）.

    与 day048 ``/finetune/dataset/validate`` 完全同一套请求解析风格：
    ``examples`` 是未解析的原始字典，逐条走 ``parse_example``，格式名与
    解析函数都复用既有实现（不另造一份"增强专用的解析器"——那样两份
    规则会各自漂移）。

    三条不变式在响应里可以被逐条验证，而不是只写在注释里：

    - **``output`` 与输入逐字相同**：增强只改 prompt 侧。动了答案，
      增强就从"换一种问法"变成"生成新答案"，而生成就可能出错；
    - ``source`` / ``license`` / ``input`` 原样继承：增强不改变数据出处；
    - ``tags`` 追加 ``aug:<算子名>``：这批数据里有多少是合成的，永远可查。

    ``ops`` 留空即用 ``DEFAULT_OPS``（不含 ``noise``）；算子按
    "样本序号 + 第几次尝试"轮转选取，**不用随机数**，因此同一请求的响应
    逐字节可复现。

    错误码：

    - 未知 ``format`` → **400**（与 day048 同一条错误码语言）；
    - 某条样本无法解析 → **400**，详情里带该条的下标与 ``parse_example``
      的原文（本端点没有 issue 列表可放，逐条报错比静默跳过有用）；
    - 未知算子名 → **400**，详情来自 ``DomainDataError`` 原文——算子名
      打错最糟的表现是**静默少生成一批样本**，报告里只是少一个键；
    - ``examples`` 为空 → **422**，由 Pydantic 的 ``min_length=1`` 拦下。
    """
    if req.format not in SUPPORTED_FORMATS:
        available = ", ".join(SUPPORTED_FORMATS)
        raise HTTPException(
            status_code=400,
            detail=f"未知数据集格式: {req.format}，可用格式: {available}",
        )

    examples = []
    for index, raw in enumerate(req.examples):
        try:
            examples.append(parse_example(raw, req.format))
        except DatasetFormatError as exc:
            raise HTTPException(
                status_code=400, detail=f"第 {index} 条样本无法解析：{exc}"
            ) from exc

    ops = list(req.ops) if req.ops else list(DEFAULT_OPS)
    try:
        augmented, report = augment_dataset(
            examples, ops=ops, max_per_example=req.max_per_example
        )
    except (DomainDataError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # 返回列表的前 len(examples) 条就是原样本本身（augment_dataset 的约定：
    # 增强只做追加、从不重排），因此"新增"这一段可以从下标切出来。
    generated = augmented[len(examples) :]
    return DomainAugmentResponse(
        inputs=len(examples),
        generated=len(generated),
        expanded=len(augmented),
        examples=[
            {
                "instruction": example.instruction,
                "output": example.output,
                "input": example.input,
                "tags": list(example.tags),
                "source": example.source,
                "license": example.license,
            }
            for example in generated
        ],
        report=report.to_dict(),
    )


@router.post("/data/domain/run", response_model=DomainRunResponse)
def data_domain_run(req: DomainRunRequest) -> DomainRunResponse:
    """用仓库内置数据源跑一次六阶段流水线，返回规模 + 清单 + 各阶段报告（day057）.

    这是本课的领域数据流水线第一次可被 HTTP 触发。它跑的是
    ``DomainDataPipeline.run_from_sources``：采集（种子样本 + 评估轨迹 +
    红队安全样本）→ 清洗 → 五维打分 → 两级去重 → 配比削峰 → 增强填谷 →
    冻结打指纹，**不落盘**（落盘由 ``scripts/`` 与流水线自己的 ``save``
    负责——HTTP 端点不该替调用方决定把产物写到哪里）。

    三个刻意的取舍：

    1. **不返回样本正文**，只返回 ``size`` + ``manifest`` + 五个阶段报告。
       清单里已经有阶段账（进/出/丢弃/新增）、来源分布、配比与占比、
       质量分布、拒绝归因与**全量参数快照**，"这次跑成什么样"由清单回答；
       正文交给落盘产物；
    2. **报告与清单并列返回**：清单描述结论、报告描述过程。清单里看不到
       "某条样本被哪条硬规则拒了"（``clean.drop_reasons``）、"削减发生在
       哪一组"（``mixing.dropped_by_group``）这类过程信息；
    3. **参数默认值来自 ``settings``**（见 ``domain_pipeline_from_settings``），
       请求体只覆盖要动的那几项，因此 ``{}`` 就是"按当前配置跑一次"。

    错误码：非法参数（``group_by`` 不在 source/origin/safety 三选一、
    ``max_group_ratio`` 不在 ``(0, 1]``、``near_dup_threshold`` 不在
    ``(0, 1]``）→ **400**，详情是 ``DomainDataError`` / ``ValueError``
    的原文。**刻意不在 Pydantic 上复制这些区间**：复制出来的第二份规则
    会漂移，而且会把"参数不合法"从 400 变成 422（与 day050~day055 的
    错误码语言冲突）。
    """
    pipeline = domain_pipeline_from_settings(
        augment=req.augment,
        mixing=req.mixing,
        group_by=req.group_by,
        max_group_ratio=req.max_group_ratio,
        quality_threshold=req.quality_threshold,
        near_dup_threshold=req.near_dup_threshold,
    )
    try:
        run = pipeline.run_from_sources(
            version=req.version, parent_fingerprint=req.parent_fingerprint
        )
    except (DomainDataError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    payload = run.to_dict()
    return DomainRunResponse(
        size=payload["size"],
        manifest=payload["manifest"],
        clean=payload["clean"],
        quality=payload["quality"],
        dedupe=payload["dedupe"],
        mixing=payload["mixing"],
        augment=payload["augment"],
    )


# --------------------------------------------------------------------------- #
# day058 模型版本管理与持续微调（M5-D9）：/registry/* 五个端点
#
# 五个端点全部**只读或纯计算**：不写磁盘、不联网、不调模型、不用随机数，
# 因此同一个请求体永远得到同一个响应。版本表的**写入**（register / set_stage）
# 只发生在持续微调流水线与脚本里——HTTP 端点在多副本部署下无法保证"谁先写"，
# 而版本表的写入必须是单点的（见 day058 教程第五章）。
#
# 请求体带**全量版本记录**而不是让服务端去读
# ``outputs/registry/versions.jsonl``：这样端点无状态、可复现、可在任何机器上
# 重放，也让"折叠事件流"这件事在接口上可见（请求里的 stage 就是折叠后的状态）。
# --------------------------------------------------------------------------- #


def registry_from_payload(versions: list[RegistryVersionInput]) -> ModelRegistry:
    """把请求里的版本记录折叠成一个内存注册表（M5-D9）.

    按**版本号升序**登记而不是按请求顺序：父版本必须先于子版本出现，
    否则 ``register`` 会以"悬空的父指针"拒绝——而调用方按什么顺序写
    请求体，不该影响服务端的判定结果。

    校验一行都不复制：三元组的长度与十六进制、版本号格式、阶段名合法性、
    父版本存在性全部由 ``registry`` 包在构造期判定，越界时抛
    ``RegistryError``；路由只负责把它转成 400。
    """
    registry = ModelRegistry()
    ordered = sorted(versions, key=lambda item: semver_sort_key(item.version))
    for item in ordered:
        registry.register(
            ModelVersion(
                triple=VersionTriple(
                    base_model=item.base_model,
                    adapter_sha256=item.adapter_sha256,
                    dataset_fingerprint=item.dataset_fingerprint,
                ),
                version=item.version,
                parent_version=item.parent_version,
                created_at=item.created_at or utc_now_iso(),
                stage=item.stage,
                artifacts={str(k): str(v) for k, v in item.artifacts.items()},
                metrics={str(k): float(v) for k, v in item.metrics.items()},
                notes=item.notes,
                tags={str(k): str(v) for k, v in item.tags.items()},
            )
        )
    return registry


def promotion_policy_from(overrides: dict | None) -> PromotionPolicy:
    """按 ``settings.retrain_*`` 装配采纳策略，并应用请求里的覆盖项（day058）.

    与 day057 的 ``domain_pipeline_from_settings`` 同一种做法：
    dataclass 的字段缺省是"通用经验值"，``settings`` 是"本课程标定值"，
    端点走配置意味着"部署层面可调"对采纳策略也成立。

    **未知键被忽略**而不是报错：策略字段会随版本演进（今天三个、明天可能四个），
    让旧客户端带上一个已删除的键就 500 是不必要的严格；而**真正非法的值**
    （负的 ``min_gain``、越界的 ``absolute_min_pass_rate``）由
    ``PromotionPolicy.__post_init__`` 拦下，路由把它转成 400。
    """
    payload: dict = {
        "min_gain": settings.retrain_min_gain,
        "regression_tolerance": settings.retrain_regression_tolerance,
        "absolute_min_pass_rate": settings.retrain_absolute_min_pass_rate,
        "max_candidate_age_hours": settings.retrain_max_candidate_age_hours,
    }
    payload.update({k: v for k, v in (overrides or {}).items() if k in payload})
    return PromotionPolicy(**payload)


def trigger_policy_from(overrides: dict | None) -> TriggerPolicy:
    """按 ``settings.retrain_*`` 装配触发策略，并应用请求里的覆盖项（day058）."""
    payload: dict = {
        "min_new_examples": settings.retrain_min_new_examples,
        "min_pass_rate": settings.retrain_min_pass_rate,
        "cooldown_hours": settings.retrain_cooldown_hours,
        "max_parallel_runs": settings.retrain_max_parallel_runs,
    }
    payload.update({k: v for k, v in (overrides or {}).items() if k in payload})
    return TriggerPolicy(**payload)


@router.get("/registry/layout")
def registry_layout() -> dict:
    """给出模型版本管理的"身份怎么算、号怎么涨、状态怎么流转"（day058）.

    这个端点**不读磁盘、不建注册表、不花钱**，它回答的是开工前的四个问题：

    1. **身份怎么算**（``version_key``）：三元组的内容寻址规则与长度；
    2. **版本号怎么涨**（``bump_kinds`` + ``bump_rules``）：三项里变的那一项
       决定递增位——``major`` 换基座、``minor`` 换数据集、``patch`` 重训；
    3. **状态怎么流转**（``stages`` + ``transitions``）：四态的允许迁移表。
       表由 ``ALLOWED_TRANSITIONS`` 现场读出，因此
       "文档说 stable 可以退回 candidate、代码里不行"这类漂移会立刻可见；
    4. **一次重训按什么顺序走**（``plan_steps``）：六步顺序即策略。

    返回的每一个数字都来自代码常量而不是手写文档，因此它可以被当作
    "这份实现的自我描述"来消费（与 day057 的 ``/data/domain/dimensions`` 同一取舍）。
    """
    stage_meanings = {
        "candidate": "已登记、尚未通过门禁的候选（head() 不会指向它）",
        "stable": "当前或曾经的生产版本（回滚目标只能从它里面选）",
        "rolled_back": "因事故被退回的版本（终止态，不再接受流转）",
        "archived": "被清理出保留窗口的版本（终止态）",
    }
    return {
        "version_key_length": VERSION_KEY_LENGTH,
        "artifact_names": list(ARTIFACT_NAMES),
        "deploy_required_artifacts": list(DEPLOY_REQUIRED_ARTIFACTS),
        "bump_kinds": list(BUMP_KINDS),
        "bump_rules": [
            {
                "kind": BUMP_PATCH,
                "changed_field": "adapter_sha256",
                "meaning": "同数据同基座重训（换种子/调步数），是最可比较的一类变更",
            },
            {
                "kind": BUMP_MINOR,
                "changed_field": "dataset_fingerprint",
                "meaning": "同一基座上的新一版领域数据；与上一版**不可比**",
            },
            {
                "kind": BUMP_MAJOR,
                "changed_field": "base_model",
                "meaning": "更换基座：旧适配器挂不上，整条适配器链作废",
            },
        ],
        "stages": [
            {"stage": name, "meaning": stage_meanings.get(name, "")} for name in STAGES
        ],
        "transitions": {
            name: list(targets) for name, targets in ALLOWED_TRANSITIONS.items()
        },
        "plan_steps": [
            {"order": index, "name": name, "meaning": STEP_MEANINGS.get(name, "")}
            for index, name in enumerate(PLAN_STEPS, start=1)
        ],
        "promotion_policy": promotion_policy_from(None).to_dict(),
        "trigger_policy": trigger_policy_from(None).to_dict(),
        "trigger_conditions": trigger_table(trigger_policy_from(None)),
    }


@router.post("/registry/versions", response_model=RegistryVersionsResponse)
def registry_versions(req: RegistryVersionsRequest) -> RegistryVersionsResponse:
    """折叠一批版本记录，返回阶段分布、当前生产版本与版本表（day058）.

    回答的是运维每天要问的那句话：**"现在线上跑的是哪一版、我手里有哪些候选"**。

    三个刻意的取舍：

    1. **``head`` 无 stable 时为 null**（不退回 candidate）："还没有生产版本"
       与"生产版本是某个候选"是两件事，后者被静默当成前者会让未验证的候选
       出现在部署脚本的视野里；
    2. **``counts`` 四个阶段键恒存在**：报告里 ``rolled_back: 0`` 与
       "没有这个键"读起来完全不同——前者是"回滚过 0 次"，后者是"没人统计过回滚"；
    3. **同时返回 ``markdown``**：接口的输出与人读的报告是同一份数据的两种渲染，
       由 ``ModelRegistry.render_markdown`` 统一生成，因此不会出现
       "接口数字对、报告数字旧"的双份真相。

    错误码：非法三元组/版本号/阶段名/悬空父版本 → **400**，详情是
    ``RegistryError`` 的原文（含"哪些字段不同"这类可操作的证据）。
    """
    try:
        registry = registry_from_payload(req.versions)
        items = registry.versions(stage=req.stage)
    except (RegistryError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    head = registry.head()
    return RegistryVersionsResponse(
        total=registry.counts()["total"],
        counts=registry.counts(),
        head=None if head is None else head.version,
        head_key=None if head is None else head.version_key,
        versions=[item.to_dict() for item in items],
        markdown=registry.render_markdown(),
    )


@router.post("/registry/candidate/evaluate", response_model=RegistryCandidateResponse)
def registry_candidate_evaluate(req: RegistryCandidateRequest) -> RegistryCandidateResponse:
    """判定"要不要把候选提升为生产版本"（day058）.

    ``current`` 留空走**首次上线**分支：只做绝对值门槛与可部署性判定。
    这个分支必须单独存在，否则第一版永远上不了线（增益无从计算）。

    三条判据的顺序与理由（逐项都进响应体的 ``decision.checks``）：

    - **可部署性**：缺少必需产物的候选不该上线。它是唯一一条"证据完整性"
      检查，因此排在最前——分数再高也不能上线一个没有产物的版本；
    - **时效**：放了三周的候选不采纳。它是为了避免"某个旧候选在别人手工
      提升时被顺手带上生产"——而它的 ``actual`` 允许为 ``null``，
      表示创建时间不可解析（跳过而不是判失败）；
    - **增益**：与 ``min_gain`` 比较。**不可比时不算差值**，改用绝对门槛
      ——把"换了数据的候选"与"当前版本"的分数相减，得到的数字既不是提升
      也不是退步，而是两种评估分布之差，而它会被下游当成前者使用。

    错误码：非法三元组 → **400**；策略值非法（负的 ``min_gain`` 等）→ **400**。
    """
    try:
        policy = promotion_policy_from(req.policy)
    except (RegistryError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    def to_record(item: RegistryVersionInput) -> ModelVersion:
        return ModelVersion(
            triple=VersionTriple(
                base_model=item.base_model,
                adapter_sha256=item.adapter_sha256,
                dataset_fingerprint=item.dataset_fingerprint,
            ),
            version=item.version,
            parent_version=item.parent_version,
            created_at=item.created_at or utc_now_iso(),
            stage=item.stage,
            artifacts={str(k): str(v) for k, v in item.artifacts.items()},
            metrics={str(k): float(v) for k, v in item.metrics.items()},
            notes=item.notes,
            tags={str(k): str(v) for k, v in item.tags.items()},
        )

    try:
        candidate = to_record(req.candidate)
        current = None if req.current is None else to_record(req.current)
        decision = evaluate_candidate(current, candidate, policy=policy)
    except (RegistryError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RegistryCandidateResponse(decision=decision.to_dict(), markdown=decision.render_markdown())


@router.post("/registry/triggers/evaluate", response_model=RegistryTriggersResponse)
def registry_triggers_evaluate(req: RegistryTriggersRequest) -> RegistryTriggersResponse:
    """判定"现在该不该重训"，并给出条件表（day058）.

    ``fired`` 与 ``vetoed_by`` **分开返回**，这是本端点唯一需要理解的语义：
    "没有理由训练"（``fired`` 为空）与"有理由但时机不允许"（``fired`` 非空
    且被否决）的运维动作完全不同——前者要去看数据与线上评估，
    后者只需要等。把否决项写成触发器的反向条件会把这两种情况压成同一个信号。

    三个可空字段（``online_pass_rate`` / ``hours_since_last_train``）的语义是
    **缺失**，不是 0：缺失不触发、不否决。把缺失当 0 会让"从未训练过"的仓库
    被冷却期永久拦住（``None → 0 < 24 → veto``），而它恰恰最该训练一次。

    错误码：非法状态（负的 ``new_examples``、越界的 ``online_pass_rate``）
    或非法策略 → **400**。
    """
    try:
        policy = trigger_policy_from(req.policy)
        state = TriggerState(
            new_examples=req.new_examples,
            dataset_fingerprint=req.dataset_fingerprint,
            stable_dataset_fingerprint=req.stable_dataset_fingerprint,
            online_pass_rate=req.online_pass_rate,
            hours_since_last_train=req.hours_since_last_train,
            active_runs=req.active_runs,
        )
        decision = evaluate_triggers(state, policy)
        conditions = trigger_table(policy)
    except (RegistryError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RegistryTriggersResponse(
        decision=decision.to_dict(),
        markdown=decision.render_markdown(),
        conditions=conditions,
    )


@router.post("/registry/rollback/plan", response_model=RegistryRollbackResponse)
def registry_rollback_plan(req: RegistryRollbackRequest) -> RegistryRollbackResponse:
    """为某个出问题的版本生成回滚计划（day058）.

    回滚目标 = 版本链上**第一个可部署的 stable 祖先**。跳过不可部署的祖先是
    必要的：一个 ``stable`` 记录可能只是"当年提升过"，它的磁盘产物早被清理，
    把回滚目标选成它等于计划了一个必然失败的动作。

    动作序列有两处刻意的设计：

    1. **``verify`` 在 ``freeze`` 之前**：回滚是不可逆动作，第一步必须确认
       "有退路"。先冻结当前版本、再发现目标不可部署，结果是
       **生产上没有任何可服务版本**——这个顺序错误只在最坏情况下暴露，
       因此它写成了代码里的固定顺序而不是注释里的建议；
    2. **``should_execute`` 为假时 ``steps`` 为空**："没有退路"必须是一个
       被输出的结论，而不是一个被吞掉的异常。

    错误码：版本记录非法 → **400**；``version`` 不在请求携带的版本表里 → **400**
    （此时``RegistryError`` 的原文是"既不是版本号也不是版本键"）。
    """
    try:
        registry = registry_from_payload(req.versions)
        plan = plan_rollback(
            registry,
            req.version,
            reason=req.reason,
            observe_window_hours=req.observe_window_hours,
        )
        lineage = registry.lineage(req.version)
    except (RegistryError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RegistryRollbackResponse(
        plan=plan.to_dict(),
        markdown=plan.render_markdown(),
        lineage=[item.to_dict() for item in lineage],
    )


# --------------------------------------------------------------------------- #
# day059 MLOps 微调流水线（M5-D10）：/mlops/* 四个端点
#
# 四个端点全部**只读或纯计算**：不写磁盘、不联网、不做真实训练、不用随机数。
# 真正的"训练 + 发布"由 ``scripts/mlops_demo.py`` 与 GitHub Actions 承担——
# **一个需要十分钟的流程不该挂在一个 HTTP 请求上**（见 ``ci.render_github_actions``）。
#
# dry-run 端点用**确定性的参考回调**驱动六阶段：它的用途不是"跑出最好的模型"，
# 而是"先看门禁会不会放行，再决定要不要真训"。
# --------------------------------------------------------------------------- #


def mlops_gates_from_settings(overrides: dict | None) -> ReleaseGates:
    """按 ``settings.mlops_*`` 装配发布门禁，并应用请求里的覆盖项（day059）.

    与 day057 的 ``domain_pipeline_from_settings``、day058 的
    ``promotion_policy_from`` 同一种做法：dataclass 字段缺省是"通用经验值"，
    ``settings`` 是"本课程标定值"，端点走配置意味着"部署层面可调"也成立。

    六个字段分两类处理：四个阈值从 ``settings`` 取，两个布尔开关
    （``require_traceability`` / ``require_artifacts``）**只在请求里显式给出时**
    才覆盖——它们是"要不要检查"，不是一个可标定的数字，给它们配一个
    部署级默认值反而会让"这次演练故意关掉产物检查"变得不显眼。
    """
    source = overrides or {}
    payload: dict = {
        "min_pass_rate": settings.mlops_min_pass_rate,
        "max_regression": settings.mlops_max_regression,
        "max_adapter_mebibytes": settings.mlops_max_adapter_mebibytes,
        "max_cost_per_1k_tokens": settings.mlops_max_cost_per_1k_tokens,
    }
    payload.update({k: v for k, v in source.items() if k in payload})
    for key in ("require_traceability", "require_artifacts"):
        if key in source:
            payload[key] = source[key]
    return ReleaseGates(**payload)


def mlops_derived_hash(*parts: str) -> str:
    """由若干字符串确定性派生一个 64 位十六进制串（适配器哈希的占位）.

    用 ``sha256`` 而不是随机数：dry-run 的响应必须逐字节可复现，
    而"每次请求都换一个适配器哈希"会让版本三元组每次都不同，
    于是注册表里会凭空多出无数条 v1.0.1。
    """
    digest = hashlib.sha256("\u0000".join(parts).encode("utf-8")).hexdigest()
    return digest


def mlops_baseline_registry(
    *, base_model: str, dataset_fingerprint: str, baseline_pass_rate: float
) -> ModelRegistry:
    """构造一个只含"当前生产版本"的内存注册表（dry-run 演练的起点）.

    基线的数据集指纹**刻意与请求不同**：dry-run 想演示的正是"用新数据训出
    一个候选"这条路径，而两者同指纹时 ``ingest`` 阶段会打印"未变化"，
    演练的价值就少了一半。
    """
    baseline_fingerprint = mlops_derived_hash("baseline", dataset_fingerprint)[:16]
    registry = ModelRegistry()
    registry.register(
        ModelVersion(
            triple=VersionTriple(
                base_model=base_model,
                adapter_sha256=mlops_derived_hash("baseline-adapter", dataset_fingerprint),
                dataset_fingerprint=baseline_fingerprint,
            ),
            version="1.0.0",
            stage=STAGE_STABLE,
            metrics={"eval_pass_rate": baseline_pass_rate},
            artifacts={
                "adapter": "outputs/lora/adapters/adapter-baseline",
                "merged": "outputs/lora/merged/baseline",
            },
            notes="dry-run 演练用的合成基线（不落盘）",
        )
    )
    return registry


def mlops_reference_callbacks(
    req: MLOpsDryRunRequest,
) -> tuple[Callable[[dict], dict], Callable[[dict[str, str]], dict[str, float]]]:
    """构造确定性的参考训练/评估回调（不碰模型、不读文件、不联网）."""

    def train(params: dict) -> dict:
        """返回一条合成的训练结果（含 adapter_sha256 与产物路径）."""
        adapter = req.adapter_sha256 or mlops_derived_hash(
            "adapter", req.dataset_fingerprint, json.dumps(params, sort_keys=True)
        )
        short = adapter[:12]
        artifacts = {
            "adapter": f"outputs/lora/adapters/{short}",
            "merged": f"outputs/lora/merged/{short}",
        }
        if not req.with_merged_artifact:
            artifacts.pop("merged")
        return {
            "adapter_sha256": adapter,
            "adapter_mebibytes": req.adapter_mebibytes,
            "train_loss": req.train_loss,
            "step": 42,
            "metrics": {"train_loss": req.train_loss},
            "artifacts": artifacts,
        }

    def evaluate(artifacts: dict[str, str]) -> dict[str, float]:
        """返回一条合成的评估结果（合格率 + 相对基线的变化）."""
        del artifacts
        return {
            "eval_pass_rate": req.pass_rate,
            "eval_pass_rate_delta": round(req.pass_rate - req.baseline_pass_rate, 4),
        }

    return train, evaluate


@router.get("/mlops/stages", response_model=MLOpsStagesResponse)
def mlops_stages() -> MLOpsStagesResponse:
    """给出这条微调流水线的"六步怎么走、门禁查什么、CI 怎么跑"（day059）.

    这个端点**不读磁盘、不跑流水线、不花钱**，它回答四个开工前的问题：

    1. **六个阶段分别依赖什么、产出什么**（``stages`` +
       ``critical_path``）：依赖表由 ``stage_table()`` 现场读出，
       并由 ``validate_stage_order()`` 在导入时校验过——"顺序写错"
       这类问题不会等到运行时才暴露；
    2. **门禁查哪六项、阈值多少**（``gates``）：表里的阈值列来自
       ``settings.mlops_*``，"文档说 64 MiB、代码是 16 MiB"会立刻可见；
    3. **CI 什么时候跑、按什么命令判定**（``ci``）：cron 与门禁脚本命令
       从 ``CIConfig`` 与 ``workflow_commands()`` 读出，
       因此"手动复现 CI"这一行不需要人去 YAML 里抄；
    4. **这套产物不能做什么**（``out_of_scope`` + ``limitations``）：
       **它比"能做什么"更值得在开工前读一遍**，而且四条限制各自带着
       前几天的实测数字（评估集 18 条、参考模型 31 万参数……）。
    """
    config = CIConfig(
        python_version=settings.mlops_ci_python_version,
        schedule=settings.mlops_ci_schedule,
        timeout_minutes=settings.mlops_ci_timeout_minutes,
        gates=mlops_gates_from_settings(None),
    )
    return MLOpsStagesResponse(
        stages=stage_table(),
        critical_path=critical_path(),
        gates=gate_table(config.gates),
        ci=workflow_summary(config),
        intended_use=list(INTENDED_USE),
        out_of_scope=list(OUT_OF_SCOPE),
        limitations=list(LIMITATIONS),
    )


@router.post("/mlops/gates/evaluate", response_model=MLOpsGateResponse)
def mlops_gates_evaluate(req: MLOpsGateRequest) -> MLOpsGateResponse:
    """按发布门禁策略判定一份指标与产物（day059）.

    这是**绝对判定**：与固定阈值比，而不是与当前版本比。两者的分工写在
    ``gates`` 模块的 docstring 里，这里只在响应上体现——缺 ``eval_pass_rate``
    的门禁**不通过**（缺证据不能发布），而 day058 的相对判定在缺数据时
    只能说"不知道"。

    ``report.warnings`` 是"记录但不拦发布"的三项缺省行为：
    没有基线（首次训练）、没有成本指标（离线训练不产生推理成本）。
    **把它们与阻塞失败分开返回**：混在一起会让真正的拦路项被淹没。

    错误码：非法策略值（负的 ``max_regression``、越界的 ``min_pass_rate``）
    → **400**（不是 422）——与 day050~day058 的错误码语言一致。
    """
    try:
        policy = mlops_gates_from_settings(req.policy)
        report = evaluate_gates(
            req.metrics,
            policy=policy,
            artifacts=req.artifacts,
            commit=req.commit,
        )
    except (MLOpsError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return MLOpsGateResponse(report=report.to_dict(), markdown=report.render_markdown())


@router.post("/mlops/pipeline/dry-run", response_model=MLOpsDryRunResponse)
def mlops_pipeline_dry_run(req: MLOpsDryRunRequest) -> MLOpsDryRunResponse:
    """用确定性参考回调演练整条六阶段流水线（day059）.

    它回答的问题是**"如果现在发，门禁会不会放行？"**——
    在花十分钟真训一次之前先把这个问题问清楚，是 dry-run 存在的全部理由。

    三个刻意的取舍：

    1. **注册表始终在内存里**：HTTP 层不做版本表写入（day058 第五章的理由
       在多副本部署下同样成立）。因此 ``version`` 字段会出现、``published``
       也可能为真，但**磁盘上什么都没写**；
    2. **``dry_run=true`` 时门禁通过也不发布**：``publish`` 阶段的状态是
       ``blocked`` 而不是 ``failed``——它表示"上游判定不允许我执行"，
       与"有东西坏了"是两件事；
    3. **五个"结果输入"参数**（合格率、体积、loss、产物开关）让调用方
       可以构造各种门禁情形。这正是 dry-run 的价值：**先看拦在哪一项**。

    错误码：空数据集指纹 → **400**；非法策略值 → **400**；
    Pydantic 层的越界（合格率超出 `[0, 1]`、体积 <= 0）→ **422**。
    """
    try:
        policy = mlops_gates_from_settings(req.policy)
        registry = mlops_baseline_registry(
            base_model=DEFAULT_BASE_MODEL,
            dataset_fingerprint=req.dataset_fingerprint,
            baseline_pass_rate=req.baseline_pass_rate,
        )
        train, evaluate = mlops_reference_callbacks(req)
        pipeline = FinetunePipeline(
            registry,
            ExperimentTracker(),
            base_model=DEFAULT_BASE_MODEL,
            train=train,
            evaluate=evaluate,
            gates=policy,
            run_name="finetune-dry-run",
        )
        outcome = pipeline.run(
            dataset_fingerprint=req.dataset_fingerprint,
            params=req.params,
            commit=req.commit,
            dry_run=req.dry_run,
        )
    except (MLOpsError, RegistryError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    payload = outcome.to_dict()
    return MLOpsDryRunResponse(
        summary=outcome.summary_line(),
        run=payload["run"],
        stages=payload["stages"],
        gate=payload["gate"],
        card=payload["card"],
        manifest=payload["manifest"],
        version=payload["version"],
        published=payload["published"],
        failed=payload["failed"],
        markdown=outcome.render_markdown(),
    )


@router.get("/mlops/ci/workflow", response_model=MLOpsWorkflowResponse)
def mlops_ci_workflow() -> MLOpsWorkflowResponse:
    """返回生成的 GitHub Actions workflow 与摘要（day059）.

    响应里的 ``yaml`` 与仓库中那份 ``.github/workflows/finetune-nightly.yml``
    **逐字相同**（``tests/test_mlops_ci.py`` 有一条测试守着）。因此"改门禁阈值"
    会同时在三个地方生效：``settings`` → 渲染器 → 仓库文件——
    而它们之间**只有一份权威**（``ReleaseGates`` 的字段）。
    """
    config = CIConfig(
        python_version=settings.mlops_ci_python_version,
        schedule=settings.mlops_ci_schedule,
        timeout_minutes=settings.mlops_ci_timeout_minutes,
        gates=mlops_gates_from_settings(None),
    )
    return MLOpsWorkflowResponse(
        summary=workflow_summary(config),
        path=f"{WORKFLOW_DIRECTORY}/{WORKFLOW_FILENAME}",
        yaml=render_github_actions(config),
        commands=workflow_commands(config),
    )


# --------------------------------------------------------------------------- #
# day060 专属模型部署与切换（M5-D11）：/serving/* 四个端点
#
# 四个端点全部**只读或纯计算**：不写磁盘、不联网、不加载任何模型、不用随机数。
# 真正加载模型、量延迟、切流量由 ``scripts/serving_demo.py`` 与部署脚本承担
# ——**一个需要 GPU 的流程不该挂在一个 HTTP 请求上**（与 day059 同一取舍）。
#
# 这一组与 day059 的 /mlops/* 有一个关键分工：
#   /mlops/*   回答"这一版**够不够格发布**"（离线，训练侧）
#   /serving/* 回答"发出去之后**线上跑的是什么、要不要切、切了值不值**"（部署侧）
# 两者的判定都是绝对判定，但**证据不同**：前者要指标，后者要端点自述与实测。
# --------------------------------------------------------------------------- #

#: 端点里默认使用的产物路径。与 day052 的 ``peft.deploy`` 生成的脚本口径一致
#: （``adapter-final`` 永不删除，``outputs/lora-merged`` 是合并脚本的默认输出）。
SERVING_DEFAULT_ADAPTER_DIR = f"{settings.peft_adapter_dir}/adapter-final"
SERVING_DEFAULT_MERGED_DIR = "outputs/lora-merged"

#: 允许在请求里覆盖的部署档案字段。**白名单而不是黑名单**：
#: 部署档案的字段就是这些，"多传一个键"应当被忽略而不是让端点 500。
SERVING_SPEC_FIELDS: tuple[str, ...] = (
    "name",
    "kind",
    "backend",
    "base_model",
    "adapter_dir",
    "merged_dir",
    "context_length",
    "max_parallel",
    "bits_per_parameter",
)


def serving_spec_from_settings(overrides: dict | None = None) -> ServingSpec:
    """按 ``settings.serving_*`` 装配部署档案，并应用请求里的覆盖项（day060）.

    与 day057~day059 的 ``*_from_settings`` 同一套路：dataclass 的字段缺省是
    "通用经验值"，``settings`` 是"本课程标定值"，端点走配置意味着
    "部署层面可调"对部署档案也成立。

    **未知键被忽略**而不是报错（与 day058 的 ``promotion_policy_from`` 同一理由：
    让旧客户端带一个已删除的键就 500 是不必要的严格）；而**真正非法的值**
    （未知形态、adapter 形态缺路径、越界位宽）由 ``ServingSpec.__post_init__``
    拦下，路由把它转成 400。
    """
    source = overrides or {}
    payload: dict = {
        "name": settings.serving_name,
        "kind": settings.serving_kind,
        "backend": settings.local_backend,
        "base_model": DEFAULT_BASE_MODEL,
        "adapter_dir": SERVING_DEFAULT_ADAPTER_DIR,
        "merged_dir": SERVING_DEFAULT_MERGED_DIR,
        "context_length": settings.local_context_length,
        "max_parallel": DEFAULT_NUM_PARALLEL,
    }
    payload.update({k: v for k, v in source.items() if k in SERVING_SPEC_FIELDS})
    return ServingSpec(**payload)


def serving_traffic_from_settings(overrides: dict | None = None) -> TrafficPolicy:
    """按 ``settings.serving_*`` 装配流量策略（day060）."""
    payload: dict = {
        "dedicated_ratio": settings.serving_dedicated_ratio,
        "shadow_ratio": settings.serving_shadow_ratio,
        "fail_open": settings.serving_fail_open,
    }
    payload.update({k: v for k, v in (overrides or {}).items() if k in payload})
    return TrafficPolicy(**payload)


def serving_verify_policy_from_settings(overrides: dict | None = None) -> VerifyPolicy:
    """按 ``settings.serving_*`` 装配上线验证策略（day060）.

    ``max_latency_ratio`` 只有**在请求里显式给出**时才生效：
    它在设置里的缺省是 ``None``（不检查），而"这次要查延迟"是一个
    针对本次切换的决定，不该被一个部署级默认值悄悄打开。
    """
    source = overrides or {}
    payload: dict = {
        "min_pass_rate": settings.serving_min_pass_rate,
        "max_regression": settings.serving_max_regression,
        "max_latency_ratio": settings.serving_max_latency_ratio,
    }
    payload.update({k: v for k, v in source.items() if k in payload})
    return VerifyPolicy(**payload)


def serving_binding_policy_from_settings(overrides: dict | None = None) -> BindingPolicy:
    """按 ``BindingPolicy`` 的缺省装配绑定校验策略（day060）.

    两个开关**刻意没有部署级缺省值**：``require_self_report`` 由后端能力决定
    （不是可标定的数字），``require_registry_head`` 是"这次要不要强制 head"，
    只有请求里显式给出时才设置——与 day059 的两个布尔开关同一处理方式。
    """
    source = overrides or {}
    payload: dict = {}
    for key in ("require_self_report", "require_registry_head"):
        if key in source:
            payload[key] = source[key]
    return BindingPolicy(**payload)


def serving_clock(latencies_ms: list[float]) -> Callable[[], float]:
    """按给定的每用例延迟构造一个确定性时钟（把"实测延迟"变成可注入的输入）.

    ``run_verification`` 会在每个用例的**开始与结束**各调一次 ``clock()``，
    因此一个按 ``[0, l1, l1, l1+l2, l1+l2, ...]`` 依次返回的时钟，
    就能让报告里的延迟精确等于调用方送上来的观测值——**不需要真的等待**，
    也不会引入随机数（与 day059 的 "dry-run 用确定性回调" 同一条纪律）。

    两个臂共用同一个时钟：云端的延迟序列在前、专属的在后，因为
    ``run_verification`` 先跑完云端臂再跑专属臂。
    """
    values: list[float] = [0.0]
    running = 0.0
    for value in latencies_ms:
        running += value / 1000.0
        values.extend([running, running])
    state = {"index": 0}

    def tick() -> float:
        index = state["index"]
        state["index"] = index + 1
        return values[index] if index < len(values) else values[-1]

    return tick


def serving_probes(probes: list[ServingProbeInput]) -> list[ProbeCase]:
    """把请求里的用例转成 ``ProbeCase``（``must_contain`` 逐条转成元组）."""
    return [
        ProbeCase(
            case_id=item.case_id,
            prompt=item.prompt,
            must_contain=tuple(item.must_contain),
        )
        for item in probes
    ]


def serving_arm_responder(
    replies: dict, error_cases: list[str], *, arm: str
) -> Callable[[ProbeCase], str]:
    """把一个推理臂的**观测结果**包装成 ``run_verification`` 需要的回调.

    两条"缺数据"的分支都必须**响亮**（而不是返回空串）：返回空串会让
    "调用方忘了填这一条"伪装成"模型答得不对"，而后者会让人去调模型。
    这与会话外的一条纪律一致：**接口报「答错了」等于宣称测过。**
    """
    missing_text = f"缺少 {arm} 臂的回复"
    failed_text = f"{arm} 臂被标记为调用失败"

    def respond(case: ProbeCase) -> str:
        if case.case_id in error_cases:
            raise ServingError(f"{failed_text}：{case.case_id}")
        if case.case_id not in replies:
            raise ServingError(f"{missing_text}：{case.case_id}")
        return str(replies[case.case_id])

    return respond


def serving_arm_latencies(replies: dict, latencies: dict, cases: list[ProbeCase]) -> list[float]:
    """按用例顺序取出一个臂的延迟（缺省 0.0，**必须是数字**）."""
    del replies
    values: list[float] = []
    for case in cases:
        value = latencies.get(case.case_id, 0.0)
        values.append(float(value))
    return values


@router.get("/serving/targets", response_model=ServingTargetsResponse)
def serving_targets() -> ServingTargetsResponse:
    """给出部署这件事的自我描述：形态、显存、流量、验证、绑定、价格（day060）.

    这个端点**不读磁盘、不加载模型、不联网**，它回答开工前的六个问题：

    1. **三种形态各要什么字段**（``kinds``）：表里的 ``required_fields``
       一列正是"声明成 adapter 却没给路径"在构造期被拦住的地方；
    2. **显存要多少**（``memory``）：权重 / KV 缓存 / 适配器三块**分开列**——
       它们的决定因素完全不同（量化、上下文与并发、挂哪个业务）；
    3. **建议用哪种形态**（``recommendation``）：只按"业务数与是否热插拔"
       两条规则给建议，并显式标出它**是否与当前配置一致**；
    4. **流量怎么分、代价是什么**（``traffic`` + ``traffic_table``）：
       ``dedicated_ratio`` / ``shadow_ratio`` / ``fail_open`` 逐项列出含义
       与代价——**每一条策略都有代价，写出来它才是一个可复核的决定**；
    5. **上线要过哪五条检查**（``verify_table``）：含"没测时算什么"一列；
    6. **部署一致性要核对哪五项**（``binding_table``）：每项都对应一种
       "不会报错但会答错"的故障。

    最后附上**价格表**与"不能做什么"：价格是外部事实，因此它只有一份出处
    （``serving.cost`` 的两个常量），文档与代码里的数字永远相同。
    """
    spec = serving_spec_from_settings(None)
    kind, reason = recommend_kind(variants=1, needs_hot_swap=False)
    traffic = serving_traffic_from_settings(None)
    memory = memory_breakdown(
        ARCH_QWEN3_8B,
        context_length=spec.context_length,
        max_parallel=spec.max_parallel,
        bits_per_parameter=spec.bits_per_parameter,
    )
    # 显存算术需要**结构参数**，而部署档案里的 ``base_model`` 是**训练侧基座**
    # （版本三元组的第一项）。两者不是同一个东西，因此这里显式写出用了哪份结构：
    # **一个不说清来源的显存数字，比没有数字更危险。**
    memory["architecture_note"] = (
        "结构参数取自 Qwen3-8B 的公开配置（36 层 / 8 个 KV 头 / head_dim 128 / "
        "约 8.2B 参数）；换基座必须重填这四个数字，见 serving/spec.py 的 ARCH_QWEN3_8B"
    )
    return ServingTargetsResponse(
        spec=spec.to_dict(),
        kinds=spec_table(),
        memory=memory,
        recommendation={
            "kind": kind,
            "reason": reason,
            "configured_kind": spec.kind,
            "matches_configured": kind == spec.kind,
        },
        traffic=traffic.to_dict(),
        traffic_table=route_table(traffic),
        verify_table=verify_table(serving_verify_policy_from_settings(None)),
        binding_table=binding_table(serving_binding_policy_from_settings(None)),
        price_book=price_book(),
        out_of_scope=list(SERVING_OUT_OF_SCOPE),
        limitations=list(SERVING_LIMITATIONS),
    )


@router.post("/serving/binding/verify", response_model=ServingBindingResponse)
def serving_binding_verify(req: ServingBindingRequest) -> ServingBindingResponse:
    """核对"注册表 / 部署记录 / 端点自述"三处版本是否指向同一份产物（day060）.

    这是本日最重要的一条护栏：**"线上跑的不是我以为的那一版"有三种写法，
    而它们都不会报错**——``adapter_dir`` 指到旧目录、基座被换过、某个副本
    还是旧配置。三条路径的共同点是**证据都在、只是没有人把它们放在一起比对**，
    这个端点做的就是这一次比对。

    五个输入里有两个是"事实"、一个是"现状"：
    ``versions`` + ``bound_version`` 是**打算部署的那一版**（注册表侧），
    ``spec`` 是**部署单元怎么声明**（配置侧），
    ``served`` 是**端点自述现在加载了什么**（运行侧）。
    端点不读磁盘、不猜：所有事实由调用方带来，因此同一个请求体永远同一个响应。

    返回两个布尔量且**必须分开看**：``passed`` 只看阻塞项，
    ``consistent`` 表示全部一致（含非阻塞项）。一个
    ``passed=true / consistent=false`` 的部署是**灰度期间最正常的状态**
    （绑定版本不是注册表 head），合成一个布尔量就再也表达不出它。

    错误码：版本不存在 / 产物不完整 / 基座不匹配 → **400**；
    Pydantic 层的形状错误（``bound_version`` 未给）→ **422**。
    """
    try:
        spec = serving_spec_from_settings(req.spec)
        registry = registry_from_payload(req.versions)
        version = registry.get(req.bound_version)
        binding = bind_version(
            spec,
            version,
            serving_name=req.serving_name,
            notes="由 /serving/binding/verify 构造（不落盘）",
        )
        served_payload = req.served
        served = ServedEndpoint(
            name=str(served_payload.get("name", "")),
            base_model=str(served_payload.get("base_model", "")),
            adapter_short_hash=str(served_payload.get("adapter_short_hash", "")),
            metadata={
                str(k): str(v) for k, v in dict(served_payload.get("metadata", {})).items()
            },
        )
        report = verify_binding(
            binding,
            served,
            head=registry.head(),
            policy=serving_binding_policy_from_settings(req.policy),
        )
    except (ServingError, RegistryError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ServingBindingResponse(
        summary=report.summary_line(),
        passed=report.passed,
        consistent=report.consistent,
        binding=binding.to_dict(),
        report=report.to_dict(),
        markdown=report.render_markdown(),
    )


@router.post("/serving/verify", response_model=ServingVerifyResponse)
def serving_verify(req: ServingVerifyRequest) -> ServingVerifyResponse:
    """在**同一批用例**上比较云端与专属模型，给出五条上线检查（day060）.

    端点**不调用任何模型**：两个臂的观测结果（回复、实测延迟、失败用例）
    由调用方带来。这不是偷懒，而是让"这一次切换够不够好"这件事在 CI 里
    可复现——真正需要 GPU 的压测属于部署脚本。

    五条检查里三条阻塞（用例数 / 合格率 / 退步 / 调用失败），
    延迟一条**缺省不检查**：``"多慢算慢"取决于部署形态``，本地 CPU 与 A10G
    的差异是十倍量级，给一个全局阈值只会让它被无脑放宽。要查就在 ``policy``
    里显式给出 ``max_latency_ratio``。

    延迟做成可注入的确定性时钟：报告里的延迟中位数精确等于请求里送来的
    观测值，因此延迟检查这条分支**可以被逐位复现地**测到，而不是只能跳过。

    错误码：非法策略值（负的 ``max_regression``、越界的 ``min_pass_rate``）、
    用例缺回复或落在 ``error_cases`` 里 → 走**正常报告路径**（缺数据是结论，
    不是错误）；空 ``probes`` → **不通过的报告**（没有证据本身就不切换）。
    """
    try:
        policy = serving_verify_policy_from_settings(req.policy)
        cases = serving_probes(req.probes)
        cloud_replies = dict(req.cloud.replies)
        dedicated_replies = dict(req.dedicated.replies)
        clock = serving_clock(
            serving_arm_latencies(cloud_replies, dict(req.cloud.latency_ms), cases)
            + serving_arm_latencies(dedicated_replies, dict(req.dedicated.latency_ms), cases)
        )
        report = run_verification(
            cases,
            cloud_respond=serving_arm_responder(
                cloud_replies, list(req.cloud.error_cases), arm="云端"
            ),
            dedicated_respond=serving_arm_responder(
                dedicated_replies, list(req.dedicated.error_cases), arm="专属"
            ),
            policy=policy,
            clock=clock,
            notes=req.notes,
        )
    except (ServingError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ServingVerifyResponse(
        summary=report.summary_line(),
        report=report.to_dict(),
        markdown=report.render_markdown(),
    )


@router.post("/serving/cost/compare", response_model=ServingCostResponse)
def serving_cost_compare(req: ServingCostRequest) -> ServingCostResponse:
    """算清"专属模型值不值"：两个成本模型 + 四个派生数字（day060）.

    结论字段 ``cheaper_side`` 只是副产品，**真正该看的是四个数**：

    - ``required_gpu_hours``：处理这批请求**用掉**的算力；
    - ``deployed_gpu_hours``：你**为之付费**的时长（``machines × 开机小时``）；
    - ``utilization_actual``：两者的比值——它解释"为什么自建在低请求量下
      一定更贵"；
    - ``breakeven_requests``：调用量要到多少，自建才真正更省。

    缺省值全部来自 ``settings.serving_*``，其中 ``gpu_tokens_per_second``
    是**待替换的占位值**：吞吐取决于模型、量化、批大小与序列长度，
    没有权威缺省，请求里请填本机实测值，否则盈亏平衡点不可信。

    错误码：未知机型键 / 未知计价方案 / 负的请求量 → **400**；
    Pydantic 层的越界（``gpu_utilization`` 超出 ``(0, 1]``、
    ``requests < 0``）→ **422**。
    """
    try:
        cloud = cloud_pricing(req.cloud_key or settings.serving_cloud_pricing_key)
        gpu = gpu_pricing(
            req.gpu_key or settings.serving_gpu_key,
            tokens_per_second=(
                req.gpu_tokens_per_second or settings.serving_gpu_tokens_per_second
            ),
            utilization=(
                req.gpu_utilization if req.gpu_utilization > 0 else settings.serving_gpu_utilization
            ),
        )
        comparison = compare_costs(
            requests=req.requests,
            input_tokens_per_request=req.input_tokens_per_request,
            output_tokens_per_request=req.output_tokens_per_request,
            cloud=cloud,
            gpu=gpu,
            monthly_hours=req.monthly_hours if req.monthly_hours > 0 else HOURS_PER_MONTH,
        )
    except (ServingError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ServingCostResponse(
        summary=comparison.summary_line(),
        comparison=comparison.to_dict(),
        markdown=comparison.render_markdown(),
    )
