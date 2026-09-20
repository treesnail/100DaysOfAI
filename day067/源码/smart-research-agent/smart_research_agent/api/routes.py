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

import base64
import hashlib
import json
import tempfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

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
    ChunkingEvaluateRequest,
    ChunkingEvaluateResponse,
    ChunkingProbe,
    ChunkingRecordsRequest,
    ChunkingRecordsResponse,
    ChunkingSplitRequest,
    ChunkingSplitResponse,
    ChunkingStrategiesResponse,
    DatasetIssue,
    DatasetStatsResponse,
    DatasetValidateRequest,
    DatasetValidateResponse,
    DocumentDetectRequest,
    DocumentDetectResponse,
    DocumentIngestRequest,
    DocumentIngestResponse,
    DocumentParseRequest,
    DocumentParseResponse,
    DocumentsLoadersResponse,
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
    HybridRetrievalRequest,
    HybridRetrievalResponse,
    IndexingBackupRequest,
    IndexingBackupResponse,
    IndexingBuildRequest,
    IndexingBuildResponse,
    IndexingPlanRequest,
    IndexingPlanResponse,
    IndexingRollbackRequest,
    IndexingRollbackResponse,
    IndexingStatusResponse,
    IndexingVerifyRequest,
    IndexingVerifyResponse,
    IndexingVersionsResponse,
    InjectionInfo,
    LexicalStatusResponse,
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
    RetrievalAnswerRequest,
    RetrievalAnswerResponse,
    RetrievalExplainResponse,
    RetrievalRoutesResponse,
    RetrievalSearchRequest,
    RetrievalSearchResponse,
    RetrievalStatusResponse,
    RetrievalTimeRange,
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
    VectorCompareRequest,
    VectorCompareResponse,
    VectorDeleteRequest,
    VectorDeleteResponse,
    VectorSearchRequest,
    VectorSearchResponse,
    VectorStatsResponse,
    VectorStoreBackendsResponse,
    VectorUpsertRequest,
    VectorUpsertResponse,
    VisionDescribeResponse,
)
from smart_research_agent.config import settings

# day061 文档解析与加载（M6-D1）：/documents/* 四个端点消费的公开接口。
# 四个端点全部只读或纯计算：不写磁盘、不联网、**不遍历目录**。
# 目录遍历与批量入库由 scripts/documents_demo.py 承担——
# 一个要读整个目录的流程不该挂在一个 HTTP 请求上（与 day059/060 同一取舍）。
from smart_research_agent.documents import (
    BLOCK_KINDS,
    DOCUMENTS_LIMITATIONS,
    DOCUMENTS_OUT_OF_SCOPE,
    MEDIA_PDF,
    Document,
    DocumentError,
    DocumentIngestor,
    MediaTypeDetection,
    decode_bytes,
    detect_media_type,
    docx_scope,
    encoding_chain,
    html_rules,
    knowledge_record_shape,
    markdown_syntax_table,
    pdf_boundaries,
)

# day062 分块策略（M6-D2）：/chunking/* 四个端点消费的公开接口。
# 与上面 /documents/* 同一纪律：**只读或纯计算**——不写磁盘、不联网、
# 不遍历目录。需要 embedding 的语义策略走 ``app.state.embedding``
# （与 /embeddings 同一个注入点），因此测试注入 MockEmbedding 即可全离线。
#
# 注意 ``knowledge_record_shape`` 与 documents 包里那个**同名**（一个回答
# "整份文档怎么交给知识库"、一个回答"片段怎么交给知识库"），这里显式改名——
# 两张表都叫"记录形状"，混用会让接口返回一份看起来合理、实际少了一层的数据。
from smart_research_agent.chunking import (
    ATOMIC_KINDS,
    CHUNKING_LIMITATIONS,
    CHUNKING_OUT_OF_SCOPE,
    STRATEGIES,
    STRATEGY_SEMANTIC,
    STRATEGY_STRUCTURAL,
    ChunkPipeline,
    ChunkPolicy,
    ChunkReport,
    ChunkingError,
    RetrievalProbe,
    chunking_boundaries,
    chunking_plan,
    default_policy,
    default_registry,
    describe_measurers,
    evaluate_strategies,
)
from smart_research_agent.chunking import (
    knowledge_record_shape as chunk_record_shape,
)

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
# day066：/retrieval/answer 需要认出"未配置 LLM"时的占位实现。
#
# 为什么认类而不是认配置：create_app 在缺密钥时会退避成 MockLLM
# （见 app.default_llm 的告警原文），而"这条链路实际拿到的是哪个对象"
# 才是判断依据——注入路径可以绕过 settings，那时看配置会得出相反的结论。
from smart_research_agent.llm.mock import MockLLM
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

# day064 向量库（M6-D3）：/vectorstore/* 六个端点消费的公开接口。
#
# 注意这一段的三个"为什么不"：
# - **不在端点里 import 任何可选后端**（faiss / chromadb）：它们的可用性是
#   调用期事实，注册表已经给出了"缺什么 / 怎么装 / 还能用什么"的三段式报错
#   （见 registry 模块 docstring），端点只负责把结论折进响应体；
# - **不自己算分数**：排序口径（越大越近）与过滤（先筛后排）都只有一处实现，
#   端点重写一遍就会得到"接口与脚本给出的 top-1 不一样"这类无法归因的差异；
# - **不缓存任何东西**：本组端点回答的是"现在这个库是什么状态"，
#   一个缓存的 state 会让 stats 在写入之后仍然回显旧值。
from smart_research_agent.vectorstore import (
    METRIC_INNER_PRODUCT,
    RECORD_ID_FIELD,
    VECTORSTORE_LIMITATIONS,
    VECTORSTORE_OUT_OF_SCOPE,
    BackendUnavailable,
    FilterError,
    RecordError,
    VectorBackend,
    VectorError,
    VectorIngestPipeline,
    VectorRecord,
    VectorStoreError,
    backend_names,
    compare_backends,
    describe_backends,
    describe_filter,
    describe_metrics,
    embedding_text,
    filter_fields,
    normalize_metric,
    record_from_knowledge,
    record_id_of,
    resolve_backend,
    verify_parity,
)

# day065 索引构建（M6-D4）：/indexing/* 七个端点消费的公开接口。
#
# 与 day064 那一段同一条纪律（依赖全走 app.state 注入），但有两处必须说清：
# - **本组有一个真写端点**（``/indexing/build``），它改的是注入的那个向量库，
#   外加版本表与（可选的）备份目录。因此它不自己造后端，也不自己造编码器：
#   两者都必须与 ``/vectorstore/*`` 是同一对实例，否则"写进库的向量"与
#   "按清单算出的向量键"会是两套口径；
# - **只读端点一个字节都不该写**（``/indexing/plan`` 最典型）：计划是"看"，
#   不是"做"。它读的上一版来自版本表，写完不会动库、不会动版本表。
from smart_research_agent.indexing import (
    BUILD_MODE_FULL,
    BUILD_MODE_INCREMENTAL,
    BUILD_MODES,
    BackupError,
    IndexBackupStore,
    IndexBuilder,
    IndexVersionStore,
    IndexingError,
    IndexingPipeline,
    IndexManifest,
    IndexPlan,
    VersionError,
    manifest_from_store,
    # 显式改名：``registry`` 那边已经有一个同名函数（见上面第 417 行的导入），
    # 直接 import 会把那个绑定**就地覆盖**——1912 / 2088 行两处注册表端点的
    # ``created_at`` 会悄悄改用本包的实现。两个函数今天行为相同，
    # 但"今天相同"不是让一个 import 静默换掉另一个实现绑定的理由。
    utc_now_iso as indexing_utc_now,
    verify_index,
)

# day066 检索（M6-D5）：/retrieval/* 五个端点消费的公开接口。
#
# 与上面两组同一条纪律（依赖全走 app.state 注入），但有三处必须说清：
# - **本段在导入期不构造任何东西**：没有模块级的后端、编码器、检索器或路由器。
#   路由器与检索器在**请求期**用 ``app.state.vector_store`` 与
#   ``app.state.embedding`` 现装（见 ``retrieval_router``），因此
#   ``import api.routes`` 不读盘、不联网、不建库——"缺 faiss 的机器上
#   import 就炸"这类事故的前提（导入期构造后端）在这里不存在；
# - **不在端点里做检索决策**：深度、阈值、多样性、空结果诊断只有一处实现
#   （``retrieval`` 包），端点只把请求体折成 ``RetrievalQuery`` 再序列化结果。
#   端点上重算一遍深度会产出"接口与脚本给出的 top-1 不一样"这种差异，
#   而它没有任何报错指向病因；
# - **多索引路由是业务决策**：缺省只有一个路由（注入的那一个库），要多个请
#   注入一个 ``StoreRouter``（挂 ``app.state.retrieval_router``）。让端点替
#   调用方挑一个库，会给出"问源码库的问题去查了手册库"这种看起来完全正常的错答。
from smart_research_agent.retrieval import (
    EMPTY_REASON_DESCRIPTIONS,
    MAX_FETCH_K,
    RETRIEVAL_LIMITATIONS,
    RETRIEVAL_OUT_OF_SCOPE,
    BM25Params,
    HybridRetriever,
    LexicalIndex,
    RagPipeline,
    RetrievalError,
    RetrievalQuery,
    RetrievalResult,
    Retriever,
    RouteDecision,
    StoreRouter,
    TimeRange,
    build_hybrid_retriever,
    build_retriever,
    describe_conditions,
    hybrid_enabled,
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


# --------------------------------------------------------------------------- #
# day061 文档解析与加载（M6-D1）：/documents/* 四个端点
#
# 四个端点全部**只读或纯计算**：不写磁盘、不联网、**不遍历目录**、不留状态。
# 目录遍历与批量入库由 ``scripts/documents_demo.py`` 承担——
# **一个要读整个目录的流程不该挂在一个 HTTP 请求上**（与 day059/060 同源）。
#
# 这一组与 day060 的 /serving/* 有一个共同点值得指出：两者的输入都是
# "调用方带来的事实"（那里是端点自述与实测延迟，这里是文件字节），
# 服务端只负责**判定与归一化**，因此同一个请求体永远得到同一个响应。
# --------------------------------------------------------------------------- #


def document_bytes(payload: dict) -> bytes:
    """从请求体里取出文件字节（``content_base64`` 与 ``text`` 二选一）.

    两条规则，都是为了让"传错了"能立刻看出来：

    - base64 解不开 → **400**（而不是静默当成空内容）；
    - 两个字段都没给 → **400**（"空文件"与"忘了填"是两件事，
      而把它们混起来会让一份空文档进库）。

    注意**空文件是合法的**（``content_base64=""`` 且 ``text=""`` 会走到上面
    第二条并被拒），这是刻意的：一份真正的空文件在入库时只会贡献一个
    "段落数为 0"的文档，而它更可能是"上游漏传了内容"。
    """
    raw = str(payload.get("content_base64", "") or "")
    text = str(payload.get("text", "") or "")
    if raw:
        try:
            return base64.b64decode(raw, validate=True)
        except ValueError as exc:
            # ``binascii.Error`` 继承自 ``ValueError``，因此这里捕一次就够——
            # 写两个更"精确"的异常类型反而会在 Python 版本之间漂移。
            raise DocumentError(f"content_base64 不是合法的 base64：{exc}") from exc
    if text:
        return text.encode("utf-8")
    raise DocumentError("content_base64 与 text 必须给出一个（空内容无法判定是漏传还是真的为空）")


def document_source_name(filename: str) -> str:
    """来源标签：文件名留空时用一个**显式的占位名**，而不是空串.

    ``Document.source`` 不允许为空（它是引用溯源的落点），而"没给文件名"
    是一个常见情形（内容从接口直接传进来）。用 ``<inline>`` 作占位，
    比让请求失败或让它悄悄变成空串都更好——**它明确说出了"这份内容
    没有文件名"这件事**。
    """
    return filename or "<inline>"


@router.get("/documents/loaders", response_model=DocumentsLoadersResponse)
def documents_loaders() -> DocumentsLoadersResponse:
    """给出解析能力的自我描述：五个加载器、编码链、六个格式的边界（day061）.

    这个端点**不读磁盘、不解析任何文件**，它回答开工前的五个问题：

    1. **哪些格式有加载器**（``loaders``）：表里的 ``suffixes`` 一列
       就是"哪些文件会被认出来"；
    2. **编码是怎么定的**（``encoding_chain``）：五步固定顺序，
       含"哪一步为什么会排在那里"；
    3. **每个格式取哪些结构**（``markdown`` / ``html`` / ``docx``）；
    4. **每个格式做不到什么**（``pdf.not_supported`` 与
       ``docx.parts_not_read``）：**它比"能做什么"更值得在开工前读一遍**；
    5. **归一化后的形状**（``block_kinds`` + ``knowledge_record``）：
       day062 的分块器就在这个形状上工作，day009 的知识库直接消费它。

    最后附上"不能做什么"：文档里那句"PDF 抽取不解析 CID 字形映射"
    与"DOCX 读不到页眉页脚"是本日最重要的两条边界。
    """
    registry = DocumentIngestor().registry
    return DocumentsLoadersResponse(
        media_types=registry.media_types,
        loaders=registry.table(),
        encoding_chain=encoding_chain(),
        markdown=markdown_syntax_table(),
        html=html_rules(),
        pdf=pdf_boundaries(),
        docx=docx_scope(),
        block_kinds=list(BLOCK_KINDS),
        knowledge_record=knowledge_record_shape(),
        max_file_mib=settings.documents_max_file_mib,
        out_of_scope=list(DOCUMENTS_OUT_OF_SCOPE),
        limitations=list(DOCUMENTS_LIMITATIONS),
    )


@router.post("/documents/detect", response_model=DocumentDetectResponse)
def documents_detect(req: DocumentDetectRequest) -> DocumentDetectResponse:
    """只做类型识别与编码探测，**不解析结构**（day061）.

    两条产物各自回答一个问题：

    - ``media_type`` + ``decided_by`` + ``conflict``：这份文件是什么，
      以及**这个判断是怎么来的**（内容优先，后缀与内容冲突时报冲突）；
    - ``encoding`` + ``encoding_confident``：正文解码用了哪种编码，
      以及**这个判断可不可信**。``encoding_confident=false`` 说明已经
      回退到 latin-1，中文多半是乱码——**它必须被看见**，
      因为乱码会一路通过分块、向量化与检索。

    错误码：空内容 → **400**；base64 非法 → **400**；
    一份二进制文件（不是文本也不是 PDF/docx）会走到这里而不报错——
    "它是二进制"是一个结论，在 ``/documents/parse`` 里才会被拒收。
    """
    try:
        data = document_bytes(req.model_dump())
        detection: MediaTypeDetection = detect_media_type(req.filename, data)
        registry = DocumentIngestor().registry
        supported = registry.supports(detection.media_type)
        encoding = ""
        decided_by = ""
        confident = True
        if detection.media_type != MEDIA_PDF and not data.startswith(b"PK\x03\x04"):
            decoded = decode_bytes(data, source=req.filename)
            encoding = decoded.encoding
            decided_by = decoded.decided_by
            confident = decoded.confident
    except (DocumentError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return DocumentDetectResponse(
        summary=(
            f"{document_source_name(req.filename)} → {detection.summary_line()}"
            + (f" | 编码 {encoding}（{decided_by}）" if encoding else " | （二进制，不做编码探测）")
        ),
        media_type=detection.media_type,
        decided_by=detection.decided_by,
        suffix_type=detection.suffix_type,
        magic_type=detection.magic_type,
        conflict=detection.conflict,
        supported=supported,
        encoding=encoding,
        encoding_decided_by=decided_by,
        encoding_confident=confident,
        size_bytes=len(data),
    )


@router.post("/documents/parse", response_model=DocumentParseResponse)
def documents_parse(req: DocumentParseRequest) -> DocumentParseResponse:
    """解析一份文档，返回归一化后的结构与内容指纹（day061）.

    ``document.doc_id`` 是**内容指纹**：同一份内容从两个文件名传进来会得到
    同一个 id。这条性质有两个直接用途——**批量入库去重**，以及
    "这份文档三个月前入过库吗"这种问题可以靠 id 直接回答。

    ``document.blocks`` 是六类结构块（标题 / 段落 / 代码 / 列表项 / 表格 / 引用）。
    day062 的分块器就以它为单位切分，因此``include_blocks=False``
    只返回统计量——一份几百页的文档，其块序列会比全文还大。

    错误码：未知媒体类型（没有加载器）→ **400**；
    不被支持的内容（加密 PDF、缺 ``document.xml``、二进制、BOM 与内容矛盾）
    → **400**（错误信息里写明是"不在支持范围"而不是"出错了"）；
    ``filename`` 与 ``content_base64`` 都为空 → **400**。
    """
    try:
        data = document_bytes(req.model_dump())
        source = document_source_name(req.filename)
        registry = DocumentIngestor().registry
        document = registry.load_bytes(data, source=source, media_type=req.media_type)
        detection = (
            MediaTypeDetection(media_type=req.media_type, decided_by="given")
            if req.media_type
            else detect_media_type(req.filename, data)
        )
    except (DocumentError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return DocumentParseResponse(
        summary=document.summary_line(),
        document=document.to_dict(include_blocks=req.include_blocks),
        detection=detection.to_dict(),
    )


@router.post("/documents/ingest", response_model=DocumentIngestResponse)
def documents_ingest(req: DocumentIngestRequest) -> DocumentIngestResponse:
    """把一批文件入库：内容指纹去重 + 三类状态 + 可复核的报告（day061）.

    三件事在响应里各有位置，**不混在一起**：

    - ``report.status_counts``：``ok`` / ``duplicate`` / ``error`` 三个键恒存在。
      **重复单独一态**——"这一批里有几份是重复内容"是一个数据治理问题，
      不是一个失败问题；
    - ``report.entries``：逐条结果。成功条目带 ``doc_id`` 与字符数，
      重复条目带 ``duplicate_of``（指回它重复的那一份），
      失败条目带 ``error`` 的原文；
    - ``report.block_counts``：六类块的汇总（**取值为 0 的也在**，
      否则两张报告没法逐列对比）。

    去重的优先级就是**清单顺序**：先出现的留下。因此同一个请求体在任何
    机器上得到同一个响应（服务端不做排序，也不遍历文件系统）。

    错误码：某个文件为空 / base64 非法 → **400**（整批不处理）；
    某个文件"不在支持范围" → **记进报告**（整批继续，这是两件事）。
    """
    try:
        payloads = [
            {"filename": item.filename, "content_base64": item.content_base64, "text": item.text}
            for item in req.files
        ]
        items = [
            (document_source_name(str(payload["filename"])), document_bytes(payload))
            for payload in payloads
        ]
        report = DocumentIngestor().ingest_bytes(items)
    except (DocumentError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return DocumentIngestResponse(
        summary=report.summary_line(),
        report=report.to_dict(include_documents=True),
        markdown=report.render_markdown(),
    )


# --------------------------------------------------------------------------- #
# day062 分块策略（M6-D2）：/chunking/* 四个端点
#
# 与上面 /documents/* 同一纪律：**只读或纯计算**——不写磁盘、不联网、
# 不遍历目录、不留状态。批量分块与目录级重切由 ``scripts/chunking_demo.py``
# 承担：**一个要读整个目录的流程不该挂在一个 HTTP 请求上**。
#
# 四个端点的输入都是"调用方带来的字节 + 参数"，因此同一个请求体
# 永远得到同一个响应（分块参数里唯一的例外是 ``measurer="tiktoken"``，
# 它依赖本机词表缓存——那一点在响应里会以 ``token_measurer`` 的形式显式返回）。
# --------------------------------------------------------------------------- #


def chunking_policy_from(payload: dict, strategy: str) -> ChunkPolicy:
    """请求体 + settings → 一份策略参数（三级优先级）.

    ```text
    请求体   >   settings（项目级基线）   >   策略默认值（DEFAULT_POLICIES）
    ```

    为什么需要"项目级基线"这一层：``settings.chunking_*`` 表达的是
    **这个项目当前选定的口径**（例如"我们统一按 512 字符切"），
    而 ``DEFAULT_POLICIES`` 是**这种策略的合理起点**。两者会在真实项目里
    分叉，因此不能只留一层。

    但有两项刻意**不接受项目级覆盖**：

    - ``overlap_tokens``：结构策略的重叠必须为 0，那是**结构性判断**
      （标题边界就是语义边界），不该被一个全局默认值改掉——真改了会直接
      报错，而报错信息里的建议会是"调小 overlap"，把人引到错误的方向；
    - ``similarity_percentile``：只有语义策略消费它，其余策略拿到它也只是
      原样带在产物里（见 ``ChunkPolicy`` 的字段注释）。

    有一条与"顺序与比例"有关的规则：``overlap_tokens`` 与 ``min_tokens``
    都在 ``max_tokens`` **定下来之后**才决定，而且**按比例**而不是按绝对值：

    ```text
    overlap = round(max_tokens × 比例)      比例 = settings.overlap / settings.max_tokens
    min_tokens = min(settings.min_tokens, max_tokens // 2)
    ```

    为什么必须是比例：``48`` 这个数只在"预算 = 320"时有意义，它是
    ``320 × 15%``。把 48 写死会让"预算调成 64"变成"75% 的重叠"——
    块一下子放大 3 倍，而**参数看起来只是变小了**。这与 day058 的
    "版本键按内容算"、day060 的"成本按实际用量算"是同一种修正：
    一个绝对数只有在它的基准不变时才成立。

    ``min_tokens`` 取"项目下限与预算一半的较小值"：它本来就是一个
    合并门限，门限高于预算时合并必然超预算（``ChunkPolicy`` 会直接报错），
    而那种报错会被理解成"框架不许我用小预算"，是错的归因。

    最终生效的参数原样出现在响应的 ``stats.metadata.policy`` 里，
    因此"到底用了多少"随时可以核对——**可见的回退不是静默。**
    """
    base = default_policy(strategy)
    overrides: dict = {
        "max_tokens": settings.chunking_max_tokens,
        "measurer": settings.chunking_measurer,
    }
    if strategy == STRATEGY_SEMANTIC:
        overrides["similarity_percentile"] = settings.chunking_similarity_percentile
    if payload.get("max_tokens"):
        overrides["max_tokens"] = int(payload["max_tokens"])
    explicit_overlap = int(payload.get("overlap_tokens", -1)) >= 0
    if base.overlap_tokens > 0 and not explicit_overlap:
        ratio = (
            settings.chunking_overlap_tokens / settings.chunking_max_tokens
            if settings.chunking_max_tokens > 0
            else 0.15
        )
        overrides["overlap_tokens"] = max(0, round(overrides["max_tokens"] * ratio))
    if explicit_overlap:
        overrides["overlap_tokens"] = int(payload["overlap_tokens"])
    if int(payload.get("min_tokens", -1)) < 0:
        overrides["min_tokens"] = min(
            settings.chunking_min_tokens, overrides["max_tokens"] // 2
        )
    else:
        overrides["min_tokens"] = int(payload["min_tokens"])
    if payload.get("measurer"):
        overrides["measurer"] = str(payload["measurer"])
    if float(payload.get("similarity_percentile", -1.0)) >= 0:
        overrides["similarity_percentile"] = float(payload["similarity_percentile"])
    return default_policy(strategy, **overrides)


def chunking_document(payload: dict, filename: str, media_type: str) -> Document:
    """请求体 → 归一化文档（复用 /documents/* 的两个助手，语义完全一致）.

    直接复用 ``document_bytes`` 与 ``document_source_name``，而不是各写一份：
    **"空内容算漏传还是真的为空"这种判断只应该有一个答案**，
    两处各写一遍迟早会在某次修改后分叉（day061 的取舍）。
    """
    data = document_bytes(payload)
    source = document_source_name(filename)
    return DocumentIngestor().registry.load_bytes(
        data, source=source, media_type=media_type
    )


def chunking_pipeline(request: Request) -> ChunkPipeline:
    """分块编排器：度量器按策略、embedding 取 app.state 的注入值.

    embedding 用注入而不是 ``default_embedding()``：与 ``/embeddings``
    走同一个提供方，测试注入 MockEmbedding 就能让语义策略全离线。
    """
    return ChunkPipeline(embedding=request.app.state.embedding)


@router.get("/chunking/strategies", response_model=ChunkingStrategiesResponse)
def chunking_strategies(request: Request) -> ChunkingStrategiesResponse:
    """给出分块能力的自我描述：四种策略、度量器、默认参数与边界（day062）.

    这个端点**不解析任何文档、不调用 embedding**，它回答开工前的四个问题：

    1. **四种策略分别在哪里切**（``strategies``）：自述表里的
       ``where_to_cut`` 一列就是答案，``strengths`` / ``weaknesses``
       则是选型的直接依据；
    2. **预算单位是谁**（``measurers``）：``deterministic`` 一列决定了
       这批块的 ``chunk_id`` 能不能跨机器对齐，而 chunk_id 是去重键；
    3. **默认参数是多少**（``default_policies``）：注意它是**按策略**给的，
       结构策略的重叠为 0——"这个参数对我没意义"和"它默认为 0"是两件事；
    4. **交出去的是什么形状**（``knowledge_record``）：与
       ``/documents/loaders`` 里那份的区别只有一层——那里是整份文档，
       这里是片段，而字段名不变。

    最后附上"不能做什么"与"块与原文之间的不变量"。
    """
    boundaries = chunking_boundaries()
    return ChunkingStrategiesResponse(
        strategies=default_registry(embedding=request.app.state.embedding).table(),
        measurers=describe_measurers(),
        default_policies=boundaries["default_policies"],
        separators=boundaries["separators"],
        atomic_kinds=list(ATOMIC_KINDS),
        knowledge_record=chunk_record_shape(),
        coverage_invariant=boundaries["coverage_invariant"],
        defaults={
            "strategy": settings.chunking_strategy,
            "max_tokens": settings.chunking_max_tokens,
            "overlap_tokens": settings.chunking_overlap_tokens,
            "min_tokens": settings.chunking_min_tokens,
            "measurer": settings.chunking_measurer,
            "similarity_percentile": settings.chunking_similarity_percentile,
            "eval_top_k": settings.chunking_eval_top_k,
        },
        out_of_scope=list(CHUNKING_OUT_OF_SCOPE),
        limitations=list(CHUNKING_LIMITATIONS),
    )


@router.post("/chunking/split", response_model=ChunkingSplitResponse)
def chunking_split(req: ChunkingSplitRequest, request: Request) -> ChunkingSplitResponse:
    """把一份文档切成块：先给代价估算，再给实测统计与明细（day062）.

    三块内容各自回答一个问题，**不混在一起**：

    - ``plan``：参数一确定就能算出来的代价（块数上界、重叠放大量）。
      **先算再切**——调 ``max_tokens`` 时不必等一次完整的分块跑完；
    - ``stats``：实测统计。三列最值得看：``coverage``（有没有切丢内容）、
      ``duplication_ratio``（重叠白花了多少）、``oversized_count``
      （有多少块注定被模型截断）；
    - ``chunks``：前 ``top_n`` 块的明细，``start_char:end_char``
      可以直接回到原文核对——**"块文本是原文的连续子串"这条不变量，
      在这个字段上是肉眼可验的。**

    错误码：文档为空 / base64 非法 / 不在支持范围 → **400**（来自 day061）；
    参数矛盾（``overlap`` 不小于 ``max_tokens``、未知策略、未知度量器）→ **400**；
    请求字段类型不对 → **422**。

    一个刻意不做的事：**不递归遍历目录**。要看整个知识库切完的样子，
    用 ``scripts/chunking_demo.py``——它会把报告写下来。
    """
    try:
        document = chunking_document(
            req.model_dump(), req.filename, req.media_type
        )
        strategy = req.strategy or settings.chunking_strategy
        policy = chunking_policy_from(req.model_dump(), strategy)
        chunk_set = chunking_pipeline(request).chunk(document, strategy, policy)
    except (ChunkingError, DocumentError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    top_n = req.top_n if req.top_n > 0 else chunk_set.count
    return ChunkingSplitResponse(
        summary=chunk_set.summary_line(),
        plan=chunking_plan(len(document.text), policy),
        stats=chunk_set.to_dict(include_text=False),
        chunks=[
            chunk.to_dict(include_text=req.include_text)
            for chunk in chunk_set.chunks[:top_n]
        ],
        document=document.to_dict(include_blocks=False),
    )


@router.post("/chunking/evaluate", response_model=ChunkingEvaluateResponse)
def chunking_evaluate(
    req: ChunkingEvaluateRequest, request: Request
) -> ChunkingEvaluateResponse:
    """用一组探针给多个分块策略打分（day062）：hit@k / MRR / 命中块均长.

    判定用的是**块文本里是否含期望片段**，于是它同时测了两件事：
    检索能不能把对的块召回来，以及**期望片段有没有被切碎**——
    后者用"平均块长"是永远看不出来的（见 ``chunking/evaluate.py``）。

    两条纪律写在这里，因为它们决定了这份分数能不能用：

    - ``probes`` 必须来自**你自己的查询分布**，抄别人的最优策略只对别人有效；
    - 期望片段在原文里必须唯一。出现多次的探针会让分数**虚高且毫不显眼**，
      因此它们被单独列在 ``evaluation.ambiguous_probes`` 里，不混进总分。

    ``markdown`` 是同一份结果的渲染版，可以直接贴进选型记录。

    错误码：``probes`` 为空 → **400**；探针缺 ``question`` 或 ``expect`` → **400**；
    未知策略 / 参数矛盾 → **400**。
    """
    if not req.probes:
        raise HTTPException(
            status_code=400,
            detail="probes 不能为空：没有期望片段就无法判定命中，"
            "而「人工看一眼觉得还行」不是一条可复现的判据",
        )
    try:
        document = chunking_document(
            req.model_dump(), req.filename, req.media_type
        )
        probes = [
            RetrievalProbe(question=item.question, expect=item.expect, note=item.note)
            for item in req.probes
        ]
        strategies = list(req.strategies) or None
        policies = (
            {
                strategy: chunking_policy_from(
                    {"max_tokens": req.max_tokens}, strategy
                )
                for strategy in (strategies or list(STRATEGIES))
            }
            if req.max_tokens
            else None
        )
        evaluation = evaluate_strategies(
            document,
            probes,
            strategies=strategies,
            policies=policies,
            top_k=req.top_k or settings.chunking_eval_top_k,
            embedding=request.app.state.embedding,
        )
    except (ChunkingError, DocumentError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ChunkingEvaluateResponse(
        summary=evaluation.summary_line(),
        evaluation=evaluation.to_dict(),
        markdown=evaluation.render_markdown(),
    )


@router.post("/chunking/records", response_model=ChunkingRecordsResponse)
def chunking_records(
    req: ChunkingRecordsRequest, request: Request
) -> ChunkingRecordsResponse:
    """把文档切成**可直接入库的记录**：day061 的下一环（day062）.

    这一步是 M6 里最容易被忽略、实际上最关键的一次"形状交接"：

    ```text
    day061  {"doc_id": <doc_id>,   "text": <全文>,  …}   ← 整份文档
    day062  {"doc_id": <chunk_id>, "text": <片段>,  …}   ← 片段，字段名不变
    ```

    三个必须说清的字段取舍：

    1. ``doc_id`` 位置放的是 ``chunk_id``——知识库里的唯一键就是片段；
    2. ``text`` 是**原文片段**（可在 ``document.text`` 里逐字核对），
       因为库里存的应当是可以直接引用给用户的原文；
    3. 向量化用的是 ``metadata.retrieval_text``（带标题面包屑），
       因此"存的"和"embed 的"不是同一段文本。``embedding_input``
       这个字段就是为这一条准备的——**不把它写出来，
       下一个人会以为向量是用 ``text`` 算的，然后"修"成一个 bug。**

    错误码：文档为空 / 不在支持范围 → **400**；未知策略 / 参数矛盾 → **400**。
    """
    try:
        document = chunking_document(
            req.model_dump(), req.filename, req.media_type
        )
        strategy = req.strategy or settings.chunking_strategy
        policy = chunking_policy_from(req.model_dump(), strategy)
        chunk_set = chunking_pipeline(request).chunk(document, strategy, policy)
    except (ChunkingError, DocumentError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    report = ChunkReport(
        sets=(chunk_set,),
        metadata={"strategy": strategy, "doc_id": document.fingerprint},
    )
    records = report.knowledge_records()
    return ChunkingRecordsResponse(
        summary=report.summary_line(),
        report=report.to_dict(include_text=False),
        records=records[: req.limit] if req.limit > 0 else records,
        embedding_input="metadata.retrieval_text",
        markdown=report.render_markdown(),
    )


# --------------------------------------------------------------------------- #
# day064 向量库（M6-D3）：/vectorstore/* 六个端点
#
# 与上面 /chunking/* 同一套注入纪律（编码器取 ``app.state.embedding``），
# 但本组有两处必须显式说清的差别：
#
# 1. **本组有两个写端点**（upsert 与 records 删除），它们改的是内存状态。
#    因此后端在 ``app.state.vector_store``、最近一次写入报告在
#    ``app.state.vectorstore_last_ingest``（常量 ``LAST_INGEST_STATE_KEY``），
#    两者都挂在**应用实例**上，**不写 settings、不写模块级全局**——
#    挂上模块全局，"同一个进程里另建一个 app 看到的是一份空库"这条性质
#    会静默失效，而它的表现只是另一个测试偶发失败。
# 2. **可选后端缺席时降级成一行报告，而不是 500**：``/vectorstore/compare``
#    的全部价值就在于"同一份请求在任何机器上返回行数相同的报告，只差一个布尔"
#    （折行逻辑在 ``evaluate.compare_backends`` 里，端点只管转发）。
# --------------------------------------------------------------------------- #

#: 最近一次写入报告挂在 ``app.state`` 上的键名（app.create_app 里初始化成 None）.
#: 抽成常量是因为它被两处使用（app.py 初始化、下面读写），
#: 而"同一个字符串写两遍"正是那种改了一处、另一处继续用旧名字的经典来源。
LAST_INGEST_STATE_KEY = "vectorstore_last_ingest"


def vectorstore_pipeline(request: Request) -> VectorIngestPipeline:
    """摄取 / 检索编排器：后端与编码器都取 ``app.state`` 的注入值.

    为什么两个依赖都走注入而不是在这里 ``create_backend()`` / ``default_embedding()``：
    - 后端：测试注入 ``FlatVectorStore(dimension=4)`` 就能精确造出"库与编码器
      不是同一套"的场景（那条路径对应 HTTP 409）；
    - 编码器：与 ``/embeddings``、``/chunking/*`` 用的是**同一个** ``app.state.embedding``
      实例。查询侧与写入侧一旦用上不同的编码器，检索结果会以"排序不太对"
      的形式表现出来，而不报任何错。

    本函数**不缓存**编排器：它只持有两个引用，构造是 O(1)，
    而缓存会让"注入的后端被换掉"这件事在某个请求上悄悄失效。
    """
    return VectorIngestPipeline(
        request.app.state.vector_store, request.app.state.embedding
    )


def vectorstore_backend_factory(name: str) -> Callable[..., VectorBackend]:
    """把一个后端名收成 ``(*, metric, dimension, path)`` 形式的小工厂.

    为什么是工厂而不是实例：``VectorBackend`` 的度量在构造时定死
    （见 ``base.metric`` 的说明——运行期换度量会让"换个度量再搜一次"
    变成一次静默的结果错乱），因此"同一个后端跑三种度量"只能各建一个实例。
    ``compare_backends`` 要求的工厂签名恰好就是这三个关键字。

    ``path`` 默认空串 = **明确不落盘**：对账是内存里的一次计算，
    让它在磁盘上留下三个快照，只会给下一次"这个目录怎么多了几个文件"增加一个谜。
    """
    def factory(*, metric: str, dimension: int, path: str = "") -> VectorBackend:
        return resolve_backend(name, metric=metric, dimension=dimension, path=path)

    return factory


def vectorstore_compare_records(
    request: Request, raw_records: list[dict]
) -> list[VectorRecord]:
    """把知识库记录逐条编码成 ``VectorRecord``（compare 用的一次性转换）.

    三处刻意对齐 / 刻意不同的地方：

    - **取值路径与 ``VectorIngestPipeline.ingest`` 完全相同**：id 取 ``doc_id``、
      编码文本由 ``embedding_text`` 决定（优先 ``metadata.retrieval_text``）。
      两处各写一遍"该编码哪个字段"迟早会分叉，而分叉的症状是
      "端点里的检索结果与脚本里的不一样"，这种差异没有任何报错指向病因；
    - **不在这里归一化**：``metric=METRIC_INNER_PRODUCT`` 是本包里"不归一化"
      的那一支（``types.make_record`` **只为 cosine** 归一化）。于是每个后端在
      ``_prepare`` 里按**自己的**度量归一化——这正是"同一个库换个度量会换答案"
      能被看见的原因。若在这里先归一化，ip 与 cosine 的两行会永远相同，
      而报告看起来完全正常（见 ``evaluate`` 模块 docstring）；
    - **一条不合格的记录直接报错，而不是跳过**：对账的产物是 ``agreed`` /
      ``overlap`` 这两个"逐位对齐"的数字，少一条记录会让两边比较的深度
      悄悄变浅——报告上只是"一致度低了一点"，没有任何迹象指向"输入少了一条"。
      报错类型选 ``RecordError``（而不是 HTTPException），是为了让"记录不合法"
      与"请求形状不对"在端点里仍然走**同一条** 400 通道。
    """
    prepared: list[VectorRecord] = []
    for index, raw in enumerate(raw_records):
        record_id = record_id_of(raw)
        if not record_id:
            raise RecordError(
                f"第 {index + 1} 条记录缺少 {RECORD_ID_FIELD!r}（知识库记录的唯一键"
                "就是 chunk_id）：对账需要一份完整的输入，缺一条会让 agreeing 的"
                "深度悄悄变浅，因此这里直接拒绝而不是跳过。"
            )
        text = embedding_text(raw)
        if not text.strip():
            raise RecordError(
                f"第 {index + 1} 条记录的 text 与 retrieval_text 都是空白："
                "编码一段空文本没有意义（零向量会被向量库直接拒收）。"
            )
        prepared.append(
            record_from_knowledge(
                raw,
                request.app.state.embedding.embed(text),
                metric=METRIC_INNER_PRODUCT,
            )
        )
    return prepared


def vectorstore_compare_metrics(metrics: list[str]) -> list[str]:
    """把请求里的度量收敛成规范名；留空表示用 ``settings.vector_metric``.

    ``normalize_metric`` 对未知名字抛 ``FilterError`` 并列出可选值与别名——
    与 day062 的 ``resolve_measurer`` 同一条纪律：**宁可报错，也不要挑一个
    "看起来差不多"的度量**（静默把 ``euclidean2`` 当成默认的 cosine，
    症状只会是"排序不太对"）。
    """
    requested = list(metrics) or [settings.vector_metric]
    return [normalize_metric(name) for name in requested]


@router.get("/vectorstore/backends", response_model=VectorStoreBackendsResponse)
def vectorstore_backends(request: Request) -> VectorStoreBackendsResponse:
    """三张表回答"能用什么、口径是什么、现在用的是哪个"（day064）.

    这个端点**不 import 任何可选后端、不碰库里的数据**，因此它在缺
    faiss/chromadb 的机器上照样可用：

    - ``backends`` ← ``describe_backends()``：每个后端的依赖、能否落盘、
      元数据在不在同一个库里、缺依赖时敲哪条命令、放弃的代价是什么；
      ``available`` 与 ``missing`` **一起给**——只给一个布尔值，
      使用者看到 ``false`` 还得自己猜缺的是哪一个包；
    - ``metrics`` ← ``describe_metrics()``：三种度量的口径（越大越近、
      要不要归一化、值域），它与 ``docs/vector_store.md`` 的距离对照表同源；
    - ``current_*``：本实例实际用的后端 / 度量 / 条数。**能力表与当前选用
      必须分开**，否则"缺 faiss 时 flat 顶上了"会被读成"我配的就是 flat"。

    它**不做什么**：不选后端（那是部署时的决定）、不落盘、不改库、
    不给"该选哪个"的结论（选型要看数据规模与运维条件，见手册的决策表）。
    """
    store = request.app.state.vector_store
    return VectorStoreBackendsResponse(
        backends=describe_backends(),
        metrics=describe_metrics(),
        current_backend=store.name,
        current_metric=store.metric,
        current_count=store.count(),
        default_backend=settings.vector_backend,
        default_metric=settings.vector_metric,
        default_top_k=settings.vector_default_top_k,
        min_score=settings.vector_min_score,
        persistence_path=settings.vector_persist_path,
        limitations=list(VECTORSTORE_LIMITATIONS),
        out_of_scope=list(VECTORSTORE_OUT_OF_SCOPE),
    )


@router.post("/vectorstore/upsert", response_model=VectorUpsertResponse)
def vectorstore_upsert(
    req: VectorUpsertRequest, request: Request
) -> VectorUpsertResponse:
    """把一批 day062 形状的记录编码后写进库，返回一份 ``IngestReport``（day064）.

    它补的是最后一道缝：``/chunking/records`` 交出的"可以直接入库的记录"
    此前只有脚本能入库。三件事因此第一次出现在 HTTP 响应里：

    - ``embedding_calls``：本次**实际**调用编码器的次数。本课是逐条编码
      （刻意的选择，见 ``vectorstore/pipeline.py`` 的模块 docstring），
      因此它等于被编码的条数——day065 的批量 + 缓存省下多少次编码，
      就是要跟这个基准比才说得清；
    - ``unchanged``：重放同一批数据时它等于条数，而 ``written`` 为 0。
      **这是"这次写入其实什么都没做"的唯一证据**；
    - ``skipped`` 与 ``failed`` 分开：前者是数据问题（缺 id、文本全空白），
      后者是环境与匹配问题（维度不一致、后端整批拒绝）。
      分开之后报告自己就指出了该找谁修。

    **内存后端的写入只活在这个应用实例里**：进程退出即消失，
    同一个进程里另建的 app 实例看到的是空库（测试里有一条专门钉这个）。
    最近的这次报告记进 ``app.state.vectorstore_last_ingest``，
    ``GET /vectorstore/stats`` 会把它带回来。

    错误码：``records`` 为空 → **400**（一次什么都没写的 upsert 只会刷新状态，
    要探测库的现状请用 stats）；记录形状不对 → **422**；
    编码器或后端抛出的 ``VectorStoreError``（维度不一致、整批被拒）→ **400**
    ——注意编码器自身抛出的**其它**异常一律上抛成 500：报告只能收留
    我们认识的失败，不认识的失败必须让调用栈把它喊出来。

    它**不做什么**：不落盘（缺省后端是内存 flat 且 ``vector_persist_path``
    为空串）、不做批量编码优化、不重建索引、不决定"该不该重建"。
    """
    if not req.records:
        raise HTTPException(
            status_code=400,
            detail="records 不能为空：一次什么都没写的 upsert 只会刷新状态而不是"
            "改变库；要探测库的现状请用 GET /vectorstore/stats。",
        )
    pipeline = vectorstore_pipeline(request)
    try:
        report = pipeline.ingest(req.records)
    except VectorStoreError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    setattr(request.app.state, LAST_INGEST_STATE_KEY, report.to_dict())
    return VectorUpsertResponse(
        summary=report.summary_line(),
        report=report.to_dict(),
        count=pipeline.backend.count(),
        dimension=pipeline.backend.dimension,
        backend=report.backend,
        metric=report.metric,
    )


@router.post("/vectorstore/search", response_model=VectorSearchResponse)
def vectorstore_search(
    req: VectorSearchRequest, request: Request
) -> VectorSearchResponse:
    """查询文本 → 编码 → 检索，返回一份完整的 ``SearchResult``（day064）.

    为什么要有它：``/vectorstore/backends`` 说"能用什么"、``stats`` 说"库里有什么"，
    但"给一段文本，最像的是哪几条"此前只能靠脚本回答。这个端点让那句问话
    变成一次可复现的调用（缺省 flat 后端逐位可复现，同一个请求体在任何机器上
    给出同一份结果）。

    ``result`` 里最值得先读的不是 ``hits`` 而是三个数字，它们把
    "**为什么只返回了两条**"变成可以直接读出来的事实：

    ```text
    candidates=0 且 filter_applied=True    → 过滤器把库里所有记录都排除了（条件写窄了）
    candidates=0 且 filter_applied=False   → 库是空的（先看 stats）
    candidates>0 而 count=0                → min_score 把全部命中切掉了
    candidates>0 且 count<candidates       → top_k 起了作用
    ```

    错误码约定（三者刻意分开，**不是**同一个 400）：

    ```text
    where 写法非法 / top_k 越界   → 400  FilterError（消息里带支持的 12 个运算符与上限值）
    查询向量与库维度不一致          → 409  VectorError（Conflict：库与查询用的编码器
                                          不是同一套，重新编码能修，换个请求参数修不了）
    请求体形状不对                 → 422
    ```

    ``top_k=0`` 表示用项目级基线（``settings.vector_default_top_k``）；
    负数照实送进校验并返回 400——**0 是"没给"，负数不是**。

    它**不做什么**：不写库、不改库、不做重排序（day068）、不做多路召回融合（day067）、
    不替调用方解释"为什么这条没被召回"（那是 day066/071 的评估层）。
    """
    query = req.query.strip()
    if not query:
        raise HTTPException(
            status_code=400,
            detail="query 不能为空：检索要有一段文本去编码，空查询会得到一个"
            "由编码器决定的任意向量（或零向量），而结果看起来仍然正常。",
        )
    try:
        result = vectorstore_pipeline(request).search(
            query,
            top_k=None if req.top_k == 0 else req.top_k,
            where=req.where,
            min_score=req.min_score,
        )
    except FilterError as exc:
        # FilterError 是 VectorStoreError 的子类，必须写在它前面；
        # 它代表"查询本身写错了"，消息里已经列出可选运算符与上限，原样带回即可。
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except VectorError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return VectorSearchResponse(
        summary=result.summary_line(),
        result=result.to_dict(),
        filter=describe_filter(req.where),
        filter_fields=filter_fields(req.where),
        query_chars=len(query),
    )


@router.get("/vectorstore/stats", response_model=VectorStatsResponse)
def vectorstore_stats(request: Request) -> VectorStatsResponse:
    """当前库状态 + **本实例**最近一次写入报告 + 编码器信息（day064）.

    三块内容各自回答一个问题，**不混在一起**：

    - ``info`` ← ``VectorBackend.info()``：后端名、度量、维度、条数、
      是否持久化、位置，以及后端特有的字段（flat 会多报快照版本——
      它不该进公共形状，否则另外两个后端会永远带着一个空字符串）；
    - ``last_ingest``：本应用实例最近一次写入的 ``IngestReport``，
      还没写过就是 ``None``。它是"**内存后端的写入只活在这个实例里**"
      这句话的可核对形式——换一个 app 实例，它必须是 ``None`` 且 ``count`` 为 0；
    - ``pipeline``：编码器与检索默认值（``top_k`` / ``min_score`` 的解析结果）、
      以及"优先用哪个字段编码"（``retrieval_text``，缺了才回落到 ``text``）。

    它**不做什么**：不落盘、不清库、不替调用方决定要不要重建索引
    （"要不要重建"取决于维度与编码器是否换过，那是人的判断）。

    状态挂在 ``app.state`` 上而不是模块级全局：一个进程里可能同时存在
    多个 app 实例（测试、多租户、影子流量），而**共享一个全局库会让
    其中一个的写入污染另一个的检索结果**，且没有任何报错。
    """
    pipeline = vectorstore_pipeline(request)
    return VectorStatsResponse(
        summary=pipeline.backend.info().summary_line(),
        info=pipeline.backend.info().to_dict(),
        pipeline=pipeline.stats(),
        last_ingest=getattr(request.app.state, LAST_INGEST_STATE_KEY, None),
    )


@router.delete("/vectorstore/records", response_model=VectorDeleteResponse)
def vectorstore_delete(
    req: VectorDeleteRequest, request: Request
) -> VectorDeleteResponse:
    """按 ``ids`` 或 ``where`` 删除记录，返回**真正删掉的**条数（day064）.

    为什么用 DELETE 带 JSON 体（而不是把条件塞进查询串）：``where`` 是一棵
    嵌套的子句树，塞进 URL 要么被截断、要么得自己发明一套编码。
    调用方用 ``client.request("DELETE", url, json={...})`` 即可（见测试）。

    四条约定，前三条都是 400：

    ```text
    ids 与 where 同时给出   → 400：无法判断调用方想删的是什么（base.delete 的契约）
    两者都不给              → 400："什么都没说"不等于"删全部"
    where 给成空字典        → 400：空子句在本包里等价于"无过滤"，
                                  用它删除等于清库——清库必须是一次写清了范围的显式操作
    删不存在的 id           → 200 且 removed=0：删除是幂等的，但返回值必须诚实
    ```

    第三条是本端点唯一一处**比底层更严**的地方，理由与它一致：
    底层把"空子句"当成"无过滤"是正确的（查询语义），但同一个值落到删除上
    就变成"清空整库"。**同一个语义在读写两条路径上的后果不对等时，
    收紧那条危险的一侧**。

    错误码：``where`` 写法非法 → 400；后端不支持删除 → 409；请求形状不对 → 422。

    **内存后端的删除只活在这个应用实例里**：它不会动磁盘上的任何快照，
    也不会影响另一个 app 实例。
    """
    ids = list(req.ids)
    where = req.where
    if ids and where is not None:
        raise HTTPException(
            status_code=400,
            detail="ids 与 where 只能给一个：同时给出会让「到底删了什么」变得不确定，"
            "而那种不确定的后果是「什么都没删」或「多删了一条」——两者都不会报错。",
        )
    if not ids and where is None:
        raise HTTPException(
            status_code=400,
            detail="ids 与 where 至少要给一个：两者都没给就不会删任何东西，"
            "而一个「删了 0 条」的成功响应很容易被当成「删干净了」。",
        )
    if where is not None and not where:
        raise HTTPException(
            status_code=400,
            detail="where 是空字典：空子句在本包里等价于「无过滤」，用它删除等于清库。"
            "请写明确的条件，或用 ids 逐条删除。",
        )
    store = request.app.state.vector_store
    try:
        removed = store.delete(ids or None, where=where)
    except BackendUnavailable as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (FilterError, RecordError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return VectorDeleteResponse(
        summary=f"删除 {removed} 条，剩余 {store.count()} 条",
        removed=removed,
        count=store.count(),
    )


@router.post("/vectorstore/compare", response_model=VectorCompareResponse)
def vectorstore_compare(
    req: VectorCompareRequest, request: Request
) -> VectorCompareResponse:
    """一批记录 × 一个查询 → 每个（后端 × 度量）一行对账结论（day064）.

    **本端点最重要的行为是"缺依赖时降级而不是 500"**：``compare_backends``
    把 ``BackendUnavailable`` / ``ImportError`` 折成一行 ``available=False``
    （带"缺什么 / 怎么装 / 还能用什么"的三段式说明），其余异常照旧上抛。
    于是同一个请求在装了 faiss 的机器与没装的机器上返回**行数相同**的报告，
    只差一个布尔——这正是这份报告能被贴进教程、能在 CI 与别人的机器上
    逐行对照的原因。若把缺依赖变成 500，第二台机器上就只能得到一句错误。

    四个刻意的选择：

    1. **临时库不落盘、也不进 ``app.state``**：对账是内存里的一次计算，
       每个 ``(后端, 度量)`` 各建一个实例（度量不可变，见 base.metric），
       ``path=""`` 明确不落盘；这些实例用完即弃，因此 compare
       **不会污染正在服务的那个库**（测试里有一条专门钉这点）；
    2. **记录在这里不归一化**（``METRIC_INNER_PRODUCT`` 就是"不归一化"那一支），
       由每个后端在写入时按自己的度量归一化。否则 ip 与 cosine 的两行永远相同，
       "换度量会换答案"这件事就在报告里消失了；
    3. **参照是降级参照**（``reference=None``）：本端点是生产代码，
       不能 import tests 里那份独立的暴力实现。因此可用行的
       ``agreed``/``overlap`` 只说明"**同一台机器上几个后端彼此一致**"，
       **不能当成 CI 的判据**——真正的对账在 ``tests/test_vectorstore_parity.py``，
       那里每一条都显式注入独立算出的期望排名；
    4. **对账的输入必须完整**（见 ``vectorstore_compare_records``）：
       缺 id 或文本全空白的记录直接 400，而不是跳过。

    错误码：``records`` 为空 / ``query`` 为空 → 400；某条记录不合格 → 400；
    未知度量 → 400（消息里带可选值与别名）；``top_k`` 越界 → 400；
    请求体形状不对 → 422。
    """
    if not req.records:
        raise HTTPException(
            status_code=400,
            detail="records 不能为空：没有记录就没有任何排名可以对照，"
            "此时返回空列表会让人误以为「后端之间一致」。",
        )
    query = req.query.strip()
    if not query:
        raise HTTPException(
            status_code=400,
            detail="query 不能为空：对账要有一个查询向量，否则没有任何名次可以对照。",
        )
    try:
        resolved_metrics = vectorstore_compare_metrics(req.metrics)
        prepared = vectorstore_compare_records(request, req.records)
        rows = compare_backends(
            prepared,
            request.app.state.embedding.embed(query),
            metrics=resolved_metrics,
            top_k=req.top_k or settings.vector_default_top_k,
            backends=[
                (name, vectorstore_backend_factory(name)) for name in backend_names()
            ],
        )
    except VectorStoreError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        # compare_backends 用 ValueError 表达"这个参数不值一算"（空记录 / top_k<1）：
        # 它们是请求的问题而不是环境的问题，因此与上面的 400 同码。
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    check = verify_parity(rows)
    available = [row for row in rows if row.available]
    agreed = [row for row in available if row.agreed >= min(row.top_k, row.count)]
    backends = sorted({row.backend for row in rows})
    return VectorCompareResponse(
        summary=(
            f"{len(backends)} 个后端 × {len(resolved_metrics)} 个度量 | 行数 {len(rows)}"
            f" | 可用 {len(available)} | 逐位一致 {len(agreed)}/{len(available)}"
        ),
        rows=[row.to_dict() for row in rows],
        check={
            "ok": check["ok"],
            "checked": check["checked"],
            "available": check["available"],
            "failures": check["failures"],
        },
        backends=backends,
        metrics=resolved_metrics,
        dimension=prepared[0].dimension,
    )


# --------------------------------------------------------------------------- #
# day065 索引构建（M6-D4）：/indexing/* 七个端点
#
# 与 /vectorstore/* 共用同一对注入对象（``app.state.vector_store`` 与
# ``app.state.embedding``），因此这一组回答的是**同一个库**的三个新问题：
#
# ```text
# 这份索引是用哪个编码器的哪个版本建的？   → identity / manifest（/indexing/status）
# 上一版到这一版，哪些块没变？             → /indexing/plan 的差集
# 我能不能回到上一版，或者照清单重建一次？   → /indexing/versions / rollback / build
# ```
#
# 三条纪律，每一条在本段里都对应一个具体的写法：
#
# 1. **只有 ``/indexing/build`` 会改状态**：plan 是看、verify 是查、
#    versions 是读、rollback 只动版本指针、backup 只往索引目录里写快照。
#    把"看"与"做"分成两个端点，是为了让"这次会花多少钱"能先被问出口。
# 2. **落盘目录缺省是空串**（``app.state.indexing_dir``）：缺省 app 不落盘，
#    版本表只在内存里、备份表连路径都没有。``/indexing/backup`` 因此在
#    缺省实例上返回 **400**（"没有配置索引目录"）而不是在进程当前目录里
#    建出一个 ``backups``——一次"顺手试试"的调用不该留下目录。
# 3. **错误一律走 400 而不是 500**：本层认识的三族异常（
#    ``IndexingError`` / ``VersionError`` / ``BackupError``）都继承自
#    ``IndexingError``，它们代表"配置与数据之间的矛盾"，调用方改一个参数
#    就能修；只有认不出来的异常才让它上抛成 500（报告只收留我们认识的失败）。
# --------------------------------------------------------------------------- #

#: 最近一次构建报告挂在 ``app.state`` 上的键名（app.create_app 里初始化成 None）.
#: 与 ``LAST_INGEST_STATE_KEY`` 同一条纪律：常量只有一处定义（这里），
#: 状态挂在**应用实例**上而不是模块级全局——挂全局会让
#: "同一个进程里另建一个 app 的 last_report 必须是 None" 这条性质静默失效，
#: 而它的表现只是另一个测试偶发失败。
INDEXING_LAST_REPORT_STATE_KEY = "indexing_last_report"


def indexing_pipeline(request: Request) -> IndexingPipeline:
    """从 ``app.state`` 取索引流水线（本组端点的公共依赖）."""
    pipeline = getattr(request.app.state, "indexing_pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="本实例没有装配索引流水线：create_app 未注入 indexing_pipeline，"
            "且默认装配也没有产出（见 app.assemble_indexing_pipeline）。",
        )
    return pipeline


def indexing_builder(request: Request) -> IndexBuilder:
    """从 ``app.state`` 取索引构建器（plan / build / verify 的执行者）.

    为什么单独取它而不是从流水线里翻：``/indexing/plan`` 要的"上一版是谁"
    与 ``/indexing/verify`` 要的"清单与库对不上在哪"都只有 ``IndexBuilder``
    认识（它同时握着编码器身份、后端与版本表）。让端点去读流水线的私有字段，
    等于把"改一个字段名就静默失效"的可能性写进契约里。
    """
    builder = getattr(request.app.state, "indexing_builder", None)
    if builder is None:
        raise HTTPException(
            status_code=503,
            detail="本实例没有可用的索引构建器（app.state.indexing_builder 为空）："
            "本端点需要 IndexBuilder 才有的口径。请用 create_app 的默认装配。",
        )
    return builder


def indexing_version_store(request: Request) -> IndexVersionStore | None:
    """从 ``app.state`` 取版本表；没有时返回 ``None``（"没有历史"是合法状态）."""
    return getattr(request.app.state, "indexing_versions", None)


def indexing_backup_store(request: Request) -> IndexBackupStore | None:
    """从 ``app.state`` 取备份表；没有时返回 ``None``."""
    return getattr(request.app.state, "indexing_backups", None)


def indexing_preview_mode(
    mode: str, plan: IndexPlan, parent: IndexManifest | None
) -> tuple[str, str]:
    """预览"这次构建会不会换挡"，返回 ``(实际会用的模式, 说明)``.

    判据与 ``IndexBuilder._apply_rebuild_threshold`` **逐条相同**：

    ```text
    mode 是 incremental   全量模式不需要换挡（它已经是全量了）
    parent 存在            没有上一版时"变更比例"没有分母（全部是新增）
    plan.total > 0         空记录集时两种模式等价，不换挡也不报错
    比例 > 阈值（严格大于）等于阈值时不换挡：阈值是"超过它才换"的那条线
    ```

    为什么不直接调构建器那个私有方法：计划端点是"看"，不该伸手进别人的
    实现里。规则的权威版本仍在构建器（它的结论落在 ``report.note``），
    这里只回答"值不值得动手"——两处若不一致，以构建报告为准。
    """
    if mode != BUILD_MODE_INCREMENTAL or parent is None or plan.total <= 0:
        return mode, ""
    ratio = plan.changed / plan.total
    threshold = float(settings.indexing_full_rebuild_threshold)
    if ratio > threshold:
        return BUILD_MODE_FULL, (
            f"变更比例 {ratio:.1%} 超过阈值 {threshold:.2f}，"
            "构建时会从 incremental 切到 full——增量不是永远更省："
            "变更比例很高时，逐条 upsert 加逐条 delete 的开销反而高于清空重写"
        )
    return mode, ""


def indexing_current_manifest(request: Request) -> tuple[IndexManifest | None, str]:
    """取"当前该被对账的那一份清单"，返回 ``(清单, 来源)``.

    三级取值顺序，每一级都对应一种真实处境：

    ```text
    version  版本表里 current 指向的那一版（正常路径：构建成功过）
    store    版本表为空 / 还没采纳过          → 按库的现状现算一份
    ```

    注意"现算一份"的清单是**从现在起**的基线，不是历史的那一版：
    库里被人删过一条这种问题，只有拿**原来那份清单**来对账才看得出来。
    """
    versions = indexing_version_store(request)
    if versions is not None and versions.current:
        manifest = versions.get(versions.current)
        if manifest is not None:
            return manifest, "current"
    builder = indexing_builder(request)
    return (
        manifest_from_store(
            builder.backend,
            identity=builder.identity,
            created_at=indexing_utc_now(),
        ),
        "store",
    )


def indexing_backup_sources(backend: VectorBackend) -> tuple[list[str], str]:
    """取"这次备份要拷哪些文件"，返回 ``(源文件列表, 说明)``.

    两种情形，**都必须如实说清**：

    ```text
    后端能落盘且有位置    → 先 persist() 刷新快照，再把它当源文件
    内存后端（无位置）     → source_files=[]，只在快照里放清单本身
    ```

    为什么内存后端也要给出 ``200`` 而不是报错：它**确实**有东西可备份——
    那份清单（"这一版该有哪些块、每个块的向量键是什么"）是唯一的、
    可以被重算的证据；而没有库文件这件事必须由 ``note`` 说出来，
    否则一份 files=[] 的快照会被当成"备份失败了"或（更糟）"备份成功了"。

    这与 ``IndexBackupStore._resolve_sources`` 拒绝"源文件缺失时静默跳过"
    是同一条纪律的两面：**缺文件要喊，没有文件要说。**
    """
    if backend.supports_persistence:
        try:
            location = backend.persist()
        except VectorStoreError:
            location = ""
        if location:
            return [location], ""
    return [], (
        "内存后端没有库文件可备份（后端位置为空）：本次只备份了清单本身，"
        "快照里没有向量数据——恢复它只能拿回「这一版该有什么」，拿不回向量。"
        "要连库文件一起备份，请把后端配成可落盘（settings.vector_persist_path）。"
    )


@router.get("/indexing/status", response_model=IndexingStatusResponse)
def indexing_status(request: Request) -> IndexingStatusResponse:
    """当前索引状态：库 + 编码器身份 + 清单 + 版本数 + 备份数 + 缓存（day065）.

    它把 ``IndexingPipeline.stats()`` 的九块原样端出来（见
    ``IndexingStatusResponse`` 的说明），再补上**本实例**最近一次构建报告。

    三处不混在一起的理由，与 day064 的 ``/vectorstore/stats`` 同源：

    - ``dimension`` 是**库**的维度（空库为 0），编码器的维度在 ``identity`` 里。
      两者不一致正是"换过编码器但没重建"这个脏状态，合并成一个字段它就消失了；
    - ``manifest`` 是"这个实例采纳过的那一版"，没有就是 ``None``——本端点
      不去磁盘上找"当前生效的清单"，因为那要回答"清单文件在哪、它是不是
      现在生效的那份"，而那两个问题的答案在版本表与配置里；
    - ``last_report`` 是"最近一次构建花了什么"，与 ``count``（库里有什么）
      是两件事：构建失败时 ``count`` 可能没变，而报告里会写着失败几条。

    它**不做什么**：不落盘、不清库、不替调用方决定要不要重建索引，
    也不把清单的 entries 塞进响应体（``include_entries=False``）。

    一处**端点层**的补全（写在明面上，因为它涉及两个"清单"的差别）：
    ``pipeline.stats()`` 的 ``manifest`` 说的是"这条流水线**亲手构建过**的那一版"，
    而 ``/indexing/build`` 走的是 ``IndexBuilder.build``（请求体里的 ``backup``
    开关只有它能如实生效，见那个端点的说明）。因此本端点在这里补一步：
    ``manifest`` 为空时取版本表 ``current`` 指向的那一版——那才是"现在生效的
    是哪一版"。补完之后 ``status`` 报的版本与 ``build`` 返回的 ``version_id``
    才是同一个（两边都已经登记并采纳过）。
    """
    pipeline = indexing_pipeline(request)
    stats = pipeline.stats()
    if stats.get("manifest") is None:
        versions = indexing_version_store(request)
        if versions is not None and versions.current:
            current = versions.get(versions.current)
            if current is not None:
                stats["manifest"] = current.to_dict(include_entries=False)
    identity = stats.get("identity") or {}
    manifest = stats.get("manifest")
    version = manifest["version_id"] if manifest else "（未构建）"
    directory = getattr(request.app.state, "indexing_dir", "")
    return IndexingStatusResponse(
        summary=(
            f"{stats['backend']}/{stats['metric']} {stats['dimension']}d | "
            f"库 {stats['count']} 条 | "
            f"编码器 {identity.get('provider', '')}/{identity.get('model', '')} "
            f"{identity.get('dimension', 0)}d | "
            f"版本 {stats['versions']} | 备份 {stats['backups']} | "
            f"当前 {version} | 索引目录 {directory or '（不落盘）'}"
        ),
        stats=stats,
        last_report=getattr(request.app.state, INDEXING_LAST_REPORT_STATE_KEY, None),
        indexing_dir=directory,
    )


@router.post("/indexing/plan", response_model=IndexingPlanResponse)
def indexing_plan(req: IndexingPlanRequest, request: Request) -> IndexingPlanResponse:
    """给一批记录 → 返回差集与模式判定（day065，**只读**）.

    这个端点回答的是"这次构建要花多少钱"：``plan.to_encode`` 是**需要重新
    编码**的条数，而那个数字决定了这次构建的价格。它存在的意义就是把
    "先看计划、再决定要不要动手"变成一次可复现的调用。

    **它一个字节都不写**——不编码、不写库、不动版本表：

    ```text
    比"再写一个只读查询"更重要的是一个纪律：计划是**看**，不是**做**。
    因此它调用的是 builder.plan(...)（与 build 走的同一个函数），
    而 build 才是那个会改库的端点。
    ```

    四个刻意的选择：

    1. **上一版来自版本表**（``versions.latest()``），与 ``builder.build``
       的解析顺序一致（显式 > 版本表 > 首版）。自己另立一套"上一版是谁"
       会让计划与构建算出不同的差集，而它们的报告都看起来正常；
    2. **模式判定是预览，不是决定**：``mode`` / ``switched`` / ``threshold``
       三件一起回答"会不会从增量切到全量"，真正的换挡在构建时执行
       （见 ``indexing_preview_mode``）；
    3. **非法 mode 一样返回 400**（消息里列出可选值）：计划阶段就报错，
       比构建到一半才发现模式拼错便宜得多；
    4. **``reason`` 直接来自 ``planner.plan_reason``**：它把"换了编码器
       导致全部重算"与"数据真的变了很多"分开——两者在报告上都只是
       "更新了 N 条"，但下一步动作完全不同。

    错误码：``records`` 为空 / ``mode`` 非法 → **400**；请求体形状不对 → 422。
    """
    if not req.records:
        raise HTTPException(
            status_code=400,
            detail="records 不能为空：没有记录就没有差集可算——一次"
            "「什么都不变」的计划会被读成「数据没问题」，而真相是没看。",
        )
    builder = indexing_builder(request)
    requested = settings.indexing_mode if req.mode is None else str(req.mode).strip()
    if requested not in BUILD_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"未知构建模式 {req.mode!r}（解析后 {requested!r}），"
            f"可选：{' / '.join(BUILD_MODES)}。"
            "全量会先清空整个库，因此本端点不接受任何'看起来差不多'的值。",
        )
    versions = indexing_version_store(request)
    parent = versions.latest() if versions is not None else None
    plan = builder.plan(req.records, parent=parent)
    mode, switch_note = indexing_preview_mode(requested, plan, parent)
    switched = mode != requested
    summary = f"{plan.summary_line()} | 模式 {mode}"
    if switched:
        summary = f"{summary}（{switch_note}）"
    return IndexingPlanResponse(
        summary=summary,
        plan=plan.to_dict(),
        reason=plan.reason,
        requested_mode=requested,
        mode=mode,
        switched=switched,
        threshold=float(settings.indexing_full_rebuild_threshold),
        parent_version=parent.version_id if parent is not None else "",
        identity=builder.identity.to_dict(),
    )


@router.post("/indexing/build", response_model=IndexingBuildResponse)
def indexing_build(req: IndexingBuildRequest, request: Request) -> IndexingBuildResponse:
    """执行一次构建，返回 ``IndexingReport``（day065，**本组唯一真正改状态的端点**）.

    它把"记录 → 计划 → 批量编码 + 缓存 → 增量写库 → 清单 + 版本 + 备份"
    这条链子跑完，并把账端出来。三个参数的去向各不相同：

    ```text
    mode     None = 按 settings.indexing_mode；非法值 → 400（消息里列出可选值）
    reason   只进备份记录（"这份快照为什么建的"），不进清单
    backup   构建成功后是否顺手创建一份快照（缺省 false：备份是运维决策）
    ```

    三件必须说清的事：

    1. **请求体里没有"选后端 / 选编码器"**：后端是 ``app.state.vector_store``、
    编码器是 ``app.state.embedding``，与 ``/vectorstore/*`` 是同一对实例。
    允许一个请求换编码器，会让"这份索引是用谁建的"变成一件要翻日志的事；
    2. **报告里的数字必须分开读**：``written`` 与 ``unchanged`` 分开，
    才看得出"这次其实什么都没做"；``encoded`` 与 ``cache_hits`` 分开，
    才看得出"库没变"与"没花编码"是两件事；``skipped`` 与 ``failed`` 分开，
    报告才自己指出了该找谁修；
    3. **只提交本次 build 的报告**到 ``app.state.indexing_last_report``：
    ``/indexing/status`` 会把它带回来，而**换一个 app 实例它必须是 ``None``**
    ——那是"内存后端的写入只活在这个实例里"的可核对形式。

    错误码：``records`` 为空 / ``mode`` 非法 / 维度护栏与编码器报错
    （``IndexingError`` 一族，含 ``EncodingError``）→ **400**；
    后端拒绝（``VectorStoreError``）→ **400**；请求体形状不对 → 422。
    **不做**大小写与别名的宽容：猜错一次模式的代价是一次全量重建。
    """
    if not req.records:
        raise HTTPException(
            status_code=400,
            detail="records 不能为空：增量模式下它会删掉上一版的全部条目"
            "（计划里全是 removed），而一次「清库」不该由一次空请求触发。",
        )
    builder = indexing_builder(request)
    try:
        manifest, report = builder.build(
            req.records,
            mode=req.mode,
            reason=req.reason,
            backup=req.backup,
        )
    except IndexingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except VectorStoreError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    setattr(request.app.state, INDEXING_LAST_REPORT_STATE_KEY, report.to_dict())
    return IndexingBuildResponse(
        summary=report.summary_line(),
        report=report.to_dict(),
        version_id=report.version_id,
        count=builder.backend.count(),
        manifest=manifest.to_dict(include_entries=False),
    )


@router.get("/indexing/versions", response_model=IndexingVersionsResponse)
def indexing_versions(request: Request) -> IndexingVersionsResponse:
    """版本历史 + 当前指针（day065）.

    两个字段各自回答一个问题，**不能合并**：

    ```text
    history   有哪几版，按**登记顺序**（旧 → 新）
    current   现在生效的是哪一版（"被采纳的"，与"最近登记的"不是一回事）
    ```

    为什么按登记顺序而不是版本号排序：版本号是内容摘要，它的字典序没有任何
    时间含义，按它排出来的"历史"不是时间线。为什么"登记的"与"采纳的"要分开：
    构建失败或验证不过的版本会被登记却从不采纳，而那个差额正是回滚要用的信息
    （按时间倒序回滚会切到一个从没生效过的版本）。

    ``lineage`` 是 ``current`` 沿 ``parent_version`` 回溯出来的血缘（旧 → 新）。
    血缘断裂时它留空而不是抛错：**"历史读不全"不该让一个只读端点变成 500**，
    真正的处置（补登记或清掉指向它的 parent_version）由人对症去做。

    历史里的每一版只给 ``version_id`` 与 ``summary_line``：把每版清单的全部
    条目塞进响应体等于把历史索引的全文都发出去，而这里要回答的只是
    "有哪几版、各是什么"。本端点不读磁盘——**要读盘请显式调用
    ``IndexVersionStore.load()``**（"悄悄用磁盘上的旧指针回滚"是一类真实的错误）。
    """
    versions = indexing_version_store(request)
    manifests = versions.history() if versions is not None else []
    current = versions.current if versions is not None else ""
    lineage: list[str] = []
    if versions is not None and current:
        try:
            lineage = [manifest.version_id for manifest in versions.lineage()]
        except VersionError:
            lineage = []
    return IndexingVersionsResponse(
        summary=f"{len(manifests)} 版 | 当前 {current or '（未采纳）'}",
        history=[
            {"version_id": manifest.version_id, "summary_line": manifest.summary_line()}
            for manifest in manifests
        ],
        current=current,
        count=len(manifests),
        lineage=lineage,
    )


@router.post("/indexing/verify", response_model=IndexingVerifyResponse)
def indexing_verify(
    req: IndexingVerifyRequest, request: Request
) -> IndexingVerifyResponse:
    """清单与库的一致性体检（day065），返回 ``verify_index`` 的结论.

    它是这一课最重要的检查点：清单与库可以**互相矛盾**，而矛盾不会在检索时
    抛异常——它只会让增量更新算出错误的差集（本该重算的块被判定为"没变"），
    症状是检索质量下降。四类不一致各有各的处置：

    ```text
    清单有、库里没有   上一次构建半途失败，或有人手工删过记录 → 重建清单
    库里有、清单没有   有人绕过索引流程直接写了库             → 重建清单
    维度不符           清单描述的是另一个编码器建的库         → 不能用它驱动增量
    度量 / 后端不符     同一批向量换个度量/后端就是另一批最近邻 → 新建实例，别改
    ```

    ``version_id`` 的三级取值：显式给了就用版本表里那一版（**不存在 → 400**，
    消息里说明"版本表里没有这一版"）；没给就用**当前采纳的**那一版；
    连当前都没有（还没构建过）就按库的现状 ``manifest_from_store`` 现算一份。
    最后一条要记牢：现算出来的清单是"从现在起"的基线，**不是历史的那一版**
    ——"库里被人删了一条"这种问题，只有拿原来那份清单来对账才看得出来。

    它**不做任何修复**：名字是 verify，不是 repair。顺手补一条缺失的记录，
    会让"库与清单不一致"这件事**下一次也查不出来**——而它的成因
    （半途失败的构建、人工改库）才是要修的东西。
    """
    builder = indexing_builder(request)
    versions = indexing_version_store(request)
    if req.version_id:
        manifest = versions.get(req.version_id) if versions is not None else None
        if manifest is None:
            raise HTTPException(
                status_code=400,
                detail=f"版本表里没有这一版 {req.version_id!r}：可用的版本号只能从"
                "GET /indexing/versions 的 history 里拿——不要凭记忆拼一个 id，"
                "一个拼错的版本号与一个「还没构建过」的版本在这里长得一模一样。",
            )
        source = "version"
    else:
        manifest, source = indexing_current_manifest(request)
    result = verify_index(manifest, builder.backend)
    problems = list(result["problems"])
    return IndexingVerifyResponse(
        summary=(
            f"体检 {manifest.version_id} | ok={result['ok']} | 问题 {len(problems)} 条"
        ),
        ok=bool(result["ok"]),
        checks=result["checks"],
        problems=problems,
        version_id=manifest.version_id,
        source=source,
    )


@router.post("/indexing/backup", response_model=IndexingBackupResponse)
def indexing_backup(
    req: IndexingBackupRequest, request: Request
) -> IndexingBackupResponse:
    """创建一份备份（day065）：把"还能回到上一版"从一句话变成一次拷贝.

    备份与清单的分工是这一课的核心：

    ```text
    清单   说得出"这一版该有哪些块、每个块的向量键是什么"（可重算的账）
    备份   固化了"当时的库文件是什么"（唯一数据的那一份拷贝）
    ```

    **``indexing_dir`` 为空时返回 400，而不是 500**（本组最硬的一条约定）：
    备份必须落在库里之外的目录上，没有配置索引目录时它连"往哪儿写"都答不出来。
    这时**不建任何目录**、也不静默降级成"备份到进程当前目录"——
    一次"顺手试试"的调用在仓库里留下一个 ``backups`` 目录，
    正是 day064 用 ``vector_persist_path = ""`` 挡掉的那类事故。

    两处如实告知（都不算失败）：

    1. ``source_files`` 取自后端的 ``persist()``；**内存后端没有库文件**时
    它为空列表，快照里只放清单本身，``note`` 里会写清"拿不回向量"；
    2. ``keep``（``settings.indexing_backups_keep``）是保留上限，``create``
    之后自动裁剪——账本里曾经有过、目录里未必还在（见 ``list`` 与 ``latest``
    的差别），而"这一版曾经备份过"是一条不该被删掉的事实。

    错误码：``indexing_dir`` 为空 → **400**；``BackupError``（磁盘不可写 /
    清单与内容不符等）→ **400**；请求体形状不对 → 422。
    """
    backups = indexing_backup_store(request)
    if backups is None or not backups.path.strip():
        raise HTTPException(
            status_code=400,
            detail="本实例没有配置索引目录（indexing_dir 为空）：备份需要一个"
            "落盘位置——它必须写在库文件之外的目录上，否则「把库写坏」会连备份"
            "一起带走。请在 create_app 时传入 indexing_dir=settings.indexing_dir，"
            "或直接调 IndexBackupStore.create(...) 指定一个目录。",
        )
    builder = indexing_builder(request)
    manifest, _ = indexing_current_manifest(request)
    sources, note = indexing_backup_sources(builder.backend)
    try:
        record = backups.create(
            source_files=sources,
            manifest=manifest,
            reason=req.reason or f"手动备份 {manifest.version_id}",
        )
    except BackupError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return IndexingBackupResponse(
        summary=record.summary_line(),
        backup=record.to_dict(),
        backup_id=record.backup_id,
        version_id=record.version_id,
        files=list(record.files),
        size_bytes=record.size_bytes,
        keep=backups.keep,
        note=note,
    )


@router.post("/indexing/rollback", response_model=IndexingRollbackResponse)
def indexing_rollback(
    req: IndexingRollbackRequest, request: Request
) -> IndexingRollbackResponse:
    """回滚到血缘上的上一版（day065，``steps`` 缺省 1）.

    **回滚是运维动作，不自动重建索引；重建请调用 ``/indexing/build``。**
    本端点只改**版本指针**：它把 ``current`` 从一版移到血缘上前 ``steps`` 步
    的那一版，既不清库、也不重算任何向量、更不改备份。那么"回滚之后检索结果
    为什么没变"就有了一个直接答案——因为库里的向量本来就没动；
    要真正回到那一版的内容，请在那之后按那一版的记录重建，或从备份 ``restore``。

    为什么是"沿血缘"而不是"按登记顺序取倒数第 N 个"：版本表里可以有
    **构建失败、从未被采纳**的版本，按时间倒序回滚会切到一个从没生效过的版本，
    而它在报告里完全看不出来（``current`` 变成了一个合法的版本号，没有异常）。

    两种拒绝都是 **400**（``VersionError``，消息里带足定位信息）：

    ```text
    steps < 1        "回退 0 步"不是一次回滚，负数没有意义
    血缘不够         消息里给出当前版本与可用步数——不静默退到"能退到的最远那一版"
    ```

    最后一条尤其重要：一次"想退 3 步只退了 1 步"的回滚如果静默成功，
    调用方会以为自己在一个自己没打算去的版本上。
    """
    versions = indexing_version_store(request)
    if versions is None:
        raise HTTPException(
            status_code=503,
            detail="本实例没有装配版本表（app.state.indexing_versions 为空）："
            "回滚必须沿 parent_version 回溯，而血缘只有版本表认识。",
        )
    previous = versions.current
    try:
        current = versions.rollback(steps=req.steps)
    except VersionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    manifest = versions.get(current)
    return IndexingRollbackResponse(
        summary=f"回滚 {req.steps} 步：{previous or '（未采纳）'} → {current}",
        previous=previous,
        current=current,
        steps=req.steps,
        manifest=manifest.to_dict(include_entries=False) if manifest is not None else {},
        note=(
            "回滚是运维动作，不自动重建索引：它只改版本指针，不动库与数据——"
            "库里现在的向量仍是回滚前那一版的内容。要真正回到这一版，"
            "请调用 /indexing/build（带上那一版的记录），或从备份 restore。"
        ),
    )


# --------------------------------------------------------------------------- #
# day066 检索（M6-D5）：/retrieval/* 五个端点
# （day067 又加了两个：/retrieval/hybrid 与 /retrieval/lexical/status，
#   见本文件末尾那一段，因此本组现在是**七个**端点。）
#
# 与上面两组共用同一对注入对象（``app.state.vector_store`` 与
# ``app.state.embedding``），因此这一组问的是**同一个库**的第四类问题：
#
# ```text
# 给一句话，取回哪几条？                → /retrieval/search
# 为什么是这几条、为什么只有这几条？      → /retrieval/explain 的 lines
# 检索不到时该说"资料里没有"还是编一个？   → /retrieval/answer（未命中时一次 LLM 都不调）
# 这句话该去哪个库、那一版索引还作数吗？   → /retrieval/routes 与 /retrieval/status
# ```
#
# 五条纪律，每一条在本段里都对应一个具体写法：
#
# 1. **导入期不构造后端、不读盘、不联网**：本段只有常量、函数与路由装饰器。
#    路由器 / 检索器 / LLM 全在**请求期**从 ``app.state`` 取或现装
#    （见 ``retrieval_router`` / ``retrieval_manifest``），因此
#    ``import api.routes`` 一次 I/O 都不做。
# 2. **端点只做"参数 → 调 retrieval → 序列化"**：深度、阈值、多样性、
#    空结果诊断全部留在 ``retrieval`` 包里（那里有它们的唯一实现）。
# 3. **空库是合法状态**：库为空时返回 200 且 ``empty_reason="no_data"``，
#    不报错也不 500——"刚建好还没写数据"与"检索坏了"是两件事。
# 4. **请求参数问题一律 400，且消息里带合法取值**：消息由检索层给出
#    （top_k 的上限、支持的时间写法、可用路由名都在里面），端点只负责选状态码。
# 5. **未配置 LLM 时 /retrieval/answer 返回 400，不拿 MockLLM 兜底**：
#    RAG 的产物是一段会被当成事实读的答案，用按脚本回复的占位实现去回答
#    知识库问题，正是这一课的护栏要拦的那类"看起来最像成功"的失败。
#
# 错误码为什么只有"400 / 200"两种（本组最需要解释的一处）：
# 本层认识的失败全部继承自 ``RetrievalError``，它们都是**可预期的拒绝**——
# 参数写错（改调用点）、库与编码器对不上（重建索引）、预算放不下（调配置）、
# 索引没装配（先建索引）。四者的出路不同，但对 HTTP 调用方而言都是
# "这次请求没做成、消息里写着为什么"，因此统一 400 并把消息原样带回
# （与 /indexing/* 的"三族异常都走 400"同一条纪律）。**认不出来的异常一律
# 上抛成 500**：报告只收留我们认识的失败。
# --------------------------------------------------------------------------- #

#: 检索路由器挂在 ``app.state`` 上的键名（可选）。**缺省不需要它**：
#: 本端点组会在请求期用 ``app.state.vector_store`` + ``app.state.embedding``
#: 现装一个单路由路由器，于是"缺省 app 的 /retrieval/* 就能用"这件事
#: 不需要 create_app 认识检索这一层。多索引（不同语料 / 不同编码器 /
#: 冻结的历史归档库）在这里注入一个 ``StoreRouter``。
RETRIEVAL_ROUTER_STATE_KEY = "retrieval_router"

#: /retrieval/answer 用的 LLM 挂在 ``app.state`` 上的键名（可选）。
#: 缺省回落到 ``app.state.llm``——与 /chat 用同一个实例，因此
#: "问答与对话各拿一个模型"这种裂缝不会因为一次默认装配就出现。
RETRIEVAL_LLM_STATE_KEY = "retrieval_llm"

#: 缺省装配出来的那条路由名（= ``Retriever(name=...)`` 的默认值）。
#: 它与 ``StoreRouter(default=...)`` 用的是**同一个字符串**：不传 route 时
#: 永远有路可走，而"默认路由名与检索器名不是同一个"会让 ``/retrieval/status``
#: 回显的 ``route.name`` 与 ``retriever.name`` 对不上——一个只会让人怀疑配置的差值。
DEFAULT_RETRIEVAL_ROUTE = "default"

#: 缺省那条路由的说明（``/retrieval/routes`` 直接展示它）。
DEFAULT_RETRIEVAL_ROUTE_DESCRIPTION = (
    "本实例的缺省库：由 app.state.vector_store 与 app.state.embedding 现装，"
    "单路由、name='default'"
)


def retrieval_manifest(request: Request) -> tuple[IndexManifest | None, str]:
    """取"这次检索该对账的那一版清单"，返回 ``(清单, 来源)``.

    ```text
    version  版本表里 current 指向的那一版（/indexing/build 登记并采纳过）
    none     还没构建过 → 不带清单检索：这时 IndexState 只报库的现状，无漂移可言
    ```

    **只读内存**：``IndexVersionStore.get`` 是一次字典查找，不读盘
    （缺省 app 的版本表连路径都没有，``indexing_dir=""``）。要读盘请显式
    ``IndexVersionStore.load()``——"悄悄用磁盘上的旧指针"是一类真实的错误。

    为什么带上清单：检索结果里的 ``index.version_id`` 与 ``drift`` 是本课
    兑现的第二个伏笔——"我查的是哪一版索引""清单与库有没有互相矛盾"必须
    跟着每一条结果走（见 ``retrieval.types.IndexState`` 的取舍：漂移只观测、
    不阻断，但绝不静默）。

    拿不到清单时**返回 None 而不是现算一份**：现算出来的清单是"从现在起"的
    基线，而"库里被人删过一条"这种问题，只有拿**原来那份**来对账才看得出来。
    """
    versions = getattr(request.app.state, "indexing_versions", None)
    if versions is not None and versions.current:
        manifest = versions.get(versions.current)
        if manifest is not None:
            return manifest, "version"
    return None, "none"


def retrieval_router(request: Request) -> StoreRouter:
    """取检索路由器：**注入优先**，否则按注入的库与编码器现装一个单路由表.

    为什么现装而不是在 create_app 里装配：两条路都能跑，但"现装"让这一层的
    依赖全部来自 ``app.state`` 上那两个既有注入点（``vector_store`` /
    ``embedding``），因此 day064 与 day065 注入的任何后端与编码器在这里
    **自动是同一对实例**——而"查询侧与写入侧各拿一个编码器"正是
    "检索结果排序不太对"却不报任何错的经典成因。

    为什么每次请求现装而不缓存：它只持有两个引用（与 ``vectorstore_pipeline``
    同一条纪律，构造是 O(1) 且只读一次 settings）。缓存会让"注入的后端被换掉"
    这件事在某个请求上悄悄失效，而那种失效没有任何报错。

    注入了一个**非 Router** 的对象时给 500 而不是 400：那是装配写错了，
    不是请求参数写错了——把它降级成 400 会让一个必然复现的配置错误
    看起来像"这次请求碰巧不对"。
    """
    injected = getattr(request.app.state, RETRIEVAL_ROUTER_STATE_KEY, None)
    if injected is None:
        manifest, _source = retrieval_manifest(request)
        retriever = build_retriever(
            request.app.state.vector_store,
            request.app.state.embedding,
            manifest=manifest,
        )
        router = StoreRouter(default=retriever.name)
        router.register(
            retriever.name,
            retriever,
            description=DEFAULT_RETRIEVAL_ROUTE_DESCRIPTION,
        )
        return router
    if not isinstance(injected, StoreRouter):
        raise HTTPException(
            status_code=500,
            detail=f"app.state.{RETRIEVAL_ROUTER_STATE_KEY} 必须是 StoreRouter，"
            f"收到 {type(injected).__name__}：这是装配问题而不是请求参数问题，"
            "把它当成 400 会让一个必然复现的配置错误看起来像一次偶然。",
        )
    return injected


def retrieval_defaults() -> dict[str, Any]:
    """``config`` 里 ``retrieval_*`` 那一组的当前取值（"为什么是这个数"的出处）.

    与 ``/vectorstore/backends`` 的 ``current_*`` 同一个用途：**能力与当前取值
    必须分开**——只给一个数字时，"这个 5 是项目默认还是谁改过"要靠翻配置才知道。
    ``max_fetch_k`` 也列进来：它是深度封顶（与 ``vectorstore.MAX_TOP_K``
    同一个常量），而"我设了 2000 为什么只取了 1000"这个问题只能靠它回答。

    day067 补了六项（``retrieval_hybrid_*`` / ``retrieval_bm25_*``）：它们与
    上面八项同属"只改变怎么问"的那一组，但多了一条"新能力默认不生效"
    ——``hybrid_enabled=False`` 时 ``/retrieval/hybrid`` 会拒绝请求，
    因此这个字段必须能被读出来（否则"端点为什么说没启用"要翻环境变量才知道）。
    """
    return {
        "top_k": settings.retrieval_top_k,
        "fetch_multiplier": settings.retrieval_fetch_multiplier,
        "min_score": settings.retrieval_min_score,
        "max_per_doc": settings.retrieval_max_per_doc,
        "time_field": settings.retrieval_time_field,
        "max_context_chars": settings.retrieval_max_context_chars,
        "per_hit_chars": settings.retrieval_per_hit_chars,
        "strict_index": settings.retrieval_strict_index,
        "max_fetch_k": MAX_FETCH_K,
        "hybrid_enabled": settings.retrieval_hybrid_enabled,
        "hybrid_strategy": settings.retrieval_hybrid_strategy,
        "hybrid_alpha": settings.retrieval_hybrid_alpha,
        "hybrid_rrf_k": settings.retrieval_hybrid_rrf_k,
        "bm25_k1": settings.retrieval_bm25_k1,
        "bm25_b": settings.retrieval_bm25_b,
    }


def retrieval_time_range(payload: RetrievalTimeRange | None) -> TimeRange | None:
    """契约对象 → 领域对象（``RetrievalTimeRange`` → ``TimeRange``）.

    转换只做一次搬字段：**校验不在这里**。ISO 解析、空字符串、倒置区间
    全部由 ``TimeRange.__post_init__`` 判，消息也由它给（"倒置的区间必然
    返回空结果"这类话只有那一层说得出来）。端点上再判一遍的后果是
    两份规则迟早分家，而分家的表现是"某个写法在 search 里被拒、在 answer 里通过"。
    """
    if payload is None:
        return None
    return TimeRange(field=payload.field, start=payload.start, end=payload.end)


def retrieval_query_of(req: RetrievalSearchRequest | RetrievalAnswerRequest) -> RetrievalQuery:
    """请求体 → ``RetrievalQuery``（**校验全在检索包里**，端点不重写一遍）.

    两个请求体共用本函数：``/retrieval/search`` 与 ``/retrieval/explain`` 的文本
    字段叫 ``query``（它就是被编码的那句话），``/retrieval/answer`` 的叫
    ``question``（它是被回答的那个问题）。名字不同是刻意的，而**其余七个参数
    逐字相同**——因此这里只区分"文本从哪个字段来"，不做两套转换。

    ``route`` 原样带进查询：``StoreRouter.retrieve`` 的规则是
    "**请求里的 route 优先于调用参数**"（一份查询是可以被存盘再回放的，
    那时"该去哪"是请求的一部分），端点因此只需填一次。
    """
    text = req.query if isinstance(req, RetrievalSearchRequest) else req.question
    return RetrievalQuery(
        text=text,
        top_k=req.top_k,
        fetch_k=req.fetch_k,
        where=req.where,
        time_range=retrieval_time_range(req.time_range),
        min_score=req.min_score,
        max_per_doc=req.max_per_doc,
        route=req.route,
    )


def retrieval_select(
    router: StoreRouter, route: str | None
) -> tuple[Retriever, RouteDecision]:
    """选路，并把检索层的拒绝翻成 400（消息原样带回，里面已经有可用路由名）.

    ``route`` 为 ``None`` 时走路由器的三条规则（默认路由 / 只有一个索引时用它 /
    多个且没默认路由则报错要求显式指定）；``route`` 不存在时
    ``routing.StoreRouter.resolve`` 的消息里**列出全部可用名**（三段式：
    现象 → 可用清单 → 出路），端点不再包装一层：包装一次就会把那份清单挤到
    第二层文本里，而它恰恰是读报的人要抄的那个东西。

    空路由器（0 个路由）在这里得到 ``IndexStateError``：它说的是
    "索引还没装配好，要先建索引"，与"route 名写错了"（``QueryError``）
    是两件事（修复人不同，见 ``errors`` 的分族依据），但两者对调用方的
    出路都是 400 + 消息，因此合成一条通道。
    """
    try:
        return router.resolve(route)
    except RetrievalError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def retrieval_run(
    router: StoreRouter, query: RetrievalQuery
) -> tuple[Retriever, RetrievalResult, RouteDecision]:
    """选路 + 检索，返回 ``(选中的检索器, 结果, 选路结论)``.

    检索走 ``StoreRouter.retrieve``（它会把选路结论追加进 ``result.notes``，
    于是"这份结果来自哪个路由"跟着结果一起走，端点不必自己往 notes 里补话——
    补话的写法一旦与路由层不一致，"同一次检索在脚本与端点里交代不同"就会发生）。
    选路结论再单独取一次给响应体回显：第二次 ``resolve`` 只是一次字典查找，
    没有 I/O、没有副作用。

    ``/retrieval/answer`` 不走本函数：``RagPipeline`` 自己持有检索器（它不认识
    路由表），因此那条链路的选路在端点层做完（见 ``retrieval_answer``）。
    """
    result = router.retrieve(query)
    retriever, decision = retrieval_select(router, query.route)
    return retriever, result, decision


def retrieval_llm(request: Request) -> BaseLLM:
    """/retrieval/answer 用的 LLM；"未配置"当场 400（**不拿 MockLLM 兜底**）.

    判据是**实际拿到的那个对象**，而不是 ``settings.openai_api_key`` 是否为空：
    注入路径可以绕过 settings（测试注入一个 ``BaseLLM`` 子类就是刻意的用法），
    那时"有没有密钥"与"这条链路拿到的模型是什么"会得出相反的结论——
    而后者才是答案的来源。

    ``MockLLM`` 被显式排除：它是 ``create_app`` 在"未配置密钥"时的退避实现
    （见 ``app.default_llm`` 的告警原文），而按脚本回复的占位实现会在
    **一个片段都没用上**的情况下给出一段语法正确、语气确定的答案。
    这正是本课护栏要拦的那类失败——它看起来最像成功。要在这条链路上跑通，
    请注入一个真实 ``BaseLLM``（离线测试注入自定义子类即可）。
    """
    llm = getattr(request.app.state, RETRIEVAL_LLM_STATE_KEY, None)
    if llm is None:
        llm = getattr(request.app.state, "llm", None)
    if llm is None:
        raise HTTPException(
            status_code=400,
            detail=f"本实例没有配置 LLM（app.state.{RETRIEVAL_LLM_STATE_KEY} 与 "
            "app.state.llm 都为空）：/retrieval/answer 要在检索到的片段上生成答案，"
            "没有模型就没有答案。出路：用 create_app(llm=...) 注入一个真实客户端。",
        )
    if isinstance(llm, MockLLM):
        raise HTTPException(
            status_code=400,
            detail="本实例的 LLM 是 MockLLM 占位实现（create_app 未注入 llm 且未配置"
            "密钥时的退避实现），/retrieval/answer **不接受**它：按脚本回复的占位实现"
            "会产出一段语法正确、看起来最像成功的答案，而它一个检索片段都没用上——"
            "这正是本课那条'检索不到不许让模型自由发挥'的护栏要拦的失败。"
            "出路：注入一个真实 BaseLLM（离线测试注入自定义子类即可），"
            "或在 app.state.retrieval_llm 上显式指定一个问答专用模型。",
        )
    return llm


@router.get("/retrieval/status", response_model=RetrievalStatusResponse)
def retrieval_status(request: Request, route: str | None = None) -> RetrievalStatusResponse:
    """当前检索器与索引状态（day066，``?route=`` 可选）.

    它把 ``Retriever.describe()`` 的九项原样端出来（名字 / 深度 / 过取倍率 /
    阈值 / 每文档上限 / 分组字段 / 时间字段 / 索引现状），再补三块**不属于
    检索器**的信息：

    ```text
    defaults       settings.retrieval_* 的当前取值 + 深度封顶 MAX_FETCH_K
                   ——"为什么是这个数"必须能被读出来，而不是靠翻配置
    router         有几条路、默认是哪一条、当前选中的是谁（含选它的理由）
    empty_reasons  四种空结果成因的人话解释（空结果不是"没找到"一句话）
    ```

    ``index`` 里最值得先读的三项：``version_id``（这一版是谁，空串 = 没有清单）、
    ``count_store`` / ``count_manifest``（库的条数与清单说的条数，相减即漂移规模）、
    ``drift``（逐条成因）。**本端点只报告、不因漂移报错**：``index_state`` 是
    观测入口（"现在漂成什么样了"是运维要看的），而 ``retrieve`` 才是取数入口
    （``strict_index=True`` 时它必须拒绝给出可能不完整的答案）。

    ``?route=`` 缺省走路由器的三条规则（默认路由 / 只有一个索引时用它）；
    多个索引且没设默认路由 → **400** 并列出可用名（"替调用方挑一个"会让
    "问源码库的问题去查了手册库"变成一个看起来完全正常的错误答案）。
    """
    retrieval_router_ = retrieval_router(request)
    retriever, decision = retrieval_select(retrieval_router_, route)
    describe = retriever.describe()
    return RetrievalStatusResponse(
        summary=retriever.index_state.summary_line(),
        retriever=describe,
        index=describe["index"],
        route=decision.to_dict(),
        router={
            "count": len(retrieval_router_),
            "default": retrieval_router_.default,
            "names": retrieval_router_.names(),
        },
        defaults=retrieval_defaults(),
        empty_reasons=dict(EMPTY_REASON_DESCRIPTIONS),
        limitations=list(RETRIEVAL_LIMITATIONS),
        out_of_scope=list(RETRIEVAL_OUT_OF_SCOPE),
    )


@router.post("/retrieval/search", response_model=RetrievalSearchResponse)
def retrieval_search(req: RetrievalSearchRequest, request: Request) -> RetrievalSearchResponse:
    """一句话 → 检索 → 一份完整的 ``RetrievalResult``（day066）.

    这个端点让"给一句话，取回哪几条"变成一次可复现的调用（缺省 flat 后端
    逐位可复现：同一个请求体在任何机器上给出同一份结果）。它的价值不在 ``hits``，
    而在**五个数字**一起把"为什么只有这几条"变成能直接读出来的事实：

    ```text
    fetch_k                  本来打算取多深（notes 里会写"按 3 倍本应取 N 条，已封顶"）
    candidates               库侧过滤之后还剩多少条可选
    dropped_below_threshold  阈值切掉几条 → 去调 settings.retrieval_min_score
    dropped_by_diversity     每文档上限挤掉几条 → 去调 settings.retrieval_max_per_doc
    dropped_by_top_k         截断到 top_k 丢掉几条 → 这只是"取够了"
    ```

    三条边界行为，都在这里写清楚：

    ```text
    空库                     → 200 且 empty_reason="no_data"（**合法状态**，不是 500）
    过滤后一条不剩           → 200 且 empty_reason="filtered_out"（候选为 0）
    阈值把命中全切了         → 200 且 empty_reason="below_threshold"
    ```

    参数非法一律 **400**，消息里带合法取值（top_k 的上限、支持的时间写法、
    where 支持的运算符、可用路由名都由检索层给出）。**维度不符与编码器坏向量
    也走 400**：那两种是"库与装配对不上"，消息里写着"请用同一个提供方重建索引"，
    而它们不是"这次请求碰巧不对"——本组因此不引入第三个状态码，
    调用方读消息即可（与 /indexing/* 把三族异常统一成 400 同源）。

    它**不做什么**：不写库、不重建索引、不做重排序（day068）、不做多路融合
    （day067）、不替调用方解释"这条为什么没被召回"（那是评估层的事）。
    """
    retrieval_router_ = retrieval_router(request)
    try:
        query = retrieval_query_of(req)
        retriever, result, decision = retrieval_run(retrieval_router_, query)
    except RetrievalError as exc:
        # 四族异常（参数 / 索引状态 / 打包配置）都继承自 RetrievalError，
        # 消息里已经带足出路，原样带回即可；认不出来的异常一律上抛成 500。
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RetrievalSearchResponse(
        summary=result.summary_line(),
        result=result.to_dict(include_text=True),
        lines=retriever.explain(result),
        conditions=describe_conditions(query.where, query.time_range),
        filter_fields=filter_fields(query.where),
        empty_reason=result.empty_reason,
        empty_reason_description=EMPTY_REASON_DESCRIPTIONS.get(result.empty_reason, ""),
        route=decision.to_dict(),
        routes=retrieval_router_.names(),
    )


@router.post("/retrieval/explain", response_model=RetrievalExplainResponse)
def retrieval_explain(
    req: RetrievalSearchRequest, request: Request
) -> RetrievalExplainResponse:
    """同一次检索 → 逐行人话诊断（day066）："把话说清楚"的出口.

    ``lines`` 固定回答四件事（顺序固定，因为读诊断的人要找的就是这四句）：

    ```text
    1) 这次查的是哪一版索引        → 版本号 + 库/清单条数 + 漂移项数
    2) 过滤条件是什么              → where + 时间范围 + 过滤后候选数
    3) 为什么条数少于 top_k        → 三个 dropped_* 数字（阈值/多样性/截断）
    4) 空结果的唯一原因            → empty_reason + 人话解释
    ```

    最后两行由 ``Retriever.explain`` 补上，它们**需要读库**，因此只能由检索器给出：
    字段拼写检查（拿"库里出现过哪些字段"的全集对照，能发现"整个库里都没有这个键"）、
    漂移处置建议（漂成什么样、下一步该做什么）。

    它与 ``/retrieval/search`` 的差别只有一处：**不返回正文**。
    分开而不是加一个 ``explain=true`` 开关，理由与 day065 的
    "/indexing/plan 是看、/indexing/build 是做"同源：一个返回体的形状
    只服务一个用途，开关会让客户端代码在两种形状之间分支。

    错误码与 ``/retrieval/search`` 完全一致（同一份请求体、同一条通道）。
    """
    retrieval_router_ = retrieval_router(request)
    try:
        query = retrieval_query_of(req)
        retriever, result, decision = retrieval_run(retrieval_router_, query)
    except RetrievalError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RetrievalExplainResponse(
        summary=result.summary_line(),
        empty_reason=result.empty_reason,
        empty_reason_description=EMPTY_REASON_DESCRIPTIONS.get(result.empty_reason, ""),
        lines=retriever.explain(result),
        count=result.count,
        ids=result.ids(),
        conditions=describe_conditions(query.where, query.time_range),
        route=decision.to_dict(),
        routes=retrieval_router_.names(),
    )


@router.post("/retrieval/answer", response_model=RetrievalAnswerResponse)
def retrieval_answer(req: RetrievalAnswerRequest, request: Request) -> RetrievalAnswerResponse:
    """问题 → 检索 → 打包 → 生成 → 引用（day066）：本课的 RAG 生成链路.

    **本端点最重要的行为是"检索为空时一次 LLM 都不调"**：
    ``RagPipeline.answer`` 把判空放在渲染提示词之前，于是空结果直接得到
    ``llm_called=False``、``answer=`` 兜底答复（带三条出路）、``citations=()``。
    这不是省一次调用，而是拦住 RAG 系统最贵的一种失败——
    **模型手里没有片段，却仍然写出一段语法正确、语气确定、引用格式也像模像样的
    答案**，而读的人看不出区别。

    **未配置 LLM 时返回 400**（消息说明原因），而且**不拿 MockLLM 兜底**：
    判据是"这条链路实际拿到的那个对象"（见 ``retrieval_llm``），
    因此缺省 app（退避成 MockLLM）会得到 400，而注入了真实客户端的 app 正常作答。
    这一条是**端点级**的：检索本身不需要模型，所以这条护栏只拦 answer。

    ```text
    空库 / 过滤后无命中 / 阈值切光 → 200 且 llm_called=False（**合法状态**，不是 400）
    LLM 未配置（或仍是 Mock 占位）  → 400，消息给出"注入一个真实 BaseLLM"的出路
    检索参数非法（top_k / 时间倒置 / 未知路由）→ 400，与 /retrieval/search 同一条通道
    ```

    选路在这里做完：``RagPipeline`` 只持有**一个**检索器（它不认识路由表），
    因此先 ``resolve`` 出该用哪一个，再把选路结论放进响应的 ``route`` 字段——
    答案不对时第一件要确认的事就是"这次该不该是它"。

    它**不做什么**：不写库、不缓存答案（语义缓存在 day046 的流水线里）、
    不做引用溯源校验（day069）、不做答案质量评估（day071）。
    """
    retrieval_router_ = retrieval_router(request)
    try:
        query = retrieval_query_of(req)
        retriever, decision = retrieval_select(retrieval_router_, query.route)
        pipeline = RagPipeline(retriever, retrieval_llm(request))
        answer = pipeline.answer(query)
    except RetrievalError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    retrieval = answer.retrieval
    return RetrievalAnswerResponse(
        summary=answer.summary_line(),
        answer=answer.answer,
        llm_called=answer.llm_called,
        prompt_version=pipeline.prompt_version,
        citations=[citation.to_dict() for citation in answer.citations],
        context=answer.context.to_dict() if answer.context is not None else None,
        retrieval=retrieval.to_dict() if retrieval is not None else None,
        notes=list(answer.notes),
        empty_reason=retrieval.empty_reason if retrieval is not None else "",
        route=decision.to_dict(),
    )


@router.get("/retrieval/routes", response_model=RetrievalRoutesResponse)
def retrieval_routes(request: Request) -> RetrievalRoutesResponse:
    """路由清单（day066）：每条路的名字、说明与它自己的现状.

    它回答"这个实例一共能问几个库、不传 route 会走哪一条"，并顺带把每条路的
    现状端出来（``retriever.describe()``：深度 / 阈值 / 索引版本 / 漂移）。
    三者缺一都会让"这一路为什么没结果"变得无法回答——而漂移恰恰是最常见的原因，
    它就在 ``routes[].retriever.index.drift`` 里。

    **本端点不受"多个索引且没有默认路由"的影响**：那条规则约束的是
    "替调用方挑一个"（``resolve``），而列清单本来就不需要挑。因此这里的
    ``default`` 可以是空串（如实报告"没设默认"），调用方随后必须显式传 route。

    空路由表（0 条路）→ **400** 且说明"要先建索引"：不是"没有东西可列"这种
    无害的空，而是"检索根本还没装配好"，返回一个 ``count=0`` 的 200 会让
    调用方以为"有一张空表"，而真正要做的是去建索引（或注入一个已注册的路由器）。

    它**不做什么**：不改路由、不落盘、不选路、不触发任何检索。
    """
    retrieval_router_ = retrieval_router(request)
    if len(retrieval_router_) == 0:
        raise HTTPException(
            status_code=400,
            detail="路由器里没有注册任何索引（0 条路由）：检索还没有可去的地方。"
            "出路：先用 day065 的 /indexing/build 把索引建起来（缺省装配会自动把"
            f"库注册成 {DEFAULT_RETRIEVAL_ROUTE!r} 这一条路由），"
            f"或往 app.state.{RETRIEVAL_ROUTER_STATE_KEY} 注入一个已注册的 StoreRouter。",
        )
    report = retrieval_router_.report()
    names = list(report["names"])
    return RetrievalRoutesResponse(
        summary=(
            f"{report['count']} 条路由 | 默认 {report['default'] or '（未设置）'} | "
            f"可用 {'、'.join(names) or '（无）'}"
        ),
        count=int(report["count"]),
        default=str(report["default"]),
        names=names,
        routes=list(report["routes"]),
        limitations=list(RETRIEVAL_LIMITATIONS),
        out_of_scope=list(RETRIEVAL_OUT_OF_SCOPE),
    )


# --------------------------------------------------------------------------- #
# day067 混合检索（M6-D6）：/retrieval/hybrid 与 /retrieval/lexical/status
#
# 这两条与上面五条**共用同一对注入对象**（``app.state.vector_store`` 与
# ``app.state.embedding``），因此问的是同一个库的第五、第六类问题：
#
# ```text
# 这句话同时用"字面"和"语义"去查，两个答案怎么合？  → /retrieval/hybrid
# 关键词那一路现在是什么状态、用了哪组参数？        → /retrieval/lexical/status
# ```
#
# 四条纪律，每一条都对应本段里的一个具体写法：
#
# 1. **导入期不构造、不读盘、不联网**（与上面五条同源）：本段只有常量、函数与
#    路由装饰器；混合检索器、关键词索引、清单全在**请求期**取或现装。
# 2. **端点只做"折请求体 → 调 retrieval → 序列化"**：深度、阈值、两路过滤、
#    融合、多样性、诊断全部留在 ``retrieval`` 包里（那里有它们的唯一实现）。
#    端点上重算一次融合会产出"接口与脚本给出的 top-1 不一样"这种差异。
# 3. **缺省关闭，而且拒绝时要说清是哪一个开关**：``retrieval_hybrid_enabled=False``
#    时返回 400（**不静默退回单路**）。静默退回会让"我明明启用了却看不出区别"
#    变成一个查不出病因的现象——而它本来只需要一句话就能解释。
# 4. **关键词索引的现状要能单独问**（``/retrieval/lexical/status``）：
#    关键词路一条都没召回时，第一个要确认的事实是"索引里到底有多少篇、词表多大"，
#    而这件事在检索响应里只能看到一个候选数。
#
# 为什么关键词索引**每次请求现建**：它与向量路共用同一个库，因此
# ``from_backend`` 是唯一能保证"两路看到的是同一批文档"的建法
# （见 ``LexicalIndex.from_backend``）。代价是 O(N) 的分词——
# 语料大时请把它建一次挂到 ``app.state.retrieval_lexical`` 上注入进来
# （与 ``app.state.retrieval_router`` 的多索引注入是同一条出路）。
# --------------------------------------------------------------------------- #

#: 混合检索器挂在 ``app.state`` 上的键名（可选）。**缺省不需要它**：
#: 开关打开时本段会用 ``app.state.vector_store`` + ``app.state.embedding`` +
#: 一份现建的关键词索引装配一个默认混合检索器。
RETRIEVAL_HYBRID_STATE_KEY = "retrieval_hybrid"

#: 关键词索引挂在 ``app.state`` 上的键名（可选）。注入它可以让"每次请求重分词"
#: 变成"每次请求一次字典查找"（大语料下这是唯一的可行做法）。
RETRIEVAL_LEXICAL_STATE_KEY = "retrieval_lexical"


def retrieval_lexical(request: Request) -> LexicalIndex:
    """取关键词索引：**注入优先**，否则按注入的库与 settings 的 BM25 参数现建一份.

    现建走 ``LexicalIndex.from_backend``——它与向量路的体检用的是同两个原语
    （``ids()`` + ``get_many()``），因此"关键词索引里有哪几篇"与"向量库里有哪些
    记录"是同一份事实的两次读取；而它**不做缓存**（与 ``retrieval_router``
    同一条理由：缓存会让"注入的后端被换掉"在某个请求上悄悄失效）。

    注入了一个非 ``LexicalIndex`` 的对象时给 500 而不是 400：那是装配写错了，
    不是请求参数写错了——把它降级成 400 会让一个必然复现的配置错误
    看起来像"这次请求碰巧不对"。
    """
    injected = getattr(request.app.state, RETRIEVAL_LEXICAL_STATE_KEY, None)
    if injected is not None:
        if not isinstance(injected, LexicalIndex):
            raise HTTPException(
                status_code=500,
                detail=f"app.state.{RETRIEVAL_LEXICAL_STATE_KEY} 必须是 LexicalIndex，"
                f"收到 {type(injected).__name__}：这是装配问题而不是请求参数问题。"
                "出路：用 LexicalIndex.from_backend(vector_store) 建一份再注入。",
            )
        return injected
    return LexicalIndex.from_backend(
        request.app.state.vector_store,
        params=BM25Params(
            k1=settings.retrieval_bm25_k1,
            b=settings.retrieval_bm25_b,
        ),
    )


def retrieval_hybrid(request: Request) -> HybridRetriever:
    """取混合检索器：**注入优先**；没有注入且开关关着 → **400**（不静默退回单路）.

    顺序是刻意的：先看注入，再看开关。注入一个 ``HybridRetriever`` 是
    "我明确要这条链路"的显式动作（测试就这么做），因此它不该被一个环境变量否掉；
    而**开关管的是缺省装配**——它是"这个实例要不要长出混合检索"的运维决定。

    拒绝时消息里给三样东西：开关名、当前值、出路。不给的话，调用方看到的
    只是一个 400，而"为什么不行"要去翻环境变量与配置源码。
    """
    injected = getattr(request.app.state, RETRIEVAL_HYBRID_STATE_KEY, None)
    if injected is not None:
        if not isinstance(injected, HybridRetriever):
            raise HTTPException(
                status_code=500,
                detail=f"app.state.{RETRIEVAL_HYBRID_STATE_KEY} 必须是 HybridRetriever，"
                f"收到 {type(injected).__name__}：这是装配问题而不是请求参数问题，"
                "把它当成 400 会让一个必然复现的配置错误看起来像一次偶然。",
            )
        return injected
    if not hybrid_enabled():
        raise HTTPException(
            status_code=400,
            detail="本实例没有启用混合检索（settings.retrieval_hybrid_enabled=False）："
            "这个开关缺省是关的，因为打开它会**改变结果集合**"
            "（关键词路会捞进向量路没召回的记录——那正是它存在的意义）。"
            "出路：把 retrieval_hybrid_enabled 设为 True（环境变量 "
            "RETRIEVAL_HYBRID_ENABLED=true），"
            f"或往 app.state.{RETRIEVAL_HYBRID_STATE_KEY} 注入一个 HybridRetriever。"
            "它**不会**静默退回单路：那样你会以为自己在看混合检索的结果，"
            "而实际上一次关键词检索都没发生。",
        )
    manifest, _source = retrieval_manifest(request)
    return build_hybrid_retriever(
        request.app.state.vector_store,
        request.app.state.embedding,
        manifest=manifest,
        lexical=retrieval_lexical(request),
    )


def hybrid_query_of(req: HybridRetrievalRequest) -> RetrievalQuery:
    """请求体 → ``RetrievalQuery``：把三个融合参数折进 ``extra``（day066 预留的槽）.

    为什么折进 ``extra`` 而不是给 ``HybridRetriever.retrieve`` 多加三个参数：
    ``extra`` 就是 day066 为**这一层**留的口子（见 ``types.RetrievalQuery`` 的
    docstring："融合参数（``{"alpha": 0.3, "k_rrf": 60}`` 之类）"）。
    走这条口子有三个好处：检索器的签名不变（端点、脚本、评估用的是同一个入口）、
    一份查询能被序列化回放（"这次用的是哪组参数"跟着查询走）、
    以及参数校验只有一份实现（``fusion.fuse``）。

    键名是封闭清单：多一个键会在检索层报 ``FusionError``（→ 400），
    而不是被静默忽略。
    """
    query = retrieval_query_of(req)
    extra = dict(query.extra)
    for key in ("strategy", "alpha", "k_rrf"):
        value = getattr(req, key)
        if value is not None:
            extra[key] = value
    return replace(query, extra=extra) if extra else query


@router.post("/retrieval/hybrid", response_model=HybridRetrievalResponse)
def retrieval_hybrid_search(
    req: HybridRetrievalRequest, request: Request
) -> HybridRetrievalResponse:
    """一句话 → **两路检索** → 融合 → 一份带多路证据的结果（day067）.

    它与 ``/retrieval/search`` 的关系值得写清楚：**同一个库、同一次查询意图、
    同一个 ``RetrievalResult`` 形状**，差别只在"取候选的那一步用了两路"。
    因此这里不重复解释那五个数字（``fetch_k`` / ``candidates`` / 三个 ``dropped_*``），
    只说明混合模式**多出来**的四块：

    ```text
    channel_candidates  每路候选数（{"vector": 10, "bm25": 1}）——"某一路空着"只能靠它看出来
    fusion              策略 / 参数 / 每路召回条数 / 每路贡献数 / 融合与去重条数
    lexical             关键词索引现状（篇数 / 词表 / avgdl / k1 / b）
    channels            本次结果里出现过的通道
    ```

    三条边界行为，每一条都对应一个**不同的**动作：

    ```text
    开关关着（缺省）        → 400，消息指出开关名与注入出路（**不静默退回单路**）
    某一路一条都没召回       → 200，notes 里点名是哪一路、为什么（词元不认识 / 被筛掉 / 索引空）
    阈值把向量路清空         → 200，关键词路照常给出它那几条，notes 说明 min_score 只作用于向量路
    ```

    融合参数（``strategy`` / ``alpha`` / ``k_rrf``）写进请求体即可逐次覆盖：
    它们被折进 ``RetrievalQuery.extra``，因此"这次用哪组参数"跟着查询一起被回显
    （``result.fusion.params``）。参数非法一律 400（消息里列出合法取值）。

    它**不做什么**：不写库、不重建关键词索引（那是装配或注入的事）、
    不重排序（day068）、不做多路以外的融合（例如按文档聚合）。
    """
    hybrid = retrieval_hybrid(request)
    try:
        query = hybrid_query_of(req)
        result = hybrid.retrieve(query)
    except RetrievalError as exc:
        # 参数问题（QueryError / FusionError）与索引问题（IndexStateError / LexicalError）
        # 都继承自 RetrievalError，消息里已经带足出路，原样带回即可；
        # 认不出来的异常一律上抛成 500（报告只收留我们认识的失败）。
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return HybridRetrievalResponse(
        summary=result.summary_line(),
        result=result.to_dict(include_text=True),
        lines=hybrid.explain(result),
        lexical=hybrid.lexical.describe(),
        channel_candidates=dict(result.channel_candidates),
        fusion=dict(result.fusion),
        empty_reason=result.empty_reason,
        empty_reason_description=EMPTY_REASON_DESCRIPTIONS.get(result.empty_reason, ""),
        conditions=describe_conditions(query.where, query.time_range),
        filter_fields=filter_fields(query.where),
        channels=list(result.channels),
        limitations=list(RETRIEVAL_LIMITATIONS),
        out_of_scope=list(RETRIEVAL_OUT_OF_SCOPE),
    )


@router.get("/retrieval/lexical/status", response_model=LexicalStatusResponse)
def retrieval_lexical_status(request: Request) -> LexicalStatusResponse:
    """关键词索引的现状（day067）：篇数 / 词表大小 / avgdl / k1 / b.

    它存在的理由是**诊断顺序**。关键词路一条都没召回时，要按这个顺序排除：

    ```text
    1. 索引是空的吗          → count（0 篇 = 还没建，与"语料里没有相关字面"是两件事）
    2. 词表有多大            → vocabulary_size（"交集为空"要有分母）
    3. 文档平均多长          → avgdl（它进 BM25 的分母，也是"长度归一化有多猛"的参照）
    4. 用的是哪组参数        → k1 / b + defaults（"结果不同"要能归因到显式参数）
    ```

    四个问题各自对应一个动作，因此四个数字一起给；而 ``defaults`` 里的六项
    来自 ``settings``，回答"这个 1.5 是项目默认还是谁改过"。

    它**不触发任何检索**、不写任何东西：这是一条纯粹的读端点
    （与 ``/retrieval/status`` 的 ``index`` 那一段同源，只是换了一路）。
    """
    index = retrieval_lexical(request)
    described = index.describe()
    return LexicalStatusResponse(
        summary=index.summary_line(),
        index=described,
        count=int(described["count"]),
        vocabulary_size=int(described["vocabulary_size"]),
        avgdl=float(described["avgdl"]),
        k1=float(described["k1"]),
        b=float(described["b"]),
        defaults={
            "bm25_k1": settings.retrieval_bm25_k1,
            "bm25_b": settings.retrieval_bm25_b,
            "hybrid_enabled": settings.retrieval_hybrid_enabled,
            "hybrid_strategy": settings.retrieval_hybrid_strategy,
            "hybrid_alpha": settings.retrieval_hybrid_alpha,
            "hybrid_rrf_k": settings.retrieval_hybrid_rrf_k,
        },
        limitations=list(RETRIEVAL_LIMITATIONS),
        out_of_scope=list(RETRIEVAL_OUT_OF_SCOPE),
    )
