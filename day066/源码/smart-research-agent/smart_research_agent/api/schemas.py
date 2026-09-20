"""API 请求/响应模型：服务的对外契约（day038）.

每个模型同时承担两个角色：
  - 运行时契约：FastAPI 在请求进入路由函数之前按这些模型解析并校验请求体，
    不合法的请求根本不会碰到业务代码，而是返回 422；
  - 文档源：这些模型被反射生成 JSON Schema，构成 /openapi.json 与
    /docs（Swagger UI）中接口文档的全部内容。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# 契约层唯一一处 import 项目内的常量：时间范围的字段名默认值。
#
# 为什么值得为它破一次例（本模块其余部分只依赖 pydantic）：
# ``RetrievalTimeRange.field`` 的默认值与 ``retrieval.filters.DEFAULT_TIME_FIELD``
# **必须是同一个值**——请求体里省略 ``field`` 时用的是这里，
# 而检索器把时间子句落到哪个元数据键上用的是那边。抄一份字面量之后，
# 改一处就能让"我明明没写字段名，为什么过滤的是另一个键"变成一个没有报错的怪事。
from smart_research_agent.retrieval import DEFAULT_TIME_FIELD


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
    """POST /chat 的响应体.

    day044 起 ``reply`` 为「输出审核 + 脱敏后」的最终文本，``moderation``
    始终返回审核报告——即使回复干净（is_safe=True）也带报告，让客户端
    契约稳定：字段永远存在，不因"是否命中"而时有时无。
    """

    reply: str = Field(description="模型回复（已做输出审核与 PII 脱敏）")
    moderation: ModerationInfo = Field(description="输出侧审核报告")


class ModerationInfo(BaseModel):
    """输出侧审核报告（day044）：回复是否安全、脱敏了什么.

    它把 ``security.ContentModerator.moderate`` 的结果原样投影到 HTTP 契约：
    ``is_safe`` 是"含不含敏感词或 PII"的布尔结论，``flagged_words`` 与
    ``pii_types`` 告诉客户端具体命中了什么——脱敏不是黑盒，审计可追踪。
    """

    is_safe: bool = Field(description="回复是否含敏感词或 PII（True=干净）")
    flagged_words: list[str] = Field(default_factory=list, description="命中的敏感词")
    pii_types: list[str] = Field(default_factory=list, description="被脱敏的 PII 类型")


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
    """POST /agent/run 的响应体：最终答案 + 可审计的执行轨迹.

    day044 起 ``answer`` 为输出审核脱敏后的文本，``moderation`` 报告
    审核结论——Agent 的输出同样不能豁免内容审核。
    """

    answer: str = Field(description="Agent 给出的最终答案（已做输出审核与脱敏）")
    steps: int = Field(description="本次任务消耗的 LLM 调用轮数")
    tool_calls: list[ToolCallRecord] = Field(
        default_factory=list, description="执行过程中的全部工具调用记录"
    )
    moderation: ModerationInfo = Field(description="输出侧审核报告")


class EmbeddingRequest(BaseModel):
    """POST /embeddings 的请求体（day041）：批量文本编码.

    ``texts`` 用列表而非单字符串入口，让客户端一次请求编码整批——
    对云端 embedding 来说就是一次 HTTP 而非 N 次。``max_length`` 限制
    单批条数，防止超大请求把编码端拖垮。
    """

    texts: list[str] = Field(
        min_length=1, max_length=32, description="待编码文本列表，非空且不超过 32 条"
    )


class EmbeddingResponse(BaseModel):
    """POST /embeddings 的响应体（day041）."""

    provider: str = Field(description="实际编码的提供方类名")
    dimension: int = Field(description="向量维度")
    vectors: list[list[float]] = Field(description="与请求 texts 顺序一一对应的向量")


class SimilarityRequest(BaseModel):
    """POST /embeddings/similarity 的请求体（day041）：两段文本比相似度."""

    text_a: str = Field(min_length=1, description="第一段文本")
    text_b: str = Field(min_length=1, description="第二段文本")


class SimilarityResponse(BaseModel):
    """POST /embeddings/similarity 的响应体（day041）."""

    similarity: float = Field(description="余弦相似度，值域 [-1, 1]")
    provider: str = Field(description="实际编码的提供方类名")


class ErrorResponse(BaseModel):
    """统一错误响应：未捕获异常经全局异常处理器转为此结构（HTTP 500）."""

    detail: str = Field(description="错误详情")
    error_type: str = Field(description="异常类名，便于客户端按类型分支处理")


class PipelineRunRequest(BaseModel):
    """POST /pipeline/run 的请求体（day046）：走完整的一体化流水线.

    ``system_prompt`` 可选：流水线把它拼进消息并以「系统提示词 + 任务」为
    缓存键，避免同一句问题在不同系统提示词下命中同一份答案。
    """

    task: str = Field(min_length=1, description="用户任务，不能为空")
    system_prompt: str | None = Field(
        default=None, description="可选系统提示词；缺省用流水线自身的配置"
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="采样温度")


class StageInfo(BaseModel):
    """流水线中一个阶段的执行记录（day046）."""

    name: str = Field(description="阶段名：guard/cache/generate/account/moderate/store")
    duration_ms: float = Field(description="该阶段耗时（毫秒）")
    detail: str = Field(default="", description="人类可读的阶段细节（命中/未命中/模型名等）")


class InjectionInfo(BaseModel):
    """输入侧护栏的扫描结论（day046）."""

    is_injection: bool = Field(description="是否命中注入规则")
    matched_patterns: list[str] = Field(
        default_factory=list, description="命中的规则名（可用于审计与误报调优）"
    )


class PipelineRunResponse(BaseModel):
    """POST /pipeline/run 的响应体（day046）：答复 + 完整的阶段账.

    与 ``/chat`` 的区别：``/chat`` 只回答"说了什么"，本接口额外回答
    "这次请求经过了哪些阶段、各花多久、花了多少钱、有没有被打回"。
    ``model`` 为 ``"cache"`` 表示缓存命中（未调用模型）、``"blocked"``
    表示被输入侧护栏拦截。
    """

    reply: str = Field(description="最终答复（已输出审核与 PII 脱敏）")
    model: str = Field(description="实际生效的模型名；cache=缓存命中，blocked=被拦截")
    cached: bool = Field(description="是否命中语义缓存")
    blocked: bool = Field(description="是否被输入侧护栏拦截")
    cost_usd: float = Field(description="本次请求的新增费用（美元），差值为单次归因")
    prompt_tokens: int = Field(description="本次请求的输入 token 数")
    completion_tokens: int = Field(description="本次请求的输出 token 数")
    total_ms: float = Field(description="流水线总耗时（毫秒）")
    stages: list[StageInfo] = Field(default_factory=list, description="各阶段执行记录")
    moderation: ModerationInfo = Field(description="输出侧审核报告")
    injection: InjectionInfo | None = Field(
        default=None, description="输入侧护栏报告；未装配护栏时为 null"
    )


class PerfBaselineResponse(BaseModel):
    """GET /pipeline/baseline 的响应体（day046）：已存档的性能基线.

    直接投影 ``evaluation.perf_baseline.PerfBaseline``——``latency_ms`` 与
    ``p95_latency_ms`` 均含 ``"total"`` 键（整体耗时）与各阶段键。
    """

    label: str = Field(description="基线标签")
    samples: int = Field(description="采集样本数")
    latency_ms: dict[str, float] = Field(
        default_factory=dict, description="各阶段（含 total）平均耗时（毫秒）"
    )
    p95_latency_ms: dict[str, float] = Field(
        default_factory=dict, description="各阶段（含 total）P95 耗时（毫秒）"
    )
    mean_tokens: float = Field(description="平均 token 数")
    mean_cost_usd: float = Field(description="平均费用（美元）")
    mean_stages: float = Field(description="平均走过的阶段数")


class FinetuneMethodInfo(BaseModel):
    """一种微调方法的结构化画像（M5-D1）：/finetune/methods 的元素.

    投影自 ``finetune.overview.MethodProfile``——方法知识在代码里是常量
    （``METHODS``），API 只是把它原样交出来，避免"文档写一套、代码另一套"。
    """

    key: str = Field(description="方法标识：sft/lora/qlora/dpo/rlhf-ppo")
    full_name: str = Field(description="方法全称（中英文对照）")
    family: str = Field(
        description="方法家族：parameter-efficient / full-parameter / alignment"
    )
    data_need: str = Field(description="经验数据需求量级")
    trainable_ratio_hint: str = Field(description="可训练参数比例的定性描述")
    stability: str = Field(description="训练稳定性与主要风险")
    best_for: str = Field(description="适用场景")
    artifacts: list[str] = Field(
        default_factory=list, description="训练产物形态（需保存/部署的文件）"
    )
    notes: str = Field(default="", description="补充说明（工程代价、推理开销等）")


class FinetuneMethodsResponse(BaseModel):
    """GET /finetune/methods 的响应体（M5-D1）：五种微调方法的总览."""

    methods: list[FinetuneMethodInfo] = Field(description="全部方法画像，顺序稳定")


class DatasetValidateRequest(BaseModel):
    """POST /finetune/dataset/validate 的请求体（M5-D1）：校验一批原始样本.

    ``examples`` 是**未解析的原始字典**而非已规范化的结构：校验接口的价值
    正在于"先告诉我这批数据能不能用"，若要求调用方先自己解析，就等于把
    要检验的工作提前做了一遍。``min_length=1`` 让空列表直接 422——
    "校验零条数据"不是一个有意义的请求。
    """

    format: str = Field(default="alpaca", description="样本格式：alpaca/chat/prompt-completion")
    examples: list[dict] = Field(
        min_length=1, description="待校验的原始样本列表，至少 1 条"
    )


class DatasetIssue(BaseModel):
    """一条不合格样本的诊断结论（索引 + 原因）."""

    index: int = Field(description="样本在请求 examples 中的下标（从 0 开始）")
    reason: str = Field(description="不合格的原因（人类可读）")


class DatasetValidateResponse(BaseModel):
    """POST /finetune/dataset/validate 的响应体（M5-D1）.

    ``issues`` 只保留前若干条（见 routes.MAX_DATASET_ISSUES）：一批数据可能
    有上万条坏样本，全量返回既撑爆响应体，对排错也没有额外价值——首条
    错误通常已经说明了问题模式。
    """

    total: int = Field(description="提交的样本总数")
    valid: int = Field(description="通过校验的样本数")
    invalid: int = Field(description="未通过校验的样本数")
    issues: list[DatasetIssue] = Field(
        default_factory=list, description="不合格样本的诊断（最多前 20 条）"
    )
    stats: dict = Field(default_factory=dict, description="合格样本的数据集画像")


class DatasetStatsResponse(BaseModel):
    """GET /finetune/dataset/stats 的响应体（M5-D1）：当前数据集画像."""

    stats: dict = Field(default_factory=dict, description="数据集画像（含来源分布）")


class SFTDefaultsResponse(BaseModel):
    """GET /finetune/sft/defaults 的响应体（M5-D2）：SFT 默认超参与依赖.

    ``training_arguments`` 里的字段名与 ``transformers.TrainingArguments``
    逐字对齐，因此可以**直接展开**成构造参数；``sft_specific`` 放的是
    ``TrainingArguments`` 不认、而 SFT 又必须的那几项（``max_length`` /
    ``truncation`` / ``ignore_index``）。
    """

    training_arguments: dict = Field(
        default_factory=dict, description="与 TrainingArguments 同名的全部超参"
    )
    sft_specific: dict = Field(
        default_factory=dict, description="SFT 专属项：max_length / truncation / ignore_index"
    )
    base_model: str = Field(description="生成脚本默认使用的基座模型")
    hf_dependencies: list[str] = Field(
        default_factory=list, description="transformers + Trainer 路径所需依赖（含版本下限）"
    )
    trl_dependencies: list[str] = Field(
        default_factory=list, description="TRL SFTTrainer 路径所需依赖"
    )
    install: dict = Field(default_factory=dict, description="可复制的安装命令（pip / uv）")


class SFTPlanRequest(BaseModel):
    """POST /finetune/sft/plan 的请求体（M5-D2）：算一次训练的派生量.

    三项全部可选：都不传时用**配置里的默认超参 + 真实数据集**（day048
    落盘的 30 / 7 条），因此"想看看当前配置会跑多少步"只需发一个空对象。
    ``overrides`` 的键名与 ``SFTTrainingArgs`` 的字段名一致，未知键会被
    忽略（向前兼容），因此前端可以放心地传它不认识的字段。
    """

    overrides: dict = Field(
        default_factory=dict, description="超参覆盖（键名同 SFTTrainingArgs）"
    )
    train_size: int | None = Field(
        default=None, ge=1, description="训练集条数；留空则用真实数据集"
    )
    eval_size: int | None = Field(
        default=None, ge=0, description="评估集条数；留空则用真实数据集"
    )


class SFTPlanResponse(BaseModel):
    """POST /finetune/sft/plan 的响应体（M5-D2）：派生量 + 风险告警.

    ``length`` 与 ``suggested_max_length`` 一起返回，是为了让"``max_length``
    该取多少"这个判断**可被复核**：调用方能看到 p50/p90/p95/max，再决定
    是接受建议值还是按自己的口径取。
    """

    training_arguments: dict = Field(
        default_factory=dict, description="本次生效的超参（与 TrainingArguments 同名）"
    )
    plan: dict = Field(default_factory=dict, description="派生量：步数/warmup/有效批…")
    warnings: list[str] = Field(
        default_factory=list, description="风险告警（每条对应一个失败模式）"
    )
    length: dict = Field(default_factory=dict, description="渲染后的 token 长度分布")
    suggested_max_length: int = Field(description="按长度分位数给出的 max_length 建议值")
    suggested_quantile: float = Field(description="建议值所用的分位（缺省 0.95）")


class SFTPreviewRequest(BaseModel):
    """POST /finetune/sft/preview 的请求体（M5-D2）：预览一条样本的监督切分.

    ``example`` 是**未解析的原始字典**（与 day048 ``/finetune/dataset/validate``
    同一取舍）：预览的价值正在于"先告诉我这条样本会变成什么样"，若要求
    调用方先自己渲染，等于把要检验的工作提前做了一遍。
    """

    example: dict = Field(description="原始 alpaca 样本：instruction / input / output")
    template: str = Field(default="chatml", description="chat 模板：chatml / llama3 / plain")
    max_length: int = Field(default=384, ge=2, description="单条样本的最大 token 数")


class SFTPreviewResponse(BaseModel):
    """POST /finetune/sft/preview 的响应体（M5-D2）.

    ``prompt_fully_masked`` 是本接口最有价值的一个字段：它是"label mask 真的
    生效了吗"的**可执行答案**。只要它为 ``false``，这条样本就不该进训练集。
    """

    template: str = Field(description="实际使用的模板")
    total_chars: int = Field(description="渲染后总字符数")
    prompt_chars: int = Field(description="前缀（system + user）字符数")
    supervised_chars: int = Field(description="监督区间字符数")
    total_tokens: int = Field(description="编码后总 token 数")
    prompt_tokens: int = Field(description="前缀 token 数")
    supervised_tokens: int = Field(description="参与 loss 的 token 数")
    masked_tokens: int = Field(description="被屏蔽的 token 数（不产生梯度）")
    supervised_ratio: float = Field(description="监督 token 占比")
    truncated: bool = Field(description="是否发生截断（答案完整时为 false）")
    prompt_fully_masked: bool = Field(
        description="前缀段的 label 是否全部为 -100（必须为 true）"
    )
    text_preview: str = Field(description="渲染文本前 200 字")
    supervised_preview: str = Field(description="监督区间前 120 字")


class LoRADefaultsResponse(BaseModel):
    """GET /finetune/lora/defaults 的响应体（M5-D3）：LoRA/QLoRA 默认配置与依赖.

    ``lora`` / ``quantization_config`` 里的字段名分别与 ``peft.LoraConfig`` /
    ``transformers.BitsAndBytesConfig`` **逐字对齐**，因此可以**直接展开**成
    构造参数，不需要一张"我们的字段 → 库字段"的映射表。

    ``reference_model`` 是本课程参考模型（557×557 bigram）的实测 LoRA 计划：
    它是"参数高效到底高效到什么程度"的第一手数字（``r=8`` 时 **2.787%**）。
    """

    lora: dict = Field(default_factory=dict, description="与 peft.LoraConfig 同名的配置")
    scaling: float = Field(default=0.0, description="实际生效的缩放 alpha/r（或 alpha/sqrt(r)）")
    scaling_formula: str = Field(default="", description="缩放的可读公式")
    quantization_config: dict = Field(
        default_factory=dict, description="与 BitsAndBytesConfig 同名的 4-bit 配置"
    )
    quantization: dict = Field(
        default_factory=dict, description="量化派生量：每参数位数/字节数与公式输入"
    )
    quantization_block_table: list[dict] = Field(
        default_factory=list, description="块大小 → 每参数存储的对照表"
    )
    target_presets: dict = Field(
        default_factory=dict, description="目标模块预设名 → 模块名后缀列表"
    )
    reference_model: dict = Field(
        default_factory=dict, description="参考模型上的 LoRA 计划（参数量与占比）"
    )
    base_model: str = Field(default="", description="生成脚本使用的缺省基座")
    peft_dependencies: list[str] = Field(default_factory=list, description="LoRA 脚本依赖")
    qlora_dependencies: list[str] = Field(default_factory=list, description="QLoRA 脚本依赖")
    install: dict = Field(default_factory=dict, description="可复制粘贴的安装命令")


class LoRAPlanRequest(BaseModel):
    """POST /finetune/lora/plan 的请求体（M5-D3）：算一次 LoRA 的参数量.

    ``model`` 必须是 ``peft.targets.MODEL_SPECS`` 里的规格名（规格里的每个
    数字都取自公开 ``config.json``，并附带参数量自检）。``overrides`` 的键名
    与 ``LoRAConfig`` 的字段名一致，未知键忽略（向前兼容）。
    """

    model: str = Field(default="llama-2-7b", description="模型规格名（见 MODEL_SPECS）")
    overrides: dict = Field(
        default_factory=dict, description="LoRA 配置覆盖（键名同 LoRAConfig）"
    )


class LoRAPlanResponse(BaseModel):
    """POST /finetune/lora/plan 的响应体（M5-D3）：参数量 + 预设/秩对照 + 告警.

    ``presets`` 与 ``rank_table`` 一起返回，是为了让"``target_modules`` 与 ``r``
    该怎么选"这个判断**可被复核**：同一份数据、同一个 ``r``，不同预设的
    参数量相差 4.8 倍；而参数量对 ``r`` 是严格线性的——容量收益与参数开销
    从来不成比例。
    """

    model: dict = Field(default_factory=dict, description="模型规格（含参数量自检值）")
    plan: dict = Field(default_factory=dict, description="LoRA 参数量计划")
    warnings: list[str] = Field(default_factory=list, description="风险告警")
    presets: list[dict] = Field(
        default_factory=list, description="目标预设对照（同一 r 下的参数量差异）"
    )
    rank_table: list[dict] = Field(
        default_factory=list, description="秩 → 参数量/占比对照（严格线性）"
    )


class LoRAMemoryRequest(BaseModel):
    """POST /finetune/lora/memory 的请求体（M5-D3）：算三种策略的显存预算."""

    model: str = Field(default="llama-2-7b", description="模型规格名（见 MODEL_SPECS）")
    overrides: dict = Field(default_factory=dict, description="LoRA 配置覆盖")
    qlora_overrides: dict = Field(default_factory=dict, description="QLoRA 配置覆盖")
    optimizer: str = Field(
        default="adamw_torch", description="优化器（adamw_torch / adamw_bnb_8bit / sgd …）"
    )
    compute_dtype: str = Field(default="bfloat16", description="计算精度")


class LoRAMemoryResponse(BaseModel):
    """POST /finetune/lora/memory 的响应体（M5-D3）：全参 / LoRA / QLoRA 的显存对照.

    ``devices`` 里的 ``fits`` **只比较权重 + 梯度 + 优化器状态，不含激活显存**。
    把这条写进字段而不是免责声明里，是因为"能放下"与"能跑起来"是两件事，
    而预算表最容易被误读成后者的答案。
    """

    model: dict = Field(default_factory=dict, description="模型规格")
    strategies: list[dict] = Field(
        default_factory=list, description="三种策略的汇总（含相对全参的节省倍数）"
    )
    details: list[dict] = Field(default_factory=list, description="逐项明细（六项字节数）")
    devices: list[dict] = Field(default_factory=list, description="单卡适配表（不含激活）")
    quantization_note: str = Field(default="", description="量化存储的一句话说明")
    optimizer: str = Field(default="", description="本次使用的优化器")
    compute_dtype: str = Field(default="", description="本次使用的计算精度")


class LoRADistributedRequest(BaseModel):
    """POST /finetune/lora/distributed 的请求体（M5-D4）：多卡/混合精度计划.

    ``devices=1`` 时所有分片策略都退化为 ddp；``train_size`` 留空则用真实
    数据集（day048 落盘的 30 条训练样本）。
    """

    model: str = Field(default="llama-2-7b", description="模型规格名（见 MODEL_SPECS）")
    devices: int = Field(default=4, ge=1, le=1024, description="训练设备数")
    strategy: str = Field(default="ddp", description="分片策略：ddp / zero2 / zero3")
    mixed_precision: str = Field(default="bf16", description="混合精度：no / fp16 / bf16")
    lr_scaling_mode: str = Field(
        default="linear", description="批大小放大后学习率的缩放法则：linear / sqrt / none"
    )
    optimizer: str = Field(default="adamw_torch", description="优化器")
    full_finetune: bool = Field(default=False, description="True 时按全参微调算（默认 LoRA）")
    train_size: int | None = Field(default=None, ge=1, description="训练集条数；留空用真实数据集")
    overrides: dict = Field(default_factory=dict, description="LoRA 配置覆盖")
    qlora_overrides: dict = Field(default_factory=dict, description="非空即按 QLoRA 算")


class LoRADistributedResponse(BaseModel):
    """POST /finetune/lora/distributed 的响应体（M5-D4）.

    ``config_yaml`` 与 ``deepspeed_config`` 都是**可以被 accelerate / DeepSpeed
    直接消费**的配置（字段名与官方一致）。把它们当成数据返回，而不是让使用者
    去跑一遍交互式 ``accelerate config``——**交互式问答产出的配置没法被 review**。
    """

    model: dict = Field(default_factory=dict, description="模型规格")
    plan: dict = Field(default_factory=dict, description="派生量：步数/全局批/每设备显存/通信")
    config_yaml: str = Field(default="", description="accelerate 配置（YAML 文本）")
    accelerate_config: dict = Field(default_factory=dict, description="同一份配置的字典形式")
    deepspeed_config: dict | None = Field(default=None, description="ZeRO 配置（ddp 时为 None）")
    launch_command: str = Field(default="", description="accelerate launch 命令行")
    device_fit: dict = Field(default_factory=dict, description="按单卡预算判定的适配结果")
    warnings: list[str] = Field(default_factory=list, description="风险告警")


class LoRADeployRequest(BaseModel):
    """POST /finetune/lora/deploy 的请求体（M5-D4）：合并与部署计划.

    ``train_steps=0``（缺省）时用一个**未训练**的适配器（``ΔW = 0``）走完整条
    流程——此时合并验证必然通过，因此响应里会明确标注 ``adapter_trained=false``。
    要一份"真实的"验证结果就传一个正数（参考模型上每步约十几毫秒）。
    """

    model: str = Field(default="llama-2-7b", description="基座规格名（用于体积对照）")
    train_steps: int = Field(default=0, ge=0, le=200, description="参考模型上先跑多少步")
    overrides: dict = Field(default_factory=dict, description="LoRA 配置覆盖")
    tags: dict = Field(default_factory=dict, description="写进清单的标签（业务/环境）")


class LoRADeployResponse(BaseModel):
    """POST /finetune/lora/deploy 的响应体（M5-D4）.

    ``verification`` 是**部署门禁**：它把 day051"合并前后逐位一致"那条不变式
    从单元测试搬到了交付流程里。``passed=false`` 时不应该上线。
    """

    reference: dict = Field(default_factory=dict, description="参考模型上的参数量账")
    manifest: dict = Field(default_factory=dict, description="适配器清单（含内容哈希）")
    verification: dict = Field(default_factory=dict, description="合并验证结果（门禁）")
    deployment: dict = Field(default_factory=dict, description="两种部署形态的体积对照")
    registry: list[dict] = Field(
        default_factory=list, description="适配器注册表（一个基座 + N 个适配器）"
    )
    adapter_files: list[str] = Field(default_factory=list, description="适配器目录里的三个文件")
    merge_script_lines: int = Field(default=0, description="合并脚本行数")
    inference_script_lines: int = Field(default=0, description="推理脚本行数")
    adapter_trained: bool = Field(default=False, description="本次流程里的适配器是否训练过")



class FinetuneEvalSuiteResponse(BaseModel):
    """GET /finetune/eval/suite 的响应体（M5-D5）：领域评估集的画像.

    ``audit`` 是难度体检：列出"声明难度 ≠ 结构负载推导难度"的用例。它不抛
    异常只报清单——**难度标注偏差会让结论变弱，但不会让流程崩掉**。

    ``leakage`` 是训练/评估切分的 n-gram 重叠体检：评估集被训练集污染时
    分数会虚高，而且虚高恰好集中在你最想发现问题的那几条用例上。
    """

    suite: dict = Field(default_factory=dict, description="评估集统计画像（含指纹）")
    audit: list[dict] = Field(default_factory=list, description="难度体检发现的问题")
    buckets: list[dict] = Field(default_factory=list, description="能力桶与含义")
    component_names: list[str] = Field(default_factory=list, description="六个分量的名字")
    weights: dict = Field(default_factory=dict, description="分量权重快照（和必须为 1）")
    train_ids: list[str] = Field(default_factory=list, description="切分后的训练子集 id")
    eval_ids: list[str] = Field(default_factory=list, description="切分后的评估子集 id")
    eval_difficulty: dict = Field(default_factory=dict, description="评估子集的难度分布")
    leakage: dict = Field(default_factory=dict, description="泄漏体检结果")
    items: list[dict] = Field(default_factory=list, description="全部用例（含事实点与格式契约）")


class FinetuneEvalRunRequest(BaseModel):
    """POST /finetune/eval/run 的请求体（M5-D5）：跑一次完整的微调评估.

    ``probe=true``（缺省）时会额外在参考模型上**真的训练一次 LoRA**，
    再做白盒探针——它给的是"模型对领域答案有多熟"这个内部信号，
    与文本侧指标是两件事，这也是本端点唯一会花掉秒级 CPU 时间的地方。
    """

    eval_ratio: float = Field(
        default=0.25, ge=0.05, le=0.9, description="评估套件的评估集占比（按桶分层）"
    )
    seed: int = Field(default=42, ge=0, description="切分与自助法的随机种子")
    min_pass_rate: float = Field(
        default=0.5, ge=0.0, le=1.0, description="合格率门禁下限"
    )
    max_regression: float = Field(
        default=0.0, ge=0.0, le=1.0, description="每个分组允许的最大合格率回退"
    )
    bootstrap_samples: int = Field(
        default=2000, ge=100, le=20000, description="配对自助法重采样次数"
    )
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0, description="显著性水平")
    probe: bool = Field(default=True, description="是否做参考模型上的白盒探针")
    probe_epochs: int = Field(default=10, ge=1, le=100, description="探针侧 LoRA 的训练轮数")
    probe_learning_rate: float = Field(
        default=2.0, gt=0.0, le=100.0, description="探针侧 LoRA 的学习率（本课标定值）"
    )


class FinetuneEvalRunResponse(BaseModel):
    """POST /finetune/eval/run 的响应体（M5-D5）.

    ``report.gates`` 是三项门禁（``min_pass_rate`` / ``no_regression`` /
    ``execution``）；``report.passed`` 只有在三项全通过时才为真。
    ``report`` 里同时带着**适配器内容哈希与评估集指纹**——没有绑定信息的
    分数不是证据。
    """

    suite: dict = Field(default_factory=dict, description="评估集统计画像（含指纹）")
    audit: list[dict] = Field(default_factory=list, description="难度体检发现的问题")
    leakage: dict = Field(default_factory=dict, description="泄漏体检结果")
    train_ids: list[str] = Field(default_factory=list, description="训练子集 id")
    eval_ids: list[str] = Field(default_factory=list, description="评估子集 id")
    comparison: dict | None = Field(default=None, description="两臂配对比较（含门禁）")
    probes: dict | None = Field(default=None, description="白盒探针（含泛化间隙）")
    report: dict = Field(default_factory=dict, description="评估报告（含门禁与失败清单）")


class AlignmentDimensionsResponse(BaseModel):
    """GET /finetune/alignment/dimensions 的响应体（M5-D6）：对齐的"要买什么".

    这个端点**不跑模型**，它回答的是开工前的三个问题：偏好哪几个维度、
    要多少条样本（按统计功效算出来的表）、以及这一步有哪些已知风险。
    """

    dimensions: list[dict] = Field(default_factory=list, description="五个对齐维度与目标")
    sample_budget: list[dict] = Field(
        default_factory=list, description="目标准确率 → 每个维度需要的偏好对数"
    )
    objectives: list[dict] = Field(
        default_factory=list, description="RLHF（PPO）与 DPO 的需求对照表"
    )
    risks: list[dict] = Field(default_factory=list, description="已知风险与缓解手段")
    notes: list[str] = Field(default_factory=list, description="开工前的流程建议")
    seed_stats: dict = Field(default_factory=dict, description="种子偏好数据的统计画像")


class AlignmentRunRequest(BaseModel):
    """POST /finetune/alignment/run 的请求体（M5-D6）：跑一次最小 DPO 对齐.

    ``epochs`` 与 ``learning_rate`` 的缺省值是本课在参考模型上标定的；
    调大它们可以观察"继续训练买到了什么"（本课实测：KL 涨 54 倍而留出
    准确率一点没涨）。
    """

    beta: float = Field(default=0.1, gt=0.0, le=10.0, description="DPO 的温度 β")
    learning_rate: float = Field(default=0.5, gt=0.0, le=50.0, description="DPO 学习率")
    epochs: int = Field(default=6, ge=1, le=200, description="训练轮数")
    valid_ratio: float = Field(
        default=0.25, ge=0.05, le=0.8, description="留出偏好对占比（按维度分层）"
    )
    kl_budget: float = Field(default=0.5, gt=0.0, description="KL 预算（过优化告警阈值）")
    target_accuracy: float = Field(
        default=0.6, gt=0.5, lt=1.0, description="样本量目标：偏好准确率要分辨出的水平"
    )
    seed: int = Field(default=42, ge=0, description="切分与训练顺序的随机种子")


class AlignmentRunResponse(BaseModel):
    """POST /finetune/alignment/run 的响应体（M5-D6）.

    ``initial_check_passed`` 是**起点自检**：未训练时策略与参考模型逐位相同，
    DPO loss 必须等于 ``ln 2``（``0.693147``）。这一条能抓住"ref 或 β 接错了"
    这类接线事故——它们在训练日志里只表现为"loss 有点怪"。
    """

    stats: dict = Field(default_factory=dict, description="偏好数据统计画像（含指纹）")
    train_ids: list[str] = Field(default_factory=list, description="训练偏好对 id")
    valid_ids: list[str] = Field(default_factory=list, description="留出偏好对 id")
    plan: dict = Field(default_factory=dict, description="对齐计划（含样本量差距）")
    reference_hash: str = Field(default="", description="参考模型参数哈希（前 12 位）")
    initial_loss: float = Field(default=0.0, description="训练前的 DPO loss（应为 ln 2）")
    zero_margin_loss: float = Field(default=0.0, description="ln 2 的参考值")
    initial_check_passed: bool = Field(default=False, description="起点自检是否通过")
    before: dict = Field(default_factory=dict, description="训练前的偏好准确率")
    after: dict = Field(default_factory=dict, description="训练后的偏好准确率")
    accuracy_gain: float = Field(default=0.0, description="留出准确率的绝对提升")
    report: dict = Field(default_factory=dict, description="DPO 训练报告（含逐步历史）")
    optimization: dict = Field(default_factory=dict, description="过优化体检（三条判据）")


# --------------------------------------------------------------------------- #
# day057 领域数据准备与增强（M5-D8）：/data/domain/* 三个端点
# --------------------------------------------------------------------------- #


class DomainDimensionsResponse(BaseModel):
    """GET /data/domain/dimensions 的响应体（M5-D8）：质量五维 + 增强算子的对照表.

    这个端点**不读数据、不跑流水线**，它回答的是开工前的三个问题：
    质量分是怎么算的（哪五维、各占多少权重、门槛取多少）、增强能做哪几种、
    以及一条流水线会按什么顺序经过哪些阶段。

    三个字段刻意**从代码现场读出**而不是抄写：``dimensions`` 来自
    ``quality.dimension_table()``（权重列由 ``QualityWeights`` 反射）、
    ``augment_ops`` 来自 ``augment.augment_ops_table()``、
    ``stage_order`` 来自 ``pipeline.STAGE_ORDER``。因此"文档说 0.25、
    代码里是 0.2"这类漂移会在接口上一眼看出来。
    """

    dimensions: list[dict] = Field(
        default_factory=list,
        description="五维质量对照表（维度/含义/计算/挡住的故障/缺省权重）",
    )
    weights: dict = Field(default_factory=dict, description="五维缺省权重（和恒为 1.0）")
    threshold: float = Field(description="质量门槛：加权总分低于它即被拒")
    augment_ops: list[dict] = Field(
        default_factory=list, description="增强算子对照表（含风险列与默认启用列）"
    )
    default_ops: list[str] = Field(
        default_factory=list, description="缺省启用的增强算子（noise 刻意不在其中）"
    )
    stage_order: list[str] = Field(
        default_factory=list, description="六阶段的固定顺序：顺序即策略"
    )


class DomainAugmentRequest(BaseModel):
    """POST /data/domain/augment 的请求体（M5-D8）：对一批样本做增强.

    ``examples`` 是**未解析的原始字典**（与 day048 ``/finetune/dataset/validate``
    同一取舍、同一解析函数 ``parse_example``）：增强的价值正在于"先告诉我
    这批样本能变成什么样"，若要求调用方先自己解析，等于把要检验的工作
    提前做了一遍。

    ``ops`` 留空即用 ``DEFAULT_OPS``（``noise`` 不在其中，它默认关闭）；
    算子名写错返回 400 而不是静默少生成样本——静默的错误比响亮的错误危险。
    """

    format: str = Field(default="alpaca", description="样本格式：alpaca/chat/prompt-completion")
    examples: list[dict] = Field(
        min_length=1, description="待增强的原始样本列表，至少 1 条"
    )
    ops: list[str] = Field(
        default_factory=list,
        description="增强算子名（prefix/constraint/paraphrase/noise）；留空用 DEFAULT_OPS",
    )
    max_per_example: int = Field(
        default=1, ge=0, description="每条原样本最多产出几个增强样本"
    )


class DomainAugmentResponse(BaseModel):
    """POST /data/domain/augment 的响应体（M5-D8）：增强样本 + 增强报告.

    ``examples`` 只放**新增**的增强样本（不含原样本），每条带
    ``instruction``/``output``/``tags``：``output`` 与入参**逐字相同**
    ——增强只改 prompt 侧，动答案就等于从"换一种问法"变成"生成新答案"。
    ``tags`` 追加了 ``aug:<算子名>``，因此"这条是合成的"永远可查。
    """

    inputs: int = Field(description="提交的原样本数")
    generated: int = Field(description="新增的增强样本数（= len(examples)）")
    expanded: int = Field(description="原样本 + 增强样本的总条数")
    examples: list[dict] = Field(
        default_factory=list, description="新增的增强样本（instruction/output/tags 等）"
    )
    report: dict = Field(
        default_factory=dict,
        description="增强报告：进出/产出/守门/无变化/撞车五本账与算子分布",
    )


class DomainRunRequest(BaseModel):
    """POST /data/domain/run 的请求体（M5-D8）：用仓库内置数据源跑一次六阶段流水线.

    八个字段**全部可选**：留空即用 ``settings.domain_*`` 的默认值
    （本课程语料标定出来的阈值），因此"想看当前配置跑出什么"只需发一个
    空对象 ``{}``，而"想试另一套参数"也只覆盖要动的那几项。

    ``group_by`` / ``max_group_ratio`` / ``near_dup_threshold`` **刻意不加
    Pydantic 范围约束**：它们各自的合法区间由 ``domain_data`` 包在构造期
    判定，越界时的错误信息来自 ``DomainDataError`` / ``ValueError`` 的原文
    ——把校验复制到 Pydantic 里会得到两份会漂移的规则，还会把"参数不合法"
    从 400 变成 422，与 day050~day055 的错误码语言不一致。
    """

    augment: bool | None = Field(
        default=None, description="是否启用增强阶段；留空用 settings.domain_augment_enabled"
    )
    mixing: bool | None = Field(
        default=None, description="是否启用配比阶段（先削峰后填谷）；留空启用"
    )
    group_by: str | None = Field(
        default=None,
        description="配比分组口径：source/origin/safety；留空取 settings.domain_group_by",
    )
    max_group_ratio: float | None = Field(
        default=None,
        description="单组占比上限，必须落在 (0, 1]；留空取 settings.domain_max_group_ratio",
    )
    quality_threshold: float | None = Field(
        default=None, description="五维加权质量门槛；留空取 settings.domain_quality_threshold"
    )
    near_dup_threshold: float | None = Field(
        default=None,
        description=(
            "近重复签名 Jaccard 阈值，须落在 (0, 1]；"
            "留空取 settings.domain_near_dup_threshold"
        ),
    )
    version: int = Field(default=1, ge=1, description="本版数据集的版本号（进清单与版本链）")
    parent_fingerprint: str = Field(
        default="", description="上一版数据集指纹（版本链）；首版留空"
    )


class DomainRunResponse(BaseModel):
    """POST /data/domain/run 的响应体（M5-D8）：规模 + 清单 + 各阶段报告.

    **刻意不返回样本正文**：一次运行产出的样本有几十条、正文有几千字，
    把它们塞进响应体既撑大报文，也无助于回答"这次跑成什么样"——而
    ``manifest`` 已经含了阶段账（进/出/丢弃/新增）、来源分布、配比、
    质量分布与**全部参数快照**，指纹因此可在另一台机器上复现。

    五个阶段报告与 ``manifest`` 并列返回，而不是塞进清单里：清单描述
    **结论**，报告描述**过程**，"某条样本为什么没进最终数据集"需要的是后者。
    """

    size: int = Field(description="最终数据集条数（= freeze 阶段的 kept）")
    manifest: dict = Field(
        default_factory=dict,
        description="数据集清单：版本/指纹/阶段账/来源分布/配比/质量分布/参数快照",
    )
    clean: dict = Field(default_factory=dict, description="清洗报告：硬规则拒绝归因")
    quality: dict = Field(
        default_factory=dict, description="质量报告：五维总分分布与按最低维归因"
    )
    dedupe: dict = Field(default_factory=dict, description="去重报告：精确 + 近重复两级")
    mixing: dict = Field(default_factory=dict, description="配比报告：逐组进/出、迭代轮数与告警")
    augment: dict = Field(
        default_factory=dict, description="增强报告：算子分布、守门拒绝与撞车丢弃"
    )


# --------------------------------------------------------------------------- #
# day058 模型版本管理与持续微调（M5-D9）：/registry/* 五个端点
#
# 五个端点全部**只读或纯计算**：不写磁盘、不联网、不调模型。版本表的写入
# （register / set_stage）只发生在流水线与脚本里——HTTP 端点在多副本部署下
# 无法保证"谁先写"，而版本表的写入必须是单点的。
# --------------------------------------------------------------------------- #


class RegistryVersionInput(BaseModel):
    """一条待折叠的版本记录（M5-D9）.

    五个端点共用这一个模型：``/registry/versions`` 用它列出现状，
    ``/registry/rollback/plan`` 用它重建版本链。**请求里带全量版本记录**
    而不是让服务端读 ``outputs/registry/versions.jsonl``，是为了让端点
    保持无状态与可复现：同一个请求体在任何机器上得到同一个响应。

    ``adapter_sha256`` 与 ``dataset_fingerprint`` 的长度下限由
    ``registry.version`` 判定（分别 16 / 8 位十六进制），**刻意不在
    Pydantic 上复制**：两份会漂移的规则比一份好规则更糟。
    """

    version: str = Field(description="语义化版本号 X.Y.Z")
    base_model: str = Field(description="基座模型名（三元组第一项）")
    adapter_sha256: str = Field(description="适配器内容哈希（三元组第二项）")
    dataset_fingerprint: str = Field(description="数据集指纹（三元组第三项）")
    parent_version: str = Field(default="", description="父版本号；首版留空")
    stage: str = Field(
        default="candidate", description="阶段：candidate/stable/rolled_back/archived"
    )
    created_at: str = Field(default="", description="创建时间（UTC ISO）；留空取当前时间")
    artifacts: dict = Field(
        default_factory=dict,
        description="产物路径槽位：adapter / merged / dataset",
    )
    metrics: dict = Field(default_factory=dict, description="指标：至少含 eval_pass_rate")
    notes: str = Field(default="", description="备注（进审计）")
    tags: dict = Field(default_factory=dict, description="自由标签（如 bump_kind/triggered_by）")


class RegistryVersionsRequest(BaseModel):
    """POST /registry/versions 的请求体（M5-D9）：折叠一批版本记录并汇总.

    ``stage`` 用于只看某一阶段（例如"当前有哪些候选等我处理"）；
    留空返回全部。**空列表合法**（返回"尚无版本"），因为"仓库里还没有
    任何版本"正是首次上线前的真实状态。
    """

    versions: list[RegistryVersionInput] = Field(
        default_factory=list, description="版本记录（按版本号升序传入更易读，但非必需）"
    )
    stage: str | None = Field(
        default=None, description="只看该阶段的记录：candidate/stable/rolled_back/archived"
    )


class RegistryVersionsResponse(BaseModel):
    """POST /registry/versions 的响应体（M5-D9）.

    ``head`` 是当前生产版本的版本号，没有 stable 时**恒为 null**——
    "还没有生产版本"与"生产版本是某个候选"是两件事，接口不把它们混起来。
    """

    total: int = Field(description="版本总数")
    counts: dict = Field(default_factory=dict, description="按阶段计数（四个键恒存在）")
    head: str | None = Field(default=None, description="当前生产版本号；无 stable 为 null")
    head_key: str | None = Field(default=None, description="当前生产版本的版本键")
    versions: list[dict] = Field(default_factory=list, description="折叠后的版本记录")
    markdown: str = Field(default="", description="同一份数据的 markdown 渲染")


class RegistryCandidateRequest(BaseModel):
    """POST /registry/candidate/evaluate 的请求体（M5-D9）：采纳判定.

    ``current`` 留空表示**首次上线**：此时没有可比较的对象，
    只做绝对值门槛与可部署性判定。这个分支必须单独存在，否则第一版
    永远上不了线（增益无从计算）。

    ``policy`` 是**策略覆盖项**（可选键：``min_gain`` / ``regression_tolerance``
    / ``absolute_min_pass_rate`` / ``max_candidate_age_hours`` /
    ``require_comparable`` / ``require_deployable``）；留空用
    ``settings.retrain_*``。未知键会被忽略（``PromotionPolicy`` 只认自己的字段）。
    """

    candidate: RegistryVersionInput = Field(description="待判定的候选版本")
    current: RegistryVersionInput | None = Field(
        default=None, description="当前生产版本；留空表示首次上线"
    )
    policy: dict = Field(default_factory=dict, description="采纳策略覆盖项")


class RegistryCandidateResponse(BaseModel):
    """POST /registry/candidate/evaluate 的响应体（M5-D9）.

    ``decision.action`` 三选一：``promote``（采纳）/ ``hold``（保持现状）。
    回滚不在这里——**拒绝候选与回滚是两个动作**（触发者、对象、后果都不同），
    混在一起会让"候选没通过评估"变成"把线上版本也退掉"。
    """

    decision: dict = Field(default_factory=dict, description="采纳判定（含逐项检查）")
    markdown: str = Field(default="", description="同一份判定的 markdown 渲染")


class RegistryTriggersRequest(BaseModel):
    """POST /registry/triggers/evaluate 的请求体（M5-D9）：重训触发判定.

    六个字段与 ``registry.triggers.TriggerState`` 逐字对应。三个可空字段
    （``online_pass_rate`` / ``hours_since_last_train``）的语义是**缺失**，
    不是 0：缺失不触发、不否决，而 0 会（见 ``triggers`` 模块的说明）。
    """

    new_examples: int = Field(default=0, ge=0, description="自上次训练以来新增的可用样本数")
    dataset_fingerprint: str = Field(default="", description="当前数据集指纹")
    stable_dataset_fingerprint: str = Field(
        default="", description="生产版本所用的数据集指纹；无生产版本留空"
    )
    online_pass_rate: float | None = Field(
        default=None, description="线上观测到的合格率；未观测留空（缺失 ≠ 0）"
    )
    hours_since_last_train: float | None = Field(
        default=None, description="距上次训练的小时数；从未训练留空（不否决冷却期）"
    )
    active_runs: int = Field(default=0, ge=0, description="正在运行的训练任务数")
    policy: dict = Field(default_factory=dict, description="触发策略覆盖项")


class RegistryTriggersResponse(BaseModel):
    """POST /registry/triggers/evaluate 的响应体（M5-D9）.

    ``decision.fired`` 与 ``decision.vetoed_by`` **分开返回**：
    "没有理由训练"（fired 为空）与"有理由但时机不允许"（fired 非空且被否决）
    的运维动作完全不同——前者要去看数据与线上评估，后者只需要等。
    """

    decision: dict = Field(default_factory=dict, description="触发判定（含逐项评估）")
    markdown: str = Field(default="", description="同一份判定的 markdown 渲染")
    conditions: list[dict] = Field(
        default_factory=list, description="本次生效的条件表（触发器与否决项）"
    )


class RegistryRollbackRequest(BaseModel):
    """POST /registry/rollback/plan 的请求体（M5-D9）：生成回滚计划.

    ``versions`` 需要**全量版本记录**（而不是只传目标版本）：回滚要走版本链
    找"最近一个可部署的 stable 祖先"，链上一环缺失就可能选不出目标。
    """

    versions: list[RegistryVersionInput] = Field(
        min_length=1, description="全量版本记录（用于重建版本链）"
    )
    version: str = Field(description="出问题的版本号或版本键")
    reason: str = Field(default="", description="回滚原因（进审计事件）")
    observe_window_hours: float = Field(
        default=24.0, gt=0.0, description="回滚后的观察窗口（小时）"
    )


class RegistryRollbackResponse(BaseModel):
    """POST /registry/rollback/plan 的响应体（M5-D9）.

    ``plan.should_execute`` 为假时 ``plan.steps`` 为空——那表示**没有退路**
    （比如当前版本是首个 stable 版本，或祖先都不可部署）。
    "做不到"必须是一个被输出的结论，而不是一个被吞掉的异常。
    """

    plan: dict = Field(default_factory=dict, description="回滚计划（含有序动作序列）")
    markdown: str = Field(default="", description="同一份计划的 markdown 渲染")
    lineage: list[dict] = Field(default_factory=list, description="出问题版本的版本链")


# --------------------------------------------------------------------------- #
# day059 MLOps 微调流水线（M5-D10）：/mlops/* 四个端点
#
# 四个端点全部**只读或纯计算**：不写磁盘、不联网、不做真实训练。
# 真正的"训练 + 发布"由 scripts/mlops_demo.py 与 GitHub Actions 承担
# ——一个需要十分钟的流程不该挂在一个 HTTP 请求上（见 ci.render_github_actions）。
# --------------------------------------------------------------------------- #


class MLOpsStagesResponse(BaseModel):
    """GET /mlops/stages 的响应体（M5-D10）：这条流水线的自我描述.

    三个表全部从代码常量现场读出，因此"文档说 64 MiB、代码是 16 MiB"
    这类漂移会立刻可见：

    - ``stages`` ← ``stage_table()``（六个阶段的依赖与产出）；
    - ``critical_path`` ← ``critical_path()``（主链：任何一步失败都让发布停下来）；
    - ``gates`` ← ``gate_table()``（六项门禁的实际值与阈值）。

    ``limitations`` / ``intended_use`` / ``out_of_scope`` 是模型卡的
    "不能做什么"三段——**它们比"能做什么"更值得在开工前读一遍**。
    """

    stages: list[dict] = Field(default_factory=list, description="六阶段表（依赖与产出）")
    critical_path: list[str] = Field(default_factory=list, description="主链上的阶段名")
    gates: list[dict] = Field(default_factory=list, description="六项发布门禁与阈值")
    ci: dict = Field(default_factory=dict, description="CI 摘要（cron/超时/门禁命令）")
    intended_use: list[str] = Field(default_factory=list, description="适用场景")
    out_of_scope: list[str] = Field(default_factory=list, description="明确不适用")
    limitations: list[str] = Field(default_factory=list, description="已知限制（含数字依据）")


class MLOpsGateRequest(BaseModel):
    """POST /mlops/gates/evaluate 的请求体（M5-D10）：发布门禁判定.

    ``metrics`` 是**平铺的** ``{指标名: 值}``。门禁只认被点名的四个指标
    （``eval_pass_rate`` / ``eval_pass_rate_delta`` / ``adapter_mebibytes`` /
    ``cost_usd_per_1k_tokens``）加上两个溯源字段（``dataset_fingerprint`` /
    ``base_model``），其余指标原样忽略（它们进追踪器，不进判定）。

    ``policy`` 是**策略覆盖项**；``artifacts`` 用于产物完整性门禁。
    """

    metrics: dict = Field(default_factory=dict, description="平铺指标字典")
    artifacts: dict = Field(default_factory=dict, description="产物槽位：adapter / merged")
    policy: dict = Field(default_factory=dict, description="门禁策略覆盖项")
    commit: str = Field(default="", description="提交号（进报告）")


class MLOpsGateResponse(BaseModel):
    """POST /mlops/gates/evaluate 的响应体（M5-D10）.

    六项检查的结果分成**三类**返回，分开命名是刻意的：

    - ``report.blocking_failures``：阻塞且未通过 → 决定 ``report.passed``；
    - ``report.warnings``：非阻塞且未通过 → 有人**显式关掉了**这项检查；
    - ``report.skipped``：非阻塞且没有观测值 → 缺证据（例如首次训练没有基线）。

    混在一起的后果是运维看到"有 2 项没通过"时无法判断该去**补数据**
    还是该去**把开关打开**——而这两件事的动作完全不同。
    """

    report: dict = Field(default_factory=dict, description="门禁报告（逐项检查）")
    markdown: str = Field(default="", description="同一份报告的 markdown 渲染")


class MLOpsDryRunRequest(BaseModel):
    """POST /mlops/pipeline/dry-run 的请求体（M5-D10）：演练整条流水线.

    用**确定性的参考回调**驱动六阶段（不训练任何模型、不联网、不写盘），
    因此同一个请求体永远得到同一个响应。五个"结果输入"参数让调用方
    可以构造各种门禁情形（合格率不达标、适配器超体积、缺合并产物……），
    这正是 dry-run 的用途：**先看门禁会不会放行，再决定要不要真训**。

    ``dry_run=true``（缺省）时门禁通过也不写注册表；端点内部的注册表
    始终是内存的（HTTP 层不做版本表写入，理由见 day058 第五章）。
    """

    dataset_fingerprint: str = Field(
        default="3f1b0c9d7e5a2468", description="数据集指纹（三元组第三项）"
    )
    params: dict = Field(default_factory=dict, description="实验参数（进追踪器与 run_id）")
    pass_rate: float = Field(default=0.65, ge=0.0, le=1.0, description="模拟的评估合格率")
    baseline_pass_rate: float = Field(
        default=0.60, ge=0.0, le=1.0, description="模拟的基座（当前生产版本）合格率"
    )
    adapter_mebibytes: float = Field(default=0.05, gt=0.0, description="模拟的适配器体积")
    train_loss: float = Field(default=0.31, description="模拟的训练 loss")
    adapter_sha256: str = Field(
        default="", description="适配器内容哈希；留空则按数据集指纹与参数确定性派生"
    )
    with_merged_artifact: bool = Field(
        default=True, description="是否提供合并模型产物（false 用于触发产物完整性门禁）"
    )
    dry_run: bool = Field(default=True, description="true 时门禁通过也不写注册表")
    policy: dict = Field(default_factory=dict, description="门禁策略覆盖项")
    commit: str = Field(default="", description="提交号（进模型卡与报告）")


class MLOpsDryRunResponse(BaseModel):
    """POST /mlops/pipeline/dry-run 的响应体（M5-D10）.

    ``stages`` 里 ``publish`` 的状态在两种情况下是 ``blocked``（而不是
    ``failed``）：门禁不通过、或 ``dry_run=true``。**这个区分是本课的核心**
    ——前者意味着"这一版不该上线"，后者意味着"只演练不发"，
    而两者都不是"有东西坏了"。
    """

    summary: str = Field(default="", description="一行结论")
    run: dict = Field(default_factory=dict, description="实验追踪记录（含指标与产物）")
    stages: list[dict] = Field(default_factory=list, description="六个阶段的逐条结果")
    gate: dict | None = Field(default=None, description="发布门禁报告")
    card: dict | None = Field(default=None, description="模型卡")
    manifest: dict = Field(default_factory=dict, description="发布清单")
    version: dict | None = Field(default=None, description="登记后的版本记录")
    published: bool = Field(default=False, description="是否真的发布了")
    failed: bool = Field(default=False, description="是否有阶段失败（blocked 不算）")
    markdown: str = Field(default="", description="整次运行的 markdown 渲染")


class MLOpsWorkflowResponse(BaseModel):
    """GET /mlops/ci/workflow 的响应体（M5-D10）：生成的 workflow 与摘要.

    ``yaml`` 与仓库里那份 ``.github/workflows/finetune-nightly.yml``
    **逐字相同**（有一条测试守着），因此改门禁阈值 = 一次可追溯的 diff。
    """

    summary: dict = Field(default_factory=dict, description="CI 摘要（cron/命令/阈值）")
    path: str = Field(default="", description="workflow 在仓库中的相对路径")
    yaml: str = Field(default="", description="渲染出的 workflow YAML 文本")
    commands: list[str] = Field(default_factory=list, description="门禁脚本命令行")


# --------------------------------------------------------------------------- #
# day060 专属模型部署与切换（M5-D11）：/serving/* 四个端点
#
# 四个端点全部**只读或纯计算**：不写磁盘、不联网、不加载模型、不用随机数。
# 真正加载模型、压测延迟、切流量由 scripts/serving_demo.py 与部署脚本承担
# ——**一个需要 GPU 的流程不该挂在一个 HTTP 请求上**（与 day059 的取舍同源）。
# --------------------------------------------------------------------------- #


class ServingTargetsResponse(BaseModel):
    """GET /serving/targets 的响应体（M5-D11）：部署这件事的自我描述.

    五张表全部从代码常量现场读出，因此"文档说 64 MiB、代码是 16 MiB"
    这类漂移在这里同样会立刻可见：

    - ``kinds`` ← ``spec_table()``（三种形态的必填字段与切换代价）；
    - ``memory`` ← ``memory_breakdown()``（权重 / KV 缓存 / 适配器三块分开列）；
    - ``traffic_table`` ← ``route_table()``（分桶、影子、降级**各自的代价**）；
    - ``verify_table`` ← ``verify_table()``（五条检查，含"没测时算什么"）；
    - ``binding_table`` ← ``binding_table()``（五条一致性检查挡住的故障）。
    """

    spec: dict = Field(default_factory=dict, description="当前配置解析出的部署档案")
    kinds: list[dict] = Field(default_factory=list, description="三种部署形态对照表")
    memory: dict = Field(default_factory=dict, description="显存分解（权重/KV/适配器）")
    recommendation: dict = Field(default_factory=dict, description="形态建议与理由")
    traffic: dict = Field(default_factory=dict, description="当前流量策略")
    traffic_table: list[dict] = Field(default_factory=list, description="流量策略逐项含义与代价")
    verify_table: list[dict] = Field(default_factory=list, description="五条上线检查")
    binding_table: list[dict] = Field(default_factory=list, description="五条部署一致性检查")
    price_book: list[dict] = Field(default_factory=list, description="价格表（每一条都标注出处口径）")
    out_of_scope: list[str] = Field(default_factory=list, description="明确不适用")
    limitations: list[str] = Field(default_factory=list, description="已知限制（含数字依据）")


class ServingBindingRequest(BaseModel):
    """POST /serving/binding/verify 的请求体（M5-D11）：三处版本比对.

    请求里带**全量版本记录**而不是让服务端读 ``outputs/registry/versions.jsonl``，
    与 day058/059 的端点同一理由：让端点保持无状态与可复现。

    ``bound_version`` 是**本次部署打算上的那一版**（由部署脚本给出），
    ``served`` 是**端点自述的现状**（``/v1/models`` 的 id、``/api/show`` 的
    基座与适配器）。两组事实都由调用方提供——**端点不猜，它只比对**。
    """

    versions: list[RegistryVersionInput] = Field(
        default_factory=list, description="注册表里的版本记录（用于折叠出 head）"
    )
    bound_version: str = Field(description="本次部署绑定的版本号 X.Y.Z")
    serving_name: str = Field(default="", description="实际部署用的服务名；留空取 spec.name")
    spec: dict = Field(default_factory=dict, description="部署档案覆盖项（kind/name/backend/...）")
    served: dict = Field(default_factory=dict, description="端点自述：name/base_model/adapter_short_hash")
    policy: dict = Field(default_factory=dict, description="绑定校验策略覆盖项（两个开关）")


class ServingBindingResponse(BaseModel):
    """POST /serving/binding/verify 的响应体（M5-D11）.

    ``passed`` 只看**阻塞项**；``consistent`` 表示**全部**一致（含非阻塞项）。

    两个布尔量分开是刻意的：一个 ``passed=true`` 但 ``consistent=false``
    的部署是**灰度期间最正常的状态**（部署的不是 head），
    把两者合成一个就再也表达不出"我放行了，但我知道它不是 head"。
    """

    summary: str = Field(default="", description="一行结论")
    passed: bool = Field(default=False, description="阻塞项是否全部通过")
    consistent: bool = Field(default=False, description="是否全部一致（含非阻塞告警）")
    binding: dict = Field(default_factory=dict, description="部署记录（三处事实的绑定结果）")
    report: dict = Field(default_factory=dict, description="逐项检查报告")
    markdown: str = Field(default="", description="同一份报告的 markdown 渲染")


class ServingProbeInput(BaseModel):
    """一条端到端验证用例（M5-D11）.

    ``must_contain`` 为空的用例**永远通过**：它退化成一条"只测连通性"的探针
    （回答内容不论，只要没抛异常）。这是有意的用法，不是漏洞。
    """

    case_id: str = Field(description="用例 id（报告里按它定位）")
    prompt: str = Field(description="发给两个推理臂的同一个问题")
    must_contain: list[str] = Field(
        default_factory=list, description="回复里必须出现的片段（大小写不敏感，顺序不限）"
    )


class ServingArmInput(BaseModel):
    """一个推理臂的**观测结果**（M5-D11）：由调用方实测后填进来.

    端点不调用模型，因此它接收的是"你已经观测到的事实"：
    ``replies`` 是按 ``case_id`` 索引的回复，``latency_ms`` 是实测延迟，
    ``error_cases`` 是抛了异常的用例 id。**三者分开**是刻意的——
    "调用失败"与"答得不对"是两类问题，混成一个"不通过"就没法定位。
    """

    replies: dict = Field(default_factory=dict, description="case_id → 回复文本")
    latency_ms: dict = Field(default_factory=dict, description="case_id → 实测延迟（毫秒）")
    error_cases: list[str] = Field(default_factory=list, description="调用失败的用例 id")


class ServingVerifyRequest(BaseModel):
    """POST /serving/verify 的请求体（M5-D11）：上线前的端到端验证.

    ``policy.max_latency_ratio`` 缺省**不检查**——"多慢算慢"取决于部署形态，
    给一个全局缺省只会让它被无脑放宽；要查就显式填一个倍数。
    """

    probes: list[ServingProbeInput] = Field(default_factory=list, description="验证用例")
    cloud: ServingArmInput = Field(default_factory=ServingArmInput, description="云端臂的观测结果")
    dedicated: ServingArmInput = Field(
        default_factory=ServingArmInput, description="专属模型臂的观测结果"
    )
    policy: dict = Field(default_factory=dict, description="验证策略覆盖项")
    notes: str = Field(default="", description="备注（进报告）")


class ServingVerifyResponse(BaseModel):
    """POST /serving/verify 的响应体（M5-D11）.

    ``report.blocking_failures`` 决定 ``report.passed``；
    ``report.skipped`` 是**没有观测值**、按策略降级为非阻塞的检查（当前只有
    "策略未给出倍数"的延迟项）。**"没测"与"没达标"必须分开返回**——
    前者要补数据，后者要改模型。
    """

    summary: str = Field(default="", description="一行结论")
    report: dict = Field(default_factory=dict, description="五条检查的完整报告")
    markdown: str = Field(default="", description="同一份报告的 markdown 渲染")


class ServingCostRequest(BaseModel):
    """POST /serving/cost/compare 的请求体（M5-D11）：专属模型值不值.

    ``gpu_tokens_per_second`` **没有权威缺省值**：吞吐取决于模型、量化、
    批大小与序列长度。请求里的缺省值来自 ``settings.serving_gpu_tokens_per_second``，
    而它是**待替换的占位值**——请填入本机实测的吞吐，否则盈亏平衡点不可信。
    """

    requests: int = Field(default=100000, ge=0, description="月度请求量")
    input_tokens_per_request: int = Field(default=1500, ge=0, description="每次请求的平均输入 token")
    output_tokens_per_request: int = Field(default=500, ge=0, description="每次请求的平均输出 token")
    gpu_key: str = Field(default="", description="GPU 机型键；留空取 settings")
    gpu_tokens_per_second: float = Field(default=0.0, ge=0.0, description="实测吞吐；0 表示取 settings")
    gpu_utilization: float = Field(default=0.0, ge=0.0, le=1.0, description="GPU 利用率；0 表示取 settings")
    cloud_key: str = Field(default="", description="云端计价方案键；留空取 settings")
    monthly_hours: float = Field(default=0.0, ge=0.0, description="月度开机小时数；0 表示取 730")


class ServingCostResponse(BaseModel):
    """POST /serving/cost/compare 的响应体（M5-D11）.

    ``comparison.cheaper_side`` 是结论，但**真正该看的是另外三个数**：
    ``required_gpu_hours`` / ``deployed_gpu_hours`` / ``utilization_actual``
    ——它们解释"为什么自建在低请求量下一定更贵"，而不是只给一个谁更便宜。
    """

    summary: str = Field(default="", description="一行结论")
    comparison: dict = Field(default_factory=dict, description="两侧成本与四个派生数字")
    markdown: str = Field(default="", description="同一份对比的 markdown 渲染")


# --------------------------------------------------------------------------- #
# day061 文档解析与加载（M6-D1）：/documents/* 四个端点
#
# 四个端点全部**只读或纯计算**：不写磁盘、不联网、不遍历目录。
# 目录遍历与批量入库由 scripts/documents_demo.py 承担——
# **一个要读整个目录的流程不该挂在一个 HTTP 请求上**（与 day059/060 同源）。
# --------------------------------------------------------------------------- #


class DocumentsLoadersResponse(BaseModel):
    """GET /documents/loaders 的响应体（M6-D1）：解析能力的自我描述.

    六张表全部从代码常量现场读出，因此"文档说支持 PDF、代码里没有加载器"
    这类漂移会立刻可见：

    - ``loaders`` ← ``LoaderRegistry.table()``（哪个媒体类型交给哪个加载器）；
    - ``encoding_chain`` ← ``encoding_chain()``（编码探测的固定顺序与理由）；
    - ``markdown`` ← ``markdown_syntax_table()``（结构 → 规范条款）；
    - ``html`` ← ``html_rules()``（跳过哪些标签、为什么）；
    - ``pdf`` ← ``pdf_boundaries()``（认识哪些操作符、**做不到什么**）；
    - ``docx`` ← ``docx_scope()``（读哪些部件、**不读哪些部件**）。
    """

    media_types: list[str] = Field(default_factory=list, description="已注册的媒体类型")
    loaders: list[dict] = Field(default_factory=list, description="媒体类型 → 加载器")
    encoding_chain: list[dict] = Field(default_factory=list, description="编码探测的固定顺序")
    markdown: list[dict] = Field(default_factory=list, description="Markdown 结构与规范条款")
    html: list[dict] = Field(default_factory=list, description="HTML 解析规则")
    pdf: dict = Field(default_factory=dict, description="PDF 支持的操作符与做不到的事")
    docx: dict = Field(default_factory=dict, description="DOCX 读与不读的部件")
    block_kinds: list[str] = Field(default_factory=list, description="六种块类型")
    knowledge_record: dict = Field(default_factory=dict, description="交给知识库的记录形状")
    max_file_mib: float = Field(default=0.0, description="单份文件的解析上限（MiB）")
    out_of_scope: list[str] = Field(default_factory=list, description="明确不适用")
    limitations: list[str] = Field(default_factory=list, description="已知限制（含依据）")


class DocumentDetectRequest(BaseModel):
    """POST /documents/detect 的请求体（M6-D1）：只做类型与编码识别.

    ``content_base64`` 承载二进制内容（PDF / docx / 图片）；
    纯文本也可以走 ``text`` 字段，此时按 UTF-8 编码后再识别——
    两条路都会做**同一套**类型识别（内容优先、冲突必报）。
    """

    filename: str = Field(default="", description="文件名（用于看后缀；留空则只看内容）")
    content_base64: str = Field(default="", description="base64 编码的文件内容")
    text: str = Field(default="", description="纯文本内容（与 content_base64 二选一）")


class DocumentDetectResponse(BaseModel):
    """POST /documents/detect 的响应体（M6-D1）.

    ``conflict`` 为真说明**文件名与内容说了两件不同的事**（例如
    ``.txt`` 里其实是 PDF）。这类记录往往是"磁盘上的文件被人改过"
    的第一手证据，因此它单独返回而不是并进某一条错误里。
    """

    summary: str = Field(default="", description="一行结论")
    media_type: str = Field(default="", description="最终采用的媒体类型（内容优先）")
    decided_by: str = Field(default="", description="判定依据：magic / suffix / given / fallback")
    suffix_type: str = Field(default="", description="按后缀得到的类型")
    magic_type: str = Field(default="", description="按内容得到的类型")
    conflict: bool = Field(default=False, description="后缀与内容是否冲突")
    supported: bool = Field(default=False, description="注册表里是否有加载器能处理它")
    encoding: str = Field(default="", description="文本解码采用的编码")
    encoding_decided_by: str = Field(default="", description="编码判定依据")
    encoding_confident: bool = Field(default=True, description="编码是否可信（false 表示已回退 latin-1）")
    size_bytes: int = Field(default=0, description="内容字节数")


class DocumentParseRequest(BaseModel):
    """POST /documents/parse 的请求体（M6-D1）：解析一份文档.

    ``media_type`` 留空时按"内容优先"的规则自动识别（见 ``/documents/detect``）；
    显式给出时**直接采用**（``decided_by=given``）——这是给"我已经知道
    这是什么格式，但文件名不可信"的场景留的入口。
    """

    filename: str = Field(default="", description="文件名（用于类型识别与 source 标签）")
    content_base64: str = Field(default="", description="base64 编码的文件内容")
    text: str = Field(default="", description="纯文本内容（与 content_base64 二选一）")
    media_type: str = Field(default="", description="显式指定媒体类型；留空则自动识别")
    include_blocks: bool = Field(default=True, description="是否返回逐块明细")


class DocumentParseResponse(BaseModel):
    """POST /documents/parse 的响应体（M6-D1）.

    ``document.doc_id`` 是**内容指纹**（规范化全文的 sha256 前 16 位）：
    同一份内容从两个文件名传进来会得到同一个 id，因此它可以直接用于去重。
    """

    summary: str = Field(default="", description="一行结论")
    document: dict = Field(default_factory=dict, description="归一化后的文档")
    detection: dict = Field(default_factory=dict, description="类型识别结果")


class DocumentIngestFile(BaseModel):
    """入库清单里的一个文件（M6-D1）.

    **清单的顺序就是去重时的优先级**：先出现的那一份被留下，
    后来者被标记 ``duplicate_of``。因此同一个请求体永远得到同一份结果
    （不需要服务端自己排序）。
    """

    filename: str = Field(description="来源标签（同时用于类型识别）")
    content_base64: str = Field(default="", description="base64 编码的文件内容")
    text: str = Field(default="", description="纯文本内容（与 content_base64 二选一）")


class DocumentIngestRequest(BaseModel):
    """POST /documents/ingest 的请求体（M6-D1）：一批文件入库.

    这一批是**内存里的文件清单**而不是一个目录：端点不遍历文件系统，
    因此同一个请求体在任何机器上得到同一个响应（与 day058/059/060 的
    "不带状态"是同一条纪律）。
    """

    files: list[DocumentIngestFile] = Field(default_factory=list, description="文件清单")


class DocumentIngestResponse(BaseModel):
    """POST /documents/ingest 的响应体（M6-D1）.

    ``status_counts`` 有三个恒存在的键：``ok`` / ``duplicate`` / ``error``。
    **重复单独一态**是刻意的——"这一批里有几份是重复内容"是一个数据治理
    问题，不是一个失败问题。
    """

    summary: str = Field(default="", description="一行结论")
    report: dict = Field(default_factory=dict, description="逐条结果与汇总")
    markdown: str = Field(default="", description="同一份报告的 markdown 渲染")


# --------------------------------------------------------------------------- #
# day062 分块策略（M6-D2）：/chunking/* 四个端点
#
# 与 /documents/* 同一纪律：**只读或纯计算**——不写磁盘、不联网、
# 不遍历目录、不留状态。分块需要 embedding 时用的是 `app.state.embedding`
# （与 /embeddings 同一个注入点），因此测试注入 MockEmbedding 即可离线跑通。
# --------------------------------------------------------------------------- #


class ChunkingStrategiesResponse(BaseModel):
    """GET /chunking/strategies 的响应体（M6-D2）：分块能力的自我描述.

    四张表全部从代码常量现场读出，因此"文档说支持四种策略、代码里只有三种"
    这类漂移会立刻可见：

    - ``strategies`` ← 四个 ``Chunker.describe()``（在哪里切、参数、强弱项）；
    - ``measurers`` ← ``describe_measurers()``（预算单位是谁，**确定性一列**）；
    - ``default_policies`` ← ``DEFAULT_POLICIES``（**默认值是策略的一部分**）；
    - ``coverage_invariant`` ← 那条"块文本必须逐字可在原文里找到"的不变量。
    """

    strategies: list[dict] = Field(default_factory=list, description="四个策略的自述表")
    measurers: list[dict] = Field(default_factory=list, description="预算度量器与确定性")
    default_policies: dict = Field(default_factory=dict, description="按策略给出的默认参数")
    separators: list[str] = Field(default_factory=list, description="递归策略的分隔符优先级表")
    atomic_kinds: list[str] = Field(default_factory=list, description="结构策略保护的块类型")
    knowledge_record: dict = Field(default_factory=dict, description="交给知识库的记录形状")
    coverage_invariant: str = Field(default="", description="块与原文之间的不变量")
    defaults: dict = Field(default_factory=dict, description="settings 里的 chunking_* 默认值")
    out_of_scope: list[str] = Field(default_factory=list, description="明确不适用")
    limitations: list[str] = Field(default_factory=list, description="已知限制（含依据）")


class ChunkingSplitRequest(BaseModel):
    """POST /chunking/split 的请求体（M6-D2）：切一份文档.

    文档输入沿用 ``/documents/parse`` 的形状（文件名 / base64 / 文本三选一），
    因为"分块的上游就是解析"。参数留空时按 ``strategy`` 的**默认策略参数**
    （``DEFAULT_POLICIES``）填，而不是全局常量——**默认值属于策略**：
    ``overlap`` 对结构策略是错的（它会跨标题回带），因此它的默认值是 0。
    """

    filename: str = Field(default="", description="文件名（用于类型识别与 source 标签）")
    content_base64: str = Field(default="", description="base64 编码的文件内容")
    text: str = Field(default="", description="纯文本内容（与 content_base64 二选一）")
    media_type: str = Field(default="", description="显式指定媒体类型；留空则自动识别")
    strategy: str = Field(default="", description="分块策略；留空用 settings.chunking_strategy")
    max_tokens: int = Field(default=0, description="单块预算；0 表示用策略默认值")
    overlap_tokens: int = Field(default=-1, description="块间重叠；负数表示用策略默认值")
    min_tokens: int = Field(default=-1, description="过短块的合并下限；负数表示用策略默认值")
    measurer: str = Field(default="", description="预算单位 chars/tiktoken；留空用 settings")
    similarity_percentile: float = Field(
        default=-1.0, description="语义策略的相似度分位数；负数表示用策略默认值"
    )
    top_n: int = Field(default=20, description="返回前多少块的明细（0 表示全部）")
    include_text: bool = Field(default=False, description="明细里是否带块文本")


class ChunkingSplitResponse(BaseModel):
    """POST /chunking/split 的响应体（M6-D2）.

    三块内容各自回答一个问题，**不混在一起**：

    - ``plan``：参数一确定就能算出来的代价（块数、重叠放大量）——
      **先算再切**，调参时不必等 embedding 跑完；
    - ``stats``：切完之后的实测（覆盖率、重复率、长度分布、超预算数）；
    - ``chunks``：前 ``top_n`` 块的明细（含 ``start_char:end_char``，
      可以直接回到原文核对）。
    """

    summary: str = Field(default="", description="一行结论")
    plan: dict = Field(default_factory=dict, description="窗口/预算的理论代价")
    stats: dict = Field(default_factory=dict, description="实测统计（不含全文）")
    chunks: list[dict] = Field(default_factory=list, description="前若干块的明细")
    document: dict = Field(default_factory=dict, description="归一化后的文档摘要")


class ChunkingProbe(BaseModel):
    """评估探针：一个问题 + 期望命中的原文片段（M6-D2）."""

    question: str = Field(description="检索时输入的问题")
    expect: str = Field(description="期望命中的原文片段（必须能在原文里逐字找到）")
    note: str = Field(default="", description="备注（例如这条探针想验证什么）")


class ChunkingEvaluateRequest(BaseModel):
    """POST /chunking/evaluate 的请求体（M6-D2）：多策略检索对照.

    ``probes`` **不能为空**：没有期望片段就无法判定命中，
    而"人工看一眼觉得还行"不是一条可复现的判据。
    """

    filename: str = Field(default="", description="文件名（用于类型识别与 source 标签）")
    content_base64: str = Field(default="", description="base64 编码的文件内容")
    text: str = Field(default="", description="纯文本内容（与 content_base64 二选一）")
    media_type: str = Field(default="", description="显式指定媒体类型；留空则自动识别")
    probes: list[ChunkingProbe] = Field(default_factory=list, description="探针集")
    strategies: list[str] = Field(default_factory=list, description="要比较的策略；留空表示全部")
    top_k: int = Field(default=0, description="检索深度；0 表示用 settings.chunking_eval_top_k")
    max_tokens: int = Field(default=0, description="统一覆盖各策略的单块预算；0 表示用各自默认值")


class ChunkingEvaluateResponse(BaseModel):
    """POST /chunking/evaluate 的响应体（M6-D2）.

    ``ambiguous_probes`` 单独返回是刻意的：一条"期望片段出现多次"的探针
    会让评分**虚高且毫不显眼**，因此它必须能与分数分开看。
    ``markdown`` 是同一份结果的渲染版，可直接贴进选型记录。
    """

    summary: str = Field(default="", description="一行结论")
    evaluation: dict = Field(default_factory=dict, description="逐策略得分与明细")
    markdown: str = Field(default="", description="同一份结果的 markdown 渲染")


class ChunkingRecordsRequest(BaseModel):
    """POST /chunking/records 的请求体（M6-D2）：切成可直接入库的记录.

    它是 day061 ``/documents/ingest`` 的下一环：那里交出的是**整份文档**，
    这里交出的是**文档切出来的片段**，而 ``doc_id/source/text/metadata``
    四个键的形状不变。
    """

    filename: str = Field(default="", description="文件名（用于类型识别与 source 标签）")
    content_base64: str = Field(default="", description="base64 编码的文件内容")
    text: str = Field(default="", description="纯文本内容（与 content_base64 二选一）")
    media_type: str = Field(default="", description="显式指定媒体类型；留空则自动识别")
    strategy: str = Field(default="", description="分块策略；留空用 settings.chunking_strategy")
    max_tokens: int = Field(default=0, description="单块预算；0 表示用策略默认值")
    overlap_tokens: int = Field(default=-1, description="块间重叠；负数表示用策略默认值")
    limit: int = Field(default=10, description="返回前多少条记录（0 表示全部）")


class ChunkingRecordsResponse(BaseModel):
    """POST /chunking/records 的响应体（M6-D2）.

    ``embedding_input`` 单独列出来，是为了把"存的"和"embed 的"分开：
    ``text`` 是原文片段（可直接引用给用户），
    向量化用的是 ``metadata.retrieval_text``（带标题面包屑）。
    **这一条必须被写出来**，否则下一个人会以为向量是用 ``text`` 算的。
    """

    summary: str = Field(default="", description="一行结论")
    report: dict = Field(default_factory=dict, description="分块报告（逐策略汇总）")
    records: list[dict] = Field(default_factory=list, description="知识库记录（前 limit 条）")
    embedding_input: str = Field(default="", description="向量化时应当使用哪个字段")
    markdown: str = Field(default="", description="报告的 markdown 渲染")


# --------------------------------------------------------------------------- #
# day064 向量库（M6-D3）：/vectorstore/* 六个端点
#
# 与 /chunking/* 同一纪律（依赖注入、响应体全部可直接 json.dumps），但有一处
# **关键差别**：本组的 upsert 与 records 删除会改内存状态。因此状态一律挂在
# 应用实例上——后端在 ``app.state.vector_store``，最近一次写入报告在
# ``app.state.vectorstore_last_ingest``——**不挂 settings、不挂模块级全局**：
# 挂上模块全局，"两个 app 实例互不影响"这条性质就成了一句空话，
# 而它的失效方式是"另一个测试偶发失败"，最难查。
#
# 响应体的每一块都直接用向量库的公开投影（``to_dict`` / ``summary_line``），
# 端点层不重新拼装字段：**两处各拼一份「有哪些字段」迟早会分叉**，
# 而分叉出来的那份会被当成契约抄进客户端。
# --------------------------------------------------------------------------- #


class VectorStoreBackendsResponse(BaseModel):
    """GET /vectorstore/backends 的响应体（M6-D3）：后端能力表 + 度量表 + 当前选用.

    两张表都从代码常量现场读出（``registry.describe_backends()`` /
    ``metrics.describe_metrics()``），因此"文档写着三个后端、代码里只有一个"
    这类漂移会立刻可见——而不是等到某次选型会议上被当成事实引用。

    ``current_*`` 三个字段说的是**这个应用实例**：能力表回答"理论上能用什么"，
    ``current_backend`` 回答"现在实际用的是哪一个"。两者必须分开，
    否则"缺 faiss 时 flat 顶上了"会被读成"我配的就是 flat"。
    """

    backends: list[dict] = Field(default_factory=list, description="三个后端的声明表")
    metrics: list[dict] = Field(default_factory=list, description="三种度量的口径说明")
    current_backend: str = Field(default="", description="本实例实际使用的后端名")
    current_metric: str = Field(default="", description="本实例实际使用的度量")
    current_count: int = Field(default=0, description="本实例当前库里的记录条数")
    default_backend: str = Field(default="", description="settings.vector_backend")
    default_metric: str = Field(default="", description="settings.vector_metric")
    default_top_k: int = Field(default=0, description="settings.vector_default_top_k")
    min_score: float | None = Field(
        default=None, description="settings.vector_min_score；None 表示不设阈值"
    )
    persistence_path: str = Field(
        default="", description="settings.vector_persist_path；空串表示不落盘"
    )
    limitations: list[str] = Field(default_factory=list, description="本层的已知限制（含依据）")
    out_of_scope: list[str] = Field(default_factory=list, description="明确排除在范围外的能力")


class VectorUpsertRequest(BaseModel):
    """POST /vectorstore/upsert 的请求体（M6-D3）：把一批知识库记录写进库.

    ``records`` 就是 day062 ``ChunkSet.knowledge_records()`` 交出来的形状
    （``doc_id`` / ``source`` / ``text`` / ``metadata`` 四个键），**字段名不变**：
    上游产出什么形状，这里就收什么形状，中间不再发明一层"入库请求格式"。

    请求体里**没有"选后端"这个字段**：后端由 ``app.state.vector_store`` 决定。
    允许一个请求换后端，会让"这份数据到底在哪个库里"变成一件要靠翻日志
    才能回答的事——而它恰恰是排查"检索结果不对"时第一个要问的问题。
    """

    records: list[dict] = Field(
        default_factory=list, description="day062 形状的知识库记录列表（四个键保持不变）"
    )


class VectorUpsertResponse(BaseModel):
    """POST /vectorstore/upsert 的响应体（M6-D3）.

    ``report`` 是 ``pipeline.IngestReport.to_dict()`` 的原样投影，五个去向
    （written / unchanged / skipped / failed）各自一个数字：

    - ``written`` 与 ``unchanged`` 分开，是为了让"这次其实什么都没写"可见
      ——day065 的增量索引就靠这个答案省钱；
    - ``skipped``（数据问题）与 ``failed``（环境问题）分开，是为了让报告
      直接指向该找谁修；
    - ``embedding_calls`` 是**本课逐条编码**留下的基准列：day065 的批量 + 缓存
      省下了多少次编码，要跟它比才说得清。
    """

    summary: str = Field(default="", description="一行结论")
    report: dict = Field(default_factory=dict, description="IngestReport 的完整投影")
    count: int = Field(default=0, description="写入之后库里的记录条数")
    dimension: int = Field(default=0, description="写入之后库的维度（空库为 0）")
    backend: str = Field(default="", description="本次写入落在哪个后端上")
    metric: str = Field(default="", description="该后端的度量")


class VectorSearchRequest(BaseModel):
    """POST /vectorstore/search 的请求体（M6-D3）：查询文本 → 编码 → 检索.

    ``top_k=0`` 表示"用项目级基线"（``settings.vector_default_top_k``），
    与 day062 的 ``max_tokens=0`` 是同一条约定：**0 是"没给"，
    而不是"要 0 条"**——负数则照实送进校验并返回 400。

    ``where`` 用的是本包的规范语法（与 Chroma 一致，12 个运算符）；
    ``min_score`` 留空表示不设阈值（**不是阈值 0**：在 ``l2`` 度量下
    分数是负数，阈值 0 会把全部命中都切掉）。
    """

    query: str = Field(default="", description="查询文本（走与入库完全相同的编码路径）")
    top_k: int = Field(default=0, description="检索深度；0 表示用 settings.vector_default_top_k")
    where: dict | None = Field(default=None, description="元数据过滤子句；None 表示不过滤")
    min_score: float | None = Field(
        default=None, description="相似度下限（越大越近）；None 表示不设阈值"
    )


class VectorSearchResponse(BaseModel):
    """POST /vectorstore/search 的响应体（M6-D3）.

    ``result`` 是 ``SearchResult.to_dict()`` 的原样投影。其中三个字段
    一起才说得出"为什么只返回了两条"：

    ```text
    candidates=0 且 filter_applied=True   → 过滤器把库里所有记录都排除了
    candidates>0 而 count=0               → min_score 把命中切掉了
    candidates>0 且 count<candidates      → top_k 起了作用
    ```

    ``filter`` 是子句的人类可读渲染（``describe_filter``），``filter_fields``
    列出子句里出现过的字段名——空结果的第一个排查动作就是核对这两项。
    """

    summary: str = Field(default="", description="一行结论")
    result: dict = Field(default_factory=dict, description="SearchResult 的完整投影")
    filter: str = Field(default="", description="过滤子句的可读渲染")
    filter_fields: list[str] = Field(default_factory=list, description="子句里出现过的元数据字段名")
    query_chars: int = Field(default=0, description="查询文本的字符数（编码的是这段文本）")


class VectorStatsResponse(BaseModel):
    """GET /vectorstore/stats 的响应体（M6-D3）：库状态 + 最近一次写入 + 编码器信息.

    ``last_ingest`` 是**本应用实例**最近一次写入的 IngestReport；还没写过就是
    ``None``。它是"内存后端的写入只活在这个实例里"这句话的可核对形式——
    换一个 app 实例，它必须是 ``None``，而那个实例的 ``info.count`` 必须是 0。
    """

    summary: str = Field(default="", description="库状态的一行结论")
    info: dict = Field(default_factory=dict, description="StoreInfo 的投影（后端自述）")
    pipeline: dict = Field(default_factory=dict, description="编码器与检索默认值")
    last_ingest: dict | None = Field(
        default=None, description="本实例最近一次写入报告；None 表示还没写过"
    )


class VectorDeleteRequest(BaseModel):
    """DELETE /vectorstore/records 的请求体（M6-D3）：按 ids 或 where 删除.

    两种方式**必须二选一**（``VectorBackend.delete`` 的契约）：同时给出会让
    "到底删了什么"变得不确定——一次条件写错的 ``delete(ids=..., where=...)``
    会变成"什么都没删"，而调用方以为删了。
    """

    ids: list[str] = Field(default_factory=list, description="要删除的记录 id 列表")
    where: dict | None = Field(default=None, description="按元数据条件删除；与 ids 二选一")


class VectorDeleteResponse(BaseModel):
    """DELETE /vectorstore/records 的响应体（M6-D3）.

    ``removed`` 是**真正删掉的**条数，而不是请求删的条数：删一个不存在的 id
    是幂等操作（不报错），但报告里必须能看出"这次其实什么都没删"。
    """

    summary: str = Field(default="", description="一行结论")
    removed: int = Field(default=0, description="真正删掉的条数")
    count: int = Field(default=0, description="删除之后库里的记录条数")


class VectorCompareRequest(BaseModel):
    """POST /vectorstore/compare 的请求体（M6-D3）：一批记录 + 一个查询做跨后端对账.

    ``metrics`` 留空表示用 ``settings.vector_metric`` 那一个度量；
    给多个度量时，报告的行数按"度量 × 后端"增长，行序是**度量优先**
    （同一个度量下各后端的行连在一起才好读）。
    """

    records: list[dict] = Field(default_factory=list, description="day062 形状的记录列表")
    query: str = Field(default="", description="查询文本（与记录走同一条编码路径）")
    top_k: int = Field(default=0, description="对账深度；0 表示用 settings.vector_default_top_k")
    metrics: list[str] = Field(
        default_factory=list, description="参与对账的度量；留空用 settings.vector_metric"
    )


class VectorCompareResponse(BaseModel):
    """POST /vectorstore/compare 的响应体（M6-D3）.

    ``rows`` 是 ``ParityRow.to_dict()`` 的列表，**每一行都有一席之地**：
    缺 faiss/chromadb 的机器上，那两个后端是 ``available=False`` 的行
    （带三段式安装指引），HTTP 状态仍是 200——同一份请求在任何机器上返回
    **行数相同**的报告，只差一个布尔。这是本端点最重要的行为。

    ``check`` 是 ``evaluate.verify_parity`` 的结论（``ok`` / ``failures`` /
    ``checked`` / ``available``）。注意本端点的参照是**降级参照**
    （生产代码不能 import tests），因此 ``ok=True`` 只说明"这台机器上几个
    后端彼此一致"，**不能当成 CI 的判据**（真正的对账在
    ``tests/test_vectorstore_parity.py`` 里，那里每条都注入独立算出的期望排名）。
    """

    summary: str = Field(default="", description="一行结论")
    rows: list[dict] = Field(default_factory=list, description="逐（后端 × 度量）的对账结论")
    check: dict = Field(default_factory=dict, description="verify_parity 的结论（降级参照）")
    backends: list[str] = Field(default_factory=list, description="报告里出现的后端名（升序去重）")
    metrics: list[str] = Field(default_factory=list, description="本次实际对账的度量（规范名）")
    dimension: int = Field(default=0, description="这批记录的向量维度")


# --------------------------------------------------------------------------- #
# day065 索引构建（M6-D4）：/indexing/* 七个端点
#
# 与 /vectorstore/* 同一纪律（依赖注入、响应体全部可直接 json.dumps），
# 但有两条**本组独有的约定**，它们写在模型这一层而不是端点里：
#
# 1. **清单只以"不含条目"的形式出现**（``IndexManifest.to_dict(include_entries=False)``）。
#    一份十万条的清单把 entries 全塞进响应体接近 10 MB，而 status / versions
#    要回答的只是"现在生效的是哪一版、有几版"——条目本身属于清单文件，
#    不属于每次调用都要读一遍的接口。
# 2. **请求体里没有"选后端 / 选编码器"**：后端与编码器由 ``app.state`` 决定
#    （与 day064 同一条纪律）。允许一个请求换编码器，会让"这份索引是用谁建的"
#    变成一件要靠翻日志才能回答的事——而它恰恰是排查"检索结果不对"时
#    第一个要问的问题。
#
# 每个响应体都带一句 ``summary``（``summary_line()`` 的原样投影或端点拼的一行）：
# 报告要能被人读完，细节由 ``stats`` / ``plan`` / ``report`` 这些结构化字段承担。
# --------------------------------------------------------------------------- #


class IndexingStatusResponse(BaseModel):
    """GET /indexing/status 的响应体（M6-D4）：当前索引状态 + 本实例最近一次构建报告.

    ``stats`` 直接是 ``IndexingPipeline.stats()`` 的原样投影，键集合固定在
    九个上（backend / metric / dimension / count / identity / manifest /
    versions / backups / cache）。三处刻意的不对称都由那一层保证：

    ```text
    dimension   报的是**库**的维度（库还没建时 0）；编码器的维度在 identity.dimension 里
    manifest    当前生效的那一版清单（不含 entries）；还没构建过就是 None
    count       报的是后端的真实条数——它才是"库里有什么"的唯一来源
    ```

    ``manifest`` 的取值在端点里补了一步：流水线"亲手构建过的那一版"为空时，
    回落到版本表 ``current`` 指向的那一版（``/indexing/build`` 直接走
    ``IndexBuilder``，因此那一版由版本表登记并采纳）。两者都是"现在生效的
    是哪一版"，而端点**不去磁盘上找**清单文件——那需要回答"清单文件在哪、
    它是不是现在生效的那份"，答案在版本表与配置里。

    ``last_report`` 是**本应用实例**最近一次 /indexing/build 的 IndexingReport；
    还没构建过就是 ``None``。它与 ``app.state.indexing_last_report`` 是同一份数据，
    因此换一个 app 实例它必须是 ``None``——那是"内存后端的写入只活在这个实例里"
    这句话的可核对形式。
    """

    summary: str = Field(default="", description="一行结论")
    stats: dict = Field(default_factory=dict, description="IndexingPipeline.stats() 的投影")
    last_report: dict | None = Field(
        default=None, description="本实例最近一次构建报告；None 表示还没构建过"
    )
    indexing_dir: str = Field(
        default="", description="本实例的索引目录；空串表示明确不落盘"
    )


class IndexingPlanRequest(BaseModel):
    """POST /indexing/plan 的请求体（M6-D4）：给一批记录 → 算差集（**只读**）.

    ``records`` 与 ``/indexing/build`` 收的是同一个形状（day062 的
    ``knowledge_records()``：``doc_id`` / ``source`` / ``text`` / ``metadata``），
    因此"先看看计划、再决定要不要构建"用的是同一份输入，不必转格式。

    ``mode`` 留空表示用 ``settings.indexing_mode``；它只影响"模式判定"那一栏，
    不会让本端点写任何东西——计划是**看**，不是**做**。
    """

    records: list[dict] = Field(default_factory=list, description="day062 形状的知识库记录")
    mode: str | None = Field(
        default=None, description="要预览的构建模式（full / incremental）；None 用 settings"
    )


class IndexingPlanResponse(BaseModel):
    """POST /indexing/plan 的响应体（M6-D4）.

    ``plan`` 是 ``IndexPlan.to_dict()``（含四组 id），``reason`` 是
    ``planner.plan_reason`` 的那句话——它把"换了编码器导致全部重算"
    与"数据真的变了很多"分开，两者的下一步动作完全不同。

    ``mode`` / ``switched`` / ``threshold`` 三件一起回答"这次会不会换挡"：

    ```text
    mode          按当前计划与阈值判定后**实际**会用的模式
    switched      是否因为变更比例超过阈值而从 incremental 切到 full
    threshold     settings.indexing_full_rebuild_threshold 的当前取值
    ```

    判定的依据只有一条：增量模式下 ``plan.changed / plan.total > threshold``。
    真正的换挡在构建时由 ``IndexBuilder`` 执行，本端点只是**预览**同一套规则。
    """

    summary: str = Field(default="", description="一行结论")
    plan: dict = Field(default_factory=dict, description="IndexPlan 的完整投影（含四组 id）")
    reason: str = Field(default="", description="plan_reason 的一句话（含配置变更的区分）")
    requested_mode: str = Field(default="", description="请求里的模式（缺省已解析成 settings）")
    mode: str = Field(default="", description="按阈值判定后实际会用的模式")
    switched: bool = Field(default=False, description="是否因超过阈值切到 full")
    threshold: float = Field(default=0.0, description="settings.indexing_full_rebuild_threshold")
    parent_version: str = Field(default="", description="差集所对的上一版；空串表示首版")
    identity: dict = Field(default_factory=dict, description="本实例编码器的身份")


class IndexingBuildRequest(BaseModel):
    """POST /indexing/build 的请求体（M6-D4）：执行一次构建.

    ``mode`` 非法 → **400**（消息里列出 ``full`` / ``incremental``，
    不做大小写与别名的宽容：猜错的代价是一次全量重建）。

    ``backup`` 缺省 ``false``：备份是**运维决策**，默认给每次构建都拷一份快照
    会让"备份为什么这么多"变成一个没人答得上来的问题。需要一个可回去的点时
    显式传 ``true``，或直接调 ``/indexing/backup``。
    """

    records: list[dict] = Field(default_factory=list, description="day062 形状的知识库记录")
    mode: str | None = Field(
        default=None, description="构建模式：full / incremental；None 用 settings.indexing_mode"
    )
    reason: str = Field(default="", description="只进备份记录：这份快照为什么建的")
    backup: bool = Field(default=False, description="构建成功后是否顺便创建一份备份")


class IndexingBuildResponse(BaseModel):
    """POST /indexing/build 的响应体（M6-D4）：``IndexingReport`` 的原样投影.

    ``report`` 里的五个数字各自回答一个问题，**必须分开读**：

    ```text
    written      本次真正写进库的条数（added + updated）→ "改了什么"
    unchanged    一条都没碰库的条数（全量模式恒为 0）  → "这次其实什么都没做"
    encoded      真正送给提供方的条数                  → "花了多少次编码"
    cache_hits   其中靠缓存免掉的                      → "省下了多少次"
    batches      向提供方发起了几批                    → "批量有没有生效"
    ```

    ``reuse_ratio`` 的口径随模式变（增量是"相对上一版没变的比例"，
    全量是"这一次请求的缓存命中率"）——见 ``IndexingReport`` 的说明。
    """

    summary: str = Field(default="", description="report.summary_line()")
    report: dict = Field(default_factory=dict, description="IndexingReport 的完整投影")
    version_id: str = Field(default="", description="本次构建的版本号")
    count: int = Field(default=0, description="构建之后库里的记录条数")
    manifest: dict = Field(
        default_factory=dict, description="本次清单的投影（不含 entries）"
    )


class IndexingVersionsResponse(BaseModel):
    """GET /indexing/versions 的响应体（M6-D4）：版本历史 + 当前指针.

    ``history`` 的每一版只给 ``version_id`` 与 ``summary_line``：
    版本表里的每一版都是一份完整清单，把它们全部塞进响应体等于把历史索引的
    全文都发出去，而这里要回答的只是"有哪几版、各是什么"。

    ``current`` 是**被采纳的**那一版（与"最近登记的"不是一回事）：
    构建失败或验证不过的版本会被登记但从不采纳，两者的差别正是回滚要用的信息。
    """

    summary: str = Field(default="", description="一行结论")
    history: list[dict] = Field(default_factory=list, description="逐版 {version_id, summary_line}")
    current: str = Field(default="", description="当前采纳的版本号；空串表示还没采纳过")
    count: int = Field(default=0, description="历史里的版本数")
    lineage: list[str] = Field(
        default_factory=list, description="current 沿 parent_version 回溯的血缘（旧 → 新）"
    )


class IndexingVerifyRequest(BaseModel):
    """POST /indexing/verify 的请求体（M6-D4）：清单与库的一致性体检.

    ``version_id`` 留空表示用**当前采纳的清单**；给了就用版本表里那一版
    （不存在 → 400，消息里说明"版本表里没有这一版"）。
    两种都不满足时（还没构建过、版本表为空）用 ``manifest_from_store``
    按库的现状现算一份——那是"清单丢了但库还在"时的补救路径。
    """

    version_id: str | None = Field(
        default=None, description="要对账的版本号；None 表示用当前清单"
    )


class IndexingVerifyResponse(BaseModel):
    """POST /indexing/verify 的响应体（M6-D4）.

    ``ok`` 为假意味着四类问题里至少有一类（清单有库没有 / 库有清单没有 /
    维度不符 / 度量或后端不符）。``problems`` 是**句子的形式**而不是错误码：
    读它的人是运维，每一句里都已经包含现象、数量与下一步。

    本端点**不做任何修复**：它的名字是 verify，不是 repair。
    顺手补一条缺失的记录会让"库与清单不一致"这件事下一次也查不出来。
    """

    summary: str = Field(default="", description="一行结论")
    ok: bool = Field(default=False, description="四类问题一个都没有才为真")
    checks: dict = Field(default_factory=dict, description="verify_index 的逐项结论")
    problems: list[str] = Field(default_factory=list, description="可直接照做的中文结论")
    version_id: str = Field(default="", description="本次对账用的是哪一版清单")
    source: str = Field(
        default="",
        description="清单来源：version（指定版本）/ current（当前采纳）/ store（按库现算）",
    )


class IndexingBackupRequest(BaseModel):
    """POST /indexing/backup 的请求体（M6-D4）：手动创建一份快照.

    ``reason`` 只进备份记录（"这份快照是因为什么建的"），不进清单——
    清单的 ``metadata`` 不参与版本号，但把一次操作的临时理由写进版本记录
    会让版本表变成日志。
    """

    reason: str = Field(default="", description="这份备份为什么建的（只进备份账本）")


class IndexingBackupResponse(BaseModel):
    """POST /indexing/backup 的响应体（M6-D4）：``BackupRecord`` 的原样投影.

    ``files`` 列出拷进快照的文件名。**内存后端没有库文件可备份**，
    此时它会是空列表，``note`` 里说明"本次只备份了清单"——
    这与"源文件缺失被静默跳过"是两件事：后者是假备份，前者是如实告知。

    ``keep`` 是保留份数上限（``settings.indexing_backups_keep``）：
    ``create`` 之后总是自动裁剪，因此账本里曾经有过、目录里未必还在。
    """

    summary: str = Field(default="", description="一行结论")
    backup: dict = Field(default_factory=dict, description="BackupRecord 的完整投影")
    backup_id: str = Field(default="", description="这份快照的存储位置（version + backend + 序号）")
    version_id: str = Field(default="", description="它备份的是哪一版（身份）")
    files: list[str] = Field(default_factory=list, description="拷进快照的文件名（升序）")
    size_bytes: int = Field(default=0, description="这份快照占用的字节数")
    keep: int = Field(default=0, description="备份保留份数上限")
    note: str = Field(default="", description="本次备份的补充说明（例如内存后端只备份清单）")


class IndexingRollbackRequest(BaseModel):
    """POST /indexing/rollback 的请求体（M6-D4）：沿血缘回到 N 步之前.

    ``steps`` 数的是**血缘上的步数**，不是登记顺序上的步数：版本表里存在
    构建失败、从未被采纳的版本时，按时间倒序回滚会切到一个从没生效过的版本。
    ``steps < 1`` → 400（"回退 0 步"不是一次回滚）。
    """

    steps: int = Field(default=1, description="要沿血缘回退的步数；必须 >= 1")


class IndexingRollbackResponse(BaseModel):
    """POST /indexing/rollback 的响应体（M6-D4）.

    **回滚只改版本指针，不动库与数据**：它把 ``current`` 从一版移到另一版，
    既不清库也不重建向量。要真正回到那一版的内容，请在那之后调
    ``/indexing/build``（带上对应的记录），或从备份 ``restore``——
    本端点的返回值里没有任何条目数变化，因为一次回滚本就不该有。
    """

    summary: str = Field(default="", description="一行结论")
    previous: str = Field(default="", description="回滚前的 current")
    current: str = Field(default="", description="回滚后的 current（新的当前版本）")
    steps: int = Field(default=1, description="本次回退的血缘步数")
    manifest: dict = Field(
        default_factory=dict, description="回滚到的这一版清单投影（不含 entries）"
    )
    note: str = Field(
        default="",
        description="回滚是运维动作：不自动重建索引；重建请调用 /indexing/build",
    )


# --------------------------------------------------------------------------- #
# day066 检索（M6-D5）：/retrieval/* 五个端点
#
# 与 /vectorstore/* 和 /indexing/* 同一纪律（依赖注入、响应体全部可直接
# json.dumps、请求体里没有"选后端 / 选编码器"），但有三条本组独有的约定：
#
# 1. **请求体只描述"怎么问"**（query / top_k / fetch_k / where / time_range /
#    min_score / max_per_doc / route），一个字段都不描述"库长什么样"。
#    这一组契约正是 ``config`` 里 ``retrieval_*`` 那一组的镜像：它不改变任何
#    产物字节，只改变"怎么问"。
# 2. **空库是合法状态**：请求体里没有"库里必须有数据"这种前置条件，
#    空库对应的是 ``empty_reason="no_data"`` 的 200，而不是 500。
# 3. **三个减法数字 + 一个空结果原因原样出门**：``dropped_below_threshold`` /
#    ``dropped_by_diversity`` / ``dropped_by_top_k`` 与 ``empty_reason`` 直接来自
#    ``RetrievalResult.to_dict()``，端点不重新解释它们——这一组的全部价值
#    就在于"为什么只有这几条"能被读出来，而重新解释一次就等于把那份交代改写了一遍。
# --------------------------------------------------------------------------- #


class RetrievalTimeRange(BaseModel):
    """检索请求里的时间范围（M6-D5）：一个字段 + 一个**闭区间**.

    三条约定与 ``retrieval.types.TimeRange`` 逐条相同，这里只写接口侧的那一半：

    ```text
    field 省略            → 用 DEFAULT_TIME_FIELD（created_at）
    两端可单独省略         → 只给 start 是"这段之后"，只给 end 是"这段之前"
    倒置的区间（start > end）→ 检索层报 400，消息里点明"倒置的区间必然返回空结果"
    ```

    为什么把它做成子对象而不是塞进 ``where``：塞进 ``where`` 之后，
    "同一个字段被两组条件同时过滤"只能变成一次静默取交集，
    而结果只会变少、不会报错（见 ``retrieval.filters.combine_where`` 的冲突检测）。
    """

    field: str = Field(
        default=DEFAULT_TIME_FIELD,
        description="时间字段名；省略时用 retrieval.filters.DEFAULT_TIME_FIELD",
    )
    start: str | None = Field(
        default=None, description="闭区间下界（ISO-8601 文本）；None 表示不设下界"
    )
    end: str | None = Field(
        default=None, description="闭区间上界（ISO-8601 文本）；None 表示不设上界"
    )


class RetrievalSearchRequest(BaseModel):
    """POST /retrieval/search 与 POST /retrieval/explain 的请求体（M6-D5）.

    八个字段里只有 ``query`` 是必填的，**其余全部可省略**：
    "用项目默认值检索一段文本"是最常见的调用，不该要求它把默认值抄一遍。
    ``None`` 与 ``0`` 的语义差别写在这里，因为它决定了"没提"与"提了"：

    ```text
    top_k / fetch_k / max_per_doc   None = 取 settings.retrieval_*；
                                   0 会被检索层拒成 400（"最多 0 条"不是一个请求）
    min_score                       None = 不设阈值（**不是阈值 0**：l2 度量下分数为负）
    where / time_range              None = 不过滤；``{}`` 与 None 在本包里等价于"无过滤"
    route                           None = 走路由器规则（默认路由 / 只有一个索引时用它）
    ```

    ``explain`` 与 ``search`` 共用本模型是刻意的：**它们是同一次检索的两个出口**
    （一份结果、一份诊断）。两份请求体一旦分家，"诊断描述的那次检索"
    与"返回结果的那次检索"就会慢慢变成两次不同的检索。
    """

    query: str = Field(default="", description="查询文本（空串或空白串 → 400）")
    top_k: int | None = Field(
        default=None, description="返回条数上限；None 用 settings.retrieval_top_k"
    )
    fetch_k: int | None = Field(
        default=None, description="召回深度（必须 >= top_k）；None 按过取倍率算"
    )
    where: dict | None = Field(
        default=None, description="元数据过滤子句（与 /vectorstore/search 同一套语法）"
    )
    time_range: RetrievalTimeRange | None = Field(
        default=None, description="时间范围（闭区间）；None 表示不设时间条件"
    )
    min_score: float | None = Field(
        default=None, description="相似度下限（越大越近）；None 表示不设阈值"
    )
    max_per_doc: int | None = Field(
        default=None, description="每篇文档最多留几条；None 用 settings.retrieval_max_per_doc"
    )
    route: str | None = Field(
        default=None, description="多索引路由名；None 走路由器规则，未知名字 → 400 并列出可用名"
    )


class RetrievalSearchResponse(BaseModel):
    """POST /retrieval/search 的响应体（M6-D5）：一份检索结果 + 它的诊断.

    ``result`` 是 ``RetrievalResult.to_dict(include_text=True)`` 的原样投影
    （正文包含在内：这个端点的用途就是"看一眼检索结果"，而报告口径的默认值
    是不带正文）。其中五个数字一起才说得出"为什么只有这几条"：

    ```text
    fetch_k                  本来打算取多深（被 MAX_FETCH_K 封顶时 notes 里会写）
    candidates               过滤之后还剩多少条可选
    dropped_below_threshold  阈值切掉几条      → 去调 settings.retrieval_min_score
    dropped_by_diversity     每文档上限挤掉几条 → 去调 settings.retrieval_max_per_doc
    dropped_by_top_k         截断到 top_k 丢掉几条 → 这只是"取够了"
    ```

    ``lines`` 是 ``Retriever.explain()`` 的逐行诊断（四问 + 字段拼写检查 + 漂移处置），
    ``conditions`` / ``filter_fields`` 是过滤条件的可读渲染与字段名——
    空结果的第一个排查动作就是核对这两项（与 ``/vectorstore/search`` 同源）。
    ``routes`` 列出本实例的可用路由名：route 写错时不必再发一次请求去问有哪些。
    ``empty_reason`` 是**唯一**原因，``empty_reason_description`` 是它的人话解释
    （来自 ``retrieval.types.EMPTY_REASON_DESCRIPTIONS``，不是端点现编的一句话）。
    """

    summary: str = Field(default="", description="一行结论")
    result: dict = Field(default_factory=dict, description="RetrievalResult 的完整投影（含正文）")
    lines: list[str] = Field(default_factory=list, description="逐行人话诊断（四问）")
    conditions: str = Field(default="", description="过滤条件的一行可读描述")
    filter_fields: list[str] = Field(default_factory=list, description="where 里出现过的字段名")
    empty_reason: str = Field(default="", description="空结果的唯一原因；有命中时是 hits")
    empty_reason_description: str = Field(default="", description="该原因的人话解释")
    route: dict = Field(default_factory=dict, description="选路结论 {name, reason, matched}")
    routes: list[str] = Field(default_factory=list, description="本实例的可用路由名（升序）")


class RetrievalExplainResponse(BaseModel):
    """POST /retrieval/explain 的响应体（M6-D5）：**只有诊断，不返回正文**.

    与 ``/retrieval/search`` 的差别只有一处：这里把 ``Retriever.explain()``
    的 ``lines`` 放到主体位置，并给出 ``count`` / ``ids`` 让调用方不必再解析
    ``result``。两个端点分开而不是加一个 ``explain=true`` 开关，理由与
    day065 的 "/indexing/plan 是看、/indexing/build 是做" 同源：
    **一个返回体的形状只服务一个用途**，"要不要诊断"这种开关会让
    客户端代码在这两种形状之间分支。

    ``lines`` 的四行固定回答四件事（哪一版索引 / 什么过滤 / 为什么条数少 /
    为什么是空的），后两行是检索器补的字段拼写检查与漂移处置建议。
    """

    summary: str = Field(default="", description="一行结论")
    empty_reason: str = Field(default="", description="空结果的唯一原因；有命中时是 hits")
    empty_reason_description: str = Field(default="", description="该原因的人话解释")
    lines: list[str] = Field(default_factory=list, description="逐行人话诊断")
    count: int = Field(default=0, description="命中条数")
    ids: list[str] = Field(default_factory=list, description="命中的记录 id（按名次）")
    conditions: str = Field(default="", description="过滤条件的一行可读描述")
    route: dict = Field(default_factory=dict, description="选路结论 {name, reason, matched}")
    routes: list[str] = Field(default_factory=list, description="本实例的可用路由名（升序）")


class RetrievalAnswerRequest(BaseModel):
    """POST /retrieval/answer 的请求体（M6-D5）：一个问题 + 同一组检索参数.

    字段与 ``RetrievalSearchRequest`` **逐项相同**（含默认值），只把
    ``query`` 换成 ``question``：名字不同是刻意的——``query`` 是被编码的
    那句话，``question`` 是被回答的那个问题，而这一条链路上它们是同一段文本
    （``RagPipeline`` 用**检索之后规范化过的**那份文本，见 ``pipeline.answer``）。

    两处刻意的不提供：

    ```text
    没有 temperature / max_tokens    RAG 问答是"有依据的复述"，采样参数属于
                                     settings 与装配（RagPipeline 的构造参数）
    没有 prompt 文本                允许一个请求换提示词，会让"这次答案质量变了"
                                     无法归因到版本号（day026 的版本化约定）
    ```
    """

    question: str = Field(default="", description="问题文本（空串或空白串 → 400）")
    top_k: int | None = Field(
        default=None, description="检索条数上限；None 用 settings.retrieval_top_k"
    )
    fetch_k: int | None = Field(
        default=None, description="召回深度（必须 >= top_k）；None 按过取倍率算"
    )
    where: dict | None = Field(default=None, description="元数据过滤子句")
    time_range: RetrievalTimeRange | None = Field(
        default=None, description="时间范围（闭区间）；None 表示不设时间条件"
    )
    min_score: float | None = Field(
        default=None, description="相似度下限（越大越近）；None 表示不设阈值"
    )
    max_per_doc: int | None = Field(
        default=None, description="每篇文档最多留几条；None 用 settings.retrieval_max_per_doc"
    )
    route: str | None = Field(default=None, description="多索引路由名；未知名字 → 400")


class RetrievalAnswerResponse(BaseModel):
    """POST /retrieval/answer 的响应体（M6-D5）.

    ``llm_called`` 是本组最重要的一个字段，它不是装饰：
    它是"**检索不到时不许让模型自由发挥**"这条护栏的唯一证据。
    检索为空时它必须是 ``False``，``answer`` 是兜底答复、``citations`` 为空、
    ``context`` 为 ``None``——判定不能靠"看答案等不等于兜底那句话"
    （那句话改一个字，这种判定就静默失效了）。

    ``context`` 与 ``retrieval`` 一起回答"模型这次看到了什么"：
    ``context.text`` 是真正进提示词的那段带 ``[n]`` 编号的片段，
    ``context.citations`` 是编号对照表，``retrieval`` 是那次检索的完整交代
    （含 ``empty_reason`` 与索引版本 / 漂移）。

    ``route`` 是选路结论：RAG 链路一次只有一个检索器，因此**选路在端点层做完**
    （``RagPipeline`` 不认识路由表），而"这次是谁答的、是显式指定还是路由器选的"
    必须被回显——结果不对时第一件要确认的事就是"该不该是它"。
    """

    summary: str = Field(default="", description="一行结论")
    answer: str = Field(default="", description="模型给的答案，或未命中时的兜底答复")
    llm_called: bool = Field(default=False, description="这次到底调没调模型（护栏是否生效的证据）")
    prompt_version: str = Field(default="", description="本次用的提示词版本号")
    citations: list[dict] = Field(default_factory=list, description="编号 → 命中的对照表")
    context: dict | None = Field(
        default=None, description="进提示词的那段文本及其账；未调模型时为 None"
    )
    retrieval: dict | None = Field(default=None, description="那次检索的完整交代")
    notes: list[str] = Field(default_factory=list, description="降级与护栏注记")
    empty_reason: str = Field(default="", description="那次检索的空结果原因（有命中时是 hits）")
    route: dict = Field(default_factory=dict, description="选路结论 {name, reason, matched}")


class RetrievalStatusResponse(BaseModel):
    """GET /retrieval/status 的响应体（M6-D5）：检索器配置 + 索引现状 + 缺省值表.

    三块内容**不混在一起**，理由与 ``/vectorstore/stats`` 的三块同源：

    ```text
    retriever   这个检索器被装配成什么样（top_k / 过取倍率 / 阈值 / 每文档上限 /
                分组字段 / 时间字段），以及 index{version_id, count, drift}
    defaults    settings 里 retrieval_* 那一组的当前取值——"为什么是这个数"的出处
    router      路由表的名字与当前选中项（谁答的、是显式指定还是路由器选的）
    ```

    ``index`` 是从 ``retriever`` 里**再取一次**的便捷字段（``describe()["index"]``）：
    "这一版索引还作数吗"是最高频的一个问题，不该要求调用方在嵌套字典里找它。
    注意 ``index.count_store`` 是**库**的条数，``count_manifest`` 是清单说的条数，
    两者相减就是漂移的规模；``has_drift`` 为真时 ``drift`` 里是逐条成因。

    ``empty_reasons`` 把四种空结果成因与人话解释一起给出：空结果不是"没找到"
    这一句话，而是四种动作完全不同的处境（见 ``retrieval.types``）。
    ``limitations`` / ``out_of_scope`` 与 ``docs/retrieval.md`` 同源，
    避免"文档写着能做、代码里没做"这类漂移。
    """

    summary: str = Field(default="", description="索引状态的一行结论")
    retriever: dict = Field(default_factory=dict, description="Retriever.describe() 的投影")
    index: dict = Field(default_factory=dict, description="索引现状（与 retriever.index 同一份）")
    route: dict = Field(default_factory=dict, description="选路结论 {name, reason, matched}")
    router: dict = Field(default_factory=dict, description="路由表概览 {count, default, names}")
    defaults: dict = Field(default_factory=dict, description="settings.retrieval_* 的当前取值")
    empty_reasons: dict = Field(default_factory=dict, description="四种空结果成因的人话解释")
    limitations: list[str] = Field(default_factory=list, description="本层的已知限制（含依据）")
    out_of_scope: list[str] = Field(default_factory=list, description="明确排除在范围外的能力")


class RetrievalRoutesResponse(BaseModel):
    """GET /retrieval/routes 的响应体（M6-D5）：路由清单.

    每个路由带三样东西：**名字**（怎么问）、**说明**（为什么有它）、
    **检索器的现状**（``describe()``：深度 / 阈值 / 索引版本 / 漂移）。
    三者缺一都会让"这一路为什么没结果"变得无法回答——例如漂移在
    ``routes[].retriever.index`` 里，而它恰恰是最常见的原因。

    ``count`` / ``names`` / ``default`` 是 ``names()`` 与 ``default`` 的投影，
    与 ``routes`` 的长度**刻意重复**：调用方最常问的两件事是"有几条路"
    与"不传 route 会走哪一条"，让它们出现在顶层就不必解析列表。
    """

    summary: str = Field(default="", description="一行结论")
    count: int = Field(default=0, description="已注册的路由数")
    default: str = Field(default="", description="默认路由名；空串表示没设默认")
    names: list[str] = Field(default_factory=list, description="全部路由名（升序）")
    routes: list[dict] = Field(
        default_factory=list, description="逐路由 {name, description, retriever}"
    )
    limitations: list[str] = Field(default_factory=list, description="本层的已知限制（含依据）")
    out_of_scope: list[str] = Field(default_factory=list, description="明确排除在范围外的能力")
