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
