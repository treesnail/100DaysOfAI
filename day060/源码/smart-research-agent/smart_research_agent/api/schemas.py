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
