"""配置管理：使用 Pydantic Settings 从环境变量与 .env 文件加载配置."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置模型.

    属性默认值提供开箱即用的本地开发体验；生产环境可通过环境变量覆盖。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_name: str = "智研 AI 助手"
    debug: bool = False
    log_level: str = "INFO"
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    default_model: str = "gpt-4o-mini"
    # embedding 选型（day041）：mock / char-ngram / sentence-transformer / openai
    embedding_provider: str = "char-ngram"
    # sentence-transformer 提供方使用的本地模型名
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    # openai 提供方使用的云端 embedding 模型名
    openai_embedding_model: str = "text-embedding-3-small"
    # 语义缓存（day043）：相似度命中阈值与最大容量
    cache_similarity_threshold: float = 0.85
    cache_max_size: int = 128
    # 本地模型部署（day045）
    # llm_backend: cloud（默认，走云端 OpenAI 兼容端点）/ local（走本机服务）
    llm_backend: str = "cloud"
    # local_backend: ollama（默认，Ollama 默认端口 11434）/ vllm（默认端口 8000）
    local_backend: str = "ollama"
    # 本地模型名，须与 ollama list / vLLM --served-model-name 一致
    local_model: str = "qwen3:8b"
    # 本地 OpenAI 兼容端点（缺 /v1 时由 normalize_base_url 自动补齐）
    local_base_url: str = "http://localhost:11434/v1"
    # 本地服务占位密钥：Ollama 不校验（"required but ignored"），vLLM 未加 --api-key 时也不校验
    local_api_key: str = "ollama"
    # 上下文窗口：对齐 Ollama 默认 4096（OLLAMA_CONTEXT_LENGTH 可覆盖）
    local_context_length: int = 4096
    # 模型空闲常驻时长（仅 Ollama 生效）：5m 默认 / -1 常驻不卸载 / 0 用完即卸
    local_keep_alive: str = "5m"
    # 本地视觉模型（如 qwen3-vl 系列）才置 True
    local_supports_vision: bool = False
    # 本地 embedding 模型（Ollama 常用 nomic-embed-text，维度 768）
    local_embedding_model: str = "nomic-embed-text"
    # 一体化流水线（day046）：把 prompt/路由/缓存/成本/安全串成一条调用链
    # 是否启用语义缓存阶段（关闭后每次请求都真实调用 LLM）
    pipeline_cache_enabled: bool = True
    # 输入侧护栏命中注入时是否直接拒答（False 则只记录不拦截）
    pipeline_block_on_injection: bool = True
    # 性能基线（day046）：参考基线文件路径（相对进程工作目录）
    perf_baseline_path: str = "data/eval/perf_baseline.json"
    # 回归容忍度（当前值 / 基线值 的倍数上限）：超过即判定回归
    perf_latency_tolerance: float = 1.5
    perf_token_tolerance: float = 1.2
    perf_cost_tolerance: float = 1.2
    # 微调数据工程（M5-D1）：种子数据目录、清洗阈值与切分参数
    # 种子样本与落盘产物所在目录（相对进程工作目录）
    finetune_data_dir: str = "data/finetune"
    # 手写种子样本文件路径（默认在 finetune_data_dir 下）
    finetune_seed_path: str = "data/finetune/seed_examples.jsonl"
    # 清洗规则：输出短于此长度视为质量不合格（清洗后按字符数计）
    finetune_min_output_chars: int = 8
    # 清洗规则：输出长于此长度视为粘贴长文而非答案
    finetune_max_output_chars: int = 4000
    # 清洗规则：指令长于此长度需单独评测长上下文，默认不放入训练集
    finetune_max_instruction_chars: int = 1000
    # 训练/评估切分比例（评估集占比，落在 [0, 1) 区间）
    finetune_eval_ratio: float = 0.2
    # 切分随机种子：固定后同一份数据永远得到同一个切分，便于对比实验
    finetune_split_seed: int = 42
    # SFT 监督微调（M5-D2）：以下九个默认值按"**参考模型 + 本课程数据集**"
    # 标定，即本机离线可跑的那套配置（`scripts/sft_demo.py` 与
    # `/finetune/sft/*` 端点都用它）。换成真实的 7B 全参训练时，学习率与
    # 批大小**必须重新标定**——学习率没有绝对尺度（见 `sft/args.py` 里
    # `LR_SOFT_RANGE` 与 `LR_SOFT_RANGE_REFERENCE` 的说明）。
    # 训练产物目录（检查点写在它下面的 checkpoint-{step} / final）
    sft_output_dir: str = "outputs/sft"
    # 单条样本的最大 token 数；缺省由本数据集渲染后的长度分位数（p100=320）
    # 向上取整到 32 的倍数得到，不要照抄别人的值
    sft_max_length: int = 384
    # 训练轮数：参考模型把 9 步提到 30 步，让 loss 下降肉眼可见
    sft_num_train_epochs: float = 10.0
    # 学习率：参考模型（557×557 bigram + 纯 SGD）的标定值，比 7B 全参大 4 个数量级
    sft_learning_rate: float = 8.0
    # 单设备训练批大小（显存的主要决定项）
    sft_per_device_train_batch_size: int = 2
    # 梯度累积步数：与批大小相乘得到有效批（缺省 2 × 4 = 8 条/次更新）
    sft_gradient_accumulation_steps: int = 4
    # warmup 占总步数的比例；注意 total_steps 小到个位数时它会取整为 0
    sft_warmup_ratio: float = 0.03
    # 学习率调度器（HF SchedulerType 的字符串取值）
    sft_lr_scheduler_type: str = "cosine"
    # 全局随机种子（权重初始化与数据打乱的可复现性）
    sft_seed: int = 42
    # LoRA / QLoRA 参数高效微调（M5-D3）：以下十一个默认值按 LoRA 官方文档与
    # QLoRA 论文的常见组合标定（r=8 / alpha=16 / dropout=0.05 / 适配 q,v /
    # nf4 + 二级量化 + 块大小 64）。换成别的模型时 `lora_target_preset` 必须
    # 重新核对：**目标模块没命中只会表现为"效果不如预期"，不会报错**
    # （见 `peft/targets.py` 的校验与 `peft/config.py` 的预设展开）。
    # 适配器与训练产物目录：LoRA 只落盘 A/B 两个小矩阵，量级是十几 MB
    lora_output_dir: str = "outputs/lora"
    # 低秩维 r：方阵单层比例 = 2r/d；参考模型（557×557）上 r=8 时全模型占比 2.787%
    lora_r: int = 8
    # 缩放分子：实际生效的缩放是 alpha/r（rsLoRA 时为 alpha/sqrt(r)），
    # 单独看 alpha 没有意义
    lora_alpha: int = 16
    # LoRA 分支的 dropout：只作用在 A 的输入上，基座路径不受影响
    lora_dropout: float = 0.05
    # 目标模块预设：attention / attention_all / mlp / all_linear / bigram
    lora_target_preset: str = "attention"
    # 偏置策略：none（全冻结，缺省）/ all / lora_only
    lora_bias: str = "none"
    # 是否使用 rank-stabilized LoRA（缩放改为 alpha/sqrt(r)，让不同秩之间可比）
    lora_use_rslora: bool = False
    # 4-bit 码本：nf4（QLoRA 的选择，正态分位数）/ fp4（E2M1）
    qlora_quant_type: str = "nf4"
    # 反量化后的计算精度（4-bit 只用于存基座，计算与适配器都是它）
    qlora_compute_dtype: str = "bfloat16"
    # 是否启用二级（常数量化）：每参数存储 4.5 bit → 4.126953 bit
    qlora_use_double_quant: bool = True
    # 一级量化的块大小：每多少个权重共享一个 absmax 常数（论文取 64）
    qlora_block_size: int = 64
    # 训练脚本与 PEFT 实践（M5-D4）：适配器生命周期 + 多卡/混合精度的五个参数。
    # 注意 `peft_adapter_save_steps` 缺省 0：只在训练结束时落盘一次适配器——
    # 中间适配器是"续训与回滚"才需要的东西，缺省打开只会让产物目录变乱。
    # 适配器落盘目录（一个适配器 = 三个文件，量级几十 KB ~ 十几 MB）
    peft_adapter_dir: str = "outputs/lora/adapters"
    # 每多少步落一次适配器；0 表示只在训练结束时落盘一次
    peft_adapter_save_steps: int = 0
    # 最多保留多少个**中间**适配器（adapter-final 永不删除）
    peft_adapter_save_total_limit: int = 2
    # 训练设备数：1 表示单卡（此时所有分片策略都退化为 ddp）
    peft_devices: int = 1
    # 分片策略：ddp（每个设备一份完整副本）/ zero2（切优化器状态与梯度）/
    # zero3（参数也切，代价是每层前向都要 all-gather）
    peft_sharding_strategy: str = "ddp"
    # 混合精度：no / fp16（需要 loss scaling）/ bf16（Ampere 及之后优先）
    peft_mixed_precision: str = "bf16"
    # 批大小被多卡放大后学习率的缩放法则：linear（同倍放大）/ sqrt（保守）/
    # none（不缩放，此时 plan_distributed 会给出"梯度变小"的告警）
    peft_lr_scaling_mode: str = "linear"
    # 微调模型评估（M5-D5）：领域评估集的切分、门禁与统计参数。
    # **注意两个 "eval ratio" 不是一回事**：上面的 `finetune_eval_ratio` 切的是
    # **微调数据**（day048 的数据工程，回答"留多少条做验证"），下面的
    # `finetune_eval_suite_ratio` 切的是**评估套件**（回答"18 条领域用例里
    # 留几条做评估"）。两者名字都含 eval，混用会让"评估集"这个词指两件事——
    # 本课把它们显式分开并在文档里点名。
    # 评估套件与报告落盘目录（报告 = JSON + markdown 两份）
    finetune_eval_dir: str = "outputs/finetune_eval"
    # 评估套件的评估集占比（按能力桶分层；每桶至少留 1 条进评估集）
    finetune_eval_suite_ratio: float = 0.25
    # 评估套件切分与配对自助法的随机种子（固定后报告可逐位复现）
    finetune_eval_seed: int = 42
    # 合格率门禁下限：低于它报告判不通过（缺省 0.5，可在请求里覆盖）
    finetune_eval_min_pass_rate: float = 0.5
    # 每个分组（能力桶/难度）允许的最大合格率回退：0.0 = 一处都不许掉
    finetune_eval_max_regression: float = 0.0
    # 配对自助法的重采样次数：越大区间越稳，2000 次在本课规模上不到 20 ms
    finetune_eval_bootstrap_samples: int = 2000
    # 配对比较的显著性水平（McNemar 精确检验与自助法区间共用）
    finetune_eval_alpha: float = 0.05
    # RLHF / DPO 对齐（M5-D6）：偏好数据切分、DPO 超参与过优化门禁。
    # 注意下面三个超参都是**本课在参考模型上实测标定**的值，不是从论文抄来的：
    # `alignment_beta` 与 KL 不是单调关系（固定步数时 KL 随 β 先升后降，
    # 见 docs/alignment.md 的 KL 表），照抄别人的 β 会得到完全不同的偏离程度。
    # 对齐产物目录（偏好数据与对齐报告落盘）
    alignment_dir: str = "outputs/alignment"
    # DPO 的 β：loss 里的温度，也是梯度系数 −β·σ(−βm) 的一部分
    alignment_beta: float = 0.1
    # DPO 的学习率（参考模型上的标定值；7B 上要比它小若干数量级）
    alignment_learning_rate: float = 0.5
    # DPO 训练轮数（每轮过一次训练偏好对；本课标定 6 轮 = 42 步）
    alignment_epochs: int = 6
    # 留出偏好对占比（按对齐维度分层，每维至少留 1 条）
    alignment_valid_ratio: float = 0.25
    # KL 预算：最终 KL 超过它就会触发过优化告警
    alignment_kl_budget: float = 0.5
    # 样本量目标：希望偏好准确率能分辨出它与 50% 的差异（查表算样本量）
    alignment_target_accuracy: float = 0.6
    # 领域数据准备与增强（M5-D8）：六阶段流水线的阈值、权重与配比口径。
    # **注意下面这几个默认值都不是从别处抄来的**：它们由本课程数据集
    # （清洗后 37 条）的实测分布标定——输出长度 67~165 字、3-gram 重复率
    # 中位数 0.0 / 最大 0.377、有许可证元信息 16/37。换成别的语料时
    # `domain_quality_threshold` 与 `domain_near_dup_threshold` 必须重标，
    # 否则会出现"整批被质量门槛拦下"或"近重复一条都没检出"这类静默失效。
    # 领域数据产物目录（train.jsonl / eval.jsonl / manifest.json）
    domain_data_dir: str = "outputs/domain_data"
    # 质量门槛：五维加权总分低于它即被拒（真实脏样本会掉到 0.45 附近）
    domain_quality_threshold: float = 0.6
    # 近重复阈值：签名 Jaccard **估计**值达到它即判重复。0.7 是本课在
    # 本课程语料上标定的值——取 0.8（很多资料给的经验值，对应长文档）
    # 会让近重复链路在短中文提问上**完全空转**，实测一条都检不出来
    domain_near_dup_threshold: float = 0.7
    # shingle 窗口（字符 n-gram）：2 假重复多、5 短句漏判，3 是中文短句的经验值
    domain_shingle_k: int = 3
    # MinHash 置换个数：64 个在 J=0.5 处的标准差约 0.062
    domain_num_perm: int = 64
    # 配比分组口径：source（来源）/ origin（原始 vs 增强）/ safety（是否安全样本）
    domain_group_by: str = "source"
    # 配比上限：任一组不超过总体的这个比例。0.5 是"安全网"档，
    # 本课程数据集在该档下不丢样本；收紧到 0.4 会因连锁削减丢掉 12/37 条
    domain_max_group_ratio: float = 0.5
    # 是否启用增强阶段（增强只在离线可复现的算子范围内做，不调模型）
    domain_augment_enabled: bool = True
    # 每条原样本最多产出几个增强样本（取 1：增强样本天然比原样本同质）
    domain_max_augment_per_example: int = 1
    # 模型版本管理与持续微调（M5-D9）：版本链、触发器与采纳策略。
    # 下面五个阈值里，**只有 `retrain_min_gain` 与 `retrain_cooldown_hours`
    # 是从 day027 的评估方差与 day030 的步数成本推出来的**，其余三个是
    # 安全网档（宁可不训、不可乱训）。换语料/换基座时必须重标。
    # 版本注册表目录（索引文件 versions.jsonl 写在它下面）
    registry_dir: str = "outputs/registry"
    # 候选版本保留上限：超出后最旧的候选被归档（**stable 永不归档**，
    # 因为它是回滚目标；见 registry/store.py 的 prune）
    registry_keep_versions: int = 20
    # 数据增量门槛：新增可用样本达到它才触发重训。8 条是"本课程 37 条种子
    # 数据的 1/5 左右"——低于它会频繁产出与上一版几乎相同的适配器
    retrain_min_new_examples: int = 8
    # 线上合格率下限：低于它触发重训（与 day053 的评估门禁同一量级）
    retrain_min_pass_rate: float = 0.5
    # 冷却期（小时）：距上次训练不足它则否决重训（从未训练时不否决）
    retrain_cooldown_hours: float = 24.0
    # 并发训练上限：达到它则否决重训（两个任务写同一份产物目录会互相覆盖）
    retrain_max_parallel_runs: int = 1
    # 采纳门槛（增益死区）：与 day027 趋势判定的 0.05 同一种防抖思想，
    # 但这里取 0.02——它是"单条评估用例翻转"在 18 条评估集上的粒度
    retrain_min_gain: float = 0.02
    # 退步容忍度：0.0 = 任何退步都拒绝采纳
    retrain_regression_tolerance: float = 0.0
    # 不可比时的绝对合格率门槛（换了数据集或基座时走这一条）
    retrain_absolute_min_pass_rate: float = 0.5
    # 候选时效上限（小时）：超龄候选不再采纳，7 天 = 一周的迭代节奏
    retrain_max_candidate_age_hours: float = 168.0
    # 回滚后的观察窗口（小时）：切换是瞬时的，事故是否停止需要窗口数据确认
    retrain_observe_window_hours: float = 24.0
    # MLOps 微调流水线（M5-D10）：实验追踪目录、发布门禁与 CI 参数。
    # 六个门禁阈值里 `mlops_min_pass_rate` 与 `mlops_max_regression` 沿用
    # day053 的评估门禁口径（0.5 / 0.0），其余是"安全网档"：
    # 64 MiB 是 LoRA 适配器在本课程参考模型上的 **1000 倍余量**（实测几十 KiB），
    # 它拦的是"挂错了目标模块导致适配器暴涨"，不是日常调优。
    # 实验追踪与产物目录（runs.jsonl / MODEL_CARD.md / release_manifest.json）
    mlops_dir: str = "outputs/mlops"
    # 发布门禁：合格率下限（与 day053 的 finetune_eval_min_pass_rate 同口径）
    mlops_min_pass_rate: float = 0.5
    # 发布门禁：相对基线的合格率允许变化（0.0 = 一处都不许掉）
    mlops_max_regression: float = 0.0
    # 发布门禁：适配器体积上限（MiB）
    mlops_max_adapter_mebibytes: float = 64.0
    # 发布门禁：单位推理成本上限（美元/千 token）
    mlops_max_cost_per_1k_tokens: float = 0.05
    # 生成的 GitHub Actions workflow：Python 版本 / cron（**按 UTC 解释**）/ 超时
    mlops_ci_python_version: str = "3.11"
    mlops_ci_schedule: str = "0 20 * * *"
    mlops_ci_timeout_minutes: int = 30
    # 专属模型部署与切换（M5-D11）：部署档案、流量策略、上线验证与成本对比。
    # 这一组分成四小类，**它们的"可调性"完全不同**，混在一起会被误调：
    #   部署档案（dir / name / kind）：改一次要重新部署，属于变更管理；
    #   流量策略（dedicated_ratio / shadow_ratio / fail_open）：**每天可调**，
    #     是灰度发布的操作面（0 → 0.05 → 0.5 → 1.0）；
    #   上线验证（min_pass_rate / max_regression / max_latency_ratio）：
    #     切流量之前问一次"够不够好"，前两个沿用 day053/059 的口径；
    #   成本参数（gpu_key / tokens_per_second / ...）：**唯一必须实测的一组**。
    # 部署记录落盘目录（deployment_binding.json 写在它下面）
    serving_dir: str = "outputs/serving"
    # 部署单元名：必须与服务端 /v1/models 返回的 id 一致（灰度时可带 -canary 后缀）
    serving_name: str = "smart-research-qwen3-8b"
    # 部署形态：base（未微调基座）/ adapter（基座 + 适配器）/ merged（合并模型）
    serving_kind: str = "adapter"
    # 专属模型直接服务的流量比例（0 = 全云端；1 = 已切完，切换器可比摘掉）
    serving_dedicated_ratio: float = 0.0
    # 影子流量比例：额外调用专属模型但只返回云端结果（灰度之前的一步）
    serving_shadow_ratio: float = 0.0
    # 专属模型失败时是否回落到云端（False = 严格模式，用于测真实成功率）
    serving_fail_open: bool = True
    # 上线验证：专属模型合格率下限（与 day053 的 finetune_eval_min_pass_rate 同口径）
    serving_min_pass_rate: float = 0.5
    # 上线验证：相对云端允许的合格率变化（0.0 = 一处都不许掉）
    serving_max_regression: float = 0.0
    # 上线验证：专属/云端延迟中位数倍数上限。**缺省 None = 不检查**——
    # "多慢算慢"取决于部署形态（本地 CPU 与 A10G 差一个数量级），
    # 给一个全局缺省只会让它被无脑放宽；要查就显式填一个数。
    serving_max_latency_ratio: float | None = None
    # 成本对比：GPU 机型键（取值来自 serving/cost.py 的 GPU_HOURLY_USD 价格表）
    serving_gpu_key: str = "aws-g5-xlarge-a10g"
    # 成本对比：实测吞吐（token/s）。**这个数没有权威缺省值**——它取决于模型、
    # 量化、批大小与序列长度，必须在本机的部署上量一遍再填。
    serving_gpu_tokens_per_second: float = 250.0
    # 成本对比：GPU 实际能用上的算力比例（批调度与长尾序列都会吃掉它）
    serving_gpu_utilization: float = 1.0
    # 成本对比：云端计价方案键（gpt-4o-mini / gpt-4o）
    serving_cloud_pricing_key: str = "gpt-4o-mini"
    # 成本对比：每次请求的平均输入/输出 token（RAG 场景输入常是输出的数倍）
    serving_input_tokens_per_request: int = 1500
    serving_output_tokens_per_request: int = 500
    # 成本对比：月度请求量（盈亏平衡点是按它来对比的）
    serving_monthly_requests: int = 100000
    # 文档解析与加载（M6-D1）：大小护栏、编码兜底与目录约定。
    # 这一组分成三小类，**它们出问题时表现完全不同**：
    #   大小上限（max_file_mib）：超出即报错，看得见；
    #   编码兜底（text_encoding_hint）：只影响"怎么猜"，猜错时表现为乱码；
    #   目录约定（dir）：只影响演示脚本往哪读，不影响任何判定。
    # 单份文件的解析上限（MiB）。**在入口拦而不是在解析器里拦**：一份 2 GiB 的
    # PDF 会把内存吃光，而错误会以 MemoryError 出现在栈的深处，无法写进报告。
    documents_max_file_mib: float = 32.0
    # 演示与批量入库默认扫描的目录（相对进程工作目录）
    documents_dir: str = "data/documents"
    # 编码兜底提示：utf-8 与 gb18030 都失败时**不**使用它，只用它做日志说明。
    # 留在这里是为了让"本项目的文档以什么编码为主"这件事有一个可写下来的位置。
    documents_text_encoding_hint: str = "utf-8"
    # 批量入库时每个目录最多处理多少个文件（0 表示不限制）。
    # 默认 0 是刻意的：截断会让"入库结果"取决于目录规模，而那种依赖
    # 一旦存在，报告里的"文档数"就不再是一份可以被比较的数字。
    documents_ingest_limit: int = 0
    # 分块（M6-D2）：默认策略、预算、重叠与语义阈值。
    # 这一组与 `documents_*` 有一个关键差别：**它们会直接改变产物的字节**
    # （块的边界、数量、chunk_id）。因此默认值一旦定下就不能随手改——
    # 改 `chunking_max_tokens` 等于让全量索引失效，与 day061 改
    # `normalize_text` 是同一量级的变更。见 docs/chunking_strategies.md。
    # 默认策略：fixed / recursive / structural / semantic。
    # 缺省是 recursive——它在"不认识结构的纯文本"与"能利用结构的文档"
    # 之间取一个中间位置，且**不需要 embedding**（离线可用）。
    chunking_strategy: str = "recursive"
    # 单块预算。缺省 320 按"一次检索命中 2~3 块进提示词"的常见用法标定：
    # 3 × 320 = 960 单位，加上系统提示与历史后仍在 4k 上下文内。
    chunking_max_tokens: int = 320
    # 块间重叠。缺省 48（约 15%）：够把一个被切断的句子补回来，
    # 又不至于让 embedding 成本明显上升（放大量 = 48/272 ≈ 17.6%）。
    chunking_overlap_tokens: int = 48
    # 过短的块与后一块合并的下限。缺省 40：低于它的块单独进检索
    # 只会贡献噪声（"阈值设为 0.85"这种片段离开上下文毫无意义）。
    chunking_min_tokens: int = 40
    # 预算的度量单位：chars（默认，确定性）/ tiktoken（真实 token，依赖词表缓存）
    chunking_measurer: str = "chars"
    # 语义策略的相似度阈值分位数（只有 semantic 消费它）。
    # 0.25 表示"在相似度最低的 25% 的缝上切"——**相对判据**，
    # 换 embedding 提供方不需要重新标定（见 chunking/semantic.py）。
    chunking_similarity_percentile: float = 0.25
    # 分块评估的检索深度：只看 top-3，因为 top-3 之外的内容大概率会被
    # 提示词长度或模型注意力稀释掉——**用"真实会被用上的块"来打分**。
    chunking_eval_top_k: int = 3
    # 向量库（M6-D3）：后端选型、度量、默认检索深度与持久化位置。
    # 这一组与 `chunking_*` 有一个关键差别：**它们不改变已有产物的字节**
    # （chunk_id 完全不受影响），因此换后端、换存储位置都不需要重建知识库。
    # 唯一的例外是 metric：**换度量会让"谁离得最近"整体改变**，
    # 于是所有按分数标定的阈值（min_score、语义缓存阈值）都必须重新标定。
    # 后端选型（M6-D3）：flat / faiss / chroma。
    # 缺省是 flat——纯 Python、零可选依赖、结果逐位可复现；
    # faiss 与 chromadb 都需要额外安装（`pip install faiss-cpu` /
    # `pip install chromadb`），缺失时 registry 会给出安装指引而不是 ImportError。
    vector_backend: str = "flat"
    # 距离度量：cosine（默认）/ ip / l2。见 vectorstore/metrics.py 的取舍。
    # 默认 cosine 的理由是**阈值可解释**：它的值域有绝对上界 1，
    # 于是 min_score=0.35 是一句跨数据集都能说的话。
    vector_metric: str = "cosine"
    # 默认检索深度。缺省 5（与 day041 起的检索默认值保持一致）。
    vector_default_top_k: int = 5
    # 相似度下限（统一口径：越大越近）。None 表示不设阈值。
    # 不设默认 0.0 是因为它在 l2 度量下是一句完全相反的话
    # （l2 的分数是 ≤ 0 的负数），而两个度量共用一个参数名。
    vector_min_score: float | None = None
    # Chroma 的集合名。注意它有命名约束：3~512 字符、首尾必须是
    # 小写字母或数字、中间只允许 . - _、不能出现连续两个点。
    vector_collection: str = "smart_research_agent"
    # 持久化位置。空串表示**不落盘**（纯内存）——这是刻意的默认值：
    # 一个"以为落盘了其实还在内存里"的向量库要到进程重启才暴露，
    # 所以默认行为必须是"明确不落盘"，由使用方显式打开。
    vector_persist_path: str = ""
    # Embedding 索引构建（M6-D4）：批量编码、向量缓存与版本/备份的约定。
    # 这一组与 `vector_*` 的差别：**它们不改变"存什么"，只改变"怎么算出要存的东西"**。
    # 因此换 batch_size / cache_path 都不需要重建索引（结果逐位相同），
    # 而换索引目录只影响清单与备份放在哪。
    # 构建模式：incremental（默认，只重算变了的块）/ full（整库重建）。
    # 缺省是 incremental——它在本课的所有场景下都不比 full 差，
    # 唯一会"更差"的情况是变更比例极高（逐条 upsert 的开销超过整库重建），
    # 那由下面的 indexing_full_rebuild_threshold 兜住。
    indexing_mode: str = "incremental"
    # 一次送给提供方的文本条数。缺省 32 是"批处理收益"与"单次请求体积"的折中：
    # 太小退化成逐条（day064 的基线），太大则一次失败要重试一整批。
    indexing_batch_size: int = 32
    # 向量缓存文件。空串表示只用内存缓存（本次进程内有效）。
    # 缓存**可以随时删**：它不承载唯一信息，只是省下重算的时间。
    indexing_cache_path: str = ""
    # 清单、备份与缓存默认所在目录（演示脚本与文档里引用它）。
    indexing_dir: str = "data/index"
    # 备份保留份数。超过就删最旧的——**备份不是日志，不设上限会吃满磁盘**。
    indexing_backups_keep: int = 3
    # 变更比例超过它时改走**全量重建**（而不是继续增量）。
    # 理由：增量不是永远更省——当九成块都变了，逐条 upsert 的开销
    # 反而高于"清空重写"。缺省 0.5 表示"一半以上变了就重建"。
    indexing_full_rebuild_threshold: float = 0.5


# 全局单例，首次导入时即完成解析
settings = Settings()
