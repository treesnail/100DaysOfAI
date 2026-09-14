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


# 全局单例，首次导入时即完成解析
settings = Settings()
