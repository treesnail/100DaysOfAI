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


# 全局单例，首次导入时即完成解析
settings = Settings()
