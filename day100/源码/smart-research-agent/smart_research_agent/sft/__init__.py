"""SFT 监督微调包（M5-D2）：渲染 → 编码 → mask → 批次 → 训练 → 落盘.

从 day048 的 ``train.jsonl`` 到一次真实训练，中间只隔一条链::

    render_supervised()                       模板渲染，产出 prompt/answer 字符偏移
        -> encode_supervised()                分别编码再拼接，prompt 段 label 置 -100
        -> collate() / iter_batches()          padding 三件套（ids/attn/labels）
        -> plan_training()                     步数、warmup、有效批大小的纯算术核定
        -> SFTTrainer.fit()                    梯度累积 + 学习率调度 + 评估 + 落盘
        -> save_checkpoint()                   超参/词表/权重/指标/逐步日志 五个文件

三个模块的边界是刻意的：

- ``template`` / ``encoding`` / ``loss`` 是**纯函数**，没有任何状态，可以
  单独测试每一个数字（"第 5 个 token 的 label 是 -100"这种断言必须成立）；
- ``args`` 是**纯数据 + 纯算术**，因此"这批数据会跑多少步"可以在训练
  之前算准（day049 已经算过一遍）；
- 只有 ``trainer`` / ``checkpoint`` 触碰外部世界（随机数、时间、磁盘）。

真实框架的落地路径由 ``hf_script`` 生成：``render_hf_sft_script()`` 产出一份
``transformers`` + ``Trainer`` 脚本，``render_trl_sft_script()`` 产出 TRL
``SFTTrainer`` 版本。两者与本包的 ``SFTTrainingArgs`` **同源**——参数只
有一个来源，因此不会出现"教程与脚本不一致"。
"""

from __future__ import annotations

from smart_research_agent.sft.args import (
    IMPLEMENTED_SCHEDULERS,
    LR_SOFT_RANGE,
    LR_SOFT_RANGE_PEFT,
    LR_SOFT_RANGE_REFERENCE,
    SCHEDULER_TYPES,
    SFTConfigError,
    SFTTrainingArgs,
    TrainingPlan,
    lr_curve,
    plan_training,
)
from smart_research_agent.sft.checkpoint import (
    CHECKPOINT_FILES,
    CheckpointError,
    checkpoint_summary,
    load_checkpoint,
    save_checkpoint,
)
from smart_research_agent.sft.encoding import (
    IGNORE_INDEX,
    PAD_TOKEN,
    PAD_TOKEN_ID,
    SPECIAL_TOKENS,
    TRUNCATION_HEAD,
    TRUNCATION_KEEP_ANSWER,
    UNK_TOKEN,
    Batch,
    CharTokenizer,
    EncodedSample,
    LengthSummary,
    SFTDataError,
    collate,
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
    generated_script_summary,
    render_hf_sft_script,
    render_trl_sft_script,
)
from smart_research_agent.sft.loss import (
    SFTLossError,
    cross_entropy,
    log_softmax,
    masked_cross_entropy,
    masked_token_accuracy,
    perplexity,
    softmax,
)
from smart_research_agent.sft.reference_model import (
    REFERENCE_LEARNING_RATE,
    ModelState,
    ReferenceSFTModel,
    top_k_next,
)
from smart_research_agent.sft.template import (
    CHATML,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPLATE,
    LLAMA3,
    PLAIN,
    SUPPORTED_TEMPLATES,
    RenderedSample,
    SFTTemplateError,
    load_training_examples,
    render_supervised,
    render_supervised_list,
)
from smart_research_agent.sft.trainer import (
    EncodingReport,
    EvalRecord,
    SFTReport,
    SFTTrainer,
    StepRecord,
    train_reference,
)

__all__ = [
    "CHATML",
    "CHECKPOINT_FILES",
    "DEFAULT_BASE_MODEL",
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_TEMPLATE",
    "HF_DEPENDENCIES",
    "IGNORE_INDEX",
    "IMPLEMENTED_SCHEDULERS",
    "LLAMA3",
    "LR_SOFT_RANGE",
    "LR_SOFT_RANGE_PEFT",
    "LR_SOFT_RANGE_REFERENCE",
    "PAD_TOKEN",
    "PAD_TOKEN_ID",
    "PLAIN",
    "REFERENCE_LEARNING_RATE",
    "SCHEDULER_TYPES",
    "SPECIAL_TOKENS",
    "SUPPORTED_TEMPLATES",
    "TRL_DEPENDENCIES",
    "TRUNCATION_HEAD",
    "TRUNCATION_KEEP_ANSWER",
    "UNK_TOKEN",
    "Batch",
    "CharTokenizer",
    "CheckpointError",
    "EncodedSample",
    "EncodingReport",
    "EvalRecord",
    "LengthSummary",
    "ModelState",
    "ReferenceSFTModel",
    "RenderedSample",
    "SFTConfigError",
    "SFTDataError",
    "SFTLossError",
    "SFTReport",
    "SFTTemplateError",
    "SFTTrainer",
    "SFTTrainingArgs",
    "StepRecord",
    "TrainingPlan",
    "checkpoint_summary",
    "collate",
    "cross_entropy",
    "dependency_commands",
    "encode_supervised",
    "generated_script_summary",
    "iter_batches",
    "length_summary",
    "load_checkpoint",
    "load_training_examples",
    "log_softmax",
    "lr_curve",
    "masked_cross_entropy",
    "masked_token_accuracy",
    "perplexity",
    "plan_training",
    "render_hf_sft_script",
    "render_supervised",
    "render_supervised_list",
    "render_trl_sft_script",
    "save_checkpoint",
    "softmax",
    "suggest_max_length",
    "top_k_next",
    "train_reference",
]
