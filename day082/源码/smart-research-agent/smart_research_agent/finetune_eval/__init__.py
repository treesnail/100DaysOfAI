"""微调模型评估（M5-D5）：把"适配器有没有变好"变成一组可复现的数字.

day052 交出了三件东西——适配器（三个文件 + 一份含哈希的清单）、
多卡的账（显存 / 通信 / 学习率缩放）、以及一道合并门禁。今天要回答的是
**下一个问题**：这份适配器到底值不值得上线？而这个问题在微调里有一个
特殊之处——适配器只改了 2.787% 的参数，输出往往"看起来差不多"，
所以"我觉得好了点"是最没有价值的判断。

七层结构，每层只干一件事：

==============================  ====================================================
``metrics``                     六个文本分量 + 权重校验（纯函数，无依赖）
``suites``                      用例数据模型、18 条种子、切分、泄漏体检、难度体检
``model_probe``                 模型侧白盒信号：逐 token 对数概率 / 困惑度 / 命中率
``evaluator``                   执行器、``passed`` 的定义、Wilcoxon 区间、脚本化两臂
``compare``                     配对比较：McNemar 精确检验 + 配对自助法 + 分桶门禁
``report``                      报告绑定适配器清单与评估集指纹，三项门禁
``pipeline``                    把上面六层串成一次调用
==============================  ====================================================

贯穿全包的三个约定，与前四天严格对齐：

1. **分母交出来**。``fact_recall`` 返回 ``(hit, total, missing)``、
   探针返回 ``(logprob, supervised)``、区间返回 ``(low, high)``——
   与 day050 的 ``masked_cross_entropy`` 交出监督 token 数是同一条纪律。
2. **产物要自描述**。报告里同时有适配器内容哈希、评估集指纹、权重快照、
   逐条分量与失败原因。**一个没有绑定信息的分数不是证据**。
3. **门禁必须有"失败时会怎样"的用例**。``no_regression``、``execution``、
   ``min_pass_rate`` 三条门禁在测试里都被构造过失败路径——day052 那句
   "一个永远不会失败的门禁不是门禁"在这里同样适用。

边界也要说清楚：``scripted_arm`` 产生的两臂是**确定性函数**，不是模型
输出。它们存在的意义是让流程可在离线环境完整跑通并被测试覆盖；
接上真实模型时，只要把 ``runner`` 换成"加载适配器 → 生成答案"的调用，
其余六层一行都不用改。
"""

from __future__ import annotations

from smart_research_agent.finetune_eval.compare import (
    DEFAULT_ALPHA,
    DEFAULT_BOOTSTRAP_SAMPLES,
    RATE_TOLERANCE,
    BootstrapResult,
    BucketDelta,
    ComparisonReport,
    McNemarResult,
    binomial_two_sided_p_value,
    compare_runs,
    mcnemar_exact,
    paired_bootstrap_delta,
    percentile,
)
from smart_research_agent.finetune_eval.evaluator import (
    BASELINE_KEEP_RATIO,
    BASELINE_SUFFIX,
    DEFAULT_Z,
    DIFFICULTY_ORDER,
    FINETUNED_HARD_KEEP_RATIO,
    SCRIPTED_ARMS,
    EvalRun,
    FineTuneEvalHarness,
    ItemOutcome,
    item_to_case,
    run_suite,
    score_item,
    scripted_arm,
    weights_snapshot,
    wilson_interval,
)
from smart_research_agent.finetune_eval.metrics import (
    COMPONENT_NAMES,
    DEFAULT_WEIGHTS,
    REFUSAL_MARKERS,
    WEIGHT_TOLERANCE,
    FactRecall,
    FinetuneEvalError,
    FormatRules,
    MetricBreakdown,
    char_ngrams,
    chrf,
    compute_components,
    fact_recall,
    forbidden_hits,
    forbidden_ratio,
    format_violations,
    is_refusal,
    lcs_length,
    mean,
    normalize_text,
    rouge_l,
    token_f1,
    tokenize,
    validate_weights,
    weighted_total,
)
from smart_research_agent.finetune_eval.model_probe import (
    MIN_PROBE_TOKENS,
    PROBE_FIELDS,
    ContextModel,
    ProbeRecord,
    ProbeResult,
    compare_probes,
    encode_probe_batches,
    probe_items,
    sequence_logprob,
    token_accuracy,
)
from smart_research_agent.finetune_eval.pipeline import (
    BASELINE_ARM_NAME,
    FINETUNED_ARM_NAME,
    PROBE_AFTER_LABEL,
    PROBE_BEFORE_LABEL,
    FineTuneEvalOutcome,
    run_finetune_evaluation,
)
from smart_research_agent.finetune_eval.report import (
    GATE_NAMES,
    MAX_FAILURES,
    REPORT_FILE,
    EvalReport,
    build_report,
    estimate_eval_seconds,
    read_report,
    render_markdown,
    verify_manifest_binding,
    write_report,
)
from smart_research_agent.finetune_eval.suites import (
    BUCKET_DESCRIPTIONS,
    BUCKETS,
    DEFAULT_EVAL_RATIO,
    DIFFICULTIES,
    EASY_MAX_LOAD,
    LEAKAGE_NGRAM,
    LEAKAGE_THRESHOLD,
    NORMAL_MAX_LOAD,
    SEED_ITEMS,
    TIGHT_CHAR_BUDGET,
    EvalItem,
    audit_references,
    audit_suite,
    build_suite,
    detect_leakage,
    jaccard,
    read_suite,
    split_suite,
    suite_fingerprint,
    suite_stats,
    write_suite,
)

__all__ = [
    "BASELINE_ARM_NAME",
    "BASELINE_KEEP_RATIO",
    "BASELINE_SUFFIX",
    "BUCKETS",
    "BUCKET_DESCRIPTIONS",
    "COMPONENT_NAMES",
    "DEFAULT_ALPHA",
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "DEFAULT_EVAL_RATIO",
    "DEFAULT_WEIGHTS",
    "DEFAULT_Z",
    "DIFFICULTIES",
    "DIFFICULTY_ORDER",
    "EASY_MAX_LOAD",
    "FINETUNED_ARM_NAME",
    "FINETUNED_HARD_KEEP_RATIO",
    "GATE_NAMES",
    "LEAKAGE_NGRAM",
    "LEAKAGE_THRESHOLD",
    "MAX_FAILURES",
    "MIN_PROBE_TOKENS",
    "NORMAL_MAX_LOAD",
    "PROBE_AFTER_LABEL",
    "PROBE_BEFORE_LABEL",
    "PROBE_FIELDS",
    "RATE_TOLERANCE",
    "REFUSAL_MARKERS",
    "REPORT_FILE",
    "SCRIPTED_ARMS",
    "SEED_ITEMS",
    "TIGHT_CHAR_BUDGET",
    "WEIGHT_TOLERANCE",
    "BootstrapResult",
    "BucketDelta",
    "ComparisonReport",
    "ContextModel",
    "EvalItem",
    "EvalReport",
    "EvalRun",
    "FactRecall",
    "FineTuneEvalHarness",
    "FineTuneEvalOutcome",
    "FinetuneEvalError",
    "FormatRules",
    "ItemOutcome",
    "McNemarResult",
    "MetricBreakdown",
    "ProbeRecord",
    "ProbeResult",
    "audit_references",
    "audit_suite",
    "binomial_two_sided_p_value",
    "build_report",
    "build_suite",
    "char_ngrams",
    "chrf",
    "compare_probes",
    "compare_runs",
    "compute_components",
    "detect_leakage",
    "encode_probe_batches",
    "estimate_eval_seconds",
    "fact_recall",
    "forbidden_hits",
    "forbidden_ratio",
    "format_violations",
    "is_refusal",
    "item_to_case",
    "jaccard",
    "lcs_length",
    "mean",
    "mcnemar_exact",
    "normalize_text",
    "paired_bootstrap_delta",
    "percentile",
    "probe_items",
    "read_report",
    "read_suite",
    "render_markdown",
    "rouge_l",
    "run_finetune_evaluation",
    "run_suite",
    "score_item",
    "scripted_arm",
    "sequence_logprob",
    "split_suite",
    "suite_fingerprint",
    "suite_stats",
    "token_accuracy",
    "token_f1",
    "tokenize",
    "validate_weights",
    "verify_manifest_binding",
    "weighted_total",
    "weights_snapshot",
    "wilson_interval",
    "write_report",
    "write_suite",
]
