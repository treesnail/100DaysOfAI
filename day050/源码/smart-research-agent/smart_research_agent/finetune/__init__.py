"""微调数据工程包（M5-D1）：格式规范 → 方法选型 → 采集 → 清洗 → 统计 → 切分落盘.

一条完整的离线流水线::

    default_collector()  ->  DataCollector.collect()  ->  DatasetBundle
        采集（seed / eval 轨迹 / 红队）        清洗+去重+归因
    compute_stats(bundle.examples)  ->  DatasetStats
    dump_bundle(bundle, "data/finetune/out")  ->  train.jsonl / eval.jsonl

方法选型（``recommend_method`` / ``METHODS`` / ``render_methods_table``）与
数据流水线同属 M5-D1："选什么方法"和"拿什么数据喂它"必须一起回答。
"""

from __future__ import annotations

from smart_research_agent.finetune.cleaner import (
    PLACEHOLDER_MARKERS,
    DatasetCleaner,
    FilterDecision,
    FilterReport,
    QualityRule,
    clean_text,
    dedupe_key,
    default_rules,
    normalize_whitespace,
    strip_control_chars,
)
from smart_research_agent.finetune.collector import (
    EVAL_TASK_SOURCE,
    REDTEAM_SOURCE,
    SAFETY_REFUSAL_TEMPLATE,
    DataCollector,
    DatasetBundle,
    DataSource,
    EvalTaskSource,
    JSONLSource,
    RedTeamSource,
    SourceSpec,
    default_collector,
    iter_jsonl_records,
)
from smart_research_agent.finetune.dataset import (
    DatasetStats,
    build_dataset,
    compute_stats,
    dump_bundle,
    split_dataset,
)
from smart_research_agent.finetune.overview import (
    DPO,
    LORA,
    METHODS,
    QLORA,
    RLHF_PPO,
    SFT,
    MethodProfile,
    MethodRecommendation,
    lora_trainable_ratio,
    recommend_method,
    render_methods_table,
)
from smart_research_agent.finetune.schema import (
    ALPACA,
    CHAT,
    DEFAULT_FORMAT,
    PROMPT_COMPLETION,
    SUPPORTED_FORMATS,
    DatasetFormatError,
    TrainingExample,
    dump_jsonl,
    load_jsonl,
    parse_example,
)

__all__ = [
    "ALPACA",
    "CHAT",
    "DEFAULT_FORMAT",
    "DPO",
    "EVAL_TASK_SOURCE",
    "LORA",
    "METHODS",
    "PLACEHOLDER_MARKERS",
    "PROMPT_COMPLETION",
    "QLORA",
    "REDTEAM_SOURCE",
    "RLHF_PPO",
    "SAFETY_REFUSAL_TEMPLATE",
    "SFT",
    "SUPPORTED_FORMATS",
    "DataCollector",
    "DataSource",
    "DatasetBundle",
    "DatasetCleaner",
    "DatasetFormatError",
    "DatasetStats",
    "EvalTaskSource",
    "FilterDecision",
    "FilterReport",
    "JSONLSource",
    "MethodProfile",
    "MethodRecommendation",
    "QualityRule",
    "RedTeamSource",
    "SourceSpec",
    "TrainingExample",
    "build_dataset",
    "clean_text",
    "compute_stats",
    "dedupe_key",
    "default_collector",
    "default_rules",
    "dump_bundle",
    "dump_jsonl",
    "iter_jsonl_records",
    "load_jsonl",
    "lora_trainable_ratio",
    "normalize_whitespace",
    "parse_example",
    "recommend_method",
    "render_methods_table",
    "split_dataset",
    "strip_control_chars",
]
