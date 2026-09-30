"""模型版本管理与持续微调包（M5-D9）：让"换不换模型"变成一份可核对的判断.

M5 前八天把"训得出来"做到了极致——数据可复现（day048/057）、训练可复现
（day050/051/052）、效果可度量（day053）、偏好可对齐（day054/055）。但它们
合起来仍然回答不了一个运维每天都要问的问题：

> **"线上这个效果是哪一份数据、哪一次训练、哪一版基座造出来的？
>   现在有一份新产物，我要不要换？换了出问题怎么退？"**

本包把这三个问题拆成五件事，每件事一个模块：

```text
version.py   三元组与内容寻址的版本键      —— "这是不是同一份产物"
record.py    版本记录与四态流转            —— "它现在是什么状态、能不能部署"
store.py     追加写事件日志 + 折叠         —— "历史是什么，生产版本是谁"
triggers.py  触发器与否决项                —— "现在该不该训"
rollback.py  采纳判定与回滚计划            —— "换不换、退不退、怎么退"
retrain.py   六步流水线                    —— 把上面五件事串成一次可复盘的动作
```

## 三条贯穿全包的纪律

1. **三元组缺一不可**：``base_model`` + ``adapter_sha256`` + ``dataset_fingerprint``。
   少任何一项，"可追溯"就只是口号（见 ``version.py`` 的对照表）；
2. **不可比就不算差值**：换了数据集或基座的两个版本，它们的分数差不是
   "提升量"。``comparable`` 与 ``evaluate_candidate`` 的分支把这条写成了代码，
   而不是留给评审去记得；
3. **状态变更是事件，不是覆盖写**：``versions.jsonl`` 是追加写的日志，
   "什么时候因为什么变了"与"现在是什么"一样重要（见 ``store.py``）。

## 与既有包的接缝

- **上游**：``domain_data.DatasetFingerprint``（数据集指纹）、
  ``peft.trainer.adapter_content_hash``（适配器哈希）、
  ``sft.checkpoint`` 与 ``finetune_eval``（指标来源）。它们一行都不用改；
- **下游**：``api.routes`` 的 ``/registry/*`` 端点只做**只读或纯计算**，
  写入动作（``register`` / ``set_stage``）只发生在流水线与脚本里——
  HTTP 端点在多副本部署下无法保证"谁先写"，而版本表的写入必须是单点的。
"""

from __future__ import annotations

from smart_research_agent.registry.errors import RegistryError
from smart_research_agent.registry.record import (
    ALLOWED_TRANSITIONS,
    ARTIFACT_ADAPTER,
    ARTIFACT_DATASET,
    ARTIFACT_MERGED,
    ARTIFACT_NAMES,
    CORE_METRICS,
    DEPLOY_REQUIRED_ARTIFACTS,
    METRIC_LATENCY_MS,
    METRIC_PASS_RATE,
    METRIC_TRAIN_LOSS,
    STAGES,
    STAGE_ARCHIVED,
    STAGE_CANDIDATE,
    STAGE_ROLLED_BACK,
    STAGE_STABLE,
    TERMINAL_STAGES,
    ModelVersion,
    utc_now_iso,
)
from smart_research_agent.registry.retrain import (
    PLAN_STEPS,
    STEP_MEANINGS,
    TRAIN_RESULT_ARTIFACTS,
    TRAIN_RESULT_REQUIRED,
    ContinualFinetunePipeline,
    Evaluator,
    RetrainOutcome,
    RetrainPlan,
    Trainer,
    build_state,
    pipeline_dataset_fingerprint,
    verify_data,
)
from smart_research_agent.registry.rollback import (
    ACTIONS,
    ACTION_HOLD,
    ACTION_PROMOTE,
    ACTION_ROLLBACK,
    CHECK_ABSOLUTE,
    CHECK_AGE,
    CHECK_COMPARABLE,
    CHECK_DEPLOYABLE,
    CHECK_GAIN,
    CHECK_METRICS,
    DEFAULT_OBSERVE_WINDOW_HOURS,
    STEP_FREEZE,
    STEP_OBSERVE,
    STEP_RECORD,
    STEP_VERIFY,
    CheckResult,
    PromotionDecision,
    PromotionPolicy,
    RollbackPlan,
    RollbackStep,
    apply_rollback,
    candidate_age_hours,
    evaluate_candidate,
    plan_rollback,
)
from smart_research_agent.registry.store import (
    DEFAULT_KEEP_VERSIONS,
    EVENT_REGISTER,
    EVENT_STAGE,
    INDEX_FILENAME,
    REGISTER_REQUIRED_FIELDS,
    ModelRegistry,
)
from smart_research_agent.registry.triggers import (
    KIND_TRIGGER,
    KIND_VETO,
    TRIGGER_DATASET_CHANGE,
    TRIGGER_DATA_GROWTH,
    TRIGGER_QUALITY_DROP,
    VETO_ACTIVE_RUN,
    VETO_COOLDOWN,
    RetrainDecision,
    TriggerEvaluation,
    TriggerPolicy,
    TriggerState,
    evaluate_triggers,
    trigger_table,
)
from smart_research_agent.registry.version import (
    BUMP_KINDS,
    BUMP_MAJOR,
    BUMP_MINOR,
    BUMP_PATCH,
    INITIAL_VERSION,
    MIN_ADAPTER_HASH_LENGTH,
    MIN_DATASET_FINGERPRINT_LENGTH,
    SEMVER_PATTERN,
    VERSION_KEY_LENGTH,
    VersionConflict,
    VersionTriple,
    bump_kind_for,
    bump_version,
    canonical_version_payload,
    comparable,
    format_semver,
    next_version,
    normalize_digest,
    parse_semver,
    semver_sort_key,
    version_key,
)

__all__ = [
    "ACTIONS",
    "ACTION_HOLD",
    "ACTION_PROMOTE",
    "ACTION_ROLLBACK",
    "ALLOWED_TRANSITIONS",
    "ARTIFACT_ADAPTER",
    "ARTIFACT_DATASET",
    "ARTIFACT_MERGED",
    "ARTIFACT_NAMES",
    "BUMP_KINDS",
    "BUMP_MAJOR",
    "BUMP_MINOR",
    "BUMP_PATCH",
    "CHECK_ABSOLUTE",
    "CHECK_AGE",
    "CHECK_COMPARABLE",
    "CHECK_DEPLOYABLE",
    "CHECK_GAIN",
    "CHECK_METRICS",
    "CORE_METRICS",
    "DEFAULT_KEEP_VERSIONS",
    "DEFAULT_OBSERVE_WINDOW_HOURS",
    "DEPLOY_REQUIRED_ARTIFACTS",
    "EVENT_REGISTER",
    "EVENT_STAGE",
    "INDEX_FILENAME",
    "INITIAL_VERSION",
    "KIND_TRIGGER",
    "KIND_VETO",
    "METRIC_LATENCY_MS",
    "METRIC_PASS_RATE",
    "METRIC_TRAIN_LOSS",
    "MIN_ADAPTER_HASH_LENGTH",
    "MIN_DATASET_FINGERPRINT_LENGTH",
    "PLAN_STEPS",
    "REGISTER_REQUIRED_FIELDS",
    "SEMVER_PATTERN",
    "STAGES",
    "STAGE_ARCHIVED",
    "STAGE_CANDIDATE",
    "STAGE_ROLLED_BACK",
    "STAGE_STABLE",
    "STEP_FREEZE",
    "STEP_MEANINGS",
    "STEP_OBSERVE",
    "STEP_RECORD",
    "STEP_VERIFY",
    "TERMINAL_STAGES",
    "TRAIN_RESULT_ARTIFACTS",
    "TRAIN_RESULT_REQUIRED",
    "TRIGGER_DATASET_CHANGE",
    "TRIGGER_DATA_GROWTH",
    "TRIGGER_QUALITY_DROP",
    "VERSION_KEY_LENGTH",
    "VETO_ACTIVE_RUN",
    "VETO_COOLDOWN",
    "CheckResult",
    "ContinualFinetunePipeline",
    "Evaluator",
    "ModelRegistry",
    "ModelVersion",
    "PromotionDecision",
    "PromotionPolicy",
    "RegistryError",
    "RetrainDecision",
    "RetrainOutcome",
    "RetrainPlan",
    "RollbackPlan",
    "RollbackStep",
    "Trainer",
    "TriggerEvaluation",
    "TriggerPolicy",
    "TriggerState",
    "VersionConflict",
    "VersionTriple",
    "apply_rollback",
    "build_state",
    "bump_kind_for",
    "bump_version",
    "candidate_age_hours",
    "canonical_version_payload",
    "comparable",
    "evaluate_candidate",
    "evaluate_triggers",
    "format_semver",
    "next_version",
    "normalize_digest",
    "parse_semver",
    "pipeline_dataset_fingerprint",
    "plan_rollback",
    "semver_sort_key",
    "trigger_table",
    "utc_now_iso",
    "verify_data",
    "version_key",
]
