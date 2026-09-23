"""MLOps 微调流水线包（M5-D10）：把训练、评估、门禁、打包、发布串成一次运行.

M5 走到第十天，散落的"零件"已经齐了——数据（day048/057）、训练
（day050/051/052）、评估（day053）、对齐（day054/055）、版本（day058）。
本包把它们收口成一条**可被 CI 驱动、可被评审读懂**的流水线：

```text
ingest → train → evaluate → gate → package → publish
取数据    训练     评估       门禁     打包      发布
```

六个模块，每个回答一个问题：

```text
tracking.py   这次实验用了什么参数、结果多少、产出了哪些文件
gates.py      这次的产物够格发布吗（绝对判定）
stages.py     六个阶段分别依赖什么、产出什么（可被程序校验）
packaging.py  它是什么、好不好、凭什么可信、**不能做什么**
ci.py         什么时候自动跑、按什么阈值判定
pipeline.py   把上面五件事串成一次可复盘的动作
```

## 三条贯穿全包的纪律

1. **"出错"与"做不到"分开报**。门禁拦下发布时 ``publish`` 的状态是
   ``blocked``，不是 ``failed``——前者是预期内的结果，后者意味着有东西坏了。
   这与 day058 把"没有退路"输出成 ``hold`` 而不是抛异常是同一条纪律；
2. **缺失就是缺失**。``Run.metric`` 返回 ``None``、门禁在缺指标时判**不通过**、
   追踪器允许 ``actual=None``——**相对判定缺数据只能说"不知道"，
   绝对判定缺数据必须说"不行"**；
3. **文档与实现同源**。门禁表（``gate_table``）、阶段表（``stage_table``）、
   CI 摘要（``workflow_summary``）全部从代码常量现场读出，而且
   **仓库里那份 workflow 文件由渲染器生成并有测试守着**。

## 与既有包的接缝

- **上游**：``registry``（版本三元组与注册表）、``domain_data``（数据集指纹）、
  ``finetune_eval`` 与 ``peft``（指标与适配器哈希的来源）。它们一行都不用改；
- **下游**：``api.routes`` 的 ``/mlops/*`` 端点只做**只读或纯计算**。
  真正的"训练 + 发布"由 ``scripts/mlops_demo.py`` 与 CI 承担——
  一个需要十分钟的流程不该挂在一个 HTTP 请求上。
"""

from __future__ import annotations

from smart_research_agent.mlops.ci import (
    GATE_SCRIPT,
    WORKFLOW_DIRECTORY,
    WORKFLOW_FILENAME,
    CIConfig,
    parse_workflow,
    render_github_actions,
    workflow_commands,
    workflow_summary,
)
from smart_research_agent.mlops.errors import MLOpsError
from smart_research_agent.mlops.gates import (
    GATE_ADAPTER_SIZE,
    GATE_ARTIFACT_COMPLETENESS,
    GATE_COST,
    GATE_DATASET_TRACEABILITY,
    GATE_METRIC_NAMES,
    GATE_METRIC_PREFIX,
    GATE_PASS_RATE,
    GATE_REGRESSION,
    REQUIRED_ARTIFACTS,
    TRACEABILITY_KEYS,
    GateCheck,
    GateReport,
    ReleaseGates,
    evaluate_gates,
    gate_table,
)
from smart_research_agent.mlops.packaging import (
    INTENDED_USE,
    LIMITATIONS,
    MODEL_CARD_FILENAME,
    OUT_OF_SCOPE,
    RELEASE_MANIFEST_FILENAME,
    ModelCard,
    build_model_card,
    build_release_manifest,
)
from smart_research_agent.mlops.pipeline import (
    EVALUATE_REQUIRED_FIELDS,
    TRACK_ARTIFACT_ADAPTER,
    TRACK_ARTIFACT_MANIFEST,
    TRACK_ARTIFACT_MERGED,
    TRACK_ARTIFACT_MODEL_CARD,
    TRAIN_REQUIRED_FIELDS,
    FinetunePipeline,
    PipelineOutcome,
)
from smart_research_agent.mlops.stages import (
    ARTIFACT_ADAPTER,
    ARTIFACT_DATASET,
    ARTIFACT_GATE_REPORT,
    ARTIFACT_METRICS,
    ARTIFACT_MODEL_CARD,
    ARTIFACT_VERSION,
    DEFAULT_ON_FAILURE_BLOCKING,
    PIPELINE_STAGES,
    STAGE_BLOCKED,
    STAGE_EVALUATE,
    STAGE_FAILED,
    STAGE_GATE,
    STAGE_INGEST,
    STAGE_OK,
    STAGE_PACKAGE,
    STAGE_PUBLISH,
    STAGE_SKIPPED,
    STAGE_SPECS,
    STAGE_STATUSES,
    STAGE_TRAIN,
    StageResult,
    StageSpec,
    critical_path,
    stage_table,
    validate_stage_order,
)
from smart_research_agent.mlops.tracking import (
    METRIC_MODE_MAX,
    METRIC_MODE_MIN,
    METRIC_MODES,
    RELATION_BETTER,
    RELATION_EQUAL,
    RELATION_WORSE,
    RUN_ID_LENGTH,
    RUNS_FILENAME,
    STATUSES,
    STATUS_FAILED,
    STATUS_FINISHED,
    STATUS_RUNNING,
    ExperimentTracker,
    Run,
    compare_runs,
    run_id_for,
)

__all__ = [
    "ARTIFACT_ADAPTER",
    "ARTIFACT_DATASET",
    "ARTIFACT_GATE_REPORT",
    "ARTIFACT_METRICS",
    "ARTIFACT_MODEL_CARD",
    "ARTIFACT_VERSION",
    "CIConfig",
    "DEFAULT_ON_FAILURE_BLOCKING",
    "EVALUATE_REQUIRED_FIELDS",
    "GATE_ADAPTER_SIZE",
    "GATE_ARTIFACT_COMPLETENESS",
    "GATE_COST",
    "GATE_DATASET_TRACEABILITY",
    "GATE_METRIC_NAMES",
    "GATE_METRIC_PREFIX",
    "GATE_PASS_RATE",
    "GATE_REGRESSION",
    "GATE_SCRIPT",
    "INTENDED_USE",
    "LIMITATIONS",
    "METRIC_MODES",
    "METRIC_MODE_MAX",
    "METRIC_MODE_MIN",
    "MODEL_CARD_FILENAME",
    "OUT_OF_SCOPE",
    "PIPELINE_STAGES",
    "RELATION_BETTER",
    "RELATION_EQUAL",
    "RELATION_WORSE",
    "RELEASE_MANIFEST_FILENAME",
    "REQUIRED_ARTIFACTS",
    "RUNS_FILENAME",
    "RUN_ID_LENGTH",
    "STATUSES",
    "STATUS_FAILED",
    "STATUS_FINISHED",
    "STATUS_RUNNING",
    "STAGE_BLOCKED",
    "STAGE_EVALUATE",
    "STAGE_FAILED",
    "STAGE_GATE",
    "STAGE_INGEST",
    "STAGE_OK",
    "STAGE_PACKAGE",
    "STAGE_PUBLISH",
    "STAGE_SKIPPED",
    "STAGE_SPECS",
    "STAGE_STATUSES",
    "STAGE_TRAIN",
    "TRACEABILITY_KEYS",
    "TRACK_ARTIFACT_ADAPTER",
    "TRACK_ARTIFACT_MANIFEST",
    "TRACK_ARTIFACT_MERGED",
    "TRACK_ARTIFACT_MODEL_CARD",
    "TRAIN_REQUIRED_FIELDS",
    "WORKFLOW_DIRECTORY",
    "WORKFLOW_FILENAME",
    "ExperimentTracker",
    "FinetunePipeline",
    "GateCheck",
    "GateReport",
    "MLOpsError",
    "ModelCard",
    "PipelineOutcome",
    "ReleaseGates",
    "Run",
    "StageResult",
    "StageSpec",
    "build_model_card",
    "build_release_manifest",
    "compare_runs",
    "critical_path",
    "evaluate_gates",
    "gate_table",
    "parse_workflow",
    "render_github_actions",
    "run_id_for",
    "stage_table",
    "validate_stage_order",
    "workflow_commands",
    "workflow_summary",
]
