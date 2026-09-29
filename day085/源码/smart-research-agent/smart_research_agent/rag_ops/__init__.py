"""``rag_ops``：RAG 链路的生产化运维（M6-D10 / day072）.

M6 的九课（day061 ~ day071）交出一条**能回答问题、也能被评估**的链路，
而 day071 最后留下的一句话是它的边界：

```text
"11 个质量指标 + RagBaseline —— day072 要把它接进定时评估：
  '索引增量更新之后质量变了吗'。这个问题今天已经有判据了——
  缺的只是'谁来定期跑、跑完通知谁'。"
```

本模块就是那个"谁来定期跑、跑完通知谁"。它要回答三个问题，
而这三个问题**都不是模型问题**：

```text
① 该不该跑？            调度：间隔 + 抖动 + 退避 + 错过窗口
② 该重算什么？          两级增量：文件级（谁变了）+ 块级（哪些向量要重算）
③ 现在能不能服务？      体检 + 监控：库空不空、清单对不对、语料新不新、质量退没退
```

## 一、这一课最核心的一件事：两级增量

"增量"这个词在 M6 里出现过两次，而它们回答的是**不同的问题**：

```text
文件级增量（本模块）   两份快照的差集         → "要不要跑这一趟"
  语料一份没变 → 一行都不写（连解析与分块都不做）
块级增量（day065）     planner 的四条判据      → "要重算哪些向量"
  一份文档改了一行 → 只重算变了的块（缓存命中，其余判 unchanged）
```

两者必须**都在**，任何一个单独拿出来都会浪费掉另一半：

```text
只做块级 → 每天都要把语料全部解析、全部切块，才能知道"哪些块变了"
只做文件级 → 一份 4000 字的文档改一个字，它的 14 块全部重新编码
```

差额本身也是一条读数：一次"改了一份文档"在文件级是 ``sources_changed = 1``，
在块级可能是 ``chunks_written = 14``——只看其中一个都会对本次改动规模的判断失准。

## 二、部署是"数据 + 判据 + 渲染"，不是"手写一份 yml"

```text
ComposeSpec（三个服务的规格）
    ↓ validate_spec：11 条静态判据（探针 / 镜像 tag / 持久卷 / 端口 / 依赖 …）
    ↓ render_compose：确定性的 YAML 文本
docker-compose.rag.yml（进 git，被测试逐字节守着）
```

"容器起来了但库是空的""镜像不可追溯""探针永远绿""卷没声明，重建即丢"
这四类事故全都能在提交时被查出来，因此它们不该留到部署时才发现。

## 三、"没体检"不许读成"健康"

health 的每一项在"缺输入"时给 ``warn`` 而不是 ``ok``，报告为空时**直接报错**：
一份"一项都没查"的报告与一份"全绿"的报告在状态上长得一模一样，
而它们的含义完全相反（与 day071 的"不下结论 ≠ 通过"是同一条纪律）。

## 模块地图（每个模块回答一个问题）

```text
errors.py    这一层会怎么失败（账本 / 调度 / 健康 / 监控 / 编排 五族，修复人各不相同）
types.py     五副形状 + 六张封闭表（动作 / 模式 / 三态 / 检查项 / 指标 / 运行状态）
snapshot.py  语料侧的快照：扫一遍目录，把"现在长什么样"变成一份可 diff 的账
sync.py      两级增量的上半段：两次快照的差集，以及"这次该怎么建"的那一个决定
schedule.py  调度：什么时候该跑、为什么、下一次在哪（纯函数，可被推到任意时刻）
health.py    健康判据：四项检查（库 / 清单 / 语料新鲜度 / 质量），探活不跑评估
metrics.py   监控采样：11 个指标 + 采样 + 序列（质量那四个是**抄** day071，不是重算）
deploy.py    编排规格：三个服务 + 11 条判据 + 确定性渲染 + 与仓库文件的一致性
pipeline.py  运维流水线：扫盘 → 差集 → 重建 → 体检 → 采样 → 落账（十步固定顺序）
worker.py    定时同步容器：循环、等待、退避（可注入 sleep 与时钟，因此能在 CI 里推演）
```

## 四条贯穿全包的纪律

1. **水位只在成功时前进**（``SyncLedger.mark_success``）。失败推进水位的后果是
   那批来源被永久跳过，而库里少的记录没有任何地方会提醒你。
2. **判定与执行分开**。``decide`` / ``plan_sync`` / ``validate_spec`` 都是纯函数；
   端点因此可以在**不产生任何副作用**的前提下回答"下一次什么时候跑""这次会改什么"。
3. **跳过不是通过**。缺清单、缺基线、没评估一律 ``warn``；只有"这一秒就在骗人"
   的事（库空、清单与库不符、质量判定回归）才是 ``fail``。
4. **状态必须落盘，且顺序不能反**。先写向量库快照、再写版本表——
   版本表是"库应该长什么样"的声明，它必须晚于库本身。

## 与既有包的接缝

- **上游**：``documents``（day061）把文件变成 ``Document``；``indexing``（day065）
  提供构建器、清单与版本表；``rag_debug``（day071）提供质量基线与门禁；
  ``observability``（day032/043）提供成本与追踪的口径；
- **脚下**：``config`` 的 ``rag_ops_*`` 一组（语料目录、账本、间隔、抖动、退避、
  新鲜度阈值、端口与镜像 tag）——它们**不改变任何产物字节**，只决定"怎么运维"；
- **下游**：``worker`` 与 ``docker-compose.rag.yml`` 是它的两个运行形态（进程内 / 容器内）；
- **端点**：``api.routes`` 的 ``/rag/ops/*``（status / health / sync）；
  演示脚本 ``scripts/rag_ops_demo.py``；手册 ``docs/rag_ops.md``。
"""

from __future__ import annotations

from smart_research_agent.rag_ops.deploy import (
    API_CONTAINER_PORT,
    API_HEALTH_PATH,
    CHROMA_HEARTBEAT_PATH,
    CHROMA_IMAGE,
    CHROMA_TAG,
    FORBIDDEN_IMAGE_TAGS,
    IMAGE_REPOSITORY,
    PROBE_TEST_FORMS,
    SERVICE_API,
    SERVICE_ROLE_DESCRIPTIONS,
    SERVICE_ROLES,
    SERVICE_STORE,
    SERVICE_WORKER,
    VALIDATION_RULES,
    ComposeSpec,
    HealthProbe,
    PortMapping,
    ServiceSpec,
    VolumeMount,
    check_committed_compose,
    compose_summary_lines,
    default_spec,
    render_compose,
    require_valid,
    validate_spec,
    write_compose,
)
from smart_research_agent.rag_ops.deploy import describe as describe_spec
from smart_research_agent.rag_ops.errors import (
    DeployError,
    HealthError,
    MetricsError,
    RagOpsError,
    ScheduleError,
    SyncError,
)
from smart_research_agent.rag_ops.health import (
    build_health_report,
    check_corpus_freshness,
    check_index_integrity,
    check_quality_regression,
    check_vector_store,
)
from smart_research_agent.rag_ops.metrics import (
    QUALITY_SAMPLE_SOURCES,
    MetricsBuffer,
    collect_samples,
    samples_from_ledger,
    samples_from_plan,
    samples_from_quality,
    samples_from_store,
)
from smart_research_agent.rag_ops.pipeline import (
    LEDGER_FILENAME,
    REPORT_FILENAME,
    Clock,
    QualityLoader,
    RagOpsPipeline,
    default_rag_ops_pipeline,
    indexer_versions,
    paths_in,
    policy_from_settings,
    quality_from_report,
)
from smart_research_agent.rag_ops.schedule import decide, jitter_seconds
from smart_research_agent.rag_ops.schedule import describe as describe_schedule
from smart_research_agent.rag_ops.snapshot import (
    entry_from_document,
    relative_source,
    resolve_source_root,
    scan_corpus,
    scan_sources,
)
from smart_research_agent.rag_ops.sync import (
    changed_lines,
    plan_sync,
    resolved_mode,
    suggest_mode,
)
from smart_research_agent.rag_ops.types import (
    ACTION_ADDED,
    ACTION_REMOVED,
    ACTION_UNCHANGED,
    ACTION_UPDATED,
    CHECK_CORPUS_FRESHNESS,
    CHECK_INDEX_INTEGRITY,
    CHECK_QUALITY_REGRESSION,
    CHECK_VECTOR_STORE,
    FINGERPRINT_LENGTH,
    HEALTH_CHECK_DESCRIPTIONS,
    HEALTH_CHECKS,
    HEALTH_STATE_DESCRIPTIONS,
    HEALTH_STATE_EXIT_CODES,
    HEALTH_STATE_RANK,
    HEALTH_STATES,
    LEDGER_VERSION,
    METRIC_CHUNKS_WRITTEN,
    METRIC_INDEX_RECORDS,
    METRIC_QUALITY_BAD_CASE,
    METRIC_QUALITY_GROUNDED,
    METRIC_QUALITY_HALLUCINATION,
    METRIC_QUALITY_RECALL,
    METRIC_RUN_FAILURES,
    METRIC_SOURCES_CHANGED,
    METRIC_SOURCES_TOTAL,
    METRIC_SYNC_AGE_MINUTES,
    METRIC_SYNC_SECONDS,
    METRIC_UNIT_COUNT,
    METRIC_UNIT_MINUTES,
    METRIC_UNIT_RATIO,
    METRIC_UNIT_SECONDS,
    METRIC_UNITS,
    MODE_BACKOFF,
    MODE_DUE,
    MODE_FIRST_RUN,
    MODE_FORCED,
    MODE_NOT_DUE,
    OPS_METRIC_DESCRIPTIONS,
    OPS_METRIC_UNITS,
    OPS_METRICS,
    OPS_REPORT_VERSION,
    OPS_STATUS_DESCRIPTIONS,
    OPS_STATUSES,
    SCHEDULE_MODE_DESCRIPTIONS,
    SCHEDULE_MODES,
    SNAPSHOT_ID_LENGTH,
    SNAPSHOT_VERSION,
    STATE_FAIL,
    STATE_OK,
    STATE_WARN,
    STATUS_FAILED,
    STATUS_NO_CHANGE,
    STATUS_NOT_DUE,
    STATUS_SYNCED,
    SYNC_ACTION_DESCRIPTIONS,
    SYNC_ACTION_IS_CHANGE,
    SYNC_ACTIONS,
    CorpusSnapshot,
    HealthFinding,
    HealthReport,
    MetricSample,
    MetricSeries,
    OpsReport,
    ScheduleDecision,
    SchedulePolicy,
    SourceEntry,
    SyncLedger,
    SyncPlan,
    minutes_between,
    parse_iso,
    shift_iso,
    snapshot_from_entries,
    snapshot_id,
    to_iso,
    utc_now,
    worst_state,
)
from smart_research_agent.rag_ops.worker import (
    MAXIMUM_WAIT_SECONDS,
    MINIMUM_WAIT_SECONDS,
    build_parser,
    main,
    run_forever,
    seconds_until_next,
)

__all__ = [
    "ACTION_ADDED",
    "ACTION_REMOVED",
    "ACTION_UNCHANGED",
    "ACTION_UPDATED",
    "API_CONTAINER_PORT",
    "API_HEALTH_PATH",
    "CHECK_CORPUS_FRESHNESS",
    "CHECK_INDEX_INTEGRITY",
    "CHECK_QUALITY_REGRESSION",
    "CHECK_VECTOR_STORE",
    "CHROMA_HEARTBEAT_PATH",
    "CHROMA_IMAGE",
    "CHROMA_TAG",
    "FINGERPRINT_LENGTH",
    "FORBIDDEN_IMAGE_TAGS",
    "HEALTH_CHECKS",
    "HEALTH_CHECK_DESCRIPTIONS",
    "HEALTH_STATES",
    "HEALTH_STATE_DESCRIPTIONS",
    "HEALTH_STATE_EXIT_CODES",
    "HEALTH_STATE_RANK",
    "IMAGE_REPOSITORY",
    "LEDGER_FILENAME",
    "LEDGER_VERSION",
    "MAXIMUM_WAIT_SECONDS",
    "METRIC_CHUNKS_WRITTEN",
    "METRIC_INDEX_RECORDS",
    "METRIC_QUALITY_BAD_CASE",
    "METRIC_QUALITY_GROUNDED",
    "METRIC_QUALITY_HALLUCINATION",
    "METRIC_QUALITY_RECALL",
    "METRIC_RUN_FAILURES",
    "METRIC_SOURCES_CHANGED",
    "METRIC_SOURCES_TOTAL",
    "METRIC_SYNC_AGE_MINUTES",
    "METRIC_SYNC_SECONDS",
    "METRIC_UNITS",
    "METRIC_UNIT_COUNT",
    "METRIC_UNIT_MINUTES",
    "METRIC_UNIT_RATIO",
    "METRIC_UNIT_SECONDS",
    "MINIMUM_WAIT_SECONDS",
    "MODE_BACKOFF",
    "MODE_DUE",
    "MODE_FIRST_RUN",
    "MODE_FORCED",
    "MODE_NOT_DUE",
    "OPS_METRICS",
    "OPS_METRIC_DESCRIPTIONS",
    "OPS_METRIC_UNITS",
    "OPS_REPORT_VERSION",
    "OPS_STATUSES",
    "OPS_STATUS_DESCRIPTIONS",
    "PROBE_TEST_FORMS",
    "QUALITY_SAMPLE_SOURCES",
    "REPORT_FILENAME",
    "SCHEDULE_MODES",
    "SCHEDULE_MODE_DESCRIPTIONS",
    "SERVICE_API",
    "SERVICE_ROLES",
    "SERVICE_ROLE_DESCRIPTIONS",
    "SERVICE_STORE",
    "SERVICE_WORKER",
    "SNAPSHOT_ID_LENGTH",
    "SNAPSHOT_VERSION",
    "STATE_FAIL",
    "STATE_OK",
    "STATE_WARN",
    "STATUS_FAILED",
    "STATUS_NOT_DUE",
    "STATUS_NO_CHANGE",
    "STATUS_SYNCED",
    "SYNC_ACTIONS",
    "SYNC_ACTION_DESCRIPTIONS",
    "SYNC_ACTION_IS_CHANGE",
    "VALIDATION_RULES",
    "Clock",
    "ComposeSpec",
    "CorpusSnapshot",
    "DeployError",
    "HealthError",
    "HealthFinding",
    "HealthProbe",
    "HealthReport",
    "MetricsBuffer",
    "MetricsError",
    "MetricSample",
    "MetricSeries",
    "OpsReport",
    "PortMapping",
    "QualityLoader",
    "RagOpsError",
    "RagOpsPipeline",
    "ScheduleDecision",
    "ScheduleError",
    "SchedulePolicy",
    "ServiceSpec",
    "SourceEntry",
    "SyncError",
    "SyncLedger",
    "SyncPlan",
    "VolumeMount",
    "build_health_report",
    "build_parser",
    "changed_lines",
    "check_committed_compose",
    "check_corpus_freshness",
    "check_index_integrity",
    "check_quality_regression",
    "check_vector_store",
    "collect_samples",
    "compose_summary_lines",
    "decide",
    "default_rag_ops_pipeline",
    "default_spec",
    "describe_schedule",
    "describe_spec",
    "entry_from_document",
    "indexer_versions",
    "jitter_seconds",
    "main",
    "minutes_between",
    "parse_iso",
    "paths_in",
    "plan_sync",
    "policy_from_settings",
    "quality_from_report",
    "relative_source",
    "render_compose",
    "require_valid",
    "resolve_source_root",
    "resolved_mode",
    "run_forever",
    "samples_from_ledger",
    "samples_from_plan",
    "samples_from_quality",
    "samples_from_store",
    "scan_corpus",
    "scan_sources",
    "seconds_until_next",
    "shift_iso",
    "snapshot_from_entries",
    "snapshot_id",
    "suggest_mode",
    "to_iso",
    "utc_now",
    "validate_spec",
    "worst_state",
    "write_compose",
]
