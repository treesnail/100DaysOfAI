"""``rag_debug``：RAG 的端到端评估与调试（M6-D9 / day071）.

M6 的前八课（day061 ~ day069）交出一条**能回答问题的链路**，而 day070 这个
复习日把这条链路画成了一张走读表：从字节到可核对答案有九个决定，每一层都
留下了自己的数字。本模块把那两天的人工流程变成代码，回答三个问题：

```text
① 这一批答案好不好？        → 11 个质量指标（进基线、可回归）
② 不好在哪一层？            → 12 类坏例标签（按链路顺序归因，只给一个）
③ 下一步动哪个旋钮？        → 每类坏例一个动作（去重成一份动作清单）
```

## 一、这一课最核心的一件事：把"召回"拆成两段

day066 的召回率衡量"金标准有没有被检索到"；day069 的覆盖率衡量"答案有没有
用上片段"。两者之间**还有一段漏斗**，而它在此之前没有被单独度量过：

```text
检索名单（retrieved）  →  打包（pack_context）  →  提示词（packed）  →  答案
      recall@k                                     packed_recall@k      grounded
      └──────────── day066~day068 的账 ────────────┘
                              ↑
                  这一段差额是**预算**的账（dropped_hits / truncated_hits）
```

金标准"在名单里但不在提示词里"是一门**独立的失败**：检索抓对了，模型却
没有机会用它。合成一个召回率之后，它会消失在均值里——而它的处置动作
（调 ``retrieval_max_context_chars``）与另两种（改检索、改提示词）完全不同。

这就是本模块把 ``StageMetrics`` 分成 ``STAGE_RETRIEVED`` 与 ``STAGE_PACKED``
两段、并额外产出 ``packed_away`` 这个条数的原因。

## 二、归因的顺序就是链路的顺序

``diagnose`` 逐条判、命中即返回，而判据的排列（``BAD_CASE_PRIORITY``）抄的是
day070 教程第三章那张人工诊断表：

```text
库为空 → 检索为空 → 检索没抓到 → 重排切掉 → 没进提示词 → 排太后 →
片段被截断 → 走了回退 → 幻觉引用 → 没有有效引用 → 覆盖率低 → 忠实度低
```

**只给一个标签**是刻意的：一条坏例往往同时满足多个判据，但它们的因果顺序
是固定的（检索没抓到的时候，答案当然也没有引用）。一次给出全部标签会把
因果关系摊平成一张清单，而读的人只会去修那个最显眼的（最后一个）。

## 三、质量基线与性能基线的方向相反

day046 的 ``evaluation.perf_baseline`` 用**比值**判回归（延迟/token/费用的
"越大越差"）；本模块的质量基线用**差值**判（召回/接地的"越小越差"）。
两处不能共用一份实现，而原因不止"方向不同"：

```text
质量指标全是 [0,1] 的比例   → 差值可以直接读成"几个百分点"
基数为 0 是常态（幻觉率 0）  → 比值在 0 基数下没有定义，差值照样可判
第三种方向存在              → llm_call_rate 只做描述，不判方向
```

## 模块地图（每个模块回答一个问题）

```text
errors.py     这一层会怎么失败（用例数据 / 归因用法 / 基线与当前不可比）
types.py      四副形状：用例、两段指标、一次运行的结果、一次归因；三张封闭表
runner.py     跑一条用例：把 RagAnswer 搬成一份扁平、可序列化、可比较的账
diagnose.py   归因：一个标签 + 一句话 + 一个动作（纯函数）
baseline.py   质量基线：11 个指标的存档、方向表、死区、三道可比性检查
report.py     报告：指标 + 逐条证据 + 坏例表 + 动作清单；A/B 对照
suite.py      套件：评测集 + 语料 + 变体表（"一次只改一个旋钮"的数据形状）
```

## 四条贯穿全包的纪律

1. **判定不靠字符串比较**（day069 起）：接地看 ``valid``/``invalid`` 三组编号，
   调没调模型看 ``llm_called``，"有没有回退"看 ``fallback_reason`` 这个封闭值。
2. **闭合清单必须逐键对齐**：坏例标签的三张表（标签 / 解释 / 动作）、质量指标
   的三张表（方向 / 容忍度 / 说明）在**导入期**就会校验一致性——少一个键
   意味着某条判据静默失效，而那与"判了没问题"在报告里长得一样。
3. **指标口径只有一份实现**：四个检索指标直接复用 ``retrieval.rerank`` 与
   ``evaluation.rag_metrics`` 的纯函数，忠实度复用 ``evaluation.rag_eval``
   的提示词与解析（day071 只把它公开，没有重写）。
4. **降级必须留痕**：评审模型输出无法解析时 ``faithfulness=None`` 并写进
   ``CaseOutcome.notes``——否则那一批的忠实度均值里会多出一个看不见的空洞。

## 与既有包的接缝

- **上游**：``retrieval``（day066 ~ day069）交出 ``RagAnswer``——
  ``retrieval.ids`` 是 retrieved 那一段、``context.citations`` 是 packed
  那一段、``check`` 是生成侧的三个编号集合、``fallback_reason`` 是六取一；
  ``evaluation``（day025 / day029 / day046）提供 harness、四个检索指标、
  忠实度评审与 ``percentile``；
- **脚下**：``config`` 的 ``rag_eval_*`` 一组（评测集路径、判据阈值、基线路径、
  三门禁参数）——它们**不改变任何产物字节**，只改变"怎么评"；
- **下游**：day072（M6-D10 生产化部署）把这份基线接进定时评估；
  报告里的 11 个指标就是"这一次部署有没有变差"的判据；
- **端点**：``api.routes`` 的 ``/rag/eval/*``；演示脚本 ``scripts/rag_debug_demo.py``；
  手册 ``docs/rag_debug.md``。
"""

from __future__ import annotations

from smart_research_agent.rag_debug.baseline import (
    DEFAULT_DEAD_BAND,
    DEFAULT_MIN_SAMPLES,
    DEFAULT_TOLERANCES,
    DIRECTION_HIGHER_BETTER,
    DIRECTION_LOWER_BETTER,
    DIRECTION_NEUTRAL,
    DIRECTIONS,
    METRIC_BAD_CASE_RATE,
    METRIC_COVERAGE,
    METRIC_DESCRIPTIONS,
    METRIC_DIRECTIONS,
    METRIC_FALLBACK_RATE,
    METRIC_GROUNDED_RATE,
    METRIC_HALLUCINATION_RATE,
    METRIC_LLM_CALL_RATE,
    METRIC_PACKED_NDCG,
    METRIC_PACKED_RECALL,
    METRIC_RETRIEVAL_NDCG,
    METRIC_RETRIEVAL_PRECISION,
    METRIC_RETRIEVAL_RECALL,
    QUALITY_METRICS,
    BaselineComparison,
    QualityDelta,
    QualityRegression,
    RagBaseline,
    RagBaselineGuard,
    compare_with_file,
    deltas_between,
    guard_from_settings,
)
from smart_research_agent.rag_debug.diagnose import (
    action_plan,
    diagnose,
    diagnose_all,
    tag_counts,
)
from smart_research_agent.rag_debug.errors import (
    BaselineError,
    CaseDataError,
    DiagnosisError,
    RagDebugError,
)
from smart_research_agent.rag_debug.report import (
    RagEvalReport,
    VariantComparison,
    build_report,
    compare_variants,
    metric_table,
)
from smart_research_agent.rag_debug.runner import (
    DEFAULT_EVAL_K,
    RagEvalRunner,
    score_faithfulness,
)
from smart_research_agent.rag_debug.suite import (
    DEFAULT_SUITE_LABEL,
    DEFAULT_VARIANTS,
    TIGHT_BUDGET_CHARS,
    RagEvalSuite,
    Variant,
    build_suite,
    index_corpus,
    load_cases,
    load_corpus,
)
from smart_research_agent.rag_debug.types import (
    BAD_CASE_ACTIONS,
    BAD_CASE_DESCRIPTIONS,
    BAD_CASE_PRIORITY,
    BAD_CASE_TAGS,
    DEFAULT_MIN_COVERAGE,
    DEFAULT_MIN_FAITHFULNESS,
    DEFAULT_MIN_RECIPROCAL_RANK,
    FALLBACK_ACTIONS,
    RETRIEVAL_METRICS,
    STAGE_PACKED,
    STAGE_RETRIEVED,
    STAGES,
    TAG_GENERATION_FALLBACK,
    TAG_HALLUCINATED_CITATION,
    TAG_LOW_COVERAGE,
    TAG_NO_DATA,
    TAG_PACKED_AWAY,
    TAG_RANK_BAD,
    TAG_RERANK_CUT,
    TAG_RETRIEVAL_EMPTY,
    TAG_RETRIEVAL_MISS,
    TAG_TRUNCATED_CHUNK,
    TAG_UNGROUNDED,
    TAG_WEAK_FAITHFULNESS,
    BadCase,
    CaseOutcome,
    RagEvalCase,
    StageMetrics,
    bad_case_rate,
    described_tags,
    measure_ids,
)

__all__ = [
    "BAD_CASE_ACTIONS",
    "BAD_CASE_DESCRIPTIONS",
    "BAD_CASE_PRIORITY",
    "BAD_CASE_TAGS",
    "DEFAULT_DEAD_BAND",
    "DEFAULT_EVAL_K",
    "DEFAULT_MIN_COVERAGE",
    "DEFAULT_MIN_FAITHFULNESS",
    "DEFAULT_MIN_RECIPROCAL_RANK",
    "DEFAULT_MIN_SAMPLES",
    "DEFAULT_SUITE_LABEL",
    "DEFAULT_TOLERANCES",
    "DEFAULT_VARIANTS",
    "DIRECTIONS",
    "DIRECTION_HIGHER_BETTER",
    "DIRECTION_LOWER_BETTER",
    "DIRECTION_NEUTRAL",
    "FALLBACK_ACTIONS",
    "METRIC_BAD_CASE_RATE",
    "METRIC_COVERAGE",
    "METRIC_DESCRIPTIONS",
    "METRIC_DIRECTIONS",
    "METRIC_FALLBACK_RATE",
    "METRIC_GROUNDED_RATE",
    "METRIC_HALLUCINATION_RATE",
    "METRIC_LLM_CALL_RATE",
    "METRIC_PACKED_NDCG",
    "METRIC_PACKED_RECALL",
    "METRIC_RETRIEVAL_NDCG",
    "METRIC_RETRIEVAL_PRECISION",
    "METRIC_RETRIEVAL_RECALL",
    "QUALITY_METRICS",
    "RETRIEVAL_METRICS",
    "STAGES",
    "STAGE_PACKED",
    "STAGE_RETRIEVED",
    "TAG_GENERATION_FALLBACK",
    "TAG_HALLUCINATED_CITATION",
    "TAG_LOW_COVERAGE",
    "TAG_NO_DATA",
    "TAG_PACKED_AWAY",
    "TAG_RANK_BAD",
    "TAG_RERANK_CUT",
    "TAG_RETRIEVAL_EMPTY",
    "TAG_RETRIEVAL_MISS",
    "TAG_TRUNCATED_CHUNK",
    "TAG_UNGROUNDED",
    "TAG_WEAK_FAITHFULNESS",
    "TIGHT_BUDGET_CHARS",
    "BadCase",
    "BaselineComparison",
    "BaselineError",
    "CaseDataError",
    "CaseOutcome",
    "DiagnosisError",
    "QualityDelta",
    "QualityRegression",
    "RagBaseline",
    "RagBaselineGuard",
    "RagDebugError",
    "RagEvalCase",
    "RagEvalReport",
    "RagEvalRunner",
    "RagEvalSuite",
    "StageMetrics",
    "Variant",
    "VariantComparison",
    "action_plan",
    "bad_case_rate",
    "build_report",
    "build_suite",
    "compare_variants",
    "compare_with_file",
    "deltas_between",
    "described_tags",
    "diagnose",
    "diagnose_all",
    "guard_from_settings",
    "index_corpus",
    "load_cases",
    "load_corpus",
    "measure_ids",
    "metric_table",
    "score_faithfulness",
    "tag_counts",
]
