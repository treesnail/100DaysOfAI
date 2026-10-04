"""``inference_optim``：推理路径上的三笔账（M7-D11 / day087）.

day086 把模型**接进来**了；今天的全部动作都发生在那之后，而只发生在一侧：
**推理**。三样东西可以动，而它们互不相干，各自换一种东西：

```text
① 缓存       别再重算已经算过的 K/V —— 省时间、花显存
② 精度       把权重从 32 位压到 8 位 / 4 位 —— 省显存、花精度
③ 批         把若干条样本绑在一起跑 —— 省单位吞吐、花首字延迟
```

## 一、今天最值钱的一句话

> **缓存不是"变快"，而是"换"：它把每步的计算量从 O(T) 降到 O(1)，代价是 O(T) 的显存。**

这句话有一个立刻可用的推论：显存紧的时候"缓存一定更好"是**错的**。
因此本课每一条读数都带**方向**（`higher` / `lower` / `neutral`），
而"越大越好"这种话一句都不写。

## 二、七条性质（判据分四类）

```text
逐位（==）    缓存路径 vs 整段路径（同一次算术）、int4 打包往返
整数相等      每步字节增量 vs 2·L·h·bytes、预算三项之和、占用槽位数
上界判定      反量化误差 ≤ scale/2   ← **今天新出现的一类：两个操作数不该相等**
两侧都查      预算里 T_max 放得下、T_max+1 放不下
```

第二类是本课唯一一处"两个数相等"与"一个数不超过另一个数"混在一起的地方——
因此 `CrossCheck` 多了一个 `upper_bound` 字段：有它的时候判据是上界，
没有的时候才是相等。把两类混成一个判据，就会出现"实测误差恰好等于 0（因为输入全是 0）
被当成通过"这种事。

## 三、八个模块

```text
errors.py     七个失败族（**缺席的仍是 GradientError**，理由第三次换了一条：量化不可微）
types.py      四种精度 / 两种方案 × 两种粒度 / 缓存两种布局与两个阶段 / 七条性质 / 十条笔记 / 五条边界
cache.py      K/V 缓存：prefill 与 decode 两阶段，与"整段重算"**逐位**相同
quantize.py   定标 / 取整 / 打包**三件事分开**：误差只发生在取整那一步
batching.py   静态批与连续批的步数、槽位占用与吞吐
budget.py     权重 + 缓存 + 激活三项的账，以及一个能被整除算出来的 T_max
verify.py     七条性质与三类对账
study.py      六张表
```

## 四、四条纪律

1. **三笔账必须能被逐项相加**：权重 + 缓存 + 激活 = 总量；Σ 每步增量 = 总缓存；
   Σ 各条长度 = 有效 token。一个不能被逐项拆开的数没法被反驳。
2. **方向必须写清楚**：三样东西都是"换来"而不是"变好"，因此每条读数都带方向。
3. **上界与相等必须分开判**：`CrossCheck.upper_bound` 有值时走"≤"，无值时走"=="。
4. **两侧都要查**：预算里只查"``T_max`` 放得下"时，一个**偏小**的最大值也会通过——
   而"保守"与"算错"读起来一样。

## 五、与既有包的接缝

- **上游**：`hf_integration`（day086：卡片、权重、`hidden_states`、`logits`、
  `run_one_block`、`embed_tokens`）、`hf_source`（day085：`split_heads` / `row_softmax` /
  `merge_heads` / `hf_scale`）、`math_foundations`（day073：`matmul` / `transpose`）；
- **脚下**：`config.py` **没有**新增配置项——精度、容量、批大小都是函数参数；
- **下游**：day088（项目底层原理串联）会用今天这三笔账写"原理 → 应用"的提纲里
  最实用的一节（"为什么这个模型在我的卡上跑不起来"）。
"""

from __future__ import annotations

from smart_research_agent.inference_optim.batching import (
    DEFAULT_MAX_BATCH,
    BatchGroup,
    BatchPlan,
    batch_line,
    continuous_batching_schedule,
    occupancy,
    padding_line,
    padding_waste,
    per_request_steps,
    plan_batches,
    throughput,
)
from smart_research_agent.inference_optim.budget import (
    activation_bytes,
    cache_bytes_for,
    check_max_tokens,
    compression_ratio,
    level_line,
    level_table,
    max_tokens,
    per_token_cost,
    plan,
    weight_bytes,
)
from smart_research_agent.inference_optim.cache import (
    DEFAULT_CAPACITY_RATIO,
    KVCache,
    attend_projected,
    attention_work,
    cache_account_line,
    compare_with_recompute,
    decode_step,
    generate_with_cache,
    make_cache,
    position_of,
    prefill,
    step_delta_is_a_formula,
    tokens_per_second,
)
from smart_research_agent.inference_optim.errors import (
    ABSENT_FAMILY,
    ABSENT_FAMILY_REASON,
    FAMILY_OUTCOMES,
    AssemblyError,
    BudgetError,
    CacheError,
    NumericError,
    OptimError,
    ParameterError,
    QuantError,
    ShapeError,
)
from smart_research_agent.inference_optim.quantize import (
    BOUND_TOLERANCE,
    LayerError,
    bound_line,
    dequantize,
    error_bound,
    error_stats,
    logits_of,
    measure,
    pack_int4,
    packed_length,
    quantize_matrix,
    round_trip_line,
    unpack_int4,
)
from smart_research_agent.inference_optim.study import (
    BATCH_LENGTHS,
    BATCH_MAX,
    BUDGET_BYTES,
    BUDGET_TOKENS,
    DECODE_STEPS,
    FORMULA_TOKENS,
    PREFILL_TOKENS,
    BudgetRow,
    CacheStepTable,
    GranularityRow,
    QuantRow,
    batch_rows,
    budget_rows,
    cache_formula_rows,
    cache_step_table,
    granularity_rows,
    note_lines,
    quant_rows,
    skewed_matrix,
    study_lines,
)
from smart_research_agent.inference_optim.types import (
    ACCELERATE_LIBRARY,
    BITS_PER_BYTE,
    BIT_WIDTHS,
    BYTES_PER_ELEMENT,
    CACHE_TABLES,
    DECODE_TOKENS_PER_STEP,
    GRANULARITIES,
    GRANULARITY_CHANNEL,
    GRANULARITY_DESCRIPTIONS,
    GRANULARITY_TENSOR,
    INT4_PER_BYTE,
    INTEGER_BITS,
    INTEGER_LEVELS,
    LAYOUTS,
    LAYOUT_DESCRIPTIONS,
    LAYOUT_GROW,
    LAYOUT_PREALLOCATED,
    LEVELS,
    LEVEL_DESCRIPTIONS,
    LEVEL_FP16,
    LEVEL_FP32,
    LEVEL_INT4,
    LEVEL_INT8,
    OPTIM_BOUNDARIES,
    OPTIM_NOTES,
    OPTIM_NOTES_ORDER,
    OPTIM_PROPERTIES,
    PROPERTY_BUDGET_IS_EXACTLY_ACCOUNTED,
    PROPERTY_CACHED_EQUALS_RECOMPUTE,
    PROPERTY_CACHE_GROWTH_IS_A_FORMULA,
    PROPERTY_DEQUANT_ERROR_WITHIN_HALF_SCALE,
    PROPERTY_DESCRIPTIONS,
    PROPERTY_FAILURE,
    PROPERTY_INT4_PACK_IS_LOSSLESS,
    PROPERTY_PREFILL_MATCHES_FULL,
    PROPERTY_SCHEDULE_OCCUPANCY_IS_COUNTED,
    QUANT_LIBRARY,
    QUANT_LIBRARY_VERSION,
    SCHEMES,
    SCHEME_ABSMAX,
    SCHEME_DESCRIPTIONS,
    SCHEME_ZERO_POINT,
    STAGES,
    STAGE_DECODE,
    STAGE_DESCRIPTIONS,
    STAGE_PREFILL,
    TRANSFORMERS_VERSION,
    BudgetBreakdown,
    CacheLayer,
    CacheState,
    ErrorStats,
    QuantSpec,
    QuantizedMatrix,
    StepRow,
    cache_bytes,
    element_bytes,
    flatten,
    norm_or_raise,
)
from smart_research_agent.inference_optim.verify import (
    INT4_BOUNDARY,
    INT4_UNSIGNED_BOUNDARY,
    CrossCheck,
    PropertyOutcome,
    PropertyReport,
    cache_state_line,
    check_all,
    check_budget_is_exactly_accounted,
    check_cache_growth_is_a_formula,
    check_cached_equals_recompute,
    check_dequant_error_within_half_scale,
    check_int4_pack_is_lossless,
    check_prefill_matches_full,
    check_schedule_occupancy_is_counted,
    finite_or_raise,
    quant_summary,
)

__all__ = [
    "ABSENT_FAMILY",
    "ABSENT_FAMILY_REASON",
    "ACCELERATE_LIBRARY",
    "BATCH_LENGTHS",
    "BATCH_MAX",
    "BITS_PER_BYTE",
    "BIT_WIDTHS",
    "BOUND_TOLERANCE",
    "BUDGET_BYTES",
    "BUDGET_TOKENS",
    "BYTES_PER_ELEMENT",
    "CACHE_TABLES",
    "DECODE_STEPS",
    "DECODE_TOKENS_PER_STEP",
    "DEFAULT_CAPACITY_RATIO",
    "DEFAULT_MAX_BATCH",
    "FAMILY_OUTCOMES",
    "FORMULA_TOKENS",
    "GRANULARITIES",
    "GRANULARITY_CHANNEL",
    "GRANULARITY_DESCRIPTIONS",
    "GRANULARITY_TENSOR",
    "INT4_BOUNDARY",
    "INT4_PER_BYTE",
    "INT4_UNSIGNED_BOUNDARY",
    "INTEGER_BITS",
    "INTEGER_LEVELS",
    "LAYOUTS",
    "LAYOUT_DESCRIPTIONS",
    "LAYOUT_GROW",
    "LAYOUT_PREALLOCATED",
    "LEVELS",
    "LEVEL_DESCRIPTIONS",
    "LEVEL_FP16",
    "LEVEL_FP32",
    "LEVEL_INT4",
    "LEVEL_INT8",
    "OPTIM_BOUNDARIES",
    "OPTIM_NOTES",
    "OPTIM_NOTES_ORDER",
    "OPTIM_PROPERTIES",
    "PREFILL_TOKENS",
    "PROPERTY_BUDGET_IS_EXACTLY_ACCOUNTED",
    "PROPERTY_CACHED_EQUALS_RECOMPUTE",
    "PROPERTY_CACHE_GROWTH_IS_A_FORMULA",
    "PROPERTY_DEQUANT_ERROR_WITHIN_HALF_SCALE",
    "PROPERTY_DESCRIPTIONS",
    "PROPERTY_FAILURE",
    "PROPERTY_INT4_PACK_IS_LOSSLESS",
    "PROPERTY_PREFILL_MATCHES_FULL",
    "PROPERTY_SCHEDULE_OCCUPANCY_IS_COUNTED",
    "QUANT_LIBRARY",
    "QUANT_LIBRARY_VERSION",
    "SCHEMES",
    "SCHEME_ABSMAX",
    "SCHEME_DESCRIPTIONS",
    "SCHEME_ZERO_POINT",
    "STAGES",
    "STAGE_DECODE",
    "STAGE_DESCRIPTIONS",
    "STAGE_PREFILL",
    "TRANSFORMERS_VERSION",
    "AssemblyError",
    "BatchGroup",
    "BatchPlan",
    "BudgetBreakdown",
    "BudgetError",
    "BudgetRow",
    "CacheError",
    "CacheLayer",
    "CacheState",
    "CacheStepTable",
    "CrossCheck",
    "ErrorStats",
    "GranularityRow",
    "KVCache",
    "LayerError",
    "NumericError",
    "OptimError",
    "ParameterError",
    "PropertyOutcome",
    "PropertyReport",
    "QuantError",
    "QuantRow",
    "QuantSpec",
    "QuantizedMatrix",
    "ShapeError",
    "StepRow",
    "activation_bytes",
    "attend_projected",
    "attention_work",
    "batch_line",
    "batch_rows",
    "bound_line",
    "budget_rows",
    "cache_account_line",
    "cache_bytes",
    "cache_bytes_for",
    "cache_formula_rows",
    "cache_state_line",
    "cache_step_table",
    "check_all",
    "check_budget_is_exactly_accounted",
    "check_cache_growth_is_a_formula",
    "check_cached_equals_recompute",
    "check_dequant_error_within_half_scale",
    "check_int4_pack_is_lossless",
    "check_max_tokens",
    "check_prefill_matches_full",
    "check_schedule_occupancy_is_counted",
    "compare_with_recompute",
    "compression_ratio",
    "continuous_batching_schedule",
    "decode_step",
    "dequantize",
    "element_bytes",
    "error_bound",
    "error_stats",
    "finite_or_raise",
    "flatten",
    "generate_with_cache",
    "granularity_rows",
    "level_line",
    "level_table",
    "logits_of",
    "make_cache",
    "max_tokens",
    "measure",
    "norm_or_raise",
    "note_lines",
    "occupancy",
    "pack_int4",
    "packed_length",
    "padding_line",
    "padding_waste",
    "per_request_steps",
    "per_token_cost",
    "plan",
    "plan_batches",
    "position_of",
    "prefill",
    "quant_rows",
    "quant_summary",
    "quantize_matrix",
    "round_trip_line",
    "skewed_matrix",
    "step_delta_is_a_formula",
    "study_lines",
    "throughput",
    "tokens_per_second",
    "unpack_int4",
    "weight_bytes",
]
