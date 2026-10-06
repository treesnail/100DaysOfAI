"""``explainability``：把注意力读出来、画出来、并给出可被断言的读数（M7-D8 / day083）.

day082 留下了三张权重表；今天回答的是它们的下一问——**这些表在说什么**。

## 一、今天最值钱的一句话

> **熵要跟天花板比。**

```text
一行的熵 ≤ ln(这一行能看到的位置数)        ← Jensen 不等式，不是经验规律
全开掩码（BERT）  第 0 行的天花板是 ln n（n = 4 时 1.386294）
因果掩码（GPT）   第 0 行的天花板是 ln 1 = 0，全表平均 (1/n)Σln(i+1) = 0.794513
```

于是"GPT 的注意力比 BERT 尖"这句话里，**有一半是掩码造成的**。
本包因此把三个数一起给：

```text
entropy              平均每行熵
ceiling              平均每行天花板
normalized_entropy   entropy / ceiling ∈ [0, 1]     ← **跨层、跨变体可比的那把尺子**
```

## 二、第二条：10 级是有损的，因此"画出来再读回去"是一条判据

```text
等级 0..9 = " .:-=+*#%@"，第 k 级的区间是 [k/9, (k+1)/9)
渲染 → 解析 ⇒ 必须**逐级**相同（不是"看起来差不多"）
```

它挡住的是"把 1 与 0 画得一样"这类实现错误——那种错误**只会让人看着图得出错的结论**，
因此它值得一条能失败的断言（第 4 条性质）。

## 三、八个模块

```text
errors.py   四族失败（**缺席的那一族是 GradientError**：本课不改任何算术）
types.py    AttentionRecord / HeadProfile / Attribution、六条性质、两条纪律
extract.py  两条读法：模型**真实**权重（self_records）与多头读法（head_records）
render.py   文本热力图：10 级、图例、以及"画回去"（parse_heatmap）
analyze.py  六个读数：熵/天花板/峰值/支撑/Frobenius/对角质量 + 头间余弦 + 偏移 + 列和
rollout.py  Â = α·A + (1−α)·I，R = Â_L ⋯ Â_1（那条 (1−α)·I 就是残差）
verify.py   六条性质（每一条都有一条"它会亮红"的反证）
study.py    五张表：逐层熵 / 头间冗余 / 偏移质量 / 滚动集中度 / 天花板对照
```

## 四、四条纪律

1. **两条读法不许混**：`self_records` 是模型真实看到的（头数 1），
   `head_records` 是同一组投影在多头划分下的读法——多头会改变投影划分与缩放
   （`1/√d_head` 而不是 `1/√d_k`）。它们之间只有一条接缝可以比：
   `head_records(heads=1)` 必须**逐位**等于 `self_records`。
2. **跨掩码的比较必须先归一化**：熵要跟天花板比，否则比的是掩码。
3. **读数要么有反证，要么不算判据**：每一条性质都在测试里有一条"它会失败"。
4. **画出来的东西必须能被读回去**：文本热力图是"可 review 的图"，
   而"渲染 → 解析逐级相同"让它成为一条断言。

## 五、一条被发现写松了的判据（第二版修的）

第一版的 `check_reads_every_position` 式的比法在 day082 上已经被修过；
本课踩到的是它的**镜像**：`head_redundancy` 若比较两张**掩码不同**的表，
比的是"能看到多少"而不是"看了哪里"。修法是在入口要求"形状与掩码都相同"：

```python
if record.mask != first.mask:
    raise AssemblyError("不同掩码的两张表之间比余弦，比的是'能看到多少'")
```

## 六、与既有包的接缝

- **上游**：``arch_variants``（day082 的三张权重表与两种掩码）、
  ``multi_head``（day076 的多头前向）、``encoder_decoder``（day079 的层与块）；
- **脚下**：``config`` **没有**新增配置项——头数、阈值、α、级数都是函数参数；
- **下游**：day085（源码精读）会看到 Hugging Face 的 ``BertAttention`` / ``GPT2Attention``
  与今天的记录一一对应；day088（项目底层原理串联）会用今天这五张表写"原理 → 应用"的分享提纲。
"""

from __future__ import annotations

from smart_research_agent.explainability.analyze import (
    DEFAULT_DEAD_THRESHOLD,
    attribution_of,
    cosine_similarity,
    dead_heads,
    diagonal_mass,
    entropy_gap,
    focus_verdict,
    frobenius_of,
    head_redundancy,
    layer_profiles,
    offset_mass,
    peak_of,
    profile_of,
    profiles_of,
    shape_of_record,
    summarise,
    support_of,
)
from smart_research_agent.explainability.errors import (
    FAMILY_OUTCOMES,
    AssemblyError,
    ExplainError,
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.explainability.extract import (
    LABEL_CROSS,
    LABEL_DECODER,
    LABEL_ENCODER,
    LABEL_LAYER,
    aggregate_heads,
    cross_records,
    head_records,
    layer_count,
    layer_inputs,
    record_of,
    records_of_layer,
    self_records,
    self_stream_records,
    single_head_matches,
    summary_lines,
    validate_heads,
    variant_of,
)
from smart_research_agent.explainability.render import (
    LEGEND,
    char_level,
    explanation_of,
    heatmap_block,
    legend_line,
    level_char,
    level_of,
    normalized_levels,
    parse_heatmap,
    profile_bar,
    profile_block,
    render_profiles,
    render_records,
    round_trip,
)
from smart_research_agent.explainability.rollout import (
    ROLLOUT_LABEL,
    aggregate_by_layer,
    identity_when_alpha_zero,
    mask_respected_after_rollout,
    preserves_causal_zeroes,
    rollout_focus,
    rollout_record,
    rollout_weights,
    row_sums,
)
from smart_research_agent.explainability.study import (
    DEFAULT_OFFSETS,
    INPUT_SEED,
    PARAMETER_SEED,
    SOURCE_SEED,
    CeilingStudy,
    LayerEntropyStudy,
    OffsetStudy,
    RedundancyStudy,
    RolloutStudy,
    all_ceilings,
    all_layer_records,
    ceiling_study,
    default_shape,
    how_to_read,
    layer_entropy_study,
    offset_study,
    redundancy_study,
    rollout_study,
    study_inputs,
    study_params,
    variant_labels,
)
from smart_research_agent.explainability.types import (
    DEFAULT_ALPHA,
    DEFAULT_HEADS,
    DEFAULT_THRESHOLD,
    EXPLAIN_NOTES,
    EXPLAIN_PROPERTIES,
    LEVEL_COUNT,
    LEVELS,
    PROPERTY_DESCRIPTIONS,
    PROPERTY_DETERMINISTIC,
    PROPERTY_ENTROPY_WITHIN_CEILING,
    PROPERTY_HEATMAP_ROUND_TRIP,
    PROPERTY_MASKED_ENTRIES_ARE_EXACT_ZERO,
    PROPERTY_ROLLOUT_IS_STOCHASTIC,
    PROPERTY_ROWS_ARE_DISTRIBUTIONS,
    STREAM_CROSS,
    STREAM_DESCRIPTIONS,
    STREAM_KINDS,
    STREAM_SELF,
    AttentionRecord,
    Attribution,
    HeadProfile,
    allowed_counts,
    causal_mask,
    ceiling_of_mask,
    entropy_ceiling,
    full_mask,
    mean_of,
    row_entropies,
    row_entropy,
)
from smart_research_agent.explainability.verify import (
    ENTROPY_TOLERANCE,
    ROLLOUT_TOLERANCE,
    PropertyOutcome,
    PropertyReport,
    check_all,
    check_deterministic,
    check_entropy_within_ceiling,
    check_heatmap_round_trip,
    check_masked_entries_are_exact_zero,
    check_properties,
    check_rollout_is_stochastic,
    check_rows_are_distributions,
)

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_DEAD_THRESHOLD",
    "DEFAULT_HEADS",
    "DEFAULT_OFFSETS",
    "DEFAULT_THRESHOLD",
    "ENTROPY_TOLERANCE",
    "EXPLAIN_NOTES",
    "EXPLAIN_PROPERTIES",
    "FAMILY_OUTCOMES",
    "INPUT_SEED",
    "LABEL_CROSS",
    "LABEL_DECODER",
    "LABEL_ENCODER",
    "LABEL_LAYER",
    "LEGEND",
    "LEVELS",
    "LEVEL_COUNT",
    "PARAMETER_SEED",
    "PROPERTY_DESCRIPTIONS",
    "PROPERTY_DETERMINISTIC",
    "PROPERTY_ENTROPY_WITHIN_CEILING",
    "PROPERTY_HEATMAP_ROUND_TRIP",
    "PROPERTY_MASKED_ENTRIES_ARE_EXACT_ZERO",
    "PROPERTY_ROLLOUT_IS_STOCHASTIC",
    "PROPERTY_ROWS_ARE_DISTRIBUTIONS",
    "ROLLOUT_LABEL",
    "ROLLOUT_TOLERANCE",
    "SOURCE_SEED",
    "STREAM_CROSS",
    "STREAM_DESCRIPTIONS",
    "STREAM_KINDS",
    "STREAM_SELF",
    "AssemblyError",
    "AttentionRecord",
    "Attribution",
    "CeilingStudy",
    "ExplainError",
    "HeadProfile",
    "LayerEntropyStudy",
    "NumericError",
    "OffsetStudy",
    "ParameterError",
    "PropertyOutcome",
    "PropertyReport",
    "RedundancyStudy",
    "RolloutStudy",
    "ShapeError",
    "aggregate_by_layer",
    "aggregate_heads",
    "all_ceilings",
    "all_layer_records",
    "allowed_counts",
    "attribution_of",
    "causal_mask",
    "ceiling_of_mask",
    "ceiling_study",
    "char_level",
    "check_all",
    "check_deterministic",
    "check_entropy_within_ceiling",
    "check_heatmap_round_trip",
    "check_masked_entries_are_exact_zero",
    "check_properties",
    "check_rollout_is_stochastic",
    "check_rows_are_distributions",
    "cosine_similarity",
    "cross_records",
    "dead_heads",
    "default_shape",
    "diagonal_mass",
    "entropy_ceiling",
    "entropy_gap",
    "explanation_of",
    "focus_verdict",
    "frobenius_of",
    "full_mask",
    "head_records",
    "head_redundancy",
    "heatmap_block",
    "how_to_read",
    "identity_when_alpha_zero",
    "layer_count",
    "layer_entropy_study",
    "layer_inputs",
    "layer_profiles",
    "legend_line",
    "level_char",
    "level_of",
    "mask_respected_after_rollout",
    "mean_of",
    "normalized_levels",
    "offset_mass",
    "offset_study",
    "parse_heatmap",
    "peak_of",
    "preserves_causal_zeroes",
    "profile_bar",
    "profile_block",
    "profile_of",
    "profiles_of",
    "record_of",
    "records_of_layer",
    "redundancy_study",
    "render_profiles",
    "render_records",
    "rollout_focus",
    "rollout_record",
    "rollout_study",
    "rollout_weights",
    "round_trip",
    "row_entropies",
    "row_entropy",
    "row_sums",
    "self_records",
    "self_stream_records",
    "shape_of_record",
    "single_head_matches",
    "study_inputs",
    "study_params",
    "summarise",
    "summary_lines",
    "support_of",
    "validate_heads",
    "variant_labels",
    "variant_of",
]
