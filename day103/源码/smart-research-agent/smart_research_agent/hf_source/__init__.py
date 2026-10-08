"""``hf_source``：Transformer 源码精读 —— 把 Hugging Face 的那几个类读成可断言的行为（M7-D9 / day085）.

day078 ~ day083 把"一个块、一条链、三种接法、五张表"都写完了；今天第一次去读
**别人写的**那几行。阅读对象是两个文件的推理路径：

```text
src/transformers/models/gpt2/modeling_gpt2.py
src/transformers/models/bert/modeling_bert.py
```

## 一、今天最值钱的一句话

> **"分头"发生在投影内部。**

day079 与 day083 都记下过同一条**接口边界**——"一个块只支持一种掩码"：
day079 的 `decoder_block` 要求自注意力是因果的，day082 的 `resolve_mask`
拒绝"因果 + 显式掩码"同时给。Hugging Face 的注意力并不接受这条边界，
而它绕开的办法只有两个形状重排：

```python
qkv   = c_attn(hidden)                          # (n, 3·hidden)
query, key, value = qkv.split(hidden, dim=-1)
query = query.view(n, heads, head_dim).transpose(0, 1)   # ← 就这里
```

于是同一个 `GPT2Attention` **既能跑 heads=12 又能跑因果掩码**——
不是因为它换了一个块类型，而是因为分头只是一次 `view` + `transpose`。
本包把这件事写成第一条性质：`split_heads` → `merge_heads` 必须**逐位**还原。

## 二、四个默认值（都是配置文件里的事实，不是"一般认为"）

```text
bert    layer_norm_eps = 1e-12    hidden_act = "gelu"        LN 在子层**之后**（post）
gpt2    layer_norm_epsilon = 1e-5 activation_function = "gelu_new"
                                                          LN 在子层**之前**（pre）
t5      layer_norm_epsilon = 1e-6  （一并记下，作为第三个数）
```

- `eps` 那一条接上 day079 第 3.2 节：`variance = σ²/(σ²+eps)`，因此
  **eps 越小，`x̂` 的方差越接近 1**——三个默认值给出三个不同的"离 1 多远"；
- 激活那一条接上 day079 的 `gelu`：`gelu_new` 是它的 tanh 近似，
  两式在 `x = 0` 处同为 `0.5`，而两侧差一个可量出来的量；
- 摆放那一条接上 day079 的 pre/post 对照表：**"norm 放在哪两个 Module 之间"
  就是那张表在真实源码里的样子**。

## 三、八个模块

```text
errors.py  五族失败（**缺席的仍是 GradientError**，理由换了一条：源码精读读的是推理路径）
types.py   两个画像 / 十个阶段 / 五种策略 / 七条性质 / 十二条源码阅读笔记 / 五条边界
vocab.py   词表与嵌入：同一条公式，两个模型的**第一个分岔**（BERT 多一次 LN）
attention.py 融合投影、投影内部分头、加性掩码（**-1e9 与 -inf 在这里逐位等价**）
blocks.py  pre/post 两个块 + gelu / gelu_new 两个激活
generation.py 四个 warper 与两种搜索（贪心 / 采样 / top-k / top-p / beam）
verify.py  七条性质与两条跨天对账（逐位 vs 容差，两类比法写清楚）
study.py   七张表（过滤器 / top-k / top-p / 四种策略 / beam / 激活 / 三个 eps）
```

## 四、四条纪律

1. **只承诺行为，不承诺行号**：本包复现的是计算语义（"`ln_1` 在自注意力之前"），
   而不是某个符号在第几行——源码会随版本重构，结构事实不会。
2. **逐位与容差必须分开写**：同一份算术里的两个量用 `==`；
   两份独立实现之间的量用 `1e-12`，并把 `max_gap` 印出来。
3. **"不适用" ≠ "通过"**：`applicable=False` 时 `passed` 也必须是 `False`
   （构造期就拒绝），否则"删掉检查"与"全通过"在报告里长得一样。
4. **不可能失败的读数也要印**：`top_p` 的"至少留一个"永远成立，
   因此它不作为判据，而是把**保留个数的最小值**如实印出来。

## 五、与既有包的接缝

- **上游**：`math_foundations`（matmul / softmax / LCG）、`transformer_core`（`AttentionParams`）、
  `encoder_decoder`（`BlockParameters` / `layer_norm` / `add_residual`）、
  `multi_head`（`head_scale` 与 `multi_head_attention`，跨天对账的另一端）；
- **脚下**：`config` **没有**新增配置项——画像、头数、策略、种子都是函数参数；
- **下游**：day086（与 HF 生态集成）会把这些行为换成**真模型**的读数；
  day087（高效推理与量化）会从"推理路径"这一条边界上接下去（KV Cache 与量化都只作用于推理）；
  day088（项目底层原理串联）用今天这七张表写"原理 → 应用"的提纲。
"""

from __future__ import annotations

from smart_research_agent.hf_source.attention import (
    BIAS_FLOOR,
    SINGLE_HEAD_TOLERANCE,
    HfAttentionForward,
    add_bias,
    additive_bias_of,
    causal_bias,
    check_score_magnitude,
    checked_mask,
    checked_scores,
    fused_weight,
    hf_attention,
    hf_scale,
    merge_heads,
    project,
    resolve_heads,
    row_softmax,
    split_fused,
    split_heads,
)
from smart_research_agent.hf_source.blocks import (
    BLOCK_TOLERANCE,
    DAY079_ACTIVATIONS,
    FFN_RATIO,
    GELU_NEW_CUBIC,
    GELU_NEW_INNER,
    BlockForward,
    activate_rows,
    activation_of,
    bert_layer,
    block_of,
    gelu_exact,
    gelu_new,
    gpt2_block,
    gpt2_mlp,
    make_hf_block,
    max_abs_gap,
    source_shape_of,
)
from smart_research_agent.hf_source.errors import (
    FAMILY_OUTCOMES,
    AssemblyError,
    GenerationError,
    NumericError,
    ParameterError,
    ShapeError,
    SourceError,
)
from smart_research_agent.hf_source.generation import (
    FILTERED_LOGIT,
    MIN_TOKENS_TO_KEEP,
    apply_repetition_penalty,
    apply_temperature,
    beam_search,
    checked_logits,
    distribution_entropy,
    generate,
    greedy_index,
    log_softmax_of,
    sample_index,
    select_token,
    softmax_of,
    strategy_of,
    top_k_filter,
    top_p_filter,
    transform_logits,
    validate_settings,
)
from smart_research_agent.hf_source.study import (
    ACTIVATION_POINTS,
    BEAM_CASES,
    DEFAULT_HIDDEN,
    DEFAULT_TOKENS,
    FILTER_CASES,
    FILTER_LOGITS,
    SAMPLING_CASES,
    TOY_SHARPNESS,
    TOY_VOCAB,
    ActivationRow,
    BeamRow,
    BlockRow,
    FilterRow,
    LayerNormRow,
    NucleusRow,
    SamplingRow,
    TopKRow,
    activation_study,
    beam_study,
    block_study,
    filter_study,
    layer_norm_study,
    nucleus_study,
    sampling_study,
    study_lines,
    top_k_study,
    toy_logits,
)
from smart_research_agent.hf_source.types import (
    ACTIVATIONS,
    ACTIVATION_BERT,
    ACTIVATION_DESCRIPTIONS,
    ACTIVATION_GPT2,
    ACTIVATION_RELU,
    ATTENTION_STAGES,
    ATTENTION_STAGE_DESCRIPTIONS,
    ATTENTION_STAGE_SHAPES,
    GENERATION_STRATEGIES,
    GENERATION_STRATEGY_DESCRIPTIONS,
    LN_EPS_BERT,
    LN_EPS_DEFAULTS,
    LN_EPS_GPT2,
    LN_EPS_T5,
    NORM_PLACEMENTS,
    NORM_PLACEMENT_DESCRIPTIONS,
    NORM_POST,
    NORM_PRE,
    PROFILES,
    PROPERTY_CAUSAL_PREFIX_IS_STABLE,
    PROPERTY_DESCRIPTIONS,
    PROPERTY_FAILURE,
    PROPERTY_FUSED_MATCHES_SEPARATE,
    PROPERTY_MASKED_ENTRIES_ARE_EXACT_ZERO,
    PROPERTY_PRE_NORM_BLOCK_MATCHES_DAY079,
    PROPERTY_ROWS_ARE_DISTRIBUTIONS,
    PROPERTY_SINGLE_HEAD_MATCHES_MULTI_HEAD,
    PROPERTY_SPLIT_MERGE_ROUND_TRIP,
    SOURCE_BOUNDARIES,
    SOURCE_FILES,
    SOURCE_LIBRARY,
    SOURCE_NOTES,
    SOURCE_NOTES_ORDER,
    SOURCE_PROPERTIES,
    SOURCE_VERSION,
    SOURCE_VERSION_SAMPLE,
    STAGE_BIAS,
    STAGE_DROP,
    STAGE_MERGE,
    STAGE_MIX,
    STAGE_OUT,
    STAGE_QKV,
    STAGE_SCALE,
    STAGE_SCORE,
    STAGE_SOFTMAX,
    STAGE_SPLIT,
    STRATEGY_BEAM,
    STRATEGY_GREEDY,
    STRATEGY_SAMPLE,
    STRATEGY_TOP_K,
    STRATEGY_TOP_P,
    WARPER_ORDER,
    WARPER_ORDER_NOTE,
    GenerationResult,
    GenerationSettings,
    GenerationStep,
    ModelProfile,
    SourceShape,
    profile_of,
)
from smart_research_agent.hf_source.verify import (
    PERTURBATION,
    ROW_SUM_TOLERANCE,
    CrossCheck,
    PropertyOutcome,
    PropertyReport,
    check_all,
    check_causal_prefix_is_stable,
    check_fused_matches_separate,
    check_masked_entries_are_exact_zero,
    check_pre_norm_block_matches_day079,
    check_rows_are_distributions,
    check_single_head_matches_multi_head,
    check_split_merge_round_trip,
    default_inputs,
    scale_agreement_line,
)
from smart_research_agent.hf_source.vocab import (
    BERT_MAX_POSITION_EMBEDDINGS,
    DEFAULT_TOKEN_TYPE,
    GPT2_N_POSITIONS,
    EmbeddingTables,
    bert_embed,
    default_token_types,
    embedding_note,
    embedding_of,
    gather_rows,
    gpt2_embed,
    position_ids,
)

__all__ = [
    "ACTIVATIONS",
    "ACTIVATION_BERT",
    "ACTIVATION_DESCRIPTIONS",
    "ACTIVATION_GPT2",
    "ACTIVATION_POINTS",
    "ACTIVATION_RELU",
    "ATTENTION_STAGES",
    "ATTENTION_STAGE_DESCRIPTIONS",
    "ATTENTION_STAGE_SHAPES",
    "BEAM_CASES",
    "BERT_MAX_POSITION_EMBEDDINGS",
    "BIAS_FLOOR",
    "BLOCK_TOLERANCE",
    "DEFAULT_HIDDEN",
    "DEFAULT_TOKENS",
    "DEFAULT_TOKEN_TYPE",
    "DAY079_ACTIVATIONS",
    "FFN_RATIO",
    "FILTERED_LOGIT",
    "FILTER_CASES",
    "FILTER_LOGITS",
    "FAMILY_OUTCOMES",
    "GELU_NEW_CUBIC",
    "GELU_NEW_INNER",
    "GENERATION_STRATEGIES",
    "GENERATION_STRATEGY_DESCRIPTIONS",
    "GPT2_N_POSITIONS",
    "LN_EPS_BERT",
    "LN_EPS_DEFAULTS",
    "LN_EPS_GPT2",
    "LN_EPS_T5",
    "MIN_TOKENS_TO_KEEP",
    "NORM_PLACEMENTS",
    "NORM_PLACEMENT_DESCRIPTIONS",
    "NORM_POST",
    "NORM_PRE",
    "PERTURBATION",
    "PROFILES",
    "PROPERTY_CAUSAL_PREFIX_IS_STABLE",
    "PROPERTY_DESCRIPTIONS",
    "PROPERTY_FAILURE",
    "PROPERTY_FUSED_MATCHES_SEPARATE",
    "PROPERTY_MASKED_ENTRIES_ARE_EXACT_ZERO",
    "PROPERTY_PRE_NORM_BLOCK_MATCHES_DAY079",
    "PROPERTY_ROWS_ARE_DISTRIBUTIONS",
    "PROPERTY_SINGLE_HEAD_MATCHES_MULTI_HEAD",
    "PROPERTY_SPLIT_MERGE_ROUND_TRIP",
    "ROW_SUM_TOLERANCE",
    "SAMPLING_CASES",
    "SINGLE_HEAD_TOLERANCE",
    "SOURCE_BOUNDARIES",
    "SOURCE_FILES",
    "SOURCE_LIBRARY",
    "SOURCE_NOTES",
    "SOURCE_NOTES_ORDER",
    "SOURCE_PROPERTIES",
    "SOURCE_VERSION",
    "SOURCE_VERSION_SAMPLE",
    "STAGE_BIAS",
    "STAGE_DROP",
    "STAGE_MERGE",
    "STAGE_MIX",
    "STAGE_OUT",
    "STAGE_QKV",
    "STAGE_SCALE",
    "STAGE_SCORE",
    "STAGE_SOFTMAX",
    "STAGE_SPLIT",
    "STRATEGY_BEAM",
    "STRATEGY_GREEDY",
    "STRATEGY_SAMPLE",
    "STRATEGY_TOP_K",
    "STRATEGY_TOP_P",
    "TOY_SHARPNESS",
    "TOY_VOCAB",
    "WARPER_ORDER",
    "WARPER_ORDER_NOTE",
    "ActivationRow",
    "AssemblyError",
    "BeamRow",
    "BlockForward",
    "BlockRow",
    "CrossCheck",
    "EmbeddingTables",
    "FilterRow",
    "GenerationError",
    "GenerationResult",
    "GenerationSettings",
    "GenerationStep",
    "HfAttentionForward",
    "LayerNormRow",
    "ModelProfile",
    "NucleusRow",
    "NumericError",
    "ParameterError",
    "PropertyOutcome",
    "PropertyReport",
    "SamplingRow",
    "ShapeError",
    "SourceError",
    "SourceShape",
    "TopKRow",
    "activate_rows",
    "activation_of",
    "activation_study",
    "add_bias",
    "additive_bias_of",
    "apply_repetition_penalty",
    "apply_temperature",
    "beam_search",
    "beam_study",
    "bert_embed",
    "bert_layer",
    "block_of",
    "block_study",
    "causal_bias",
    "check_all",
    "check_causal_prefix_is_stable",
    "check_fused_matches_separate",
    "check_masked_entries_are_exact_zero",
    "check_pre_norm_block_matches_day079",
    "check_rows_are_distributions",
    "check_score_magnitude",
    "check_single_head_matches_multi_head",
    "check_split_merge_round_trip",
    "checked_logits",
    "checked_mask",
    "checked_scores",
    "default_inputs",
    "default_token_types",
    "distribution_entropy",
    "embedding_note",
    "embedding_of",
    "filter_study",
    "fused_weight",
    "gather_rows",
    "gelu_exact",
    "gelu_new",
    "generate",
    "gpt2_block",
    "gpt2_embed",
    "gpt2_mlp",
    "greedy_index",
    "hf_attention",
    "hf_scale",
    "layer_norm_study",
    "log_softmax_of",
    "make_hf_block",
    "max_abs_gap",
    "merge_heads",
    "nucleus_study",
    "position_ids",
    "profile_of",
    "project",
    "resolve_heads",
    "row_softmax",
    "sample_index",
    "sampling_study",
    "scale_agreement_line",
    "select_token",
    "softmax_of",
    "source_shape_of",
    "split_fused",
    "split_heads",
    "strategy_of",
    "study_lines",
    "top_k_filter",
    "top_k_study",
    "top_p_filter",
    "toy_logits",
    "transform_logits",
    "validate_settings",
]
