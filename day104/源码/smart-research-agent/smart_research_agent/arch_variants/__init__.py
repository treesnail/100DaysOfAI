"""``arch_variants``：三个变体，只差三处（M7-D7 / day082）.

day081 把一条链**更新了 40 次**，并留下一句话：*块是唯一被复制的单位*。
今天回答的是它的下一问——**同一个块该往哪几处接**。三篇论文在这件事上只差三处：

```text
                 ┌── 自注意力掩码 ──┬── 第二路（交叉注意力）── 训练目标 ──┐
encoder_only     │ 全开（双向）      │ 无                       │ MLM
decoder_only     │ 因果（只看过去）   │ 无                       │ CLM
encoder_decoder  │ 全开 + 因果       │ 有（K/V 来自另一路）        │ 去噪
                 └─────────────────┴──────────────────────────┴──────────┘
```

## 一、今天最值钱的一句话

> **因果性不是读代码读出来的，是用扰动**量**出来的。**

“这个模型是因果的吗”通常靠看有没有 `causal=True` 来确认，而今天换一条路：

```text
扰动输入的第 j 个 token（乘 1.5），重新前向，看输出第 i 行变了多少
  被掩码挡掉的 (i, j)     Δ **恰好 0.0**（不是 1e-16：被掩码的打分从来没有被读过）
  允许的 (i, j)           Δ 明显大于 0
```

于是“因果”变成一张可以打印出来的 0/非 0 表，而它有三条判据：

```text
逐位       挡掉的格子必须**逐位**相等（`== 0.0`，本课不留容差）
双向       允许的格子里最小的那个读数必须 > 0（否则“把一切都挡住”也会通过）
反证       把掩码换错，同一张表立刻亮红（越界 6 格 / 缺失 6 格）
```

## 二、七个模块

```text
errors.py    五族失败：形状 / 参数 / **组装** / 数值 / 梯度
types.py     三个变体、两张口径表（掩码与训练目标）、六条性质、六条记录
masks.py     三张掩码与一次组合（全开/因果**转发** day073/075；填充是本课新增）
stacks.py    接线：按变体分配掩码、接上两条路、重放前向做反向、数参数
probe.py     实测依赖表（**这一课的判据**）：主流 / 编码器流 / 交叉那一路
verify.py    六条性质 + 整条变体的梯度校验（解析 vs 中心差分）
assembly.py  生成 PyTorch 脚本，并用 ast 把那段文本解析回结构
```

## 三、五条纪律

1. **不写新的注意力算术**：三个变体的每一层都是 day079/080 的
   ``self_attention`` / ``encoder_block`` / ``decoder_block``——
   ``decoder_only`` 用的就是 ``encoder_block``，只换一张掩码。
2. **掩码必须传给正确的流**：给错、漏传、反向没穿，三种失败**都不会报错**；
   因此第 1 条性质用 ``==`` 断言权重表，第 2~4 条用扰动实测。
3. **一次只改一个旋钮**：换变体时参数、样本与种子全部不变（连编码器与解码器的
   前馈都刻意同源），于是差别只可能来自“接法”。
4. **实测表与声明的掩码逐格比**：越界与缺失**两个方向**都量——
   只量前者时，一张“把一切都挡住”的掩码也会全绿。
5. **不适用 ≠ 通过**：``encoder_only`` 上没有“解码器只读过去”这条性质，
   报告里它记作 ``applicable=False``——否则“删掉这条检查”与“它通过了”长得一样。

## 四、踩到的两个真实的坑（都只表现为“梯度对不上”）

```text
① pre-LN 下注意力作用在 LN(x) 上   把算好的注意力账传进来时用了原始 x
                                   ⇒ 前向形状全对、读数全对，而 encoder_block_backward
                                   按“注意力在 LN(x) 上”分梯度 ⇒ 两边不是同一个函数
② decoder_block 要求因果自注意力   它与“显式填充掩码”不能同时给（day075 的 resolve_mask）
                                   ⇒ encoder_decoder + pads 必须**显式拒绝**，而不是静默忽略
```

两处都写进了代码注释与教程第 10 章——**它们在前面几课都出现过**，
而今天以“变体接线”的形式再出现一次。

## 五、与既有包的接缝

- **上游**：``encoder_decoder``（day079 的块与两种反向）、``transformer_core``
  （day075 的注意力与 ``full_mask``）、``math_foundations``（day073 的 ``causal_mask``
  与 LCG 随机数）、``transformer_stack``（day080 的逐层读数口径）；
- **脚下**：``config`` **没有**新增配置项——变体名、掩码名、层数、倍数都是
  “这一次调用或这一次对照的判据”，它们进的是函数参数（与 day073~081 同一条纪律）；
- **下游**：day083（可解释性与注意力可视化）要的正是今天留下的三张权重表
  （``block_weights`` / ``decoder_self_weights`` / ``cross_weights``）；
  day085（源码精读）会看到 Hugging Face 的 ``BertEncoder`` / ``GPT2Model``
  与今天的三个变体逐层对应；day086（与 HF 生态集成）会把生成的脚本换成真模型。
"""

from __future__ import annotations

from smart_research_agent.arch_variants.assembly import (
    ASSEMBLY_CLASSES,
    ASSEMBLY_CONFIG_KEYS,
    ASSEMBLY_MASKS,
    ASSEMBLY_MODULES,
    ASSEMBLY_REQUIREMENT,
    all_scripts,
    assembly_facts,
    class_names,
    config_matches_shape,
    facts_match_variant,
    function_names,
    module_names,
    parse_config,
    parse_script,
    script_requires_torch,
    variant_script,
)
from smart_research_agent.arch_variants.errors import (
    FAMILY_OUTCOMES,
    AssemblyError,
    GradientError,
    NumericError,
    ParameterError,
    ShapeError,
    VariantError,
)
from smart_research_agent.arch_variants.masks import (
    MASK_CAUSAL,
    MASK_DESCRIPTIONS,
    MASK_FULL,
    MASK_KINDS,
    Mask,
    combine_masks,
    dependency_from_floats,
    mask_allowed_counts,
    mask_allowed_pairs,
    mask_entropy_ceiling,
    mask_fits_sequence,
    mask_is_causal,
    mask_is_shape_only,
    mask_leaks,
    mask_of,
    mask_summary,
    mask_to_floats,
    padding_mask,
    shape_of_mask,
)
from smart_research_agent.arch_variants.probe import (
    DEFAULT_PERTURBATION,
    DEFAULT_PROBE_TOLERANCE,
    STREAM_ENCODER,
    STREAM_KINDS,
    STREAM_MAIN,
    STREAM_SOURCE,
    DependencyReport,
    causal_verdict,
    changed_rows,
    contract_gap,
    cross_dependency,
    dependency_matrix,
    dependency_of,
    encoder_dependency,
    is_rectangular,
    largest_change,
    perturb_row,
    report_matches,
    row_change_profile,
)
from smart_research_agent.arch_variants.stacks import (
    ATTENTION_SEED_STRIDE,
    BLOCK_SEED_STRIDE,
    CROSS_SEED_STRIDE,
    DEFAULT_SOURCE_SEED,
    VariantGradients,
    block_shape_of,
    census_of,
    check_mask_fits,
    encoder_stream_is_causal,
    loss_gradient,
    make_block_stack,
    make_cross_parameters,
    make_decoder_stack,
    make_variant_parameters,
    make_variant_shape,
    mask_for_streams,
    params_mask_of,
    sample_matrix,
    stream_is_causal,
    variant_backward,
    variant_forward,
    variant_loss,
    visible_positions,
)
from smart_research_agent.arch_variants.study import (
    DEFAULT_INPUT_SEED,
    DEFAULT_SOURCE_SEED_SAMPLE,
    CensusRow,
    CensusStudy,
    LeakRow,
    LeakStudy,
    PrefixRow,
    PrefixStudy,
    census_study,
    entropy_ceiling_table,
    keystone_check,
    leak_study,
    mask_catalogue,
    mask_pair_ratio,
    objectives_table,
    prefix_study,
    study_summary,
    variant_descriptions,
    variant_parameter_total,
)
from smart_research_agent.arch_variants.types import (
    ARCH_NOTES,
    ARCH_PROPERTIES,
    DEFAULT_FFN_RATIO,
    DEFAULT_HIDDEN,
    DEFAULT_INIT_SCALE,
    DEFAULT_LAYERS,
    DEFAULT_SOURCES,
    DEFAULT_TOKENS,
    PARAMETER_BLOCK_ORDER,
    PROPERTY_CROSS_SPANS_ALL_SOURCES,
    PROPERTY_DECODER_READS_ONLY_PAST,
    PROPERTY_DESCRIPTIONS,
    PROPERTY_DETERMINISTIC,
    PROPERTY_ENCODER_READS_EVERY_POSITION,
    PROPERTY_MASK_ZEROES_ARE_EXACT,
    PROPERTY_STACK_PRESERVES_SHAPE,
    VARIANTS,
    VARIANT_DESCRIPTIONS,
    VARIANT_DECODER_ONLY,
    VARIANT_ENCODER_DECODER,
    VARIANT_ENCODER_ONLY,
    VARIANT_EXAMPLES,
    VARIANT_MASKS,
    VARIANT_OBJECTIVES,
    BlockStackParameters,
    DecoderStackParameters,
    VariantCensus,
    VariantForward,
    VariantParameters,
    VariantShape,
    frobenius,
    max_absolute,
    ones_vector,
    relative_matrix_error,
    validate_variant,
    zeros_vector,
)
from smart_research_agent.arch_variants.verify import (
    GRADIENT_TOLERANCE,
    NUMERICAL_STEP,
    GradientOutcome,
    GradientReport,
    PropertyOutcome,
    PropertyReport,
    check_all,
    check_cross_spans_all_sources,
    check_deterministic,
    check_properties,
    check_reads_every_position,
    check_reads_only_past,
    check_shape_preserved,
    check_variant_gradients,
    check_weights_are_zero_where_masked,
    numerical_stream_gradient,
)

__all__ = [
    "ARCH_NOTES",
    "ARCH_PROPERTIES",
    "ASSEMBLY_CLASSES",
    "ASSEMBLY_CONFIG_KEYS",
    "ASSEMBLY_MASKS",
    "ASSEMBLY_MODULES",
    "ASSEMBLY_REQUIREMENT",
    "ATTENTION_SEED_STRIDE",
    "BLOCK_SEED_STRIDE",
    "CROSS_SEED_STRIDE",
    "DEFAULT_FFN_RATIO",
    "DEFAULT_HIDDEN",
    "DEFAULT_INIT_SCALE",
    "DEFAULT_INPUT_SEED",
    "DEFAULT_LAYERS",
    "DEFAULT_PERTURBATION",
    "DEFAULT_PROBE_TOLERANCE",
    "DEFAULT_SOURCE_SEED",
    "DEFAULT_SOURCE_SEED_SAMPLE",
    "DEFAULT_SOURCES",
    "DEFAULT_TOKENS",
    "FAMILY_OUTCOMES",
    "GRADIENT_TOLERANCE",
    "MASK_CAUSAL",
    "MASK_DESCRIPTIONS",
    "MASK_FULL",
    "MASK_KINDS",
    "NUMERICAL_STEP",
    "PARAMETER_BLOCK_ORDER",
    "PROPERTY_CROSS_SPANS_ALL_SOURCES",
    "PROPERTY_DECODER_READS_ONLY_PAST",
    "PROPERTY_DESCRIPTIONS",
    "PROPERTY_DETERMINISTIC",
    "PROPERTY_ENCODER_READS_EVERY_POSITION",
    "PROPERTY_MASK_ZEROES_ARE_EXACT",
    "PROPERTY_STACK_PRESERVES_SHAPE",
    "STREAM_ENCODER",
    "STREAM_KINDS",
    "STREAM_MAIN",
    "STREAM_SOURCE",
    "VARIANTS",
    "VARIANT_DESCRIPTIONS",
    "VARIANT_DECODER_ONLY",
    "VARIANT_ENCODER_DECODER",
    "VARIANT_ENCODER_ONLY",
    "VARIANT_EXAMPLES",
    "VARIANT_MASKS",
    "VARIANT_OBJECTIVES",
    "AssemblyError",
    "BlockStackParameters",
    "CensusRow",
    "CensusStudy",
    "DecoderStackParameters",
    "DependencyReport",
    "GradientError",
    "GradientOutcome",
    "GradientReport",
    "LeakRow",
    "LeakStudy",
    "Mask",
    "NumericError",
    "ParameterError",
    "PrefixRow",
    "PrefixStudy",
    "PropertyOutcome",
    "PropertyReport",
    "ShapeError",
    "VariantCensus",
    "VariantError",
    "VariantForward",
    "VariantGradients",
    "VariantParameters",
    "VariantShape",
    "all_scripts",
    "assembly_facts",
    "block_shape_of",
    "causal_verdict",
    "census_of",
    "census_study",
    "changed_rows",
    "check_all",
    "check_cross_spans_all_sources",
    "check_deterministic",
    "check_mask_fits",
    "check_properties",
    "check_reads_every_position",
    "check_reads_only_past",
    "check_shape_preserved",
    "check_variant_gradients",
    "check_weights_are_zero_where_masked",
    "class_names",
    "combine_masks",
    "config_matches_shape",
    "contract_gap",
    "cross_dependency",
    "dependency_from_floats",
    "dependency_matrix",
    "dependency_of",
    "encoder_dependency",
    "encoder_stream_is_causal",
    "entropy_ceiling_table",
    "facts_match_variant",
    "frobenius",
    "function_names",
    "is_rectangular",
    "keystone_check",
    "largest_change",
    "leak_study",
    "loss_gradient",
    "make_block_stack",
    "make_cross_parameters",
    "make_decoder_stack",
    "make_variant_parameters",
    "make_variant_shape",
    "mask_allowed_counts",
    "mask_allowed_pairs",
    "mask_catalogue",
    "mask_entropy_ceiling",
    "mask_fits_sequence",
    "mask_for_streams",
    "mask_is_causal",
    "mask_is_shape_only",
    "mask_leaks",
    "mask_of",
    "mask_pair_ratio",
    "mask_summary",
    "mask_to_floats",
    "max_absolute",
    "module_names",
    "numerical_stream_gradient",
    "objectives_table",
    "ones_vector",
    "padding_mask",
    "params_mask_of",
    "parse_config",
    "parse_script",
    "perturb_row",
    "prefix_study",
    "relative_matrix_error",
    "report_matches",
    "row_change_profile",
    "sample_matrix",
    "script_requires_torch",
    "shape_of_mask",
    "stream_is_causal",
    "study_summary",
    "validate_variant",
    "variant_backward",
    "variant_descriptions",
    "variant_forward",
    "variant_loss",
    "variant_parameter_total",
    "variant_script",
    "visible_positions",
    "zeros_vector",
]
