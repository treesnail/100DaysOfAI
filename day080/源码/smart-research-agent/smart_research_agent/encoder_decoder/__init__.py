"""``encoder_decoder``：把三个子层拼成一个块（M7-D4 / day079）.

day075 交出一层可训练的注意力，day078 交出“把位置加进输入”的那一行。
今天把三样东西拼成一个**块**，而“块”是 Transformer 论文里唯一被复制的单位：

```text
          ┌──────────────── 一个 Encoder 块 ────────────────┐
x ──► LN ──► 注意力 ──► ⊕ ──► LN ──► 前馈 ──► ⊕ ──► y        ⊕ 是残差（跨层直连）
          └──────────────────────────────────────────────┘
```

## 一、三个子层，一个跨天的结论

```text
LayerNorm   每一行标准化       **逐行** → 不跨行取统计量（与 BatchNorm 的分水岭）
前馈         逐位置的两层网络    **逐行** → 同样不破坏置换等变性
残差         y = x + F(x)     反向里多出一条**不减值的路**
```

前两条合起来把 day078 的一句话补完整了：**打破置换等变性的只有位置编码与掩码**，
而 LN 与前馈都是逐行算子。第 4、5 条性质把它们写成两条**逐位**断言。

## 二、pre 与 post 只差 LN 的位置，而效果差一个数量级

```text
pre-LN    y = x + F(LN(x))      现代实现的主流
post-LN   y = LN(x + F(x))      原论文的写法
```

形状一样、参数量一样，而第 9 章的深度实验把差别量出来：
堆 8 层之后，``bare``（关掉 +x）的输入梯度范数衰减到第 1 层的万分之一量级，
而 ``pre`` 基本保持不变。**这一课的值钱结论就在这里。**

## 三、交叉注意力**不能**加因果掩码

解码器块比编码器块多一个子层，而那个子层的 ``K/V`` 来自**另一路**：

```text
Q 来自解码器、K/V 来自编码器 → 权重是 (n_tgt, n_src) 的**长方形**，不是方阵
```

于是“给它加一份方阵掩码”这件事只在 ``n_tgt == n_src`` 时**不报错**，
而那时它悄悄把源序列的后半段删掉了。本包把它做成一条显式拒绝（``AssemblyError``），
并在第 8.8 条性质里量出“加错的代价”。

## 四、六个模块

```text
errors.py   五族失败：形状 / 参数 / **组装** / 数值 / 梯度
types.py    形状、两张口径表（六个阶段与九项梯度 + 交叉六项）、八条性质、九条记录
layers.py   LN(+反向) / 前馈(+反向) / 残差 / 编码器块(+反向) / 交叉注意力(+反向) / 解码器块(+反向)
verify.py   九项块级校验 + 六项交叉校验 + 两项解码器校验、八条性质
depth.py    深度实验：三个变体的 ‖∂loss/∂x‖ 表（这一课的值钱结论）
```

## 五、四条纪律

1. **两个各自合法的部件可以拼不成一个整体**：因此有了 ``AssemblyError``——
   跨路掩码、跨路隐藏维、自注意力不是因果的，都属于“要同时看两个部件”的失败。
2. **“逐行”的算子用 ``==`` 断言**：LN 与前馈的换行序性质是**逐位**的，
   因为它们不产生任何求和顺序的变化；而残差那条 +1 路只能容差比较。
3. **判据要能量出代价**：第 8 条不止断言“它会拒绝”，还量出“加错之后差多少”。
4. **一次只改一个旋钮**：深度实验的三个变体共用同一个目标与同一批初始参数，
   只差摆放位置与残差开关。

## 六、与既有包的接缝

- **上游**：``transformer_core``（day075 的 ``self_attention`` / ``attention_backward`` /
  ``softmax_backward_row`` / ``default_parameters``）、``math_foundations``（day073 的
  ``matmul``/``softmax``/``transpose`` 与 ``uniforms``、day074 的 ``calculus.gradient``）；
- **脚下**：``config`` **没有**新增配置项——``eps``、``d_ff`` 的倍数、摆放位置都是
  “这一次调用或这一次对照的判据”，它们进的是函数参数（与 day073~078 同一条纪律）；
- **下游**：day080（从零实现 Transformer 块）会把这些子层组装成可堆叠的一层；
  day082（变体架构）会用到“pre/post”与“encoder/decoder”这两组轴；
  day085（源码精读）会看到 Hugging Face 里 ``LayerNorm`` 的 ``eps`` 与本文档同源。
"""

from __future__ import annotations

from smart_research_agent.encoder_decoder.depth import (
    DEFAULT_DEPTHS,
    DEFAULT_INIT_SCALE,
    VARIANTS,
    VARIANT_BARE,
    VARIANT_DESCRIPTIONS,
    VARIANT_POST_RESIDUAL,
    VARIANT_PRE_RESIDUAL,
    DepthRow,
    DepthStudy,
    block_parameter_count,
    depth_study,
    ffn_output_scale,
    make_block_parameters,
    make_stack_parameters,
    matrix_frobenius,
    stack_forward,
    stack_input_gradient_norm,
    stack_loss,
    target_matrix,
)
from smart_research_agent.encoder_decoder.errors import (
    FAMILY_OUTCOMES,
    AssemblyError,
    EncoderDecoderError,
    GradientError,
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.encoder_decoder.layers import (
    activate,
    activation_backward,
    add_matrices,
    add_residual,
    block_attention,
    cross_attention,
    cross_attention_backward,
    decoder_block,
    decoder_block_backward,
    encoder_block,
    encoder_block_backward,
    feed_forward,
    feed_forward_backward,
    layer_norm,
    layer_norm_backward,
    stage_order,
    zero_matrix,
)
from smart_research_agent.encoder_decoder.types import (
    ACTIVATIONS,
    ACTIVATION_DESCRIPTIONS,
    ACTIVATION_GELU,
    ACTIVATION_RELU,
    BLOCK_GRADIENT_DESCRIPTIONS,
    BLOCK_GRADIENT_FORMULAS,
    BLOCK_GRADIENT_TARGETS,
    CROSS_GRADIENT_FORMULAS,
    CROSS_GRADIENT_TARGETS,
    DECODER_BLOCK_STAGES,
    DECODER_STAGE_DESCRIPTIONS,
    DECODER_STAGE_SHAPES,
    DEFAULT_EPSILON,
    DEFAULT_FFN_RATIO,
    ENCODER_BLOCK_STAGES,
    ENCODER_DECODER_NOTES,
    ENCODER_DECODER_PROPERTIES,
    GRAD_FFN_B_IN,
    GRAD_FFN_B_OUT,
    GRAD_FFN_W_IN,
    GRAD_FFN_W_OUT,
    GRAD_INPUTS,
    GRAD_NORM1_BETA,
    GRAD_NORM1_GAMMA,
    GRAD_NORM2_BETA,
    GRAD_NORM2_GAMMA,
    GRAD_SOURCE_INPUTS,
    GRAD_TARGET_INPUTS,
    GRAD_W_KEY,
    GRAD_W_OUTPUT,
    GRAD_W_QUERY,
    GRAD_W_VALUE,
    NORM_PLACEMENTS,
    NORM_PLACEMENT_DESCRIPTIONS,
    NORM_POST,
    NORM_PRE,
    PROPERTY_CROSS_NOT_CAUSAL,
    PROPERTY_DESCRIPTIONS,
    PROPERTY_FFN_POSITION_WISE,
    PROPERTY_NORM_ROW_INDEPENDENT,
    PROPERTY_RESIDUAL_IDENTITY,
    PROPERTY_RESIDUAL_UNIT_PATH,
    PROPERTY_ROWS_STANDARDIZED,
    PROPERTY_SCALE_EQUIVARIANT,
    PROPERTY_SHIFT_INVARIANT,
    STAGE_ADD1,
    STAGE_ADD2,
    STAGE_BRANCH1,
    STAGE_BRANCH2,
    STAGE_DESCRIPTIONS,
    STAGE_NORM1,
    STAGE_NORM2,
    STAGE_SHAPES,
    BlockForward,
    BlockGradients,
    BlockParameters,
    BlockShape,
    CrossForward,
    CrossGradients,
    CrossParameters,
    CrossShape,
    DecoderGradients,
    FFNCache,
    FFNGradients,
    FFNWeights,
    NormCache,
    NormGradients,
    matrix_max_absolute,
    relative_matrix_error,
    validate_epsilon,
)
from smart_research_agent.encoder_decoder.verify import (
    GRADIENT_TOLERANCE,
    KIND_BLOCK,
    KIND_CROSS,
    KIND_DECODER,
    KINDS,
    GradientOutcome,
    GradientReport,
    PropertyOutcome,
    PropertyReport,
    block_loss,
    causal_mask_damage,
    check_block_gradients,
    check_cross_attention_is_not_causal,
    check_cross_gradients,
    check_decoder_input_gradients,
    check_feed_forward_is_position_wise,
    check_norm_is_row_independent,
    check_properties,
    check_residual_identity,
    check_residual_unit_path,
    check_rows_are_standardized,
    check_scale_equivariance,
    check_shift_invariance,
    cross_loss,
    numerical_block_gradients,
    numerical_cross_gradients,
    permute_rows,
    reverse_permutation,
)

__all__ = [
    "ACTIVATIONS",
    "ACTIVATION_DESCRIPTIONS",
    "ACTIVATION_GELU",
    "ACTIVATION_RELU",
    "BLOCK_GRADIENT_DESCRIPTIONS",
    "BLOCK_GRADIENT_FORMULAS",
    "BLOCK_GRADIENT_TARGETS",
    "CROSS_GRADIENT_FORMULAS",
    "CROSS_GRADIENT_TARGETS",
    "DECODER_BLOCK_STAGES",
    "DECODER_STAGE_DESCRIPTIONS",
    "DECODER_STAGE_SHAPES",
    "DEFAULT_DEPTHS",
    "DEFAULT_EPSILON",
    "DEFAULT_FFN_RATIO",
    "DEFAULT_INIT_SCALE",
    "ENCODER_BLOCK_STAGES",
    "ENCODER_DECODER_NOTES",
    "ENCODER_DECODER_PROPERTIES",
    "FAMILY_OUTCOMES",
    "GRADIENT_TOLERANCE",
    "GRAD_FFN_B_IN",
    "GRAD_FFN_B_OUT",
    "GRAD_FFN_W_IN",
    "GRAD_FFN_W_OUT",
    "GRAD_INPUTS",
    "GRAD_NORM1_BETA",
    "GRAD_NORM1_GAMMA",
    "GRAD_NORM2_BETA",
    "GRAD_NORM2_GAMMA",
    "GRAD_SOURCE_INPUTS",
    "GRAD_TARGET_INPUTS",
    "GRAD_W_KEY",
    "GRAD_W_OUTPUT",
    "GRAD_W_QUERY",
    "GRAD_W_VALUE",
    "KINDS",
    "KIND_BLOCK",
    "KIND_CROSS",
    "KIND_DECODER",
    "NORM_PLACEMENTS",
    "NORM_PLACEMENT_DESCRIPTIONS",
    "NORM_POST",
    "NORM_PRE",
    "PROPERTY_CROSS_NOT_CAUSAL",
    "PROPERTY_DESCRIPTIONS",
    "PROPERTY_FFN_POSITION_WISE",
    "PROPERTY_NORM_ROW_INDEPENDENT",
    "PROPERTY_RESIDUAL_IDENTITY",
    "PROPERTY_RESIDUAL_UNIT_PATH",
    "PROPERTY_ROWS_STANDARDIZED",
    "PROPERTY_SCALE_EQUIVARIANT",
    "PROPERTY_SHIFT_INVARIANT",
    "STAGE_ADD1",
    "STAGE_ADD2",
    "STAGE_BRANCH1",
    "STAGE_BRANCH2",
    "STAGE_DESCRIPTIONS",
    "STAGE_NORM1",
    "STAGE_NORM2",
    "STAGE_SHAPES",
    "VARIANTS",
    "VARIANT_BARE",
    "VARIANT_DESCRIPTIONS",
    "VARIANT_POST_RESIDUAL",
    "VARIANT_PRE_RESIDUAL",
    "AssemblyError",
    "BlockForward",
    "BlockGradients",
    "BlockParameters",
    "BlockShape",
    "CrossForward",
    "CrossGradients",
    "CrossParameters",
    "CrossShape",
    "DecoderGradients",
    "DepthRow",
    "DepthStudy",
    "EncoderDecoderError",
    "FFNCache",
    "FFNGradients",
    "FFNWeights",
    "GradientError",
    "GradientOutcome",
    "GradientReport",
    "NormCache",
    "NormGradients",
    "NumericError",
    "ParameterError",
    "PropertyOutcome",
    "PropertyReport",
    "ShapeError",
    "activate",
    "activation_backward",
    "add_matrices",
    "add_residual",
    "block_attention",
    "block_loss",
    "block_parameter_count",
    "causal_mask_damage",
    "check_block_gradients",
    "check_cross_attention_is_not_causal",
    "check_cross_gradients",
    "check_decoder_input_gradients",
    "check_feed_forward_is_position_wise",
    "check_norm_is_row_independent",
    "check_properties",
    "check_residual_identity",
    "check_residual_unit_path",
    "check_rows_are_standardized",
    "check_scale_equivariance",
    "check_shift_invariance",
    "cross_attention",
    "cross_attention_backward",
    "cross_loss",
    "decoder_block",
    "decoder_block_backward",
    "depth_study",
    "encoder_block",
    "encoder_block_backward",
    "feed_forward",
    "feed_forward_backward",
    "ffn_output_scale",
    "layer_norm",
    "layer_norm_backward",
    "make_block_parameters",
    "make_stack_parameters",
    "matrix_frobenius",
    "matrix_max_absolute",
    "numerical_block_gradients",
    "numerical_cross_gradients",
    "permute_rows",
    "relative_matrix_error",
    "reverse_permutation",
    "stack_forward",
    "stack_input_gradient_norm",
    "stack_loss",
    "stage_order",
    "target_matrix",
    "validate_epsilon",
    "zero_matrix",
]
