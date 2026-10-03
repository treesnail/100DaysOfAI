"""``positional_encoding``：位置编码 —— 把“在第几位”加进输入（M7-D3 / day078）.

day075 交出一层**可训练**的自注意力，并在同一天留下一个可断言的性质：
**无掩码时它是置换等变的**——把输入的行换一个顺序，输出只是跟着换。
day076 把 `d_k` 切成 `heads` 段，回答了“看什么”，一个字也没回答“在第几位”。

今天只做一件事：

```text
day075   output = attention(x)                  输出只依赖 x 的**内容**
day078   output = attention(x + table[pos])     把“第几位”加进去
```

## 一、一行加法，两道护栏

```text
① 造表     sinusoidal（公式，无参数、可外推）/ learnable（参数，定长、越界必须拒绝）
② 相加     injected[i] = inputs[i] + table[positions[i]]     ← 这一课的全部算术
③ 反向     dx = dInjected（逐位）；dTable[p] = Σ_{同位置的行} dInjected[i]（**累加**）
```

第 ③ 步是“漏掉不会报错”的错误在本层的化身：把“按位置累加”写成“按行覆盖”，
**形状完全正确、也不会报错**，只在位置重复时让表的梯度偏小。

## 二、两条性质（只在**对齐频率**时成立）

```text
行范数恒定   ‖PE(p)‖ = √(d/2)          每一对贡献 sin² + cos² = 1
位移律       <PE(p), PE(q)> = Σ_i cos((p−q)/f_i)      只依赖 p − q
理由         PE(p + δ) = R_δ · PE(p)   R_δ 是分块**正交旋转** ⇒ 保持范数与内积
```

day073 的 `math_foundations.attention.positional_encoding` 把 `cos` 用了**隔壁**的频率
（`base^((2i+1)/d)`），于是上面两条**一起失效**。本课把两种配对都实现出来
（`pairing=aligned / staggered`），并让性质检查**同时钉住两边**。

## 三、这一课唯一“值钱”的结论：一个可证的**下界**

任务：长度为 `n` 的序列，**每一行都要输出“第 0 位那个 token”**。
模型若是置换等变的（无掩码 + 不加位置编码就是），则它在位移轨道上的平均损失

```text
>= (1 − 1/V) / V          V = 6 时是 0.138889
```

推导见 :mod:`smart_research_agent.positional_encoding.symmetry`，而它是**紧的**：
一个常数预测器（每行输出均匀分布）恰好达到它，而常数预测器也是等变的。
一旦注入位置编码（或开因果掩码），模型就不在那个类里了——**下界对它不适用**，
因此“损失跑到下界之下”是位置信息真的被用上的**证据**。

## 四、六个模块

```text
errors.py      五族失败：形状 / 参数 / **位置越界** / 数值 / 梯度
types.py       形状、两张口径表（六个阶段 / 六项梯度）、七条性质、三条记录
layers.py      造表（正弦 / 可学习 / 全零）+ 注入 + 前向 + 反向（**复用 day075 那层**）
verify.py      六项梯度校验、七条性质、置换缺口读数
symmetry.py    位置选择任务、下界 (1−1/V)/V、常数预测器见证、三个变体
train.py       训练回路（轨道平均损失与梯度）、四个编码刻度、下界判决书
```

## 五、四条纪律

1. **不替调用方外推**：可学习表越界当场 `RangeError`。把“第 20 位”截成“第 7 位”
   不报错、不产生非有限数，只会让模型看到**另一个位置**。
2. **该逐位相等的地方就说逐位相等**：`dx == dInjected`、基准层的置换缺口 `== 0.0`；
   而闭式与实测的内积是两条求和路径，判据取 `1e-12`。
3. **“没有出现过的状态”与“处理了但没发生”必须分开**：`causal=True` 时
   “位置编码打破了等变性”那一条返回**跳过**而不是通过。
4. **判据必须能反向检验**：只在对齐正弦表上测过的两条性质，
   分不清“实现对了”与“判据太松”——错位表必须被它们挡住。

## 六、与既有包的接缝

- **上游**：`transformer_core`（day075 的 `self_attention` / `attention_backward` /
  `masked_mean_squared_error` / `default_parameters` / `AttentionParams`）、
  `math_foundations`（day073 的 `POSITIONAL_BASE` 与 `uniforms`、
  day074 的 `calculus.gradient` 与四种优化器）；
- **脚下**：`config` **没有**新增配置项——表长、基数、容差都是“这一次调用或这一次
  对照的判据“，它们进的是函数参数（与 day073~076 同一条纪律）；
- **下游**：day079（Encoder / Decoder 结构）会把“注入 + 注意力”这一段放进残差与
  LayerNorm 之间，而本课的 `dx` 正是那段残差所需要的输入梯度；
  day080（从零实现 Transformer 块）会把它组装成可堆叠的一层；
  day083（注意力可视化）会把 `head_weights` 画成热力图——**位置编码是理解
  那张图上“为什么第 i 行会看向第 j 列”的钥匙之一**。
"""

from __future__ import annotations

from smart_research_agent.positional_encoding.errors import (
    FAMILY_OUTCOMES,
    GradientError,
    NumericError,
    ParameterError,
    PositionalError,
    RangeError,
    ShapeError,
)
from smart_research_agent.positional_encoding.layers import (
    POSITIONAL_BASE,
    batch_accuracy,
    batch_loss,
    batch_mean_injection_ratio,
    closed_form_offset_inner,
    frequency_of,
    inject,
    learnable_table,
    loss_gradient,
    positional_backward,
    positional_forward,
    positional_loss_from_flat,
    rotate,
    rotation_factor,
    sinusoidal_row,
    sinusoidal_table,
    staggered_table,
    unflatten_flat,
    zero_table,
)
from smart_research_agent.positional_encoding.symmetry import (
    FLOOR_TOLERANCE,
    VARIANT_CAUSAL,
    VARIANT_DESCRIPTIONS,
    VARIANT_EQUIVARIANT,
    VARIANT_SINUSOIDAL,
    FloorVariant,
    ReadoutTask,
    SymmetryFloorReport,
    assemble_floor_report,
    check_task_is_a_position_readout,
    floor_witness,
    make_readout_task,
    orbit_accuracy,
    orbit_floor,
    orbit_loss_spread,
    orbit_losses,
    orbit_mean_loss,
    sample_positions,
    bag_predictor,
)
from smart_research_agent.positional_encoding.train import (
    ENCODING_LABELS,
    ENCODING_LABEL_DESCRIPTIONS,
    GRADIENT_SOURCE_ANALYTIC,
    LABEL_LEARNABLE_FROM_SINE,
    LABEL_LEARNABLE_RANDOM,
    LABEL_NONE,
    LABEL_SINUSOIDAL,
    TRAINABLE_BLOCKS,
    EncodingComparison,
    EncodingComparisonRow,
    PositionalTrainingReport,
    analytic_objective,
    compare_encodings,
    default_parameters,
    freeze_blocks,
    is_equivariant_table,
    orbit_gradients,
    resolve_trainable,
    symmetry_floor_study,
    train_positions,
)
from smart_research_agent.positional_encoding.types import (
    ENCODING_DESCRIPTIONS,
    ENCODING_KINDS,
    ENCODING_LEARNABLE,
    ENCODING_SINUSOIDAL,
    GRAD_INPUTS,
    GRAD_TABLE,
    GRAD_W_KEY,
    GRAD_W_OUTPUT,
    GRAD_W_QUERY,
    GRAD_W_VALUE,
    INITIALIZERS,
    INITIALIZER_DESCRIPTIONS,
    INIT_RANDOM,
    INIT_SCALE,
    INIT_SINUSOIDAL,
    OFFSET_TOLERANCE,
    PAIRING_ALIGNED,
    PAIRING_DESCRIPTIONS,
    PAIRING_STAGGERED,
    PAIRINGS,
    POSITIONAL_GRADIENT_DESCRIPTIONS,
    POSITIONAL_GRADIENT_FORMULAS,
    POSITIONAL_GRADIENT_TARGETS,
    POSITIONAL_NOTES,
    POSITIONAL_PROPERTIES,
    POSITIONAL_PROPERTY_DESCRIPTIONS,
    POSITIONAL_STAGES,
    POSITIONAL_STAGE_DESCRIPTIONS,
    POSITIONAL_STAGE_SHAPES,
    PROPERTY_ADDITIVE_BACKWARD,
    PROPERTY_BREAKS_EQUIVARIANCE,
    PROPERTY_CONSTANT_NORM,
    PROPERTY_LENGTH_IS_A_HARD_BOUND,
    PROPERTY_OFFSET_ONLY,
    PROPERTY_SHIFT_IS_ROTATION,
    PROPERTY_STAGGERED_BREAKS_THE_LAWS,
    ROW_NORM_TOLERANCE,
    STAGE_ADD,
    STAGE_ATTEND,
    STAGE_LOSS,
    STAGE_POSITIONS,
    STAGE_PROJECT,
    STAGE_TABLE,
    EncodingTable,
    PositionalForward,
    PositionalGradients,
    PositionalShape,
    add_matrices,
    check_rows_are_aligned,
    default_positions,
    flatten_parameters,
    gather_rows,
    matrix_row_norms,
    scatter_add_rows,
    unflatten_parameters,
    validate_positions,
    validate_settings,
)
from smart_research_agent.positional_encoding.verify import (
    GRADIENT_TOLERANCE,
    OFFSET_LAW_TOLERANCE,
    ROTATION_TOLERANCE,
    PositionalGradientOutcome,
    PositionalGradientReport,
    PositionalPropertyOutcome,
    PositionalPropertyReport,
    SymmetryBreak,
    check_additive_backward,
    check_injection_breaks_equivariance,
    check_offset_law,
    check_positional_gradients,
    check_properties,
    check_row_norm_is_constant,
    check_shift_is_rotation,
    check_staggered_pairing_breaks_both_laws,
    check_table_length_is_a_hard_bound,
    default_permutation,
    numerical_positional_gradients,
    permute_rows,
    symmetry_break,
)

__all__ = [
    "ENCODING_DESCRIPTIONS",
    "ENCODING_KINDS",
    "ENCODING_LABELS",
    "ENCODING_LABEL_DESCRIPTIONS",
    "ENCODING_LEARNABLE",
    "ENCODING_SINUSOIDAL",
    "FAMILY_OUTCOMES",
    "FLOOR_TOLERANCE",
    "GRADIENT_SOURCE_ANALYTIC",
    "GRADIENT_TOLERANCE",
    "GRAD_INPUTS",
    "GRAD_TABLE",
    "GRAD_W_KEY",
    "GRAD_W_OUTPUT",
    "GRAD_W_QUERY",
    "GRAD_W_VALUE",
    "INITIALIZERS",
    "INITIALIZER_DESCRIPTIONS",
    "INIT_RANDOM",
    "INIT_SCALE",
    "INIT_SINUSOIDAL",
    "LABEL_LEARNABLE_FROM_SINE",
    "LABEL_LEARNABLE_RANDOM",
    "LABEL_NONE",
    "LABEL_SINUSOIDAL",
    "OFFSET_LAW_TOLERANCE",
    "OFFSET_TOLERANCE",
    "PAIRINGS",
    "PAIRING_ALIGNED",
    "PAIRING_DESCRIPTIONS",
    "PAIRING_STAGGERED",
    "POSITIONAL_BASE",
    "POSITIONAL_GRADIENT_DESCRIPTIONS",
    "POSITIONAL_GRADIENT_FORMULAS",
    "POSITIONAL_GRADIENT_TARGETS",
    "POSITIONAL_NOTES",
    "POSITIONAL_PROPERTIES",
    "POSITIONAL_PROPERTY_DESCRIPTIONS",
    "POSITIONAL_STAGES",
    "POSITIONAL_STAGE_DESCRIPTIONS",
    "POSITIONAL_STAGE_SHAPES",
    "PROPERTY_ADDITIVE_BACKWARD",
    "PROPERTY_BREAKS_EQUIVARIANCE",
    "PROPERTY_CONSTANT_NORM",
    "PROPERTY_LENGTH_IS_A_HARD_BOUND",
    "PROPERTY_OFFSET_ONLY",
    "PROPERTY_SHIFT_IS_ROTATION",
    "PROPERTY_STAGGERED_BREAKS_THE_LAWS",
    "ROTATION_TOLERANCE",
    "ROW_NORM_TOLERANCE",
    "STAGE_ADD",
    "STAGE_ATTEND",
    "STAGE_LOSS",
    "STAGE_POSITIONS",
    "STAGE_PROJECT",
    "STAGE_TABLE",
    "TRAINABLE_BLOCKS",
    "VARIANT_CAUSAL",
    "VARIANT_DESCRIPTIONS",
    "VARIANT_EQUIVARIANT",
    "VARIANT_SINUSOIDAL",
    "EncodingComparison",
    "EncodingComparisonRow",
    "EncodingTable",
    "FloorVariant",
    "GradientError",
    "NumericError",
    "ParameterError",
    "PositionalError",
    "PositionalForward",
    "PositionalGradientOutcome",
    "PositionalGradientReport",
    "PositionalGradients",
    "PositionalPropertyOutcome",
    "PositionalPropertyReport",
    "PositionalShape",
    "PositionalTrainingReport",
    "RangeError",
    "ReadoutTask",
    "ShapeError",
    "SymmetryBreak",
    "SymmetryFloorReport",
    "add_matrices",
    "analytic_objective",
    "assemble_floor_report",
    "batch_accuracy",
    "batch_loss",
    "batch_mean_injection_ratio",
    "check_additive_backward",
    "check_injection_breaks_equivariance",
    "check_offset_law",
    "check_positional_gradients",
    "check_properties",
    "check_row_norm_is_constant",
    "check_rows_are_aligned",
    "check_shift_is_rotation",
    "check_staggered_pairing_breaks_both_laws",
    "check_table_length_is_a_hard_bound",
    "check_task_is_a_position_readout",
    "closed_form_offset_inner",
    "compare_encodings",
    "default_permutation",
    "default_parameters",
    "default_positions",
    "flatten_parameters",
    "floor_witness",
    "freeze_blocks",
    "frequency_of",
    "gather_rows",
    "inject",
    "is_equivariant_table",
    "learnable_table",
    "loss_gradient",
    "make_readout_task",
    "matrix_row_norms",
    "numerical_positional_gradients",
    "orbit_accuracy",
    "orbit_floor",
    "orbit_gradients",
    "orbit_loss_spread",
    "orbit_losses",
    "orbit_mean_loss",
    "permute_rows",
    "positional_backward",
    "positional_forward",
    "positional_loss_from_flat",
    "resolve_trainable",
    "rotate",
    "rotation_factor",
    "sample_positions",
    "scatter_add_rows",
    "sinusoidal_row",
    "sinusoidal_table",
    "staggered_table",
    "symmetry_break",
    "symmetry_floor_study",
    "train_positions",
    "bag_predictor",
    "unflatten_flat",
    "unflatten_parameters",
    "validate_positions",
    "validate_settings",
    "zero_table",
]
