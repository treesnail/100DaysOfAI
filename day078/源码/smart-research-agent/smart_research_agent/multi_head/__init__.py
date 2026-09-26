"""``multi_head``：多头注意力 —— 把"一份混合系数"变成"heads 份"（M7-D2 / day076）.

day075 交出了一层**可训练**的自注意力：``softmax(QKᵀ/√d_k)·V``，
Q/K/V 由四个投影矩阵给出，七步前向、七步反向、五项梯度校验。
那一层里每一行只有**一份**混合系数。

今天只动一个字：把"一份"变成"``heads`` 份"。

```text
day075   context = softmax(QKᵀ/√d_k) · V                每行**一个**分布
day076   context = ‖_h softmax(Q_h K_hᵀ/√d_h) · V_h     每行 **heads** 个分布，再拼接
```

## 一、这一课的三件事

```text
① 九步前向       七个阶段 + split + merge（多出来的两步是一次切片与一次拼接）
② 九步反向       heads 条链各走一遍，再**按行拼回**融合的三块矩阵
③ 四道护栏       五项梯度校验、六条性质、头间差异读数、可达集合的分离见证
```

第 ③ 件是这一课与"把 d_k 切一刀"的差别。多头里"漏掉不会报错"的地方
比 day075 多了一处，而且它最像正确实现：

```text
① 每头缩放分母是 √d_h       不是 √d_k；用错时打分偏小 √heads 倍（heads=4 → 2 倍）
② 每头掩码位置置 0          同一张掩码要发给每一头
③ 每头 softmax 的雅可比     要跑 heads 遍（只跑第一遍形状完全正确）
④ 分头之后必须**相加**      四个投影被所有头共用 → 漏一头只会让那一头学得慢
```

## 二、六个模块，一条从"多一个数字"到"多一块可达集合"的线

```text
errors.py         五族失败：形状 / 参数 / **划分** / 数值 / 梯度
types.py          形状（多出 heads）、九阶段口径表、六条性质、两条记录
layers.py         前向九步 + 反向九步 + 逐头梯度 + "逐头投影再相加"的恒等式
verify.py         五项梯度校验、六条性质、头间差异（全变差距离）
reachability.py   可达集合：单头是一个凸包，多头是 heads 个凸包的闵可夫斯基和
train.py          同参数量下的 heads 对照表（损失 / 峰值 / 命中率 / 头间差异）
```

## 三、五条纪律

1. **权重按行切、激活按列切**：``W_q`` 的形状是 ``(d_k, d_in)``、``d_k`` 在它的**行**上；
   而 ``Q = x·W_qᵀ`` 的形状是 ``(n, d_k)``、``d_k`` 在它的**列**上。
   同一个维度、两张不同的表——切错了不会报错，只会让每一头看到"别人和其他人的混合"。

2. **``heads=1`` 必须与 day075 逐位相同**：连续块划分在 ``heads=1`` 时的唯一划分
   就是整张矩阵，于是每一步的算子、顺序、循环都退化回那七个阶段。
   本包把它写成 ``forward.output == classic.output``——
   **一条"逐位相等"的断言比一条"误差很小"的断言难伪造得多。**

3. **乘法口径只有一条**（继承 day075）：``W`` 的形状是 ``(d_out, d_in)``、投影是 ``x·Wᵀ``；
   于是 ``dW = gradᵀ·x``、``dx = grad·W``。多头不新增任何参数——
   ``heads`` 只决定"这些维度被切成几段"。

4. **不替调用方猜**：``d_k`` 不能被 ``heads`` 整除当场拒绝（``PartitionError``），
   **绝不挑一个"最接近的合法头数"**；``head_dim < |A_i|`` 时拒绝算可达集合
   （一个偏大的可达集合会把"到不了"说成"到得了"）；``d_out != 2`` 时拒绝算距离。

5. **该近似的地方就说近似**：``merge→project`` 恒等式与"头序只是记账"两条性质
   在浮点下只到 ``1e-18`` 量级（本样本 6.94e-18；两种写法求和顺序不同），因此判据是 ``1e-12``；
   而 "heads=1 等于 day075" 与"往返恒等"用的是 ``==``。

## 四、与既有包的接缝

- **上游**：``transformer_core``（day075 的 ``AttentionParams`` / ``project`` /
  ``resolve_mask`` / ``softmax_backward_row`` / MSE / ``InductionTask`` /
  ``default_parameters`` / ``Optimizer``）与 ``math_foundations``（day073 的显式掩码、
  day074 的 ``calculus.gradient`` 与四种优化器）；
- **脚下**：``config`` **没有**新增配置项——头数与容差都是"这一次调用或这一次
  对照的判据"，它们进的是函数参数（与 day073/074/075 同一条纪律）；
- **下游**：day078（位置编码）会解决 day075 用"置换等变"提出的问题；
  day079~080（Encoder 与从零实现）会把这一层堆起来——多头只是"堆之前的最后一个
  横切面"，而 ``grad_inputs`` 仍然是堆叠的前提；
  day083（注意力可视化）会把 ``head_weights`` 画成 ``heads`` 张热力图。
"""

from __future__ import annotations

from smart_research_agent.multi_head.errors import (
    FAMILY_OUTCOMES,
    GradientError,
    MultiHeadError,
    NumericError,
    ParameterError,
    PartitionError,
    ShapeError,
)
from smart_research_agent.multi_head.layers import (
    head_parameter_gradients,
    head_scale,
    head_sequence,
    head_value_of,
    multi_head_attention,
    multi_head_backward,
    project_heads_separately,
)
from smart_research_agent.multi_head.reachability import (
    HULL_TOLERANCE,
    MAX_VERTEX_COMBINATIONS,
    WITNESS_HEAD_ONE_PULLS,
    WITNESS_HEAD_ZERO_PULLS,
    WITNESS_ONE_HEAD_DISTANCE,
    WITNESS_TARGET,
    WITNESS_WEIGHTS,
    DesignedWitness,
    ReachabilityReport,
    allowed_positions,
    convex_hull_2d,
    designed_witness,
    distance_to_convex_hull_2d,
    head_contributions,
    minkowski_vertices,
    multi_head_shape_of,
    reachability_report,
    realize_with_weights,
    single_head_contributions,
    witness_parameters_are_shared,
    witness_report,
)
from smart_research_agent.multi_head.train import (
    DEFAULT_INIT_SCALE,
    MULTIHEAD_GRADIENT_SOURCES,
    MULTIHEAD_PARAMETER_BLOCKS,
    HeadsComparison,
    HeadsComparisonRow,
    MultiHeadTrainingReport,
    analytic_objective,
    batch_accuracy,
    batch_gradients,
    batch_loss,
    batch_mean_disagreement,
    batch_mean_peak_weight,
    compare_heads,
    matrix_totals,
    train_multi_head,
)
from smart_research_agent.multi_head.types import (
    MULTIHEAD_GRADIENT_DESCRIPTIONS,
    MULTIHEAD_GRADIENT_FORMULAS,
    MULTIHEAD_GRADIENT_TARGETS,
    MULTIHEAD_PROPERTIES,
    MULTIHEAD_PROPERTY_DESCRIPTIONS,
    MULTIHEAD_STAGE_DESCRIPTIONS,
    MULTIHEAD_STAGE_SHAPES,
    MULTIHEAD_STAGES,
    PROPERTY_CAUSAL_NO_LEAK_PER_HEAD,
    PROPERTY_HEAD_NON_NEGATIVE,
    PROPERTY_HEAD_ORDER_IS_BOOKKEEPING,
    PROPERTY_MERGE_INVERTS_SPLIT,
    PROPERTY_PER_HEAD_ROW_STOCHASTIC,
    PROPERTY_SINGLE_HEAD_MATCHES_CLASSIC,
    STAGE_MASK,
    STAGE_MERGE,
    STAGE_MIX,
    STAGE_OUTPUT,
    STAGE_PROJECT,
    STAGE_SCALE,
    STAGE_SCORE,
    STAGE_SOFTMAX,
    STAGE_SPLIT,
    HeadGradients,
    HeadPartition,
    MultiHeadForward,
    MultiHeadShape,
    head_gradient_norms,
    head_gradient_shares,
)
from smart_research_agent.multi_head.verify import (
    GRADIENT_TOLERANCE,
    IDENTITY_TOLERANCE,
    HeadDisagreement,
    MultiHeadGradientOutcome,
    MultiHeadGradientReport,
    MultiHeadPropertyOutcome,
    MultiHeadPropertyReport,
    check_causal_no_leak_per_head,
    check_head_non_negative,
    check_head_order_is_bookkeeping,
    check_merge_inverts_split,
    check_multihead_gradients,
    check_per_head_row_stochastic,
    check_properties,
    check_single_head_matches_classic,
    checked_head_order,
    default_head_order,
    head_disagreement,
    head_weight_table,
    numerical_multihead_gradients,
    permute_head_blocks,
    total_variation,
)

__all__ = [
    "DEFAULT_INIT_SCALE",
    "FAMILY_OUTCOMES",
    "GRADIENT_TOLERANCE",
    "HULL_TOLERANCE",
    "IDENTITY_TOLERANCE",
    "MAX_VERTEX_COMBINATIONS",
    "MULTIHEAD_GRADIENT_DESCRIPTIONS",
    "MULTIHEAD_GRADIENT_FORMULAS",
    "MULTIHEAD_GRADIENT_SOURCES",
    "MULTIHEAD_GRADIENT_TARGETS",
    "MULTIHEAD_PARAMETER_BLOCKS",
    "MULTIHEAD_PROPERTIES",
    "MULTIHEAD_PROPERTY_DESCRIPTIONS",
    "MULTIHEAD_STAGES",
    "MULTIHEAD_STAGE_DESCRIPTIONS",
    "MULTIHEAD_STAGE_SHAPES",
    "PROPERTY_CAUSAL_NO_LEAK_PER_HEAD",
    "PROPERTY_HEAD_NON_NEGATIVE",
    "PROPERTY_HEAD_ORDER_IS_BOOKKEEPING",
    "PROPERTY_MERGE_INVERTS_SPLIT",
    "PROPERTY_PER_HEAD_ROW_STOCHASTIC",
    "PROPERTY_SINGLE_HEAD_MATCHES_CLASSIC",
    "STAGE_MASK",
    "STAGE_MERGE",
    "STAGE_MIX",
    "STAGE_OUTPUT",
    "STAGE_PROJECT",
    "STAGE_SCALE",
    "STAGE_SCORE",
    "STAGE_SOFTMAX",
    "STAGE_SPLIT",
    "WITNESS_HEAD_ONE_PULLS",
    "WITNESS_HEAD_ZERO_PULLS",
    "WITNESS_ONE_HEAD_DISTANCE",
    "WITNESS_TARGET",
    "WITNESS_WEIGHTS",
    "DesignedWitness",
    "GradientError",
    "HeadDisagreement",
    "HeadGradients",
    "HeadPartition",
    "HeadsComparison",
    "HeadsComparisonRow",
    "MultiHeadError",
    "MultiHeadForward",
    "MultiHeadGradientOutcome",
    "MultiHeadGradientReport",
    "MultiHeadPropertyOutcome",
    "MultiHeadPropertyReport",
    "MultiHeadShape",
    "MultiHeadTrainingReport",
    "NumericError",
    "ParameterError",
    "PartitionError",
    "ReachabilityReport",
    "ShapeError",
    "allowed_positions",
    "analytic_objective",
    "batch_accuracy",
    "batch_gradients",
    "batch_loss",
    "batch_mean_disagreement",
    "batch_mean_peak_weight",
    "check_causal_no_leak_per_head",
    "check_head_non_negative",
    "check_head_order_is_bookkeeping",
    "check_merge_inverts_split",
    "check_multihead_gradients",
    "check_per_head_row_stochastic",
    "check_properties",
    "check_single_head_matches_classic",
    "checked_head_order",
    "compare_heads",
    "convex_hull_2d",
    "default_head_order",
    "designed_witness",
    "distance_to_convex_hull_2d",
    "head_contributions",
    "head_disagreement",
    "head_gradient_norms",
    "head_gradient_shares",
    "head_parameter_gradients",
    "head_scale",
    "head_sequence",
    "head_value_of",
    "head_weight_table",
    "matrix_totals",
    "minkowski_vertices",
    "multi_head_attention",
    "multi_head_backward",
    "multi_head_shape_of",
    "numerical_multihead_gradients",
    "permute_head_blocks",
    "project_heads_separately",
    "reachability_report",
    "realize_with_weights",
    "single_head_contributions",
    "total_variation",
    "train_multi_head",
    "witness_parameters_are_shared",
    "witness_report",
]
