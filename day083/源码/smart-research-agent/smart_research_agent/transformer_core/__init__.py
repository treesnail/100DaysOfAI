"""``transformer_core``：可训练的注意力（M7-D1 / day075）.

day073 把注意力的**公式**钉死了：``softmax(QKᵀ/√d_k)·V``，七个算子拼起来、逐点对照过。
但那一份里的 Q/K/V 是**给定的输入**——它能回答"公式算得对"，不能回答"权重该往哪调"。

今天补上这一句：

```text
day073  Attention(Q, K, V)      Q/K/V 是输入        纯函数，无参数
day075  SelfAttention(x; W)     Q = x·W_qᵀ 等四个矩阵  可训练（因此"可微"不再是一个形容词）
```

## 一、这一课的三件事

```text
① 前向七步       投影 → 打分 → 缩放 → 掩码 → softmax → 加权 → 输出投影
② 反向七步       顺序恰好相反，每一步都是"上游梯度 × 局部导数"
③ 四道护栏       梯度校验（解析 vs 数值）、性质检查、检索类比、induction 训练
```

第 ③ 件是这一课与"写一个能跑的实现"的差别：
反向传播里最危险的三处（softmax 的雅可比、缩放系数、掩码位置的梯度）
**写错了都不会报错**——只会让训练慢一点、或者让"看不到未来"这条性质悄悄失效。
因此每一处都配了一条能独立失败的检查。

## 二、五个模块，一条从"公式"到"训练曲线"的线

```text
errors.py     五族失败：形状 / 参数 / 数值 / **梯度**（多出的一族属于控制流）
types.py      形状、三张口径表（七个阶段 / 五项梯度校验 / 四项性质）与四条记录
layers.py     前向七步 + 反向七步 + 逐元素 MSE（损失平凡，因此失败只可能来自注意力）
verify.py     五项梯度校验、四条性质、与检索的类比（峰值一致 / top-k 重合 / 秩相关）
train.py      induction 任务、批量损失与梯度、训练回路与三列读数的报告
```

## 三、五条纪律

1. **乘法口径只有一条**：``W`` 的形状是 ``(d_out, d_in)``、投影是 ``x·Wᵀ``
   （与 ``nn.Linear`` 一致）。于是 ``dW = gradᵀ·x``、``dx = grad·W``——
   两条式子各有一条用手算小矩阵核对的测试。

2. **掩码是显式的**：被掩码的位置权重**恰好是 0.0**、且不计入 softmax 的分母
   （沿用 day073 的 ``masked_softmax_rows``，而不是把打分写成 ``-inf``）。
   反向里同样要把那些位置的梯度置 0——**这一条最隐蔽**：
   前向看起来完全正确，只有 key 的梯度里混进了"未来位置"的贡献。

3. **损失取 MSE**：``∂MSE/∂y = 2(y−t)/N`` 一行就能手算，
   于是"外部输入的梯度"是平凡的，任何梯度校验失败都只可能来自注意力那七步。
   这与 day074 用碗形函数验证优化器是同一个手法：**一次只留一个不确定性**。

4. **不替调用方猜**：输入行全零当场拒绝（它会让 softmax 给出均匀分布，
   而"均匀分布"看起来像"模型还没学到"）；``causal=True`` 与显式掩码不能同时给
   （两个来源同时生效时"到底用哪张"要读代码才知道）。

5. **三个读数一起看**：损失（混得多准）、峰值权重（注意力多尖）、命中率（argmax 对不对）。
   只有损失会骗人——它可能降了，而机制并没有学到"去找那个 token"。

## 四、与既有包的接缝

- **上游**：``math_foundations``（day073 的显式掩码与 softmax、day074 的
  ``calculus.gradient`` 与四种优化器）；
- **脚下**：``config`` **没有**新增配置项：容差、步长、初始化幅度、学习率
  都是"这一次计算或这一次训练的判据"，它们进的是函数参数；
- **下游**：day076（多头）会把 ``d_k`` 切成若干段、day078（位置编码）
  会解决"置换等变"那条性质指出的问题、day079~080（Encoder 与从零实现）
  会把这一层堆起来——而 ``grad_inputs`` 就是为那一步准备的（没有它，层没法堆）。
"""

from __future__ import annotations

from smart_research_agent.transformer_core.errors import (
    GradientError,
    NumericError,
    ParameterError,
    ShapeError,
    TransformerError,
)
from smart_research_agent.transformer_core.layers import (
    attention_backward,
    full_mask,
    masked_mean_squared_error,
    masked_mse_gradient,
    mean_squared_error,
    mse_gradient,
    resolve_mask,
    row_argmax_hits,
    self_attention,
    softmax_backward_row,
)
from smart_research_agent.transformer_core.train import (
    DEFAULT_INIT_SCALE,
    GRADIENT_SOURCES,
    GRADIENT_SOURCE_ANALYTIC,
    GRADIENT_SOURCE_NUMERIC,
    PARAMETER_BLOCKS,
    AttentionTrainingReport,
    InductionTask,
    analytic_objective,
    batch_accuracy,
    batch_gradients,
    batch_loss,
    batch_mean_peak_weight,
    default_parameters,
    make_induction_batch,
    make_induction_task,
    train_attention,
)
from smart_research_agent.transformer_core.types import (
    ATTENTION_STAGES,
    ATTENTION_STAGE_DESCRIPTIONS,
    ATTENTION_STAGE_SHAPES,
    DEFAULT_SUM_TOLERANCE,
    GRADIENT_TARGETS,
    GRADIENT_TARGET_DESCRIPTIONS,
    GRADIENT_TARGET_FORMULAS,
    GRAD_INPUTS,
    GRAD_W_KEY,
    GRAD_W_OUTPUT,
    GRAD_W_QUERY,
    GRAD_W_VALUE,
    PROPERTY_CHECKS,
    PROPERTY_CHECK_DESCRIPTIONS,
    PROPERTY_CAUSAL_NO_LEAK,
    PROPERTY_NON_NEGATIVE,
    PROPERTY_PERMUTATION_EQUIVARIANCE,
    PROPERTY_ROW_STOCHASTIC,
    STAGE_MASK,
    STAGE_MIX,
    STAGE_OUTPUT,
    STAGE_PROJECT,
    STAGE_SCALE,
    STAGE_SCORE,
    STAGE_SOFTMAX,
    AttentionForward,
    AttentionParams,
    AttentionShape,
    ParameterGradients,
    assert_rows_are_distributions,
    check_weights_are_a_distribution,
    matrix_max_absolute,
    project,
    relative_matrix_error,
    softmax_shape_scale,
)
from smart_research_agent.transformer_core.verify import (
    GRADIENT_TOLERANCE,
    GradientCheckOutcome,
    GradientCheckReport,
    PropertyOutcome,
    PropertyReport,
    RetrievalComparison,
    check_attention_gradients,
    check_causal_no_leak,
    check_non_negative,
    check_permutation_equivariance,
    check_properties,
    check_row_stochastic,
    compare_with_retrieval,
    default_permutation,
    numerical_parameter_gradients,
    permutation_gap,
    rank_correlation,
)

__all__ = [
    "ATTENTION_STAGES",
    "ATTENTION_STAGE_DESCRIPTIONS",
    "ATTENTION_STAGE_SHAPES",
    "DEFAULT_INIT_SCALE",
    "DEFAULT_SUM_TOLERANCE",
    "GRADIENT_SOURCES",
    "GRADIENT_SOURCE_ANALYTIC",
    "GRADIENT_SOURCE_NUMERIC",
    "GRADIENT_TARGETS",
    "GRADIENT_TARGET_DESCRIPTIONS",
    "GRADIENT_TARGET_FORMULAS",
    "GRADIENT_TOLERANCE",
    "GRAD_INPUTS",
    "GRAD_W_KEY",
    "GRAD_W_OUTPUT",
    "GRAD_W_QUERY",
    "GRAD_W_VALUE",
    "PROPERTY_CHECKS",
    "PROPERTY_CHECK_DESCRIPTIONS",
    "PROPERTY_CAUSAL_NO_LEAK",
    "PROPERTY_NON_NEGATIVE",
    "PROPERTY_PERMUTATION_EQUIVARIANCE",
    "PROPERTY_ROW_STOCHASTIC",
    "STAGE_MASK",
    "STAGE_MIX",
    "STAGE_OUTPUT",
    "STAGE_PROJECT",
    "STAGE_SCALE",
    "STAGE_SCORE",
    "STAGE_SOFTMAX",
    "AttentionForward",
    "AttentionParams",
    "AttentionShape",
    "AttentionTrainingReport",
    "GradientCheckOutcome",
    "GradientCheckReport",
    "GradientError",
    "InductionTask",
    "NumericError",
    "PARAMETER_BLOCKS",
    "ParameterError",
    "ParameterGradients",
    "PropertyOutcome",
    "PropertyReport",
    "RetrievalComparison",
    "ShapeError",
    "TransformerError",
    "analytic_objective",
    "assert_rows_are_distributions",
    "attention_backward",
    "batch_accuracy",
    "batch_gradients",
    "batch_loss",
    "batch_mean_peak_weight",
    "check_attention_gradients",
    "check_causal_no_leak",
    "check_non_negative",
    "check_permutation_equivariance",
    "check_properties",
    "check_row_stochastic",
    "check_weights_are_a_distribution",
    "compare_with_retrieval",
    "default_parameters",
    "default_permutation",
    "full_mask",
    "make_induction_batch",
    "make_induction_task",
    "masked_mean_squared_error",
    "masked_mse_gradient",
    "matrix_max_absolute",
    "mean_squared_error",
    "mse_gradient",
    "numerical_parameter_gradients",
    "permutation_gap",
    "project",
    "rank_correlation",
    "relative_matrix_error",
    "resolve_mask",
    "row_argmax_hits",
    "self_attention",
    "softmax_backward_row",
    "softmax_shape_scale",
    "train_attention",
]
