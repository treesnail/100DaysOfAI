"""``math_foundations``：Transformer 与深度学习之前的数学地基（Math-D1~D2 / day073~day074）.

M6 结束时（day072）我们已经有一条能跑、能评估、也能运营的 RAG 链路。
而从 day073 起进入**底层**：微积分与优化 → Attention → Transformer →
从零实现 → 深度学习。要读懂那些公式，先把几样东西的**数学含义**钉死：

```text
向量与矩阵      点积、模长、余弦、矩阵乘法、投影 —— 注意力打分的全部零件   （day073）
概率分布        熵、交叉熵、KL、期望、条件概率、贝叶斯 —— LLM 输出的"一个分布"（day073）
注意力          把上面两样合起来：softmax(QKᵀ/√d_k)·V                    （day073）
微分            差分、梯度、链式法则、雅可比 —— "变化率"与它的近似        （day074）
反向传播        把链式法则自动跑一遍（标量自动微分）                      （day074）
优化            三种优化器 + 四种学习率调度 + 梯度裁剪 —— "走多远"         （day074）
梯度对照        解析式 vs 数值差分：六项逐点对照                          （day074）
```

## 一、这一课最值钱的一句话：注意力 = 可微的检索

```text
检索（day066~day071）  cos(query, doc) 排序 → 取前 k 条 → 拼进提示词
                       **离散的选择**：取或不取，梯度传不回去
注意力（day073）       softmax(q·k) 打分 → 一组权重 → 对 value 加权平均
                       **可微的混合**：想要什么是一个概率分布，梯度能一路回传
```

两者都在回答"哪些内容与当前问题相关"。差别在于注意力**不做**那个不可导的取舍，
而是把"想要什么"变成分布——于是模型可以"学"出该看哪里。
而"能学"这句话的技术前提正是 day074 的三块：**导数（往哪走）、反向传播（怎么算）、
优化器（走多远）**。没有它们，"可微"只是一个形容词。

## 二、十个模块，一条从几何到训练的线

```text
errors.py       四族失败（形状 / 数值 / 参数 / 概率表），修复人各不相同
types.py        形状与**八张口径表**：算子（含公式）、注意力变体、对照项、分布判据、
                差分方法（含误差阶数）、优化器（含更新公式）、调度、梯度对照项
linalg.py       十个算子：dot / norm / normalize / cosine / matmul / transpose /
                projection / softmax / power_iteration / low_rank
probability.py  一个分布（Distribution）与一张联合表（JointTable）                     [day073]
attention.py    缩放点积注意力（可选因果掩码与温度）、多头、位置编码，                   [day073]
                以及一个把"为什么要除以 √d_k"**量出来**的数值实验
bridge.py       与项目里既有实现的八项逐点对照（"公式只有一份真相"）                     [day073]
calculus.py     三种差分（前向/后向/中心）+ 梯度/方向导数/雅可比 + 步长误差实验 + 链式法则 [day074]
autograd.py     标量自动微分：计算图、拓扑序、反向累积（`+=`）、value_and_grad         [day074]
optim.py        SGD / 动量 / Adam（含偏差修正）+ 四种调度 + 梯度裁剪 + 训练回路          [day074]
gradcheck.py    六项梯度对照：解析式 vs 数值差分（含 softmax 雅可比与自动微分链）         [day074]
```

## 三、六条贯穿全包的纪律

1. **公式必须能被复核。** 每个算子的公式都写在 ``types`` 的口径表里，
   再由 ``bridge``（函数值）与 ``gradcheck``（变化率）把它与项目里已有的实现对上：
   八项值对照里七项逐位一致、一项（零向量的归一化）是**有意记录的约定差异**；
   六项梯度对照全部逐点一致。

   ```text
   教学实现   零向量归一化 → 报错（零向量没有方向，这不是数学层能替调用方决定的）
   生产实现   零向量归一化 → 原样返回（写入侧不能因为"没有方向"就抛异常）
   ```

   两条都对。危险的不是"不一样"，而是"不一样而没人知道"。

2. **单位一律 nats。** 熵、交叉熵、KL 全部与 ``math.log`` 同底：
   ``sft.loss`` 的交叉熵是 nats、``perplexity = exp(loss)`` 也是 nats。
   若这里改用 bits，同一个数字在两处报告里会差一个 ``ln 2`` 倍，
   而它们各自的说明都"看起来对"。

3. **数值稳定不是优化，是正确性。** ``softmax`` 先减最大值、
   ``log_softmax`` 不经过 ``log(softmax)``、``sigmoid`` 按符号分支——
   这三处的朴素写法在 ``z = [1000, 1001]`` 或 ``x = −800`` 上会得到
   ``nan`` / ``−inf``，而它们在常见输入上完全正确（因此极难在测试里被发现）。

4. **随机性必须可注入。** 采样要吃调用方给的 ``u``，随机数由
   ``uniforms(count, seed)`` 这串确定性数列提供——于是"按这个分布采 10 万次"
   这条结论可以被逐位复现。

5. **形状先于语义。** day073 的 ``validate_vector`` / ``validate_matrix`` 挡的是
   ``zip`` 的静默截断；day074 的差分同样第一行校验步长与点的**定义域**
   （``h = 0`` 时导数没有定义）。"先校验，再谈语义"是同一个动作在两个尺度上的重复。

6. **误差必须被量出来。** ``sampled_dot_product_variance``（day073）把
   "点积方差随维度线性增长"量出来；``step_size_study``（day074）把
   "前向 O(h) / 中心 O(h²)"与"h 太小之后误差反而上升"量出来。
   能算出来的东西不要只用一句话带过——**结论是给人的，"量出来"是给复核的**。

## 四、与既有包的接缝

- **上游**：无（这一层不依赖智能体的任何模块——它是纯数学）；
- **脚下**：``config`` **没有**为本包新增配置项：容差、温度、秩、步长、学习率曲线
  都是"这一次计算的判据"，它们进的是函数参数。写进配置会让人以为
  "改了容差就等于改了结论"（而容差只影响这一份报告的判定口径）；
- **下游**：day075（Attention）会用 ``gradcheck`` 验证注意力权重的梯度、
  用 ``optim`` 的训练回路把一个可训练的注意力层跑起来；
  day079~day080（Encoder/Decoder 与从零实现）会用到 ``linalg`` 的全部算子与
  ``autograd`` 的链式法则；day051 的 LoRA 与 day073 的 ``low_rank_approximation``
  是同一件事的两种说法（一个是工程实现，一个是数学原型）；
  day054 的 DPO 目标与 day074 的 ``cross_entropy`` 梯度对照是同一条公式的两种用途。
"""

from __future__ import annotations

from smart_research_agent.math_foundations.attention import (
    POSITIONAL_BASE,
    DotProductStudy,
    MultiHeadReport,
    attention_scores,
    causal_mask,
    masked_softmax_rows,
    merge_heads,
    multi_head_attention,
    positional_encoding,
    sampled_dot_product_variance,
    scaled_dot_product_attention,
    scaling_factor,
    split_heads,
)
from smart_research_agent.math_foundations.autograd import (
    Scalar,
    ScalarExpression,
    TraceRow,
    ensure_scalar,
    gradients_of,
    topological_order,
    value_and_grad,
)
from smart_research_agent.math_foundations.bridge import (
    CHECK_FUNCTIONS,
    BridgeReport,
    CheckOutcome,
    cross_check_all,
    sample_outputs,
)
from smart_research_agent.math_foundations.calculus import (
    DEFAULT_STEP,
    DEFAULT_STEP_SIZES,
    ScalarFunction,
    StepRecord,
    StepSizeStudy,
    VectorFunction,
    backward_difference,
    best_step_for_central,
    central_difference,
    chain_rule,
    compose,
    derivative,
    directional_derivative,
    forward_difference,
    gradient,
    jacobian,
    linear_approximation,
    second_difference,
    step_size_study,
    taylor_accuracy,
)
from smart_research_agent.math_foundations.errors import (
    MathError,
    NumericError,
    ParameterError,
    ShapeError,
    TableError,
)
from smart_research_agent.math_foundations.gradcheck import (
    AUTOGRAD_CASES,
    FLOAT_EPSILON,
    GRADIENT_CHECKS,
    GRADIENT_TOLERANCE,
    LOGIT_VECTORS,
    PERPLEXITY_INPUTS,
    SCALAR_POINTS,
    TARGET_INDICES,
    GradientOutcome,
    GradientReport,
    check_autograd_chain,
    check_cross_entropy_gradient,
    check_gradients_all,
    check_log_sigmoid_gradient,
    check_perplexity_gradient,
    check_sigmoid_gradient,
    check_softmax_jacobian,
    difference_resolution,
    max_absolute_gap,
    max_scaled_gap,
)
from smart_research_agent.math_foundations.linalg import (
    DEFAULT_ITERATION_TOLERANCE,
    DEFAULT_MAX_ITERATIONS,
    add,
    argmax,
    cosine,
    dot,
    frobenius_norm,
    identity,
    log_softmax,
    low_rank_approximation,
    matmul,
    matrix_add,
    matrix_scale,
    matvec,
    norm,
    normalize,
    outer,
    power_iteration,
    projection,
    reconstruction_error,
    row_normalize,
    scale,
    softmax,
    subtract,
    top_k_indices,
    transpose,
    zeros,
)
from smart_research_agent.math_foundations.optim import (
    DEFAULT_EPSILON,
    OPTIMIZER_CLASSES,
    SCHEDULE_FACTORIES,
    AdamOptimizer,
    MomentumOptimizer,
    Objective,
    Optimizer,
    SGDOptimizer,
    Schedule,
    TrainingTrace,
    clip_by_global_norm,
    clip_by_value,
    constant_schedule,
    cosine_schedule,
    flatten_matrices,
    global_norm,
    make_optimizer,
    make_schedule,
    minimize,
    step_decay_schedule,
    unflatten_matrices,
    warmup_cosine_schedule,
)
from smart_research_agent.math_foundations.probability import (
    LCG_INCREMENT,
    LCG_MODULUS,
    LCG_MULTIPLIER,
    Distribution,
    JointTable,
    cross_entropy,
    entropy,
    expectation,
    kl_divergence,
    max_entropy,
    perplexity_from_entropy,
    sample_index,
    uniforms,
    variance,
)
from smart_research_agent.math_foundations.types import (
    ATTENTION_CAUSAL,
    ATTENTION_DOT,
    ATTENTION_MULTI_HEAD,
    ATTENTION_SCALED,
    ATTENTION_VARIANT_DESCRIPTIONS,
    ATTENTION_VARIANTS,
    BACKWARD,
    BRIDGE_COSINE,
    BRIDGE_CROSS_ENTROPY,
    BRIDGE_LOG_SIGMOID,
    BRIDGE_LOG_SOFTMAX,
    BRIDGE_NORMALIZE,
    BRIDGE_PERPLEXITY,
    BRIDGE_SIGMOID,
    BRIDGE_SOFTMAX,
    BRIDGE_TARGET_DESCRIPTIONS,
    BRIDGE_TARGET_SOURCES,
    BRIDGE_TARGETS,
    CALCULUS_METHOD_DESCRIPTIONS,
    CALCULUS_METHOD_FORMULAS,
    CALCULUS_METHOD_ORDERS,
    CALCULUS_METHODS,
    CALCULUS_ORDER_CENTRAL,
    CALCULUS_ORDER_FORWARD,
    CENTRAL,
    CHECK_LENGTH_MATCHES_LABELS,
    CHECK_NON_EMPTY,
    CHECK_NON_NEGATIVE,
    CHECK_SUMS_TO_ONE,
    DEFAULT_SUM_TOLERANCE,
    DEFAULT_TOLERANCE,
    DISTRIBUTION_CHECK_DESCRIPTIONS,
    DISTRIBUTION_CHECKS,
    FORWARD,
    GRADIENT_AUTOGRAD_CHAIN,
    GRADIENT_CROSS_ENTROPY,
    GRADIENT_LOG_SIGMOID,
    GRADIENT_PERPLEXITY,
    GRADIENT_SIGMOID,
    GRADIENT_SOFTMAX,
    GRADIENT_STATUSES,
    GRADIENT_TARGET_DESCRIPTIONS,
    GRADIENT_TARGET_FORMULAS,
    GRADIENT_TARGET_SOURCES,
    GRADIENT_TARGETS,
    LINALG_OP_DESCRIPTIONS,
    LINALG_OP_FORMULAS,
    LINALG_OPS,
    OP_COSINE,
    OP_DOT,
    OP_LOW_RANK,
    OP_MATMUL,
    OP_NORM,
    OP_NORMALIZE,
    OP_POWER_ITERATION,
    OP_PROJECTION,
    OP_SOFTMAX,
    OP_TRANSPOSE,
    OPTIMIZER_ADAM,
    OPTIMIZER_DESCRIPTIONS,
    OPTIMIZER_MOMENTUM,
    OPTIMIZER_SGD,
    OPTIMIZER_UPDATE_FORMULAS,
    OPTIMIZERS,
    SCHEDULE_CONSTANT,
    SCHEDULE_COSINE,
    SCHEDULE_DESCRIPTIONS,
    SCHEDULE_FORMULAS,
    SCHEDULE_STEP_DECAY,
    SCHEDULE_WARMUP_COSINE,
    SCHEDULES,
    AttentionReport,
    Matrix,
    Vector,
    VectorSummary,
    check_distribution,
    close,
    is_finite,
    is_square,
    matrix_shape,
    normalize_rows,
    relative_error,
    require_same_dimension,
    require_square,
    row_sums,
    validate_matrix,
    validate_vector,
)

__all__ = [
    "ATTENTION_CAUSAL",
    "ATTENTION_DOT",
    "ATTENTION_MULTI_HEAD",
    "ATTENTION_SCALED",
    "ATTENTION_VARIANTS",
    "ATTENTION_VARIANT_DESCRIPTIONS",
    "AUTOGRAD_CASES",
    "BACKWARD",
    "BRIDGE_COSINE",
    "BRIDGE_CROSS_ENTROPY",
    "BRIDGE_LOG_SIGMOID",
    "BRIDGE_LOG_SOFTMAX",
    "BRIDGE_NORMALIZE",
    "BRIDGE_PERPLEXITY",
    "BRIDGE_SIGMOID",
    "BRIDGE_SOFTMAX",
    "BRIDGE_TARGETS",
    "BRIDGE_TARGET_DESCRIPTIONS",
    "BRIDGE_TARGET_SOURCES",
    "CALCULUS_METHODS",
    "CALCULUS_METHOD_DESCRIPTIONS",
    "CALCULUS_METHOD_FORMULAS",
    "CALCULUS_METHOD_ORDERS",
    "CALCULUS_ORDER_CENTRAL",
    "CALCULUS_ORDER_FORWARD",
    "CENTRAL",
    "CHECK_FUNCTIONS",
    "CHECK_LENGTH_MATCHES_LABELS",
    "CHECK_NON_EMPTY",
    "CHECK_NON_NEGATIVE",
    "CHECK_SUMS_TO_ONE",
    "DEFAULT_EPSILON",
    "DEFAULT_ITERATION_TOLERANCE",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_STEP",
    "DEFAULT_STEP_SIZES",
    "DEFAULT_SUM_TOLERANCE",
    "DEFAULT_TOLERANCE",
    "DISTRIBUTION_CHECKS",
    "DISTRIBUTION_CHECK_DESCRIPTIONS",
    "FLOAT_EPSILON",
    "FORWARD",
    "GRADIENT_AUTOGRAD_CHAIN",
    "GRADIENT_CHECKS",
    "GRADIENT_CROSS_ENTROPY",
    "GRADIENT_LOG_SIGMOID",
    "GRADIENT_PERPLEXITY",
    "GRADIENT_SIGMOID",
    "GRADIENT_SOFTMAX",
    "GRADIENT_STATUSES",
    "GRADIENT_TARGETS",
    "GRADIENT_TARGET_DESCRIPTIONS",
    "GRADIENT_TARGET_FORMULAS",
    "GRADIENT_TARGET_SOURCES",
    "GRADIENT_TOLERANCE",
    "LCG_INCREMENT",
    "LCG_MODULUS",
    "LCG_MULTIPLIER",
    "LINALG_OPS",
    "LINALG_OP_DESCRIPTIONS",
    "LINALG_OP_FORMULAS",
    "LOGIT_VECTORS",
    "OP_COSINE",
    "OP_DOT",
    "OP_LOW_RANK",
    "OP_MATMUL",
    "OP_NORMALIZE",
    "OP_NORM",
    "OP_POWER_ITERATION",
    "OP_PROJECTION",
    "OP_SOFTMAX",
    "OP_TRANSPOSE",
    "OPTIMIZERS",
    "OPTIMIZER_ADAM",
    "OPTIMIZER_CLASSES",
    "OPTIMIZER_DESCRIPTIONS",
    "OPTIMIZER_MOMENTUM",
    "OPTIMIZER_SGD",
    "OPTIMIZER_UPDATE_FORMULAS",
    "PERPLEXITY_INPUTS",
    "POSITIONAL_BASE",
    "SCALAR_POINTS",
    "SCHEDULES",
    "SCHEDULE_CONSTANT",
    "SCHEDULE_COSINE",
    "SCHEDULE_DESCRIPTIONS",
    "SCHEDULE_FACTORIES",
    "SCHEDULE_FORMULAS",
    "SCHEDULE_STEP_DECAY",
    "SCHEDULE_WARMUP_COSINE",
    "TARGET_INDICES",
    "AdamOptimizer",
    "AttentionReport",
    "BridgeReport",
    "CheckOutcome",
    "Distribution",
    "DotProductStudy",
    "GradientOutcome",
    "GradientReport",
    "JointTable",
    "MathError",
    "Matrix",
    "MomentumOptimizer",
    "MultiHeadReport",
    "NumericError",
    "Objective",
    "Optimizer",
    "ParameterError",
    "SGDOptimizer",
    "Scalar",
    "ScalarExpression",
    "ScalarFunction",
    "Schedule",
    "ShapeError",
    "StepRecord",
    "StepSizeStudy",
    "TableError",
    "TraceRow",
    "TrainingTrace",
    "Vector",
    "VectorFunction",
    "VectorSummary",
    "add",
    "argmax",
    "attention_scores",
    "backward_difference",
    "best_step_for_central",
    "causal_mask",
    "central_difference",
    "chain_rule",
    "check_autograd_chain",
    "check_cross_entropy_gradient",
    "check_distribution",
    "check_gradients_all",
    "check_log_sigmoid_gradient",
    "check_perplexity_gradient",
    "check_sigmoid_gradient",
    "check_softmax_jacobian",
    "clip_by_global_norm",
    "clip_by_value",
    "close",
    "compose",
    "constant_schedule",
    "cosine",
    "cosine_schedule",
    "cross_check_all",
    "cross_entropy",
    "derivative",
    "difference_resolution",
    "directional_derivative",
    "dot",
    "ensure_scalar",
    "entropy",
    "expectation",
    "flatten_matrices",
    "forward_difference",
    "frobenius_norm",
    "global_norm",
    "gradient",
    "gradients_of",
    "identity",
    "is_finite",
    "is_square",
    "jacobian",
    "kl_divergence",
    "linear_approximation",
    "log_softmax",
    "low_rank_approximation",
    "make_optimizer",
    "make_schedule",
    "masked_softmax_rows",
    "matmul",
    "matrix_add",
    "matrix_scale",
    "matrix_shape",
    "matvec",
    "max_absolute_gap",
    "max_entropy",
    "max_scaled_gap",
    "merge_heads",
    "minimize",
    "multi_head_attention",
    "norm",
    "normalize",
    "normalize_rows",
    "outer",
    "perplexity_from_entropy",
    "positional_encoding",
    "power_iteration",
    "projection",
    "reconstruction_error",
    "relative_error",
    "require_same_dimension",
    "require_square",
    "row_normalize",
    "row_sums",
    "sample_index",
    "sample_outputs",
    "sampled_dot_product_variance",
    "scale",
    "scaled_dot_product_attention",
    "scaling_factor",
    "second_difference",
    "softmax",
    "split_heads",
    "step_decay_schedule",
    "step_size_study",
    "subtract",
    "taylor_accuracy",
    "top_k_indices",
    "topological_order",
    "transpose",
    "unflatten_matrices",
    "uniforms",
    "validate_matrix",
    "validate_vector",
    "value_and_grad",
    "variance",
    "warmup_cosine_schedule",
    "zeros",
]
