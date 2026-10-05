"""``backprop``：把 day089 那条链的**梯度**补上（M8-D2 / day090）.

day089 结束时，那句话是被明写的：**"有前向、有损失、没有反向，而且没有交给自动微分"**。
今天补上那一行反向，并第一次让 ``GradientError``（day075 定义、连续缺席八天）真正被抛出来。

```text
一个神经元的导数  →  一层的三块梯度  →  整条 MLP 的逐层回传  →  一次真的下降
        ↘                ↘                    ↘
      逐元素激活的导数    一张张量级计算图      与生产实现逐位对账
```

## 一、今天最值钱的一句话

> **反向传播 = 上游梯度 × 局部导数，然后累加。**

前半句是链式法则，后半句是纪律。真正会把人绊倒的是后半句：

```text
y = x·x        两条边都通向 x ⇒ dy/dx = x + x = 2x
写成 = 而不是 +=   它不报错，只会给出一个偏小的梯度
                  而"梯度偏小"看起来像"学习率设小了"——于是有人去调学习率
```

## 二、七条性质（判据分两类）

```text
逐点导数    五个逐元素激活的解析导数 vs 中心差分（day074 的尺子）
一个约定    relu 在 x = 0 处取次梯度 0：恒负输入的梯度**恰好**是 0
跨天对账①   softmax 雅可比 vs day074 的数值雅可比（并读 day074 的结论）
恒等式      (Jᵀv)_i = p_i(v_i − ⟨p,v⟩) 与显式矩阵乘一致（偏差 <= 1e-12）
跨天对账②   交叉熵梯度 = p − onehot，vs 对 sft.loss 的数值差分
整条网络    整个 MLP 的参数梯度（解析）vs 数值差分（相对误差 <= 1e-5）
跨天对账③   ffn_backward 与 encoder_decoder.layers.feed_forward_backward **逐位**一致
```

判据只有两类：**相等**（逐位 / 整数 / 计数）与**不超过某个上界**（相对误差）。
因此 :class:`GradientCheck` 带一个 ``upper_bound`` 字段——有它时是"≤"，没有时才是"=="。
把两类混成一个判据，就会出现"偏差恰好是 0（因为输入全是 0）被当成通过"这种事。

## 三、十个模块

```text
errors.py     八个失败族（**GradientError 回来了**：连续缺席八天之后第一次被真的抛出）
types.py      六个激活的导数公式 / 两个损失的梯度公式 / 逐算子规则表 / 七条性质 /
              十条笔记 / 五条边界 / torch 对照表 / 四个记录
gradients.py  局部导数（含 softmax 的显式雅可比与那条 O(n) 的 JVP）
graph.py      一张**张量级**计算图（自动微分：同一批梯度再算一遍）
layers.py     Dense 的三块梯度（dW、db、dx）+ 参数的压平与还原
network.py    MLP 的逐层回传 + 前馈（与 encoder_decoder 的接缝）
train.py      用解析反奔跑一次真的下降（优化器复用 day074）
verify.py     七条性质与两类判据（含三条跨天对账）
study.py      五张表（导数 / 一层反向 / 网络反向 / 校验 / 性质）
__init__.py   本文件
```

## 四、四条纪律（与前面各天逐字相同）

1. **一个量只写一遍**：``relu_derivative`` 只在 :mod:`gradients` 里实现一次；
   ``dense_backward`` 只在一个地方写；softmax 的反向只有那个 O(n) 的 JVP。
2. **读数必须现场算出**：``study`` 里不存数字；每一张表的每一行都来自函数调用。
3. **跨天对账必须调用别人的包**：三条对账分别调 ``math_foundations``（day073/074）、
   ``sft``（生产损失）、``encoder_decoder``（day079 的前馈反向）——自证是不成立的。
4. **缺席要可断言**：``RETURNED_FAMILY`` / ``ABSENT_FAMILY`` 都是常量，
   因此"这一天第一次没有缺席者"是一件可以被测试逐字钉住的事实，而不是一段散文。

## 五、与既有包的接缝（M8-D1 → M8-D2 的过渡）

- **上游（本包调用的真实实现）**：
  ``neural_basics``（day089：``activations`` / ``layers.Dense`` / ``losses`` / ``network``）、
  ``math_foundations``（day073/074：``calculus.gradient`` / ``jacobian`` / ``gradcheck`` /
  ``optim``）、``sft``（day050 起：``loss.cross_entropy``）、
  ``encoder_decoder``（day079：``layers.feed_forward`` / ``feed_forward_backward``）、
  ``transformer_core``（day075：``GradientError`` 等四族）；
- **传承**：day074 给了"数值差分是尺子、自动微分是推导的执行"；
  day089 把一条链的前向从最小零件搭起来——**今天在每个零件上各取一次导数**，
  再用 day074 的尺子量一遍、用 day079 的真实前馈反向对一遍；
- **脚下**：``config`` **没有**新增配置项——激活名、步长、容差、种子都是函数参数；
- **下游**：day092（优化器）会沿着同一条链问"往哪走、走多远"；
  今天只回答"**往哪走**"，而且要求它可被三条独立路径同时确认。
"""

from __future__ import annotations

from smart_research_agent.backprop import errors as _errors
from smart_research_agent.backprop import gradients as _gradients
from smart_research_agent.backprop import graph as _graph
from smart_research_agent.backprop import layers as _layers
from smart_research_agent.backprop import network as _network
from smart_research_agent.backprop import study as _study
from smart_research_agent.backprop import train as _train
from smart_research_agent.backprop import types as _types
from smart_research_agent.backprop import verify as _verify
from smart_research_agent.backprop.errors import (
    ABSENT_FAMILY,
    ABSENT_FAMILY_REASON,
    FAMILY_OUTCOMES,
    RETURNED_FAMILY,
    RETURNED_FAMILY_REASON,
    BackpropError,
    BackwardError,
    ChainError,
    GradientError,
    NumericError,
    ParameterError,
    ShapeError,
    StepError,
)
from smart_research_agent.backprop.gradients import (
    LOSS_GRADIENT_NAMES,
    SIGMOID_DERIVATIVE_CEILING,
    SQRT_TWO,
    SQRT_TWO_OVER_PI,
    SQRT_TWO_PI,
    activation_derivative,
    cross_entropy_grad,
    cross_entropy_grad_rows,
    elementwise_backward,
    gelu_derivative,
    gelu_tanh_derivative,
    leaky_relu_derivative,
    mse_grad,
    relu_derivative,
    sigmoid_derivative,
    sigmoid_derivative_from_output,
    softmax_backward_rows,
    softmax_jacobian,
    softmax_jacobian_vector_product,
    tanh_derivative,
    tanh_derivative_from_output,
)
from smart_research_agent.backprop.graph import (
    Node,
    Tensor,
    add_bias,
    add_values,
    constant,
    cross_entropy,
    dense,
    ensure_node,
    flatten_value,
    gelu,
    grad_of,
    is_matrix,
    is_scalar,
    is_vector,
    matmul,
    mse,
    ones_like,
    relu,
    softmax_rows,
    sum_all,
    tensor_shape,
    topological_order,
    variable,
    zeros_like,
)
from smart_research_agent.backprop.layers import (
    LayerShape,
    ParameterPair,
    affine_backward,
    dense_backward,
    flatten_parameters,
    gradient_summary,
    parameter_count,
    unflatten_parameters,
)
from smart_research_agent.backprop.network import (
    FFN_ACTIVATIONS,
    BackwardTrace,
    ForwardCache,
    activation_backward,
    build_chain,
    build_parameters,
    chain_forward,
    chain_forward_with_cache,
    chain_from_parameters,
    ffn_backward,
    ffn_forward,
    gradient_norm,
    mlp_backward,
    mlp_forward_with_cache,
    mlp_loss_gradients,
    parameter_objective,
    softmax_loss_gradients,
    spec_shapes,
)
from smart_research_agent.backprop.study import (
    DENSE_GRAD_OUTPUT,
    DENSE_INPUTS,
    DENSE_SAMPLE,
    DERIVATIVE_SAMPLE_X,
    PROPERTY_NAMES,
    SOFTMAX_SAMPLE_ROW,
    CheckRow,
    DerivativeRow,
    PropertyRow,
    check_rows,
    dense_layer_gradients,
    derivative_rows,
    network_layer_rows,
    note_lines,
    property_rows,
    study_lines,
)
from smart_research_agent.backprop.train import (
    DEFAULT_OPTIMIZER,
    TRAIN_LOSS,
    TrainReport,
    evaluate_loss,
    train_mlp,
)
from smart_research_agent.backprop.types import (
    ACTIVATION_DERIVATIVE_FORMULAS,
    BACKPROP_BOUNDARIES,
    BACKPROP_NOTES,
    BACKPROP_NOTES_ORDER,
    BACKPROP_PROPERTIES,
    BACKWARD_RULES,
    ELEMENTWISE_ACTIVATIONS,
    LOSSES,
    LOSS_CROSS_ENTROPY,
    LOSS_GRADIENT_FORMULAS,
    LOSS_MSE,
    PROPERTY_ACTIVATION_DERIVATIVES_MATCH_NUMERICAL,
    PROPERTY_CROSS_ENTROPY_GRADIENT_IS_P_MINUS_ONEHOT,
    PROPERTY_DESCRIPTIONS,
    PROPERTY_FFN_BACKWARD_MATCHES_ENCODER_DECODER,
    PROPERTY_FAILURE,
    PROPERTY_MLP_BACKWARD_MATCHES_NUMERICAL,
    PROPERTY_RELU_SUBGRADIENT_IS_ZERO,
    PROPERTY_SOFTMAX_JACOBIAN_MATCHES_NUMERICAL,
    PROPERTY_SOFTMAX_JVP_AVOIDS_MATRIX,
    TORCH_COUNTERPARTS,
    DenseGradients,
    FFNGradients,
    GradientReading,
    LayerGradients,
)
from smart_research_agent.backprop.verify import (
    CE_CASES,
    DERIVATIVE_POINTS,
    DERIVATIVE_TOLERANCE,
    DIFFERENCE_STEP,
    FFN_GRAD_OUTPUT,
    FFN_HIDDEN,
    FFN_INPUTS,
    FFN_SEED,
    IDENTITY_TOLERANCE,
    JVP_CASES,
    MLP_INPUTS,
    MLP_SPEC,
    MLP_TARGETS,
    NETWORK_TOLERANCE,
    GradientCheck,
    PropertyOutcome,
    PropertyReport,
    check_all,
    check_cross_entropy_gradient_is_p_minus_onehot,
    check_ffn_backward_matches_encoder_decoder,
    check_mlp_backward_matches_numerical,
    check_relu_subgradient_is_zero,
    check_softmax_jacobian_matches_numerical,
    check_softmax_jvp_avoids_matrix,
    difference_resolution,
    evaluate_derivatives_vs_numerical,
    max_scaled_gap,
    numerical_gradient,
)

#: 本包的九个功能模块（``__all__`` 由它们的公开名单合并而来——"一个量只写一遍"）。
_MODULES = (_errors, _types, _gradients, _graph, _layers, _network, _train, _verify, _study)

#: 本包的公开名单：**九个模块各自 ``__all__`` 的并集**，字母序。
#:
#: 故意不逐个手写：手写两份名单（一份 import、一份 ``__all__``）一定会分家，
#: 而"某一个名字在 ``__all__`` 里、却没有人真的导入它"这种失败
#: 在报告里长得和"它不存在"一模一样。下面那条导入期不变式代替人来核对这件事。
__all__ = sorted({name for module in _MODULES for name in module.__all__})

_MISSING = [name for name in __all__ if name not in globals()]
if _MISSING:  # pragma: no cover - 只在有人改名单时触发
    raise BackpropError(
        f"__all__ 里的 {_MISSING} 没有被真正导入："
        "``from backprop import *`` 会静默地少几个名字，而调用方只会看到 NameError。"
    )

_SUBMODULES = {"errors", "types", "gradients", "graph", "layers", "network", "train", "verify", "study"}
if _SUBMODULES & set(__all__):  # pragma: no cover - 只在有人改名单时触发
    raise BackpropError(
        f"__all__ 不能包含子模块名：{sorted(_SUBMODULES & set(__all__))}——"
        "否则 `from backprop import layers` 会拿到函数还是模块取决于导入顺序。"
    )
