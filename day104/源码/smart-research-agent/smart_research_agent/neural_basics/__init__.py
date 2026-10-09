"""``neural_basics``：从神经元到 FFN 的一条链（M8-D1 / day089）.

day088 把 M7 的十二块拼图串成了一张可校验的图；今天是 **M8（深度学习基础）的第一天**，
动作从"把既有的东西串起来"回到"从最小的零件搭起"：

```text
一个神经元  →  一层（Dense）  →  一个网络（MLP）  →  一个损失（MSE / CE）
                                          ↘  接回 Transformer 的 FFN（前馈块）
```

## 一、今天最值钱的一句话

> **深度本身不产生非线性：整条链上唯一让"多层"区别于"一层"的东西，是激活函数。**

这句话有一个立刻可用的推论，并写成了本课第 4 条性质：把恒等激活堆三层，
得到的仍是一个仿射映射（``mlp_forward`` 与 ``compose_affine`` 逐点相等）。
拿掉非线性，深度就退回一层——这不是一个比喻，是一次矩阵乘。

## 二、七条性质（判据分四类）

```text
有限性     六个激活在写死的网格（含 ±1000）上都有限
值域       激活的值域被遵守（relu≥0、sigmoid∈(0,1)、tanh∈(-1,1)、softmax 行和为 1）
跨天对账①  softmax 与 math_foundations.linalg.softmax **逐位**一致（day073）
结构事实   恒等激活的多层网络 == 合成后的单层仿射映射（逐点，容差 1e-12）
损失       mse 在预测等于目标时恰为 0.0，且等于逐元素平方差的均值
跨天对账②  CE 两条路径一致、CE ≥ 0，且与 sft.loss.cross_entropy **逐位**一致
跨天对账③  ffn_block 与 encoder_decoder.layers.feed_forward **逐点**一致（day079）
```

第二类是本课唯一一处"两个数相等"与"一个数不超过另一个数"混在一起的地方——
因此 ``CrossCheck`` 带一个 ``upper_bound`` 字段（与 day087 / day088 同源）：有它时判据是"≤"，
没有时才是"=="。把两类混成一个判据，就会出现"偏差恰好是 0（因为输入全是 0）被当成通过"这种事。

## 三、九个模块

```text
errors.py       七个失败族（**缺席的仍是 GradientError**，理由第五次换了一条：
                有前向、有损失、没有反向，而且没有交给自动微分）
types.py        六个激活 / 三个损失 / 五种初始化 / 七条性质 / 十条笔记 / 五条边界 /
                纯 Python ↔ PyTorch 对照表 / 五个记录
activations.py  六个激活的前向 + 三个数值稳定点（sigmoid 按符号分支、softmax 先减最大值、gelu 用精确 erf）
layers.py       Dense 层（W 形状 (out, in)、前向 x·Wᵀ + b）+ 五种初始化（LCG）+ 仿射合成
network.py      MLP 前向 / forward_trace / 恒等塌缩 / ffn_block（接回 Transformer 的前馈）
losses.py       mse / mae / cross_entropy（稳定路径）+ log_softmax + accuracy + perplexity
verify.py       七条性质与两类判据（含三条跨天对账）
study.py        五张表（激活 / 初始化 / 层次账 / 损失 / 性质）
__init__.py     本文件
```

## 四、四条纪律

1. **一个量只写一遍**：``softmax`` 只在 :mod:`activations` 里实现一次；
   ``log_softmax`` 只在 :mod:`losses` 里实现一次（另一条稳定路径，不是重复）。
2. **读数必须现场算出**：``study`` 里不存数字；每一张表的每一行都来自函数调用，
   某一天公式改了，读数会跟着改，而不会"悄悄地不再是真的"。
3. **跨天对账必须调用别人的包**：三条对账分别调 ``math_foundations``（day073）、
   ``sft``（day050 起）、``encoder_decoder``（day079）、``hf_source``（day085）——
   自证是不成立的，跨包对账才是。
4. **缺席要可断言**：``ABSENT_FAMILY`` 与 ``ABSENT_FAMILY_REASON`` 是常量，
   因此"本课没有 GradientError"是一件可以被测试逐字钉住的事实，而不是一段散文。

## 五、与既有包的接缝（M7 → M8 的过渡）

- **上游（本包调用的真实实现）**：
  ``math_foundations``（day073：``linalg.softmax`` / ``probability.cross_entropy``）、
  ``sft``（day050 起：``loss.softmax`` / ``loss.log_softmax`` / ``loss.cross_entropy``）、
  ``encoder_decoder``（day079：``layers.feed_forward`` / ``types.FFNWeights``）、
  ``hf_source``（day085：``blocks.gelu_exact`` / ``blocks.gelu_new``）、
  ``transformer_core``（day075：三族基础异常）；
- **传承**：day073 给数学地基、day075 给可微注意力、day085 读 HF 源码、
  day087 算推理三笔账、day088 把十二块拼图串成一张图——**今天把这些"零件"重新拆到最小**，
  从一个神经元重新搭起，再用第 7 条性质把 ``ffn_block`` 接回 day079 的前馈；
- **脚下**：``config`` **没有**新增配置项——激活名、初始化名、种子、容差都是函数参数；
- **下游**：day090（反向传播）会给这条链补上**唯一缺席的那一族** ``GradientError``——
  今天每一个前向算子都是可微的，但一行反向都没有。
"""

from __future__ import annotations

from smart_research_agent.neural_basics.activations import (
    ACTIVATION_RANGES,
    FINITE_GRID,
    GELU_TANH_INNER,
    RANGE_GRID,
    ActivationRange,
    activate,
    activation_at,
    activation_range,
    gelu,
    gelu_tanh,
    leaky_relu,
    relu,
    sigmoid,
    softmax,
    tanh,
)
from smart_research_agent.neural_basics.errors import (
    ABSENT_FAMILY,
    ABSENT_FAMILY_REASON,
    FAMILY_OUTCOMES,
    ActivationError,
    ForwardError,
    InitializationError,
    LossError,
    NeuralError,
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.neural_basics.layers import (
    LCG_INCREMENT,
    LCG_MODULUS,
    LCG_MULTIPLIER,
    Dense,
    affine_forward,
    compose_affine,
    dense_forward,
    dense_linear,
    he_scale,
    initialize,
    layer_accounting,
    lcg_stream,
    xavier_limit,
)
from smart_research_agent.neural_basics.losses import (
    PERPLEXITY_CEILING,
    accuracy,
    cross_entropy,
    cross_entropy_via_probability,
    log_softmax,
    mae,
    mse,
    perplexity,
)
from smart_research_agent.neural_basics.network import (
    FFN_RATIO,
    FFNParams,
    build_ffn_params,
    collapse_identity_mlp,
    ffn_block,
    ffn_ratio,
    forward_trace,
    identity_spec,
    mlp_forward,
    neuron_forward,
)
from smart_research_agent.neural_basics.study import (
    ACTIVATION_SAMPLE_X,
    ACTIVATION_SOFTMAX_ROW,
    INIT_SAMPLE,
    LAYER_SAMPLE,
    LOSS_LOGITS,
    LOSS_PRED,
    LOSS_TARGET,
    LOSS_TARGET_INDEX,
    ActivationRow,
    InitializationRow,
    LayerRow,
    LossRow,
    PropertyRow,
    activation_rows,
    initialization_rows,
    layer_rows,
    loss_rows,
    note_lines,
    property_rows,
    study_lines,
)
from smart_research_agent.neural_basics.types import (
    ACTIVATIONS,
    ACTIVATION_DESCRIPTIONS,
    ACTIVATION_PROFILES,
    ACT_GELU,
    ACT_LEAKY_RELU,
    ACT_RELU,
    ACT_SIGMOID,
    ACT_SOFTMAX,
    ACT_TANH,
    GELU_TANH_CUBIC,
    INITIALIZATIONS,
    INITIALIZATION_DESCRIPTIONS,
    INIT_HE,
    INIT_NORMAL,
    INIT_UNIFORM,
    INIT_XAVIER,
    INIT_ZEROS,
    LEAKY_SLOPE,
    LOSSES,
    LOSS_CROSS_ENTROPY,
    LOSS_DESCRIPTIONS,
    LOSS_MAE,
    LOSS_MSE,
    NEURAL_BOUNDARIES,
    NEURAL_NOTES,
    NEURAL_NOTES_ORDER,
    NEURAL_PROPERTIES,
    PROPERTY_ACTIVATIONS_ARE_FINITE,
    PROPERTY_CROSS_ENTROPY_PATHS_AGREE,
    PROPERTY_DESCRIPTIONS,
    PROPERTY_FFN_MATCHES_TRANSFORMER_STACK,
    PROPERTY_FAILURE,
    PROPERTY_IDENTITY_STACK_COLLAPSES,
    PROPERTY_MSE_IS_ZERO_AT_PERFECT,
    PROPERTY_RANGES_ARE_RESPECTED,
    PROPERTY_SOFTMAX_ROWS_ARE_DISTRIBUTIONS,
    TORCH_COUNTERPARTS,
    ActivationProfile,
    DenseSpec,
    ForwardTrace,
    LossReport,
    MLPSpec,
    NeuronSpec,
)
from smart_research_agent.neural_basics.verify import (
    CE_CASES,
    FFN_HIDDEN,
    FFN_INPUTS,
    FFN_SEED,
    IDENTITY_INPUTS,
    IDENTITY_SPEC,
    IDENTITY_TOLERANCE,
    MSE_PRED,
    MSE_TARGET,
    PATH_TOLERANCE,
    ROW_TOLERANCE,
    SOFTMAX_CASES,
    CrossCheck,
    PropertyOutcome,
    PropertyReport,
    check_activations_are_finite,
    check_all,
    check_cross_entropy_paths_agree,
    check_ffn_matches_transformer_stack,
    check_identity_stack_collapses,
    check_mse_is_zero_at_perfect,
    check_ranges_are_respected,
    check_softmax_rows_are_distributions,
)

__all__ = [
    "ABSENT_FAMILY",
    "ABSENT_FAMILY_REASON",
    "ACTIVATIONS",
    "ACTIVATION_DESCRIPTIONS",
    "ACTIVATION_PROFILES",
    "ACTIVATION_RANGES",
    "ACTIVATION_SAMPLE_X",
    "ACTIVATION_SOFTMAX_ROW",
    "ACT_GELU",
    "ACT_LEAKY_RELU",
    "ACT_RELU",
    "ACT_SIGMOID",
    "ACT_SOFTMAX",
    "ACT_TANH",
    "ActivationError",
    "ActivationProfile",
    "ActivationRange",
    "ActivationRow",
    "CE_CASES",
    "CrossCheck",
    "Dense",
    "DenseSpec",
    "FAMILY_OUTCOMES",
    "FFNParams",
    "FFN_HIDDEN",
    "FFN_INPUTS",
    "FFN_RATIO",
    "FFN_SEED",
    "FINITE_GRID",
    "ForwardError",
    "ForwardTrace",
    "GELU_TANH_CUBIC",
    "GELU_TANH_INNER",
    "IDENTITY_INPUTS",
    "IDENTITY_SPEC",
    "IDENTITY_TOLERANCE",
    "INITIALIZATIONS",
    "INITIALIZATION_DESCRIPTIONS",
    "INIT_HE",
    "INIT_NORMAL",
    "INIT_SAMPLE",
    "INIT_UNIFORM",
    "INIT_XAVIER",
    "INIT_ZEROS",
    "InitializationError",
    "InitializationRow",
    "LAYER_SAMPLE",
    "LCG_INCREMENT",
    "LCG_MODULUS",
    "LCG_MULTIPLIER",
    "LEAKY_SLOPE",
    "LOSSES",
    "LOSS_CROSS_ENTROPY",
    "LOSS_DESCRIPTIONS",
    "LOSS_LOGITS",
    "LOSS_MAE",
    "LOSS_MSE",
    "LOSS_PRED",
    "LOSS_TARGET",
    "LOSS_TARGET_INDEX",
    "LayerRow",
    "LossError",
    "LossReport",
    "LossRow",
    "MLPSpec",
    "MSE_PRED",
    "MSE_TARGET",
    "NEURAL_BOUNDARIES",
    "NEURAL_NOTES",
    "NEURAL_NOTES_ORDER",
    "NEURAL_PROPERTIES",
    "NeuralError",
    "NeuronSpec",
    "NumericError",
    "PATH_TOLERANCE",
    "PERPLEXITY_CEILING",
    "PROPERTY_ACTIVATIONS_ARE_FINITE",
    "PROPERTY_CROSS_ENTROPY_PATHS_AGREE",
    "PROPERTY_DESCRIPTIONS",
    "PROPERTY_FAILURE",
    "PROPERTY_FFN_MATCHES_TRANSFORMER_STACK",
    "PROPERTY_IDENTITY_STACK_COLLAPSES",
    "PROPERTY_MSE_IS_ZERO_AT_PERFECT",
    "PROPERTY_RANGES_ARE_RESPECTED",
    "PROPERTY_SOFTMAX_ROWS_ARE_DISTRIBUTIONS",
    "ParameterError",
    "PropertyOutcome",
    "PropertyReport",
    "PropertyRow",
    "RANGE_GRID",
    "ROW_TOLERANCE",
    "SOFTMAX_CASES",
    "ShapeError",
    "TORCH_COUNTERPARTS",
    "accuracy",
    "activate",
    "activation_at",
    "activation_range",
    "activation_rows",
    "affine_forward",
    "build_ffn_params",
    "check_activations_are_finite",
    "check_all",
    "check_cross_entropy_paths_agree",
    "check_ffn_matches_transformer_stack",
    "check_identity_stack_collapses",
    "check_mse_is_zero_at_perfect",
    "check_ranges_are_respected",
    "check_softmax_rows_are_distributions",
    "collapse_identity_mlp",
    "compose_affine",
    "cross_entropy",
    "cross_entropy_via_probability",
    "dense_forward",
    "dense_linear",
    "ffn_block",
    "ffn_ratio",
    "forward_trace",
    "gelu",
    "gelu_tanh",
    "he_scale",
    "identity_spec",
    "initialization_rows",
    "initialize",
    "layer_accounting",
    "layer_rows",
    "lcg_stream",
    "leaky_relu",
    "log_softmax",
    "loss_rows",
    "mae",
    "mlp_forward",
    "mse",
    "neuron_forward",
    "note_lines",
    "perplexity",
    "property_rows",
    "relu",
    "sigmoid",
    "softmax",
    "study_lines",
    "tanh",
    "xavier_limit",
]
