"""``conv_net`` 的口径表（day093 / M8-D4）.

一次性把这一课的"名词表"写全：**两种填充 / 两种池化 / 七条性质 / 十条笔记 /
五条边界 / 一张 PyTorch 对照表 / 三张公式**。全部是常量，因此可以被测试逐键检查。

```text
两种填充    valid（不填充） / same（步长 1 时补到同尺寸）
两种池化    max（保留窗口极值） / avg（窗口均值）
七条性质    与"定义"对账 4 条（滑窗点积 / 线性 / same / 输出尺寸）+ 梯度 1 条 + 池化 1 条 + 感受野 1 条
```

## 这一课与 day089 / day090 的边界

day089 的 ``Dense`` 是"**每个输入位置用一个独立的权重**"；day093 的卷积是
"**同一组权重在输入的不同位置被重复使用**"。这一句话就是本课全部内容的来源：

```text
权重共享   ⇒ 参数量从 O(H·W) 降到 O(k²)，且同一特征在全图可复用
局部连接   ⇒ 每个输出只看一个 k×k 的窗口（感受野）
```

因此本课复用 day074 的数值差分当尺子、复用 day090 的 ``flatten_parameters`` 口径压平参数、
复用 day092 的 ``TrainingOptimizer`` 做一次真的训练——**卷积本身只有一份实现**。
"""

from __future__ import annotations

from smart_research_agent.conv_net.errors import ConvError
from smart_research_agent.neural_basics.types import ACTIVATIONS

# --------------------------------------------------------------------------- #
# 闭合表 1：两种填充
# --------------------------------------------------------------------------- #

PADDING_VALID = "valid"
PADDING_SAME = "same"

#: 两种填充（顺序 = 不填充 → 补到同尺寸）.
PADDING_MODES: tuple[str, ...] = (PADDING_VALID, PADDING_SAME)

#: 每种填充的一句话解释.
PADDING_DESCRIPTIONS: dict[str, str] = {
    PADDING_VALID: "不填充：输出尺寸 = ⌊(H − k)/s⌋ + 1，特征图每经过一层都会缩小",
    PADDING_SAME: "补到同尺寸：步长 1 时两侧各补 ⌊(k−1)/2⌋（仅对奇数 k 保证精确同尺寸）",
}

#: 每种填充的公式.
PADDING_FORMULAS: dict[str, str] = {
    PADDING_VALID: "pad = 0",
    PADDING_SAME: "pad = (dilation·(k−1)) // 2（要求 stride = 1）",
}

# --------------------------------------------------------------------------- #
# 闭合表 2：两种池化
# --------------------------------------------------------------------------- #

POOL_MAX = "max"
POOL_AVG = "avg"

#: 两种池化（顺序 = 取极值 → 取均值）.
POOL_MODES: tuple[str, ...] = (POOL_MAX, POOL_AVG)

#: 每种池化的一句话解释.
POOL_DESCRIPTIONS: dict[str, str] = {
    POOL_MAX: "最大池化：窗口取最大值，保留最强的响应（对微小平移不敏感）",
    POOL_AVG: "平均池化：窗口取均值，保留整体强度（对噪声更平滑）",
}

#: 每种池化的输出尺寸公式（同卷积，核即窗口）.
POOL_FORMULAS: dict[str, str] = {
    POOL_MAX: "out = ⌊(H − k)/s⌋ + 1；窗口内取 max",
    POOL_AVG: "out = ⌊(H − k)/s⌋ + 1；窗口内取 mean",
}

# --------------------------------------------------------------------------- #
# 闭合表 3：三张公式
# --------------------------------------------------------------------------- #

#: 卷积 / 池化的输出尺寸公式（步长 s、填充 p、膨胀 d、核 k）.
OUTPUT_SIZE_FORMULA = "out = ⌊(H + 2p − d·(k−1) − 1)/s⌋ + 1"

#: 感受野的递推公式（从输入往输出累加）.
RECEPTIVE_FIELD_FORMULA = "r₀ = 1；r_{i+1} = r_i + (k_{i+1} − 1)·∏_{j≤i} s_j"

#: 卷积层参数量公式（含偏置）.
CONV_PARAM_FORMULA = "params = C_out·(C_in·k_h·k_w) + C_out"

#: 卷积层可用的激活名（**是 day089 那张 ACTIVATIONS 表的子集**）.
#: 刻意不含 ``softmax``：它是"按行归一成概率"的激活，用在特征图上会破坏空间结构；
#: 恒等（不激活）用 ``None`` 表示，与 day089 ``Dense(activation=None)`` 的口径一致。
CONV_ACTIVATIONS: tuple[str, ...] = ("relu", "leaky_relu", "sigmoid", "tanh", "gelu")

if not set(CONV_ACTIVATIONS) <= set(ACTIVATIONS):  # pragma: no cover - 导入期不变式
    raise ConvError(
        "卷积可用的激活名必须是 day089 ACTIVATIONS 表的子集："
        "自造一个激活名会让它在 forward 里静默地走不到任何分支。"
    )

# --------------------------------------------------------------------------- #
# 闭合表 4：七条性质
# --------------------------------------------------------------------------- #

PROPERTY_CONV_MATCHES_SLIDING_DOT = "conv_matches_sliding_dot"
PROPERTY_CONV_IS_LINEAR = "conv_is_linear"
PROPERTY_SAME_PADDING_PRESERVES_SIZE = "same_padding_preserves_size"
PROPERTY_OUTPUT_SIZE_FORMULA_MATCHES_SHAPE = "output_size_formula_matches_shape"
PROPERTY_CONV_BACKWARD_MATCHES_NUMERICAL = "conv_backward_matches_numerical"
PROPERTY_POOL_RETURNS_WINDOW_EXTREME = "pool_returns_window_extreme"
PROPERTY_RECEPTIVE_FIELD_MATCHES_DIRECT = "receptive_field_matches_direct"

#: 七条性质（顺序 = 从"定义对不对"到"反向与感受野对不对"）.
CONV_PROPERTIES: tuple[str, ...] = (
    PROPERTY_CONV_MATCHES_SLIDING_DOT,
    PROPERTY_CONV_IS_LINEAR,
    PROPERTY_SAME_PADDING_PRESERVES_SIZE,
    PROPERTY_OUTPUT_SIZE_FORMULA_MATCHES_SHAPE,
    PROPERTY_CONV_BACKWARD_MATCHES_NUMERICAL,
    PROPERTY_POOL_RETURNS_WINDOW_EXTREME,
    PROPERTY_RECEPTIVE_FIELD_MATCHES_DIRECT,
)

#: 每条性质在讲什么（一句话）.
PROPERTY_DESCRIPTIONS: dict[str, str] = {
    PROPERTY_CONV_MATCHES_SLIDING_DOT: "conv2d 的输出与手写滑窗点积逐位一致",
    PROPERTY_CONV_IS_LINEAR: "卷积是线性算子：conv(αx + βy) = α·conv(x) + β·conv(y)",
    PROPERTY_SAME_PADDING_PRESERVES_SIZE: "same 填充（奇数核、步长 1）让输出与输入同尺寸",
    PROPERTY_OUTPUT_SIZE_FORMULA_MATCHES_SHAPE: "尺寸公式给出的数与实际特征图形状一致",
    PROPERTY_CONV_BACKWARD_MATCHES_NUMERICAL: "卷积解析梯度（dK / dB / dX）与数值差分一致",
    PROPERTY_POOL_RETURNS_WINDOW_EXTREME: "池化的每个输出都真的是那个窗口的极值 / 均值",
    PROPERTY_RECEPTIVE_FIELD_MATCHES_DIRECT: "感受野递推公式与逐层直接累加的结果一致",
}

#: 每条性质"失败意味着什么"（**不通过时要去看哪里**）.
PROPERTY_FAILURE: dict[str, str] = {
    PROPERTY_CONV_MATCHES_SLIDING_DOT: "滑窗的边界或翻转被写错了（卷积核没反，只是取值错了位置）",
    PROPERTY_CONV_IS_LINEAR: "偏置被当成了非线性项，或填充引入了随位置变化的贡献",
    PROPERTY_SAME_PADDING_PRESERVES_SIZE: "same 的补边数算错（奇数核两侧补 ⌊(k−1)/2⌋）",
    PROPERTY_OUTPUT_SIZE_FORMULA_MATCHES_SHAPE: "公式与实际取窗口的方式分家（floor 用错）",
    PROPERTY_CONV_BACKWARD_MATCHES_NUMERICAL: "dK 或 dX 的索引写错了（这两块最容易互相串）",
    PROPERTY_POOL_RETURNS_WINDOW_EXTREME: "池化窗口起点 / 步长写错，取到了相邻窗口的元素",
    PROPERTY_RECEPTIVE_FIELD_MATCHES_DIRECT: "步长的乘积项漏乘了某一层",
}

if not (
    set(CONV_PROPERTIES) == set(PROPERTY_DESCRIPTIONS) == set(PROPERTY_FAILURE)
):  # pragma: no cover
    raise ConvError("七条性质的三张表不一致：名单 / 说明 / 失败意味着什么必须逐键对齐。")

# --------------------------------------------------------------------------- #
# 闭合表 5：十条笔记 / 五条边界 / 一张 PyTorch 对照表
# --------------------------------------------------------------------------- #

CONV_NOTES_ORDER: tuple[str, ...] = (
    "weight_sharing",
    "local_connection",
    "receptive_field",
    "output_size",
    "padding_same",
    "pooling_role",
    "channel_mixing",
    "backward_shapes",
    "linearity",
    "preprocessing",
)

#: 十条笔记（键 -> 一句话）.
CONV_NOTES: dict[str, str] = {
    "weight_sharing": "卷积的全部秘密是一句话：**同一组权重在输入的不同位置被重复使用**。",
    "local_connection": "每个输出只看一个 k×k 窗口：局部连接换来了平移等变性与更少的参数。",
    "receptive_field": "感受野随层数增长：堆两层 3×3 的感受野是 5×5，而参数比一层 5×5 少。",
    "output_size": "输出尺寸是一个 floor：⌊(H + 2p − d(k−1) − 1)/s⌋ + 1，先算它再谈别的。",
    "padding_same": "same 只在步长 1、奇数核时精确成立：补边数 = ⌊(k−1)/2⌋。",
    "pooling_role": "池化的作用是**降采样**：它让后面的层用更少的数看到更大的范围。",
    "channel_mixing": "多通道卷积在**通道维**上也做加权求和：C_in → C_out 是一次全连接式的混合。",
    "backward_shapes": "卷积反向有三块：dK、dB、dX；dK 与 dX 的形状最容易互相写串。",
    "linearity": "卷积是线性算子（不含激活时）：conv(αx+βy) = α·conv(x)+β·conv(y)。",
    "preprocessing": "CNN 在本项目里的位置是**视觉预处理**：把图像变成特征，再交给上层的 Agent。",
}

#: 五条边界（**这一课明确不承诺的事**）.
CONV_BOUNDARIES: tuple[str, ...] = (
    "只用单/多通道的二维卷积：不做 3D 卷积、分组卷积、空洞卷积的工程优化。",
    "不做 GPU / 向量化：纯 Python 的滑窗实现，为的是可读与可对账，不是速度。",
    "不做批量（batch）维：一次前向处理一张图（批量是工程上的循环，不是数学）。",
    "不做池化的反向之外的复杂反向：卷积 / 池化 / 全连接三块，不做残差与跳连。",
    "不新增第三方依赖：全部纯标准库实现，不 import torch / numpy。",
)

#: 纯 Python ↔ PyTorch 对照表（**只核对语义，本仓库不安装也不调用 torch**）.
TORCH_COUNTERPARTS: dict[str, str] = {
    "conv2d": "torch.nn.functional.conv2d(input, weight, bias, stride, padding, dilation)",
    "conv_layer": "torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)",
    "max_pool2d": "torch.nn.functional.max_pool2d(x, kernel_size, stride)",
    "avg_pool2d": "torch.nn.functional.avg_pool2d(x, kernel_size, stride)",
    "padding": "torch.nn.functional.pad(x, (left, right, top, bottom))",
    "conv_backward": "weight.grad / bias.grad / input.grad（由 autograd 自动求出）",
    "receptive_field": "没有现成 API，按同一个递推公式手算",
    "weight_sharing": "nn.Conv2d 的 weight 形状 (C_out, C_in, k_h, k_w)——一组核在全图复用",
    "flatten": "torch.flatten(x, start_dim=1) 之后接 nn.Linear",
    "optimizer": "torch.optim.Adam(model.parameters(), lr)（day092 的同一套更新规则）",
}

#: 这一课的性质名单（供报告核对：表里的行数必须等于它）.
PROPERTIES = CONV_PROPERTIES

__all__ = [
    "CONV_ACTIVATIONS",
    "CONV_BOUNDARIES",
    "CONV_NOTES",
    "CONV_NOTES_ORDER",
    "CONV_PARAM_FORMULA",
    "CONV_PROPERTIES",
    "OUTPUT_SIZE_FORMULA",
    "PADDING_DESCRIPTIONS",
    "PADDING_FORMULAS",
    "PADDING_MODES",
    "PADDING_SAME",
    "PADDING_VALID",
    "POOL_AVG",
    "POOL_DESCRIPTIONS",
    "POOL_FORMULAS",
    "POOL_MAX",
    "POOL_MODES",
    "PROPERTIES",
    "PROPERTY_CONV_BACKWARD_MATCHES_NUMERICAL",
    "PROPERTY_CONV_IS_LINEAR",
    "PROPERTY_CONV_MATCHES_SLIDING_DOT",
    "PROPERTY_DESCRIPTIONS",
    "PROPERTY_FAILURE",
    "PROPERTY_OUTPUT_SIZE_FORMULA_MATCHES_SHAPE",
    "PROPERTY_POOL_RETURNS_WINDOW_EXTREME",
    "PROPERTY_RECEPTIVE_FIELD_MATCHES_DIRECT",
    "PROPERTY_SAME_PADDING_PRESERVES_SIZE",
    "RECEPTIVE_FIELD_FORMULA",
    "TORCH_COUNTERPARTS",
]
