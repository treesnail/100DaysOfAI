"""``neural_basics`` 的常量表与记录（day089 / M8-D1）.

day088 把 M7 的十二块拼图串成了一张图；今天是 **M8（深度学习基础）的第一天**，
动作从"串起来"回到"从最小的零件搭起"：

```text
一个神经元  →  一层（Dense）  →  一个网络（MLP）  →  一个损失（MSE / CE）
                                          ↘  接回 Transformer 的 FFN（前馈块）
```

本模块只放**口径**：六个激活、三个损失、四种初始化、七条性质、十条笔记、五条边界，
以及那条链上的五个记录。算术一律在别的模块里（``activations`` / ``layers`` /
``network`` / ``losses``）——"一个量只写一遍"。

## 一、六个激活的口径表（**是否饱和 / 是否零中心 / 值域 / 公式**）

```text
relu        max(0, x)                                     不饱和、非零中心、[0, +∞)
leaky_relu  x if x > 0 else 0.01x                         不饱和、非零中心、(-∞, +∞)
sigmoid     1 / (1 + e^{-x})                              饱和、  非零中心、(0, 1)
tanh        (e^x - e^{-x}) / (e^x + e^{-x})               饱和、  零中心、 (-1, 1)
gelu        0.5x(1 + erf(x/√2))                           不饱和、非零中心、[-0.17, +∞)
softmax     e^{z_i - max} / Σ_j e^{z_j - max}             不饱和、非零中心、(0, 1) 且行和为 1
```

## 二、一条纪律：**torch 只对照、不调用**

本仓库全程纯 Python（**不依赖 torch，也不依赖 numpy**）。因此 PyTorch 的语义只写在
:data:`TORCH_COUNTERPARTS` 这张对照表里——它对着**公式与语义**核对，
不是一次"实测 torch 输出"。本课**不声称**任何 torch 版本号；具体版本语义以官方文档为准。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from smart_research_agent.neural_basics.errors import (
    ForwardError,
    NumericError,
    ParameterError,
)

# --------------------------------------------------------------------------------------
# 1. 六个激活
# --------------------------------------------------------------------------------------

ACT_RELU = "relu"
ACT_LEAKY_RELU = "leaky_relu"
ACT_SIGMOID = "sigmoid"
ACT_TANH = "tanh"
ACT_GELU = "gelu"
ACT_SOFTMAX = "softmax"

#: 六个激活（**顺序 = 口径表的顺序 = study 表的打印顺序**）.
ACTIVATIONS: tuple[str, ...] = (
    ACT_RELU,
    ACT_LEAKY_RELU,
    ACT_SIGMOID,
    ACT_TANH,
    ACT_GELU,
    ACT_SOFTMAX,
)

#: ``leaky_relu`` 的默认斜率（写进常量，让"0.01"是一个可断言的事实）.
LEAKY_SLOPE = 0.01

#: ``gelu_new``（tanh 近似）的三次项系数——与 day085 的 ``hf_source.blocks.GELU_NEW_CUBIC`` 同源.
GELU_TANH_CUBIC = 0.044715

ACTIVATION_DESCRIPTIONS: dict[str, str] = {
    ACT_RELU: "ReLU：max(0, x)——负半轴恒为 0，因此它是**分段线性**的，且非零中心",
    ACT_LEAKY_RELU: f"LeakyReLU：x>0 时是 x、x<=0 时是 {LEAKY_SLOPE}·x——给负半轴留一条小坡度",
    ACT_SIGMOID: "Sigmoid：1/(1+e^{-x})——把它写成**按符号分支**两条式子，两端都不会上溢",
    ACT_TANH: "Tanh：(e^x-e^{-x})/(e^x+e^{-x})——零中心、值域 (-1, 1)，是唯一零中心的激活",
    ACT_GELU: "GELU：0.5x(1+erf(x/√2))——本课采用**精确 erf 式**（tanh 近似另列为 gelu_tanh）",
    ACT_SOFTMAX: "Softmax：按行归一化的指数——先减最大值，因此它对整体平移不变、且不上溢",
}


@dataclass(frozen=True)
class ActivationProfile:
    """一个激活的口径：是否饱和 / 是否零中心 / 值域 / 公式（**四列都要有**）."""

    saturating: bool
    zero_centered: bool
    value_range: str
    formula: str

    def line(self) -> str:
        """一行口径：``饱和 否 | 零中心 是 | 值域 (-1, 1) | 公式 ...``."""
        saturated = "是" if self.saturating else "否"
        centered = "是" if self.zero_centered else "否"
        return (
            f"饱和 {saturated} | 零中心 {centered} | 值域 {self.value_range} | 公式 {self.formula}"
        )


ACTIVATION_PROFILES: dict[str, ActivationProfile] = {
    ACT_RELU: ActivationProfile(
        saturating=False,
        zero_centered=False,
        value_range="[0, +inf)",
        formula="max(0, x)",
    ),
    ACT_LEAKY_RELU: ActivationProfile(
        saturating=False,
        zero_centered=False,
        value_range="(-inf, +inf)",
        formula=f"x if x > 0 else {LEAKY_SLOPE}·x",
    ),
    ACT_SIGMOID: ActivationProfile(
        saturating=True,
        zero_centered=False,
        value_range="(0, 1)",
        formula="1 / (1 + e^{-x})",
    ),
    ACT_TANH: ActivationProfile(
        saturating=True,
        zero_centered=True,
        value_range="(-1, 1)",
        formula="(e^x - e^{-x}) / (e^x + e^{-x})",
    ),
    ACT_GELU: ActivationProfile(
        saturating=False,
        zero_centered=False,
        value_range="[-0.170, +inf)",
        formula="0.5x(1 + erf(x/√2))",
    ),
    ACT_SOFTMAX: ActivationProfile(
        saturating=False,
        zero_centered=False,
        value_range="(0, 1)（每行和为 1）",
        formula="e^{z_i - max} / Σ_j e^{z_j - max}",
    ),
}

# --------------------------------------------------------------------------------------
# 2. 三个损失
# --------------------------------------------------------------------------------------

LOSS_MSE = "mse"
LOSS_MAE = "mae"
LOSS_CROSS_ENTROPY = "cross_entropy"

#: 三个损失（**顺序 = 损失表的顺序**）.
LOSSES: tuple[str, ...] = (LOSS_MSE, LOSS_MAE, LOSS_CROSS_ENTROPY)

LOSS_DESCRIPTIONS: dict[str, str] = {
    LOSS_MSE: "均方误差：mean((pred − target)²)——预测等于目标时**恰好**为 0.0",
    LOSS_MAE: "平均绝对误差：mean(|pred − target|)——对离群点比 MSE 温和",
    LOSS_CROSS_ENTROPY: "交叉熵：−log softmax(logits)[target]——从 logits 走稳定路径，标签越界当场拒绝",
}

# --------------------------------------------------------------------------------------
# 3. 四种初始化（外加可选的 He）
# --------------------------------------------------------------------------------------

INIT_ZEROS = "zeros"
INIT_UNIFORM = "uniform"
INIT_NORMAL = "normal"
INIT_XAVIER = "xavier"
INIT_HE = "he"

#: 五种初始化（**顺序 = 初始化表的顺序**）.
INITIALIZATIONS: tuple[str, ...] = (
    INIT_ZEROS,
    INIT_UNIFORM,
    INIT_NORMAL,
    INIT_XAVIER,
    INIT_HE,
)

INITIALIZATION_DESCRIPTIONS: dict[str, str] = {
    INIT_ZEROS: "全零：W = 0、b = 0——它让每一个神经元**完全相同**（对称性不破），只适合对账",
    INIT_UNIFORM: "均匀：U(−1/√fan_in, 1/√fan_in)——尺度只看 fan_in，与 nn.Linear 的默认同阶",
    INIT_NORMAL: "正态：N(0, 1/√fan_in)——用 Box-Muller 从 LCG 造，因此可复现",
    INIT_XAVIER: "Xavier：U(−a, a)、a = √(6/(fan_in + fan_out))——fan_in + fan_out = 0 时无定义",
    INIT_HE: "He：N(0, √(2/fan_in))——为 ReLU 一类非对称激活设计的下游尺度",
}

# --------------------------------------------------------------------------------------
# 4. 七条性质（**判据分四类**）
# --------------------------------------------------------------------------------------

PROPERTY_ACTIVATIONS_ARE_FINITE = "activations_are_finite"
PROPERTY_RANGES_ARE_RESPECTED = "activation_ranges_are_respected"
PROPERTY_SOFTMAX_ROWS_ARE_DISTRIBUTIONS = "softmax_rows_are_distributions"
PROPERTY_IDENTITY_STACK_COLLAPSES = "identity_stack_collapses_to_affine"
PROPERTY_MSE_IS_ZERO_AT_PERFECT = "mse_is_zero_at_perfect"
PROPERTY_CROSS_ENTROPY_PATHS_AGREE = "cross_entropy_paths_agree"
PROPERTY_FFN_MATCHES_TRANSFORMER_STACK = "ffn_matches_transformer_stack"

#: 七条性质（**顺序 = verify 报告的顺序**）.
NEURAL_PROPERTIES: tuple[str, ...] = (
    PROPERTY_ACTIVATIONS_ARE_FINITE,
    PROPERTY_RANGES_ARE_RESPECTED,
    PROPERTY_SOFTMAX_ROWS_ARE_DISTRIBUTIONS,
    PROPERTY_IDENTITY_STACK_COLLAPSES,
    PROPERTY_MSE_IS_ZERO_AT_PERFECT,
    PROPERTY_CROSS_ENTROPY_PATHS_AGREE,
    PROPERTY_FFN_MATCHES_TRANSFORMER_STACK,
)

PROPERTY_DESCRIPTIONS: dict[str, str] = {
    PROPERTY_ACTIVATIONS_ARE_FINITE: "六个激活在写死的网格（含 ±1000）上都有限，不出现 nan / inf",
    PROPERTY_RANGES_ARE_RESPECTED: "激活的值域被遵守：relu≥0、sigmoid∈(0,1)、tanh∈(-1,1)、softmax 行和为 1",
    PROPERTY_SOFTMAX_ROWS_ARE_DISTRIBUTIONS: "**跨天对账**：与 ``math_foundations.softmax`` 在同一输入上逐位一致",
    PROPERTY_IDENTITY_STACK_COLLAPSES: "恒等激活的多层网络 == 合成后的单层仿射映射（逐点，容差 1e-12）",
    PROPERTY_MSE_IS_ZERO_AT_PERFECT: "``mse`` 在预测等于目标时恰为 0.0，且等于逐元素平方差的均值",
    PROPERTY_CROSS_ENTROPY_PATHS_AGREE: "**跨天对账**：CE 的两条路径一致且 CE ≥ 0，并与 ``sft.loss.cross_entropy`` 一致",
    PROPERTY_FFN_MATCHES_TRANSFORMER_STACK: "**跨天对账**：本包 ``ffn_block`` 与 ``encoder_decoder.layers.feed_forward`` 逐点一致",
}

PROPERTY_FAILURE: dict[str, str] = {
    PROPERTY_ACTIVATIONS_ARE_FINITE: "sigmoid / softmax 没做数值稳定（exp 上溢）⇒ 极端点上出现 inf 或 nan",
    PROPERTY_RANGES_ARE_RESPECTED: "值域被破坏——多半是公式抄错了符号，而'越界一个点'在图上几乎看不出来",
    PROPERTY_SOFTMAX_ROWS_ARE_DISTRIBUTIONS: "本包的 softmax 与 day073 的口径分家（平移不变性或多减了一个数）",
    PROPERTY_IDENTITY_STACK_COLLAPSES: "多层的仿射没塌缩成一层——说明某处偷偷插了一个非线性",
    PROPERTY_MSE_IS_ZERO_AT_PERFECT: "损失里少了平方、或分了两次母——'恰好 0'这条判据会第一个发现",
    PROPERTY_CROSS_ENTROPY_PATHS_AGREE: "把概率当 logits、或 log_softmax 少减了一次最大值",
    PROPERTY_FFN_MATCHES_TRANSFORMER_STACK: "本包的前馈与项目里真实的 FFN 口径不同（权重转置、激活不同）",
}

# --------------------------------------------------------------------------------------
# 5. 十条笔记（这是"从神经元到 FFN"这条链的成品）
# --------------------------------------------------------------------------------------

NEURAL_NOTES: dict[str, str] = {
    "neuron_is_a_dot_plus_bias": (
        "一个神经元 = 一次点积 + 一个偏置 + 一个非线性。"
        "没有非线性时，再多层也只是**一个**仿射映射——这正是第 4 条性质要钉的事实。"
    ),
    "activation_is_the_only_nonlinearity": (
        "整条链上唯一产生非线性的地方是激活函数。"
        "拿掉它，网络的表达力退回一层；保留它，深度才开始有意义。"
    ),
    "sigmoid_and_tanh_saturate": (
        "sigmoid 与 tanh 在两端饱和：导数趋近 0。"
        "这不是「数值误差」，而是它们作为激活在两端的**定义域代价**——因此才有 relu 一族。"
    ),
    "relu_is_not_zero_centered": (
        "relu 的输出恒非负，因此不是零中心：下一层的输入会整体偏正。"
        "tanh 是这六个里唯一零中心的——这条口径差是「选哪个激活」的第一判据。"
    ),
    "softmax_subtracts_the_max": (
        "softmax 先减最大值：这是**恒等变形**（分子分母同乘 e^{-max}），"
        "它不改变数学结果，只把 exp 的自变量压到 ≤ 0——不做这一步，z=1000 会去算 e^{1000}"
        "（CPython 抛 OverflowError，numpy / torch 里是 inf 再 inf/inf=nan）。"
    ),
    "cross_entropy_prefers_log_softmax": (
        "交叉熵走 log_softmax：−log(softmax) 在概率极小时先下溢成 0、再取 log 得到 -inf；"
        "log_softmax 在同样的输入下仍给出有限值。两条路径在正常区间上一致，在极端点上分家。"
    ),
    "mse_is_exactly_zero_at_perfect": (
        "MSE 在预测等于目标时**恰好**是 0.0（不是「接近 0」）。"
        "这条「恰好」是一条硬判据：任何多余的正则项或错误的分母都会让它不再是 0。"
    ),
    "xavier_divides_by_fan_sum": (
        "Xavier 的尺度是 √(6/(fan_in + fan_out))：它同时看两侧的宽度。"
        "分母为 0 时这个式子没有定义——本包当场抛 InitializationError，而不是崩在别处。"
    ),
    "identity_stack_collapses": (
        "把恒等激活堆三层，得到的仍是一个仿射映射："
        "W = W₃·W₂·W₁、b = W₃·W₂·b₁ + W₃·b₂ + b₃。这条塌缩不是一个比喻，是一次矩阵乘。"
    ),
    "ffn_is_dense_act_dense": (
        "Transformer 的前馈块就是 Dense → 激活 → Dense（先扩张 4 倍再压回）。"
        "它与本包的 ffn_block 落在**同一个值**上——这条对账把「神经网络基础」接回了 M7。"
    ),
}

#: 十条笔记的键（顺序即写入顺序，报告里读它）.
NEURAL_NOTES_ORDER: tuple[str, ...] = tuple(NEURAL_NOTES)

# --------------------------------------------------------------------------------------
# 6. 边界（**这一课明确不承诺的事**）
# --------------------------------------------------------------------------------------

NEURAL_BOUNDARIES: tuple[str, ...] = (
    "本包实现的是**前向与损失**，一行反向都没有：反向传播是 day090 的主题",
    "本包不安装、也不调用 torch / numpy：PyTorch 的语义只写在 TORCH_COUNTERPARTS 对照表里，"
    "对照的是公式与语义，不是「实测 torch 输出」；具体版本以官方文档为准",
    "六个激活只覆盖课程口径下的那一组（relu / leaky_relu / sigmoid / tanh / gelu / softmax），"
    "不是「所有激活函数」",
    "权重由 LCG 生成、输入写死，因此本包复现的是**读数**、不是统计规律；"
    "它不承诺任何「训练效果」",
    "本包不承诺浮点意义上的「位逐位跨平台一致」——跨天对账的容差由各自的口径给出",
)

# --------------------------------------------------------------------------------------
# 7. 纯 Python ↔ PyTorch 对照表（**只对照，不调用**）
# --------------------------------------------------------------------------------------

#: 纯 Python 实现 ↔ PyTorch API 的对照（例：``"dense_forward": "torch.nn.Linear"``）.
#: 只核对**公式与语义**，不声明任何 torch 版本号。
TORCH_COUNTERPARTS: dict[str, str] = {
    "neuron": "单个神经元 ≈ torch.nn.Linear(1, 1)（含偏置）",
    "dense_forward": "torch.nn.Linear（x·Wᵀ + b，权重形状 (out, in)）",
    "dense_bias": "nn.Linear(bias=True)（本包默认带偏置）",
    "relu": "torch.nn.functional.relu / nn.ReLU",
    "leaky_relu": "torch.nn.functional.leaky_relu / nn.LeakyReLU",
    "sigmoid": "torch.sigmoid / torch.nn.functional.sigmoid / nn.Sigmoid",
    "tanh": "torch.tanh / torch.nn.functional.tanh / nn.Tanh",
    "gelu": "torch.nn.functional.gelu / nn.GELU",
    "softmax": "torch.nn.functional.softmax(dim=-1) / nn.Softmax",
    "log_softmax": "torch.nn.functional.log_softmax(dim=-1) / nn.LogSoftmax",
    "mse": "torch.nn.functional.mse_loss / nn.MSELoss",
    "mae": "torch.nn.functional.l1_loss / nn.L1Loss",
    "cross_entropy": "torch.nn.functional.cross_entropy / nn.CrossEntropyLoss（从 logits 出发）",
    "accuracy": "torch.argmax(logits) == target（未内置损失，需手写）",
    "perplexity": "torch.exp(loss)",
    "zeros_init": "torch.nn.init.zeros_",
    "uniform_init": "torch.nn.init.uniform_",
    "normal_init": "torch.nn.init.normal_",
    "xavier_init": "torch.nn.init.xavier_uniform_",
    "he_init": "torch.nn.init.kaiming_normal_（mode='fan_in', nonlinearity='relu'）",
}

# --------------------------------------------------------------------------------------
# 8. 记录
# --------------------------------------------------------------------------------------


def _require_positive_int(value: object, *, name: str) -> int:
    """正整数护栏（``bool`` 被显式拒绝，因为 ``True == 1`` 会偷偷溜过）."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ParameterError(f"{name} 必须是正整数，收到 {value!r}：宽度不能是 0 或非整数。")
    return value


def _require_activation(value: object, *, name: str) -> str | None:
    """激活名护栏：允许 ``None``（恒等），其余必须在六个之列."""
    if value is None:
        return None
    if value not in ACTIVATIONS:
        raise ParameterError(f"{name} 未知的激活 {value!r}：可选 {list(ACTIVATIONS)} 或 None（恒等）。")
    return str(value)


def _require_initialization(value: object) -> str:
    """初始化名护栏."""
    if value not in INITIALIZATIONS:
        raise ParameterError(f"未知的初始化 {value!r}：可选 {list(INITIALIZATIONS)}。")
    return str(value)


@dataclass(frozen=True)
class NeuronSpec:
    """一个神经元：``inputs`` 个输入 → 一次点积 + 偏置 → 一个激活."""

    inputs: int
    activation: str | None = ACT_RELU
    init: str = INIT_ZEROS
    seed: int = 0

    def __post_init__(self) -> None:
        _require_positive_int(self.inputs, name="NeuronSpec.inputs")
        _require_activation(self.activation, name="NeuronSpec.activation")
        _require_initialization(self.init)

    @property
    def parameter_count(self) -> int:
        """参数个数 = ``inputs`` 个权重 + 1 个偏置."""
        return self.inputs + 1

    def to_dict(self) -> dict[str, object]:
        """摊平成一行字段."""
        return {
            "inputs": self.inputs,
            "activation": self.activation,
            "init": self.init,
            "seed": self.seed,
            "parameter_count": self.parameter_count,
        }

    def line(self) -> str:
        """一行说明：``neuron(3→1) | 激活 relu | 初始化 zeros | 参数 4``."""
        act = self.activation if self.activation is not None else "identity"
        return f"neuron({self.inputs}→1) | 激活 {act} | 初始化 {self.init} | 参数 {self.parameter_count}"


@dataclass(frozen=True)
class DenseSpec:
    """一层全连接：权重形状 ``(out_features, in_features)``，前向 ``x·Wᵀ + b``（与 ``nn.Linear`` 一致）."""

    in_features: int
    out_features: int
    activation: str | None = ACT_RELU
    init: str = INIT_XAVIER
    seed: int = 0

    def __post_init__(self) -> None:
        _require_positive_int(self.in_features, name="DenseSpec.in_features")
        _require_positive_int(self.out_features, name="DenseSpec.out_features")
        _require_activation(self.activation, name="DenseSpec.activation")
        _require_initialization(self.init)

    @property
    def weight_count(self) -> int:
        """权重元素个数 = ``in_features × out_features``."""
        return self.in_features * self.out_features

    @property
    def parameter_count(self) -> int:
        """参数总数 = ``in_features × out_features + out_features``（权重 + 偏置）."""
        return self.weight_count + self.out_features

    def to_dict(self) -> dict[str, object]:
        """摊平成一行字段."""
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "activation": self.activation,
            "init": self.init,
            "seed": self.seed,
            "parameter_count": self.parameter_count,
        }

    def line(self) -> str:
        """一行说明：``dense(3→4) | 激活 relu | 初始化 xavier | 参数 16``."""
        act = self.activation if self.activation is not None else "identity"
        return (
            f"dense({self.in_features}→{self.out_features}) | 激活 {act} | "
            f"初始化 {self.init} | 参数 {self.parameter_count}"
        )


@dataclass(frozen=True)
class MLPSpec:
    """一个前馈网络：一串 ``DenseSpec``（相邻两层的宽度必须接得上）."""

    layers: tuple[DenseSpec, ...]

    def __post_init__(self) -> None:
        if not self.layers:
            raise ParameterError("MLPSpec.layers 不能为空：一个网络至少也要有一层。")
        for index, layer in enumerate(self.layers):
            if not isinstance(layer, DenseSpec):
                raise ParameterError(f"MLPSpec 的第 {index} 层不是 DenseSpec，收到 {layer!r}。")
        for index in range(len(self.layers) - 1):
            left = self.layers[index]
            right = self.layers[index + 1]
            if left.out_features != right.in_features:
                raise ForwardError(
                    f"第 {index} 层的输出宽度 {left.out_features} 与第 {index + 1} 层的"
                    f"输入宽度 {right.in_features} 接不上：前向的定义要求相邻两层共享同一个宽度。"
                )

    @property
    def input_width(self) -> int:
        """整个网络的输入宽度."""
        return self.layers[0].in_features

    @property
    def output_width(self) -> int:
        """整个网络的输出宽度."""
        return self.layers[-1].out_features

    @property
    def parameter_count(self) -> int:
        """所有层的参数之和."""
        return sum(layer.parameter_count for layer in self.layers)

    @property
    def widths(self) -> tuple[int, ...]:
        """宽度链：``(in_0, out_0, out_1, ..., out_n)``."""
        return (self.layers[0].in_features,) + tuple(layer.out_features for layer in self.layers)

    def to_dict(self) -> dict[str, object]:
        """摊平成一行字段（层数 / 宽度链 / 参数总数）."""
        return {
            "layers": len(self.layers),
            "widths": list(self.widths),
            "parameter_count": self.parameter_count,
        }

    def line(self) -> str:
        """一行说明：``MLP 3→4→2 | 2 层 | 参数 26``."""
        chain = "→".join(str(width) for width in self.widths)
        return f"MLP {chain} | {len(self.layers)} 层 | 参数 {self.parameter_count}"


@dataclass(frozen=True)
class ForwardTrace:
    """一次前向的账：每一层的输入 / 输出宽度，以及中间张量的形状字符串."""

    input_width: int
    layer_widths: tuple[tuple[int, int], ...] = field(default=())

    def __post_init__(self) -> None:
        _require_positive_int(self.input_width, name="ForwardTrace.input_width")
        if not self.layer_widths:
            raise ParameterError("ForwardTrace.layer_widths 不能为空：一次前向至少穿过一层。")
        if self.layer_widths[0][0] != self.input_width:
            raise ForwardError(
                f"第一层的输入宽度 {self.layer_widths[0][0]} 与 trace 的输入宽度 "
                f"{self.input_width} 不一致。"
            )
        for index in range(len(self.layer_widths) - 1):
            if self.layer_widths[index][1] != self.layer_widths[index + 1][0]:
                raise ForwardError(f"第 {index} 层与第 {index + 1} 层的宽度接不上。")

    @property
    def output_width(self) -> int:
        """整个前向的输出宽度（最后一层的输出）."""
        return self.layer_widths[-1][1]

    @property
    def widths(self) -> tuple[int, ...]:
        """宽度链：``(in, out_0, out_1, ...)``."""
        return (self.input_width,) + tuple(out for _in, out in self.layer_widths)

    def shapes(self) -> tuple[str, ...]:
        """逐层的形状字符串：``"3 -> 4"``（用箭头读起来就是数据流的方向）."""
        return tuple(f"{in_w} -> {out_w}" for in_w, out_w in self.layer_widths)

    def to_dict(self) -> dict[str, object]:
        """摊平成一行字段."""
        return {
            "input_width": self.input_width,
            "layer_widths": [list(pair) for pair in self.layer_widths],
            "widths": list(self.widths),
            "shapes": list(self.shapes()),
        }

    def line(self) -> str:
        """一行说明：``3 → 4 → 2 | 层形状 3 -> 4、4 -> 2``."""
        chain = " → ".join(str(width) for width in self.widths)
        return f"{chain} | 层形状 {'、'.join(self.shapes())}"


@dataclass(frozen=True)
class LossReport:
    """一次损失的账：名称 + 标量 + 样本数 + 平均方式."""

    name: str
    value: float
    samples: int
    reduction: str = "mean"

    def __post_init__(self) -> None:
        if self.name not in LOSSES:
            raise ParameterError(f"未知的损失 {self.name!r}：可选 {list(LOSSES)}。")
        if not math.isfinite(self.value):
            raise NumericError(f"损失读数必须有限，收到 {self.value!r}：非有限数会让比较静默为假。")
        if self.value < 0.0 and self.name != LOSS_MAE:
            raise NumericError(f"{self.name} 不应为负，收到 {self.value!r}。")
        if self.samples < 1:
            raise ParameterError(f"样本数必须 >= 1，收到 {self.samples}。")
        if not self.reduction:
            raise ParameterError("损失的平均方式（reduction）不能为空。")

    def to_dict(self) -> dict[str, object]:
        """摊平成一行字段."""
        return {
            "name": self.name,
            "value": self.value,
            "samples": self.samples,
            "reduction": self.reduction,
        }

    def line(self) -> str:
        """一行说明：``mse = 0.000000e+00 | 样本 6 | mean``."""
        return f"{self.name} = {self.value:.6e} | 样本 {self.samples} | {self.reduction}"


__all__ = [
    "ACTIVATIONS",
    "ACTIVATION_DESCRIPTIONS",
    "ACTIVATION_PROFILES",
    "ACT_GELU",
    "ACT_LEAKY_RELU",
    "ACT_RELU",
    "ACT_SIGMOID",
    "ACT_SOFTMAX",
    "ACT_TANH",
    "ActivationProfile",
    "DenseSpec",
    "ForwardTrace",
    "GELU_TANH_CUBIC",
    "INITIALIZATIONS",
    "INITIALIZATION_DESCRIPTIONS",
    "INIT_HE",
    "INIT_NORMAL",
    "INIT_UNIFORM",
    "INIT_XAVIER",
    "INIT_ZEROS",
    "LEAKY_SLOPE",
    "LOSSES",
    "LOSS_CROSS_ENTROPY",
    "LOSS_DESCRIPTIONS",
    "LOSS_MAE",
    "LOSS_MSE",
    "LossReport",
    "MLPSpec",
    "NEURAL_BOUNDARIES",
    "NEURAL_NOTES",
    "NEURAL_NOTES_ORDER",
    "NEURAL_PROPERTIES",
    "NeuronSpec",
    "PROPERTY_ACTIVATIONS_ARE_FINITE",
    "PROPERTY_CROSS_ENTROPY_PATHS_AGREE",
    "PROPERTY_DESCRIPTIONS",
    "PROPERTY_FFN_MATCHES_TRANSFORMER_STACK",
    "PROPERTY_FAILURE",
    "PROPERTY_IDENTITY_STACK_COLLAPSES",
    "PROPERTY_MSE_IS_ZERO_AT_PERFECT",
    "PROPERTY_RANGES_ARE_RESPECTED",
    "PROPERTY_SOFTMAX_ROWS_ARE_DISTRIBUTIONS",
    "TORCH_COUNTERPARTS",
]
