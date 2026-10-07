"""``backprop`` 的常量表与记录（day090 / M8-D2）.

day089 把"从神经元到 FFN"这条链的**前向**搭好，并明确写下"一行反向都没有"。
今天补上那一行反向。这一课的全部口径可以压成一句话：

> **反向传播 = 上游梯度 × 局部导数，然后累加。**

本模块只放**口径**：六个激活的导数公式、两个损失的梯度公式、七条性质、十条笔记、
五条边界、一个"逐算子"的规则表，以及那次反向留下的四个记录。算术一律在别的模块里
（``gradients`` / ``graph`` / ``layers`` / ``network``）——"一个量只写一遍"。

## 一、六个激活的导数（**这是本课最值钱的一张表**）

```text
relu         d/dx = 1 if x > 0 else 0                在 x = 0 处取**次梯度 0**（约定，不是事实）
leaky_relu   d/dx = 1 if x > 0 else 0.01             负半轴留的那条坡度在导数里也是 0.01
sigmoid      d/dx = σ(x)(1 − σ(x))                   值域 (0, 1) ⇒ 导数上界 0.25
tanh         d/dx = 1 − tanh²(x)                     值域 (−1, 1) ⇒ 导数上界 1
gelu         d/dx = 0.5(1 + erf(x/√2)) + x·e^{−x²/2}/√(2π)    在 0 处恰好是 0.5
softmax      J[i][j] = p_i(δ_ij − p_j)               一整行一起看，不是一个数对一个数
```

前五行都是"逐元素"的（一个数对一个数），只有 ``softmax`` 是**逐行**的——
它是这一课唯一一个"局部导数是一张矩阵"的激活。

## 二、一条纪律：**softmax 的雅可比不显式乘出来**

``softmax`` 的雅可比是 ``n × n`` 的，但反向真正要算的是 ``Jᵀv``：

```text
显式路径   先造 n×n 的 J，再乘 v        ⇒ O(n²) 个中间量
恒等变形   (Jᵀv)_i = p_i(v_i − ⟨p, v⟩)   ⇒ O(n) 个中间量（本课采用）
```

两式**恒等**，本课把它写成一条性质（第 4 条）逐点量出来。生产实现里这条变形
是 softmax 反向的全部内容——没有一处会真的去建那张矩阵。

## 三、torch 只对照、不调用

与 day089 同一条纪律：本仓库全程纯 Python（**不依赖 torch，也不依赖 numpy**）。
``TORCH_COUNTERPARTS`` 这张表对着**公式与语义**核对，不是一次"实测 torch 输出"，
也不声明任何 torch 版本号。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from smart_research_agent.backprop.errors import NumericError, ParameterError
from smart_research_agent.math_foundations.types import Matrix, Vector
from smart_research_agent.neural_basics.types import ACTIVATIONS

# --------------------------------------------------------------------------------------
# 1. 六个激活的导数公式
# --------------------------------------------------------------------------------------

#: 激活名 → 它的导数公式（**顺序 = :data:`neural_basics.types.ACTIVATIONS`**）.
ACTIVATION_DERIVATIVE_FORMULAS: dict[str, str] = {
    "relu": "1 if x > 0 else 0（x = 0 处取次梯度 0）",
    "leaky_relu": "1 if x > 0 else 0.01",
    "sigmoid": "σ(x)·(1 − σ(x))（值域 (0,1) ⇒ 导数上界 0.25）",
    "tanh": "1 − tanh²(x)（值域 (−1,1) ⇒ 导数上界 1）",
    "gelu": "0.5(1 + erf(x/√2)) + x·e^{−x²/2}/√(2π)",
    "softmax": "J[i][j] = p_i(δ_ij − p_j)（**逐行**，不是逐元素）",
}

#: **逐元素**的五个激活（``softmax`` 不在其中：它的局部导数是一张矩阵）.
ELEMENTWISE_ACTIVATIONS: tuple[str, ...] = ("relu", "leaky_relu", "sigmoid", "tanh", "gelu")

if set(ELEMENTWISE_ACTIVATIONS) | {"softmax"} != set(ACTIVATIONS):  # pragma: no cover - 导入期不变式
    raise ParameterError(
        "逐元素激活表加 softmax 必须恰好等于 neural_basics 的六个激活："
        "少一个的那个激活在反向里会静默地走不到分支，而'走不到'与'它没有反向'长得一样。"
    )

# --------------------------------------------------------------------------------------
# 2. 两个损失的梯度公式
# --------------------------------------------------------------------------------------

LOSS_MSE = "mse"
LOSS_CROSS_ENTROPY = "cross_entropy"

#: 本课实现反向的两个损失（顺序 = 公式表的顺序）.
LOSSES: tuple[str, ...] = (LOSS_MSE, LOSS_CROSS_ENTROPY)

LOSS_GRADIENT_FORMULAS: dict[str, str] = {
    LOSS_MSE: "∂L/∂pred = 2(pred − target)/N（N = 全部元素个数）",
    LOSS_CROSS_ENTROPY: "∂L/∂logits = (p − onehot(target))/N（p = softmax(logits)）",
}

# --------------------------------------------------------------------------------------
# 3. 逐算子的反向规则表（**这一课最该背下来的一张表**）
# --------------------------------------------------------------------------------------

#: 算子 → "前向是什么 / 局部导数是谁 / 上游从哪来"。反向传播的**全部**内容就是这张表。
BACKWARD_RULES: dict[str, str] = {
    "add": "z = x + y ⇒ dx = dy = dZ（加法把上游**原样**分给两条边）",
    "mul": "z = x·y ⇒ dx = dZ·y、dy = dZ·x（局部导数是'另一个乘数的**前向值**'）",
    "matmul": "C = A·B ⇒ dA = dC·Bᵀ、dB = Aᵀ·dC（两侧各自乘对方的转置）",
    "add_bias": "Y = Z + b（逐行广播）⇒ dZ = dY、db = dY 的**列求和**",
    "relu": "y = max(0, x) ⇒ dx = dY·[x > 0]（在 x = 0 处取 0）",
    "sigmoid": "y = σ(x) ⇒ dx = dY·y·(1 − y)（用**前向输出**，不必再算一次 σ）",
    "tanh": "y = tanh(x) ⇒ dx = dY·(1 − y²)（同样用前向输出）",
    "gelu": "y = gelu(x) ⇒ dx = dY·[0.5(1 + erf(x/√2)) + x·e^{−x²/2}/√(2π)]（需要**输入 x**）",
    "softmax": "P = softmax(Z) ⇒ dZ = P⊙(dP − ⟨dP, P⟩行)（不建 n×n 的雅可比）",
    "mse": "L = mean((P − T)²) ⇒ dP = 2(P − T)/N",
    "cross_entropy": "L = −log softmax(Z)[k] ⇒ dZ = p − onehot(k)",
}

# --------------------------------------------------------------------------------------
# 4. 七条性质（**判据分两类**）
# --------------------------------------------------------------------------------------

PROPERTY_ACTIVATION_DERIVATIVES_MATCH_NUMERICAL = "activation_derivatives_match_numerical"
PROPERTY_RELU_SUBGRADIENT_IS_ZERO = "relu_subgradient_is_zero_at_origin"
PROPERTY_SOFTMAX_JACOBIAN_MATCHES_NUMERICAL = "softmax_jacobian_matches_numerical"
PROPERTY_SOFTMAX_JVP_AVOIDS_MATRIX = "softmax_jvp_avoids_matrix"
PROPERTY_CROSS_ENTROPY_GRADIENT_IS_P_MINUS_ONEHOT = "cross_entropy_gradient_is_p_minus_onehot"
PROPERTY_MLP_BACKWARD_MATCHES_NUMERICAL = "mlp_backward_matches_numerical"
PROPERTY_FFN_BACKWARD_MATCHES_ENCODER_DECODER = "ffn_backward_matches_encoder_decoder"

#: 七条性质（**顺序 = verify 报告的顺序**）.
BACKPROP_PROPERTIES: tuple[str, ...] = (
    PROPERTY_ACTIVATION_DERIVATIVES_MATCH_NUMERICAL,
    PROPERTY_RELU_SUBGRADIENT_IS_ZERO,
    PROPERTY_SOFTMAX_JACOBIAN_MATCHES_NUMERICAL,
    PROPERTY_SOFTMAX_JVP_AVOIDS_MATRIX,
    PROPERTY_CROSS_ENTROPY_GRADIENT_IS_P_MINUS_ONEHOT,
    PROPERTY_MLP_BACKWARD_MATCHES_NUMERICAL,
    PROPERTY_FFN_BACKWARD_MATCHES_ENCODER_DECODER,
)

PROPERTY_DESCRIPTIONS: dict[str, str] = {
    PROPERTY_ACTIVATION_DERIVATIVES_MATCH_NUMERICAL: "五个逐元素激活的解析导数与数值差分逐点一致（相对误差 <= 1e-6）",
    PROPERTY_RELU_SUBGRADIENT_IS_ZERO: "``relu`` 在 ``x = 0`` 处取次梯度 0：反向里一个恒为负的输入**恰好**得到 0",
    PROPERTY_SOFTMAX_JACOBIAN_MATCHES_NUMERICAL: "**跨天对账**：本包 softmax 雅可比与 day074 ``calculus.jacobian`` 的数值雅可比一致",
    PROPERTY_SOFTMAX_JVP_AVOIDS_MATRIX: "恒等式 ``(Jᵀv)_i = p_i(v_i − ⟨p,v⟩)`` 与显式矩阵乘的偏差 <= 1e-12",
    PROPERTY_CROSS_ENTROPY_GRADIENT_IS_P_MINUS_ONEHOT: "**跨天对账**：CE 的梯度等于 ``p − onehot``，与 day074 的数值差分一致",
    PROPERTY_MLP_BACKWARD_MATCHES_NUMERICAL: "整条 MLP 的解析梯度（对权重 / 偏置 / 输入）与数值差分一致（相对误差 <= 1e-5）",
    PROPERTY_FFN_BACKWARD_MATCHES_ENCODER_DECODER: "**跨天对账**：本包 ``ffn_backward`` 与 ``encoder_decoder.layers.feed_forward_backward`` **逐位**一致",
}

PROPERTY_FAILURE: dict[str, str] = {
    PROPERTY_ACTIVATION_DERIVATIVES_MATCH_NUMERICAL: "某个激活的导数抄错了（漏乘一项、符号写反），而前向仍然完全正确",
    PROPERTY_RELU_SUBGRADIENT_IS_ZERO: "在 x = 0 处取了 1（右侧导数）⇒ 一个恒为负的输入会一直拿到梯度",
    PROPERTY_SOFTMAX_JACOBIAN_MATCHES_NUMERICAL: "softmax 的雅可比漏了 −p_j 那一项（对角线对了、非对角线错了）",
    PROPERTY_SOFTMAX_JVP_AVOIDS_MATRIX: "把 'Jᵀv' 写成了 'Jv'，或漏掉了行内积 ⟨p,v⟩——两者形状都对",
    PROPERTY_CROSS_ENTROPY_GRADIENT_IS_P_MINUS_ONEHOT: "忘记除以行数、或把 onehot 加成了 p（符号写反）",
    PROPERTY_MLP_BACKWARD_MATCHES_NUMERICAL: "逐层回传时漏了一层、或把 dW 与 dx 的转置方向搞反",
    PROPERTY_FFN_BACKWARD_MATCHES_ENCODER_DECODER: "本包的前馈反向与项目里真实的口径不同（累加顺序或转置方向）",
}

# --------------------------------------------------------------------------------------
# 5. 十条笔记（这是"一次反向"这条链的成品）
# --------------------------------------------------------------------------------------

BACKPROP_NOTES: dict[str, str] = {
    "chain_rule_is_mul_then_add": (
        "反向传播只有一条规则：**上游梯度 × 局部导数**；只有一条纪律：**累加**。"
        "一个值被用了 k 次，它的梯度就是 k 条路径之和——写成 = 会让梯度偏小，而不报错。"
    ),
    "local_derivative_uses_forward_value": (
        "乘法的局部导数取的是**另一个乘数的前向值**，与梯度无关。"
        "把 dZ 当成局部导数是这一层最常见的错——它在只有一条链时甚至能算对。"
    ),
    "relu_is_a_switch": (
        "relu 的反向是一个开关：x > 0 时原样通过，否则**恰好**是 0。"
        "这就是'关掉的神经元在反向里梯度恰好是 0'的严格含义。"
    ),
    "sigmoid_derivative_uses_its_own_output": (
        "σ'(x) = σ(x)(1 − σ(x))：局部导数可以用**前向输出**直接算，不必再算一次 σ。"
        "tanh 同理（1 − tanh²）。这条省下一次 exp，也让实现更难写错。"
    ),
    "gelu_derivative_needs_the_input": (
        "gelu 与 sigmoid / tanh 不同：它的导数**必须知道输入 x**，只有输出是不够的。"
        "因此逐元素反向的签名里必须带上激活**前**的值（pre_activation）。"
    ),
    "softmax_jacobian_is_a_matrix": (
        "softmax 是唯一一个'局部导数是一张矩阵'的激活：J[i][j] = p_i(δ_ij − p_j)。"
        "但它从来不真的被建出来——见下一条。"
    ),
    "softmax_jvp_avoids_the_matrix": (
        "反向只需要 Jᵀv，而 (Jᵀv)_i = p_i(v_i − ⟨p,v⟩)。"
        "这条恒等变形把 O(n²) 的中间量压成 O(n)，是生产实现里 softmax 反向的全部内容。"
    ),
    "cross_entropy_gradient_cancels_the_jacobian": (
        "∂(−log p_k)/∂z = p − onehot(k)：softmax 的雅可比与 −log 的导数相乘之后**化简**了。"
        "于是'softmax 雅可比写错'这件事在交叉熵的梯度上**看不见**——它必须被单独验一次。"
    ),
    "mse_gradient_is_twice_the_residual": (
        "∂mean((p−t)²)/∂p = 2(p−t)/N：系数 2 来自平方的导数，分母 N 来自'均值'。"
        "少一个 2 会让学习率看起来需要减半——而它不报错。"
    ),
    "zero_grad_is_not_optional": (
        "图可以复用，梯度不会自己清零：第二次 backward 会把梯度**加上去**。"
        "漏掉清零不会报错，只会让'这一步的梯度'变成'这两步梯度之和'。"
    ),
}

#: 十条笔记的键（顺序即写入顺序，报告里读它）.
BACKPROP_NOTES_ORDER: tuple[str, ...] = tuple(BACKPROP_NOTES)

# --------------------------------------------------------------------------------------
# 6. 边界（**这一课明确不承诺的事**）
# --------------------------------------------------------------------------------------

BACKPROP_BOUNDARIES: tuple[str, ...] = (
    "本包实现的是**一阶**反向：只算梯度，不算 Hessian / 二阶导数，也不做二阶优化",
    "本包不安装、也不调用 torch / numpy：PyTorch 的语义只写在 TORCH_COUNTERPARTS 对照表里，"
    "对照的是公式与语义，不是'实测 torch 输出'；具体版本以官方文档为准",
    "本包的自动微分只覆盖一个 MLP 用得到的算子（matmul / 加偏置 / 五个逐元素激活 / softmax / "
    "mse / 交叉熵），不是'任意计算图'",
    "权重由 LCG 生成、输入写死，因此本包复现的是**读数**、不是统计规律；"
    "它不承诺任何'训练到收敛'的结果",
    "本包不承诺浮点意义上的'位逐位跨平台一致'——跨天对账的容差由各自的口径给出",
)

# --------------------------------------------------------------------------------------
# 7. 纯 Python ↔ PyTorch 对照表（**只对照，不调用**）
# --------------------------------------------------------------------------------------

#: 本包的反向 ↔ PyTorch 的反向 API 对照。只核对**公式与语义**，不声明任何 torch 版本号。
TORCH_COUNTERPARTS: dict[str, str] = {
    "backward": "torch.Tensor.backward()（从标量损失出发做反向模式自动微分）",
    "zero_grad": "torch.optim.Optimizer.zero_grad() / param.grad.zero_()",
    "graph": "torch.autograd.Function + torch.Tensor.grad_fn 构成的动态图",
    "retain_graph": "torch.Tensor.backward(retain_graph=True)（同一张图被反两次时必须显式说明）",
    "dense_backward": "nn.Linear 的 grad_weight = dYᵀ·X、grad_bias = dY 的列求和",
    "relu_backward": "torch.nn.functional.relu 的导数（torch 里由 autograd 现场推出）",
    "sigmoid_backward": "torch.sigmoid 的导数 σ(1−σ)",
    "tanh_backward": "torch.tanh 的导数 1 − tanh²",
    "gelu_backward": "torch.nn.functional.gelu 的导数（需保留输入）",
    "softmax_backward": "torch.nn.functional.softmax 的反向（内部同样不做 n×n 雅可比）",
    "mse_grad": "torch.nn.functional.mse_loss 的梯度 2(p−t)/N",
    "cross_entropy_grad": "torch.nn.functional.cross_entropy 的梯度 p − onehot",
    "grad_check": "torch.autograd.gradcheck（用数值差分校验解析梯度）",
    "no_grad": "torch.no_grad()（推理时不建图）",
}

# --------------------------------------------------------------------------------------
# 8. 记录
# --------------------------------------------------------------------------------------


def _require_positive_float(value: object, *, name: str) -> float:
    """正有限数护栏（``bool`` 被显式拒绝，因为 ``True == 1`` 会偷偷溜过）."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParameterError(f"{name} 必须是实数，收到 {value!r}。")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ParameterError(
            f"{name} 必须是正的有限数，收到 {value!r}："
            "负的步长会把数值差分变成反向差分，负的容差会让任何偏差都'通过'。"
        )
    return resolved


@dataclass(frozen=True)
class DenseGradients:
    """一层全连接的三块梯度：``grad_weight``、``grad_bias``、``grad_inputs``.

    ```text
    grad_weight  (out, in)   = dYᵀ·X            （回传梯度的每一列 × 输入）
    grad_bias    (out,)      = Σ_i dY[i]        （偏置被所有行共用 ⇒ 按行求和）
    grad_inputs  (rows, in)  = dY·W             （按 W 的**列**收，而不是按行）
    ```

    三块的形状刻意与 :class:`backprop.types.FFNGradients` 的后三块一致：
    一层全连接与前馈的一半是同一个算子，因此它们的梯度也该长得一样。
    """

    grad_weight: Matrix
    grad_bias: Vector
    grad_inputs: Matrix

    @property
    def weight_shape(self) -> tuple[int, int]:
        """``grad_weight`` 的形状 ``(out_features, in_features)``."""
        return (len(self.grad_weight), len(self.grad_weight[0]) if self.grad_weight else 0)

    def to_dict(self) -> dict[str, object]:
        """摊平成一行字段（**不含**梯度本体，只含形状与各块的和）."""
        return {
            "weight_shape": list(self.weight_shape),
            "grad_bias_len": len(self.grad_bias),
            "rows": len(self.grad_inputs),
            "sum_grad_weight": math.fsum(value for row in self.grad_weight for value in row),
            "sum_grad_bias": math.fsum(self.grad_bias),
        }

    def line(self) -> str:
        """一行说明：``dW (4, 3) | db 4 | dx 2 行 | ΣdW=...``."""
        rows, columns = self.weight_shape
        weight_total = math.fsum(value for row in self.grad_weight for value in row)
        return (
            f"dW ({rows}, {columns}) | db {len(self.grad_bias)} | dx {len(self.grad_inputs)} 行 | "
            f"ΣdW={weight_total:+.6f}"
        )


@dataclass(frozen=True)
class FFNGradients:
    """前馈块的五块梯度（**字段名与 ``encoder_decoder.types.FFNGradients`` 逐一相同**）.

    与那一份保持同名不是巧合：两条实现要能被逐位对账，就得先能被**同一个读取器**读。
    """

    grad_w_in: Matrix
    grad_b_in: Vector
    grad_w_out: Matrix
    grad_b_out: Vector
    grad_inputs: Matrix

    def to_dict(self) -> dict[str, object]:
        """摊平成一行字段（形状 + 各块的和）."""
        return {
            "w_in_shape": [len(self.grad_w_in), len(self.grad_w_in[0]) if self.grad_w_in else 0],
            "w_out_shape": [len(self.grad_w_out), len(self.grad_w_out[0]) if self.grad_w_out else 0],
            "b_in_len": len(self.grad_b_in),
            "b_out_len": len(self.grad_b_out),
            "rows": len(self.grad_inputs),
        }

    def line(self) -> str:
        """一行说明：``dw_in (16, 4) | dw_out (4, 16) | db 16/4 | dx 2 行``."""
        in_shape = (len(self.grad_w_in), len(self.grad_w_in[0]) if self.grad_w_in else 0)
        out_shape = (len(self.grad_w_out), len(self.grad_w_out[0]) if self.grad_w_out else 0)
        return (
            f"dw_in {in_shape} | dw_out {out_shape} | "
            f"db {len(self.grad_b_in)}/{len(self.grad_b_out)} | dx {len(self.grad_inputs)} 行"
        )


@dataclass(frozen=True)
class GradientReading:
    """一个梯度读数：名字 + 标量 + 形状 + 公式（**四列都要有**）.

    一行只写"梯度算出来了"的表是没法反驳的；因此每一行都要同时给出
    **它是什么**（公式）、**它多长**（形状）与**它现在等于多少**（标量读数）。
    """

    name: str
    value: float
    shape: str
    formula: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise NumericError(
                f"梯度读数 {self.name!r} 非有限（{self.value!r}）："
                "非有限的读数会让任何比较静默为假，因此它在入口就被拒绝。"
            )
        if not self.shape or not self.formula:
            raise ParameterError(f"梯度读数 {self.name!r} 缺少形状或公式：两列都是判据的一部分。")

    def to_dict(self) -> dict[str, object]:
        """摊平成一行字段."""
        return {
            "name": self.name,
            "value": self.value,
            "shape": self.shape,
            "formula": self.formula,
        }

    def line(self) -> str:
        """一行说明：``relu        = +1.000000 | 逐元素 | d/dx = ...``."""
        return f"{self.name:<12} = {self.value:+.6f} | {self.shape} | {self.formula}"


@dataclass(frozen=True)
class LayerGradients:
    """一层反向的账：序号 + 三块梯度的**范数** + 权重形状.

    为什么记的是范数而不是全部元素：一层 ``(8, 4)`` 的权重有 32 个偏导数，
    把它们全部印出来读不出"这一层在学什么"；而三个范数一眼能看出
    "这一层的梯度是不是比上一层大一个量级"——那是梯度爆炸最早的可读信号。
    """

    index: int
    weight_shape: tuple[int, int]
    weight_norm: float
    bias_norm: float
    input_norm: float

    def __post_init__(self) -> None:
        for label, value in (
            ("weight_norm", self.weight_norm),
            ("bias_norm", self.bias_norm),
            ("input_norm", self.input_norm),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise NumericError(f"第 {self.index} 层的 {label} 必须是非负有限数，收到 {value!r}。")

    def to_dict(self) -> dict[str, object]:
        """摊平成一行字段."""
        return {
            "index": self.index,
            "weight_shape": list(self.weight_shape),
            "weight_norm": self.weight_norm,
            "bias_norm": self.bias_norm,
            "input_norm": self.input_norm,
        }

    def line(self) -> str:
        """一行说明：``第 1 层  W (8, 4) | ‖dW‖=... ‖db‖=... ‖dx‖=...``."""
        rows, columns = self.weight_shape
        return (
            f"第 {self.index} 层  W ({rows}, {columns}) | "
            f"‖dW‖={self.weight_norm:.6e} ‖db‖={self.bias_norm:.6e} ‖dx‖={self.input_norm:.6e}"
        )


__all__ = [
    "ACTIVATION_DERIVATIVE_FORMULAS",
    "BACKPROP_BOUNDARIES",
    "BACKPROP_NOTES",
    "BACKPROP_NOTES_ORDER",
    "BACKPROP_PROPERTIES",
    "BACKWARD_RULES",
    "DenseGradients",
    "ELEMENTWISE_ACTIVATIONS",
    "FFNGradients",
    "GradientReading",
    "LOSSES",
    "LOSS_CROSS_ENTROPY",
    "LOSS_GRADIENT_FORMULAS",
    "LOSS_MSE",
    "LayerGradients",
    "PROPERTY_ACTIVATION_DERIVATIVES_MATCH_NUMERICAL",
    "PROPERTY_CROSS_ENTROPY_GRADIENT_IS_P_MINUS_ONEHOT",
    "PROPERTY_DESCRIPTIONS",
    "PROPERTY_FFN_BACKWARD_MATCHES_ENCODER_DECODER",
    "PROPERTY_FAILURE",
    "PROPERTY_MLP_BACKWARD_MATCHES_NUMERICAL",
    "PROPERTY_RELU_SUBGRADIENT_IS_ZERO",
    "PROPERTY_SOFTMAX_JACOBIAN_MATCHES_NUMERICAL",
    "PROPERTY_SOFTMAX_JVP_AVOIDS_MATRIX",
    "TORCH_COUNTERPARTS",
]
