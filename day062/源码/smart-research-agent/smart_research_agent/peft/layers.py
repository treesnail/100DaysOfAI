"""LoRA 的矩阵算术与参考层（M5-D3）.

LoRA 的全部内容就是一条式子：

    W' = W + ΔW,        ΔW = (alpha / r) · B @ A

其中 ``W ∈ R^{out×in}`` 是**冻结**的基座权重，``A ∈ R^{r×in}``、
``B ∈ R^{out×r}`` 是**新增**的两个小矩阵，``r ≪ min(in, out)``。

本模块只做两件在 day050 就已经定下的事，把它们从 bigram 推广到一般矩阵：

1. **可手算的算术**（纯函数）：增量矩阵、合并权重、秩上界、参数量；
2. **一个真的能反向传播的层**（``LoRALinear``）：前向、反向、梯度累积、
   一次 SGD 更新。梯度是**解析求出来的**，不是数值近似——因为
   ``∂L/∂B`` 与 ``∂L/∂A`` 都能直接写出来：设 ``a = A·x``、
   ``g = ∂L/∂output``（长度 ``out``），则

       ∂L/∂B[i][j] = scaling · g[i] · a[j]
       ∂L/∂a[j]    = scaling · Σ_i g[i] · B[i][j]
       ∂L/∂A[j][k] = ∂L/∂a[j] · x[k]

   这三行是理解"LoRA 为什么能训"的全部内容：**基座没有梯度，增量有梯度**。

为什么要把梯度写成解析式而不是用数值差分：数值差分的误差量级是
``O(ε)``，而 LoRA 的增量本身也很小（``B`` 初始为 0），两者量级接近时
"梯度算得对不对"就无法判别。解析式可以逐元素对照（见
``tests/test_lora_layers.py`` 里与数值差分的交叉验证）。
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from smart_research_agent.peft.config import LoRAConfig, PEFTConfigError

#: 判断"两个浮点数是否可视为 0"的缺省容差（用于"第 0 步增量恒为 0"这类断言）.
ZERO_TOLERANCE = 1e-12


def _zeros(rows: int, cols: int) -> list[list[float]]:
    """构造 ``rows × cols`` 的全零矩阵."""
    return [[0.0] * cols for _ in range(rows)]


def _check_matrix(matrix: Sequence[Sequence[float]], *, name: str) -> tuple[int, int]:
    """校验二维矩阵是**矩形**的，返回 ``(rows, cols)``.

    矩形校验放在这里而不是交给下游：一个"第 3 行少一列"的矩阵在矩阵乘法里
    会抛 ``IndexError``，而在逐元素加法里会静默跑完——**同一个错误在不同
    路径上的表现不一致，是最难排查的一类问题**。
    """
    rows = len(matrix)
    if rows == 0:
        raise PEFTConfigError(f"{name} 不能为空矩阵")
    cols = len(matrix[0])
    if cols == 0:
        raise PEFTConfigError(f"{name} 的列数不能为 0")
    for index, row in enumerate(matrix):
        if len(row) != cols:
            raise PEFTConfigError(
                f"{name} 不是矩形：第 {index} 行有 {len(row)} 列，第 0 行有 {cols} 列"
            )
    return rows, cols


def init_lora_weights(
    in_features: int,
    out_features: int,
    r: int,
    *,
    init_mode: str = "gaussian",
    seed: int = 42,
    std_scale: float = 1.0,
) -> tuple[list[list[float]], list[list[float]]]:
    """初始化 ``(A, B)``，返回两个**可变**的嵌套列表.

    ``gaussian``（缺省，也是 peft 的缺省语义）做两件事：

    - ``A[i][j] ~ N(0, (std_scale / sqrt(in_features))²)``——标准差按输入
      维度缩放。这里只保证**量级**是 ``Θ(1/sqrt(fan_in))``：peft 用的是
      ``kaiming_uniform(a=√5)``，等价于 ``U(-1/√fan_in, 1/√fan_in)``，方差
      是 ``1/(3·fan_in)``，与本实现相差 3 倍。数量级相同即可，因为真正决定
      增量大小的是后续的 ``scaling`` 与训练本身；
    - ``B`` 取**全零**——于是 ``ΔW = 0``，第 0 步的模型输出与基座逐位一致。
      这条性质是 LoRA 能被"先挂上、再评估、再训练"的原因：接适配器不会
      让效果立刻变差。

    ``random``（``init_mode="random"``）让 ``B`` 也随机，**破坏 ``ΔW = 0``**，
    只用于对照实验。

    RNG 用局部 ``random.Random(seed)``，不污染全局随机状态——与 day048
    ``split_dataset``、day050 ``SFTTrainer`` 是同一条纪律。
    """
    if in_features <= 0 or out_features <= 0:
        raise PEFTConfigError(
            f"in_features / out_features 必须为正整数，收到 {in_features} / {out_features}"
        )
    if r <= 0:
        raise PEFTConfigError(f"r 必须为正整数，收到 {r}")
    if init_mode not in ("gaussian", "random"):
        raise PEFTConfigError(
            f"参考实现只支持 gaussian / random 两种初始化，收到 {init_mode!r}；"
            "loftq / eva / olora / orthogonal 需要 peft 运行时"
        )
    if std_scale <= 0:
        raise PEFTConfigError(f"std_scale 必须为正数，收到 {std_scale}")

    rng = random.Random(seed)
    a_std = std_scale / math.sqrt(in_features)
    b_std = std_scale / math.sqrt(r) if init_mode == "random" else 0.0
    a_matrix = [[rng.gauss(0.0, a_std) for _ in range(in_features)] for _ in range(r)]
    b_matrix = [[rng.gauss(0.0, b_std) for _ in range(r)] for _ in range(out_features)]
    return a_matrix, b_matrix


def matmul(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> list[list[float]]:
    """矩阵乘法（纯 Python、三层循环；本课的矩阵规模是教学级的）.

    显式三层循环而不是调库：``r=8`` 时一次 ``B @ A`` 是
    ``557×8×557 ≈ 248 万`` 次乘加，几十毫秒就能跑完，**而"哪一层是秩维"
    在代码里看得见**——这正是本课要讲的东西。
    """
    left_rows, left_cols = _check_matrix(left, name="左矩阵")
    right_rows, right_cols = _check_matrix(right, name="右矩阵")
    if left_cols != right_rows:
        raise PEFTConfigError(
            f"矩阵维度不匹配：左矩阵 {left_rows}×{left_cols}，右矩阵 {right_rows}×{right_cols}"
        )
    result = _zeros(left_rows, right_cols)
    for i in range(left_rows):
        left_row = left[i]
        for k in range(left_cols):
            factor = left_row[k]
            if factor == 0.0:
                continue
            right_row = right[k]
            for j in range(right_cols):
                result[i][j] += factor * right_row[j]
    return result


def add_matrices(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> list[list[float]]:
    """逐元素相加（形状不一致直接报错，不做广播）."""
    left_rows, left_cols = _check_matrix(left, name="左矩阵")
    right_rows, right_cols = _check_matrix(right, name="右矩阵")
    if (left_rows, left_cols) != (right_rows, right_cols):
        raise PEFTConfigError(
            f"矩阵形状不一致，无法相加：{left_rows}×{left_cols} vs {right_rows}×{right_cols}"
        )
    return [
        [left[i][j] + right[i][j] for j in range(left_cols)] for i in range(left_rows)
    ]


def lora_delta(
    a_matrix: Sequence[Sequence[float]],
    b_matrix: Sequence[Sequence[float]],
    scaling: float,
) -> list[list[float]]:
    """增量矩阵 ``ΔW = scaling · B @ A``（形状 ``out × in``）.

    注意 ``B @ A`` 的乘积形状与基座 ``W`` **完全一致**，这是"可以直接相加"
    的前提，也是"LoRA 不需要改动模型结构、只多两条分支"的原因。
    """
    product = matmul(b_matrix, a_matrix)
    return [[value * scaling for value in row] for row in product]


def merge_lora_weight(
    base_weight: Sequence[Sequence[float]],
    a_matrix: Sequence[Sequence[float]],
    b_matrix: Sequence[Sequence[float]],
    scaling: float,
) -> list[list[float]]:
    """合并后的权重 ``W + scaling · B @ A``.

    合并的意义是**推理时不再需要额外的矩阵乘法**：``W'`` 与基座同形同 dtype，
    可以像普通权重一样部署（调用 ``peft`` 的 ``merge_and_unload()`` 做的是
    同一件事）。代价是增量被固化了——想换一个适配器就得重新合并。
    """
    return add_matrices(base_weight, lora_delta(a_matrix, b_matrix, scaling))


def context_delta(
    a_matrix: Sequence[Sequence[float]],
    b_matrix: Sequence[Sequence[float]],
    index: int,
    scaling: float,
) -> list[float]:
    """``ΔW`` 的第 ``index`` **列**：``scaling · B @ A[:, index]``（长度 ``out``）.

    为什么"上下文 token 的增量"是 ``ΔW`` 的一列：适配器分支算的是
    ``B·(A·e_index) = B·A[:, index]``，而 ``ΔW = B @ A`` 的第 ``index`` 列
    正是 ``B @ A[:, index]``。

    bigram / 查表式权重（``in == out``）的每次前向只需要这一个向量，因此
    参考模型的每次前向只做 ``r × out`` 次乘加，而不是先把整个 ``out × in``
    的 ``ΔW`` 物化出来——真实 LoRA 也是这么做的（``x @ Aᵀ @ Bᵀ``，从不
    显式构造 ``B @ A``）。

    **注意它与基座的行是两回事**：基座取 ``W[index]``（一行），增量取
    ``ΔW[:, index]``（一列）。方阵上两种下标空间相同，这正是"行/列约定"
    容易混起来的原因（见 ``forward_context`` 的说明）。
    """
    if not 0 <= index < len(a_matrix[0]):
        raise PEFTConfigError(f"列下标 {index} 越界：A 有 {len(a_matrix[0])} 列")
    out_features = len(b_matrix)
    r = len(a_matrix)
    result = [0.0] * out_features
    for i in range(out_features):
        total = 0.0
        b_row = b_matrix[i]
        for j in range(r):
            total += b_row[j] * a_matrix[j][index]
        result[i] = total * scaling
    return result


def matrix_rank(matrix: Sequence[Sequence[float]], *, tolerance: float = 1e-9) -> int:
    """用带部分主元的高斯消元估算矩阵的数值秩.

    用途只有一个：**验证 LoRA 的秩上界是真的**。``ΔW = scaling·B@A`` 的秩
    不超过 ``min(r, in, out)``，这条性质在浮点下表现为"消元后非零主元的个数
    不超过 r"。它是 LoRA 里少数**能被一条断言钉死**的结构性质。
    """
    rows, cols = _check_matrix(matrix, name="待求秩矩阵")
    work = [[float(value) for value in row] for row in matrix]
    rank = 0
    for col in range(cols):
        pivot = -1
        best = tolerance
        for row in range(rank, rows):
            magnitude = abs(work[row][col])
            if magnitude > best:
                best = magnitude
                pivot = row
        if pivot < 0:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        pivot_value = work[rank][col]
        for row in range(rank + 1, rows):
            factor = work[row][col] / pivot_value
            if factor == 0.0:
                continue
            for inner in range(col, cols):
                work[row][inner] -= factor * work[rank][inner]
        rank += 1
        if rank == rows:
            break
    return rank


def is_rank_one(matrix: Sequence[Sequence[float]], *, tolerance: float = 1e-9) -> bool:
    """判断矩阵是否为秩 1（``r = 1`` 时 ``ΔW`` 的每一行都成比例）."""
    return matrix_rank(matrix, tolerance=tolerance) <= 1


def frobenius_norm(matrix: Sequence[Sequence[float]]) -> float:
    """矩阵的 Frobenius 范数（增量"有多大"的直接度量）."""
    _check_matrix(matrix, name="待求范数矩阵")
    return math.sqrt(sum(value * value for row in matrix for value in row))


def max_abs(matrix: Sequence[Sequence[float]]) -> float:
    """矩阵元素的最大绝对值（用于"增量确实非零"的断言）."""
    _check_matrix(matrix, name="待求最大值矩阵")
    return max(abs(value) for row in matrix for value in row)


def adapter_state_size(
    a_matrix: Sequence[Sequence[float]], b_matrix: Sequence[Sequence[float]]
) -> int:
    """适配器的参数个数 ``r·in + out·r``（= ``r(in+out)``）."""
    a_rows, a_cols = _check_matrix(a_matrix, name="A")
    b_rows, b_cols = _check_matrix(b_matrix, name="B")
    if a_rows != b_cols:
        raise PEFTConfigError(f"B 的列数 {b_cols} 必须等于 A 的行数 {a_rows}（即秩 r）")
    return a_rows * a_cols + b_rows * b_cols


def describe_delta(
    a_matrix: Sequence[Sequence[float]],
    b_matrix: Sequence[Sequence[float]],
    scaling: float,
) -> dict[str, float | int]:
    """增量的画像：形状、参数量、范数、最大元素、数值秩.

    这份画像在 demo 里是"LoRA 到底训出了什么"的第一手证据：训练前
    ``max_abs = 0``（``B`` 全零），训练后是一个非零但明显小于基座权重的
    数——``ΔW`` 的范数与 ``W`` 的范数之比，就是"这次微调改动了多少"。
    """
    delta = lora_delta(a_matrix, b_matrix, scaling)
    return {
        "out_features": len(delta),
        "in_features": len(delta[0]),
        "parameters": adapter_state_size(a_matrix, b_matrix),
        "scaling": scaling,
        "frobenius": frobenius_norm(delta),
        "max_abs": max_abs(delta),
        "numerical_rank": matrix_rank(delta),
        "rank_upper_bound": min(len(a_matrix), len(delta), len(delta[0])),
    }


class LoRALinear:
    """带 LoRA 分支的线性层参考实现（基座冻结、只训 ``A`` / ``B``）.

    接口与 day050 ``ReferenceSFTModel`` 的梯度累积纪律一致，分成两步：

    - ``accumulate(x, grad_out)``：前向 + 反向，把梯度**累加**进缓冲；
    - ``apply_update(lr)``：把累加梯度按 ``1/N`` 缩放（``N`` = 自上次更新
      以来累积的位置数）并做一次 SGD，然后清零缓冲。

    为什么梯度也要"先累加、后除 N"：与 day050 第五章同一条理由——前者
    精确地等于"把整个累积窗口当成一个大 batch"，后者在窗口内各位置数不等
    时会引入偏差。

    dropout 的语义（``lora_dropout > 0`` 且 ``training=True``）：对**输入**
    ``x`` 做一次反向缩放（``x_i/(1-p)`` 或以概率 ``p`` 置零），并把掩码
    缓存下来，反向时用同一个掩码。这是 peft ``LoraLayer`` 的做法，也是
    "训练与推理必须用同一套口径"的一个实例。
    """

    def __init__(
        self,
        base_weight: Sequence[Sequence[float]],
        config: LoRAConfig,
        *,
        seed: int = 42,
        std_scale: float = 1.0,
    ):
        config.validate()
        out_features, in_features = _check_matrix(base_weight, name="基座权重")
        self._config = config
        self._base = [[float(value) for value in row] for row in base_weight]
        self._a, self._b = init_lora_weights(
            in_features,
            out_features,
            config.r,
            init_mode="random" if config.init_lora_weights == "random" else "gaussian",
            seed=seed,
            std_scale=std_scale,
        )
        self._grad_a = _zeros(config.r, in_features)
        self._grad_b = _zeros(out_features, config.r)
        self._pending = 0
        self._updates = 0
        self._rng = random.Random(seed)
        self._cached_input: list[float] | None = None
        self._cached_activation: list[float] | None = None
        self._cached_mask: list[float] | None = None

    # ------------------------------------------------------------------ 属性
    @property
    def config(self) -> LoRAConfig:
        """本层使用的 LoRA 配置."""
        return self._config

    @property
    def in_features(self) -> int:
        """输入维度."""
        return len(self._base[0])

    @property
    def out_features(self) -> int:
        """输出维度."""
        return len(self._base)

    @property
    def r(self) -> int:
        """低秩维（秩）."""
        return self._config.r

    @property
    def scaling(self) -> float:
        """缩放系数（``alpha/r`` 或 ``alpha/sqrt(r)``）."""
        return self._config.scaling

    @property
    def updates(self) -> int:
        """已完成的参数更新次数."""
        return self._updates

    @property
    def pending_positions(self) -> int:
        """尚未应用到参数的累积位置数（= ``apply_update`` 的分母）."""
        return self._pending

    @property
    def base_weight(self) -> list[list[float]]:
        """基座权重的一份**拷贝**（防止调用方就地改动冻结参数）."""
        return [row[:] for row in self._base]

    @property
    def frozen_parameters(self) -> int:
        """冻结参数个数（= ``out × in``）."""
        return self.out_features * self.in_features

    @property
    def trainable_parameters(self) -> int:
        """可训练参数个数（= ``r × (in + out)``）."""
        return adapter_state_size(self._a, self._b)

    @property
    def trainable_ratio(self) -> float:
        """可训练参数占全参的比例（含基座）."""
        return self.trainable_parameters / (
            self.trainable_parameters + self.frozen_parameters
        )

    # ------------------------------------------------------------------ 前向
    def forward(self, x: Sequence[float], *, training: bool = False) -> list[float]:
        """``W·x + scaling · B·(A·x')``，其中 ``x'`` 是**施加过 dropout** 的 ``x``.

        dropout **只作用在 LoRA 分支的输入上**，基座路径用的是原始 ``x``——
        这与 peft ``LoraLayer`` 的实现一致（``base_layer(x) + lora_B(lora_A(dropout(x)))``）。
        把它作用到整层输入上会让基座输出也被随机缩放，等于给冻结的基座
        引入了噪声，这是很多自写实现的偏差来源。
        """
        if len(x) != self.in_features:
            raise PEFTConfigError(
                f"输入维度不匹配：本层需要 {self.in_features} 维，收到 {len(x)}"
            )
        dropout = self._config.lora_dropout
        if training and dropout > 0.0:
            keep = 1.0 - dropout
            mask = [0.0 if self._rng.random() < dropout else 1.0 / keep for _ in x]
        else:
            mask = [1.0] * len(x)
        effective = [value * mask[index] for index, value in enumerate(x)]

        base_out = [
            sum(row[index] * x[index] for index in range(self.in_features))
            for row in self._base
        ]
        activation = [
            sum(self._a[j][index] * effective[index] for index in range(self.in_features))
            for j in range(self.r)
        ]
        delta_out = [
            self.scaling * sum(self._b[i][j] * activation[j] for j in range(self.r))
            for i in range(self.out_features)
        ]

        self._cached_input = list(x)
        self._cached_activation = activation
        self._cached_mask = mask
        return [base_out[i] + delta_out[i] for i in range(self.out_features)]

    # ------------------------------------------------------------------ 反向
    def backward(self, grad_out: Sequence[float]) -> list[float]:
        """给定 ``∂L/∂output``，累加 ``∂L/∂A`` / ``∂L/∂B``，返回 ``∂L/∂x``.

        返回 ``∂L/∂x`` 是为了让本层能被**串联**使用（前一层需要的梯度），
        尽管本课的参考模型只需要最底层的一次反向。
        """
        if self._cached_activation is None or self._cached_input is None:
            raise PEFTConfigError("backward 之前必须先调用 forward（缺少缓存的激活）")
        if len(grad_out) != self.out_features:
            raise PEFTConfigError(
                f"输出梯度维度不匹配：本层输出 {self.out_features} 维，收到 {len(grad_out)}"
            )
        activation = self._cached_activation
        mask = self._cached_mask or [1.0] * self.in_features
        scaling = self.scaling

        grad_activation = [0.0] * self.r
        for i in range(self.out_features):
            factor = scaling * grad_out[i]
            if factor == 0.0:
                continue
            b_row = self._b[i]
            grad_row = self._grad_b[i]
            for j in range(self.r):
                grad_row[j] += factor * activation[j]
                grad_activation[j] += factor * b_row[j]

        grad_input = [0.0] * self.in_features
        for i in range(self.out_features):
            factor = grad_out[i]
            if factor == 0.0:
                continue
            base_row = self._base[i]
            for index in range(self.in_features):
                grad_input[index] += factor * base_row[index]
        for j in range(self.r):
            factor = grad_activation[j]
            if factor == 0.0:
                continue
            a_row = self._a[j]
            grad_row = self._grad_a[j]
            for index in range(self.in_features):
                grad_row[index] += factor * self._cached_input[index] * mask[index]
                grad_input[index] += factor * a_row[index] * mask[index]

        self._pending += 1
        return grad_input

    def accumulate(
        self, x: Sequence[float], grad_out: Sequence[float], *, training: bool = False
    ) -> list[float]:
        """``forward`` + ``backward`` 的便捷组合（返回 ``∂L/∂x``）."""
        self.forward(x, training=training)
        return self.backward(grad_out)

    # -------------------------------------------------------- one-hot 快速路径
    def _onehot_mask(self, index: int, *, training: bool) -> float:
        """one-hot 输入下的 dropout 系数.

        输入只有一个非零元素时，逐元素 dropout 的**唯一**效果是：以概率
        ``p`` 把这个非零元素置零，否则放大 ``1/(1-p)``。于是"给输入加噪声"
        退化为"以概率 ``p`` 整条适配器分支不被激活"——这个语义值得写出来，
        因为它说明 **LoRA dropout 作用在 embedding 查表这类 one-hot 输入上
        时，实际做的是样本级的分支丢弃**，与作用在稠密激活上不是一回事。
        """
        dropout = self._config.lora_dropout
        if not training or dropout <= 0.0:
            return 1.0
        if self._rng.random() < dropout:
            return 0.0
        return 1.0 / (1.0 - dropout)

    def forward_onehot(self, index: int, *, training: bool = False) -> list[float]:
        """one-hot 输入（第 ``index`` 个基向量）下的前向，复杂度 ``O(r·out)``.

        与 ``forward(e_index)`` **逐位等价**（测试里直接断言）：输入是 one-hot
        时，``W @ e_j`` 取的是 ``W`` 的第 ``j`` **列**。本方法只是省掉了
        ``O(in·out)`` 的乘法，并没有改变数学含义。
        """
        if not 0 <= index < self.in_features:
            raise PEFTConfigError(f"one-hot 下标 {index} 越界：本层输入维度 {self.in_features}")
        mask_value = self._onehot_mask(index, training=training)
        base_out = [self._base[i][index] for i in range(self.out_features)]
        activation = [self._a[j][index] * mask_value for j in range(self.r)]
        delta_out = [
            self.scaling * sum(self._b[i][j] * activation[j] for j in range(self.r))
            for i in range(self.out_features)
        ]
        cached_input = [0.0] * self.in_features
        cached_input[index] = 1.0
        self._cached_input = cached_input
        self._cached_activation = activation
        self._cached_mask = [1.0] * self.in_features
        self._cached_mask[index] = mask_value
        return [base_out[i] + delta_out[i] for i in range(self.out_features)]

    def forward_context(self, index: int, *, training: bool = False) -> list[float]:
        """**查表式（bigram）权重**的前向：``W[index] + scaling · B @ A[:, index]``.

        与 ``forward_onehot`` 的区别只有一处，但这一处很关键：这里取的是
        ``W`` 的第 ``index`` **行**，而增量取的是 ``ΔW`` 的第 ``index`` **列**。

        .. code-block:: text

            dense（forward / forward_onehot）：out = W@e_index = W[:, index]   （W 的一列）
            context（forward_context）      ：out = W[index] + ΔW[:, index]（W 一行 + ΔW 一列）

        为什么会有这种"混用"：day050 的 ``ReferenceSFTModel.logits(ctx)`` 定义为
        ``W[context] + b``，等价于把 ``Wᵀ`` 当作这一层的权重矩阵。把 LoRA
        挂到 ``Wᵀ`` 上，增量就是 ``scaling · B @ A``，于是

            (Wᵀ + scaling·B@A) · e_ctx = W[ctx] + scaling·B@A[:, ctx]

        也就是"基座取行、增量取列"。**这不是笔误，而是同一个矩阵被解读了两次。**

        由此得到一条会被测试钉死的关系：按上下文合并出来的矩阵
        ``M[i][j] = W[i][j] + ΔW[j][i]`` 是密集约定下的合并矩阵
        ``W + ΔW`` 的**转置**。

        **本课实现时真的踩到过这个坑**：把 dense 口径的增量加到 context 口径
        的基座上，模型照样训练、loss 照样下降，但"合并后的模型"与"带适配器的
        模型"在同一个上下文 token 上给出不同的 logits（实测最大差 0.038）。
        守这条不变式的测试是：**``ΔW = 0`` 时（刚挂上适配器、还没训练），
        ``LoRAReferenceModel.logits(ctx)`` 必须与基座逐位相同。**

        仅支持方阵：``W[index]`` 的长度是 ``in_features``、``ΔW[:, index]``
        的长度是 ``out_features``，只有 ``in == out`` 时两者才能相加——
        bigram 矩阵正好是方阵。
        """
        if self.in_features != self.out_features:
            raise PEFTConfigError(
                f"forward_context 只适用于方阵（bigram / 查表式权重），本层是 "
                f"{self.out_features}×{self.in_features}"
            )
        if not 0 <= index < self.out_features:
            raise PEFTConfigError(f"行下标 {index} 越界：本层有 {self.out_features} 行")
        mask_value = self._onehot_mask(index, training=training)
        base_row = self._base[index]
        activation = [self._a[j][index] * mask_value for j in range(self.r)]
        delta_out = [
            self.scaling * sum(self._b[i][j] * activation[j] for j in range(self.r))
            for i in range(self.out_features)
        ]
        cached_input = [0.0] * self.in_features
        cached_input[index] = 1.0
        self._cached_input = cached_input
        self._cached_activation = activation
        self._cached_mask = [1.0] * self.in_features
        self._cached_mask[index] = mask_value
        return [base_row[i] + delta_out[i] for i in range(self.out_features)]

    def _backward_into_column(self, index: int, grad_out: Sequence[float]) -> None:
        """把 ``∂L/∂A`` / ``∂L/∂B`` 累加进缓冲，``A`` 只更新第 ``index`` 列.

        行约定与列约定的反向**完全一样**：基座是冻结的，梯度只流向 ``A`` / ``B``，
        而 ``∂L/∂a_j = scaling · Σ_i g_i B[i][j]`` 与"基座用的是哪一行/哪一列"
        无关。这也是为什么"合并后与适配器模型逐位一致"必须靠**前向**的不变式
        来守，而不能靠反向。

        前置条件（"必须先走一次前向"）由 ``_cached_index()`` 负责检查——两个
        调用方都会先经过它，所以这里不再重复一遍。**一个永远走不到的防御分支
        会让覆盖率这个指标失去意义，比"少写一行防御"更糟。**
        """
        if len(grad_out) != self.out_features:
            raise PEFTConfigError(
                f"输出梯度维度不匹配：本层输出 {self.out_features} 维，收到 {len(grad_out)}"
            )
        activation = self._cached_activation
        mask_value = (self._cached_mask or [1.0] * self.in_features)[index]
        scaling = self.scaling
        grad_activation = [0.0] * self.r
        for i in range(self.out_features):
            factor = scaling * grad_out[i]
            if factor == 0.0:
                continue
            b_row = self._b[i]
            grad_row = self._grad_b[i]
            for j in range(self.r):
                grad_row[j] += factor * activation[j]
                grad_activation[j] += factor * b_row[j]
        for j in range(self.r):
            self._grad_a[j][index] += grad_activation[j] * mask_value
        self._pending += 1

    def _cached_index(self) -> int:
        """从缓存里取出上次前向用的行/列下标（one-hot 位置）."""
        if self._cached_input is None:
            raise PEFTConfigError("缺少前向缓存：请先调用前向")
        index = next(
            (position for position, value in enumerate(self._cached_input) if value != 0.0),
            -1,
        )
        if index < 0:  # pragma: no cover - 前向必定写入一个非零位
            raise PEFTConfigError("缓存里没有非零位：请使用 forward_onehot / forward_context")
        return index

    def backward_onehot(self, grad_out: Sequence[float]) -> None:
        """``forward_onehot`` 之后的反向（``O(r·out)``）."""
        self._backward_into_column(self._cached_index(), grad_out)

    def backward_context(self, grad_out: Sequence[float]) -> None:
        """``forward_context`` 之后的反向（``O(r·out)``）.

        ``A`` 的这次更新只碰**一列**（第 ``index`` 列），从计算图上看就是
        "每个上下文 token 只训练到它自己那一列"——训练数据里出现得少的
        上下文 token，其对应列几乎学不到东西。这是 LoRA 在小数据上的一个
        结构性限制，也是本课 rank 扫里"加大秩不一定更好"的原因之一。
        """
        self._backward_into_column(self._cached_index(), grad_out)

    # ------------------------------------------------------------------ 更新
    def zero_grad(self) -> None:
        """清零梯度缓冲与待处理计数（丢弃未应用的累积梯度）."""
        self._grad_a = _zeros(self.r, self.in_features)
        self._grad_b = _zeros(self.out_features, self.r)
        self._pending = 0

    def gradient_snapshot(self) -> tuple[list[list[float]], list[list[float]], int]:
        """记录梯度缓冲的当前状态（评估前后恢复用）.

        评估必须**不污染梯度缓冲**，否则"训练 — 评估 — 继续训练"会把评估
        的梯度混进下一次更新。与 day050 ``ReferenceSFTModel.evaluate`` 的
        "记下 → 评估 → 恢复"是同一种做法：比"相信没人会忘记 zero_grad"可靠。
        """
        return (
            [row[:] for row in self._grad_a],
            [row[:] for row in self._grad_b],
            self._pending,
        )

    def restore_gradients(
        self, snapshot: tuple[list[list[float]], list[list[float]], int]
    ) -> None:
        """恢复 ``gradient_snapshot`` 记录的梯度缓冲状态."""
        grad_a, grad_b, pending = snapshot
        self._grad_a = [row[:] for row in grad_a]
        self._grad_b = [row[:] for row in grad_b]
        self._pending = pending

    def apply_update(self, learning_rate: float) -> None:
        """按 ``1/N`` 缩放累加梯度并做一次 SGD，然后清零缓冲.

        ``N`` 是自上次更新以来累积的**位置数**（不是批数、也不是参数量）。
        用错分母会让有效学习率被悄悄缩放——这与 day050 里"loss 的分母只数
        监督 token"是同一个纪律在梯度层面的体现。
        """
        if learning_rate <= 0:
            raise PEFTConfigError(f"learning_rate 必须为正数，收到 {learning_rate}")
        if self._pending == 0:
            raise PEFTConfigError("没有待应用的梯度：apply_update 之前必须先 accumulate")
        scale = 1.0 / self._pending
        for j in range(self.r):
            a_row = self._a[j]
            grad_row = self._grad_a[j]
            for index in range(self.in_features):
                a_row[index] -= learning_rate * grad_row[index] * scale
        for i in range(self.out_features):
            b_row = self._b[i]
            grad_row = self._grad_b[i]
            for j in range(self.r):
                b_row[j] -= learning_rate * grad_row[j] * scale
        self._updates += 1
        self.zero_grad()

    # --------------------------------------------------------------- 结构信息
    def delta_for_context(self, index: int) -> list[float]:
        """给定上下文 token，增量向量 ``scaling · B @ A[:, index]``（``O(r·out)``）.

        = ``ΔW`` 的第 ``index`` 列，长度 ``out_features``。逐行合并
        （``merged_by_context``）就建立在它之上。
        """
        return context_delta(self._a, self._b, index, self.scaling)

    def delta(self) -> list[list[float]]:
        """当前增量矩阵 ``scaling · B @ A``."""
        return lora_delta(self._a, self._b, self.scaling)

    def merged_weight(self) -> list[list[float]]:
        """密集约定下的合并权重 ``W + scaling · B @ A``（与基座同形）.

        **它是 ``forward`` 对应的合并结果，不是 ``forward_context`` 的。**
        ``forward_context`` 用的是 ``Wᵀ``，因此它对应的合并矩阵是
        ``W + ΔWᵀ``（见 ``merged_by_context``）——两条约定用的增量互为转置，
        在非对称基座上落错一个，输出会全错而不会有任何报错。
        """
        return merge_lora_weight(self._base, self._a, self._b, self.scaling)

    def merged_by_context(self) -> list[list[float]]:
        """按上下文合并出的权重矩阵 ``M = W + ΔWᵀ``，即 ``M[i][j] = W[i][j] + ΔW[j][i]``.

        它是 ``merged_weight()`` 在增量上取了转置的结果（基座不变）。

        参考模型走的是这一条：``LoRAReferenceModel.merge()`` 把 ``M`` 交给
        一个普通 ``ReferenceSFTModel``，于是"带适配器的模型"与"合并后的
        模型"在全部上下文 token 上给出**逐位相同**的 logits（实测差 0.0e+00）。
        """
        if self.in_features != self.out_features:
            raise PEFTConfigError(
                "merged_by_context 只适用于方阵（查表式权重），本层是 "
                f"{self.out_features}×{self.in_features}"
            )
        return [
            [
                self._base[index][value] + delta
                for value, delta in enumerate(self.delta_for_context(index))
            ]
            for index in range(self.out_features)
        ]

    def delta_is_zero(self, *, tolerance: float = ZERO_TOLERANCE) -> bool:
        """增量是否为 0（``gaussian`` 初始化下恒为 True，直到第一次更新）."""
        return max_abs(self.delta()) <= tolerance

    def describe(self) -> dict[str, float | int]:
        """本层增量的画像（转发 ``describe_delta``）."""
        return describe_delta(self._a, self._b, self.scaling)

    def adapter_state(self) -> dict[str, object]:
        """适配器状态（``A`` / ``B`` / 更新次数），用于落盘与恢复."""
        return {
            "a": [row[:] for row in self._a],
            "b": [row[:] for row in self._b],
            "updates": self._updates,
            "r": self.r,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "scaling": self.scaling,
        }

    def load_adapter_state(self, state: dict[str, object]) -> None:
        """载入适配器状态；形状不一致直接拒绝（避免静默错位）."""
        a_matrix = state.get("a")
        b_matrix = state.get("b")
        if not isinstance(a_matrix, list) or not isinstance(b_matrix, list):
            raise PEFTConfigError("适配器状态缺少 a / b 矩阵")
        rows_a, cols_a = _check_matrix(a_matrix, name="A")
        rows_b, cols_b = _check_matrix(b_matrix, name="B")
        if (rows_a, cols_a) != (self.r, self.in_features):
            raise PEFTConfigError(
                f"A 的形状应为 {self.r}×{self.in_features}，收到 {rows_a}×{cols_a}"
            )
        if (rows_b, cols_b) != (self.out_features, self.r):
            raise PEFTConfigError(
                f"B 的形状应为 {self.out_features}×{self.r}，收到 {rows_b}×{cols_b}"
            )
        if float(state.get("scaling", self.scaling)) != self.scaling:
            raise PEFTConfigError(
                "适配器状态的 scaling 与本层配置不一致："
                f"{state.get('scaling')} vs {self.scaling}（换 r 或 alpha 后必须重新合并）"
            )
        self._a = [[float(value) for value in row] for row in a_matrix]
        self._b = [[float(value) for value in row] for row in b_matrix]
        self._updates = int(state.get("updates", 0))  # type: ignore[arg-type]
        self.zero_grad()

    def snapshot_matrices(self) -> tuple[list[list[float]], list[list[float]]]:
        """返回 ``(A, B)`` 的拷贝（测试与诊断用，避免外部改动内部状态）."""
        return [row[:] for row in self._a], [row[:] for row in self._b]


__all__ = [
    "ZERO_TOLERANCE",
    "LoRALinear",
    "adapter_state_size",
    "add_matrices",
    "describe_delta",
    "frobenius_norm",
    "init_lora_weights",
    "is_rank_one",
    "lora_delta",
    "matmul",
    "matrix_rank",
    "max_abs",
    "merge_lora_weight",
    "context_delta",
]
