"""``positional_encoding`` 的形状、口径表与三条记录（day078 / M7-D3）.

day075 交给下游两句话：**这一层的 ``grad_inputs`` 是为了“堆叠”准备的**，
以及**无掩码时注意力是置换等变的——这正是需要位置编码的原因**。
day076 把“看什么”切成 ``heads`` 份，但一个字也没有回答“在第几位”。

今天只做一件事：**在进那一层之前，把“第几位”加到输入上**。

```text
day075   attention(x)                    输出只依赖 x 的**内容**（无掩码时换序就换序）
day078   attention(inject(x, PE))        注入之后换序**不再**等变——这才是重点
```

## 一、六个阶段（比 day075 多出“造表”与“相加”两步，而且它们在**最前面**）

```text
positions  n → (n,)                     选位置：这一批 token 在第几位
table      (n,) → (n, d)                造表：每个位置一行（**唯一的“位置知识”来源**）
add        (n, d_in) + (n, d) → (n, d)  相加：注入——形状必须逐行对齐
attend     day075 的七步（本课把它当黑盒复用，一行未改）
project    (n, d_v) → (n, d_out)        W_o：位置信息在这里也能被**选择性忽略**
loss       只在监督行上的 MSE
```

多出来的两步都在最前面，这一点有具体后果：**反向时它们也在最后面**，
而“最后一步”恰好是最容易被写错的一步（见 :func:`scatter_add_rows`）。

## 二、三条记录

```text
EncodingTable       一张表：种类 / 行 / 频率基数 / 配对方式 / 注记
PositionalForward   一次前向：原始输入、取出的表行、注入结果、注意力账、损失
PositionalGradients 一次反向：表 / 输入 / 四个投影（**比 day075 多一项**）
```

## 三、六项梯度校验的名单（比 day075 多一项）

```text
w_output → w_value → w_query → w_key → table → inputs
                                        └── 这一项是本课的全部内容
```

顺序上 ``table`` 排在 ``inputs`` 之前，因为它离损失**更近一步**：
``injected = inputs + table[positions]``，反向时
``dTable`` 是“按位置聚合”，``dInputs`` 是“逐位原样传回”。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

from smart_research_agent.math_foundations.types import (
    DEFAULT_TOLERANCE,
    Matrix,
    Vector,
    matrix_shape,
    validate_matrix,
    validate_vector,
)
from smart_research_agent.positional_encoding.errors import (
    NumericError,
    ParameterError,
    RangeError,
    ShapeError,
)
from smart_research_agent.transformer_core.types import (
    AttentionForward,
    AttentionParams,
)

#: 位置编码的两种“长相”（本课的全部内容就是这两行的差别）.
ENCODING_SINUSOIDAL = "sinusoidal"
ENCODING_LEARNABLE = "learnable"
ENCODING_KINDS: tuple[str, ...] = (ENCODING_SINUSOIDAL, ENCODING_LEARNABLE)

ENCODING_DESCRIPTIONS: dict[str, str] = {
    ENCODING_SINUSOIDAL: (
        "固定正弦/余弦：位置 p 的行由**公式**给出，没有任何参数——"
        "因此表长之外仍然有定义（外推是算出来的，不是猜出来的）"
    ),
    ENCODING_LEARNABLE: (
        "可学习表：每个位置一行、行本身是参数——它只能覆盖表内的位置，"
        "越界**必须拒绝**（表长是一条硬边界，不是一个建议）"
    ),
}

#: 频率的两种“配对方式”（原论文用对齐；day073 的 ``attention.positional_encoding`` 用了错位）.
PAIRING_ALIGNED = "aligned"
PAIRING_STAGGERED = "staggered"
PAIRINGS: tuple[str, ...] = (PAIRING_ALIGNED, PAIRING_STAGGERED)

PAIRING_DESCRIPTIONS: dict[str, str] = {
    PAIRING_ALIGNED: (
        "对齐：``sin`` 与 ``cos`` 用**同一个**频率 "
        "``f_i = base^(2i/d)``（原论文）——于是每一对的和是 ``sin² + cos² = 1``，"
        "而位移 δ 恰好是一个正交旋转"
    ),
    PAIRING_STAGGERED: (
        "错位：``cos`` 用**隔壁**那个频率 "
        "``base^((2i+1)/d)``（day073 的写法）——于是 ``sin² + cos²`` 不成立、"
        "位移也不再是一个旋转，两条性质一起失效"
    ),
}

#: 可学习表的两种初始化（**从正弦表出发**是实践中的默认做法）.
INIT_SINUSOIDAL = "sinusoidal"
INIT_RANDOM = "random"
INITIALIZERS: tuple[str, ...] = (INIT_SINUSOIDAL, INIT_RANDOM)

INITIALIZER_DESCRIPTIONS: dict[str, str] = {
    INIT_SINUSOIDAL: "初始化为正弦表：训练从一个**已经知道怎么按位置读**的点出发",
    INIT_RANDOM: "随机初始化：位置知识必须**全部**从数据里学出来（本课用它做对照）",
}

#: 可学习表随机初始化的幅度（与 day075 的 ``DEFAULT_INIT_SCALE`` 同值）。
INIT_SCALE = 0.25

#: 位移律的判据：``<PE(p), PE(q)>`` 只依赖 ``p − q``。
#:
#: 取 ``1e-12`` 而不是 ``==``：闭式 ``Σ_i cos(δ/f_i)`` 与实测的“两行做点积"
#: 是**两条不同的求和路径**（一个按频率对相加，一个按维度相加），
#: 浮点下差最后几位。day076 的 ``merge→project`` 恒等式是同一类情形。
OFFSET_TOLERANCE = 1e-12

#: 行范数的判据：正弦表每一行的 L2 范数**恰好**是 ``√(d/2)``。
ROW_NORM_TOLERANCE = 1e-12

#: 六个阶段的名字（**顺序就是数据流的顺序**）.
STAGE_POSITIONS = "positions"
STAGE_TABLE = "table"
STAGE_ADD = "add"
STAGE_ATTEND = "attend"
STAGE_PROJECT = "project"
STAGE_LOSS = "loss"

POSITIONAL_STAGES: tuple[str, ...] = (
    STAGE_POSITIONS,
    STAGE_TABLE,
    STAGE_ADD,
    STAGE_ATTEND,
    STAGE_PROJECT,
    STAGE_LOSS,
)

POSITIONAL_STAGE_DESCRIPTIONS: dict[str, str] = {
    STAGE_POSITIONS: "选位置：这一批 token 的每一行在第几位（**可以重复**——一批两条序列时位置会重复）",
    STAGE_TABLE: "造表：位置 → 一行 d 维向量（正弦表由公式给出、可学习表由参数给出）",
    STAGE_ADD: "相加：injected = inputs + table[positions]，**逐行对齐**——这就是“注入”的全部动作",
    STAGE_ATTEND: "注意力：day075 的七步，作用在 injected 上（本课一行未改那一层）",
    STAGE_PROJECT: "输出投影：y = context·W_oᵀ（W_o 可以**选择性忽略**位置那部分分量）",
    STAGE_LOSS: "损失：只在监督行上的 MSE（与 day075 同一口径）",
}

POSITIONAL_STAGE_SHAPES: dict[str, str] = {
    STAGE_POSITIONS: "n → (n,)（每个元素都必须是表里存在的行下标）",
    STAGE_TABLE: "(n,) → (n, d)",
    STAGE_ADD: "(n, d_in) + (n, d) → (n, d)，要求 d_in == d",
    STAGE_ATTEND: "(n, d) × 四个投影 → (n, d_v)",
    STAGE_PROJECT: "(n, d_v) × (d_out, d_v)ᵀ → (n, d_out)",
    STAGE_LOSS: "(n, d_out) × (n, d_out) → 标量",
}

#: 六个梯度目标（**比 day075 多出 ``table``**）.
GRAD_TABLE = "table"
GRAD_INPUTS = "inputs"
GRAD_W_QUERY = "w_query"
GRAD_W_KEY = "w_key"
GRAD_W_VALUE = "w_value"
GRAD_W_OUTPUT = "w_output"

POSITIONAL_GRADIENT_TARGETS: tuple[str, ...] = (
    GRAD_W_OUTPUT,
    GRAD_W_VALUE,
    GRAD_W_QUERY,
    GRAD_W_KEY,
    GRAD_TABLE,
    GRAD_INPUTS,
)

POSITIONAL_GRADIENT_FORMULAS: dict[str, str] = {
    GRAD_W_OUTPUT: "dW_o = dOutᵀ · attention_sum：与 day075 **逐字相同**（位置编码不进这一项）",
    GRAD_W_VALUE: "dV = weightsᵀ · dInjected；dW_v = dVᵀ · injected",
    GRAD_W_QUERY: "dScores ← softmax 反向；dQ = dRaw · K；dW_q = dQᵀ · injected",
    GRAD_W_KEY: "dK = dRawᵀ · Q；dW_k = dKᵀ · injected",
    GRAD_TABLE: "dTable[p] = Σ_{i: positions[i]=p} dInjected[i]——**共享表要按位置相加**",
    GRAD_INPUTS: "dx = dInjected（**逐位**：加法注入的导数恰好是恒等映射）",
}

POSITIONAL_GRADIENT_DESCRIPTIONS: dict[str, str] = {
    GRAD_W_OUTPUT: "输出投影：反向的第一站，本课与 day075 完全一致",
    GRAD_W_VALUE: "value 投影：位置信息经由 V 进入加权平均的那条路",
    GRAD_W_QUERY: "query 投影：位置信息也可以改变'这一行想问什么'",
    GRAD_W_KEY: "key 投影：位置信息还可以改变'这一行有多容易被问中'",
    GRAD_TABLE: "位置表：**本课唯一新增的一项**，它把 batch 里所有同位置的行加起来",
    GRAD_INPUTS: "输入：加法注入的偏导数恰好是 1，因此它是逐位原样传回（不会变小）",
}

#: 七条性质的名单.
PROPERTY_CONSTANT_NORM = "rows_have_constant_norm"
PROPERTY_OFFSET_ONLY = "dot_product_depends_only_on_offset"
PROPERTY_SHIFT_IS_ROTATION = "shift_is_an_orthogonal_rotation"
PROPERTY_BREAKS_EQUIVARIANCE = "additive_injection_breaks_permutation_equivariance"
PROPERTY_ADDITIVE_BACKWARD = "additive_backward_is_the_identity"
PROPERTY_LENGTH_IS_A_HARD_BOUND = "learnable_table_length_is_a_hard_bound"
PROPERTY_STAGGERED_BREAKS_THE_LAWS = "staggered_pairing_breaks_both_laws"

POSITIONAL_PROPERTIES: tuple[str, ...] = (
    PROPERTY_CONSTANT_NORM,
    PROPERTY_OFFSET_ONLY,
    PROPERTY_SHIFT_IS_ROTATION,
    PROPERTY_BREAKS_EQUIVARIANCE,
    PROPERTY_ADDITIVE_BACKWARD,
    PROPERTY_LENGTH_IS_A_HARD_BOUND,
    PROPERTY_STAGGERED_BREAKS_THE_LAWS,
)

POSITIONAL_PROPERTY_DESCRIPTIONS: dict[str, str] = {
    PROPERTY_CONSTANT_NORM: (
        "正弦表（对齐频率）每一行的 L2 范数**恰好**是 ``√(d/2)``——"
        "这一条只在偶数维、且 ``cos`` 用同一个频率时成立"
    ),
    PROPERTY_OFFSET_ONLY: (
        "``<PE(p), PE(q)>`` 只依赖 ``p − q``——**相对距离可以从绝对编码里读出来**，"
        "这正是原论文那句'模型可以学会用相对位置'的全部依据"
    ),
    PROPERTY_SHIFT_IS_ROTATION: (
        "``PE(p + δ)`` **逐位**等于 ``R_δ · PE(p)``（``R_δ`` 是分块正交旋转）——"
        "它是上一条的**构造性理由**，而不是又一条观察"
    ),
    PROPERTY_BREAKS_EQUIVARIANCE: (
        "注入之后，把输入行换序**不再**是'输出行按同样方式换序'——"
        "**这是本课要的效果，不是 bug**：注意力原本不知道顺序，位置编码把它变成了知道"
    ),
    PROPERTY_ADDITIVE_BACKWARD: (
        "``dx`` **逐位**等于 ``dInjected``，而 ``dTable[p]`` 是“所有落在第 p 位的行之和”——"
        "把“按位置相加”写成“按行覆盖”不会报错，只会让共享表的梯度少掉一部分"
    ),
    PROPERTY_LENGTH_IS_A_HARD_BOUND: (
        "可学习表对越界位置**拒绝**（``RangeError``），而正弦表对同一个位置**算得出来**——"
        "两种编码的语义差别就在这里"
    ),
    PROPERTY_STAGGERED_BREAKS_THE_LAWS: (
        "把 ``cos`` 换成“隔壁”的频率（day073 的写法）之后，"
        "**行范数恒定**与**位移律**一起失效——判据必须能抓住“看起来对但公式错位”的实现"
    ),
}

#: 位置表里“没写出来的东西”的三条纪律（供报告引用）.
POSITIONAL_NOTES: tuple[str, ...] = (
    "正弦表没有任何参数：它的每一行都是一个公式的取值，因此 parameter_count() 恒为 0",
    "可学习表的行数与宽度都是参数：它的表长是一条硬边界，越界必须拒绝而不是外推",
    "位置信息与内容信息在加法里**互相叠加**：注入之后无法再把两者分开，"
    "所以'模型到底用了位置还是用了内容'要由消融实验回答，不能由一次前向回答",
)


def _checked_positive_int(value: Any, *, name: str) -> int:
    """校验“正整数”（非整数或 < 1 抛 ``ParameterError``）."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ParameterError(f"{name} 必须是整数，收到 {value!r}。")
    if value < 1:
        raise ParameterError(f"{name} 必须 >= 1，收到 {value}。")
    return value


def _checked_offset(value: Any, *, name: str = "offset") -> int:
    """校验“非负整数位移”（位移可以是 0，但不可以是负数）."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ParameterError(f"{name} 必须是整数，收到 {value!r}。")
    if value < 0:
        raise ParameterError(f"{name} 必须 >= 0，收到 {value}。")
    return value


@dataclass(frozen=True)
class PositionalShape:
    """这一层的形状：**四个维度，其中两个必须相等**.

    ```text
    inputs      d_in     输入的列数
    positions   n        这一批有多少行（每一行一个位置下标）
    dimension   d        位置表的宽度（**必须等于 d_in**）
    outputs     d_out    输出投影的列数
    ```

    为什么要同时留下 ``inputs`` 与 ``dimension`` 两个字段——它们相等，
    这件事本身就是要被写下来的：**加法要求两边逐行对齐**，
    而“表比输入宽”或“表比输入窄”都不是“少算几项”的问题，是“这个加法没有定义”。
    把相等写进 ``__post_init__`` 之后，“两个字段相等”变成一条**可被断言的契约**，
    而不是一句注释。
    """

    inputs: int
    positions: int
    dimension: int
    outputs: int

    def __post_init__(self) -> None:
        for name in ("inputs", "positions", "dimension", "outputs"):
            _checked_positive_int(getattr(self, name), name=name)
        if self.dimension != self.inputs:
            raise ShapeError(
                f"位置表的宽度 {self.dimension} 与输入列数 {self.inputs} 不一致："
                "注入是**逐位相加**，它要求两边的行宽逐行对齐——"
                "宽度不同时这个加法根本没有定义（不是'少算几项'）。"
            )

    @property
    def table_shape(self) -> tuple[int, int]:
        """位置表的形状 ``(positions, dimension)``."""
        return (self.positions, self.dimension)

    @property
    def pairs(self) -> int:
        """频率对的个数 ``d // 2``（奇数维时最后一维没有配对）."""
        return self.dimension // 2

    @property
    def even_dimension(self) -> bool:
        """维数是否为偶数——**正弦表的两条性质都要求它**."""
        return self.dimension % 2 == 0

    @property
    def row_norm(self) -> float:
        """正弦表每一行的范数 ``√(d/2)``.

        它只在偶数维时是“真实范数”：奇数维的最后一维只有一个 ``sin``、
        没有配对的 ``cos``，于是那一项的取值范围是 ``[0, 1]`` 而不是 ``1``——
        本属性仍然返回 ``√(d/2)``，是为了让“奇数维对不上”这件事**在比较中暴露出来**，
        而不是让调用方拿到一个“看起来对”的常数。
        """
        return math.sqrt(self.dimension / 2.0)

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状（含两个派生量）."""
        return {
            "inputs": self.inputs,
            "positions": self.positions,
            "dimension": self.dimension,
            "outputs": self.outputs,
            "pairs": self.pairs,
            "even_dimension": self.even_dimension,
            "row_norm": self.row_norm,
        }

    def summary_line(self) -> str:
        """一行说明：``d_in=4 d=4 d_out=4 | n=4 | 频率对 2 | 每行范数 √2 = 1.414214``."""
        parity = "偶数维" if self.even_dimension else "**奇数维**（最后一位没有配对）"
        return (
            f"d_in={self.inputs} d={self.dimension} d_out={self.outputs} | "
            f"n={self.positions} | {parity} | 频率对 {self.pairs} | "
            f"每行范数 √{self.dimension}/2 = {self.row_norm:.6f}"
        )


def validate_positions(
    positions: Any,
    *,
    expected: int,
    table_positions: int,
) -> tuple[int, ...]:
    """校验一串位置下标：长度对得上、每一个都落在表的行范围内.

    ``table_positions`` 是表的行数。越界抛 :class:`RangeError` 而**不是**截断：
    把“第 20 位”截成“第 7 位”不会报错、不会产生非有限数，
    只会让模型看到另一个位置——而报告里“我查了第 20 位”这句话仍然原样写着。
    """
    if isinstance(positions, (str, bytes)) or not isinstance(positions, (tuple, list)):
        raise ShapeError(f"positions 必须是序列，收到 {type(positions).__name__}。")
    resolved: list[int] = []
    for index, value in enumerate(positions):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ShapeError(f"positions[{index}] 必须是整数，收到 {value!r}。")
        if value < 0:
            raise RangeError(
                f"positions[{index}] = {value} 是负位置：位置从 0 开始编号，"
                "负下标在 Python 里会**从尾部取**——那是一个静默的错误取值。"
            )
        if value >= table_positions:
            raise RangeError(
                f"positions[{index}] = {value} 超出了表的范围 [0, {table_positions})："
                "表不会替调用方外推——可学习表的第 "
                f"{value} 行根本不存在，而正弦表应当**直接算**而不是查表。"
            )
        resolved.append(value)
    if len(resolved) != expected:
        raise ShapeError(
            f"位置个数 {len(resolved)} 与输入行数 {expected} 不一致："
            "注入是逐行相加，一个位置对应一行——个数不同时'哪一行用哪个位置'没有定义。"
        )
    return tuple(resolved)


def default_positions(size: int) -> tuple[int, ...]:
    """缺省的位置序列 ``(0, 1, …, size-1)``（一条长度为 ``size`` 的序列）."""
    _checked_positive_int(size, name="size")
    return tuple(range(size))


def gather_rows(table: Matrix, positions: tuple[int, ...]) -> Matrix:
    """按位置从表里**取行**（``gather``）：返回 ``(n, d)`` 的矩阵.

    它只是“取”——不涉及任何算术，因此它是这一层最不容易出错的一步，
    也是反向时唯一**不需要**求导的一步（下标是可微的：``dk/dθ = 0``）。
    """
    checked = validate_matrix(table, name="table")
    rows: list[Vector] = []
    for index, position in enumerate(positions):
        if position < 0 or position >= len(checked):
            raise RangeError(
                f"positions[{index}] = {position} 越界（表有 {len(checked)} 行）："
                "取行之前必须先用 validate_positions 校验，"
                "因为 Python 的负下标会静默地从尾部取。"
            )
        rows.append(checked[position])
    if not rows:
        raise ShapeError("positions 为空：没有行可以取。")
    return tuple(rows)


def scatter_add_rows(
    row_gradients: Matrix,
    positions: tuple[int, ...],
    *,
    table_positions: int,
) -> Matrix:
    """把“逐行的梯度”按位置**累加**回表（``scatter-add``）：反向里唯一新增的一步.

    ## 为什么必须是“累加”而不是“覆盖”

    前向是 ``injected[i] = inputs[i] + table[positions[i]]``。一行梯度
    ``dInjected[i]`` 因此**整份**属于 ``table[positions[i]]``。当两个下标
    ``i ≠ j`` 有 ``positions[i] == positions[j]`` 时（一批两条序列、或者
    一条序列里同一个位置被用了两次），第 ``p`` 行拿到的梯度是**两份之和**。

    ```python
    for i in range(n):
        grad_table[positions[i]] = row_gradients[i]      # ← 静默错误
    ```

    上面这种写法**形状完全正确**、也不会报错：当所有位置互不相同时它是对的，
    而当位置重复时它只保留了**最后一次**写入——表的梯度于是偏小，
    而“偏小”在训练里只表现为“位置编码学得慢一点”。

    这正是 day075/076 那类“漏掉不会报错”的错误在本层的化身，而它的判据只有一条：
    **解析梯度必须与数值差分对得上**（重复位置的那条样本专门测它）。
    """
    checked = validate_matrix(row_gradients, name="row_gradients")
    _checked_positive_int(table_positions, name="table_positions")
    width = matrix_shape(checked)[1]
    columns: list[list[float]] = [[0.0] * width for _ in range(table_positions)]
    for row, position in zip(checked, positions, strict=True):
        if position < 0 or position >= table_positions:
            raise RangeError(
                f"位置 {position} 越界（表有 {table_positions} 行）："
                "累加回表之前必须先用 validate_positions 校验。"
            )
        target = columns[position]
        for index, value in enumerate(row):
            target[index] += value
    return tuple(tuple(row) for row in columns)


def add_matrices(left: Matrix, right: Matrix) -> Matrix:
    """逐位相加（形状不同抛 ``ShapeError``）——**注入**的全部算术就是这一行."""
    checked_left = validate_matrix(left, name="left")
    checked_right = validate_matrix(right, name="right")
    if matrix_shape(checked_left) != matrix_shape(checked_right):
        raise ShapeError(
            f"相加的两个矩阵形状不同：{matrix_shape(checked_left)} 与 "
            f"{matrix_shape(checked_right)}——注入要求两边逐行对齐。"
        )
    return tuple(
        tuple(a + b for a, b in zip(left_row, right_row, strict=True))
        for left_row, right_row in zip(checked_left, checked_right, strict=True)
    )


def matrix_row_norms(matrix: Matrix) -> Vector:
    """每一行的 L2 范数（用于“行范数恒定”这条性质）."""
    checked = validate_matrix(matrix, name="matrix")
    return tuple(math.sqrt(math.fsum(value * value for value in row)) for row in checked)


@dataclass(frozen=True)
class EncodingTable:
    """一张位置表：**本课唯一的“位置知识”载体**.

    ```text
    kind       sinusoidal / learnable（两种“长相”，见 ENCODING_DESCRIPTIONS）
    table      (positions, dimension) 的矩阵——两种编码都最终落到“一张表”上
    base       频率基数（只有正弦表有；缺省 10000.0，与 day073 同值）
    pairing    aligned / staggered（只有正弦表有；见 PAIRING_DESCRIPTIONS）
    notes      注记（“它到底是怎么来的”）
    ```

    为什么两种编码共用一条记录：**注入那一步对两者一无所知**——
    它拿到的是“一张 (n, d) 的表”和“一串位置下标”，然后逐位相加。
    这个刻意的统一有具体好处：正弦表与可学习表可以走**同一条**前向、反向与
    梯度校验，于是“换一种编码”这件事在代码里只发生在**造表**那一行。
    """

    kind: str
    table: Matrix
    base: float | None = None
    pairing: str | None = None
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.kind not in ENCODING_KINDS:
            raise ParameterError(
                f"未知的编码种类 {self.kind!r}：可选 {', '.join(ENCODING_KINDS)}——"
                "本包不为未知种类挑一个默认值。"
            )
        checked = validate_matrix(self.table, name="table")
        if not checked:
            raise ShapeError("位置表不能为空：至少要有一行。")
        object.__setattr__(self, "table", checked)
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))
        if self.kind == ENCODING_SINUSOIDAL:
            if self.base is None:
                raise ParameterError("正弦表必须给出频率基数 base。")
            if not math.isfinite(self.base) or self.base <= 1.0:
                raise ParameterError(
                    f"频率基数必须 > 1，收到 {self.base!r}：base = 1 时所有频率都是 1，"
                    "每一维都在同一个尺度上振荡（位置信息会退化成'全都一样')。"
                )
            if self.pairing not in PAIRINGS:
                raise ParameterError(
                    f"未知的频率配对方式 {self.pairing!r}：可选 {', '.join(PAIRINGS)}。"
                )
        else:
            if self.base is not None or self.pairing is not None:
                raise ParameterError(
                    "可学习表没有频率基数与配对方式：它的每一行都是参数，"
                    "不是任何公式的取值——把 base/pairing 留在记录里会让人以为"
                    "它还有一个'公式'可以对照。"
                )

    # ------------------------------------------------------------------ 派生量

    @property
    def positions(self) -> int:
        """表的行数（**它是一条硬边界**：可学习表只能覆盖 ``[0, positions)``）."""
        return len(self.table)

    @property
    def dimension(self) -> int:
        """表的宽度."""
        return matrix_shape(self.table)[1]

    @property
    def learnable(self) -> bool:
        """这张表的行是不是参数（决定“位置知识”能不能被训练改）.  """
        return self.kind == ENCODING_LEARNABLE

    @property
    def parameter_count(self) -> int:
        """参数个数：正弦表恒为 ``0``，可学习表是 ``positions × dimension``."""
        return self.positions * self.dimension if self.learnable else 0

    @property
    def pairing_is_aligned(self) -> bool:
        """是不是“对齐频率”的正弦表（两条正弦表性质都以它为**前提**）."""
        return self.kind == ENCODING_SINUSOIDAL and self.pairing == PAIRING_ALIGNED

    @property
    def norms(self) -> Vector:
        """每一行的 L2 范数."""
        return matrix_row_norms(self.table)

    @property
    def aligned_within(self) -> bool:
        """每一行的范数是否都等于 ``√(d/2)``（**恰好**，容差 ``ROW_NORM_TOLERANCE``）.

        偶数维 + 对齐频率时它为 ``True``；错位频率或奇数维时为 ``False``。
        一个“看起来像正弦表”的错位实现会在这一条上露出来。
        """
        target = math.sqrt(self.dimension / 2.0)
        return all(
            abs(value - target) <= ROW_NORM_TOLERANCE * max(1.0, target)
            for value in self.norms
        )

    # ------------------------------------------------------------------ 取用

    def row(self, position: int) -> Vector:
        """取第 ``position`` 行（越界抛 :class:`RangeError`，**绝不外推**）."""
        if isinstance(position, bool) or not isinstance(position, int):
            raise ParameterError(f"position 必须是整数，收到 {position!r}。")
        if position < 0 or position >= self.positions:
            raise RangeError(
                f"位置 {position} 超出表的范围 [0, {self.positions})："
                f"这张表是 {ENCODING_DESCRIPTIONS[self.kind].split('：')[0]}，"
                "它不会替调用方外推——要更长的范围请造一张更长的表。"
            )
        return self.table[position]

    def extends_to(self, position: int) -> bool:
        """这张表能否为 ``position`` 给出一个值（正弦表恒为 ``True``）."""
        _checked_offset(position, name="position")
        if self.kind == ENCODING_SINUSOIDAL:
            return True
        return position < self.positions

    def inner_product(self, left: int, right: int) -> float:
        """第 ``left`` 行与第 ``right`` 行的内积（两行都必须在表内）."""
        checked_left = self.row(_checked_offset(left, name="left"))
        checked_right = self.row(_checked_offset(right, name="right"))
        return math.fsum(a * b for a, b in zip(checked_left, checked_right, strict=True))

    # ------------------------------------------------------------------ 压平 / 记录

    def flatten(self) -> tuple[Vector, tuple[tuple[int, int], ...]]:
        """把表压平成一串数，返回 ``(向量, 形状表)``（与 ``AttentionParams`` 同口径）."""
        flat: list[float] = []
        for row in self.table:
            flat.extend(row)
        return tuple(flat), (matrix_shape(self.table),)

    def with_table(self, table: Matrix) -> EncodingTable:
        """换一张同形的表（**保留** kind/base/pairing/notes）——反向与数值差分都用它."""
        checked = validate_matrix(table, name="table")
        if matrix_shape(checked) != matrix_shape(self.table):
            raise ShapeError(
                f"新表的形状 {matrix_shape(checked)} 与原来的 "
                f"{matrix_shape(self.table)} 不一致：换表不能改变形状，"
                "否则注入那一步会立刻对不上。"
            )
        return replace(self, table=checked)

    def describe(self) -> str:
        """一行说明这张表长什么样、参数有几个."""
        if self.learnable:
            origin = f"{self.positions} 行参数（{self.parameter_count} 个）"
        else:
            pairing = PAIRING_ALIGNED if self.pairing_is_aligned else PAIRING_STAGGERED
            origin = f"公式生成（base={self.base:g}、{pairing}）"
        return (
            f"{self.kind} 表 {self.positions}×{self.dimension} | {origin} | "
            f"行范数 {'恒定' if self.aligned_within else '不恒定'}"
        )

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状（表本身放在 ``table`` 里）."""
        return {
            "kind": self.kind,
            "description": ENCODING_DESCRIPTIONS[self.kind],
            "positions": self.positions,
            "dimension": self.dimension,
            "parameter_count": self.parameter_count,
            "base": self.base,
            "pairing": self.pairing,
            "row_norms": list(self.norms),
            "aligned_within": self.aligned_within,
            "table": [list(row) for row in self.table],
            "notes": list(self.notes),
        }

    def summary_line(self) -> str:
        """一行说明（用于演示脚本与报告）."""
        return self.describe()


@dataclass(frozen=True)
class PositionalForward:
    """一次位置注入 + 注意力前向的账.

    ```text
    inputs      原始输入 (n, d_in)——**没有**位置信息的那一份
    gathered    按 positions 取出的表行 (n, d)
    injected    inputs + gathered (n, d)——进注意力层的那一份
    attention   day075 那层的账（权重、输出、熵、峰值……）
    output      attention.output，也就是本层的输出
    loss        只在 supervised 行上的 MSE（没有给 target 时为 None）
    ```

    三条记录一起留下是刻意的：**“模型看到了什么”与“模型原本看到什么”必须同时可见**，
    否则“位置编码到底改了什么”就只能靠猜。演示脚本第 2 节把这三张表并排打印。
    """

    shape: PositionalShape
    table: EncodingTable
    positions: tuple[int, ...]
    inputs: Matrix
    gathered: Matrix
    injected: Matrix
    attention: AttentionForward
    target: Matrix | None = None
    supervised: tuple[int, ...] = field(default=())
    loss: float | None = None
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        checked_inputs = validate_matrix(self.inputs, name="inputs")
        checked_gathered = validate_matrix(self.gathered, name="gathered")
        checked_injected = validate_matrix(self.injected, name="injected")
        if matrix_shape(checked_inputs) != matrix_shape(checked_gathered):
            raise ShapeError(
                f"输入 {matrix_shape(checked_inputs)} 与取出的表行 "
                f"{matrix_shape(checked_gathered)} 形状不同：注入之前两者必须逐行对齐。"
            )
        if matrix_shape(checked_inputs) != matrix_shape(checked_injected):
            raise ShapeError(
                f"注入前后的形状不一致：{matrix_shape(checked_inputs)} 与 "
                f"{matrix_shape(checked_injected)}——相加不会改变形状。"
            )
        if len(self.positions) != matrix_shape(checked_inputs)[0]:
            raise ShapeError(
                f"位置个数 {len(self.positions)} 与输入行数 "
                f"{matrix_shape(checked_inputs)[0]} 不一致。"
            )
        if self.target is None:
            if self.loss is not None:
                raise NumericError(
                    "没有给 target 却带着 loss：损失是'输出与目标'之间的距离，"
                    "缺一边时它没有定义。"
                )
            if self.supervised:
                raise NumericError(
                    "没有给 target 却有监督行：监督行的意义就是'这些行参与损失'，"
                    "没有目标时它没有意义。"
                )
        else:
            checked_target = validate_matrix(self.target, name="target")
            if matrix_shape(checked_target) != matrix_shape(self.attention.output):
                raise ShapeError(
                    f"目标形状 {matrix_shape(checked_target)} 与输出形状 "
                    f"{matrix_shape(self.attention.output)} 不一致。"
                )
            if self.loss is None:
                raise NumericError("给了 target 却没有 loss：这一份前向的账不完整。")
        object.__setattr__(self, "inputs", checked_inputs)
        object.__setattr__(self, "gathered", checked_gathered)
        object.__setattr__(self, "injected", checked_injected)
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def tokens(self) -> int:
        """行数（这一批多少行）."""
        return matrix_shape(self.inputs)[0]

    @property
    def learnable(self) -> bool:
        """位置表是不是可学习的."""
        return self.table.learnable

    @property
    def input_norms(self) -> Vector:
        """注入**之前**每一行的范数（“内容”有多大）."""
        return matrix_row_norms(self.inputs)

    @property
    def injection_ratios(self) -> Vector:
        """每一行“位置占了多少”：``‖table[positions[i]]‖ / ‖inputs[i]‖``.

        分母为 0 的行（全零输入）返回 ``math.inf``——它**不是**一个可以比较的数，
        但这比“悄悄跳过这一行”好：一个全零的输入行在 day075 那层本来就会被拒绝
        （见 ``_check_inputs``），因此这个 ``inf`` 只会在“还没进那层”时出现，
        而它恰好说明“这一行没有任何内容可以注入位置”。
        """
        return tuple(
            (row_norm / input_norm) if input_norm > 0 else math.inf
            for row_norm, input_norm in zip(
                matrix_row_norms(self.gathered), self.input_norms, strict=True
            )
        )

    @property
    def mean_injection_ratio(self) -> float:
        """所有行的平均“位置/内容”比（有限值的平均）."""
        finite = [value for value in self.injection_ratios if math.isfinite(value)]
        if not finite:
            raise NumericError("所有输入行都是全零：没有任何'内容'可以比较位置的大小。")
        return math.fsum(finite) / len(finite)

    @property
    def supervised_rows(self) -> tuple[int, ...]:
        """参与损失的行."""
        return self.supervised

    def output_row(self, index: int) -> Vector:
        """输出的一行（越界抛 ``RangeError``）."""
        checked = self.attention.output
        if index < 0 or index >= len(checked):
            raise RangeError(f"输出行下标 {index} 越界（共 {len(checked)} 行）。")
        return checked[index]

    def summary_line(self) -> str:
        """一行说明：``n=4 d=4 | sinusoidal | 位置 (0,1,2,3) | 位置/内容比 1.414 | 损失 0.123456``."""
        loss = "—" if self.loss is None else f"{self.loss:.6f}"
        shown = ", ".join(str(value) for value in self.positions[:6])
        if len(self.positions) > 6:
            shown += ", …"
        return (
            f"n={self.tokens} d={self.shape.dimension} | {self.table.kind} | "
            f"位置 ({shown}) | 位置/内容比 {self.mean_injection_ratio:.3f} | 损失 {loss}"
        )

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状（含三张并排的表）."""
        return {
            "shape": self.shape.to_dict(),
            "table": self.table.to_dict(),
            "positions": list(self.positions),
            "inputs": [list(row) for row in self.inputs],
            "gathered": [list(row) for row in self.gathered],
            "injected": [list(row) for row in self.injected],
            "output": [list(row) for row in self.attention.output],
            "supervised": list(self.supervised),
            "loss": self.loss,
            "injection_ratios": list(self.injection_ratios),
            "attention": self.attention.to_dict(),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class PositionalGradients:
    """一次反向的账：**六块**（比 day075 多出 ``grad_table``）.

    ``grad_table`` 是本课唯一新增的一项，而它也是唯一**会按位置累加**的一项：
    其余五块的形状与 day075 逐字相同（前四块是参数、第五块是输入）。
    """

    grad_table: Matrix
    grad_inputs: Matrix
    grad_w_query: Matrix
    grad_w_key: Matrix
    grad_w_value: Matrix
    grad_w_output: Matrix

    def __post_init__(self) -> None:
        for name in (
            "grad_table",
            "grad_inputs",
            "grad_w_query",
            "grad_w_key",
            "grad_w_value",
            "grad_w_output",
        ):
            object.__setattr__(self, name, validate_matrix(getattr(self, name), name=name))

    def matrices(self) -> tuple[Matrix, ...]:
        """按“压平时的顺序”给出全部矩阵（与 :meth:`flatten` 一致）."""
        return (
            self.grad_w_query,
            self.grad_w_key,
            self.grad_w_value,
            self.grad_w_output,
            self.grad_table,
        )

    def flatten(self) -> Vector:
        """压平成一串数（顺序与 ``AttentionParams.flatten()`` 的四个矩阵**再加表**一致）."""
        flat: list[float] = []
        for matrix in self.matrices():
            for row in matrix:
                flat.extend(row)
        return tuple(flat)

    def max_absolute(self) -> float:
        """全部六块里最大的绝对值（“这次反向的量级有多大”）."""
        largest = 0.0
        for matrix in self.matrices():
            for row in matrix:
                for value in row:
                    largest = max(largest, abs(value))
        return largest

    def as_dict(self) -> dict[str, Matrix]:
        """按名字给出六块（键与 ``POSITIONAL_GRADIENT_TARGETS`` 前五项一致）."""
        return {
            GRAD_W_QUERY: self.grad_w_query,
            GRAD_W_KEY: self.grad_w_key,
            GRAD_W_VALUE: self.grad_w_value,
            GRAD_W_OUTPUT: self.grad_w_output,
            GRAD_TABLE: self.grad_table,
            GRAD_INPUTS: self.grad_inputs,
        }

    def summary_line(self) -> str:
        """一行说明：``dW_q 0.000184 | dW_k 0.000183 | dW_v 0.010605 | dW_o 0.000000 | dTable 0.002237 | dx 0.002237``."""
        names = (
            ("dW_q", self.grad_w_query),
            ("dW_k", self.grad_w_key),
            ("dW_v", self.grad_w_value),
            ("dW_o", self.grad_w_output),
            ("dTable", self.grad_table),
            ("dx", self.grad_inputs),
        )
        parts = []
        for label, matrix in names:
            largest = 0.0
            for row in matrix:
                for value in row:
                    largest = max(largest, abs(value))
            parts.append(f"{label} {largest:.6f}")
        return " | ".join(parts)


def relative_matrix_error(approximate: Matrix, reference: Matrix) -> float:
    """两块同形矩阵的最大相对逐点误差（**转发 day075 的实现**）.

    转发而不是重写：本课与 day075 的比较必须用**同一个**口径，
    否则“两边都对吧”这类争论会从“实现错了”变成“口径不同”。
    """
    from smart_research_agent.transformer_core.types import (
        relative_matrix_error as core_relative_matrix_error,
    )

    return core_relative_matrix_error(approximate, reference)


def check_rows_are_aligned(left: Matrix, right: Matrix, *, tolerance: float = DEFAULT_TOLERANCE) -> bool:
    """两块矩阵是否逐位相等（在给定容差内）——报告里“形状契约”那一条用它."""
    if matrix_shape(left) != matrix_shape(right):
        return False
    return all(
        abs(a - b) <= tolerance * max(1.0, abs(a), abs(b))
        for left_row, right_row in zip(left, right, strict=True)
        for a, b in zip(left_row, right_row, strict=True)
    )


def flatten_parameters(
    params: AttentionParams,
    table: EncodingTable,
) -> tuple[Vector, tuple[tuple[int, int], ...]]:
    """把“四个投影 + 位置表”压成一串数（**数值梯度与优化器都用这一条口径**）.

    顺序是 ``(w_query, w_key, w_value, w_output, table)``。它与 day075 的
    ``AttentionParams.flatten()`` 只差最后一截：表被接在后面。
    为什么把表接在**末尾**而不是插在中间——这样 day075 的四个块在前缀上位置不变，
    “多出来的那一项”在 diff 里永远出现在最后一行。
    """
    from smart_research_agent.math_foundations.optim import flatten_matrices

    matrices = params.matrices() + (table.table,)
    return flatten_matrices(matrices)


def unflatten_parameters(
    flat: Vector,
    shapes: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    *,
    template: EncodingTable,
) -> tuple[AttentionParams, EncodingTable]:
    """把一串数还原成“四个投影 + 位置表”（形状对不上时当场报错）."""
    from smart_research_agent.math_foundations.optim import unflatten_matrices

    matrices = unflatten_matrices(flat, shapes)
    if len(matrices) != 5:
        raise ShapeError(
            f"参数块个数 {len(matrices)} 与预期的 5 个（四个投影 + 一张表）不一致。"
        )
    params = AttentionParams(
        w_query=matrices[0],
        w_key=matrices[1],
        w_value=matrices[2],
        w_output=matrices[3],
    )
    return params, template.with_table(matrices[4])


def validate_settings(tolerance: float, *, name: str = "tolerance") -> float:
    """校验容差是正的有限数（报告的每条判据都走它）."""
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ParameterError(f"{name} 必须是正的有限数，收到 {tolerance!r}。")
    return float(tolerance)


__all__ = [
    "ENCODING_DESCRIPTIONS",
    "ENCODING_KINDS",
    "ENCODING_LEARNABLE",
    "ENCODING_SINUSOIDAL",
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
    "OFFSET_TOLERANCE",
    "PAIRING_ALIGNED",
    "PAIRING_DESCRIPTIONS",
    "PAIRING_STAGGERED",
    "PAIRINGS",
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
    "ROW_NORM_TOLERANCE",
    "STAGE_ADD",
    "STAGE_ATTEND",
    "STAGE_LOSS",
    "STAGE_POSITIONS",
    "STAGE_PROJECT",
    "STAGE_TABLE",
    "EncodingTable",
    "PositionalForward",
    "PositionalGradients",
    "PositionalShape",
    "add_matrices",
    "check_rows_are_aligned",
    "default_positions",
    "flatten_parameters",
    "gather_rows",
    "matrix_row_norms",
    "relative_matrix_error",
    "scatter_add_rows",
    "unflatten_parameters",
    "validate_positions",
    "validate_settings",
]
