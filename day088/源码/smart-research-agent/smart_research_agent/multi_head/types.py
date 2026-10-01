"""``multi_head`` 的形状、口径表与三条记录（day076 / M7-D2）.

day075 交出了一层**可训练**的自注意力：``Q = x·W_qᵀ`` 等四个投影，
七个阶段、七步反向、五项梯度校验，全流程零依赖、可手算。
那一层里每一行只有**一份**混合系数——softmax 给出**一个**分布。

今天要动的是这句话里的一个字：把"一个分布"变成"``heads`` 个分布"。

```text
day075   context = softmax(QKᵀ/√d_k) · V              每行**一个**分布
day076   context = ‖_h softmax(Q_h K_hᵀ/√d_h) · V_h   每行 **heads** 个分布，再拼接
```

## 一、九个阶段：多出来的两步正是这一课的全部内容

```text
project → split → score → scale → mask → softmax → mix → merge → output
           ^^^^^                                              ^^^^^
           把 d_k 与 d_v 各切成 heads 段                 把 heads 段拼回去
```

``split`` 与 ``merge`` 是这一课唯一的新算术，而它们**只是一次切片与一次拼接**——
真正的难点不在这里，而在下面三件事：

```text
① 缩放系数        每头除的是 √(d_k/heads)，**不是** √d_k；
                  用错时打分整体偏小 √heads 倍（heads = 4 时是 2 倍）→ 注意力过软
② 参数是共享的    四个投影矩阵**被所有头共用**，因此反向时 heads 条链必须相加；
                  漏掉一条不报错，只会让那一头的参数更新偏小
③ 划分是记账      头与维度的对应关系是一种**约定**（连续块），
                  约定本身不影响"函数是什么"——但错了不会报错，只会静默地
                  让每一头看到"若干个维度的混合"
```

## 二、这一课的关键数字：``scale_ratio = √heads``

单头时打分除 ``√d_k``；多头时每一头只看 ``d_h = d_k/heads`` 个维度，
因此除的是 ``√d_h``。两者的比就是

```text
scale_ratio = (1/√d_h) / (1/√d_k) = √(d_k/d_h) = √heads
```

它不是一句口号，而是本模块的一个**派生属性**（``MultiHeadShape.scale_ratio``）：
实现里写的是 `self.scale / self.attention.scale`，
而测试断言它必须等于 `math.sqrt(heads)`——**代数结论被当成断言钉住**。

## 三、三个维度再加一个：``heads``

day075 的 ``AttentionShape`` 有四个维度（inputs / keys / values / outputs），
今天在它外面包一层：

```text
MultiHeadShape(attention=AttentionShape(...), heads=2)
    d_in = 6   d_k = 6   d_v = 6   d_out = 6
    heads = 2  →  head_dim = d_k/heads = 3   head_value = d_v/heads = 3
```

为什么是"包一层"而不是把 ``heads`` 塞进 ``AttentionShape``：
``AttentionShape`` 描述的是"这一层的输入输出有多宽"，它对头数**一无所知**；
头数是"这一层内部怎么组织这些维度"的一个约定。
把两件事塞进一个 dataclass，会让"换头数"变成一次**形状变更**——
而换头数其实不改变任何一个张量的对外形状（输入输出宽度完全一样）。

**这条区分有一处立刻可验证的后果**：``AttentionParams`` 一个字节都不用改。
``W_q`` 的形状仍然是 ``(d_k, d_in)``、``W_v`` 仍然是 ``(d_v, d_in)``、
``W_o`` 仍然是 ``(d_out, d_v)``——多头**不新增任何参数**。
换句话说：

> **多头不是"更多参数买更多能力"，而是"同一批参数买到 heads 份注意力"。**

## 四、口径表：三张表，逐键对齐

```text
MULTIHEAD_STAGES           九个阶段（七个 + split + merge）
MULTIHEAD_GRADIENT_FORMULAS 五项梯度（**与 day075 同名**，公式多一个 Σ_h）
MULTIHEAD_PROPERTIES        六条性质（day075 的四条 + 两条本层专有）
```

第二张表值得单独说一句：五项的名字**刻意与 day075 一模一样**
（``w_query`` / ``w_key`` / ``w_value`` / ``w_output`` / ``inputs``），
因为它们是"同一个参数矩阵的梯度"；变的只是公式——

```text
day075  dW_q = dQᵀ · x
day076  dW_q = Σ_h dQ_hᵀ · x          ← 因为 W_q 被所有头共用
```

**同名而不同式**是这一课最容易出错的地方：报告里两天的 ``w_query`` 长得一样，
只有公式那一列能告诉你"这一份是多头的那一份"。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.math_foundations.types import (
    Matrix,
    Vector,
    matrix_shape,
    validate_matrix,
)
from smart_research_agent.multi_head.errors import (
    MultiHeadError,
    ParameterError,
    PartitionError,
    ShapeError,
)
from smart_research_agent.transformer_core.types import (
    GRADIENT_TARGETS,
    AttentionParams,
    AttentionShape,
    ParameterGradients,
)

# --------------------------------------------------------------------------- #
# 闭合表 1：九个阶段
# --------------------------------------------------------------------------- #

STAGE_PROJECT = "project"
STAGE_SPLIT = "split"
STAGE_SCORE = "score"
STAGE_SCALE = "scale"
STAGE_MASK = "mask"
STAGE_SOFTMAX = "softmax"
STAGE_MIX = "mix"
STAGE_MERGE = "merge"
STAGE_OUTPUT = "output"

#: 九个阶段（顺序 = 一次前向的执行顺序，也是教程与手册里的排列）.
MULTIHEAD_STAGES: tuple[str, ...] = (
    STAGE_PROJECT,
    STAGE_SPLIT,
    STAGE_SCORE,
    STAGE_SCALE,
    STAGE_MASK,
    STAGE_SOFTMAX,
    STAGE_MIX,
    STAGE_MERGE,
    STAGE_OUTPUT,
)

#: 每个阶段的一句话解释.
MULTIHEAD_STAGE_DESCRIPTIONS: dict[str, str] = {
    STAGE_PROJECT: "投影：用四个矩阵把输入变成 Q/K/V——与 day075 完全相同的四个矩阵",
    STAGE_SPLIT: "拆分：把 Q/K/V 的行（输出维）各切成 heads 段，每段是一头自己的子空间",
    STAGE_SCORE: "打分：每一头 raw_h = Q_h·K_hᵀ，heads 张 (n, n) 各自独立",
    STAGE_SCALE: "缩放：scores_h = raw_h / √d_h（**分母是每头的宽度，不是 d_k**）",
    STAGE_MASK: "掩码：heads 张表共用同一份掩码——因果性是位置的性质，不是某一头的偏好",
    STAGE_SOFTMAX: "softmax：每一头按行归一化，于是每一行有 **heads 个**分布、每头各自和为 1",
    STAGE_MIX: "加权：context_h = weights_h · V_h，heads 个 (n, d_vh) 各自独立",
    STAGE_MERGE: "拼接：heads 个 context_h 按列拼成 (n, d_v)——**它是 split 的严格逆**",
    STAGE_OUTPUT: "输出投影：y = context · W_oᵀ，一次投影把 heads 个子空间混合回 d_out",
}

#: 每个阶段的形状变化（报告里能读到"这一步之后形状是什么"）.
MULTIHEAD_STAGE_SHAPES: dict[str, str] = {
    STAGE_PROJECT: "(n, d_in) → Q/K/V 各 (n, d_k)/(n, d_k)/(n, d_v)",
    STAGE_SPLIT: "(n, d_k) → heads × (n, d_h)；V 同理 → heads × (n, d_vh)",
    STAGE_SCORE: "每头 (n, d_h) × (n, d_h)ᵀ → (n, n)；共 heads 张",
    STAGE_SCALE: "(n, n) × heads → (n, n) × heads",
    STAGE_MASK: "(n, n) → (n, n)（同一张表发给每一头）",
    STAGE_SOFTMAX: "(n, n) × heads → (n, n) × heads，**每头每一行**和为 1",
    STAGE_MIX: "每头 (n, n) × (n, d_vh) → (n, d_vh)",
    STAGE_MERGE: "heads × (n, d_vh) → (n, d_v)",
    STAGE_OUTPUT: "(n, d_v) × (d_out, d_v)ᵀ → (n, d_out)",
}

if not (
    set(MULTIHEAD_STAGES) == set(MULTIHEAD_STAGE_DESCRIPTIONS) == set(MULTIHEAD_STAGE_SHAPES)
):
    raise MultiHeadError(
        "多头阶段的两张表不一致：MULTIHEAD_STAGES / MULTIHEAD_STAGE_DESCRIPTIONS / "
        "MULTIHEAD_STAGE_SHAPES 必须逐键对齐，否则某个阶段在报告里只有名字、"
        "没有解释也没有形状，而'缺一行'与'这一项没问题'读起来一样。"
    )

# --------------------------------------------------------------------------- #
# 闭合表 2：五项梯度校验（**与 day075 同名，公式多一个 Σ_h**）
# --------------------------------------------------------------------------- #

GRAD_W_QUERY = "w_query"
GRAD_W_KEY = "w_key"
GRAD_W_VALUE = "w_value"
GRAD_W_OUTPUT = "w_output"
GRAD_INPUTS = "inputs"

#: 五项（顺序与 day075 逐位相同：输出投影 → value → 权重 → 打分 → 输入投影）.
MULTIHEAD_GRADIENT_TARGETS: tuple[str, ...] = GRADIENT_TARGETS

#: 每一项的解析梯度公式（**每一式都带一个 Σ_h 或一次 split**）.
MULTIHEAD_GRADIENT_FORMULAS: dict[str, str] = {
    GRAD_W_OUTPUT: "dW_o = dOutᵀ · merged_context（W_o 消费的是拼接结果，**不属于任何一头**）",
    GRAD_W_VALUE: "dV = Σ_h splitᵀ(dContext)·weights_h；dW_v = Σ_h dV_hᵀ · x",
    GRAD_W_QUERY: "dScores_h ← 每头各自的 softmax 反向；dQ = Σ_h dQ_h；dW_q = Σ_h dQ_hᵀ · x",
    GRAD_W_KEY: "dK = Σ_h dK_h；dW_k = Σ_h dK_hᵀ · x",
    GRAD_INPUTS: "dx = Σ_h (dQ_h·W_q^h + dK_h·W_k^h + dV_h·W_v^h)",
}

#: 每一项的说明（它回答"这一块梯度是怎么来的"）.
MULTIHEAD_GRADIENT_DESCRIPTIONS: dict[str, str] = {
    GRAD_W_OUTPUT: "输出投影：唯一**不被分头**的一块——它把 heads 个子空间混回 d_out",
    GRAD_W_VALUE: "value 投影：heads 条链在**行维**上相加（W_v 被所有头共用）",
    GRAD_W_QUERY: "query 投影：每一头的梯度要穿过**它自己那一张**权重表的 softmax，再相加",
    GRAD_W_KEY: "key 投影：与 query 共享同一份软打分，但分头之后梯度也各走各的",
    GRAD_INPUTS: "输入：heads × 三条链的和——少一条不报错，只会让下层学得慢",
}

if not (
    set(MULTIHEAD_GRADIENT_TARGETS)
    == set(MULTIHEAD_GRADIENT_FORMULAS)
    == set(MULTIHEAD_GRADIENT_DESCRIPTIONS)
):
    raise MultiHeadError(
        "梯度校验项的两张表不一致：MULTIHEAD_GRADIENT_TARGETS / "
        "MULTIHEAD_GRADIENT_FORMULAS / MULTIHEAD_GRADIENT_DESCRIPTIONS 必须逐键对齐——"
        "少一个键的那一块梯度会静默地不被校验，而'没校验'与'校验通过'在报告里长得一样。"
    )

# --------------------------------------------------------------------------- #
# 闭合表 3：六条性质（day075 的四条 + 本层两条）
# --------------------------------------------------------------------------- #

PROPERTY_PER_HEAD_ROW_STOCHASTIC = "per_head_row_stochastic"
PROPERTY_HEAD_NON_NEGATIVE = "head_non_negative"
PROPERTY_CAUSAL_NO_LEAK_PER_HEAD = "causal_no_leak_per_head"
PROPERTY_MERGE_INVERTS_SPLIT = "merge_inverts_split"
PROPERTY_SINGLE_HEAD_MATCHES_CLASSIC = "single_head_matches_classic"
PROPERTY_HEAD_ORDER_IS_BOOKKEEPING = "head_order_is_bookkeeping"

#: 六条性质（顺序 = 从"每头的账"到"这一层的划分只是一个约定"）.
MULTIHEAD_PROPERTIES: tuple[str, ...] = (
    PROPERTY_PER_HEAD_ROW_STOCHASTIC,
    PROPERTY_HEAD_NON_NEGATIVE,
    PROPERTY_CAUSAL_NO_LEAK_PER_HEAD,
    PROPERTY_MERGE_INVERTS_SPLIT,
    PROPERTY_SINGLE_HEAD_MATCHES_CLASSIC,
    PROPERTY_HEAD_ORDER_IS_BOOKKEEPING,
)

#: 每条性质的一句话解释.
MULTIHEAD_PROPERTY_DESCRIPTIONS: dict[str, str] = {
    PROPERTY_PER_HEAD_ROW_STOCHASTIC: "**每一头的每一行**之和为 1——是 heads·n 个条件分布，不是一个",
    PROPERTY_HEAD_NON_NEGATIVE: "每个权重 >= 0；负权重会让某一头的'加权平均'变成减法",
    PROPERTY_CAUSAL_NO_LEAK_PER_HEAD: "因果模式下**每一头**的上三角恰好是 0.0——位置 i 看不到未来",
    PROPERTY_MERGE_INVERTS_SPLIT: "merge(split(M)) 逐位等于 M（往返恒等）——否则头部会静默串味",
    PROPERTY_SINGLE_HEAD_MATCHES_CLASSIC: "heads=1 时输出与 day075 的自注意力"
    "**逐位相同**（含反向）",
    PROPERTY_HEAD_ORDER_IS_BOOKKEEPING: "一致地重排头块之后输出不变——头编号是一个记账约定",
}

if set(MULTIHEAD_PROPERTIES) != set(MULTIHEAD_PROPERTY_DESCRIPTIONS):
    raise MultiHeadError(
        "性质的两张表不一致：MULTIHEAD_PROPERTIES / MULTIHEAD_PROPERTY_DESCRIPTIONS "
        "必须逐键对齐。"
    )


# --------------------------------------------------------------------------- #
# 头划分：把一条轴切成 heads 段连续块
# --------------------------------------------------------------------------- #


def _checked_positive_int(value: int, *, name: str) -> int:
    """校验一个 >= 1 的整数（``bool`` 不算整数——它是 ``int`` 的子类）."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ParameterError(f"{name} 必须是整数，收到 {value!r}。")
    if value < 1:
        raise ParameterError(
            f"{name} 必须 >= 1，收到 {value}："
            "零宽的子空间上既没有点积、也没有分布、也没有加权平均。"
        )
    return value


@dataclass(frozen=True)
class HeadPartition:
    """把一条长度为 ``total`` 的轴切成 ``heads`` 段**连续**的块.

    ```text
    total = 6, heads = 3   →  bounds = ((0, 2), (2, 4), (4, 6))   每头宽度 2
    ```

    ## 为什么必须是"连续块"

    切法有很多种（交错切、按模取、任意集合），本包选**连续块**，理由是三条：

    ```text
    ① 与生产实现一致   PyTorch 的 reshape(..., heads, head_dim).transpose 就是连续块
    ② 可打印           一段连续块只需要两个数（start, stop），
                       而"任意集合"要打印一整个列表——报告里读不出"第 3 头是谁"
    ③ 可逆             连续块的拼回是唯一的；交错切也能拼回，但"拼回对不对"
                       要读完两份实现才知道
    ```

    第 ③ 条是这一课的一条硬约束：``merge_rows(split_rows(M))`` 必须**逐位等于** ``M``。
    如果切法写成"第 h 头取下标 ≡ h (mod heads)"，那么一条"切对了但拼回去顺序不同"的
    实现会让每一头看到**混合了所有头**的维度——它不报错，只是每一头的语义都变了。

    ## 一个不得不说的限制

    ``total`` 必须能被 ``heads`` 整除，否则抛 :class:`PartitionError`——
    **本包不会替你挑一个"最接近的合法头数"**。自动修正是一种"看起来友好"的静默失败：
    调用方设了 4 个头，报告里的头数是 3，而报告看起来完全正常。
    """

    total: int
    heads: int

    def __post_init__(self) -> None:
        _checked_positive_int(self.total, name="total")
        _checked_positive_int(self.heads, name="heads")
        if self.total % self.heads != 0:
            raise PartitionError(
                f"宽度 {self.total} 不能被头数 {self.heads} 整除："
                "每头的宽度必须是一个整数，而本包不会替你挑一个'最接近的合法头数'——"
                "自动修正会让'我设了几个头'这件事在报告里消失。"
            )

    @property
    def width(self) -> int:
        """每一头的宽度（``total / heads``）."""
        return self.total // self.heads

    @property
    def bounds(self) -> tuple[tuple[int, int], ...]:
        """每一头的 ``(start, stop)``（**可以打印出来看的那张表**）."""
        return tuple(
            (head * self.width, (head + 1) * self.width) for head in range(self.heads)
        )

    def rows_of(self, head: int) -> tuple[int, ...]:
        """第 ``head`` 头占用的下标（用于"按行取子矩阵"）."""
        start, stop = self.bounds[self._checked_head(head)]
        return tuple(range(start, stop))

    def _checked_head(self, head: int) -> int:
        """校验头号落在 ``[0, heads)`` 内."""
        if isinstance(head, bool) or not isinstance(head, int):
            raise ParameterError(f"头号必须是整数，收到 {head!r}。")
        if not 0 <= head < self.heads:
            raise ParameterError(f"头号 {head} 落在 [0, {self.heads}) 之外。")
        return head

    def split_rows(self, matrix: Matrix) -> tuple[Matrix, ...]:
        """沿**行**切（``W_q`` / ``W_k`` / ``W_v`` 的输出维就是行）."""
        checked = validate_matrix(matrix, name="matrix")
        rows, columns = matrix_shape(checked)
        if rows != self.total:
            raise ShapeError(
                f"矩阵行数 {rows} 与划分的宽度 {self.total} 不一致："
                "行对应的是'这一层输出多少维'，划分必须作用在它上面。"
            )
        return tuple(
            tuple(checked[row] for row in range(start, stop))
            for start, stop in self.bounds
        )

    def merge_rows(self, parts: tuple[Matrix, ...] | list[Matrix]) -> Matrix:
        """把按行切开的若干块拼回一张矩阵（``split_rows`` 的严格逆）."""
        checked = self._checked_parts(parts)
        columns = matrix_shape(checked[0])[1]
        if matrix_shape(checked[0])[0] != self.width:
            raise ShapeError(
                f"第 0 块的宽度 {matrix_shape(checked[0])[0]} 与每头宽度 {self.width} 不一致。"
            )
        for index, part in enumerate(checked):
            if matrix_shape(part) != (self.width, columns):
                raise ShapeError(
                    f"第 {index} 块形状 {matrix_shape(part)} 与 "
                    f"({self.width}, {columns}) 不一致。"
                )
        return tuple(row for part in checked for row in part)

    def split_columns(self, matrix: Matrix) -> tuple[Matrix, ...]:
        """沿**列**切（``W_o`` 的输入维就是列——它消费拼接结果）."""
        checked = validate_matrix(matrix, name="matrix")
        rows, columns = matrix_shape(checked)
        if columns != self.total:
            raise ShapeError(
                f"矩阵列数 {columns} 与划分的宽度 {self.total} 不一致："
                "``W_o`` 的列对应的是它消费的 context 宽度。"
            )
        return tuple(
            tuple(row[start:stop] for row in checked) for start, stop in self.bounds
        )

    def merge_columns(self, parts: tuple[Matrix, ...] | list[Matrix]) -> Matrix:
        """把按列切开的若干块拼回一张矩阵（``split_columns`` 的严格逆）."""
        checked = self._checked_parts(parts)
        rows = matrix_shape(checked[0])[0]
        for index, part in enumerate(checked):
            if matrix_shape(part) != (rows, self.width):
                raise ShapeError(
                    f"第 {index} 块形状 {matrix_shape(part)} 与 "
                    f"({rows}, {self.width}) 不一致。"
                )
        return tuple(
            tuple(value for part in checked for value in part[row]) for row in range(rows)
        )

    def split_vector(self, vector: Vector) -> tuple[Vector, ...]:
        """把一条向量切成若干段（报告里"第几头是哪些维度"的读数用它）."""
        values = tuple(float(value) for value in vector)
        if len(values) != self.total:
            raise ShapeError(
                f"向量长度 {len(values)} 与划分的宽度 {self.total} 不一致。"
            )
        return tuple(values[start:stop] for start, stop in self.bounds)

    def merge_vector(self, parts: tuple[Vector, ...] | list[Vector]) -> Vector:
        """把若干段拼回一条向量（``split_vector`` 的严格逆）."""
        checked = tuple(tuple(float(value) for value in part) for part in parts)
        if len(checked) != self.heads:
            raise ShapeError(
                f"给出 {len(checked)} 段，而头数是 {self.heads}："
                "段数与头数必须一一对应，否则某一头会被装到另一头的位置上。"
            )
        for index, part in enumerate(checked):
            if len(part) != self.width:
                raise ShapeError(
                    f"第 {index} 段长度 {len(part)} 与每头宽度 {self.width} 不一致。"
                )
        return tuple(value for part in checked for value in part)

    def _checked_parts(self, parts: tuple[Matrix, ...] | list[Matrix]) -> tuple[Matrix, ...]:
        """校验块的个数与头数一致（**先数个数，再看形状**）."""
        if len(parts) != self.heads:
            raise ShapeError(
                f"给出 {len(parts)} 块，而头数是 {self.heads}："
                "块数与头数必须一一对应，否则某一头会被静默地装到另一头的位置上。"
            )
        return tuple(parts)

    def describe(self) -> str:
        """一行说明：``d_k=6 → 3 头 × 2 维 | 边界 ((0,2),(2,4),(4,6))``."""
        bounds = "、".join(f"({start},{stop})" for start, stop in self.bounds)
        return f"{self.total} 维 → {self.heads} 头 × {self.width} 维 | 边界 {bounds}"


# --------------------------------------------------------------------------- #
# 形状
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MultiHeadShape:
    """一次多头注意力：``AttentionShape`` 之外再加一个头数.

    ```text
    attention.inputs     d_in       输入行宽（四个投影的公共输入维）
    attention.keys       d_k        打分维度（= heads × head_dim）
    attention.values     d_v        value 维度（= heads × head_value）
    attention.outputs    d_out      最终输出维（= W_o 的行数）
    heads                heads      头数（**必须整除 d_k 与 d_v**）
    ```

    派生量与它们各自的来源：

    ```text
    head_dim       d_k / heads      每一头打分用几个维度 → 决定缩放系数 √d_h
    head_value     d_v / heads      每一头混合出几个维度 → 决定 context 分块宽度
    scale          1/√head_dim      多头**真正**用的缩放系数
    single_head_scale 1/√d_k        day075 用的那个（**只在 heads=1 时相等**）
    scale_ratio    scale/单头       等于 √heads（派生出来，因此"代数结论"变成一条断言）
    ```
    """

    attention: AttentionShape
    heads: int

    def __post_init__(self) -> None:
        if not isinstance(self.attention, AttentionShape):
            raise ShapeError(
                f"attention 必须是 AttentionShape，收到 {type(self.attention).__name__}。"
            )
        _checked_positive_int(self.heads, name="heads")
        for name, width in (
            ("d_k（attention.keys）", self.attention.keys),
            ("d_v（attention.values）", self.attention.values),
        ):
            if width % self.heads != 0:
                raise PartitionError(
                    f"{name} = {width} 不能被头数 {self.heads} 整除："
                    "每头的宽度必须是一个整数，而本包不会替你挑一个'最接近的合法头数'。"
                )

    @property
    def keys_partition(self) -> HeadPartition:
        """``d_k`` 的划分（``W_q`` / ``W_k`` 按它切行）."""
        return HeadPartition(self.attention.keys, self.heads)

    @property
    def values_partition(self) -> HeadPartition:
        """``d_v`` 的划分（``W_v`` 按它切行、``W_o`` 按它切列）."""
        return HeadPartition(self.attention.values, self.heads)

    @property
    def head_dim(self) -> int:
        """每一头的打分维度 ``d_k / heads``."""
        return self.keys_partition.width

    @property
    def head_value(self) -> int:
        """每一头的 value 维度 ``d_v / heads``."""
        return self.values_partition.width

    @property
    def scale(self) -> float:
        """多头**真正**用的缩放系数 ``1/√head_dim``."""
        return 1.0 / math.sqrt(self.head_dim)

    @property
    def single_head_scale(self) -> float:
        """day075 用的那个 ``1/√d_k``（**只在 heads=1 时与 ``scale`` 相等**）."""
        return self.attention.scale

    @property
    def scale_ratio(self) -> float:
        """``scale / 单头缩放``——**派生量**，测试断言它等于 ``√heads``."""
        return self.scale / self.single_head_scale

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "heads": self.heads,
            "head_dim": self.head_dim,
            "head_value": self.head_value,
            "scale": self.scale,
            "single_head_scale": self.single_head_scale,
            "scale_ratio": self.scale_ratio,
            "attention": self.attention.to_dict(),
        }

    def summary_line(self) -> str:
        """一行说明：``d_in=6 d_k=6 d_v=6 d_out=6 | 2 头 × (3, 3) | scale=0.577350(√2×)``."""
        return (
            f"d_in={self.attention.inputs} d_k={self.attention.keys} "
            f"d_v={self.attention.values} d_out={self.attention.outputs} | "
            f"{self.heads} 头 × ({self.head_dim}, {self.head_value}) | "
            f"scale={self.scale:.6f}（单头 {self.single_head_scale:.6f} 的 "
            f"{self.scale_ratio:.4f} 倍）"
        )


# --------------------------------------------------------------------------- #
# 前向记录
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MultiHeadForward:
    """一次多头前向的全部记录（**反向传播要用到的每一个中间量**）.

    ```text
    inputs          (n, d_in)              输入（反向算 dW = dQᵀ·x 要用）
    queries/keys    (n, d_k)               融合投影（**未分头**，反向算 dW_q 要用）
    values          (n, d_v)               同上
    head_queries    heads × (n, d_h)       分头之后的 Q（每一头自己的子空间）
    head_keys       heads × (n, d_h)
    head_values     heads × (n, d_vh)
    head_raw        heads × (n, n)         Q_h·K_hᵀ（**未缩放**）
    head_scores     heads × (n, n)         raw_h · scale（softmax 的直接输入）
    head_weights    heads × (n, n)         每一头按行 softmax（被掩码的位置恰好 0.0）
    head_contexts   heads × (n, d_vh)      weights_h · V_h
    merged_context  (n, d_v)               heads 段按列拼接（**W_o 的输入**）
    output          (n, d_out)             merged_context · W_oᵀ
    ```

    ## 为什么"融合的 Q/K/V"与"分头的 Q/K/V"都要留

    两者**只差一次切片**，看起来留一份就够。但反向里它们各自负责一件事：

    ```text
    融合投影 queries    算 dW_q 要用它（dW_q = dQᵀ · x 里的 x，以及"这一头对应哪几行"）
    分头投影 head_keys  算 dQ_h = dRaw_h · K_h 要用它
    ```

    只留分头版本的话，反向必须**先把 heads 张 dRaw 拼成一张 (n, d_k) 的大矩阵**——
    而"拼接顺序错了"这种事在数值上表现为"训练变慢"，不会报错。
    留一份融合投影（它就是"拼好的那一份"）把这件事变成了**一次断言**：
    ``merge_rows(head_keys) == keys``。
    """

    params: AttentionParams
    shape: MultiHeadShape
    inputs: Matrix
    queries: Matrix
    keys: Matrix
    values: Matrix
    head_queries: tuple[Matrix, ...]
    head_keys: tuple[Matrix, ...]
    head_values: tuple[Matrix, ...]
    head_raw: tuple[Matrix, ...]
    head_scores: tuple[Matrix, ...]
    head_weights: tuple[Matrix, ...]
    head_contexts: tuple[Matrix, ...]
    merged_context: Matrix
    output: Matrix
    causal: bool
    mask: tuple[tuple[bool, ...], ...]
    scale: float
    head_entropies: tuple[Vector, ...]
    head_peaks: tuple[Vector, ...]
    head_indices: tuple[tuple[int, ...], ...]
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        shape = self.shape
        if not isinstance(self.params, AttentionParams):
            raise ShapeError(
                f"params 必须是 AttentionParams，收到 {type(self.params).__name__}。"
            )
        if self.params.shape != shape.attention:
            raise ShapeError(
                f"参数的形状 {self.params.shape.summary_line()} 与本层声明的形状 "
                f"{shape.attention.summary_line()} 不一致："
                "头数是一个组织约定，它必须与四个矩阵的实际形状相容。"
            )
        groups = (
            ("head_queries", self.head_queries),
            ("head_keys", self.head_keys),
            ("head_values", self.head_values),
            ("head_raw", self.head_raw),
            ("head_scores", self.head_scores),
            ("head_weights", self.head_weights),
            ("head_contexts", self.head_contexts),
            ("head_entropies", self.head_entropies),
            ("head_peaks", self.head_peaks),
            ("head_indices", self.head_indices),
        )
        for name, group in groups:
            if len(group) != shape.heads:
                raise ShapeError(
                    f"{name} 有 {len(group)} 项，而头数是 {shape.heads}："
                    "少一项的那一头在报告里会静默消失，而'少了第二头'与"
                    "'第二头没问题'在读数上长得一样。"
                )
        rows, columns = matrix_shape(self.head_weights[0])
        if rows != columns:
            raise ShapeError(
                f"每一头的权重矩阵必须是方阵（自己看自己），收到 {(rows, columns)}。"
            )
        expected = {
            "head_queries": (rows, shape.head_dim),
            "head_keys": (rows, shape.head_dim),
            "head_values": (rows, shape.head_value),
            "head_raw": (rows, rows),
            "head_scores": (rows, rows),
            "head_weights": (rows, rows),
            "head_contexts": (rows, shape.head_value),
        }
        for head in range(shape.heads):
            for name, wanted in expected.items():
                matrix = getattr(self, name)[head]
                if matrix_shape(matrix) != wanted:
                    raise ShapeError(
                        f"{name}[{head}] 的形状 {matrix_shape(matrix)} 与 {wanted} 不一致。"
                    )
            for name, vector in (
                ("head_entropies", self.head_entropies[head]),
                ("head_peaks", self.head_peaks[head]),
                ("head_indices", self.head_indices[head]),
            ):
                if len(vector) != rows:
                    raise ShapeError(
                        f"{name}[{head}] 有 {len(vector)} 项，而行数是 {rows}。"
                    )
        if matrix_shape(self.merged_context) != (rows, shape.attention.values):
            raise ShapeError(
                f"merged_context 的形状 {matrix_shape(self.merged_context)} 与 "
                f"({rows}, {shape.attention.values}) 不一致。"
            )
        if matrix_shape(self.output)[0] != rows:
            raise ShapeError("output 的行数必须与权重行数一致（一行输入一行输出）。")
        if len(self.mask) != rows or (rows and len(self.mask[0]) != rows):
            raise ShapeError(f"掩码形状与每头权重 {(rows, rows)} 不一致。")
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    # ----------------------------------------------------------------- #
    # 派生读数
    # ----------------------------------------------------------------- #

    @property
    def tokens(self) -> int:
        """行数（= 序列长度）."""
        return matrix_shape(self.head_weights[0])[0]

    @property
    def heads(self) -> int:
        """头数（从 ``head_weights`` 的项数读出，**不重复声明**）."""
        return len(self.head_weights)

    @property
    def distributions(self) -> int:
        """这一层一共产生了多少个条件分布：``heads × n``.

        它是这一课最容易说错的一个数：单头时是 ``n`` 个，多头时是 ``heads × n`` 个。
        "每一行是一个分布"这句话在多头下**不再成立**——
        每一行是 ``heads`` 个分布，各自独立、各自和为 1。
        """
        return self.heads * self.tokens

    def head_weight_row(self, head: int, row: int) -> Vector:
        """第 ``head`` 头第 ``row`` 行的那一份分布."""
        self.shape.keys_partition._checked_head(head)
        if not 0 <= row < self.tokens:
            raise ParameterError(f"行号 {row} 落在 [0, {self.tokens}) 之外。")
        return self.head_weights[head][row]

    def weights_tensor(self) -> tuple[Matrix, ...]:
        """``heads`` 张 ``(n, n)`` 的表（"同一行的 heads 份分布"最直观的读法）."""
        return tuple(self.head_weights)

    @property
    def mean_entropy(self) -> float:
        """所有 (头, 行) 的平均熵（nats）：0 = 各自全押一处，ln(n) = 完全均匀."""
        values = [value for head in self.head_entropies for value in head]
        if not values:
            return 0.0
        return math.fsum(values) / len(values)

    def max_entropy(self) -> float:
        """理论上界 ``ln(n)``."""
        return math.log(self.tokens) if self.tokens > 0 else 0.0

    def focus_ratio(self) -> float:
        """平均集中度 ``1 − 平均熵/ln(n)``."""
        ceiling = self.max_entropy()
        if ceiling <= 0:
            return 0.0
        return max(0.0, 1.0 - self.mean_entropy / ceiling)

    @property
    def mean_peak_weight(self) -> float:
        """所有 (头, 行) 的平均峰值权重（"注意力有多尖"的总读数）."""
        values = [value for head in self.head_peaks for value in head]
        if not values:
            return 0.0
        return math.fsum(values) / len(values)

    def peak_weights_of_head(self, head: int) -> Vector:
        """某一头各行的峰值权重."""
        self.shape.keys_partition._checked_head(head)
        return self.head_peaks[head]

    def head_summary_line(self, head: int) -> str:
        """某一头的一行读数（报告里"这一头在干什么"用它）."""
        partition = self.shape.keys_partition
        partition._checked_head(head)
        peaks = self.head_peaks[head]
        entropies = self.head_entropies[head]
        indices = self.head_indices[head]
        return (
            f"head {head} | 维度 {partition.bounds[head]} | "
            f"平均峰值 {math.fsum(peaks) / len(peaks):.4f} | "
            f"平均熵 {math.fsum(entropies) / len(entropies):.4f} | "
            f"各行 argmax {indices}"
        )

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "causal": self.causal,
            "tokens": self.tokens,
            "heads": self.heads,
            "distributions": self.distributions,
            "scale": self.scale,
            "shape": self.shape.to_dict(),
            "head_weights": [[list(row) for row in head] for head in self.head_weights],
            "head_peaks": [list(head) for head in self.head_peaks],
            "head_entropies": [list(head) for head in self.head_entropies],
            "head_indices": [list(head) for head in self.head_indices],
            "merged_context": [list(row) for row in self.merged_context],
            "output": [list(row) for row in self.output],
            "mean_entropy": self.mean_entropy,
            "mean_peak_weight": self.mean_peak_weight,
            "focus_ratio": self.focus_ratio(),
            "notes": list(self.notes),
        }

    def summary_line(self) -> str:
        """一行说明：``causal | n=5 | 2 头 × (3, 3) → d_out=6 | 平均熵 …``."""
        return (
            f"{'causal' if self.causal else 'full'} | n={self.tokens} | "
            f"{self.heads} 头 → d_out={matrix_shape(self.output)[1]} | "
            f"平均熵 {self.mean_entropy:.4f} | 集中度 {self.focus_ratio():.1%}"
        )


@dataclass(frozen=True)
class HeadGradients:
    """**单一一头**对参数的梯度贡献（四块里少一块：没有 ``grad_w_output``）.

    ```text
    grad_w_query   (head_dim, d_in)      W_q 的第 h 段行
    grad_w_key     (head_dim, d_in)      W_k 的第 h 段行
    grad_w_value   (head_value, d_in)    W_v 的第 h 段行
    grad_inputs    (n, d_in)             这一头传回输入的梯度（三层链之和）
    ```

    ## 为什么没有 ``grad_w_output``

    这是这一课最重要的一处**不对称**：

    ```text
    W_q / W_k / W_v   按行分头   → 第 h 头负责其中一段行，梯度各管各的
    W_o               **不分头**  → 它消费的是拼接结果，是所有头的公共出口
    ```

    换句话说，``W_o`` 的梯度里**无法区分**"贡献来自哪一头"——
    它看到的只有那张 ``(n, d_v)`` 的拼接结果。[``头梯度范数``][head_gradient_norms]
    这个读数因此只统计前三块，而报告里必须把这件事写出来：
    **一个"某头不学"的结论绝不能只凭 W_o 的梯度下**。

    一个直接可验证的推论：把 heads 份 :class:`HeadGradients` 按行拼起来，
    必须**逐位等于** :func:`layers.multi_head_backward` 给出的融合梯度
    （每一行只由一头贡献，因此不存在求和顺序的歧义）。
    """

    head: int
    grad_w_query: Matrix
    grad_w_key: Matrix
    grad_w_value: Matrix
    grad_inputs: Matrix

    def __post_init__(self) -> None:
        _checked_positive_int(self.head + 1, name="head + 1")
        for name in ("grad_w_query", "grad_w_key", "grad_w_value", "grad_inputs"):
            object.__setattr__(self, name, validate_matrix(getattr(self, name), name=name))

    def parameter_matrices(self) -> tuple[Matrix, ...]:
        """三块**按行**的参数梯度（顺序固定：q / k / v）."""
        return (self.grad_w_query, self.grad_w_key, self.grad_w_value)

    def norm(self) -> float:
        """三块参数梯度的 Frobenius 范数之和（就是这个读数让"哪头在学"可比较）."""
        total = 0.0
        for matrix in self.parameter_matrices():
            for row in matrix:
                for value in row:
                    total += value * value
        return math.sqrt(total)

    def summary_line(self) -> str:
        """一行说明：``head 0 | ‖dW_q‖=0.1234 |dW_k|=… | 输入梯度 max …``."""
        parts = []
        for name, matrix in zip(("dW_q", "dW_k", "dW_v"), self.parameter_matrices()):
            worst = max((abs(value) for row in matrix for value in row), default=0.0)
            parts.append(f"|{name}|={worst:.6f}")
        input_worst = max(
            (abs(value) for row in self.grad_inputs for value in row), default=0.0
        )
        return (
            f"head {self.head} | " + " ".join(parts)
            + f" | 输入梯度 max {input_worst:.6f} | 范数 {self.norm():.6f}"
        )


def head_gradient_norms(
    gradients: ParameterGradients,
    shape: MultiHeadShape,
) -> tuple[float, ...]:
    """每一头在 q/k/v 三块梯度上的 **Frobenius 范数之和**（诊断"哪些头在学"）.

    参数块与头的对应关系由划分决定：第 ``h`` 头拥有

    ```text
    W_q 的第 h 段行      W_k 的第 h 段行      W_v 的第 h 段行
    ```

    ``W_o`` **不在任何一头里**——它消费的是拼接结果，是 heads 个子空间的公共出口。
    （这也是这一课要记住的一处不对称：三块投影按行分头，第四块不分头。）

    这个读数回答的问题是"梯度有没有被某一头独吞"：

    ```text
    各头范数接近        多头真的都在学（梯度被平分）
    某一头独大          其它头可能已经"死掉"（这一头的输出被 W_o 忽略）
    全部集中在同一头    多头退化成了单头——参数多了一倍，注意力只有一份
    ```
    """
    if not isinstance(shape, MultiHeadShape):
        raise ShapeError(f"shape 必须是 MultiHeadShape，收到 {type(shape).__name__}。")
    keys_partition = shape.keys_partition
    values_partition = shape.values_partition
    query_rows = keys_partition.split_rows(gradients.grad_w_query)
    key_rows = keys_partition.split_rows(gradients.grad_w_key)
    value_rows = values_partition.split_rows(gradients.grad_w_value)
    norms: list[float] = []
    for head in range(shape.heads):
        total = 0.0
        for matrix in (query_rows[head], key_rows[head], value_rows[head]):
            for row in matrix:
                for value in row:
                    total += value * value
        norms.append(math.sqrt(total))
    return tuple(norms)


def head_gradient_shares(
    gradients: ParameterGradients,
    shape: MultiHeadShape,
) -> tuple[float, ...]:
    """把 :func:`head_gradient_norms` 归一成占比（总和为 1，全零时均分为 0）.

    "全零时返回全 0"而不是抛异常：一片零梯度是一个**真实的观测**
    （学习率为 0、或者损失已经到极小），它应当被读成"没有头在动"，
    而不是一个需要捕获的错误。
    """
    norms = head_gradient_norms(gradients, shape)
    total = math.fsum(norms)
    if total == 0.0:
        return tuple(0.0 for _ in norms)
    return tuple(value / total for value in norms)


__all__ = [
    "GRADIENT_TARGETS",
    "GRAD_INPUTS",
    "GRAD_W_KEY",
    "GRAD_W_OUTPUT",
    "GRAD_W_QUERY",
    "GRAD_W_VALUE",
    "MULTIHEAD_GRADIENT_DESCRIPTIONS",
    "MULTIHEAD_GRADIENT_FORMULAS",
    "MULTIHEAD_GRADIENT_TARGETS",
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
    "HeadGradients",
    "HeadPartition",
    "MultiHeadForward",
    "MultiHeadShape",
    "head_gradient_norms",
    "head_gradient_shares",
]
