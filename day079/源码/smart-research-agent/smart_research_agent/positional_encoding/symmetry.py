"""``positional_encoding`` 的**对称性下界**：这一课唯一“值钱”的结论（day078 / M7-D3）.

## 问题

day075 给本课留了一句可断言的话：**无掩码时注意力是置换等变的**。等变意味着

```text
把输入的行换一个顺序，输出的行**按同样的方式**换顺序
⇒ 模型对“哪一行在第几位”这件事一无所知
```

如果任务要求“第 0 位是什么”，会发生什么？答案是：**有一个下界，而且它可以被算出来**。

## 任务：位置选择（readout）

```text
输入   一个长度为 n 的序列，每一行是它那个 token 的 one-hot（V 维）
目标   **每一行都输出“第 0 位那个 token”的 one-hot**
监督   所有行
轨道   这个序列的 n 个循环位移——每一个位移的“第 0 位”是另一个 token
```

一句话：**同一个“袋子”（同样的 n 个 token），要读出“谁是第一位”。**

## 下界（一步一步推）

设模型是置换等变的 ``g``（无掩码、无位置编码的注意力层就是这一类），
记 ``G = g(x)``、``o_k = onehot(t_k)``、``x^{(k)}`` 是第 ``k`` 个位移：

```text
① 等变性        g(x^{(k)}) = π_k · G
② 逐行损失      ‖g(x^{(k)})_i − o_k‖² 对 i 求和 = ‖G_j − o_k‖² 对 j 求和（只是换了个下标）
③ 轨道平均      平均损失 = (1/(n²V)) · Σ_j Σ_k ‖G_j − o_k‖²
④ 每一行独立最优  min_{G_j} Σ_k ‖G_j − o_k‖² 在 G_j = mean_k o_k 处取到
                 ——**那是“袋子分布”**（在轨道涉及的那 n 个 token 上各 1/n），**不是均匀分布 1/V**
⑤ 代回去        Σ_k ‖b − o_k‖² = n·(1 − 1/n) = n − 1，一共 n 行
⇒ **下界 = n(n−1) / (n²V) = (n − 1) / (n·V)**
```

本课样本 ``n = 4``、``V = 6`` 时它是 ``3/24 = 0.125``。

> **这一步错一次**（第 13 章记了全过程）：第一版把 ④ 写成了“在 ``1/V`` 的均匀分布处取到”，
> 于是闭式 ``(1 − 1/V)/V = 0.138889`` 与“见证”**自洽地一起错了**——
> 因为两边用的是同一个错误假设。把实测落在真正的轨道上之后，
> 数字变成 0.125，闭式与实测这才分开。
> 这正是“判据要能反向检验”那条纪律的反面教材：**同一个错误假设下的两条路径会一起通过。**

## 三件必须一起说清的事

**其一：这个下界是“紧”的。** 取那个常数预测器（每一行都输出袋子分布 ``b``），
它在轨道上的平均损失**恰好**等于下界——而常数预测器本身也是置换等变的。
于是“等变模型类的最优值”就是它，而不是“某个比它小但够不着的数”。
:func:`floor_witness` 把这个数算出来，与闭式对照。

它对模型的含义很直白：**没有位置信息时，“第 0 位是哪个 token”最多只能被读成
“这一袋里有哪几个 token”**——两项之间的差 ``(n−1)/(nV)`` 就是位置知识的价格。

**其二：打破对称性有两条路。** 位置编码是第一条；**因果掩码**是第二条
（``0..i`` 这件事本身就把顺序写进了权重表）。本课把两条都作为对照变体测一遍——
否则“位置编码有用”这句话无法与“掩码已经提供了顺序”区分开。

**其三：下界管的是等变模型，不是“这个实现”。** 一旦注入位置编码，
模型就**不在**那个类里了，下界对它不适用。因此“低于下界”是**位置编码起作用的
一个充分证据**（在下界有效的那一侧它做不到）——这正是本课要的那个判决。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from smart_research_agent.math_foundations.probability import sample_index, uniforms
from smart_research_agent.math_foundations.types import (
    Matrix,
    matrix_shape,
    validate_matrix,
)
from smart_research_agent.positional_encoding.errors import (
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.positional_encoding.layers import positional_forward
from smart_research_agent.positional_encoding.types import EncodingTable
from smart_research_agent.transformer_core.layers import row_argmax_hits
from smart_research_agent.transformer_core.types import AttentionParams

#: 轨道上“平均损失必须 >= 下界”的判据（把浮点求和顺序的差压在外面）.
FLOOR_TOLERANCE = 1e-12

#: 变体的三个固定名字（报告里逐行对照用）.
VARIANT_EQUIVARIANT = "no_mask_no_positions"
VARIANT_SINUSOIDAL = "no_mask_sinusoidal"
VARIANT_CAUSAL = "causal_no_positions"

VARIANT_DESCRIPTIONS: dict[str, str] = {
    VARIANT_EQUIVARIANT: (
        "无掩码 + 不加位置编码：**置换等变**，因此受下界约束——"
        "它应该停在 (n−1)/(n·V) 附近（本课样本 n=4、V=6 时是 0.125）"
    ),
    VARIANT_SINUSOIDAL: (
        "无掩码 + 正弦位置编码：打破了置换等变性，**不受下界约束**——它应该跑到下界之下"
    ),
    VARIANT_CAUSAL: (
        "因果掩码 + 不加位置编码：掩码自己就带了顺序，同样不受下界约束——"
        "它是“位置编码不是唯一出路”的对照"
    ),
}


def make_readout_task(
    *,
    seed: int = 42,
    vocabulary: int = 6,
    length: int = 4,
) -> ReadoutTask:
    """造一个“位置选择”任务：``length`` 个**互不相同**的 token，要读出第 0 位.

    两条约束都有理由：

    ```text
    length <= vocabulary   token 必须互不相同——否则“第 0 位是哪一个”这件事
                           在多行之间会有重名，等变性那一步的推导就不干净
    length >= 2            长度为 1 时“位移轨道”只有一个元素，下界退化成 0
    ```
    """
    if isinstance(vocabulary, bool) or not isinstance(vocabulary, int):
        raise ParameterError(f"vocabulary 必须是整数，收到 {vocabulary!r}。")
    if vocabulary < 2:
        raise ParameterError(f"vocabulary 必须 >= 2，收到 {vocabulary}。")
    if isinstance(length, bool) or not isinstance(length, int):
        raise ParameterError(f"length 必须是整数，收到 {length!r}。")
    if length < 2:
        raise ParameterError(
            f"length 必须 >= 2，收到 {length}：长度为 1 时轨道只有一个元素，"
            "下界会退化成 0（那一条就没有内容了）。"
        )
    if length > vocabulary:
        raise ParameterError(
            f"length 必须 <= vocabulary（{vocabulary}），收到 {length}："
            "token 要互不相同，否则'第 0 位是哪个 token'这件事会有重名。"
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ParameterError(f"seed 必须是整数，收到 {seed!r}。")
    # 用确定性随机数在同一批下标里**不放回**地抽 length 个 token。
    picked: list[int] = []
    cursor = 0
    raw = uniforms(vocabulary * length + 8, seed=seed)
    while len(picked) < length:
        if cursor >= len(raw):  # pragma: no cover - 长度足够时不会走到
            raw = raw + uniforms(vocabulary, seed=seed + cursor)
        candidate = min(int(raw[cursor] * vocabulary), vocabulary - 1)
        if candidate not in picked:
            picked.append(candidate)
        cursor += 1
    return ReadoutTask(
        vocabulary=vocabulary,
        length=length,
        tokens=tuple(picked),
        seed=seed,
    )


@dataclass(frozen=True)
class ReadoutTask:
    """“位置选择”任务的一条基序列（+ 它生成的整个位移轨道）.

    ```text
    inputs       (n, V) 的 one-hot 矩阵——第 i 行是 tokens[i] 的 one-hot
    target       (n, V) 的矩阵，**每一行都是 tokens[0] 的 one-hot**
    supervised   所有行
    ```

    为什么“每一行都要输出第 0 位”而不是“只在第 0 行输出”：
    目标是“读位置”的能力，而不是“某一行的输出”；行与行使用**同一份**
    ``W_o`` 与同一张位置表，因此“每一行都要读第 0 位”把
    **位置表按位置共享**这件事也压进了同一个损失里。
    """

    vocabulary: int
    length: int
    tokens: tuple[int, ...]
    seed: int = 42

    def __post_init__(self) -> None:
        if len(self.tokens) != self.length:
            raise ShapeError(
                f"token 个数 {len(self.tokens)} 与声明的长度 {self.length} 不一致。"
            )
        if len(set(self.tokens)) != len(self.tokens):
            raise ShapeError(
                f"token 必须互不相同，收到 {self.tokens}："
                "重名会让'第 0 位是哪个 token'这件事在行之间失去意义。"
            )
        for index, token in enumerate(self.tokens):
            if isinstance(token, bool) or not isinstance(token, int):
                raise ShapeError(f"tokens[{index}] 必须是整数，收到 {token!r}。")
            if token < 0 or token >= self.vocabulary:
                raise ShapeError(
                    f"tokens[{index}] = {token} 超出 [0, {self.vocabulary})。"
                )

    @property
    def inputs(self) -> Matrix:
        """``(n, V)`` 的 one-hot 输入矩阵."""
        return tuple(
            tuple(1.0 if column == token else 0.0 for column in range(self.vocabulary))
            for token in self.tokens
        )

    @property
    def target(self) -> Matrix:
        """``(n, V)``：**每一行都是第 0 位那个 token 的 one-hot**."""
        head = self.tokens[0]
        row = tuple(1.0 if column == head else 0.0 for column in range(self.vocabulary))
        return tuple(row for _ in range(self.length))

    @property
    def supervised(self) -> tuple[int, ...]:
        """所有行都参与损失."""
        return tuple(range(self.length))

    def shifted(self, offset: int) -> ReadoutTask:
        """循环位移 ``offset`` 位之后的那条序列（轨道里的第 ``offset`` 个元素）."""
        if isinstance(offset, bool) or not isinstance(offset, int):
            raise ParameterError(f"offset 必须是整数，收到 {offset!r}。")
        if offset < 0:
            raise ParameterError(f"offset 必须 >= 0，收到 {offset}。")
        resolved = offset % self.length
        moved = self.tokens[resolved:] + self.tokens[:resolved]
        return ReadoutTask(
            vocabulary=self.vocabulary,
            length=self.length,
            tokens=moved,
            seed=self.seed,
        )

    def orbit(self) -> tuple[ReadoutTask, ...]:
        """整条轨道：``n`` 个循环位移（含自身）."""
        return tuple(self.shifted(offset) for offset in range(self.length))

    def describe(self) -> str:
        """一行说明这条基序列长什么样."""
        return (
            f"位置选择任务：V={self.vocabulary}、n={self.length}、"
            f"基序列 {self.tokens}、轨道 {self.length} 个位移"
        )

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "vocabulary": self.vocabulary,
            "length": self.length,
            "tokens": list(self.tokens),
            "seed": self.seed,
            "orbit": [list(item.tokens) for item in self.orbit()],
            "describe": self.describe(),
        }


def check_task_is_a_position_readout(task: ReadoutTask) -> tuple[int, ...]:
    """返回“第 0 位 token 在轨道上的取值序列”（下界推导里 ``o_k`` 的那个下标）."""
    return tuple(item.tokens[0] for item in task.orbit())


def orbit_floor(task: ReadoutTask) -> float:
    """闭式下界 ``(n − 1) / (n·V)``（**置换等变模型在轨道上的平均损失不可能更低**）.

    推导见模块文档，最后两步是：

    ```text
    min_{G_j} Σ_k ‖G_j − o_k‖² 在 G_j = 袋子分布 b（轨道那 n 个 token 上各 1/n）处取到
    Σ_k ‖b − o_k‖² = n·(1 − 1/n) = n − 1        一共 n 行
    平均损失 = n(n−1)/(n²V) = (n − 1)/(n·V)
    ```

    注意它同时含 ``n`` 与 ``V``：**轨道只覆盖 n 个类别**，
    因此“每一行输出均匀分布 ``1/V``”**不是**最优的等变预测器
    （那是第一版写错的地方，见模块文档的注记）。
    """
    length = task.length
    vocabulary = task.vocabulary
    return (length - 1) / (length * vocabulary)


def bag_predictor(task: ReadoutTask) -> Matrix:
    """袋子预测器：每一行都输出序列里那 ``n`` 个 token 的经验分布（各 ``1/n``）.

    它是**最优的置换等变预测器**（在轨道平均损失的意义下），
    而它本身也是一个等变映射（输出与行的顺序无关）。
    “第 0 位是什么”与“这一袋里有什么”之间的差，就是位置知识的价格。
    """
    share = 1.0 / task.length
    counts = [0] * task.vocabulary
    for token in task.tokens:
        counts[token] += 1
    row = tuple(share * count for count in counts)
    return tuple(row for _ in range(task.length))


def _orbit_mean_loss_of_outputs(
    task: ReadoutTask,
    outputs: Sequence[Matrix],
) -> float:
    """给定轨道上每一步的输出，算平均 MSE（与 ``self_attention`` 的 MSE 同口径）."""
    if len(outputs) != task.length:
        raise ShapeError(
            f"输出个数 {len(outputs)} 与轨道长度 {task.length} 不一致。"
        )
    total = 0.0
    for shifted, output in zip(task.orbit(), outputs, strict=True):
        checked = validate_matrix(output, name="output")
        target = shifted.target
        if matrix_shape(checked) != matrix_shape(target):
            raise ShapeError(
                f"输出形状 {matrix_shape(checked)} 与目标 {matrix_shape(target)} 不一致。"
            )
        rows, columns = matrix_shape(checked)
        total += (
            math.fsum(
                (value - expected) ** 2
                for row, expected_row in zip(checked, target, strict=True)
                for value, expected in zip(row, expected_row, strict=True)
            )
            / (rows * columns)
        )
    return total / task.length


def floor_witness(task: ReadoutTask) -> float:
    """把**袋子预测器**代进轨道，实测它的平均损失——它必须等于闭式下界.

    这一步的意义是“下界是紧的”：一个**等变的**函数（常数映射）已经达到了它，
    因此“等变模型类的最优值”就是这个数。少了这一步，下界只是“某个不能超过的数”。

    它与 :func:`orbit_floor` 是两条**不同**的路径：一个是代数推导，
    一个是把矩阵代进去逐项算。第一版两条路径都用了同一个错误假设（均匀分布），
    因此它们“自洽地一起错了”——这也是为什么本课的判据必须让一侧走真实轨道。
    """
    return _orbit_mean_loss_of_outputs(task, [bag_predictor(task)] * task.length)


def orbit_losses(
    params: AttentionParams,
    table: EncodingTable,
    task: ReadoutTask,
    *,
    causal: bool = False,
) -> tuple[float, ...]:
    """模型在轨道上每一步的损失（**同一批参数、同一张表、n 条位移序列**）."""
    losses: list[float] = []
    for shifted in task.orbit():
        forward = positional_forward(
            params,
            table,
            shifted.inputs,
            target=shifted.target,
            supervised=shifted.supervised,
            causal=causal,
        )
        if forward.loss is None:  # pragma: no cover - target 已给定
            raise NumericError("轨道上的某一步没有算出损失。")
        losses.append(forward.loss)
    return tuple(losses)


def orbit_mean_loss(
    params: AttentionParams,
    table: EncodingTable,
    task: ReadoutTask,
    *,
    causal: bool = False,
) -> float:
    """轨道上的**平均**损失（下界就是对这个量说的）."""
    losses = orbit_losses(params, table, task, causal=causal)
    return math.fsum(losses) / len(losses)


def orbit_loss_spread(
    params: AttentionParams,
    table: EncodingTable,
    task: ReadoutTask,
    *,
    causal: bool = False,
) -> float:
    """轨道上损失极差 ``max − min``.

    等变模型在这个量上是**大**的（不同位移的目标不同，而模型给不出不同的输出），
    而一个好的位置感知模型上它是**小**的。它是“模型的输出真的随位移变了”的读数。
    """
    losses = orbit_losses(params, table, task, causal=causal)
    return max(losses) - min(losses)


def orbit_accuracy(
    params: AttentionParams,
    table: EncodingTable,
    task: ReadoutTask,
    *,
    causal: bool = False,
) -> float:
    """轨道上的平均命中率（每一行的 argmax 是否命中第 0 位那个 token）."""
    hits = 0.0
    for shifted in task.orbit():
        forward = positional_forward(
            params, table, shifted.inputs, causal=causal
        )
        hits += row_argmax_hits(
            forward.attention.output, shifted.target, list(shifted.supervised)
        )
    return hits / task.length


def sample_positions(count: int, *, seed: int = 7) -> tuple[int, ...]:
    """从轨道里抽 ``count`` 个起点（用确定性随机数）——训练时不必跑全部位移."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ParameterError(f"count 必须是 >= 1 的整数，收到 {count!r}。")
    weights = tuple(1.0 for _ in range(count))
    indices: list[int] = []
    raw = uniforms(count * 3, seed=seed)
    for index in range(count):
        item = sample_index(weights, u=raw[index % len(raw)])
        indices.append(item)
    return tuple(indices)


@dataclass(frozen=True)
class FloorVariant:
    """一个变体（“某个旋钮的一次设置”）在轨道上的读数.

    ```text
    equivariant   这个变体是否**落在下界适用的那个类里**（无掩码 + 位置表全零/不加表）
    causal        是否开了因果掩码
    floor         它要越过的那个下界（等变变体是“不能低于”，其余是“应该低于”）
    initial/final 第 0 步与最后一步的**轨道平均损失**
    accuracy      最后一步的轨道平均命中率
    trajectory    步数 + 1 个损失读数（含第 0 步）
    ```
    """

    name: str
    description: str
    equivariant: bool
    causal: bool
    table_kind: str | None
    floor: float
    initial_loss: float
    final_loss: float
    accuracy: float
    trajectory: tuple[float, ...]
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.name not in VARIANT_DESCRIPTIONS:
            raise ParameterError(
                f"未知的变体名 {self.name!r}：可选 {', '.join(VARIANT_DESCRIPTIONS)}。"
            )
        for name in ("floor", "initial_loss", "final_loss", "accuracy"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or math.isnan(value):
                raise NumericError(f"{name} 必须是数，收到 {value!r}。")
        if not 0.0 <= self.accuracy <= 1.0:
            raise NumericError(f"accuracy 必须落在 [0, 1]，收到 {self.accuracy!r}。")
        if len(self.trajectory) < 2:
            raise NumericError("轨迹至少要有两个读数（含第 0 步）。")
        object.__setattr__(self, "trajectory", tuple(float(v) for v in self.trajectory))
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def below_floor(self) -> bool:
        """轨道平均损失是否**低于下界**（等变变体做到这件事说明实现有问题）."""
        return self.final_loss < self.floor - FLOOR_TOLERANCE

    @property
    def respects_floor(self) -> bool:
        """轨道平均损失是否**不低于下界**（等变变体应该满足它）."""
        return not self.below_floor

    @property
    def improvement(self) -> float:
        """从第 0 步到最后一步降了多少（比例，可能为 0）."""
        if self.initial_loss <= 0:  # pragma: no cover - 初始损失恒为正
            return 0.0
        return 1.0 - self.final_loss / self.initial_loss

    @property
    def verdict(self) -> str:
        """“等变 → 停在界上”还是“破对称 → 越过界”的一句话判决."""
        if self.equivariant:
            return "受下界约束" + ("（**越过了，实现有问题**）" if self.below_floor else "（停在界上）")
        return "不受下界约束" + ("（**越过了**）" if self.below_floor else "（没有越过）")

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "name": self.name,
            "description": self.description,
            "equivariant": self.equivariant,
            "causal": self.causal,
            "table_kind": self.table_kind,
            "floor": self.floor,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "accuracy": self.accuracy,
            "below_floor": self.below_floor,
            "improvement": self.improvement,
            "verdict": self.verdict,
            "trajectory": list(self.trajectory),
            "notes": list(self.notes),
        }

    def summary_line(self) -> str:
        """一行说明：``无掩码+无位置编码 | 0.1663 → 0.1389（下界 0.138889）| 命中率 17% | 受下界约束（停在界上）``."""
        return (
            f"{self.description.split('：')[0]} | {self.initial_loss:.4f} → "
            f"{self.final_loss:.6f}（下界 {self.floor:.6f}）| 命中率 {self.accuracy:.0%} | "
            f"{self.verdict}"
        )


@dataclass(frozen=True)
class SymmetryFloorReport:
    """三个变体在一张表上的对照（**这一课的结论就落在这里**）."""

    task: ReadoutTask
    floor: float
    witness: float
    variants: tuple[FloorVariant, ...]
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        resolved = tuple(self.variants)
        if not resolved:
            raise ParameterError("对照报告至少要有一个变体。")
        names = {item.name for item in resolved}
        if len(resolved) == len(VARIANT_DESCRIPTIONS) and names != set(VARIANT_DESCRIPTIONS):
            raise NumericError(
                f"三个变体必须齐备：缺 {sorted(set(VARIANT_DESCRIPTIONS) - names)}——"
                "少了对照的那一份表，'位置编码有用'这句话就没有参照物。"
            )
        object.__setattr__(self, "variants", resolved)
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def equivariant_variants(self) -> tuple[FloorVariant, ...]:
        """落在下界适用类里的变体."""
        return tuple(item for item in self.variants if item.equivariant)

    @property
    def breaking_variants(self) -> tuple[FloorVariant, ...]:
        """打破对称性的变体."""
        return tuple(item for item in self.variants if not item.equivariant)

    @property
    def floor_violations(self) -> tuple[FloorVariant, ...]:
        """**不该低于下界**却低于下界的变体（数学上不可能 ⇒ 实现有问题）."""
        return tuple(item for item in self.equivariant_variants if item.below_floor)

    @property
    def crossers(self) -> tuple[FloorVariant, ...]:
        """越过了下界的变体（位置编码/掩码真的带来了顺序信息）."""
        return tuple(item for item in self.variants if item.below_floor)

    @property
    def tight(self) -> bool:
        """下界是否被“常数预测器”精确达到（它是紧的，不是估计出来的）."""
        return abs(self.witness - self.floor) <= FLOOR_TOLERANCE

    @property
    def ok(self) -> bool:
        """判决：等变变体**没有**越过下界，且至少有一个破对称变体**越过了**."""
        return not self.floor_violations and bool(self.crossers) and self.tight

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "ok": self.ok,
            "task": self.task.to_dict(),
            "floor": self.floor,
            "witness": self.witness,
            "tight": self.tight,
            "variants": [item.to_dict() for item in self.variants],
            "notes": list(self.notes),
        }

    def summary_line(self) -> str:
        """一行说明：``位置选择下界 0.138889（常数预测器实测 0.138889）| 3 个变体：2 个越界、1 个停在界上``."""
        return (
            f"位置选择下界 {self.floor:.6f}（常数预测器实测 {self.witness:.6f}）| "
            f"{len(self.variants)} 个变体：{len(self.crossers)} 个越界、"
            f"{len(self.variants) - len(self.crossers)} 个停在界上"
        )

    def table_lines(self) -> tuple[str, ...]:
        """把三个变体排成一张表（每一行一个变体）."""
        header = (
            f"  {'变体':<32} | {'初始损失':>9} | {'最终损失':>9} | {'命中率':>6} | 低于下界"
        )
        lines = [header, "  " + "-" * 88]
        for item in self.variants:
            label = item.description.split("：")[0]
            lines.append(
                f"  {label:<32} | {item.initial_loss:>9.4f} | {item.final_loss:>9.6f} | "
                f"{item.accuracy:>6.0%} | {'是' if item.below_floor else '否'}"
            )
        return tuple(lines)


def assemble_floor_report(
    task: ReadoutTask,
    variants: Sequence[FloorVariant],
) -> SymmetryFloorReport:
    """把三个变体组装成报告（下界与见证都现算，不从变体里读）."""
    return SymmetryFloorReport(
        task=task,
        floor=orbit_floor(task),
        witness=floor_witness(task),
        variants=tuple(variants),
        notes=(
            "下界 (1−1/V)/V 由等变性推出来，且被常数预测器精确达到（紧的）",
            "等变变体低于下界 = 实现有问题；破对称变体低于下界 = 位置信息真的被用上了",
            "因果掩码是打破对称性的另一条路，因此它也是必要的对照",
        ),
    )


__all__ = [
    "FLOOR_TOLERANCE",
    "VARIANT_CAUSAL",
    "VARIANT_DESCRIPTIONS",
    "VARIANT_EQUIVARIANT",
    "VARIANT_SINUSOIDAL",
    "FloorVariant",
    "ReadoutTask",
    "SymmetryFloorReport",
    "assemble_floor_report",
    "check_task_is_a_position_readout",
    "floor_witness",
    "make_readout_task",
    "orbit_accuracy",
    "orbit_floor",
    "orbit_loss_spread",
    "orbit_losses",
    "orbit_mean_loss",
    "sample_positions",
    "bag_predictor",
]
