"""``positional_encoding`` 的训练回路与两组对照实验（day078 / M7-D3）.

训练这一课的东西需要一个**位置敏感**的任务，否则“有没有位置编码”根本看不出来。
本模块用 :class:`~smart_research_agent.positional_encoding.symmetry.ReadoutTask`
（“每一行都读出第 0 位那个 token”），因为它在等变模型上有一个**可证的下界**：

```text
等变模型（无掩码 + 不加位置编码）  轨道平均损失 >= (1 − 1/V)/V
破对称模型（加位置编码 / 开因果掩码）  不受该下界约束
```

于是两组实验都落在一句话上：**“低于下界”是位置信息真的被用上的证据。**

```text
实验一 compare_encodings      四个旋钮：不加 / 正弦表 / 可学习表（随机）/ 可学习表（从正弦出发）
实验二 symmetry_floor_study   三个变体：等变 / 正弦 / 因果掩码——下界的判决书
```

两组实验都用同一个任务、同一批初始参数、同一个优化器配置，
**只换一个旋钮**（day068 起反复强调的纪律：一次只改一个）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from smart_research_agent.math_foundations.optim import (
    Objective,
    Optimizer,
    Schedule,
    flatten_matrices,
)
from smart_research_agent.math_foundations.types import Vector
from smart_research_agent.positional_encoding.errors import (
    NumericError,
    ParameterError,
)
from smart_research_agent.positional_encoding.layers import (
    learnable_table,
    loss_gradient,
    positional_backward,
    positional_forward,
    sinusoidal_table,
    zero_table,
)
from smart_research_agent.positional_encoding.symmetry import (
    VARIANT_CAUSAL,
    VARIANT_DESCRIPTIONS,
    VARIANT_EQUIVARIANT,
    VARIANT_SINUSOIDAL,
    FloorVariant,
    ReadoutTask,
    SymmetryFloorReport,
    assemble_floor_report,
    orbit_accuracy,
    orbit_floor,
    orbit_loss_spread,
    orbit_losses,
    orbit_mean_loss,
)
from smart_research_agent.positional_encoding.types import (
    GRAD_TABLE,
    GRAD_W_KEY,
    GRAD_W_OUTPUT,
    GRAD_W_QUERY,
    GRAD_W_VALUE,
    EncodingTable,
    flatten_parameters,
    unflatten_parameters,
)
from smart_research_agent.transformer_core.train import default_parameters

#: 两组实验共用的四个“编码”标签（**一次只换一个旋钮**的四个刻度）.
LABEL_NONE = "none"
LABEL_SINUSOIDAL = "sinusoidal"
LABEL_LEARNABLE_RANDOM = "learnable_random"
LABEL_LEARNABLE_FROM_SINE = "learnable_from_sine"

ENCODING_LABELS: tuple[str, ...] = (
    LABEL_NONE,
    LABEL_SINUSOIDAL,
    LABEL_LEARNABLE_RANDOM,
    LABEL_LEARNABLE_FROM_SINE,
)

ENCODING_LABEL_DESCRIPTIONS: dict[str, str] = {
    LABEL_NONE: "不加位置编码（全零表、冻结）——等变，因此受下界约束",
    LABEL_SINUSOIDAL: "正弦表（冻结）——它没有参数可训，位置知识是公式给的",
    LABEL_LEARNABLE_RANDOM: "可学习表（随机初始化）——位置知识必须全部从数据里学出来",
    LABEL_LEARNABLE_FROM_SINE: "可学习表（从正弦表出发）——先给一个“会读位置”的起点，再让它调",
}

#: 可训练的四个参数块 + 位置表（``inputs`` 不是参数，永远不会被训练）.
TRAINABLE_BLOCKS: tuple[str, ...] = (
    GRAD_W_QUERY,
    GRAD_W_KEY,
    GRAD_W_VALUE,
    GRAD_W_OUTPUT,
    GRAD_TABLE,
)

#: 梯度来源（与 day075/076 同名同义；本课的解析反向是手写的那个）.
GRADIENT_SOURCE_ANALYTIC = "analytic"


def _checked_steps(steps: Any) -> int:
    """校验步数（>= 1 的整数）."""
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise ParameterError(f"steps 必须是整数，收到 {steps!r}。")
    if steps < 1:
        raise ParameterError(f"steps 必须 >= 1，收到 {steps}。")
    return steps


def resolve_trainable(trainable: Sequence[str] | None) -> tuple[str, ...]:
    """校验“这一轮训练哪几块”（未知块、重复块、一块都不训都当场拒绝）."""
    if trainable is None:
        return TRAINABLE_BLOCKS
    resolved: list[str] = []
    for index, name in enumerate(trainable):
        if name not in TRAINABLE_BLOCKS:
            raise ParameterError(
                f"trainable[{index}] = {name!r} 不是可训练的参数块："
                f"可选 {', '.join(TRAINABLE_BLOCKS)}（inputs 不是参数，永远不训）。"
            )
        if name in resolved:
            raise ParameterError(f"trainable 里出现了重复的块 {name!r}。")
        resolved.append(name)
    if not resolved:
        raise ParameterError(
            "至少要训练一块：一块都不训时“训练”这个词没有意义——"
            "如果只想前向，请直接调用 positional_forward。"
        )
    return tuple(resolved)


def is_equivariant_table(table: EncodingTable, *, causal: bool) -> bool:
    """这张表 + 这个掩码设置是否落在“置换等变”那个类里（下界适用）.

    判据只有一条：**每一行的位置向量都是同一个（全是 0）**。
    全零表等价于“没有加任何东西”，因此它不破坏等变性；
    而“可学习表恰好初始化成全零、然后被训练”这件事不在这里判——
    报告判的是**这一次训练开始时**的类别（见 ``PositionalTrainingReport.equivariant_class``）。
    """
    if causal:
        return False
    return all(value == 0.0 for row in table.table for value in row)


def _block_slices(
    shapes: Sequence[tuple[int, int]],
    total: int,
) -> dict[str, range]:
    """算出压平向量里每一块的**下标区间**（用于把不训练的块置 0）.

    ``shapes`` 是 :func:`types.flatten_parameters` 给出的**五个**形状
    （四个投影 + 位置表），顺序与 ``TRAINABLE_BLOCKS`` 逐项对齐。
    这里刻意检查个数：少一项时“表的梯度落在哪一段”会静默地指向前一块。
    """
    if len(shapes) != len(TRAINABLE_BLOCKS):
        raise NumericError(
            f"形状表有 {len(shapes)} 项，而参数块有 {len(TRAINABLE_BLOCKS)} 个"
            "（四个投影 + 位置表）：两者必须逐项对齐，"
            "否则“哪一段属于哪一块”会静默地错位。"
        )
    sizes = [rows * columns for rows, columns in shapes]
    if sum(sizes) != total:
        raise NumericError(
            f"五块的元素个数 {sum(sizes)} 与压平向量长度 {total} 不一致。"
        )
    slices: dict[str, range] = {}
    cursor = 0
    for name, size in zip(TRAINABLE_BLOCKS, sizes, strict=True):
        slices[name] = range(cursor, cursor + size)
        cursor += size
    return slices


def freeze_blocks(
    flat: Vector,
    shapes: Sequence[tuple[int, int]],
    trainable: Sequence[str],
) -> Vector:
    """按**真实分块区间**把不训练的块置 0（形状无关，逐块精确）.

    为什么是“置 0”而不是“从向量里删掉”：位置表在压平向量里的**下标区间是固定的**，
    删掉一块会让后面所有块前移——而“表那一项在第几段”正是这一课要盯的东西。
    置 0 的代价只是名称仍然出现在报告里，好处是“冻结”这件事显式可见
    （见 :attr:`PositionalTrainingReport.frozen`）。
    """
    resolved = resolve_trainable(trainable)
    frozen = {name for name in TRAINABLE_BLOCKS if name not in resolved}
    if not frozen:
        return flat
    slices = _block_slices(shapes, len(flat))
    values = list(flat)
    for name in frozen:
        for index in slices[name]:
            values[index] = 0.0
    return tuple(values)


def orbit_gradients(
    params: AttentionParams,
    table: EncodingTable,
    task: ReadoutTask,
    *,
    causal: bool = False,
) -> Vector:
    """轨道上**平均**的六块梯度（压平成一串，顺序与 ``flatten_parameters`` 一致）.

    为什么在轨道上平均：损失是“对轨道说的”，梯度就必须是同一个函数对同一个量的梯度——
    抽一部分位移去算梯度会得到一个“对另一条曲线”的方向，
    而它与被测的那个损失之间的偏差**不会报错**，只会让曲线难看。
    """
    total: Vector | None = None
    count = 0
    for shifted in task.orbit():
        forward = positional_forward(
            params,
            table,
            shifted.inputs,
            target=shifted.target,
            supervised=shifted.supervised,
            causal=causal,
        )
        flat = positional_backward(forward, loss_gradient(forward)).flatten()
        if total is None:
            total = flat
        else:
            total = tuple(a + b for a, b in zip(total, flat, strict=True))
        count += 1
    if total is None:  # pragma: no cover - 轨道至少有一个元素
        raise NumericError("轨道为空：没有任何梯度可以算。")
    return tuple(value / count for value in total)


def analytic_objective(
    params: AttentionParams,
    table: EncodingTable,
    task: ReadoutTask,
    *,
    causal: bool = False,
) -> Objective:
    """轨道平均损失作为 ``θ ↦ 损失`` 的目标函数（供 ``calculus.gradient`` 手工核对）."""
    from smart_research_agent.positional_encoding.layers import positional_loss_from_flat

    _flat, shapes = flatten_parameters(params, table)
    ordinates = tuple(task.orbit())

    def objective(theta: Vector) -> float:
        """给定压平后的参数，返回轨道上的平均损失."""
        rebuilt_params, rebuilt_table = unflatten_parameters(
            theta, shapes, template=table
        )
        total = 0.0
        for shifted in ordinates:
            single = positional_loss_from_flat(
                rebuilt_params,
                rebuilt_table,
                shifted.inputs,
                target=shifted.target,
                supervised=shifted.supervised,
                causal=causal,
            )
            total += single(theta)
        return total / len(ordinates)

    return objective


@dataclass(frozen=True)
class PositionalTrainingReport:
    """一次训练的账（**三列读数**：轨道平均损失、轨道平均命中率、位置表的变化）.

    ```text
    trajectory      每一步的**轨道平均损失**，长度 steps + 1（含第 0 步）
    accuracies      每一步的轨道平均命中率，长度 steps + 1
    initial_table   训练前的位置表（用来量“表动了多少”）
    final_table     训练后的位置表
    ```

    三列都留下是刻意的：损失降了但命中率不涨、命中率涨了但表没动，
    这三种情形对应完全不同的结论（延续 day075/076 的“四个读数一起看”）。
    """

    task: ReadoutTask
    causal: bool
    trajectory: tuple[float, ...]
    accuracies: tuple[float, ...]
    initial_table: EncodingTable
    final_table: EncodingTable
    final_parameters: AttentionParams
    gradient_source: str = GRADIENT_SOURCE_ANALYTIC
    trainable: tuple[str, ...] = TRAINABLE_BLOCKS
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if len(self.trajectory) != len(self.accuracies):
            raise NumericError(
                f"两列读数的长度不一致（{len(self.trajectory)} 与 "
                f"{len(self.accuracies)}）：错开一行在图上只是“曲线略陡”。"
            )
        if len(self.trajectory) < 2:
            raise NumericError("轨迹至少要有两个读数（含第 0 步）。")
        for name in ("trajectory", "accuracies"):
            values = getattr(self, name)
            if any(not math.isfinite(value) for value in values):
                raise NumericError(f"{name} 里出现了非有限数。")
        if self.gradient_source != GRADIENT_SOURCE_ANALYTIC:
            raise ParameterError(
                f"未知的梯度来源 {self.gradient_source!r}：本课只提供解析反向。"
            )
        object.__setattr__(self, "trainable", resolve_trainable(self.trainable))
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def frozen(self) -> tuple[str, ...]:
        """被冻结的块（“这一次没训什么”必须写在报告里）."""
        return tuple(
            name for name in TRAINABLE_BLOCKS if name not in self.trainable
        )

    @property
    def steps(self) -> int:
        """训练步数（= 轨迹长度 − 1）."""
        return len(self.trajectory) - 1

    @property
    def initial_loss(self) -> float:
        """第 0 步的轨道平均损失."""
        return self.trajectory[0]

    @property
    def final_loss(self) -> float:
        """最后一步的轨道平均损失."""
        return self.trajectory[-1]

    @property
    def floor(self) -> float:
        """这个任务的下界（等变模型不能低于它）."""
        return orbit_floor(self.task)

    @property
    def equivariant_class(self) -> bool:
        """这一次训练是否落在下界适用的那个类里（无掩码 + 位置表全零）."""
        zero = all(value == 0.0 for row in self.final_table.table for value in row)
        initial_zero = all(value == 0.0 for row in self.initial_table.table for value in row)
        return (not self.causal) and zero and initial_zero

    @property
    def improvement_ratio(self) -> float:
        """从第 0 步到最后一步降了多少（比例）."""
        if self.initial_loss <= 0:  # pragma: no cover - 初始损失恒为正
            return 0.0
        return 1.0 - self.final_loss / self.initial_loss

    @property
    def initial_accuracy(self) -> float:
        """第 0 步的轨道平均命中率."""
        return self.accuracies[0]

    @property
    def final_accuracy(self) -> float:
        """最后一步的轨道平均命中率."""
        return self.accuracies[-1]

    @property
    def table_movement(self) -> float:
        """位置表动了多少：``max |final − initial|``（冻结表恒为 0.0）."""
        if self.initial_table.table == self.final_table.table:
            return 0.0
        return max(
            abs(a - b)
            for initial_row, final_row in zip(
                self.initial_table.table, self.final_table.table, strict=True
            )
            for a, b in zip(initial_row, final_row, strict=True)
        )

    @property
    def ok(self) -> bool:
        """这次训练是否“看起来正常”：损失没涨、读数有限、等变类没有越过下界."""
        if self.final_loss > self.initial_loss + 1e-12:
            return False
        if self.equivariant_class and self.final_loss < self.floor - 1e-12:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "task": self.task.to_dict(),
            "causal": self.causal,
            "steps": self.steps,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "floor": self.floor,
            "improvement_ratio": self.improvement_ratio,
            "initial_accuracy": self.initial_accuracy,
            "final_accuracy": self.final_accuracy,
            "table_movement": self.table_movement,
            "trainable": list(self.trainable),
            "frozen": list(self.frozen),
            "notes": list(self.notes),
        }

    def summary_line(self) -> str:
        """一行说明：``no_mask_sinusoidal 80 步 | 损失 0.1663 → 0.0112（↓93.3%）| 命中率 17% → 100% | 表动了 0.000000``."""
        frozen = "、".join(self.frozen) if self.frozen else "无"
        return (
            f"{self.final_table.kind} 表 {self.steps} 步 | 损失 {self.initial_loss:.4f} → "
            f"{self.final_loss:.6f}（↓{self.improvement_ratio:.1%}）| 命中率 "
            f"{self.initial_accuracy:.0%} → {self.final_accuracy:.0%} | "
            f"表动了 {self.table_movement:.6f} | 冻结：{frozen}"
        )

    def epoch_line(self, index: int) -> str:
        """第 ``index`` 步的一行读数（越界抛 ``ParameterError``）."""
        if index < 0 or index >= len(self.trajectory):
            raise ParameterError(
                f"步下标 {index} 越界（轨迹长度 {len(self.trajectory)}，含第 0 步）。"
            )
        return (
            f"  step {index:>4} | 损失 {self.trajectory[index]:.8f} | "
            f"命中率 {self.accuracies[index]:.0%}"
        )


def train_positions(
    initial_params: AttentionParams,
    initial_table: EncodingTable,
    task: ReadoutTask,
    *,
    optimizer: Optimizer,
    steps: int,
    causal: bool = False,
    trainable: Sequence[str] | None = None,
    schedule: Schedule | None = None,
) -> PositionalTrainingReport:
    """在轨道上训练“位置注入 + 注意力层”（**损失与梯度都在整条轨道上平均**）.

    ```text
    每一步：① 损失 = 轨道平均损失（与报告里的那一列同一个函数）
            ② 梯度 = 轨道平均梯度（解析反向，六块）
            ③ 冻结的块置 0（形状不变）
            ④ optimizer.step 压平向量 → 还原成“四个投影 + 一张表”
    ```
    """
    resolved_steps = _checked_steps(steps)
    resolved_trainable = resolve_trainable(trainable)
    params = initial_params
    table = initial_table
    _flat, shapes = flatten_parameters(params, table)
    losses = [orbit_mean_loss(params, table, task, causal=causal)]
    accuracies = [orbit_accuracy(params, table, task, causal=causal)]
    for step in range(resolved_steps):
        if schedule is not None:
            learning_rate = schedule(step)
            if not math.isfinite(learning_rate) or learning_rate <= 0:
                raise ParameterError(
                    f"第 {step} 步的学习率 {learning_rate!r} 不是正的有限数。"
                )
            optimizer.learning_rate = float(learning_rate)
        flat_gradient = orbit_gradients(params, table, task, causal=causal)
        shaped = freeze_blocks(flat_gradient, shapes, resolved_trainable)
        current = flatten_matrices(params.matrices() + (table.table,))[0]
        updated = optimizer.step(current, shaped)
        params, table = unflatten_parameters(updated, shapes, template=table)
        losses.append(orbit_mean_loss(params, table, task, causal=causal))
        accuracies.append(orbit_accuracy(params, table, task, causal=causal))
    return PositionalTrainingReport(
        task=task,
        causal=causal,
        trajectory=tuple(losses),
        accuracies=tuple(accuracies),
        initial_table=initial_table,
        final_table=table,
        final_parameters=params,
        gradient_source=GRADIENT_SOURCE_ANALYTIC,
        trainable=resolved_trainable,
        notes=(
            f"优化器：{optimizer.describe()}",
            "损失与梯度都在**整条位移轨道**上平均——抽一部分位移会让方向对不上损失",
        ),
    )


@dataclass(frozen=True)
class EncodingComparisonRow:
    """一个“编码刻度”的读数."""

    label: str
    description: str
    table_kind: str
    table_parameters: int
    table_trainable: bool
    equivariant: bool
    floor: float
    initial_loss: float
    final_loss: float
    accuracy: float
    table_movement: float

    def __post_init__(self) -> None:
        if self.label not in ENCODING_LABELS:
            raise ParameterError(
                f"未知的编码标签 {self.label!r}：可选 {', '.join(ENCODING_LABELS)}。"
            )
        for name in ("floor", "initial_loss", "final_loss", "accuracy", "table_movement"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise NumericError(f"{name} 必须是有限数，收到 {value!r}。")

    @property
    def below_floor(self) -> bool:
        """轨道平均损失是否低于下界."""
        return self.final_loss < self.floor - 1e-12

    @property
    def improvement_ratio(self) -> float:
        """降幅比例."""
        return 1.0 - self.final_loss / self.initial_loss if self.initial_loss > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "label": self.label,
            "description": self.description,
            "table_kind": self.table_kind,
            "table_parameters": self.table_parameters,
            "table_trainable": self.table_trainable,
            "equivariant": self.equivariant,
            "floor": self.floor,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "accuracy": self.accuracy,
            "table_movement": self.table_movement,
            "below_floor": self.below_floor,
            "improvement_ratio": self.improvement_ratio,
        }

    def summary_line(self) -> str:
        """一行说明：``sinusoidal | 表参数 0 | 0.1663 → 0.0112 | 命中率 100% | 低于下界 是``."""
        return (
            f"{self.label:<22} | 表参数 {self.table_parameters:>3} | "
            f"{self.initial_loss:.4f} → {self.final_loss:.6f} | "
            f"命中率 {self.accuracy:.0%} | 表动了 {self.table_movement:.6f} | "
            f"低于下界 {'是' if self.below_floor else '否'}"
        )


@dataclass(frozen=True)
class EncodingComparison:
    """四个“编码刻度”的对照表（**同一批初始参数、同一个任务、同一个优化器配置**）."""

    task: ReadoutTask
    rows: tuple[EncodingComparisonRow, ...]
    steps: int
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        resolved = tuple(self.rows)
        if len(resolved) < 2:
            raise ParameterError("对照至少要有两行：只有一行没有可比较的东西。")
        if len({row.label for row in resolved}) != len(resolved):
            raise NumericError("对照表里有重复的标签：同一行出现两次会让“对照”变成“重复”。")
        _checked_steps(self.steps)
        object.__setattr__(self, "rows", resolved)
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def floor(self) -> float:
        """这个任务的下界."""
        return orbit_floor(self.task)

    @property
    def equivariant_violations(self) -> tuple[EncodingComparisonRow, ...]:
        """等变的那些行里低于下界的（数学上不可能 ⇒ 实现有问题）."""
        return tuple(
            row for row in self.rows if row.equivariant and row.below_floor
        )

    @property
    def ok(self) -> bool:
        """对照是否自洽：等变行没有越过下界。"""
        return not self.equivariant_violations

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "ok": self.ok,
            "task": self.task.to_dict(),
            "steps": self.steps,
            "floor": self.floor,
            "rows": [row.to_dict() for row in self.rows],
            "notes": list(self.notes),
        }

    def summary_line(self) -> str:
        """一行说明：``编码对照 4 行 / 80 步 | 下界 0.138889 | 越过下界 3 行``."""
        return (
            f"编码对照 {len(self.rows)} 行 / {self.steps} 步 | 下界 {self.floor:.6f} | "
            f"越过下界 {sum(1 for row in self.rows if row.below_floor)} 行"
        )

    def table_lines(self) -> tuple[str, ...]:
        """把对照表排成一张表."""
        header = (
            f"  {'编码':<22} | {'表参数':>6} | {'初始损失':>9} | {'最终损失':>9} | "
            f"{'命中率':>6} | {'表动了':>9} | 低于下界"
        )
        lines = [header, "  " + "-" * 104]
        for row in self.rows:
            lines.append(
                f"  {row.label:<22} | {row.table_parameters:>6} | {row.initial_loss:>9.4f} | "
                f"{row.final_loss:>9.6f} | {row.accuracy:>6.0%} | {row.table_movement:>9.6f} | "
                f"{'是' if row.below_floor else '否'}"
            )
        return tuple(lines)


def _encoding_specs(
    task: ReadoutTask,
    *,
    seed: int,
    init_scale: float,
) -> tuple[tuple[str, EncodingTable, bool], ...]:
    """四个刻度各自的“表 + 表是否可训”（**唯一的旋钮就是这里**）."""
    positions, dimension = task.length, task.vocabulary
    return (
        (LABEL_NONE, zero_table(positions, dimension), False),
        (LABEL_SINUSOIDAL, sinusoidal_table(positions, dimension), False),
        (
            LABEL_LEARNABLE_RANDOM,
            learnable_table(
                positions, dimension, initializer="random", scale=init_scale, seed=seed
            ),
            True,
        ),
        (
            LABEL_LEARNABLE_FROM_SINE,
            learnable_table(positions, dimension, initializer="sinusoidal"),
            True,
        ),
    )


def compare_encodings(
    initial_params: AttentionParams,
    task: ReadoutTask,
    *,
    optimizer_factory: Callable[[], Optimizer],
    steps: int = 80,
    causal: bool = False,
    seed: int = 11,
    init_scale: float = 0.25,
) -> EncodingComparison:
    """四个“编码刻度”的对照：**一次只换一个旋钮**.

    ``optimizer_factory`` 是**工厂**而不是实例：优化器带状态（动量），
    复用同一个实例会让第二次训练带着第一次的动量起步——
    表现为“某一行收敛得莫名其妙地快”（day075/076 的同一条纪律）。
    """
    resolved_steps = _checked_steps(steps)
    rows: list[EncodingComparisonRow] = []
    for label, table, trainable in _encoding_specs(
        task, seed=seed, init_scale=init_scale
    ):
        report = train_positions(
            initial_params,
            table,
            task,
            optimizer=optimizer_factory(),
            steps=resolved_steps,
            causal=causal,
            trainable=TRAINABLE_BLOCKS if trainable else TRAINABLE_BLOCKS[:-1],
        )
        rows.append(
            EncodingComparisonRow(
                label=label,
                description=ENCODING_LABEL_DESCRIPTIONS[label],
                table_kind=table.kind,
                table_parameters=table.parameter_count if trainable else 0,
                table_trainable=trainable,
                equivariant=is_equivariant_table(table, causal=causal),
                floor=orbit_floor(task),
                initial_loss=report.initial_loss,
                final_loss=report.final_loss,
                accuracy=report.final_accuracy,
                table_movement=report.table_movement,
            )
        )
    return EncodingComparison(
        task=task,
        rows=tuple(rows),
        steps=resolved_steps,
        notes=(
            "四个刻度共用同一批初始参数与同一个任务；表的种类与“是否可训”是唯一的旋钮",
            "表参数一列给出“位置知识”有多少个自由度：正弦表恒为 0",
        ),
    )


def symmetry_floor_study(
    initial_params: AttentionParams,
    task: ReadoutTask,
    *,
    optimizer_factory: Callable[[], Optimizer],
    steps: int = 80,
) -> SymmetryFloorReport:
    """三个变体的下界判决书（等变 / 正弦 / 因果掩码）.

    返回 :class:`~smart_research_agent.positional_encoding.symmetry.SymmetryFloorReport`。
    判决只有一句：**等变变体停在界上，破对称变体越过界**。
    """
    resolved_steps = _checked_steps(steps)
    positions, dimension = task.length, task.vocabulary
    # 三个变体都用**冻结**的位置表（全零表或正弦表），因此它们只差两个旋钮：
    # 用哪张表、开不开掩码。"表被训练"这件事留给 encoding 对照，
    # 否则等变变体会自己学出一张非零的表，于是它**不再**落在下界适用的那个类里。
    frozen_blocks = TRAINABLE_BLOCKS[:-1]
    settings = (
        (VARIANT_EQUIVARIANT, zero_table(positions, dimension), False, True),
        (VARIANT_SINUSOIDAL, sinusoidal_table(positions, dimension), False, False),
        (VARIANT_CAUSAL, zero_table(positions, dimension), True, False),
    )
    variants: list[FloorVariant] = []
    for name, table, causal, equivariant in settings:
        report = train_positions(
            initial_params,
            table,
            task,
            optimizer=optimizer_factory(),
            steps=resolved_steps,
            causal=causal,
            trainable=frozen_blocks,
        )
        variants.append(
            FloorVariant(
                name=name,
                description=VARIANT_DESCRIPTIONS[name],
                equivariant=equivariant,
                causal=causal,
                table_kind=table.kind if table.parameter_count else None,
                floor=orbit_floor(task),
                initial_loss=report.initial_loss,
                final_loss=report.final_loss,
                accuracy=report.final_accuracy,
                trajectory=report.trajectory,
                notes=(
                    f"轨道损失极差（最后一步）{orbit_loss_spread(report.final_parameters, report.final_table, task, causal=causal):.6f}",
                    f"优化器：{report.steps} 步",
                ),
            )
        )
    return assemble_floor_report(task, variants)


def task_batch(task: ReadoutTask) -> tuple[tuple[Any, Any], ...]:
    """把轨道变成一个 ``(inputs, target)`` 批（供 ``layers.batch_*`` 使用）."""
    return tuple((item.inputs, item.target) for item in task.orbit())


def orbit_loss_vector(
    params: AttentionParams,
    table: EncodingTable,
    task: ReadoutTask,
    *,
    causal: bool = False,
) -> tuple[float, ...]:
    """轨道上每一步的损失（转发 ``symmetry.orbit_losses``，便于本模块单点引用）."""
    return orbit_losses(params, table, task, causal=causal)


__all__ = [
    "ENCODING_LABELS",
    "ENCODING_LABEL_DESCRIPTIONS",
    "GRADIENT_SOURCE_ANALYTIC",
    "LABEL_LEARNABLE_FROM_SINE",
    "LABEL_LEARNABLE_RANDOM",
    "LABEL_NONE",
    "LABEL_SINUSOIDAL",
    "TRAINABLE_BLOCKS",
    "EncodingComparison",
    "EncodingComparisonRow",
    "PositionalTrainingReport",
    "analytic_objective",
    "compare_encodings",
    "default_parameters",
    "freeze_blocks",
    "is_equivariant_table",
    "orbit_gradients",
    "orbit_loss_vector",
    "resolve_trainable",
    "symmetry_floor_study",
    "task_batch",
    "train_positions",
]
