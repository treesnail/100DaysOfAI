"""``batching``：把若干条样本绑在一起跑（day087 / M7-D11）.

批处理换来的是**吞吐**，代价是两样东西，而两样都要算清楚：

```text
① 填充       一次前向要求等宽 ⇒ 短的那几条要陪着等到最长的那条
② 首字延迟   一条要等全批算完才出第一个 token（**延迟与吞吐是两个方向**）
```

因此"批越大越好"这句话本课一句都不写。它换成一张账：

```text
静态批       把请求按 max_batch 分好组，一组一起跑到底 ⇒ 总步数 = Σ 每组的最长
连续批       一条算完就把它那个位置让给新的一条     ⇒ 总步数 = 最长的那一条
```

两个都是**整数**，因此"连续批比静态批省多少步"是一个能被算出来的数。

## 一条必须写下来的口径：槽位效率的分母

```text
槽位效率 = Σ 各条长度 / (总步数 × max_batch)
```

分母里那个 `总步数 × max_batch` 是"这一次**有机会**算的 token 数"，
分子是"真的算了有意义工作的那部分"。静态批之所以低，是因为短的那条
在填充位上占了位置却没有产出——**那就是填充浪费在批这一层的同一个数**。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from smart_research_agent.hf_integration.types import TextBatch
from smart_research_agent.inference_optim.errors import ParameterError, ShapeError

#: 静态批默认的批大小（与 day086 的演示同量级，便于对照）。
DEFAULT_MAX_BATCH = 3


def padding_waste(batch: TextBatch) -> tuple[float, int, int]:
    """一批的填充浪费：``(占比, 有效 token, 总槽位)``.

    三个数一起返回，因为占比单独看没有意义：一条样本的批占比是 0.0，
    而"没有浪费"与"什么都没查"必须分得开（day085 起的那条纪律）。
    """
    if batch.batch_size == 0:
        raise ShapeError("空批没有填充可说：它既没有浪费、也没有有效 token。")
    effective = sum(row.real_length for row in batch.rows)
    slots = batch.batch_size * batch.width
    waste = 0.0 if slots == 0 else 1.0 - effective / slots
    return waste, effective, slots


def _check_lengths(lengths: tuple[int, ...]) -> tuple[int, ...]:
    if not lengths:
        raise ShapeError("长度表为空：没有请求就没有批。")
    for index, length in enumerate(lengths):
        if length < 1:
            raise ShapeError(
                f"第 {index} 条的长度是 {length}：零长度的请求没有 token 可算，"
                "而它会让'批的最大长度'这个数凭空变大或变小。"
            )
    return lengths


@dataclass(frozen=True)
class BatchGroup:
    """静态批里的一组：成员下标 + 这一组的宽度 + 步数."""

    members: tuple[int, ...]
    width: int

    @property
    def steps(self) -> int:
        """这一组要跑多少步（= 组内最长的那条）."""
        return self.width

    def line(self) -> str:
        """一行说明：``组 [0, 1, 2]：宽 9 ⇒ 9 步``."""
        return f"组 {list(self.members)}：宽 {self.width} ⇒ {self.steps} 步"


@dataclass(frozen=True)
class BatchPlan:
    """一次批处理的账：静态与连续两种做法各自的步数与效率."""

    lengths: tuple[int, ...]
    max_batch: int
    strategy: str
    groups: tuple[BatchGroup, ...] = field(default=())

    @property
    def tokens(self) -> int:
        """有效 token 总数（就是各条长度之和）."""
        return sum(self.lengths)

    @property
    def static_steps(self) -> int:
        """静态批的总步数（Σ 每组的最长）."""
        return sum(group.steps for group in self.groups)

    @property
    def continuous_steps(self) -> int:
        """连续批的**下界**步数（= 最长的那一条；**槽位够用时它就是实际步数**）.

        请求比槽位多时实际步数会更长——那一个数在
        :func:`continuous_batching_schedule` 返回的表长度里。
        """
        return max(self.lengths)

    @property
    def steps(self) -> int:
        """这一次选了哪一条路线（由 ``strategy`` 决定）."""
        return self.static_steps if self.strategy == "static" else self.continuous_steps

    @property
    def slots(self) -> int:
        """这一次**有机会**算的 token 数（分母）."""
        return self.steps * self.max_batch

    @property
    def efficiency(self) -> float:
        """槽位效率：有效 token / 机会槽位（1.0 = 一个位置都没浪费）."""
        if self.slots == 0:  # pragma: no cover - steps 与 max_batch 都为正
            return 0.0
        return self.tokens / self.slots

    @property
    def saved_steps(self) -> int:
        """连续批比静态批省下的步数（**整数**，因此它能被逐项解释）."""
        return self.static_steps - self.continuous_steps

    def to_dict(self) -> dict[str, object]:
        """摊平成可 JSON 化的字段（两种路线一起印）."""
        return {
            "lengths": list(self.lengths),
            "max_batch": self.max_batch,
            "strategy": self.strategy,
            "tokens": self.tokens,
            "static_steps": self.static_steps,
            "continuous_steps": self.continuous_steps,
            "steps": self.steps,
            "slots": self.slots,
            "efficiency": self.efficiency,
            "saved_steps": self.saved_steps,
        }

    def lines(self) -> tuple[str, ...]:
        """逐行读数（先分组、再一行总计）."""
        head = tuple(group.line() for group in self.groups)
        return head + (
            f"静态批 {self.static_steps} 步 / 连续批 {self.continuous_steps} 步 | "
            f"省下 {self.saved_steps} 步 | 这一次走 {self.strategy}：{self.steps} 步、"
            f"槽位效率 {self.efficiency:.1%}（有效 {self.tokens} / 机会 {self.slots}）",
        )


def plan_batches(
    lengths: tuple[int, ...] | list[int],
    *,
    max_batch: int = DEFAULT_MAX_BATCH,
    strategy: str = "static",
) -> BatchPlan:
    """按 ``max_batch`` 把请求分组，并算出两种路线的步数.

    ``strategy`` 只决定"这一次按哪条路线算步数"，两种路线**都要印出来**——
    因为"连续批省了多少"这件事就是静态批与连续批之差（一个整数）。
    """
    checked = _check_lengths(tuple(int(length) for length in lengths))
    if max_batch < 1:
        raise ParameterError(f"max_batch 必须 >= 1，收到 {max_batch}。")
    if strategy not in ("static", "continuous"):
        raise ParameterError(
            f"未知的批策略 {strategy!r}：本包只认 'static' 与 'continuous'。"
            "回退到静态批的后果是——一份'省了 12 步'的读数被印成'一步没省'。"
        )
    groups: list[BatchGroup] = []
    for start in range(0, len(checked), max_batch):
        members = tuple(range(start, min(start + max_batch, len(checked))))
        width = max(checked[index] for index in members)
        groups.append(BatchGroup(members=members, width=width))
    return BatchPlan(
        lengths=checked, max_batch=max_batch, strategy=strategy, groups=tuple(groups)
    )


def continuous_batching_schedule(
    lengths: tuple[int, ...] | list[int],
    *,
    max_batch: int = DEFAULT_MAX_BATCH,
) -> tuple[tuple[int, ...], ...]:
    """连续批的处理表：每一步"占用槽位的那几条请求"的下标.

    形状是"步 × 槽位"，空位用 ``-1`` 表示。它存在的理由是
    "连续批省了多少"必须能被**看见**，而不是只出现在一个总数里：

    ```text
    步 0   [0, 1, 2]     三条一起进
    步 2   [0, 1, 3]     第 2 条（长度 2）算完了，第 3 条填进它那个位置
    ```

    ## 一条必须逐条核对的不变量

    每一条请求在表里出现**恰好** ``它的长度`` 次——这是"它在每一步都在算"的
    唯一证据。少了这个核对，"某一条被提前扔掉"与"它算完了"在表里长得一样
    （前者会让占用率虚高，而**总步数不变**）。

    ## 总步数不一定是"最长的那一条"

    :attr:`BatchPlan.continuous_steps` 给出的 ``max(lengths)`` 是**槽位够用时的下界**：
    请求比槽位多的时候，排在后面的那几条要等前面算完才进得来，因此**实际步数会更长**。
    本函数循环到"全部请求都算完"为止，因此返回的表长度就是**实际步数**——
    这一点值得写下来，因为"连续批的总步数 = 最长的那条"是一句只在
    ``len(lengths) <= max_batch`` 时成立的话。
    """
    checked = _check_lengths(tuple(int(length) for length in lengths))
    if max_batch < 1:
        raise ParameterError(f"max_batch 必须 >= 1，收到 {max_batch}。")
    remaining = list(checked)
    queue = list(range(len(checked)))
    active: list[int] = []
    schedule: list[tuple[int, ...]] = []
    while queue or active:
        while queue and len(active) < max_batch:
            active.append(queue.pop(0))
        schedule.append(tuple(list(active) + [-1] * (max_batch - len(active))))
        for member in active:
            remaining[member] -= 1
        active = [member for member in active if remaining[member] > 0]
    return tuple(schedule)


def per_request_steps(
    schedule: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    """处理表里**每一条请求各出现了几次**（下标即请求号）.

    它是一条不变量：第 i 条必须出现恰好 ``lengths[i]`` 次。
    缺了这个核对，"某一条被提前扔掉"与"它算完了"在表里长得一样。
    """
    if not schedule:
        raise ShapeError("处理表为空：没有步就没有请求。")
    counts = [0] * (max((member for row in schedule for member in row), default=-1) + 1)
    for row in schedule:
        for member in row:
            if member >= 0:
                counts[member] += 1
    return tuple(counts)


def occupancy(schedule: tuple[tuple[int, ...], ...]) -> tuple[float, int, int]:
    """处理表的占用率：``(占用率, 真的算了工作的槽位, 总槽位)``.

    它比 :attr:`BatchPlan.efficiency` 更细：后者算的是"有效 token / 机会槽位"，
    而这里算的是"**有请求占着**的槽位 / 总槽位"——两者之差就是"请求在跑但位置被填"。
    """
    if not schedule:
        raise ShapeError("处理表为空：没有步就没有槽位。")
    total = 0
    occupied = 0
    for row in schedule:
        if not row:
            raise ShapeError("处理表里有一行是空的：空行没有槽位可以占用。")
        total += len(row)
        occupied += sum(1 for member in row if member >= 0)
    return occupied / total, occupied, total


def throughput(tokens: int, steps: int) -> float:
    """每一步产出多少有效 token（**吞吐的方向是 higher**）."""
    if steps < 1:
        raise ShapeError(f"步数必须 >= 1，收到 {steps}。")
    return tokens / steps


def batch_line(plan: BatchPlan) -> str:
    """一行读数：两种路线的步数、省下的步数与槽位效率."""
    return (
        f"长度 {list(plan.lengths)} 批大小 {plan.max_batch} | 静态 {plan.static_steps} 步、"
        f"连续 {plan.continuous_steps} 步（省 {plan.saved_steps}）| "
        f"槽位效率 {plan.efficiency:.1%} | 每步吞吐 {throughput(plan.tokens, plan.steps):.2f} token"
    )


def padding_line(batch: TextBatch) -> str:
    """一行读数：一批的填充浪费（与 day086 的池化表同源的那件事）."""
    waste, effective, slots = padding_waste(batch)
    return (
        f"{batch.batch_size} 条 × 宽 {batch.width} = {slots} 槽位 | 有效 {effective} | "
        f"浪费 {waste:.1%}（**这就是批这一层的填充代价**）"
    )


__all__ = [
    "DEFAULT_MAX_BATCH",
    "BatchGroup",
    "BatchPlan",
    "batch_line",
    "continuous_batching_schedule",
    "occupancy",
    "padding_line",
    "padding_waste",
    "per_request_steps",
    "plan_batches",
    "throughput",
]
