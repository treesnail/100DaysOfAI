"""``dataloader``：按批切分、可选丢弃尾批，并把多进程分片接进来（day096 / M8-D7）.

```text
Batch                    一批：下标 + 输入 + 标签（三者等长，且逐位对应）
collate(dataset, indices) 把一串下标取成一批（越界下标当场拒绝）
DataLoader               构造 DataLoader → 每个 epoch 打乱 → 先分片 → 再切批
```

## 一、三个读数的口径写死在这里

```text
批数          drop_last=True：n // b；drop_last=False：ceil(n / b) = (n + b − 1) // b
丢弃样本      dropped = n − b × 批数
本 epoch 批数 ``len(loader)`` = 这份 worker 分片的批数（workers=1 时 n 就是数据集大小）
```

三者是同一个式子的三面：批数定了，丢弃与"最后一批评不满"的关系就定了。
:data:`types.PROPERTY_DROP_LAST_MATCHES_BATCH_FORMULA` 因此可以把"批数"与
"``n // b``"逐位对一次账。

## 二、顺序是"先分片、再切批"

```text
Sampler.order(epoch)   →   一串全局访问顺序（同 seed 同 epoch 逐位相同）
shard_indices(…, k)    →   属于 worker k 的那些**位置**
按 batch_size 切块       →   这个 worker 本 epoch 的批
```

顺序刻意写成这两步：**分片发生在切批之前**，因此"一个样本只能进一个 worker 的批"
是一条结构性事实，而不是靠"块大小恰好整除"侥幸成立。
"""

from __future__ import annotations

from dataclasses import dataclass

from smart_research_agent.torch_pipeline.datasets import (
    Input,
    Sample,
    TabularDataset,
)
from smart_research_agent.torch_pipeline.errors import (
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.torch_pipeline.sampler import Sampler, shard_indices


@dataclass(frozen=True)
class Batch:
    """一批样本：下标、输入、标签三者等长（顺序即"这一批的喂入顺序"）."""

    indices: tuple[int, ...]
    inputs: tuple[Input, ...]
    labels: tuple[int, ...]

    def __post_init__(self) -> None:
        if not (len(self.indices) == len(self.inputs) == len(self.labels)):
            raise ShapeError(
                f"批的三个分量长度不一致：下标 {len(self.indices)}、输入 {len(self.inputs)}、"
                f"标签 {len(self.labels)}——zip 会静默截断成较短的那一边。"
            )
        for index, label in enumerate(self.labels):
            if isinstance(label, bool) or not isinstance(label, int):
                raise ShapeError(f"第 {index} 个标签必须是整数，收到 {label!r}。")

    @property
    def size(self) -> int:
        """这一批有多少条样本."""
        return len(self.indices)

    def line(self) -> str:
        """一行说明：``批 | 8 条 | 下标 0..7 | 标签 正 5 / 负 3``."""
        if not self.indices:
            return "批 | 0 条（空批）"
        return (
            f"批 | {self.size} 条 | 下标 {self.indices[0]}..{self.indices[-1]} | "
            f"标签 正 {sum(self.labels)} / 负 {self.size - sum(self.labels)}"
        )


def collate(dataset: TabularDataset, indices: tuple[int, ...]) -> Batch:
    """把一串下标取成一批（越界 / 非整数下标当场拒绝，**不静默取模**）."""
    if not isinstance(dataset, TabularDataset):
        raise ParameterError(f"dataset 必须是 TabularDataset，收到 {type(dataset).__name__}。")
    if isinstance(indices, (str, bytes)) or not hasattr(indices, "__iter__"):
        raise ShapeError(f"indices 必须是一串下标，收到 {type(indices).__name__}。")
    checked: list[int] = []
    for position, index in enumerate(indices):
        if isinstance(index, bool) or not isinstance(index, int):
            raise ShapeError(f"第 {position} 个下标必须是整数，收到 {index!r}。")
        if not 0 <= index < len(dataset):
            raise ShapeError(
                f"第 {position} 个下标 {index} 越界（数据集只有 {len(dataset)} 条）："
                "越界取样本会静默拿到错误的样本，而不是报错。"
            )
        checked.append(index)
    pairs = [dataset[index] for index in checked]
    return Batch(
        indices=tuple(checked),
        inputs=tuple(inputs for inputs, _label in pairs),
        labels=tuple(label for _inputs, label in pairs),
    )


def batch_samples(batch: Batch) -> tuple[Sample, ...]:
    """把一批还原成一串样本（``(输入, 标签)``）——交给 ``regularization`` 的就是它."""
    return tuple(zip(batch.inputs, batch.labels))


class DataLoader:
    """按 ``batch_size`` 切分的装载器：可打乱、可丢尾批、可只取某个 worker 的分片.

    它不是"迭代器"，而是一个**纯函数式**的批生成器：``batches(epoch)`` 每次现算，
    同样的 ``(epoch, seed, batch_size, …)`` 永远给出同一串批。这样"第 3 轮的批"
    才是一个可以被两次实验共同引用的对象。
    """

    def __init__(
        self,
        dataset: TabularDataset,
        batch_size: int,
        *,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int = 0,
        workers: int = 1,
        worker_id: int = 0,
    ) -> None:
        if not isinstance(dataset, TabularDataset):
            raise ParameterError(
                f"dataset 必须是 TabularDataset，收到 {type(dataset).__name__}。"
            )
        if len(dataset) == 0:  # pragma: no cover - 数据集构造期已拒绝空集
            raise NumericError("空数据集切不出任何批。")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ParameterError(
                f"batch_size 必须是 >= 1 的整数，收到 {batch_size!r}："
                "批大小为 0 时'批数'没有定义。"
            )
        if not isinstance(shuffle, bool):
            raise ParameterError(f"shuffle 必须是布尔值，收到 {shuffle!r}。")
        if not isinstance(drop_last, bool):
            raise ParameterError(f"drop_last 必须是布尔值，收到 {drop_last!r}。")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.workers = workers
        self.worker_id = worker_id
        # 这两次调用同时完成"参数合法性"与"本 worker 分片大小"两件事
        self.sampler = Sampler(size=len(dataset), shuffle=shuffle, seed=seed)
        self.worker_shard = shard_indices(len(dataset), workers=workers, worker_id=worker_id)

    # ------------------------------------------------------------------ 读数

    def steps_per_epoch(self) -> int:
        """本 epoch 的批数（= :meth:`__len__`，两个名字指同一件事）."""
        return len(self)

    def dropped_samples(self) -> int:
        """本 epoch 被丢掉的样本数：``max(0, n − b × 批数)``.

        ``drop_last=True`` 时它等于尾批（``n % b``）；``drop_last=False`` 时尾批**被保留**，
        因此它是 0——那个 ``max(0, …)`` 正是"保留尾批"与"丢弃尾批"这件事的判据所在。
        """
        return max(0, len(self.worker_shard) - self.batch_size * len(self))

    def __len__(self) -> int:
        """本 epoch 的批数：``n // b``（drop_last）或 ``ceil(n / b)``."""
        size = len(self.worker_shard)
        if self.drop_last:
            return size // self.batch_size
        return (size + self.batch_size - 1) // self.batch_size

    # ------------------------------------------------------------------ 生成

    def batches(self, epoch: int) -> tuple[Batch, ...]:
        """第 ``epoch`` 轮的全部批（先按 :class:`Sampler` 定序，再分片，最后切块）."""
        order = self.sampler.order(epoch)
        positions = shard_indices(len(order), workers=self.workers, worker_id=self.worker_id)
        chosen = tuple(order[position] for position in positions)
        produced: list[Batch] = []
        for start in range(0, len(chosen), self.batch_size):
            chunk = chosen[start : start + self.batch_size]
            if self.drop_last and len(chunk) < self.batch_size:
                continue
            produced.append(collate(self.dataset, chunk))
        return tuple(produced)

    def line(self) -> str:
        """一行说明：``装载器 | sign | b=8 | shuffle=True | drop_last=True | workers=1#0 | 批数 4 | 丢弃 0``."""
        return (
            f"装载器 | {self.dataset.name} | b={self.batch_size} | shuffle={self.shuffle} | "
            f"drop_last={self.drop_last} | workers={self.workers}#{self.worker_id} | "
            f"批数 {len(self)} | 丢弃 {self.dropped_samples()}"
        )


__all__ = [
    "Batch",
    "DataLoader",
    "batch_samples",
    "collate",
]
