"""``sampler``：把 epoch 号映射成一串**确定**的访问顺序，并把数据分给多个工作进程（day096 / M8-D7）.

```text
shuffle_indices(size, seed)          一次 Fisher–Yates 打乱（复用 LCG，不是全局随机源）
Sampler.order(epoch)                 第 epoch 轮的访问顺序（不洗牌时就是 0,1,2,…）
shard_indices(size, workers, k)      worker k 的那一份下标（互不相交、并集为全集）
```

## 一、为什么"打乱只依赖 (seed, epoch)"

```text
若打乱依赖一个全局随机源
  ⇒ 同一个 seed 在两台机器上给出两份不同的「第 3 轮的批」
  ⇒ 而它们都叫「第 3 轮」——报告里看不出差别，对账时才发现对不上
因此本模块把打乱写成 (seed, epoch) 的纯函数：**只要这两个数一样，顺序就逐位一样**。
```

## 二、为什么分片必须是"取模"，而不是"切块"

```text
取模（i % workers == k）   每个 worker 拿到的是「稀疏但均匀」的一份；并集 = 全集、两两不相交
切块（range(k·b,(k+1)·b)） 换一个 workers 就会漏样本 / 重叠——而它**不会报错**，只会少训几条
```

本课选择取模：它让"分片性"变成一条可以直接断言的性质
（见 :data:`types.PROPERTY_WORKER_SHARDS_PARTITION_THE_DATASET`，一条**下界**判据）。
"""

from __future__ import annotations

from dataclasses import dataclass

from smart_research_agent.math_foundations.probability import uniforms
from smart_research_agent.torch_pipeline.errors import NumericError, ParameterError


def _checked_size(size: object, *, name: str = "size") -> int:
    """``size`` / ``workers`` / ``worker_id`` 共用的整数校验（布尔不是整数）."""
    if isinstance(size, bool) or not isinstance(size, int):
        raise ParameterError(f"{name} 必须是整数，收到 {size!r}。")
    return int(size)


def _checked_seed(seed: object) -> int:
    """种子必须是整数（浮点种子会让"同一个种子"这句话失去意义）."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ParameterError(f"seed 必须是整数，收到 {seed!r}。")
    return int(seed)


def shuffle_indices(size: int, *, seed: int) -> tuple[int, ...]:
    """一次确定性的 **Fisher–Yates** 打乱（复用 day073 的 LCG ``uniforms``）.

    同一个 ``(size, seed)`` 永远同一串顺序。打乱源**不自己造**：
    它调用 :func:`math_foundations.probability.uniforms`，因此"这一串顺序"可以被逐位复核。
    """
    checked_size = _checked_size(size)
    if checked_size < 0:
        raise ParameterError(f"size 不能为负，收到 {size}。")
    checked_seed = _checked_seed(seed)
    draws = uniforms(checked_size, seed=checked_seed)
    order = list(range(checked_size))
    for position in range(checked_size - 1, 0, -1):
        swap = int(draws[position] * (position + 1))
        order[position], order[swap] = order[swap], order[position]
    return tuple(order)


def _mixed_seed(seed: int, epoch: int) -> int:
    """把 ``(seed, epoch)`` 混成 LCG 的种子（乘积常数取自 32 位黄金比例）."""
    return (seed * 2654435761 + epoch * 40503) % (2**32)


@dataclass(frozen=True)
class Sampler:
    """一个 epoch 的访问顺序：``shuffle=False`` 时是恒等顺序，否则是确定的打乱."""

    size: int
    shuffle: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        checked_size = _checked_size(self.size)
        if checked_size < 1:
            raise NumericError(f"size 必须 >= 1，收到 {self.size}：空数据集没有访问顺序。")
        if not isinstance(self.shuffle, bool):
            raise ParameterError(f"shuffle 必须是布尔值，收到 {self.shuffle!r}。")
        checked_seed = _checked_seed(self.seed)
        object.__setattr__(self, "size", checked_size)
        object.__setattr__(self, "seed", checked_seed)

    def order(self, epoch: int) -> tuple[int, ...]:
        """第 ``epoch`` 轮的访问顺序（``epoch`` 从 0 / 1 开始都可以，只影响打乱的种子）."""
        checked_epoch = _checked_size(epoch, name="epoch")
        if checked_epoch < 0:
            raise ParameterError(f"epoch 不能为负，收到 {epoch}。")
        if not self.shuffle:
            return tuple(range(self.size))
        return shuffle_indices(self.size, seed=_mixed_seed(self.seed, checked_epoch))

    def line(self) -> str:
        """一行说明：``采样器 | size=32 | shuffle=True | seed=100``."""
        return f"采样器 | size={self.size} | shuffle={self.shuffle} | seed={self.seed}"


def shard_indices(size: int, *, workers: int, worker_id: int) -> tuple[int, ...]:
    """把 ``0..size-1`` 分给 ``workers`` 个进程里的第 ``worker_id`` 个（**取模分片**）.

    ```text
    并集    ⋃_k  shard_indices(size, workers, k)  =  {0, 1, …, size−1}
    互斥    shard_indices(size, workers, j) ∩ shard_indices(size, workers, k) = ∅  （j ≠ k）
    ```
    这两条正是 :data:`types.PROPERTY_WORKER_SHARDS_PARTITION_THE_DATASET` 要钉住的事。
    参数非法（``workers < 1`` 或 ``worker_id`` 越界）一律 :class:`ParameterError`。
    """
    checked_size = _checked_size(size)
    if checked_size < 1:
        raise NumericError(f"size 必须 >= 1，收到 {size}。")
    checked_workers = _checked_size(workers, name="workers")
    if checked_workers < 1:
        raise ParameterError(f"workers 必须 >= 1，收到 {workers}。")
    checked_worker = _checked_size(worker_id, name="worker_id")
    if not 0 <= checked_worker < checked_workers:
        raise ParameterError(
            f"worker_id 必须落在 [0, {checked_workers})，收到 {worker_id}："
            "越界的 worker 拿到的是空分片，而'空'与'没有数据'在读报告时长得一样。"
        )
    return tuple(range(checked_worker, checked_size, checked_workers))


__all__ = [
    "Sampler",
    "shard_indices",
    "shuffle_indices",
]
