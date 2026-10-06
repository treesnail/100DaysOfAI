"""``datasets``：一份**不可变**的数据集与它的内容指纹（day096 / M8-D7）.

```text
Sample           一条样本：(输入, 标签)
TabularDataset   一批不可变样本 + 名字 + 内容指纹（sha256 前 16 位）
make_dataset     由一串样本造数据集（校验非空、标签只能是 0 / 1）
split_dataset    用复用的 LCG 打乱后按比例切成 (训练, 评估) 两半，**两侧都非空**
```

## 一、为什么样本必须"进数据集就不再改"

一个数据集要能被两次实验比较，前提是"它是同一份"。若样本可以被原地改掉，
那么"上一轮跑出的准确率"与"这一轮跑出的准确率"之间就少了一个可以查证的锚点。
因此本模块的 :class:`TabularDataset` 是 ``@dataclass(frozen=True)``，
而 :func:`datasets.dataset_fingerprint` 把内容压成 ``sha256`` 的前 16 位——
**改一个字、加一条样本，指纹都会变**（与 day052 ``suite_fingerprint`` 同一条纪律）。

## 二、随机性必须复用 LCG，而不是自己造随机源

打乱用的是 day073 的 :func:`math_foundations.probability.uniforms`（一个线性同余
生成器）：同一个 ``(seed, n)`` 永远同一串数，于是"这一份切分"可以被逐位复现。
自己 ``import random`` 会让"同一个种子两台机器上切出两份数据"——
而它们**都叫同一个名字**，报告里却看不出差别。

## 三、为什么这一课的样本与 day094 / day095 同形

本课不重写网络：训练与推理都要把样本原样交给 ``regularization`` 的
:func:`loss_and_grad` / :func:`regularized_forward`。因此这里的 ``Sample``
就是 day094 的 ``Sample``（``(一串 D 维向量, 标签)``）——**一条也是它，一批也是它**。
"""

from __future__ import annotations

import hashlib
import json
import math

from dataclasses import dataclass

from smart_research_agent.math_foundations.types import Vector
from smart_research_agent.regularization.network import as_batch
from smart_research_agent.sequence_models.train import Sample
from smart_research_agent.torch_pipeline.errors import (
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.torch_pipeline.sampler import shuffle_indices
from smart_research_agent.transformer_core.errors import ShapeError as CoreShapeError

#: 一条样本的输入（``tuple[Vector, ...]``：本课的任务里它是一串 D 维向量）.
Input = tuple[Vector, ...]


def as_samples(samples: object, *, name: str = "samples") -> tuple[Sample, ...]:
    """收敛成一份样本元组（复用 day095 的批校验，再收紧到二分类标签 0 / 1）.

    形状与"标签是整数"这两件事**不在本包重写**：它调用 ``regularization.network.as_batch``，
    只额外要求标签落在 ``{0, 1}``——本课只做二分类，"标签是 3"必须当场拒绝。
    """
    try:
        checked = as_batch(samples, name=name)
    except CoreShapeError as exc:
        raise ShapeError(f"{name} 不是一份合法样本：{exc}") from exc
    for index, (_inputs, label) in enumerate(checked):
        if label not in (0, 1):
            raise NumericError(
                f"{name} 的第 {index} 项标签是 {label}，不在 {{0, 1}}："
                "本课只做二分类——多分类的 logits 与交叉熵在 day089 已经写好了。"
            )
    return checked


def dataset_fingerprint(samples: tuple[Sample, ...]) -> str:
    """一份样本的**内容指纹**（``sha256`` 前 16 位，与 day052 的 ``suite_fingerprint`` 同算法）.

    规范化 JSON（``sort_keys=True`` + 紧凑分隔符）是必需的：键顺序变了就换指纹，
    等于没回答"是不是同一份"。空样本集直接拒绝——"没有数据"不是一个可以取指纹的对象。
    """
    checked = as_samples(samples)
    payload = json.dumps(
        [
            {"inputs": [list(step) for step in inputs], "label": label}
            for inputs, label in checked
        ],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _shuffled_indices(size: int, seed: int) -> tuple[int, ...]:
    """打乱的**唯一实现**在 :mod:`sampler`（:func:`sampler.shuffle_indices`）里——本处只转发."""
    return shuffle_indices(size, seed=seed)


@dataclass(frozen=True)
class TabularDataset:
    """一份不可变的数据集：名字 + 样本 + 内容指纹（派生量用 :meth:`fingerprint` 现算）."""

    name: str
    samples: tuple[Sample, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ParameterError(f"数据集的名字必须是非空字符串，收到 {self.name!r}。")
        object.__setattr__(self, "samples", as_samples(self.samples, name=f"{self.name}.samples"))

    def __len__(self) -> int:
        """样本条数."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        """取一条样本（下标越界交给元组自己抛 ``IndexError``，**不静默取模**）."""
        return self.samples[index]

    def labels(self) -> tuple[int, ...]:
        """全部标签（顺序与样本顺序一致）."""
        return tuple(label for _inputs, label in self.samples)

    def fingerprint(self) -> str:
        """这份数据集的内容指纹（``sha256`` 前 16 位）."""
        return dataset_fingerprint(self.samples)

    def line(self) -> str:
        """一行说明：``sign | 32 条样本 | 正 16 / 负 16 | 指纹 a1b2c3d4e5f60718``."""
        labels = self.labels()
        return (
            f"{self.name} | {len(self)} 条样本 | 正 {sum(labels)} / 负 {len(labels) - sum(labels)}"
            f" | 指纹 {self.fingerprint()}"
        )


def make_dataset(samples: tuple[Sample, ...], *, name: str) -> TabularDataset:
    """由一串样本造一个数据集（校验都在 :class:`TabularDataset` 的构造里）."""
    return TabularDataset(name=name, samples=samples)


def split_dataset(
    dataset: TabularDataset, *, eval_ratio: float, seed: int
) -> tuple[TabularDataset, TabularDataset]:
    """用 LCG 打乱后按比例切成 ``(训练集, 评估集)``——**两侧都非空**.

    ``eval_ratio`` 必须落在开区间 ``(0, 1)``：取 0 会让评估集为空
    （"每轮评估"变成"每轮评估一个空集"），取 1 会让训练集为空。
    切点由 ``round(n × eval_ratio)`` 定，再**夹到** ``[1, n−1]``：
    这条夹取规则的目的正是保证两侧都非空，它是写进手册的口径，不是静默兜底。

    打乱只依赖 ``(seed, len(dataset))``：同一个 seed 永远切出同一对数据集
    （这正是 :data:`types.PROPERTY_RESUME_MATCHES_UNINTERRUPTED` 的前提）。
    """
    if isinstance(eval_ratio, bool) or not isinstance(eval_ratio, (int, float)):
        raise ParameterError(f"eval_ratio 必须是数，收到 {eval_ratio!r}。")
    ratio = float(eval_ratio)
    if not math.isfinite(ratio) or not 0.0 < ratio < 1.0:
        raise ParameterError(
            f"eval_ratio 必须落在开区间 (0, 1)，收到 {eval_ratio!r}："
            "取 0 时评估集为空、取 1 时训练集为空，两种都会让'每轮评估'失去意义。"
        )
    size = len(dataset)
    if size < 2:
        raise NumericError(
            f"数据集只有 {size} 条样本，切不出「训练 + 评估」两半："
            "本课要求两侧都非空。"
        )
    order = _shuffled_indices(size, int(seed))
    eval_count = int(round(size * ratio))
    eval_count = min(max(eval_count, 1), size - 1)
    eval_indices = order[:eval_count]
    train_indices = order[eval_count:]
    train = make_dataset(tuple(dataset[index] for index in train_indices), name=f"{dataset.name}/train")
    evaluation = make_dataset(tuple(dataset[index] for index in eval_indices), name=f"{dataset.name}/eval")
    return train, evaluation


__all__ = [
    "Input",
    "Sample",
    "TabularDataset",
    "as_samples",
    "dataset_fingerprint",
    "make_dataset",
    "split_dataset",
]
