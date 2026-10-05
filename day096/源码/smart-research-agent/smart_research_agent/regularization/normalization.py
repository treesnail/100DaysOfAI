"""``normalization``：批量归一化（BatchNorm）的**可复现实现**（day095 / M8-D6）.

```text
batch_statistics      逐特征算批内的 μ 与 σ²（**有偏方差**：分母是 N）
running_update        running ← (1−m)·running + m·batch   （momentum 是"每个批占多少"）
batch_norm_forward    训练相用本批统计量、推理相用 running ⇒ 返回 (y, cache, running')
batch_norm_backward   训练相**三项**、推理相**一项**（统计量依赖不依赖 x 的差别）
```

## 一、两条归一化轴：这一课的第一句话

```text
BatchNorm（本包）   固定一个特征，在**批**上算 μ/σ    ⇒ 与批里有哪些样本有关
LayerNorm（day079） 固定一条样本，在**特征**上算 μ/σ  ⇒ 与批里有多少条样本无关
```

把批矩阵**转置**之后，两条轴互换——这就是第 ② 条性质：

```text
batch_norm_forward(X)  ==  transpose(layer_norm(transpose(X), γ 按批重复, β 按批重复))
```

"逐位相同"这句话之所以成立，是因为本包的 ``DEFAULT_EPSILON`` 与
day079 的 ``encoder_decoder.DEFAULT_EPSILON`` **取同一个值**（``1e-5``）——
两处各自取一个 eps 是跨天对账最常见的失效方式。

## 二、两个相：这一课唯一"写反了不会报错"的地方

```text
训练相   μ/σ 来自**这一批**，同时把 running 统计量往这批靠一步
推理相   μ/σ 来自 **running**（**必须由调用方给出**，否则抛 PhaseError）
```

把它写反的症状很具体：单条样本推理时"批内方差"恒为 0，
整层输出被压成常数 β——而形状、有限性、覆盖率全部正常。

## 三、反向：三项 vs 一项

```text
训练相   μ/σ 也是 x 的函数 ⇒ dx = (1/σ)·(dŷ − mean_batch(dŷ) − x̂ ⊙ mean_batch(dŷ ⊙ x̂))
推理相   μ/σ 是常数        ⇒ dx = γ ⊙ dy / σ
```

这不是"两种实现"，而是**同一个式子在两种相对关系下**的样子。
它同时也是"两相必须配对"最硬的证据：把推理相当成训练相来求导，
数值上**不会**对（第 ④ / ⑤ 条性质把两条路都钉住）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from smart_research_agent.math_foundations.types import Matrix, Vector
from smart_research_agent.regularization.errors import (
    NumericError,
    ParameterError,
    PhaseError,
    ShapeError,
)
from smart_research_agent.regularization.types import (
    DEFAULT_EPSILON,
    DEFAULT_MOMENTUM,
    PHASE_EVAL,
    PHASE_TRAIN,
    PHASES,
)


def _checked_epsilon(value: object) -> float:
    """ε：必须是**正的**有限数（0 会让 0 方差那一格除零）."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParameterError(f"epsilon 必须是数，收到 {value!r}。")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ParameterError(
            f"epsilon 必须是正的有限数，收到 {value!r}："
            "0 会让'批内方差恰好是 0'那一格除零，而它不会报错，只会给出 inf / nan。"
        )
    return resolved


def _checked_momentum(value: object) -> float:
    """momentum：落在 ``[0, 1]``（它是"新统计量占多少"，不是"遗忘率"）."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParameterError(f"momentum 必须是数，收到 {value!r}。")
    resolved = float(value)
    if not math.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
        raise ParameterError(
            f"momentum 必须落在 [0, 1]，收到 {value!r}："
            "它是'这一批的统计量占多少'——m=1 就是'完全丢掉历史'。"
        )
    return resolved


def _checked_phase(value: object) -> str:
    """相的名字必须是 ``train`` / ``eval`` 之一（**不给它挑一个默认值**）."""
    if value not in PHASES:
        raise PhaseError(f"不认识的相 {value!r}：可用取值 {list(PHASES)}。")
    return str(value)


def as_matrix(batch: object, *, name: str = "batch") -> Matrix:
    """把二维序列收敛成 ``Matrix``（非空、行等长、元素有限）."""
    if isinstance(batch, (str, bytes)) or not hasattr(batch, "__iter__"):
        raise ShapeError(f"{name} 必须是若干行（可迭代的行的序列），收到 {type(batch).__name__}。")
    rows = list(batch)
    if not rows:
        raise ShapeError(f"{name} 不能为空：批里至少要有一条样本、一个特征。")
    checked: list[tuple[float, ...]] = []
    width: int | None = None
    for index, row in enumerate(rows):
        if isinstance(row, (str, bytes)) or not hasattr(row, "__iter__"):
            raise ShapeError(f"{name} 的第 {index} 行不是序列。")
        values = tuple(float(value) for value in row)
        if not values:
            raise ShapeError(f"{name} 的第 {index} 行为空。")
        if any(not math.isfinite(value) for value in values):
            raise NumericError(f"{name} 的第 {index} 行含有非有限数（nan / inf）。")
        if width is None:
            width = len(values)
        elif len(values) != width:
            raise ShapeError(f"{name} 的第 {index} 行宽度 {len(values)} 与首行 {width} 不一致。")
        checked.append(values)
    return tuple(checked)


def as_vector(values: object, *, name: str) -> Vector:
    """把一串数收敛成 ``Vector``（非空、元素有限）."""
    if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
        raise ShapeError(f"{name} 必须是一串数，收到 {type(values).__name__}。")
    items = tuple(values)
    if not items:
        raise ShapeError(f"{name} 不能为空。")
    checked: list[float] = []
    for index, item in enumerate(items):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ShapeError(f"{name}[{index}] 必须是实数，收到 {item!r}。")
        number = float(item)
        if not math.isfinite(number):
            raise NumericError(f"{name}[{index}] 是非有限数（{item!r}）。")
        checked.append(number)
    return tuple(checked)


def matrix_shape(matrix: Matrix) -> tuple[int, int]:
    """矩阵形状 ``(行数 = 批大小, 列数 = 特征数)``."""
    return len(matrix), len(matrix[0])


def transpose(matrix: Matrix) -> Matrix:
    """转置（第 ② 条性质要用它把"沿批"与"沿特征"两条轴对上）."""
    rows, columns = matrix_shape(matrix)
    return tuple(tuple(matrix[row][column] for row in range(rows)) for column in range(columns))


@dataclass(frozen=True)
class BatchStatistics:
    """一批样本上的逐特征统计量（**有偏方差**：分母是批大小 N）."""

    mean: Vector
    variance: Vector

    @property
    def features(self) -> int:
        """特征数 F."""
        return len(self.mean)

    def line(self) -> str:
        """一行说明：``μ 均值 +0.123456 | σ² 范围 [0.500000, 1.500000]``."""
        if not self.variance:
            return "空统计量"
        return (
            f"μ 均值 {math.fsum(self.mean) / len(self.mean):+.6f} | "
            f"σ² 范围 [{min(self.variance):.6f}, {max(self.variance):.6f}]"
        )


@dataclass(frozen=True)
class RunningStatistics:
    """跨批累积的统计量（推理相用它）."""

    mean: Vector
    variance: Vector
    batches: int = 0

    @property
    def features(self) -> int:
        """特征数 F."""
        return len(self.mean)

    def line(self) -> str:
        """一行说明：``running（3 批）| μ 均值 ... | σ² 范围 [...]``."""
        if not self.variance:
            return "空 running 统计量"
        return (
            f"running（{self.batches} 批）| μ 均值 "
            f"{math.fsum(self.mean) / len(self.mean):+.6f} | "
            f"σ² 范围 [{min(self.variance):.6f}, {max(self.variance):.6f}]"
        )


@dataclass(frozen=True)
class BatchNormCache:
    """一次前向留下的中间量（反向需要的**全部**东西）."""

    phase: str
    mean: Vector
    std: Vector
    normalized: Matrix
    gamma: Vector

    @property
    def features(self) -> int:
        """特征数 F."""
        return len(self.mean)


@dataclass(frozen=True)
class BatchNormGradients:
    """一次反向的三块梯度：``dγ`` / ``dβ`` / ``dx``."""

    d_gamma: Vector
    d_beta: Vector
    d_inputs: Matrix

    def norm(self) -> float:
        """整体 L2 范数（"这一步被推动了多大"）."""
        return math.sqrt(
            math.fsum(value * value for value in self.d_gamma)
            + math.fsum(value * value for value in self.d_beta)
            + math.fsum(value * value for row in self.d_inputs for value in row)
        )


def initial_running(features: int) -> RunningStatistics:
    """running 统计量的初值：``μ = 0``、``σ² = 1``（"还没见过任何一批"）."""
    if isinstance(features, bool) or not isinstance(features, int) or features < 1:
        raise ParameterError(f"特征数必须是 >= 1 的整数，收到 {features!r}。")
    return RunningStatistics(mean=(0.0,) * features, variance=(1.0,) * features, batches=0)


def batch_statistics(batch: Matrix) -> BatchStatistics:
    """逐特征算批内均值与**有偏**方差.

    ```text
    μ_j  = (1/N)·Σ_n x_nj
    σ²_j = (1/N)·Σ_n (x_nj − μ_j)²
    ```

    分母是 ``N`` 而不是 ``N−1``：它与 PyTorch 的 BatchNorm 以及 day079 的 LayerNorm
    **同口径**。用 ``N−1`` 的后果在批很小时很明显（N=2 时方差会大一倍），
    而它只在"批大小为 1"时才报错——那正是第 ⑤ 条笔记要写下来的边界。
    """
    checked = as_matrix(batch, name="batch")
    rows, columns = matrix_shape(checked)
    means = tuple(math.fsum(checked[row][column] for row in range(rows)) / rows for column in range(columns))
    variances = tuple(
        math.fsum((checked[row][column] - means[column]) ** 2 for row in range(rows)) / rows
        for column in range(columns)
    )
    return BatchStatistics(mean=means, variance=variances)


def running_update(
    running: RunningStatistics, stats: BatchStatistics, *, momentum: float = DEFAULT_MOMENTUM
) -> RunningStatistics:
    """把一批统计量折进 running：``r ← (1−m)·r + m·这批``."""
    resolved = _checked_momentum(momentum)
    if running.features != stats.features:
        raise ShapeError(
            f"running 有 {running.features} 个特征而这一批有 {stats.features} 个："
            "两者必须描述同一组特征。"
        )
    return RunningStatistics(
        mean=tuple(
            (1.0 - resolved) * old + resolved * new
            for old, new in zip(running.mean, stats.mean, strict=True)
        ),
        variance=tuple(
            (1.0 - resolved) * old + resolved * new
            for old, new in zip(running.variance, stats.variance, strict=True)
        ),
        batches=running.batches + 1,
    )


def batch_norm_forward(
    batch: Matrix,
    *,
    gamma: Vector | None = None,
    beta: Vector | None = None,
    epsilon: float = DEFAULT_EPSILON,
    phase: str = PHASE_TRAIN,
    running: RunningStatistics | None = None,
    momentum: float = DEFAULT_MOMENTUM,
) -> tuple[Matrix, BatchNormCache, RunningStatistics]:
    """批量归一化的前向，返回 ``(y, cache, running')``.

    ``gamma`` / ``beta`` 缺省是"什么都不改"的那一组（``γ = 1``、``β = 0``）——
    此时输出就是纯粹的标准化结果。它们可学，且长度必须等于**特征数 F**
    （逐特征，不逐样本）。

    推理相（``phase="eval"``）**必须**给出 ``running``：拿本批统计量去做推理
    正是这一课要拦下的那件事（单条样本时方差为 0，整层被压成常数 β）。
    """
    checked = as_matrix(batch, name="batch")
    rows, columns = matrix_shape(checked)
    resolved_phase = _checked_phase(phase)
    resolved_epsilon = _checked_epsilon(epsilon)
    features = columns
    resolved_gamma = as_vector((1.0,) * features if gamma is None else gamma, name="gamma")
    resolved_beta = as_vector((0.0,) * features if beta is None else beta, name="beta")
    if len(resolved_gamma) != features:
        raise ShapeError(
            f"gamma 长度 {len(resolved_gamma)} 与特征数 {features} 不一致："
            "γ/β 是**逐特征**的，不是逐样本的。"
        )
    if len(resolved_beta) != features:
        raise ShapeError(f"beta 长度 {len(resolved_beta)} 与特征数 {features} 不一致。")

    if resolved_phase == PHASE_TRAIN:
        stats = batch_statistics(checked)
        base = initial_running(features) if running is None else running
        running_out = running_update(base, stats, momentum=momentum)
        means, variances = stats.mean, stats.variance
    else:
        if running is None:
            raise PhaseError(
                "推理相必须给出 running 统计量："
                "拿本批统计量去推理时，单条样本的批内方差恒为 0，"
                "整层输出会被压成常数 β——而形状与有限性全都正常。"
            )
        if running.features != features:
            raise ShapeError(
                f"running 统计量有 {running.features} 个特征而这一批有 {features} 个。"
            )
        running_out = running
        means, variances = running.mean, running.variance

    std = tuple(math.sqrt(value + resolved_epsilon) for value in variances)
    normalized = tuple(
        tuple(
            (checked[row][column] - means[column]) / std[column] for column in range(features)
        )
        for row in range(rows)
    )
    output = tuple(
        tuple(
            resolved_gamma[column] * normalized[row][column] + resolved_beta[column]
            for column in range(features)
        )
        for row in range(rows)
    )
    cache = BatchNormCache(
        phase=resolved_phase, mean=means, std=std, normalized=normalized, gamma=resolved_gamma
    )
    return output, cache, running_out


def batch_norm_backward(
    cache: BatchNormCache, grad_output: Matrix, *, beta: Vector | None = None
) -> BatchNormGradients:
    """批量归一化的反向（**训练相三项、推理相一项**）.

    ```text
    dβ_j = Σ_n dy_nj
    dγ_j = Σ_n dy_nj · x̂_nj
    dŷ   = γ ⊙ dy
    训练相  dx_nj = (1/σ_j)·( dŷ_nj − mean_n(dŷ_j) − x̂_nj·mean_n(dŷ_j ⊙ x̂_j) )
    推理相  dx_nj = dŷ_nj / σ_j
    ```

    训练相那三项之所以存在，是因为 ``μ`` 与 ``σ`` **也是 x 的函数**；
    推理相的 ``μ/σ`` 来自 running（常数），那两个减项就不该出现。
    把推理相写成训练相（或反过来）**形状全对**，只有数值差分能发现——
    这就是第 ④ / ⑤ 条性质各钉一条路的原因。
    """
    if not isinstance(cache, BatchNormCache):
        raise ParameterError(f"cache 必须是 BatchNormCache，收到 {type(cache).__name__}。")
    checked_grad = as_matrix(grad_output, name="grad_output")
    rows, columns = matrix_shape(checked_grad)
    features = cache.features
    if columns != features:
        raise ShapeError(
            f"回传梯度有 {columns} 列而缓存有 {features} 个特征：两者必须来自同一次前向。"
        )
    if rows != len(cache.normalized):
        raise ShapeError(
            f"回传梯度有 {rows} 行而这一批有 {len(cache.normalized)} 行："
            "反向的梯度必须与**同一次**前向的批配对。"
        )
    if beta is not None and len(as_vector(beta, name="beta")) != features:
        raise ShapeError(f"beta 长度与特征数 {features} 不一致。")

    d_gamma = tuple(
        math.fsum(checked_grad[row][column] * cache.normalized[row][column] for row in range(rows))
        for column in range(features)
    )
    d_beta = tuple(
        math.fsum(checked_grad[row][column] for row in range(rows)) for column in range(features)
    )
    scaled = tuple(
        tuple(cache.gamma[column] * checked_grad[row][column] for column in range(features))
        for row in range(rows)
    )
    if cache.phase == PHASE_TRAIN:
        first = tuple(
            math.fsum(scaled[row][column] for row in range(rows)) / rows for column in range(features)
        )
        second = tuple(
            math.fsum(scaled[row][column] * cache.normalized[row][column] for row in range(rows))
            / rows
            for column in range(features)
        )
        d_inputs = tuple(
            tuple(
                (
                    scaled[row][column]
                    - first[column]
                    - cache.normalized[row][column] * second[column]
                )
                / cache.std[column]
                for column in range(features)
            )
            for row in range(rows)
        )
    else:
        d_inputs = tuple(
            tuple(scaled[row][column] / cache.std[column] for column in range(features))
            for row in range(rows)
        )
    return BatchNormGradients(d_gamma=d_gamma, d_beta=d_beta, d_inputs=d_inputs)


def batch_size_one_report(batch: Matrix) -> dict[str, float]:
    """``批大小为 1`` 时的读数：方差恒为 0 ⇒ 标准化结果恒为 0 ⇒ 输出恒为 β.

    这是一个**独立于实现的读数**（它直接来自定义），因此可以被写成一条可断言的结论。
    """
    checked = as_matrix(batch, name="batch")
    rows, columns = matrix_shape(checked)
    stats = batch_statistics(checked)
    output, _cache, _running = batch_norm_forward(checked)
    return {
        "rows": float(rows),
        "columns": float(columns),
        "max_variance": max(stats.variance),
        "max_abs_output": max(abs(value) for row in output for value in row),
    }


__all__ = [
    "BatchNormCache",
    "BatchNormGradients",
    "BatchStatistics",
    "RunningStatistics",
    "as_matrix",
    "as_vector",
    "batch_norm_backward",
    "batch_norm_forward",
    "batch_size_one_report",
    "batch_statistics",
    "initial_running",
    "matrix_shape",
    "running_update",
    "transpose",
]
