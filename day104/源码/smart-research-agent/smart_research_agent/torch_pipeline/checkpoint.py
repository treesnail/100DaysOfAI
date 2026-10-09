"""``checkpoint``：把"训练到哪了"完整落成**五件套**，并且可校验、可恢复（day096 / M8-D7）.

```text
CHECKPOINT_FILES = meta.json + params.json + optimizer.json + metrics.json + history.jsonl

meta.json        清单：step / 权重哈希 / 优化器哈希 / 文件表 / 参数个数
params.json      压平后的权重（flatten_params 的产物）
optimizer.json   优化器的**跨步状态**（存什么由 OPTIMIZER_STATE_KEYS 决定）
metrics.json     这一轮的汇总读数（含 running 统计量与 initial_loss，供恢复用）
history.jsonl    逐步 / 逐轮的 EpochRecord（一行一条，便于画曲线）
```

## 一、为什么是"五件套"而不是"一个权重文件"

```text
只有权重   ⇒ 别人拿到手第一件事是问"这是哪个网络、训到第几步、优化器状态呢"
五件套     ⇒ 目录本身就是一份**自描述**的记录：能继续训（resume）、也能被审计
```

这与 day050 ``sft/checkpoint.py`` 留下的那条纪律同源（**产物要自描述**），
只是今天文件名独立（``params.json`` 而不是 ``model_state.json``），
因为本课的"参数"是 ``regularization`` 的那一份 ``RegularizedParams``。

## 二、清单里的哈希**只覆盖**权重与优化器状态

```text
params_sha256      覆盖 params.json 里的权重数组
optimizer_sha256   覆盖 optimizer.json 里被 OPTIMIZER_STATE_KEYS 选中的那些键
```

指标与历史**不参与校验**：它们是可读记录，允许被追加（恢复训练时会继续往
``history.jsonl`` 里写）。把可变记录纳入"完整性校验"会让"追加一行历史"
莫名其妙地把检查点判成损坏。

## 三、缺文件必须**点名**

一个缺了 ``params.json`` 的检查点**看着能用**，实际会丢权重——
所以 :func:`load_checkpoint` 在缺文件时把缺的那几个名字写进错误信息里。
这是 day050 那条纪律的直接延续："不返回部分可用的对象"。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smart_research_agent.optimizers.optimizer import SUPPORTED_OPTIMIZERS
from smart_research_agent.optimizers.types import OPTIMIZER_STATE_KEYS
from smart_research_agent.regularization.network import RegularizedParams, flatten_params
from smart_research_agent.regularization.train import EpochRecord
from smart_research_agent.torch_pipeline.errors import (
    CheckpointError,
    ParameterError,
    ShapeError,
)

#: 检查点固定包含的文件名（**顺序即写盘顺序**；``meta.json`` 最后写）.
CHECKPOINT_FILES: tuple[str, ...] = (
    "meta.json",
    "params.json",
    "optimizer.json",
    "metrics.json",
    "history.jsonl",
)

#: 优化器的标量簿记：每个优化器都存，但**不足以**恢复状态（恢复靠跨步状态键表）.
OPTIMIZER_SCALAR_KEYS: tuple[str, ...] = (
    "name",
    "core_rule",
    "learning_rate",
    "step_count",
    "weight_decay",
    "decay_mode",
    "max_grad_norm",
    "last_grad_norm",
    "last_clip_factor",
)


def _canonical(payload: Any) -> str:
    """规范化 JSON（键排序 + 紧凑分隔符）：哈希要能回答"是不是同一份"."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_of(payload: Any) -> str:
    """规范化 JSON 的 ``sha256``（完整 64 位——清单里的哈希不做截断）."""
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Any) -> Path:
    """写一个 JSON 文件（UTF-8、不转义中文、两空格缩进便于 diff）."""
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return path


def optimizer_payload(optimizer_state: dict[str, Any]) -> dict[str, Any]:
    """按 :data:`OPTIMIZER_STATE_KEYS` 选出"该存"的优化器状态.

    ```text
    标量簿记（step_count / learning_rate / …）  每个优化器都存
    跨步状态（velocity / first_moment / …）     由 OPTIMIZER_STATE_KEYS[name] 决定
    base（day074 的内部状态）                   若存在则原样保留（共享规则的动量在里面）
    ```
    """
    if not isinstance(optimizer_state, dict):
        raise ParameterError(
            f"optimizer_state 必须是 dict（optimizer.state() 的产物），"
            f"收到 {type(optimizer_state).__name__}。"
        )
    name = optimizer_state.get("name")
    if name not in OPTIMIZER_STATE_KEYS:
        raise ParameterError(
            f"优化器状态里的 name={name!r} 不在 {list(SUPPORTED_OPTIMIZERS)} 里："
            "没有那张键表就不知道该存哪些跨步状态。"
        )
    payload: dict[str, Any] = {
        key: optimizer_state[key] for key in OPTIMIZER_SCALAR_KEYS if key in optimizer_state
    }
    if "base" in optimizer_state:
        payload["base"] = optimizer_state["base"]
    for key in OPTIMIZER_STATE_KEYS[name]:
        if key in optimizer_state:
            payload[key] = optimizer_state[key]
    return payload


def record_payload(record: EpochRecord) -> dict[str, Any]:
    """把一个 :class:`EpochRecord` 变成可写进 ``history.jsonl`` 的字典（**不是 ``to_dict``**）."""
    if not isinstance(record, EpochRecord):
        raise ParameterError(f"history 的每一项必须是 EpochRecord，收到 {type(record).__name__}。")
    return {
        "epoch": record.epoch,
        "train_loss": record.train_loss,
        "eval_loss": record.eval_loss,
        "eval_accuracy": record.eval_accuracy,
        "learning_rate": record.learning_rate,
    }


def record_from_payload(payload: dict[str, Any]) -> EpochRecord:
    """把 ``history.jsonl`` 的一行还原成 :class:`EpochRecord`（恢复训练时需要它）."""
    try:
        return EpochRecord(
            epoch=int(payload["epoch"]),
            train_loss=float(payload["train_loss"]),
            eval_loss=float(payload["eval_loss"]),
            eval_accuracy=float(payload["eval_accuracy"]),
            learning_rate=float(payload["learning_rate"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointError(
            f"history.jsonl 的一行不是合法的 EpochRecord：{payload!r}（{exc}）"
        ) from exc


@dataclass(frozen=True)
class CheckpointManifest:
    """清单：这一份检查点"到第几步、有多少参数、两个哈希是多少"."""

    step: int
    params_sha256: str
    optimizer_sha256: str
    files: tuple[str, ...]
    params_count: int

    def __post_init__(self) -> None:
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0:
            raise ParameterError(f"step 必须是 >= 0 的整数，收到 {self.step!r}。")
        if isinstance(self.params_count, bool) or not isinstance(self.params_count, int):
            raise ParameterError(f"params_count 必须是整数，收到 {self.params_count!r}。")
        if self.params_count < 1:
            raise ShapeError(f"params_count 必须 >= 1，收到 {self.params_count}——空模型没有检查点。")
        object.__setattr__(self, "files", tuple(str(name) for name in self.files))

    def line(self) -> str:
        """一行说明：``step=24 | 参数 44 | 权重 a1b2…c3d4 | 优化器 e5f6…0718 | 文件 5``."""
        return (
            f"step={self.step} | 参数 {self.params_count} | 权重 {self.params_sha256[:12]}…"
            f"{self.params_sha256[-4:]} | 优化器 {self.optimizer_sha256[:12]}…"
            f"{self.optimizer_sha256[-4:]} | 文件 {len(self.files)}"
        )


def _manifest_payload(manifest: CheckpointManifest) -> dict[str, Any]:
    """清单 -> ``meta.json`` 的内容（**不是 ``to_dict``**）."""
    return {
        "step": manifest.step,
        "params_sha256": manifest.params_sha256,
        "optimizer_sha256": manifest.optimizer_sha256,
        "files": list(manifest.files),
        "params_count": manifest.params_count,
    }


@dataclass(frozen=True)
class Checkpoint:
    """读回来的检查点：权重是**压平后的**元组，优化器状态是那份被选中的字典."""

    step: int
    params: tuple[float, ...]
    optimizer_state: dict[str, Any]
    metrics: dict[str, Any]
    history: tuple[dict[str, Any], ...]
    manifest: CheckpointManifest

    def line(self) -> str:
        """一行说明：``检查点 step=24 | 参数 44 | 历史 3 轮 | step=24 …``."""
        return (
            f"检查点 {self.manifest.line()} | 参数 {len(self.params)} 个 | "
            f"历史 {len(self.history)} 轮"
        )


def save_checkpoint(
    directory: str | Path,
    *,
    params: RegularizedParams,
    optimizer_state: dict[str, Any],
    step: int,
    metrics: dict[str, Any],
    history: tuple[EpochRecord, ...],
) -> CheckpointManifest:
    """写出五件套并返回清单（``meta.json`` **最后写**：它在，说明前面都写完了）.

    哈希只覆盖**权重**与**优化器状态**：指标与历史是可读记录、允许被继续追加。
    """
    if not isinstance(params, RegularizedParams):
        raise ParameterError(f"params 必须是 RegularizedParams，收到 {type(params).__name__}。")
    if not isinstance(metrics, dict):
        raise ParameterError(f"metrics 必须是 dict，收到 {type(metrics).__name__}。")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ParameterError(f"step 必须是 >= 0 的整数，收到 {step!r}。")
    flat = flatten_params(params)
    payload = optimizer_payload(optimizer_state)

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    _write_json(target / "params.json", {"count": len(flat), "params": list(flat)})
    _write_json(target / "optimizer.json", payload)
    _write_json(target / "metrics.json", metrics)
    with (target / "history.jsonl").open("w", encoding="utf-8") as handle:
        for record in history:
            handle.write(
                json.dumps(record_payload(record), ensure_ascii=False, sort_keys=True) + "\n"
            )

    manifest = CheckpointManifest(
        step=int(step),
        params_sha256=_sha256_of(list(flat)),
        optimizer_sha256=_sha256_of(payload),
        files=CHECKPOINT_FILES,
        params_count=len(flat),
    )
    _write_json(target / "meta.json", _manifest_payload(manifest))
    return manifest


def _read_json(path: Path) -> Any:
    """读一个 JSON 文件（解析失败时**点名文件**）."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CheckpointError(f"{path.name} 不是合法 JSON：{exc}") from exc


def load_checkpoint(directory: str | Path) -> Checkpoint:
    """读回五件套（缺任何一个都抛 :class:`CheckpointError` 并**点名是哪一个**）."""
    target = Path(directory)
    if not target.is_dir():
        raise CheckpointError(f"检查点目录不存在：{target}")
    missing = [name for name in CHECKPOINT_FILES if not (target / name).exists()]
    if missing:
        raise CheckpointError(
            f"检查点不完整，缺少 {', '.join(missing)}（目录：{target}）："
            "一个缺了文件的检查点看着能用，实际 load 出来的东西是错的。"
        )
    params_payload = _read_json(target / "params.json")
    optimizer_state = _read_json(target / "optimizer.json")
    metrics = _read_json(target / "metrics.json")
    meta = _read_json(target / "meta.json")
    if not isinstance(params_payload, dict) or "params" not in params_payload:
        raise CheckpointError("params.json 缺少 params 字段。")
    history: list[dict[str, Any]] = []
    for number, line in enumerate(
        (target / "history.jsonl").read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            history.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise CheckpointError(f"history.jsonl 第 {number} 行解析失败：{exc}") from exc
    try:
        manifest = CheckpointManifest(
            step=int(meta["step"]),
            params_sha256=str(meta["params_sha256"]),
            optimizer_sha256=str(meta["optimizer_sha256"]),
            files=tuple(str(name) for name in meta["files"]),
            params_count=int(meta["params_count"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointError(f"meta.json 字段不完整：{exc}") from exc
    return Checkpoint(
        step=manifest.step,
        params=tuple(float(value) for value in params_payload["params"]),
        optimizer_state=dict(optimizer_state),
        metrics=dict(metrics),
        history=tuple(history),
        manifest=manifest,
    )


def verify_checkpoint(directory: str | Path, manifest: CheckpointManifest) -> None:
    """对账：文件表齐全、且权重 / 优化器状态的哈希与清单一致（不一致抛 ``CheckpointError``）."""
    if not isinstance(manifest, CheckpointManifest):
        raise ParameterError(f"manifest 必须是 CheckpointManifest，收到 {type(manifest).__name__}。")
    target = Path(directory)
    if not target.is_dir():
        raise CheckpointError(f"检查点目录不存在：{target}")
    missing = [name for name in manifest.files if not (target / name).exists()]
    if missing:
        raise CheckpointError(
            f"检查点不完整，缺少 {', '.join(missing)}（目录：{target}）。"
        )
    params_payload = _read_json(target / "params.json")
    if not isinstance(params_payload, dict) or "params" not in params_payload:
        raise CheckpointError("params.json 缺少 params 字段。")
    weights = [float(value) for value in params_payload["params"]]
    if _sha256_of(weights) != manifest.params_sha256:
        raise CheckpointError(
            "params.json 的权重哈希与清单不一致：权重被谁改过，或这份检查点没写完。"
        )
    optimizer_payload_read = optimizer_payload(_read_json(target / "optimizer.json"))
    if _sha256_of(optimizer_payload_read) != manifest.optimizer_sha256:
        raise CheckpointError(
            "optimizer.json 的哈希与清单不一致：优化器状态被谁改过——"
            "用它恢复训练会得到一个'看着正常、其实动量不对'的起点。"
        )


def resume_step(checkpoint: Checkpoint) -> int:
    """从检查点读"恢复到第几步"（就是清单里的 ``step``）."""
    if not isinstance(checkpoint, Checkpoint):
        raise ParameterError(f"checkpoint 必须是 Checkpoint，收到 {type(checkpoint).__name__}。")
    return checkpoint.step


__all__ = [
    "CHECKPOINT_FILES",
    "OPTIMIZER_SCALAR_KEYS",
    "Checkpoint",
    "CheckpointManifest",
    "load_checkpoint",
    "optimizer_payload",
    "record_from_payload",
    "record_payload",
    "resume_step",
    "save_checkpoint",
    "verify_checkpoint",
]
