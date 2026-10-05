"""SFT 检查点：把"训练到底发生了什么"完整落盘（M5-D2）.

一次训练结束后，最有价值的产物不是权重文件，而是**一次可被审计的记录**：
用哪组超参、在哪个数据集上、每一步的 loss 与学习率、评估结果是多少。
没有这份记录，两天后回看 ``outputs/sft`` 里的一堆目录，没有人能说清
"这一份是哪次实验"。

所以 ``save_checkpoint`` 固定写五个文件：

| 文件 | 内容 | 为什么必须有 |
|------|------|-------------|
| ``training_args.json`` | 完整超参（含不属于 HF 的字段） | 复现这次训练的唯一依据 |
| ``tokenizer.json`` | 词表 | 没有词表，``input_ids`` 无法还原为文本 |
| ``model_state.json`` | 权重 + 偏置 + 已更新次数 | 可继续训练（resume）或做推理 |
| ``metrics.json`` | 汇总指标 + 派生量 plan | 一眼判断"这次训练有没有意义" |
| ``train_log.jsonl`` | 逐步的 ``StepRecord`` | 画 loss 曲线、定位发散的那一步 |

**词表必须随检查点落盘**：这与 day041 ``EmbeddingResponse`` 带
``provider`` + ``dimension`` 是同一种纪律——**产物要自描述**。一个只有
权重的目录，别人拿到手第一件事就是问"这是哪个分词器训的"。

写盘用 UTF-8 + ``ensure_ascii=False``：与 day048 ``dump_jsonl`` 同一口径，
中文样本落盘后要能被 review。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from smart_research_agent.sft.encoding import CharTokenizer
from smart_research_agent.sft.reference_model import ModelState

#: 检查点固定包含的文件名（顺序即写盘顺序，便于日志）
CHECKPOINT_FILES: tuple[str, ...] = (
    "training_args.json",
    "tokenizer.json",
    "model_state.json",
    "metrics.json",
    "train_log.jsonl",
)

#: ``metrics.json`` 里记录的本课快照标识：让产物能指回"哪一天的快照生成的"
SNAPSHOT = "day050"


class CheckpointError(RuntimeError):
    """检查点读写失败（目录缺失、文件损坏、字段不完整）."""


class _SerializableReport(Protocol):
    """只需提供 ``to_dict``：检查点层不依赖 trainer 的具体类型."""

    def to_dict(self) -> dict[str, Any]: ...  # pragma: no cover - 协议声明


class _SerializableRecord(Protocol):
    """逐步记录：同样只需要 ``to_dict``."""

    def to_dict(self) -> dict[str, Any]: ...  # pragma: no cover - 协议声明


class _HasState(Protocol):
    """模型协议：需要 ``state_dict`` 与 ``vocab_size``."""

    @property
    def vocab_size(self) -> int: ...  # pragma: no cover - 协议声明

    def state_dict(self) -> ModelState: ...  # pragma: no cover - 协议声明


class _HasTokenizerDict(Protocol):
    """分词器协议：需要 ``to_dict`` 与 ``vocab_size``."""

    @property
    def vocab_size(self) -> int: ...  # pragma: no cover - 协议声明

    def to_dict(self) -> dict[str, Any]: ...  # pragma: no cover - 协议声明


def _write_json(path: Path, payload: Any) -> Path:
    """写一个 JSON 文件（UTF-8、不转义中文、两空格缩进便于 diff）."""
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return path


def save_checkpoint(
    out_dir: str | Path,
    *,
    args: Any,
    model: _HasState,
    tokenizer: _HasTokenizerDict,
    report: _SerializableReport,
    records: Sequence[_SerializableRecord] = (),
) -> dict[str, Path]:
    """把一次训练完整落盘，返回 ``{文件名: 路径}``.

    返回路径字典而不是"成功"布尔值：调用方需要能直接打印/上传这些文件
    （与 day048 ``dump_bundle`` 同一种设计）。

    ``records`` 单独传而不是从 ``report`` 里取，是为了让调用方可以**只存
    一部分步**（例如只存 loss 异常的那几步）；``report`` 负责汇总、
    ``records`` 负责明细，两者的生命周期不同。
    """
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    paths["training_args.json"] = _write_json(target / "training_args.json", args.to_dict())
    paths["tokenizer.json"] = _write_json(target / "tokenizer.json", tokenizer.to_dict())
    paths["model_state.json"] = _write_json(
        target / "model_state.json", model.state_dict().to_dict()
    )
    metrics = report.to_dict()
    # 落盘时补上自描述信息：产物要能回答"我是谁生成的"
    metrics.setdefault("_meta", {})
    metrics["_meta"].update(
        {
            "snapshot": SNAPSHOT,
            "vocab_size": model.vocab_size,
            "tokenizer_vocab_size": tokenizer.vocab_size,
        }
    )
    paths["metrics.json"] = _write_json(target / "metrics.json", metrics)

    log_path = target / "train_log.jsonl"
    with log_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    paths["train_log.jsonl"] = log_path
    return paths


def load_checkpoint(out_dir: str | Path) -> dict[str, Any]:
    """读回检查点，返回 ``{args, tokenizer, state, metrics, records}``.

    缺任何一个必需文件都抛 ``CheckpointError``——**不返回"部分可用的对象"**。
    一个缺了 ``tokenizer.json`` 的检查点看着能用，实际会在推理时给出
    完全错误的文本（id 与字符错位），这比直接报错危险得多。
    """
    target = Path(out_dir)
    if not target.is_dir():
        raise CheckpointError(f"检查点目录不存在：{target}")
    missing = [name for name in CHECKPOINT_FILES if not (target / name).exists()]
    if missing:
        raise CheckpointError(
            f"检查点不完整，缺少 {', '.join(missing)}（目录：{target}）"
        )

    from smart_research_agent.sft.args import SFTTrainingArgs

    try:
        args_payload = json.loads((target / "training_args.json").read_text(encoding="utf-8"))
        tokenizer_payload = json.loads((target / "tokenizer.json").read_text(encoding="utf-8"))
        state_payload = json.loads((target / "model_state.json").read_text(encoding="utf-8"))
        metrics = json.loads((target / "metrics.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CheckpointError(f"检查点文件不是合法 JSON：{exc}") from exc

    records: list[dict[str, Any]] = []
    for lineno, line in enumerate(
        (target / "train_log.jsonl").read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            records.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise CheckpointError(f"train_log.jsonl 第 {lineno} 行解析失败：{exc}") from exc

    return {
        "args": SFTTrainingArgs.from_dict(args_payload),
        "tokenizer": CharTokenizer.from_dict(tokenizer_payload),
        "state": ModelState.from_dict(state_payload),
        "metrics": metrics,
        "records": records,
    }


def checkpoint_summary(out_dir: str | Path) -> str:
    """人类可读的一行摘要（demo 与日志直接打印）.

    只读 ``metrics.json``，不加载权重——运维巡检时不需要为看一眼结果
    而把整个检查点读进内存。
    """
    metrics_path = Path(out_dir) / "metrics.json"
    if not metrics_path.exists():
        raise CheckpointError(f"找不到 metrics.json：{out_dir}")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    return (
        f"检查点 {out_dir} | 步数 {payload.get('total_steps')} | "
        f"loss {payload.get('initial_loss')} -> {payload.get('final_loss')} | "
        f"评估 loss {payload.get('eval_loss')} | 监督 token {payload.get('supervised_tokens_seen')}"
    )
