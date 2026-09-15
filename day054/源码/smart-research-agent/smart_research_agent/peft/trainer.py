"""LoRA 训练器与适配器检查点：把"能训"变成"能交付"（M5-D4）.

day051 证明了一件事：``LoRAReferenceModel`` 与 ``ReferenceSFTModel`` 接口一致，
因此 day050 的 ``SFTTrainer`` 一行不改就能训 LoRA。这一天要在它之上补上
**适配器的生命周期**——而这件事恰好落在训练循环的内部：

.. code-block:: text

    全参微调：每 N 步存一份完整权重（7B 是十几 GB）
    LoRA    ：每 N 步存一份适配器（同一模型上只有十几 MB）

两者要存的东西差三个数量级，所以"怎么存、存几份、怎么恢复"是三件不能靠
"照抄全参流程"解决的事。本模块把它们做成三个明确的动作：

1. **``save_adapter``**：写三个文件（配置、权重、训练状态），并给出一份
   **内容哈希**——"线上跑的是哪个适配器"必须能被回答；
2. **``load_adapter``**：缺文件就报错，不做"部分可用"的降级（与 day050
   ``load_checkpoint`` 同一条纪律）；
3. **``prune_adapters``**：按 ``save_total_limit`` 清理最旧的中间适配器，
   ``adapter-final`` 永不删除。

循环本身**不复制**：day052 给 ``SFTTrainer`` 加了一个 ``on_step`` 回调，
``LoRATrainer`` 用它把"落盘"插进循环。这样训练逻辑仍然只有一份，而
"每步做什么额外动作"由调用方决定——测试里有一条用例守着"回调拿到的
step 与 records 与报告完全一致"。

``LoRATrainingReport`` 比 day050 的 ``SFTReport`` 多三块：适配器规模、适配器
检查点清单、以及**与全参口径的对照**（同一个模型的 LoRA 与全参各要落多少盘）。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from smart_research_agent.peft.config import LoRAConfig, PEFTConfigError
from smart_research_agent.peft.models import LoRAReferenceModel, reference_lora_accounting
from smart_research_agent.sft.args import SFTTrainingArgs
from smart_research_agent.sft.encoding import CharTokenizer
from smart_research_agent.sft.trainer import SFTReport, SFTTrainer, StepRecord
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: 一个适配器目录固定包含的三个文件（顺序即写盘顺序）.
#:
#: 为什么是三个而不是一个：``adapter_config.json`` 是可复现的前提（``r`` /
#: ``alpha`` / 目标模块），``adapter_model.json`` 是权重，``training_state.json``
#: 是"训到哪了"。**缺任何一个都不该被加载**——缺配置会导致用错缩放、缺状态会
#: 导致续训从第 0 步开始（把学习率调度重置，损失曲线从此不可解释）。
ADAPTER_FILES: tuple[str, ...] = (
    "adapter_config.json",
    "adapter_model.json",
    "training_state.json",
)

#: 末次适配器目录名（永不删除）；中间适配器的命名与 HF 习惯一致
FINAL_ADAPTER_NAME = "adapter-final"
INTERMEDIATE_PREFIX = "adapter-step-"


class AdapterCheckpointError(RuntimeError):
    """适配器检查点读写失败（目录缺失、文件不全、JSON 损坏）."""


@dataclass(frozen=True)
class AdapterCheckpoint:
    """一次适配器落盘的记录."""

    directory: str
    boundary: str
    step: int
    files: tuple[str, ...]
    adapter_bytes: int
    trainable_parameters: int
    content_sha256: str
    train_loss: float | None
    learning_rate: float | None

    @property
    def short_hash(self) -> str:
        """哈希前 12 位——用于日志与清单展示（完整哈希留在 ``content_sha256``）."""
        return self.content_sha256[:12]

    def to_dict(self) -> dict[str, Any]:
        """投影为可 json.dumps 的字典."""
        payload = asdict(self)
        payload["files"] = list(self.files)
        payload["short_hash"] = self.short_hash
        return payload

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        return (
            f"{self.boundary} step={self.step} | {self.adapter_bytes} B"
            f"（{self.adapter_bytes / 1024:.1f} KiB）| 可训练 {self.trainable_parameters} | "
            f"sha256:{self.short_hash}"
        )


def _canonical_bytes(payload: Any) -> bytes:
    """把字典序列化成**规范化**的字节（排序键 + 紧凑分隔符）.

    "内容哈希"只有在序列化规范固定之后才有意义：同一份数据换了键顺序就会
    得到不同的哈希，那么"线上跑的是哪个适配器"这个问题就仍然答不出来。
    所以这里显式 `sort_keys=True` + `separators=(",", ":")`。
    """
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def adapter_content_hash(*, config: dict, model: dict) -> str:
    """适配器内容的 sha256（只覆盖配置与权重，不含训练状态）.

    刻意把 ``training_state.json`` 排除在哈希之外：同一个适配器续训一步之后
    权重没变、但"训到哪了"变了。**哈希要回答的是"权重是不是同一份"**，
    把时间戳/步数混进去会让这个问题的答案随无关量漂移。
    """
    digest = hashlib.sha256()
    digest.update(_canonical_bytes({"config": config}))
    digest.update(_canonical_bytes({"model": model}))
    return digest.hexdigest()


def save_adapter(
    out_dir: str | Path,
    *,
    model: LoRAReferenceModel,
    args: SFTTrainingArgs,
    step: int,
    boundary: str,
    records: Sequence[StepRecord] = (),
) -> AdapterCheckpoint:
    """把一个 LoRA 适配器完整落盘，返回检查点记录.

    写三个文件，并返回路径级信息与内容哈希：

    ==========================  ================================================
    ``adapter_config.json``     ``LoRAConfig.to_dict()`` + 基座参数量与缩放公式
    ``adapter_model.json``      ``A`` / ``B`` 两个矩阵（**只有它们**）
    ``training_state.json``     步数、学习率、loss、超参快照、词表摘要
    ==========================  ================================================

    三个文件加起来的体积就是"这次微调需要交付的东西"——在参考模型上是几十 KiB，
    在 7B 上是十几 MB，而**同一个模型的基座是十几 GB**。
    """
    if step < 0:
        raise PEFTConfigError(f"step 不能为负数，收到 {step}")
    if not boundary.strip():
        raise PEFTConfigError("boundary 不能为空（final / step-N 之类）")
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)

    adapter_state = model.adapter_state_dict()
    config_payload = adapter_state["config"]
    model_payload = {
        "a": adapter_state["a"],
        "b": adapter_state["b"],
        "updates": adapter_state["updates"],
        "trainable_parameters": model.trainable_parameters,
        "frozen_parameters": model.frozen_parameters,
    }
    last = records[-1] if records else None
    state_payload = {
        "step": step,
        "boundary": boundary,
        "base_updates": adapter_state["base_updates"],
        "adapter_updates": adapter_state["updates"],
        "learning_rate": last.learning_rate if last else args.learning_rate,
        "train_loss": last.loss if last else None,
        "training_arguments": args.to_dict(),
        "vocab_size": model.vocab_size,
        "record_count": len(records),
        "reference_model": reference_lora_accounting(model.vocab_size, model.config),
    }

    (target / "adapter_config.json").write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (target / "adapter_model.json").write_text(
        json.dumps(model_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (target / "training_state.json").write_text(
        json.dumps(state_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    checkpoint = AdapterCheckpoint(
        directory=str(target),
        boundary=boundary,
        step=step,
        files=ADAPTER_FILES,
        adapter_bytes=sum((target / name).stat().st_size for name in ADAPTER_FILES),
        trainable_parameters=model.trainable_parameters,
        content_sha256=adapter_content_hash(config=config_payload, model=model_payload),
        train_loss=last.loss if last else None,
        learning_rate=last.learning_rate if last else args.learning_rate,
    )
    logger.info("已保存适配器 %s（%s）", target, checkpoint.summary_line())
    return checkpoint


def load_adapter(out_dir: str | Path) -> dict[str, Any]:
    """读回一个适配器，返回 ``{config, model, state, sha256, directory}``.

    缺任何一个必需文件都抛 ``AdapterCheckpointError``——与 day050 ``load_checkpoint``
    一样，**不返回"部分可用的对象"**。一个缺了 ``adapter_config.json`` 的目录
    看着能用，实际会在加载时用上默认的 ``alpha``，于是缩放悄悄变了一个倍数。
    """
    target = Path(out_dir)
    if not target.is_dir():
        raise AdapterCheckpointError(f"适配器目录不存在：{target}")
    missing = [name for name in ADAPTER_FILES if not (target / name).exists()]
    if missing:
        raise AdapterCheckpointError(
            f"适配器不完整，缺少 {', '.join(missing)}（目录：{target}）"
        )
    try:
        config_payload = json.loads((target / "adapter_config.json").read_text(encoding="utf-8"))
        model_payload = json.loads((target / "adapter_model.json").read_text(encoding="utf-8"))
        state_payload = json.loads((target / "training_state.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AdapterCheckpointError(f"适配器文件不是合法 JSON：{exc}") from exc
    return {
        "config": config_payload,
        "model": model_payload,
        "state": state_payload,
        "sha256": adapter_content_hash(config=config_payload, model=model_payload),
        "directory": str(target),
    }


def adapter_summary(out_dir: str | Path) -> str:
    """只读 ``training_state.json`` 的一句话摘要（巡检不需要加载权重）."""
    state_path = Path(out_dir) / "training_state.json"
    if not state_path.exists():
        raise AdapterCheckpointError(f"找不到 training_state.json：{out_dir}")
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    return (
        f"适配器 {out_dir} | {payload.get('boundary')} step={payload.get('step')} | "
        f"可训练 {payload.get('reference_model', {}).get('trainable_parameters')} | "
        f"loss {payload.get('train_loss')} | lr {payload.get('learning_rate')}"
    )


def prune_adapters(output_dir: str | Path, *, limit: int) -> list[str]:
    """按 ``limit`` 清理最旧的中间适配器，返回被删除的目录名.

    ``adapter-final`` **永不删除**：它是这次实验的交付物。中间目录按步号
    从小到大排序，保留最后 ``limit`` 个。与 day050 ``_prune_checkpoints``
    是同一条纪律——**磁盘不是无限的，而"训练跑到一半磁盘满"是最令人沮丧的
    失败方式**；差别只是这里要清的是适配器，量级小得多，因此删除动作更安全。
    """
    if limit <= 0:
        raise PEFTConfigError(f"limit 必须为正整数，收到 {limit}")
    root = Path(output_dir)
    if not root.is_dir():
        return []
    intermediates = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir() and path.name.startswith(INTERMEDIATE_PREFIX)
        ),
        key=lambda path: int(path.name[len(INTERMEDIATE_PREFIX) :]),
    )
    removed: list[str] = []
    for stale in intermediates[: max(0, len(intermediates) - limit)]:
        for child in stale.iterdir():
            child.unlink()
        stale.rmdir()
        removed.append(stale.name)
    return removed


@dataclass
class LoRATrainingReport:
    """一次 LoRA 训练的完整汇总（比 ``SFTReport`` 多三块）."""

    lora: dict[str, Any]
    reference: dict[str, Any]
    sft: dict[str, Any]
    adapter_checkpoints: list[AdapterCheckpoint] = field(default_factory=list)
    adapter_bytes_final: int = 0
    full_finetune_bytes: int = 0
    adapter_save_steps: int = 0

    @property
    def adapter_saving_factor(self) -> float:
        """相对全参微调的落盘节省倍数（适配器体积 vs 完整权重）."""
        if self.adapter_bytes_final <= 0:
            return 0.0
        return self.full_finetune_bytes / self.adapter_bytes_final

    def to_dict(self) -> dict[str, Any]:
        """投影为可 json.dumps 的字典."""
        return {
            "lora": self.lora,
            "reference": self.reference,
            "sft": self.sft,
            "adapter_checkpoints": [item.to_dict() for item in self.adapter_checkpoints],
            "adapter_bytes_final": self.adapter_bytes_final,
            "full_finetune_bytes": self.full_finetune_bytes,
            "adapter_saving_factor": round(self.adapter_saving_factor, 4),
            "adapter_save_steps": self.adapter_save_steps,
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        return (
            f"可训练 {self.reference.get('trainable_parameters')} "
            f"（{self.reference.get('trainable_ratio', 0.0):.4%}）| "
            f"最终适配器 {self.adapter_bytes_final} B vs 全参 "
            f"{self.full_finetune_bytes} B（省 {self.adapter_saving_factor:.1f}×）| "
            f"落盘 {len(self.adapter_checkpoints)} 次"
        )


#: ``LoRATrainer`` 里用来区分"没传这个参数"与"显式传了 None"的哨兵.
#:
#: 为什么需要它：day050 的 ``system_prompt=None`` 有明确语义——**不要 system 轮**，
#: 与"不传（用 ``DEFAULT_SYSTEM_PROMPT``）"是两件事。如果本类把 ``None`` 与
#: "没传"混为一谈，调用方传 ``None`` 想训"无 system"的数据时会被静默换成
#: 默认提示词——**训练数据与推理输入不一致，而脚本不会报任何错**（day050 第
#: 2.1 节把这类缺陷列在最隐蔽的一类和）。本课实现时踩到过，所以这里用哨兵区分。
_UNSET: Any = object()


class LoRATrainer:
    """在 day050 的 ``SFTTrainer`` 之上补上适配器生命周期（训练循环仍然只有一份）.

    用法::

        trainer = LoRATrainer(model, tokenizer, args, adapter_save_steps=10)
        report = trainer.fit(train_examples, eval_examples)

    ``adapter_save_steps=0``（缺省）表示只在训练结束时落盘一次；给正数则沿途
    按间隔落盘，并按 ``save_total_limit`` 清理最旧的中间副本。

    **为什么不是把 ``SFTTrainer`` 的循环复制一份**：循环里每一步都有 day050
    辛苦钉住的口径（累积分母、warmup 第 0 步、drop_last 的步数算术）。复制
    一份的代价不是那 60 行代码，而是**那些口径会各自漂移**。所以这里只提供
    一个回调，把"额外动作"插进唯一的那个循环。
    """

    def __init__(
        self,
        model: LoRAReferenceModel,
        tokenizer: CharTokenizer,
        args: SFTTrainingArgs,
        *,
        adapter_save_steps: int = 0,
        adapter_output_dir: str | None = None,
        save_total_limit: int | None = None,
        template: Any = _UNSET,
        system_prompt: Any = _UNSET,
        mask_prompt: bool = True,
    ):
        if adapter_save_steps < 0:
            raise PEFTConfigError(f"adapter_save_steps 不能为负数，收到 {adapter_save_steps}")
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.adapter_save_steps = adapter_save_steps
        self.adapter_output_dir = adapter_output_dir or f"{args.output_dir}/adapters"
        self.save_total_limit = (
            save_total_limit if save_total_limit is not None else args.save_total_limit
        )
        # 沿用 SFTTrainer 的两条缺省（模板与 system 提示词）——刻意不在本类里
        # 重新定义，避免"两处缺省不一致"。**显式传 None 必须被转发**（它的语义是
        # "不要 system 轮"，与"没传"不同），所以用哨兵而不是 ``is not None`` 判断。
        sft_defaults: dict[str, Any] = {"mask_prompt": mask_prompt}
        if template is not _UNSET:
            sft_defaults["template"] = template
        if system_prompt is not _UNSET:
            sft_defaults["system_prompt"] = system_prompt
        self._sft_defaults = sft_defaults

    # ------------------------------------------------------------------ 落盘
    def _save(self, step: int, boundary: str, records: Sequence[StepRecord]) -> AdapterCheckpoint:
        """落一个适配器，并按 limit 清理旧的中间副本."""
        checkpoint = save_adapter(
            Path(self.adapter_output_dir) / boundary,
            model=self.model,
            args=self.args,
            step=step,
            boundary=boundary,
            records=records,
        )
        prune_adapters(self.adapter_output_dir, limit=self.save_total_limit)
        return checkpoint

    def restore(self, directory: str | Path) -> dict[str, Any]:
        """断点续训的加载动作：把适配器状态灌回模型.

        返回被加载的适配器元信息（含内容哈希）。**不加载基座权重**——LoRA
        的续训只需要适配器，基座由调用方按同样的 ``model_name`` 重新载入。
        这也是一处必须说清的语义差异：全参微调的续训是"加载整份权重"，
        LoRA 的续训是"加载一个小矩阵"，两者失败的后果完全不同（后者错了
        只会得到一份错的增量，不会毁掉基座）。
        """
        payload = load_adapter(directory)
        config = LoRAConfig.from_dict(payload["config"])
        if config.r != self.model.config.r or config.lora_alpha != self.model.config.lora_alpha:
            raise AdapterCheckpointError(
                "适配器配置与当前模型不一致："
                f"r={config.r}/alpha={config.lora_alpha} vs "
                f"r={self.model.config.r}/alpha={self.model.config.lora_alpha}"
            )
        self.model.load_adapter_state(
            {
                "a": payload["model"]["a"],
                "b": payload["model"]["b"],
                "updates": payload["model"].get("updates", 0),
                "scaling": config.scaling,
            }
        )
        return payload

    # ------------------------------------------------------------------ 训练
    def fit(
        self,
        train_examples: Sequence[Any],
        eval_examples: Sequence[Any] = (),
        *,
        save_at_end: bool = True,
    ) -> LoRATrainingReport:
        """跑完一次 LoRA 训练并返回含适配器清单的报告.

        六步顺序与 day050 一致（校验 → 编码 → 计划 → 切批 → 循环 → 收尾），
        只是循环里多了一个"按间隔落适配器"的回调。
        """
        checkpoints: list[AdapterCheckpoint] = []

        def on_step(step: int, model: Any, records: Sequence[StepRecord]) -> None:
            del model  # 回调拿到的模型就是 self.model；保留参数是为了签名通用
            if self.adapter_save_steps and step % self.adapter_save_steps == 0:
                checkpoints.append(
                    self._save(step, f"{INTERMEDIATE_PREFIX}{step}", records)
                )

        trainer = SFTTrainer(
            self.model, self.tokenizer, self.args, on_step=on_step, **self._sft_defaults
        )
        report: SFTReport = trainer.fit(train_examples, eval_examples, save_at_end=save_at_end)
        if save_at_end:
            checkpoints.append(
                self._save(report.optimizer_steps, FINAL_ADAPTER_NAME, report.records)
            )

        reference = reference_lora_accounting(self.model.vocab_size, self.model.config)
        final = next(
            (item for item in checkpoints if item.boundary == FINAL_ADAPTER_NAME), None
        )
        # 全参微调的落盘体积：词表大小 × 词表大小（权重）+ 词表（偏置），按
        # 4 字节算——参考模型的"完整权重"就是这个量级。用一个显式公式而不是
        # 常量，是为了让这条对照随模型规模自动成立。
        full_bytes = (self.model.vocab_size**2 + self.model.vocab_size) * 4
        return LoRATrainingReport(
            lora=self.model.config.to_dict(),
            reference=reference,
            sft=report.to_dict(),
            adapter_checkpoints=checkpoints,
            adapter_bytes_final=final.adapter_bytes if final else 0,
            full_finetune_bytes=full_bytes,
            adapter_save_steps=self.adapter_save_steps,
        )


__all__ = [
    "ADAPTER_FILES",
    "FINAL_ADAPTER_NAME",
    "INTERMEDIATE_PREFIX",
    "AdapterCheckpoint",
    "AdapterCheckpointError",
    "LoRATrainer",
    "LoRATrainingReport",
    "adapter_content_hash",
    "adapter_summary",
    "load_adapter",
    "prune_adapters",
    "save_adapter",
]
