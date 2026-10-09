"""合并与部署：把适配器变成"可以上线的模型"（M5-D4）.

LoRA 的部署形态有两种，选择的关键是"适配器要不要热插拔"：

.. code-block:: text

    形态 A：一个基座 + N 个适配器（推理时动态加载）
        优点：磁盘与显存都只存一份基座；换业务只换十几 MB
        代价：推理多两次小矩阵乘法（开销通常可忽略）；必须管好"哪个适配器配哪个基座"

    形态 B：合并成一个完整模型（merge_and_unload → save_pretrained）
        优点：部署链路与普通模型完全一致（不需要 peft 运行时）
        代价：每个业务一份完整权重；换适配器要重新合并

本模块把两种形态都做成可复核的产物：**适配器清单**（含内容哈希，回答
"线上跑的是哪一份"）、**合并验证报告**（回答"合并前后是不是同一个模型"）、
以及两份可直接运行的脚本（合并脚本、推理脚本）。

合并本身在 day051 已经实现（``LoRAReferenceModel.merge()``），今天的重点在
**验证**：合并是一个"会把小矩阵乘进大矩阵"的不可逆动作，一旦行/列约定用错
（day051 第三章），合并后的模型会与适配器模型给出不同的输出，而**它不会报错**。
所以 ``verify_merge`` 是部署流水线里的一个硬门禁——它把 day051 那条逐位一致
的不变式从单元测试搬到了交付流程里。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from smart_research_agent.peft.config import LoRAConfig, PEFTConfigError
from smart_research_agent.peft.models import LoRAReferenceModel
from smart_research_agent.peft.trainer import (
    ADAPTER_FILES,
    AdapterCheckpointError,
    load_adapter,
    save_adapter,
)
from smart_research_agent.sft.encoding import Batch
from smart_research_agent.sft.hf_script import DEFAULT_BASE_MODEL
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: 适配器清单的文件名（清单与适配器放在同一目录，随适配器一起被复制）.
MANIFEST_FILE = "adapter_manifest.json"


@dataclass(frozen=True)
class MergeVerification:
    """合并验证的结果（部署门禁）."""

    contexts_checked: int
    bitwise_identical: bool
    max_logit_difference: float
    adapter_loss: float
    merged_loss: float
    supervised_tokens: int
    tolerance: float

    @property
    def passed(self) -> bool:
        """是否通过门禁：要么逐位一致，要么差异在容差内."""
        return self.bitwise_identical or self.max_logit_difference <= self.tolerance

    @property
    def loss_difference(self) -> float:
        """两次评估的 loss 之差（合并后 - 适配器）."""
        return self.merged_loss - self.adapter_loss

    def to_dict(self) -> dict[str, Any]:
        """投影为可 json.dumps 的字典."""
        payload = asdict(self)
        payload["passed"] = self.passed
        payload["loss_difference"] = self.loss_difference
        return payload

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        return (
            f"检查 {self.contexts_checked} 个上下文 | 逐位一致 {self.bitwise_identical} | "
            f"最大 logits 差 {self.max_logit_difference:.3e} | "
            f"loss {self.adapter_loss:.10f} vs {self.merged_loss:.10f}"
            f"（差 {self.loss_difference:+.3e}）| 通过 {self.passed}"
        )


def verify_merge(
    model: LoRAReferenceModel,
    batches: Sequence[Batch],
    *,
    tolerance: float = 0.0,
) -> MergeVerification:
    """验证"合并后的模型"与"带适配器的模型"是同一个模型.

    ``tolerance=0.0``（缺省）表示要求**逐位一致**。这不是苛刻：两条路径做的
    是同一批浮点运算（``W[i][j] + ΔW[j][i]`` 一次算完 vs 每次前向现算一次），
    结果本就应当相同。本课实测差值是 ``0.00e+00``。

    如果容忍一个正数容差，说明调用方接受了某种数值差异（例如真实框架里
    合并会改变张量布局、触发不同的 kernel）。**但这个容差必须显式给出**——
    因为"合并错了"与"合并对了但有 1e-7 的浮点误差"在数值上难以区分，
    唯一的区别就是量级：约定错误带来的是 O(1) 的差异（实测 0.0381），
    浮点重排带来的是 O(1e-7)。
    """
    if tolerance < 0:
        raise PEFTConfigError(f"tolerance 不能为负数，收到 {tolerance}")
    merged = model.merge()
    max_difference = 0.0
    identical = True
    for context in range(model.vocab_size):
        left = merged.logits(context)
        right = model.logits(context)
        for a, b in zip(left, right):
            if a == b:
                continue
            identical = False
            max_difference = max(max_difference, abs(a - b))
    adapter_loss, tokens = model.evaluate(batches)
    merged_loss, _ = merged.evaluate(batches)
    verification = MergeVerification(
        contexts_checked=model.vocab_size,
        bitwise_identical=identical,
        max_logit_difference=max_difference,
        adapter_loss=adapter_loss,
        merged_loss=merged_loss,
        supervised_tokens=tokens,
        tolerance=tolerance,
    )
    logger.info("合并验证：%s", verification.summary_line())
    return verification


@dataclass
class AdapterManifest:
    """适配器清单：回答"这是哪一份适配器、配哪个基座、怎么复现"."""

    adapter_dir: str
    base_model: str
    lora: dict[str, Any]
    trainable_parameters: int
    adapter_bytes: int
    content_sha256: str
    step: int
    train_loss: float | None
    learning_rate: float | None
    merged: bool = False
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """投影为可 json.dumps 的字典."""
        payload = asdict(self)
        payload["short_hash"] = self.content_sha256[:12]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AdapterManifest:
        """从 ``to_dict`` 的产物还原（未知键忽略）."""
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in payload.items() if key in known})

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        # ``LoRAConfig.to_dict()`` 里的键是 ``resolved_targets``（展开后的模块名），
        # 不是 ``targets``——本课实现时这里一度打印出 ``targets=None``，
        # **一个只出现在日志里的字段永远没人核对**，所以两个键都试一遍并回退到原始写法。
        targets = self.lora.get("resolved_targets") or self.lora.get("target_modules")
        return (
            f"{Path(self.adapter_dir).name} | 基座 {self.base_model} | "
            f"r={self.lora.get('r')} alpha={self.lora.get('lora_alpha')} "
            f"targets={targets} | step {self.step} | "
            f"{self.adapter_bytes} B | sha256:{self.content_sha256[:12]} | 已合并 {self.merged}"
        )


def build_manifest(
    adapter_dir: str | Path,
    *,
    base_model: str = DEFAULT_BASE_MODEL,
    merged: bool = False,
    tags: dict[str, str] | None = None,
) -> AdapterManifest:
    """读 adapter 目录，产出一份清单（不加载权重，只读 JSON）.

    清单存在的理由很具体：一个只装了 ``A`` / ``B`` 的目录，别人拿到手第一件
    要问的就是"这是哪个基座、哪个 ``r``、训练到第几步"。**产物要自描述**——
    这与 day050 把词表随检查点落盘是同一条纪律。
    """
    payload = load_adapter(adapter_dir)
    state = payload["state"]
    reference = state.get("reference_model", {})
    return AdapterManifest(
        adapter_dir=str(adapter_dir),
        base_model=base_model,
        lora=payload["config"],
        trainable_parameters=int(reference.get("trainable_parameters", 0)),
        adapter_bytes=sum((Path(adapter_dir) / name).stat().st_size for name in ADAPTER_FILES),
        content_sha256=payload["sha256"],
        step=int(state.get("step", 0)),
        train_loss=state.get("train_loss"),
        learning_rate=state.get("learning_rate"),
        merged=merged,
        tags=dict(tags or {}),
    )


def write_manifest(directory: str | Path, manifest: AdapterManifest) -> Path:
    """把清单写进适配器目录（同目录，随适配器一起被复制）."""
    target = Path(directory) / MANIFEST_FILE
    target.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def read_manifest(directory: str | Path) -> AdapterManifest:
    """读回清单；文件缺失抛 ``AdapterCheckpointError``."""
    target = Path(directory) / MANIFEST_FILE
    if not target.exists():
        raise AdapterCheckpointError(f"找不到适配器清单：{target}")
    return AdapterManifest.from_dict(json.loads(target.read_text(encoding="utf-8")))


def deployment_report(
    *,
    spec_parameters: int,
    manifest: AdapterManifest,
    base_bits_per_parameter: float = 16.0,
) -> dict[str, Any]:
    """两种部署形态的体积对照（一个基座 + N 个适配器 / 合并模型）.

    三个数字放在一起才有意义：

    - **基座**：``spec_parameters × base_bits_per_parameter / 8``——7B 上
      16-bit 是 13.48 GB；
    - **适配器**：几 MB 到十几 MB；
    - **合并模型**：与基座同量级（它就是一个完整模型）。

    于是"一个基座 + 10 个适配器"与"10 个合并模型"的差别是
    ``1 份基座 + 10 份适配器`` vs ``10 份基座``——**相差接近 10 倍**。
    """
    if spec_parameters <= 0:
        raise PEFTConfigError(f"spec_parameters 必须为正整数，收到 {spec_parameters}")
    if base_bits_per_parameter <= 0:
        raise PEFTConfigError(f"base_bits_per_parameter 必须为正数，收到 {base_bits_per_parameter}")
    base_bytes = int(spec_parameters * base_bits_per_parameter / 8)
    multi_adapter_bytes = base_bytes + manifest.adapter_bytes
    ten_adapters_bytes = base_bytes + manifest.adapter_bytes * 10
    return {
        "base_model_bytes": base_bytes,
        "base_model_gib": round(base_bytes / 1024**3, 4),
        "adapter_bytes": manifest.adapter_bytes,
        "adapter_mebibytes": round(manifest.adapter_bytes / 1024**2, 4),
        "one_base_plus_one_adapter_bytes": multi_adapter_bytes,
        "merged_model_bytes": base_bytes,
        "adapter_ratio": round(manifest.adapter_bytes / base_bytes, 8),
        "ten_merged_models_bytes": base_bytes * 10,
        "one_base_plus_ten_adapters_bytes": ten_adapters_bytes,
        "multi_adapter_saving_factor": round(base_bytes * 10 / ten_adapters_bytes, 6),
    }


def adapter_registry(manifests: Sequence[AdapterManifest]) -> list[dict[str, Any]]:
    """把 N 份清单整理成一张注册表（"一个基座 + N 个适配器"的运维视图）.

    输出里刻意保留 ``base_model`` 与 ``content_sha256``：前者用于回答
    "这个适配器能不能挂到我的基座上"（**基座不匹配的适配器不会报错，
    只会给出莫名其妙的结果**），后者用于回答"线上到底跑的哪一份"。
    """
    rows: list[dict[str, Any]] = []
    for manifest in manifests:
        rows.append(
            {
                "adapter": Path(manifest.adapter_dir).name,
                "base_model": manifest.base_model,
                "step": manifest.step,
                "train_loss": manifest.train_loss,
                "trainable_parameters": manifest.trainable_parameters,
                "adapter_bytes": manifest.adapter_bytes,
                "short_hash": manifest.content_sha256[:12],
                "merged": manifest.merged,
                "tags": dict(manifest.tags),
            }
        )
    return sorted(rows, key=lambda row: (row["adapter"], row["step"]))


def render_merge_script(
    lora: LoRAConfig,
    *,
    base_model: str = DEFAULT_BASE_MODEL,
    output_dir: str = "outputs/lora-merged",
) -> str:
    """生成"合并并保存"的脚本（``peft`` 的 ``merge_and_unload``）.

    三行核心代码，但每一步都有要求：

    .. code-block:: python

        model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, adapter_dir)     # 先挂上适配器
        model = model.merge_and_unload()                          # 再合并（不可逆）
        model.save_pretrained(output_dir)                         # 落盘成普通模型

    ``merge_and_unload`` 会**就地替换**权重并丢弃适配器，因此它必须发生在
    "所有需要适配器的动作之后"。脚本里在合并前打印适配器的内容哈希——
    **合并是一个不可逆动作，落盘的模型必须能指回它是由哪一份适配器产生的**。
    """
    lora.validate()
    payload = json.dumps(lora.to_peft_dict(), ensure_ascii=False, indent=2)
    return f'''#!/usr/bin/env python
"""把 LoRA 适配器合并成完整模型——由 smart_research_agent.peft.deploy 生成.

安装依赖::

    pip install "transformers>=5.16" "torch>=2.5" "peft>=0.20"

运行::

    python merge_adapter.py --base_model {base_model} \\
        --adapter_dir outputs/lora/adapters/adapter-final \\
        --output_dir {output_dir}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

#: 与训练时使用的配置逐字对齐（由 LoRAConfig.to_peft_dict() 生成）
LORA_CONFIG = json.loads(
    """{payload}"""
)


def adapter_fingerprint(adapter_dir: str) -> str:
    """打印适配器目录里的内容哈希（有清单时读清单，没有就算一次）."""
    manifest = Path(adapter_dir) / "adapter_manifest.json"
    if manifest.exists():
        return json.loads(manifest.read_text(encoding="utf-8"))["content_sha256"][:12]
    return "未知（目录里没有 adapter_manifest.json）"


def main() -> None:
    parser = argparse.ArgumentParser(description="合并 LoRA 适配器")
    parser.add_argument("--base_model", default="{base_model}")
    parser.add_argument("--adapter_dir", default="outputs/lora/adapters/adapter-final")
    parser.add_argument("--output_dir", default="{output_dir}")
    args = parser.parse_args()

    print(f"基座 {{args.base_model}} + 适配器 {{args.adapter_dir}}"
          f"（sha256:{{adapter_fingerprint(args.adapter_dir)}}）")
    print(f"适配器配置：r={{LORA_CONFIG['r']}} alpha={{LORA_CONFIG['lora_alpha']}} "
          f"targets={{LORA_CONFIG['target_modules']}}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    # 合并是就地、不可逆的：之后的模型不再需要 peft 运行时
    model = model.merge_and_unload()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"合并完成：{{args.output_dir}}（与基座同量级的完整模型）")


if __name__ == "__main__":
    main()
'''


def render_inference_script(
    *,
    base_model: str = DEFAULT_BASE_MODEL,
    adapter_dir: str = "outputs/lora/adapters/adapter-final",
    system_prompt: str = "你是智研 AI 助手，一个善于检索与推理的研究助理。",
) -> str:
    """生成推理脚本：**适配器形态**（一个基座 + N 个适配器）.

    与合并形态的差别只有一处：加载时多一行 ``PeftModel.from_pretrained``，
    而基座只有一份。脚本同时打印当前挂载的适配器哈希——多适配器场景下
    "我这次加载的是哪一个"必须能从日志里看出来。
    """
    return f'''#!/usr/bin/env python
"""用 LoRA 适配器做推理（不合并）——由 smart_research_agent.peft.deploy 生成.

一个基座 + N 个适配器：换业务只换 ``--adapter_dir``，基座只加载一次。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = """{system_prompt}"""


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA 适配器推理")
    parser.add_argument("--base_model", default="{base_model}")
    parser.add_argument("--adapter_dir", default="{adapter_dir}")
    parser.add_argument("--prompt", default="什么是 LoRA？")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    manifest_path = Path(args.adapter_dir) / "adapter_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(f"当前适配器：{{manifest['adapter_dir']}} "
              f"sha256:{{manifest['content_sha256'][:12]}} step={{manifest['step']}}")
        if manifest["base_model"] != args.base_model:
            print(f"[警告] 清单记录的基座是 {{manifest['base_model']}}，"
                  f"而本次加载的是 {{args.base_model}}——基座不匹配不会报错，"
                  "只会给出莫名其妙的结果")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()

    messages = [
        {{"role": "system", "content": SYSTEM_PROMPT}},
        {{"role": "user", "content": args.prompt}},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        output = model.generate(inputs, max_new_tokens=args.max_new_tokens)
    print(tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
'''


def repackage_adapter(
    source_dir: str | Path,
    target_dir: str | Path,
    *,
    base_model: str = DEFAULT_BASE_MODEL,
    tags: dict[str, str] | None = None,
) -> AdapterManifest:
    """把一份适配器**连同清单**复制到新目录（发布动作）.

    复制的是三个文件而不是"目录整体"，因此清单是**重新生成**的（而不是原样
    拷贝）：这样"发布出来的那一份"与"实验目录里的那一份"各自带着自己的
    哈希与元信息，发布目录里不会留下训练过程中的临时文件。
    """
    payload = load_adapter(source_dir)
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    for name in ("adapter_config.json", "adapter_model.json", "training_state.json"):
        (target / name).write_text(
            (Path(source_dir) / name).read_text(encoding="utf-8"), encoding="utf-8"
        )
    manifest = AdapterManifest(
        adapter_dir=str(target),
        base_model=base_model,
        lora=payload["config"],
        trainable_parameters=int(
            payload["state"].get("reference_model", {}).get("trainable_parameters", 0)
        ),
        adapter_bytes=sum((target / name).stat().st_size for name in ADAPTER_FILES),
        content_sha256=payload["sha256"],
        step=int(payload["state"].get("step", 0)),
        train_loss=payload["state"].get("train_loss"),
        learning_rate=payload["state"].get("learning_rate"),
        merged=False,
        tags=dict(tags or {}),
    )
    write_manifest(target, manifest)
    return manifest


__all__ = [
    "MANIFEST_FILE",
    "AdapterManifest",
    "MergeVerification",
    "adapter_registry",
    "build_manifest",
    "deployment_report",
    "read_manifest",
    "render_inference_script",
    "render_merge_script",
    "repackage_adapter",
    "save_adapter",
    "verify_merge",
    "write_manifest",
]
