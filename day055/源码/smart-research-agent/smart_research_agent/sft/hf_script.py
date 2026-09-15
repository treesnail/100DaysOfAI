"""生成可直接运行的 SFT 训练脚本（``transformers`` + ``Trainer`` / TRL）（M5-D2）.

本课的训练循环用纯 Python 实现（``trainer.py``），好处是每一步都能被手算
验证、离线可跑、零依赖。但**真实训练必须交给真实框架**：混合精度、
梯度检查点、多卡 DDP/DeepSpeed、FlashAttention，这些都不是本课要重造
的轮子。

所以本模块负责"把两者的接缝写清楚"：``render_hf_sft_script`` 生成一份
**完整、语法正确、参数与本课 ``SFTTrainingArgs`` 一一对应**的训练脚本。
生成而不是手写的好处与 day048 ``render_methods_table`` 相同——**参数值
只有一个来源**（``SFTTrainingArgs.to_hf_dict()``），不会出现"教程里的
参数和脚本里的不一致"。

脚本里的三个关键实现点，正是本课的核心机制在真实框架里的写法：

1. **前缀与答案分别 tokenize 再拼接**（``prompt_ids[-budget:] + answer_ids``），
   而不是编码整段再按字符切片——BPE 的合并不认字符边界；
2. **答案放不下的样本返回 ``None`` 被丢弃**，而不是把答案截断。截断答案
   等于把"说了一半就停"教给模型，是本课最不能接受的脏数据；
3. **labels 的 padding 补 ``-100``**，与 ``input_ids`` 补 ``pad_token_id``、
   ``attention_mask`` 补 ``0`` 三件套一起做。少任何一个都会出错。

依赖版本（本课核对于 2026-09）：``transformers>=5.16``、``datasets>=3.0``、
``accelerate>=1.0``、``trl>=1.5``、``peft>=0.20``。
"""

from __future__ import annotations

import json
from typing import Any

from smart_research_agent.sft.args import SFTTrainingArgs

#: 生成脚本运行所需的第三方依赖（与 pyproject 的 optional-dependencies 分开：
#: 训练依赖体积大且需要 GPU，不应装进日常测试环境）
HF_DEPENDENCIES: tuple[str, ...] = (
    "transformers>=5.16",
    "datasets>=3.0",
    "accelerate>=1.0",
    "torch>=2.5",
)

#: 走 TRL 路径时额外需要的依赖
TRL_DEPENDENCIES: tuple[str, ...] = (*HF_DEPENDENCIES, "trl>=1.5")

#: 默认基座模型：0.6B 级别的指令模型，单卡即可 SFT，适合作为第一次跑通的目标
DEFAULT_BASE_MODEL = "Qwen/Qwen3-0.6B"

#: 生成脚本里用的 system 提示词（必须与本课 ``DEFAULT_SYSTEM_PROMPT`` 一致，
#: 否则"训练时看到的 prompt"与"推理时给的 prompt"不同，属于最隐蔽的缺陷之一）
DEFAULT_SYSTEM_IN_SCRIPT = (
    "你是智研 AI 助手，一个善于检索与推理的研究助理。"
    "回答要准确、具体，涉及步骤时按条列出。"
)


_HF_SCRIPT = '''#!/usr/bin/env python
"""SFT 训练脚本（transformers + Trainer）——由 smart_research_agent.sft.hf_script 生成.

安装依赖::

    pip install "transformers>=5.16" "datasets>=3.0" "accelerate>=1.0" "torch>=2.5"

运行::

    python sft_train_transformers.py \\
        --model_name __BASE_MODEL__ \\
        --train_file data/finetune/out/train.jsonl \\
        --eval_file data/finetune/out/eval.jsonl

设计要点（与本课教程一一对应）：

1. 前缀与答案**分别** tokenize 再拼接——BPE 的合并不认字符边界；
2. 答案放不下的样本直接丢弃并计数，**绝不截断答案**；
3. labels 的 padding 补 -100，与 input_ids/attention_mask 三件套一起做；
4. 训练前先把"监督 token 占比"打印出来——它是这笔算力花得值不值的度量。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

#: 屏蔽位：PyTorch CrossEntropyLoss 的 ignore_index 默认值
IGNORE_INDEX = -100

#: 与训练数据渲染时使用的 system 提示词保持一致
SYSTEM_PROMPT = """__SYSTEM_PROMPT__"""

#: 由 SFTTrainingArgs.to_hf_dict() 生成，字段名与 TrainingArguments 逐字对齐
TRAINING_ARGUMENTS = json.loads(
    """__HF_ARGS_JSON__"""
)

#: SFT 专属参数（TrainingArguments 不认，由本脚本的数据准备阶段消费）
MAX_LENGTH = __MAX_LENGTH__


def build_prefix_and_answer(example: dict) -> tuple[str, str]:
    """把一条 alpaca 样本拆成（前缀文本, 监督文本）.

    ``instruction`` 是任务描述、``input`` 是待处理材料，两者都非空时用
    空行分隔——与 day048 ``TrainingExample.prompt_text`` 的口径完全一致。
    答案末尾带上轮次终止符：模型必须学会"在哪里停下"。
    """
    instruction = str(example.get("instruction") or "")
    extra = str(example.get("input") or "")
    prompt = f"{instruction}\\n\\n{extra}" if extra else instruction
    prefix = (
        f"<|im_start|>system\\n{SYSTEM_PROMPT}<|im_end|>\\n"
        f"<|im_start|>user\\n{prompt}<|im_end|>\\n"
        "<|im_start|>assistant\\n"
    )
    answer = str(example.get("output") or "") + "<|im_end|>\\n"
    return prefix, answer


def tokenize_example(example: dict, tokenizer, max_length: int) -> dict | None:
    """渲染 + 编码 + 打 mask；返回 ``None`` 表示该样本不可用（会被丢弃）.

    ``None`` 的两种情形：前缀为空、答案为空。答案放不下 max_length 时
    也返回 None——**宁可少一条样本，也不教模型"说一半就停"**。
    """
    prefix, answer = build_prefix_and_answer(example)
    prompt_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    if not prompt_ids or not answer_ids:
        return None
    budget = max_length - len(answer_ids)
    if budget < 1:
        return None
    kept_prompt = prompt_ids[-budget:]  # 从左侧裁掉前缀，保留靠近答案的指令
    input_ids = kept_prompt + answer_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        # 关键：prompt 段的 label 置 -100，只有答案参与 loss
        "labels": [IGNORE_INDEX] * len(kept_prompt) + answer_ids,
    }


def make_collator(pad_token_id: int):
    """构造 padding 的 collate 函数（三件套必须一起补）.

    等价于库里的 ``DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100)``，
    这里手写一遍是为了让"padding 到底补了什么"可见。
    """

    def collate(features: list[dict]) -> dict:
        longest = max(len(item["input_ids"]) for item in features)
        input_ids, attention_mask, labels = [], [], []
        for item in features:
            pad = longest - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_token_id] * pad)
            attention_mask.append(item["attention_mask"] + [0] * pad)
            labels.append(item["labels"] + [IGNORE_INDEX] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def preprocess_logits_for_metrics(logits, labels):
    """评估时先把 logits 压成 argmax，避免把 (batch, seq, vocab) 全留在显存里.

    这是把"评估 OOM"从常见故障变成不存在的标准做法：token 级准确率只需要
    argmax，不需要完整分布。
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def build_compute_metrics():
    """构造 ``mean_token_accuracy``：监督位置上的下一 token 命中率."""

    def compute_metrics(eval_pred) -> dict:
        predictions, labels = eval_pred
        mask = labels != IGNORE_INDEX
        total = int(mask.sum())
        if total == 0:
            return {"mean_token_accuracy": float("nan")}
        hits = int((predictions[mask] == labels[mask]).sum())
        return {"mean_token_accuracy": hits / total}

    return compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT 训练（transformers + Trainer）")
    parser.add_argument("--model_name", default="__BASE_MODEL__")
    parser.add_argument("--train_file", default="data/finetune/out/train.jsonl")
    parser.add_argument("--eval_file", default="data/finetune/out/eval.jsonl")
    parser.add_argument("--output_dir", default=TRAINING_ARGUMENTS["output_dir"])
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.bfloat16)
    model.config.use_cache = False  # 训练时关掉 KV cache，省显存

    raw = load_dataset(
        "json", data_files={"train": args.train_file, "eval": args.eval_file}
    )
    columns = raw["train"].column_names
    tokenized = raw.map(
        lambda example: tokenize_example(example, tokenizer, args.max_length),
        remove_columns=columns,
        desc="渲染 + 编码 + 打 mask",
    )
    # map 返回 None 的行会被丢弃；这里的 filter 是双保险，并顺手挡掉长度不足 2 的样本
    tokenized = tokenized.filter(
        lambda row: row["input_ids"] is not None and len(row["input_ids"]) > 1,
        desc="丢弃不可用样本（答案为空或放不下）",
    )

    for name in ("train", "eval"):
        rows = tokenized[name]
        total = sum(len(row) for row in rows["input_ids"])
        supervised = sum(
            1 for row in rows["labels"] for value in row if value != IGNORE_INDEX
        )
        if total:
            print(
                f"[{name}] {len(rows)} 条 | {total} token | 监督 {supervised} "
                f"({supervised / total:.1%})"
            )

    training_args = TrainingArguments(output_dir=args.output_dir, **TRAINING_ARGUMENTS)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["eval"],
        data_collator=make_collator(tokenizer.pad_token_id),
        compute_metrics=build_compute_metrics(),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    trainer.train()

    metrics = trainer.evaluate()
    metrics["train_examples"] = len(tokenized["train"])
    metrics["eval_examples"] = len(tokenized["eval"])
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "final_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"训练完成，产物目录：{args.output_dir}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
'''


_TRL_SCRIPT = '''#!/usr/bin/env python
"""SFT 训练脚本（TRL SFTTrainer）——由 smart_research_agent.sft.hf_script 生成.

安装依赖::

    pip install "trl>=1.5" "transformers>=5.16" "datasets>=3.0" "accelerate>=1.0"

运行::

    python sft_train_trl.py --model_name __BASE_MODEL__

与本课 ``trainer.py`` 的概念对照（TRL 把这些都封装好了）：

| 本课实现 | TRL 的等价开关 |
|----------|---------------|
| prompt 段 label 置 -100 | ``assistant_only_loss=True`` |
| ``max_length`` 截断（保留答案） | ``max_length``（旧名 ``max_seq_length``） |
| 不做序列拼接 | ``packing=False`` |
| 只用 (prompt, completion) | ``completion_only_loss=True`` |

注意 ``packing=True``（把多条短样本拼进一条序列）会**改变 loss 的分母
口径与样本边界**，本课刻意关闭它：在 37 条样本这个规模上，吞吐收益远
小于"每步看到什么不可复现"的代价。
"""

from __future__ import annotations

import json

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

#: 与 TrainingArguments 逐字对齐，另加三个 SFT 专属项
SFT_CONFIG = json.loads("""__TRL_ARGS_JSON__""")

SYSTEM_PROMPT = """__SYSTEM_PROMPT__"""


def to_prompt_completion(example: dict) -> dict:
    """转成 TRL 的 prompt-completion 会话格式（自动套用 chat 模板）."""
    instruction = str(example.get("instruction") or "")
    extra = str(example.get("input") or "")
    prompt = f"{instruction}\\n\\n{extra}" if extra else instruction
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "completion": [{"role": "assistant", "content": str(example.get("output") or "")}],
    }


def main() -> None:
    dataset = load_dataset(
        "json",
        data_files={
            "train": "data/finetune/out/train.jsonl",
            "eval": "data/finetune/out/eval.jsonl",
        },
    ).map(
        to_prompt_completion,
        remove_columns=["instruction", "input", "output"],
        desc="转成 prompt-completion 会话格式",
    )
    trainer = SFTTrainer(
        model="__BASE_MODEL__",
        args=SFTConfig(**SFT_CONFIG),
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
    )
    trainer.train()
    trainer.save_model(SFT_CONFIG["output_dir"])


if __name__ == "__main__":
    main()
'''


def render_hf_sft_script(
    args: SFTTrainingArgs,
    *,
    base_model: str = DEFAULT_BASE_MODEL,
    system_prompt: str = DEFAULT_SYSTEM_IN_SCRIPT,
) -> str:
    """生成 ``transformers`` + ``Trainer`` 的训练脚本（返回脚本文本）.

    生成物是**可执行的 Python**：``test_sft_hf_script.py`` 会 ``compile()``
    它并断言关键片段的取值，因此"教程里的参数"与"脚本里的参数"不可能漂移。
    """
    args.validate()
    return (
        _HF_SCRIPT.replace("__BASE_MODEL__", base_model)
        .replace("__SYSTEM_PROMPT__", system_prompt)
        .replace("__HF_ARGS_JSON__", json.dumps(args.to_hf_dict(), ensure_ascii=False, indent=2))
        .replace("__MAX_LENGTH__", str(args.max_length))
    )


def render_trl_sft_script(
    args: SFTTrainingArgs,
    *,
    base_model: str = DEFAULT_BASE_MODEL,
    system_prompt: str = DEFAULT_SYSTEM_IN_SCRIPT,
) -> str:
    """生成 TRL ``SFTTrainer`` 的训练脚本（更短，但等价）. """
    args.validate()
    return (
        _TRL_SCRIPT.replace("__BASE_MODEL__", base_model)
        .replace("__SYSTEM_PROMPT__", system_prompt)
        .replace("__TRL_ARGS_JSON__", json.dumps(args.to_trl_dict(), ensure_ascii=False, indent=2))
    )


def dependency_commands(*, use_trl: bool = False) -> dict[str, str]:
    """生成安装命令（让脚本的依赖可被复制粘贴，而不是散在注释里）."""
    packages = TRL_DEPENDENCIES if use_trl else HF_DEPENDENCIES
    joined = " ".join(f'"{item}"' for item in packages)
    return {
        "pip": f"pip install {joined}",
        "uv": f"uv pip install {joined}",
        "packages": ", ".join(packages),
    }


def generated_script_summary(args: SFTTrainingArgs) -> dict[str, Any]:
    """生成脚本的自述信息（供 API 与教程直接展示）.

    这是"契约自带元信息"在脚本生成上的延伸：调用方不需要读脚本文本就能
    知道"这份脚本会用哪些超参、需要什么依赖"。
    """
    return {
        "base_model": DEFAULT_BASE_MODEL,
        "max_length": args.max_length,
        "packing": False,
        "assistant_only_loss": True,
        "hf_dependencies": list(HF_DEPENDENCIES),
        "trl_dependencies": list(TRL_DEPENDENCIES),
        "training_arguments": args.to_hf_dict(),
        "sft_specific": {
            "max_length": args.max_length,
            "truncation": args.truncation,
            "ignore_index": -100,
        },
    }
