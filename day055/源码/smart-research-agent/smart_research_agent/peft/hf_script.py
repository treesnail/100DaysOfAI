"""生成可运行的 LoRA / QLoRA 训练脚本（M5-D3）.

day050 的 ``sft/hf_script.py`` 生成了全参 SFT 的 ``transformers`` + ``Trainer``
脚本；本模块在它之上只加**两行**：

.. code-block:: python

    peft_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, peft_config)

这就是 LoRA 在工程上的全部侵入性。数据准备（渲染 → 分别 tokenize → label
mask → padding 三件套）**一行都不用改**——day050 第五章的标准
``DataCollatorForSeq2Seq(label_pad_token_id=-100)`` 与手写 collator 在这里原样复用。

QLoRA 版本再多三处，且每一处都有明确目的：

.. code-block:: python

    bnb_config = BitsAndBytesConfig(**QUANT_CONFIG)        # 基座按 4-bit 载入
    model = prepare_model_for_kbit_training(model)          # 冻结基座 + 打开梯度检查点
    model = get_peft_model(model, peft_config)              # 适配器仍是 bf16

``prepare_model_for_kbit_training`` 做三件事：把基座参数 ``requires_grad=False``、
把 LayerNorm 提升到 fp32（4-bit 反量化后的数值稳定性）、启用输入梯度（梯度
检查点需要）。**漏掉它会得到一个"训练在跑但 loss 不降"的脚本**——所以生成的
脚本里把这一步与它的理由一起写出来。

生成而不是手写，理由与 day050 相同：**参数只有一个来源**。``LORA_CONFIG``
来自 ``LoRAConfig.to_peft_dict()``、``QUANT_CONFIG`` 来自
``QLoRAConfig.to_bnb_dict()``、``TRAINING_ARGUMENTS`` 来自
``SFTTrainingArgs.to_hf_dict()``，测试会把它们从脚本文本里抠出来 ``json.loads``
再与来源逐键比对——**教程里的参数与脚本里的参数不可能漂移**。

依赖版本（2026-09 核对于 PyPI）：``peft 0.20.0``、``bitsandbytes 0.50.2``。
"""

from __future__ import annotations

import json
from typing import Any

from smart_research_agent.peft.config import LoRAConfig, QLoRAConfig
from smart_research_agent.peft.targets import DecoderSpec, plan_lora
from smart_research_agent.sft.args import SFTTrainingArgs
from smart_research_agent.sft.hf_script import (
    DEFAULT_BASE_MODEL,
    DEFAULT_SYSTEM_IN_SCRIPT,
    HF_DEPENDENCIES,
)

#: LoRA 训练脚本的第三方依赖（在 day050 的全参依赖上只多了 ``peft``）
PEFT_DEPENDENCIES: tuple[str, ...] = (*HF_DEPENDENCIES, "peft>=0.20")

#: QLoRA 训练脚本的依赖（再多一个 4-bit 算子库）
QLORA_DEPENDENCIES: tuple[str, ...] = (*PEFT_DEPENDENCIES, "bitsandbytes>=0.50")

#: 生成脚本里的 dtype 字符串 → torch dtype 的映射（JSON 里存字符串是刻意的：
#: ``BitsAndBytesConfig`` 实际接受 ``torch.dtype``，但把 ``torch.bfloat16``
#: 序列化进模板会让模板与 torch 版本耦合；脚本里显式映射一次更清楚）
DTYPE_MAPPING_SNIPPET = '''COMPUTE_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}'''


COMMON_DATA_PREP = '''def build_prefix_and_answer(example: dict) -> tuple[str, str]:
    """把一条 alpaca 样本拆成（前缀文本, 监督文本）——与 day050 逐行相同."""
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
    """渲染 + 编码 + 打 mask；返回 None 表示该样本不可用（会被丢弃）.

    **LoRA / QLoRA 不改这一段**：label mask 是数据准备的事，与"训哪些参数"
    正交。答案放不下时宁可丢弃样本，也不截断答案。
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
        "labels": [IGNORE_INDEX] * len(kept_prompt) + answer_ids,
    }


def report_supervised_ratio(dataset, name: str) -> float:
    """打印监督 token 占比——训练开始前"这笔算力花得值不值"的第一手数字."""
    total = sum(len(row) for row in dataset["input_ids"])
    supervised = sum(
        1 for row in dataset["labels"] for value in row if value != IGNORE_INDEX
    )
    if total == 0:
        print(f"[{name}] 没有可用样本")
        return 0.0
    print(f"[{name}] {len(dataset)} 条 | {total} token | 监督 {supervised} "
          f"({supervised / total:.1%})")
    return supervised / total


def count_trainable(model) -> tuple[int, int]:
    """返回 ``(可训练参数, 总参数)``，并在偏差过大时**直接报警**.

    ``peft`` 的 ``print_trainable_parameters()`` 只打印，本脚本额外做一次
    核对：如果实测占比与本课算术算出的预期相差超过 20%，说明
    ``target_modules`` 没命中预期模块（比如模型用的是 ``Wqkv`` 这类合并命名）。
    **静默少训一半参数，比直接报错难发现得多。**
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"可训练 {trainable:,} / 总 {total:,} = {trainable / total:.4%}")
    expected = EXPECTED_TRAINABLE
    if expected and abs(trainable - expected) / expected > 0.2:
        print(f"[警告] 实测可训练参数 {trainable:,} 与预期 {expected:,} 偏差超过 20%，"
              "请检查 target_modules 是否命中了预期的模块")
    return trainable, total


def make_collator(pad_token_id: int):
    """padding 三件套（ids 补 pad / labels 补 -100 / mask 补 0）。"""

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


def load_tokenized(tokenizer, train_file: str, eval_file: str, max_length: int):
    """加载并预处理数据集（与 day050 的脚本同一套流程）."""
    raw = load_dataset("json", data_files={"train": train_file, "eval": eval_file})
    columns = raw["train"].column_names
    tokenized = raw.map(
        lambda example: tokenize_example(example, tokenizer, max_length),
        remove_columns=columns,
        desc="渲染 + 编码 + 打 mask",
    )
    return tokenized.filter(
        lambda row: row["input_ids"] is not None and len(row["input_ids"]) > 1,
        desc="丢弃不可用样本（答案为空或放不下）",
    )'''

_LORA_SCRIPT = '''#!/usr/bin/env python
"""LoRA 训练脚本（transformers + peft）——由 smart_research_agent.peft.hf_script 生成.

安装依赖::

    pip install "transformers>=5.16" "datasets>=3.0" "accelerate>=1.0" \\
                "torch>=2.5" "peft>=0.20"

运行::

    python lora_train.py \\
        --model_name __BASE_MODEL__ \\
        --train_file data/finetune/out/train.jsonl \\
        --eval_file data/finetune/out/eval.jsonl

与 day050 全参脚本的差别只有两行（``LoraConfig`` + ``get_peft_model``）：
**数据准备与 label mask 一行都没变**。这正是 LoRA 的工程价值——它不要求
你重写训练管线。

跑完之后 ``--output_dir`` 里只有适配器（十几 MB），基座不落盘：
加载时用 ``PeftModel.from_pretrained(base, adapter_dir)`` 重新拼装。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

IGNORE_INDEX = -100

SYSTEM_PROMPT = """__SYSTEM_PROMPT__"""

#: 由 SFTTrainingArgs.to_hf_dict() 生成
TRAINING_ARGUMENTS = json.loads(
    """__HF_ARGS_JSON__"""
)

#: 由 LoRAConfig.to_peft_dict() 生成（字段名与 peft.LoraConfig 逐字对齐）
LORA_CONFIG = json.loads(
    """__LORA_JSON__"""
)

MAX_LENGTH = __MAX_LENGTH__

#: 本课算术算出的可训练参数个数（来自 peft.targets.plan_lora）——用于核对
EXPECTED_TRAINABLE = __EXPECTED_TRAINABLE__

__DTYPE_MAPPING__


__DATA_PREP__


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA 训练（transformers + peft）")
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
    model.config.use_cache = False

    # ---- LoRA 的全部侵入性：两行 ----
    task_type = TaskType[LORA_CONFIG["task_type"]]
    peft_config = LoraConfig(
        task_type=task_type,
        **{key: value for key, value in LORA_CONFIG.items() if key != "task_type"},
    )
    model = get_peft_model(model, peft_config)
    count_trainable(model)

    dataset = load_tokenized(tokenizer, args.train_file, args.eval_file, args.max_length)
    report_supervised_ratio(dataset["train"], "train")
    report_supervised_ratio(dataset["eval"], "eval")

    training_args = TrainingArguments(output_dir=args.output_dir, **TRAINING_ARGUMENTS)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        data_collator=make_collator(tokenizer.pad_token_id),
    )
    trainer.train()

    # 只保存适配器：这是 LoRA 部署形态的核心——一个基座 + N 个小适配器
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    metrics = trainer.evaluate()
    (Path(args.output_dir) / "final_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"训练完成，适配器目录：{args.output_dir}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
'''

_QLORA_SCRIPT = '''#!/usr/bin/env python
"""QLoRA 训练脚本（4-bit 基座 + peft）——由 smart_research_agent.peft.hf_script 生成.

安装依赖::

    pip install "transformers>=5.16" "datasets>=3.0" "accelerate>=1.0" \\
                "torch>=2.5" "peft>=0.20" "bitsandbytes>=0.50"

运行::

    python qlora_train.py --model_name __BASE_MODEL__

三处与 LoRA 脚本不同的地方，每一处都有明确目的：

1. ``BitsAndBytesConfig``：基座按 4-bit 载入（``nf4`` + 二级量化 + 块大小 64），
   每参数存储 __BYTES_PER_PARAMETER__ 字节，而 bf16 是 2 字节；
2. ``prepare_model_for_kbit_training``：冻结基座、把 LayerNorm 提升到 fp32、
   启用输入梯度。**漏掉这一步会得到"训练在跑但 loss 不降"的脚本**；
3. 训练参数里 ``bf16=True``：4-bit 反量化后的计算精度是 bf16——适配器
   自始至终都是 bf16，**"QLoRA 用 4-bit 训练"这个说法是不准确的**。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

IGNORE_INDEX = -100

SYSTEM_PROMPT = """__SYSTEM_PROMPT__"""

TRAINING_ARGUMENTS = json.loads(
    """__HF_ARGS_JSON__"""
)

LORA_CONFIG = json.loads(
    """__LORA_JSON__"""
)

#: 由 QLoRAConfig.to_bnb_dict() 生成（字段名与 BitsAndBytesConfig 逐字对齐）
QUANT_CONFIG = json.loads(
    """__BNB_JSON__"""
)

MAX_LENGTH = __MAX_LENGTH__

EXPECTED_TRAINABLE = __EXPECTED_TRAINABLE__

__DTYPE_MAPPING__


__DATA_PREP__


def build_quantization_config() -> BitsAndBytesConfig:
    """把 JSON 里的字符串配置转成 BitsAndBytesConfig（dtype 需要真实 torch 对象）."""
    payload = dict(QUANT_CONFIG)
    payload["bnb_4bit_compute_dtype"] = COMPUTE_DTYPES[QUANT_CONFIG["bnb_4bit_compute_dtype"]]
    return BitsAndBytesConfig(**payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA 训练（4-bit 基座 + peft）")
    parser.add_argument("--model_name", default="__BASE_MODEL__")
    parser.add_argument("--train_file", default="data/finetune/out/train.jsonl")
    parser.add_argument("--eval_file", default="data/finetune/out/eval.jsonl")
    parser.add_argument("--output_dir", default=TRAINING_ARGUMENTS["output_dir"])
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 1. 4-bit 基座 ----
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=build_quantization_config(),
        device_map="auto",
    )
    model.config.use_cache = False

    # ---- 2. 为 k-bit 训练做好准备（冻结基座 / LayerNorm 升 fp32 / 输入梯度）----
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # ---- 3. 适配器（bf16，与基座的 4-bit 存储互不影响）----
    peft_config = LoraConfig(
        task_type=TaskType[LORA_CONFIG["task_type"]],
        **{key: value for key, value in LORA_CONFIG.items() if key != "task_type"},
    )
    model = get_peft_model(model, peft_config)
    count_trainable(model)

    dataset = load_tokenized(tokenizer, args.train_file, args.eval_file, args.max_length)
    report_supervised_ratio(dataset["train"], "train")
    report_supervised_ratio(dataset["eval"], "eval")

    training_args = TrainingArguments(output_dir=args.output_dir, **TRAINING_ARGUMENTS)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        data_collator=make_collator(tokenizer.pad_token_id),
    )
    trainer.train()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"训练完成，适配器目录：{args.output_dir}")


if __name__ == "__main__":
    main()
'''


def _render(
    template: str,
    *,
    args: SFTTrainingArgs,
    lora: LoRAConfig,
    qlora: QLoRAConfig | None,
    base_model: str,
    system_prompt: str,
    expected_trainable: int | None,
) -> str:
    """把公共占位符替换进模板（两个脚本共用，避免两处漂移）."""
    text = (
        template.replace("__BASE_MODEL__", base_model)
        .replace("__SYSTEM_PROMPT__", system_prompt)
        .replace("__HF_ARGS_JSON__", json.dumps(args.to_hf_dict(), ensure_ascii=False, indent=2))
        .replace("__LORA_JSON__", json.dumps(lora.to_peft_dict(), ensure_ascii=False, indent=2))
        .replace("__MAX_LENGTH__", str(args.max_length))
        .replace("__DTYPE_MAPPING__", DTYPE_MAPPING_SNIPPET)
        .replace("__DATA_PREP__", COMMON_DATA_PREP)
        .replace("__EXPECTED_TRAINABLE__", str(expected_trainable) if expected_trainable else "0")
    )
    if qlora is not None:
        text = text.replace(
            "__BNB_JSON__", json.dumps(qlora.to_bnb_dict(), ensure_ascii=False, indent=2)
        ).replace("__BYTES_PER_PARAMETER__", f"{qlora.bytes_per_parameter:.6f}")
    return text


def render_lora_script(
    args: SFTTrainingArgs,
    lora: LoRAConfig,
    *,
    base_model: str = DEFAULT_BASE_MODEL,
    system_prompt: str = DEFAULT_SYSTEM_IN_SCRIPT,
    expected_trainable: int | None = None,
) -> str:
    """生成 ``transformers`` + ``peft`` 的 LoRA 训练脚本（返回脚本文本）.

    ``expected_trainable`` 由 ``peft.targets.plan_lora`` 提供（真实模型规格），
    参考模型上可以传 ``None``——脚本里的核对逻辑在预期为 0 时跳过。
    """
    args.validate()
    lora.validate()
    return _render(
        _LORA_SCRIPT,
        args=args,
        lora=lora,
        qlora=None,
        base_model=base_model,
        system_prompt=system_prompt,
        expected_trainable=expected_trainable,
    )


def render_qlora_script(
    args: SFTTrainingArgs,
    lora: LoRAConfig,
    qlora: QLoRAConfig | None = None,
    *,
    base_model: str = DEFAULT_BASE_MODEL,
    system_prompt: str = DEFAULT_SYSTEM_IN_SCRIPT,
    expected_trainable: int | None = None,
) -> str:
    """生成 4-bit 基座 + ``peft`` 的 QLoRA 训练脚本（返回脚本文本）."""
    args.validate()
    lora.validate()
    effective = qlora or QLoRAConfig()
    effective.validate()
    effective.to_bnb_dict()  # 提前拦下 int4 这类对照基线口径
    return _render(
        _QLORA_SCRIPT,
        args=args,
        lora=lora,
        qlora=effective,
        base_model=base_model,
        system_prompt=system_prompt,
        expected_trainable=expected_trainable,
    )


_ACCELERATE_SCRIPT = '''#!/usr/bin/env python
"""多卡 LoRA / QLoRA 训练脚本（Accelerate）——由 smart_research_agent.peft.hf_script 生成.

准备配置与启动::

    # 1. 生成配置（本脚本同级目录下的 accelerate_config.yaml 由 deploy 流程产出）
    # 2. 启动（4 个进程 / bf16 / DeepSpeed ZeRO-2 时 accelerate 会自动读配置）
    __LAUNCH_COMMAND__

与单卡 LoRA 脚本的差别只有三处，每处都有明确目的：

1. ``Accelerator`` 负责混合精度、设备放置与梯度同步——**不要把
   ``TrainingArguments.bf16`` 与 Accelerate 的 ``mixed_precision`` 同时打开
   两套开关**：它们会各自做一次 autocast 与缩放，结果是一半的算力用在重复计算上；
2. ``accelerator.prepare(...)`` 会把模型、优化器、数据加载器一起改造（DDP 包装、
   分片、device 放置）。**必须在构造 Trainer 之前调用**；
3. 只在主进程保存：``if accelerator.is_main_process``。四个进程同时往一个目录
   写适配器，轻则相互覆盖、重则文件损坏——**这是多卡训练最常见的"看起来没问题"的错**。

适配器仍然与单卡完全一样（同一份 ``LORA_CONFIG``）：**分片与混合精度改的是
"怎么算"，不改"训什么"**。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

IGNORE_INDEX = -100

SYSTEM_PROMPT = """__SYSTEM_PROMPT__"""

TRAINING_ARGUMENTS = json.loads(
    """__HF_ARGS_JSON__"""
)

LORA_CONFIG = json.loads(
    """__LORA_JSON__"""
)

MAX_LENGTH = __MAX_LENGTH__

EXPECTED_TRAINABLE = __EXPECTED_TRAINABLE__

DTYPE_MAPPING = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


__DATA_PREP__


def main() -> None:
    parser = argparse.ArgumentParser(description="多卡 LoRA 训练（Accelerate）")
    parser.add_argument("--model_name", default="__BASE_MODEL__")
    parser.add_argument("--train_file", default="data/finetune/out/train.jsonl")
    parser.add_argument("--eval_file", default="data/finetune/out/eval.jsonl")
    parser.add_argument("--output_dir", default=TRAINING_ARGUMENTS["output_dir"])
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    args = parser.parse_args()

    # mixed_precision 由 accelerate 的配置决定；**不要**再往 TrainingArguments 里传 bf16
    accelerator = Accelerator(gradient_accumulation_steps=__ACCUM__)
    accelerator.print(f"设备数 {{accelerator.num_processes}} | "
                      f"混合精度 {{accelerator.mixed_precision}}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=DTYPE_MAPPING["bfloat16"]
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType[LORA_CONFIG["task_type"]],
            **{key: value for key, value in LORA_CONFIG.items() if key != "task_type"},
        ),
    )
    if accelerator.is_main_process:
        count_trainable(model)

    dataset = load_tokenized(tokenizer, args.train_file, args.eval_file, args.max_length)
    if accelerator.is_main_process:
        report_supervised_ratio(dataset["train"], "train")
        report_supervised_ratio(dataset["eval"], "eval")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        # 梯度累积交给 Accelerator，TrainingArguments 里**留 1**，
        # 否则两处各累积一次，有效批会变成 accum² 倍。
        # 注意 TRAINING_ARGUMENTS 里**已经剔除了** gradient_accumulation_steps：
        # `f(x=1, **{"x": 4})` 在 Python 里直接抛 TypeError——本课实现时踩过，
        # 所以渲染时就把那个键从内嵌 JSON 里删掉（见 render_accelerate_lora_script）。
        gradient_accumulation_steps=1,
        **TRAINING_ARGUMENTS,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        data_collator=make_collator(tokenizer.pad_token_id),
    )
    # 必须在构造 Trainer 之前 prepare
    model, trainer = accelerator.prepare(model, trainer)
    trainer.train()

    # 多进程同时写一个目录 = 相互覆盖；只在主进程保存
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        Path(args.output_dir, "run_summary.json").write_text(
            json.dumps(
                {
                    "devices": accelerator.num_processes,
                    "mixed_precision": accelerator.mixed_precision,
                    "global_batch_size": TRAINING_ARGUMENTS["per_device_train_batch_size"]
                    * __ACCUM__
                    * accelerator.num_processes,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"训练完成，适配器目录：{{args.output_dir}}（仅主进程写入）")


if __name__ == "__main__":
    main()
'''


def render_accelerate_lora_script(
    args: SFTTrainingArgs,
    lora: LoRAConfig,
    *,
    devices: int = 2,
    strategy: str = "ddp",
    config_file: str = "accelerate_config.yaml",
    base_model: str = DEFAULT_BASE_MODEL,
    system_prompt: str = DEFAULT_SYSTEM_IN_SCRIPT,
    expected_trainable: int | None = None,
) -> str:
    """生成多卡 LoRA 训练脚本（Accelerate）（day052）.

    与 ``render_lora_script`` 的差别集中在三处（混合精度、``prepare``、
    主进程才落盘），它们分别对应多卡训练的三个真实坑。**数据准备与 label
    mask 一行没变**——与 day051 的结论一致：换训练规模不要求重写数据管线。

    ``__LAUNCH_COMMAND__`` 由 ``accelerate.launch_command`` 给出，因此
    "怎么启动"也是从同一个来源派生的（不会出现"脚本里的设备数与命令里的不一致"）。
    """
    from smart_research_agent.peft.accelerate import launch_command

    args.validate()
    lora.validate()
    command = launch_command(
        f"train_lora_{strategy}.py", devices=devices, strategy=strategy, config_file=config_file
    )
    # 关键：把 gradient_accumulation_steps 从内嵌 JSON 里**删掉**。
    # 脚本里显式写 `gradient_accumulation_steps=1`（累积交给 Accelerator），
    # 如果 JSON 里也带这个键，展开时会抛
    # `TypeError: got multiple values for keyword argument`——**脚本语法合法、
    # compile 也通过，只有在真正运行时才崩**。本课实现时踩过这一次。
    training_arguments = args.to_hf_dict()
    training_arguments.pop("gradient_accumulation_steps", None)
    text = (
        _ACCELERATE_SCRIPT.replace("__BASE_MODEL__", base_model)
        .replace("__SYSTEM_PROMPT__", system_prompt)
        .replace(
            "__HF_ARGS_JSON__",
            json.dumps(training_arguments, ensure_ascii=False, indent=2),
        )
        .replace("__LORA_JSON__", json.dumps(lora.to_peft_dict(), ensure_ascii=False, indent=2))
        .replace("__MAX_LENGTH__", str(args.max_length))
        .replace("__DTYPE_MAPPING__", DTYPE_MAPPING_SNIPPET)
        .replace("__DATA_PREP__", COMMON_DATA_PREP)
        .replace("__EXPECTED_TRAINABLE__", str(expected_trainable) if expected_trainable else "0")
        .replace("__ACCUM__", str(args.gradient_accumulation_steps))
        .replace("__LAUNCH_COMMAND__", command)
    )
    return text


def render_merge_script(
    lora: LoRAConfig,
    *,
    base_model: str = DEFAULT_BASE_MODEL,
    output_dir: str = "outputs/lora-merged",
) -> str:
    """转发到 ``peft.deploy.render_merge_script``（保持脚本生成的单一入口）."""
    from smart_research_agent.peft.deploy import render_merge_script as _render

    return _render(lora, base_model=base_model, output_dir=output_dir)


def peft_dependency_commands(*, use_qlora: bool = False) -> dict[str, str]:
    """生成安装命令（把依赖从注释里搬进可复制粘贴的一段）."""
    packages = QLORA_DEPENDENCIES if use_qlora else PEFT_DEPENDENCIES
    joined = " ".join(f'"{item}"' for item in packages)
    return {
        "pip": f"pip install {joined}",
        "uv": f"uv pip install {joined}",
        "packages": ", ".join(packages),
    }


def rendered_peft_summary(
    args: SFTTrainingArgs,
    lora: LoRAConfig,
    qlora: QLoRAConfig | None = None,
    *,
    spec: DecoderSpec | None = None,
    base_model: str = DEFAULT_BASE_MODEL,
) -> dict[str, Any]:
    """生成脚本的自述信息（供 API 与教程直接展示，不必读脚本文本）.

    ``spec`` 非空时附带 ``peft.targets.plan_lora`` 的参数量计划——这正是脚本里
    ``EXPECTED_TRAINABLE`` 的来源，也是"脚本训了多少参数"的唯一依据。

    ``base_model`` 必须与调用 ``render_lora_script`` / ``render_qlora_script`` 时
    传的值一致：自述信息的用途是"不必读脚本文本就知道这份脚本会用哪些
    超参"，如果基座名对不上，这个承诺就不成立了。
    """
    effective = qlora or QLoRAConfig()
    payload: dict[str, Any] = {
        "base_model": base_model,
        "lora": lora.to_peft_dict(),
        "scaling": lora.scaling,
        "scaling_formula": lora.scaling_formula,
        "quantization": effective.to_bnb_dict() if qlora is not None else None,
        "quantization_bytes_per_parameter": (
            round(effective.bytes_per_parameter, 6) if qlora is not None else None
        ),
        "training_arguments": args.to_hf_dict(),
        "max_length": args.max_length,
        "peft_dependencies": list(PEFT_DEPENDENCIES),
        "qlora_dependencies": list(QLORA_DEPENDENCIES),
        "install": peft_dependency_commands(use_qlora=qlora is not None),
    }
    if spec is not None:
        payload["plan"] = plan_lora(spec, lora).to_dict()
    return payload


__all__ = [
    "COMMON_DATA_PREP",
    "DTYPE_MAPPING_SNIPPET",
    "PEFT_DEPENDENCIES",
    "QLORA_DEPENDENCIES",
    "peft_dependency_commands",
    "render_accelerate_lora_script",
    "render_lora_script",
    "render_merge_script",
    "render_qlora_script",
    "rendered_peft_summary",
]
