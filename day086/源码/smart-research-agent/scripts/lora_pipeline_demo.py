#!/usr/bin/env python
"""day052 LoRA 训练与部署流水线演示：适配器生命周期 → 多卡计划 → 合并验证 → 生成脚本.

用法（在 ``day052/源码/smart-research-agent`` 下）::

    python scripts/lora_pipeline_demo.py

本脚本不依赖任何深度学习框架：训练复用 day050/day051 的纯 Python 参考模型，
"多卡"是**算术**（每设备显存、全局批、学习率、通信量），配置是**文本**
（accelerate 的 YAML 与 DeepSpeed 的 JSON）。整段演示十几秒跑完，每一步都能
在本机复算。

真实的多卡训练路径由第 5 节生成的 Accelerate 脚本承担。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from smart_research_agent.finetune import build_dataset, split_dataset
from smart_research_agent.peft import (
    ADAPTER_FILES,
    MODEL_SPECS,
    LoRAConfig,
    LoRAReferenceModel,
    accelerate_config,
    adapter_registry,
    build_manifest,
    deepspeed_zero_config,
    default_reference_lora_config,
    deployment_report,
    device_fit_summary,
    launch_command,
    load_adapter,
    plan_distributed,
    prune_adapters,
    reference_lora_accounting,
    render_accelerate_lora_script,
    render_config_yaml,
    render_inference_script,
    render_merge_script,
    repackage_adapter,
    save_adapter,
    verify_merge,
    write_manifest,
)
from smart_research_agent.sft import (
    REFERENCE_LEARNING_RATE,
    CharTokenizer,
    ReferenceSFTModel,
    SFTTrainingArgs,
    encode_supervised,
    iter_batches,
    render_supervised_list,
)

STEPS = 4


def banner(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def build_env():
    """准备数据、词表与批次（与 day050/day051 同一套渲染 / 编码流程）."""
    bundle = build_dataset()
    train, eval_set = split_dataset(bundle.examples, eval_ratio=0.2, seed=42)
    rendered = render_supervised_list(list(train) + list(eval_set))
    tokenizer = CharTokenizer.from_texts([item.text for item in rendered])
    train_batches = iter_batches(
        [
            encode_supervised(item, tokenizer, max_length=320)
            for item in render_supervised_list(list(train))
        ],
        batch_size=2,
        drop_last=True,
    )
    eval_batches = iter_batches(
        [
            encode_supervised(item, tokenizer, max_length=320)
            for item in render_supervised_list(list(eval_set))
        ],
        batch_size=2,
        drop_last=False,
    )
    return tokenizer, train, eval_set, train_batches, eval_batches


def section_1_adapter_lifecycle(tokenizer: CharTokenizer, train_batches) -> tuple:
    banner("[1] 适配器生命周期：三个文件 + 一份清单")
    model = LoRAReferenceModel(
        ReferenceSFTModel(tokenizer.vocab_size, seed=42),
        default_reference_lora_config(r=8, lora_alpha=16),
        seed=42,
    )
    args = SFTTrainingArgs(
        output_dir="outputs/lora",
        learning_rate=REFERENCE_LEARNING_RATE,
        num_train_epochs=2.0,
        max_length=320,
        save_strategy="no",
        eval_strategy="no",
    )
    console = []
    for step in range(STEPS):
        batch = train_batches[step % len(train_batches)]
        loss_sum, count = model.accumulate(batch)
        model.apply_update(REFERENCE_LEARNING_RATE)
        console.append(loss_sum / count)
    print(
        f"训练 {STEPS} 步：loss {console[0]:.4f} -> {console[-1]:.4f} | "
        f"可训练 {model.trainable_parameters}（{model.trainable_ratio:.4%}）"
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "adapters"
        checkpoint = save_adapter(
            root / "adapter-final",
            model=model,
            args=args,
            step=STEPS,
            boundary="adapter-final",
        )
        print("适配器目录里的三个文件（缺任何一个都不该被加载）：")
        for name in ADAPTER_FILES:
            size = (root / "adapter-final" / name).stat().st_size
            print(f"  {name:<24} {size:>8} B")
        print(f"  合计 {checkpoint.adapter_bytes} B | {checkpoint.summary_line()}")

        payload = load_adapter(root / "adapter-final")
        print(
            f"回读：sha256:{payload['sha256'][:12]} | updates="
            f"{payload['model']['updates']} | step={payload['state']['step']}"
        )
        manifest = build_manifest(
            root / "adapter-final", base_model="Qwen/Qwen3-0.6B", tags={"env": "demo"}
        )
        write_manifest(root / "adapter-final", manifest)
        print(f"清单：{manifest.summary_line()}")

        seven_b = deployment_report(
            spec_parameters=MODEL_SPECS["llama-2-7b"].total_parameters(), manifest=manifest
        )
        full_bytes = (tokenizer.vocab_size**2 + tokenizer.vocab_size) * 2
        print(
            f"对照：参考模型（V=557）的完整权重（bf16）是 {full_bytes} B，"
            f"适配器小 {full_bytes / checkpoint.adapter_bytes:.1f} 倍"
        )
        print(
            "  但参考模型上这个倍数没有意义（V=557 时 V² 与 2rV 同量级）。"
            "真实量级看 llama-2-7b："
            f"基座 {seven_b['base_model_bytes']} B vs 适配器 {seven_b['adapter_bytes']} B "
            f"→ 小 {1 / seven_b['adapter_ratio']:.0f} 倍"
        )

        print("\n清理策略（save_total_limit=2，adapter-final 永不删除）：")
        for step in (2, 4, 6):
            save_adapter(
                root / f"adapter-step-{step}",
                model=model,
                args=args,
                step=step,
                boundary=f"adapter-step-{step}",
            )
        print(f"  落盘 3 个中间适配器后：{sorted(p.name for p in root.iterdir())}")
        removed = prune_adapters(root, limit=2)
        print(f"  删除 {removed}")
        print(f"  剩余 {sorted(p.name for p in root.iterdir())}（adapter-final 永不删除）")

        published = repackage_adapter(
            root / "adapter-final", Path(tmp) / "published", tags={"env": "staging"}
        )
        print(f"发布目录：{sorted(p.name for p in (Path(tmp) / 'published').iterdir())}")
        print(f"发布清单：{published.summary_line()}")
    return model, args


def section_2_distributed() -> None:
    banner("[2] 多卡与混合精度：D 个设备的账不是「除以 D」")
    args = SFTTrainingArgs(
        learning_rate=2e-4,
        num_train_epochs=2.0,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        save_strategy="no",
        eval_strategy="no",
    )
    lora = LoRAConfig(r=8, lora_alpha=16, target_modules="attention_all")
    for label, full in (("LoRA", False), ("全参微调", True)):
        print(f"-- {label}（llama-2-7b / 4 个设备 / 1000 条训练样本）")
        for strategy in ("ddp", "zero2", "zero3"):
            plan = plan_distributed(
                MODEL_SPECS["llama-2-7b"],
                args,
                devices=4,
                strategy=strategy,
                train_size=1000,
                lora_config=None if full else lora,
                full_finetune=full,
            )
            print(
                f"   {strategy:6s} 每设备 {plan.per_device_gib:6.2f} GiB"
                f"（单卡 {plan.single_device_gib:6.2f}，省 {plan.memory_saving_factor:.2f}×）"
                f" | 通信/步 {plan.communication_bytes_per_step / 1e9:9.3f} GB"
                f" | 全局批 {plan.global_batch_size} | lr {plan.scaled_learning_rate:g}"
            )
        print()
    plan = plan_distributed(
        MODEL_SPECS["llama-2-7b"],
        args,
        devices=4,
        strategy="zero3",
        train_size=1000,
        lora_config=lora,
    )
    if plan.warnings:
        print("zero3 的告警（每条对应一个真实代价）：")
        for message in plan.warnings:
            print(f"  - {message}")

    print("\naccelerate 配置（可被 `accelerate launch --config_file` 直接消费）：")
    config = accelerate_config(
        devices=4, strategy="zero2", mixed_precision="bf16", deepspeed_config_file="ds_zero2.json"
    )
    for line in render_config_yaml(config).splitlines():
        print(f"  {line}")
    print("DeepSpeed ZeRO-2 配置：")
    print(
        "  "
        + json.dumps(deepspeed_zero_config(stage=2, mixed_precision="bf16"), ensure_ascii=False)
    )
    print(f"启动命令：{launch_command('train_lora_zero2.py', devices=4, strategy='zero2')}")
    fit = device_fit_summary(plan, budget_gib=24.0)
    print(f"单卡预算 24 GiB 下：{json.dumps(fit, ensure_ascii=False)}")


def section_3_merge(model: LoRAReferenceModel, eval_batches, tokenizer: CharTokenizer) -> None:
    banner("[3] 合并与验证：部署门禁")
    verification = verify_merge(model, eval_batches)
    print(f"合并验证：{verification.summary_line()}")
    print(
        "  ↑ 这是部署流水线的硬门禁：它把 day051「合并前后逐位一致」那条不变式"
        "从单元测试搬到了交付流程里"
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "adapters"
        args = SFTTrainingArgs(
            output_dir=tmp,
            learning_rate=REFERENCE_LEARNING_RATE,
            save_strategy="no",
            eval_strategy="no",
        )
        save_adapter(
            root / "adapter-final",
            model=model,
            args=args,
            step=STEPS,
            boundary="adapter-final",
        )
        manifest = build_manifest(root / "adapter-final")
        report = deployment_report(
            spec_parameters=MODEL_SPECS["llama-2-7b"].total_parameters(), manifest=manifest
        )
        print("\n两种部署形态的体积对照（llama-2-7b 基座）：")
        print(
            f"  基座                {report['base_model_bytes']:>14} B"
            f"（{report['base_model_gib']:.2f} GiB）"
        )
        print(
            f"  适配器              {report['adapter_bytes']:>14} B"
            f"（{report['adapter_mebibytes']:.4f} MiB）"
        )
        print(f"  适配器 / 基座        {report['adapter_ratio']:.3e}")
        print(f"  10 个合并模型       {report['ten_merged_models_bytes']:>14} B")
        print(
            f"  1 基座 + 10 适配器  {report['one_base_plus_ten_adapters_bytes']:>14} B"
            f" → 省 {report['multi_adapter_saving_factor']:.2f}×"
        )
        rows = adapter_registry([manifest])
        print(f"\n适配器注册表（{len(rows)} 行）：")
        for row in rows:
            print(f"  {json.dumps(row, ensure_ascii=False)}")
    accounting = reference_lora_accounting(
        tokenizer.vocab_size, default_reference_lora_config()
    )
    print(f"\n参考模型上的账（用于核对）：{json.dumps(accounting, ensure_ascii=False)}")
    print("  注意区分：参考模型（V=557）与 llama-2-7b 的参数量相差四个数量级，"
          "但两者的『适配器/基座』比例关系是同一套算术。")


def section_4_scripts() -> None:
    banner("[4] 生成的脚本：多卡训练 / 合并 / 推理")
    args = SFTTrainingArgs(
        learning_rate=2e-4,
        num_train_epochs=2.0,
        gradient_accumulation_steps=4,
        max_length=320,
        save_strategy="no",
        eval_strategy="no",
    )
    lora = LoRAConfig(r=8, lora_alpha=16, target_modules="attention")
    accelerate_script = render_accelerate_lora_script(
        args, lora, devices=4, strategy="zero2"
    )
    scripts = (
        ("多卡训练（Accelerate）", accelerate_script),
        ("合并（merge_and_unload）", render_merge_script(lora)),
        ("推理（适配器形态）", render_inference_script()),
    )
    for name, script in scripts:
        compile(script, f"{name}.py", "exec")
        print(f"{name}：{len(script.splitlines())} 行，compile 通过")
    print("\n多卡脚本里必须出现的三处（对应多卡训练的三个真实坑）：")
    main_process_markers = (
        "accelerator = ",
        "model, trainer = ",
        "if accelerator.is_main_process",
    )
    for line in accelerate_script.splitlines():
        stripped = line.strip()
        if stripped.startswith(main_process_markers):
            print(f"  {stripped}")
    print("\n合并脚本的核心三行：")
    merge_markers = ("model = PeftModel", "model = model.merge_and_unload", "model.save_pretrained")
    for line in scripts[1][1].splitlines():
        stripped = line.strip()
        if stripped.startswith(merge_markers):
            print(f"  {stripped}")


def main() -> None:
    tokenizer, _, _, train_batches, eval_batches = build_env()
    print(f"数据：词表 V={tokenizer.vocab_size} | train {len(train_batches)} 个 micro-batch")
    model, _ = section_1_adapter_lifecycle(tokenizer, train_batches)
    section_2_distributed()
    section_3_merge(model, eval_batches, tokenizer)
    section_4_scripts()
    banner("完成")
    print("真实多卡训练请使用第 4 节生成的 Accelerate 脚本 + 第 2 节生成的配置文件。")


if __name__ == "__main__":
    main()
