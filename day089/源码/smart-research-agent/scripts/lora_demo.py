#!/usr/bin/env python
"""day051 LoRA / QLoRA 端到端演示：参数算术 → 量化 → 训练 → 合并 → 显存 → 生成脚本.

用法（在 ``day051/源码/smart-research-agent`` 下）::

    python scripts/lora_demo.py

本脚本**不依赖任何深度学习框架**：数据来自 day048 的落盘产物，LoRA 训练
复用 day050 的纯 Python 参考模型（真实梯度下降），量化用本课自实现的
NF4 / FP4 / INT4 码本与分块量化。整段演示在本机约 40 秒跑完，每一步都能
逐项核对。

真实训练路径由第 7 节生成的 ``peft`` / ``bitsandbytes`` 脚本承担。
"""

from __future__ import annotations

import random
import time

from smart_research_agent.finetune import build_dataset, split_dataset
from smart_research_agent.peft import (
    MODEL_SPECS,
    LoRAConfig,
    LoRAReferenceModel,
    QLoRAConfig,
    adapter_footprint,
    compare_codecs,
    compare_strategies,
    default_reference_lora_config,
    describe_delta,
    describe_model,
    device_fit_table,
    format_bytes,
    fp4_levels,
    int4_levels,
    minimal_device,
    nf4_levels,
    peft_dependency_commands,
    plan_lora,
    quantization_error_table,
    quantization_memory_note,
    quantization_table,
    quantize_reference_model,
    reference_lora_accounting,
    reference_matrix_ratios,
    render_lora_script,
    render_qlora_script,
    savings_table,
    target_preset_table,
    theoretical_adapter_parameter_cap,
)
from smart_research_agent.sft import (
    CharTokenizer,
    ReferenceSFTModel,
    SFTTrainingArgs,
    encode_supervised,
    iter_batches,
    render_supervised_list,
)

#: 训练预算：2 个 epoch × 15 个 micro-batch = 30 步（与 day050 的全参基线同预算）
EPOCHS = 2


def banner(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def build_batches():
    """准备数据与批次（与 day050 同一套渲染 / 编码 / padding 流程）."""
    bundle = build_dataset()
    train, eval_set = split_dataset(bundle.examples, eval_ratio=0.2, seed=42)
    rendered = render_supervised_list(list(train) + list(eval_set))
    tokenizer = CharTokenizer.from_texts([item.text for item in rendered])
    train_encoded = [
        encode_supervised(item, tokenizer, max_length=320)
        for item in render_supervised_list(list(train))
    ]
    eval_encoded = [
        encode_supervised(item, tokenizer, max_length=320)
        for item in render_supervised_list(list(eval_set))
    ]
    train_batches = iter_batches(train_encoded, batch_size=2, drop_last=True)
    eval_batches = iter_batches(eval_encoded, batch_size=2, drop_last=False)
    return tokenizer, train, eval_set, train_batches, eval_batches


def section_1_accounting(tokenizer: CharTokenizer) -> None:
    banner("[1] 参数量算术：LoRA 把平方级降到线性级")
    print("day049 预先算出的四个「单矩阵」比例（= 2r/d，方阵）：")
    for row in reference_matrix_ratios():
        print(
            f"  d={int(row['dimension']):<6} r={int(row['r']):<3} -> {row['ratio_percent']:.4f}%"
        )

    print(f"\n参考模型（本课程数据集 V={tokenizer.vocab_size}）的账：")
    accounting = reference_lora_accounting(
        tokenizer.vocab_size, default_reference_lora_config()
    )
    print(
        f"  基座 {accounting['frozen_parameters']}（W {tokenizer.vocab_size ** 2} + "
        f"b {tokenizer.vocab_size}）| 适配器 {accounting['trainable_parameters']} | "
        f"占比 {accounting['trainable_ratio']:.4%} | 秩上界 {accounting['rank_upper_bound']}"
    )

    print("\nllama-2-7b 上的目标预设对照（r=8）：同一份数据、同一个 r，参数量差 4.75 倍")
    for row in target_preset_table(MODEL_SPECS["llama-2-7b"], r=8):
        print(
            f"  {row['preset']:<15} {row['adapter_parameters']:>10,} 参数 "
            f"({row['trainable_ratio_percent']:.4f}%) 每层 {row['parameters_per_layer']:,}"
        )
    full_rank_cap = theoretical_adapter_parameter_cap(MODEL_SPECS["llama-2-7b"])
    print(
        f"  全秩上限（r = min(in,out)）：{full_rank_cap:,}"
        " ← 比整个模型还大，所以在 r 上乱加参数会让「参数高效」彻底失效"
    )
    footprint = adapter_footprint(r=8, in_features=1024, out_features=2048)
    print(
        "\n适配器落盘体积（attention_all / r=8 / 单个 1024×2048 投影）："
        f"{footprint['mebibytes']:.4f} MiB（压缩 {footprint['compression_ratio']:.1f}×）"
    )


def section_2_quantization(tokenizer: CharTokenizer) -> None:
    banner("[2] 量化：码本、每参数存储、以及「为什么是 NF4」")
    print("NF4 的 16 个码点（正态分布等概率分位数，归一化到 [-1,1]）：")
    for row in quantization_error_table(range(16), nf4_levels()):
        print(f"  code {int(row['code']):>2} ({row['binary']}) -> {row['level']:+.6f}")
    print(f"  NF4 里没有 0：最接近 0 的两个码点是 ±{abs(nf4_levels()[7]):.6f}")
    print(f"  FP4（E2M1）16 个编码只有 {len(set(fp4_levels()))} 个不同取值（0 有 ± 两个编码）")
    print(f"  INT4（对照基线，等间距）：{tuple(round(v, 4) for v in int4_levels()[:4])} …")

    print("\n块大小 → 每参数存储（4-bit 并没有把权重压到 0.5 字节）：")
    for row in quantization_table():
        print(
            f"  block={row['block_size']:<4} 二级量化 {row['bytes_per_parameter']:.6f} B/参数 "
            f"| 单重 {row['bytes_per_parameter_single_quant']:.6f} B/参数 "
            f"| 常数开销 {row['constant_bits_per_parameter']:.6f} bit"
        )
    print(f"  {quantization_memory_note(QLoRAConfig())}")

    print("\n三种码本在两份合成权重上的误差（block=64）——NF4 的优势依赖分布假设：")
    rng = random.Random(7)
    vocab = tokenizer.vocab_size
    gaussian = [[rng.gauss(0.0, 0.02) for _ in range(vocab)] for _ in range(vocab)]
    uniform = ReferenceSFTModel(vocab, seed=42).state_dict().weights
    for label, matrix in (
        ("均匀初始化 U(-0.02,0.02)（本课基座就是它）", uniform),
        ("高斯初始化 N(0,0.02²)（NF4 的目标分布）", gaussian),
    ):
        print(f"  -- {label}")
        for row in compare_codecs(matrix, block_size=64):
            print(
                f"     {row['quant_type']:5s} RMSE {row['rmse']:.10f} | "
                f"相对误差 {row['relative_error']:.6f}"
            )
    print("  两行的最优码本不同：NF4 的分位数布局偏向正态分布，均匀分布上反而输给 INT4。")


def section_3_train(
    tokenizer: CharTokenizer, train_batches, eval_batches
) -> tuple:
    banner("[3] LoRA 真的能学：初始化不变式 + 学习率扫 + ΔW 画像")
    vocab = tokenizer.vocab_size
    base = ReferenceSFTModel(vocab, seed=42)
    fresh = LoRAReferenceModel(base, default_reference_lora_config(r=8, lora_alpha=16), seed=42)
    mismatch = sum(1 for ctx in range(vocab) if fresh.logits(ctx) != base.logits(ctx))
    print(
        f"初始化不变式：ΔW=0 时 {vocab} 个上下文 token 中有 {mismatch} 个与基座不同 "
        f"（必须为 0）| ΔW 是否为 0 = {fresh.delta_is_zero()}"
    )
    print(
        f"可训练 {fresh.trainable_parameters} / 总 {fresh.total_parameters} = "
        f"{fresh.trainable_ratio:.4%}"
    )

    print("\n学习率扫（r=8，1 个 epoch = 15 步；参考模型的学习率没有绝对尺度）：")
    results: list[tuple[float, float]] = []
    for lr in (2.0, 8.0, 20.0):
        model = LoRAReferenceModel(
            ReferenceSFTModel(vocab, seed=42),
            default_reference_lora_config(r=8, lora_alpha=16),
            seed=42,
        )
        started = time.perf_counter()
        losses: list[float] = []
        for batch in train_batches:
            loss_sum, count = model.accumulate(batch)
            model.apply_update(lr)
            losses.append(loss_sum / count)
        eval_loss, _ = model.evaluate(eval_batches)
        results.append((lr, eval_loss))
        print(
            f"  lr={lr:<5g} train loss {losses[0]:.4f} -> {losses[-1]:.4f} | "
            f"评估 loss {eval_loss:.4f} | {time.perf_counter() - started:.1f}s"
        )
    best_lr = min(results, key=lambda row: row[1])[0]
    print(f"  最优学习率 {best_lr:g}（LR_SOFT_RANGE_PEFT = (1e-4, 5e-4) 是 7B LoRA 的经验区间，"
          "对参考模型相差四个数量级——照抄只会得到一条水平的 loss 曲线）")

    print(f"\n完整训练（r=8，lr={best_lr:g}，{EPOCHS} 个 epoch = 30 步，与 day050 全参同预算）：")
    model = LoRAReferenceModel(
        ReferenceSFTModel(vocab, seed=42),
        default_reference_lora_config(r=8, lora_alpha=16),
        seed=42,
    )
    started = time.perf_counter()
    losses = []
    for _ in range(EPOCHS):
        for batch in train_batches:
            loss_sum, count = model.accumulate(batch)
            model.apply_update(best_lr)
            losses.append(loss_sum / count)
    eval_loss, tokens = model.evaluate(eval_batches)
    print(
        f"  loss {losses[0]:.4f} -> {losses[-1]:.4f} | 评估 loss {eval_loss:.4f}"
        f"（{tokens} 个监督 token）| 耗时 {time.perf_counter() - started:.1f}s"
    )
    info = describe_delta(*model.layer.snapshot_matrices(), model.config.scaling)
    print(
        f"  ΔW 画像：秩 {int(info['numerical_rank'])}（上界 {int(info['rank_upper_bound'])}）| "
        f"Frobenius {info['frobenius']:.4f} | 最大元素 {info['max_abs']:.6f}"
    )
    return model, eval_loss, best_lr


def section_4_merge(model: LoRAReferenceModel, eval_batches, eval_loss: float) -> None:
    banner("[4] 合并：适配器消失，模型行为逐位不变")
    vocab = model.vocab_size
    merged = model.merge()
    same = all(merged.logits(ctx) == model.logits(ctx) for ctx in range(vocab))
    merged_loss, tokens = merged.evaluate(eval_batches)
    print(f"合并后与适配器模型的 logits 逐位一致（全部 {vocab} 个上下文）：{same}")
    print(
        f"合并后评估 loss {merged_loss:.10f} | 适配器模型 {eval_loss:.10f} | "
        f"差值 {abs(merged_loss - eval_loss):.2e}（{tokens} 个监督 token）"
    )
    print("合并的代价是增量被固化：想换一个适配器就得重新合并（day052 会做落盘与验证流程）。")


def section_5_qlora(
    tokenizer: CharTokenizer, train_batches, eval_batches, best_lr: float, lora_eval: float
) -> None:
    banner("[5] QLoRA：4-bit 基座 + bf16 适配器")
    vocab = tokenizer.vocab_size
    config = QLoRAConfig()
    print(f"量化配置：{config.describe()}")
    started = time.perf_counter()
    quantized_base, report = quantize_reference_model(ReferenceSFTModel(vocab, seed=42), config)
    print(f"量化报告：{report.summary_line()}")
    print(
        f"  存储 {report.total_bytes} B（码点 {report.code_bytes} + 常数 {report.constant_bytes}）"
        f" vs 16-bit {report.fp16_bytes} B → 省 {report.savings_vs_fp16:.2%}"
        f" | 量化耗时 {time.perf_counter() - started:.2f}s"
    )
    print(
        f"  解析式 {report.analytic_bytes_per_parameter:.6f} B/参数 vs 实测 "
        f"{report.measured_bytes_per_parameter:.6f}（差 "
        f"{report.measured_bytes_per_parameter - report.analytic_bytes_per_parameter:.6f}，"
        "来自 ceil 取整）"
    )
    model = LoRAReferenceModel(
        quantized_base, default_reference_lora_config(r=8, lora_alpha=16), seed=42
    )
    losses = []
    for _ in range(EPOCHS):
        for batch in train_batches:
            loss_sum, count = model.accumulate(batch)
            model.apply_update(best_lr)
            losses.append(loss_sum / count)
    qlora_eval, _ = model.evaluate(eval_batches)
    print(
        f"QLoRA r=8 lr={best_lr:g}：loss {losses[0]:.4f} -> {losses[-1]:.4f} | "
        f"评估 loss {qlora_eval:.4f}"
    )
    print(
        f"对照 bf16 基座 LoRA {lora_eval:.4f} → 退化 {qlora_eval - lora_eval:+.4f}"
        f"（{(qlora_eval - lora_eval) / lora_eval:+.2%}）"
    )
    print("注意：适配器自始至终是 bf16——「QLoRA 用 4-bit 训练」这个说法是不准确的。")


def section_6_memory() -> None:
    banner("[6] 显存算术：全参 / LoRA / QLoRA")
    for name, spec in MODEL_SPECS.items():
        print(f"-- {name}")
        print(f"   {describe_model(spec)}")
    lora_config = LoRAConfig(r=8, lora_alpha=16, target_modules="attention_all")
    for name in ("llama-2-7b", "qwen3-0.6b"):
        spec = MODEL_SPECS[name]
        plans = compare_strategies(spec, lora_config=lora_config)
        print(f"-- {name} / attention_all / r=8（不含激活显存）")
        for row in savings_table(plans):
            extra = f" | 省 {row['savings_factor']}×" if "savings_factor" in row else ""
            bits = row["base_bits_per_parameter"]
            print(
                f"   {row['strategy']:5s} 可训练 {row['trainable_parameters']:>12,} | "
                f"{format_bytes(row['total_bytes'])} | 基座 {bits} bit/参数{extra}"
            )
        for plan in plans:
            print(f"     最少可用卡（{plan.strategy}）：{minimal_device(plan.as_gib)}")
    print("\n单卡适配表（llama-2-7b / attention_all / r=8）：")
    for row in device_fit_table(
        compare_strategies(MODEL_SPECS["llama-2-7b"], lora_config=lora_config)
    ):
        fits = " ".join(f"{key}={'√' if value else '×'}" for key, value in row["fits"].items())
        print(f"   {row['device']:24s} {fits}")
    print("   （只比较权重 + 梯度 + 优化器状态；激活显存另算，「放得下」不等于「跑得起来」）")


def section_7_scripts() -> None:
    banner("[7] 生成真实训练脚本（peft / bitsandbytes）")
    args = SFTTrainingArgs(
        learning_rate=2e-4,
        num_train_epochs=2.0,
        max_length=320,
        save_strategy="no",
        eval_strategy="no",
    )
    lora = LoRAConfig(r=8, lora_alpha=16, target_modules="attention")
    qlora = QLoRAConfig()
    plan = plan_lora(MODEL_SPECS["qwen3-0.6b"], lora)
    trainable = plan.trainable_parameters
    scripts = (
        (
            "LoRA（transformers + peft）",
            render_lora_script(args, lora, expected_trainable=trainable),
        ),
        (
            "QLoRA（4-bit 基座 + peft）",
            render_qlora_script(args, lora, qlora, expected_trainable=trainable),
        ),
    )
    for name, script in scripts:
        compile(script, f"{name}.py", "exec")
        print(f"{name}：{len(script.splitlines())} 行，compile 通过")
    print("\n关键片段：")
    for line in scripts[1][1].splitlines():
        stripped = line.strip()
        if stripped.startswith(("model = ", "bnb_config", "payload[")) or any(
            key in stripped
            for key in (
                "BitsAndBytesConfig(", "prepare_model_for_kbit_training(",
                "get_peft_model(", "model.save_pretrained(",
            )
        ):
            print(f"  {stripped}")
    print("\n安装命令：")
    for key, value in peft_dependency_commands(use_qlora=True).items():
        if key != "packages":
            print(f"  {key}: {value}")


def main() -> None:
    tokenizer, _, _, train_batches, eval_batches = build_batches()
    print(
        f"数据：词表 V={tokenizer.vocab_size} | train {len(train_batches)} 个 micro-batch"
        f"（drop_last）| eval {len(eval_batches)} 个批次"
    )
    section_1_accounting(tokenizer)
    section_2_quantization(tokenizer)
    model, lora_eval, best_lr = section_3_train(tokenizer, train_batches, eval_batches)
    section_4_merge(model, eval_batches, lora_eval)
    section_5_qlora(tokenizer, train_batches, eval_batches, best_lr, lora_eval)
    section_6_memory()
    section_7_scripts()
    banner("完成")
    print(f"LoRA r=8（lr={best_lr:g}）评估 loss {lora_eval:.4f} | 全参 30 步基线 5.3602（lr=8）")
    print("真实训练请使用第 7 节生成的脚本，并把 --model_name 换成本地可用的基座。")


if __name__ == "__main__":
    main()
