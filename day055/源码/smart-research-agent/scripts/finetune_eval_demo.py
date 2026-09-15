#!/usr/bin/env python
"""微调评估 demo（M5-D5）：一条命令跑完 day053 的全部数字.

离线、确定性、零 GPU：脚本化两臂 + 参考模型上的真实 LoRA 训练。
运行::

    python scripts/finetune_eval_demo.py

输出六段：评估集画像 → 切分与泄漏体检 → 两臂配对比较 → 真实模型的
白盒探针（含过拟合对照组）→ 适配器清单与评估报告（含绑定核对）→ 预算外推。
"""

from __future__ import annotations

import math
import tempfile
import time
from pathlib import Path

from smart_research_agent.finetune_eval import (
    audit_suite,
    build_report,
    build_suite,
    compare_probes,
    compare_runs,
    detect_leakage,
    encode_probe_batches,
    estimate_eval_seconds,
    probe_items,
    read_report,
    render_markdown,
    run_suite,
    scripted_arm,
    split_suite,
    suite_stats,
    verify_manifest_binding,
    write_report,
)
from smart_research_agent.peft import (
    build_manifest,
    default_reference_lora_config,
    save_adapter,
    train_lora_reference,
    write_manifest,
)
from smart_research_agent.sft.args import SFTTrainingArgs
from smart_research_agent.sft.encoding import Batch, CharTokenizer
from smart_research_agent.sft.reference_model import REFERENCE_LEARNING_RATE, ReferenceSFTModel

#: 标定出来的学习率：本课在 (epochs, lr) 的 12 个组合上扫过一遍，
#: 这个取值在评估集上给出最低的困惑度（见教程第二章）。
CALIBRATED_LEARNING_RATE = 2.0

#: 训练轮数（评估集困惑度在 10 轮时最好；再多就只降训练集困惑度了）
CALIBRATED_EPOCHS = 10

#: 过拟合对照组：与 day050 全参 SFT 的标定值一致，但步数远不够它收敛
OVERFIT_LEARNING_RATE = REFERENCE_LEARNING_RATE

#: 发散对照组：证明"学习率没有绝对尺度"
DIVERGING_LEARNING_RATE = 20.0

#: 参考模型的基座名**前缀**（写进适配器清单，回答"这个适配器配哪个基座"）.
#:
#: 词表大小作为后缀拼进去（``char-bigram-v{词表}``）：``CharTokenizer`` 的词表是
#: 从语料现造的，**改一个字都可能让词表变化**（本课实测：把一条用例的参考
#: 答案改了几个字，词表从 421 变成 422）。名字里带上这个数字，是为了让
#: "适配器配的是哪个词表的基座"这件事在清单里**可见**——一个只写
#: ``char-bigram`` 的基座名，配错词表的适配器不会报错，只会给出乱码。
REFERENCE_BASE_MODEL_PREFIX = "char-bigram-v"


def _fmt(value: float) -> str:
    """格式化困惑度 / loss：发散时真的会出现 ``inf`` 与 ``nan``，不能直接 ``%.4f``.

    ``lr=20`` 那一行就是活例子——**让它以 ``inf`` 的样子被打出来，
    比把它藏起来更能说明"学习率没有绝对尺度"**。
    """
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf"
    if abs(value) >= 1e6:
        return f"{value:.3e}"
    return f"{value:.4f}"


def batches_of(items, tokenizer) -> list[Batch]:
    """把用例编码成单样本 Batch（**prompt 段被 -100 屏蔽**，day050 的纪律）.

    直接复用 ``finetune_eval.encode_probe_batches``：探针侧的微调数据与
    "模型看到的那段文本"必须是同一份，否则"训练时见过的"与"探针量的"
    不是一个东西——这正是 day052 第六章那类"配置转发改了语义"的坑。
    """
    return encode_probe_batches(items, tokenizer)


def main() -> None:
    # ---------------------------------------------------------------- 1. 评估集
    suite = build_suite()
    stats = suite_stats(suite)
    print("=" * 78)
    print("① 评估集画像")
    print("=" * 78)
    print(f"指纹 {stats['fingerprint']} | 共 {stats['total']} 条")
    print(f"桶分布 {stats['by_bucket']}")
    print(f"难度分布 {stats['by_difficulty']}")
    print(
        f"事实点合计 {stats['required_facts_total']} | 禁项合计 "
        f"{stats['forbidden_facts_total']} | 平均结构负载 {stats['mean_structural_load']}"
        f" | 平均字数预算 {stats['mean_char_budget']}"
    )
    print(f"难度体检（声明 ≠ 推导）: {audit_suite(suite) or '无差异'}")
    for item in suite[:3]:
        print("  ", item.summary_line())

    # ------------------------------------------------------------ 2. 切分与泄漏
    train_items, eval_items = split_suite(suite)
    leakage = detect_leakage(train_items, eval_items)
    print()
    print("=" * 78)
    print("② 切分与泄漏体检")
    print("=" * 78)
    print(f"训练集 {len(train_items)} 条：{[item.id for item in train_items]}")
    print(f"评估集 {len(eval_items)} 条：{[item.id for item in eval_items]}")
    difficulty_counts = {
        level: sum(1 for i in eval_items if i.difficulty == level)
        for level in ("easy", "normal", "hard")
    }
    print(f"评估集难度分布：{difficulty_counts}")
    print(
        f"泄漏体检：{leakage['ngram']}-gram Jaccard 最大 {leakage['max_jaccard']}"
        f"（阈值 {leakage['threshold']}）| 可疑对 {len(leakage['suspicious_pairs'])} 个"
        f" | 通过 {leakage['passed']}"
    )

    # -------------------------------------------------------------- 3. 两臂比较
    baseline_run = run_suite(eval_items, scripted_arm(arm="baseline"), name="scratch-baseline")
    finetuned_run = run_suite(eval_items, scripted_arm(arm="finetuned"), name="lora-finetuned")
    comparison = compare_runs(baseline_run, finetuned_run, max_regression=0.0)
    print()
    print("=" * 78)
    print("③ 两臂配对比较（脚本化对照臂，非模型输出）")
    print("=" * 78)
    print(baseline_run.summary_line())
    print(finetuned_run.summary_line())
    print()
    print("逐条结果：")
    for outcome in finetuned_run.outcomes:
        marker = "合格" if outcome.passed else "不合格"
        print(
            f"  {outcome.item_id:9s} {outcome.bucket:12s} {outcome.difficulty:7s} "
            f"{marker} 总分 {outcome.score:.4f} {outcome.failure_reason()}"
        )
    print()
    print("分桶：")
    for bucket in comparison.buckets:
        print("  ", bucket.summary_line())
    print("分难度：")
    for bucket in comparison.difficulties:
        print("  ", bucket.summary_line())
    print()
    print(comparison.mcnemar.summary_line())
    print(comparison.bootstrap.summary_line())
    print("门禁：", "通过" if comparison.passed else f"未通过 {comparison.regressions}")

    # ------------------------------------------------------ 4. 真实模型的白盒探针
    tokenizer = CharTokenizer.from_texts(
        [item.instruction + item.reference for item in suite]
    )
    base_model = ReferenceSFTModel(tokenizer.vocab_size, seed=42)
    train_batches = batches_of(train_items, tokenizer)
    eval_batches = batches_of(eval_items, tokenizer)
    print()
    print("=" * 78)
    print("④ 真实模型探针（参考模型 + 真实 LoRA 梯度的白盒信号）")
    print("=" * 78)
    print(
        f"词表 {tokenizer.vocab_size} | 训练批次 {len(train_batches)}"
        f" | 评估批次 {len(eval_batches)}"
    )
    base_eval_loss, base_tokens = base_model.evaluate(eval_batches)
    print(f"基座评估 loss {base_eval_loss:.6f}（{base_tokens} 个监督 token）")
    print()
    print("| 配置 | 步数 | 训练集困惑度 | 评估集困惑度 | 评估 loss | 泛化间隙 |")
    print("|------|------|--------------|--------------|-----------|----------|")
    base_train_perplexity = probe_items(base_model, tokenizer, train_items).perplexity
    base_eval_perplexity = probe_items(base_model, tokenizer, eval_items).perplexity
    print(
        f"| 基座（未微调） | 0 | {_fmt(base_train_perplexity)} "
        f"| {_fmt(base_eval_perplexity)} "
        f"| {_fmt(base_eval_loss)} | — |"
    )
    for label, learning_rate in (
        ("标定配置 lr=2.0", CALIBRATED_LEARNING_RATE),
        ("过拟合对照 lr=8.0", OVERFIT_LEARNING_RATE),
        ("发散对照 lr=20.0", DIVERGING_LEARNING_RATE),
    ):
        model, losses, _ = train_lora_reference(
            base_model,
            train_batches,
            config=default_reference_lora_config(),
            learning_rate=learning_rate,
            epochs=CALIBRATED_EPOCHS,
            seed=42,
        )
        train_probe = probe_items(model, tokenizer, train_items)
        eval_probe = probe_items(model, tokenizer, eval_items)
        loss, _ = model.evaluate(eval_batches)
        gap = compare_probes(
            probe_items(base_model, tokenizer, train_items),
            train_probe,
        )["perplexity_delta"] - compare_probes(
            probe_items(base_model, tokenizer, eval_items), eval_probe
        )["perplexity_delta"]
        print(
            f"| {label} | {len(losses)} | {_fmt(train_probe.perplexity)} "
            f"| {_fmt(eval_probe.perplexity)} | {_fmt(loss)} | {_fmt(gap)} |"
        )
        if abs(learning_rate - CALIBRATED_LEARNING_RATE) < 1e-9:
            calibrated_model = model
            calibrated_steps = len(losses)
            calibrated_eval_probe = eval_probe
            calibrated_train_probe = train_probe
    print()
    print("标定配置的逐桶探针（评估集）：")
    for bucket, row in calibrated_eval_probe.by_bucket().items():
        print(
            f"  {bucket:12s} 条数 {row['items']} 监督 {row['supervised']:4d} "
            f"困惑度 {_fmt(row['perplexity'])} 命中率 {row['accuracy']:.4f}"
        )

    # --------------------------------------------- 5. 适配器清单 + 评估报告 + 绑定
    with tempfile.TemporaryDirectory() as workdir:
        adapter_dir = Path(workdir) / "adapter-final"
        save_adapter(
            adapter_dir,
            model=calibrated_model,
            args=SFTTrainingArgs(
                learning_rate=CALIBRATED_LEARNING_RATE,
                num_train_epochs=float(CALIBRATED_EPOCHS),
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
            ),
            step=calibrated_steps,
            boundary="finetune-eval",
        )
        manifest = build_manifest(
            adapter_dir,
            base_model=f"{REFERENCE_BASE_MODEL_PREFIX}{tokenizer.vocab_size}",
            tags={"suite": stats["fingerprint"]},
        )
        write_manifest(adapter_dir, manifest)
        report = build_report(
            run=finetuned_run,
            suite=stats,
            manifest=manifest,
            comparison=comparison,
            min_pass_rate=0.5,
            notes=(
                "两臂为脚本化对照臂（确定性函数），不是真实模型输出",
                "白盒探针在参考模型（char-bigram）上真实训练得到",
            ),
        )
        verify_manifest_binding(report, manifest)
        report_path = write_report(
            Path(workdir) / "report.json",
            report,
            markdown_path=Path(workdir) / "report.md",
        )
        reloaded = read_report(report_path)
        print()
        print("=" * 78)
        print("⑤ 适配器清单与评估报告")
        print("=" * 78)
        print("清单：", manifest.summary_line())
        print("报告：", reloaded.summary_line())
        print("门禁：", {name: gate["passed"] for name, gate in reloaded.gates.items()})
        print("绑定核对：通过（哈希 / 基座 / 步数三项一致）")
        print()
        print("报告 markdown 前 24 行：")
        for line in render_markdown(reloaded).splitlines()[:24]:
            print("  ", line)

    # ---------------------------------------------------------------- 6. 预算外推
    print()
    print("=" * 78)
    print("⑥ 预算外推（线性，不外推并行效率）")
    print("=" * 78)
    started = time.perf_counter()
    probe_items(calibrated_model, tokenizer, eval_items)
    probe_seconds = time.perf_counter() - started
    per_item = probe_seconds / len(eval_items)
    print(
        f"白盒探针实测：{len(eval_items)} 条 / {probe_seconds * 1000:.1f} 毫秒"
        f"（单条 {per_item * 1000:.2f} 毫秒，含每条约 70 次 bigram 前向）"
    )
    print(
        f"脚本化两臂实测：{finetuned_run.total} 条 / {finetuned_run.wall_seconds * 1000:.3f} 毫秒"
        "（纯文本函数，不计入算力预算——接上真模型后这里换成推理耗时）"
    )
    for target in (100, 1000):
        seconds = estimate_eval_seconds(
            seconds_per_item=per_item, target_items=target, concurrency=4
        )
        print(f"按白盒探针口径外推到 {target} 条 / 4 并发：{seconds:.4f} 秒")
    print()
    print(f"评估集指纹 {stats['fingerprint']} | 标定配置评估集困惑度 "
          f"{calibrated_eval_probe.perplexity:.4f}（训练集 "
          f"{calibrated_train_probe.perplexity:.4f}）")


if __name__ == "__main__":
    main()
