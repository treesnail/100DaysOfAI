#!/usr/bin/env python
"""day050 SFT 端到端演示：数据 → 渲染 → 编码 → mask → 训练 → 落盘 → 生成脚本.

用法（在 ``day050/源码/smart-research-agent`` 下）::

    python scripts/sft_demo.py

本脚本**不依赖任何深度学习框架**：数据来自 day048 的落盘产物，训练用
``sft/reference_model.py`` 的纯 Python bigram 模型（真实梯度下降），
因此每一步都能在本机几秒内跑完并逐项核对。

真实训练路径由第 7 节生成的 ``transformers`` + ``Trainer`` 脚本承担。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from smart_research_agent.finetune import build_dataset, dump_bundle, split_dataset
from smart_research_agent.sft import (
    IGNORE_INDEX,
    LR_SOFT_RANGE_REFERENCE,
    REFERENCE_LEARNING_RATE,
    CharTokenizer,
    ReferenceSFTModel,
    SFTTrainer,
    SFTTrainingArgs,
    checkpoint_summary,
    dependency_commands,
    encode_supervised,
    iter_batches,
    length_summary,
    load_checkpoint,
    plan_training,
    render_hf_sft_script,
    render_supervised,
    render_trl_sft_script,
    suggest_max_length,
    top_k_next,
)


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def section_1_data() -> tuple[list, list]:
    banner("[1] 数据：day048 的落盘产物（采集 → 清洗 → 切分）")
    bundle = build_dataset()
    train, eval_set = split_dataset(bundle.examples, eval_ratio=0.2, seed=42)
    print(f"清洗后 {bundle.report.kept} 条（载入 {bundle.report.total}）→ "
          f"train {len(train)} / eval {len(eval_set)}")
    with tempfile.TemporaryDirectory() as tmp:
        paths = dump_bundle(bundle, tmp)
        for key, path in paths.items():
            lines = Path(path).read_text(encoding="utf-8").strip().splitlines()
            print(f"  {key:<5} -> {Path(path).name}（{len(lines)} 行）")
    return train, eval_set


def section_2_render_encode(train: list, eval_set: list) -> tuple[CharTokenizer, list, str]:
    banner("[2] 渲染 + 编码：prompt 段 label 置 -100")
    sample = render_supervised(train[0])
    print("渲染结构（ChatML）：")
    print(
        f"  总字符 {sample.total_chars} = prompt {sample.prompt_chars} "
        f"+ 监督 {sample.supervised_chars}"
    )
    tail = sample.text[sample.prompt_chars - 30 : sample.prompt_chars]
    print(f"  前缀尾部（含 assistant 起手符）：{tail!r}")
    print(f"  监督区间前 40 字：{sample.supervised_text[:40]}...")

    rendered = [render_supervised(e) for e in list(train) + list(eval_set)]
    tokenizer = CharTokenizer.from_texts([item.text for item in rendered])
    print(f"\n字符级分词器词表：{tokenizer.vocab_size} 个 token（<pad>=0, <unk>=1）")

    encoded = encode_supervised(sample, tokenizer, max_length=512)
    print(f"\n编码：tokens={encoded.total_tokens} prompt={encoded.prompt_tokens} "
          f"监督={encoded.supervised_tokens} 占比={encoded.supervised_ratio:.2%}")
    print(f"  labels[:{encoded.prompt_tokens}] 全为 -100 -> "
          f"{all(v == IGNORE_INDEX for v in encoded.labels[: encoded.prompt_tokens])}")
    print(f"  labels[{encoded.prompt_tokens}:] 全非 -100 -> "
          f"{all(v != IGNORE_INDEX for v in encoded.labels[encoded.prompt_tokens :])}")
    print(f"  可逆性（decode(encode(prompt)) == prompt）-> "
          f"{tokenizer.decode(tokenizer.encode(sample.prompt_text)) == sample.prompt_text}")
    return tokenizer, rendered, sample.template


def section_3_lengths(tokenizer: CharTokenizer, rendered: list) -> int:
    banner("[3] 长度分位数 → max_length（本课最重要的实操教训）")
    summary = length_summary(rendered, tokenizer)
    print(summary.summary_line())
    print("\n与 day048 画像的差距：")
    print("  day048 DatasetStats 估计：平均 84.24 token/条（原始 instruction+output）")
    print(f"  渲染并分词之后实测      ：平均 {summary.mean:.1f} token/条")
    print(
        f"  差距倍数               ：{summary.mean / 84.24:.2f}×"
        "（系统提示词 + 角色标记 + 字符级分词）"
    )
    print("\nmax_length 候选与建议：")
    candidates = (("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p100（一条都不截断）", 1.0))
    for label, ratio in candidates:
        print(f"  {label:<20} = {summary.quantile(ratio):>4} token")
    suggested = suggest_max_length(rendered, tokenizer, quantile=0.95)
    strict = suggest_max_length(rendered, tokenizer, quantile=1.0)
    print(f"  suggest_max_length(p95, 32 的倍数) = {suggested}")
    print(f"  suggest_max_length(p100)          = {strict}")
    return suggested


def section_4_plan(train: list, eval_set: list, max_length: int) -> None:
    banner("[4] 训练计划：步数 / warmup / 有效批（day049 预先算过的六个数字）")
    args = SFTTrainingArgs(max_length=max_length)
    plan = plan_training(
        args,
        train_size=len(train),
        eval_size=len(eval_set),
        dataset_avg_tokens=250.9,
        dataset_max_tokens=320,
    )
    payload = {k: v for k, v in plan.to_dict().items() if k != "warnings"}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("\nlearning_rate 曲线（cosine，6 个采样点）：")
    from smart_research_agent.sft import lr_curve

    for step, value in lr_curve(args, len(train)):
        print(f"  step {step:>2} -> lr {value:.8f}")
    print("\nwarnings（每一条都对应一个真实失败模式）：")
    for message in plan.warnings:
        print(f"  - {message}")


def section_5_train(train: list, eval_set: list, max_length: int) -> tuple:
    banner("[5] 训练：参考模型（真实梯度下降，离线可跑）")
    args = SFTTrainingArgs(
        learning_rate=REFERENCE_LEARNING_RATE,
        num_train_epochs=10.0,
        max_length=max_length,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
    )
    with tempfile.TemporaryDirectory() as tmp:
        args = args.with_overrides(output_dir=tmp)
        tokenizer = CharTokenizer.from_texts(
            [render_supervised(e).text for e in list(train) + list(eval_set)]
        )
        model = ReferenceSFTModel(tokenizer.vocab_size, seed=42)
        trainer = SFTTrainer(model, tokenizer, args)
        report = trainer.fit(train, eval_set)

        print(f"参考模型：词表 {tokenizer.vocab_size}，参数 {tokenizer.vocab_size ** 2}（W）+ "
              f"{tokenizer.vocab_size}（b），纯 SGD，lr={REFERENCE_LEARNING_RATE}")
        print(f"均匀分布参考 loss = ln({tokenizer.vocab_size}) = {model.uniform_loss:.4f}")
        print(f"\n训练集编码：{report.train.summary_line()}")
        print(f"评估集编码：{report.eval.summary_line() if report.eval else '-'}")
        print(f"\n优化器步数 {report.optimizer_steps}（epochs_run={report.epochs_run}）"
              f" | 监督 token 使用量 {report.supervised_tokens_seen}")
        print("loss 轨迹：")
        for record in report.records:
            print(f"  step {record.step:>2} | epoch {record.epoch} | loss {record.loss:.4f} | "
                  f"lr {record.learning_rate:.4f} | 监督 token {record.supervised_tokens} | "
                  f"micro-batch {record.micro_batches}")
        print(f"\nloss {report.initial_loss:.4f} -> {report.final_loss:.4f} "
              f"（降幅 {report.loss_drop_ratio:+.2%}）")
        print(f"评估 loss {report.eval_loss:.4f} | 困惑度 {report.eval_perplexity:.2f}"
              f"（均匀分布 = {tokenizer.vocab_size}）")
        print(f"耗时 {report.elapsed_seconds:.2f}s")

        print("\n检查点：")
        for path in report.checkpoints:
            print(f"  写过 {Path(path).name}")
        print(f"  落盘后实际保留：{sorted(p.name for p in Path(args.output_dir).iterdir())}"
              f"（save_total_limit={args.save_total_limit}，最旧的中间检查点已被清理）")
        print(f"  {checkpoint_summary(Path(report.checkpoints[-1]))}")
        loaded = load_checkpoint(report.checkpoints[-1])
        print(f"  回读：args.max_length={loaded['args'].max_length} "
              f"updates={loaded['state'].updates} records={len(loaded['records'])} "
              f"metrics.total_steps={loaded['metrics']['total_steps']}")

        print("\n模型确实学到了东西（上下文 token 的下一 token 概率分布）：")
        context = tokenizer.encode("答")[0]
        top3 = top_k_next(model, context, k=3)
        print(f"  上下文 {context!r} 的 top-3：(id, prob) = "
              f"{[(i, round(p, 4)) for i, p in top3]}")
        print(f"  均匀分布下每个 token 的概率 = {1 / tokenizer.vocab_size:.6f}")
        print(f"  最高概率 / 均匀概率 = {top3[0][1] * tokenizer.vocab_size:.2f}×")
        return report, model, tokenizer, args


def section_6_mask_ablation(train: list, eval_set: list, max_length: int) -> None:
    banner("[6] 对照实验：屏蔽 prompt vs 不屏蔽（都用同一把尺子评估）")
    args = SFTTrainingArgs(
        learning_rate=REFERENCE_LEARNING_RATE,
        num_train_epochs=10.0,
        max_length=max_length,
    )
    tokenizer = CharTokenizer.from_texts(
        [render_supervised(e).text for e in list(train) + list(eval_set)]
    )
    # 统一的"尺子"：评估集一律按**屏蔽 prompt** 的口径编码，衡量的是
    # "模型在答案 token 上的预测能力"，与训练时用哪种口径无关。
    ruler_trainer = SFTTrainer(
        ReferenceSFTModel(tokenizer.vocab_size, seed=42), tokenizer, args, mask_prompt=True
    )
    ruler_batches = iter_batches(
        ruler_trainer.encode(eval_set)[0], batch_size=args.per_device_eval_batch_size
    )

    results: dict[str, dict[str, float]] = {}
    for label, mask in (("屏蔽 prompt（标准 SFT）", True), ("不屏蔽 prompt", False)):
        model = ReferenceSFTModel(tokenizer.vocab_size, seed=42)
        trainer = SFTTrainer(model, tokenizer, args, mask_prompt=mask)
        report = trainer.fit(train, eval_set, save_at_end=False)
        # 用同一把尺子重测：评估集按屏蔽口径编码
        ruler_loss, ruler_tokens = model.evaluate(ruler_batches)
        results[label] = {
            "train_loss": report.final_loss,
            "train_reported": report.final_loss,
            "ruler_loss": ruler_loss,
            "supervised_ratio": report.train.supervised_ratio,
            "tokens": ruler_tokens,
        }
        print(f"\n{label}")
        print(f"  训练集监督 token 占比：{report.train.supervised_ratio:.2%}"
              f"（监督 {report.train.supervised_tokens} / 共 {report.train.total_tokens}）")
        print(f"  训练报告 loss（各自口径）：{report.initial_loss:.4f} -> {report.final_loss:.4f}")
        print(f"  **统一尺子**下的答案 loss：{ruler_loss:.4f}"
              f"（{ruler_tokens} 个监督 token 上求均值）")

    masked = results["屏蔽 prompt（标准 SFT）"]["ruler_loss"]
    unmasked = results["不屏蔽 prompt"]["ruler_loss"]
    print(f"\n结论：统一尺子下 屏蔽={masked:.4f} / 不屏蔽={unmasked:.4f}，"
          f"屏蔽{'更好' if masked < unmasked else '更差'} "
          f"（差值 {abs(masked - unmasked):.4f}）")
    print("原因：不屏蔽时 prompt 段的梯度没有被屏蔽，"
          "被用去预测「用户会怎么提问」——而提问在推理时是给定的输入。")


def section_7_scripts(args: SFTTrainingArgs) -> None:
    banner("[7] 生成真实训练脚本（transformers + Trainer / TRL）")
    hf_script = render_hf_sft_script(args)
    trl_script = render_trl_sft_script(args)
    for name, script in (("transformers + Trainer", hf_script), ("TRL SFTTrainer", trl_script)):
        compile(script, f"{name}.py", "exec")
        print(f"{name}: {len(script.splitlines())} 行，compile 通过")
    print("\nHF 脚本里的关键片段：")
    for line in hf_script.splitlines():
        stripped = line.strip()
        if any(
            key in stripped
            for key in (
                'IGNORE_INDEX = -100',
                'labels": [IGNORE_INDEX] * len(kept_prompt)',
                'kept_prompt = prompt_ids[-budget:]',
                '"max_length"',
                'eval_strategy',
                'learning_rate',
            )
        ):
            print(f"  {stripped}")
    print("\n安装命令：")
    for key, value in dependency_commands(use_trl=True).items():
        if key != "packages":
            print(f"  {key}: {value}")
    print("\nTRL 版本里三个 SFT 专属开关（与本课实现的对应关系）：")
    for line in trl_script.splitlines():
        stripped = line.strip()
        if any(k in stripped for k in ('"max_length"', '"packing"', '"assistant_only_loss"')):
            print(f"  {stripped}")


def main() -> None:
    train, eval_set = section_1_data()
    tokenizer, rendered, template = section_2_render_encode(train, eval_set)
    max_length = section_3_lengths(tokenizer, rendered)
    section_4_plan(train, eval_set, max_length)
    _, _, _, demo_args = section_5_train(train, eval_set, max_length)
    section_6_mask_ablation(train, eval_set, max_length)
    section_7_scripts(demo_args)
    banner("完成")
    print(f"模板 {template} | 建议 max_length {max_length} | 学习率区间（参考模型）"
          f" {LR_SOFT_RANGE_REFERENCE}")
    print("真实训练请使用第 7 节生成的脚本，并把 --model_name 换成本地可用的基座。")


if __name__ == "__main__":
    main()
