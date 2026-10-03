#!/usr/bin/env python
"""DPO 对齐实践 demo（M5-D7）：一条命令跑完 day055 的全部数字.

离线、确定性、零 GPU：偏好数据从磁盘上的种子样本构造，损失与梯度是纯函数，
步数算术不依赖任何框架。运行::

    python scripts/dpo_demo.py

输出五段：三种损失的对照表与起点自检 → 解析梯度 vs 中心差分 → 训练配置与
步数算术（含 warmup 退化检查）→ 偏好数据构造报告 → TRL 数据集落盘回合。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from smart_research_agent.dpo import (
    LOSS_TYPES,
    DPOTrainingConfig,
    build_preferences,
    dataset_report,
    load_seed_examples,
    loss_gradient,
    loss_table,
    numeric_gradient,
    read_safety_pairs,
    read_trl_dataset,
    safety_category_table,
    to_trl_rows,
    warmup_floor,
    write_trl_dataset,
    zero_margin_losses,
)

BETA = 0.1


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def demo_loss_table() -> None:
    section("① 三种损失函数的对照表与起点自检值")
    for row in loss_table(beta=BETA):
        print(
            f"{row['loss_type']:8s} {row['formula']:22s} "
            f"起点={row['zero_margin_loss']:.6f} | {row['beta_role']}"
        )
    print(f"\n起点期望值（beta={BETA}）："
          f"{ {k: round(v, 6) for k, v in zero_margin_losses(beta=BETA).items()} }")
    print("其中 sigmoid 的起点值是 ln2，可与 day054 的 ZERO_MARGIN_LOSS 直接对照。")


def demo_gradient_check() -> None:
    section("② 解析梯度 vs 中心差分（数值独立的第二意见）")
    print(f"{'loss':8s} {'margin':>8s} {'解析':>16s} {'数值':>16s} {'误差':>12s}")
    worst = 0.0
    for name in LOSS_TYPES:
        for margin in (-2.0, 0.0, 1.5, 6.0):
            analytic = loss_gradient(name, margin, beta=BETA)
            numeric = numeric_gradient(name, margin, beta=BETA)
            error = abs(analytic - numeric)
            worst = max(worst, error)
            print(f"{name:8s} {margin:8.1f} {analytic:16.10f} {numeric:16.10f} {error:12.2e}")
    print(f"\n最大误差 {worst:.2e}——量级在浮点噪声内，说明梯度式子与实现一致。")


def demo_training_plan() -> None:
    section("③ 训练配置与步数算术（开训前就能算准）")
    cfg = DPOTrainingConfig()
    print(cfg.summary_line())
    print(f"\nTRL DPOConfig 投影（键名逐字对齐）：{sorted(cfg.to_dpo_config())}")

    train_pairs = 16
    plan = cfg.step_plan(train_pairs)
    print(f"\n{train_pairs} 条偏好对的训练计划：")
    for key in (
        "micro_batches_per_epoch",
        "optimizer_steps_per_epoch",
        "complete_epochs",
        "total_steps",
        "warmup_steps",
        "effective_batch_size",
        "pairs_consumed",
        "pairs_dropped_per_epoch",
        "evaluations",
    ):
        print(f"  {key:28s} = {plan[key]}")
    print(f"  warmup_degenerate            = {plan['warmup_degenerate']}")
    print(
        f"\nwarmup 退化判据：{plan['total_steps']} 步要让 warmup 生效，"
        f"warmup_ratio 至少需 {warmup_floor(plan['total_steps']):.4f}"
        f"（默认 {cfg.warmup_ratio} 必然退化成 0 步）"
    )


def demo_preferences() -> None:
    section("④ 偏好数据构造（金标准必须真的是最好的）")
    examples = load_seed_examples()
    pairs, report = build_preferences(examples)
    print(f"种子样本 {len(examples)} 条 → {report.summary_line()}")
    print(f"\n构造报告：{report.to_dict()}")
    print(f"\n数据集画像：{dataset_report(pairs)}")

    print("\n第一条偏好对的三个字段（TRL 只认这三列）：")
    row = to_trl_rows(pairs[:1])[0]
    for key in ("prompt", "chosen", "rejected"):
        value = row[key].replace("\n", " ")
        print(f"  {key:8s}: {value[:70]}{'…' if len(value) > 70 else ''}")

    print("\n安全偏好集（由 day031 红队 payload 派生）：")
    safe = read_safety_pairs()
    print(f"  {len(safe)} 对，类别分布 {safety_category_table(safe)}")


def demo_trl_roundtrip() -> None:
    section("⑤ TRL 数据集落盘与回读（含 meta 的取舍）")
    pairs, _ = build_preferences(load_seed_examples())
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "dpo_train.jsonl"
        write_trl_dataset(path, pairs)
        back = read_trl_dataset(path)
        print(f"写入 {path.name}：{len(back)} 行")
        print(f"列名：{sorted(back[0])}")
        print(f"首行 chosen 前 40 字：{back[0]['chosen'][:40]}…")
        print("\n默认只写 TRL 认识的三个列——多一列都会被 json 直读路径带进去；")
        print("需要溯源时用 include_meta=True 追加元信息。")


def main() -> None:
    demo_loss_table()
    demo_gradient_check()
    demo_training_plan()
    demo_preferences()
    demo_trl_roundtrip()
    print()
    print("=" * 72)
    print("day055 全部数字演示完成（离线、确定性、零 GPU）")
    print("=" * 72)


if __name__ == "__main__":
    main()
