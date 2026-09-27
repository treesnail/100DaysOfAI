"""领域数据准备与增强一键演示（day057 · M5-D8）.

跑法::

    cd day057/源码/smart-research-agent
    python scripts/domain_data_demo.py

九段输出，全部离线、确定、零 GPU、零网络：

1. **质量维度表与增强算子表**——由代码生成（``render_dimension_table`` /
   ``render_augment_table``），文档与实现同源；
2. **采集**：用 day048 的三个数据源载入**原始**样本（不清洗）；
3. **六阶段流水线**：clean → quality → near_dedup → mixing → augment → freeze，
   逐阶段打印进/出计数（``dropped`` 与 ``added`` 分开记，增强阶段不会印出负数）；
4. **数据集清单**：版本、指纹、来源分布、配比、质量分布、参数快照；
5. **配比连锁削减实证**：把上限从 0.5 收到 0.4，看不动点迭代怎么把
   ``16 : 5 : 16`` 收敛成 ``10 : 5 : 10``，以及代价是多少条；
6. **不可达的配比**：``group_by="safety"`` + 0.4 会一路崩到 2 条，
   并显式报出 ``feasible=False``；
7. **配比缺口**：等权目标下"谁多了、谁少了"，这是增强的靶子；
8. **近重复标定**：七组文本对的精确 Jaccard 与 MinHash 估计对照，
   以及阈值 0.6/0.7/0.8/0.9 的敏感性表；
9. **增强不变式与增量合并**：证明 ``output`` 逐字不动、``source/license``
   原样继承，并演示跨批次增量去重；
10. **脏样本的质量判定**：占位符样本与车轱辘话样本各自被哪一维挡下；
11. **落盘**：``train.jsonl`` / ``eval.jsonl`` / ``manifest.json``，
    并回读清单验证指纹与规模。
"""

from __future__ import annotations

import json
from pathlib import Path

from smart_research_agent.domain_data import (
    DEFAULT_OPS,
    DEFAULT_NEAR_DUP_THRESHOLD,
    DomainDataPipeline,
    NearDuplicateIndex,
    apply_mix_plan,
    augment_dataset,
    deficit_report,
    estimate_jaccard,
    exact_key,
    filter_by_quality,
    fingerprint_example,
    group_counts,
    is_augmented,
    jaccard_exact,
    plan_mixing,
    render_augment_table,
    render_dimension_table,
    shingles,
)
from smart_research_agent.finetune.schema import TrainingExample

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "domain_data"


def banner(title: str) -> None:
    """打印一段分隔标题."""
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def probe_jaccard(left: str, right: str) -> tuple[float, float, bool]:
    """返回（精确 Jaccard, MinHash 估计, 精确键是否相同）."""
    example_left = TrainingExample(instruction=left, output="占位答案。")
    example_right = TrainingExample(instruction=right, output="占位答案。")
    fp_left = fingerprint_example(example_left)
    fp_right = fingerprint_example(example_right)
    return (
        jaccard_exact(shingles(left), shingles(right)),
        estimate_jaccard(fp_left.signature, fp_right.signature),
        fp_left.key == fp_right.key,
    )


def main() -> None:
    pipeline = DomainDataPipeline()

    # ------------------------------------------------------------------ 1
    banner("1. 质量维度表（由代码生成）与增强算子表")
    print(render_dimension_table())
    print(render_augment_table())
    print(f"缺省启用的算子：{DEFAULT_OPS}")

    # ------------------------------------------------------------------ 2
    banner("2. 采集：三个数据源的**原始**样本（不清洗）")
    raw, raw_counts = pipeline.load_raw_batch()
    print(f"raw_counts = {raw_counts}，合计 {len(raw)} 条")

    # ------------------------------------------------------------------ 3
    banner("3. 六阶段流水线")
    run = pipeline.run(raw, version=1, raw_counts=raw_counts)
    for line in run.summary_lines():
        print(" ", line)
    print()
    for stage in run.manifest.stages:
        print(
            f"  [{stage.name:<11}] 进 {stage.total_in:>3} → 出 {stage.kept:>3} | "
            f"丢弃 {stage.dropped:>3} 新增 {stage.added:>3} | "
            f"{json.dumps(stage.detail, ensure_ascii=False)}"
        )

    # ------------------------------------------------------------------ 4
    banner("4. 数据集清单（manifest）")
    print(run.manifest.render_markdown())

    # ------------------------------------------------------------------ 5
    banner("5. 配比连锁削减：上限 0.5 → 0.4")
    cleaned, _ = pipeline.cleaner.run(raw)
    print(f"清洗后分组（来源口径）：{group_counts(cleaned, group_by='source')}")
    for ratio in (0.5, 0.4):
        plan = plan_mixing(cleaned, group_by="source", max_ratio=ratio)
        print(f"  max_ratio={ratio}: {json.dumps(plan.to_dict(), ensure_ascii=False)}")
    plan_04 = plan_mixing(cleaned, group_by="source", max_ratio=0.4)
    mixed, report = apply_mix_plan(cleaned, plan_04)
    print(f"  按 0.4 执行：{report.summary_line()}")

    # ------------------------------------------------------------------ 6
    banner("6. 一个**不可达**的配比：safety 口径 + 0.4")
    plan_safety = plan_mixing(cleaned, group_by="safety", max_ratio=0.4)
    print(json.dumps(plan_safety.to_dict(), ensure_ascii=False, indent=2))

    # ------------------------------------------------------------------ 7
    banner("7. 配比缺口（等权目标）：增强的靶子")
    deficit = deficit_report(
        run.examples,
        group_by="source",
        weights={"seed": 1.0, "eval/agent_tasks": 1.0, "eval/redteam_cases": 1.0},
    )
    print(json.dumps(deficit.to_dict(), ensure_ascii=False, indent=2))

    # ------------------------------------------------------------------ 8
    banner("8. 近重复标定：精确 Jaccard vs MinHash 估计")
    pairs = (
        ("逐字相同", "如何评估 RAG 的检索质量？", "如何评估 RAG 的检索质量？"),
        ("加 2 字前缀", "如何评估 RAG 的检索质量？", "请问如何评估 RAG 的检索质量？"),
        ("加 6 字前缀", "如何评估 RAG 的检索质量？", "请帮我看看：如何评估 RAG 的检索质量？"),
        ("尾部追加一句", "如何评估 RAG 的检索质量？", "如何评估 RAG 的检索质量？请给出指标。"),
        ("同义改写（换疑问词）", "如何评估 RAG 的检索质量？", "怎么评估 RAG 的检索质量？"),
        ("同义改写（换两个词）", "如何评估 RAG 的检索质量？", "怎么评价 RAG 的检索效果？"),
        ("完全不同的问", "如何评估 RAG 的检索质量？", "DPO 的 beta 是什么？"),
    )
    print(f"{'场景':<20}{'精确 Jaccard':>14}{'MinHash 估计':>16}{'精确键相同':>12}")
    for name, left, right in pairs:
        exact, estimated, same_key = probe_jaccard(left, right)
        print(f"{name:<20}{exact:>14.4f}{estimated:>16.4f}{str(same_key):>12}")

    print()
    print("阈值敏感性（37 条真实样本 + 16 条逐字复制 + 16 条前缀复制 = 69 条）：")
    probe_corpus = list(cleaned)
    probe_corpus += [
        TrainingExample(instruction=item.instruction, output=item.output, source="copy")
        for item in cleaned[:16]
    ]
    probe_corpus += [
        TrainingExample(
            instruction=f"请帮我看看：{item.instruction}", output=item.output, source="prefixed"
        )
        for item in cleaned[:16]
    ]
    print(f"{'threshold':>10}{'保留':>8}{'精确重复':>10}{'近重复':>8}")
    for threshold in (0.6, DEFAULT_NEAR_DUP_THRESHOLD, 0.8, 0.9):
        index = NearDuplicateIndex(threshold=threshold)
        kept, dedupe = index.filter(probe_corpus)
        print(
            f"{threshold:>10.1f}{len(kept):>8}{dedupe.exact_duplicates:>10}"
            f"{dedupe.near_duplicates:>8}"
        )

    # ------------------------------------------------------------------ 9
    banner("9. 增强不变式与跨批次增量合并")
    head = cleaned[:5]
    augmented, augment_report = augment_dataset(head)
    print(augment_report.summary_line())
    fresh = augmented[len(head) :]
    print(f"新增 {len(fresh)} 条，逐条核对不变式：")
    for item in fresh:
        parent = next(p for p in head if p.output == item.output)
        print(
            f"  op={','.join(t for t in item.tags if t.startswith('aug:')):<14} "
            f"output 与亲本逐字相同={item.output == parent.output} "
            f"source 继承={item.source == parent.source} "
            f"license 继承={item.license == parent.license} "
            f"被识别为增强样本={is_augmented(item)}"
        )
        print(f"    亲本：{parent.instruction[:34]}")
        print(f"    增强：{item.instruction[:34]}")

    print()
    history = cleaned[:20]
    new_batch = cleaned[15:25]  # 其中 5 条与历史重叠
    merged, merge_report = pipeline.merge_with_history(new_batch, history)
    print(f"增量合并：历史 {len(history)} 条 + 新批次 {len(new_batch)} 条")
    print(f"  {merge_report.summary_line()} → 合并后 {len(merged)} 条")

    # ------------------------------------------------------------------ 10
    banner("10. 真实脏样本的质量判定")
    dirty = (
        ("占位符 + 太短", "TODO：待补充。"),
        ("车轱辘话 + 无收尾", "好的好的好的好的好的"),
        ("正常答案（无许可证）", "检索增强生成（RAG）把外部知识检索结果作为上下文交给模型。"),
    )
    for label, output in dirty:
        scored = filter_by_quality(
            [TrainingExample(instruction="什么是 RAG？", output=output, source="probe")]
        )
        score = scored.scores[0]
        print(
            f"  {label:<20} total={score.total:.4f} 通过={score.passed} "
            f"最低维={score.weakest()} 明细="
            f"{ {key: round(value, 4) for key, value in score.dimensions.items()} }"
        )
    print()
    print(f"参考：指纹长度 {len(exact_key('示例'))} 位十六进制；"
          f"最终数据集指纹 {run.manifest.fingerprint}")

    # ------------------------------------------------------------------ 11
    banner("11. 落盘（train.jsonl / eval.jsonl / manifest.json）")
    paths = pipeline.save(run, OUTPUT_DIR)
    for name, path in sorted(paths.items()):
        print(f"  {name:<9} → {path.relative_to(PROJECT_ROOT)}")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    print(f"  清单回读：版本 v{manifest['version']} / 规模 {manifest['size']} / "
          f"指纹 {manifest['fingerprint']}")


if __name__ == "__main__":
    main()
