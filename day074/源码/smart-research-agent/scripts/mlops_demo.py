"""MLOps 微调流水线一键演示（day059 · M5-D10）.

跑法::

    cd day059/源码/smart-research-agent
    PYTHONPATH=. python scripts/mlops_demo.py

八段输出，全部离线、确定、零 GPU、零网络、零随机：

1. **阶段表与主链**——六阶段的依赖/产出由代码生成，并演示"依赖写错会被拒"；
2. **门禁表**——六项门禁的实际阈值（与 ``settings`` 同源）；
3. **单因素对照**——每次只改一个指标，看是哪一项把发布拦下来的；
4. **实验追踪**——确定性 run_id、幂等复用与 ``-2`` 后缀、指标 step 冲突、
   两次 run 的三种比较关系；
5. **端到端流水线**——``dry_run`` 与真正发布两条路径的逐阶段状态；
6. **失败与阻塞的区分**——门禁不通过时 ``publish`` 是 ``blocked`` 而不是
   ``failed``（这正是本课的核心区分）；
7. **模型卡与发布清单**——产物自描述；
8. **CI 集成**——渲染 GitHub Actions workflow，并核对仓库里那份文件。
"""

from __future__ import annotations

import json
from pathlib import Path

from smart_research_agent.mlops import (
    CIConfig,
    ExperimentTracker,
    FinetunePipeline,
    LIMITATIONS,
    ReleaseGates,
    Run,
    StageSpec,
    compare_runs,
    critical_path,
    evaluate_gates,
    gate_table,
    render_github_actions,
    stage_table,
    validate_stage_order,
    workflow_commands,
    workflow_summary,
)
from smart_research_agent.registry import ModelRegistry, ModelVersion, VersionTriple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = "Qwen3-8B"
DATA_A = "3f1b0c9d7e5a2468"
DATA_B = "aa77c31e90f4b258"
ADAPTER_NEW = "5c6d7e8f9012345678" + "b" * 46
ADAPTER_OLD = "1a2b3c4d5e6f70819293a4b5c6d7e8f90123456789abcdef" + "0" * 8


def banner(title: str) -> None:
    """打印一段分隔标题."""
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def make_train(adapter: str = ADAPTER_NEW, *, merged: bool = True, size_mib: float = 0.05):
    """构造确定性训练回调（不碰模型，只返回约定字段）."""

    def train(params: dict) -> dict:
        artifacts = {
            "adapter": f"outputs/lora/adapters/{adapter[:12]}",
            "merged": f"outputs/lora/merged/{adapter[:12]}",
        }
        if not merged:
            artifacts.pop("merged")
        return {
            "adapter_sha256": adapter,
            "adapter_mebibytes": size_mib,
            "train_loss": 0.3187,
            "step": 42,
            "metrics": {"train_loss": 0.3187},
            "artifacts": artifacts,
        }

    return train


def make_evaluate(pass_rate: float, baseline: float = 0.60):
    """构造确定性评估回调（合格率 + 相对基线的变化）."""

    def evaluate(artifacts: dict) -> dict:
        del artifacts
        return {
            "eval_pass_rate": pass_rate,
            "eval_pass_rate_delta": round(pass_rate - baseline, 4),
        }

    return evaluate


def seeded_registry(*, dataset: str = DATA_A, pass_rate: float = 0.60) -> ModelRegistry:
    """一个已有生产版本的注册表（演练的起点）."""
    registry = ModelRegistry()
    registry.register(
        ModelVersion(
            triple=VersionTriple(
                base_model=BASE_MODEL,
                adapter_sha256=ADAPTER_OLD,
                dataset_fingerprint=dataset,
            ),
            version="1.0.0",
            stage="stable",
            metrics={"eval_pass_rate": pass_rate, "train_loss": 0.41},
            artifacts={
                "adapter": "outputs/lora/adapters/adapter-final",
                "merged": "outputs/lora/merged/adapter-final",
            },
        )
    )
    return registry


def main() -> None:  # noqa: C901 - 演示脚本本来就是一长串顺序展示
    # ------------------------------------------------------------------ 1
    banner("1. 阶段表与主链（由代码生成）")
    for row in stage_table():
        print(
            f"  {row['order']}. {row['name']:<9} 依赖 {row['requires'] or ['（无）']} "
            f"→ 产出 {row['produces']}"
        )
        print(f"     {row['description']}")
    print(f"\n主链：{' → '.join(critical_path())}")
    print(f"依赖表自检：{validate_stage_order() or '通过'}")
    broken = (
        StageSpec(name="ingest", description="", requires=(), produces=("dataset",)),
        StageSpec(
            name="publish",
            description="故意把 publish 挪到 train 之前",
            requires=("adapter",),
            produces=("version",),
        ),
    )
    try:
        validate_stage_order(broken)
    except Exception as exc:  # noqa: BLE001 - 演示"依赖写错会被拒"
        print(f"  依赖写错的表被拒：{exc}")

    # ------------------------------------------------------------------ 2
    banner("2. 门禁表（阈值与 settings 同源）")
    for row in gate_table(ReleaseGates()):
        print(
            f"  {row['name']:<22} 阈值 {str(row['threshold']):<18} "
            f"阻塞 {'是' if row['blocking'] else '否'}  {row['meaning'][:44]}"
        )

    # ------------------------------------------------------------------ 3
    banner("3. 单因素对照：每次只改一个指标")
    good = {
        "eval_pass_rate": 0.65,
        "eval_pass_rate_delta": 0.05,
        "adapter_mebibytes": 0.05,
        "dataset_fingerprint": DATA_A,
        "base_model": BASE_MODEL,
    }
    artifacts = {"adapter": "a/adapter", "merged": "m/merged"}
    scenarios = [
        ("全项达标", good, artifacts),
        ("合格率 0.42", {**good, "eval_pass_rate": 0.42}, artifacts),
        ("相对基线 -0.03", {**good, "eval_pass_rate_delta": -0.03}, artifacts),
        ("适配器 80 MiB", {**good, "adapter_mebibytes": 80.0}, artifacts),
        ("缺数据集指纹", {k: v for k, v in good.items() if k != "dataset_fingerprint"}, artifacts),
        ("缺合并产物", good, {"adapter": "a/adapter"}),
        ("加了成本指标 0.09", {**good, "cost_usd_per_1k_tokens": 0.09}, artifacts),
    ]
    for label, metrics, arts in scenarios:
        report = evaluate_gates(metrics, policy=ReleaseGates(), artifacts=arts)
        failed = [item.name for item in report.blocking_failures]
        warned = [item.name for item in report.warnings]
        print(
            f"  {'通过' if report.passed else '不通过'} | {label:<18} "
            f"阻塞失败 {failed or '无'} | 告警 {warned or '无'}"
        )

    # ------------------------------------------------------------------ 4
    banner("4. 实验追踪：确定性 run_id 与三种比较关系")
    tracker = ExperimentTracker()
    params = {"lora_r": 8, "learning_rate": 1e-4, "dataset_fingerprint": DATA_A}
    first = tracker.start_run("lora-ablation", params=params)
    print(f"  第一次：{first.summary_line()}")
    reused = tracker.start_run("lora-ablation", params=params, reuse=True)
    print(f"  reuse=True  → 同一个 run_id：{reused.run_id == first.run_id}")
    suffix = tracker.start_run("lora-ablation", params=params)
    print(f"  reuse=False → 新 id（同参数第二次运行）：{suffix.run_id}")
    other = tracker.start_run("lora-ablation", params={**params, "lora_r": 16})
    print(f"  改一个参数 → 全新 id：{other.run_id}")

    tracker.log_metrics(first, {"train_loss": 0.62}, step=0)
    tracker.log_metrics(first, {"train_loss": 0.3187}, step=42)
    print(f"  逐步指标：{first.history('train_loss')}")
    try:
        tracker.log_metrics(first, {"train_loss": 0.30}, step=42)
    except Exception as exc:  # noqa: BLE001 - 演示"同 step 冲突会被拒"
        print(f"  step 冲突被拒：{str(exc)[:76]}…")

    tracker.log_metrics(first, {"eval_pass_rate": 0.5493}, step=0)
    tracker.log_metrics(first, {"eval_pass_rate": 0.65})
    tracker.log_metrics(other, {"eval_pass_rate": 0.58})
    first.status = "finished"
    other.status = "finished"
    for mode in ("max", "min"):
        payload = compare_runs(first, other, metric="eval_pass_rate", mode=mode)
        print(
            f"  compare(mode={mode}): left={payload['left']} right={payload['right']} "
            f"delta={payload['delta']} → {payload['relation']}"
        )
    missing = compare_runs(first, Run(run_id="x", name="empty"), metric="eval_pass_rate")
    print(f"  缺指标时：comparable={missing['comparable']} relation={missing['relation']}")

    # ------------------------------------------------------------------ 5
    banner("5. 端到端流水线：dry_run 与真正发布")
    for label, dry in (("dry_run=True", True), ("dry_run=False", False)):
        registry = seeded_registry()
        pipeline = FinetunePipeline(
            registry,
            ExperimentTracker(),
            base_model=BASE_MODEL,
            train=make_train(),
            evaluate=make_evaluate(0.7222),
            gates=ReleaseGates(),
        )
        outcome = pipeline.run(
            dataset_fingerprint=DATA_B, params={"lora_r": 8, "learning_rate": 1e-4},
            commit="deadbeef", dry_run=dry,
        )
        print(f"■ {label}：{outcome.summary_line()}")
        for stage in outcome.stages:
            print(f"    [{stage.status:<7}] {stage.name:<9} {stage.duration_ms:7.2f} ms  {stage.detail[:64]}")
        print(f"    版本表规模 {registry.counts()['total']}，head={registry.head().version}")

    # ------------------------------------------------------------------ 6
    banner("6. 门禁不通过：blocked 而不是 failed")
    registry = seeded_registry()
    registry.register(
        ModelVersion(
            triple=VersionTriple(
                base_model=BASE_MODEL, adapter_sha256=ADAPTER_NEW, dataset_fingerprint=DATA_B
            ),
            version="1.1.0",
            parent_version="1.0.0",
            artifacts={"adapter": "a/new"},  # 刻意缺 merged
            metrics={"eval_pass_rate": 0.7222},
        )
    )
    pipeline = FinetunePipeline(
        registry,
        ExperimentTracker(),
        base_model=BASE_MODEL,
        train=make_train(merged=False),
        evaluate=make_evaluate(0.7222),
        gates=ReleaseGates(),
    )
    outcome = pipeline.run(dataset_fingerprint=DATA_B, params={"lora_r": 8})
    print(f"  {outcome.summary_line()}")
    for stage in outcome.stages[-3:]:
        print(f"    [{stage.status:<7}] {stage.name:<9} {stage.detail[:70]}")
    print(f"  failed={outcome.failed}（**blocked 不算失败**）")
    print(f"  追踪器里的 run 状态：{outcome.run.status}")

    # ------------------------------------------------------------------ 7
    banner("7. 模型卡与发布清单")
    registry = seeded_registry()
    pipeline = FinetunePipeline(
        registry,
        ExperimentTracker(),
        base_model=BASE_MODEL,
        train=make_train(),
        evaluate=make_evaluate(0.7222),
        gates=ReleaseGates(),
    )
    published = pipeline.run(
        dataset_fingerprint=DATA_B, params={"lora_r": 8, "learning_rate": 1e-4},
        commit="deadbeef",
    )
    print(published.card.render_markdown())
    print("发布清单（截断展示）：")
    print(json.dumps(published.manifest, ensure_ascii=False, indent=2)[:900])
    print(f"\n已知限制 {len(LIMITATIONS)} 条 / 适用场景 {len(published.card.intended_use)} 条 / "
          f"不适用 {len(published.card.out_of_scope)} 条")

    # ------------------------------------------------------------------ 8
    banner("8. CI 集成：渲染的 workflow 与仓库文件")
    config = CIConfig()
    summary = workflow_summary(config)
    print(f"  文件：{summary['path']}")
    print(f"  cron：{summary['schedule']}（{summary['schedule_note']}）")
    print(f"  超时：{summary['timeout_minutes']} 分钟 | Python {summary['python_version']}")
    print(f"  门禁命令：{' '.join(workflow_commands(config))}")
    rendered = render_github_actions(config)
    committed = PROJECT_ROOT / ".github" / "workflows" / "finetune-nightly.yml"
    same = committed.exists() and committed.read_text(encoding="utf-8") == rendered
    print(f"  仓库里的 finetune-nightly.yml 与渲染结果逐字相同：{same}")
    print(f"  yaml 行数 {len(rendered.splitlines())}，仓库文件存在 {committed.exists()}")

    # 顺带打印门禁报告的实际渲染（评审看的就是它）
    print()
    print(evaluate_gates(good, policy=ReleaseGates(), artifacts=artifacts).render_markdown())


if __name__ == "__main__":
    main()
