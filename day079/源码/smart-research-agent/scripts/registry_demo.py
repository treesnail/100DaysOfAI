"""模型版本管理与持续微调一键演示（day058 · M5-D9）.

跑法::

    cd day058/源码/smart-research-agent
    python scripts/registry_demo.py

九段输出，全部离线、确定、零 GPU、零网络、零随机：

1. **三元组与版本键**——同一份产物的不同写法得到同一个键，三项各变一次
   得到三个不同的键；
2. **版本号递增规则**——一条四步演进（首版 → 重训 → 换数据 → 换基座）
   的 ``bump_kind`` 与 semver 对照；
3. **追加写事件日志与折叠**——``versions.jsonl`` 的真实内容、重载后的
   状态、以及"非法状态迁移会被拒绝"；
4. **采纳判定七场景**——首次上线 / 增益达标 / 落在死区 / 退步 / 缺分数 /
   超龄 / 缺产物，逐项打印检查表；
5. **不可比却硬算差值**——``require_comparable=False`` 下那个数字有多离谱；
6. **触发器与否决项矩阵**——五种输入状态的判定对照；
7. **回滚计划**——正常回滚的 4 步、无退路、祖先不可部署三种情形；
8. **持续微调流水线端到端**——dry_run、真实训练（注入确定性假回调）、
   以及"重跑同一个三元组复用版本号"；
9. **数据集指纹接桥**——用 day057 的领域数据流水线算出真实的
   ``dataset_fingerprint``，并把它写进一条版本记录。
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from smart_research_agent.registry import (
    ARTIFACT_ADAPTER,
    ARTIFACT_DATASET,
    ARTIFACT_MERGED,
    PLAN_STEPS,
    STAGES,
    STAGE_STABLE,
    ContinualFinetunePipeline,
    ModelRegistry,
    ModelVersion,
    PromotionPolicy,
    RegistryError,
    TriggerPolicy,
    TriggerState,
    VersionTriple,
    bump_kind_for,
    evaluate_candidate,
    evaluate_triggers,
    next_version,
    plan_rollback,
    trigger_table,
)
from smart_research_agent.registry.retrain import pipeline_dataset_fingerprint

BASE = "Qwen3-8B"
BIG = "Qwen3-14B"
DATA_V1 = "3f1b0c9d7e5a2468"
DATA_V2 = "aa77c31e90f4b258"
ADAPTER_1 = "1a2b3c4d5e6f70819293a4b5c6d7e8f90123456789abcdef" + "0" * 8
ADAPTER_2 = "5c6d7e8f9012345678" + "b" * 46
ADAPTER_3 = "9a8b7c6d5e4f3021" + "c" * 48
NOW = datetime(2026, 9, 16, 12, 0, 0, tzinfo=timezone.utc)


def banner(title: str) -> None:
    """打印一段分隔标题."""
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def make_record(
    *,
    version: str,
    adapter: str,
    dataset: str = DATA_V1,
    base: str = BASE,
    parent: str = "",
    stage: str = "candidate",
    rating: float | None = 0.60,
    artifacts: dict[str, str] | None = None,
    created_at: str | None = None,
) -> ModelVersion:
    """构造一条版本记录（demo 里的语法糖，产物路径给默认值）."""
    metrics = {} if rating is None else {"eval_pass_rate": rating, "train_loss": 0.41}
    return ModelVersion(
        triple=VersionTriple(
            base_model=base, adapter_sha256=adapter, dataset_fingerprint=dataset
        ),
        version=version,
        parent_version=parent,
        created_at=created_at or NOW.isoformat(),
        stage=stage,
        metrics=metrics,
        artifacts=(
            {
                ARTIFACT_ADAPTER: f"outputs/lora/adapters/{version}",
                ARTIFACT_MERGED: f"outputs/lora/merged/{version}",
                ARTIFACT_DATASET: "outputs/domain_data",
            }
            if artifacts is None
            else artifacts
        ),
    )


def main() -> None:  # noqa: C901 - 演示脚本本来就是一长串顺序展示
    # ------------------------------------------------------------------ 1
    banner("1. 三元组与版本键：内容寻址")
    canonical = VersionTriple(base_model=BASE, adapter_sha256=ADAPTER_1, dataset_fingerprint=DATA_V1)
    sloppy = VersionTriple(
        base_model="  Qwen3-8B  ",
        adapter_sha256=f"sha256:{ADAPTER_1.upper()}",
        dataset_fingerprint=DATA_V1.upper(),
    )
    print(f"规范写法：{canonical.describe()}")
    print(f"脏写法  ：{sloppy.describe()}")
    print(f"三者相等 {canonical == sloppy} | 键相同 {canonical.key == sloppy.key}")
    print()
    variants = [
        ("原始", canonical),
        ("换适配器", VersionTriple(BASE, ADAPTER_2, DATA_V1)),
        ("换数据集", VersionTriple(BASE, ADAPTER_1, DATA_V2)),
        ("换基座  ", VersionTriple(BIG, ADAPTER_1, DATA_V1)),
    ]
    for label, triple in variants:
        print(f"  {label} → {triple.key}  适配器 {triple.short_adapter}")
    print()
    try:
        VersionTriple(BASE, "xyz", DATA_V1)
    except RegistryError as exc:
        print(f"  非法三元组被拒：{exc}")

    # ------------------------------------------------------------------ 2
    banner("2. 版本号递增规则：变化的那一项决定递增位")
    evolution = [
        ("首版", None, BASE, DATA_V1, ADAPTER_1),
        ("同数据重训", VersionTriple(BASE, ADAPTER_1, DATA_V1), BASE, DATA_V1, ADAPTER_2),
        ("换数据集", VersionTriple(BASE, ADAPTER_2, DATA_V1), BASE, DATA_V2, ADAPTER_3),
        ("换基座", VersionTriple(BASE, ADAPTER_3, DATA_V2), BIG, DATA_V2, ADAPTER_3),
    ]
    history: list[str] = []
    print("| 步骤 | 变化项 | 递增位 | 新版本号 |")
    print("|------|--------|--------|----------|")
    for label, previous, base, dataset, adapter in evolution:
        kind = bump_kind_for(previous, base_model=base, dataset_fingerprint=dataset)
        version = next_version(history, kind=kind)
        history.append(version)
        changed = (
            "（首版）"
            if previous is None
            else "base_model"
            if previous.base_model != base
            else "dataset_fingerprint"
            if previous.dataset_fingerprint != dataset
            else "adapter_sha256"
        )
        print(f"| {label} | {changed} | {kind} | v{version} |")
    print(f"\n最终版本序列：{history}")

    # ------------------------------------------------------------------ 3
    banner("3. 追加写事件日志与折叠")
    with tempfile.TemporaryDirectory() as tmp:
        index = Path(tmp) / "versions.jsonl"
        registry = ModelRegistry(index)
        registry.register(make_record(version="1.0.0", adapter=ADAPTER_1, rating=0.60))
        registry.set_stage("1.0.0", STAGE_STABLE, reason="首次上线，人工门禁通过")
        registry.register(
            make_record(
                version="1.0.1", adapter=ADAPTER_2, parent="1.0.0", rating=0.66
            )
        )
        print("versions.jsonl 的真实内容（每行一个事件）：")
        for index_no, line in enumerate(index.read_text(encoding="utf-8").splitlines(), start=1):
            payload = json.loads(line)
            print(f"  {index_no}. type={payload['type']:<8} version={payload['version']}")
            if payload["type"] == "stage":
                print(f"     stage={payload['stage']} reason={payload['reason']}")
        print()
        print(f"折叠后：{registry.counts()} | head={registry.head().version}")
        reloaded = ModelRegistry(index)
        print(
            f"重新加载后：{reloaded.counts()} | head={reloaded.head().version} | "
            f"键一致 {reloaded.head().version_key == registry.head().version_key}"
        )
        print(f"版本链（v1.0.1 → 祖先）：{[item.version for item in reloaded.lineage('1.0.1')]}")
        print(f"可回滚的 stable 祖先：{[item.version for item in reloaded.stable_ancestors('1.0.1')]}")
        print()
        try:
            reloaded.get("1.0.0").with_stage("candidate")
        except RegistryError as exc:
            print(f"  非法迁移被拒：{exc}")
        try:
            reloaded.register(make_record(version="1.0.0", adapter=ADAPTER_3, rating=0.5))
        except RegistryError as exc:
            print(f"  版本号冲突被拒：{exc}")
        print(f"  幂等重登记：{reloaded.register(make_record(version='1.0.0', adapter=ADAPTER_1, rating=0.60)).version}")

    # ------------------------------------------------------------------ 4
    banner("4. 采纳判定七场景")
    current = make_record(version="1.0.0", adapter=ADAPTER_1, stage=STAGE_STABLE, rating=0.60)
    fresh = NOW.isoformat()
    stale = (NOW - timedelta(days=21)).isoformat()
    scenarios = [
        ("首次上线（无 current）", None, make_record(version="1.0.0", adapter=ADAPTER_1, rating=0.58, created_at=fresh)),
        ("增益达标 +0.12", current, make_record(version="1.0.1", adapter=ADAPTER_2, parent="1.0.0", rating=0.72, created_at=fresh)),
        ("落在死区 +0.01", current, make_record(version="1.0.1", adapter=ADAPTER_2, parent="1.0.0", rating=0.61, created_at=fresh)),
        ("退步 -0.08", current, make_record(version="1.0.1", adapter=ADAPTER_2, parent="1.0.0", rating=0.52, created_at=fresh)),
        ("候选缺分数", current, make_record(version="1.0.1", adapter=ADAPTER_2, parent="1.0.0", rating=None, created_at=fresh)),
        ("候选超龄 21 天", current, make_record(version="1.0.1", adapter=ADAPTER_2, parent="1.0.0", rating=0.80, created_at=stale)),
        (
            "候选缺合并产物",
            current,
            make_record(
                version="1.0.1",
                adapter=ADAPTER_2,
                parent="1.0.0",
                rating=0.80,
                created_at=fresh,
                artifacts={ARTIFACT_ADAPTER: "outputs/lora/adapters/1.0.1"},
            ),
        ),
    ]
    for label, base_record, candidate in scenarios:
        decision = evaluate_candidate(base_record, candidate, policy=PromotionPolicy(), now=NOW)
        gain = "不可比" if decision.gain is None else f"{decision.gain:+.4f}"
        print(f"■ {label} → {decision.action}（gain {gain}，可比 {decision.comparable}）")
        for check in decision.checks:
            print(f"    {check.summary_line()}")

    # ------------------------------------------------------------------ 5
    banner("5. 不可比却硬算差值：一个被实测出来的错误")
    current_v1 = make_record(version="1.0.0", adapter=ADAPTER_1, stage=STAGE_STABLE, rating=0.60)
    next_dataset = make_record(
        version="1.1.0", adapter=ADAPTER_3, dataset=DATA_V2, parent="1.0.0",
        rating=0.44, created_at=fresh,
    )
    default_policy = PromotionPolicy()
    loose_policy = PromotionPolicy(require_comparable=False)
    for label, policy in (("缺省（require_comparable=True）", default_policy), ("关掉可比性要求", loose_policy)):
        decision = evaluate_candidate(current_v1, next_dataset, policy=policy, now=NOW)
        gain = "未计算" if decision.gain is None else f"{decision.gain:+.4f}"
        print(f"  {label}: action={decision.action} gain={gain}")
        print(f"    {decision.reason}")
    print(
        "\n  ↑ 后者那个 -0.1600 既不是提升也不是退步，"
        "而是两种评估分布之差——它会被下游当成提升量使用。"
    )

    # ------------------------------------------------------------------ 6
    banner("6. 触发器与否决项矩阵")
    print("条件表（由 trigger_table() 生成，文档与实现同源）：")
    for row in trigger_table(TriggerPolicy()):
        print(f"  [{row['kind']:<7}] {row['name']:<15} 阈值 {row['threshold']} —— {row['meaning']}")
    print()
    states = [
        ("首次训练（从未训练，全部样本都算新增）", dict(new_examples=37, dataset_fingerprint=DATA_V1, stable_dataset_fingerprint="", hours_since_last_train=None)),
        ("数据涨 30 条、冷却已过", dict(new_examples=30, dataset_fingerprint=DATA_V2, stable_dataset_fingerprint=DATA_V1, hours_since_last_train=30.0)),
        ("数据涨 30 条、冷却只过了 2h", dict(new_examples=30, dataset_fingerprint=DATA_V2, stable_dataset_fingerprint=DATA_V1, hours_since_last_train=2.0)),
        ("数据没涨、线上掉到 0.42", dict(new_examples=0, dataset_fingerprint=DATA_V1, stable_dataset_fingerprint=DATA_V1, online_pass_rate=0.42, hours_since_last_train=48.0)),
        ("数据没涨、线上也没掉", dict(new_examples=0, dataset_fingerprint=DATA_V1, stable_dataset_fingerprint=DATA_V1, online_pass_rate=0.72, hours_since_last_train=48.0)),
        ("已有训练任务在跑", dict(new_examples=30, dataset_fingerprint=DATA_V2, stable_dataset_fingerprint=DATA_V1, hours_since_last_train=48.0, active_runs=1)),
    ]
    for label, payload in states:
        decision = evaluate_triggers(TriggerState(**payload))
        flag = "[训练]  " if decision.should_retrain else "[不训练]"
        print(f"  {flag} {label}")
        print(f"      命中 {list(decision.fired) or '（无）'} | 否决 {list(decision.vetoed_by) or '（无）'}")
        print(f"      {decision.notes}")

    # ------------------------------------------------------------------ 7
    banner("7. 回滚计划三种情形")
    def build_chain(*specs):
        registry = ModelRegistry()
        for spec in specs:
            record = make_record(**spec)
            registry.register(record)
        return registry

    good_1 = dict(version="1.0.0", adapter=ADAPTER_1, stage=STAGE_STABLE, rating=0.60)
    good_2 = dict(version="1.0.1", adapter=ADAPTER_2, parent="1.0.0", stage=STAGE_STABLE, rating=0.68)
    bad_3 = dict(version="1.0.2", adapter=ADAPTER_3, parent="1.0.1", stage=STAGE_STABLE, rating=0.71)

    print("■ 情形 A：正常回滚（三个 stable，退到 v1.0.1）")
    registry_a = build_chain(good_1, good_2, bad_3)
    plan_a = plan_rollback(registry_a, "1.0.2", reason="上线后拒答率异常升高")
    print(f"  {plan_a.summary_line()}")
    for step in plan_a.steps:
        print(f"    {step.summary_line()}")

    print("\n■ 情形 B：无退路（只有首个 stable）")
    registry_b = build_chain(good_1)
    plan_b = plan_rollback(registry_b, "1.0.0", reason="尝试回滚首个版本")
    print(f"  {plan_b.summary_line()}")
    print(f"  should_execute={plan_b.should_execute} steps={len(plan_b.steps)}")

    print("\n■ 情形 C：最近的祖先不可部署，自动往前找")
    undeployable = dict(
        version="1.0.1",
        adapter=ADAPTER_2,
        parent="1.0.0",
        stage=STAGE_STABLE,
        rating=0.68,
        artifacts={ARTIFACT_ADAPTER: "outputs/lora/adapters/1.0.1"},
    )
    registry_c = build_chain(good_1, undeployable, bad_3)
    plan_c = plan_rollback(registry_c, "1.0.2", reason="产物目录被清理后的回滚演练")
    print(f"  {plan_c.summary_line()}")
    print(f"  目标版本 {plan_c.target_version}（跳过了不可部署的 v1.0.1）")

    # ------------------------------------------------------------------ 8
    banner("8. 持续微调流水线端到端")
    with tempfile.TemporaryDirectory() as tmp:
        registry = ModelRegistry(Path(tmp) / "versions.jsonl")
        registry.register(make_record(version="1.0.0", adapter=ADAPTER_1, stage=STAGE_STABLE, rating=0.60))

        train_calls: list[str] = []

        def trainer(plan):  # type: ignore[no-untyped-def]
            """确定性假训练器：产出适配器哈希 + 产物路径（不碰任何模型）."""
            train_calls.append(plan.version)
            return {
                "adapter_sha256": ADAPTER_2,
                "step": 42,
                "train_loss": 0.3187,
                "artifacts": {
                    ARTIFACT_ADAPTER: f"outputs/lora/adapters/{plan.version}",
                    ARTIFACT_MERGED: f"outputs/lora/merged/{plan.version}",
                },
            }

        def evaluator(artifacts):  # type: ignore[no-untyped-def]
            """确定性假评估器：按产物路径给一个可解释的合格率."""
            del artifacts
            return {"eval_pass_rate": 0.7222, "train_loss": 0.3187}

        pipeline = ContinualFinetunePipeline(
            registry, base_model=BASE, trainer=trainer, evaluator=evaluator
        )
        state = pipeline.build_state(
            new_examples=30,
            dataset_fingerprint=DATA_V2,
            hours_since_last_train=30.0,
        )
        print(f"策略快照：{json.dumps(pipeline.policy_snapshot(), ensure_ascii=False)}")
        print(f"步骤顺序：{list(PLAN_STEPS)}")
        print()

        dry = pipeline.run(state, dry_run=True)
        print(f"dry_run：{dry.summary_line()}")
        print(f"  trained={dry.trained} trainer 被调用次数={len(train_calls)}")
        print()

        first = pipeline.run(state)
        print(f"第一次真实运行：{first.summary_line()}")
        for item in first.timeline:
            print(f"    [{'✓' if item['ok'] else '✗'}] {item['step']:<12} {item['detail'][:96]}")
        print(f"  head 现在是 {registry.head().version}，trainer 调用次数={len(train_calls)}")
        print()

        second = pipeline.run(
            pipeline.build_state(
                new_examples=30, dataset_fingerprint=DATA_V2, hours_since_last_train=48.0
            )
        )
        print(f"第二次运行（同一份数据、同一个适配器）：{second.summary_line()}")
        print(f"  登记的版本号 {second.version.version} | 版本表规模 {registry.counts()['total']}")
        print(f"  trainer 调用次数={len(train_calls)}（三元组相同仍会重新训练一次，但复用版本号）")

    # ------------------------------------------------------------------ 9
    banner("9. 数据集指纹接桥：day057 流水线 → 版本三元组")
    fingerprint = pipeline_dataset_fingerprint()
    print(f"DomainDataManifest.fingerprint = {fingerprint}")
    record = make_record(version="1.0.0", adapter=ADAPTER_1, dataset=fingerprint, rating=0.60)
    linked = ModelRegistry()
    linked.register(record)
    print(f"这条版本记录因此可以回答'是哪份数据训出来的'：{record.triple.describe()}")
    print(f"注册表自描述：{json.dumps(linked.to_dict()['counts'], ensure_ascii=False)}")
    print(f"阶段枚举：{list(STAGES)}")


if __name__ == "__main__":
    main()
