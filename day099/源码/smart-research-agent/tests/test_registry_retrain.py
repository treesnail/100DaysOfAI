"""day058 持续微调流水线测试（M5-D9）：六步顺序、门禁前置、复用版本号.

这个文件守的是编排层的四件事，每一件都对应一个"只在某些输入下才出错"的
顺序或边界：

1. **``gate`` 在 ``register`` 之前**：没通过门禁的候选不该进注册表。
   先进表再判断会让版本表堆满"登记了但永远不该上线"的记录，
   而 ``lineage`` 会开始被它们污染（它不看阶段）；
2. **``promote`` 在 ``register`` 之后**：采纳是把一个**已经存在的**记录
   升为 stable。反过来会出现"生产指针指向注册表里还没有的版本"；
3. **``verify_data`` 在最前**：训练前必须确认数据确实变了（或增量够数），
   否则等于"先花完算力再发现数据没变"；
4. **三元组复用**：同一份数据 + 同一个适配器重跑时**复用版本号**，
   并且如果它已经在线上，就跳过采纳判定——因为此时 gain 恒为 0，
   任何 ``min_gain`` 都会判 hold，而那个 hold 会把"它正在服务"
   这个事实说成"未采纳"。

流水线的训练与评估用**注入的确定性假回调**（不碰模型、不联网、不写盘），
因此这些用例全部毫秒级完成。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from smart_research_agent.registry import (
    PLAN_STEPS,
    STAGE_CANDIDATE,
    STAGE_STABLE,
    STEP_MEANINGS,
    ContinualFinetunePipeline,
    ModelRegistry,
    ModelVersion,
    PromotionPolicy,
    RegistryError,
    TriggerPolicy,
    TriggerState,
    VersionTriple,
    evaluate_triggers,
)
from smart_research_agent.registry.retrain import (
    TRAIN_RESULT_ARTIFACTS,
    TRAIN_RESULT_REQUIRED,
    RetrainOutcome,
    RetrainPlan,
    build_state,
    pipeline_dataset_fingerprint,
    verify_data,
)

BASE = "Qwen3-8B"
BIG = "Qwen3-14B"
DATA_A = "3f1b0c9d7e5a2468"
DATA_B = "aa77c31e90f4b258"
NEW_ADAPTER = "5c6d7e8f9012345678" + "b" * 46


def hexdigest64(seed: str) -> str:
    """由任意字符串派生 64 位十六进制串（确定性适配器哈希）."""
    raw = "".join(f"{byte:02x}" for byte in seed.encode("utf-8"))
    return (raw * 8)[:64]


def make(
    version: str,
    *,
    adapter: str | None = None,
    dataset: str = DATA_A,
    base: str = BASE,
    parent: str = "",
    stage: str = STAGE_CANDIDATE,
    rating: float | None = 0.6,
) -> ModelVersion:
    """构造一条版本记录（产物默认齐全）."""
    return ModelVersion(
        triple=VersionTriple(
            base_model=base,
            adapter_sha256=adapter or hexdigest64(version),
            dataset_fingerprint=dataset,
        ),
        version=version,
        parent_version=parent,
        stage=stage,
        metrics={} if rating is None else {"eval_pass_rate": rating},
        artifacts={"adapter": f"a/{version}", "merged": f"m/{version}"},
    )


def registry_with_head(rating: float = 0.60, dataset: str = DATA_A) -> ModelRegistry:
    """一个已有生产版本的注册表（head = v1.0.0）."""
    registry = ModelRegistry()
    registry.register(make("1.0.0", stage=STAGE_STABLE, rating=rating, dataset=dataset))
    return registry


class FakeTrainer:
    """确定性假训练器：记录调用、返回可配置的适配器哈希与产物."""

    def __init__(
        self,
        *,
        adapter: str = NEW_ADAPTER,
        artifacts: dict[str, str] | None = None,
        extra: dict | None = None,
    ) -> None:
        self.adapter = adapter
        self.artifacts = (
            artifacts if artifacts is not None else {"adapter": "a/new", "merged": "m/new"}
        )
        self.extra = extra or {}
        self.calls: list[RetrainPlan] = []

    def __call__(self, plan: RetrainPlan) -> dict:
        """返回训练产物（含回调契约要求的 ``adapter_sha256``）."""
        self.calls.append(plan)
        return {
            "adapter_sha256": self.adapter,
            "step": 42,
            "train_loss": 0.3187,
            "artifacts": dict(self.artifacts),
            **self.extra,
        }


class FakeEvaluator:
    """确定性假评估器：返回固定合格率，并记录被调用的产物路径."""

    def __init__(self, rating: float = 0.72) -> None:
        self.rating = rating
        self.calls: list[dict] = []

    def __call__(self, artifacts: dict[str, str]) -> dict:
        """返回一条 ``eval_pass_rate``（门禁唯一消费的指标）。"""
        self.calls.append(dict(artifacts))
        return {"eval_pass_rate": self.rating, "train_loss": 0.3187}


# --------------------------------------------------------------------------- #
# 常量与计划
# --------------------------------------------------------------------------- #


def test_plan_steps_are_six_in_fixed_order() -> None:
    """六步顺序即策略——它进 ``RetrainPlan.steps``、报告与 API 的自我描述。"""
    assert PLAN_STEPS == (
        "verify_data",
        "train",
        "evaluate",
        "gate",
        "register",
        "promote",
    )
    assert PLAN_STEPS.index("gate") < PLAN_STEPS.index("register") < PLAN_STEPS.index("promote")


def test_step_meanings_cover_every_step() -> None:
    """每个步骤都要有一句话说明（报告里的表格直接取它）。"""
    assert set(STEP_MEANINGS) == set(PLAN_STEPS)
    assert all(STEP_MEANINGS[name] for name in PLAN_STEPS)


def test_trainer_contract_constants() -> None:
    """回调契约里唯一必填的字段是适配器哈希（没有它算不出版本键）。"""
    assert TRAIN_RESULT_REQUIRED == ("adapter_sha256",)
    assert set(TRAIN_RESULT_ARTIFACTS) == {"adapter", "merged", "dataset"}


def test_retrain_plan_projection_summary_and_markdown() -> None:
    """计划要能打印、能归档——它是训练脚本的输入与复盘的入口。"""
    registry = registry_with_head()
    pipeline = ContinualFinetunePipeline(registry, base_model=BASE)
    outcome = pipeline.run(
        pipeline.build_state(new_examples=30, dataset_fingerprint=DATA_B), dry_run=True
    )
    plan = outcome.plan
    assert plan is not None
    payload = plan.to_dict()
    assert payload["version"] == plan.version
    assert payload["steps"] == list(PLAN_STEPS)
    assert payload["parent_version"] == "1.0.0"
    assert "计划 v" in plan.summary_line()
    markdown = plan.render_markdown()
    assert "# 重训计划" in markdown
    assert "verify_data" in markdown
    assert "# 重训触发判定" in markdown


# --------------------------------------------------------------------------- #
# build_state
# --------------------------------------------------------------------------- #


def test_build_state_pulls_stable_dataset_fingerprint() -> None:
    """状态里的"生产版本所用数据指纹"由流水线自己从注册表补出。"""
    registry = registry_with_head(dataset=DATA_A)
    state = build_state(registry, new_examples=30, dataset_fingerprint=DATA_B)
    assert state.stable_dataset_fingerprint == DATA_A
    assert state.dataset_changed() is True


def test_build_state_without_head_leaves_fingerprint_blank() -> None:
    """没有生产版本时基线指纹为空 → 数据集变化不触发（"没有基线"不是"变了"）。"""
    state = build_state(ModelRegistry(), new_examples=37, dataset_fingerprint=DATA_A)
    assert state.stable_dataset_fingerprint == ""
    assert state.dataset_changed() is False


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #


def test_plan_reserves_version_and_links_parent() -> None:
    """计划预留版本号（训练脚本需要知道"这次要产出 v几"）并带上父版本。"""
    registry = registry_with_head()
    pipeline = ContinualFinetunePipeline(registry, base_model=BASE)
    state = pipeline.build_state(new_examples=30, dataset_fingerprint=DATA_B)
    plan = pipeline.plan(state, pipeline.run(state, dry_run=True).decision)
    assert plan.version == "1.1.0"
    assert plan.parent_version == "1.0.0"
    assert plan.bump_kind == "minor"


def test_plan_uses_patch_when_only_adapter_will_change() -> None:
    """同数据同基座 → patch（持续微调里最高频的动作）。"""
    registry = registry_with_head(dataset=DATA_A)
    pipeline = ContinualFinetunePipeline(registry, base_model=BASE)
    state = pipeline.build_state(new_examples=30, dataset_fingerprint=DATA_A)
    decision = pipeline.run(state, dry_run=True).decision
    assert pipeline.plan(state, decision).bump_kind == "patch"
    assert pipeline.plan(state, decision).version == "1.0.1"


def test_plan_uses_major_when_base_changes() -> None:
    """换基座 → major。"""
    registry = registry_with_head()
    pipeline = ContinualFinetunePipeline(registry, base_model=BIG)
    state = pipeline.build_state(new_examples=30, dataset_fingerprint=DATA_B)
    decision = pipeline.run(state, dry_run=True).decision
    assert pipeline.plan(state, decision).version == "2.0.0"


def test_pipeline_rejects_blank_base_model() -> None:
    """空基座名会让三元组只剩两项，构造期就拒绝。"""
    with pytest.raises(RegistryError, match="base_model 不能为空"):
        ContinualFinetunePipeline(ModelRegistry(), base_model="   ")


def test_policy_snapshot_contains_everything_that_affects_the_result() -> None:
    """快照里要有全部影响结果的参数，否则"同一次判定可复现"不成立。"""
    pipeline = ContinualFinetunePipeline(ModelRegistry(), base_model=BASE)
    snapshot = pipeline.policy_snapshot()
    assert snapshot["base_model"] == BASE
    assert set(snapshot) == {
        "base_model",
        "trigger_policy",
        "promotion_policy",
        "observe_window_hours",
        "steps",
    }


# --------------------------------------------------------------------------- #
# run：跳过与 dry_run
# --------------------------------------------------------------------------- #


def test_run_skips_without_trigger_and_never_calls_trainer() -> None:
    """一条触发器都没命中 → 立刻返回，连计划都不生成（省算力的第一道闸）。"""
    trainer = FakeTrainer()
    pipeline = ContinualFinetunePipeline(
        registry_with_head(), base_model=BASE, trainer=trainer, evaluator=FakeEvaluator()
    )
    outcome = pipeline.run(pipeline.build_state(new_examples=1, dataset_fingerprint=DATA_A))
    assert outcome.plan is None
    assert outcome.trained is False
    assert trainer.calls == []
    assert "未启动训练" in outcome.summary_line()


def test_run_dry_run_produces_plan_without_touching_callbacks() -> None:
    """dry_run 只到"计划"为止：**它恰好能回答"这次到底该不该训"这个问题**."""
    trainer = FakeTrainer()
    evaluator = FakeEvaluator()
    pipeline = ContinualFinetunePipeline(
        registry_with_head(), base_model=BASE, trainer=trainer, evaluator=evaluator
    )
    outcome = pipeline.run(
        pipeline.build_state(new_examples=30, dataset_fingerprint=DATA_B), dry_run=True
    )
    assert outcome.plan is not None
    assert outcome.trained is False
    assert trainer.calls == [] and evaluator.calls == []
    assert outcome.timeline[0]["step"] == "decide"
    assert outcome.timeline[1]["step"] == "plan"
    assert "dry_run" in outcome.skipped_reason


# --------------------------------------------------------------------------- #
# run：端到端
# --------------------------------------------------------------------------- #


def test_run_happy_path_registers_candidate_then_promotes() -> None:
    """完整路径：登记为候选 → 采纳判定 promote → 升为 stable，head 跟上。"""
    registry = registry_with_head(rating=0.60)
    trainer = FakeTrainer()
    evaluator = FakeEvaluator(rating=0.72)
    pipeline = ContinualFinetunePipeline(
        registry, base_model=BASE, trainer=trainer, evaluator=evaluator
    )
    outcome = pipeline.run(pipeline.build_state(new_examples=30, dataset_fingerprint=DATA_B))

    assert outcome.trained is True
    assert outcome.promoted is True
    assert [item["step"] for item in outcome.timeline] == [
        "decide",
        "plan",
        "verify_data",
        "train",
        "evaluate",
        "register",
        "gate",
        "promote",
    ]
    assert registry.head().version == outcome.version.version
    assert registry.get("1.1.0").stage == STAGE_STABLE
    assert len(trainer.calls) == 1 and len(evaluator.calls) == 1
    assert registry.counts()["total"] == 2


def test_run_gate_failure_keeps_candidate_out_of_production() -> None:
    """门禁不通过：候选留在 candidate，head 不变（**上线是显式动作**）.

    这里刻意用**同一份数据集**：可比时才算增益，0.55 vs 0.70 的 -0.15
    会被 ``regression_tolerance=0`` 直接拦下。
    """
    registry = registry_with_head(rating=0.70, dataset=DATA_A)
    pipeline = ContinualFinetunePipeline(
        registry,
        base_model=BASE,
        trainer=FakeTrainer(),
        evaluator=FakeEvaluator(rating=0.55),
    )
    outcome = pipeline.run(pipeline.build_state(new_examples=30, dataset_fingerprint=DATA_A))

    assert outcome.promoted is False
    assert registry.head().version == "1.0.0"
    assert registry.get(outcome.version.version).stage == STAGE_CANDIDATE
    assert outcome.promotion is not None
    assert outcome.promotion.action == "hold"


def test_run_records_candidate_even_when_gate_holds() -> None:
    """门禁不通过**仍然登记**候选：候选阶段让 head 保持不变，
    但记录留下来了（下一次可以用同一份产物重跑门禁，不必重新训练）。"""
    registry = registry_with_head(rating=0.70, dataset=DATA_A)
    pipeline = ContinualFinetunePipeline(
        registry,
        base_model=BASE,
        trainer=FakeTrainer(),
        evaluator=FakeEvaluator(rating=0.71),
    )
    outcome = pipeline.run(pipeline.build_state(new_examples=30, dataset_fingerprint=DATA_A))
    assert outcome.version is not None
    assert outcome.version.version in registry
    assert registry.counts() == {
        STAGE_CANDIDATE: 1,
        STAGE_STABLE: 1,
        "rolled_back": 0,
        "archived": 0,
        "total": 2,
    }


def test_run_reuses_version_number_for_identical_triple() -> None:
    """同一份数据 + 同一个适配器重跑：复用版本号，不新占一个.

    训练回调仍然会被调用一次（流水线无法知道"这次训练会产出什么"），
    但注册表里不会多出一个版本——这正是"CI 重试"应有的行为。
    """
    registry = registry_with_head(rating=0.60)
    trainer = FakeTrainer()
    evaluator = FakeEvaluator(rating=0.72)
    pipeline = ContinualFinetunePipeline(
        registry, base_model=BASE, trainer=trainer, evaluator=evaluator
    )
    state = pipeline.build_state(new_examples=30, dataset_fingerprint=DATA_B)
    first = pipeline.run(state)
    second = pipeline.run(state)

    assert registry.counts()["total"] == 2
    assert second.version.version == first.version.version == "1.1.0"
    assert len(trainer.calls) == 2  # 训练了两次（流水线无法预知产物）
    assert second.promoted is True


def test_run_skips_gate_when_reused_version_is_already_stable() -> None:
    """复用 + 已上线 → 跳过采纳判定.

    此时 current 与 candidate 是同一条记录，gain 恒为 0，任何 ``min_gain``
    都会判 hold——而那个 hold 会把"它正在服务"这个事实说成"未采纳"。
    """
    registry = registry_with_head(rating=0.60)
    pipeline = ContinualFinetunePipeline(
        registry,
        base_model=BASE,
        trainer=FakeTrainer(),
        evaluator=FakeEvaluator(rating=0.72),
    )
    state = pipeline.build_state(new_examples=30, dataset_fingerprint=DATA_B)
    pipeline.run(state)
    repeat = pipeline.run(state)

    assert repeat.promotion is None  # 没有做采纳判定
    assert repeat.promoted is True  # 但"它正在服务"依然成立
    details = {item["step"]: item["detail"] for item in repeat.timeline}
    assert "无需采纳判定" in details["gate"]
    assert "已处于 stable" in details["promote"]
    assert "复用版本号" in repeat.skipped_reason


def test_run_reruns_gate_for_reused_candidate() -> None:
    """复用的是一个**候选**时照常走门禁（它还没上线，需要判定）.

    第一次用严格策略（``min_gain=0.05``）拦下 0.01 的增益，
    第二次换回缺省策略（``min_gain=0.02``）——**同一份产物、同一个版本号**
    被采纳，训练一次都没重跑（这里 trainer 仍被调用，但注册表里没有新版本）。
    """
    registry = registry_with_head(rating=0.70, dataset=DATA_A)
    trainer = FakeTrainer()
    evaluator = FakeEvaluator(rating=0.71)
    state_holder = ContinualFinetunePipeline(
        registry,
        base_model=BASE,
        trainer=trainer,
        evaluator=evaluator,
        promotion_policy=PromotionPolicy(min_gain=0.05),
    )
    state = state_holder.build_state(new_examples=30, dataset_fingerprint=DATA_A)
    first = state_holder.run(state)
    assert first.promoted is False
    assert registry.get(first.version.version).stage == STAGE_CANDIDATE

    relaxed = ContinualFinetunePipeline(
        registry,
        base_model=BASE,
        trainer=trainer,
        evaluator=evaluator,
        promotion_policy=PromotionPolicy(min_gain=0.0),
    )
    second = relaxed.run(state)
    assert second.version.version == first.version.version
    assert second.promoted is True
    assert registry.counts()["total"] == 2


def test_run_filters_unknown_artifact_slots() -> None:
    """训练回调返回的未知产物槽位被静默丢弃（不写进记录的 ``artifacts``）."""
    registry = registry_with_head()
    trainer = FakeTrainer(artifacts={"adapter": "a/new", "merged": "m/new", "junk": "x"})
    pipeline = ContinualFinetunePipeline(
        registry, base_model=BASE, trainer=trainer, evaluator=FakeEvaluator()
    )
    outcome = pipeline.run(pipeline.build_state(new_examples=30, dataset_fingerprint=DATA_B))
    assert set(outcome.version.artifacts) == {"adapter", "merged"}


def test_run_propagates_version_tags() -> None:
    """登记时把递增位与命中的触发器写进标签（"为什么有这一版"永远可查）."""
    registry = registry_with_head()
    pipeline = ContinualFinetunePipeline(
        registry,
        base_model=BASE,
        trainer=FakeTrainer(),
        evaluator=FakeEvaluator(),
    )
    outcome = pipeline.run(pipeline.build_state(new_examples=30, dataset_fingerprint=DATA_B))
    assert outcome.version.tags["bump_kind"] == "minor"
    assert "data_growth" in outcome.version.tags["triggered_by"]
    assert "自动登记" in outcome.version.notes


# --------------------------------------------------------------------------- #
# run：回调契约与前置校验
# --------------------------------------------------------------------------- #


def test_run_requires_trainer_when_not_dry_run() -> None:
    """非 dry_run 缺少训练回调 → 立刻报错（而不是静默产出一份空记录）。"""
    pipeline = ContinualFinetunePipeline(registry_with_head(), base_model=BASE)
    state = pipeline.build_state(new_examples=30, dataset_fingerprint=DATA_B)
    with pytest.raises(RegistryError, match="未注入 trainer"):
        pipeline.run(state)


def test_run_requires_evaluator_when_not_dry_run() -> None:
    """只有训练回调也不行：没有评估就没有门禁的判据。"""
    pipeline = ContinualFinetunePipeline(
        registry_with_head(), base_model=BASE, trainer=FakeTrainer()
    )
    state = pipeline.build_state(new_examples=30, dataset_fingerprint=DATA_B)
    with pytest.raises(RegistryError, match="未注入 evaluator"):
        pipeline.run(state)


def test_run_rejects_trainer_without_adapter_hash() -> None:
    """训练回调必须给出适配器内容哈希——没有它就算不出版本键."""
    pipeline = ContinualFinetunePipeline(
        registry_with_head(),
        base_model=BASE,
        trainer=lambda plan: {"step": 1},
        evaluator=FakeEvaluator(),
    )
    state = pipeline.build_state(new_examples=30, dataset_fingerprint=DATA_B)
    with pytest.raises(RegistryError, match="adapter_sha256"):
        pipeline.run(state)


def test_run_rejects_blank_dataset_fingerprint() -> None:
    """数据指纹为空的训练无法追溯，**在生成计划时就被拦下**.

    实际的拦点是版本号提议（``bump_kind_for`` → ``normalize_digest``），
    它比 ``verify_data`` 更早：一个空指纹连"下一个版本号是多少"都算不出来。
    两道闸门都保留——它们的触发条件不同，而且在不同的调用路径上生效。
    """
    pipeline = ContinualFinetunePipeline(
        registry_with_head(),
        base_model=BASE,
        trainer=FakeTrainer(),
        evaluator=FakeEvaluator(),
    )
    state = pipeline.build_state(
        new_examples=30, dataset_fingerprint="", online_pass_rate=0.1
    )
    with pytest.raises(RegistryError, match="dataset_fingerprint 不能为空"):
        pipeline.run(state)


def test_verify_data_rejects_blank_dataset_fingerprint() -> None:
    """``verify_data`` 自己也拦一次空指纹（供不经过版本号提议的调用路径使用）."""
    state = TriggerState(new_examples=30, dataset_fingerprint="   ")
    with pytest.raises(RegistryError, match="dataset_fingerprint 为空"):
        verify_data(state, evaluate_triggers(state))


def test_verify_data_rejects_training_without_any_data_change_or_quality_drop() -> None:
    """数据没变、质量也没掉却要训练 → 多半是调度器配置写错了.

    这条护栏在**端到端路径上走不到**（触发器早已把这类状态挡在外面），
    因此它必须能被单独驱动——这正是 ``verify_data`` 是模块级函数、
    而不是流水线私有方法的原因。一个永远走不到的 ``raise`` 分支
    既拿不到覆盖率，也没人会发现它的条件写反了。
    """
    state = TriggerState(
        new_examples=0,
        dataset_fingerprint=DATA_A,
        stable_dataset_fingerprint=DATA_A,
    )
    # 先确认触发器本身也不会放行这种状态（两道闸门是冗余而不是替代）
    assert evaluate_triggers(state).should_retrain is False
    with pytest.raises(RegistryError, match="既没有新增样本"):
        verify_data(state, evaluate_triggers(state))


def test_verify_data_allows_same_data_retrain_when_quality_dropped() -> None:
    """线上质量下降触发的同数据重训是**合法**的，理由会被写进时间线."""
    registry = registry_with_head(dataset=DATA_A, rating=0.60)
    pipeline = ContinualFinetunePipeline(
        registry,
        base_model=BASE,
        trainer=FakeTrainer(),
        evaluator=FakeEvaluator(rating=0.66),
    )
    state = pipeline.build_state(
        new_examples=0,
        dataset_fingerprint=DATA_A,
        online_pass_rate=0.42,
        hours_since_last_train=48.0,
    )
    assert verify_data(state, evaluate_triggers(state)).startswith("数据集与生产版本一致")
    outcome = pipeline.run(state)
    assert outcome.plan is not None
    detail = next(item for item in outcome.timeline if item["step"] == "verify_data")
    assert "同数据重训" in detail["detail"]
    assert outcome.promoted is True


def test_verify_data_reports_fingerprint_change_and_baseline_state() -> None:
    """说明里要能区分"指纹已变化"与"未设基线"（两者都放行，但含义不同）."""
    changed = verify_data(
        TriggerState(
            new_examples=0, dataset_fingerprint=DATA_B, stable_dataset_fingerprint=DATA_A
        ),
        evaluate_triggers(TriggerState(dataset_fingerprint=DATA_B)),
    )
    assert "指纹已变化" in changed
    baseline_missing = verify_data(
        TriggerState(new_examples=5, dataset_fingerprint=DATA_A),
        evaluate_triggers(TriggerState(new_examples=5)),
    )
    assert "未设基线" in baseline_missing


def test_trigger_policy_override_can_block_the_run() -> None:
    """把增量门槛调到 100 之后，30 条增量不再触发训练.

    这里刻意让数据集指纹与生产版本一致（同数据重训），否则
    ``dataset_change`` 触发器会独立点火，掩盖门槛的效果。
    """
    pipeline = ContinualFinetunePipeline(
        registry_with_head(dataset=DATA_A),
        base_model=BASE,
        trigger_policy=TriggerPolicy(min_new_examples=100),
        trainer=FakeTrainer(),
        evaluator=FakeEvaluator(),
    )
    outcome = pipeline.run(pipeline.build_state(new_examples=30, dataset_fingerprint=DATA_A))
    assert outcome.plan is None
    assert outcome.trained is False


# --------------------------------------------------------------------------- #
# RetrainOutcome 投影
# --------------------------------------------------------------------------- #


def test_outcome_to_dict_is_json_serialisable_and_complete() -> None:
    """投影要能被 ``json.dumps`` 直接消费（API 端点会这么做）."""
    import json

    pipeline = ContinualFinetunePipeline(
        registry_with_head(),
        base_model=BASE,
        trainer=FakeTrainer(),
        evaluator=FakeEvaluator(),
    )
    outcome = pipeline.run(pipeline.build_state(new_examples=30, dataset_fingerprint=DATA_B))
    payload = outcome.to_dict()
    assert json.loads(json.dumps(payload, ensure_ascii=False))["promoted"] is True
    assert set(payload) == {
        "decision",
        "plan",
        "version",
        "promotion",
        "trained",
        "promoted",
        "skipped_reason",
        "timeline",
    }
    assert payload["version"]["version"] == "1.1.0"
    assert payload["promotion"]["action"] == "promote"


def test_outcome_summary_line_mentions_production_version_only_when_promoted() -> None:
    """摘要里"生产版本"一列只在真的上线时才印出版本号."""
    idle = RetrainOutcome(decision=evaluate_triggers(TriggerState()))
    assert "未启动训练" in idle.summary_line()

    registry = registry_with_head(rating=0.70, dataset=DATA_A)
    pipeline = ContinualFinetunePipeline(
        registry,
        base_model=BASE,
        trainer=FakeTrainer(),
        evaluator=FakeEvaluator(rating=0.50),
    )
    outcome = pipeline.run(pipeline.build_state(new_examples=30, dataset_fingerprint=DATA_A))
    assert outcome.promoted is False
    assert "（未采纳）" in outcome.summary_line()


def test_outcome_summary_line_with_plan_but_no_version() -> None:
    """有 plan 但还没登记（dry_run）时摘要仍可用（不抛 AttributeError）."""
    pipeline = ContinualFinetunePipeline(registry_with_head(), base_model=BASE)
    outcome = pipeline.run(
        pipeline.build_state(new_examples=30, dataset_fingerprint=DATA_B), dry_run=True
    )
    line = outcome.summary_line()
    assert "计划 v1.1.0" in line
    assert "已训练 False" in line


# --------------------------------------------------------------------------- #
# 与 day057 的接桥
# --------------------------------------------------------------------------- #


def test_pipeline_dataset_fingerprint_matches_domain_data_manifest() -> None:
    """**接桥测试**：模型版本里的数据指纹必须与 day057 清单里的指纹是同一个值.

    否则"这个效果是哪份数据训出来的"这句话在跨天之后就不成立了
    （两份指纹各自漂移，谁也说不清哪一个才是训练时用的）。
    """
    fingerprint = pipeline_dataset_fingerprint()
    assert len(fingerprint) == 16
    assert all(char in "0123456789abcdef" for char in fingerprint)

    from smart_research_agent.domain_data.pipeline import default_pipeline

    assert default_pipeline().run_from_sources().manifest.fingerprint == fingerprint


def test_registered_version_uses_domain_data_fingerprint(tmp_path: Path) -> None:
    """落盘一条版本记录，重载后三元组三项齐全（含真实的数据集指纹）."""
    index = tmp_path / "versions.jsonl"
    registry = ModelRegistry(index)
    registry.register(
        make(
            "1.0.0",
            stage=STAGE_STABLE,
            dataset=pipeline_dataset_fingerprint(),
        )
    )
    reloaded = ModelRegistry(index)
    head = reloaded.head()
    assert head is not None
    assert head.dataset_fingerprint == pipeline_dataset_fingerprint()
    assert head.triple.describe().count("|") == 3
