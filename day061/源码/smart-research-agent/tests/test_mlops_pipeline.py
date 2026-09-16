"""day059 端到端流水线测试（M5-D10）：六阶段、blocked ≠ failed、版本复用与卡片一致.

这个文件守的是三件编排层的核心性质：

1. **``blocked`` 不是 ``failed``**。门禁拦下发布时 ``publish`` 的状态是
   ``blocked``、``outcome.failed`` 为 ``False``、追踪器里的 run 状态是
   ``finished``。把它报成 ``failed`` 会让 CI 的红灯掩盖真正需要看的门禁报告；
2. **回调契约缺失必须响亮**。训练回调缺 ``adapter_sha256``、评估回调缺
   ``eval_pass_rate`` 都要立刻报错——否则"评估器写错了"会被伪装成
   "效果不达标"，而后者会让人去调超参；
3. **卡片版本号必须等于注册表里的版本号**。同一份产物重跑时，卡片若仍印着
   新算出来的号，就会出现"同一个东西两个编号"——**它不会报错，只会让追溯失效**。

失败路径同样是重点：任何阶段抛异常都要**留档**（run 被标为 failed 且错误
写进 tags），因为失败那次 run 的参数才是复现问题所需要的东西。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from smart_research_agent.mlops import (
    ARTIFACT_ADAPTER,
    ARTIFACT_GATE_REPORT,
    ARTIFACT_METRICS,
    ARTIFACT_MODEL_CARD,
    ARTIFACT_VERSION,
    PIPELINE_STAGES,
    STAGE_BLOCKED,
    STAGE_EVALUATE,
    STAGE_FAILED,
    STAGE_GATE,
    STAGE_INGEST,
    STAGE_OK,
    STAGE_PACKAGE,
    STAGE_PUBLISH,
    STAGE_TRAIN,
    STATUS_FAILED,
    STATUS_FINISHED,
    TRAIN_REQUIRED_FIELDS,
    ExperimentTracker,
    FinetunePipeline,
    MLOpsError,
    ReleaseGates,
)
from smart_research_agent.registry import (
    STAGE_STABLE,
    ModelRegistry,
    ModelVersion,
    VersionTriple,
)

BASE_MODEL = "Qwen3-8B"
DATA_A = "3f1b0c9d7e5a2468"
DATA_B = "aa77c31e90f4b258"
ADAPTER_OLD = "1a2b3c4d5e6f70819293a4b5c6d7e8f90123456789abcdef" + "0" * 8
ADAPTER_NEW = "5c6d7e8f9012345678" + "b" * 46
PARAMS = {"lora_r": 8, "learning_rate": 1e-4}


def seeded_registry(*, adapter: str = ADAPTER_OLD, dataset: str = DATA_A, rating: float = 0.60) -> ModelRegistry:
    """一个有生产版本的内存注册表（演练起点）."""
    registry = ModelRegistry()
    registry.register(
        ModelVersion(
            triple=VersionTriple(
                base_model=BASE_MODEL, adapter_sha256=adapter, dataset_fingerprint=dataset
            ),
            version="1.0.0",
            stage=STAGE_STABLE,
            metrics={"eval_pass_rate": rating},
            artifacts={"adapter": "a/baseline", "merged": "m/baseline"},
        )
    )
    return registry


def make_train(
    adapter: str = ADAPTER_NEW,
    *,
    merged: bool = True,
    size_mib: float = 0.05,
    extra: dict | None = None,
):
    """确定性训练回调."""

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
            **(extra or {}),
        }

    return train


def make_evaluate(pass_rate: float = 0.7222, baseline: float = 0.60, **extra: float):
    """确定性评估回调."""

    def evaluate(artifacts: dict) -> dict:
        del artifacts
        return {
            "eval_pass_rate": pass_rate,
            "eval_pass_rate_delta": round(pass_rate - baseline, 4),
            **extra,
        }

    return evaluate


def build_pipeline(
    registry: ModelRegistry | None = None,
    tracker: ExperimentTracker | None = None,
    **kwargs,
):  # type: ignore[no-untyped-def]
    """装配一条缺省可达标的流水线（各用例只覆盖要改的那一项）."""
    kwargs.setdefault("train", make_train())
    kwargs.setdefault("evaluate", make_evaluate())
    kwargs.setdefault("gates", ReleaseGates())
    return FinetunePipeline(
        registry if registry is not None else seeded_registry(),
        tracker if tracker is not None else ExperimentTracker(),
        base_model=BASE_MODEL,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# 构造期
# --------------------------------------------------------------------------- #


def test_pipeline_rejects_blank_base_model() -> None:
    """基座名是三元组的一项，不能空。"""
    with pytest.raises(MLOpsError, match="base_model 不能为空"):
        FinetunePipeline(ModelRegistry(), ExperimentTracker(), base_model="  ")


def test_train_required_fields_constant() -> None:
    """回调契约里唯一必填的字段就是适配器哈希。"""
    assert TRAIN_REQUIRED_FIELDS == ("adapter_sha256",)


# --------------------------------------------------------------------------- #
# 空指纹
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fingerprint", ["", "   "])
def test_blank_dataset_fingerprint_is_rejected(fingerprint: str) -> None:
    """空指纹连三元组都构不成——必须在训练之前拦下。"""
    pipeline = build_pipeline()
    with pytest.raises(MLOpsError, match="dataset_fingerprint 不能为空"):
        pipeline.run(dataset_fingerprint=fingerprint)


# --------------------------------------------------------------------------- #
# dry_run 与真正发布
# --------------------------------------------------------------------------- #


def test_dry_run_publishes_nothing_but_marks_publish_blocked() -> None:
    """``dry_run=True``：六阶段照跑，``publish`` 是 ``blocked``，注册表规模不变。"""
    registry = seeded_registry()
    outcome = build_pipeline(registry).run(
        dataset_fingerprint=DATA_B, params=PARAMS, dry_run=True
    )
    assert [item.name for item in outcome.stages] == list(PIPELINE_STAGES)
    assert [item.status for item in outcome.stages] == [
        STAGE_OK,
        STAGE_OK,
        STAGE_OK,
        STAGE_OK,
        STAGE_OK,
        STAGE_BLOCKED,
    ]
    assert outcome.published is False
    assert outcome.failed is False
    assert registry.counts()["total"] == 1
    assert "dry_run" in outcome.skipped_reason
    assert outcome.run.status == STATUS_FINISHED


def test_publish_creates_a_stable_version_linked_to_the_baseline() -> None:
    """真正发布：新版本进注册表、升为 stable、父版本指向基线。"""
    registry = seeded_registry()
    outcome = build_pipeline(registry).run(dataset_fingerprint=DATA_B, params=PARAMS)
    assert outcome.published is True
    assert outcome.failed is False
    assert registry.counts()["total"] == 2
    assert registry.head().version == outcome.version.version
    assert outcome.version.stage == STAGE_STABLE
    assert outcome.version.parent_version == "1.0.0"
    assert outcome.version.tags["run_id"] == outcome.run.run_id
    # 注意两个命名空间：阶段产物叫 adapter_sha256，注册表的产物槽位叫 adapter
    assert outcome.version.artifacts["adapter"]
    assert outcome.version.artifacts["dataset"] == DATA_B


def test_publish_records_the_commit_in_the_run_tags() -> None:
    """提交号写进 run 的 tags（追溯"这一版是哪次提交跑的"）。"""
    outcome = build_pipeline().run(
        dataset_fingerprint=DATA_B, params=PARAMS, commit="deadbeef"
    )
    assert outcome.run.tags["commit"] == "deadbeef"
    assert outcome.card.commit == "deadbeef"


def test_ingest_reports_dataset_change_against_the_baseline() -> None:
    """``ingest`` 阶段说明里带基线版本与"数据集已变化/未变化"。"""
    changed = build_pipeline().run(dataset_fingerprint=DATA_B, params=PARAMS)
    assert "数据集已变化" in changed.stage(STAGE_INGEST).detail
    same = build_pipeline().run(dataset_fingerprint=DATA_A, params=PARAMS)
    assert "数据集未变化" in same.stage(STAGE_INGEST).detail


def test_ingest_without_baseline_says_so() -> None:
    """没有基线时 ``ingest`` 明确写出"尚无 stable"。"""
    outcome = build_pipeline(ModelRegistry()).run(dataset_fingerprint=DATA_B, params=PARAMS)
    assert "尚无 stable" in outcome.stage(STAGE_INGEST).detail
    assert outcome.version.parent_version == ""


# --------------------------------------------------------------------------- #
# 阶段产物
# --------------------------------------------------------------------------- #


def test_stages_declare_the_documents_the_next_stage_needs() -> None:
    """每个阶段的 ``produced`` 里带着下游需要的 key（依赖表与实际执行一致）。"""
    outcome = build_pipeline().run(dataset_fingerprint=DATA_B, params=PARAMS)
    assert ARTIFACT_ADAPTER in outcome.stage(STAGE_TRAIN).produced
    assert ARTIFACT_METRICS in outcome.stage(STAGE_EVALUATE).produced
    assert ARTIFACT_GATE_REPORT in outcome.stage(STAGE_GATE).produced
    assert ARTIFACT_MODEL_CARD in outcome.stage(STAGE_PACKAGE).produced
    assert ARTIFACT_VERSION in outcome.stage(STAGE_PUBLISH).produced


def test_duration_is_recorded_in_milliseconds() -> None:
    """耗时用毫秒（六步里四步是纯计算，用秒会得到一列全 0 的耗时列）。"""
    outcome = build_pipeline().run(dataset_fingerprint=DATA_B, params=PARAMS)
    assert all(item.duration_ms >= 0 for item in outcome.stages)
    assert any(item.duration_ms > 0 for item in outcome.stages)


def test_stage_lookup_returns_none_for_unknown_name() -> None:
    """``outcome.stage`` 对不存在的名字返回 ``None``（便于调用方兜底）。"""
    outcome = build_pipeline().run(dataset_fingerprint=DATA_B, params=PARAMS)
    assert outcome.stage("deploy") is None


# --------------------------------------------------------------------------- #
# 门禁阻塞
# --------------------------------------------------------------------------- #


def test_gate_failure_blocks_publish_without_failing_the_pipeline() -> None:
    """门禁不通过：``gate`` 与 ``publish`` 都是 ``blocked``，``failed`` 为 False."""
    outcome = build_pipeline(evaluate=make_evaluate(0.30)).run(
        dataset_fingerprint=DATA_B, params=PARAMS
    )
    assert outcome.stage(STAGE_GATE).status == STAGE_BLOCKED
    assert outcome.stage(STAGE_PUBLISH).status == STAGE_BLOCKED
    assert outcome.published is False
    assert outcome.failed is False
    assert "门禁不通过" in outcome.skipped_reason
    assert outcome.run.status == STATUS_FINISHED


def test_gate_failure_still_produces_the_card_and_manifest() -> None:
    """门禁不通过**仍然打包**：那份报告正是要给人看的东西。"""
    outcome = build_pipeline(evaluate=make_evaluate(0.30)).run(
        dataset_fingerprint=DATA_B, params=PARAMS
    )
    assert outcome.stage(STAGE_PACKAGE).status == STAGE_OK
    assert outcome.card is not None and outcome.card.gate_passed is False
    assert outcome.manifest["gate_passed"] is False


def test_missing_merged_artifact_blocks_on_completeness() -> None:
    """缺合并产物 → 门禁在 ``artifact_completeness`` 上阻塞（且不是失败）。"""
    outcome = build_pipeline(train=make_train(merged=False)).run(
        dataset_fingerprint=DATA_B, params=PARAMS
    )
    assert outcome.gate.blocking_failures[0].name == "artifact_completeness"
    assert outcome.published is False
    assert outcome.failed is False


def test_oversized_adapter_blocks_publish() -> None:
    """适配器体积超限 → 阻塞（LoRA 不该长到与基座同量级）。"""
    outcome = build_pipeline(train=make_train(size_mib=80.0)).run(
        dataset_fingerprint=DATA_B, params=PARAMS
    )
    assert "adapter_size" in [item.name for item in outcome.gate.blocking_failures]


def test_gates_can_be_relaxed_via_the_policy() -> None:
    """放宽门禁后同一个候选会被放行（策略是可注入的）。"""
    outcome = build_pipeline(
        train=make_train(merged=False),
        gates=ReleaseGates(require_artifacts=False),
    ).run(dataset_fingerprint=DATA_B, params=PARAMS)
    assert outcome.published is True


# --------------------------------------------------------------------------- #
# 版本复用与卡片一致性
# --------------------------------------------------------------------------- #


def test_rerun_with_the_same_adapter_reuses_the_version_number() -> None:
    """同一份产物重跑：注册表不新增版本，且跳过一次阶段流转."""
    registry = seeded_registry()
    pipeline = build_pipeline(registry)
    first = pipeline.run(dataset_fingerprint=DATA_B, params=PARAMS)
    second = pipeline.run(dataset_fingerprint=DATA_B, params=PARAMS)
    assert registry.counts()["total"] == 2
    assert second.version.version == first.version.version
    assert second.published is True


def test_rerun_card_version_matches_the_registry_record() -> None:
    """**一致性回归测试**：重跑时卡片的版本号必须等于注册表里的版本号.

    若不这么做，卡片会印出新算出来的号（例如 v1.1.1），而注册表里那条记录
    是 v1.1.0——两者指向同一条记录。"同一个东西两个编号"不会报错，
    只会让追溯失效。
    """
    registry = seeded_registry()
    pipeline = build_pipeline(registry)
    pipeline.run(dataset_fingerprint=DATA_B, params=PARAMS)
    second = pipeline.run(dataset_fingerprint=DATA_B, params=PARAMS)
    assert second.card.version == second.version.version
    assert "复用版本号" in second.card.notes


def test_new_adapter_produces_a_new_patch_version() -> None:
    """换了适配器 → 新三元组 → patch 号；同数据同基座的重训是最可比的一类变更。"""
    registry = seeded_registry()
    baseline = registry.head()
    assert baseline.dataset_fingerprint == DATA_A
    first = build_pipeline(registry).run(dataset_fingerprint=DATA_A, params=PARAMS)
    assert first.version.version == "1.0.1"
    assert first.version.tags["bump_kind"] == "patch"


# --------------------------------------------------------------------------- #
# 回调契约
# --------------------------------------------------------------------------- #


def test_missing_train_callback_raises_and_is_recorded() -> None:
    """未注入训练回调 → 失败留档（run 被标 failed，错误写进 tags）。"""
    pipeline = FinetunePipeline(
        seeded_registry(), ExperimentTracker(), base_model=BASE_MODEL, evaluate=make_evaluate()
    )
    outcome = pipeline.run(dataset_fingerprint=DATA_B, params=PARAMS)
    assert outcome.failed is True
    assert outcome.stage(STAGE_TRAIN).status == STAGE_FAILED
    assert outcome.run.status == STATUS_FAILED
    assert "未注入 train 回调" in outcome.run.tags["error"]


def test_missing_evaluate_callback_raises_and_is_recorded() -> None:
    """未注入评估回调 → 同样留档（没有评估就没有门禁的判据）。"""
    pipeline = FinetunePipeline(
        seeded_registry(), ExperimentTracker(), base_model=BASE_MODEL, train=make_train()
    )
    outcome = pipeline.run(dataset_fingerprint=DATA_B, params=PARAMS)
    assert outcome.failed is True
    assert outcome.stage(STAGE_EVALUATE).status == STAGE_FAILED
    assert "未注入 evaluate 回调" in outcome.run.tags["error"]


def test_train_callback_without_adapter_hash_fails_loudly() -> None:
    """训练回调缺适配器哈希 → 报错（没有它就算不出版本键）。"""
    outcome = build_pipeline(train=lambda params: {"step": 1}).run(
        dataset_fingerprint=DATA_B, params=PARAMS
    )
    assert outcome.failed is True
    assert "adapter_sha256" in outcome.stage(STAGE_TRAIN).detail


def test_evaluate_callback_without_pass_rate_fails_loudly() -> None:
    """评估回调缺合格率 → 报错（否则"评估器写错了"会被伪装成"效果不达标"）."""
    outcome = build_pipeline(evaluate=lambda artifacts: {"latency_ms": 12.0}).run(
        dataset_fingerprint=DATA_B, params=PARAMS
    )
    assert outcome.failed is True
    assert "eval_pass_rate" in outcome.stage(STAGE_EVALUATE).detail


def test_train_callback_exception_is_captured() -> None:
    """回调抛异常也要留档（失败那次 run 的参数才是复现问题所需的东西）。"""

    def boom(params: dict) -> dict:
        raise ValueError("CUDA out of memory")

    outcome = build_pipeline(train=boom).run(dataset_fingerprint=DATA_B, params=PARAMS)
    assert outcome.failed is True
    assert "CUDA out of memory" in outcome.run.tags["error"]
    assert outcome.stage(STAGE_TRAIN).status == STAGE_FAILED


def test_failure_lands_on_the_next_stage_name() -> None:
    """失败被记在"下一个该执行的阶段"上（而不是一个名字随便取的阶段）.

    没有训练回调时，前两个阶段只有 ``ingest`` 会留下记录，
    因此失败落在 ``train``——而 ``StageResult`` 会因未知阶段名直接抛
    ``MLOpsError``，那是"处理异常时又抛异常"，最难查的一类。
    """
    pipeline = FinetunePipeline(
        seeded_registry(), ExperimentTracker(), base_model=BASE_MODEL, evaluate=make_evaluate()
    )
    outcome = pipeline.run(dataset_fingerprint=DATA_B, params=PARAMS)
    assert [item.name for item in outcome.stages] == [STAGE_INGEST, STAGE_TRAIN]
    assert outcome.stages[-1].status == STAGE_FAILED
    assert outcome.stages[-1].name == STAGE_TRAIN


def test_failure_after_training_lands_on_evaluate() -> None:
    """训练成功但评估回调缺失时，失败落在 ``evaluate``（第三条记录）。"""
    pipeline = FinetunePipeline(
        seeded_registry(), ExperimentTracker(), base_model=BASE_MODEL, train=make_train()
    )
    outcome = pipeline.run(dataset_fingerprint=DATA_B, params=PARAMS)
    assert [item.name for item in outcome.stages] == [STAGE_INGEST, STAGE_TRAIN, STAGE_EVALUATE]
    assert outcome.stages[-1].status == STAGE_FAILED


# --------------------------------------------------------------------------- #
# 投影与追踪器
# --------------------------------------------------------------------------- #


def test_outcome_projection_is_json_ready() -> None:
    """投影能被 ``json.dumps`` 直接消费，且字段齐全。"""
    import json

    outcome = build_pipeline().run(dataset_fingerprint=DATA_B, params=PARAMS)
    payload = outcome.to_dict()
    assert json.loads(json.dumps(payload, ensure_ascii=False))["published"] is True
    assert set(payload) == {
        "run",
        "stages",
        "gate",
        "card",
        "manifest",
        "version",
        "published",
        "failed",
        "skipped_reason",
    }
    assert payload["card"]["version"] == payload["version"]["version"]


def test_outcome_summary_and_markdown() -> None:
    """摘要与 markdown 渲染（六阶段表 + 门禁 + 卡片）。"""
    outcome = build_pipeline().run(dataset_fingerprint=DATA_B, params=PARAMS)
    line = outcome.summary_line()
    assert "已发布" in line and "全部完成" in line
    markdown = outcome.render_markdown()
    assert "微调流水线运行" in markdown
    assert "发布门禁：通过" in markdown
    assert "| 1 | `ingest` |" in markdown


def test_tracker_receives_metrics_artifacts_and_gate_flags() -> None:
    """追踪器里能查到全部指标、产物与门禁结论标记。"""
    tracker = ExperimentTracker()
    outcome = build_pipeline(tracker=tracker).run(dataset_fingerprint=DATA_B, params=PARAMS)
    run = tracker.get(outcome.run.run_id)
    flat = run.flat_metrics()
    assert flat["eval_pass_rate"] == pytest.approx(0.7222)
    assert flat["train_loss"] == pytest.approx(0.3187)
    assert flat["gate_pass_rate"] == 1.0
    assert "adapter" in run.artifacts and "model_card" in run.artifacts
    assert run.status == STATUS_FINISHED


def test_tracker_is_persisted(tmp_path: Path) -> None:
    """追踪器落盘后可重载（CI 的构建产物就是它）。"""
    tracker = ExperimentTracker(tmp_path / "runs.jsonl")
    outcome = build_pipeline(tracker=tracker).run(dataset_fingerprint=DATA_B, params=PARAMS)
    reloaded = ExperimentTracker(tmp_path / "runs.jsonl")
    assert reloaded.get(outcome.run.run_id).metric("eval_pass_rate") == pytest.approx(0.7222)


def test_train_callback_with_only_the_required_field_works() -> None:
    """回调只给最低要求的字段也要能跑通（其余指标/产物都是可选的）.

    这条用例同时覆盖"没有 ``metrics`` 就不写指标"与"某个可选字段缺席"
    两条支路——**契约只强制一样东西，因此其余路径都必须能被走通**。
    """

    def minimal(params: dict) -> dict:
        del params
        return {"adapter_sha256": ADAPTER_NEW}

    tracker = ExperimentTracker()
    outcome = build_pipeline(train=minimal, tracker=tracker).run(
        dataset_fingerprint=DATA_B, params=PARAMS
    )
    assert outcome.failed is False
    run = tracker.get(outcome.run.run_id)
    # 只有评估回调给的指标进了追踪器；train 侧一个都没记
    assert "train_loss" not in run.flat_metrics()
    assert run.flat_metrics()["eval_pass_rate"] == pytest.approx(0.7222)


def test_markdown_renders_for_a_failed_run_without_gate_or_card() -> None:
    """失败时（没有门禁报告、没有模型卡）渲染也不能崩——那份报告正是要看的."""
    pipeline = FinetunePipeline(
        seeded_registry(), ExperimentTracker(), base_model=BASE_MODEL, evaluate=make_evaluate()
    )
    outcome = pipeline.run(dataset_fingerprint=DATA_B, params=PARAMS)
    assert outcome.gate is None and outcome.card is None
    markdown = outcome.render_markdown()
    assert "微调流水线运行" in markdown
    assert "发布门禁" not in markdown  # 没有门禁时不编造一段


def test_run_id_is_stable_across_identical_runs() -> None:
    """参数相同的两次运行共用同一个 run_id（可复盘、不产生内容重复的副本）."""
    first = build_pipeline().run(dataset_fingerprint=DATA_B, params=PARAMS)
    second = build_pipeline().run(dataset_fingerprint=DATA_B, params=PARAMS)
    assert first.run.run_id == second.run.run_id


def test_run_id_differs_when_params_differ() -> None:
    """参数变了就是另一次实验。"""
    first = build_pipeline().run(dataset_fingerprint=DATA_B, params=PARAMS)
    second = build_pipeline().run(dataset_fingerprint=DATA_B, params={"lora_r": 16})
    assert first.run.run_id != second.run.run_id
