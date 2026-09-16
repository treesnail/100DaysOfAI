"""端到端微调流水线：把六步串成一次可复盘的运行（M5-D10）.

前九天各自解决了一段：day050~day052 训练、day053 评估、day057 数据、
day058 版本与采纳、今天的追踪与门禁。本模块把它们串成**一次运行**：

```text
ingest → train → evaluate → gate → package → publish
取数据    训练     评估       门禁     打包      发布
    ↓        ↓        ↓         ↓        ↓        ↓
  指纹    适配器    指标     门禁报告  模型卡   版本记录
```

## 三个贯穿全模块的取舍

### 1. 门禁不通过**不是失败**

门禁拦下发布时，``publish`` 的状态是 ``blocked``（不是 ``failed``），
追踪器里的 run 状态是 ``finished``（不是 ``failed``）。
理由在 ``stages`` 的 docstring 里写过一次，这里再钉一次：

> **``failed`` 意味着"有东西坏了，需要人去修"，``blocked`` 意味着
> "上游判定不允许我执行，这是预期内的结果"。** 把后者报成前者，
> 会让 CI 的红灯掩盖真正需要看的那份门禁报告——而那份报告就在同一个
> 构建产物里，被一个"失败"的标签挡住了。

### 2. 产物与指标来自**注入的回调**，流水线不持有模型

```python
train(params) -> {"adapter_sha256": ..., "adapter_mebibytes": ...,
                  "train_loss": ..., "artifacts": {...}}
evaluate(artifacts) -> {"eval_pass_rate": ..., "eval_pass_rate_delta": ...}
```

这与 day058 ``ContinualFinetunePipeline`` 的取舍一致：编排层的职责是
**顺序、记录与判定**，不是训练。好处是本模块的全部用例都是毫秒级的
纯逻辑测试，而"真的训一次"由 ``scripts/mlops_demo.py`` 承担。

### 3. ``dry_run`` 跳过的是**发布**，不是训练

day058 的 ``dry_run`` 是"只到计划为止"（因为那一步的昂贵动作是训练）。
今天的昂贵动作同样是训练，但今天多了一个更值得演练的东西：**发布**。
``dry_run=True`` 会照常训练、评估、门禁、打包，只是**不写注册表**——
它回答的问题是"如果现在发，门禁会不会放行？"。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.mlops.errors import MLOpsError
from smart_research_agent.mlops.gates import (
    GateReport,
    ReleaseGates,
    evaluate_gates,
)
from smart_research_agent.mlops.packaging import (
    MODEL_CARD_FILENAME,
    RELEASE_MANIFEST_FILENAME,
    ModelCard,
    build_model_card,
    build_release_manifest,
)
from smart_research_agent.mlops.stages import (
    ARTIFACT_ADAPTER,
    ARTIFACT_DATASET,
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
    StageResult,
)
from smart_research_agent.mlops.tracking import (
    STATUS_FINISHED,
    ExperimentTracker,
    Run,
)
from smart_research_agent.registry import (
    ARTIFACT_ADAPTER as REGISTRY_ARTIFACT_ADAPTER,
    ARTIFACT_DATASET as REGISTRY_ARTIFACT_DATASET,
    ARTIFACT_MERGED as REGISTRY_ARTIFACT_MERGED,
    STAGE_STABLE,
    ModelRegistry,
    ModelVersion,
    RegistryError,
    VersionTriple,
)
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: 训练回调的必填字段（缺了它就没有版本键，也就没有版本记录）。
TRAIN_REQUIRED_FIELDS: tuple[str, ...] = ("adapter_sha256",)

#: 评估回调的必填字段（门禁第一项就靠它）。
EVALUATE_REQUIRED_FIELDS: tuple[str, ...] = ("eval_pass_rate",)

#: 追踪器里记录的产物名（与阶段声明的产物名刻意分开：
#: 前者是"文件路径"，后者是"阶段之间传递的东西"）。
TRACK_ARTIFACT_ADAPTER = "adapter"
TRACK_ARTIFACT_MERGED = "merged"
TRACK_ARTIFACT_MODEL_CARD = "model_card"
TRACK_ARTIFACT_MANIFEST = "release_manifest"


def _elapsed_ms(start: float) -> float:
    """从 ``start`` 到现在经过的毫秒数（保留三位小数）."""
    return round((time.perf_counter() - start) * 1000, 3)


@dataclass
class PipelineOutcome:
    """一次流水线运行的完整产物（含逐阶段结果，便于复盘"哪一步决定了什么"）."""

    run: Run
    stages: list[StageResult] = field(default_factory=list)
    gate: GateReport | None = None
    card: ModelCard | None = None
    manifest: dict[str, Any] = field(default_factory=dict)
    version: ModelVersion | None = None
    published: bool = False
    skipped_reason: str = ""

    @property
    def failed(self) -> bool:
        """是否有阶段**失败**（``blocked`` 与 ``skipped`` 都不算）."""
        return any(item.status == STAGE_FAILED for item in self.stages)

    def stage(self, name: str) -> StageResult | None:
        """按名字取阶段结果（不存在返回 ``None``）."""
        return next((item for item in self.stages if item.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return {
            "run": self.run.to_dict(),
            "stages": [item.to_dict() for item in self.stages],
            "gate": None if self.gate is None else self.gate.to_dict(),
            "card": None if self.card is None else self.card.to_dict(),
            "manifest": dict(self.manifest),
            "version": None if self.version is None else self.version.to_dict(),
            "published": self.published,
            "failed": self.failed,
            "skipped_reason": self.skipped_reason,
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        published = (
            f"已发布 v{self.version.version}" if self.published else "未发布"
        )
        return (
            f"{self.run.run_id} | {'有阶段失败' if self.failed else '全部完成'} | "
            f"{published} | "
            f"门禁 {'通过' if self.gate is not None and self.gate.passed else '不通过/未执行'}"
        )

    def render_markdown(self) -> str:
        """把运行渲染成 markdown（六阶段表 + 门禁报告 + 模型卡）."""
        lines = [
            f"# 微调流水线运行 `{self.run.run_id}`",
            "",
            f"- 结论：**{self.summary_line()}**",
            f"- 备注：{self.skipped_reason or '（无）'}",
            "",
            "| 序 | 阶段 | 状态 | 耗时(ms) | 说明 |",
            "|----|------|------|----------|------|",
        ]
        lines.extend(
            f"| {index} | `{item.name}` | {item.status} | {item.duration_ms:.2f} | {item.detail} |"
            for index, item in enumerate(self.stages, start=1)
        )
        lines.append("")
        if self.gate is not None:
            lines.append(self.gate.render_markdown())
        if self.card is not None:
            lines.append(self.card.render_markdown())
        return "\n".join(lines)


class FinetunePipeline:
    """端到端微调流水线（六阶段，可注入训练/评估回调）.

    用法::

        pipeline = FinetunePipeline(
            registry, tracker, base_model="Qwen3-8B",
            train=train_fn, evaluate=eval_fn,
        )
        outcome = pipeline.run(dataset_fingerprint=fingerprint, params={"lora_r": 8})
        if outcome.published:
            print("已发布", outcome.version.version)
    """

    def __init__(
        self,
        registry: ModelRegistry,
        tracker: ExperimentTracker,
        *,
        base_model: str,
        train: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        evaluate: Callable[[dict[str, str]], dict[str, float]] | None = None,
        gates: ReleaseGates | None = None,
        run_name: str = "finetune",
    ):
        if not base_model.strip():
            raise MLOpsError("base_model 不能为空（它是版本三元组的一项）")
        self.registry = registry
        self.tracker = tracker
        self.base_model = base_model.strip()
        self.train = train
        self.evaluate = evaluate
        self.gates = gates or ReleaseGates()
        self.run_name = run_name

    # ------------------------------------------------------------------ 运行
    def run(  # noqa: C901 - 六阶段顺序展开写，是这一课的核心产物
        self,
        *,
        dataset_fingerprint: str,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
        commit: str = "",
        dry_run: bool = False,
        publish: bool = True,
    ) -> PipelineOutcome:
        """跑一次完整流水线（``dry_run=True`` 时训练照跑，但不写注册表）."""
        if not dataset_fingerprint.strip():
            raise MLOpsError(
                "dataset_fingerprint 不能为空：版本三元组需要它来回答"
                "'这个效果是哪份数据训出来的'，缺了它登记的记录无法追溯"
            )
        run_params = dict(params or {})
        run_params["dataset_fingerprint"] = dataset_fingerprint
        run_params["base_model"] = self.base_model
        run = self.tracker.start_run(
            self.run_name,
            params=run_params,
            tags={
                **{str(k): str(v) for k, v in (tags or {}).items()},
                **({"commit": commit} if commit else {}),
            },
        )
        outcome = PipelineOutcome(run=run)

        try:
            baseline = self.registry.head()
            # ---- 阶段 1：取数据
            start = time.perf_counter()
            changed = baseline is not None and baseline.dataset_fingerprint != dataset_fingerprint
            outcome.stages.append(
                StageResult(
                    name=STAGE_INGEST,
                    status=STAGE_OK,
                    detail=(
                        f"数据指纹 {dataset_fingerprint} | "
                        f"基线 {baseline.version if baseline else '（尚无 stable）'} | "
                        f"数据集{'已变化' if changed else '未变化'}"
                    ),
                    duration_ms=_elapsed_ms(start),
                    produced={
                        ARTIFACT_DATASET: dataset_fingerprint,
                        "baseline_version": "" if baseline is None else baseline.version,
                    },
                )
            )

            # ---- 阶段 2：训练
            if self.train is None:
                raise MLOpsError("未注入 train 回调：流水线不自己训练")
            start = time.perf_counter()
            train_result = dict(self.train(run_params))
            missing = [
                name for name in TRAIN_REQUIRED_FIELDS if not train_result.get(name)
            ]
            if missing:
                raise MLOpsError(
                    f"训练回调缺少必需字段 {', '.join(missing)}："
                    "没有适配器内容哈希就算不出版本键，而版本键是版本记录的身份证"
                )
            adapter_sha256 = str(train_result["adapter_sha256"])
            artifacts = {
                str(k): str(v) for k, v in dict(train_result.get("artifacts", {})).items()
            }
            train_metrics = {
                name: float(value)
                for name, value in dict(train_result.get("metrics", {})).items()
            }
            for extra in ("train_loss", "adapter_mebibytes", "step"):
                if extra in train_result:
                    train_metrics[extra] = float(train_result[extra])
            if train_metrics:
                self.tracker.log_metrics(run, train_metrics)
            for name, path in artifacts.items():
                self.tracker.log_artifact(run, name, path)
            outcome.stages.append(
                StageResult(
                    name=STAGE_TRAIN,
                    status=STAGE_OK,
                    detail=(
                        f"适配器 sha256:{adapter_sha256[:12]} | "
                        f"{len(artifacts)} 个产物 | {sorted(train_metrics)}"
                    ),
                    duration_ms=_elapsed_ms(start),
                    produced={ARTIFACT_ADAPTER: adapter_sha256, "artifacts": artifacts},
                )
            )

            # ---- 阶段 3：评估
            if self.evaluate is None:
                raise MLOpsError("未注入 evaluate 回调：门禁唯一的判据来源就是它")
            start = time.perf_counter()
            eval_metrics = {
                str(k): float(v) for k, v in dict(self.evaluate(artifacts)).items()
            }
            missing_metrics = [
                name for name in EVALUATE_REQUIRED_FIELDS if name not in eval_metrics
            ]
            if missing_metrics:
                raise MLOpsError(
                    f"评估回调缺少必需指标 {', '.join(missing_metrics)}："
                    "门禁第一项就是合格率，缺了它门禁必然不通过——"
                    "但那会把'评估器写错了'伪装成'效果不达标'"
                )
            self.tracker.log_metrics(run, eval_metrics)
            outcome.stages.append(
                StageResult(
                    name=STAGE_EVALUATE,
                    status=STAGE_OK,
                    detail=f"指标 { {k: eval_metrics[k] for k in sorted(eval_metrics)} }",
                    duration_ms=_elapsed_ms(start),
                    produced={ARTIFACT_METRICS: eval_metrics},
                )
            )

            # ---- 阶段 4：门禁（绝对判定：缺证据就是不通过）
            start = time.perf_counter()
            gate_metrics: dict[str, Any] = dict(train_metrics)
            gate_metrics.update(eval_metrics)
            gate_metrics["dataset_fingerprint"] = dataset_fingerprint
            gate_metrics["base_model"] = self.base_model
            gate = evaluate_gates(
                gate_metrics,
                policy=self.gates,
                artifacts=artifacts,
                commit=commit,
                notes=f"run {run.run_id}",
            )
            outcome.gate = gate
            self.tracker.log_metrics(
                run,
                {
                    f"gate_{name}": (1.0 if passed else 0.0)
                    for name, passed in (
                        (item.name, item.passed) for item in gate.checks
                    )
                },
            )
            outcome.stages.append(
                StageResult(
                    name=STAGE_GATE,
                    status=STAGE_OK if gate.passed else STAGE_BLOCKED,
                    detail=gate.summary_line(),
                    duration_ms=_elapsed_ms(start),
                    produced={ARTIFACT_GATE_REPORT: gate.to_dict()},
                )
            )

            # ---- 阶段 5：打包（门禁不通过也打包：报告正是要给人看的东西）
            start = time.perf_counter()
            version_number, bump_kind = self.registry.propose_version(
                base_model=self.base_model, dataset_fingerprint=dataset_fingerprint
            )
            # 三元组**在打包前就算出来**：同一份产物重跑时，注册表里已经有一条
            # 记录，此时卡片必须用**已有的版本号**——否则卡片上印着 v1.1.1、
            # 注册表里是 v1.1.0，而两者指向同一条记录。
            # 这类"同一个东西两个编号"的缺陷不会报错，只会让追溯失效。
            triple = VersionTriple(
                base_model=self.base_model,
                adapter_sha256=adapter_sha256,
                dataset_fingerprint=dataset_fingerprint,
            )
            existing = self.registry.find(triple.key)
            reused_version = existing.version if existing is not None else ""
            if existing is not None:
                version_number = existing.version
                bump_kind = existing.tags.get("bump_kind", bump_kind)
            card = build_model_card(
                version=version_number,
                base_model=self.base_model,
                adapter_sha256=adapter_sha256,
                dataset_fingerprint=dataset_fingerprint,
                run=run,
                gate=gate,
                artifacts=artifacts,
                parent_version="" if baseline is None else baseline.version,
                commit=commit,
                notes=(
                    "本卡由 FinetunePipeline 自动生成；"
                    f"递增位 {bump_kind} 由三元组差异推出。"
                    + (f"（三元组已存在，复用版本号 v{reused_version}）" if reused_version else "")
                ),
            )
            manifest = build_release_manifest(
                card=card,
                run=run,
                gate=gate,
                extra={"bump_kind": bump_kind, "dry_run": dry_run or not publish},
            )
            self.tracker.log_artifact(run, TRACK_ARTIFACT_MODEL_CARD, MODEL_CARD_FILENAME)
            self.tracker.log_artifact(
                run, TRACK_ARTIFACT_MANIFEST, RELEASE_MANIFEST_FILENAME
            )
            outcome.card = card
            outcome.manifest = manifest
            outcome.stages.append(
                StageResult(
                    name=STAGE_PACKAGE,
                    status=STAGE_OK,
                    detail=(
                        f"模型卡 v{version_number}（{bump_kind}）| "
                        f"{len(card.metrics)} 个指标 | {len(card.limitations)} 条已知限制"
                    ),
                    duration_ms=_elapsed_ms(start),
                    produced={ARTIFACT_MODEL_CARD: card.to_dict(), "manifest": manifest},
                )
            )

            # ---- 阶段 6：发布（被门禁或 dry_run 阻塞时不是 failed）
            start = time.perf_counter()
            if not gate.passed:
                outcome.stages.append(
                    StageResult(
                        name=STAGE_PUBLISH,
                        status=STAGE_BLOCKED,
                        detail=(
                            "门禁不通过，未写注册表：阻塞失败 "
                            f"{[item.name for item in gate.blocking_failures]}"
                        ),
                        duration_ms=_elapsed_ms(start),
                    )
                )
                outcome.skipped_reason = "门禁不通过：这一版不该上线（报告与模型卡仍已生成）"
            elif dry_run or not publish:
                outcome.stages.append(
                    StageResult(
                        name=STAGE_PUBLISH,
                        status=STAGE_BLOCKED,
                        detail="dry_run / publish=False：只演练到打包，未写注册表",
                        duration_ms=_elapsed_ms(start),
                    )
                )
                outcome.skipped_reason = "dry_run：门禁通过，但未写注册表"
            else:
                # ``triple`` / ``existing`` 在打包阶段已经算好（同一份产物只该有一个版本号）
                if existing is None:
                    record = ModelVersion(
                        triple=triple,
                        version=version_number,
                        parent_version="" if baseline is None else baseline.version,
                        metrics={
                            "eval_pass_rate": eval_metrics.get("eval_pass_rate", 0.0),
                            **{
                                name: train_metrics[name]
                                for name in ("train_loss",)
                                if name in train_metrics
                            },
                        },
                        artifacts={
                            REGISTRY_ARTIFACT_ADAPTER: artifacts.get(TRACK_ARTIFACT_ADAPTER, ""),
                            REGISTRY_ARTIFACT_MERGED: artifacts.get(TRACK_ARTIFACT_MERGED, ""),
                            REGISTRY_ARTIFACT_DATASET: dataset_fingerprint,
                        },
                        tags={"bump_kind": bump_kind, "run_id": run.run_id},
                        notes=f"由 FinetunePipeline 登记（run {run.run_id}）",
                    )
                    outcome.version = self.registry.register(record, reason="流水线发布")
                else:
                    outcome.version = existing
                if outcome.version.stage != STAGE_STABLE:
                    outcome.version = self.registry.set_stage(
                        outcome.version.version,
                        STAGE_STABLE,
                        reason=f"发布门禁通过（run {run.run_id}）",
                    )
                outcome.published = True
                outcome.stages.append(
                    StageResult(
                        name=STAGE_PUBLISH,
                        status=STAGE_OK,
                        detail=(
                            f"v{outcome.version.version} 已发布并置为 stable"
                            f"（键 {outcome.version.version_key}）"
                        ),
                        duration_ms=_elapsed_ms(start),
                        produced={ARTIFACT_VERSION: outcome.version.to_dict()},
                    )
                )

            self.tracker.finish(
                run,
                status=STATUS_FINISHED,
                notes=outcome.skipped_reason or "六个阶段全部完成",
            )
        except (MLOpsError, RegistryError, KeyError, TypeError, ValueError) as exc:
            # 失败也要留档：`fail` 会把错误写进 run 的 tags，而**失败的那次 run
            # 往往是最需要看的那一条**——它的参数要用来复现问题。
            outcome.stages.append(
                StageResult(
                    name=self._next_stage_name(outcome.stages),
                    status=STAGE_FAILED,
                    detail=str(exc),
                )
            )
            self.tracker.fail(run, error=str(exc))
            outcome.skipped_reason = str(exc)
            logger.warning("流水线失败：%s", exc)
        logger.info("流水线完成：%s", outcome.summary_line())
        return outcome

    @staticmethod
    def _next_stage_name(stages: list[StageResult]) -> str:
        """下一个该执行的阶段名（失败落在哪一步）.

        没有它，失败会被记在一个"名字随便取"的阶段上，而
        ``StageResult`` 会因未知阶段名直接抛 ``MLOpsError``——
        那是"处理异常时又抛异常"，最难查的一类。
        """
        done = {item.name for item in stages}
        return next((name for name in PIPELINE_STAGES if name not in done), PIPELINE_STAGES[-1])


__all__ = [
    "EVALUATE_REQUIRED_FIELDS",
    "TRACK_ARTIFACT_ADAPTER",
    "TRACK_ARTIFACT_MANIFEST",
    "TRACK_ARTIFACT_MERGED",
    "TRACK_ARTIFACT_MODEL_CARD",
    "TRAIN_REQUIRED_FIELDS",
    "FinetunePipeline",
    "PipelineOutcome",
]
