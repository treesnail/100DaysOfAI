"""持续微调流水线：把"该不该训、训完要不要换、出问题怎么退"串成一条链（M5-D9）.

本模块是 day058 的收口，也是 M5 前半段所有"离线可复现"设施第一次被串起来：

```text
数据集(day048/057) → 训练(day050/051/052) → 评估(day053) → 门禁(今天) → 注册(今天) → 采纳(今天)
       指纹              适配器内容哈希          合格率          采纳策略        版本链      生产指针
```

## 六个步骤的顺序与它为什么是这个顺序

``RetrainPlan.steps`` 固定为：

```text
verify_data → train → evaluate → gate → register → promote
```

四处理由，每一处都是"换位只在某些输入下出错"的那种：

1. **``verify_data`` 在最前**：训练前必须确认数据集指纹确实变了（或增量确实
   够数）。放到后面等于"先花完算力再发现数据没变"——这正是 day058 触发器
   要防的事，但触发器看的是**调用方报上来的状态**，而状态可能与磁盘上的
   数据不一致，所以流水线自己要再核一次；
2. **``evaluate`` 在 ``gate`` 之前**：门禁是**对评估结果的判定**，
   没有结果就没有判据。反过来写只能得到一个"用空指标通过的门禁"；
3. **``gate`` 在 ``register`` 之前**：没通过门禁的候选**不该进注册表**。
   先进表再判断，会让版本表里堆满"登记了但永远不该上线"的记录，
   而 ``lineage`` 与 ``stable_ancestors`` 都会开始被它们污染
   （``stable_ancestors`` 只看 stable，但 ``lineage`` 不看阶段）；
4. **``promote`` 在 ``register`` 之后**：采纳是把一个**已经存在的**记录
   升为 stable。反过来（先 promote 再 register）会出现"生产指针指向一个
   注册表里还没有的版本"，而回滚逻辑依赖注册表来找祖先。

## 先登记为候选、再按判定提升

这与"训练脚本自己把结果标成 stable"的做法只差一行，但差别很大：

- 候选阶段让 ``head()`` 保持不变，**上线是一个显式的、有记录的动作**；
- 采纳判定拿到的 ``current`` 与 ``candidate`` 都是注册表里的正式记录，
  没有"半登记状态"；
- 门禁不通过时注册表里留下的是 ``candidate``，**回滚链上一个都没多**。

## 计划（plan）与实际版本号

``RetrainPlan.version`` 是**预留**的版本号（由 ``registry.propose_version``
按"变化的那一项"算出递增位）。训练完成之后拿到适配器哈希，如果这个三元组
已经在注册表里（同一份数据、同一个适配器——重跑一次 CI），流水线会
**复用已有版本号**并把这件事记进时间线，而不是新占一个号。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.registry.errors import RegistryError
from smart_research_agent.registry.record import (
    ARTIFACT_ADAPTER,
    ARTIFACT_DATASET,
    ARTIFACT_MERGED,
    STAGE_STABLE,
    ModelVersion,
)
from smart_research_agent.registry.rollback import (
    DEFAULT_OBSERVE_WINDOW_HOURS,
    PromotionDecision,
    PromotionPolicy,
    evaluate_candidate,
)
from smart_research_agent.registry.store import ModelRegistry
from smart_research_agent.registry.triggers import (
    TRIGGER_QUALITY_DROP,
    RetrainDecision,
    TriggerPolicy,
    TriggerState,
    evaluate_triggers,
)
from smart_research_agent.registry.version import VersionTriple
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: 流水线的六个步骤（顺序即策略，见模块 docstring）.
PLAN_STEPS: tuple[str, ...] = (
    "verify_data",
    "train",
    "evaluate",
    "gate",
    "register",
    "promote",
)

#: 训练回调必须返回的字段（缺了它就没法算版本键）.
TRAIN_RESULT_REQUIRED: tuple[str, ...] = ("adapter_sha256",)

#: 允许训练回调返回的产物槽位（与 ``record.ARTIFACT_NAMES`` 同集合，这里显式
#: 列出是为了在**回调契约层面**给出错误信息，而不是等到构造记录时才发现）。
TRAIN_RESULT_ARTIFACTS: tuple[str, ...] = (ARTIFACT_ADAPTER, ARTIFACT_MERGED, ARTIFACT_DATASET)

#: 训练回调契约：输入计划，输出 ``{adapter_sha256, step?, train_loss?, artifacts?}``。
Trainer = Callable[["RetrainPlan"], dict[str, Any]]

#: 评估回调契约：输入产物字典，输出 ``{指标名: 数值}``（至少含 eval_pass_rate）。
Evaluator = Callable[[dict[str, str]], dict[str, float]]


@dataclass(frozen=True)
class RetrainPlan:
    """一次重训的完整计划（可打印、可归档、可作为训练脚本的输入）."""

    version: str
    bump_kind: str
    base_model: str
    dataset_fingerprint: str
    parent_version: str
    decision: RetrainDecision
    steps: tuple[str, ...] = PLAN_STEPS

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return {
            "version": self.version,
            "bump_kind": self.bump_kind,
            "base_model": self.base_model,
            "dataset_fingerprint": self.dataset_fingerprint,
            "parent_version": self.parent_version,
            "steps": list(self.steps),
            "fired": list(self.decision.fired),
            "notes": self.decision.notes,
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        parent = self.parent_version or "（首版）"
        return (
            f"计划 v{self.version}（{self.bump_kind}，父 {parent}）| "
            f"基座 {self.base_model} | 数据 {self.dataset_fingerprint} | "
            f"{len(self.steps)} 步 | 触发 {', '.join(self.decision.fired)}"
        )

    def render_markdown(self) -> str:
        """把计划渲染成 markdown（含既定步骤顺序与触发证据）."""
        lines = [
            f"# 重训计划：v{self.version}（{self.bump_kind}）",
            "",
            f"- 基座：{self.base_model}",
            f"- 数据集指纹：`{self.dataset_fingerprint}`",
            f"- 父版本：{self.parent_version or '（首版）'}",
            f"- 命中的触发器：{', '.join(self.decision.fired) or '（无）'}",
            f"- 备注：{self.decision.notes or '（无）'}",
            "",
            "| 序 | 步骤 | 说明 |",
            "|----|------|------|",
        ]
        for index, name in enumerate(self.steps, start=1):
            lines.append(f"| {index} | `{name}` | {STEP_MEANINGS.get(name, '')} |")
        lines.extend(["", self.decision.render_markdown()])
        return "\n".join(lines)


#: 每个步骤的一句话说明（进计划报告；与 ``PLAN_STEPS`` 同源，不会各自漂移）.
STEP_MEANINGS: dict[str, str] = {
    "verify_data": "确认数据集指纹确实变化（或增量够数）——先核一次，别花完算力才发现没变",
    "train": "调用训练回调，产出适配器（内容哈希是版本键的一部分）",
    "evaluate": "在领域评估集上跑一次评估，产出合格率等指标",
    "gate": "按采纳策略判定，未通过则停在候选阶段（不进版本表）",
    "register": "把候选登记进注册表（含三元组、产物路径与指标）",
    "promote": "采纳判定为 promote 时提升为 stable，head() 随之指向新版本",
}


@dataclass
class RetrainOutcome:
    """一次持续微调运行的完整产物（含时间线，便于复盘"哪一步决定了什么"）."""

    decision: RetrainDecision
    plan: RetrainPlan | None = None
    version: ModelVersion | None = None
    promotion: PromotionDecision | None = None
    trained: bool = False
    skipped_reason: str = ""
    timeline: list[dict[str, Any]] = field(default_factory=list)

    @property
    def promoted(self) -> bool:
        """运行结束时，本次产出的版本是否**正处于 stable 阶段**.

        定义刻意基于 ``version.stage`` 而不是 ``promotion.promoted``：
        复用已有版本时（三元组相同、CI 重跑），这条流水线根本没有做采纳判定
        ——而"它已经是生产版本"这个事实依然成立。用阶段定义，两种路径
        给出同一个答案；用判定定义，重跑会得出"未采纳"这个**误导性**的结论
        （实测复跑时 gain=0.0000 < 0.02，判定必然是 hold）。
        """
        return self.version is not None and self.version.stage == STAGE_STABLE

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return {
            "decision": self.decision.to_dict(),
            "plan": None if self.plan is None else self.plan.to_dict(),
            "version": None if self.version is None else self.version.to_dict(),
            "promotion": None if self.promotion is None else self.promotion.to_dict(),
            "trained": self.trained,
            "promoted": self.promoted,
            "skipped_reason": self.skipped_reason,
            "timeline": [dict(item) for item in self.timeline],
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        if self.plan is None:
            return f"未启动训练 | {self.skipped_reason or self.decision.summary_line()}"
        if self.version is None:
            # dry_run 的真实路径：有计划、没登记。它不是"防御式分支"——
            # 这条路径每次"先算计划再决定要不要花算力"都会走到。
            return (
                f"{self.decision.summary_line()} | 计划 v{self.plan.version} | "
                f"已训练 {self.trained} | 未登记"
            )
        on_production = self.version.version if self.promoted else "（未采纳）"
        return (
            f"{self.decision.summary_line()} | 计划 v{self.plan.version} | "
            f"已训练 {self.trained} | 生产版本 {on_production}"
        )


def build_state(
    registry: ModelRegistry,
    *,
    new_examples: int = 0,
    dataset_fingerprint: str = "",
    online_pass_rate: float | None = None,
    hours_since_last_train: float | None = None,
    active_runs: int = 0,
) -> TriggerState:
    """从注册表补出 ``stable_dataset_fingerprint``，构造一个完整状态.

    "查一次注册表"这个动作刻意放在流水线之外：触发器模块只管判定，
    不持有注册表（见 ``triggers`` 的 docstring）。副作用是测试与接口
    可以手写状态，不需要先造版本历史。
    """
    head = registry.head()
    return TriggerState(
        new_examples=new_examples,
        dataset_fingerprint=dataset_fingerprint,
        stable_dataset_fingerprint="" if head is None else head.dataset_fingerprint,
        online_pass_rate=online_pass_rate,
        hours_since_last_train=hours_since_last_train,
        active_runs=active_runs,
    )


def verify_data(state: TriggerState, decision: RetrainDecision) -> str:
    """训练前再核一次数据集状态（计划里的第一步），返回一行说明.

    两条判据：

    1. **数据指纹不能为空**（硬约束）。三元组需要它来回答"这个效果是哪份
       数据训出来的"，缺了它登记下来的记录无法追溯——而事后补是补不上的；
    2. **"数据没变却要训练"必须有一个说得出口的理由**。质量下降触发的
       重训是合法的（同数据重训，靠换超参或对齐手段改进），此时理由被
       写进时间线；而"既没有新增样本、指纹也没变、品质也没掉"却还在训练，
       多半是调度器配置写错了——那正是本课触发器要防的算力浪费。

    只做**自洽性**校验，不读磁盘：要检查的是"调用方报上来的状态有没有
    内部矛盾"。读盘校验留给集成脚本，流水线刻意不持有数据目录。

    它是一个模块级函数而不是流水线的私有方法，原因很实际：**它是唯一
    一段"在触发器已判定为应该训练之后仍然会否决训练"的逻辑**，必须能被
    单独驱动（否则那两条 ``raise`` 分支在端到端路径上永远走不到——
    触发器早已把这类状态挡在外面了）。
    """
    if not state.dataset_fingerprint.strip():
        raise RegistryError(
            "dataset_fingerprint 为空：版本三元组需要它来回答"
            "'这个效果是哪份数据训出来的'，缺了它登记的记录无法追溯"
        )
    if state.new_examples == 0 and not state.dataset_changed():
        if TRIGGER_QUALITY_DROP in decision.fired:
            return "数据集与生产版本一致：本次属于同数据重训（线上质量下降触发）"
        raise RegistryError(
            "既没有新增样本，数据集指纹也没有变化："
            "这次训练产出的适配器哈希极可能与上一版相同，"
            "请先用 dry_run 检查触发判定"
        )
    return (
        f"数据集指纹已核对：新增样本 {state.new_examples} 条，"
        f"指纹{'已变化' if state.dataset_changed() else '未设基线'}"
    )


def pipeline_dataset_fingerprint() -> str:
    """用 day057 的领域数据流水线算当前数据集指纹（把两条链路接起来）.

    这是"数据集指纹 → 模型版本三元组"那座桥的实体：模型版本里的
    ``dataset_fingerprint`` 必须与 ``DomainDataManifest.fingerprint`` 是**同一个值**，
    否则"这个效果是哪份数据训出来的"这句话在跨天之后就不成立了。
    延迟导入：``domain_data`` 只在这一步被需要，而它的导入代价不小。
    """
    from smart_research_agent.domain_data.pipeline import default_pipeline

    return default_pipeline().run_from_sources().manifest.fingerprint


class ContinualFinetunePipeline:
    """持续微调流水线（触发 → 计划 → 训练 → 评估 → 门禁 → 登记 → 采纳）.

    用法::

        registry = ModelRegistry("outputs/registry/versions.jsonl")
        pipeline = ContinualFinetunePipeline(
            registry, base_model="Qwen3-8B", trainer=train_fn, evaluator=eval_fn
        )
        outcome = pipeline.run(pipeline.build_state(new_examples=30))
        if outcome.promoted:
            print("已上线", outcome.version.version)
    """

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        base_model: str,
        trigger_policy: TriggerPolicy | None = None,
        promotion_policy: PromotionPolicy | None = None,
        trainer: Trainer | None = None,
        evaluator: Evaluator | None = None,
        observe_window_hours: float = DEFAULT_OBSERVE_WINDOW_HOURS,
    ):
        if not base_model.strip():
            raise RegistryError("base_model 不能为空")
        self.registry = registry
        self.base_model = base_model.strip()
        self.trigger_policy = trigger_policy or TriggerPolicy()
        self.promotion_policy = promotion_policy or PromotionPolicy()
        self.trainer = trainer
        self.evaluator = evaluator
        self.observe_window_hours = observe_window_hours

    # ------------------------------------------------------------------ 计划
    def build_state(self, **kwargs: Any) -> TriggerState:
        """便捷方法：转调模块级 ``build_state``（持有注册表，因此不用重复传）."""
        return build_state(self.registry, **kwargs)

    def plan(self, state: TriggerState, decision: RetrainDecision) -> RetrainPlan:
        """生成重训计划（预留版本号，父版本取当前 head）.

        版本号**预留**而不是训练后再取：训练脚本需要知道"我这次要产出 v几"，
        才能把版本号写进检查点与日志；训练完再取会让"这次训练的产物叫什么"
        在训练期间处于未知状态。
        """
        head = self.registry.head()
        version, kind = self.registry.propose_version(
            base_model=self.base_model, dataset_fingerprint=state.dataset_fingerprint
        )
        return RetrainPlan(
            version=version,
            bump_kind=kind,
            base_model=self.base_model,
            dataset_fingerprint=state.dataset_fingerprint,
            parent_version="" if head is None else head.version,
            decision=decision,
        )

    # ------------------------------------------------------------------ 运行
    def run(self, state: TriggerState, *, dry_run: bool = False) -> RetrainOutcome:
        """跑一次持续微调（``dry_run=True`` 时只到"计划"为止，不调训练回调）."""
        decision = evaluate_triggers(state, self.trigger_policy)
        outcome = RetrainOutcome(decision=decision)
        outcome.timeline.append(
            {"step": "decide", "detail": decision.summary_line(), "ok": decision.should_retrain}
        )
        if not decision.should_retrain:
            outcome.skipped_reason = decision.summary_line()
            logger.info("跳过本次重训：%s", outcome.skipped_reason)
            return outcome

        outcome.plan = self.plan(state, decision)
        outcome.timeline.append(
            {"step": "plan", "detail": outcome.plan.summary_line(), "ok": True}
        )
        if dry_run:
            outcome.skipped_reason = "dry_run：只产出计划，未调用训练回调"
            return outcome

        verified = verify_data(state, decision)
        outcome.timeline.append({"step": "verify_data", "detail": verified, "ok": True})

        if self.trainer is None:
            raise RegistryError("未注入 trainer：非 dry_run 运行必须提供训练回调")
        if self.evaluator is None:
            raise RegistryError("未注入 evaluator：非 dry_run 运行必须提供评估回调")

        train_result = dict(self.trainer(outcome.plan))
        missing = [name for name in TRAIN_RESULT_REQUIRED if not train_result.get(name)]
        if missing:
            raise RegistryError(
                f"训练回调缺少必需字段 {', '.join(missing)}："
                "没有适配器内容哈希就算不出版本键，而版本键是这条记录的身份证"
            )
        artifacts = {
            str(k): str(v)
            for k, v in dict(train_result.get("artifacts", {})).items()
            if k in TRAIN_RESULT_ARTIFACTS
        }
        outcome.trained = True
        outcome.timeline.append(
            {
                "step": "train",
                "detail": (
                    f"适配器 sha256:{str(train_result['adapter_sha256'])[:12]} | "
                    f"step {train_result.get('step', '?')} | "
                    f"loss {train_result.get('train_loss', '?')}"
                ),
                "ok": True,
            }
        )

        metrics = {str(k): float(v) for k, v in dict(self.evaluator(artifacts)).items()}
        outcome.timeline.append(
            {"step": "evaluate", "detail": f"指标 {metrics}", "ok": True}
        )

        triple = VersionTriple(
            base_model=self.base_model,
            adapter_sha256=str(train_result["adapter_sha256"]),
            dataset_fingerprint=state.dataset_fingerprint,
        )
        existing = self.registry.find(triple.key)
        if existing is not None:
            # 同一个三元组重跑（CI 重试）：复用已有版本号，不新占一个
            outcome.version = existing
            outcome.timeline.append(
                {
                    "step": "register",
                    "detail": (
                        f"三元组键 {triple.key} 已存在（v{existing.version}），"
                        "复用已有记录而不是新占版本号"
                    ),
                    "ok": True,
                }
            )
            if existing.stage == STAGE_STABLE:
                # 复用 + 已上线：不再做采纳判定。此时两者的评估分数必然相同
                # （同一条记录），gain 恒为 0，任何 min_gain 都会判 hold——
                # 那个 hold 会把"它正在服务"这个事实说成"未采纳"。
                outcome.timeline.append(
                    {
                        "step": "gate",
                        "detail": "复用已上线版本：无需采纳判定（同一条记录，gain 恒为 0）",
                        "ok": True,
                    }
                )
                outcome.timeline.append(
                    {
                        "step": "promote",
                        "detail": (
                            f"v{existing.version} 已处于 stable，无需重复提升"
                        ),
                        "ok": True,
                    }
                )
                outcome.skipped_reason = "三元组已上线：复用版本号并跳过采纳判定"
                logger.info("持续微调完成（复用已上线版本）：%s", outcome.summary_line())
                return outcome
        else:
            record = ModelVersion(
                triple=triple,
                version=outcome.plan.version,
                parent_version=outcome.plan.parent_version,
                metrics=metrics,
                artifacts=artifacts,
                tags={
                    "bump_kind": outcome.plan.bump_kind,
                    "triggered_by": ",".join(decision.fired),
                },
                notes=f"由持续微调流水线自动登记（{decision.summary_line()}）",
            )
            outcome.version = self.registry.register(record, reason="流水线自动登记")
            outcome.timeline.append(
                {
                    "step": "register",
                    "detail": outcome.version.summary_line(),
                    "ok": True,
                }
            )

        promotion = evaluate_candidate(
            self.registry.head(), outcome.version, policy=self.promotion_policy
        )
        outcome.promotion = promotion
        outcome.timeline.append(
            {"step": "gate", "detail": promotion.summary_line(), "ok": promotion.promoted}
        )
        if promotion.promoted:
            outcome.version = self.registry.set_stage(
                outcome.version.version,
                STAGE_STABLE,
                reason=f"采纳判定为 promote（{promotion.reason}）",
            )
            outcome.timeline.append(
                {
                    "step": "promote",
                    "detail": (
                        f"v{outcome.version.version} 提升为 stable，"
                        f"观察窗口 {self.observe_window_hours}h"
                    ),
                    "ok": True,
                }
            )
        else:
            outcome.timeline.append(
                {
                    "step": "promote",
                    "detail": f"未提升阶段：{promotion.reason}",
                    "ok": False,
                }
            )
        logger.info("持续微调完成：%s", outcome.summary_line())
        return outcome

    # ------------------------------------------------------------------ 内部
    def policy_snapshot(self) -> dict[str, Any]:
        """影响结果的全部参数快照（落进报告，让"同一判定可复现"成立）."""
        return {
            "base_model": self.base_model,
            "trigger_policy": self.trigger_policy.to_dict(),
            "promotion_policy": self.promotion_policy.to_dict(),
            "observe_window_hours": self.observe_window_hours,
            "steps": list(PLAN_STEPS),
        }


__all__ = [
    "PLAN_STEPS",
    "STEP_MEANINGS",
    "TRAIN_RESULT_ARTIFACTS",
    "TRAIN_RESULT_REQUIRED",
    "ContinualFinetunePipeline",
    "Evaluator",
    "RetrainOutcome",
    "RetrainPlan",
    "Trainer",
    "build_state",
    "pipeline_dataset_fingerprint",
    "verify_data",
]
