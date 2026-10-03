"""重训触发条件：什么时候值得再训一次（M5-D9）.

持续微调最容易做错的一步是"上定时任务，每天训一次"。它有三个具体代价：

1. **算力浪费**：数据一天没变，训出来的适配器哈希与上一版几乎相同，
   而评估、门禁、部署检查一样都不能省；
2. **版本表被噪声灌满**：每天多一个版本，回滚目标的选择空间被稀释，
   而"哪一版是好的"这个问题越来越难回答；
3. **评估方差被当成信号**：评估有 ±0.02 的噪声（day027 的死区），
   天天换版本等于天天在噪声上做决策。

所以本模块把"该不该训"做成一份**可核对的判断**，而不是一个 ``if now - last > 1h``。

## 触发器（trigger）与否决项（veto）是两类东西

这是本模块唯一需要记住的区分：

| 类别 | 语义 | 例子 |
|------|------|------|
| ``trigger`` | **有一个理由支持训练** | 数据涨了 30 条、线上合格率掉到 0.42、数据集重建过 |
| ``veto`` | **有一个理由禁止训练** | 冷却期没过、已有训练任务在跑 |

判定规则：

```text
should_retrain = any(trigger.fired) and not vetoed_by
```

为什么不能把 veto 写成"触发器之一的反向条件"（例如把冷却期做成一个
``cooldown_passed`` 触发器，要求它 fired 才算数）？因为那样报告里就分不清：

```text
fired = []                          → 没有理由训练          （正确：什么都不做）
fired = [] 且 vetoed_by = [cooldown]→ 有理由，但时机不允许  （正确：等一会儿）
```

两者的运维动作完全不同——前者要去看数据/评估，后者只需要等。
把 veto 混进触发器会把这两种情况压成同一个信号。

## 缺失值一律"不触发、不否决"

``online_pass_rate=None``（还没跑线上评估）**不**触发质量下降；
``hours_since_last_train=None``（从未训练）**不**否决冷却期。

这与 day058 的 ``pass_rate`` 返回 ``None`` 是同一条纪律：**缺失与 0 是两件事**。
把缺失当 0 会让"从未训练过"的仓库被冷却期永久拦住（hours=None → 0 < 24 → veto），
而它恰恰是最该训练一次的那种状态。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from smart_research_agent.registry.errors import RegistryError

#: 触发器名字（进报告，逐项可核对）。
TRIGGER_DATA_GROWTH = "data_growth"
TRIGGER_DATASET_CHANGE = "dataset_change"
TRIGGER_QUALITY_DROP = "quality_drop"

#: 否决项名字。
VETO_COOLDOWN = "cooldown"
VETO_ACTIVE_RUN = "active_run"

#: 评估类别标签。
KIND_TRIGGER = "trigger"
KIND_VETO = "veto"


@dataclass
class TriggerState:
    """重训判定需要的全部输入（一个值对象，便于从任意来源构造）.

    刻意**不持有注册表**：状态是一个纯数据快照，因此测试与 API 端点都可以
    手写一个状态来驱动判定，不需要先造一个真实的版本历史。
    ``retrain.build_state`` 负责从注册表里把 ``stable_dataset_fingerprint``
    填好——那是"查一次"的动作，不是判定逻辑的一部分。
    """

    new_examples: int = 0
    dataset_fingerprint: str = ""
    stable_dataset_fingerprint: str = ""
    online_pass_rate: float | None = None
    hours_since_last_train: float | None = None
    active_runs: int = 0

    def __post_init__(self) -> None:
        if self.new_examples < 0:
            raise RegistryError(f"new_examples 不能为负数，收到 {self.new_examples}")
        if self.active_runs < 0:
            raise RegistryError(f"active_runs 不能为负数，收到 {self.active_runs}")
        if self.online_pass_rate is not None and not 0.0 <= self.online_pass_rate <= 1.0:
            raise RegistryError(
                f"online_pass_rate 必须落在 [0, 1] 或为 None，收到 {self.online_pass_rate}"
            )
        if self.hours_since_last_train is not None and self.hours_since_last_train < 0:
            raise RegistryError(
                f"hours_since_last_train 不能为负数（None 表示从未训练），"
                f"收到 {self.hours_since_last_train}"
            )

    def dataset_changed(self) -> bool:
        """当前数据集指纹是否与生产版本所用的不同.

        ``stable_dataset_fingerprint`` 为空（还没有生产版本）返回 ``False``：
        "没有基线"不是"变了"。这个区分让首次上线不会被误判成数据漂移。
        """
        if not self.stable_dataset_fingerprint:
            return False
        return self.dataset_fingerprint != self.stable_dataset_fingerprint

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        payload = asdict(self)
        payload["dataset_changed"] = self.dataset_changed()
        return payload


@dataclass(frozen=True)
class TriggerPolicy:
    """触发策略：五个数字就是"什么时候该训"的操作化定义."""

    min_new_examples: int = 8
    min_pass_rate: float = 0.5
    cooldown_hours: float = 24.0
    max_parallel_runs: int = 1

    def __post_init__(self) -> None:
        if self.min_new_examples < 1:
            raise RegistryError(
                f"min_new_examples 必须 >= 1，收到 {self.min_new_examples}"
                "（0 会让'数据一条没多'也触发训练）"
            )
        if not 0.0 <= self.min_pass_rate <= 1.0:
            raise RegistryError(
                f"min_pass_rate 必须落在 [0, 1]，收到 {self.min_pass_rate}"
            )
        if self.cooldown_hours < 0:
            raise RegistryError(
                f"cooldown_hours 不能为负数，收到 {self.cooldown_hours}"
            )
        if self.max_parallel_runs < 1:
            raise RegistryError(
                f"max_parallel_runs 必须 >= 1，收到 {self.max_parallel_runs}"
            )

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return asdict(self)


@dataclass(frozen=True)
class TriggerEvaluation:
    """一条触发/否决项的判定结果.

    ``actual`` 与 ``threshold`` 都允许为 ``None``：前者表示"这个量没有观测值"
    （例如从未评估过），后者表示"这一项没有阈值"（例如数据集变化是布尔判据）。
    **允许 None 而不是用哨兵数字，是为了让报告里"没有数据"与"刚好等于 0"
    不会被打印成同一个样子。**
    """

    name: str
    kind: str
    fired: bool
    actual: float | str | None
    threshold: float | str | None
    reason: str

    def __post_init__(self) -> None:
        if self.kind not in (KIND_TRIGGER, KIND_VETO):
            raise RegistryError(
                f"未知类别 {self.kind!r}，可选 {KIND_TRIGGER} / {KIND_VETO}"
            )

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return asdict(self)

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        mark = "命中" if self.fired else "未命中"
        return (
            f"[{self.kind}/{mark}] {self.name}: 实际 {self.actual} / "
            f"阈值 {self.threshold} —— {self.reason}"
        )


@dataclass(frozen=True)
class RetrainDecision:
    """一次"要不要重训"的完整判定."""

    should_retrain: bool
    fired: tuple[str, ...] = ()
    vetoed_by: tuple[str, ...] = ()
    evaluations: tuple[TriggerEvaluation, ...] = ()
    notes: str = ""

    def triggers(self) -> list[TriggerEvaluation]:
        """只看触发器那一类（便于报告分块展示）."""
        return [item for item in self.evaluations if item.kind == KIND_TRIGGER]

    def vetoes(self) -> list[TriggerEvaluation]:
        """只看否决项那一类."""
        return [item for item in self.evaluations if item.kind == KIND_VETO]

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        if self.should_retrain:
            return f"应该重训 | 命中触发器 {', '.join(self.fired)}"
        if self.vetoed_by:
            return (
                f"暂不重训 | 触发器 {', '.join(self.fired) or '（无）'} "
                f"被否决项 {', '.join(self.vetoed_by)} 拦下"
            )
        return "无需重训 | 没有任何触发器命中"

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return {
            "should_retrain": self.should_retrain,
            "fired": list(self.fired),
            "vetoed_by": list(self.vetoed_by),
            "notes": self.notes,
            "evaluations": [item.to_dict() for item in self.evaluations],
        }

    def render_markdown(self) -> str:
        """把判定渲染成 markdown（先触发器、后否决项）."""
        lines = [
            "# 重训触发判定",
            "",
            f"- 结论：**{'应该重训' if self.should_retrain else '暂不重训'}**",
            f"- 命中的触发器：{', '.join(self.fired) or '（无）'}",
            f"- 生效的否决项：{', '.join(self.vetoed_by) or '（无）'}",
            f"- 备注：{self.notes or '（无）'}",
            "",
            "| 类别 | 名称 | 命中 | 实际 | 阈值 | 理由 |",
            "|------|------|------|------|------|------|",
        ]
        lines.extend(
            f"| {item.kind} | `{item.name}` | {'是' if item.fired else '否'} | "
            f"{item.actual} | {item.threshold} | {item.reason} |"
            for item in self.evaluations
        )
        lines.append("")
        return "\n".join(lines)


def evaluate_triggers(
    state: TriggerState, policy: TriggerPolicy | None = None
) -> RetrainDecision:
    """按策略评估全部触发项与否决项，给出重训判定.

    评估顺序固定为**三个触发器 → 两个否决项**，与 ``render_markdown`` 的
    展示顺序一致：报告的顺序与代码的顺序不同，是最容易让"读报告的人"
    与"改代码的人"对不上的一种漂移。
    """
    resolved = policy or TriggerPolicy()
    evaluations: list[TriggerEvaluation] = []

    # ---- 触发器 1：数据增量
    growth_fired = state.new_examples >= resolved.min_new_examples
    evaluations.append(
        TriggerEvaluation(
            name=TRIGGER_DATA_GROWTH,
            kind=KIND_TRIGGER,
            fired=growth_fired,
            actual=state.new_examples,
            threshold=resolved.min_new_examples,
            reason=(
                f"新增可用样本 {state.new_examples} 条"
                f"{'达到' if growth_fired else '未达到'}增量门槛 "
                f"{resolved.min_new_examples}"
            ),
        )
    )

    # ---- 触发器 2：数据集重建（指纹变了）
    changed = state.dataset_changed()
    evaluations.append(
        TriggerEvaluation(
            name=TRIGGER_DATASET_CHANGE,
            kind=KIND_TRIGGER,
            fired=changed,
            actual=(
                "（无基线）"
                if not state.stable_dataset_fingerprint
                else f"{state.dataset_fingerprint} vs {state.stable_dataset_fingerprint}"
            ),
            threshold="指纹不同",
            reason=(
                "数据集指纹与生产版本不同：同一批样本换过清洗口径也算重建"
                if changed
                else "数据集指纹与生产版本一致"
                if state.stable_dataset_fingerprint
                else "尚无生产版本，没有可比较的基线指纹"
            ),
        )
    )

    # ---- 触发器 3：线上质量下降
    if state.online_pass_rate is None:
        quality_fired = False
        quality_reason = "未观测到线上合格率：缺失不触发（缺失与 0 是两件事）"
    else:
        quality_fired = state.online_pass_rate < resolved.min_pass_rate
        quality_reason = (
            f"线上合格率 {state.online_pass_rate:.4f}"
            f"{'低于' if quality_fired else '不低于'}下限 {resolved.min_pass_rate}"
        )
    evaluations.append(
        TriggerEvaluation(
            name=TRIGGER_QUALITY_DROP,
            kind=KIND_TRIGGER,
            fired=quality_fired,
            actual=state.online_pass_rate,
            threshold=resolved.min_pass_rate,
            reason=quality_reason,
        )
    )

    # ---- 否决项 1：冷却期
    if state.hours_since_last_train is None:
        cooldown_fired = False
        cooldown_reason = "从未训练过：冷却期不否决（首次训练不该被冷却期拦住）"
    else:
        cooldown_fired = state.hours_since_last_train < resolved.cooldown_hours
        cooldown_reason = (
            f"距上次训练 {state.hours_since_last_train:.2f}h"
            f"{'不足' if cooldown_fired else '已过'}冷却期 {resolved.cooldown_hours}h"
        )
    evaluations.append(
        TriggerEvaluation(
            name=VETO_COOLDOWN,
            kind=KIND_VETO,
            fired=cooldown_fired,
            actual=state.hours_since_last_train,
            threshold=resolved.cooldown_hours,
            reason=cooldown_reason,
        )
    )

    # ---- 否决项 2：并发训练
    parallel_fired = state.active_runs >= resolved.max_parallel_runs
    evaluations.append(
        TriggerEvaluation(
            name=VETO_ACTIVE_RUN,
            kind=KIND_VETO,
            fired=parallel_fired,
            actual=state.active_runs,
            threshold=resolved.max_parallel_runs,
            reason=(
                f"已有 {state.active_runs} 个训练任务在跑"
                f"{'，达到' if parallel_fired else '，未达到'}并发上限 "
                f"{resolved.max_parallel_runs}"
            ),
        )
    )

    fired = tuple(item.name for item in evaluations if item.kind == KIND_TRIGGER and item.fired)
    vetoed = tuple(item.name for item in evaluations if item.kind == KIND_VETO and item.fired)
    notes = ""
    if fired and vetoed:
        notes = "有触发理由但被否决项拦下：这是'再等等'而不是'不需要'"
    elif not fired:
        notes = "没有任何触发理由：去看数据与线上评估，而不是调冷却期"
    return RetrainDecision(
        should_retrain=bool(fired) and not vetoed,
        fired=fired,
        vetoed_by=vetoed,
        evaluations=tuple(evaluations),
        notes=notes,
    )


def trigger_table(policy: TriggerPolicy | None = None) -> list[dict[str, Any]]:
    """把策略渲染成一张"条件表"（API 的自我描述端点与文档同源）.

    表里的每一行都对应 ``evaluate_triggers`` 里的一段判定，
    因此"文档说 min_new_examples 是 8、代码是 50"这类漂移会立刻可见。
    """
    resolved = policy or TriggerPolicy()
    return [
        {
            "name": TRIGGER_DATA_GROWTH,
            "kind": KIND_TRIGGER,
            "threshold": resolved.min_new_examples,
            "meaning": "自上次训练以来新增的可用样本数达到门槛",
            "reviewed_when": "数据流水线跑完一轮之后",
        },
        {
            "name": TRIGGER_DATASET_CHANGE,
            "kind": KIND_TRIGGER,
            "threshold": "指纹与生产版本不同",
            "meaning": "数据集被重建（换了清洗口径或补了样本）",
            "reviewed_when": "manifest.fingerprint 更新之后",
        },
        {
            "name": TRIGGER_QUALITY_DROP,
            "kind": KIND_TRIGGER,
            "threshold": resolved.min_pass_rate,
            "meaning": "线上观测到的合格率低于下限",
            "reviewed_when": "线上评估窗口结束之后",
        },
        {
            "name": VETO_COOLDOWN,
            "kind": KIND_VETO,
            "threshold": resolved.cooldown_hours,
            "meaning": "距上次训练不足冷却期（从未训练时不否决）",
            "reviewed_when": "每次判定",
        },
        {
            "name": VETO_ACTIVE_RUN,
            "kind": KIND_VETO,
            "threshold": resolved.max_parallel_runs,
            "meaning": "已有训练任务在跑，避免两个任务写同一份产物目录",
            "reviewed_when": "每次判定",
        },
    ]


__all__ = [
    "KIND_TRIGGER",
    "KIND_VETO",
    "TRIGGER_DATASET_CHANGE",
    "TRIGGER_DATA_GROWTH",
    "TRIGGER_QUALITY_DROP",
    "VETO_ACTIVE_RUN",
    "VETO_COOLDOWN",
    "RetrainDecision",
    "TriggerEvaluation",
    "TriggerPolicy",
    "TriggerState",
    "evaluate_triggers",
    "trigger_table",
]
