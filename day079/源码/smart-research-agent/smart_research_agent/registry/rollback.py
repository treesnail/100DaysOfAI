"""采纳与回滚：把"要不要换版本"做成一份可以核对的决策（M5-D9）.

持续微调带来的第一个新问题不是"怎么训"，而是**"训完了要不要换"**。
本模块把这个判断拆成三件互不混淆的事：

```text
拒绝候选（hold）    候选不如当前 → 什么都不发生，当前版本继续服务
采纳候选（promote） 候选明显更好 → 候选升为 stable，成为新的 head
回滚（rollback）    当前版本出问题 → 退回到最近一个 stable 祖先
```

**"拒绝候选"与"回滚"必须分开**，这是本模块最重要的一条纪律。它们是两个
不同的动作、不同的触发者、不同的风险：

| | 拒绝候选 | 回滚 |
|---|---|---|
| 触发者 | 离线评估（还没上线） | 线上事故（已经上线） |
| 对象 | 新候选 | 当前生产版本 |
| 后果 | 什么都不发生 | 生产版本换人，有服务窗口 |
| 能否"改回去" | 不需要，本来就没换 | 不能——回滚本身要被审计 |

把两者混成一个 ``if score < threshold: revert()`` 会把"候选没通过评估"变成
"把线上版本也退掉"，而后者的代价远大于前者。

## 两个数字：``min_gain`` 与 ``regression_tolerance``

- ``min_gain``（缺省 0.02）是**采纳门槛**，也是"死区"：模型评估有方差，
  ±0.02 的波动是噪声不是提升（day027 的趋势判定用的是同一个数）。
  没有死区，自动流水线会天天在不同版本之间来回切；
- ``regression_tolerance``（缺省 0.0）是**退步容忍度**：候选可以比当前低
  多少还仍然"值得考虑"。缺省 0 意味着"任何退步都拒绝"。

两个数字都必须在报告里被打印出来——**一个没有写出阈值的判定，无法被复核**。

## 不可比时不能算差值

``version.comparable`` 定义了"两个版本的效果可否直接比较"（基座相同 + 数据集
指纹相同）。不可比时本模块**拒绝计算 gain**，改用一个绝对值门槛
（``absolute_min_pass_rate``）。理由在 day057 已经写过一次：**同一份数据在
两套口径下得到两个结论，是数据治理最忌讳的事**。这里同理——把"换了数据的
候选"与"当前版本"的分数相减，得到的数字既不是提升也不是退步，而是两种
评估分布之间的差，而它会被下游当成前者使用。

## 回滚的顺序：第一步必须是"确认有退路"

``plan_rollback`` 生成的动作序列刻意把 ``verify``（校验目标版本可部署）
放在 ``freeze``（把出问题的版本置为 rolled_back）**之前**。反过来写也能
通过绝大多数测试，代价只在最坏情况下暴露：

> 先冻结当前版本、再发现目标版本不可部署，结果是**生产上没有任何可服务版本**。

这与 day057 六阶段里"mixing 必须在 augment 之前"是同一种推理方式：
每一步的位置都有理由，换位只在某些输入下才出错——**而那种错误最难在测试里
被发现**。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from smart_research_agent.registry.errors import RegistryError
from smart_research_agent.registry.record import (
    STAGE_ROLLED_BACK,
    STAGE_STABLE,
    ModelVersion,
)
from smart_research_agent.registry.store import ModelRegistry
from smart_research_agent.registry.version import comparable
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: 三个动作名。``hold`` 是"什么都不做"，**它是一个正式结论而不是失败**：
#: 报告里 "hold / 增益不足" 与 "hold / 缺少评估分数" 是两条可区分的信息。
ACTION_PROMOTE = "promote"
ACTION_ROLLBACK = "rollback"
ACTION_HOLD = "hold"
ACTIONS: tuple[str, ...] = (ACTION_PROMOTE, ACTION_ROLLBACK, ACTION_HOLD)

#: 检查项名字（进报告，逐项可核对）。
CHECK_COMPARABLE = "comparable"
CHECK_DEPLOYABLE = "deployable"
CHECK_METRICS = "metrics"
CHECK_AGE = "age"
CHECK_GAIN = "gain"
CHECK_ABSOLUTE = "absolute_pass_rate"

#: 回滚动作序列里每一步的动作名。
STEP_VERIFY = "verify"
STEP_FREEZE = "freeze"
STEP_RECORD = "record"
STEP_OBSERVE = "observe"

#: 观察窗口时长（小时）。回滚之后**不立刻宣布成功**：切换动作是瞬时的，
#: 而"事故是否停止"需要一段窗口的线上数据才能确认。这个字段进计划，
#: 是为了让"回滚完成"这个说法有一个明确的时间点。
DEFAULT_OBSERVE_WINDOW_HOURS = 24.0


@dataclass(frozen=True)
class PromotionPolicy:
    """采纳策略：几个阈值就是"什么算更好"的操作化定义.

    ``require_comparable=True``（缺省）表示**不可比的候选拒绝算增益**、
    改用绝对值门槛；置 False 会让不可比的候选也落进增益分支——那是本模块
    要防的用法，保留这个开关只是为了让"不可比时算出的差值有多离谱"
    可以被实测出来（教程里给了一组实测数字）。
    """

    min_gain: float = 0.02
    regression_tolerance: float = 0.0
    absolute_min_pass_rate: float = 0.5
    max_candidate_age_hours: float | None = 168.0
    require_comparable: bool = True
    require_deployable: bool = True

    def __post_init__(self) -> None:
        if self.min_gain < 0:
            raise RegistryError(f"min_gain 不能为负数，收到 {self.min_gain}")
        if self.regression_tolerance < 0:
            raise RegistryError(
                f"regression_tolerance 不能为负数，收到 {self.regression_tolerance}"
            )
        if not 0.0 <= self.absolute_min_pass_rate <= 1.0:
            raise RegistryError(
                "absolute_min_pass_rate 必须落在 [0, 1]，"
                f"收到 {self.absolute_min_pass_rate}"
            )
        if self.max_candidate_age_hours is not None and self.max_candidate_age_hours <= 0:
            raise RegistryError(
                f"max_candidate_age_hours 必须为正数或 None，收到 {self.max_candidate_age_hours}"
            )

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return asdict(self)


@dataclass(frozen=True)
class CheckResult:
    """一条检查的结果：实际值、阈值、结论、理由.

    四个字段缺一不可。只有 ``passed`` 的报告在运维上是不可用的——
    看到"没通过"之后唯一的动作是去读代码；而看到"实际 0.03 / 阈值 0.02，
    通过"就能直接判断阈值定得对不对。
    """

    name: str
    passed: bool
    actual: float | str | None
    threshold: float | str | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return asdict(self)

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        return (
            f"[{'通过' if self.passed else '不通过'}] {self.name}: "
            f"实际 {self.actual} / 阈值 {self.threshold} —— {self.reason}"
        )


@dataclass(frozen=True)
class PromotionDecision:
    """一次"采纳还是保留"的完整判定."""

    action: str
    reason: str
    current_version: str
    candidate_version: str
    target_version: str
    gain: float | None = None
    comparable: bool = True
    checks: tuple[CheckResult, ...] = ()

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise RegistryError(f"未知动作 {self.action!r}，可选 {', '.join(ACTIONS)}")

    @property
    def promoted(self) -> bool:
        """是否采纳了候选（``action == "promote"`` 的便捷读取）."""
        return self.action == ACTION_PROMOTE

    def failed_checks(self) -> list[CheckResult]:
        """未通过的检查项（用于定位"为什么没换版本"）."""
        return [item for item in self.checks if not item.passed]

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return {
            "action": self.action,
            "reason": self.reason,
            "current_version": self.current_version,
            "candidate_version": self.candidate_version,
            "target_version": self.target_version,
            "gain": self.gain,
            "comparable": self.comparable,
            "promoted": self.promoted,
            "checks": [item.to_dict() for item in self.checks],
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        gain = "不可比" if self.gain is None else f"{self.gain:+.4f}"
        return (
            f"{self.action} | 当前 {self.current_version or '（无）'} → 候选 "
            f"{self.candidate_version} | gain {gain} | 目标 "
            f"{self.target_version or '（不变）'} | {self.reason}"
        )

    def render_markdown(self) -> str:
        """把判定渲染成 markdown（表 + 逐项检查）."""
        lines = [
            f"# 采纳判定：{self.action}",
            "",
            f"- 当前版本：{self.current_version or '（尚无 stable）'}",
            f"- 候选版本：{self.candidate_version}",
            f"- 目标版本：{self.target_version or '（不变）'}",
            f"- 增益：{'不可比' if self.gain is None else f'{self.gain:+.4f}'}",
            f"- 结论：{self.reason}",
            "",
            "| 检查 | 实际 | 阈值 | 通过 | 理由 |",
            "|------|------|------|------|------|",
        ]
        lines.extend(
            f"| `{item.name}` | {item.actual} | {item.threshold} | "
            f"{'是' if item.passed else '否'} | {item.reason} |"
            for item in self.checks
        )
        lines.append("")
        return "\n".join(lines)


def _hold(
    *,
    reason: str,
    current: ModelVersion | None,
    candidate: ModelVersion,
    checks: tuple[CheckResult, ...],
    gain: float | None,
    is_comparable: bool,
) -> PromotionDecision:
    """构造一个 ``hold`` 判定（当前版本保持不变）."""
    return PromotionDecision(
        action=ACTION_HOLD,
        reason=reason,
        current_version="" if current is None else current.version,
        candidate_version=candidate.version,
        target_version="" if current is None else current.version,
        gain=gain,
        comparable=is_comparable,
        checks=checks,
    )


def candidate_age_hours(candidate: ModelVersion, *, now: datetime | None = None) -> float | None:
    """候选的年龄（小时）；创建时间缺失或不可解析时返回 ``None``.

    返回 ``None`` 而不是 ``0.0`` 或 ``inf``：这个函数的结果会被写进
    时效检查的 ``actual`` 字段，"时间戳解析失败"与"刚好 0 小时"必须能区分开。
    """
    if not candidate.created_at:
        return None
    raw = candidate.created_at.replace("Z", "+00:00")
    try:
        created = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    reference = now if now is not None else datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    return (reference - created).total_seconds() / 3600.0


def evaluate_candidate(
    current: ModelVersion | None,
    candidate: ModelVersion,
    *,
    policy: PromotionPolicy | None = None,
    now: datetime | None = None,
) -> PromotionDecision:
    """判定"要不要把候选提升为生产版本".

    ``current=None``（还没有任何 stable 版本）走**首次上线**分支：
    此时没有可比较的对象，只能看绝对值门槛与可部署性。这个分支必须单独处理，
    否则第一版永远上不了线（gain 无从计算，任何与 ``min_gain`` 的比较都是 undefined）。
    """
    resolved = policy or PromotionPolicy()
    is_comparable = current is not None and comparable(current.triple, candidate.triple)
    checks: list[CheckResult] = []

    # ---- 检查 1：可部署性（证据完整性）
    missing = candidate.missing_artifacts()
    deployable_ok = not missing or not resolved.require_deployable
    checks.append(
        CheckResult(
            name=CHECK_DEPLOYABLE,
            passed=deployable_ok,
            actual=", ".join(missing) if missing else "齐全",
            threshold="缺失为空" if resolved.require_deployable else "不检查",
            reason=(
                "必需产物齐全"
                if deployable_ok and not missing
                else f"缺少必需产物 {', '.join(missing)}"
                if missing
                else "策略未要求检查可部署性"
            ),
        )
    )

    # ---- 检查 2：时效（避免把一个放了三周的候选突然推上线）
    age = candidate_age_hours(candidate, now=now)
    if resolved.max_candidate_age_hours is None:
        age_ok, age_reason = True, "策略未设时效上限"
    elif age is None:
        age_ok, age_reason = True, "创建时间不可解析，跳过时效检查"
    else:
        age_ok = age <= resolved.max_candidate_age_hours
        age_reason = (
            f"候选年龄 {age:.2f}h"
            f"{'在' if age_ok else '超过'} {resolved.max_candidate_age_hours}h 上限"
        )
    checks.append(
        CheckResult(
            name=CHECK_AGE,
            passed=age_ok,
            actual=None if age is None else round(age, 4),
            threshold=resolved.max_candidate_age_hours,
            reason=age_reason,
        )
    )

    # ---- 检查 3：可比性（**信息项，不阻塞**）
    #
    # 它刻意不参与 "不通过就 hold" 的判定：不可比不是候选的错，
    # 换数据集本来就是持续微调的常态。它决定了**走哪条判定分支**，
    # 而"走了一半发现不可比就拒绝"会把每一次数据更新都变成一次上不了线。
    checks.append(
        CheckResult(
            name=CHECK_COMPARABLE,
            passed=True,
            actual=is_comparable,
            threshold="决定判定分支",
            reason=(
                "与当前版本可比（同基座、同数据指纹）：按增益判定"
                if is_comparable
                else "与当前版本不可比（基座或数据集指纹不同）：按绝对门槛判定"
                if current is not None
                else "尚无 stable 版本，走首次上线分支"
            ),
        )
    )

    candidate_rate = candidate.pass_rate
    current_rate = None if current is None else current.pass_rate

    # ---- 分支 A：首次上线
    if current is None:
        absolute_ok = candidate_rate is not None and candidate_rate >= (
            resolved.absolute_min_pass_rate
        )
        checks.append(
            CheckResult(
                name=CHECK_ABSOLUTE,
                passed=absolute_ok,
                actual=candidate_rate,
                threshold=resolved.absolute_min_pass_rate,
                reason=(
                    "还没有 stable 版本：只做绝对值门槛判定"
                    + ("" if candidate_rate is not None else "（候选缺少评估分数）")
                ),
            )
        )
        blocking = [item for item in checks if not item.passed]
        if blocking:
            return _hold(
                reason=f"首次上线被拦下：{blocking[0].reason}",
                current=current,
                candidate=candidate,
                checks=tuple(checks),
                gain=None,
                is_comparable=False,
            )
        return PromotionDecision(
            action=ACTION_PROMOTE,
            reason=(
                f"首次上线：候选合格率 {candidate_rate:.4f} 达到绝对门槛 "
                f"{resolved.absolute_min_pass_rate}"
            ),
            current_version="",
            candidate_version=candidate.version,
            target_version=candidate.version,
            gain=None,
            comparable=False,
            checks=tuple(checks),
        )

    # ---- 分支 B：有当前版本
    #
    # ``require_comparable`` 为真（缺省）时，不可比的候选**拒绝算增益**，
    # 改用绝对值门槛。置为 False 会让代码落进增益分支——那是本模块要防的
    # 用法，保留这个开关只是为了"不可比时算出来的差值有多离谱"可以被实测。
    use_gain_branch = is_comparable or not resolved.require_comparable
    if not use_gain_branch:
        absolute_ok = candidate_rate is not None and candidate_rate >= (
            resolved.absolute_min_pass_rate
        )
        checks.append(
            CheckResult(
                name=CHECK_ABSOLUTE,
                passed=absolute_ok,
                actual=candidate_rate,
                threshold=resolved.absolute_min_pass_rate,
                reason=(
                    "不可比：不算差值，改用绝对值门槛"
                    + ("" if candidate_rate is not None else "（候选缺少评估分数）")
                ),
            )
        )
        blocking = [item for item in checks if not item.passed]
        if blocking:
            return _hold(
                reason=(
                    "不可比且未达绝对门槛：候选 "
                    f"{'无分数' if candidate_rate is None else f'{candidate_rate:.4f}'}"
                    f" < {resolved.absolute_min_pass_rate}"
                ),
                current=current,
                candidate=candidate,
                checks=tuple(checks),
                gain=None,
                is_comparable=is_comparable,
            )
        return PromotionDecision(
            action=ACTION_PROMOTE,
            reason=(
                f"不可比（换了数据集或基座），按绝对门槛判定："
                f"{candidate_rate:.4f} >= {resolved.absolute_min_pass_rate}"
            ),
            current_version=current.version,
            candidate_version=candidate.version,
            target_version=candidate.version,
            gain=None,
            comparable=False,
            checks=tuple(checks),
        )

    # ---- 增益分支（可比，或调用方显式关掉了可比性要求）
    if candidate_rate is None or current_rate is None:
        checks.append(
            CheckResult(
                name=CHECK_METRICS,
                passed=False,
                actual=f"当前 {'无' if current_rate is None else f'{current_rate:.4f}'}"
                f" / 候选 {'无' if candidate_rate is None else f'{candidate_rate:.4f}'}",
                threshold="两边都有分数",
                reason="缺少评估分数，无法计算增益（缺失与 0 是两件事）",
            )
        )
        return _hold(
            reason="缺少评估分数，无法计算增益",
            current=current,
            candidate=candidate,
            checks=tuple(checks),
            gain=None,
            is_comparable=is_comparable,
        )

    checks.append(
        CheckResult(
            name=CHECK_METRICS,
            passed=True,
            actual=f"{current_rate:.4f} → {candidate_rate:.4f}",
            threshold="两边都有分数",
            reason="两边都有评估分数，可以计算增益",
        )
    )

    gain = candidate_rate - current_rate
    checks.append(
        CheckResult(
            name=CHECK_GAIN,
            passed=gain >= -resolved.regression_tolerance,
            actual=round(gain, 4),
            threshold=(
                f">= {-resolved.regression_tolerance}"
                + ("" if is_comparable else "（警告：两版本不可比，此差值跨了评估分布）")
            ),
            reason=(
                f"增益 {gain:+.4f}"
                f"{'达到' if gain >= resolved.min_gain else '未达到'}采纳门槛 "
                f"{resolved.min_gain}"
                + ("" if is_comparable else "；本次差值不可作为提升证据")
            ),
        )
    )

    blocking = [item for item in checks if not item.passed]
    if blocking:
        return _hold(
            reason=f"被拦下：{blocking[0].reason}",
            current=current,
            candidate=candidate,
            checks=tuple(checks),
            gain=gain,
            is_comparable=is_comparable,
        )
    if gain < resolved.min_gain:
        return _hold(
            reason=(
                f"增益 {gain:+.4f} 落在死区（< {resolved.min_gain}）："
                "评估方差量级内不值得切换生产版本"
            ),
            current=current,
            candidate=candidate,
            checks=tuple(checks),
            gain=gain,
            is_comparable=is_comparable,
        )
    decision = PromotionDecision(
        action=ACTION_PROMOTE,
        reason=(
            f"增益 {gain:+.4f} 达到采纳门槛 {resolved.min_gain}，"
            f"候选 {candidate.version} 提升为生产版本"
            + ("" if is_comparable else "（注意：两版本不可比，本次采纳仅凭显式关闭的可比性要求）")
        ),
        current_version=current.version,
        candidate_version=candidate.version,
        target_version=candidate.version,
        gain=gain,
        comparable=is_comparable,
        checks=tuple(checks),
    )
    logger.info("采纳判定：%s", decision.summary_line())
    return decision


@dataclass(frozen=True)
class RollbackStep:
    """回滚动作序列里的一步.

    ``blocking=True`` 表示"这一步失败必须中止整个回滚"。序列里哪些步是
    阻塞的不是风格问题：``verify`` 与 ``freeze`` 失败时继续往下走，
    结果会是"生产上没有可服务版本"（见模块 docstring）。
    """

    order: int
    action: str
    detail: str
    blocking: bool

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return asdict(self)

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        return (
            f"{self.order}. [{self.action}] {self.detail}"
            f"{'（阻塞）' if self.blocking else '（非阻塞）'}"
        )


@dataclass(frozen=True)
class RollbackPlan:
    """回滚计划：决策 + 有序动作序列.

    计划与执行是分开的（本模块只产出计划）。这不是偷懒，
    而是因为**回滚必须先被看见再被执行**：把计划打印出来给人确认，
    是唯一能在"自动回滚把好版本也退掉"这类事故前踩到刹车的做法。
    """

    action: str
    reason: str
    version: str
    target_version: str = ""
    target_key: str = ""
    steps: tuple[RollbackStep, ...] = ()
    observe_window_hours: float = DEFAULT_OBSERVE_WINDOW_HOURS

    def __post_init__(self) -> None:
        if self.action not in (ACTION_ROLLBACK, ACTION_HOLD):
            raise RegistryError(
                f"回滚计划的动作只能是 {ACTION_ROLLBACK} / {ACTION_HOLD}，收到 {self.action!r}"
            )

    @property
    def should_execute(self) -> bool:
        """是否应当执行（``hold`` 计划没有任何步可执行）."""
        return self.action == ACTION_ROLLBACK and bool(self.steps)

    def blocking_steps(self) -> list[RollbackStep]:
        """阻塞步列表（执行器遇到它们失败时必须中止）."""
        return [step for step in self.steps if step.blocking]

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return {
            "action": self.action,
            "reason": self.reason,
            "version": self.version,
            "target_version": self.target_version,
            "target_key": self.target_key,
            "should_execute": self.should_execute,
            "observe_window_hours": self.observe_window_hours,
            "steps": [step.to_dict() for step in self.steps],
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        return (
            f"{self.action} | {self.version} → {self.target_version or '（无目标）'} | "
            f"{len(self.steps)} 步（阻塞 {len(self.blocking_steps())}） | {self.reason}"
        )

    def render_markdown(self) -> str:
        """把计划渲染成 markdown（可直接贴进事故工单）."""
        lines = [
            f"# 回滚计划：{self.action}",
            "",
            f"- 出问题的版本：v{self.version}",
            f"- 回滚目标：{('v' + self.target_version) if self.target_version else '（无）'}",
            f"- 目标版本键：`{self.target_key}`" if self.target_key else "- 目标版本键：（无）",
            f"- 观察窗口：{self.observe_window_hours}h",
            f"- 结论：{self.reason}",
            "",
        ]
        if not self.steps:
            lines.append("（无可执行动作）")
            lines.append("")
            return "\n".join(lines)
        lines.extend(["| 序 | 动作 | 阻塞 | 说明 |", "|----|------|------|------|"])
        lines.extend(
            f"| {step.order} | `{step.action}` | {'是' if step.blocking else '否'} | "
            f"{step.detail} |"
            for step in self.steps
        )
        lines.append("")
        return "\n".join(lines)


def plan_rollback(
    registry: ModelRegistry,
    version: str,
    *,
    reason: str = "",
    observe_window_hours: float = DEFAULT_OBSERVE_WINDOW_HOURS,
) -> RollbackPlan:
    """为某个版本生成回滚计划（找最近一个可部署的 stable 祖先）.

    目标选择规则：``stable_ancestors`` 里**第一个可部署**的版本（从近到远）。
    跳过不可部署的祖先是必要的——一个 ``stable`` 记录可能只是"当年提升过"，
    它的磁盘产物早就被清理了；把回滚目标选成它，等于计划了一个必然失败的动作。
    """
    if observe_window_hours <= 0:
        raise RegistryError(f"observe_window_hours 必须为正数，收到 {observe_window_hours}")
    current = registry.get(version)
    if current.stage != STAGE_STABLE:
        # 只有 stable 版本才谈得上"回滚"：候选退回候选不需要任何动作
        return RollbackPlan(
            action=ACTION_HOLD,
            reason=(
                f"v{current.version} 当前阶段是 {current.stage}，不是 stable："
                "回滚只针对正在服务的版本"
            ),
            version=current.version,
            observe_window_hours=observe_window_hours,
        )

    candidates = registry.stable_ancestors(current.version)
    deployable = [item for item in candidates if item.is_deployable()]
    if not deployable:
        return RollbackPlan(
            action=ACTION_HOLD,
            reason=(
                f"v{current.version} 没有可部署的 stable 祖先"
                f"（祖先 {len(candidates)} 个，可部署 0 个）："
                "回滚目标必须是一个真的能兜住流量的版本"
            ),
            version=current.version,
            observe_window_hours=observe_window_hours,
        )

    target = deployable[0]
    steps = (
        RollbackStep(
            order=1,
            action=STEP_VERIFY,
            detail=(
                f"校验目标 v{target.version}（键 {target.version_key}）产物齐全："
                f"适配器={bool(target.artifacts.get('adapter'))}, "
                f"合并模型={bool(target.artifacts.get('merged'))}"
            ),
            # 第一步必须确认"有退路"：先冻结再校验，一旦校验失败就没有可服务版本
            blocking=True,
        ),
        RollbackStep(
            order=2,
            action=STEP_FREEZE,
            detail=(
                f"把 v{current.version} 置为 rolled_back"
                "（head() 会自动落回上一个 stable，无需反向改写目标版本）"
            ),
            blocking=True,
        ),
        RollbackStep(
            order=3,
            action=STEP_RECORD,
            detail=f"写入审计事件：原因「{reason or '未说明'}」",
            blocking=False,
        ),
        RollbackStep(
            order=4,
            action=STEP_OBSERVE,
            detail=(
                f"按 {observe_window_hours}h 观察窗口监控 v{target.version}："
                "切换是瞬时动作，事故是否停止需要窗口内的线上数据才能确认"
            ),
            blocking=False,
        ),
    )
    plan = RollbackPlan(
        action=ACTION_ROLLBACK,
        reason=reason or f"从 v{current.version} 回滚到 v{target.version}",
        version=current.version,
        target_version=target.version,
        target_key=target.version_key,
        steps=steps,
        observe_window_hours=observe_window_hours,
    )
    logger.info("回滚计划：%s", plan.summary_line())
    return plan


def apply_rollback(registry: ModelRegistry, plan: RollbackPlan) -> ModelVersion:
    """执行回滚计划（只做``freeze``这一步的副作用），返回被冻结的记录.

    刻意**不接受"任意计划"**：``should_execute`` 为假时抛错。
    一个 ``hold`` 计划被误执行，等于在没有目标的情况下把生产版本退掉
    ----这是本模块能造成的最大伤害，所以在唯一有副作用的入口拦一次。
    """
    if not plan.should_execute:
        raise RegistryError(
            f"计划不可执行（action={plan.action}，步数 {len(plan.steps)}）："
            f"{plan.reason}"
        )
    current = registry.get(plan.version)
    return registry.set_stage(
        current.version,
        STAGE_ROLLED_BACK,
        reason=f"回滚到 v{plan.target_version}：{plan.reason}",
    )


__all__ = [
    "ACTIONS",
    "ACTION_HOLD",
    "ACTION_PROMOTE",
    "ACTION_ROLLBACK",
    "CHECK_ABSOLUTE",
    "CHECK_AGE",
    "CHECK_COMPARABLE",
    "CHECK_DEPLOYABLE",
    "CHECK_GAIN",
    "CHECK_METRICS",
    "DEFAULT_OBSERVE_WINDOW_HOURS",
    "STEP_FREEZE",
    "STEP_OBSERVE",
    "STEP_RECORD",
    "STEP_VERIFY",
    "CheckResult",
    "PromotionDecision",
    "PromotionPolicy",
    "RollbackPlan",
    "RollbackStep",
    "apply_rollback",
    "candidate_age_hours",
    "evaluate_candidate",
    "plan_rollback",
]
