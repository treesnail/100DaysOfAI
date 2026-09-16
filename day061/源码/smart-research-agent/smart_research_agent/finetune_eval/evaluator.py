"""评估执行器：逐条跑、逐条记、把"合格"定义清楚（M5-D5）.

执行器本身很薄，它的价值在于三件被显式化的约定：

1. **``passed`` 的定义**。合格 = 事实点全覆盖 + 没有禁项 + 格式合规 +
   拒答行为与预期一致。它是一个**合取**，不是加权平均的阈值——因为
   "编造了一个数字"和"少说了一个要点"不该互相抵消。加权总分
   （``metrics.weighted_total``）单独报告，用来回答"好多少"。
2. **单条崩溃不拖垮整场评估**。这与 day030 ``Harness.run`` 的纪律一致：
   runner 抛异常时记 ``error`` 并把这条算作**不合格**（而不是"跳过、不计入
   分母"）。跳过会让最差的那条用例凭空消失，分母一变，分数就再也不能
   与上一次比较了。
3. **延迟与总数一起报**。80.2 秒跑 18 条本身就是"评估要花多少机器时间"
   的答案，而 day052 的 ``plan_distributed`` 已经教过：**先算预算再跑**。

另外提供两个**脚本化对照臂**（``scripted_arm``）：它们是确定性的函数，
不是真的模型输出。用它们的理由很实际——本课要在离线、零 GPU 的条件
下演示"微调前 vs 微调后"的完整对比流程，而流程的正确性与"这两段文本
是不是某个真实模型生成的"无关。教程与报告里都如实标注这一点。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.evaluation.harness import (
    CaseResult,
    EvalCase,
    EvaluationHarness,
)
from smart_research_agent.finetune_eval.metrics import (
    COMPONENT_NAMES,
    DEFAULT_WEIGHTS,
    FinetuneEvalError,
    MetricBreakdown,
    compute_components,
    fact_recall,
    forbidden_hits,
    format_violations,
    is_refusal,
)
from smart_research_agent.finetune_eval.suites import EvalItem
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: 对照臂的名字（进报告，必须如实标注"脚本化"）.
SCRIPTED_ARMS: tuple[str, ...] = ("baseline", "finetuned")

#: 基线臂保留的文本比例（截断模拟"基座模型只说了前半句"）.
BASELINE_KEEP_RATIO = 0.5

#: 基线臂的固定尾注（它会让紧字数预算的用例超限，这正是要观察的失败）.
BASELINE_SUFFIX = "以上内容仅供参考，具体以文档为准。"

#: 微调臂在**难例**上保留的文本比例（模拟"难例上仍有尾部丢失"）.
#:
#: 这个数字是本课标定出来的，不是拍脑袋：取 0.85 时难例也全线通过，
#: 于是 ``by_difficulty`` 那张表全是 100%，**评估集里的难例等于白放**；
#: 取 0.70 时难例恰好开始掉（尾部的事实点与段落标记被截掉）。
#: "让对照臂的退化程度可标定"本身就是一条工程经验——一个怎么跑都
#: 满分（或怎么跑都零分）的对照臂，测不出任何东西。
FINETUNED_HARD_KEEP_RATIO = 0.7

#: Wilson 区间缺省的 95% 置信水平对应的 z 值.
DEFAULT_Z = 1.96

#: 难度档的固定顺序（报告与测试都依赖它）.
DIFFICULTY_ORDER: tuple[str, ...] = ("easy", "normal", "hard")


@dataclass(frozen=True)
class ItemOutcome:
    """单条用例的评估结果（判定 + 分量 + 证据，三样一起留档）."""

    item_id: str
    bucket: str
    difficulty: str
    output: str = ""
    metric: MetricBreakdown = field(default_factory=MetricBreakdown)
    passed: bool = False
    missing_facts: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    violations: tuple[str, ...] = ()
    refusal_correct: bool = True
    latency_seconds: float = 0.0
    error: str = ""

    @property
    def score(self) -> float:
        """加权总分（``metrics.DEFAULT_WEIGHTS`` 下的加权和）."""
        return self.metric.total

    @property
    def components(self) -> dict[str, float]:
        """六个分量."""
        return self.metric.components

    def failure_reason(self) -> str:
        """把失败原因拼成一句可直接进日志的话（合格时返回空串）.

        它存在的原因是**报告里最常见的需求是"哪几条为什么不合格"**：
        只有布尔值时，看报告的人得自己去翻原文；把原因写进结果，
        那一条用例就不需要二次查询了。
        """
        if self.passed:
            return ""
        if self.error:
            return f"执行异常：{self.error}"
        reasons: list[str] = []
        if self.missing_facts:
            reasons.append("缺事实点 " + "、".join(self.missing_facts))
        if self.forbidden:
            reasons.append("命中禁项 " + "、".join(self.forbidden))
        if self.violations:
            reasons.extend(self.violations)
        if not self.refusal_correct:
            reasons.append("拒答行为与预期不符")
        return "；".join(reasons) if reasons else "未通过（原因未记录）"

    def to_dict(self) -> dict[str, Any]:
        """投影为可 json.dumps 的字典."""
        return {
            "item_id": self.item_id,
            "bucket": self.bucket,
            "difficulty": self.difficulty,
            "score": self.score,
            "passed": self.passed,
            "components": {name: self.components[name] for name in COMPONENT_NAMES},
            "missing_facts": list(self.missing_facts),
            "forbidden": list(self.forbidden),
            "violations": list(self.violations),
            "refusal_correct": self.refusal_correct,
            "latency_seconds": self.latency_seconds,
            "error": self.error,
            "failure_reason": self.failure_reason(),
        }


@dataclass
class EvalRun:
    """一次完整评估运行（逐条结果 + 汇总），可直接交给 ``compare.compare_runs``."""

    name: str
    outcomes: list[ItemOutcome] = field(default_factory=list)
    wall_seconds: float = 0.0

    @property
    def total(self) -> int:
        """用例总数（**分母包含执行失败的用例**）."""
        return len(self.outcomes)

    @property
    def passed(self) -> int:
        """合格条数."""
        return sum(1 for outcome in self.outcomes if outcome.passed)

    @property
    def pass_rate(self) -> float:
        """合格率；空运行时为 0.0（而不是抛异常：报告要能打印半成品）."""
        return self.passed / self.total if self.total else 0.0

    @property
    def mean_score(self) -> float:
        """平均加权总分；空运行时为 0.0."""
        return (
            sum(outcome.score for outcome in self.outcomes) / self.total
            if self.total
            else 0.0
        )

    @property
    def error_count(self) -> int:
        """runner 抛异常的条数（它们已经计入分母）."""
        return sum(1 for outcome in self.outcomes if outcome.error)

    @property
    def mean_latency_seconds(self) -> float:
        """平均单条延迟（测的是 runner 的调用耗时，不含指标计算）."""
        return (
            sum(outcome.latency_seconds for outcome in self.outcomes) / self.total
            if self.total
            else 0.0
        )

    def by_bucket(self) -> dict[str, dict[str, Any]]:
        """按能力桶聚合（桶名排序，便于两次运行逐行对比）."""
        grouped: dict[str, list[ItemOutcome]] = {}
        for outcome in self.outcomes:
            grouped.setdefault(outcome.bucket, []).append(outcome)
        return {
            bucket: {
                "total": len(members),
                "passed": sum(1 for member in members if member.passed),
                "pass_rate": sum(1 for member in members if member.passed) / len(members),
                "mean_score": sum(member.score for member in members) / len(members),
            }
            for bucket, members in sorted(grouped.items())
        }

    def by_difficulty(self) -> dict[str, dict[str, Any]]:
        """按难度聚合（顺序固定为 easy / normal / hard）."""
        grouped: dict[str, list[ItemOutcome]] = {}
        for outcome in self.outcomes:
            grouped.setdefault(outcome.difficulty, []).append(outcome)
        return {
            level: {
                "total": len(grouped[level]),
                "passed": sum(1 for member in grouped[level] if member.passed),
                "pass_rate": sum(1 for member in grouped[level] if member.passed)
                / len(grouped[level]),
                "mean_score": sum(member.score for member in grouped[level])
                / len(grouped[level]),
            }
            for level in DIFFICULTY_ORDER
            if level in grouped
        }

    def failures(self) -> list[ItemOutcome]:
        """所有不合格条目（报告里"最该先看的一段"）."""
        return [outcome for outcome in self.outcomes if not outcome.passed]

    def wilson(self, *, z: float = DEFAULT_Z) -> tuple[float, float]:
        """合格率的 Wilson 95% 区间（小样本下比正态近似可靠）."""
        return wilson_interval(self.passed, self.total, z=z)

    def to_dict(self) -> dict[str, Any]:
        """投影为可 json.dumps 的字典（含区间与两层聚合）."""
        low, high = self.wilson()
        return {
            "name": self.name,
            "total": self.total,
            "passed": self.passed,
            "pass_rate": self.pass_rate,
            "pass_rate_ci95": [low, high],
            "mean_score": self.mean_score,
            "error_count": self.error_count,
            "mean_latency_seconds": self.mean_latency_seconds,
            "wall_seconds": self.wall_seconds,
            "by_bucket": self.by_bucket(),
            "by_difficulty": self.by_difficulty(),
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        low, high = self.wilson()
        return (
            f"{self.name} | {self.passed}/{self.total} 合格"
            f"（{self.pass_rate:.2%}，95% CI {low:.2%}~{high:.2%}）"
            f" | 平均总分 {self.mean_score:.4f}"
            f" | 异常 {self.error_count} 条 | 耗时 {self.wall_seconds:.3f}s"
        )


def wilson_interval(successes: int, total: int, *, z: float = DEFAULT_Z) -> tuple[float, float]:
    """合格率的 Wilson 得分区间（**不用正态近似**）.

    为什么不用 ``p ± z·sqrt(p(1-p)/n)``：当 ``p = 1.0``（全对）时那个公式
    给出宽度 0 的区间——"18 条全对"被说成"合格率精确等于 100%"，
    而在 18 条样本上这个结论显然过强。Wilson 区间在 ``p`` 接近 0 或 1 时
    仍然给出合理宽度，这也是它在小样本评估里被普遍采用的原因。

    边界：``total = 0`` 时返回 ``(0.0, 1.0)``——**空样本的诚实答案是
    "什么都不知道"**，而不是 0 或 1。
    """
    if total < 0:
        raise FinetuneEvalError(f"total 不能为负数，收到 {total}")
    if not 0 <= successes <= total:
        raise FinetuneEvalError(f"successes={successes} 必须落在 [0, {total}]")
    if z <= 0:
        raise FinetuneEvalError(f"z 必须为正数，收到 {z}")
    if total == 0:
        return (0.0, 1.0)
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * (
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        ** 0.5
        / denominator
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def score_item(item: EvalItem, output: str) -> ItemOutcome:
    """给一条输出打分并给出合格判定（不测延迟：那是 ``run_suite`` 的事）."""
    metric = compute_components(
        output,
        item.reference,
        required_facts=item.required_facts,
        forbidden_facts=item.forbidden_facts,
        rules=item.format_rules,
        expect_refusal=item.expect_refusal,
    )
    recall = fact_recall(output, item.required_facts)
    forbidden = forbidden_hits(output, item.forbidden_facts)
    violations = format_violations(output, item.format_rules)
    refusal_correct = is_refusal(output) == item.expect_refusal
    passed = (
        not recall.missing
        and not forbidden
        and not violations
        and refusal_correct
    )
    return ItemOutcome(
        item_id=item.id,
        bucket=item.bucket,
        difficulty=item.difficulty,
        output=output,
        metric=metric,
        passed=passed,
        missing_facts=recall.missing,
        forbidden=forbidden,
        violations=violations,
        refusal_correct=refusal_correct,
    )


def run_suite(
    items: Sequence[EvalItem],
    runner: Callable[[EvalItem], str],
    *,
    name: str = "run",
    clock: Callable[[], float] = time.perf_counter,
) -> EvalRun:
    """对评估集逐条执行 runner 并汇总（runner 异常被隔离成"不合格"）.

    ``clock`` 可注入是为了让延迟可测：真实评估里"这一条为什么慢"经常
    比"这一条为什么错"更值得查（超长输出、工具超时、重试风暴）。
    """
    if not items:
        raise FinetuneEvalError("不能对空评估集运行评估")
    started = clock()
    run = EvalRun(name=name)
    for item in items:
        item_started = clock()
        try:
            output = runner(item)
        except Exception as exc:  # noqa: BLE001 - 单条崩溃必须被隔离
            elapsed = clock() - item_started
            logger.warning("用例 %s 的 runner 异常：%s", item.id, exc)
            run.outcomes.append(
                ItemOutcome(
                    item_id=item.id,
                    bucket=item.bucket,
                    difficulty=item.difficulty,
                    metric=MetricBreakdown(
                        components={component: 0.0 for component in COMPONENT_NAMES},
                        total=0.0,
                    ),
                    passed=False,
                    error=f"{type(exc).__name__}: {exc}",
                    latency_seconds=elapsed,
                )
            )
            continue
        elapsed = clock() - item_started
        outcome = score_item(item, output)
        run.outcomes.append(
            ItemOutcome(
                item_id=outcome.item_id,
                bucket=outcome.bucket,
                difficulty=outcome.difficulty,
                output=outcome.output,
                metric=outcome.metric,
                passed=outcome.passed,
                missing_facts=outcome.missing_facts,
                forbidden=outcome.forbidden,
                violations=outcome.violations,
                refusal_correct=outcome.refusal_correct,
                latency_seconds=elapsed,
            )
        )
    run.wall_seconds = clock() - started
    logger.info("评估完成：%s", run.summary_line())
    return run


def scripted_arm(*, arm: str) -> Callable[[EvalItem], str]:
    """构造一个**脚本化对照臂**（确定性函数，不是真的模型输出）.

    ============  ==================================================================
    ``baseline``  去掉引用尾注（``来源：…``）+ 截断到 ``BASELINE_KEEP_RATIO`` + 追加固定尾注
    ``finetuned`` 参考答案原文；**难例**截断到 ``FINETUNED_HARD_KEEP_RATIO``
    ============  ==================================================================

    两个比例是常量而不是字面量，理由是它们**被标定过**（见常量处的说明）：
    截得太多会让基线臂全线失败，截得太少会让微调臂全线通过，两种情况下
    ``by_difficulty`` 那张表都失去信息量。

    三种退化各对应一类真实失败：**不给出处**（引用能力）、**说一半**
    （召回能力）、**废话超限**（简洁度）。它们不是"随便找个办法造出
    高低差"——每一条都能在真实微调里找到对应现象，因此用它们演示的
    流程（跑两臂 → 配对比较 → 门禁）与接上真模型时完全一致。
    """
    if arm not in SCRIPTED_ARMS:
        raise FinetuneEvalError(f"未知的对照臂 {arm!r}，可选：{', '.join(SCRIPTED_ARMS)}")

    def baseline(item: EvalItem) -> str:
        text = item.reference
        if "来源：" in text:
            text = text.split("来源：", 1)[0]
        keep = max(1, int(len(text) * BASELINE_KEEP_RATIO))
        return text[:keep] + BASELINE_SUFFIX

    def finetuned(item: EvalItem) -> str:
        text = item.reference
        if item.difficulty == "hard":
            keep = max(1, int(len(text) * FINETUNED_HARD_KEEP_RATIO))
            text = text[:keep]
        return text

    return baseline if arm == "baseline" else finetuned


def item_to_case(item: EvalItem) -> EvalCase:
    """把 ``EvalItem`` 投影成 day025 的 ``EvalCase``（标签 = 桶 + 难度）.

    映射刻意是**单向投影**：``EvalCase`` 只承载"执行与分组"需要的东西
    （id / 输入 / 参考 / 标签），事实点与格式契约留在 ``EvalItem`` 里。
    把两套字段硬塞进一个类，会让"哪一层负责判定"重新变模糊。
    """
    return EvalCase(
        id=item.id,
        name=item.id,
        input=item.instruction,
        expected=item.reference,
        tags=list(item.tags),
        metadata={"bucket": item.bucket, "difficulty": item.difficulty},
    )


class FineTuneEvalHarness(EvaluationHarness):
    """把领域评估集接进 day025 的 ``EvaluationHarness``（零重复实现）.

    复用的是 harness 的**执行与汇总骨架**：``run(list[EvalCase])`` 逐条调用
    ``evaluate``，``summary()`` 给出 ``by_tag`` 分组，``write_report`` 落盘。
    不复用的是它的数据集加载器：``EvalItem`` 的字段比 ``EvalCase`` 多
    （事实点、禁项、格式契约），所以套件从 ``suites.read_suite`` 读，
    再逐条 ``add_item`` 注册进 harness——**同一份数据，两种视图**。
    """

    def __init__(
        self,
        runner: Callable[[EvalItem], str],
        *,
        name: str = "finetune-eval",
    ) -> None:
        super().__init__()
        self.runner = runner
        self.name = name
        self._items: dict[str, EvalItem] = {}
        self._outcomes: list[ItemOutcome] = []

    @property
    def items(self) -> list[EvalItem]:
        """已注册的用例（按注册顺序）."""
        return list(self._items.values())

    @property
    def outcomes(self) -> list[ItemOutcome]:
        """逐条判定结果（按执行顺序，与 ``self.results`` 一一对应）."""
        return list(self._outcomes)

    def add_item(self, item: EvalItem) -> None:
        """注册一条用例（同 id 重复注册会**替换**原条目并记警告）.

        替换而不是追加：``self.cases`` 与 ``self._items`` 必须一一对应，
        否则 ``run(self.cases)`` 会对同一条用例执行两次，而 ``summary()``
        的 ``by_tag`` 分母跟着变大——**报出来的是同一份用例，统计上却是两份**。
        """
        case = item_to_case(item)
        if item.id in self._items:
            logger.warning("用例 %s 被重复注册：后者替换前者", item.id)
            self.cases = [existing for existing in self.cases if existing.id != item.id]
        self._items[item.id] = item
        self.cases.append(case)

    def load_suite(self, items: Sequence[EvalItem]) -> None:
        """批量注册（``read_suite`` 的产物直接喂进来即可）."""
        for item in items:
            self.add_item(item)

    def to_run(self) -> EvalRun:
        """把已执行的结果打包成 ``EvalRun``（可直接与另一臂比较）."""
        return EvalRun(
            name=self.name,
            outcomes=list(self._outcomes),
            wall_seconds=sum(outcome.latency_seconds for outcome in self._outcomes),
        )

    def evaluate(self, case: EvalCase) -> CaseResult:
        """执行一条用例（harness 的扩展点）：runner → 打分 → CaseResult."""
        item = self._items.get(case.id)
        if item is None:
            raise FinetuneEvalError(f"用例 {case.id} 尚未注册：请先 add_item / load_suite")
        started = time.perf_counter()
        try:
            output = self.runner(item)
        except Exception as exc:  # noqa: BLE001 - 与 run_suite 同一套隔离纪律
            logger.warning("用例 %s 的 runner 异常：%s", item.id, exc)
            outcome = ItemOutcome(
                item_id=item.id,
                bucket=item.bucket,
                difficulty=item.difficulty,
                metric=MetricBreakdown(
                    components={component: 0.0 for component in COMPONENT_NAMES},
                    total=0.0,
                ),
                passed=False,
                error=f"{type(exc).__name__}: {exc}",
                latency_seconds=time.perf_counter() - started,
            )
        else:
            outcome = score_item(item, output)
            outcome = ItemOutcome(
                item_id=outcome.item_id,
                bucket=outcome.bucket,
                difficulty=outcome.difficulty,
                output=outcome.output,
                metric=outcome.metric,
                passed=outcome.passed,
                missing_facts=outcome.missing_facts,
                forbidden=outcome.forbidden,
                violations=outcome.violations,
                refusal_correct=outcome.refusal_correct,
                latency_seconds=time.perf_counter() - started,
            )
        self._outcomes.append(outcome)
        return CaseResult(
            case_id=item.id,
            case=case,
            output=outcome.output,
            score=outcome.score,
            passed=outcome.passed,
            latency_seconds=outcome.latency_seconds,
            details=outcome.to_dict(),
            detail=outcome.failure_reason(),
        )


def weights_snapshot() -> dict[str, float]:
    """返回当前缺省权重的副本（报告里留档用，防止"分数变了说不清为什么"）."""
    return dict(DEFAULT_WEIGHTS)


__all__ = [
    "BASELINE_KEEP_RATIO",
    "BASELINE_SUFFIX",
    "DEFAULT_Z",
    "DIFFICULTY_ORDER",
    "FINETUNED_HARD_KEEP_RATIO",
    "SCRIPTED_ARMS",
    "EvalRun",
    "FineTuneEvalHarness",
    "ItemOutcome",
    "item_to_case",
    "run_suite",
    "score_item",
    "scripted_arm",
    "weights_snapshot",
    "wilson_interval",
]
