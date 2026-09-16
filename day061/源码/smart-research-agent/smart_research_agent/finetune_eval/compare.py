"""配对比较与回归门禁：微调到底把什么改好了、把什么改坏了（M5-D5）.

"微调前的 62.5% vs 微调后的 83.3%"这句话在评估报告里几乎总是**不充分**的。
它至少缺三样东西：

1. **配对的证据**。两臂跑的是同一份评估集、同一条用例，所以能看的不是
   两个独立比例，而是**同一条用例上的翻转**：``b`` = 之前过、之后不过；
   ``c`` = 之前不过、之后过。翻转型统计量（McNemar）比"两个比例相减"
   敏感得多，因为它把"两边都过"的那些用例从分母里剔掉了。
2. **不确定度**。18 条用例上 62.5% → 83.3% 是"变好了"还是"运气"？
   配对自助法（paired bootstrap）给出差值的区间——注意必须是**配对**的，
   对两臂各自抽样的区间会宽得多，因为那样丢掉了配对信息。
3. **分桶的方向**。总体涨了不等于处处涨：一个桶涨 50%、另一个桶掉 25%
   也能凑出"总体变好"。所以门禁按**桶**设，而不是只看总分。

三个实现细节值得单独说：

- **McNemar 用精确二项检验**，不用卡方近似。``b + c`` 很小（本课实测
  6 个不一致对）时卡方近似不可靠；精确检验在这个规模上只要几个组合数，
  而且是"定义明确"的那个答案。卡方统计量仍然作为参考值报出来。
- **自助法用固定种子的 ``random.Random``**，所以同一份结果重跑永远得到
  同一个区间——评估报告里的数字**必须可复现**，否则它就不是证据。
- **回归判定带一个极小容差**。浮点相减得到的 ``-0.0`` 或 ``1e-16`` 不该
  被写成"回退"，否则门禁会被浮点噪声反复触发——**一个总是报警的门禁
  很快就会被忽略**。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.finetune_eval.evaluator import EvalRun, ItemOutcome
from smart_research_agent.finetune_eval.metrics import FinetuneEvalError
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: 回归判定的浮点容差：差值绝对值不超过它就不算回退.
RATE_TOLERANCE = 1e-9

#: 自助法的默认重采样次数（2000 次在本课规模上不到 20 ms）.
DEFAULT_BOOTSTRAP_SAMPLES = 2000

#: 默认显著性水平.
DEFAULT_ALPHA = 0.05


@dataclass(frozen=True)
class McNemarResult:
    """配对翻转型检验的结果（``b`` / ``c`` 的定义见模块文档）."""

    before_pass_after_fail: int
    before_fail_after_pass: int
    discordant: int
    chi_square: float
    p_value: float
    alpha: float

    @property
    def significant(self) -> bool:
        """是否在给定显著性水平下显著（``p < alpha``）."""
        return self.p_value < self.alpha

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        return (
            f"McNemar 精确检验：过→不过 {self.before_pass_after_fail} 条，"
            f"不过→过 {self.before_fail_after_pass} 条（不一致 {self.discordant} 条）"
            f" | p = {self.p_value:.6f} | 显著 {self.significant}"
        )

    def to_dict(self) -> dict[str, Any]:
        """投影为可 json.dumps 的字典."""
        return {
            "before_pass_after_fail": self.before_pass_after_fail,
            "before_fail_after_pass": self.before_fail_after_pass,
            "discordant": self.discordant,
            "chi_square": self.chi_square,
            "p_value": self.p_value,
            "alpha": self.alpha,
            "significant": self.significant,
        }


@dataclass(frozen=True)
class BootstrapResult:
    """配对自助法的差值区间."""

    mean_delta: float
    ci_low: float
    ci_high: float
    samples: int
    alpha: float

    @property
    def excludes_zero(self) -> bool:
        """区间是否完全落在 0 的同一侧（"变化方向明确"）."""
        return self.ci_low > 0 or self.ci_high < 0

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        level = (1 - self.alpha) * 100
        return (
            f"配对自助法（{self.samples} 次）：平均总分差 {self.mean_delta:+.4f}"
            f" | {level:.0f}% 区间 [{self.ci_low:+.4f}, {self.ci_high:+.4f}]"
            f" | 不含 0 {self.excludes_zero}"
        )

    def to_dict(self) -> dict[str, Any]:
        """投影为可 json.dumps 的字典."""
        return {
            "mean_delta": self.mean_delta,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "samples": self.samples,
            "alpha": self.alpha,
            "excludes_zero": self.excludes_zero,
        }


@dataclass(frozen=True)
class BucketDelta:
    """一个分组（桶或难度档）上的配对比对结果."""

    name: str
    total: int
    before_pass_rate: float
    after_pass_rate: float
    before_mean_score: float
    after_mean_score: float
    regressed: bool = False

    @property
    def pass_rate_delta(self) -> float:
        """合格率差值（后 - 前）."""
        return self.after_pass_rate - self.before_pass_rate

    @property
    def score_delta(self) -> float:
        """平均总分差值（后 - 前）."""
        return self.after_mean_score - self.before_mean_score

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        flag = " ⚠回退" if self.regressed else ""
        return (
            f"{self.name}（{self.total} 条）：合格率 {self.before_pass_rate:.1%} → "
            f"{self.after_pass_rate:.1%}（{self.pass_rate_delta:+.1%}），"
            f"总分 {self.before_mean_score:.4f} → {self.after_mean_score:.4f}"
            f"（{self.score_delta:+.4f}）{flag}"
        )

    def to_dict(self) -> dict[str, Any]:
        """投影为可 json.dumps 的字典."""
        return {
            "name": self.name,
            "total": self.total,
            "before_pass_rate": self.before_pass_rate,
            "after_pass_rate": self.after_pass_rate,
            "pass_rate_delta": self.pass_rate_delta,
            "before_mean_score": self.before_mean_score,
            "after_mean_score": self.after_mean_score,
            "score_delta": self.score_delta,
            "regressed": self.regressed,
        }


@dataclass
class ComparisonReport:
    """一次"微调前 vs 微调后"的完整比对报告."""

    before_name: str
    after_name: str
    total: int
    before_pass_rate: float
    after_pass_rate: float
    before_mean_score: float
    after_mean_score: float
    buckets: list[BucketDelta] = field(default_factory=list)
    difficulties: list[BucketDelta] = field(default_factory=list)
    mcnemar: McNemarResult | None = None
    bootstrap: BootstrapResult | None = None
    regressions: list[str] = field(default_factory=list)
    max_regression: float = 0.0

    @property
    def pass_rate_delta(self) -> float:
        """总体合格率差值（后 - 前）."""
        return self.after_pass_rate - self.before_pass_rate

    @property
    def score_delta(self) -> float:
        """总体平均总分差值（后 - 前）."""
        return self.after_mean_score - self.before_mean_score

    @property
    def passed(self) -> bool:
        """回归门禁：**没有任何分组超过允许的回退幅度**."""
        return not self.regressions

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        return (
            f"{self.before_name} → {self.after_name} | {self.total} 条 | "
            f"合格率 {self.before_pass_rate:.1%} → {self.after_pass_rate:.1%}"
            f"（{self.pass_rate_delta:+.1%}）| 总分 {self.before_mean_score:.4f} → "
            f"{self.after_mean_score:.4f}（{self.score_delta:+.4f}）| "
            f"门禁 {'通过' if self.passed else '未通过'}"
            f"（回退分组：{', '.join(self.regressions) or '无'}）"
        )

    def to_dict(self) -> dict[str, Any]:
        """投影为可 json.dumps 的字典."""
        return {
            "before_name": self.before_name,
            "after_name": self.after_name,
            "total": self.total,
            "before_pass_rate": self.before_pass_rate,
            "after_pass_rate": self.after_pass_rate,
            "pass_rate_delta": self.pass_rate_delta,
            "before_mean_score": self.before_mean_score,
            "after_mean_score": self.after_mean_score,
            "score_delta": self.score_delta,
            "buckets": [bucket.to_dict() for bucket in self.buckets],
            "difficulties": [bucket.to_dict() for bucket in self.difficulties],
            "mcnemar": self.mcnemar.to_dict() if self.mcnemar else None,
            "bootstrap": self.bootstrap.to_dict() if self.bootstrap else None,
            "regressions": list(self.regressions),
            "max_regression": self.max_regression,
            "passed": self.passed,
        }

    def format_lines(self) -> list[str]:
        """渲染成 markdown 表格行（报告文件与控制台共用同一份渲染）."""
        lines = [
            f"| {bucket.name} | {bucket.total} | {bucket.before_pass_rate:.1%} | "
            f"{bucket.after_pass_rate:.1%} | {bucket.pass_rate_delta:+.1%} | "
            f"{bucket.score_delta:+.4f} | {'是' if bucket.regressed else '否'} |"
            for bucket in self.buckets
        ]
        return lines


def binomial_two_sided_p_value(successes: int, total: int) -> float:
    """``p = 0.5`` 下两尾精确二项检验的 p 值（``min(1, 2·Σ_{k≤min(b,c)} C(n,k)/2^n)``）.

    ``total = 0`` 时返回 ``1.0``：不一致对为 0 意味着"两臂完全一样"，
    此时**没有任何理由说它们不同**（p 值接近 1 才是诚实的答案）。
    两尾求和会超过 1（当分布重心偏向一侧时），所以最后要取 ``min(1, ·)``——
    这是离散分布做两尾检验的常规处理，不是修补。
    """
    if total < 0:
        raise FinetuneEvalError(f"total 不能为负数，收到 {total}")
    if not 0 <= successes <= total:
        raise FinetuneEvalError(f"successes={successes} 必须落在 [0, {total}]")
    if total == 0:
        return 1.0
    tail = min(successes, total - successes)
    cumulative = sum(math.comb(total, k) for k in range(tail + 1))
    return min(1.0, 2 * cumulative / (2**total))


def mcnemar_exact(
    before_pass_after_fail: int,
    before_fail_after_pass: int,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> McNemarResult:
    """McNemar 精确检验：只看**翻转**的那两格.

    两个计数必须是非负整数；``b + c = 0`` 时 p 值为 1.0（无法拒绝原假设）。
    卡方统计量带连续性修正（``(|b-c|-1)²/(b+c)``）**只作为参考值**报出，
    不作为判定依据——本课规模下精确检验就那么几个组合数，没有理由用近似。
    """
    if before_pass_after_fail < 0 or before_fail_after_pass < 0:
        raise FinetuneEvalError("McNemar 的两个计数不能为负数")
    if not 0.0 < alpha < 1.0:
        raise FinetuneEvalError(f"alpha 必须落在 (0, 1)，收到 {alpha}")
    discordant = before_pass_after_fail + before_fail_after_pass
    if discordant == 0:
        chi_square = 0.0
    else:
        difference = abs(before_pass_after_fail - before_fail_after_pass)
        chi_square = max(0.0, difference - 1) ** 2 / discordant
    return McNemarResult(
        before_pass_after_fail=before_pass_after_fail,
        before_fail_after_pass=before_fail_after_pass,
        discordant=discordant,
        chi_square=chi_square,
        p_value=binomial_two_sided_p_value(
            before_fail_after_pass, discordant
        ),
        alpha=alpha,
    )


def percentile(sorted_values: list[float], quantile: float) -> float:
    """线性插值分位数（``quantile`` 落在 ``[0, 1]``）.

    实现按"秩 = q·(n-1) 再线性插值"的常规做法，而不是取最近秩：
    取最近秩会让 2000 次重采样的区间在 ``n`` 很小时出现台阶，
    而报告里的区间宽度会被这些台阶影响。
    """
    if not sorted_values:
        raise FinetuneEvalError("不能对空序列取分位数")
    if not 0.0 <= quantile <= 1.0:
        raise FinetuneEvalError(f"quantile 必须落在 [0, 1]，收到 {quantile}")
    position = quantile * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def paired_bootstrap_delta(
    before_scores: list[float],
    after_scores: list[float],
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 42,
    alpha: float = DEFAULT_ALPHA,
) -> BootstrapResult:
    """配对自助法：对**逐条差值**重采样，给出平均总分差的区间.

    抽的是"条目"而不是"两次独立运行的结果"：配对信息（同一份评估集、
    同一条用例）必须被保留，否则区间会宽到得不出结论。
    """
    if len(before_scores) != len(after_scores):
        raise FinetuneEvalError(
            f"配对自助法要求两臂逐条对应：{len(before_scores)} != {len(after_scores)}"
        )
    if not before_scores:
        raise FinetuneEvalError("配对自助法需要至少一条用例")
    if samples <= 0:
        raise FinetuneEvalError(f"samples 必须为正整数，收到 {samples}")
    if not 0.0 < alpha < 1.0:
        raise FinetuneEvalError(f"alpha 必须落在 (0, 1)，收到 {alpha}")
    deltas = [
        after - before for before, after in zip(before_scores, after_scores, strict=True)
    ]
    count = len(deltas)
    mean_delta = sum(deltas) / count
    rng = random.Random(seed)
    resampled_means: list[float] = []
    for _ in range(samples):
        total = 0.0
        for _ in range(count):
            total += deltas[rng.randrange(count)]
        resampled_means.append(total / count)
    resampled_means.sort()
    return BootstrapResult(
        mean_delta=mean_delta,
        ci_low=percentile(resampled_means, alpha / 2),
        ci_high=percentile(resampled_means, 1 - alpha / 2),
        samples=samples,
        alpha=alpha,
    )


def _align(before: EvalRun, after: EvalRun) -> list[tuple[ItemOutcome, ItemOutcome]]:
    """按 item_id 对齐两臂，并要求**顺序一致**（顺序不一致也拒绝）.

    顺序也校验，是因为报告的 ``format_lines`` 是按顺序逐行渲染的：
    允许乱序会让同一份数据渲染出两种表格，而人眼对比表格时不会去核对
    每行的 id。
    """
    if before.total != after.total:
        raise FinetuneEvalError(
            f"两臂的用例数不一致：{before.total} != {after.total}"
        )
    if before.total == 0:
        raise FinetuneEvalError("不能对比空运行")
    pairs = list(zip(before.outcomes, after.outcomes, strict=True))
    for left, right in pairs:
        if left.item_id != right.item_id:
            raise FinetuneEvalError(
                f"两臂的用例顺序不一致：{left.item_id} != {right.item_id}"
            )
    return pairs


def _group_deltas(
    pairs: list[tuple[ItemOutcome, ItemOutcome]],
    *,
    key: str,
    max_regression: float,
    order: tuple[str, ...] | None = None,
) -> list[BucketDelta]:
    """按 ``key``（``bucket`` 或 ``difficulty``）分组，产出配对比对表."""
    grouped: dict[str, list[tuple[ItemOutcome, ItemOutcome]]] = {}
    for left, right in pairs:
        name = left.bucket if key == "bucket" else left.difficulty
        grouped.setdefault(name, []).append((left, right))
    # ``order`` 只负责"已知分组"的次序；不在 ``order`` 里的名字（例如某个
    # 非法的难度档）**追加在后面而不是被丢掉**——静默少一行会让报告看起来
    # "一切正常"，而那正是最难发现的报告缺陷。
    names = ([] if order is None else list(order))
    names += sorted(set(grouped) - set(names))
    table: list[BucketDelta] = []
    for name in names:
        members = grouped.get(name)
        if not members:
            continue
        before_rate = sum(1 for left, _ in members if left.passed) / len(members)
        after_rate = sum(1 for _, right in members if right.passed) / len(members)
        before_score = sum(left.score for left, _ in members) / len(members)
        after_score = sum(right.score for _, right in members) / len(members)
        table.append(
            BucketDelta(
                name=name,
                total=len(members),
                before_pass_rate=before_rate,
                after_pass_rate=after_rate,
                before_mean_score=before_score,
                after_mean_score=after_score,
                regressed=(before_rate - after_rate) > max_regression + RATE_TOLERANCE,
            )
        )
    return table


def compare_runs(
    before: EvalRun,
    after: EvalRun,
    *,
    max_regression: float = 0.0,
    alpha: float = DEFAULT_ALPHA,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 42,
) -> ComparisonReport:
    """完整比对两臂：总体差值 + 分桶/分难度差值 + McNemar + 配对自助法 + 门禁.

    ``max_regression`` 是**每个分组**允许的最大合格率回退（缺省 0.0，
    即"一处都不许掉"）。把它设成 0 而不是"总体不降"，是因为微调的
    典型事故恰恰是"总体涨了、某一类掉得厉害"——而那一类往往是安全类
    用例，掉下去不会被总分反映出来。
    """
    if max_regression < 0:
        raise FinetuneEvalError(f"max_regression 不能为负数，收到 {max_regression}")
    pairs = _align(before, after)
    buckets = _group_deltas(pairs, key="bucket", max_regression=max_regression)
    difficulties = _group_deltas(
        pairs, key="difficulty", max_regression=max_regression, order=("easy", "normal", "hard")
    )
    before_pass_after_fail = sum(1 for left, right in pairs if left.passed and not right.passed)
    before_fail_after_pass = sum(1 for left, right in pairs if not left.passed and right.passed)
    report = ComparisonReport(
        before_name=before.name,
        after_name=after.name,
        total=len(pairs),
        before_pass_rate=before.pass_rate,
        after_pass_rate=after.pass_rate,
        before_mean_score=before.mean_score,
        after_mean_score=after.mean_score,
        buckets=buckets,
        difficulties=difficulties,
        mcnemar=mcnemar_exact(
            before_pass_after_fail, before_fail_after_pass, alpha=alpha
        ),
        bootstrap=paired_bootstrap_delta(
            [left.score for left, _ in pairs],
            [right.score for _, right in pairs],
            samples=samples,
            seed=seed,
            alpha=alpha,
        ),
        regressions=[bucket.name for bucket in buckets if bucket.regressed],
        max_regression=max_regression,
    )
    logger.info("对比完成：%s", report.summary_line())
    return report


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "RATE_TOLERANCE",
    "BootstrapResult",
    "BucketDelta",
    "ComparisonReport",
    "McNemarResult",
    "binomial_two_sided_p_value",
    "compare_runs",
    "mcnemar_exact",
    "paired_bootstrap_delta",
    "percentile",
]
