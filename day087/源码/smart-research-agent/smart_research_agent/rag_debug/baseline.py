"""质量基线：把"这次比上次差吗"从一句感觉变成一次带方向的判定（day071）.

day046 的 ``evaluation.perf_baseline`` 已经交过一份基线（性能），本模块是它的
**质量侧对应物**，而它与那一份有一处根本差别，也是这一课最该被记住的一点：

```text
性能基线    延迟 / token / 费用    越大越差  → 用**比值**判（current / baseline > tolerance）
质量基线    召回 / 接地 / 覆盖率    越小越差  → 用**差值**判（delta < -tolerance）
                                    ↑ 方向相反
```

## 为什么质量侧必须用差值而不是比值

三条理由，每一条都能独立成立：

1. **方向反转**：性能指标的"变差"是变大，质量指标的"变差"是变小。若两块
   共用一份"比值 > 阈值"的实现，质量侧的每一次改善都会被判成回归——
   而一个"把好事报成坏事"的门禁会在第一天就被人关掉。
2. **基数为 0 是常态**：``hallucination_rate`` 与 ``fallback_rate`` 在健康
   系统上的基线值**就是 0**。比值在 0 基数下没有定义（day046 的处理是
   "跳过这条指标"），而差值在 0 基数下照样可判：``0 → 0.05`` 就是 5 个百分点
   的幻觉率上升。**跳过一条指标会让"从零变坏"这种事永远不被发现**。
3. **量纲可比**：质量指标全都是 ``[0, 1]`` 区间里的比例，差值就是"几个百分点"，
   可以直接写进报告给人读（"可核对率掉了 8 个百分点"）。

## 三张方向表：越高越好 / 越低越好 / 只做描述

``METRIC_DIRECTIONS`` 把每个指标钉死在一个方向上，而第三种方向
（``DIRECTION_NEUTRAL``）是刻意存在的：

```text
higher_better  召回 / 精确 / NDCG / packed 召回 / 接地率 / 覆盖率
lower_better   幻觉率 / 回退率 / 坏例率
neutral        llm_call_rate（这一批里有几次真的调了模型）
```

``llm_call_rate`` 为什么**不判方向**：它下降既可能是好事（检索为空时按护栏
不调模型——那是设计如此），也可能是坏事（该调没调）。给它硬安一个方向之后，
报告里会多出一类"看起来像回归、其实是护栏生效"的告警——那正是误报的来源。
它仍然**进基线、进报告、进差值表**，只是不参与回归判定；而"检索为空"这类
异常会由 ``fallback_rate`` / ``bad_case_rate`` 如实报出来。

## 死区与样本量

``dead_band``（缺省 0.02）防的是评估噪声：模型评分方差、采样随机性都会带来
±0.01 量级的波动，而一个对噪声过敏的门禁等于没有门禁（与 day027 的
``aggregate`` 死区同一条理由）。``min_samples``（缺省 3）防的是"用两条用例
宣告回归"——**样本不足时本模块拒绝给结论**，而不是给一个"通过"。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from smart_research_agent.rag_debug.errors import BaselineError
from smart_research_agent.rag_debug.types import BAD_CASE_TAGS
from smart_research_agent.retrieval.generation import FALLBACK_REASON_DESCRIPTIONS
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# 指标名（封闭清单）
# --------------------------------------------------------------------------- #

#: 检索名单那一段的召回（金标准有没有被检索到）.
METRIC_RETRIEVAL_RECALL = "retrieval_recall"

#: 检索名单那一段的精确率（端上来的盘子里有多少是用户要的）.
METRIC_RETRIEVAL_PRECISION = "retrieval_precision"

#: 检索名单那一段的 NDCG（命中几条 + 排得多靠前）.
METRIC_RETRIEVAL_NDCG = "retrieval_ndcg"

#: **进了提示词**那一段的召回——这一课新增的那个指标（见 types 的模块 docstring）.
METRIC_PACKED_RECALL = "packed_recall"

#: 进了提示词那一段的 NDCG（次序对不对，模型看到的是这个）。
METRIC_PACKED_NDCG = "packed_ndcg"

#: 接地率：答案里有可核对依据的比例（day069 的 ``GroundingReport.grounded``）。
METRIC_GROUNDED_RATE = "grounded_rate"

#: 平均覆盖率（有效引用 / 给出去的片段数）。
METRIC_COVERAGE = "mean_coverage"

#: 幻觉引用率：答案里出现"不存在的编号"的用例比例。
METRIC_HALLUCINATION_RATE = "hallucination_rate"

#: 模型真的被调用的比例（**描述性指标，不判方向**）。
METRIC_LLM_CALL_RATE = "llm_call_rate"

#: 走了回退的比例（六种原因各自还会单独计数）。
METRIC_FALLBACK_RATE = "fallback_rate"

#: 坏例率（归因判为坏例的用例比例）。
METRIC_BAD_CASE_RATE = "bad_case_rate"

#: 质量指标的**封闭清单**（基线与当前必须逐键一致，多一个少一个都报错）。
QUALITY_METRICS: tuple[str, ...] = (
    METRIC_RETRIEVAL_RECALL,
    METRIC_RETRIEVAL_PRECISION,
    METRIC_RETRIEVAL_NDCG,
    METRIC_PACKED_RECALL,
    METRIC_PACKED_NDCG,
    METRIC_GROUNDED_RATE,
    METRIC_COVERAGE,
    METRIC_HALLUCINATION_RATE,
    METRIC_LLM_CALL_RATE,
    METRIC_FALLBACK_RATE,
    METRIC_BAD_CASE_RATE,
)

#: 越高越好（质量类的正向指标）.
DIRECTION_HIGHER_BETTER = "higher_better"

#: 越低越好（失败率类的负向指标）.
DIRECTION_LOWER_BETTER = "lower_better"

#: 只做描述（进报告、进差值表，**不参与回归判定**）.
DIRECTION_NEUTRAL = "neutral"

#: 三种方向（封闭清单）.
DIRECTIONS: tuple[str, ...] = (
    DIRECTION_HIGHER_BETTER,
    DIRECTION_LOWER_BETTER,
    DIRECTION_NEUTRAL,
)

#: 每个指标的方向（三张表合一；见模块 docstring 的"方向反转"一节）.
METRIC_DIRECTIONS: dict[str, str] = {
    METRIC_RETRIEVAL_RECALL: DIRECTION_HIGHER_BETTER,
    METRIC_RETRIEVAL_PRECISION: DIRECTION_HIGHER_BETTER,
    METRIC_RETRIEVAL_NDCG: DIRECTION_HIGHER_BETTER,
    METRIC_PACKED_RECALL: DIRECTION_HIGHER_BETTER,
    METRIC_PACKED_NDCG: DIRECTION_HIGHER_BETTER,
    METRIC_GROUNDED_RATE: DIRECTION_HIGHER_BETTER,
    METRIC_COVERAGE: DIRECTION_HIGHER_BETTER,
    METRIC_HALLUCINATION_RATE: DIRECTION_LOWER_BETTER,
    METRIC_LLM_CALL_RATE: DIRECTION_NEUTRAL,
    METRIC_FALLBACK_RATE: DIRECTION_LOWER_BETTER,
    METRIC_BAD_CASE_RATE: DIRECTION_LOWER_BETTER,
}

#: 每个指标的缺省容忍度（**差值**口径：允许变差多少个百分点）.
#: 越贴近"用户能感知"的指标容忍度越小：接地率与幻觉率各 0.05，覆盖率 0.08，
#: 检索指标 0.05 —— 它们是"改一个旋钮就能动 5 个百分点"的量级。
DEFAULT_TOLERANCES: dict[str, float] = {
    METRIC_RETRIEVAL_RECALL: 0.05,
    METRIC_RETRIEVAL_PRECISION: 0.05,
    METRIC_RETRIEVAL_NDCG: 0.05,
    METRIC_PACKED_RECALL: 0.05,
    METRIC_PACKED_NDCG: 0.05,
    METRIC_GROUNDED_RATE: 0.05,
    METRIC_COVERAGE: 0.08,
    METRIC_HALLUCINATION_RATE: 0.05,
    METRIC_LLM_CALL_RATE: 0.20,
    METRIC_FALLBACK_RATE: 0.05,
    METRIC_BAD_CASE_RATE: 0.10,
}

#: 缺省死区（低于它的变化不算信号）.
DEFAULT_DEAD_BAND = 0.02

#: 缺省最小样本量（少于它时**拒绝给结论**）.
DEFAULT_MIN_SAMPLES = 3

#: 指标的中文说明（报告与 ``--explain`` 直接读它，避免两处各写一句）。
METRIC_DESCRIPTIONS: dict[str, str] = {
    METRIC_RETRIEVAL_RECALL: "检索名单的召回率（金标准有没有被检索到）",
    METRIC_RETRIEVAL_PRECISION: "检索名单的精确率（端上来的盘子里有多少是要的）",
    METRIC_RETRIEVAL_NDCG: "检索名单的 NDCG（命中几条 + 排得多靠前）",
    METRIC_PACKED_RECALL: "**进了提示词**那一段的召回率（预算丢尾之后的账）",
    METRIC_PACKED_NDCG: "**进了提示词**那一段的 NDCG（模型看到的次序）",
    METRIC_GROUNDED_RATE: "接地率（答案里有可核对依据的比例）",
    METRIC_COVERAGE: "平均覆盖率（有效引用 / 给出去的片段数）",
    METRIC_HALLUCINATION_RATE: "幻觉引用率（答案里出现不存在的编号）",
    METRIC_LLM_CALL_RATE: "模型真的被调用的比例（描述性，不判方向）",
    METRIC_FALLBACK_RATE: "走了回退的比例",
    METRIC_BAD_CASE_RATE: "坏例率（归因判为坏例的比例）",
}

# 三张表必须逐键对齐（封闭清单的三种投影）：少一个键时"这一条的方向"或
# "它的缺省容忍度"就无处可查，而那种缺失表现为一次静默的跳过。
if set(METRIC_DIRECTIONS) != set(QUALITY_METRICS) or set(DEFAULT_TOLERANCES) != set(
    QUALITY_METRICS
) or set(METRIC_DESCRIPTIONS) != set(QUALITY_METRICS):
    raise BaselineError(
        "质量指标的三张表不一致："
        f"directions={sorted(METRIC_DIRECTIONS)}、"
        f"tolerances={sorted(DEFAULT_TOLERANCES)}、"
        f"descriptions={sorted(METRIC_DESCRIPTIONS)}、metrics={sorted(QUALITY_METRICS)}。"
        "三者必须逐键对齐（同一份封闭清单的三种投影），否则某个指标会"
        "静默地不被判定——而「静默地不判」与「判了没问题」在报告里长得一样。"
    )
for _name, _direction in METRIC_DIRECTIONS.items():
    if _direction not in DIRECTIONS:
        raise BaselineError(
            f"指标 {_name!r} 的方向 {_direction!r} 不在 {list(DIRECTIONS)} 里："
            "方向是一张封闭清单（越高越好 / 越低越好 / 只做描述），"
            "自由文本会让回归判定失去依据。"
        )


# --------------------------------------------------------------------------- #
# 基线
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RagBaseline:
    """一次质量采集的存档：**指标表 + 两张分布表 + 条件 + 延迟**（day071）.

    ```text
    label            这次采集的名字（报告按它分组，例如 "baseline:baseline"）
    samples          评了几条用例（少于 min_samples 时守卫拒绝下结论）
    k                四个检索指标共用的那个 k（口径的一部分）
    prompt_version   这一批用的提示词版本（**可以不同**——那是 A/B 的自变量）
    index_version    这一批查的索引版本（**不同则不可比**：换了索引，历史作废）
    metrics          11 个质量指标的取值（封闭清单，逐键必须齐全）
    fallback_counts  六种回退各自的条数（只列真的出现过的）
    tag_counts       各类坏例的条数（只列真的出现过的）
    p95_latency_ms   端到端延迟的 p95（最近秩法，day046 的 percentile）
    mean_latency_ms  端到端延迟的均值
    ```

    ``index_version`` 与 ``prompt_version`` 的待遇刻意**不同**：

```text
prompt_version 不同  这是一次 A/B（换一版提示词之后可核对率变了吗）
                     → 允许对比，并在对比结果里记下"这一次的自变量是它"
index_version 不同   这批数字的起点变了（day065：换了编码器必须重建索引）
                     → 结论不可比，守卫拒绝判定（而不是报一个假的回归）
```
    """

    label: str
    samples: int
    k: int
    metrics: dict[str, float] = field(default_factory=dict)
    prompt_version: str = ""
    index_version: str = ""
    fallback_counts: dict[str, int] = field(default_factory=dict)
    tag_counts: dict[str, int] = field(default_factory=dict)
    p95_latency_ms: float = 0.0
    mean_latency_ms: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise BaselineError(
                f"RagBaseline.label 必须是非空字符串，收到 {self.label!r}："
                "报告按它分组（'这次是哪个变体'），空标签会让两次采集在报告里合并。"
            )
        if isinstance(self.samples, bool) or not isinstance(self.samples, int) or self.samples < 0:
            raise BaselineError(
                f"RagBaseline.samples 必须是非负整数，收到 {self.samples!r}"
            )
        if isinstance(self.k, bool) or not isinstance(self.k, int) or self.k < 1:
            raise BaselineError(
                f"RagBaseline.k 必须是 >= 1 的整数，收到 {self.k!r}："
                "k 是口径的一部分，两次采集的 k 不同就不可比。"
            )
        missing = sorted(set(QUALITY_METRICS) - set(self.metrics))
        extra = sorted(set(self.metrics) - set(QUALITY_METRICS))
        if missing or extra:
            raise BaselineError(
                f"RagBaseline.metrics 必须恰好是 QUALITY_METRICS 那 11 个键，"
                f"缺 {missing}、多 {extra}。"
                "指标表是一张**封闭清单**：多一个键意味着'这个数字从哪来'没有定义，"
                "少一个键意味着那个指标永远不参与回归判定（而「静默地不判」与"
                "'判了没问题'在报告里长得一样）。"
            )
        for name in QUALITY_METRICS:
            value = self.metrics[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise BaselineError(
                    f"RagBaseline.metrics[{name!r}] 必须是数字，收到 {type(value).__name__}"
                )
            number = float(value)
            if number != number or not 0.0 <= number <= 1.0:
                raise BaselineError(
                    f"RagBaseline.metrics[{name!r}]={value!r} 必须落在 [0, 1]："
                    "11 个质量指标全是比例（差值才因此可以直接读成'几个百分点'）。"
                )
        for label, table, allowed in (
            ("fallback_counts", self.fallback_counts, set(FALLBACK_REASON_DESCRIPTIONS)),
            ("tag_counts", self.tag_counts, set(BAD_CASE_TAGS)),
        ):
            if not isinstance(table, dict):
                raise BaselineError(
                    f"RagBaseline.{label} 必须是字典，收到 {type(table).__name__}"
                )
            unknown = sorted(set(table) - allowed)
            if unknown:
                raise BaselineError(
                    f"RagBaseline.{label} 里出现了清单之外的键 {unknown}："
                    f"合法取值是 {sorted(allowed)}。"
                    "两张分布表读的是同一份封闭清单（回退原因 / 坏例标签），"
                    "自由文本会让'这一类各有几条'无法统计。"
                )
            for key, count in table.items():
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise BaselineError(
                        f"RagBaseline.{label}[{key!r}] 必须是非负整数，收到 {count!r}"
                    )
        for name in ("p95_latency_ms", "mean_latency_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise BaselineError(
                    f"RagBaseline.{name} 必须是数字，收到 {type(value).__name__}"
                )
            number = float(value)
            if number != number or number < 0.0 or number == float("inf"):
                raise BaselineError(
                    f"RagBaseline.{name} 必须是非负有限数，收到 {value!r}"
                )

    # ------------------------------------------------------------------ 只读视图

    def value(self, name: str) -> float:
        """按名字取一个指标（名字必须是 ``QUALITY_METRICS`` 里的）."""
        if name not in self.metrics:
            raise BaselineError(
                f"不认识的指标名 {name!r}：可用的是 {list(QUALITY_METRICS)}。"
            )
        return float(self.metrics[name])

    def direction(self, name: str) -> str:
        """这个指标的方向（越高越好 / 越低越好 / 只做描述）."""
        if name not in METRIC_DIRECTIONS:
            raise BaselineError(f"不认识的指标名 {name!r}")
        return METRIC_DIRECTIONS[name]

    def total_bad_cases(self) -> int:
        """这次采集判出多少条坏例（``tag_counts`` 的和）."""
        return sum(self.tag_counts.values())

    def total_fallbacks(self) -> int:
        """这次采集走了多少次回退（``fallback_counts`` 里除空串之外的和）."""
        return sum(
            count
            for reason, count in self.fallback_counts.items()
            if reason != ""
        )

    # ------------------------------------------------------------------ 存档

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典（键序固定，便于 diff）."""
        return {
            "label": self.label,
            "samples": self.samples,
            "k": self.k,
            "prompt_version": self.prompt_version,
            "index_version": self.index_version,
            "metrics": {
                name: round(self.metrics[name], 6) for name in QUALITY_METRICS
            },
            "fallback_counts": {
                key: self.fallback_counts[key] for key in sorted(self.fallback_counts)
            },
            "tag_counts": {key: self.tag_counts[key] for key in sorted(self.tag_counts)},
            "p95_latency_ms": round(float(self.p95_latency_ms), 4),
            "mean_latency_ms": round(float(self.mean_latency_ms), 4),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RagBaseline:
        """从字典还原（``save`` 的逆操作，用于读取已提交的基线文件）."""
        return cls(
            label=str(data.get("label", "baseline")),
            samples=int(data.get("samples", 0)),
            k=int(data.get("k", 1)),
            metrics={
                str(name): float(value)
                for name, value in (data.get("metrics") or {}).items()
            },
            prompt_version=str(data.get("prompt_version", "")),
            index_version=str(data.get("index_version", "")),
            fallback_counts={
                str(key): int(value)
                for key, value in (data.get("fallback_counts") or {}).items()
            },
            tag_counts={
                str(key): int(value) for key, value in (data.get("tag_counts") or {}).items()
            },
            p95_latency_ms=float(data.get("p95_latency_ms", 0.0)),
            mean_latency_ms=float(data.get("mean_latency_ms", 0.0)),
        )

    def save(self, path: str | Path) -> Path:
        """写入 JSON（自动创建父目录），返回实际写入的路径.

        与 ``PerfBaseline.save`` 同一形状：基线随仓库提交，它就是这次迭代的
        "质量体检底片"——没有它，明天的所有数字都只是"看起来还行"。
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("质量基线已保存: %s（%d 条用例）", target, self.samples)
        return target

    @classmethod
    def load(cls, path: str | Path) -> RagBaseline:
        """读取基线文件；文件不存在时抛 ``FileNotFoundError``（不返回空基线）.

        刻意不兜底，理由与 ``PerfBaseline.load`` 逐字相同：**没有基线**与
        "基线全是零"是两回事，后者会静默地让所有回归判定变成"从 0 到 0"，
        那是最危险的失败方式。
        """
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"质量基线文件不存在: {target}")
        return cls.from_dict(json.loads(target.read_text(encoding="utf-8")))

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        return (
            f"基线 {self.label}（{self.samples} 条 / k={self.k}）"
            f" | 提示词 {self.prompt_version or '（未记录）'}"
            f" | 索引 {self.index_version or '（未记录）'}"
            f" | 检索召回 {self.value(METRIC_RETRIEVAL_RECALL):.4f}"
            f" | packed 召回 {self.value(METRIC_PACKED_RECALL):.4f}"
            f" | 接地率 {self.value(METRIC_GROUNDED_RATE):.4f}"
            f" | 幻觉率 {self.value(METRIC_HALLUCINATION_RATE):.4f}"
            f" | 坏例 {self.total_bad_cases()} 条"
            f" | p95 {self.p95_latency_ms:.1f}ms"
        )


# --------------------------------------------------------------------------- #
# 回归判定
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class QualityDelta:
    """一个指标的前后差（**差值**口径，不是比值）.

    ``verdict`` 三取一，而第三档 ``"unchanged"`` 的含义比"没变"更宽：
    它是"落在死区内或方向为 neutral"——两种情况在报告里都读作"这次不判它"。
    """

    metric: str
    baseline: float
    current: float
    delta: float
    direction: str
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return {
            "metric": self.metric,
            "direction": self.direction,
            "baseline": round(self.baseline, 6),
            "current": round(self.current, 6),
            "delta": round(self.delta, 6),
            "verdict": self.verdict,
            "description": METRIC_DESCRIPTIONS.get(self.metric, ""),
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要（差值用百分号读，因为它就是几个百分点）."""
        mark = {"improved": "↑", "degraded": "↓", "unchanged": "＝"}[self.verdict]
        return (
            f"{mark} {self.metric}: {self.baseline:.4f} → {self.current:.4f}"
            f"（{self.delta:+.4f}）[{self.direction}]"
        )


@dataclass(frozen=True)
class QualityRegression:
    """一条被判定为回归的指标（带方向与容忍度，便于回读"为什么判它"）."""

    metric: str
    baseline: float
    current: float
    delta: float
    tolerance: float
    message: str

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return {
            "metric": self.metric,
            "baseline": round(self.baseline, 6),
            "current": round(self.current, 6),
            "delta": round(self.delta, 6),
            "tolerance": self.tolerance,
            "message": self.message,
        }


@dataclass(frozen=True)
class BaselineComparison:
    """当前采集 vs 参考基线的对比结论（**可能"不下结论"**）.

    ``conclusive=False`` 的三种成因（它们都会写进 ``reasons``，一句一条）：

```text
样本不足            current.samples < min_samples（两条用例不足以宣告回归）
口径变了            k 不同（前 3 条与前 5 条不是同一件事）
条件变了            index_version 不同（换了索引，历史数字作废）
```

    "不下结论"**不等于"通过"**：``ok`` 的定义是"下了结论且没有回归"，
    因此样本不足时 ``ok`` 为 ``False`` 而 ``regressions`` 为空——
    读报告的人看到的是 ``inconclusive``，而不是一个会让人放松警惕的绿灯。
    这是本模块最重要的一条取舍：**门禁最危险的行为是在数据不足时装作有结论。**
    """

    baseline: RagBaseline
    current: RagBaseline
    deltas: tuple[QualityDelta, ...] = ()
    regressions: tuple[QualityRegression, ...] = ()
    reasons: tuple[str, ...] = ()
    variables: tuple[str, ...] = ()

    @property
    def conclusive(self) -> bool:
        """这次对比给出结论了吗（样本、口径、条件三关都过）."""
        return not self.reasons

    @property
    def ok(self) -> bool:
        """**下了结论且没有回归**才算通过（不结论时返回 ``False``）."""
        return self.conclusive and not self.regressions

    @property
    def improved(self) -> tuple[QualityDelta, ...]:
        """被判为改善的那些指标."""
        return tuple(item for item in self.deltas if item.verdict == "improved")

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return {
            "ok": self.ok,
            "conclusive": self.conclusive,
            "baseline_label": self.baseline.label,
            "current_label": self.current.label,
            "reasons": list(self.reasons),
            "variables": list(self.variables),
            "regressions": [item.to_dict() for item in self.regressions],
            "improvements": [item.metric for item in self.improved],
            "deltas": [item.to_dict() for item in self.deltas],
        }

    def summary_line(self) -> str:
        """一句话结论，供 CI 日志与演示脚本直接打印."""
        if not self.conclusive:
            return (
                f"不下结论（{self.current.samples} 条用例）："
                + "; ".join(self.reasons)
            )
        if self.ok:
            changed = "、".join(item.metric for item in self.improved) or "无明显变化"
            return (
                f"质量未回归：{self.current.samples} 条用例，改善指标 {changed}；"
                f"接地率 {self.baseline.value(METRIC_GROUNDED_RATE):.4f} → "
                f"{self.current.value(METRIC_GROUNDED_RATE):.4f}"
            )
        return f"质量回归 {len(self.regressions)} 项：" + "; ".join(
            item.message for item in self.regressions
        )


class RagBaselineGuard:
    """质量门禁：用**方向表 + 死区 + 容忍度**把"抖动"与"真回归"分开.

    判定规则（对每个方向不为 ``neutral`` 的指标，``delta = current - baseline``）：

```text
|delta| <= dead_band                    → unchanged（噪声，不判）
方向是 neutral                          → unchanged（描述性指标，永不判）
变化方向是"好"（higher_better 时 delta > 0）→ improved
变化方向是"差"且 |delta| > 容忍度        → degraded
变化方向是"差"但 |delta| <= 容忍度       → unchanged（**变差但在容忍度内**）
```

    最后一行是这套规则里最容易写错的一处：容忍度的语义是**"允许变差多少"**，
    因此它只对"变差"生效；把"差得不多"记成 improved 会让一次轻微劣化
    在报告里显示为改善，而符号反了的报告比没有报告更坏。

    三处与 ``PerformanceGuard`` 的刻意差别：**差值而非比值**（见模块 docstring）、
    **方向表而非单一方向**、**三档判定而非两档**（``unchanged`` 显式存在，
    因此报告里能区分"变了但没用"与"没变"）。
    """

    def __init__(
        self,
        baseline: RagBaseline,
        *,
        tolerances: dict[str, float] | None = None,
        dead_band: float = DEFAULT_DEAD_BAND,
        min_samples: int = DEFAULT_MIN_SAMPLES,
    ) -> None:
        if not isinstance(baseline, RagBaseline):
            raise BaselineError(
                f"baseline 必须是 RagBaseline，收到 {type(baseline).__name__}"
            )
        table = dict(DEFAULT_TOLERANCES) if tolerances is None else dict(tolerances)
        unknown = sorted(set(table) - set(QUALITY_METRICS))
        if unknown:
            raise BaselineError(
                f"tolerances 里出现了清单之外的指标 {unknown}："
                f"合法取值是 {list(QUALITY_METRICS)}。"
                "多出来的键永远不会被读到——而'我明明调了它的容忍度'这句话"
                "在报告里没有任何痕迹。"
            )
        for name, value in table.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) < 0.0:
                raise BaselineError(
                    f"tolerances[{name!r}]={value!r} 必须是非负数字（单位是百分点）"
                )
        if (
            isinstance(dead_band, bool)
            or not isinstance(dead_band, (int, float))
            or float(dead_band) < 0.0
        ):
            raise BaselineError(
                f"dead_band={dead_band!r} 必须是非负数字："
                "它是「多大的变化算噪声」的那条线（缺省 0.02）。"
            )
        if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 1:
            raise BaselineError(
                f"min_samples 必须是 >= 1 的整数，收到 {min_samples!r}："
                "它是「几条用例以下拒绝下结论」的那条线。"
            )
        self.baseline = baseline
        self.tolerances = {name: float(table[name]) for name in table}
        self.dead_band = float(dead_band)
        self.min_samples = min_samples

    def tolerance(self, metric: str) -> float:
        """某个指标的容忍度（没显式给过就用缺省表里的值）."""
        if metric not in QUALITY_METRICS:
            raise BaselineError(f"不认识的指标名 {metric!r}")
        return self.tolerances.get(metric, DEFAULT_TOLERANCES[metric])

    def compare(self, current: RagBaseline) -> BaselineComparison:
        """逐指标比对，并先做三道"能不能比"的检查（不通过就不给结论）.

        顺序刻意是"先查能不能比、再比"：把一个不可比的对比算出一堆差值，
        比不出一份报告更危险——那些差值看起来完全合理。
        """
        if not isinstance(current, RagBaseline):
            raise BaselineError(
                f"current 必须是 RagBaseline，收到 {type(current).__name__}"
            )
        reasons: list[str] = []
        if current.k != self.baseline.k:
            reasons.append(
                f"口径变了：k 从 {self.baseline.k} 变成 {current.k}"
                "（前 3 条与前 5 条不是同一件事，两边的指标不可比）"
            )
        if current.index_version != self.baseline.index_version:
            reasons.append(
                f"条件变了：索引版本从 {self.baseline.index_version or '（未记录）'} "
                f"变成 {current.index_version or '（未记录）'}"
                "（换了索引/编码器之后，历史数字立刻作废——需要新开一个基线）"
            )
        if current.samples < self.min_samples or self.baseline.samples < self.min_samples:
            reasons.append(
                f"样本不足：当前 {current.samples} 条、基线 {self.baseline.samples} 条，"
                f"低于下限 {self.min_samples}（两条用例不足以宣告回归）"
            )
        variables: list[str] = []
        if current.prompt_version != self.baseline.prompt_version:
            variables.append(
                f"提示词版本 {self.baseline.prompt_version or '（未记录）'} → "
                f"{current.prompt_version or '（未记录）'}"
            )

        deltas = deltas_between(
            self.baseline,
            current,
            tolerances=self.tolerances,
            dead_band=self.dead_band,
        )
        if reasons:
            comparison = BaselineComparison(
                baseline=self.baseline,
                current=current,
                deltas=deltas,
                regressions=(),
                reasons=tuple(reasons),
                variables=tuple(variables),
            )
            logger.warning("质量对比不下结论: %s", comparison.summary_line())
            return comparison

        regressions: list[QualityRegression] = []
        for item in deltas:
            if item.verdict != "degraded":
                continue
            tolerance = self.tolerance(item.metric)
            regressions.append(
                QualityRegression(
                    metric=item.metric,
                    baseline=item.baseline,
                    current=item.current,
                    delta=item.delta,
                    tolerance=tolerance,
                    message=(
                        f"{item.metric} {item.baseline:.4f} → {item.current:.4f}"
                        f"（{item.delta:+.4f}，超出容忍度 {tolerance}）"
                    ),
                )
            )
        comparison = BaselineComparison(
            baseline=self.baseline,
            current=current,
            deltas=deltas,
            regressions=tuple(regressions),
            reasons=(),
            variables=tuple(variables),
        )
        if comparison.ok:
            logger.info("质量对比通过: %s", comparison.summary_line())
        else:
            logger.warning("质量对比发现回归: %s", comparison.summary_line())
        return comparison

    def _delta(self, metric: str, current: RagBaseline) -> QualityDelta:
        """算一个指标的差与它的判定（**单指标入口，便于逐条复核**）."""
        return deltas_between(
            self.baseline,
            current,
            tolerances={metric: self.tolerance(metric)},
            dead_band=self.dead_band,
        )[QUALITY_METRICS.index(metric)]


def deltas_between(
    baseline: RagBaseline,
    current: RagBaseline,
    *,
    tolerances: dict[str, float] | None = None,
    dead_band: float = DEFAULT_DEAD_BAND,
) -> tuple[QualityDelta, ...]:
    """逐指标算差与判定（**判定只有一份实现**：守卫与 A/B 对比都读它）.

    为什么把它抽成模块级函数：``RagBaselineGuard.compare`` 要它，
    ``report.compare_variants`` 也要它（A/B 实验逐指标看差值）。两处各写一遍
    一定会分家，而分家的表现是"同一个指标在回归报告里是 degraded、
    在 A/B 报告里是 improved"——两个互相矛盾的读数比一个错读数更坏。
    """
    if not isinstance(baseline, RagBaseline) or not isinstance(current, RagBaseline):
        raise BaselineError("deltas_between 的两个入参都必须是 RagBaseline")
    table = dict(DEFAULT_TOLERANCES)
    if tolerances is not None:
        unknown = sorted(set(tolerances) - set(QUALITY_METRICS))
        if unknown:
            raise BaselineError(
                f"tolerances 里出现了清单之外的指标 {unknown}："
                f"合法取值是 {list(QUALITY_METRICS)}。"
            )
        table.update({name: float(value) for name, value in tolerances.items()})
    if (
        isinstance(dead_band, bool)
        or not isinstance(dead_band, (int, float))
        or float(dead_band) < 0.0
    ):
        raise BaselineError(f"dead_band={dead_band!r} 必须是非负数字")
    results: list[QualityDelta] = []
    for metric in QUALITY_METRICS:
        base_value = baseline.value(metric)
        current_value = current.value(metric)
        delta = round(current_value - base_value, 6)
        direction = METRIC_DIRECTIONS[metric]
        verdict = "unchanged"
        # 三档判定的完整规则（**容忍度只对"变差"生效**）：
        #   落在死区内            → unchanged（低于噪声线，不判）
        #   方向是 neutral        → unchanged（描述性指标，永不判）
        #   变化方向是"好"         → improved
        #   变化方向是"差"         → 超出容忍度才 degraded，否则仍然 unchanged
        # 第三行那个 else 很容易写错：把"差得不多"记成 improved，
        # 会让一次轻微劣化在报告里显示为改善——**符号反了的报告比没有报告更坏**。
        if direction != DIRECTION_NEUTRAL and abs(delta) > float(dead_band):
            tolerance = float(table[metric])
            if direction == DIRECTION_HIGHER_BETTER:
                better = delta > 0.0
            else:
                better = delta < 0.0
            if better:
                verdict = "improved"
            elif abs(delta) > tolerance:
                verdict = "degraded"
        results.append(
            QualityDelta(
                metric=metric,
                baseline=base_value,
                current=current_value,
                delta=delta,
                direction=direction,
                verdict=verdict,
            )
        )
    return tuple(results)


def guard_from_settings(baseline: RagBaseline) -> RagBaselineGuard:
    """按 ``settings`` 的 ``rag_eval_*`` 一组构建门禁（阈值外置，调参不改代码）."""
    from smart_research_agent.config import settings

    tolerances = {
        METRIC_RETRIEVAL_RECALL: settings.rag_eval_recall_tolerance,
        METRIC_GROUNDED_RATE: settings.rag_eval_grounded_tolerance,
        METRIC_COVERAGE: settings.rag_eval_coverage_tolerance,
        METRIC_HALLUCINATION_RATE: settings.rag_eval_hallucination_tolerance,
    }
    return RagBaselineGuard(
        baseline,
        tolerances=tolerances,
        dead_band=settings.rag_eval_dead_band,
        min_samples=settings.rag_eval_min_samples,
    )


def compare_with_file(
    current: RagBaseline, baseline_path: str | Path
) -> BaselineComparison:
    """读取基线文件并与当前采集对比（CI 与演示脚本的一行入口）."""
    baseline = RagBaseline.load(baseline_path)
    return guard_from_settings(baseline).compare(current)


__all__ = [
    "DEFAULT_DEAD_BAND",
    "DEFAULT_MIN_SAMPLES",
    "DEFAULT_TOLERANCES",
    "DIRECTIONS",
    "DIRECTION_HIGHER_BETTER",
    "DIRECTION_LOWER_BETTER",
    "DIRECTION_NEUTRAL",
    "METRIC_BAD_CASE_RATE",
    "METRIC_COVERAGE",
    "METRIC_DESCRIPTIONS",
    "METRIC_DIRECTIONS",
    "METRIC_FALLBACK_RATE",
    "METRIC_GROUNDED_RATE",
    "METRIC_HALLUCINATION_RATE",
    "METRIC_LLM_CALL_RATE",
    "METRIC_PACKED_NDCG",
    "METRIC_PACKED_RECALL",
    "METRIC_RETRIEVAL_NDCG",
    "METRIC_RETRIEVAL_PRECISION",
    "METRIC_RETRIEVAL_RECALL",
    "QUALITY_METRICS",
    "BaselineComparison",
    "QualityDelta",
    "QualityRegression",
    "RagBaseline",
    "RagBaselineGuard",
    "compare_with_file",
    "deltas_between",
    "guard_from_settings",
]
