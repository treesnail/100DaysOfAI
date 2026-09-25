"""监控采样：把一次同步与一份评估折成 11 个可上报的数字（M6-D10 / day072）.

day032 的 ``observability`` 已经能回答"这一次请求花了多少钱"，
day071 的 ``rag_debug`` 已经能回答"这一批答案好不好"。本模块要回答的是
运维最关心的那个问题——**"这个知识库服务现在是什么状态"**，
并且它必须能在**没有模型、没有网络、没有评估**的情况下被回答：

```text
语料侧   sources_total / sources_changed      （有几份、变了几份）
构建侧   chunks_written / sync_seconds        （写了多少块、花了多久）
状态侧   sync_age_minutes / index_records / run_failures
质量侧   quality_*（4 个）                     ← 从 day071 的 RagBaseline 抄过来
```

## 为什么质量指标是"抄"，不是"算"

``samples_from_quality`` 只做一件事：把 ``RagBaseline.metrics`` 里的四个键
改名搬进监控指标表。它**不重算**任何东西——重算一次就会出现两个质量读数
（评估报告里的与监控里的），而两个互相矛盾的读数比一个错读数更坏
（day071 的"归因只读账"是同一条理由）。

代价要说清楚：**没有评估就没有质量采样**。此时那四个指标**不出现在样本里**，
而不是以 0 出现——0 会被读成"幻觉率 0，很好"，而真相是"没人量过"。

## 采样为什么带 ``at`` 且必须是有序的

监控系统的第一条性质是"时间是横轴"。因此每个采样自带时刻，
``MetricSeries`` 在构造期校验同一指标名与时间非降序：
混了两个指标的序列会让均值变成两个东西的平均，
时间倒序会让"最新值"变成"最后一个元素"。两条都不报错，只让图表说谎。
"""

from __future__ import annotations

from datetime import datetime

from smart_research_agent.indexing.types import IndexingReport
from smart_research_agent.rag_debug.baseline import (
    METRIC_BAD_CASE_RATE,
    METRIC_GROUNDED_RATE,
    METRIC_HALLUCINATION_RATE,
    METRIC_RETRIEVAL_RECALL,
    QUALITY_METRICS,
    RagBaseline,
)
from smart_research_agent.rag_ops.errors import MetricsError
from smart_research_agent.rag_ops.types import (
    METRIC_CHUNKS_WRITTEN,
    METRIC_INDEX_RECORDS,
    METRIC_QUALITY_BAD_CASE,
    METRIC_QUALITY_GROUNDED,
    METRIC_QUALITY_HALLUCINATION,
    METRIC_QUALITY_RECALL,
    METRIC_RUN_FAILURES,
    METRIC_SOURCES_CHANGED,
    METRIC_SOURCES_TOTAL,
    METRIC_SYNC_AGE_MINUTES,
    METRIC_SYNC_SECONDS,
    OPS_METRIC_DESCRIPTIONS,
    OPS_METRIC_UNITS,
    OPS_METRICS,
    MetricSample,
    MetricSeries,
    SyncLedger,
    SyncPlan,
    to_iso,
    utc_now,
)
from smart_research_agent.vectorstore.base import VectorBackend

#: 从 day071 的 11 个质量指标搬进监控的那四个（**只抄不改名**：
#: 监控里叫 ``quality_retrieval_recall``，评估报告里叫 ``retrieval_recall``，
#: 两者的对应关系就在这张表里，而不是散在代码的字符串里）.
QUALITY_SAMPLE_SOURCES: dict[str, str] = {
    METRIC_QUALITY_RECALL: METRIC_RETRIEVAL_RECALL,
    METRIC_QUALITY_GROUNDED: METRIC_GROUNDED_RATE,
    METRIC_QUALITY_HALLUCINATION: METRIC_HALLUCINATION_RATE,
    METRIC_QUALITY_BAD_CASE: METRIC_BAD_CASE_RATE,
}

if not set(QUALITY_SAMPLE_SOURCES.values()) <= set(QUALITY_METRICS):
    raise MetricsError(
        "质量采样的来源表里出现了不在 day071 指标集里的名字："
        f"{sorted(set(QUALITY_SAMPLE_SOURCES.values()) - set(QUALITY_METRICS))}——"
        "监控要抄的是评估**真实产出**的那些指标，抄错名字只会得到一片 0。"
    )
if len(set(QUALITY_SAMPLE_SOURCES.values())) != len(QUALITY_SAMPLE_SOURCES):
    raise MetricsError("质量采样的来源表里有重复的源指标名：四个监控指标必须各对一项。")


def _moment(at: datetime | str | None) -> str:
    """把"采样时刻"统一成 ISO 串（``None`` 表示现在）."""
    if at is None:
        return to_iso(utc_now())
    if isinstance(at, datetime):
        return to_iso(at)
    return str(at)


def samples_from_plan(
    plan: SyncPlan,
    *,
    at: datetime | str | None = None,
    index_report: IndexingReport | None = None,
    seconds: float = 0.0,
) -> tuple[MetricSample, ...]:
    """语料侧与构建侧的四个采样（**文件级的账 + 块级的账**）.

    ``sources_changed`` 与 ``chunks_written`` 必须**同时**上报，这是本模块
    最想让人看见的一对数字：前者回答"这次改了几份文件"，
    后者回答"这次写了几块"。一次"只改了一份大文档"会让前者是 1、
    后者是十几甚至几十——只看其中一个都会对这次改动规模的判断失准。
    """
    stamp = _moment(at)
    if seconds < 0:
        raise MetricsError(f"sync_seconds 不能为负：{seconds}。")
    written = int(index_report.written) if index_report is not None else 0
    return (
        MetricSample(name=METRIC_SOURCES_TOTAL, value=len(plan.current_sources), at=stamp),
        MetricSample(name=METRIC_SOURCES_CHANGED, value=plan.changed_count, at=stamp),
        MetricSample(name=METRIC_CHUNKS_WRITTEN, value=written, at=stamp),
        MetricSample(name=METRIC_SYNC_SECONDS, value=round(float(seconds), 4), at=stamp),
    )


def samples_from_store(
    store: VectorBackend,
    *,
    at: datetime | str | None = None,
) -> tuple[MetricSample, ...]:
    """库侧的一个采样：库里有几条记录（= 可被检索的块数）."""
    return (
        MetricSample(
            name=METRIC_INDEX_RECORDS,
            value=int(store.count()),
            at=_moment(at),
            labels=(("backend", store.name), ("metric", store.metric)),
        ),
    )


def samples_from_ledger(
    ledger: SyncLedger,
    *,
    now: datetime | None = None,
    at: datetime | str | None = None,
) -> tuple[MetricSample, ...]:
    """状态侧的两个采样：多久没成功同步了、连续失败几次.

    "从未成功同步"时 ``sync_age_minutes`` 报 **0**，这不是在说"很新鲜"——
    它只是"算不出年龄"。真正的判定在 ``health.check_corpus_freshness``
    （那一项会给出 ``warn``），而这里只有数字。
    这条分工要写清楚：**采样不许替判据做决定，判据不许编数字。**
    """
    moment = now if now is not None else utc_now()
    age = ledger.age_minutes(moment)
    return (
        MetricSample(
            name=METRIC_SYNC_AGE_MINUTES,
            value=0.0 if age is None else round(float(age), 4),
            at=_moment(at),
            labels=(("has_watermark", "true" if ledger.has_watermark else "false"),),
        ),
        MetricSample(
            name=METRIC_RUN_FAILURES,
            value=int(ledger.consecutive_failures),
            at=_moment(at),
        ),
    )


def samples_from_quality(
    quality: RagBaseline | None,
    *,
    at: datetime | str | None = None,
) -> tuple[MetricSample, ...]:
    """质量侧的四个采样（``None`` 时**返回空**，不是四个 0）.

    返回空而不是全 0，是这一课最容易被"顺手补全"的一处：一份没有评估过的
    部署如果上报 ``hallucination_rate = 0``，看板会显示"幻觉率 0"，
    而这句结论没有任何一次测量支撑它。
    """
    if quality is None:
        return ()
    stamp = _moment(at)
    samples: list[MetricSample] = []
    for name, source in QUALITY_SAMPLE_SOURCES.items():
        if source not in quality.metrics:
            raise MetricsError(
                f"评估基线里缺少指标 {source!r}（监控要抄成 {name}）："
                "缺一项时**不能**用 0 顶上——'没量过'与'量出来是 0'是两回事。"
            )
        samples.append(
            MetricSample(
                name=name,
                value=float(quality.metrics[source]),
                at=stamp,
                labels=(("label", quality.label), ("samples", str(quality.samples))),
            )
        )
    return tuple(samples)


def collect_samples(
    *,
    at: datetime | str | None = None,
    now: datetime | None = None,
    plan: SyncPlan | None = None,
    index_report: IndexingReport | None = None,
    seconds: float = 0.0,
    store: VectorBackend | None = None,
    ledger: SyncLedger | None = None,
    quality: RagBaseline | None = None,
) -> tuple[MetricSample, ...]:
    """一次运行的全部采样（**顺序 = 语料 → 构建 → 状态 → 质量**）.

    每个可选的输入缺席时，对应那几个指标就**不出现在样本里**：
    这份函数的产物是"这次运行真的量到了什么"，而不是"一张 11 格的表"。
    """
    samples: list[MetricSample] = []
    if plan is not None:
        samples.extend(
            samples_from_plan(plan, at=at, index_report=index_report, seconds=seconds)
        )
    if store is not None:
        samples.extend(samples_from_store(store, at=at))
    if ledger is not None:
        samples.extend(samples_from_ledger(ledger, now=now, at=at))
    samples.extend(samples_from_quality(quality, at=at))
    return tuple(samples)


class MetricsBuffer:
    """一份内存监控缓冲：按指标名分组、可查最新值与汇总表.

    它刻意**不落盘**（落盘的是报告与账本）：监控数据的归宿是监控系统
    （Prometheus / 看板），而不是这个项目的数据目录。写进 ``data/`` 的
    后果是"两个真相源"，而它们迟早会分家。
    """

    def __init__(self, samples: tuple[MetricSample, ...] = ()) -> None:
        self._samples: list[MetricSample] = []
        self.extend(samples)

    def __len__(self) -> int:
        """采样总数."""
        return len(self._samples)

    def record(self, sample: MetricSample) -> MetricSample:
        """记一个采样，返回它（便于链式调用）."""
        if not isinstance(sample, MetricSample):
            raise MetricsError(
                f"只能记 MetricSample，收到 {type(sample).__name__}："
                "形状不对的采样进不了序列校验（指标名、单位、时间顺序都会失守）。"
            )
        self._samples.append(sample)
        return sample

    def extend(self, samples: tuple[MetricSample, ...] | list[MetricSample]) -> int:
        """批量记采样，返回本次记入的条数."""
        count = 0
        for sample in samples:
            self.record(sample)
            count += 1
        return count

    def names(self) -> tuple[str, ...]:
        """出现过的指标名（按 ``OPS_METRICS`` 的顺序，而不是插入顺序）."""
        present = {sample.name for sample in self._samples}
        return tuple(name for name in OPS_METRICS if name in present)

    def samples(self, name: str) -> tuple[MetricSample, ...]:
        """取某个指标的全部采样（按插入顺序）."""
        if name not in OPS_METRICS:
            raise MetricsError(f"不认识的指标名 {name!r}：可用取值 {list(OPS_METRICS)}。")
        return tuple(sample for sample in self._samples if sample.name == name)

    def series(self, name: str) -> MetricSeries:
        """把某个指标折成一条序列（时间必须非降序，否则当场报错）."""
        return MetricSeries(name=name, samples=self.samples(name))

    def latest(self, name: str) -> MetricSample | None:
        """某个指标的最新采样；没有时返回 ``None``."""
        series = self.series(name)
        return series.latest

    def describe(self) -> list[dict[str, object]]:
        """汇总表：每个出现过一次的指标一行（指标名 + 单位 + 方向 + 最新值 + 计数）.

        只列**出现过**的指标，而不是把 11 个全列出来：一行 "quality_recall: 无"
        与一行 "quality_recall: 0.0" 在后端看板上长得不一样，
        但在这一层的语义里前者才是实话。
        """
        rows: list[dict[str, object]] = []
        for name in self.names():
            series = self.series(name)
            summary = series.summary()
            latest = series.latest
            rows.append(
                {
                    "name": name,
                    "unit": OPS_METRIC_UNITS[name],
                    "description": OPS_METRIC_DESCRIPTIONS[name],
                    "direction": latest.direction if latest is not None else "neutral",
                    "count": summary["count"],
                    "min": summary["min"],
                    "max": summary["max"],
                    "mean": summary["mean"],
                    "latest": summary["latest"],
                    "latest_at": summary.get("latest_at", ""),
                }
            )
        return rows

    def to_dict(self) -> dict[str, object]:
        """可 json.dumps 的形状（汇总 + 逐指标序列）."""
        return {
            "count": len(self._samples),
            "metrics": self.names(),
            "rows": self.describe(),
            "series": {name: self.series(name).to_dict() for name in self.names()},
        }

    def clear(self) -> int:
        """清空缓冲，返回清掉的条数（**只清这一层的副本，不碰监控系统**）."""
        count = len(self._samples)
        self._samples = []
        return count


__all__ = [
    "QUALITY_SAMPLE_SOURCES",
    "MetricsBuffer",
    "collect_samples",
    "samples_from_ledger",
    "samples_from_plan",
    "samples_from_quality",
    "samples_from_store",
]
