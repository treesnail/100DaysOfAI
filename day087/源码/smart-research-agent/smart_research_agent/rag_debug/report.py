"""报告：把一次评估折成"十一个数字 + 一张坏例表 + 一份动作清单"（day071）.

一份评估的结果只有三种读者，而本模块的三节正好对应他们：

```text
① 十一个质量指标（进基线的部分）       → CI / 回归门禁："这次比上次差吗"
② 逐条证据表（per_case）              → 排查的人："这一条为什么被判坏"
③ 坏例表 + 动作清单（bad_cases / actions）→ 要动手的人："下一步改哪个旋钮"
```

三节都在同一份报告里，**不能只留一节**：只留 ① 的门禁无法定位病灶；
只留 ② 的报告读到第三十条就没人看了；只留 ③ 的报告无法进 CI。

## 报告必须自述结论的适用范围

``RagEvalReport.limitations`` 直接搬 ``retrieval.RETRIEVAL_LIMITATIONS``
（day066 起就写在那里的那段自述）：**一份报告不说清"它的结论在什么范围内
成立"，就等于让人把一次实验室里的测量当成生产结论**。day069 的
``GroundingReport`` 只做编号层面核对、拒答检测靠固定句式——这两条边界
会跟着每一个"接地率"一起被打印出来。

## A/B：换一版提示词之后可核对率变了吗

``compare_variants`` 是 day069 留下的那处伏笔的兑现：两份基线（同一索引、
同一批用例、只差提示词版本）逐指标看差值。它与 ``RagBaselineGuard``
共用同一份判定实现（``baseline.deltas_between``），但**不判回归**——
A/B 的结论是"哪个变体更好"，而回归门禁的结论是"这次能不能发"，
两个问题共用一个函数会让"实验"与"发布"的语义混在一起。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smart_research_agent.evaluation.perf_baseline import percentile
from smart_research_agent.rag_debug.baseline import (
    DEFAULT_DEAD_BAND,
    METRIC_BAD_CASE_RATE,
    METRIC_COVERAGE,
    METRIC_DESCRIPTIONS,
    METRIC_DIRECTIONS,
    METRIC_FALLBACK_RATE,
    METRIC_GROUNDED_RATE,
    METRIC_HALLUCINATION_RATE,
    METRIC_LLM_CALL_RATE,
    METRIC_PACKED_NDCG,
    METRIC_PACKED_RECALL,
    METRIC_RETRIEVAL_NDCG,
    METRIC_RETRIEVAL_PRECISION,
    METRIC_RETRIEVAL_RECALL,
    QUALITY_METRICS,
    QualityDelta,
    RagBaseline,
    deltas_between,
)
from smart_research_agent.rag_debug.diagnose import action_plan, tag_counts
from smart_research_agent.rag_debug.errors import RagDebugError
from smart_research_agent.rag_debug.types import (
    BAD_CASE_DESCRIPTIONS,
    BadCase,
    CaseOutcome,
    bad_case_rate,
    mean_of,
)
from smart_research_agent.retrieval.generation import FALLBACK_REASON_DESCRIPTIONS
from smart_research_agent.retrieval.types import RETRIEVAL_LIMITATIONS


@dataclass(frozen=True)
class RagEvalReport:
    """一次端到端评估的完整交代（指标 + 分布 + 逐条证据 + 坏例 + 动作）.

    ```text
    label           这次评估的名字（变体名进报告，便于两次实验互相区分）
    k               四个检索指标共用的那个 k（口径的一部分）
    prompt_version  这一批用的提示词版本（A/B 的自变量，进基线）
    index_version   这一批查的索引版本（条件变了就不可比，进基线）
    cases           评了几条用例（**0 条也如实写出来**）
    metrics         11 个质量指标（封闭清单，进基线的就是它）
    fallback_counts 六种回退各自的条数
    tag_counts      各类坏例的条数
    per_case        逐条证据（CaseOutcome.to_dict() 的列表）
    bad_cases       坏例表（只含被归因的那些）
    actions         去重后的动作清单（"下一步做什么"）
    p95/mean_latency_ms  端到端延迟的两个数（最近秩法 p95）
    limitations     这份报告的适用范围（搬 RETRIEVAL_LIMITATIONS）
    ```

    ``cases == 0`` 与"指标全是 0"必须一起被看见：一份"评了 0 条"的报告
    在字节上与"全部失败"的报告很像（都是 0），而 ``cases`` 是唯一能把
    两者分开的字段——这正是 day025 起就立下的那条纪律。
    """

    label: str
    k: int
    cases: int
    metrics: dict[str, float]
    per_case: tuple[dict[str, Any], ...] = ()
    bad_cases: tuple[BadCase, ...] = ()
    actions: tuple[str, ...] = ()
    fallback_counts: dict[str, int] | None = None
    tag_counts: dict[str, int] | None = None
    prompt_version: str = ""
    index_version: str = ""
    p95_latency_ms: float = 0.0
    mean_latency_ms: float = 0.0
    limitations: tuple[str, ...] = RETRIEVAL_LIMITATIONS

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise RagDebugError(
                f"RagEvalReport.label 必须是非空字符串，收到 {self.label!r}"
            )
        if isinstance(self.k, bool) or not isinstance(self.k, int) or self.k < 1:
            raise RagDebugError(f"RagEvalReport.k 必须是 >= 1 的整数，收到 {self.k!r}")
        if isinstance(self.cases, bool) or not isinstance(self.cases, int) or self.cases < 0:
            raise RagDebugError(
                f"RagEvalReport.cases 必须是非负整数，收到 {self.cases!r}"
            )
        missing = sorted(set(QUALITY_METRICS) - set(self.metrics))
        extra = sorted(set(self.metrics) - set(QUALITY_METRICS))
        if missing or extra:
            raise RagDebugError(
                f"RagEvalReport.metrics 必须恰好是 QUALITY_METRICS 那 11 个键，"
                f"缺 {missing}、多 {extra}（报告是基线的来源，键不齐就没法进基线）。"
            )
        if len(self.bad_cases) > self.cases:
            raise RagDebugError(
                f"RagEvalReport 的账不自洽：坏例 {len(self.bad_cases)} 条 > "
                f"用例 {self.cases} 条（每条用例最多归因一次）。"
            )
        if not isinstance(self.per_case, tuple):
            raise RagDebugError(
                f"RagEvalReport.per_case 必须是 tuple，收到 {type(self.per_case).__name__}"
            )
        for name, table in (
            ("fallback_counts", self.fallback_counts),
            ("tag_counts", self.tag_counts),
        ):
            if table is None:
                continue
            allowed = (
                set(FALLBACK_REASON_DESCRIPTIONS)
                if name == "fallback_counts"
                else set(BAD_CASE_DESCRIPTIONS)
            )
            unknown = sorted(set(table) - allowed)
            if unknown:
                raise RagDebugError(
                    f"RagEvalReport.{name} 里出现了清单之外的键 {unknown}"
                )

    # ------------------------------------------------------------------ 只读视图

    @property
    def bad_case_count(self) -> int:
        """判为坏例的条数."""
        return len(self.bad_cases)

    @property
    def clean_rate(self) -> float:
        """"这一批里有多少条没被判坏"（坏例率的反面，报告里两个都印）."""
        if self.cases == 0:
            return 0.0
        return round(1.0 - bad_case_rate(self.bad_case_count, self.cases), 4)

    def value(self, name: str) -> float:
        """按名字取一个指标."""
        if name not in self.metrics:
            raise RagDebugError(f"不认识的指标名 {name!r}")
        return float(self.metrics[name])

    def to_baseline(self) -> RagBaseline:
        """把这份报告折成一份基线（**进基线的就是这 11 个数字与两张分布表**）.

        ``per_case`` 与 ``bad_cases`` 刻意**不进基线**：基线要被提交进仓库、
        要被 diff、要被明天读一遍，而一份 40 条用例的逐条证据表会让它变成
        一份没人读的档案。要看坏例请读那一天的 ``outputs/`` 报告文件——
        基线与报告的分工，和 day046 的"基线 vs 性能采集明细"完全一致。
        """
        return RagBaseline(
            label=self.label,
            samples=self.cases,
            k=self.k,
            metrics={name: self.metrics[name] for name in QUALITY_METRICS},
            prompt_version=self.prompt_version,
            index_version=self.index_version,
            fallback_counts=dict(self.fallback_counts or {}),
            tag_counts=dict(self.tag_counts or {}),
            p95_latency_ms=self.p95_latency_ms,
            mean_latency_ms=self.mean_latency_ms,
        )

    # ------------------------------------------------------------------ 投影

    def to_dict(self, *, include_per_case: bool = True) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典.

        ``include_per_case=True`` 是默认值（与 ``RetrievalResult.to_dict`` 相反）：
        这份报告的主要消费者是**排查一次评估**，而逐条证据正是他要的东西。
        只要那一小段摘要（比如写进 CI 输出）请显式传 ``False``。
        """
        return {
            "label": self.label,
            "k": self.k,
            "cases": self.cases,
            "prompt_version": self.prompt_version,
            "index_version": self.index_version,
            "metrics": {name: round(self.metrics[name], 6) for name in QUALITY_METRICS},
            "bad_case_count": self.bad_case_count,
            "clean_rate": self.clean_rate,
            "fallback_counts": {
                key: value for key, value in sorted((self.fallback_counts or {}).items())
            },
            "tag_counts": {
                key: value for key, value in sorted((self.tag_counts or {}).items())
            },
            "p95_latency_ms": round(float(self.p95_latency_ms), 4),
            "mean_latency_ms": round(float(self.mean_latency_ms), 4),
            "bad_cases": [case.to_dict() for case in self.bad_cases],
            "actions": list(self.actions),
            "limitations": list(self.limitations),
            "per_case": [dict(row) for row in self.per_case] if include_per_case else [],
        }

    def summary_line(self) -> str:
        """一句话摘要（CI 输出与演示脚本直接打印它）."""
        return (
            f"RAG 评估 {self.label}（{self.cases} 条 / k={self.k} / 提示词 "
            f"{self.prompt_version or '（未记录）'}）："
            f"检索召回 {self.value(METRIC_RETRIEVAL_RECALL):.4f} → "
            f"packed 召回 {self.value(METRIC_PACKED_RECALL):.4f} | "
            f"接地率 {self.value(METRIC_GROUNDED_RATE):.4f} | "
            f"覆盖率 {self.value(METRIC_COVERAGE):.4f} | "
            f"幻觉率 {self.value(METRIC_HALLUCINATION_RATE):.4f} | "
            f"坏例 {self.bad_case_count} 条（干净率 {self.clean_rate:.1%}）| "
            f"p95 {self.p95_latency_ms:.1f}ms"
        )

    def explain(self, *, limit: int = 5) -> list[str]:
        """人类可读诊断（六段：口径 → 指标 → 漏斗 → 回退 → 坏例 → 动作）.

        与 ``RetrievalResult.explain`` / ``GroundingReport.explain`` 同一形状：
        固定几段、每段回答一个固定问题——读报告的人要找的就是那几句。
        """
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise RagDebugError(f"limit 必须是非负整数，收到 {limit!r}")
        lines = [
            f"口径：{self.cases} 条用例、k={self.k}、提示词 "
            f"{self.prompt_version or '（未记录）'}、索引 "
            f"{self.index_version or '（未记录）'}",
            "指标：" + "、".join(
                f"{name} {self.value(name):.4f}" for name in QUALITY_METRICS
            ),
            "召回漏斗：检索名单 "
            f"{self.value(METRIC_RETRIEVAL_RECALL):.4f} → 进了提示词 "
            f"{self.value(METRIC_PACKED_RECALL):.4f}"
            f"（差 {self.value(METRIC_RETRIEVAL_RECALL) - self.value(METRIC_PACKED_RECALL):+.4f}："
            "这一段差额是**预算**的账，不是检索或生成的错）",
        ]
        if self.fallback_counts:
            rendered = "、".join(
                f"{reason or '（无回退）'} {count} 条"
                for reason, count in sorted(self.fallback_counts.items())
            )
            lines.append(f"回退分布：{rendered}")
        if self.bad_cases:
            lines.append(f"坏例 {self.bad_case_count} 条（按链路顺序列出）：")
            for case in self.bad_cases[:limit]:
                lines.append(
                    f"  - [{case.tag}] {case.query[:24]} —— {case.reason}"
                )
            if self.bad_case_count > limit:
                lines.append(
                    f"  …… 其余 {self.bad_case_count - limit} 条见 to_dict() 的 bad_cases"
                )
        if self.actions:
            lines.append("下一步动作（去重后按首次出现顺序）：")
            lines.extend(f"  {index}. {action}" for index, action in enumerate(self.actions, 1))
        lines.append("适用范围（本次结论只在以下范围内成立）：")
        lines.extend(f"  - {item}" for item in self.limitations)
        return lines

    def save(self, path: str | Path) -> Path:
        """把整份报告写成 JSON（自动创建父目录），返回实际写入的路径.

        与基线分开存：基线是"要进 git 的那 11 个数字"，报告是"这次排查的全套
        证据"（含逐条用例）。把两者混在一个文件里，仓库里的基线就会随用例数
        一起膨胀，而"diff 一下基线看这次动了多少"这件事会变得没法做。
        """
        import json

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return target


@dataclass(frozen=True)
class VariantComparison:
    """两个变体（或两次采集）的逐指标对照（**A/B，不判回归**）.

    ```text
    label_a / label_b   两边各自的名字
    deltas              逐指标的差值（含方向与三档判定）
    variables           两边**不同**的那些条件（例如"提示词版本 v2 → v1"）
    inconclusive        两边条件不可比时的原因（k 不同 / 索引版本不同）
    ```

    ``variables`` 与 ``inconclusive`` 的分工是本模块最要紧的一处：
    **"我们在比较什么"与"能不能比较"是两件事**。换提示词版本就是要比较的
    自变量（进 ``variables``）；而换了索引版本意味着两边的数字起点不同
    （进 ``inconclusive``，此时差值表仍然给出，但结论不成立）。
    """

    label_a: str
    label_b: str
    deltas: tuple[QualityDelta, ...] = ()
    variables: tuple[str, ...] = ()
    inconclusive: tuple[str, ...] = ()

    @property
    def improved(self) -> tuple[str, ...]:
        """B 相对 A 变好的指标."""
        return tuple(item.metric for item in self.deltas if item.verdict == "improved")

    @property
    def degraded(self) -> tuple[str, ...]:
        """B 相对 A 变差的指标."""
        return tuple(item.metric for item in self.deltas if item.verdict == "degraded")

    @property
    def unchanged(self) -> tuple[str, ...]:
        """落在死区里或方向为 neutral 的指标."""
        return tuple(item.metric for item in self.deltas if item.verdict == "unchanged")

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return {
            "label_a": self.label_a,
            "label_b": self.label_b,
            "variables": list(self.variables),
            "inconclusive": list(self.inconclusive),
            "improved": list(self.improved),
            "degraded": list(self.degraded),
            "unchanged": list(self.unchanged),
            "deltas": [item.to_dict() for item in self.deltas],
        }

    def summary_line(self) -> str:
        """一句话结论（"变好了几个、变差了几个"）."""
        head = f"A/B {self.label_a} → {self.label_b}"
        if self.variables:
            head += f"（变量：{'；'.join(self.variables)}）"
        if self.inconclusive:
            return f"{head}：不下结论（{'；'.join(self.inconclusive)}）"
        return (
            f"{head}：改善 {len(self.improved)} 项、退化 {len(self.degraded)} 项、"
            f"未变 {len(self.unchanged)} 项"
        )


def build_report(
    outcomes: Sequence[CaseOutcome],
    bad_cases: Sequence[BadCase] = (),
    *,
    label: str = "baseline",
    k: int = 1,
    prompt_version: str = "",
    index_version: str = "",
    limitations: tuple[str, ...] = RETRIEVAL_LIMITATIONS,
) -> RagEvalReport:
    """把一批运行结果折成一份报告（**11 个指标 + 分布 + 逐条证据 + 动作**）.

    十一个指标的口径都写在 ``baseline.METRIC_DESCRIPTIONS`` 里，这里只做
    "对一批 ``CaseOutcome`` 取均值"这一件事；而**"取均值"这个动作本身
    也有口径**，三处值得写出来：

```text
比例类指标（接地率 / 幻觉率 / 调用率 / 回退率）按**用例**取均值：
    一条有 3 个幻觉引用的用例与一条有 1 个的，在"幻觉率"上贡献相同——
    因为这个指标回答的是"这一批里多少条答案不可信"，
    而不是"总共有几个幻觉引用"（后者是 tag_counts 与 hallucinated 的账）
两段召回各自取均值（不合并）：合并之后 packed_away 就看不见了
延迟用**最近秩法** p95（day046 的 percentile），不用插值：
    小样本下插值会造出观测不到的耗时
```
    """
    rows = tuple(outcome.to_dict() for outcome in outcomes)
    cases = len(outcomes)
    metrics = {
        METRIC_RETRIEVAL_RECALL: mean_of([item.retrieved.recall for item in outcomes]),
        METRIC_RETRIEVAL_PRECISION: mean_of(
            [item.retrieved.precision for item in outcomes]
        ),
        METRIC_RETRIEVAL_NDCG: mean_of([item.retrieved.ndcg for item in outcomes]),
        METRIC_PACKED_RECALL: mean_of([item.packed.recall for item in outcomes]),
        METRIC_PACKED_NDCG: mean_of([item.packed.ndcg for item in outcomes]),
        METRIC_GROUNDED_RATE: mean_of([1.0 if item.grounded else 0.0 for item in outcomes]),
        METRIC_COVERAGE: mean_of([item.coverage for item in outcomes]),
        METRIC_HALLUCINATION_RATE: mean_of(
            [1.0 if item.hallucinated > 0 else 0.0 for item in outcomes]
        ),
        METRIC_LLM_CALL_RATE: mean_of([1.0 if item.llm_called else 0.0 for item in outcomes]),
        METRIC_FALLBACK_RATE: mean_of(
            [1.0 if item.fallback_reason else 0.0 for item in outcomes]
        ),
        METRIC_BAD_CASE_RATE: bad_case_rate(len(bad_cases), cases),
    }
    latencies = [float(item.latency_ms) for item in outcomes]
    return RagEvalReport(
        label=label,
        k=k,
        cases=cases,
        metrics=metrics,
        per_case=rows,
        bad_cases=tuple(bad_cases),
        actions=action_plan(bad_cases),
        fallback_counts=_fallback_counts(outcomes),
        tag_counts=tag_counts(bad_cases),
        prompt_version=prompt_version,
        index_version=index_version,
        p95_latency_ms=round(percentile(latencies, 95), 4),
        mean_latency_ms=mean_of(latencies),
        limitations=limitations,
    )


def compare_variants(
    baseline: RagBaseline,
    current: RagBaseline,
    *,
    dead_band: float = DEFAULT_DEAD_BAND,
) -> VariantComparison:
    """对照两份基线（A/B 实验的落地：**逐指标差值 + 变量 + 可比性**）.

    与 ``RagBaselineGuard`` 共用 ``deltas_between``（判定只有一份实现），
    但**不产出回归结论**：A/B 回答"哪个变体更好"，门禁回答"这次能不能发"。
    把两件事合成一个函数，会让一次实验的结果被读成一次发版评审。

    两处"不可比"的判据与门禁逐字相同（k 口径、索引版本），但它们是
    ``inconclusive`` 而不是异常：**A/B 是探索，探索里"这两组没法比"是
    一个结论**；而门禁里同样的情况必须拒绝给出绿灯（``ok=False``）。
    """
    inconclusive: list[str] = []
    if baseline.k != current.k:
        inconclusive.append(f"k 口径不同：{baseline.k} vs {current.k}")
    if baseline.index_version != current.index_version:
        inconclusive.append(
            f"索引版本不同：{baseline.index_version or '（未记录）'} vs "
            f"{current.index_version or '（未记录）'}"
        )
    variables: list[str] = []
    if baseline.prompt_version != current.prompt_version:
        variables.append(
            f"提示词版本 {baseline.prompt_version or '（未记录）'} → "
            f"{current.prompt_version or '（未记录）'}"
        )
    if baseline.samples != current.samples:
        variables.append(f"用例数 {baseline.samples} → {current.samples}")
    return VariantComparison(
        label_a=baseline.label,
        label_b=current.label,
        deltas=deltas_between(baseline, current, dead_band=dead_band),
        variables=tuple(variables),
        inconclusive=tuple(inconclusive),
    )


def metric_table() -> list[dict[str, str]]:
    """十一个指标的"名字 / 方向 / 说明"三列（端点与演示脚本直接展示它）.

    与 ``retrieval.describe_metrics`` / ``vectorstore.describe_backends``
    同一用途：**把口径本身也做成可读的输出**——一份报告里的数字只有配上
    "它是什么意思、往哪个方向算好"才可读，而这三列在这里各只有一份定义。
    """
    return [
        {
            "metric": name,
            "direction": METRIC_DIRECTIONS[name],
            "description": METRIC_DESCRIPTIONS[name],
        }
        for name in QUALITY_METRICS
    ]


def _fallback_counts(outcomes: Sequence[CaseOutcome]) -> dict[str, int]:
    """统计回退原因分布（**含空串那一格**：没有回退的条数也要有账）.

    为什么把"没有回退"也数出来：一份只有回退条数的分布表无法回答
    "这一批里正常的占多少"——读者要自己拿总数去减，而那个总数（用例数）
    在报告的另一个字段里。三项相减一旦有人算错，"回退率"与分布表
    就会互相矛盾。
    """
    counts: dict[str, int] = {}
    for outcome in outcomes:
        key = outcome.fallback_reason
        counts[key] = counts.get(key, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


__all__ = [
    "RagEvalReport",
    "VariantComparison",
    "build_report",
    "compare_variants",
    "metric_table",
]
