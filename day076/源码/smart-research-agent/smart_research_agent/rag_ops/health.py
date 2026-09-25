"""健康判据：这个实例现在能不能服务，以及不能服务的原因（M6-D10 / day072）.

四项检查，顺序就是"从能不能用到用得对不对"：

```text
vector_store        库里有多少条记录       0 条 → 它答不了任何知识库问题
index_integrity     清单与库对不对得上     对不上 → 检索结果不可信（少条/多条/口径不符）
corpus_freshness    距上次**成功**同步多久  太久 → 答案可能基于过期语料
quality_regression  质量相对基线退没退化    退化 → 服务是活的，但答案是坏的
```

## 三条纪律

**1. 探活不许跑评估。**

``quality_regression`` **只读**最近一次评估的结论（day071 的 ``RagBaseline``），
自己一行都不评。理由很直接：容器探针每 30 秒被调一次，而一次 6 条用例的
端到端评估要跑检索 + 生成；把它挂进探针，等于让"健康检查"本身成为
这个服务最重的负载，并且把探针的延迟与模型提供方的可用性绑在一起——
提供方抖动会一次性摘掉**全部**实例。

**2. "跳过"永远不是 ``ok``。**

任何一项因为"缺输入"而没查成，结论都是 ``warn`` 而不是 ``ok``：

```text
没有清单          → warn（这个实例不知道库里的记录是谁建的）
没有基线文件      → warn（没有基线 ≠ 基线全是零，见 day071 的 RagBaseline.load）
没有评估结果      → warn（没评估和评估通过是两件事）
```

与 day071 的"不下结论 ≠ 通过"是同一条纪律：报告里不许出现
一个**看起来没问题**的空洞。区别只在严重度——这些都不该让容器被摘掉，
因此是 ``warn`` 而不是 ``fail``。

**3. 只有"这一秒就在骗人"的事才算 ``fail``。**

```text
fail   库是空的 / 清单与库对不上 / 质量门禁判定为回归
warn   从未成功同步过 / 语料偏旧 / 缺清单 / 缺基线 / 缺评估结果
```

``fail`` 会让探针退出码变成 1（``HEALTH_STATE_EXIT_CODES``），
于是编排层停止向这个实例导流量。把 ``warn`` 也塞进 ``fail``，
只会让一个"还没灌库"的新实例永远起不来。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from smart_research_agent.config import settings
from smart_research_agent.indexing.manifest import verify_index
from smart_research_agent.indexing.types import IndexManifest
from smart_research_agent.rag_debug.baseline import (
    BaselineComparison,
    RagBaseline,
    guard_from_settings,
)
from smart_research_agent.rag_ops.errors import HealthError
from smart_research_agent.rag_ops.types import (
    CHECK_CORPUS_FRESHNESS,
    CHECK_INDEX_INTEGRITY,
    CHECK_QUALITY_REGRESSION,
    CHECK_VECTOR_STORE,
    STATE_FAIL,
    STATE_OK,
    STATE_WARN,
    HealthFinding,
    HealthReport,
    SyncLedger,
    to_iso,
    utc_now,
)
from smart_research_agent.vectorstore.base import VectorBackend


def check_vector_store(
    store: VectorBackend,
    *,
    require_non_empty: bool = True,
) -> HealthFinding:
    """查向量库里有多少条记录（**不读盘、不编码**，一次 ``count()``）.

    ``require_non_empty=False`` 的用法只有一种：**灰度/预发实例**——
    它在等第一次同步，此时"空库"是预期状态而不是故障。缺省 ``True``，
    因为"一个对外服务、库里 0 条记录"在绝大多数时候就是部署漏了一步。
    """
    count = int(store.count())
    detail = {
        "count": count,
        "backend": store.name,
        "metric": store.metric,
        "dimension": store.dimension,
        "location": store.location,
    }
    if count == 0 and require_non_empty:
        return HealthFinding(
            check=CHECK_VECTOR_STORE,
            state=STATE_FAIL,
            message=(
                "向量库是空的（0 条记录）：这个实例答不了任何知识库问题——"
                "先跑一次 POST /rag/ops/sync 把语料灌进去"
            ),
            detail=detail,
        )
    if count == 0:
        return HealthFinding(
            check=CHECK_VECTOR_STORE,
            state=STATE_WARN,
            message="向量库是空的，但本次允许空库（灰度实例等第一次同步）",
            detail=detail,
        )
    return HealthFinding(
        check=CHECK_VECTOR_STORE,
        state=STATE_OK,
        message=f"向量库里有 {count} 条记录（{store.name} / {store.metric}）",
        detail=detail,
    )


def check_index_integrity(
    manifest: IndexManifest | None,
    store: VectorBackend,
) -> HealthFinding:
    """把清单与库对账（复用 day065 的 ``verify_index``，**本模块不做任何修复**）.

    ``manifest`` 为 ``None`` 时给 ``warn`` 而不是 ``ok``：没有清单只能说明
    "没发现矛盾"，不能说明"库是对的"——检查项要拿证据说话。
    对账的四类问题（清单有库无 / 库有清单无 / 维度不符 / 口径不符）
    由 ``verify_index`` 给出，本模块只把它们的条数翻成人话。
    """
    if manifest is None:
        return HealthFinding(
            check=CHECK_INDEX_INTEGRITY,
            state=STATE_WARN,
            message=(
                "没有清单可比对：这个实例不知道库里的记录是哪一版、由谁建的——"
                "跑一次同步会生成清单（POST /rag/ops/sync）"
            ),
            detail={"count_store": int(store.count())},
        )
    verdict = verify_index(manifest, store)
    problems = list(verdict["problems"])
    detail = {
        "version_id": manifest.version_id,
        "count_manifest": manifest.count,
        "count_store": int(store.count()),
        "checks": dict(verdict["checks"]),
        "problems": problems,
    }
    if problems:
        return HealthFinding(
            check=CHECK_INDEX_INTEGRITY,
            state=STATE_FAIL,
            message=(
                f"清单与库对不上（版本 {manifest.version_id}）："
                f"{'；'.join(problems)}——检索结果会少条或多条，"
                "请先跑一次同步重建索引，而不是继续对外服务"
            ),
            detail=detail,
        )
    return HealthFinding(
        check=CHECK_INDEX_INTEGRITY,
        state=STATE_OK,
        message=(
            f"清单与库一致（版本 {manifest.version_id}，"
            f"{manifest.count} 条记录，维度与口径都对得上）"
        ),
        detail=detail,
    )


def check_corpus_freshness(
    ledger: SyncLedger | None,
    *,
    max_age_minutes: int | None = None,
    now: datetime | None = None,
) -> HealthFinding:
    """查距上次**成功**同步有多久（用 ``last_success_at``，不是 ``last_run_at``）.

    用错那一个字段是本项最容易犯的错：一个"每小时都在跑、每次都失败"的实例
    在 ``last_run_at`` 上看起来非常新鲜，而它的语料可能已经过期几天。
    """
    limit = int(settings.rag_ops_freshness_minutes if max_age_minutes is None else max_age_minutes)
    if limit < 0:
        raise HealthError(f"新鲜度阈值不能为负：{limit}。")
    moment = now if now is not None else utc_now()
    if ledger is None:
        return HealthFinding(
            check=CHECK_CORPUS_FRESHNESS,
            state=STATE_WARN,
            message="没有账本：无法判断语料有多新（从未同步过与账本丢了在这里同形）",
            detail={"max_age_minutes": limit},
        )
    age = ledger.age_minutes(moment)
    detail = {
        "max_age_minutes": limit,
        "age_minutes": age,
        "last_success_at": ledger.last_success_at,
        "last_run_at": ledger.last_run_at,
        "consecutive_failures": ledger.consecutive_failures,
    }
    if age is None:
        return HealthFinding(
            check=CHECK_CORPUS_FRESHNESS,
            state=STATE_WARN,
            message=(
                "从未成功同步过（水位为空）：库里的内容是手工灌的或来自上一次部署——"
                "跑一次 POST /rag/ops/sync 建立水位"
            ),
            detail=detail,
        )
    if age > limit:
        return HealthFinding(
            check=CHECK_CORPUS_FRESHNESS,
            state=STATE_WARN,
            message=(
                f"语料已 {age:.1f} 分钟没更新（阈值 {limit} 分钟）："
                f"上级同步状态为连续失败 {ledger.consecutive_failures} 次"
                f"{'（最近原因：' + ledger.last_error + '）' if ledger.last_error else ''}——"
                "答案是活着的，但可能基于过期语料"
            ),
            detail=detail,
        )
    return HealthFinding(
        check=CHECK_CORPUS_FRESHNESS,
        state=STATE_OK,
        message=f"语料 {age:.1f} 分钟前更新过（阈值 {limit} 分钟）",
        detail=detail,
    )


def check_quality_regression(
    current: RagBaseline | None,
    *,
    baseline_path: str | Path | None = None,
    comparison: BaselineComparison | None = None,
) -> HealthFinding:
    """读最近一次评估的结论，判断质量有没有相对基线退化（**不跑评估**）.

    两个入口都在这里，是因为它们的来源不同：

    ```text
    current        一次评估产出的 11 个指标（day071 的 RagEvalReport.to_baseline()）
    baseline_path  随仓库提交的基线文件（缺省 settings.rag_eval_baseline_path）
    comparison     调用方已经算好的 BaselineComparison（端点里已经比过一次时复用它）
    ```

    给了 ``comparison`` 就不再看另外两个参数：**同一个结论不该算两遍**——
    两遍之间只要有一次输入不同（例如基线文件在这期间被改过），
    报告里就会出现两个互相矛盾的质量读数，而那种矛盾比一个错读数更坏。
    """
    path = Path(baseline_path if baseline_path is not None else settings.rag_eval_baseline_path)
    if comparison is None:
        if current is None:
            return HealthFinding(
                check=CHECK_QUALITY_REGRESSION,
                state=STATE_WARN,
                message=(
                    "最近没有评估结果：本次不判质量（**跳过不等于通过**）——"
                    "跑一次 POST /rag/eval/run 会产出可供判定的 11 个指标"
                ),
                detail={"baseline_path": str(path), "baseline_exists": path.exists()},
            )
        if not path.exists():
            return HealthFinding(
                check=CHECK_QUALITY_REGRESSION,
                state=STATE_WARN,
                message=(
                    f"没有质量基线文件 {str(path)!r}：没有基线与'基线全是零'是两回事，"
                    "本次不判质量——先跑一次评估并把基线提交进仓库"
                ),
                detail={"baseline_path": str(path), "baseline_exists": False},
            )
        comparison = guard_from_settings(RagBaseline.load(path)).compare(current)

    report = comparison
    detail = {
        "baseline_path": str(path),
        "baseline_exists": path.exists(),
        "conclusive": report.conclusive,
        "reasons": list(report.reasons),
        "regressions": [item.to_dict() for item in report.regressions],
        "summary": report.summary_line(),
    }
    if report.regressions:
        first = report.regressions[0]
        return HealthFinding(
            check=CHECK_QUALITY_REGRESSION,
            state=STATE_FAIL,
            message=(
                f"质量相对基线退化（{len(report.regressions)} 项）：{first.message}——"
                "服务是活的，但答案是坏的，请回滚索引版本或回滚提示词版本"
            ),
            detail=detail,
        )
    if not report.conclusive:
        return HealthFinding(
            check=CHECK_QUALITY_REGRESSION,
            state=STATE_WARN,
            message=(
                "本次评估与基线**不可比**，因此不给质量结论（不当成通过）："
                + "；".join(report.reasons)
            ),
            detail=detail,
        )
    return HealthFinding(
        check=CHECK_QUALITY_REGRESSION,
        state=STATE_OK,
        message=f"质量与基线可比且没有退化：{report.summary_line()}",
        detail=detail,
    )


def build_health_report(
    store: VectorBackend,
    *,
    manifest: IndexManifest | None = None,
    ledger: SyncLedger | None = None,
    quality: RagBaseline | None = None,
    quality_comparison: BaselineComparison | None = None,
    max_age_minutes: int | None = None,
    baseline_path: str | Path | None = None,
    require_non_empty: bool = True,
    now: datetime | None = None,
) -> HealthReport:
    """跑完四项检查，装成一份报告（**顺序 = ``HEALTH_CHECKS`` 的顺序**）.

    它**不跑任何评估、不读语料、不写任何东西**：四项检查的输入全部是
    "别人已经算好的账"（库、清单、账本、评估基线）。因此这个函数可以被
    容器探针每 30 秒调用一次。
    """
    moment = now if now is not None else utc_now()
    findings = (
        check_vector_store(store, require_non_empty=require_non_empty),
        check_index_integrity(manifest, store),
        check_corpus_freshness(ledger, max_age_minutes=max_age_minutes, now=moment),
        check_quality_regression(
            quality, baseline_path=baseline_path, comparison=quality_comparison
        ),
    )
    return HealthReport(findings=findings, checked_at=to_iso(moment))


__all__ = [
    "build_health_report",
    "check_corpus_freshness",
    "check_index_integrity",
    "check_quality_regression",
    "check_vector_store",
]
