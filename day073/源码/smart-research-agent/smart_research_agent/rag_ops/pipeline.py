"""运维流水线：扫盘 → 差集 → 重建 → 体检 → 采样 → 落账（M6-D10 / day072）.

它是这一课的**最高一层**，也是唯一一个会"动手"的模块：前面六个模块
（``types`` / ``snapshot`` / ``sync`` / ``schedule`` / ``health`` / ``deploy``）
全是判据与形状，本模块把它们按一条固定顺序串起来，并把每次运行的结果
留在账本与报告里。

```text
① 读账本        SyncLedger.load          （账本不存在 = 首次运行，不是错误）
② 判调度        schedule.decide          （没到点就**结束**，一次编码都不做）
③ 扫语料        snapshot.scan_corpus     （文件只读一遍：快照与文档一起拿）
④ 算差集        sync.plan_sync           （文件级的账：谁变了）
⑤ 定模式        sync.resolved_mode       （变更比例高 → 整库重建）
⑥ 重建索引      indexer.build_from_documents  ← day065 的块级增量在这里
⑦ 落盘          store.persist + versions.persist  （内存库必须落盘，见下）
⑧ 前进水位      ledger.mark_success / mark_failure
⑨ 体检 + 采样   health.build_health_report / metrics.collect_samples
⑩ 写报告 + 账本 OpsReport.save / ledger.save
```

## 为什么第 ⑦ 步是这个模块不可推卸的责任

day064 的 ``FlatVectorStore`` 默认**不落盘**（``vector_persist_path`` 缺省空串），
这是刻意的：一个"以为落盘了其实还在内存里"的向量库要到进程重启才暴露。
而容器化的世界里"进程重启"是**常态**（滚动发布、节点漂移、OOM）。
因此"重建完索引之后把它写下来"这件事必须发生在编排层——
它是唯一知道"这一次重建是不是完整成功"的那一层。

## 三处"必须留痕"的降级

```text
落盘路径为空        → 不落盘，但报告里留一条 note（否则容器重建后"库是空的"
                      会被读成一次新的故障，而不是一次已知的取舍）
评估报告读不出来    → 质量采样缺席，体检给出 warn（"没评估"永远不等于"通过"）
语料目录一份都没有  → 照常记一次成功（水位前进到"空"），并留一条 note
```

三个降级都**不抛异常**：运维流水线的价值在于"每次都留下可读的结论"，
而不是"遇到不完美就整体停下"。相反，**编码失败、清单写不进、维度不符
这类错误一律上抛**（它们会让库与账本不一致，必须由人看见）。

## 一次运行只读两次时钟

``run_once`` 只在开头读一次时钟（``started``），在每条出口前读一次
（``finished``），中间的每一步都用 ``started``。这样做的收益不只是可复现：
报告里的 ``started_at / finished_at / seconds`` 与账本里的 ``last_run_at``
**来自同一批读数**，因此不会出现"报告说跑了 0.4 秒、账本说跨了两分钟"
这种两个时间源打架的情况（它们在同一个进程里，却来自两次不同的时钟调用）。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from smart_research_agent.config import settings
from smart_research_agent.documents.base import LoaderRegistry
from smart_research_agent.indexing.errors import IndexingError
from smart_research_agent.indexing.manifest import MANIFEST_HISTORY_FILE
from smart_research_agent.indexing.pipeline import (
    IndexingPipeline,
    default_indexing_pipeline,
)
from smart_research_agent.indexing.types import IndexingReport, IndexManifest
from smart_research_agent.indexing.versioning import IndexVersionStore
from smart_research_agent.llm.embedding import EmbeddingProvider
from smart_research_agent.rag_debug.baseline import RagBaseline
from smart_research_agent.rag_debug.errors import RagDebugError
from smart_research_agent.rag_ops.deploy import default_spec
from smart_research_agent.rag_ops.deploy import describe as describe_spec
from smart_research_agent.rag_ops.errors import RagOpsError
from smart_research_agent.rag_ops.health import build_health_report
from smart_research_agent.rag_ops.metrics import (
    MetricsBuffer,
    collect_samples,
    samples_from_ledger,
)
from smart_research_agent.rag_ops.schedule import decide
from smart_research_agent.rag_ops.schedule import describe as describe_schedule
from smart_research_agent.rag_ops.snapshot import scan_corpus, scan_sources
from smart_research_agent.rag_ops.sync import plan_sync, resolved_mode
from smart_research_agent.rag_ops.types import (
    OPS_REPORT_VERSION,
    STATUS_FAILED,
    STATUS_NO_CHANGE,
    STATUS_NOT_DUE,
    STATUS_SYNCED,
    CorpusSnapshot,
    HealthReport,
    MetricSample,
    OpsReport,
    ScheduleDecision,
    SchedulePolicy,
    SyncLedger,
    SyncPlan,
    to_iso,
    utc_now,
)
from smart_research_agent.vectorstore.base import VectorBackend
from smart_research_agent.vectorstore.errors import VectorStoreError
from smart_research_agent.vectorstore.registry import create_backend

#: 时钟函数：返回"现在"。可注入是这一层能被测试的前提
#: （否则"跨了三个计划点才醒过来"这种用例要等三天才能真实发生）。
Clock = Callable[[], datetime]

#: 质量来源：返回最近一次评估的 11 个指标（``None`` 表示"没有评估结果"）。
QualityLoader = Callable[[], RagBaseline | None]

#: 账本文件名（``paths_in`` 用它拼路径；与 ``settings.rag_ops_ledger_path`` 的末段一致）。
LEDGER_FILENAME = "sync_ledger.json"

#: 报告文件名（同上）。
REPORT_FILENAME = "ops_report.json"


def paths_in(directory: str | Path) -> tuple[str, str]:
    """把"一个目录"折成 ``(账本路径, 报告路径)``；**空目录返回两个空串**.

    端点（``create_app`` 的 ``rag_ops_dir``）需要一个"只给一个目录"的入口，
    而"空串 = 明确不落盘"是本项目从 day064 起的一条固定纪律：

    ```text
    vector_persist_path = ""    默认不落盘（一个"以为落盘了"的库要到重启才暴露）
    indexing_dir = ""           默认不落盘（否则跑一次测试就在仓库里留下 data/index）
    rag_ops_dir = ""            同上：一次测试不该在仓库里留下 data/ops
    ```

    因此这里**不**在空目录时回落到 settings：回落的后果是"我以为注入了一个空目录、
    结果它写进了仓库"，而那种事在 CI 上的表现是"工作区变脏"，
    在本地则是"我的 data/ops 里多了一份别人的账本"。
    """
    raw = str(directory or "").strip()
    if not raw:
        return "", ""
    base = Path(raw)
    return str(base / LEDGER_FILENAME), str(base / REPORT_FILENAME)


def quality_from_report(path: str | Path) -> RagBaseline | None:
    """从 day071 的评估报告落盘文件里读回一份 ``RagBaseline``（**读不到就返回 None**）.

    为什么读"报告"而不是重新跑一次评估：探活与定时同步都不该调用模型
    （见 ``health`` 模块的第 1 条纪律）。因此运维层消费的是评估**已经产出**
    的那份账——这正是 day071 把 ``RagEvalReport.save`` 单独留在那里的理由。

    三种"读不到"都给 ``None``，而不是抛异常：文件不存在、不是 JSON、
    指标不全（真跑过一次评估就会写全 11 个）。它们会让体检给出 ``warn``，
    而"没评估"本来就不是"评估通过"。
    """
    target = Path(path)
    if not target.exists():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("metrics"), dict):
        return None
    try:
        return RagBaseline(
            label=str(payload.get("label", "report")),
            samples=int(payload.get("cases", 0)),
            k=int(payload.get("k", 1)),
            metrics={str(key): float(value) for key, value in payload["metrics"].items()},
            prompt_version=str(payload.get("prompt_version", "")),
            index_version=str(payload.get("index_version", "")),
            fallback_counts=dict(payload.get("fallback_counts") or {}),
            tag_counts=dict(payload.get("tag_counts") or {}),
            p95_latency_ms=float(payload.get("p95_latency_ms", 0.0)),
            mean_latency_ms=float(payload.get("mean_latency_ms", 0.0)),
        )
    except (RagDebugError, TypeError, ValueError):
        return None


def indexer_versions(indexer: IndexingPipeline) -> IndexVersionStore | None:
    """从索引流水线手里取那份版本表（取不到就 ``None``，**绝不另建一个**）.

    另建一个指向同一目录的 ``IndexVersionStore`` 是本课最容易踩的一处静默 bug：
    构建器会把新版本登记进**它自己**那份表里，而运维层那份仍然是空的——
    于是体检永远报告"没有清单可比对"，而磁盘上其实躺着一份完整的清单。
    """
    return getattr(indexer, "versions", None)


class RagOpsPipeline:
    """一条运维流水线：一次同步 = 一次 :meth:`run_once`（见模块 docstring 的十步）.

    它**不做判定**：该不该跑由 ``schedule`` 说、谁变了由 ``sync`` 说、
    怎么建由 ``sync.resolved_mode`` 说、能不能服务由 ``health`` 说。
    本类的职责是**顺序**与**留痕**——这两件事恰恰不能分散在各模块里，
    因为"哪一步在什么时候发生了"决定了报告能回答什么问题。
    """

    def __init__(
        self,
        indexer: IndexingPipeline,
        *,
        store: VectorBackend,
        source_dir: str | None = None,
        policy: SchedulePolicy | None = None,
        ledger_path: str | None = None,
        report_path: str | None = None,
        versions: IndexVersionStore | None = None,
        persist_path: str | None = None,
        quality_loader: QualityLoader | None = None,
        registry: LoaderRegistry | None = None,
        limit: int | None = None,
        strict: bool = True,
        buffer: MetricsBuffer | None = None,
        clock: Clock | None = None,
    ) -> None:
        """装配流水线（**构造无副作用**：不建目录、不读盘、不写任何东西）.

        ``store`` 与 ``indexer`` 是两个不同的东西，这不是冗余：

        ```text
        indexer   负责"把记录变成索引"（day065 的构建器 + 缓存 + 版本 + 备份）
        store     负责"现在库里有什么"（health 的一次 count、落盘时的 persist）
        ```
        """
        self._indexer = indexer
        self._store = store
        self._source_dir = str(settings.rag_ops_source_dir if source_dir is None else source_dir)
        self._policy = policy if policy is not None else policy_from_settings()
        self._ledger_path = str(
            settings.rag_ops_ledger_path if ledger_path is None else ledger_path
        )
        self._report_path = str(
            settings.rag_ops_report_path if report_path is None else report_path
        )
        self._versions = versions if versions is not None else indexer_versions(indexer)
        self._persist_path = str(
            settings.vector_persist_path if persist_path is None else persist_path
        )
        resolved_loader: QualityLoader = (
            quality_loader
            if quality_loader is not None
            else (lambda: quality_from_report(settings.rag_eval_report_path))
        )
        self._quality_loader = resolved_loader
        self._registry = registry
        self._limit = limit
        self._strict = strict
        self._buffer = buffer if buffer is not None else MetricsBuffer()
        self._clock: Clock = clock if clock is not None else utc_now

    # ------------------------------------------------------------------ 元信息

    @property
    def source_dir(self) -> str:
        """语料根目录."""
        return self._source_dir

    @property
    def ledger_path(self) -> str:
        """账本落盘路径（空串表示只留在内存里）."""
        return self._ledger_path

    @property
    def report_path(self) -> str:
        """报告落盘路径."""
        return self._report_path

    @property
    def policy(self) -> SchedulePolicy:
        """当前的调度策略."""
        return self._policy

    @property
    def store(self) -> VectorBackend:
        """体检与落盘用的向量库（**它必须就是服务用的那一个**）.

        把它暴露出来是为了让装配方（``create_app`` 与演示脚本）能把
        "同一个库"这件事变成代码里的同一份引用，而不是一句承诺：
        体检一个库、服务另一个库的实现会让"健康"这两个字失去意义。
        """
        return self._store

    @property
    def buffer(self) -> MetricsBuffer:
        """监控缓冲（本次进程内的采样都在这里）."""
        return self._buffer

    # ------------------------------------------------------------------ 只读

    def load_ledger(self) -> SyncLedger:
        """读账本（空路径 = 只在内存里，因此每次都是空账本）."""
        if not self._ledger_path:
            return SyncLedger.empty()
        return SyncLedger.load(self._ledger_path)

    def current_manifest(self) -> IndexManifest | None:
        """当前生效的清单（版本表里最近登记的那一版；没有版本表时 ``None``）.

        刻意用 ``latest()`` 而不是 ``current``：``current`` 是"被采纳的版本号"，
        它在版本表没读盘时是空串——而体检要回答的是"库里那批记录对应哪份清单"，
        构建成功的那一刻两者是同一个东西（``builder`` 会 register + adopt）。
        """
        if self._versions is None:
            return None
        return self._versions.latest()

    def snapshot(self, *, now: datetime | None = None) -> CorpusSnapshot:
        """只扫一遍语料（**只取身份，不取文档**，供只读端点与预览用）."""
        moment = now if now is not None else self._clock()
        return scan_sources(
            self._source_dir,
            registry=self._registry,
            limit=self._limit,
            strict=self._strict,
            clock=moment,
        )

    def plan(self) -> SyncPlan:
        """只算差集（不构建、不落盘）——"这次会做什么"的预览."""
        ledger = self.load_ledger()
        return plan_sync(ledger.snapshot, self.snapshot())

    def health_report(self, *, now: datetime | None = None) -> HealthReport:
        """只做体检（**不跑评估、不构建**，见 ``health`` 模块的纪律）."""
        moment = now if now is not None else self._clock()
        return build_health_report(
            self._store,
            manifest=self.current_manifest(),
            ledger=self.load_ledger(),
            quality=self._quality(),
            baseline_path=settings.rag_eval_baseline_path,
            require_non_empty=True,
            now=moment,
        )

    def status(self, *, now: datetime | None = None, force: bool = False) -> dict[str, Any]:
        """一段完整的只读状态（``/rag/ops/status`` 的全部内容）.

        四块内容，都**不产生任何副作用**——这也是它们能同时出现在
        一个 GET 端点里的原因：

        ```text
        index     索引现状（day065 的 stats：后端、维度、计数、版本、备份、缓存）
        schedule  账本 + 调度判定（"下一次什么时候"与"为什么这次不跑"）
        metrics   本次进程内的监控采样汇总
        deploy    编排规格的静态校验结论 + 与仓库里那份 yml 的一致性
        ```
        """
        moment = now if now is not None else self._clock()
        ledger = self.load_ledger()
        return {
            "source_dir": self._source_dir,
            "ledger_path": self._ledger_path,
            "report_path": self._report_path,
            "index": self._indexer.stats(),
            "schedule": describe_schedule(self._policy, ledger, now=moment, force=force),
            "metrics": self._buffer.describe(),
            "deploy": describe_spec(default_spec()),
            "last_report": self._last_report(),
        }

    # ------------------------------------------------------------------ 写入

    def run_once(self, *, now: datetime | None = None, force: bool = False) -> OpsReport:
        """跑一次同步（**唯一会改变世界的入口**，见模块 docstring 的十步）.

        三种"没做事"的结果都是**正常返回**，而不是异常：

        ```text
        not_due    没到计划时间（连语料目录都没碰，账本也不写）
        no_change  到点了，但一份来源都没变（一行都没写，水位照常前进）
        failed     跑了但失败了（水位不动，原因留在账本与报告里）
        ```

        把前两种实现成异常，会逼调用方在每个调用点写一次 try——
        而那些"正常的空转"本来就该出现在报告里，与异常处理混在一起之后，
        "这个月跑了 30 次、其中 28 次没变化"这种结论就再也统计不出来。
        """
        started = now if now is not None else self._clock()
        ledger = self.load_ledger()
        decision = decide(self._policy, ledger, now=started, force=force)

        if not decision.due:
            return self._finish(
                started=started,
                finished=self._clock(),
                decision=decision,
                status=STATUS_NOT_DUE,
                plan=None,
                index_report=None,
                health=None,
                samples=(),
                notes=(
                    "本次未执行：不读语料、不构建、不写账本，"
                    "也**不覆盖上一次落盘的报告**（调度判定照常进本次响应）",
                ),
                error="",
                persist=False,
            )

        notes: list[str] = []
        try:
            snapshot, documents = scan_corpus(
                self._source_dir,
                registry=self._registry,
                limit=self._limit,
                strict=self._strict,
                clock=started,
            )
            plan = plan_sync(ledger.snapshot, snapshot)
            if snapshot.count == 0:
                notes.append(
                    "语料目录里一份受支持的文档都没有：本次不会有任何内容进库——"
                    "请确认 rag_ops_source_dir 指向了正确的目录"
                )
            if snapshot.skipped:
                notes.append(
                    f"有 {len(snapshot.skipped)} 份来源读不进来（strict=False）："
                    + "；".join(f"{src}（{err}）" for src, err in snapshot.skipped[:3])
                )

            index_report: IndexingReport | None = None
            if plan.empty:
                notes.append("语料一份来源都没变：本次不重建索引（一行都不会写）")
                status = STATUS_NO_CHANGE
            else:
                mode, mode_note = resolved_mode(plan)
                notes.append(mode_note)
                index_report = self._indexer.build_from_documents(documents, mode=mode)
                notes.append(
                    f"构建报告：{index_report.summary_line()}（mode={index_report.mode}，"
                    f"父版本 {index_report.parent_version or '（首版）'}）"
                )
                notes.extend(self._persist_state(index_report.version_id))
                status = STATUS_SYNCED

            finished = self._clock()
            seconds = max(0.0, (finished - started).total_seconds())
            updated = ledger.mark_success(
                at=to_iso(started),
                snapshot=snapshot,
                index_version=index_report.version_id if index_report is not None else "",
            )
            self._save_ledger(updated)
            quality = self._quality()
            samples = collect_samples(
                at=started,
                now=finished,
                plan=plan,
                index_report=index_report,
                seconds=seconds,
                store=self._store,
                ledger=updated,
                quality=quality,
            )
            self._buffer.extend(samples)
            report_health = self._build_health(ledger=updated, quality=quality, now=finished)
            return self._finish(
                started=started,
                finished=finished,
                decision=decision,
                status=status,
                plan=plan,
                index_report=index_report,
                health=report_health,
                samples=samples,
                notes=tuple(notes),
                error="",
            )
        except (RagOpsError, IndexingError, VectorStoreError) as exc:
            finished = self._clock()
            failed = ledger.mark_failure(at=to_iso(finished), error=str(exc))
            self._save_ledger(failed)
            notes.append(
                "本次失败：水位未前进（下一趟会重试同一批来源），"
                "而失败路径不做体检——出错之后的状态由下一次运行体检"
            )
            samples = samples_from_ledger(failed, now=finished, at=finished)
            self._buffer.extend(samples)
            return self._finish(
                started=started,
                finished=finished,
                decision=decision,
                status=STATUS_FAILED,
                plan=None,
                index_report=None,
                health=None,
                samples=samples,
                notes=tuple(notes),
                error=str(exc),
            )

    # ------------------------------------------------------------------ 内部

    def _persist_state(self, version_id: str) -> list[str]:
        """落盘两样东西：向量库快照与版本表（**顺序不能反**）.

        先落库、再落版本表：版本表里的那一版是"库应该长什么样"的**声明**，
        它必须晚于库本身的落盘。反过来会出现"版本表说库里是这一版，
        而实际落盘的还是上一版"——重启后检索结果与清单不符，
        而体检只会把它报成"清单与库对不上"，查不到根因。
        """
        notes: list[str] = []
        if self._persist_path:
            written = self._store.persist(self._persist_path)
            notes.append(f"向量库已落盘：{written}")
        else:
            notes.append(
                "向量库未落盘（vector_persist_path 为空）：内存库在进程重启后为空——"
                "这是配置的显式取舍，容器化部署请务必设置它"
            )
        if self._versions is not None:
            directory = self._versions.persist()
            notes.append(f"版本表已落盘：{directory}（当前版本 {version_id}）")
        return notes

    def _save_ledger(self, ledger: SyncLedger) -> None:
        """写账本（空路径 = 不落盘，见 ``SyncLedger.load`` 的说明）."""
        if self._ledger_path:
            ledger.save(self._ledger_path)

    def _quality(self) -> RagBaseline | None:
        """读最近一次评估的结果（读不到就 ``None``，由体检给出 warn）."""
        if self._quality_loader is None:
            return None
        try:
            return self._quality_loader()
        except RagDebugError:
            return None

    def _build_health(
        self,
        *,
        ledger: SyncLedger,
        quality: RagBaseline | None,
        now: datetime,
    ) -> HealthReport:
        """体检（四项检查，见 ``health.build_health_report``）.

        ``quality`` **没有缺省值**：它必须由调用方在"体检之前"取一次。
        给一个缺省 ``None`` 会让"忘了取质量"与"确实没有评估结果"
        在代码里长得一样，而它们对体检的含义相反（后者要 warn）。
        """
        return build_health_report(
            self._store,
            manifest=self.current_manifest(),
            ledger=ledger,
            quality=quality,
            baseline_path=settings.rag_eval_baseline_path,
            require_non_empty=True,
            now=now,
        )

    def _finish(
        self,
        *,
        started: datetime,
        finished: datetime,
        decision: ScheduleDecision,
        status: str,
        plan: SyncPlan | None,
        index_report: IndexingReport | None,
        health: HealthReport | None,
        samples: Sequence[MetricSample],
        notes: Sequence[str],
        error: str,
        persist: bool = True,
    ) -> OpsReport:
        """装配报告、写盘、返回（**四条出口都走这里**）.

        只留一个出口是为了让"每一次运行都有报告"变成结构上的必然：
        三条成功路径与一条失败路径如果各自拼一份报告，
        迟早会有一条忘了带上 ``schedule``（于是"这次为什么没跑"再也答不出来）。

        ``persist`` 只有"未到点"那一档会传 ``False``：空转**不许覆盖**
        上一次真正跑过的那份报告，否则 ``/rag/ops/status`` 读到的
        ``last_report`` 会被一串"未到点"盖住——而那正是值班的人最需要看到的
        "上一次同步到底成功没有"的证据（与"空报告不是健康"是同一条纪律）。
        """
        seconds = max(0.0, (finished - started).total_seconds())
        report = OpsReport(
            started_at=to_iso(started),
            finished_at=to_iso(finished),
            status=status,
            schedule=decision,
            plan=plan,
            index=index_report.to_dict() if index_report is not None else None,
            health=health,
            metrics=tuple(samples),
            notes=tuple(notes),
            error=error,
            seconds=round(seconds, 4),
            version=OPS_REPORT_VERSION,
        )
        if persist and self._report_path:
            report.save(self._report_path)
        return report

    def _last_report(self) -> dict[str, Any] | None:
        """读上一次落盘的报告（没有就返回 ``None``，**不造假报告**）."""
        target = Path(self._report_path)
        if not target.exists():
            return None
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"error": f"{self._report_path} 读不出来（坏 JSON）"}
        return payload if isinstance(payload, dict) else None


def policy_from_settings() -> SchedulePolicy:
    """按 ``settings.rag_ops_*`` 装配调度策略（端点、脚本、worker 三处共用）."""
    return SchedulePolicy(
        interval_minutes=int(settings.rag_ops_interval_minutes),
        jitter_seconds=int(settings.rag_ops_jitter_seconds),
        max_lag_minutes=int(settings.rag_ops_max_lag_minutes),
        backoff_base_seconds=int(settings.rag_ops_backoff_base_seconds),
        backoff_max_seconds=int(settings.rag_ops_backoff_max_seconds),
    )


def default_rag_ops_pipeline(
    *,
    embedding: EmbeddingProvider | None = None,
    backend: VectorBackend | None = None,
    indexer: IndexingPipeline | None = None,
    source_dir: str | None = None,
    policy: SchedulePolicy | None = None,
    ledger_path: str | None = None,
    report_path: str | None = None,
    persist_path: str | None = None,
    clock: Clock | None = None,
) -> RagOpsPipeline:
    """按 ``settings`` 装配一条完整的运维流水线（端点与演示脚本的统一入口）.

    装配清单，以及**为什么每一项都必须在这里装配**：

    ```text
    backend       create_backend()           vector_backend / vector_metric / vector_persist_path
    store         backend 本身               落盘与 count 都要它；它与 indexer 用的是同一个对象
    indexer       default_indexing_pipeline(backend=backend)
                                             **同一个后端实例**，否则"重建了哪个库"会有两个答案
    versions      <indexing_dir>             读盘（文件存在才 load：不做磁盘上的猜测）
    policy        policy_from_settings()     间隔 / 抖动 / 退避
    persist_path  settings.vector_persist_path
    quality       读 outputs/rag_eval/report.json（day071 的产物）
    ```

    ``clock`` 会一路传下去（快照的 ``created_at``、调度判定、报告时间戳），
    因此演示脚本可以用一个固定时刻跑出可复现的输出。
    """
    resolved_backend = backend if backend is not None else create_backend()

    if indexer is None:
        resolved_indexer = default_indexing_pipeline(
            embedding=embedding, backend=resolved_backend
        )
    else:
        resolved_indexer = indexer

    # 版本表**从构建器手里拿**，而不是另建一个指向同一目录的实例：
    # 两个实例会让"刚构建完的那一版"只存在于构建器内存里，
    # 而运维层读到的永远是空表（见 IndexingPipeline.versions 的说明）。
    versions = resolved_indexer.versions
    directory = str(settings.indexing_dir)
    history = Path(directory) / MANIFEST_HISTORY_FILE
    if versions is not None and history.exists():
        versions.load()
    return RagOpsPipeline(
        resolved_indexer,
        store=resolved_backend,
        source_dir=source_dir,
        policy=policy,
        ledger_path=ledger_path,
        report_path=report_path,
        persist_path=persist_path,
        versions=versions,
        clock=clock,
    )


__all__ = [
    "LEDGER_FILENAME",
    "REPORT_FILENAME",
    "Clock",
    "QualityLoader",
    "RagOpsPipeline",
    "default_rag_ops_pipeline",
    "indexer_versions",
    "paths_in",
    "policy_from_settings",
    "quality_from_report",
]
