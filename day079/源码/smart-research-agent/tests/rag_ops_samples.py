"""``rag_ops`` 的共享样本：三份语料、一个可复现的时钟、一个完全隔离的流水线.

三个约定（与 ``rag_debug_samples`` 同一取向）：

```text
① 全部离线        向量后端用 flat、编码器用默认的 char-ngram，零可选依赖、零网络
② 时间可注入      FakeClock 让"跨了三个计划点才醒过来"这类用例在 0 秒内跑完
③ 目录全隔离      账本 / 报告 / 索引 / 版本表全部写在 tmp_path 下，
                  绝不在仓库里留下 data/ops 或 data/index
```

第 ③ 条不是洁癖：``default_indexing_pipeline()`` 的版本表路径来自
``settings.indexing_dir``，直接用它会在仓库里建出 ``data/index``——
而"测试跑完工作区变脏"会让 ``git status`` 从此不再是一份可信的信号。
因此 :func:`isolated_indexer` 手工装配构建器，把四个路径都指向临时目录
（与 ``api.app.assemble_indexing_pipeline`` 同一个套路）。
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path

from smart_research_agent.indexing import (
    EmbeddingCache,
    IndexBackupStore,
    IndexBuilder,
    IndexingPipeline,
    IndexVersionStore,
    describe_embedding,
)
from smart_research_agent.llm.embedding import default_embedding
from smart_research_agent.rag_ops import (
    LEDGER_FILENAME,
    REPORT_FILENAME,
    RagOpsPipeline,
    SchedulePolicy,
    SourceEntry,
    SyncLedger,
    SyncPlan,
    snapshot_id,
)
from smart_research_agent.rag_ops.types import (
    CorpusSnapshot,
    HealthFinding,
    HealthReport,
    MetricSample,
    to_iso,
)
from smart_research_agent.vectorstore.registry import create_backend

#: 全部用例共用的"现在"（2026-10-06 03:00 UTC = 本课程那次自动运行的时刻）.
FIXED_NOW = datetime(2026, 10, 6, 3, 0, 0, tzinfo=timezone.utc)

#: 三份语料：一份带小标题、一份是运维笔记、一份更短（用来制造"只改一份"的场景）.
CORPUS_FILES: dict[str, str] = {
    "rag.md": "# 检索增强生成\n\nRAG 把检索与生成结合起来。\n\n## 向量检索\n\n余弦相似度常用。\n",
    "ops.md": "# 运维\n\n索引需要定期增量更新。\n",
    "sched.md": "# 调度\n\n每天凌晨跑一次同步。\n",
}

#: 只留一份语料（用来验证"来源数 1 份""全量重建"等结论）.
SINGLE_FILE: dict[str, str] = {"rag.md": CORPUS_FILES["rag.md"]}


class FakeClock:
    """可推动的时钟（``advance`` 之后所有取时间的调用都会看到新时刻）."""

    def __init__(self, moment: datetime | None = None) -> None:
        self.moment = moment if moment is not None else FIXED_NOW

    def __call__(self) -> datetime:
        """取当前时刻（可以直接当 ``Clock`` 传进流水线）."""
        return self.moment

    def advance(self, **delta: float) -> datetime:
        """把时钟往前推（``minutes=30`` / ``hours=1`` / ``days=2``）."""
        self.moment = self.moment + timedelta(**delta)
        return self.moment


def fingerprint(seed: str) -> str:
    """由种子算一个确定性的 16 位指纹（**不碰文件**，用于手工构造快照）."""
    import hashlib

    return hashlib.sha256(seed.encode()).hexdigest()[:16]


def entry(
    source: str,
    *,
    seed: str | None = None,
    char_count: int = 10,
    media_type: str = "text/markdown",
) -> SourceEntry:
    """造一条快照条目（``seed`` 缺省与路径相同 → 同路径同指纹）."""
    return SourceEntry(
        source=source,
        fingerprint=fingerprint(seed if seed is not None else source),
        char_count=char_count,
        media_type=media_type,
    )


def snapshot(
    entries: tuple[SourceEntry, ...] | None = None,
    *,
    skipped: tuple[tuple[str, str], ...] = (),
    created_at: str = "",
) -> CorpusSnapshot:
    """造一份快照（缺省是两份来源：docs/a.md 与 docs/b.md）."""
    resolved = entries if entries is not None else (entry("docs/a.md"), entry("docs/b.md"))
    return CorpusSnapshot(entries=resolved, skipped=skipped, created_at=created_at)


def plan(
    *,
    added: tuple[str, ...] = (),
    updated: tuple[str, ...] = (),
    removed: tuple[str, ...] = (),
    unchanged: tuple[str, ...] = (),
    previous_sources: tuple[str, ...] | None = None,
    current_sources: tuple[str, ...] | None = None,
    reason: str = "手工构造的差集",
) -> SyncPlan:
    """手工造一份差集（两侧来源清单缺省由四组动作推出来）."""
    previous = (
        tuple(sorted({*removed, *updated, *unchanged}))
        if previous_sources is None
        else previous_sources
    )
    current = (
        tuple(sorted({*added, *updated, *unchanged}))
        if current_sources is None
        else current_sources
    )
    return SyncPlan(
        added=added,
        updated=updated,
        removed=removed,
        unchanged=unchanged,
        previous_id=snapshot_id(tuple(entry(name) for name in previous)) if previous else "",
        current_id=snapshot_id(tuple(entry(name) for name in current)) if current else "",
        previous_sources=previous,
        current_sources=current,
        reason=reason,
    )


def policy(**overrides: object) -> SchedulePolicy:
    """造一个调度策略（缺省：每天一次、抖动 300 秒、退避 60→3600 秒）."""
    fields: dict[str, object] = {
        "interval_minutes": 1440,
        "jitter_seconds": 300,
        "max_lag_minutes": 720,
        "backoff_base_seconds": 60,
        "backoff_max_seconds": 3600,
    }
    fields.update(overrides)
    return SchedulePolicy(**fields)  # type: ignore[arg-type]


def ledger(
    snapshot_value: CorpusSnapshot | None = None,
    *,
    last_run_at: str = "",
    last_success_at: str = "",
    last_index_version: str = "",
    consecutive_failures: int = 0,
    runs: int = 0,
    failures: int = 0,
    last_error: str = "",
) -> SyncLedger:
    """造一份账本（缺省是"全新的、没有任何水位"）."""
    return SyncLedger(
        snapshot=snapshot_value,
        last_run_at=last_run_at,
        last_success_at=last_success_at,
        last_index_version=last_index_version,
        consecutive_failures=consecutive_failures,
        runs=runs,
        failures=failures,
        last_error=last_error,
    )


def ran_ledger(
    snapshot_value: CorpusSnapshot | None = None,
    *,
    moment: datetime | None = None,
    index_version: str = "",
) -> SyncLedger:
    """造一份"刚成功同步过"的账本（水位 = 传入的快照）."""
    at = to_iso(moment if moment is not None else FIXED_NOW)
    water = snapshot_value if snapshot_value is not None else snapshot()
    return ledger(
        water,
        last_run_at=at,
        last_success_at=at,
        last_index_version=index_version,
        runs=1,
    )


def finding(
    check: str = "vector_store",
    *,
    state: str = "ok",
    message: str = "手工构造的判据",
    detail: dict | None = None,
) -> HealthFinding:
    """造一条健康判据."""
    return HealthFinding(check=check, state=state, message=message, detail=detail or {})


def health_report(*findings: HealthFinding) -> HealthReport:
    """造一份健康报告（缺省四项检查全 ok，可用参数逐个替换）."""
    resolved = findings or (
        finding("vector_store"),
        finding("index_integrity"),
        finding("corpus_freshness"),
        finding("quality_regression"),
    )
    return HealthReport(findings=tuple(resolved), checked_at=to_iso(FIXED_NOW))


def sample(name: str = "sources_total", value: float = 1.0, **kwargs: object) -> MetricSample:
    """造一个监控采样（缺省时刻是 FIXED_NOW）."""
    fields: dict[str, object] = {"at": to_iso(FIXED_NOW)}
    fields.update(kwargs)
    return MetricSample(name=name, value=value, **fields)  # type: ignore[arg-type]


def write_corpus(root: Path, files: Mapping[str, str] | None = None) -> Path:
    """把语料写进 ``root`` 并返回它（缺省三份，顺序与字典一致）."""
    resolved = dict(files if files is not None else CORPUS_FILES)
    root.mkdir(parents=True, exist_ok=True)
    for name, body in resolved.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    return root


def isolated_indexer(work: Path, *, backend=None, batch_size: int = 8) -> IndexingPipeline:
    """装配一条**完全隔离**的索引流水线（清单、版本表、备份都写在 ``work`` 下）.

    它回答的是"为什么不能直接用 ``default_indexing_pipeline()``"：
    那个工厂的版本表路径来自 ``settings.indexing_dir``，于是在测试里
    ``versions.persist()`` 会在仓库根目录建出 ``data/index``。
    """
    resolved_backend = backend if backend is not None else create_backend(
        backend="flat", metric="cosine", path=""
    )
    embedding = default_embedding()
    identity = describe_embedding(embedding)
    directory = work / "index"
    versions = IndexVersionStore(path=str(directory))
    backups = IndexBackupStore(path=str(directory / "backups"))
    cache = EmbeddingCache(
        path="",
        provider=identity.provider,
        model=identity.model,
        dimension=identity.dimension,
    )
    builder = IndexBuilder(
        resolved_backend,
        embedding,
        cache=cache,
        identity=identity,
        batch_size=batch_size,
        versions=versions,
        backups=backups,
    )
    return IndexingPipeline(builder, versions=versions, backups=backups)


def build_pipeline(
    tmp_path: Path,
    *,
    files: Mapping[str, str] | None = None,
    clock: FakeClock | None = None,
    source_dir: str | None = None,
    policy_value: SchedulePolicy | None = None,
    ledger_name: str = LEDGER_FILENAME,
    report_name: str = REPORT_FILENAME,
    persist: bool = True,
    strict: bool = True,
    limit: int | None = None,
    quality_loader=None,
) -> RagOpsPipeline:
    """装配一条隔离的运维流水线（账本 / 报告 / 索引全在 ``tmp_path`` 下）."""
    work = Path(tmp_path)
    work.mkdir(parents=True, exist_ok=True)
    corpus = (
        Path(source_dir)
        if source_dir is not None
        else write_corpus(work / "corpus", files)
    )
    backend = create_backend(backend="flat", metric="cosine", path="")
    indexer = isolated_indexer(work, backend=backend)
    return RagOpsPipeline(
        indexer,
        store=backend,
        source_dir=str(corpus),
        policy=policy_value if policy_value is not None else policy(),
        ledger_path=str(work / ledger_name) if ledger_name else "",
        report_path=str(work / report_name) if report_name else "",
        persist_path=str(work / "index" / "vectors.json") if persist else "",
        versions=indexer.versions,
        quality_loader=quality_loader,
        limit=limit,
        strict=strict,
        clock=clock if clock is not None else FakeClock(FIXED_NOW),
    )


__all__ = [
    "CORPUS_FILES",
    "FIXED_NOW",
    "SINGLE_FILE",
    "FakeClock",
    "build_pipeline",
    "entry",
    "finding",
    "fingerprint",
    "health_report",
    "isolated_indexer",
    "ledger",
    "plan",
    "policy",
    "ran_ledger",
    "sample",
    "snapshot",
    "write_corpus",
]
