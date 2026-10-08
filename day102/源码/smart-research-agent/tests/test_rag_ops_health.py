"""``rag_ops.health`` 与 ``rag_ops.metrics``：能不能服务、量到了什么（day072）.

本文件钉住三条纪律：

```text
① 探活不跑评估    四项检查只读"别人已经算好的账"（库计数 / 清单 / 账本 / 基线）
② 跳过不是通过    缺清单、缺基线、没评估一律 warn，绝不因为"没查成"而报 ok
③ 只有骗人的事才 fail  库空 / 清单与库不符 / 质量判定回归 —— 这三种才摘流量
```
"""

from __future__ import annotations

from pathlib import Path

import pytest

from smart_research_agent.indexing import build_manifest, describe_embedding
from smart_research_agent.indexing.manifest import manifest_from_store, verify_index
from smart_research_agent.indexing.types import IndexEntry
from smart_research_agent.llm.embedding import default_embedding
from smart_research_agent.rag_debug.baseline import RagBaseline, RagBaselineGuard
from smart_research_agent.rag_ops.errors import HealthError, MetricsError
from smart_research_agent.rag_ops.health import (
    build_health_report,
    check_corpus_freshness,
    check_index_integrity,
    check_quality_regression,
    check_vector_store,
)
from smart_research_agent.rag_ops.metrics import (
    QUALITY_SAMPLE_SOURCES,
    MetricsBuffer,
    collect_samples,
    samples_from_ledger,
    samples_from_plan,
    samples_from_quality,
    samples_from_store,
)
from smart_research_agent.rag_ops.types import (
    CHECK_CORPUS_FRESHNESS,
    CHECK_INDEX_INTEGRITY,
    CHECK_QUALITY_REGRESSION,
    CHECK_VECTOR_STORE,
    HEALTH_CHECKS,
    METRIC_CHUNKS_WRITTEN,
    METRIC_INDEX_RECORDS,
    METRIC_QUALITY_HALLUCINATION,
    METRIC_RUN_FAILURES,
    METRIC_SOURCES_CHANGED,
    METRIC_SYNC_AGE_MINUTES,
    STATE_FAIL,
    STATE_OK,
    STATE_WARN,
    to_iso,
)
from smart_research_agent.vectorstore.registry import create_backend
from smart_research_agent.vectorstore.types import make_record
from tests.rag_debug_samples import metrics_row
from tests.rag_ops_samples import (
    FIXED_NOW,
    build_pipeline,
    fingerprint,
    ledger,
    plan,
    ran_ledger,
    sample,
)


def _store(*ids: str):
    """造一个装了若干条记录的 flat 库（维度由编码器在第一次 upsert 时定下）.

    注意 ``EmbeddingProvider.embed`` 的入参是**一段文本**而不是一批文本
    （批处理在 ``indexing.BatchEncoder`` 那一层），因此这里逐条编码。
    """
    store = create_backend(backend="flat", metric="cosine", path="")
    if not ids:
        return store
    provider = default_embedding()
    store.upsert(
        [
            make_record(record_id=name, vector=provider.embed(name), text=name)
            for name in ids
        ]
    )
    return store


class TestVectorStoreCheck:
    """库空不空：对外服务的实例空库是 fail，灰度实例可以放开."""

    def test_empty_store_fails_by_default(self) -> None:
        result = check_vector_store(_store())
        assert result.state == STATE_FAIL and result.check == CHECK_VECTOR_STORE
        assert "空" in result.message and result.detail["count"] == 0

    def test_empty_store_is_only_a_warning_when_allowed(self) -> None:
        result = check_vector_store(_store(), require_non_empty=False)
        assert result.state == STATE_WARN and "灰度" in result.message

    def test_populated_store_is_ok(self) -> None:
        result = check_vector_store(_store("r1", "r2"))
        assert result.state == STATE_OK and result.detail["count"] == 2
        assert result.detail["backend"] == "flat"


class TestIndexIntegrityCheck:
    """清单与库对不对得上：没有清单只能给 warn（"没发现矛盾"不是"库是对的"）."""

    def test_without_manifest_it_warns(self) -> None:
        result = check_index_integrity(None, _store("r1"))
        assert result.state == STATE_WARN and result.detail["count_store"] == 1

    def test_matching_manifest_is_ok(self) -> None:
        store = _store("r1", "r2")
        identity = describe_embedding(default_embedding())
        manifest = manifest_from_store(store, identity=identity)
        assert verify_index(manifest, store)["ok"] is True
        result = check_index_integrity(manifest, store)
        assert result.state == STATE_OK
        assert result.detail["version_id"] == manifest.version_id

    def test_manifest_with_a_missing_record_fails(self) -> None:
        store = _store("r1")
        identity = describe_embedding(default_embedding())
        manifest = build_manifest(
            (
                IndexEntry(
                    record_id="r1",
                    fingerprint=fingerprint("r1"),
                    vector_key=fingerprint("r1-vector"),
                ),
                IndexEntry(
                    record_id="ghost",
                    fingerprint=fingerprint("ghost"),
                    vector_key=fingerprint("ghost-vector"),
                ),
            ),
            identity=identity,
            metric="cosine",
            backend="flat",
        )
        result = check_index_integrity(manifest, store)
        assert result.state == STATE_FAIL
        assert "ghost" in result.message
        assert result.detail["problems"]


class TestCorpusFreshnessCheck:
    """新鲜度：用 last_success_at 而不是 last_run_at（"每小时都失败"看起来也很勤快）."""

    def test_without_ledger_it_warns(self) -> None:
        result = check_corpus_freshness(None, now=FIXED_NOW)
        assert result.state == STATE_WARN

    def test_never_succeeded_warns(self) -> None:
        result = check_corpus_freshness(ledger(), now=FIXED_NOW)
        assert result.state == STATE_WARN and "从未成功同步" in result.message
        assert result.detail["age_minutes"] is None

    def test_fresh_ledger_is_ok(self) -> None:
        state = ran_ledger(moment=FIXED_NOW)
        result = check_corpus_freshness(state, now=FIXED_NOW)
        assert result.state == STATE_OK and result.detail["age_minutes"] == 0.0

    def test_stale_ledger_warns_and_quotes_the_failure(self) -> None:
        state = ledger(
            None,
            last_run_at=to_iso(FIXED_NOW),
            last_success_at=to_iso(FIXED_NOW),
            consecutive_failures=2,
            runs=3,
            failures=2,
            last_error="编码器挂了",
        )
        result = check_corpus_freshness(
            state, max_age_minutes=60, now=FIXED_NOW.replace(hour=6)
        )
        assert result.state == STATE_WARN
        assert "编码器挂了" in result.message and "连续失败 2 次" in result.message

    def test_negative_threshold_is_rejected(self) -> None:
        with pytest.raises(HealthError):
            check_corpus_freshness(ledger(), max_age_minutes=-1, now=FIXED_NOW)


class TestQualityRegressionCheck:
    """质量：读 day071 的结论，缺输入一律 warn（**跳过不等于通过**）."""

    def _baseline(self, **overrides: float) -> RagBaseline:
        return RagBaseline(
            label="baseline:baseline",
            samples=6,
            k=3,
            metrics=metrics_row(**overrides),
        )

    def test_without_evaluation_it_warns(self, tmp_path: Path) -> None:
        result = check_quality_regression(None, baseline_path=tmp_path / "base.json")
        assert result.state == STATE_WARN and "最近没有评估结果" in result.message

    def test_without_baseline_file_it_warns(self, tmp_path: Path) -> None:
        result = check_quality_regression(
            self._baseline(), baseline_path=tmp_path / "base.json"
        )
        assert result.state == STATE_WARN and "没有质量基线文件" in result.message

    def test_regression_fails(self) -> None:
        committed = self._baseline()
        degraded = self._baseline(retrieval_recall=0.2)
        comparison = RagBaselineGuard(committed, min_samples=3).compare(degraded)
        assert comparison.regressions
        result = check_quality_regression(None, comparison=comparison)
        assert result.state == STATE_FAIL
        assert "退化" in result.message
        assert result.detail["regressions"]

    def test_inconclusive_comparison_warns(self) -> None:
        committed = self._baseline()
        other_k = RagBaseline(label="now", samples=6, k=5, metrics=metrics_row())
        comparison = RagBaselineGuard(committed, min_samples=3).compare(other_k)
        assert comparison.conclusive is False
        result = check_quality_regression(None, comparison=comparison)
        assert result.state == STATE_WARN and "不可比" in result.message

    def test_comparable_and_equal_is_ok(self) -> None:
        committed = self._baseline()
        comparison = RagBaselineGuard(committed, min_samples=3).compare(self._baseline())
        result = check_quality_regression(None, comparison=comparison)
        assert result.state == STATE_OK

    def test_missing_baseline_file_on_disk_is_read_from_settings(self, tmp_path: Path) -> None:
        # 不传 comparison 且当前基线为空 → 直接给"没有评估结果"的 warn（不读盘）
        result = check_quality_regression(None, baseline_path=tmp_path / "none.json")
        assert result.state == STATE_WARN


class TestHealthReport:
    """四项检查的装订：顺序固定、总状态取最坏、退出码随之而变."""

    def test_warns_but_stays_alive(self, tmp_path: Path) -> None:
        pipeline = build_pipeline(tmp_path)
        report = pipeline.health_report(now=FIXED_NOW)
        assert [item.check for item in report.findings] == list(HEALTH_CHECKS)
        assert report.state == STATE_FAIL  # 还没同步：库是空的
        assert report.exit_code == 1
        assert report.findings[0].check == CHECK_VECTOR_STORE

    def test_after_a_sync_the_report_turns_green_except_quality(
        self, tmp_path: Path
    ) -> None:
        pipeline = build_pipeline(tmp_path)
        pipeline.run_once()
        report = pipeline.health_report()
        by_check = report.by_check()
        assert by_check[CHECK_VECTOR_STORE].state == STATE_OK
        assert by_check[CHECK_INDEX_INTEGRITY].state == STATE_OK
        assert by_check[CHECK_CORPUS_FRESHNESS].state == STATE_OK
        assert by_check[CHECK_QUALITY_REGRESSION].state == STATE_WARN
        assert report.state == STATE_WARN and report.exit_code == 0

    def test_build_health_report_accepts_explicit_inputs(self, tmp_path: Path) -> None:
        store = _store("r1")
        report = build_health_report(
            store,
            manifest=None,
            ledger=None,
            quality=None,
            require_non_empty=True,
            now=FIXED_NOW,
        )
        assert len(report.findings) == len(HEALTH_CHECKS)
        assert report.checked_at == to_iso(FIXED_NOW)


class TestMetricExtractors:
    """采样器：四个来源各自只报自己量得到的东西（缺输入就不出现）."""

    def test_plan_samples_carry_both_granularities(self) -> None:
        built = plan(added=("docs/a.md",), unchanged=("docs/b.md",))
        samples = samples_from_plan(built, at=FIXED_NOW, seconds=1.5)
        values = {item.name: item.value for item in samples}
        assert values[METRIC_SOURCES_CHANGED] == 1
        assert values[METRIC_CHUNKS_WRITTEN] == 0  # 没有构建报告 → 0 块
        assert len(samples) == 4

    def test_plan_samples_reject_negative_seconds(self) -> None:
        with pytest.raises(MetricsError):
            samples_from_plan(plan(), seconds=-1.0)

    def test_store_samples_carry_backend_labels(self) -> None:
        samples = samples_from_store(_store("r1", "r2"), at=FIXED_NOW)
        assert samples[0].name == METRIC_INDEX_RECORDS
        assert samples[0].value == 2.0
        assert dict(samples[0].labels) == {"backend": "flat", "metric": "cosine"}

    def test_ledger_samples_mark_the_missing_watermark(self) -> None:
        fresh = samples_from_ledger(ledger(), now=FIXED_NOW, at=FIXED_NOW)
        assert {item.name for item in fresh} == {METRIC_SYNC_AGE_MINUTES, METRIC_RUN_FAILURES}
        assert fresh[0].value == 0.0
        assert dict(fresh[0].labels) == {"has_watermark": "false"}

    def test_quality_samples_are_copied_not_recomputed(self) -> None:
        quality = RagBaseline(
            label="baseline:baseline",
            samples=6,
            k=3,
            metrics=metrics_row(retrieval_recall=0.75, hallucination_rate=0.25),
        )
        samples = samples_from_quality(quality, at=FIXED_NOW)
        assert {item.name for item in samples} == set(QUALITY_SAMPLE_SOURCES)
        values = {item.name: item.value for item in samples}
        assert values["quality_retrieval_recall"] == 0.75
        assert values[METRIC_QUALITY_HALLUCINATION] == 0.25
        assert dict(samples[0].labels)["label"] == "baseline:baseline"

    def test_quality_samples_are_empty_without_an_evaluation(self) -> None:
        assert samples_from_quality(None) == ()

    def test_quality_samples_reject_a_missing_metric(self) -> None:
        quality = RagBaseline(
            label="x", samples=6, k=3, metrics=metrics_row()
        )
        # 直接改坏那份账（真实场景是别人写了一份不全的报告）：
        # 缺一项时必须报错，而不是用 0 顶上——"没量过"与"量出来是 0"是两回事。
        del quality.metrics["retrieval_recall"]
        with pytest.raises(MetricsError):
            samples_from_quality(quality, at=FIXED_NOW)

    def test_collect_samples_respects_input_order_and_absence(self) -> None:
        built = plan(added=("docs/a.md",), unchanged=("docs/b.md",))
        everything = collect_samples(
            at=FIXED_NOW,
            now=FIXED_NOW,
            plan=built,
            index_report=None,
            seconds=2.0,
            store=_store("r1"),
            ledger=ran_ledger(moment=FIXED_NOW),
            quality=None,
        )
        assert [item.name for item in everything] == [
            "sources_total",
            "sources_changed",
            "chunks_written",
            "sync_seconds",
            METRIC_INDEX_RECORDS,
            METRIC_SYNC_AGE_MINUTES,
            METRIC_RUN_FAILURES,
        ]
        # 什么输入都不给 → 什么都不报（"没量到"不写成 0）
        assert collect_samples() == ()


class TestMetricsBuffer:
    """内存缓冲：按指标分组、汇总表只列出现过的指标、清空只清这一层."""

    def test_record_and_describe(self) -> None:
        buffer = MetricsBuffer()
        assert buffer.extend((sample("sources_total", 2.0), sample("chunks_written", 5.0))) == 2
        assert buffer.record(sample("sources_total", 3.0, at=to_iso(FIXED_NOW))) is not None
        assert buffer.names() == ("sources_total", METRIC_CHUNKS_WRITTEN)
        assert len(buffer) == 3
        assert buffer.latest("sources_total") is not None
        assert buffer.series("sources_total").count == 2
        rows = buffer.describe()
        assert [row["name"] for row in rows] == ["sources_total", METRIC_CHUNKS_WRITTEN]
        assert rows[0]["unit"] == "count" and rows[0]["latest"] == 3.0
        assert buffer.to_dict()["count"] == 3

    def test_empty_buffer_reports_nothing(self) -> None:
        buffer = MetricsBuffer()
        assert buffer.names() == () and buffer.describe() == []

    def test_unknown_metric_is_rejected(self) -> None:
        buffer = MetricsBuffer()
        with pytest.raises(MetricsError):
            buffer.samples("custom_metric")
        with pytest.raises(MetricsError):
            buffer.record("not a sample")  # type: ignore[arg-type]

    def test_clear_only_touches_this_copy(self) -> None:
        buffer = MetricsBuffer((sample("sources_total", 1.0),))
        assert buffer.clear() == 1 and len(buffer) == 0

    def test_constructor_accepts_samples(self) -> None:
        buffer = MetricsBuffer((sample("run_failures", 2.0),))
        assert buffer.latest(METRIC_RUN_FAILURES) is not None
        assert buffer.latest("chunks_written") is None
