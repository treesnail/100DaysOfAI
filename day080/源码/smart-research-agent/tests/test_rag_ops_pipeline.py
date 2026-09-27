"""``rag_ops.pipeline`` 与 ``/rag/ops/*``：一次同步的全过程与三个端点（day072）.

本文件是这一课的**端到端**用例：一条真实（但完全离线）的流水线跑四次——
首次全量、当天第二次（未到点）、次日（没变）、改一份文档之后（增量）——
再把同样的四次通过 HTTP 端点跑一遍，确认两条入口给出同一份结论。

```text
run_once 的四档状态：not_due / no_change / synced / failed
两个一致性的锚点：  "评估对象 = 服务对象"（同一个库）与"空转不覆盖证据"
```
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from smart_research_agent.api.app import create_app
from smart_research_agent.llm.embedding import default_embedding
from smart_research_agent.rag_ops.errors import SyncError
from smart_research_agent.rag_ops.pipeline import (
    LEDGER_FILENAME,
    REPORT_FILENAME,
    default_rag_ops_pipeline,
    indexer_versions,
    paths_in,
    policy_from_settings,
    quality_from_report,
)
from smart_research_agent.rag_ops.types import (
    STATUS_FAILED,
    STATUS_NOT_DUE,
    STATUS_NO_CHANGE,
    STATUS_SYNCED,
    SyncLedger,
    to_iso,
)
from tests.rag_debug_samples import BoomLLM, metrics_row
from tests.rag_ops_samples import (
    CORPUS_FILES,
    FIXED_NOW,
    FakeClock,
    build_pipeline,
    policy,
    write_corpus,
)


class TestRunOnceHappyPath:
    """四次真实运行：每一次的状态、差集、落盘与监控都要对得上."""

    def test_first_run_builds_everything(self, tmp_path: Path) -> None:
        pipeline = build_pipeline(tmp_path)
        report = pipeline.run_once()
        assert report.status == STATUS_SYNCED
        assert report.plan is not None and report.plan.added == (
            "ops.md",
            "rag.md",
            "sched.md",
        )
        assert report.index is not None and report.index["written"] > 0
        assert report.changed == 3
        assert report.metric("sources_changed") == 3.0
        assert report.health is not None and report.health.findings[0].state == "ok"
        assert (tmp_path / "index" / "vectors.json").exists()
        assert (tmp_path / LEDGER_FILENAME).exists()
        assert (tmp_path / REPORT_FILENAME).exists()
        ledger = SyncLedger.load(tmp_path / LEDGER_FILENAME)
        assert ledger.has_watermark and ledger.last_index_version

    def test_second_run_within_the_interval_is_not_due(self, tmp_path: Path) -> None:
        pipeline = build_pipeline(tmp_path)
        pipeline.run_once()
        first = (tmp_path / REPORT_FILENAME).read_text(encoding="utf-8")
        report = pipeline.run_once()
        assert report.status == STATUS_NOT_DUE
        assert report.plan is None and report.health is None
        assert "未执行" in report.notes[0]
        # 空转**不覆盖**上一次落盘的报告：值班的人要看到的是"上次同步成没成"
        assert (tmp_path / REPORT_FILENAME).read_text(encoding="utf-8") == first

    def test_next_day_without_changes_writes_nothing(self, tmp_path: Path) -> None:
        clock = FakeClock()
        pipeline = build_pipeline(tmp_path, clock=clock)
        pipeline.run_once()
        clock.advance(days=1, minutes=30)
        report = pipeline.run_once()
        assert report.status == STATUS_NO_CHANGE
        assert report.index is None
        assert report.plan is not None and report.plan.changed_count == 0
        assert report.metric("chunks_written") == 0.0
        assert "一行都不会写" in " ".join(report.notes)

    def test_edited_file_is_the_only_thing_rebuilt(self, tmp_path: Path) -> None:
        clock = FakeClock()
        corpus = write_corpus(tmp_path / "corpus", CORPUS_FILES)
        pipeline = build_pipeline(tmp_path, clock=clock, source_dir=str(corpus))
        pipeline.run_once()
        (corpus / "ops.md").write_text(
            "# 运维\n\n索引需要定期增量更新。\n\n## 调度\n\n每天凌晨跑一次。\n",
            encoding="utf-8",
        )
        clock.advance(days=1, hours=1)  # 抖动最多 5 分钟，因此"整一天"还差一点
        report = pipeline.run_once()
        assert report.status == STATUS_SYNCED
        assert report.plan is not None
        assert report.plan.updated == ("ops.md",)
        assert report.plan.unchanged == ("rag.md", "sched.md")
        assert report.plan.churn_ratio == pytest.approx(1 / 3, abs=1e-6)

    def test_force_ignores_the_schedule(self, tmp_path: Path) -> None:
        pipeline = build_pipeline(tmp_path)
        pipeline.run_once()
        report = pipeline.run_once(force=True)
        assert report.status == STATUS_NO_CHANGE
        assert report.schedule.mode == "forced"
        assert "force=True" in report.schedule.reason

    def test_notes_explain_persistence(self, tmp_path: Path) -> None:
        pipeline = build_pipeline(tmp_path, persist=False)
        report = pipeline.run_once()
        assert any("未落盘" in note for note in report.notes)
        assert any("版本表已落盘" in note for note in report.notes)


class TestRunOnceFailures:
    """失败：水位不动、原因留下、报告照发；修好之后能自动恢复."""

    def test_missing_corpus_fails_without_advancing_the_watermark(self, tmp_path: Path) -> None:
        pipeline = build_pipeline(tmp_path, source_dir=str(tmp_path / "nope"))
        report = pipeline.run_once()
        assert report.status == STATUS_FAILED
        assert "nope" in report.error and report.health is None
        assert report.metric("run_failures") == 1.0
        ledger = SyncLedger.load(tmp_path / LEDGER_FILENAME)
        assert ledger.has_watermark is False
        assert ledger.consecutive_failures == 1
        assert ledger.last_error == report.error

    def test_a_later_success_clears_the_failure(self, tmp_path: Path) -> None:
        work = tmp_path / "corpus"
        clock = FakeClock(FIXED_NOW)
        pipeline = build_pipeline(tmp_path, source_dir=str(work), clock=clock)
        assert pipeline.run_once().status == STATUS_FAILED
        write_corpus(work, CORPUS_FILES)
        clock.advance(minutes=30)  # 退避基数 60 秒 → 半小时后早已出了重试窗口
        report = pipeline.run_once()
        assert report.status == STATUS_SYNCED
        ledger = SyncLedger.load(tmp_path / LEDGER_FILENAME)
        assert ledger.consecutive_failures == 0 and ledger.last_error == ""

    def test_failed_run_in_a_backoff_window_does_not_retry(self, tmp_path: Path) -> None:
        pipeline = build_pipeline(tmp_path, source_dir=str(tmp_path / "nope"))
        pipeline.run_once()
        again = pipeline.run_once()
        assert again.status == STATUS_NOT_DUE
        assert again.schedule.mode == "backoff"

    def test_broken_ledger_is_raised_not_swallowed(self, tmp_path: Path) -> None:
        pipeline = build_pipeline(tmp_path)
        (tmp_path / LEDGER_FILENAME).write_text("{ broken", encoding="utf-8")
        with pytest.raises(SyncError):
            pipeline.run_once()


class TestReadOnlyViews:
    """只读视图：预览、体检、状态都不许留下副作用."""

    def test_plan_preview_writes_nothing(self, tmp_path: Path) -> None:
        pipeline = build_pipeline(tmp_path)
        preview = pipeline.plan()
        assert preview.added == ("ops.md", "rag.md", "sched.md")
        assert not (tmp_path / LEDGER_FILENAME).exists()
        assert not (tmp_path / REPORT_FILENAME).exists()
        assert pipeline.snapshot().count == 3

    def test_manifest_is_unknown_before_the_build(self, tmp_path: Path) -> None:
        pipeline = build_pipeline(tmp_path)
        assert pipeline.current_manifest() is None
        pipeline.run_once()
        manifest = pipeline.current_manifest()
        assert manifest is not None and manifest.count > 0
        assert manifest.version_id == SyncLedger.load(tmp_path / LEDGER_FILENAME).last_index_version

    def test_status_exposes_four_blocks(self, tmp_path: Path) -> None:
        pipeline = build_pipeline(tmp_path, clock=FakeClock(FIXED_NOW))
        payload = pipeline.status()
        assert set(payload) == {
            "source_dir",
            "ledger_path",
            "report_path",
            "index",
            "schedule",
            "metrics",
            "deploy",
            "last_report",
        }
        assert payload["schedule"]["decision"]["mode"] == "first_run"
        assert payload["last_report"] is None
        assert payload["deploy"]["valid"] is True
        assert payload["deploy"]["compose_drift"] == ""
        pipeline.run_once()
        assert pipeline.status()["last_report"]["status"] == STATUS_SYNCED
        assert pipeline.status()["metrics"]

    def test_health_report_is_cheap_and_read_only(self, tmp_path: Path) -> None:
        pipeline = build_pipeline(tmp_path)
        report = pipeline.health_report(now=FIXED_NOW)
        assert report.exit_code == 1  # 库是空的
        assert not (tmp_path / LEDGER_FILENAME).exists()


class TestQualityAndAssembly:
    """质量来源与装配：报告读不出来只影响质量那一项，不挡住整条链路."""

    def test_quality_from_a_missing_report_is_none(self, tmp_path: Path) -> None:
        assert quality_from_report(tmp_path / "none.json") is None

    def test_quality_from_a_broken_report_is_none(self, tmp_path: Path) -> None:
        target = tmp_path / "broken.json"
        target.write_text("{ broken", encoding="utf-8")
        assert quality_from_report(target) is None
        target.write_text("[1, 2]", encoding="utf-8")
        assert quality_from_report(target) is None
        target.write_text(json.dumps({"no_metrics": True}), encoding="utf-8")
        assert quality_from_report(target) is None

    def test_quality_from_a_real_report(self, tmp_path: Path) -> None:
        target = tmp_path / "report.json"
        target.write_text(
            json.dumps(
                {
                    "label": "baseline:baseline",
                    "cases": 6,
                    "k": 3,
                    "metrics": metrics_row(retrieval_recall=0.8),
                    "prompt_version": "v2",
                    "index_version": "",
                    "fallback_counts": {"": 6},
                    "tag_counts": {},
                    "p95_latency_ms": 1.0,
                    "mean_latency_ms": 0.8,
                }
            ),
            encoding="utf-8",
        )
        baseline = quality_from_report(target)
        assert baseline is not None
        assert baseline.value("retrieval_recall") == 0.8
        assert baseline.samples == 6

    def test_quality_loader_feeds_metrics_and_health(self, tmp_path: Path) -> None:
        target = tmp_path / "report.json"
        target.write_text(
            json.dumps(
                {
                    "label": "now",
                    "cases": 6,
                    "k": 3,
                    "metrics": metrics_row(),
                }
            ),
            encoding="utf-8",
        )
        pipeline = build_pipeline(
            tmp_path, quality_loader=lambda: quality_from_report(target)
        )
        report = pipeline.run_once()
        assert report.metric("quality_retrieval_recall") == 0.5
        assert report.health is not None
        # 这份质量报告对照的是**仓库里那份基线**（settings.rag_eval_baseline_path，
        # 这是产线的行为）：0.5 的召回率相对基线的 0.9167 是明确退化 → fail。
        quality = report.health.by_check()["quality_regression"]
        assert quality.state == "fail"
        assert quality.detail["regressions"]
        assert quality.detail["baseline_exists"] is True

    def test_paths_in_helpers(self) -> None:
        assert paths_in("") == ("", "")
        assert paths_in("data/ops") == (
            str(Path("data/ops") / LEDGER_FILENAME),
            str(Path("data/ops") / REPORT_FILENAME),
        )

    def test_policy_from_settings_reads_the_rag_ops_group(self) -> None:
        from smart_research_agent.config import settings

        built = policy_from_settings()
        assert built.interval_minutes == settings.rag_ops_interval_minutes
        assert built.backoff_max_seconds == settings.rag_ops_backoff_max_seconds

    def test_default_pipeline_is_assembled_from_settings(self) -> None:
        from smart_research_agent.config import settings

        pipeline = default_rag_ops_pipeline()
        assert pipeline.source_dir == settings.rag_ops_source_dir
        assert pipeline.ledger_path == settings.rag_ops_ledger_path
        assert pipeline.report_path == settings.rag_ops_report_path
        assert pipeline.store is not None
        # 版本表与构建器手里的是**同一份对象**（两个实例会让清单永远读不到）
        assert indexer_versions(pipeline) is indexer_versions(pipeline)

    def test_indexer_versions_falls_back_to_none(self) -> None:
        class _Bare:
            """没有 versions 属性的替身（模拟"没接版本表"的流水线）."""

        assert indexer_versions(_Bare()) is None  # type: ignore[arg-type]


def make_client(pipeline, *, embedding=None) -> TestClient:
    """造一个只注入运维流水线的 app（LLM 用"一调就炸"的替身证明不调模型）."""
    app = create_app(
        llm=BoomLLM(),
        embedding=embedding if embedding is not None else default_embedding(),
        vector_store=pipeline.store,
        rag_ops_pipeline=pipeline,
    )
    return TestClient(app)


class TestOpsStatusEndpoint:
    """GET /rag/ops/status：四块内容 + 0 次模型调用."""

    def test_reports_the_four_blocks(self, tmp_path: Path) -> None:
        pipeline = build_pipeline(tmp_path)
        with make_client(pipeline) as client:
            payload = client.get("/rag/ops/status").json()
        assert payload["summary"]
        assert payload["source_dir"] == pipeline.source_dir
        assert payload["ledger_path"] == pipeline.ledger_path
        assert payload["schedule"]["decision"]["mode"] == "first_run"
        assert payload["deploy"]["valid"] is True
        assert payload["index"]["count"] == 0
        assert payload["last_report"] is None
        assert payload["metrics"] == []
        json.dumps(payload)

    def test_last_report_appears_after_a_sync(self, tmp_path: Path) -> None:
        pipeline = build_pipeline(tmp_path)
        with make_client(pipeline) as client:
            assert client.post("/rag/ops/sync", json={}).json()["status"] == STATUS_SYNCED
            payload = client.get("/rag/ops/status").json()
        assert payload["last_report"]["status"] == STATUS_SYNCED
        assert payload["metrics"]
        assert payload["index"]["count"] > 0

    def test_missing_pipeline_is_a_503_with_a_way_out(self, tmp_path: Path) -> None:
        pipeline = build_pipeline(tmp_path)
        with make_client(pipeline) as client:
            client.app.state.rag_ops_pipeline = None
            response = client.get("/rag/ops/status")
        assert response.status_code == 503
        assert "rag_ops_pipeline" in response.json()["detail"]


class TestOpsHealthEndpoint:
    """GET /rag/ops/health：探针用它，因此 fail 必须是 503."""

    def test_empty_store_is_a_503(self, tmp_path: Path) -> None:
        pipeline = build_pipeline(tmp_path)
        with make_client(pipeline) as client:
            response = client.get("/rag/ops/health")
        assert response.status_code == 503
        payload = response.json()
        assert payload["state"] == "fail" and payload["exit_code"] == 1
        assert len(payload["findings"]) == 4
        assert payload["rules"] and all(":" in item for item in payload["rules"])

    def test_after_a_sync_it_is_a_200_with_warnings(self, tmp_path: Path) -> None:
        pipeline = build_pipeline(tmp_path)
        with make_client(pipeline) as client:
            client.post("/rag/ops/sync", json={"force": True})
            response = client.get("/rag/ops/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["state"] == "warn" and payload["ok"] is True
        assert payload["exit_code"] == 0
        checks = {item["check"]: item["state"] for item in payload["findings"]}
        assert checks["vector_store"] == "ok"
        assert checks["quality_regression"] == "warn"


class TestOpsSyncEndpoint:
    """POST /rag/ops/sync：四档状态全在 200 的响应体里，400 只给"跑不了"."""

    def test_first_sync_then_not_due(self, tmp_path: Path) -> None:
        pipeline = build_pipeline(tmp_path)
        with make_client(pipeline) as client:
            first = client.post("/rag/ops/sync", json={})
            second = client.post("/rag/ops/sync", json={})
        assert first.status_code == 200
        payload = first.json()
        assert payload["status"] == STATUS_SYNCED
        assert payload["report"]["index"]["written"] > 0
        assert payload["persisted"]["ledger"] == pipeline.ledger_path
        assert second.json()["status"] == STATUS_NOT_DUE
        json.dumps(payload)

    def test_force_reruns_immediately(self, tmp_path: Path) -> None:
        pipeline = build_pipeline(tmp_path)
        with make_client(pipeline) as client:
            client.post("/rag/ops/sync", json={"force": True})
            payload = client.post("/rag/ops/sync", json={"force": True}).json()
        assert payload["status"] == STATUS_NO_CHANGE
        assert payload["report"]["schedule"]["mode"] == "forced"

    def test_broken_ledger_is_a_400(self, tmp_path: Path) -> None:
        pipeline = build_pipeline(tmp_path)
        (tmp_path / LEDGER_FILENAME).write_text("{ broken", encoding="utf-8")
        with make_client(pipeline) as client:
            response = client.post("/rag/ops/sync", json={})
        assert response.status_code == 400
        assert "坏账本" in response.json()["detail"]

    def test_missing_pipeline_is_a_503(self, tmp_path: Path) -> None:
        pipeline = build_pipeline(tmp_path)
        with make_client(pipeline) as client:
            client.app.state.rag_ops_pipeline = None
            response = client.post("/rag/ops/sync", json={})
        assert response.status_code == 503

    def test_persisted_paths_are_empty_without_a_directory(self, tmp_path: Path) -> None:
        pipeline = build_pipeline(tmp_path, ledger_name="", report_name="")
        with make_client(pipeline) as client:
            payload = client.post("/rag/ops/sync", json={"force": True}).json()
        assert payload["persisted"]["ledger"] == ""
        assert payload["persisted"]["report"] == ""
        assert not (tmp_path / LEDGER_FILENAME).exists()
