"""``rag_ops.types`` 的形状校验与封闭表（day072）.

全部离线、零网络、零文件（除了两处显式的 save/load 用例用 tmp_path）。
本文件钉住四类性质：

```text
① 六张封闭表逐键对齐（少一个键 = 某一类在报告里只有名字、没有解释）
② 每一个形状都在构造期拒绝"看起来像、其实不对"的输入（绝不静默兜底）
③ 时间戳一律秒级 UTC：微秒被砍掉、时区被抹平、空串不是时间
④ 派生量（快照号 / 变更比例 / 最坏状态 / 退出码）与我们写下的定义逐位一致
```
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from smart_research_agent.rag_ops.errors import HealthError, MetricsError, ScheduleError, SyncError
from smart_research_agent.rag_ops.types import (
    ACTION_ADDED,
    ACTION_REMOVED,
    ACTION_UNCHANGED,
    ACTION_UPDATED,
    CHECK_CORPUS_FRESHNESS,
    HEALTH_CHECKS,
    HEALTH_CHECK_DESCRIPTIONS,
    HEALTH_STATES,
    HEALTH_STATE_DESCRIPTIONS,
    HEALTH_STATE_EXIT_CODES,
    METRIC_UNITS,
    OPS_METRICS,
    OPS_METRIC_DESCRIPTIONS,
    OPS_METRIC_UNITS,
    OPS_STATUSES,
    OPS_STATUS_DESCRIPTIONS,
    SCHEDULE_MODES,
    SCHEDULE_MODE_DESCRIPTIONS,
    STATE_FAIL,
    STATE_OK,
    STATE_WARN,
    STATUS_FAILED,
    STATUS_NOT_DUE,
    STATUS_NO_CHANGE,
    STATUS_SYNCED,
    SYNC_ACTIONS,
    SYNC_ACTION_DESCRIPTIONS,
    SYNC_ACTION_IS_CHANGE,
    CorpusSnapshot,
    HealthFinding,
    HealthReport,
    MetricSample,
    MetricSeries,
    OpsReport,
    ScheduleDecision,
    SchedulePolicy,
    SourceEntry,
    SyncLedger,
    SyncPlan,
    minutes_between,
    parse_iso,
    shift_iso,
    snapshot_from_entries,
    snapshot_id,
    to_iso,
    utc_now,
    worst_state,
)
from tests.rag_ops_samples import (
    FIXED_NOW,
    entry,
    finding,
    fingerprint,
    health_report,
    ledger,
    plan,
    policy,
    ran_ledger,
    sample,
    snapshot,
)


class TestClosedTables:
    """六张封闭表必须逐键对齐（不一致就该在导入期炸掉，而不是在报告里留空洞）."""

    def test_sync_action_tables_align(self) -> None:
        assert set(SYNC_ACTIONS) == set(SYNC_ACTION_DESCRIPTIONS) == set(SYNC_ACTION_IS_CHANGE)
        assert SYNC_ACTIONS == (ACTION_ADDED, ACTION_UPDATED, ACTION_REMOVED, ACTION_UNCHANGED)
        assert SYNC_ACTION_IS_CHANGE[ACTION_UNCHANGED] is False
        assert SYNC_ACTION_IS_CHANGE[ACTION_REMOVED] is True

    def test_schedule_mode_tables_align(self) -> None:
        assert set(SCHEDULE_MODES) == set(SCHEDULE_MODE_DESCRIPTIONS)
        assert len(SCHEDULE_MODES) == 5

    def test_health_tables_align(self) -> None:
        assert set(HEALTH_STATES) == set(HEALTH_STATE_DESCRIPTIONS) == set(
            HEALTH_STATE_EXIT_CODES
        )
        assert set(HEALTH_CHECKS) == set(HEALTH_CHECK_DESCRIPTIONS)
        assert HEALTH_STATE_EXIT_CODES[STATE_FAIL] == 1
        assert HEALTH_STATE_EXIT_CODES[STATE_WARN] == 0

    def test_metric_tables_align(self) -> None:
        assert set(OPS_METRICS) == set(OPS_METRIC_UNITS) == set(OPS_METRIC_DESCRIPTIONS)
        assert set(OPS_METRIC_UNITS.values()) <= set(METRIC_UNITS)
        assert len(OPS_METRICS) == 11

    def test_run_status_tables_align(self) -> None:
        assert set(OPS_STATUSES) == set(OPS_STATUS_DESCRIPTIONS)
        assert OPS_STATUSES == (STATUS_NOT_DUE, STATUS_NO_CHANGE, STATUS_SYNCED, STATUS_FAILED)


class TestTime:
    """时间：秒级 UTC 字符串（微秒被砍、时区被抹平、空串不是时间）."""

    def test_round_trip_drops_microseconds(self) -> None:
        moment = datetime(2026, 10, 6, 3, 0, 0, 123456, tzinfo=timezone.utc)
        assert to_iso(moment) == "2026-10-06T03:00:00Z"
        assert parse_iso("2026-10-06T03:00:00Z") == datetime(
            2026, 10, 6, 3, 0, 0, tzinfo=timezone.utc
        )

    def test_naive_datetime_is_read_as_utc(self) -> None:
        naive = datetime(2026, 10, 6, 3, 0, 0)
        assert to_iso(naive) == "2026-10-06T03:00:00Z"

    def test_offset_is_normalized_to_z(self) -> None:
        moment = datetime(2026, 10, 6, 11, 0, 0, tzinfo=timezone(timedelta(hours=8)))
        assert to_iso(moment) == "2026-10-06T03:00:00Z"
        assert parse_iso("2026-10-06T11:00:00+08:00") == datetime(
            2026, 10, 6, 3, 0, 0, tzinfo=timezone.utc
        )

    @pytest.mark.parametrize("value", ["", "   ", "昨天", "2026/10/06 03:00:00", "03:00:00"])
    def test_bad_timestamps_are_rejected(self, value: str) -> None:
        with pytest.raises(SyncError):
            parse_iso(value)

    def test_minutes_between_and_shift(self) -> None:
        later = FIXED_NOW + timedelta(minutes=90)
        assert minutes_between(to_iso(FIXED_NOW), later) == 90.0
        assert minutes_between(to_iso(later), FIXED_NOW) == -90.0
        assert shift_iso(FIXED_NOW, hours=1) == "2026-10-06T04:00:00Z"

    def test_utc_now_is_aware(self) -> None:
        assert utc_now().tzinfo is timezone.utc


class TestSourceEntry:
    """一条来源身份的四条校验（路径风格、指纹形状、字数非负）."""

    def test_ok_and_round_trip(self) -> None:
        item = entry("docs/a.md", char_count=12)
        assert item.to_dict()["source"] == "docs/a.md"
        assert "docs/a.md" in item.summary_line()

    @pytest.mark.parametrize(
        "source",
        ["", "/abs/a.md", "C:/data/a.md", "docs\\a.md", "../outside.md", "docs/../a.md"],
    )
    def test_bad_sources_are_rejected(self, source: str) -> None:
        with pytest.raises(SyncError):
            SourceEntry(source=source, fingerprint=fingerprint("x"))

    @pytest.mark.parametrize("value", ["", "abc", "0" * 15, "0" * 17, "A" * 16, "g" * 16])
    def test_bad_fingerprints_are_rejected(self, value: str) -> None:
        with pytest.raises(SyncError):
            SourceEntry(source="docs/a.md", fingerprint=value)

    def test_negative_char_count_is_rejected(self) -> None:
        with pytest.raises(SyncError):
            SourceEntry(source="docs/a.md", fingerprint=fingerprint("a"), char_count=-1)


class TestCorpusSnapshot:
    """一圈扫盘的账：排序、唯一性、跳过项、以及"快照号只由 entries 算出"."""

    def test_entries_are_sorted_and_deduped_by_source(self) -> None:
        snap = snapshot((entry("docs/b.md"), entry("docs/a.md")))
        assert snap.sources == ("docs/a.md", "docs/b.md")
        assert snap.count == 2

    def test_duplicate_source_is_rejected(self) -> None:
        with pytest.raises(SyncError):
            snapshot((entry("docs/a.md", seed="one"), entry("docs/a.md", seed="two")))

    def test_skipped_requires_a_reason(self) -> None:
        with pytest.raises(SyncError):
            snapshot((), skipped=(("docs/a.md", ""),))

    def test_skipped_cannot_overlap_entries(self) -> None:
        with pytest.raises(SyncError):
            snapshot((entry("docs/a.md"),), skipped=(("docs/a.md", "读不了"),))

    def test_unknown_version_is_rejected(self) -> None:
        with pytest.raises(SyncError):
            CorpusSnapshot(entries=(), version=99)

    def test_digest_ignores_created_at_and_skipped(self) -> None:
        base = snapshot((entry("docs/a.md"),), created_at=to_iso(FIXED_NOW))
        later = snapshot((entry("docs/a.md"),), created_at=to_iso(FIXED_NOW + timedelta(days=3)))
        with_skip = snapshot((entry("docs/a.md"),), skipped=(("docs/b.md", "读不了"),))
        assert base.digest == later.digest == with_skip.digest

    def test_digest_changes_with_fingerprint(self) -> None:
        assert snapshot((entry("docs/a.md"),)).digest != snapshot(
            (entry("docs/a.md", seed="changed"),)
        ).digest

    def test_digest_is_16_hex(self) -> None:
        digest = snapshot().digest
        assert digest == snapshot_id(snapshot().entries)
        assert len(digest) == 16 and all(ch in "0123456789abcdef" for ch in digest)

    def test_lookup_helpers(self) -> None:
        snap = snapshot()
        assert snap.by_source()["docs/a.md"].source == "docs/a.md"
        assert snap.entry("docs/a.md") is not None
        assert snap.entry("docs/missing.md") is None
        assert snap.total_chars == 20
        assert "2 份来源" in snap.summary_line()

    def test_round_trip_and_projection(self) -> None:
        snap = snapshot((entry("docs/a.md"),), skipped=(("docs/b.md", "读不了"),))
        payload = snap.to_dict()
        restored = CorpusSnapshot.from_dict(json.loads(json.dumps(payload)))
        assert restored.digest == snap.digest
        assert restored.skipped == snap.skipped
        thin = snap.to_dict(include_entries=False)
        assert "entries" not in thin and thin["count"] == 1

    def test_snapshot_from_entries_factory_and_bad_created_at(self) -> None:
        assert snapshot_from_entries((entry("docs/a.md"),)).count == 1
        with pytest.raises(SyncError):
            CorpusSnapshot(entries=(), created_at="昨天")


class TestSyncPlan:
    """差集：四组互斥、两侧被完全解释、变更比例用并集做分母."""

    def test_normalizes_and_sorts(self) -> None:
        built = plan(added=("docs/b.md", "docs/a.md"))
        assert built.added == ("docs/a.md", "docs/b.md")

    def test_overlap_between_groups_is_rejected(self) -> None:
        with pytest.raises(SyncError):
            SyncPlan(
                added=("docs/a.md",),
                unchanged=("docs/a.md",),
                current_sources=("docs/a.md",),
            )

    def test_incomplete_previous_snapshot_is_rejected(self) -> None:
        with pytest.raises(SyncError):
            plan(unchanged=("docs/a.md",), previous_sources=("docs/a.md", "docs/b.md"))

    def test_incomplete_current_snapshot_is_rejected(self) -> None:
        with pytest.raises(SyncError):
            plan(
                unchanged=("docs/a.md",),
                previous_sources=("docs/a.md",),
                current_sources=("docs/a.md", "docs/b.md"),
            )

    def test_derived_counts_and_churn(self) -> None:
        built = plan(added=("docs/c.md",), updated=("docs/a.md",), unchanged=("docs/b.md",))
        assert built.changed == ("docs/a.md", "docs/c.md")
        assert built.changed_count == 2
        assert built.empty is False
        # 变更 2 / 并集 3 → 0.666667
        assert built.churn_ratio == pytest.approx(2 / 3, abs=1e-6)
        assert built.sources(ACTION_UNCHANGED) == ("docs/b.md",)
        assert "变更比例 66.7%" in built.summary_line()

    def test_empty_plan_has_zero_churn(self) -> None:
        built = plan(unchanged=("docs/a.md",))
        assert built.empty is True and built.churn_ratio == 0.0

    def test_empty_union_is_zero(self) -> None:
        assert SyncPlan().churn_ratio == 0.0

    def test_unknown_action_is_rejected(self) -> None:
        with pytest.raises(SyncError):
            plan().sources("deleted")

    def test_bad_snapshot_ids_are_rejected(self) -> None:
        with pytest.raises(SyncError):
            SyncPlan(current_id="xyz")

    def test_projection(self) -> None:
        built = plan(added=("docs/c.md",), unchanged=("docs/a.md",))
        payload = built.to_dict()
        assert payload["counts"] == {
            ACTION_ADDED: 1,
            ACTION_UPDATED: 0,
            ACTION_REMOVED: 0,
            ACTION_UNCHANGED: 1,
        }
        assert payload["current_sources"] == ["docs/a.md", "docs/c.md"]
        assert "previous_sources" not in built.to_dict(include_sources=False)


class TestSyncLedger:
    """账本：**水位只在成功时前进**、失败要留原因、读不到文件时返回空账本."""

    def test_empty_ledger_has_no_watermark(self) -> None:
        fresh = SyncLedger.empty()
        assert fresh.has_watermark is False
        assert fresh.age_minutes(FIXED_NOW) is None
        assert "水位 （无）" in fresh.summary_line()

    def test_mark_success_advances_everything(self) -> None:
        water = snapshot()
        updated = ledger().mark_success(
            at=to_iso(FIXED_NOW), snapshot=water, index_version=fingerprint("v1")
        )
        assert updated.snapshot is not None and updated.has_watermark
        assert updated.runs == 1 and updated.consecutive_failures == 0
        assert updated.age_minutes(FIXED_NOW) == 0.0
        assert updated.describe(now=FIXED_NOW)["watermark"] == water.digest

    def test_mark_success_keeps_index_version_when_not_given(self) -> None:
        first = ledger().mark_success(
            at=to_iso(FIXED_NOW), snapshot=snapshot(), index_version=fingerprint("v1")
        )
        again = first.mark_success(at=to_iso(FIXED_NOW), snapshot=snapshot())
        assert again.last_index_version == first.last_index_version

    def test_mark_failure_never_advances_watermark(self) -> None:
        water = snapshot()
        ok = ledger().mark_success(at=to_iso(FIXED_NOW), snapshot=water)
        failed = ok.mark_failure(at=to_iso(FIXED_NOW), error="编码器挂了")
        assert failed.snapshot is not None and failed.snapshot.digest == water.digest
        assert failed.consecutive_failures == 1 and failed.failures == 1 and failed.runs == 2
        assert failed.last_error == "编码器挂了"

    def test_failure_without_reason_is_rejected(self) -> None:
        with pytest.raises(SyncError):
            ledger().mark_failure(at=to_iso(FIXED_NOW), error="")

    @pytest.mark.parametrize(
        "fields",
        [
            {"version": 99},
            {"consecutive_failures": -1},
            {"runs": 1, "failures": 2},
            {"consecutive_failures": 1},
            {"last_index_version": "zz"},
            {"last_run_at": "昨天"},
        ],
    )
    def test_shape_violations_are_rejected(self, fields: dict) -> None:
        with pytest.raises(SyncError):
            SyncLedger(**fields)

    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "ledger.json"
        water = snapshot()
        saved = ran_ledger(water, index_version=fingerprint("v1"))
        assert saved.save(target) == target
        restored = SyncLedger.load(target)
        assert restored.to_dict() == saved.to_dict()

    def test_missing_file_is_an_empty_ledger_not_an_error(self, tmp_path: Path) -> None:
        restored = SyncLedger.load(tmp_path / "nope.json")
        assert restored.has_watermark is False and restored.runs == 0

    def test_broken_json_is_an_error(self, tmp_path: Path) -> None:
        target = tmp_path / "bad.json"
        target.write_text("{ not json", encoding="utf-8")
        with pytest.raises(SyncError):
            SyncLedger.load(target)

    def test_non_object_json_is_an_error(self, tmp_path: Path) -> None:
        target = tmp_path / "list.json"
        target.write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(SyncError):
            SyncLedger.load(target)

    def test_none_snapshot_in_payload_means_no_watermark(self, tmp_path: Path) -> None:
        target = tmp_path / "empty.json"
        target.write_text(json.dumps(SyncLedger.empty().to_dict()), encoding="utf-8")
        assert SyncLedger.load(target).has_watermark is False


class TestScheduleShapes:
    """调度策略与判定：五个参数的量纲、指数退避的封顶、两种"矛盾"要当场拒绝."""

    def test_policy_projection_and_backoff(self) -> None:
        built = policy(backoff_base_seconds=60, backoff_max_seconds=1000)
        assert built.to_dict()["interval_minutes"] == 1440
        assert "退避 60→1000s" in built.summary_line()
        assert built.backoff_seconds(0) == 0
        assert built.backoff_seconds(1) == 60
        assert built.backoff_seconds(2) == 120
        assert built.backoff_seconds(3) == 240
        assert built.backoff_seconds(10) == 1000  # 封顶
        assert built.backoff_seconds(40) == 1000  # 指数不溢出

    @pytest.mark.parametrize(
        "overrides",
        [
            {"interval_minutes": 0},
            {"jitter_seconds": -1},
            {"jitter_seconds": 1440 * 60},
            {"max_lag_minutes": -1},
            {"backoff_base_seconds": -1},
            {"backoff_base_seconds": 100, "backoff_max_seconds": 50},
        ],
    )
    def test_bad_policies_are_rejected(self, overrides: dict) -> None:
        with pytest.raises(ScheduleError):
            policy(**overrides)

    def test_decision_projection_and_summary(self) -> None:
        decision = ScheduleDecision(
            mode="due",
            due=True,
            reason="到点了",
            now=to_iso(FIXED_NOW),
            effective_at=to_iso(FIXED_NOW),
            next_run_at=to_iso(FIXED_NOW + timedelta(days=1)),
            lag_minutes=12.5,
            missed_runs=1,
        )
        payload = decision.to_dict()
        assert payload["mode_description"].startswith("已到计划时间")
        assert "该跑" in decision.summary_line()

    @pytest.mark.parametrize(
        "fields",
        [
            {"mode": "whenever", "due": True},
            {"mode": "not_due", "due": True},
            {"mode": "due", "due": False},
            {"mode": "forced", "due": False},
            {"mode": "first_run", "due": False},
            {"reason": ""},
            {"lag_minutes": -1},
            {"missed_runs": -1},
            {"consecutive_failures": -1},
            {"now": "昨天"},
        ],
    )
    def test_bad_decisions_are_rejected(self, fields: dict) -> None:
        base = {
            "mode": "due",
            "due": True,
            "reason": "到点了",
            "now": to_iso(FIXED_NOW),
        }
        base.update(fields)
        # 时间戳那一项由 parse_iso 报 SyncError（它只有一份实现，见 types 的说明），
        # 其余几项都是"这个判定自己前后矛盾"，报 ScheduleError。
        with pytest.raises((ScheduleError, SyncError)):
            ScheduleDecision(**base)


class TestHealthShapes:
    """健康：三态取最坏、**空报告当场报错**、退出码由状态决定."""

    def test_worst_state(self) -> None:
        assert worst_state(()) == STATE_OK
        assert worst_state((STATE_OK, STATE_WARN)) == STATE_WARN
        assert worst_state((STATE_WARN, STATE_FAIL, STATE_OK)) == STATE_FAIL
        with pytest.raises(HealthError):
            worst_state(("unknown",))

    def test_finding_validation_and_projection(self) -> None:
        ok = finding("vector_store", state=STATE_WARN, detail={"count": 0})
        assert ok.to_dict()["state_description"].startswith("警告")
        assert "[warn] vector_store" in ok.summary_line()
        with pytest.raises(HealthError):
            HealthFinding(check="unknown", state=STATE_OK, message="x")
        with pytest.raises(HealthError):
            HealthFinding(check=CHECK_CORPUS_FRESHNESS, state="bad", message="x")
        with pytest.raises(HealthError):
            HealthFinding(check=CHECK_CORPUS_FRESHNESS, state=STATE_OK, message="")

    def test_empty_report_is_rejected(self) -> None:
        with pytest.raises(HealthError):
            HealthReport(findings=())
        with pytest.raises(HealthError):
            HealthReport(findings=(finding(), finding()))

    def test_report_derivations(self) -> None:
        report = health_report(
            finding("vector_store", state=STATE_OK),
            finding("index_integrity", state=STATE_WARN, message="没有清单"),
            finding("corpus_freshness", state=STATE_FAIL, message="库是空的"),
            finding("quality_regression", state=STATE_OK),
        )
        assert report.state == STATE_FAIL
        assert report.ok is False and report.exit_code == 1
        assert len(report.failed) == 1 and len(report.warnings) == 1
        assert report.by_check()["index_integrity"].message == "没有清单"
        payload = report.to_dict()
        assert payload["counts"] == {STATE_OK: 2, STATE_WARN: 1, STATE_FAIL: 1}
        assert "探针退出码 1" in report.summary_line()

    def test_warn_only_report_is_still_healthy(self) -> None:
        report = health_report(
            finding("vector_store"),
            finding("index_integrity"),
            finding("corpus_freshness", state=STATE_WARN, message="语料偏旧"),
            finding("quality_regression"),
        )
        assert report.state == STATE_WARN and report.ok is True and report.exit_code == 0

    def test_bad_checked_at_is_rejected(self) -> None:
        with pytest.raises(SyncError):
            HealthReport(findings=(finding(),), checked_at="昨天")


class TestMetricsShapes:
    """监控采样与序列：指标名封闭、值必须有限、同序列不许混指标或时间倒序."""

    def test_sample_unit_and_direction_come_from_the_table(self) -> None:
        assert sample("sync_seconds", 1.5).unit == "seconds"
        assert sample("sync_age_minutes", 5).direction == "lower_better"
        assert sample("quality_retrieval_recall", 0.9).direction == "higher_better"
        assert sample("sources_changed", 3).direction == "neutral"
        assert "sources_total = 1.0 count" in sample().summary_line()

    def test_sample_projection_carries_label_and_direction(self) -> None:
        payload = sample("index_records", 7, labels=(("backend", "flat"),)).to_dict()
        assert payload["labels"] == {"backend": "flat"}
        assert payload["unit"] == "count" and payload["description"]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"name": "custom_metric", "value": 1.0},
            {"name": "sources_total", "value": float("nan")},
            {"name": "sources_total", "value": float("inf")},
            {"name": "sources_total", "value": 1.0, "at": "昨天"},
            {"name": "sources_total", "value": 1.0, "labels": (("", "v"),)},
            {"name": "sources_total", "value": 1.0, "labels": (("k", "a\nb"),)},
        ],
    )
    def test_bad_samples_are_rejected(self, kwargs: dict) -> None:
        fields = {"at": to_iso(FIXED_NOW)}
        fields.update(kwargs)
        # 时刻那一项由 parse_iso 报 SyncError（单一实现），其余是本层的 MetricsError。
        with pytest.raises((MetricsError, SyncError)):
            MetricSample(**fields)

    def test_series_summary(self) -> None:
        series = MetricSeries(
            name="sources_changed",
            samples=(
                sample("sources_changed", 1.0),
                sample("sources_changed", 3.0, at=to_iso(FIXED_NOW + timedelta(days=1))),
            ),
        )
        assert series.count == 2 and series.latest is not None
        summary = series.summary()
        assert summary["min"] == 1.0 and summary["max"] == 3.0 and summary["mean"] == 2.0
        assert summary["latest"] == 3.0
        assert series.to_dict()["summary"]["count"] == 2

    def test_empty_series(self) -> None:
        empty = MetricSeries(name="sources_changed")
        assert empty.latest is None and empty.values == ()
        assert empty.summary()["mean"] is None

    def test_series_rejects_unknown_name_and_mixed_samples(self) -> None:
        with pytest.raises(MetricsError):
            MetricSeries(name="custom")
        with pytest.raises(MetricsError):
            MetricSeries(
                name="sources_changed", samples=(sample("sources_total", 1.0),)
            )

    def test_series_rejects_time_regression(self) -> None:
        with pytest.raises(MetricsError):
            MetricSeries(
                name="sources_changed",
                samples=(
                    sample("sources_changed", 1.0, at=to_iso(FIXED_NOW + timedelta(days=1))),
                    sample("sources_changed", 2.0, at=to_iso(FIXED_NOW)),
                ),
            )


class TestOpsReport:
    """一次运行的账：状态与证据必须对得上（"同步了"就得有构建报告）."""

    def _decision(self) -> ScheduleDecision:
        return ScheduleDecision(
            mode="due",
            due=True,
            reason="到点了",
            now=to_iso(FIXED_NOW),
            effective_at=to_iso(FIXED_NOW),
            next_run_at=to_iso(FIXED_NOW + timedelta(days=1)),
        )

    def _report(self, **overrides: object) -> OpsReport:
        fields: dict[str, object] = {
            "started_at": to_iso(FIXED_NOW),
            "finished_at": to_iso(FIXED_NOW),
            "status": STATUS_SYNCED,
            "schedule": self._decision(),
            "index": {"written": 2},
            "health": health_report(),
            "seconds": 0.5,
        }
        fields.update(overrides)
        return OpsReport(**fields)  # type: ignore[arg-type]

    def test_ok_report_derivations(self) -> None:
        report = self._report(plan=plan(added=("docs/a.md",)))
        assert report.changed == 1 and report.healthy is True
        assert report.to_dict()["status_description"].startswith("已同步")
        assert "[synced]" in report.summary_line()

    def test_metric_lookup(self) -> None:
        report = self._report(metrics=(sample("chunks_written", 4.0),))
        assert report.metric("chunks_written") == 4.0
        assert report.metric("missing_metric") is None

    def test_no_health_means_not_healthy(self) -> None:
        assert self._report(health=None).healthy is False

    @pytest.mark.parametrize(
        "overrides",
        [
            {"version": 99},
            {"started_at": "昨天"},
            {"status": "unknown"},
            {"seconds": -1.0},
            {"status": STATUS_SYNCED, "index": None},
            {"status": STATUS_FAILED, "error": "", "index": None},
            {"status": STATUS_NOT_DUE, "index": {"written": 0}},
        ],
    )
    def test_shape_violations_are_rejected(self, overrides: dict) -> None:
        with pytest.raises(SyncError):
            self._report(**overrides)

    def test_failed_report_keeps_the_reason(self) -> None:
        report = self._report(
            status=STATUS_FAILED, index=None, error="编码器挂了", health=None
        )
        assert report.to_dict()["error"] == "编码器挂了"
        assert report.changed == 0

    def test_save_writes_json(self, tmp_path: Path) -> None:
        target = tmp_path / "out" / "report.json"
        self._report().save(target)
        payload = json.loads(target.read_text(encoding="utf-8"))
        assert payload["status"] == STATUS_SYNCED
        assert payload["schedule"]["mode"] == "due"
