"""评估指标模块测试：RunRecord 与 MetricsTracker 的聚合统计."""

from __future__ import annotations

from smart_research_agent.evaluation import MetricsTracker, RunRecord


class TestRunRecord:
    def test_fields(self):
        record = RunRecord(success=True, steps=3, latency_seconds=1.5, tool_calls=2)
        assert record.success is True
        assert record.steps == 3
        assert record.latency_seconds == 1.5
        assert record.tool_calls == 2

    def test_tool_calls_default_zero(self):
        record = RunRecord(success=False, steps=10, latency_seconds=5.0)
        assert record.tool_calls == 0


class TestMetricsTracker:
    def test_empty_tracker_metrics_are_zero(self):
        tracker = MetricsTracker()
        assert tracker.total_runs == 0
        assert tracker.success_rate == 0.0
        assert tracker.avg_steps == 0.0
        assert tracker.avg_latency == 0.0

    def test_success_rate(self):
        tracker = MetricsTracker()
        tracker.record_run(success=True, steps=2, latency_seconds=1.0)
        tracker.record_run(success=True, steps=4, latency_seconds=2.0)
        tracker.record_run(success=False, steps=10, latency_seconds=6.0)
        assert abs(tracker.success_rate - 2 / 3) < 1e-9

    def test_avg_steps_and_latency(self):
        tracker = MetricsTracker()
        tracker.record_run(success=True, steps=2, latency_seconds=1.0)
        tracker.record_run(success=True, steps=4, latency_seconds=3.0)
        assert tracker.avg_steps == 3.0
        assert tracker.avg_latency == 2.0

    def test_avg_tool_calls(self):
        tracker = MetricsTracker()
        tracker.record_run(success=True, steps=2, latency_seconds=1.0, tool_calls=1)
        tracker.record_run(success=True, steps=4, latency_seconds=2.0, tool_calls=3)
        assert tracker.avg_tool_calls == 2.0

    def test_report_dict_shape(self):
        tracker = MetricsTracker()
        tracker.record_run(success=True, steps=2, latency_seconds=1.0, tool_calls=1)
        tracker.record_run(success=False, steps=6, latency_seconds=3.0, tool_calls=3)

        report = tracker.report()
        assert report == {
            "total_runs": 2,
            "success_rate": 0.5,
            "avg_steps": 4.0,
            "avg_latency_seconds": 2.0,
            "avg_tool_calls": 2.0,
        }

    def test_report_empty(self):
        report = MetricsTracker().report()
        assert report["total_runs"] == 0
        assert report["success_rate"] == 0.0

    def test_record_accepts_run_record_object(self):
        tracker = MetricsTracker()
        tracker.record(RunRecord(success=True, steps=1, latency_seconds=0.5))
        assert tracker.total_runs == 1
        assert tracker.success_rate == 1.0
