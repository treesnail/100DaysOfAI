"""day032 可观测性测试：token 用量统计、成本追踪、轻量 trace、告警判定."""

from __future__ import annotations

import pytest

from smart_research_agent.llm.base import Message
from smart_research_agent.llm.mock import MockLLM, estimate_tokens
from smart_research_agent.observability.alerting import check_thresholds, render_alerts
from smart_research_agent.observability.cost_tracker import CostTracker
from smart_research_agent.observability.tracing import Tracer, run_with_trace


# ----------------------------------------------------------------------
# MockLLM 用量统计（day032 新增，向后兼容）
# ----------------------------------------------------------------------


class TestMockLLMUsage:
    def test_chat_return_value_unchanged(self):
        """向后兼容：加统计后 chat 的返回值与旧行为完全一致."""
        llm = MockLLM(responses=["回答1", "回答2"], default="兜底")
        assert llm.chat([Message(role="user", content="hi")]) == "回答1"
        assert llm.chat([Message(role="user", content="hi")]) == "回答2"
        assert llm.chat([Message(role="user", content="hi")]) == "兜底"

    def test_usage_log_records_each_call(self):
        llm = MockLLM(responses=["ab"] * 2)
        llm.chat([Message(role="user", content="12345678")])  # 8 chars -> 2 tokens
        llm.chat([Message(role="user", content="1234")])  # 4 chars -> 1 token
        assert len(llm.usage_log) == 2
        assert llm.usage_log[0] == {"prompt_tokens": 2, "completion_tokens": 1}
        assert llm.usage_log[1] == {"prompt_tokens": 1, "completion_tokens": 1}

    def test_cumulative_counters(self):
        llm = MockLLM(responses=["12345678"])
        llm.chat([Message(role="user", content="12345678")])
        assert llm.total_prompt_tokens == 2
        assert llm.total_completion_tokens == 2

    def test_completion_tokens_capped_by_max_tokens(self):
        llm = MockLLM(responses=["x" * 400])  # 估算 100 token
        llm.chat([Message(role="user", content="hi")], max_tokens=10)
        assert llm.usage_log[0]["completion_tokens"] == 10

    def test_estimate_tokens_minimum_one(self):
        assert estimate_tokens("") == 1
        assert estimate_tokens("ab") == 1
        assert estimate_tokens("a" * 100) == 25


# ----------------------------------------------------------------------
# CostTracker
# ----------------------------------------------------------------------


class TestCostTracker:
    def test_record_and_cost_calculation(self):
        tracker = CostTracker()
        tracker.record("gpt-4o-mini", prompt_tokens=1000, completion_tokens=500)
        # (1000*0.00015 + 500*0.0006) / 1000 = 0.00045
        assert tracker.total_cost == pytest.approx(0.00045)

    def test_unknown_model_raises(self):
        """模型不在价格表时宁可报错，不可按 0 元漏算."""
        tracker = CostTracker()
        with pytest.raises(KeyError):
            tracker.record("unknown-model", prompt_tokens=1, completion_tokens=1)

    def test_custom_price_table(self):
        tracker = CostTracker(price_table={"my-model": {"input": 0.001, "output": 0.002}})
        tracker.record("my-model", prompt_tokens=2000, completion_tokens=1000)
        # (2000*0.001 + 1000*0.002) / 1000 = 0.004
        assert tracker.total_cost == pytest.approx(0.004)

    def test_breakdown_by_model(self):
        tracker = CostTracker()
        tracker.record("gpt-4o-mini", prompt_tokens=1000, completion_tokens=0)
        tracker.record("gpt-4o", prompt_tokens=1000, completion_tokens=0)
        tracker.record("gpt-4o-mini", prompt_tokens=1000, completion_tokens=0)
        by_model = tracker.breakdown("model")
        assert by_model["gpt-4o-mini"]["prompt_tokens"] == 2000
        assert by_model["gpt-4o"]["prompt_tokens"] == 1000

    def test_breakdown_by_endpoint(self):
        tracker = CostTracker()
        tracker.record("gpt-4o-mini", 100, 50, endpoint="chat")
        tracker.record("gpt-4o-mini", 200, 0, endpoint="embedding")
        by_ep = tracker.breakdown("endpoint")
        assert set(by_ep) == {"chat", "embedding"}
        assert by_ep["embedding"]["prompt_tokens"] == 200

    def test_record_from_llm_consumes_increment(self):
        """record_from_llm 只消费自上次以来的增量，重复调用不重复计费."""
        tracker = CostTracker()
        llm = MockLLM(responses=["ab"] * 3)
        llm.chat([Message(role="user", content="1234")])
        assert tracker.record_from_llm(llm, model="gpt-4o-mini") == 1
        llm.chat([Message(role="user", content="1234")])
        llm.chat([Message(role="user", content="1234")])
        assert tracker.record_from_llm(llm, model="gpt-4o-mini") == 2
        assert tracker.record_from_llm(llm, model="gpt-4o-mini") == 0
        assert len(tracker.records) == 3

    def test_report_structure(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", prompt_tokens=500, completion_tokens=500)
        report = tracker.report()
        assert report["total_calls"] == 1
        assert report["total_tokens"] == 1000
        assert "gpt-4o" in report["by_model"]
        assert "chat" in report["by_endpoint"]

    def test_empty_tracker_report(self):
        report = CostTracker().report()
        assert report["total_calls"] == 0
        assert report["total_cost_usd"] == 0.0


# ----------------------------------------------------------------------
# Tracer
# ----------------------------------------------------------------------


class TestTracer:
    def test_trace_records_root_and_child_spans(self, tmp_path):
        tracer = Tracer(sink_path=tmp_path / "traces.jsonl")
        with tracer.start_trace("agent.run", task="t") as ctx:
            with ctx.span("llm.chat"):
                pass
            with ctx.span("tool.calculator"):
                pass
        assert len(tracer.spans) == 3
        names = [s.name for s in tracer.spans]
        assert names == ["llm.chat", "tool.calculator", "agent.run"]

    def test_child_span_parent_is_root(self):
        tracer = Tracer()
        with tracer.start_trace() as ctx:
            with ctx.span("child") as span:
                assert span.parent_id == ctx.root.span_id
                assert span.trace_id == ctx.trace_id

    def test_span_duration_and_status(self):
        tracer = Tracer()
        with tracer.start_trace() as ctx:
            with ctx.span("op") as span:
                pass
        assert span.status == "ok"
        assert span.duration_ms is not None and span.duration_ms >= 0

    def test_exception_marks_span_error_and_reraises(self):
        tracer = Tracer()
        with pytest.raises(ValueError):
            with tracer.start_trace() as ctx:
                with ctx.span("failing"):
                    raise ValueError("boom")
        errors = tracer.query(status="error")
        # 子 span 与 root span 都被标记为 error
        assert {s.name for s in errors} == {"failing", "agent.run"}

    def test_jsonl_sink_and_replay(self, tmp_path):
        path = tmp_path / "traces.jsonl"
        tracer = Tracer(sink_path=path)
        with tracer.start_trace("agent.run") as ctx:
            with ctx.span("llm.chat", messages=2):
                pass
            trace_id = ctx.trace_id
        records = Tracer.load(path)
        assert len(records) == 2
        assert all(r["trace_id"] == trace_id for r in records)
        chat = next(r for r in records if r["name"] == "llm.chat")
        assert chat["attributes"]["messages"] == 2

    def test_get_trace_replays_in_time_order(self):
        tracer = Tracer()
        with tracer.start_trace() as ctx:
            for i in range(3):
                with ctx.span(f"step-{i}"):
                    pass
            trace_id = ctx.trace_id
        spans = tracer.get_trace(trace_id)
        assert [s.name for s in spans] == ["agent.run", "step-0", "step-1", "step-2"]

    def test_query_by_name(self):
        tracer = Tracer()
        with tracer.start_trace() as ctx:
            with ctx.span("llm.chat"):
                pass
            with ctx.span("llm.chat"):
                pass
        assert len(tracer.query(name="llm.chat")) == 2


class TestRunWithTrace:
    def _make_agent(self):
        from smart_research_agent.agent.react_agent import ReactAgent
        from smart_research_agent.tools.calculator import CalculatorTool
        from smart_research_agent.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(CalculatorTool())
        llm = MockLLM(responses=["Thought: 直接答\nFinal Answer: 42"])
        return ReactAgent(llm=llm, registry=registry), llm

    def test_agent_run_produces_llm_spans(self, tmp_path):
        agent, llm = self._make_agent()
        tracer = Tracer(sink_path=tmp_path / "traces.jsonl")
        answer = run_with_trace(agent, "计算 1+1", tracer)
        assert answer == "42"
        chat_spans = tracer.query(name="llm.chat")
        assert len(chat_spans) == len(llm.calls) == 1
        root = tracer.query(name="agent.run")[0]
        assert root.attributes["answer"] == "42"
        assert root.attributes["history_steps"] == 1

    def test_agent_llm_restored_after_run(self):
        agent, llm = self._make_agent()
        run_with_trace(agent, "计算 1+1", Tracer())
        assert agent.llm is llm


# ----------------------------------------------------------------------
# 告警判定
# ----------------------------------------------------------------------


class TestAlerting:
    def test_below_threshold_produces_alert(self):
        alerts = check_thresholds({"completion_rate": 0.5}, {"completion_rate": 0.8})
        assert len(alerts) == 1
        assert alerts[0].metric == "completion_rate"
        assert alerts[0].value == 0.5

    def test_at_threshold_passes(self):
        assert check_thresholds({"block_rate": 1.0}, {"block_rate": 1.0}) == []

    def test_missing_metric_skipped(self):
        assert check_thresholds({}, {"completion_rate": 0.8}) == []

    def test_render_alerts_marks_red(self):
        alerts = check_thresholds({"block_rate": 0.8}, {"block_rate": 0.95})
        text = render_alerts(alerts)
        assert "告警" in text and "color:red" in text and "block_rate" in text

    def test_render_empty_when_no_alerts(self):
        assert render_alerts([]) == ""
