"""性能基线与回归门禁测试（day046）：度量、存档、判定三段的确定性验证.

耗时全部来自注入的假时钟，因此断言的是**计算逻辑**而不是机器性能。
"""

from __future__ import annotations

import json

import pytest

from smart_research_agent.config import settings
from smart_research_agent.evaluation.perf_baseline import (
    DEFAULT_MIN_LATENCY_MS,
    DEFAULT_MIN_TOKENS,
    TOTAL_KEY,
    BaselineComparison,
    PerfBaseline,
    PerformanceGuard,
    Regression,
    compare_with_file,
    guard_from_settings,
    measure,
    percentile,
)
from smart_research_agent.integration.pipeline import IntegratedPipeline
from smart_research_agent.llm.base import BaseLLM, Message
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.observability.cost_tracker import CostTracker


class FixedClock:
    """每次读取前进固定步长的时钟（与时钟调用次数线性对应）."""

    def __init__(self, step: float = 0.001):
        self._now = 0.0
        self._step = step

    def __call__(self) -> float:
        value = self._now
        self._now += self._step
        return value


class StubLLM(BaseLLM):
    """固定回复的桩.

    刻意**不**提供 ``usage_log``：成本追踪器要求 LLM 自报用量（``record_from_llm``
    的契约），桩不提供正好用来验证"未装配追踪器时用量指标为 0"这条口径；
    需要真实用量时用 ``MockLLM``（它按 day032 的约定记录 ``usage_log``）。
    """

    def __init__(self, reply: str = "四字回复"):
        self.reply = reply
        self.calls = 0

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        self.calls += 1
        return self.reply


def make_baseline(**overrides) -> PerfBaseline:
    """构造一份基线，字段可用关键字覆盖（默认值是一份"已知"的参考）."""
    data = {
        "label": "pipeline",
        "samples": 3,
        "latency_ms": {TOTAL_KEY: 100.0, "generate": 60.0, "guard": 1.0},
        "p95_latency_ms": {TOTAL_KEY: 120.0, "generate": 70.0, "guard": 1.0},
        "mean_tokens": 100.0,
        "mean_cost_usd": 0.01,
        "mean_stages": 6.0,
    }
    data.update(overrides)
    return PerfBaseline(**data)


class TestPercentile:
    """最近秩法百分位数：只返回真实观测值."""

    def test_empty(self):
        assert percentile([], 50) == 0.0

    def test_single_value(self):
        assert percentile([7.0], 95) == 7.0

    def test_min_and_max_boundaries(self):
        values = [3.0, 1.0, 2.0]
        assert percentile(values, 0) == 1.0
        assert percentile(values, 100) == 3.0

    def test_nearest_rank_is_an_observed_value(self):
        """20 个样本取 P95 -> 第 19 个（下标 18），是真实样本而非插值."""
        values = [float(i) for i in range(1, 21)]
        assert percentile(values, 95) == 19.0

    def test_median_of_four_samples(self):
        assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.0

    def test_invalid_q(self):
        with pytest.raises(ValueError, match="0~100"):
            percentile([1.0], 101)
        with pytest.raises(ValueError, match="0~100"):
            percentile([1.0], -1)


class TestPerfBaselineSerialization:
    """序列化：随仓库提交的 JSON 必须能无损往返."""

    def test_round_trip(self):
        baseline = make_baseline()
        restored = PerfBaseline.from_dict(baseline.to_dict())
        assert restored == baseline

    def test_from_dict_with_missing_fields(self):
        restored = PerfBaseline.from_dict({})
        assert restored.label == "pipeline"
        assert restored.samples == 0
        assert restored.latency_ms == {}
        assert restored.mean_tokens == 0.0

    def test_to_dict_rounds(self):
        baseline = make_baseline(mean_cost_usd=0.012345678901)
        assert baseline.to_dict()["mean_cost_usd"] == 0.01234568

    def test_save_and_load(self, tmp_path):
        path = tmp_path / "nested" / "perf_baseline.json"
        baseline = make_baseline(label="day046")
        saved = baseline.save(path)
        assert saved == path and path.exists()
        assert PerfBaseline.load(path) == baseline

    def test_saved_file_is_readable_json(self, tmp_path):
        path = tmp_path / "baseline.json"
        make_baseline().save(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["label"] == "pipeline"
        assert payload["latency_ms"]["total"] == 100.0

    def test_load_missing_file_raises(self, tmp_path):
        """"没有基线"必须显式报错，不能返回零值基线静默放行所有回归."""
        with pytest.raises(FileNotFoundError, match="不存在"):
            PerfBaseline.load(tmp_path / "nothing.json")


class TestMeasure:
    """采集：复用流水线自报的耗时，不额外包一层计时."""

    def test_measure_collects_stage_and_total(self):
        pipe = IntegratedPipeline(
            MockLLM(responses=["回答"]), clock=FixedClock(), cache=None
        )
        baseline = measure(pipe, ["任务一", "任务二"], label="demo")
        assert baseline.label == "demo"
        assert baseline.samples == 2
        assert baseline.latency_ms[TOTAL_KEY] == 13.0  # 满路径固定 13 次时钟读取
        assert baseline.p95_latency_ms[TOTAL_KEY] == 13.0
        assert set(baseline.latency_ms) >= {"total", "guard", "generate", "moderate"}
        assert baseline.mean_stages == 6.0

    def test_measure_without_tracker_has_no_numbers(self):
        """没有账本就没有数字：未装配 CostTracker 时 token 与费用都是 0.

        这不是缺陷而是口径声明——token 的唯一来源是成本归集阶段（它读 LLM
        的 ``usage_log``），没有归集就没有可核对的用量，绝不用"估算"顶替。
        """
        pipe = IntegratedPipeline(StubLLM(), clock=FixedClock())
        baseline = measure(pipe, ["a", "b", "c"])
        assert baseline.mean_tokens == 0.0
        assert baseline.mean_cost_usd == 0.0

    def test_measure_tokens_and_cost_with_tracker(self):
        tracker = CostTracker()
        tracker.price_table["MockLLM"] = {"input": 0.001, "output": 0.002}
        pipe = IntegratedPipeline(
            MockLLM(responses=["四字回复"]), clock=FixedClock(), tracker=tracker
        )
        baseline = measure(pipe, ["a", "b", "c"])
        assert baseline.mean_tokens > 0
        assert baseline.mean_cost_usd > 0

    def test_measure_empty_task_set(self):
        pipe = IntegratedPipeline(MockLLM(), clock=FixedClock())
        baseline = measure(pipe, [])
        assert baseline.samples == 0
        assert baseline.latency_ms[TOTAL_KEY] == 0.0
        assert baseline.mean_tokens == 0.0
        assert baseline.mean_stages == 0.0

    def test_measure_with_system_prompt(self):
        pipe = IntegratedPipeline(MockLLM(responses=["回答"]), clock=FixedClock())
        baseline = measure(pipe, ["同一任务"], system_prompt="你是助手")
        assert baseline.samples == 1


class TestPerformanceGuard:
    """门禁：容忍倍数 + 判定下限，把抖动与真回归分开."""

    def test_tolerances_must_be_positive(self):
        with pytest.raises(ValueError, match="latency_tolerance"):
            PerformanceGuard(make_baseline(), latency_tolerance=0)
        with pytest.raises(ValueError, match="token_tolerance"):
            PerformanceGuard(make_baseline(), token_tolerance=-1)
        with pytest.raises(ValueError, match="cost_tolerance"):
            PerformanceGuard(make_baseline(), cost_tolerance=0)

    def test_no_regression(self):
        guard = PerformanceGuard(make_baseline())
        comparison = guard.compare(make_baseline())
        assert comparison.ok is True
        assert comparison.regressions == []
        assert "未回归" in comparison.summary()

    def test_latency_regression_detected(self):
        current = make_baseline(latency_ms={TOTAL_KEY: 200.0, "generate": 60.0, "guard": 1.0})
        comparison = PerformanceGuard(make_baseline()).compare(current)
        assert comparison.ok is False
        metrics = {r.metric for r in comparison.regressions}
        assert f"latency_ms[{TOTAL_KEY}]" in metrics
        assert "回归" in comparison.summary()
        assert comparison.to_dict()["regressions"][0]["baseline"] == 100.0

    def test_within_tolerance_is_not_regression(self):
        current = make_baseline(latency_ms={TOTAL_KEY: 140.0, "generate": 60.0, "guard": 1.0})
        assert PerformanceGuard(make_baseline(), latency_tolerance=1.5).compare(current).ok

    def test_stage_level_regression_detected(self):
        current = make_baseline(
            latency_ms={TOTAL_KEY: 110.0, "generate": 200.0, "guard": 1.0}
        )
        comparison = PerformanceGuard(make_baseline()).compare(current)
        assert [r.metric for r in comparison.regressions] == ["latency_ms[generate]"]

    def test_tiny_baseline_is_ignored(self):
        """低于判定下限的指标跳过比值判定：1ms -> 10ms 是 10 倍噪声，不是回归."""
        baseline = make_baseline(latency_ms={"total": 1.0})
        current = make_baseline(latency_ms={"total": 10.0})
        assert PerformanceGuard(baseline).compare(current).ok is True
        assert DEFAULT_MIN_LATENCY_MS == 5.0

    def test_metric_missing_in_current_is_skipped(self):
        """当前采集里没有的阶段（如缓存命中跳过了 generate）不参与判定."""
        baseline = make_baseline(latency_ms={"total": 100.0, "generate": 60.0})
        current = make_baseline(latency_ms={"total": 100.0})
        assert PerformanceGuard(baseline).compare(current).ok is True

    def test_token_regression_detected(self):
        baseline = make_baseline(mean_tokens=100.0)
        current = make_baseline(mean_tokens=150.0)
        comparison = PerformanceGuard(baseline, token_tolerance=1.2).compare(current)
        assert [r.metric for r in comparison.regressions] == ["mean_tokens"]

    def test_tiny_token_baseline_is_ignored(self):
        baseline = make_baseline(mean_tokens=0.5)
        current = make_baseline(mean_tokens=50.0)
        assert PerformanceGuard(baseline).compare(current).ok is True
        assert DEFAULT_MIN_TOKENS == 1.0

    def test_cost_regression_detected(self):
        baseline = make_baseline(mean_cost_usd=0.01)
        current = make_baseline(mean_cost_usd=0.02)
        comparison = PerformanceGuard(baseline, cost_tolerance=1.2).compare(current)
        assert [r.metric for r in comparison.regressions] == ["mean_cost_usd"]

    def test_zero_cost_baseline_is_skipped(self):
        """全本地模型的基线成本为 0：0 的任意倍数仍是 0，比值判定无意义."""
        baseline = make_baseline(mean_cost_usd=0.0)
        current = make_baseline(mean_cost_usd=0.5)
        assert PerformanceGuard(baseline).compare(current).ok is True

    def test_comparison_to_dict(self):
        comparison = PerformanceGuard(make_baseline()).compare(make_baseline())
        payload = comparison.to_dict()
        assert payload["ok"] is True
        assert payload["regressions"] == []
        assert payload["current"]["label"] == "pipeline"

    def test_regression_to_dict(self):
        regression = Regression(
            metric="mean_tokens",
            baseline=10.0,
            current=20.0,
            ratio=2.0,
            threshold=1.2,
            message="翻倍",
        )
        assert regression.to_dict() == {
            "metric": "mean_tokens",
            "baseline": 10.0,
            "current": 20.0,
            "ratio": 2.0,
            "threshold": 1.2,
            "message": "翻倍",
        }

    def test_guard_from_settings_uses_configured_tolerances(self, monkeypatch):
        monkeypatch.setattr(settings, "perf_latency_tolerance", 1.1)
        monkeypatch.setattr(settings, "perf_token_tolerance", 1.1)
        monkeypatch.setattr(settings, "perf_cost_tolerance", 1.1)
        guard = guard_from_settings(make_baseline())
        assert guard.latency_tolerance == 1.1
        current = make_baseline(mean_tokens=120.0)  # 1.2x > 1.1x
        assert [r.metric for r in guard.compare(current).regressions] == ["mean_tokens"]

    def test_compare_with_file_end_to_end(self, tmp_path):
        path = tmp_path / "perf_baseline.json"
        make_baseline().save(path)
        assert compare_with_file(make_baseline(), path).ok is True
        regressed = make_baseline(latency_ms={TOTAL_KEY: 500.0, "generate": 60.0, "guard": 1.0})
        assert compare_with_file(regressed, path).ok is False

    def test_compare_with_file_missing_baseline(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            compare_with_file(make_baseline(), tmp_path / "absent.json")

    def test_comparison_dataclass_shape(self):
        comparison = BaselineComparison(baseline=make_baseline(), current=make_baseline())
        assert comparison.ok is True
        assert comparison.summary().startswith("性能未回归")
