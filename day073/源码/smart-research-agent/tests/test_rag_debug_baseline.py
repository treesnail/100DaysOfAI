"""``rag_debug.baseline`` 的质量基线与三门门禁（day071）.

这一份测试的重心是**方向**与**能不能比**：

```text
方向    同样 +0.10 的变化，对 grounded_rate 是改善、对 hallucination_rate 是回归
        而 llm_call_rate 两个方向都不判（它是描述性指标）
能否比  样本不足 / k 口径变了 / 索引版本变了 → **不下结论**（而不是"通过"）
```
"""

from __future__ import annotations

import json

import pytest

from smart_research_agent.rag_debug.baseline import (
    DEFAULT_DEAD_BAND,
    DEFAULT_MIN_SAMPLES,
    DEFAULT_TOLERANCES,
    DIRECTION_HIGHER_BETTER,
    DIRECTION_LOWER_BETTER,
    DIRECTION_NEUTRAL,
    DIRECTIONS,
    METRIC_BAD_CASE_RATE,
    METRIC_COVERAGE,
    METRIC_DESCRIPTIONS,
    METRIC_DIRECTIONS,
    METRIC_FALLBACK_RATE,
    METRIC_GROUNDED_RATE,
    METRIC_HALLUCINATION_RATE,
    METRIC_LLM_CALL_RATE,
    METRIC_PACKED_RECALL,
    METRIC_RETRIEVAL_RECALL,
    QUALITY_METRICS,
    RagBaseline,
    RagBaselineGuard,
    compare_with_file,
    deltas_between,
    guard_from_settings,
)
from smart_research_agent.rag_debug.errors import BaselineError
from tests.rag_debug_samples import metrics_row


def baseline(**overrides) -> RagBaseline:
    """造一份基线（指标缺省全 0.5，可逐项覆盖）."""
    payload = {
        "label": "baseline:baseline",
        "samples": 6,
        "k": 3,
        "metrics": metrics_row(),
        "prompt_version": "v2",
        "index_version": "idx-1",
    }
    payload.update(overrides)
    return RagBaseline(**payload)


class TestMetricTables:
    def test_three_tables_align(self):
        assert set(METRIC_DIRECTIONS) == set(DEFAULT_TOLERANCES)
        assert set(METRIC_DESCRIPTIONS) == set(METRIC_DIRECTIONS)

    def test_there_are_three_directions(self):
        assert DIRECTIONS == (
            DIRECTION_HIGHER_BETTER,
            DIRECTION_LOWER_BETTER,
            DIRECTION_NEUTRAL,
        )
        assert METRIC_DIRECTIONS[METRIC_LLM_CALL_RATE] == DIRECTION_NEUTRAL
        assert METRIC_DIRECTIONS[METRIC_GROUNDED_RATE] == DIRECTION_HIGHER_BETTER
        assert METRIC_DIRECTIONS[METRIC_HALLUCINATION_RATE] == DIRECTION_LOWER_BETTER

    def test_eleven_metrics(self):
        assert len(QUALITY_METRICS) == 11
        assert METRIC_PACKED_RECALL in QUALITY_METRICS
        assert METRIC_COVERAGE in QUALITY_METRICS
        assert METRIC_FALLBACK_RATE in QUALITY_METRICS
        assert METRIC_BAD_CASE_RATE in QUALITY_METRICS


class TestRagBaseline:
    def test_projects_and_summarizes(self):
        item = baseline()
        data = item.to_dict()
        assert list(data["metrics"]) == list(QUALITY_METRICS)
        assert data["k"] == 3
        assert "基线 baseline:baseline" in item.summary_line()
        assert item.value(METRIC_GROUNDED_RATE) == pytest.approx(0.5)
        assert item.direction(METRIC_HALLUCINATION_RATE) == DIRECTION_LOWER_BETTER

    def test_counts_of_bad_cases_and_fallbacks(self):
        item = baseline(
            tag_counts={"no_data": 1, "low_coverage": 2},
            fallback_counts={"": 4, "llm_error": 2},
        )
        assert item.total_bad_cases() == 3
        assert item.total_fallbacks() == 2

    def test_round_trip_through_dict(self):
        item = baseline(tag_counts={"no_data": 1}, p95_latency_ms=12.5, mean_latency_ms=8.0)
        restored = RagBaseline.from_dict(item.to_dict())
        assert restored.to_dict() == item.to_dict()

    def test_save_and_load(self, tmp_path):
        item = baseline()
        target = item.save(tmp_path / "nested" / "rag_baseline.json")
        assert target.exists()
        loaded = RagBaseline.load(target)
        assert loaded.metrics == pytest.approx(item.metrics)

    def test_load_missing_file_raises(self, tmp_path):
        # 刻意不兜底："没有基线"与"基线全为零"是两回事。
        with pytest.raises(FileNotFoundError, match="质量基线文件不存在"):
            RagBaseline.load(tmp_path / "absent.json")

    def test_label_must_be_non_empty(self):
        with pytest.raises(BaselineError, match="非空字符串"):
            baseline(label="  ")

    def test_samples_must_be_non_negative_int(self):
        with pytest.raises(BaselineError, match="非负整数"):
            baseline(samples=-1)

    def test_k_must_be_positive_int(self):
        with pytest.raises(BaselineError, match=">= 1 的整数"):
            baseline(k=0)

    def test_metrics_table_is_closed(self):
        row = metrics_row()
        row.pop(METRIC_RETRIEVAL_RECALL)
        with pytest.raises(BaselineError, match="封闭清单|缺"):
            baseline(metrics=row)
        extra = metrics_row()
        extra["hit_rate"] = 0.5
        with pytest.raises(BaselineError, match="多"):
            baseline(metrics=extra)

    def test_metrics_values_must_be_ratios(self):
        with pytest.raises(BaselineError, match=r"\[0, 1\]"):
            baseline(metrics=metrics_row(**{METRIC_GROUNDED_RATE: 1.4}))

    def test_metrics_values_must_be_numbers(self):
        row = metrics_row()
        row[METRIC_GROUNDED_RATE] = "高"
        with pytest.raises(BaselineError, match="必须是数字"):
            baseline(metrics=row)

    def test_fallback_counts_must_be_dict(self):
        with pytest.raises(BaselineError, match="必须是字典"):
            baseline(fallback_counts=["llm_error"])

    def test_fallback_counts_keys_are_closed(self):
        with pytest.raises(BaselineError, match="清单之外的键"):
            baseline(fallback_counts={"模型挂了": 1})

    def test_tag_counts_keys_are_closed(self):
        with pytest.raises(BaselineError, match="清单之外的键"):
            baseline(tag_counts={"something": 1})

    def test_count_values_must_be_non_negative(self):
        with pytest.raises(BaselineError, match="非负整数"):
            baseline(tag_counts={"no_data": -1})

    def test_latency_must_be_finite(self):
        with pytest.raises(BaselineError, match="非负有限数"):
            baseline(p95_latency_ms=float("nan"))

    def test_latency_must_be_number(self):
        with pytest.raises(BaselineError, match="必须是数字"):
            baseline(mean_latency_ms="快")

    def test_unknown_metric_lookup_rejected(self):
        with pytest.raises(BaselineError, match="不认识的指标名"):
            baseline().value("hit_rate")
        with pytest.raises(BaselineError, match="不认识的指标名"):
            baseline().direction("hit_rate")


class TestDeltasBetween:
    def test_higher_better_improvement(self):
        rows = deltas_between(
            baseline(),
            baseline(metrics=metrics_row(**{METRIC_GROUNDED_RATE: 0.9})),
        )
        item = next(entry for entry in rows if entry.metric == METRIC_GROUNDED_RATE)
        assert item.delta == pytest.approx(0.4)
        assert item.verdict == "improved"
        assert item.direction == DIRECTION_HIGHER_BETTER

    def test_higher_better_regression(self):
        rows = deltas_between(
            baseline(),
            baseline(metrics=metrics_row(**{METRIC_GROUNDED_RATE: 0.1})),
        )
        item = next(entry for entry in rows if entry.metric == METRIC_GROUNDED_RATE)
        assert item.verdict == "degraded"

    def test_lower_better_direction_is_reversed(self):
        rows = deltas_between(
            baseline(),
            baseline(metrics=metrics_row(**{METRIC_HALLUCINATION_RATE: 0.9})),
        )
        item = next(entry for entry in rows if entry.metric == METRIC_HALLUCINATION_RATE)
        assert item.verdict == "degraded"
        assert "↓" in item.summary_line()

    def test_two_thousandths_is_unchanged(self):
        rows = deltas_between(
            baseline(),
            baseline(metrics=metrics_row(**{METRIC_GROUNDED_RATE: 0.502})),
        )
        item = next(entry for entry in rows if entry.metric == METRIC_GROUNDED_RATE)
        assert item.verdict == "unchanged"

    def test_neutral_metric_is_never_judged(self):
        rows = deltas_between(
            baseline(),
            baseline(metrics=metrics_row(**{METRIC_LLM_CALL_RATE: 0.0})),
        )
        item = next(entry for entry in rows if entry.metric == METRIC_LLM_CALL_RATE)
        assert item.direction == DIRECTION_NEUTRAL
        assert item.verdict == "unchanged"

    def test_zero_baseline_still_yields_a_verdict(self):
        # 质量侧用差值判定的理由之一：0 基数下差值照样可判。
        rows = deltas_between(
            baseline(metrics=metrics_row(**{METRIC_HALLUCINATION_RATE: 0.0})),
            baseline(metrics=metrics_row(**{METRIC_HALLUCINATION_RATE: 0.2})),
        )
        item = next(entry for entry in rows if entry.metric == METRIC_HALLUCINATION_RATE)
        assert item.verdict == "degraded"

    def test_unknown_tolerance_rejected(self):
        with pytest.raises(BaselineError, match="清单之外的指标"):
            deltas_between(baseline(), baseline(), tolerances={"hit_rate": 0.1})

    def test_inputs_must_be_baselines(self):
        with pytest.raises(BaselineError, match="都必须是 RagBaseline"):
            deltas_between(baseline(), {"label": "x"})  # type: ignore[arg-type]

    def test_dead_band_must_be_non_negative(self):
        with pytest.raises(BaselineError, match="非负数字"):
            deltas_between(baseline(), baseline(), dead_band=-0.1)


class TestGuard:
    def test_no_regression_when_identical(self):
        comparison = RagBaselineGuard(baseline()).compare(baseline())
        assert comparison.ok is True
        assert comparison.conclusive is True
        assert comparison.regressions == ()
        assert "质量未回归" in comparison.summary_line()

    def test_regression_is_reported_with_direction(self):
        guard = RagBaselineGuard(baseline())
        current = baseline(metrics=metrics_row(**{METRIC_GROUNDED_RATE: 0.2}))
        comparison = guard.compare(current)
        assert comparison.ok is False
        assert len(comparison.regressions) == 1
        assert comparison.regressions[0].metric == METRIC_GROUNDED_RATE
        assert "超出容忍度" in comparison.regressions[0].message
        assert "质量回归 1 项" in comparison.summary_line()

    def test_improvement_makes_ok_true(self):
        guard = RagBaselineGuard(baseline())
        current = baseline(metrics=metrics_row(**{METRIC_GROUNDED_RATE: 1.0}))
        comparison = guard.compare(current)
        assert comparison.ok is True
        assert [item.metric for item in comparison.improved] == [METRIC_GROUNDED_RATE]

    def test_insufficient_samples_is_not_a_pass(self):
        guard = RagBaselineGuard(baseline(), min_samples=5)
        comparison = guard.compare(baseline(samples=2))
        assert comparison.conclusive is False
        assert comparison.ok is False
        assert comparison.regressions == ()
        assert "不下结论" in comparison.summary_line()

    def test_k_change_is_inconclusive(self):
        guard = RagBaselineGuard(baseline())
        comparison = guard.compare(baseline(k=5))
        assert comparison.conclusive is False
        assert any("口径变了" in reason for reason in comparison.reasons)

    def test_index_version_change_is_inconclusive(self):
        guard = RagBaselineGuard(baseline())
        comparison = guard.compare(baseline(index_version="idx-2"))
        assert comparison.conclusive is False
        assert any("条件变了" in reason for reason in comparison.reasons)

    def test_prompt_version_change_is_a_variable_not_a_blocker(self):
        guard = RagBaselineGuard(baseline())
        comparison = guard.compare(baseline(prompt_version="v1"))
        assert comparison.conclusive is True
        assert comparison.variables
        assert "提示词版本" in comparison.variables[0]

    def test_custom_tolerances_and_dead_band(self):
        guard = RagBaselineGuard(baseline(), tolerances={METRIC_GROUNDED_RATE: 0.4})
        # 掉了 0.3：按缺省容忍度 0.05 是回归，按自定义的 0.4 不是。
        current = baseline(metrics=metrics_row(**{METRIC_GROUNDED_RATE: 0.2}))
        assert guard.compare(current).ok is True
        assert RagBaselineGuard(baseline()).compare(current).ok is False

    def test_tolerance_lookup(self):
        guard = RagBaselineGuard(baseline())
        assert guard.tolerance(METRIC_GROUNDED_RATE) == pytest.approx(0.05)
        with pytest.raises(BaselineError, match="不认识的指标名"):
            guard.tolerance("hit_rate")

    def test_guard_validates_inputs(self):
        with pytest.raises(BaselineError, match="必须是 RagBaseline"):
            RagBaselineGuard({"label": "x"})  # type: ignore[arg-type]

    def test_guard_rejects_unknown_tolerances(self):
        with pytest.raises(BaselineError, match="清单之外的指标"):
            RagBaselineGuard(baseline(), tolerances={"hit_rate": 0.1})

    def test_guard_rejects_bad_dead_band_and_samples(self):
        with pytest.raises(BaselineError, match="非负数字"):
            RagBaselineGuard(baseline(), dead_band=-1.0)
        with pytest.raises(BaselineError, match=">= 1 的整数"):
            RagBaselineGuard(baseline(), min_samples=0)

    def test_compare_requires_baseline(self):
        with pytest.raises(BaselineError, match="必须是 RagBaseline"):
            RagBaselineGuard(baseline()).compare({"label": "x"})  # type: ignore[arg-type]

    def test_single_metric_delta_entry(self):
        guard = RagBaselineGuard(baseline())
        item = guard._delta(METRIC_GROUNDED_RATE, baseline())
        assert item.metric == METRIC_GROUNDED_RATE
        assert item.verdict == "unchanged"

    def test_comparison_projection(self):
        guard = RagBaselineGuard(baseline())
        data = guard.compare(baseline()).to_dict()
        assert data["ok"] is True
        assert data["conclusive"] is True
        assert len(data["deltas"]) == len(QUALITY_METRICS)
        assert data["regressions"] == []
        # 投影里的每一项都要能被 json.dumps 序列化（端点直接返回它）。
        assert json.loads(json.dumps(data, ensure_ascii=False))["ok"] is True


class TestSettingsWiring:
    def test_guard_from_settings_uses_resolved_values(self):
        guard = guard_from_settings(baseline())
        assert guard.dead_band == pytest.approx(DEFAULT_DEAD_BAND)
        assert guard.min_samples == DEFAULT_MIN_SAMPLES
        assert guard.tolerance(METRIC_GROUNDED_RATE) == pytest.approx(0.05)

    def test_compare_with_file(self, tmp_path):
        target = baseline().save(tmp_path / "rag_baseline.json")
        comparison = compare_with_file(baseline(), target)
        assert comparison.ok is True

    def test_compare_with_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            compare_with_file(baseline(), tmp_path / "absent.json")
