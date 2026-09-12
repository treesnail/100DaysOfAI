"""本地/云端对比评估测试（day045）：延迟、成本、一致性三个口径（全离线）."""

from __future__ import annotations

import pytest

from smart_research_agent.evaluation.local_eval import (
    CaseResult,
    ComparisonReport,
    LocalModelEvaluator,
    ModelRun,
)
from smart_research_agent.llm.embedding import MockEmbedding
from smart_research_agent.llm.mock import MockLLM, estimate_tokens


def _clock_from(ticks: list[float]):
    """把一串时间刻度变成 clock 函数（每次调用取下一个）."""
    iterator = iter(ticks)

    def clock() -> float:
        return next(iterator)

    return clock


class TestModelRun:
    def test_empty_run_has_zero_latency(self):
        run = ModelRun(label="local")
        assert run.mean_latency_s == 0.0
        assert run.total_latency_s == 0.0
        assert run.total_tokens == 0
        assert run.estimated_cost() == 0.0

    def test_cost_is_tokens_per_1k_times_price(self):
        run = ModelRun(
            label="cloud",
            results=[
                CaseResult(
                    task="t",
                    reply="r",
                    latency_s=0.5,
                    prompt_tokens=100,
                    completion_tokens=100,
                )
            ],
            price_per_1k=0.01,
        )
        assert run.total_tokens == 200
        assert run.estimated_cost() == pytest.approx(0.002)


class TestLocalModelEvaluator:
    def test_identical_replies_reach_full_agreement(self):
        tasks = ["什么是 RAG？", "解释向量检索"]
        local = MockLLM(responses=["检索增强生成。", "用向量做相似度检索。"] * 1)
        cloud = MockLLM(responses=["检索增强生成。", "用向量做相似度检索。"])
        report = LocalModelEvaluator(local, cloud, clock=_clock_from([0.0] * 8)).run(tasks)
        assert report.agreement == pytest.approx(1.0)
        assert report.local.results[0].task == "什么是 RAG？"
        assert len(report.local.results) == 2
        assert "可替代" in report.verdict()

    def test_divergent_replies_trigger_verdict_warning(self):
        tasks = ["t1", "t2"]
        local = MockLLM(responses=["苹果香蕉橘子", "本地小模型的口语化回答"])
        cloud = MockLLM(responses=["苹果香蕉橘子", "向量数据库通过余弦相似度完成最近邻检索"])
        report = LocalModelEvaluator(
            local, cloud, clock=_clock_from([0.0] * 8), agreement_threshold=0.99
        ).run(tasks)
        assert report.similarities[0] == pytest.approx(1.0)
        assert report.agreement < 0.99
        assert "需谨慎" in report.verdict()
        worst = report.worst_case()
        assert worst is not None
        assert worst.task == "t2"

    def test_latency_uses_injected_clock(self):
        tasks = ["a", "b"]
        local = MockLLM(responses=["x", "y"])
        cloud = MockLLM(responses=["x", "y"])
        # 本地每条 1s/2s，云端每条 0.5s/1.0s
        ticks = [0.0, 1.0, 1.0, 3.0, 0.0, 0.5, 0.5, 1.5]
        report = LocalModelEvaluator(local, cloud, clock=_clock_from(ticks)).run(tasks)
        assert report.local.mean_latency_s == pytest.approx(1.5)
        assert report.cloud.mean_latency_s == pytest.approx(0.75)
        assert report.latency_ratio == pytest.approx(2.0)

    def test_cost_saved_counts_local_price(self):
        tasks = ["a"]
        local = MockLLM(responses=["相同的回答"])
        cloud = MockLLM(responses=["相同的回答"])
        report = LocalModelEvaluator(
            local,
            cloud,
            clock=_clock_from([0.0, 0.1, 0.0, 0.1]),
            cloud_price_per_1k=0.02,
            local_price_per_1k=0.001,
        ).run(tasks)
        assert report.cloud.estimated_cost() > report.local.estimated_cost()
        assert report.cost_saved == pytest.approx(
            report.cloud.estimated_cost() - report.local.estimated_cost()
        )

    def test_system_prompt_counted_in_prompt_tokens(self):
        system_prompt = "你是一个严谨的研究助手。"
        task = "介绍一下 RAG"
        local = MockLLM(responses=["回答"])
        cloud = MockLLM(responses=["回答"])
        report = LocalModelEvaluator(
            local,
            cloud,
            clock=_clock_from([0.0, 0.1, 0.0, 0.1]),
            system_prompt=system_prompt,
        ).run([task])
        expected = estimate_tokens(system_prompt) + estimate_tokens(task)
        assert report.local.results[0].prompt_tokens == expected
        assert report.cloud.results[0].prompt_tokens == expected

    def test_empty_tasks_yield_empty_report(self):
        report = LocalModelEvaluator(MockLLM(), MockLLM()).run([])
        assert report.agreement == 0.0
        assert report.latency_ratio == 0.0
        assert report.worst_case() is None
        assert "无评估样本" in report.verdict()
        assert report.to_dict()["cases"] == 0

    def test_rejects_out_of_range_threshold(self):
        with pytest.raises(ValueError, match="agreement_threshold"):
            LocalModelEvaluator(MockLLM(), MockLLM(), agreement_threshold=1.5)

    def test_injected_embedder_is_used(self):
        """注入 MockEmbedding：同文本得到同一伪向量，相似度仍为 1.0."""
        report = LocalModelEvaluator(
            MockLLM(responses=["一致的回复"]),
            MockLLM(responses=["一致的回复"]),
            embedder=MockEmbedding(dimension=16),
            clock=_clock_from([0.0, 0.1, 0.0, 0.1]),
        ).run(["任务"])
        assert report.agreement == pytest.approx(1.0)

    def test_to_dict_exposes_decision_fields(self):
        report = LocalModelEvaluator(
            MockLLM(responses=["回答"]),
            MockLLM(responses=["回答"]),
            clock=_clock_from([0.0, 0.1, 0.0, 0.2]),
        ).run(["任务"])
        data = report.to_dict()
        assert set(data) == {
            "local",
            "cloud",
            "cases",
            "agreement",
            "agreement_threshold",
            "mean_latency_s",
            "latency_ratio",
            "cost",
        }
        assert data["mean_latency_s"] == {"local": 0.1, "cloud": 0.2}
        assert data["cost"]["local"] == 0.0


class TestComparisonReportHelpers:
    def test_verdict_without_samples(self):
        report = ComparisonReport(local=ModelRun("local"), cloud=ModelRun("cloud"))
        assert report.worst_case() is None
        assert "无评估样本" in report.verdict()

    def test_latency_ratio_guards_zero_cloud_latency(self):
        """云端均延迟为 0（如即时缓存命中）时不能除零."""
        report = ComparisonReport(
            local=ModelRun(
                label="local",
                results=[CaseResult("t", "r", 1.0, 10, 10)],
            ),
            cloud=ModelRun(
                label="cloud",
                results=[CaseResult("t", "r", 0.0, 10, 10)],
            ),
        )
        assert report.latency_ratio == 0.0

    def test_worst_case_none_when_similarity_length_mismatch(self):
        report = ComparisonReport(
            local=ModelRun(label="local", results=[CaseResult("t", "r", 1.0, 1, 1)]),
            cloud=ModelRun(label="cloud", results=[CaseResult("t", "r", 1.0, 1, 1)]),
            similarities=[],
        )
        assert report.worst_case() is None
