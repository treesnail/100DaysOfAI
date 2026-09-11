"""Agent 评估模块测试：指标手算用例 + 端到端评估 + 报告生成."""

from __future__ import annotations

import pytest

from smart_research_agent.agent.react_agent import ReactAgent
from smart_research_agent.evaluation.agent_eval import (
    AgentEvaluator,
    AgentRunTrace,
    _lcs_length,
    completion_rate,
    step_efficiency,
    tool_accuracy,
)
from smart_research_agent.evaluation.harness import Harness, load_jsonl
from smart_research_agent.evaluation.metrics import accuracy, contains_rate, mean
from smart_research_agent.evaluation.report import render_report, write_report
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.tools.calculator import CalculatorTool
from smart_research_agent.tools.registry import ToolRegistry

DATASET_PATH = "data/eval/agent_tasks.jsonl"


class FakeSearchTool(CalculatorTool):
    """离线测试用的假搜索工具：不联网，返回固定文本."""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "联网搜索，返回与查询相关的资料摘要"

    def execute(self, expression: str = "", **kwargs) -> str:  # noqa: ARG002
        return "模拟搜索结果"


def make_agent(responses: list[str]) -> ReactAgent:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(FakeSearchTool())
    return ReactAgent(llm=MockLLM(responses=responses), registry=registry, max_steps=6)


# ---------------------------------------------------------------------------
# 指标函数：手算用例
# ---------------------------------------------------------------------------


class TestCompletionRate:
    def test_all_success(self):
        traces = [AgentRunTrace(success=True, steps=1) for _ in range(3)]
        assert completion_rate(traces) == 1.0

    def test_partial_success(self):
        traces = [
            AgentRunTrace(success=True, steps=1),
            AgentRunTrace(success=False, steps=1),
            AgentRunTrace(success=True, steps=1),
            AgentRunTrace(success=False, steps=1),
        ]
        assert completion_rate(traces) == 0.5

    def test_empty(self):
        assert completion_rate([]) == 0.0


class TestStepEfficiency:
    def test_perfect_efficiency(self):
        # 期望 1 个工具 -> 最小必要步数 2，实际也走 2 步
        traces = [AgentRunTrace(success=True, steps=2, expected_tools=["calculator"])]
        assert step_efficiency(traces) == 1.0

    def test_redundant_step_halves_efficiency(self):
        # 最小必要 2 步，实际走 4 步 -> 2/4 = 0.5
        traces = [AgentRunTrace(success=True, steps=4, expected_tools=["calculator"])]
        assert step_efficiency(traces) == 0.5

    def test_zero_steps_scores_zero(self):
        traces = [AgentRunTrace(success=False, steps=0, expected_tools=["calculator"])]
        assert step_efficiency(traces) == 0.0

    def test_fewer_steps_capped_at_one(self):
        # 期望 2 个工具（最小 3 步）但 1 步就收工 -> 截断到 1.0，不奖励"少做事"
        traces = [
            AgentRunTrace(success=True, steps=1, expected_tools=["web_search", "calculator"])
        ]
        assert step_efficiency(traces) == 1.0

    def test_average_over_traces(self):
        traces = [
            AgentRunTrace(success=True, steps=2, expected_tools=["calculator"]),  # 1.0
            AgentRunTrace(success=True, steps=4, expected_tools=["calculator"]),  # 0.5
        ]
        assert step_efficiency(traces) == 0.75


class TestToolAccuracy:
    def test_perfect_match(self):
        traces = [
            AgentRunTrace(
                success=True,
                steps=3,
                tool_calls=["web_search", "calculator"],
                expected_tools=["web_search", "calculator"],
            )
        ]
        result = tool_accuracy(traces)
        assert result == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "order": 1.0}

    def test_extra_tool_lowers_precision(self):
        # 实际调了 [web_search, calculator]，期望只有 [calculator]
        # precision = 1/2, recall = 1, f1 = 2*0.5*1/1.5 = 2/3
        traces = [
            AgentRunTrace(
                success=True,
                steps=3,
                tool_calls=["web_search", "calculator"],
                expected_tools=["calculator"],
            )
        ]
        result = tool_accuracy(traces)
        assert result["precision"] == 0.5
        assert result["recall"] == 1.0
        assert result["f1"] == pytest.approx(2 / 3)

    def test_missing_tool_lowers_recall(self):
        # 实际只调了 [web_search]，期望 [web_search, calculator]
        traces = [
            AgentRunTrace(
                success=True,
                steps=2,
                tool_calls=["web_search"],
                expected_tools=["web_search", "calculator"],
            )
        ]
        result = tool_accuracy(traces)
        assert result["precision"] == 1.0
        assert result["recall"] == 0.5
        assert result["f1"] == pytest.approx(2 / 3)

    def test_wrong_order_lowers_order_score(self):
        # 集合完全一致（f1=1），但顺序颠倒：LCS=1，order = 1/2
        traces = [
            AgentRunTrace(
                success=True,
                steps=3,
                tool_calls=["calculator", "web_search"],
                expected_tools=["web_search", "calculator"],
            )
        ]
        result = tool_accuracy(traces)
        assert result["f1"] == 1.0
        assert result["order"] == 0.5

    def test_order_only_counts_successful_traces(self):
        # 失败轨迹不参与顺序分统计
        traces = [
            AgentRunTrace(
                success=False,
                steps=1,
                tool_calls=["calculator"],
                expected_tools=["web_search", "calculator"],
            )
        ]
        assert tool_accuracy(traces)["order"] == 0.0  # 无成功轨迹 -> mean([]) = 0.0

    def test_no_tools_called_with_expectation(self):
        traces = [AgentRunTrace(success=False, steps=1, expected_tools=["calculator"])]
        result = tool_accuracy(traces)
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0


class TestLcsLength:
    def test_identical(self):
        assert _lcs_length(["a", "b"], ["a", "b"]) == 2

    def test_reversed(self):
        assert _lcs_length(["a", "b"], ["b", "a"]) == 1

    def test_subsequence_with_noise(self):
        assert _lcs_length(["a", "x", "b"], ["a", "b"]) == 2

    def test_empty(self):
        assert _lcs_length([], ["a"]) == 0


# ---------------------------------------------------------------------------
# Harness 与通用指标
# ---------------------------------------------------------------------------


class TestHarness:
    def test_load_jsonl_skips_blank_lines(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text('{"id": "a"}\n\n{"id": "b"}\n', encoding="utf-8")
        assert load_jsonl(path) == [{"id": "a"}, {"id": "b"}]

    def test_load_jsonl_invalid_line_raises(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"id": "a"}\n这不是 JSON\n', encoding="utf-8")
        with pytest.raises(ValueError, match="第 2 行"):
            load_jsonl(path)

    def test_run_captures_case_exception(self):
        def runner(case: dict) -> dict:
            if case["id"] == "bad":
                raise RuntimeError("爆炸")
            return {"ok": True}

        result = Harness().run([{"id": "good"}, {"id": "bad"}], runner)
        assert result.total == 2
        assert len(result.case_results) == 1
        assert result.case_results[0]["id"] == "good"
        assert result.errors == ["bad: 爆炸"]


class TestMetrics:
    def test_accuracy(self):
        assert accuracy(["a", "b"], ["a", "c"]) == 0.5
        assert accuracy([], []) == 0.0

    def test_accuracy_length_mismatch(self):
        with pytest.raises(ValueError):
            accuracy(["a"], [])

    def test_contains_rate(self):
        assert contains_rate(["答案"], ["结果 42"]) == 0.0
        assert contains_rate(["42"], ["最终答案是 42"]) == 1.0

    def test_mean_empty(self):
        assert mean([]) == 0.0


# ---------------------------------------------------------------------------
# 端到端：AgentEvaluator 跑真实数据集（MockLLM 驱动，完全离线）
# ---------------------------------------------------------------------------


@pytest.fixture
def evaluation() -> dict:
    return AgentEvaluator(agent_factory=make_agent).evaluate(DATASET_PATH)


class TestAgentEvaluatorEndToEnd:
    def test_total_tasks(self, evaluation):
        assert evaluation["metrics"]["total"] == 8

    def test_completion_rate(self, evaluation):
        # task-01/02/03/07/08 成功，task-04/05/06 失败 -> 5/8
        assert evaluation["metrics"]["completion_rate"] == 0.625

    def test_step_efficiency(self, evaluation):
        # 各轨迹效率: 1, 1, 1, 1, 1(截断), 0, 2/3, 2/3 -> 平均 19/24
        assert evaluation["metrics"]["step_efficiency"] == pytest.approx(19 / 24)

    def test_tool_f1(self, evaluation):
        # 各轨迹 f1: 1,1,1,0,0,0,2/3,1 -> 平均 17/24
        assert evaluation["metrics"]["tool_accuracy"]["f1"] == pytest.approx(17 / 24)

    def test_order_score(self, evaluation):
        # 成功轨迹（01/02/03/07/08）的顺序分都是 1.0
        assert evaluation["metrics"]["tool_accuracy"]["order"] == 1.0

    def test_failure_attribution(self, evaluation):
        categories = {r.task_id: r.category for r in evaluation["results"]}
        assert categories["task-04"] == "wrong_tool"  # 该用 calculator 却调了 web_search
        assert categories["task-05"] == "planning_error"  # 直接给错误答案
        assert categories["task-06"] == "parse_failure"  # 输出无 Thought 字段

    def test_trace_records_tool_sequence(self, evaluation):
        by_id = {r.task_id: r for r in evaluation["results"]}
        assert by_id["task-03"].trace.tool_calls == ["web_search", "calculator"]
        assert by_id["task-07"].trace.tool_calls == ["web_search", "calculator"]
        assert by_id["task-03"].trace.steps == 3

    def test_latency_recorded(self, evaluation):
        for r in evaluation["results"]:
            if r.trace.error is None:
                assert r.trace.latency >= 0.0


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------


class TestReport:
    def test_render_contains_all_sections(self, evaluation):
        text = render_report(evaluation["metrics"], evaluation["results"])
        assert "## 指标总览" in text
        assert "## 薄弱维度归类" in text
        assert "## 逐条明细" in text
        assert "任务完成率 completion_rate | 0.62" in text or "0.63" in text

    def test_render_groups_failures_by_category(self, evaluation):
        text = render_report(evaluation["metrics"], evaluation["results"])
        assert "规划错误" in text
        assert "工具选错" in text
        assert "解析失败" in text
        assert "task-05" in text
        assert "task-06" in text

    def test_write_report_creates_file(self, evaluation, tmp_path):
        output = write_report(
            evaluation["metrics"], evaluation["results"], tmp_path / "report.md"
        )
        assert output.exists()
        content = output.read_text(encoding="utf-8")
        assert content.startswith("# Agent 评估报告")
        assert "| task-01 | ✅ |" in content
        assert "❌" in content

    def test_render_all_pass_case(self):
        metrics = {
            "total": 1,
            "completion_rate": 1.0,
            "step_efficiency": 1.0,
            "tool_accuracy": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "order": 1.0},
        }
        text = render_report(metrics, [])
        assert "全部任务通过，无失败用例。" in text
