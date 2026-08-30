"""评估框架测试：数据集加载、evaluate 流程、summary 统计与报告写盘."""

from __future__ import annotations

import json

import pytest

from smart_research_agent.agent.react_agent import ReactAgent
from smart_research_agent.evaluation import (
    AgentEvalHarness,
    EvalCase,
    EvaluationHarness,
)
from smart_research_agent.evaluation.agent_harness import (
    contains_scorer,
    exact_match_scorer,
)
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.tools.calculator import CalculatorTool
from smart_research_agent.tools.registry import ToolRegistry

DATASET_JSONL = (
    '{"id": "case-1", "input": "计算 2 + 3 等于几", "expected": "5", "tags": ["math"]}\n'
    '{"id": "case-2", "input": "法国的首都是哪里？", "expected": "巴黎", "tags": ["qa", "knowledge"]}\n'
)


def _write_dataset(tmp_path, content: str = DATASET_JSONL):
    path = tmp_path / "eval.jsonl"
    path.write_text(content, encoding="utf-8")
    return path


class TestLoadDataset:
    def test_load_basic(self, tmp_path):
        cases = EvaluationHarness.load_dataset(_write_dataset(tmp_path))
        assert len(cases) == 2
        assert cases[0] == EvalCase(id="case-1", input="计算 2 + 3 等于几", expected="5", tags=["math"])
        assert cases[1].tags == ["qa", "knowledge"]

    def test_tags_default_empty(self, tmp_path):
        path = _write_dataset(tmp_path, '{"id": "c", "input": "i", "expected": "e"}\n')
        assert EvaluationHarness.load_dataset(path)[0].tags == []

    def test_blank_lines_skipped(self, tmp_path):
        path = _write_dataset(tmp_path, "\n" + DATASET_JSONL + "\n\n")
        assert len(EvaluationHarness.load_dataset(path)) == 2

    def test_malformed_json_raises(self, tmp_path):
        path = _write_dataset(tmp_path, '{"id": "c1", 不是合法 JSON\n')
        with pytest.raises(ValueError, match="JSON"):
            EvaluationHarness.load_dataset(path)

    def test_missing_field_raises(self, tmp_path):
        path = _write_dataset(tmp_path, '{"id": "c1", "input": "只有输入"}\n')
        with pytest.raises(ValueError, match="字段"):
            EvaluationHarness.load_dataset(path)


class TestScorers:
    def test_exact_match(self):
        assert exact_match_scorer("5", "5") == 1.0
        assert exact_match_scorer(" 5 ", "5") == 1.0
        assert exact_match_scorer("答案是 5", "5") == 0.0

    def test_contains(self):
        assert contains_scorer("答案是 5", "5") == 1.0
        assert contains_scorer("答案是 6", "5") == 0.0


def _make_factory(scripts: dict[str, list[str]]):
    """按用例 id 选择 MockLLM 脚本的 Agent 工厂."""

    def factory(case: EvalCase) -> ReactAgent:
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        llm = MockLLM(responses=list(scripts[case.id]))
        return ReactAgent(llm=llm, registry=registry)

    return factory


PASS_SCRIPT = [
    "Thought: 这是算术题，调用计算器\nAction: calculator\nAction Input: 2 + 3",
    "Thought: 计算器返回 5，可以回答了\nFinal Answer: 5",
]
FAIL_SCRIPT = ["Thought: 我不知道\nFinal Answer: 伦敦"]


class TestAgentEvalHarness:
    def _run(self, tmp_path):
        scripts = {"case-1": PASS_SCRIPT, "case-2": FAIL_SCRIPT}
        harness = AgentEvalHarness(agent_factory=_make_factory(scripts))
        harness.run(_write_dataset(tmp_path))
        return harness

    def test_evaluate_single_case(self, tmp_path):
        harness = AgentEvalHarness(
            agent_factory=_make_factory({"case-1": PASS_SCRIPT})
        )
        result = harness.evaluate(
            EvalCase(id="case-1", input="计算 2 + 3 等于几", expected="5")
        )
        assert result.case_id == "case-1"
        assert result.output == "5"
        assert result.score == 1.0
        assert result.passed is True
        assert result.latency_seconds >= 0.0
        assert result.details["expected"] == "5"

    def test_run_all_cases(self, tmp_path):
        harness = self._run(tmp_path)
        assert [r.case_id for r in harness.results] == ["case-1", "case-2"]
        assert harness.results[0].passed is True
        assert harness.results[1].passed is False

    def test_summary_statistics(self, tmp_path):
        summary = self._run(tmp_path).summary()
        assert summary["total"] == 2
        assert summary["passed"] == 1
        assert summary["pass_rate"] == 0.5
        assert summary["avg_score"] == 0.5
        assert summary["avg_latency_seconds"] >= 0.0

    def test_summary_by_tag(self, tmp_path):
        summary = self._run(tmp_path).summary()
        assert summary["by_tag"]["math"] == {
            "total": 1,
            "passed": 1,
            "pass_rate": 1.0,
            "avg_score": 1.0,
        }
        assert summary["by_tag"]["knowledge"]["passed"] == 0
        assert summary["by_tag"]["qa"]["total"] == 1

    def test_summary_empty(self, tmp_path):
        harness = AgentEvalHarness(agent_factory=_make_factory({}))
        harness.run(_write_dataset(tmp_path, "\n"))
        assert harness.summary() == {
            "total": 0,
            "passed": 0,
            "pass_rate": 0.0,
            "avg_score": 0.0,
            "avg_latency_seconds": 0.0,
            "by_tag": {},
        }

    def test_custom_scorer(self, tmp_path):
        scripts = {"case-2": ["Thought: 思考一下\nFinal Answer: 法国的首都是巴黎"]}
        harness = AgentEvalHarness(
            agent_factory=_make_factory(scripts), scorer=exact_match_scorer
        )
        result = harness.evaluate(
            EvalCase(id="case-2", input="法国的首都是哪里？", expected="巴黎")
        )
        # 精确匹配下"法国的首都是巴黎"不等于"巴黎"
        assert result.score == 0.0
        assert result.passed is False

    def test_write_report(self, tmp_path):
        harness = self._run(tmp_path)
        report_path = harness.write_report(harness.summary(), tmp_path / "report.json")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["summary"]["total"] == 2
        assert report["summary"]["pass_rate"] == 0.5
        assert len(report["results"]) == 2
        assert report["results"][0]["case_id"] == "case-1"
        assert report["results"][0]["passed"] is True
