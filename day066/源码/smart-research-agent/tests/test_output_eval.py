"""输出质量评估测试：格式校验、数字一致性、双重打分、质量日志（MockLLM 驱动，完全离线）."""

from __future__ import annotations

import json

import pytest

from smart_research_agent.agent.react_agent import ReactAgent
from smart_research_agent.evaluation.harness import EvaluationHarness
from smart_research_agent.evaluation.output_eval import OutputEvaluator, OutputScore
from smart_research_agent.evaluation.quality_logger import OutputQualityLogger
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.tools.calculator import CalculatorTool
from smart_research_agent.tools.registry import ToolRegistry

RUBRIC_OK = '{"helpfulness": 5, "accuracy": 4, "issues": []}'


def _make_evaluator(**kwargs) -> OutputEvaluator:
    kwargs.setdefault("llm", MockLLM(responses=[RUBRIC_OK]))
    return OutputEvaluator(**kwargs)


class TestFormatCheck:
    def test_json_format_valid(self):
        ev = _make_evaluator(expected_format="json")
        score = ev.evaluate('{"结论": "RAG 值得采用"}')
        assert score.format_compliance == 1.0
        assert score.issues == []

    def test_json_format_with_code_fence(self):
        ev = _make_evaluator(expected_format="json")
        score = ev.evaluate('```json\n{"a": 1}\n```')
        assert score.format_compliance == 1.0

    def test_json_format_invalid(self):
        ev = _make_evaluator(expected_format="json")
        score = ev.evaluate("这不是 JSON，只是一段话")
        assert score.format_compliance == 0.5
        assert any("JSON" in i for i in score.issues)

    def test_markdown_format_missing_structure(self):
        ev = _make_evaluator(expected_format="markdown")
        score = ev.evaluate("就一句话，没有任何 markdown 结构")
        assert score.format_compliance == 0.5

    def test_length_too_short(self):
        ev = _make_evaluator(min_length=10)
        score = ev.evaluate("太短")
        assert score.format_compliance == 0.5
        assert any("过短" in i for i in score.issues)

    def test_length_too_long(self):
        ev = _make_evaluator(max_length=10)
        score = ev.evaluate("这一段话远远超过了十个字符的长度上限")
        assert score.format_compliance == 0.5
        assert any("过长" in i for i in score.issues)

    def test_multiple_violations_floor_at_zero(self):
        ev = _make_evaluator(expected_format="json", min_length=100)
        score = ev.evaluate("短")
        assert score.format_compliance == 0.0
        assert len(score.issues) == 2

    def test_unsupported_format_raises(self):
        with pytest.raises(ValueError, match="expected_format"):
            _make_evaluator(expected_format="xml")


class TestNumericConsistency:
    def test_numbers_grounded_in_observation(self):
        ev = _make_evaluator()
        score = ev.evaluate("计算结果是 5.0", observations=["5.0"])
        assert score.accuracy == pytest.approx(0.5 * 1.0 + 0.5 * 0.8)  # 规则满分 + 模型 4/5
        assert not any("编造" in i for i in score.issues)

    def test_hallucinated_number_flagged(self):
        ev = _make_evaluator()
        score = ev.evaluate("答案是 42", observations=["5.0"])
        assert any("42" in i and "编造" in i for i in score.issues)
        # 规则通道 0 分，拉低 accuracy
        assert score.accuracy == pytest.approx(0.5 * 0.0 + 0.5 * 0.8)

    def test_no_observations_skips_check(self):
        ev = _make_evaluator()
        score = ev.evaluate("任何数字 999 都不检查", observations=[])
        assert score.accuracy == pytest.approx(0.5 * 1.0 + 0.5 * 0.8)

    def test_output_without_numbers_skips_check(self):
        ev = _make_evaluator()
        score = ev.evaluate("结论是框架 A 更好", observations=["检索到 3 个框架"])
        assert not any("编造" in i for i in score.issues)


class TestModelScoring:
    def test_rubric_scores_reflected(self):
        llm = MockLLM(responses=['{"helpfulness": 5, "accuracy": 2, "issues": ["数据过时"]}'])
        ev = OutputEvaluator(llm=llm)
        score = ev.evaluate("一份不错的答案")
        assert score.helpfulness == 1.0
        assert score.accuracy == pytest.approx(0.5 * 1.0 + 0.5 * 0.4)
        assert "数据过时" in score.issues

    def test_rubric_json_with_surrounding_text(self):
        llm = MockLLM(responses=['好的，评分如下：{"helpfulness": 4, "accuracy": 4, "issues": []} 完毕'])
        ev = OutputEvaluator(llm=llm)
        score = ev.evaluate("答案")
        assert score.helpfulness == 0.8

    def test_invalid_model_output_falls_back_to_rules(self):
        llm = MockLLM(responses=["我拒绝打分"])
        ev = OutputEvaluator(llm=llm)
        score = ev.evaluate("一份正常的答案", observations=[])
        # 降级为纯规则：helpfulness 用长度启发，accuracy 用数字规则
        assert score.helpfulness == 1.0
        assert score.accuracy == 1.0
        assert score.overall == pytest.approx(1.0)

    def test_out_of_range_scores_fall_back(self):
        llm = MockLLM(responses=['{"helpfulness": 99, "accuracy": 4, "issues": []}'])
        ev = OutputEvaluator(llm=llm)
        score = ev.evaluate("一份正常的答案")
        assert score.helpfulness == 1.0  # 已走降级分支

    def test_weighted_overall(self):
        llm = MockLLM(responses=['{"helpfulness": 5, "accuracy": 5, "issues": []}'])
        ev = OutputEvaluator(llm=llm, expected_format="json", weights=(0.4, 0.4, 0.2))
        score = ev.evaluate("不是 json")  # 格式违规 -> format=0.5
        expected = 0.4 * 1.0 + 0.4 * 1.0 + 0.2 * 0.5
        assert score.overall == pytest.approx(expected)


class TestOutputScore:
    def test_to_dict_rounds_and_keeps_issues(self):
        s = OutputScore(
            helpfulness=0.12345, accuracy=1.0, format_compliance=0.5, overall=0.8, issues=["x"]
        )
        d = s.to_dict()
        assert d["helpfulness"] == 0.1235
        assert d["issues"] == ["x"]


class TestQualityLogger:
    def _score(self, overall: float) -> OutputScore:
        return OutputScore(
            helpfulness=overall,
            accuracy=overall,
            format_compliance=overall,
            overall=overall,
            issues=["示例问题"] if overall < 1.0 else [],
        )

    def test_log_appends_jsonl(self, tmp_path):
        log = OutputQualityLogger(log_path=tmp_path / "q.jsonl")
        log.log("任务一", "输出一", self._score(0.8))
        log.log("任务二", "输出二", self._score(1.0))
        lines = (tmp_path / "q.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        record = json.loads(lines[0])
        assert record["task"] == "任务一"
        assert record["overall"] == 0.8
        assert "ts" in record

    def test_aggregate_empty(self, tmp_path):
        log = OutputQualityLogger(log_path=tmp_path / "q.jsonl")
        assert log.aggregate() == {"count": 0, "trend": "no_data"}

    def test_aggregate_averages_and_issues(self, tmp_path):
        log = OutputQualityLogger(log_path=tmp_path / "q.jsonl")
        log.log("t1", "o1", self._score(0.5))
        log.log("t2", "o2", self._score(1.0))
        agg = log.aggregate()
        assert agg["count"] == 2
        assert agg["avg_overall"] == 0.75
        assert agg["trend"] == "insufficient_history"
        assert agg["top_issues"] == [("示例问题", 1)]

    def test_aggregate_trend_improving(self, tmp_path):
        log = OutputQualityLogger(log_path=tmp_path / "q.jsonl", recent_window=2)
        for _ in range(3):
            log.log("t", "o", self._score(0.4))
        for _ in range(2):
            log.log("t", "o", self._score(0.9))
        assert log.aggregate()["trend"] == "improving"

    def test_aggregate_trend_degrading(self, tmp_path):
        log = OutputQualityLogger(log_path=tmp_path / "q.jsonl", recent_window=2)
        for _ in range(3):
            log.log("t", "o", self._score(0.9))
        for _ in range(2):
            log.log("t", "o", self._score(0.4))
        assert log.aggregate()["trend"] == "degrading"


class TestHarness:
    def test_load_and_run(self, tmp_path):
        dataset = tmp_path / "cases.jsonl"
        dataset.write_text(
            '{"input": "1+1=?", "expected": "2"}\n{"input": "2+2=?", "expected": "4"}\n',
            encoding="utf-8",
        )
        cases = EvaluationHarness.load_cases(dataset)
        assert len(cases) == 2
        assert cases[0].expected == "2"

        harness = EvaluationHarness(
            fn=lambda text: text.split("=")[0] + "= 4",  # 永远答 4
            scorer=lambda output, expected: 1.0 if expected and expected in output else 0.0,
        )
        summary = harness.run(cases)
        assert summary["total"] == 2
        assert summary["passed"] == 1
        assert summary["pass_rate"] == 0.5
        assert summary["avg_score"] == 0.5


class TestReactAgentIntegration:
    def _make_agent(self, log_path, responses) -> ReactAgent:
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        evaluator = OutputEvaluator(llm=MockLLM(responses=[RUBRIC_OK]))
        return ReactAgent(
            llm=MockLLM(responses=responses),
            registry=registry,
            output_evaluator=evaluator,
            quality_logger=OutputQualityLogger(log_path=log_path),
        )

    def test_final_answer_is_evaluated_and_logged(self, tmp_path):
        log_path = tmp_path / "q.jsonl"
        agent = self._make_agent(
            log_path,
            [
                "Thought: 需要先计算 2+3\nAction: calculator\nAction Input: 2+3",
                "Thought: 已得到结果\nFinal Answer: 计算结果是 5.0",
            ],
        )
        answer = agent.run("2+3 等于几？")
        assert "5" in answer
        record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
        assert record["task"] == "2+3 等于几？"
        assert record["format_compliance"] == 1.0
        assert record["overall"] > 0.8  # 数字有 Observation 支撑，模型打满分

    def test_hallucinated_number_logged_as_issue(self, tmp_path):
        log_path = tmp_path / "q.jsonl"
        agent = self._make_agent(
            log_path,
            [
                "Thought: 先算一下\nAction: calculator\nAction Input: 2+3",
                "Thought: 编个别的数\nFinal Answer: 计算结果是 42",
            ],
        )
        agent.run("2+3 等于几？")
        record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
        assert any("编造" in i for i in record["issues"])

    def test_without_evaluator_behavior_unchanged(self, tmp_path):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        agent = ReactAgent(
            llm=MockLLM(responses=["Thought: 直接答\nFinal Answer: 答案是 5"]),
            registry=registry,
        )
        assert agent.run("任意任务") == "答案是 5"
        assert not (tmp_path / "q.jsonl").exists()

    def test_evaluator_exception_does_not_break_run(self):
        class BrokenEvaluator:
            def evaluate(self, *args, **kwargs):
                raise RuntimeError("打分服务宕机")

        registry = ToolRegistry()
        registry.register(CalculatorTool())
        agent = ReactAgent(
            llm=MockLLM(responses=["Thought: 直接答\nFinal Answer: 答案是 5"]),
            registry=registry,
            output_evaluator=BrokenEvaluator(),
        )
        assert agent.run("任意任务") == "答案是 5"
