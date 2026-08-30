"""EvaluationHarness 基座测试：批量执行与汇总."""

from __future__ import annotations

from smart_research_agent.evaluation.harness import CaseResult, EvalCase, EvaluationHarness


class EchoHarness(EvaluationHarness):
    """最简具体实现：expected 是 input 的子串则通过."""

    def run_case(self, case: EvalCase) -> CaseResult:
        passed = case.expected is None or case.expected in case.input
        return CaseResult(case=case, output=case.input, passed=passed)


class TestEvaluationHarness:
    def test_run_returns_results_in_order(self):
        cases = [
            EvalCase(name="c1", input="hello world", expected="world"),
            EvalCase(name="c2", input="hello", expected="missing"),
        ]
        results = EchoHarness().run(cases)
        assert [r.case.name for r in results] == ["c1", "c2"]
        assert [r.passed for r in results] == [True, False]

    def test_summary_counts(self):
        results = EchoHarness().run(
            [
                EvalCase(name="c1", input="abc", expected="a"),
                EvalCase(name="c2", input="abc", expected="b"),
                EvalCase(name="c3", input="abc", expected="z"),
            ]
        )
        summary = EchoHarness.summary(results)
        assert summary == {"total": 3, "passed": 2, "failed": 1, "pass_rate": round(2 / 3, 4)}

    def test_summary_empty(self):
        assert EchoHarness.summary([]) == {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0}
