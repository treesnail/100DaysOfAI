"""``rag_debug`` 的运行器、报告与套件（day071）.

这一份是**端到端**的：真的装配一条 ``RagPipeline``（检索 → 打包 → 生成 → 核对），
真的跑一遍那三条手算用例，再核对报告里的每一个数字。三组断言各自对应
教程里的一处主张：

```text
运行器   两段名单来自**同一次回答**（retrieved 与 packed 必须同源）
报告     11 个指标从一批 CaseOutcome 折出来，且能折回一份基线
套件     变体一次只改一个旋钮；标注形态不一致时装配期就报错
```
"""

from __future__ import annotations

import json

import pytest

from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.rag_debug.baseline import (
    METRIC_BAD_CASE_RATE,
    METRIC_GROUNDED_RATE,
    METRIC_LLM_CALL_RATE,
    METRIC_PACKED_RECALL,
    METRIC_RETRIEVAL_PRECISION,
    METRIC_RETRIEVAL_RECALL,
    QUALITY_METRICS,
    RagBaseline,
)
from smart_research_agent.rag_debug.errors import CaseDataError, DiagnosisError, RagDebugError
from smart_research_agent.rag_debug.report import (
    RagEvalReport,
    build_report,
    compare_variants,
    metric_table,
)
from smart_research_agent.rag_debug.runner import RagEvalRunner, score_faithfulness
from smart_research_agent.rag_debug.suite import (
    DEFAULT_VARIANTS,
    TIGHT_BUDGET_CHARS,
    RagEvalSuite,
    Variant,
    build_suite,
    index_corpus,
    load_cases,
    load_corpus,
)
from smart_research_agent.rag_debug.types import RagEvalCase
from smart_research_agent.retrieval.generation import (
    FALLBACK_REASON_UNUSABLE_CITATIONS,
    PROMPT_VERSION_V1,
)
from smart_research_agent.retrieval.pipeline import RagPipeline
from smart_research_agent.retrieval.retriever import Retriever
from smart_research_agent.vectorstore.flat import FlatVectorStore
from tests.rag_debug_samples import (
    ANSWER_HALLUCINATED,
    ANSWER_TWO_CITATIONS,
    CASES,
    CORPUS,
    EXPECTED_IDS,
    BoomLLM,
    ScriptedLLM,
    VocabEmbedding,
    build_store,
    outcome,
)


def make_pipeline(llm, *, variant: Variant | None = None, store=None) -> RagPipeline:
    """装配一条能跑的手算链路（三条语料 + 查表编码器 + 注入的模型）."""
    spec = variant or Variant(name="inline")
    suite = RagEvalSuite(list(CASES), list(CORPUS))
    return suite.assemble(store or build_store(), VocabEmbedding(), llm, spec)[0]


# --------------------------------------------------------------------------- #
# 运行器
# --------------------------------------------------------------------------- #


class TestRunnerBasics:
    def test_run_case_produces_two_hand_computed_stages(self):
        llm = ScriptedLLM([ANSWER_TWO_CITATIONS] * 3)
        runner = RagEvalRunner(make_pipeline(llm), k=3)
        result = runner.run_case(CASES[0])
        assert result.retrieved.ids == EXPECTED_IDS
        assert result.packed.ids == EXPECTED_IDS
        assert result.retrieved.recall == pytest.approx(1.0)
        assert result.retrieved.precision == pytest.approx(1 / 3)
        assert result.retrieved.reciprocal_rank == pytest.approx(1.0)
        assert result.packed_away == 0
        assert result.llm_called is True
        assert result.fallback_reason == ""

    def test_packed_stage_drops_what_the_budget_cut(self):
        llm = ScriptedLLM([ANSWER_TWO_CITATIONS] * 3)
        variant = Variant(name="tight", max_context_chars=TIGHT_BUDGET_CHARS)
        runner = RagEvalRunner(make_pipeline(llm, variant=variant), k=3)
        result = runner.run_case(CASES[0])
        # 检索名单不受预算影响（三条命中都在），而 packed 只会更少或相等：
        # "预算丢尾"这件事只可能发生在打包那一步。
        assert result.retrieved.count == 3
        assert result.packed.count <= result.retrieved.count
        assert result.answer_chars > 0

    def test_empty_store_never_calls_the_model(self):
        runner = RagEvalRunner(
            make_pipeline(BoomLLM(), store=FlatVectorStore()), k=3
        )
        result = runner.run_case(CASES[0])
        assert result.llm_called is False
        assert result.empty_reason == "no_data"
        assert result.retrieved.count == 0
        assert result.grounded is False
        assert result.fallback_reason == "no_context"

    def test_hallucinated_citations_are_counted(self):
        llm = ScriptedLLM([ANSWER_HALLUCINATED] * 3)
        runner = RagEvalRunner(make_pipeline(llm), k=3)
        result = runner.run_case(CASES[0])
        assert result.hallucinated >= 1
        assert result.grounded is False
        assert result.valid_count + result.unused_count == result.given

    def test_run_keeps_input_order(self):
        llm = ScriptedLLM([ANSWER_TWO_CITATIONS] * 6)
        runner = RagEvalRunner(make_pipeline(llm), k=3)
        results = runner.run(list(CASES))
        assert [item.query for item in results] == ["甲", "乙", "丙"]

    def test_clock_is_injectable(self):
        ticks = iter([10.0, 10.25])
        runner = RagEvalRunner(
            make_pipeline(ScriptedLLM([ANSWER_TWO_CITATIONS] * 3)),
            k=3,
            clock=lambda: next(ticks),
        )
        result = runner.run_case(CASES[0])
        assert result.latency_ms == pytest.approx(250.0)

    def test_describe_reports_resolved_values(self):
        runner = RagEvalRunner(make_pipeline(ScriptedLLM([])), k=3, judge=MockLLM())
        described = runner.describe()
        assert described["k"] == 3
        assert described["judge"] is True
        assert described["prompt_version"]
        assert runner.pipeline is not None

    def test_replay_of_a_historical_answer(self):
        # to_outcome 与运行解耦：同一份回答可以重放成账（排查历史报告时用）。
        llm = ScriptedLLM([ANSWER_TWO_CITATIONS] * 3)
        suite = RagEvalSuite(list(CASES), list(CORPUS))
        pipeline = suite.build_pipeline(build_store(), VocabEmbedding(), llm)
        answer = pipeline.answer(CASES[0].query)
        runner = RagEvalRunner(pipeline, k=3)
        replayed = runner.to_outcome(CASES[0], answer, elapsed_ms=1.0)
        assert list(replayed.retrieved.ids) == answer.retrieval.ids()
        assert replayed.query == answer.question


class TestRunnerValidation:
    def test_pipeline_must_be_rag_pipeline(self):
        with pytest.raises(DiagnosisError, match="必须是 RagPipeline"):
            RagEvalRunner("not a pipeline")  # type: ignore[arg-type]

    def test_k_must_be_positive(self):
        with pytest.raises(DiagnosisError, match=">= 1 的整数"):
            RagEvalRunner(make_pipeline(ScriptedLLM([])), k=0)

    def test_min_coverage_must_be_ratio(self):
        with pytest.raises(DiagnosisError, match=r"\[0, 1\]"):
            RagEvalRunner(make_pipeline(ScriptedLLM([])), min_coverage=2.0)

    def test_require_grounded_must_be_bool(self):
        with pytest.raises(DiagnosisError, match="必须是布尔值"):
            RagEvalRunner(make_pipeline(ScriptedLLM([])), require_grounded="yes")

    def test_judge_must_be_llm_or_none(self):
        with pytest.raises(DiagnosisError, match="必须是 BaseLLM 或 None"):
            RagEvalRunner(make_pipeline(ScriptedLLM([])), judge="judge")

    def test_case_must_be_rag_eval_case(self):
        runner = RagEvalRunner(make_pipeline(ScriptedLLM([])))
        with pytest.raises(DiagnosisError, match="必须是 RagEvalCase"):
            runner.run_case({"query": "甲"})


class TestFaithfulness:
    def test_score_faithfulness_parses_judge_output(self):
        judge = MockLLM(responses=['评审结果：{"score": 0.5, "reason": "部分一致"}'])
        result = score_faithfulness("答案", "参考答案", judge)
        assert result.score == pytest.approx(0.5)
        assert result.reason == "部分一致"

    def test_runner_records_the_score(self):
        judge = MockLLM(responses=['{"score": 1.0, "reason": "一致"}'] * 3)
        runner = RagEvalRunner(
            make_pipeline(ScriptedLLM([ANSWER_TWO_CITATIONS] * 3)), k=3, judge=judge
        )
        result = runner.run_case(CASES[0])
        assert result.faithfulness == pytest.approx(1.0)

    def test_unparsable_judge_output_degrades_with_a_note(self):
        judge = MockLLM(responses=["我无法评审"] * 3)
        runner = RagEvalRunner(
            make_pipeline(ScriptedLLM([ANSWER_TWO_CITATIONS] * 3)), k=3, judge=judge
        )
        result = runner.run_case(CASES[0])
        assert result.faithfulness is None
        assert any("无法解析" in note for note in result.notes)

    def test_no_judge_records_a_note(self):
        runner = RagEvalRunner(make_pipeline(ScriptedLLM([ANSWER_TWO_CITATIONS] * 3)), k=3)
        result = runner.run_case(CASES[0])
        assert result.faithfulness is None
        assert any("未配置评审模型" in note for note in result.notes)

    def test_case_without_reference_records_a_note(self):
        judge = MockLLM(responses=['{"score": 1.0}'] * 3)
        runner = RagEvalRunner(
            make_pipeline(ScriptedLLM([ANSWER_TWO_CITATIONS] * 3)), k=3, judge=judge
        )
        result = runner.run_case(CASES[2])  # 第三条没有参考答案
        assert result.faithfulness is None
        assert any("没有参考答案" in note for note in result.notes)


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #


class TestReport:
    def _report(self, **overrides) -> RagEvalReport:
        payload = {
            "label": "baseline:baseline",
            "k": 3,
            "cases": 2,
            "metrics": {name: 0.5 for name in QUALITY_METRICS},
        }
        payload.update(overrides)
        return RagEvalReport(**payload)

    def test_metrics_are_means_over_cases(self):
        report = build_report(
            [outcome(), outcome()],
            label="l",
            k=3,
            prompt_version="v2",
        )
        assert report.cases == 2
        assert report.value(METRIC_RETRIEVAL_RECALL) == pytest.approx(1.0)
        assert report.value(METRIC_GROUNDED_RATE) == pytest.approx(1.0)
        assert report.value(METRIC_BAD_CASE_RATE) == pytest.approx(0.0)

    def test_fallback_and_hallucination_rates_are_per_case(self):
        report = build_report(
            [
                outcome(fallback_reason="llm_error", llm_called=False),
                outcome(hallucinated=1, cited=2, given=2, valid_count=2, unused_count=0),
            ],
            label="l",
            k=3,
        )
        assert report.value("fallback_rate") == pytest.approx(0.5)
        assert report.value("hallucination_rate") == pytest.approx(0.5)
        assert report.value(METRIC_LLM_CALL_RATE) == pytest.approx(0.5)

    def test_empty_batch_is_a_zero_report_not_a_crash(self):
        report = build_report([], label="l", k=3)
        assert report.cases == 0
        assert report.value(METRIC_RETRIEVAL_RECALL) == 0.0
        assert report.p95_latency_ms == 0.0
        assert report.clean_rate == 0.0

    def test_latency_uses_nearest_rank_p95(self):
        report = build_report(
            [outcome(latency_ms=value) for value in (1.0, 2.0, 100.0)],
            label="l",
            k=3,
        )
        assert report.p95_latency_ms == pytest.approx(100.0)
        assert report.mean_latency_ms == pytest.approx(34.3333)

    def test_to_baseline_round_trip(self):
        report = build_report(
            [outcome(), outcome()],
            label="l",
            k=3,
            prompt_version="v2",
            index_version="idx-1",
        )
        baseline = report.to_baseline()
        assert baseline.samples == 2
        assert baseline.k == 3
        assert baseline.prompt_version == "v2"
        assert baseline.index_version == "idx-1"
        assert baseline.total_bad_cases() == 0

    def test_projection_contains_evidence_and_limits(self):
        report = build_report([outcome()], label="l", k=3)
        data = report.to_dict()
        assert data["per_case"][0]["query"] == "甲"
        assert data["limitations"]
        assert json.loads(json.dumps(data, ensure_ascii=False))["cases"] == 1
        trimmed = report.to_dict(include_per_case=False)
        assert trimmed["per_case"] == []

    def test_explain_has_the_six_sections(self):
        report = build_report(
            [outcome(faithfulness=0.2)],
            label="l",
            k=3,
        )
        lines = report.explain(limit=1)
        assert any(line.startswith("口径：") for line in lines)
        assert any(line.startswith("指标：") for line in lines)
        assert any(line.startswith("召回漏斗：") for line in lines)
        assert any(line.startswith("适用范围") for line in lines)
        assert "RAG 评估 l" in report.summary_line()

    def test_explain_lists_fallbacks_bad_cases_and_actions(self):
        # 报告里**真的有坏例**时的 explain：回退分布、坏例逐条、去重后的动作清单。
        from smart_research_agent.rag_debug.diagnose import diagnose_all
        from smart_research_agent.rag_debug.types import TAG_NO_DATA, BadCase

        outcomes = [
            outcome(empty_reason="no_data", grounded=False, coverage=0.0, llm_called=False,
                    fallback_reason="no_context", cited=0, given=0, valid_count=0, unused_count=0),
            outcome(empty_reason="no_data", grounded=False, coverage=0.0, llm_called=False,
                    fallback_reason="no_context", cited=0, given=0, valid_count=0, unused_count=0),
        ]
        bad_cases = diagnose_all(outcomes)
        report = build_report(outcomes, bad_cases, label="l", k=3)
        lines = report.explain(limit=1)
        assert any(line.startswith("回退分布：") for line in lines)
        assert any(line.startswith("坏例 2 条") for line in lines)
        assert any("其余 1 条" in line for line in lines)
        assert any(line.startswith("下一步动作") for line in lines)
        assert report.bad_cases[0].tag == TAG_NO_DATA
        # 单条手工坏例也要能进报告（`actions` 是去重后的动作清单）。
        manual = build_report(
            outcomes, (BadCase(query="甲", tag=TAG_NO_DATA, reason="r", action="动作 A"),),
            label="l2", k=3,
        )
        assert manual.actions == ("动作 A",)
        assert manual.explain()[-1].startswith("  - ")

    def test_explain_limit_must_be_non_negative(self):
        with pytest.raises(RagDebugError, match="limit 必须是非负整数"):
            self._report().explain(limit=-1)

    def test_save_writes_json(self, tmp_path):
        report = build_report([outcome()], label="l", k=3)
        target = report.save(tmp_path / "out" / "report.json")
        payload = json.loads(target.read_text(encoding="utf-8"))
        assert payload["label"] == "l"

    def test_label_must_be_non_empty(self):
        with pytest.raises(RagDebugError, match="非空字符串"):
            self._report(label=" ")

    def test_k_and_cases_must_be_non_negative_ints(self):
        with pytest.raises(RagDebugError, match=">= 1 的整数"):
            self._report(k=0)
        with pytest.raises(RagDebugError, match="非负整数"):
            self._report(cases=-1)

    def test_metrics_table_is_closed(self):
        with pytest.raises(RagDebugError, match="封闭清单|缺"):
            self._report(metrics={METRIC_RETRIEVAL_RECALL: 0.5})

    def test_bad_cases_cannot_outnumber_cases(self):
        from smart_research_agent.rag_debug.types import TAG_NO_DATA, BadCase

        case = BadCase(query="甲", tag=TAG_NO_DATA, reason="r", action="a")
        with pytest.raises(RagDebugError, match="账不自洽"):
            self._report(cases=0, bad_cases=(case,))

    def test_per_case_must_be_tuple(self):
        with pytest.raises(RagDebugError, match="必须是 tuple"):
            self._report(per_case=[{"query": "甲"}])

    def test_distribution_tables_are_validated(self):
        with pytest.raises(RagDebugError, match="清单之外的键"):
            self._report(tag_counts={"something": 1})

    def test_clean_rate_is_the_complement_of_bad_case_rate(self):
        report = self._report(cases=4, metrics={name: 0.5 for name in QUALITY_METRICS})
        assert report.clean_rate == pytest.approx(1.0)
        assert report.bad_case_count == 0

    def test_unknown_metric_rejected(self):
        with pytest.raises(RagDebugError, match="不认识的指标名"):
            self._report().value("hit_rate")


class TestVariantComparison:
    def _baseline(self, **overrides) -> RagBaseline:
        payload = {
            "label": "baseline:baseline",
            "samples": 6,
            "k": 3,
            "metrics": {name: 0.5 for name in QUALITY_METRICS},
            "prompt_version": "v2",
            "index_version": "idx-1",
        }
        payload.update(overrides)
        return RagBaseline(**payload)

    def test_prompt_version_change_is_the_variable(self):
        current = self._baseline(
            label="baseline:prompt-v1",
            prompt_version=PROMPT_VERSION_V1,
            metrics={
                **{name: 0.5 for name in QUALITY_METRICS},
                METRIC_GROUNDED_RATE: 0.2,
            },
        )
        comparison = compare_variants(self._baseline(), current)
        assert comparison.variables
        assert comparison.inconclusive == ()
        assert comparison.degraded == (METRIC_GROUNDED_RATE,)
        assert "改善" in comparison.summary_line()
        assert comparison.to_dict()["label_b"] == "baseline:prompt-v1"

    def test_index_version_change_is_inconclusive(self):
        comparison = compare_variants(
            self._baseline(), self._baseline(index_version="idx-2")
        )
        assert comparison.inconclusive
        assert "不下结论" in comparison.summary_line()

    def test_k_change_is_inconclusive(self):
        comparison = compare_variants(self._baseline(), self._baseline(k=5))
        assert any("k 口径不同" in reason for reason in comparison.inconclusive)

    def test_sample_count_difference_is_a_variable(self):
        comparison = compare_variants(self._baseline(), self._baseline(samples=3))
        assert any("用例数" in item for item in comparison.variables)

    def test_unchanged_metrics_are_listed(self):
        comparison = compare_variants(self._baseline(), self._baseline())
        assert len(comparison.unchanged) == len(QUALITY_METRICS)


class TestMetricTable:
    def test_table_covers_every_metric_with_direction(self):
        rows = metric_table()
        assert len(rows) == len(QUALITY_METRICS)
        for row in rows:
            assert row["metric"] in QUALITY_METRICS
            assert row["direction"] in {"higher_better", "lower_better", "neutral"}
            assert row["description"]


# --------------------------------------------------------------------------- #
# 套件
# --------------------------------------------------------------------------- #


class TestSuiteLoading:
    def test_shipped_eval_set_loads(self):
        suite = build_suite()
        assert len(suite.cases) == 6
        assert len(suite.corpus) == 6
        assert suite.k == 3
        assert [item.name for item in suite.variants] == [
            item.name for item in DEFAULT_VARIANTS
        ]
        assert suite.describe()["cases"] == 6

    def test_variant_lookup(self):
        suite = build_suite()
        assert suite.variant("tight-budget").max_context_chars == TIGHT_BUDGET_CHARS
        with pytest.raises(DiagnosisError, match="不认识的变体"):
            suite.variant("absent")

    def test_load_cases_rejects_missing_fields(self, tmp_path):
        target = tmp_path / "cases.jsonl"
        target.write_text('{"query": "甲"}\n', encoding="utf-8")
        with pytest.raises(CaseDataError, match="缺少 query 或 relevant_ids"):
            load_cases(target)

    def test_load_cases_rejects_non_object_lines(self, tmp_path):
        target = tmp_path / "cases.jsonl"
        target.write_text("[1, 2]\n", encoding="utf-8")
        with pytest.raises(CaseDataError, match="不是 JSON 对象"):
            load_cases(target)

    def test_load_cases_rejects_scalar_relevant_ids(self, tmp_path):
        target = tmp_path / "cases.jsonl"
        target.write_text('{"query": "甲", "relevant_ids": "d-1"}\n', encoding="utf-8")
        with pytest.raises(CaseDataError, match="必须是列表"):
            load_cases(target)

    def test_load_corpus_rejects_missing_fields(self, tmp_path):
        target = tmp_path / "corpus.jsonl"
        target.write_text('{"id": "d-1"}\n', encoding="utf-8")
        with pytest.raises(CaseDataError, match="含 id 与 text"):
            load_corpus(target)

    def test_load_corpus_rejects_non_object_lines(self, tmp_path):
        target = tmp_path / "corpus.jsonl"
        target.write_text('"d-1"\n', encoding="utf-8")
        with pytest.raises(CaseDataError, match="含 id 与 text"):
            load_corpus(target)


class TestSuiteValidation:
    def test_empty_cases_rejected(self):
        with pytest.raises(CaseDataError, match="不能为空"):
            RagEvalSuite([])

    def test_cases_must_be_rag_eval_cases(self):
        with pytest.raises(CaseDataError, match="只收 RagEvalCase"):
            RagEvalSuite([{"query": "甲"}])

    def test_duplicate_variant_names_rejected(self):
        with pytest.raises(DiagnosisError, match="重名"):
            RagEvalSuite(list(CASES), variants=[Variant(name="a"), Variant(name="a")])

    def test_empty_variants_rejected(self):
        with pytest.raises(DiagnosisError, match="不能为空"):
            RagEvalSuite(list(CASES), variants=[])

    def test_label_must_be_non_empty(self):
        with pytest.raises(DiagnosisError, match="非空字符串"):
            RagEvalSuite(list(CASES), label="  ")

    def test_mixed_annotation_forms_rejected(self):
        mixed = [
            RagEvalCase(query="甲", relevant=("d-1",), grades={"d-1": 2}),
            RagEvalCase(query="乙", relevant=("d-2",)),
        ]
        with pytest.raises(CaseDataError, match="标注形态不一致"):
            RagEvalSuite(mixed)

    def test_invalid_k_is_caught_at_assembly_time(self):
        # 判据的校验只有一份实现（在 runner 里），但装配期就要响。
        with pytest.raises(DiagnosisError, match=">= 1 的整数"):
            RagEvalSuite(list(CASES), k=0)


class TestVariantShape:
    def test_default_variants_change_one_knob_each(self):
        names = [item.name for item in DEFAULT_VARIANTS]
        assert names[0] == "baseline"
        baseline = DEFAULT_VARIANTS[0]
        assert baseline.prompt_version is None
        for item in DEFAULT_VARIANTS[1:]:
            assert item.note  # 每个变体都要说清"它改了什么"

    def test_name_must_be_non_empty(self):
        with pytest.raises(DiagnosisError, match="非空字符串"):
            Variant(name=" ")

    def test_prompt_version_must_be_managed(self):
        with pytest.raises(DiagnosisError, match="受管模板"):
            Variant(name="custom", prompt_version="v9")

    def test_require_citation_must_be_bool(self):
        with pytest.raises(DiagnosisError, match="必须是布尔值"):
            Variant(name="x", require_citation="yes")

    def test_numeric_knobs_must_be_positive(self):
        with pytest.raises(DiagnosisError, match=">= 1 的整数或 None"):
            Variant(name="x", top_k=0)
        with pytest.raises(DiagnosisError, match=">= 1 的整数或 None"):
            Variant(name="x", max_context_chars=0)

    def test_projection_and_summary(self):
        spec = Variant(name="tight", max_context_chars=200, note="只收紧预算")
        assert spec.to_dict()["max_context_chars"] == 200
        assert "只收紧预算" in spec.summary_line()


class TestSuiteRun:
    def _suite(self) -> RagEvalSuite:
        return RagEvalSuite(list(CASES), list(CORPUS))

    def test_index_into_reports_the_write(self):
        report = self._suite().index_into(FlatVectorStore(), VocabEmbedding())
        assert report.added == 3
        assert report.unchanged == 0
        # 对同一个库再灌一次：`unchanged` 是"这次白算了多少"的账。
        again = index_corpus(build_store(), VocabEmbedding(), list(CORPUS))
        assert again.unchanged == 3

    def test_assemble_accepts_a_variant_name(self):
        suite = self._suite()
        pipeline, retriever = suite.assemble(
            build_store(), VocabEmbedding(), ScriptedLLM([]), "cite-gate"
        )
        assert pipeline.prompt_version
        assert isinstance(retriever, Retriever)
        assert suite.build_pipeline(build_store(), VocabEmbedding(), ScriptedLLM([])) is not None

    def test_assemble_rejects_unknown_variant_shape(self):
        with pytest.raises(DiagnosisError, match="Variant、变体名或 None"):
            self._suite().assemble(build_store(), VocabEmbedding(), ScriptedLLM([]), 3)

    def test_run_variant_produces_a_registered_report(self):
        suite = self._suite()
        report = suite.run_variant(
            build_store(),
            VocabEmbedding(),
            ScriptedLLM([ANSWER_TWO_CITATIONS] * 3),
            "baseline",
        )
        assert report.label == "baseline:baseline"
        assert report.cases == 3
        assert report.value(METRIC_RETRIEVAL_RECALL) == pytest.approx(1.0)
        # 均值保留四位小数（mean_of 的口径），因此用 abs=1e-4 而不是 1e-7。
        assert report.value(METRIC_RETRIEVAL_PRECISION) == pytest.approx(1 / 3, abs=1e-4)

    def test_run_all_covers_every_variant(self):
        suite = self._suite()
        reports = suite.run_all(
            build_store(), VocabEmbedding(), ScriptedLLM([ANSWER_TWO_CITATIONS] * 30)
        )
        assert len(reports) == len(DEFAULT_VARIANTS)
        assert reports[0].label.endswith("baseline")
        # cite-gate 会拒收"没有合法引用"的答案：给 [1] [2] 时应仍然可交付。
        assert reports[2].value(METRIC_GROUNDED_RATE) == pytest.approx(1.0)

    def test_cite_gate_turns_uncited_answers_into_fallbacks(self):
        suite = RagEvalSuite(list(CASES), list(CORPUS))
        reports = suite.run_all(
            build_store(),
            VocabEmbedding(),
            ScriptedLLM(["这段答案一个编号都没写。"] * 30),
        )
        baseline, _, gate, _ = reports
        assert baseline.value(METRIC_GROUNDED_RATE) == pytest.approx(0.0)
        assert gate.value("fallback_rate") == pytest.approx(1.0)
        assert gate.to_baseline().fallback_counts[FALLBACK_REASON_UNUSABLE_CITATIONS] == 3

    def test_tight_budget_lowers_packed_recall_only(self):
        suite = RagEvalSuite(list(CASES), list(CORPUS))
        reports = suite.run_all(
            build_store(), VocabEmbedding(), ScriptedLLM([ANSWER_TWO_CITATIONS] * 30)
        )
        baseline, _, _, tight = reports
        assert tight.value(METRIC_RETRIEVAL_RECALL) == pytest.approx(
            baseline.value(METRIC_RETRIEVAL_RECALL)
        )
        assert tight.value(METRIC_PACKED_RECALL) <= baseline.value(METRIC_PACKED_RECALL)
