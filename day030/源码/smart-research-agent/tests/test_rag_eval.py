"""RAG 评估模块测试：四个指标的手算用例 + 端到端评测集跑通.

所有期望值均先在纸上手算（推导过程见教程第三章），再固化进断言。
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from smart_research_agent.evaluation.harness import EvalCase, EvalHarness
from smart_research_agent.evaluation.rag_eval import (
    FaithfulnessParseError,
    RagEvaluator,
    load_jsonl,
)
from smart_research_agent.evaluation.rag_metrics import (
    mrr,
    ndcg,
    retrieval_precision,
    retrieval_recall,
)
from smart_research_agent.llm.embedding import EmbeddingProvider, MockEmbedding
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.memory.vector_store import InMemoryVectorStore

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "eval"


class TestRetrievalRecall:
    def test_half_recalled(self):
        # retrieved ∩ relevant = {a}，relevant 共 2 个 → 1/2
        assert retrieval_recall(["a", "b", "c"], ["a", "d"]) == pytest.approx(0.5)

    def test_full_recall(self):
        assert retrieval_recall(["a", "d"], ["a", "d"]) == pytest.approx(1.0)

    def test_zero_recall(self):
        assert retrieval_recall(["x", "y"], ["a"]) == pytest.approx(0.0)

    def test_empty_relevant_is_vacuous_full(self):
        # 没有该召回的文档，约定 1.0
        assert retrieval_recall(["a"], []) == pytest.approx(1.0)

    def test_duplicates_count_once(self):
        # 同一文档重复返回不算重复召回
        assert retrieval_recall(["a", "a", "a"], ["a", "d"]) == pytest.approx(0.5)


class TestRetrievalPrecision:
    def test_one_of_three(self):
        # retrieved 共 3 个，命中 {a} → 1/3
        assert retrieval_precision(["a", "b", "c"], ["a", "d"]) == pytest.approx(1 / 3)

    def test_perfect_precision(self):
        assert retrieval_precision(["a", "d"], ["a", "d"]) == pytest.approx(1.0)

    def test_empty_retrieved(self):
        assert retrieval_precision([], ["a"]) == pytest.approx(0.0)

    def test_no_hits(self):
        assert retrieval_precision(["x", "y"], ["a"]) == pytest.approx(0.0)


class TestMrr:
    def test_first_hit_at_rank_1(self):
        assert mrr(["a", "b"], ["a"]) == pytest.approx(1.0)

    def test_first_hit_at_rank_2(self):
        assert mrr(["x", "a", "y"], ["a"]) == pytest.approx(0.5)

    def test_first_hit_at_rank_3(self):
        assert mrr(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)

    def test_no_hit(self):
        assert mrr(["x", "y"], ["a"]) == pytest.approx(0.0)

    def test_only_first_hit_matters(self):
        # 第一个命中在第 1 位，后面排什么都不影响
        assert mrr(["a", "x"], ["a", "x"]) == pytest.approx(1.0)


class TestNdcg:
    def test_perfect_order_is_one(self):
        # 实际排序 = 理想排序（等级 2 在前、1 在后）→ NDCG = 1
        assert ndcg(["a", "b", "c"], {"a": 2, "b": 1, "c": 0}) == pytest.approx(1.0)

    def test_inverted_pair(self):
        # retrieved = [b(rel=1), a(rel=2)]：
        # DCG  = 1/log2(2) + 3/log2(3) = 1 + 1.8927893 = 2.8927893
        # IDCG = 3/log2(2) + 1/log2(3) = 3 + 0.6309298 = 3.6309298
        # NDCG = 2.8927893 / 3.6309298 ≈ 0.7967
        expected = (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3))
        assert ndcg(["b", "a"], {"a": 2, "b": 1}) == pytest.approx(expected)
        assert expected == pytest.approx(0.7967, abs=1e-4)

    def test_ungraded_doc_treated_as_zero(self):
        # "x" 不在 grades 里 → rel=0，排第 1 位零增益但仍占坑位
        # DCG = 0 + 3/log2(3)，IDCG = 3 → NDCG = 1/log2(3) ≈ 0.6309
        assert ndcg(["x", "a"], {"a": 2}) == pytest.approx(1 / math.log2(3))

    def test_empty_grades(self):
        assert ndcg(["a"], {}) == pytest.approx(0.0)

    def test_all_zero_grades(self):
        assert ndcg(["a"], {"a": 0}) == pytest.approx(0.0)


class ToyEmbedding(EmbeddingProvider):
    """词表到标准基的确定性 embedding：已知词各占一个维度，未知词忽略.

    维度 4，词表 {"向量": 0, "索引": 1, "排序": 2, "检索": 3}。
    这样每个语料的向量、每次检索的排序都可以完全手算。
    """

    VOCAB = {"向量": 0, "索引": 1, "排序": 2, "检索": 3}

    @property
    def dimension(self) -> int:
        return 4

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * 4
        for word, idx in self.VOCAB.items():
            if word in text:
                vec[idx] = 1.0
        return vec


class TestRetrievalEndToEnd:
    """受控语料的端到端检索评估：排序与四个指标全部手算核对.

    语料（ToyEmbedding 向量）：
        t1 "向量 检索"  → [1,0,0,1]
        t2 "索引"       → [0,1,0,0]
        t3 "排序 索引"  → [0,1,1,0]
    """

    CORPUS = [
        {"id": "t1", "text": "向量 检索"},
        {"id": "t2", "text": "索引"},
        {"id": "t3", "text": "排序 索引"},
    ]

    EVAL_SET = [
        # q1 向量 [1,0,0,1]：余弦 t1=1.0, t2=0, t3=0（并列按插入序）→ top2 = [t1, t2]
        {
            "query": "向量 检索",
            "relevant_ids": ["t1"],
            "relevance_grades": {"t1": 2, "t2": 1},
        },
        # q2 向量 [0,1,0,0]：余弦 t1=0, t2=1.0, t3=1/√2≈0.7071 → top2 = [t2, t3]
        {
            "query": "索引",
            "relevant_ids": ["t3"],
            "relevance_grades": {"t3": 2, "t2": 1},
        },
    ]

    def _run(self):
        evaluator = RagEvaluator(vector_store=InMemoryVectorStore(), embedding=ToyEmbedding())
        evaluator.index_documents(self.CORPUS)
        return evaluator.evaluate_retrieval(self.EVAL_SET, top_k=2)

    def test_per_query_metrics(self):
        report = self._run()
        q1, q2 = report.per_query

        assert q1.retrieved_ids == ["t1", "t2"]
        assert q1.recall == pytest.approx(1.0)  # {t1} 全部召回
        assert q1.precision == pytest.approx(0.5)  # 2 条结果 1 条相关
        assert q1.mrr == pytest.approx(1.0)  # 首个命中在第 1 位
        assert q1.ndcg == pytest.approx(1.0)  # 排序即理想排序

        assert q2.retrieved_ids == ["t2", "t3"]
        assert q2.recall == pytest.approx(1.0)
        assert q2.precision == pytest.approx(0.5)
        assert q2.mrr == pytest.approx(0.5)  # 首个命中在第 2 位
        # DCG = 1/log2(2) + 3/log2(3)，IDCG = 3 + 1/log2(3)
        assert q2.ndcg == pytest.approx((1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3)))

    def test_mean_metrics(self):
        report = self._run()
        assert report.mean_recall == pytest.approx(1.0)
        assert report.mean_precision == pytest.approx(0.5)
        assert report.mean_mrr == pytest.approx(0.75)
        expected_ndcg = (1.0 + (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3))) / 2
        assert report.mean_ndcg == pytest.approx(expected_ndcg)

    def test_report_to_dict_structure(self):
        data = self._run().to_dict()
        assert data["top_k"] == 2
        assert data["num_queries"] == 2
        assert len(data["per_query"]) == 2
        assert set(data["per_query"][0]) == {
            "query", "retrieved_ids", "recall", "precision", "mrr", "ndcg",
        }


class TestShippedEvalSet:
    """项目自带的 data/eval/ 评测集：MockEmbedding 无真实语义，

    只断言流水线跑通与输出结构/值域合法，不断言具体指标数值。
    """

    def test_eval_set_runs_end_to_end(self):
        corpus = load_jsonl(DATA_DIR / "rag_corpus.jsonl")
        eval_set = load_jsonl(DATA_DIR / "rag_eval.jsonl")
        assert len(eval_set) == 6

        evaluator = RagEvaluator(vector_store=InMemoryVectorStore(), embedding=MockEmbedding())
        evaluator.index_documents(corpus)
        report = evaluator.evaluate_retrieval(eval_set, top_k=3)

        assert len(report.per_query) == 6
        for row in report.per_query:
            assert len(row.retrieved_ids) == 3
            for value in (row.recall, row.precision, row.mrr, row.ndcg):
                assert 0.0 <= value <= 1.0
        for value in (
            report.mean_recall,
            report.mean_precision,
            report.mean_mrr,
            report.mean_ndcg,
        ):
            assert 0.0 <= value <= 1.0


class TestAnswerFaithfulness:
    def _evaluator(self) -> RagEvaluator:
        return RagEvaluator(vector_store=InMemoryVectorStore(), embedding=MockEmbedding())

    def test_full_score(self):
        judge = MockLLM(responses=['{"score": 1.0, "reason": "关键事实完全一致"}'])
        result = self._evaluator().answer_faithfulness("答案", "参考答案", judge)
        assert result.score == pytest.approx(1.0)
        assert result.reason == "关键事实完全一致"

    def test_json_wrapped_in_prose(self):
        judge = MockLLM(responses=['评审结果：{"score": 0.5, "reason": "部分一致"} 完毕'])
        result = self._evaluator().answer_faithfulness("答案", "参考答案", judge)
        assert result.score == pytest.approx(0.5)

    def test_prompt_contains_answer_and_reference(self):
        judge = MockLLM(responses=['{"score": 1.0}'])
        self._evaluator().answer_faithfulness("我的答案", "标准答案", judge)
        prompt = judge.calls[0][0].content
        assert "我的答案" in prompt and "标准答案" in prompt

    def test_no_json_raises(self):
        judge = MockLLM(responses=["无法评审"])
        with pytest.raises(FaithfulnessParseError):
            self._evaluator().answer_faithfulness("答案", "参考答案", judge)

    def test_score_out_of_range_raises(self):
        judge = MockLLM(responses=['{"score": 2.0}'])
        with pytest.raises(FaithfulnessParseError, match="越界"):
            self._evaluator().answer_faithfulness("答案", "参考答案", judge)

    def test_score_not_numeric_raises(self):
        judge = MockLLM(responses=['{"score": "很高"}'])
        with pytest.raises(FaithfulnessParseError):
            self._evaluator().answer_faithfulness("答案", "参考答案", judge)


class TestHarness:
    """简版 harness：用例隔离与通过率汇总."""

    def test_pass_and_fail_are_aggregated(self):
        harness = EvalHarness()
        harness.add(EvalCase(name="ok", data={}, check=lambda d: (True, "")))
        harness.add(EvalCase(name="bad", data={}, check=lambda d: (False, "不符合预期")))
        report = harness.run()
        assert report.total == 2
        assert report.passed == 1
        assert report.pass_rate == pytest.approx(0.5)

    def test_exception_is_caught_not_propagated(self):
        def boom(_data):
            raise RuntimeError("炸了")

        harness = EvalHarness()
        harness.add(EvalCase(name="boom", data={}, check=boom))
        harness.add(EvalCase(name="ok", data={}, check=lambda d: (True, "")))
        report = harness.run()
        assert report.results[0].passed is False
        assert "炸了" in report.results[0].detail
        assert report.results[1].passed is True  # 后续用例不受影响

    def test_empty_suite(self):
        report = EvalHarness().run()
        assert report.total == 0
        assert report.pass_rate == pytest.approx(0.0)
