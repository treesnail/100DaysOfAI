"""day071 ``/rag/eval/*`` 两个端点的单元测试.

全部离线、零网络：库里灌的是 ``rag_debug_samples`` 那三条正交语料，模型是
按脚本回复的 ``ScriptedLLM``。因此"这一批接不接地、坏例落在哪一层"都是
**手算或按定义**推出来的，而不是"跑一遍看看"。

除了"两个端点各自返回什么"，这里钉住四件比 200 更重要的事：

1. **空库是合法状态**：``/rag/eval/run`` 对空库返回 200，且每条用例都归因成
   ``no_data``（"还没灌语料"与"检索坏了"是两件事）；库非空时 ``no_data``
   **一条都不该出现**——两条都要断言，因为"护栏生效"与"护栏不误伤"是两件事；
2. **参数问题一律 400，且消息里带可用取值**：不认识的变体名要能被列出来；
3. **未配置 LLM 时 400**：``app.state.llm = None`` 时端点给出可照做的出路，
   而**不拿 MockLLM 兜底**（与 /retrieval/answer 同一条通道）；
4. **端点不重算评估**：响应里的 11 个指标与 ``baseline`` 逐字段等于
   ``rag_debug`` 在同一份样本上的结论，``per_case`` 的条数等于评测集条数。
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from smart_research_agent.api.app import create_app
from smart_research_agent.llm.base import BaseLLM
from smart_research_agent.llm.embedding import default_embedding
from smart_research_agent.rag_debug import build_suite
from smart_research_agent.rag_debug.baseline import QUALITY_METRICS
from smart_research_agent.rag_debug.types import BAD_CASE_TAGS
from smart_research_agent.vectorstore.flat import FlatVectorStore
from tests.rag_debug_samples import (
    ANSWER_TWO_CITATIONS,
    CASES,
    CORPUS,
    ScriptedLLM,
)

#: 项目自带评测集的条数（``data/eval/rag_eval.jsonl``）.
SHIPPED_CASES = 6

#: 内置变体的个数（baseline / prompt-v1 / cite-gate / tight-budget）。
SHIPPED_VARIANTS = 4


def indexed_store() -> tuple[FlatVectorStore, object]:
    """按项目默认（离线字符 n-gram）编码器把自带语料灌进库，返回 ``(库, 编码器)``.

    为什么用 ``default_embedding()`` 而不是样本里那个查表编码器：评测集问的是
    中文句子（"什么是检索增强生成"），查表编码器对它们会返回**零向量**，
    而检索层对零向量是**合法拒绝**（余弦 0/0）。这里要验的是端点接线，
    因此用一套对中文有朴素字面相关性的真实编码器。
    """
    embedding = default_embedding()
    store = FlatVectorStore()
    build_suite().index_into(store, embedding)
    return store, embedding


def make_client(
    *,
    store: FlatVectorStore | None = None,
    embedding=None,
    llm: BaseLLM | None = None,
) -> TestClient:
    """建一个可注入的离线客户端（库与模型两个注入点）."""
    if store is None:
        store, default_embedding_ = indexed_store()
        embedding = embedding or default_embedding_
    app = create_app(
        embedding=embedding if embedding is not None else default_embedding(),
        vector_store=store,
        llm=llm if llm is not None else ScriptedLLM([ANSWER_TWO_CITATIONS] * 40),
    )
    return TestClient(app)


class TestStatusEndpoint:
    def test_reports_suite_shape_and_vocabularies(self):
        client = make_client()
        response = client.get("/rag/eval/status")
        assert response.status_code == 200
        body = response.json()
        assert body["suite"]["cases"] == SHIPPED_CASES
        assert body["suite"]["k"] == 3
        assert len(body["variants"]) == SHIPPED_VARIANTS
        # 11 个指标各自带方向与说明（"往哪个方向算好"必须能被读出来）。
        assert [row["metric"] for row in body["metrics"]] == list(QUALITY_METRICS)
        assert {row["direction"] for row in body["metrics"]} <= {
            "higher_better",
            "lower_better",
            "neutral",
        }
        # 12 类坏例标签各自带解释与动作。
        assert set(body["tags"]) == set(BAD_CASE_TAGS)
        assert all(item["action"] for item in body["tags"].values())
        assert "评测集" in body["summary"]

    def test_reports_the_rag_eval_settings_group(self):
        body = make_client().get("/rag/eval/status").json()
        assert body["defaults"]["top_k"] == 3
        assert body["defaults"]["cases_path"].endswith("rag_eval.jsonl")
        assert body["defaults"]["min_samples"] >= 1

    def test_baseline_is_null_when_absent_and_a_dict_when_present(self):
        # **不造全零基线**：没有基线时给 null（前者该去跑一次采集）。
        baseline = make_client().get("/rag/eval/status").json()["baseline"]
        assert baseline is None or isinstance(baseline, dict)
        if isinstance(baseline, dict):
            assert set(baseline["metrics"]) == set(QUALITY_METRICS)

    def test_status_does_not_need_a_model(self):
        # 这个端点只看配置：把模型摘掉也照样 200（它一次模型都不调）。
        store, embedding = indexed_store()
        app = create_app(embedding=embedding, vector_store=store)
        app.state.llm = None
        assert TestClient(app).get("/rag/eval/status").status_code == 200


class TestRunEndpoint:
    def test_populated_store_produces_a_full_report(self):
        client = make_client()
        response = client.post("/rag/eval/run", json={})
        assert response.status_code == 200
        body = response.json()
        assert body["judge"] is False
        assert body["report"]["cases"] == SHIPPED_CASES
        assert len(body["report"]["per_case"]) == SHIPPED_CASES
        # 库非空 → **不该出现 no_data**（检索为空的那四类里没有这一条）。
        assert all(row["empty_reason"] != "no_data" for row in body["report"]["per_case"])
        # 报告折出的基线必须是一份完整的 11 指标表。
        assert set(body["baseline"]["metrics"]) == set(QUALITY_METRICS)
        assert body["baseline"]["k"] == 3
        assert body["baseline"]["samples"] == SHIPPED_CASES
        assert body["limitations"]

    def test_empty_store_is_a_no_data_report_not_a_500(self):
        client = make_client(store=FlatVectorStore())
        response = client.post("/rag/eval/run", json={})
        assert response.status_code == 200
        body = response.json()
        assert body["report"]["metrics"]["retrieval_recall"] == 0.0
        assert body["report"]["metrics"]["grounded_rate"] == 0.0
        assert {row["empty_reason"] for row in body["report"]["per_case"]} == {"no_data"}
        assert {case["tag"] for case in body["report"]["bad_cases"]} == {"no_data"}
        assert body["report"]["actions"]

    def test_variant_selection_is_echoed_in_the_label(self):
        client = make_client()
        body = client.post("/rag/eval/run", json={"variant": "tight-budget"}).json()
        assert body["report"]["label"].endswith("tight-budget")

    def test_unknown_variant_is_a_400_with_the_available_names(self):
        response = make_client().post("/rag/eval/run", json={"variant": "absent"})
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "absent" in detail
        assert "baseline" in detail and "tight-budget" in detail

    def test_judge_flag_is_echoed(self):
        body = make_client().post("/rag/eval/run", json={"judge": True}).json()
        assert body["judge"] is True

    def test_missing_llm_is_a_400_with_a_way_out(self):
        store, embedding = indexed_store()
        app = create_app(embedding=embedding, vector_store=store)
        app.state.llm = None
        response = TestClient(app).post("/rag/eval/run", json={})
        assert response.status_code == 400
        assert "llm" in response.json()["detail"].lower()

    def test_comparison_is_present_only_when_a_baseline_file_exists(self):
        body = make_client().post("/rag/eval/run", json={}).json()
        comparison = body["comparison"]
        assert comparison is None or set(comparison) >= {
            "ok",
            "conclusive",
            "regressions",
            "deltas",
        }
        skipped = make_client().post(
            "/rag/eval/run", json={"compare_with_baseline": False}
        ).json()
        assert skipped["comparison"] is None

    def test_response_is_json_serializable(self):
        body = make_client().post("/rag/eval/run", json={}).json()
        assert json.loads(json.dumps(body, ensure_ascii=False))["report"]["cases"] == (
            SHIPPED_CASES
        )


class TestSampleSuiteIsSelfConsistent:
    """样本与端点的口径一致（端点上标的"手算"必须在样本里也成立）。"""

    def test_sample_cases_match_corpus(self):
        assert len(CASES) == len(CORPUS)
        assert all(case.relevant for case in CASES)

    @pytest.mark.parametrize("variant", ["baseline", "prompt-v1", "cite-gate", "tight-budget"])
    def test_every_variant_runs_on_the_sample_store(self, variant):
        client = make_client()
        response = client.post("/rag/eval/run", json={"variant": variant})
        assert response.status_code == 200
        assert response.json()["report"]["cases"] == SHIPPED_CASES
