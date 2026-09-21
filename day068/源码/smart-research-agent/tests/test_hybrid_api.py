"""day067 ``/retrieval/hybrid`` 与 ``/retrieval/lexical/status`` 两个端点的单元测试.

全部离线、确定性、零网络：样本来自 ``tests/hybrid_samples.py``（十条记录 +
两路各自可控的分数），因此"两路各召回几条、融合之后谁第一"都是**手算得出的数**。

除了"两个端点各自返回什么"，这里钉住五件比 200 更重要的事：

1. **缺省关闭，而且拒绝时说清是哪一个开关**：``retrieval_hybrid_enabled=False``
   时返回 400（**不静默退回单路**）——静默退回会让"我明明启用了却看不出区别"
   变成一个查不出病因的现象；
2. **注入优先于开关**：注入一个 ``HybridRetriever`` 是"我明确要这条链路"的显式动作，
   它不该被一个环境变量否掉（开关管的是**缺省装配**）；
3. **端点不重算任何检索决策**：``channel_candidates`` / ``fusion`` / ``lines``
   逐字段等于 ``HybridRetriever`` 在同一份样本上的结论——
   "接口给出的那一次检索"必须是"脚本给出的那一次检索"；
4. **day066 的五个端点行为不变**：开关关着时 ``/retrieval/search`` 的结果里
   ``channel_candidates`` 与 ``fusion`` 是**空字典**（表示"这次是单路"），
   而不是被混合检索顺带改写成非空；
5. **融合参数走请求体**：``strategy`` / ``alpha`` / ``k_rrf`` 逐次覆盖，
   并回显在 ``result.fusion.params`` 里（"这次用哪组参数"跟着结果走）。
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from smart_research_agent.api.app import create_app
from smart_research_agent.config import settings
from smart_research_agent.retrieval import (
    RETRIEVAL_LIMITATIONS,
    RETRIEVAL_OUT_OF_SCOPE,
    LexicalDocument,
    LexicalIndex,
    build_hybrid_retriever,
)
from smart_research_agent.retrieval.lexical import BM25Params
from tests.hybrid_samples import (
    EXPECTED_VOCABULARY_SIZE,
    HYBRID_DEFAULT,
    HYBRID_SHALLOW,
    QUERY_BOTH,
    QUERY_CODE,
    QUERY_EXACT,
    QUERY_SEMANTIC,
    RECORD_IDS,
    empty_store,
    hybrid_embedding,
    hybrid_lexical,
    hybrid_store,
)
from tests.retrieval_samples import sample_manifest

#: 两个新端点的路径（断言"检索端点从 5 个变 7 个"时用）.
HYBRID_PATH = "/retrieval/hybrid"
LEXICAL_STATUS_PATH = "/retrieval/lexical/status"

#: 四个查询（名次对照的用例用它做参数化）.
ALL_QUERIES: tuple[str, ...] = (QUERY_EXACT, QUERY_CODE, QUERY_SEMANTIC, QUERY_BOTH)

#: 顶层与融合有关的响应字段（"证据原样出门"的清单）.
FUSION_FIELDS: tuple[str, ...] = ("channel_candidates", "fusion", "channels", "lexical")


def make_client(
    *,
    hybrid: Any = None,
    lexical: Any = None,
    store: Any = None,
    enabled: bool = False,
) -> TestClient:
    """建一个可注入的离线客户端（三个注入点各自对应一类用例）.

    ``enabled`` 直接改 ``settings.retrieval_hybrid_enabled``：这个开关的真实用法就是
    "环境变量 / 配置打开它"，因此测试也走同一处（**不传构造参数**——
    ``create_app`` 没有这个参数，它读的正是 settings）。
    """
    settings.retrieval_hybrid_enabled = bool(enabled)
    app = create_app(
        embedding=hybrid_embedding(),
        vector_store=store if store is not None else hybrid_store(),
        llm=None,
    )
    if hybrid is not None:
        setattr(app.state, "retrieval_hybrid", hybrid)
    if lexical is not None:
        setattr(app.state, "retrieval_lexical", lexical)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _restore_flag():
    """每个用例结束后把开关复位（``settings`` 是进程级单例，必须自己收拾）. """
    yield
    settings.retrieval_hybrid_enabled = False


def hybrid_call(client: TestClient, **body: Any) -> dict:
    """发一次混合检索（``query`` 缺省用 ``ERR-2043``）."""
    payload: dict = {"query": QUERY_EXACT}
    payload.update(body)
    response = client.post(HYBRID_PATH, json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def lexical_status(client: TestClient) -> dict:
    """读一次关键词索引状态."""
    response = client.get(LEXICAL_STATUS_PATH)
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture
def client() -> TestClient:
    """注入了混合检索器的客户端（**注入优先**，因此不受开关影响）."""
    store = hybrid_store()
    return make_client(
        hybrid=build_hybrid_retriever(store, hybrid_embedding()),
        store=store,
    )


# --------------------------------------------------------------------------- #
# 端点清单
# --------------------------------------------------------------------------- #


class TestEndpointRegistry:
    """检索端点从 5 个变 7 个（这一条是"今天加了什么"的机器可核对的证据）.

    day068 又加了三个（``POST /retrieval/rerank``、``GET /retrieval/rerank/status``、
    ``POST /retrieval/rerank/compare``），因此现在这个前缀下一共 10 条——
    断言值跟着改，而**计数方式**（前缀过滤 + 路径在集合里）一个字没动。
    """

    def test_ten_retrieval_endpoints_are_registered(self) -> None:
        client = make_client(enabled=True)
        paths = [
            path
            for path in client.get("/openapi.json").json()["paths"]
            if path.startswith("/retrieval")
        ]

        assert len(paths) == 10
        assert HYBRID_PATH in paths
        assert LEXICAL_STATUS_PATH in paths

    def test_the_two_new_paths_keep_their_methods(self) -> None:
        schema = make_client(enabled=True).get("/openapi.json").json()["paths"]

        assert set(schema[HYBRID_PATH]) == {"post"}
        assert set(schema[LEXICAL_STATUS_PATH]) == {"get"}


# --------------------------------------------------------------------------- #
# 开关：缺省关闭，而且拒绝时说清是哪一个开关
# --------------------------------------------------------------------------- #


class TestHybridDisabledByDefault:
    """``retrieval_hybrid_enabled=False``（缺省）时**拒绝**请求，不静默退回单路. """

    def test_default_setting_is_off(self) -> None:
        assert settings.retrieval_hybrid_enabled is False

    def test_request_is_rejected_with_400(self) -> None:
        client = make_client()

        response = client.post(HYBRID_PATH, json={"query": QUERY_EXACT})

        assert response.status_code == 400

    def test_the_message_names_the_switch(self) -> None:
        client = make_client()

        detail = client.post(HYBRID_PATH, json={"query": QUERY_EXACT}).json()["detail"]

        assert "retrieval_hybrid_enabled" in detail
        assert "RETRIEVAL_HYBRID_ENABLED" in detail
        assert "app.state.retrieval_hybrid" in detail

    def test_the_message_explains_why_it_is_off_by_default(self) -> None:
        """缺省关闭的理由必须写在消息里（否则它看起来只是一个"还没做"的功能）. """
        client = make_client()

        detail = client.post(HYBRID_PATH, json={"query": QUERY_EXACT}).json()["detail"]

        assert "改变结果集合" in detail
        assert "**不会**静默退回单路" in detail

    def test_the_switch_turns_it_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "retrieval_hybrid_enabled", True)
        client = make_client(enabled=True)

        payload = hybrid_call(client, top_k=3)

        assert payload["result"]["channel_candidates"]["bm25"] == 1

    def test_lexical_status_is_available_even_when_hybrid_is_off(self) -> None:
        """状态读端点不受开关影响：诊断要先能看到现状（它不改变任何结果集合）. """
        payload = lexical_status(make_client())

        assert payload["count"] == len(RECORD_IDS)


# --------------------------------------------------------------------------- #
# POST /retrieval/hybrid
# --------------------------------------------------------------------------- #


class TestHybridEndpoint:
    """一次混合检索的完整交代（名次用手算常量，证据逐字段核对）. """

    @pytest.mark.parametrize("query", ALL_QUERIES)
    def test_default_depth_matches_the_hand_computed_order(
        self, client: TestClient, query: str
    ) -> None:
        payload = hybrid_call(client, query=query)

        assert [hit["record_id"] for hit in payload["result"]["hits"]] == list(
            HYBRID_DEFAULT[query]
        )

    @pytest.mark.parametrize("query", ALL_QUERIES)
    def test_shallow_depth_matches_the_hand_computed_order(
        self, client: TestClient, query: str
    ) -> None:
        payload = hybrid_call(client, query=query, top_k=3, fetch_k=3)

        assert [hit["record_id"] for hit in payload["result"]["hits"]] == list(
            HYBRID_SHALLOW[query]
        )

    def test_the_exact_match_record_is_first(self, client: TestClient) -> None:
        """主案例：向量路对它没有方向，关键词路把它顶到第一. """
        payload = hybrid_call(client)
        first = payload["result"]["hits"][0]

        assert first["record_id"] == "h-t-01"
        assert first["channel"] == "bm25"
        assert first["channels"] == ["bm25", "vector"]

    def test_channel_candidates_are_reported_per_channel(self, client: TestClient) -> None:
        payload = hybrid_call(client)

        assert payload["channel_candidates"] == {
            "vector": len(RECORD_IDS),
            "bm25": 1,
        }
        assert payload["result"]["channel_candidates"] == payload["channel_candidates"]

    def test_fusion_summary_is_complete(self, client: TestClient) -> None:
        payload = hybrid_call(client, query=QUERY_BOTH)
        fusion = payload["fusion"]

        assert fusion["strategy"] == "rrf"
        assert fusion["params"] == {"k_rrf": 60}
        assert fusion["recall"] == {"bm25": 1, "vector": len(RECORD_IDS)}
        assert fusion["deduped"] == 1
        assert fusion["fused"] == len(RECORD_IDS)

    def test_channels_lists_the_channels_that_appear(self, client: TestClient) -> None:
        """``channels`` 按**首次出现**的顺序：``ERR-2043`` 的第一条来自关键词路. """
        assert hybrid_call(client)["channels"] == ["bm25", "vector"]

    def test_channels_order_follows_the_hits(self, client: TestClient) -> None:
        """换一个查询（第一条来自向量路）→ 顺序跟着换（它不是一个常量列表）. """
        assert hybrid_call(client, query=QUERY_SEMANTIC)["channels"] == ["vector"]

    def test_the_lexical_block_describes_the_keyword_index(self, client: TestClient) -> None:
        lexical = hybrid_call(client)["lexical"]

        assert lexical["count"] == len(RECORD_IDS)
        assert lexical["vocabulary_size"] == EXPECTED_VOCABULARY_SIZE
        assert lexical["k1"] == 1.5
        assert lexical["b"] == 0.75

    def test_lines_include_the_hybrid_specific_diagnosis(self, client: TestClient) -> None:
        joined = "\n".join(hybrid_call(client)["lines"])

        assert "两路明细" in joined
        assert "融合：" in joined
        assert "阈值口径" in joined
        assert "关键词索引" in joined

    def test_lines_answer_the_four_original_questions(self, client: TestClient) -> None:
        joined = "\n".join(hybrid_call(client)["lines"])

        assert "索引版本" in joined
        assert "过滤条件" in joined
        assert "条数" in joined
        assert "空结果原因" in joined

    def test_result_is_included_with_text(self, client: TestClient) -> None:
        """这个端点带着正文出门（"看一眼检索结果"是它的用途，与 search 同取向）. """
        payload = hybrid_call(client)

        assert payload["result"]["hits"][0]["text"]
        assert "ERR-2043" in payload["result"]["hits"][0]["text"]

    def test_the_keyword_channel_emptiness_is_named(self, client: TestClient) -> None:
        payload = hybrid_call(client, query=QUERY_SEMANTIC)
        notes = " ".join(payload["result"]["notes"])

        assert payload["result"]["channel_candidates"]["bm25"] == 0
        assert "关键词路一条都没召回" in notes
        assert "交集为空" in notes

    def test_the_threshold_scope_is_named(self, client: TestClient) -> None:
        payload = hybrid_call(client, min_score=1.5)
        notes = " ".join(payload["result"]["notes"])

        assert payload["result"]["channel_candidates"]["vector"] == 0
        assert "只作用于向量通道" in notes

    def test_filtered_out_is_reported_with_one_reason(self, client: TestClient) -> None:
        payload = hybrid_call(client, where={"strategy": "nope"})

        assert payload["empty_reason"] == "filtered_out"
        assert payload["empty_reason_description"]
        assert payload["result"]["hits"] == []

    def test_conditions_and_filter_fields_are_echoed(self, client: TestClient) -> None:
        payload = hybrid_call(client, where={"parent_doc_id": "doc-trouble"})

        assert "parent_doc_id" in payload["conditions"]
        assert payload["filter_fields"] == ["parent_doc_id"]

    def test_limitations_and_out_of_scope_are_echoed(self, client: TestClient) -> None:
        payload = hybrid_call(client)

        assert payload["limitations"] == list(RETRIEVAL_LIMITATIONS)
        assert payload["out_of_scope"] == list(RETRIEVAL_OUT_OF_SCOPE)

    def test_every_response_field_is_json_serializable(self, client: TestClient) -> None:
        """响应体是一个**能原样往返**的 JSON 对象（端点不把 dataclass 直接塞进去）. """
        for query in ALL_QUERIES:
            payload = hybrid_call(client, query=query)
            rendered = json.dumps(payload, ensure_ascii=False)

            assert json.loads(rendered) == payload
            assert "h-t-01" in rendered or "h-t-03" in rendered or "h-t-02" in rendered

    def test_the_manifest_version_flows_into_the_result(self) -> None:
        """清单被采纳时，版本号跟着结果走（与 day066 的 /retrieval/search 同源）.

        这里**不注入**混合检索器：走缺省装配那条路，验证
        ``retrieval_hybrid`` 把 ``retrieval_manifest`` 读到的清单传给了它。
        """
        store = hybrid_store()
        client = make_client(store=store, enabled=True)
        manifest = sample_manifest(store)
        versions = client.app.state.indexing_versions  # type: ignore[attr-defined]
        versions.register(manifest)
        versions.adopt(manifest.version_id)

        payload = hybrid_call(client)

        assert payload["result"]["index"]["version_id"] == manifest.version_id
        assert payload["result"]["index"]["has_drift"] is False

    def test_drift_is_observable_through_the_endpoint(self) -> None:
        """库被人删过一条 → 结果里带漂移，而且**照样返回结果**（观测而不是阻断）. """
        store = hybrid_store()
        client = make_client(store=store, enabled=True)
        manifest = sample_manifest(store)
        versions = client.app.state.indexing_versions  # type: ignore[attr-defined]
        versions.register(manifest)
        versions.adopt(manifest.version_id)
        store.delete(["h-c-04"])

        payload = hybrid_call(client)

        assert payload["result"]["index"]["has_drift"] is True
        assert payload["result"]["hits"]


class TestHybridRequestParameters:
    """请求体的九个字段各自的作用（八个检索字段 + 三个融合字段，其中三个可省略）. """

    def test_top_k_and_fetch_k_are_honoured(self, client: TestClient) -> None:
        payload = hybrid_call(client, top_k=2, fetch_k=4)

        assert payload["result"]["count"] == 2
        assert payload["result"]["fetch_k"] == 4

    def test_where_is_honoured(self, client: TestClient) -> None:
        payload = hybrid_call(client, where={"parent_doc_id": "doc-concept"})

        assert payload["result"]["channel_candidates"]["bm25"] == 0
        assert all(
            hit["metadata"]["parent_doc_id"] == "doc-concept"
            for hit in payload["result"]["hits"]
        )

    def test_min_score_is_honoured(self, client: TestClient) -> None:
        payload = hybrid_call(client, min_score=1.0)

        assert payload["result"]["dropped_below_threshold"] == len(RECORD_IDS) - 1

    def test_max_per_doc_is_honoured(self, client: TestClient) -> None:
        payload = hybrid_call(client, max_per_doc=1)

        assert payload["result"]["count"] == 3
        assert payload["result"]["dropped_by_diversity"] == len(RECORD_IDS) - 3

    def test_time_range_is_honoured(self, client: TestClient) -> None:
        payload = hybrid_call(
            client,
            time_range={"field": "created_at", "start": "2026-09-15", "end": "2026-09-30"},
        )

        assert "h-t-03" not in [hit["record_id"] for hit in payload["result"]["hits"]]

    def test_strategy_override_is_echoed(self, client: TestClient) -> None:
        payload = hybrid_call(client, strategy="weighted")

        assert payload["fusion"]["strategy"] == "weighted"
        assert payload["fusion"]["params"] == {"alpha": 0.5}

    def test_alpha_override_is_echoed(self, client: TestClient) -> None:
        payload = hybrid_call(client, strategy="weighted", alpha=0.25)

        assert payload["fusion"]["params"]["alpha"] == 0.25

    def test_k_rrf_override_is_echoed(self, client: TestClient) -> None:
        payload = hybrid_call(client, k_rrf=5)

        assert payload["fusion"]["params"]["k_rrf"] == 5

    def test_alpha_override_changes_the_order(self, client: TestClient) -> None:
        """``alpha=0.1`` 时关键词路的话语权很大：``h-t-01`` 依然第一；
        ``alpha=0.9`` 时向量路压过来，``h-c-01`` 翻到第一（两条的分数就是 alpha 与 1-alpha）. """
        low = hybrid_call(client, top_k=3, fetch_k=3, strategy="weighted", alpha=0.1)
        high = hybrid_call(client, top_k=3, fetch_k=3, strategy="weighted", alpha=0.9)

        assert low["result"]["hits"][0]["record_id"] == "h-t-01"
        assert high["result"]["hits"][0]["record_id"] == "h-c-01"

    def test_unknown_json_fields_are_ignored(self, client: TestClient) -> None:
        """HTTP 层只有三个具名字段：多给的字段被 pydantic 忽略（与其余端点一致）.

        这与库内 ``RetrievalQuery.extra`` 的"写错键名报错"不是同一条通道
        （见 ``HybridRetrievalRequest`` 的 docstring 末段）。这条用例把**真实行为**
        钉住，免得"我传了 weight 却没生效"变成一次没有答案的排查。
        """
        payload = hybrid_call(client, weight=0.9, strategy="weighted")

        assert "weight" not in payload["fusion"]["params"]
        assert payload["fusion"]["params"] == {"alpha": 0.5}


class TestHybridRequestValidation:
    """参数非法一律 400，且消息里带合法取值（与 day066 的五条端点同一条通道）. """

    @pytest.mark.parametrize("strategy", ["nope", "RRF", ""])
    def test_unknown_strategy_is_rejected(self, client: TestClient, strategy: str) -> None:
        response = client.post(HYBRID_PATH, json={"query": QUERY_EXACT, "strategy": strategy})

        assert response.status_code == 400
        assert "rrf" in response.json()["detail"]

    @pytest.mark.parametrize("alpha", [1.5, -0.1, 2])
    def test_alpha_out_of_range_is_rejected(self, client: TestClient, alpha: float) -> None:
        response = client.post(HYBRID_PATH, json={"query": QUERY_EXACT, "alpha": alpha})

        assert response.status_code == 400
        assert "alpha" in response.json()["detail"]

    @pytest.mark.parametrize("k_rrf", [0, -1])
    def test_bad_k_rrf_is_rejected(self, client: TestClient, k_rrf: int) -> None:
        response = client.post(HYBRID_PATH, json={"query": QUERY_EXACT, "k_rrf": k_rrf})

        assert response.status_code == 400
        assert "k_rrf" in response.json()["detail"]

    @pytest.mark.parametrize("query", ["", "   "])
    def test_blank_query_is_rejected(self, client: TestClient, query: str) -> None:
        response = client.post(HYBRID_PATH, json={"query": query})

        assert response.status_code == 400
        assert "空查询" in response.json()["detail"]

    def test_bad_where_syntax_is_rejected(self, client: TestClient) -> None:
        response = client.post(
            HYBRID_PATH, json={"query": QUERY_EXACT, "where": {"topic": {"$bad": 1}}}
        )

        assert response.status_code == 400
        assert "where" in response.json()["detail"]

    def test_top_k_beyond_the_ceiling_is_rejected(self, client: TestClient) -> None:
        response = client.post(HYBRID_PATH, json={"query": QUERY_EXACT, "top_k": 5000})

        assert response.status_code == 400

    def test_a_bad_body_shape_is_422(self, client: TestClient) -> None:
        """字段类型不对是**契约**问题（422），不是检索参数问题（400）——与其余端点一致. """
        response = client.post(HYBRID_PATH, json={"query": QUERY_EXACT, "where": ["topic"]})

        assert response.status_code == 422

    def test_hybrid_and_search_reject_the_same_parameters(self, client: TestClient) -> None:
        """同一个坏参数在两条通道上得到同一个状态码（它们共用同一套校验）. """
        body = {"query": QUERY_EXACT, "top_k": 0}

        assert client.post(HYBRID_PATH, json=body).status_code == 400
        assert client.post("/retrieval/search", json=body).status_code == 400


# --------------------------------------------------------------------------- #
# GET /retrieval/lexical/status
# --------------------------------------------------------------------------- #


class TestLexicalStatusEndpoint:
    """关键词索引的现状：四个数字各自回答一个诊断问题. """

    def test_reports_the_index_shape(self) -> None:
        payload = lexical_status(make_client())

        assert payload["count"] == len(RECORD_IDS)
        assert payload["vocabulary_size"] == EXPECTED_VOCABULARY_SIZE
        assert payload["avgdl"] == pytest.approx(34.3, rel=1e-9)
        assert payload["k1"] == 1.5
        assert payload["b"] == 0.75

    def test_index_block_matches_the_top_level_fields(self) -> None:
        payload = lexical_status(make_client())

        assert payload["index"]["count"] == payload["count"]
        assert payload["index"]["vocabulary_size"] == payload["vocabulary_size"]
        assert payload["index"]["avgdl"] == payload["avgdl"]
        assert payload["index"]["params"] == {"k1": payload["k1"], "b": payload["b"]}
        assert payload["index"]["channel"] == "bm25"

    def test_summary_line_carries_the_key_numbers(self) -> None:
        payload = lexical_status(make_client())

        assert f"{len(RECORD_IDS)} 篇" in payload["summary"]
        assert str(EXPECTED_VOCABULARY_SIZE) in payload["summary"]
        assert "BM25(k1=1.5, b=0.75)" in payload["summary"]

    def test_defaults_come_from_the_settings(self) -> None:
        payload = lexical_status(make_client())

        assert payload["defaults"] == {
            "bm25_k1": settings.retrieval_bm25_k1,
            "bm25_b": settings.retrieval_bm25_b,
            "hybrid_enabled": settings.retrieval_hybrid_enabled,
            "hybrid_strategy": settings.retrieval_hybrid_strategy,
            "hybrid_alpha": settings.retrieval_hybrid_alpha,
            "hybrid_rrf_k": settings.retrieval_hybrid_rrf_k,
        }

    def test_an_injected_index_with_custom_params_is_reflected(self) -> None:
        index = hybrid_lexical(params=BM25Params(k1=1.1, b=0.4), name="手册关键词")
        payload = lexical_status(make_client(lexical=index))

        assert payload["k1"] == 1.1
        assert payload["b"] == 0.4
        assert payload["index"]["name"] == "手册关键词"

    def test_an_empty_store_reports_a_zero_index(self) -> None:
        """空库：200 且四个数字都是 0（不是 500——刚建好还没写数据是正常处境）. """
        payload = lexical_status(make_client(store=empty_store()))

        assert payload["count"] == 0
        assert payload["vocabulary_size"] == 0
        assert payload["avgdl"] == 0.0
        assert "0 篇" in payload["summary"]

    def test_limitations_are_echoed(self) -> None:
        payload = lexical_status(make_client())

        assert payload["limitations"] == list(RETRIEVAL_LIMITATIONS)
        assert payload["out_of_scope"] == list(RETRIEVAL_OUT_OF_SCOPE)
        assert any("2-gram" in item for item in payload["limitations"])

    def test_the_endpoint_is_read_only(self) -> None:
        """两次读给出同一份答案（它不触发检索、不写任何状态）. """
        client = make_client()

        assert lexical_status(client) == lexical_status(client)

    def test_body_is_json_serializable(self) -> None:
        assert json.dumps(lexical_status(make_client()))


# --------------------------------------------------------------------------- #
# 注入与装配错误
# --------------------------------------------------------------------------- #


class TestInjection:
    """三个注入点（hybrid / lexical / vector_store+embedding）各自的优先级与报错. """

    def test_an_injected_hybrid_is_used_even_when_the_switch_is_off(self) -> None:
        """注入是显式动作，它不该被环境变量否掉（开关管的是缺省装配）. """
        client = make_client(hybrid=build_hybrid_retriever(hybrid_store(), hybrid_embedding()))

        payload = hybrid_call(client, top_k=3)

        assert payload["result"]["count"] == 3

    def test_an_injected_lexical_index_is_reused(self) -> None:
        index = LexicalIndex.build([LexicalDocument(record_id="only", text="独一份的正文")])
        client = make_client(hybrid=None, lexical=index, enabled=True)

        payload = hybrid_call(client)

        assert payload["lexical"]["count"] == 1
        assert payload["channel_candidates"]["bm25"] == 0

    def test_a_wrong_injected_hybrid_is_a_500(self) -> None:
        """装配写错 → 500（不是 400）：把它降级成 400 会让一个必然复现的配置错误
        看起来像"这次请求碰巧不对"。"""
        client = make_client(hybrid="not-a-retriever")

        response = client.post(HYBRID_PATH, json={"query": QUERY_EXACT})

        assert response.status_code == 500
        assert "HybridRetriever" in response.json()["detail"]

    def test_a_wrong_injected_lexical_index_is_a_500(self) -> None:
        client = make_client(lexical="not-an-index")

        response = client.get(LEXICAL_STATUS_PATH)

        assert response.status_code == 500
        assert "LexicalIndex" in response.json()["detail"]

    def test_the_lexical_index_is_built_from_the_injected_store(self) -> None:
        """缺省装配（开关打开）时，关键词索引来自 ``app.state.vector_store``. """
        client = make_client(enabled=True)

        payload = hybrid_call(client, top_k=3)

        assert payload["lexical"]["count"] == len(RECORD_IDS)
        assert payload["channel_candidates"]["bm25"] == 1

    def test_state_is_per_app_instance(self) -> None:
        """状态挂在**应用实例**上：另建一个 app 看到的是另一个库. """
        first = make_client(enabled=True)
        second = make_client(enabled=True)

        assert hybrid_call(first, top_k=3)["lexical"]["count"] == len(RECORD_IDS)
        assert lexical_status(second)["count"] == len(RECORD_IDS)
        assert first is not second


# --------------------------------------------------------------------------- #
# day066 的五个端点行为不变
# --------------------------------------------------------------------------- #


class TestExistingEndpointsUnchanged:
    """开关关着时，day066 的端点逐字段保持原样（"新能力默认不生效"的证据）. """

    def test_search_still_works_and_stays_single_path(self) -> None:
        client = make_client()

        payload = client.post("/retrieval/search", json={"query": QUERY_EXACT, "top_k": 3})

        assert payload.status_code == 200
        result = payload.json()["result"]
        assert result["channels"] == ["vector"]
        assert result["channel_candidates"] == {}
        assert result["fusion"] == {}

    def test_hybrid_evidence_fields_exist_but_are_empty_for_single_path(self) -> None:
        """两个新字段**始终存在**（响应体形状稳定），单路时是空字典. """
        client = make_client()

        result = client.post("/retrieval/search", json={"query": QUERY_EXACT}).json()["result"]

        assert "channel_candidates" in result
        assert "fusion" in result

    def test_status_reports_the_new_defaults(self) -> None:
        client = make_client()

        defaults = client.get("/retrieval/status").json()["defaults"]

        assert defaults["hybrid_enabled"] is False
        assert defaults["hybrid_strategy"] == "rrf"
        assert defaults["hybrid_rrf_k"] == 60
        assert defaults["bm25_k1"] == 1.5
        assert defaults["bm25_b"] == 0.75
        assert defaults["max_fetch_k"]

    def test_routes_and_explain_are_untouched(self) -> None:
        client = make_client()

        assert client.get("/retrieval/routes").status_code == 200
        explain = client.post("/retrieval/explain", json={"query": QUERY_EXACT})
        assert explain.status_code == 200
        assert "lines" in explain.json()
