"""day066 ``/retrieval/*`` 五个端点的单元测试.

全部离线、确定性、零网络：样本来自 ``tests/retrieval_samples.py``（八条记录 +
``TableEmbedding`` 查表编码器 + ``FlatVectorStore`` 参照后端），因此
"哪条最近""切掉了几条""候选剩几条"都是**手算得出来的数**，不是"跑一遍看看"。
``axis`` 查询在样本上的分数是 ``1.0 / 1.0 / 0.8 / 0.6 / 0.5 / 0 / 0 / 0``
（手算依据见那份样本的模块 docstring），下面每一条期望值都由它推出来。

除了"五个端点各自返回什么"，这里钉住四件比 200 更重要的事：

1. **空库是合法状态**：``/retrieval/search`` 与 ``/retrieval/explain`` 对空库
   返回 200 且 ``empty_reason="no_data"``（``fetch_k`` 照样算过），
   而 ``/retrieval/answer`` **一次 LLM 都不调**（``llm_called=False``）——
   这条要用一个"一旦被调用就抛异常"的假模型来证明，而不是看答案像不像兜底话术；
2. **参数问题一律 400，且消息里带合法取值**：top_k 的上下界 / ``fetch_k < top_k`` /
   ``max_per_doc=0`` / 时间范围倒置与格式 / ``where`` 语法 / 未知路由名 /
   多个索引却没有默认路由——每一条都断言"消息里有没有那句能照做的话"；
3. **未配置 LLM 时 ``/retrieval/answer`` 返回 400，且不拿 MockLLM 兜底**：
   缺省 app（``llm=MockLLM()``）得到 400，而注入一个自定义 ``BaseLLM`` 子类时
   正常作答——两条都要断言，因为"护栏生效"与"护栏不误伤"是两件不同的事；
4. **端点不重算检索决策**：三个 ``dropped_*`` 与 ``empty_reason`` 逐字段等于
   ``Retriever`` 在同一份样本上的结论，``explain`` 的 ``lines`` 与 ``search``
   的同一次请求逐行相同——"诊断描述的那次检索"必须是"返回结果的那次检索"。

另外两条跨端点的性质也在这里：状态挂在**应用实例**上（另建一个 app 看到的是
另一个库），以及所有响应体都能被 ``json.dumps``（端点不把 dataclass 直接塞进响应）。
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from smart_research_agent.api.app import create_app
from smart_research_agent.config import settings
from smart_research_agent.llm.base import BaseLLM, Message
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.retrieval import (
    EMPTY_REASONS,
    FALLBACK_NO_CONTEXT,
    MAX_FETCH_K,
    RAG_ANSWER_PROMPT_VERSION,
    RETRIEVAL_LIMITATIONS,
    RETRIEVAL_OUT_OF_SCOPE,
    StoreRouter,
    build_retriever,
)
from smart_research_agent.vectorstore.flat import FlatVectorStore
from tests.retrieval_samples import (
    EXPECTED_AXIS_ORDER,
    EXPECTED_AXIS_SCORES,
    MISSING_CREATED_AT_ID,
    QUERY_TEXTS,
    RECORD_IDS,
    RECORD_TEXTS,
    SOURCE_ALPHA,
    TIE_IDS,
    VECTOR_DIMENSION,
    TableEmbedding,
    empty_store,
    sample_manifest,
    sample_store,
)

#: 三个查询（它们是 ``TableEmbedding`` 查表的键，本身没有语义）.
AXIS = QUERY_TEXTS["axis"]
TILT = QUERY_TEXTS["tilt"]
ORTHO = QUERY_TEXTS["ortho"]

#: 三个端点共用的空结果原因取值（``EMPTY_REASONS`` 里除 "hits" 之外的三个）.
NO_DATA = "no_data"
FILTERED_OUT = "filtered_out"
BELOW_THRESHOLD = "below_threshold"

#: ``where={"strategy": "structural"}`` 之后 ``axis`` 查询的期望命中（手算见样本 docstring）.
STRUCTURAL_IDS: tuple[str, ...] = ("c-a-01", "c-b-01", "c-a-02", "c-c-02")

#: 时间范围 ``[2026-09-02, 2026-09-30]`` 内的期望命中。``c-c-01`` **不在**里面：
#: 它是唯一没有 ``created_at`` 的记录，被排除是显式约定（不是 bug）。
SEPTEMBER_IDS: tuple[str, ...] = ("c-b-01", "c-a-02", "c-a-03", "c-b-02", "c-c-03")

#: 只给下界 ``>= 2026-01-01`` 时的期望候选数：八条里去掉那条缺字段的。
SINGLE_SIDED_CANDIDATES = len(RECORD_IDS) - 1

#: ``max_per_doc=1`` 之后 ``axis`` 查询留下的三条（每篇文档一条）.
DIVERSITY_IDS: tuple[str, ...] = ("c-a-01", "c-b-01", "c-c-02")

#: 未加过滤时库侧的候选数（= 库里的条数）。
ALL_CANDIDATES = len(RECORD_IDS)


class RecordingLLM(BaseLLM):
    """离线假模型：记下收到的提示词，返回一段带 ``[1]`` ``[2]`` 的答案.

    为什么不直接用 ``MockLLM``：``/retrieval/answer`` **显式拒绝** MockLLM
    （它是 ``create_app`` 在未配置密钥时的占位实现），因此要测命中路径就必须
    拿一个非 Mock 的对象。它在这个文件里承担两件事：证明护栏**不误伤**，
    以及留下"模型这次到底看到了什么"的证据（``calls`` / ``prompt_text``）。
    """

    #: 默认答案。带 ``[1]`` ``[2]`` 是刻意的：提示词里的编号能不能对上，
    #: 要靠它来验（答案里的编号与 ``citations`` 是同一份编号）。
    REPLY = "依据 [1]，阈值用分位数标定。\n依据 [2]，换编码器必须重新标定。"

    def __init__(self, reply: str = REPLY) -> None:
        self._reply = reply
        self.calls: list[list[Message]] = []

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """记下整轮消息并返回预设回复（不打网络、不读盘、不改任何状态）."""
        self.calls.append(list(messages))
        return self._reply

    @property
    def prompt_text(self) -> str:
        """最近一次调用收到的提示词（没有调用过时是空串）."""
        if not self.calls:
            return ""
        return "\n".join(message.content for message in self.calls[-1])


class ExplodingLLM(BaseLLM):
    """一旦被调用就抛异常：用来证明"检索为空时一次 LLM 都不调".

    用抛异常而不是"数调用次数"来当判据，是因为它**拦得住**：
    护栏失效时用例会以 ``AssertionError`` 失败，而不是悄悄地多调一次模型。
    ``calls`` 仍然记着，方便失败时看清是哪一步走到了这里。
    """

    def __init__(self) -> None:
        self.calls = 0

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """任何一次调用都是 bug：本模型只应该待在"不该被调用"的用例里."""
        self.calls += 1
        raise AssertionError("检索为空时不该调用 LLM：护栏没有生效")


def make_client(
    *,
    store: FlatVectorStore | None = None,
    embedding: TableEmbedding | None = None,
    llm: BaseLLM | None = None,
    router: StoreRouter | None = None,
) -> TestClient:
    """建一个可注入的离线客户端（四个注入点各自对应一类用例）.

    ``llm`` 缺省时 **不传**（交给 ``create_app`` 的默认装配）：
    ``/retrieval/answer`` 的"未配置 LLM → 400"那条路径要的正是缺省装配的结果，
    而其余端点根本不看模型。
    """
    app = create_app(
        embedding=embedding if embedding is not None else TableEmbedding(),
        vector_store=store if store is not None else sample_store(),
        llm=llm,
    )
    if router is not None:
        # 路由表**注入在 app.state 上**（键名见 routes.RETRIEVAL_ROUTER_STATE_KEY）：
        # 多索引是业务决策，端点不会替调用方猜（见 routes.retrieval_router）。
        setattr(app.state, "retrieval_router", router)
    return TestClient(app)


def two_route_router(*, default: str | None = None) -> StoreRouter:
    """两条路由（handbook / source）指向**同一份样本**：选路用例的素材.

    两个检索器共用一份库是刻意的：这里要验的是"选路有没有发生"，
    而库的内容已经在搜索用例里验过——让第二个库不一样，只会让每条断言
    同时依赖两件互不相干的事。
    """
    store = sample_store()
    embedding = TableEmbedding()
    router = StoreRouter(default=default)
    router.register(
        "handbook",
        build_retriever(store, embedding, name="handbook"),
        description="手册库：产品手册与配置说明",
    )
    router.register(
        "source",
        build_retriever(store, embedding, name="source"),
        description="源码库：函数签名与实现片段",
    )
    return router


# --------------------------------------------------------------------------- #
# 调用助手：每个端点一个，"这一步必须成功"由助手钉住
# --------------------------------------------------------------------------- #


def status(client: TestClient, **params: object) -> dict:
    """读一次检索与索引状态（``GET /retrieval/status``）."""
    response = client.get("/retrieval/status", params=params)
    assert response.status_code == 200, response.text
    return response.json()


def search(client: TestClient, **body: object) -> dict:
    """发一次检索请求（``query`` 缺省用 ``axis`` 那句话）."""
    payload: dict = {"query": AXIS}
    payload.update(body)
    response = client.post("/retrieval/search", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def explain(client: TestClient, **body: object) -> dict:
    """发一次诊断请求（与 ``search`` 同一份请求体）."""
    payload: dict = {"query": AXIS}
    payload.update(body)
    response = client.post("/retrieval/explain", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def answer(client: TestClient, **body: object) -> dict:
    """发一次问答请求（``question`` 缺省用 ``axis`` 那句话）."""
    payload: dict = {"question": AXIS}
    payload.update(body)
    response = client.post("/retrieval/answer", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def routes(client: TestClient) -> dict:
    """读一次路由清单（``GET /retrieval/routes``）."""
    response = client.get("/retrieval/routes")
    assert response.status_code == 200, response.text
    return response.json()


def result_of(client: TestClient, **body: object) -> dict:
    """只取 ``/retrieval/search`` 的 ``result``（断言里最常读的那一块）."""
    return search(client, **body)["result"]


@pytest.fixture
def client() -> TestClient:
    """缺省客户端：八条样本 + 查表编码器 + 缺省 ``app.state`` 装配（单路由）."""
    return make_client()


# --------------------------------------------------------------------------- #
# GET /retrieval/status
# --------------------------------------------------------------------------- #


def test_status_reports_the_retriever_config_and_the_index_state(client: TestClient) -> None:
    """正常路径：检索器的九项配置 + 索引现状 + 缺省值表 + 路由概览一起给."""
    payload = status(client)
    retriever = payload["retriever"]
    assert retriever["name"] == "default"
    assert retriever["top_k"] == settings.retrieval_top_k
    assert retriever["fetch_multiplier"] == settings.retrieval_fetch_multiplier
    assert retriever["max_per_doc"] == settings.retrieval_max_per_doc
    assert retriever["doc_id_field"] == "parent_doc_id"
    assert retriever["time_field"] == settings.retrieval_time_field
    # index 是 retriever 里那一块的便捷投影：两者必须是同一份，而不是各算一次
    assert payload["index"] == retriever["index"]
    index = payload["index"]
    assert index["count_store"] == ALL_CANDIDATES
    assert index["count_manifest"] is None
    assert index["version_id"] == ""
    assert index["has_drift"] is False
    assert index["metric"] == "cosine"
    assert index["dimension"] == VECTOR_DIMENSION
    assert "索引版本" in payload["summary"]
    # 缺省值表：能力与当前取值必须分开（"这个 5 是项目默认还是谁改过"）
    assert payload["defaults"]["top_k"] == settings.retrieval_top_k
    assert payload["defaults"]["max_fetch_k"] == MAX_FETCH_K
    assert payload["defaults"]["time_field"] == settings.retrieval_time_field
    assert set(payload["empty_reasons"]) == set(EMPTY_REASONS)
    assert payload["limitations"] == list(RETRIEVAL_LIMITATIONS)
    assert payload["out_of_scope"] == list(RETRIEVAL_OUT_OF_SCOPE)


def test_status_reports_the_adopted_manifest_version(client: TestClient) -> None:
    """清单被采纳时：版本号与库/清单两个条数一起报出来，且无漂移."""
    store = sample_store()
    apps = make_client(store=store, embedding=TableEmbedding())
    manifest = sample_manifest(store)
    versions = apps.app.state.indexing_versions  # type: ignore[attr-defined]
    versions.register(manifest)
    versions.adopt(manifest.version_id)

    index = status(apps)["index"]
    assert index["version_id"] == manifest.version_id
    assert index["count_manifest"] == ALL_CANDIDATES
    assert index["count_store"] == ALL_CANDIDATES
    assert index["has_drift"] is False
    assert index["drift"] == []


def test_status_reports_drift_after_a_record_disappears() -> None:
    """库里被人删过一条 → ``drift`` 非空、``has_drift`` 为真、摘要里写着漂移几项.

    这是本课兑现的第二个伏笔：清单与库互相矛盾时**观测而不是阻断**
    （检索照跑、结果可能少召回），但那件事必须被读出来。
    """
    store = sample_store()
    client = make_client(store=store, embedding=TableEmbedding())
    manifest = sample_manifest(store)
    versions = client.app.state.indexing_versions  # type: ignore[attr-defined]
    versions.register(manifest)
    versions.adopt(manifest.version_id)

    store.delete(["c-a-02"])
    payload = status(client)
    assert payload["index"]["count_store"] == ALL_CANDIDATES - 1
    assert payload["index"]["count_manifest"] == ALL_CANDIDATES
    assert payload["index"]["has_drift"] is True
    assert "c-a-02" in payload["index"]["drift"][0]
    assert "漂移 1 项" in payload["summary"]


def test_status_on_an_empty_store_is_a_legal_state() -> None:
    """空库：200 且报告"库 0 条"，不是 500（刚建好还没写数据是正常处境）."""
    payload = status(make_client(store=empty_store()))
    assert payload["index"]["count_store"] == 0
    assert payload["index"]["count_manifest"] is None
    assert payload["index"]["has_drift"] is False
    assert "库 0 条" in payload["summary"]


def test_status_returns_the_route_the_caller_asked_for() -> None:
    """``?route=source`` 显式指定：回显的 name 是它，且 ``matched=True``."""
    payload = status(make_client(router=two_route_router()), route="source")
    assert payload["retriever"]["name"] == "source"
    assert payload["route"]["name"] == "source"
    assert payload["route"]["matched"] is True
    assert payload["router"] == {
        "count": 2,
        "default": "",
        "names": ["handbook", "source"],
    }


def test_status_falls_back_to_the_router_default() -> None:
    """不传 route 且设了默认：用默认那一版，并说明"是路由器选定的"."""
    payload = status(make_client(router=two_route_router(default="handbook")))
    assert payload["retriever"]["name"] == "handbook"
    assert payload["route"]["name"] == "handbook"
    assert payload["route"]["matched"] is False
    assert "default" in payload["route"]["reason"]


def test_status_rejects_an_unknown_route_and_lists_the_available_names() -> None:
    """未知路由 → 400，消息里列出**全部**可用名（读报的人要抄的就是它）."""
    response = make_client(router=two_route_router()).get(
        "/retrieval/status", params={"route": "handbok"}
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "未知路由" in detail
    assert "可用路由" in detail
    for name in ("handbook", "source"):
        assert name in detail


def test_status_requires_an_explicit_route_when_several_routes_exist() -> None:
    """多个索引且没有默认路由 → 400（"随便挑一个"会给出看起来正常的错误答案）."""
    response = make_client(router=two_route_router()).get("/retrieval/status")
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "必须显式指定 route" in detail
    assert "handbook" in detail and "source" in detail


def test_status_rejects_an_empty_router() -> None:
    """空路由表 → 400 且说明"要先建索引"（这不是参数写错，而是索引还没装配）."""
    response = make_client(router=StoreRouter()).get("/retrieval/status")
    assert response.status_code == 400
    assert "没有注册任何索引" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# POST /retrieval/search：正常路径
# --------------------------------------------------------------------------- #


def test_search_returns_the_top_k_hits_in_order(client: TestClient) -> None:
    """``top_k=3``：名次、分数、深度、候选数、三个减法数字一起对齐."""
    payload = search(client, top_k=3)
    result = payload["result"]
    assert result["count"] == 3
    assert [hit["record_id"] for hit in result["hits"]] == list(EXPECTED_AXIS_ORDER[:3])
    assert [hit["score"] for hit in result["hits"]] == [
        round(score, 6) for score in EXPECTED_AXIS_SCORES[:3]
    ]
    # rank 从 0 起连续：三道减法会留下空洞，重排名次就是为了消掉它
    assert [hit["rank"] for hit in result["hits"]] == [0, 1, 2]
    assert result["fetch_k"] == 9
    assert result["candidates"] == ALL_CANDIDATES
    assert result["filter_applied"] is False
    assert result["dropped_below_threshold"] == 0
    assert result["dropped_by_diversity"] == 0
    assert result["dropped_by_top_k"] == 5
    assert result["empty_reason"] == "hits"
    assert result["channels"] == ["vector"]
    assert payload["empty_reason"] == "hits"
    assert payload["empty_reason_description"] == "有命中，无需诊断"
    assert payload["routes"] == ["default"]
    # 命中自带正文与展示口径（这个端点要的就是"看一眼检索结果"）
    first = result["hits"][0]
    assert first["text"] == RECORD_TEXTS["c-a-01"]
    assert first["doc_id"] == "doc-alpha"
    assert first["heading_path"] == "检索手册 > 分块"
    assert first["channel"] == "vector"
    assert first["char_count"] == len(RECORD_TEXTS["c-a-01"])
    assert result["doc_ids"] == ["doc-alpha", "doc-beta"]


def test_search_uses_the_project_default_top_k_when_the_body_omits_it(
    client: TestClient,
) -> None:
    """省略 ``top_k``：用 ``settings.retrieval_top_k``，深度按三倍过取."""
    result = result_of(client)
    assert result["query"]["top_k"] == settings.retrieval_top_k
    assert result["count"] == settings.retrieval_top_k
    assert result["fetch_k"] == settings.retrieval_top_k * settings.retrieval_fetch_multiplier
    assert [hit["record_id"] for hit in result["hits"]] == list(
        EXPECTED_AXIS_ORDER[: settings.retrieval_top_k]
    )


def test_search_strips_the_query_before_encoding_it() -> None:
    """首尾空格被收敛：编码器收到的与回显的都是 strip 之后的那一份.

    不收敛的后果是"同一个问题因为多了两个空格而被当成两次不同的检索"
    （缓存与评估都会各算一次），而它不会报错。
    """
    embedding = TableEmbedding()
    client = make_client(embedding=embedding)
    result = result_of(client, query=f"  {AXIS}  ")
    assert result["query"]["text"] == AXIS
    assert embedding.calls == [AXIS]


def test_search_applies_where_and_reports_the_candidate_set(client: TestClient) -> None:
    """``where`` 过滤：候选数从 8 降到 4，条数、条件渲染与字段名一起给."""
    payload = search(client, where={"strategy": "structural"})
    result = payload["result"]
    assert result["filter_applied"] is True
    assert result["candidates"] == len(STRUCTURAL_IDS)
    assert [hit["record_id"] for hit in result["hits"]] == list(STRUCTURAL_IDS)
    assert result["dropped_by_top_k"] == 0
    assert payload["filter_fields"] == ["strategy"]
    assert "structural" in payload["conditions"]


def test_search_applies_a_closed_time_range(client: TestClient) -> None:
    """时间范围是闭区间：两端都给的期望命中是手算出来的那五条."""
    payload = search(
        client,
        time_range={"start": "2026-09-02", "end": "2026-09-30"},
    )
    result = payload["result"]
    assert result["filter_applied"] is True
    assert result["candidates"] == len(SEPTEMBER_IDS)
    assert [hit["record_id"] for hit in result["hits"]] == list(SEPTEMBER_IDS)
    assert "created_at ∈ [2026-09-02, 2026-09-30]" in payload["conditions"]


def test_search_excludes_the_record_that_has_no_time_field(client: TestClient) -> None:
    """**只给下界**时也要能看出"缺字段的记录被排除"（显式约定，不是 bug）.

    八条记录里有七条带 ``created_at``，因此候选数是 7 而不是 8；
    被排除的那条 ``c-c-01`` 不能出现在命中里——它既不在区间内也不在区间外。
    """
    payload = search(client, time_range={"start": "2026-01-01"})
    result = payload["result"]
    assert result["candidates"] == SINGLE_SIDED_CANDIDATES
    assert MISSING_CREATED_AT_ID not in [hit["record_id"] for hit in result["hits"]]
    assert result["query"]["time_range"]["start"] == "2026-01-01"
    assert result["query"]["time_range"]["end"] is None


def test_search_keeps_hits_exactly_at_the_threshold(client: TestClient) -> None:
    """阈值用 ``>=``：分数恰好等于阈值的两条必须留下，被切掉的条数如实记账."""
    payload = search(client, min_score=1.0)
    result = payload["result"]
    assert [hit["record_id"] for hit in result["hits"]] == list(TIE_IDS)
    assert result["dropped_below_threshold"] == ALL_CANDIDATES - len(TIE_IDS)
    assert result["empty_reason"] == "hits"


def test_search_below_threshold_is_reported_as_the_single_empty_reason(
    client: TestClient,
) -> None:
    """阈值把命中全切了：``empty_reason`` 只能是 ``below_threshold``，不是"没数据"."""
    payload = search(client, query=ORTHO, min_score=0.1)
    result = payload["result"]
    assert result["count"] == 0
    assert result["candidates"] == ALL_CANDIDATES
    assert result["dropped_below_threshold"] == ALL_CANDIDATES
    assert result["empty_reason"] == BELOW_THRESHOLD
    assert payload["empty_reason"] == BELOW_THRESHOLD
    assert payload["empty_reason_description"]
    assert result["hits"] == []


def test_search_filtered_out_reports_a_zero_candidate_set(client: TestClient) -> None:
    """过滤把候选筛成 0：原因归到过滤器这一边，并留下"检查字段名"的注记."""
    payload = search(client, where={"stratgy": "structural"})
    result = payload["result"]
    assert result["candidates"] == 0
    assert result["filter_applied"] is True
    assert result["empty_reason"] == FILTERED_OUT
    assert any("过滤条件把候选筛成了 0 条" in note for note in result["notes"])


def test_search_on_an_empty_store_returns_no_data_instead_of_500() -> None:
    """空库是**合法状态**：200 + ``no_data``，深度照样报出来（15 不是 0）."""
    payload = search(make_client(store=empty_store()))
    result = payload["result"]
    assert result["count"] == 0
    assert result["candidates"] == 0
    assert result["fetch_k"] == settings.retrieval_top_k * settings.retrieval_fetch_multiplier
    assert result["empty_reason"] == NO_DATA
    assert any("库是空的" in note for note in result["notes"])
    assert payload["empty_reason_description"] == "库是空的（合法状态）：请先建索引"


def test_search_trims_by_diversity_and_renumbers_the_ranks(client: TestClient) -> None:
    """``max_per_doc=1``：同一篇文档只留一条，被挤掉的计入 ``dropped_by_diversity``."""
    payload = search(client, max_per_doc=1)
    result = payload["result"]
    assert [hit["record_id"] for hit in result["hits"]] == list(DIVERSITY_IDS)
    assert result["dropped_by_diversity"] == ALL_CANDIDATES - len(DIVERSITY_IDS)
    assert result["dropped_by_top_k"] == 0
    assert [hit["rank"] for hit in result["hits"]] == [0, 1, 2]
    assert sorted(result["doc_ids"]) == ["doc-alpha", "doc-beta", "doc-gamma"]


def test_search_notes_a_capped_fetch_depth(client: TestClient) -> None:
    """深度封顶：``top_k=999`` 时深度是 ``MAX_FETCH_K``，且注记里写明"已封顶"."""
    result = result_of(client, top_k=MAX_FETCH_K - 1)
    assert result["fetch_k"] == MAX_FETCH_K
    assert any("已封顶" in note for note in result["notes"])
    # 库只有 8 条：深度再大也只是白扫，命中数仍然是 8
    assert result["count"] == ALL_CANDIDATES


def test_search_patterns_the_expected_order_for_the_tilt_query(client: TestClient) -> None:
    """换一个查询（``tilt``）得到另一份手算次序——证明排序不是"总是样本顺序"."""
    result = result_of(client, query=TILT, top_k=3)
    assert [hit["record_id"] for hit in result["hits"]] == ["c-c-02", "c-a-02", "c-a-03"]
    assert [hit["score"] for hit in result["hits"]] == [1.0, 0.96, 0.8]
    assert result["doc_ids"] == ["doc-gamma", "doc-alpha"]


# --------------------------------------------------------------------------- #
# POST /retrieval/search：400 / 404 分支
# --------------------------------------------------------------------------- #


def test_search_rejects_a_blank_query(client: TestClient) -> None:
    """空白查询 → 400：空白串会编码成一个固定向量，于是每次查询都返回同一批."""
    response = client.post("/retrieval/search", json={"query": "   "})
    assert response.status_code == 400
    assert "空查询" in response.json()["detail"]


def test_search_rejects_top_k_zero(client: TestClient) -> None:
    """``top_k=0`` 不是"没给"（本组的 None 才是"没给"）→ 400."""
    response = client.post("/retrieval/search", json={"query": AXIS, "top_k": 0})
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "top_k 必须 >= 1" in detail
    assert "backend.count()" in detail


def test_search_rejects_top_k_beyond_the_ceiling(client: TestClient) -> None:
    """``top_k`` 超过上限 → 400，消息里带上限值（与 vectorstore 同一个常量）."""
    response = client.post(
        "/retrieval/search", json={"query": AXIS, "top_k": MAX_FETCH_K + 1}
    )
    assert response.status_code == 400
    assert str(MAX_FETCH_K) in response.json()["detail"]


def test_search_rejects_fetch_k_smaller_than_top_k(client: TestClient) -> None:
    """深度小于条数 → 400：那些"本来能进 top_k"的候选根本不会被取回来."""
    response = client.post(
        "/retrieval/search", json={"query": AXIS, "top_k": 3, "fetch_k": 2}
    )
    assert response.status_code == 400
    assert "召回深度必须 >= 要返回的条数" in response.json()["detail"]


def test_search_rejects_max_per_doc_zero(client: TestClient) -> None:
    """``max_per_doc=0`` → 400："最多 0 条"不是一个请求（``None`` 才是不限）."""
    response = client.post(
        "/retrieval/search", json={"query": AXIS, "max_per_doc": 0}
    )
    assert response.status_code == 400
    assert "max_per_doc 必须 >= 1" in response.json()["detail"]


def test_search_rejects_an_inverted_time_range(client: TestClient) -> None:
    """时间范围倒置 → 400：倒置的闭区间必然返回空集，当场拒掉而不是去库里筛."""
    response = client.post(
        "/retrieval/search",
        json={"query": AXIS, "time_range": {"start": "2026-09-30", "end": "2026-09-01"}},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "时间范围倒置" in detail
    assert "start <= end" in detail


def test_search_rejects_a_time_range_that_is_not_iso(client: TestClient) -> None:
    """时间写法不认识 → 400，消息里列出接受的三种写法（读报的人不用去翻文档）."""
    response = client.post(
        "/retrieval/search",
        json={"query": AXIS, "time_range": {"start": "2026/09/01"}},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "ISO-8601" in detail
    assert "2026-09-20" in detail


def test_search_rejects_a_time_field_that_also_appears_in_where(client: TestClient) -> None:
    """同一个字段被 where 与 time_range 同时过滤 → 400（两组条件请自己合并成区间）."""
    response = client.post(
        "/retrieval/search",
        json={
            "query": AXIS,
            "where": {"created_at": "2026-09-01"},
            "time_range": {"start": "2026-09-01"},
        },
    )
    assert response.status_code == 400
    assert "过滤条件冲突" in response.json()["detail"]


def test_search_rejects_a_filter_conflict_even_on_an_empty_store() -> None:
    """空库也不该把"条件冲突"降级成 ``no_data``：同一种调用方错误必须行为一致.

    ``Retriever.retrieve`` 的第 5 步（合成过滤条件）排在"库是空的"那条早退
    **之前**，因此这条 400 在任何库状态下都成立。若早退跑到前面，同一个写错的
    条件会从 400 变成"库里没数据"——后者看起来像是"条件生效了，只是恰好没数据"，
    而真正要改的是那份冲突的条件。
    """
    response = make_client(store=empty_store()).post(
        "/retrieval/search",
        json={
            "query": AXIS,
            "where": {"created_at": "2026-09-01"},
            "time_range": {"start": "2026-09-02"},
        },
    )
    assert response.status_code == 400
    assert "过滤条件冲突" in response.json()["detail"]


def test_search_rejects_an_invalid_where_clause(client: TestClient) -> None:
    """``where`` 语法非法 → 400，消息里列出支持的运算符（与 /vectorstore/search 同源）."""
    response = client.post(
        "/retrieval/search",
        json={"query": AXIS, "where": {"strategy": {"$regex": "a"}}},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "支持的运算符" in detail
    assert "$contains" in detail


def test_search_rejects_an_unknown_route_name() -> None:
    """未知路由名 → 400，消息里列出可用名 + 三段式的出路."""
    response = make_client(router=two_route_router(default="handbook")).post(
        "/retrieval/search", json={"query": AXIS, "route": "handbok"}
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "未知路由 'handbok'" in detail
    assert "可用路由" in detail
    assert "不传 route" in detail


def test_search_requires_an_explicit_route_when_several_routes_exist() -> None:
    """没传 route、有两条路、又没有默认路由 → 400（不替调用方挑一个）."""
    response = make_client(router=two_route_router()).post(
        "/retrieval/search", json={"query": AXIS}
    )
    assert response.status_code == 400
    assert "必须显式指定 route" in response.json()["detail"]


def test_search_rejects_an_empty_router() -> None:
    """空路由表 → 400：这是"索引还没装配好"，不是"没找到"."""
    response = make_client(router=StoreRouter()).post(
        "/retrieval/search", json={"query": AXIS}
    )
    assert response.status_code == 400
    assert "没有注册任何索引" in response.json()["detail"]


def test_search_rejects_a_bad_body_shape_with_422(client: TestClient) -> None:
    """字段类型不对是 422（请求形状问题），与业务规则的 400 分开."""
    response = client.post("/retrieval/search", json={"query": AXIS, "top_k": "三条"})
    assert response.status_code == 422


def test_search_reports_the_route_decision_in_the_result_notes() -> None:
    """显式选路：响应回显 ``route``，而结果的 ``notes`` 里也有同一句话.

    notes 里那一行来自 ``StoreRouter.retrieve``（路由层自己的交代），
    端点上不再补写一遍——两处各写一份，迟早会不一致。
    """
    payload = search(
        make_client(router=two_route_router()), query=AXIS, route="source"
    )
    assert payload["route"]["name"] == "source"
    assert payload["route"]["matched"] is True
    assert any("路由 source（显式）" in note for note in payload["result"]["notes"])


# --------------------------------------------------------------------------- #
# POST /retrieval/explain
# --------------------------------------------------------------------------- #


def test_explain_answers_the_four_questions_for_a_hit(client: TestClient) -> None:
    """正常路径：四行固定回答四件事，检索器再补一行"分组口径"."""
    payload = explain(client)
    lines = payload["lines"]
    assert payload["empty_reason"] == "hits"
    assert payload["count"] == settings.retrieval_top_k
    assert payload["ids"] == list(EXPECTED_AXIS_ORDER[: settings.retrieval_top_k])
    assert lines[0].startswith("索引版本")
    assert lines[1].startswith("过滤条件")
    assert "阈值切掉 0 条" in lines[2]
    assert "top_k 截断丢掉 3 条" in lines[2]
    assert "空结果原因：hits" in lines[3]
    assert any(line.startswith("分组口径") for line in lines)
    assert "（无过滤）" in lines[1]


def test_explain_lines_are_the_same_diagnosis_as_search(client: TestClient) -> None:
    """同一次请求：``explain`` 的 ``lines`` 与 ``search`` 的诊断逐行相同.

    两个端点分开是"一个形状服务一个用途"，而不是"两次不同的检索"——
    这条断言就是在钉这一点。
    """
    body = {"query": AXIS, "where": {"strategy": "structural"}}
    assert explain(client, **body)["lines"] == search(client, **body)["lines"]


def test_explain_names_the_single_reason_on_an_empty_store() -> None:
    """空库：原因唯一是 ``no_data``，行里写着"库是空的（合法状态）"."""
    payload = explain(make_client(store=empty_store()))
    assert payload["empty_reason"] == NO_DATA
    assert payload["count"] == 0
    assert payload["ids"] == []
    assert any("库是空的（合法状态）" in line for line in payload["lines"])


def test_explain_names_the_threshold_when_it_empties_the_result(client: TestClient) -> None:
    """阈值切光：诊断行给出"threshold 切掉 8 条"与"降低阈值"的出路."""
    payload = explain(client, query=ORTHO, min_score=0.1)
    assert payload["empty_reason"] == BELOW_THRESHOLD
    assert any("阈值切掉 8 条" in line for line in payload["lines"])
    assert any("min_score 把命中的全切了" in line for line in payload["lines"])


def test_explain_reports_a_field_spelling_mistake(client: TestClient) -> None:
    """字段拼写检查（需要读库，因此只能由检索器给出）：拼错的字段被点名."""
    payload = explain(client, where={"stratgy": "structural"})
    assert payload["empty_reason"] == FILTERED_OUT
    assert any(
        "字段拼写检查" in line and "stratgy" in line for line in payload["lines"]
    )
    assert any("strategy" in line for line in payload["lines"])


def test_explain_rejects_the_same_bad_parameters_as_search(client: TestClient) -> None:
    """诊断端点与检索端点是同一条错误通道（400 + 消息里的合法取值）."""
    zero = client.post("/retrieval/explain", json={"query": AXIS, "top_k": 0})
    assert zero.status_code == 400
    assert "top_k 必须 >= 1" in zero.json()["detail"]
    inverted = client.post(
        "/retrieval/explain",
        json={"query": AXIS, "time_range": {"start": "2026-09-30", "end": "2026-09-01"}},
    )
    assert inverted.status_code == 400
    assert "时间范围倒置" in inverted.json()["detail"]


def test_explain_rejects_an_unknown_route() -> None:
    """未知路由 → 400，消息里列出可用名."""
    response = make_client(router=two_route_router(default="handbook")).post(
        "/retrieval/explain", json={"query": AXIS, "route": "handbok"}
    )
    assert response.status_code == 400
    assert "可用路由" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# POST /retrieval/answer
# --------------------------------------------------------------------------- #


def test_answer_calls_the_llm_with_a_numbered_context() -> None:
    """命中路径：模型收到带 ``[1]`` 的提示词，响应带回答案、引用与上下文账."""
    llm = RecordingLLM()
    payload = answer(make_client(llm=llm))
    assert payload["llm_called"] is True
    assert payload["answer"] == RecordingLLM.REPLY
    assert payload["prompt_version"] == RAG_ANSWER_PROMPT_VERSION
    assert len(llm.calls) == 1
    prompt = llm.prompt_text
    assert "[1]" in prompt and "[2]" in prompt
    assert AXIS in prompt
    # 编号对照表与提示词一一对应：第一条就是那次检索的第一名
    assert [citation["marker"] for citation in payload["citations"]] == [1, 2, 3, 4, 5]
    assert payload["citations"][0]["record_id"] == EXPECTED_AXIS_ORDER[0]
    assert payload["citations"][0]["source"] == SOURCE_ALPHA
    assert payload["citations"][0]["heading_path"] == "检索手册 > 分块"
    assert payload["context"]["count"] == settings.retrieval_top_k
    assert payload["context"]["truncated_hits"] == 0
    assert payload["context"]["dropped"] == []
    assert payload["retrieval"]["count"] == settings.retrieval_top_k
    assert payload["retrieval"]["index"]["count_store"] == ALL_CANDIDATES
    assert payload["empty_reason"] == "hits"
    assert payload["route"]["name"] == "default"


def test_answer_rejects_the_mock_placeholder_llm() -> None:
    """未配置 LLM（退避成 MockLLM）→ 400，且消息说明**为什么不拿它兜底**."""
    response = make_client(llm=MockLLM()).post(
        "/retrieval/answer", json={"question": AXIS}
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "MockLLM" in detail
    assert "护栏" in detail
    assert "BaseLLM" in detail


def test_answer_rejects_a_missing_llm() -> None:
    """连占位实现都没有（``app.state.llm is None``）→ 400 且给出注入的出路."""
    app = create_app(
        embedding=TableEmbedding(), vector_store=sample_store(), llm=MockLLM()
    )
    app.state.llm = None
    response = TestClient(app).post("/retrieval/answer", json={"question": AXIS})
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "没有配置 LLM" in detail
    assert "create_app(llm=...)" in detail


def test_answer_does_not_call_the_llm_on_an_empty_store() -> None:
    """空库：护栏生效——``llm_called=False``、兜底答复、引用为空、一次都没调."""
    llm = ExplodingLLM()
    payload = answer(make_client(store=empty_store(), llm=llm))
    assert llm.calls == 0
    assert payload["llm_called"] is False
    assert payload["answer"] == FALLBACK_NO_CONTEXT
    assert payload["citations"] == []
    assert payload["context"] is None
    assert payload["empty_reason"] == NO_DATA
    assert any("没有调用 LLM" in note for note in payload["notes"])


def test_answer_does_not_call_the_llm_when_the_filter_matches_nothing() -> None:
    """过滤后无命中：同样一次都不调，且原因归到过滤器这一边."""
    llm = ExplodingLLM()
    payload = answer(
        make_client(llm=llm), where={"strategy": "nonexistent-strategy"}
    )
    assert llm.calls == 0
    assert payload["llm_called"] is False
    assert payload["empty_reason"] == FILTERED_OUT
    assert payload["retrieval"]["candidates"] == 0


def test_answer_rejects_a_blank_question() -> None:
    """空白问题 → 400（与检索端点是同一条消息）."""
    response = make_client(llm=RecordingLLM()).post(
        "/retrieval/answer", json={"question": "  "}
    )
    assert response.status_code == 400
    assert "空查询" in response.json()["detail"]


def test_answer_rejects_an_unknown_route() -> None:
    """未知路由 → 400；选路发生在检索之前，因此它不会被"空库"掩盖过去."""
    response = make_client(
        router=two_route_router(default="handbook"), llm=RecordingLLM()
    ).post("/retrieval/answer", json={"question": AXIS, "route": "handbok"})
    assert response.status_code == 400
    assert "可用路由" in response.json()["detail"]


def test_answer_respects_the_selected_route_and_reports_it() -> None:
    """显式选路：答案、引用与 ``route`` 都来自被选中的那一条路."""
    payload = answer(
        make_client(router=two_route_router(default="handbook"), llm=RecordingLLM()),
        route="source",
    )
    assert payload["route"]["name"] == "source"
    assert payload["route"]["matched"] is True
    assert payload["llm_called"] is True
    assert payload["retrieval"]["query"]["route"] == "source"


def test_answer_reads_the_retrieval_llm_override_first() -> None:
    """``app.state.retrieval_llm`` 优先于 ``app.state.llm``：可给问答单独配模型."""
    app = create_app(
        embedding=TableEmbedding(), vector_store=sample_store(), llm=MockLLM()
    )
    override = RecordingLLM(reply="只用检索专用模型回答。")
    setattr(app.state, "retrieval_llm", override)
    payload = answer(TestClient(app))
    assert payload["answer"] == "只用检索专用模型回答。"
    assert len(override.calls) == 1
    assert payload["llm_called"] is True


# --------------------------------------------------------------------------- #
# GET /retrieval/routes
# --------------------------------------------------------------------------- #


def test_routes_lists_the_default_single_route(client: TestClient) -> None:
    """缺省 app：一条路由，名字与默认名都是 ``default``，现状带在里面."""
    payload = routes(client)
    assert payload["count"] == 1
    assert payload["names"] == ["default"]
    assert payload["default"] == "default"
    assert "1 条路由" in payload["summary"]
    route = payload["routes"][0]
    assert route["name"] == "default"
    assert route["description"]
    assert route["retriever"]["name"] == "default"
    assert route["retriever"]["index"]["count_store"] == ALL_CANDIDATES
    assert payload["limitations"] == list(RETRIEVAL_LIMITATIONS)
    assert payload["out_of_scope"] == list(RETRIEVAL_OUT_OF_SCOPE)


def test_routes_lists_every_registered_route_with_its_description() -> None:
    """两条路由：名字升序、逐条带说明与自己的现状；没设默认就如实报告空串."""
    payload = routes(make_client(router=two_route_router()))
    assert payload["count"] == 2
    assert payload["names"] == ["handbook", "source"]
    assert payload["default"] == ""
    assert "（未设置）" in payload["summary"]
    assert [route["name"] for route in payload["routes"]] == ["handbook", "source"]
    assert payload["routes"][1]["description"] == "源码库：函数签名与实现片段"
    assert payload["routes"][0]["retriever"]["name"] == "handbook"


def test_routes_rejects_an_empty_router() -> None:
    """空路由表 → 400（"有一张空表"会让调用方以为检索是可用的）."""
    response = make_client(router=StoreRouter()).get("/retrieval/routes")
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "没有注册任何索引" in detail
    assert "/indexing/build" in detail


def test_routes_is_read_only_and_reproducible(client: TestClient) -> None:
    """路由清单是纯读：连读两次逐字节相同，且库的条数一条不变."""
    first = routes(client)
    second = routes(client)
    assert first == second
    assert status(client)["index"]["count_store"] == ALL_CANDIDATES


# --------------------------------------------------------------------------- #
# 契约层面的两个收口
# --------------------------------------------------------------------------- #


def test_two_app_instances_do_not_share_the_index() -> None:
    """状态挂在应用实例上：一个 app 里删掉一条，另一个 app 看到的仍是八条."""
    store_a = sample_store()
    client_a = make_client(store=store_a, embedding=TableEmbedding())
    client_b = make_client()
    store_a.delete(["c-a-01"])
    assert status(client_a)["index"]["count_store"] == ALL_CANDIDATES - 1
    assert status(client_b)["index"]["count_store"] == ALL_CANDIDATES


def test_every_response_body_is_json_serializable(client: TestClient) -> None:
    """五个端点的响应体都能被 ``json.dumps``：端点不把 dataclass 直接塞进响应.

    ``TestClient`` 已经把响应体解析成了 Python 对象，因此这里再序列化一次
    是**多余的**——但多余得很值：它挡住的是"某天有人往响应里塞了一个
    ``RetrievalResult`` 对象、而 FastAPI 恰好也能编码它"这种混用。
    """
    bodies = [
        status(client),
        search(client, where={"strategy": "structural"}),
        explain(client, min_score=1.0),
        routes(client),
    ]
    for body in bodies:
        assert json.dumps(body, ensure_ascii=False)
    assert "application/json" in client.get("/retrieval/routes").headers["content-type"]
