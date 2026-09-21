"""day068 ``/retrieval/rerank`` 系列三个端点的单元测试.

全部离线、确定性、零网络：样本来自 ``tests/hybrid_samples.py``（十条记录 +
``TableEmbedding`` 查表编码器），因此"重排把谁从第几名提到第几名"
是一个**能在纸上算出来的数**，不是"跑一遍看看"。

```text
查询 ERR-2043（样本里唯一含这个编号的正文是 h-t-01，而它的向量被刻意
              放在第 8 根轴上——见 hybrid_samples 的模块 docstring）
第一阶段   分数 1.0 / 0 / …（h-c-01 拿 1.0，其余并列 0.0，按 id 升序）
重排       h-t-01 的四个特征：覆盖率 1.0、整句包含 1.0、邻近度 1/3、长度 1.0
           → 0.5 + 0.25 + 0.125/3 + 0.125 = 0.916667
           其余七条只有 length_penalty 那一项 → 0.125（文本都短于 IDEAL_LEN）
```

除了"三个端点各自返回什么"，这里钉住七件比 200 更重要的事：

1. **强制开启，而且拒绝要说得清**：``enabled=False`` 返回 400（**不静默忽略**），
   而 ``retrieval_rerank_enabled=False``（缺省）时这条通道**照样重排**
   ——它管的是检索链路上的缺省装配，不是这条显式通道；
2. **前后两个次序一起给**：``before_ids`` 与 ``after_ids`` 都用手算常量断言，
   这是本端点唯一的存在理由（只给重排后的名单，读的人无法判断次序动没动）；
3. **写回的字段是两份分数**：命中上的 ``rerank_score`` / ``stage1_rank`` 已回填，
   而 ``score`` 仍然是第一阶段的分数（口径不变）；
4. **窗口纪律**：``top_n=1`` 时窗口外的 7 条一条都没被打分（``rerank_score`` 是
   0.0 而不是 None，由 ``scored`` 与注记负责解释），且**那次重排什么都没改动**；
5. **``min_score`` 只落一次刀**（落在重排那一侧）：请求体里那一个数字在折成检索
   查询时被显式清空，因此它**不会**顺带把召回也切一遍（两处量的不是同一种分数）；
6. **三条指标能手算**：``recall@5`` / ``RR`` / ``nDCG@5`` 的前后值都写成常量，
   ``lift == after - before`` 逐项核对，并额外钉住"重排不是总更好"（负提升）；
7. **day066/067 的七个端点逐位不变**：开关关着时 ``rerank == {}``、
   ``dropped_by_rerank == 0``、命中里 ``rerank_score is None``。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest
from fastapi.testclient import TestClient

from smart_research_agent.api.app import create_app
from smart_research_agent.api.routes import RETRIEVAL_RERANK_STATE_KEY
from smart_research_agent.config import settings
from smart_research_agent.llm.base import BaseLLM, Message
from smart_research_agent.retrieval import (
    DEFAULT_RERANK_BATCH,
    EMPTY_REASON_DESCRIPTIONS,
    IDEAL_LEN,
    RERANK_FEATURE_NAMES,
    RERANK_MODES,
    RETRIEVAL_LIMITATIONS,
    RETRIEVAL_OUT_OF_SCOPE,
    BaseReranker,
    CrossEncoderReranker,
)
from tests.hybrid_samples import (
    HYBRID_DEFAULT,
    QUERY_EXACT,
    RECORD_TEXTS,
    empty_store,
    hybrid_embedding,
    hybrid_store,
)

#: 三个新端点的路径（断言"检索端点从 7 个变 10 个"时用）.
RERANK_PATH = "/retrieval/rerank"
RERANK_STATUS_PATH = "/retrieval/rerank/status"
RERANK_COMPARE_PATH = "/retrieval/rerank/compare"

#: 主案例的请求体：``top_k=8, fetch_k=8`` 让窗口里有八条，
#: 于是"重排把第 8 名（0 起第 7）提到第 1 名"这件事看得见。
WIDE_BODY: dict[str, Any] = {"query": QUERY_EXACT, "top_k": 8, "fetch_k": 8}

#: 重排前的 id 次序（第一阶段：1.0 的那条在前，其余 0.0 并列、按 id 升序）.
BEFORE_IDS: tuple[str, ...] = (
    "h-c-01",
    "h-c-02",
    "h-c-03",
    "h-c-04",
    "h-m-01",
    "h-m-02",
    "h-m-03",
    "h-t-01",
)

#: 重排后的 id 次序（含编号那条被提到第 1 名，其余七条保持相对次序）.
AFTER_IDS: tuple[str, ...] = (
    "h-t-01",
    "h-c-01",
    "h-c-02",
    "h-c-03",
    "h-c-04",
    "h-m-01",
    "h-m-02",
    "h-m-03",
)

#: 含编号那条（``ERR-2043``）的 id 与重排分。手算：
#:
#: ```text
#: term_coverage 1.0（err 与 2043 两个词元都在正文里）  × 0.5
#: exact_phrase  1.0（正文以 ERR-2043 开头）             × 0.25
#: proximity     1/(1+2)（两个词元相邻，窗口宽度 2）      × 0.125
#: length_penalty 1.0（正文远短于 IDEAL_LEN=240）        × 0.125
#: = 0.875 + 1/24 = 0.916666…（结果里 round 到 6 位）
#: ```
MATCH_ID = "h-t-01"
MATCH_SCORE = 0.916667

#: 其余七条的重排分：四个特征里只有 ``length_penalty`` 那一项
#: （0.125 × 1.0）——它们既不覆盖查询词元、也不含查询原文、更谈不上邻近度。
BASE_SCORE = 0.125

#: 含编号那条在第一阶段里的位置（也是它的 ``stage1_rank``）.
MATCH_STAGE1_RANK = 7

#: 主案例里被命中正文的那四个特征（逐项可手算，见 MATCH_SCORE 的算式）.
MATCH_FEATURES: dict[str, float] = {
    "term_coverage": 1.0,
    "exact_phrase": 1.0,
    "proximity": 0.333333,
    "length_penalty": 1.0,
}

#: 没命中的那七条的特征：三个 0.0 加一个 1.0（它们都短于 IDEAL_LEN）.
MISS_FEATURES: dict[str, float] = {
    "term_coverage": 0.0,
    "exact_phrase": 0.0,
    "proximity": 0.0,
    "length_penalty": 1.0,
}

#: compare 用的那句话：一个两字术语，样本里**两条**正文含它（h-m-01 / h-m-02），
#: 而向量路在这句话上没有方向（它不在 ``TableEmbedding`` 的表里 → 兜底向量
#: 指向 h-m-01），于是第一阶段的名单里 h-m-01 在前、h-m-02 在后。
THRESHOLD_QUERY = "阈值"

#: compare 的第一阶段名单（默认深度 top_k=5；向量分数 1.0 / 0.8 / 0.6 / 0 / 0）.
THRESHOLD_BEFORE_IDS: tuple[str, ...] = ("h-m-01", "h-m-02", "h-c-03", "h-c-01", "h-c-02")

#: 重排之后 h-m-02 与 h-m-01 换位：两条的覆盖率与整句包含都是 1.0，
#: 差别只在**邻近度**——h-m-02 的正文更短、词元挨得更近（1/(1+9)=0.1），
#: h-m-01 的是 1/(1+13)=0.071429。于是 0.8875 > 0.883929。
THRESHOLD_AFTER_IDS: tuple[str, ...] = ("h-m-02", "h-m-01", "h-c-03", "h-c-01", "h-c-02")

#: 两条命中者的重排分（同样是 0.875 + 0.125 × proximity）.
THRESHOLD_MATCH_SCORES: dict[str, float] = {"h-m-02": 0.8875, "h-m-01": 0.883929}

#: 换位之后被挪动的两条（按**新**名次顺序）.
THRESHOLD_MOVED: tuple[str, ...] = ("h-m-02", "h-m-01")

#: 三条指标的期望值（k=5）。金标准取 h-m-02：它在第一阶段排第 2、重排后排第 1。
#:
#: ```text
#: recall@5   |前 5 条 ∩ {h-m-02}| / 1 = 1.0（两侧都在前 5 条里）
#: RR         before 1/(1+1)=0.5；after 1/(0+1)=1.0
#: nDCG@5     before 1/log2(1+2)=0.630930；after 1/log2(0+2)=1.0
#: lift       after - before（round 到 6 位）
#: ```
COMPARE_BEFORE: dict[str, float] = {"recall": 1.0, "reciprocal_rank": 0.5, "ndcg": 0.63093}
COMPARE_AFTER: dict[str, float] = {"recall": 1.0, "reciprocal_rank": 1.0, "ndcg": 1.0}
COMPARE_LIFT: dict[str, float] = {"recall": 0.0, "reciprocal_rank": 0.5, "ndcg": 0.36907}

#: 金标准换成"被挤下去的那一条"（h-m-01）时的负提升：它从第 1 名掉到第 2 名。
#: 这一条刻意留着——**重排不是总更好**，而报告必须能把这件事说出来。
DEMOTED_LIFT: dict[str, float] = {"recall": 0.0, "reciprocal_rank": -0.5, "ndcg": -0.36907}

#: 状态端点里 ``model`` 的键集合（``CrossEncoderReranker.describe()`` 的封闭形状）.
MODEL_KEYS: frozenset[str] = frozenset(
    {
        "name",
        "kind",
        "features",
        "weights",
        "batch_size",
        "ideal_len",
        "calls",
        "scored_pairs",
        "batches",
    }
)

#: 教学替身的四个特征权重（顺序 = ``RERANK_FEATURE_NAMES``）.
FEATURE_WEIGHTS: dict[str, float] = {
    "term_coverage": 0.5,
    "exact_phrase": 0.25,
    "proximity": 0.125,
    "length_penalty": 0.125,
}


class ReverseLengthReranker(BaseReranker):
    """按正文长度打分的自定义重排器（越短分越高）——**证明注入的那个真的在被用**.

    它的分数与四个特征毫无关系，而且 ``explain_pairs`` 走 ``BaseReranker`` 的
    缺省实现（返回空字典）：于是"注入生效"这件事有两个独立证据
    ——次序按长度排、以及逐条的特征是 ``{}``（教学替身才会摊开四个特征）。
    """

    def __init__(self, name: str = "reverse-by-length") -> None:
        self._name = name

    @property
    def name(self) -> str:
        """进报告的名字（缺省装配下响应里的 ``rerank.model`` 就是它）."""
        return self._name

    @property
    def dimension(self) -> int:
        """打分的维度（自定义实现随便取一个非零数，这里取 1）."""
        return 1

    def score_pairs(self, query: str, texts: Sequence[str]) -> list[float]:
        """``1 / (1 + len(text))``：与查询无关，只由正文长度决定."""
        return [1.0 / (1.0 + len(text)) for text in texts]

    def describe(self) -> dict[str, Any]:
        """自述（状态端点直接返回它）."""
        return {"name": self._name, "kind": "custom(test)", "dimension": self.dimension}


class RecordingLLM(BaseLLM):
    """离线假模型：只为让 ``/retrieval/answer`` 走通（答案内容不是本文件的重点）."""

    REPLY = "依据 [1]，阈值用分位数标定。"

    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """记下整轮消息并返回预设回复（不打网络、不读盘）."""
        self.calls.append(list(messages))
        return self.REPLY


def make_client(
    *,
    reranker: Any = None,
    store: Any = None,
    llm: BaseLLM | None = None,
) -> TestClient:
    """建一个可注入的离线客户端（两个注入点各自对应一类用例）.

    ``reranker`` 直接挂到 ``app.state`` 上（键名见 ``routes.RETRIEVAL_RERANK_STATE_KEY``），
    与 ``app.state.retrieval_hybrid`` 的注入方式逐字相同：重排器是"我明确要这条链路"
    的显式动作，因此它优先于缺省装配（而开关管的是缺省装配）。
    """
    app = create_app(
        embedding=hybrid_embedding(),
        vector_store=store if store is not None else hybrid_store(),
        llm=llm,
    )
    if reranker is not None:
        setattr(app.state, RETRIEVAL_RERANK_STATE_KEY, reranker)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _restore_settings():
    """每个用例结束后把两个开关复位（``settings`` 是进程级单例，必须自己收拾）."""
    yield
    settings.retrieval_rerank_enabled = False
    settings.retrieval_hybrid_enabled = False


def rerank_call(client: TestClient, **body: Any) -> dict:
    """发一次重排请求（缺省用主案例的请求体）."""
    payload: dict = dict(WIDE_BODY)
    payload.update(body)
    response = client.post(RERANK_PATH, json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def status_call(client: TestClient) -> dict:
    """读一次重排状态."""
    response = client.get(RERANK_STATUS_PATH)
    assert response.status_code == 200, response.text
    return response.json()


def compare_call(client: TestClient, **body: Any) -> dict:
    """发一次前后对照请求（缺省：``阈值`` + 金标准 h-m-02 + k=5）."""
    payload: dict = {"query": THRESHOLD_QUERY, "relevant": ["h-m-02"], "k": 5}
    payload.update(body)
    response = client.post(RERANK_COMPARE_PATH, json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def ids_of(hits: list[dict]) -> list[str]:
    """一组命中里的 id（按名次）."""
    return [hit["record_id"] for hit in hits]


def length_order(before_ids: list[str]) -> list[str]:
    """按正文长度升序、同长按原来的位置——这正是注入那个重排器给出的次序.

    期望值在用例里**现算**而不是抄一份常量：它写的是注入那个重排器的定义
    （``1 / (1 + len(text))``），因此"次序变了"这件事本身就是"注入生效了"的证据。
    """
    return sorted(before_ids, key=lambda rid: (len(RECORD_TEXTS[rid]), before_ids.index(rid)))


@pytest.fixture
def client() -> TestClient:
    """缺省客户端：十条样本 + 查表编码器 + 缺省装配（重排器**没有注入**）."""
    return make_client()


# --------------------------------------------------------------------------- #
# 端点清单
# --------------------------------------------------------------------------- #


class TestEndpointRegistry:
    """检索端点从 7 个变 10 个（这一条是"今天加了什么"的机器可核对的证据）. """

    def test_ten_retrieval_endpoints_are_registered(self) -> None:
        client = make_client()
        paths = [
            path
            for path in client.get("/openapi.json").json()["paths"]
            if path.startswith("/retrieval")
        ]

        assert len(paths) == 10
        assert RERANK_PATH in paths
        assert RERANK_STATUS_PATH in paths
        assert RERANK_COMPARE_PATH in paths

    def test_the_three_new_paths_keep_their_methods(self) -> None:
        schema = make_client().get("/openapi.json").json()["paths"]

        assert set(schema[RERANK_PATH]) == {"post"}
        assert set(schema[RERANK_STATUS_PATH]) == {"get"}
        assert set(schema[RERANK_COMPARE_PATH]) == {"post"}

    def test_the_request_models_are_exported_to_openapi(self) -> None:
        """三个请求/响应模型都进了契约文档（"接口长什么样"只在这里定义一次）."""
        schemas = make_client().get("/openapi.json").json()["components"]["schemas"]

        assert "RerankRequest" in schemas
        assert "RerankResponse" in schemas
        assert "RerankStatusResponse" in schemas
        assert "RerankCompareRequest" in schemas
        assert "RerankCompareResponse" in schemas
        # 请求体继承自检索那一个：六个重排字段都在（``min_score`` 也是其中之一）。
        properties = schemas["RerankRequest"]["properties"]
        assert {"enabled", "mode", "top_n", "weight", "min_score", "model"} <= set(properties)


# --------------------------------------------------------------------------- #
# 缺省关闭，而这条通道强制开启
# --------------------------------------------------------------------------- #


class TestRerankDisabledByDefault:
    """``retrieval_rerank_enabled=False``（缺省）时，这条通道**照样重排**."""

    def test_default_setting_is_off(self) -> None:
        assert settings.retrieval_rerank_enabled is False

    def test_the_endpoint_still_reorders_while_the_switch_is_off(self, client: TestClient) -> None:
        """缺省关闭的是"检索链路上的重排"，不是这条显式通道——次序**确实被改了**.

        这一条是本端点的价值所在：它必须让人看见"重排动了什么"，
        而不是让人看见"开关没开、所以什么都没发生"。
        """
        payload = rerank_call(client)

        assert settings.retrieval_rerank_enabled is False
        assert tuple(payload["before_ids"]) == BEFORE_IDS
        assert tuple(payload["after_ids"]) == AFTER_IDS
        assert payload["before_ids"] != payload["after_ids"]

    def test_the_switch_off_leaves_the_search_channel_untouched(self, client: TestClient) -> None:
        """同一次请求体在 ``/retrieval/search`` 上什么都不改（day066/067 口径）."""
        result = client.post(
            "/retrieval/search", json={"query": QUERY_EXACT, "top_k": 8, "fetch_k": 8}
        ).json()["result"]

        assert result["rerank"] == {}
        assert result["dropped_by_rerank"] == 0
        assert [hit["rerank_score"] for hit in result["hits"]] == [None] * 8
        assert ids_of(result["hits"]) == list(BEFORE_IDS)

    def test_enabling_the_switch_reorders_the_search_channel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """开关打开后，``/retrieval/search`` 给出的次序与 /retrieval/rerank **逐位相同**.

        两条通道共用同一个 ``rerank_hits`` 与同一个窗口口径，因此"接口给的那次重排"
        必须是"脚本/检索链路给的那次重排"——分家的表现是"两边差一点点"，
        而那种差异没有任何报错指向它。
        """
        monkeypatch.setattr(settings, "retrieval_rerank_enabled", True)
        client = make_client()

        result = client.post(
            "/retrieval/search", json={"query": QUERY_EXACT, "top_k": 8, "fetch_k": 8}
        ).json()["result"]
        payload = rerank_call(client)

        assert ids_of(result["hits"]) == list(AFTER_IDS)
        assert payload["after_ids"] == ids_of(result["hits"])
        assert result["rerank"]["scored"] == 8
        assert result["rerank"]["window"] == 8
        assert result["dropped_by_rerank"] == 0

    def test_status_reports_the_switch_off(self, client: TestClient) -> None:
        """状态端点如实报告 ``enabled=False``（它回答"现在是什么状态"）."""
        payload = status_call(client)

        assert payload["enabled"] is False
        assert payload["defaults"]["enabled"] is False


# --------------------------------------------------------------------------- #
# POST /retrieval/rerank
# --------------------------------------------------------------------------- #


class TestRerankEndpoint:
    """一次强制重排的完整交代（名次与分数都用手算常量核对）."""

    def test_both_orders_are_reported(self, client: TestClient) -> None:
        payload = rerank_call(client)

        assert tuple(payload["before_ids"]) == BEFORE_IDS
        assert tuple(payload["after_ids"]) == AFTER_IDS

    def test_the_window_book_keeping(self, client: TestClient) -> None:
        """窗口的四个数：候选 8 / 打分 8 / 窗口 8 / top_n 取 settings（20）."""
        payload = rerank_call(client)
        rerank = payload["rerank"]

        assert rerank["candidates"] == 8
        assert rerank["scored"] == 8
        assert rerank["window"] == 8
        assert rerank["top_n"] == settings.retrieval_rerank_top_n
        assert rerank["dropped_by_min_score"] == 0
        assert rerank["mode"] == "replace"
        assert rerank["model"] == settings.retrieval_rerank_model
        # ``rerank``（RerankResult.to_dict）里的 ``moved`` 是**逐条 id**：
        # 含编号那条上移 7 位、其余七条各被挤下一位，因此八条都"动过"。
        assert len(rerank["moved"]) == 8
        assert rerank["moved"][0] == MATCH_ID

    def test_hits_are_renumbered_from_zero(self, client: TestClient) -> None:
        hits = rerank_call(client)["result"]["hits"]

        assert [hit["rank"] for hit in hits] == list(range(8))

    def test_rerank_score_and_stage1_rank_are_written_back(self, client: TestClient) -> None:
        """两份分数各占一个字段：新名次 + 重排分 + 它原来在第几."""
        hits = {hit["record_id"]: hit for hit in rerank_call(client)["result"]["hits"]}

        assert hits[MATCH_ID]["rank"] == 0
        assert hits[MATCH_ID]["rerank_score"] == MATCH_SCORE
        assert hits[MATCH_ID]["stage1_rank"] == MATCH_STAGE1_RANK
        assert hits["h-c-01"]["rank"] == 1
        assert hits["h-c-01"]["stage1_rank"] == 0
        assert hits["h-c-01"]["rerank_score"] == BASE_SCORE
        for hit in hits.values():
            if hit["record_id"] == MATCH_ID:
                continue
            assert hit["rerank_score"] == BASE_SCORE
            assert hit["stage1_rank"] == hit["rank"] - 1

    def test_the_first_stage_score_is_untouched(self, client: TestClient) -> None:
        """``score`` 仍然是第一阶段的分数（口径不变），因此它可以与重排分背离.

        ``h-c-01`` 的第一阶段分数是 1.0（最高），重排分却只有 0.125；
        ``h-t-01`` 的第一阶段分数是 0.0，重排分却是 0.916667。
        把 ``score`` 覆盖成重排分会更"好看"，但代价是丢掉"它原来得了多少分"。
        """
        hits = {hit["record_id"]: hit for hit in rerank_call(client)["result"]["hits"]}

        assert hits["h-c-01"]["score"] == 1.0
        assert hits[MATCH_ID]["score"] == 0.0

    def test_the_four_features_are_hand_computable(self, client: TestClient) -> None:
        rows = {hit["record_id"]: hit for hit in rerank_call(client)["rerank"]["hits"]}

        assert rows[MATCH_ID]["features"] == MATCH_FEATURES
        assert rows["h-c-01"]["features"] == MISS_FEATURES
        assert rows[MATCH_ID]["score"] == MATCH_SCORE

    def test_lines_include_the_window_note(self, client: TestClient) -> None:
        joined = "\n".join(rerank_call(client)["lines"])

        assert "窗口：候选 8 条" in joined
        assert "只对前 top_n=20 条打分" in joined
        assert "窗口外 0 条" in joined
        assert "模式：replace" in joined
        assert f"#0 {MATCH_ID}" in joined
        assert "移动 +7" in joined
        assert "注记：只对前 N 条打分" in joined

    def test_summary_and_empty_reason(self, client: TestClient) -> None:
        payload = rerank_call(client)

        assert payload["empty_reason"] == "hits"
        assert payload["empty_reason_description"] == EMPTY_REASON_DESCRIPTIONS["hits"]
        assert payload["result"]["empty_reason"] == "hits"
        assert MATCH_ID in payload["summary"]
        assert payload["result"]["count"] == 8

    def test_the_route_decision_is_echoed(self, client: TestClient) -> None:
        payload = rerank_call(client)

        assert payload["route"]["name"] == "default"
        assert payload["result"]["query"]["route"] is None

    def test_limitations_and_out_of_scope_are_echoed(self, client: TestClient) -> None:
        payload = rerank_call(client)

        assert payload["limitations"] == list(RETRIEVAL_LIMITATIONS)
        assert payload["out_of_scope"] == list(RETRIEVAL_OUT_OF_SCOPE)
        assert any("只作用于前 N 条窗口" in item for item in payload["limitations"])
        assert any("教学级交叉编码器替身" in item for item in payload["limitations"])

    def test_defaults_echo_the_settings(self, client: TestClient) -> None:
        defaults = rerank_call(client)["defaults"]

        assert defaults["enabled"] == settings.retrieval_rerank_enabled
        assert defaults["model"] == settings.retrieval_rerank_model
        assert defaults["top_n"] == settings.retrieval_rerank_top_n
        assert defaults["mode"] == settings.retrieval_rerank_mode
        assert defaults["weight"] == settings.retrieval_rerank_weight
        assert defaults["min_score"] == settings.retrieval_rerank_min_score
        assert defaults["modes"] == list(RERANK_MODES)
        assert defaults["features"] == list(RERANK_FEATURE_NAMES)

    def test_a_narrow_window_leaves_everything_unchanged(self, client: TestClient) -> None:
        """``top_n=1``：窗口外的 7 条**一条都没打分**，于是次序一动不动.

        这正是窗口纪律的代价：含编号那条本应被提到第一，但它在窗口外——
        "名次可能本应更靠前，而这次重排看不见它"必须能被读出来（注记）。
        """
        payload = rerank_call(client, top_n=1)
        hits = payload["result"]["hits"]

        assert payload["after_ids"] == payload["before_ids"] == list(BEFORE_IDS)
        assert payload["rerank"]["scored"] == 1
        assert payload["rerank"]["window"] == 1
        assert payload["rerank"]["candidates"] == 8
        assert payload["rerank"]["moved"] == []
        assert hits[0]["record_id"] == "h-c-01"
        assert hits[0]["rerank_score"] == BASE_SCORE
        assert [hit["rerank_score"] for hit in hits[1:]] == [0.0] * 7
        assert [hit["stage1_rank"] for hit in hits] == list(range(8))
        assert any("窗口外的 7 条" in note for note in payload["rerank"]["notes"])

    def test_blend_weight_decides_who_wins(self, client: TestClient) -> None:
        """blend 下两份分数各自归一化后加权，权重决定结果（三个值都是手算的）.

        ```text
        weight=1.0  只信重排分      → h-t-01（blended 1.0）
        weight=0.5  两边各占一半    → h-c-01 与 h-t-01 的 blended **都是 0.5**，
                                    于是次序由第三键 stage1_rank 决定 → h-c-01 在前
        weight=0.0  只信第一阶段    → 次序回到第一阶段那一份
        ```
        """
        high = rerank_call(client, mode="blend", weight=1.0)
        half = rerank_call(client, mode="blend", weight=0.5)
        low = rerank_call(client, mode="blend", weight=0.0)

        assert high["after_ids"] == list(AFTER_IDS)
        assert high["rerank"]["hits"][0]["blended"] == 1.0
        assert half["after_ids"][:2] == ["h-c-01", MATCH_ID]
        assert half["rerank"]["hits"][0]["blended"] == 0.5
        assert half["rerank"]["hits"][1]["blended"] == 0.5
        assert low["after_ids"] == list(BEFORE_IDS)
        assert low["rerank"]["weight"] == 0.0

    def test_blend_records_the_weight_and_replace_notes_it(self, client: TestClient) -> None:
        """replace 下权重不参与排序，但**记录并说明**（不报错，与 fusion 的取舍不同）."""
        payload = rerank_call(client, mode="replace")

        assert payload["rerank"]["weight"] == settings.retrieval_rerank_weight
        assert any("不看权重" in note for note in payload["rerank"]["notes"])

    def test_min_score_only_cuts_once_and_on_the_rerank_side(self, client: TestClient) -> None:
        """``min_score`` 是**重排阈值**：它切掉的是重排分低于阈值的条.

        0.916667 的那条留下、七条 0.125 的切掉 → ``dropped_by_rerank == 7``；
        而 ``before_ids`` 仍然是八条——**第一阶段的阈值没有被顺带切一遍**
        （两个数字量的不是同一种分数：请求体里这一个在折成检索查询时被清空）。
        """
        payload = rerank_call(client, min_score=0.5)

        assert payload["before_ids"] == list(BEFORE_IDS)
        assert payload["after_ids"] == [MATCH_ID]
        assert payload["result"]["count"] == 1
        assert payload["result"]["dropped_by_rerank"] == 7
        assert payload["result"]["dropped_below_threshold"] == 0
        assert payload["empty_reason"] == "hits"

    def test_a_threshold_above_every_score_empties_the_result(self, client: TestClient) -> None:
        """阈值定得比全部分数都高 → 空名单，但**不是**重排失败（8 条一起记账）."""
        payload = rerank_call(client, min_score=0.99)

        assert payload["after_ids"] == []
        assert payload["result"]["count"] == 0
        assert payload["result"]["dropped_by_rerank"] == 8
        assert payload["empty_reason"] == "below_threshold"
        assert payload["empty_reason_description"] == EMPTY_REASON_DESCRIPTIONS["below_threshold"]
        assert any("阈值定得比全部分数都高" in note for note in payload["rerank"]["notes"])

    def test_the_model_parameter_only_enters_the_report(self, client: TestClient) -> None:
        """``model`` 不加载任何模型，只进报告；与实现名不符时记一条注记."""
        payload = rerank_call(client, model="my-rerank-v2")
        joined = "\n".join(payload["lines"])

        assert payload["rerank"]["model"] == "my-rerank-v2"
        assert "不按名字加载模型" in joined
        assert payload["after_ids"] == list(AFTER_IDS)

    def test_the_retrieval_parameters_are_honoured(self, client: TestClient) -> None:
        """继承来的检索字段照样生效（它们是同一次检索的两种排法）."""
        payload = rerank_call(client, top_k=3, fetch_k=3)

        assert payload["result"]["count"] == 3
        assert payload["result"]["fetch_k"] == 3
        assert payload["after_ids"] == ["h-c-01", "h-c-02", "h-c-03"]

    def test_where_is_honoured(self, client: TestClient) -> None:
        payload = rerank_call(client, where={"parent_doc_id": "doc-concept"})

        assert payload["before_ids"] == ["h-c-01", "h-c-02", "h-c-03", "h-c-04"]
        assert payload["after_ids"] == ["h-c-01", "h-c-02", "h-c-03", "h-c-04"]

    def test_an_empty_store_is_a_legal_state(self) -> None:
        """空库：200，重排**一次都不调用重排器**，原因仍是上游那一个（no_data）."""
        payload = rerank_call(make_client(store=empty_store()))

        assert payload["before_ids"] == []
        assert payload["after_ids"] == []
        assert payload["empty_reason"] == "no_data"
        assert payload["rerank"]["scored"] == 0
        assert payload["rerank"]["candidates"] == 0
        assert any("空输入不调用重排器" in note for note in payload["rerank"]["notes"])

    def test_every_response_field_is_json_serializable(self, client: TestClient) -> None:
        payload = rerank_call(client)
        rendered = json.dumps(payload, ensure_ascii=False)

        assert json.loads(rendered) == payload
        assert MATCH_ID in rendered


# --------------------------------------------------------------------------- #
# GET /retrieval/rerank/status
# --------------------------------------------------------------------------- #


class TestRerankStatusEndpoint:
    """重排的现状：模型自述 / 开关 / 四个参数 / 缺省值表. """

    def test_the_model_block_has_the_closed_shape(self, client: TestClient) -> None:
        model = status_call(client)["model"]

        assert set(model) == MODEL_KEYS
        assert model["name"] == settings.retrieval_rerank_model
        assert "teaching" in model["kind"]
        assert model["features"] == list(RERANK_FEATURE_NAMES)
        assert model["weights"] == FEATURE_WEIGHTS
        assert sum(model["weights"].values()) == 1.0
        assert model["batch_size"] == DEFAULT_RERANK_BATCH
        assert model["ideal_len"] == IDEAL_LEN

    def test_counters_stay_zero_without_injection(self, client: TestClient) -> None:
        """缺省装配下重排器**每次请求现装**，因此三个计数器永远是 0.

        这是"不缓存"这条纪律的代价，也是它的证据：一个现装的实例如实报告
        "我还没打过分"，而不是把一个跨请求累计的数冒充成全局统计。
        """
        assert status_call(client)["model"]["calls"] == 0
        rerank_call(client)
        described = status_call(client)["model"]

        assert described["calls"] == 0
        assert described["scored_pairs"] == 0
        assert described["batches"] == 0

    def test_counters_accumulate_when_an_instance_is_injected(self) -> None:
        """注入之后三个计数器才是累计账，而 ``scored_pairs`` 正是窗口纪律的账.

        一次 8 条的窗口 = 1 次调用 / 8 对 / 1 批（``DEFAULT_RERANK_BATCH=16``）。
        """
        injected = CrossEncoderReranker()
        client = make_client(reranker=injected)

        rerank_call(client)
        after_one = status_call(client)["model"]
        rerank_call(client)
        after_two = status_call(client)["model"]

        assert after_one["calls"] == 1
        assert after_one["scored_pairs"] == 8
        assert after_one["batches"] == 1
        assert after_two["calls"] == 2
        assert after_two["scored_pairs"] == 16
        assert after_two["batches"] == 2
        assert injected.scored_pairs == 16

    def test_the_note_says_it_is_a_teaching_stand_in(self, client: TestClient) -> None:
        payload = status_call(client)

        assert "教学级" in payload["note"]
        assert "没有真实语义" in payload["note"]
        assert "BaseReranker" in payload["note"]
        assert "教学级交叉编码器替身" in payload["summary"]

    def test_the_four_parameters_come_from_the_settings(self, client: TestClient) -> None:
        payload = status_call(client)

        assert payload["enabled"] is settings.retrieval_rerank_enabled
        assert payload["top_n"] == settings.retrieval_rerank_top_n
        assert payload["mode"] == settings.retrieval_rerank_mode
        assert payload["weight"] == settings.retrieval_rerank_weight
        assert payload["min_score"] == settings.retrieval_rerank_min_score
        assert payload["min_score"] is None

    def test_defaults_repeat_the_settings(self, client: TestClient) -> None:
        """``defaults`` 与顶层那四个数**刻意重复**（"这个数是默认还是被改过"）."""
        payload = status_call(client)

        assert payload["defaults"]["top_n"] == payload["top_n"]
        assert payload["defaults"]["mode"] == payload["mode"]
        assert payload["defaults"]["weight"] == payload["weight"]
        assert payload["defaults"]["min_score"] == payload["min_score"]
        assert payload["defaults"]["model"] == settings.retrieval_rerank_model
        assert payload["defaults"]["modes"] == list(RERANK_MODES)

    def test_limitations_and_out_of_scope_are_echoed(self, client: TestClient) -> None:
        payload = status_call(client)

        assert payload["limitations"] == list(RETRIEVAL_LIMITATIONS)
        assert payload["out_of_scope"] == list(RETRIEVAL_OUT_OF_SCOPE)

    def test_the_endpoint_is_read_only(self, client: TestClient) -> None:
        """两次读给出同一份答案（它不检索、不重排、不写任何状态）."""
        assert status_call(client) == status_call(client)

    def test_body_is_json_serializable(self, client: TestClient) -> None:
        assert json.dumps(status_call(client))


# --------------------------------------------------------------------------- #
# POST /retrieval/rerank/compare
# --------------------------------------------------------------------------- #


class TestRerankCompareEndpoint:
    """三条指标的前后对照：**重排有没有用**（数字全部手算）."""

    def test_the_three_metrics_are_hand_computable(self, client: TestClient) -> None:
        payload = compare_call(client)

        assert payload["before"] == COMPARE_BEFORE
        assert payload["after"] == COMPARE_AFTER
        assert payload["lift"] == COMPARE_LIFT
        assert payload["queries"] == 1
        assert payload["k"] == 5
        assert payload["metrics"] == ["ndcg", "recall", "reciprocal_rank"]

    def test_lift_equals_after_minus_before(self, client: TestClient) -> None:
        payload = compare_call(client)

        for name, value in payload["lift"].items():
            assert value == pytest.approx(payload["after"][name] - payload["before"][name])
        assert payload["per_query"][0]["lift"] == payload["lift"]

    def test_per_query_keeps_the_shape_of_measure_lift(self, client: TestClient) -> None:
        """``per_query`` 恒为 1 条，里面是与 ``measure_lift`` 逐字相同的明细."""
        row = compare_call(client)["per_query"][0]

        assert row["query"] == THRESHOLD_QUERY
        assert row["relevant"] == ["h-m-02"]
        assert row["k"] == 5
        assert tuple(row["before_ids"]) == THRESHOLD_BEFORE_IDS
        assert tuple(row["after_ids"]) == THRESHOLD_AFTER_IDS
        assert tuple(row["moved"]) == THRESHOLD_MOVED
        assert row["candidates"] == 5
        assert row["scored"] == 5
        assert row["top_n"] == settings.retrieval_rerank_top_n
        assert row["mode"] == "replace"
        assert row["weight"] == settings.retrieval_rerank_weight
        assert row["before"] == COMPARE_BEFORE
        assert row["after"] == COMPARE_AFTER

    def test_the_swap_comes_from_the_proximity_term(self, client: TestClient) -> None:
        """换位的两条重排分逐项手算：0.875 + 0.125 × 邻近度."""
        rows = {
            hit["record_id"]: hit
            for hit in rerank_call(client, query=THRESHOLD_QUERY, top_k=5)["rerank"]["hits"]
        }

        for record_id, score in THRESHOLD_MATCH_SCORES.items():
            assert rows[record_id]["score"] == score
            assert rows[record_id]["features"]["term_coverage"] == 1.0
            assert rows[record_id]["features"]["exact_phrase"] == 1.0
            assert rows[record_id]["score"] == pytest.approx(
                0.875 + 0.125 * rows[record_id]["features"]["proximity"]
            )

    def test_a_demoted_gold_standard_has_a_negative_lift(self, client: TestClient) -> None:
        """把金标准换成被挤下去的那一条 → **负提升**.

        这一条刻意留着：它说明这张表不是用来给重排背书的，而是用来回答
        "这一次它把谁的利益换给了谁"——只看正数的那种报告会掩盖另一半事实。
        """
        payload = compare_call(client, relevant=["h-m-01"])

        assert payload["lift"] == DEMOTED_LIFT
        assert payload["before"]["reciprocal_rank"] == 1.0
        assert payload["after"]["reciprocal_rank"] == 0.5

    def test_k_out_of_range_is_rejected(self, client: TestClient) -> None:
        for k in (0, -1):
            response = client.post(RERANK_COMPARE_PATH, json={"query": THRESHOLD_QUERY,
                                                             "relevant": ["h-m-02"], "k": k})

            assert response.status_code == 400
            assert "k 必须是 >= 1 的整数" in response.json()["detail"]

    def test_an_empty_gold_standard_is_rejected(self, client: TestClient) -> None:
        """空标注会让 recall@k 变成 0/0，而它算出来的 0.0 看起来像"一条都没召回到"."""
        response = client.post(
            RERANK_COMPARE_PATH, json={"query": THRESHOLD_QUERY, "relevant": [], "k": 5}
        )

        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "不能为空" in detail
        assert "分母" in detail

    def test_an_unknown_mode_is_rejected(self, client: TestClient) -> None:
        response = client.post(
            RERANK_COMPARE_PATH,
            json={"query": THRESHOLD_QUERY, "relevant": ["h-m-02"], "mode": "nope"},
        )

        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "未知重排模式" in detail
        assert "replace" in detail
        assert "blend" in detail

    def test_a_bad_window_is_rejected(self, client: TestClient) -> None:
        response = client.post(
            RERANK_COMPARE_PATH,
            json={"query": THRESHOLD_QUERY, "relevant": ["h-m-02"], "top_n": 0},
        )

        assert response.status_code == 400
        assert "top_n 必须是 >= 1 的整数" in response.json()["detail"]

    def test_an_empty_store_reports_zeroes_without_failing(self) -> None:
        """空库：200 且三条指标都是 0——"评了 1 条查询但一条都没召回到"要能被看见."""
        payload = compare_call(make_client(store=empty_store()))

        assert payload["before"] == {"recall": 0.0, "reciprocal_rank": 0.0, "ndcg": 0.0}
        assert payload["after"] == payload["before"]
        assert payload["lift"] == payload["before"]
        assert payload["queries"] == 1
        assert payload["per_query"][0]["before_ids"] == []
        assert payload["per_query"][0]["after_ids"] == []

    def test_defaults_and_scope_are_echoed(self, client: TestClient) -> None:
        payload = compare_call(client)

        assert payload["defaults"]["top_n"] == settings.retrieval_rerank_top_n
        assert payload["defaults"]["modes"] == list(RERANK_MODES)
        assert payload["limitations"] == list(RETRIEVAL_LIMITATIONS)
        assert payload["out_of_scope"] == list(RETRIEVAL_OUT_OF_SCOPE)
        assert "重排提升" in payload["summary"]
        assert json.dumps(payload)


# --------------------------------------------------------------------------- #
# 请求体校验：参数问题 → 400（消息里带合法取值）
# --------------------------------------------------------------------------- #


class TestRerankRequestValidation:
    """参数非法一律 400，且消息里带合法取值（与 day066/067 同一条通道）.

    **400 与 422 的分界**（与 ``test_hybrid_api.py`` / ``test_retrieval_api.py``
    完全同一口径）：**业务参数**写错（值越界、名字不在封闭清单里、文本为空）
    走检索层的 ``RerankError`` → 400，因为消息里带着"合法取值是什么"与出路；
    **请求体的形状/类型**写错（``where`` 给了列表、``top_n`` 给了字符串）
    在 pydantic 那一层就被拦下 → 422，业务代码根本看不到它。
    """

    @pytest.mark.parametrize("top_n", [0, -1])
    def test_bad_top_n_is_rejected(self, client: TestClient, top_n: int) -> None:
        response = client.post(RERANK_PATH, json={"query": QUERY_EXACT, "top_n": top_n})

        assert response.status_code == 400
        assert "top_n 必须是 >= 1 的整数" in response.json()["detail"]

    @pytest.mark.parametrize("weight", [1.5, -0.1, 2])
    def test_weight_out_of_range_is_rejected(self, client: TestClient, weight: float) -> None:
        response = client.post(RERANK_PATH, json={"query": QUERY_EXACT, "weight": weight})

        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "必须落在 [0, 1]" in detail

    @pytest.mark.parametrize("mode", ["nope", "RRF", ""])
    def test_unknown_mode_is_rejected(self, client: TestClient, mode: str) -> None:
        response = client.post(RERANK_PATH, json={"query": QUERY_EXACT, "mode": mode})

        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "未知重排模式" in detail
        assert "replace、blend" in detail

    @pytest.mark.parametrize("query", ["", "   "])
    def test_blank_query_is_rejected(self, client: TestClient, query: str) -> None:
        response = client.post(RERANK_PATH, json={"query": query})

        assert response.status_code == 400
        assert "空查询" in response.json()["detail"]

    def test_enabled_false_is_rejected_with_a_way_out(self, client: TestClient) -> None:
        """本端点**强制开启**重排：``enabled=False`` 是明确被拒的请求.

        消息里给三样东西：为什么不接受、关掉重排的**正确位置**（那个开关管的是
        检索链路上的缺省装配）、以及不被静默忽略的理由。
        """
        response = client.post(RERANK_PATH, json={"query": QUERY_EXACT, "enabled": False})

        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "强制开启" in detail
        assert "retrieval_rerank_enabled" in detail
        assert "RETRIEVAL_RERANK_ENABLED" in detail
        assert "/retrieval/search" in detail

    def test_enabled_true_behaves_like_omitting_it(self, client: TestClient) -> None:
        payload = rerank_call(client, enabled=True)

        assert payload["after_ids"] == list(AFTER_IDS)

    def test_a_bad_window_shape_is_422(self, client: TestClient) -> None:
        """字段类型不对是**契约**问题（422），不是检索参数问题（400）.

        注意这里给的是**一个列表**而不是 ``"3"``：pydantic 会把数字文本
        （``"3"``）按宽松模式转成整数，而"该是整数的地方给了一个列表"
        是它无论如何都接不住的写法。
        """
        response = client.post(RERANK_PATH, json={"query": QUERY_EXACT, "top_n": [3]})

        assert response.status_code == 422

    def test_a_bad_where_shape_is_422(self, client: TestClient) -> None:
        response = client.post(RERANK_PATH, json={"query": QUERY_EXACT, "where": ["topic"]})

        assert response.status_code == 422

    def test_rerank_and_search_share_the_retrieval_parameter_channel(
        self, client: TestClient
    ) -> None:
        """同一个坏参数在两条通道上得到同一个状态码（它们共用同一套校验）."""
        body = {"query": QUERY_EXACT, "top_k": 0}

        assert client.post(RERANK_PATH, json=body).status_code == 400
        assert client.post("/retrieval/search", json=body).status_code == 400

    def test_a_bad_where_syntax_is_rejected_with_the_supported_operators(
        self, client: TestClient
    ) -> None:
        response = client.post(RERANK_PATH, json={"query": QUERY_EXACT,
                                                  "where": {"topic": {"$bad": 1}}})

        assert response.status_code == 400
        assert "$bad" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# 注入与装配错误
# --------------------------------------------------------------------------- #


class TestInjection:
    """``app.state.retrieval_rerank`` 注入点的优先级、报错与实例隔离. """

    def test_the_injected_reranker_is_used(self) -> None:
        """注入的分数（长短）与教学替身（四个特征）毫无关系，因此次序就是证据."""
        client = make_client(reranker=ReverseLengthReranker())
        payload = rerank_call(client)
        expected = length_order(payload["before_ids"])

        assert payload["after_ids"] == expected
        assert payload["after_ids"] != list(AFTER_IDS)
        assert payload["rerank"]["model"] == "reverse-by-length"
        assert payload["rerank"]["hits"][0]["score"] == round(
            1.0 / (1.0 + len(RECORD_TEXTS[expected[0]])), 6
        )
        assert payload["rerank"]["hits"][0]["features"] == {}

    def test_the_injected_reranker_is_used_by_the_compare_endpoint(self) -> None:
        """对照端点也走同一个注入点（否则"接口量的那次重排"会是另一次重排）."""
        client = make_client(reranker=ReverseLengthReranker())
        payload = compare_call(client)
        expected = length_order(list(THRESHOLD_BEFORE_IDS))

        assert tuple(payload["per_query"][0]["after_ids"]) == tuple(expected)
        assert tuple(payload["per_query"][0]["after_ids"]) != THRESHOLD_AFTER_IDS

    def test_the_status_endpoint_reports_the_injected_reranker(self) -> None:
        client = make_client(reranker=ReverseLengthReranker())

        payload = status_call(client)

        assert payload["model"]["name"] == "reverse-by-length"
        assert payload["model"]["kind"] == "custom(test)"
        assert payload["summary"] == "重排器 'reverse-by-length'（1 维打分）"

    def test_a_wrong_injected_reranker_is_a_500(self) -> None:
        """装配写错 → 500（不是 400）：把它降级成 400 会让一个必然复现的配置错误
        看起来像"这次请求碰巧不对"。"""
        client = make_client(reranker="not-a-reranker")

        response = client.post(RERANK_PATH, json={"query": QUERY_EXACT})

        assert response.status_code == 500
        assert "BaseReranker" in response.json()["detail"]

    def test_a_wrong_injected_reranker_is_a_500_on_the_read_endpoint(self) -> None:
        client = make_client(reranker="not-a-reranker")

        response = client.get(RERANK_STATUS_PATH)

        assert response.status_code == 500
        assert RETRIEVAL_RERANK_STATE_KEY in response.json()["detail"]

    def test_the_state_key_is_initialised_and_empty_per_app_instance(self) -> None:
        """状态挂在**应用实例**上，而且缺省是 ``None``.

        ``None`` 的含义是"没注入"（于是每次请求现装一个替身），**不是**"禁用重排"：
        同一个进程里另建一个 app，那个位置仍然是空的——
        "注入只活在这个应用实例里"因此是一件可断言的事，而不是一句承诺。
        """
        first = make_client(reranker=ReverseLengthReranker())
        second = make_client()

        assert RETRIEVAL_RERANK_STATE_KEY == "retrieval_rerank"
        assert isinstance(getattr(first.app.state, RETRIEVAL_RERANK_STATE_KEY), BaseReranker)
        assert getattr(second.app.state, RETRIEVAL_RERANK_STATE_KEY) is None
        assert first.app is not second.app

    def test_injection_does_not_leak_into_the_search_channel(self) -> None:
        """注入的是这条通道的重排器：``/retrieval/search`` 关着重排时一个字段都不变."""
        client = make_client(reranker=ReverseLengthReranker())

        result = client.post("/retrieval/search", json={"query": QUERY_EXACT}).json()["result"]

        assert result["rerank"] == {}
        assert [hit["rerank_score"] for hit in result["hits"]] == [None] * len(result["hits"])


# --------------------------------------------------------------------------- #
# day066 / day067 的七个端点行为不变
# --------------------------------------------------------------------------- #


class TestExistingEndpointsUnchanged:
    """重排关着（缺省）时，search / hybrid / explain / answer 逐位保持 day067 口径."""

    def test_search_is_unchanged(self, client: TestClient) -> None:
        result = client.post("/retrieval/search", json={"query": QUERY_EXACT}).json()["result"]

        assert result["rerank"] == {}
        assert result["dropped_by_rerank"] == 0
        assert ids_of(result["hits"]) == ["h-c-01", "h-c-02", "h-c-03", "h-c-04", "h-m-01"]
        for hit in result["hits"]:
            assert hit["rerank_score"] is None
            assert hit["stage1_rank"] is None

    def test_hybrid_is_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "retrieval_hybrid_enabled", True)
        client = make_client()

        result = client.post("/retrieval/hybrid", json={"query": QUERY_EXACT}).json()["result"]

        assert ids_of(result["hits"]) == list(HYBRID_DEFAULT[QUERY_EXACT])
        assert result["rerank"] == {}
        assert result["dropped_by_rerank"] == 0
        assert [hit["rerank_score"] for hit in result["hits"]] == [None] * len(result["hits"])

    def test_explain_is_unchanged(self, client: TestClient) -> None:
        payload = client.post("/retrieval/explain", json={"query": QUERY_EXACT}).json()

        assert payload["ids"] == ["h-c-01", "h-c-02", "h-c-03", "h-c-04", "h-m-01"]
        assert payload["lines"]

    def test_answer_is_unchanged(self) -> None:
        """RAG 那条链路拿到的是同一份"没有重排"的结果（重排关着时逐位不变）."""
        llm = RecordingLLM()
        client = make_client(llm=llm)

        payload = client.post("/retrieval/answer", json={"question": QUERY_EXACT}).json()
        retrieval = payload["retrieval"]

        assert payload["llm_called"] is True
        assert llm.calls
        assert retrieval["rerank"] == {}
        assert retrieval["dropped_by_rerank"] == 0
        assert [hit["rerank_score"] for hit in retrieval["hits"]] == [None] * len(
            retrieval["hits"]
        )
        assert payload["citations"]

    def test_the_status_defaults_report_the_new_knobs(self, client: TestClient) -> None:
        """``/retrieval/status`` 的缺省值表也补上了重排这一组（与 day067 同一条纪律）."""
        defaults = client.get("/retrieval/status").json()["defaults"]

        assert defaults["rerank_enabled"] is False
        assert defaults["rerank_model"] == settings.retrieval_rerank_model
        assert defaults["rerank_top_n"] == settings.retrieval_rerank_top_n
        assert defaults["rerank_mode"] == settings.retrieval_rerank_mode
        assert defaults["rerank_weight"] == settings.retrieval_rerank_weight
        assert defaults["rerank_min_score"] is None

    def test_the_shared_reranker_keeps_the_three_verdicts_apart(self, client: TestClient) -> None:
        """同一份请求体在三条通道上的"重排账"互不串味.

        ```text
        /retrieval/search        空 rerank（链路关着）
        /retrieval/rerank        有 rerank（这条通道强制开启）
        /retrieval/rerank/status 不产生任何重排（纯读）
        ```
        """
        search = client.post("/retrieval/search", json=WIDE_BODY).json()["result"]
        forced = rerank_call(client)
        before_status = status_call(client)

        assert search["rerank"] == {}
        assert forced["rerank"]["scored"] == 8
        assert before_status["model"]["calls"] == 0
        assert json.dumps(forced)
