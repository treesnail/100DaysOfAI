"""day069 ``/retrieval/generate`` 系列三个端点的单元测试.

全部离线、确定性、零网络：样本来自 ``tests/hybrid_samples.py``（十条记录 +
``TableEmbedding`` 查表编码器），模型是下面那个只记账、不打网络的
``RecordingLLM``，因此"这次给了几条、引用了哪几条、覆盖率是多少"
全部是**能在纸上算出来的数**，不是"跑一遍看看"。

```text
查询 ERR-2043（缺省 top_k=5）
命中的五条（= 提示词里的编号表，名次即编号）
[1] h-c-01  [2] h-c-02  [3] h-c-03  [4] h-c-04  [5] h-m-01
模型答 "依据 [1]… 依据 [2]…"
→ cited=(1,2)  valid=(1,2)  invalid=()  unused=(3,4,5)  given=5  coverage=2/5=0.4
→ grounded=True（valid 非空且 invalid 为空）
```

除了"三个端点各自返回什么"，这里钉住七件比 200 更重要的事：

1. **生成参数只在请求体里换，不换模板本身**：``prompt_version`` 只接受受管版本名，
   清单外的取值（``v3`` / ``V2`` / 一段模板文本）一律 400 并列出合法取值；
2. **六个回退都不是错误**：``no_context`` 一次 LLM 都不调（护栏），
   ``llm_error`` / ``empty_reply`` / ``model_declined`` / ``unusable_citations``
   都**确实调用过**模型（``llm_called=True``）——这四个的判断全在
   ``fallback_reason`` 里，而不是靠状态码；
3. **``grounding`` 与 ``check`` 是同一份报告的整体与明细**（六个数字 + 逐编号的行），
   幻觉引用的那一行 ``record_id`` 是 ``None``；
4. **核对可以单独问一次，而且它一次模型都不调**：``/retrieval/grounding/verify``
   连着三个用例——注入了会计数的假 LLM 时 ``calls`` 恒为 0、连
   ``app.state.llm`` 都为空时它照样 200；
5. **提示词的元信息是算出来的，不是抄来的**：v1 四条回答要求、v2 六条约束
   （``## 输出格式`` 那三条不计入），字符数与 ``RAG_PROMPTS`` 逐个对得上；
6. **回退原因表覆盖全部六个取值**（含表示"没有回退"的空串）：这份表的用途是
   按原因分组统计，少一个取值就意味着有一批结果无法归组；
7. **day066~068 的十条端点逐位不变，只多了两个键**：``/retrieval/answer`` 的
   ``check`` / ``fallback_reason`` 是**新增**的，旧键的值一个都没改
   （``test_answer_keeps_its_values_and_gains_two_keys`` 把键集合也钉住）。
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from smart_research_agent.api.app import create_app
from smart_research_agent.api.routes import (
    RETRIEVAL_LLM_STATE_KEY,
    prompt_constraints,
    prompt_placeholders,
)
from smart_research_agent.config import settings
from smart_research_agent.llm.base import BaseLLM, Message
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.retrieval import (
    CURRENT_PROMPT_VERSION,
    DECLINE_MARKERS,
    DEFAULT_PROMPT_VERSION,
    FALLBACK_NO_CONTEXT,
    FALLBACK_REASON_DESCRIPTIONS,
    FALLBACK_REASONS,
    PROMPT_CHANGELOG,
    PROMPT_VERSIONS,
    RAG_PROMPTS,
    RETRIEVAL_LIMITATIONS,
    RETRIEVAL_OUT_OF_SCOPE,
)
from tests.hybrid_samples import (
    HYBRID_DEFAULT,
    QUERY_EXACT,
    empty_store,
    hybrid_embedding,
    hybrid_store,
)

#: 三个新端点的路径（断言"检索端点从 10 个变 13 个"时用）.
GENERATE_PATH = "/retrieval/generate"
PROMPT_VERSIONS_PATH = "/retrieval/prompt/versions"
GROUNDING_VERIFY_PATH = "/retrieval/grounding/verify"

#: 主案例的五条命中（缺省 top_k=5；编号 = 名次 + 1）。
#: 这一份次序与 ``/retrieval/search`` 的缺省结果逐位相同（day066 口径不变）。
HIT_IDS: tuple[str, ...] = ("h-c-01", "h-c-02", "h-c-03", "h-c-04", "h-m-01")

#: 主案例的六个核对数字（手算依据见模块 docstring）。
EXPECTED_GIVEN = 5
EXPECTED_CITED: tuple[int, ...] = (1, 2)
EXPECTED_VALID: tuple[int, ...] = (1, 2)
EXPECTED_UNUSED: tuple[int, ...] = (3, 4, 5)
EXPECTED_COVERAGE = 0.4

#: 幻觉引用的答案：``[9]`` 不在那次提示词里（只给了 [1]~[5]）。
HALLUCINATED_REPLY = "依据 [9]，结论成立。"
HALLUCINATED_MARKER = 9

#: 一句**没有引用**的答案：它不是幻觉引用，而是"没有可核对的东西"。
CITATION_FREE_REPLY = "阈值用分位数标定。"

#: 命中 v2 固定拒答句式的答案（句式来自 ``DECLINE_MARKERS``）。
DECLINED_REPLY = "资料未提及该编号的具体含义。"

#: 空库时那条护栏的期望值（``generation.FALLBACK_NO_CONTEXT`` 的原文，字面值不变）。
NO_CONTEXT_REASON = "no_context"

#: ``/retrieval/answer`` 的键集合：旧十个 + day069 新增的两个（**只增不改**）。
ANSWER_KEYS: frozenset[str] = frozenset(
    {
        "summary",
        "answer",
        "llm_called",
        "prompt_version",
        "citations",
        "context",
        "retrieval",
        "notes",
        "empty_reason",
        "route",
        "check",
        "fallback_reason",
    }
)

#: 离线核对用的编号表（三个编号，少而够用：一条命中、一条未引用、一条幻觉）。
NUMBERING: list[dict[str, Any]] = [
    {"marker": 1, "record_id": "doc-1"},
    {"marker": 2, "record_id": "doc-2"},
    {"marker": 3, "record_id": "doc-3"},
]

#: 全部合法的答案（引用 [1] 与 [3]；[2] 是那条"给了没引用"的）。
ALL_VALID_ANSWER = "依据 [1]，结论。\n依据 [3]，补充。"

#: 有幻觉引用的答案（引用 [1] [2] [9]；[9] 不在编号表里）。
HALLUCINATED_ANSWER = "依据 [1] 与 [9]，结论成立。\n依据 [2] 补充。"

#: 带着一个非编号的 [0] 的答案（它既不算引用、也不算幻觉引用）。
NOISE_ANSWER = "见 [0] 与 [2]，两处说法不同。"


class RecordingLLM(BaseLLM):
    """离线假模型：记下整轮消息与两个采样参数，返回预设回复.

    为什么不用 ``MockLLM``：这两条链路都**显式拒绝**它（它是未配置密钥时的
    占位实现）。它在这里承担三件事：证明护栏不误伤、留下"模型这次到底看到了什么"
    的证据（``calls``）、以及钉住"请求体里的 temperature / max_tokens 真的
    传给了模型"（``samplings``）。
    """

    #: 默认答案。带 ``[1]`` ``[2]`` 是刻意的：核对要拿它当输入。
    REPLY = "依据 [1]，阈值用分位数标定。\n依据 [2]，换编码器必须重新标定。"

    def __init__(self, reply: str = REPLY) -> None:
        self._reply = reply
        self.calls: list[list[Message]] = []
        self.samplings: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """记下整轮消息与两个采样参数并返回预设回复（不打网络、不读盘）."""
        self.calls.append(list(messages))
        self.samplings.append({"temperature": temperature, "max_tokens": max_tokens})
        return self._reply

    @property
    def prompt_text(self) -> str:
        """最近一次调用收到的提示词（没有调用过时是空串）."""
        if not self.calls:
            return ""
        return "\n".join(message.content for message in self.calls[-1])


class ExplodingLLM(BaseLLM):
    """一旦被调用就抛异常：用来走通 ``llm_error`` 那条路（超时 / 鉴权的替身）.

    它同时是"空检索时一次都不调"的判据（护栏失效时用例会以异常失败，
    而不是悄悄多调一次模型——与 day066 的用法逐字相同）。
    """

    def __init__(self) -> None:
        self.calls = 0

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """任何一次调用都记一笔并抛出（调用方把它折成 ``llm_error``）."""
        self.calls += 1
        raise RuntimeError("超时：这是一次可预期的失败，不是一个 bug")


def make_client(
    *,
    store: Any = None,
    llm: BaseLLM | None = None,
    retrieval_llm: BaseLLM | None = None,
) -> TestClient:
    """建一个可注入的离线客户端（两个注入点各自对应一类用例）.

    ``retrieval_llm`` 直接挂到 ``app.state`` 上（键名见
    ``routes.RETRIEVAL_LLM_STATE_KEY``）：它就是 ``/retrieval/generate`` 与
    ``/retrieval/answer`` 用的那个模型（``retrieval_llm`` 里"注入优先"那条规则）。
    """
    app = create_app(
        embedding=hybrid_embedding(),
        vector_store=store if store is not None else hybrid_store(),
        llm=llm,
    )
    if retrieval_llm is not None:
        setattr(app.state, RETRIEVAL_LLM_STATE_KEY, retrieval_llm)
    return TestClient(app)


def generate_call(client: TestClient, **body: Any) -> dict:
    """发一次生成请求（``query`` 缺省用 ``ERR-2043``）."""
    payload: dict = {"query": QUERY_EXACT}
    payload.update(body)
    response = client.post(GENERATE_PATH, json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def prompt_versions(client: TestClient) -> dict:
    """读一次受管提示词清单（``GET /retrieval/prompt/versions``）."""
    response = client.get(PROMPT_VERSIONS_PATH)
    assert response.status_code == 200, response.text
    return response.json()


def verify_call(client: TestClient, **body: Any) -> dict:
    """发一次离线核对（``answer`` 与 ``citations`` 缺省用主案例的两个常量）."""
    payload: dict = {"answer": ALL_VALID_ANSWER, "citations": NUMBERING}
    payload.update(body)
    response = client.post(GROUNDING_VERIFY_PATH, json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def answer_call(client: TestClient) -> dict:
    """发一次 ``/retrieval/answer``（``question`` 用 ``ERR-2043``）——口径不变的对照组."""
    response = client.post("/retrieval/answer", json={"question": QUERY_EXACT})
    assert response.status_code == 200, response.text
    return response.json()


def six_numbers(payload: dict) -> dict:
    """一份核对报告里的六个数字（``/retrieval/generate`` 与核对端点共用）."""
    return {
        "given": payload["given"],
        "cited": payload["cited"],
        "valid": payload["valid"],
        "invalid": payload["invalid"],
        "unused": payload["unused"],
        "coverage": payload["coverage"],
    }


@pytest.fixture(autouse=True)
def _restore_settings():
    """每个用例结束后把两个开关复位（``settings`` 是进程级单例，必须自己收拾）."""
    yield
    settings.retrieval_hybrid_enabled = False
    settings.retrieval_rerank_enabled = False


@pytest.fixture
def client() -> TestClient:
    """缺省客户端：十条样本 + 查表编码器 + 一个记账假模型."""
    return make_client(llm=RecordingLLM())


# --------------------------------------------------------------------------- #
# 端点清单
# --------------------------------------------------------------------------- #


class TestEndpointRegistry:
    """检索端点从 10 个变 13 个（这一条是"今天加了什么"的机器可核对的证据）. """

    def test_thirteen_retrieval_endpoints_are_registered(self) -> None:
        client = make_client()
        paths = [
            path
            for path in client.get("/openapi.json").json()["paths"]
            if path.startswith("/retrieval")
        ]

        assert len(paths) == 13
        assert GENERATE_PATH in paths
        assert PROMPT_VERSIONS_PATH in paths
        assert GROUNDING_VERIFY_PATH in paths

    def test_the_three_new_paths_keep_their_methods(self) -> None:
        schema = make_client().get("/openapi.json").json()["paths"]

        assert set(schema[GENERATE_PATH]) == {"post"}
        assert set(schema[PROMPT_VERSIONS_PATH]) == {"get"}
        assert set(schema[GROUNDING_VERIFY_PATH]) == {"post"}

    def test_the_request_models_are_exported_to_openapi(self) -> None:
        """五个新模型都进了契约文档（"接口长什么样"只在这里定义一次）."""
        schemas = make_client().get("/openapi.json").json()["components"]["schemas"]

        assert {
            "GenerateRequest",
            "GenerateResponse",
            "PromptVersionsResponse",
            "GroundingVerifyCitation",
            "GroundingVerifyRequest",
            "GroundingVerifyResponse",
        } <= set(schemas)

    def test_the_generate_request_extends_the_retrieval_one(self) -> None:
        """八个检索字段是继承来的（含 ``query``），四个生成字段是加上的.

        同时钉住"**没有**模板文本字段"：提示词只能用版本号说——它一旦能被
        请求方改写，``prompt_version`` 就不再指着一份确定的模板。
        """
        schemas = make_client().get("/openapi.json").json()["components"]["schemas"]
        properties = schemas["GenerateRequest"]["properties"]

        assert {
            "query",
            "top_k",
            "fetch_k",
            "where",
            "time_range",
            "min_score",
            "max_per_doc",
            "route",
        } <= set(properties)
        assert {"prompt_version", "temperature", "max_tokens", "require_citation"} <= set(
            properties
        )
        assert "prompt" not in properties
        assert "template" not in properties

    def test_the_grounding_citation_model_mirrors_the_domain_shape(self) -> None:
        """编号表那一行的五个字段与 ``context.Citation`` 逐项对应（三个可省略）."""
        schemas = make_client().get("/openapi.json").json()["components"]["schemas"]
        properties = schemas["GroundingVerifyCitation"]["properties"]

        assert set(properties) == {"marker", "record_id", "source", "heading_path", "score"}
        assert set(schemas["GroundingVerifyCitation"]["required"]) == {"marker", "record_id"}


# --------------------------------------------------------------------------- #
# GET /retrieval/prompt/versions
# --------------------------------------------------------------------------- #


class TestPromptVersionsEndpoint:
    """受管提示词的清单与元信息：纯读端点，机器可核对的受管清单. """

    def test_the_closed_list_of_versions(self, client: TestClient) -> None:
        payload = prompt_versions(client)

        assert payload["versions"] == ["v1", "v2"]
        assert payload["versions"] == list(PROMPT_VERSIONS)
        assert payload["current"] == CURRENT_PROMPT_VERSION
        assert payload["current"] == "v2"
        assert payload["default"] == DEFAULT_PROMPT_VERSION
        assert payload["default"] == settings.retrieval_prompt_version

    def test_the_summary_counts_what_the_body_gives(self, client: TestClient) -> None:
        payload = prompt_versions(client)

        assert f"受管提示词 {len(PROMPT_VERSIONS)} 版" in payload["summary"]
        assert f"当前 {CURRENT_PROMPT_VERSION}" in payload["summary"]
        assert f"固定拒答句式 {len(DECLINE_MARKERS)} 条" in payload["summary"]
        assert f"回退原因 {len(FALLBACK_REASONS)} 种" in payload["summary"]

    def test_each_version_keeps_its_five_metadata_columns(self, client: TestClient) -> None:
        rows = {row["version"]: row for row in prompt_versions(client)["templates"]}

        assert list(rows) == list(PROMPT_VERSIONS)
        for version in PROMPT_VERSIONS:
            row = rows[version]
            assert set(row) == {"version", "chars", "changelog", "placeholders", "constraints"}
            assert row["chars"] == len(RAG_PROMPTS[version])
            assert row["changelog"] == PROMPT_CHANGELOG[version]
            assert row["changelog"].strip()
            # 两个占位符都必须在：缺一个就是一次"没有资料的提问"。
            assert row["placeholders"] == ["context", "question"]

    def test_the_constraint_counts_are_hand_computable(self, client: TestClient) -> None:
        """v1 四条回答要求、v2 六条约束（``## 输出格式`` 那三条**不计入**）.

        口径写在 ``prompt_constraints`` 的 docstring 里：数到下一个二级标题为止。
        把它改成"数全文所有编号条目"的话，v2 会变成 9——而 9 回答不了
        "v2 比 v1 多挡了哪两种失败"这个问题。
        """
        rows = {row["version"]: row for row in prompt_versions(client)["templates"]}

        assert rows["v1"]["constraints"] == 4
        assert rows["v2"]["constraints"] == 6
        # v2 是四段式重写，字符数也更长（它是"更贵"的那一版）。
        assert rows["v2"]["chars"] > rows["v1"]["chars"]

    def test_the_fallback_table_covers_every_value(self, client: TestClient) -> None:
        """六个取值一个不少（含表示"没有回退"的空串），解释直接来自那一份字典."""
        reasons = [row["reason"] for row in prompt_versions(client)["fallback_reasons"]]

        assert reasons == ["", *FALLBACK_REASONS]
        assert set(reasons) == set(FALLBACK_REASON_DESCRIPTIONS)
        for row in prompt_versions(client)["fallback_reasons"]:
            assert row["description"] == FALLBACK_REASON_DESCRIPTIONS[row["reason"]]
            assert row["description"].strip()

    def test_the_decline_markers_are_the_three_sentences(self, client: TestClient) -> None:
        """检测清单与 v2 的约束 2 是同一份：三句话都要能在模板里读出来."""
        payload = prompt_versions(client)

        assert payload["decline_markers"] == [
            "资料中没有相关内容",
            "资料未提及",
            "无法依据资料回答",
        ]
        assert payload["decline_markers"] == list(DECLINE_MARKERS)
        for marker in DECLINE_MARKERS:
            assert marker in RAG_PROMPTS["v2"]
            assert marker in RAG_PROMPTS["v2"][RAG_PROMPTS["v2"].index("## 约束") :]

    def test_the_constraint_rule_is_the_one_the_endpoint_uses(self) -> None:
        """端点的两个数字来自 ``prompt_constraints``（同一个函数，不是抄来的常量）.

        顺带钉住它的边界：**没有约束小节的模板数出来是 0**——而不是把整份模板里的
        编号条目都算进去（那样 v2 会从 6 变成 9，而 9 回答不了"多了哪两种失败"）。
        """
        assert prompt_constraints(RAG_PROMPTS["v1"]) == 4
        assert prompt_constraints(RAG_PROMPTS["v2"]) == 6
        assert prompt_constraints("一段没有约束小节的模板。\n1. 只有一条编号条目。") == 0

    def test_the_placeholder_rule_reads_the_format_string(self) -> None:
        """转义的 ``{{context}}`` 不算占位符（"在字符串里搜"的写法会把它算进去）."""
        assert prompt_placeholders("{{context}} 与 {context} 与 {question}") == [
            "context",
            "question",
        ]

    def test_defaults_echo_the_settings(self, client: TestClient) -> None:
        defaults = prompt_versions(client)["defaults"]

        assert defaults["prompt_version"] == settings.retrieval_prompt_version
        assert defaults["answer_temperature"] == settings.retrieval_answer_temperature
        assert defaults["answer_max_tokens"] == settings.retrieval_answer_max_tokens
        assert defaults["require_citation"] == settings.retrieval_require_citation
        assert defaults["prompt_versions"] == list(PROMPT_VERSIONS)
        assert defaults["decline_markers"] == list(DECLINE_MARKERS)

    def test_limitations_and_out_of_scope_are_echoed(self, client: TestClient) -> None:
        payload = prompt_versions(client)

        assert payload["limitations"] == list(RETRIEVAL_LIMITATIONS)
        assert payload["out_of_scope"] == list(RETRIEVAL_OUT_OF_SCOPE)

    def test_the_endpoint_is_read_only(self, client: TestClient) -> None:
        """两次读给出同一份答案（它不检索、不生成、不写任何状态）."""
        assert prompt_versions(client) == prompt_versions(client)

    def test_body_is_json_serializable(self, client: TestClient) -> None:
        assert json.dumps(prompt_versions(client))


# --------------------------------------------------------------------------- #
# POST /retrieval/generate
# --------------------------------------------------------------------------- #


class TestGenerateEndpoint:
    """一次生成的完整交代：答案 / 问法 / 核对三块都用手算常量核对. """

    def test_the_happy_path_reports_the_whole_book_keeping(self, client: TestClient) -> None:
        llm = RecordingLLM()
        payload = generate_call(make_client(llm=llm))

        assert payload["llm_called"] is True
        assert payload["answer"] == RecordingLLM.REPLY
        assert payload["prompt_version"] == CURRENT_PROMPT_VERSION
        assert payload["fallback_reason"] == ""
        assert payload["fallback_reason_description"] == FALLBACK_REASON_DESCRIPTIONS[""]
        assert payload["empty_reason"] == "hits"
        assert payload["notes"] == []  # 正常路径不占注记（day066 那条纪律仍在）
        assert len(llm.calls) == 1

    def test_the_prompt_text_carries_the_numbered_context_and_the_question(
        self, client: TestClient
    ) -> None:
        """``prompt_text`` 是"模型看到了什么"的唯一证据：编号、问题、四段标题都在."""
        prompt = generate_call(client)["prompt_text"]

        assert "[1]" in prompt and "[5]" in prompt
        assert QUERY_EXACT in prompt
        assert "## 资料片段" in prompt
        assert "## 问题" in prompt
        assert "## 约束" in prompt
        assert "## 输出格式" in prompt

    def test_the_six_grounding_numbers_are_hand_computable(self, client: TestClient) -> None:
        grounding = generate_call(client)["grounding"]

        assert six_numbers(grounding) == {
            "given": EXPECTED_GIVEN,
            "cited": list(EXPECTED_CITED),
            "valid": list(EXPECTED_VALID),
            "invalid": [],
            "unused": list(EXPECTED_UNUSED),
            "coverage": EXPECTED_COVERAGE,
        }
        assert grounding["grounded"] is True

    def test_the_citations_are_the_numbering_table_of_that_prompt(
        self, client: TestClient
    ) -> None:
        """编号表与那次提示词一一对应：第 1 号就是那次检索的第一名."""
        payload = generate_call(client)

        assert [citation["marker"] for citation in payload["citations"]] == [1, 2, 3, 4, 5]
        assert [citation["record_id"] for citation in payload["citations"]] == list(HIT_IDS)
        # check 是同一份报告的明细（不是另一份）：给得少一条就会不一致。
        assert payload["check"] == payload["grounding"]["checks"]
        assert [row["marker"] for row in payload["check"]] == [1, 2, 3, 4, 5]

    def test_a_hallucinated_marker_makes_grounded_false(self, client: TestClient) -> None:
        """引用了不存在的编号：[9] 进了 invalid，于是接地不通过——但**没有回退**.

        ``require_citation`` 缺省是关的，因此这份"不可核对"的答案仍然交付：
        ``grounded=False`` 是报告里的一个事实，不是一道闸门（闸门是那条开关）。
        """
        payload = generate_call(make_client(llm=RecordingLLM(reply=HALLUCINATED_REPLY)))
        grounding = payload["grounding"]

        assert grounding["cited"] == [HALLUCINATED_MARKER]
        assert grounding["valid"] == []
        assert grounding["invalid"] == [HALLUCINATED_MARKER]
        assert grounding["unused"] == [1, 2, 3, 4, 5]
        assert grounding["coverage"] == 0.0
        assert grounding["grounded"] is False
        assert payload["check"][-1] == {
            "marker": HALLUCINATED_MARKER,
            "record_id": None,
            "hallucinated": True,
            "used": True,
        }
        assert payload["fallback_reason"] == ""
        assert payload["answer"] == HALLUCINATED_REPLY

    def test_an_answer_without_any_marker_is_not_a_hallucination(
        self, client: TestClient
    ) -> None:
        """一个 ``[n]`` 都没有：``invalid`` 仍为空（它是"没有引用"，不是"引用错了"）."""
        payload = generate_call(make_client(llm=RecordingLLM(reply=CITATION_FREE_REPLY)))
        grounding = payload["grounding"]

        assert grounding["cited"] == []
        assert grounding["invalid"] == []
        assert grounding["unused"] == [1, 2, 3, 4, 5]
        assert grounding["grounded"] is False
        assert payload["fallback_reason"] == ""

    def test_no_context_does_not_call_the_llm_at_all(self, client: TestClient) -> None:
        """空库：护栏生效——``llm_called=False``、``citations=[]``、答案取兜底答复."""
        llm = RecordingLLM()
        payload = generate_call(make_client(store=empty_store(), llm=llm))

        assert llm.calls == []
        assert payload["llm_called"] is False
        assert payload["citations"] == []
        assert payload["prompt_text"] == ""
        assert payload["answer"] == FALLBACK_NO_CONTEXT
        assert payload["fallback_reason"] == NO_CONTEXT_REASON
        assert payload["fallback_reason_description"] == FALLBACK_REASON_DESCRIPTIONS[
            NO_CONTEXT_REASON
        ]
        assert payload["empty_reason"] == "no_data"
        # 检索为空时 context 是一份**空包**（不是 None）：本链路照样走完了打包。
        assert payload["context"]["count"] == 0
        assert payload["context"]["text"] == ""
        assert payload["grounding"]["given"] == 0
        assert any("一个 LLM 都没有调" in note for note in payload["notes"])

    def test_a_filter_that_matches_nothing_hits_the_same_guardrail(
        self, client: TestClient
    ) -> None:
        llm = RecordingLLM()
        payload = generate_call(make_client(llm=llm), where={"strategy": "nonexistent"})

        assert llm.calls == []
        assert payload["llm_called"] is False
        assert payload["fallback_reason"] == NO_CONTEXT_REASON
        assert payload["empty_reason"] == "filtered_out"

    def test_an_llm_error_is_a_fallback_not_an_error(self, client: TestClient) -> None:
        """调用抛异常 → 200 且原因写在 ``fallback_reason`` 里（这次**确实调过**）."""
        llm = ExplodingLLM()
        payload = generate_call(make_client(llm=llm))

        assert llm.calls == 1
        assert payload["fallback_reason"] == "llm_error"
        assert payload["llm_called"] is True
        assert payload["answer"] == FALLBACK_NO_CONTEXT
        assert any("RuntimeError" in note for note in payload["notes"])

    def test_an_empty_reply_keeps_the_empty_answer(self, client: TestClient) -> None:
        """空回复不是"资料里没有相关内容"：``answer`` 就是那段空串."""
        payload = generate_call(make_client(llm=RecordingLLM(reply="   ")))

        assert payload["fallback_reason"] == "empty_reply"
        assert payload["answer"] == "   "
        assert payload["llm_called"] is True

    def test_a_declined_answer_names_the_matched_sentence(self, client: TestClient) -> None:
        """命中固定句式 → ``model_declined``，注记里写着命中的是哪一句."""
        payload = generate_call(make_client(llm=RecordingLLM(reply=DECLINED_REPLY)))

        assert payload["fallback_reason"] == "model_declined"
        assert payload["llm_called"] is True
        assert payload["answer"] == DECLINED_REPLY
        assert any("资料未提及" in note for note in payload["notes"])

    def test_require_citation_turns_missing_citations_into_a_fallback(
        self, client: TestClient
    ) -> None:
        """开闸门之后，一条合法引用都没有的答案会被整条丢掉（答案取兜底答复）."""
        payload = generate_call(
            make_client(llm=RecordingLLM(reply=CITATION_FREE_REPLY)), require_citation=True
        )

        assert payload["fallback_reason"] == "unusable_citations"
        assert payload["answer"] == FALLBACK_NO_CONTEXT
        assert payload["llm_called"] is True
        assert payload["grounding"]["valid"] == []

    def test_require_citation_false_leaves_the_same_answer_alone(
        self, client: TestClient
    ) -> None:
        """同一个答案在开关关着时照常交付（区别只在 ``fallback_reason``）."""
        payload = generate_call(make_client(llm=RecordingLLM(reply=CITATION_FREE_REPLY)))

        assert payload["fallback_reason"] == ""
        assert payload["answer"] == CITATION_FREE_REPLY

    def test_the_two_sampling_parameters_reach_the_model(self, client: TestClient) -> None:
        llm = RecordingLLM()
        generate_call(make_client(llm=llm), temperature=0.3, max_tokens=64)

        assert llm.samplings == [{"temperature": 0.3, "max_tokens": 64}]

    def test_omitting_them_uses_the_settings(self, client: TestClient) -> None:
        """``None`` = 取 settings（"项目默认只有一处定义"在这一点上可见）."""
        llm = RecordingLLM()
        generate_call(make_client(llm=llm))

        assert llm.samplings == [
            {
                "temperature": settings.retrieval_answer_temperature,
                "max_tokens": settings.retrieval_answer_max_tokens,
            }
        ]

    def test_the_prompt_version_can_be_switched_per_request(self, client: TestClient) -> None:
        """``v1`` 与 ``v2`` 是两段不同的模板：标题、约束条数与长度都能读出来."""
        payload = generate_call(client, prompt_version="v1")
        current = generate_call(client)

        assert payload["prompt_version"] == "v1"
        assert current["prompt_version"] == CURRENT_PROMPT_VERSION
        assert "回答要求：" in payload["prompt_text"]
        assert "## 约束" not in payload["prompt_text"]
        assert "## 约束" in current["prompt_text"]
        # v1 是"资料与要求混在一段里"的那一版：同一份上下文下它更短。
        assert len(payload["prompt_text"]) < len(current["prompt_text"])
        # 两个版本问的是同一个问题：资料片段与问题都照旧进提示词。
        assert QUERY_EXACT in payload["prompt_text"]
        assert "[1]" in payload["prompt_text"]

    def test_the_retrieval_parameters_are_honoured(self, client: TestClient) -> None:
        """继承来的检索字段照样生效：只取两条时，编号表与分母一起缩小."""
        payload = generate_call(client, top_k=2, fetch_k=2)

        assert payload["retrieval"]["count"] == 2
        assert [citation["record_id"] for citation in payload["citations"]] == ["h-c-01", "h-c-02"]
        assert payload["grounding"]["given"] == 2
        assert payload["grounding"]["valid"] == [1, 2]
        assert payload["grounding"]["coverage"] == 1.0

    def test_where_is_honoured(self, client: TestClient) -> None:
        payload = generate_call(client, where={"parent_doc_id": "doc-concept"})

        assert [hit["record_id"] for hit in payload["retrieval"]["hits"]] == [
            "h-c-01",
            "h-c-02",
            "h-c-03",
            "h-c-04",
        ]

    def test_the_route_decision_is_echoed(self, client: TestClient) -> None:
        payload = generate_call(client)

        assert payload["route"]["name"] == "default"
        assert payload["retrieval"]["query"]["route"] is None

    def test_defaults_echo_the_settings(self, client: TestClient) -> None:
        defaults = generate_call(client)["defaults"]

        assert defaults["prompt_version"] == settings.retrieval_prompt_version
        assert defaults["answer_temperature"] == settings.retrieval_answer_temperature
        assert defaults["answer_max_tokens"] == settings.retrieval_answer_max_tokens
        assert defaults["require_citation"] == settings.retrieval_require_citation
        assert defaults["prompt_versions"] == list(PROMPT_VERSIONS)
        assert defaults["decline_markers"] == list(DECLINE_MARKERS)

    def test_limitations_and_out_of_scope_are_echoed(self, client: TestClient) -> None:
        payload = generate_call(client)

        assert payload["limitations"] == list(RETRIEVAL_LIMITATIONS)
        assert payload["out_of_scope"] == list(RETRIEVAL_OUT_OF_SCOPE)

    def test_every_response_field_is_json_serializable(self, client: TestClient) -> None:
        payload = generate_call(client)
        rendered = json.dumps(payload, ensure_ascii=False)

        assert json.loads(rendered) == payload
        assert QUERY_EXACT in rendered

    def test_the_summary_line_carries_the_verdict_and_the_coverage(
        self, client: TestClient
    ) -> None:
        summary = generate_call(client)["summary"]

        assert "提示词 v2" in summary
        assert "接地通过" in summary
        assert "覆盖 40.0%" in summary


# --------------------------------------------------------------------------- #
# 请求体校验：业务参数 → 400，形状/类型 → 422
# --------------------------------------------------------------------------- #


class TestGenerateRequestValidation:
    """参数非法一律 400，且消息里带合法取值（与 day066~068 同一条通道）.

    **400 与 422 的分界**（与 ``test_rerank_api.py`` / ``test_retrieval_api.py``
    完全同一口径）：**业务参数**写错（版本名不在封闭清单里、温度越界、
    ``max_tokens < 1``、文本为空）由检索层给 400，因为消息里带着"合法取值是什么"
    与出路；**请求体的形状/类型**写错（``temperature`` 给字符串、``max_tokens``
    给列表）在 pydantic 那一层就被拦下 → 422，业务代码根本看不到它。
    """

    @pytest.mark.parametrize("version", ["v3", "V2", "v2.0", "V1"])
    def test_an_unknown_prompt_version_is_rejected(
        self, client: TestClient, version: str
    ) -> None:
        response = client.post(
            GENERATE_PATH, json={"query": QUERY_EXACT, "prompt_version": version}
        )

        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "受管清单" in detail
        assert "['v1', 'v2']" in detail
        assert "custom" in detail

    @pytest.mark.parametrize("version", ["", "   "])
    def test_a_blank_prompt_version_is_rejected(self, client: TestClient, version: str) -> None:
        response = client.post(
            GENERATE_PATH, json={"query": QUERY_EXACT, "prompt_version": version}
        )

        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "必须是非空字符串" in detail
        assert "分组键" in detail

    def test_a_template_text_is_not_a_version(self, client: TestClient) -> None:
        """把模板原文当成版本号传进来 → 400（提示词**只能**用版本号说）."""
        response = client.post(
            GENERATE_PATH, json={"query": QUERY_EXACT, "prompt_version": RAG_PROMPTS["v2"]}
        )

        assert response.status_code == 400
        assert "受管清单" in response.json()["detail"]

    def test_the_custom_sentinel_is_not_a_request_level_value(self, client: TestClient) -> None:
        """``custom`` 是**装配期**的标记：请求体里给它会 400（没有那样一份模板）."""
        response = client.post(
            GENERATE_PATH, json={"query": QUERY_EXACT, "prompt_version": "custom"}
        )

        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "未知的提示词版本" in detail
        assert "custom" in detail

    @pytest.mark.parametrize("temperature", [2.5, -0.1, 3])
    def test_temperature_out_of_range_is_rejected(
        self, client: TestClient, temperature: float
    ) -> None:
        response = client.post(
            GENERATE_PATH, json={"query": QUERY_EXACT, "temperature": temperature}
        )

        assert response.status_code == 400
        assert "超出 [0, 2]" in response.json()["detail"]

    @pytest.mark.parametrize("max_tokens", [0, -1])
    def test_max_tokens_below_one_is_rejected(self, client: TestClient, max_tokens: int) -> None:
        response = client.post(
            GENERATE_PATH, json={"query": QUERY_EXACT, "max_tokens": max_tokens}
        )

        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "没有空间说话" in detail
        assert "空回复" in detail

    @pytest.mark.parametrize("query", ["", "   "])
    def test_a_blank_query_is_rejected(self, client: TestClient, query: str) -> None:
        response = client.post(GENERATE_PATH, json={"query": query})

        assert response.status_code == 400
        assert "空查询" in response.json()["detail"]

    def test_the_text_field_is_query_not_question(self, client: TestClient) -> None:
        """文本字段叫 ``query``（继承自 ``RetrievalSearchRequest``）.

        把文本写在 ``question`` 里**不会报错**——pydantic 忽略未知字段，
        于是这次请求退化成一次空查询（400）。这与 ``/retrieval/answer``
        的 ``question`` 是两个名字：那一条链路的请求体不继承检索那一个，
        因此它必须自己命名（见 schemas 的两处字段说明）。
        """
        response = client.post(GENERATE_PATH, json={"question": QUERY_EXACT})

        assert response.status_code == 400
        assert "空查询" in response.json()["detail"]

    def test_a_bad_temperature_shape_is_422(self, client: TestClient) -> None:
        response = client.post(
            GENERATE_PATH, json={"query": QUERY_EXACT, "temperature": "low"}
        )

        assert response.status_code == 422

    def test_a_bad_max_tokens_shape_is_422(self, client: TestClient) -> None:
        response = client.post(GENERATE_PATH, json={"query": QUERY_EXACT, "max_tokens": [3]})

        assert response.status_code == 422

    def test_a_bad_require_citation_shape_is_422(self, client: TestClient) -> None:
        response = client.post(
            GENERATE_PATH, json={"query": QUERY_EXACT, "require_citation": "maybe"}
        )

        assert response.status_code == 422

    def test_a_bad_retrieval_parameter_shares_the_channel(self, client: TestClient) -> None:
        """同一个坏参数在两条通道上得到同一个状态码（同一套校验、同一个 400）."""
        body = {"query": QUERY_EXACT, "top_k": 0}

        assert client.post(GENERATE_PATH, json=body).status_code == 400
        assert client.post("/retrieval/search", json=body).status_code == 400

    def test_an_unknown_route_is_rejected(self, client: TestClient) -> None:
        """未知路由 → 400，消息里列出可用路由（与 ``/retrieval/search`` 同一条通道）."""
        response = client.post(
            GENERATE_PATH, json={"query": QUERY_EXACT, "route": "handbook"}
        )

        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "未知路由" in detail
        assert "可用路由：default" in detail


# --------------------------------------------------------------------------- #
# POST /retrieval/grounding/verify
# --------------------------------------------------------------------------- #


class TestGroundingVerifyEndpoint:
    """离线核对：六个数字手算，而且**一次模型都不调**. """

    def test_the_six_numbers_are_hand_computable(self, client: TestClient) -> None:
        payload = verify_call(client)

        assert six_numbers(payload) == {
            "given": 3,
            "cited": [1, 3],
            "valid": [1, 3],
            "invalid": [],
            "unused": [2],
            "coverage": 0.6667,
        }
        assert payload["grounded"] is True
        assert payload["hallucinated"] == 0

    def test_a_hallucinated_marker_is_named(self, client: TestClient) -> None:
        payload = verify_call(client, answer=HALLUCINATED_ANSWER)

        assert six_numbers(payload) == {
            "given": 3,
            "cited": [1, 2, 9],
            "valid": [1, 2],
            "invalid": [9],
            "unused": [3],
            "coverage": 0.6667,
        }
        assert payload["grounded"] is False
        assert payload["hallucinated"] == 1
        assert "幻觉 1" in payload["summary"]
        assert "接地未通过" in payload["summary"]

    def test_the_check_rows_are_the_whole_report(self, client: TestClient) -> None:
        """checks = 先 1..n 逐条，再按升序接上越界编号（幻觉那几行的 id 是 None）."""
        grounded = verify_call(client)
        hallucinated = verify_call(client, answer=HALLUCINATED_ANSWER)

        assert [row["marker"] for row in grounded["checks"]] == [1, 2, 3]
        assert [row["used"] for row in grounded["checks"]] == [True, False, True]
        assert [row["record_id"] for row in grounded["checks"]] == ["doc-1", "doc-2", "doc-3"]
        assert [row["marker"] for row in hallucinated["checks"]] == [1, 2, 3, 9]
        assert hallucinated["checks"][-1]["hallucinated"] is True
        assert hallucinated["checks"][-1]["record_id"] is None

    def test_an_answer_without_any_marker_is_not_a_hallucination(
        self, client: TestClient
    ) -> None:
        payload = verify_call(client, answer="阈值用分位数标定，换编码器必须重新标定。")

        assert payload["cited"] == []
        assert payload["invalid"] == []
        assert payload["unused"] == [1, 2, 3]
        assert payload["coverage"] == 0.0
        assert payload["grounded"] is False
        assert any("没有出现任何 [n]" in note for note in payload["notes"])

    def test_a_zero_marker_is_noise_not_a_citation(self, client: TestClient) -> None:
        """``[0]`` 不是编号：它既不进 cited/valid，也不算一次幻觉引用."""
        payload = verify_call(client, answer=NOISE_ANSWER)

        assert payload["cited"] == [2]
        assert payload["invalid"] == []
        assert payload["unused"] == [1, 3]
        assert payload["coverage"] == 0.3333
        assert any("[0]" in note for note in payload["notes"])

    def test_an_empty_numbering_table_is_a_legal_state(self, client: TestClient) -> None:
        payload = verify_call(client, citations=[])

        assert payload["given"] == 0
        assert payload["grounded"] is False
        assert payload["coverage"] == 0.0
        assert any("没有任何片段可核对" in note for note in payload["notes"])

    def test_the_endpoint_does_not_call_any_model(self, client: TestClient) -> None:
        """注入一个会计数的假 LLM：核对走完，它的 ``calls`` 仍然是 0."""
        llm = RecordingLLM()
        payload = verify_call(make_client(retrieval_llm=llm))

        assert llm.calls == []
        assert payload["grounded"] is True

    def test_it_works_even_when_the_app_has_no_llm_at_all(self) -> None:
        """连 ``app.state.llm`` 都为空也照样 200：这条通道不需要模型.

        这一条比"假 LLM 没被调用"更强：它证明核对**不依赖**任何模型对象，
        因此它是排查时的第一站（在还没有配好模型的环境里就能用）。
        """
        app = create_app(embedding=hybrid_embedding(), vector_store=hybrid_store(), llm=MockLLM())
        app.state.llm = None
        response = TestClient(app).post(
            GROUNDING_VERIFY_PATH, json={"answer": ALL_VALID_ANSWER, "citations": NUMBERING}
        )

        assert response.status_code == 200
        assert response.json()["grounded"] is True

    def test_a_marker_below_one_is_rejected(self, client: TestClient) -> None:
        response = client.post(
            GROUNDING_VERIFY_PATH,
            json={"answer": "见 [1]。", "citations": [{"marker": 0, "record_id": "doc-0"}]},
        )

        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "从 1 起" in detail
        assert "[0]" in detail

    def test_a_blank_record_id_is_rejected(self, client: TestClient) -> None:
        response = client.post(
            GROUNDING_VERIFY_PATH,
            json={"answer": "见 [1]。", "citations": [{"marker": 1, "record_id": "   "}]},
        )

        assert response.status_code == 400
        assert "没有 id 的引用" in response.json()["detail"]

    @pytest.mark.parametrize("body", [{}, {"answer": "见 [1]。"}, {"citations": NUMBERING}])
    def test_missing_fields_are_422(self, client: TestClient, body: dict) -> None:
        """两个字段都必填：离线核对没有"默认答案"可言（形状错 → 422）."""
        response = client.post(GROUNDING_VERIFY_PATH, json=body)

        assert response.status_code == 422

    def test_defaults_and_scope_are_echoed(self, client: TestClient) -> None:
        payload = verify_call(client)

        assert payload["defaults"]["prompt_version"] == settings.retrieval_prompt_version
        assert payload["limitations"] == list(RETRIEVAL_LIMITATIONS)
        assert payload["out_of_scope"] == list(RETRIEVAL_OUT_OF_SCOPE)
        assert "核对" in payload["summary"]
        assert json.dumps(payload)

    def test_it_reports_the_same_numbers_as_the_generating_channel(
        self, client: TestClient
    ) -> None:
        """同一份答案 + 同一份编号表：两条通道给出同一组数字.

        这一条是"核对只有一份实现"的行为证据：端点若自己再写一遍解析，
        两份结论迟早会对同一个输入给出不同的数（而那种差异没有报错指向它）。
        """
        generated = generate_call(client)["grounding"]
        numbering = [
            {"marker": citation["marker"], "record_id": citation["record_id"]}
            for citation in generate_call(client)["citations"]
        ]
        verified = verify_call(client, answer=RecordingLLM.REPLY, citations=numbering)

        assert six_numbers(verified) == six_numbers(generated)
        assert verified["grounded"] == generated["grounded"]
        assert verified["checks"] == generated["checks"]


# --------------------------------------------------------------------------- #
# 注入
# --------------------------------------------------------------------------- #


class TestInjection:
    """``app.state.retrieval_llm`` 注入点的优先级与实例隔离. """

    def test_the_injected_llm_is_used(self) -> None:
        override = RecordingLLM(reply="只用检索专用模型回答。")
        client = make_client(llm=MockLLM(), retrieval_llm=override)

        payload = generate_call(client)

        assert payload["answer"] == "只用检索专用模型回答。"
        assert len(override.calls) == 1

    def test_the_injected_llm_wins_over_the_app_llm(self) -> None:
        """两个都能用时，``retrieval_llm`` 优先（问答可以单独配一个模型）."""
        app_llm = RecordingLLM(reply="app.state.llm 给的答案。")
        retrieval_llm = RecordingLLM(reply="注入的检索专用模型给的答案。")
        client = make_client(llm=app_llm, retrieval_llm=retrieval_llm)

        payload = generate_call(client)

        assert payload["answer"] == "注入的检索专用模型给的答案。"
        assert len(retrieval_llm.calls) == 1
        assert app_llm.calls == []

    def test_a_mock_placeholder_is_rejected_on_both_channels(self) -> None:
        """同一个判据：MockLLM 占位实现在这两条链路上都被拦（400 + 同一句话）."""
        client = make_client(llm=MockLLM())

        generate = client.post(GENERATE_PATH, json={"query": QUERY_EXACT})
        answer = client.post("/retrieval/answer", json={"question": QUERY_EXACT})

        assert generate.status_code == 400
        assert answer.status_code == 400
        assert "MockLLM" in generate.json()["detail"]
        assert "MockLLM" in answer.json()["detail"]

    def test_a_missing_llm_is_rejected_with_a_way_out(self) -> None:
        app = create_app(embedding=hybrid_embedding(), vector_store=hybrid_store(), llm=MockLLM())
        app.state.llm = None
        response = TestClient(app).post(GENERATE_PATH, json={"query": QUERY_EXACT})

        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "没有配置 LLM" in detail
        assert RETRIEVAL_LLM_STATE_KEY in detail

    def test_the_state_key_is_empty_per_app_instance(self) -> None:
        """注入挂在**应用实例**上：同一个进程里另建一个 app，那个位置仍然是空的.

        ``None`` 的含义是"没注入"（于是回落到 ``app.state.llm``），
        而"注入只活在这个应用实例里"因此是一件可断言的事。
        """
        first = make_client(llm=RecordingLLM(), retrieval_llm=RecordingLLM())
        second = make_client(llm=RecordingLLM())

        assert RETRIEVAL_LLM_STATE_KEY == "retrieval_llm"
        assert getattr(first.app.state, RETRIEVAL_LLM_STATE_KEY) is not None
        assert getattr(second.app.state, RETRIEVAL_LLM_STATE_KEY, None) is None
        assert first.app is not second.app

    def test_injection_does_not_change_the_grounding_verifier(self) -> None:
        """核对端点根本不读模型：注入什么、注没注入，它的结论逐位相同."""
        with_llm = verify_call(make_client(retrieval_llm=RecordingLLM()))
        without_llm = verify_call(make_client())

        assert six_numbers(with_llm) == six_numbers(without_llm)


# --------------------------------------------------------------------------- #
# day066~068 的十条端点行为不变（day069 只增键）
# --------------------------------------------------------------------------- #


class TestExistingEndpointsUnchanged:
    """``search`` / ``answer`` / ``rerank`` / ``hybrid`` 的旧口径逐位不变."""

    def test_search_is_unchanged(self, client: TestClient) -> None:
        result = client.post("/retrieval/search", json={"query": QUERY_EXACT}).json()["result"]

        assert [hit["record_id"] for hit in result["hits"]] == list(HIT_IDS)
        assert result["rerank"] == {}
        assert result["dropped_by_rerank"] == 0
        for hit in result["hits"]:
            assert hit["rerank_score"] is None
            assert hit["stage1_rank"] is None

    def test_answer_keeps_its_values_and_gains_two_keys(self) -> None:
        """键集合**只增**：旧十个键一个不少、值逐位不变，新增 ``check`` 与 ``fallback_reason``."""
        payload = answer_call(make_client(llm=RecordingLLM()))

        assert set(payload) == ANSWER_KEYS
        assert payload["llm_called"] is True
        assert payload["answer"] == RecordingLLM.REPLY
        assert payload["prompt_version"] == CURRENT_PROMPT_VERSION
        assert [citation["marker"] for citation in payload["citations"]] == [1, 2, 3, 4, 5]
        assert [citation["record_id"] for citation in payload["citations"]] == list(HIT_IDS)
        assert payload["context"]["count"] == 5
        assert payload["retrieval"]["count"] == 5
        assert payload["empty_reason"] == "hits"
        assert payload["route"]["name"] == "default"
        # 旧口径里那三个"没有重排"的字段仍然逐位不变。
        assert payload["retrieval"]["rerank"] == {}
        assert payload["retrieval"]["dropped_by_rerank"] == 0
        assert [hit["rerank_score"] for hit in payload["retrieval"]["hits"]] == [None] * 5

    def test_answer_reports_the_new_keys(self) -> None:
        """新键的值：同一份答案的核对结论 + "没有回退"的空串."""
        payload = answer_call(make_client(llm=RecordingLLM()))

        assert payload["fallback_reason"] == ""
        assert payload["check"]["given"] == 5
        assert payload["check"]["valid"] == [1, 2]
        assert payload["check"]["invalid"] == []
        assert payload["check"]["coverage"] == EXPECTED_COVERAGE
        assert payload["check"]["grounded"] is True
        assert [row["marker"] for row in payload["check"]["checks"]] == [1, 2, 3, 4, 5]

    def test_answer_on_an_empty_store_keeps_check_none(self) -> None:
        """空检索那条路一次都没走到生成：``check`` 是 None，而原因是 ``no_context``."""
        payload = answer_call(make_client(store=empty_store(), llm=RecordingLLM()))

        assert payload["llm_called"] is False
        assert payload["answer"] == FALLBACK_NO_CONTEXT
        assert payload["check"] is None
        assert payload["fallback_reason"] == NO_CONTEXT_REASON
        assert payload["empty_reason"] == "no_data"

    def test_the_answer_request_model_is_unchanged(self) -> None:
        """``/retrieval/answer`` 的请求体仍然**没有**采样参数与模板字段（day066 的取舍）."""
        schemas = make_client().get("/openapi.json").json()["components"]["schemas"]
        properties = schemas["RetrievalAnswerRequest"]["properties"]

        assert "question" in properties
        assert "temperature" not in properties
        assert "max_tokens" not in properties
        assert "prompt_version" not in properties
        assert "prompt" not in properties

    def test_rerank_is_unchanged(self, client: TestClient) -> None:
        """``/retrieval/rerank`` 的两个次序仍是 day068 手算的那两份."""
        payload = client.post(
            "/retrieval/rerank", json={"query": QUERY_EXACT, "top_k": 8, "fetch_k": 8}
        ).json()

        assert payload["before_ids"] == [
            "h-c-01",
            "h-c-02",
            "h-c-03",
            "h-c-04",
            "h-m-01",
            "h-m-02",
            "h-m-03",
            "h-t-01",
        ]
        assert payload["after_ids"][0] == "h-t-01"
        assert payload["rerank"]["scored"] == 8
        assert payload["rerank"]["window"] == 8

    def test_hybrid_is_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "retrieval_hybrid_enabled", True)
        client = make_client(llm=RecordingLLM())

        result = client.post("/retrieval/hybrid", json={"query": QUERY_EXACT}).json()["result"]

        assert [hit["record_id"] for hit in result["hits"]] == list(HYBRID_DEFAULT[QUERY_EXACT])
        assert result["rerank"] == {}
        assert result["dropped_by_rerank"] == 0

    def test_the_status_defaults_report_the_new_knobs(self, client: TestClient) -> None:
        """``/retrieval/status`` 的缺省值表也补上了生成这一组（与 day067/068 同一条纪律）."""
        defaults = client.get("/retrieval/status").json()["defaults"]

        assert defaults["prompt_version"] == settings.retrieval_prompt_version
        assert defaults["answer_temperature"] == settings.retrieval_answer_temperature
        assert defaults["answer_max_tokens"] == settings.retrieval_answer_max_tokens
        assert defaults["require_citation"] == settings.retrieval_require_citation

    def test_the_generating_channel_does_not_change_the_others(self, client: TestClient) -> None:
        """同一份请求体在三条通道上的"口令"互不串味.

        ```text
        /retrieval/generate   有 grounding（它自己算的那一份）
        /retrieval/search     没有 grounding/check/fallback_reason 这些键
        /retrieval/answer     两个新键都在（day069 只增不改）
        ```
        """
        before = client.post("/retrieval/search", json={"query": QUERY_EXACT}).json()
        generated = generate_call(client)
        after = client.post("/retrieval/search", json={"query": QUERY_EXACT}).json()

        assert "grounding" not in before["result"]
        assert "check" not in before["result"]
        assert generated["grounding"]["given"] == 5
        # 同一份请求体在生成前后给出同一份名单（耗时那类字段不参与比较）。
        assert [hit["record_id"] for hit in before["result"]["hits"]] == [
            hit["record_id"] for hit in after["result"]["hits"]
        ]
        assert before["summary"] == after["summary"]
