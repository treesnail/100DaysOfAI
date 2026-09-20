"""day062 ``/chunking/*`` 四个端点的单元测试.

全部离线：注入 ``MockEmbedding`` 的 ``TestClient``，不联网、不写盘、不遍历目录。
除了"四个端点各自返回什么"，这里还钉住了一条**参数优先级的规则**：
``请求体 > settings > 策略默认值``，而且重叠与下限是**按比例**跟着预算缩放的
——把 48 写死会让"预算调成 64"变成"75% 的重叠"，而参数看起来只是变小了。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from smart_research_agent.api.app import create_app
from smart_research_agent.chunking import (
    CHUNKING_LIMITATIONS,
    CHUNKING_OUT_OF_SCOPE,
    STRATEGIES,
    STRATEGY_FIXED,
    STRATEGY_RECURSIVE,
    STRATEGY_SEMANTIC,
    STRATEGY_STRUCTURAL,
)
from smart_research_agent.config import settings
from smart_research_agent.llm.embedding import MockEmbedding
from tests.chunking_samples import GUIDE_MARKDOWN, PLAIN_TEXT

PROBES = [
    {"question": "分块预算怎么定", "expect": "固定长度分块每 320 个字符切一刀"},
    {"question": "相似度阈值设多少", "expect": "相似度低于 0.35 的直接丢掉"},
]


@pytest.fixture
def client() -> TestClient:
    """注入 MockEmbedding 的离线客户端（语义策略因此完全离线）."""
    return TestClient(create_app(embedding=MockEmbedding()))


def split_body(**overrides: object) -> dict:
    body: dict = {"filename": "guide.md", "text": GUIDE_MARKDOWN}
    body.update(overrides)
    return body


# --------------------------------------------------------------------------- #
# GET /chunking/strategies
# --------------------------------------------------------------------------- #


def test_strategies_endpoint_describes_the_whole_capability(client: TestClient) -> None:
    """四张表全部从代码常量现场读出：文档与实现不会漂移."""
    response = client.get("/chunking/strategies")
    assert response.status_code == 200
    payload = response.json()
    assert [row["name"] for row in payload["strategies"]] == list(STRATEGIES)
    assert {row["name"] for row in payload["measurers"]} == {"chars", "tiktoken"}
    assert set(payload["default_policies"]) == set(STRATEGIES)
    assert payload["separators"][0] == "\n\n"
    assert payload["atomic_kinds"] == ["code", "table"]
    assert payload["knowledge_record"]["fields"]["doc_id"].startswith("块 id")
    assert "chunk.text" in payload["coverage_invariant"]
    assert payload["defaults"]["strategy"] == settings.chunking_strategy
    assert payload["defaults"]["max_tokens"] == settings.chunking_max_tokens
    assert payload["limitations"] == list(CHUNKING_LIMITATIONS)
    assert payload["out_of_scope"] == list(CHUNKING_OUT_OF_SCOPE)


def test_strategies_endpoint_reports_the_injected_embedding(client: TestClient) -> None:
    """语义策略的自述里要报出当前用的提供方（注入的是 Mock）."""
    payload = client.get("/chunking/strategies").json()
    semantic = next(row for row in payload["strategies"] if row["name"] == STRATEGY_SEMANTIC)
    assert semantic["embedding"] == "MockEmbedding"
    assert semantic["uses_similarity_percentile"] is True


# --------------------------------------------------------------------------- #
# POST /chunking/split
# --------------------------------------------------------------------------- #


def test_split_returns_plan_stats_and_chunks(client: TestClient) -> None:
    """三块内容各答一个问题：代价估算、实测统计、逐块明细."""
    response = client.post(
        "/chunking/split", json=split_body(strategy=STRATEGY_STRUCTURAL)
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"].startswith("structural |")
    assert payload["plan"]["strategy"] == STRATEGY_STRUCTURAL
    assert payload["plan"]["estimated_chunks_exact"] is False
    assert payload["stats"]["count"] >= 1
    assert payload["stats"]["coverage"] > 0.9
    assert payload["document"]["doc_id"]
    assert len(payload["chunks"]) == payload["stats"]["count"]
    # 明细里带 start_char / end_char：**"块是原文的连续子串"在这两个数上肉眼可验**
    assert payload["chunks"][0]["end_char"] > payload["chunks"][0]["start_char"]
    assert payload["chunks"][0]["heading_path"] == ["检索手册"]
    assert "text" not in payload["chunks"][0]


def test_split_can_return_text_and_limit_the_listing(client: TestClient) -> None:
    """``include_text`` 与 ``top_n`` 是两个独立的开关（明细要不要带全文、列几条）."""
    response = client.post(
        "/chunking/split",
        json=split_body(strategy=STRATEGY_RECURSIVE, include_text=True, top_n=2),
    )
    payload = response.json()
    assert len(payload["chunks"]) == 2
    assert payload["chunks"][0]["text"]
    assert payload["chunks"][0]["retrieval_text"] == payload["chunks"][0]["text"]


def test_split_applies_settings_baseline_by_default(client: TestClient) -> None:
    """不传参数时用 ``settings.chunking_*``：项目级基线优先于策略默认值."""
    payload = client.post(
        "/chunking/split", json=split_body(strategy=STRATEGY_RECURSIVE)
    ).json()
    policy = payload["stats"]["metadata"]["policy"]
    assert policy["max_tokens"] == settings.chunking_max_tokens
    assert policy["overlap_tokens"] == settings.chunking_overlap_tokens
    assert policy["min_tokens"] == settings.chunking_min_tokens
    assert policy["measurer"] == settings.chunking_measurer


def test_split_scales_overlap_and_floor_with_the_budget(client: TestClient) -> None:
    """重叠与下限**按比例**跟着预算走：48/320 = 15%，因此 64 的预算配 10 的重叠.

    写死 48 会让"预算调成 64"变成"75% 的重叠"——块一下子放大 3 倍，
    而参数看起来只是变小了。
    """
    payload = client.post(
        "/chunking/split",
        json=split_body(strategy=STRATEGY_RECURSIVE, max_tokens=64),
    ).json()
    policy = payload["stats"]["metadata"]["policy"]
    assert policy["max_tokens"] == 64
    assert policy["overlap_tokens"] == 10
    assert policy["min_tokens"] == 32


def test_split_respects_explicit_parameters(client: TestClient) -> None:
    """请求体最高优先级：显式给的重叠与下限直接生效."""
    payload = client.post(
        "/chunking/split",
        json=split_body(
            strategy=STRATEGY_RECURSIVE,
            max_tokens=64,
            overlap_tokens=0,
            min_tokens=0,
            similarity_percentile=0.5,
        ),
    ).json()
    policy = payload["stats"]["metadata"]["policy"]
    assert policy["overlap_tokens"] == 0
    assert policy["min_tokens"] == 0
    assert policy["similarity_percentile"] == 0.5


def test_split_keeps_structural_overlap_at_zero(client: TestClient) -> None:
    """结构策略的重叠恒为 0：项目基线不改这一条**结构性判断**."""
    payload = client.post(
        "/chunking/split", json=split_body(strategy=STRATEGY_STRUCTURAL)
    ).json()
    assert payload["stats"]["metadata"]["policy"]["overlap_tokens"] == 0
    assert payload["stats"]["duplication_ratio"] == 0.0


def test_split_protects_and_reports_oversized_atoms(client: TestClient) -> None:
    """结构策略在代码块上"宁可不切"，因此报告里必须出现一个超预算的块."""
    payload = client.post(
        "/chunking/split",
        json=split_body(strategy=STRATEGY_STRUCTURAL, max_tokens=40, overlap_tokens=0),
    ).json()
    assert payload["stats"]["oversized_count"] == 1
    oversized = [chunk for chunk in payload["chunks"] if chunk["oversized"]]
    assert oversized and oversized[0]["reason"] == "atom:code"


def test_split_rejects_unknown_strategy(client: TestClient) -> None:
    response = client.post("/chunking/split", json=split_body(strategy="rolling"))
    assert response.status_code == 400
    assert "rolling" in response.json()["detail"]


def test_split_rejects_overlap_on_structural(client: TestClient) -> None:
    """结构策略 + 非零重叠 → 400，且错误信息说明为什么（不是"不支持"）."""
    response = client.post(
        "/chunking/split",
        json=split_body(strategy=STRATEGY_STRUCTURAL, overlap_tokens=8),
    )
    assert response.status_code == 400
    assert "标题边界" in response.json()["detail"]


def test_split_rejects_a_contradictory_policy(client: TestClient) -> None:
    """``overlap >= max_tokens`` 是参数矛盾，必须报错而不是原地打转."""
    response = client.post(
        "/chunking/split",
        json=split_body(strategy=STRATEGY_FIXED, max_tokens=8, overlap_tokens=8),
    )
    assert response.status_code == 400
    assert "原地踏步" in response.json()["detail"]


def test_split_rejects_empty_content(client: TestClient) -> None:
    """空内容算"漏传"而不是"真的为空"（与 /documents/parse 同一判断）."""
    response = client.post("/chunking/split", json={"filename": "empty.txt"})
    assert response.status_code == 400
    assert "必须给出一个" in response.json()["detail"]


def test_split_rejects_a_bad_request_shape(client: TestClient) -> None:
    """字段类型不对是 422，与"业务规则不允许"的 400 分开."""
    response = client.post("/chunking/split", json={"text": GUIDE_MARKDOWN, "top_n": "许多"})
    assert response.status_code == 422


def test_split_reports_the_measurer_it_used(client: TestClient) -> None:
    """``token_measurer`` 跟着产物走：**这批块按什么单位切的随时可查**.

    ``tiktoken`` 依赖本机词表缓存，因此这里两种结果都接受——
    要么成功并报出 tiktoken，要么 400 明说不可用（**不允许静默降级**）。
    """
    response = client.post(
        "/chunking/split", json=split_body(strategy=STRATEGY_FIXED, measurer="tiktoken")
    )
    if response.status_code == 200:
        assert response.json()["stats"]["token_measurer"] == "tiktoken"
    else:
        assert response.status_code == 400
        assert "tiktoken" in response.json()["detail"]


def test_split_rejects_an_unknown_measurer(client: TestClient) -> None:
    response = client.post(
        "/chunking/split", json=split_body(measurer="wordpiece")
    )
    assert response.status_code == 400
    assert "wordpiece" in response.json()["detail"]


def test_split_handles_a_headingless_plain_text_document(client: TestClient) -> None:
    """纯文本文档没有标题：结构策略退化为按段落切，仍要给出正确的报告."""
    payload = client.post(
        "/chunking/split",
        json={"filename": "notes.txt", "text": PLAIN_TEXT, "strategy": STRATEGY_STRUCTURAL},
    ).json()
    assert payload["stats"]["metadata"]["sections"] == "1"
    assert payload["stats"]["coverage"] > 0.9


# --------------------------------------------------------------------------- #
# POST /chunking/evaluate
# --------------------------------------------------------------------------- #


def test_evaluate_returns_scores_and_markdown(client: TestClient) -> None:
    response = client.post(
        "/chunking/evaluate",
        json={
            "filename": "guide.md",
            "text": GUIDE_MARKDOWN,
            "probes": PROBES,
            "top_k": 8,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"].startswith("4 个策略")
    strategies = [row["strategy"] for row in payload["evaluation"]["strategies"]]
    assert strategies == list(STRATEGIES)
    assert all(row["hit_rate"] == 1.0 for row in payload["evaluation"]["strategies"])
    assert "# 分块策略检索评估" in payload["markdown"]


def test_evaluate_can_compare_a_subset_and_default_top_k(client: TestClient) -> None:
    """``strategies`` 留空表示全部；``top_k`` 留空用 ``settings.chunking_eval_top_k``."""
    payload = client.post(
        "/chunking/evaluate",
        json={
            "filename": "guide.md",
            "text": GUIDE_MARKDOWN,
            "probes": PROBES,
            "strategies": [STRATEGY_STRUCTURAL],
        },
    ).json()
    assert [row["strategy"] for row in payload["evaluation"]["strategies"]] == [
        STRATEGY_STRUCTURAL
    ]
    assert payload["evaluation"]["strategies"][0]["top_k"] == settings.chunking_eval_top_k


def test_evaluate_uniform_budget_override(client: TestClient) -> None:
    """``max_tokens`` 统一覆盖各策略的预算（选型时最常用的一步）."""
    payload = client.post(
        "/chunking/evaluate",
        json={
            "filename": "guide.md",
            "text": GUIDE_MARKDOWN,
            "probes": PROBES,
            "max_tokens": 64,
            "top_k": 8,
        },
    ).json()
    assert payload["evaluation"]["metadata"]["document"] == "guide.md"
    assert all(row["chunks"] >= 1 for row in payload["evaluation"]["strategies"])


def test_evaluate_rejects_empty_probes(client: TestClient) -> None:
    """没有期望片段就无法判定命中——这一条必须挡住，而不是给一份 0 分的表."""
    response = client.post(
        "/chunking/evaluate", json={"filename": "guide.md", "text": GUIDE_MARKDOWN}
    )
    assert response.status_code == 400
    assert "probes 不能为空" in response.json()["detail"]


def test_evaluate_rejects_a_probe_without_expectation(client: TestClient) -> None:
    """缺字段是 **422**（请求形状不对），空字符串是 **400**（业务规则不允许）.

    两种拒绝要分开：前者是写错了请求，后者是"你确实没给期望片段"——
    而后者恰恰是最容易发生的情形（把 expect 留成默认空串）。
    """
    missing = client.post(
        "/chunking/evaluate",
        json={"text": GUIDE_MARKDOWN, "probes": [{"question": "缺少期望片段"}]},
    )
    assert missing.status_code == 422
    blank = client.post(
        "/chunking/evaluate",
        json={"text": GUIDE_MARKDOWN, "probes": [{"question": "空期望片段", "expect": "  "}]},
    )
    assert blank.status_code == 400
    assert "expect" in blank.json()["detail"]


def test_evaluate_surfaces_ambiguous_probes(client: TestClient) -> None:
    """出现多次的期望片段单独返回：它会让分数虚高，且**虚高得毫不显眼**."""
    payload = client.post(
        "/chunking/evaluate",
        json={
            "filename": "dup.txt",
            "text": "重复的一句。\n\n重复的一句。",
            "probes": [{"question": "重复", "expect": "重复的一句。"}],
        },
    ).json()
    assert payload["evaluation"]["ambiguous_probes"] == [
        "重复 → 期望片段在原文中出现 2 次"
    ]
    assert "歧义探针" in payload["markdown"]


# --------------------------------------------------------------------------- #
# POST /chunking/records
# --------------------------------------------------------------------------- #


def test_records_returns_knowledge_base_shape(client: TestClient) -> None:
    """四个键、片段身份、以及**明确写出的向量化字段**."""
    response = client.post(
        "/chunking/records",
        json=split_body(strategy=STRATEGY_STRUCTURAL),
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["embedding_input"] == "metadata.retrieval_text"
    assert payload["summary"].startswith("1 份文档 × 1 个策略")
    assert payload["report"]["chunks"] == len(payload["records"])
    first = payload["records"][0]
    assert set(first) == {"doc_id", "source", "text", "metadata"}
    assert first["metadata"]["parent_doc_id"]
    assert first["metadata"]["retrieval_text"]
    assert "# 分块报告" in payload["markdown"]


def test_records_limit_zero_returns_everything(client: TestClient) -> None:
    """``limit=0`` 表示不截断（默认 10 只是为了让响应体小一点）."""
    payload = client.post(
        "/chunking/records",
        json=split_body(strategy=STRATEGY_FIXED, limit=0, max_tokens=40, overlap_tokens=0),
    ).json()
    assert len(payload["records"]) == payload["report"]["chunks"] > 10


def test_records_rejects_a_broken_document(client: TestClient) -> None:
    response = client.post(
        "/chunking/records", json={"filename": "broken.txt", "content_base64": "%%%"}
    )
    assert response.status_code == 400
