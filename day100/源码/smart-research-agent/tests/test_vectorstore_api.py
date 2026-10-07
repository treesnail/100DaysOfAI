"""day064 ``/vectorstore/*`` 六个端点的单元测试.

全部离线、确定性、零网络：注入 ``FlatVectorStore``（零可选依赖的参照实现）与
``TableEmbedding``（按文本查表的确定性编码器），因此"哪条最近""候选有几条"
都是手算得出来的数，不是"跑一遍看看"。

除了"六个端点各自返回什么"，这里钉住三件比 200 更重要的事：

1. **写入状态挂在应用实例上**（``app.state.vector_store`` /
   ``app.state.vectorstore_last_ingest``）：同一个进程里另建一个 app，
   它的库必须是空的、最近一次写入报告必须是 ``None``——否则
   "内存后端的写入只活在这个实例里"就是一句无法核对的话；
2. **缺可选依赖时 compare 降级而不是 500**：报告在任何机器上都是同样的行数，
   缺 faiss/chromadb 的行只是 ``available=False``（带三段式安装指引）；
3. **三种"没返回"要分得开**：``candidates`` / ``filter_applied`` / ``count``
   三个数字一起才说得出"为什么只命中了 0 条"。

样本与期望值的来源：``tests/vectorstore_samples.py`` 的六条记录里，
``r_a``（下标 0）与 ``r_d``（下标 3）对 ``q_axis`` 的**余弦精确并列**（都是 1.0），
而 ``r_d`` 的模长是 10 倍——这一条样本同时支撑了"平局按 id 断"与
"ip 与 cosine 的 top-1 分家"两个断言。
"""

from __future__ import annotations

import importlib.util
import json

import pytest
from fastapi.testclient import TestClient

from smart_research_agent.api.app import create_app
from smart_research_agent.config import settings
from smart_research_agent.vectorstore import (
    MAX_TOP_K,
    METRICS,
    VECTORSTORE_LIMITATIONS,
    VECTORSTORE_OUT_OF_SCOPE,
    FlatVectorStore,
    backend_names,
    describe_backends,
    describe_metrics,
    embedding_text,
    record_from_knowledge,
)
from tests.vectorstore_samples import (
    RECORD_IDS,
    RECORD_SPECS,
    VECTOR_DIMENSION,
    TableEmbedding,
    knowledge_records,
)

#: 与六条样本都正交的查询文本（它的向量是最后一根坐标轴，与所有样本余弦为 0）.
#: 它专门用来制造"**候选不为空、但一条都过不了阈值**"那种局面。
QUERY_ORTHOGONAL = "一个与六条样本都正交的查询"


def retrieval_view(index: int) -> str:
    """第 ``index`` 条样本的**检索视图**（``heading_path + 正文``）.

    它才是被送进编码器的那份文本（``pipeline.embedding_text`` 优先取
    ``metadata.retrieval_text``），因此把它当查询就等于"用这条记录去查它自己"。
    """
    spec = RECORD_SPECS[index]
    return f"{spec[3]['heading_path']}\n{spec[2]}"


def api_embedding(extra: dict[str, tuple[float, ...]] | None = None) -> TableEmbedding:
    """构造端点测试用的查表编码器（正文与检索视图都映射到同一条向量）.

    为什么两张键都要写：``text`` 是**落库**的那份原文，``retrieval_text`` 是
    **被编码**的那份检索视图。只注册其中一个，另一边就会落到 ``default`` 上——
    那样写出来的断言测的是兜底向量，而不是样本的语义。
    """
    table: dict[str, tuple[float, ...]] = {}
    for spec in RECORD_SPECS:
        table[spec[2]] = spec[1]
        table[f"{spec[3]['heading_path']}\n{spec[2]}"] = spec[1]
    table.update(extra or {})
    return TableEmbedding(table, dimension=VECTOR_DIMENSION)


def api_records() -> list[dict]:
    """六条 day062 形状的记录，外加一个**数组型**元数据 ``tags``.

    补 ``tags`` 的理由：``$contains`` / ``$not_contains`` 只对数组有意义，
    而 day062 交出来的那份记录里一个数组都没有——不补的话
    "数组过滤"这条能力在端点层就永远没被验证过。
    """
    records = knowledge_records()
    for record, spec in zip(records, RECORD_SPECS):
        record["metadata"]["tags"] = list(spec[3]["tags"])
    return records


@pytest.fixture
def store() -> FlatVectorStore:
    """每个用例一个空库：内存后端不落盘，因此用例之间不会互相污染."""
    return FlatVectorStore()


@pytest.fixture
def embedding() -> TableEmbedding:
    """注入 ``app.state.embedding`` 的确定性编码器（与入库用同一个实例）."""
    return api_embedding()


@pytest.fixture
def client(store: FlatVectorStore, embedding: TableEmbedding) -> TestClient:
    """注入 flat 后端 + 查表编码器的离线客户端（缺 faiss/chromadb 也能跑）."""
    return TestClient(create_app(embedding=embedding, vector_store=store))


def make_client(
    *,
    vector_store: FlatVectorStore | None = None,
    embedding: TableEmbedding | None = None,
) -> TestClient:
    """按需另建一个客户端（用于"另一个应用实例"与"维度不一致"两条路径）."""
    return TestClient(
        create_app(
            embedding=embedding or api_embedding(),
            vector_store=vector_store if vector_store is not None else FlatVectorStore(),
        )
    )


def upsert_all(client: TestClient, records: list[dict] | None = None) -> dict:
    """写入六条样本并断言这一步是成功的（后续用例的前置条件）."""
    response = client.post(
        "/vectorstore/upsert", json={"records": records or api_records()}
    )
    assert response.status_code == 200
    return response.json()


def stats(client: TestClient) -> dict:
    """读一次库状态（``/vectorstore/stats``）."""
    response = client.get("/vectorstore/stats")
    assert response.status_code == 200
    return response.json()


def search(client: TestClient, **body: object) -> dict:
    """发一次检索请求；``query`` 默认用第一条样本的检索视图."""
    payload: dict = {"query": retrieval_view(0)}
    payload.update(body)
    response = client.post("/vectorstore/search", json=payload)
    assert response.status_code == 200
    return response.json()


def delete_records(client: TestClient, **body: object):
    """发一次删除请求（DELETE 带 JSON 体，见路由 docstring）."""
    return client.request("DELETE", "/vectorstore/records", json=body)


# --------------------------------------------------------------------------- #
# GET /vectorstore/backends
# --------------------------------------------------------------------------- #


def test_backends_lists_three_backends_and_flat_is_always_available(
    client: TestClient,
) -> None:
    """三个后端都在表里，且 flat 的 ``available`` 恒为 True（它 ``requires=()``）."""
    response = client.get("/vectorstore/backends")
    assert response.status_code == 200
    payload = response.json()
    assert [row["name"] for row in payload["backends"]] == list(backend_names())
    assert len(payload["backends"]) == 3
    flat = next(row for row in payload["backends"] if row["name"] == "flat")
    assert flat["available"] is True
    assert flat["missing"] == []
    assert flat["requires"] == []
    # available 与 missing 一起给：只给布尔值时看到 False 还得猜缺哪个包
    for row in payload["backends"]:
        assert row["available"] is (row["missing"] == [])
        if not row["available"]:
            assert row["install_hint"].startswith("pip install ")


def test_backends_table_is_echoed_from_the_registry_not_reassembled(
    client: TestClient,
) -> None:
    """两张表直接来自 ``describe_backends()`` / ``describe_metrics()``.

    端点自己拼一份字段列表，就会出现"文档与接口各有一份说法"，
    而它们在下次加字段时必然分叉。
    """
    payload = client.get("/vectorstore/backends").json()
    assert payload["backends"] == describe_backends()
    assert payload["metrics"] == describe_metrics()
    assert [row["metric"] for row in payload["metrics"]] == list(METRICS)


def test_backends_separates_capability_table_from_current_choice(
    client: TestClient,
) -> None:
    """能力表回答"能用什么"，``current_*`` 回答"这个实例现在用的是哪个"."""
    payload = client.get("/vectorstore/backends").json()
    assert payload["current_backend"] == "flat"
    assert payload["current_metric"] == "cosine"
    assert payload["current_count"] == 0
    assert payload["default_backend"] == settings.vector_backend
    assert payload["default_metric"] == settings.vector_metric
    assert payload["default_top_k"] == settings.vector_default_top_k
    assert payload["min_score"] == settings.vector_min_score
    assert payload["persistence_path"] == settings.vector_persist_path


def test_backends_reports_the_current_count_of_the_injected_store(
    client: TestClient,
) -> None:
    """写入之后 ``current_count`` 跟着变——它读的是注入的那个后端，不是配置."""
    upsert_all(client)
    assert client.get("/vectorstore/backends").json()["current_count"] == 6


def test_backends_declares_limits_and_out_of_scope(client: TestClient) -> None:
    """边界与"明确不做"照原样回显（它们是从代码常量读的，不会漂移）."""
    payload = client.get("/vectorstore/backends").json()
    assert payload["limitations"] == list(VECTORSTORE_LIMITATIONS)
    assert payload["out_of_scope"] == list(VECTORSTORE_OUT_OF_SCOPE)


# --------------------------------------------------------------------------- #
# POST /vectorstore/upsert
# --------------------------------------------------------------------------- #


def test_upsert_writes_six_records_with_six_embedding_calls(
    client: TestClient,
) -> None:
    """六条样本 → ``written=6`` 且 ``embedding_calls=6``（本课刻意逐条编码）."""
    payload = upsert_all(client)
    report = payload["report"]
    assert report["seen"] == 6
    assert report["written"] == 6
    assert report["unchanged"] == 0
    assert report["skipped"] == 0
    assert report["failed"] == 0
    assert report["embedding_calls"] == 6
    assert report["dimension"] == VECTOR_DIMENSION
    assert report["ok"] is True
    assert payload["count"] == 6
    assert payload["dimension"] == VECTOR_DIMENSION
    assert payload["backend"] == "flat"
    assert payload["metric"] == "cosine"
    assert payload["summary"].startswith("flat/cosine/8d")
    # 统计端点必须看到同一件事（两个端点读的是同一个库）
    assert stats(client)["info"]["count"] == 6


def test_the_same_batch_is_six_added_at_the_backend_level() -> None:
    """端点的 ``written`` 是后端 ``WriteReport`` 的 ``added + updated``.

    端点返回 ``IngestReport``，它没有单独的 ``added`` 字段（那是 ``WriteReport``
    的四个数之一，被折进 ``written``）。这条用例把那份底账直接摆出来：
    **六条全新记录 → ``added=6``、``updated=0``**，因此"重放得到 ``unchanged=6``"
    说的确实是"一条都没有被改写"，而不是"六条都被覆盖了一遍"。
    """
    embedding = api_embedding()
    batch = [
        record_from_knowledge(raw, embedding.embed(embedding_text(raw)))
        for raw in api_records()
    ]
    report = FlatVectorStore().upsert(batch)
    assert report.added == 6
    assert report.updated == 0
    assert report.written == 6


def test_upsert_report_is_json_serializable_and_self_consistent(
    client: TestClient,
) -> None:
    """``IngestReport`` 的两个约定：能直接 ``json.dumps``、且恒等式成立."""
    report = upsert_all(client)["report"]
    assert json.loads(json.dumps(report)) == report
    assert report["seen"] == (
        report["written"] + report["unchanged"] + report["skipped"] + report["failed"]
    )
    # failures 与 skipped 的条数一一对应（报告不藏任何一条）
    assert len(report["failures"]) == report["skipped"] + report["failed"]


def test_upsert_replay_changes_nothing(client: TestClient) -> None:
    """重放同一批：``unchanged=6`` 且 ``written=0``——"这次其实什么都没写"的证据.

    注意 ``embedding_calls`` 仍然是 6：本课**没有**向量缓存（day065 才加），
    "库没变"与"没花编码"是两件事，而报告把它们分开摆着。
    """
    upsert_all(client)
    report = upsert_all(client)["report"]
    assert report["written"] == 0
    assert report["unchanged"] == 6
    assert report["embedding_calls"] == 6
    assert stats(client)["info"]["count"] == 6


def test_upsert_skips_records_without_an_id(client: TestClient) -> None:
    """缺 ``doc_id`` 的记录被跳过（数据问题），其余五条照常入库，且不花它的编码."""
    records = api_records()
    records[0]["doc_id"] = "   "
    report = upsert_all(client, records)["report"]
    assert report["seen"] == 6
    assert report["written"] == 5
    assert report["skipped"] == 1
    assert report["failed"] == 0
    assert report["embedding_calls"] == 5
    assert report["ok"] is False
    assert report["failures"][0]["record_id"] == ""
    assert "跳过" in report["failures"][0]["reason"]
    assert stats(client)["info"]["count"] == 5


def test_upsert_rejects_an_empty_batch(client: TestClient) -> None:
    """空批次 → 400：一次什么都没写的 upsert 只会刷新状态，不该是一次成功调用."""
    response = client.post("/vectorstore/upsert", json={"records": []})
    assert response.status_code == 400
    assert "records 不能为空" in response.json()["detail"]


def test_upsert_rejects_a_bad_request_shape(client: TestClient) -> None:
    """字段类型不对是 422（请求形状问题），与业务规则的 400 分开."""
    response = client.post("/vectorstore/upsert", json={"records": "六条"})
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# GET /vectorstore/stats
# --------------------------------------------------------------------------- #


def test_stats_reports_store_state_and_encoder_info(client: TestClient) -> None:
    """三块内容各答一个问题：库状态、编码器、检索默认值."""
    upsert_all(client)
    payload = stats(client)
    info = payload["info"]
    assert info["backend"] == "flat"
    assert info["metric"] == "cosine"
    assert info["dimension"] == VECTOR_DIMENSION
    assert info["count"] == 6
    assert info["persistent"] is False
    # 后端私有概念留在 extra 里（flat 报快照版本；另两个后端不该带着空字符串）
    assert info["extra"]["snapshot_version"] == "1"
    pipeline = payload["pipeline"]
    assert pipeline["embedding"] == "TableEmbedding"
    assert pipeline["embedding_dimension"] == VECTOR_DIMENSION
    assert pipeline["embedding_input_field"] == "retrieval_text"
    assert pipeline["top_k"] == settings.vector_default_top_k
    assert pipeline["min_score"] is None
    assert payload["summary"].startswith("flat | cosine | 8d")


def test_stats_is_empty_before_any_write(client: TestClient) -> None:
    """还没写过：``count=0``、``dimension=0``（维度未定的哨兵）、``last_ingest=None``."""
    payload = stats(client)
    assert payload["info"]["count"] == 0
    assert payload["info"]["dimension"] == 0
    assert payload["last_ingest"] is None


def test_stats_remembers_the_last_ingest_of_this_instance(client: TestClient) -> None:
    """最近一次写入报告记在 ``app.state`` 上，并原样带回（第二次会覆盖第一次）."""
    upsert_all(client)
    assert stats(client)["last_ingest"]["written"] == 6
    upsert_all(client)
    assert stats(client)["last_ingest"]["unchanged"] == 6


def test_writes_do_not_leak_into_another_app_instance(client: TestClient) -> None:
    """写入只活在这个应用实例里：另建一个 app 看到的是**空库**.

    这一条同时否掉了两种实现：把库放在模块级全局上（两个实例共享一份数据），
    或把最近一次报告放进 ``settings``（那会让"谁写的"永远说不清）。
    """
    upsert_all(client)
    other = make_client()
    payload = stats(other)
    assert payload["info"]["count"] == 0
    assert payload["last_ingest"] is None
    # 原实例的数据仍然在（不是被另一个实例清掉了）
    assert stats(client)["info"]["count"] == 6


# --------------------------------------------------------------------------- #
# POST /vectorstore/search
# --------------------------------------------------------------------------- #


def test_search_uses_the_settings_baseline_top_k_by_default(
    client: TestClient,
) -> None:
    """``top_k=0`` 是"没给"，用项目级基线；候选数是库里的全部记录."""
    upsert_all(client)
    result = search(client)["result"]
    assert result["top_k"] == settings.vector_default_top_k
    assert result["candidates"] == 6
    assert result["filter_applied"] is False
    assert result["count"] == settings.vector_default_top_k


def test_search_honours_top_k_and_breaks_ties_by_id(client: TestClient) -> None:
    """``top_k`` 生效；并列时按 id 升序——**不是**按插入顺序.

    ``r_a`` 与 ``r_d`` 对这条查询的余弦都是 1.0（向量同向、只是模长不同），
    因此名次必须由 id 决定。少了这个第二排序键，同一次查询两次运行
    会给出不同的 top-1，而那取决于字典的迭代顺序。
    """
    upsert_all(client)
    result = search(client, top_k=2)["result"]
    assert result["count"] == 2
    assert [hit["record_id"] for hit in result["hits"]] == [
        RECORD_IDS[0],
        RECORD_IDS[3],
    ]
    assert result["hits"][0]["score"] == pytest.approx(1.0)
    assert result["hits"][1]["score"] == pytest.approx(1.0)
    # 命中明细要带正文（命中之后要给用户看的就是它）与元数据（溯源）
    assert result["hits"][0]["text"] == RECORD_SPECS[0][2]
    assert result["hits"][0]["metadata"]["strategy"] == "recursive"


def test_search_reports_the_filter_it_applied(client: TestClient) -> None:
    """``where`` 过滤生效：候选数变小，且报告里能读到这次是什么条件."""
    upsert_all(client)
    payload = search(client, where={"strategy": "structural"})
    result = payload["result"]
    assert result["filter_applied"] is True
    assert result["candidates"] == 2
    assert result["count"] == 2
    assert payload["filter"] == '{"strategy": "structural"}'
    assert payload["filter_fields"] == ["strategy"]
    assert payload["query_chars"] == len(retrieval_view(0))


def test_search_supports_in_and_contains_operators(client: TestClient) -> None:
    """``$in`` 与 ``$contains`` 各一次：数组过滤走的是同一条编译路径."""
    upsert_all(client)
    inside = search(client, where={"strategy": {"$in": ["recursive", "fixed"]}})["result"]
    assert inside["candidates"] == 3
    contained = search(client, where={"tags": {"$contains": "cost"}})["result"]
    assert contained["candidates"] == 2
    assert sorted(hit["record_id"] for hit in contained["hits"]) == sorted(
        [RECORD_IDS[2], RECORD_IDS[5]]
    )


def test_search_min_score_can_empty_the_hits_but_keep_candidates(
    client: TestClient,
) -> None:
    """``candidates=1`` 而 ``count=0``：阈值把命中切光了，而库里**有**候选.

    这是本组字段设计最直白的回报：没有 ``candidates`` 这个数，
    "一条都没命中"与"库里根本没有这条"在响应体里长得一模一样。
    """
    upsert_all(client)
    payload = search(
        client,
        query=QUERY_ORTHOGONAL,
        where={"strategy": "semantic"},
        min_score=0.5,
    )["result"]
    assert payload["count"] == 0
    assert payload["hits"] == []
    assert payload["candidates"] == 1
    assert payload["filter_applied"] is True


def test_search_reports_a_filter_that_excludes_everything(
    client: TestClient,
) -> None:
    """过滤器把库筛空：``candidates=0`` 且 ``filter_applied=True``（第三种空结果）."""
    upsert_all(client)
    payload = search(client, where={"strategy": "不存在这个策略"})["result"]
    assert payload["candidates"] == 0
    assert payload["count"] == 0
    assert payload["filter_applied"] is True


def test_search_rejects_an_unknown_operator_with_the_supported_list(
    client: TestClient,
) -> None:
    """非法 ``where`` → 400，且消息里带**支持的运算符列表**（照它就能改对）."""
    upsert_all(client)
    response = client.post(
        "/vectorstore/search",
        json={"query": retrieval_view(0), "where": {"strategy": {"$regex": "a"}}},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "$regex" in detail
    assert "支持的运算符" in detail
    for operator in ("$contains", "$and", "$nin"):
        assert operator in detail


def test_search_rejects_a_logical_clause_mixed_with_field_conditions(
    client: TestClient,
) -> None:
    """``$and`` 混写在字段条件里 → 400（语义不确定，编译器拒收而不是猜）."""
    response = client.post(
        "/vectorstore/search",
        json={"query": retrieval_view(0), "where": {"$and": [{"a": 1}], "b": 2}},
    )
    assert response.status_code == 400
    assert "$and / $or 必须单独占一层" in response.json()["detail"]


def test_search_rejects_top_k_beyond_the_ceiling(client: TestClient) -> None:
    """``top_k`` 越界 → 400（FilterError），消息里带上限值."""
    response = client.post(
        "/vectorstore/search", json={"query": retrieval_view(0), "top_k": MAX_TOP_K + 1}
    )
    assert response.status_code == 400
    assert str(MAX_TOP_K) in response.json()["detail"]


def test_search_rejects_a_negative_top_k(client: TestClient) -> None:
    """负数不是"没给"：0 用基线，负数照实送进校验 → 400."""
    response = client.post(
        "/vectorstore/search", json={"query": retrieval_view(0), "top_k": -3}
    )
    assert response.status_code == 400
    assert "top_k 必须 >= 1" in response.json()["detail"]


def test_search_rejects_a_blank_query(client: TestClient) -> None:
    """空白查询 → 400：空查询会得到一个任意（或零）向量，而结果看起来正常."""
    response = client.post("/vectorstore/search", json={"query": "   "})
    assert response.status_code == 400
    assert "query 不能为空" in response.json()["detail"]


def test_search_dimension_mismatch_returns_409() -> None:
    """维度不一致 → **409**（Conflict：重新编码能修，改请求参数修不了）.

    这条契约写进了端点 docstring，因此必须在这里被断言——
    否则"409 还是 400"会变成下一个人凭感觉改的一行代码。
    """
    mismatch = make_client(vector_store=FlatVectorStore(dimension=4))
    response = mismatch.post("/vectorstore/search", json={"query": retrieval_view(0)})
    assert response.status_code == 409
    assert "查询向量是 8 维" in response.json()["detail"]


def test_search_on_an_empty_store_returns_no_hits(client: TestClient) -> None:
    """空库也要给出一份可读的结果（候选 0、命中 0），而不是报错."""
    result = search(client)["result"]
    assert result["candidates"] == 0
    assert result["count"] == 0
    assert result["hits"] == []


# --------------------------------------------------------------------------- #
# DELETE /vectorstore/records
# --------------------------------------------------------------------------- #


def test_delete_by_ids_returns_what_was_really_removed(client: TestClient) -> None:
    """按 ids 删除：返回值是**真正删掉的**条数（删两次第二次是 0）."""
    upsert_all(client)
    response = delete_records(client, ids=[RECORD_IDS[0]])
    assert response.status_code == 200
    assert response.json() == {
        "summary": "删除 1 条，剩余 5 条",
        "removed": 1,
        "count": 5,
    }
    again = delete_records(client, ids=[RECORD_IDS[0]])
    assert again.json()["removed"] == 0
    assert again.json()["count"] == 5


def test_delete_by_where_removes_every_match(client: TestClient) -> None:
    """按 ``where`` 删除：返回剩余条数，且条件外的一条都不动."""
    upsert_all(client)
    response = delete_records(client, where={"strategy": "structural"})
    assert response.status_code == 200
    assert response.json()["removed"] == 2
    assert response.json()["count"] == 4
    remaining = search(client, top_k=10)["result"]["hits"]
    assert RECORD_IDS[3] not in [hit["record_id"] for hit in remaining]


def test_delete_rejects_missing_or_duplicate_criteria(client: TestClient) -> None:
    """三种写法各挡一条：同时给、都不给、给空字典."""
    both = delete_records(client, ids=[RECORD_IDS[0]], where={"strategy": "structural"})
    assert both.status_code == 400
    assert "只能给一个" in both.json()["detail"]
    neither = delete_records(client)
    assert neither.status_code == 400
    assert "至少要给一个" in neither.json()["detail"]
    empty = delete_records(client, where={})
    assert empty.status_code == 400
    assert "等于清库" in empty.json()["detail"]


def test_delete_rejects_an_invalid_where(client: TestClient) -> None:
    """``where`` 写法非法 → 400（与检索同一条错误通道）."""
    response = delete_records(client, where={"strategy": {"$regex": "a"}})
    assert response.status_code == 400
    assert "支持的运算符" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# POST /vectorstore/compare
# --------------------------------------------------------------------------- #


def test_compare_reports_every_backend_even_when_dependencies_are_missing(
    client: TestClient,
) -> None:
    """缺 faiss/chromadb 时 **200**，且那两个后端是 ``available=False`` 的行.

    这是本端点最重要的行为：报告的行数在任何机器上都相同，只差一个布尔。
    若把缺依赖变成 500，第二台机器上就只能得到一句错误，
    而"两份报告逐行对照"这件事永远做不到。
    """
    response = client.post(
        "/vectorstore/compare",
        json={"records": api_records(), "query": retrieval_view(0), "top_k": 3},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["backends"] == sorted(backend_names())
    assert len(payload["rows"]) == len(backend_names())
    flat = next(row for row in payload["rows"] if row["backend"] == "flat")
    assert flat["available"] is True
    assert flat["count"] == 6
    assert flat["agreed"] == 3
    assert flat["overlap"] == pytest.approx(1.0)
    assert flat["note"] == f"top-1={RECORD_IDS[0]}"
    assert payload["check"]["checked"] == len(payload["rows"])
    assert payload["dimension"] == VECTOR_DIMENSION


def test_compare_degrades_unavailable_backends_into_rows(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """把依赖探测打成"什么都没装"，faiss/chroma 必须变成两行而不是一次 500.

    直接 monkeypatch ``importlib.util.find_spec``（registry 刻意用属性访问
    调用它，就是为了让这条分支能被真实走到），因此这条用例在装了 faiss
    的机器上同样有效——**覆盖哪一半不该由本机装了什么决定**。
    """
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *args: object, **kwargs: object):
        if name.split(".")[0] in {"faiss", "numpy", "chromadb"}:
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    response = client.post(
        "/vectorstore/compare",
        json={"records": api_records(), "query": retrieval_view(0), "top_k": 2},
    )
    assert response.status_code == 200
    payload = response.json()
    rows = {row["backend"]: row for row in payload["rows"]}
    assert set(rows) == set(backend_names())
    for name in ("faiss", "chroma"):
        assert rows[name]["available"] is False
        assert rows[name]["first_divergence"] == "后端不可用"
        # 三段式：缺什么 → 怎么装 → 还能用什么
        assert "不可用：缺少" in rows[name]["note"]
        assert "安装：" in rows[name]["note"]
        assert "或者改用：" in rows[name]["note"]
    assert rows["flat"]["available"] is True
    assert payload["check"]["available"] == 1
    assert len(payload["check"]["failures"]) == 2


def test_compare_keeps_inner_product_and_cosine_apart(client: TestClient) -> None:
    """同一个后端跑两个度量：ip 的 top-1 是 ``r_d``，cosine 的 top-1 是 ``r_a``.

    这条用例钉住的是端点里那个**刻意的选择**：记录在这里不归一化，
    由每个后端按自己的度量归一化。若端点先把向量归一化，
    两行的 top-1 会完全一样，而"换度量会换答案"这件事就从报告里消失了——
    报告仍然全绿，只是它再也不说明任何事情。
    """
    response = client.post(
        "/vectorstore/compare",
        json={
            "records": api_records(),
            "query": retrieval_view(0),
            "top_k": 2,
            "metrics": ["cosine", "ip"],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["metrics"] == ["cosine", "ip"]
    assert len(payload["rows"]) == 2 * len(backend_names())
    cosine = next(
        row for row in payload["rows"] if row["backend"] == "flat" and row["metric"] == "cosine"
    )
    inner = next(
        row for row in payload["rows"] if row["backend"] == "flat" and row["metric"] == "ip"
    )
    assert cosine["note"] == f"top-1={RECORD_IDS[0]}"
    assert inner["note"] == f"top-1={RECORD_IDS[3]}"


def test_compare_does_not_touch_the_serving_store(client: TestClient) -> None:
    """对账用的是"用完即弃"的临时库：它不进 ``app.state``，也不落盘.

    否则一次对账就会把正在服务的库改成"最后那个度量的库"，
    而使用者只是"看了一眼报告"。
    """
    response = client.post(
        "/vectorstore/compare",
        json={"records": api_records(), "query": retrieval_view(0)},
    )
    assert response.status_code == 200
    payload = stats(client)
    assert payload["info"]["count"] == 0
    assert payload["info"]["dimension"] == 0
    assert payload["last_ingest"] is None


def test_compare_rejects_an_incomplete_input(client: TestClient) -> None:
    """对账的输入必须完整：缺记录、缺 id、未知度量各 400."""
    assert (
        client.post("/vectorstore/compare", json={"query": retrieval_view(0)}).status_code
        == 400
    )
    assert (
        client.post(
            "/vectorstore/compare", json={"records": api_records()}
        ).status_code
        == 400
    )
    broken = api_records()
    broken[2]["doc_id"] = ""
    missing = client.post(
        "/vectorstore/compare", json={"records": broken, "query": retrieval_view(0)}
    )
    assert missing.status_code == 400
    assert "第 3 条记录缺少 'doc_id'" in missing.json()["detail"]
    unknown = client.post(
        "/vectorstore/compare",
        json={"records": api_records(), "query": retrieval_view(0), "metrics": ["hamming"]},
    )
    assert unknown.status_code == 400
    assert "hamming" in unknown.json()["detail"]


# --------------------------------------------------------------------------- #
# 契约层面的两个收口
# --------------------------------------------------------------------------- #


def test_vectorstore_paths_are_published_in_openapi(client: TestClient) -> None:
    """六个端点都进了 OpenAPI（客户端代码生成与 /docs 都靠它）."""
    paths = client.get("/openapi.json").json()["paths"]
    assert {
        "/vectorstore/backends",
        "/vectorstore/upsert",
        "/vectorstore/search",
        "/vectorstore/stats",
        "/vectorstore/records",
        "/vectorstore/compare",
    } <= set(paths)
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    for model in (
        "VectorStoreBackendsResponse",
        "VectorUpsertRequest",
        "VectorUpsertResponse",
        "VectorSearchRequest",
        "VectorSearchResponse",
        "VectorStatsResponse",
        "VectorDeleteRequest",
        "VectorDeleteResponse",
        "VectorCompareRequest",
        "VectorCompareResponse",
    ):
        assert model in schemas


def test_default_app_uses_an_in_memory_flat_store() -> None:
    """``create_app()`` 的缺省后端是 flat 且不落盘（``vector_persist_path`` 为空串）.

    默认"落盘"是一个很贵的默认值：它会让一次测试运行在仓库里留下快照，
    而下次运行读到的是一份上一轮的库——症状是"检索结果多出来几条"。
    """
    default = make_client()
    payload = stats(default)
    assert payload["info"]["backend"] == "flat"
    assert payload["info"]["persistent"] is False
    assert payload["info"]["location"] == ""
    assert settings.vector_persist_path == ""
