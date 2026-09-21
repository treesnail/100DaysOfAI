"""day065 ``/indexing/*`` 七个端点的单元测试.

全部离线、确定性、零网络：注入 ``FlatVectorStore``（零可选依赖的参照实现）与
``TableEmbedding``（按文本查表的确定性编码器），因此 ``encoded`` / ``batches`` /
``written`` / ``reuse_ratio`` 这些数字都是**手算得出来的**，不是"跑一遍看看"。

除了"七个端点各自返回什么"，这里钉住四件比 200 更重要的事：

1. **``indexing_dir`` 缺省为空串 = 明确不落盘**：缺省 app 的版本表与备份表
   都没有路径（``test_default_app_keeps_versions_and_backups_in_memory``），
   而 ``/indexing/backup`` 在缺省实例上返回 **400**（不是 500，也不偷偷建目录）
   ——这是本组最硬的一条约定，因为它的失效方式是"跑一次测试就在仓库里
   留下一个 ``backups``/``data/index`` 目录"；
2. **``/indexing/plan`` 只读**：调用之后库的条数与版本表的条数都必须原样
   （``test_plan_does_not_touch_the_store_or_the_version_table``）。计划是"看"，
   不是"做"——把两者合成一个端点，"这次要花多少钱"就永远要先付出去才知道；
3. **写入只活在这个 app 实例里**：另建一个 app，它的库是空的、版本表是空的、
   ``last_report`` 是 ``None``（``test_build_does_not_leak_into_another_app_instance``）；
4. **回滚只改版本指针**：``/indexing/rollback`` 之后库里的条数一条都不变
   （``test_rollback_moves_the_pointer_without_touching_the_store``）。

样本的形状与 day064 的 ``tests/vectorstore_samples.knowledge_records()`` 逐键一致
（``doc_id`` / ``source`` / ``text`` / ``metadata``），因此端点测试与
``/vectorstore/*`` 的测试用的是**同一份输入契约**——两份各造一套样本，
会让"两个端点组能不能接上"这件事永远没被验证过。这里自带一份而不是 import
那份样本，是因为本文件要造**第二版内容**（改一条记录）来演示增量与版本，
而共享样本是只读的。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from smart_research_agent.api.app import create_app
from smart_research_agent.config import settings
from smart_research_agent.vectorstore import FlatVectorStore
from tests.vectorstore_samples import TableEmbedding

#: 样本向量维度。取 8 与 day064 的样本一致：期望值能手算。
DIM = 8

#: 四条记录的 id（形态取自 day062 的 ``chunk_id``：16 位十六进制）。
RECORD_IDS: tuple[str, ...] = (
    "4a6680cde33da45e",
    "90a20c398e18fb0b",
    "1c9f4b7a2e6d8035",
    "7b3e0d5c9a14f286",
)

#: 四条记录的面包屑与正文（``(heading_path, text)``）.
TEXTS: tuple[tuple[str, str], ...] = (
    ("检索手册 > 分块", "固定长度分块每 320 个字符切一刀，重叠 48 个字符。"),
    ("检索手册 > 检索", "默认返回前 5 条，相似度低于 0.35 的直接丢掉。"),
    ("检索手册 > 成本", "一次全量重建索引大约 12 万条记录，每条 320 token。"),
    ("检索手册 > 版本", "版本号由内容摘要决定，时间戳不进版本号。"),
)

#: 四条记录的向量（写成手算得出来的定值，向量表由它构成）。
VECTORS: tuple[tuple[float, ...], ...] = (
    (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.8, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
)

#: "第二版"里被改掉的那一条的下标。**只改一条**是刻意的：增量构建的
#: ``updated == 1`` / ``reuse_ratio == 0.75`` 因此都是手算得出来的数。
REVISED_INDEX = 2


def fingerprint_of(text: str) -> str:
    """一段文本的内容身份（16 位十六进制，形态与 day062 的 ``content_id`` 相同）.

    这里刻意不 import ``documents.types.content_id``：本文件要的是"文本变了、
    指纹就变"这一条**可读的**性质，而不是复述那个函数的实现。
    ``planner.entry_from_record`` 只把 ``metadata["fingerprint"]`` 当作给定的值
    使用，因此任何稳定的 16 位十六进制都能表达"内容身份"。
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def make_records(*, revised: bool = False) -> list[dict]:
    """造一批 day062 形状的记录；``revised=True`` 时把第 3 条改掉.

    改一条记录要同时改三处，**缺一处就会漏判**，这正是 day065 那一课的重点：

    ```text
    text                 原文 → 影响 fingerprint（内容身份）
    metadata.retrieval_text  编码视图（面包屑 + 正文）→ 影响 vector_key（该不该复用向量）
    metadata.fingerprint 内容身份本身
    ```
    """
    records: list[dict] = []
    for index, (record_id, (heading, text)) in enumerate(zip(RECORD_IDS, TEXTS)):
        if revised and index == REVISED_INDEX:
            heading = f"{heading} > 修订"
            text = f"{text}（这一句是新加的）"
        records.append(
            {
                "doc_id": record_id,
                "source": f"docs/{record_id}.md",
                "text": text,
                "metadata": {
                    "parent_doc_id": "7feafd1c95ab3577",
                    "strategy": "recursive",
                    "index": str(index),
                    "token_count": str(60 + index),
                    "token_measurer": "chars",
                    "heading_path": heading,
                    "fingerprint": fingerprint_of(text),
                    "retrieval_text": f"{heading}\n{text}",
                    "oversized": "false",
                },
            }
        )
    return records


def api_embedding() -> TableEmbedding:
    """注入 ``app.state.embedding`` 的查表编码器（两版的检索视图都在表里）.

    两张视图都要注册：编码的是 ``retrieval_text``（day064 的 ``embedding_text``
    优先取它），只注册正文会让第二版的记录落到兜底向量上——那样写出来的断言
    测的是"兜底向量长什么样"，而不是样本的语义。
    """
    table: dict[str, tuple[float, ...]] = {}
    for records in (make_records(), make_records(revised=True)):
        for index, record in enumerate(records):
            table[record["metadata"]["retrieval_text"]] = VECTORS[index]
    return TableEmbedding(table, dimension=DIM)


def make_client(
    *,
    vector_store: FlatVectorStore | None = None,
    embedding: TableEmbedding | None = None,
    indexing_dir: str = "",
) -> TestClient:
    """按需建一个客户端（``indexing_dir`` 缺省空串 = 不落盘）."""
    return TestClient(
        create_app(
            embedding=embedding or api_embedding(),
            vector_store=vector_store if vector_store is not None else FlatVectorStore(),
            indexing_dir=indexing_dir,
        )
    )


@pytest.fixture
def client() -> TestClient:
    """缺省客户端：注入 flat 后端 + 查表编码器，**不落盘**（indexing_dir=""）."""
    return make_client()


# --------------------------------------------------------------------------- #
# 调用助手：每个端点一个，顺便把"这一步必须成功"钉住
# --------------------------------------------------------------------------- #


def status(client: TestClient) -> dict:
    """读一次索引状态（``/indexing/status``）."""
    response = client.get("/indexing/status")
    assert response.status_code == 200
    return response.json()


def plan(client: TestClient, records: list[dict] | None = None, **body: object) -> dict:
    """算一次差集；``records`` 缺省用第一版."""
    payload: dict = {"records": make_records() if records is None else records}
    payload.update(body)
    response = client.post("/indexing/plan", json=payload)
    assert response.status_code == 200
    return response.json()


def build(
    client: TestClient, records: list[dict] | None = None, **body: object
):
    """发一次构建请求（不替调用方断言状态码）."""
    payload: dict = {"records": make_records() if records is None else records}
    payload.update(body)
    return client.post("/indexing/build", json=payload)


def build_ok(
    client: TestClient, records: list[dict] | None = None, **body: object
) -> dict:
    """构建并断言成功（后续用例的前置条件）."""
    response = build(client, records, **body)
    assert response.status_code == 200, response.text
    return response.json()


def versions(client: TestClient) -> dict:
    """读一次版本历史（``/indexing/versions``）."""
    response = client.get("/indexing/versions")
    assert response.status_code == 200
    return response.json()


def verify(client: TestClient, **body: object) -> dict:
    """发一次体检请求并断言成功."""
    response = client.post("/indexing/verify", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def backup(client: TestClient, **body: object) -> dict:
    """发一次备份请求并断言成功."""
    response = client.post("/indexing/backup", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def rollback(client: TestClient, **body: object) -> dict:
    """发一次回滚请求并断言成功."""
    response = client.post("/indexing/rollback", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def delete_records(client: TestClient, **body: object):
    """借 ``/vectorstore/records`` 人为破坏库（制造"清单与库不一致"）."""
    return client.request("DELETE", "/vectorstore/records", json=body)


# --------------------------------------------------------------------------- #
# GET /indexing/status
# --------------------------------------------------------------------------- #

#: ``IndexingPipeline.stats()`` 的九个键。**写死在这里**：少了任何一个，
#: 客户端就得靠"这个字段有时在有时不在"来写代码——而那种代码的失败方式
#: 是运行时的一个 KeyError，不是一个可读的契约变化。
STATS_KEYS = {
    "backend",
    "metric",
    "dimension",
    "count",
    "identity",
    "manifest",
    "versions",
    "backups",
    "cache",
}


def test_status_reports_the_documented_keys_on_a_fresh_instance(
    client: TestClient,
) -> None:
    """未构建过：九个键齐全、``manifest`` 为 ``None``、``last_report`` 为 ``None``.

    三处**不合并**的字段在这里各显一次：

    ```text
    dimension=0        库还是空的（维度未定）；编码器的维度在 identity.dimension=8
    manifest=None      "还没构建过"与"构建了一个空的"是两件事
    last_report=None   最近一次构建报告；它挂在 app 实例上，不是模块级全局
    ```
    """
    payload = status(client)
    assert set(payload["stats"]) == STATS_KEYS
    stats = payload["stats"]
    assert stats["backend"] == "flat"
    assert stats["metric"] == "cosine"
    assert stats["dimension"] == 0
    assert stats["count"] == 0
    assert stats["manifest"] is None
    assert stats["versions"] == 0
    assert stats["backups"] == 0
    assert stats["identity"] == {
        "provider": "TableEmbedding",
        "model": "default",
        "dimension": DIM,
        "key": stats["identity"]["key"],
    }
    assert stats["cache"]["size"] == 0
    assert payload["last_report"] is None
    assert payload["indexing_dir"] == ""
    assert "（不落盘）" in payload["summary"]


def test_status_manifest_matches_the_version_returned_by_build(
    client: TestClient,
) -> None:
    """构建之后：``manifest.version_id`` 必须等于 ``build`` 返回的 ``version_id``.

    这条断言把两个端点钉在同一份清单上：``status`` 报的"当前是哪一版"与
    ``build`` 报的"这次建的是哪一版"若不是同一个版本号，增量构建的"上一版"
    就无从谈起。
    """
    built = build_ok(client)
    payload = status(client)
    assert payload["stats"]["count"] == len(RECORD_IDS)
    assert payload["stats"]["dimension"] == DIM
    assert payload["stats"]["manifest"]["version_id"] == built["version_id"]
    # 清单以"不含条目"的形式出现：entries 不进响应体（十万条会接近 10 MB）
    assert "entries" not in payload["stats"]["manifest"]
    assert payload["stats"]["manifest"]["count"] == len(RECORD_IDS)
    assert payload["stats"]["versions"] == 1
    assert payload["last_report"]["written"] == len(RECORD_IDS)


def test_status_is_json_serializable(client: TestClient) -> None:
    """响应体必须能被 ``json.dumps`` 直接序列化（不把 entries / 自定义对象塞进去）."""
    build_ok(client)
    payload = status(client)
    assert json.loads(json.dumps(payload)) == payload


# --------------------------------------------------------------------------- #
# POST /indexing/plan
# --------------------------------------------------------------------------- #


def test_plan_on_a_first_build_is_all_added(client: TestClient) -> None:
    """首次构建：四条全是 ``added``，没有可复用的条目."""
    payload = plan(client)
    assert payload["plan"]["added"] == len(RECORD_IDS)
    assert payload["plan"]["added_ids"] == sorted(RECORD_IDS)
    assert payload["plan"]["updated"] == 0
    assert payload["plan"]["unchanged"] == 0
    assert payload["plan"]["reuse_ratio"] == 0.0
    assert payload["parent_version"] == ""
    # 没有上一版时不换挡：变更比例没有分母（全部是新增）
    assert payload["mode"] == "incremental"
    assert payload["switched"] is False
    assert payload["reason"].startswith("首次构建")
    assert payload["identity"]["provider"] == "TableEmbedding"


def test_plan_replay_is_all_unchanged_with_full_reuse(client: TestClient) -> None:
    """重放同一批：四条全 ``unchanged`` 且 ``reuse_ratio == 1.0``.

    这是"这次不需要任何编码调用"的唯一证据——而它之所以成立，是因为
    ``fingerprint``（内容身份）与 ``vector_key``（该不该复用向量）**都没变**。
    """
    built = build_ok(client)
    payload = plan(client)
    assert payload["plan"]["unchanged"] == len(RECORD_IDS)
    assert payload["plan"]["added"] == 0
    assert payload["plan"]["updated"] == 0
    assert payload["plan"]["removed"] == 0
    assert payload["plan"]["reuse_ratio"] == 1.0
    assert payload["parent_version"] == built["version_id"]
    assert "没有变化" in payload["reason"]
    assert payload["switched"] is False


def test_plan_does_not_touch_the_store_or_the_version_table(
    client: TestClient,
) -> None:
    """**计划是只读的**：调用之后库的条数与版本表的条数都必须原样.

    两条都要断言，因为两种"手滑"都可能发生：plan 顺手把记录写进库
    （那是 build 的活），或顺手登记一版（那是版本表的活）。
    """
    empty = status(client)
    assert empty["stats"]["count"] == 0
    assert empty["stats"]["versions"] == 0
    plan(client)
    after_plan = status(client)
    assert after_plan["stats"]["count"] == 0
    assert after_plan["stats"]["versions"] == 0
    assert versions(client)["history"] == []

    built = build_ok(client)
    before = status(client)
    plan(client, make_records(revised=True))
    after = status(client)
    assert after["stats"]["count"] == before["stats"]["count"] == len(RECORD_IDS)
    assert after["stats"]["versions"] == before["stats"]["versions"] == 1
    assert after["stats"]["manifest"]["version_id"] == built["version_id"]
    assert versions(client)["current"] == built["version_id"]


def test_plan_marks_only_the_changed_record_as_updated(client: TestClient) -> None:
    """改一条记录 → 只有它落进 ``updated``，其余三条复用（``reuse_ratio=0.75``）."""
    build_ok(client)
    payload = plan(client, make_records(revised=True))
    assert payload["plan"]["updated_ids"] == [RECORD_IDS[REVISED_INDEX]]
    assert payload["plan"]["unchanged"] == len(RECORD_IDS) - 1
    assert payload["plan"]["reuse_ratio"] == 0.75
    assert "增量更新" in payload["reason"]


def test_plan_previews_the_threshold_switch(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """把阈值调到 0.1：25% 的变更比例超线，计划预告会切到 ``full``.

    它演示的是"增量不是永远更省"：变更比例很高时，逐条 upsert 加逐条 delete
    的开销反而高于清空重写。真正的换挡在构建时执行，计划只是**预告**。
    """
    build_ok(client)
    monkeypatch.setattr(settings, "indexing_full_rebuild_threshold", 0.1)
    payload = plan(client, make_records(revised=True))
    assert payload["threshold"] == 0.1
    assert payload["requested_mode"] == "incremental"
    assert payload["mode"] == "full"
    assert payload["switched"] is True
    assert "超过阈值" in payload["summary"]


def test_plan_rejects_an_empty_batch(client: TestClient) -> None:
    """空批次 → 400：一次"什么都不变"的计划会被读成"数据没问题"."""
    response = client.post("/indexing/plan", json={"records": []})
    assert response.status_code == 400
    assert "records 不能为空" in response.json()["detail"]


def test_plan_rejects_an_unknown_mode(client: TestClient) -> None:
    """非法 ``mode`` → 400，且消息里列出 ``full`` 与 ``incremental``（照它就能改对）."""
    response = client.post(
        "/indexing/plan", json={"records": make_records(), "mode": "rebuild"}
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "full" in detail
    assert "incremental" in detail
    assert "rebuild" in detail


# --------------------------------------------------------------------------- #
# POST /indexing/build
# --------------------------------------------------------------------------- #


def test_build_writes_and_encodes_every_record(client: TestClient) -> None:
    """首次构建：``written=4`` 且 ``encoded=4``（一批装下全部四条）."""
    payload = build_ok(client)
    report = payload["report"]
    assert report["mode"] == "incremental"
    assert report["seen"] == len(RECORD_IDS)
    assert report["written"] == len(RECORD_IDS)
    assert report["unchanged"] == 0
    assert report["encoded"] == len(RECORD_IDS)
    assert report["cache_hits"] == 0
    assert report["batches"] == 1
    assert report["failed"] == 0
    assert report["ok"] is True
    assert report["parent_version"] == ""
    assert report["dimension"] == DIM
    assert report["provider"] == "TableEmbedding"
    assert payload["count"] == len(RECORD_IDS)
    assert payload["version_id"] == report["version_id"]
    assert len(payload["version_id"]) == 16
    assert payload["manifest"]["count"] == len(RECORD_IDS)
    assert json.loads(json.dumps(report)) == report


def test_build_replay_writes_and_encodes_nothing(client: TestClient) -> None:
    """重放同一批：``written=0`` 且 ``encoded=0``——库没变，钱也没花.

    这是 day065 相对 day064 的那个改进：day064 的逐条写入在重放时
    ``embedding_calls`` 仍然是 6（库没变、钱照花），而这里因为差集为空，
    **一次编码调用都没有发起**（``batches=0``）。
    """
    build_ok(client)
    report = build_ok(client)["report"]
    assert report["written"] == 0
    assert report["unchanged"] == len(RECORD_IDS)
    assert report["encoded"] == 0
    assert report["batches"] == 0
    assert report["cache_hits"] == 0
    assert report["reuse_ratio"] == 1.0


def test_build_explicit_full_mode_is_honoured(client: TestClient) -> None:
    """``mode="full"`` 显式生效：库被重写一遍，但缓存让 ``encoded=0``.

    "把库重写一遍"与"把向量重算一遍"是两件事——全量重建仍然照用缓存。
    """
    build_ok(client)
    report = build_ok(client, mode="full")["report"]
    assert report["mode"] == "full"
    assert report["written"] == len(RECORD_IDS)
    # 全量模式下 unchanged 恒为 0：clear() 之后库里没有"没动过"的东西
    assert report["unchanged"] == 0
    assert report["encoded"] == 0
    assert report["cache_hits"] == len(RECORD_IDS)
    assert report["batches"] == 0
    assert report["reuse_ratio"] == 1.0


def test_build_incremental_updates_only_the_changed_record(client: TestClient) -> None:
    """增量：改一条 → ``written=1`` / ``encoded=1`` / ``unchanged=3``."""
    build_ok(client)
    report = build_ok(client, make_records(revised=True))["report"]
    assert report["mode"] == "incremental"
    assert report["written"] == 1
    assert report["encoded"] == 1
    assert report["unchanged"] == len(RECORD_IDS) - 1
    assert report["reuse_ratio"] == 0.75
    assert report["parent_version"] != ""
    assert report["version_id"] != report["parent_version"]


def test_build_rejects_an_unknown_mode_with_the_choices(client: TestClient) -> None:
    """非法 ``mode`` → 400，消息里带 ``full`` 与 ``incremental``（不做别名宽容）."""
    response = build(client, mode="nope")
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "full" in detail
    assert "incremental" in detail
    assert "nope" in detail


def test_build_rejects_an_empty_batch(client: TestClient) -> None:
    """空批次 → 400（增量模式下它会删掉上一版的全部条目——一次清库）."""
    response = build(client, records=[])
    assert response.status_code == 400
    assert "records 不能为空" in response.json()["detail"]


def test_build_remembers_the_last_report_of_this_instance(client: TestClient) -> None:
    """最近一次构建报告记在 ``app.state.indexing_last_report`` 上并原样带回."""
    first = build_ok(client)
    assert status(client)["last_report"]["version_id"] == first["version_id"]
    second = build_ok(client, make_records(revised=True))
    remembered = status(client)["last_report"]
    assert remembered["version_id"] == second["version_id"]
    assert remembered["written"] == 1


def test_build_rejects_a_bad_request_shape(client: TestClient) -> None:
    """字段类型不对是 422（请求形状问题），与业务规则的 400 分开."""
    response = client.post("/indexing/build", json={"records": "四条"})
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# GET /indexing/versions
# --------------------------------------------------------------------------- #


def test_versions_is_empty_before_any_build(client: TestClient) -> None:
    """"还没有历史"是一个合法状态：空列表 + 空指针，而不是 404."""
    payload = versions(client)
    assert payload["history"] == []
    assert payload["current"] == ""
    assert payload["count"] == 0
    assert payload["lineage"] == []


def test_versions_lists_two_versions_and_current_is_the_second(
    client: TestClient,
) -> None:
    """构建两版（内容不同）→ 两条历史，``current`` 是第二版.

    两版必须是**不同的内容**：版本号由内容摘要决定，同一批内容重建两次
    会算出同一个版本号（``register`` 幂等），因此"构建两次"不等于"两版"。
    """
    first = build_ok(client)
    second = build_ok(client, make_records(revised=True))
    assert first["version_id"] != second["version_id"]
    payload = versions(client)
    assert payload["count"] == 2
    assert [item["version_id"] for item in payload["history"]] == [
        first["version_id"],
        second["version_id"],
    ]
    assert payload["current"] == second["version_id"]
    assert payload["lineage"] == [first["version_id"], second["version_id"]]
    for item in payload["history"]:
        assert item["summary_line"].startswith("版本 ")


# --------------------------------------------------------------------------- #
# POST /indexing/verify
# --------------------------------------------------------------------------- #


def test_verify_is_ok_right_after_a_build(client: TestClient) -> None:
    """刚构建完的清单与库必然对得上：``ok`` 为真、``problems`` 为空."""
    built = build_ok(client)
    payload = verify(client)
    assert payload["ok"] is True
    assert payload["problems"] == []
    assert payload["version_id"] == built["version_id"]
    assert payload["source"] == "current"
    assert payload["checks"]["missing_in_store"] == []
    assert payload["checks"]["orphan_in_store"] == []
    assert payload["checks"]["dimension_match"] is True


def test_verify_accepts_an_explicit_version_id(client: TestClient) -> None:
    """显式给 ``version_id``：用的是版本表里那一版（``source="version"``）."""
    built = build_ok(client)
    payload = verify(client, version_id=built["version_id"])
    assert payload["ok"] is True
    assert payload["source"] == "version"
    assert payload["version_id"] == built["version_id"]


def test_verify_detects_a_record_removed_from_the_store(client: TestClient) -> None:
    """人为从库里删一条 → ``ok`` 为假，``problems`` 里给出具体 id 与下一步.

    **没有**这条用例，``verify`` 就是一个永远返回 ``ok=True`` 的装饰品：
    它存在的唯一理由是让"库与清单对不上"变成当场可见的事件。
    """
    built = build_ok(client)
    removed = delete_records(client, ids=[RECORD_IDS[0]])
    assert removed.status_code == 200
    assert removed.json()["removed"] == 1
    payload = verify(client, version_id=built["version_id"])
    assert payload["ok"] is False
    assert payload["checks"]["missing_in_store"] == [RECORD_IDS[0]]
    assert len(payload["problems"]) == 1
    assert RECORD_IDS[0] in payload["problems"][0]
    assert "在库里找不到" in payload["problems"][0]


def test_verify_rejects_an_unknown_version_id(client: TestClient) -> None:
    """版本表里没有这一版 → 400（而不是"体检通过"或 500）."""
    build_ok(client)
    response = client.post("/indexing/verify", json={"version_id": "f" * 16})
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "版本表里没有这一版" in detail
    assert "f" * 16 in detail


def test_verify_falls_back_to_the_store_when_nothing_was_built(
    client: TestClient,
) -> None:
    """还没构建过 → 按库的现状现算一份清单（``source="store"``）.

    注意这时 ``ok`` 是**假**的：现算的清单带着编码器的身份（8 维），
    而空库的维度是"未定"（0）——维度不符正是"换过编码器但没重建"这个
    脏状态的样子。这条断言刻意留着：把"空库体检通过"当成正常，
    会让维度护栏在下一次真正写库时才发现问题。
    """
    payload = verify(client)
    assert payload["source"] == "store"
    assert len(payload["version_id"]) == 16
    assert payload["ok"] is False
    assert payload["checks"]["dimension_match"] is False
    assert any("维度不一致" in problem for problem in payload["problems"])


# --------------------------------------------------------------------------- #
# POST /indexing/backup
# --------------------------------------------------------------------------- #


def test_backup_creates_a_snapshot_under_the_indexing_dir(tmp_path: Path) -> None:
    """``indexing_dir=tmp_path`` → 200，快照落在 ``<dir>/backups/<backup_id>/``.

    内存后端没有库文件可备份，因此 ``files`` 是空列表、``note`` 里说明
    "本次只备份了清单"——这与"源文件缺失被静默跳过"是两件事：
    后者是假备份，前者是如实告知。
    """
    client = make_client(indexing_dir=str(tmp_path))
    built = build_ok(client)
    payload = backup(client, reason="演示一次手动备份")
    assert payload["backup_id"].startswith(built["version_id"])
    assert payload["version_id"] == built["version_id"]
    assert payload["files"] == []
    assert payload["size_bytes"] > 0
    assert payload["keep"] == settings.indexing_backups_keep
    assert "只备份了清单" in payload["note"]
    assert payload["backup"]["reason"] == "演示一次手动备份"
    snapshot = tmp_path / "backups" / payload["backup_id"]
    assert (snapshot / "manifest.json").is_file()
    assert (tmp_path / "backups" / "backups.jsonl").is_file()


def test_backup_copies_the_store_file_when_the_backend_can_persist(
    tmp_path: Path,
) -> None:
    """后端能落盘时快照里会带上库文件（``files == ["store.json"]``，``note`` 为空）."""
    store = FlatVectorStore(path=str(tmp_path / "store.json"))
    client = make_client(vector_store=store, indexing_dir=str(tmp_path))
    build_ok(client)
    payload = backup(client)
    assert payload["files"] == ["store.json"]
    assert payload["note"] == ""
    assert (tmp_path / "backups" / payload["backup_id"] / "store.json").is_file()


def test_backup_without_an_indexing_dir_returns_400(client: TestClient) -> None:
    """``indexing_dir=""`` → **400**（不是 500，也不在进程当前目录建目录）.

    理由与 day064 的 ``vector_persist_path = ""`` 逐字相同：备份必须落在库
    文件**之外**的目录上，而"没有配置索引目录"时它连往哪儿写都答不出来。
    这里顺带断言备份表本身没有路径——400 是**提前**拦下来的，
    而不是写失败之后才报的。
    """
    build_ok(client)
    assert client.app.state.indexing_backups.path == ""
    response = client.post("/indexing/backup", json={})
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "索引目录" in detail
    assert "indexing_dir" in detail


# --------------------------------------------------------------------------- #
# POST /indexing/rollback
# --------------------------------------------------------------------------- #


def test_rollback_moves_the_pointer_without_touching_the_store(
    client: TestClient,
) -> None:
    """构建两版后 ``steps=1`` → ``current`` 回到第一版，而**库里的数据一条没变**.

    "回滚只改版本指针"这句话必须能被断言，否则它只是一句承诺：
    回滚是运维动作，重建要另调 ``/indexing/build``。
    """
    first = build_ok(client)
    second = build_ok(client, make_records(revised=True))
    before = status(client)["stats"]["count"]
    payload = rollback(client, steps=1)
    assert payload["previous"] == second["version_id"]
    assert payload["current"] == first["version_id"]
    assert payload["manifest"]["version_id"] == first["version_id"]
    assert "不自动重建索引" in payload["note"]
    assert versions(client)["current"] == first["version_id"]
    # 库与数据都没动：条数不变，版本历史仍然是两条
    assert status(client)["stats"]["count"] == before == len(RECORD_IDS)
    assert versions(client)["count"] == 2
    assert client.get("/vectorstore/stats").json()["info"]["count"] == len(RECORD_IDS)


def test_rollback_defaults_to_one_step(client: TestClient) -> None:
    """缺省 ``steps=1``：请求体可以什么都不给."""
    first = build_ok(client)
    build_ok(client, make_records(revised=True))
    payload = rollback(client)
    assert payload["steps"] == 1
    assert payload["current"] == first["version_id"]


def test_rollback_rejects_zero_steps(client: TestClient) -> None:
    """``steps=0`` → 400（"回退 0 步"不是一次回滚）."""
    build_ok(client)
    response = client.post("/indexing/rollback", json={"steps": 0})
    assert response.status_code == 400
    assert "必须 >= 1" in response.json()["detail"]


def test_rollback_rejects_a_lineage_that_is_too_short(client: TestClient) -> None:
    """还没采纳过任何版本 → 400（消息里说明没有"上一版"可退）."""
    response = client.post("/indexing/rollback", json={})
    assert response.status_code == 400
    assert "还没有采纳任何版本" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# 两个写入隔离与契约收口
# --------------------------------------------------------------------------- #


def test_build_does_not_leak_into_another_app_instance(client: TestClient) -> None:
    """写入只活在这个 app 实例里：另建一个 app 看到的是空库、空版本表.

    这一条同时否掉了三种实现：把库放在模块级全局上（两个实例共享数据）、
    把版本表放进 ``settings``、把最近一次报告放进模块级变量——
    它们的失效方式都是"另一个测试偶发失败"，最难查。
    """
    build_ok(client)
    other = make_client()
    payload = status(other)
    assert payload["stats"]["count"] == 0
    assert payload["stats"]["versions"] == 0
    assert payload["stats"]["manifest"] is None
    assert payload["last_report"] is None
    assert versions(other)["history"] == []
    # 原实例的数据仍然在（不是被另一个实例清掉了）
    assert status(client)["stats"]["count"] == len(RECORD_IDS)


def test_default_app_keeps_versions_and_backups_in_memory() -> None:
    """``create_app()`` 的缺省装配**不落盘**：版本表与备份表都没有路径.

    ``settings.indexing_dir`` 的缺省是 ``data/index``，而本参数刻意不取它：
    一个"只想看看状态"的缺省 app 不该把版本表与备份指向仓库内的目录。
    **要落盘请显式传 ``indexing_dir=settings.indexing_dir``。**
    """
    default = create_app()
    assert default.state.indexing_dir == ""
    assert default.state.indexing_versions.path == ""
    assert default.state.indexing_backups.path == ""
    # 配置里那个值仍然存在——两者分开，正是为了"缺省不落盘"可被断言
    assert settings.indexing_dir == "data/index"
    assert status(TestClient(default))["indexing_dir"] == ""


def test_indexing_paths_are_published_in_openapi(client: TestClient) -> None:
    """七个端点与它们的请求/响应模型都进了 OpenAPI（客户端代码生成与 /docs 靠它）."""
    document = client.get("/openapi.json").json()
    assert {
        "/indexing/status",
        "/indexing/plan",
        "/indexing/build",
        "/indexing/versions",
        "/indexing/verify",
        "/indexing/backup",
        "/indexing/rollback",
    } <= set(document["paths"])
    schemas = document["components"]["schemas"]
    for model in (
        "IndexingStatusResponse",
        "IndexingPlanRequest",
        "IndexingPlanResponse",
        "IndexingBuildRequest",
        "IndexingBuildResponse",
        "IndexingVersionsResponse",
        "IndexingVerifyRequest",
        "IndexingVerifyResponse",
        "IndexingBackupRequest",
        "IndexingBackupResponse",
        "IndexingRollbackRequest",
        "IndexingRollbackResponse",
    ):
        assert model in schemas
