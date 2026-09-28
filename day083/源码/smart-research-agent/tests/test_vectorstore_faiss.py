"""``faiss_backend`` 的单元测试：**注入假 faiss 模块**，全程离线、无真库（M6-D3）.

本机没有（也不该为了跑测试去装）``faiss``，因此这里唯一被允许的"依赖注入"
是 ``tests/faiss_fakes.py`` —— 它用 numpy 做精确检索，并把真库的几条硬行为
（float32/int64、``IndexIDMap2.add`` 失败、``-1`` 填充、``+Inf`` 哨兵）写死。

测试分五组，对应五类"会悄悄错掉"的事：

```text
对账    与 brute_force_top 逐条比 —— 排序错、换算错、过滤少返回都在这里露馅
过滤    返回条数必须等于"满足条件的条数"，而不是"先取 top-k 再过滤"
id      哈希映射稳定、值域正确、撞号必须报错
维度    FAISS 建索引前必需维度；写入维度不一致要被拦
落盘    索引 + 旁挂记录两个文件必须成对，且载入时逐项核对
```

数值断言全部写成具体值（或与参照实现逐条比较），不用 ``is not None``
这类软断言——**不会变红的断言等于没有断言**。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from smart_research_agent.vectorstore import faiss_backend
from smart_research_agent.vectorstore.errors import (
    BackendUnavailable,
    RecordError,
    VectorError,
    VectorStoreError,
)
from smart_research_agent.vectorstore.faiss_backend import (
    FAISS_ID_NOT_FOUND,
    INT64_ID_MODULUS,
    RECORDS_SUFFIX,
    FaissVectorStore,
    from_int64_id,
    to_int64_id,
)
from smart_research_agent.vectorstore.types import make_record
from tests.faiss_fakes import (
    FakeFaissModuleMissingIDSelectorBatch,
    FakeFaissModuleMissingIndexFlat,
    FakeIDSelectorBatch,
    FakeIndexFlatIP,
    FakeIndexFlatL2,
    FakeIndexIDMap2,
    make_fake_faiss_module,
)
from tests.vectorstore_samples import (
    RECORD_IDS,
    SAMPLE_QUERIES,
    VECTOR_DIMENSION,
    brute_force_top,
    sample_records,
)

#: 两个查询（``q_orthogonal`` 与所有样本向量正交，用来测 0 分的边界，本文件不用）.
ACCOUNTED_QUERIES = ("q_axis", "q_tilt")


def empty_store(metric: str = "cosine") -> FaissVectorStore:
    """一个空的 faiss 库（每个测试各拿一个假模块，避免状态串味）."""
    return FaissVectorStore(
        metric=metric,
        dimension=VECTOR_DIMENSION,
        faiss_module=make_fake_faiss_module(),
    )


def loaded_store(metric: str = "cosine") -> FaissVectorStore:
    """装了六条样本记录的 faiss 库（写入走 ``upsert``，与真实用法一致）."""
    store = empty_store(metric)
    store.upsert(sample_records(metric=metric))
    return store


def reference_ids(metric: str, query_name: str, top_k: int) -> list[str]:
    """参照实现给出的 id 顺序（**独立的第二份计算**，见 samples 的说明）."""
    expected = brute_force_top(
        sample_records(metric=metric), list(SAMPLE_QUERIES[query_name]), metric, top_k
    )
    return [record_id for record_id, _ in expected]


class MinimalIndex:
    """一个"最小可用"的索引替身：只有 ``d`` / ``ntotal`` / ``id_map``，没有 ``metric_type``.

    真库的索引都有 ``metric_type``，但适配器对"接管的索引"只应要求
    "能表达外部 id"。这个替身用来覆盖那条"缺少 ``metric_type`` 也不该误报"
    的防御性分支——**它证明的是"我们没有把真库没承诺的东西当成必需项"**。
    """

    def __init__(self, dimension: int) -> None:
        self.d = dimension
        self.ntotal = 0

    @property
    def id_map(self) -> np.ndarray:
        """行号 → 外部 id 的映射（空索引所以是空数组）."""
        return np.zeros(0, dtype=np.int64)


# --------------------------------------------------------------------------- #
# 生命周期：写入 / 查询 / 取值 / 删除
# --------------------------------------------------------------------------- #


def test_injected_fake_module_drives_the_whole_lifecycle() -> None:
    """注入假模块后，六个原语与公共行为串起来跑一遍（写出 / 查询 / 删除 / 统计）."""
    store = empty_store()
    records = sample_records(metric="cosine")
    report = store.upsert(records)

    assert report.added == 6
    assert report.updated == 0
    assert report.unchanged == 0
    assert report.written == 6
    assert store.count() == 6
    assert store.ids() == sorted(RECORD_IDS)
    assert store.get(RECORD_IDS[0]).text == records[0].text
    assert store.get(RECORD_IDS[0]).vector == records[0].vector
    assert store.get("不存在的记录 id") is None

    result = store.query(list(SAMPLE_QUERIES["q_axis"]), top_k=2)
    assert result.ids() == [RECORD_IDS[0], RECORD_IDS[3]]
    assert result.scores == pytest.approx([1.0, 1.0])
    assert result.candidates == 6
    assert result.filter_applied is False

    info = store.info()
    assert info.backend == "faiss"
    assert info.metric == "cosine"
    assert info.dimension == VECTOR_DIMENSION
    assert info.count == 6
    assert info.persistent is False
    assert info.extra["index_type"] == "IndexFlatIP"
    assert info.extra["index_ntotal"] == "6"
    assert info.extra["records"] == "6"
    assert info.extra["records_sidecar"] == RECORDS_SUFFIX
    assert info.extra["snapshot_version"] == "1"
    assert info.extra["id_range"] == f"1..{INT64_ID_MODULUS}"

    assert store.delete(ids=[RECORD_IDS[0]]) == 1
    assert store.count() == 5
    assert store._index.ntotal == 5
    assert store.describe_extra()["index_ntotal"] == "5"


def test_upsert_replay_reports_unchanged_and_touches_nothing() -> None:
    """重放同一批：六条全是 ``unchanged``，索引条数与记录表条数都不变."""
    store = loaded_store()
    report = store.upsert(sample_records(metric="cosine"))

    assert (report.added, report.updated, report.unchanged) == (0, 0, 6)
    assert report.total == 6
    assert report.changed == 0
    assert store.count() == 6
    assert store._index.ntotal == 6


def test_upsert_updated_vector_rewrites_the_index_in_place() -> None:
    """改一条向量 → ``updated=1``；索引被替换而不是追加（ntotal 仍是 6）."""
    store = loaded_store()
    records = sample_records(metric="cosine")
    changed = make_record(
        record_id=RECORD_IDS[2],
        vector=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        text="改成与查询完全同向",
        metadata=dict(records[2].metadata),
        metric="cosine",
    )

    report = store.upsert([changed])

    assert (report.added, report.updated, report.unchanged) == (0, 1, 0)
    assert store.count() == 6
    assert store._index.ntotal == 6
    assert store.get(RECORD_IDS[2]).text == "改成与查询完全同向"
    # 三条向量此时逐位相同（都是 e1），分数并列 1.0 → 名次由 id 升序决定
    assert store.query(list(SAMPLE_QUERIES["q_axis"]), top_k=3).ids() == [
        RECORD_IDS[2],
        RECORD_IDS[0],
        RECORD_IDS[3],
    ]


def test_add_conflict_rejects_the_whole_batch() -> None:
    """``add()`` 撞 id 时整体拒绝，**库与索引都不动**（不做部分写入）."""
    store = loaded_store()
    fresh = make_record(
        record_id="新增记录00000001",
        vector=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        text="这条本来应该被写进去",
        metric="cosine",
    )

    with pytest.raises(RecordError, match="撞上"):
        store.add([fresh, sample_records(metric="cosine")[0]])

    assert store.count() == 6
    assert store._index.ntotal == 6
    assert store.get("新增记录00000001") is None


def test_get_many_skips_missing_ids() -> None:
    """批量取记录跳过缺失项（顺序仍与入参一致）."""
    store = loaded_store()
    found = store.get_many([RECORD_IDS[0], "不存在的记录 id", RECORD_IDS[5]])

    assert [record.record_id for record in found] == [RECORD_IDS[0], RECORD_IDS[5]]


def test_clear_keeps_metric_and_dimension() -> None:
    """``clear()` 清空两个库，但度量与维度保留（便于复用同一个实例重建）."""
    store = loaded_store()
    store.clear()

    assert store.count() == 0
    assert store.ids() == []
    assert store._index.ntotal == 0
    assert store.metric == "cosine"
    assert store.dimension == VECTOR_DIMENSION

    store.upsert(sample_records(metric="cosine")[:2])
    assert store.count() == 2
    assert store._index.ntotal == 2


def test_delete_by_where_and_by_missing_id() -> None:
    """按 ``where`` 删除走记录表预筛；删不存在的 id 返回 0 而不是报错."""
    store = loaded_store()

    assert store.delete(where={"strategy": "recursive"}) == 2
    assert store.count() == 4
    assert store._index.ntotal == 4
    assert store.ids() == sorted(
        [RECORD_IDS[2], RECORD_IDS[3], RECORD_IDS[4], RECORD_IDS[5]]
    )

    assert store.delete(ids=["不存在的记录 id"]) == 0
    assert store.delete(ids=[]) == 0
    assert store.delete(where={"strategy": "不存在的策略"}) == 0


def test_remove_reconciliation_failure_is_reported() -> None:
    """索引与记录表被人为弄成不同步时，删除必须报错（而不是静默少删）."""
    store = loaded_store()
    key = np.asarray([to_int64_id(RECORD_IDS[0])], dtype=np.int64)
    # 绕过适配器直接动索引：模拟"有人手改了索引文件"这种不同步
    assert store._index.remove_ids(FakeIDSelectorBatch(key)) == 1

    with pytest.raises(VectorStoreError, match="删除对账失败"):
        store.delete(ids=[RECORD_IDS[0]])


def test_empty_store_query_and_empty_batch_write() -> None:
    """空库的两个退化路径：查询返回空结果（不报错）、空批次写入全 0."""
    store = empty_store()

    result = store.query(list(SAMPLE_QUERIES["q_axis"]), top_k=3)

    assert result.hits == ()
    assert result.count == 0
    assert result.candidates == 0
    assert result.filter_applied is False

    report = store.upsert([])

    assert (report.added, report.updated, report.unchanged) == (0, 0, 0)
    assert store.count() == 0
    assert store._index.ntotal == 0


def test_query_skips_index_entries_without_a_record() -> None:
    """索引里多出来的 id（没有对应记录）在读路径上被跳过，而不是让整次查询失败.

    这条对应用户必然会遇到的一种脏状态：有人绕过适配器直接往索引里塞了向量。
    读路径的选择是"跳过并继续"——一次查询不该因为一条脏数据整体失败。
    """
    store = loaded_store()
    orphan = np.asarray([[1.0] + [0.0] * 7], dtype="float32")
    store._index.add_with_ids(orphan, np.asarray([12345], dtype=np.int64))

    result = store.query(list(SAMPLE_QUERIES["q_axis"]), top_k=10)

    assert store._index.ntotal == 7
    assert result.count == 6
    assert result.ids() == reference_ids("cosine", "q_axis", 6)


# --------------------------------------------------------------------------- #
# 与参照实现对账 + 过滤的正确性
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("metric", ["cosine", "ip", "l2"])
@pytest.mark.parametrize("query_name", ACCOUNTED_QUERIES)
def test_query_matches_brute_force_reference(metric: str, query_name: str) -> None:
    """三种度量 × 两个查询：top-3 的 id 顺序与分数都与参照实现逐条相同.

    这一条同时盯住了三件事：索引类型选对（IP/L2）、FAISS 的 D 口径换算对
    （内积是相似度、L2 是距离）、排序规则与 ``sort_hits`` 一致（并列按 id 升序）。
    """
    store = loaded_store(metric)
    query = list(SAMPLE_QUERIES[query_name])
    expected = brute_force_top(sample_records(metric=metric), query, metric, 3)
    result = store.query(query, top_k=3)

    assert result.ids() == [record_id for record_id, _ in expected]
    assert result.scores == pytest.approx(
        [score for _, score in expected], rel=1e-5, abs=1e-4
    )
    assert result.metric == metric
    assert result.candidates == 6
    assert result.filter_applied is False


def test_where_filter_returns_every_matching_record_not_top_k_then_filter() -> None:
    """过滤的正确性：top-3 里有 2 条被筛掉时，仍要返回**候选集内**的前若干条.

    样本里 "最像的 3 条" 是 ``[r_a, r_d, r_c]``（id 见 ``RECORD_IDS``），
    而 ``strategy=structural`` 只命中 ``{r_d, r_f}``。因此：

    ```text
    先取 top-3 再过滤（错） → 只剩 r_d 一条
    先在候选里排序（对）     → [r_d, r_f]，条数等于满足条件的条数
    ```
    """
    store = loaded_store()
    top_three_before_filter = store.query(list(SAMPLE_QUERIES["q_axis"]), top_k=3).ids()
    assert top_three_before_filter == [RECORD_IDS[0], RECORD_IDS[3], RECORD_IDS[2]]

    result = store.query(
        list(SAMPLE_QUERIES["q_axis"]), top_k=3, where={"strategy": "structural"}
    )

    assert result.ids() == [RECORD_IDS[3], RECORD_IDS[5]]
    assert result.count == 2
    assert result.candidates == 2
    assert result.filter_applied is True
    assert all(hit.record.metadata["strategy"] == "structural" for hit in result.hits)
    # 被筛掉的 r_c 在半路，naive 的"先 top-k 再过滤"会把它算进候选而漏掉 r_f
    assert RECORD_IDS[2] not in result.ids()


def test_where_filter_with_more_matches_returns_exactly_top_k() -> None:
    """候选多于 top_k 时返回的正好是 top_k 条，且全部满足条件."""
    store = loaded_store()

    result = store.query(
        list(SAMPLE_QUERIES["q_axis"]), top_k=3, where={"oversized": False}
    )

    assert result.ids() == [RECORD_IDS[0], RECORD_IDS[3], RECORD_IDS[2]]
    assert result.count == 3
    assert result.candidates == 5
    assert result.filter_applied is True
    assert RECORD_IDS[5] not in result.ids()  # 唯一 oversized=True 的那条
    assert all(hit.record.metadata["oversized"] is False for hit in result.hits)


def test_where_filter_that_matches_nothing_returns_empty_result() -> None:
    """条件筛掉全部记录：hits 为空、``candidates=0``、``filter_applied=True``."""
    store = loaded_store()

    result = store.query(
        list(SAMPLE_QUERIES["q_axis"]), top_k=3, where={"strategy": "不存在的策略"}
    )

    assert result.hits == ()
    assert result.count == 0
    assert result.candidates == 0
    assert result.filter_applied is True


def test_min_score_cuts_hits_without_touching_candidates() -> None:
    """``min_score`` 只切命中（并列的 r_a/r_d 保留、0.8 的 r_c 被切掉）."""
    store = loaded_store()

    result = store.query(list(SAMPLE_QUERIES["q_axis"]), top_k=6, min_score=0.9)

    assert result.ids() == [RECORD_IDS[0], RECORD_IDS[3]]
    assert result.candidates == 6
    assert result.filter_applied is False


def test_find_ids_scans_the_record_table_since_faiss_has_no_metadata() -> None:
    """``native_metadata`` 为 False：``_find_ids`` 退化成扫记录表，语义与过滤一致."""
    store = loaded_store()

    assert store._find_ids({"strategy": "semantic"}) == {RECORD_IDS[4]}
    assert store._find_ids(None) == set(RECORD_IDS)
    assert store._find_ids({"$and": [{"topic": "glossary"}, {"oversized": True}]}) == {
        RECORD_IDS[5]
    }


# --------------------------------------------------------------------------- #
# -1 填充 / 类型要求（假模块的忠实性）
# --------------------------------------------------------------------------- #


def test_query_top_k_larger_than_library_returns_every_record() -> None:
    """``top_k`` 大于库内条数：返回条数等于库内条数（不会多出 -1 的占位）."""
    store = loaded_store()

    result = store.query(list(SAMPLE_QUERIES["q_axis"]), top_k=10)

    assert result.count == 6
    assert result.ids() == reference_ids("cosine", "q_axis", 6)
    assert FAISS_ID_NOT_FOUND not in store._index.id_map.tolist()


def test_padding_entries_are_discarded_when_ntotal_is_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ntotal`` 过期（读到之后、search 之前有并发删除）时，``-1`` 填充必须被丢弃.

    这条路不是假设：适配器按"读到的 ntotal"决定检索深度，而两次调用之间
    索引可能变小，于是 ``I`` 的尾部会出现 ``-1``。适配器一旦把 ``-1``
    当成 id 去查记录表，命中的就是"没有内容的记录"。
    """
    store = loaded_store()
    monkeypatch.setattr(FakeIndexIDMap2, "ntotal", property(lambda self: 10))

    result = store.query(list(SAMPLE_QUERIES["q_axis"]), top_k=10)

    assert result.count == 6
    assert result.ids() == reference_ids("cosine", "q_axis", 6)


def test_fake_index_pads_with_minus_one_and_distance_sentinels() -> None:
    """假模块的填充必须与真库一致：``L2 → +Inf``、``IP → -Inf``、``I → -1``."""
    l2_index = FakeIndexFlatL2(3)
    l2_index.add_with_ids(
        np.asarray([[0.0, 0.0, 0.0]], dtype="float32"), np.asarray([7], dtype=np.int64)
    )
    distances, ids = l2_index.search(
        np.asarray([[1.0, 0.0, 0.0]], dtype="float32"), 3
    )
    assert ids.tolist() == [[7, FAISS_ID_NOT_FOUND, FAISS_ID_NOT_FOUND]]
    assert distances[0][0] == pytest.approx(1.0)
    assert distances[0][1] == float("inf")
    assert distances[0][2] == float("inf")

    ip_index = FakeIndexFlatIP(3)
    ip_index.add_with_ids(
        np.asarray([[0.0, 0.0, 0.0]], dtype="float32"), np.asarray([9], dtype=np.int64)
    )
    ip_distances, ip_ids = ip_index.search(
        np.asarray([[1.0, 0.0, 0.0]], dtype="float32"), 2
    )
    assert ip_ids.tolist() == [[9, FAISS_ID_NOT_FOUND]]
    assert ip_distances[0][0] == pytest.approx(0.0)
    assert ip_distances[0][1] == float("-inf")


def test_fake_index_flat_rejects_unknown_keyword_arguments() -> None:
    """假模块不许适配器"发明参数"：真库的 ``IndexFlat*`` 只接受 ``d``."""
    with pytest.raises(TypeError, match="不接受关键字参数"):
        FakeIndexFlatIP(VECTOR_DIMENSION, nprobe=16)
    with pytest.raises(TypeError, match="不接受关键字参数"):
        FakeIndexFlatL2(VECTOR_DIMENSION, metric_type=1)


def test_fake_index_id_map_2_add_must_fail() -> None:
    """真库的 ``IndexIDMap2.add`` 会断言失败：写入只能走 ``add_with_ids``."""
    wrapper = FakeIndexIDMap2(FakeIndexFlatIP(VECTOR_DIMENSION))

    with pytest.raises(RuntimeError, match="add_with_ids"):
        wrapper.add(np.zeros((1, VECTOR_DIMENSION), dtype="float32"))


def test_fake_requires_float32_vectors_and_int64_ids() -> None:
    """类型要求照真 wrapper：``x`` 必须 float32、``ids`` 必须 int64."""
    index = FakeIndexFlatIP(VECTOR_DIMENSION)

    with pytest.raises(TypeError, match="float32"):
        index.add_with_ids(
            np.zeros((1, VECTOR_DIMENSION), dtype="float64"),
            np.asarray([1], dtype=np.int64),
        )
    with pytest.raises(TypeError, match="int64"):
        index.add_with_ids(
            np.zeros((1, VECTOR_DIMENSION), dtype="float32"),
            np.asarray([1], dtype=np.int32),
        )
    with pytest.raises(TypeError, match="float32"):
        index.search(np.zeros((1, VECTOR_DIMENSION), dtype="float64"), 1)
    with pytest.raises(TypeError, match="int64"):
        FakeIDSelectorBatch(np.asarray([1, 2], dtype=np.int32))


def test_adapter_passes_float32_and_int64_to_the_wrapper_signature() -> None:
    """适配器自己完成类型转换：``add_with_ids`` 收 float32+int64，``search`` 收 float32.

    同时断言"假模块的严格是真的会响"：直接喂 float64 必须 ``TypeError``，
    说明适配器没有绕过真 wrapper 的类型要求（它没有把 dtype 交给对方兜底）。
    """
    store = empty_store()
    store.upsert(sample_records(metric="cosine")[:1])

    assert store._index.add_calls == [(np.dtype("float32"), np.dtype("int64"))]

    store.query(list(SAMPLE_QUERIES["q_axis"]), top_k=1)
    assert store._index.searches[-1].dtype == np.dtype("float32")
    assert store._index.searches[-1].shape == (1, VECTOR_DIMENSION)
    assert store._index.searches[-1].flags["C_CONTIGUOUS"]

    with pytest.raises(TypeError, match="float32"):
        store._index.search(
            np.zeros((1, VECTOR_DIMENSION), dtype="float64"), 1
        )


def test_class_attributes_declare_the_optional_dependency() -> None:
    """类属性是 registry 判定的依据（D 号任务据此给安装指引）."""
    assert FaissVectorStore.name == "faiss"
    assert FaissVectorStore.requires == ("faiss", "numpy")
    assert FaissVectorStore.supports_persistence is True
    assert FaissVectorStore.supports_filter is True
    assert FaissVectorStore.supports_delete is True
    assert FaissVectorStore.native_metadata is False


# --------------------------------------------------------------------------- #
# id 映射
# --------------------------------------------------------------------------- #


def test_to_int64_id_is_stable_and_stays_in_range() -> None:
    """同一 id 永远同一个值；六条样本落在 ``[1, 2**63-2]`` 且互不相同."""
    assert INT64_ID_MODULUS == 2**63 - 2
    assert to_int64_id(RECORD_IDS[0]) == to_int64_id(RECORD_IDS[0])

    values = [to_int64_id(record_id) for record_id in RECORD_IDS]

    assert all(1 <= value <= INT64_ID_MODULUS for value in values)
    assert all(value != FAISS_ID_NOT_FOUND for value in values)
    assert all(value != 0 for value in values)
    assert len(set(values)) == len(RECORD_IDS)


def test_from_int64_id_guards_the_reserved_value_and_the_range() -> None:
    """``from_int64_id`` 不是逆函数，它负责挡住 ``-1`` 与值域外的整数."""
    value = to_int64_id(RECORD_IDS[0])

    assert from_int64_id(value) == value
    assert isinstance(from_int64_id(np.int64(value)), int)

    with pytest.raises(VectorError, match="-1"):
        from_int64_id(FAISS_ID_NOT_FOUND)
    with pytest.raises(VectorError, match="值域"):
        from_int64_id(0)
    with pytest.raises(VectorError, match="值域"):
        from_int64_id(INT64_ID_MODULUS + 1)


def test_hash_collision_is_reported_and_nothing_is_written(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """人为制造一次撞号（常量摘要）→ ``VectorStoreError``，且**整批拒绝**."""
    store = empty_store()
    first, second = sample_records(metric="cosine")[:2]
    monkeypatch.setattr(faiss_backend, "_id_digest", lambda record_id: b"\x2a" * 32)
    store.upsert([first])

    with pytest.raises(VectorStoreError, match="哈希冲突"):
        store.upsert([second])

    assert store.ids() == [first.record_id]
    assert store.count() == 1
    assert store._index.ntotal == 1
    assert store.get(second.record_id) is None

    # 同一批里的两条相撞也在预检阶段被拦住（库里一个字节都没写）
    fresh = empty_store()
    with pytest.raises(VectorStoreError, match="哈希冲突"):
        fresh.upsert([first, second])

    assert fresh.count() == 0
    assert fresh._index.ntotal == 0


def test_load_detects_a_hash_collision_inside_the_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """快照里的两个 id 若映射到同一个 int64（换了哈希实现），载入必须拒绝."""
    store = loaded_store()
    index_path = tmp_path / "faiss.index"
    store.persist(str(index_path))
    monkeypatch.setattr(faiss_backend, "_id_digest", lambda record_id: b"\x2a" * 32)

    with pytest.raises(VectorStoreError, match="快照里的 id 发生哈希冲突"):
        empty_store().load(str(index_path))


# --------------------------------------------------------------------------- #
# 维度
# --------------------------------------------------------------------------- #


def test_dimension_is_required_when_the_store_starts_empty() -> None:
    """构造时未给维度且库为空 → 报错，消息里必须给出两条出路."""
    with pytest.raises(VectorStoreError) as excinfo:
        FaissVectorStore(faiss_module=make_fake_faiss_module())

    message = str(excinfo.value)
    assert "FAISS 在建索引之前就要知道维度，请传入 dimension，或先写入一批数据" in message
    assert "flat" in message


def test_writing_a_different_dimension_raises_vector_error() -> None:
    """写入维度与库不一致 → ``VectorError``，且索引里一个向量都没进去."""
    store = empty_store()
    wrong = make_record(
        record_id="维度不对的记录", vector=[1.0, 2.0], text="两维", metric="cosine"
    )

    with pytest.raises(VectorError, match="维度不一致"):
        store.upsert([wrong])

    assert store.count() == 0
    assert store._index.ntotal == 0


def test_dimension_must_be_positive() -> None:
    """``dimension=0`` 是哨兵值而不是合法维度."""
    with pytest.raises(VectorError, match="dimension 必须是 >= 1"):
        FaissVectorStore(dimension=0, faiss_module=make_fake_faiss_module())


def test_adopting_an_index_derives_missing_dimension() -> None:
    """接管一个 IDMap2（空）时，维度可以从 ``index.d`` 得到."""
    index = FakeIndexIDMap2(FakeIndexFlatIP(4))
    store = FaissVectorStore(faiss_module=make_fake_faiss_module(), index=index)

    assert store.dimension == 4
    assert store._index is index
    assert store.describe_extra()["index_type"] == "IndexFlatIP"


def test_adopting_a_bare_flat_index_wraps_it_in_an_id_map() -> None:
    """裸 ``IndexFlat`` 没有外部 id，接管时必须包一层 ``IndexIDMap2``."""
    store = FaissVectorStore(
        faiss_module=make_fake_faiss_module(), index=FakeIndexFlatIP(VECTOR_DIMENSION)
    )

    assert isinstance(store._index, FakeIndexIDMap2)
    assert store.dimension == VECTOR_DIMENSION
    assert store._index.ntotal == 0

    store.upsert(sample_records(metric="cosine")[:2])
    assert store.query(list(SAMPLE_QUERIES["q_axis"]), top_k=1).ids() == [RECORD_IDS[0]]


def test_adopting_a_non_empty_bare_index_is_rejected() -> None:
    """非空的裸索引无法还原"行号 → 记录 id"，必须拒绝接管."""
    raw = FakeIndexFlatIP(VECTOR_DIMENSION)
    raw.add(np.asarray([[1.0] + [0.0] * 7], dtype="float32"))

    with pytest.raises(VectorError, match="裸索引"):
        FaissVectorStore(faiss_module=make_fake_faiss_module(), index=raw)


def test_adopting_an_index_with_the_wrong_metric_is_rejected() -> None:
    """索引的度量类型与本实例的 metric 不一致 → 拒绝（否则排序会整体反向）."""
    with pytest.raises(VectorError, match="度量类型"):
        FaissVectorStore(
            metric="l2",
            faiss_module=make_fake_faiss_module(),
            index=FakeIndexIDMap2(FakeIndexFlatIP(VECTOR_DIMENSION)),
        )


def test_adopting_an_index_without_metric_type_still_works() -> None:
    """接管的索引缺少 ``metric_type``（自定义实现）时不该误报——只要求它能表达外部 id."""
    store = FaissVectorStore(
        faiss_module=make_fake_faiss_module(), index=MinimalIndex(VECTOR_DIMENSION)
    )

    assert store.dimension == VECTOR_DIMENSION
    assert store._index.id_map.tolist() == []


def test_index_dimension_must_match_the_declared_dimension() -> None:
    """``dimension`` 与 ``index.d`` 不一致 → ``VectorError``."""
    with pytest.raises(VectorError, match="索引的维度"):
        FaissVectorStore(
            dimension=VECTOR_DIMENSION,
            faiss_module=make_fake_faiss_module(),
            index=FakeIndexIDMap2(FakeIndexFlatIP(4)),
        )


# --------------------------------------------------------------------------- #
# 依赖缺失 / 版本不匹配
# --------------------------------------------------------------------------- #


def test_missing_faiss_module_gives_install_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """没装 faiss 时给的是"该装哪个包"，而不是一个裸的 ImportError."""
    monkeypatch.setitem(sys.modules, "faiss", None)

    with pytest.raises(BackendUnavailable, match="pip install faiss-cpu"):
        FaissVectorStore(dimension=VECTOR_DIMENSION)


@pytest.mark.parametrize(
    ("module", "missing"),
    [
        (FakeFaissModuleMissingIndexFlat(), "IndexFlatIP"),
        (FakeFaissModuleMissingIDSelectorBatch(), "IDSelectorBatch"),
    ],
)
def test_incomplete_faiss_module_is_detected_at_construction(
    module: object, missing: str
) -> None:
    """对方版本不匹配（少符号）→ 构造点就报错，并点名缺的是什么."""
    with pytest.raises(BackendUnavailable, match=missing):
        FaissVectorStore(dimension=VECTOR_DIMENSION, faiss_module=module)


def test_injected_module_is_used_without_importing_faiss() -> None:
    """注入优先：即使 ``sys.modules["faiss"]`` 被弄坏，注入的模块仍然可用."""
    store = FaissVectorStore(
        metric="l2",
        dimension=VECTOR_DIMENSION,
        faiss_module=make_fake_faiss_module(),
        index=None,
    )
    store.upsert(sample_records(metric="l2"))

    assert store.describe_extra()["index_type"] == "IndexFlatL2"
    assert store.query(list(SAMPLE_QUERIES["q_axis"]), top_k=3).ids() == reference_ids(
        "l2", "q_axis", 3
    )


# --------------------------------------------------------------------------- #
# 落盘：索引 + 旁挂记录两个文件
# --------------------------------------------------------------------------- #


def test_persist_and_load_round_trip(tmp_path: Path) -> None:
    """``persist`` 写出两个文件；``load`` 之后 ids/记录逐条相同且仍可检索."""
    store = loaded_store()
    index_path = tmp_path / "faiss.index"
    location = store.persist(str(index_path))

    assert location == str(index_path)
    assert index_path.exists()
    sidecar = Path(str(index_path) + RECORDS_SUFFIX)
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["backend"] == "faiss"
    assert payload["metric"] == "cosine"
    assert payload["dimension"] == VECTOR_DIMENSION
    assert payload["count"] == 6
    assert payload["index_ntotal"] == 6
    assert [item["record_id"] for item in payload["records"]] == sorted(RECORD_IDS)
    assert store.info().persistent is True
    assert store.info().location == str(index_path)

    restored = empty_store()
    restored.load(str(index_path))

    assert restored.ids() == store.ids()
    assert restored.count() == 6
    assert restored._index.ntotal == 6
    assert restored.location == str(index_path)
    for record_id in store.ids():
        assert restored.get(record_id).vector == store.get(record_id).vector
        assert restored.get(record_id).text == store.get(record_id).text
        assert restored.get(record_id).metadata == store.get(record_id).metadata
    assert restored.query(list(SAMPLE_QUERIES["q_tilt"]), top_k=3).ids() == reference_ids(
        "cosine", "q_tilt", 3
    )


def test_persist_uses_the_constructor_path_and_creates_directories(
    tmp_path: Path,
) -> None:
    """``path`` 只是默认落盘位置（构造时不自动 load）；目录不存在时自动建."""
    index_path = tmp_path / "nested" / "dir" / "faiss.index"
    store = FaissVectorStore(
        metric="cosine",
        dimension=VECTOR_DIMENSION,
        faiss_module=make_fake_faiss_module(),
        path=str(index_path),
    )
    assert store.count() == 0

    store.upsert(sample_records(metric="cosine"))
    assert store.persist() == str(index_path)
    assert index_path.exists()

    reopened = FaissVectorStore(
        metric="cosine",
        dimension=VECTOR_DIMENSION,
        faiss_module=make_fake_faiss_module(),
        path=str(index_path),
    )
    assert reopened.count() == 0  # 不自动 load（见 __init__ 的说明）
    reopened.load()
    assert reopened.ids() == sorted(RECORD_IDS)


def test_persist_without_any_path_raises(tmp_path: Path) -> None:
    """既没有构造 path 也没有本次 path → ``BackendUnavailable``，不静默降级."""
    store = empty_store()

    with pytest.raises(BackendUnavailable, match="没有可写的路径"):
        store.persist()


def test_load_without_any_path_raises() -> None:
    """没有可读路径 → ``BackendUnavailable``."""
    store = empty_store()

    with pytest.raises(BackendUnavailable, match="没有可读的路径"):
        store.load()


def test_load_rejects_missing_files(tmp_path: Path) -> None:
    """两个文件必须成对：缺索引、缺记录表各有明确的报错."""
    store = loaded_store()
    index_path = tmp_path / "only-index.faiss"
    store._module.write_index(store._index, str(index_path))

    with pytest.raises(BackendUnavailable, match="旁挂文件不存在"):
        empty_store().load(str(index_path))
    with pytest.raises(BackendUnavailable, match="索引文件不存在"):
        empty_store().load(str(tmp_path / "nope.index"))


def test_load_rejects_a_corrupt_index_file(tmp_path: Path) -> None:
    """索引文件不是本类写出的（这里塞了 JSON 文本）→ 报可读的错误."""
    store = loaded_store()
    index_path = tmp_path / "faiss.index"
    store.persist(str(index_path))
    Path(index_path).write_text("{}", encoding="utf-8")

    with pytest.raises(BackendUnavailable, match="读取 faiss 索引失败"):
        empty_store().load(str(index_path))


def _rewrite_sidecar(index_path: Path, mutate) -> None:  # type: ignore[no-untyped-def]
    """改一改旁挂文件（测试"载入时必须逐项核对"这条纪律）."""
    sidecar = Path(str(index_path) + RECORDS_SUFFIX)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    mutate(payload)
    sidecar.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_load_rejects_a_version_mismatch(tmp_path: Path) -> None:
    """快照版本不匹配 → ``VectorStoreError``（报文里点明"快照版本"）."""
    store = loaded_store()
    index_path = tmp_path / "faiss.index"
    store.persist(str(index_path))
    _rewrite_sidecar(index_path, lambda payload: payload.update(version=999))

    with pytest.raises(VectorStoreError, match="快照版本与当前代码不匹配"):
        empty_store().load(str(index_path))


def test_load_rejects_a_metric_mismatch(tmp_path: Path) -> None:
    """度量不一致 → ``VectorError``，并给出"用快照的度量新建实例"这条出路."""
    store = loaded_store()
    index_path = tmp_path / "faiss.index"
    store.persist(str(index_path))
    other = empty_store(metric="l2")

    with pytest.raises(VectorError) as excinfo:
        other.load(str(index_path))

    message = str(excinfo.value)
    assert "快照的 metric='cosine' 与本实例的 metric='l2' 不一致" in message
    assert "用 snapshot 的 metric 新建实例" in message


def test_load_rejects_a_dimension_mismatch(tmp_path: Path) -> None:
    """维度不一致 → ``VectorError``."""
    store = loaded_store()
    index_path = tmp_path / "faiss.index"
    store.persist(str(index_path))
    other = FaissVectorStore(
        metric="cosine", dimension=4, faiss_module=make_fake_faiss_module()
    )

    with pytest.raises(VectorError, match="快照的 dimension=8"):
        other.load(str(index_path))


def test_load_rejects_a_self_contradictory_sidecar(tmp_path: Path) -> None:
    """``count`` 与 ``records`` 条数对不上 → 文件被截断或手改过."""
    store = loaded_store()
    index_path = tmp_path / "faiss.index"
    store.persist(str(index_path))
    _rewrite_sidecar(index_path, lambda payload: payload.update(records=payload["records"][:-1]))

    with pytest.raises(VectorStoreError, match="快照自相矛盾"):
        empty_store().load(str(index_path))


def test_load_rejects_index_and_records_count_desync(tmp_path: Path) -> None:
    """旁挂文件自洽（5 条）但索引里还有 6 条 → "两个库"不同步."""
    store = loaded_store()
    index_path = tmp_path / "faiss.index"
    store.persist(str(index_path))

    def shrink(payload: dict) -> None:
        payload["records"] = payload["records"][:-1]
        payload["count"] = 5
        payload["index_ntotal"] = 5

    _rewrite_sidecar(index_path, shrink)

    with pytest.raises(VectorError, match="索引与记录表不同步"):
        empty_store().load(str(index_path))


def test_load_rejects_index_and_records_id_desync(tmp_path: Path) -> None:
    """条数相同但 id 集合不同（两个文件不是一次写出的）→ ``VectorError``."""
    store = loaded_store()
    index_path = tmp_path / "faiss.index"
    store.persist(str(index_path))
    _rewrite_sidecar(
        index_path,
        lambda payload: payload["records"][0].update(record_id="ffffffffffffffff"),
    )

    with pytest.raises(VectorError, match="id 集合"):
        empty_store().load(str(index_path))


def test_load_replaces_state_and_keeps_metric_and_dimension(tmp_path: Path) -> None:
    """``load`` 是"替换"：库里原来的记录不会残留；度量与维度不变."""
    store = loaded_store()
    index_path = tmp_path / "faiss.index"
    store.persist(str(index_path))

    other = empty_store()
    other.upsert(sample_records(metric="cosine")[:1])
    other.load(str(index_path))

    assert other.ids() == sorted(RECORD_IDS)
    assert other.count() == 6
    assert other.metric == "cosine"
    assert other.dimension == VECTOR_DIMENSION
