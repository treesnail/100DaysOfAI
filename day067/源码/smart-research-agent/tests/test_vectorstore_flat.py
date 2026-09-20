"""day064 ``vectorstore.flat`` 的单元测试：参照实现的全部原语与持久化.

全部离线、确定性、零网络；落盘只用 pytest 的 ``tmp_path``。
``flat`` 是另外两个后端的**基准**，所以这里的断言刻意写得"硬"：

```text
写入     added / updated / unchanged 三个计数各自对得上
查询     具体到某几个 id、且同分由 id 升序决定
限制内排序  构造"全库 top-1 不在候选集里"的数据，逼出"先取 top-k 再求交集"的错法
持久化   快照字段、往返逐条相同、三种不一致各自报错
```

其中"限制内排序"那一条是本文件最重要的用例：那种写错**不报错**，
只表现为"加了过滤之后结果变少了"。
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from smart_research_agent.vectorstore.base import MAX_TOP_K
from smart_research_agent.vectorstore.errors import (
    BackendUnavailable,
    FilterError,
    RecordError,
    VectorError,
    VectorStoreError,
)
from smart_research_agent.vectorstore.flat import (
    SNAPSHOT_VERSION,
    FlatVectorStore,
    build_flat_store,
)
from smart_research_agent.vectorstore.types import VectorRecord, make_record
from tests.vectorstore_samples import (
    RECORD_IDS,
    SAMPLE_QUERIES,
    brute_force_top,
    sample_records,
    sample_vectors,
)

#: 与样本一致的维度（``vectorstore_samples.VECTOR_DIMENSION`` 是 8）.
DIMENSION = 8


def cosine_store(*, path: str = "", dimension: int | None = None) -> FlatVectorStore:
    """一个装好六条样本（cosine）的 flat 库."""
    store = FlatVectorStore(metric="cosine", dimension=dimension, path=path)
    store.upsert(sample_records(metric="cosine"))
    return store


def unit_axis(index: int = 0) -> tuple[float, ...]:
    """第 ``index`` 个分量是 1 的单位向量（造"改向量"的数据用）."""
    return tuple(1.0 if position == index else 0.0 for position in range(DIMENSION))


# --------------------------------------------------------------------------- #
# 写入：added / updated / unchanged
# --------------------------------------------------------------------------- #


def test_upsert_adds_all_records() -> None:
    store = FlatVectorStore(metric="cosine")

    report = store.upsert(sample_records(metric="cosine"))

    assert report.added == 6
    assert report.updated == 0
    assert report.unchanged == 0
    assert report.written == 6
    assert store.count() == 6


def test_replaying_the_same_batch_is_unchanged_and_keeps_count() -> None:
    """重放同一批 → ``unchanged=6`` 且库不变（这是增量索引的地基）."""
    store = FlatVectorStore(metric="cosine")
    store.upsert(sample_records(metric="cosine"))

    report = store.upsert(sample_records(metric="cosine"))

    assert report.added == 0
    assert report.updated == 0
    assert report.unchanged == 6
    assert store.count() == 6


def test_replaying_via_stored_objects_is_also_unchanged() -> None:
    """直接把库里取出来的记录再写回去也必须 ``unchanged``.

    ``base._prepare`` 会用 ``make_record`` 重建每条记录（cosine 会再归一化一次），
    因此这条断言实际上在要求**归一化是幂等的**：否则每次重放都会报 ``updated``，
    而 day065 的"省掉了多少次编码"就会永远算成 0。
    """
    store = cosine_store()
    stored = [store.get(record_id) for record_id in store.ids()]

    report = store.upsert([item for item in stored if item is not None])

    assert report.unchanged == 6
    assert report.updated == 0


def test_upsert_reports_a_single_vector_change() -> None:
    store = cosine_store()
    records = sample_records(metric="cosine")
    changed = list(records)
    changed[3] = make_record(
        RECORD_IDS[3],
        unit_axis(1),
        records[3].text,
        dict(records[3].metadata),
        metric="cosine",
    )

    report = store.upsert(changed)

    assert report.updated == 1
    assert report.unchanged == 5
    assert report.added == 0
    assert store.count() == 6
    assert store.get(RECORD_IDS[3]).vector == unit_axis(1)


def test_upsert_reports_a_single_text_change() -> None:
    store = cosine_store()
    records = sample_records(metric="cosine")
    raw = sample_vectors()
    changed = list(records)
    changed[0] = make_record(
        RECORD_IDS[0],
        raw[RECORD_IDS[0]],
        "改写过的正文",
        dict(records[0].metadata),
        metric="cosine",
    )

    report = store.upsert(changed)

    assert report.updated == 1
    assert report.unchanged == 5
    assert store.get(RECORD_IDS[0]).text == "改写过的正文"


def test_upsert_reports_a_single_metadata_change() -> None:
    store = cosine_store()
    records = sample_records(metric="cosine")
    raw = sample_vectors()
    metadata = dict(records[0].metadata)
    metadata["topic"] = "rewritten"
    changed = list(records)
    changed[0] = make_record(
        RECORD_IDS[0],
        raw[RECORD_IDS[0]],
        records[0].text,
        metadata,
        metric="cosine",
    )

    report = store.upsert(changed)

    assert report.updated == 1
    assert report.unchanged == 5
    assert store.get(RECORD_IDS[0]).metadata["topic"] == "rewritten"


def test_unchanged_does_not_replace_the_stored_object() -> None:
    """``unchanged`` 时库里那个对象必须**没有被换掉**（身份不变）.

    计数对得上但对象被换掉，是一个"报告看起来没错、复用却静默失效"的故障。
    """
    store = cosine_store()
    before = store.get(RECORD_IDS[0])

    store.upsert(sample_records(metric="cosine"))

    assert store.get(RECORD_IDS[0]) is before


def test_ids_are_sorted_not_insertion_order() -> None:
    """``ids()`` 升序：逆序写入也要得到同一份列表（可复现性最基本的检查）."""
    store = FlatVectorStore(metric="cosine")

    store.upsert(list(reversed(sample_records(metric="cosine"))))

    assert store.ids() == sorted(RECORD_IDS)


def test_dimension_is_learned_on_first_write() -> None:
    store = FlatVectorStore(metric="cosine")
    assert store.has_dimension is False
    assert store.dimension == 0

    store.upsert(sample_records(metric="cosine")[:1])

    assert store.has_dimension is True
    assert store.dimension == DIMENSION


def test_upsert_dimension_mismatch_raises() -> None:
    """库里已有 8 维数据后写进 4 维 → ``VectorError``（指向"编码器换过"）."""
    store = cosine_store()

    with pytest.raises(VectorError) as excinfo:
        store.upsert([make_record("short-0001", (1.0, 0.0, 0.0, 0.0))])

    assert "维度不一致" in str(excinfo.value)
    assert store.count() == 6


# --------------------------------------------------------------------------- #
# add：整体拒绝
# --------------------------------------------------------------------------- #


def test_add_on_empty_store_succeeds() -> None:
    store = FlatVectorStore(metric="cosine")

    report = store.add(sample_records(metric="cosine"))

    assert report.added == 6
    assert store.count() == 6


def test_add_conflict_is_rejected_atomically() -> None:
    """撞 id → ``RecordError``，且**整批都不写**（连新 id 也不进库）."""
    store = cosine_store()
    before = store.ids()
    new_id = "00000000000000ff"

    with pytest.raises(RecordError) as excinfo:
        store.add(
            [
                make_record(new_id, unit_axis(0), "新记录"),
                make_record(RECORD_IDS[0], unit_axis(0), "撞车"),
            ]
        )

    assert "add() 撞上" in str(excinfo.value)
    assert store.ids() == before
    assert store.get(new_id) is None
    assert store.count() == 6


# --------------------------------------------------------------------------- #
# 查询：top_k / min_score / where
# --------------------------------------------------------------------------- #


def test_query_top_k_follows_the_tie_rule() -> None:
    """``q_axis`` 下 ``r_a`` 与 ``r_d`` 并列 1.0，同分由 id 升序决定成败."""
    store = cosine_store()

    result = store.query(SAMPLE_QUERIES["q_axis"], top_k=3)

    assert result.ids() == [RECORD_IDS[0], RECORD_IDS[3], RECORD_IDS[2]]
    assert result.scores[0] == pytest.approx(1.0)
    assert result.scores[1] == pytest.approx(1.0)
    assert result.candidates == 6
    assert result.filter_applied is False
    assert [hit.rank for hit in result.hits] == [0, 1, 2]


def test_query_min_score_drops_low_hits() -> None:
    """阈值切掉 2 条：``candidates`` 仍是 6（阈值不是过滤，不改变候选数）."""
    store = cosine_store()

    result = store.query(SAMPLE_QUERIES["q_axis"], top_k=5, min_score=0.5)

    assert result.ids() == [
        RECORD_IDS[0],
        RECORD_IDS[3],
        RECORD_IDS[2],
        RECORD_IDS[5],
    ]
    assert result.count == 4
    assert result.candidates == 6
    assert result.filter_applied is False


def test_query_where_narrows_candidates() -> None:
    store = cosine_store()

    result = store.query(SAMPLE_QUERIES["q_axis"], top_k=5, where={"strategy": "structural"})

    assert result.ids() == [RECORD_IDS[3], RECORD_IDS[5]]
    assert result.candidates == 2
    assert result.filter_applied is True


def test_query_sorts_within_restrict_not_globally() -> None:
    """**必须先在 restrict 内排序**：全库 top-1 不在候选集里时也要返回一条.

    数据是刻意构造的：``q_tilt`` 下全库第 1 名是 ``r_c``（不在 structural 里），
    而 structural 的 top-1 是 ``r_f``。若实现写成"先全库取 top-1 再求交集"，
    这里会返回**空结果**——不报错，只是少返回。
    """
    store = cosine_store()
    query = SAMPLE_QUERIES["q_tilt"]
    global_top = store.query(query, top_k=1)
    assert global_top.ids() == [RECORD_IDS[2]]  # 不在 restrict 里

    result = store.query(query, top_k=1, where={"strategy": "structural"})

    assert result.ids() == [RECORD_IDS[5]]
    assert result.scores[0] == pytest.approx(0.7)
    assert result.candidates == 2


def test_query_where_matching_nothing_reports_zero_candidates() -> None:
    """``candidates=0`` 且 ``filter_applied=True``＝"过滤器把记录全排除了"."""
    store = cosine_store()

    result = store.query(SAMPLE_QUERIES["q_axis"], top_k=5, where={"strategy": "缺失的策略"})

    assert result.count == 0
    assert result.ids() == []
    assert result.candidates == 0
    assert result.filter_applied is True
    assert result.top() is None


def test_query_on_empty_store_is_not_an_error() -> None:
    """空库查询返回空结果（``filter_applied=False``），而不是抛错或"库是空的"特例."""
    store = FlatVectorStore(metric="cosine")

    result = store.query(unit_axis(0), top_k=5)

    assert result.count == 0
    assert result.candidates == 0
    assert result.filter_applied is False


def test_query_rejects_top_k_below_one() -> None:
    store = cosine_store()

    with pytest.raises(FilterError) as excinfo:
        store.query(SAMPLE_QUERIES["q_axis"], top_k=0)

    assert "top_k 必须 >= 1" in str(excinfo.value)


def test_query_rejects_top_k_above_the_cap() -> None:
    """上限是防御性的：一次把整库搬进响应体没有意义."""
    store = cosine_store()

    with pytest.raises(FilterError) as excinfo:
        store.query(SAMPLE_QUERIES["q_axis"], top_k=MAX_TOP_K + 1)

    assert "超过上限" in str(excinfo.value)


def test_query_rejects_dimension_mismatch() -> None:
    store = cosine_store()

    with pytest.raises(VectorError) as excinfo:
        store.query((1.0, 0.0, 0.0, 0.0), top_k=1)

    assert "查询向量是 4 维" in str(excinfo.value)


@pytest.mark.parametrize("metric", ["cosine", "ip", "l2"])
@pytest.mark.parametrize("query_name", ["q_axis", "q_tilt"])
def test_query_matches_brute_force_reference(metric: str, query_name: str) -> None:
    """三种度量 × 两个查询，与独立参照实现逐条对账（含分数与名次）."""
    records = sample_records(metric=metric)
    store = FlatVectorStore(metric=metric)
    store.upsert(records)
    query = SAMPLE_QUERIES[query_name]

    expected = brute_force_top(records, query, metric, top_k=3)
    result = store.query(query, top_k=3)

    assert result.ids() == [record_id for record_id, _ in expected]
    for hit, (_, value) in zip(result.hits, expected):
        assert hit.score == pytest.approx(value)


# --------------------------------------------------------------------------- #
# 删除与清空
# --------------------------------------------------------------------------- #


def test_delete_by_ids_returns_real_count() -> None:
    """删不存在的 id 是幂等操作：不报错，但返回值必须诚实（返回 0）."""
    store = cosine_store()

    assert store.delete(ids=[RECORD_IDS[0], "missing-id"]) == 1
    assert store.count() == 5
    assert store.get(RECORD_IDS[0]) is None
    assert store.delete(ids=["missing-id"]) == 0
    assert store.delete(ids=[]) == 0


def test_delete_by_where_removes_matching_ids() -> None:
    store = cosine_store()

    removed = store.delete(where={"strategy": "structural"})

    assert removed == 2
    assert store.count() == 4
    assert store.get(RECORD_IDS[3]) is None
    assert store.get(RECORD_IDS[5]) is None


def test_delete_rejects_both_ids_and_where() -> None:
    store = cosine_store()

    with pytest.raises(RecordError) as excinfo:
        store.delete(ids=[RECORD_IDS[0]], where={"strategy": "structural"})

    assert "不能同时给出" in str(excinfo.value)
    assert store.count() == 6


def test_delete_needs_ids_or_where() -> None:
    store = cosine_store()

    with pytest.raises(RecordError) as excinfo:
        store.delete()

    assert "需要 ids 或 where" in str(excinfo.value)


def test_delete_by_where_matching_nothing_returns_zero() -> None:
    store = cosine_store()

    assert store.delete(where={"strategy": "缺失的策略"}) == 0
    assert store.count() == 6


def test_clear_keeps_metric_and_dimension() -> None:
    """清空只清数据：度量与维度保留，便于复用同一个实例重建."""
    store = cosine_store()

    store.clear()

    assert store.count() == 0
    assert store.ids() == []
    assert store.metric == "cosine"
    assert store.dimension == DIMENSION


# --------------------------------------------------------------------------- #
# 持久化
# --------------------------------------------------------------------------- #


def test_persist_writes_snapshot_and_creates_parent_dirs(tmp_path: Any) -> None:
    """快照字段逐个核对；目录不存在时由 ``persist`` 创建."""
    store = cosine_store()
    target = tmp_path / "nested" / "flat_snapshot.json"

    written = store.persist(str(target))

    assert written == str(target)
    assert target.exists()
    assert store.location == str(target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["version"] == SNAPSHOT_VERSION
    assert payload["backend"] == "flat"
    assert payload["metric"] == "cosine"
    assert payload["dimension"] == DIMENSION
    assert payload["count"] == 6
    assert len(payload["records"]) == 6
    assert set(payload["records"][0]) == {"record_id", "vector", "text", "metadata"}
    assert [item["record_id"] for item in payload["records"]] == sorted(RECORD_IDS)


def test_persist_then_load_roundtrip(tmp_path: Any) -> None:
    """新实例带 ``path`` 构造时自动载入，逐条与源库相同."""
    store = cosine_store()
    target = tmp_path / "flat_snapshot.json"
    store.persist(str(target))

    reloaded = FlatVectorStore(metric="cosine", path=str(target))

    assert reloaded.count() == 6
    assert reloaded.ids() == store.ids()
    assert reloaded.dimension == DIMENSION
    for record_id in store.ids():
        mine = store.get(record_id)
        theirs = reloaded.get(record_id)
        assert mine is not None
        assert theirs is not None
        assert isinstance(theirs, VectorRecord)
        assert theirs.vector == mine.vector
        assert theirs.text == mine.text
        assert theirs.metadata == mine.metadata


def test_constructor_with_missing_path_does_not_load(tmp_path: Any) -> None:
    """``path`` 指向不存在的文件时构造成功（还没有库是最常见的第一次运行）."""
    store = FlatVectorStore(metric="cosine", path=str(tmp_path / "not-yet.json"))

    assert store.count() == 0


def test_load_rejects_metric_mismatch(tmp_path: Any) -> None:
    """度量不一致 → ``VectorError``，消息里给出"用 snapshot 的 metric 新建实例"."""
    store = cosine_store()
    target = tmp_path / "flat_snapshot.json"
    store.persist(str(target))
    other = FlatVectorStore(metric="ip")

    with pytest.raises(VectorError) as excinfo:
        other.load(str(target))

    message = str(excinfo.value)
    assert "cosine" in message
    assert "ip" in message
    assert "新建实例" in message
    assert other.count() == 0


def test_load_rejects_dimension_mismatch(tmp_path: Any) -> None:
    store = cosine_store()
    target = tmp_path / "flat_snapshot.json"
    store.persist(str(target))
    other = FlatVectorStore(metric="cosine", dimension=4)

    with pytest.raises(VectorError) as excinfo:
        other.load(str(target))

    assert "dimension" in str(excinfo.value)


def test_load_rejects_snapshot_version_mismatch(tmp_path: Any) -> None:
    target = tmp_path / "old.json"
    target.write_text(
        json.dumps(
            {
                "version": SNAPSHOT_VERSION + 98,
                "backend": "flat",
                "metric": "cosine",
                "dimension": DIMENSION,
                "count": 0,
                "records": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(VectorStoreError) as excinfo:
        FlatVectorStore(metric="cosine").load(str(target))

    assert "快照版本" in str(excinfo.value)


def test_load_rejects_truncated_snapshot(tmp_path: Any) -> None:
    """``count`` 与 ``records`` 条数对不上＝这份文件被截断或手改过."""
    target = tmp_path / "truncated.json"
    target.write_text(
        json.dumps(
            {
                "version": SNAPSHOT_VERSION,
                "backend": "flat",
                "metric": "cosine",
                "dimension": DIMENSION,
                "count": 6,
                "records": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(VectorStoreError) as excinfo:
        FlatVectorStore(metric="cosine").load(str(target))

    assert "自相矛盾" in str(excinfo.value)


def test_persist_without_path_raises() -> None:
    """没有落盘位置时**不静默降级**（"以为落盘了"要到重启才暴露）."""
    store = FlatVectorStore(metric="cosine")

    with pytest.raises(BackendUnavailable) as excinfo:
        store.persist()

    assert "没有可写的路径" in str(excinfo.value)


def test_load_without_path_raises() -> None:
    with pytest.raises(BackendUnavailable):
        FlatVectorStore(metric="cosine").load()


def test_load_missing_file_raises(tmp_path: Any) -> None:
    with pytest.raises(BackendUnavailable) as excinfo:
        FlatVectorStore(metric="cosine").load(str(tmp_path / "nope.json"))

    assert "不存在" in str(excinfo.value)


def test_persist_updates_location_for_later_load(tmp_path: Any) -> None:
    """``persist`` 返回的路径会写回 ``location``，因此之后可以不传 path 直接 load."""
    store = cosine_store()
    target = tmp_path / "flat_snapshot.json"
    store.persist(str(target))

    store.clear()
    assert store.count() == 0
    store.load()

    assert store.count() == 6


# --------------------------------------------------------------------------- #
# 元信息与工厂
# --------------------------------------------------------------------------- #


def test_describe_extra_and_info() -> None:
    store = cosine_store()

    info = store.info()

    assert info.backend == "flat"
    assert info.metric == "cosine"
    assert info.dimension == DIMENSION
    assert info.count == 6
    assert info.persistent is False
    assert info.extra["records"] == "6"
    assert info.extra["snapshot_version"] == str(SNAPSHOT_VERSION)


def test_class_flags_match_the_contract() -> None:
    """四个能力开关 + ``requires`` 是"参照实现"这件事的公开声明."""
    assert FlatVectorStore.name == "flat"
    assert FlatVectorStore.requires == ()
    assert FlatVectorStore.supports_filter is True
    assert FlatVectorStore.supports_delete is True
    assert FlatVectorStore.supports_persistence is True
    assert FlatVectorStore.native_metadata is True


def test_build_flat_store_with_and_without_records() -> None:
    built = build_flat_store(sample_records(metric="cosine"))
    assert built.count() == 6
    assert built.metric == "cosine"
    assert built.dimension == DIMENSION

    empty = build_flat_store()
    assert empty.count() == 0

    l2_store = build_flat_store(sample_records(metric="l2"), metric="l2")
    assert l2_store.metric == "l2"
    assert l2_store.count() == 6
