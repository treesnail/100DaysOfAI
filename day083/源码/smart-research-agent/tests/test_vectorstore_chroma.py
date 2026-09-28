"""day064 ``vectorstore.chroma_backend`` 的单元测试：用假 chromadb 注入再逐条对账.

真库既没装、也不允许为了跑测试去装（它默认的 embedding function 会在
"只是建了个集合"的时候去下载 MiniLM 的权重）。因此每一个用例都通过
``tests.chroma_fakes`` 注入一份忠实的假库，断言的对象因此有三类：

```text
交出去的参数   where 有没有被原样下推、embedding_function 是不是 None、
              n_results 是「过滤之后」的还是「过滤之前」的
收回来的数值   score 与 distance 的换算方向、与 brute_force_top 的逐条对账
抛出来的消息   坏输入必须给出能照着改的具体片段，而不是一句"参数错误"
```

其中"与参照实现对账"是这一课的重点：``brute_force_top`` 是一段**不复用**
``metrics.py`` 的独立实现，再加上"三种度量给出的 top-1 必须不同"这条探针
——它专门用来抓"把三个 space 都按同一个公式算"这类不报错的静默错误。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from smart_research_agent.vectorstore import (
    MAX_TOP_K,
    METRIC_COSINE,
    METRIC_INNER_PRODUCT,
    METRIC_L2,
    BackendUnavailable,
    FilterError,
    RecordError,
    VectorError,
    VectorRecord,
    make_record,
)
from smart_research_agent.vectorstore import chroma_backend as chroma_module
from smart_research_agent.vectorstore.chroma_backend import (
    COLLECTION_NAME_MAX_LENGTH,
    COLLECTION_NAME_MIN_LENGTH,
    DEFAULT_COLLECTION_NAME,
    SPACE_BY_METRIC,
    ChromaVectorStore,
    validate_collection_name,
)
from tests.chroma_fakes import (
    DEFAULT_N_RESULTS,
    FakeChromaModule,
    FakeChromaModuleMissingPersistentClient,
    FakeClient,
    FakeCollection,
    to_float32,
)
from tests.vectorstore_samples import (
    RECORD_IDS,
    SAMPLE_QUERIES,
    VECTOR_DIMENSION,
    brute_force_top,
    sample_records,
)

#: "与查询完全同向、但模长是 10 倍"的那条：``ip`` 的 top-1 是它，``cosine`` 不是.
OVERSIZED_ID = RECORD_IDS[3]

#: 单位长度的查询轴（与 ``SAMPLE_QUERIES['q_axis']`` 同向，单独写方便直接传给假集合）.
AXIS_QUERY: list[float] = [1.0] + [0.0] * (VECTOR_DIMENSION - 1)


def build_store(
    metric: str = METRIC_COSINE,
    *,
    path: str = "",
    collection: str = DEFAULT_COLLECTION_NAME,
    dimension: int | None = None,
    module: Any = None,
    client: Any = None,
) -> tuple[ChromaVectorStore, Any]:
    """建一个注入了假 chromadb 的后端，并把假模块一并返回（断言要用到它）."""
    resolved_module = FakeChromaModule() if module is None else module
    store = ChromaVectorStore(
        metric=metric,
        dimension=dimension,
        path=path,
        collection=collection,
        client=client,
        chromadb_module=resolved_module,
    )
    return store, resolved_module


def fake_collection(store: ChromaVectorStore, module: Any) -> FakeCollection:
    """取出适配器正在用的那个假集合（"交出去的参数"要靠它记录）."""
    return module.last_client.get_collection(store.describe_extra()["collection"])


def seeded(metric: str = METRIC_COSINE) -> tuple[ChromaVectorStore, FakeCollection]:
    """建库 + 写入六条样本记录，返回后端与它的假集合."""
    store, module = build_store(metric=metric)
    store.upsert(sample_records(metric=metric))
    return store, fake_collection(store, module)


# --------------------------------------------------------------------- 基础往返


def test_collection_is_created_without_embedding_function() -> None:
    """建集合必须显式传 ``embedding_function=None``（否则会触发一次模型下载）."""
    store, module = build_store()
    kwargs = module.last_client.collection_kwargs[-1]

    assert kwargs["name"] == DEFAULT_COLLECTION_NAME
    assert kwargs["embedding_function"] is None
    assert kwargs["configuration"] == {"hnsw": {"space": "cosine"}}
    assert store.describe_extra()["embedding_function"] == "none"
    assert store.name == "chroma"
    assert store.requires == ("chromadb",)
    assert store.native_metadata is True


def test_upsert_query_get_delete_roundtrip() -> None:
    """写入 / 查询 / 取单条 / 删除 / count 的最小闭环."""
    store, collection = seeded()

    assert store.count() == 6
    assert store.ids() == sorted(RECORD_IDS)
    record = store.get(RECORD_IDS[0])
    assert record is not None
    assert record.dimension == VECTOR_DIMENSION
    assert record.metadata["strategy"] == "recursive"
    assert record.text.startswith("固定长度分块")
    assert store.get("这个 id 不存在") is None

    result = store.query(list(SAMPLE_QUERIES["q_axis"]), 3)
    assert result.ids() == [RECORD_IDS[0], OVERSIZED_ID, RECORD_IDS[2]]
    assert result.candidates == 6
    assert result.filter_applied is False

    assert store.delete([RECORD_IDS[0], "这个 id 不存在"]) == 1
    assert store.count() == 5
    assert collection.count() == 5
    assert store.delete(["还是不存在"]) == 0


def test_add_rejects_existing_ids_and_leaves_the_store_untouched() -> None:
    """``add`` 撞 id 要整体拒绝，且库里状态一个字节都不变."""
    store, _ = seeded()

    with pytest.raises(RecordError) as excinfo:
        store.add(sample_records())

    assert "add() 撞上 6 个已存在的 id" in str(excinfo.value)
    assert "要覆盖请用 upsert()" in str(excinfo.value)
    assert store.count() == 6


@pytest.mark.parametrize("metric", [METRIC_COSINE, METRIC_INNER_PRODUCT, METRIC_L2])
def test_replay_reports_unchanged_and_never_touches_the_collection(metric: str) -> None:
    """重放同一批 → ``unchanged=6``，且**没有发生任何一次写入**.

    这条同时守住两件事：一是 ``unchanged`` 这个态在 Chroma 上真的可达
    （向量存成 float32、读回来带量化误差，因此比较必须带容差——
    三种度量都要过，因为 ``l2`` 的分量可以到 10 的量级，容差也得撑得住）；
    二是"未变"的记录确实没被交给库（写入次数没有增加）。
    """
    store, collection = seeded(metric=metric)
    assert len(collection.add_calls) == 1

    report = store.upsert(sample_records(metric=metric))

    assert report.added == 0
    assert report.updated == 0
    assert report.unchanged == 6
    assert report.written == 0
    assert report.changed == 0
    assert len(collection.add_calls) == 1
    assert store.count() == 6


def test_changed_vector_reports_exactly_one_update() -> None:
    """换掉一条记录的向量 → 恰好一条 ``updated``，其余仍然 ``unchanged``."""
    store, _ = seeded()
    records = sample_records()
    original = records[2]
    records[2] = make_record(
        original.record_id,
        [0.2, 0.96**0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        original.text,
        dict(original.metadata),
        metric=METRIC_COSINE,
    )

    report = store.upsert(records)

    assert (report.added, report.updated, report.unchanged) == (0, 1, 5)
    stored = store.get(RECORD_IDS[2])
    assert stored is not None
    assert stored.vector[0] == pytest.approx(0.2, abs=1e-6)
    assert stored.vector[1] == pytest.approx(0.96**0.5, abs=1e-6)


def test_changed_metadata_reports_updated() -> None:
    """只改元数据也算"变了"：文本与元数据是精确比较的，不吃向量的容差."""
    store, _ = seeded()
    records = sample_records()
    original = records[0]
    records[0] = make_record(
        original.record_id,
        list(original.vector),
        original.text,
        {**original.metadata, "index": 99},
        metric=METRIC_COSINE,
    )

    report = store.upsert(records)

    assert (report.added, report.updated, report.unchanged) == (0, 1, 5)
    stored = store.get(RECORD_IDS[0])
    assert stored is not None
    assert stored.metadata["index"] == 99


# ------------------------------------------------------------------ 与参照实现对账


@pytest.mark.parametrize("metric", [METRIC_COSINE, METRIC_INNER_PRODUCT, METRIC_L2])
@pytest.mark.parametrize("query_name", ["q_axis", "q_tilt"])
def test_parity_with_brute_force_top(metric: str, query_name: str) -> None:
    """三种度量 × 两个查询：top-3 的 id 与分数逐条对上独立参照实现."""
    store, _ = seeded(metric=metric)
    query = list(SAMPLE_QUERIES[query_name])

    result = store.query(query, 3)
    expected = brute_force_top(sample_records(metric=metric), query, metric, 3)

    assert result.ids() == [record_id for record_id, _ in expected]
    assert result.scores == pytest.approx([score for _, score in expected], abs=1e-6)
    assert [hit.rank for hit in result.hits] == [0, 1, 2]


@pytest.mark.parametrize(
    ("metric", "expected_top1"),
    [
        (METRIC_COSINE, RECORD_IDS[0]),
        (METRIC_INNER_PRODUCT, OVERSIZED_ID),
        (METRIC_L2, RECORD_IDS[0]),
    ],
)
def test_metric_decides_who_is_nearest(metric: str, expected_top1: str) -> None:
    """三个度量必须给出各自的结果：``ip`` 认模长，``cosine``/``l2`` 不认.

    ``r_d`` 与查询完全同向、模长却是 10 倍，因此它是 ``ip`` 的 top-1；
    ``cosine`` 把这个模长抹掉之后它与 ``r_a`` 并列（按 id 升序排前面的是 ``r_a``）。
    **若适配器把三个 space 都按同一个公式算，这个用例会红。**
    """
    store, _ = seeded(metric=metric)
    assert store.describe_extra()["space"] == SPACE_BY_METRIC[metric]

    result = store.query(list(SAMPLE_QUERIES["q_axis"]), 1)

    assert result.ids() == [expected_top1]


def test_cosine_tie_is_broken_by_id_not_by_insertion_order() -> None:
    """并列时的断法是"id 升序"，与 ``types.sort_hits`` 的规则一致."""
    store, _ = seeded()
    result = store.query(list(SAMPLE_QUERIES["q_axis"]), 2)

    assert result.ids() == [RECORD_IDS[0], OVERSIZED_ID]
    assert result.scores == pytest.approx([1.0, 1.0], abs=1e-6)


# ------------------------------------------------------------------ 距离换算


def test_library_distances_are_smaller_is_nearer() -> None:
    """先钉住库给的是**距离**：数值升序、最近的排第一（与 FAISS 的分数方向相反）."""
    _, collection = seeded()
    raw = collection.query(
        query_embeddings=[list(AXIS_QUERY)], n_results=3, include=["documents", "distances"]
    )

    assert raw["ids"][0] == [RECORD_IDS[0], OVERSIZED_ID, RECORD_IDS[2]]
    assert raw["distances"][0] == pytest.approx([0.0, 0.0, 0.2], abs=1e-6)


@pytest.mark.parametrize("metric", [METRIC_COSINE, METRIC_INNER_PRODUCT])
def test_score_is_one_minus_the_library_distance(metric: str) -> None:
    """``cosine`` / ``ip``：``score == 1 − distance``，且 distance 就是库给的那个数."""
    store, collection = seeded(metric=metric)
    result = store.query(list(SAMPLE_QUERIES["q_axis"]), 3)
    raw = collection.query(
        query_embeddings=[list(SAMPLE_QUERIES["q_axis"])], n_results=3, include=["distances"]
    )

    assert result.ids() == raw["ids"][0]
    for hit, raw_distance in zip(result.hits, raw["distances"][0]):
        assert hit.distance == pytest.approx(raw_distance, abs=1e-9)
        assert hit.score == pytest.approx(1.0 - raw_distance, abs=1e-9)


def test_score_is_negative_the_library_distance_for_l2() -> None:
    """``l2``：``score == −distance``；用 ``1 − distance`` 那个公式是**错的**."""
    store, collection = seeded(metric=METRIC_L2)
    result = store.query(list(SAMPLE_QUERIES["q_axis"]), 3)
    raw = collection.query(
        query_embeddings=[list(SAMPLE_QUERIES["q_axis"])], n_results=3, include=["distances"]
    )

    assert result.ids() == [RECORD_IDS[0], RECORD_IDS[2], RECORD_IDS[5]]
    for hit, raw_distance in zip(result.hits, raw["distances"][0]):
        assert hit.distance == pytest.approx(raw_distance, abs=1e-9)
        assert hit.score == pytest.approx(-raw_distance, abs=1e-9)
        assert hit.score != pytest.approx(1.0 - raw_distance, abs=1e-6)
    assert result.hits[0].distance == min(raw["distances"][0])


# ------------------------------------------------------------------ 集合名校验


@pytest.mark.parametrize(
    ("name", "fragment"),
    [
        ("ab", "不在允许范围 3~512"),
        ("x" * (COLLECTION_NAME_MAX_LENGTH + 1), "不在允许范围 3~512"),
        ("SmartResearch", "必须以小写字母或数字开头"),
        ("smart.", "必须以小写字母或数字结尾"),
        ("smart..research", "不能出现连续两个点"),
        ("192.168.1.1", "是一个合法 IP 地址"),
        ("10.0.0.1", "是一个合法 IP 地址"),
        ("smart research", "含有不允许的字符"),
        ("smartResearch", "含有不允许的字符"),
    ],
)
def test_collection_name_violations(name: str, fragment: str) -> None:
    """九种违反官方约束的写法各自抛 ``RecordError``，消息里点明是哪一条."""
    with pytest.raises(RecordError) as excinfo:
        validate_collection_name(name)
    assert fragment in str(excinfo.value)


@pytest.mark.parametrize(
    "name",
    [DEFAULT_COLLECTION_NAME, "abc", "a-b.c_2", "1repo", "x" * COLLECTION_NAME_MAX_LENGTH],
)
def test_collection_name_accepts_official_shapes(name: str) -> None:
    """合法的名字**原样返回**（不转小写、不替换字符：静默改名比报错危险得多）."""
    assert validate_collection_name(name) == name
    assert COLLECTION_NAME_MIN_LENGTH == 3
    assert COLLECTION_NAME_MAX_LENGTH == 512


def test_invalid_collection_name_is_rejected_before_any_client_is_built() -> None:
    """名字不合法时连客户端都不该建：校验必须发生在动库之前."""
    module = FakeChromaModule()

    with pytest.raises(RecordError):
        ChromaVectorStore(collection="ab", chromadb_module=module)

    assert module.clients == []


@pytest.mark.parametrize("bad_name", [123, None, b"smart", ["smart"]])
def test_collection_name_must_be_a_string(bad_name: Any) -> None:
    """传进来的根本不是字符串：这条要在正则校验之前就报出来（否则是 TypeError）."""
    with pytest.raises(RecordError) as excinfo:
        validate_collection_name(bad_name)
    assert "collection 必须是字符串" in str(excinfo.value)
    assert DEFAULT_COLLECTION_NAME in str(excinfo.value)


# ------------------------------------------------------------------ where 下推


def test_where_is_pushed_down_to_the_native_query() -> None:
    """``where`` 原样交给原生查询（不改写、不拆成两趟）."""
    store, collection = seeded()
    where = {"strategy": "structural"}

    result = store.query(list(SAMPLE_QUERIES["q_axis"]), 3, where=where)

    pushed = collection.last_query["where"]
    assert pushed == {"strategy": "structural"}
    assert sorted(pushed) == sorted(where)
    assert collection.last_query["include"] == ["documents", "metadatas", "distances"]
    assert result.filter_applied is True
    assert result.ids() == [OVERSIZED_ID, RECORD_IDS[5]]


def test_n_results_counts_after_filtering_not_before() -> None:
    """``n_results`` 必须是**过滤之后**要的条数（"先取 top-k 再过滤"会少返回）.

    数据刻意构造过：全局最像的 3 条里，只有 1 条满足 ``strategy=structural``。
    """
    store, collection = seeded()
    where = {"strategy": "structural"}

    result = store.query(list(SAMPLE_QUERIES["q_axis"]), 3, where=where)

    assert result.ids() == [OVERSIZED_ID, RECORD_IDS[5]]
    assert result.count == 2
    assert result.candidates == 2
    assert collection.last_query["n_results"] == 2

    # 对照组：不过滤时最像的 3 条里只有一条满足条件，正是"先排序再过滤"的受害者
    naive = store.query(list(SAMPLE_QUERIES["q_axis"]), 3)
    assert naive.ids() == [RECORD_IDS[0], OVERSIZED_ID, RECORD_IDS[2]]
    assert [record_id for record_id in naive.ids() if record_id == OVERSIZED_ID] == [OVERSIZED_ID]


def test_where_with_no_match_short_circuits_without_querying() -> None:
    """过滤后一条都没有 → 直接返回空结果，且**不去打扰库**."""
    store, collection = seeded()

    result = store.query(list(SAMPLE_QUERIES["q_axis"]), 3, where={"strategy": "nope"})

    assert result.count == 0
    assert result.candidates == 0
    assert result.filter_applied is True
    assert collection.query_calls == []


def test_query_skips_rows_that_cannot_be_read_back(monkeypatch: Any) -> None:
    """某条向量取不回来时**跳过并继续**，而不是让整次查询失败.

    ``candidates`` 仍然是 6 且库确实被查过——这说明空结果是"记录表出了问题"，
    不是"过滤器把候选清零"，两种情形在报告里必须能区分开。
    """
    store, collection = seeded()
    monkeypatch.setattr(store, "_records", lambda _ids: {})

    result = store.query(list(SAMPLE_QUERIES["q_axis"]), 3)

    assert result.count == 0
    assert result.candidates == 6
    assert result.filter_applied is False
    assert collection.query_calls != []


def test_delete_by_where_uses_the_native_filter() -> None:
    """``delete(where=...)`` 走 ``get(where=...)`` 取 id，并返回真正删掉的条数."""
    store, collection = seeded()
    glossary_ids = sorted([RECORD_IDS[4], RECORD_IDS[5]])

    assert store.delete(where={"topic": "glossary"}) == 2

    assert collection.delete_calls[-1]["ids"] == glossary_ids
    assert store.ids() == sorted(set(RECORD_IDS) - set(glossary_ids))
    assert store.count() == 4

    with pytest.raises(RecordError) as excinfo:
        store.delete([RECORD_IDS[0]], where={"topic": "glossary"})
    assert "只能按 ids 或按 where 之一删除" in str(excinfo.value)


# ----------------------------------------------------------- 原语：_ranked_ids


def test_ranked_ids_sorts_by_score_and_reports_the_raw_distance_include() -> None:
    """``_ranked_ids`` 走一次原生 ``query``，分数降序；并列按 id 升序."""
    store, collection = seeded()

    ranked = store._ranked_ids(list(SAMPLE_QUERIES["q_axis"]), limit=3)

    assert [record_id for record_id, _ in ranked] == [RECORD_IDS[0], OVERSIZED_ID, RECORD_IDS[2]]
    assert [score for _, score in ranked] == pytest.approx([1.0, 1.0, 0.8], abs=1e-6)
    assert collection.last_query["include"] == ["distances"]
    assert collection.last_query["where"] is None


def test_ranked_ids_ranks_inside_the_restrict_set() -> None:
    """``restrict`` 必须在**集合内**排序，而不是"全库取 limit 再求交集".

    ``restrict`` 里只有全局分数最低的那一条，而 ``limit=2``：
    若按 ``min(limit, count)`` 取深度，全库前 2 名都在 restrict 之外，
    结果会变成空——**少返回而不是报错**，正是这一条要拦住的那种故障。
    """
    store, collection = seeded()
    restrict = frozenset({RECORD_IDS[4]})

    ranked = store._ranked_ids(list(SAMPLE_QUERIES["q_axis"]), limit=2, restrict=restrict)

    assert [record_id for record_id, _ in ranked] == [RECORD_IDS[4]]
    assert collection.last_query["n_results"] == 6


def test_ranked_ids_short_circuits_on_empty_store_or_bad_limit() -> None:
    """空库与 ``limit < 1`` 都直接返回空列表（不去问库：库里也没有）."""
    empty, empty_module = build_store()
    assert empty._ranked_ids(list(SAMPLE_QUERIES["q_axis"]), limit=3) == []
    assert empty_module.last_client.last_collection.query_calls == []

    store, _ = seeded()
    assert store._ranked_ids(list(SAMPLE_QUERIES["q_axis"]), limit=0) == []


def test_ranked_ids_with_an_empty_restrict_set_returns_nothing() -> None:
    """``restrict`` 是空集合时返回空（不是"不过滤"）——两者相差的是整库结果."""
    store, _ = seeded()

    ranked = store._ranked_ids(
        list(SAMPLE_QUERIES["q_axis"]), limit=3, restrict=frozenset()
    )

    assert ranked == []


def test_find_ids_none_means_no_filter_not_empty_match() -> None:
    """``_find_ids(None)`` 返回 ``None``（"不过滤"），空集合才是"一条都没匹配上"."""
    store, _ = seeded()

    assert store._find_ids(None) is None
    assert store._find_ids({"strategy": "structural"}) == {OVERSIZED_ID, RECORD_IDS[5]}
    assert store._find_ids({"strategy": "nope"}) == set()


def test_find_ids_asks_the_library_for_ids_only() -> None:
    """取候选 id 时要 ``include=[]``：为了数一遍 id 不该把整库的正文搬进内存."""
    store, collection = seeded()

    store._find_ids({"strategy": "structural"})

    assert collection.get_calls[-1]["where"] == {"strategy": "structural"}
    assert collection.get_calls[-1]["include"] == []


# ------------------------------------------------------------ top_k / min_score


def test_top_k_out_of_range_messages_match_the_base_class() -> None:
    """``top_k`` 的两条边界与 ``base.query`` 用同一套措辞（调用方不必分后端处理）."""
    store, _ = seeded()
    query = list(SAMPLE_QUERIES["q_axis"])

    with pytest.raises(FilterError) as low:
        store.query(query, 0)
    assert "top_k 必须 >= 1，收到 0" in str(low.value)
    assert "请用 count()" in str(low.value)

    with pytest.raises(FilterError) as high:
        store.query(query, MAX_TOP_K + 1)
    assert f"top_k={MAX_TOP_K + 1} 超过上限 {MAX_TOP_K}" in str(high.value)
    assert "ids() + get_many()" in str(high.value)


def test_min_score_cuts_hits_but_not_candidates() -> None:
    """``min_score`` 切掉的是命中，不是候选：``candidates`` 仍报过滤后的候选数."""
    store, _ = seeded()

    result = store.query(list(SAMPLE_QUERIES["q_axis"]), 5, min_score=0.9)

    assert result.ids() == [RECORD_IDS[0], OVERSIZED_ID]
    assert result.candidates == 6
    assert result.scores == pytest.approx([1.0, 1.0], abs=1e-6)


def test_top_k_larger_than_the_collection_returns_everything() -> None:
    """库比要求的小：返回库里的全部（而不是拿占位值补齐）."""
    store, collection = seeded()

    result = store.query(list(SAMPLE_QUERIES["q_axis"]), 20)

    assert result.count == 6
    assert collection.last_query["n_results"] == 6
    assert result.ids() == [
        RECORD_IDS[0],
        OVERSIZED_ID,
        RECORD_IDS[2],
        RECORD_IDS[5],
        RECORD_IDS[1],
        RECORD_IDS[4],
    ]


# ------------------------------------------------------------------ 维度


def test_clear_keeps_metric_and_dimension() -> None:
    """``clear()`` 只清数据：度量与维度留着，便于复用同一个实例重建."""
    store, _ = seeded()

    store.clear()

    assert store.count() == 0
    assert store.ids() == []
    assert store.metric == METRIC_COSINE
    assert store.dimension == VECTOR_DIMENSION
    assert store.query(list(SAMPLE_QUERIES["q_axis"]), 3).count == 0


def test_given_dimension_rejects_a_different_one() -> None:
    """构造时给了 dimension，写入不同维度 → ``VectorError``（在本包这一层就拦下）."""
    store, _ = build_store(dimension=VECTOR_DIMENSION)

    with pytest.raises(VectorError) as excinfo:
        store.upsert([make_record("short", [1.0, 0.0, 0.0, 0.0])])

    assert "维度不一致：本库是 8 维，收到 4 维" in str(excinfo.value)
    assert "换过 embedding 提供方就要重建库" in str(excinfo.value)
    assert store.count() == 0


@pytest.mark.parametrize("dimension", [0, -1])
def test_zero_dimension_is_rejected_by_the_constructor(dimension: int) -> None:
    """``dimension=0`` 不是"缺省"而是非法：0 是 ``UNKNOWN_DIMENSION`` 的哨兵值.

    校验来自 ``base.VectorBackend._check_dimension``（全包唯一的实现，
    子类不再重复覆写）——``base.__init__`` 拿到 ``dimension`` 就会调用它。
    """
    module = FakeChromaModule()

    with pytest.raises(VectorError) as excinfo:
        ChromaVectorStore(dimension=dimension, chromadb_module=module)

    assert "dimension 必须是 >= 1 的整数" in str(excinfo.value)
    assert "UNKNOWN_DIMENSION" in str(excinfo.value)
    assert module.clients == []


def test_dimension_is_learned_from_the_first_write() -> None:
    """未给 dimension：第一次写入定下维度，之后不同维度就报错."""
    store, _ = build_store()
    assert store.dimension == 0

    store.upsert(sample_records())

    assert store.dimension == VECTOR_DIMENSION
    with pytest.raises(VectorError):
        store.upsert([make_record("short", [1.0, 0.0, 0.0, 0.0])])


def test_dimension_is_recovered_from_an_existing_collection(tmp_path: Path) -> None:
    """重开一个已有数据的库：维度要从库里读回来（否则 base 的护栏会失效）."""
    path = str(tmp_path / "chroma_reopen")
    first, _ = build_store(path=path)
    first.upsert(sample_records())

    second, _ = build_store(path=path)

    assert second.count() == 6
    assert second.dimension == VECTOR_DIMENSION
    assert second.query(list(SAMPLE_QUERIES["q_axis"]), 1).ids() == [RECORD_IDS[0]]
    reopened = second.get(RECORD_IDS[0])
    assert reopened is not None
    assert reopened.text.startswith("固定长度分块")


def test_query_dimension_mismatch_raises_vector_error() -> None:
    """查询向量维度与库不一致 → ``VectorError``，消息指向"编码器不是同一个"."""
    store, _ = seeded()

    with pytest.raises(VectorError) as excinfo:
        store.query([1.0, 0.0, 0.0, 0.0], 3)

    assert "查询向量是 4 维，本库是 8 维" in str(excinfo.value)


class _ProbeWithoutEmbeddings:
    """只为覆盖"维度探测拿不到向量"这条护栏的最小替身.

    正常安装的 Chroma 一定会在 ``include=["embeddings"]`` 时给出向量，
    因此这一条是防御性的：库没有给出向量时**保持"维度未定"而不是猜一个值**，
    否则 base 的维度护栏会拿一个猜出来的数字去拦下合法的写入。
    """

    def __init__(self, total: int) -> None:
        self._total = total

    def count(self) -> int:
        return self._total

    def get(self, **kwargs: Any) -> dict[str, Any]:
        return {"ids": ["a"]}


def test_dimension_probe_keeps_the_dimension_unknown_when_rows_have_no_vector(
    monkeypatch: Any,
) -> None:
    """探测失败 → 维度仍是未定（0），而不是被猜成一个具体数字."""
    store, _ = build_store()
    monkeypatch.setattr(store, "_collection", _ProbeWithoutEmbeddings(3))

    store._detect_dimension()

    assert store.has_dimension is False
    assert store.dimension == 0


# ------------------------------------------------------------------ 持久化


def test_persist_on_memory_client_is_refused() -> None:
    """内存客户端（``chromadb.Client()``）没有目录可落盘 → 报错而不是假成功."""
    store, _ = build_store()

    assert store.describe_extra()["client"] == "memory"
    assert store.info().persistent is False

    with pytest.raises(BackendUnavailable) as excinfo:
        store.persist()

    assert "内存客户端不落盘" in str(excinfo.value)
    assert "请用 path=... 构造" in str(excinfo.value)
    assert "chromadb.PersistentClient(path=...)" in str(excinfo.value)


def test_persist_on_persistent_client_returns_the_directory(tmp_path: Path) -> None:
    """带 path 的客户端：``persist()`` 返回目录本身（数据从第一条起就在盘上）."""
    directory = tmp_path / "chroma_store"
    store, _ = build_store(path=str(directory))
    store.upsert(sample_records())

    assert store.persist() == str(directory)
    assert directory.is_dir()
    assert (directory / "fake_chroma.pkl").exists()
    assert store.describe_extra()["client"] == "persistent"

    info = store.info()
    assert info.persistent is True
    assert info.location == str(directory)
    assert info.count == 6


def test_persist_to_another_directory_is_refused(tmp_path: Path) -> None:
    """Chroma 的目录绑定在客户端上：改存到别处要新建实例，不能静默忽略."""
    store, _ = build_store(path=str(tmp_path / "here"))

    with pytest.raises(BackendUnavailable) as excinfo:
        store.persist(str(tmp_path / "there"))

    assert "落盘目录在客户端构造时就已定死" in str(excinfo.value)
    assert "请用 path=" in str(excinfo.value)


def test_load_reacquires_the_collection_and_guards_the_path(tmp_path: Path) -> None:
    """``load()`` 只重新拿句柄；换目录则报错（数据已在盘上，不需要重写）."""
    store, _ = build_store(path=str(tmp_path / "chroma_load"))
    store.upsert(sample_records())

    store.load()

    assert store.count() == 6
    assert store.dimension == VECTOR_DIMENSION

    with pytest.raises(BackendUnavailable) as excinfo:
        store.load(str(tmp_path / "elsewhere"))
    assert "没法改读" in str(excinfo.value)

    memory, _ = build_store()
    with pytest.raises(BackendUnavailable) as memory_error:
        memory.load()
    assert "没有可读取的目录" in str(memory_error.value)


# ------------------------------------------------------------------ 模块解析


def test_missing_chromadb_module_gives_an_install_hint(monkeypatch: Any) -> None:
    """没装 chromadb：抛 ``BackendUnavailable`` 并说明装什么、以及退路是什么.

    用 ``sys.modules`` 里塞一个 ``None`` 来把"没装"这件事变成确定性的：
    Python 的导入机制在这种情况下一定抛 ``ImportError``，
    于是断言不依赖"本机到底装没装 chromadb"。
    """
    monkeypatch.setitem(sys.modules, "chromadb", None)

    with pytest.raises(BackendUnavailable) as excinfo:
        ChromaVectorStore()

    message = str(excinfo.value)
    assert "缺少可选依赖 chromadb" in message
    assert "pip install chromadb" in message
    assert "flat 后端零依赖" in message


def test_module_without_persistent_client_is_rejected() -> None:
    """装了但缺 ``PersistentClient``（版本过旧/残缺）：在建客户端之前就拦下."""
    module = FakeChromaModuleMissingPersistentClient()

    with pytest.raises(BackendUnavailable) as excinfo:
        ChromaVectorStore(chromadb_module=module)

    assert "缺少 PersistentClient" in str(excinfo.value)
    assert "pip install -U chromadb" in str(excinfo.value)
    assert module.clients == []


def test_injected_client_works_without_a_module() -> None:
    """注入客户端时跳过模块级检查：能力全在那个对象上（测试与自建封装的入口）."""
    client = FakeClient()

    store, _ = build_store(client=client, module=None)

    assert store.count() == 0
    assert store.describe_extra()["client"] == "memory"
    assert client.collection_kwargs[-1]["embedding_function"] is None


def test_injected_persistent_client_exposes_its_directory(tmp_path: Path) -> None:
    """注入的客户端若带 path，落盘目录要被认下来（persist 才有东西可返回）."""
    directory = tmp_path / "injected"

    store, _ = build_store(client=FakeClient(str(directory)))

    assert store.location == str(directory)
    assert store.persist() == str(directory)
    assert store.describe_extra()["client"] == "persistent"


def test_explicit_path_wins_over_the_injected_client_directory(tmp_path: Path) -> None:
    """同时给了 ``path`` 与 ``client`` 时以 ``path`` 为准：显式参数胜过推断."""
    explicit = tmp_path / "explicit"
    store, _ = build_store(path=str(explicit), client=FakeClient(str(tmp_path / "other")))

    assert store.location == str(explicit)


def test_module_found_on_sys_modules_is_used(monkeypatch: Any) -> None:
    """本机"确实装了 chromadb"的那条分支：``import`` 成功，随后做 API 检查.

    不传 ``chromadb_module``，而是把假模块塞进 ``sys.modules``——这样既走到了
    "导入成功"的路径，又完全不碰网络与磁盘（真库默认要下模型）。
    """
    module = FakeChromaModule()
    monkeypatch.setitem(sys.modules, "chromadb", module)

    store = ChromaVectorStore()

    assert store.count() == 0
    assert module.last_client is not None
    assert module.last_client.collection_kwargs[-1]["embedding_function"] is None


# --------------------------------------------------------- 假库本身的忠实性


def test_fake_collection_has_no_catch_all_kwargs() -> None:
    """假库的每个方法都是显式签名：适配器一旦发明参数就会 ``TypeError``."""
    owner = FakeClient()
    collection = owner.create_collection(name="abc")

    with pytest.raises(TypeError):
        collection.add(ids=["a"], embeddings=[[1.0]], unknown_argument=True)
    with pytest.raises(TypeError):
        collection.get(top_k=3)
    with pytest.raises(TypeError):
        collection.query(query_embeddings=[[1.0]], top_k=3)
    with pytest.raises(TypeError):
        owner.create_collection(name="xyz", top_k=3)


@pytest.mark.parametrize("bad_value", [None, {"nested": 1}, [], [1, "mixed"], [1.0, True]])
def test_fake_collection_rejects_illegal_metadata_values(bad_value: Any) -> None:
    """元数据只收标量或**同类型标量数组**：None / dict / 空数组 / 混类型都要报错."""
    owner = FakeClient()
    collection = owner.create_collection(name="abc")

    with pytest.raises(ValueError) as excinfo:
        collection.upsert(ids=["a"], embeddings=[[1.0, 0.0]], metadatas=[{"field": bad_value}])

    assert "metadata value" in str(excinfo.value)


def test_fake_collection_accepts_legal_metadata_values() -> None:
    """对照组：四种标量与三种同类型数组都必须收下（否则上面的报错没有意义）."""
    owner = FakeClient()
    collection = owner.create_collection(name="abc")

    collection.upsert(
        ids=["a"],
        embeddings=[[1.0, 0.0]],
        metadatas=[
            {"name": "x", "count": 2, "score": 0.5, "flag": True, "tags": ["a", "b"]}
        ],
        documents=["正文"],
    )

    assert collection.count() == 1
    stored = collection.get(ids=["a"], include=["metadatas", "documents"])
    assert stored["metadatas"][0]["tags"] == ["a", "b"]
    assert stored["documents"] == ["正文"]


def test_fake_collection_rejects_embedding_dimension_mismatch() -> None:
    """集合级维度写死：第二次写入换个维度 → 报维度错误（真库如此）."""
    owner = FakeClient()
    collection = owner.create_collection(name="abc")
    collection.add(ids=["a"], embeddings=[[1.0, 0.0, 0.0]])

    with pytest.raises(ValueError) as excinfo:
        collection.add(ids=["b"], embeddings=[[1.0, 0.0]])

    assert "does not match index dimensionality" in str(excinfo.value)


def test_fake_query_is_two_dimensional_while_get_is_flat() -> None:
    """形状必须与真库一致：``query`` 二维（外层每个查询一组）、``get``/``peek`` 一维."""
    store, collection = seeded()

    query_result = collection.query(query_embeddings=[list(AXIS_QUERY)], n_results=2)
    flat_result = collection.get(include=["documents"])
    peeked = collection.peek(limit=2)

    assert len(query_result["ids"]) == 1
    assert len(query_result["ids"][0]) == 2
    assert len(query_result["metadatas"][0]) == 2
    assert isinstance(flat_result["ids"][0], str)
    assert len(flat_result["ids"]) == 6
    assert "documents" in flat_result
    assert "metadatas" not in flat_result
    assert isinstance(peeked["ids"][0], str)
    assert len(peeked["ids"]) == 2
    assert store.count() == 6


def test_fake_update_only_touches_existing_ids() -> None:
    """``update`` 与 ``upsert`` 不同：id 不存在要报错（本包的适配器因此只用 upsert）."""
    owner = FakeClient()
    collection = owner.create_collection(name="abc")
    collection.add(ids=["a"], embeddings=[[1.0, 0.0]], documents=["旧"])

    collection.update(ids=["a"], embeddings=[[0.0, 1.0]], documents=["新"])

    assert collection.get(ids=["a"], include=["documents"])["documents"] == ["新"]
    with pytest.raises(ValueError) as excinfo:
        collection.update(ids=["missing"], embeddings=[[1.0, 0.0]])
    assert "Could not find ids in collection" in str(excinfo.value)


def test_fake_query_defaults_to_ten_results() -> None:
    """真库的 ``n_results`` 默认是 10 而不是 5：这个默认值会被调用方严重忽略."""
    _, collection = seeded()

    collection.query(query_embeddings=[list(AXIS_QUERY)])

    assert DEFAULT_N_RESULTS == 10
    assert collection.last_query["n_results"] == 10


def test_fake_ranks_each_query_vector_separately() -> None:
    """多个查询向量各自排各自的（外层是"每个查询一组"，不是"每条命中一组"）."""
    _, collection = seeded()

    result = collection.query(
        query_embeddings=[list(SAMPLE_QUERIES["q_axis"]), list(SAMPLE_QUERIES["q_tilt"])],
        n_results=1,
        include=["distances"],
    )

    assert result["ids"] == [[RECORD_IDS[0]], [RECORD_IDS[2]]]
    assert result["distances"][0] == pytest.approx([0.0], abs=1e-6)
    assert result["distances"][1] == pytest.approx([0.04], abs=1e-6)


# ------------------------------------------------------- 适配器的解析与比较工具


def test_records_helper_guards_empty_input_and_rows_without_vectors() -> None:
    """``_records`` / ``_records_from`` 对"缺字段的行"的处理：空输入返回空、缺向量跳过.

    ``include`` 里没要 ``embeddings`` 时库就会返回"只有 id 的行"，
    这种行构不成 ``VectorRecord``（向量是必填的），因此只能跳过。
    """
    store, _ = seeded()

    assert store._records([]) == {}
    assert chroma_module._records_from({"ids": ["a"], "documents": ["正文"]}) == {}

    parsed = chroma_module._records_from(
        {
            "ids": ["a"],
            "embeddings": [[1.0, 0.0]],
            "documents": [None],
            "metadatas": [None],
        }
    )
    assert parsed["a"].vector == (1.0, 0.0)
    assert parsed["a"].text == ""
    assert parsed["a"].metadata == {}


def test_first_row_tolerates_missing_rows() -> None:
    """``_first_row`` 对空值返回空列表（库没命中时 ``ids`` 是 ``[[]]`` 或 ``[]``）."""
    assert chroma_module._first_row(None) == []
    assert chroma_module._first_row([]) == []
    assert chroma_module._first_row([[1, 2], [3, 4]]) == [1, 2]


def test_records_equal_tolerates_float32_noise_but_not_content_changes() -> None:
    """``_records_equal`` 的比较规则：向量吃 float32 容差，其余字段精确比较.

    这条是 ``unchanged`` 能不能成立的全部依据，因此五个分支各自断言一次：
    向量量化噪声算"没变"，而文本、元数据、维度、向量实质变化都算"变了"。
    """
    original = VectorRecord(
        record_id="a", vector=(0.6, 0.8), text="正文", metadata={"index": 1}
    )
    quantized = VectorRecord(
        record_id="a",
        vector=(to_float32(0.6), to_float32(0.8)),
        text="正文",
        metadata={"index": 1},
    )

    assert chroma_module._records_equal(original, quantized) is True
    assert (
        chroma_module._records_equal(
            original, VectorRecord("a", (0.6, 0.8), "改过的正文", {"index": 1})
        )
        is False
    )
    assert (
        chroma_module._records_equal(
            original, VectorRecord("a", (0.6, 0.8), "正文", {"index": 2})
        )
        is False
    )
    assert (
        chroma_module._records_equal(
            original, VectorRecord("a", (0.6, 0.8, 0.0), "正文", {"index": 1})
        )
        is False
    )
    assert (
        chroma_module._records_equal(
            original, VectorRecord("a", (0.2, 0.9798), "正文", {"index": 1})
        )
        is False
    )
