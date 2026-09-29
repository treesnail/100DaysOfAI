"""``vectorstore.evaluate`` 的对账测试：用**独立实现**的期望值核对三个后端（M6-D3）.

为什么这里的断言值得写：``evaluate.py`` 的缺省参照是 ``metrics.score``，
它与被测后端共享同一段排序代码，于是"一致"只证明它等于它自己
（见该模块 docstring 的"降级参照为什么弱"）。因此本文件里**每一条真正的对账
都显式传入 ``brute_force_top`` 或手工排出的期望值**——

```text
判定的依据必须与被判定的实现相互独立，否则它只是在复述实现。
```

用例分成六组，对应六类"会悄悄错掉"的事：

```text
指标        recall_at_k / rank_agreement 的四个边界（空参照、k=0、短候选、首名就不同）
对账        flat 对三种度量与 brute_force_top 逐位一致（agreed==top_k、overlap==1.0）
刻意不一致  把参照换成"把 r_d 顶到第一"的假参照 → 分歧说明里必须有"期望""实际"
可选依赖    用 sys.modules 伪造"faiss/chromadb 没装" → 折成 available=False 的行
度量差异    sample_records(metric="ip") 下 cosine 与 ip 的 top-1 必须不同（最重要的断言）
体检        index_health 在正常库上四项全过，在两个脏状态上各自报出来
```

FAISS / Chroma 一律用 ``tests/faiss_fakes.py`` / ``tests/chroma_fakes.py`` 注入，
全程离线、无真库、零网络。
"""

from __future__ import annotations

import json
import sys
from typing import Any

import pytest

from smart_research_agent.vectorstore.chroma_backend import ChromaVectorStore
from smart_research_agent.vectorstore.errors import BackendUnavailable, VectorStoreError
from smart_research_agent.vectorstore.evaluate import (
    DEFAULT_PARITY_TOP_K,
    SCORE_TOLERANCE,
    ParityRow,
    compare_backends,
    compare_metrics,
    index_health,
    rank_agreement,
    recall_at_k,
    scores_within_tolerance,
    verify_parity,
)
from smart_research_agent.vectorstore.faiss_backend import FaissVectorStore
from smart_research_agent.vectorstore.flat import FlatVectorStore
from smart_research_agent.vectorstore.types import VectorRecord, WriteReport
from tests.chroma_fakes import FakeChromaModule
from tests.faiss_fakes import make_fake_faiss_module
from tests.vectorstore_samples import (
    RECORD_IDS,
    SAMPLE_QUERIES,
    VECTOR_DIMENSION,
    brute_force_top,
    sample_records,
)

#: 与样本一致的维度（``vectorstore_samples.VECTOR_DIMENSION`` 是 8）.
DIMENSION = VECTOR_DIMENSION

#: "与查询完全同向、但模长是 10 倍"的那条：``ip`` 的 top-1 是它，``cosine`` 不是.
OVERSIZED_ID = RECORD_IDS[3]

#: 单位长度的查询轴（与 ``SAMPLE_QUERIES['q_axis']`` 同向）.
AXIS_QUERY: list[float] = list(SAMPLE_QUERIES["q_axis"])

QUERY_EXAMPLE: list[float] = [1.0] + [0.0] * (DIMENSION - 1)


def flat_records(metric: str = "cosine") -> list[Any]:
    """六条样本记录（``metric='ip'`` 时不归一化，模长差才留得住）."""
    return sample_records(metric=metric)


def expected_ranking(records: list[Any], metric: str, top_k: int = DEFAULT_PARITY_TOP_K):
    """独立实现（``brute_force_top``）给出的期望排名，不经过本包任何生产代码."""
    return brute_force_top(records, AXIS_QUERY, metric, top_k)


def faiss_factory(*, metric: str, dimension: int, path: str = "") -> FaissVectorStore:
    """构造一个注入假 faiss 模块的后端（``compare_backends`` 的工厂签名）."""
    return FaissVectorStore(
        metric=metric,
        dimension=dimension,
        path=path,
        faiss_module=make_fake_faiss_module(),
    )


def chroma_factory(*, metric: str, dimension: int, path: str = "") -> ChromaVectorStore:
    """构造一个注入假 chromadb 模块的后端（每个实例一个全新的假模块）."""
    return ChromaVectorStore(
        metric=metric,
        dimension=dimension,
        path=path,
        chromadb_module=FakeChromaModule(),
    )


def bare_faiss_factory(*, metric: str, dimension: int, path: str = "") -> FaissVectorStore:
    """**不注入**假模块的 faiss 工厂：走真实的惰性 import，用来测"缺依赖"那条路径.

    与 ``faiss_factory`` 的差别只有一处，但它是决定性的：注入了 ``faiss_module``
    之后 ``resolve_faiss_module`` 永远不会去 import faiss，于是"本机没装 faiss"
    这件事在这条路径上根本不可达（这也正是"假模块注入"与"可选依赖缺失"
    必须用两个不同工厂来测的原因）。
    """
    return FaissVectorStore(metric=metric, dimension=dimension, path=path)


def bare_chroma_factory(*, metric: str, dimension: int, path: str = "") -> ChromaVectorStore:
    """**不注入**假模块的 chroma 工厂（同上：只有这样才能走到真实的缺失分支）."""
    return ChromaVectorStore(metric=metric, dimension=dimension, path=path)


def loaded_flat(metric: str = "cosine") -> FlatVectorStore:
    """一个装好六条样本的 flat 库."""
    store = FlatVectorStore(metric=metric, dimension=DIMENSION)
    store.upsert(flat_records(metric))
    return store


# --------------------------------------------------------------------------- #
# 指标：recall_at_k / rank_agreement
# --------------------------------------------------------------------------- #


def test_recall_at_k_full_overlap() -> None:
    assert recall_at_k(RECORD_IDS, RECORD_IDS, 5) == 1.0
    assert recall_at_k(RECORD_IDS, RECORD_IDS, len(RECORD_IDS)) == 1.0


def test_recall_at_k_partial_overlap_and_short_reference() -> None:
    assert recall_at_k(["a", "b", "c"], ["b", "c", "d"], 3) == pytest.approx(2 / 3)
    # 参照不足 k 条时分母取参照长度：只看得到 2 条、2 条都在 → 1.0
    assert recall_at_k(["a", "b"], ["a", "b", "z"], 5) == 1.0


def test_recall_at_k_returns_zero_for_empty_reference_and_zero_k() -> None:
    assert recall_at_k([], ["a", "b"], 5) == 0.0
    assert recall_at_k(["a", "b"], ["a", "b"], 0) == 0.0
    assert recall_at_k([], [], 0) == 0.0


def test_rank_agreement_counts_the_identical_prefix() -> None:
    assert rank_agreement(RECORD_IDS, RECORD_IDS) == len(RECORD_IDS)
    assert rank_agreement(["a", "b", "c"], ["a", "b", "x"]) == 2


def test_rank_agreement_zero_when_the_first_rank_differs() -> None:
    assert rank_agreement(RECORD_IDS, list(reversed(RECORD_IDS))) == 0


def test_rank_agreement_truncates_to_the_shorter_candidate() -> None:
    assert rank_agreement(["a", "b", "c"], ["a"]) == 1
    assert rank_agreement([], ["a"]) == 0


def test_scores_within_tolerance_separates_precision_from_ordering() -> None:
    assert scores_within_tolerance(1.0, 0.9999999999999998) is True
    assert scores_within_tolerance(1.0, 1.0 - 10 * SCORE_TOLERANCE) is False


# --------------------------------------------------------------------------- #
# 对账：flat 对三种度量与独立参照实现逐位一致
# --------------------------------------------------------------------------- #


def test_compare_backends_flat_matches_brute_force_on_three_metrics() -> None:
    records = flat_records(metric="ip")
    references: dict[str, Any] = {
        metric: expected_ranking(records, metric) for metric in ("cosine", "ip", "l2")
    }

    rows = compare_backends(
        records,
        AXIS_QUERY,
        metrics=("cosine", "ip", "l2"),
        backends=[("flat", FlatVectorStore)],
        reference=references,
    )

    assert [row.metric for row in rows] == ["cosine", "ip", "l2"]
    assert [row.backend for row in rows] == ["flat", "flat", "flat"]
    for row in rows:
        assert row.available is True
        assert row.count == len(RECORD_IDS)
        assert row.top_k == DEFAULT_PARITY_TOP_K
        assert row.agreed == DEFAULT_PARITY_TOP_K
        assert row.overlap == 1.0
        assert row.first_divergence == ""
        assert row.note.startswith("top-1=")


def test_compare_backends_without_reference_only_proves_self_consistency() -> None:
    """降级参照与被测实现共享全部代码：全绿说明不了任何事（这是刻意的演示）."""
    rows = compare_backends(
        flat_records(metric="ip"),
        AXIS_QUERY,
        metrics=("cosine", "ip", "l2"),
        backends=[FlatVectorStore],
    )

    assert len(rows) == 3
    for row in rows:
        assert row.available is True
        assert row.agreed == row.top_k
        assert row.overlap == 1.0
        assert row.first_divergence == ""


def test_compare_backends_accepts_instances_and_skips_other_metrics() -> None:
    store = loaded_flat()

    rows = compare_backends(
        flat_records(),
        AXIS_QUERY,
        metrics=("cosine", "ip"),
        backends=[store],
        reference=expected_ranking(flat_records(), "cosine"),
    )

    # 实例的度量在构造时就定死了（base.metric 不允许运行期切换），ip 那一行不出。
    assert len(rows) == 1
    assert rows[0].backend == "flat"
    assert rows[0].metric == "cosine"
    assert rows[0].agreed == DEFAULT_PARITY_TOP_K
    # 实例被写入了同一批记录：对账不要求调用方先自己 load 一遍。
    assert store.count() == len(RECORD_IDS)


def test_compare_backends_defaults_to_the_flat_reference() -> None:
    """缺省 ``backends=None`` 只对 flat 自己 —— 它是"参照实现的自述"，不是对账."""
    rows = compare_backends(flat_records(), AXIS_QUERY)

    assert len(rows) == 1
    assert rows[0].backend == "flat"
    assert rows[0].metric == "cosine"


def test_compare_backends_accepts_a_mapping_of_backends() -> None:
    records = flat_records()

    rows = compare_backends(
        records,
        AXIS_QUERY,
        backends={"flat": FlatVectorStore, "faiss": faiss_factory},
        reference=expected_ranking(records, "cosine"),
    )

    assert [row.backend for row in rows] == ["flat", "faiss"]
    assert [row.available for row in rows] == [True, True]


def test_compare_backends_reports_first_divergence_with_expected_and_actual() -> None:
    """刻意制造一次不一致：把参照换成"把 ``r_d`` 排到第一"的假排名."""
    records = flat_records()
    real = expected_ranking(records, "cosine")
    # 假的参照 = 真参照里把 r_d 顶到第一（其余顺序不动）
    fake_reference = [(OVERSIZED_ID, 2.0)] + [
        item for item in real if item[0] != OVERSIZED_ID
    ]

    rows = compare_backends(
        records,
        AXIS_QUERY,
        backends=[("flat", FlatVectorStore)],
        reference=fake_reference,
    )
    row = rows[0]

    assert row.available is True
    assert row.agreed == 0
    assert row.agreed < row.top_k
    assert row.first_divergence != ""
    assert "期望" in row.first_divergence
    assert "实际" in row.first_divergence
    assert OVERSIZED_ID in row.first_divergence  # 期望（假参照的 top-1）
    assert RECORD_IDS[0] in row.first_divergence  # 实际（flat 的 top-1）
    assert row.note == f"top-1={RECORD_IDS[0]}"
    # 集合完全相同、只有顺序被参照改掉：这正是"两个指标各管一半"的证据。
    assert row.overlap == 1.0


def test_first_divergence_mentions_the_tolerance_when_scores_are_equal() -> None:
    records = flat_records()
    real = expected_ranking(records, "cosine")
    swapped = [real[1], real[0], *real[2:]]  # 前两名对调，分数逐位相同

    rows = compare_backends(
        records, AXIS_QUERY, backends=[FlatVectorStore], reference=swapped
    )
    row = rows[0]

    assert row.agreed == 0
    assert "容差" in row.first_divergence
    assert "平局断法" in row.first_divergence


def test_compare_backends_flags_a_shorter_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    """候选比参照短（库里只有前两名）→ 一致长度够、但集合对不上."""
    records = flat_records()
    store = FlatVectorStore(metric="cosine", dimension=DIMENSION)
    store.upsert([records[0], records[3]])  # 恰好是参照的前两名
    monkeypatch.setattr(store, "upsert", lambda incoming: WriteReport())

    rows = compare_backends(
        records,
        AXIS_QUERY,
        backends=[("flat", store)],
        reference=expected_ranking(records, "cosine"),
    )
    row = rows[0]

    assert row.count == 2
    assert row.agreed == 2  # 前两名逐位相同
    assert "（缺失）" in row.first_divergence
    assert row.overlap == pytest.approx(0.4)

    report = verify_parity(rows)
    assert report["ok"] is False
    assert "集合对不上" in report["failures"][0]


def test_compare_backends_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="至少需要一条记录"):
        compare_backends([], QUERY_EXAMPLE)
    with pytest.raises(ValueError, match="top_k"):
        compare_backends(flat_records(), QUERY_EXAMPLE, top_k=0)
    with pytest.raises(ValueError, match="至少要给出一个度量"):
        compare_backends(flat_records(), QUERY_EXAMPLE, metrics=())


def test_compare_metrics_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="至少需要一条记录"):
        compare_metrics([], QUERY_EXAMPLE)
    with pytest.raises(ValueError, match="top_k"):
        compare_metrics(flat_records(), QUERY_EXAMPLE, top_k=0)
    with pytest.raises(ValueError, match="至少要给出一个度量"):
        compare_metrics(flat_records(), QUERY_EXAMPLE, metrics=())


def test_compare_backends_rejects_an_unknown_spec_and_a_short_tuple() -> None:
    with pytest.raises(VectorStoreError, match="无法识别的后端规格"):
        compare_backends(flat_records(), QUERY_EXAMPLE, backends=[12345])
    with pytest.raises(VectorStoreError, match="二元组"):
        compare_backends(flat_records(), QUERY_EXAMPLE, backends=[("flat",)])
    with pytest.raises(VectorStoreError, match="必须是 VectorBackend 实例或可调用工厂"):
        compare_backends(flat_records(), QUERY_EXAMPLE, backends=[("flat", 42)])


def test_compare_backends_refuses_a_reference_without_the_metric() -> None:
    references = {"cosine": expected_ranking(flat_records(), "cosine")}

    with pytest.raises(VectorStoreError, match="期望排名"):
        compare_backends(
            flat_records(),
            AXIS_QUERY,
            metrics=("cosine", "ip"),
            backends=[FlatVectorStore],
            reference=references,
        )


# --------------------------------------------------------------------------- #
# 可选依赖缺失：折成一行 available=False，而不是让整次对账失败
# --------------------------------------------------------------------------- #


def test_compare_backends_marks_missing_faiss_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "faiss", None)  # 伪造"本机没装 faiss"

    rows = compare_backends(
        flat_records(),
        AXIS_QUERY,
        backends=[("flat", FlatVectorStore), ("faiss", bare_faiss_factory)],
    )
    by_backend = {row.backend: row for row in rows}

    assert by_backend["flat"].available is True
    row = by_backend["faiss"]
    assert row.available is False
    assert row.count == 0
    assert row.agreed == 0
    assert row.overlap == 0.0
    assert row.first_divergence == "后端不可用"
    assert "pip install faiss-cpu" in row.note
    assert "flat" in row.note  # 三段式错误信息里的"还能用什么"


def test_compare_backends_marks_missing_chromadb_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "chromadb", None)

    rows = compare_backends(
        flat_records(), AXIS_QUERY, backends=[("chroma", bare_chroma_factory)]
    )
    row = rows[0]

    assert row.available is False
    assert row.first_divergence == "后端不可用"
    assert "pip install chromadb" in row.note


def test_compare_metrics_marks_a_missing_dependency_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "chromadb", None)

    rows = compare_metrics(
        flat_records(),
        AXIS_QUERY,
        metrics=("cosine",),
        backend=("chroma", bare_chroma_factory),
    )

    assert rows[0].backend == "chroma"
    assert rows[0].available is False
    assert rows[0].first_divergence == "后端不可用"
    assert "pip install chromadb" in rows[0].note
    assert verify_parity(rows)["ok"] is False


def test_a_factory_raising_backend_unavailable_is_recorded_not_raised() -> None:
    def unavailable_factory(*, metric: str, dimension: int, path: str = "") -> Any:
        raise BackendUnavailable(
            "后端 demo 不可用：缺少 demo（numpy 已满足）。安装：pip install demo。"
            "或者改用：flat（零可选依赖，结果逐位可复现）"
        )

    rows = compare_backends(
        flat_records(),
        AXIS_QUERY,
        backends=[("flat", FlatVectorStore), ("demo", unavailable_factory)],
    )

    assert [row.available for row in rows] == [True, False]
    assert "pip install demo" in rows[1].note


# --------------------------------------------------------------------------- #
# 三个后端互相对账：假 faiss / 假 chroma 与 flat 必须给出同一份排名
# --------------------------------------------------------------------------- #


def test_faiss_and_chroma_agree_with_flat_and_the_reference() -> None:
    records = flat_records()
    expected = expected_ranking(records, "cosine")

    rows = compare_backends(
        records,
        AXIS_QUERY,
        metrics=("cosine",),
        backends=[
            ("flat", FlatVectorStore),
            ("faiss", faiss_factory),
            ("chroma", chroma_factory),
        ],
        reference=expected,
    )

    assert [row.backend for row in rows] == ["flat", "faiss", "chroma"]
    for row in rows:
        assert row.available is True
        assert row.count == len(RECORD_IDS)
        assert row.agreed == DEFAULT_PARITY_TOP_K
        assert row.overlap == 1.0
        assert row.first_divergence == ""
        assert row.note == f"top-1={RECORD_IDS[0]}"

    report = verify_parity(rows)
    assert report["ok"] is True
    assert report["failures"] == []
    assert report["checked"] == 3
    assert report["available"] == 3


def test_verify_parity_builds_a_json_serializable_report() -> None:
    records = flat_records()
    rows = compare_backends(
        records,
        AXIS_QUERY,
        backends=[("flat", FlatVectorStore), ("faiss", faiss_factory)],
        reference=expected_ranking(records, "cosine"),
    )

    report = verify_parity(rows)

    assert json.loads(json.dumps(report)) == report
    assert [item["backend"] for item in report["rows"]] == ["flat", "faiss"]


def test_verify_parity_reports_the_reason_of_an_unavailable_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "faiss", None)

    rows = compare_backends(
        flat_records(), AXIS_QUERY, backends=[("faiss", bare_faiss_factory)]
    )
    report = verify_parity(rows)

    assert report["ok"] is False
    assert len(report["failures"]) == 1
    assert "faiss" in report["failures"][0]
    assert "不可用" in report["failures"][0]
    assert "pip install faiss-cpu" in report["failures"][0]
    assert report["available"] == 0


def test_verify_parity_reports_a_ranking_mismatch() -> None:
    records = flat_records()
    real = expected_ranking(records, "cosine")
    fake_reference = [(OVERSIZED_ID, 2.0)] + [
        item for item in real if item[0] != OVERSIZED_ID
    ]

    report = verify_parity(
        compare_backends(
            records, AXIS_QUERY, backends=[FlatVectorStore], reference=fake_reference
        )
    )

    assert report["ok"] is False
    assert "名次与参照不一致" in report["failures"][0]
    assert OVERSIZED_ID in report["failures"][0]


# --------------------------------------------------------------------------- #
# compare_metrics：三种度量对同一份数据给出的 top-1 不一样（最重要的对照）
# --------------------------------------------------------------------------- #


def test_compare_metrics_top1_differs_between_cosine_and_ip() -> None:
    """本模块最重要的一条断言：用 ``metric="ip"`` 构造才看得出模长带来的差异."""
    records = flat_records(metric="ip")

    rows = compare_metrics(records, AXIS_QUERY, metrics=("cosine", "ip", "l2"))
    by_metric = {row.metric: row for row in rows}

    assert list(by_metric) == ["cosine", "ip", "l2"]
    assert all(row.backend == "flat" for row in rows)

    cosine, ip, l2 = by_metric["cosine"], by_metric["ip"], by_metric["l2"]
    # 基线度量与自己对照：它是后续度量的锚，必然逐位一致。
    assert cosine.agreed == DEFAULT_PARITY_TOP_K
    assert cosine.first_divergence == ""
    assert cosine.note == f"top-1={RECORD_IDS[0]}"

    # ip 的 top-1 被 r_d 的 10 倍模长顶走 —— 与 cosine 的 top-1 不是同一条。
    assert ip.note == f"top-1={OVERSIZED_ID}"
    assert ip.agreed == 0
    assert "期望" in ip.first_divergence
    assert "实际" in ip.first_divergence
    assert RECORD_IDS[0] in ip.first_divergence  # 期望 = cosine 的 top-1
    assert OVERSIZED_ID in ip.first_divergence  # 实际 = ip 的 top-1

    # l2 的第一名还是 r_a（与查询同向），但从第二名起就与 cosine 分家。
    assert l2.note == f"top-1={RECORD_IDS[0]}"
    assert l2.agreed == 1
    assert l2.overlap == pytest.approx(0.8)
    # 期望 = cosine 的第 2 名（模长 10 倍、被归一化后与 r_a 并列的那条）
    assert OVERSIZED_ID in l2.first_divergence


def test_compare_metrics_on_normalized_records_hides_the_magnitude_gap() -> None:
    """``metric="cosine"`` 构造的记录已被归一化，此时 ip 与 cosine 名次完全相同."""
    rows = compare_metrics(
        flat_records(metric="cosine"), AXIS_QUERY, metrics=("cosine", "ip")
    )

    assert rows[1].metric == "ip"
    assert rows[1].agreed == DEFAULT_PARITY_TOP_K
    assert rows[1].overlap == 1.0
    assert rows[1].first_divergence == ""
    assert rows[1].note == rows[0].note


def test_compare_metrics_matches_brute_force_for_all_three_metrics() -> None:
    records = flat_records(metric="ip")
    references = {
        metric: expected_ranking(records, metric)
        for metric in ("cosine", "ip", "l2")
    }

    rows = compare_metrics(
        records, AXIS_QUERY, metrics=("cosine", "ip", "l2"), reference=references
    )

    assert [row.metric for row in rows] == ["cosine", "ip", "l2"]
    for row in rows:
        assert row.agreed == DEFAULT_PARITY_TOP_K
        assert row.overlap == 1.0
        assert row.first_divergence == ""


def test_compare_metrics_accepts_an_injected_backend_instance() -> None:
    store = FaissVectorStore(
        metric="cosine", dimension=DIMENSION, faiss_module=make_fake_faiss_module()
    )

    rows = compare_metrics(
        flat_records(), AXIS_QUERY, metrics=("cosine", "ip"), backend=("faiss", store)
    )

    assert len(rows) == 1  # 实例的度量定死在 cosine，ip 那一行被跳过
    assert rows[0].backend == "faiss"
    assert rows[0].metric == "cosine"
    assert rows[0].agreed == DEFAULT_PARITY_TOP_K
    assert rows[0].note == f"top-1={RECORD_IDS[0]}"


# --------------------------------------------------------------------------- #
# index_health：四项体检
# --------------------------------------------------------------------------- #


def test_index_health_passes_a_healthy_store() -> None:
    health = index_health(loaded_flat())

    assert health["healthy"] is True
    assert [item["check"] for item in health["checks"]] == [
        "count",
        "unique_ids",
        "dimension",
        "retrievable",
    ]
    assert all(item["ok"] is True for item in health["checks"])
    assert health["backend"] == "flat"
    assert health["count"] == len(RECORD_IDS)
    assert health["dimension"] == DIMENSION
    assert health["duplicate_ids"] == []
    assert health["missing_ids"] == []
    assert health["dimension_mismatches"] == []
    assert health["summary"] == "四项体检全部通过"


def test_index_health_handles_an_empty_store() -> None:
    """空库没有脏状态可检：四项都必须给出可读结论（而不是"因为没数据所以跳过了"）."""
    health = index_health(FlatVectorStore(metric="cosine"))

    assert health["healthy"] is True
    assert health["count"] == 0
    assert health["dimension"] == 0
    dimension = next(item for item in health["checks"] if item["check"] == "dimension")
    assert dimension["detail"] == "维度未定（空库）"


def test_index_health_detects_a_record_the_table_lost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """向量索引里有、记录表里没有 —— 那个最典型的脏状态."""
    store = loaded_flat()
    victim = store.ids()[0]
    original = store.get

    def broken_get(record_id: str) -> Any:
        return None if record_id == victim else original(record_id)

    monkeypatch.setattr(store, "get", broken_get)

    health = index_health(store)

    assert health["healthy"] is False
    assert health["count"] == len(RECORD_IDS)  # 两边都还说着 6 条
    assert health["missing_ids"] == [victim]
    assert health["summary"] == "未通过：retrievable"
    retrievable = next(item for item in health["checks"] if item["check"] == "retrievable")
    assert retrievable["ok"] is False
    assert "记录表里取不回" in retrievable["detail"]

    # 症状本身：读路径跳过那条脏数据，于是"命中少了一条"（不报错）。
    result = store.query(AXIS_QUERY, DEFAULT_PARITY_TOP_K)
    assert result.count == DEFAULT_PARITY_TOP_K - 1


def test_index_health_detects_a_dimension_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = loaded_flat()
    victim = store.ids()[0]
    original = store.get
    foreign = VectorRecord(
        record_id=victim, vector=(0.5,) * 16, text="异维记录（换过编码器？）"
    )

    def broken_get(record_id: str) -> Any:
        return foreign if record_id == victim else original(record_id)

    monkeypatch.setattr(store, "get", broken_get)

    health = index_health(store)

    assert health["healthy"] is False
    assert health["dimension_mismatches"] == [f"{victim}=16d"]
    assert health["missing_ids"] == []  # 记录取回来了，只是维度不对
    assert health["summary"] == "未通过：dimension"


# --------------------------------------------------------------------------- #
# ParityRow 的投影
# --------------------------------------------------------------------------- #


def test_parity_row_projects_to_json_and_a_summary_line() -> None:
    row = ParityRow(
        backend="flat",
        metric="cosine",
        available=True,
        count=6,
        top_k=5,
        agreed=5,
        overlap=1.0,
        first_divergence="",
        note="top-1=4a6680cde33da45e",
    )

    payload = row.to_dict()
    assert payload == {
        "backend": "flat",
        "metric": "cosine",
        "available": True,
        "count": 6,
        "top_k": 5,
        "agreed": 5,
        "overlap": 1.0,
        "first_divergence": "",
        "note": "top-1=4a6680cde33da45e",
    }
    assert json.loads(json.dumps(payload)) == payload
    assert "逐位一致" in row.summary_line()
    assert "5/5" in row.summary_line()


def test_parity_row_summary_line_covers_diverged_and_unavailable() -> None:
    diverged = ParityRow(
        backend="flat",
        metric="l2",
        available=True,
        count=6,
        top_k=5,
        agreed=1,
        overlap=0.8,
        first_divergence="第 2 名：期望 '7b3e0d5c9a14f286'，实际 '1c9f4b7a2e6d8035'",
        note="top-1=4a6680cde33da45e",
    )
    assert "首个分歧在第 2 名" in diverged.summary_line()
    assert "80.0%" in diverged.summary_line()

    missing = ParityRow(
        backend="faiss",
        metric="cosine",
        available=False,
        count=0,
        top_k=5,
        agreed=0,
        overlap=0.0,
        first_divergence="后端不可用",
        note="缺少 faiss：pip install faiss-cpu",
    )
    assert "不可用" in missing.summary_line()
    assert "pip install faiss-cpu" in missing.summary_line()
