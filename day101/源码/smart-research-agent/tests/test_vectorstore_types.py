"""day064 ``vectorstore.types`` 的单元测试：四种形状的校验、算术与投影.

全部离线、确定性。这一层的测试重点是**契约挂在哪里**：

```text
VectorRecord   校验挂在类型上（不是挂在 make_record 上）
WriteReport    unchanged 是一个独立的态，不是 updated 的一部分
SearchResult   candidates / filter_applied 让"为什么只返回了 2 条"可读
```

因此断言都指向具体数值与具体错误消息片段，而不是 ``is not None``——
一条"软断言"在实现被改坏之后仍然会通过，它证明不了任何事。
"""

from __future__ import annotations

from typing import Any

import pytest

from smart_research_agent.vectorstore.errors import RecordError
from smart_research_agent.vectorstore.types import (
    MAX_RECORD_ID_LENGTH,
    SearchHit,
    SearchResult,
    StoreInfo,
    VectorRecord,
    WriteReport,
    make_record,
    score_hit,
    sort_hits,
)
from tests.vectorstore_samples import RECORD_IDS, SAMPLE_QUERIES, sample_records


def record(**overrides: Any) -> VectorRecord:
    """造一条合法的记录（覆盖时才用给定值）."""
    payload: dict[str, Any] = {
        "record_id": "rec-0001",
        "vector": (1.0, 0.0),
        "text": "一段正文",
        "metadata": {"strategy": "fixed"},
    }
    payload.update(overrides)
    return VectorRecord(**payload)


# --------------------------------------------------------------------------- #
# VectorRecord：契约挂在类型上
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        # id：空串 / 全空白 / 超过长度上限
        ({"record_id": ""}, "record_id 必须是非空字符串"),
        ({"record_id": "   "}, "record_id 必须是非空字符串"),
        (
            {"record_id": "x" * (MAX_RECORD_ID_LENGTH + 1)},
            f"超过上限 {MAX_RECORD_ID_LENGTH}",
        ),
        # 向量：nan / inf / 零向量
        ({"vector": (float("nan"), 1.0)}, "不是有限数"),
        ({"vector": (float("inf"), 1.0)}, "不是有限数"),
        ({"vector": (0.0, 0.0)}, "零向量"),
        # 元数据：嵌套 dict / 空 list / 混类型 list
        ({"metadata": {"nested": {"a": 1}}}, "metadata['nested']"),
        ({"metadata": {"tags": []}}, "metadata['tags']"),
        ({"metadata": {"tags": ["guide", 1]}}, "metadata['tags']"),
        ({"metadata": {"nothing": None}}, "metadata['nothing']"),
    ],
)
def test_vector_record_rejects_invalid_fields(kwargs: dict[str, Any], fragment: str) -> None:
    """直接构造也要被拦（校验不能只挂在 ``make_record`` 上）.

    每一条都对应一种"不报错但此后全错"的输入：``nan`` 参与的比较恒为 False，
    零向量让余弦变成 0/0，嵌套 dict 在 Chroma 上必然被拒。
    """
    with pytest.raises(RecordError) as excinfo:
        record(**kwargs)

    assert fragment in str(excinfo.value)


@pytest.mark.parametrize(
    "metadata",
    [
        {"flag": True},
        {"count": 3},
        {"score": 0.5},
        {"name": "x"},
        {"tags": ["a", "b"]},
        {"tags": [1, 2]},
        {"tags": (1.5, 2.5)},
    ],
)
def test_vector_record_accepts_contract_metadata(metadata: dict[str, Any]) -> None:
    """``bool`` 必须被当成布尔而不是整数（``isinstance(True, int)`` 是 True）."""
    created = record(metadata=metadata)

    assert created.metadata == metadata


def test_vector_is_frozen_into_tuple() -> None:
    """传进来的 list 会被冻结成 tuple，改原 list 不影响记录.

    frozen dataclass 只是禁止**重新赋值**，字段本身若是 list
    仍然可以被 ``record.vector[0] = 0.0`` 改掉——那会绕过所有校验。
    """
    values = [1.0, 0.0]
    created = VectorRecord(record_id="rec-0001", vector=values)

    values[0] = 99.0

    assert isinstance(created.vector, tuple)
    assert created.vector == (1.0, 0.0)
    assert created.dimension == 2


def test_vector_is_frozen_even_for_tuple_input() -> None:
    """tuple 输入也会被逐元素 ``float()`` 一遍（避免 int 混进向量）."""
    created = VectorRecord(record_id="rec-0001", vector=(1, 0))

    assert created.vector == (1.0, 0.0)
    assert all(isinstance(value, float) for value in created.vector)


def test_text_defaults_to_empty_and_char_count_follows() -> None:
    """``text`` 缺省是空串（只做向量过滤的场景不需要原文）."""
    created = VectorRecord(record_id="rec-0001", vector=(1.0, 0.0))

    assert created.text == ""
    assert created.char_count == 0
    assert record(text="\n换行\n").char_count == 4


# --------------------------------------------------------------------------- #
# WriteReport：unchanged 是一个独立的态
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["added", "updated", "unchanged", "skipped", "removed"])
def test_write_report_rejects_negative(name: str) -> None:
    """五个计数都非负；负数意味着某处把"减出来的差值"当成了计数."""
    with pytest.raises(RecordError) as excinfo:
        WriteReport(**{name: -1})

    assert f"WriteReport.{name} 必须非负" in str(excinfo.value)


def test_write_report_arithmetic_includes_removed() -> None:
    """``written`` / ``total`` / ``changed`` 三个派生量各自的口径.

    ``changed`` **含 removed 而不含 unchanged**：它是"这次调用让库变了多少"，
    而删除确实让库变了。
    """
    report = WriteReport(added=2, updated=1, unchanged=3, skipped=1, removed=2)

    assert report.written == 3  # added + updated
    assert report.total == 7  # added + updated + unchanged + skipped
    assert report.changed == 5  # added + updated + removed


def test_write_report_to_dict_has_derived_keys() -> None:
    """投影里也带派生量，便于测试直接逐键断言."""
    payload = WriteReport(added=2, updated=1, unchanged=3, skipped=1, removed=2).to_dict()

    assert set(payload) == {
        "added",
        "updated",
        "unchanged",
        "skipped",
        "removed",
        "written",
        "total",
        "changed",
    }
    assert payload["changed"] == 5


def test_write_report_merge_keeps_unchanged_separate() -> None:
    """分批写库时 ``unchanged`` 必须能累加——day065 靠它算"省掉多少次编码"."""
    merged = WriteReport(added=1, unchanged=2).merge(WriteReport(updated=3, unchanged=4))

    assert merged.added == 1
    assert merged.updated == 3
    assert merged.unchanged == 6
    assert merged.written == 4


def test_write_report_summary_line_lists_every_state() -> None:
    """五个态都要在摘要里出现（少一个就说明它被并进了别的态）."""
    line = WriteReport(added=1, updated=2, unchanged=3, skipped=4, removed=5).summary_line()

    for fragment in ("新增 1", "覆盖 2", "未变 3", "跳过 4", "删除 5"):
        assert fragment in line


# --------------------------------------------------------------------------- #
# SearchHit：两个口径同时保留
# --------------------------------------------------------------------------- #


def test_search_hit_keeps_both_score_and_distance() -> None:
    """``score`` 越大越近、``distance`` 越小越近，两者都要带."""
    created = make_record("rec-0001", (1.0, 0.0), "正文", {"k": 1})

    hit = score_hit(created, (1.0, 0.0), "cosine", rank=0)

    assert hit.score == pytest.approx(1.0)
    assert hit.distance == pytest.approx(0.0)


def test_search_hit_to_dict_key_set() -> None:
    """投影的键集合是端点契约的一部分（多一个少一个都算接口变了）."""
    hit = SearchHit(
        record=make_record("rec-0001", (1.0, 0.0), "正文", {"k": 1}),
        score=0.75,
        distance=0.25,
        rank=2,
    )

    payload = hit.to_dict()

    assert set(payload) == {
        "rank",
        "record_id",
        "score",
        "distance",
        "dimension",
        "metadata",
        "text",
    }
    assert payload["rank"] == 2
    assert payload["score"] == 0.75
    assert payload["record_id"] == "rec-0001"


def test_search_hit_to_dict_can_hide_text() -> None:
    """``include_text=False`` 只给排序与身份（批量响应不该带全文）."""
    hit = SearchHit(
        record=make_record("rec-0001", (1.0, 0.0), "正文", {"k": 1}),
        score=0.75,
        distance=0.25,
        rank=0,
    )

    lean = hit.to_dict(include_text=False)

    assert "text" not in lean
    assert lean["dimension"] == 2


def test_search_hit_summary_line_is_not_empty_and_carries_rank() -> None:
    hit = SearchHit(
        record=make_record("rec-0001", (1.0, 0.0), "正文", {"k": 1}),
        score=0.75,
        distance=0.25,
        rank=3,
    )

    line = hit.summary_line()

    assert line.startswith("#3 rec-0001")
    assert "score=+0.750000" in line


# --------------------------------------------------------------------------- #
# SearchResult：让"为什么只返回了这么点"可读
# --------------------------------------------------------------------------- #


def build_result(*hits: SearchHit, **overrides: Any) -> SearchResult:
    """由若干命中造一个结果（默认两个候选、无过滤）."""
    payload: dict[str, Any] = {
        "hits": hits,
        "metric": "cosine",
        "top_k": 5,
        "candidates": 2,
        "filter_applied": False,
    }
    payload.update(overrides)
    return SearchResult(**payload)


def test_search_result_accessors() -> None:
    first = SearchHit(make_record("rec-a", (1.0, 0.0)), 0.9, 0.1, 0)
    second = SearchHit(make_record("rec-b", (0.0, 1.0)), 0.4, 0.6, 1)
    result = build_result(first, second)

    assert result.count == 2
    assert result.ids() == ["rec-a", "rec-b"]
    assert result.scores == [0.9, 0.4]
    assert result.top() is first


def test_search_result_empty_has_no_top() -> None:
    """空结果返回 ``None`` 而不是抛异常（"没找到"是正常结论）."""
    empty = SearchResult(hits=(), metric="cosine", top_k=5)

    assert empty.count == 0
    assert empty.ids() == []
    assert empty.scores == []
    assert empty.top() is None


def test_search_result_to_dict_key_set() -> None:
    """``candidates`` 与 ``filter_applied`` 必须在投影里——它们是诊断字段."""
    result = build_result(
        SearchHit(make_record("rec-a", (1.0, 0.0)), 0.9, 0.1, 0),
        candidates=5,
        filter_applied=True,
    )

    payload = result.to_dict()

    assert set(payload) == {
        "metric",
        "top_k",
        "candidates",
        "filter_applied",
        "count",
        "hits",
    }
    assert payload["candidates"] == 5
    assert payload["filter_applied"] is True
    assert payload["count"] == 1
    assert payload["hits"][0]["record_id"] == "rec-a"


def test_search_result_summary_line_is_not_empty() -> None:
    result = build_result(
        SearchHit(make_record("rec-a", (1.0, 0.0)), 0.9, 0.1, 0),
        candidates=3,
    )

    line = result.summary_line()

    assert line.startswith("cosine | top_k=5")
    assert "候选 3" in line
    assert "命中 1" in line


# --------------------------------------------------------------------------- #
# sort_hits：并列由 id 决定，名次是 0..n-1
# --------------------------------------------------------------------------- #


def test_sort_hits_breaks_ties_by_record_id() -> None:
    """**用样本里那对余弦并列的记录做断言**（分数精确相等，不是近似相等）.

    正序传入时两种实现都"看起来对"；这里刻意**逆序传入**，
    让"排序键里有没有 id"这件事真的能被区分出来。
    """
    records = sample_records(metric="cosine")
    query = SAMPLE_QUERIES["q_axis"]
    hits = [score_hit(record, query, "cosine", rank=0) for record in reversed(records)]

    ordered = sort_hits(hits)

    assert ordered[0].score == pytest.approx(1.0)
    assert ordered[1].score == pytest.approx(1.0)
    assert ordered[0].score == ordered[1].score
    assert [hit.record.record_id for hit in ordered[:2]] == [RECORD_IDS[0], RECORD_IDS[3]]


def test_sort_hits_reranks_from_zero() -> None:
    """重排之后 ``rank`` 是 0..n-1（原 rank 全部被丢掉，不是保留）."""
    records = sample_records(metric="cosine")
    query = SAMPLE_QUERIES["q_axis"]
    hits = [score_hit(record, query, "cosine", rank=99) for record in records]

    ordered = sort_hits(hits)

    assert [hit.rank for hit in ordered] == list(range(len(records)))


def test_sort_hits_orders_scores_descending() -> None:
    records = sample_records(metric="cosine")
    query = SAMPLE_QUERIES["q_axis"]
    scores = [hit.score for hit in sort_hits([score_hit(r, query, "cosine", 0) for r in records])]

    assert scores == sorted(scores, reverse=True)


# --------------------------------------------------------------------------- #
# StoreInfo
# --------------------------------------------------------------------------- #


def test_store_info_to_dict_and_summary_line() -> None:
    info = StoreInfo(
        backend="flat",
        metric="cosine",
        dimension=8,
        count=6,
        persistent=True,
        location="snap.json",
        extra={"records": "6"},
    )

    payload = info.to_dict()

    assert set(payload) == {
        "backend",
        "metric",
        "dimension",
        "count",
        "persistent",
        "location",
        "extra",
    }
    assert info.summary_line() == "flat | cosine | 8d | 6 条 | 持久化 是 | snap.json"


def test_store_info_marks_memory_store() -> None:
    """没有 location 时摘要写"（内存）"——"以为落盘了"必须一眼可见."""
    info = StoreInfo(backend="flat", metric="ip", dimension=4, count=0)

    assert info.summary_line() == "flat | ip | 4d | 0 条 | 持久化 否 | （内存）"
