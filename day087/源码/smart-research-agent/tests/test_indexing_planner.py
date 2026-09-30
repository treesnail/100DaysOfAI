"""``indexing.planner`` 的测试：两个指纹、四条判据、一句原因（M6-D4）.

``planner`` 的全部价值在于把两个**不同的问题**分开判：

```text
fingerprint   内容变了没有        （原文的内容身份）
vector_key    该不该复用向量      （编码器身份 + 该编码的那段文本）
```

因此本文件的断言围绕两件事展开：

```text
四条判据     新增 / 更新（内容变）/ 更新（编码文本变）/ 不变 / 删除 各有一条
两个反例     只比 fingerprint 会漏"换编码器"；只比 vector_key 会漏"同长度覆盖写"
排序与原因   输出升序；身份变了要在 reason 里被单独写出来（"编码器"）
```

全部离线、零网络：清单由契约层的 ``IndexEntry`` / ``IndexManifest``
直接构造（不依赖 B 号同刻在写的 ``manifest.py``），因此本文件可以独立运行。
"""

from __future__ import annotations

from typing import Any

from smart_research_agent.documents.types import content_id
from smart_research_agent.indexing.planner import (
    entry_from_record,
    plan_index,
    plan_reason,
)
from smart_research_agent.indexing.types import (
    EmbeddingIdentity,
    IndexManifest,
    entries_digest,
    index_version_id,
    vector_key,
)

#: 三个 16 位十六进制 id（形态取自 day062 的 ``chunk_id``）.
ID_A = "0a1b2c3d4e5f6071"
ID_B = "1b2c3d4e5f607182"
ID_C = "2c3d4e5f60718293"

#: 基准身份（provider / model / dimension 三者缺一不可）.
IDENTITY = EmbeddingIdentity(provider="FakeEmbedding", model="m1", dimension=8)

#: 换了模型的同一族身份：维度相同，只有模型名变了.
IDENTITY_OTHER_MODEL = EmbeddingIdentity(provider="FakeEmbedding", model="m2", dimension=8)

#: 换了输出维度的同一族身份.
IDENTITY_OTHER_DIMENSION = EmbeddingIdentity(
    provider="FakeEmbedding", model="m1", dimension=16
)


def make_record(
    record_id: str,
    text: str,
    *,
    retrieval_text: str | None = None,
    fingerprint: str | None = None,
    token_count: Any = None,
) -> dict[str, Any]:
    """造一条 day062 ``knowledge_records()`` 形状的记录（四个键）.

    每个可选参数都对应一个**独立**的取数口径，因此测试能把
    "内容"与"该编码的文本"分离开来单独检验（见两个反例）。
    """
    metadata: dict[str, Any] = {}
    if retrieval_text is not None:
        metadata["retrieval_text"] = retrieval_text
    if fingerprint is not None:
        metadata["fingerprint"] = fingerprint
    if token_count is not None:
        metadata["token_count"] = token_count
    return {
        "doc_id": record_id,
        "source": "docs/demo.md",
        "text": text,
        "metadata": metadata,
    }


def manifest_for(
    records: list[dict[str, Any]],
    identity: EmbeddingIdentity = IDENTITY,
    *,
    metric: str = "cosine",
    backend: str = "flat",
) -> IndexManifest:
    """把这批记录编成一份**可自洽**的清单（version_id 按契约重算）.

    这里直接调用契约层的 ``index_version_id`` / ``entries_digest``，
    因此不依赖 B 号的 ``manifest.py``——本文件的测试对象只有 planner。
    """
    entries = tuple(entry_from_record(record, identity) for record in records)
    return IndexManifest(
        version_id=index_version_id(
            identity_key=identity.key,
            metric=metric,
            backend=backend,
            digest=entries_digest(entries),
        ),
        identity=identity,
        metric=metric,
        backend=backend,
        entries=entries,
    )


def base_records() -> list[dict[str, Any]]:
    """三条记录：它们是下面大多数用例的共同起点（改动只发生在其中一条上）."""
    return [
        make_record(ID_A, "固定长度的块每 320 个字符切一刀。"),
        make_record(ID_B, "默认返回前 5 条，低于阈值的直接丢掉。"),
        make_record(ID_C, "一次全量重建索引大约 12 万条记录。"),
    ]


# --------------------------------------------------------------------------- #
# 四条判据
# --------------------------------------------------------------------------- #


def test_previous_none_marks_everything_added() -> None:
    """没有上一版 → 全部 ``added``，复用率为 0。"""
    plan = plan_index(None, base_records(), IDENTITY)

    assert plan.added == (ID_A, ID_B, ID_C)
    assert plan.updated == ()
    assert plan.removed == ()
    assert plan.unchanged == ()
    assert plan.reuse_ratio == 0.0
    assert plan.to_encode == (ID_A, ID_B, ID_C)
    assert "首次构建" in plan.reason


def test_identical_batches_are_all_unchanged() -> None:
    """完全相同的两批 → 全部 ``unchanged``，``to_encode`` 为空，复用率 1.0。"""
    records = base_records()
    previous = manifest_for(records)

    plan = plan_index(previous, base_records(), IDENTITY)

    assert plan.unchanged == (ID_A, ID_B, ID_C)
    assert plan.added == ()
    assert plan.updated == ()
    assert plan.removed == ()
    assert plan.to_encode == ()
    assert plan.reuse_ratio == 1.0
    assert "没有变化" in plan.reason


def test_changed_text_with_different_length_is_updated() -> None:
    """改一条文本（长度不同）→ 该条 ``updated``，其余复用."""
    previous = manifest_for(base_records())
    changed = base_records()
    changed[0]["text"] = "固定长度的块每 320 个字符切一刀，重叠 48 个字符。"

    plan = plan_index(previous, changed, IDENTITY)

    assert plan.updated == (ID_A,)
    assert plan.unchanged == (ID_B, ID_C)
    assert plan.added == () and plan.removed == ()
    assert plan.to_encode == (ID_A,)
    assert "更新 1 条" in plan.reason


def test_same_length_different_content_is_updated() -> None:
    """**同长度但不同内容** → 仍然 ``updated``（这条挡"只比长度"的实现）."""
    previous = manifest_for(base_records())
    changed = base_records()
    # "AAAA" 与 "BBBB" 长度相同、内容不同：按长度比较的实现会把它判成 unchanged。
    changed[0]["text"] = "AAAA"
    before_same_length = base_records()
    before_same_length[0]["text"] = "BBBB"
    previous = manifest_for(before_same_length)

    plan = plan_index(previous, changed, IDENTITY)

    assert len("AAAA") == len("BBBB")
    assert plan.updated == (ID_A,)
    assert plan.unchanged == (ID_B, ID_C)


def test_removed_record_appears_in_removed() -> None:
    """删掉一条 → 出现在 ``removed``，其余复用；两条被删时按 id 升序."""
    previous = manifest_for(base_records())

    plan_one = plan_index(previous, base_records()[:2], IDENTITY)
    assert plan_one.removed == (ID_C,)
    assert plan_one.unchanged == (ID_A, ID_B)
    assert plan_one.updated == () and plan_one.added == ()
    assert plan_one.total == 3

    plan_two = plan_index(previous, base_records()[:1], IDENTITY)
    assert plan_two.removed == (ID_B, ID_C)


def test_identity_change_marks_everything_updated() -> None:
    """换编码器（模型或维度）→ 全部 ``updated``，且 reason 里含"编码器"."""
    records = base_records()
    previous = manifest_for(records, IDENTITY)

    model_changed = plan_index(previous, records, IDENTITY_OTHER_MODEL)
    assert model_changed.updated == (ID_A, ID_B, ID_C)
    assert model_changed.unchanged == ()
    assert model_changed.added == () and model_changed.removed == ()
    assert "编码器" in model_changed.reason
    assert "这不是数据变更" in model_changed.reason

    dimension_changed = plan_index(previous, records, IDENTITY_OTHER_DIMENSION)
    assert dimension_changed.updated == (ID_A, ID_B, ID_C)
    assert "编码器" in dimension_changed.reason


# --------------------------------------------------------------------------- #
# 两个反例：两个指纹必须分别判
# --------------------------------------------------------------------------- #


def test_fingerprint_change_without_vector_key_change_is_updated() -> None:
    """只比 ``vector_key`` 会漏掉这种：原文变了，但该编码的检索视图没变.

    构造方式：固定 ``retrieval_text``（向量键的输入），只改 ``text``
    （``fingerprint`` 的输入）。旧实现若只比向量键，会把它判成 unchanged——
    于是库里留着旧内容的向量，命中之后给出的却是新原文。
    """
    before = [make_record(ID_A, "内容甲乙", retrieval_text="检索视图（固定）")]
    after = [make_record(ID_A, "内容甲乙丙", retrieval_text="检索视图（固定）")]
    previous = manifest_for(before)

    before_entry = entry_from_record(before[0], IDENTITY)
    after_entry = entry_from_record(after[0], IDENTITY)
    assert before_entry.fingerprint != after_entry.fingerprint
    assert before_entry.vector_key == after_entry.vector_key

    plan = plan_index(previous, after, IDENTITY)

    assert plan.updated == (ID_A,)
    assert plan.unchanged == ()


def test_vector_key_change_without_fingerprint_change_is_updated() -> None:
    """只比 ``fingerprint`` 会漏掉这种：原文没变，但编码文本变了.

    构造方式：固定 ``text``（``fingerprint`` 的输入），只改 ``retrieval_text``
    （向量键的输入）。旧实现若只比内容指纹，会把它判成 unchanged——
    于是旧向量被复用，而它其实是用**另一段文本**编码出来的。
    """
    before = [make_record(ID_A, "内容甲乙", retrieval_text="检索视图 一")]
    after = [make_record(ID_A, "内容甲乙", retrieval_text="检索视图 二")]
    previous = manifest_for(before)

    before_entry = entry_from_record(before[0], IDENTITY)
    after_entry = entry_from_record(after[0], IDENTITY)
    assert before_entry.fingerprint == after_entry.fingerprint
    assert before_entry.vector_key != after_entry.vector_key

    plan = plan_index(previous, after, IDENTITY)

    assert plan.updated == (ID_A,)
    assert plan.unchanged == ()


# --------------------------------------------------------------------------- #
# 输出形状：升序、无重复
# --------------------------------------------------------------------------- #


def test_output_ids_are_sorted_for_unordered_input() -> None:
    """乱序输入 → 输出仍然升序（报告要能被逐行 diff）."""
    shuffled = [base_records()[2], base_records()[0], base_records()[1]]

    plan = plan_index(None, shuffled, IDENTITY)

    assert plan.added == tuple(sorted((ID_A, ID_B, ID_C)))
    assert plan.added == (ID_A, ID_B, ID_C)


def test_mixed_actions_are_each_sorted_and_do_not_overlap() -> None:
    """一次同时新增/更新/删除：每组各自升序，且同一个 id 只出现在一个组里."""
    previous = manifest_for(base_records())
    changed = [
        make_record(ID_B, "这一条的内容被改写了，长度也变了。"),  # updated
        make_record("3d4e5f60718293a4", "全新的一条。"),  # added
        make_record("4e5f60718293a4b5", "另一条全新的。"),  # added
    ]

    plan = plan_index(previous, changed, IDENTITY)

    assert plan.added == ("3d4e5f60718293a4", "4e5f60718293a4b5")
    assert plan.updated == (ID_B,)
    assert plan.removed == (ID_A, ID_C)
    groups = [set(plan.added), set(plan.updated), set(plan.removed), set(plan.unchanged)]
    assert sum(len(group) for group in groups) == plan.total
    assert plan.total == 5


# --------------------------------------------------------------------------- #
# entry_from_record：取哪个字段
# --------------------------------------------------------------------------- #


def test_entry_from_record_prefers_metadata_fingerprint() -> None:
    """``metadata["fingerprint"]`` 存在时直接采用（day062 给的内容身份）."""
    record = make_record(ID_A, "内容甲", fingerprint="abcdef0123456789", token_count=7)

    entry = entry_from_record(record, IDENTITY)

    assert entry.record_id == ID_A
    assert entry.fingerprint == "abcdef0123456789"
    assert entry.token_count == 7


def test_entry_from_record_falls_back_to_content_id_of_raw_text() -> None:
    """没有 ``metadata["fingerprint"]`` → 按**原文**现算，且向量键用检索视图."""
    record = make_record(ID_A, "原文内容", retrieval_text="面包屑 > 原文内容")

    entry = entry_from_record(record, IDENTITY)

    assert entry.fingerprint == content_id("原文内容")
    assert entry.vector_key == vector_key(
        IDENTITY.provider, IDENTITY.model, IDENTITY.dimension, "面包屑 > 原文内容"
    )
    assert entry.vector_key != vector_key(
        IDENTITY.provider, IDENTITY.model, IDENTITY.dimension, "原文内容"
    )


def test_entry_from_record_reads_raw_text_from_metadata_layer() -> None:
    """原文可能落在 ``metadata`` 那一层（同一个字段两种位置都要能取到）."""
    record: dict[str, Any] = {
        "doc_id": ID_A,
        "source": "docs/demo.md",
        "metadata": {"text": "元数据里的原文"},
    }

    entry = entry_from_record(record, IDENTITY)

    assert entry.fingerprint == content_id("元数据里的原文")


def test_entry_token_count_edge_cases() -> None:
    """``token_count`` 取整数；取不到、不是整数、为负 → 0（它只用于统计）."""
    assert entry_from_record(make_record(ID_A, "文本"), IDENTITY).token_count == 0
    assert (
        entry_from_record(make_record(ID_A, "文本", token_count="12"), IDENTITY).token_count
        == 12
    )
    assert (
        entry_from_record(make_record(ID_A, "文本", token_count="abc"), IDENTITY).token_count
        == 0
    )
    assert (
        entry_from_record(make_record(ID_A, "文本", token_count="320.7"), IDENTITY).token_count
        == 0
    )
    assert (
        entry_from_record(make_record(ID_A, "文本", token_count=-5), IDENTITY).token_count
        == 0
    )
    top_level = {"doc_id": ID_A, "text": "文本", "token_count": 33}
    assert entry_from_record(top_level, IDENTITY).token_count == 33


# --------------------------------------------------------------------------- #
# plan_reason：三种取值
# --------------------------------------------------------------------------- #


def test_plan_reason_is_independent_of_plan_reason_field() -> None:
    """``plan_reason`` 可以直接被外部调用（它只看身份与计数，不看 plan.reason）."""
    previous = manifest_for(base_records())
    plan = plan_index(previous, base_records(), IDENTITY)

    assert plan_reason(previous, IDENTITY, plan) == plan.reason
    assert "复用率 100.00%" in plan.reason


def test_plan_reason_identity_change_mentions_both_identities() -> None:
    """身份变更的那句话要能分别说出旧身份与新身份（否则看不出换了什么）."""
    previous = manifest_for(base_records(), IDENTITY)
    plan = plan_index(previous, base_records(), IDENTITY_OTHER_MODEL)

    reason = plan_reason(previous, IDENTITY_OTHER_MODEL, plan)

    assert "FakeEmbedding/m1 (8d)" in reason
    assert "FakeEmbedding/m2 (8d)" in reason
    assert "全部 3 条需重算" in reason
