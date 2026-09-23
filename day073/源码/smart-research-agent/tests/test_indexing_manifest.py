"""``indexing.manifest`` 的测试：版本号可重算、清单能被读回、与库对账能说人话.

本文件盯住四件事，每一件都对应模块 docstring 里的一条取舍：

```text
版本号的构成   同一份内容不同顺序 → 同号；改 fingerprint/metric/backend/dimension → 变号；
               改 created_at / parent_version / metadata → **不变号**（本课的重点）
落盘往返       write/read 一致；目录与文件两种写法都认；被改过的清单读回来就炸
从库重建       FlatVectorStore 上的样本能重建出条目；空库得到 0 条；没有原文又
               没有 fingerprint 的记录必须报错（不许猜一个指纹出来）
与库对账       missing / orphan / dimension / metric / backend 各自能单独变红，
               problems 里的句子含具体 id
```

全部离线、确定性、零网络：条目里的两个指纹是写死的常量，
向量由 ``vectorstore_samples`` 的样本给出，期望值手算得出来。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smart_research_agent.documents.types import content_id
from smart_research_agent.indexing.errors import ManifestError
from smart_research_agent.indexing.manifest import (
    MANIFEST_FILE,
    build_manifest,
    compare_with_store,
    manifest_from_store,
    read_manifest,
    verify_index,
    write_manifest,
)
from smart_research_agent.indexing.types import (
    EmbeddingIdentity,
    IndexEntry,
    entries_digest,
    index_version_id,
    vector_key,
)
from smart_research_agent.vectorstore.flat import FlatVectorStore
from smart_research_agent.vectorstore.types import make_record
from tests.vectorstore_samples import RECORD_IDS, sample_records

#: 清单的编码器身份（本文件直连 ``build_manifest``，与真实编码器无关）.
IDENTITY = EmbeddingIdentity(provider="MockEmbedding", model="mock-64", dimension=64)

#: 库里样本记录在 manifest_from_store 里应该得到的身份（8 维，与样本一致）.
SAMPLE_IDENTITY = EmbeddingIdentity(
    provider="TableEmbedding", model="sample-8", dimension=8
)

#: 三个内容指纹与三个向量键：都是 16 位小写十六进制（``IndexEntry`` 的硬要求）.
FP_A = "0123456789abcdef"
FP_B = "fedcba9876543210"
FP_C = "aabbccddeeff0011"
VK_A = "0011223344556677"
VK_B = "8899aabbccddeeff"
VK_C = "1122334455667788"

#: 一条"绕过索引流程直接写进库"的记录 id（与样本的六个 id 都不相同）.
EXTRA_ID = "f0e1d2c3b4a59687"

#: 三条条目：第三条带 token 数，用来验证统计字段的搬运.
ENTRIES: tuple[IndexEntry, ...] = (
    IndexEntry(record_id="1c9f4b7a2e6d8035", fingerprint=FP_A, vector_key=VK_A),
    IndexEntry(
        record_id="4a6680cde33da45e",
        fingerprint=FP_B,
        vector_key=VK_B,
        token_count=12,
    ),
    IndexEntry(
        record_id="90a20c398e18fb0b",
        fingerprint=FP_C,
        vector_key=VK_C,
        token_count=34,
    ),
)


# --------------------------------------------------------------------------- #
# 构造器
# --------------------------------------------------------------------------- #


def _entry(
    record_id: str,
    *,
    fingerprint: str = FP_A,
    vector_key: str = VK_A,
    token_count: int = 0,
) -> IndexEntry:
    """造一条条目（默认值让"只想改一个字段"的用例只写一行）."""
    return IndexEntry(
        record_id=record_id,
        fingerprint=fingerprint,
        vector_key=vector_key,
        token_count=token_count,
    )


def _manifest(
    entries: tuple[IndexEntry, ...] = ENTRIES,
    *,
    identity: EmbeddingIdentity = IDENTITY,
    metric: str = "cosine",
    backend: str = "flat",
    parent_version: str = "",
    created_at: str = "",
    metadata: dict[str, str] | None = None,
):
    """``build_manifest`` 的便捷包装（本文件所有用例都从这里出发）."""
    return build_manifest(
        entries,
        identity=identity,
        metric=metric,
        backend=backend,
        parent_version=parent_version,
        created_at=created_at,
        metadata=metadata,
    )


def _store_with_samples() -> FlatVectorStore:
    """一个装了六条样本记录的 flat 库（维度 8、度量 cosine）."""
    store = FlatVectorStore(metric="cosine")
    store.upsert(sample_records())
    return store


# --------------------------------------------------------------------------- #
# 版本号的构成
# --------------------------------------------------------------------------- #


def test_version_id_is_recomputable_from_content() -> None:
    """版本号必须能被任何人在不看代码的情况下重算出来."""
    manifest = _manifest()
    expected = index_version_id(
        identity_key=IDENTITY.key,
        metric="cosine",
        backend="flat",
        digest=entries_digest(ENTRIES),
    )
    assert manifest.version_id == expected
    assert manifest.verify_version_id() == expected


def test_entry_order_does_not_change_version_id() -> None:
    """同一份内容按不同顺序传进来 → 同一个版本号（否则重建一次就多一版）."""
    forward = _manifest(ENTRIES)
    backward = _manifest(tuple(reversed(ENTRIES)))
    assert forward.version_id == backward.version_id
    assert forward.record_ids == tuple(sorted(item.record_id for item in ENTRIES))
    assert forward.to_dict() == backward.to_dict()


def test_fingerprint_change_changes_version_id() -> None:
    """改一个内容指纹 → 版本号必须变（内容变了）."""
    tweaked = (_entry("1c9f4b7a2e6d8035", fingerprint=FP_B), *ENTRIES[1:])
    assert _manifest(tweaked).version_id != _manifest().version_id


def test_metric_change_changes_version_id() -> None:
    """换度量 → 最近邻整体改变，版本号必须变."""
    assert _manifest(metric="l2").version_id != _manifest().version_id


def test_backend_change_changes_version_id() -> None:
    """换后端 → 可以重建，但不能与旧版共用一份清单."""
    assert _manifest(backend="chroma").version_id != _manifest().version_id


def test_dimension_change_changes_version_id() -> None:
    """同一个 provider/model、不同输出维度 → 不是同一套编码器."""
    other = EmbeddingIdentity(provider="MockEmbedding", model="mock-64", dimension=128)
    assert _manifest(identity=other).version_id != _manifest().version_id


def test_created_at_and_notes_do_not_change_version_id() -> None:
    """本课的重点：``created_at`` / ``parent_version`` / ``metadata`` 都不进版本号.

    时间戳进版本号会让"同一份内容在两台机器上得到两个版本号"，
    于是"这个版本在哪台机器上建的"变成一个必须回答的问题——而它本来不该是一个问题。
    """
    naked = _manifest()
    decorated = _manifest(
        created_at="2026-09-19T12:00:00Z",
        parent_version=FP_A,
        metadata={"trigger": "test", "operator": "B"},
    )
    assert decorated.created_at == "2026-09-19T12:00:00Z"
    assert decorated.parent_version == FP_A
    assert decorated.version_id == naked.version_id


# --------------------------------------------------------------------------- #
# 落盘与读回
# --------------------------------------------------------------------------- #


def test_write_and_read_manifest_round_trip(tmp_path: Path) -> None:
    """写进目录再读回来，清单逐字段相同（含 entries）."""
    manifest = _manifest(created_at="2026-09-19T12:00:00Z", metadata={"trigger": "test"})
    written = write_manifest(manifest, str(tmp_path))
    assert Path(written) == tmp_path / MANIFEST_FILE
    assert isinstance(json.loads(Path(written).read_text(encoding="utf-8")), dict)
    restored = read_manifest(str(tmp_path))
    assert restored.to_dict() == manifest.to_dict()
    assert restored.verify_version_id() == manifest.version_id


def test_write_manifest_accepts_directory_file_and_missing_directory(tmp_path: Path) -> None:
    """目录、已有文件、还不存在的无后缀目录 —— 三种写法都要指到对的地方."""
    manifest = _manifest()
    nested = tmp_path / "index"  # 不存在且没有后缀 → 按目录处理
    assert Path(write_manifest(manifest, str(nested))) == nested / MANIFEST_FILE

    custom = tmp_path / "custom.json"  # 有后缀 → 按文件处理
    assert write_manifest(manifest, str(custom)) == str(custom)
    assert read_manifest(str(custom)).version_id == manifest.version_id
    assert read_manifest(str(nested)).version_id == manifest.version_id


def test_write_manifest_rejects_empty_path() -> None:
    """空路径会让清单落到进程当前目录，而调用方以为它写在别处."""
    with pytest.raises(ManifestError) as excinfo:
        write_manifest(_manifest(), "   ")
    assert "清单需要一个路径" in str(excinfo.value)


def test_write_manifest_rejects_tampered_manifest(tmp_path: Path) -> None:
    """磁盘上不该出现一份读不回来的清单——校验放在写的一侧."""
    from smart_research_agent.indexing.types import IndexManifest

    tampered = IndexManifest(
        version_id=FP_B,
        identity=IDENTITY,
        metric="cosine",
        backend="flat",
        entries=ENTRIES,
    )
    with pytest.raises(ManifestError) as excinfo:
        write_manifest(tampered, str(tmp_path / "index"))
    assert "version_id 与内容不一致" in str(excinfo.value)
    assert not (tmp_path / "index").exists()


def test_read_manifest_rejects_tampered_content(tmp_path: Path) -> None:
    """改一个 fingerprint 而不改版本号 → 读回来就炸（这就是"可被任何人重算"）."""
    path = tmp_path / MANIFEST_FILE
    payload = _manifest().to_dict()
    payload["entries"][0]["fingerprint"] = "ffffffffffffffff"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with pytest.raises(ManifestError) as excinfo:
        read_manifest(str(path))
    assert "version_id 与内容不一致" in str(excinfo.value)


def test_read_manifest_rejects_unknown_format_version(tmp_path: Path) -> None:
    """格式版本不匹配时拒读（用旧格式驱动增量会让差集算错）."""
    path = tmp_path / MANIFEST_FILE
    payload = _manifest().to_dict()
    payload["manifest_version"] = 99
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with pytest.raises(ManifestError) as excinfo:
        read_manifest(str(path))
    assert "清单格式版本是 99" in str(excinfo.value)


def test_read_manifest_rejects_broken_json_and_missing_file(tmp_path: Path) -> None:
    """半截写入的文件与不存在的文件是两种失败，都要有各自的说法."""
    half = tmp_path / "half.json"
    half.write_text('{"version_id": "012345', encoding="utf-8")
    with pytest.raises(ManifestError) as excinfo:
        read_manifest(str(half))
    assert "不是合法的 JSON" in str(excinfo.value)

    with pytest.raises(ManifestError) as missing:
        read_manifest(str(tmp_path / "nope.json"))
    assert "清单文件不存在" in str(missing.value)


def test_read_manifest_rejects_non_object_payload(tmp_path: Path) -> None:
    """顶层不是对象 → 不是一份清单（读成列表再往下走只会得到更难懂的错误）."""
    path = tmp_path / MANIFEST_FILE
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ManifestError) as excinfo:
        read_manifest(str(path))
    assert "顶层必须是对象" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# 从库重建清单
# --------------------------------------------------------------------------- #


class _DesyncedStore(FlatVectorStore):
    """一个"向量索引里有、记录表里没有"的库（删除只做了一半时的形状）.

    真实世界里这个状态来自"先删了记录表、还没删索引"或两个文件不是一次写出的；
    它的危害是检索命中一个没有原文的 id（见 ``vectorstore.evaluate.index_health``）。
    重建清单时遇到它必须报错，而不是把这条幻影记成一个条目。
    """

    PHANTOM = "0f2c4b8e1d3a7695"

    def ids(self) -> list[str]:
        return sorted([*super().ids(), self.PHANTOM])


def test_manifest_from_store_rejects_desynced_store() -> None:
    """库自相矛盾（``ids()`` 说有一条、``get()`` 取不到）时拒建清单."""
    store = _DesyncedStore(metric="cosine")
    store.upsert(sample_records())
    with pytest.raises(ManifestError) as excinfo:
        manifest_from_store(store, identity=SAMPLE_IDENTITY)
    message = str(excinfo.value)
    assert "库自相矛盾" in message
    assert _DesyncedStore.PHANTOM in message
    assert "index_health" in message


def test_manifest_from_store_rebuilds_fingerprints_and_keys() -> None:
    """库里没有 fingerprint 时，内容身份由**原文**现算；向量键走编码文本规则."""
    store = _store_with_samples()
    manifest = manifest_from_store(store, identity=SAMPLE_IDENTITY)

    assert manifest.backend == "flat"
    assert manifest.metric == "cosine"
    assert manifest.count == len(RECORD_IDS)
    assert manifest.record_ids == tuple(sorted(RECORD_IDS))
    assert manifest.version_id == manifest.verify_version_id()

    entries = manifest.entry_map()
    for record in sample_records():
        item = entries[record.record_id]
        assert item.fingerprint == content_id(record.text)
        assert item.vector_key == vector_key(
            SAMPLE_IDENTITY.provider, SAMPLE_IDENTITY.model, SAMPLE_IDENTITY.dimension,
            record.text,
        )
    expected_tokens = sum(int(record.metadata["token_count"]) for record in sample_records())
    assert manifest.total_tokens == expected_tokens


def test_manifest_from_store_metric_override_changes_version_id() -> None:
    """``metric=None`` 取库的度量；显式覆盖要真的改变版本号（口径变了）."""
    store = _store_with_samples()
    default = manifest_from_store(store, identity=SAMPLE_IDENTITY)
    overridden = manifest_from_store(store, identity=SAMPLE_IDENTITY, metric="l2")
    assert overridden.metric == "l2"
    assert overridden.version_id != default.version_id


def test_manifest_from_store_on_empty_store_yields_empty_manifest() -> None:
    """空库是合法输入：得到 0 条清单，而不是异常."""
    manifest = manifest_from_store(
        FlatVectorStore(), identity=SAMPLE_IDENTITY, parent_version=FP_A
    )
    assert manifest.count == 0
    assert manifest.record_ids == ()
    assert manifest.digest == entries_digest(())
    assert manifest.parent_version == FP_A


def test_manifest_from_store_prefers_metadata_fingerprint_without_text() -> None:
    """原文不在库里时用 metadata["fingerprint"]（day062 的内容身份）."""
    store = FlatVectorStore(dimension=8)
    store.upsert(
        [
            make_record(
                "a1b2c3d4e5f60718",
                [1.0] + [0.0] * 7,
                "",
                {"fingerprint": FP_A, "token_count": "7"},
            )
        ]
    )
    manifest = manifest_from_store(store, identity=SAMPLE_IDENTITY)
    item = manifest.entries[0]
    assert item.fingerprint == FP_A
    assert item.token_count == 7
    # 编码文本回落到空串（原文与 retrieval_text 都不在），但仍然算得出来一个键
    assert item.vector_key == vector_key("TableEmbedding", "sample-8", 8, "")


def test_manifest_from_store_treats_unparsable_token_count_as_zero() -> None:
    """``token_count`` 只是统计量：取不出整数时记 0，不能让整份清单建不出来."""
    store = FlatVectorStore(dimension=8)
    store.upsert(
        [
            make_record(
                "c1b2c3d4e5f60718",
                [1.0] + [0.0] * 7,
                "这一条的 token_count 被上游写成了非数字。",
                {"token_count": "很多"},
            )
        ]
    )
    manifest = manifest_from_store(store, identity=SAMPLE_IDENTITY)
    assert manifest.entries[0].token_count == 0
    assert manifest.total_tokens == 0


def test_manifest_from_store_rejects_record_without_text_or_fingerprint() -> None:
    """两者都没有时**必须报错**：不许猜一个指纹出来."""
    store = FlatVectorStore(dimension=8)
    store.upsert([make_record("b1b2c3d4e5f60718", [1.0] + [0.0] * 7, "", {})])
    with pytest.raises(ManifestError) as excinfo:
        manifest_from_store(store, identity=SAMPLE_IDENTITY)
    message = str(excinfo.value)
    assert "无法重建内容身份" in message
    assert "b1b2c3d4e5f60718" in message
    assert "从 chunk 记录重建清单" in message


# --------------------------------------------------------------------------- #
# 与库对账
# --------------------------------------------------------------------------- #


def test_compare_with_store_fully_consistent() -> None:
    """完全一致 → 差集为空、口径三项全对、token 两边相等."""
    store = _store_with_samples()
    manifest = manifest_from_store(store, identity=SAMPLE_IDENTITY)
    report = compare_with_store(manifest, store)

    assert report["missing_in_store"] == []
    assert report["orphan_in_store"] == []
    assert report["count_manifest"] == len(RECORD_IDS)
    assert report["count_store"] == len(RECORD_IDS)
    assert report["dimension_match"] is True
    assert report["metric_match"] is True
    assert report["backend_match"] is True
    assert report["tokens_match"] is True
    assert report["extra_tokens"]["diff"] == 0
    json.dumps(report)  # 端点会直接返回它，因此必须可 json 序列化


def test_compare_with_store_reports_missing_after_delete() -> None:
    """库里删一条 → 它在 missing_in_store 里（这是"上一次构建半途失败"的形状）."""
    store = _store_with_samples()
    manifest = manifest_from_store(store, identity=SAMPLE_IDENTITY)
    assert store.delete([RECORD_IDS[0]]) == 1

    report = compare_with_store(manifest, store)
    assert report["missing_in_store"] == [RECORD_IDS[0]]
    assert report["orphan_in_store"] == []
    assert report["count_store"] == len(RECORD_IDS) - 1


def test_compare_with_store_reports_orphan_after_external_write() -> None:
    """有人绕过索引流程直接写了库 → 它在 orphan_in_store 里."""
    store = _store_with_samples()
    manifest = manifest_from_store(store, identity=SAMPLE_IDENTITY)
    store.upsert(
        [
            make_record(
                EXTRA_ID,
                [0.0] * 7 + [1.0],
                "这条是绕过索引流程直接写进来的。",
                {"token_count": 5},
            )
        ]
    )

    report = compare_with_store(manifest, store)
    assert report["orphan_in_store"] == [EXTRA_ID]
    assert report["missing_in_store"] == []
    assert report["count_store"] == len(RECORD_IDS) + 1
    assert report["tokens_match"] is False
    assert report["extra_tokens"]["diff"] == 5


def test_compare_with_store_detects_dimension_mismatch() -> None:
    """用另一个维度的清单比 → dimension_match 为假（清单描述的是另一个库）."""
    store = _store_with_samples()
    wide = EmbeddingIdentity(provider="TableEmbedding", model="sample-16", dimension=16)
    manifest = manifest_from_store(store, identity=wide)

    report = compare_with_store(manifest, store)
    assert report["dimension_match"] is False
    assert report["metric_match"] is True
    assert report["backend_match"] is True
    assert report["missing_in_store"] == []
    assert report["orphan_in_store"] == []


# --------------------------------------------------------------------------- #
# 体检结论
# --------------------------------------------------------------------------- #


def test_verify_index_ok_when_manifest_matches_store() -> None:
    """清单与库对得上 → ok 为真、problems 为空."""
    store = _store_with_samples()
    manifest = manifest_from_store(store, identity=SAMPLE_IDENTITY)
    report = verify_index(manifest, store)

    assert report["ok"] is True
    assert report["problems"] == []
    assert report["checks"]["missing_in_store"] == []
    assert report["checks"]["orphan_in_store"] == []
    assert report["checks"]["dimension_match"] is True


def test_verify_index_problems_name_the_records() -> None:
    """ok 为假时，problems 里每一句都要能照着做（含具体 id）. """
    store = _store_with_samples()
    manifest = manifest_from_store(store, identity=SAMPLE_IDENTITY)
    store.delete([RECORD_IDS[0]])
    store.upsert(
        [
            make_record(
                EXTRA_ID,
                [0.0] * 7 + [1.0],
                "这条是绕过索引流程直接写进来的。",
                {},
            )
        ]
    )

    report = verify_index(manifest, store)
    assert report["ok"] is False
    assert len(report["problems"]) == 2

    orphan_line = next(line for line in report["problems"] if "不在清单里" in line)
    assert EXTRA_ID in orphan_line
    assert "不会被增量更新覆盖" in orphan_line
    assert "请先重建清单" in orphan_line

    missing_line = next(line for line in report["problems"] if "在库里找不到" in line)
    assert RECORD_IDS[0] in missing_line


def test_verify_index_rejects_manifest_from_another_encoder() -> None:
    """维度不符必须让 ok 为假（否则"库与清单对不上"会被伪装成"数据全变了"）."""
    store = _store_with_samples()
    wide = EmbeddingIdentity(provider="TableEmbedding", model="sample-16", dimension=16)
    manifest = manifest_from_store(store, identity=wide)

    report = verify_index(manifest, store)
    assert report["ok"] is False
    assert report["checks"]["dimension_match"] is False
    assert any("维度不一致" in line for line in report["problems"])
    assert any("16 维" in line and "8 维" in line for line in report["problems"])


def test_verify_index_rejects_metric_and_backend_mismatch() -> None:
    """条目一模一样、口径不同 → 差集干净但 ok 仍为假（度量与后端各一句结论）."""
    store = _store_with_samples()
    manifest = manifest_from_store(store, identity=SAMPLE_IDENTITY)
    other = build_manifest(
        manifest.entries,
        identity=SAMPLE_IDENTITY,
        metric="l2",
        backend="chroma",
    )

    report = verify_index(other, store)
    assert report["ok"] is False
    assert report["checks"]["dimension_match"] is True
    assert report["checks"]["metric_match"] is False
    assert report["checks"]["backend_match"] is False
    assert any("度量不一致" in line for line in report["problems"])
    assert any("后端不一致" in line for line in report["problems"])
