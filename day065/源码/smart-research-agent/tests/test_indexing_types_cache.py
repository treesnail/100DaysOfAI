"""``indexing.types`` 与 ``indexing.cache`` 的契约测试：形状、报错分支、三条硬校验.

这两个模块是整个 ``indexing`` 包的**契约文件**：``types`` 定义"清单长什么样、
什么算合法"，``cache`` 定义"向量键属于谁、什么时候拒读"。它们自己不搬数据、
不碰磁盘格式，因此能被单独测；而正因为它们是契约，**每一条报错分支都值得单独变红**——
一个不会炸的校验等于没有校验。

本文件盯住三件事：

```text
types  每个 dataclass 的 __post_init__ 校验各自能单独触发；
       报错句子含具体字段名与收到的值；版本号/摘要可重算且顺序无关
cache  键里含编码器身份；命中/未命中分开计数；has 不计数；
       prune 只删不在 keep 里的；load 的版本/身份/维度/count 四条校验
       与 persist/load 的"没给路径"分支各自能单独变红
```

全部离线、确定性、零网络：向量是手写的小整数列表，
键与摘要的期望值在测试里用 ``hashlib`` 独立重算一遍，
落盘只用 ``tmp_path``。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from smart_research_agent.indexing import cache as cache_module
from smart_research_agent.indexing.cache import CACHE_VERSION, EmbeddingCache
from smart_research_agent.indexing.errors import (
    EncodingError,
    IndexingError,
    ManifestError,
)
from smart_research_agent.indexing.types import (
    ACTIONS,
    ACTION_ADDED,
    ACTION_REMOVED,
    ACTION_UNCHANGED,
    ACTION_UPDATED,
    MANIFEST_VERSION,
    VECTOR_KEY_LENGTH,
    EmbeddingIdentity,
    EncodeReport,
    IndexEntry,
    IndexManifest,
    IndexingReport,
    IndexPlan,
    entries_digest,
    index_version_id,
    vector_key,
)

#: 本文件统一的编码器身份与维度。取 3 而不是 384：向量一眼能看完，期望值能手算。
PROVIDER = "MockEmbedding"
MODEL = "mock-3"
DIMENSION = 3

#: 三个 16 位小写十六进制常量：``IndexEntry`` 与 ``vector_key`` 的硬要求就是这个形状。
FP_A = "0123456789abcdef"
FP_B = "fedcba9876543210"
VK_A = "0011223344556677"
VK_B = "8899aabbccddeeff"


# --------------------------------------------------------------------------- #
# 构造器与落盘小工具
# --------------------------------------------------------------------------- #


def _identity(dimension: int = DIMENSION) -> EmbeddingIdentity:
    """造一个合法的编码器身份（只想改一个字段的用例只写一行）."""
    return EmbeddingIdentity(provider=PROVIDER, model=MODEL, dimension=dimension)


def _entry(
    record_id: str,
    *,
    fingerprint: str = FP_A,
    vector_key_value: str = VK_A,
    token_count: int = 0,
) -> IndexEntry:
    """造一条合法条目（默认值让"只想改一个字段"的用例只写一行）."""
    return IndexEntry(
        record_id=record_id,
        fingerprint=fingerprint,
        vector_key=vector_key_value,
        token_count=token_count,
    )


def _manifest(
    entries: tuple[IndexEntry, ...] = (),
    *,
    version_id: str = FP_A,
    identity: EmbeddingIdentity | None = None,
    metric: str = "cosine",
    backend: str = "flat",
    parent_version: str = "",
    created_at: str = "",
    metadata: dict[str, str] | None = None,
) -> IndexManifest:
    """``IndexManifest`` 的便捷包装（默认 version_id 只求形状合法，不求内容一致）."""
    return IndexManifest(
        version_id=version_id,
        identity=identity if identity is not None else _identity(),
        metric=metric,
        backend=backend,
        entries=entries,
        parent_version=parent_version,
        created_at=created_at,
        metadata=metadata if metadata is not None else {},
    )


def _cache(**kwargs: object) -> EmbeddingCache:
    """造一份缓存：身份默认与本文件常量一致，可按用例覆盖单个字段."""
    params: dict[str, object] = {
        "provider": PROVIDER,
        "model": MODEL,
        "dimension": DIMENSION,
    }
    params.update(kwargs)
    return EmbeddingCache(**params)  # type: ignore[arg-type]


def _payload(
    entries: dict[str, list[float]] | None = None,
    **overrides: object,
) -> dict[str, object]:
    """造一份合法的缓存文件内容；``count`` 默认与实际条数一致（可被覆盖）."""
    body: dict[str, object] = {
        "cache_version": CACHE_VERSION,
        "provider": PROVIDER,
        "model": MODEL,
        "dimension": DIMENSION,
        "identity_key": _identity().key,
        "count": len(entries or {}),
        "entries": dict(entries or {}),
    }
    body.update(overrides)
    return body


def _write(tmp_path: Path, body: dict[str, object], name: str = "cache.json") -> Path:
    """把缓存文件内容写进 ``tmp_path`` 并返回路径（落盘只用 tmp_path）."""
    path = tmp_path / name
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


class _BlankPath:
    """``__str__`` 为空的 Path 替身，用来单独触发"没给路径"那条护栏.

    为什么需要它：真实世界里 ``Path("")`` 会折叠成 ``Path(".")``——
    ``str`` 是 ``"."``（非空），因此 ``persist()`` / ``load()`` 里
    ``if not str(target):`` 这条护栏**用真实参数永远进不去**。
    用一个替身把它的 True 分支单独变成红：护栏哪天被删掉，本用例就会失败。
    """

    def __str__(self) -> str:
        return ""


def _blank_path(*_args: object, **_kwargs: object) -> _BlankPath:
    """替换 ``cache.Path`` 的工厂：无论收到什么参数都返回空串路径."""
    return _BlankPath()



# --------------------------------------------------------------------------- #
# 纯函数：向量键、内容摘要、版本号
# --------------------------------------------------------------------------- #


def test_vector_key_is_sha256_prefix_and_identity_sensitive() -> None:
    """键 = sha256("provider|model|dimension\\ntext")[:16]；四个输入一个都不能少."""
    text = "你好"
    expected = hashlib.sha256(f"{PROVIDER}|{MODEL}|3\n{text}".encode()).hexdigest()[
        :VECTOR_KEY_LENGTH
    ]
    assert len(expected) == 16
    assert vector_key(PROVIDER, MODEL, 3, text) == expected
    # 同一输入可复现
    assert vector_key(PROVIDER, MODEL, 3, text) == vector_key(PROVIDER, MODEL, 3, text)
    # 四个输入各自改变都必须换键（尤其是换模型：那是"不报错的一类错误"）
    assert vector_key(PROVIDER, MODEL, 3, text) != vector_key("Other", MODEL, 3, text)
    assert vector_key(PROVIDER, MODEL, 3, text) != vector_key(PROVIDER, "other", 3, text)
    assert vector_key(PROVIDER, MODEL, 3, text) != vector_key(PROVIDER, MODEL, 4, text)
    assert vector_key(PROVIDER, MODEL, 3, text) != vector_key(PROVIDER, MODEL, 3, "再见")


def test_entries_digest_is_order_independent_and_ignores_stats() -> None:
    """摘要按 record_id 升序重算，因此顺序无关；token_count 只做统计、不进摘要."""
    left = _entry("record-a", fingerprint=FP_A, vector_key_value=VK_A, token_count=1)
    right = _entry("record-b", fingerprint=FP_B, vector_key_value=VK_B, token_count=2)
    ordered = (
        f"{left.record_id}|{left.fingerprint}|{left.vector_key}\n"
        f"{right.record_id}|{right.fingerprint}|{right.vector_key}"
    )
    expected = hashlib.sha256(ordered.encode()).hexdigest()[:VECTOR_KEY_LENGTH]
    assert entries_digest((left, right)) == expected
    # 顺序无关：同一份内容按不同顺序算得同一个摘要
    assert entries_digest((right, left)) == expected
    # 只改统计字段 → 摘要不变（否则"改一个统计字段"会变成"整库失效"）
    bumped = _entry("record-b", fingerprint=FP_B, vector_key_value=VK_B, token_count=999)
    assert entries_digest((left, bumped)) == expected
    # 改内容指纹 → 摘要必须变
    tweaked = _entry("record-b", fingerprint=FP_A, vector_key_value=VK_B, token_count=2)
    assert entries_digest((left, tweaked)) != expected


def test_index_version_id_depends_on_all_four_inputs() -> None:
    """版本号由"身份 + 度量 + 后端 + 内容摘要"决定，四个里任一个变了就换号."""
    args = {"identity_key": FP_A, "metric": "cosine", "backend": "flat", "digest": FP_B}
    expected = hashlib.sha256(
        f"{MANIFEST_VERSION}|{FP_A}|cosine|flat|{FP_B}".encode()
    ).hexdigest()[:VECTOR_KEY_LENGTH]
    assert index_version_id(**args) == expected
    assert index_version_id(**{**args, "identity_key": VK_A}) != expected
    assert index_version_id(**{**args, "metric": "l2"}) != expected
    assert index_version_id(**{**args, "backend": "chroma"}) != expected
    assert index_version_id(**{**args, "digest": VK_B}) != expected


def test_action_constants_keep_report_order() -> None:
    """``ACTIONS`` 的顺序就是报告里四行的顺序（差集打印依赖它）."""
    assert ACTIONS == (ACTION_ADDED, ACTION_UPDATED, ACTION_REMOVED, ACTION_UNCHANGED)


# --------------------------------------------------------------------------- #
# EmbeddingIdentity：三个字段缺一不可
# --------------------------------------------------------------------------- #


def test_identity_rejects_blank_provider() -> None:
    """provider 空白串 → 报错（空白不是"没填"，是"填错了"）."""
    with pytest.raises(ManifestError) as excinfo:
        EmbeddingIdentity(provider="   ", model=MODEL, dimension=DIMENSION)
    assert "EmbeddingIdentity.provider 不能为空" in str(excinfo.value)


def test_identity_rejects_blank_model() -> None:
    """model 空白串 → 报错，且提示"没有模型的编码器身份无法回答用谁建的"."""
    with pytest.raises(ManifestError) as excinfo:
        EmbeddingIdentity(provider=PROVIDER, model=" ", dimension=DIMENSION)
    message = str(excinfo.value)
    assert "EmbeddingIdentity.model 不能为空" in message
    assert "n-gram" in message


@pytest.mark.parametrize("dimension", [0, -3])
def test_identity_rejects_non_positive_dimension(dimension: int) -> None:
    """维度必须 >= 1：0 与负数各报一次，消息里带上收到的值."""
    with pytest.raises(ManifestError) as excinfo:
        EmbeddingIdentity(provider=PROVIDER, model=MODEL, dimension=dimension)
    message = str(excinfo.value)
    assert "EmbeddingIdentity.dimension 必须 >= 1" in message
    assert f"收到 {dimension}" in message


def test_identity_key_summary_and_round_trip() -> None:
    """``key`` 走 dimension=0 的特别口径；``summary_line`` 与 to/from_dict 可往返."""
    identity = _identity()
    assert identity.key == vector_key(PROVIDER, MODEL, 0, f"dimension={DIMENSION}")
    # key 与"向量键"不同：它把维度写进 text 而不是走 dimension 参数
    assert identity.key != vector_key(PROVIDER, MODEL, DIMENSION, f"dimension={DIMENSION}")
    assert identity.summary_line() == f"{PROVIDER}/{MODEL} ({DIMENSION}d)"
    assert identity.to_dict() == {
        "provider": PROVIDER,
        "model": MODEL,
        "dimension": DIMENSION,
        "key": identity.key,
    }
    assert EmbeddingIdentity.from_dict(identity.to_dict()) == identity


@pytest.mark.parametrize("missing", ["provider", "model", "dimension"])
def test_identity_from_dict_reports_missing_field(missing: str) -> None:
    """缺字段时给出可读错误：消息里点名缺的是哪一个，并说明三者缺一不可."""
    payload = _identity().to_dict()
    del payload[missing]
    with pytest.raises(ManifestError) as excinfo:
        EmbeddingIdentity.from_dict(payload)
    message = str(excinfo.value)
    assert f"编码器身份缺少字段 '{missing}'" in message
    assert "三者缺一不可" in message


# --------------------------------------------------------------------------- #
# IndexEntry：三个字段 + 一个统计字段
# --------------------------------------------------------------------------- #


def test_entry_rejects_blank_record_id() -> None:
    """record_id 空白 → 报错（无法定位要更新/删除的记录）."""
    with pytest.raises(ManifestError) as excinfo:
        _entry("   ")
    assert "IndexEntry.record_id 不能为空" in str(excinfo.value)


@pytest.mark.parametrize("bad", ["abc", "0123456789abcde", "zzzzzzzzzzzzzzzz"])
def test_entry_rejects_bad_fingerprint(bad: str) -> None:
    """指纹必须恰好 16 位小写十六进制：太短、太短一点点、长度够但非十六进制都要炸."""
    with pytest.raises(ManifestError) as excinfo:
        _entry("record-a", fingerprint=bad)
    message = str(excinfo.value)
    assert "IndexEntry.fingerprint 必须是 16 位十六进制" in message
    assert f"收到 {bad!r}" in message


@pytest.mark.parametrize("bad", ["", "001122334455667", "GGGGGGGGGGGGGGGG"])
def test_entry_rejects_bad_vector_key(bad: str) -> None:
    """向量键与指纹同形：空串、15 位、大写字母都算坏形状."""
    with pytest.raises(ManifestError) as excinfo:
        _entry("record-a", vector_key_value=bad)
    message = str(excinfo.value)
    assert "IndexEntry.vector_key 必须是 16 位十六进制" in message
    assert f"收到 {bad!r}" in message


def test_entry_rejects_negative_token_count() -> None:
    """token_count 只用于统计，但仍然必须非负（负数说明上游算错了）."""
    with pytest.raises(ManifestError) as excinfo:
        _entry("record-a", token_count=-1)
    assert "IndexEntry.token_count 必须非负，收到 -1" in str(excinfo.value)


def test_entry_to_dict_summary_and_from_dict() -> None:
    """投影逐字段可控；``token_count`` 缺省时回落到 0."""
    entry = _entry("record-a", token_count=5)
    assert entry.summary_line() == f"record-a | fp={FP_A} | vk={VK_A} | 5 token"
    assert entry.to_dict() == {
        "record_id": "record-a",
        "fingerprint": FP_A,
        "vector_key": VK_A,
        "token_count": 5,
    }
    assert IndexEntry.from_dict(entry.to_dict()) == entry
    minimal = IndexEntry.from_dict(
        {"record_id": "record-a", "fingerprint": FP_A, "vector_key": VK_A}
    )
    assert minimal.token_count == 0


@pytest.mark.parametrize("missing", ["record_id", "fingerprint", "vector_key"])
def test_entry_from_dict_reports_missing_field(missing: str) -> None:
    """缺字段时点名是哪一条缺的哪一个（而不是一个裸 KeyError）."""
    payload = {"record_id": "record-a", "fingerprint": FP_A, "vector_key": VK_A}
    del payload[missing]
    with pytest.raises(ManifestError) as excinfo:
        IndexEntry.from_dict(payload)
    message = str(excinfo.value)
    assert "三者缺一不可" in message
    assert f"缺少 '{missing}'" in message


# --------------------------------------------------------------------------- #
# IndexManifest：形状校验 + 排序 + 版本号可重算 + 血缘
# --------------------------------------------------------------------------- #


def test_manifest_rejects_bad_version_id() -> None:
    """version_id 必须是 16 位十六进制（12 位这种"看起来像"的要拒掉）."""
    with pytest.raises(ManifestError) as excinfo:
        _manifest(version_id="not-a-hex-id")
    message = str(excinfo.value)
    assert "version_id 必须是 16 位十六进制" in message
    assert "not-a-hex-id" in message


def test_manifest_parent_version_allows_empty_but_rejects_bad() -> None:
    """parent_version 允许空串（首版），但非空时必须是 16 位十六进制."""
    assert _manifest(parent_version="").parent_version == ""
    assert _manifest(parent_version=FP_B).parent_version == FP_B
    with pytest.raises(ManifestError) as excinfo:
        _manifest(parent_version="XYZ")
    message = str(excinfo.value)
    assert "parent_version 必须是 16 位十六进制或空串" in message
    assert "'XYZ'" in message


def test_manifest_rejects_blank_backend_and_metric() -> None:
    """后端与度量都不能空白：清单必须说清它描述的是哪个库、按什么距离."""
    with pytest.raises(ManifestError) as excinfo:
        _manifest(backend="   ")
    assert "IndexManifest.backend 不能为空" in str(excinfo.value)
    with pytest.raises(ManifestError) as excinfo:
        _manifest(metric="")
    assert "IndexManifest.metric 不能为空" in str(excinfo.value)


def test_manifest_rejects_duplicate_record_id() -> None:
    """同一份索引不可能有两条同 id 的记录：重复通常意味着两次构建被拼在一起."""
    entries = (_entry("record-a"), _entry("record-a", fingerprint=FP_B))
    with pytest.raises(ManifestError) as excinfo:
        _manifest(entries=entries)
    message = str(excinfo.value)
    assert "清单里出现重复的 record_id 'record-a'" in message
    assert "两次构建的结果被拼在了一起" in message


def test_manifest_sorts_entries_and_exposes_stats() -> None:
    """构造时强制按 record_id 升序；count/total_tokens/entry_map 与内容一致."""
    entries = (_entry("record-c", token_count=3), _entry("record-a", token_count=7))
    manifest = _manifest(entries=entries)
    assert manifest.record_ids == ("record-a", "record-c")
    assert manifest.count == 2
    assert manifest.total_tokens == 10
    assert manifest.entry_map()["record-c"].token_count == 3
    assert manifest.digest == entries_digest(manifest.entries)


def test_manifest_verify_version_id_accepts_and_rejects() -> None:
    """版本号可被任何人重算：一致时返回重算值，被改过时当场炸."""
    entries = (_entry("record-a"),)
    identity = _identity()
    good = index_version_id(
        identity_key=identity.key,
        metric="cosine",
        backend="flat",
        digest=entries_digest(entries),
    )
    manifest = _manifest(entries=entries, version_id=good, identity=identity)
    assert manifest.verify_version_id() == good

    tampered = _manifest(entries=entries, version_id=FP_B, identity=identity)
    with pytest.raises(ManifestError) as excinfo:
        tampered.verify_version_id()
    assert "version_id 与内容不一致" in str(excinfo.value)


def test_manifest_to_dict_can_omit_entries() -> None:
    """``include_entries=False`` 时只给身份与统计（十万条清单不该被整份序列化）."""
    manifest = _manifest(
        entries=(_entry("record-a"),),
        created_at="2026-09-19T12:00:00Z",
        metadata={"k": "v"},
    )
    full = manifest.to_dict()
    assert full["manifest_version"] == MANIFEST_VERSION
    assert full["count"] == 1
    assert full["created_at"] == "2026-09-19T12:00:00Z"
    assert full["entries"] == [
        {"record_id": "record-a", "fingerprint": FP_A, "vector_key": VK_A, "token_count": 0}
    ]
    json.dumps(full)  # 端点会直接返回它，因此必须可 json 序列化

    lean = manifest.to_dict(include_entries=False)
    assert "entries" not in lean
    assert lean["metadata"] == {"k": "v"}
    assert lean["digest"] == manifest.digest


def test_manifest_from_dict_round_trip() -> None:
    """to_dict → from_dict 逐字段相同（含 entries / metadata / parent_version）."""
    manifest = _manifest(
        entries=(_entry("record-a", token_count=4),),
        version_id=FP_A,
        parent_version=FP_B,
        created_at="2026-09-19T12:00:00Z",
        metadata={"trigger": "test"},
    )
    restored = IndexManifest.from_dict(manifest.to_dict())
    assert restored.to_dict() == manifest.to_dict()
    assert restored.count == 1
    assert restored.entries[0].token_count == 4


def test_manifest_from_dict_rejects_unknown_format_version() -> None:
    """格式版本不匹配时拒读（用旧格式驱动增量会让差集算错）."""
    payload = _manifest().to_dict()
    payload["manifest_version"] = 99
    with pytest.raises(ManifestError) as excinfo:
        IndexManifest.from_dict(payload)
    assert "清单格式版本是 99" in str(excinfo.value)


def test_manifest_from_dict_rejects_non_list_entries() -> None:
    """entries 不是列表 → 不是一份清单（读成 dict 再往下走只会得到更难懂的错误）."""
    payload = _manifest().to_dict()
    payload["entries"] = {"record-a": {}}
    with pytest.raises(ManifestError) as excinfo:
        IndexManifest.from_dict(payload)
    assert "清单的 entries 必须是列表" in str(excinfo.value)


def test_manifest_summary_line_marks_first_version() -> None:
    """首版没有父版本，摘要里显示"（首版）"而不是一个空白."""
    manifest = _manifest(version_id=FP_A)
    line = manifest.summary_line()
    assert line.startswith(f"版本 {FP_A} ← （首版）")
    assert f"{PROVIDER}/{MODEL} ({DIMENSION}d)" in line
    assert "摘要" in line


def test_manifest_lineage_four_shapes() -> None:
    """血缘的四种写法：没有上一版 / 紧接 / 接在别的版本之后 / 没声明父版本."""
    first = _manifest(version_id=FP_A)
    assert first.lineage(None) == "首版（没有上一版）"

    child = _manifest(version_id=FP_B, parent_version=FP_A)
    assert child.lineage(first) == f"紧接 {FP_A}"
    # 父版本不是"最近的一个"：说清它接在谁之后，并点名它不是谁
    assert child.lineage(_manifest(version_id=VK_A)) == f"接在 {FP_A} 之后（不是 {VK_A}）"

    orphan = _manifest(version_id=VK_B)
    assert orphan.lineage(first) == f"没有声明父版本（当前最新是 {FP_A}）"


# --------------------------------------------------------------------------- #
# IndexPlan：四种去向互斥、升序且不重复
# --------------------------------------------------------------------------- #


def test_plan_rejects_duplicate_ids() -> None:
    """同一个动作里出现重复 id → 报错（报告要能被逐行 diff）."""
    with pytest.raises(IndexingError) as excinfo:
        IndexPlan(added=("b", "b"))
    assert "IndexPlan.added 里出现重复 id" in str(excinfo.value)


def test_plan_requires_ascending_ids() -> None:
    """每个动作的 id 列表必须升序（乱序的 id 列表每次都会长得不一样）."""
    with pytest.raises(IndexingError) as excinfo:
        IndexPlan(updated=("b", "a"))
    assert "IndexPlan.updated 必须是升序" in str(excinfo.value)


def test_plan_rejects_id_in_two_actions() -> None:
    """一个块只能有一个去向；跨动作重复要当场点名."""
    with pytest.raises(IndexingError) as excinfo:
        IndexPlan(added=("a",), updated=("a",))
    assert "同一个 id 不能同时出现在两个动作里，例如 'a'" in str(excinfo.value)


def test_plan_metrics_and_projection() -> None:
    """total / changed / to_encode / reuse_ratio 手算可验；投影可按需省略 id 列表."""
    plan = IndexPlan(
        added=("a",),
        updated=("c",),
        removed=("e",),
        unchanged=("b", "d"),
        reason="test",
    )
    assert plan.total == 5
    assert plan.changed == 3
    assert plan.to_encode == ("a", "c")
    assert plan.reuse_ratio == 0.4
    assert plan.summary_line() == "新增 1 / 更新 1 / 删除 1 / 未变 2 | 复用率 40.00% | test"

    full = plan.to_dict()
    assert full["added_ids"] == ["a"]
    assert full["updated_ids"] == ["c"]
    assert full["removed_ids"] == ["e"]
    lean = plan.to_dict(include_ids=False)
    assert "added_ids" not in lean
    assert lean["total"] == 5
    assert lean["reason"] == "test"


def test_plan_reuse_ratio_on_empty_plan_is_zero() -> None:
    """空差集的复用率是 0.0，不是除零错误."""
    empty = IndexPlan()
    assert empty.total == 0
    assert empty.reuse_ratio == 0.0
    assert empty.changed == 0
    assert empty.to_encode == ()


# --------------------------------------------------------------------------- #
# EncodeReport 与 IndexingReport：两张报告的投影与摘要
# --------------------------------------------------------------------------- #


def test_encode_report_hit_ratio_and_projection() -> None:
    """texts=0 时命中率是 0.0；有请求时按 cache_hits/texts 手算."""
    assert EncodeReport().hit_ratio == 0.0
    report = EncodeReport(
        texts=4, cache_hits=3, encoded=1, batches=1, dimension=DIMENSION, provider=PROVIDER
    )
    assert report.hit_ratio == 0.75
    payload = report.to_dict()
    assert payload["texts"] == 4
    assert payload["cache_hits"] == 3
    assert payload["encoded"] == 1
    assert payload["batches"] == 1
    assert payload["hit_ratio"] == 0.75
    assert payload["provider"] == PROVIDER
    assert report.summary_line() == (
        f"4 份文本 → 命中 3 / 编码 1 | 1 批 | 命中率 75.00% | {PROVIDER} {DIMENSION}d"
    )


def test_indexing_report_rejects_unknown_mode() -> None:
    """mode 只有 full / incremental 两种（拼错时点名收到的值）."""
    with pytest.raises(IndexingError) as excinfo:
        IndexingReport(
            mode="rebuild",
            version_id=FP_A,
            parent_version="",
            backend="flat",
            metric="cosine",
            dimension=DIMENSION,
            provider=PROVIDER,
        )
    message = str(excinfo.value)
    assert "mode 只能是 full 或 incremental" in message
    assert "'rebuild'" in message


def test_indexing_report_ok_count_and_projection() -> None:
    """ok 只由 failed 决定；ok_count = 写入 + 未变；count 只在要求时才给."""
    report = IndexingReport(
        mode="full",
        version_id=FP_A,
        parent_version="",
        backend="flat",
        metric="cosine",
        dimension=DIMENSION,
        provider=PROVIDER,
        seen=5,
        written=4,
        unchanged=1,
        failed=0,
    )
    assert report.ok is True
    assert report.ok_count == 5
    assert "count" not in report.to_dict()
    assert report.to_dict(include_ids=True)["count"] == 5
    assert report.summary_line() == (
        f"full 构建 | {FP_A} ← （首版） | 5 条 → 写入 4 / 未变 1 / 删除 0"
        f" | 编码 0（命中 0，0 批） | 复用率 0.00%"
    )


def test_indexing_report_summary_mentions_failures_note_and_parent() -> None:
    """有失败与备注时，摘要末尾各追加一段（且都带具体数字/句子）."""
    report = IndexingReport(
        mode="incremental",
        version_id=FP_A,
        parent_version=FP_B,
        backend="flat",
        metric="cosine",
        dimension=DIMENSION,
        provider=PROVIDER,
        failed=2,
        note="变更比例 62% 超过阈值，已从 incremental 切到 full",
    )
    assert report.ok is False
    line = report.summary_line()
    assert f"← {FP_B}" in line
    assert "失败 2" in line
    assert "变更比例 62% 超过阈值" in line


# --------------------------------------------------------------------------- #
# EmbeddingCache：身份、读写、计数
# --------------------------------------------------------------------------- #


def test_cache_identity_and_key_follow_the_encoder() -> None:
    """键走声明维度；身份指纹走 dimension=0 的特别口径；默认只在内存里."""
    cache = _cache()
    assert cache.provider == PROVIDER
    assert cache.model == MODEL
    assert cache.dimension == DIMENSION
    assert cache.path == ""
    assert cache.identity_key == vector_key(PROVIDER, MODEL, 0, f"dimension={DIMENSION}")
    assert cache.key("你好") == vector_key(PROVIDER, MODEL, DIMENSION, "你好")


def test_cache_get_has_and_counters() -> None:
    """has 不计数；get 命中记 hits、未命中记 misses；取回的是副本."""
    cache = _cache()
    stored = cache.put("你好", [1, 2, 3])
    assert stored == vector_key(PROVIDER, MODEL, DIMENSION, "你好")
    assert cache.size == 1
    assert cache.has("你好") is True
    assert cache.has("没有这条") is False
    # has 是"先探测再决策"用的：它绝不能污染命中率
    assert (cache.hits, cache.misses) == (0, 0)

    assert cache.get("你好") == [1.0, 2.0, 3.0]
    assert cache.hits == 1
    assert cache.get("没有这条") is None
    assert cache.misses == 1
    # 返回的是副本：调用方改它不会污染缓存
    cached = cache.get("你好")
    assert cached is not None
    cached.append(99.0)
    assert cache.get("你好") == [1.0, 2.0, 3.0]


def test_cache_reset_counters_keeps_entries() -> None:
    """清零只动计数、不动条目（一次构建的命中率不该被上一次累加）."""
    cache = _cache()
    cache.put("你好", [1, 2, 3])
    cache.get("你好")
    cache.get("没有这条")
    assert (cache.hits, cache.misses) == (1, 1)
    cache.reset_counters()
    assert (cache.hits, cache.misses) == (0, 0)
    assert cache.size == 1
    assert cache.get("你好") == [1.0, 2.0, 3.0]


def test_cache_keys_sorted_and_stats() -> None:
    """keys 升序；stats 的 lookups / hit_ratio 在零查询与有查询下都算得对."""
    cache = _cache()
    cache.put("b", [1, 2, 3])
    cache.put("a", [4, 5, 6])
    assert cache.keys() == sorted(cache.keys())

    empty = cache.stats()
    assert empty["lookups"] == 0
    assert empty["hit_ratio"] == 0.0
    assert empty["size"] == 2

    cache.get("b")
    cache.get("没有这条")
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["lookups"] == 2
    assert stats["hit_ratio"] == 0.5
    assert stats["provider"] == PROVIDER
    assert stats["model"] == MODEL
    assert stats["dimension"] == DIMENSION
    assert stats["identity_key"] == cache.identity_key
    assert stats["path"] == ""


def test_cache_put_rejects_dimension_mismatch() -> None:
    """写入向量长度与声明维度不符 → EncodingError，且一个字节都不该落下."""
    cache = _cache()
    with pytest.raises(EncodingError) as excinfo:
        cache.put("你好", [1, 2])
    message = str(excinfo.value)
    assert "写入缓存的向量是 2 维" in message
    assert f"而这份缓存声明的是 {DIMENSION} 维" in message
    assert f"（{PROVIDER}/{MODEL}）" in message
    assert cache.size == 0


def test_cache_put_without_declared_dimension_accepts_any_length() -> None:
    """dimension=0 表示"不限"：任意长度都能写入，键里的维度也是 0."""
    cache = _cache(dimension=0)
    assert cache.put("你好", [1, 2]) == vector_key(PROVIDER, MODEL, 0, "你好")
    assert cache.size == 1
    assert cache.get("你好") == [1.0, 2.0]


def test_cache_put_vectors_stores_batch_and_returns_count() -> None:
    """批量写入返回条数，逐条都能按文本键取回."""
    cache = _cache()
    assert cache.put_vectors([("a", [1, 2, 3]), ("b", [4, 5, 6])]) == 2
    assert cache.keys() == sorted([cache.key("a"), cache.key("b")])
    assert cache.get("a") == [1.0, 2.0, 3.0]
    assert cache.get("b") == [4.0, 5.0, 6.0]


def test_cache_put_vectors_rejects_whole_batch() -> None:
    """批量里有一条维度不符 → 整批拒绝（部分写入会让"缓存里有什么"不可预测）."""
    cache = _cache()
    bad_text = "这一条只有两维所以整批被拒"
    with pytest.raises(EncodingError) as excinfo:
        cache.put_vectors([("a", [1, 2, 3]), (bad_text, [1, 2])])
    message = str(excinfo.value)
    assert "批量写入里有一条 2 维的向量" in message
    assert f"与缓存声明的 {DIMENSION} 维不符" in message
    assert f"文本前 20 字：{bad_text!r}" in message
    assert "整批拒绝而不是跳过" in message
    assert cache.size == 0


def test_cache_put_vectors_without_declared_dimension() -> None:
    """dimension=0 时批量写入同样不限长度."""
    cache = _cache(dimension=0)
    assert cache.put_vectors([("a", [1]), ("b", [1, 2, 3, 4])]) == 2
    assert cache.size == 2


# --------------------------------------------------------------------------- #
# EmbeddingCache：修剪与清空
# --------------------------------------------------------------------------- #


def test_cache_prune_deletes_only_stale_keys() -> None:
    """prune 只删不在 keep_keys 里的；重复调用不再删东西."""
    cache = _cache()
    cache.put("keep", [1, 2, 3])
    cache.put("stale-1", [1, 2, 3])
    cache.put("stale-2", [1, 2, 3])
    assert cache.prune([cache.key("keep")]) == 2
    assert cache.keys() == [cache.key("keep")]
    # 再修剪一次：没有新的可删
    assert cache.prune([cache.key("keep")]) == 0


def test_cache_clear_resets_entries_and_counters() -> None:
    """clear 把条目与计数一起清掉（计数描述的正是那些条目）."""
    cache = _cache()
    cache.put("a", [1, 2, 3])
    cache.get("a")
    cache.get("missing")
    cache.clear()
    assert cache.size == 0
    assert cache.keys() == []
    assert (cache.hits, cache.misses) == (0, 0)


# --------------------------------------------------------------------------- #
# EmbeddingCache：落盘 persist
# --------------------------------------------------------------------------- #


def test_cache_persist_requires_a_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """构造与调用都没给 path → 明确报错（纯内存缓存是合法用法，但不该落盘）.

    这里把 ``cache.Path`` 换成空串替身：真实 ``Path("")`` 会折叠成 ``Path(".")``，
    这条护栏用真实参数进不去（见 ``_BlankPath`` 的说明）。
    """
    monkeypatch.setattr(cache_module, "Path", _blank_path)
    with pytest.raises(ManifestError) as excinfo:
        _cache().persist()
    message = str(excinfo.value)
    assert "缓存没有落盘位置" in message
    assert "纯内存缓存是合法用法" in message


def test_cache_persist_and_load_round_trip(tmp_path: Path) -> None:
    """写出的文件字段可核对；读回后条目与路径都恢复."""
    cache = _cache()
    cache.put("a", [1, 2, 3])
    target = tmp_path / "nested" / "cache.json"
    written = cache.persist(str(target))
    assert written == str(target)
    assert cache.path == str(target)

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["cache_version"] == CACHE_VERSION
    assert payload["provider"] == PROVIDER
    assert payload["model"] == MODEL
    assert payload["dimension"] == DIMENSION
    assert payload["count"] == 1
    assert payload["identity_key"] == cache.identity_key
    assert payload["entries"][cache.key("a")] == [1.0, 2.0, 3.0]

    restored = _cache()
    assert restored.load(str(target)) == 1
    assert restored.path == str(target)
    assert restored.get("a") == [1.0, 2.0, 3.0]


def test_cache_persist_uses_constructor_path(tmp_path: Path) -> None:
    """persist() 不带参时用构造时给的 path."""
    target = tmp_path / "cache.json"
    cache = _cache(path=str(target))
    cache.put("a", [1, 2, 3])
    assert cache.persist() == str(target)
    assert json.loads(target.read_text(encoding="utf-8"))["count"] == 1


# --------------------------------------------------------------------------- #
# EmbeddingCache：读回 load 的四条硬校验
# --------------------------------------------------------------------------- #


def test_cache_load_requires_a_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """构造与调用都没给 path → 明确报错（同样需要空串路径替身，理由见 persist 那条）."""
    monkeypatch.setattr(cache_module, "Path", _blank_path)
    with pytest.raises(ManifestError) as excinfo:
        _cache().load()
    assert "缓存没有可读位置" in str(excinfo.value)


def test_cache_load_rejects_missing_file(tmp_path: Path) -> None:
    """文件不存在要说清路径，并提醒"首次构建时缓存本来就是空的"."""
    missing = tmp_path / "nope.json"
    with pytest.raises(ManifestError) as excinfo:
        _cache().load(str(missing))
    message = str(excinfo.value)
    assert "缓存文件不存在" in message
    assert str(missing) in message


def test_cache_load_rejects_unknown_version(tmp_path: Path) -> None:
    """格式版本不匹配 → 拒读（读进来一半比读不进来更糟）."""
    path = _write(tmp_path, _payload(cache_version=99))
    with pytest.raises(ManifestError) as excinfo:
        _cache().load(str(path))
    message = str(excinfo.value)
    assert "缓存文件格式版本是 99" in message
    assert f"本代码只认识 {CACHE_VERSION}" in message


def test_cache_load_rejects_identity_mismatch(tmp_path: Path) -> None:
    """文件身份与本实例不一致 → 拒读（否则会让人以为缓存能跨模型共享）."""
    path = _write(tmp_path, _payload(provider="OtherEmbedding"))
    with pytest.raises(ManifestError) as excinfo:
        _cache().load(str(path))
    message = str(excinfo.value)
    assert "缓存身份不匹配" in message
    assert "OtherEmbedding" in message
    assert f"本实例是 {PROVIDER}/{MODEL}" in message
    assert "换了编码器就新建一份缓存" in message


def test_cache_load_rejects_non_object_entries(tmp_path: Path) -> None:
    """entries 不是对象 → 拒读（不是一份键到向量的表）."""
    body = _payload({FP_A: [1, 2, 3]}, count=1)
    body["entries"] = []  # 直接改 payload：列表不是"键 → 向量"的表
    path = _write(tmp_path, body)
    with pytest.raises(ManifestError) as excinfo:
        _cache().load(str(path))
    assert "entries 必须是对象" in str(excinfo.value)


def test_cache_load_rejects_bad_key_length(tmp_path: Path) -> None:
    """条目键长度不对 → 拒读（键形状坏了说明文件被改过）."""
    path = _write(tmp_path, _payload({"short": [1, 2, 3]}, count=1))
    with pytest.raises(ManifestError) as excinfo:
        _cache().load(str(path))
    assert f"缓存里有一个 5 位的键（应为 {VECTOR_KEY_LENGTH} 位）" in str(excinfo.value)


def test_cache_load_rejects_entry_dimension_mismatch(tmp_path: Path) -> None:
    """某条向量维度与声明不符 → 拒读（文件被改过或写到一半）."""
    path = _write(tmp_path, _payload({FP_A: [1, 2]}, count=1))
    with pytest.raises(ManifestError) as excinfo:
        _cache().load(str(path))
    message = str(excinfo.value)
    assert f"缓存条目 {FP_A} 是 2 维" in message
    assert f"与声明的 {DIMENSION} 维不符" in message
    assert "文件可能被改过或写到一半" in message


def test_cache_load_rejects_self_contradicting_count(tmp_path: Path) -> None:
    """声明的 count 与实际条数不符 → 拒读（文件自相矛盾）."""
    path = _write(tmp_path, _payload({FP_A: [1, 2, 3]}, count=99))
    with pytest.raises(ManifestError) as excinfo:
        _cache().load(str(path))
    assert "缓存文件自相矛盾：声明的 count=99，实际有 1 条" in str(excinfo.value)


def test_cache_load_accepts_consistent_payload(tmp_path: Path) -> None:
    """合法文件（count 一致）能读回，返回条数即条目数."""
    path = _write(tmp_path, _payload({FP_A: [1, 2, 3]}, count=1))
    cache = _cache()
    assert cache.load(str(path)) == 1
    assert cache.size == 1
    assert cache.path == str(path)


def test_cache_load_tolerates_missing_count_field(tmp_path: Path) -> None:
    """没有 count 字段是合法输入（只是少了一条自检，不是坏文件）."""
    body = _payload({FP_A: [1, 2, 3]})
    del body["count"]
    path = _write(tmp_path, body)
    assert _cache().load(str(path)) == 1


def test_cache_load_without_declared_dimension_accepts_any_length(tmp_path: Path) -> None:
    """dimension=0 的缓存读回时不校验条目长度（"不限"两侧都要成立）."""
    body = _payload({FP_A: [1, 2]}, dimension=0, count=1)
    path = _write(tmp_path, body)
    cache = _cache(dimension=0)
    assert cache.load(str(path)) == 1
    assert cache.size == 1


# --------------------------------------------------------------------------- #
# EmbeddingCache：一行摘要
# --------------------------------------------------------------------------- #


def test_cache_describe_reports_memory_and_disk(tmp_path: Path) -> None:
    """没落盘时显示"（内存）"，落盘后显示真实路径，且计数是当时的值."""
    cache = _cache()
    cache.put("a", [1, 2, 3])
    cache.get("a")
    assert cache.describe() == (
        f"缓存 {PROVIDER}/{MODEL} ({DIMENSION}d) | 1 条 | 命中 1 / 未命中 0 | （内存）"
    )

    target = tmp_path / "cache.json"
    cache.persist(str(target))
    line = cache.describe()
    assert str(target) in line
    assert "（内存）" not in line
