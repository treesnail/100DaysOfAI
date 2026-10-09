"""``indexing.versioning`` 的测试：登记、血缘、回滚与两版差集.

本文件的头号用例是规格 B.3 第 1 条的那个反例：

```text
v1 ─► v2（采纳） ─► v3（构建失败，从未采纳）
                     按时间倒序，回滚会回到 v3 —— 一个从没生效过的版本
                     按血缘回溯，回滚回到 v1
```

按"登记顺序上的上一版"实现的 ``rollback`` 会在这里变红——这正是本文件存在的理由。
其余用例覆盖登记幂等、血缘断裂与成环、``persist`` / ``load`` 往返
（含 ``current`` 指针）以及 ``diff`` 的四个分组。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smart_research_agent.indexing.errors import ManifestError, VersionError
from smart_research_agent.indexing.manifest import MANIFEST_HISTORY_FILE, build_manifest
from smart_research_agent.indexing.types import EmbeddingIdentity, IndexEntry, IndexManifest
from smart_research_agent.indexing.versioning import CURRENT_FILE, IndexVersionStore

#: 版本表不关心编码器是谁，只要三个字段稳定（它只参与版本号的计算）.
IDENTITY = EmbeddingIdentity(provider="MockEmbedding", model="mock-64", dimension=64)

FP_A = "0123456789abcdef"
FP_B = "fedcba9876543210"
FP_C = "aabbccddeeff0011"
VK_A = "0011223344556677"
VK_B = "8899aabbccddeeff"
VK_C = "1122334455667788"

#: 一个"一定不在版本表里"的 16 位十六进制串（用于查不存在的东西）.
UNKNOWN_ID = "0" * 16


def _entry(
    record_id: str,
    *,
    fingerprint: str = FP_A,
    vector_key: str = VK_A,
) -> IndexEntry:
    """造一条条目（默认值让"只想改一个字段"的用例只写一行）."""
    return IndexEntry(
        record_id=record_id,
        fingerprint=fingerprint,
        vector_key=vector_key,
    )


def _build(
    entries: tuple[IndexEntry, ...],
    *,
    parent_version: str = "",
    metric: str = "cosine",
    backend: str = "flat",
) -> IndexManifest:
    """``build_manifest`` 的便捷包装（版本号由它算，调用方不传）."""
    return build_manifest(
        entries,
        identity=IDENTITY,
        metric=metric,
        backend=backend,
        parent_version=parent_version,
    )


def _three_versions() -> tuple[IndexManifest, IndexManifest, IndexManifest]:
    """v1 → v2 → v3，内容逐版扩充（因此三个版本号互不相同）.

    血缘是显式连起来的：只有 v2 声明了 ``parent_version=v1``，
    v3 声明了 ``parent_version=v2``，``rollback`` 才有"上一版"可以走。
    """
    v1 = _build((_entry("a"), _entry("b")))
    v2 = _build((_entry("a"), _entry("b", fingerprint=FP_B)), parent_version=v1.version_id)
    v3 = _build(
        (
            _entry("a"),
            _entry("b", fingerprint=FP_B),
            _entry("c", fingerprint=FP_C, vector_key=VK_C),
        ),
        parent_version=v2.version_id,
    )
    return v1, v2, v3


def _store_with(*manifests: IndexManifest) -> IndexVersionStore:
    """登记给定的几版，返回版本表（登记顺序 = 传入顺序）."""
    store = IndexVersionStore()
    for manifest in manifests:
        store.register(manifest)
    return store


# --------------------------------------------------------------------------- #
# 登记与查询
# --------------------------------------------------------------------------- #


def test_register_history_latest_and_get() -> None:
    """三版登记后：history 按登记顺序、latest 是第三版、get 能取回同一对象."""
    v1, v2, v3 = _three_versions()
    store = _store_with(v1, v2, v3)

    assert len(store) == 3
    assert [item.version_id for item in store.history()] == [
        v1.version_id,
        v2.version_id,
        v3.version_id,
    ]
    assert store.latest() is not None
    assert store.latest().version_id == v3.version_id
    assert store.get(v2.version_id) is v2
    assert store.get(UNKNOWN_ID) is None


def test_register_is_idempotent_for_the_same_version() -> None:
    """同一份内容重新算一遍 → 同一个版本号；再登记一次不产生第二条历史."""
    v1, _, _ = _three_versions()
    rebuilt = _build((_entry("a"), _entry("b")))
    assert rebuilt.version_id == v1.version_id

    store = _store_with(v1)
    assert store.register(rebuilt) == v1.version_id
    assert len(store) == 1
    assert len(store.history()) == 1


def test_register_rejects_manifest_whose_version_id_does_not_match() -> None:
    """版本号与内容不符的清单不许进表（否则它会被当成"上一版"）."""
    v1, _, _ = _three_versions()
    tampered = IndexManifest(
        version_id=FP_B,  # 形状合法，但与内容对不上
        identity=v1.identity,
        metric=v1.metric,
        backend=v1.backend,
        entries=v1.entries,
    )
    store = IndexVersionStore()
    with pytest.raises(ManifestError) as excinfo:
        store.register(tampered)
    assert "version_id 与内容不一致" in str(excinfo.value)
    assert len(store) == 0


def test_adopt_rejects_unknown_version_and_keeps_current() -> None:
    """采纳一个不存在的版本要报错，且不能把 current 改坏."""
    store = _store_with(*_three_versions())
    with pytest.raises(VersionError) as excinfo:
        store.adopt(UNKNOWN_ID)
    assert "不在版本表里" in str(excinfo.value)
    assert store.current == ""


# --------------------------------------------------------------------------- #
# 血缘
# --------------------------------------------------------------------------- #


def test_lineage_returns_old_to_new_and_needs_a_start() -> None:
    """血缘是"旧 → 新"的一条链；没有 current 时不能用"从当前回溯"这条路."""
    v1, v2, v3 = _three_versions()
    store = _store_with(v1, v2, v3)

    assert [item.version_id for item in store.lineage(v3.version_id)] == [
        v1.version_id,
        v2.version_id,
        v3.version_id,
    ]
    with pytest.raises(VersionError) as excinfo:
        store.lineage()
    assert "还没有采纳任何版本" in str(excinfo.value)

    store.adopt(v2.version_id)
    assert [item.version_id for item in store.lineage()] == [v1.version_id, v2.version_id]


def test_lineage_rejects_broken_and_unknown_parents() -> None:
    """血缘断裂与版本不在表里都是 ``VersionError``（不许"能走多远走多远"）."""
    v1, v2, v3 = _three_versions()
    store = _store_with(v1, v2)

    # 真正的"断裂"：v2 声明了 parent=v1，但 v1 不在表里
    lonely = _store_with(v2)
    with pytest.raises(VersionError) as excinfo:
        lonely.lineage(v2.version_id)
    assert "断裂" in str(excinfo.value)
    assert v1.version_id in str(excinfo.value)

    with pytest.raises(VersionError) as unknown:
        store.lineage(UNKNOWN_ID)
    assert "不在版本表里" in str(unknown.value)

    with pytest.raises(VersionError) as absent:
        store.lineage(v3.version_id)  # v3 根本没登记
    assert "断裂" in str(absent.value)
    assert v3.version_id in str(absent.value)


def test_lineage_detects_a_cycle_in_a_crafted_history(tmp_path: Path) -> None:
    """成环的血缘会让"上一版"有无穷多个答案——读进来之后必须当场报错.

    环只能从磁盘上来：``register`` 不会让一版覆盖另一版（幂等），
    但一份被手工拼过的 JSONL 完全可以让两个版本的 ``parent_version`` 互相指向。
    这正是本模块坚持"读回来的每一行都要重算版本号"的原因——
    能通过校验的内容也可能组成一条走不通的血缘。
    """
    root = _build((_entry("a"),))
    child = _build((_entry("a"), _entry("b")), parent_version=root.version_id)
    # 同一份内容、声明另一个父版本：parent_version 不进版本号，因此这一步是合法的
    looped = IndexManifest(
        version_id=root.version_id,
        identity=root.identity,
        metric=root.metric,
        backend=root.backend,
        entries=root.entries,
        parent_version=child.version_id,
    )
    assert looped.verify_version_id() == root.version_id

    directory = tmp_path / "index"
    directory.mkdir()
    lines = [
        json.dumps(looped.to_dict(), ensure_ascii=False),
        json.dumps(child.to_dict(), ensure_ascii=False),
    ]
    (directory / MANIFEST_HISTORY_FILE).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    store = IndexVersionStore()
    assert store.load(str(directory)) == 2
    with pytest.raises(VersionError) as excinfo:
        store.lineage(root.version_id)
    assert "环" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# 回滚（本文件的重点）
# --------------------------------------------------------------------------- #


def test_rollback_follows_lineage_not_time_order() -> None:
    """规格 B.3 第 1 条的反例：v3 未采纳时，回滚必须回到 v1，而不是 v3.

    按"登记顺序的上一版"实现的 ``rollback`` 会返回 v3 —— 一个从没生效过的
    版本（它之所以没被采纳，通常正是因为构建失败），
    而在报告里只会看到"当前版本变成了 v3"。
    """
    v1, v2, v3 = _three_versions()
    store = _store_with(v1, v2, v3)
    store.adopt(v2.version_id)  # v3 构建失败，从未采纳

    assert store.latest() is not None
    assert store.latest().version_id == v3.version_id  # 时间上最新的是 v3
    assert store.rollback(steps=1) == v1.version_id  # 血缘上的"上一版"是 v1
    assert store.current == v1.version_id
    assert store.current != v3.version_id


def test_adopt_then_rollback_walks_back_step_by_step() -> None:
    """采纳最新版之后，``steps=1`` 一次退一版，且 current 跟着变."""
    v1, v2, v3 = _three_versions()
    store = _store_with(v1, v2, v3)
    assert store.adopt(v3.version_id) == v3.version_id

    assert store.rollback() == v2.version_id
    assert store.current == v2.version_id
    assert store.rollback(steps=1) == v1.version_id
    assert store.current == v1.version_id


def test_rollback_rejects_zero_steps_and_missing_current() -> None:
    """``steps < 1`` 与"还没采纳过任何版本"是两种不同的调用错误."""
    v1, v2, _ = _three_versions()
    store = _store_with(v1, v2)

    with pytest.raises(VersionError) as excinfo:
        store.rollback(steps=0)
    assert "steps 必须 >= 1" in str(excinfo.value)

    with pytest.raises(VersionError) as no_current:
        store.rollback()
    assert "还没有采纳任何版本" in str(no_current.value)


def test_rollback_reports_current_version_and_available_steps() -> None:
    """血缘不够时，消息里必须给出"当前版本与可用步数"（否则不知道该退几步）."""
    v1, v2, _ = _three_versions()
    store = _store_with(v1, v2)
    store.adopt(v2.version_id)

    with pytest.raises(VersionError) as excinfo:
        store.rollback(steps=2)
    message = str(excinfo.value)
    assert v2.version_id in message
    assert "最多只能回退 1 步" in message
    assert store.current == v2.version_id  # 失败的调用不改状态


# --------------------------------------------------------------------------- #
# 差集
# --------------------------------------------------------------------------- #


def test_diff_reports_each_group_with_exactly_its_ids() -> None:
    """两版之间：一条没变、一条内容变了、一条新增、一条删除 —— 四组各含一条."""
    left = _build((_entry("a"), _entry("b"), _entry("d", fingerprint=FP_C, vector_key=VK_C)))
    right = _build(
        (
            _entry("a"),
            _entry("b", fingerprint=FP_B),
            _entry("c", fingerprint=FP_C, vector_key=VK_C),
        ),
        parent_version=left.version_id,
    )
    store = _store_with(left, right)

    plan = store.diff(left.version_id, right.version_id)
    assert plan.added == ("c",)
    assert plan.updated == ("b",)
    assert plan.removed == ("d",)
    assert plan.unchanged == ("a",)
    assert plan.reason == f"{left.version_id} → {right.version_id}"
    assert plan.total == 4
    assert plan.reuse_ratio == 0.25
    assert plan.to_encode == ("b", "c")


def test_diff_treats_vector_key_change_as_update() -> None:
    """``vector_key`` 变（编码器身份或编码文本变了）也算"更新"——与 planner 同口径."""
    left = _build((_entry("a"),))
    right = _build((_entry("a", vector_key=VK_B),), parent_version=left.version_id)
    store = _store_with(left, right)

    plan = store.diff(left.version_id, right.version_id)
    assert plan.updated == ("a",)
    assert plan.unchanged == ()
    assert plan.added == ()
    assert plan.removed == ()


def test_diff_rejects_unknown_versions() -> None:
    """拿一个不在表里的版本号做差集 → ``VersionError``（并列出可用的版本号）."""
    v1, _, _ = _three_versions()
    store = _store_with(v1)
    with pytest.raises(VersionError) as excinfo:
        store.diff(v1.version_id, UNKNOWN_ID)
    assert "不在版本表里" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# 落盘与读回
# --------------------------------------------------------------------------- #


def test_persist_and_load_round_trip_including_current(tmp_path: Path) -> None:
    """历史写成 JSONL（一行一版）+ ``current.txt`` 指针，读回来一字不差."""
    v1, v2, v3 = _three_versions()
    store = _store_with(v1, v2, v3)
    store.adopt(v2.version_id)

    directory = store.persist(str(tmp_path / "index"))
    assert Path(directory) == tmp_path / "index"
    assert store.path == str(tmp_path / "index")
    lines = (tmp_path / "index" / MANIFEST_HISTORY_FILE).read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(lines) == 3  # 一版一行
    pointer = (tmp_path / "index" / CURRENT_FILE).read_text(encoding="utf-8")
    assert pointer == v2.version_id

    restored = IndexVersionStore(path=str(tmp_path / "index"))
    assert restored.current == ""  # 构造时不去读盘（见模块 docstring 的取舍表）
    assert restored.latest() is None
    assert restored.load() == 3

    assert [item.version_id for item in restored.history()] == [
        v1.version_id,
        v2.version_id,
        v3.version_id,
    ]
    assert restored.current == v2.version_id
    assert restored.latest().version_id == v3.version_id
    assert restored.rollback() == v1.version_id  # 读回来的血缘仍然可用


def test_persist_uses_constructor_path_and_load_accepts_history_file(tmp_path: Path) -> None:
    """``persist()`` 用构造时给的目录；``load`` 也认历史文件本身的路径."""
    v1, _, _ = _three_versions()
    directory = tmp_path / "index"
    store = IndexVersionStore(path=str(directory))
    store.register(v1)
    assert store.persist() == str(directory)
    assert (directory / MANIFEST_HISTORY_FILE).exists()

    other = IndexVersionStore()
    assert other.load(str(directory / MANIFEST_HISTORY_FILE)) == 1
    assert other.latest() is not None
    assert other.latest().version_id == v1.version_id


def test_load_rejects_missing_history_file(tmp_path: Path) -> None:
    """一版都没构建过时历史文件不存在——那与"路径写错了"要区分开."""
    with pytest.raises(ManifestError) as excinfo:
        IndexVersionStore().load(str(tmp_path / "empty"))
    assert "版本历史文件不存在" in str(excinfo.value)


def test_load_rejects_unknown_manifest_format(tmp_path: Path) -> None:
    """某行的 ``manifest_version`` 不匹配 → 拒读整份历史."""
    directory = tmp_path / "index"
    directory.mkdir()
    payload = _build((_entry("a"),)).to_dict()
    payload["manifest_version"] = 99
    (directory / MANIFEST_HISTORY_FILE).write_text(
        json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with pytest.raises(ManifestError) as excinfo:
        IndexVersionStore().load(str(directory))
    assert "清单格式版本是 99" in str(excinfo.value)


def test_load_rejects_broken_json_line_and_pointer_to_unknown_version(
    tmp_path: Path,
) -> None:
    """半截的一行与指向不存在版本的指针，都不能被无声地接受."""
    directory = tmp_path / "index"
    directory.mkdir()
    (directory / MANIFEST_HISTORY_FILE).write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ManifestError) as excinfo:
        IndexVersionStore().load(str(directory))
    assert "第 1 行不是合法 JSON" in str(excinfo.value)

    v1 = _build((_entry("a"),))
    (directory / MANIFEST_HISTORY_FILE).write_text(
        json.dumps(v1.to_dict(), ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (directory / CURRENT_FILE).write_text(UNKNOWN_ID, encoding="utf-8")
    with pytest.raises(ManifestError) as pointer:
        IndexVersionStore().load(str(directory))
    assert "current 指针指向" in str(pointer.value)


def test_load_skips_blank_lines_and_rejects_empty_path(tmp_path: Path) -> None:
    """空行是"上一次追加留下的痕迹"，跳过它；空路径则是把历史写到进程当前目录."""
    v1, v2, _ = _three_versions()
    directory = tmp_path / "index"
    directory.mkdir()
    body = "\n".join(
        [
            "",
            json.dumps(v1.to_dict(), ensure_ascii=False),
            "",
            json.dumps(v2.to_dict(), ensure_ascii=False),
            "",
        ]
    )
    (directory / MANIFEST_HISTORY_FILE).write_text(body, encoding="utf-8")
    assert IndexVersionStore().load(str(directory)) == 2

    with pytest.raises(ManifestError) as excinfo:
        IndexVersionStore().persist("  ")
    assert "版本历史需要一个目录" in str(excinfo.value)
    with pytest.raises(ManifestError) as loaded:
        IndexVersionStore().load("")
    assert "版本历史需要一个目录" in str(loaded.value)


def test_latest_is_none_on_empty_store_even_with_a_path(tmp_path: Path) -> None:
    """没有历史是合法状态（第一次构建之前就是这样），不是异常."""
    assert IndexVersionStore().latest() is None

    store = IndexVersionStore(path=str(tmp_path / "index"))
    assert store.latest() is None
    assert store.current == ""
    assert store.history() == []
    assert len(store) == 0
