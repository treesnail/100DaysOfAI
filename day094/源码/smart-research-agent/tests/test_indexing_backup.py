"""day065 索引备份与恢复测试（M6-D4）：快照的四个动作.

本文件守的是 ``backup.py`` 存在的那个理由：**"重建把库写坏了"要能回去**。
环绕这个理由的四组断言：

```text
create   真拷贝（逐字节相同 + 记录齐备）——"备份"不能只是一行日志
id       只由 (version_id, backend, 序号) 决定——时间戳不进 id
prune    保留份数有上限，且裁剪只删目录不改账本
restore  先校验后动手，且**不动**目标目录里已有的其它文件
```

另外两条容易被写错的边界：``load`` 的返回值是"**实际可用**份数"而不是记录条数；
``read_manifest`` 只读不动——"我想看看那一版是什么，但不想动现在的库"。
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from smart_research_agent.config import settings
from smart_research_agent.indexing import backup as backup_module
from smart_research_agent.indexing.backup import (
    BACKUP_INDEX_FILE,
    BACKUP_VERSION,
    DEFAULT_KEEP,
    BackupRecord,
    IndexBackupStore,
    utc_now_iso,
)
from smart_research_agent.indexing.errors import BackupError, ManifestError
from smart_research_agent.indexing.types import (
    EmbeddingIdentity,
    IndexEntry,
    IndexManifest,
    entries_digest,
    index_version_id,
)

FIXED_TIME = "2026-01-02T03:04:05+00:00"
MANIFEST_NAME = "manifest.json"


# --------------------------------------------------------------------------- #
# 构造工具（清单一律用契约文件里的 to_dict/from_dict/verify_version_id 那条路）
# --------------------------------------------------------------------------- #


def fixed_clock() -> str:
    """固定时钟：让 ``created_at`` 可预期，不依赖真实时间."""
    return FIXED_TIME


def make_clock(*values: str) -> Callable[[], str]:
    """按顺序吐值的时钟（用于制造"两次创建时间不同"的场景）."""
    iterator = iter(values)
    return lambda: next(iterator)


def entry(
    record_id: str = "c1",
    *,
    fingerprint: str = "0" * 16,
    vector_key: str = "1" * 16,
    token_count: int = 0,
) -> IndexEntry:
    """造一条清单条目（两个 16 位十六进制字段是契约要求的形状）."""
    return IndexEntry(
        record_id=record_id,
        fingerprint=fingerprint,
        vector_key=vector_key,
        token_count=token_count,
    )


def make_manifest(
    entries: tuple[IndexEntry, ...] = (),
    *,
    provider: str = "mock",
    model: str = "demo-ngram-3",
    dimension: int = 4,
    metric: str = "cosine",
    backend: str = "flat",
    created_at: str = "",
) -> IndexManifest:
    """按契约算法造一份自洽的清单（version_id 由内容重算，而不是随手编）."""
    identity = EmbeddingIdentity(provider=provider, model=model, dimension=dimension)
    ordered = tuple(entries)
    return IndexManifest(
        version_id=index_version_id(
            identity_key=identity.key,
            metric=metric,
            backend=backend,
            digest=entries_digest(ordered),
        ),
        identity=identity,
        metric=metric,
        backend=backend,
        entries=ordered,
        created_at=created_at,
    )


def write_source(directory: Path, name: str, payload: bytes) -> Path:
    """在指定目录里写一个源文件并返回路径."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(payload)
    return path


def snapshot(root: Path) -> dict[str, tuple[int, int]]:
    """目录树的快照（相对路径 → 字节数 + mtime）——用来断言"没被动过"."""
    return {
        str(item.relative_to(root)): (item.stat().st_size, item.stat().st_mtime_ns)
        for item in sorted(root.rglob("*"))
    }


def index_lines(path: Path) -> list[dict[str, object]]:
    """读账本并逐行解析（``load`` 之外的另一双眼睛）."""
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
# 常量与默认时钟
# --------------------------------------------------------------------------- #


def test_constants_are_frozen() -> None:
    """三个常量是契约（装配阶段的脚本按它们找文件与算缺省值）。"""
    assert BACKUP_VERSION == 1
    assert BACKUP_INDEX_FILE == "backups.jsonl"
    assert DEFAULT_KEEP == 3


def test_utc_now_iso_is_utc_and_second_precision() -> None:
    """默认时钟给出 UTC、秒级精度的 ISO 串（测试一律注入固定时钟）。"""
    stamp = utc_now_iso()
    parsed = datetime.fromisoformat(stamp)
    assert parsed.utcoffset() == timedelta(0)
    assert parsed.microsecond == 0
    assert len(stamp) == len("2026-01-02T03:04:05+00:00")


# --------------------------------------------------------------------------- #
# create：真拷贝
# --------------------------------------------------------------------------- #


def test_create_copies_files_byte_for_byte(tmp_path: Path) -> None:
    """``create`` 必须**真拷贝**：备份目录里逐字节相同，且 ``size_bytes`` 量得准."""
    base = tmp_path / "backups"
    payload_a = bytes(range(256))
    payload_b = "索引数据".encode()
    source_a = write_source(tmp_path / "store", "vectors.json", payload_a)
    source_b = write_source(tmp_path / "store", "raw.bin", payload_b)

    store = IndexBackupStore(path=str(base), clock=fixed_clock)
    record = store.create(
        source_files=[str(source_a), str(source_b)],
        manifest=make_manifest(),
        reason="全量重建前",
    )

    backup_dir = base / record.backup_id
    assert (backup_dir / "vectors.json").read_bytes() == payload_a
    assert (backup_dir / "raw.bin").read_bytes() == payload_b
    assert record.files == ("raw.bin", "vectors.json")
    assert record.size_bytes == sum(
        item.stat().st_size for item in backup_dir.iterdir() if item.is_file()
    )
    assert record.size_bytes == (
        len(payload_a) + len(payload_b) + (backup_dir / MANIFEST_NAME).stat().st_size
    )


def test_create_records_the_manifest_fields(tmp_path: Path) -> None:
    """记录填实：version_id/backend/metric/dimension/created_at/reason 全部来自清单与时钟."""
    manifest = make_manifest((entry("c1"),), metric="dot", backend="flat", dimension=7)
    store = IndexBackupStore(path=str(tmp_path / "backups"), clock=fixed_clock)

    record = store.create(source_files=[], manifest=manifest, reason="换模型前")

    assert record.version_id == manifest.version_id
    assert record.backend == "flat"
    assert record.metric == "dot"
    assert record.dimension == 7
    assert record.created_at == FIXED_TIME
    assert record.reason == "换模型前"
    assert record.backup_version == BACKUP_VERSION
    assert record.backup_id == f"{manifest.version_id}-flat-0001"


def test_create_without_source_files_keeps_manifest_only(tmp_path: Path) -> None:
    """没有源文件也是合法备份（只保清单）：``files`` 为空、目录里只剩 manifest.json."""
    base = tmp_path / "backups"
    store = IndexBackupStore(path=str(base), clock=fixed_clock)

    record = store.create(source_files=[], manifest=make_manifest((entry("c1"),)))

    backup_dir = base / record.backup_id
    assert record.files == ()
    assert sorted(item.name for item in backup_dir.iterdir()) == [MANIFEST_NAME]
    assert record.size_bytes == (backup_dir / MANIFEST_NAME).stat().st_size


def test_backup_id_sanitises_the_backend_name(tmp_path: Path) -> None:
    """后端名里的空格等字符被清洗成 ``-``：备份 id 要能直接当目录名用."""
    base = tmp_path / "backups"
    manifest = make_manifest(backend="flat store")
    record = IndexBackupStore(path=str(base), clock=fixed_clock).create(
        source_files=[], manifest=manifest
    )

    assert record.backup_id == f"{manifest.version_id}-flat-store-0001"
    assert (base / record.backup_id).is_dir()


def test_create_missing_source_leaves_no_partial_backup(tmp_path: Path) -> None:
    """缺源文件 → ``BackupError``，而且**不留半成品目录**、不写账本.

    先全部校验再拷：留下一个"看起来有数据"的目录，比直接失败危险得多。
    """
    base = tmp_path / "backups"
    good = write_source(tmp_path / "store", "vectors.json", b"x")
    store = IndexBackupStore(path=str(base), clock=fixed_clock)

    with pytest.raises(BackupError, match="备份源文件不存在"):
        store.create(
            source_files=[str(good), str(tmp_path / "store" / "missing.json")],
            manifest=make_manifest(),
        )

    assert store.list() == []
    assert [item for item in base.iterdir() if item.is_dir()] == []
    assert not (base / BACKUP_INDEX_FILE).exists()


def test_create_rejects_a_directory_as_source(tmp_path: Path) -> None:
    """目录不能当源文件（展开成文件列表是调用方的事）。"""
    directory = tmp_path / "store"
    directory.mkdir()
    (directory / "vectors.json").write_bytes(b"1")
    store = IndexBackupStore(path=str(tmp_path / "backups"), clock=fixed_clock)

    with pytest.raises(BackupError, match="不是文件"):
        store.create(source_files=[str(directory)], manifest=make_manifest())


def test_create_rejects_duplicate_basenames(tmp_path: Path) -> None:
    """两个源文件同名 → 拒绝（备份目录是平铺的，同名会互相覆盖）."""
    first = write_source(tmp_path / "a", "vectors.json", b"1")
    second = write_source(tmp_path / "b", "vectors.json", b"2")
    store = IndexBackupStore(path=str(tmp_path / "backups"), clock=fixed_clock)

    with pytest.raises(BackupError, match="同名"):
        store.create(source_files=[str(first), str(second)], manifest=make_manifest())


def test_create_rejects_a_source_named_manifest(tmp_path: Path) -> None:
    """源文件不能叫 ``manifest.json``：那个位置留给清单本身."""
    source = write_source(tmp_path / "store", MANIFEST_NAME, b"{}")
    store = IndexBackupStore(path=str(tmp_path / "backups"), clock=fixed_clock)

    with pytest.raises(BackupError, match="manifest.json"):
        store.create(source_files=[str(source)], manifest=make_manifest())


def test_create_rejects_a_bare_string_as_source_files(tmp_path: Path) -> None:
    """``source_files`` 传单个字符串会被拒（它会被逐字符当成路径，很晚才报错）."""
    source = write_source(tmp_path / "store", "vectors.json", b"1")
    store = IndexBackupStore(path=str(tmp_path / "backups"), clock=fixed_clock)

    with pytest.raises(BackupError, match="路径序列"):
        store.create(source_files=str(source), manifest=make_manifest())  # type: ignore[arg-type]


def test_create_verifies_the_manifest_before_copying(tmp_path: Path) -> None:
    """拿一份被改过的清单去备份会被拦下：否则等于把"错误的上一版"固化下来."""
    base = tmp_path / "backups"
    manifest = make_manifest((entry("c1"),))
    tampered = dataclasses.replace(manifest, version_id="0" * 16)
    store = IndexBackupStore(path=str(base), clock=fixed_clock)

    with pytest.raises(ManifestError, match="不一致"):
        store.create(source_files=[], manifest=tampered)

    assert [item for item in base.iterdir() if item.is_dir()] == []
    assert store.list() == []


def test_create_requires_a_path(tmp_path: Path) -> None:
    """没给 path 就不能建备份（备份必须落在库之外的目录上）."""
    store = IndexBackupStore(clock=fixed_clock)

    with pytest.raises(BackupError, match="没有位置"):
        store.create(source_files=[], manifest=make_manifest())


def test_create_appends_one_line_per_backup(tmp_path: Path) -> None:
    """每次 ``create`` 追加**一行**，行里带格式版本（账本自证格式）."""
    base = tmp_path / "backups"
    store = IndexBackupStore(path=str(base), clock=fixed_clock)
    manifest = make_manifest((entry("c1"),))

    first = store.create(source_files=[], manifest=manifest)
    second = store.create(source_files=[], manifest=manifest)

    lines = index_lines(base / BACKUP_INDEX_FILE)
    assert len(lines) == 2
    assert [line["backup_id"] for line in lines] == [first.backup_id, second.backup_id]
    assert all(line["backup_version"] == BACKUP_VERSION for line in lines)


# --------------------------------------------------------------------------- #
# backup_id：时间戳不进 id
# --------------------------------------------------------------------------- #


def test_same_version_twice_gets_two_ids_and_one_version_id(tmp_path: Path) -> None:
    """同一版备份两次 → **id 不同、version_id 相同**（存储位置 ≠ 身份）.

    这条断言正是"时间戳不进 id"的代价与目的：id 靠序号区分，
    而"这两个备份其实是同一版"这句话仍然成立（两个 version_id 相等）。
    """
    base = tmp_path / "backups"
    store = IndexBackupStore(path=str(base), clock=make_clock("t1", "t2"))
    manifest = make_manifest((entry("c1"),))

    first = store.create(source_files=[], manifest=manifest)
    second = store.create(source_files=[], manifest=manifest)

    assert first.backup_id != second.backup_id
    assert first.version_id == second.version_id == manifest.version_id
    assert first.backup_id.endswith("-0001")
    assert second.backup_id.endswith("-0002")
    assert first.backup_id.rsplit("-", 1)[0] == second.backup_id.rsplit("-", 1)[0]
    assert (first.created_at, second.created_at) == ("t1", "t2")
    assert "t1" not in first.backup_id


def test_sequence_does_not_reuse_freed_slots(tmp_path: Path) -> None:
    """裁剪之后再创建不会重号（否则下一次创建会原地覆盖一份还在的快照）."""
    base = tmp_path / "backups"
    store = IndexBackupStore(path=str(base), keep=2, clock=fixed_clock)
    manifest = make_manifest((entry("c1"),))

    store.create(source_files=[], manifest=manifest)
    store.create(source_files=[], manifest=manifest)
    third = store.create(source_files=[], manifest=manifest)
    fourth = store.create(source_files=[], manifest=manifest)

    assert third.backup_id.endswith("-0003")
    assert fourth.backup_id.endswith("-0004")
    assert len({record.backup_id for record in store.list()}) == 2


# --------------------------------------------------------------------------- #
# keep 与 prune
# --------------------------------------------------------------------------- #


def test_keep_defaults_to_settings_and_explicit_value_wins(tmp_path: Path) -> None:
    """``keep`` 缺省取 ``settings.indexing_backups_keep``，显式传值覆盖它."""
    default_store = IndexBackupStore(path=str(tmp_path / "a"), clock=fixed_clock)
    assert default_store.keep == settings.indexing_backups_keep
    assert IndexBackupStore(path=str(tmp_path / "b"), keep=1, clock=fixed_clock).keep == 1


def test_keep_must_be_positive(tmp_path: Path) -> None:
    """``keep <= 0`` 在构造时就拒绝：不保留任何备份的备份机制是自相矛盾的."""
    with pytest.raises(BackupError, match="保留份数必须 >= 1"):
        IndexBackupStore(path=str(tmp_path / "a"), keep=0, clock=fixed_clock)


def test_create_auto_prunes_the_oldest(tmp_path: Path) -> None:
    """``keep=2`` 时创建 3 份：最旧的一份目录消失，但账本仍留着那一行."""
    base = tmp_path / "backups"
    store = IndexBackupStore(path=str(base), keep=2, clock=fixed_clock)
    manifest = make_manifest((entry("c1"),))

    records = [store.create(source_files=[], manifest=manifest) for _ in range(3)]

    assert [record.backup_id for record in store.list()] == [
        records[2].backup_id,
        records[1].backup_id,
    ]
    assert not (base / records[0].backup_id).is_dir()
    assert (base / records[1].backup_id).is_dir()
    assert len(index_lines(base / BACKUP_INDEX_FILE)) == 3


def test_prune_returns_deleted_ids_oldest_first(tmp_path: Path) -> None:
    """``prune(keep=...)`` 返回被删的 id（从最旧开始），并真的删掉目录."""
    base = tmp_path / "backups"
    store = IndexBackupStore(path=str(base), clock=fixed_clock)
    manifest = make_manifest((entry("c1"),))
    records = [store.create(source_files=[], manifest=manifest) for _ in range(3)]

    deleted = store.prune(keep=1)

    assert deleted == [records[0].backup_id, records[1].backup_id]
    assert [record.backup_id for record in store.list()] == [records[2].backup_id]
    assert all(not (base / backup_id).is_dir() for backup_id in deleted)
    assert store.prune() == []


def test_prune_rejects_a_non_positive_limit(tmp_path: Path) -> None:
    """显式传 ``keep=0`` 给 ``prune`` 同样被拒（那是一次不可逆的数据删除）."""
    store = IndexBackupStore(path=str(tmp_path / "backups"), clock=fixed_clock)
    store.create(source_files=[], manifest=make_manifest())

    with pytest.raises(BackupError, match="保留份数必须 >= 1"):
        store.prune(keep=0)


# --------------------------------------------------------------------------- #
# read_manifest：只看不动
# --------------------------------------------------------------------------- #


def test_read_manifest_returns_the_version_and_touches_nothing(tmp_path: Path) -> None:
    """``read_manifest`` 只读：清单能还原，且备份目录与目标目录的文件都没被动过."""
    base = tmp_path / "backups"
    target = tmp_path / "live"
    target.mkdir()
    (target / "sentinel.txt").write_text("keep me", encoding="utf-8")
    source = write_source(tmp_path / "store", "vectors.json", b"payload")
    store = IndexBackupStore(path=str(base), clock=fixed_clock)
    record = store.create(
        source_files=[str(source)], manifest=make_manifest((entry("c1"),))
    )

    before_backup = snapshot(base)
    before_target = snapshot(target)
    manifest = store.read_manifest(record.backup_id)
    after_backup = snapshot(base)
    after_target = snapshot(target)

    assert manifest.version_id == record.version_id
    assert manifest.count == 1
    assert manifest.record_ids == ("c1",)
    assert manifest.verify_version_id() == record.version_id
    assert before_backup == after_backup
    assert before_target == after_target


def test_read_manifest_rejects_an_unknown_id(tmp_path: Path) -> None:
    """账本里没有这个 id → ``BackupError``（不是 ``None``：调用方要的是清单）."""
    store = IndexBackupStore(path=str(tmp_path / "backups"), clock=fixed_clock)

    with pytest.raises(BackupError, match="没有备份"):
        store.read_manifest("deadbeefdeadbeef-flat-0009")


# --------------------------------------------------------------------------- #
# restore：先校验后动手
# --------------------------------------------------------------------------- #


def test_restore_copies_files_and_the_manifest_back(tmp_path: Path) -> None:
    """恢复到另一个目录：源文件逐字节回来，清单可读且自洽."""
    base = tmp_path / "backups"
    source = write_source(tmp_path / "store", "vectors.json", b"payload-1")
    store = IndexBackupStore(path=str(base), clock=fixed_clock)
    record = store.create(
        source_files=[str(source)], manifest=make_manifest((entry("c1"),))
    )
    target = tmp_path / "restored"

    returned = store.restore(record.backup_id, target_dir=str(target))

    assert returned == record
    assert (target / "vectors.json").read_bytes() == b"payload-1"
    payload = json.loads((target / MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest = IndexManifest.from_dict(payload)
    assert manifest.verify_version_id() == record.version_id


def test_restore_does_not_clean_the_target_directory(tmp_path: Path) -> None:
    """恢复**不删除**目标目录里已有的其它文件（避免"恢复备份顺带清库"）."""
    base = tmp_path / "backups"
    source = write_source(tmp_path / "store", "vectors.json", b"new")
    store = IndexBackupStore(path=str(base), clock=fixed_clock)
    record = store.create(source_files=[str(source)], manifest=make_manifest())
    target = tmp_path / "restored"
    target.mkdir()
    (target / "untouched.json").write_bytes(b"older")

    store.restore(record.backup_id, target_dir=str(target))

    assert (target / "untouched.json").read_bytes() == b"older"
    assert (target / "vectors.json").read_bytes() == b"new"


def test_restore_rejects_a_tampered_manifest(tmp_path: Path) -> None:
    """清单被改过一个字符 → ``BackupError``，且**目标目录一点都不动**."""
    base = tmp_path / "backups"
    store = IndexBackupStore(path=str(base), clock=fixed_clock)
    record = store.create(source_files=[], manifest=make_manifest((entry("c1"),)))
    manifest_path = base / record.backup_id / MANIFEST_NAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["version_id"] = "0" * 16 if payload["version_id"] != "0" * 16 else "1" * 16
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    target = tmp_path / "restored"

    with pytest.raises(BackupError, match="不一致"):
        store.read_manifest(record.backup_id)
    with pytest.raises(BackupError, match="不一致"):
        store.restore(record.backup_id, target_dir=str(target))

    assert not target.exists()


def test_restore_rejects_an_unknown_backup_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """快照格式版本与代码不符 → 拒恢复（用旧格式恢复出一半比空库更难发现）."""
    base = tmp_path / "backups"
    store = IndexBackupStore(path=str(base), clock=fixed_clock)
    record = store.create(source_files=[], manifest=make_manifest())
    assert record.backup_version == BACKUP_VERSION

    monkeypatch.setattr(backup_module, "BACKUP_VERSION", BACKUP_VERSION + 1)
    with pytest.raises(BackupError, match="格式版本"):
        store.restore(record.backup_id, target_dir=str(tmp_path / "restored"))


def test_restore_rejects_an_unknown_id(tmp_path: Path) -> None:
    """账本里没有这个 id → ``BackupError``."""
    store = IndexBackupStore(path=str(tmp_path / "backups"), clock=fixed_clock)

    with pytest.raises(BackupError, match="没有备份"):
        store.restore("deadbeefdeadbeef-flat-0009", target_dir=str(tmp_path / "restored"))


def test_restore_reports_a_vanished_directory(tmp_path: Path) -> None:
    """记录在但目录被删 → ``BackupError``（账本里有记录不等于目录还在）."""
    base = tmp_path / "backups"
    store = IndexBackupStore(path=str(base), clock=fixed_clock)
    record = store.create(source_files=[], manifest=make_manifest())
    shutil.rmtree(base / record.backup_id)

    with pytest.raises(BackupError, match="目录不存在"):
        store.restore(record.backup_id, target_dir=str(tmp_path / "restored"))


# --------------------------------------------------------------------------- #
# persist / load：账本与"实际可用份数"
# --------------------------------------------------------------------------- #


def test_persist_rewrites_the_index_file(tmp_path: Path) -> None:
    """``persist`` 重写账本（把已裁剪的记录从文件里去掉），返回实际路径."""
    base = tmp_path / "backups"
    store = IndexBackupStore(path=str(base), clock=fixed_clock)
    manifest = make_manifest((entry("c1"),))
    records = [store.create(source_files=[], manifest=manifest) for _ in range(3)]
    assert len(index_lines(base / BACKUP_INDEX_FILE)) == 3

    store.prune(keep=1)
    written = store.persist()

    assert written == str(base / BACKUP_INDEX_FILE)
    lines = index_lines(base / BACKUP_INDEX_FILE)
    assert len(lines) == 1
    assert lines[0]["backup_id"] == records[2].backup_id


def test_persist_and_load_roundtrip(tmp_path: Path) -> None:
    """``load`` 把账本读回来：顺序新 → 旧、latest/get 可用、path 被记住."""
    base = tmp_path / "backups"
    store = IndexBackupStore(path=str(base), clock=fixed_clock)
    first = store.create(
        source_files=[write_source(tmp_path / "s", "vectors.json", b"1")],
        manifest=make_manifest((entry("c1"),)),
        reason="重建前",
    )
    second = store.create(
        source_files=[],
        manifest=make_manifest((entry("c1"), entry("c2"))),
        reason="增量前",
    )

    reloaded = IndexBackupStore(clock=fixed_clock)
    available = reloaded.load(str(base))

    assert available == 2
    assert reloaded.path == str(base)
    assert [record.backup_id for record in reloaded.list()] == [
        second.backup_id,
        first.backup_id,
    ]
    assert reloaded.latest() == second
    assert reloaded.get(first.backup_id) == first
    assert reloaded.list()[0].reason == "增量前"
    assert reloaded.list()[0].files == ()


def test_load_accepts_the_index_file_path_itself(tmp_path: Path) -> None:
    """目录与 ``backups.jsonl`` 文件两种写法都能读（路径解析规则只有一条）."""
    base = tmp_path / "backups"
    store = IndexBackupStore(path=str(base), clock=fixed_clock)
    store.create(source_files=[], manifest=make_manifest())

    reloaded = IndexBackupStore(clock=fixed_clock)

    assert reloaded.load(str(base / BACKUP_INDEX_FILE)) == 1
    assert reloaded.path == str(base)


def test_load_on_an_empty_store_returns_zero(tmp_path: Path) -> None:
    """没有账本是合法状态：``load`` 返回 0、``latest`` 返回 ``None``（都不是异常）."""
    store = IndexBackupStore(path=str(tmp_path / "empty"), clock=fixed_clock)

    assert store.load() == 0
    assert store.list() == []
    assert store.latest() is None


def test_load_rejects_an_unknown_backup_version(tmp_path: Path) -> None:
    """账本里出现别的格式版本 → 拒读（逐行去看"哪几份能恢复"是更坏的选择）."""
    index = tmp_path / BACKUP_INDEX_FILE
    index.write_text(
        json.dumps(
            {
                "backup_version": BACKUP_VERSION + 1,
                "backup_id": "deadbeefdeadbeef-flat-0001",
                "version_id": "deadbeefdeadbeef",
                "backend": "flat",
                "metric": "cosine",
                "dimension": 4,
                "created_at": FIXED_TIME,
                "files": [],
                "size_bytes": 0,
                "reason": "",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    store = IndexBackupStore(clock=fixed_clock)

    with pytest.raises(BackupError, match="格式版本"):
        store.load(str(tmp_path))


def test_load_reports_only_the_available_backups(tmp_path: Path) -> None:
    """目录被删的记录仍在账本里，但 ``load`` 如实报告**实际可用**份数."""
    base = tmp_path / "backups"
    store = IndexBackupStore(path=str(base), clock=fixed_clock)
    records = [
        store.create(source_files=[], manifest=make_manifest((entry(f"c{index}"),)))
        for index in range(3)
    ]
    shutil.rmtree(base / records[1].backup_id)

    reloaded = IndexBackupStore(clock=fixed_clock)
    available = reloaded.load(str(base))

    assert available == 2
    assert len(reloaded.list()) == 3
    assert reloaded.get(records[1].backup_id) == records[1]


def test_latest_skips_a_vanished_directory(tmp_path: Path) -> None:
    """最新的那份目录没了 → ``latest`` 给下一份可用的（否则它必然 restore 失败）."""
    base = tmp_path / "backups"
    store = IndexBackupStore(path=str(base), clock=fixed_clock)
    records = [
        store.create(source_files=[], manifest=make_manifest((entry(f"c{index}"),)))
        for index in range(3)
    ]
    shutil.rmtree(base / records[2].backup_id)

    reloaded = IndexBackupStore(clock=fixed_clock)
    reloaded.load(str(base))

    assert reloaded.latest() == records[1]
    assert reloaded.get(records[2].backup_id) == records[2]


# --------------------------------------------------------------------------- #
# BackupRecord 自身
# --------------------------------------------------------------------------- #


def test_record_projection_and_summary(tmp_path: Path) -> None:
    """记录可投影为 JSON、可往返、摘要里带上位置与身份."""
    store = IndexBackupStore(path=str(tmp_path / "backups"), clock=fixed_clock)
    record = store.create(
        source_files=[], manifest=make_manifest((entry("c1"),)), reason="演示"
    )

    payload = record.to_dict()
    assert payload["backup_version"] == BACKUP_VERSION
    assert payload["files"] == []
    assert json.loads(json.dumps(payload, ensure_ascii=False))["backup_id"] == record.backup_id
    assert BackupRecord.from_dict(payload) == record

    line = record.summary_line()
    assert record.backup_id in line
    assert record.version_id in line
    assert "演示" in line


def test_record_rejects_invalid_fields() -> None:
    """三个护栏各自可触发：id 不能空、维度不能为 0、字节数不能为负."""
    common = {
        "backup_id": "x-flat-0001",
        "version_id": "a" * 16,
        "backend": "flat",
        "metric": "cosine",
        "dimension": 4,
        "created_at": FIXED_TIME,
    }
    with pytest.raises(BackupError, match="backup_id 不能为空"):
        BackupRecord(**(common | {"backup_id": "  "}))  # type: ignore[arg-type]
    with pytest.raises(BackupError, match="dimension 必须 >= 1"):
        BackupRecord(**(common | {"dimension": 0}))  # type: ignore[arg-type]
    with pytest.raises(BackupError, match="size_bytes 不能为负"):
        BackupRecord(**(common | {"size_bytes": -1}))  # type: ignore[arg-type]


def test_record_from_dict_rejects_missing_fields() -> None:
    """账本一行缺字段 → ``BackupError``（那条记录无法还原出一份备份）."""
    with pytest.raises(BackupError, match="缺少字段"):
        BackupRecord.from_dict({"backup_id": "x-flat-0001"})
