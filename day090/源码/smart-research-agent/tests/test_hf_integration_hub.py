"""``hf_integration.hub`` / ``hf_integration.errors``：文件怎么被找齐（day086 / M7-D10）."""

from __future__ import annotations

import pytest

from smart_research_agent.hf_integration import errors, hub
from smart_research_agent.hf_integration.hub import (
    DEFAULT_CACHE_DIR,
    CacheStore,
    HubResolver,
    InMemoryHub,
    blob_path,
    check_readable,
    flat_paths,
    known_offered,
    repo_folder,
    select_files,
    sha256_of,
    snapshot_root,
)
from smart_research_agent.hf_integration.types import CONFIG_FILE, VOCAB_FILE

from tests.hf_integration_samples import (
    MAIN_COMMIT,
    SECOND_COMMIT,
    build_case,
    make_hub,
    repo_files,
)

REPO = "org/tiny"


# --------------------------------------------------------------------------- 目录命名


def test_repo_folder_uses_double_dash() -> None:
    """``org/name`` → ``models--org--name``（真实的目录命名规则）."""
    assert repo_folder("org/tiny") == "models--org--tiny"
    assert repo_folder("meta-llama/Llama-3.1-8B") == "models--meta-llama--Llama-3.1-8B"


@pytest.mark.parametrize("repo_id", ["tiny", "a/b/c", "/tiny", "tiny/", ""])
def test_repo_folder_rejects_non_two_segment_ids(repo_id: str) -> None:
    """两段之外一律拒绝：目录名里的斜杠被换成双横线之后是**不可逆**的."""
    with pytest.raises(errors.ParameterError):
        repo_folder(repo_id)


def test_snapshot_root_and_blob_path_layout() -> None:
    """快照根与实体路径的两段结构（三个常量各自出现在正确的位置）."""
    root = snapshot_root("C:/cache", REPO, MAIN_COMMIT)
    assert root == f"C:/cache/models--org--tiny/snapshots/{MAIN_COMMIT}"
    assert hub.SNAPSHOT_DIR in root
    digest = sha256_of(b"abc")
    assert blob_path("C:/cache", REPO, digest) == f"C:/cache/models--org--tiny/blobs/{digest}"


def test_default_cache_dir_is_the_real_default() -> None:
    """默认缓存根就是真实的那个（``$HF_HOME/hub`` 的默认值）."""
    assert DEFAULT_CACHE_DIR == "~/.cache/huggingface/hub"


def test_sha256_is_stable_and_distinct() -> None:
    """sha256 是缓存的寻址键：同内容同名、不同内容不同名."""
    assert sha256_of(b"abc") == sha256_of(b"abc")
    assert sha256_of(b"abc") != sha256_of(b"abd")
    assert len(sha256_of(b"")) == 64


# --------------------------------------------------------------------------- 假仓库


def test_in_memory_hub_records_calls() -> None:
    """每次 ``list_files`` 都记一笔（"下载了几次"由此可被断言）."""
    fake = make_hub()
    assert fake.repo_exists(REPO) is True
    assert fake.repo_exists("org/nope") is False
    assert fake.calls == []
    files = fake.list_files(REPO, MAIN_COMMIT)
    assert set(files) == {"config.json", "vocab.json", "merges.txt", "model.safetensors"}
    assert fake.calls == [(REPO, MAIN_COMMIT)]


def test_in_memory_hub_reports_two_kinds_of_missing() -> None:
    """仓库不存在与版本不存在**分别报**（合成一句"找不到"会指错方向）."""
    fake = make_hub()
    with pytest.raises(errors.HubError, match="不存在"):
        fake.resolve_commit("org/nope", "main")
    with pytest.raises(errors.HubError, match="版本"):
        fake.resolve_commit(REPO, "v9")
    with pytest.raises(errors.HubError, match="commit"):
        fake.list_files(REPO, "deadbeef")


def test_offline_with_empty_cache_is_a_hub_error() -> None:
    """离线 + 空缓存 ⇒ HubError（**不是**参数错）."""
    resolver = HubResolver(hub=make_hub(), store=CacheStore())
    with pytest.raises(errors.HubError, match="离线"):
        resolver.resolve(REPO)
    with pytest.raises(errors.HubError):
        resolver.resolve(REPO, revision="v9")


def test_first_online_resolve_downloads_everything_once() -> None:
    """第一次联网解析：每个文件都"新下"，随后 refs / snapshots / blobs 三段都写上了."""
    resolver = HubResolver(hub=make_hub(), store=CacheStore(), cache_dir="C:/cache")
    snapshot = resolver.resolve(REPO, local_files_only=False)
    assert snapshot.downloaded_count == snapshot.file_count == 4
    assert snapshot.cached_count == 0
    assert snapshot.downloaded_bytes == snapshot.total_bytes
    assert resolver.store.snapshot_count == 1
    assert resolver.store.blob_count == 4
    assert resolver.store.describe()["refs"] == 1


def test_second_online_resolve_hits_every_file() -> None:
    """**逐文件**命中：第二次联网解析的下载量恰好为 0（这是本课的核心读数）."""
    resolver = HubResolver(hub=make_hub(), store=CacheStore())
    resolver.resolve(REPO, local_files_only=False)
    second = resolver.resolve(REPO, local_files_only=False)
    assert second.downloaded_count == 0
    assert second.cached_count == second.file_count
    assert second.downloaded_bytes == 0


def test_resolve_is_idempotent_across_modes() -> None:
    """联网一次之后，离线解析给出**同一个** commit 与 root."""
    resolver = HubResolver(hub=make_hub(), store=CacheStore(), cache_dir="C:/cache")
    online = resolver.resolve(REPO, local_files_only=False)
    offline = resolver.resolve(REPO)
    assert offline.commit == online.commit
    assert offline.root == online.root
    assert offline.local_files_only is True


def test_switching_revision_switches_snapshot_directory() -> None:
    """换 revision ⇒ 换 commit ⇒ 换快照目录（这也解释了为什么"钉住 revision"是纪律）."""
    fake = make_hub(revisions={"main": MAIN_COMMIT, "v1": SECOND_COMMIT})
    resolver = HubResolver(hub=fake, store=CacheStore(), cache_dir="C:/cache")
    main = resolver.resolve(REPO, local_files_only=False)
    tagged = resolver.resolve(REPO, revision="v1", local_files_only=False)
    assert main.commit != tagged.commit
    assert main.root != tagged.root
    assert resolver.store.snapshot_count == 2


def test_shared_content_is_stored_once() -> None:
    """两个版本内容相同 ⇒ 实体只存一份（blobs 按 sha256 寻址的直接后果）."""
    files = repo_files()
    fake = make_hub(
        revisions={"main": MAIN_COMMIT, "v1": SECOND_COMMIT},
        trees={MAIN_COMMIT: files, SECOND_COMMIT: dict(files)},
    )
    resolver = HubResolver(hub=fake, store=CacheStore())
    resolver.resolve(REPO, local_files_only=False)
    resolver.resolve(REPO, revision="v1", local_files_only=False)
    assert resolver.store.snapshot_count == 2
    assert resolver.store.blob_count == 4


def test_missing_blob_while_offline_is_a_hub_error() -> None:
    """快照表里有名字、实体被清掉了 ⇒ HubError（'名字在'与'东西在'是两件事）."""
    resolver = HubResolver(hub=make_hub(), store=CacheStore())
    resolver.resolve(REPO, local_files_only=False)
    resolver.store.blobs.clear()
    with pytest.raises(errors.HubError, match="实体"):
        resolver.resolve(REPO)


def test_allow_patterns_narrows_the_manifest() -> None:
    """``allow_patterns`` 会过滤掉文件；过滤到必需文件时如实报错."""
    resolver = HubResolver(hub=make_hub(), store=CacheStore(), cache_dir="C:/cache")
    snapshot = resolver.resolve(
        REPO, allow_patterns=("config.json", "vocab.json"), local_files_only=False
    )
    assert snapshot.file_count == 2
    with pytest.raises(errors.HubError, match="必需文件"):
        resolver.resolve(REPO, allow_patterns=("vocab.json",), local_files_only=False)


def test_empty_manifest_is_a_hub_error() -> None:
    """仓库里一个文件都没有 ⇒ HubError（空清单与"没写这个参数"是两件事）."""
    fake = InMemoryHub(commits={REPO: {"main": MAIN_COMMIT}}, trees={REPO: {MAIN_COMMIT: {}}})
    resolver = HubResolver(hub=fake, store=CacheStore())
    with pytest.raises(errors.HubError, match="一个文件都没有"):
        resolver.resolve(REPO, local_files_only=False)


@pytest.mark.parametrize(
    ("patterns", "expected"),
    [
        (None, 4),
        (("*.json",), 2),
        (("merges.txt",), 1),
        (("vocab.*",), 1),
    ],
)
def test_select_files_matching(patterns: tuple[str, ...] | None, expected: int) -> None:
    """``select_files`` 支持 ``*`` 与 ``?``；``None`` 表示全要."""
    manifest = repo_files()
    assert len(select_files(manifest, allow_patterns=patterns)) == expected


def test_select_files_empty_tuple_is_rejected() -> None:
    """空元组表示"一个都不要"，与 ``None``（全要）必须分开."""
    with pytest.raises(errors.ParameterError, match="空元组"):
        select_files(repo_files(), allow_patterns=())


# --------------------------------------------------------------------------- 读文件


def test_read_text_and_json_from_the_cache() -> None:
    """两个读入口都从 ``store`` 里取（**不碰文件系统**）."""
    case = build_case("gpt2")
    payload = case.resolver.read_json(case.snapshot, CONFIG_FILE)
    assert payload["model_type"] == "gpt2"
    assert payload["n_embd"] == 16
    assert case.resolver.read_bytes(case.snapshot, VOCAB_FILE).startswith(b"{")


def test_read_missing_path_is_a_hub_error() -> None:
    """快照里没有这条路径 ⇒ HubError（消息里带上"它有的文件"）."""
    case = build_case("gpt2")
    with pytest.raises(errors.HubError, match="没有"):
        case.resolver.read_text(case.snapshot, "nope.json")


def test_non_utf8_config_is_a_config_error() -> None:
    """非 UTF-8 的配置⇒ ConfigError（用别的编码写的文件在这里变成一次解析失败）."""
    files = repo_files()
    files[CONFIG_FILE] = b"\xff\xfe\x00"
    fake = make_hub(trees={MAIN_COMMIT: files})
    resolver = HubResolver(hub=fake, store=CacheStore())
    snapshot = resolver.resolve(REPO, local_files_only=False)
    with pytest.raises(errors.ConfigError, match="UTF-8"):
        resolver.read_text(snapshot, CONFIG_FILE)


def test_broken_json_is_a_config_error() -> None:
    """坏 JSON ⇒ ConfigError（消息里带上行号）."""
    files = repo_files()
    files[CONFIG_FILE] = b'{"model_type": "gpt2",}'
    resolver = HubResolver(hub=make_hub(trees={MAIN_COMMIT: files}), store=CacheStore())
    snapshot = resolver.resolve(REPO, local_files_only=False)
    with pytest.raises(errors.ConfigError, match="JSON"):
        resolver.read_json(snapshot, CONFIG_FILE)


def test_json_top_level_must_be_an_object() -> None:
    """顶层不是对象 ⇒ ConfigError（数组与字符串都不是一份配置）."""
    files = repo_files()
    files[CONFIG_FILE] = b"[1, 2, 3]"
    resolver = HubResolver(hub=make_hub(trees={MAIN_COMMIT: files}), store=CacheStore())
    snapshot = resolver.resolve(REPO, local_files_only=False)
    with pytest.raises(errors.ConfigError, match="对象"):
        resolver.read_json(snapshot, CONFIG_FILE)


# --------------------------------------------------------------------------- 平铺与自检


def test_flat_paths_and_known_offered() -> None:
    """平铺：四个名字全部不带内部前缀；四个都认得."""
    case = build_case("gpt2")
    names = case.snapshot.names()
    assert flat_paths(case.snapshot) == names
    assert set(known_offered(case.snapshot)) == set(names)


def test_flat_paths_flags_an_internal_name() -> None:
    """只要有一个名字落在 blobs/ 里，"平铺"这条性质就不再成立."""
    case = build_case("gpt2")
    from smart_research_agent.hf_integration.types import BLOB_DIR, CacheEntry, Snapshot

    broken = Snapshot(
        repo_id=case.snapshot.repo_id,
        revision=case.snapshot.revision,
        commit=case.snapshot.commit,
        root=case.snapshot.root,
        files=case.snapshot.files
        + (CacheEntry(path=f"{BLOB_DIR}/deadbeef", size=1, cached=True),),
    )
    assert len(flat_paths(broken)) == len(broken.files) - 1


def test_check_readable_passes_and_fails() -> None:
    """自检入口：缺哪一条就报哪一条."""
    case = build_case("gpt2")
    check_readable(case.snapshot, (CONFIG_FILE,))
    with pytest.raises(errors.ShapeError, match="缺少"):
        check_readable(case.snapshot, ("nope.json",))


def test_snapshot_entry_lookup() -> None:
    """``Snapshot.entry`` 按路径取；缺失时是 ``KeyError``（上层再翻成 HubError）."""
    case = build_case("gpt2")
    assert case.snapshot.entry(CONFIG_FILE).path == CONFIG_FILE
    with pytest.raises(KeyError):
        case.snapshot.entry("nope")


def test_cache_layout_table_is_closed() -> None:
    """缓存布局的五个角色都在表里，且描述非空."""
    from smart_research_agent.hf_integration.types import CACHE_LAYOUT, CACHE_PREFIX

    assert set(CACHE_LAYOUT) == {"root", "refs", "snapshots", "blobs", "folder"}
    assert CACHE_PREFIX == "models--"
    assert all(CACHE_LAYOUT.values())
