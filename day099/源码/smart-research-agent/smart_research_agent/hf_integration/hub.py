"""``hub``：``from_pretrained`` 的第一步——**把文件找齐**（day086 / M7-D10）.

`from_pretrained("gpt2")` 这个调用的第一步**不是**加载权重，而是把四份文件找齐：
``config.json`` / ``vocab.json`` / ``merges.txt`` / ``model.safetensors``。
找齐这件事有一个被写进协议的三段结构，本模块把它完整复刻：

```text
① 名字 → commit     refs/<revision> 里存着一串 commit hash
                    （"main" 只是那串 hash 的一个**可读别名**）
② commit → 文件     快照目录 snapshots/<commit>/ 里的相对路径**与仓库一致**（平铺）
③ 文件 → 实体       blobs/<sha256> 存实体；快照里的路径与实体之间是"两个名字"
```

## 为什么这一层值得单独写一个模块

因为"东西在本地"和"东西能用"是两件不同的事，而两者失败时**长得一样**：

```text
参数全对 + 缓存里没有   →  LocalEntryNotFoundError    （本包的 HubError）
参数全对 + 网络断了     →  同样一句连接错误            （本包的 HubError）
参数写错（max_length=0）→  ParameterError
```

把这三件事混进同一个族里，修法就会指错方向：一个"该去联网"的失败
会被当成"该去改参数"，于是有人把 ``local_files_only`` 改成 ``False`` 再试一遍——
而网络那一步的失败长得**完全一样**（day086 的失败族表里那条）。

## 本模块不碰文件系统

缓存被建模成一个**内存结构** :class:`CacheStore`（对应磁盘上的 ``$HF_HOME/hub``），
网络侧被建模成一个**可注入的客户端** :class:`HubClient`。
两条都刻意的：这样"缓存命中是逐文件的"这句话可以被**逐条断言**，
而不是靠"跑一遍看看下没下"。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Protocol

from smart_research_agent.hf_integration.errors import (
    ConfigError,
    HubError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.hf_integration.types import (
    BLOB_DIR,
    CACHE_PREFIX,
    CONFIG_FILE,
    DEFAULT_REVISION,
    KNOWN_FILES,
    REF_DIR,
    SNAPSHOT_DIR,
    CacheEntry,
    Snapshot,
)

#: 缓存根的默认位置（真实事实：``$HF_HOME/hub``，而 ``HF_HOME`` 默认 ``~/.cache/huggingface``）.
DEFAULT_CACHE_DIR = "~/.cache/huggingface/hub"

#: 一次解析至少要有的那一个文件。缺了它，后面全都无从谈起。
REQUIRED_FILES: tuple[str, ...] = (CONFIG_FILE,)


class HubClient(Protocol):
    """远端 Hub 的最小接口（**三个方法**，正好够回答"有没有、哪一版、有哪些文件"）.

    把它做成协议而不是一个具体类，是为了让"离线也能跑"成为一条**设计属性**：
    测试注入内存假仓库，生产注入真客户端，而 :func:`HubResolver.resolve` 一行不改。
    """

    def repo_exists(self, repo_id: str) -> bool:
        """这个仓库存不存在（对应真实世界的 ``RepositoryNotFoundError`` 判据）."""

    def resolve_commit(self, repo_id: str, revision: str) -> str:
        """把一个可读版本名翻译成 commit hash（对应 ``refs/<revision>`` 的内容）."""

    def list_files(self, repo_id: str, commit: str) -> dict[str, bytes]:
        """列出某个 commit 下的全部文件：``相对路径 → 字节``."""


@dataclass
class InMemoryHub:
    """一个内存里的假仓库（**测试与演示都用它，因此不碰网络**）.

    结构是三层字典，与真实的 Hub 协议一一对应：

    ```text
    commits[repo_id][revision] = commit          ← 对应 refs/<revision>
    trees[repo_id][commit]     = {path: bytes}   ← 对应 snapshots/<commit>/ 下的文件清单
    ```
    """

    commits: dict[str, dict[str, str]] = field(default_factory=dict)
    trees: dict[str, dict[str, dict[str, bytes]]] = field(default_factory=dict)
    #: 每一次 ``list_files`` 调用都记一笔——"下载了几次"这件事由此可被断言.
    calls: list[tuple[str, str]] = field(default_factory=list)

    def repo_exists(self, repo_id: str) -> bool:
        """仓库存在 ⇔ 它有至少一个 commit 映射."""
        return bool(self.commits.get(repo_id))

    def resolve_commit(self, repo_id: str, revision: str) -> str:
        """把版本名翻译成 commit（两种失败分别报，而不是合成一句"找不到"）."""
        if not self.repo_exists(repo_id):
            raise HubError(
                f"仓库 {repo_id!r} 不存在：请检查组织名与仓库名"
                "（真实的 huggingface_hub 在这里抛 RepositoryNotFoundError——"
                "私有库没有权限时是**同一条**错误，因此不能靠它区分'名字错'与'没权限'）。"
            )
        table = self.commits[repo_id]
        if revision not in table:
            raise HubError(
                f"版本 {revision!r} 在仓库 {repo_id!r} 里不存在："
                f"可用版本 {sorted(table)}（真实世界抛 RevisionNotFoundError）。"
            )
        return table[revision]

    def list_files(self, repo_id: str, commit: str) -> dict[str, bytes]:
        """列出某个 commit 下的文件（``commit`` 直接查表，不做二次翻译）."""
        self.calls.append((repo_id, commit))
        tree = self.trees.get(repo_id, {}).get(commit)
        if tree is None:
            raise HubError(
                f"仓库 {repo_id!r} 里没有 commit {commit!r} 的文件树："
                "commit 必须来自 resolve_commit，手写一个 hash 是不被承认的。"
            )
        return dict(tree)


@dataclass
class CacheStore:
    """一个内存里的缓存根（对应磁盘上的 ``$HF_HOME/hub``）.

    三段结构与真实缓存逐一对应：``refs`` / ``snapshots`` / ``blobs``。
    **``blobs`` 按 sha256 存**，因此两个快照可以共享同一份实体文件——
    这正是"两个名字"（快照里的相对路径、blobs 里的哈希名）存在的理由。
    """

    refs: dict[str, str] = field(default_factory=dict)
    #: ``"<repo_folder>/<commit>" -> {相对路径: sha256}``
    snapshots: dict[str, dict[str, str]] = field(default_factory=dict)
    #: ``sha256 -> 字节``
    blobs: dict[str, bytes] = field(default_factory=dict)

    @property
    def blob_count(self) -> int:
        """实体文件的个数（去重之后）——两个快照共享同一份时它不变."""
        return len(self.blobs)

    @property
    def snapshot_count(self) -> int:
        """快照目录的个数（每个 commit 一个）."""
        return len(self.snapshots)

    def total_bytes(self) -> int:
        """缓存占用的总字节数（**按 blobs 算**，因此共享的文件只算一次）."""
        return sum(len(data) for data in self.blobs.values())

    def describe(self) -> dict[str, object]:
        """摊平成一行字段（报告里读它）。"""
        return {
            "refs": len(self.refs),
            "snapshots": self.snapshot_count,
            "blobs": self.blob_count,
            "bytes": self.total_bytes(),
        }


def repo_folder(repo_id: str) -> str:
    """``org/name`` → ``models--org--name``（真实的目录命名规则）.

    ``repo_id`` 必须是 ``org/name`` 两段：多一段少一段都会被拒绝，
    因为斜杠在目录名里被换成了双横线，**改写是不可逆的**——
    一个 ``a/b/c`` 会与 ``a--b/c`` 落到同一个目录里。
    """
    text = (repo_id or "").strip()
    if text.count("/") != 1 or text.startswith("/") or text.endswith("/"):
        raise ParameterError(
            f"repo_id 必须是 'org/name' 两段，收到 {repo_id!r}："
            "目录名里的斜杠会被换成双横线，因此三段及以上是不可逆的"
            "（'a/b/c' 与 'a--b/c' 会撞到同一个目录）。"
        )
    return CACHE_PREFIX + text.replace("/", "--")


def snapshot_root(cache_dir: str, repo_id: str, commit: str) -> str:
    """快照目录的路径 ``<cache>/<folder>/snapshots/<commit>``（**平铺的根**）."""
    return f"{cache_dir}/{repo_folder(repo_id)}/{SNAPSHOT_DIR}/{commit}"


def blob_path(cache_dir: str, repo_id: str, digest: str) -> str:
    """实体文件的路径 ``<cache>/<folder>/blobs/<sha256>``.

    注意它**只由 sha256 决定文件名**（前缀仍是仓库目录）——
    因此同一个文件在两个 revision 里只存一份，而"两个名字指向一份实体"
    正是快照与 blobs 之间那层间接的意义。
    """
    return f"{cache_dir}/{repo_folder(repo_id)}/{BLOB_DIR}/{digest}"


def sha256_of(data: bytes) -> str:
    """一个字节串的 sha256（缓存里的实体就是这样被寻址的）."""
    return hashlib.sha256(data).hexdigest()


def select_files(
    manifest: dict[str, bytes],
    *,
    allow_patterns: tuple[str, ...] | None = None,
) -> dict[str, bytes]:
    """按 ``allow_patterns`` 过滤文件清单（支持 ``*`` 与 ``?``，与真实口径同形）.

    ``None`` 表示**全要**。空元组表示**一个都不要**——而它与 ``None`` 必须分开：
    "没写这个参数"与"写了个空的"在报告里长得一样，却完全是两件事。
    """
    if allow_patterns is None:
        return dict(manifest)
    if not allow_patterns:
        raise ParameterError(
            "allow_patterns 是空元组：它表示'一个文件都不要'，"
            "而'没写这个参数'（None）表示'全要'。两者必须分开写。"
        )
    return {
        path: data
        for path, data in manifest.items()
        if any(fnmatch(path, pattern) for pattern in allow_patterns)
    }


@dataclass
class HubResolver:
    """把"名字 → commit → 文件 → 实体"三段串起来的那一个对象.

    ``local_files_only=True``（默认）是**离线优先**：任何需要联网才能补上的文件
    都会当场变成 :class:`HubError`，而不是悄悄去连一次网络。
    这个默认值与真实库相反（那里默认允许联网），而本课程刻意反过来——
    因为"我以为它在本地"与"它其实下了 12 秒"在报告里长得一样。
    """

    hub: HubClient
    store: CacheStore = field(default_factory=CacheStore)
    cache_dir: str = DEFAULT_CACHE_DIR

    def resolve(
        self,
        repo_id: str,
        *,
        revision: str = DEFAULT_REVISION,
        allow_patterns: tuple[str, ...] | None = None,
        required: tuple[str, ...] = REQUIRED_FILES,
        local_files_only: bool = True,
    ) -> Snapshot:
        """解析出一个快照：命中缓存的逐文件复用，缺的那些（若允许）当场下载."""
        folder = repo_folder(repo_id)
        commit = self._commit_of(repo_id, folder, revision, local_files_only=local_files_only)
        key = f"{folder}/{commit}"
        snapshot_map: dict[str, str] = dict(self.store.snapshots.get(key, {}))

        # 离线时"有哪些文件"只能来自缓存里的快照表；联网时来自远端清单。
        # 两条路都必须**先有清单再逐个文件判缓存**——这正是"逐文件命中"的实现。
        if local_files_only:
            if not snapshot_map:
                raise HubError(
                    f"离线模式下缓存里没有 {repo_id!r} 的 {revision!r}："
                    f"快照目录 {SNAPSHOT_DIR}/{commit}/ 还不存在。"
                    "把它当成参数错的后果是——有人会把 local_files_only 改成 False 再试，"
                    "而网络那一步的失败与它长得完全一样。"
                )
            manifest: dict[str, bytes] = {}
            for path in sorted(snapshot_map):
                digest = snapshot_map[path]
                if digest not in self.store.blobs:
                    raise HubError(
                        f"离线模式下缓存里没有 {path!r} 的实体文件（blob {digest[:12]}… 缺失）："
                        "快照表里有它的名字、但那份实体已经被清掉了——"
                        "'名字在'与'东西在'是两件事。"
                    )
                manifest[path] = self.store.blobs[digest]
        else:
            manifest = self._fetch(repo_id, commit, allow_patterns=allow_patterns)

        entries: list[CacheEntry] = []
        for path in sorted(manifest):
            data = manifest[path]
            digest = sha256_of(data)
            cached = snapshot_map.get(path) == digest and digest in self.store.blobs
            if not cached:
                if local_files_only:  # pragma: no cover - 离线分支的清单来自缓存，必然命中
                    raise HubError(f"离线模式下缓存里没有 {path!r}。")
                self.store.blobs[digest] = data
            snapshot_map[path] = digest
            entries.append(CacheEntry(path=path, size=len(data), cached=cached))

        self.store.snapshots[key] = snapshot_map
        names = {entry.path for entry in entries}
        for name in required:
            if name not in names:
                raise HubError(
                    f"快照里缺少必需文件 {name!r}：现有的文件是 {sorted(names)}。"
                    "allow_patterns 把它过滤掉了，或者这个仓库本来就不是模型仓库。"
                )
        return Snapshot(
            repo_id=repo_id,
            revision=revision,
            commit=commit,
            root=snapshot_root(self.cache_dir, repo_id, commit),
            files=tuple(entries),
            local_files_only=local_files_only,
        )

    def _commit_of(
        self,
        repo_id: str,
        folder: str,
        revision: str,
        *,
        local_files_only: bool,
    ) -> str:
        """把版本名翻译成 commit：先查 ``refs``，查不到再（若允许）去远端问一次."""
        ref_key = f"{folder}/{revision}"
        pinned = self.store.refs.get(ref_key)
        if pinned is not None:
            return pinned
        if local_files_only:
            raise HubError(
                f"离线模式下缓存里没有版本 {revision!r} 的引用："
                f"refs/{revision} 还不存在，因此连 commit 都不知道。"
            )
        commit = self.hub.resolve_commit(repo_id, revision)
        self.store.refs[ref_key] = commit
        return commit

    def _fetch(
        self,
        repo_id: str,
        commit: str,
        *,
        allow_patterns: tuple[str, ...] | None,
    ) -> dict[str, bytes]:
        """把远端清单拉下来并过滤（**只有这一步会碰客户端**）."""
        manifest = self.hub.list_files(repo_id, commit)
        if not manifest:
            raise HubError(f"仓库 {repo_id!r} 的 commit {commit!r} 里一个文件都没有。")
        return select_files(manifest, allow_patterns=allow_patterns)

    def read_bytes(self, snapshot: Snapshot, path: str) -> bytes:
        """读一个文件（**不碰文件系统**：从 ``store`` 里按 sha256 取）."""
        folder = repo_folder(snapshot.repo_id)
        key = f"{folder}/{snapshot.commit}"
        table = self.store.snapshots.get(key)
        if not table or path not in table:
            raise HubError(
                f"快照 {snapshot.repo_id}@{snapshot.commit} 里没有 {path!r}："
                f"它有的文件是 {sorted(table or {})}。"
            )
        return self.store.blobs[table[path]]

    def read_text(self, snapshot: Snapshot, path: str) -> str:
        """读一个文本文件（UTF-8）。读不出来时按"这份文件坏了"报."""
        try:
            return self.read_bytes(snapshot, path).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ConfigError(
                f"{path!r} 不是合法的 UTF-8 文本（{exc}）："
                "配置文件与词表都必须是 UTF-8——用别的编码写出来的 config.json "
                "会在这里变成一个'解析失败'，而它的形状看起来是合法的。"
            ) from exc

    def read_json(self, snapshot: Snapshot, path: str) -> dict:
        """读一个 JSON 文件并校验它是一个对象（不是数组、不是字符串）."""
        text = self.read_text(snapshot, path)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(
                f"{path!r} 不是合法的 JSON（{exc.msg}，第 {exc.lineno} 行）："
                "配置文件坏掉时最省事的修法是不要手改它。"
            ) from exc
        if not isinstance(payload, dict):
            raise ConfigError(
                f"{path!r} 的顶层是 {type(payload).__name__}，不是对象："
                "配置文件必须是一个 JSON 对象（键值对），数组与字符串都不行。"
            )
        return payload


def flat_paths(snapshot: Snapshot) -> tuple[str, ...]:
    """快照里那些**没有**被写成内部路径的名字（这是"平铺"的可断言形式）.

    只要有一个名字落在 ``blobs/`` 或 ``snapshots/`` 里面，它就不可能被
    ``os.path.join(root, name)`` 找到——而那时 ``config.json`` 的缺席
    会被报成"仓库里没有这个文件"，指错方向。
    """
    return tuple(
        entry.path
        for entry in snapshot.files
        if not entry.path.startswith(f"{BLOB_DIR}/")
        and not entry.path.startswith(f"{SNAPSHOT_DIR}/")
    )


def known_offered(snapshot: Snapshot) -> tuple[str, ...]:
    """快照里那几个"本包认得"的文件（用于演示与自检）."""
    return tuple(name for name in snapshot.names() if name in KNOWN_FILES)


def check_readable(snapshot: Snapshot, names: tuple[str, ...]) -> None:
    """自检：这几条路径必须在快照里（缺一条就报形状错，指到具体那一条）."""
    offered = set(snapshot.names())
    missing = [name for name in names if name not in offered]
    if missing:
        raise ShapeError(
            f"快照里缺少 {missing}：它有的文件是 {sorted(offered)}。"
            "（在真实世界这里会是一次 FileNotFoundError，而报错点离出错点很远。）"
        )


__all__ = [
    "DEFAULT_CACHE_DIR",
    "REQUIRED_FILES",
    "CacheStore",
    "HubClient",
    "HubResolver",
    "InMemoryHub",
    "blob_path",
    "check_readable",
    "flat_paths",
    "known_offered",
    "repo_folder",
    "select_files",
    "sha256_of",
    "snapshot_root",
]
