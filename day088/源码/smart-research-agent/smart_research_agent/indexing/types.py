"""索引的形状：清单、条目、差集、编码报告（M6-D4）.

day064 的产物是一个**库**：存了向量与原文，能按相似度查。
今天要在这上面加一层**账**——回答三个"库本身答不了"的问题：

```text
1. 这份索引是用哪个编码器的哪个版本建的？   → EmbeddingIdentity
2. 上一版到这一版，哪些块没变、哪些变了？     → IndexEntry / IndexPlan
3. 我能不能回到上一版，或者照清单重建一次？   → IndexManifest.version_id / parent_version
```

三条里第 2 条是省钱的来源，第 1 条是**正确性的来源**，第 3 条是可运维性的来源。

## 为什么键里必须有"编码器身份"

这是本模块最容易写错、也最毒的一条。向量的缓存键取
``sha256(text)[:16]`` 看似够了：同一段文本得到同一个向量，天经地义。
但换一次 embedding 模型之后：

```text
文本没变 → 缓存键没变 → **命中旧模型算出的向量**
→ 新旧向量混在同一个库里 → 检索结果整体错乱
```

而这个过程**不会有任何异常**：缓存命中了、写入成功了、查询也返回了，
只是"谁更近"变了。因此本包的向量键是：

```python
vector_key(provider, model, dimension, text) = sha256(f"{provider}|{model}|{dimension}\n{text}")[:16]
```

**身份进键**，换来的是"换模型必须重建索引"这件事变成**必然**而不是**靠自觉**：
换了身份，所有键都变，差集算出"全部需要重算"。这与 day062 把
``strategy`` 与 ``index`` 放进 ``chunk_id``、day058 把三元组放进版本键
是同一条纪律：**让口径的每一次变化都显式地改变身份。**

## 时间戳为什么不能进版本号

``IndexManifest`` 带 ``created_at``，但它**不参与** ``version_id`` 的计算。
理由与 day062 用 ``chars`` 而不是 ``tiktoken`` 作默认度量完全一样：

```text
时间戳进版本号 → 同一份内容在两台机器上得到两个版本号
                → "这个版本在哪台机器上建的？" 变成必须回答的问题
                → 而它本来不该是一个问题
```

版本号要能被任何人**重算**：给定同样的内容、同样的编码器身份、同样的度量，
在任何时间任何机器上算出来的是同一个 16 位十六进制数。
``created_at`` 是给人看的备注，不是身份的一部分。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.indexing.errors import IndexingError, ManifestError

#: 清单格式版本。**不匹配时拒读**（见 ``errors.BackupError`` 的说明）。
MANIFEST_VERSION = 1

#: 向量键与版本号的长度：sha256 前 16 位十六进制。
#: 与 day061 的 ``doc_id``、day062 的 ``chunk_id`` 取同一个量级——
#: 它们会一起出现在同一份报告里。
VECTOR_KEY_LENGTH = 16

#: 差集里的三种动作名（**顺序就是报告里的顺序**）。
ACTION_ADDED = "added"
ACTION_UPDATED = "updated"
ACTION_REMOVED = "removed"
ACTION_UNCHANGED = "unchanged"
ACTIONS: tuple[str, ...] = (ACTION_ADDED, ACTION_UPDATED, ACTION_REMOVED, ACTION_UNCHANGED)


def vector_key(provider: str, model: str, dimension: int, text: str) -> str:
    """算一份文本在某个编码器下的向量键（见模块 docstring 的取舍）.

    ``dimension`` 也进键：同一个模型名在不同输出维度设置下
    （例如 OpenAI 的 ``dimensions`` 截断参数）会给出不同长度的向量，
    而"维度不同"的向量混进同一个库是 day064 已经拦过的那类错误。
    """
    payload = f"{provider}|{model}|{dimension}\n{text}".encode()
    return hashlib.sha256(payload).hexdigest()[:VECTOR_KEY_LENGTH]


def entries_digest(entries: tuple[IndexEntry, ...]) -> str:
    """算一批条目的内容摘要（**逐条三字段、按 id 升序**）.

    摘要只取 ``record_id | fingerprint | vector_key`` 三样：

    ```text
    record_id     哪一条
    fingerprint   内容有没有变（day062 的内容身份）
    vector_key    该不该复用向量（编码器身份 + 文本）
    ```

    ``token_count`` 这一类只用于统计的字段**不进摘要**：
    它们的变化不需要重建索引，进了摘要就会让"改一个统计字段"
    变成"整库失效"——这种"过度敏感"的版本号比不准的版本号更麻烦。

    **函数内部自己排序**，不依赖调用方传进来的顺序。这一条是必须的：
    摘要要只取决于"内容集合"，而"谁先谁后"是一个与内容无关的事实。
    不排序的后果是同一份内容按不同顺序算得两个版本号，
    于是重建一次索引就凭空多出一个版本——而两个版本的差别
    在报告里**什么也看不出来**（条目集合完全相同）。
    """
    lines = [
        f"{entry.record_id}|{entry.fingerprint}|{entry.vector_key}"
        for entry in sorted(entries, key=lambda item: item.record_id)
    ]
    payload = "\n".join(lines).encode()
    return hashlib.sha256(payload).hexdigest()[:VECTOR_KEY_LENGTH]


def index_version_id(
    *,
    identity_key: str,
    metric: str,
    backend: str,
    digest: str,
) -> str:
    """算索引版本号：由"编码器身份 + 度量 + 后端 + 内容摘要"决定.

    四个输入都被显式列出来，因为**它们中任何一个变了，索引的语义就变了**：

    ```text
    identity_key  换模型 → 全部向量作废
    metric        换度量 → "谁更近"整体改变（day064 已论证）
    backend       换后端 → 可以重建，但**不能与旧版共用一份清单**
    digest        内容变了 → 只有它是"数据"的变化，其余三个是"配置"的变化
    ```
    """
    payload = f"{MANIFEST_VERSION}|{identity_key}|{metric}|{backend}|{digest}".encode()
    return hashlib.sha256(payload).hexdigest()[:VECTOR_KEY_LENGTH]


def _is_hex_id(value: str) -> bool:
    return len(value) == VECTOR_KEY_LENGTH and all(
        char in "0123456789abcdef" for char in value
    )


@dataclass(frozen=True)
class EmbeddingIdentity:
    """编码器的身份：三个字段一起才够用.

    ```text
    provider   实现类名（MockEmbedding / CharNgramEmbedding / …）
    model      模型或配置标识（模型名、n-gram 阶数、维度设置等）
    dimension  输出维度
    ```

    为什么"三个一起"而不是只留模型名：``CharNgramEmbedding`` 没有模型名，
    它的身份藏在 ``ngram_sizes`` 里；而同一个模型名在两种输出维度下
    也会给出不同长度的向量。**只留一个字段必然会在某个提供方上失效。**
    """

    provider: str
    model: str
    dimension: int

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ManifestError("EmbeddingIdentity.provider 不能为空")
        if not self.model.strip():
            raise ManifestError(
                "EmbeddingIdentity.model 不能为空：没有模型的编码器身份"
                "无法回答'这份索引是用谁建的'，请传一个稳定的配置标识"
                "（例如 CharNgramEmbedding 的 n-gram 阶数）"
            )
        if self.dimension < 1:
            raise ManifestError(
                f"EmbeddingIdentity.dimension 必须 >= 1，收到 {self.dimension}"
            )

    @property
    def key(self) -> str:
        """身份指纹（进向量键与版本号的就是它）."""
        return vector_key(self.provider, self.model, 0, f"dimension={self.dimension}")

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return {
            "provider": self.provider,
            "model": self.model,
            "dimension": self.dimension,
            "key": self.key,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EmbeddingIdentity:
        """从字典还原（缺字段时给出可读的 ``ManifestError``）."""
        try:
            return cls(
                provider=str(payload["provider"]),
                model=str(payload["model"]),
                dimension=int(payload["dimension"]),
            )
        except KeyError as exc:
            raise ManifestError(
                f"编码器身份缺少字段 {exc.args[0]!r}：清单里的 provider/model/dimension 三者缺一不可，"
                "少了任何一个都无法判断'这份索引是用谁建的'"
            ) from exc

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        return f"{self.provider}/{self.model} ({self.dimension}d)"


@dataclass(frozen=True)
class IndexEntry:
    """清单里的一条：一个块，以及"它的向量该怎么算"凭据.

    三个字段各回答一个问题，**缺一个就会让某类变更无法被发现**：

    | 字段 | 回答 | 少了它会怎样 |
    |------|------|-------------|
    | ``record_id`` | 这是哪一条 | 无法定位要更新/删除的记录 |
    | ``fingerprint`` | 内容变了没有 | 只改了原文但向量键没变（同长度覆盖写）时会漏判 |
    | ``vector_key`` | 该不该复用向量 | 无法区分"内容没变"与"编码器变了" |

    ``token_count`` 只用于统计（成本估算、报告），**不进摘要**——
    见 ``entries_digest`` 的说明。
    """

    record_id: str
    fingerprint: str
    vector_key: str
    token_count: int = 0

    def __post_init__(self) -> None:
        if not self.record_id.strip():
            raise ManifestError("IndexEntry.record_id 不能为空")
        if not _is_hex_id(self.fingerprint):
            raise ManifestError(
                f"IndexEntry.fingerprint 必须是 {VECTOR_KEY_LENGTH} 位十六进制，"
                f"收到 {self.fingerprint!r}（day062 的 chunk fingerprint 就是这个形状）"
            )
        if not _is_hex_id(self.vector_key):
            raise ManifestError(
                f"IndexEntry.vector_key 必须是 {VECTOR_KEY_LENGTH} 位十六进制，"
                f"收到 {self.vector_key!r}"
            )
        if self.token_count < 0:
            raise ManifestError(f"IndexEntry.token_count 必须非负，收到 {self.token_count}")

    def to_dict(self) -> dict[str, Any]:
        """投影为字典."""
        return {
            "record_id": self.record_id,
            "fingerprint": self.fingerprint,
            "vector_key": self.vector_key,
            "token_count": self.token_count,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> IndexEntry:
        """从字典还原（字段缺失时给出可读错误）."""
        try:
            return cls(
                record_id=str(payload["record_id"]),
                fingerprint=str(payload["fingerprint"]),
                vector_key=str(payload["vector_key"]),
                token_count=int(payload.get("token_count", 0)),
            )
        except KeyError as exc:
            raise ManifestError(
                f"清单条目的 record_id/fingerprint/vector_key 三者缺一不可，"
                f"缺少 {exc.args[0]!r}"
            ) from exc

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        return (
            f"{self.record_id} | fp={self.fingerprint} | vk={self.vector_key} | "
            f"{self.token_count} token"
        )


@dataclass(frozen=True)
class IndexManifest:
    """一份索引的清单：它是"上一版长什么样"的唯一证据.

    字段分成三组：

    ```text
    身份    version_id / parent_version              版本与血缘
    口径    identity / metric / backend              "用什么建的"
    内容    entries / digest                          "建了哪些"
    ```

    ``entries`` 在构造时被**强制成"按 record_id 升序且不重复"**。
    这不是美化：清单要参与版本号计算，而"同一批条目按不同顺序排列"
    必须得到同一个版本号——**否则重建一次索引就会凭空多出一个版本**。
    """

    version_id: str
    identity: EmbeddingIdentity
    metric: str
    backend: str
    entries: tuple[IndexEntry, ...] = ()
    parent_version: str = ""
    created_at: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _is_hex_id(self.version_id):
            raise ManifestError(
                f"version_id 必须是 {VECTOR_KEY_LENGTH} 位十六进制，收到 {self.version_id!r}"
            )
        if self.parent_version and not _is_hex_id(self.parent_version):
            raise ManifestError(
                f"parent_version 必须是 {VECTOR_KEY_LENGTH} 位十六进制或空串，"
                f"收到 {self.parent_version!r}"
            )
        if not self.backend.strip():
            raise ManifestError("IndexManifest.backend 不能为空：清单必须说清它描述的是哪个后端的库")
        if not self.metric.strip():
            raise ManifestError("IndexManifest.metric 不能为空")

        ordered = tuple(sorted(self.entries, key=lambda entry: entry.record_id))
        object.__setattr__(self, "entries", ordered)
        seen: set[str] = set()
        for entry in ordered:
            if entry.record_id in seen:
                raise ManifestError(
                    f"清单里出现重复的 record_id {entry.record_id!r}："
                    "同一份索引不可能同时存在两条同 id 的记录——"
                    "重复通常意味着两次构建的结果被拼在了一起"
                )
            seen.add(entry.record_id)

    @property
    def count(self) -> int:
        """条目数."""
        return len(self.entries)

    @property
    def digest(self) -> str:
        """内容摘要（见 ``entries_digest``）."""
        return entries_digest(self.entries)

    @property
    def record_ids(self) -> tuple[str, ...]:
        """全部 id（已升序，可直接与 ``backend.ids()`` 比对）."""
        return tuple(entry.record_id for entry in self.entries)

    @property
    def total_tokens(self) -> int:
        """条目 token 数之和（只用于成本估算，不参与版本号）."""
        return sum(entry.token_count for entry in self.entries)

    def entry_map(self) -> dict[str, IndexEntry]:
        """id → 条目的映射（差集计算用）."""
        return {entry.record_id: entry for entry in self.entries}

    def verify_version_id(self) -> str:
        """重算版本号并与 ``version_id`` 核对；不一致则抛 ``ManifestError``.

        这是"清单可被任何人重算"这条纪律的**执行器**：
        一份被手工改过（或截断）的清单会在下一次构建时当场炸掉，
        而不是被安静地当成"上一版"。
        """
        expected = index_version_id(
            identity_key=self.identity.key,
            metric=self.metric,
            backend=self.backend,
            digest=self.digest,
        )
        if expected != self.version_id:
            raise ManifestError(
                f"清单的 version_id 与内容不一致：记录的是 {self.version_id}，"
                f"按内容重算应为 {expected}。"
                "清单可能被手工改动过——请按当前内容重新构建，而不要沿用这份清单。"
            )
        return expected

    def to_dict(self, *, include_entries: bool = True) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典.

        ``include_entries=False`` 时只返回身份与统计：
        一份十万条的清单把全部条目序列化出来接近 10 MB。
        """
        payload: dict[str, Any] = {
            "manifest_version": MANIFEST_VERSION,
            "version_id": self.version_id,
            "parent_version": self.parent_version,
            "identity": self.identity.to_dict(),
            "metric": self.metric,
            "backend": self.backend,
            "count": self.count,
            "digest": self.digest,
            "total_tokens": self.total_tokens,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }
        if include_entries:
            payload["entries"] = [entry.to_dict() for entry in self.entries]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> IndexManifest:
        """从字典还原清单（格式版本不匹配时拒读）."""
        version = int(payload.get("manifest_version", 0))
        if version != MANIFEST_VERSION:
            raise ManifestError(
                f"清单格式版本是 {version}，本代码只认识 {MANIFEST_VERSION}。"
                "拒绝读取：用旧格式的清单去驱动增量构建，会让'哪些块要重算'算错，"
                "而算错的表现只是检索质量下降。请重新构建一份清单。"
            )
        raw_entries = payload.get("entries", [])
        if not isinstance(raw_entries, list):
            raise ManifestError("清单的 entries 必须是列表")
        return cls(
            version_id=str(payload["version_id"]),
            identity=EmbeddingIdentity.from_dict(dict(payload["identity"])),
            metric=str(payload["metric"]),
            backend=str(payload["backend"]),
            entries=tuple(IndexEntry.from_dict(dict(item)) for item in raw_entries),
            parent_version=str(payload.get("parent_version", "")),
            created_at=str(payload.get("created_at", "")),
            metadata={str(k): str(v) for k, v in dict(payload.get("metadata", {})).items()},
        )

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        parent = self.parent_version or "（首版）"
        return (
            f"版本 {self.version_id} ← {parent} | {self.identity.summary_line()} | "
            f"{self.metric} | {self.backend} | {self.count} 条 | 摘要 {self.digest}"
        )

    def lineage(self, other: IndexManifest | None) -> str:
        """血缘关系的一句话（版本表里逐行打印）."""
        if other is None:
            return "首版（没有上一版）"
        if self.parent_version == other.version_id:
            return f"紧接 {other.version_id}"
        if self.parent_version:
            return f"接在 {self.parent_version} 之后（不是 {other.version_id}）"
        return f"没有声明父版本（当前最新是 {other.version_id}）"


@dataclass(frozen=True)
class IndexPlan:
    """一次增量构建的差集：四种去向，各自是一批 id.

    与 day064 的 ``WriteReport`` 是同一个设计（四个数分家），
    区别是这里的粒度是**一批 id 而不是计数**：报告要能回答
    "哪些块需要重新编码"，而不只是"有几个块需要"。
    """

    added: tuple[str, ...] = ()
    updated: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    unchanged: tuple[str, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        for name in ACTIONS:
            values = getattr(self, name)
            if len(set(values)) != len(values):
                raise IndexingError(f"IndexPlan.{name} 里出现重复 id")
            if list(values) != sorted(values):
                raise IndexingError(
                    f"IndexPlan.{name} 必须是升序："
                    "报告要能被逐行 diff，而乱序的 id 列表每次都会长得不一样"
                )
        groups = [set(getattr(self, name)) for name in ACTIONS]
        for index, left in enumerate(groups):
            for right in groups[index + 1 :]:
                overlap = left & right
                if overlap:
                    raise IndexingError(
                        f"同一个 id 不能同时出现在两个动作里，例如 {sorted(overlap)[0]!r}："
                        "一个块要么新增、要么更新、要么删除、要么没变"
                    )

    @property
    def total(self) -> int:
        """这一批总共涉及多少条.``"""
        return len(self.added) + len(self.updated) + len(self.removed) + len(self.unchanged)

    @property
    def to_encode(self) -> tuple[str, ...]:
        """需要**重新编码**的 id（新增 + 更新）.

        这是本课省钱的那个数字：``len(plan.to_encode)`` 越小，
        这次构建就越便宜——而它是否真的变小，取决于
        "内容身份"与"向量键"这两层判据是否分开（见 ``planner``）。
        """
        return tuple(sorted((*self.added, *self.updated)))

    @property
    def reuse_ratio(self) -> float:
        """复用率 = 未变条目 / 总条目.

        它是"这次增量值不值得"的直接依据：复用率很低时，
        逐条 upsert 的开销可能反而高于整库重建（见 ``builder``）。
        """
        if self.total <= 0:
            return 0.0
        return round(len(self.unchanged) / self.total, 4)

    @property
    def changed(self) -> int:
        """让索引发生变化的条数（新增 + 更新 + 删除）."""
        return len(self.added) + len(self.updated) + len(self.removed)

    def to_dict(self, *, include_ids: bool = True) -> dict[str, Any]:
        """投影为字典（``include_ids=False`` 时只给计数与复用率）."""
        payload: dict[str, Any] = {
            "added": len(self.added),
            "updated": len(self.updated),
            "removed": len(self.removed),
            "unchanged": len(self.unchanged),
            "total": self.total,
            "changed": self.changed,
            "reuse_ratio": self.reuse_ratio,
            "reason": self.reason,
        }
        if include_ids:
            payload["added_ids"] = list(self.added)
            payload["updated_ids"] = list(self.updated)
            payload["removed_ids"] = list(self.removed)
        return payload

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        return (
            f"新增 {len(self.added)} / 更新 {len(self.updated)} / "
            f"删除 {len(self.removed)} / 未变 {len(self.unchanged)} | "
            f"复用率 {self.reuse_ratio:.2%} | {self.reason}"
        )


@dataclass(frozen=True)
class EncodeReport:
    """一次批量编码的账：请求数、命中数、真正编码数、批次数.

    ```text
    texts       请求了多少份文本
    cache_hits  其中直接命中缓存的
    encoded     真正送给提供方的
    batches     向提供方发起了几批（与 day064 的"逐条调用"对照）
    ```

    ``batches`` 与 ``encoded`` 一起看才有意义：
    ``encoded=6, batches=1`` 是"一批装下 6 条"，
    ``encoded=6, batches=6`` 是"逐条调用"——**同一份数据、同一个结果，成本不同**。
    """

    texts: int = 0
    cache_hits: int = 0
    encoded: int = 0
    batches: int = 0
    dimension: int = 0
    provider: str = ""

    @property
    def hit_ratio(self) -> float:
        """缓存命中率（``texts`` 为 0 时是 0.0，不是除零错误）."""
        if self.texts <= 0:
            return 0.0
        return round(self.cache_hits / self.texts, 4)

    def to_dict(self) -> dict[str, Any]:
        """投影为字典."""
        return {
            "texts": self.texts,
            "cache_hits": self.cache_hits,
            "encoded": self.encoded,
            "batches": self.batches,
            "hit_ratio": self.hit_ratio,
            "dimension": self.dimension,
            "provider": self.provider,
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        return (
            f"{self.texts} 份文本 → 命中 {self.cache_hits} / 编码 {self.encoded} | "
            f"{self.batches} 批 | 命中率 {self.hit_ratio:.2%} | {self.provider} {self.dimension}d"
        )


@dataclass(frozen=True)
class IndexingReport:
    """一次索引构建的完整账（``/indexing/build`` 直接返回它）."""

    mode: str
    version_id: str
    parent_version: str
    backend: str
    metric: str
    dimension: int
    provider: str
    seen: int = 0
    cache_hits: int = 0
    encoded: int = 0
    batches: int = 0
    written: int = 0
    unchanged: int = 0
    removed: int = 0
    failed: int = 0
    reuse_ratio: float = 0.0
    failures: tuple[tuple[str, str], ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        if self.mode not in ("full", "incremental"):
            raise IndexingError(
                f"mode 只能是 full 或 incremental，收到 {self.mode!r}"
            )

    @property
    def ok(self) -> bool:
        """这次构建是否"没有一条需要人再看一眼"."""
        return self.failed == 0

    @property
    def ok_count(self) -> int:
        """成功处理的条数（写入 + 未变）."""
        return self.written + self.unchanged

    def to_dict(self, *, include_ids: bool = False) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        payload: dict[str, Any] = {
            "mode": self.mode,
            "version_id": self.version_id,
            "parent_version": self.parent_version,
            "backend": self.backend,
            "metric": self.metric,
            "dimension": self.dimension,
            "provider": self.provider,
            "seen": self.seen,
            "cache_hits": self.cache_hits,
            "encoded": self.encoded,
            "batches": self.batches,
            "written": self.written,
            "unchanged": self.unchanged,
            "removed": self.removed,
            "failed": self.failed,
            "reuse_ratio": self.reuse_ratio,
            "ok": self.ok,
            "note": self.note,
            "failures": [list(item) for item in self.failures],
        }
        if include_ids:
            payload["count"] = self.ok_count
        return payload

    def summary_line(self) -> str:
        """人类可读的一行摘要.

        ``note`` 单独追加在末尾而不参与前面的字段：它是"这次构建为什么长这样"的
        一句解释（例如"变更比例 62% 超过阈值，已从 incremental 切到 full"）。
        把它塞进某个数字字段会让人以为那是一个度量。
        """
        return (
            f"{self.mode} 构建 | {self.version_id} ← {self.parent_version or '（首版）'} | "
            f"{self.seen} 条 → 写入 {self.written} / 未变 {self.unchanged} / 删除 {self.removed}"
            f" | 编码 {self.encoded}（命中 {self.cache_hits}，{self.batches} 批）"
            f" | 复用率 {self.reuse_ratio:.2%}"
            + (f" | 失败 {self.failed}" if self.failed else "")
            + (f" | {self.note}" if self.note else "")
        )


__all__ = [
    "ACTIONS",
    "ACTION_ADDED",
    "ACTION_REMOVED",
    "ACTION_UNCHANGED",
    "ACTION_UPDATED",
    "MANIFEST_VERSION",
    "VECTOR_KEY_LENGTH",
    "EmbeddingIdentity",
    "EncodeReport",
    "IndexEntry",
    "IndexManifest",
    "IndexPlan",
    "IndexingReport",
    "entries_digest",
    "index_version_id",
    "vector_key",
]
