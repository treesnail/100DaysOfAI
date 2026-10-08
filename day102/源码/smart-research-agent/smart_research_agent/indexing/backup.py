"""索引备份与恢复：把"还能回到上一版"从一句话变成一次拷贝（M6-D4）.

``types.py`` 让"上一版长什么样"可被重算，``versioning`` 让版本表能被回溯。
但清单只是一张**图**：它说得出"这一版该有哪些块、每个块的向量键是什么"，
却说不出"当时的向量在哪"。day064 的库（``FlatVectorStore`` 落盘的
``store.json`` 这一类文件）才是那份唯一数据，而它只有一份。

本模块补的就是这一份：在重建之前把库文件与清单**一起**拷成一份快照，
于是"重建把库写坏了"从一次事故变成一次 ``restore``。

```text
IndexManifest   上一版该有什么        （可重算的账）
backup          上一版实际有什么      （拷出来的文件）
restore         把后者放回原位        （本模块存在的理由）
```

## 备份不是日志：为什么必须有保留上限

日志可以无限追加，因为它的价值在"完整"；备份的价值在"能回去"，而
**能回去这件事只需要最近几份**。不设上限的备份目录会一直长到吃满磁盘，
而吃满磁盘的那一刻，正在写库的那次构建会失败——**备份把主流程搞挂了**，
这是备份机制最讽刺的失败方式。因此 ``create`` 之后总是自动 ``prune``，
``keep <= 0`` 直接报错：一个一份都不保留的备份机制是自相矛盾的。

## 两个身份字段的取舍

``backup_id`` 只由 ``(version_id, backend, 序号)`` 决定，**``created_at`` 不进 id**：

```text
时间戳进 id → 同一份内容在同一台机器上备份两次得到两个 id
            → "这两个备份其实是同一版"这件事再也说不清
            → 而清理与对账正是靠"哪些备份是同一版"来做的
```

代价是"两次创建得到同一个 id 前缀"，于是必须补一个**序号**（四位补零）
把它区分开；序号不随时间走，只随"这个目录里已经有多少份"走，
所以同一份内容备份两次得到 ``…-flat-0001`` 与 ``…-flat-0002``：
**id 不同、version_id 相同**——前者是存储位置，后者才是身份。
``created_at`` 只作为记录字段（由注入的 ``clock()`` 提供），它是给人看的。

## 谁依赖它

装配阶段的构建器（``builder``）与 ``/indexing/backup`` 端点：每次全量重建前
先 ``create`` 一份，重建后若校验不过就 ``restore``。本模块**不判断**该不该备份，
也不碰向量库——它只做"拷贝、登记、裁剪、放回"四件事，
因此它可以在没有向量提供方、没有网络的条件下被单独测试。
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from smart_research_agent.config import settings
from smart_research_agent.indexing.errors import BackupError, ManifestError
from smart_research_agent.indexing.types import IndexManifest

#: 备份目录的格式版本。**不匹配时拒读/拒恢复**（见 ``errors.BackupError``）：
#: 一个旧格式的快照被读进来一半，得到的是一个"看起来有数据"的库，
#: 而它的字段含义可能与当前代码不一致。
BACKUP_VERSION = 1

#: 备份账本的文件名（一行一条记录，追加写）。
BACKUP_INDEX_FILE = "backups.jsonl"

#: 保留份数的缺省值。实际缺省取 ``settings.indexing_backups_keep``，
#: 这个常量是"配置读不到时"的兜底，也是文档里引用的那个数。
DEFAULT_KEEP = 3

#: 每份备份里清单的文件名。刻意**不**从 ``manifest.MANIFEST_FILE`` 导入：
#: 本模块与 ``manifest.py`` 是并行开发的兄弟模块，备份要能在对方还没落地时
#: 独立跑起来——**契约文件（``types``/``errors``/``cache``）之外不互相 import**。
_MANIFEST_FILENAME = "manifest.json"

#: 备份 id 里序号的位数：四位补零，便于按字符串排序（``0002`` < ``0011``）。
_SEQ_WIDTH = 4


def utc_now_iso() -> str:
    """当前 UTC 时间的 ISO 字符串（**默认时钟**）.

    单独做成函数是为了能被注入：测试要断言 ``created_at`` 的具体取值，
    而"断言里带上真实时间"的测试只是在断言"机器的时间在走"。
    秒级精度足够——备份的时间戳用于报告排序与人工排查，不用于计算。
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_token(value: str) -> str:
    """把后端名清洗成可作目录名的一段（保住 id 能被直接当路径用）."""
    cleaned = "".join(
        char if (char.isalnum() or char in "-_") else "-" for char in value.strip()
    )
    return cleaned or "backend"


def _seq_of(backup_id: str) -> int:
    """从备份 id 里取回序号（取不到当 0：排序时排到最旧）."""
    tail = backup_id.rsplit("-", 1)[-1]
    try:
        return int(tail)
    except ValueError:
        return 0


@dataclass(frozen=True)
class BackupRecord:
    """一份备份的登记条目：它是账本里的一行，也能被 ``json.dumps`` 直接写下.

    ```text
    backup_id     存储位置（version_id + backend + 序号）
    version_id    它备份的是哪一版（**身份**，用于判断"两份是不是同一版"）
    dimension     这一版的向量维度（恢复时先看它对不对，比读文件便宜得多）
    files         拷进来的文件（相对名，排序后），清单文件不在其中
    size_bytes    这份备份占了多少字节（对账用）
    ```

    ``backup_version`` 不在规格的字段清单里，但它必须有一个**承载处**：
    ``load`` 要逐行校验、``restore`` 要在动手前校验，而清单文件本身
    （``IndexManifest``）不含这个字段。放进记录里，账本的一行就能自证格式。
    """

    backup_id: str
    version_id: str
    backend: str
    metric: str
    dimension: int
    created_at: str
    files: tuple[str, ...] = ()
    size_bytes: int = 0
    reason: str = ""
    backup_version: int = BACKUP_VERSION

    def __post_init__(self) -> None:
        if not self.backup_id.strip():
            raise BackupError("BackupRecord.backup_id 不能为空：没有 id 就无法定位这份备份")
        if self.dimension < 1:
            raise BackupError(
                f"BackupRecord.dimension 必须 >= 1，收到 {self.dimension}："
                "维度是恢复前最先要核对的东西，为 0 说明这条记录没被正确填上"
            )
        if self.size_bytes < 0:
            raise BackupError(f"BackupRecord.size_bytes 不能为负，收到 {self.size_bytes}")

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典（账本一行就是它的 JSON）."""
        return {
            "backup_version": self.backup_version,
            "backup_id": self.backup_id,
            "version_id": self.version_id,
            "backend": self.backend,
            "metric": self.metric,
            "dimension": self.dimension,
            "created_at": self.created_at,
            "files": list(self.files),
            "size_bytes": self.size_bytes,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BackupRecord:
        """从账本里的一行还原（缺字段时给出可读的 ``BackupError``）."""
        required = ("backup_id", "version_id", "backend", "metric", "dimension", "created_at")
        missing = [name for name in required if name not in payload]
        if missing:
            raise BackupError(
                f"备份记录缺少字段 {missing}：账本的一行必须能独立还原出一份备份的位置与身份，"
                "缺任何一项这条记录都无法用于恢复"
            )
        return cls(
            backup_id=str(payload["backup_id"]),
            version_id=str(payload["version_id"]),
            backend=str(payload["backend"]),
            metric=str(payload["metric"]),
            dimension=int(payload["dimension"]),
            created_at=str(payload["created_at"]),
            files=tuple(str(item) for item in payload.get("files", ())),
            size_bytes=int(payload.get("size_bytes", 0)),
            reason=str(payload.get("reason", "")),
            backup_version=int(payload.get("backup_version", BACKUP_VERSION)),
        )

    def summary_line(self) -> str:
        """人类可读的一行摘要（报告里逐行打印）."""
        tail = f" | {self.reason}" if self.reason else ""
        return (
            f"备份 {self.backup_id} | 版本 {self.version_id} | "
            f"{self.backend}/{self.metric} {self.dimension}d | "
            f"{len(self.files)} 个文件 / {self.size_bytes} B | {self.created_at}{tail}"
        )


def _resolve_keep(keep: int | None) -> int:
    """缺省保留份数取 ``settings.indexing_backups_keep``（显式值优先）.

    规格把 ``keep`` 的缺省写成常量 ``DEFAULT_KEEP``，同时要求"缺省取
    ``settings.indexing_backups_keep``"。两者用同一个默认值不可能同时成立
    （改配置就无效了），因此这里用 ``None`` 作哨兵：**只有显式传值才覆盖配置**。
    """
    if keep is not None:
        return int(keep)
    return int(settings.indexing_backups_keep)


class IndexBackupStore:
    """一个目录里的若干份快照：创建、登记、裁剪、恢复.

    目录布局（``path`` 是**基准目录**）：

    ```text
    <path>/backups.jsonl          账本：一行一次 create（追加写）
    <path>/<backup_id>/           一份快照：源文件 + manifest.json
    ```

    账本是追加写的，快照是可删的：``prune`` 只删目录，账本里的那一行留着。
    于是"这份备份曾经存在过、现在没了"在账本上是可读的事实，
    而不是一个需要靠时间戳猜的空白。**可用性由目录是否存在判定**，
    这也正是 ``load`` 的返回值语义（见下）。
    """

    def __init__(
        self,
        *,
        path: str = "",
        keep: int | None = None,
        clock: Callable[[], str] = utc_now_iso,
    ) -> None:
        self._path = path
        self._keep = _resolve_keep(keep)
        if self._keep <= 0:
            raise BackupError(
                f"备份保留份数必须 >= 1，收到 {self._keep}："
                "一个不保留任何备份的备份机制是自相矛盾的——"
                "它在报告里显示'备份成功'，而真出事时没有任何东西可以回去"
            )
        self._clock = clock
        self._records: list[BackupRecord] = []

    # ------------------------------------------------------------------ 属性

    @property
    def path(self) -> str:
        """基准目录（空串表示还没落过盘）."""
        return self._path

    @property
    def keep(self) -> int:
        """保留份数上限（``create`` 之后自动裁剪到这个数）."""
        return self._keep

    # ------------------------------------------------------------------ 内部

    def _base_dir(self) -> Path:
        """取基准目录并确保它存在（空路径即拒绝）."""
        if not self._path.strip():
            raise BackupError(
                "备份没有位置：构造时没给 path。"
                "备份必须落在**库里之外**的目录上，否则'把库写坏'会连备份一起带走"
            )
        base = Path(self._path)
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _dir_of(self, record: BackupRecord) -> Path:
        return Path(self._path) / record.backup_id

    def _index_path(self, path: str | None) -> Path:
        """把 ``path`` 解析成账本文件的位置.

        规则只有一条：**名为 ``backups.jsonl`` 的路径被当成文件，其余都当成目录**。
        这条规则是刻意的——目录与文件两种写法都有人用，
        而"看后缀猜"会在一个叫 ``index_v2`` 的目录上猜错。
        """
        raw = path if path is not None else ""
        if not raw and self._path:
            raw = str(Path(self._path) / BACKUP_INDEX_FILE)
        if not str(raw).strip():
            raise BackupError("备份账本没有位置：构造与调用都没有给 path")
        target = Path(raw)
        if target.name == BACKUP_INDEX_FILE:
            return target
        return target / BACKUP_INDEX_FILE

    def _next_seq(self) -> int:
        """下一个序号 = 现有最大序号 + 1.

        **不能用"现有份数 + 1"**：裁剪之后份数会变小，于是下一次创建会得到
        一个**仍然存在**的 id，把上一份快照原地覆盖掉。
        用最大序号 + 1，序号只增不减，路径永不重号。
        """
        return max((_seq_of(record.backup_id) for record in self._records), default=0) + 1

    def _resolve_sources(self, source_files: Sequence[str]) -> list[tuple[Path, str]]:
        """把源文件列表校验并解析成 ``(源路径, 备份内文件名)``.

        **先全部校验，再开始拷**：源文件缺一个就在动手前拒绝。
        "少备份一个文件而报告说成功"是备份机制最不该有的行为，
        而"拷到一半才发现第三个源文件不在"会留下一个半成品目录——
        它看起来像一份可用的备份，其实缺文件。
        """
        if isinstance(source_files, str | bytes):
            raise BackupError(
                "source_files 要的是路径序列，收到的是单个字符串："
                "把它当成路径序列会逐字符去建目录，而报错会发生在很后面"
            )
        resolved: list[tuple[Path, str]] = []
        seen: set[str] = set()
        for raw in source_files:
            source = Path(raw)
            if not source.exists():
                raise BackupError(
                    f"备份源文件不存在：{source}。"
                    "不静默跳过：缺一个文件的备份会在恢复后表现为'库少了几条记录'，"
                    "而那要等到检索结果变差才会被发现"
                )
            if not source.is_file():
                raise BackupError(f"备份源不是文件：{source}（目录请由调用方展开成文件列表）")
            name = source.name
            if name == _MANIFEST_FILENAME:
                raise BackupError(
                    f"源文件不能叫 {_MANIFEST_FILENAME!r}：这个位置留给清单本身，"
                    "撞名会让恢复出来的清单覆盖掉源数据"
                )
            if name in seen:
                raise BackupError(
                    f"有两个源文件同名 {name!r}：备份目录是平铺的，"
                    "同名会互相覆盖——请由调用方先把它们改名或放进各自的子目录"
                )
            seen.add(name)
            resolved.append((source, name))
        return resolved

    def _append_index(self, record: BackupRecord) -> None:
        """把一条记录追加到账本（一次写一行）."""
        base = self._base_dir()
        line = json.dumps(record.to_dict(), ensure_ascii=False)
        with (base / BACKUP_INDEX_FILE).open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    # ------------------------------------------------------------------ 创建

    def create(
        self,
        *,
        source_files: Sequence[str],
        manifest: IndexManifest,
        reason: str = "",
    ) -> BackupRecord:
        """把 ``source_files`` 与 ``manifest`` 拷成一份快照并登记（返回新记录）.

        四步，顺序不能换：

        ```text
        1. verify_version_id()   清单先自证：拿一份被改过的清单去备份，
                                 等于把"错误的上一版"固化下来
        2. 校验全部源文件        见 _resolve_sources（不做半成品备份）
        3. 拷贝 + 写 manifest.json + 追加账本
        4. prune()               备份不是日志，保留份数有上限
        ```

        ``size_bytes`` 是这份快照目录里**所有文件**的字节数之和（源文件 + 清单）。
        它用于对账（"这一版有多大"）而不是用于配额，所以直接量目录。
        """
        base = self._base_dir()
        manifest.verify_version_id()
        sources = self._resolve_sources(source_files)

        backup_id = f"{manifest.version_id}-{_safe_token(manifest.backend)}-{self._next_seq():04d}"
        target = base / backup_id
        target.mkdir(parents=True, exist_ok=True)
        for source, name in sources:
            shutil.copy2(source, target / name)
        (target / _MANIFEST_FILENAME).write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        size_bytes = sum(
            item.stat().st_size for item in sorted(target.iterdir()) if item.is_file()
        )
        record = BackupRecord(
            backup_id=backup_id,
            version_id=manifest.version_id,
            backend=manifest.backend,
            metric=manifest.metric,
            dimension=manifest.identity.dimension,
            created_at=self._clock(),
            files=tuple(sorted(name for _, name in sources)),
            size_bytes=size_bytes,
            reason=reason,
            backup_version=BACKUP_VERSION,
        )
        self._records.append(record)
        self._append_index(record)
        self.prune()
        return record

    # ------------------------------------------------------------------ 查询

    def list(self) -> list[BackupRecord]:
        """全部登记过的备份，**新 → 旧**.

        ``list()`` 给的是**账本**：目录被 ``prune`` 删掉的记录也还在，
        因为"这一版曾经备份过"是一条真实发生过的事实。
        "现在还能恢复哪几份"要问 ``load`` 的返回值或 ``latest()``。
        """
        return sorted(self._records, key=lambda record: _seq_of(record.backup_id), reverse=True)

    def latest(self) -> BackupRecord | None:
        """最近一份**目录仍在**的备份；一份都没有时返回 ``None``.

        刻意跳过目录已丢失的记录：``restore(latest())`` 是调用方最自然的写法，
        让 ``latest()`` 返回一个必然失败的记录，会把"备份目录被人删了"
        伪装成"恢复功能坏了"。
        """
        for record in self.list():
            if self._dir_of(record).is_dir():
                return record
        return None

    def get(self, backup_id: str) -> BackupRecord | None:
        """按 id 取记录（目录是否还在都返回；**没有**则 ``None``）."""
        for record in self._records:
            if record.backup_id == backup_id:
                return record
        return None

    def read_manifest(self, backup_id: str) -> IndexManifest:
        """只读地取出某份备份里的清单——"我想看看那一版是什么，但不想动现在的库".

        这条路径是恢复的**校验前置**：能读出 + ``verify_version_id()`` 通过，
        才说明这份快照自洽。清单损坏一律转成 ``BackupError``：
        ``ManifestError`` 的修复建议是"请重新构建清单"，而对一份快照而言，
        正确的动作是"换一份备份"或"承认这份备份坏了"，两者的处置不同。
        """
        record = self.get(backup_id)
        if record is None:
            raise BackupError(f"没有备份 {backup_id!r}：账本里查不到这个 id，无法读出清单")
        manifest_path = self._dir_of(record) / _MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise BackupError(
                f"备份 {backup_id!r} 里找不到 {_MANIFEST_FILENAME}："
                "没有清单的快照说不清它备份的是哪一版，不能用于恢复"
            )
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise BackupError(
                f"备份 {backup_id!r} 的清单不是合法 JSON（{exc}）："
                "文件可能被改过或写到一半，请换一份备份"
            ) from exc
        if not isinstance(payload, dict):
            raise BackupError(f"备份 {backup_id!r} 的清单必须是对象（键 → 值）")
        try:
            manifest = IndexManifest.from_dict(payload)
            manifest.verify_version_id()
        except ManifestError as exc:
            raise BackupError(
                f"备份 {backup_id!r} 的清单与内容不一致：{exc}"
            ) from exc
        if manifest.version_id != record.version_id:
            raise BackupError(
                f"备份 {backup_id!r} 的账本记录是版本 {record.version_id}，"
                f"清单里是 {manifest.version_id}："
                "两者不符说明目录被人工改过，这份快照的身份不可信"
            )
        return manifest

    # ------------------------------------------------------------------ 恢复

    def restore(self, backup_id: str, *, target_dir: str) -> BackupRecord:
        """把某份快照里的文件拷回 ``target_dir``，返回该记录.

        三条纪律：

        ```text
        先校验  清单能读出且 verify_version_id() 通过；格式版本必须是 BACKUP_VERSION
        不清理  目标目录里已有的其它文件**不动**（避免"恢复备份顺带清库"）
        拷清单  清单本身也拷回去——恢复出来的库如果没有账，下一次增量构建
                会把它当成"没有上一版"而整库重算
        ```
        """
        record = self.get(backup_id)
        if record is None:
            raise BackupError(f"没有备份 {backup_id!r}：账本里查不到这个 id，无法恢复")
        if record.backup_version != BACKUP_VERSION:
            raise BackupError(
                f"备份 {backup_id!r} 的格式版本是 {record.backup_version}，"
                f"本代码只认识 {BACKUP_VERSION}：拒绝恢复。"
                "用旧格式的快照恢复出一半，会得到一个'看起来有数据'的库，"
                "而它的字段含义可能与当前代码不一致——那种库比空库更难发现"
            )
        source_dir = self._dir_of(record)
        if not source_dir.is_dir():
            raise BackupError(
                f"备份 {backup_id!r} 的目录不存在：{source_dir}。"
                "账本里有记录不代表目录还在（prune 或人工删除都会这样）"
            )
        self.read_manifest(backup_id)
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        for item in sorted(source_dir.iterdir()):
            if item.is_file():
                shutil.copy2(item, target / item.name)
        return record

    # ------------------------------------------------------------------ 裁剪

    def prune(self, keep: int | None = None) -> list[str]:
        """删掉最旧的若干份，返回被删的 ``backup_id``（**从最旧开始**）.

        只删目录，不改账本：账本是"曾经发生过什么"的记录，
        而裁剪是"现在还留着什么"的决策，两者不是一回事。
        """
        limit = self._keep if keep is None else int(keep)
        if limit <= 0:
            raise BackupError(
                f"保留份数必须 >= 1，收到 {limit}："
                "把备份全删掉不是一次清理，而是一次不可逆的数据删除"
            )
        ordered = self.list()
        stale = ordered[limit:]
        if not stale:
            return []
        base = Path(self._path) if self._path else None
        for record in stale:
            if base is not None:
                shutil.rmtree(base / record.backup_id, ignore_errors=True)
        removed = {record.backup_id for record in stale}
        self._records = [record for record in self._records if record.backup_id not in removed]
        return [record.backup_id for record in reversed(stale)]

    # ------------------------------------------------------------------ 落盘

    def persist(self, path: str | None = None) -> str:
        """把账本**重写**成 JSONL，返回实际写入的路径.

        为什么是 JSONL 而不是一个大 JSON 数组：追加一份备份是**一次写一行**，
        中途失败只会丢掉最后一行，历史文件仍然可读；
        而重写一个 JSON 数组时断电，得到的是一个语法不完整的文件
        ——**账本读不出来，等于所有备份都定位不到**。
        """
        index = self._index_path(path)
        index.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(self._records, key=lambda record: _seq_of(record.backup_id))
        lines = [json.dumps(record.to_dict(), ensure_ascii=False) for record in ordered]
        index.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        self._path = str(index.parent)
        return str(index)

    def load(self, path: str | None = None) -> int:
        """逐行读回账本，返回**实际可用的备份份数**.

        返回"可用份数"而不是"记录条数"是刻意的取舍：

        ```text
        记录说存在但目录没了 → 不报错（prune 与人工删除都会造成这种状态，
                              而它不该让整个账本读不出来）
                             但必须**如实计入返回值**——
                             否则 load() 说"读了 3 份"，restore 却只成功 2 次
        ```

        账本本身仍然完整读回（``list()`` / ``get()`` 能看到那些记录），
        所以"这一版曾经备份过"这条事实不会丢。格式版本不符则**拒读**。
        """
        index = self._index_path(path)
        if not index.is_file():
            return 0
        records: list[BackupRecord] = []
        for lineno, raw in enumerate(index.read_text(encoding="utf-8").splitlines(), start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise BackupError(f"备份账本第 {lineno} 行不是合法 JSON：{exc}") from exc
            if not isinstance(payload, dict):
                raise BackupError(f"备份账本第 {lineno} 行必须是对象")
            try:
                version = int(payload.get("backup_version", 0))
            except (TypeError, ValueError) as exc:
                raise BackupError(
                    f"备份账本第 {lineno} 行的 backup_version 不是整数："
                    f"{payload.get('backup_version')!r}"
                ) from exc
            if version != BACKUP_VERSION:
                raise BackupError(
                    f"备份账本第 {lineno} 行的格式版本是 {version}，"
                    f"本代码只认识 {BACKUP_VERSION}：拒读。"
                    "账本里混着两种格式时，'哪几份能恢复'会变成一个逐行去看的问题"
                )
            records.append(BackupRecord.from_dict(payload))
        self._records = records
        self._path = str(index.parent)
        base = Path(self._path)
        return sum(1 for record in records if (base / record.backup_id).is_dir())


__all__ = [
    "BACKUP_INDEX_FILE",
    "BACKUP_VERSION",
    "DEFAULT_KEEP",
    "BackupRecord",
    "IndexBackupStore",
    "utc_now_iso",
]
