"""语料侧的快照：把"一个目录现在长什么样"变成一份可以 diff 的账（M6-D10 / day072）.

day061 的 ``documents`` 回答"**一份文件**是什么"，day065 的 ``indexing`` 回答
"**一批记录**与上一版差在哪"。本模块补的是它们之间那一层：

```text
documents.iter_supported_files(root)   目录里都有哪些文件（确定性的排序）
        ↓ LoaderRegistry.load_path     逐份解析成 Document（day061 的 5 个加载器）
        ↓ Document.fingerprint         内容身份（sha256(正文)[:16]）
CorpusSnapshot                         这一次扫盘的账：路径 + 指纹 + 字数 + 媒体类型
```

## 为什么这一层的粒度是"文件"，而不是"块"

day065 的 ``planner`` 比的是**块**（``chunk_id + fingerprint + vector_key``），
它回答"哪些块的向量要重算"。本模块比的是**文件**，它回答另外三个问题：

```text
① 要不要跑这一趟？        一份都没变 → 一行都不写（省下的是一次整库扫描的开销）
② 语料侧到底改了什么？     运维视角的账：新增几份、改了几份、删了几份
③ 这次改动有多大？        变更比例 = 要不要从"增量"换成"整库重建"的判据
```

两本账的**差额**本身就是一条有用的读数：一次"改了一份 4000 字的文档"
在文件级是 ``sources_changed = 1``，在块级却可能是 ``chunks_written = 14``。
只看前者会以为"改动很小"，只看后者会以为"语料大改"。

## 三条纪律

1. **来源路径一律是相对的 posix 风格**（``docs/a.md``）。
   绝对路径会让同一份语料在两台机器上算出两个快照号，于是"变了"这件事
   变成一个永远为真的判定——而那看起来只是"调度很勤快"（见 ``types`` 的校验）。
2. **快照号只由（路径，指纹）算出**，不含时间、不含机器、不含 skipped。
   与 day065 的"能重算的才叫版本号"是同一条纪律。
3. **读不进来的文件默认让整次扫描失败**（``strict=True``）。
   静默跳过一份文件的后果是"库里少一条记录"，而它没有任何症状指向那个文件；
   要在生产里放宽，请显式 ``strict=False``——那时它会进 ``snapshot.skipped``，
    **留在账上**，而不是消失。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from smart_research_agent.documents.base import LoaderRegistry, default_registry
from smart_research_agent.documents.errors import DocumentError
from smart_research_agent.documents.pipeline import iter_supported_files
from smart_research_agent.documents.types import Document
from smart_research_agent.rag_ops.errors import SyncError
from smart_research_agent.rag_ops.types import (
    CorpusSnapshot,
    SourceEntry,
    to_iso,
    utc_now,
)


def resolve_source_root(root: str | Path) -> Path:
    """把语料根目录解析成一个绝对路径，并检查它**确实是个目录**.

    检查放在这里而不是等 ``iter_supported_files`` 抛 ``DocumentError``：
    两族的处置人不同（``DocumentError`` 是"这份文件有问题"，
    本模块要说的是"整个语料目录没配对"），而端点的 400 通道只认 ``RagOpsError``。
    消息里带上当前配置项名字，是因为最常见的成因就是它没配。

    ``""`` 与 ``"."`` 都被拒绝：``Path("")`` 在 Python 里等于 ``Path(".")``，
    于是"没配源目录"会静默变成"拿进程当前目录当语料"——
    在仓库根目录跑一次同步，``pyproject.toml``、测试文件与文档会被一起灌进库里，
    而它们全都是合法的 markdown 与文本。
    """
    raw = str(root or "").strip()
    if not raw or raw == ".":
        raise SyncError(
            f"语料根目录不能是空串或 '.'（收到 {raw!r}）：请显式给一个目录，"
            "或配置 rag_ops_source_dir——否则同步会拿进程当前目录当语料。"
        )
    path = Path(raw)
    if not path.exists():
        raise SyncError(
            f"语料根目录 {str(path)!r} 不存在："
            "请先创建它（并把要入库的文档放进去），或把 rag_ops_source_dir 指向正确的位置。"
        )
    if not path.is_dir():
        raise SyncError(f"语料根目录 {str(path)!r} 不是目录。")
    return path


def relative_source(path: Path, root: Path) -> str:
    """把绝对路径折成相对语料根目录的 posix 路径（``docs/a.md``）.

    ``as_posix()`` 不是装饰：Windows 上扫出来的 ``docs\\a.md`` 一旦进快照，
    在 Linux 上就再也匹配不上同一份文件（而重复入库与漏更新都不报错）。
    """
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:  # pragma: no cover - iter_supported_files 保证在 root 之下
        raise SyncError(
            f"文件 {str(path)!r} 不在语料根目录 {str(root)!r} 之下，无法算出相对路径。"
        ) from exc


def entry_from_document(document: Document, *, source: str) -> SourceEntry:
    """把一份 ``Document`` 折成一条快照条目（媒体类型空时留空，不猜）."""
    return SourceEntry(
        source=source,
        fingerprint=document.fingerprint,
        char_count=document.char_count,
        media_type=document.media_type,
    )


def scan_corpus(
    root: str | Path,
    *,
    registry: LoaderRegistry | None = None,
    suffixes: Sequence[str] | None = None,
    limit: int | None = None,
    strict: bool = True,
    clock: datetime | None = None,
) -> tuple[CorpusSnapshot, tuple[Document, ...]]:
    """扫一个目录，返回 ``(快照, 文档序列)``——**文件只读一遍**.

    分两次读（先扫快照、再扫文档）看起来更清晰，但代价是每个文件被解析两遍；
    而这里返回的 ``Document`` 正是下一步要喂给分块器的东西，因此一次读完。

    ```text
    strict=True   某份文件读不进来 → 当场抛 SyncError（消息里带路径与原因）
    strict=False  跳过它，并把 (路径, 原因) 记进 snapshot.skipped —— **留在账上**
    limit         只处理排序后的前 N 份（None / 0 = 不限），用于演示与灰度
    ```

    ``clock`` 只影响快照里的 ``created_at``（**不进快照号**）：
    这个参数存在是为了让报告里的时间可复现，而不是让判定依赖它。
    """
    base = resolve_source_root(root)
    resolved_registry = registry if registry is not None else default_registry()
    try:
        paths = iter_supported_files(base, suffixes=suffixes)
    except DocumentError as exc:
        raise SyncError(
            f"扫描语料根目录 {str(base)!r} 失败：{exc}"
        ) from exc

    capped = paths if not limit else paths[: int(limit)]
    entries: list[SourceEntry] = []
    documents: list[Document] = []
    skipped: list[tuple[str, str]] = []
    for path in capped:
        source = relative_source(path, base)
        try:
            document = resolved_registry.load_path(path)
        except DocumentError as exc:
            if strict:
                raise SyncError(
                    f"来源 {source!r} 读不进来：{exc}"
                    "（strict=True 时一份读不进来会让整次扫描失败——"
                    "静默跳过的后果是库里少一条记录，而它没有任何症状指向这个文件；"
                    "确实要跳过请显式传 strict=False）"
                ) from exc
            skipped.append((source, str(exc)))
            continue
        entries.append(entry_from_document(document, source=source))
        documents.append(document)

    moment = clock if clock is not None else utc_now()
    snapshot = CorpusSnapshot(
        entries=tuple(entries),
        skipped=tuple(skipped),
        created_at=to_iso(moment),
    )
    return snapshot, tuple(documents)


def scan_sources(
    root: str | Path,
    *,
    registry: LoaderRegistry | None = None,
    suffixes: Sequence[str] | None = None,
    limit: int | None = None,
    strict: bool = True,
    clock: datetime | None = None,
) -> CorpusSnapshot:
    """只取快照（不要文档）——**报告与对账只需要身份，不需要正文**.

    它仍然要把每份文件读一遍：内容指纹是对正文算的
    （``documents.types.content_id``）。要连文档一起用，请用 :func:`scan_corpus`，
    它会顺手把结果给你，而不用再解析一遍。
    """
    snapshot, _ = scan_corpus(
        root,
        registry=registry,
        suffixes=suffixes,
        limit=limit,
        strict=strict,
        clock=clock,
    )
    return snapshot


__all__ = [
    "entry_from_document",
    "relative_source",
    "resolve_source_root",
    "scan_corpus",
    "scan_sources",
]
