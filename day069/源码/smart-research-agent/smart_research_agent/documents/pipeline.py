"""批量入库：把"一堆文件"变成"一份可复核的文档集合"（M6-D1）.

单个文件的解析是 ``loader`` 的事，本模块处理的是**批量**特有的三个问题：

```text
1. 去重     同一份内容出现两次（复制、重下、导出）时保留哪一份
2. 报错     一份文件解析不了时，这一批是整体失败还是继续
3. 排序     同一批文件在不同机器上必须得到同一份结果
```

## 去重按 ``doc_id``（内容指纹），保留"第一个"

```text
先看到的留下，后看到的标记 duplicate_of=<先看到的那份的 source>
```

"先看到"的定义必须是**确定的**，否则去重结果会随文件系统的遍历顺序变化：
因此 ``ingest_dir`` 会把文件列表**排序之后**再处理（``sorted``，字典序）。
这不是洁癖：day057 的"可复现版本链"、day058 的"确定性 run_id"、
day060 的"确定性分桶"都在同一个位置上——**同一份输入要在任何机器上
得到同一份输出**，否则"这次入库比上次少了三个文档"永远说不清为什么。

## 报错分两类，而且只有一类被吞掉

```text
DocumentError（含 UnsupportedDocument）  记进报告的 error 字段，这一批继续
其余异常                                  **向上抛出**
```

第二行是刻意的：把"我们自己的 bug"记成报告里的一行错误，会让它
永远不被修——因为报告看起来"只是一份文件失败了"。这与 day059 把
``blocked`` 与 ``failed`` 分开、day060 把"未知机型"与"没用过这个机型"
分开是同一条纪律：**"做不到"要被记成结论，"出错了"要响亮。**

## 与知识库的接缝

``IngestReport.knowledge_records()`` 产出的形状就是 day009 的
``VectorStore.add`` 需要的东西（``doc_id`` / ``text`` / ``metadata``）。
day062 会在它上面加一层分块：**今天交出去的是整份文档，
明天交出去的是文档切出来的片段**，而接口形状不变。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from smart_research_agent.documents.base import (
    MEDIA_TYPES,
    SUFFIX_MEDIA_TYPES,
    LoaderRegistry,
    default_registry,
)
from smart_research_agent.documents.errors import DocumentError
from smart_research_agent.documents.types import BLOCK_KINDS, Document
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: 入库这件事**做不到什么**。每条都带一个可核对的依据——
#: 与 day059 模型卡、day060 部署限制同一条纪律：
#: **无法否掉结论的限制等于免责声明。**
DOCUMENTS_LIMITATIONS: tuple[str, ...] = (
    "PDF 抽取不解析 CID 字体的字形映射（规范 §9.10.2 判定为不可确定），"
    "因此 Identity-H 编码的中文 PDF 会抽到乱码或空文本——元数据里的 "
    "`pdf_cid_font_detected=true` 是唯一的提示",
    "只管 FlateDecode 一种流过滤器：LZW / ASCIIHex / DCTDecode（扫描件）"
    "全部走不到，表现为「解析成功但段落数为 0」",
    "DOCX 只读 word/document.xml：页眉页脚、脚注、批注、文本框里的内容"
    "**不会被读到**，而「没读到」与「文档里没有」在结果里长得一样——"
    "只能靠元数据的 docx_parts_not_read 分辨",
    "编码探测在 utf-8 与 gb18030 都不成立时回退 latin-1 并标记 "
    "`encoding_confident=false`：此时中文已是乱码，而乱码会一路走进索引",
    "只做行距分段这一个版面推断（PDF 32000-1 里没有「段落」这个概念），"
    "双栏与表格的阅读顺序不做还原",
)

#: 明确排除的用途（比"不适用于生产"具体得多）。
DOCUMENTS_OUT_OF_SCOPE: tuple[str, ...] = (
    "直接按本课的抽取结果做法律或财务文档的处理：PDF 的阅读顺序与表格还原"
    "都不可靠，需要人工复核",
    "把入库当作「文件已备份」：本模块只读不写，且不做任何完整性校验",
    "用本模块替代 pypdf / pdfplumber / python-docx 这类生产级库："
    "不复用它们只为了保证离线零依赖，不是因为在质量上等价",
)


@dataclass(frozen=True)
class IngestEntry:
    """一个文件的入库结果（成功 / 重复 / 不被支持，三种状态各有一个字段）."""

    source: str
    media_type: str = ""
    doc_id: str = ""
    char_count: int = 0
    block_count: int = 0
    duplicate_of: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        """解析成功（**重复也算成功**：重复是内容级的结论，不是失败）."""
        return not self.error

    @property
    def is_duplicate(self) -> bool:
        """是否与前一份文档内容相同."""
        return bool(self.duplicate_of) and not self.error

    @property
    def status(self) -> str:
        """三态之一：``ok`` / ``duplicate`` / ``error``.

        重复**单独一态**而不是并入 ``ok``：报告要能回答"这一批里有几份
        是重复内容"——那是一个关于数据治理的问题，不是一个关于失败的问题。
        """
        if self.error:
            return "error"
        return "duplicate" if self.is_duplicate else "ok"

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        payload = asdict(self)
        payload["status"] = self.status
        return payload

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        if self.error:
            return f"[error] {self.source} —— {self.error}"
        if self.is_duplicate:
            return f"[重复] {self.source}（与 {self.duplicate_of} 内容相同）"
        return (
            f"[ok] {self.source} | {self.media_type} | {self.char_count} 字符 / "
            f"{self.block_count} 块 | {self.doc_id}"
        )


@dataclass(frozen=True)
class IngestReport:
    """一批文件的入库报告（逐条结果 + 去重后的文档集合）."""

    entries: tuple[IngestEntry, ...] = ()
    documents: tuple[Document, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def files(self) -> int:
        """这一批处理了多少个文件."""
        return len(self.entries)

    def status_counts(self) -> dict[str, int]:
        """按状态计数（**三个键恒存在**，为 0 也在——报告的表头不该随内容变化）."""
        counts = {"ok": 0, "duplicate": 0, "error": 0}
        for entry in self.entries:
            counts[entry.status] += 1
        return counts

    @property
    def unique_documents(self) -> int:
        """去重后的文档数（= 会真正进知识库的数量）."""
        return len(self.documents)

    @property
    def total_chars(self) -> int:
        """去重后的总字符数（**不是文件大小的合计**，是解析出来的正文长度）."""
        return sum(document.char_count for document in self.documents)

    def block_counts(self) -> dict[str, int]:
        """按块类型汇总（六个键恒存在，为 0 也在）."""
        counts = dict.fromkeys(BLOCK_KINDS, 0)
        for document in self.documents:
            for kind, value in document.block_counts().items():
                counts[kind] += value
        return counts

    def media_type_counts(self) -> dict[str, int]:
        """按媒体类型汇总去重后的文档数."""
        counts = {media_type: 0 for media_type in MEDIA_TYPES}
        for document in self.documents:
            counts[document.media_type] = counts.get(document.media_type, 0) + 1
        return counts

    def errors(self) -> list[IngestEntry]:
        """失败条目（"哪些文件解析不了"是这批数据的第一手画像）."""
        return [entry for entry in self.entries if entry.error]

    def duplicates(self) -> list[IngestEntry]:
        """重复条目（每一份都能指回它重复的是哪一份）."""
        return [entry for entry in self.entries if entry.is_duplicate]

    def knowledge_records(self) -> list[dict[str, Any]]:
        """产出可交给知识库的记录（day009 的 ``VectorStore`` 形状）.

        形状只有四个字段，而且**刻意不做分块**：分块是 day062 的事。
        今天交出去的是"整份文档"，明天起交出去的是"文档切出来的片段"，
        而**接口形状不变**——这正是先做归一化的价值。
        """
        return [
            {
                "doc_id": document.fingerprint,
                "source": document.source,
                "text": document.text,
                "metadata": {
                    "media_type": document.media_type,
                    "char_count": str(document.char_count),
                    "block_count": str(len(document.blocks)),
                    **dict(document.metadata),
                },
            }
            for document in self.documents
        ]

    def to_dict(self, *, include_documents: bool = False) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        payload: dict[str, Any] = {
            "files": self.files,
            "unique_documents": self.unique_documents,
            "total_chars": self.total_chars,
            "status_counts": self.status_counts(),
            "block_counts": self.block_counts(),
            "media_type_counts": self.media_type_counts(),
            "metadata": dict(self.metadata),
            "entries": [entry.to_dict() for entry in self.entries],
        }
        if include_documents:
            payload["documents"] = [
                document.to_dict(include_blocks=False) for document in self.documents
            ]
        return payload

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        counts = self.status_counts()
        return (
            f"{self.files} 个文件 → {self.unique_documents} 份文档 | "
            f"ok {counts['ok']} / 重复 {counts['duplicate']} / 失败 {counts['error']} | "
            f"{self.total_chars} 字符"
        )

    def render_markdown(self) -> str:
        """把报告渲染成 markdown（可直接贴进数据治理工单）."""
        counts = self.status_counts()
        lines = [
            "# 文档入库报告",
            "",
            f"- 文件数：{self.files}",
            f"- 去重后文档数：{self.unique_documents}",
            f"- 正文总字符数：{self.total_chars}",
            f"- 状态：ok {counts['ok']} / 重复 {counts['duplicate']} / 失败 {counts['error']}",
            "",
            "| 媒体类型 | 文档数 |",
            "|---------|-------|",
        ]
        lines.extend(
            f"| `{media_type}` | {count} |"
            for media_type, count in sorted(self.media_type_counts().items())
            if count
        )
        lines.extend(["", "| 块类型 | 数量 |", "|--------|------|"])
        lines.extend(
            f"| `{kind}` | {count} |" for kind, count in self.block_counts().items()
        )
        lines.extend(["", "## 逐条结果", "", "| 状态 | 来源 | 字符 | 块 | 说明 |", "|------|------|------|----|------|"])
        lines.extend(
            f"| {entry.status} | `{entry.source}` | {entry.char_count} | "
            f"{entry.block_count} | {entry.error or entry.duplicate_of or '—'} |"
            for entry in self.entries
        )
        lines.append("")
        return "\n".join(lines)


def iter_supported_files(
    root: str | Path, *, suffixes: Sequence[str] | None = None
) -> list[Path]:
    """列出目录下可解析的文件（**排序后返回**，见模块 docstring 的第 3 条）.

    默认只收注册表认识的后缀，因此 ``.git`` 里的文件、``.DS_Store``、
    图片与视频都不会进来——但**目录本身会被递归进去**。
    隐藏目录（以 ``.`` 开头）会被跳过：那里通常是工具的缓存。
    """
    base = Path(root)
    if not base.is_dir():
        raise DocumentError(f"{base} 不是一个目录")
    # 不给 suffixes 时按**注册表认识的后缀**过滤，而不是"全部收下"：
    # 一个目录里通常还躺着图片、压缩包与 .git 里的东西，把它们都读进来
    # 只会得到一屏"不在支持范围"的错误，而真正的信号会被淹没。
    if suffixes is None:
        wanted = set(SUFFIX_MEDIA_TYPES)
    else:
        wanted = {suffix.lower() for suffix in suffixes}
    found: list[Path] = []
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.relative_to(base).parts):
            continue
        if path.suffix.lower() not in wanted:
            continue
        found.append(path)
    return sorted(found)


class DocumentIngestor:
    """批量入库器：一个注册表 + 一次遍历 + 一份报告."""

    def __init__(self, registry: LoaderRegistry | None = None) -> None:
        self.registry = registry or default_registry()

    def _entry_for(self, document: Document) -> IngestEntry:
        """由文档构造一条成功条目."""
        return IngestEntry(
            source=document.source,
            media_type=document.media_type,
            doc_id=document.fingerprint,
            char_count=document.char_count,
            block_count=len(document.blocks),
        )

    def ingest_bytes(self, items: Sequence[tuple[str, bytes]]) -> IngestReport:
        """解析一批内存字节（``(来源标签, 内容)`` 的有序序列）.

        顺序**就是**优先级：先出现的那一份在去重时被留下。因此调用方
        传进来的顺序必须自己是确定的（``ingest_dir`` 用排序保证这一点）。
        """
        entries: list[IngestEntry] = []
        kept: dict[str, Document] = {}
        owner: dict[str, str] = {}
        for source, data in items:
            try:
                document = self.registry.load_bytes(data, source=source, media_type="")
            except DocumentError as exc:
                # 只有"我们认识的失败"被记进报告；其余异常向上抛（见模块 docstring）。
                logger.warning("%s 解析失败：%s", source, exc)
                entries.append(IngestEntry(source=source, error=str(exc)))
                continue
            key = document.fingerprint
            if key in owner:
                entries.append(
                    IngestEntry(
                        source=source,
                        media_type=document.media_type,
                        doc_id=key,
                        char_count=document.char_count,
                        block_count=len(document.blocks),
                        duplicate_of=owner[key],
                    )
                )
                continue
            owner[key] = source
            kept[key] = document
            entries.append(self._entry_for(document))
        report = IngestReport(entries=tuple(entries), documents=tuple(kept.values()))
        logger.info("入库完成：%s", report.summary_line())
        return report

    def ingest_dir(
        self,
        root: str | Path,
        *,
        suffixes: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> IngestReport:
        """把一个目录里的文件全部入库（排序遍历，可选截断）.

        ``limit`` 是给"先看一眼这个目录里有什么"用的：它作用在**排序后**的
        列表上，因此"看前 10 个"在任何机器上都是同样那 10 个。
        """
        if limit is not None and limit < 0:
            raise DocumentError(f"limit 不能为负数，收到 {limit}")
        paths = iter_supported_files(root, suffixes=suffixes)
        if limit is not None:
            paths = paths[:limit]
        items = [(str(path), path.read_bytes()) for path in paths]
        report: IngestReport = self.ingest_bytes(items)
        return IngestReport(
            entries=report.entries,
            documents=report.documents,
            metadata={"root": str(root), "scanned": str(len(paths))},
        )

    def table(self) -> list[dict[str, Any]]:
        """注册表的可读视图（转交给 ``LoaderRegistry.table``）."""
        return self.registry.table()


def knowledge_record_shape() -> dict[str, Any]:
    """知识库记录的字段说明（报告与端点同源）."""
    return {
        "fields": {
            "doc_id": "内容指纹（sha256 前 16 位），去重与溯源都按它",
            "source": "来源标签（路径或 URL），只作为标签使用",
            "text": "规范化后的全文；day062 会在这里切出片段",
            "metadata": "媒体类型、字符数、块数 + 加载器自己写的诊断字段",
        },
        "next_step": "day062 的分块器会以 blocks 为单位切分，接口形状不变",
        "loader": "day009 的 VectorStore.add 直接消费这个形状",
    }


__all__ = [
    "DOCUMENTS_LIMITATIONS",
    "DOCUMENTS_OUT_OF_SCOPE",
    "DocumentIngestor",
    "IngestEntry",
    "IngestReport",
    "iter_supported_files",
    "knowledge_record_shape",
]
