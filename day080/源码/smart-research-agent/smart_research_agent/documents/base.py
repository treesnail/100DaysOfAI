"""加载器接口与注册表：把"格式各异的文件"接到一个统一的入口上（M6-D1）.

day058 的注册表管的是**模型版本**，day060 的绑定管的是**部署事实**，
今天的注册表管的是**解析器**——三者的形状完全一样：

```text
键（可查询的身份）  →  值（干活的实现）
媒体类型 / 后缀     →  DocumentLoader
```

## 三个刻意的设计决定

### 1. 加载器只接受 ``bytes``，不接受路径

```python
def load_bytes(self, data: bytes, *, source: str, media_type: str) -> Document
```

理由有三条，每条都对应一类真实需求：

- **同一份内容可能来自任何地方**：本地文件、HTTP 响应、对象存储、
  用户上传的内存字节。把"读文件"放进接口，等于让每个加载器各自处理
  路径与编码，而这两件事与"解析 PDF"毫无关系；
- **``source`` 是标签不是句柄**：它是给人看的来源标识（路径或 URL），
  加载器只把它写进结果，**不允许再去读它**——否则"同一个解析器
  既能解析内存数据又能偷偷访问磁盘"会让测试无法离线；
- **测试可以完全不碰文件系统**：dev 与 CI 上构造一个 200 字节的字节串
  就能覆盖一个解析器的全部分支。

### 2. 类型识别由**内容优先**，并且把冲突报出来

``detect_media_type`` 同时看后缀与魔数，两者不一致时**以内容为准**并
把冲突记下来：

```text
一份 .txt 文件以 %PDF- 开头    → 按 PDF 解析（内容更可信），并记下冲突
一份 .pdf 文件其实是 ZIP       → 按 ZIP 处理 → 报"不在支持范围"（见 errors 模块）
```

只信后缀的代价在真实数据里非常具体：**导出工具改名不改内容**、
**下载器按 URL 猜后缀**、**磁盘上有人手工改过扩展名**。
把它报成一个"有冲突但按内容处理"的记录，比静默按后缀处理要安全得多
——后者会得到一个"解析成功但内容全空"的文档，而那种结果最容易被当成
"这份文件本来就没内容"。

### 3. 未知类型是一个**可读的结论**，不是异常堆栈

注册表找不到加载器时抛 ``DocumentError`` 并在消息里列出**当前有哪些
加载器可用**。批量入库时这条消息会直接进报告，而"你要的是不是其中之一"
比"KeyError: 'application/zip'"有用得多（day045 列候选模型名是同一条思路）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from smart_research_agent.config import settings
from smart_research_agent.documents.errors import DocumentError
from smart_research_agent.documents.types import Document, make_document
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: 五种媒体类型（**这里只有五种，而加载器有五个**——一一对应是刻意的：
#: 一个加载器声称支持两种媒体类型时，它内部一定有一个 if，
#: 而那个 if 应该变成两个加载器）。
MEDIA_TEXT = "text/plain"
MEDIA_MARKDOWN = "text/markdown"
MEDIA_HTML = "text/html"
MEDIA_PDF = "application/pdf"
MEDIA_DOCX = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)
MEDIA_TYPES: tuple[str, ...] = (MEDIA_TEXT, MEDIA_MARKDOWN, MEDIA_HTML, MEDIA_PDF, MEDIA_DOCX)

#: 后缀 → 媒体类型。取值都带点（``.md`` 而不是 ``md``）：
#: 一个不带点的表会让 ``"README"`` 与 ``"README.md"`` 这类判断出现
#: 难以察觉的错配。
SUFFIX_MEDIA_TYPES: dict[str, str] = {
    ".txt": MEDIA_TEXT,
    ".text": MEDIA_TEXT,
    ".log": MEDIA_TEXT,
    ".md": MEDIA_MARKDOWN,
    ".markdown": MEDIA_MARKDOWN,
    ".html": MEDIA_HTML,
    ".htm": MEDIA_HTML,
    ".pdf": MEDIA_PDF,
    ".docx": MEDIA_DOCX,
}

#: 魔数 → 媒体类型。**只列"能确定唯一类型"的签名**：
#: ``PK\\x03\\x04`` 是 ZIP 家族（docx / xlsx / pptx 都是），
#: 单靠它分不出 docx，因此它在这里只用来**判冲突**，不作为最终答案。
MAGIC_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"%PDF-", MEDIA_PDF),
    (b"PK\x03\x04", "application/zip"),
)

#: 类型识别的三种来源。
DECIDED_BY_MAGIC = "magic"
DECIDED_BY_SUFFIX = "suffix"
DECIDED_BY_FALLBACK = "fallback"


@dataclass(frozen=True)
class MediaTypeDetection:
    """一次类型识别的结果（含冲突记录）.

    ``conflict`` 为真时 ``media_type`` 取的是**魔数**判断的结果。
    这类记录必须进报告：它往往是"磁盘上的文件被人改过"的第一手证据。
    """

    media_type: str
    decided_by: str
    suffix_type: str = ""
    magic_type: str = ""
    conflict: bool = False

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return asdict(self)

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        if not self.conflict:
            return f"{self.media_type}（按{self.decided_by}判断）"
        return (
            f"{self.media_type}（**冲突**：后缀说 {self.suffix_type}、"
            f"内容说 {self.magic_type}；按内容处理）"
        )


def suffix_media_type(name: str) -> str:
    """按文件名后缀猜媒体类型；未知后缀返回空串.

    用小写化之后再查表：``.MD`` 与 ``.md`` 是同一个后缀，
    而大小写在 Windows 与 Linux 上都可能出现。
    """
    return SUFFIX_MEDIA_TYPES.get(Path(name).suffix.lower(), "")


def magic_media_type(data: bytes) -> str:
    """按文件头魔数判断媒体类型；无法确定时返回空串.

    HTML 没有魔数，因此它靠"前若干字节里出现 ``<html`` 或 ``<!doctype html``"
    来判断——这是一个**启发式**，所以它只在后缀也给不出答案时才生效
    （见 ``detect_media_type``）。
    """
    head = data[:1024]
    for signature, media_type in MAGIC_SIGNATURES:
        if head.startswith(signature):
            return media_type
    lowered = head.lstrip()[:512].lower()
    if b"<!doctype html" in lowered or b"<html" in lowered:
        return MEDIA_HTML
    return ""


def detect_media_type(name: str, data: bytes = b"") -> MediaTypeDetection:
    """综合后缀与内容给出媒体类型（**内容优先，冲突必报**）.

    四种组合各有明确处理：

    | 后缀 | 内容 | 结果 |
    |------|------|------|
    | 有 | 有且一致 | 按它 |
    | 有 | 有但不同 | **按内容**，``conflict=True`` |
    | 有 | 无 | 按后缀 |
    | 无 | 有 | 按内容 |
    | 无 | 无 | ``text/plain``（``decided_by="fallback"``） |

    最后一行的兜底是刻意的：**没有证据时按纯文本处理**是最保守的选择
    （它不会尝试解压、不会解析二进制），而"拒绝加载"会把一个普通的
    无后缀日志文件也变成一次失败。
    """
    by_suffix = suffix_media_type(name)
    by_magic = magic_media_type(data)
    if by_magic and by_suffix and by_magic != by_suffix:
        # ``PK\x03\x04`` 与 docx 的关系需要单独说明：ZIP 签名只能证明
        # "这是一个 ZIP"，docx 的判定还要看里面有没有 ``word/document.xml``。
        # 因此当后缀是 docx 而魔数是 zip 时**不算冲突**——那是同一件事的
        # 两种描述，由 docx 加载器在解包时给出最终结论。
        if by_magic == "application/zip" and by_suffix == MEDIA_DOCX:
            return MediaTypeDetection(
                media_type=by_suffix,
                decided_by=DECIDED_BY_SUFFIX,
                suffix_type=by_suffix,
                magic_type=by_magic,
            )
        return MediaTypeDetection(
            media_type=by_magic,
            decided_by=DECIDED_BY_MAGIC,
            suffix_type=by_suffix,
            magic_type=by_magic,
            conflict=True,
        )
    if by_magic:
        return MediaTypeDetection(
            media_type=by_magic,
            decided_by=DECIDED_BY_MAGIC,
            suffix_type=by_suffix,
            magic_type=by_magic,
        )
    if by_suffix:
        return MediaTypeDetection(
            media_type=by_suffix,
            decided_by=DECIDED_BY_SUFFIX,
            suffix_type=by_suffix,
        )
    return MediaTypeDetection(
        media_type=MEDIA_TEXT,
        decided_by=DECIDED_BY_FALLBACK,
        suffix_type=by_suffix,
        magic_type=by_magic,
    )


class DocumentLoader(ABC):
    """一个格式的解析器：``bytes`` + 来源标签 → ``Document``."""

    #: 本加载器支持的媒体类型（注册表按它建索引）。
    media_types: tuple[str, ...] = ()
    #: 本加载器认识的后缀（用于报告里的"这个我会"清单）。
    suffixes: tuple[str, ...] = ()
    #: 一句话说明（报告与 ``/documents/loaders`` 端点直接读它）。
    description: str = ""

    @abstractmethod
    def load_bytes(self, data: bytes, *, source: str, media_type: str) -> Document:
        """把字节解析成 ``Document``；不支持的内容抛 ``UnsupportedDocument``."""

    def supported_line(self) -> str:
        """人类可读的一行"支持范围"（报告里逐行打印）."""
        return (
            f"{type(self).__name__}: {', '.join(self.media_types)} "
            f"（后缀 {', '.join(self.suffixes) or '（无）'}）—— {self.description}"
        )


class LoaderRegistry:
    """加载器注册表：按媒体类型路由，并统一做文件大小与类型识别.

    ``max_bytes`` 是**入口处的一次护栏**：一份 2 GiB 的 PDF 在解析器里
    会把内存吃光，而错误会以 ``MemoryError`` 的形式出现在栈的深处。
    在入口拦一次，得到的是一条能写进报告的结论。
    """

    def __init__(
        self,
        loaders: Sequence[DocumentLoader] = (),
        *,
        max_bytes: int | None = None,
    ) -> None:
        self._loaders: dict[str, DocumentLoader] = {}
        for loader in loaders:
            self.register(loader)
        self.max_bytes = (
            settings.documents_max_file_mib * 1024 * 1024 if max_bytes is None else max_bytes
        )

    # ------------------------------------------------------------------ 注册
    def register(self, loader: DocumentLoader) -> None:
        """登记一个加载器；媒体类型重复时抛 ``DocumentError``.

        **重复即拒绝**而不是"后者覆盖前者"：两个加载器声称解析同一种
        格式时，实际生效的是哪一个取决于导入顺序——而"行为取决于导入顺序"
        是最难复现的一类问题（day057 的 `STAGE_ORDER`、day059 的依赖表
        都在防同一件事）。
        """
        if not loader.media_types:
            raise DocumentError(
                f"{type(loader).__name__} 没有声明 media_types："
                "注册表按媒体类型路由，不声明就无法被选中"
            )
        for media_type in loader.media_types:
            if media_type in self._loaders:
                raise DocumentError(
                    f"媒体类型 {media_type} 已由 "
                    f"{type(self._loaders[media_type]).__name__} 注册，"
                    f"不能再由 {type(loader).__name__} 注册"
                )
            self._loaders[media_type] = loader

    def get(self, media_type: str) -> DocumentLoader:
        """按媒体类型取加载器；没有则抛错并列出全部可用类型."""
        loader = self._loaders.get(media_type)
        if loader is None:
            available = ", ".join(sorted(self._loaders)) or "（注册表为空）"
            raise DocumentError(
                f"没有能解析 {media_type} 的加载器；当前可用：{available}"
            )
        return loader

    @property
    def loaders(self) -> list[DocumentLoader]:
        """已注册的加载器（按媒体类型排序，报告与端点的稳定顺序）."""
        return list(self._loaders.values())

    @property
    def media_types(self) -> list[str]:
        """已注册的媒体类型（排序后，便于逐行对比）."""
        return sorted(self._loaders)

    def supports(self, media_type: str) -> bool:
        """是否有加载器能处理这个媒体类型."""
        return media_type in self._loaders

    # ------------------------------------------------------------------ 入口
    def check_size(self, data: bytes, *, source: str) -> None:
        """大小护栏（**在解析之前拦**，见类 docstring）."""
        if len(data) > self.max_bytes:
            raise DocumentError(
                f"{source} 有 {len(data) / 1024 / 1024:.2f} MiB，"
                f"超过上限 {self.max_bytes / 1024 / 1024:.2f} MiB："
                "在入口拦一次才能得到一条能写进报告的结论，"
                "否则它会以 MemoryError 出现在栈的深处"
            )

    def load_bytes(
        self, data: bytes, *, source: str, media_type: str = ""
    ) -> Document:
        """解析一段字节（``media_type`` 留空时按来源名与内容自动识别）."""
        self.check_size(data, source=source)
        if media_type:
            detection = MediaTypeDetection(media_type=media_type, decided_by="given")
        else:
            detection = detect_media_type(source, data)
        loader = self.get(detection.media_type)
        document = loader.load_bytes(
            data, source=source, media_type=detection.media_type
        )
        # 类型识别的结论**进文档元数据**：解析成功但类型曾发生冲突这件事，
        # 必须跟着文档走（它是"这份文件被人改过"的第一手证据）。
        merged = dict(document.metadata)
        merged.setdefault("detected_media_type", detection.media_type)
        merged.setdefault("media_type_decided_by", detection.decided_by)
        if detection.suffix_type:
            merged.setdefault("suffix_media_type", detection.suffix_type)
        if detection.magic_type:
            merged.setdefault("magic_media_type", detection.magic_type)
        merged["media_type_conflict"] = "true" if detection.conflict else "false"
        return Document(
            source=document.source,
            media_type=document.media_type,
            text=document.text,
            blocks=document.blocks,
            metadata=merged,
            doc_id=document.doc_id,
        )

    def load_path(self, path: str | Path) -> Document:
        """读一个文件并解析（**唯一允许碰文件系统的地方**）."""
        target = Path(path)
        if not target.is_file():
            raise DocumentError(f"{target} 不是一个文件（目录或无权限）")
        return self.load_bytes(
            target.read_bytes(), source=str(target), media_type=""
        )

    def table(self) -> list[dict[str, Any]]:
        """把注册表渲染成一张表（报告与 ``/documents/loaders`` 端点同源）."""
        rows: list[dict[str, Any]] = []
        for media_type in self.media_types:
            loader = self._loaders[media_type]
            rows.append(
                {
                    "media_type": media_type,
                    "loader": type(loader).__name__,
                    "suffixes": list(loader.suffixes),
                    "description": loader.description,
                }
            )
        return rows


def default_registry() -> LoaderRegistry:
    """构建默认注册表（五个加载器一一对应五种媒体类型）.

    工厂函数**在调用时导入**各加载器，而不是在模块顶层导入：
    这样 ``documents.base`` 不会因为某个加载器内部的导入失败而整体不可用，
    "哪一个加载器起不来"也能被单独看见（day059 的 CI 渲染器是同一条思路）。
    """
    from smart_research_agent.documents.docx_loader import DocxLoader
    from smart_research_agent.documents.html_loader import HtmlLoader
    from smart_research_agent.documents.markdown_loader import MarkdownLoader
    from smart_research_agent.documents.pdf_loader import PdfLoader
    from smart_research_agent.documents.text_loader import TextLoader

    registry = LoaderRegistry(
        [TextLoader(), MarkdownLoader(), HtmlLoader(), PdfLoader(), DocxLoader()]
    )
    logger.info("文档加载器就绪：%s", ", ".join(registry.media_types))
    return registry


__all__ = [
    "DECIDED_BY_FALLBACK",
    "DECIDED_BY_MAGIC",
    "DECIDED_BY_SUFFIX",
    "MAGIC_SIGNATURES",
    "MEDIA_DOCX",
    "MEDIA_HTML",
    "MEDIA_MARKDOWN",
    "MEDIA_PDF",
    "MEDIA_TEXT",
    "MEDIA_TYPES",
    "SUFFIX_MEDIA_TYPES",
    "DocumentLoader",
    "LoaderRegistry",
    "MediaTypeDetection",
    "default_registry",
    "detect_media_type",
    "magic_media_type",
    "make_document",
    "suffix_media_type",
]
