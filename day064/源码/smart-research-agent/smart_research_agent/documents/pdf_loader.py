"""PDF 加载器：用标准库抽出文本（M6-D1）.

PDF 是 M6 里唯一**为打印而设计**的格式：它描述的是"每个字符画在纸上的哪个位置"，
而不是"这是一段话"。因此"从 PDF 提取文本"本质上是一次**逆向工程**，
本模块把这件事做到"够用且诚实"的程度。

## 它怎么做

```text
1. 扫描文件里全部 stream ... endstream 段
2. 用 zlib 解 FlateDecode（PDF 32000-1:2008 §7.4；规范里 FlateDecode 的
   定义直接引 RFC 1950/1951，也就是 zlib/deflate）；解不开就按原文用
3. 在内容流里跑一个小型词法分析器，识别文本相关操作符
4. 把"一串字符 + 它在页面上的纵坐标"收集成行，再按行距把行聚成段落
```

## 它认识哪些操作符

| 操作符 | 含义 | 规范条款 |
|--------|------|---------|
| ``BT`` / ``ET`` | 文本对象的开始与结束 | §9.4.1 |
| ``Td`` / ``TD`` / ``T*`` / ``Tm`` | 文本定位（换行、接排、换矩阵） | §9.4.2 |
| ``Tj`` / ``'`` / ``"`` / ``TJ`` | 显示文本（``'`` 与 ``"`` 自带换行） | §9.4.3 |
| ``Tf`` | 选字体（本模块只记录，不做字形映射） | §9.3 |

## 它**做不到**什么（这一段比上面重要）

1. **Identity-H 编码的 CID 字体**：中文 PDF 常用这种编码，字符码是**字形索引**
   而不是 Unicode。规范 §9.10.2 的原话是："如果这些方法都得不到 Unicode 值，
   **就无法确定字符码代表什么**"。也就是说：这不是本模块的缺陷，而是规范层面的
   不可判定——**除非** PDF 内嵌了 ``/ToUnicode`` CMap，而本模块不解析 CMap。
   遇到这种情况，元数据里会有 ``pdf_cid_font_detected=true`` 与一条明确的告警；
2. **不以文本形式存储的内容**：扫描件（整页是图片）、矢量轮廓化的文字；
3. **阅读顺序的完全还原**：双栏排版、表格、脚注会被读成"按纵坐标排序的行流"。
   本模块只做**行距分段**这一个版面推断（见 ``_group_paragraphs``）；
4. **加密的 PDF**：``/Encrypt`` 出现即拒收（抛 ``UnsupportedDocument``），
   而不是返回一堆乱码。

**这四条边界全部写进元数据**，因为"没抽到的内容"最容易被当成
"文档里本来就没有"——而那一类判断会一路影响到检索质量，且无从察觉。

## 与 pypdf / pdfplumber 的关系

它们的当前版本分别是 6.18.0（2026-09-08）与 0.11.10，质量都比本模块高得多。
本课程不复用它们的原因只有一条且与质量无关：**整门课的源码要能在
离线 CI 里零依赖跑通**。因此本模块的定位是"教学实现 + 质量基线"：
它给出可复现的抽取结果与一组明确的边界，而不是一个生产级 PDF 解析器。
"""

from __future__ import annotations

import re
import statistics
import zlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.documents.base import MEDIA_PDF, DocumentLoader
from smart_research_agent.documents.errors import UnsupportedDocument
from smart_research_agent.documents.types import BLOCK_PARAGRAPH, Block, Document, make_document
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: PDF 文件头（也是 day061 类型识别里的魔数）.
PDF_SIGNATURE = b"%PDF-"

#: ``stream`` 与 ``endstream`` 关键字（规范要求它们各自独占一行，
#: 但真实文件里常有额外的空格，因此用宽松一点的匹配）。
STREAM_START = re.compile(rb"\bstream\r?\n")
STREAM_END = re.compile(rb"\bendstream\b")

#: 页面对象计数：``/Type /Page``（**不匹配 ``/Pages``**，
#: 那个是页面树的中间节点，页数会因此多算一堆）。
PAGE_OBJECT = re.compile(rb"/Type\s*/Page[^s]")

#: 加密字典：出现即拒收（见模块 docstring 第 4 条边界）。
ENCRYPT_MARKER = b"/Encrypt"

#: CID 字体与 CMap 的痕迹（用于给出"抽出来的文本可能不可用"的告警）。
CID_MARKERS: tuple[tuple[bytes, str], ...] = (
    (b"/Identity-H", "Identity-H 编码（字符码 = 字形索引）"),
    (b"/Type0", "Type0 复合字体"),
    (b"/ToUnicode", "内嵌 ToUnicode CMap（本模块不解析它）"),
)

#: 词法分析用的字符类。
WHITESPACE = b"\x00\t\n\x0c\r "
DELIMITERS = b"()<>[]{}/%"

#: ``TJ`` 数组里，数值元素绝对值超过它时判定为"词间空格"。
#: 规范里这个数是**千分之一 em**（``-1000`` = 一个 em 的反向位移），
#: 因此阈值 100 相当于"间隔超过 0.1 em 就当有空格"——这是排版里
#: 词间距的常见量级（很多资料用 120~250，取 100 偏保守，宁可多插空格）。
TJ_SPACE_THRESHOLD = 100.0

#: 段落判定的行距倍数：相邻两行的纵坐标差超过**中位行距的 1.5 倍**
#: 时判为段落边界。用中位数而不是平均值，因为标题与图注会把平均值拉飞。
PARAGRAPH_GAP_FACTOR = 1.5

#: 一行最多接纳多少个字符（防御性上限：一份损坏的内容流可能读出
#: 几百万个"字符"，而那会让 downstream 的窗口爆掉）。
MAX_LINE_CHARS = 4096


@dataclass
class _TextState:
    """内容流解析过程中的可变状态（行收集 + 位置跟踪）."""

    lines: list[tuple[float, str]] = field(default_factory=list)
    current: list[str] = field(default_factory=list)
    y: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def commit_line(self) -> None:
        """把当前缓冲落成一行（**连续的空行只留一个**）.

        空行是有意义的：它是内容流里显式的"这里断开"，也是段落判定的一个信号。
        但连续的空行没有额外信息（``ET`` 与收尾各提交一次就会产生两个），
        因此这里把它们折叠——这与 day061 的文本规范化把连续空行折叠到一个是
        同一种处理，**而它们的理由也相同：同一份内容不该有两种空行写法**。
        """
        text = "".join(self.current).strip()
        self.current.clear()
        if not text and self.lines and not self.lines[-1][1]:
            return
        self.lines.append((self.y, text))


def decode_literal_string(raw: bytes) -> str:
    """解码 PDF 的字面字符串（``(...)``）.

    两条规则：

    - **UTF-16BE**：PDF 2.0 起允许字符串以 ``FE FF`` 开头表示 UTF-16BE，
      这是非 ASCII 文本唯一可移植的写法；
    - 其余按 **latin-1** 解：PDF 的简单字体用它各自的编码表，而
      WinAnsiEncoding 与 latin-1 在 ASCII 区域完全一致——**本模块不做
      字体编码表的映射**，这一点写在模块 docstring 的边界里。
    """
    if raw.startswith(b"\xfe\xff"):
        return raw[2:].decode("utf-16-be", errors="replace")
    return raw.decode("latin-1", errors="replace")


def _unescape_literal(data: bytes, start: int) -> tuple[bytes, int]:
    """从 ``data[start]``（``(`` 之后）开始读一个字面字符串，返回 ``(内容, 新位置)``.

    处理的转义与规范 §7.3.4.2 一致：``\\n \\r \\t \\b \\f \\( \\) \\\\``
    以及 ``\\ddd``（一到三位八进制，**最多三位**：``\\0536`` 是 ``+6`` 而不是
    一个码点 536）。行末的反斜杠表示续行，续行处**不插入任何字符**。
    """
    out = bytearray()
    depth = 1
    index = start
    while index < len(data):
        char = data[index : index + 1]
        if char == b"\\":
            index += 1
            if index >= len(data):
                break
            escape = data[index : index + 1]
            simple = {
                b"n": b"\n",
                b"r": b"\r",
                b"t": b"\t",
                b"b": b"\b",
                b"f": b"\f",
                b"(": b"(",
                b")": b")",
                b"\\": b"\\",
            }
            if escape in simple:
                out.extend(simple[escape])
                index += 1
                continue
            if escape in (b"\n", b"\r"):
                index += 1
                continue
            digits = b""
            while index < len(data) and len(digits) < 3 and data[index : index + 1].isdigit():
                digits += data[index : index + 1]
                index += 1
            out.append(int(digits, 8) & 0xFF if digits else ord("\\"))
            continue
        if char == b"(":
            depth += 1
        elif char == b")":
            depth -= 1
            if depth == 0:
                return bytes(out), index + 1
        out.extend(char)
        index += 1
    return bytes(out), index


def _unescape_hex(data: bytes, start: int) -> tuple[bytes, int]:
    """读一个十六进制字符串 ``<...>``（空白忽略；奇数位补 0）."""
    digits = bytearray()
    index = start
    while index < len(data) and data[index : index + 1] != b">":
        char = data[index : index + 1]
        if char not in WHITESPACE:
            digits.extend(char)
        index += 1
    text = bytes(digits)
    if len(text) % 2:
        text += b"0"
    try:
        return bytes.fromhex(text.decode("ascii")), index + 1
    except ValueError:
        return b"", index + 1


class _Lexer:
    """内容流的小型词法分析器（只认文本抽取需要的几种记号）."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def _skip(self) -> None:
        """跳过空白与注释（``%`` 到行尾）."""
        while self.pos < len(self.data):
            char = self.data[self.pos : self.pos + 1]
            if char in WHITESPACE:
                self.pos += 1
                continue
            if char == b"%":
                while self.pos < len(self.data) and self.data[self.pos : self.pos + 1] not in (
                    b"\n",
                    b"\r",
                ):
                    self.pos += 1
                continue
            return

    def tokens(self) -> Iterator[tuple[str, Any]]:
        """逐个产出 ``("num"|"str"|"name"|"op"|"arr", value)``."""
        while True:
            self._skip()
            if self.pos >= len(self.data):
                return
            char = self.data[self.pos : self.pos + 1]
            if char == b"(":
                raw, self.pos = _unescape_literal(self.data, self.pos + 1)
                yield "str", raw
                continue
            if char == b"<":
                if self.data[self.pos : self.pos + 2] == b"<<":
                    self.pos += 2
                    yield "op", "<<"
                    continue
                raw, self.pos = _unescape_hex(self.data, self.pos + 1)
                yield "str", raw
                continue
            if char == b"[":
                self.pos += 1
                items: list[Any] = []
                for kind, value in self.tokens():
                    if kind == "op" and value == "]":
                        break
                    items.append(value)
                yield "arr", items
                continue
            if char in b"]>{}":
                self.pos += 1
                yield "op", char.decode("ascii")
                continue
            if char == b"/":
                self.pos += 1
                start = self.pos
                while self.pos < len(self.data) and (
                    self.data[self.pos : self.pos + 1] not in WHITESPACE
                    and self.data[self.pos : self.pos + 1] not in DELIMITERS
                ):
                    self.pos += 1
                yield "name", self.data[start : self.pos].decode("latin-1")
                continue
            start = self.pos
            while self.pos < len(self.data) and (
                self.data[self.pos : self.pos + 1] not in WHITESPACE
                and self.data[self.pos : self.pos + 1] not in DELIMITERS
            ):
                self.pos += 1
            token = self.data[start : self.pos].decode("latin-1")
            if not token:
                self.pos += 1
                continue
            try:
                yield "num", float(token)
            except ValueError:
                yield "op", token


def _string_token(kind: str, value: Any) -> str:
    """把 ``str`` / ``arr`` 记号转成文本（``arr`` 就是 ``TJ`` 的参数）.

    只有这两类记号会被传进来（``extract_lines`` 只挑它们），因此这里
    **不留一个"其他类型的兜底返回值"**：那个分支永远执行不到，
    而它会让"传错了类型"变成一个静默的空字符串。
    """
    if kind == "arr":
        parts: list[str] = []
        for item in value:
            if isinstance(item, bytes):
                parts.append(decode_literal_string(item))
            elif isinstance(item, float) and item < -TJ_SPACE_THRESHOLD:
                parts.append(" ")
        return "".join(parts)
    return decode_literal_string(value)


def extract_lines(content: bytes) -> tuple[list[tuple[float, str]], list[str]]:
    """从一条内容流里抽出行 ``(纵坐标, 文本)`` 与告警.

    位置跟踪只用 ``Td`` / ``TD`` / ``T*`` / ``Tm`` 的**纵坐标**：
    本模块不实现完整的文本矩阵与图形状态栈，因为"把字精确放回纸上"
    对检索没有价值，而"这一行在下一行上面还是下面"有价值。
    """
    state = _TextState()
    pending: list[Any] = []
    for kind, value in _Lexer(content).tokens():
        if kind in ("num", "str", "name", "arr"):
            pending.append((kind, value))
            if len(pending) > 16:
                pending.pop(0)
            continue
        operator = value
        if operator == "BT":
            state.current.clear()
            state.y = 0.0
        elif operator == "ET":
            state.commit_line()
        elif operator in ("Td", "TD"):
            numbers = [item for item in pending if item[0] == "num"]
            if len(numbers) >= 2:
                delta = float(numbers[-1][1])
                if delta:
                    state.commit_line()
                    state.y += delta
        elif operator in ("T*", "'", '"'):
            state.commit_line()
            state.y -= 12.0  # 缺省行距：PDF 的 `TL` 未被显式设置时的常见值
            if operator in ("'", '"'):
                for token in reversed(pending):
                    if token[0] in ("str", "arr"):
                        state.current.append(_string_token(*token))
                        break
        elif operator == "Tm":
            numbers = [item for item in pending if item[0] == "num"]
            if len(numbers) >= 6:
                new_y = float(numbers[-1][1])
                if state.current or new_y != state.y:
                    state.commit_line()
                state.y = new_y
        elif operator in ("Tj", "TJ"):
            for token in reversed(pending):
                if token[0] in ("str", "arr"):
                    text = _string_token(*token)
                    if len("".join(state.current)) + len(text) <= MAX_LINE_CHARS:
                        state.current.append(text)
                    else:
                        state.warnings.append("单行字符数超过上限，已截断（内容流可能损坏）")
                    break
        pending = []
    # 收尾只在**还有内容**时落行：无条件落会多出一个空行，而那个空行会让
    # "最后一行是什么"这个问题在调用方那里少看一行（本课第一版就是这样，
    # 单测里 `lines[-1][1]` 永远是空串）。
    if state.current:
        state.commit_line()
    return state.lines, state.warnings


def _group_paragraphs(lines: list[tuple[float, str]]) -> list[str]:
    """按行距把行聚成段落（本模块唯一的版面推断，见模块 docstring 边界第 3 条）.

    规则只有两条：

    - 空行 → 段落边界（它是内容流里显式的"这里断开"）；
    - 相邻非空行的纵坐标差 > **中位行距 × 1.5** → 段落边界
      （标题、图注、跨小节之间都会出现这种大间距）。

    用**中位数**而不是平均值：标题与图注的间距会把平均值抬到"正常行距的两倍"，
    于是所有正常的段落边界都判不出来（而那种失效看起来只是"段落变少了"）。
    """
    kept = [(y, text) for y, text in lines if text]
    if not kept:
        return []
    deltas = [
        abs(kept[index][0] - kept[index - 1][0])
        for index in range(1, len(kept))
        if kept[index][0] != kept[index - 1][0]
    ]
    median = statistics.median(deltas) if deltas else 0.0
    threshold = median * PARAGRAPH_GAP_FACTOR if median else 0.0
    paragraphs: list[str] = []
    buffer: list[str] = []
    previous_y: float | None = None
    for y, text in kept:
        gap = None if previous_y is None else abs(y - previous_y)
        if buffer and gap is not None and gap > threshold:
            paragraphs.append("\n".join(buffer))
            buffer = []
        buffer.append(text)
        previous_y = y
    # 这里**不再判** ``if buffer``：``kept`` 非空时循环至少append 一次，
    # 因此那个条件恒为真（本课第一版写了它，覆盖率报告指出不可达）。
    # 与 day059 那条被删掉的"至少要有一个无依赖阶段"是同一种清理。
    paragraphs.append("\n".join(buffer))
    return paragraphs


def iter_streams(data: bytes) -> Iterator[bytes]:
    """产出文件里的每一段 ``stream`` 原始字节（**不去解压**）.

    用"关键字扫描"而不是完整解析 PDF 对象（xref 表、对象流、交叉引用流）：
    后者需要实现一整套对象模型，而本模块只需要"拿到内容流"。
    代价是要容忍两种常见偏差：``stream`` 后面不一定是 ``\\n`` 而是 ``\\r\\n``，
    以及 ``endstream`` 前面可能有一两个填充字节——解压时一并容忍。
    """
    position = 0
    while True:
        start = STREAM_START.search(data, position)
        if start is None:
            return
        end = STREAM_END.search(data, start.end())
        if end is None:
            return
        yield data[start.end() : end.start()]
        position = end.end()


def maybe_inflate(raw: bytes) -> tuple[bytes, bool]:
    """尝试 FlateDecode（zlib）；失败时原样返回并标记未解压.

    失败的三种真实原因：① 这条流本来就没压缩（图片、``/Filter`` 为
    ``ASCIIHexDecode`` 等）；② 流的长度估算多了几个字节（``endstream``
    前的填充）；③ 流真的坏了。三种都**不该让整份文档失败**——
    跳过它、把"有多少条流解不开"记进元数据，比抛异常更有用。
    """
    for candidate in (raw, raw.rstrip(b"\r\n"), raw.strip(b"\r\n")):
        if not candidate:
            continue
        try:
            return zlib.decompress(candidate), True
        except zlib.error:
            continue
    return raw, False


@dataclass(frozen=True)
class PdfTextExtraction:
    """一次 PDF 文本抽取的结果（行、段落与全部诊断信息）."""

    lines: list[tuple[float, str]]
    paragraphs: list[str]
    pages: int
    streams: int
    inflated_streams: int
    warnings: list[str]
    cid_font: bool
    encrypted: bool

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典（不含行级细节）."""
        return {
            "paragraphs": len(self.paragraphs),
            "lines": len([item for item in self.lines if item[1]]),
            "pages": self.pages,
            "streams": self.streams,
            "inflated_streams": self.inflated_streams,
            "cid_font": self.cid_font,
            "encrypted": self.encrypted,
            "warnings": list(self.warnings),
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        return (
            f"{self.pages} 页 / {self.streams} 条流（解压 {self.inflated_streams}）| "
            f"{len([item for item in self.lines if item[1]])} 行 → "
            f"{len(self.paragraphs)} 段 | CID 字体 {self.cid_font}"
        )


def extract_pdf_text(data: bytes, *, source: str = "") -> PdfTextExtraction:
    """抽出 PDF 的文本与全部诊断信息（**不抛异常，除加密与非法头**）."""
    if not data.startswith(PDF_SIGNATURE):
        raise UnsupportedDocument(
            f"{source or '（未知来源）'} 不是 PDF（头四个字节不是 %PDF-）"
        )
    if ENCRYPT_MARKER in data:
        raise UnsupportedDocument(
            f"{source or '（未知来源）'} 是加密的 PDF（含 /Encrypt）："
            "返回一堆乱码比拒收更糟——乱码会一路走进索引"
        )

    warnings: list[str] = []
    cid_font = False
    for marker, description in CID_MARKERS:
        if marker in data:
            cid_font = cid_font or marker != b"/ToUnicode"
            warnings.append(f"检测到 {description}")
    if cid_font:
        warnings.append(
            "CID 字体的字符码是字形索引而不是 Unicode（规范 §9.10.2："
            "无法确定字符码代表什么），本模块不做 CMap 映射——中文可能是乱码或空"
        )

    lines: list[tuple[float, str]] = []
    streams = 0
    inflated = 0
    for raw in iter_streams(data):
        streams += 1
        content, ok = maybe_inflate(raw)
        if ok:
            inflated += 1
        if b"Tj" not in content and b"TJ" not in content:
            continue
        extracted, stream_warnings = extract_lines(content)
        lines.extend(extracted)
        warnings.extend(stream_warnings)
    # 只在"既没解压成功、也没抽到文本"时才告警：未压缩的内容流是合法写法，
    # 把它报成一条诊断会让"这份 PDF 有问题"这个信号被噪声稀释
    # （本课第一版就是这样，一份正常的未压缩 PDF 也带着一条警告）。
    if streams and not inflated and not any(text for _, text in lines):
        warnings.append("既没有流解压成功、也没有抽到任何文本：内容可能是扫描件或罕见编码")

    deduped = sorted(set(warnings))
    return PdfTextExtraction(
        lines=lines,
        paragraphs=_group_paragraphs(lines),
        pages=len(PAGE_OBJECT.findall(data)),
        streams=streams,
        inflated_streams=inflated,
        warnings=deduped,
        cid_font=cid_font,
        encrypted=False,
    )


class PdfLoader(DocumentLoader):
    """PDF 加载器：零依赖文本抽取（能力边界见模块 docstring）."""

    media_types: tuple[str, ...] = (MEDIA_PDF,)
    suffixes: tuple[str, ...] = (".pdf",)
    description: str = "PDF：扫描内容流 + zlib 解压 + 文本操作符抽取 + 行距分段"

    def load_bytes(self, data: bytes, *, source: str, media_type: str) -> Document:
        """解析 PDF 字节；加密或非 PDF 抛 ``UnsupportedDocument``."""
        extraction = extract_pdf_text(data, source=source)
        blocks = [
            Block(kind=BLOCK_PARAGRAPH, text=paragraph)
            for paragraph in extraction.paragraphs
        ]
        metadata = {
            # PDF 的字符串是"字节 + 字体编码"，因此这里**没有单一编码可报**：
            # 写 latin-1 会让读者以为已经解对了，而真相是"取决于字体"。
            "encoding": "取决于字体（本模块按 WinAnsi/latin-1 与 UTF-16BE 处理）",
            "encoding_confident": "false" if extraction.cid_font else "true",
            "pdf_pages": str(extraction.pages),
            "pdf_streams": str(extraction.streams),
            "pdf_inflated_streams": str(extraction.inflated_streams),
            "pdf_cid_font_detected": "true" if extraction.cid_font else "false",
            "pdf_warnings": " | ".join(extraction.warnings),
            "paragraph_count": str(len(blocks)),
        }
        if extraction.warnings:
            logger.warning("%s 的 PDF 抽取有 %d 条诊断", source, len(extraction.warnings))
        return make_document(
            source=source, media_type=media_type, blocks=blocks, metadata=metadata
        )


def pdf_boundaries() -> dict[str, Any]:
    """PDF 抽取支持的操作符与做不到的事（报告与端点同源）."""
    return {
        "operators": [
            {"operator": "BT / ET", "meaning": "文本对象的开始与结束", "spec": "PDF 32000-1 §9.4.1"},
            {
                "operator": "Td / TD / T* / Tm",
                "meaning": "文本定位（换行 / 接排 / 换矩阵）",
                "spec": "PDF 32000-1 §9.4.2",
            },
            {
                "operator": "Tj / ' / \" / TJ",
                "meaning": "显示文本（' 与 \" 自带换行；TJ 数组里的负位移判为空格）",
                "spec": "PDF 32000-1 §9.4.3",
            },
            {"operator": "Tf", "meaning": "选字体（只记录，不做字形映射）", "spec": "PDF 32000-1 §9.3"},
        ],
        "filters": [
            {
                "name": "FlateDecode",
                "implementation": "zlib.decompress（规范直接引用 RFC 1950/1951）",
                "spec": "PDF 32000-1 §7.4",
            }
        ],
        "not_supported": [
            "Identity-H 等 CID 字体的字形索引 → Unicode 映射（规范 §9.10.2 判定为不可确定）",
            "内嵌 /ToUnicode CMap 的解析",
            "扫描件与矢量轮廓化的文字（它们不是文本）",
            "双栏、表格等版面的阅读顺序还原（只做行距分段）",
            "加密 PDF（/Encrypt 出现即拒收）",
            "除 FlateDecode 以外的过滤器（LZW / ASCIIHex / RunLength / DCTDecode 等）",
        ],
        "reference_versions": {
            "pypdf": "6.18.0（2026-09-08）",
            "pdfplumber": "0.11.10",
            "note": "它们的质量远高于本模块；本课程不复用只为保证离线零依赖",
        },
    }


__all__ = [
    "CID_MARKERS",
    "ENCRYPT_MARKER",
    "MAX_LINE_CHARS",
    "PARAGRAPH_GAP_FACTOR",
    "PDF_SIGNATURE",
    "TJ_SPACE_THRESHOLD",
    "PdfLoader",
    "PdfTextExtraction",
    "decode_literal_string",
    "extract_lines",
    "extract_pdf_text",
    "iter_streams",
    "maybe_inflate",
    "pdf_boundaries",
]
