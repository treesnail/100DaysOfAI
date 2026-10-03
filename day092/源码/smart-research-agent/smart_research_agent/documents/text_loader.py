"""纯文本与编码探测：把"一串字节"变成"一串已知编码的文本"（M6-D1）.

这是整个 M6 里最容易被跳过、也最容易埋雷的一步。理由很具体：

> **猜错编码不会报错。** 它只会让中文变成 ``æˆ‘ä»¬`` 这样的乱码，
> 而乱码会一路通过分块、向量化、检索，最后以一个"模型答非所问"的形式
> 出现在用户面前——排查要从用户反馈倒推回编码，中间隔着六天的工作量。

因此本模块做两件事：

```text
1. 按一个**固定顺序**尝试若干编码，并把"是怎么定的"记录下来
2. 定不下来时**明确标记为不可信**，让上层决定要不要拒收
```

## 尝试顺序与每一步的理由

| 顺序 | 依据 | 为什么排在这里 |
|------|------|---------------|
| 1 | BOM | 有 BOM 就是明示，不需要猜（**注意 UTF-32 的前缀与 UTF-16 相同，必须先判 UTF-32**） |
| 2 | 二进制探测 | NUL 字节出现在头 8 KiB 且没有 BOM → 这不是文本（一份被改名成 `.txt` 的可执行文件） |
| 3 | UTF-8 严格解码 | 现代文档的绝对主流；**严格**解码让它成为一个可靠的判据 |
| 4 | GB18030 | 中文旧文档的兜底；它是 GBK/GB2312 的超集，一个编码覆盖三代 |
| 5 | Latin-1 | 永不失败（任何字节序列都是合法 Latin-1），因此**只能排最后** |

第 3 步的"严格"是关键：UTF-8 有一个很强的性质——**不合法的字节序列很少**，
因此"严格解码成功"本身就是一条相当可靠的证据。反过来，
如果用 ``errors="replace"`` 去试，那么任何字节序列都会"成功"，
这个判据就完全失效了。

第 5 步的 Latin-1 是**最后一张网**：它保证函数总能返回一个字符串，
因此调用方需要显式处理 ``confident=False`` 这种情形，而不是处理异常。
这与 day059 门禁的 ``when_missing`` 一列是同一种表达：
**把"我没把握"做成一个字段，而不是一个沉默。**
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from smart_research_agent.documents.base import MEDIA_TEXT, DocumentLoader
from smart_research_agent.documents.errors import UnsupportedDocument
from smart_research_agent.documents.types import BLOCK_PARAGRAPH, Block, Document, make_document
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: 编码判定的四种来源。
ENCODING_FROM_BOM = "bom"
ENCODING_FROM_STRICT = "strict"
ENCODING_FROM_FALLBACK = "fallback"
ENCODING_FROM_LATIN1 = "latin-1"

#: BOM（字节序标记）→ 编码名。**顺序敏感**：
#: ``\\xff\\xfe`` 是 UTF-16-LE 的 BOM，而 UTF-32-LE 的 BOM 是
#: ``\\xff\\xfe\\x00\\x00``——如果先判 UTF-16-LE，UTF-32 的文件会被
#: 解成"每个字符后面跟一个 NUL"，而那种乱码看起来很像文件损坏。
BOMS: tuple[tuple[bytes, str], ...] = (
    (b"\x00\x00\xfe\xff", "utf-32-be"),
    (b"\xff\xfe\x00\x00", "utf-32-le"),
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xfe\xff", "utf-16-be"),
    (b"\xff\xfe", "utf-16-le"),
)

#: 二进制探测的扫描窗口（字节）。参考 UNIX ``file`` 的用法取前 8 KiB：
#: 足够覆盖任何文本文件的开头，又不会为一个大文件多读一次。
BINARY_SCAN_BYTES = 8192

#: 中文字旧的兜底编码：GB18030 是 GBK 与 GB2312 的超集，
#: 用它一个就够（**注意不是 "gbk"**：纯 GBK 会漏掉 GB18030 扩展区的字符）。
CJK_FALLBACK_ENCODING = "gb18030"


@dataclass(frozen=True)
class DecodedText:
    """一次解码的结果：文本 + 编码 + 依据 + 是否有把握."""

    text: str
    encoding: str
    decided_by: str
    confident: bool

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return asdict(self)

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        mark = "可信" if self.confident else "**不可信**"
        return f"{self.encoding}（按{self.decided_by}判断，{mark}，{len(self.text)} 字符）"


def detect_bom(data: bytes) -> tuple[str, int]:
    """识别 BOM，返回 ``(编码名, BOM 长度)``；没有 BOM 时返回 ``("", 0)``."""
    for bom, encoding in BOMS:
        if data.startswith(bom):
            return encoding, len(bom)
    return "", 0


def looks_binary(data: bytes) -> bool:
    """判断前若干字节是否"像二进制"（判据是 NUL 字节）.

    只用 NUL 一个判据，是因为它在文本里几乎不可能出现：
    即便是 UTF-16 的文本，也已经由 BOM 那一步先拦下了。
    更复杂的启发式（控制字符比例、可打印字符比例）会带来误判，
    而**误判的代价是不对称的**：把文本判成二进制会让一份正常文档入库失败，
    把二进制判成文本只会得到一段明显是乱的文本（而且不可信标记会提示它）。
    """
    return b"\x00" in data[:BINARY_SCAN_BYTES]


def decode_bytes(data: bytes, *, source: str = "") -> DecodedText:
    """按固定顺序尝试若干编码，返回文本与判定依据.

    ``source`` 只用于日志（错误信息里能指出是哪一份文件）。
    """
    if not data:
        return DecodedText(text="", encoding="utf-8", decided_by=ENCODING_FROM_STRICT, confident=True)

    bom_encoding, bom_length = detect_bom(data)
    if bom_encoding:
        body = data[bom_length:]
        # BOM 明确写明了编码，因此这里允许它失败——失败说明文件本身坏了，
        # 而不是"我们猜错了"。用 ``utf-8-sig`` 时要注意它自己会吃掉 BOM，
        # 但我们已经手工切掉了 BOM，因此统一按"去掉 BOM 的编码"来解。
        encoding = {"utf-8-sig": "utf-8"}.get(bom_encoding, bom_encoding)
        try:
            return DecodedText(
                text=body.decode(encoding),
                encoding=bom_encoding,
                decided_by=ENCODING_FROM_BOM,
                confident=True,
            )
        except UnicodeDecodeError as exc:
            raise UnsupportedDocument(
                f"{source or '（未知来源）'} 的 BOM 声明为 {bom_encoding}，但按它解码失败：{exc}"
            ) from exc

    if looks_binary(data):
        raise UnsupportedDocument(
            f"{source or '（未知来源）'} 看起来是二进制文件（头 {BINARY_SCAN_BYTES} 字节里出现 NUL）："
            "改名为 .txt 不会让它变成文本"
        )

    try:
        return DecodedText(
            text=data.decode("utf-8"),
            encoding="utf-8",
            decided_by=ENCODING_FROM_STRICT,
            confident=True,
        )
    except UnicodeDecodeError:
        pass

    try:
        return DecodedText(
            text=data.decode(CJK_FALLBACK_ENCODING),
            encoding=CJK_FALLBACK_ENCODING,
            decided_by=ENCODING_FROM_FALLBACK,
            confident=True,
        )
    except UnicodeDecodeError:
        pass

    return DecodedText(
        text=data.decode("latin-1"),
        encoding="latin-1",
        decided_by=ENCODING_FROM_LATIN1,
        confident=False,
    )


def split_paragraphs(text: str) -> list[str]:
    """按空行切段（并丢掉只含空白的段）.

    不用"按行切"：一行一句的做法会把"一段说明文字"切成一堆碎片，
    而 day062 的分块器再按字符数切一次之后，每个片段都会失去上下文。
    段落是纯文本里**唯一稳定存在的结构**，用它当第一层单位最稳妥。
    """
    paragraphs: list[str] = []
    buffer: list[str] = []
    for line in text.split("\n"):
        if line.strip():
            buffer.append(line)
            continue
        if buffer:
            paragraphs.append("\n".join(buffer))
            buffer = []
    if buffer:
        paragraphs.append("\n".join(buffer))
    return paragraphs


class TextLoader(DocumentLoader):
    """纯文本加载器：解码 → 按空行切段."""

    media_types: tuple[str, ...] = (MEDIA_TEXT,)
    suffixes: tuple[str, ...] = (".txt", ".text", ".log")
    description: str = "纯文本：按固定顺序探测编码，再按空行切段"

    def load_bytes(self, data: bytes, *, source: str, media_type: str) -> Document:
        """解析纯文本字节；二进制内容抛 ``UnsupportedDocument``."""
        decoded = decode_bytes(data, source=source)
        blocks = [
            Block(kind=BLOCK_PARAGRAPH, text=paragraph)
            for paragraph in split_paragraphs(decoded.text)
        ]
        metadata = {
            "encoding": decoded.encoding,
            "encoding_decided_by": decoded.decided_by,
            "encoding_confident": "true" if decoded.confident else "false",
            "paragraph_count": str(len(blocks)),
        }
        if not decoded.confident:
            logger.warning(
                "%s 的编码只能按 latin-1 兜底，中文很可能已经是乱码（请在上游修正编码）",
                source,
            )
        return make_document(
            source=source, media_type=media_type, blocks=blocks, metadata=metadata
        )


#: 默认的编码兜底顺序说明（报告与端点直接读它）。
ENCODING_CHAIN: tuple[str, ...] = (
    "BOM（utf-32-be / utf-32-le / utf-8-sig / utf-16-be / utf-16-le）",
    "二进制探测（头 8 KiB 出现 NUL 即拒收）",
    "utf-8 严格解码",
    CJK_FALLBACK_ENCODING,
    "latin-1（永不失败，因此只能排最后；结果标记 encoding_confident=false）",
)


def encoding_chain() -> list[dict[str, str]]:
    """编码探测链的对照表（报告与 ``/documents/loaders`` 端点同源）."""
    reasons = (
        "有 BOM 就是明示，不需要猜；UTF-32 必须排在 UTF-16 之前",
        "不是文本就不该进这条链路；判据只有 NUL 一个（误判的代价不对称）",
        "现代文档的主流；**严格**解码让「成功」本身成为可靠证据",
        "中文旧文档的兜底；GB18030 是 GBK 与 GB2312 的超集",
        "最后一张网：保证总能返回字符串，代价是必须显式看 confident",
    )
    return [
        {"step": str(index + 1), "rule": rule, "reason": reason}
        for index, (rule, reason) in enumerate(zip(ENCODING_CHAIN, reasons))
    ]


__all__ = [
    "BINARY_SCAN_BYTES",
    "BOMS",
    "CJK_FALLBACK_ENCODING",
    "ENCODING_CHAIN",
    "ENCODING_FROM_BOM",
    "ENCODING_FROM_FALLBACK",
    "ENCODING_FROM_LATIN1",
    "ENCODING_FROM_STRICT",
    "DecodedText",
    "TextLoader",
    "decode_bytes",
    "detect_bom",
    "encoding_chain",
    "looks_binary",
    "split_paragraphs",
]
