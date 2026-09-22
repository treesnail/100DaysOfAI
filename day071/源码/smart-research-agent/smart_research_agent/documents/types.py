"""文档与块的统一结构：把"格式各异的文件"收敛成一种可被下游消费的形状（M6-D1）.

M6 的第一天要解决一件看起来平淡、实际上决定后面九天上限的事：

> **下游（分块、向量化、检索、评估）只应该认识一种东西。**

如果分块器要分别处理 Markdown 的标题、PDF 的文本流、docx 的表格，
那么每加一种格式就要动一次分块逻辑——而分块策略本身已经够复杂了
（day062 要讲固定长度、递归、语义、结构四种）。因此今天先做一次
**结构归一**：

```text
各种格式  →  Document（全文 + 块序列 + 元数据）  →  下游只认识 Document
```

## 为什么是"块序列"而不是"一段纯文本"

纯文本丢掉了 M6 后面七天要用的全部信息：

| 信息 | 谁需要它 |
|------|---------|
| 标题层级 | day062 的结构分块（按标题切）、day069 的 RAG 提示（把标题当上下文） |
| 代码块与其语言 | day062 不该把代码从中间切开 |
| 表格的行列结构 | day069 的表格问答（渲染成 Markdown 再喂给模型） |
| 每一块的顺序 | 引用溯源（"这句话出自文档的第几段"） |

## `doc_id` 是内容指纹，不是文件名

```python
doc_id = sha256(规范化后的全文)[:16]
```

三条性质，每条都有具体用途：

1. **同一份内容从两个路径读进来得到同一个 id**：复制了一份文件、从两个
   渠道各下了一份同一份规范，在入库时会被认成同一份文档（day061 的去重）；
2. **改一个字节 id 就变**：id 变了说明内容变了，这与 day058 的三元组
   "内容寻址"完全是同一种设计；
3. **它不包含文件名、路径与时间**：那些是"这份内容放在哪"，
   而不是"这是哪份内容"——把它们混进 id 会让同一份内容因为换了个目录
   就变成两份，而去重也就失效了。

因此**规范化的规则必须被写死并被测试覆盖**（``normalize_text``）：
换行统一、行尾空白去掉、连续空行折叠。任何一条改动都会改变全量文档的
`doc_id`，属于"会让历史索引全部失效"的变更。
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

from smart_research_agent.documents.errors import DocumentError

#: 六种块类型。刻意保持得很少：**每一种都要有下游消费者**，
#: 加进来一个"没人用"的类型只会让分块器多一个要考虑的分支。
BLOCK_HEADING = "heading"
BLOCK_PARAGRAPH = "paragraph"
BLOCK_CODE = "code"
BLOCK_LIST_ITEM = "list_item"
BLOCK_TABLE = "table"
BLOCK_QUOTE = "quote"
BLOCK_KINDS: tuple[str, ...] = (
    BLOCK_HEADING,
    BLOCK_PARAGRAPH,
    BLOCK_CODE,
    BLOCK_LIST_ITEM,
    BLOCK_TABLE,
    BLOCK_QUOTE,
)

#: ``doc_id`` 的长度：``sha256`` 的前 16 位十六进制（64 bit）。
#: 与 day058 的版本键取同一个量级：在"最多几十万份文档"的规模上，
#: 64 bit 的碰撞概率可以忽略，而它在日志与文件名里都够短。
DOC_ID_LENGTH = 16

#: 连续空行的折叠上限：段落之间最多留**一个**空行。
#: 不折叠的话，"同一段文字被不同的编辑器导出"会产生不同的指纹——
#: 而中文文档在这一点上尤其混乱（有的导出行尾带 `\r`，有的带两个空行）。
MAX_BLANK_RUN = 2


def normalize_text(text: str) -> str:
    """把文本收敛成**唯一的标准形**（``doc_id`` 就建立在它上面）.

    四条规则，每条都对应一类"看起来一样、字节不同"的差异：

    | 规则 | 消除的差异 |
    |------|-----------|
    | ``\\r\\n`` / ``\\r`` → ``\\n`` | Windows 与 Unix 换行 |
    | 去掉每行行尾空白 | 编辑器留下的尾随空格 |
    | 连续空行折叠到 1 个 | 导出的多余空行 |
    | 去掉首尾空白并在结尾补一个 ``\\n`` | 有无结尾换行 |

    最后一条的"补一个换行"是刻意的：**一个"没有结尾换行"的文件与
    "有一个结尾换行"的文件是同一份内容**，而它们在字节上不同。
    统一补一个，让两者得到同一个指纹。

    不做的事也要说清楚：**不做 Unicode 规范化**（不把全角标点换成半角、
    不折叠空格）。理由是中英文混排的技术文档里，全角逗号与半角逗号
    经常是内容差异而不是格式差异，把它们合并会改变检索结果——
    而这类"看起来更干净"的规范化最难被发现。
    """
    if not text:
        return ""
    unified = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in unified.split("\n")]
    collapsed: list[str] = []
    blanks = 0
    for line in lines:
        if line:
            blanks = 0
            collapsed.append(line)
            continue
        blanks += 1
        if blanks < MAX_BLANK_RUN:
            collapsed.append(line)
    body = "\n".join(collapsed).strip("\n")
    return f"{body}\n" if body else ""


def content_id(text: str) -> str:
    """由规范化文本算内容指纹（``doc_id``）."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:DOC_ID_LENGTH]


@dataclass(frozen=True)
class Block:
    """文档里的一个结构块.

    ``text`` 是所有块类型都有的（表格时它是**渲染成 Markdown 的文本**，
    便于直接进提示词），结构化的部分放在各自的专用字段里。
    这样"只想拿文本"的调用方不需要区分六种类型，而"需要结构"的调用方
    也不会被迫去解析文本。
    """

    kind: str
    text: str
    level: int = 0
    language: str = ""
    rows: tuple[tuple[str, ...], ...] = ()
    meta: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in BLOCK_KINDS:
            raise DocumentError(
                f"未知块类型 {self.kind!r}，可选 {', '.join(BLOCK_KINDS)}"
            )
        if self.kind == BLOCK_HEADING and self.level not in (1, 2, 3, 4, 5, 6):
            raise DocumentError(
                f"标题层级必须是 1~6，收到 {self.level}（Markdown 与 HTML 都只有这六档）"
            )
        if self.kind == BLOCK_TABLE and not self.rows:
            raise DocumentError("表格块必须有 rows：没有行列结构的表格与普通段落没有区别")

    @property
    def is_empty(self) -> bool:
        """文本为空且没有表格行时视为空块（加载器不该把它们放进来）."""
        return not self.text.strip() and not self.rows

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        payload = asdict(self)
        payload["rows"] = [list(row) for row in self.rows]
        return payload

    def summary_line(self) -> str:
        """人类可读的一行摘要（报告里按它逐块打印）."""
        head = self.kind if self.level == 0 else f"{self.kind}{self.level}"
        extra = f" [{self.language}]" if self.language else ""
        if self.kind == BLOCK_TABLE:
            extra += f" {len(self.rows)}×{len(self.rows[0])}"
        preview = self.text.replace("\n", " ")[:48]
        return f"{head}{extra}: {preview}"


@dataclass(frozen=True)
class Document:
    """一份归一化后的文档：全文 + 块序列 + 元数据 + 内容指纹."""

    source: str
    media_type: str
    text: str
    blocks: tuple[Block, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)
    doc_id: str = ""

    def __post_init__(self) -> None:
        if not self.source:
            raise DocumentError("文档必须有 source：报告与引用溯源都要按它定位")
        if not self.media_type:
            raise DocumentError("文档必须有 media_type：加载器选择与统计都按它分流")
        if self.doc_id and self.doc_id != content_id(self.text):
            raise DocumentError(
                "给定的 doc_id 与文本内容不符：doc_id 是内容指纹，"
                "外部传入一个不匹配的值会让去重与溯源同时失效"
            )

    @property
    def char_count(self) -> int:
        """正文字符数（含空白，与 ``len(text)`` 一致）."""
        return len(self.text)

    @property
    def fingerprint(self) -> str:
        """内容指纹（``doc_id`` 缺失时现场算一次）."""
        return self.doc_id or content_id(self.text)

    def headings(self) -> list[Block]:
        """全部标题块（day062 的结构分块按它切）."""
        return [block for block in self.blocks if block.kind == BLOCK_HEADING]

    def block_counts(self) -> dict[str, int]:
        """按类型统计块数（**六种类型都出现，取值为 0 的也在**）.

        固定键集合是刻意的：报告里的表头不该随文档内容变化，
        否则两张报告没法逐列对比。
        """
        counts = dict.fromkeys(BLOCK_KINDS, 0)
        for block in self.blocks:
            counts[block.kind] += 1
        return counts

    def to_dict(self, *, include_blocks: bool = True) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典.

        ``include_blocks=False`` 时只返回摘要：一份几百页的文档，
        其块序列会比全文还大，而报告通常只需要统计量。
        """
        payload: dict[str, Any] = {
            "doc_id": self.fingerprint,
            "source": self.source,
            "media_type": self.media_type,
            "char_count": self.char_count,
            "block_count": len(self.blocks),
            "block_counts": self.block_counts(),
            "metadata": dict(self.metadata),
            "preview": self.text[:120],
        }
        if include_blocks:
            payload["blocks"] = [block.to_dict() for block in self.blocks]
        return payload

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        counts = self.block_counts()
        detail = " ".join(
            f"{kind}={counts[kind]}" for kind in BLOCK_KINDS if counts[kind]
        )
        return (
            f"{self.fingerprint} {self.media_type} {self.char_count} 字符 | "
            f"{len(self.blocks)} 块（{detail or '空'}）| {self.source}"
        )


def render_table(rows: tuple[tuple[str, ...], ...] | list[tuple[str, ...]]) -> str:
    """把表格渲染成 Markdown 文本（``Block.text`` 用的就是它）.

    ``Block.rows`` 保留结构，``Block.text`` 保留"可以直接进提示词"的形态。
    两份是同一个东西的两种视图，因此 **text 由 rows 生成**，
    不允许各自独立构造（与 ``make_document`` 全文由块拼出是同一条纪律）。

    三个格式（Markdown / HTML / docx）的表格都走这一个渲染器：
    **同一份事实只有一份渲染实现**，否则三种格式的表格会以三种样子进提示词。
    """
    if not rows:
        return ""
    header = list(rows[0])
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows[1:]:
        cells = list(row) + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(cells[: len(header)]) + " |")
    return "\n".join(lines)


def make_document(
    *,
    source: str,
    media_type: str,
    blocks: list[Block] | tuple[Block, ...],
    metadata: dict[str, str] | None = None,
) -> Document:
    """由块序列组装一份文档（**全文由块拼出，不接受另外传入的 text**）.

    这是本模块最重要的一条约束：全文**只能**由块序列拼接得到，
    因此"全文"与"块序列"永远是同一份内容的两种视图，不会出现
    "分块器按块切、检索器按 text 找"这种两套内容并存的情况。

    一开始的实现允许调用方传一个 ``text`` 覆盖拼接结果，理由是
    "加载器可能有更准确的全文"。它被删掉了：**那个参数的存在本身
    就是在允许两份内容不一致**，而一旦不一致，两份都会有人依赖——
    这类缺陷不会报错，只会让"检索命中的段落"与"引用到的原文"
    长期对不上。相关的测试是
    ``test_full_text_is_always_derived_from_the_blocks``。

    空块会被丢掉：一个只含空白的段落对下游毫无价值，而它会白白占掉
    分块器的一个预算单位（day062 的 token 预算）。
    """
    kept = [block for block in blocks if not block.is_empty]
    body = normalize_text("\n\n".join(block.text for block in kept))
    return Document(
        source=source,
        media_type=media_type,
        text=body,
        blocks=tuple(kept),
        metadata=dict(metadata or {}),
        doc_id=content_id(body),
    )


__all__ = [
    "BLOCK_CODE",
    "BLOCK_HEADING",
    "BLOCK_KINDS",
    "BLOCK_LIST_ITEM",
    "BLOCK_PARAGRAPH",
    "BLOCK_QUOTE",
    "BLOCK_TABLE",
    "DOC_ID_LENGTH",
    "MAX_BLANK_RUN",
    "Block",
    "Document",
    "content_id",
    "make_document",
    "normalize_text",
    "render_table",
]
