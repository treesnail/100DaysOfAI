"""块与块集合的形状：分块的产物长什么样（M6-D2）.

day061 把"格式各异的文件"收敛成了 ``Document``；今天的产物是它下面一层：
``Chunk`` 与 ``ChunkSet``。这一层比 ``Document`` 更容易被草率设计，因为
"切出来的片段"看起来只是一个字符串列表，好像不需要什么结构。但下游有
**四个**消费者，每个要的东西都不一样：

| 消费者 | 要什么 | 因此哪个字段必须存在 |
|--------|--------|-------------------|
| 向量库（day009/064） | 稳定的 entry id | ``chunk_id`` |
| 引用溯源（day069） | 这句话出自哪一段、哪一节 | ``start_char`` / ``end_char`` / ``heading_path`` |
| 回归对比（本日） | 是不是同一个策略、同一套参数切的 | ``strategy`` / ``token_measurer`` |
| 质量报告（day071） | 有没有超预算的块、覆盖率是多少 | ``oversized`` / ``token_count`` |

## ``chunk_id`` 与 ``fingerprint`` 是两个不同的问题

这是本模块最值得记住的一条：

```text
chunk_id     sha256(doc_id | strategy | index | text)[:16]    ←「这一块是谁」
fingerprint  sha256(text)[:16]                                ←「这是哪段内容」
```

**只按文本算 id 会坏在"同一段话在同一份文档里出现两次"**（工程文档里的
"注意"小节经常长得一模一样，命令示例更是反复出现）：两个块的 id 相同，
向量库按 id 覆盖，于是"命中了 3 次"在库里只剩 1 条。而把 ``index`` 放进 id
之后，两个块各有各的位置，都能被检索到。

**而跨文档去重恰恰要的相反**：同一段话在 A 文档和 B 文档里各存一份，
是重复存储。所以 ``fingerprint`` 单独留着，用于"这批块里有多少是重复内容"
——它与 day061 的 ``doc_id``、day058 的三元组版本键是同一个东西。

两个问题、两个字段，**混成一个就会出现"要么去重失效、要么块互相覆盖"**，
而两种失效都不会报错。

## ``text`` 与 ``retrieval_text`` 也是两个不同的问题

```text
text            原文里的一段连续子串 —— 用来展示、引用、逐字核对
retrieval_text  heading_path 面包屑 + text —— 用来做 embedding
```

分块器有一条硬不变量：**``document.text[start_char:end_char] == chunk.text``**。
它保证"检索命中的片段"与"引用给出的原文"永远是同一份东西。
把标题前缀写进 ``text`` 会让这条等式失效（前缀不在原文里），于是引用
会引到一段**原文中并不存在**的文字——而这种错误在演示里看不出来，
只有在用户点开出处时才暴露。所以前缀只出现在 ``retrieval_text`` 里，
**它是检索用的视图，不是内容**（与 day061 "全文只能由块拼出"是同一条纪律）。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.chunking.errors import ChunkingError
from smart_research_agent.chunking.tokens import MEASURER_CHARS
from smart_research_agent.documents.types import content_id

#: 四种策略名。**顺序就是决策表的顺序**，也是端点返回的顺序。
STRATEGY_FIXED = "fixed"
STRATEGY_RECURSIVE = "recursive"
STRATEGY_STRUCTURAL = "structural"
STRATEGY_SEMANTIC = "semantic"
STRATEGIES: tuple[str, ...] = (
    STRATEGY_FIXED,
    STRATEGY_RECURSIVE,
    STRATEGY_STRUCTURAL,
    STRATEGY_SEMANTIC,
)

#: ``chunk_id`` 的长度：sha256 前 16 位十六进制（64 bit）.
#: 与 day061 的 ``DOC_ID_LENGTH`` 取同一个量级——两者会拼在同一个 key 里。
CHUNK_ID_LENGTH = 16


def chunk_id_for(doc_id: str, strategy: str, index: int, text: str) -> str:
    """由"文档 + 策略 + 序号 + 文本"算块 id（见模块 docstring 的取舍一）."""
    payload = f"{doc_id}|{strategy}|{index}\n{text}".encode()
    return hashlib.sha256(payload).hexdigest()[:CHUNK_ID_LENGTH]


@dataclass(frozen=True)
class Chunk:
    """一段被切出来的内容块.

    字段分成三组，**每组服务一组不同的读者**：

    ```text
    身份    chunk_id / doc_id / strategy / index     向量库与去重
    内容    text / start_char / end_char             引用溯源
    度量    token_count / token_measurer / oversized  预算与质量报告
    结构    start_block / end_block / heading_path   结构策略的产物
    ```

    ``start_block`` / ``end_block`` 取 ``-1`` 表示"本策略不追踪块区间"
    （固定长度与语义策略只在字符维度上工作）。用一个哨兵值而不是 0，
    因为 0 是一个合法的块下标——**"没有"和"第一块"必须能区分开**。
    """

    chunk_id: str
    doc_id: str
    source: str
    text: str
    index: int
    strategy: str
    token_count: int
    token_measurer: str = MEASURER_CHARS
    start_char: int = 0
    end_char: int = 0
    start_block: int = -1
    end_block: int = -1
    heading_path: tuple[str, ...] = ()
    reason: str = ""
    oversized: bool = False
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ChunkingError(
                f"未知策略 {self.strategy!r}，可选 {', '.join(STRATEGIES)}"
            )
        if not self.text.strip():
            raise ChunkingError(
                "块不允许为空：一个只有空白的块对检索毫无价值，"
                "却会占掉向量库的一条记录与一次 embedding 调用"
            )
        if self.index < 0:
            raise ChunkingError(f"块序号必须非负，收到 {self.index}")
        if self.token_count < 0:
            raise ChunkingError(f"token_count 必须非负，收到 {self.token_count}")
        if self.end_char < self.start_char:
            raise ChunkingError(
                f"区间非法：start_char={self.start_char} 大于 end_char={self.end_char}"
            )
        if len(self.chunk_id) != CHUNK_ID_LENGTH:
            raise ChunkingError(
                f"chunk_id 必须是 {CHUNK_ID_LENGTH} 位十六进制，收到 {self.chunk_id!r}"
            )
        if self.end_block >= 0 and self.end_block < self.start_block:
            raise ChunkingError(
                f"块区间非法：start_block={self.start_block} 大于 end_block={self.end_block}"
            )

    @property
    def char_count(self) -> int:
        """正文字符数（**不含面包屑前缀**：那是检索视图，不是内容）."""
        return len(self.text)

    @property
    def fingerprint(self) -> str:
        """内容指纹：跨文档的"这是哪段内容"（见模块 docstring）."""
        return content_id(self.text)

    @property
    def heading_text(self) -> str:
        """标题路径的字符串形式（``"A > B > C"``；没有标题时为空串）."""
        return " > ".join(self.heading_path)

    @property
    def retrieval_text(self) -> str:
        """进 embedding 的文本 = 标题面包屑 + 正文.

        前缀只在**检索视图**里出现。为什么要加它：一句"阈值设为 0.85"
        单独看没有任何话题信息，而它上面的标题（"语义缓存"）才是
        查询"缓存相似度怎么配"能命中它的原因。**标题是这一块所属语境
        的最短摘要**，白扔掉太可惜。
        """
        if not self.heading_path:
            return self.text
        return f"{self.heading_text}\n{self.text}"

    def to_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典.

        ``include_text=False`` 时只返回度量与身份：一份 500 块的知识库，
        把全文返回一遍足够撑爆任何响应体，而报告通常只需要统计量。
        """
        payload: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source": self.source,
            "index": self.index,
            "strategy": self.strategy,
            "token_count": self.token_count,
            "token_measurer": self.token_measurer,
            "char_count": self.char_count,
            "fingerprint": self.fingerprint,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "start_block": self.start_block,
            "end_block": self.end_block,
            "heading_path": list(self.heading_path),
            "reason": self.reason,
            "oversized": self.oversized,
            "metadata": dict(self.metadata),
        }
        if include_text:
            payload["text"] = self.text
            payload["retrieval_text"] = self.retrieval_text
        return payload

    def summary_line(self) -> str:
        """人类可读的一行摘要（报告里逐块打印）."""
        mark = "!" if self.oversized else " "
        head = self.heading_text or "（无标题）"
        preview = self.text.replace("\n", " ")[:40]
        return (
            f"{mark} #{self.index:<3} {self.token_count:>4} {self.token_measurer} | "
            f"[{self.start_char}:{self.end_char}] | {head} | {preview}"
        )


@dataclass(frozen=True)
class ChunkSet:
    """一份文档在某一种策略下切出的全部块（+ 这一批的度量口径）."""

    doc_id: str
    source: str
    strategy: str
    chunks: tuple[Chunk, ...] = ()
    token_measurer: str = MEASURER_CHARS
    doc_chars: int = 0
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ChunkingError(
                f"未知策略 {self.strategy!r}，可选 {', '.join(STRATEGIES)}"
            )
        if self.doc_chars < 0:
            raise ChunkingError(f"doc_chars 必须非负，收到 {self.doc_chars}")
        expected = tuple(range(len(self.chunks)))
        actual = tuple(chunk.index for chunk in self.chunks)
        if actual != expected:
            # 序号必须是 0..n-1 连续递增。跳号或乱序意味着"有块丢了"，
            # 而那种丢失在下游表现为"检索少了几个片段"——**必须在这里响**。
            raise ChunkingError(
                f"块序号必须是 0..{len(self.chunks) - 1} 的连续递增序列，收到 {actual}"
            )
        for chunk in self.chunks:
            if chunk.doc_id != self.doc_id:
                raise ChunkingError(
                    f"块 #{chunk.index} 的 doc_id={chunk.doc_id!r} 与所属集合"
                    f"{self.doc_id!r} 不一致：那说明两批块被拼到了一起"
                )
            if chunk.strategy != self.strategy:
                raise ChunkingError(
                    f"块 #{chunk.index} 的策略 {chunk.strategy!r} 与集合"
                    f"{self.strategy!r} 不一致"
                )

    @property
    def count(self) -> int:
        """块数."""
        return len(self.chunks)

    @property
    def total_chars(self) -> int:
        """所有块的字符数之和（**重叠会算两次**，见 ``duplication_ratio``）."""
        return sum(chunk.char_count for chunk in self.chunks)

    @property
    def total_tokens(self) -> int:
        """所有块的预算单位之和（按 ``token_measurer`` 的口径）."""
        return sum(chunk.token_count for chunk in self.chunks)

    @property
    def coverage(self) -> float:
        """覆盖率 = 块字符总数 / 原文字符数.

        底下是一条不变量：**每块的文本都是原文的一段连续子串，且原文里
        每一个非空白字符都至少属于一个块**。因此它不会被显著拉低——
        唯一的差额来自"块首尾的空白被修剪掉"（两个块之间的那个空行
        既不属于前一块的结尾，也不属于后一块的开头）。

        这个差额有上界，因此它是**可核对的**：

        ```text
        doc_chars - total_chars <= 2 × (块数 - 1) + 1
                                    ↑ 块间空行        ↑ 原文结尾的换行
        ```

        于是"覆盖率 98% 是否正常"不再是一个感觉问题：一份 310 字符、
        4 块的文档，差额上界是 7——实测 7 就正常，实测 40 就是丢了内容。
        （不变量本身由 ``Chunker._assert_spans`` 强制，违反会直接报错，
        因此覆盖率只可能在上面那个界限内轻微偏低。）
        """
        if self.doc_chars <= 0:
            return 0.0
        return round(self.total_chars / self.doc_chars, 4)

    @property
    def duplication_ratio(self) -> float:
        """重复率 = 重叠带来的放大量（块字符总数相对原文多出来的比例）.

        它直接回答一个成本问题：**这些重叠有没有白花 embedding 的钱**。
        ``overlap=48`` 在 ``max_tokens=320`` 下理论放大 ``48/(320-48) = 17.6%``，
        实测值应当与它接近；差得远说明切分没有按预期走（例如大量块因为
        碰到原子块而过短，重叠相对于块长就显得更大）。

        下界裁到 0：上一条说的"块首尾空白被修剪"会让差额略小于 0，
        而那与重叠是两件不相干的事——**把它显示成 -0.6% 会让人以为
        重叠算法算错了。**
        """
        if self.total_chars <= 0:
            return 0.0
        return max(0.0, round(1 - self.doc_chars / self.total_chars, 4))

    @property
    def oversized_count(self) -> int:
        """超预算且不可切的块数（原子块被整块保留的情形）."""
        return sum(1 for chunk in self.chunks if chunk.oversized)

    def token_stats(self) -> dict[str, float]:
        """块长度的分布（报告里与"预算"并排看）.

        用 **p90** 而不是平均值看上限：平均值会被一堆短块拉下来，
        而真正决定"会不会被模型截断"的是最长的那几个块。
        """
        if not self.chunks:
            return {"min": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0, "mean": 0.0}
        values = sorted(chunk.token_count for chunk in self.chunks)
        return {
            "min": float(values[0]),
            "p50": float(_percentile(values, 0.5)),
            "p90": float(_percentile(values, 0.9)),
            "max": float(values[-1]),
            "mean": round(sum(values) / len(values), 2),
        }

    def heading_paths(self) -> list[tuple[str, ...]]:
        """出现过的标题路径（去重后按字典序，便于逐次运行对比）."""
        return sorted({chunk.heading_path for chunk in self.chunks})

    def oversized(self) -> list[Chunk]:
        """超预算的块（"为什么这一块这么大"要能一眼看到）."""
        return [chunk for chunk in self.chunks if chunk.oversized]

    def knowledge_records(self) -> list[dict[str, Any]]:
        """产出可交给知识库的记录（day009 的 ``VectorStore`` 形状）.

        day061 交出的是**整份文档**，今天交出的是**文档切出来的片段**，
        而**形状不变**——``doc_id`` / ``source`` / ``text`` / ``metadata``
        四个键，这正是 day061 先把结构归一化的回报。

        三个取舍写在明面上：

        1. ``doc_id`` 位置放的是 ``chunk_id``（知识库里的唯一键就是片段）；
        2. ``text`` 放的是**原文片段**而不是带面包屑的检索视图——
           库里存的应该是能直接引用给用户的原文；
        3. 检索视图放 ``metadata["retrieval_text"]``，向量化时用它，
           因此"存的"和"embed 的"不是同一段文本，**这一条必须写出来**
           （否则下一个人会以为向量是用 ``text`` 算的，然后"修"成一个 bug）。
        """
        return [
            {
                "doc_id": chunk.chunk_id,
                "source": chunk.source,
                "text": chunk.text,
                "metadata": {
                    "parent_doc_id": chunk.doc_id,
                    "strategy": chunk.strategy,
                    "index": str(chunk.index),
                    "token_count": str(chunk.token_count),
                    "token_measurer": chunk.token_measurer,
                    "heading_path": chunk.heading_text,
                    "fingerprint": chunk.fingerprint,
                    "oversized": "true" if chunk.oversized else "false",
                    "retrieval_text": chunk.retrieval_text,
                    **dict(chunk.metadata),
                },
            }
            for chunk in self.chunks
        ]

    def to_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        payload: dict[str, Any] = {
            "doc_id": self.doc_id,
            "source": self.source,
            "strategy": self.strategy,
            "token_measurer": self.token_measurer,
            "doc_chars": self.doc_chars,
            "count": self.count,
            "total_chars": self.total_chars,
            "total_tokens": self.total_tokens,
            "coverage": self.coverage,
            "duplication_ratio": self.duplication_ratio,
            "oversized_count": self.oversized_count,
            "token_stats": self.token_stats(),
            "heading_paths": [" > ".join(path) for path in self.heading_paths()],
            "metadata": dict(self.metadata),
        }
        payload["chunks"] = [
            chunk.to_dict(include_text=include_text) for chunk in self.chunks
        ]
        return payload

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        stats = self.token_stats()
        return (
            f"{self.strategy} | {self.count} 块 | 覆盖 {self.coverage:.2%} | "
            f"重复 {self.duplication_ratio:.2%} | 长度 p50 {stats['p50']:.0f} / "
            f"p90 {stats['p90']:.0f} / max {stats['max']:.0f} {self.token_measurer}"
            + (f" | 超预算 {self.oversized_count}" if self.oversized_count else "")
        )

    def table(self) -> list[dict[str, Any]]:
        """逐块的可读视图（报告渲染与端点共用一份）."""
        return [
            {
                "index": chunk.index,
                "chunk_id": chunk.chunk_id,
                "token_count": chunk.token_count,
                "char_count": chunk.char_count,
                "range": f"{chunk.start_char}:{chunk.end_char}",
                "heading_path": chunk.heading_text,
                "reason": chunk.reason,
                "oversized": chunk.oversized,
                "preview": chunk.text.replace("\n", " ")[:60],
            }
            for chunk in self.chunks
        ]


def _percentile(values: list[int], fraction: float) -> float:
    """线性插值分位数（``values`` 必须已升序）.

    与 day059 报告里用的是同一个口径：``rank = fraction * (n - 1)``，
    在相邻两点之间线性插值。**不用"取第 k 个"那种写法**——样本数变化时
    它会跳变，而"这次比上次 p90 高了 3"这种结论需要连续可比。
    """
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    rank = fraction * (len(values) - 1)
    low = int(rank)
    high = min(low + 1, len(values) - 1)
    weight = rank - low
    return values[low] * (1 - weight) + values[high] * weight


__all__ = [
    "CHUNK_ID_LENGTH",
    "STRATEGIES",
    "STRATEGY_FIXED",
    "STRATEGY_RECURSIVE",
    "STRATEGY_SEMANTIC",
    "STRATEGY_STRUCTURAL",
    "Chunk",
    "ChunkSet",
    "chunk_id_for",
]
