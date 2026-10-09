"""关键词一路：**零依赖的 BM25**（M6-D6）.

向量检索回答"意思像不像"，关键词检索回答"**这个词在不在**"。两者错的不是同一批
查询，这正是混合检索的全部理由：

```text
向量路擅长的    "怎么让缓存不要每次都重算" → 命中"语义缓存用分位数标定阈值"
关键词路擅长的  "ERR-2043"              → 命中"ERR-2043 表示向量维度不一致"
```

反过来各有一个死穴：向量路对**编号、函数名、报错码、专有名词**常常"差不多就行"
（它们在训练语料里太稀，编码器只好把它们映射到一个模糊的方向），
而关键词路对**同义改写**完全无力（"重算"与"重新计算"在它眼里是两个词）。
把两路合起来不是为了"更强"，而是为了**两种失败不再是同一种失败**。

## 为什么不引入分词库（本模块最该被记住的一处取舍）

``jieba`` / ``rank_bm25`` 都能让这段代码短一半。不用它们的理由有两条，缺一不可：

```text
1. 不新增依赖   本课的全部产物必须能在"只有 Python + 标准库"的机器上复现，
                而 BM25 与分词都是**几十行的算术**，不是一个值得为之引入依赖的能力。
                （jieba 还会带一个约 5 MB 的词典文件，它会把"离线可复现"变成
                "词典版本也会影响结果"——那正是评估最怕的东西。）
2. 评估要可复现  分词器是有版本的（jieba 换版本会切出不同的词），
                于是同一天、同一份语料、同一批查询会给出不同的 BM25 分数，
                而 day071 的 RAG 评估要求"分数变了必须能归因到某个显式参数"。
                相邻 2-gram 的分词法**没有版本**：它的定义就是"字符串的相邻两字"。
```

代价是真实的，写在这里免得被当成"没有缺点"：2-gram 会把"语义"与"义语"当作两个
不同的词元（词表比真实词表大、精确率略低），也不认识"未登录词"。这两条都是
**已知边界**（``types.RETRIEVAL_LIMITATIONS`` 里各有一条）。

## 分词规则（确定性，无参数）

```text
先按 [a-z0-9_]+ | [\\u4e00-\\u9fff]+ 切段（全角标点、空格、连字符都是分隔符）
拉丁/数字段    原样保留（"err"、"2043"、"retry_budget" 各是一个词元；大小写先折叠）
汉字段长度 1   保留该单字（"词" → 词）
汉字段长度 ≥2  单字 + **相邻 2-gram**（"语义缓存" → 语/义/缓/存/语义/义缓/缓存）
```

"单字 + 2-gram"同时给出两种粒度的证据：单字让"重算"与"重新计算"共享 重/算 两个
词元（在不引入同义词表的前提下蹭到一点召回），2-gram 让"语义"与"义语"区分开
（提高精确率）。**两个都留**而不是只留 2-gram：只留 2-gram 时单字查询
（"熵"、"税"）会一个词元都对不上，而单字查询在中文里是真实存在的。

## 打分：BM25，两个参数各自管一件事

```text
idf(t) = log(1 + (N - df + 0.5) / (df + 0.5))              ← 恒正（见下）
score  = Σ_t idf(t) · tf · (k1 + 1) / (tf + k1 · (1 - b + b · dl / avgdl))
```

``k1``（词频饱和）与 ``b``（长度归一化）的缺省值 ``1.5`` / ``0.75`` 出自
Robertson & Zaragoza 的经典取值（*The Probabilistic Relevance Framework:
BM25 and Beyond*, 2009），也是几乎所有实现里的默认值；它们不是本课标定出来的，
**引用而不是发明**在这里很重要：一个"自己调的 1.7"会让别人无法判断
"结果不同是因为语料还是因为参数"。

**``log(1 + …)`` 里的那个 1 是本模块刻意选的写法**。经典 BM25 用
``log((N - df + 0.5) / (df + 0.5))``，它在 ``df > N/2`` 时变成**负数**——
于是一个"到处都出现的词"会给文档**减分**。两种写法的排序在多数语料上几乎一样，
但负 IDF 会带来一个很难解释的现象：``score`` 越小越不相关**和**``score`` 为负
意味着"减去相关性"混在一起，而 0 分又被本模块用作"完全没有共同词元"的哨兵
（见 ``search``）。恒正的 IDF 让"0 分"这件事只有一个含义。

## 只返回"至少命中一个词元"的文档（本模块的第二个关键取舍）

BM25 会给**完全不相关的文档**一个 0 分，而它们在名次上位于所有真实命中之后。
如果把它们也算作"关键词路的命中"，那么在名次靠后的位置上会躺着一批
"和查询毫无关系"的记录——而 RRF（``fusion.py``）恰恰**只看名次不看分数**，
它会老老实实给"关键词路第 3 名"一个 1/(k+3) 的贡献。那条证据是假的。

因此 ``search`` 只返回 ``score > 0`` 的文档，并把被排除的条数写进 ``notes``：
"0 分不是'相关性很低'，而是'这次查询完全没提到它'"。这条规则让
"关键词路一条都没召回"成为一个**能说清楚的结论**，而不是"返回了 5 条 0 分记录"。
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.retrieval.errors import LexicalError, QueryError
from smart_research_agent.retrieval.filters import validate_where
from smart_research_agent.retrieval.types import CHANNEL_BM25
from smart_research_agent.vectorstore.base import DEFAULT_TOP_K, MAX_TOP_K
from smart_research_agent.vectorstore.filters import compile_filter

#: 词频饱和参数的缺省值。出处：Robertson & Zaragoza（2009）的经典取值，
#: 与 Lucene / Elasticsearch 的 ``k1`` 默认值同源。**不是本课标定出来的**——
#: 引用一个公开取值，别人才分得清"结果不同"是语料造成的还是参数造成的。
DEFAULT_K1 = 1.5

#: 长度归一化强度的缺省值。出处同上（``b`` 的经典取值 0.75）。
#: ``b=0.75`` 的意思是"长文档打个折，但不把长度当成决定性因素"。
DEFAULT_B = 0.75

#: 切段规则：拉丁/数字/下划线一段，汉字一段。标点、空白、连字符都是分隔符
#: （因此 ``"ERR-2043"`` 会切成 ``err`` 与 ``2043`` 两个词元——
#: 这是刻意的：编号的"前缀"与"序号"分别命中时都能给分）。
TOKEN_PATTERN = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]+")

#: 判断一个切段是不是汉字段（用来决定要不要生成 2-gram）。
CJK_ONLY = re.compile(r"^[\u4e00-\u9fff]+$")


# --------------------------------------------------------------------------- #
# 分词
# --------------------------------------------------------------------------- #


def tokenize(text: str) -> list[str]:
    """把一段文本切成词元（规则见模块 docstring，**确定性、零依赖、无参数**）.

    为什么不是"按空格切 + 去标点"：中文没有空格，那种切法会把一整句话当成一个
    词元，于是 BM25 退化成"整句精确匹配"——一个看起来在工作、实际上永远
    只命中原句的检索器（而它的表现是"关键词路召回率很低"，不是报错）。

    为什么不是"逐字切"（只留单字）：单字粒度会把"语义"与"义语"混在一起
    （它们共享 语/义/义/语），也会让"缓存"与"存缓"等价。2-gram 补上区分度，
    单字补上单字查询与跨词边界的召回，两者一起才是这里的方案。

    返回的列表**保留重复**（同一个词元在文本里出现两次就出现两次）：
    "词频"这件事必须能被读出来（``tf`` 是 BM25 的两个输入之一），
    去重是打分那一步的事（那里对查询词元去重，避免"写两遍翻倍"）。
    """
    if not isinstance(text, str):
        raise QueryError(
            f"tokenize 需要字符串，收到 {type(text).__name__}。"
            "要索引一条记录请用 LexicalDocument（它会先校验类型）；"
            "要给它一个向量请用 vectorstore 的那一层。"
        )
    lowered = text.lower()
    tokens: list[str] = []
    for chunk in TOKEN_PATTERN.findall(lowered):
        if not CJK_ONLY.match(chunk):
            tokens.append(chunk)
            continue
        if len(chunk) == 1:
            tokens.append(chunk)
            continue
        # 单字在前、2-gram 在后：顺序本身也是确定性的一部分（同一个输入
        # 在任何机器上给出同一个列表，逐位可比）。
        tokens.extend(chunk)
        tokens.extend(chunk[index : index + 2] for index in range(len(chunk) - 1))
    return tokens


def _idf(total_docs: int, doc_freq: int) -> float:
    """``log(1 + (N - df + 0.5) / (df + 0.5))``：**恒正**的 IDF（理由见模块 docstring）.

    取对数而不是直接比值：比值会让"df 从 1 变 2"与"df 从 101 变 102"给出
    同样的降幅，而对数下的降幅与"稀有度"的关系更接近人对"这个词有多特别"的直觉。

    它对本模块是**公开可核对**的（测试直接对它断言单调性）：BM25 的排序
    不变量（词越稀有权重越大、``df`` 增大时权重单调下降）都落在这一个函数上。
    """
    return math.log(1.0 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))


# --------------------------------------------------------------------------- #
# 三副形状
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BM25Params:
    """BM25 的两个参数（``k1`` 词频饱和 / ``b`` 长度归一化）.

    做成一个 frozen dataclass 而不是两个散参数，理由与
    ``vectorstore.metrics.MetricSpec`` 相同：它们是**一对**——
    "k1 调大、b 调小"是同一个实验的两个旋钮，而它们必须一起进报告
    （``describe()`` 会把它们端出来，"这次用的是哪组参数"不能被省略）。
    """

    k1: float = DEFAULT_K1
    b: float = DEFAULT_B

    def __post_init__(self) -> None:
        for label, value in (("k1", self.k1), ("b", self.b)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise QueryError(
                    f"BM25Params.{label} 必须是数字，收到 {type(value).__name__}。"
                    f"字符串形式的参数会在打分时变成 TypeError——请在这里就写成数字。"
                )
            if not math.isfinite(float(value)):
                raise QueryError(
                    f"BM25Params.{label} 必须是有限数，收到 {value!r}。"
                    "nan 参与的比较恒为 False，于是**任何查询都不再返回结果**，"
                    "而它在配置里看起来只是一个参数。"
                )
        k1 = float(self.k1)
        b = float(self.b)
        if k1 < 0.0:
            raise QueryError(
                f"BM25Params.k1 必须 >= 0，收到 {k1}。"
                "k1 是词频饱和项：它必须非负（分母 tf + k1·… 里出现它），"
                "而 k1=0 有明确含义——'完全忽略词频，只看出现过没有'。"
            )
        if not 0.0 <= b <= 1.0:
            raise QueryError(
                f"BM25Params.b 必须落在 [0, 1]，收到 {b}。"
                "b=1 表示完全按文档长度归一化（长文档里的同一个词更不值钱），"
                "b=0 表示完全不归一化；超出这个区间的取值会让分母变成负数，"
                "于是'长文档'反而得到更高的分数——一个不会报错、只会反过来的排序。"
            )
        object.__setattr__(self, "k1", k1)
        object.__setattr__(self, "b", b)

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return {"k1": self.k1, "b": self.b}

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        return f"BM25(k1={self.k1}, b={self.b})"


@dataclass(frozen=True)
class LexicalDocument:
    """关键词索引里的一篇文档：**id + 正文 + 元数据**（向量那条记录的投影）.

    它与 ``vectorstore.types.VectorRecord`` 是同一份数据的两个视角：
    那条带向量（给排序用），这条不带（关键词一路**一个字节的向量都不需要**）。
    刻意不写成"``VectorRecord`` 去掉 vector"的继承关系：两个形状的
    **校验规则不同**（这条要求正文非空——一篇没有正文的文档在关键词索引里
    没有意义，而向量记录允许空正文，因为"只按向量过滤"是一种真实用法）。

    ``metadata`` 原样带进来，只为一件事：``where`` 过滤。它不参与打分，
    因此"关键词一路的分数只由正文决定"这句话是能被验证的。
    """

    record_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, str) or not self.record_id.strip():
            raise QueryError(
                f"LexicalDocument.record_id 必须是非空字符串，收到 {self.record_id!r}："
                "没有 id 就无法被去重、无法与向量路的命中对齐、无法回库取原文。"
            )
        if not isinstance(self.text, str):
            raise QueryError(
                f"LexicalDocument.text 必须是字符串，收到 {type(self.text).__name__}"
            )
        if self.text and not self.text.strip():
            raise QueryError(
                f"LexicalDocument.text 不能只有空白（id={self.record_id!r}）："
                "它与空串不同——空串是'这条没有可索引的正文'这个事实，"
                "而空白串会让分词给出一个非空但全是标点的结果。"
            )
        if not isinstance(self.metadata, dict):
            raise QueryError(
                f"LexicalDocument.metadata 必须是字典，收到 {type(self.metadata).__name__}"
            )

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return {
            "record_id": self.record_id,
            "char_count": len(self.text),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LexicalHit:
    """关键词一路的一条命中：id + BM25 分数 + 名次 + 正文与元数据.

    与 ``types.RetrievalHit`` 有**同一组字段名**（``record_id`` / ``score`` /
    ``rank`` / ``text`` / ``metadata``），这不是巧合：融合层要对两路的命中
    做同一件事（读名次、读分数、按 id 去重），字段名一致才能让融合写成
    一份实现而不是两份。差别只在 ``channel``——这里没有这个字段，
    因为它恒为 ``CHANNEL_BM25``（用一个属性给出，见下）。
    """

    record_id: str
    score: float
    rank: int
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, str) or not self.record_id.strip():
            raise QueryError(
                f"LexicalHit.record_id 必须是非空字符串，收到 {self.record_id!r}"
            )
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            # 类型检查必须在 isfinite 之前：``float("1.0")`` 是能转换的，
            # 于是"分数是个字符串"会一路活到排序那一步（那里比较字符串与浮点
            # 会得到一个与相关性无关的顺序）。这里当场拒掉。
            raise QueryError(
                f"LexicalHit.score 必须是数字，收到 {type(self.score).__name__}"
                f"（{self.score!r}）。BM25 的分数是算出来的浮点数，"
                "字符串形式的分数会让排序变成字典序比较。"
            )
        if not math.isfinite(float(self.score)):
            raise QueryError(
                f"LexicalHit.score 必须是有限数，收到 {self.score!r}："
                "BM25 的分数必须是有限数，inf 会让它永远排第一。"
            )
        if not isinstance(self.rank, int) or isinstance(self.rank, bool) or self.rank < 0:
            raise QueryError(
                f"LexicalHit.rank 必须是非负整数（从 0 起），收到 {self.rank!r}"
            )
        if not isinstance(self.metadata, dict):
            raise QueryError(
                f"LexicalHit.metadata 必须是字典，收到 {type(self.metadata).__name__}"
            )

    @property
    def channel(self) -> str:
        """这一路的名字（``CHANNEL_BM25``）；它是属性而不是字段，理由见类 docstring."""
        return CHANNEL_BM25

    def to_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典（``include_text`` 与 ``RetrievalHit`` 同义）."""
        payload: dict[str, Any] = {
            "rank": self.rank,
            "record_id": self.record_id,
            "score": round(self.score, 6),
            "channel": self.channel,
            "metadata": dict(self.metadata),
        }
        if include_text:
            payload["text"] = self.text
        return payload

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        preview = self.text.replace("\n", " ")[:36] if self.text else ""
        return f"#{self.rank} {self.record_id} [bm25] score={self.score:.6f} | {preview}"


@dataclass(frozen=True)
class LexicalSearchResult:
    """关键词一路的一次检索：命中 + **"为什么一条都没召回"的可回答证据**.

    ``matched_terms`` / ``missing_terms`` 是本模块的卖点之一。关键词路的失败
    与向量路的失败形状不同：向量路总能返回"最像的 K 条"（哪怕都不像），
    而关键词路会**干脆地一条都不返回**。那时唯一能回答"为什么"的事实就是
    "查询的那些词元里，有几个在词表里出现过"：

    ```text
    missing_terms 全等于 query_terms  → 一个词都不认识（纯语义改写 / 用了别的语言）
    missing_terms 为空                → 词都认识，但没有任何一篇同时包含它们
    matched_terms 非空且 hits 非空     → 正常命中
    ```

    前两种都表现为"空结果"，但处置动作不同（前者要换查询说法或补语料，
    后者要检查过滤条件与语料分布），因此这里把它们**分开列出来**而不是
    合成一个布尔值——与 ``types.EMPTY_REASONS`` 的那条纪律同源。
    """

    hits: tuple[LexicalHit, ...] = ()
    candidates: int = 0
    filter_applied: bool = False
    query_terms: tuple[str, ...] = ()
    matched_terms: tuple[str, ...] = ()
    missing_terms: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.candidates < 0:
            raise QueryError(f"LexicalSearchResult.candidates 必须非负，收到 {self.candidates}")
        if not isinstance(self.hits, tuple):
            raise QueryError(
                f"LexicalSearchResult.hits 必须是 tuple，收到 {type(self.hits).__name__}"
            )

    @property
    def count(self) -> int:
        """命中条数."""
        return len(self.hits)

    @property
    def is_empty(self) -> bool:
        """这一路是否一条都没召回（**它是融合要看的那个信号**）."""
        return not self.hits

    def ids(self) -> list[str]:
        """命中的记录 id，按名次排列."""
        return [hit.record_id for hit in self.hits]

    def to_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典（默认不带正文，与 ``RetrievalResult`` 同取向）."""
        return {
            "count": self.count,
            "candidates": self.candidates,
            "filter_applied": self.filter_applied,
            "query_terms": list(self.query_terms),
            "matched_terms": list(self.matched_terms),
            "missing_terms": list(self.missing_terms),
            "notes": list(self.notes),
            "hits": [hit.to_dict(include_text=include_text) for hit in self.hits],
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        head = "、".join(f"{hit.record_id}:{hit.score:.4f}" for hit in self.hits[:3])
        return (
            f"bm25 | 候选 {self.candidates} | 命中 {self.count} | "
            f"词元 {len(self.query_terms)} 个（缺 {len(self.missing_terms)}）| {head or '（无）'}"
        )


# --------------------------------------------------------------------------- #
# 索引
# --------------------------------------------------------------------------- #


class LexicalIndex:
    """一份**内存里**的倒排索引 + BM25 打分（零依赖、可逐位复现）.

    它刻意不做持久化，也不做增量更新：关键词索引的构建成本是"分词 + 计数"，
    在一份百万字的语料上是秒级，而**落盘会引入第二种版本口径**
    （向量索引有清单与版本号，见 ``indexing``）——两份版本号一定会分家，
    而分家之后"这次检索用的是哪一版"会取决于读的是哪一份文件。
    因此这一路的口径是：**每次进程启动/每次注入时从向量库重建**
    （``from_backend``），"当前索引是什么样"由 ``describe()`` 如实报告。

    ``add`` 对**已存在的 id 是替换**（不是拒绝）。理由：这个索引是
    "这批文档现在长什么样"的投影，而一份投影的正确语义是"以最后一次写入为准"；
    拒绝会让"改了正文之后重建"变成一次需要先清空的麻烦事。
    真正的防错在别处：``vectorstore.add``（"这批数据必须是新的"）已经在
    写入侧拦住了重复建库，这里再拦一次只会让重建路径难写。
    """

    def __init__(self, *, params: BM25Params | None = None, name: str = "lexical") -> None:
        if not isinstance(name, str) or not name.strip():
            raise QueryError(f"LexicalIndex.name 必须是非空字符串，收到 {name!r}")
        if params is not None and not isinstance(params, BM25Params):
            raise QueryError(
                f"params 必须是 BM25Params，收到 {type(params).__name__}。"
                "写法：LexicalIndex(params=BM25Params(k1=1.2, b=0.6))"
            )
        self._name = name.strip()
        self._params = params if params is not None else BM25Params()
        # 四份事实各自独立存：文档本身、每篇的词频、每篇的长度、每个词元的 df。
        # 不做"存一份分词结果再每次重算 df"的省事写法——那会让每次 search 变成 O(N)，
        # 而 df 正是"加一篇"时唯一需要增量维护的东西。
        self._documents: dict[str, LexicalDocument] = {}
        self._term_freq: dict[str, dict[str, int]] = {}
        self._lengths: dict[str, int] = {}
        self._doc_freq: dict[str, int] = {}
        self._total_length = 0

    # ------------------------------------------------------------------ 装配

    @classmethod
    def build(
        cls,
        documents: Iterable[LexicalDocument],
        *,
        params: BM25Params | None = None,
        name: str = "lexical",
    ) -> LexicalIndex:
        """从一批文档建一份索引（逐篇 ``add``，重复 id 以最后一篇为准）."""
        index = cls(params=params, name=name)
        for document in documents:
            index.add(document)
        return index

    @classmethod
    def from_backend(
        cls,
        backend: Any,
        *,
        params: BM25Params | None = None,
        name: str = "lexical",
    ) -> LexicalIndex:
        """从**向量库后端**把正文与元数据搬过来重建一次（本模块推荐入口）.

        走 ``ids()`` + ``get_many()``——与 ``Retriever`` 体检时用的是同两个原语，
        因此"关键词索引里有哪几篇"与"向量库里有哪些记录"是同一份事实的两次读取。

        ``get_many`` 的实现是"**跳过不存在的**"（``vectorstore.base`` 的原文），
        这里刻意把它升级成错误：库里报 N 条、只取回 M 条，意味着记录表与索引
        不同步（day064 那个经典故障）。如果照常建索引，后果是**关键词路静默少召回**
        ——少的那几条不会出现在任何数字里，只会表现为"关键词路好像漏了什么"。
        因此宁可在这里响（``LexicalError``，"去重建"那一族）。
        """
        for method in ("ids", "get_many"):
            if not hasattr(backend, method):
                raise QueryError(
                    f"from_backend 需要一个实现了 ids() / get_many() 的后端，"
                    f"收到 {type(backend).__name__}（缺 {method}()）。"
                    "要直接给一批文档请用 LexicalIndex.build([LexicalDocument(...)])。"
                )
        ids = [str(record_id) for record_id in backend.ids()]
        records = list(backend.get_many(ids))
        if len(records) != len(ids):
            raise LexicalError(
                f"库里报 {len(ids)} 条记录，但只取回 {len(records)} 条："
                "记录表与向量索引不同步（get_many 会跳过不存在的 id）。"
                "关键词索引**不能**照常建起来——那会让关键词路静默少召回那几条，"
                "而少了多少不会出现在任何报告里。出路：重建关键词索引之前先修库"
                "（补回缺失记录或重建清单），再走 LexicalIndex.from_backend。"
            )
        return cls.build(
            (
                LexicalDocument(
                    record_id=str(record.record_id),
                    text=str(record.text),
                    metadata=dict(record.metadata),
                )
                for record in records
            ),
            params=params,
            name=name,
        )

    def add(self, document: LexicalDocument) -> None:
        """加入/替换一篇文档（重复 id → 先忘掉旧的那一篇，再按新的算）."""
        if not isinstance(document, LexicalDocument):
            raise QueryError(
                f"add 只接受 LexicalDocument，收到 {type(document).__name__}。"
                "字典形状的输入请先经 LexicalIndex.from_backend（它直接从库里取），"
                "或显式构造 LexicalDocument(record_id=..., text=..., metadata=...)。"
            )
        if document.record_id in self._documents:
            self._forget(document.record_id)
        tokens = tokenize(document.text)
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        self._documents[document.record_id] = document
        self._term_freq[document.record_id] = counts
        self._lengths[document.record_id] = len(tokens)
        self._total_length += len(tokens)
        for term in counts:
            self._doc_freq[term] = self._doc_freq.get(term, 0) + 1

    def _forget(self, record_id: str) -> None:
        """把一篇文档从四份事实里一起摘掉（``add`` 的替换语义要靠它）."""
        counts = self._term_freq.pop(record_id, {})
        self._total_length -= self._lengths.pop(record_id, 0)
        self._documents.pop(record_id, None)
        for term in counts:
            remaining = self._doc_freq.get(term, 0) - 1
            if remaining > 0:
                self._doc_freq[term] = remaining
            else:
                self._doc_freq.pop(term, None)

    # ------------------------------------------------------------------ 只读视图

    @property
    def name(self) -> str:
        """这份索引的名字（报告里回显它）."""
        return self._name

    @property
    def params(self) -> BM25Params:
        """这一路用的 BM25 参数（"为什么是这个数"的出处）."""
        return self._params

    @property
    def count(self) -> int:
        """索引里的文档篇数."""
        return len(self._documents)

    @property
    def vocabulary_size(self) -> int:
        """词表大小（出现过多少个不同的词元）."""
        return len(self._doc_freq)

    @property
    def avgdl(self) -> float:
        """文档平均词元数（``dl`` 的归一化基准）.

        空索引返回 ``0.0`` 而不是报错（空索引是合法状态，见 ``search``）；
        ``dl`` 为 0 的文档（正文为空串）也算进平均——"没有正文"是它自己的事实，
        把它排除在平均之外会让 ``avgdl`` 变成"非空文档的平均"，而那个数字
        在报告里解释不通（读者会以为它就是全体的平均）。
        """
        if not self._documents:
            return 0.0
        return self._total_length / len(self._documents)

    def vocabulary(self) -> dict[str, int]:
        """词表快照：``词元 → df``（字典序，确定性）——解释"为什么没命中"时要用."""
        return {term: self._doc_freq[term] for term in sorted(self._doc_freq)}

    def ids(self) -> list[str]:
        """全部文档 id（**升序**，与 ``VectorBackend.ids()`` 同一条纪律）."""
        return sorted(self._documents)

    def document(self, record_id: str) -> LexicalDocument | None:
        """取一篇文档；不存在返回 ``None``（不抛异常：查不到是正常结果）."""
        return self._documents.get(str(record_id))

    def describe(self) -> dict[str, Any]:
        """这份索引的现状（``/retrieval/lexical/status`` 直接返回它）."""
        return {
            "name": self._name,
            "channel": CHANNEL_BM25,
            "count": self.count,
            "vocabulary_size": self.vocabulary_size,
            "avgdl": round(self.avgdl, 6),
            "total_terms": self._total_length,
            "params": self._params.to_dict(),
            "k1": self._params.k1,
            "b": self._params.b,
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        return (
            f"关键词索引 {self._name!r}：{self.count} 篇 / 词表 {self.vocabulary_size} / "
            f"avgdl {self.avgdl:.4f} | {self._params.summary_line()}"
        )

    def __len__(self) -> int:
        """文档篇数（``len(index)`` 比 ``len(index.ids())`` 直白）."""
        return len(self._documents)

    # ------------------------------------------------------------------ 检索

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        where: dict[str, Any] | None = None,
    ) -> LexicalSearchResult:
        """给一句话，取回 BM25 分数最高的 ``top_k`` 篇（可带元数据过滤）.

        七步，顺序与 ``Retriever.retrieve`` 对齐（两条路的"问法"必须能逐条对照）：

        ```text
        1. 校验        查询非空 + top_k 范围 + where 语法（复用 vectorstore.compile_filter）
        2. 分词        查询词元（保留重复）与去重后的"打分用词元"
        3. 对账词表    分出 matched_terms / missing_terms —— 这是"为什么一条都没召回"的证据
        4. 空索引早退  0 篇 → 空结果 + notes（**合法状态，不抛异常**，与向量路同一口径）
        5. 过滤        与向量路**同一份**谓词语义（compile_filter），candidates = 过滤后篇数
        6. 打分排序    BM25；只留 score > 0；排序键 (-score, record_id)
        7. 截断与注记  截断到 top_k，并把"0 分被排除了几篇"写进 notes
        ```

        ``where`` 的校验与执行**完全复用** ``vectorstore.filters.compile_filter``：
        两路各写一份过滤实现的话，"同一个 where 在向量路筛掉 3 条、在关键词路筛掉 2 条"
        这种分家会永远存在而且不报错（``hybrid`` 的"过滤在两路都落刀"要的正是同一份语义）。
        """
        _require_query_text(query)
        _check_top_k(top_k)
        validate_where(where)

        raw_terms = tokenize(query)
        unique_terms = sorted(set(raw_terms))
        known = set(self._doc_freq)
        matched = tuple(term for term in unique_terms if term in known)
        missing = tuple(term for term in unique_terms if term not in known)
        filter_applied = bool(where)
        notes: list[str] = []

        # 第 4 步：空索引早退。**注意它排在"过滤之后"的语义里**：库是空的时候，
        # 过滤条件根本没有作用对象，因此 candidates 给 0 而不是"过滤掉了多少"。
        if not self._documents:
            notes.append(
                "关键词索引是空的（0 篇文档）：这是合法状态，直接返回空结果而不报错"
                "（与向量路 count=0 的 no_data 同一口径）"
            )
            return LexicalSearchResult(
                candidates=0,
                filter_applied=filter_applied,
                query_terms=tuple(raw_terms),
                matched_terms=matched,
                missing_terms=missing,
                notes=tuple(notes),
            )

        if self._total_length == 0:
            # 有文档但一个词元都没有：索引被"空的正文"毒化了（avgdl = 0 会让
            # BM25 的分母退化）。这是索引侧的事（去修入库口径或重建），不是调用方的参数问题。
            raise LexicalError(
                f"关键词索引里有 {self.count} 篇文档，但它们的词元总数为 0："
                "索引无法打分（BM25 的分母里有 avgdl）。这通常意味着入库的正文"
                "全是没有可索引内容的串（纯标点 / 纯空白）。出路：修入库口径"
                "（正文至少要有可索引的字）或重建关键词索引——"
                "本方法与向量路一致，**不把这种情况当成'没有命中'**。"
            )

        # 第 5 步：过滤。与向量路同一份 compile_filter（见 docstring）。
        predicate = compile_filter(where)
        candidates = [
            document
            for document in self._documents.values()
            if predicate(document.metadata)
        ]
        if not candidates:
            notes.append(
                "where 把关键词路的候选筛成了 0 条：请检查字段名与取值，或放宽过滤条件"
                "（这一路的过滤与向量路是同一份语义，因此两路会同时被筛没）"
            )
            return LexicalSearchResult(
                candidates=0,
                filter_applied=filter_applied,
                query_terms=tuple(raw_terms),
                matched_terms=matched,
                missing_terms=missing,
                notes=tuple(notes),
            )

        if not matched:
            notes.append(
                f"关键词路一条都没召回：查询词元与词表交集为空（纯语义改写常见）——"
                f"missing_terms={list(missing)}（共 {len(unique_terms)} 个词元，全不在词表里），"
                f"词表里有 {self.vocabulary_size} 个词元。"
                "这不是索引坏了，而是这一路的固有边界：它只认字面。"
            )
            return LexicalSearchResult(
                candidates=len(candidates),
                filter_applied=filter_applied,
                query_terms=tuple(raw_terms),
                matched_terms=matched,
                missing_terms=missing,
                notes=tuple(notes),
            )

        # 第 6 步：打分。idf 与 avgdl 都是**全库统计**（不受 where 影响）：
        # 它们是语料的性质，不是"这次过滤出来的那几条"的性质。
        # 用过滤后的子集重算 idf 会让同一个词在两次不同过滤下有不同的权重，
        # 于是"换个过滤条件"会静默改变每条的分数——而分数是要跨查询比较的。
        total_docs = self.count
        avgdl = self.avgdl
        k1 = self._params.k1
        b = self._params.b
        scored: list[tuple[str, float]] = []
        zero_scored = 0
        for document in candidates:
            counts = self._term_freq[document.record_id]
            length = self._lengths[document.record_id]
            score = 0.0
            for term in matched:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                idf = _idf(total_docs, self._doc_freq[term])
                score += idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * length / avgdl))
            if score > 0.0:
                scored.append((document.record_id, score))
            else:
                zero_scored += 1

        # 第 6 步（续）：排序键 (-score, record_id)——与向量路同一条纪律：
        # 分数相同时必须有一个**不依赖插入顺序**的第二键，否则同一份数据
        # 两次运行会给出不同的名次（同分的来源在关键词路上很常见：
        # 两篇同样长度、同样词频的文档，分数逐位相同）。
        scored.sort(key=lambda item: (-item[1], item[0]))

        if zero_scored:
            notes.append(
                f"有 {zero_scored} 篇候选文档与查询没有任何共同词元（BM25 给 0 分），"
                "已排除在名次之外：0 分不是'相关性很低'，而是'这次查询完全没提到它'。"
                "把 0 分文档按零分排进名次，会让只看名次的融合策略"
                "（RRF）把'毫无关系'当成'第 N 名证据'。"
            )

        # 第 7 步：截断与装配。
        if len(scored) > top_k:
            notes.append(
                f"关键词路有 {len(scored)} 条真命中，截断到 top_k={top_k} "
                f"丢掉 {len(scored) - top_k} 条（余下的都由查询词元命中）"
            )
        hits = tuple(
            LexicalHit(
                record_id=record_id,
                score=score,
                rank=rank,
                text=self._documents[record_id].text,
                metadata=dict(self._documents[record_id].metadata),
            )
            for rank, (record_id, score) in enumerate(scored[:top_k])
        )
        if missing:
            notes.append(
                f"查询里有 {len(missing)} 个词元不在词表里（{list(missing)}）："
                "它们不会贡献任何分数——这是'未登录词'这条真实边界"
                "（2-gram 分词法不认识未收录的字串）。"
            )
        return LexicalSearchResult(
            hits=hits,
            candidates=len(candidates),
            filter_applied=filter_applied,
            query_terms=tuple(raw_terms),
            matched_terms=matched,
            missing_terms=missing,
            notes=tuple(notes),
        )

    def search_many(
        self,
        queries: Sequence[str],
        top_k: int = DEFAULT_TOP_K,
        where: dict[str, Any] | None = None,
    ) -> list[LexicalSearchResult]:
        """批量检索（逐条调用 ``search``，顺序与入参一致）.

        与 ``Retriever.retrieve_many`` 同一条理由：**刻意不做批处理**——
        这里的"批量"是报告与评估的便利入口，不是性能特性；逐条调用让每次
        检索都有一份独立的证据（词元对账结果不同、候选数不同）。
        """
        return [self.search(item, top_k=top_k, where=where) for item in queries]


# --------------------------------------------------------------------------- #
# 入参校验（三个小函数，错误消息各自指出出路）
# --------------------------------------------------------------------------- #


def _require_query_text(query: str) -> None:
    """查询必须是非空字符串（与 ``RetrievalQuery.text`` 同一条纪律）.

    空白查询在关键词路上不是"返回空结果"而是**没有定义**：它会被分成 0 个词元，
    于是"一条都没召回"的原因永远说不清（是词表里没有，还是这次根本没有查询？）。
    与向量路的空查询一样当场报错，两路的入口规则保持一致。
    """
    if not isinstance(query, str):
        raise QueryError(
            f"关键词检索的查询必须是字符串，收到 {type(query).__name__}。"
            "要用一批 id 取记录请用 from_backend 的那条路。"
        )
    if not query.strip():
        raise QueryError(
            "空查询在关键词检索里没有定义：请给一段非空文本。"
            "空白串会被分成 0 个词元，于是'一条都没召回'与'这次根本没查询'"
            "在报告里长得一样——而它们的处置完全不同。"
        )


def _check_top_k(top_k: int) -> None:
    """``top_k`` 必须是 ``[1, MAX_TOP_K]`` 里的整数（上限与向量路同一个常量）.

    上限**复用 ``vectorstore.MAX_TOP_K``**：两条路的深度上限一旦分成两个数字，
    混合检索取候选时就会出现"关键词路取 1000、向量路只能取 800"这种
    自相矛盾的配置，而它的表现是"两路的候选数不一样"——一个必须靠读代码
    才能解释的差值。
    """
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise QueryError(
            f"top_k 必须是 >= 1 的整数，收到 {top_k!r}。"
            "'要 0 条'不是一个有意义的检索请求；要探测'索引里有没有文档'"
            "请读 len(index) 或 index.count。"
        )
    if top_k > MAX_TOP_K:
        raise QueryError(
            f"top_k={top_k} 超过上限 {MAX_TOP_K}（与 vectorstore.MAX_TOP_K 同一个上限）："
            "一次把整库搬进内存没有意义——要取全部请遍历 index.ids()。"
        )


__all__ = [
    "CJK_ONLY",
    "DEFAULT_B",
    "DEFAULT_K1",
    "TOKEN_PATTERN",
    "BM25Params",
    "LexicalDocument",
    "LexicalHit",
    "LexicalIndex",
    "LexicalSearchResult",
    "tokenize",
]
