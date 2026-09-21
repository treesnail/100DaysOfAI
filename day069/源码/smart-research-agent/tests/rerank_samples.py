"""day068 重排测试用的确定性小语料（十条记录 + 可手算的四个特征 + 五条探针）.

单独一个模块而不是把样本复制进三个测试文件，理由与 ``retrieval_samples`` /
``hybrid_samples`` 完全相同：**同一份语料要被 core / pipeline / lift 三个文件
同时用到**，复制多份会让"样本里改一个字"变成要改许多处，而漏改的那一处只表现为
"某个测试不再覆盖那条语义"。

## 这批样本刻意埋了五个差异点（每一个都对应一条要钉住的纪律）

```text
1. 第一阶段把"最像的"排第 0、把"最相关的"排第 7
   → 向量路给 r-t-01 满分（它一个查询词都不含），而真正含 "window rerank"
     的 r-t-08 在第一阶段排第 7（0 分并列里 id 靠后）
   → 重排把 r-t-08 顶到第 0：这就是"两段式"的全部意义

2. 重排分**精确并列**：r-t-02 / r-t-05 / r-t-07 都是 0.375
   → 排序第二键（stage1_rank）有东西可兜：输入序 1 / 4 / 6，输出也是这个序
   → 若第二键被换成 record_id，输出会变成 r-t-02 / r-t-05 / r-t-07（恰好相同），
     因此**再配一条输入序与 id 序相反的用例**（见 TestRerankHitsOrdering）

3. 长度惩罚的**两个分支**都出现：r-t-09 长 300 字 → 240/300 = 0.8；
   其余九条都短于 240 字 → 1.0
   → "短于标准长度不罚、长于它按比例打折"是一个可核对的数字，不是一句描述

4. 词汇能力的**边界**：r-t-02 写的是 "reranking"（不是 "rerank"）
   → 覆盖率只算 0.5，而 proximity 是 0.0（缺一个词元就判"无法覆盖"）
   → 教学替身"认不出同义改写/词形变化"这件事因此可核对，而不是一句免责声明

5. 三路名次各不相同（向量 / BM25 / 融合），且**融合次序可手算**
   → RRF 的算术写在 ``HAND_FUSED_RRF`` 的注释里：两路名次都是手算出来的
   → 于是"重排在多样性之前落刀"能用一个**只翻一个开关**的对照钉住
```

## 期望值为什么能手算

```text
1) 查询取纯拉丁词 "window rerank"
   → tokenize 只切出两个词元（"window" / "rerank"），没有 CJK 的单字 + 2-gram，
     因此覆盖率、跨度、最小窗口全都是小整数算术
2) 向量只有 0 / 0.6 / 0.8 / 1.0 四种分量，且都是单位向量（cosine 下模长全为 1）
   → 查询向量 (1,0,…,0) 与它们的点积只有 1.0 / 0.8 / 0.6 / 0.0 四种取值
   → 第一阶段的次序（0 分并列按 id 升序）是一个**精确的事实**
3) 四个特征的权重是二进制精确可表示的 (1/2, 1/4, 1/8, 1/8)
   → 每个分数都是分母为 24 或 12 的分数，可以逐位写下来（见 HAND_SCORES）
4) 探针的金标准只有 1~2 条 id
   → recall@k / RR / nDCG@k 三条指标的分母都很小，改动一位就能对出来
```

全部离线、确定性、零网络、不写仓库外文件。
"""

from __future__ import annotations

from typing import Any

from smart_research_agent.retrieval.hybrid import HybridRetriever
from smart_research_agent.retrieval.lexical import LexicalIndex
from smart_research_agent.retrieval.rerank import (
    BaseReranker,
    CrossEncoderReranker,
    LiftProbe,
    RerankFeatures,
)
from smart_research_agent.retrieval.retriever import Retriever, build_retriever
from smart_research_agent.retrieval.types import (
    RetrievalHit,
    RetrievalQuery,
    RetrievalResult,
)
from smart_research_agent.vectorstore.flat import FlatVectorStore
from smart_research_agent.vectorstore.types import VectorRecord, make_record
from tests.retrieval_samples import TableEmbedding

# --------------------------------------------------------------------------- #
# 语料的骨架
# --------------------------------------------------------------------------- #

#: 样本向量维度。取 8（与 ``retrieval_samples`` / ``hybrid_samples`` 同一个量级）：
#: 分量只有 0 / 0.6 / 0.8 / 1.0，因此 cosine 分数是 1.0 / 0.8 / 0.6 / 0.0 四个数。
VECTOR_DIMENSION = 8

#: 本次唯一的查询：一个**只有两个拉丁词元**的查询。
#:
#: 取拉丁词不是为了"像英文"，而是为了 ``tokenize`` 只给两个词元——CJK 会切出
#: 单字 + 2-gram（一份 4 字中文查询是 7 个词元），跨度与覆盖率就不再是手算级别。
RERANK_QUERY = "window rerank"

#: 查询向量：恰好指向第一条记录的方向（那条记录一个查询词都不含）。
QUERY_VECTOR: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

#: 三条记录共用一个文档 id —— ``max_per_doc=1`` 时它们互相挤（多样性裁剪的素材）.
DOC_WINDOW = "doc-window"

#: 那条**长于 IDEAL_LEN** 的记录（长度惩罚的"打折"一侧）。长度是 300 字符：
#: ``240 / 300 = 0.8``，因此它的 ``length_penalty`` 是 0.8 而不是 1.0。
LONG_TEXT = (
    "the context packer walks every hit and appends its body until the character "
    "budget runs out, then it records how many hits were skipped so that a report can "
    "tell apart a short answer from a truncated one and a truncated one from an "
    "empty one at last the packer prints the exact number of skipped hits"
)

#: ``LONG_TEXT`` 的字符数（**写死**，好让"改了正文就要改惩罚常量"变成一条断言）.
LONG_TEXT_LENGTH = 300

#: 十条记录：``(id, 向量, 正文, parent_doc_id)`` 摊平成元组。
#:
#: 向量的排序**刻意与语义相反**：讲 "window rerank" 的那几条（r-t-06 / r-t-08）
#: 被放在第 4 / 6 根轴上，而查询指向第 1 根轴——于是"语义上最像的"是 r-t-01
#: （"cosine similarity from a bi encoder"，一个查询词都不含）。
RECORD_SPECS: tuple[tuple[str, tuple[float, ...], str, str], ...] = (
    (
        "r-t-01",
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "cosine similarity from a bi encoder",
        "doc-01",
    ),
    (
        # **词形陷阱**：这里写的是 "reranking"，而查询词元是 "rerank"。
        # exact_phrase 也匹配不上（它判的是查询原文 "window rerank"）。
        "r-t-02",
        (0.8, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "reranking needs a window of candidates",
        "doc-02",
    ),
    (
        "r-t-03",
        (0.6, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "recall depth is three times top k",
        "doc-03",
    ),
    (
        "r-t-04",
        (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "chunking strategies for long documents",
        "doc-04",
    ),
    (
        "r-t-05",
        (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "the window slides over every token",
        DOC_WINDOW,
    ),
    (
        # 覆盖齐全 + 紧挨着：span = 2 → proximity = 1/3
        "r-t-06",
        (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        "rerank window discipline keeps cost bounded",
        DOC_WINDOW,
    ),
    (
        "r-t-07",
        (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        "the rerank model is a teaching stub",
        "doc-07",
    ),
    (
        # **最强的那个分数**：覆盖率 1.0 + 整句包含 1.0 + 邻近度 1/3
        "r-t-08",
        (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        "an exact phrase like window rerank matches fully",
        DOC_WINDOW,
    ),
    (
        # 唯一一条长于 IDEAL_LEN 的记录：length_penalty = 240/300 = 0.8
        "r-t-09",
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        LONG_TEXT,
        "doc-09",
    ),
    (
        "r-t-10",
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        "answer generation prompt template",
        "doc-10",
    ),
)

#: 十条记录的 id（按插入顺序 = 第一阶段的输入序，也是 0 分并列时的 id 序）.
RECORD_IDS: tuple[str, ...] = tuple(spec[0] for spec in RECORD_SPECS)

#: id → 正文.
RECORD_TEXTS: dict[str, str] = {spec[0]: spec[2] for spec in RECORD_SPECS}

#: id → 第一阶段的分数（= 与 ``QUERY_VECTOR`` 的 cosine）。
#:
#: ```text
#: (1,0,…) 与 (1,0,…)   点积 1.0 / 模长 1×1 → 1.0
#: (1,0,…) 与 (0.8,0.6) 点积 0.8 / 模长 1×1 → 0.8
#: (1,0,…) 与 (0.6,0.8) 点积 0.6 / 模长 1×1 → 0.6
#: 其余七条与查询正交       → 0.0（并列，按 id 升序）
#: ```
STAGE1_SCORES: dict[str, float] = {
    "r-t-01": 1.0,
    "r-t-02": 0.8,
    "r-t-03": 0.6,
    "r-t-04": 0.0,
    "r-t-05": 0.0,
    "r-t-06": 0.0,
    "r-t-07": 0.0,
    "r-t-08": 0.0,
    "r-t-09": 0.0,
    "r-t-10": 0.0,
}

#: 第一阶段的完整次序（分数降序 + 同分按 id 升序）——**十条都进来**.
#:
#: 它就是"重排之前"的名单：``measure_lift`` 的 ``before_ids`` 必须逐项等于它。
HAND_STAGE1_ORDER: tuple[str, ...] = RECORD_IDS

#: 记录 id → 它第一阶段排在**第几名**（0 起）.
HAND_STAGE1_RANK: dict[str, int] = {rid: index for index, rid in enumerate(RECORD_IDS)}


# --------------------------------------------------------------------------- #
# 四个特征：手算常量（算术过程写在注释里）
# --------------------------------------------------------------------------- #

#: 查询词元（去重后排序）：只有两个。
#: ``tokenize("window rerank")`` → ``["window", "rerank"]``（拉丁段整段成词元）。
QUERY_TERMS: tuple[str, ...] = ("rerank", "window")

#: 相邻时的邻近度：跨度恰好等于词元数 2 → ``1 / (1 + 2) = 1/3``.
PROXIMITY_ADJACENT = 1.0 / 3.0

#: 每条记录的四个特征：``(term_coverage, exact_phrase, proximity, length_penalty)``.
#:
#: ```text
#: 记录     覆盖率                整句    邻近度           长度惩罚
#: r-t-01   0/2 = 0.0            0.0     0.0（缺 window）  min(1, 240/35) = 1.0
#: r-t-02   1/2 = 0.5（window）   0.0     0.0（缺 rerank）  1.0
#: r-t-03   0/2 = 0.0            0.0     0.0              1.0
#: r-t-04   0/2 = 0.0            0.0     0.0              1.0
#: r-t-05   1/2 = 0.5（window）   0.0     0.0（缺 rerank）  1.0
#: r-t-06   2/2 = 1.0            0.0     1/3（span=2）     1.0
#: r-t-07   1/2 = 0.5（rerank）   0.0     0.0（缺 window）  1.0
#: r-t-08   2/2 = 1.0            1.0     1/3（span=2）     1.0
#: r-t-09   0/2 = 0.0            0.0     0.0              240/300 = 0.8
#: r-t-10   0/2 = 0.0            0.0     0.0              1.0
#: ```
#:
#: 两个词形细节值得单独看：``r-t-02`` 的 "reranking" **不等于** "rerank"（覆盖率
#: 只算 0.5），``r-t-08`` 的 "window rerank" 逐字出现（整句包含 = 1.0）。
HAND_FEATURES: dict[str, tuple[float, float, float, float]] = {
    "r-t-01": (0.0, 0.0, 0.0, 1.0),
    "r-t-02": (0.5, 0.0, 0.0, 1.0),
    "r-t-03": (0.0, 0.0, 0.0, 1.0),
    "r-t-04": (0.0, 0.0, 0.0, 1.0),
    "r-t-05": (0.5, 0.0, 0.0, 1.0),
    "r-t-06": (1.0, 0.0, PROXIMITY_ADJACENT, 1.0),
    "r-t-07": (0.5, 0.0, 0.0, 1.0),
    "r-t-08": (1.0, 1.0, PROXIMITY_ADJACENT, 1.0),
    "r-t-09": (0.0, 0.0, 0.0, 0.8),
    "r-t-10": (0.0, 0.0, 0.0, 1.0),
}

#: 每条记录的重排分 = ``Σ 权重 × 特征``（权重 1/2、1/4、1/8、1/8）.
#:
#: ```text
#: r-t-01  0 + 0 + 0            + 1/8 = 3/24 = 0.125
#: r-t-02  1/4 + 0 + 0          + 1/8 = 9/24 = 0.375
#: r-t-03  0 + 0 + 0            + 1/8 = 3/24 = 0.125
#: r-t-04  0 + 0 + 0            + 1/8 = 3/24 = 0.125
#: r-t-05  1/4 + 0 + 0          + 1/8 = 9/24 = 0.375
#: r-t-06  1/2 + 0 + 1/24       + 1/8 = 16/24 = 2/3   = 0.666667
#: r-t-07  1/4 + 0 + 0          + 1/8 = 9/24 = 0.375
#: r-t-08  1/2 + 1/4 + 1/24     + 1/8 = 22/24 = 11/12 = 0.916667
#: r-t-09  0 + 0 + 0            + 1/10 = 0.1   （0.8/8 = 0.1）
#: r-t-10  0 + 0 + 0            + 1/8 = 3/24 = 0.125
#: ```
#:
#: 三个 0.375（r-t-02 / r-t-05 / r-t-07）是**精确并列**，不是"差不多相等"：
#: 它们的算术过程逐项相同。
HAND_SCORES: dict[str, float] = {
    "r-t-01": 0.125,
    "r-t-02": 0.375,
    "r-t-03": 0.125,
    "r-t-04": 0.125,
    "r-t-05": 0.375,
    "r-t-06": 2.0 / 3.0,
    "r-t-07": 0.375,
    "r-t-08": 11.0 / 12.0,
    "r-t-09": 0.1,
    "r-t-10": 0.125,
}

#: ``mode="replace"``、``top_n=10``、不设阈值时的重排名单（**本课的主案例**）.
#:
#: ```text
#: 排序键 (-score, stage1_rank, record_id)：
#:   11/12      r-t-08（stage1 #7）
#:   2/3        r-t-06（stage1 #5）
#:   0.375      r-t-02（#1）→ r-t-05（#4）→ r-t-07（#6）   ← 三键并列，按 stage1_rank
#:   0.125      r-t-01（#0）→ r-t-03（#2）→ r-t-04（#3）→ r-t-10（#9）
#:   0.1        r-t-09（#8）
#: ```
HAND_RERANK_ORDER: tuple[str, ...] = (
    "r-t-08",
    "r-t-06",
    "r-t-02",
    "r-t-05",
    "r-t-07",
    "r-t-01",
    "r-t-03",
    "r-t-04",
    "r-t-10",
    "r-t-09",
)

#: ``HAND_RERANK_ORDER`` 逐项对应的重排分（保留 6 位）.
HAND_RERANK_SCORES: tuple[float, ...] = (
    0.916667,
    0.666667,
    0.375,
    0.375,
    0.375,
    0.125,
    0.125,
    0.125,
    0.125,
    0.1,
)

#: ``HAND_RERANK_ORDER`` 逐项对应的 **stage1_rank**（它原来在第几）.
HAND_RERANK_STAGE1_RANKS: tuple[int, ...] = (7, 5, 1, 4, 6, 0, 2, 3, 9, 8)

#: ``HAND_RERANK_ORDER`` 逐项对应的**名次变化** ``stage1_rank - rank``.
#:
#: ```text
#: 7-0=+7  5-1=+4  1-2=-1  4-3=+1  6-4=+2  0-5=-5  2-6=-4  3-7=-4  9-8=+1  8-9=-1
#: ```
#: 十条**全都**动了 — 这正是"重排改变了次序"的完整证据（`moved_ids` 长度 10）。
HAND_RERANK_MOVED: tuple[int, ...] = (7, 4, -1, 1, 2, -5, -4, -4, 1, -1)

#: ``top_n=3`` 时的名单：**窗口内换过序，窗口外原样接在后面**.
#:
#: ```text
#: 窗口 = 第一阶段前 3 条 = r-t-01(#0, 0.125) / r-t-02(#1, 0.375) / r-t-03(#2, 0.125)
#: 重排后窗口内：r-t-02(0.375) → r-t-01(0.125) → r-t-03(0.125)（后两条并列，按原名次）
#: 尾部 7 条（r-t-04 … r-t-10）**一条都没打分**，原顺序接在后面
#: ```
HAND_WINDOW3_ORDER: tuple[str, ...] = (
    "r-t-02",
    "r-t-01",
    "r-t-03",
    "r-t-04",
    "r-t-05",
    "r-t-06",
    "r-t-07",
    "r-t-08",
    "r-t-09",
    "r-t-10",
)

#: ``mode="blend"``、``weight=0.5``、``top_n=10`` 时的名单与加权分.
#:
#: 归一化的基准是**窗口内全部十条被打分的条**（在 min_score 落刀**之前**）：
#:
#: ```text
#: 重排分 min/max = 0.1（r-t-09）/ 11/12（r-t-08），跨度 49/60
#: 第一阶段 min/max = 0.0 / 1.0，跨度 1.0（于是归一化后就是它自己）
#:
#: r-t-02  0.5·((3/8−1/10)÷49/60) + 0.5·0.8   = 0.5·33/98 + 0.4   = 0.568367
#: r-t-01  0.5·((1/8−1/10)÷49/60) + 0.5·1.0   = 0.5·3/98  + 0.5   = 0.515306
#: r-t-08  0.5·1.0               + 0.5·0.0                  = 0.5
#: r-t-06  0.5·((2/3−1/10)÷49/60) + 0.5·0.0   = 0.5·34/49        = 0.346939
#: r-t-03  0.5·3/98              + 0.5·0.6                  = 0.315306
#: r-t-05  0.5·33/98 + 0.0                                   = 0.168367
#: r-t-07  0.5·33/98 + 0.0                                   = 0.168367
#: r-t-04  0.5·3/98  + 0.0                                   = 0.015306
#: r-t-10  0.5·3/98  + 0.0                                   = 0.015306
#: r-t-09  0.0                       + 0.0                   = 0.0
#: ```
#:
#: 它与 ``replace`` 的名单**不同**（replace 的第一名是 r-t-08，blend 是 r-t-02），
#: 因为 blend 里第一阶段的分数也占一半——这正是这个模式存在的理由。
HAND_BLEND_ORDER: tuple[str, ...] = (
    "r-t-02",
    "r-t-01",
    "r-t-08",
    "r-t-06",
    "r-t-03",
    "r-t-05",
    "r-t-07",
    "r-t-04",
    "r-t-10",
    "r-t-09",
)

#: ``HAND_BLEND_ORDER`` 逐项的加权分（保留 6 位）.
HAND_BLEND_VALUES: tuple[float, ...] = (
    0.568367,
    0.515306,
    0.5,
    0.346939,
    0.315306,
    0.168367,
    0.168367,
    0.015306,
    0.015306,
    0.0,
)

#: ``mode="blend"`` + ``min_score=0.2`` 时的名单（窗口内只留下分数 ≥ 0.2 的五条）.
#:
#: **它同时钉住"先归一化、后落刀"**：``r-t-06`` 的加权分仍然是 0.346939
#: （= 用**全部十条**的 min/max 归一化出来的那个数）。若实现先把 < 0.2 的切掉再
#: 归一化，基准会变成 min=0.375、max=0.916667，``r-t-06`` 会变成
#: ``0.5·(0.291667÷0.541667) = 0.269231``——一个不同的数。
HAND_BLEND_MIN_SCORE_ORDER: tuple[str, ...] = (
    "r-t-02",
    "r-t-08",
    "r-t-06",
    "r-t-05",
    "r-t-07",
)

#: ``min_score=0.5``（``replace``）时切掉的条数：0.916667 / 2/3 两条留下，其余八条被切.
HAND_MIN_SCORE_050_DROPPED = 8

#: 阈值高到把窗口切空时留下的那句话（``empty_reason``）的**关键片段**.
#: 它必须同时说清"是谁切的"与"出路是什么"。
EMPTY_REASON_FRAGMENT = "请降低重排阈值或不设阈值"

#: 窗口纪律注记的**关键片段**：无论 top_n 是多少、有没有尾部，它都必须出现。
WINDOW_NOTE_FRAGMENT = "一条都没打分"


# --------------------------------------------------------------------------- #
# 第二阶段（混合检索）的手算次序
# --------------------------------------------------------------------------- #

#: BM25 这一路的次序（**也是手算出来的**，公式与 ``test_hybrid_bm25`` 的
#: 独立实现同源：``idf·tf·(k1+1) / (tf + k1·(1 − b + b·dl/avgdl))``）。
#:
#: ```text
#: 只有五条含查询词元：r-t-06(window+rerank) / r-t-08(window+rerank) /
#: r-t-07(rerank) / r-t-02(window) / r-t-05(window)
#: 两条同时含两个词元的排在前面，而 r-t-06（43 字）比 r-t-08（48 字）短
#:   → 长度归一化把 r-t-06 抬高：2.577454 > 2.339779
#: r-t-07 只含较稀有的 rerank（df=3）：1.377603
#: r-t-02 与 r-t-05 都只含 window（df=4）且长度相近：1.129883（**并列，按 id**）
#: ```
HAND_LEXICAL_ORDER: tuple[str, ...] = (
    "r-t-06",
    "r-t-08",
    "r-t-07",
    "r-t-02",
    "r-t-05",
)

#: 融合（RRF，k=60）之后的次序。分数是纯算术，逐条写在下面：
#:
#: ```text
#: r-t-02  vector #1 → 1/62 + bm25 #3 → 1/64 = 0.031754  ← 第一名
#: r-t-06  vector #5 → 1/66 + bm25 #0 → 1/61 = 0.031545
#: r-t-08  vector #7 → 1/68 + bm25 #1 → 1/62 = 0.030835
#: r-t-07  vector #6 → 1/67 + bm25 #2 → 1/63 = 0.030798
#: r-t-05  vector #4 → 1/65 + bm25 #4 → 1/65 = 0.030769
#: r-t-01  vector #0 → 1/61                 = 0.016393
#: r-t-03  vector #2 → 1/63                 = 0.015873
#: r-t-04  vector #3 → 1/64                 = 0.015625
#: r-t-09  vector #8 → 1/69                 = 0.014493
#: r-t-10  vector #9 → 1/70                 = 0.014286
#: ```
#:
#: 前五名的差距都在 1e-4 量级（融合的两个名次都很靠前），因此"换个 alpha 就变"
#: 这件事在这里是可核对的；而**只有 r-t-06 与 r-t-08 属于同一个文档**
#: （``DOC_WINDOW``），这给了多样性裁剪一个只涉及两条的干净场景。
HAND_FUSED_ORDER: tuple[str, ...] = (
    "r-t-02",
    "r-t-06",
    "r-t-08",
    "r-t-07",
    "r-t-05",
    "r-t-01",
    "r-t-03",
    "r-t-04",
    "r-t-09",
    "r-t-10",
)

#: 融合分（逐条）——与上一条注释里的算术一一对应。
HAND_FUSED_RRF: dict[str, float] = {
    "r-t-02": 1.0 / 62.0 + 1.0 / 64.0,
    "r-t-06": 1.0 / 66.0 + 1.0 / 61.0,
    "r-t-08": 1.0 / 68.0 + 1.0 / 62.0,
    "r-t-07": 1.0 / 67.0 + 1.0 / 63.0,
    "r-t-05": 1.0 / 65.0 + 1.0 / 65.0,
    "r-t-01": 1.0 / 61.0,
    "r-t-03": 1.0 / 63.0,
    "r-t-04": 1.0 / 64.0,
    "r-t-09": 1.0 / 69.0,
    "r-t-10": 1.0 / 70.0,
}

#: 融合之后**再重排**（``replace`` / ``top_n=10``）的次序。
#:
#: 输入是 ``HAND_FUSED_ORDER``（因此 ``stage1_rank`` 是融合名次 0…9），
#: 排序键仍是 ``(-重排分, stage1_rank, record_id)``：
#:
#: ```text
#: 11/12  r-t-08（融合 #2）      2/3  r-t-06（融合 #1）
#: 0.375  r-t-02（#0）→ r-t-07（#3）→ r-t-05（#4）
#: 0.125  r-t-01（#5）→ r-t-03（#6）→ r-t-04（#7）→ r-t-10（#9）
#: 0.1    r-t-09（#8）
#: ```
HAND_HYBRID_RERANK_ORDER: tuple[str, ...] = (
    "r-t-08",
    "r-t-06",
    "r-t-02",
    "r-t-07",
    "r-t-05",
    "r-t-01",
    "r-t-03",
    "r-t-04",
    "r-t-10",
    "r-t-09",
)

#: ``max_per_doc=1``、**关着重排**时留下的那个 ``DOC_WINDOW`` 成员：融合第一.
#: 同一文档的三条（r-t-05 / r-t-06 / r-t-08）里，融合名次最好的是 r-t-06（#1）。
HAND_DIVERSITY_KEPT_WITHOUT_RERANK = "r-t-06"

#: ``max_per_doc=1``、**开着重排**时留下的那个成员：重排把它顶到了融合之前。
#: 重排之后 r-t-08 在整份名单里排第 0，因此多样性保它、切掉 r-t-06 与 r-t-05。
HAND_DIVERSITY_KEPT_WITH_RERANK = "r-t-08"

#: ``max_per_doc=1`` 下被多样性切掉的条数（两种开关状态都是 2 条）.
HAND_DIVERSITY_DROPPED = 2


# --------------------------------------------------------------------------- #
# 探针与三条指标的手算值
# --------------------------------------------------------------------------- #

#: 五条探针共用的**查询文本**——就是 ``RERANK_QUERY``。
#:
#: 这不是偷懒：``measure_lift`` 会把 ``probe.query`` **既送去检索、又送去重排**，
#: 而三条指标只依赖"名单 + 金标准"。让五条探针共用一句话、只换金标准，
#: 是为了让 before/after 的差**只**来自"有没有重排"这一个变量——
#: 换成五句不同的话，差里就会混进"两次不同检索"的差异，
#: 而那正是 ``measure_lift`` 的 docstring 明说要避免的事。
PROBE_QUERY = RERANK_QUERY

#: 五条探针：``(名字, 金标准 id)``。金标准刻意挑成三种角色：
#:
#: ```text
#: p-a  第一阶段第一名（重排把它压到第 5）  → 重排**变差**的对照
#: p-b  第一阶段第 7 名（重排顶到第 0）     → 重排**提升**的对照（最干净的一条）
#: p-c  第一阶段第 5 名（重排升到第 1）     → 名次上移但没到第一（nDCG 只有 0.63093）
#: p-d  长文档（两个名单里都在 9 名之后）   → 提升与变差都不明显（RR 略降）
#: p-e  两条金标准（一条升、一条降）        → 同时看"命中几条"与"排得多靠前"
#: ```
PROBE_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("p-a", ("r-t-01",)),
    ("p-b", ("r-t-08",)),
    ("p-c", ("r-t-06",)),
    ("p-d", ("r-t-09",)),
    ("p-e", ("r-t-02", "r-t-08")),
)

#: 指标里共用的 k.
LIFT_K = 5

#: 逐条探针的 ``(before 三条指标, after 三条指标)`` —— 顺序都是
#: ``(recall@k, reciprocal_rank, nDCG@k)``。
#:
#: 算术过程（``before`` 用 ``HAND_STAGE1_ORDER``，``after`` 用 ``HAND_RERANK_ORDER``）：
#:
#: ```text
#: p-a  before: 前 5 条含 r-t-01（位置 0）→ recall 1/1=1.0、RR 1/1=1.0、nDCG 1/1=1.0
#:      after : r-t-01 在第 5 位 → recall 0.0、RR 1/6=0.166667、nDCG 0.0
#: p-b  before: r-t-01 在第 7 位 → recall 0.0（前 5 条没有它）、RR 1/8=0.125、nDCG 0.0
#:      after : 位置 0 → recall 1.0、RR 1.0、nDCG 1/log2(2)/1.0 = 1.0
#: p-c  before: 第 5 位 → recall 0.0、RR 1/6=0.166667、nDCG 0.0
#:      after : 第 1 位 → recall 1.0、RR 1/2=0.5、nDCG (1/log2(3))/1 = 0.630930
#: p-d  before: 第 8 位 → recall 0.0、RR 1/9=0.111111、nDCG 0.0
#:      after : 第 9 位 → recall 0.0、RR 1/10=0.1、nDCG 0.0
#: p-e  before: 前 5 条含 r-t-02（位置 1）→ recall 1/2=0.5、RR 1/2=0.5、
#:             nDCG = (1/log2(3)) / (1/log2(2)+1/log2(3)) = 0.386853
#:      after : 前 5 条含 r-t-08（位置 0）与 r-t-02（位置 2）→ recall 2/2=1.0、
#:             RR 1/1=1.0、nDCG = (1/log2(2)+1/log2(4)) / (1+1/log2(3)) = 0.919721
#: ```
HAND_PROBE_METRICS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "p-a": ((1.0, 1.0, 1.0), (0.0, 0.166667, 0.0)),
    "p-b": ((0.0, 0.125, 0.0), (1.0, 1.0, 1.0)),
    "p-c": ((0.0, 0.166667, 0.0), (1.0, 0.5, 0.63093)),
    "p-d": ((0.0, 0.111111, 0.0), (0.0, 0.1, 0.0)),
    "p-e": ((0.5, 0.5, 0.386853), (1.0, 1.0, 0.919721)),
}

#: 逐条探针的 ``lift`` = after − before（顺序 ``(recall, RR, nDCG)``）.
HAND_PROBE_LIFT: dict[str, tuple[float, float, float]] = {
    "p-a": (-1.0, -0.833333, -1.0),
    "p-b": (1.0, 0.875, 1.0),
    "p-c": (1.0, 0.333333, 0.63093),
    "p-d": (0.0, -0.011111, 0.0),
    "p-e": (0.5, 0.5, 0.532868),
}

#: 报告里的三条均值（``before`` / ``after``）与 ``lift``。
#:
#: ```text
#: before recall = (1.0 + 0 + 0 + 0 + 0.5) / 5                              = 0.3
#: after  recall = (0 + 1.0 + 1.0 + 0 + 1.0) / 5                            = 0.6
#: before RR     = (1.0 + 0.125 + 0.166667 + 0.111111 + 0.5) / 5            = 0.380556
#: after  RR     = (0.166667 + 1.0 + 0.5 + 0.1 + 1.0) / 5                   = 0.553333
#: before nDCG   = (1.0 + 0 + 0 + 0 + 0.386853) / 5                         = 0.277371
#: after  nDCG   = (0 + 1.0 + 0.63093 + 0 + 0.919721) / 5                   = 0.51013
#: lift          = 0.3 / 0.172777 / 0.232759（after − before，逐位相减后 round 6）
#: ```
HAND_LIFT_BEFORE: dict[str, float] = {
    "recall": 0.3,
    "reciprocal_rank": 0.380556,
    "ndcg": 0.277371,
}

HAND_LIFT_AFTER: dict[str, float] = {
    "recall": 0.6,
    "reciprocal_rank": 0.553333,
    "ndcg": 0.51013,
}

HAND_LIFT: dict[str, float] = {
    "recall": 0.3,
    "reciprocal_rank": 0.172777,
    "ndcg": 0.232759,
}

#: 三条指标的**报告内键名**（与 ``rerank._METRIC_NAMES`` 同序）.
METRIC_NAMES: tuple[str, ...] = ("recall", "reciprocal_rank", "ndcg")


# --------------------------------------------------------------------------- #
# 构造器
# --------------------------------------------------------------------------- #


def rerank_embedding(
    table: dict[str, tuple[float, ...]] | None = None,
) -> TableEmbedding:
    """按 ``{RERANK_QUERY: QUERY_VECTOR}`` 查表的确定性编码器（8 维）.

    ``default`` 也用 ``QUERY_VECTOR``：任何别的查询文本都会得到同一条方向，
    因此"这次检索的分数分布"只由查询向量决定，不会被兜底向量搅乱。
    """
    return TableEmbedding(
        table=dict(table or {RERANK_QUERY: QUERY_VECTOR}),
        dimension=VECTOR_DIMENSION,
        default=QUERY_VECTOR,
    )


def sample_metadata() -> dict[str, dict[str, Any]]:
    """id → 元数据（每次调用都新建，避免用例之间互相污染）."""
    return {
        spec[0]: {
            "parent_doc_id": spec[3],
            "source": f"docs/{spec[3]}.md",
            "topic": "rerank",
        }
        for spec in RECORD_SPECS
    }


def sample_records(*, metric: str = "cosine", reverse: bool = False) -> list[VectorRecord]:
    """十条 ``VectorRecord``（``reverse=True`` 时插入顺序相反，用于可复现性用例）."""
    specs = list(RECORD_SPECS)
    if reverse:
        specs.reverse()
    return [
        make_record(
            record_id=spec[0],
            vector=tuple(float(value) for value in spec[1]),
            text=spec[2],
            metadata={
                "parent_doc_id": spec[3],
                "source": f"docs/{spec[3]}.md",
                "topic": "rerank",
            },
            metric=metric,
        )
        for spec in specs
    ]


def rerank_store(*, metric: str = "cosine", reverse: bool = False) -> FlatVectorStore:
    """装好十条样本记录的 flat 库（本模块所有流水线用例的默认起点）."""
    store = FlatVectorStore(metric=metric, dimension=VECTOR_DIMENSION)
    store.upsert(sample_records(metric=metric, reverse=reverse))
    return store


def rerank_lexical(
    store: FlatVectorStore | None = None,
    *,
    name: str = "lexical",
) -> LexicalIndex:
    """从样本库现建一份关键词索引（``store`` 缺省用 ``rerank_store()``）."""
    return LexicalIndex.from_backend(store if store is not None else rerank_store(), name=name)


def empty_store(*, metric: str = "cosine") -> FlatVectorStore:
    """一个**空**库（维度固定为 8，与 ``rerank_store`` 同规格）.

    空库是合法状态：两条流水线都为它返回 ``empty_reason="no_data"`` 而不报错，
    而且这次的早退发生在**重排之前**——因此重排器一次都不会被调用
    （它的 ``calls`` 计数器保持 0）。
    """
    return FlatVectorStore(metric=metric, dimension=VECTOR_DIMENSION)


def rerank_vector_retriever(
    store: FlatVectorStore | None = None,
    *,
    embedding: TableEmbedding | None = None,
    **overrides: Any,
) -> Retriever:
    """单路向量检索器（``**overrides`` 逐个覆盖构造参数，例如 ``rerank_enabled=True``）."""
    return build_retriever(
        store if store is not None else rerank_store(),
        embedding if embedding is not None else rerank_embedding(),
        **overrides,
    )


def rerank_hybrid_retriever(
    store: FlatVectorStore | None = None,
    *,
    embedding: TableEmbedding | None = None,
    lexical: LexicalIndex | None = None,
    **overrides: Any,
) -> HybridRetriever:
    """两路齐全的混合检索器.

    **注意**：``HybridRetriever`` 会从向量那一半"只增不减地"继承 ``rerank_enabled``，
    而本工厂刻意为向量那一半交一个**关着重排**的 ``Retriever``——
    否则"混合层的重排开关"与"向量层的重排开关"就分不开了。
    """
    resolved = store if store is not None else rerank_store()
    return HybridRetriever(
        rerank_vector_retriever(resolved, embedding=embedding),
        lexical if lexical is not None else rerank_lexical(resolved),
        **overrides,
    )


def rerank_reranker(**overrides: Any) -> CrossEncoderReranker:
    """教学级替身（``**overrides`` 用于 ``batch_size`` / ``ideal_len`` / ``name``）."""
    return CrossEncoderReranker(**overrides)


def query_of(text: str = RERANK_QUERY, **overrides: Any) -> RetrievalQuery:
    """一个字段齐全的查询：``top_k=fetch_k=10``（十条样本一次取全）."""
    payload: dict[str, Any] = {"text": text, "top_k": 10, "fetch_k": 10}
    payload.update(overrides)
    return RetrievalQuery(**payload)


def stage1_hits(
    *,
    channels: dict[str, tuple[str, ...]] | None = None,
) -> list[RetrievalHit]:
    """第一阶段（重排之前）的十条命中——``rerank_hits`` 的典型输入.

    交错使用 ``RetrievalHit`` 而不是自己造一个形状：上游给过来的就是它，
    而"重排只读 id / 分数 / 名次 / 正文 / 通道"这句话必须能在这份输入上验证。
    """
    resolved_channels = channels or {}
    return [
        RetrievalHit(
            record_id=record_id,
            score=STAGE1_SCORES[record_id],
            rank=index,
            text=RECORD_TEXTS[record_id],
            metadata={"parent_doc_id": RECORD_SPECS[index][3]},
            channels=resolved_channels.get(record_id, ("vector",)),
        )
        for index, record_id in enumerate(RECORD_IDS)
    ]


def hand_features(record_id: str) -> RerankFeatures:
    """把 ``HAND_FEATURES`` 变成一副 ``RerankFeatures``（断言里逐项对照用）."""
    coverage, phrase, proximity, penalty = HAND_FEATURES[record_id]
    return RerankFeatures(
        term_coverage=coverage,
        exact_phrase=phrase,
        proximity=proximity,
        length_penalty=penalty,
    )


def lift_probes() -> list[LiftProbe]:
    """五条探针（``PROBE_SPECS`` 的 ``LiftProbe`` 形式，查询文本统一是 ``PROBE_QUERY``）."""
    return [LiftProbe(query=PROBE_QUERY, relevant=relevant) for _name, relevant in PROBE_SPECS]


def probe_of(name: str) -> LiftProbe:
    """按名字取一条探针（名字拼错时让 ``KeyError`` 指出可用名）."""
    table = dict(PROBE_SPECS)
    return LiftProbe(query=PROBE_QUERY, relevant=table[name])


class CountingRetriever:
    """把每一次检索都记下来的包装器（**"只检索一次"这件事的账**）.

    它存在的理由只有一条：``measure_lift`` 的正确性建立在"before 与 after
    来自**同一批命中**"之上。若它每条探针检索两次（先取 before、再取 after），
    那两条指标之差里就混着"两次检索的随机性"——用一个会计数的检索器，
    这条纪律就从一个说法变成一行断言（``calls == 探针数``）。
    """

    def __init__(
        self,
        retriever: Retriever | None = None,
        *,
        top_k: int = 10,
        fetch_k: int = 10,
    ) -> None:
        self._retriever = retriever if retriever is not None else rerank_vector_retriever()
        self._top_k = top_k
        self._fetch_k = fetch_k
        self.queries: list[str] = []

    @property
    def calls(self) -> int:
        """``retrieve`` 被调过几次（**应该恰好等于探针数**）."""
        return len(self.queries)

    def retrieve(self, query: str) -> RetrievalResult:
        """取一次结果（深度固定，好让 before 的名单长度是常量 10）."""
        self.queries.append(query)
        return self._retriever.retrieve(
            RetrievalQuery(text=query, top_k=self._top_k, fetch_k=self._fetch_k)
        )


# --------------------------------------------------------------------------- #
# 几个"可编程"的重排器（钉住那些内置替身无法构造的分支）
# --------------------------------------------------------------------------- #


class FixedReranker(BaseReranker):
    """按**正文**查表给分的重排器（分数由用例给定，因此次序完全可控）.

    键取正文而不是 id：``BaseReranker.score_pairs`` 只拿得到文本，而本模块的
    十条正文两两不同（见 ``RECORD_TEXTS``），于是 "id → 分数" 在构造时
    折成 "正文 → 分数"，用起来与按 id 指定一样直接。
    """

    def __init__(
        self,
        by_id: dict[str, float] | None = None,
        *,
        default: float = 0.0,
        name: str = "fixed-reranker",
        dimension: int = 1,
    ) -> None:
        self._by_text = {
            RECORD_TEXTS[record_id]: float(score) for record_id, score in (by_id or {}).items()
        }
        self._default = float(default)
        self._name = name
        self._dimension = dimension

    @property
    def name(self) -> str:
        """这个名字会进报告（用例断言它不被替换掉）."""
        return self._name

    @property
    def dimension(self) -> int:
        """一维打分（真实交叉编码器就是这个形状：一个相关性 logit）."""
        return self._dimension

    def score_pairs(self, query: str, texts: ListLike) -> list[float]:
        """按正文查表（未登记过的正文拿 ``default``）."""
        return [self._by_text.get(text, self._default) for text in texts]

    def describe(self) -> dict[str, Any]:
        """自述（不含任何逐次变化的账）."""
        return {
            "name": self._name,
            "kind": "fixed",
            "dimension": self._dimension,
            "entries": len(self._by_text),
        }


class ReversingReranker(FixedReranker):
    """把一批文本**逆序**打分（最后一条拿最高分）.

    它让"这次检索的次序确实来自重排"这件事变得无法含混：逆序后的名单必须
    逐项等于第一阶段的**倒序**（窗口内），而这与任何相似度都无关。
    """

    def __init__(self, *, name: str = "reversing-reranker") -> None:
        super().__init__(name=name, dimension=1)

    def score_pairs(self, query: str, texts: ListLike) -> list[float]:
        """第 i 条拿 ``i + 1``（越大越相关 → **最后一条**排第一）."""
        return [float(index + 1) for index in range(len(list(texts)))]


class TruncatingReranker(FixedReranker):
    """少返回一个分数（**错位不会报错**，因此必须由本层拦下）."""

    def score_pairs(self, query: str, texts: ListLike) -> list[float]:
        """故意比 ``texts`` 短一项."""
        return super().score_pairs(query, texts)[:-1]


class TextExplainReranker(FixedReranker):
    """``explain_pairs`` 返回一个**字符串**（提供方把一行当成整批）."""

    def explain_pairs(self, query: str, texts: ListLike) -> Any:
        """故意返回非序列."""
        return "not-a-feature-table"


class ShortExplainReranker(FixedReranker):
    """``explain_pairs`` 的行数比 ``texts`` 少（特征会错位到别的条上）."""

    def explain_pairs(self, query: str, texts: ListLike) -> list[dict[str, float]]:
        """故意少一行."""
        return [{} for _ in list(texts)[:-1]]


class NonDictExplainReranker(FixedReranker):
    """``explain_pairs`` 的每一项不是字典（键名与数值的对应关系就此丢失）."""

    def explain_pairs(self, query: str, texts: ListLike) -> list[Any]:
        """故意给出一串字符串."""
        return [f"row-{index}" for index, _ in enumerate(list(texts))]


class HitsLikeWithoutId:
    """有 ``score`` / ``rank`` 但**没有** ``record_id`` 的东西（上游给错了形状）."""

    def __init__(self, score: float = 1.0, rank: int = 0) -> None:
        self.score = score
        self.rank = rank


#: ``score_pairs`` 的 ``texts`` 形参的注解（本模块只读它，故用宽松别名）.
ListLike = Any


__all__ = [
    "DOC_WINDOW",
    "EMPTY_REASON_FRAGMENT",
    "HAND_BLEND_MIN_SCORE_ORDER",
    "HAND_BLEND_ORDER",
    "HAND_BLEND_VALUES",
    "HAND_DIVERSITY_DROPPED",
    "HAND_DIVERSITY_KEPT_WITHOUT_RERANK",
    "HAND_DIVERSITY_KEPT_WITH_RERANK",
    "HAND_FEATURES",
    "HAND_FUSED_ORDER",
    "HAND_FUSED_RRF",
    "HAND_HYBRID_RERANK_ORDER",
    "HAND_LEXICAL_ORDER",
    "HAND_LIFT",
    "HAND_LIFT_AFTER",
    "HAND_LIFT_BEFORE",
    "HAND_MIN_SCORE_050_DROPPED",
    "HAND_PROBE_LIFT",
    "HAND_PROBE_METRICS",
    "HAND_RERANK_MOVED",
    "HAND_RERANK_ORDER",
    "HAND_RERANK_SCORES",
    "HAND_RERANK_STAGE1_RANKS",
    "HAND_SCORES",
    "HAND_STAGE1_ORDER",
    "HAND_STAGE1_RANK",
    "HAND_WINDOW3_ORDER",
    "LIFT_K",
    "LONG_TEXT",
    "LONG_TEXT_LENGTH",
    "METRIC_NAMES",
    "PROBE_QUERY",
    "PROBE_SPECS",
    "PROXIMITY_ADJACENT",
    "QUERY_TERMS",
    "QUERY_VECTOR",
    "RECORD_IDS",
    "RECORD_SPECS",
    "RECORD_TEXTS",
    "RERANK_QUERY",
    "STAGE1_SCORES",
    "VECTOR_DIMENSION",
    "WINDOW_NOTE_FRAGMENT",
    "BaseReranker",
    "CountingRetriever",
    "CrossEncoderReranker",
    "FixedReranker",
    "HitsLikeWithoutId",
    "LexicalIndex",
    "LiftProbe",
    "NonDictExplainReranker",
    "RerankFeatures",
    "RetrievalHit",
    "RetrievalQuery",
    "RetrievalResult",
    "ReversingReranker",
    "ShortExplainReranker",
    "TableEmbedding",
    "TextExplainReranker",
    "TruncatingReranker",
    "empty_store",
    "hand_features",
    "lift_probes",
    "probe_of",
    "query_of",
    "rerank_embedding",
    "rerank_hybrid_retriever",
    "rerank_lexical",
    "rerank_reranker",
    "rerank_store",
    "rerank_vector_retriever",
    "sample_metadata",
    "sample_records",
    "stage1_hits",
]
