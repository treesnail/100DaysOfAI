#!/usr/bin/env python
"""day068 演示脚本：重排序（M6-D7）——交叉编码器**只对前 N 条**打分.

九节，全部**离线、确定性、零网络、零新增依赖**，只依赖本包与标准库：

```text
1. 第一阶段次序 vs 重排后次序  同一份名单两列并排，标出每一条移动了几个名次
2. 窗口纪律实测               12 条候选、top_n=5 → 打分了 5 条、尾部一条都没打分
3. 四个特征的实际数值         覆盖 / 整句 / 邻近 / 长度，加权和与引擎逐位对照
4. replace vs blend           两份分数各自 min-max 的中间值 + blended 手算对照
5. 阈值落刀                   min_score 只切窗口内；尾部 score=0.0 却不受它影响
6. 排序确定性                 并列时由谁说算（三次运行给出同一份名单）
7. 评估提升                   recall@k / RR / nDCG@k 的 before → after → lift
8. 两种集成的证据字段          单路（第 6.5 步）与混合（第 10.5 步）的 rerank_score / stage1_rank
9. 关闭重排                   rerank == {}、两个新字段是 None、名单逐位不变
```

## 这一课要证明的那句话

```text
双塔（bi-encoder）   查询与文档**分开**编码 → 文档侧可预计算 → 能扫百万条，但两两不相见
交叉编码器            查询与文档**一起**过模型 → 更准，也贵得多（每对一次前向推理）
```

因此标准姿势是两段式（第一阶段**召回**别漏，第二阶段**重排**排准）；本脚本要打印的
正是第二段那件事：**它把谁挪到了前面、代价是什么、账记在哪里**。

## 诚实声明：这里的"交叉编码器"是确定性替身

真实交叉编码器（bge-reranker / ms-marco-MiniLM 之类）要下载权重、要推理框架、通常还要
GPU——**本课一件都不许有**。``CrossEncoderReranker`` 因此是**教学级替身**：分数由四个
可手算的特征加权而成，**它没有真实语义**（"重算"与"重新计算"在它眼里是两个不同的字串，
语序它读不懂）。生产环境请替换为真实重排模型（接口 ``BaseReranker``）；换来的是
"每一个分数都能拿纸笔验算"——这正是"这次评估为什么变了"要的东西。

## 为什么这里的向量要写死成查表

要证明的是一条关于两段式分工的断言：**向量路对编号（``ERR-2043``）没有方向**，它会去
命中"讲向量是什么"的那条；而重排（词元覆盖 / 整句包含 / 邻近度）会把真正含这个编号的
那条捞到第 1 名。这条断言只能靠一份写死的映射构造（真实编码器的行为要看训练分布），
而**分数仍然是真算的**：余弦相似度在两个已知向量之间，手算能对上。

## 样本与输出

``tests/`` 里也有一份语料，但**脚本不 import tests**——脚本要能被复制走单独运行。因此
这里自带 12 条记录（正文与 day066/day067 的样本同源，便于对着文档核对）。结果同时打印到
stdout 并写入 ``outputs/rerank_demo.txt``；运行（cwd 为仓库根）::

    python scripts/rerank_demo.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

# 本脚本要能直接 ``python scripts/rerank_demo.py`` 跑起来（cwd 是仓库根）。
# 以脚本方式启动时 ``sys.path[0]`` 是 ``scripts/``，仓库根不在其中，
# 而本快照的包**没有 pip 安装**——因此这里显式把仓库根塞进 sys.path。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.config import settings  # noqa: E402
from smart_research_agent.llm.embedding import EmbeddingProvider  # noqa: E402
from smart_research_agent.retrieval import (  # noqa: E402
    DEFAULT_RERANK_MODEL,
    DEFAULT_RERANK_TOP_N,
    DEFAULT_RERANK_WEIGHT,
    IDEAL_LEN,
    RERANK_FEATURE_NAMES,
    RERANK_MODE_BLEND,
    RERANK_MODE_REPLACE,
    RERANK_MODES,
    RERANK_OVERRIDE_KEYS,
    CrossEncoderReranker,
    LexicalIndex,
    LiftProbe,
    RetrievalHit,
    RetrievalQuery,
    build_hybrid_retriever,
    build_retriever,
    measure_lift,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    rerank_hits,
    tokenize,
)
from smart_research_agent.retrieval.types import (  # noqa: E402
    CHANNEL_BM25,
    CHANNEL_VECTOR,
)
from smart_research_agent.vectorstore import FlatVectorStore  # noqa: E402
from smart_research_agent.vectorstore.types import make_record  # noqa: E402

#: 本层的边界（原文供教程引用，见 ``retrieval/types.py`` 的 ``RETRIEVAL_LIMITATIONS``）.
LAYER_BOUNDARY = (
    "重排只作用于前 N 条窗口：窗口之外的命中一条都不打分，"
    "因此它的名次只由第一阶段的分数决定；而教学级替身没有真实语义。"
)

#: 输出目录与文件名（``outputs/`` 在仓库 ``.gitignore`` 里，不污染仓库）.
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_NAME = "rerank_demo.txt"

#: 样本向量维度。取 8、分量只用 0 / 0.6 / 0.8 / 1.0（都已归一），
#: 于是余弦分数**可以手算**（0.8×0.8 + 0.6×0.6 = 1.0 这类），
#: 而"0 分并列"这件事是一个精确的事实（并列正是第 6 节要用的素材）。
DEMO_DIMENSION = 8

#: 三份文档（day062 的 ``parent_doc_id``；``retrieval_max_per_doc`` 缺省 0 = 不限）。
DOC_MANUAL = "doc-manual"
DOC_TROUBLE = "doc-trouble"
DOC_CONCEPT = "doc-concept"

#: 那条刻意写长的正文：它让 ``length_penalty`` 真的小于 1（标准长度 IDEAL_LEN=240）。
#: 它的长度由 ``len(text)`` 在运行时算出来（第 3 节打印），这里不做任何假设。
LONG_TEXT = (
    "排查 ERR-2043 的完整顺序：先确认编码器版本，再确认清单里的 dimension，"
    "然后确认入库时的 metric 是否一致，最后才怀疑数据本身。文档里每一个编号都应该能"
    "追到一段正文：编号以 err 打头、序号是 2043，这两件事分开看更容易定位。"
    "这一段刻意写得比标准长度 240 字更长，为的是让 length_penalty 这一项真的小于 1，"
    "从而四个特征都能在一张表里看到非平凡的取值。而标准长度本身是一个显式旋钮"
    "（CrossEncoderReranker(ideal_len=…)），不是藏在实现里的常数：它只影响这一项，"
    "且在总权重里占 1/8。把这段话说到底：长度惩罚只是从另一侧对冲排序模型的冗长偏好，"
    "它不是一条硬规则，也不该被读成'短文档更相关'。"
)

#: 十二条记录：``(id, 向量, 正文, 文档, strategy)``.
#:
#: 四处人为设计（缺一个第 1 节就讲不清）：d-01 与查询**同向**（cos=1.0）但一个编号
#: 都不含（阶段一的第 1 名因此是"瞎的"）；d-05 / d-11 含 ERR-2043 整句但向量 cos
#: 只有 0.0 / 0.48；d-11 与 d-05 的重排分**逐位相同**（第 6 节的并列素材）；
#: d-12 含 ERR-2043 且很长（四个特征都非平凡、长度惩罚 < 1）。
DEMO_SPECS: tuple[tuple[Any, ...], ...] = (
    ("d-01", (0.8, 0.6, 0, 0, 0, 0, 0, 0),
     "把一句话映射到低维空间的向量，越近表示意思越像。", DOC_CONCEPT, "structural"),
    ("d-02", (1.0, 0, 0, 0, 0, 0, 0, 0),
     "语义缓存用分位数标定阈值，换编码器必须重新标定。", DOC_MANUAL, "structural"),
    ("d-03", (0, 1.0, 0, 0, 0, 0, 0, 0),
     "召回深度按三倍过取，阈值在检索层落刀。", DOC_MANUAL, "structural"),
    ("d-04", (0.6, 0.8, 0, 0, 0, 0, 0, 0),
     "一次全量重建索引大约 12 万条记录，按每条 320 token 计。", DOC_MANUAL, "fixed"),
    ("d-05", (0, 0, 0, 0, 0, 0, 0, 1.0),
     "ERR-2043 表示向量维度不一致，先重建索引再重试。", DOC_TROUBLE, "errors"),
    ("d-06", (0, 0, 1.0, 0, 0, 0, 0, 0),
     "重试预算 retry_budget 缺省 3 次，超时按指数退避。", DOC_TROUBLE, "errors"),
    ("d-07", (0, 0, 0, 1.0, 0, 0, 0, 0),
     "报错码 401 与 403 的处置方式完全不同。", DOC_TROUBLE, "errors"),
    ("d-08", (0, 0, 0, 0, 1.0, 0, 0, 0),
     "术语表给出缩写与英文全称的对照。", DOC_CONCEPT, "glossary"),
    ("d-09", (0, 0, 0, 0, 0, 1.0, 0, 0),
     "关键词检索擅长精确匹配：编号、函数名、报错码，但它不认同义改写。", DOC_CONCEPT, "keyword"),
    ("d-10", (0, 0, 0, 0, 0, 0, 1.0, 0),
     "向量维度不一致时先看编码器版本，再看清单里的 dimension。", DOC_TROUBLE, "semantic"),
    ("d-11", (0.6, 0, 0, 0, 0, 0, 0.8, 0),
     "ERR-2043 这个编号既是整句包含，又只隔着一个空格，覆盖与整句两项都拿满。",
     DOC_TROUBLE, "errors"),
    ("d-12", (0, 0.6, 0, 0, 0, 0, 0.8, 0),
     LONG_TEXT, DOC_TROUBLE, "errors"),
)

#: 十二条记录 id（顺序 = 插入顺序）：用它做"整库都返回"的 top_k。
RECORD_IDS: tuple[str, ...] = tuple(str(spec[0]) for spec in DEMO_SPECS)

#: id → 正文（特征表与人工核对都用它）.
TEXT_BY_ID: dict[str, str] = {str(spec[0]): str(spec[2]) for spec in DEMO_SPECS}

#: 查询文本 → 查询向量（见模块 docstring：这张表是**断言**的一部分）.
#:
#: ``QUERY_EXACT`` 指向 d-01（"讲向量是什么"的那条），而真正含 ``ERR-2043`` 的是
#: d-05 / d-11 / d-12——这正是"双塔在编号面前差不多就行"的复现。
QUERY_VECTORS: dict[str, tuple[float, ...]] = {
    "ERR-2043": (0.8, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "401 报错码": (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "retry_budget": (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
}

#: 兜底查询向量（查询文本不在表里时用它）：第 1 根轴 = d-02 的方向。
DEFAULT_QUERY_VECTOR: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

#: 三个演示查询（名字与语义见上面那张表）.
QUERY_EXACT = "ERR-2043"
QUERY_CODE = "401 报错码"
QUERY_BOTH = "retry_budget"

#: 只用于特征表的"部分覆盖"查询：它的词元里有一个（401）不在 d-09 里。
QUERY_PARTIAL = "重试预算 退避"


def title(text: str) -> None:
    """打印一节标题."""
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def short(text: str, width: int = 34) -> str:
    """一行摘要（超长截断加省略号）——正文只用来核对，不占满屏幕."""
    return text if len(text) <= width else text[: width - 1] + "…"


def ids_of(result: Any) -> str:
    """一份结果/名单的 id 列表（打印用）."""
    return "、".join(hit.record_id for hit in result.hits)


def min_max(values: dict[str, float]) -> dict[str, float]:
    """手写的 min-max 归一化：与 ``fusion._min_max_normalize`` **同一条定义**.

    ```text
    一般情况   (v - min) / (max - min)
    min == max 定义为 1.0（**不是除零，也不是 0**）——一路只有一条命中时它必然
               min == max，归一化成 0.0 会让"唯一的那条证据"在加权和里消失
    ```

    第 4 节用它把 blend 的中间值算出来，再与引擎给的 ``RerankHit.blended`` 逐项对照。
    """
    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    if high == low:
        return {key: 1.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


# --------------------------------------------------------------------------- #
# 样本构造（离线小库 / 关键词索引 / 两个检索器）
# --------------------------------------------------------------------------- #


class TableEmbedding(EmbeddingProvider):
    """按文本查表的确定性编码器（本脚本的向量表，理由见模块 docstring）."""

    def __init__(
        self,
        *,
        table: dict[str, tuple[float, ...]],
        dimension: int,
        default: tuple[float, ...],
    ) -> None:
        self._table = dict(table)
        self._dimension = dimension
        self._default = tuple(default)

    @property
    def dimension(self) -> int:
        """向量维度（与库的维度必须一致，否则检索器在编码那一步就拒）."""
        return self._dimension

    def embed(self, text: str) -> list[float]:
        """查表；表里没有的文本用兜底向量（与 day066/day067 的演示同一做法）."""
        return list(self._table.get(text, self._default))


def demo_records() -> list[Any]:
    """十二条 ``VectorRecord``（元数据按 day062 的口径：``parent_doc_id`` / ``source`` / …）."""
    return [
        make_record(
            record_id=str(spec[0]),
            vector=tuple(float(value) for value in spec[1]),
            text=str(spec[2]),
            metadata={
                "parent_doc_id": str(spec[3]),
                "source": f"docs/{spec[3]}.md",
                "strategy": str(spec[4]),
            },
            metric="cosine",
        )
        for spec in DEMO_SPECS
    ]


def demo_store() -> FlatVectorStore:
    """装好十二条样本记录的 flat 库（维度与编码器必须一致，否则检索器在编码那一步就拒）."""
    store = FlatVectorStore(metric="cosine", dimension=DEMO_DIMENSION)
    store.upsert(demo_records())
    return store


def demo_embedding() -> TableEmbedding:
    """按 ``QUERY_VECTORS`` 查表的确定性编码器（8 维，与样本库同规格）."""
    return TableEmbedding(
        table=QUERY_VECTORS,
        dimension=DEMO_DIMENSION,
        default=DEFAULT_QUERY_VECTOR,
    )


def demo_lexical(store: FlatVectorStore) -> LexicalIndex:
    """从样本库现建一份关键词索引（BM25 的两个参数取 settings 的缺省）."""
    return LexicalIndex.from_backend(store)


def demo_vector_retriever(store: FlatVectorStore, **overrides: Any) -> Any:
    """只有向量一路的检索器（``build_retriever`` 装配；``**overrides`` 逐个覆盖）.

    **第 7 节的 ``retrieve`` 必须用它**（不带重排的那一份）：否则 "before" 里
    已经含着重排的结论，整份对照表就变成了"重排与重排比"。
    """
    return build_retriever(store, demo_embedding(), **overrides)


def stage1_hits(query: str = QUERY_EXACT, *, top_k: int = len(RECORD_IDS)) -> list[RetrievalHit]:
    """第一阶段（向量路）的完整名单：分数降序，同分按 ``record_id`` 升序."""
    result = demo_vector_retriever(demo_store()).retrieve(
        RetrievalQuery(text=query, top_k=top_k)
    )
    return list(result.hits)


# --------------------------------------------------------------------------- #
# 第 1 节：第一阶段次序 vs 重排后次序
# --------------------------------------------------------------------------- #


def section_1_stage1_vs_rerank() -> None:
    """同一份名单两列并排：重排把谁挪到了前面、挪了几个名次."""
    title("第 1 节：第一阶段次序 vs 重排后次序（同一份名单，两列并排）")
    print(f"边界（本层的原文）：{LAYER_BOUNDARY}")
    print(f"\n库：{len(RECORD_IDS)} 条 / {DEMO_DIMENSION} 维 / 度量 cosine；"
          f"编码器 TableEmbedding（按一张写死的向量表查表）")
    print(f"重排器：{CrossEncoderReranker().summary_line()}")
    print(f"常量：DEFAULT_RERANK_MODEL={DEFAULT_RERANK_MODEL!r}、"
          f"DEFAULT_RERANK_TOP_N={DEFAULT_RERANK_TOP_N}、IDEAL_LEN={IDEAL_LEN}")

    hits = stage1_hits(QUERY_EXACT)
    result = rerank_hits(hits, QUERY_EXACT, top_n=DEFAULT_RERANK_TOP_N)

    print(f"\n查询 {QUERY_EXACT!r}（词元：{tokenize(QUERY_EXACT)}）")
    print(f"  第一阶段（向量路）返回 {len(hits)} 条，分数降序、同分按 record_id 升序：")
    for hit in hits:
        print(f"      #{hit.rank:<2} {hit.record_id}  score={hit.score:+.6f} | {short(hit.text)}")

    print(f"\n  重排之后（mode={result.mode!r}，窗口 top_n={result.top_n}，"
          f"打分 {result.scored} 条）：")
    for hit in result.hits:
        print(f"      {hit.summary_line()}")

    before = [hit.record_id for hit in hits]
    after = [hit.record_id for hit in result.hits]
    print(f"\n  两份名单逐位相同：{before == after}；"
          f"名次真的变了的记录（按**新**名次）：{result.moved_ids()}")
    print(f"  第一名：第一阶段 {before[0]} → 重排之后 {after[0]}")
    print(
        "\n  读法：第一阶段把 d-01（'讲向量是什么'的那条）排在第 1 名（余弦 1.0），但它一个"
        "\n  编号都不含；真正含 ERR-2043 的三条（d-05 / d-11 / d-12）在第一阶段里分别是"
        "\n  第 6、4、5 名。重排把它们提到前三——这就是两段式要换来的东西。而它只能重排"
        f"\n  **它看得见的那一段**：本次 top_n={result.top_n} 覆盖了全部 {result.candidates} 条，"
        "第 2 节把窗口关小。"
    )


# --------------------------------------------------------------------------- #
# 第 2 节：窗口纪律实测
# --------------------------------------------------------------------------- #


def section_2_window() -> None:
    """候选 12 条 + top_n=5：打分了 5 条，尾部 7 条一条都没打分（计数器为证）."""
    title("第 2 节：窗口纪律实测（只对前 N 条打分）")
    hits = stage1_hits(QUERY_EXACT)
    print(f"候选 {len(hits)} 条（向量路把整库都返回了）；窗口 = 输入序列的**前 top_n 条**，"
          "尾部保持原顺序、一个都不进重排器。")

    for top_n in (5, 10):
        reranker = CrossEncoderReranker()
        before_pairs = reranker.scored_pairs
        before_calls = reranker.calls
        before_batches = reranker.batches
        result = rerank_hits(hits, QUERY_EXACT, reranker=reranker, top_n=top_n)
        scored_ids = [hit.record_id for hit in result.hits if hit.scored]
        unscored_ids = [hit.record_id for hit in result.hits if not hit.scored]
        print(f"\n  top_n={top_n}：候选 {result.candidates} 条 → "
              f"打分了 {len(scored_ids)} 条、没打分 {len(unscored_ids)} 条")
        print(f"      打过分：{scored_ids}")
        print(f"      没打分：{unscored_ids}（score=0.0、scored=False、stage1_rank 保持原值）")
        print(f"      结果的账：candidates={result.candidates}、scored={result.scored}、"
              f"window={result.window}、top_n={result.top_n}")
        print(f"      重排器计数器差量：calls +{reranker.calls - before_calls}、"
              f"scored_pairs +{reranker.scored_pairs - before_pairs}、"
              f"batches +{reranker.batches - before_batches}（累计 calls="
              f"{reranker.calls}、scored_pairs={reranker.scored_pairs}）；注记 "
              f"{len(result.notes)} 条：{short(result.notes[0], 52)}")

    print(
        "  为什么尾部那一批的 score 是 0.0 而不是 None：形状上保持'每条都有一个分数'，"
        "\n  '它其实没被打分'由 scored 标志与 notes 里那条无条件注记负责——0.0 不会被读成"
        "\n  '它最不相关'：窗口外的那批按原顺序排在窗口结果之后，名次与分数是两套证据。"
    )


# --------------------------------------------------------------------------- #
# 第 3 节：四个特征的实际数值
# --------------------------------------------------------------------------- #


def section_3_features() -> None:
    """四特征 + 加权和：手算与引擎逐位对照，外加一条"覆盖率满、邻近度低"的对照."""
    title("第 3 节：四个特征的实际数值（以及加权和与引擎的逐位对照）")
    reranker = CrossEncoderReranker()
    weights = CrossEncoderReranker.FEATURE_WEIGHTS
    print(f"特征清单（封闭）：{'、'.join(RERANK_FEATURE_NAMES)}")
    print(f"权重（顺序一一对应）：{weights} → 和 = {sum(weights)}（逐位等于 1.0）")
    print("四个特征都落在 [0, 1]，因此加权和也落在 [0, 1]——min_score 的可标定性正建立在这条上。")

    cases: tuple[tuple[str, str, str], ...] = (
        ("整句包含，编号紧挨", QUERY_EXACT, "d-05"),
        ("整句包含，另一个编号", QUERY_EXACT, "d-11"),
        ("整句包含但正文很长", QUERY_EXACT, "d-12"),
        ("一个编号都不含（阶段一的第 1 名）", QUERY_EXACT, "d-01"),
        ("覆盖率 6/6、邻近度很低", QUERY_PARTIAL, "d-06"),
    )
    print(f"\n  {'情形':<26} | {'id':<5} | {'覆盖':>6} | {'整句':>5} | {'邻近':>6} | "
          f"{'长度':>6} | {'手算和':>9} | {'引擎':>9} | 一致")
    for label, query, record_id in cases:
        text = TEXT_BY_ID[record_id]
        features = reranker.features_of(query, text)
        hand = 0.0
        for weight, name in zip(weights, RERANK_FEATURE_NAMES):
            hand += weight * getattr(features, name)
        engine = reranker.score_pairs(query, [text])[0]
        print(
            f"  {label:<26} | {record_id:<5} | {features.term_coverage:>6.4f} | "
            f"{features.exact_phrase:>5.1f} | {features.proximity:>6.4f} | "
            f"{features.length_penalty:>6.4f} | {hand:>9.6f} | {engine:>9.6f} | "
            f"{hand == engine}"
        )
        print(f"     查询 {query!r} → 词元 {tokenize(query)}；正文 {len(text)} 字 → "
              f"长度项 = min(1.0, {IDEAL_LEN} / {len(text)}) = {features.length_penalty:.6f}；"
              f"{features.summary_line()}")

    print(
        "\n  三件必须看出来的事：1) d-05 与 d-11 的四项**逐位相同** → 两个分数精确并列"
        "\n  （第 6 节的素材）；2) d-12 的长度项小于 1（冗长偏好被从另一侧对冲，但它只占"
        "\n  1/8 权重）；3) 邻近度只要求'所有词元都出现'才谈跨度——覆盖率是连续量。"
    )


# --------------------------------------------------------------------------- #
# 第 4 节：replace 与 blend
# --------------------------------------------------------------------------- #


def section_4_modes() -> None:
    """replace vs blend：窗口内两份分数各自归一化的中间值 + blended 手算对照."""
    title("第 4 节：两种模式 replace 与 blend（blend 的 min-max 中间值手算）")
    print(f"RERANK_MODES = {RERANK_MODES}；缺省 {RERANK_MODE_REPLACE!r}")
    print("replace  窗口内**只**按重排分排（第一阶段的分数降级为兜底的排序键）")
    print("blend    窗口内把两份分数**各自** min-max 归一化到 [0, 1]，再按 weight 加权")
    print(
        "量纲问题在这一层是**第二次**出现：第一阶段的分数量纲取决于上游（余弦 / BM25 /"
        "\nRRF 之和 / weighted 融合），而重排分落在 [0, 1]；直接加权等于让'量纲差'冒充"
        "\n'相关性差'——因此这里复用 fusion 的同一个归一化实现。"
    )

    hits = stage1_hits(QUERY_EXACT)
    window_n = 6
    weight = DEFAULT_RERANK_WEIGHT
    replaced = rerank_hits(hits, QUERY_EXACT, top_n=window_n, mode=RERANK_MODE_REPLACE)
    blended = rerank_hits(
        hits, QUERY_EXACT, top_n=window_n, mode=RERANK_MODE_BLEND, weight=weight
    )
    window_ids = [hit.record_id for hit in hits[:window_n]]
    print(f"\n窗口 top_n={window_n}（窗口内 {replaced.scored} 条，窗口外 "
          f"{replaced.candidates - replaced.scored} 条原样接在后面）、weight={weight}")
    print(f"  窗口的输入次序（= 第一阶段次序）：{'、'.join(window_ids)}")
    print(f"  replace 名单：{ids_of(replaced)}")
    print(f"  blend   名单：{ids_of(blended)}")
    print(f"  两份名单逐位相同："
          f"{[hit.record_id for hit in replaced.hits] == [hit.record_id for hit in blended.hits]}")

    stage1 = {hit.record_id: hit.score for hit in hits[:window_n]}
    rerank_score = {hit.record_id: hit.score for hit in replaced.hits[:window_n]}
    norm_stage1 = min_max(stage1)
    norm_rerank = min_max(rerank_score)
    blend_by_id = {hit.record_id: hit for hit in blended.hits}

    print("\n  两列分数各自在**窗口内**做 min-max（手算，只用窗口里这 6 条）：")
    print(f"      阶段一：min={min(stage1.values()):.6f}、max={max(stage1.values()):.6f}")
    print(f"      重排分：min={min(rerank_score.values()):.6f}、"
          f"max={max(rerank_score.values()):.6f}")
    print(
        f"\n  {'id':<5} | {'阶段一':>9} | {'重排分':>9} | {'归一阶段一':>10} | "
        f"{'归一重排':>10} | {'blended手算':>11} | {'blended引擎':>11} | 差"
    )
    for record_id in window_ids:
        engine_value = blend_by_id[record_id].blended
        hand = weight * norm_rerank[record_id] + (1.0 - weight) * norm_stage1[record_id]
        gap = abs(hand - engine_value) if engine_value is not None else float("nan")
        print(
            f"  {record_id:<5} | {stage1[record_id]:>9.6f} | {rerank_score[record_id]:>9.6f} | "
            f"{norm_stage1[record_id]:>10.6f} | {norm_rerank[record_id]:>10.6f} | "
            f"{hand:>11.6f} | {engine_value:>11.6f} | {gap:.1e}"
        )
    print("  逐项差都在浮点噪声量级（同一串运算 → 多数行精确为 0.0）")
    print(f"\n  replace 与 blend 这次的差别：\n      replace："
          f"{'、'.join(hit.record_id for hit in replaced.hits[:6])}\n      blend  ："
          f"{'、'.join(hit.record_id for hit in blended.hits[:6])}")
    print("      blend 保留了'第一阶段也说得过去'的 d-01（它归一化后是 1.0），"
          "代价是 d-12 被压到第 3 —— 这就是 weight=0.5 的含义：两边都不占先。")

    wider = rerank_hits(hits, QUERY_EXACT, top_n=8, mode=RERANK_MODE_BLEND, weight=weight)
    wider_by_id = {hit.record_id: hit for hit in wider.hits if hit.blended is not None}
    pairs = [
        (record_id, blend_by_id[record_id].blended, wider_by_id[record_id].blended)
        for record_id in window_ids
        if record_id in wider_by_id
    ]
    shift = max(pairs, key=lambda item: abs(item[1] - item[2]))
    print("\n  blend 的代价（相对量）：归一化只在**这一次窗口内**做，因此换个窗口就会变——")
    print(f"      {shift[0]} 的 blended={shift[1]:.6f}（top_n={window_n}）→ "
          f"{shift[2]:.6f}（top_n=8），差 {shift[2] - shift[1]:+.6f}；"
          "它只要求'同一次重排内部可比'。")


# --------------------------------------------------------------------------- #
# 第 5 节：阈值落刀
# --------------------------------------------------------------------------- #


def section_5_min_score() -> None:
    """min_score 只作用于被打分过的条；尾部连分数都没有，因此它切不到."""
    title("第 5 节：阈值落刀（只切**被打分过**的那批）")
    print(
        "min_score 比的是**重排分**（不是第一阶段的分数）：教学替身落在 [0, 1]，"
        "\n因此这个阈值有绝对含义——与 BM25 的'没有绝对标度、给它一个数字是假的安全感'"
        "\n正好相反。而它切掉几条，记在**第四个** dropped_* 数里"
        "（RetrievalResult.dropped_by_rerank）。"
    )
    hits = stage1_hits(QUERY_EXACT)
    ordered = rerank_hits(hits, QUERY_EXACT, top_n=DEFAULT_RERANK_TOP_N)
    print("\n  12 条的分数分布（replace 模式，只列去重后的取值）：")
    seen: dict[float, list[str]] = {}
    for hit in ordered.hits:
        seen.setdefault(round(hit.score, 6), []).append(hit.record_id)
    for value in sorted(seen, reverse=True):
        print(f"      {value:.6f} → {'、'.join(seen[value])}")

    for top_n, threshold in ((5, 0.9), (5, 2.0), (DEFAULT_RERANK_TOP_N, 2.0)):
        out = rerank_hits(hits, QUERY_EXACT, top_n=top_n, min_score=threshold)
        kept = [hit.record_id for hit in out.hits if hit.scored]
        tail = [hit.record_id for hit in out.hits if not hit.scored]
        print(f"\n  top_n={top_n}、min_score={threshold}：窗口 {out.window} 条 → "
              f"留下 {len(kept)} 条、切掉 {out.dropped_by_min_score} 条；"
              f"留下的：{kept or '（窗口内一条都没留下）'}")
        tail_note = (
            f"窗口外的 {len(tail)} 条：{'、'.join(tail)} ——它们的 score 全是 0.0，"
            "**没有一条被阈值切掉**"
            if tail
            else "窗口外没有尾部：窗口覆盖了全部候选，被切掉的每一条都真的消失了"
        )
        print(f"      {tail_note}")
        print(f"      结果：命中 {out.count} 条、empty_reason="
              f"{short(out.empty_reason, 30) or '（空串）'}；注记："
              f"{short(next(n for n in out.notes if 'min_score' in n), 60)}")

    print(
        "\n  三条要点：1) 第二行（top_n=5、min_score=2.0）里窗口被切光了，但名单**不是空的**："
        "\n  窗口外那 7 条照样在（score=0.0、scored=False）——'切光窗口'与'空结果'是两件事。"
        "\n  2) 第三行（窗口覆盖全部 12 条）才是真的空结果，此时 empty_reason 必须给出唯一成因，"
        "\n  并由 dropped_by_min_score 记账。3) 阈值切不到尾部：**它没有分数可切**。"
    )


# --------------------------------------------------------------------------- #
# 第 6 节：排序确定性
# --------------------------------------------------------------------------- #


def tie_order(first_id: str, second_id: str) -> tuple[list[str], float]:
    """两条**同分同正文**的命中（输入顺序由参数给出），返回重排后的 id 次序与分数."""
    text = TEXT_BY_ID["d-05"]
    pair = [
        RetrievalHit(record_id=first_id, score=0.5, rank=0, text=text),
        RetrievalHit(record_id=second_id, score=0.5, rank=1, text=text),
    ]
    out = rerank_hits(pair, QUERY_EXACT, top_n=5)
    return [hit.record_id for hit in out.hits], out.hits[0].score


def section_6_determinism() -> None:
    """三次运行同一份名单；并列由第二键（输入位置）决定，第三键是防御性兜底."""
    title("第 6 节：排序确定性（并列时由谁说算）")
    print(
        "排序键（代码里逐字写着）：replace → (-score, stage1_rank, record_id)、"
        "\nblend → (-blended, stage1_rank, record_id)。三个键缺一不可：只按分数排序时，"
        "\n'两条分数逐位相同'会让名次要靠 sorted 的稳定性——那是**没写下来的**排序规则，"
        "\n而同一份数据两次运行给出不同 top-1 是最难查的一类 bug。"
    )

    hits = stage1_hits(QUERY_EXACT)
    runs = [rerank_hits(hits, QUERY_EXACT, top_n=DEFAULT_RERANK_TOP_N) for _ in range(3)]
    orders = [[hit.record_id for hit in run.hits] for run in runs]
    print("\n  同一批输入跑三次（每次都是新的 RerankResult）：")
    for index, order in enumerate(orders, start=1):
        print(f"      第 {index} 次：{'、'.join(order)}")
    print(f"      三次逐位相同：{orders[0] == orders[1] == orders[2]}")

    print("\n  语料里就躺着一对**精确并列**：d-05 与 d-11 的四个特征逐位相同")
    ties = [hit for hit in runs[0].hits if hit.record_id in ("d-05", "d-11")]
    for hit in ties:
        print(f"      {hit.record_id}  rerank={hit.score!r}  "
              f"stage1_rank={hit.stage1_rank}  rank={hit.rank}")
    print(f"      两条分数相等：{ties[0].score == ties[1].score}；次序 "
          f"{ties[0].record_id} → {ties[1].record_id}，由第二键 stage1_rank"
          f"（{ties[0].stage1_rank} < {ties[1].stage1_rank}）决定")
    print("      注意：**不是**由 record_id 决定（d-05 < d-11 只是碰巧同向），"
          "也不代表'重排器认为 d-05 更相关'（两个分数一模一样）。")

    forward, score_forward = tie_order("tie-b", "tie-a")
    backward, score_backward = tie_order("tie-a", "tie-b")
    print("\n  把并列造得更极端：同一段正文、同一个阶段一分数（0.5），只换输入顺序")
    print(f"      输入 ['tie-b', 'tie-a'] → {forward}；输入 ['tie-a', 'tie-b'] → {backward}"
          f"（两条重排分逐位相同：{score_forward == score_backward}）")
    print("      次序**跟着输入位置走**（这正是第二键的定义）；第三键 record_id 在这条路径上"
          "\n      不会被触发——stage1_rank 是输入序列的位置，它本身已经把每一条区分开了，"
          "\n      写在那里是**防御性兜底**：让'同分'在任何调用方式下都有一个确定的答案。")


# --------------------------------------------------------------------------- #
# 第 7 节：评估提升
# --------------------------------------------------------------------------- #

#: 三条评估探针：一句话 + 它的金标准 id（``relevant`` 是本报告里唯一来自外部的输入）.
#:
#: 三条刻意选出三种结果：第一条**明显提升**（阶段一的第 1 名与金标准无关）、
#: 第二条**从 0 到 1**（金标准那条在阶段一的第 6 名，被 top_k 挡在门外）、
#: 第三条**一动不动**（阶段一已经把它排第 1）——"提升"不是必然的，报告要敢写 0。
PROBES: tuple[LiftProbe, ...] = (
    LiftProbe(query=QUERY_EXACT, relevant=("d-05", "d-11")),
    LiftProbe(query=QUERY_CODE, relevant=("d-07",)),
    LiftProbe(query=QUERY_BOTH, relevant=("d-06",)),
)


def section_7_lift() -> None:
    """LiftProbe 列表 → measure_lift → before / after / lift 三个字典与逐条明细."""
    title("第 7 节：评估提升（recall@k / RR / nDCG@k 的 before → after）")
    print(
        "三条指标的公式（实现里逐字对应）：recall@k = |前 k 条 ∩ 金标准| / |去重后的金标准|；"
        "\nRR = 1 / (第一个相关命中的位置 + 1)，一个都没有则 0.0；nDCG@k = DCG@k / IDCG@k，"
        "\nDCG@k = Σ_rel_i / log2(i + 2)（二值增益，i 从 0 起），"
        "\nIDCG@k = Σ 1 / log2(i + 2)（i < min(|金标准|, k)）。三者必须一起看：只看"
        "\nrecall@k 时'把第 4 名提到第 1 名'完全不可见，只看 RR 时'前 k 条命中 3 条还是"
        "\n1 条'不可见——nDCG@k 是唯一同时看'命中几条'与'排得多靠前'的那条。"
    )

    k = 5
    retriever = demo_vector_retriever(demo_store())

    def retrieve_deep(text: str) -> Any:
        """取一份**比 k 更深**的名单：重排才有机会把窗口外的条提进前 k 条."""
        return retriever.retrieve(RetrievalQuery(text=text, top_k=len(RECORD_IDS)))

    print(f"\n探针 {len(PROBES)} 条（k={k}）；retrieve 用**不开重排**的那一份检索器，并刻意取满"
          f" {len(RECORD_IDS)} 条（比 k 深）：measure_lift 的 after 只能重排 retrieve 交上来的"
          "\n      那份名单——若它已被 top_k 截到 k 条，'召回提升'在这份报告里**不可能出现**。"
          "\n      变量只有一个'有没有重排'：不是检索两次，而是同一批命中跑两遍。")
    for probe in PROBES:
        print(f"      {probe.summary_line()}")

    report = measure_lift(PROBES, retrieve_deep, k=k, mode=RERANK_MODE_REPLACE)
    print(f"\n  {report.summary_line()}")
    print(f"      before = {report.before}")
    print(f"      after  = {report.after}")
    print(f"      lift   = {report.lift}（差值而不是比值：before=0 时比值是无穷）")

    print("\n  逐条明细：")
    for row in report.per_query:
        print(f"      查询 {row['query']!r}（金标准 {row['relevant']}）")
        print(f"          before：{'、'.join(row['before_ids'])}")
        print(f"          after ：{'、'.join(row['after_ids'])}")
        print(f"          指标 before={row['before']} after={row['after']} lift={row['lift']}")
        print(f"          名次变化：{row['moved'] or '（无）'}；窗口 candidates="
              f"{row['candidates']}、scored={row['scored']}、top_n={row['top_n']}、"
              f"mode={row['mode']!r}、weight={row['weight']}")

    probe = PROBES[0]
    before_ids = report.per_query[0]["before_ids"]
    pool = set(probe.relevant)
    head = before_ids[:k]
    first_position = next(
        (index for index, record_id in enumerate(before_ids) if record_id in pool), None
    )
    dcg = sum(
        1.0 / math.log2(index + 2.0)
        for index, record_id in enumerate(head)
        if record_id in pool
    )
    ideal = sum(1.0 / math.log2(index + 2.0) for index in range(min(len(pool), k)))
    print("\n  手算核对第 1 条探针的 before（纸笔，不从引擎拿数）：")
    print(f"      名单 {head}、金标准 {sorted(pool)}（分母 = 去重后 {len(pool)} 条）、"
          f"命中位置 {[index for index, rid in enumerate(head) if rid in pool]}")
    print(f"      DCG@k = Σ 1/log2(i+2) = {dcg:.6f}；"
          f"IDCG@k = Σ(i<min({len(pool)},{k})) = {ideal:.6f}")
    print(f"      recall@k = {len(set(head) & pool)}/{len(pool)} = "
          f"{len(set(head) & pool) / len(pool):.6f}"
          f"（引擎 recall_at_k = {recall_at_k(before_ids, probe.relevant, k):.6f}）；"
          f"nDCG@k = {dcg / ideal:.6f}"
          f"（引擎 ndcg_at_k = {ndcg_at_k(before_ids, probe.relevant, k):.6f}）")
    print(f"      RR = 1/({first_position}+1) = {1.0 / (first_position + 1):.6f}"
          f"（引擎 reciprocal_rank = {reciprocal_rank(before_ids, probe.relevant):.6f}）")

    print(f"\n  这份报告的形状：LiftReport(queries / k / before / after / lift / per_query)；"
          f"三张指标表必须逐键对齐（{list(report.metrics)}）——少一个键'这一条提升了多少'"
          "\n  就答不出来。逐条明细里留下了两侧名单、三条指标、谁的名次动了、窗口的账。")


# --------------------------------------------------------------------------- #
# 第 8 节：两种集成下的证据字段
# --------------------------------------------------------------------------- #


def section_8_integration() -> None:
    """单路（第 6.5 步）与混合（第 10.5 步）两条流水线的重排证据：同一对原语."""
    title("第 8 节：两种集成下的证据字段（单路第 6.5 步 / 混合第 10.5 步）")
    print(
        "两处调用共用同一对原语：rerank_hits（纯函数：切窗口 / 打分 / 排序）与"
        "\nretriever._rebuild_hits（写回：次序与名次取自重排，其余全部取自原命中）——"
        "\n两处各写一份写回逻辑一定会分家（一份忘了 channels、一份忘了 stage1_rank），"
        "\n而分家不会报错，只会让报告少一列。"
    )
    print("\n  单路 retriever.py 第 6.5 步：阈值落刀**之后**、多样性裁剪**之前**；"
          "\n  混合 hybrid.py 第 10.5 步：融合**之后**、多样性裁剪**之前**。")
    print("  顺序只有这一种排法：多样性会按 doc_id 丢掉同文档的其余记录，"
          "若先裁后重排，被丢掉的那些候选**再也回不到重排器眼前**。")

    store = demo_store()
    embedding = demo_embedding()

    single = demo_vector_retriever(
        store,
        rerank_enabled=True,
        rerank_top_n=8,
        rerank_mode=RERANK_MODE_REPLACE,
    )
    result = single.retrieve(RetrievalQuery(text=QUERY_EXACT, top_k=5))
    print("\n  [A] 单路：rerank_enabled=True、rerank_top_n=8、mode='replace'、top_k=5")
    print(f"      fetch_k={result.fetch_k}、candidates={result.candidates}、"
          f"命中 {result.count} 条")
    print(f"      {'rank':>4} | {'id':<5} | {'score(阶段一)':>14} | {'rerank_score':>12} | "
          f"{'stage1_rank':>11} | {'moved':>5}")
    for hit in result.hits:
        moved = 0 if hit.stage1_rank is None else hit.stage1_rank - hit.rank
        print(f"      {hit.rank:>4} | {hit.record_id:<5} | {hit.score:>14.6f} | "
              f"{hit.rerank_score:>12.6f} | {hit.stage1_rank:>11} | {moved:>+5d}")
    score_sequence = [round(hit.score, 6) for hit in result.hits]
    print(f"      score 序列 = {score_sequence}；它在重排过的名单里**不再单调**："
          f"{all(a >= b for a, b in zip(score_sequence, score_sequence[1:]))}"
          "——score 的口径没有变（仍是第一阶段的分数），"
          "因此'排序依据看 rerank_score，跨版本对比看 score'。")
    print(f"      result.rerank = {result.rerank}")
    print(f"      result.dropped_by_rerank = {result.dropped_by_rerank}（第四个 dropped_* 数）；"
          f"四道减法：below_threshold={result.dropped_below_threshold}、"
          f"by_diversity={result.dropped_by_diversity}、"
          f"by_rerank={result.dropped_by_rerank}、by_top_k={result.dropped_by_top_k}")
    print("      notes 里与重排有关的行，以及 explain() 里那一行重排口径：")
    for note in result.notes:
        if "重排" in note:
            print(f"          {short(note, 72)}")
    for line in single.explain(result):
        if "重排" in line:
            print(f"          explain → {short(line, 132)}")
    print("      那一行里的'只对前 N 条'读的是摘要的 params['top_n']（回落 window）；"
          "摘要顶层没有\n      top_n 这个键——早先按顶层键读时这里会显示 0，"
          "与同一行的 scored 自相矛盾（已修）。")

    lexical = demo_lexical(store)
    hybrid_off = build_hybrid_retriever(
        store, embedding, lexical=lexical, strategy="weighted", alpha=0.9
    )
    hybrid_on = build_hybrid_retriever(
        store, embedding, lexical=lexical, strategy="weighted", alpha=0.9,
        rerank_enabled=True, rerank_top_n=8,
    )
    full_query = RetrievalQuery(text=QUERY_EXACT, top_k=len(RECORD_IDS))
    fused_full = hybrid_off.retrieve(full_query)
    reranked_full = hybrid_on.retrieve(full_query)
    fused_order = [hit.record_id for hit in fused_full.hits]
    stage1_order = [
        hit.record_id for hit in sorted(reranked_full.hits, key=lambda item: item.stage1_rank or 0)
    ]
    print("\n  [B] 混合：strategy='weighted'、alpha=0.9（几乎只看向量路）")
    print(f"      融合之后的次序：{'、'.join(fused_order)}；两路召回 vector "
          f"{fused_full.channel_candidates[CHANNEL_VECTOR]} 条 / bm25 "
          f"{fused_full.channel_candidates[CHANNEL_BM25]} 条，并集 {fused_full.candidates} 条")
    print(f"      重排的输入次序（按 stage1_rank 复原）与融合次序逐位相同："
          f"{fused_order == stage1_order}")

    reranked = hybrid_on.retrieve(RetrievalQuery(text=QUERY_EXACT, top_k=5))
    print("\n      开重排之后的名单（每条带证据）与它原来的位置：")
    for hit in reranked.hits:
        moved = 0 if hit.stage1_rank is None else hit.stage1_rank - hit.rank
        print(f"          #{hit.rank} {hit.record_id} | rank {hit.stage1_rank} → {hit.rank}"
              f"（{moved:+d}） | channel={hit.channel!r} | channels={hit.channels} | "
              f"score={hit.score:.6f} → rerank={hit.rerank_score:.6f}")
    print(f"      result.rerank = {reranked.rerank}；"
          f"dropped_by_rerank = {reranked.dropped_by_rerank}")
    for line in hybrid_on.explain(reranked):
        if "重排口径" in line:
            print(f"      融合口径那一行（重排在融合之后）：{short(line, 120)}")
    print(
        "\n  两条集成的同一句结论：重排**不改 score**、不改 channels、不改 text，"
        "\n  它只写两个新字段（rerank_score / stage1_rank）与最终次序。"
        "\n  因此'重排把多路证据弄丢了'这种事在形状上不可能发生。"
    )


# --------------------------------------------------------------------------- #
# 第 9 节：关闭重排
# --------------------------------------------------------------------------- #


def section_9_disabled() -> None:
    """关着重排：一个字段都不改写、rerank == {}、名单逐位不变."""
    title("第 9 节：关闭重排（逐位不变的对照）")
    print(f"settings.retrieval_rerank_enabled = {settings.retrieval_rerank_enabled}"
          "（config.py 缺省 False = 新能力默认不生效）；extra 的封闭清单 "
          f"RERANK_OVERRIDE_KEYS = {RERANK_OVERRIDE_KEYS}")
    print("      注意 min_score **不在**这份清单里：阈值只能由构造参数或 settings 决定，"
          "逐次覆盖\n      能改的是 enabled / mode / model / top_n / weight 这五个。")

    store = demo_store()
    query = RetrievalQuery(text=QUERY_EXACT, top_k=5)
    off = demo_vector_retriever(store).retrieve(query)
    off_explicit = demo_vector_retriever(store, rerank_enabled=False).retrieve(query)
    on = demo_vector_retriever(store, rerank_enabled=True, rerank_top_n=8).retrieve(query)

    print(f"\n  关闭（走 settings 缺省）：{ids_of(off)}")
    print(f"  关闭（显式 False）      ：{ids_of(off_explicit)}")
    print(f"  打开（rerank_top_n=8）  ：{ids_of(on)}")
    print(f"  两次关闭逐位相同：{ids_of(off) == ids_of(off_explicit)}；"
          f"分数逐位相同："
          f"{[hit.score for hit in off.hits] == [hit.score for hit in off_explicit.hits]}")
    print(f"  关闭时 rerank == {{}}：{off.rerank == dict()}、dropped_by_rerank="
          f"{off.dropped_by_rerank}、两个新字段全 None："
          f"{[hit.rerank_score for hit in off.hits]} / {[hit.stage1_rank for hit in off.hits]}")
    print(f"\n  打开时 result.rerank 的键：{sorted(on.rerank.keys())}")
    print(f"  打开时 rerank_score：{[hit.rerank_score for hit in on.hits]}")
    print(f"  打开时 stage1_rank：{[hit.stage1_rank for hit in on.hits]}")
    fresh = demo_vector_retriever(demo_store()).retrieve(query)
    print(f"\n  '逐位不变'再确认一次：关闭时的名单 == 一个'根本没配过重排'的检索器给的名单："
          f"{ids_of(off) == ids_of(fresh)}")
    print(f"\n  extra 的两种结局（见手册第 8 节）：{'、'.join(RERANK_OVERRIDE_KEYS)} 五个键"
          "可以逐次覆盖，\n  覆盖生效必须进 notes；清单外的键由**两份清单的并集**判定，"
          "一律报错（'静默忽略\n  一个覆盖参数'这条纪律一个字都没放松）。")


# --------------------------------------------------------------------------- #
# 输出：同时写终端与文件
# --------------------------------------------------------------------------- #


class _Tee:
    """把写往 stdout 的内容**同时**送到终端与文件（模块 docstring 的硬要求）."""

    def __init__(self, *streams: Any) -> None:
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def main() -> int:
    """按节运行演示，并把完整输出同时写入 ``outputs/rerank_demo.txt``."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / OUTPUT_NAME
    original = sys.stdout
    with output_path.open("w", encoding="utf-8") as handle:
        sys.stdout = _Tee(original, handle)
        try:
            for section in (
                section_1_stage1_vs_rerank,
                section_2_window,
                section_3_features,
                section_4_modes,
                section_5_min_score,
                section_6_determinism,
                section_7_lift,
                section_8_integration,
                section_9_disabled,
            ):
                section()
            print(f"\n演示完成（工作目录 {Path.cwd()}）；输出已同时写入 {output_path}")
            print("九节全部离线：零网络、零新增依赖；重排分、四个特征、"
                  "blend 的中间值与三条指标都是真算出来的。")
        finally:
            sys.stdout = original
    return 0


if __name__ == "__main__":
    sys.exit(main())
