"""混合检索：**两路一起答，并说清每一条是哪一路顶上来的**（M6-D6 / M6-D7）.

day066 交了单路向量检索器（一行流水线九步，三道减法各自留数字）。
day067 这一层不重写那条流水线，只回答一个新问题：

```text
向量路与关键词路各给一份名单 → 合成的那一份该长什么样？
                              → 而每一条的"凭什么在这里"还能不能被读出来？
```

## 固定顺序（每一步都在为"可解释"服务）

```text
1.  规范化查询     str → RetrievalQuery；None 字段沿用向量路的默认值
2.  融合参数解析   ctor 默认 ← RetrievalQuery.extra（day066 预留的那个槽）
2.5 重排参数解析   同上（ctor 默认 ← extra）；键用 RERANK_OVERRIDE_KEYS 封闭清单
3.  算召回深度     fetch_k 由向量路那**一份**实现算出，两路共用同一个深度
4.  索引体检       条数与清单漂移（沿用向量路的 _inspect_index，不写第二套判据）
5.  合成过滤子句   combine_where 只调一次，两路拿到的是**同一个对象**
6.  关键词路取候选 lexical.search(text, fetch_k, where=那同一份子句)
7.  向量路取候选   编码 + backend.query(fetch_k, where=那同一份子句)
8.  阈值落刀（只对向量路）  见下"阈值语义"
9.  融合           两路名次 → 一份带完整证据的名单（fusion.fuse）
10. 装配命中       channel = 贡献最大的那一路；channels = 全部证据
10.5 重排（可选）  只对前 top_n 条重打分，按重排分重编名次（day068）
11. 多样性裁剪     按 doc_id 分组（沿用 _apply_diversity）
12. 截断到 top_k   截断前先重排名次（融合的次序是权威次序）
13. 诊断与装配     empty_reason / channel_candidates / fusion / rerank 摘要 / notes
```

第 10.5 步的位置只有一种排法，理由与单路那条流水线完全相同（day068）：
``多样性的动作是丢掉同文档的其余记录``，若先裁后重排，被丢掉的那些候选
**再也回不到重排器眼前**——"同一文档的第 2 段其实更贴题"这个事实就无法被发现。
因此顺序是"重排（在完整候选集上比较）→ 多样性（在重排后的次序上取舍）"，
而重排放在**融合之后**是因为：融合已经把所有通道的证据摆好了，
重排看到的每一段文本都是"被某些路真实召回过的"，而不是某一路的内部抽签结果。

## 为什么第 5 步的"同一份子句"值得单独讲（本课最值得讲的一个坑）

``where`` 过滤**必须在两路都落刀**，而且必须是**同一份**子句。这条看似显然，
但它有一个非常安静的失败模式：只在一路过滤时，另一路会把"已经被排除的东西"
重新抬进结果里——**而且不报错**。

```text
只过滤向量路    用户 where={"strategy": "structural"} → 向量路只取 structural 的
                关键词路不过滤 → 它把 fixed/semantic 的记录也捞了回来
                融合之后：结果里出现了过滤条件明确排除掉的记录
                而报告里 filter_applied=True（"过滤生效了"），没有任何异常
```

这类"少而不报"的反面——"多而不报"——比少更危险：少了几条还能靠对比两次运行发现，
多了几条在报告里看起来完全正常（谁也不会去核对每一条的 strategy 是不是 structural）。
因此本模块把子句合成放在**两路取候选之前的一次调用**里，并把"两路用的是同一份"
写进代码结构（同一个变量交给两个调用点），而不是指望以后加第三路的人记得同步。

## 阈值语义：``min_score`` **只作用于向量通道**（在融合前落一次刀）

```text
向量通道   有绝对标度（余弦 ∈ [-1, 1]）→ 可以设阈值，而且这次落刀的数字要报出来
关键词通道 没有绝对标度（BM25 ∈ [0, ∞)，同一份文档在两个查询上的 7.31 不可比）
           → **不设阈值**：给它一个数字是"假的安全感"（见下）
```

"假的安全感"是这段话里最要紧的词。BM25 的分数分布取决于语料（词频、文档长度、
词表大小）与查询长度，因此 ``min_score=3.0`` 在一份 10 篇文档的手册上可能切掉一半，
在一份 100 万篇的语料上可能一条都不切。给一个**看起来能调**的参数，
后果是出问题的人去调它、发现没有稳定效果、于是转去调别的参数——
**一个不存在的旋钮比没有旋钮更贵**。

它与 ``config.retrieval_min_score = None`` 的缺省理由是同一句话：
"标定之前给一个数字是假的安全感"。

被忽略这件事**必须进 notes**（即使这次它一条都没切）：
"min_score=X 只作用于向量通道"这句话要出现在每一次带阈值的混合检索里，
否则"我设了阈值为什么关键词路还在返回低分记录"会变成一个反复出现的疑问。

## 结果里多出来的两类证据

```text
RetrievalResult.channels            综合之后出现过的全部通道
RetrievalResult.candidates          两路候选的**并集**（同一条两路都召回只算一次）
RetrievalResult.channel_candidates  每路各自的候选数（"哪一路没捞到东西"看它）
RetrievalResult.fusion              策略 / 参数 / 每路召回数 / 每路贡献数 / 去重条数
RetrievalResult.rerank              重排摘要（模型 / 模式 / 窗口 / 打分 / 切掉几条）← day068
RetrievalResult.dropped_by_rerank   被重排阈值切掉几条（第四个 dropped_* 数）    ← day068
RetrievalHit.channel                贡献最大的那一路（并列时向量优先）
RetrievalHit.channels               这一条被哪几路召回（按名次好坏排序）
RetrievalHit.rerank_score           重排分（None = 这次没开重排）                ← day068
RetrievalHit.stage1_rank            重排**前**它在名单里的位置（None = 没开重排） ← day068
```

``channel_candidates`` 与 ``candidates`` 必须一起看：并集小于两路之和的差值，
就是"被两路同时命中"的条数（融合最强的那类证据），而这个数字**只能**从
两个数一起读出来。

day068 加的两组字段是**只增不改**的：``rerank`` 是空字典、``rerank_score`` /
``stage1_rank`` 是 ``None`` 时，就表示"这次没开重排"——名单与 day067 逐位相同。
打开之后要看的是"名次动了哪几条"（``stage1_rank`` 与 ``rank`` 的差）
与"有多少条连分都没拿到"（``rerank.scored`` 与 ``rerank.candidates`` 的差）：
后者是重排窗口的代价，它**必须**能被读出来（那条窗口注记在 notes 里是无条件的）。
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from smart_research_agent.config import settings
from smart_research_agent.indexing.types import IndexManifest
from smart_research_agent.llm.embedding import EmbeddingProvider
from smart_research_agent.retrieval import retriever as retriever_module
from smart_research_agent.retrieval.errors import (
    FusionError,
    IndexStateError,
    QueryError,
    RerankError,
)
from smart_research_agent.retrieval.filters import combine_where, unknown_filter_fields
from smart_research_agent.retrieval.fusion import (
    FUSION_OVERRIDE_KEYS,
    FUSION_RRF,
    FUSION_WEIGHTED,
    FusedHit,
    contribution_counts,
    fuse,
)
from smart_research_agent.retrieval.lexical import BM25Params, LexicalIndex
from smart_research_agent.retrieval.rerank import (
    RERANK_OVERRIDE_KEYS,
    BaseReranker,
    CrossEncoderReranker,
    rerank_hits,
)
from smart_research_agent.retrieval.retriever import Retriever, build_retriever
from smart_research_agent.retrieval.types import (
    CHANNEL_BM25,
    CHANNEL_VECTOR,
    CHANNELS,
    DEFAULT_DOC_ID_FIELD,
    EMPTY_REASON_NO_DATA,
    IndexState,
    RetrievalHit,
    RetrievalQuery,
    RetrievalResult,
)
from smart_research_agent.vectorstore.base import VectorBackend

#: 混合检索器的默认名字（报告与端点回显它）。
DEFAULT_HYBRID_NAME = "hybrid"


@dataclass(frozen=True)
class _RerankPlan:
    """一次检索**实际生效**的重排参数（第 2.5 步的产物；私有形状）.

    它存在的理由与 ``RetrievalQuery`` 相同：**"这次用的是哪组参数"必须能独立复述**。
    两层来源（本层构造参数 ← ``RetrievalQuery.extra``）合起来之后，
    六个值散在四个局部变量里对不上号——而第 10.5 步要用它们的**全部**。

    ``model`` 是唯一一个"只进报告"的字段（缺省 ``None`` 表示用
    ``reranker.name``）：本模块不按名字加载模型，但允许调用方在报告里
    标出"这次理应是谁"（真实部署里通常是网关侧的路由名）。
    """

    enabled: bool
    reranker: BaseReranker
    top_n: int
    mode: str
    weight: float
    min_score: float | None
    model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典（诊断里回显"这次用的哪组参数"）."""
        return {
            "enabled": self.enabled,
            "top_n": self.top_n,
            "mode": self.mode,
            "weight": self.weight,
            "min_score": self.min_score,
            "model": self.model if self.model is not None else self.reranker.name,
        }


class HybridRetriever:
    """混合检索器：向量路 + 关键词路，融合成一份**带多路证据**的结果.

    构造参数分三组，与 day066 的 ``Retriever`` 同一风格：

    ```text
    融合类    strategy / alpha / k_rrf / weights     "两路怎么合"
    决策类    max_per_doc / strict_index             "留下哪些、漂移要不要阻断"
    重排类    rerank_enabled / reranker /            "融合之后那一步怎么排"（day068）
              rerank_top_n / rerank_mode /
              rerank_weight / rerank_min_score
    ```

    ``None`` 在融合类与重排类参数上有明确含义：**"没指定，去读 settings"**
    （``retrieval_hybrid_strategy`` / ``retrieval_hybrid_alpha`` /
    ``retrieval_hybrid_rrf_k``，以及 day068 的 ``retrieval_rerank_*``）。
    于是"项目默认值"只有一处定义，端点与演示脚本不必把三个数字抄两遍。

    **重排参数的优先级是两级**（本层构造参数 ← ``RetrievalQuery.extra``）：
    构造参数是这次装配的口径，而 ``extra`` 里的五个键（``RERANK_OVERRIDE_KEYS``）
    可以逐次覆盖它——键名是封闭清单，清单外的键一律报 ``RerankError``
    （见 ``_resolve_rerank``）。``reranker=None`` 的含义与单路那层一致：
    **用缺省替身**（按 ``settings.retrieval_rerank_model`` 建一个）。

    **它刻意不继承 ``Retriever``**：那条流水线（九步 + 一步重排）与这条
    （十三步 + 一步重排）在"什么时候落阈值、什么时候裁多样性"上顺序**不同**
    ——混合模式必须把多样性放在融合之后（否则两路各自裁一遍，被某一路裁掉的记录
    就没有机会从另一路被救回来）。继承会强迫这两套顺序共用一套方法名，
    而它们的语义差别恰恰是本课要讲的东西。

    它也不认识语料：``_lexical`` 由调用方给出（``build_hybrid_retriever``
    会从向量库现建一份），因此"两路用的是同一批文档"是一件构造期就能说清的事。
    """

    def __init__(
        self,
        vector: Retriever,
        lexical: LexicalIndex,
        *,
        strategy: str | None = None,
        alpha: float | None = None,
        k_rrf: int | None = None,
        weights: Mapping[str, float] | None = None,
        max_per_doc: int | None = None,
        strict_index: bool = False,
        rerank_enabled: bool = False,
        reranker: BaseReranker | None = None,
        rerank_top_n: int | None = None,
        rerank_mode: str | None = None,
        rerank_weight: float | None = None,
        rerank_min_score: float | None = None,
        name: str = DEFAULT_HYBRID_NAME,
    ) -> None:
        if not isinstance(vector, Retriever):
            raise QueryError(
                f"HybridRetriever 的第一个参数必须是 Retriever，"
                f"收到 {type(vector).__name__}。"
                "混合检索的向量一半**复用** day066 的那条流水线，"
                "自己再写一份编码/体检/漂移判定，会让两条路的判据分家。"
            )
        if not isinstance(lexical, LexicalIndex):
            raise QueryError(
                f"HybridRetriever 的第二个参数必须是 LexicalIndex，"
                f"收到 {type(lexical).__name__}。"
                "关键词那一半要一份倒排索引（见 lexical.LexicalIndex.build / from_backend）。"
            )
        self._vector = vector
        self._lexical = lexical
        self._strategy = _resolve_strategy(strategy)
        self._alpha = _resolve_alpha(alpha)
        self._k_rrf = _resolve_k_rrf(k_rrf)
        self._weights = _resolve_weights(weights)
        # 融合参数的校验只有一份实现（``fusion.fuse``）：构造期拿一份**空输入**
        # 先过一遍它，于是"参数写错"在**装配那一刻**就响，而不是等到第一次检索才响
        # ——后者会把一个配置错误伪装成"这次请求碰巧不对"（那是两种完全不同的动作）。
        # 这一次调用同时覆盖了"strategy='rrf' 却给了 weights"这个组合。
        fuse(
            {},
            strategy=self._strategy,
            alpha=self._alpha,
            k_rrf=self._k_rrf,
            weights=self._weights,
        )
        # 直接读向量检索器的三个**配置**字段（构造期定下的纯值，没有任何 I/O）。
        # 走 describe() 会顺带做一次索引体检（count + 清单对账），
        # 而"构造一个检索器"必须是零副作用的（与 routes 的导入期纪律同源）。
        self._doc_id_field = str(getattr(vector, "_doc_id_field", DEFAULT_DOC_ID_FIELD))
        self._vector_max_per_doc = int(getattr(vector, "_max_per_doc", 0))
        if max_per_doc is not None and (
            not isinstance(max_per_doc, int) or isinstance(max_per_doc, bool) or max_per_doc < 0
        ):
            raise QueryError(
                f"max_per_doc 必须是 >= 0 的整数（0 表示不限），收到 {max_per_doc!r}"
            )
        self._max_per_doc = max_per_doc
        # 严格模式是**只增不减**的护栏（与 Retriever 同一句）：构造参数只能把它打开，
        # 不能覆盖 settings 里已经打开的 True，也不能覆盖向量路已经打开的 True。
        self._strict_index = (
            bool(strict_index)
            or bool(settings.retrieval_strict_index)
            or bool(getattr(vector, "_strict_index", False))
        )
        if not isinstance(name, str) or not name.strip():
            raise QueryError(f"name 必须是非空字符串，收到 {name!r}")
        self._name = name.strip()
        # 重排的六个参数（day068）。``rerank_enabled`` 是**只增不减**的护栏
        # （与严格模式同一句）：构造参数只能把它打开，不能覆盖 settings 里
        # 已经打开的 True，也不能覆盖向量路已经打开的 True——那是"这台机器
        # 一律要重排"的部署侧决定。
        self._rerank_enabled = (
            bool(rerank_enabled)
            or bool(settings.retrieval_rerank_enabled)
            or bool(getattr(vector, "_rerank_enabled", False))
        )
        self._reranker = (
            reranker
            if reranker is not None
            else CrossEncoderReranker(name=settings.retrieval_rerank_model)
        )
        self._rerank_top_n = (
            settings.retrieval_rerank_top_n if rerank_top_n is None else rerank_top_n
        )
        self._rerank_mode = settings.retrieval_rerank_mode if rerank_mode is None else rerank_mode
        self._rerank_weight = (
            settings.retrieval_rerank_weight if rerank_weight is None else rerank_weight
        )
        self._rerank_min_score = (
            settings.retrieval_rerank_min_score if rerank_min_score is None else rerank_min_score
        )
        # 参数校验只有一份实现（``rerank_hits``）：拿一份**空输入**先过一遍它，
        # 于是"参数写错"在装配那一刻就响，而不是等到第一次检索才响
        # （与上面 ``fuse({}, ...)`` 的做法逐字同理）。空输入不会调用重排器。
        rerank_hits(
            (),
            "构造期参数校验",
            reranker=self._reranker,
            top_n=self._rerank_top_n,
            mode=self._rerank_mode,
            weight=self._rerank_weight,
            min_score=self._rerank_min_score,
        )

    # ------------------------------------------------------------------ 只读视图

    @property
    def name(self) -> str:
        """这个检索器的名字（报告里回显它）."""
        return self._name

    @property
    def strategy(self) -> str:
        """缺省融合策略（``RetrievalQuery.extra`` 可以逐次覆盖它）."""
        return self._strategy

    @property
    def lexical(self) -> LexicalIndex:
        """关键词索引（``/retrieval/lexical/status`` 直接读它）."""
        return self._lexical

    @property
    def vector(self) -> Retriever:
        """复用的那个向量检索器（它的 ``describe()`` 描述着两路共用的口径）."""
        return self._vector

    @property
    def index_state(self) -> IndexState:
        """当前索引状态（**观测**入口：漂移时照常返回，不抛错）."""
        return self._vector.index_state

    def describe(self) -> dict[str, Any]:
        """这个混合检索器的配置 + 两路的现状（端点直接返回它）."""
        return {
            "name": self._name,
            "strategy": self._strategy,
            "alpha": self._alpha,
            "k_rrf": self._k_rrf,
            "weights": self._effective_weights(),
            "max_per_doc": self._max_per_doc
            if self._max_per_doc is not None
            else self._vector_max_per_doc,
            "strict_index": self._strict_index,
            "doc_id_field": self._doc_id_field,
            "channels": list(CHANNELS),
            # day068 加的一组"重排口径"（与上面几项同一性质：**构造参数**，
            # 而 ``RetrievalQuery.extra`` 可以逐次覆盖其中五个键）。
            "rerank": self._rerank_describe(),
            "vector": self._vector.describe(),
            "lexical": self._lexical.describe(),
        }

    def _rerank_describe(self) -> dict[str, Any]:
        """重排那一组的现状（含重排器自述与"逐次可覆盖"的键清单）.

        ``override_keys`` 出现在这里不是装饰：``extra`` 是唯一一处能让同一次查询
        改变重排参数的地方，而"我能覆盖哪几个键"必须在端点响应里查得到
        （清单外的键会报 ``RerankError``，因此清单本身是一份对外协议）。
        """
        return {
            "enabled": self._rerank_enabled,
            "top_n": self._rerank_top_n,
            "mode": self._rerank_mode,
            "weight": self._rerank_weight,
            "min_score": self._rerank_min_score,
            "override_keys": list(RERANK_OVERRIDE_KEYS),
            "model": self._reranker.describe(),
        }

    # ------------------------------------------------------------------ 检索入口

    def retrieve(self, query: str | RetrievalQuery) -> RetrievalResult:
        """执行一次混合检索（十三步见模块 docstring）."""
        started = time.perf_counter()

        # 第 1 步：规范化查询（沿用向量路那一条：默认值只有一处定义）。
        resolved, notes = self._normalize_query(query)

        # 第 2 步：融合参数（ctor 默认 ← query.extra）。
        strategy, alpha, k_rrf, weights = self._resolve_fusion(resolved, notes)

        # 第 2.5 步：重排参数（同样的两级优先级：ctor 默认 ← query.extra）。
        # 它必须在**取候选之前**解析：extra 里可以把 enabled 打开，
        # 而"这次到底要不要重排"决定了后面那一步做不做。
        rerank_plan = self._resolve_rerank(resolved, notes)

        # 第 3 步：召回深度。**由向量路那唯一一份实现算出**（它同时负责过取倍率、
        # 封顶与"已封顶"的注记），两路共用同一个数字——两路深度不同时，
        # 融合会把"这一路取得浅"读成"这一路没有它"。
        fetch_k = self._vector._fetch_depth(resolved, notes)

        # 第 4 步：索引体检（条数 + 清单漂移），沿用向量路的判据与严格模式语义。
        state = self._vector._inspect_index()
        if state.has_drift:
            self._vector._warn_drift(state)
            notes.append(f"索引漂移 {len(state.drift)} 项：{state.drift[0]}")
        if self._strict_index and state.has_drift:
            raise IndexStateError(
                f"strict_index=True 且索引漂移 {len(state.drift)} 项：{state.drift[0]}\n"
                "严格模式的选择是**拒绝服务**，而不是给出一个可能少召回的结果；"
                "请先重建清单（indexing.manifest.manifest_from_store）或补回缺失记录；"
                "若接受'能跑就先跑'，请关掉 strict_index（默认关闭）。"
            )
        filter_applied = resolved.has_filter

        # 第 5 步：过滤子句**只合成一次**（两路拿到同一个对象，见模块 docstring 的坑）。
        combined = combine_where(resolved.where, resolved.time_range)

        if state.count_store == 0:
            # 空库是合法状态：与 day066 同一口径（返回空结果而不是报错）。
            # 深度已经算过（第 3 步），因此 fetch_k 不是 0——"本来打算取多深"照样可读。
            notes.append("库是空的（count=0）：这是合法状态，直接返回空结果而不报错")
            return RetrievalResult(
                query=resolved,
                fetch_k=fetch_k,
                candidates=0,
                filter_applied=filter_applied,
                empty_reason=EMPTY_REASON_NO_DATA,
                metric=state.metric,
                index_state=state,
                latency_ms=retriever_module._elapsed_ms(started),
                notes=tuple(notes),
                channel_candidates={CHANNEL_VECTOR: 0, CHANNEL_BM25: 0},
                fusion=self._fusion_summary(
                    strategy, alpha, k_rrf, weights,
                    {CHANNEL_VECTOR: 0, CHANNEL_BM25: 0}, (), 0,
                ),
            )

        # 第 7 步（先做向量路，因为它的编码校验会在参数有问题时先响）：
        vector_hits, dropped_threshold = self._vector_channel(
            resolved, state, fetch_k, combined, notes
        )
        if not vector_hits:
            notes.append(
                "向量通道一条都没活下来："
                + (
                    f"min_score={resolved.min_score} 在融合前把它的命中全切了"
                    if dropped_threshold
                    else "库侧这次没有返回任何候选（过滤条件或索引内容）"
                )
                + "——这是**融合要看的信号**：一路被阈值清零与一路本来就没召回，"
                "对融合是两件不同的事（前者降权、后者只能补其他路的权重）。"
            )

        # 第 6 步：关键词路（同一个 where、同一个深度）。
        lexical_result = self._lexical.search(resolved.text, top_k=fetch_k, where=combined)
        lexical_hits = list(lexical_result.hits)
        if lexical_result.is_empty:
            notes.append(
                f"关键词路一条都没召回：{_lexical_empty_reason(self._lexical, lexical_result)}"
                "——这正是 day066 留下的那个钩子：这一路空着，"
                "融合就只能靠向量路，而报告里必须说得出来它为什么空着。"
            )
        else:
            notes.extend(f"关键词路：{note}" for note in lexical_result.notes)
            if lexical_result.missing_terms:
                notes.append(
                    f"关键词路有 {len(lexical_result.missing_terms)} 个查询词元不在词表里"
                    f"（{list(lexical_result.missing_terms)}）：它们是这一路这次"
                    "没有贡献的原因之一（未登录词/纯语义改写），"
                    "而向量路恰好擅长这些——这正是混合检索的互补来源。"
                )

        # 第 9 步：融合（唯一入口，两路的原始分数与名次都在里面变成证据）。
        channel_hits = {CHANNEL_VECTOR: vector_hits, CHANNEL_BM25: lexical_hits}
        fused = fuse(
            channel_hits,
            strategy=strategy,
            alpha=alpha,
            k_rrf=k_rrf,
            weights=weights,
        )
        union_candidates = len({hit.record_id for hit in vector_hits}
                               | {hit.record_id for hit in lexical_hits})

        # 第 10 步：装配（channel = 贡献最大的那一路；channels = 全部证据）。
        lookup = _text_lookup(vector_hits, lexical_hits)
        hits = [self._to_hit(item, lookup) for item in fused]

        # 第 10.5 步：重排（可选）。**必须在多样性之前**——多样性会按 doc_id
        # 砍掉同一文档的其余记录，若先砍后重排，被砍掉的候选就再也回不到
        # 重排器眼前（理由见模块 docstring）。关着重排时这一步原样返回。
        hits, rerank_summary, dropped_rerank = self._apply_rerank(
            hits, resolved, rerank_plan, notes
        )

        # 第 11 步：多样性裁剪（在**融合之后**做，因此两路都有机会把同文档的
        # 其他记录带进来；在融合之前裁会把"这一路裁掉的"直接判死）。
        kept, dropped_diversity = retriever_module._apply_diversity(
            hits,
            resolved.max_per_doc,
            self._doc_id_field,
        )

        # 第 12 步：截断（次序是融合的权威次序；重排开启时它就是重排的权威次序，
        # 因此这里同样只重编号、不重排——重排分不在 ``score`` 里，
        # 按 score 排一遍会把重排的结论原样抹掉）。
        top_k = resolved.top_k if resolved.top_k is not None else self._vector.default_top_k
        final = tuple(
            replace(hit, rank=position) for position, hit in enumerate(kept[:top_k])
        )
        dropped_top_k = len(kept) - len(final)

        # 阈值口径的注记**必须无条件出现**（不许静默地忽略一个用户设过的参数）。
        if resolved.min_score is not None:
            notes.append(
                f"min_score={resolved.min_score} **只作用于向量通道**（在融合前落刀，"
                f"本次切掉 {dropped_threshold} 条）；关键词通道不设阈值——"
                "BM25 的分数没有绝对标度，给它一个数字是假的安全感"
                "（与 settings.retrieval_min_score 缺省 None 的理由同源）。"
            )

        # 第 13 步：诊断与装配。
        empty_reason = retriever_module._diagnose(
            hits=final,
            count_store=state.count_store,
            candidates=union_candidates,
            filter_applied=filter_applied,
            dropped_below_threshold=dropped_threshold,
            dropped_by_diversity=dropped_diversity,
            dropped_by_rerank=dropped_rerank,
        )
        return RetrievalResult(
            query=resolved,
            hits=final,
            fetch_k=fetch_k,
            candidates=union_candidates,
            filter_applied=filter_applied,
            dropped_below_threshold=dropped_threshold,
            dropped_by_diversity=dropped_diversity,
            dropped_by_top_k=dropped_top_k,
            empty_reason=empty_reason,
            metric=state.metric,
            index_state=state,
            latency_ms=retriever_module._elapsed_ms(started),
            notes=tuple(notes),
            channel_candidates={
                CHANNEL_VECTOR: len(vector_hits),
                CHANNEL_BM25: len(lexical_hits),
            },
            fusion=self._fusion_summary(
                strategy, alpha, k_rrf, weights,
                {CHANNEL_VECTOR: len(vector_hits), CHANNEL_BM25: len(lexical_hits)},
                fused,
                union_candidates,
            ),
            rerank=rerank_summary,
            dropped_by_rerank=dropped_rerank,
        )

    def retrieve_many(self, queries: Sequence[str | RetrievalQuery]) -> list[RetrievalResult]:
        """批量检索（逐条调用 ``retrieve``，顺序与入参一致）.

        与 ``Retriever.retrieve_many`` 同样是**刻意的逐条**：每次混合检索的
        两路召回数与融合证据都是独立的，而"这一批总共花了多久"与
        "其中一次为什么某一路没贡献"是两个不同的问题（后者才是要查的那个）。
        """
        return [self.retrieve(item) for item in queries]

    def explain(self, result: RetrievalResult) -> list[str]:
        """在 ``result.explain()`` 之上补**混合模式特有**的五行诊断.

        ``result.explain()`` 的四问（哪一版索引 / 什么过滤 / 为什么少 /
        为什么空）在混合模式下全部照旧；这里补的是那条通道回答不了的五件事：

        ```text
        两路分别召回了几条、最终名单里各贡献了几条   → 只看并集看不出"某一路空着"
        融合用的是哪一组参数                        → "为什么换 alpha 没变化"的第一手证据
        阈值的作用范围（只对向量路）                 → 一个被忽略的参数必须被说出来
        重排的作用范围与参数（day068）               → "为什么配了 top_n 却看不出区别"
        关键词索引的现状（篇数 / 词表 / avgdl / k1,b） → "这一路凭什么空着"的第二种回答
        ```
        """
        lines = list(result.explain())
        lines.append(
            f"两路明细：向量路召回 {result.channel_candidates.get(CHANNEL_VECTOR, 0)} 条、"
            f"关键词路召回 {result.channel_candidates.get(CHANNEL_BM25, 0)} 条；"
            f"最终名单里各贡献 "
            f"{result.fusion.get('contributions', {}).get(CHANNEL_VECTOR, 0)} / "
            f"{result.fusion.get('contributions', {}).get(CHANNEL_BM25, 0)} 条"
            f"（并集 {result.candidates} 条 = 两路之和减去被两路同时命中的 "
            f"{result.fusion.get('deduped', 0)} 条）"
        )
        lines.append(
            f"融合：策略 {result.fusion.get('strategy', self._strategy)}"
            f"，参数 {result.fusion.get('params', {})}"
            f"（缺省来自 HybridRetriever(strategy={self._strategy!r}, "
            f"alpha={self._alpha}, k_rrf={self._k_rrf})）"
        )
        lines.append(
            "阈值口径：min_score 只在向量通道、只在融合前落一次刀；"
            "关键词通道**不设阈值**（BM25 的分数没有绝对标度，"
            "给它一个数字是假的安全感）——因此 '关键词路为什么返回了一条低分记录' "
            "这个问题的答案是'BM25 的低分不等于不相关，只等于字面重叠少'"
        )
        lines.append(
            f"重排口径：重排在**融合之后、多样性之前**（第 10.5 步）——"
            f"本层默认 enabled={self._rerank_enabled}、只对前 {self._rerank_top_n} 条打分、"
            f"模式 {self._rerank_mode!r}、权重 {self._rerank_weight}"
            f"（可用 RetrievalQuery.extra 逐次覆盖：{'、'.join(RERANK_OVERRIDE_KEYS)}）；"
            f"关着的时候一个字段都不会被改写（结果与 day067 逐位相同）"
        )
        lines.append(f"关键词索引：{self._lexical.summary_line()}")
        lines.append(
            f"分组口径：多样性按 metadata[{self._doc_id_field!r}] 分组"
            f"（在**融合之后**裁剪，因此两路的证据都已经在名单里）；"
            f"本检索器的名字 {self._name!r}"
        )
        # 字段拼写检查：与 day066 的 ``Retriever.explain`` 是**同一份判据**（它要读库，
        # 因此只能由持有后端的那一层给出）。混合模式下它比单路更要紧：过滤条件要在
        # **两路**都落刀，一个拼错的字段名会让两路同时空掉，而"两路都空"
        # 很容易被读成"语料里没有相关内容"——那时该改的是字段名，不是语料。
        if result.query.where:
            known = self._vector._metadata_fields()
            unknown = unknown_filter_fields(result.query.where, known)
            if unknown:
                lines.append(
                    f"字段拼写检查：{unknown} 在本次库里从未出现过"
                    f"（库里出现过的字段：{sorted(known)}）——过滤条件的字段名很可能写错了。"
                    "注意：它只能发现「整个库里都没有这个字段」，发现不了「值写错」。"
                    "关键词路与向量路用的是**同一份**过滤语义，因此拼错字段时两路会同时空掉。"
                )
            else:
                lines.append(
                    "字段拼写检查：过滤用到的字段在库里都出现过"
                    "（这只证明字段名对，不能证明值对——值域是业务知识）"
                )
        return lines

    # ------------------------------------------------------------------ 十三步实现

    def _normalize_query(
        self, query: str | RetrievalQuery
    ) -> tuple[RetrievalQuery, list[str]]:
        """第 1 步：沿用向量路的规范化，再把 ``max_per_doc`` 换成本层的口径.

        为什么 ``max_per_doc`` 要单独收一次：向量路的 ``_normalize_query`` 会把
        ``None`` 填成**它自己的**默认值，而混合模式的多样性是在融合之后做的
        （见模块 docstring 第 11 步），因此这里的"这次到底允许每个文档几条"
        必须由本层决定：``HybridRetriever(max_per_doc=...)`` 优先，
        否则沿用向量路的配置（"两路共用同一份口径"这句话在这里也成立）。
        """
        raw = query if isinstance(query, RetrievalQuery) else RetrievalQuery(text=query)
        resolved, notes = self._vector._normalize_query(raw)
        max_per_doc = raw.max_per_doc
        if max_per_doc is None:
            max_per_doc = self._max_per_doc
        if max_per_doc is None:
            # 向量路那 0 表示"不限"，而 RetrievalQuery 里 None 才是"不限"（0 会被拒）。
            max_per_doc = self._vector_max_per_doc or None
        return replace(resolved, max_per_doc=max_per_doc), notes

    def _resolve_fusion(
        self,
        query: RetrievalQuery,
        notes: list[str],
    ) -> tuple[str, float, int, Mapping[str, float] | None]:
        """第 2 步：融合参数 = 构造默认 ← ``RetrievalQuery.extra``（day066 预留的槽）.

        键名是**封闭清单**（``fusion.FUSION_OVERRIDE_KEYS``）：多一个键就报
        ``FusionError`` 并列出合法取值。理由与 ``RagPipeline`` 的
        ``OVERRIDE_KEYS`` 完全相同——静默忽略一个覆盖参数会让调用方以为它生效了
        （"我把 ``alpha`` 拼成 ``alhpa``，结果看起来只是没有变化"）。

        参数来自 ``extra`` 时**必须进 notes**：一次"这次用的是哪组参数"的改写
        必须可见，否则同一份查询在两个调用点会给出不同结果而无法解释。

        **"不认识的键"按两份清单的并集判定**（day068 补上）。``extra`` 是**一个**字典，
        而融合层与重排层各自守着一份封闭清单（``FUSION_OVERRIDE_KEYS`` /
        ``RERANK_OVERRIDE_KEYS``，两者不重叠）。只按自己那份判会有一个安静的后果：
        ``extra={"mode": "blend"}`` 在这一层被拒，于是**重排的覆盖参数永远用不上**
        （"我明明在 extra 里写了 mode"会变成一个查不出病因的现象）。
        并集判定仍然拦得住真正的错键（``"alhpa"`` / ``"weigth"``），
        "静默忽略一个覆盖参数"这条纪律一个字都没有放松。
        """
        extra = dict(query.extra)
        known = set(FUSION_OVERRIDE_KEYS) | set(RERANK_OVERRIDE_KEYS)
        unknown = sorted(set(extra) - known)
        if unknown:
            raise FusionError(
                f"RetrievalQuery.extra 里有本层不认识的键 {unknown}："
                f"融合层只认 {'、'.join(FUSION_OVERRIDE_KEYS)}、"
                f"重排层只认 {'、'.join(RERANK_OVERRIDE_KEYS)}。"
                "静默忽略一个覆盖参数会让调用方以为它生效了——请核对键名。"
            )
        strategy = extra.get("strategy", self._strategy)
        alpha = extra.get("alpha", self._alpha)
        k_rrf = extra.get("k_rrf", self._k_rrf)
        weights = extra.get("weights", self._weights)
        # 参数校验只有一份实现（``fusion.fuse``）。这里拿一份**空输入**先过一遍它，
        # 于是"库是空的"那条早退路径也照样会在参数错误上响——否则一个写错的
        # strategy 会被"库恰好为空"掩盖掉，而在库有数据时才突然冒出来。
        fuse({}, strategy=strategy, alpha=alpha, k_rrf=k_rrf, weights=weights)
        if extra:
            applied = sorted(extra)
            notes.append(
                f"融合参数来自 RetrievalQuery.extra（本层默认 strategy={self._strategy!r}、"
                f"alpha={self._alpha}、k_rrf={self._k_rrf}）：本次覆盖了 {'、'.join(applied)}"
            )
        return strategy, alpha, k_rrf, weights  # type: ignore[return-value]

    def _resolve_rerank(self, query: RetrievalQuery, notes: list[str]) -> _RerankPlan:
        """第 2.5 步：重排参数 = 构造默认 ← ``RetrievalQuery.extra``.

        键名是**封闭清单**（``rerank.RERANK_OVERRIDE_KEYS``）：多一个键就报
        ``RerankError``。理由与 ``_resolve_fusion`` 逐字相同——静默忽略一个覆盖参数
        会让调用方以为它生效了（"我把 ``mode`` 拼成 ``mod``，结果看起来只是没有变化"）。
        两个封闭清单**各自管自己那一层**：``extra`` 是同一个字典，
        因此这里的清单必须含 ``strategy/alpha`` 之外的键，而那五个键不能再被融合层认领。

        参数来自 ``extra`` 时必须进 ``notes``（一次"这次用的是哪组参数"的改写必须可见）；
        ``enabled`` 尤其要写清楚，因为它是唯一一个能**逐次打开/关掉整层能力**的键。

        ``enabled`` 的取值必须是真布尔：``"yes"`` / ``1`` 这类"看起来是真的"写法
        一律报错（``bool("no")`` 是 True——一个会让人把"关闭"写成"打开"的坑）。
        """
        extra = dict(query.extra)
        applied = sorted(set(extra) & set(RERANK_OVERRIDE_KEYS))
        enabled_raw = extra.get("enabled", self._rerank_enabled)
        if not isinstance(enabled_raw, bool):
            raise RerankError(
                f"RetrievalQuery.extra['enabled'] 必须是布尔值，收到 "
                f"{type(enabled_raw).__name__}（{enabled_raw!r}）。"
                "它的值是 True/False，不是字符串或数字——"
                "bool('no') 是 True，那会把'这次先关掉重排'静默地变成'打开'。"
            )
        plan = _RerankPlan(
            enabled=enabled_raw,
            reranker=self._reranker,
            top_n=extra.get("top_n", self._rerank_top_n),
            mode=extra.get("mode", self._rerank_mode),
            weight=extra.get("weight", self._rerank_weight),
            min_score=self._rerank_min_score,
            model=extra.get("model"),
        )
        # 参数校验只有一份实现（``rerank_hits``）：拿一份**空输入**先过一遍它，
        # 于是"库是空的"那条早退路径也照样会在参数错误上响——否则一个写错的
        # mode 会被"库恰好为空"掩盖掉，而在库有数据时才突然冒出来。
        rerank_hits(
            (),
            "extra 参数校验",
            reranker=plan.reranker,
            top_n=plan.top_n,
            mode=plan.mode,
            weight=plan.weight,
            min_score=plan.min_score,
            model=plan.model,
        )
        if applied:
            if plan.enabled:
                notes.append(
                    f"重排参数来自 RetrievalQuery.extra（本层默认 enabled="
                    f"{self._rerank_enabled}、top_n={self._rerank_top_n}、"
                    f"mode={self._rerank_mode!r}、weight={self._rerank_weight}）："
                    f"本次覆盖了 {'、'.join(applied)}"
                )
            else:
                # 一个**真的没生效**的覆盖参数必须被说出来（与"min_score 只作用于
                # 向量通道"那条注记同源）：否则"我写了 top_n 却没变化"会变成一个
                # 反复出现的疑问，而答案其实只有一条——重排没开。
                notes.append(
                    f"重排未启用（enabled=False），因此 extra 里的 "
                    f"{'、'.join(applied)} 这次**没有作用**："
                    "要在这一次打开重排请同时给 enabled=True"
                    "（整层开关也可以由 HybridRetriever(rerank_enabled=True) "
                    "或 settings.retrieval_rerank_enabled 打开）"
                )
        return plan

    def _apply_rerank(
        self,
        hits: list[RetrievalHit],
        query: RetrievalQuery,
        plan: _RerankPlan,
        notes: list[str],
    ) -> tuple[list[RetrievalHit], dict[str, Any], int]:
        """第 10.5 步：重排（关闭时**原样返回**，一个字段都不改写）.

        与单路那条流水线的第 6.5 步是同一个动作，因此**共用模块级的两个原语**：
        ``rerank_hits``（纯函数，做窗口与排序）与 ``retriever._rebuild_hits``
        （把结果写回 ``RetrievalHit``）。这里只负责"把本层的参数递进去"——
        两份各写一遍的写回逻辑一定会分家，而分家不会报错，只会让报告少一列。

        传进重排的是**融合之后的命中**（``_to_hit`` 已经装配好的那一份）：
        顺序是融合的权威次序，``channels`` 带着多路证据，``text`` 来自两路的正文表。
        重排只读它的 id / 分数 / 名次 / 正文 / 通道，不会碰任何一路的原始证据。
        """
        if not plan.enabled:
            return hits, {}, 0
        result = rerank_hits(
            hits,
            query.text,
            reranker=plan.reranker,
            top_n=plan.top_n,
            mode=plan.mode,
            weight=plan.weight,
            min_score=plan.min_score,
            model=plan.model,
        )
        notes.extend(f"重排：{note}" for note in result.notes)
        return (
            retriever_module._rebuild_hits(hits, result),
            result.to_summary(),
            result.dropped_by_min_score,
        )

    def _vector_channel(
        self,
        query: RetrievalQuery,
        state: IndexState,
        fetch_k: int,
        combined: dict[str, Any] | None,
        notes: list[str],
    ) -> tuple[list[RetrievalHit], int]:
        """第 7、8 步：向量路取候选 + 阈值落刀（**只落这一路**）.

        阈值在这里落刀而不是交给 ``backend.query(min_score=...)``：与 day066
        第 6 步同一条理由——"切掉了几条"这个数字必须能被报告，而库侧切完之后
        那个数字拿不到（``candidates`` 会给过滤后的总数，而不是切掉几条）。
        """
        vector = self._vector._encode_query(query.text)
        notes.extend(self._vector._encoder_notes(state, len(vector)))
        search = self._vector._backend.query(vector, fetch_k, where=combined)
        if combined and search.candidates == 0:
            notes.append("过滤条件把向量路的候选筛成了 0 条：请检查字段名与取值，或放宽条件")
        hits = [self._vector._to_hit(item) for item in search.hits]
        kept, dropped = retriever_module._apply_threshold(hits, query.min_score)
        return kept, dropped

    def _to_hit(
        self,
        item: FusedHit,
        lookup: Mapping[str, tuple[str, dict[str, Any]]],
    ) -> RetrievalHit:
        """第 10 步：``FusedHit`` → ``RetrievalHit``（补上正文、元数据与通道归属）.

        ``channel`` 取**贡献最大的那一路**；并列时按 ``CHANNEL_VECTOR`` 优先
        （与 ``fusion._ordered_channels`` 是同一条规则，因此
        ``channels[0] == channel`` 永远成立——它们本来就是同一个问题的两个问法）。
        """
        text, metadata = lookup.get(item.record_id, ("", {}))
        return RetrievalHit(
            record_id=item.record_id,
            score=item.score,
            rank=item.rank,
            text=text,
            metadata=dict(metadata),
            channel=_dominant_channel(item.contributions),
            channels=item.channels,
        )

    def _effective_weights(self) -> dict[str, float]:
        """这次实际会用的权重表（rrf 下是空字典——"RRF 不加权"必须看得出来）."""
        if self._strategy != FUSION_WEIGHTED:
            return {}
        if self._weights is not None:
            return {name: float(value) for name, value in self._weights.items()}
        return {CHANNEL_VECTOR: self._alpha, CHANNEL_BM25: 1.0 - self._alpha}

    def _fusion_summary(
        self,
        strategy: str,
        alpha: float,
        k_rrf: int,
        weights: Mapping[str, float] | None,
        recall: Mapping[str, int],
        fused: Sequence[FusedHit],
        union_candidates: int,
    ) -> dict[str, Any]:
        """融合摘要："为什么这条排第一"的全部输入（进 ``RetrievalResult.fusion``）.

        ``deduped`` 的定义值得记牢：``两路召回之和 - 并集``。它不是"丢掉了多少"，
        而是"**有多少条被两路同时命中**"——融合里最强的那类证据。
        取这个名字是为了与"去重"这个说法对齐（去重的结果是名单变短），
        但语义上它是"合并了多少条重复证据"。
        """
        params: dict[str, Any] = (
            {"k_rrf": k_rrf} if strategy == FUSION_RRF else {"alpha": alpha}
        )
        if strategy == FUSION_WEIGHTED and weights is not None:
            params["weights"] = {name: float(value) for name, value in weights.items()}
        return {
            "strategy": strategy,
            "params": params,
            "recall": {name: int(count) for name, count in sorted(recall.items())},
            "contributions": contribution_counts(fused),
            "fused": len(fused),
            "deduped": max(0, int(sum(recall.values())) - int(union_candidates)),
        }


# --------------------------------------------------------------------------- #
# 小工具（三个纯函数，各自回答一个具体问题）
# --------------------------------------------------------------------------- #


def _dominant_channel(contributions: Mapping[str, float]) -> str:
    """贡献最大的那一路（并列时向量优先；没有证据时回落向量路）.

    为什么需要"归属"：``RetrievalHit.channel`` 是 day066 就定下的形状，
    而混合模式下一条记录可能同时来自两路——必须给出**一个**名字（它是"这条
    主要是谁找回来的"），同时用 ``channels`` 保留"另一路也召回了它"这个证据。
    两处都说同一个名字会让最强证据消失；两处都不说会让报告无法排序原因。
    """
    if not contributions:
        return CHANNEL_VECTOR
    name, _value = max(
        contributions.items(),
        key=lambda pair: (pair[1], pair[0] == CHANNEL_VECTOR),
    )
    return name


def _text_lookup(
    vector_hits: Sequence[RetrievalHit],
    lexical_hits: Sequence[Any],
) -> dict[str, tuple[str, dict[str, Any]]]:
    """``record_id → (正文, 元数据)``：融合后的名单要能显示正文.

    两路的正文是**同一份**（都来自同一个库），因此这里只是"从哪一侧取"的问题。
    先放向量路（它带的是 ``RetrievalHit``，字段更全），再用关键词路补上
    向量路没有的那几条（``setdefault``：不会覆盖已有的更全的那一份）。
    """
    lookup: dict[str, tuple[str, dict[str, Any]]] = {}
    for hit in vector_hits:
        lookup[str(hit.record_id)] = (hit.text, dict(hit.metadata))
    for hit in lexical_hits:
        lookup.setdefault(str(hit.record_id), (str(hit.text), dict(hit.metadata)))
    return lookup


def _lexical_empty_reason(index: LexicalIndex, result: Any) -> str:
    """关键词路空着的原因（**四种成因各自一句话**，与 ``EMPTY_REASONS`` 同一纪律）.

    四种成因的处置动作完全不同，因此不能合成一句"关键词路没有结果"：

    ```text
    索引是空的（0 篇）        → 先建关键词索引（from_backend）
    被 where 筛成 0 条        → 放宽过滤条件（这一路的过滤与向量路是同一份）
    查询词元与词表交集为空    → 纯语义改写 / 未登录词——**这是最正常的一种**，
                              向量路正好擅长它，混合检索的价值在这里体现
    词元都认识但没有一篇全中  → 语料里确实没有同时提到这几个词的内容（要去看语料）
    ```
    """
    if index.count == 0:
        return "关键词索引是空的（0 篇文档），请先重建关键词索引"
    if result.candidates == 0:
        return "where 把关键词路的候选筛成了 0 条（与向量路是同一份过滤语义）"
    if not result.matched_terms:
        return (
            f"查询词元与词表交集为空（纯语义改写常见）——"
            f"missing_terms={list(result.missing_terms)}"
        )
    return "查询词元都能对上词表，但没有任何一篇同时包含它们（语料里没有这段内容）"


# --------------------------------------------------------------------------- #
# 配置解析（构造期一次算清，之后不再看 settings）
# --------------------------------------------------------------------------- #


def _resolve_strategy(value: str | None) -> str:
    """``strategy``：``None`` → ``settings.retrieval_hybrid_strategy``."""
    resolved = settings.retrieval_hybrid_strategy if value is None else value
    if not isinstance(resolved, str) or not resolved.strip():
        raise FusionError(
            f"strategy 必须是非空字符串，收到 {resolved!r}。"
            "可用策略：rrf（只看名次）/ weighted（先归一化再加权）"
        )
    return resolved.strip()


def _resolve_alpha(value: float | None) -> float:
    """``alpha``：``None`` → ``settings.retrieval_hybrid_alpha``（范围校验交给 fusion）."""
    resolved = settings.retrieval_hybrid_alpha if value is None else value
    if isinstance(resolved, bool) or not isinstance(resolved, (int, float)):
        raise FusionError(f"alpha 必须是数字，收到 {type(resolved).__name__}")
    return float(resolved)


def _resolve_k_rrf(value: int | None) -> int:
    """``k_rrf``：``None`` → ``settings.retrieval_hybrid_rrf_k``."""
    resolved = settings.retrieval_hybrid_rrf_k if value is None else value
    if not isinstance(resolved, int) or isinstance(resolved, bool):
        raise FusionError(f"k_rrf 必须是整数，收到 {type(resolved).__name__}")
    return resolved


def _resolve_weights(
    value: Mapping[str, float] | None,
) -> dict[str, float] | None:
    """``weights``：``None`` 保持 ``None``（weighted 时按 alpha 展开）；否则存一份副本.

    这里把五项校验都做在构造期（通道名、类型、有限性、非负、扁平化），
    而 ``fusion._resolve_weights`` 在每次融合时还会再校验一次"覆盖了每个通道"——
    两处的分工是"构造期管参数本身，运行期管这次的通道表"。

    存副本而不是持有调用方的字典：它可能在构造之后被改，
    而"构造参数不该在运行期被外部改掉"是 frozen 数据报的教条在可变对象上的那一半。
    """
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise FusionError(
            f"weights 必须是 {{通道名: 权重}} 这样的映射，收到 {type(value).__name__}"
        )
    resolved: dict[str, float] = {}
    for name, weight in value.items():
        if not isinstance(name, str) or not name.strip():
            raise FusionError(f"weights 的键必须是非空通道名，收到 {name!r}")
        key = name.strip()
        if key not in CHANNELS:
            raise FusionError(
                f"weights 里有未知通道名 {key!r}：本层认识的通道是 {'、'.join(CHANNELS)}。"
                "拼错的名字不会报错、只会让真正的通道缺权重——那一路的证据随后静默消失。"
            )
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise FusionError(
                f"weights[{key!r}] 必须是数字，收到 {type(weight).__name__}"
            )
        if not math.isfinite(float(weight)):
            raise FusionError(f"weights[{key!r}] 必须是有限数，收到 {weight!r}")
        if float(weight) < 0.0:
            raise FusionError(
                f"weights[{key!r}]={weight} 是负数：负权重意味着"
                "'这一路召回得越好、最终排名越差'——那不是一个可以用权重表达的需求。"
            )
        resolved[key] = float(weight)
    return resolved


# --------------------------------------------------------------------------- #
# 装配入口
# --------------------------------------------------------------------------- #


def hybrid_enabled() -> bool:
    """混合检索是否在这个进程里被启用（``settings.retrieval_hybrid_enabled``）.

    默认 ``False``：混合检索会**改变结果集合**（关键词路会捞进向量路没召回的记录，
    那正是它存在的意义），因此"换一层检索"必须是一次显式决定，
    而不是升级之后自动发生的事。端点层用它决定要不要拒绝请求
    （而不是静默退回单路——那会让"我明明启用了却看不出区别"变成一个谜）。
    """
    return bool(settings.retrieval_hybrid_enabled)


def build_hybrid_retriever(
    backend: VectorBackend,
    embedding: EmbeddingProvider,
    *,
    manifest: IndexManifest | None = None,
    lexical: LexicalIndex | None = None,
    params: BM25Params | None = None,
    strategy: str | None = None,
    alpha: float | None = None,
    k_rrf: int | None = None,
    weights: Mapping[str, float] | None = None,
    max_per_doc: int | None = None,
    strict_index: bool | None = None,
    reranker: BaseReranker | None = None,
    rerank_enabled: bool | None = None,
    rerank_top_n: int | None = None,
    rerank_mode: str | None = None,
    rerank_weight: float | None = None,
    rerank_min_score: float | None = None,
    name: str | None = None,
) -> HybridRetriever:
    """按 ``settings`` 装配一个混合检索器（端点与演示脚本的统一入口）.

    三处刻意的决定，都与 ``build_retriever`` 是同一条纪律：

    ```text
    1. 向量一半走 build_retriever    时间字段与 strict_index 由那条装配路径
                                    显式指向 settings（理由见 build_retriever）
    2. 关键词一半**现建**            lexical=None 时从 backend 现建一份
                                    （BM25 的两个参数取 settings.retrieval_bm25_*）；
                                    语料大时请自己建一次并注入（见下）
    3. 注意关键词索引是**一次性快照** 它不会跟着向量库的后续写入自动更新——
                                    混合检索的两个一半来自同一次读取，
                                    而"关键词索引落后于向量库"会让关键词路静默少召回。
                                    因此：库变了就重建（from_backend），
                                    或直接注入一个你自己管生命周期的 LexicalIndex。
    ```

    第 3 条不是缺陷而是本层的边界声明（``LexicalIndex`` 刻意不持久化、
    不做增量，理由写在那个类的 docstring 里）。把它写在这里，是因为
    "装配函数每次现建"很容易被读成"每次检索都会重建"——**不是**：
    这个函数被调用一次，索引就只建一次；"每次请求现建"发生在
    ``api.routes.retrieval_hybrid`` 那条路径上，那里也把这件事写进了注释。

    day068 追加的重排那一组与前两组同一条纪律：``None`` 一律去读
    ``settings.retrieval_rerank_*``（重排**是否生效**由 ``rerank_enabled``
    与 ``settings.retrieval_rerank_enabled`` 只增不减地决定），
    而 ``reranker=None`` 表示"用缺省替身"——真实部署要接自己的重排模型时，
    在这里注入一个 ``BaseReranker`` 实例即可，**不需要**改本项以外的配置。
    """
    vector = build_retriever(backend, embedding, manifest=manifest)
    resolved_lexical = lexical if lexical is not None else LexicalIndex.from_backend(
        backend,
        params=params
        if params is not None
        else BM25Params(k1=settings.retrieval_bm25_k1, b=settings.retrieval_bm25_b),
    )
    return HybridRetriever(
        vector,
        resolved_lexical,
        strategy=strategy if strategy is not None else settings.retrieval_hybrid_strategy,
        alpha=alpha if alpha is not None else settings.retrieval_hybrid_alpha,
        k_rrf=k_rrf if k_rrf is not None else settings.retrieval_hybrid_rrf_k,
        weights=weights,
        max_per_doc=max_per_doc,
        strict_index=(
            strict_index if strict_index is not None else settings.retrieval_strict_index
        ),
        reranker=reranker,
        rerank_enabled=(
            settings.retrieval_rerank_enabled if rerank_enabled is None else rerank_enabled
        ),
        rerank_top_n=(
            settings.retrieval_rerank_top_n if rerank_top_n is None else rerank_top_n
        ),
        rerank_mode=settings.retrieval_rerank_mode if rerank_mode is None else rerank_mode,
        rerank_weight=(
            settings.retrieval_rerank_weight if rerank_weight is None else rerank_weight
        ),
        rerank_min_score=(
            settings.retrieval_rerank_min_score if rerank_min_score is None else rerank_min_score
        ),
        name=name if name is not None else DEFAULT_HYBRID_NAME,
    )


__all__ = [
    "DEFAULT_HYBRID_NAME",
    "HybridRetriever",
    "build_hybrid_retriever",
    "hybrid_enabled",
]
