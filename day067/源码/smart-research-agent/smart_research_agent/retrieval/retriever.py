"""检索器：**给一句话，取回哪几条、为什么是这几条、为什么只有这几条**（M6-D5）.

## 检索器与向量库的分工（本模块最重要的一句话）

```text
向量库    给一个向量，谁最近                  → 排序问题
检索器    给一句话，取回哪几条、               → 决策问题
          为什么是这几条、为什么只有这几条
```

向量库**答不了**后面那半句话，因为它的输入里没有"一句话"：编码器在它之外，
阈值、多样性、深度、路由都在它之外。把这三件事塞进后端会得到一个
每个后端都要各写一遍的"检索策略"——而策略是三行配置的事，
不该变成三个实现里各一份、互相对不齐的代码（见 ``vectorstore.base`` 的边界声明）。

反过来，检索器也**不碰向量**：它不认识度量公式、不做归一化、不改排序键，
一切"谁更近"的判断仍然是 ``vectorstore.metrics`` 的 ``score``（越大越近）。

## 固定的九步（教程会逐条讲，代码里的注释逐条对应）

```text
1. 规范化查询      str → RetrievalQuery；None 字段填构造参数 / settings
2. 索引体检        库空 → 直接返回 no_data（空库是合法状态，不报错）；
                   有清单 → compare_with_store 得到 drift（严格模式直接抛）
3. 算召回深度      fetch_k = max(top_k, ceil(top_k × multiplier))，封顶 MAX_FETCH_K
4. 编码查询        embed(text) + 三项校验（维度 / 有限 / 非零）
5. 库侧过滤        combine_where(where, time_range) 交给 backend.query
6. 阈值落刀        在本层按 min_score 过滤并记「切掉了几条」
7. 多样性裁剪      按 doc_id 分组，每组留前 max_per_doc 条，其余记「挤掉了几条」
8. 截断到 top_k    记「截断丢掉了几条」
9. 诊断与装配      定 empty_reason、算延迟、收集 notes
```

### 第 6 步为什么必须在**本层**落刀

``VectorBackend.query`` 也接受 ``min_score``，但检索器**故意不传**它：

```text
交给库过滤   candidates=5、hits=2      → "切掉了几条"这个数字拿不到
在本层落刀   candidates=5、dropped=3   → 数字拿得到（正是 day067 融合要用的）
```

day067 的融合需要按通道判断"**这一路是不是一条都没活下来**"——一路被阈值清零
与一路本来就没召回，对融合是两件完全不同的事（前者降权、后者可以补其他路的权重）。
语义与 ``VectorBackend.query(min_score=...)`` 完全一致：**同一个 ``>=``、
同一个分数口径**（越大越近），因此两种写法的结果集合逐条相同——
差别只在"这个数字有没有被记下来"。

### 排序稳定性：同分按 ``record_id`` 升序

与 ``vectorstore.types.sort_hits`` 同一条纪律，并且**在本层再执行一次**
（第 8 步之前统一排序并重排名次）。理由不是不信任后端，而是这一层的上限
是"任何实现 ``VectorBackend`` 的后端"：第三方的后端可能没读过那段注释，
而后果是"同一次查询两次运行给出不同的 top-1"——**不确定性不来自算法，
来自没写下来的排序规则**，所以把它写在这一层。
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from smart_research_agent.config import settings
from smart_research_agent.indexing.manifest import compare_with_store
from smart_research_agent.indexing.types import IndexManifest
from smart_research_agent.llm.embedding import EmbeddingProvider
from smart_research_agent.retrieval.errors import IndexStateError, QueryError
from smart_research_agent.retrieval.filters import (
    DEFAULT_TIME_FIELD,
    combine_where,
    unknown_filter_fields,
)
from smart_research_agent.retrieval.types import (
    CHANNEL_VECTOR,
    DEFAULT_DOC_ID_FIELD,
    EMPTY_REASON_BELOW_THRESHOLD,
    EMPTY_REASON_DIVERSITY,
    EMPTY_REASON_FILTERED_OUT,
    EMPTY_REASON_NO_DATA,
    EMPTY_REASON_NONE,
    MAX_FETCH_K,
    IndexState,
    RetrievalHit,
    RetrievalQuery,
    RetrievalResult,
)
from smart_research_agent.utils.logger import get_logger
from smart_research_agent.vectorstore.base import VectorBackend
from smart_research_agent.vectorstore.types import SearchHit

logger = get_logger(__name__)


class Retriever:
    """单路向量检索器：Top-K + 深度 + 过滤 + 阈值 + 多样性 + 空结果诊断 + 漂移告警.

    构造参数分三组，**每组各自决定一件可以被单独讨论的事**：

    ```text
    深度类    default_top_k / fetch_multiplier      "取多深"
    决策类    min_score / max_per_doc               "留下哪些"
    口径类    doc_id_field / time_field / manifest  "按什么口径认文档与时间"
    ```

    ``None`` 在这些参数上有明确含义：**"没指定，去读 settings"**
    （``retrieval_top_k`` / ``retrieval_fetch_multiplier`` / ``retrieval_min_score`` /
    ``retrieval_max_per_doc``）。于是"项目默认值"只有一处定义，
    而端点与演示脚本不需要把 8 个数字抄两遍。

    ``max_per_doc`` 的 ``0`` 表示**不限**（与 settings 的口径一致）；
    而查询里不允许写 0（见 ``RetrievalQuery``）——"最多 0 条"不是一个请求。
    """

    def __init__(
        self,
        backend: VectorBackend,
        embedding: EmbeddingProvider,
        *,
        default_top_k: int | None = None,
        fetch_multiplier: int | None = None,
        min_score: float | None = None,
        max_per_doc: int | None = None,
        doc_id_field: str = DEFAULT_DOC_ID_FIELD,
        time_field: str = DEFAULT_TIME_FIELD,
        manifest: IndexManifest | None = None,
        name: str = "default",
        strict_index: bool = False,
    ) -> None:
        self._backend = backend
        self._embedding = embedding
        self._default_top_k = _resolve_top_k(default_top_k)
        self._fetch_multiplier = _resolve_multiplier(fetch_multiplier)
        self._min_score = _resolve_min_score(min_score)
        self._max_per_doc = _resolve_max_per_doc(max_per_doc)
        self._doc_id_field = _resolve_field_name("doc_id_field", doc_id_field)
        # time_field 的默认值写成字面量（签名要求），因此它与 settings 撞在同一个值上：
        # "等于默认值"时按"没指定"处理，去读 settings.retrieval_time_field。
        # 这一条写在 docstring 里而不是藏着：不然"我明明传了 created_at"这种疑问没法解释。
        self._time_field = (
            _resolve_field_name("time_field", settings.retrieval_time_field)
            if time_field == DEFAULT_TIME_FIELD
            else _resolve_field_name("time_field", time_field)
        )
        self._manifest = manifest
        self._name = _resolve_field_name("name", name)
        # 严格模式是**只增不减**的护栏：构造参数只能把它打开，
        # 不能覆盖 settings 里已经打开的 True——它是"这台机器一律不接受漂移"的运维开关。
        self._strict_index = bool(strict_index) or bool(settings.retrieval_strict_index)

    # ------------------------------------------------------------------ 只读视图

    @property
    def name(self) -> str:
        """这个检索器的名字（多索引路由与报告里回显它）."""
        return self._name

    @property
    def default_top_k(self) -> int:
        """默认的返回条数上限."""
        return self._default_top_k

    @property
    def index_state(self) -> IndexState:
        """当前索引状态（``/retrieval/status`` 用它；**只报告，不因漂移抛错**）.

        与 ``retrieve`` 的差别是刻意的：``index_state`` 是**观测**入口
        （运维想看看"现在漂成什么样了"），因此它在漂移时照常返回状态；
        而 ``retrieve`` 是**取数**入口，``strict_index=True`` 时必须拒绝给出
        可能不完整的答案（见 ``IndexState`` 的取舍说明）。
        """
        return self._inspect_index()

    def describe(self) -> dict[str, Any]:
        """这个检索器的配置 + 索引现状（端点直接返回它）."""
        return {
            "name": self._name,
            "top_k": self._default_top_k,
            "fetch_multiplier": self._fetch_multiplier,
            "min_score": self._min_score,
            "max_per_doc": self._max_per_doc,
            "doc_id_field": self._doc_id_field,
            "time_field": self._time_field,
            "index": self.index_state.to_dict(),
        }

    # ------------------------------------------------------------------ 检索入口

    def retrieve(self, query: str | RetrievalQuery) -> RetrievalResult:
        """执行一次检索（九步流水线见模块 docstring）."""
        started = time.perf_counter()

        # 第 1 步：规范化查询。str → RetrievalQuery；None 字段填默认值。
        resolved, notes = self._normalize_query(query)

        # 第 3 步：算召回深度。**在体检之前算**，这样"库是空的"这条早退路径
        # 也能报告"本来打算取多深"（一个空结果里的 fetch_k=0 会让人以为
        # 这次没设深度，而实际上深度是算过的）。
        fetch_k = self._fetch_depth(resolved, notes)

        # 第 2 步：索引体检（条数 + 清单漂移）。
        state = self._inspect_index()
        if state.has_drift:
            self._warn_drift(state)
            notes.append(f"索引漂移 {len(state.drift)} 项：{state.drift[0]}")
        if self._strict_index and state.has_drift:
            raise IndexStateError(
                f"strict_index=True 且索引漂移 {len(state.drift)} 项：{state.drift[0]}\n"
                "严格模式的选择是**拒绝服务**，而不是给出一个可能少召回的结果；"
                "请先重建清单（indexing.manifest.manifest_from_store）或补回缺失记录；"
                "若接受'能跑就先跑'，请关掉 strict_index（默认关闭）。"
            )
        filter_applied = _filter_applied(resolved)

        # 第 5 步（子句部分）：合成过滤条件。**必须排在"库是空的"这条早退之前**：
        # 它只是拼字典，而"where 与 time_range 撞在同一个字段"这种调用方错误
        # 在任何库状态下都该报出来——否则同一个错误会在库非空时抛 QueryError、
        # 在库为空时被静默降级成 empty_reason=no_data，而后者让调用方以为
        # "条件是合并生效的，只是恰好没数据"。（这一条是被 day066 的测试
        # `test_filter_conflict_is_reported_even_on_an_empty_store` 逼出来的：
        # 最初的实现把早退放在这里之前，行为与这段注释相反。）
        combined = combine_where(resolved.where, resolved.time_range)

        if state.count_store == 0:
            # 空库是**合法状态**：刚建好、刚清空重建、或在演示脚本的第二段里
            # 都是一个正常存在的库。为它报错会让"调用方"去修一个不存在的问题。
            notes.append("库是空的（count=0）：这是合法状态，直接返回空结果而不报错")
            return RetrievalResult(
                query=resolved,
                fetch_k=fetch_k,
                candidates=0,
                filter_applied=filter_applied,
                empty_reason=EMPTY_REASON_NO_DATA,
                metric=state.metric,
                index_state=state,
                latency_ms=_elapsed_ms(started),
                notes=tuple(notes),
            )

        # 第 4 步：编码查询。三项校验（维度 / 有限 / 非零）都在 _encode_query 里。
        vector = self._encode_query(resolved.text)
        notes.extend(self._encoder_notes(state, len(vector)))

        # 第 5 步：库侧查询。min_score **不传给库**——理由见模块 docstring 第 6 步。
        search = self._backend.query(vector, fetch_k, where=combined)
        if filter_applied and search.candidates == 0:
            notes.append("过滤条件把候选筛成了 0 条：请检查字段名与取值，或放宽条件")
        hits = [self._to_hit(item) for item in search.hits]

        # 第 6 步：阈值落刀（在本层）。
        kept, dropped_threshold = _apply_threshold(hits, resolved.min_score)

        # 第 7 步：多样性裁剪。
        kept, dropped_diversity = _apply_diversity(kept, resolved.max_per_doc, self._doc_id_field)

        # 第 8 步：重排名次 + 截断到 top_k。
        ranked = _rerank(kept)
        top_k = resolved.top_k if resolved.top_k is not None else self._default_top_k
        final = tuple(ranked[:top_k])
        dropped_top_k = len(ranked) - len(final)

        # 第 9 步：诊断与装配。
        empty_reason = _diagnose(
            hits=final,
            count_store=state.count_store,
            candidates=search.candidates,
            filter_applied=search.filter_applied,
            dropped_below_threshold=dropped_threshold,
            dropped_by_diversity=dropped_diversity,
        )
        return RetrievalResult(
            query=resolved,
            hits=final,
            fetch_k=fetch_k,
            candidates=search.candidates,
            filter_applied=search.filter_applied,
            dropped_below_threshold=dropped_threshold,
            dropped_by_diversity=dropped_diversity,
            dropped_by_top_k=dropped_top_k,
            empty_reason=empty_reason,
            metric=search.metric,
            index_state=state,
            latency_ms=_elapsed_ms(started),
            notes=tuple(notes),
        )

    def retrieve_many(self, queries: Sequence[str | RetrievalQuery]) -> list[RetrievalResult]:
        """批量检索（逐条调用 ``retrieve``，顺序与入参一致）.

        **刻意不做任何批处理**：这一层的"批量"是报告与评估的便利入口，
        不是性能特性。逐条调用让每次检索都有一份独立可复现的
        ``latency_ms`` 与 ``index_state``——而"这一批总共花了多久"
        与"其中一次特别慢"是两个不同的问题（后者才是要查的那个）。
        """
        return [self.retrieve(item) for item in queries]

    def explain(self, result: RetrievalResult) -> list[str]:
        """在 ``result.explain()`` 之上补充"字段拼写检查"与"漂移处置建议".

        四问（哪一版索引 / 什么过滤 / 为什么少 / 为什么空）由
        ``RetrievalResult.explain()`` 回答——那些信息都在结果里，
        不需要再碰库。这里补的两条**都需要读库**，因此只能由检索器给出：

        ```text
        字段拼写检查   需要"库里出现过哪些元数据字段"这个全集
        漂移处置建议   需要知道这次到底漂在哪（并给出怎么修）
        ```
        """
        lines = list(result.explain())
        if result.query.where:
            known = self._metadata_fields()
            unknown = unknown_filter_fields(result.query.where, known)
            if unknown:
                lines.append(
                    f"字段拼写检查：{unknown} 在本次库里从未出现过"
                    f"（库里出现过的字段：{sorted(known)}）——"
                    "过滤条件的字段名很可能写错了。注意：它只能发现"
                    "「整个库里都没有这个字段」，发现不了「值写错」。"
                )
            else:
                lines.append(
                    "字段拼写检查：过滤用到的字段在库里都出现过"
                    "（这只证明字段名对，不能证明值对——值域是业务知识）"
                )
        if result.index_state.has_drift:
            lines.append(
                "漂移处置建议：清单与库互相矛盾，本次结果可能少召回——"
                "请重建清单（indexing.manifest.manifest_from_store）或补回缺失记录；"
                "要让这一点直接变成失败，请用 Retriever(strict_index=True)。"
            )
        lines.append(
            f"分组口径：多样性按 metadata[{self._doc_id_field!r}] 分组，"
            f"时间字段默认 {self._time_field!r}"
            f"（本检索器的名字 {self._name!r}，默认 top_k={self._default_top_k}）"
        )
        return lines

    # ------------------------------------------------------------------ 九步实现

    def _normalize_query(self, query: str | RetrievalQuery) -> tuple[RetrievalQuery, list[str]]:
        """第 1 步：把入参收敛成一个**字段齐全**的 ``RetrievalQuery``，并记下参数解析的注记.

        ``None`` → 默认值的三条规则：

        ```text
        top_k        默认 default_top_k（构造参数缺省时来自 settings.retrieval_top_k）
        min_score    默认 self._min_score（None 表示**不设阈值**，不是 0）
        max_per_doc  默认 self._max_per_doc，其中 0 归一成 None（None = 不限）
        ```

        一个边界情形在下面显式处理：调用方只给了 ``fetch_k`` 而它小于默认
        ``top_k`` 时，把 ``top_k`` **收敛到 fetch_k** 而不是报错——
        因为"深度 >= 条数"这条不变量必须成立，而调用方显然想要那个更小的深度。
        这条收敛会进 ``notes``（一次静默的改写必须可见）。
        """
        raw = query if isinstance(query, RetrievalQuery) else RetrievalQuery(text=query)
        notes: list[str] = []
        top_k = raw.top_k if raw.top_k is not None else self._default_top_k
        if raw.fetch_k is not None and raw.fetch_k < top_k:
            notes.append(
                f"显式 fetch_k={raw.fetch_k} 小于默认 top_k={top_k}："
                f"已把本次 top_k 收敛为 {raw.fetch_k}（深度不能小于条数）"
            )
            top_k = raw.fetch_k
        resolved = RetrievalQuery(
            text=raw.text,
            top_k=top_k,
            fetch_k=raw.fetch_k,
            where=raw.where,
            time_range=raw.time_range,
            min_score=raw.min_score if raw.min_score is not None else self._min_score,
            max_per_doc=(
                raw.max_per_doc if raw.max_per_doc is not None else (self._max_per_doc or None)
            ),
            route=raw.route,
            extra=dict(raw.extra),
        )
        return resolved, notes

    def _fetch_depth(self, query: RetrievalQuery, notes: list[str]) -> int:
        """第 3 步：算召回深度 ``fetch_k``（过取倍率 + 封顶）.

        ```text
        没有显式 fetch_k → ceil(top_k × multiplier)，再取 max(top_k, …)
        有显式 fetch_k   → 用它（RetrievalQuery 已保证 >= top_k）
        两条路都封顶     → MAX_FETCH_K，并在 notes 里说明"已封顶"
        ```

        ``max(top_k, …)`` 里的那个 ``top_k`` 不是冗余：``multiplier=1`` 时
        ``ceil`` 恰好等于 ``top_k``，但一旦有人把 ``multiplier`` 配成 0
        （哪怕是误配），``max`` 这一层能保证"深度绝不会小于要返回的条数"。
        """
        top_k = query.top_k if query.top_k is not None else self._default_top_k
        desired = (
            query.fetch_k
            if query.fetch_k is not None
            else int(math.ceil(top_k * self._fetch_multiplier))
        )
        fetch_k = max(top_k, desired)
        if fetch_k > MAX_FETCH_K:
            notes.append(
                f"召回深度按 {self._fetch_multiplier} 倍本应取 {fetch_k} 条，"
                f"已封顶到 MAX_FETCH_K={MAX_FETCH_K}（与 vectorstore.MAX_TOP_K 同一个上限）"
            )
            fetch_k = MAX_FETCH_K
        return fetch_k

    def _inspect_index(self) -> IndexState:
        """第 2 步：索引体检（条数 / 维度 / 度量 / 清单漂移）.

        漂移的判定**复用 day065 的 ``compare_with_store``**（它内部就是"清单与库
        逐项对照"的那一份规则）。这里不做第二套判据：两份规则一定会分家，
        而分家之后"清单到底算不算漂"会取决于谁先被调用。

        漂移的**措辞**与 ``indexing.verify_index`` 的 ``problems`` 对齐
        （同一种现象在两处应该长得一样），只是这里不合成 ``ok`` 判定——
        要不要阻断由 ``strict_index`` 决定（见 ``IndexState`` 的取舍）。
        """
        count_store = self._backend.count()
        if self._manifest is None:
            return IndexState(
                version_id="",
                count_store=count_store,
                count_manifest=None,
                dimension=self._backend.dimension,
                metric=self._backend.metric,
                drift=(),
            )
        comparison = compare_with_store(self._manifest, self._backend)
        return IndexState(
            version_id=self._manifest.version_id,
            count_store=count_store,
            count_manifest=int(comparison["count_manifest"]),
            dimension=self._backend.dimension,
            metric=self._backend.metric,
            drift=_drift_messages(comparison, self._manifest, self._backend),
        )

    def _warn_drift(self, state: IndexState) -> None:
        """漂移记一条 WARNING 日志（"可观测的告警"而不是静默的质量下降）.

        用 WARNING 而不是 ERROR：检索仍然会返回结果，它不是失败；
        但"清单过期"这件事必须在一个不依赖调用方配合的地方留下痕迹——
        日志是唯一这样的地方（端点可能被绕过，demo 可能不看返回值）。
        """
        logger.warning(
            "检索器 %s 检测到索引漂移 %d 项（版本 %s，库 %d 条 / 清单 %s 条）：%s",
            self._name,
            len(state.drift),
            state.version_id or "（无）",
            state.count_store,
            state.count_manifest,
            state.drift[0] if state.drift else "（无明细）",
        )

    def _encode_query(self, text: str) -> list[float]:
        """第 4 步：编码查询并做三项校验，任何一项不过都是 ``IndexStateError``.

        ```text
        长度 != 库维度   编码器与建库的不是同一套 → 排序会整体失去意义
        含 nan / inf     比较恒为 False → 这条查询的排序不可复现
        模长为 0         余弦相似度 0/0 → "平等地命中或不命中一切"
        ```

        三项都属于 **``IndexStateError`` 而不是 ``QueryError``**：调用方那句话
        没有问题，问题在"这一层配的编码器与这个库对不上"，
        修它的人要去改装配代码或重建索引（见 ``errors`` 的分族依据）。
        """
        raw = self._embedding.embed(text)
        if not isinstance(raw, (list, tuple)):
            raise IndexStateError(
                f"编码器返回了 {type(raw).__name__}，而本层要求 list[float]。"
                "请检查 EmbeddingProvider.embed 的实现（它必须返回定长浮点序列）。"
            )
        try:
            values = [float(item) for item in raw]
        except (TypeError, ValueError) as exc:
            raise IndexStateError(
                f"编码器返回的向量里有无法转成浮点数的分量：{exc}。"
                "这通常意味着提供方把 ndarray 或字符串塞进了结果里。"
            ) from exc
        if not values:
            raise IndexStateError(
                "编码器返回了空向量：空向量无法与任何向量比距离，"
                "让它进检索会得到'永远返回空集'——看起来像'库里没数据'。"
            )
        dimension = self._backend.dimension
        if dimension and len(values) != dimension:
            raise IndexStateError(
                f"查询向量 {len(values)} 维、本库 {dimension} 维："
                "查询用的编码器与建库用的不是同一套，请用同一个提供方重建索引"
                "（截断或补零都会让排序静默失去意义）。"
            )
        for position, value in enumerate(values):
            if not math.isfinite(value):
                raise IndexStateError(
                    f"查询向量的第 {position} 个分量不是有限数（{value!r}）："
                    "nan/inf 参与的比较恒为 False，排序结果不可复现。"
                )
        if math.fsum(value * value for value in values) == 0.0:
            raise IndexStateError(
                "编码器返回了零向量（模长为 0）：余弦相似度对它是 0/0，"
                "这次查询会'平等地命中或不命中一切'。零向量通常意味着"
                "编码器收到了空文本或内部失败，请检查 EmbeddingProvider。"
            )
        return values

    def _encoder_notes(self, state: IndexState, vector_dimension: int) -> list[str]:
        """收集"编码器与清单身份不符"的两条注记（只是注记，不是错误）.

        为什么不是错误：**它仍然能查出结果**，而错配的证据已经在
        ``IndexState`` 里（维度不符本身就是一项漂移）。把它升级成异常会让
        "清单里 provider 名字写得不完全一样"这种无害情形变成一次服务不可用。
        """
        if self._manifest is None:
            return []
        notes: list[str] = []
        declared = self._manifest.identity
        if declared.dimension != vector_dimension:
            notes.append(
                f"编码器维度 {vector_dimension} 与清单记的 {declared.dimension} 不一致："
                "这份清单多半是另一套编码器建的，本次结果的排序口径可能与清单无关"
            )
        provider = type(self._embedding).__name__
        if provider != declared.provider:
            notes.append(
                f"编码器实现 {provider} 与清单记的 {declared.provider} 不同："
                "结论仍然可用，但「这份索引是用谁建的」这件事对不上，"
                "建议核对 settings.embedding_provider 与清单的 identity"
            )
        return notes

    def _to_hit(self, item: SearchHit) -> RetrievalHit:
        """``SearchHit`` → ``RetrievalHit``（丢掉向量与 distance，补上通道名）."""
        return RetrievalHit(
            record_id=item.record.record_id,
            score=float(item.score),
            rank=item.rank,
            text=item.record.text,
            metadata=dict(item.record.metadata),
            channel=CHANNEL_VECTOR,
        )

    def _metadata_fields(self) -> set[str]:
        """库里出现过的全部元数据字段名（``explain`` 的字段拼写检查用它）.

        代价是遍历整库一次（``ids()`` + ``get_many``）。它只被
        ``explain`` 调用，而 ``explain`` 是"人要看诊断"时才会走的路径——
        检索主路径上没有任何一次全库遍历（那是向量库的活）。
        """
        names: set[str] = set()
        for record in self._backend.get_many(self._backend.ids()):
            names.update(str(key) for key in record.metadata)
        return names


# --------------------------------------------------------------------------- #
# 配置解析（构造期一次算清，之后不再看 settings）
# --------------------------------------------------------------------------- #


def _resolve_top_k(value: int | None) -> int:
    """``default_top_k``：``None`` → ``settings.retrieval_top_k``，并校验范围."""
    resolved = settings.retrieval_top_k if value is None else value
    if not isinstance(resolved, int) or isinstance(resolved, bool) or resolved < 1:
        raise QueryError(f"default_top_k 必须是 >= 1 的整数，收到 {resolved!r}")
    if resolved > MAX_FETCH_K:
        raise QueryError(
            f"default_top_k={resolved} 超过上限 {MAX_FETCH_K}"
            "（与 vectorstore.MAX_TOP_K 同一个上限）"
        )
    return resolved


def _resolve_multiplier(value: int | None) -> int:
    """``fetch_multiplier``：``None`` → ``settings.retrieval_fetch_multiplier``."""
    resolved = settings.retrieval_fetch_multiplier if value is None else value
    if not isinstance(resolved, int) or isinstance(resolved, bool) or resolved < 1:
        raise QueryError(
            f"fetch_multiplier 必须是 >= 1 的整数，收到 {resolved!r}。"
            "倍率 < 1 会让'深度 >= 条数'无法成立（那正是要取回条数的下限）。"
        )
    return resolved


def _resolve_min_score(value: float | None) -> float | None:
    """``min_score``：``None`` → ``settings.retrieval_min_score``（可能仍是 None = 不设阈值）."""
    resolved = settings.retrieval_min_score if value is None else value
    if resolved is None:
        return None
    if not isinstance(resolved, (int, float)) or isinstance(resolved, bool):
        raise QueryError(f"min_score 必须是数字或 None，收到 {type(resolved).__name__}")
    if not math.isfinite(float(resolved)):
        raise QueryError(f"min_score 必须是有限数，收到 {resolved!r}")
    return float(resolved)


def _resolve_max_per_doc(value: int | None) -> int:
    """``max_per_doc``：``None`` → ``settings.retrieval_max_per_doc``（``0`` = 不限）."""
    resolved = settings.retrieval_max_per_doc if value is None else value
    if not isinstance(resolved, int) or isinstance(resolved, bool) or resolved < 0:
        raise QueryError(
            f"max_per_doc 必须是 >= 0 的整数（0 表示不限），收到 {resolved!r}"
        )
    return resolved


def _resolve_field_name(label: str, value: str) -> str:
    """``doc_id_field`` / ``time_field`` / ``name`` 共用：必须是非空字符串."""
    if not isinstance(value, str) or not value.strip():
        raise QueryError(f"{label} 必须是非空字符串，收到 {value!r}")
    return value.strip()


# --------------------------------------------------------------------------- #
# 三道减法（纯函数：给定一批命中，回答"留下哪些、切掉几条"）
# --------------------------------------------------------------------------- #


def _filter_applied(query: RetrievalQuery) -> bool:
    """这次到底有没有过滤（``{}`` 与 ``None`` 都算"没有"）."""
    return query.has_filter


def _apply_threshold(
    hits: list[RetrievalHit],
    min_score: float | None,
) -> tuple[list[RetrievalHit], int]:
    """第 6 步：按 ``min_score`` 落刀（语义与 ``backend.query(min_score=...)`` 一致）.

    用 ``>=`` 而不是 ``>``：与 ``VectorBackend.query`` 里那一行完全相同。
    一个 ``>`` 与 ``>=`` 的差别在"阈值恰好等于某条命中分数"时体现出来，
    而那种情形**必然出现**——同一个确定性编码器对同文本给出逐位相同的向量。
    """
    if min_score is None:
        return list(hits), 0
    threshold = float(min_score)
    kept: list[RetrievalHit] = []
    dropped = 0
    for hit in hits:
        if hit.score >= threshold:
            kept.append(hit)
        else:
            dropped += 1
    return kept, dropped


def _apply_diversity(
    hits: list[RetrievalHit],
    max_per_doc: int | None,
    doc_id_field: str,
) -> tuple[list[RetrievalHit], int]:
    """第 7 步：每文档最多留 ``max_per_doc`` 条（``None`` / ``0`` = 不限）.

    入参已经按分数降序，因此"每组留前 N 条"就是"按顺序遇到就收"——
    不需要分组再排序（分组会丢掉"同一分数时按 id"的全局秩序）。

    两个刻意的决定：

    ```text
    缺 doc_id 的记录自成一类（键为 ""）  它不属于任何文档，因此不该被别人的上限挤掉；
                                        代价是"一批没有 parent 的记录"会互相挤，
                                        这条上限对它们仍然生效（同一个规则对所有人）
    key 用 metadata[doc_id_field] 而非 record_id   多样性要治的正是"同一文档刷屏"，
                                        而 record_id 永远唯一（治不了任何东西）
    ```
    """
    if not max_per_doc:
        return list(hits), 0
    seen: dict[str, int] = {}
    kept: list[RetrievalHit] = []
    dropped = 0
    for hit in hits:
        key = str(hit.metadata.get(doc_id_field, "") or "")
        used = seen.get(key, 0)
        if used >= max_per_doc:
            dropped += 1
            continue
        seen[key] = used + 1
        kept.append(hit)
    return kept, dropped


def _rerank(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    """第 8 步：按"分数降序 + id 升序"重排，并把 ``rank`` 重编成 0 起连续.

    重排的必要性见模块 docstring（写在这一层的排序规则不依赖后端实现）；
    重编 ``rank`` 的必要性是"三道减法会留下空洞"：阈值切掉第 2 名之后，
    后端的 rank 会变成 0、1、3、4——**一个带空洞的名次列表不能用来索引**
    （"第 3 条"到底指哪一个）。因此 ``rank`` 在最终结果里永远从 0 连续。
    """
    ordered = sorted(hits, key=lambda hit: (-hit.score, hit.record_id))
    return [replace(hit, rank=position) for position, hit in enumerate(ordered)]


def _diagnose(
    *,
    hits: tuple[RetrievalHit, ...],
    count_store: int,
    candidates: int,
    filter_applied: bool,
    dropped_below_threshold: int,
    dropped_by_diversity: int,
) -> str:
    """第 9 步：按 ``EMPTY_REASONS`` 的优先级定出**唯一**原因.

    ```text
    有命中                     → "hits"
    库是空的                   → "no_data"（合法状态）
    过滤后候选为 0             → "filtered_out"
    阈值切掉了东西             → "below_threshold"
    多样性挤掉了东西           → "diversity_trimmed"
    过滤开着但候选非空、又没切掉任何东西 → "filtered_out"（仍是过滤器这一边的事）
    其余                       → "no_data"（防御性兜底，见下）
    ```

    优先级把**更上游的成因**排在前面：库是空的时候说"被阈值切掉"没有意义
    （根本没东西可切）。最后两条分支各挡一种"报告不能是空的"：
    诊断字段永远要有一个值，而它是这五种之一（见 ``EMPTY_REASONS``）。

    **一个必须写下来的可达性事实**：``"diversity_trimmed"`` 这一支在当前算法下
    **端到端不可达**——第 7 步里每一组至少会留下一条（``used >= limit`` 只对
    第二、第三条之后的命中成立），因此 ``dropped_by_diversity > 0`` 时
    ``kept`` 必然非空。保留这一支的理由是**诊断规则不该依赖某一步的内部细节**：
    一旦第 7 步改成"每组至少两条"或"只保留主文档"，这个分支就会立刻变成活路径，
    而那时它必须在正确的位置（阈值之后、兜底之前）。要验证它只能直接调用本函数
    （单元级构造），这一条写在这里，免得下一个人拿着"四种空结果"的清单
    在端到端路径上找一个不存在的情形。
    """
    if hits:
        return EMPTY_REASON_NONE
    if count_store == 0:
        return EMPTY_REASON_NO_DATA
    if filter_applied and candidates == 0:
        return EMPTY_REASON_FILTERED_OUT
    if dropped_below_threshold:
        return EMPTY_REASON_BELOW_THRESHOLD
    if dropped_by_diversity:
        return EMPTY_REASON_DIVERSITY
    if filter_applied:
        return EMPTY_REASON_FILTERED_OUT
    # 库非空、无过滤、无阈值、无多样性，却一条都没排出来：
    # 只可能是后端在这一层返回了空列表（restrict 命中不到任何记录、或记录表与索引不同步）。
    # 这不是任何一种"被切掉"，因此归到"库里没有可用的数据"这一类。
    return EMPTY_REASON_NO_DATA


def _drift_messages(
    comparison: dict[str, Any],
    manifest: IndexManifest,
    backend: VectorBackend,
) -> tuple[str, ...]:
    """把 ``compare_with_store`` 的结果折成几句话（措辞对齐 ``verify_index``）.

    四类不一致各自一句话，顺序与 ``indexing.verify_index`` 的 ``problems`` 相同：
    **同一种现象在两处长得一样**，读日志的人不需要重新学一遍措辞。
    """
    problems: list[str] = []
    missing = list(comparison["missing_in_store"])
    orphan = list(comparison["orphan_in_store"])
    if missing:
        problems.append(
            f"清单里有 {len(missing)} 条记录在库里找不到：{_ids_preview(missing)}——"
            "上一次构建可能半途失败，或有人手工删过记录；"
            "请先重建清单（或用这些 id 补回向量），不要在这份清单上做增量。"
        )
    if orphan:
        problems.append(
            f"库里有 {len(orphan)} 条记录不在清单里：{_ids_preview(orphan)}——"
            "它们不会被增量更新覆盖，请先重建清单。"
        )
    if not bool(comparison["dimension_match"]):
        problems.append(
            f"维度不一致：清单记的是 {manifest.identity.dimension} 维，"
            f"库里是 {backend.dimension} 维——这份清单描述的是另一个编码器建的库，"
            "不能拿它判断本次检索的召回口径。"
        )
    if not bool(comparison["metric_match"]):
        problems.append(
            f"度量不一致：清单记的是 {manifest.metric!r}，库是 {backend.metric!r}——"
            "同一批向量在两种度量下的最近邻不是同一批。"
        )
    if not bool(comparison["backend_match"]):
        problems.append(
            f"后端不一致：清单记的是 {manifest.backend!r}，库是 {backend.name!r}——"
            "换后端可以重建，但不能与旧版共用一份清单。"
        )
    return tuple(problems)


def _ids_preview(ids: Sequence[str], limit: int = 3) -> str:
    """把一批 id 写成一行预览（``problems`` 是要被人读完的，不能整列贴进去）."""
    shown = ", ".join(repr(str(item)) for item in list(ids)[:limit])
    suffix = ", …" if len(ids) > limit else ""
    return f"[{shown}{suffix}]"


def _elapsed_ms(started: float) -> float:
    """从 ``perf_counter`` 起点到现在的毫秒数（保留 3 位，够用且不会太长）."""
    return round((time.perf_counter() - started) * 1000, 3)


# --------------------------------------------------------------------------- #
# 装配入口
# --------------------------------------------------------------------------- #


def build_retriever(
    backend: VectorBackend,
    embedding: EmbeddingProvider,
    *,
    manifest: IndexManifest | None = None,
    **overrides: Any,
) -> Retriever:
    """按 ``settings.retrieval_*`` 装配一个检索器（端点与演示脚本的统一入口）.

    这里把两个"签名默认值是字面量、但 settings 里也有同名配置"的字段
    **显式指过去**（``time_field`` 与 ``strict_index``）：

    ```text
    time_field    构造签名默认 "created_at"（与 settings 默认值相同，
                  因此无法区分"没传"与"传了默认值"）→ 在装配路径上用 settings 的值
    strict_index  构造签名默认 False → 在装配路径上取 settings.retrieval_strict_index
    ```

    其余参数（``top_k`` / ``fetch_multiplier`` / ``min_score`` / ``max_per_doc``）
    传 ``None`` 时由 ``Retriever`` 自己读 settings，因此**不必在这里抄一遍**——
    抄一遍就会有两处默认值，而它们迟早会不一致。

    ``**overrides`` 原样透传给 ``Retriever``，因此"显式传参"永远赢过 settings。
    """
    params: dict[str, Any] = {
        "time_field": settings.retrieval_time_field,
        "strict_index": settings.retrieval_strict_index,
    }
    params.update(overrides)
    return Retriever(backend, embedding, manifest=manifest, **params)


__all__ = [
    "Retriever",
    "build_retriever",
]
