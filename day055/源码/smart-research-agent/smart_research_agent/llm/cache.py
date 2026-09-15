"""语义缓存：用 Embedding 命中相似历史查询，降低重复 LLM 调用成本（day043）.

LLM 调用是智能体运行成本的大头。观察真实流量会发现大量重复：

- 用户反复问同一个问题（措辞不同）；
- 长会话里系统提示词与上下文前缀反复注入（相同的提示词段）；
- 多个 Agent 角色对同一个子问题各问一遍。

缓存的思想是：把「查询 → 回复」存下来，下次遇到相同或足够相似的查询
直接返回，省掉一次 API 调用。day041 给了我们真正的语义 embedding——
语义相近的文本得到相近的向量，这正是「语义缓存」的地基：不需要字面
完全相同，措辞不同但意思相近也能命中。

与 day032 成本追踪的衔接：``CachedLLM`` 在命中缓存时把「本应发生的
调用」折算成美元累计到 ``savings_usd``——缓存到底省了多少钱，是数字
而不是口号；这些数字可以与 ``CostTracker`` 的实际账单合并出完整成本
图景（见 cost_tracker.report 的 top_paths 与 scripts/cache_demo.py）。
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from smart_research_agent.config import settings
from smart_research_agent.llm.base import BaseLLM, Message
from smart_research_agent.llm.embedding import EmbeddingProvider
from smart_research_agent.llm.tokenizer import TokenCounter
from smart_research_agent.memory.vector_store import cosine_similarity
from smart_research_agent.observability.cost_tracker import DEFAULT_PRICE_TABLE


@dataclass
class CacheEntry:
    """一条缓存记录：查询、回复与查询向量.

    查询向量在写入时算好并随条目保存，命中判断不再重复编码——一次
    ``put`` 一次 ``embed``，之后的所有 ``get`` 只做余弦比较（O(1) 的
    向量点积），不为每次查找重新向量化查询文本。
    """

    query: str
    reply: str
    embedding: list[float]


class SemanticCache:
    """语义缓存：精确匹配快路径 + 语义相似匹配 + LRU 淘汰.

    用法::

        cache = SemanticCache(embedding=CharNgramEmbedding())
        cache.get("什么是 RAG")            # 未命中 -> None
        cache.put("什么是 RAG", "RAG 是检索增强生成")
        cache.get("RAG 是什么")            # 语义命中 -> "RAG 是检索增强生成"

    命中分两级：先做零成本的精确匹配（``strip`` 后字符串相等），再对
    全部缓存条目做余弦最近邻（O(n) 次点积）。阈值由 ``similarity_threshold``
    控制，取 1.0 时退化为「仅精确匹配」，取 0.85 时允许同义改写命中——
    阈值越高越保守（少命中、省得少、但不误答），越低越激进（多命中、
    省得多、但可能张冠李戴），调阈值就是调「省钱」与「正确」的天平。
    """

    def __init__(
        self,
        embedding: EmbeddingProvider,
        similarity_threshold: float = 0.85,
        max_size: int = 128,
    ):
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold 必须在 [0, 1] 区间")
        if max_size <= 0:
            raise ValueError("max_size 必须为正整数")
        self._embedding = embedding
        self._threshold = similarity_threshold
        self._max_size = max_size
        #: 有序字典即 LRU：队尾是最近使用的，队头是待淘汰的
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self.exact_hits = 0
        self.semantic_hits = 0
        self.misses = 0
        self.evictions = 0

    @staticmethod
    def _normalize(query: str) -> str:
        """把查询规整为缓存键：去首尾空白。大小写敏感性留给语义层兜底."""
        return query.strip()

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def similarity_threshold(self) -> float:
        return self._threshold

    @property
    def hits(self) -> int:
        return self.exact_hits + self.semantic_hits

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def _touch(self, key: str) -> None:
        """把 key 移到队尾（最近使用），淘汰时从队头开始."""
        entry = self._entries.pop(key)
        self._entries[key] = entry

    def get(self, query: str) -> str | None:
        """命中返回缓存的回复，未命中返回 None.

        两条匹配路径（顺序即优先级）：
        1. 精确匹配——strip 后字符串相等，零成本、无歧义；
        2. 语义匹配——与全部条目的余弦最近邻达到阈值即命中，措辞不同
           但意思相近也能复用；代价是 O(n) 次点积，n 为缓存条目数。
        """
        key = self._normalize(query)
        if key in self._entries:
            self._touch(key)
            self.exact_hits += 1
            return self._entries[key].reply

        if self._entries:
            qvec = self._embedding.embed(key)
            best_key: str | None = None
            best_entry: CacheEntry | None = None
            best_score = -1.0
            for k, entry in self._entries.items():
                score = cosine_similarity(entry.embedding, qvec)
                if score > best_score:
                    best_score = score
                    best_key = k
                    best_entry = entry
            if best_entry is not None and best_score >= self._threshold:
                self._touch(best_key)  # type: ignore[arg-type]
                self.semantic_hits += 1
                return best_entry.reply

        self.misses += 1
        return None

    def put(self, query: str, reply: str) -> None:
        """写入一条缓存；已存在的键更新回复，超容量时淘汰最久未用的条目."""
        key = self._normalize(query)
        if key in self._entries:
            self._touch(key)
            self._entries[key] = CacheEntry(
                query=key, reply=reply, embedding=self._embedding.embed(key)
            )
            return
        self._entries[key] = CacheEntry(
            query=key, reply=reply, embedding=self._embedding.embed(key)
        )
        while len(self._entries) > self._max_size:
            self._entries.popitem(last=False)
            self.evictions += 1


class CachedLLM(BaseLLM):
    """带缓存的 LLM 包装器：对上层透明，命中缓存时跳过真实调用.

    ``CachedLLM`` 实现 ``BaseLLM`` 接口，可包在任意 LLM（乃至
    ``ModelRouter``）外面，Agent 与 API 层无感知。命中时返回缓存回复
    并累计「节省成本」——把这次本应发生的调用按价格表折算成美元。
    成本估算复用 day034 的 ``TokenCounter``：输入 token 按查询文本计，
    输出 token 按缓存回复文本计，二者都是调用发生前的诚实估算。

    归类上它与 ``ModelRouter`` 同属**包装器**：包装器本身一个 token 都
    不产生，用量全落在被包的那一层。因此它必须交代清楚"我替谁说话"
    （``last_used_llm``），否则 day046 的 ``IntegratedPipeline._account``
    在归因时会读到包装器、发现没有 ``usage_log`` 而断链，把真实用量
    记成 0——这正是 day047 复盘要修的遗留缺陷。
    """

    def __init__(
        self,
        llm: BaseLLM,
        cache: SemanticCache,
        model: str = "gpt-4o-mini",
        price_table: dict[str, dict[str, float]] | None = None,
        token_counter: TokenCounter | None = None,
    ):
        self._llm = llm
        self._cache = cache
        self._model = model
        self._price_table = (
            {k: dict(v) for k, v in price_table.items()}
            if price_table is not None
            else {k: dict(v) for k, v in DEFAULT_PRICE_TABLE.items()}
        )
        # 默认强制走字符级估算：离线测试全程确定，不依赖 tiktoken 词表文件
        self._counter = token_counter or TokenCounter(prefer_tiktoken=False)
        #: 累计节省成本（美元）：每次命中把本应发生的调用折算累加
        self.savings_usd = 0.0
        self.cache_hits = 0
        self.cache_misses = 0

    @property
    def last_used_llm(self):
        """最后一次实际产生用量的叶子模型（day047 复盘遗留项）.

        透传规则：内层若自己也是包装器（例如 ``ModelRouter``），就继续
        向内取它的 ``last_used_llm``，把路由真正选中的叶子暴露给计费器；
        内层是普通客户端时直接返回它。从未发生过调用、或内层路由还没
        做决策时返回内层对象本身——由 ``Pipeline._account`` 的
        ``getattr(..., None) or llm`` 兜底，语义是"账记在它头上"。

        本属性**只读且不改变任何既有行为**：缓存命中路径依旧直接返回，
        不调用底层模型、不产生任何用量。
        """
        return getattr(self._llm, "last_used_llm", None) or self._llm

    @staticmethod
    def _last_user_message(messages: list[Message]) -> str:
        """取最近一条 user 消息作为缓存键（与 ModelRouter 的决策输入一致）."""
        return next((m.content for m in reversed(messages) if m.role == "user"), "")

    def _estimate_call_cost(self, query: str, reply: str) -> float:
        """把「一次查询 + 一次回复」按价格表折算成美元（day034 口径）."""
        expected_completion = self._counter.count_tokens(reply, self._model)
        info = self._counter.estimate_cost(
            query,
            self._model,
            self._price_table,
            expected_completion_tokens=expected_completion,
        )
        return info["total_cost_usd"]

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """先查缓存，未命中才调用底层 LLM 并把结果写回缓存."""
        query = self._last_user_message(messages)
        cached = self._cache.get(query)
        if cached is not None:
            self.cache_hits += 1
            self.savings_usd += self._estimate_call_cost(query, cached)
            return cached
        self.cache_misses += 1
        reply = self._llm.chat(messages, temperature=temperature, max_tokens=max_tokens)
        self._cache.put(query, reply)
        return reply


def default_cache(embedding: EmbeddingProvider | None = None) -> SemanticCache:
    """按 ``settings`` 构建默认语义缓存（day043）.

    阈值与容量读自配置（``cache_similarity_threshold`` / ``cache_max_size``），
    embedding 缺省走 ``default_embedding()`` 工厂——与 LLM、embedding 的
    「离线可启动」哲学一致：默认 CharNgramEmbedding 离线可用，缓存功能
    无需任何外部服务即可运行与测试。
    """
    from smart_research_agent.llm.embedding import default_embedding

    return SemanticCache(
        embedding=embedding or default_embedding(),
        similarity_threshold=settings.cache_similarity_threshold,
        max_size=settings.cache_max_size,
    )
