"""上下文打包与引用：把命中折成**一份带编号、有预算的提示词片段**（M6-D5）.

检索器交出来的是一批 ``RetrievalHit``（id + 分数 + 正文 + 元数据），
而提示词只认一段**文本**。这两者之间隔着的三件事全都发生在本模块：

```text
预算    给模型留多少空间        → max_chars（默认取 settings.retrieval_max_context_chars）
截断    单条太长怎么办          → per_hit_chars（先截单条，再从尾部整条丢）
编号    答案里那些 [1] 指谁     → Citation.marker（只给真的进了上下文的那些）
```

## 为什么"先截单条，再丢尾部"

两种做法都能把包压进预算，但它们的**可回答性**完全不同：

```text
先截单条再丢尾   一条过长 → 它被截断（truncated_hits += 1），其余条照常进包
                 整包还超 → 从最后一条开始整条丢掉（dropped_hits 里能看到它们）
平均切每一段     每条都变短，看不出"是哪一条吃掉了预算"
```

后者把"这次回答为什么信息不全"这个问题抹掉了：报告里只剩一个"包满了"。
因此截断**只发生在单条上**，而丢东西**只发生在包的尾部**——两件事都留下数字。

## 预算单位为什么默认是 chars

与 day062 的默认度量同一条理由：**确定性、离线、可复现**。
``len()`` 在任何机器上给出同一个数，而 token 计数要引入分词器与版本。
需要真实 token 口径时把 ``measurer`` 注入进来即可（例如接
``llm.tokenizer``）——本模块不 import 任何分词器，因此它不认识 token。

## 引用编号为什么只给进了上下文的那些

编号是"**在这一次提示词里的位置**"，不是记录的身份。被丢掉的尾部如果也占着
编号，模型就可能写出一个指向不存在片段的 ``[7]``——而这种错误看起来
只是"引用了一个没给的片段"，查起来却要重新跑一遍打包。因此
``Citation.marker`` 从 1 起连续编号，与 ``text`` 里的 ``[n]`` 一一对应。

## 谁依赖它

```text
pipeline.RagPipeline   命中 → pack_context → 渲染提示词 → 调 LLM
tests/                 单条截断 / 整包丢尾 / 至少保一条 / 预算过小 / 编号连续
```
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from smart_research_agent.config import settings
from smart_research_agent.retrieval.errors import ContextError
from smart_research_agent.retrieval.types import DEFAULT_MIN_HIT_CHARS, RetrievalHit

#: 块与块之间的分隔符。用空行而不是单换行：**每个块自己就带换行**
#: （``[1] 来源 › 标题`` 与正文之间），单换行会让"下一条命中从哪开始"看不出来。
BLOCK_SEPARATOR = "\n\n"

#: 单条被截断时补的标记。它的作用是让"这段是半截的"在提示词里**看得见**——
#: 不补标记时，模型读到的是一段以逗号结尾的正常文本，它会顺手把话补完。
TRUNCATION_MARKER = "…"


def _char_measure(text: str) -> int:
    """默认度量：字符数（``len()``）.

    与 day062 的默认度量口径一致：确定、离线、不依赖分词器版本。
    写成模块级函数而不是 lambda，是为了让"默认口径是什么"在一个能被引用的
    地方有一个名字（报告与教程都直接称它为 chars 口径）。
    """
    return len(text)


@dataclass(frozen=True)
class Citation:
    """一条引用：**编号 + 这条命中是什么**（M6-D5）.

    ``marker`` 与 ``record_id`` 分别服务两个方向：

    ```text
    marker     从答案回到片段（模型写的 [1] 是哪一条）  → 提示词里的位置
    record_id  从片段回到库（这一条到底是哪一块）        → 溯源与去重
    ```

    ``source`` / ``heading_path`` 是给人看的那半句"这句话出自哪里"，
    它们的回落规则与 ``RetrievalHit.citation`` 完全一致
    （``source`` → ``doc_id`` → ``record_id``；``heading_path`` 缺失即省略）。
    """

    marker: int
    record_id: str
    source: str
    heading_path: str
    score: float

    def __post_init__(self) -> None:
        if not isinstance(self.marker, int) or isinstance(self.marker, bool) or self.marker < 1:
            raise ContextError(
                f"Citation.marker 必须是从 1 起的整数，收到 {self.marker!r}。"
                "编号 0 或负数会让提示词里出现一个指不到任何片段的 [0]——"
                "引用编号从 1 起是与 day069 的引用溯源共用的约定。"
            )
        if not isinstance(self.record_id, str) or not self.record_id.strip():
            raise ContextError(
                f"Citation.record_id 必须是非空字符串，收到 {self.record_id!r}："
                "没有 id 的引用无法回库取原文，只能当作一句无法核对的话。"
            )

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return {
            "marker": self.marker,
            "record_id": self.record_id,
            "source": self.source,
            "heading_path": self.heading_path,
            "score": round(self.score, 6),
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要（``[1] docs/retrieval.md › 检索手册 > 阈值``）."""
        place = f"{self.source} › {self.heading_path}" if self.heading_path else self.source
        return f"[{self.marker}] {place} score={self.score:+.6f}"


@dataclass(frozen=True)
class PackedContext:
    """打包结果：**提示词里的那段文本 + 它的账**（M6-D5）.

    ```text
    text            真正进提示词的文本（[n] 编号已就位）
    citations       编号 → 命中 的对照表（只有进了上下文的那些）
    used_hits       进了上下文的命中（顺序 = 文本里的顺序）
    dropped_hits    预算放不下、从尾部被整条丢掉的命中
    truncated_hits  被单条上限截断的条数（**只算真的进了上下文的那些**）
    char_count      实测占用（默认口径 = 字符数）
    max_chars       这次的预算（char_count > max_chars 只可能发生在"至少保一条"上）
    ```

    ``char_count`` 与 ``max_chars`` 一起给出的理由：**"这次有没有被预算削过"
    必须是一个能被读出来的事实**。只有一个 ``char_count`` 时，
    读的人不知道它是"包刚好这么长"还是"被削到这么长"。

    ``dropped_hits`` 与 ``truncated_hits`` 分开的理由同 ``RetrievalResult``
    的三个 ``dropped_*``：一个要去调预算，一个要去调单条上限，
    合成一个数字之后"该调哪个参数"就答不出来了。
    """

    text: str
    citations: tuple[Citation, ...]
    used_hits: tuple[RetrievalHit, ...]
    dropped_hits: tuple[RetrievalHit, ...]
    truncated_hits: int
    char_count: int
    max_chars: int

    def __post_init__(self) -> None:
        if self.truncated_hits < 0 or self.char_count < 0 or self.max_chars < 1:
            raise ContextError(
                f"PackedContext 的计数字段非法：truncated_hits={self.truncated_hits}、"
                f"char_count={self.char_count}、max_chars={self.max_chars}。"
                "它们是报告里给人看的数字，负数与零预算都只会让报告失去意义。"
            )
        if len(self.citations) != len(self.used_hits):
            raise ContextError(
                f"引用 {len(self.citations)} 条，而进了上下文的命中 {len(self.used_hits)} 条："
                "两者必须一一对应——多出来的引用会指向提示词里不存在的片段。"
            )
        expected = list(range(1, len(self.citations) + 1))
        actual = [citation.marker for citation in self.citations]
        if actual != expected:
            raise ContextError(
                f"引用编号必须是 1 起连续的 {expected}，收到 {actual}："
                "编号是'在这一次提示词里的位置'，中间断号会让模型写出指向空位的 [n]。"
            )

    @property
    def count(self) -> int:
        """进了上下文的命中条数."""
        return len(self.used_hits)

    @property
    def is_empty(self) -> bool:
        """是否没有进任何命中（没有命中可打包时是这样，不是异常）."""
        return not self.used_hits

    @property
    def fill_ratio(self) -> float:
        """预算占用率（``char_count / max_chars``，保留 4 位）.

        "包很满"本身不是问题，"包很满且还有命中被丢掉"才是——
        因此它总是与 ``dropped_hits`` 一起被读。
        """
        return round(self.char_count / self.max_chars, 4)

    def marker_for(self, record_id: str) -> int | None:
        """某个记录在这份上下文里的编号（**不在里面时返回 ``None``**）.

        返回 ``None`` 而不是 0 或抛异常：调用方（例如 day069 的引用溯源）
        拿着一个可能被丢掉的 id 来问编号，而"它没进上下文"是一个正常答案。
        """
        for citation in self.citations:
            if citation.record_id == record_id:
                return citation.marker
        return None

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return {
            "char_count": self.char_count,
            "max_chars": self.max_chars,
            "fill_ratio": self.fill_ratio,
            "count": self.count,
            "truncated_hits": self.truncated_hits,
            "citations": [citation.to_dict() for citation in self.citations],
            "used": [hit.record_id for hit in self.used_hits],
            "dropped": [hit.record_id for hit in self.dropped_hits],
            "text": self.text,
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要（演示脚本与端点逐行打印它）."""
        return (
            f"上下文 {self.char_count}/{self.max_chars} 字（占用 {self.fill_ratio:.1%}）"
            f" | 引用 {self.count} 条 | 截断 {self.truncated_hits} 条"
            f" | 丢尾 {len(self.dropped_hits)} 条"
        )


def pack_context(
    hits: Sequence[RetrievalHit],
    *,
    max_chars: int | None = None,
    per_hit_chars: int | None = None,
    min_hit_chars: int = DEFAULT_MIN_HIT_CHARS,
    measurer: Callable[[str], int] | None = None,
) -> PackedContext:
    """把一批命中打包成一段带编号、不超预算的文本（规则见模块 docstring）.

    五条规则的执行顺序是**固定的**，因为每一步都会改变"还剩多少预算"：

    ```text
    1. 解析预算     None → settings.retrieval_max_context_chars / retrieval_per_hit_chars
    2. 单条截断     超 per_hit_chars 的块被截断并记 truncated_hits（它仍然在包里）
    3. 拼包         用 BLOCK_SEPARATOR 连成一段
    4. 丢尾         整包超预算 → 从**最后一条**开始整条丢，记进 dropped_hits
    5. 至少保一条   只剩一条还超预算 → 把它截到 max_chars，**返回它而不是返回空**
    ```

    第 5 条是这一层最重要的护栏：**"预算紧"不等于"没有上下文"**。
    返回空上下文会让 RAG 链路走进"检索为空"的分支，于是模型一次都不被调用
    ——一个因为打包而发生的静默降级，看起来与"库里没有相关内容"一模一样。

    第 2 步与第 5 步的截断**只记第 2 步**：``truncated_hits`` 的定义是
    "被单条上限截断的条数"。第 5 步那次截断的证据在
    ``char_count == max_chars`` 上（包被削到刚好占满）。

    空 ``hits`` 返回空上下文而不是异常（第 6 条规则）。但**配置检查在它之前**：
    预算配错的时候，"这次恰好没有命中"不该让这个错误被跳过——
    配错的预算会在下一次有命中的时候以同样的方式错下去。

    ``max_chars`` / ``per_hit_chars`` 与 ``min_hit_chars`` 用**同一个度量单位**：
    默认是字符数；注入 ``measurer`` 之后三者都变成那个 measurer 的单位
    （例如 token 计数）。**它们必须同单位**，否则"预算放不放得下一条"
    这个比较就没有意义——而它恰恰是这一层唯一会拒绝服务的判据。
    """
    measure = measurer or _char_measure
    budget = _resolve_limit(
        "max_chars",
        max_chars,
        settings.retrieval_max_context_chars,
        hint="请调大 settings.retrieval_max_context_chars",
    )
    cap = _resolve_limit(
        "per_hit_chars",
        per_hit_chars,
        settings.retrieval_per_hit_chars,
        hint="请调大 settings.retrieval_per_hit_chars",
    )
    floor = _resolve_min_hit_chars(min_hit_chars)
    if budget < floor:
        raise ContextError(
            f"一条命中至少需要 {floor} 字，本预算只给了 {budget} 字："
            "预算放不下任何一条命中，而'放不下'不该被静默处理成'空上下文'——"
            "空上下文会让整条 RAG 链路走进'检索为空'的分支（一次 LLM 都不调），"
            "于是一次配置错误被伪装成'库里没有相关内容'。"
            "出路：调大 settings.retrieval_max_context_chars，"
            f"或把单条下限从 {floor} 降到能放下的值。"
        )

    blocks: list[str] = []
    cut: list[bool] = []
    for marker, hit in enumerate(hits, start=1):
        block = _render_block(hit, marker)
        capped = _truncate(block, cap, measure)
        cut.append(capped != block)
        blocks.append(capped)

    # 从尾部丢：入参已经是检索器的名次顺序（越靠后越不相关），
    # 因此"最后一条"就是最该被牺牲的那一条。丢整条而不是继续缩短，
    # 是因为半截片段进上下文只是噪声（见 types.DEFAULT_MIN_HIT_CHARS）。
    while len(blocks) > 1 and measure(BLOCK_SEPARATOR.join(blocks)) > budget:
        blocks.pop()

    if blocks and measure(blocks[0]) > budget:
        # 至少保一条：把仅剩的那条截到预算（这一条**不计**进 truncated_hits，
        # 它的证据是 char_count == max_chars）。
        blocks[0] = _truncate(blocks[0], budget, measure)

    used = tuple(hits[: len(blocks)])
    dropped = tuple(hits[len(blocks) :])
    text = BLOCK_SEPARATOR.join(blocks)
    citations = tuple(_citation_of(hit, marker) for marker, hit in enumerate(used, start=1))
    return PackedContext(
        text=text,
        citations=citations,
        used_hits=used,
        dropped_hits=dropped,
        truncated_hits=sum(cut[: len(blocks)]),
        char_count=measure(text),
        max_chars=budget,
    )


# --------------------------------------------------------------------------- #
# 渲染与裁剪
# --------------------------------------------------------------------------- #


def _render_block(hit: RetrievalHit, marker: int) -> str:
    """把一条命中渲染成 ``[n] 来源 › 标题路径`` + 换行 + 正文.

    头部直接复用 ``RetrievalHit.citation``：**"来源与标题怎么回落"
    只该有一份实现**（``source`` → ``doc_id`` → ``record_id``）。
    这里不重写它，也就不会与检索器那边的展示分家。

    ``heading_path`` 为空时那一段自然消失（``citation`` 已经处理）；
    ``text`` 为空时只留头部，而不是补一行空洞的换行。
    """
    header = hit.citation(marker)
    if not hit.text:
        return header
    return f"{header}\n{hit.text}"


def _citation_of(hit: RetrievalHit, marker: int) -> Citation:
    """把一条命中折成 ``Citation``（``source`` 的回落规则与头部同源）."""
    return Citation(
        marker=marker,
        record_id=hit.record_id,
        source=_source_of(hit),
        heading_path=hit.heading_path,
        score=hit.score,
    )


def _source_of(hit: RetrievalHit) -> str:
    """这条命中的"来源"（``source`` → ``doc_id`` → ``record_id`` 三级回落）.

    与 ``RetrievalHit.citation`` 里的回落顺序**逐条相同**：
    提示词里的头部与 ``Citation.to_dict()`` 里的 source 必须是同一个值，
    否则"答案里的 [2] 指哪份文档"会取决于读的是哪一份数据。
    """
    return str(hit.metadata.get("source") or hit.metadata.get("doc_id") or hit.record_id)


def _truncate(text: str, limit: int, measure: Callable[[str], int]) -> str:
    """把 ``text`` 截到"按 ``measure`` 量不超过 ``limit``"，并补上截断标记.

    默认口径下它就是一次切片（``len`` 与字符一一对应）；而 ``measurer``
    是注入的任意函数时，**"切到第几个字符"与"量出来是多少"不再成正比**
    （例如长度除以 4 的 token 口径：24 个中文字约 6 个 token，
    而 24 个英文字符约 6 个 token 也不是同一个数）。因此这里用
    **二分找最长的可行前缀**，而不是"按字符切一刀再看量得过去吗"：

    ```text
    可行前缀    text[:n] + 标记 量得过去 → n 可以更大（测量单调不减）
    不可行前缀  量超了 → n 必须更小
    返回值      **一定**是量过的那一个（循环里测过才取）；测不出来的只有空串
    ```

    二分而不是逐字符回收的理由有两条：一是注入口径下能把预算**尽量填满**
    （逐字符回收会明显少填，例如 token 口径下只填了四分之一）；
    二是调用次数从 O(n) 降到 O(log n)——同一段文本要试的次数与长度无关。

    假设 ``measure`` 对前缀长度单调不减（字符数、token 数都满足）。
    即使这个假设不成立也不会切出超限的片段：**返回值在循环里被实测过**，
    最多是"这次截得比理论上更短或更长一点"，而不会破坏预算这条硬约束。
    """
    if measure(text) <= limit:
        return text
    marker = TRUNCATION_MARKER
    if measure(marker) <= limit:
        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            if measure(text[:middle] + marker) <= limit:
                low = middle
            else:
                high = middle - 1
        candidate = text[:low] + marker
        if measure(candidate) <= limit:
            return candidate
    # 连标记都放不下：退化成"按测量口径能放下多少算多少"（可能返回空串）。
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if measure(text[:middle]) <= limit:
            low = middle
        else:
            high = middle - 1
    return text[:low]


def _resolve_limit(label: str, value: int | None, default: int, *, hint: str) -> int:
    """解析并校验一个"必须 >= 1 的整数上限"（``None`` → settings 里的默认值）."""
    resolved = default if value is None else value
    if not isinstance(resolved, int) or isinstance(resolved, bool):
        raise ContextError(
            f"{label} 必须是整数，收到 {type(resolved).__name__}（值为 {resolved!r}）："
            f"它是打包预算，{hint} 或显式传一个整数。"
        )
    if resolved < 1:
        raise ContextError(
            f"{label}={resolved}：为 0 的上限意味着**没有任何片段能进上下文**，"
            "而那会让一次正常的检索退化成一个空上下文（于是 LLM 一次都不被调用）。"
            f"出路：{hint}；要'不限'请给一个足够大的正数，不要给 0。"
        )
    return resolved


def _resolve_min_hit_chars(value: int) -> int:
    """校验单条命中的最低字数（它只参与预算校验，不参与单条截断）.

    与 ``per_hit_chars`` 的关系值得写清楚：``min_hit_chars`` 回答的是
    "**整包预算至少要能放下一条**"（否则这一层根本不该被调用），
    而 ``per_hit_chars`` 是"单条硬上限"。两者冲突时（上限更小）
    以硬上限为准——截断是明确可见的（``truncated_hits``），
    而为了下限去放宽上限会静默地违背调用方给的数字。

    它的单位与 ``max_chars`` 相同（默认 chars，注入 ``measurer`` 即随之改变）：
    把 token 口径的预算配上字符口径的下限，会得到一个看似宽松、
    实际上永远为真的比较——那样这条下限就白写了。
    """
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ContextError(
            f"min_hit_chars 必须是 >= 1 的整数，收到 {value!r}："
            "它是'单条命中至少要有多少字'，0 会让'预算够不够'这个判断失去尺度"
            "（任何预算都算够）。要放宽请显式调大预算，而不是取消这条下限。"
        )
    return value


__all__ = [
    "BLOCK_SEPARATOR",
    "TRUNCATION_MARKER",
    "Citation",
    "PackedContext",
    "pack_context",
]
