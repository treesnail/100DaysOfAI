"""分块预算的度量单位：在切之前先回答"谁来数"（M6-D2）.

四种分块策略里，有三种的公共参数是"一块最多装多少"。这句话在写下来的时候
就已经埋了一个未定义的东西：**多少什么？**

```text
max_tokens = 320      # 320 个什么？
                      #   token？字符？汉字？字节？
```

这个坑非常具体：一份中文技术文档，同一个 ``320`` 在不同度量下差出 2~3 倍——
``cl100k_base`` 下常见汉字约 1~1.5 token/字，于是"320 token"大约 250 个汉字；
而"320 字符"就是 320 个汉字。**两者差 28%，而参数名是一样的。**
写文档的人按 token 想，读文档的人按字符猜，最后表现为"检索出来的片段太碎/太糊"。

因此本模块把度量单位**显式化成一个可注入的对象**：

```python
class TokenMeasurer(Protocol):
    name: str
    def count(self, text: str) -> int: ...
```

## 为什么默认是 ``chars`` 而不是 ``tiktoken``

直觉上应该默认用真实分词器（day034 的 ``TokenCounter`` 就是干这个的）。
但分块有一条第 6 天就立下的纪律：**同一份输入要在任何机器上得到同一份输出**
（day057 的可复现版本链、day058 的确定性 run_id、day061 的排序遍历）。
``tiktoken`` 的可用性依赖本机是否有词表缓存：

```text
有缓存 → 精确 token 数         → 分块结果 A
没缓存 → 字符级估算（day034）  → 分块结果 B
```

于是同一份文档在两台机器上切出**两套不同的 chunk_id**，而知识库是按
chunk_id 去重的——"这条知识为什么重复了两份"会变成一个查不出来的问题。
字符度量器没有这个性质：它只依赖 ``len()``，永远给同一个答案。

所以本包的分工是：

| 度量器 | 确定性 | 与计费单位一致 | 默认 |
|--------|--------|---------------|------|
| ``chars`` | **是**（只依赖 ``len``） | 否 | **是** |
| ``tiktoken`` | 否（依赖本机词表缓存） | 是 | 否 |

要按真实 token 预算时显式切到 ``tiktoken``，并接受"换台机器结果可能变"——
这一点写在 ``ChunkSet.token_measurer`` 里跟着产物走，**排查时能直接看到
这批块是按什么单位切的**（与 day061 把能力边界写进元数据是同一条纪律）。

## 二分：``longest_prefix_within``

"从某个位置起最多能切多长"是四种策略共用的一个原子操作。实现用二分，
而二分的合法性依赖一条**必须被说出来的假设**：

> ``count(text[:k])`` 关于 ``k`` **单调不减**。

它在两个度量器上都成立（字符数当然单调；BPE 编码的前缀 token 数也不会
因为多了一个字符而变少），但它不是"显然"的——如果将来换一个会在
分块边界上"压缩"的度量器（比如带词干还原的计数），二分就会给出错误答案
且**不报错**。相关的测试是 ``test_prefix_count_is_monotonic``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from smart_research_agent.chunking.errors import ChunkingError
from smart_research_agent.llm.tokenizer import TokenCounter

#: 字符度量器：1 个字符 = 1 个预算单位。确定性来自"它只调用 ``len``"。
MEASURER_CHARS = "chars"

#: 真实分词器度量器：按模型对应的 tiktoken encoding 计数。
MEASURER_TIKTOKEN = "tiktoken"

#: 可用的度量器（端点与自述表按这个顺序列出，因此它也是"文档的列序"）。
MEASURERS: tuple[str, ...] = (MEASURER_CHARS, MEASURER_TIKTOKEN)


@runtime_checkable
class TokenMeasurer(Protocol):
    """预算度量器：把一段文本折算成"预算单位"的个数.

    用 ``Protocol`` 而不是抽象基类，因为**实现它的东西不需要继承任何东西**
    （与 day040 的 ``supports_vision``、day041 的 ``embed_batch`` 默认实现
    是同一取向：能少一层耦合就少一层）。
    """

    name: str

    def count(self, text: str) -> int:
        """这段文本占多少个预算单位."""
        ...


@dataclass(frozen=True)
class CharMeasurer:
    """字符度量器：1 个字符 = 1 个预算单位（默认）.

    它看起来"没用真实分词器那么专业"，但它换来的是**可复现**：
    同一份文档在任何机器、任何 Python 版本上切出完全一样的块。
    对教学与测试而言这是最重要的性质——**一个会随环境变化的基线，
    没法用来讨论'改了这个参数以后变好了还是变坏了'。**
    """

    name: str = MEASURER_CHARS

    def count(self, text: str) -> int:
        return len(text)


@dataclass
class TiktokenMeasurer:
    """真实分词器度量器：按模型对应的 encoding 计数（day034 的 ``TokenCounter``）.

    ``model`` 留空时用 ``TokenCounter`` 的默认 encoding（``cl100k_base``）。
    构造时**不检查**分词器是否可用——可用性由 ``backend`` 暴露、由
    ``resolve_measurer`` 决定要不要拒绝，这与 day045 ``LocalModel``
    "构造期不探活"是同一条：**能建出来的对象不该因为环境而无法构造。**
    """

    model: str | None = None
    name: str = MEASURER_TIKTOKEN
    _counter: TokenCounter = field(default_factory=TokenCounter, init=False, repr=False)

    @property
    def backend(self) -> str:
        """``"tiktoken"``（精确）或 ``"fallback"``（day034 的字符级估算）."""
        return self._counter.backend

    def count(self, text: str) -> int:
        return self._counter.count_tokens(text, self.model)


def resolve_measurer(name: str, *, model: str | None = None) -> TokenMeasurer:
    """按名字建一个度量器；不可用时**拒绝而不是静默降级**.

    这里有一处刻意的严格：``tiktoken`` 在本机不可用时抛 ``ChunkingError``，
    而不是悄悄退回字符度量。理由是本模块最上面那条纪律的镜像面——

    > 静默降级会让"我要的是 token 预算"变成"我拿到的是字符预算"，
    > 而两者差 28%，且结果里看不出来。

    调用方要的就是"要么按 token 切，要么明确告诉我切不了"。
    """
    if name == MEASURER_CHARS:
        return CharMeasurer()
    if name == MEASURER_TIKTOKEN:
        measurer = TiktokenMeasurer(model=model)
        if measurer.backend != "tiktoken":
            raise ChunkingError(
                "本机没有可用的 tiktoken 词表缓存，无法按真实 token 计数（"
                "day034 的 TokenCounter 会退化成字符级估算）。"
                "改用 measurer='chars'，或在有网络的环境先预热词表。"
            )
        return measurer
    raise ChunkingError(f"未知度量器 {name!r}，可选 {', '.join(MEASURERS)}")


def describe_measurers() -> list[dict[str, str]]:
    """度量器的自述表（端点 ``GET /chunking/strategies`` 直接读它）.

    ``deterministic`` 一列是本表存在的理由：**它决定了这批块的 chunk_id
    能不能跨机器对齐**，而 chunk_id 是知识库的去重键。
    """
    return [
        {
            "name": MEASURER_CHARS,
            "unit": "1 个字符",
            "deterministic": "yes",
            "consistent_with_billing": "no",
            "note": "默认。只依赖 len()，任何机器上结果一致；"
            "中文文档下与 token 数相差约 1.0~1.5 倍",
        },
        {
            "name": MEASURER_TIKTOKEN,
            "unit": "1 个 token（模型词表）",
            "deterministic": "no",
            "consistent_with_billing": "yes",
            "note": "按 cl100k_base / o200k_base 精确计数，依赖本机词表缓存；"
            "缓存缺失时 resolve_measurer 直接拒绝，不做静默降级",
        },
    ]


def longest_prefix_within(text: str, budget: int, measurer: TokenMeasurer) -> int:
    """返回最长的 ``k``，使 ``measurer.count(text[:k]) <= budget``.

    实现是二分，因此依赖模块 docstring 里那条假设（``count`` 对前缀长度
    单调不减）。两个出口都是明确的：

    - ``k == len(text)``：整段放得下；
    - ``k == 0``：**连第一个字符都放不下**（``budget`` 太小）。
      这个情形不能返回"随便切一个字符"——那会让调用方以为预算是够的，
      于是产生一批每一块都超预算的结果。调用方必须显式处理它。
    """
    if budget < 0:
        raise ChunkingError(f"预算不能为负数，收到 {budget}")
    if not text:
        return 0
    if measurer.count(text) <= budget:
        return len(text)
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if measurer.count(text[:mid]) <= budget:
            low = mid
        else:
            high = mid - 1
    return low


def count_tokens(text: str, measurer: TokenMeasurer) -> int:
    """一行封装：把度量器用在单段文本上（报告与 chunk 都按它记数）."""
    return measurer.count(text) if text else 0


__all__ = [
    "MEASURERS",
    "MEASURER_CHARS",
    "MEASURER_TIKTOKEN",
    "CharMeasurer",
    "TiktokenMeasurer",
    "TokenMeasurer",
    "count_tokens",
    "describe_measurers",
    "longest_prefix_within",
    "resolve_measurer",
]
