"""分块器接口、策略参数与收尾器：四种策略的公共骨架（M6-D2）.

四种分块策略听起来是四套东西，实际上**只有一个问题不同**：

```text
在哪里切？
  fixed       每 max_tokens 个单位切一刀（不看内容）
  recursive   优先在语义最强的分隔符处切（段落 > 换行 > 句号 > 逗号）
  structural  在标题层级处切（用 day061 保留下来的 blocks）
  semantic    在相邻段落相似度最低处切（用 embedding）
```

**切完之后的事，四者完全一样**：

```text
收尾（finalize）
  1. 修剪       去掉每段首尾的空白，并调整区间（否则块文本会以空白开头）
  2. 合并       过短的块（< min_tokens）与相邻块合并，避免碎块污染检索
  3. 校验       区间合法 + 起点严格递增 + 「原文非空白字符全被覆盖」
  4. 编号       index = 0..n-1（序号是身份的一部分，见 types.py）
  5. 度量       用同一个度量器算 token_count
  6. 成块       make_chunk → Chunk → ChunkSet
```

因此本模块用**模板方法**把它们钉在一起：子类只实现 ``_spans()`` 返回一串
**字符区间 + 切在这里的理由**，``finalize()`` 统一收尾。这样做的好处不是
少写几行代码，而是**四条纪律只写一次**：

> 一个策略如果自己收尾，它就会自己漏掉一条；而"漏掉哪一条"没有任何报错，
> 只表现为某个指标悄悄变差。

## 收尾里最重要的一件事：覆盖率是被接口保证的

``_assert_coverage`` 会逐个字符核对：**原文里每一个非空白字符都必须至少
落在一个块里**。这不是"测出来的性质"，而是"接口拒绝违反它"——

```text
切丢了内容 → ChunkingError，而不是一个 coverage=0.94 的报告
```

day061 用"逐行核对"守住了"只分类不丢弃"；今天用"逐字符核对"守住
"只搬运不丢弃"。两者都在同一个位置：**结构化处理最容易犯的错就是让内容
悄悄消失，而它的表现只是"结果比原文短"。**

## 参数默认值是策略的一部分

``overlap_tokens`` 在固定长度与递归策略下是必要的（切在句子中间时，
两侧都读不到完整的句子），在结构策略下**是错的**（标题边界本身
就是语义边界，跨边界重叠会把上一节的内容灌进下一节）。因此默认值
不能是一张全局表，而必须是 ``DEFAULT_POLICIES`` 这张**按策略给出的表**，
并且结构策略显式地把重叠设为 0——**"这个参数对我没意义"和"这个参数
默认为 0"是两件不同的事，要能说出是哪一件。**
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from smart_research_agent.chunking.errors import ChunkingError, UnsupportedStrategy
from smart_research_agent.chunking.tokens import (
    MEASURER_CHARS,
    MEASURERS,
    TokenMeasurer,
    longest_prefix_within,
    resolve_measurer,
)
from smart_research_agent.chunking.types import (
    STRATEGIES,
    STRATEGY_FIXED,
    STRATEGY_RECURSIVE,
    STRATEGY_SEMANTIC,
    STRATEGY_STRUCTURAL,
    Chunk,
    ChunkSet,
    chunk_id_for,
)
from smart_research_agent.documents.types import BLOCK_HEADING, Document
from smart_research_agent.llm.embedding import EmbeddingProvider
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: 分块这件事**做不到什么**。每条都带一个可核对的依据——
#: 与 day059 模型卡、day060 部署限制、day061 解析边界同一条纪律：
#: **无法否掉结论的限制等于免责声明。**
CHUNKING_LIMITATIONS: tuple[str, ...] = (
    "覆盖率保证的是「原文的每个非空白字符至少属于一个块」，不是「每块都是一句"
    "完整的话」：固定长度策略必然会切在句子中间，这是它的定义性行为",
    "结构策略的 start_block / end_block 只在「小节能被精确定位」时才有值；"
    "定位失败的小节退化为按段落切分，此时两个字段记 -1 而不是一个猜出来的值",
    "语义策略只认识「自然段」（原文里被空行分开的段），它不看块类型："
    "一个含空行的代码块会被它从中间切开，要保护代码块请用结构策略",
    "``tiktoken`` 度量器依赖本机词表缓存，同一份文档在不同机器上可能切出"
    "不同的 chunk_id；要用它就必须接受「分块结果不可跨机器对齐」",
    "重叠（overlap）只在 fixed / recursive / semantic 三种策略下生效，"
    "结构策略显式拒绝非零重叠：跨标题的重叠会把上一节内容灌进下一节",
)

#: 明确排除的用途（比"不适用于生产"具体得多）。
CHUNKING_OUT_OF_SCOPE: tuple[str, ...] = (
    "指望某一种策略在所有语料上最优：四种策略的优劣取决于查询分布，"
    "选型必须用自己的查询集跑 /chunking/evaluate 之后再决定",
    "把分块当作「语义理解」：四种策略里只有 semantic 看内容相似度，"
    "而它用的 embedding 若没有真实语义（如离线的 char-ngram），"
    "切出来的边界只是「字面差异最大的地方」",
    "用分块结果做「原文重建」：带重叠的块集合重建原文会得到重复内容，"
    "重建请用 day061 的 Document.text",
)


#: 被当作"天然断点"的字符（重叠回带的落点会对齐到它们）。
#: 中英文标点都在表里：只放 `"\n"` 会让中文段落的重叠永远落在句子中间，
#: 而那种重叠看起来只是"多带了一点前面的话"，实际上把上一句切成了两半。
BOUNDARY_CHARS = "\n。！？；：，、" + ".!?;:, "


def snap_to_boundary(
    text: str, low: int, high: int, boundary_chars: str = BOUNDARY_CHARS
) -> int:
    """在 ``[low, high)`` 里找第一个"紧跟断点字符之后"的位置，找不到就用 ``low``.

    它服务的是重叠（overlap）的落点：重叠本身是"多带一点上文"，
    如果带过来的上文是从一个句子中间开始的，那一句在两侧各自断了一半——
    **重叠的目的是让两侧都能读到完整的句子，切在句子中间等于白带。**

    ``high`` 之前找不到断点就退回 ``low``：宁可重叠落点不完美，
    也不要为了对齐而把重叠长度翻倍（那会连带影响成本）。

    还有一条比"对齐断点"更容易被忽略的偏好：**落点应当是内容而不是空白**。
    两个块之间隔着一段空行时，``low`` 很可能就落在那段空行里——
    回带过来的是"两个换行"，它既没有语义，又会让引用看起来像是
    从半空中开始的。因此这里优先返回"紧跟断点、且本身不是空白"的位置，
    退一步才接受"任意非空白位置"，最后才退回 ``low``。
    """
    if low < 0 or high > len(text) or high <= low:
        return max(0, min(low, len(text)))
    fallback = -1
    for position in range(max(low, 1), high):
        if text[position].isspace():
            continue
        if fallback < 0:
            fallback = position
        if text[position - 1] in boundary_chars:
            return position
    return fallback if fallback >= 0 else low


#: "自然段"的分界：连续**两个及以上**换行。一个换行只是排版折行
#: （day061 的 PDF 加载器就是按行距推段落的），两个才是段落。
PARAGRAPH_GAP = re.compile(r"\n{2,}")


def paragraph_spans(text: str, start: int = 0, end: int | None = None) -> list[tuple[int, int]]:
    """把 ``[start, end)`` 按空行切成自然段的字符区间（结构/语义策略共用）.

    刻意用正则而不是 ``text.split("\\n\\n")``：split 会把**偏移量丢掉**，
    而这一层所有产物的身份都建立在偏移量上（``start_char`` 是引用溯源的锚点）。
    用 ``finditer`` 拿到每个分隔段的位置，段落区间就是"上一个分隔段的结束"
    到"这一个分隔段的开始"之间。

    只有空白的段落会被丢掉：它们覆盖的是空白字符，而覆盖率核对
    只要求非空白字符被覆盖（见 ``_assert_spans``）。

    最后一段的**尾随空白也会被剪掉**（原文结尾一定有一个换行）。
    不剪的话，"最后一段"与其他段落在语义上就不一样了——
    例如三份完全相同的段落，前两段相似度是 1.0，而最后一段因为多了一个
    换行而略低，于是语义策略会**凭空在末尾切一刀**。这类偏差非常隐蔽：
    块数只多了一个，而报告里看不出原因。
    """
    if end is None:
        end = len(text)
    spans: list[tuple[int, int]] = []
    cursor = start
    for match in PARAGRAPH_GAP.finditer(text, start, end):
        if match.start() > cursor:
            spans.append((cursor, match.start()))
        cursor = match.end()
    if cursor < end:
        spans.append((cursor, end))
    kept: list[tuple[int, int]] = []
    for left, right in spans:
        while right > left and text[right - 1].isspace():
            right -= 1
        if right > left:
            kept.append((left, right))
    return kept


def heading_positions(document: Document) -> list[tuple[int, tuple[str, ...]]]:
    """标题在 ``document.text`` 里的偏移 + 它累积出来的标题路径.

    标题栈的规则与 ``structural.build_sections`` 完全一致（``level >= 当前``
    就弹栈），因为**"这一节叫什么"这个问题只能有一个答案**——两个模块
    各写一套，迟早会出现"结构策略说它在第 2 节、语义策略说它在第 3 节"。

    定位不到的标题**直接跳过**而不是报错：标题路径只是一个语境标签，
    为它让整次分块失败不值得；而且跳过它不会影响覆盖率
    （覆盖率由区块间的完整性保证，与标题定位无关）。
    """
    text = document.text
    stack: list[tuple[int, str]] = []
    found: list[tuple[int, tuple[str, ...]]] = []
    cursor = 0
    for block in document.blocks:
        if block.kind != BLOCK_HEADING:
            continue
        title = block.text.strip()
        while stack and stack[-1][0] >= block.level:
            stack.pop()
        stack.append((block.level, title))
        position = text.find(block.text, cursor)
        length = len(block.text)
        if position < 0:
            position = text.find(title, cursor)
            length = len(title)
        if position < 0:
            continue
        cursor = position + length
        found.append((position, tuple(item[1] for item in stack)))
    return found


def path_for_offset(
    offset: int, positions: list[tuple[int, tuple[str, ...]]]
) -> tuple[str, ...]:
    """某个偏移量所处的小节路径（``positions`` 必须按偏移升序）."""
    path: tuple[str, ...] = ()
    for position, candidate in positions:
        if position > offset:
            break
        path = candidate
    return path


@dataclass(frozen=True)
class ChunkPolicy:
    """一次分块的参数（四种策略共用一张表，默认值按策略区分）."""
    strategy: str = STRATEGY_RECURSIVE
    max_tokens: int = 320
    #: 重叠额度：**0 表示不重叠**。字段默认值刻意取 0，因为"默认重叠多少"
    #: 是一个**策略决定**（结构策略必须为 0，否则会把上一节内容灌进下一节），
    #: 所以它属于 ``DEFAULT_POLICIES`` 而不是数据类的字段默认值。
    #: 直接构造 ``ChunkPolicy`` 得到的是**最保守的一套**；要策略默认值
    #: 请用 ``default_policy(strategy)``。
    overlap_tokens: int = 0
    min_tokens: int = 0
    measurer: str = MEASURER_CHARS
    model: str | None = None
    #: **只有语义策略消费**这个参数：相邻段落相似度低于该分位数处切一刀。
    #: 其余三个策略把它原样带在产物里（``ChunkSet.metadata.policy``），
    #: 但不会用到它——因此它们的 ``describe()`` 里写着
    #: ``uses_similarity_percentile: false``。**"参数对我没意义"要说出来，
    #: 否则下一个人会以为调它没效果是因为分块器忽略了参数。**
    similarity_percentile: float = 0.25

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise UnsupportedStrategy(
                f"未知策略 {self.strategy!r}，可选 {', '.join(STRATEGIES)}"
            )
        if self.max_tokens < 1:
            raise ChunkingError(f"max_tokens 必须至少为 1，收到 {self.max_tokens}")
        if self.overlap_tokens < 0:
            raise ChunkingError(
                f"overlap_tokens 不能为负数，收到 {self.overlap_tokens}"
            )
        if self.overlap_tokens >= self.max_tokens:
            # 这条护栏挡的是一次真实的死循环：``step = max - overlap <= 0``
            # 时窗口原地踏步，切分永远不前进。**参数矛盾的报错应该在构造期，
            # 而不是在一次跑不完的循环里。**
            raise ChunkingError(
                f"overlap_tokens({self.overlap_tokens}) 必须小于 "
                f"max_tokens({self.max_tokens})：否则切分步长为 0，窗口原地踏步"
            )
        if self.min_tokens < 0:
            raise ChunkingError(f"min_tokens 不能为负数，收到 {self.min_tokens}")
        if self.min_tokens > self.max_tokens:
            raise ChunkingError(
                f"min_tokens({self.min_tokens}) 不能大于 max_tokens({self.max_tokens})："
                "那会让合并后的块必然超过预算"
            )
        if self.measurer not in MEASURERS:
            raise ChunkingError(
                f"未知度量器 {self.measurer!r}，可选 {', '.join(MEASURERS)}"
            )
        if not 0.0 <= self.similarity_percentile <= 1.0:
            # 分位数是一个比例。传 25 而不是 0.25 是最常见的笔误，
            # 而它的表现是"每次都在最低的那一刀切"，看起来像算法失效。
            raise ChunkingError(
                f"similarity_percentile 必须是 [0, 1] 的比例，收到 "
                f"{self.similarity_percentile}（想要 25% 请写 0.25）"
            )

    @property
    def step(self) -> int:
        """固定长度策略的窗口步长（``max_tokens - overlap_tokens``，恒为正）."""
        return self.max_tokens - self.overlap_tokens

    def replace(self, **overrides: Any) -> ChunkPolicy:
        """派生一份改了若干字段的策略（测试与端点都按它做参数覆盖）."""
        payload = asdict(self)
        unknown = set(overrides) - set(payload)
        if unknown:
            raise ChunkingError(
                f"未知参数 {', '.join(sorted(unknown))}；"
                f"可选 {', '.join(sorted(payload))}"
            )
        payload.update(overrides)
        return ChunkPolicy(**payload)

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典（含派生的 step）."""
        payload = asdict(self)
        payload["step"] = self.step
        return payload


#: 按策略给出的默认参数表（见模块 docstring 最后一节）。
DEFAULT_POLICIES: dict[str, dict[str, Any]] = {
    STRATEGY_FIXED: {"max_tokens": 320, "overlap_tokens": 48, "min_tokens": 0},
    STRATEGY_RECURSIVE: {"max_tokens": 320, "overlap_tokens": 48, "min_tokens": 0},
    # 结构策略的重叠**必须**为 0：标题边界就是语义边界（见模块 docstring）。
    STRATEGY_STRUCTURAL: {"max_tokens": 320, "overlap_tokens": 0, "min_tokens": 0},
    # 语义策略的最小块偏大：它一次切一段话题，碎块会让"话题"信息量不足。
    STRATEGY_SEMANTIC: {"max_tokens": 400, "overlap_tokens": 40, "min_tokens": 60},
}


def default_policy(strategy: str, **overrides: Any) -> ChunkPolicy:
    """按策略取一份默认参数（``overrides`` 覆盖其中若干项）."""
    if strategy not in DEFAULT_POLICIES:
        raise UnsupportedStrategy(
            f"未知策略 {strategy!r}，可选 {', '.join(STRATEGIES)}"
        )
    payload = {"strategy": strategy, **DEFAULT_POLICIES[strategy], **overrides}
    return ChunkPolicy(**payload)


@dataclass(frozen=True)
class Span:
    """切分器交给收尾器的中间产物：一段**原文的字符区间** + 切在这里的理由.

    刻意只带区间而不带文本：文本由 ``document.text[start:end]`` 现取。
    这样"块文本是原文的连续子串"这条不变量**不可能被违反**——
    因为没有第二个地方可以放文本（与 day061 ``make_document``
    删掉 ``text`` 参数是同一种做法：**让错误的东西无处可写**）。
    """

    start_char: int
    end_char: int
    reason: str = ""
    heading_path: tuple[str, ...] = ()
    start_block: int = -1
    end_block: int = -1
    oversized: bool = False


class Chunker(ABC):
    """分块器基类：子类只实现 ``_spans``，收尾统一由 ``finalize`` 做."""

    #: 策略名（与 ``types.STRATEGIES`` 一一对应）.
    name: str = ""

    #: 重叠是否由收尾器统一施加（见 ``finalize`` 的顺序说明）.
    #: 递归与语义策略为真；固定长度策略在自己的窗口里带重叠（假）；
    #: 结构策略恒不重叠（它的 ``__init__`` 会拒绝非零重叠）。
    apply_overlap: bool = False

    def __init__(
        self,
        policy: ChunkPolicy | None = None,
        *,
        measurer: TokenMeasurer | None = None,
        embedding: EmbeddingProvider | None = None,
    ) -> None:
        self.policy = policy or default_policy(self.name)
        if self.policy.strategy != self.name:
            raise ChunkingError(
                f"{type(self).__name__} 只处理 {self.name!r} 策略，"
                f"收到 {self.policy.strategy!r}"
            )
        # ``measurer`` 显式注入优先；否则按策略里的度量器名解析。
        # 测试注入 ``CharMeasurer`` 即可让四种策略全部离线、确定性可复现。
        self.measurer = measurer or resolve_measurer(
            self.policy.measurer, model=self.policy.model
        )
        self.embedding = embedding
        #: 切分器**实际使用的预算**与它留给重叠的额度。两个属性存在的理由
        #: 是让 ``max_tokens`` 在四种策略下含义统一为"**一个块最多多少**"：
        #:
        #: ```text
        #: fixed      窗口就是 max_tokens，重叠靠"下一刀往前挪"实现（窗口内自带重叠）
        #: recursive  先按 max_tokens - overlap 切，再回带 overlap → 合计不超过 max_tokens
        #: semantic   同 recursive
        #: structural 重叠恒为 0，窗口就是 max_tokens
        #: ```
        #:
        #: 没有这一层会出现一个很隐蔽的问题：重叠让块**超出**预算，而超预算
        #: 会被 ``_make_chunk`` 记成 ``oversized``——于是"为了不丢句子而回带
        #: 的重叠"看起来像"有一个原子块没切开"。**同一个字段表达两件事，
        #: 报告就再也说不清是哪种。**
        self.window_budget: int = self.policy.max_tokens
        self.window_overlap: int = 0
        #: 子类在 ``_spans`` 里写下的"这一次切分发生过什么"（段落数、
        #: 相似度断层数、退化的小节数……）。它跟着 ``ChunkSet.metadata``
        #: 走进产物：**报告要能回答"为什么这批块长这样"，
        #: 而不只是"这批块长什么样"。**
        self.stats: dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # 子类接口
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _spans(self, document: Document) -> list[Span]:
        """把文档切成若干字符区间（**不做收尾**，交给 ``finalize``）."""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """策略自述：单位、参数、强项、弱点、确定性来源（端点直接读它）."""

    # ------------------------------------------------------------------ #
    # 模板方法
    # ------------------------------------------------------------------ #

    def split(self, document: Document) -> ChunkSet:
        """切一份文档：``_spans`` + ``finalize``（唯一的公开入口）."""
        spans = self._spans(document)
        result = self.finalize(document, spans)
        logger.info("分块完成：%s | %s", result.summary_line(), document.source)
        return result

    def finalize(self, document: Document, spans: Sequence[Span]) -> ChunkSet:
        """收尾：修剪 → 合并 → 校验 → 编号 → 度量 → 成块（见模块 docstring）."""
        text = document.text
        if not text.strip():
            # 空文档返回**空集合**而不是抛错：它不是一个错误，是一个结论
            # （day061 的"解析结果为空"就是这种情形）。抛错会让批量分块
            # 因为一份空文件而整体中断，而真正需要的是报告里一条"0 块"。
            logger.warning("%s 没有可切的内容，返回空块集", document.source)
            return ChunkSet(
                doc_id=document.fingerprint,
                source=document.source,
                strategy=self.name,
                token_measurer=self.measurer.name,
                doc_chars=0,
                metadata={"policy": self.policy.to_dict(), "empty": "true", **self.stats},
            )
        trimmed = [span for span in (self._trim(text, span) for span in spans) if span]
        # 收尾的固定顺序：**修剪 → 重叠 → 合并 → 校验**。
        # 重叠必须排在修剪之后：重叠会把起点往回挪，而挪过之后可能落进空白，
        # 若先重叠再修剪，两个块会被修剪到同一个位置（起点不再严格递增）。
        # ``apply_overlap`` 为真的策略（递归/语义）在这一步统一回带；
        # 固定长度策略在自己的窗口里实现重叠（``window_overlap``），因此为假。
        if self.apply_overlap:
            trimmed = self._with_overlap(text, trimmed)
        cleaned = self._dedupe_starts(trimmed)
        merged = self._merge(text, cleaned)
        self._assert_spans(text, merged)
        chunks = [self._make_chunk(document, span, index) for index, span in enumerate(merged)]
        result = ChunkSet(
            doc_id=document.fingerprint,
            source=document.source,
            strategy=self.name,
            chunks=tuple(chunks),
            token_measurer=self.measurer.name,
            doc_chars=len(text),
            metadata={"policy": self.policy.to_dict(), **self.stats},
        )
        if result.oversized_count:
            logger.warning(
                "%s 有 %d 个块超过预算且不可切（原子块，见 CHUNKING_LIMITATIONS）",
                document.source,
                result.oversized_count,
            )
        return result

    # ------------------------------------------------------------------ #
    # 收尾的六步（每一步都可以单独测）
    # ------------------------------------------------------------------ #

    def _trim(self, text: str, span: Span) -> Span | None:
        """去掉区间首尾的空白，并同步移动偏移；全是空白则丢弃.

        不修剪的后果很具体：块文本以 ``"\\n\\n"`` 开头，它在 embedding 里
        贡献不了任何语义，却会让"引用里显示的那段话"看起来像是从半空中开始的。
        """
        start, end = span.start_char, span.end_char
        start = max(0, min(start, len(text)))
        end = max(start, min(end, len(text)))
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end <= start:
            return None
        return Span(
            start_char=start,
            end_char=end,
            reason=span.reason,
            heading_path=span.heading_path,
            start_block=span.start_block,
            end_block=span.end_block,
            oversized=span.oversized,
        )

    def _dedupe_starts(self, spans: list[Span]) -> list[Span]:
        """起点相同的块只保留最后一个（区间有序且终点递增时，后者包含前者）.

        它挡的是一种很隐蔽的情形：两个相邻区间之间只隔着一小段空白，
        修剪（或重叠回带）之后两条起点撞到同一个位置。此时前一块的内容
        几乎完全被后一块包含，留着它只会带来**重复内容与一次多余的
        embedding 调用**，而 ``_assert_spans`` 会把它当成"窗口没有前进"报错。

        触发时会记一条告警：**能自愈不等于不该知道**——如果告警频繁出现，
        说明真正的病因在重叠比例（``2 × overlap >= max_tokens``），
        而不是这一层。
        """
        cleaned: list[Span] = []
        for span in spans:
            if (
                cleaned
                and span.start_char <= cleaned[-1].start_char
                and span.end_char >= cleaned[-1].end_char
            ):
                # 只有"后一块完全包含前一块"时才自愈。其它情形（起点倒退、
                # 后一块更短）是真正的切分缺陷，**必须让 ``_assert_spans`` 响亮地报错**
                # ——自愈会把"窗口没有前进"这种病变成一堆悄无声息的短块。
                dropped = cleaned.pop()
                logger.warning(
                    "块起点重复（%d），丢弃被包含的那一块 [%d:%d]",
                    span.start_char,
                    dropped.start_char,
                    dropped.end_char,
                )
            cleaned.append(span)
        return cleaned

    def _merge(self, text: str, spans: list[Span]) -> list[Span]:
        """把过短的块与后一块合并，直到达到 ``min_tokens``.

        两条边界：

        - **合并不越过预算**：合并后超过 ``max_tokens`` 就停下。宁可留一个
          短块，也不要造一个注定被模型截断的块；
        - **不合并原子块**：``oversized`` 的块本身已经超预算，把它再并进来
          只会得到更大的块——它需要的是被单独标记，而不是被稀释。

        合并的块文本仍然取自原文区间，因此"块是原文连续子串"不受影响。
        """
        if self.policy.min_tokens <= 0 or len(spans) <= 1:
            return spans
        merged: list[Span] = []
        for span in spans:
            if (
                merged
                and not merged[-1].oversized
                and not span.oversized
                and merged[-1].heading_path == span.heading_path
                and self.measurer.count(text[merged[-1].start_char : merged[-1].end_char])
                < self.policy.min_tokens
                and self.measurer.count(text[merged[-1].start_char : span.end_char])
                <= self.window_budget
            ):
                previous = merged.pop()
                merged.append(
                    Span(
                        start_char=previous.start_char,
                        end_char=span.end_char,
                        reason=f"{previous.reason}+merge",
                        heading_path=previous.heading_path,
                        start_block=previous.start_block,
                        end_block=span.end_block,
                        oversized=False,
                    )
                )
                continue
            merged.append(span)
        return merged

    def _assert_spans(self, text: str, spans: Sequence[Span]) -> None:
        """区间合法性 + 起点严格递增 + 非空白字符全覆盖（见模块 docstring）."""
        if not spans:
            raise ChunkingError(
                "切分没有产出任何块：原文非空却切不出块，说明切分逻辑丢掉了内容"
            )
        for span in spans:
            if span.start_char < 0 or span.end_char > len(text):
                raise ChunkingError(
                    f"区间越界：[{span.start_char}:{span.end_char}] 不在 "
                    f"0..{len(text)} 之内"
                )
            if span.end_char <= span.start_char:
                raise ChunkingError(
                    f"区间为空：[{span.start_char}:{span.end_char}]（收尾应已把它丢掉）"
                )
        starts = [span.start_char for span in spans]
        if any(later <= earlier for earlier, later in zip(starts, starts[1:])):
            raise ChunkingError(
                f"块的起点必须严格递增，收到 {starts}：不递增意味着窗口没有前进，"
                "切分会在原地打转"
            )
        covered = bytearray(len(text))
        for span in spans:
            covered[span.start_char : span.end_char] = b"\x01" * (
                span.end_char - span.start_char
            )
        missing = [
            index
            for index, char in enumerate(text)
            if not covered[index] and not char.isspace()
        ]
        if missing:
            raise ChunkingError(
                f"有 {len(missing)} 个非空白字符没有被任何块覆盖"
                f"（首个位置 {missing[0]}）：切分丢了内容，"
                "这与 day061「只分类不丢弃」是同一条纪律"
            )

    def _make_chunk(self, document: Document, span: Span, index: int) -> Chunk:
        """由区间造一个块（``text`` 只能来自 ``document.text`` 的切片）."""
        text = document.text[span.start_char : span.end_char]
        return Chunk(
            chunk_id=chunk_id_for(document.fingerprint, self.name, index, text),
            doc_id=document.fingerprint,
            source=document.source,
            text=text,
            index=index,
            strategy=self.name,
            token_count=self.measurer.count(text),
            token_measurer=self.measurer.name,
            start_char=span.start_char,
            end_char=span.end_char,
            start_block=span.start_block,
            end_block=span.end_block,
            heading_path=span.heading_path,
            reason=span.reason,
            oversized=span.oversized
            or self.measurer.count(text) > self.policy.max_tokens,
        )

    # ------------------------------------------------------------------ #
    # 两个共用的切分原语
    # ------------------------------------------------------------------ #

    def _hard_split(self, text: str, start: int, end: int, reason: str) -> list[Span]:
        """按预算硬切（**所有分隔符都用完之后的最后手段**）.

        它会出现的地方比想象中多：一段没有标点的长串（哈希、日志行、
        压缩后的文本）在任何分隔符策略下都会退到这里。因此它必须
        永远前进——每刀至少一个字符，否则就是死循环。
        """
        spans: list[Span] = []
        cursor = start
        while cursor < end:
            length = self._longest_span(text, cursor, end)
            if length <= 0:
                raise ChunkingError(
                    f"预算 {self.window_budget} {self.measurer.name} 装不下 "
                    f"位置 {cursor} 的一个字符：请调大 max_tokens"
                )
            spans.append(Span(start_char=cursor, end_char=cursor + length, reason=reason))
            cursor += length
        return spans

    def _longest_span(self, text: str, start: int, end: int) -> int:
        """从 ``start`` 起不超过预算的最大字符数（用度量器二分）."""
        return longest_prefix_within(text[start:end], self.window_budget, self.measurer)

    def _fits(self, text: str, start: int, end: int) -> bool:
        """``text[start:end]`` 是否装得进预算（三个策略共用的判据）."""
        if end <= start:
            return True
        return self.measurer.count(text[start:end]) <= self.window_budget

    def _pack(self, text: str, spans: list[Span], reason: str = "+pack") -> list[Span]:
        """把相邻的小区间贪心地合并到预算上限（"尽量填满，但不超"）.

        不合并的后果很具体：递归策略在 ``\\n\\n`` 处切开后，每个自然段
        各自成块——一份 200 段的文档会切出 200 个不到 100 字的块。
        碎块对检索的伤害是双重的：**话题信息不足**（一块里只有一句话）
        且**索引条目暴涨**（embedding 调用量与向量库容量同步翻倍）。
        """
        if len(spans) <= 1:
            return spans
        packed: list[Span] = [spans[0]]
        for span in spans[1:]:
            previous = packed[-1]
            if (
                previous.heading_path == span.heading_path
                and not previous.oversized
                and not span.oversized
                and self._fits(text, previous.start_char, span.end_char)
            ):
                packed.pop()
                packed.append(
                    Span(
                        start_char=previous.start_char,
                        end_char=span.end_char,
                        # 理由只追加一次：连续合并会写出
                        # `sep:段落+pack+pack+pack…` 这种读不出信息的字符串，
                        # 而报告里那一列是要给人看的。
                        reason=(
                            previous.reason
                            if reason in previous.reason
                            else previous.reason + reason
                        ),
                        heading_path=previous.heading_path,
                        start_block=previous.start_block,
                        end_block=span.end_block,
                    )
                )
                continue
            packed.append(span)
        return packed

    def _with_overlap(self, text: str, spans: list[Span]) -> list[Span]:
        """给每块（首块除外）往回带一段上文，落点对齐到天然断点.

        实现方式是**把下一块的起点往回挪**，而不是"给上一块追加一段尾巴"：
        两者在文本上等价，但前者的每一块仍然是一个区间，
        于是"覆盖率核对"和"重复率计算"都还成立（见 types.ChunkSet）。

        三条护栏：

        - 不吞掉整块：往回挪之后起点仍必须**严格大于**上一块的起点；
        - 不超过重叠预算：最多回挪 ``overlap_tokens`` 个单位；
        - 落点对齐断点：否则重叠带过来的是一句被切成两半的话。
        另加一条与结构有关的：**不跨标题重叠**。两块的小节路径不同就不回带——
        否则上一节的内容会挂在新一节的 ``heading_path`` 下，形成一条
        错误的、而且看起来很合理的来源标签（结构策略因此直接禁止重叠）。
        """
        overlap = self.policy.overlap_tokens
        if overlap <= 0 or len(spans) <= 1:
            return spans
        overlapped: list[Span] = [spans[0]]
        for span in spans[1:]:
            previous_start = overlapped[-1].start_char
            if span.heading_path != overlapped[-1].heading_path:
                overlapped.append(span)
                continue
            low = max(previous_start + 1, span.start_char - overlap)
            if low >= span.start_char:
                # 两块贴在一起，没有回带空间（重叠额度相对块长偏大时会出现）：
                # 保持原样比"挪一点点"更好——后者的起点会撞上上一块。
                overlapped.append(span)
                continue
            start = min(
                span.start_char - 1,
                max(snap_to_boundary(text, low, span.start_char), low),
            )
            if text[start].isspace():
                # 落点仍落在空白里（回带区间整段都是空白）：不做重叠。
                # 带一段空行进块只会让引用看起来像从半空中开始。
                overlapped.append(span)
                continue
            overlapped.append(
                Span(
                    start_char=start,
                    end_char=span.end_char,
                    reason=span.reason + "+overlap",
                    heading_path=span.heading_path,
                    start_block=span.start_block,
                    end_block=span.end_block,
                    oversized=span.oversized,
                )
            )
        return overlapped

    def _back_off(self, text: str, start: int, end: int, reason: str) -> list[Span]:
        """在 ``[start, end)`` 上做"带重叠的窗口切分"（fixed 与退化路径共用）.

        ``window_overlap`` 的实现方式是**把下一块的起点往回挪**，
        而不是"给上一块追加一段尾巴"——两者在文本上等价，但前者保持
        "每一块都是一个区间"，于是覆盖率核对与去重计算都还成立。

        注意它使用的是 ``window_overlap`` 而不是 ``policy.overlap_tokens``：
        递归与语义策略的全局重叠由 ``_with_overlap`` 在最后统一施加，
        这里再叠一次就会**双重回带**（放大 40% 而不是 20%），
        而"多花了一点 embedding 钱"这种代价没人会去查。
        """
        spans: list[Span] = []
        cursor = start
        while cursor < end:
            length = self._longest_span(text, cursor, end)
            if length <= 0:
                raise ChunkingError(
                    f"预算 {self.window_budget} {self.measurer.name} 装不下 "
                    f"位置 {cursor} 的一个字符：请调大 max_tokens"
                )
            span_end = min(cursor + length, end)
            spans.append(Span(start_char=cursor, end_char=span_end, reason=reason))
            if span_end >= end:
                break
            advance = max(1, length - self.window_overlap)
            cursor += advance
        return spans


# ---------------------------------------------------------------------- #
# 注册表与工厂
# ---------------------------------------------------------------------- #


class ChunkerRegistry:
    """策略名 → 分块器实例的注册表（形状与 day058/060/061 的注册表一致）."""

    def __init__(self) -> None:
        self._chunkers: dict[str, Chunker] = {}

    def register(self, chunker: Chunker) -> None:
        """注册一个分块器；同名重复注册直接报错.

        不做"后来者覆盖"是刻意的：一个策略名对应两套实现时，
        ``/chunking/split`` 的结果取决于**导入顺序**——而那种不确定性
        在报告里看不出来。
        """
        if chunker.name in self._chunkers:
            raise ChunkingError(f"策略 {chunker.name!r} 已经注册过了")
        self._chunkers[chunker.name] = chunker

    def get(self, strategy: str) -> Chunker:
        """按名取分块器；未注册时列出**当前可用的策略**（day045 同款报错）."""
        if strategy not in self._chunkers:
            raise UnsupportedStrategy(
                f"没有名为 {strategy!r} 的分块策略，"
                f"当前可用：{', '.join(self.strategies) or '（空）'}"
            )
        return self._chunkers[strategy]

    @property
    def strategies(self) -> tuple[str, ...]:
        """已注册的策略名（按 ``STRATEGIES`` 的顺序，**不按注册顺序**）."""
        return tuple(name for name in STRATEGIES if name in self._chunkers)

    def table(self) -> list[dict[str, Any]]:
        """四张自述表（端点 ``GET /chunking/strategies`` 直接返回它）."""
        return [self._chunkers[name].describe() for name in self.strategies]


def build_chunker(
    strategy: str,
    policy: ChunkPolicy | None = None,
    *,
    measurer: TokenMeasurer | None = None,
    embedding: EmbeddingProvider | None = None,
) -> Chunker:
    """按策略名建一个分块器（端点与演示脚本都走它）.

    四个实现**在这里惰性导入**：``semantic`` 依赖 embedding 提供方，
    而 ``default_embedding()`` 会读 settings——放在模块顶层会让
    ``import smart_research_agent.chunking`` 顺带把配置解析一遍，
    测试里替换环境变量就来不及了。
    """
    from smart_research_agent.chunking.fixed import FixedSizeChunker
    from smart_research_agent.chunking.recursive import RecursiveChunker
    from smart_research_agent.chunking.semantic import SemanticChunker
    from smart_research_agent.chunking.structural import StructuralChunker

    builders: dict[str, type[Chunker]] = {
        STRATEGY_FIXED: FixedSizeChunker,
        STRATEGY_RECURSIVE: RecursiveChunker,
        STRATEGY_STRUCTURAL: StructuralChunker,
        STRATEGY_SEMANTIC: SemanticChunker,
    }
    if strategy not in builders:
        raise UnsupportedStrategy(
            f"没有名为 {strategy!r} 的分块策略，可有：{', '.join(STRATEGIES)}"
        )
    return builders[strategy](
        policy or default_policy(strategy), measurer=measurer, embedding=embedding
    )


def default_registry(
    *,
    measurer: TokenMeasurer | None = None,
    embedding: EmbeddingProvider | None = None,
) -> ChunkerRegistry:
    """四种策略各一份默认实例组成的注册表（自述表与演示默认用它）."""
    registry = ChunkerRegistry()
    for strategy in STRATEGIES:
        registry.register(build_chunker(strategy, measurer=measurer, embedding=embedding))
    return registry


def chunking_boundaries() -> dict[str, Any]:
    """分块能力的自述（端点与文档同源，避免"文档说有一套、代码里没有"）."""
    return {
        "strategies": list(STRATEGIES),
        "measurers": list(MEASURERS),
        "default_policies": DEFAULT_POLICIES,
        "separators": _separator_preview(),
        "limitations": list(CHUNKING_LIMITATIONS),
        "out_of_scope": list(CHUNKING_OUT_OF_SCOPE),
        "coverage_invariant": (
            "document.text[start_char:end_char] == chunk.text，"
            "且原文每个非空白字符至少属于一个块（由 finalize 强制，违反即报错）"
        ),
        "next_step": "day064/065 的向量库按 chunk_id 入库，day066 的检索器按 retrieval_text 召回",
    }


def _separator_preview() -> list[str]:
    """递归策略的分隔符表（自述用；真实定义在 recursive 模块，避免循环导入）."""
    from smart_research_agent.chunking.recursive import DEFAULT_SEPARATORS

    return list(DEFAULT_SEPARATORS)


__all__ = [
    "BOUNDARY_CHARS",
    "CHUNKING_LIMITATIONS",
    "CHUNKING_OUT_OF_SCOPE",
    "DEFAULT_POLICIES",
    "PARAGRAPH_GAP",
    "ChunkPolicy",
    "Chunker",
    "ChunkerRegistry",
    "Span",
    "build_chunker",
    "chunking_boundaries",
    "default_policy",
    "default_registry",
    "heading_positions",
    "paragraph_spans",
    "path_for_offset",
    "snap_to_boundary",
]
