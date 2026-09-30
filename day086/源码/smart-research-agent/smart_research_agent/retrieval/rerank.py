"""交叉编码器重排：**把"最像"换成"最相关"，而且只对前 N 条这么做**（M6-D7）.

双塔检索（day064~day067）把查询与文档**分开**编码，因此文档向量可以在离线算好
——这正是它能扫百万条的原因。代价是查询与文档**从来没有见过面**：两边各自被压成
一个点，相似度只是两个点的夹角。

```text
双塔（bi-encoder）        查询 → 向量 ┐
                         文档 → 向量 ┘ → 点积        文档侧可预计算，但两两不相见
交叉编码器（cross-encoder） [CLS] 查询 [SEP] 文档 → transformer → 相关性 logit
                                                  两两一起过模型：更准，也贵得多
```

贵多少必须写出来，否则"为什么不干脆全用交叉编码器"会变成一个反复出现的疑问：

```text
双塔对 N 篇文档是 N 次点积      可批量、可在索引里做、可预计算文档侧
交叉编码器是 N 次前向推理      每一对都要把两个文本拼起来跑一遍模型
```

因此工业界的标准姿势是**两段式**，也就是本模块的形状：

```text
第一阶段（召回）  双塔 / BM25 / 融合 → 几十~几百条候选    追求"别漏"，允许粗糙
第二阶段（重排）  交叉编码器只对**前 N 条**打分 → 重排     追求"排准"，必须精确
```

## 窗口纪律：只对前 N 条打分（本模块的第一条纪律）

它有一个好消息与一个坏消息，两个都必须写出来：

```text
好消息   成本从 O(候选) 降到 O(N)，N 是一个可配的旋钮（retrieval_rerank_top_n）
坏消息   第 N+1 条**永远没有机会**证明自己——它的名次只由第一阶段的分数决定
```

坏消息那一半**无条件进 notes**：只报"重排生效了"却不报"有一批条连分都没拿到"，
等于把一次**有边界的**改进说成了一次全面的改进。这条注记是本模块最容易被删掉、
也最不该删掉的一行。

## 教学级替身：它没有真实语义（诚实声明，写在最前面）

真实的交叉编码器（bge-reranker / ms-marco-MiniLM 之类）要下载权重、要推理框架、
通常还要 GPU——**这三件事本课一件都不许有**（离线、确定性、零依赖）。
因此 ``CrossEncoderReranker`` 是一个**确定性替身**：分数由四个可手算的特征
加权而成。**它没有真实语义**，这句话不含糊其辞：

```text
它认不出同义改写    "重算"与"重新计算"在它眼里是两个不同的字串
它读不懂语序        "甲打败了乙"与"乙打败了甲"的特征几乎一样
```

**生产环境应当把它替换为真实重排模型**——接口就是 ``BaseReranker``
（与 ``llm.embedding.MockEmbedding`` 的诚实写法同源：形状留好、替身说清）。
换来的东西恰恰是本课要的：**分数可以手算核对**。四个特征都在下面，
任何一个分数都能拿纸笔验算——而一个"必须先下载 2 GB 权重才能复现"的分数，
在"这次评估为什么变了"这个问题前面是哑的。

## 四个特征（全部落在 [0, 1]，因此加权和也落在 [0, 1]）

```text
term_coverage    查询词元里出现在文档中的比例（**去重后**算）
                 → 替代真实模型的词汇匹配能力
exact_phrase     文档包含查询原文（strip + 折叠大小写）→ 1.0，否则 0.0
                 → 替代真实模型对整句的强匹配
proximity        1 / (1 + 最小跨度)：所有查询词元都出现时的最小窗口宽度
                 → 替代真实模型的语序/邻近敏感
length_penalty   min(1.0, IDEAL_LEN / max(len(text), 1))
                 → 从**另一侧**对冲排序模型的冗长偏好，理由见下
```

``term_coverage`` 与 ``proximity`` 复用 ``lexical.tokenize``（**同一套分词**）：
重排与关键词路如果各用一套分词，"同一个查询能不能命中同一篇"就会有两份答案，
而这两份答案在报告里长得完全一样——那是评估最怕的一种不一致
（同一条纪律也用在 day067 的"两路共用一份过滤子句"上）。

``length_penalty`` 单独说明：交叉编码器的位置偏置里最顽固的一条是**冗长偏好**
——长文档里"恰好出现某个词"的概率本来就高，注意力也更容易找到一段能与查询对上
的文字，于是分数系统性地偏高。把长度罚项放进分数，是在**重排这一侧**对冲它；
而"长文档更该被看见吗"没有普适答案，因此它只占 1/8 的权重
（见 ``CrossEncoderReranker.FEATURE_WEIGHTS``），而不是一条硬规则。

## 两种模式：replace 与 blend（量纲问题在这里**第二次**出现）

```text
replace   窗口内**只**按重排分重排（第一阶段的分数降级为兜底的排序键）
blend     窗口内把两份分数**各自 min-max 归一化**到 [0, 1]，再按 weight 加权
```

``blend`` 里那一步归一化不是可选项：第一阶段的分数是一个**没有上界、
量纲取决于上游**的数（余弦 ∈ [-1,1]、BM25 ∈ [0,∞)、RRF 是 1/(k+r) 之和、
weighted 融合又是一套归一化分），而重排分落在 [0, 1]。直接加权等于让"量纲差"
冒充"相关性差"——与 ``fusion`` 模块是同一条纪律，因此这里**复用同一个函数**
（``fusion._min_max_normalize``）：同一条纪律只写一份实现，两份实现一定会分家。

代价也必须写下来：归一化只在**这一次窗口内**做，因此同一篇文档的归一化分
**随窗口变化**（换一条查询、换一个 ``top_n``，它都会变）。这是相对量的固有性质，
不是实现缺陷——只要求"同一次重排内部可比"，而那正是加权求和需要的全部。

与之配套的一条小口径：``min_score`` 比的是**重排分**（不是第一阶段的分数）。
理由是可标定性——教学替身的重排分落在 ``[0, 1]``，因此一个阈值在这里有绝对含义
（与 BM25 的"没有绝对标度、给它一个数字是假的安全感"正好相反）。

## 评估提升：三条指标都是"名单 + 金标准"的函数

"重排到底有没有用"不是靠感觉回答的，而是靠一组标注好的查询（``LiftProbe``：
一句话 + 它的金标准 id）与三条指标。三条的公式写在这里，实现里逐字对应：

```text
recall@k   = |前 k 条 ∩ 金标准| / |金标准|                    （命中面：有没有捞回来）
RR         = 1 / (第一个相关命中的位置 + 1)（无则 0.0）        （头部质量：第一条要多久才相关）
nDCG@k     = DCG@k / IDCG@k
             DCG@k  = Σ_{i=0}^{k-1} rel_i / log2(i + 2)        二值增益 rel_i ∈ {0,1}
             IDCG@k = Σ_{i=0}^{m-1} 1 / log2(i + 2)，
                      m = min(|金标准|, k)                     （理想名单的 DCG）
```

三者必须一起看：只看 ``recall@k`` 时，"把相关的那条从第 4 名提到第 1 名"这件事
完全不可见（都在前 k 条里）；只看 ``RR`` 时，"前 k 条命中了 3 条还是 1 条"
不可见。``nDCG@k`` 是唯一同时看"命中几条"与"排得多靠前"的那条。

## 与既有包的接缝

- **上游**：``RetrievalHit`` 序列——单路在**阈值之后**（``retriever`` 第 6 步后）、
  混合在**融合之后**（``hybrid`` 第 10 步后）。本模块只读它的
  ``record_id`` / ``score`` / ``rank`` / ``text`` / ``channels``，
  写回两个**新增**字段 ``rerank_score`` 与 ``stage1_rank``：``score``
  仍然是第一阶段的分数（口径不变），因此"重排分是多少"与"它原来在第几"
  各有一个能查的地方，而"越大的分越靠前"这条不变量在重排列表里**不再成立**
  ——这一点写在 ``RetrievalHit`` 的字段说明里，不藏着；
- **脚下**：``lexical.tokenize``（同一套分词）与 ``fusion._min_max_normalize``
  （同一条归一化纪律），两者都不重写；
- **下游**：day071 的 RAG 评估用 ``measure_lift`` 量"重排有没有用"，
  它与真实模型无关（三条指标只依赖名单与金标准），因此**替身也能跑出一份对照表**
  ——这正是"评估提升"这条学习目标要的落地。
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.retrieval.errors import RerankError
from smart_research_agent.retrieval.fusion import _min_max_normalize
from smart_research_agent.retrieval.lexical import tokenize
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: 缺省的重排模型名。它是**教学级替身**的名字（见模块 docstring），
#: 刻意把 "teaching" 写进名字里：一份报告里出现它，读的人立刻知道
#: "这里的分数不是真实模型的分数"，而不是先怀疑数据有问题。
DEFAULT_RERANK_MODEL = "cross-encoder-teaching-v1"

#: 只对前 N 条打分。缺省 20 是"交叉编码器比双塔贵两个数量级"与"第 21 名还值不值得救"
#: 之间的折中：N 越大越准、越慢；N 之外的一条都不会被打分（窗口纪律）。
DEFAULT_RERANK_TOP_N = 20

#: ``blend`` 模式下**重排分**的权重（第一阶段分数的权重是 ``1 - weight``）。
#: 缺省 0.5 是"两边都不占先"的起点：它不代表两个信号一样有用，
#: 只代表**还没有证据**说明该偏向哪一边——先跑一遍 ``replace`` 看清重排把谁动了，
#: 再决定要不要 blend。
DEFAULT_RERANK_WEIGHT = 0.5

#: 一次送给重排器的文本条数。缺省 16 是"批处理收益"与"单批体积"的折中，
#: 与 ``indexing_batch_size`` 同一条理由：太小退化成逐条，太大则一次失败要重算一整批。
#: 注意它**不改变任何分数**（特征逐条算），只改变"分成几次算"——因此它是一个纯开销旋钮。
DEFAULT_RERANK_BATCH = 16

#: 重排模式之一：窗口内**只**按重排分重排。
RERANK_MODE_REPLACE = "replace"

#: 重排模式之二：窗口内两份分数各自归一化后加权。
RERANK_MODE_BLEND = "blend"

#: 全部模式（顺序 = 报告里的顺序）。``RetrievalQuery.extra`` 与 ``settings``
#: 都必须落在这个元组里，否则 ``rerank_hits`` 当场报 ``RerankError``。
RERANK_MODES: tuple[str, ...] = (RERANK_MODE_REPLACE, RERANK_MODE_BLEND)

#: ``rerank_hits`` 允许从 ``RetrievalQuery.extra`` 读到的键（见 ``hybrid``）。
#: **封闭清单**：多一个键就报错，理由与 ``FUSION_OVERRIDE_KEYS`` 逐字相同——
#: 静默忽略一个覆盖参数会让调用方以为它生效了（"我把 ``mode`` 拼成 ``mod``，
#: 结果看起来只是没有变化"）。
RERANK_OVERRIDE_KEYS: tuple[str, ...] = ("enabled", "mode", "model", "top_n", "weight")

#: 教学替身摊开来的四个特征名（顺序 = ``CrossEncoderReranker.FEATURE_WEIGHTS`` 的顺序）。
#: 它是一份**封闭清单**：``RerankHit.features`` 的键必须落在里面，
#: 否则"这一条为什么得到这个分"就无法按一张固定的表读出来。
RERANK_FEATURE_NAMES: tuple[str, ...] = (
    "term_coverage",
    "exact_phrase",
    "proximity",
    "length_penalty",
)

#: 长度惩罚的"标准长度"（字符）。缺省 240 按本课语料的块长标定
#: （day062 的结构化分块大多落在 150~350 字）：短于它不罚，长于它按比例打折。
#: 它只影响 ``length_penalty`` 一项，且在总权重里占 1/8——因此它不是一个"调不动"
#: 的隐藏参数，而是一个可以显式改的旋钮（``CrossEncoderReranker(ideal_len=...)``）。
IDEAL_LEN = 240

#: 提升报告里的三条指标名（顺序 = 报告里的顺序）。**私有**：它是本模块的口径，
#: 不是给外部拼字符串用的清单（三条指标的公式见模块 docstring）。
_METRIC_NAMES: tuple[str, ...] = ("recall", "reciprocal_rank", "ndcg")


# --------------------------------------------------------------------------- #
# 特征形状
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RerankFeatures:
    """一次 (查询, 文档) 打分用到的**四个可手算特征**，全部落在 ``[0, 1]``.

    它之所以是一个形状而不是四个浮点参数，理由与 ``BM25Params`` 相同：
    这四个数是**一起**进报告的——"这个分数是怎么来的"必须能被逐项核对，
    而一个方法返回四元组会让"哪一项是哪一项"取决于调用点的顺序。

    四个特征的定义与"它替代真实模型的哪一种能力"写在模块 docstring 里；
    这里只强调一件事：**每个特征都在 [0, 1]**，因此加权和也在 [0, 1]。
    这条不变量正是 ``min_score`` 可以标定的前提（见模块 docstring）。
    """

    term_coverage: float
    exact_phrase: float
    proximity: float
    length_penalty: float

    def __post_init__(self) -> None:
        for name in RERANK_FEATURE_NAMES:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RerankError(
                    f"RerankFeatures.{name} 必须是数字，收到 {type(value).__name__}"
                    f"（{value!r}）。四个特征都是算出来的浮点数；字符串形式的特征"
                    "会让加权和变成字符串拼接——一个不会报错、只会得到怪分数的结果。"
                )
            if not math.isfinite(float(value)):
                raise RerankError(
                    f"RerankFeatures.{name} 必须是有限数，收到 {value!r}。"
                    "nan 参与的比较恒为 False，于是**这条命中的名次不再由分数决定**；"
                    "请检查算这个特征的那段代码（除零是最常见的原因）。"
                )
            number = float(value)
            if not 0.0 <= number <= 1.0:
                raise RerankError(
                    f"RerankFeatures.{name}={number} 越界：四个特征都必须落在 [0, 1]。"
                    "越界的特征会让加权和超出 [0, 1]，而 min_score 的可标定性"
                    "（[0, 1] 上的一个阈值）正是建立在'分数不会越界'之上的。"
                )
            object.__setattr__(self, name, number)

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return {name: round(getattr(self, name), 6) for name in RERANK_FEATURE_NAMES}

    def summary_line(self) -> str:
        """人类可读的一行摘要（按固定顺序列出四项，便于逐条对照两次运行）."""
        rendered = "、".join(f"{name}={getattr(self, name):.4f}" for name in RERANK_FEATURE_NAMES)
        return f"特征：{rendered}"


# --------------------------------------------------------------------------- #
# 重排器接口 + 教学级替身
# --------------------------------------------------------------------------- #


class BaseReranker(ABC):
    """重排器的接口：**给一句话与一批文本，给出每一条的相关性分数**（越大越相关）.

    它是本模块唯一的扩展点，因此刻意只有三件必须实现的事与一件可选的事：

    ```text
    name            不许无声的模型：报告里要回显它（"这次是谁打的分"）
    dimension       打分的维度（见 CrossEncoderReranker.dimension 的诚实说明）
    score_pairs     核心：一次调用给出一批分数（**对外是一次调用**，内部分批是开销问题）
    describe        给端点/报告用的自述
    explain_pairs   可选的第二入口：把分数用到的特征摊开（缺省返回空字典）
    ```

    **接真实模型的正确姿势**：写一个子类包住真实重排服务（本地权重或 HTTP 都行），
    ``score_pairs`` 里做真正的 ``[CLS] query [SEP] doc`` 前向推理，
    ``explain_pairs`` 可以照旧返回空字典（真实模型没有"可手算的四个特征"，
    硬凑一份反而是编出来的证据）。**本模块的其余部分不需要改动**——
    窗口纪律、两种模式、回填的字段、提升指标都与"分数是怎么来的"无关。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """这个重排器的名字（进报告；缺省用 ``DEFAULT_RERANK_MODEL``）."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """打分的维度（教学替身是 4，真实交叉编码器通常是 1，见实现类里的说明）."""

    @abstractmethod
    def score_pairs(self, query: str, texts: Sequence[str]) -> list[float]:
        """``(query, text)`` 逐对打分（**对外一次调用**，内部可以分批）.

        返回的列表必须与 ``texts`` **等长同序**：错位不会报错（两个都是列表），
        只会让"这一段为什么得到这个分"变成编出来的（见 ``rerank_hits`` 的等长校验）。
        """

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """这个重排器的自述（端点直接返回它；**不含任何逐次变化的账**）."""

    def explain_pairs(self, query: str, texts: Sequence[str]) -> list[dict[str, float]]:
        """把"分数用到了哪些特征"摊开（缺省什么都不摊：``{}``）.

        它不是抽象方法：真实重排模型的分数来自一个隐层，没有可以摆出来的特征，
        因此"返回空字典"是**正确**的实现而不是偷懒。教学替身覆盖它，
        好让"这个 0.8125 是怎么来的"能被逐项核对。
        """
        return [{} for _ in texts]

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        return f"重排器 {self.name!r}（{self.dimension} 维打分）"


class CrossEncoderReranker(BaseReranker):
    """**确定性离线**的交叉编码器替身：分数 = 四个特征的加权和.

    真实交叉编码器做的是：``[CLS] 查询 [SEP] 文档`` 拼起来过一遍 transformer，
    取相关性 logit。这里用四个可手算的特征替代它，理由与代价都写在模块 docstring
    （**它没有真实语义**，生产环境请换成真实重排模型，接口就是 ``BaseReranker``）。

    换来的性质里有两条值得单独说：

    ```text
    分数可以手算核对   0.8125 这样的分数能被纸笔验算，而"评估为什么变了"要的正是这个
    确定性             同一个输入在任何机器、任何 Python 版本上给出逐位相同的分数
                       （没有浮点累加顺序问题：四个项、固定顺序）
    ```

    ``dimension`` 在这里返回 **4** 而不是 1：真实交叉编码器只产出一个 logit，
    而替身把这一维摊成了四个可解释的特征。**这个差异是刻意暴露的**——
    如果有人拿 ``dimension == 4`` 去和真实模型对齐，他会立刻发现对不上，
    而一个"看起来是 1"的替身会让这件事晚一步才被发现。
    """

    #: 四个特征的权重，顺序与 ``RERANK_FEATURE_NAMES`` 一一对应。
    #:
    #: 四个数取 1/2、1/4、1/8、1/8——它们都是**二进制精确可表示**的数，
    #: 因此和**逐位等于 1.0**，构造期可以用精确比较校验而不必引入一个 epsilon
    #: （与 ``fusion._min_max_normalize`` 里"精确的零跨度"是同一条纪律：
    #: 一个凭空取出的 tolerance 会变成第三个需要标定的参数，而它的表现是
    #: "某些权重组合明明是 1 却被拒绝了"）。
    #:
    #: 排序本身的含义：覆盖率是主项（它替代词汇匹配），整句包含是强证据，
    #: 邻近度与长度各占 1/8（它们是对冲项，不是决定性因素）。
    FEATURE_WEIGHTS: tuple[float, ...] = (0.5, 0.25, 0.125, 0.125)

    def __init__(
        self,
        *,
        name: str = DEFAULT_RERANK_MODEL,
        batch_size: int = DEFAULT_RERANK_BATCH,
        ideal_len: int = IDEAL_LEN,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise RerankError(
                f"CrossEncoderReranker.name 必须是非空字符串，收到 {name!r}："
                "名字会进报告（'这次是谁打的分'），空名字会让重排这一段在报告里失名。"
            )
        self._name = name.strip()
        self._batch_size = _check_positive_int("batch_size", batch_size)
        self._ideal_len = _check_positive_int("ideal_len", ideal_len)
        _check_feature_weights(type(self).FEATURE_WEIGHTS)
        # 三个计数器是"真的跑了几次"的账：测试要断言"top_n 之外的条数一次都没被打分"，
        # 靠的就是它们（读一次差量即可）。它们**只增不减**：一个实例可以被多次查询复用，
        # "这次调用打了多少分"用调用前后的差值回答。
        self._calls = 0
        self._scored_pairs = 0
        self._batches = 0

    # ------------------------------------------------------------------ 只读视图

    @property
    def name(self) -> str:
        """这个重排器的名字（进报告）."""
        return self._name

    @property
    def dimension(self) -> int:
        """打分的维度：替身是 4（四个可手算特征）——真实模型是 1，见类 docstring."""
        return len(RERANK_FEATURE_NAMES)

    @property
    def batch_size(self) -> int:
        """内部分批的批大小（**不改变任何分数**，只改变分成几次算）."""
        return self._batch_size

    @property
    def ideal_len(self) -> int:
        """长度惩罚的"标准长度"（``length_penalty`` 的唯一参数）."""
        return self._ideal_len

    @property
    def calls(self) -> int:
        """``score_pairs`` 被调用了几次（**累计值**，含空输入的调用）."""
        return self._calls

    @property
    def scored_pairs(self) -> int:
        """累计真正打过分的 ``(查询, 文档)`` 对数量（**窗口纪律的账**）."""
        return self._scored_pairs

    @property
    def batches(self) -> int:
        """累计真正算过的批次数（``ceil(scored_pairs / batch_size)`` 的累计）."""
        return self._batches

    def describe(self) -> dict[str, Any]:
        """自述：名字 / 类型 / 特征表 / 权重 / 批大小 + 三个计数器.

        ``kind`` 里带着 ``(teaching)`` 是刻意的：端点的响应体里必须看得出
        "这次打分的是替身而不是真实模型"（与 ``name`` 的 ``teaching`` 同一个目的，
        两处都写是为了防止有人只截了 ``kind`` 就去做结论）。
        """
        return {
            "name": self._name,
            "kind": "cross-encoder(teaching)",
            "features": list(RERANK_FEATURE_NAMES),
            "weights": {
                feature: weight
                for feature, weight in zip(RERANK_FEATURE_NAMES, self.FEATURE_WEIGHTS)
            },
            "batch_size": self._batch_size,
            "ideal_len": self._ideal_len,
            "calls": self._calls,
            "scored_pairs": self._scored_pairs,
            "batches": self._batches,
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        return (
            f"重排器 {self._name!r}（教学级交叉编码器替身，"
            f"{len(RERANK_FEATURE_NAMES)} 维可手算特征，批大小 {self._batch_size}）"
        )

    # ------------------------------------------------------------------ 打分

    def score_pairs(self, query: str, texts: Sequence[str]) -> list[float]:
        """逐个 ``(query, text)`` 打分（内部分批，**对外是一次调用**）.

        分批只是开销问题：分数逐条算，与批大小无关，因此
        ``score_pairs(..., batch_size=1)`` 与 ``batch_size=1024`` 给出一字不差的列表。
        这条性质写在这里，是为了让"换批大小之后结果变了"永远指向别的原因。

        空列表是合法输入（窗口可能是空的）：返回 ``[]``，并且**不产生批次**
        （``batches`` 加 0）——"算过 0 批"与"算过 1 批空批"是两件事。
        """
        _check_query_text(query)
        cleaned = _check_text_sequence(texts, label="CrossEncoderReranker.score_pairs")
        self._calls += 1
        scores: list[float] = []
        for start in range(0, len(cleaned), self._batch_size):
            self._batches += 1
            scores.extend(self._score_chunk(query, cleaned[start : start + self._batch_size]))
        self._scored_pairs += len(cleaned)
        return scores

    def explain_pairs(self, query: str, texts: Sequence[str]) -> list[dict[str, float]]:
        """把每条文本的四个特征摊开（**不计数**：它不是打分，只是把打分用的数摆出来）.

        特征会被算第二遍（打分时一遍、这里一遍）。这是刻意的：缓存会让
        "这段分数对应这些特征"变成一个需要维护的状态，而本模块的价值恰恰在于
        "任何一个数都能独立验算"。四个特征的算术成本与一次分词同量级，
        换来的是一份无状态的、可重放的实现。
        """
        _check_query_text(query)
        cleaned = _check_text_sequence(texts, label="CrossEncoderReranker.explain_pairs")
        return [self._features(query, text).to_dict() for text in cleaned]

    def features_of(self, query: str, text: str) -> RerankFeatures:
        """一对 ``(query, text)`` 的四个特征（公开一点点，便于教学与测试逐项核对）."""
        _check_query_text(query)
        if not isinstance(text, str):
            raise RerankError(
                f"features_of 的 text 必须是字符串，收到 {type(text).__name__}。"
                "一段文本请直接给 str；一个批次请用 explain_pairs。"
            )
        return self._features(query, text)

    def _score_chunk(self, query: str, chunk: Sequence[str]) -> list[float]:
        """一批文本的分数：``Σ 权重_i × 特征_i``（**顺序固定**，因此逐位可复现）."""
        weights = self.FEATURE_WEIGHTS
        scores: list[float] = []
        for text in chunk:
            features = self._features(query, text)
            total = 0.0
            for weight, feature in zip(weights, RERANK_FEATURE_NAMES):
                total += weight * getattr(features, feature)
            scores.append(total)
        return scores

    def _features(self, query: str, text: str) -> RerankFeatures:
        """算四个特征（定义与模块 docstring 逐条对应，这里是唯一的实现）."""
        query_terms = sorted(set(tokenize(query)))
        doc_terms = tokenize(text)
        doc_set = set(doc_terms)

        # 1) 词元覆盖率：去重后的查询词元里，有多少出现在文档里。
        #    查询一个词元都没有（纯标点）时定义为 0.0 而不是 1.0：没有证据
        #    不等于证据充分——把它算成 1.0 会让"什么都匹配"的文档白拿 0.5 分。
        coverage = (
            sum(1 for term in query_terms if term in doc_set) / len(query_terms)
            if query_terms
            else 0.0
        )

        # 2) 整句包含：折叠大小写后判包含（与 tokenize 的折叠口径一致）。
        #    strip 之后为空串时给 0.0：空串"在"任何文本里，那会让每条都拿到这一项。
        stripped = query.strip().casefold()
        phrase = 1.0 if stripped and stripped in text.casefold() else 0.0

        # 3) 邻近度：所有查询词元都出现时的最小窗口宽度（越小越像"整句命中"）。
        #    1/(1+span) 把 "全部紧挨着"（span=词元数）映射到接近 1，
        #    而"散落全文"映射到接近 0；无法覆盖时是 0.0（不是"很远"）。
        span = _min_span(doc_terms, query_terms)
        proximity = 1.0 / (1.0 + span) if span is not None else 0.0

        # 4) 长度惩罚：短于标准长度不罚，长于它按比例打折（理由见模块 docstring：
        #    交叉编码器的冗长偏好从另一侧对冲）。
        length_penalty = min(1.0, self._ideal_len / max(len(text), 1))

        return RerankFeatures(
            term_coverage=coverage,
            exact_phrase=phrase,
            proximity=proximity,
            length_penalty=length_penalty,
        )


# --------------------------------------------------------------------------- #
# 两副结果形状
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RerankHit:
    """重排之后的一条命中：**新名次 + 它凭什么排这里 + 它原来在哪**.

    ```text
    record_id        哪一条
    score            重排分（**越大越相关**，教学替身落在 [0, 1]）
    rank             重排后的名次（0 起连续，整份名单一起编号）
    stage1_score     第一阶段给它的分数（原样带来，用来回答"变了多少"）
    stage1_rank      它在第一阶段的名次（**输入序列的位置**，0 起）
    blended          blend 模式下的加权分（replace 模式是 None）
    features         重排分用到的四个特征（窗口外的条是空字典）
    channels         它被哪几路召回（从上游的命中带过来，融合证据不丢）
    scored           这一条**到底有没有被重排器打过分**
    ```

    ``stage1_rank`` 用的是**输入序列的位置**而不是上游那个 ``hit.rank``：
    阈值落刀之后 ``hit.rank`` 会带空洞（0、1、3、4），而重排的排序第二键
    需要一个稠密的、不含空洞的次序（"没有空洞"时两者逐位相同）。

    ``scored=False`` 的那一批（窗口之外）里，``score`` 是 **0.0** 而不是 ``None``：
    形状上保持"每条都有一个分数"，而"它其实没被打分"这件事由 ``scored``
    与 ``RerankResult.notes`` 里那条**无条件**注记负责。0.0 不会被误读成
    "它最不相关"，因为窗口外的那批按原顺序排在窗口结果**之后**——
    名次与分数在这里是两套证据，而不是一个排序键。
    """

    record_id: str
    score: float
    rank: int
    stage1_score: float
    stage1_rank: int
    blended: float | None = None
    features: dict[str, float] = field(default_factory=dict)
    channels: tuple[str, ...] = ()
    scored: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, str) or not self.record_id.strip():
            raise RerankError(
                f"RerankHit.record_id 必须是非空字符串，收到 {self.record_id!r}："
                "重排的第一步就是按 id 把名次写回去，没有 id 就没有'这一条'。"
            )
        for label, value in (("score", self.score), ("stage1_score", self.stage1_score)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RerankError(
                    f"RerankHit.{label} 必须是数字，收到 {type(value).__name__}"
                    f"（{value!r}）。分数是算出来的浮点数，字符串会让排序变成字典序比较。"
                )
            if not math.isfinite(float(value)):
                raise RerankError(
                    f"RerankHit.{label} 必须是有限数，收到 {value!r}："
                    "nan 会让这条落到任意位置，而重排的全部意义就是名次。"
                )
        if self.blended is not None:
            if isinstance(self.blended, bool) or not isinstance(self.blended, (int, float)):
                raise RerankError(
                    f"RerankHit.blended 必须是数字或 None，收到 {type(self.blended).__name__}。"
                    "replace 模式请留 None——'没算过'与'算出来是 0'是两件事。"
                )
            if not math.isfinite(float(self.blended)):
                raise RerankError(f"RerankHit.blended 必须是有限数，收到 {self.blended!r}")
        for label, value in (("rank", self.rank), ("stage1_rank", self.stage1_rank)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise RerankError(
                    f"RerankHit.{label} 必须是非负整数（从 0 起），收到 {value!r}。"
                    "名次从 1 起会让'第几条'这个问题在报告里整体错一格。"
                )
        if not isinstance(self.features, dict):
            raise RerankError(f"RerankHit.features 必须是字典，收到 {type(self.features).__name__}")
        unknown = sorted(str(name) for name in self.features if name not in RERANK_FEATURE_NAMES)
        if unknown:
            raise RerankError(
                f"RerankHit.features 里有本模块不认识的键 {unknown}："
                f"教学级替身的特征是 {'、'.join(RERANK_FEATURE_NAMES)}。"
                "不在清单里的键会让'这一条为什么得到这个分'无法按一张固定的表读出来——"
                "自定义重排器请让 explain_pairs 返回空字典（真实模型的分数没有可摆出的特征）。"
            )
        normalized: dict[str, float] = {}
        for name, value in self.features.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RerankError(
                    f"RerankHit.features[{name!r}] 必须是数字，收到 {type(value).__name__}"
                )
            if not math.isfinite(float(value)):
                raise RerankError(f"RerankHit.features[{name!r}] 必须是有限数，收到 {value!r}")
            normalized[str(name)] = float(value)
        if not isinstance(self.channels, tuple):
            raise RerankError(
                f"RerankHit.channels 必须是 tuple，收到 {type(self.channels).__name__}。"
                "写法：channels=('vector', 'bm25')；空元组 () 表示'未记录多路证据'——"
                "frozen 形状里塞一个可变对象会让'这份结果不曾被改过'不再成立。"
            )
        if not isinstance(self.scored, bool):
            raise RerankError(
                f"RerankHit.scored 必须是布尔值，收到 {type(self.scored).__name__}："
                "它是'这一条到底有没有被打分'的唯一标记，非布尔的写法会让它被当成真值用。"
            )
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "stage1_score", float(self.stage1_score))
        if self.blended is not None:
            object.__setattr__(self, "blended", float(self.blended))
        object.__setattr__(self, "features", normalized)

    @property
    def moved(self) -> int:
        """名次变化：``stage1_rank - rank``（**正数 = 上移**，负数 = 下移）.

        注意它把"别人被切掉"也算进来：重排阈值切掉几条之后，后面每一条的
        ``rank`` 都会前移，于是它们的 ``moved`` 也是正的。它是**名次差**，
        不是"重排器认为它更好"——要问后者请看 ``features``。
        """
        return self.stage1_rank - self.rank

    def to_dict(self, *, include_features: bool = True) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典.

        ``include_features=False`` 用于"只看名次怎么动"的场景：四个特征 × 20 条
        会把一次 diff 淹掉，而"重排把谁提到了前面"这件事只需要 id 与名次。
        """
        payload: dict[str, Any] = {
            "rank": self.rank,
            "record_id": self.record_id,
            "score": round(self.score, 6),
            "stage1_score": round(self.stage1_score, 6),
            "stage1_rank": self.stage1_rank,
            "moved": self.moved,
            "blended": None if self.blended is None else round(self.blended, 6),
            "channels": list(self.channels),
            "scored": self.scored,
        }
        if include_features:
            payload["features"] = {name: round(value, 6) for name, value in self.features.items()}
        return payload

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        blend = "" if self.blended is None else f" blended={self.blended:.4f}"
        if not self.scored:
            return (
                f"#{self.rank} {self.record_id} stage1 #{self.stage1_rank}"
                f"（score={self.stage1_score:.4f}）| **窗口外，未打分**"
            )
        return (
            f"#{self.rank} {self.record_id} rerank={self.score:.4f}{blend} | "
            f"stage1 #{self.stage1_rank} → #{self.rank}（移动 {self.moved:+d}）"
        )


@dataclass(frozen=True)
class RerankResult:
    """一次重排的完整交代：新名单 + **窗口与两种模式的账**.

    ```text
    query_text            这次问的是什么（重排分是对 (查询, 文档) 打的，查询必须跟着走）
    hits                  重排后的完整名单（窗口内重排结果 + 窗口外原样尾部）
    candidates            输入条数（上游给过来几条）
    scored                真正被打分的条数（**恰好是 min(candidates, top_n)**）
    top_n                 窗口大小（"只对前 N 条打分"里的 N）
    model / mode / weight 这一组参数（报告里必须能复述"用的是哪种重排"）
    dropped_by_min_score  重排阈值切掉几条（只对被打分过的那批生效）
    empty_reason          没有命中时的**唯一**原因（有命中时是空串）
    latency_ms            这次重排花了多久
    notes                 窗口纪律、被忽略的权重、模型名不符等"必须被看见"的事
    ```

    ``weight`` 只在 ``mode="blend"`` 下参与排序。``mode="replace"`` 且调用方
    显式给了 ``weight`` 时，它照样被记录在这里，并进 ``notes`` 说明
    "replace 不看权重"——**记录并说明**而不是报错，与 ``fusion`` 里
    "``strategy='rrf'`` 却给了 ``weights`` 直接报错"是**两个刻意的不同选择**：
    那里权重在 RRF 的定义里根本不存在（给了它说明调用方认为自己在用另一种策略），
    这里权重只是另一种模式的参数（报错会让"试一下 blend 的区别"
    变成一次必须先删参数的实验）。
    """

    query_text: str
    hits: tuple[RerankHit, ...] = ()
    candidates: int = 0
    scored: int = 0
    top_n: int = 0
    model: str = ""
    mode: str = RERANK_MODE_REPLACE
    weight: float = 0.0
    dropped_by_min_score: int = 0
    empty_reason: str = ""
    latency_ms: float = 0.0
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.query_text, str) or not self.query_text.strip():
            raise RerankError(
                f"RerankResult.query_text 必须是非空字符串，收到 {self.query_text!r}："
                "重排分是对 (查询, 文档) 打的，没有查询这一侧的记录，"
                "这份名单就成了一份'不知道在问什么'的名次。"
            )
        if not isinstance(self.hits, tuple):
            raise RerankError(f"RerankResult.hits 必须是 tuple，收到 {type(self.hits).__name__}")
        if self.mode not in RERANK_MODES:
            raise RerankError(
                f"未知重排模式 {self.mode!r}：可用模式是 {'、'.join(RERANK_MODES)}。"
                "（replace 只按重排分重排；blend 先各自归一化再加权——"
                "两者的差别正是'信不信第一阶段的分数'）"
            )
        for name in ("candidates", "scored", "top_n", "dropped_by_min_score"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise RerankError(
                    f"RerankResult.{name} 必须是非负整数，收到 {value!r}。"
                    "这四个数各自是一笔账（输入 / 打分 / 窗口 / 阈值切掉），"
                    "负数说明有一笔账算错了。"
                )
        if self.scored > self.candidates:
            raise RerankError(
                f"RerankResult 的账不自洽：scored={self.scored} 大于 "
                f"candidates={self.candidates}。打分的条数不可能多于输入条数——"
                "两个数一起读是'窗口有没有生效'的唯一证据，不一致时它什么都不说明。"
            )
        if isinstance(self.weight, bool) or not isinstance(self.weight, (int, float)):
            raise RerankError(f"RerankResult.weight 必须是数字，收到 {type(self.weight).__name__}")
        if not math.isfinite(float(self.weight)) or not 0.0 <= float(self.weight) <= 1.0:
            raise RerankError(
                f"RerankResult.weight 必须是 [0, 1] 里的有限数，收到 {self.weight!r}。"
                "它是**重排分的权重**（第一阶段分数的权重是 1 - weight），"
                "越界会让 blend 的两个权重连同号或超过 1。"
            )
        if not math.isfinite(float(self.latency_ms)) or float(self.latency_ms) < 0.0:
            raise RerankError(f"RerankResult.latency_ms 必须是非负有限数，收到 {self.latency_ms!r}")
        if self.hits and self.empty_reason:
            raise RerankError(
                f"有 {len(self.hits)} 条命中却报告 empty_reason={self.empty_reason!r}："
                "两个字段互相矛盾的结果会让'为什么没有命中'这个问题失去意义。"
            )
        if not self.hits and not self.empty_reason:
            raise RerankError(
                "没有命中时必须给出 empty_reason（一句话说明为什么）："
                "空名单可能是'上游没给候选'、'被重排阈值切完'、'输入本身是空的'——"
                "三者的处置动作完全不同，合成一句'没有结果'等于什么都没说。"
            )
        for note in self.notes:
            if not isinstance(note, str) or not note.strip():
                raise RerankError(
                    f"RerankResult.notes 里出现了空条目 {note!r}："
                    "注记是给人读的，空串只会让报告里多一行空白。"
                )
        object.__setattr__(self, "weight", float(self.weight))
        object.__setattr__(self, "latency_ms", float(self.latency_ms))

    @property
    def count(self) -> int:
        """命中条数."""
        return len(self.hits)

    @property
    def is_empty(self) -> bool:
        """是否没有命中（``empty_reason`` 一定非空）."""
        return not self.hits

    @property
    def window(self) -> int:
        """窗口大小（本次真正送进重排器的那一段长度）."""
        return min(self.scored, self.top_n)

    def moved_ids(self) -> list[str]:
        """名次真的变了的记录 id（按**新**名次顺序）.

        "变了"包括两种情况，且两者都值得看：被重排器重新排序的那几条，
        以及因为别人被重排阈值切掉而前移的那几条（见 ``RerankHit.moved``）。
        """
        return [hit.record_id for hit in self.hits if hit.moved != 0]

    def top(self) -> RerankHit | None:
        """第一名（没有命中时返回 ``None``，而不是抛异常）."""
        return self.hits[0] if self.hits else None

    def to_dict(self, *, include_features: bool = True) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return {
            "query_text": self.query_text,
            "count": self.count,
            "candidates": self.candidates,
            "scored": self.scored,
            "window": self.window,
            "top_n": self.top_n,
            "model": self.model,
            "mode": self.mode,
            "weight": round(self.weight, 6),
            "dropped_by_min_score": self.dropped_by_min_score,
            "empty_reason": self.empty_reason,
            "latency_ms": self.latency_ms,
            "moved": self.moved_ids(),
            "notes": list(self.notes),
            "hits": [hit.to_dict(include_features=include_features) for hit in self.hits],
        }

    def to_summary(self) -> dict[str, Any]:
        """进上游结果的那一小段摘要（``RetrievalResult.rerank`` 直接用它）.

        为什么要有它：单路（``retriever``）与混合（``hybrid``）两个集成点都要
        把"这次重排做了什么"带进各自的 ``RetrievalResult``，而两份手写的摘要
        一定会分家（一个多了 ``mode``、一个忘了 ``scored``）。形状只在这里定义一次。

        **它刻意不含正文与特征**：摘要进的是每一份检索结果的公共字段，
        而 20 条 × 4 个特征放在那里会把它变成第二份 ``hits``——
        要看细节的人手里本来就有完整的 ``RerankResult``。

        **两个键名要看清楚，它们与 ``RetrievalResult`` 上的同名键不是一回事**：

        ```text
        candidates   重排的**输入条数**（阈值/融合之后交给它的那批）
                     ≠ RetrievalResult.candidates（库侧过滤之后的候选数）
        window       本次真正送进重排器的那一段长度 = min(scored, top_n)
        ```

        沿用形状自己的键名（而不是另起 ``incoming`` 之类的别名）是为了让
        ``result.rerank["candidates"]`` 与 ``RerankResult.candidates`` 逐字对得上；
        代价就是上面这一行必须写下来——同名字段两个口径，是这份代码里最容易
        读错的一处。
        """
        return {
            "model": self.model,
            "mode": self.mode,
            "params": {"top_n": self.top_n, "weight": self.weight},
            "candidates": self.candidates,
            "scored": self.scored,
            "window": self.window,
            "dropped_by_min_score": self.dropped_by_min_score,
            "moved": len(self.moved_ids()),
            "empty_reason": self.empty_reason,
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        head = "、".join(f"{hit.record_id}:{hit.score:.4f}" for hit in self.hits[:3])
        return (
            f"重排[{self.mode}] {self.model} | 候选 {self.candidates} | "
            f"打分 {self.scored}（窗口 {self.top_n}）| 阈值切掉 {self.dropped_by_min_score} | "
            f"命中 {self.count} | {head or '（无）'}"
        )

    def explain(self) -> list[str]:
        """人类可读诊断：窗口策略 / 模式与量纲 / 名次变化 / 注记.

        四段与四个问题的对应关系是固定的（"读一份重排诊断"的人要找的就是这四句）：

        ```text
        1) 只对多少条打了分、外面还有多少条       → 窗口纪律（以及它的代价）
        2) 用的是哪种模式、量纲怎么处理的         → "为什么分数不可比"这件事怎么被绕开的
        3) 谁的名次动了、动了多少                 → 重排的全部可见效果
        4) 阈值切掉了哪一批                       → 只有被打分过的条会被切
        ```
        """
        lines = [
            f"窗口：候选 {self.candidates} 条，只对前 top_n={self.top_n} 条打分"
            f"（真正打分 {self.scored} 条、窗口内 {self.window} 条）；"
            f"窗口外 {max(0, self.candidates - self.window)} 条一条都没打分，"
            "因此它的名次可能本应更靠前",
            f"模式：{self.mode}"
            + (
                f"，重排分权重 {self.weight}（第一阶段分数权重 {round(1.0 - self.weight, 6)}）"
                "—— 两份分数在**窗口内**各自 min-max 归一化后才相加"
                if self.mode == RERANK_MODE_BLEND
                else "—— 窗口内只按重排分排序，权重不参与（同一次重排内部无需归一化）"
            ),
        ]
        if self.dropped_by_min_score:
            lines.append(
                f"重排阈值：切掉 {self.dropped_by_min_score} 条（只作用于被打分过的 "
                f"{self.scored} 条）；窗口外的条既不会被它切、也不会被它救"
            )
        for hit in self.hits:
            if hit.moved:
                lines.append(f"名次变化：{hit.summary_line()}")
        if not any(hit.moved for hit in self.hits):
            lines.append(
                "名次变化：无（窗口内的重排次序与第一阶段一致——"
                "这**不代表重排没用**，只代表这次它没有改动任何一对相邻次序）"
            )
        if self.empty_reason:
            lines.append(f"空结果原因：{self.empty_reason}")
        for note in self.notes:
            lines.append(f"注记：{note}")
        return lines


# --------------------------------------------------------------------------- #
# 重排主入口
# --------------------------------------------------------------------------- #


def rerank_hits(
    hits: Sequence[Any],
    query_text: str,
    *,
    reranker: BaseReranker | None = None,
    top_n: int | None = None,
    mode: str | None = None,
    weight: float | None = None,
    min_score: float | None = None,
    model: str | None = None,
    elapsed: float | None = None,
) -> RerankResult:
    """给一批命中（**融合之后或阈值之后**）与一句话，重排它们（纯函数）.

    七步，顺序就是纪律：

    ```text
    1. 解析参数     None → 缺省值；所有非法参数当场报 RerankError（不静默纠正）
    2. 空输入早退   0 条 → 空结果（**不调用重排器**：一次都不打分为空结果负责）
    3. 切窗口       前 top_n 条进窗口，其余原样留作尾部
    4. 打分         reranker.score_pairs(query, 窗口内全部正文) —— **一次调用**
    5. 算归一化     blend 模式下两份分数各自在**窗口内** min-max（量纲不可比，见 docstring）
    6. 阈值落刀     min_score 只作用于被打分过的条；切掉几条记进 dropped_by_min_score
    7. 排序与装配   窗口内按定序键重排、重编名次，窗口外原顺序接在后面
    ```

    ```text
    replace  排序键 (-score,          stage1_rank, record_id)
    blend    排序键 (-blended,        stage1_rank, record_id)
    ```

    三个键缺一不可：只按分数排序时，"两条分数逐位相同"会让名次取决于
    ``sorted`` 的稳定性（那是**没写下来的**排序规则）；``record_id`` 是最后的兜底，
    保证同一份数据两次运行给出同一份名单。

    排序**第三键的对照物**是 ``stage1_rank`` 而不是上游的 ``rank``：
    前者是输入序列的位置（稠密），后者在阈值落刀之后会带空洞。
    """
    started = time.perf_counter()
    _check_query_text(query_text)
    resolved_reranker = _resolve_reranker(reranker)
    resolved_top_n = _resolve_top_n(top_n)
    resolved_mode = _resolve_mode(mode)
    resolved_weight = _resolve_weight(weight)
    _check_min_score(min_score)
    resolved_model = _resolve_model_name(model, resolved_reranker)
    _check_elapsed(elapsed)

    if isinstance(hits, (str, bytes)) or not isinstance(hits, Sequence):
        raise RerankError(
            f"rerank_hits 需要一批命中（序列），收到 {type(hits).__name__}。"
            "要重排请传 RetrievalHit 的列表/元组（本模块只读它的 "
            "record_id / score / rank / text / channels）；"
            "要重排一批纯文本请直接用 reranker.score_pairs(query, texts)。"
        )
    candidates = list(hits)

    # 第 2 步：空输入早退。**不调用重排器**：一次都不打分，因此它的计数器不动，
    # 而"这条查询没有候选"这件事由 empty_reason 说清楚（不是"重排没起作用"）。
    if not candidates:
        return RerankResult(
            query_text=query_text,
            candidates=0,
            scored=0,
            top_n=resolved_top_n,
            model=resolved_model,
            mode=resolved_mode,
            weight=_weight_for(resolved_mode, resolved_weight),
            empty_reason=(
                "没有候选可比：上游（阈值 / 融合）之后一条都没有，"
                "因此这次**没有调用重排器**（它的 calls 计数器仍然是 0）"
            ),
            latency_ms=_elapsed_ms(started, elapsed),
            notes=(
                "空输入不调用重排器：'没有候选'不是重排的问题，"
                "要查它请回头看上游那两道减法（dropped_below_threshold / "
                "dropped_by_diversity）的账",
            ),
        )

    # 第 3 步：切窗口。窗口是"前 top_n 条"，尾部保持原顺序（**一个都不打分**）。
    window = candidates[:resolved_top_n]
    tail = candidates[resolved_top_n:]
    stage1 = [_read_stage1_hit(item, position=position) for position, item in enumerate(window)]
    texts = [row.text for row in stage1]

    # 第 4 步：一次调用。**窗口之外一条都不进来**——这是本模块的成本纪律，
    # 也是"top_n 之外的条数一次都没被打分"这条断言能被验证的原因。
    scores = resolved_reranker.score_pairs(query_text, texts)
    if len(scores) != len(texts):
        raise RerankError(
            f"重排器 {resolved_reranker.name!r} 给了 {len(scores)} 个分数，"
            f"而窗口里有 {len(texts)} 条文本：长度不一致。分数与文本错位**不会报错**"
            "（两个都是列表），只会让每一段拿到别人的分数。"
            "出路：让 score_pairs 返回与 texts 等长同序的列表。"
        )
    clean_scores = [_check_score(value, label="score_pairs") for value in scores]
    feature_rows = _feature_rows(resolved_reranker, query_text, texts)

    # 第 5 步：blend 模式下两份分数**各自**在窗口内归一化（量纲不可比，
    # 与 fusion._min_max_normalize 是同一条纪律，因此复用同一个实现）。
    blended_of: dict[int, float] = {}
    if resolved_mode == RERANK_MODE_BLEND:
        resolved_weight_value = _weight_for(resolved_mode, resolved_weight)
        normalized_rerank = _min_max_normalize(
            {str(index): value for index, value in enumerate(clean_scores)}
        )
        normalized_stage1 = _min_max_normalize(
            {str(index): row.score for index, row in enumerate(stage1)}
        )
        blended_of = {
            index: resolved_weight_value * normalized_rerank[str(index)]
            + (1.0 - resolved_weight_value) * normalized_stage1[str(index)]
            for index in range(len(stage1))
        }

    # 第 6 步：阈值落刀（只作用于被打分过的条）。
    payloads = [
        _RerankPayload(
            index=index,
            record_id=row.record_id,
            score=clean_scores[index],
            blended=blended_of.get(index),
            stage1_score=row.score,
            stage1_rank=row.stage1_rank,
            features=feature_rows[index],
            channels=row.channels,
        )
        for index, row in enumerate(stage1)
    ]
    kept: list[_RerankPayload] = []
    dropped = 0
    for payload in payloads:
        if min_score is not None and payload.score < float(min_score):
            dropped += 1
            continue
        kept.append(payload)

    # 第 7 步：排序与装配。窗口内按定序键重排，窗口外原顺序接在后面，
    # 名次在**整份名单**上重编成 0 起连续（"第 3 条"必须只有一个含义）。
    if resolved_mode == RERANK_MODE_BLEND:
        kept.sort(key=lambda item: (-(item.blended or 0.0), item.stage1_rank, item.record_id))
    else:
        kept.sort(key=lambda item: (-item.score, item.stage1_rank, item.record_id))

    hits_out = [
        RerankHit(
            record_id=payload.record_id,
            score=payload.score,
            rank=position,
            stage1_score=payload.stage1_score,
            stage1_rank=payload.stage1_rank,
            blended=payload.blended,
            features=dict(payload.features),
            channels=payload.channels,
        )
        for position, payload in enumerate(kept)
    ]
    for offset, item in enumerate(tail):
        row = _read_stage1_hit(item, position=len(window) + offset)
        hits_out.append(
            RerankHit(
                record_id=row.record_id,
                score=0.0,
                rank=len(kept) + offset,
                stage1_score=row.score,
                stage1_rank=row.stage1_rank,
                blended=None,
                features={},
                channels=row.channels,
                scored=False,
            )
        )

    # 空名单只有一种成因（上游给了候选、但窗口内全被重排阈值切掉，且没有窗口外尾部），
    # 而 ``RerankResult`` 的不变量要求"没有命中时必须给出一句话"——因此这里必须
    # **显式**把它写出来，否则这一条合法路径会变成一次构造期异常（"切完之后为空"
    # 是阈值定得太高，不是重排失败）。
    notes = _build_notes(
        candidates=len(candidates),
        window=len(window),
        tail=len(tail),
        top_n=resolved_top_n,
        mode=resolved_mode,
        weight=resolved_weight,
        min_score=min_score,
        dropped=dropped,
        reranker=resolved_reranker,
        model=model,
    )
    empty_reason = ""
    if not hits_out:
        empty_reason = (
            f"重排阈值 {min_score} 把窗口内的 {len(window)} 条全切掉了，"
            "而窗口外没有别的候选：请降低重排阈值或不设阈值"
            "（上游给的候选不是空的——被切掉的是**被打过分的**那批，"
            "它由 dropped_by_min_score 记账）"
        )
        notes.append(
            "这次重排一条都没留下：**不是**重排器失败，而是阈值定得比全部分数都高。"
            "处置：降低 retrieval_rerank_min_score，或把它设回 None 先看清分数分布。"
        )
    result = RerankResult(
        query_text=query_text,
        hits=tuple(hits_out),
        candidates=len(candidates),
        scored=len(window),
        top_n=resolved_top_n,
        model=resolved_model,
        mode=resolved_mode,
        weight=_weight_for(resolved_mode, resolved_weight),
        dropped_by_min_score=dropped,
        empty_reason=empty_reason,
        latency_ms=_elapsed_ms(started, elapsed),
        notes=tuple(notes),
    )
    logger.debug(
        "重排：查询 %r，候选 %d 条，窗口 %d 条（打分 %d 条），模式 %s，移动 %d 条，阈值切掉 %d 条",
        query_text[:40],
        result.candidates,
        result.window,
        result.scored,
        result.mode,
        len(result.moved_ids()),
        result.dropped_by_min_score,
    )
    return result


@dataclass(frozen=True)
class _RerankPayload:
    """窗口内一条命中的**中间态**（打分完、排序前）——私有，只为让排序键只写一份.

    它不进 ``__all__``：这不是对外形状，而是"算好了但还没定序"的那一步的容器。
    之所以要它，是因为 ``RerankHit.rank`` 依赖最终次序，而最终次序依赖
    （分数、blended、stage1_rank、record_id）四个数——没有它就得把这些数
    并排躺在几个平行列表里，而那正是"字段错位"最容易发生的地方。
    """

    index: int
    record_id: str
    score: float
    blended: float | None
    stage1_score: float
    stage1_rank: int
    features: dict[str, float]
    channels: tuple[str, ...]


@dataclass(frozen=True)
class _Stage1Row:
    """窗口/尾部的输入行（从上游命中里读出五样东西）——私有，见 ``_read_stage1_hit``."""

    record_id: str
    score: float
    rank: int
    text: str
    channels: tuple[str, ...]
    stage1_rank: int


def _read_stage1_hit(item: Any, *, position: int) -> _Stage1Row:
    """从一条上游命中里读出 ``(id, 分数, 名次, 正文, 通道)``（缺什么就说什么）.

    只读五个属性、**不做任何鸭子类型上的宽容**：传进来一个字典
    （``{"record_id": ...}``）时要当场响，因为"字典也能取到这几个键"
    会让错误的用法活到排序那一步（那里比较两个不同类型的值不会报错）。
    """
    where = f"第 {position} 条命中"
    for attribute in ("record_id", "score", "rank"):
        if not hasattr(item, attribute):
            raise RerankError(
                f"{where} 没有 {attribute} 属性（收到 {type(item).__name__}）："
                "重排只要求命中提供 record_id / score / rank（外加可选的 text / channels），"
                "RetrievalHit 与 LexicalHit 都满足。"
                "出路：传这两者之一，或用 reranker.score_pairs 直接对纯文本打分。"
            )
    record_id = getattr(item, "record_id")
    if not isinstance(record_id, str) or not record_id.strip():
        raise RerankError(f"{where} 的 record_id 必须是非空字符串，收到 {record_id!r}")
    score = _check_score(getattr(item, "score"), label=f"{where} 的 score")
    rank = getattr(item, "rank")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
        raise RerankError(
            f"{where} 的 rank 必须是非负整数（从 0 起），收到 {rank!r}。"
            "它会被原样记进 stage1_rank（'它原来在第几'的答案）。"
        )
    text = getattr(item, "text", "")
    if not isinstance(text, str):
        raise RerankError(
            f"{where} 的 text 必须是字符串，收到 {type(text).__name__}："
            "重排分是对 (查询, 正文) 算的，没有正文本模块无从打分"
            "（空串是合法的，它会让这条特征全为 0）。"
        )
    channels = getattr(item, "channels", ())
    if not isinstance(channels, tuple):
        raise RerankError(
            f"{where} 的 channels 必须是 tuple，收到 {type(channels).__name__}："
            "它是从上游带过来的多路证据（未记录时用空元组 ()），"
            "重排会把它原样传给 RerankHit。"
        )
    return _Stage1Row(
        record_id=record_id,
        score=score,
        rank=rank,
        text=text,
        channels=channels,
        stage1_rank=position,
    )


def _feature_rows(
    reranker: BaseReranker,
    query: str,
    texts: Sequence[str],
) -> list[dict[str, float]]:
    """问重排器要一份"每条的四个特征"（真实模型返回空字典，这是正确实现）."""
    rows = reranker.explain_pairs(query, texts)
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise RerankError(
            f"重排器 {reranker.name!r} 的 explain_pairs 返回了 {type(rows).__name__}，"
            "而本层要求序列（与 texts 等长同序）。出路：自定义重排器里"
            "给每条返回一个 dict；真实模型没有可摆出的特征时返回 {} 就是正确实现。"
        )
    if len(rows) != len(texts):
        raise RerankError(
            f"重排器 {reranker.name!r} 的 explain_pairs 返回 {len(rows)} 行特征，"
            f"而窗口里有 {len(texts)} 条文本。"
            "出路：让它的长度与 texts 一致（或直接删掉这个方法的覆盖，"
            "用 BaseReranker 的缺省实现返回空字典）。"
        )
    result: list[dict[str, float]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise RerankError(
                f"重排器 {reranker.name!r} 的 explain_pairs 返回了 {type(row).__name__} 项，"
                "而每一项必须是一个 dict（键落在 RERANK_FEATURE_NAMES 里）。"
            )
        result.append({str(name): float(value) for name, value in row.items()})
    return result


def _build_notes(
    *,
    candidates: int,
    window: int,
    tail: int,
    top_n: int,
    mode: str,
    weight: float | None,
    min_score: float | None,
    dropped: int,
    reranker: BaseReranker,
    model: str | None,
) -> list[str]:
    """拼出这次重排的注记（**窗口纪律那条是无条件的**，其余按情况）."""
    notes = [
        f"只对前 N 条打分（N=top_n={top_n}）：候选 {candidates} 条，"
        f"窗口内 {window} 条全部打过分；窗口外的 {tail} 条**一条都没打分**——"
        "因此它的名次可能本应更靠前，而这次重排看不见它。"
        "这条纪律换来的是成本：交叉编码器是每对一次前向推理，"
        "不切窗口就等于把整份候选都重算一遍。"
    ]
    if tail:
        notes.append(
            f"窗口外的 {tail} 条按**原顺序**接在重排结果之后（它们的 stage1_rank "
            "保持原值、scored=False、score=0.0）：0.0 不是'最不相关'，"
            "而是'没有分'——要让它有机会被提前，请调大 top_n。"
        )
    if min_score is not None:
        notes.append(
            f"min_score={min_score} **只作用于被打分过的 {window} 条**（本次切掉 "
            f"{dropped} 条）：窗口外的 {tail} 条不受它影响——它们连分数都没有。"
            "比的是重排分（教学替身落在 [0, 1]，因此这个阈值是可以标定的；"
            "与关键词路 BM25 的'没有绝对标度'正好相反）。"
        )
    if mode == RERANK_MODE_REPLACE and weight is not None:
        notes.append(
            f"mode='replace' 不看权重：weight={weight} 已记录在结果里但没有参与排序。"
            "它与 fusion 的做法是**刻意的不同选择**（那里 strategy='rrf' 收到 weights "
            "时直接报错）：RRF 的定义里没有权重，而这里权重只是另一种模式的参数——"
            "报错会让'试一下 blend 的区别'变成一次必须先删参数的实验。"
        )
    if model is not None and model != reranker.name:
        notes.append(
            f"报告的模型名 {model!r} 与实现的名字 {reranker.name!r} 不一致："
            "本模块**不按名字加载模型**（那是生产环境接真实重排服务的事），"
            f"这次打分用的仍然是 {reranker.name!r}——这个名字只进报告。"
        )
    return notes


# --------------------------------------------------------------------------- #
# 提升评估（三条指标 + 一份对照报告）
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LiftProbe:
    """一条评估探针：**一句话 + 它的金标准相关 id**.

    ``relevant`` 是这份报告里唯一来自外部的东西（人工标注或既有评测集），
    因此它的形状要足够严格：不能是空元组（没有金标准就没有分母，
    ``recall@k`` 会变成 0/0）。**它必须是 tuple**（照 ``RetrievalHit.channels``
    的纪律：frozen 形状里塞一个可变对象，"这份标注不曾被改过"就不再成立）。
    """

    query: str
    relevant: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise RerankError(
                f"LiftProbe.query 必须是非空字符串，收到 {self.query!r}："
                "探针要被真的送进检索器，空查询在检索那一层本来就会被拒。"
            )
        if not isinstance(self.relevant, tuple):
            raise RerankError(
                f"LiftProbe.relevant 必须是 tuple，收到 {type(self.relevant).__name__}。"
                "写法：relevant=('c-a-01', 'c-b-02')；frozen 形状里塞一个 list "
                "会让'这份金标准不曾被改过'不再成立。"
            )
        if not self.relevant:
            raise RerankError(
                f"LiftProbe.relevant 不能为空（query={self.query!r}）："
                "三条指标的分母都由它给出，空标注会让 recall@k 变成 0/0——"
                "而它算出来的 0.0 看起来像是'一条都没召回到'。"
                "出路：给这条查询标出至少一条相关记录，或把这条探针删掉。"
            )
        for item in self.relevant:
            if not isinstance(item, str) or not item.strip():
                raise RerankError(
                    f"LiftProbe.relevant 里出现了非字符串或空串：{item!r}。"
                    "金标准里存的是记录 id，空 id 永远匹配不上任何命中。"
                )

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return {"query": self.query, "relevant": list(self.relevant)}

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        return f"探针 {self.query!r} | 金标准 {len(self.relevant)} 条：{list(self.relevant)}"


def recall_at_k(ids: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """``recall@k = |前 k 条 ∩ 金标准| / |金标准|``——**有没有把它捞回来**.

    两个口径写在明面上，因为它们决定了同一个数字：

    ```text
    分母   **去重后**的金标准条数（标了两次同一条不会把分母抬高）
    分子   前 k 条里落在金标准里的**条数**（同一条在前 k 条里出现两次只算一次）
    ```

    这条指标对"排名"完全不敏感：把相关的那条从第 5 名提到第 1 名，它一动不动。
    因此它必须与 ``reciprocal_rank`` / ``ndcg_at_k`` 一起看（见模块 docstring）。
    """
    checked_k = _check_k(k)
    pool = _check_relevant(relevant, label="recall_at_k")
    head = _check_ids(ids, label="recall_at_k")[:checked_k]
    if not pool:
        return 0.0
    return len(set(head) & pool) / len(pool)


def reciprocal_rank(ids: Sequence[str], relevant: Sequence[str]) -> float:
    """``RR = 1 / (第一个相关命中的位置 + 1)``（一个都没有则为 ``0.0``）.

    位置从 0 起，因此"第一条就相关"给 1.0、"第二条才相关"给 0.5——
    与 ``fusion`` 里 RRF 的 ``1/(k+rank+1)`` 是同一个形状（那里从名次算，
    这里从位置算），两处的"第几名"都从 0 起，这一点一致很重要。
    """
    pool = _check_relevant(relevant, label="reciprocal_rank")
    for position, record_id in enumerate(_check_ids(ids, label="reciprocal_rank")):
        if record_id in pool:
            return 1.0 / (position + 1)
    return 0.0


def ndcg_at_k(ids: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """``nDCG@k = DCG@k / IDCG@k``（二值增益）——**命中几条**与**排得多靠前**一起看.

    ```text
    DCG@k  = Σ_{i=0}^{k-1} rel_i / log2(i + 2)      rel_i = 1 若第 i 条相关（首次出现）
    IDCG@k = Σ_{i=0}^{m-1} 1 / log2(i + 2)          m = min(|金标准|, k)
    ```

    三个细节都是刻意选的：

    ```text
    折损用 log2(i + 2)      i 从 0 起，因此第一条的折损是 log2(2)=1（不折损）
    理想名单用 min(|金标准|, k) 条   金标准比 k 多时，理想 DCG 也只数前 k 条
    重复 id 只算一次        第 2 次出现按 0 增益——否则一份"把同一条刷满"的名单
                          会拿到大于 1 的 nDCG（分母却只按去重的金标准算）
    ```
    """
    checked_k = _check_k(k)
    pool = _check_relevant(relevant, label="ndcg_at_k")
    head = _check_ids(ids, label="ndcg_at_k")[:checked_k]
    if not pool or not head:
        return 0.0
    dcg = 0.0
    seen: set[str] = set()
    for position, record_id in enumerate(head):
        if record_id in pool and record_id not in seen:
            dcg += 1.0 / math.log2(position + 2.0)
        seen.add(record_id)
    ideal = sum(1.0 / math.log2(position + 2.0) for position in range(min(len(pool), checked_k)))
    if ideal <= 0.0:
        return 0.0
    return dcg / ideal


@dataclass(frozen=True)
class LiftReport:
    """重排前后的对照报告：**三条指标的均值 + 逐条明细**.

    ```text
    queries    评了几条探针
    k          指标里的那个 k（三条指标共用它，因此它只写一次）
    before     重排前的三条均值（{"recall": …, "reciprocal_rank": …, "ndcg": …}）
    after      重排后的三条均值
    lift       after - before（**正数 = 重排在这条指标上更好**）
    per_query  逐条明细：两侧的名单、三条指标、谁的名次动了、窗口的账
    ```

    ``lift`` 的差值而不是比值：比值在 ``before == 0`` 时是无穷（"从 0 提升到
    0.2"没有倍数可言），而差值在两条指标都落在 ``[0, 1]`` 时始终可读。

    空探针列表返回一份"三条指标全 0"的报告而不是报错：与
    ``aggregate_results`` 的空批次、``mean()`` 的"没有数据应得 0 分"是同一条纪律
    （但 ``queries=0`` 会如实写在那里——0 分与"评了 0 条"必须一起被看见）。
    """

    queries: int
    k: int
    before: dict[str, float] = field(default_factory=dict)
    after: dict[str, float] = field(default_factory=dict)
    lift: dict[str, float] = field(default_factory=dict)
    per_query: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.queries, int) or isinstance(self.queries, bool) or self.queries < 0:
            raise RerankError(
                f"LiftReport.queries 必须是非负整数，收到 {self.queries!r}："
                "它是'这份报告基于几条探针'，而均值正是在它上面算的。"
            )
        if not isinstance(self.k, int) or isinstance(self.k, bool) or self.k < 1:
            raise RerankError(
                f"LiftReport.k 必须是 >= 1 的整数，收到 {self.k!r}："
                "三条指标共用这个 k（recall@k / nDCG@k），k=0 会让两个指标恒为 0。"
            )
        if set(self.before) != set(self.after) or set(self.before) != set(self.lift):
            raise RerankError(
                "LiftReport 的三份指标表键不一致："
                f"before={sorted(self.before)}、after={sorted(self.after)}、"
                f"lift={sorted(self.lift)}。三张表必须逐键对齐——"
                "少一个键时'这一条提升了多少'就无法回答，而它正是这份报告的全部意义。"
            )
        for label, table in (
            ("before", self.before),
            ("after", self.after),
            ("lift", self.lift),
        ):
            for name, value in table.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise RerankError(
                        f"LiftReport.{label}[{name!r}] 必须是数字，收到 {type(value).__name__}"
                    )
                if not math.isfinite(float(value)):
                    raise RerankError(
                        f"LiftReport.{label}[{name!r}] 必须是有限数，收到 {value!r}："
                        "nan 的均值会让'这次提升是正是负'永远无法回答。"
                    )
        if not isinstance(self.per_query, tuple):
            raise RerankError(
                f"LiftReport.per_query 必须是 tuple，收到 {type(self.per_query).__name__}"
            )

    @property
    def metrics(self) -> tuple[str, ...]:
        """这份报告里的指标名（**字典序**，与投影里的顺序一致）."""
        return tuple(sorted(self.before))

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return {
            "queries": self.queries,
            "k": self.k,
            "metrics": list(self.metrics),
            "before": {name: round(self.before[name], 6) for name in self.metrics},
            "after": {name: round(self.after[name], 6) for name in self.metrics},
            "lift": {name: round(self.lift[name], 6) for name in self.metrics},
            "per_query": [dict(item) for item in self.per_query],
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要（逐指标一行太长，因此拼在同一行里）."""
        rendered = "、".join(
            f"{name} {self.before[name]:.4f} → {self.after[name]:.4f}（{self.lift[name]:+.4f}）"
            for name in self.metrics
        )
        return f"重排提升（{self.queries} 条探针 / k={self.k}）：{rendered or '（无指标）'}"


def measure_lift(
    probes: Sequence[LiftProbe],
    retrieve: Callable[[str], Any],
    reranker: BaseReranker | None = None,
    *,
    k: int = 5,
    top_n: int | None = None,
    mode: str | None = None,
    weight: float | None = None,
) -> LiftReport:
    """逐条探针量"重排前后"的三条指标，返回一份对照报告（**评估提升的落地**）.

    每条探针做四件事，顺序固定：

    ```text
    1. retrieve(probe.query)       拿一份 RetrievalResult（本模块只读它的 .hits/.ids()）
    2. before_ids = result.ids()   算重排前的三条指标
    3. rerank_hits(result.hits, …) 用**同一批命中**重排（因此 before/after 只差一个变量：
                                   "有没有重排"，不是"检索了两次"）
    4. after_ids                   算重排后的三条指标，并把逐条明细记下来
    ```

    第 3 步用**同一批命中**这一点很要紧：如果 after 是"重排之后重新检索一遍"，
    那么两条指标的差里就混着"两次检索的随机性"，而这份报告的全部价值在于
    **把变量减到一个**（有重排 / 无重排）。

    ``retrieve`` 是一个可调用对象而不是"检索器实例"：探针的描述里只有一句话，
    怎么把它变成一份结果（单路 / 混合 / 带过滤）由调用方决定——
    本函数因此可以对单路与混合两种检索器给出**同一份口径**的提升报告。

    三条指标的公式见模块 docstring；它们的实现是三个纯函数，
    因此这张对照表可以被手算核对（不需要任何真实模型）。
    """
    checked_k = _check_k(k)
    if not isinstance(probes, Sequence) or isinstance(probes, (str, bytes)):
        raise RerankError(
            f"measure_lift 需要一批 LiftProbe（序列），收到 {type(probes).__name__}。"
            "写法：measure_lift([LiftProbe(query='…', relevant=('c-a-01',))], retriever)"
        )
    if not callable(retrieve):
        raise RerankError(
            f"measure_lift 的 retrieve 必须可调用（一句话 → RetrievalResult），"
            f"收到 {type(retrieve).__name__}。"
            "出路：传 retriever.retrieve 或 hybrid.retrieve（不要传检索器本身）。"
        )
    for index, probe in enumerate(probes):
        if not isinstance(probe, LiftProbe):
            raise RerankError(
                f"第 {index} 条探针不是 LiftProbe，收到 {type(probe).__name__}。"
                "写法：LiftProbe(query='阈值怎么设', relevant=('c-a-01',))"
            )
    resolved_reranker = _resolve_reranker(reranker)

    before_rows: dict[str, list[float]] = {name: [] for name in _METRIC_NAMES}
    after_rows: dict[str, list[float]] = {name: [] for name in _METRIC_NAMES}
    per_query: list[dict[str, Any]] = []
    for probe in probes:
        result = retrieve(probe.query)
        if not hasattr(result, "hits") or not hasattr(result, "ids"):
            raise RerankError(
                f"retrieve({probe.query!r}) 返回了 {type(result).__name__}，"
                "而本层要的是一份 RetrievalResult（要有 .hits 与 .ids()）。"
                "出路：传 retriever.retrieve / hybrid.retrieve，"
                "不要传一个自己拼的字典。"
            )
        before_ids = [str(item) for item in result.ids()]
        reranked = rerank_hits(
            result.hits,
            probe.query,
            reranker=resolved_reranker,
            top_n=top_n,
            mode=mode,
            weight=weight,
        )
        after_ids = [hit.record_id for hit in reranked.hits]
        before_metrics = _metrics_of(before_ids, probe.relevant, checked_k)
        after_metrics = _metrics_of(after_ids, probe.relevant, checked_k)
        row: dict[str, Any] = {
            "query": probe.query,
            "relevant": list(probe.relevant),
            "k": checked_k,
            "before_ids": before_ids,
            "after_ids": after_ids,
            "before": before_metrics,
            "after": after_metrics,
            "moved": reranked.moved_ids(),
            "candidates": reranked.candidates,
            "scored": reranked.scored,
            "top_n": reranked.top_n,
            "mode": reranked.mode,
            "weight": reranked.weight,
        }
        row["lift"] = {
            name: round(after_metrics[name] - before_metrics[name], 6) for name in _METRIC_NAMES
        }
        for name in _METRIC_NAMES:
            before_rows[name].append(float(before_metrics[name]))
            after_rows[name].append(float(after_metrics[name]))
        per_query.append(row)

    before = {name: _mean(values) for name, values in before_rows.items()}
    after = {name: _mean(values) for name, values in after_rows.items()}
    return LiftReport(
        queries=len(probes),
        k=checked_k,
        before=before,
        after=after,
        lift={name: round(after[name] - before[name], 6) for name in _METRIC_NAMES},
        per_query=tuple(per_query),
    )


def _metrics_of(ids: Sequence[str], relevant: Sequence[str], k: int) -> dict[str, float]:
    """三条指标一起算（**一次**遍历口径的入口，避免逐条明细里出现四份近似代码）."""
    return {
        "recall": round(recall_at_k(ids, relevant, k), 6),
        "reciprocal_rank": round(reciprocal_rank(ids, relevant), 6),
        "ndcg": round(ndcg_at_k(ids, relevant, k), 6),
    }


def _mean(values: Sequence[float]) -> float:
    """均值：没有数据时给 ``0.0``（而不是除零）——与 ``aggregate_results`` 同一条纪律."""
    if not values:
        return 0.0
    return round(math.fsum(values) / len(values), 6)


# --------------------------------------------------------------------------- #
# 入参校验（构造期用 _resolve_*，纯校验用 _check_*）
# --------------------------------------------------------------------------- #


def _resolve_reranker(value: BaseReranker | None) -> BaseReranker:
    """``reranker``：``None`` → 教学级替身（``CrossEncoderReranker``）.

    ``None`` 的语义与 ``Retriever`` 的其它 ``None`` 参数一致：**"没指定，用缺省"**。
    缺省值是替身而不是"不重排"——要不要重排由 ``enabled`` 决定，
    而"重排一次都没打分"从来不是一个选项（那等于没开重排却记了一笔账）。
    """
    if value is None:
        return CrossEncoderReranker()
    if not isinstance(value, BaseReranker):
        raise RerankError(
            f"reranker 必须是 BaseReranker，收到 {type(value).__name__}。"
            "要接真实重排模型请写一个 BaseReranker 子类（实现 score_pairs 与 name），"
            "不要传一个函数或一个字符串模型名——本模块**不按名字加载模型**。"
        )
    return value


def _resolve_top_n(value: int | None) -> int:
    """``top_n``：``None`` → ``DEFAULT_RERANK_TOP_N``（必须是 >= 1 的整数）."""
    resolved = DEFAULT_RERANK_TOP_N if value is None else value
    if not isinstance(resolved, int) or isinstance(resolved, bool) or resolved < 1:
        raise RerankError(
            f"top_n 必须是 >= 1 的整数，收到 {resolved!r}。"
            "'只对前 0 条打分'不是一个请求；要关掉重排请把 rerank_enabled 设为 False"
            "（那条路上一个字段都不会被改写，因此结果逐位不变）。"
        )
    return resolved


def _resolve_mode(value: str | None) -> str:
    """``mode``：``None`` → ``RERANK_MODE_REPLACE``（必须是封闭清单里的一个）."""
    resolved = RERANK_MODE_REPLACE if value is None else value
    if not isinstance(resolved, str) or resolved not in RERANK_MODES:
        raise RerankError(
            f"未知重排模式 {resolved!r}：可用模式是 {'、'.join(RERANK_MODES)}。"
            "（replace 只按重排分重排；blend 先把两份分数各自归一化再加权）"
            "拼错的模式名会让调用方以为换了一种策略，而结果看起来'只是没什么变化'。"
        )
    return resolved


def _resolve_weight(value: float | None) -> float | None:
    """``weight``：``None`` 保持 ``None``（由模式决定缺省），否则校验落在 ``[0, 1]``."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RerankError(f"weight 必须是数字或 None，收到 {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise RerankError(
            f"weight 必须是有限数，收到 {value!r}。"
            "nan 参与的比较恒为 False，于是 blend 的加权和会整列变成 nan。"
        )
    if not 0.0 <= number <= 1.0:
        raise RerankError(
            f"weight={number} 越界：它是**重排分的权重**，必须落在 [0, 1]"
            "（第一阶段分数的权重是 1 - weight）。"
            "要表达'只信重排分'请用 mode='replace' 或 weight=1.0。"
        )
    return number


def _resolve_model_name(value: str | None, reranker: BaseReranker) -> str:
    """``model``：``None`` → ``reranker.name``；给了就必须是非空字符串（只进报告）."""
    if value is None:
        return reranker.name
    if not isinstance(value, str) or not value.strip():
        raise RerankError(
            f"model 必须是非空字符串或 None，收到 {value!r}。"
            "它是**报告里的模型名**（缺省取 reranker.name）；空串会让报告里"
            "'这次是谁打的分'这一栏空白，而空白最容易被读成'没有重排'。"
        )
    return value.strip()


def _check_feature_weights(weights: tuple[float, ...]) -> None:
    """校验四个特征的权重：等长、非负有限、**和逐位等于 1.0**（理由见特征权重常量）."""
    if len(weights) != len(RERANK_FEATURE_NAMES):
        raise RerankError(
            f"FEATURE_WEIGHTS 有 {len(weights)} 项，而特征有 "
            f"{len(RERANK_FEATURE_NAMES)} 个（{'、'.join(RERANK_FEATURE_NAMES)}）："
            "两者必须一一对应，否则分数会漏掉某个特征或把两套顺序错位。"
        )
    for name, weight in zip(RERANK_FEATURE_NAMES, weights):
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise RerankError(
                f"FEATURE_WEIGHTS 里 {name} 的权重必须是数字，收到 {type(weight).__name__}"
            )
        if not math.isfinite(float(weight)) or float(weight) < 0.0:
            raise RerankError(
                f"FEATURE_WEIGHTS 里 {name} 的权重必须是非负有限数，收到 {weight!r}。"
                "负权重会让'特征越明显、分数越低'，那不是任何模型想要的偏置。"
            )
    if math.fsum(float(item) for item in weights) != 1.0:
        raise RerankError(
            f"FEATURE_WEIGHTS 之和必须逐位等于 1.0，当前是 "
            f"{math.fsum(float(item) for item in weights)!r}。"
            "和等于 1 保证分数始终落在 [0, 1]（四个特征各自都不超过 1），"
            "而 min_score 的可标定性正建立在这条不变量上；"
            "本模块刻意选二进制精确可表示的权重（1/2、1/4、1/8、1/8），"
            "因此这里可以用精确比较，而不必引入一个需要标定的 tolerance。"
        )


def _check_query_text(query: str) -> None:
    """查询必须是非空字符串（与 ``RetrievalQuery.text`` 同一条纪律）."""
    if not isinstance(query, str):
        raise RerankError(
            f"重排的查询必须是字符串，收到 {type(query).__name__}。"
            "要重排一批 id 请先把它们变成命中（那需要正文）；"
            "要重排一段文本请直接用 reranker.score_pairs(query, texts)。"
        )
    if not query.strip():
        raise RerankError(
            "空查询在重排里没有定义：请给一段非空文本。"
            "空白串会让'整句包含'这一项对每条都无意义，而分数看起来仍然正常——"
            "一个不会报错、只会让重排失去依据的输入。"
        )


def _check_text_sequence(texts: Sequence[str], *, label: str) -> list[str]:
    """一批待打分的文本：必须是序列、每一项都是字符串（空序列合法）."""
    if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
        raise RerankError(
            f"{label} 的 texts 必须是序列（列表/元组），收到 {type(texts).__name__}。"
            "一段文本会被按字符逐个当成'一篇文章'——一个能把逐字打分当成批量打分的坑。"
            "出路：传 ['第一段', '第二段']，空输入传 []。"
        )
    cleaned: list[str] = []
    for position, text in enumerate(texts):
        if not isinstance(text, str):
            raise RerankError(
                f"{label} 的第 {position} 项必须是字符串，收到 {type(text).__name__}。"
                "命中对象请先取出它的 text；本方法只认文本。"
            )
        cleaned.append(text)
    return cleaned


def _check_score(value: Any, *, label: str) -> float:
    """一个分数必须是非布尔的有限数（``float("1.0")`` 能被转出来，因此先判类型）."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RerankError(
            f"{label} 必须是数字，收到 {type(value).__name__}（{value!r}）。"
            "字符串形式的分数会被排序当成字典序比较——一个与相关性无关的顺序。"
        )
    if not math.isfinite(float(value)):
        raise RerankError(
            f"{label} 必须是有限数，收到 {value!r}。"
            "nan 会让'这条排第几'变成未定义（它比任何数都既不大于也不小于）。"
        )
    return float(value)


def _check_min_score(min_score: float | None) -> None:
    """``min_score``：``None`` 或有限数（阈值本身不设范围：重排分不一定在 [0, 1]）."""
    if min_score is None:
        return
    if isinstance(min_score, bool) or not isinstance(min_score, (int, float)):
        raise RerankError(f"min_score 必须是数字或 None，收到 {type(min_score).__name__}")
    if not math.isfinite(float(min_score)):
        raise RerankError(
            f"min_score 必须是有限数，收到 {min_score!r}。"
            "nan 与任何分数比较都返回 False，于是**一条都不会留下**，"
            "而它在配置里看起来只是一个阈值。"
        )


def _check_elapsed(elapsed: float | None) -> None:
    """``elapsed``：``None``（自己量）或非负有限数（调用方量好了传进来）."""
    if elapsed is None:
        return
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
        raise RerankError(
            f"elapsed 必须是数字或 None，收到 {type(elapsed).__name__}。"
            "它是'调用方已经量好的毫秒数'，没有量过请留 None（本函数自己量）。"
        )
    if not math.isfinite(float(elapsed)) or float(elapsed) < 0.0:
        raise RerankError(
            f"elapsed 必须是非负有限数，收到 {elapsed!r}：耗时不是负数，"
            "而负数会让'这次重排贵不贵'这个问题得到一个编出来的答案。"
        )


def _check_k(k: int) -> int:
    """``k`` 必须是 >= 1 的整数（三条指标共用它；``k=0`` 会让两条指标恒为 0）."""
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise RerankError(
            f"k 必须是 >= 1 的整数，收到 {k!r}。"
            "recall@k 与 nDCG@k 的名字里带着它：k=0 不是一个'只看零条'的请求，"
            "而是让两条指标恒为 0（看起来像'重排毫无效果'）。"
        )
    return k


def _check_relevant(relevant: Sequence[str], *, label: str) -> set[str]:
    """金标准：序列、每一项是非空字符串（**允许重复，会被去重**）."""
    if isinstance(relevant, (str, bytes)) or not isinstance(relevant, Sequence):
        raise RerankError(
            f"{label} 的 relevant 必须是序列（列表/元组），收到 {type(relevant).__name__}。"
            "一条金标准请写成 ('c-a-01',)（注意那个逗号），一个字符串会被按字符拆开。"
        )
    pool: set[str] = set()
    for item in relevant:
        if not isinstance(item, str) or not item.strip():
            raise RerankError(
                f"{label} 的 relevant 里出现了非字符串或空串：{item!r}。"
                "金标准里存的是记录 id，空 id 永远匹配不上任何命中，"
                "而它会让分母变大、指标变小——看起来像'检索变差了'。"
            )
        pool.add(item)
    return pool


def _check_ids(ids: Sequence[str], *, label: str) -> list[str]:
    """一份名单：序列、每一项是非空字符串（**保留重复**：重复位置按名次参与计算）."""
    if isinstance(ids, (str, bytes)) or not isinstance(ids, Sequence):
        raise RerankError(
            f"{label} 的 ids 必须是序列（列表/元组），收到 {type(ids).__name__}。"
            "出路：传 result.ids()（RetrievalResult 已经按名次排好了）或 "
            "[hit.record_id for hit in reranked.hits]。"
        )
    return [str(item) for item in ids]


def _check_positive_int(label: str, value: int) -> int:
    """``batch_size`` / ``ideal_len`` 共用：必须是 >= 1 的整数."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise RerankError(
            f"{label} 必须是 >= 1 的整数，收到 {value!r}。"
            "它们各自是一个下限（批大小 / 长度惩罚的标准长度），"
            "0 会让'每批 0 条'或'所有文本都打满折'这种错法静默发生。"
        )
    return value


def _weight_for(mode: str, weight: float | None) -> float:
    """这次实际生效的权重.

    ```text
    blend    weight 缺省 → DEFAULT_RERANK_WEIGHT（0.5，两边都不占先的起点）
    replace  weight 缺省 → 0.0（它不参与排序，记 0.0 的意思是"没有权重这件事"）
    replace  显式给了 weight → 原样记录，并由 notes 说明"replace 不看权重"
    ```

    "记录而不是报错"这个选择写在 ``RerankResult`` 的 docstring 里（与 ``fusion``
    里 RRF 收到 weights 直接报错的差别，那里是刻意的不同）。
    """
    if weight is not None:
        return weight
    if mode == RERANK_MODE_BLEND:
        return DEFAULT_RERANK_WEIGHT
    return 0.0


def _min_span(tokens: Sequence[str], needed: Sequence[str]) -> int | None:
    """最小跨度：**所有** ``needed`` 都出现时，最短的连续窗口宽度（找不到时 ``None``）.

    滑动窗口一次遍历（``O(len(tokens))``）：右端每前进一步把词收进窗口，
    窗口一旦"覆盖齐全"就一直收缩左端并记录最短宽度。两个细节：

    ```text
    返回宽度而不是位置    "跨度"是长度概念（1 表示紧挨着的一条），
                        而位置依赖文档怎么切，跨文档不可比
    找不到时给 None      不用 -1 或 0 当哨兵：0 是一个合法的"宽度"（虽然取不到），
                        而 -1 会一路漏进 1/(1+span) 变成 1/0
    ```
    """
    need = set(needed)
    if not need or not tokens:
        return None
    counts: dict[str, int] = {}
    missing = len(need)
    best: int | None = None
    left = 0
    for right, token in enumerate(tokens):
        if token in need:
            counts[token] = counts.get(token, 0) + 1
            if counts[token] == 1:
                missing -= 1
        while missing == 0:
            span = right - left + 1
            if best is None or span < best:
                best = span
            dropped = tokens[left]
            if dropped in need:
                counts[dropped] -= 1
                if counts[dropped] == 0:
                    missing += 1
            left += 1
    return best


def _elapsed_ms(started: float, elapsed: float | None) -> float:
    """这次重排的毫秒数：调用方量过就用它的，否则本函数自己量.

    "自己量"这件事对一个纯函数看着奇怪，但 ``RetrievalResult`` 里每个形状都
    带 ``latency_ms``（"这次贵不贵"是报告的一部分），而重排恰好是整条链路上
    最贵的一步——把它量出来比留一个 0.0 诚实。
    """
    if elapsed is not None:
        return round(float(elapsed), 3)
    return round((time.perf_counter() - started) * 1000, 3)


__all__ = [
    "DEFAULT_RERANK_BATCH",
    "DEFAULT_RERANK_MODEL",
    "DEFAULT_RERANK_TOP_N",
    "DEFAULT_RERANK_WEIGHT",
    "IDEAL_LEN",
    "RERANK_FEATURE_NAMES",
    "RERANK_MODES",
    "RERANK_MODE_BLEND",
    "RERANK_MODE_REPLACE",
    "RERANK_OVERRIDE_KEYS",
    "BaseReranker",
    "CrossEncoderReranker",
    "LiftProbe",
    "LiftReport",
    "RerankFeatures",
    "RerankHit",
    "RerankResult",
    "measure_lift",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "rerank_hits",
]
