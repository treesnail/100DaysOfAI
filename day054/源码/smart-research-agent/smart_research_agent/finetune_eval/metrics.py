"""领域文本指标：微调前后的输出该怎么比（M5-D5）.

day025~day030 的评估器回答的是"**任务**做对了吗"（用例通过 / 未通过），
这一课要回答的是另一个问题：**同一个问题的两个答案，哪个更好、好多少**。
微调评估绕不开这个比较：适配器只改了一百来 MB 的参数，输出往往"看起来
差不多"，只有把差别拆成可复现的数字，才能判断这次训练值不值。

本模块给出六个互补的分量，全部是**纯函数**——给同一对输入就必须给出同一个
输出，不依赖模型、不依赖随机数、不依赖环境：

======================  ==========================================================
``fact_recall``         参考答案里的事实点命中了几个（**语义**尺度）
``forbidden_ratio``     有没有踩到"禁项"（幻觉数字、被误用的结论）
``token_f1``            字符多重集的 F1（**词面**尺度，对改写敏感）
``rouge_l``             最长公共子序列的 F 度量（**顺序**敏感，对调序敏感）
``chrf``                字符 n-gram（n=1..6）的 F 度量（**局部片段**敏感）
``format``              必须片段 / 禁止片段 / 长度上限（**契约**尺度）
``refusal``             拒答行为是否与预期一致（**安全**尺度）
======================  ==========================================================

三条刻意的取舍，都是"看起来更简单、实际会出错"的地方：

1. **字符级而不是词级**。本课程语料以中文为主，中文没有天然空格分词，
   引一个分词器既增加依赖、又让指标随分词器版本漂移。字符级 F1 / chrF 在
   中文上是被广泛使用的做法；代价是英文单词被拆成字母，这一点在文档里
   写清楚，而不是假装没有。
2. **格式规则用原文匹配，不做规范化**。``结论：`` 是带全角冒号的四个字符，
   规范化会把标点抹掉，于是"必须写结论："这条规则对任何含"结论"的文本
   都成立——**规则形同虚设**。所以事实点走规范化（容忍标点与大小写差异），
   格式片段走原文（它本来就是字面契约）。
3. **六个分量不合成一个"总分"就交给调用方**。本模块只负责算分量与
   加权和，``pass`` 与否由 ``evaluator.score_item`` 定义（事实全覆盖 +
   无禁项 + 格式合规），因为"合格"是一个业务判断，不该藏在指标里。

数值稳定性：所有比值在分母为 0 时有**显式定义**（见各函数文档），
不返回 ``nan``——``nan`` 会让一切比较恒为 ``False``，于是"全是坏结果"
与"指标算错了"在报告里长得一模一样。
"""

from __future__ import annotations

import math
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

#: 规范化时剔除的 Unicode 大类：标点、分隔符、控制符、符号。
#: 不剔除字母 / 数字 / 组合记号（``M``）——后者在部分语言的规范化里承载信息。
_STRIPPED_CATEGORIES = frozenset({"P", "Z", "C", "S"})

#: 拒答检测的标记词。这份清单刻意保守：它只用来判断"模型是否拒绝了请求"，
#: 不用来判断"拒绝得好不好"（后者是 M3-D6 红队评估的题目）。
REFUSAL_MARKERS: tuple[str, ...] = (
    "无法",
    "不能",
    "不会",
    "抱歉",
    "拒绝",
    "不便",
    "不予",
    "不建议",
    "违反",
)

#: 六个分量的默认权重（和必须为 1.0，``validate_weights`` 会校验）.
#:
#: 为什么 ``fact_recall`` 占 0.40：微调要买到的正是"该说的说全、不该说
#: 的别说"，而词面指标（token_f1 / rouge_l / chrf 合计 0.50）对"把事实
#: 说清楚"这件事**给不出区分度**——把一句话换个说法，词面分掉一半，
#: 事实一个没少。格式与拒答各 0.05：它们是硬规则，权重低但不为零，
#: 因为"格式全错"这件事必须能从总分里看出来。
DEFAULT_WEIGHTS: dict[str, float] = {
    "fact_recall": 0.40,
    "token_f1": 0.20,
    "rouge_l": 0.15,
    "chrf": 0.15,
    "format": 0.05,
    "refusal": 0.05,
}

#: 分量名的固定顺序（报告与测试都依赖它，顺序变了报告就不可比对）.
COMPONENT_NAMES: tuple[str, ...] = tuple(DEFAULT_WEIGHTS)

#: 权重之和的允许误差（浮点加法不满足结合律，用精确相等会误伤）.
WEIGHT_TOLERANCE = 1e-9


class FinetuneEvalError(ValueError):
    """微调评估的非法输入（空规则、越界的 n、权重不合法等）.

    与 ``SFTLossError`` / ``PEFTConfigError`` 一样继承 ``ValueError``：
    调用方（路由层）可以一次捕获，把"字段非法"与"取值越界"统一映射成 400。
    """


# ---------------------------------------------------------------------------
# 规范化与分词
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """规范化文本：NFKC → 大小写折叠 → 剔除标点 / 空白 / 控制符 / 符号.

    这样做是为了让"事实点命中"这件事**容忍表达差异**：``ΔW = B·A`` 与
    ``ΔW=B*A`` 规范化后都是 ``δwb*a``（``·`` 与 ``*`` 都被剔除），于是
    "参考答案里的事实点在输出里出现了吗"不会被一个乘号绊住。

    代价必须说清楚：规范化之后 ``B`` 与 ``b`` 等价、标点全部消失。所以它是
    **子串命中**的前置处理，不是"语义相等"的判据——顺序差异它照样分不清
    （``normalize_text("ABA") != normalize_text("AAB")``，真正把两者混为一谈的
    是事实点判定所用的**子串包含**：``"ab"`` 同时是 ``aba`` 与 ``aab`` 的子串）。
    """
    folded = unicodedata.normalize("NFKC", text).casefold()
    return "".join(
        character
        for character in folded
        if unicodedata.category(character)[0] not in _STRIPPED_CATEGORIES
    )


def tokenize(text: str) -> list[str]:
    """字符级分词：规范化后的每个字符算一个 token.

    中文没有空格分词，而引一个分词器会让指标随分词器版本漂移；字符级
    在中文上是常规做法。英文单词因此被拆成字母——这对 F1 的影响是
    "把单词内部的拼写错误也算进去"，属于**更严格**的方向。
    """
    return list(normalize_text(text))


# ---------------------------------------------------------------------------
# 词面指标
# ---------------------------------------------------------------------------


def token_f1(prediction: str, reference: str) -> float:
    """字符多重集 F1：``2PR/(P+R)``，重叠按**多重集**计数（重复字符也算）.

    显式定义三种边界：

    - 两边都为空 → ``1.0``（空答案对比空答案是"完全一致"）；
    - 只有一边为空 → ``0.0``；
    - 重叠为 0 → ``0.0``（不做"都含标点所以相似"这类加分）。
    """
    predicted = Counter(tokenize(prediction))
    expected = Counter(tokenize(reference))
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    overlap = sum((predicted & expected).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(expected.values())
    return 2 * precision * recall / (precision + recall)


def lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    """最长公共子序列长度（滚动数组 DP，空间 ``O(len(right))``）.

    LCS 对**顺序**敏感：``A B C`` 与 ``C B A`` 的重叠 F1 是 1.0，而 LCS
    只有 1——这正是同时保留两个指标的理由，它们量的不是同一件事。
    """
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_item in left:
        current = [0] * (len(right) + 1)
        for index, right_item in enumerate(right, start=1):
            if left_item == right_item:
                current[index] = previous[index - 1] + 1
            else:
                current[index] = max(previous[index], current[index - 1])
        previous = current
    return previous[-1]


def rouge_l(prediction: str, reference: str, *, beta: float = 1.0) -> float:
    """ROUGE-L 的 F 度量：``(1+β²)PR / (R + β²P)``，``P`` / ``R`` 由 LCS 给出.

    ``beta`` 是刻意暴露的参数，因为**它有两种常见取法**：Lin 原文取
    ``β = P/R``（让 F 退化为一种调和形式），工程上更常见的是 ``β = 1``
    （即 F1）。两者在 ``P ≠ R`` 时给出**不同数值**，而"哪个是标准答案"
    取决于引用哪一份定义。本课缺省 ``β = 1`` 并写进签名——**把有争议的
    常数显式化，比让它藏在一个数字里好**。

    边界：两边都空 → ``1.0``；只有一边空 → ``0.0``；LCS 为 0 → ``0.0``。
    """
    if beta <= 0:
        raise FinetuneEvalError(f"beta 必须为正数，收到 {beta}")
    predicted = tokenize(prediction)
    expected = tokenize(reference)
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    common = lcs_length(predicted, expected)
    if common == 0:
        return 0.0
    precision = common / len(predicted)
    recall = common / len(expected)
    beta_squared = beta * beta
    return (1 + beta_squared) * precision * recall / (recall + beta_squared * precision)


def char_ngrams(characters: Sequence[str], order: int) -> Counter[tuple[str, ...]]:
    """字符 n-gram 多重集（``order`` 大于序列长度时返回空）."""
    if order <= 0:
        raise FinetuneEvalError(f"n-gram 的阶必须为正整数，收到 {order}")
    return Counter(
        tuple(characters[index : index + order])
        for index in range(len(characters) - order + 1)
    )


def chrf(prediction: str, reference: str, *, n: int = 6, beta: float = 2.0) -> float:
    """chrF：字符 1..n-gram 精确率的宏平均，再按 ``β`` 合成 F 度量.

    两个参数都按 chrF 的常见取值：``n = 6``、``β = 2``（**召回权重是精确率
    的 4 倍**）。``β > 1`` 的理由很实在：在生成任务里"把该说的说全"比
    "不要多说"更重要，而 chrF 的设计目标正是机器翻译的**充分性**。

    排序取用 sacrebleu 的形式 ``(1+β²)·chrP·chrR / (β²·chrP + chrR)``——
    这与 ROUGE-L 的写法**分母不同**（``chrR`` 与 ``chrP`` 的位置互换了），
    照抄任一个都会算错。本课按各自文献的实现写，并在测试里对
    ``β = 1`` 退化成调和平均这条不变式做断言。

    关于 ``β`` 的方向：``β → 0`` 收敛到精确率、``β → ∞`` 收敛到召回率。
    因此"β 越大越严格"是**错的**——当召回低于精确率时，增大 β 会让分数
    **下降**。

    边界：两边都空 → ``1.0``；只有一边空 → ``0.0``；宏平均任一项为 0
    （重叠为 0）→ ``0.0``；**某一阶 n-gram 在两侧都不存在时跳过该阶**
    （文本比 ``n`` 短时，这一阶量不到任何东西）。
    """
    if n < 1:
        raise FinetuneEvalError(f"chrf 的 n 至少为 1，收到 {n}")
    if beta <= 0:
        raise FinetuneEvalError(f"beta 必须为正数，收到 {beta}")
    predicted = tokenize(prediction)
    expected = tokenize(reference)
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    precisions: list[float] = []
    recalls: list[float] = []
    for order in range(1, n + 1):
        predicted_grams = char_ngrams(predicted, order)
        expected_grams = char_ngrams(expected, order)
        if not predicted_grams and not expected_grams:
            # 两侧都没有这一阶的 n-gram（文本比 n 还短）：**这一阶量不到任何
            # 东西**，跳过它。不跳过的后果是"完全相同的短文本拿不到 1.0"——
            # 本课实现时真的踩到过：默认 n=6 时 chrf("中文测试","中文测试")
            # 只有 0.6667，因为 5/6 两阶被记成了"零重叠"。
            continue
        overlap = (
            0
            if not predicted_grams or not expected_grams
            else sum((predicted_grams & expected_grams).values())
        )
        precisions.append(overlap / sum(predicted_grams.values()) if predicted_grams else 0.0)
        recalls.append(overlap / sum(expected_grams.values()) if expected_grams else 0.0)
    if not precisions:  # pragma: no cover - 两侧都非空时至少 1 阶可量
        raise FinetuneEvalError("chrF 没有任何可量的 n-gram 阶")
    mean_precision = sum(precisions) / len(precisions)
    mean_recall = sum(recalls) / len(recalls)
    if mean_precision == 0 or mean_recall == 0:
        return 0.0
    beta_squared = beta * beta
    return (1 + beta_squared) * mean_precision * mean_recall / (
        beta_squared * mean_precision + mean_recall
    )


# ---------------------------------------------------------------------------
# 事实点与禁项
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactRecall:
    """事实点召回的结果（**把分母交出来**，与 day050 的 loss 分母同一条纪律）."""

    hit: int
    total: int
    score: float
    missing: tuple[str, ...] = ()

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        state = "全部命中" if not self.missing else "缺失 " + "、".join(self.missing)
        return f"事实点 {self.hit}/{self.total}（{state}）"


def fact_recall(prediction: str, required_facts: Sequence[str]) -> FactRecall:
    """事实点召回：规范化后做子串命中，返回命中数与缺失清单.

    ``required_facts`` 为空时 ``score = 1.0``（"没有要求"等于"要求都满足"），
    这与 ``rag_metrics.retrieval_recall`` 在相关集合为空时返回 1.0 同构。

    空事实点直接报错，而不是当成"没有要求"：``""`` 是任何文本的子串，
    放进清单会让这条用例永远满分——**一个恒真的断言不是断言**。
    """
    checklist = list(required_facts)
    for fact in checklist:
        if not fact.strip():
            raise FinetuneEvalError("fact_recall 的事实点不能为空字符串")
    if not checklist:
        return FactRecall(hit=0, total=0, score=1.0, missing=())
    normalized_prediction = normalize_text(prediction)
    missing = tuple(
        fact for fact in checklist if normalize_text(fact) not in normalized_prediction
    )
    hit = len(checklist) - len(missing)
    return FactRecall(hit=hit, total=len(checklist), score=hit / len(checklist), missing=missing)


def forbidden_hits(prediction: str, forbidden_facts: Sequence[str]) -> tuple[str, ...]:
    """禁项命中：出现了哪些**不该出现**的片段（幻觉数字、被误用的结论）.

    与事实点用同一套规范化，因为两者都是"意思层面的存在性判断"。
    返回原文片段而不是规范化结果——报告是给人看的。
    """
    normalized_prediction = normalize_text(prediction)
    return tuple(
        fact for fact in forbidden_facts if normalize_text(fact) in normalized_prediction
    )


def forbidden_ratio(prediction: str, forbidden_facts: Sequence[str]) -> float:
    """禁项命中比例：``命中数 / 禁项总数``；没有禁项时定义为 ``0.0``（无瑕）.

    与 ``fact_recall`` 的"空即为 1.0"相反，这里空集合的语义是
    **"没有需要避免的东西"，所以瑕疵为 0**。
    """
    checklist = list(forbidden_facts)
    if not checklist:
        return 0.0
    return len(forbidden_hits(prediction, checklist)) / len(checklist)


# ---------------------------------------------------------------------------
# 格式契约
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FormatRules:
    """格式契约：必须出现的片段、禁止出现的片段、长度上限（字符数）."""

    must_contain: tuple[str, ...] = ()
    must_not_contain: tuple[str, ...] = ()
    max_chars: int | None = None

    def validate(self) -> None:
        """校验规则的**自洽性**（不检查内容是否合理，那是用例设计的事）."""
        if self.max_chars is not None and self.max_chars <= 0:
            raise FinetuneEvalError(f"max_chars 必须为正整数，收到 {self.max_chars}")
        if not self.must_contain and self.must_not_contain and self.max_chars is None:
            # 只有"禁止"没有"要求"+没有长度限制 = 这条规则几乎必然通过，
            # 属于设计缺陷而不是运行时错误，本函数仍然拒绝它：
            # **评估集里不许存在注定满分的用例**（day025 的纪律沿用）。
            raise FinetuneEvalError(
                "格式规则至少要有 must_contain 或 max_chars 之一："
                "只写 must_not_contain 的规则对绝大多数输出都成立"
            )

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        parts: list[str] = []
        if self.must_contain:
            parts.append("必须含 " + "、".join(self.must_contain))
        if self.must_not_contain:
            parts.append("禁止含 " + "、".join(self.must_not_contain))
        if self.max_chars is not None:
            parts.append(f"不超 {self.max_chars} 字")
        return "；".join(parts) if parts else "无格式约束"


def format_violations(text: str, rules: FormatRules) -> tuple[str, ...]:
    """逐条检查格式契约，返回违规说明（**空元组 = 合规**）.

    必须片段与禁止片段都用**原文**匹配（不做 ``normalize_text``）：
    ``结论：`` 里的全角冒号是契约的一部分，规范化会把标点抹掉，于是
    禁止片段 ``不好`` 也会命中 ``不 好``——**规则一旦被规范化软化，
    它就再也拦不住任何东西**。
    """
    rules.validate()
    violations: list[str] = []
    for fragment in rules.must_contain:
        if fragment not in text:
            violations.append(f"缺少必需片段：{fragment!r}")
    for fragment in rules.must_not_contain:
        if fragment in text:
            violations.append(f"出现禁止片段：{fragment!r}")
    if rules.max_chars is not None and len(text) > rules.max_chars:
        violations.append(f"长度 {len(text)} 超出上限 {rules.max_chars}")
    return tuple(violations)


def is_refusal(text: str) -> bool:
    """是否是一次拒答（按标记词判断，保守清单 ``REFUSAL_MARKERS``）.

    它不是"安全评估"：一个模型可能拒答得当、也可能拒答得毫无帮助，
    本函数只回答"拒了吗"。**过度拒答（该答的也拒）与拒答缺失是两种
    不同的失败**，两者都要被看见，所以这里绝不返回"安全/不安全"。

    **已知的假阳性**：清单里的"不能 / 不会 / 不建议"在正常解释里也很常见
    （"这个改动不会影响性能"）。本课刻意不做上下文判别——判别需要语义
    模型，那是 M3-D6 红队评估的题目；这里保留"保守清单 + 原文子串匹配"
    的口径，并把假阳性代价写清楚：真命中会让 ``refusal`` 分量打成 0。
    因此**在"该不该拒答"会决定成败的用例上，宁可写得长一点、把话说全**。
    """
    return any(marker in text for marker in REFUSAL_MARKERS)


# ---------------------------------------------------------------------------
# 权重与加权和
# ---------------------------------------------------------------------------


def validate_weights(weights: dict[str, float]) -> None:
    """校验权重字典：键集合与 ``COMPONENT_NAMES`` 一致，且和等于 1.0.

    两项都校验的理由：少一个键会让某个分量**静默不参与**打分；和不为 1
    会让总分不再落在 ``[0, 1]``，于是"通过阈值"这个约定失效。
    """
    unknown = sorted(set(weights) - set(COMPONENT_NAMES))
    if unknown:
        raise FinetuneEvalError(f"未知的分量：{', '.join(unknown)}")
    missing = sorted(set(COMPONENT_NAMES) - set(weights))
    if missing:
        raise FinetuneEvalError(f"缺少分量权重：{', '.join(missing)}")
    for name, value in weights.items():
        if not math.isfinite(value) or value < 0:
            raise FinetuneEvalError(f"分量 {name} 的权重必须是非负有限数，收到 {value}")
    total = sum(weights.values())
    if abs(total - 1.0) > WEIGHT_TOLERANCE:
        raise FinetuneEvalError(f"权重之和必须为 1.0，当前为 {total!r}")


def weighted_total(components: dict[str, float], weights: dict[str, float] | None = None) -> float:
    """分量加权和（缺省用 ``DEFAULT_WEIGHTS``）.

    每个分量都必须落在 ``[0, 1]``：越界说明某个分量的实现坏了，此时
    让它静默算出总分比直接报错更糟——**总分越界会被当成"模型很好"
    或"模型很差"，而真正的问题是指标本身**。
    """
    active = DEFAULT_WEIGHTS if weights is None else weights
    validate_weights(active)
    unknown = sorted(set(components) - set(COMPONENT_NAMES))
    if unknown:
        raise FinetuneEvalError(f"未知的分量：{', '.join(unknown)}")
    missing = sorted(set(COMPONENT_NAMES) - set(components))
    if missing:
        raise FinetuneEvalError(f"缺少分量：{', '.join(missing)}")
    for name in COMPONENT_NAMES:
        value = components[name]
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise FinetuneEvalError(f"分量 {name} 必须落在 [0, 1]，收到 {value}")
    return sum(components[name] * active[name] for name in COMPONENT_NAMES)


def mean(values: Iterable[float]) -> float:
    """安全求均值：空序列返回 ``0.0``（与 ``evaluation.metrics.mean`` 同约定）."""
    collected = list(values)
    if not collected:
        return 0.0
    return sum(collected) / len(collected)


@dataclass(frozen=True)
class MetricBreakdown:
    """一条输出上的六个分量与加权和（**分量与总分一起留档**）."""

    components: dict[str, float] = field(default_factory=dict)
    total: float = 0.0

    def __post_init__(self) -> None:
        """把缺的分量补成 0.0，让"六个分量永远都在"成为结构不变式.

        为什么要补：``evaluator`` 的异常路径会构造一个"全 0"的分解，
        调用方也可能只想给几个分量。若允许"缺键"存在，``to_dict()`` /
        ``summary_line()`` 就会在**打印一份本来就坏掉的结果**时再抛一次
        ``KeyError``——**报告工具在遇到坏数据时最不该做的事就是崩掉**。

        注意 ``total`` 不被重算：它是"加权和"这个判断的结果，由
        ``weighted_total`` 算出并显式传进来；在这里静默重算会让"传进来的
        总分"与"实际打印的总分"不一致，那比缺键更难查。
        """
        merged = {name: self.components.get(name, 0.0) for name in COMPONENT_NAMES}
        if merged != self.components:
            object.__setattr__(self, "components", merged)

    def to_dict(self) -> dict[str, Any]:
        """投影为可 json.dumps 的字典（分量按固定顺序排序）."""
        return {
            "components": {name: self.components[name] for name in COMPONENT_NAMES},
            "total": self.total,
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        parts = " ".join(f"{name}={self.components[name]:.3f}" for name in COMPONENT_NAMES)
        return f"总分 {self.total:.4f} | {parts}"


def compute_components(
    prediction: str,
    reference: str,
    *,
    required_facts: Sequence[str] = (),
    forbidden_facts: Sequence[str] = (),
    rules: FormatRules | None = None,
    expect_refusal: bool = False,
) -> MetricBreakdown:
    """把一次输出拆成六个分量与加权和（**纯计算，不做合格判定**）.

    ``format`` 分量是"违规条数"的衰减而不是 0/1 二值：一条违规扣一半
    （``1/(1+违规数)``），因为"少写一个必须片段"与"长度超一倍"不该
    得到同一个分数。而 ``refusal`` 分量是二值的：拒答行为只有"对"和
    "错"两种——**"拒了一半"不是一个状态**。
    """
    active_rules = rules if rules is not None else FormatRules()
    recall = fact_recall(prediction, required_facts)
    violations = format_violations(prediction, active_rules)
    refusal_correct = is_refusal(prediction) == expect_refusal
    components = {
        "fact_recall": recall.score,
        "token_f1": token_f1(prediction, reference),
        "rouge_l": rouge_l(prediction, reference),
        "chrf": chrf(prediction, reference),
        "format": 1.0 / (1 + len(violations)),
        "refusal": 1.0 if refusal_correct else 0.0,
    }
    return MetricBreakdown(components=components, total=weighted_total(components))


__all__ = [
    "COMPONENT_NAMES",
    "DEFAULT_WEIGHTS",
    "REFUSAL_MARKERS",
    "WEIGHT_TOLERANCE",
    "FactRecall",
    "FinetuneEvalError",
    "FormatRules",
    "MetricBreakdown",
    "char_ngrams",
    "chrf",
    "compute_components",
    "fact_recall",
    "forbidden_hits",
    "forbidden_ratio",
    "format_violations",
    "is_refusal",
    "lcs_length",
    "mean",
    "normalize_text",
    "rouge_l",
    "token_f1",
    "tokenize",
    "validate_weights",
    "weighted_total",
]
