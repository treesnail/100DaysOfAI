"""多维质量打分：把"通过 / 拒绝"升级成"好多少"（M5-D8）.

day048 的质量过滤是**布尔规则**：七条规则，任一条不通过即拒，原因逐条
记账。它解决的是"这条样本能不能进训练集"，却回答不了另一个问题——
**两条都能进，哪条更该留？** 配比削减（``mixing``）与增强补样都要按
"好多少"排序，布尔规则给不出顺序。

本模块把质量拆成**五个可计算的维度**，加权成 0~1 的连续分：

| 维度 | 计算 | 挡住什么故障 |
|------|------|-------------|
| ``length`` | 输出字符数的梯形分（``low → ideal → high``） | 太短没信息 / 太长是粘贴 |
| ``repetition`` | 3-gram 重复率的反向线性分 | 车轱辘话（模型退化的典型输出） |
| ``structure`` | 三档：有结构化收尾 1.0 / 有句末标点 0.75 / 都没有 0.25 | 半句话、被截断的输出 |
| ``traceability`` | ``source`` 与 ``license`` 各 0.5 | 来源不可追溯、许可证不明 |
| ``placeholder`` | 命中 TODO/待补充 记 0.0，否则 1.0 | 答案没写完的占位样本 |

**打分不是硬规则的替代品**。流水线里的顺序是"先硬规则、后软打分"
（``DomainDataPipeline.run``）：占位符样本必须被清洗器直接拒掉，
而不是"扣 0.15 分之后仍然侥幸过线"。用打分器冒充门禁，是这类系统里
最常见的一种自欺——分数看起来在起作用，实际上放过了一条永远不会
被真正学会的样本。

**覆盖不了的维度就不要假装能打分**（day055 的同一条纪律）：
``format``（格式合规）、``refusal``（拒答边界）、``factuality``（事实性）
都需要结构化或人工判定，本课**不给它们分数**。用一个字符级统计量
冒充"事实正确"，比承认测不了更糟。

常量标定依据（本课程数据集，37 条清洗后样本实测）：
输出长度 67~165 字（中位数 78）、3-gram 重复率中位数 0.0 / 均值 0.045 /
最大 0.377、有许可证元信息 16/37、结构化收尾 5/37。**这些数字决定了
下面的阈值怎么取，而不是反过来。**
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from smart_research_agent.domain_data.errors import DomainDataError
from smart_research_agent.domain_data.fingerprint import normalize_for_fingerprint
from smart_research_agent.evaluation.perf_baseline import percentile
from smart_research_agent.finetune.cleaner import PLACEHOLDER_MARKERS
from smart_research_agent.finetune.schema import TrainingExample

#: 五个维度的固定顺序。它同时决定：**并列最低分时取哪一个当拒绝原因**、
#: 维度对照表的行顺序、以及 ``QualityWeights.as_dict()`` 的键顺序。
#: 三处同源，因此"文档、报表、代码"不会各说一套。
DIMENSIONS: tuple[str, ...] = (
    "length",
    "repetition",
    "structure",
    "traceability",
    "placeholder",
)

#: 输出长度的梯形分四段点。低端对齐 day048 的 ``finetune_min_output_chars``
#: （8 字符以下信息量太低），高端对齐 ``finetune_max_output_chars`` 的
#: 十分之一档（4000 字以上基本是把长文粘进来了）。中间 40~800 字是
#: **实测的舒适区**：本课程样本落在 67~165 字，全部拿满分。
LENGTH_LOW = 8
LENGTH_IDEAL_LOW = 40
LENGTH_IDEAL_HIGH = 800
LENGTH_HIGH = 4000

#: 3-gram 重复率的反向区间：低于 ``BEST`` 拿满分，高于 ``WORST`` 得 0 分。
#: ``BEST`` 取 0.10 而不是 0.0，是因为正常中文文本也会有一定量的 3-gram
#: 复用（"检索增强生成"这类术语会重复出现）；把 0 当满分点会让所有样本
#: 都被扣分，分数分布被压扁，反而失去区分度。
REPETITION_BEST = 0.10
REPETITION_WORST = 0.35

#: 3-gram 窗口：比指纹用的 ``DEFAULT_SHINGLE_K`` 一致（同为 3），
#: 但**用途不同**——指纹要的是"两条文本像不像"，重复率要的是
#: "这段话自己重复了几遍"。同一个 k 让两个数字可以互相印证。
REPETITION_NGRAM = 3

#: ``structure`` 的三档取值（常量而不是字面量，报表里直接引用）
STRUCTURE_WITH_MARKER = 1.0
STRUCTURE_WITH_TERMINAL = 0.75
STRUCTURE_OPEN_ENDED = 0.25

#: 结构化收尾标记。ReAct 轨迹以 ``Final Answer:`` 收尾，说明它是一条
#: **完整**的推理轨迹；缺了它而只剩半截 ``Thought:``，是最危险的训练数据
#: ——模型学到的会是"想到一半就停"。
STRUCTURE_MARKERS: tuple[str, ...] = ("Final Answer:", "结论：", "总结：", "答案：")

#: 句末标点集合（中英文并列）。列成常量而不是内联字符串，
#: 是为了让测试能直接引用同一份定义去构造用例。
TERMINAL_PUNCTUATION = "。！？.!?；;）)\"'”’"

#: 权重之和的容差：浮点加法不满足结合律，四个 0.15 相加未必恰好是 0.6。
WEIGHT_SUM_TOLERANCE = 1e-6

#: 质量门槛。**它到底在拦什么，必须算清楚，否则这个数字就是玄学**：
#:
#: 门槛 0.6 精确地等价于"**加权质量损失超过 0.4 才拒**"（总分满分 1.0）。
#: 五个维度里权重最大的是 0.25，也就是说——
#:
#: > **任何一个维度单独归零，总分都还在 0.75 以上，不会被拒。**
#:
#: 这不是漏洞，而是与 day048 的硬规则刻意分工：**单点致命问题**
#: （占位符答案、空指令、超长粘贴）由那七条布尔规则一票否决；
#: 质量分负责的是"都合格但有优劣"的排序，因此必须对单点不敏感。
#: 实测把两类真实脏样本喂进来，被拒的正是**多维度同时不合格**的那些：
#:
#: ==================================================  ==========  ==================
#: 样本                                                总分        结论（门槛 0.6）
#: ==================================================  ==========  ==================
#: 正常样本（67~165 字、有来源）                        0.90~0.95   通过
#: ``"TODO：待补充。"``（占位 + 6 字太短）              0.4828      拒（最低维 placeholder）
#: ``"好的好的好的好的好的"``（车轱辘话 + 无收尾）      0.2906      拒（最低维 repetition）
#: ==================================================  ==========  ==================
#:
#: 取 0.6 而不是 0.5999999 这类"刚好卡住某条样本"的值，
#: 是为了让门槛能被复述、能在评审会上被算一遍。
DEFAULT_QUALITY_THRESHOLD = 0.6


def band_score(
    value: float, *, low: float, ideal_low: float, ideal_high: float, high: float
) -> float:
    """梯形打分：``low`` 以下 0 分，``[ideal_low, ideal_high]`` 满分，``high`` 以上 0 分.

    未使用平滑曲线（sigmoid 之类）是刻意的：梯形分的每一段都能手算复核，
    报告里的"这条样本长度分 0.375"可以被评审一眼还原成"输出 25 字，
    落在 8~40 的上升段"。曲线给不出这种可核对性。

    边界行为：``value <= low`` 与 ``value >= high`` 都返回 0.0，
    ``[ideal_low, ideal_high]`` 闭区间返回 1.0。四段点必须满足
    ``low <= ideal_low <= ideal_high <= high``，否则抛 ``DomainDataError``
    ——退化区间（例如 ``ideal_low > high``）会让函数静默返回错误的分数。
    """
    if not low <= ideal_low <= ideal_high <= high:
        raise DomainDataError(
            f"梯形分的四段点必须满足 low <= ideal_low <= ideal_high <= high，"
            f"收到 ({low}, {ideal_low}, {ideal_high}, {high})"
        )
    if value <= low or value >= high:
        return 0.0
    if value < ideal_low:
        return (value - low) / (ideal_low - low)
    if value <= ideal_high:
        return 1.0
    return (high - value) / (high - ideal_high)


def descending_score(value: float, *, best: float, worst: float) -> float:
    """反向线性打分：``value <= best`` 满分，``value >= worst`` 0 分，中间线性.

    这是梯形分在"单侧最优"场景下的特例（最优平台收缩成一个点），
    单独成函数是因为重复率、噪声比例这类"越小越好"的指标全都用它。
    """
    if best > worst:
        raise DomainDataError(f"best 必须不大于 worst，收到 best={best}, worst={worst}")
    if value <= best:
        return 1.0
    if value >= worst:
        return 0.0
    if worst == best:
        return 0.0
    return (worst - value) / (worst - best)


def ngram_repetition(text: str, *, n: int = REPETITION_NGRAM) -> float:
    """n-gram 重复率 = ``1 - 唯一 n-gram 数 / n-gram 总数``.

    在**归一化后**的文本上统计（与指纹共用 ``normalize_for_fingerprint``），
    否则全半角与大小写的差异会让"看起来一模一样的两段话"算出不同的重复率。

    文本短于 ``n`` 时返回 0.0：连一个 n-gram 都凑不出来，
    谈不上重复——把它记成 1.0（"完全重复"）会让所有极短样本被判死。
    """
    if n < 1:
        raise DomainDataError(f"n-gram 窗口必须 >= 1，收到 {n}")
    normalized = normalize_for_fingerprint(text)
    if len(normalized) < n:
        return 0.0
    grams = [normalized[index : index + n] for index in range(len(normalized) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def has_terminal_punctuation(text: str) -> bool:
    """输出是否以句末标点收尾（去掉尾部空白后判最后一个字符）."""
    stripped = text.strip()
    return bool(stripped) and stripped[-1] in TERMINAL_PUNCTUATION


def has_structure_marker(text: str) -> bool:
    """输出是否含结构化收尾标记（``Final Answer:`` / ``结论：`` 之类）."""
    return any(marker in text for marker in STRUCTURE_MARKERS)


def structure_score(text: str) -> float:
    """结构完整性三档分：结构化收尾 1.0 / 有句末标点 0.75 / 都没有 0.25.

    **为什么最低档是 0.25 而不是 0.0**：一个既没有句末标点、也没有结构化
    收尾的输出，读起来像"被截断了"，但它未必真的错——短答句（"是。"）
    是合法的。给 0.25 表示"可疑但不清零"，把它交给总分与其他维度一起裁决。
    若直接判 0，一条长度、相关性都完美的短答会被这一维一票否决，
    而真正的截断样本本来就还有 ``length`` 维在下压。
    """
    if has_structure_marker(text):
        return STRUCTURE_WITH_MARKER
    if has_terminal_punctuation(text):
        return STRUCTURE_WITH_TERMINAL
    return STRUCTURE_OPEN_ENDED


@dataclass(frozen=True)
class QualitySignals:
    """从一条样本抽出的**原始信号**（未加权、未归一）.

    把"抽信号"与"算分数"分开，好处是评审能看到原始量：报告里写
    "总分 0.62"没人能反驳，写"总分 0.62，其中 3-gram 重复率 0.31、
    无许可证、无结构化收尾"就能立刻定位问题。信号是事实，分数是判断。
    """

    output_chars: int
    instruction_chars: int
    repetition: float
    structure: float
    has_source: bool
    has_license: bool
    has_placeholder: bool

    def to_dict(self) -> dict:
        """投影为可直接 json.dumps 的字典."""
        return {
            "output_chars": self.output_chars,
            "instruction_chars": self.instruction_chars,
            "repetition": round(self.repetition, 4),
            "structure": self.structure,
            "has_source": self.has_source,
            "has_license": self.has_license,
            "has_placeholder": self.has_placeholder,
        }


def extract_signals(example: TrainingExample) -> QualitySignals:
    """抽取一条样本的质量信号（纯函数，不读磁盘、不调模型）."""
    lowered = example.output.lower()
    return QualitySignals(
        output_chars=len(example.output),
        instruction_chars=len(example.instruction),
        repetition=ngram_repetition(example.output),
        structure=structure_score(example.output),
        has_source=bool(example.source.strip()),
        has_license=bool(example.license.strip()),
        has_placeholder=any(marker in lowered for marker in PLACEHOLDER_MARKERS),
    )


@dataclass(frozen=True)
class QualityWeights:
    """五个维度的权重（构造期校验：非负且和为 1.0）.

    权重之和必须恰好是 1.0，这条约束不是形式主义：它保证不同批次的
    ``total`` 落在同一个尺度上，"这批平均 0.82、上批平均 0.79"才是
    可比较的两个数字。允许不加校验，就会出现"某天权重写成 1.2，
    全批分数集体超过 1"这种没人会立刻发现的错误。
    """

    length: float = 0.25
    repetition: float = 0.25
    structure: float = 0.20
    traceability: float = 0.15
    placeholder: float = 0.15

    def __post_init__(self) -> None:
        for name in DIMENSIONS:
            value = getattr(self, name)
            if value < 0.0:
                raise DomainDataError(f"维度 {name} 的权重不能为负，收到 {value}")
        total = self.total()
        if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise DomainDataError(
                f"五个维度权重之和必须为 1.0（容差 {WEIGHT_SUM_TOLERANCE}），收到 {total!r}"
            )

    def total(self) -> float:
        """权重之和（构造期用它做校验，报告里也可以直接展示）."""
        return sum(getattr(self, name) for name in DIMENSIONS)

    def as_dict(self) -> dict[str, float]:
        """按 ``DIMENSIONS`` 顺序投影为字典."""
        return {name: getattr(self, name) for name in DIMENSIONS}


def default_weights() -> QualityWeights:
    """默认权重：长度与重复率各占四分之一，另三维合计一半.

    20 组权重组合在真实数据集上试过，结论写在这里而不是留给学生去搜：
    **把 ``traceability`` 提到 0.3 以上会把 21 条"有来源但无许可证"的
    样本整体压到 0.75 附近**，于是配比削减会优先砍掉红队安全样本
    （它们的许可证恰好是空的）——一个纯治理指标不该有这种副作用。
    """
    return QualityWeights()


@dataclass(frozen=True)
class QualityScore:
    """一条样本的质量分：分维度明细 + 加权总分 + 判定.

    ``threshold`` 随分数一起存下来，是为了让分数**自带判定依据**。
    只存 ``total`` 的报告在门槛改了之后就失去意义：读到 0.62 的人
    无法判断它当时是"过"还是"没过"。
    """

    dimensions: dict[str, float]
    total: float
    threshold: float

    @property
    def passed(self) -> bool:
        """是否达到门槛（``>=`` 而不是 ``>``：门槛值本身算通过）."""
        return self.total >= self.threshold

    def weakest(self) -> str:
        """得分最低的维度名（并列时按 ``DIMENSIONS`` 顺序取第一个）.

        并列取第一个而不是"随便一个"，是为了让拒绝原因的统计可复现：
        同一份数据两次跑出来的 ``rejection_by_dimension`` 必须逐键相同。
        """
        return min(
            DIMENSIONS,
            key=lambda name: (self.dimensions.get(name, 0.0), DIMENSIONS.index(name)),
        )

    def to_dict(self) -> dict:
        """投影为可直接 json.dumps 的字典."""
        return {
            "dimensions": {name: round(value, 4) for name, value in self.dimensions.items()},
            "total": round(self.total, 4),
            "threshold": self.threshold,
            "passed": self.passed,
            "weakest": self.weakest(),
        }


def score_example(
    example: TrainingExample,
    *,
    weights: QualityWeights | None = None,
    threshold: float = DEFAULT_QUALITY_THRESHOLD,
) -> QualityScore:
    """给一条样本打分：五个维度加权求和.

    每个维度都先被换算到 ``[0, 1]``，因此加权和天然落在 ``[0, 1]``
    ——不需要再做一次归一化，也不会出现"总分超过 1"这种让门槛失效的值。
    """
    resolved = weights if weights is not None else default_weights()
    signals = extract_signals(example)
    dimensions = {
        "length": band_score(
            signals.output_chars,
            low=LENGTH_LOW,
            ideal_low=LENGTH_IDEAL_LOW,
            ideal_high=LENGTH_IDEAL_HIGH,
            high=LENGTH_HIGH,
        ),
        "repetition": descending_score(
            signals.repetition, best=REPETITION_BEST, worst=REPETITION_WORST
        ),
        "structure": signals.structure,
        "traceability": 0.5 * signals.has_source + 0.5 * signals.has_license,
        "placeholder": 0.0 if signals.has_placeholder else 1.0,
    }
    total = sum(dimensions[name] * getattr(resolved, name) for name in DIMENSIONS)
    return QualityScore(dimensions=dimensions, total=total, threshold=threshold)


#: 维度对照表的元信息：人读的部分（含义 / 计算 / 挡什么）与算的部分（权重）
#: 分开存，是为了让 ``dimension_table()`` 能**从代码生成文档**
#: （day048 ``render_methods_table`` 的同一条纪律：代码改了文档没改立刻可见）。
_DIMENSION_META: tuple[tuple[str, str, str, str], ...] = (
    (
        "length",
        "输出长度落在舒适区",
        f"梯形分：<{LENGTH_LOW} / {LENGTH_LOW}~{LENGTH_IDEAL_LOW} 上升 / "
        f"{LENGTH_IDEAL_LOW}~{LENGTH_IDEAL_HIGH} 满分 / {LENGTH_IDEAL_HIGH}~{LENGTH_HIGH} 下降",
        "太短没信息量、太长是粘贴长文",
    ),
    (
        "repetition",
        "自我重复程度",
        f"3-gram 重复率的反向线性分：<={REPETITION_BEST} 满分、>={REPETITION_WORST} 零分",
        "车轱辘话（模型退化的典型输出）",
    ),
    (
        "structure",
        "结构完整性",
        f"三档：结构化收尾 {STRUCTURE_WITH_MARKER} / 有句末标点 "
        f"{STRUCTURE_WITH_TERMINAL} / 都没有 {STRUCTURE_OPEN_ENDED}",
        "半句话、被截断的输出",
    ),
    (
        "traceability",
        "来源可追溯性",
        "source 与 license 各计 0.5",
        "来源不明、许可证不明的语料",
    ),
    (
        "placeholder",
        "无占位符",
        "命中 TODO/待补充/lorem 等标记记 0.0，否则 1.0",
        "答案没写完的占位样本",
    ),
)


def dimension_table(weights: QualityWeights | None = None) -> list[dict]:
    """维度对照表：**权重列由 ``QualityWeights`` 现场读出**，不是抄写.

    表格里出现的每一个数字（权重、四段点、三档取值）都来自模块常量；
    测试会断言"表里的权重 == ``default_weights().as_dict()``"，
    因此"文档说 0.25、代码是 0.2"这类漂移会立刻变红。
    """
    resolved = weights if weights is not None else default_weights()
    weight_map = resolved.as_dict()
    return [
        {
            "dimension": name,
            "meaning": meaning,
            "computation": computation,
            "guards_against": guards,
            "weight": weight_map[name],
        }
        for name, meaning, computation, guards in _DIMENSION_META
    ]


def render_dimension_table(weights: QualityWeights | None = None) -> str:
    """把维度对照表渲染成 markdown（文档由代码生成，见 ``dimension_table``）."""
    header = (
        "| 维度 | 含义 | 计算 | 挡住的故障 | 缺省权重 |\n"
        "|------|------|------|-----------|---------|\n"
    )
    rows = "".join(
        f"| `{row['dimension']}` | {row['meaning']} | {row['computation']} | "
        f"{row['guards_against']} | {row['weight']:.2f} |\n"
        for row in dimension_table(weights)
    )
    return header + rows


@dataclass
class QualityFilterResult:
    """质量过滤结果：保留样本 + 全量分数 + 门槛 + 权重.

    ``scores`` 与入参 ``examples`` **等长同序**（包括被拒的那些），
    因此"第 i 条为什么被拒"永远可以当场回答。只留被保留样本的分数
    会让被拒样本的原因消失在报告之外——而"被丢掉了多少、为什么"正是
    数据治理最需要留档的那一栏。
    """

    examples: list[TrainingExample]
    scores: list[QualityScore]
    weights: QualityWeights
    threshold: float

    @property
    def total(self) -> int:
        """进过滤器的样本数."""
        return len(self.examples)

    @property
    def kept_indices(self) -> list[int]:
        """被保留样本在入参里的下标（升序）."""
        return [index for index, score in enumerate(self.scores) if score.passed]

    @property
    def rejected_indices(self) -> list[int]:
        """被拒样本在入参里的下标（升序）."""
        return [index for index, score in enumerate(self.scores) if not score.passed]

    @property
    def kept(self) -> list[TrainingExample]:
        """被保留的样本（保持入参相对顺序）."""
        return [self.examples[index] for index in self.kept_indices]

    @property
    def rejected(self) -> int:
        """被拒条数."""
        return self.total - len(self.kept_indices)

    @property
    def keep_rate(self) -> float:
        """保留率（空输入按 0.0 计，与 day048 的 ``FilterReport`` 同一约定）."""
        return len(self.kept_indices) / self.total if self.total else 0.0

    def rejection_by_dimension(self) -> dict[str, int]:
        """被拒样本按**最低分维度**归因的计数（**每一次丢弃都要有名字**）.

        归因到"最低分维度"而不是"第一个不达标的维度"：分数是连续量，
        没有"不达标"这条线，只有"哪一维拖了后腿"。这条口径与 day055 的
        ``drop_reasons`` 是同一种记账思路，只是判据从布尔换成了排序。
        """
        counts: dict[str, int] = {}
        for index in self.rejected_indices:
            name = self.scores[index].weakest()
            counts[name] = counts.get(name, 0) + 1
        return counts

    def distribution(self) -> dict[str, float]:
        """总分分布：最值 + 最近秩法分位数 + 均值.

        分位数复用 day046 的 ``percentile``（**最近秩法、不插值**）：
        37 条样本的 P75 必须落在某一条真实样本上，"0.8271"这种插值出来的
        数字无法被回溯核对。
        """
        values = [score.total for score in self.scores]
        if not values:
            return {"min": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "max": 0.0, "mean": 0.0}
        return {
            "min": round(min(values), 4),
            "p25": round(percentile(values, 25), 4),
            "p50": round(percentile(values, 50), 4),
            "p75": round(percentile(values, 75), 4),
            "max": round(max(values), 4),
            "mean": round(sum(values) / len(values), 4),
        }

    def to_dict(self) -> dict:
        """投影为可直接 json.dumps 的字典."""
        return {
            "total": self.total,
            "kept": len(self.kept_indices),
            "rejected": self.rejected,
            "keep_rate": round(self.keep_rate, 4),
            "threshold": self.threshold,
            "weights": self.weights.as_dict(),
            "distribution": self.distribution(),
            "rejection_by_dimension": self.rejection_by_dimension(),
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要（demo 与日志直接打印）."""
        return (
            f"质量过滤：{self.total} 条进 / {len(self.kept_indices)} 条留 | "
            f"门槛 {self.threshold} | 拒绝归因 {self.rejection_by_dimension() or '无'}"
        )


def validate_threshold(threshold: float) -> float:
    """校验质量门槛落在 ``[0, 1]``（本包"校验必须在构造期/进流水线前"的纪律）.

    这条校验看起来多余——门槛超过 1 会怎样？答案是：**整批样本被静默拒光**，
    流水线返回一个空数据集，而报告里只有一行 "kept: 0"。没有异常、没有告警，
    排查的人会先去怀疑数据而不是怀疑那个写错的门槛。
    一个"看起来在跑、其实一行数据都没留下"的配置，正是最该被提前拦下的一类。
    """
    if not 0.0 <= threshold <= 1.0:
        raise DomainDataError(f"质量门槛必须落在 [0, 1] 区间，收到 {threshold}")
    return threshold


def filter_by_quality(
    examples: Sequence[TrainingExample],
    *,
    weights: QualityWeights | None = None,
    threshold: float = DEFAULT_QUALITY_THRESHOLD,
) -> QualityFilterResult:
    """按质量分过滤一批样本，返回完整结果（含被拒样本的分数）."""
    resolved = weights if weights is not None else default_weights()
    validate_threshold(threshold)
    scores = [
        score_example(example, weights=resolved, threshold=threshold) for example in examples
    ]
    return QualityFilterResult(
        examples=list(examples),
        scores=scores,
        weights=resolved,
        threshold=threshold,
    )


def quality_scores(
    examples: Sequence[TrainingExample], *, weights: QualityWeights | None = None
) -> list[float]:
    """只取总分（配比削减时的排序键）.

    单独开这个薄接口，是为了让调用处一眼看出"这里只需要顺序"：
    配比削减要的是"谁的分数高"，不需要五个维度的明细。
    """
    resolved = weights if weights is not None else default_weights()
    return [score_example(example, weights=resolved).total for example in examples]
