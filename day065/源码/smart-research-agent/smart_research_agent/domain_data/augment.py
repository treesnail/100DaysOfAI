"""数据增强：在**不改动答案**的前提下扩充指令样本（M5-D8）.

day049 已经把账算清：37 条样本训不出有用的模型，**500 条以下先补数据**
（day048 的那条无条件告警）。补数据有两条路——去外面采，或者把手上这批
"用不同的方式再问一遍"。本模块做后者，并给它装上三条纪律。

## 纪律一：只改 prompt 侧，``output`` 逐字不动

增强样本的答案是**从原样本继承**的。只要动了 output，增强就从"换一种问法"
变成了"生成新答案"，而生成就意味着可能出错——一条错了的增强样本会在
训练里被当成金标准反复强化。因此 ``augment_example`` 的返回值里
``output`` 恒等于入参的 ``output``，这条性质在测试里对**每一个算子**
逐条断言，不靠注释保证。

## 纪律二：能改写答案正确性的增强，必须自证"答案本来就满足"

``constraint`` 算子往指令后面追加输出约束（"请用中文回答。"）。如果原答案
是 300 字而约束写"200 字以内"，这条增强样本就是**自相矛盾**的：训练的
结果要么是模型学会忽略指令，要么是模型学会把答案砍短（而参考答案并没有
砍短）。两种都是污染。

所以每个约束都带一个 ``satisfied_by`` 判定，**只有原答案本来就满足这条约束
时才追加**。不满足就跳过并记进 ``guard_rejected``——跳过是正确行为，
不是失败。

同理，``paraphrase`` 只使用受控的同义词对（``PARAPHRASE_PAIRS``），
不做自由改写：自由改写需要模型参与，也就把随机性引进了"可复现"的
数据流水线。

## 纪律三：默认不启用会降低质量的算子

``noise`` 算子往指令里塞口语填充词（"嗯，……，你懂我意思吧。"），它模拟的是
真实用户的口语输入。它**默认关闭**（``DEFAULT_OPS`` 不含它），理由是：

> 噪声增强造成的质量下降**恰好不在质量打分的五个维度里**——填充词不会
> 让长度越界、不产生 n-gram 重复、也不影响收尾与来源。也就是说
> ``quality`` 拦不住它。这种"打分器看不见的风险"只能靠**显式标签**
> （``aug:noise``）与默认关闭来治理，不能指望下游自动发现。

这是一条可以带走的判断：**给流水线加一个算子之前，先问"它出错时
谁会报警"**。没人报警的算子，就必须默认关闭。

## 算子的轮转：覆盖度优先于单条样本的丰富度

每条原样本只产出 ``max_per_example`` 个增强样本（缺省 1 个），算子按
"样本序号 + 第几次尝试"轮转选取（``ops[(index + attempt) % len(ops)]``），
**不使用随机数**。这与 day055 的 ``pick_rejected`` 是同一条取舍：
随机抽样会让一部分算子完全抽不到样本，而"哪个算子对模型更有用"这个
问题必须有数据才能回答。确定性轮转保证每个算子在每个批次里都被行使。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from smart_research_agent.domain_data.dedup import (
    DEFAULT_NEAR_DUP_THRESHOLD,
    NearDuplicateIndex,
)
from smart_research_agent.domain_data.errors import DomainDataError
from smart_research_agent.domain_data.fingerprint import (
    DEFAULT_NUM_PERM,
    DEFAULT_SHINGLE_K,
    MINHASH_SEED,
    normalize_for_fingerprint,
)
from smart_research_agent.finetune.schema import TrainingExample

#: 增强样本的标签前缀。它的作用是让"这批数据里有多少是合成的"永远可查：
#: 报告按 ``aug:*`` 统计一次就行，不需要额外维护一张映射表。
AUG_TAG_PREFIX = "aug:"

#: 算子的三种结论（用常量而不是裸字符串，报表键不会拼错）
AUG_APPLIED = "applied"
AUG_GUARD_REJECTED = "guard_rejected"
AUG_NO_CHANGE = "no_change"

#: 全部可用算子。``noise`` 在列但默认不启用，见模块开头的纪律三。
AUGMENT_OPS: tuple[str, ...] = ("prefix", "constraint", "paraphrase", "noise")

#: 缺省启用的算子（不含 ``noise``）
DEFAULT_OPS: tuple[str, ...] = ("prefix", "constraint", "paraphrase")

#: 每条原样本缺省产出几个增强样本。取 1 而不是 3：
#: 增强样本天然比原样本同质，一次补太多会把语义分布往少数几个知识点上压。
DEFAULT_MAX_PER_EXAMPLE = 1

#: ``prefix`` 的口语化前缀：模拟"真实用户会怎么开头问"。
#: 四个变体全部是纯前置语，不含任何事实或限定条件——因此**不可能改变答案的正确性**。
PREFIX_VARIANTS: tuple[str, ...] = (
    "请帮我看看：",
    "想请教一个问题：",
    "麻烦展开说说：",
    "我这边有个疑问：",
)

#: ``noise`` 的首尾口语填充（默认关闭）。首尾成对出现，
#: 单独加个"嗯，"读起来不像人的口吻，也就达不到"模拟口语输入"的目的。
NOISE_VARIANTS: tuple[tuple[str, str], ...] = (
    ("嗯，", "，大概就是这个意思。"),
    ("那个，", "，你能明白我意思吧。"),
    ("就是，", "，反正差不多是这样。"),
)

#: ``paraphrase`` 的受控同义词对（左：原文，右：替换）。**只在指令侧替换，
#: 且每个样本只替换第一处**：换得越多，语义漂移的风险越大，而"问法不同"
#: 这个目的换一处就已经达到。
PARAPHRASE_PAIRS: tuple[tuple[str, str], ...] = (
    ("如何", "怎么"),
    ("为什么", "为何"),
    ("什么是", "什么叫"),
    ("区别", "差别"),
    ("介绍", "讲讲"),
    ("是否", "是不是"),
    ("可以", "能"),
    ("以及", "和"),
)

#: 判定"答案是否以中文为主"的下限。0.3 是实测选出的：
#: 本课程 37 条样本里中文占比最低的 4 条低于它（多为代码/英文术语密集的
#: 答案），这 4 条会被 ``请用中文回答。`` 这条约束**正确地**跳过——
#: 给一条英文答案追加"请用中文回答"是自相矛盾。
CHINESE_RATIO_FLOOR = 0.3

#: ``回答请控制在 N 字以内。`` 约束的字数上限
LENGTH_CONSTRAINT_CHARS = 200

#: "答案开头是否在复述问题"的探测窗口（归一化后的字符数）
RESTATEMENT_PROBE = 8


def cjk_ratio(text: str) -> float:
    """文本中中日韩统一表意文字（U+4E00~U+9FFF）的占比（空文本按 0.0）."""
    if not text:
        return 0.0
    count = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    return count / len(text)


def is_mostly_chinese(text: str) -> bool:
    """答案是否以中文为主（中文占比达到 ``CHINESE_RATIO_FLOOR``）."""
    return cjk_ratio(text) >= CHINESE_RATIO_FLOOR


def within_chars(limit: int) -> Callable[[TrainingExample], bool]:
    """生成一个"输出不超过 limit 字"的判定器（清洗后的字符数口径）."""

    def check(example: TrainingExample) -> bool:
        return len(example.output) <= limit

    return check


def does_not_restate(example: TrainingExample) -> bool:
    """答案开头是否**没有**复述问题.

    复述问题（"问题是：什么是 RAG？答：RAG 是……"）在人工撰写的语料里
    很常见，但它不是好的监督信号——模型会学会"先把问题抄一遍"。
    探测方式：取答案归一化后的前 ``RESTATEMENT_PROBE`` 个字符，
    若它在问题里原样出现，就判定为复述。
    """
    head = normalize_for_fingerprint(example.output)[:RESTATEMENT_PROBE]
    question = normalize_for_fingerprint(example.instruction)
    if not head or not question:
        return True
    return head not in question


@dataclass(frozen=True)
class ConstraintSpec:
    """一条输出约束：文本 + "原答案是否本来就满足"的判定 + 判定理由.

    ``reason`` 是给人看的：报告里出现 ``guard_rejected`` 时，评审需要
    立刻知道"是答案不像中文被跳过，还是答案太长了被跳过"。
    """

    text: str
    satisfied_by: Callable[[TrainingExample], bool]
    reason: str


#: 三条约束。全部满足"答案本来就满足才追加"这一条纪律。
CONSTRAINTS: tuple[ConstraintSpec, ...] = (
    ConstraintSpec(
        text="请用中文回答。",
        satisfied_by=lambda example: is_mostly_chinese(example.output),
        reason="答案不是以中文为主，追加『请用中文回答』会与答案自相矛盾",
    ),
    ConstraintSpec(
        text=f"回答请控制在 {LENGTH_CONSTRAINT_CHARS} 字以内。",
        satisfied_by=within_chars(LENGTH_CONSTRAINT_CHARS),
        reason=f"答案超过 {LENGTH_CONSTRAINT_CHARS} 字，追加长度上限会让参考答案越界",
    ),
    ConstraintSpec(
        text="请直接给出结论，不要复述问题。",
        satisfied_by=does_not_restate,
        reason="答案开头复述了问题，与『不要复述问题』的约束冲突",
    ),
)


@dataclass(frozen=True)
class OpResult:
    """一次算子尝试的结果：结论 + 新指令 + 人读的说明.

    ``text`` 只在 ``status == AUG_APPLIED`` 时有意义。把"结论"与"文本"
    放在同一个对象里，是为了让算子函数无法只返回一半——只返回文本的话，
    "没变化"与"被守门拒绝"就都只能用 ``None`` 表示，而这两种情况在
    报表里必须分开计数。
    """

    status: str
    text: str = ""
    detail: str = ""


def _result(status: str, text: str = "", detail: str = "") -> OpResult:
    """内部便捷构造器（避免每处都写全三个字段）."""
    return OpResult(status=status, text=text, detail=detail)


def transform_prefix(example: TrainingExample, offset: int = 0) -> OpResult:
    """``prefix``：在指令前加一句口语化开场（固定变体表，轮转选取）."""
    total = len(PREFIX_VARIANTS)
    for step in range(total):
        prefix = PREFIX_VARIANTS[(offset + step) % total]
        text = f"{prefix}{example.instruction}"
        if text != example.instruction:
            return _result(AUG_APPLIED, text, f"开场白 {prefix!r}")
    return _result(AUG_NO_CHANGE, detail="所有开场白变体都不改变指令")


def transform_constraint(example: TrainingExample, offset: int = 0) -> OpResult:
    """``constraint``：追加一条**原答案本来就满足**的输出约束.

    轮转到一条约束后先过 ``satisfied_by``；不满足就试下一条。
    全部不满足时返回 ``AUG_GUARD_REJECTED``——这个结论会被单独计数，
    因为"守门拦下了多少条"本身就是一个质量信号：占比过高说明
    数据集里的答案风格与预设约束体系不匹配。
    """
    total = len(CONSTRAINTS)
    blocked: list[str] = []
    for step in range(total):
        spec = CONSTRAINTS[(offset + step) % total]
        if not spec.satisfied_by(example):
            blocked.append(spec.reason)
            continue
        text = f"{example.instruction}\n{spec.text}"
        if text != example.instruction:
            return _result(AUG_APPLIED, text, f"输出约束 {spec.text!r}")
    if blocked:
        return _result(
            AUG_GUARD_REJECTED,
            detail=f"{len(blocked)} 条约束均被守门拦下：{blocked[0]}",
        )
    return _result(AUG_NO_CHANGE, detail="约束未改变指令")


def transform_paraphrase(example: TrainingExample, offset: int = 0) -> OpResult:
    """``paraphrase``：按受控词表替换指令里的**第一处**同义词.

    只替换一处、只换受控词表里的词，是"可复现"与"语义安全"两条要求的
    交点。自由改写需要模型参与，会把随机性和幻觉一起带进数据流水线。
    """
    total = len(PARAPHRASE_PAIRS)
    for step in range(total):
        source, target = PARAPHRASE_PAIRS[(offset + step) % total]
        if source in example.instruction:
            text = example.instruction.replace(source, target, 1)
            if text != example.instruction:
                return _result(AUG_APPLIED, text, f"同义替换 {source!r} → {target!r}")
    return _result(AUG_NO_CHANGE, detail="指令里没有可替换的受控同义词")


def transform_noise(example: TrainingExample, offset: int = 0) -> OpResult:
    """``noise``：首尾插入口语填充词（**默认不启用**，见模块开头纪律三）."""
    total = len(NOISE_VARIANTS)
    for step in range(total):
        head, tail = NOISE_VARIANTS[(offset + step) % total]
        text = f"{head}{example.instruction}{tail}"
        if text != example.instruction:
            return _result(AUG_APPLIED, text, f"口语噪声 {head!r} … {tail!r}")
    return _result(AUG_NO_CHANGE, detail="填充词未改变指令")


@dataclass(frozen=True)
class AugmentOp:
    """一个增强算子：名字、它改善什么、风险等级、以及纯函数实现.

    ``risky=True`` 表示"这个算子会让数据质量下降，且质量打分**看不出来**"。
    标记出来而不是口头说明，是为了让默认算子表（``DEFAULT_OPS``）与
    文档表格都能从同一份定义生成——加算子时忘了更新文档这件事，
    在项目里发生过太多次。
    """

    name: str
    dimension: str
    description: str
    risky: bool
    transform: Callable[[TrainingExample, int], OpResult]


#: 算子登记表。键顺序与 ``AUGMENT_OPS`` 一致，测试会断言两者同源。
OPS: dict[str, AugmentOp] = {
    "prefix": AugmentOp(
        name="prefix",
        dimension="问法多样性",
        description="在指令前加一句口语化开场白（四个固定变体轮转）",
        risky=False,
        transform=transform_prefix,
    ),
    "constraint": AugmentOp(
        name="constraint",
        dimension="指令跟随",
        description="追加一条原答案本来就满足的输出约束（三条约束，先过守门）",
        risky=False,
        transform=transform_constraint,
    ),
    "paraphrase": AugmentOp(
        name="paraphrase",
        dimension="词汇鲁棒性",
        description="按受控词表替换指令里的第一处同义词",
        risky=False,
        transform=transform_paraphrase,
    ),
    "noise": AugmentOp(
        name="noise",
        dimension="噪声鲁棒性",
        description="首尾插入口语填充词（会降低质量，且质量打分察觉不到，默认关闭）",
        risky=True,
        transform=transform_noise,
    ),
}


def resolve_ops(names: Sequence[str]) -> tuple[AugmentOp, ...]:
    """把算子名序列解析成算子对象序列（未知名字抛 ``DomainDataError``）.

    提前解析而不是"遇到再查表"，是因为算子名打错时最糟的表现是**静默少生成
    一批样本**：报告里的 ``by_op`` 少一个键，但没有任何报错。
    """
    resolved: list[AugmentOp] = []
    for name in names:
        if name not in OPS:
            raise DomainDataError(
                f"未知的增强算子 {name!r}，可选：{', '.join(AUGMENT_OPS)}"
            )
        resolved.append(OPS[name])
    return tuple(resolved)


def augment_ops_table() -> list[dict]:
    """算子对照表：**从 ``OPS`` 现场读出**，不抄写（含 ``默认启用`` 列）."""
    return [
        {
            "op": name,
            "dimension": op.dimension,
            "description": op.description,
            "risky": op.risky,
            "enabled_by_default": name in DEFAULT_OPS,
        }
        for name, op in OPS.items()
    ]


def render_augment_table() -> str:
    """把算子对照表渲染成 markdown（文档由代码生成）."""
    header = (
        "| 算子 | 改善维度 | 做法 | 风险 | 默认启用 |\n"
        "|------|---------|------|------|---------|\n"
    )
    rows = "".join(
        f"| `{row['op']}` | {row['dimension']} | {row['description']} | "
        f"{'是' if row['risky'] else '否'} | {'是' if row['enabled_by_default'] else '否'} |\n"
        for row in augment_ops_table()
    )
    return header + rows


def augmented_ops(example: TrainingExample) -> tuple[str, ...]:
    """一条样本身上带的所有 ``aug:*`` 算子名（未增强则为空元组）."""
    return tuple(
        tag[len(AUG_TAG_PREFIX) :]
        for tag in example.tags
        if tag.startswith(AUG_TAG_PREFIX)
    )


def is_augmented(example: TrainingExample) -> bool:
    """这条样本是不是增强产物（按 ``aug:*`` 标签判定）."""
    return bool(augmented_ops(example))


@dataclass(frozen=True)
class AugmentationOutcome:
    """一次 ``augment_example`` 的结果：结论 + 说明 + （成功时的）新样本."""

    status: str
    op: str
    detail: str
    example: TrainingExample | None = None

    def to_dict(self) -> dict:
        """投影为可直接 json.dumps 的字典."""
        return {
            "status": self.status,
            "op": self.op,
            "detail": self.detail,
            "instruction": self.example.instruction if self.example else "",
        }


def augment_example(
    example: TrainingExample, op: AugmentOp, *, offset: int = 0
) -> AugmentationOutcome:
    """用指定算子增强一条样本（纯函数，不读磁盘、不用随机数）.

    返回的新样本满足三条不变式：

    1. ``output`` 与入参逐字相同；
    2. ``source`` / ``license`` / ``system`` / ``input`` 原样继承
       （**来源与许可证必须可追溯**：增强不改变数据出处）；
    3. ``tags`` 追加 ``aug:<op>`` 一个标签（因此"这条是合成的"永远可查）。
    """
    result = op.transform(example, offset)
    if result.status != AUG_APPLIED:
        return AugmentationOutcome(status=result.status, op=op.name, detail=result.detail)
    augmented = TrainingExample(
        instruction=result.text,
        output=example.output,
        input=example.input,
        system=example.system,
        source=example.source,
        tags=example.tags + (f"{AUG_TAG_PREFIX}{op.name}",),
        license=example.license,
    )
    return AugmentationOutcome(
        status=AUG_APPLIED, op=op.name, detail=result.detail, example=augmented
    )


@dataclass
class AugmentReport:
    """增强报告：产出、守门、无变化与重复丢弃四本账.

    四本账的恒等式（测试里钉住）：``inputs + generated == 最终数据集条数``，
    其中 ``generated = applied - dropped_duplicates``。对不上就说明有样本
    在流水线里消失了。
    """

    inputs: int
    applied: int
    dropped_duplicates: int
    threshold: float
    by_op: dict[str, int] = field(default_factory=dict)
    guard_rejected: dict[str, int] = field(default_factory=dict)
    no_change: dict[str, int] = field(default_factory=dict)
    details: list[dict] = field(default_factory=list)

    @property
    def generated(self) -> int:
        """真正进入数据集的增强样本数（已扣掉因重复被丢弃的）."""
        return self.applied - self.dropped_duplicates

    @property
    def expansion_ratio(self) -> float:
        """扩容倍数：``generated / inputs``（空输入按 0.0 计）."""
        return self.generated / self.inputs if self.inputs else 0.0

    @property
    def blocked(self) -> int:
        """被守门拦下的尝试次数（不是样本数——同一条样本可能被拦多次）."""
        return sum(self.guard_rejected.values())

    @property
    def unchanged(self) -> int:
        """算子判定"这条没法增强"的尝试次数."""
        return sum(self.no_change.values())

    def to_dict(self) -> dict:
        """投影为可直接 json.dumps 的字典."""
        return {
            "inputs": self.inputs,
            "applied": self.applied,
            "dropped_duplicates": self.dropped_duplicates,
            "generated": self.generated,
            "expansion_ratio": round(self.expansion_ratio, 4),
            "threshold": self.threshold,
            "by_op": dict(self.by_op),
            "guard_rejected": dict(self.guard_rejected),
            "no_change": dict(self.no_change),
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要（demo 与日志直接打印）."""
        return (
            f"增强：{self.inputs} 条进 → 新增 {self.generated} 条"
            f"（扩容 {self.expansion_ratio:.2f}×）| 算子分布 {self.by_op} | "
            f"守门拦下 {self.blocked} / 无变化 {self.unchanged} / 撞车丢弃 "
            f"{self.dropped_duplicates}"
        )


def augment_dataset(
    examples: Sequence[TrainingExample],
    *,
    ops: Sequence[str] = DEFAULT_OPS,
    max_per_example: int = DEFAULT_MAX_PER_EXAMPLE,
    dedupe: bool = True,
    threshold: float = DEFAULT_NEAR_DUP_THRESHOLD,
    k: int = DEFAULT_SHINGLE_K,
    num_perm: int = DEFAULT_NUM_PERM,
    seed: int = MINHASH_SEED,
) -> tuple[list[TrainingExample], AugmentReport]:
    """批量增强：返回（原样本 + 增强样本, 报告）.

    返回列表的前 ``len(examples)`` 条**就是原样本本身且顺序不变**——
    增强只做追加，从不重排或替换。这条约定让"增强前后逐条对照"成为
    一行代码的事，也让落盘数据集的下游切分（day048 的 ``split_dataset``）
    在增强前后保持可比较。

    ``dedupe=True`` 时，增强样本要过一遍近重复索引。**索引里只放增强样本，
    不放原样本**——这是一条必须写明的取舍，因为它看起来像"漏了"：

    去重的判据是"prompt 字面高度重叠"，而 ``prefix`` / ``constraint`` 类
    增强**天生就是要产出与亲本高度重叠的样本**（"请帮我看看：X" 与 "X"
    的 Jaccard 实测 0.8667）。若把原样本也放进索引，增强样本会先被自己的
    亲本判成重复，整条增强链路大面积空转：本课实测 13 次 ``prefix``
    尝试里有 **5 次**因此被丢弃，白白浪费 38% 的产出。

    所以增强阶段的去重范围被收窄为**增强样本之间**：原样本已经在
    ``near_dedup`` 阶段（含历史批次）去过重，而"与自己的亲本像"是设计意图，
    不是缺陷。**要防的是另一件事**——两条本来就相近的原样本被同一算子
    改成几乎一样的问法，那才会造出真正的重复。

    另外，**原样本优先**：撞车时丢弃的是增强样本，不是原样本——
    原样本的来源可追溯，增强样本可再造。
    """
    resolved = resolve_ops(ops)
    if max_per_example < 0:
        raise DomainDataError(f"max_per_example 不能为负，收到 {max_per_example}")

    result: list[TrainingExample] = list(examples)
    index = NearDuplicateIndex(threshold=threshold, k=k, num_perm=num_perm, seed=seed)

    by_op: dict[str, int] = {}
    guard_rejected: dict[str, int] = {}
    no_change: dict[str, int] = {}
    applied = 0
    dropped = 0

    if resolved and max_per_example:
        for index_of_example, example in enumerate(examples):
            for attempt in range(max_per_example):
                offset = index_of_example + attempt
                op = resolved[offset % len(resolved)]
                outcome = augment_example(example, op, offset=offset)
                if outcome.status == AUG_GUARD_REJECTED:
                    guard_rejected[op.name] = guard_rejected.get(op.name, 0) + 1
                    continue
                if outcome.status != AUG_APPLIED or outcome.example is None:
                    no_change[op.name] = no_change.get(op.name, 0) + 1
                    continue
                applied += 1
                # 登记在"成功产出"之后：撞车的增强样本不进数据集，
                # 但**要计数**——它是"这批增强的多样性不足"的证据。
                if dedupe and index.add(outcome.example).is_duplicate:
                    dropped += 1
                    continue
                by_op[op.name] = by_op.get(op.name, 0) + 1
                result.append(outcome.example)

    report = AugmentReport(
        inputs=len(examples),
        applied=applied,
        dropped_duplicates=dropped,
        threshold=threshold,
        by_op=by_op,
        guard_rejected=guard_rejected,
        no_change=no_change,
    )
    return result, report
