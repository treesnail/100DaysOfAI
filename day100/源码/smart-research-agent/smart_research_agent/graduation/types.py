"""``graduation`` 的口径表（day100 / G2-D1）.

一次性把这一课的"名词表"写全：**4 份交付物 / 4 级自评量程 / 7 条性质（判据分三类）/
10 条笔记 / 5 条边界**。全部是常量，因此可以被测试逐键检查——本课最值钱的一句话
就是这些表能不能**逐键对上**。

```text
4 份交付物  演示剧本 / 课程清单 / 能力自评 / 后续规划（用户拿得走的东西）
4 级量程    未建（0）/ 已建（1）/ 可用（2）/ 已交付（3）
7 条性质    判据分三类：相等（==）/ 上界（<=）/ 下界（>=）
```

## 一、今天最值钱的一句话

> **"交付"不是把结果贴出来，而是把结果变成**别人能复算**的产物：
> 剧本要能被重放、清单要能被 ``importlib`` 核对、自评要能落到证据上、
> 规划要能从缺口推出——四份都如此，交付才算完成。**

day099 把八项能力装成一条链，并给了它一条"同一输入两次运行逐位相同"的判据。
今天在这条链之上再往前一步：**链跑得对 ≠ 交付做得成**。
交付要求的是四份"拿起就能用、放下能被反驳"的产物。

## 二、为什么"自评"要有量程（0~3），而不是一个布尔值

```text
布尔值   "这一项能力有没有"  →  一把尺子只有一个刻度，它只能回答"有/没有"
量程     "这一项能力做到哪一步" →  4 个刻度各自对应一条**可核对的证据**
```

四级的含义（:data:`LEVEL_DESCRIPTIONS`）是"证据走到了哪一步"，不是"感觉怎么样"：

```text
LEVEL_NOT_BUILT  (0)  有承担子包不在场            证据：在场数 < 承担数
LEVEL_BUILT      (1)  承担子包全部在场，但公开符号太少    证据：符号数 < 门槛
LEVEL_USABLE     (2)  前两项成立，但链上那一段没通过        证据：capstone.run 的阶段 ok=False
LEVEL_DELIVERED  (3)  三项都成立                     证据：在场 / 符号 / 阶段全过
```

**把量程写进常量而不是写进某一行的 if**：因为"为什么这一项是 2 分"这件事
必须能被报告逐条回答，而不是留在实现里。

## 三、一条纪律：交付物是**四份**，不是"一份报告"

```text
演示剧本  把"这一次运行"变成一份能被重放的文本（读者不必信任我们，他可以自己跑）
课程清单  把"这门课有什么"变成一份能被 importlib 核对的名册（数出来，不是抄出来）
能力自评  把"我们做到了哪一步"变成 8 行带证据的评级（不是一句"已完成"）
后续规划  把"还差什么"变成一份从**真实缺口**推出的下一步（不是许愿）
```

四份合起来才算"交付完成"——因此第 ② 条性质是"四份都非空"，
而这条性质在报告里的读数就是**缺了几份**。
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------------------
# 1. 四份交付物
# --------------------------------------------------------------------------------------

DELIVERABLE_DEMO = "demo"
DELIVERABLE_SUMMARY = "summary"
DELIVERABLE_ASSESSMENT = "assessment"
DELIVERABLE_ROADMAP = "roadmap"

#: 四份交付物的 id（顺序 = 渲染与检查的顺序）.
DELIVERABLE_ORDER: tuple[str, ...] = (
    DELIVERABLE_DEMO,
    DELIVERABLE_SUMMARY,
    DELIVERABLE_ASSESSMENT,
    DELIVERABLE_ROADMAP,
)

#: 每份交付物的一句话解释（"它回答什么问题、reader 会拿它做什么"）.
DELIVERABLE_DESCRIPTIONS: dict[str, str] = {
    DELIVERABLE_DEMO: "演示剧本：把一次端到端运行变回一份能被重放的文本（读者可自己复跑）",
    DELIVERABLE_SUMMARY: "课程清单：用 importlib 数遍全部子包，给出可核对的名册与计数",
    DELIVERABLE_ASSESSMENT: "能力自评：8 项能力逐项评级，每一级都挂一条证据",
    DELIVERABLE_ROADMAP: "后续规划：从真实缺口推出确定顺序的下一步清单",
}

if set(DELIVERABLE_ORDER) != set(DELIVERABLE_DESCRIPTIONS):  # pragma: no cover - 导入期不变式
    raise ValueError(
        "四份交付物的两张表不一致：DELIVERABLE_ORDER 与 DELIVERABLE_DESCRIPTIONS "
        "必须逐键对齐——少一个键的交付物在报告里只有名字、没有它回答什么问题。"
    )


def require_deliverable(deliverable: str) -> str:
    """校验一份交付物的 id（未知交付物当场拒绝）."""
    if deliverable not in DELIVERABLE_DESCRIPTIONS:
        from smart_research_agent.graduation.errors import ParameterError

        raise ParameterError(
            f"未知的交付物 {deliverable!r}：可选 {list(DELIVERABLE_ORDER)}——"
            "报告按这份名单逐份印；名单外的名字既不会被渲染、也不会被检查。"
        )
    return deliverable


# --------------------------------------------------------------------------------------
# 2. 里程碑
# --------------------------------------------------------------------------------------

#: 这门课的总天数（"100 天计划"这个名字本身就是一个可被核对的常数）.
MILESTONE_TOTAL_DAYS = 100

#: 里程碑的标题（报告与归档正文里印它）.
MILESTONE_TITLE = "100 天 AI 学习计划 · 结业里程碑"

#: 里程碑的一句话说明（"它归档的是什么"）.
MILESTONE_DESCRIPTION = (
    "100 天走完的一段学习轨迹：从 Python 工程化一路到深度学习底层，"
    "主线项目「智研 AI 助手」从空目录长成一个可评估、可复算的系统。"
)

# --------------------------------------------------------------------------------------
# 3. 自评量程（0~3）
# --------------------------------------------------------------------------------------

LEVEL_NOT_BUILT = 0
LEVEL_BUILT = 1
LEVEL_USABLE = 2
LEVEL_DELIVERED = 3

#: 四级量程（顺序 = 从"还没建"到"已交付"）.
LEVELS: tuple[int, ...] = (LEVEL_NOT_BUILT, LEVEL_BUILT, LEVEL_USABLE, LEVEL_DELIVERED)

#: 每一级的名字与它对应的一条证据（**证据是量程的一部分**，不是附录）.
LEVEL_DESCRIPTIONS: dict[int, str] = {
    LEVEL_NOT_BUILT: "未建：有承担子包不在场（证据：在场数 < 承担数）",
    LEVEL_BUILT: "已建：承担子包全部在场，但公开符号数低于门槛（证据：符号数 < 门槛）",
    LEVEL_USABLE: "可用：承担子包在场且符号达标，但端到端链上那一段没有通过（证据：阶段 ok=False）",
    LEVEL_DELIVERED: "已交付：承担子包在场、符号达标、链上阶段通过（证据：三项全过）",
}

#: 自评的**最高级**（"已交付"）——它是 :data:`PROPERTY_ASSESSMENT_WITHIN_SCALE` 的界.
TOP_LEVEL = LEVEL_DELIVERED

#: 一项能力"够用"所需的公开符号数门槛（低于它算"已建但不可用"）.
SYMBOL_FLOOR = 8

if set(LEVELS) != set(LEVEL_DESCRIPTIONS):  # pragma: no cover - 导入期不变式
    raise ValueError(
        "自评量程的两张表不一致：LEVELS 与 LEVEL_DESCRIPTIONS 必须逐键对齐。"
    )

if TOP_LEVEL != max(LEVELS):  # pragma: no cover - 导入期不变式
    raise ValueError(
        f"TOP_LEVEL={TOP_LEVEL} 必须等于 LEVELS 的最大值 {max(LEVELS)}——"
        "否则'最高级'与量程本身会对不上，上界判据会拿一个错的界去兜。"
    )


def require_level(level: int) -> int:
    """校验一个自评评级（越界当场拒绝）."""
    if level not in LEVELS:
        from smart_research_agent.graduation.errors import AssessmentError

        raise AssessmentError(
            f"未知的自评评级 {level!r}：可选 {list(LEVELS)}——"
            "量程外的评级在报告里会被当成越过最高级，而上界判据会因此永远失败。"
        )
    return level


def require_positive_threshold(label: str, value: int) -> int:
    """校验一个 >= 1 的整数阈值（门槛类参数在入口就拒绝非法值）."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        from smart_research_agent.graduation.errors import ParameterError

        raise ParameterError(
            f"{label} 必须是 >= 1 的整数，收到 {value!r}："
            "为 0 的门槛意味着任何一项能力都'达标'，这条自评就失去了分辨力。"
        )
    return value


# --------------------------------------------------------------------------------------
# 4. 七条性质（判据分三类）
# --------------------------------------------------------------------------------------

CRITERION_EQUALITY = "equality"
CRITERION_UPPER_BOUND = "upper_bound"
CRITERION_LOWER_BOUND = "lower_bound"

#: 三类判据（顺序 = 从"逐位相同"到"有方向的界"）.
CRITERIA: tuple[str, ...] = (CRITERION_EQUALITY, CRITERION_UPPER_BOUND, CRITERION_LOWER_BOUND)

CRITERION_DESCRIPTIONS: dict[str, str] = {
    CRITERION_EQUALITY: "相等（==）：读数与期望逐位 / 整数相同（没有容差空间）",
    CRITERION_UPPER_BOUND: "上界（<=）：读数不超过某个界（越少越好）",
    CRITERION_LOWER_BOUND: "下界（>=）：读数不低于某个底（越多越好）",
}

PROPERTY_DEMO_REPLAYS_IDENTICALLY = "demo_replays_identically"
PROPERTY_DELIVERABLES_ARE_COMPLETE = "deliverables_are_complete"
PROPERTY_INVENTORY_ACCOUNTS_FOR_ALL = "inventory_accounts_for_all"
PROPERTY_ROADMAP_COVERS_EVERY_GAP = "roadmap_covers_every_gap"
PROPERTY_ASSESSMENT_WITHIN_SCALE = "assessment_within_scale"
PROPERTY_ASSESSMENT_HAS_EVIDENCE = "assessment_has_evidence"
PROPERTY_SUMMARY_MATCHES_INVENTORY = "summary_matches_inventory"


@dataclass(frozen=True)
class PropertySpec:
    """一条性质：id + 说明 + 判据类别 + "失败意味着什么".

    ``criterion`` 必须是 :data:`CRITERIA` 之一——三类判据里**至少各有一条**，
    这条闭合检查在导入期兑现（见文件末尾）。
    """

    id: str
    description: str
    criterion: str
    failure: str

    def __post_init__(self) -> None:
        from smart_research_agent.graduation.errors import ParameterError

        if not self.id or not self.description or not self.failure:
            raise ParameterError(f"性质 {self.id!r} 的说明与失败语义都不能为空。")
        if self.criterion not in CRITERIA:
            raise ParameterError(
                f"性质 {self.id!r} 的判据 {self.criterion!r} 未知：可选 {list(CRITERIA)}。"
            )

    def to_dict(self) -> dict[str, object]:
        """摊平成一行字段."""
        return {
            "id": self.id,
            "description": self.description,
            "criterion": self.criterion,
            "failure": self.failure,
        }

    def line(self) -> str:
        """一行说明：``[equality] demo_replays_identically | ...``."""
        return f"[{self.criterion}] {self.id} | {self.description}"


#: 7 条性质（顺序 = 交付检查的顺序）.
PROPERTY_SPECS: dict[str, PropertySpec] = {
    PROPERTY_DEMO_REPLAYS_IDENTICALLY: PropertySpec(
        id=PROPERTY_DEMO_REPLAYS_IDENTICALLY,
        description="演示剧本重放两次**逐位相同**（同一份剧本，不是同一次运气）",
        criterion=CRITERION_EQUALITY,
        failure="剧本里混进了未固定的量（时间 / uuid / 采样），第二次重放与第一次不同",
    ),
    PROPERTY_DELIVERABLES_ARE_COMPLETE: PropertySpec(
        id=PROPERTY_DELIVERABLES_ARE_COMPLETE,
        description="四份交付物全部非空（演示剧本 / 课程清单 / 能力自评 / 后续规划）",
        criterion=CRITERION_EQUALITY,
        failure="某一份交付物是空白的，而'没写'与'这一份不需要'读起来一样",
    ),
    PROPERTY_INVENTORY_ACCOUNTS_FOR_ALL: PropertySpec(
        id=PROPERTY_INVENTORY_ACCOUNTS_FOR_ALL,
        description="清单既不重复也不漏项（发现集合 == 记录集合）",
        criterion=CRITERION_EQUALITY,
        failure="同一子包被数了两遍，或有一个子包从名册里消失了",
    ),
    PROPERTY_ROADMAP_COVERS_EVERY_GAP: PropertySpec(
        id=PROPERTY_ROADMAP_COVERS_EVERY_GAP,
        description="每一个真实缺口都有对应规划项（未覆盖的缺口数 == 0）",
        criterion=CRITERION_EQUALITY,
        failure="有一个缺口没有下一步——一份'看起来完整'的规划漏掉了最该补的那一项",
    ),
    PROPERTY_ASSESSMENT_WITHIN_SCALE: PropertySpec(
        id=PROPERTY_ASSESSMENT_WITHIN_SCALE,
        description="每一项能力的自评评级不超过最高级（读数 <= TOP_LEVEL）",
        criterion=CRITERION_UPPER_BOUND,
        failure="某一项评级越过了量程——一个越界的分数在报告里会被读成'已经交付'",
    ),
    PROPERTY_ASSESSMENT_HAS_EVIDENCE: PropertySpec(
        id=PROPERTY_ASSESSMENT_HAS_EVIDENCE,
        description="每一项能力都至少挂一条证据（证据条数 >= 1）",
        criterion=CRITERION_LOWER_BOUND,
        failure="某一项能力只有分数没有证据——一个没有证据的 3 分与'我猜的 3 分'一样",
    ),
    PROPERTY_SUMMARY_MATCHES_INVENTORY: PropertySpec(
        id=PROPERTY_SUMMARY_MATCHES_INVENTORY,
        description="课程清单与课程总结对同一批子包给出一致的计数（差异项数 == 0）",
        criterion=CRITERION_EQUALITY,
        failure="总结里的子包数与清单里数出来的不一样——两个数各说各的",
    ),
}

#: 7 条性质的 id（与 :data:`PROPERTY_SPECS` 逐键对齐，供 :mod:`graduation.verify` 对名单）.
GRADUATION_PROPERTIES: tuple[str, ...] = tuple(PROPERTY_SPECS)


def require_property(property_id: str) -> str:
    """校验一条性质 id（未知性质当场拒绝）."""
    if property_id not in PROPERTY_SPECS:
        from smart_research_agent.graduation.errors import ParameterError

        raise ParameterError(f"未知的性质 {property_id!r}：可选 {list(GRADUATION_PROPERTIES)}。")
    return property_id


#: 三类判据各自至少要有一条性质（否则报告里那一类判据是空的）.
_CRITERION_COVERAGE: dict[str, int] = {
    criterion: sum(1 for spec in PROPERTY_SPECS.values() if spec.criterion == criterion)
    for criterion in CRITERIA
}
if any(count == 0 for count in _CRITERION_COVERAGE.values()):  # pragma: no cover - 导入期不变式
    raise ValueError(
        f"三类判据至少要各有一条性质，当前分布 {_CRITERION_COVERAGE}——"
        "缺一类判据的后果是：'只能从下面兜住'的性质被写成了相等判据，"
        "于是它会在变得更好时误报失败。"
    )

# --------------------------------------------------------------------------------------
# 5. 十条笔记（这是"交付起来"这件事的成品）
# --------------------------------------------------------------------------------------

GRADUATION_NOTES: dict[str, str] = {
    "delivery_must_be_reproducible": (
        "交付的判据是**别人能复算**，不是我们能跑通：剧本要能被重放，"
        "清单要能被 importlib 核对——'我这台机器跑过'不是一份产物。"
    ),
    "four_deliverables_not_one_report": (
        "交付物是四份而不是一份报告：剧本、清单、自评、规划各自回答一个独立的问题，"
        "合成一坨之后，任何一个问题失真的都看不出来。"
    ),
    "score_needs_evidence": (
        "自评的每一级都必须挂一条证据：'在场 / 符号 / 阶段'三项全过才是已交付。"
        "没有证据的满分，与'我猜的满分'在报告里长得一模一样。"
    ),
    "inventory_is_counted_not_copied": (
        "课程清单是**数出来**的（importlib 遍历全部子包），不是抄出来的："
        "抄一遍的名字不会随着代码变化而变化。"
    ),
    "roadmap_comes_from_real_gaps": (
        "规划只从**真实缺口**推出：无人认领的子包、没到最高级的能力。"
        "从愿望出发的规划读起来更好，但它无法被反驳。"
    ),
    "milestone_is_a_version_event": (
        "里程碑是一次**版本事件**：它给出一份摘要，摘要相同即同一份产物。"
        "把'100 天'写成一个手写的 100，与'我们真的走完了 100 天'读起来一样。"
    ),
    "demo_shows_instead_of_telling": (
        "剧本的价值是**展示而不是宣称**：它把九段读数原样贴出来，"
        "读者可以自己跑一遍对照，而不必相信任何一句话。"
    ),
    "replay_catches_unfixed_quantities": (
        "重放是抓'未固定量'最省力的办法：时间戳、uuid、字典序、采样——"
        "凡是没有被固定的量，第二次重放都会把它们露出来。"
    ),
    "summary_and_inventory_must_agree": (
        "总结与清单必须给出**同一个计数**：两个数各说各的时，"
        "读者只会相信看起来更合理的那一个，而正确的那个已经无从判断。"
    ),
    "boundaries_are_part_of_the_delivery": (
        "边界是交付的一部分：一份不说清'我不承诺什么'的交付，"
        "会让读者把'这一次没做'读成'这件事本来就该这样'。"
    ),
}

#: 十条笔记的键（顺序即写入顺序，报告里读它）.
GRADUATION_NOTES_ORDER: tuple[str, ...] = tuple(GRADUATION_NOTES)

# --------------------------------------------------------------------------------------
# 6. 边界（**这一课明确不承诺的事**）
# --------------------------------------------------------------------------------------

GRADUATION_BOUNDARIES: tuple[str, ...] = (
    "本包**不新增任何第三方依赖**：四份交付物用到的每一个读数都来自快照内的既有子系统",
    "本包**不改动任何既有模块、既有测试、pyproject 与 docs 下的既有文件**——"
    "它只新增文件，既有子系统一行未改",
    "本包交付的是**产物与判据**，不是新的模型能力：它不重写检索器、生成器或指标",
    "本包的全部读数**离线、确定性**：复用 day099 的固定底座，不联网、不读任何环境变量密钥",
    "能力自评是**基于证据的评级**，不是对真实世界效果的承诺——"
    "它衡量的是'这一项能力在本仓库里被建到哪一步'，不是'它在生产环境里有多强'",
)

# --------------------------------------------------------------------------------------
# 7. 记录
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DeliverableSpec:
    """一份交付物：id + 标题 + 说明 + 它回答的问题.

    ``question`` 是这份交付物**唯一要回答**的那一个问题：四份交付物各有一个，
    合起来不重不漏——否则读者会以为某一个问题已经被回答过了。
    """

    id: str
    title: str
    description: str
    question: str

    def __post_init__(self) -> None:
        from smart_research_agent.graduation.errors import ParameterError

        if self.id not in DELIVERABLE_DESCRIPTIONS:
            raise ParameterError(f"未知的交付物 {self.id!r}：可选 {list(DELIVERABLE_ORDER)}。")
        if not self.title or not self.description or not self.question:
            raise ParameterError(f"交付物 {self.id!r} 的标题 / 说明 / 问题都不能为空。")

    def to_dict(self) -> dict[str, object]:
        """摊平成一行字段."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "question": self.question,
        }

    def line(self) -> str:
        """一行说明：``demo | 演示剧本 | 它回答：这一次运行能被重放吗``."""
        return f"{self.id:<11} | {self.title} | 它回答：{self.question}"


#: 4 份交付物（键与 :data:`DELIVERABLE_ORDER` 逐项对齐）.
DELIVERABLE_SPECS: dict[str, DeliverableSpec] = {
    DELIVERABLE_DEMO: DeliverableSpec(
        id=DELIVERABLE_DEMO,
        title="演示剧本",
        description=DELIVERABLE_DESCRIPTIONS[DELIVERABLE_DEMO],
        question="这一次端到端运行，能不能被别人原样重放？",
    ),
    DELIVERABLE_SUMMARY: DeliverableSpec(
        id=DELIVERABLE_SUMMARY,
        title="课程清单",
        description=DELIVERABLE_DESCRIPTIONS[DELIVERABLE_SUMMARY],
        question="这门课到底交付了哪些子包，各有多少东西？",
    ),
    DELIVERABLE_ASSESSMENT: DeliverableSpec(
        id=DELIVERABLE_ASSESSMENT,
        title="能力自评",
        description=DELIVERABLE_DESCRIPTIONS[DELIVERABLE_ASSESSMENT],
        question="八项能力分别建到了哪一步，各自的证据是什么？",
    ),
    DELIVERABLE_ROADMAP: DeliverableSpec(
        id=DELIVERABLE_ROADMAP,
        title="后续规划",
        description=DELIVERABLE_DESCRIPTIONS[DELIVERABLE_ROADMAP],
        question="按真实缺口排，下一步该补什么？",
    ),
}

if set(DELIVERABLE_SPECS) != set(DELIVERABLE_ORDER):  # pragma: no cover - 导入期不变式
    raise ValueError(
        "4 份交付物的两张表不一致：DELIVERABLE_SPECS 的键必须与 DELIVERABLE_ORDER 逐项对应——"
        "少一份的交付物既不会被渲染、也不会被完整性检查发现。"
    )


@dataclass(frozen=True)
class LevelSpec:
    """自评量程的一格：级别 + 名字 + 证据口径."""

    level: int
    name: str
    criterion: str

    def __post_init__(self) -> None:
        from smart_research_agent.graduation.errors import AssessmentError

        if self.level not in LEVELS:
            raise AssessmentError(f"未知的自评评级 {self.level!r}：可选 {list(LEVELS)}。")
        if not self.name or not self.criterion:
            raise AssessmentError(f"评级 {self.level!r} 的名字与证据口径都不能为空。")

    def to_dict(self) -> dict[str, object]:
        """摊平成一行字段."""
        return {"level": self.level, "name": self.name, "criterion": self.criterion}

    def line(self) -> str:
        """一行说明：``3 已交付 | 证据：在场 / 符号 / 阶段全过``."""
        return f"{self.level} {self.name} | 证据：{self.criterion}"


#: 4 格量程（名字与证据口径逐格写死——量程是判据，不是从长句里切出来的）。
LEVEL_SPECS: dict[int, LevelSpec] = {
    LEVEL_NOT_BUILT: LevelSpec(
        level=LEVEL_NOT_BUILT,
        name="未建",
        criterion="在场数 < 承担数（有承担子包解析不到）",
    ),
    LEVEL_BUILT: LevelSpec(
        level=LEVEL_BUILT,
        name="已建",
        criterion="符号数 < 门槛（子包在场，但公开面太薄）",
    ),
    LEVEL_USABLE: LevelSpec(
        level=LEVEL_USABLE,
        name="可用",
        criterion="端到端链上那一段 ok=False（接口在，但这一步没跑通）",
    ),
    LEVEL_DELIVERED: LevelSpec(
        level=LEVEL_DELIVERED,
        name="已交付",
        criterion="在场 / 符号 / 阶段三项全过",
    ),
}

if set(LEVEL_SPECS) != set(LEVELS):  # pragma: no cover - 导入期不变式
    raise ValueError("4 格量程的两张表不一致：LEVEL_SPECS 的键必须与 LEVELS 逐项对应。")


def deliverables() -> tuple[DeliverableSpec, ...]:
    """按次序返回 4 份交付物（渲染与检查共用同一份顺序）."""
    return tuple(DELIVERABLE_SPECS[key] for key in DELIVERABLE_ORDER)


def level_specs() -> tuple[LevelSpec, ...]:
    """按次序返回 4 格量程（报告与测试共用同一份顺序）."""
    return tuple(LEVEL_SPECS[level] for level in LEVELS)


def property_specs() -> tuple[PropertySpec, ...]:
    """按次序返回 7 条性质（verify 与报告共用同一份顺序）."""
    return tuple(PROPERTY_SPECS[key] for key in GRADUATION_PROPERTIES)


#: 交付物名单的一个别名（按名字取一份交付物的规格；与 :data:`DELIVERABLE_SPECS` 同一份）.
DELIVERABLES: dict[str, DeliverableSpec] = DELIVERABLE_SPECS


__all__ = [
    "CRITERIA",
    "CRITERION_DESCRIPTIONS",
    "CRITERION_EQUALITY",
    "CRITERION_LOWER_BOUND",
    "CRITERION_UPPER_BOUND",
    "DELIVERABLES",
    "DELIVERABLE_ASSESSMENT",
    "DELIVERABLE_DEMO",
    "DELIVERABLE_DESCRIPTIONS",
    "DELIVERABLE_ORDER",
    "DELIVERABLE_ROADMAP",
    "DELIVERABLE_SPECS",
    "DELIVERABLE_SUMMARY",
    "GRADUATION_BOUNDARIES",
    "GRADUATION_NOTES",
    "GRADUATION_NOTES_ORDER",
    "GRADUATION_PROPERTIES",
    "LEVELS",
    "LEVEL_BUILT",
    "LEVEL_DELIVERED",
    "LEVEL_DESCRIPTIONS",
    "LEVEL_NOT_BUILT",
    "LEVEL_SPECS",
    "LEVEL_USABLE",
    "MILESTONE_DESCRIPTION",
    "MILESTONE_TITLE",
    "MILESTONE_TOTAL_DAYS",
    "PROPERTY_ASSESSMENT_HAS_EVIDENCE",
    "PROPERTY_ASSESSMENT_WITHIN_SCALE",
    "PROPERTY_DELIVERABLES_ARE_COMPLETE",
    "PROPERTY_DEMO_REPLAYS_IDENTICALLY",
    "PROPERTY_INVENTORY_ACCOUNTS_FOR_ALL",
    "PROPERTY_ROADMAP_COVERS_EVERY_GAP",
    "PROPERTY_SPECS",
    "PROPERTY_SUMMARY_MATCHES_INVENTORY",
    "SYMBOL_FLOOR",
    "TOP_LEVEL",
    "DeliverableSpec",
    "LevelSpec",
    "PropertySpec",
    "deliverables",
    "level_specs",
    "property_specs",
    "require_deliverable",
    "require_level",
    "require_positive_threshold",
    "require_property",
]
