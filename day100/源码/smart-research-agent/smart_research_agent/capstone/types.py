"""``capstone`` 的口径表（day099 / G1-D1）.

一次性把这一课的"名词表"写全：**8 项能力 / 9 个阶段 / 12 个候选子包 /
7 条性质（判据分三类）/ 10 条笔记 / 5 条边界**。全部是常量，因此可以被测试
逐键检查——本包最值钱的一句话就是这些表能不能**逐键对上**。

```text
8 项能力   用户能看见的八种能力：输入护栏 / 任务规划 / 混合检索 / 上下文打包 /
           接地生成 / 离线评估 / 成本与追踪 / 工具调用
9 个阶段   guard → plan → retrieve → pack → generate → ground → evaluate → account → trace
12 个子包  能力清单里的承担者与"无人认领"的支撑包（清单报告要说得出来）
7 条性质   判据分三类：相等（==）/ 上界（<=）/ 下界（>=）
```

## 一、今天最值钱的一句话

> **"装上了"与"接上了"是两件事：前者是 import 得到，后者是一次端到端调用里
> 每个阶段都真的跑过、而且读数能被复算。**

因此本课不新增任何第三方依赖、不重写任何既有子系统，只做两件事：

```text
1) 把分散在 security / agent / retrieval / evaluation / observability / tools /
   llm / vectorstore 里的能力，**真实装配**成一条能一次跑完的调用链
   （assembly.SystemAssembly.run）；
2) 从清单把这条链渲染成**最终版 README 与架构文档**
   （document.render_readme / render_architecture），并检查文档覆盖全部 8 项能力。
```

## 二、为什么判据要分三类

与 day088 的 `principle_map.types` 同源，但今天多出第三类：

```text
相等（==）   读数与期望**逐位相同**（覆盖计数、阶段序列、两次运行的一致性、
             文档缺失条数）
上界（<=）   读数**不超过**某个界（幻觉引用条数 <= 0、成本相对差 <= 容差）
下界（>=）   读数**不低于**某个底（端到端检索的召回 >= 1.0）
```

第三类不是凑数：召回是一条"只能从下面兜住"的性质——把召回写成"等于 1.0"的
相等判据，会在**取回更多条**（更好）时误报失败；写成下界才是它真正的方向。
把三类混成一个 `==`，报告里就会出现"提高了召回反而没通过"这种事。

## 三、一条纪律：能力必须落到**子包**，而不是落到"某个函数"里

day088 的图上，落点是一个 `模块.函数` 字符串；今天上移一层——落点是**子包**，
因为"结业项目整合"要回答的问题是"这一项能力由哪个包负责"，而不是"由哪一行实现"。
因此 :class:`Capability` 带的是一份 ``owners``（承担子包清单），
而 :class:`~capstone.manifest.Manifest` 会真的用 ``importlib`` 把每个子包解析一遍、
数出它的公开符号数——解析不到的承担者会被点名，而不是被静默跳过。
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------------------
# 1. 十二个候选子包（能力清单的"落点池"）
# --------------------------------------------------------------------------------------

#: 本课允许出现在能力清单里的子包（顺序即清单报告的打印顺序）.
CANDIDATE_SUBPACKAGES: tuple[str, ...] = (
    "security",
    "agent",
    "retrieval",
    "evaluation",
    "observability",
    "tools",
    "llm",
    "vectorstore",
    "indexing",
    "mcp_server",
    "finetune",
    "api",
)

#: 每个候选子包的一句话解释（"它是干什么的、它属不属于这 8 项能力"）.
SUBPACKAGE_DESCRIPTIONS: dict[str, str] = {
    "security": "输入侧护栏：注入检测与内容审核（8 项能力之一的承担者）",
    "agent": "Agent 规划：把目标拆成一串可执行子任务（8 项能力之一的承担者）",
    "retrieval": "检索与生成：混合检索 / 上下文打包 / 接地生成（三项能力的承担者）",
    "evaluation": "离线评估：召回 / 精确 / MRR / NDCG 四条纯函数指标（一项能力的承担者）",
    "observability": "观测：成本追踪与 trace（一项能力的承担者）",
    "tools": "工具：计算器（一项能力的承担者）",
    "llm": "模型层：MockLLM / 编码器——被多项能力**借用**的支撑包",
    "vectorstore": "向量库：被检索能力**借用**的支撑包",
    "indexing": "索引清单与版本：本项目已交付，但**不属于**这 8 项能力",
    "mcp_server": "MCP 服务端：本项目已交付，但**不属于**这 8 项能力清单",
    "finetune": "微调数据管线：本项目已交付，但**不属于**这 8 项能力",
    "api": "HTTP 端点：本项目已交付，但**不属于**这 8 项能力（它是外壳）",
}

if set(CANDIDATE_SUBPACKAGES) != set(  # pragma: no cover - 导入期不变式
    SUBPACKAGE_DESCRIPTIONS
):
    raise ValueError(
        "候选子包的两张表不一致：CANDIDATE_SUBPACKAGES 与 SUBPACKAGE_DESCRIPTIONS "
        "必须逐键对齐——少一个键的子包在清单报告里只有名字、没有它属于哪一项能力。"
    )

# --------------------------------------------------------------------------------------
# 2. 八个阶段（顺序 = 数据流动的顺序）
# --------------------------------------------------------------------------------------

STAGE_GUARD = "guard"
STAGE_PLAN = "plan"
STAGE_RETRIEVE = "retrieve"
STAGE_PACK = "pack"
STAGE_GENERATE = "generate"
STAGE_GROUND = "ground"
STAGE_EVALUATE = "evaluate"
STAGE_ACCOUNT = "account"
STAGE_TRACE = "trace"

#: 九个阶段（顺序 = 端到端调用链的顺序）.
ASSEMBLY_STAGES: tuple[str, ...] = (
    STAGE_GUARD,
    STAGE_PLAN,
    STAGE_RETRIEVE,
    STAGE_PACK,
    STAGE_GENERATE,
    STAGE_GROUND,
    STAGE_EVALUATE,
    STAGE_ACCOUNT,
    STAGE_TRACE,
)

#: 每个阶段的一句话解释（"这一步在端到端链里干什么"）.
STAGE_DESCRIPTIONS: dict[str, str] = {
    STAGE_GUARD: "护栏：注入检测 + 内容审核，决定这次请求能不能往下走",
    STAGE_PLAN: "规划：把问题拆成有序子任务（Planner + MockLLM 剧本）",
    STAGE_RETRIEVE: "检索：混合检索器取回 top-k 命中（两路共用一份过滤与深度）",
    STAGE_PACK: "打包：把命中折成带编号、有预算的提示词片段（[n] 从 1 起连续）",
    STAGE_GENERATE: "生成：把上下文交给模型，要求它给出可核对的引用",
    STAGE_GROUND: "接地：把答案里的 [n] 与那次提示词的编号对账（幻觉引用 = 0）",
    STAGE_EVALUATE: "评估：召回 / 精确 / MRR / NDCG 四条离线指标现场算出",
    STAGE_ACCOUNT: "记账：按模型与接口归因 token 与费用（价格表折算）",
    STAGE_TRACE: "追踪：把这一整次调用记成一条 trace（root + 每个阶段一个 span）",
}

#: 每个阶段**由哪一项能力**承担（阶段 ↔ 能力的映射，要能被逐键检查）.
STAGE_CAPABILITIES: dict[str, str] = {
    STAGE_GUARD: "input_guard",
    STAGE_PLAN: "task_planning",
    STAGE_RETRIEVE: "hybrid_retrieval",
    STAGE_PACK: "context_packing",
    STAGE_GENERATE: "grounded_generation",
    STAGE_GROUND: "grounded_generation",
    STAGE_EVALUATE: "offline_evaluation",
    STAGE_ACCOUNT: "cost_and_tracing",
    STAGE_TRACE: "cost_and_tracing",
}

#: 每个阶段"读什么"（一句话口径，报告里印它）.
STAGE_READERS: dict[str, str] = {
    STAGE_GUARD: "报告的 flagged 条数（越少越好，0 = 干净）",
    STAGE_PLAN: "拆出的子任务条数（本课写死 3 条）",
    STAGE_RETRIEVE: "取回的命中条数（top-k 内的）",
    STAGE_PACK: "进了上下文的引用条数（= used_hits 条数）",
    STAGE_GENERATE: "这次调没调模型（1 = 调了，0 = 被护栏拦下）",
    STAGE_GROUND: "幻觉引用条数（越少越好，0 = 全部编号都能对上）",
    STAGE_EVALUATE: "端到端检索的召回（金标准命中率）",
    STAGE_ACCOUNT: "这一次调用的总费用（美元，价格表折算）",
    STAGE_TRACE: "trace 里的 span 条数（root + 每个阶段一个）",
}

if not (
    set(ASSEMBLY_STAGES) == set(STAGE_DESCRIPTIONS) == set(STAGE_CAPABILITIES) == set(STAGE_READERS)
):  # pragma: no cover - 导入期不变式
    raise ValueError(
        "九个阶段的四张表不一致：ASSEMBLY_STAGES / STAGE_DESCRIPTIONS / "
        "STAGE_CAPABILITIES / STAGE_READERS 必须逐键对齐——少一个键的阶段"
        "在报告里只有名字、没有'它由哪项能力承担'。"
    )

# --------------------------------------------------------------------------------------
# 3. 八项能力（**用户能看见的八种能力**）
# --------------------------------------------------------------------------------------

CAPABILITY_INPUT_GUARD = "input_guard"
CAPABILITY_TASK_PLANNING = "task_planning"
CAPABILITY_HYBRID_RETRIEVAL = "hybrid_retrieval"
CAPABILITY_CONTEXT_PACKING = "context_packing"
CAPABILITY_GROUNDED_GENERATION = "grounded_generation"
CAPABILITY_OFFLINE_EVALUATION = "offline_evaluation"
CAPABILITY_COST_AND_TRACING = "cost_and_tracing"
CAPABILITY_TOOL_EXECUTION = "tool_execution"

#: 八项能力的 id（顺序 = README 里列举的顺序）.
CAPABILITY_ORDER: tuple[str, ...] = (
    CAPABILITY_INPUT_GUARD,
    CAPABILITY_TASK_PLANNING,
    CAPABILITY_HYBRID_RETRIEVAL,
    CAPABILITY_CONTEXT_PACKING,
    CAPABILITY_GROUNDED_GENERATION,
    CAPABILITY_OFFLINE_EVALUATION,
    CAPABILITY_COST_AND_TRACING,
    CAPABILITY_TOOL_EXECUTION,
)


@dataclass(frozen=True)
class Capability:
    """一项能力：标题 + 说明 + **承担子包清单** + 它落在哪个阶段.

    ``owners`` 是**字符串的子包名**（不是对象引用）：名字可以被 ``importlib``
    解析、被计数、被报告——一个对象引用只能"看起来对"。
    ``stage`` 是这项能力在端到端链里**最能被看见**的那一段（用于阶段表）。
    """

    id: str
    title: str
    description: str
    owners: tuple[str, ...]
    stage: str

    def __post_init__(self) -> None:
        from smart_research_agent.capstone.errors import CapabilityError, ParameterError

        if self.id not in CAPABILITY_ORDER:
            raise CapabilityError(
                f"未知的能力 {self.id!r}：可选 {list(CAPABILITY_ORDER)}。"
                "回退到某一项能力的后果是——两项能力在报告里被算成了同一个名字。"
            )
        if not self.title or not self.description:
            raise CapabilityError(f"能力 {self.id!r} 必须有标题与说明。")
        if not self.owners:
            raise CapabilityError(
                f"能力 {self.id!r} 没有承担子包：一项无人认领的能力在清单里"
                "会以'已覆盖'的形式出现，而它其实没有落点。"
            )
        for owner in self.owners:
            if owner not in CANDIDATE_SUBPACKAGES:
                raise ParameterError(
                    f"能力 {self.id!r} 的承担子包 {owner!r} 不在候选清单里："
                    f"可选 {list(CANDIDATE_SUBPACKAGES)}——"
                    "清单外的名字既不会被解析、也不会被计入覆盖，只会让覆盖报告看起来是对的。"
                )
        if self.stage not in ASSEMBLY_STAGES:
            raise ParameterError(
                f"能力 {self.id!r} 的阶段 {self.stage!r} 未知：可选 {list(ASSEMBLY_STAGES)}。"
            )

    def to_dict(self) -> dict[str, object]:
        """摊平成一行字段."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "owners": list(self.owners),
            "stage": self.stage,
        }

    def line(self) -> str:
        """一行说明：``[guard] input_guard | 承担：security、llm``."""
        return f"[{self.stage}] {self.id} | {self.title} | 承担：{'、'.join(self.owners)}"


#: 8 项能力（README 的最终能力清单；键与 :data:`CAPABILITY_ORDER` 逐项对齐）.
CAPABILITIES: dict[str, Capability] = {
    CAPABILITY_INPUT_GUARD: Capability(
        id=CAPABILITY_INPUT_GUARD,
        title="输入侧护栏",
        description="注入检测与内容审核：把最常见的公开攻击话术与 PII 拦在链路之外",
        owners=("security", "llm"),
        stage=STAGE_GUARD,
    ),
    CAPABILITY_TASK_PLANNING: Capability(
        id=CAPABILITY_TASK_PLANNING,
        title="任务规划",
        description="把目标拆成一串有序子任务，交给执行层（Planner + MockLLM 剧本）",
        owners=("agent", "llm"),
        stage=STAGE_PLAN,
    ),
    CAPABILITY_HYBRID_RETRIEVAL: Capability(
        id=CAPABILITY_HYBRID_RETRIEVAL,
        title="混合检索",
        description="向量路 + 关键词路融合成一份带多路证据的名单（过滤与深度两路共用一份）",
        owners=("retrieval", "vectorstore"),
        stage=STAGE_RETRIEVE,
    ),
    CAPABILITY_CONTEXT_PACKING: Capability(
        id=CAPABILITY_CONTEXT_PACKING,
        title="上下文打包",
        description="按预算截断单条、整包丢尾、[n] 从 1 起连续编号，并留下三条计数",
        owners=("retrieval",),
        stage=STAGE_PACK,
    ),
    CAPABILITY_GROUNDED_GENERATION: Capability(
        id=CAPABILITY_GROUNDED_GENERATION,
        title="接地生成",
        description="生成答案并做编号层面核对：有效引用 / 幻觉引用 / 未引用三组编号",
        owners=("retrieval", "llm"),
        stage=STAGE_GENERATE,
    ),
    CAPABILITY_OFFLINE_EVALUATION: Capability(
        id=CAPABILITY_OFFLINE_EVALUATION,
        title="离线评估",
        description="召回 / 精确 / MRR / NDCG 四条纯函数指标，只看名单与金标准",
        owners=("evaluation",),
        stage=STAGE_EVALUATE,
    ),
    CAPABILITY_COST_AND_TRACING: Capability(
        id=CAPABILITY_COST_AND_TRACING,
        title="成本与追踪",
        description="按模型与接口归因 token 与费用，并把一次请求记成一条 trace",
        owners=("observability", "llm"),
        stage=STAGE_ACCOUNT,
    ),
    CAPABILITY_TOOL_EXECUTION: Capability(
        id=CAPABILITY_TOOL_EXECUTION,
        title="工具调用",
        description="基于 AST 白名单的安全算术求值（在规划阶段做一次确定性自检）",
        owners=("tools",),
        stage=STAGE_PLAN,
    ),
}

if set(CAPABILITIES) != set(CAPABILITY_ORDER):  # pragma: no cover - 导入期不变式
    raise ValueError(
        "8 项能力的两张表不一致：CAPABILITIES 的键必须与 CAPABILITY_ORDER 逐项对应——"
        "少一项的能力既不会被渲染进 README、也不会被覆盖检查发现。"
    )


def require_capability(capability: str) -> str:
    """校验一个能力 id（未知能力当场拒绝）."""
    if capability not in CAPABILITIES:
        from smart_research_agent.capstone.errors import CapabilityError

        raise CapabilityError(f"未知的能力 {capability!r}：可选 {list(CAPABILITY_ORDER)}。")
    return capability


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

PROPERTY_CAPABILITIES_ARE_COVERED = "capabilities_are_covered"
PROPERTY_STAGES_MATCH_SPEC = "stages_match_spec"
PROPERTY_ASSEMBLY_IS_REPRODUCIBLE = "assembly_is_reproducible"
PROPERTY_RETRIEVAL_RECALL_MEETS_FLOOR = "retrieval_recall_meets_floor"
PROPERTY_GROUNDING_HAS_NO_HALLUCINATION = "grounding_has_no_hallucination"
PROPERTY_COST_MATCHES_HAND_FORMULA = "cost_matches_hand_formula"
PROPERTY_DOCUMENT_COVERS_ALL_CAPABILITIES = "document_covers_all_capabilities"


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
        from smart_research_agent.capstone.errors import CapabilityError, ParameterError

        if not self.id or not self.description or not self.failure:
            raise CapabilityError(f"性质 {self.id!r} 的说明与失败语义都不能为空。")
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
        """一行说明：``[equality] capabilities_are_covered | ...``."""
        return f"[{self.criterion}] {self.id} | {self.description}"


#: 7 条性质（顺序 = 端到端检查的顺序）.
PROPERTY_SPECS: dict[str, PropertySpec] = {
    PROPERTY_CAPABILITIES_ARE_COVERED: PropertySpec(
        id=PROPERTY_CAPABILITIES_ARE_COVERED,
        description="8 项能力全部被覆盖：每一项都有承担子包，且子包能被解析",
        criterion=CRITERION_EQUALITY,
        failure="某一行清单写了一个 importlib 解析不到的子包——它是一条指向空气的边",
    ),
    PROPERTY_STAGES_MATCH_SPEC: PropertySpec(
        id=PROPERTY_STAGES_MATCH_SPEC,
        description="运行记录的阶段序列与 ASSEMBLY_STAGES 逐位相同（不多不少不重不乱）",
        criterion=CRITERION_EQUALITY,
        failure="有人加了一个阶段却忘了更新表，或某个阶段被跑了两次——链的形状变了",
    ),
    PROPERTY_ASSEMBLY_IS_REPRODUCIBLE: PropertySpec(
        id=PROPERTY_ASSEMBLY_IS_REPRODUCIBLE,
        description="同一输入两次运行**逐位相同**（装配链里没有任何未固定的随机性）",
        criterion=CRITERION_EQUALITY,
        failure="读到了未固定的随机性（时间 / 哈希序 / 全局状态），两次读数不同",
    ),
    PROPERTY_RETRIEVAL_RECALL_MEETS_FLOOR: PropertySpec(
        id=PROPERTY_RETRIEVAL_RECALL_MEETS_FLOOR,
        description="端到端检索的召回不低于 1.0（金标准文档必须出现在名单里）",
        criterion=CRITERION_LOWER_BOUND,
        failure="筛掉了该召回的那一条——检索层的三道减法里有一道落错了地方",
    ),
    PROPERTY_GROUNDING_HAS_NO_HALLUCINATION: PropertySpec(
        id=PROPERTY_GROUNDING_HAS_NO_HALLUCINATION,
        description="答案里没有幻觉引用（引用的编号都能在提示词里找到）",
        criterion=CRITERION_UPPER_BOUND,
        failure="答案写了一个不在提示词里的 [n]——它指向一份并不存在的依据",
    ),
    PROPERTY_COST_MATCHES_HAND_FORMULA: PropertySpec(
        id=PROPERTY_COST_MATCHES_HAND_FORMULA,
        description="记出来的费用与手算公式（token × 单价 / 1000）相对差不超过 1e-12",
        criterion=CRITERION_UPPER_BOUND,
        failure="价格表被读错了一档，或 token 被重复 / 漏记了一次",
    ),
    PROPERTY_DOCUMENT_COVERS_ALL_CAPABILITIES: PropertySpec(
        id=PROPERTY_DOCUMENT_COVERS_ALL_CAPABILITIES,
        description="渲染出的文档覆盖全部 8 项能力（每一个 id 都出现在正文里）",
        criterion=CRITERION_EQUALITY,
        failure="文档漏掉了某一项能力，而'没写'与'不成立'读起来一样",
    ),
}

#: 7 条性质的 id（与 :data:`PROPERTY_SPECS` 逐键对齐，供 :mod:`capstone.verify` 对名单）.
CAPSTONE_PROPERTIES: tuple[str, ...] = tuple(PROPERTY_SPECS)


def require_property(property_id: str) -> str:
    """校验一条性质 id（未知性质当场拒绝）."""
    if property_id not in PROPERTY_SPECS:
        from smart_research_agent.capstone.errors import CapabilityError

        raise CapabilityError(f"未知的性质 {property_id!r}：可选 {list(CAPSTONE_PROPERTIES)}。")
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
# 5. 十条笔记（这是"装配起来"这件事的成品）
# --------------------------------------------------------------------------------------

CAPSTONE_NOTES: dict[str, str] = {
    "capstone_is_assembly_not_rewrite": (
        "结业项目的整合**不重写任何子系统**：它把已经各自成立的能力装成一条链。"
        "重写会让你同时面对两处未知——而整合只面对一处：'接缝对不对'。"
    ),
    "capability_is_a_subpackage": (
        "一项能力要落到**子包**上，而不是落到某一行上："
        "'由哪个包负责'是可以被别人核对的问题，'由哪一行实现'不是。"
    ),
    "guard_is_before_everything": (
        "护栏排在最前，因为它是唯一一处**能省下后面所有花费**的阶段："
        "把一次攻击话术拦下来，检索与生成就一次都不会跑。"
    ),
    "plan_is_cheap_and_visible": (
        "规划只花一次模型调用，却把'这次要做什么'变成一份可读的清单——"
        "后面对不上号时，第一步要看的往往就是它。"
    ),
    "retrieval_keeps_evidence": (
        "混合检索的每一条命中都留着'哪几路召回了它、各贡献了多少'："
        "只看名单看不出'某一路空着'，而那正是最该被看见的失败。"
    ),
    "packing_owns_the_budget": (
        "打包是唯一一处能回答'上下文为什么这么短'的地方："
        "截断与丢尾各自留数字，编号只给真的进了包的那些。"
    ),
    "no_context_must_not_call_model": (
        "空上下文时**一次模型都不许调**：产物看起来最像成功的那种失败，"
        "正是一个没有片段却写得很通顺的答案。"
    ),
    "grounding_is_number_level": (
        "接地的判据是**编号**而不是字符串：写在这里的只有三组编号，"
        "因此换一版提示词、改一个字，这条性质都不会静默失效。"
    ),
    "eval_needs_a_gold_set": (
        "离线的四条指标只认识'名单 + 金标准'，因此它可以在没有模型的时候跑——"
        "而它量的是**检索**的质量，不是答案的质量。"
    ),
    "cost_and_trace_are_the_receipt": (
        "记账与追踪是同一次调用的两张收据：一张回答'花了多少'，"
        "一张回答'经历了几步'——两个问题都不该靠日志去猜。"
    ),
}

#: 十条笔记的键（顺序即写入顺序，报告里读它）.
CAPSTONE_NOTES_ORDER: tuple[str, ...] = tuple(CAPSTONE_NOTES)

# --------------------------------------------------------------------------------------
# 6. 边界（**这一课明确不承诺的事**）
# --------------------------------------------------------------------------------------

CAPSTONE_BOUNDARIES: tuple[str, ...] = (
    "本包**不新增任何第三方依赖**：装配用到的每一个子系统都是快照内既有的",
    "本包**不改动任何既有模块、既有测试、pyproject 与 docs 下的既有文件**——"
    "它只新增文件，既有子系统一行未改",
    "本包复现的是**装配与读数**，不是新的模型能力：它不重写检索器、生成器或指标",
    "本包的全部读数**离线、确定性**：用 MockLLM、固定语料与固定编码器，"
    "不联网、不读任何环境变量密钥",
    "README 与架构文档是**从清单渲染**出来的文本，本包不承诺它的文笔——"
    "它承诺的是'8 项能力每一项都出现在正文里'这条可断言的事实",
)

# --------------------------------------------------------------------------------------
# 7. 记录
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StageSpec:
    """一个阶段：次序 + 说明 + 它承担的能力 + 它读什么（**口径的一句话**）.

    ``index`` 是 1 起的连续序号（与 :data:`ASSEMBLY_STAGES` 的顺序一致）：
    端到端报告的阶段表就按它逐行印出来，"少一行"能一眼看出来。
    """

    key: str
    index: int
    description: str
    capability: str
    reader: str

    def __post_init__(self) -> None:
        from smart_research_agent.capstone.errors import ParameterError, StageError

        if self.key not in ASSEMBLY_STAGES:
            raise StageError(f"未知的阶段 {self.key!r}：可选 {list(ASSEMBLY_STAGES)}。")
        expected = ASSEMBLY_STAGES.index(self.key) + 1
        if self.index != expected:
            raise ParameterError(
                f"阶段 {self.key!r} 的序号应当是 {expected}，收到 {self.index}："
                "序号是'这一节在第几行'，错位会让报告里的阶段线读起来是乱的。"
            )
        if not self.description or not self.reader:
            raise StageError(f"阶段 {self.key!r} 必须有说明与读数口径。")
        if self.capability not in CAPABILITIES:
            raise ParameterError(
                f"阶段 {self.key!r} 的能力 {self.capability!r} 未知："
                f"可选 {list(CAPABILITY_ORDER)}。"
            )

    def to_dict(self) -> dict[str, object]:
        """摊平成一行字段."""
        return {
            "key": self.key,
            "index": self.index,
            "description": self.description,
            "capability": self.capability,
            "reader": self.reader,
        }

    def line(self) -> str:
        """一行说明：``1. [guard] 护栏：…… | 承担 input_guard | 读 flagged 条数``."""
        return (
            f"{self.index}. [{self.key}] {self.description} | "
            f"承担 {self.capability} | 读 {self.reader}"
        )


#: 9 个阶段（键与 :data:`ASSEMBLY_STAGES` 逐项对齐）.
STAGE_SPECS: dict[str, StageSpec] = {
    key: StageSpec(
        key=key,
        index=index,
        description=STAGE_DESCRIPTIONS[key],
        capability=STAGE_CAPABILITIES[key],
        reader=STAGE_READERS[key],
    )
    for index, key in enumerate(ASSEMBLY_STAGES, start=1)
}

if set(STAGE_SPECS) != set(ASSEMBLY_STAGES):  # pragma: no cover - 导入期不变式
    raise ValueError(
        "9 个阶段的两张表不一致：STAGE_SPECS 的键必须与 ASSEMBLY_STAGES 逐项对应。"
    )

#: 一份"清单之外的说明"：阶段表与能力表的交叉检查在导入期兑现.
_UNBACKED_STAGES = tuple(
    key for key, spec in STAGE_SPECS.items() if spec.capability not in CAPABILITIES
)
if _UNBACKED_STAGES:  # pragma: no cover - 导入期不变式
    raise ValueError(
        f"这些阶段没有任何能力承担：{list(_UNBACKED_STAGES)}——"
        "一个没人认领的阶段在报告里会以'跑过了'的形式出现，而它其实没有归属。"
    )


def stage_specs() -> tuple[StageSpec, ...]:
    """按次序返回 9 个阶段的口径（报告与测试共用同一份顺序）."""
    return tuple(STAGE_SPECS[key] for key in ASSEMBLY_STAGES)


def capabilities() -> tuple[Capability, ...]:
    """按次序返回 8 项能力（README 与清单共用同一份顺序）."""
    return tuple(CAPABILITIES[key] for key in CAPABILITY_ORDER)


def property_specs() -> tuple[PropertySpec, ...]:
    """按次序返回 7 条性质（verify 与报告共用同一份顺序）."""
    return tuple(PROPERTY_SPECS[key] for key in CAPSTONE_PROPERTIES)


__all__ = [
    "ASSEMBLY_STAGES",
    "CAPABILITIES",
    "CAPABILITY_CONTEXT_PACKING",
    "CAPABILITY_COST_AND_TRACING",
    "CAPABILITY_GROUNDED_GENERATION",
    "CAPABILITY_HYBRID_RETRIEVAL",
    "CAPABILITY_INPUT_GUARD",
    "CAPABILITY_OFFLINE_EVALUATION",
    "CAPABILITY_ORDER",
    "CAPABILITY_TASK_PLANNING",
    "CAPABILITY_TOOL_EXECUTION",
    "CAPSTONE_BOUNDARIES",
    "CAPSTONE_NOTES",
    "CAPSTONE_NOTES_ORDER",
    "CAPSTONE_PROPERTIES",
    "CANDIDATE_SUBPACKAGES",
    "CRITERIA",
    "CRITERION_DESCRIPTIONS",
    "CRITERION_EQUALITY",
    "CRITERION_LOWER_BOUND",
    "CRITERION_UPPER_BOUND",
    "PROPERTY_ASSEMBLY_IS_REPRODUCIBLE",
    "PROPERTY_CAPABILITIES_ARE_COVERED",
    "PROPERTY_COST_MATCHES_HAND_FORMULA",
    "PROPERTY_DOCUMENT_COVERS_ALL_CAPABILITIES",
    "PROPERTY_GROUNDING_HAS_NO_HALLUCINATION",
    "PROPERTY_RETRIEVAL_RECALL_MEETS_FLOOR",
    "PROPERTY_SPECS",
    "PROPERTY_STAGES_MATCH_SPEC",
    "STAGE_ACCOUNT",
    "STAGE_CAPABILITIES",
    "STAGE_DESCRIPTIONS",
    "STAGE_EVALUATE",
    "STAGE_GENERATE",
    "STAGE_GROUND",
    "STAGE_GUARD",
    "STAGE_PACK",
    "STAGE_PLAN",
    "STAGE_READERS",
    "STAGE_RETRIEVE",
    "STAGE_SPECS",
    "STAGE_TRACE",
    "SUBPACKAGE_DESCRIPTIONS",
    "Capability",
    "PropertySpec",
    "StageSpec",
    "capabilities",
    "property_specs",
    "require_capability",
    "require_property",
    "stage_specs",
]
