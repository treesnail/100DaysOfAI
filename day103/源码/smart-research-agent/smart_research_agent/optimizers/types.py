"""``optimizers`` 的口径表（day092 / M8-D3）.

一次性把这一课的"名词表"写全：**六个优化器 / 三种衰减 / 七条性质 / 十条笔记 /
五条边界 / 一张 PyTorch 对照表**。它们全部是常量，因此可以被测试逐键检查——
"这一步到底实现了哪几个更新规则"永远是一个可反驳的事实，而不是一段散文。

```text
六个优化器    sgd → momentum → nesterov → rmsprop → adam → adamw
              （顺序 = 状态越来越多、等价步长越来越"自适应"）
三种衰减      none → l2（耦合进梯度）→ decoupled（AdamW：不进梯度，单独缩参数）
七条性质      与 day074 逐位对账 2 条 / 新规则的性质 5 条
```

## 这一课与 day074 的边界

day074 的 ``math_foundations.optim`` 已经实现了三个更新公式（sgd / momentum / adam）、
四种学习率调度与两个裁剪原语。**本课不重写它们**：共享的三个规则一字不改地复用，
本课只补 day074 **刻意没有**的四件事——

```text
① Nesterov        动量里先"看一眼"再迈步（lookahead），与普通动量不是一回事
② RMSProp         只保留二阶动量的开方归一化（Adam 的一半）
③ AdamW           解耦权重衰减：λθ 不进梯度，而是单独把参数缩一下
④ Adam + L2       对照项：λθ 进梯度——它会被自适应分母除一下，因此与 AdamW 不同
```

因此 ``UPDATE_RULE_FAMILIES`` 把六个规则分成两档：``shared``（复用 day074）
与 ``new``（本课新增）。测试会检查这张表既不空、也不越界。
"""

from __future__ import annotations

from smart_research_agent.math_foundations.types import (
    OPTIMIZER_ADAM as MATH_ADAM,
)
from smart_research_agent.math_foundations.types import (
    OPTIMIZER_MOMENTUM as MATH_MOMENTUM,
)
from smart_research_agent.math_foundations.types import (
    OPTIMIZER_SGD as MATH_SGD,
)
from smart_research_agent.optimizers.errors import OptimizerError

# --------------------------------------------------------------------------- #
# 闭合表 1：六个优化器
# --------------------------------------------------------------------------- #

#: 复用 day074 的三个（**一字不改地复用**，本课只是把它们接进训练期回路）.
OPTIMIZER_SGD = MATH_SGD
OPTIMIZER_MOMENTUM = MATH_MOMENTUM
OPTIMIZER_ADAM = MATH_ADAM

#: 本课新增的三个（day074 刻意留白的那一半）.
OPTIMIZER_NESTEROV = "nesterov"
OPTIMIZER_RMSPROP = "rmsprop"
OPTIMIZER_ADAMW = "adamw"

#: 六个优化器（顺序 = 状态越来越多、等价步长越来越自适应）.
TRAIN_OPTIMIZERS: tuple[str, ...] = (
    OPTIMIZER_SGD,
    OPTIMIZER_MOMENTUM,
    OPTIMIZER_NESTEROV,
    OPTIMIZER_RMSPROP,
    OPTIMIZER_ADAM,
    OPTIMIZER_ADAMW,
)

#: 复用 day074 的规则（``shared``）与本课新增的规则（``new``）——两档都非空.
UPDATE_RULE_FAMILIES: dict[str, str] = {
    OPTIMIZER_SGD: "shared",
    OPTIMIZER_MOMENTUM: "shared",
    OPTIMIZER_ADAM: "shared",
    OPTIMIZER_NESTEROV: "new",
    OPTIMIZER_RMSPROP: "new",
    OPTIMIZER_ADAMW: "new",
}

#: 每种优化器的一句话解释.
OPTIMIZER_DESCRIPTIONS: dict[str, str] = {
    OPTIMIZER_SGD: "随机梯度下降：θ ← θ − lr·g（无状态，也是最诚实的基线）",
    OPTIMIZER_MOMENTUM: "动量：v ← βv + g；θ ← θ − lr·v（把历史梯度累成速度，冲过平坦区）",
    OPTIMIZER_NESTEROV: "Nesterov：先按动量迈一步再取梯度（lookahead），比普通动量更早刹车",
    OPTIMIZER_RMSPROP: "RMSProp：s ← ρs + (1−ρ)g²；θ ← θ − lr·g/(√s + ε)"
    "（只归一化尺度，不做一阶动量）",
    OPTIMIZER_ADAM: "Adam：一阶动量 m 与二阶动量 v 各带偏差修正，θ ← θ − lr·m̂/(√v̂ + ε)",
    OPTIMIZER_ADAMW: "AdamW：Adam 的解耦权重衰减版——λθ 不进梯度，而是单独把参数缩一下",
}

#: 每种优化器实际保留的状态变量（``state()`` 里会出现的东西）.
OPTIMIZER_STATE_KEYS: dict[str, tuple[str, ...]] = {
    OPTIMIZER_SGD: (),
    OPTIMIZER_MOMENTUM: ("velocity",),
    OPTIMIZER_NESTEROV: ("velocity",),
    OPTIMIZER_RMSPROP: ("square_avg",),
    OPTIMIZER_ADAM: ("first_moment", "second_moment"),
    OPTIMIZER_ADAMW: ("first_moment", "second_moment"),
}

#: 每种优化器的更新公式（**这就是可以被逐行读出来的算法**）.
OPTIMIZER_FORMULAS: dict[str, str] = {
    OPTIMIZER_SGD: "θ ← θ − lr·g",
    OPTIMIZER_MOMENTUM: "v ← β·v + g；θ ← θ − lr·v",
    OPTIMIZER_NESTEROV: "v ← β·v + g；θ ← θ − lr·(β·v + g)（用**迈步后**的梯度）",
    OPTIMIZER_RMSPROP: "s ← ρ·s + (1−ρ)·g²；θ ← θ − lr·g/(√s + ε)",
    OPTIMIZER_ADAM: "m ← β₁m + (1−β₁)g；v ← β₂v + (1−β₂)g²；m̂ = m/(1−β₁ᵗ)；"
    "v̂ = v/(1−β₂ᵗ)；θ ← θ − lr·m̂/(√v̂ + ε)",
    OPTIMIZER_ADAMW: "Adam 的 θ 更新之后，再 θ ← θ·(1 − lr·λ)（解耦衰减，**不除分母**）",
}

if not (
    set(TRAIN_OPTIMIZERS)
    == set(OPTIMIZER_DESCRIPTIONS)
    == set(OPTIMIZER_FORMULAS)
    == set(OPTIMIZER_STATE_KEYS)
    == set(UPDATE_RULE_FAMILIES)
):  # pragma: no cover - 导入期不变式
    raise OptimizerError(
        "六个优化器的五张表不一致：TRAIN_OPTIMIZERS / OPTIMIZER_DESCRIPTIONS / "
        "OPTIMIZER_FORMULAS / OPTIMIZER_STATE_KEYS / UPDATE_RULE_FAMILIES 必须逐键对齐——"
        "少一个键的那个规则在报告里只有名字、没有它实际执行的那一行更新。"
    )

if not set(UPDATE_RULE_FAMILIES.values()) <= {"shared", "new"}:  # pragma: no cover
    raise OptimizerError("UPDATE_RULE_FAMILIES 的取值只能是 'shared' 或 'new'。")

# --------------------------------------------------------------------------- #
# 闭合表 2：三种衰减
# --------------------------------------------------------------------------- #

DECAY_NONE = "none"
DECAY_L2 = "l2"
DECAY_DECOUPLED = "decoupled"

#: 三种衰减（顺序 = 不衰减 → 耦合进梯度 → 解耦缩参数）.
DECAY_MODES: tuple[str, ...] = (DECAY_NONE, DECAY_L2, DECAY_DECOUPLED)

#: 每种衰减的一句话解释.
DECAY_DESCRIPTIONS: dict[str, str] = {
    DECAY_NONE: "不衰减：参数只由梯度推动（对照组的基线）",
    DECAY_L2: "L2 权重衰减（耦合）：把 λθ 加进梯度——它会被自适应分母除一下，"
    "因此实际衰减强度随梯度尺度变化",
    DECAY_DECOUPLED: "解耦权重衰减（AdamW）：λθ **不进梯度**，而是单独把参数缩 (1−lr·λ)——"
    "衰减强度与自适应分母无关",
}

#: 每种衰减的公式.
DECAY_FORMULAS: dict[str, str] = {
    DECAY_NONE: "无附加项",
    DECAY_L2: "g ← g + λ·θ（在更新之前）",
    DECAY_DECOUPLED: "θ ← θ·(1 − lr·λ)（在更新之后）",
}

if not (set(DECAY_MODES) == set(DECAY_DESCRIPTIONS) == set(DECAY_FORMULAS)):  # pragma: no cover
    raise OptimizerError(
        "三种衰减的三张表不一致：少一个键的那种衰减会静默地没有公式。"
    )

# --------------------------------------------------------------------------- #
# 闭合表 3：七条性质
# --------------------------------------------------------------------------- #

PROPERTY_SHARED_RULES_MATCH_DAY074 = "shared_rules_match_day074"
PROPERTY_NESTEROV_IS_NOT_MOMENTUM = "nesterov_is_not_momentum"
PROPERTY_DECOUPLED_DECAY_DIFFERS_UNDER_VARYING_GRADIENTS = (
    "decoupled_decay_differs_under_varying_gradients"
)
PROPERTY_RMSPROP_EQUALIZES_SCALE = "rmsprop_equalizes_scale"
PROPERTY_CLIP_PRESERVES_DIRECTION = "clip_preserves_direction"
PROPERTY_CLIPPING_MATCHES_DAY074 = "clipping_matches_day074"
PROPERTY_SCHEDULE_MATCHES_DAY074 = "schedule_matches_day074"

#: 七条性质（顺序 = 从"复用对不对"到"新增对不对"）.
OPTIMIZER_PROPERTIES: tuple[str, ...] = (
    PROPERTY_SHARED_RULES_MATCH_DAY074,
    PROPERTY_NESTEROV_IS_NOT_MOMENTUM,
    PROPERTY_DECOUPLED_DECAY_DIFFERS_UNDER_VARYING_GRADIENTS,
    PROPERTY_RMSPROP_EQUALIZES_SCALE,
    PROPERTY_CLIP_PRESERVES_DIRECTION,
    PROPERTY_CLIPPING_MATCHES_DAY074,
    PROPERTY_SCHEDULE_MATCHES_DAY074,
)

#: 每条性质在讲什么（一句话）.
PROPERTY_DESCRIPTIONS: dict[str, str] = {
    PROPERTY_SHARED_RULES_MATCH_DAY074: "共享的三个规则（sgd/momentum/adam）逐步与 day074 逐位一致",
    PROPERTY_NESTEROV_IS_NOT_MOMENTUM: "Nesterov 与普通动量在同一轨迹上给出不同的一步",
    PROPERTY_DECOUPLED_DECAY_DIFFERS_UNDER_VARYING_GRADIENTS: (
        "梯度尺度变化时，AdamW 与 Adam+L2 的轨迹显著不同（解耦不是摆设）"
    ),
    PROPERTY_RMSPROP_EQUALIZES_SCALE: "RMSProp 把相差 100 倍的梯度归一到几乎相同的等效步长",
    PROPERTY_CLIP_PRESERVES_DIRECTION: "整体范数裁剪只改长度、不改方向",
    PROPERTY_CLIPPING_MATCHES_DAY074: "本课的裁剪与 day074 的 clip_by_global_norm 返回同一个缩放系数",
    PROPERTY_SCHEDULE_MATCHES_DAY074: "本课接线的热身+余弦与 day074 的 warmup_cosine_schedule 逐点一致",
}

#: 每条性质"失败意味着什么"（**不通过时要去看哪里**）.
PROPERTY_FAILURE: dict[str, str] = {
    PROPERTY_SHARED_RULES_MATCH_DAY074: "共享规则被改动了——同一条公式写了两遍，",
    PROPERTY_NESTEROV_IS_NOT_MOMENTUM: "lookahead 项被漏掉，Nesterov 退化成普通动量",
    PROPERTY_DECOUPLED_DECAY_DIFFERS_UNDER_VARYING_GRADIENTS: "衰减被加进了梯度（耦合），AdamW 名不副实",
    PROPERTY_RMSPROP_EQUALIZES_SCALE: "分母少了开方或 ε，尺度没有被真正归一",
    PROPERTY_CLIP_PRESERVES_DIRECTION: "裁剪写成了逐分量（clip_by_value），方向被改了",
    PROPERTY_CLIPPING_MATCHES_DAY074: "裁剪的阈值判据（<= 还是 <）与 day074 分家",
    PROPERTY_SCHEDULE_MATCHES_DAY074: "热身的计步口径（1-based）与 day074 分家",
}

if not (
    set(OPTIMIZER_PROPERTIES) == set(PROPERTY_DESCRIPTIONS) == set(PROPERTY_FAILURE)
):  # pragma: no cover
    raise OptimizerError(
        "七条性质的三张表不一致：名单 / 说明 / 失败意味着什么必须逐键对齐。"
    )

# --------------------------------------------------------------------------- #
# 闭合表 4：十条笔记 / 五条边界 / 一张 PyTorch 对照表
# --------------------------------------------------------------------------- #

OPTIMIZER_NOTES_ORDER: tuple[str, ...] = (
    "direction",
    "distance",
    "momentum_form",
    "nesterov_lookahead",
    "rmsprop_half_adam",
    "adam_bias",
    "adamw_decoupling",
    "global_norm",
    "schedule_rise_fall",
    "state_reset",
)

#: 十条笔记（键 -> 一句话）.
OPTIMIZER_NOTES: dict[str, str] = {
    "direction": "方向由梯度给（day090），本课只决定**走多远**——两者是两件事，不能一起调。",
    "distance": "一步 = 学习率 × 更新规则。学习率是标量，更新规则决定这个标量作用在什么上。",
    "momentum_form": "动量有累加型（v += g）与 EMA 型（v = βv + (1−β)g）两种写法；"
    "本仓库统一用累加型，换写法等价于把 lr 除以 (1−β)，混用会让'换了优化器'变成'重新调 lr'。",
    "nesterov_lookahead": "Nesterov 不是'换个动量系数'：它取的是**迈步之后**那一点的梯度，"
    "因此在快要过冲时会比普通动量更早刹车。",
    "rmsprop_half_adam": "RMSProp 是 Adam 的一半：只有二阶动量，没有一阶动量。"
    "因此它在梯度方向来回变时会抖，而 Adam 不会。",
    "adam_bias": "Adam 的两个滑动平均从 0 出发，第一步只有真值的 1−β；"
    "不做偏差修正时第一步会大 3 倍多，而它看起来只是'前期有点抖'。",
    "adamw_decoupling": "AdamW 的 λθ **不进梯度**：耦合版里 λθ 会被 √v̂ 除一下，"
    "于是梯度大的参数衰减得更弱——这不是我们想要的'人人等量衰减'。",
    "global_norm": "裁剪用**整体范数**（所有参数共享一个尺度），逐分量裁剪会改变方向。",
    "schedule_rise_fall": "学习率曲线先升后降：前期热身避免打乱随机初始化，后期退火避免在谷底横跳。",
    "state_reset": "换任务换形状前必须 reset()：上一段的动量被带进新任务的前几步，"
    "表现为'换了任务之后前几步莫名地大'。",
}

#: 五条边界（**这一课明确不承诺的事**）.
OPTIMIZER_BOUNDARIES: tuple[str, ...] = (
    "只做一阶方法：AdamW / RMSProp / Nesterov 都是一阶，不涉及二阶方法（Newton / L-BFGS）。",
    "不做参数分组：所有参数共享一个学习率与一个衰减系数（分层学习率是工程问题，不是数学问题）。",
    "不做分布式/混合精度：梯度累积、ZeRO、bf16 优化器状态都不在本课范围。",
    "不做收敛证明：本课只给出确定性的对比读数，不给任何优化器的收敛性定理。",
    "不新增第三方依赖：全部纯标准库实现，不 import torch / numpy。",
)

#: 纯 Python ↔ PyTorch 对照表（**只核对语义，本仓库不安装也不调用 torch**）.
TORCH_COUNTERPARTS: dict[str, str] = {
    OPTIMIZER_SGD: "torch.optim.SGD(params, lr)（weight_decay 即 L2 耦合衰减）",
    OPTIMIZER_MOMENTUM: "torch.optim.SGD(params, lr, momentum=β)",
    OPTIMIZER_NESTEROV: "torch.optim.SGD(params, lr, momentum=β, nesterov=True)",
    OPTIMIZER_RMSPROP: "torch.optim.RMSprop(params, lr, alpha=ρ, eps=ε)",
    OPTIMIZER_ADAM: "torch.optim.Adam(params, lr, betas=(β₁, β₂), eps=ε)",
    OPTIMIZER_ADAMW: "torch.optim.AdamW(params, lr, weight_decay=λ)",
    "decoupled_weight_decay": "optimizer 构造里的 weight_decay（AdamW 才是解耦的）",
    "global_norm_clip": "torch.nn.utils.clip_grad_norm_(params, max_norm)",
    "lr_scheduler": "torch.optim.lr_scheduler.LambdaLR / CosineAnnealingLR / OneCycleLR",
    "zero_grad": "optimizer.zero_grad(set_to_none=True)",
}

#: 这一课的性质名单（供报告核对：表里的行数必须等于它）.
PROPERTIES = OPTIMIZER_PROPERTIES

__all__ = [
    "DECAY_DECOUPLED",
    "DECAY_DESCRIPTIONS",
    "DECAY_FORMULAS",
    "DECAY_L2",
    "DECAY_MODES",
    "DECAY_NONE",
    "OPTIMIZER_ADAM",
    "OPTIMIZER_ADAMW",
    "OPTIMIZER_BOUNDARIES",
    "OPTIMIZER_DESCRIPTIONS",
    "OPTIMIZER_FORMULAS",
    "OPTIMIZER_MOMENTUM",
    "OPTIMIZER_NESTEROV",
    "OPTIMIZER_NOTES",
    "OPTIMIZER_NOTES_ORDER",
    "OPTIMIZER_PROPERTIES",
    "OPTIMIZER_RMSPROP",
    "OPTIMIZER_SGD",
    "OPTIMIZER_STATE_KEYS",
    "PROPERTIES",
    "PROPERTY_CLIPPING_MATCHES_DAY074",
    "PROPERTY_CLIP_PRESERVES_DIRECTION",
    "PROPERTY_DECOUPLED_DECAY_DIFFERS_UNDER_VARYING_GRADIENTS",
    "PROPERTY_DESCRIPTIONS",
    "PROPERTY_FAILURE",
    "PROPERTY_NESTEROV_IS_NOT_MOMENTUM",
    "PROPERTY_RMSPROP_EQUALIZES_SCALE",
    "PROPERTY_SCHEDULE_MATCHES_DAY074",
    "PROPERTY_SHARED_RULES_MATCH_DAY074",
    "TORCH_COUNTERPARTS",
    "TRAIN_OPTIMIZERS",
    "UPDATE_RULE_FAMILIES",
]
