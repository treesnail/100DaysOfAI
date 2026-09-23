"""配比控制：给"某一组样本占了多少"装上一条可执行的护栏（M5-D8）.

day049 的复习里留了一个**光说了没做**的结论：

> 红队安全样本 16 条占了 ``16/37 = 43.2%``。直接丢进 SFT，模型会倾向
> "多拒答"。正确做法不是删安全样本，而是**调权或分阶段训练**。

这句结论在当时是对的（也确实不能靠"删样本"解决），但它缺一件东西：
**一条能报警的线**。43.2% 是人工算出来的，下一个批次变成 55% 时
没有人会被通知。本模块补的就是这条线——把"某一组不超过总体的某个比例"
写成可以被断言、可以被复现、可以在报告里被检查的约束。

## 两种口径：上限（``max_ratio``）与目标缺口（``deficit_report``）

- ``plan_mixing``：**上限口径**。任一组占比超过 ``max_ratio`` 就按配额削减。
  它是一道安全网——缺省 0.5 的含义是"没有任何单一来源能超过一半"。
- ``deficit_report``：**目标口径**。给定各组的相对权重，算出"谁多了、谁少了"。
  它本身不裁剪任何样本，只回答"要真正补齐配比，还差多少条"——
  而这个数字正是``augment`` 的靶子。**先算缺口、再做增强**，
  比"随便增强一批再看配比"靠谱。

## 一个必须写清楚的数学事实：上限约束会连锁收紧

给"任一组不超过 40%"这条约束，直觉答案是"把超过的组砍到 40%"。
**这是错的**：砍掉一组之后总数变小，其余组的占比反而变大，
于是必须重算。本课数据集（清洗后 37 条，来源分布 16 : 5 : 16）上，
这个不动点迭代跑了 **6 轮**：

```text
16 : 5 : 16  （T=37，cap=14）→  14 : 5 : 14  （T=33，cap=13）
  →  13 : 5 : 13  （T=31，cap=12）→  12 : 5 : 12  （T=29，cap=11）
  →  11 : 5 : 11  （T=27，cap=10）→  10 : 5 : 10  （T=25，cap=10）收敛
```

最终 25 条，两个大组各占 ``10/25 = 40.0%``。**代价是丢掉 12 条样本
（32%）**——这就是"为什么 day049 说该用调权/分阶段训练而不是硬删"的
定量答案：硬删的账太贵了。上限口径因此适合当**安全网**（防垄断），
不适合当**日常配比手段**。

## 一个必须存在但很容易漏的字段：``feasible``

离散样本下"任一组不超过 X%"**未必可达**。同一个数据集换成
``group_by="safety"`` 口径（21 组正常 : 16 条安全样本），再收 0.4：

```text
21 : 16  →  14 : 14  →  11 : 11  →  8 : 8  →  6 : 6
       →  4 : 4  →  3 : 3  →  2 : 2  →  1 : 1 （触发 MIN_GROUP_KEEP 保底）
```

**9 轮之后剩下 2 条**，两组各占 50%——约束根本达不到（两组各占一半时，
必然有一组 ≥ 50%）。此时 ``MixPlan.feasible`` 为 ``False`` 并附带一条
告警。一个悄悄违反自己约束的护栏，比没有护栏更糟：它会让评审以为
"配比已经守住 40% 了"。**"这个目标做不到"本身就是一个必须被输出的结论。**

收敛性可以证明：每一轮里只要存在超额组，就把它精确压到 ``cap``，
而 ``n > cap`` 意味着总数**严格下降**；总数是正整数且有下界，
因此一定在有限步内收敛。**收敛不等于可达**——上面那个例子收敛了，
但收敛到的点违反约束。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from smart_research_agent.domain_data.augment import is_augmented
from smart_research_agent.domain_data.errors import DomainDataError
from smart_research_agent.domain_data.quality import quality_scores
from smart_research_agent.finetune.schema import TrainingExample

#: 三种分组口径
GROUP_BY_SOURCE = "source"
GROUP_BY_ORIGIN = "origin"
GROUP_BY_SAFETY = "safety"

#: 可选分组口径全集（新增口径时同步改 ``group_key``）
GROUP_BY_CHOICES: tuple[str, ...] = (GROUP_BY_SOURCE, GROUP_BY_ORIGIN, GROUP_BY_SAFETY)

#: 缺省分组口径：按来源。来源是唯一"一定存在、且一定单一"的治理字段。
DEFAULT_GROUP_BY = GROUP_BY_SOURCE

#: 缺省上限：任一组不超过一半。取 0.5 而不是更紧的值，是因为
#: 上限约束的代价是**连锁削减**（见模块开头），缺省值应该是"安全网"
#: 而不是"日常手段"——本课程数据集在这个缺省下不丢任何样本。
DEFAULT_MAX_GROUP_RATIO = 0.5

#: 每组至少保留 1 条。没有这条保底，``int(max_ratio * total)`` 在小样本上
#: 会算出 0，把整组清空——而"某一组样本被清空"从来不是配比控制的意图。
MIN_GROUP_KEEP = 1

#: 不动点迭代的上限。理论上一定收敛（见模块开头），这里只是防御死循环。
MAX_MIX_ITERATIONS = 1000

#: 浮点误差保护：``int(0.3 * 10)`` 在 IEEE 754 下是 ``2``（``0.3*10`` 得到
#: ``2.9999999999999996``），而正确答案是 3。差一条样本在配比这种
#: "刚好卡在边界"的场景里足以让结论翻转，所以取整前统一加一个极小量。
_FLOOR_EPSILON = 1e-9

#: ``origin`` 口径的两个组名
ORIGIN_ORIGINAL = "original"
ORIGIN_AUGMENTED = "augmented"

#: ``safety`` 口径的两个组名
GROUP_SAFETY = "safety"
GROUP_NORMAL = "normal"

#: 无来源样本的兜底组名。它出现即说明数据治理有问题，
#: 但**不能丢弃**——把它归到一个显式命名的组里，问题才会出现在报表上。
GROUP_UNKNOWN = "(unknown)"


def group_key(example: TrainingExample, *, group_by: str = DEFAULT_GROUP_BY) -> str:
    """一条样本落在哪个配比组里（未知口径抛 ``DomainDataError``）.

    三种口径的设计取舍：``source`` 是治理字段，``origin`` 区分原始与增强
    （**增强样本不能无声地把配比撑歪**），``safety`` 直接对应 day049 那条
    "安全样本占 43.2%"的观察。三者都用**单个**分组，因此"占比之和为 1"
    这条恒等式总是成立——多标签分组会让一个样本同时算进两组，
    占比之和超过 100%，任何基于占比的判断都会失效。
    """
    if group_by == GROUP_BY_SOURCE:
        return example.source or GROUP_UNKNOWN
    if group_by == GROUP_BY_ORIGIN:
        return ORIGIN_AUGMENTED if is_augmented(example) else ORIGIN_ORIGINAL
    if group_by == GROUP_BY_SAFETY:
        return GROUP_SAFETY if example.is_safety_example() else GROUP_NORMAL
    raise DomainDataError(
        f"未知的分组口径 {group_by!r}，可选：{', '.join(GROUP_BY_CHOICES)}"
    )


def group_counts(
    examples: Sequence[TrainingExample], *, group_by: str = DEFAULT_GROUP_BY
) -> dict[str, int]:
    """按分组口径统计各组的样本数（键顺序 = 首次出现的顺序，可复现）."""
    counts: dict[str, int] = {}
    for example in examples:
        key = group_key(example, group_by=group_by)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _floor_ratio(max_ratio: float, total: int) -> int:
    """``floor(max_ratio * total)``，带浮点误差保护与每组保底.

    这个函数存在的唯一理由是那条 ``int(0.3 * 10) == 2`` 的陷阱。
    写在一处而不是散在循环里，是为了让"取整口径"只有一个出处。
    """
    return max(MIN_GROUP_KEEP, math.floor(max_ratio * total + _FLOOR_EPSILON))


@dataclass
class MixPlan:
    """配比方案：每组的保留上限 + 迭代过程 + 是否真的可达.

    ``feasible`` 是一个**必须存在**的字段。离散样本下"任一组不超过 X%"
    未必可达：两组各 1 条时，无论怎么砍都会有一组占 50%，取 ``max_ratio=0.4``
    就无解。此时正确的做法不是假装成功，而是把结论标出来——
    一个悄悄违反自己约束的护栏，比没有护栏更糟。
    """

    group_by: str
    max_ratio: float
    counts: dict[str, int]
    quotas: dict[str, int]
    iterations: int
    warnings: list[str] = field(default_factory=list)

    @property
    def total_in(self) -> int:
        """进配比之前的样本数."""
        return sum(self.counts.values())

    @property
    def total_out(self) -> int:
        """按本方案保留的样本数."""
        return sum(self.quotas.values())

    @property
    def dropped(self) -> int:
        """被削减掉的样本数."""
        return self.total_in - self.total_out

    def final_ratios(self) -> dict[str, float]:
        """削减后各组的实际占比（空集返回空字典）."""
        total = self.total_out
        if not total:
            return {}
        return {name: quota / total for name, quota in self.quotas.items()}

    @property
    def feasible(self) -> bool:
        """削减后的实际占比是否真的都落在 ``max_ratio`` 以内."""
        ratios = self.final_ratios().values()
        return all(ratio <= self.max_ratio + _FLOOR_EPSILON for ratio in ratios)

    def to_dict(self) -> dict:
        """投影为可直接 json.dumps 的字典."""
        return {
            "group_by": self.group_by,
            "max_ratio": self.max_ratio,
            "counts": dict(self.counts),
            "quotas": dict(self.quotas),
            "total_in": self.total_in,
            "total_out": self.total_out,
            "dropped": self.dropped,
            "iterations": self.iterations,
            "feasible": self.feasible,
            "final_ratios": {
                name: round(value, 4) for name, value in self.final_ratios().items()
            },
            "warnings": list(self.warnings),
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要（demo 与日志直接打印）."""
        return (
            f"配比：{self.group_by} 口径 / 上限 {self.max_ratio} | "
            f"{self.total_in} 条 → {self.total_out} 条（削 {self.dropped}）| "
            f"迭代 {self.iterations} 轮 | 可达={self.feasible}"
        )


def plan_mixing(
    examples: Sequence[TrainingExample],
    *,
    group_by: str = DEFAULT_GROUP_BY,
    max_ratio: float = DEFAULT_MAX_GROUP_RATIO,
) -> MixPlan:
    """算出配比方案（上限口径，不动点迭代到收敛）.

    迭代过程见模块开头：每一轮把超额组压到 ``floor(max_ratio × 当前总数)``，
    直到没有超额组。返回的 ``quotas`` 就是最终的每组保留上限。
    """
    if not 0.0 < max_ratio <= 1.0:
        raise DomainDataError(f"配比上限必须落在 (0, 1] 区间，收到 {max_ratio}")

    counts = group_counts(examples, group_by=group_by)
    warnings: list[str] = []
    alive = dict(counts)
    iterations = 0

    if len(counts) < 2:
        # 只有 0/1 个组时"某组不超过 X%"要么无意义（一组占 100%），
        # 要么根本不可达。直接放行并告警，不做无意义的迭代。
        warnings.append("样本只落在一个分组里，配比上限无意义（不裁剪任何样本）")
    else:
        while iterations < MAX_MIX_ITERATIONS:
            iterations += 1
            total = sum(alive.values())
            if total <= 0:
                break
            cap = _floor_ratio(max_ratio, total)
            over = [name for name, count in alive.items() if count > cap]
            if not over:
                break
            for name in over:
                alive[name] = cap
        if iterations >= MAX_MIX_ITERATIONS:
            warnings.append(f"迭代达到上限 {MAX_MIX_ITERATIONS} 次仍未收敛")

    plan = MixPlan(
        group_by=group_by,
        max_ratio=max_ratio,
        counts=counts,
        quotas=alive,
        iterations=iterations,
        warnings=warnings,
    )
    if not plan.feasible:
        # 保底（MIN_GROUP_KEEP）与取整会让约束在极端分布下不可达。
        # 说出来，而不是让它悄悄失效。
        plan.warnings.append(
            f"实际占比无法全部落在 {max_ratio} 以内（每组至少保留 "
            f"{MIN_GROUP_KEEP} 条与取整所致），最终占比 {plan.final_ratios()}"
        )
    return plan


@dataclass
class MixingReport:
    """配比执行报告：进/出条数、逐组明细、迭代轮数与告警."""

    group_by: str
    max_ratio: float
    total_in: int
    total_out: int
    by_group_in: dict[str, int]
    by_group_out: dict[str, int]
    iterations: int
    feasible: bool
    warnings: list[str] = field(default_factory=list)

    @property
    def dropped(self) -> int:
        """被削减掉的样本数."""
        return self.total_in - self.total_out

    def dropped_by_group(self) -> dict[str, int]:
        """逐组的削减数（只列出非零项）."""
        return {
            name: self.by_group_in.get(name, 0) - self.by_group_out.get(name, 0)
            for name in self.by_group_in
            if self.by_group_in.get(name, 0) - self.by_group_out.get(name, 0)
        }

    def final_ratios(self) -> dict[str, float]:
        """削减后各组的实际占比."""
        if not self.total_out:
            return {}
        return {
            name: count / self.total_out for name, count in self.by_group_out.items()
        }

    def to_dict(self) -> dict:
        """投影为可直接 json.dumps 的字典."""
        return {
            "group_by": self.group_by,
            "max_ratio": self.max_ratio,
            "total_in": self.total_in,
            "total_out": self.total_out,
            "dropped": self.dropped,
            "iterations": self.iterations,
            "feasible": self.feasible,
            "by_group_in": dict(self.by_group_in),
            "by_group_out": dict(self.by_group_out),
            "dropped_by_group": self.dropped_by_group(),
            "final_ratios": {
                name: round(value, 4) for name, value in self.final_ratios().items()
            },
            "warnings": list(self.warnings),
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要（demo 与日志直接打印）."""
        return (
            f"配比执行：{self.total_in} 条 → {self.total_out} 条 | "
            f"削减 {self.dropped_by_group() or '无'} | 迭代 {self.iterations} 轮"
        )


def apply_mix_plan(
    examples: Sequence[TrainingExample],
    plan: MixPlan,
    *,
    scores: Sequence[float] | None = None,
) -> tuple[list[TrainingExample], MixingReport]:
    """按方案削减样本，返回（保留样本, 报告）.

    **组内保留谁**由一个刻意的规则决定：

    - 给了 ``scores``（通常来自 ``quality.quality_scores``）→ 组内按分数
      **降序**保留，同分时保持原顺序（``sorted`` 稳定）；
    - 没给 → 按原顺序保留前 ``quota`` 条。

    两种都确定可复现。给分数的那条路是配比与质量两个模块**唯一的接口**，
    也是"削减时该砍谁"的答案：砍分数最低的，而不是砍排在最后的。
    没给分数时按顺序保留，是为了让"只想看看配比怎么变"的调用方
    不必先跑一遍质量打分。
    """
    scores_list = list(scores) if scores is not None else None
    if scores_list is not None and len(scores_list) != len(examples):
        raise DomainDataError(
            f"scores 长度（{len(scores_list)}）必须与 examples（{len(examples)}）一致"
        )

    buckets: dict[str, list[int]] = {}
    for index, example in enumerate(examples):
        key = group_key(example, group_by=plan.group_by)
        buckets.setdefault(key, []).append(index)

    kept_indices: list[int] = []
    by_group_out: dict[str, int] = {}
    for name, indices in buckets.items():
        quota = plan.quotas.get(name, 0)
        if scores_list is not None and len(indices) > quota:
            ordered = sorted(indices, key=lambda item: (-scores_list[item], item))
            selected = ordered[:quota]
        else:
            selected = indices[:quota]
        kept_indices.extend(selected)
        by_group_out[name] = len(selected)

    # 恢复入参相对顺序：削减后的数据集仍然"看起来像原来那份的子集"，
    # 逐条对照时不需要在脑子里重排。
    kept_indices.sort()
    kept = [examples[index] for index in kept_indices]

    report = MixingReport(
        group_by=plan.group_by,
        max_ratio=plan.max_ratio,
        total_in=len(examples),
        total_out=len(kept),
        by_group_in=dict(plan.counts),
        by_group_out=by_group_out,
        iterations=plan.iterations,
        feasible=plan.feasible,
        warnings=list(plan.warnings),
    )
    return kept, report


@dataclass
class MixDeficit:
    """目标配比下的缺口报告：谁超额、谁不足、各差多少条.

    ``targets`` 用**最大余数法**分配（先取整、再按小数部分从大到小补足
    余数），因此 ``sum(targets) == sum(counts)`` 恒成立——目标数之和必须
    等于实际总数，否则"缺口"就变成了"凭空要变出样本"。
    """

    group_by: str
    counts: dict[str, int]
    targets: dict[str, int]
    weights: dict[str, float]

    @property
    def total(self) -> int:
        """样本总数."""
        return sum(self.counts.values())

    def deficits(self) -> dict[str, int]:
        """逐组缺口：正数 = 不足（需要补），负数 = 超额（需要削）."""
        return {name: self.targets[name] - self.counts[name] for name in self.counts}

    def under(self) -> dict[str, int]:
        """不足的组及其缺口（只列正数）."""
        return {name: gap for name, gap in self.deficits().items() if gap > 0}

    def over(self) -> dict[str, int]:
        """超额的组及其超出量（只列正数）."""
        return {name: -gap for name, gap in self.deficits().items() if gap < 0}

    def to_dict(self) -> dict:
        """投影为可直接 json.dumps 的字典."""
        return {
            "group_by": self.group_by,
            "weights": dict(self.weights),
            "counts": dict(self.counts),
            "targets": dict(self.targets),
            "deficits": self.deficits(),
            "under": self.under(),
            "over": self.over(),
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要（demo 与日志直接打印）."""
        return (
            f"配比缺口：{self.group_by} 口径 | 目标 {self.targets} | "
            f"不足 {self.under() or '无'} | 超额 {self.over() or '无'}"
        )


def deficit_report(
    examples: Sequence[TrainingExample],
    *,
    group_by: str = DEFAULT_GROUP_BY,
    weights: dict[str, float],
) -> MixDeficit:
    """按目标权重算配比缺口（**不裁剪任何样本**）.

    ``weights`` 必须覆盖实际出现的**全部**组：漏掉一个组的权重，
    就等于宣称"这一组的目标数是 0"，而它多半只是被忘了。
    安静地把一整组判成 0，是这类接口最危险的一种默认行为，所以这里直接拒绝。
    """
    counts = group_counts(examples, group_by=group_by)
    missing = [name for name in counts if name not in weights]
    if missing:
        raise DomainDataError(
            f"权重表缺少分组 {missing}，必须覆盖全部实际分组：{sorted(counts)}"
        )
    if any(value <= 0 for value in weights.values()):
        raise DomainDataError("目标权重必须为正数（权重为 0 的组无法被表达，请直接从表中删掉）")

    total = sum(counts.values())
    active = {name: weights[name] for name in counts}
    weight_sum = sum(active.values())
    targets: dict[str, int] = {}
    for name in counts:
        targets[name] = math.floor(total * active[name] / weight_sum + _FLOOR_EPSILON)
    # 最大余数法补齐取整损失：按小数部分从大到小发剩下的名额，
    # 小数部分相同时按组名升序（保证同一份数据永远得到同一份目标）。
    remainder = total - sum(targets.values())
    if remainder > 0:
        order = sorted(
            counts,
            key=lambda name: (
                -(total * active[name] / weight_sum - targets[name]),
                name,
            ),
        )
        for name in order[:remainder]:
            targets[name] += 1
    return MixDeficit(
        group_by=group_by, counts=counts, targets=targets, weights=dict(active)
    )


def mixing_scores(
    examples: Sequence[TrainingExample], *, weights=None
) -> list[float]:
    """配比削减用的排序键（``quality.quality_scores`` 的别名）.

    保留这个薄别名，是为了让 ``pipeline`` 的调用处读起来是
    "按质量分削减"，而不是"调用 quality 模块"——两者语义不同：
    前者是配比的策略，后者是打分的实现。
    """
    return quality_scores(examples, weights=weights)
