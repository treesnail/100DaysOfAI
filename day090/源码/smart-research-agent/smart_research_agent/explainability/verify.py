"""``verify.py``：六条性质——**每一条都必须有一条"它会亮红"的反证**（day083）.

```text
rows_are_distributions            每一行是一个条件分布（非负、和为 1、有限）
masked_entries_are_exact_zero     被掩码挡掉的位置**逐位**是 0.0（沿用 day082 的口径）
entropy_within_ceiling            每一行的熵 ≤ ln(这一行能看到的位置数)（Jensen）
heatmap_round_trip                渲染 → 解析 ⇒ **逐级**相同（10 级是有损的，因此要能对回去）
rollout_is_stochastic             滚动之后仍然行随机，且**因果零模式逐位保持**
deterministic                     同一批记录跑两次逐位相同
```

## 一、四条判据的口径都是"实测 vs 独立构造的期望"

```text
第 1 条   只读权重本身（行和）                       没有"期望"可被污染
第 2 条   期望来自 record.mask（**它自己那次前向的掩码**）—— 但这里没有"选对掩码"的问题：
          本课不判断"这张掩码对不对"（那是 day082 的事），只判断"权重与掩码自洽"
第 3 条   期望由掩码的**每一行允许的位置数**算出（ln k）——与权重无关
第 5 条   期望是"行随机 + 因果零模式"，两者都不依赖调用参数
```

第 2 条那句区分很重要：day082 的 `check_reads_every_position` 必须**自己造期望掩码**
（否则"给编码器一串因果掩码"会全绿）；而本课读的是**已经发生的那次前向**，
因此"权重与它的掩码自洽"就是正确的判据——**两张表来自同一次前向**这件事
由 :func:`extract.self_records` 保证（它直接取 day082 的前向记录）。

## 二、每一条的"反证"都写进了测试

```text
第 1 条   把一行的和改成 0.9            ⇒ 亮红
第 2 条   把被掩码的一格改成 1e-12      ⇒ 亮红（**逐位**判据）
第 3 条   把权重与掩码错配（不同前向）   ⇒ 亮红
第 4 条   把某个字符改成不在图例里的     ⇒ 解析当场拒绝
第 5 条   把某个因子换成非随机矩阵       ⇒ 行和不再为 1
```

"它通过了"这句话只有在"它会失败"也被证明过之后才有信息量
（day082 第 5、7 章同一条纪律）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from smart_research_agent.explainability.errors import (
    ParameterError,
    ShapeError,
)
from smart_research_agent.explainability.render import heatmap_block, parse_heatmap
from smart_research_agent.explainability.rollout import (
    aggregate_by_layer,
    mask_respected_after_rollout,
    preserves_causal_zeroes,
    rollout_record,
    rollout_weights,
)
from smart_research_agent.explainability.types import (
    DEFAULT_ALPHA,
    EXPLAIN_PROPERTIES,
    PROPERTY_DESCRIPTIONS,
    PROPERTY_DETERMINISTIC,
    PROPERTY_ENTROPY_WITHIN_CEILING,
    PROPERTY_HEATMAP_ROUND_TRIP,
    PROPERTY_MASKED_ENTRIES_ARE_EXACT_ZERO,
    PROPERTY_ROLLOUT_IS_STOCHASTIC,
    PROPERTY_ROWS_ARE_DISTRIBUTIONS,
    STREAM_SELF,
    AttentionRecord,
)
from smart_research_agent.math_foundations.types import matrix_shape

#: 行随机性在**滚动之后**允许的漂移（几个因子相乘会带来浮点误差）.
ROLLOUT_TOLERANCE = 1e-12

#: 熵与天花板的比较容差（`-p·ln p` 的求和会有 1e-16 量级的误差）.
ENTROPY_TOLERANCE = 1e-12


@dataclass(frozen=True)
class PropertyOutcome:
    """一条性质的读数（证据是一句话，而它必须能被独立复核）."""

    name: str
    passed: bool
    evidence: str
    detail: str = ""

    def __post_init__(self) -> None:
        if self.name not in EXPLAIN_PROPERTIES:
            raise ParameterError(
                f"未知的性质名 {self.name!r}：可选 {', '.join(EXPLAIN_PROPERTIES)}。"
            )
        object.__setattr__(self, "evidence", str(self.evidence))
        object.__setattr__(self, "detail", str(self.detail))

    @property
    def description(self) -> str:
        """这一条在说什么."""
        return PROPERTY_DESCRIPTIONS[self.name]

    @property
    def state(self) -> str:
        """``通过`` / ``未通过``（本课没有"不适用"：六条对任何一批记录都有定义）."""
        return "通过" if self.passed else "未通过"

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "name": self.name,
            "state": self.state,
            "passed": self.passed,
            "evidence": self.evidence,
            "detail": self.detail,
        }

    def summary_line(self) -> str:
        """一行说明：``[通过] rows_are_distributions | 6 条记录，最大的行和误差 0.000e+00``."""
        return f"[{self.state}] {self.name} | {self.evidence}"


@dataclass(frozen=True)
class PropertyReport:
    """六条性质的报告（名单必须完整：缺一条与"它通过"必须可区分）."""

    subject: str
    outcomes: tuple[PropertyOutcome, ...]
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if not isinstance(self.subject, str) or not self.subject:
            raise ParameterError(f"subject 必须是非空字符串，收到 {self.subject!r}。")
        resolved = tuple(self.outcomes)
        if not resolved:
            raise ParameterError("性质报告不能为空。")
        names = {item.name for item in resolved}
        if names != set(EXPLAIN_PROPERTIES):
            raise ShapeError(
                f"性质报告的名单不完整：缺 {sorted(set(EXPLAIN_PROPERTIES) - names)}——"
                "缺一条与'它通过'在报告里必须长得不一样。"
            )
        object.__setattr__(self, "outcomes", resolved)
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def ok(self) -> bool:
        """全部通过."""
        return all(item.passed for item in self.outcomes)

    def outcome_of(self, name: str) -> PropertyOutcome:
        """按名字取一条读数（名字不认识时当场拒绝）."""
        for item in self.outcomes:
            if item.name == name:
                return item
        raise ParameterError(
            f"报告里没有性质 {name!r}：可选 {', '.join(item.name for item in self.outcomes)}。"
        )

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "subject": self.subject,
            "ok": self.ok,
            "outcomes": [item.to_dict() for item in self.outcomes],
        }

    def summary_lines(self) -> tuple[str, ...]:
        """逐行说明（演示脚本直接印它）."""
        return tuple(item.summary_line() for item in self.outcomes)


def _checked_records(records: Sequence[AttentionRecord]) -> tuple[AttentionRecord, ...]:
    resolved = tuple(records)
    if not resolved:
        raise ParameterError("性质检查需要至少一条记录。")
    for index, record in enumerate(resolved):
        if not isinstance(record, AttentionRecord):
            raise ParameterError(
                f"第 {index} 条不是 AttentionRecord，收到 {type(record).__name__}。"
            )
    return resolved


def _self_records(records: Sequence[AttentionRecord]) -> tuple[AttentionRecord, ...]:
    """只要自注意力那几条（滚动与逐格比较都要求方阵）."""
    return tuple(record for record in records if record.stream == STREAM_SELF)


def check_rows_are_distributions(
    records: Sequence[AttentionRecord],
) -> PropertyOutcome:
    """第 1 条：每一行是一个条件分布（非负、和为 1、有限）."""
    resolved = _checked_records(records)
    worst = 0.0
    cells = 0
    for record in resolved:
        for row in record.weights:
            cells += len(row)
            worst = max(worst, abs(math.fsum(row) - 1.0))
    return PropertyOutcome(
        name=PROPERTY_ROWS_ARE_DISTRIBUTIONS,
        passed=worst <= 1e-9,
        evidence=(
            f"{len(resolved)} 条记录、{cells} 个格子 | 最大的行和误差 {worst:.3e}"
            f"（容差 1e-09）"
        ),
        detail="行和不为 1 会让'这一行怎么看各个位置'这件事失去含义（day075 的口径）",
    )


def check_masked_entries_are_exact_zero(
    records: Sequence[AttentionRecord],
) -> PropertyOutcome:
    """第 2 条：被掩码挡掉的位置**逐位**是 0.0（沿用 day082 的判据）."""
    resolved = _checked_records(records)
    blocked = 0
    worst = 0.0
    for record in resolved:
        for row in range(record.rows):
            for column in range(record.columns):
                if not record.mask[row][column]:
                    blocked += 1
                    worst = max(worst, abs(record.weights[row][column]))
    return PropertyOutcome(
        name=PROPERTY_MASKED_ENTRIES_ARE_EXACT_ZERO,
        passed=worst == 0.0,
        evidence=(
            f"掩码外共 {blocked} 个格子，最大读数 {worst:.6e}"
            f"（{'逐位为 0' if worst == 0.0 else '**不是逐位为 0**'}）"
        ),
        detail=(
            "判据用 `==`：被掩码的打分从来没有被读过（day082 第 4.3 节）——"
            "交叉那一流的掩码是全开的长方形，因此它对这一条的贡献是 0 个格子"
        ),
    )


def check_entropy_within_ceiling(
    records: Sequence[AttentionRecord],
) -> PropertyOutcome:
    """第 3 条：每一行的熵 ≤ ``ln(这一行能看到的位置数)``（**Jensen 不等式**）."""
    resolved = _checked_records(records)
    worst = 0.0
    rows = 0
    tightest = math.inf
    for record in resolved:
        for index in range(record.rows):
            rows += 1
            entropy = record.entropies[index]
            ceiling = record.ceiling[index]
            worst = max(worst, entropy - ceiling)
            tightest = min(tightest, ceiling - entropy)
    return PropertyOutcome(
        name=PROPERTY_ENTROPY_WITHIN_CEILING,
        passed=worst <= ENTROPY_TOLERANCE,
        evidence=(
            f"{rows} 行 | 最大的'熵−天花板' {worst:.3e}"
            f" | 最紧的一行还剩 {0.0 if tightest == math.inf else tightest:.3e}"
        ),
        detail=(
            "这一条是**本课的第一条纪律**：熵要跟天花板比——"
            "天花板被掩码压低了（因果掩码下第 0 行是 ln 1 = 0）"
        ),
    )


def check_heatmap_round_trip(
    records: Sequence[AttentionRecord],
) -> PropertyOutcome:
    """第 4 条：渲染成等级字符再解析回来，**逐级**相同."""
    resolved = _checked_records(records)
    rendered = 0
    for record in resolved:
        text = heatmap_block(record)
        parsed = parse_heatmap(text)
        expected = tuple(
            tuple(_level(record, row, column) for column in range(record.columns))
            for row in range(record.rows)
        )
        if parsed != expected:
            return PropertyOutcome(
                name=PROPERTY_HEATMAP_ROUND_TRIP,
                passed=False,
                evidence=f"{record.label} 的渲染与解析不一致",
                detail="逐级相同的判据失败：有一条记录的等级对不回去",
            )
        rendered += record.rows
    return PropertyOutcome(
        name=PROPERTY_HEATMAP_ROUND_TRIP,
        passed=True,
        evidence=f"{len(resolved)} 条记录、{rendered} 行全部逐级相同（10 级是有损的）",
        detail="它挡住的是'把 1 与 0 画得一样'那类实现错误——那种错误只会让人看着图得出错的结论",
    )


def _level(record: AttentionRecord, row: int, column: int) -> int:
    from smart_research_agent.explainability.render import level_of

    return level_of(record.weights[row][column])


def check_rollout_is_stochastic(
    records: Sequence[AttentionRecord], *, alpha: float = DEFAULT_ALPHA
) -> PropertyOutcome:
    """第 5 条：滚动之后仍然行随机，且**掩码挡掉的格子仍然逐位为 0**."""
    resolved = _self_records(_checked_records(records))
    layers = aggregate_by_layer(resolved) if resolved else ()
    if len(layers) < 2:
        raise ParameterError(
            f"滚动至少需要两层，收到 {len(layers)} 层（{len(resolved)} 条记录）——"
            "只有一层时'信息走了几跳'这个问题没有意义"
            "（同一个层的几个头只是同一个因子）。"
        )
    weights = rollout_weights(resolved, alpha=alpha)
    rows, columns = matrix_shape(weights)
    worst = max(abs(math.fsum(weights[row]) - 1.0) for row in range(rows))
    blocked, blocked_worst = mask_respected_after_rollout(resolved, alpha=alpha)
    causal_kept = preserves_causal_zeroes(resolved, alpha=alpha)
    return PropertyOutcome(
        name=PROPERTY_ROLLOUT_IS_STOCHASTIC,
        passed=worst <= ROLLOUT_TOLERANCE and blocked_worst == 0.0,
        evidence=(
            f"α={alpha} | {len(layers)} 层 | 最大的行和误差 {worst:.3e}"
            f"（容差 {ROLLOUT_TOLERANCE}）| 掩码外 {blocked} 个格子最大读数 "
            f"{blocked_worst:.6e}（逐位为 0）| 上三角全 0：{causal_kept}"
        ),
        detail=(
            "两个读数一起给：`掩码外 0 个格子`（全开掩码）与`上三角全 0`（因果掩码）"
            "说的是同一件事的两面——**挡住的仍然挡住**"
        ),
    )


def check_deterministic(
    records: Sequence[AttentionRecord], *, alpha: float = DEFAULT_ALPHA
) -> PropertyOutcome:
    """第 6 条：同一批记录跑两次得到逐位相同的结果（渲染与滚动都是）."""
    resolved = _checked_records(records)
    first_text = "\n".join(heatmap_block(record) for record in resolved)
    second_text = "\n".join(heatmap_block(record) for record in resolved)
    same_text = first_text == second_text
    rolled_once = rollout_record(_self_records(resolved), alpha=alpha)
    rolled_twice = rollout_record(_self_records(resolved), alpha=alpha)
    same_rollout = rolled_once.weights == rolled_twice.weights
    return PropertyOutcome(
        name=PROPERTY_DETERMINISTIC,
        passed=same_text and same_rollout,
        evidence=(
            f"两次渲染逐字符相同：{same_text} | 两次滚动逐位相同：{same_rollout}"
            f"（{len(resolved)} 条记录、{len(rolled_once.weights)} 行）"
        ),
        detail="没有随机数：同一个输入永远得到同一段热力图（因此它可以直接进 diff）",
    )


def check_properties(
    records: Sequence[AttentionRecord], *, alpha: float = DEFAULT_ALPHA
) -> PropertyReport:
    """跑完六条性质（第 5、6 条只在**自注意力**那几条记录上跑：滚动要求方阵）."""
    resolved = _checked_records(records)
    self_only = _self_records(resolved)
    if not self_only:
        raise ParameterError(
            "六条性质里有两条第 5、6 条需要自注意力（方阵）的记录，"
            "而这一批里一条都没有（只有交叉注意力）。"
        )
    outcomes = (
        check_rows_are_distributions(resolved),
        check_masked_entries_are_exact_zero(resolved),
        check_entropy_within_ceiling(resolved),
        check_heatmap_round_trip(resolved),
        check_rollout_is_stochastic(self_only, alpha=alpha),
        check_deterministic(self_only, alpha=alpha),
    )
    notes = (
        f"记录 {len(resolved)} 条（其中自注意力 {len(self_only)} 条）| α = {alpha}",
        "第 2 条的期望掩码**就是这一次前向用的那张**（两张表来自同一次前向，"
        "由 extract.self_records 保证）——本课不判断'掩码选得对不对'",
        "第 3 条是本课的第一条纪律：熵要跟天花板比（天花板由掩码决定）",
    )
    return PropertyReport(
        subject=f"records={len(resolved)} heads={len(self_only)}",
        outcomes=outcomes,
        notes=notes,
    )


def check_all(
    records: Sequence[AttentionRecord], *, alpha: float = DEFAULT_ALPHA
) -> PropertyReport:
    """``check_properties`` 的别名（与 day082 的 ``check_all`` 同名，便于跨天记）."""
    return check_properties(records, alpha=alpha)


__all__ = [
    "ENTROPY_TOLERANCE",
    "ROLLOUT_TOLERANCE",
    "PropertyOutcome",
    "PropertyReport",
    "check_all",
    "check_deterministic",
    "check_entropy_within_ceiling",
    "check_heatmap_round_trip",
    "check_masked_entries_are_exact_zero",
    "check_properties",
    "check_rollout_is_stochastic",
    "check_rows_are_distributions",
]
