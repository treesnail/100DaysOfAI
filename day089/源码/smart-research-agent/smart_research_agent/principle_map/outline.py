"""``outline``：分享提纲生成器 **与** 原理文档渲染器（day088 / M7-D12）.

今天要交两样东西，而它们是同一件事的两种说法：

```text
build_outline()      一场"原理 → 应用"的分享提纲：按层分节、每节 3 条要点 + demo + 时长
render_document()    《底层模型原理解析》文档：10 节 markdown，每个数字来自 reconcile 的实测读数
```

把两者放在同一个模块里是刻意的：**提纲的次序与文档的次序必须一致**，
否则"讲的时候先讲推理、写的时候先写数学"这种分歧会以"听众觉得乱"的形式出现，
而它不会报错。这里的唯一次序来源是 :data:`types.LAYER_ORDER`（数学 → 注意力 → 表征 → 推理）。

## 一条纪律：**次序是可以被检查的**

:func:`check_order` 把"提纲的层次序"变成一次拓扑检验：只要某一节的层
排在它前面出现过的层的**后面**（按前置依赖），就抛 :class:`errors.OrderError`。
"讲反了"与"讲对了"在文本上都是合法的 markdown——只有这条检查能把它们分开。
"""

from __future__ import annotations

import pathlib

from smart_research_agent.principle_map import claims, graph as graph_module, reconcile
from smart_research_agent.principle_map.errors import CoverageError, OrderError
from smart_research_agent.principle_map.types import (
    LAYER_ATTENTION,
    LAYER_INFERENCE,
    LAYER_MATH,
    LAYER_ORDER,
    LAYER_REPRESENTATION,
    PRINCIPLE_NOTES,
    PRINCIPLE_NOTES_ORDER,
    PRINCIPLE_PROPERTIES,
    PRINCIPLE_BOUNDARIES,
    PROPERTY_FAILURE,
    TalkSection,
)

#: 每一层对应的 demo 命令（**真实存在的 scripts/*.py**）.
SECTION_DEMOS: dict[str, str] = {
    LAYER_MATH: "python scripts/math_foundations_demo.py",
    LAYER_ATTENTION: "python scripts/attention_demo.py",
    LAYER_REPRESENTATION: "python scripts/retrieval_demo.py",
    LAYER_INFERENCE: "python scripts/inference_optim_demo.py",
}

#: 每一层的标题（顺序即 :data:`types.LAYER_ORDER`）.
SECTION_TITLES: dict[str, str] = {
    LAYER_MATH: "第一层：数学地基 —— 为什么『看哪里』能被学出来",
    LAYER_ATTENTION: "第二层：注意力与结构 —— 分布、掩码、分头与位置",
    LAYER_REPRESENTATION: "第三层：表征与检索 —— 方向、排序与缓存复用",
    LAYER_INFERENCE: "第四层：推理与部署 —— 缓存、量化与预算三笔账",
}

#: 每一节的时长（分钟，整数）——它由"讲三个要点 + 跑一次 demo"的节奏给出.
SECTION_MINUTES: dict[str, int] = {
    LAYER_MATH: 8,
    LAYER_ATTENTION: 10,
    LAYER_REPRESENTATION: 8,
    LAYER_INFERENCE: 10,
}

#: 项目根（``scripts/`` 与 ``docs/`` 都在它下面）.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: 文档的十节标题（顺序即渲染顺序）.
DOCUMENT_SECTIONS: tuple[str, ...] = (
    "十二块拼图总览",
    "第一层：数学地基",
    "第二层：注意力与结构",
    "第三层：表征与检索",
    "第四层：推理与部署",
    "覆盖：四层 × 六应用",
    "七条性质与两类判据",
    "失败族与缺席的 GradientError",
    "十条笔记",
    "五条边界与后续接缝",
)


def check_order(sections: tuple[TalkSection, ...]) -> None:
    """检验提纲的层次序满足前置依赖（违反时抛 :class:`errors.OrderError`）.

    判据是"层的序号非递减"：``LAYER_ORDER`` 里的下标只能往后走，不能倒退。
    同层可以有多节（非递减，而不是严格递增）——一个层用两节讲是允许的。
    """
    rank = {layer: index for index, layer in enumerate(LAYER_ORDER)}
    previous = -1
    for section in sections:
        current = rank.get(section.layer, -1)
        if current < previous:
            raise OrderError(
                f"第 {section.index} 节的层 {section.layer!r} 排在了更靠后的位置："
                "前置依赖是 数学 → 注意力 → 表征 → 推理，讲反了听众会在第一页就掉队。"
            )
        previous = current


def build_outline(graph: graph_module.PrincipleGraph | None = None) -> tuple[TalkSection, ...]:
    """按层生成分享提纲（每节 3 条要点 + 一条 demo + 时长），并检验次序."""
    resolved = graph or graph_module._default_graph()
    evidence = {item.principle: item for item in resolved.evidence}

    def reading(principle_id: str) -> str:
        item = evidence[principle_id]
        if item.upper_bound is not None:
            return f"读数 {item.reading:.3e} ≤ 上界 {item.upper_bound:.3e}"
        return f"读数 {item.reading}"

    bullets: dict[str, tuple[str, ...]] = {
        LAYER_MATH: (
            "注意力 = 可微的检索：softmax(QKᵀ/√d_k)·V 是一组概率对 V 的加权平均——"
            f"这就是『能学』的前提（{reading('attention_is_differentiable_retrieval')}）。",
            "余弦 = 归一化点积：cos(a,b) = a·b/(‖a‖‖b‖)，只看方向不看长度"
            f"（{reading('cosine_is_normalized_dot')}）。",
            "落点：math_foundations.attention.scaled_dot_product_attention 与 "
            "math_foundations.linalg.cosine——两条命题都能被现场算出来。",
        ),
        LAYER_ATTENTION: (
            "注意力权重矩阵的每一行是一个和为 1 的分布——热力图可读的前提"
            f"（{reading('attention_rows_are_distributions')}）。",
            "因果掩码是一张可打印的布尔表，被挡住的位置权重恰好是 0.0"
            f"（{reading('causal_mask_blocks_future')}）。",
            "『分头』发生在投影内部（merge(split(x)) == x 逐位），"
            f"而位置注入打破了置换等变性（{reading('position_encoding_breaks_permutation')}）。",
        ),
        LAYER_REPRESENTATION: (
            "方向相似：cos(5a, a) = 1.0、cos(-a, a) = -1.0，因此入库前要归一化"
            f"（{reading('embedding_similarity_is_direction')}）。",
            "注意力排序与检索排序可以逐行对照：恒等投影 + 单位行时两者完全一致"
            f"（{reading('rank_ordering_matches_attention_peaks')}）。",
            "缓存复用逐位相同：用缓存的 decode 与整段重算的最后一行 logits 相等"
            f"（{reading('cache_reuse_is_bitwise_exact')}）。",
        ),
        LAYER_INFERENCE: (
            "缓存字节数是一条公式：2 · L · T · h · bytes，'公式'与'逐层相加'整数相等"
            f"（{reading('cache_bytes_is_a_formula')}）。",
            "量化误差不超过半格：实测 <= scale/2（上界判定，不是相等）"
            f"（{reading('quantization_error_bounded_by_half_step')}）。",
            "生成的预算是一条和：权重 + 缓存 + 激活 = 总量，而 T_max 的两侧都要查"
            f"（{reading('generation_respects_budget_breakdown')}）。",
        ),
    }

    sections: list[TalkSection] = []
    for index, layer in enumerate(LAYER_ORDER, start=1):
        sections.append(
            TalkSection(
                index=index,
                title=SECTION_TITLES[layer],
                bullets=bullets[layer],
                demo=SECTION_DEMOS[layer],
                minutes=SECTION_MINUTES[layer],
                layer=layer,
            )
        )
    resolved_sections = tuple(sections)
    check_order(resolved_sections)
    return resolved_sections


def outline_minutes(sections: tuple[TalkSection, ...]) -> int:
    """提纲的总时长（分钟，整数）——它必须被算出来，而不是被估计."""
    return sum(section.minutes for section in sections)


def missing_demos(sections: tuple[TalkSection, ...]) -> tuple[str, ...]:
    """提纲里指向的、但**磁盘上不存在**的 demo 脚本（应当为空）."""
    root = PROJECT_ROOT
    missing: list[str] = []
    for section in sections:
        parts = section.demo.split()
        if len(parts) < 2:  # pragma: no cover - 提纲里的命令都有两段
            missing.append(section.demo)
            continue
        script = root / parts[1]
        if not script.is_file():
            missing.append(section.demo)
    return tuple(missing)


def outline_lines(sections: tuple[TalkSection, ...] | None = None) -> tuple[str, ...]:
    """提纲逐行打印（含总时长与 demo 存在性）."""
    resolved = build_outline() if sections is None else sections
    lines: list[str] = []
    for section in resolved:
        lines.append(section.line())
        for bullet in section.bullets:
            lines.append(f"  - {bullet}")
        lines.append(f"  demo：{section.demo}")
    lines.append(f"总时长 {outline_minutes(resolved)} 分钟 | 缺失 demo：{list(missing_demos(resolved))}")
    return tuple(lines)


def _principle_line(principle_id: str, graph: graph_module.PrincipleGraph) -> str:
    item = claims.principle(principle_id)
    evidence = graph.evidence_of(principle_id)
    return (
        f"- **{item.id}**（{item.layer} → {item.application}，{item.source_day}）｜"
        f"落点 `{item.artifact}`｜{evidence.note}"
    )


def document_lines(graph: graph_module.PrincipleGraph | None = None) -> tuple[str, ...]:
    """渲染《底层模型原理解析》文档（10 节）的逐行 markdown."""
    resolved = graph or graph_module._default_graph()
    report = resolved.coverage()
    outline = build_outline(resolved)
    lines: list[str] = [
        "# 底层模型原理解析（day088 / M7-D12）",
        "",
        "> 本文档的每一个数字都来自本课快照的可复现读数：",
        "> `python scripts/principle_map_demo.py`（十一节）与 `python -m pytest`。",
        "> 十二块拼图由探针现场量出，权重由 LCG 生成、输入写死，因此同一个数可以被重新跑出来。",
        "",
        f"## 一、{DOCUMENT_SECTIONS[0]}",
        "",
        "```text",
        f"十二条原理 / {len(resolved.applications)} 个应用 / {report.line()}",
        "```",
        "",
        "| id | 层 | 支撑应用 | 实现落点 | 来源 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in resolved.principles:
        lines.append(f"| {item.id} | {item.layer} | {item.application} | `{item.artifact}` | {item.source_day} |")
    lines.append("")

    for index, (layer, title) in enumerate(
        zip(LAYER_ORDER, DOCUMENT_SECTIONS[1:5], strict=True), start=1
    ):
        lines.append(f"## {['二', '三', '四', '五'][index - 1]}、{title}")
        lines.append("")
        members = claims.by_layer(layer)
        lines.append(f"本层共 {len(members)} 条原理：")
        lines.append("")
        for item in members:
            lines.append(_principle_line(item.id, resolved))
        lines.append("")
        lines.append("```text")
        for item in members:
            lines.append(resolved.evidence_of(item.id).line())
        lines.append("```")
        lines.append("")

    lines.append(f"## 六、{DOCUMENT_SECTIONS[5]}")
    lines.append("")
    lines.append("```text")
    lines.append("按层：" + "、".join(f"{layer}={count}" for layer, count in resolved.layer_counts().items()))
    lines.append(
        "按应用：" + "、".join(f"{app}={count}" for app, count in resolved.application_counts().items())
    )
    lines.append(report.line())
    lines.append("```")
    lines.append("")
    lines.append("没有入边的应用（应为空）：" + str(list(resolved.unsupported)))
    lines.append("")

    lines.append(f"## 七、{DOCUMENT_SECTIONS[6]}")
    lines.append("")
    lines.append("七条性质与它们各自的失败意味着什么：")
    lines.append("")
    for name in PRINCIPLE_PROPERTIES:
        lines.append(f"- `{name}` —— {PROPERTY_FAILURE[name]}")
    lines.append("")

    lines.append(f"## 八、{DOCUMENT_SECTIONS[7]}")
    lines.append("")
    from smart_research_agent.principle_map.errors import ABSENT_FAMILY, ABSENT_FAMILY_REASON, FAMILY_OUTCOMES

    lines.append("七个失败族各自该谁去修：")
    lines.append("")
    for name, outcome in FAMILY_OUTCOMES.items():
        lines.append(f"- `{name}` —— {outcome}")
    lines.append("")
    lines.append(f"连续缺席的那一族：**{ABSENT_FAMILY}**。理由（**第四条，与前三天的都不同**）：")
    lines.append("")
    lines.append(f"> {ABSENT_FAMILY_REASON}")
    lines.append("")

    lines.append(f"## 九、{DOCUMENT_SECTIONS[8]}")
    lines.append("")
    for index, key in enumerate(PRINCIPLE_NOTES_ORDER, start=1):
        lines.append(f"{index}. **{key}**：{PRINCIPLE_NOTES[key]}")
    lines.append("")

    lines.append(f"## 十、{DOCUMENT_SECTIONS[9]}")
    lines.append("")
    lines.append("本课明确**不承诺**的事：")
    lines.append("")
    for index, boundary in enumerate(PRINCIPLE_BOUNDARIES, start=1):
        lines.append(f"{index}. {boundary}")
    lines.append("")
    lines.append(
        "与后续的接缝：这份文档的次序（数学 → 注意力 → 表征 → 推理）就是分享提纲的次序——"
        f"总时长 {outline_minutes(outline)} 分钟，缺口（缺失 demo）{list(missing_demos(outline))}。"
    )
    lines.append("")
    return tuple(lines)


def render_document(graph: graph_module.PrincipleGraph | None = None) -> str:
    """渲染整份文档为一个 markdown 字符串."""
    return "\n".join(document_lines(graph)) + "\n"


def missing_principles(document: str) -> tuple[str, ...]:
    """文档里**没有出现**的原理 id（应当为空）."""
    return tuple(item.id for item in claims.principles() if item.id not in document)


def ensure_document_covers_all(graph: graph_module.PrincipleGraph | None = None) -> None:
    """文档必须覆盖全部十二条原理（缺一条就抛 ``CoverageError``）."""
    document = render_document(graph)
    missing = missing_principles(document)
    if missing:
        raise CoverageError(
            "原理文档漏掉了以下命题：" + "、".join(missing) + "。'没写'与'不成立'读起来一样。"
        )


def demo_scripts() -> tuple[str, ...]:
    """提纲里用到的全部 demo 脚本路径（去重、保序）."""
    seen: list[str] = []
    for section in build_outline():
        script = section.demo.split()[-1]
        if script not in seen:
            seen.append(script)
    return tuple(seen)


__all__ = [
    "DOCUMENT_SECTIONS",
    "PROJECT_ROOT",
    "SECTION_DEMOS",
    "SECTION_MINUTES",
    "SECTION_TITLES",
    "build_outline",
    "check_order",
    "demo_scripts",
    "document_lines",
    "ensure_document_covers_all",
    "missing_demos",
    "missing_principles",
    "outline_lines",
    "outline_minutes",
    "render_document",
]
