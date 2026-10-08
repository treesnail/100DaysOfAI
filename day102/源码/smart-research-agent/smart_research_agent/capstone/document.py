"""``document``：从清单渲染**最终版 README 与架构文档**，并检查它覆盖全部 8 项能力（day099）.

README 与架构文档是本课的两个"交付物文本"。它们不是手写的，而是**从清单渲染**出来的：

```text
manifest.build_manifest()   →  覆盖表（哪一项能力由哪些子包负责、覆没覆上）
render_readme()             →  「最终版 README」：8 项能力 + 9 个阶段 + 7 条性质 + 边界
render_architecture()       →  「架构文档」：装配总览 + 逐子包职责 + 接缝 + 复算口径 + 失败族
check_document_covers_all() →  文档必须覆盖全部 8 项能力（否则抛 DocumentError）
```

## 一、今天最值钱的一句话

> **"文档写全了"不该靠人肉核对：把 8 项能力的 id 渲染进正文，再用一次子串检查
> 证明"每一个 id 都出现过"——这是可以断言的，而"读起来很完整"不是。**

因此 :func:`missing_capabilities` 返回的是**缺了哪些能力 id**（一个可被反驳的清单），
而 :func:`check_document_covers_all` 在缺项时抛 :class:`DocumentError`——
那个名字今天**回来**（见 :mod:`capstone.errors` 的第三节）。

## 二、一条纪律：渲染是**确定性的**

同样的清单渲染两次得到**逐字节相同**的文本：没有时间戳、没有"生成于"、
没有哈希序。因此 :func:`render_documents` 的两次调用可以被一条 `==` 判定，
而"文档说能力 A 由包 B 负责"这句话在两次渲染里是同一句话。

## 三、为什么渲染而不是覆写既有 README

本课**不改动任何既有文件**（见 types 的边界表）：因此最终版 README 与架构文档
是**新渲染出来的文本**，由 :func:`write_documents` 写到给定目录（演示脚本写到
``outputs/capstone/``），而不是覆盖仓库根的那份 ``README.md``。
"新渲染一份"与"改掉旧的那份"是两件事——前者可复算，后者会掩盖 diff。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from smart_research_agent.capstone.errors import FAMILY_OUTCOMES, DocumentError, ParameterError
from smart_research_agent.capstone.manifest import Manifest, build_manifest
from smart_research_agent.capstone.types import (
    CAPABILITIES,
    CAPABILITY_ORDER,
    CAPSTONE_BOUNDARIES,
    CANDIDATE_SUBPACKAGES,
    SUBPACKAGE_DESCRIPTIONS,
    capabilities,
    property_specs,
    stage_specs,
)

#: 渲染出来的两个文件的名字（**新文件**，不覆盖仓库里既有的 README.md）.
README_FILENAME = "README.capstone.md"
ARCHITECTURE_FILENAME = "architecture.capstone.md"

#: 文档名（:func:`render_documents` 的键）.
DOC_README = "readme"
DOC_ARCHITECTURE = "architecture"

#: 文档名的清单（顺序 = 渲染顺序）.
DOCUMENT_NAMES: tuple[str, ...] = (DOC_README, DOC_ARCHITECTURE)


def _manifest(manifest: Manifest | None) -> Manifest:
    """取一份清单（``None`` 时现建一份）."""
    return build_manifest() if manifest is None else manifest


def render_readme(manifest: Manifest | None = None) -> str:
    """渲染「最终版 README」的正文（**确定性**：同一清单 → 逐字节相同的文本）."""
    resolved = _manifest(manifest)
    lines: list[str] = []
    lines.append("# 智研 AI 助手 · 最终版 README（capstone 渲染）")
    lines.append("")
    lines.append(
        f"> 本文件由 `smart_research_agent.capstone.document.render_readme` 从清单渲染而来；"
        f"清单覆盖 {len(resolved.covered)}/{len(CAPABILITY_ORDER)} 项能力。"
    )
    lines.append("")
    lines.append("## 一、八项最终能力")
    lines.append("")
    lines.append("| id | 标题 | 说明 | 承担子包 | 覆盖 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for cap in capabilities():
        coverage = resolved.coverage_of(cap.id)
        mark = "是" if coverage.covered else "否"
        lines.append(
            f"| `{cap.id}` | {cap.title} | {cap.description} "
            f"| {'、'.join(cap.owners)} | {mark} |"
        )
    lines.append("")
    lines.append("## 二、端到端调用链（九个阶段）")
    lines.append("")
    lines.append("| # | 阶段 | 承担能力 | 读什么 |")
    lines.append("| --- | --- | --- | --- |")
    for spec in stage_specs():
        lines.append(
            f"| {spec.index} | `{spec.key}` | `{spec.capability}` | {spec.reader} |"
        )
    lines.append("")
    lines.append("## 三、七条性质（判据分三类）")
    lines.append("")
    lines.append("| 性质 | 判据 | 说明 |")
    lines.append("| --- | --- | --- |")
    for spec in property_specs():
        lines.append(f"| `{spec.id}` | `{spec.criterion}` | {spec.description} |")
    lines.append("")
    lines.append("## 四、被认领与无人认领的子包")
    lines.append("")
    lines.append(f"- 被认领（{len(resolved.claimed)}）：{'、'.join(resolved.claimed)}")
    lines.append(
        f"- 无人认领（{len(resolved.unclaimed)}）："
        + ("、".join(resolved.unclaimed) if resolved.unclaimed else "（无）")
    )
    lines.append("")
    lines.append("## 五、边界（本课明确不承诺的事）")
    lines.append("")
    lines.extend(f"- {item}" for item in CAPSTONE_BOUNDARIES)
    lines.append("")
    lines.append("## 六、怎么跑")
    lines.append("")
    lines.append("```text")
    lines.append("python scripts/capstone_demo.py          # 十一节离线演示（含本节全部读数）")
    lines.append("python -m pytest tests/test_capstone.py -q -o addopts=\"\"")
    lines.append("```")
    return "\n".join(lines)


def render_architecture(manifest: Manifest | None = None) -> str:
    """渲染「架构文档」的正文（**确定性**：同一清单 → 逐字节相同的文本）."""
    resolved = _manifest(manifest)
    lines: list[str] = []
    lines.append("# 智研 AI 助手 · 架构文档（capstone 渲染）")
    lines.append("")
    lines.append(
        "> 架构文档回答一个问题：**一次请求进来，经过了哪些包、留下了什么读数**。"
        f"当前清单覆盖 {len(resolved.covered)}/{len(CAPABILITY_ORDER)} 项能力。"
    )
    lines.append("")
    lines.append("## 一、装配总览")
    lines.append("")
    lines.append("```text")
    lines.append("一次请求 →")
    for spec in stage_specs():
        lines.append(f"  {spec.index}. {spec.key:<9} ← {spec.capability:<22} | {spec.description}")
    lines.append("```")
    lines.append("")
    lines.append("## 二、八项能力与它们的落点")
    lines.append("")
    for cap in capabilities():
        coverage = resolved.coverage_of(cap.id)
        state = "已覆盖" if coverage.covered else "未覆盖"
        lines.append(f"### {cap.title}（`{cap.id}`）")
        lines.append("")
        lines.append(f"- 说明：{cap.description}")
        lines.append(f"- 最近可见的阶段：`{cap.stage}`")
        lines.append(f"- 承担子包：{'、'.join(cap.owners)}")
        lines.append(
            "- 解析结论："
            + "；".join(
                f"{module.name}({module.symbol_count})" for module in coverage.modules
            )
            + f" → {state}"
        )
        lines.append("")
    lines.append("## 三、十二个候选子包的职责")
    lines.append("")
    for name in CANDIDATE_SUBPACKAGES:
        role = "被认领" if name in resolved.claimed else "无人认领"
        lines.append(f"- `{name}`（{role}）：{SUBPACKAGE_DESCRIPTIONS[name]}")
    lines.append("")
    lines.append("## 四、与既有包的接缝")
    lines.append("")
    lines.append("```text")
    lines.append("guard     security.injection_detector / security.content_moderator")
    lines.append("plan      agent.planner.Planner（+ tools.calculator.CalculatorTool）")
    lines.append("retrieve  retrieval.hybrid.HybridRetriever（build_retriever + LexicalIndex）")
    lines.append("pack      retrieval.context.pack_context")
    lines.append("generate  retrieval.generation.RAGGenerator（配 llm.mock.MockLLM）")
    lines.append("ground    retrieval.generation.GroundingReport")
    lines.append("evaluate  evaluation.rag_metrics（recall / precision / mrr / ndcg）")
    lines.append("account   observability.cost_tracker.CostTracker")
    lines.append("trace     observability.tracing.Tracer")
    lines.append("library   vectorstore.FlatVectorStore / llm.embedding.CharNgramEmbedding")
    lines.append("```")
    lines.append("")
    lines.append("## 五、复算口径")
    lines.append("")
    lines.append(
        "`SystemRun.comparable()` 只含确定性字段：问题 / 九段记录 / 答案 / 四条指标 / "
        "费用 / token / span 条数。**不含**时间戳、uuid、耗时与日志文本——"
        "因此同一输入两次运行逐位相同（`diff_count == 0`）。"
    )
    lines.append("")
    lines.append("## 六、失败族（按「该谁去修」分）")
    lines.append("")
    lines.append("| 族 | 该谁去修 |")
    lines.append("| --- | --- |")
    for name, outcome in FAMILY_OUTCOMES.items():
        lines.append(f"| `{name}` | {outcome} |")
    return "\n".join(lines)


def render_documents(manifest: Manifest | None = None) -> dict[str, str]:
    """一次渲染两份文档，返回 ``{文档名: 正文}``（键为 :data:`DOCUMENT_NAMES`）."""
    resolved = _manifest(manifest)
    return {
        DOC_README: render_readme(resolved),
        DOC_ARCHITECTURE: render_architecture(resolved),
    }


def _text_of(documents: dict[str, str]) -> str:
    """把两份文档拼成一段文本（覆盖检查在拼接文本上做）."""
    if not isinstance(documents, dict) or not documents:
        raise ParameterError("documents 必须是非空字典（{文档名: 正文}）。")
    return "\n".join(documents.values())


def missing_capabilities(
    documents: dict[str, str],
    *,
    table: dict[str, Any] | None = None,
) -> tuple[str, ...]:
    """返回文档里**没有出现**的能力 id（空元组表示 8 项全部覆盖）.

    判据是**子串**：只要某一项能力的 id 出现在任意一份文档的正文里，就算覆盖。
    这不是"读起来很完整"的近似——它是一个可以被反驳的清单。
    """
    resolved = CAPABILITIES if table is None else table
    text = _text_of(documents)
    return tuple(key for key in CAPABILITY_ORDER if key in resolved and key not in text)


def check_document_covers_all(
    documents: dict[str, str],
    *,
    table: dict[str, Any] | None = None,
) -> tuple[str, ...]:
    """文档覆盖全部 8 项能力时返回空元组，否则抛 :class:`DocumentError`.

    它与 :func:`missing_capabilities` 的分工是"记录 vs 拒绝"：
    前者给覆盖报告用（不抛），后者给"拒绝交付"那条路用（抛）。
    """
    missing = missing_capabilities(documents, table=table)
    if missing:
        raise DocumentError(
            "渲染出的文档没有覆盖全部能力，缺：" + "、".join(missing)
            + "——'没写'与'不成立'读起来一样，因此覆盖检查必须点名缺的是哪几项。"
        )
    return missing


def document_lines(documents: dict[str, str] | None = None) -> tuple[str, ...]:
    """文档表：逐份文档一行（名字 / 字符数 / 覆盖了多少项能力）."""
    resolved = render_documents() if documents is None else documents
    lines: list[str] = []
    for name in DOCUMENT_NAMES:
        text = resolved.get(name, "")
        covered = sum(1 for key in CAPABILITY_ORDER if key in text)
        lines.append(
            f"{name:<14} | {len(text)} 字符 | 覆盖能力 {covered}/{len(CAPABILITY_ORDER)}"
        )
    lines.append(f"合并覆盖：缺 {list(missing_capabilities(resolved)) or '（无）'}")
    return tuple(lines)


def write_documents(
    directory: str | Path,
    *,
    manifest: Manifest | None = None,
) -> tuple[Path, Path]:
    """把两份文档写到给定目录，返回落盘的两个路径（**不覆盖仓库既有的 README.md**）.

    目录不存在时创建它；文件名固定为 :data:`README_FILENAME` /
    :data:`ARCHITECTURE_FILENAME`。
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    documents = render_documents(manifest)
    readme_path = target / README_FILENAME
    arch_path = target / ARCHITECTURE_FILENAME
    readme_path.write_text(documents[DOC_README] + "\n", encoding="utf-8")
    arch_path.write_text(documents[DOC_ARCHITECTURE] + "\n", encoding="utf-8")
    return readme_path, arch_path


__all__ = [
    "ARCHITECTURE_FILENAME",
    "DOCUMENT_NAMES",
    "DOC_ARCHITECTURE",
    "DOC_README",
    "README_FILENAME",
    "check_document_covers_all",
    "document_lines",
    "missing_capabilities",
    "render_architecture",
    "render_documents",
    "render_readme",
    "write_documents",
]
