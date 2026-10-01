"""评估报告渲染：把 AgentEvaluator 的产出写成 Markdown 文件.

报告三段式结构：
  1. 指标总表：completion_rate / step_efficiency / tool_accuracy 四个子项
  2. 薄弱维度归类：按归因类别分组列出失败用例（规划错误 / 工具选错 / 解析失败）
  3. 逐条明细：每条任务的成功与否、步数、工具序列对比、耗时
"""

from __future__ import annotations

from pathlib import Path

from smart_research_agent.evaluation.agent_eval import TaskResult

# 归因类别 -> 报告中展示的中文名与对应的组件
CATEGORY_LABELS = {
    "planning_error": "规划错误（未能在步数限制内完成任务）",
    "wrong_tool": "工具选错（调用了任务不需要的工具）",
    "parse_failure": "解析失败（LLM 输出不符合 ReAct 格式）",
}


def _fmt_tools(tools: list[str]) -> str:
    return " -> ".join(tools) if tools else "（无）"


def render_report(metrics: dict, results: list[TaskResult]) -> str:
    """把评估结果渲染为 Markdown 文本."""
    tool = metrics["tool_accuracy"]
    lines = [
        "# Agent 评估报告",
        "",
        "## 指标总览",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
        f"| 评估任务总数 | {metrics['total']} |",
        f"| 任务完成率 completion_rate | {metrics['completion_rate']:.2f} |",
        f"| 步数效率 step_efficiency | {metrics['step_efficiency']:.2f} |",
        f"| 工具精确率 precision | {tool['precision']:.2f} |",
        f"| 工具召回率 recall | {tool['recall']:.2f} |",
        f"| 工具 F1 | {tool['f1']:.2f} |",
        f"| 工具顺序分 order | {tool['order']:.2f} |",
        "",
        "## 薄弱维度归类",
        "",
    ]

    failures = [r for r in results if not r.success]
    if not failures:
        lines.append("全部任务通过，无失败用例。")
        lines.append("")
    else:
        for category, label in CATEGORY_LABELS.items():
            group = [r for r in failures if r.category == category]
            if not group:
                continue
            lines.append(f"### {label}（{len(group)} 条）")
            lines.append("")
            for r in group:
                lines.append(f"- `{r.task_id}` {r.task}")
                if r.trace.error:
                    lines.append(f"  - 异常：{r.trace.error}")
                else:
                    lines.append(f"  - 实际工具：{_fmt_tools(r.trace.tool_calls)}")
                    lines.append(f"  - 期望工具：{_fmt_tools(r.trace.expected_tools)}")
            lines.append("")

    lines += [
        "## 逐条明细",
        "",
        "| 任务 | 结果 | 步数 | 实际工具序列 | 期望工具序列 | 耗时(s) |",
        "|------|------|------|--------------|--------------|---------|",
    ]
    for r in results:
        mark = "✅" if r.success else "❌"
        lines.append(
            f"| {r.task_id} | {mark} | {r.trace.steps} "
            f"| {_fmt_tools(r.trace.tool_calls)} | {_fmt_tools(r.trace.expected_tools)} "
            f"| {r.trace.latency:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_report(metrics: dict, results: list[TaskResult], path: str | Path) -> Path:
    """渲染报告并写入文件，返回写入路径."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(metrics, results), encoding="utf-8")
    return output
