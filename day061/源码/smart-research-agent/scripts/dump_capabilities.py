"""生成 MCP 能力文档：内存会话发现 Server 能力，输出 docs/mcp_capabilities.md.

用法::

    python scripts/dump_capabilities.py

原理：用 ``create_connected_server_and_client_session`` 在内存中完成握手与
list_tools / list_resource_templates / list_prompts，把结果渲染成 Markdown 表格。
能力文档因此永远与代码保持一致——它是"跑出来的"，不是"手写记的"。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from mcp.shared.memory import create_connected_server_and_client_session

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.mcp_server.server import app  # noqa: E402

OUTPUT_PATH = PROJECT_ROOT / "docs" / "mcp_capabilities.md"


async def collect_capabilities() -> str:
    """连接内存 Server，拉取全部能力并渲染为 Markdown."""
    async with create_connected_server_and_client_session(app._mcp_server) as session:
        tools = (await session.list_tools()).tools
        templates = (await session.list_resource_templates()).resourceTemplates
        prompts = (await session.list_prompts()).prompts

    lines = [
        "# SmartResearch MCP Server 能力文档",
        "",
        "> 本文件由 `python scripts/dump_capabilities.py` 自动生成，请勿手改。",
        "",
        f"Server 名称：`{app.name}`",
        "",
        "## Tools（工具）",
        "",
        "| 名称 | 说明 | 参数（JSON Schema） |",
        "|------|------|---------------------|",
    ]
    for t in tools:
        props = t.inputSchema.get("properties", {})
        required = set(t.inputSchema.get("required", []))
        params = "; ".join(
            f"`{name}`: {schema.get('type', 'any')}"
            f"{'（必填）' if name in required else '（可选）'}"
            for name, schema in props.items()
        )
        lines.append(f"| `{t.name}` | {t.description or '-'} | {params or '-'} |")

    lines += [
        "",
        "## Resources（资源）",
        "",
        "| URI 模板 | 名称 | 说明 | MIME 类型 |",
        "|----------|------|------|-----------|",
    ]
    for r in templates:
        lines.append(f"| `{r.uriTemplate}` | {r.name} | {r.description or '-'} | {r.mimeType or '-'} |")

    lines += [
        "",
        "## Prompts（提示词）",
        "",
        "| 名称 | 说明 | 参数 |",
        "|------|------|------|",
    ]
    for p in prompts:
        args = "; ".join(
            f"`{a.name}`{'（必填）' if a.required else '（可选，有默认值）'}"
            for a in (p.arguments or [])
        )
        lines.append(f"| `{p.name}` | {p.description or '-'} | {args or '-'} |")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    markdown = asyncio.run(collect_capabilities())
    OUTPUT_PATH.write_text(markdown, encoding="utf-8")
    print(f"能力文档已生成: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
