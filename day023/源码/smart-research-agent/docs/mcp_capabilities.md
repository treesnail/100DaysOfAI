# SmartResearch MCP Server 能力文档

> 本文件由 `python scripts/dump_capabilities.py` 自动生成，请勿手改。

Server 名称：`smart-research-agent`

## Tools（工具）

| 名称 | 说明 | 参数（JSON Schema） |
|------|------|---------------------|
| `calculator` | 计算数学表达式，支持 + - * / ** % 与括号 | `expression`: string（必填） |

## Resources（资源）

| URI 模板 | 名称 | 说明 | MIME 类型 |
|----------|------|------|-----------|
| `research://knowledge/{doc_id}` | knowledge | 按 doc_id 读取内置知识库文档. | text/plain |

## Prompts（提示词）

| 名称 | 说明 | 参数 |
|------|------|------|
| `research_report` | 生成一份调研报告的撰写提示词 | `topic`（必填）; `style`（可选，有默认值） |
