# day021 源码说明

复习日无新增代码。

当前最新代码快照见 [../../day020/源码/smart-research-agent/](../../day020/源码/smart-research-agent/)（M2-D5 MCP Client 集成完成后的完整项目）。

建议配合 [../教程/教程.md](../教程/教程.md) 第七章的"代码走读路线"做复盘，推荐顺序：

1. [../../day016/源码/smart-research-agent/smart_research_agent/mcp_server/protocol.py](../../day016/源码/smart-research-agent/smart_research_agent/mcp_server/protocol.py)（day016，JSON-RPC 2.0 与三原语描述符建模）
2. [../../day017/源码/smart-research-agent/smart_research_agent/mcp_server/](../../day017/源码/smart-research-agent/smart_research_agent/mcp_server/)（day017，Resources：URI 模板与读取）
3. [../../day018/源码/smart-research-agent/](../../day018/源码/smart-research-agent/)（day018，MCP Tools 与 Prompts）
4. [../../day019/源码/smart-research-agent/](../../day019/源码/smart-research-agent/)（day019，FastMCP Server 完整搭建，stdio/SSE 双传输）
5. [../../day020/源码/smart-research-agent/](../../day020/源码/smart-research-agent/)（day020，MCP Client 集成与工具动态接入）

走读时带着一个问题：**如果把这个组件删掉，整个系统会退化出什么症状？** 能答上来，说明你真的理解了它在架构中的位置。
