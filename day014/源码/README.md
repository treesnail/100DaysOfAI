# day014 源码说明

复习日无新增代码。

当前最新代码快照见 [../../day013/源码/smart-research-agent/](../../day013/源码/smart-research-agent/)。

> 说明：day011~day013 的源码快照均为累积式——每天的快照包含截至当天的全部模块。因此 day013 快照里已包含 day011 的反思模块与 day012 的多 Agent 模块，下面第 4~6 项可直接在同一份快照中走读。

建议配合 [../教程/教程.md](../教程/教程.md) 第六章的"完整架构串讲"做代码走读，推荐顺序：

1. `smart_research_agent/memory/short_term.py`、`memory/session.py`（day008，短期记忆与会话隔离）
2. `smart_research_agent/memory/vector_store.py`、`memory/long_term.py`、`llm/embedding.py`（day009，长期记忆与向量检索）
3. `smart_research_agent/agent/planner.py`（day010，规划模块）
4. day011 快照中的反思模块（Reflexion：失败检测、反思生成、重试注入）
5. day012 快照中的多 Agent 模块（研究员 / 分析师 / 撰稿人角色与消息交接）
6. day013 快照中的 LangGraph 状态图编排（StateGraph、条件边、MemorySaver 持久化）

走读时带着一个问题：**如果把这个组件删掉，整个系统会退化出什么症状？** 能答上来，说明你真的理解了它在架构中的位置。
