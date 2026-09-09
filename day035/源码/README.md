# day035 源码说明

复习日无新增代码。

当前最新代码快照见 [../../day034/源码/smart-research-agent/](../../day034/源码/smart-research-agent/)（M4-D2 Tokenization 完成后的完整快照，515 个测试全绿）。

全部快照均为**累积式**：day034 的快照已包含 M3 评估体系（day025~day032）与 M4 前两课（day033~day034）的全部代码与测试，走读时以 day034 快照为主即可。

本日复习涉及的两段代码：

- M3 评估体系（day025~day032）：[../../day034/源码/smart-research-agent/smart_research_agent/evaluation/](../../day034/源码/smart-research-agent/smart_research_agent/evaluation/)、[../../day034/源码/smart-research-agent/smart_research_agent/observability/](../../day034/源码/smart-research-agent/smart_research_agent/observability/)、[../../day034/源码/smart-research-agent/smart_research_agent/security/](../../day034/源码/smart-research-agent/smart_research_agent/security/)
- M4 前两课（day033~day034）：[../../day034/源码/smart-research-agent/smart_research_agent/llm/](../../day034/源码/smart-research-agent/smart_research_agent/llm/)

建议配合 [../教程/教程.md](../教程/教程.md) 做代码走读，推荐路线：

1. **评估主线**：`evaluation/harness.py`（流水线骨架）→ `evaluation/prompt_eval.py`（输入侧双通道）→ `evaluation/output_eval.py` + `evaluation/quality_logger.py`（输出侧双通道+日志）→ `evaluation/rag_eval.py` / `evaluation/agent_eval.py`（检索与轨迹两个特化）→ `evaluation/redteam.py`（攻击视角）；
2. **可观测性主线**：`observability/cost_tracker.py`（成本归因）→ `observability/tracing.py`（span 树）→ `observability/alerting.py`（指标下游消费者）；
3. **M4 主线**：`llm/sampling.py`（温度与核采样的数学）→ `llm/router.py`（多模型路由）→ `llm/tokenizer.py`（token 计数与成本预估）→ `llm/bpe_demo.py`（BPE 核心循环教具）。

每条主线都请对照 `tests/` 下的同名测试文件阅读——MockLLM 驱动的离线测试是这些模块"行为契约"的最精确描述。
