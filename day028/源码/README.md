# day028 源码说明

复习日无新增代码。

当前最新代码快照见 [../../day027/源码/smart-research-agent/](../../day027/源码/smart-research-agent/)（M3-D3 输出质量评估完成后的完整快照）。

本日复习涉及的三天代码：

- day025（M3-D1 评估指标与 Harness 框架）：[../../day025/源码/smart-research-agent/smart_research_agent/evaluation/](../../day025/源码/smart-research-agent/smart_research_agent/evaluation/)
- day026（M3-D2 Prompt 评估）：[../../day026/源码/smart-research-agent/smart_research_agent/evaluation/](../../day026/源码/smart-research-agent/smart_research_agent/evaluation/)
- day027（M3-D3 输出质量评估）：[../../day027/源码/smart-research-agent/smart_research_agent/evaluation/](../../day027/源码/smart-research-agent/smart_research_agent/evaluation/)

建议配合 [../教程/教程.md](../教程/教程.md) 第四章的"全链路串讲"做代码走读：从 `smart_research_agent/evaluation/harness.py` 的 `run()` 出发，依次阅读 `evaluation/output_eval.py`、`evaluation/quality_logger.py`，以及 day026 快照中的 `evaluation/prompt_eval.py`，并打开对应的 `tests/test_output_eval.py` 等用例，对照 MockLLM 驱动的裁判通道测试。
