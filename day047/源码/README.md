# day047 源码说明

复盘缓冲日无新增代码。

当前最新代码快照见 [../../day046/源码/smart-research-agent/](../../day046/源码/smart-research-agent/)（M4-D12 大模型应用集成与优化完成后的完整累积快照，**956 个测试全绿，覆盖率 95.99%**）。

全部快照均为**累积式**：day046 的快照已包含 M4 全段（day036~day046）的完整代码与测试，以及更早的 M3 评估体系与 M1/M2 基础。走读时以 day046 快照为主即可。

本日复盘涉及的四条主线与代码：

- **主线一 · 成本（day043）**：
  - [../../day046/源码/smart-research-agent/smart_research_agent/observability/cost_tracker.py](../../day046/源码/smart-research-agent/smart_research_agent/observability/cost_tracker.py)（`UsageRecord`、`CostTracker.record`/`record_from_llm`/`breakdown`/`top_cost_paths`/`report`）
  - [../../day046/源码/smart-research-agent/smart_research_agent/llm/cache.py](../../day046/源码/smart-research-agent/smart_research_agent/llm/cache.py)（`SemanticCache` 精确 + 语义 + LRU、`CachedLLM` 透明包装与省成本归因、`default_cache`）
- **主线二 · 合规（day044）**：
  - [../../day046/源码/smart-research-agent/smart_research_agent/security/content_moderator.py](../../day046/源码/smart-research-agent/smart_research_agent/security/content_moderator.py)（`ContentModerator`、`PII_PATTERNS`、`_luhn_valid`、`_mask_bank_cards`）
  - [../../day046/源码/smart-research-agent/smart_research_agent/security/injection_detector.py](../../day046/源码/smart-research-agent/smart_research_agent/security/injection_detector.py)（`PromptInjectionDetector`）
  - [../../day046/源码/smart-research-agent/smart_research_agent/api/middleware.py](../../day046/源码/smart-research-agent/smart_research_agent/api/middleware.py)（`AccessLogStore`、`AccessLogMiddleware`）
  - [../../day046/源码/smart-research-agent/smart_research_agent/security/audit.py](../../day046/源码/smart-research-agent/smart_research_agent/security/audit.py)（`AuditLogger`，工具调用审计）
- **主线三 · 部署（day045）**：
  - [../../day046/源码/smart-research-agent/smart_research_agent/llm/local_model.py](../../day046/源码/smart-research-agent/smart_research_agent/llm/local_model.py)（`LocalModelSpec`、`LocalModel`、`normalize_base_url`、探活与清单、`chat_with_tools` 覆写）
  - [../../day046/源码/smart-research-agent/smart_research_agent/llm/local_runtime.py](../../day046/源码/smart-research-agent/smart_research_agent/llm/local_runtime.py)（`estimate_vram_gb`、`bytes_per_parameter`、`HardwarePlan`、`plan_deployment`、`max_context_tokens`、`OllamaRuntime`）
  - [../../day046/源码/smart-research-agent/smart_research_agent/evaluation/local_eval.py](../../day046/源码/smart-research-agent/smart_research_agent/evaluation/local_eval.py)（一致性 / 延迟 / 成本三指标对比）
- **主线四 · 集成（day046）**：
  - [../../day046/源码/smart-research-agent/smart_research_agent/integration/pipeline.py](../../day046/源码/smart-research-agent/smart_research_agent/integration/pipeline.py)（`IntegratedPipeline` 六阶段、`StageRecord`、`PipelineResult`、`build_hybrid_router`、`default_pipeline`）
  - [../../day046/源码/smart-research-agent/smart_research_agent/evaluation/perf_baseline.py](../../day046/源码/smart-research-agent/smart_research_agent/evaluation/perf_baseline.py)（`percentile` 最近秩法、`PerfBaseline`、`PerformanceGuard`、`Regression`、`compare_with_file`）
  - [../../day046/源码/smart-research-agent/smart_research_agent/llm/router.py](../../day046/源码/smart-research-agent/smart_research_agent/llm/router.py)（`last_used_llm`——修复"路由后计费断链"）
  - [../../day046/源码/smart-research-agent/scripts/integration_demo.py](../../day046/源码/smart-research-agent/scripts/integration_demo.py)（六阶段流水线端到端演示）

建议配合 [../教程/教程.md](../教程/教程.md) 做代码走读，推荐路线：

1. **成本线**：`observability/cost_tracker.py`（三种归集维度 + 高成本路径）→ `llm/cache.py`（两级匹配 → LRU → 省成本归因）；
2. **合规线**：`security/content_moderator.py`（敏感词 + 校验型 PII）→ `api/middleware.py`（HTTP 层审计）→ 对照 `security/audit.py`（工具层审计）看"一内一外"；
3. **部署线**：`llm/local_runtime.py`（先算显存再选量化）→ `llm/local_model.py`（同一套 `BaseLLM` 接口）→ `evaluation/local_eval.py`（换模型前先量）；
4. **集成线**：`integration/pipeline.py`（六阶段顺序即策略）→ `evaluation/perf_baseline.py`（把"变慢"变成可判定事实）→ `llm/router.py` 的 `last_used_llm`（计费落叶子）。

每条主线都请对照 `tests/` 下的同名测试文件阅读——`MockLLM`/`TestClient` 驱动的离线测试是这些模块"行为契约"最精确的描述（如缓存的阈值与 LRU 断言、PII 的 Luhn 边界、显存估算的公式对照、性能基线的假时钟与最近秩法）。复盘日涉及的 M3 评估框架代码（`evaluation/` 下的 `metrics.py`/`redteam.py`/`agent_eval.py`/`rag_eval.py`/`prompt_eval.py`/`quality_logger.py`）同样在 day046 快照内，可直接对照走读。

上一阶段的缓冲日文档见 [../../day042/源码/README.md](../../day042/源码/README.md)（M4 前半段，覆盖 day036~day041 的走读路线），更早见 [../../day035/源码/README.md](../../day035/源码/README.md)（M3+M4 前半），可与之对照衔接。
