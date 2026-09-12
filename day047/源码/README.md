# day047 源码说明

复习日无新增代码。

当前最新代码快照见 [../../day046/源码/smart-research-agent/](../../day046/源码/smart-research-agent/)（M4-D12 应用集成完成后的完整累积快照，**956 个测试全绿，覆盖率 95.99%**）。

全部快照均为**累积式**：day046 的快照已包含 M3 评估体系（day025~032）与 M4 大模型应用开发（day033~046）的全部代码与测试，以及更早的 M1/M2 基础。走读时以 day046 快照为主即可。

> 本日与 day035、day042 的区别：day035 复盘 M3 + M4 前半（day025~034），day042 复盘 M4 前半段（day036~041）；**day047 是 R2 阶段复习日，把 M3 与 M4 合并收口**，并额外做了一次真实核对（见下节「本日的真实核对」）。

本日复习涉及的三段代码：

- **M3 路线 · 尺子（day025~day032）**：
  - [../../day046/源码/smart-research-agent/smart_research_agent/evaluation/harness.py](../../day046/源码/smart-research-agent/smart_research_agent/evaluation/harness.py)（四步骨架 `load_dataset` / `evaluate` / `run` / `summary`，含 `by_tag` 分组汇总）
  - [../../day046/源码/smart-research-agent/smart_research_agent/evaluation/prompt_eval.py](../../day046/源码/smart-research-agent/smart_research_agent/evaluation/prompt_eval.py)（`RULE_WEIGHT=0.4` + `JUDGE_WEIGHT=0.6`、`TIE_THRESHOLD=0.05`、`JUDGE_FALLBACK_SCORE=3.0`）
  - [../../day046/源码/smart-research-agent/smart_research_agent/evaluation/output_eval.py](../../day046/源码/smart-research-agent/smart_research_agent/evaluation/output_eval.py)（`0.4/0.4/0.2` 三因子、`_is_valid_json` / `_check_numeric` 纯代码判定）
  - [../../day046/源码/smart-research-agent/smart_research_agent/evaluation/quality_logger.py](../../day046/源码/smart-research-agent/smart_research_agent/evaluation/quality_logger.py)（`aggregate` 最近 10 条 + 0.05 死区）
  - [../../day046/源码/smart-research-agent/smart_research_agent/evaluation/rag_metrics.py](../../day046/源码/smart-research-agent/smart_research_agent/evaluation/rag_metrics.py)（`retrieval_recall` / `retrieval_precision` / `mrr` / `ndcg` 的边界值）
  - [../../day046/源码/smart-research-agent/smart_research_agent/evaluation/rag_eval.py](../../day046/源码/smart-research-agent/smart_research_agent/evaluation/rag_eval.py)（`RagEvaluator.evaluate_retrieval` / `answer_faithfulness`）
  - [../../day046/源码/smart-research-agent/smart_research_agent/evaluation/agent_eval.py](../../day046/源码/smart-research-agent/smart_research_agent/evaluation/agent_eval.py)（`completion_rate` / `step_efficiency` / `tool_accuracy` + LCS 顺序分）
  - [../../day046/源码/smart-research-agent/smart_research_agent/evaluation/redteam.py](../../day046/源码/smart-research-agent/smart_research_agent/evaluation/redteam.py)（`RedTeamEvaluator` / `default_attack_suite` / `RedTeamReport`）
  - [../../day046/源码/smart-research-agent/smart_research_agent/observability/](../../day046/源码/smart-research-agent/smart_research_agent/observability/)（`cost_tracker.py` / `tracing.py` / `alerting.py`）
- **M4 路线 · 系统（day033~day046）**：
  - [../../day046/源码/smart-research-agent/smart_research_agent/llm/sampling.py](../../day046/源码/smart-research-agent/smart_research_agent/llm/sampling.py)（temperature / top_p 的确定性模拟）
  - [../../day046/源码/smart-research-agent/smart_research_agent/llm/tokenizer.py](../../day046/源码/smart-research-agent/smart_research_agent/llm/tokenizer.py)（`TokenCounter`：tiktoken 精确计数 + 字符级降级）
  - [../../day046/源码/smart-research-agent/smart_research_agent/agent/prompt_library.py](../../day046/源码/smart-research-agent/smart_research_agent/agent/prompt_library.py) 与 [agent/techniques.py](../../day046/源码/smart-research-agent/smart_research_agent/agent/techniques.py)（版本化模板 + 四种技巧）
  - [../../day046/源码/smart-research-agent/smart_research_agent/llm/function_calling.py](../../day046/源码/smart-research-agent/smart_research_agent/llm/function_calling.py) 与 [agent/fc_agent.py](../../day046/源码/smart-research-agent/smart_research_agent/agent/fc_agent.py)（`ToolCallError` 三类 + tool 角色回填循环）
  - [../../day046/源码/smart-research-agent/smart_research_agent/api/](../../day046/源码/smart-research-agent/smart_research_agent/api/)（`app.py` 工厂 + `schemas.py` 契约 + `routes.py` 端点 + `middleware.py` 审计）
  - [../../day046/源码/smart-research-agent/smart_research_agent/multimodal/image.py](../../day046/源码/smart-research-agent/smart_research_agent/multimodal/image.py)（magic bytes 校验 / data URL 组装 / `IMAGE_TOKEN_BUDGET=85`）
  - [../../day046/源码/smart-research-agent/smart_research_agent/llm/embedding.py](../../day046/源码/smart-research-agent/smart_research_agent/llm/embedding.py)（四提供方 + `default_embedding()` 工厂）
  - [../../day046/源码/smart-research-agent/smart_research_agent/llm/cache.py](../../day046/源码/smart-research-agent/smart_research_agent/llm/cache.py)（`SemanticCache` + `CachedLLM`，**本日核对出的断链点**）
  - [../../day046/源码/smart-research-agent/smart_research_agent/security/](../../day046/源码/smart-research-agent/smart_research_agent/security/)（`injection_detector.py` / `content_moderator.py` / `permissions.py` / `audit.py`）
  - [../../day046/源码/smart-research-agent/smart_research_agent/llm/local_model.py](../../day046/源码/smart-research-agent/smart_research_agent/llm/local_model.py) 与 [local_runtime.py](../../day046/源码/smart-research-agent/smart_research_agent/llm/local_runtime.py)（本地部署与显存估算）
- **集成路线 · 焊接（day046）**：
  - [../../day046/源码/smart-research-agent/smart_research_agent/integration/pipeline.py](../../day046/源码/smart-research-agent/smart_research_agent/integration/pipeline.py)（`STAGE_ORDER` 六阶段、`StageRecord`、`PipelineResult`、`_account` 的叶子归因链）
  - [../../day046/源码/smart-research-agent/smart_research_agent/evaluation/perf_baseline.py](../../day046/源码/smart-research-agent/smart_research_agent/evaluation/perf_baseline.py)（`percentile` 最近秩法 + `PerformanceGuard` 判定下限）

## 本日的真实核对（可复现）

复习日不是"只读一遍"。本日对 day046 快照做了一次**只读核对**，方法是用脚本分别 probe 项目里现存的四种"LLM 形态"：

| LLM 形态 | 持有 `usage_log` | `CostTracker.record_from_llm` 直接归因 |
|----------|-----------------|--------------------------------------|
| `MockLLM`（叶子） | 有 | 成功，登记 1 条 |
| `CachedLLM`（缓存包装器） | **无** | `AttributeError: 'CachedLLM' object has no attribute 'usage_log'` ← **遗留缺陷** |
| `ModelRouter`（路由包装器） | 无 | 直接传它失败；传 `router.last_used_llm` 成功（day046 的修补有效） |
| `_TimedLLMProxy`（追踪代理） | 有（`__getattr__` 全量透传） | 成功，登记 1 条 |

同时复核了 `pipeline.py` 的三条路径在假时钟下的确定耗时：满路径 **13.0ms / 6 阶段**、拒答路径 **3.0ms / 1 阶段（`llm.calls == []`）**、缓存命中路径 **7.0ms / 3 阶段**；以及 `build_hybrid_router()` 的分流（复杂度 1 → `qwen3:8b`，复杂度 4 → `gpt-4o-mini`）。结论与 day046 教程一致。

核对出的缺陷已作为工单交给下一份快照：给 `CachedLLM` 增加只读属性 `last_used_llm`，让它像 `ModelRouter` 一样"交代清楚我替谁说话"，使 `IntegratedPipeline._account` 能把用量归因到真正产生 token 的叶子。修复的落地与回归测试见 [../../day048/源码/smart-research-agent/](../../day048/源码/smart-research-agent/) 的 `smart_research_agent/llm/cache.py` 与 `tests/test_wrapper_accounting.py`。

## 推荐走读路线

1. **M3 路线（尺子）**：`evaluation/harness.py` → `prompt_eval.py` → `output_eval.py` + `quality_logger.py` → `rag_metrics.py` + `rag_eval.py` → `agent_eval.py` + `metrics.py` + `report.py` → `redteam.py` → `observability/`；
2. **M4 路线（系统）**：`llm/sampling.py` → `llm/tokenizer.py` → `agent/prompt_library.py` + `agent/techniques.py` → `llm/function_calling.py` + `agent/fc_agent.py` → `api/app.py` + `schemas.py` + `routes.py` → `multimodal/image.py` + `tools/image_analysis.py` → `llm/embedding.py` + `memory/vector_store.py` → `llm/cache.py` → `security/` → `llm/local_model.py` + `local_runtime.py` + `evaluation/local_eval.py`；
3. **集成路线（焊接）**：`integration/pipeline.py`（逐行对照 `STAGE_ORDER` 与 `_account`）→ `evaluation/perf_baseline.py`（`percentile` + `PerformanceGuard`）→ `api/routes.py` 的 `/pipeline/run` 与 `/pipeline/baseline`。

每条路线都请对照 `tests/` 下的同名测试文件阅读——`MockLLM` / `TestClient` / 假时钟驱动的离线测试是这些模块"行为契约"最精确的描述（如 `test_pipeline.py` 的阶段序列与耗时断言、`test_rag_eval.py` 的指标边界值、`test_redteam.py` 的四类拦截率、`test_perf_baseline.py` 的最近秩法与判定下限）。

上一阶段的复习日文档见 [../../day042/源码/README.md](../../day042/源码/README.md)（M4 前半段，day036~041）与 [../../day035/源码/README.md](../../day035/源码/README.md)（M3 + M4 前两课，day025~034），可与之对照衔接。
