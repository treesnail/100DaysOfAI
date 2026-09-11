# day042 源码说明

复习日无新增代码。

当前最新代码快照见 [../../day041/源码/smart-research-agent/](../../day041/源码/smart-research-agent/)（M4-D8 Embedding 完成后的完整累积快照，**706 个测试全绿，覆盖率 94.60%**）。

全部快照均为**累积式**：day041 的快照已包含 M4 前半段（day036~day041）的完整代码与测试，以及更早的 M3 评估体系与 M1/M2 基础。走读时以 day041 快照为主即可。

本日复习涉及的三段代码：

- **主线一 · 输出端控制（day036~day037）**：
  - [../../day041/源码/smart-research-agent/smart_research_agent/agent/techniques.py](../../day041/源码/smart-research-agent/smart_research_agent/agent/techniques.py)（zero/few-shot、CoT、ToT 的 message 构造器）
  - [../../day041/源码/smart-research-agent/smart_research_agent/agent/prompt_library.py](../../day041/源码/smart-research-agent/smart_research_agent/agent/prompt_library.py)（`PromptTemplate`/`PromptLibrary`/`PromptRenderError`）
  - [../../day041/源码/smart-research-agent/smart_research_agent/llm/function_calling.py](../../day041/源码/smart-research-agent/smart_research_agent/llm/function_calling.py)（`FunctionSpec.from_tool`/`to_openai_tool`、`parse_tool_calls`、`validate_arguments`、三类 `ToolCallError`）
  - [../../day041/源码/smart-research-agent/smart_research_agent/agent/fc_agent.py](../../day041/源码/smart-research-agent/smart_research_agent/agent/fc_agent.py)（`FunctionCallingAgent`）
- **主线二 · 服务化（day038~day039）**：
  - [../../day041/源码/smart-research-agent/smart_research_agent/api/app.py](../../day041/源码/smart-research-agent/smart_research_agent/api/app.py)（`create_app` 工厂 + 依赖注入 + `default_registry`）
  - [../../day041/源码/smart-research-agent/smart_research_agent/api/routes.py](../../day041/源码/smart-research-agent/smart_research_agent/api/routes.py)（`/health` `/chat` `/chat/stream` `/models` `/agent/run` `/vision/describe` `/embeddings`）
  - [../../day041/源码/smart-research-agent/smart_research_agent/api/schemas.py](../../day041/源码/smart-research-agent/smart_research_agent/api/schemas.py)（请求/响应契约 + `ErrorResponse`）
  - [../../day041/源码/smart-research-agent/smart_research_agent/llm/router.py](../../day041/源码/smart-research-agent/smart_research_agent/llm/router.py)（`ModelRouter` 作为 `BaseLLM` 完整替身：`stream`/`chat_with_tools`/`get_model`/`chat_vision`）
- **主线三 · 输入模态语义化（day040~day041）**：
  - [../../day041/源码/smart-research-agent/smart_research_agent/multimodal/image.py](../../day041/源码/smart-research-agent/smart_research_agent/multimodal/image.py)（`validate_image` magic bytes、`build_data_url`、`build_vision_payload`、`IMAGE_TOKEN_BUDGET=85`）
  - [../../day041/源码/smart-research-agent/smart_research_agent/tools/image_analysis.py](../../day041/源码/smart-research-agent/smart_research_agent/tools/image_analysis.py)（`ImageAnalysisTool`，拒绝远程 URL 防 SSRF）
  - [../../day041/源码/smart-research-agent/smart_research_agent/llm/embedding.py](../../day041/源码/smart-research-agent/smart_research_agent/llm/embedding.py)（`EmbeddingProvider`、`CharNgramEmbedding`、`SentenceTransformerEmbedding`、`OpenAIEmbedding`、`default_embedding` 工厂）
  - [../../day041/源码/smart-research-agent/smart_research_agent/memory/vector_store.py](../../day041/源码/smart-research-agent/smart_research_agent/memory/vector_store.py)（`min_score` 过滤、`cosine_similarity`）

建议配合 [../教程/教程.md](../教程/教程.md) 做代码走读，推荐路线：

1. **主线一**：`agent/techniques.py`（四种技巧的纯函数构造器）→ `agent/prompt_library.py`（版本化渲染）→ `llm/function_calling.py`（规格→解析→校验）→ `agent/fc_agent.py`（tool 角色回填的完整循环）；
2. **主线二**：`api/app.py`（工厂+注入）→ `api/schemas.py`（契约）→ `api/routes.py`（各端点与 `resolve_model`）→ `llm/router.py`（路由替身）；
3. **主线三**：`multimodal/image.py`（校验→编码→组装）→ `tools/image_analysis.py`（Agent 工具）→ `llm/embedding.py`（四提供方+工厂）→ `memory/vector_store.py`（`min_score`/余弦）。

每条主线都请对照 `tests/` 下的同名测试文件阅读——`MockLLM`/`TestClient` 驱动的离线测试是这些模块"行为契约"最精确的描述（如 function calling 的离线脚本化模拟、视觉接口的 422/400/404 错误码、embedding 的批量与相似度断言）。

上一阶段的复习日文档见 [../../day035/源码/README.md](../../day035/源码/README.md)（M3+M4 前半，覆盖 day025~day034 的走读路线，可与之对照衔接）。