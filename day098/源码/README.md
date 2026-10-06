# day098 源码说明

今天是 **休息日/缓冲日（11/01 周日）**，**不新增代码**，因此本目录不存放代码快照。

最新完整代码快照请见：[../../day096/源码/smart-research-agent/](../../day096/源码/smart-research-agent/)
（M8-D7「PyTorch 高级与综合实践」完成后的完整项目）。day020 起快照为**累积式**，
包含截至当天全部模块、测试与文档，因此 day096 的快照就是"全栈走完之后的最终状态"——
它里面既有 P 的工程底座、M1 ~ M6 的全部应用/模型能力，也有 M7 的 Transformer 与
M8 的深度学习。

今天的学习材料：

- 教程：[../教程/教程.md](../教程/教程.md) —— 四层全栈地图（应用层 → 评估与系统层 →
  模型与数据层 → 原理层）；M1 Agent 与 M2 MCP 按「本质·为什么·误区」复盘（含两阶段时间线）；
  M3 Harness 与 M4 大模型应用的复盘与互相牵引；M5 微调与 M6 RAG 的复盘与关键读数；
  原理层一句话收束（指回 day097）；**结项素材盘点**（50 个子包 → README 的 8 项最终能力、
  `data/` 三类资产、可复用的判据与护栏、CI 三级门禁）；结项待兑现清单与三条风险；
  15 条自查清单（对齐到具体某一天）
- 习题：[../习题/习题.md](../习题/习题.md) —— 18 题（10 速答 + 1 连线 + 3 对照 + 2 诊断 + 2 盘点）
- 习题答案：[../习题答案/习题答案.md](../习题答案/习题答案.md)

## 快照现状（day096）

核对 day096 快照的顶层结构（都是相对快照根的路径）：

```text
smart_research_agent/   主包，50 个子包（不含 __pycache__）
tests/                  顶层 285 个文件，其中 261 个 test_*.py
scripts/                57 个 *.py（各天的确定性演示脚本）
docs/                   43 份 *.md（各天的权威手册）
data/                   eval / finetune / knowledge 三个目录（见下）
.github/workflows/      ci.yml（三级门禁）与 finetune-nightly.yml（夜间微调）
pyproject.toml          覆盖率硬护栏 fail_under = 90
```

## 建议的代码走读路线

带着一个问题读每一层：**"这一层的产物，依赖下一层哪一条结论？"**
答不上来的地方，就是这一层你还没有真正读懂。推荐顺序
（都是 day096 快照内的相对路径）。

应用层（M1 / M2）：

1. `agent/react_agent.py` 与 `agent/parser.py` —— ReAct 循环与解析器：
   **循环为什么必须有 `max_steps`？解析失败为什么必须抛 `ReActParseError`？**（day006）
2. `tools/base.py` 与 `tools/calculator.py` —— 工具四元组与 AST 白名单：
   **`description` 写给谁看？为什么不能用 `eval`？**（day004）
3. `llm/base.py` 与 `llm/mock.py` —— `BaseLLM` 抽象与 `MockLLM`：
   **没有依赖倒置，后面的测试还能离线吗？**（day005）
4. `mcp_server/protocol.py` —— JSON-RPC 2.0 三件套与三原语：
   **为什么自建协议包绝不能命名为 `mcp`？**（day016）

评估与系统层（M3 / M4）：

1. `evaluation/harness.py` —— 四步骨架与 `by_tag`：
   **为什么"只报总通过率"会掩盖塌方？**（day025）
2. `evaluation/redteam.py` 与 `data/eval/redteam_cases.jsonl` —— 16 条攻击用例：
   **基线 56.3% 是怎么变成 100% 的、修复动作落在哪两个模块里？**（day031）
3. `integration/pipeline.py` —— 六阶段与 `STAGE_ORDER`：
   **为什么"护栏最先、审核在回写缓存之前"？顺序反了会怎样？**（day046）

模型与数据层（M5 / M6）：

1. `finetune/dataset.py` 与 `domain_data/pipeline.py` —— 数据工程与领域数据：
   **"这一批数据能不能用"与"下一批来了怎么办"分别由谁回答？**（day048 / day057）
2. `sft/trainer.py` 与 `peft/layers.py` —— SFT 与 LoRA：
   **`max_length` 为什么必须按渲染并分词后的长度分位定？LoRA 改了哪些参数？**（day050 / day051）
3. `serving/binding.py` 与 `serving/switch.py` —— 绑定检查与灰度切换：
   **五条检查各自把"哪份声明"与"哪处自述"对上？为什么"不是 head"不阻塞？**（day060）
4. `retrieval/hybrid.py` 与 `retrieval/rerank.py` —— 混合检索与重排序：
   **过滤为什么必须在两路都落刀？重排窗口外为什么"没有分"而不是"分很低"？**（day067 / day068）
5. `rag_ops/sync.py` 与 `rag_ops/health.py` —— 两级增量与体检：
   **文件级与块级各回答什么？为什么"跳过"永远不是 `ok`？**（day072）

原理层（M7 / M8）的走读路线见 day097 的源码说明：
[../../day097/源码/README.md](../../day097/源码/README.md)。

## `data/` 资产清单（结项素材）

```text
data/eval/agent_tasks.jsonl       8 行    Agent 任务集（day030）
data/eval/rag_corpus.jsonl        6 行    RAG 语料
data/eval/rag_eval.jsonl          6 行    RAG 评估集
data/eval/redteam_cases.jsonl    16 行    红队攻击用例（day031 的 16 条）
data/eval/safety_pairs.jsonl     16 行    安全偏好对
data/eval/perf_baseline.json       —      day046 流水线性能基线（label=day046-pipeline，samples=3）
data/eval/rag_baseline.json        —      RAG 质量基线（samples=6，k=3，prompt_version=v2）
data/knowledge/rag.md              —      RAG 知识库文档（离线演示的语料来源）
data/finetune/seed_examples.jsonl 18 行   微调种子样本（day048 数据工程入口）
```

两份基线是结项对比的"锚"，且都带**内容身份**（数据 / 提示词版本 / 样本数）——
换数据就必须重采，否则"涨了 / 跌了"是一句无证据的话（day053 的纪律）。

## 可复现的验证方式

快照内的演示脚本与测试都可以直接跑（离线、确定性、不需要 API Key）。
先进入最新快照（`cd day096/源码/smart-research-agent`），再按层挑着跑：

```bash
# M1 / M2：Agent 与 MCP
python scripts/registry_demo.py          # 工具注册表与 LLM 调用层
python scripts/dump_capabilities.py      # MCP Server 的能力清单
# M3 / M4：评估与集成
python scripts/demo_prompt_eval.py       # Prompt 评估（规则 0.4 + 裁判 0.6）
python scripts/run_agent_eval.py --min-completion-rate 0.6 --min-step-efficiency 0.7
python scripts/run_redteam.py --min-block-rate 1.0
python scripts/integration_demo.py       # 六阶段流水线（并更新 perf_baseline.json）
# M5：微调
python scripts/finetune_data_demo.py     # 数据工程
python scripts/sft_demo.py               # SFT 渲染 / 编码 / 计划
python scripts/lora_pipeline_demo.py     # 适配器生命周期 + 合并门禁
python scripts/serving_demo.py           # 部署 + 切换 + 验证 + 成本
# M6：RAG
python scripts/hybrid_demo.py            # 混合检索
python scripts/rerank_demo.py            # 重排序
python scripts/generation_demo.py        # 生成与引用核对
python scripts/rag_ops_demo.py           # 增量同步 + 体检
```

单跑某一天（秒级）：
`python -m pytest tests/test_serving_binding.py tests/test_rag_ops_sync.py -q --no-cov`

全量回归与覆盖率（这就是"测试全绿才 push"那条护栏）：

```bash
cd day096/源码/smart-research-agent
python -m pytest -q
# 本机实测（day096 快照）：collected 14800（14799 passed / 1 skipped），总覆盖率 98.20%
#   逐天累计：14372（day090）→ 14482（day092）→ 14570（day093）→
#             14650（day094）→ 14719（day095）→ 14800 collected（day096）
#   覆盖率硬护栏：pyproject.toml 的 fail_under = 90
```

CI 侧的三级门禁（`.github/workflows/ci.yml`）：

```text
① 单元测试 + 覆盖率（fail_under = 90）
② Agent 评估（--min-completion-rate 0.6 --min-step-efficiency 0.7）
③ 红队拦截率（--min-block-rate 1.0，安全零容忍）
```

## 今天与明天

今天盘出来的三张清单，是结业项目（day099 / day100）的起点：

```text
今天的盘点                                   →  结业项目的用法
"50 个子包 → 8 项最终能力"                    挑选要联调的模块，先分清"用户可见的 29 个"
                                             与"支撑的 21 个"
"data/ 三类资产 + 两份基线"                    演示与回归的数据锚；改数据就必须重采基线
"可复用的判据与护栏"（M1~M8 + CI 三级门禁）     上线检查单：每条能力都要能被反驳
```

而 day097 与 day098 交班的两句话也在这里被接住：

```text
day097（纵向）  M7 决定"能不能表达"，M8 决定"能不能学会、能不能复现"
day098（横向）  P + M1~M8 铺成四层地图，并落到结项素材
→   把两张图叠起来，联调成一个可运行的系统，就是 day099（G1 结业项目整合（一））。
```

- 教程：[../教程/教程.md](../教程/教程.md)
- 习题：[../习题/习题.md](../习题/习题.md)
- 习题答案：[../习题答案/习题答案.md](../习题答案/习题答案.md)
- 最新累积快照：[../../day096/源码/smart-research-agent/](../../day096/源码/smart-research-agent/)
- 昨天的复盘：[../../day097/源码/README.md](../../day097/源码/README.md)
