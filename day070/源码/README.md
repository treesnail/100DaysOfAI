# day070 源码说明

今天是 **M6 缓冲复习日（day061 ~ day069 收口）**，**不新增代码**，因此本目录不存放代码快照。

最新完整代码快照请见：[../../day069/源码/smart-research-agent/](../../day069/源码/smart-research-agent/)（M6-D8「RAG 生成器与 Prompt」完成后的完整项目）。day020 起快照为**累积式**，包含截至当天全部模块、测试与文档，因此 day069 的快照就是 M6 走完八课之后的最终状态。

今天的学习材料：

- 教程：[../教程/教程.md](../教程/教程.md) —— M6 全景图、八课复盘（本质 → 为什么 → 误区）、全链路走读与对账表、概念对照表、误区清单、阶段自查清单、实验数据整理、day071 衔接
- 习题：[../习题/习题.md](../习题/习题.md) —— 15 题（6 选择 + 4 简答 + 3 诊断 + 2 编程）
- 习题答案：[../习题答案/习题答案.md](../习题答案/习题答案.md)

## 建议的代码走读路线

带着一个问题读每一层：**"这一层留下了哪个数字，能回答'这一层丢了几条'？"** 答不上来的模块，在明天的评估报告里就只是一个黑盒。推荐顺序（都是 day069 快照内的相对路径）：

1. `smart_research_agent/documents/types.py` —— `normalize_text` 与 `content_id`：内容指纹即身份（day061）
2. `smart_research_agent/chunking/base.py` —— `Chunker.finalize` 的四条收尾纪律与 `Span`（day062）
3. `smart_research_agent/vectorstore/flat.py` —— `_find_ids` 与 `_ranked_ids`：过滤为何必须早于排序（day064）
4. `smart_research_agent/indexing/types.py` —— `vector_key` / `index_version_id` / `entries_digest`（day065）
5. `smart_research_agent/retrieval/types.py` —— `RetrievalResult` 的四个 `dropped_*` 与 `EMPTY_REASONS`（day066）
6. `smart_research_agent/retrieval/fusion.py` —— RRF 与加权融合：量纲不可比时的两种出路（day067）
7. `smart_research_agent/retrieval/rerank.py` —— `rerank_hits` 的窗口纪律与 `measure_lift`（day068）
8. `smart_research_agent/retrieval/generation.py` —— `GroundingReport` 的三组编号与 `FALLBACK_REASONS`（day069）

## 可复现的验证方式

快照内所有离线演示脚本都可以直接跑（不需要 API Key，全部确定性）：

```bash
cd day069/源码/smart-research-agent
python scripts/documents_demo.py       # day061 解析
python scripts/chunking_demo.py        # day062 分块
python scripts/vectorstore_demo.py     # day064 向量库
python scripts/indexing_demo.py        # day065 索引
python scripts/retrieval_demo.py       # day066 检索器
python scripts/hybrid_demo.py          # day067 混合检索
python scripts/rerank_demo.py          # day068 重排
python scripts/generation_demo.py      # day069 生成与核对
```

全量回归与覆盖率（这就是"测试全绿才 push"那条护栏）：

```bash
cd day069/源码/smart-research-agent
python -m pytest -q
```

## 今天与明天

day070 整理出来的实验数据表（教程第七章）就是 day071「RAG 评估与调试」的输入：今天的人工表字段会在明天变成结构里的字段，今天的走读表会变成归因的优先级清单。
