# day063 源码说明

今天是 **M6 前半段的缓冲复习日**（复习范围：day061 文档解析与加载 + day062 分块策略，
并回看 M5 与 M6 的交接），**不新增代码**，因此本目录不存放代码快照。

最新完整代码快照请见：[../../day062/源码/smart-research-agent/](../../day062/源码/smart-research-agent/)
（M6-D2 的最终状态，包含 `documents/` 与 `chunking/` 两个包、八个 M6 端点、
两份手册与两套演示脚本）。day020 起快照为累积式，包含截至当天全部模块与测试。

## 为什么复习日不放代码快照

与 day042 / day047 / day049 / day056 同一个理由：**复习日的产物是"理解"，
不是文件**。把 day062 的快照复制一份到 day063 只会得到两个二进制相同的目录，
而"哪一份才是最新的"会立刻变成一个要靠 `diff` 才能回答的问题。

复习日真正需要的三样东西都在别处：

```text
教程      ../教程/教程.md            两天地图 + 四条自查清单 + 四组只读核对
习题      ../习题/习题.md            12 题（两道编程题必做）
答案      ../习题答案/习题答案.md     含可直接运行的核对代码
```

## 今日的四组只读核对（可复现）

核对脚本在 day062 快照上运行，**只读、离线、确定性**：

```bash
cd ../../day062/源码/smart-research-agent
PYTHONPATH=. python <核对脚本>
```

| 核对 | 回答什么 | 实测结论 |
|------|---------|---------|
| 一 · 两个身份 | `doc_id` 与 `chunk_id` / `fingerprint` 有什么区别 | 文档 `7feafd1c95ab3577`；块 `chunk_id=4a6680cde33da45e`、`fingerprint=90a20c398e18fb0b`；块文本逐字可回原文 |
| 二 · 记录形状 | `knowledge_records()` 从"整份文档"变成"片段"时字段有没有变 | 键集合完全相同（4 个键）；`text` 从 614 字变成 `[36, 153, 210, 102, 104]`；`metadata` 多出 9 个块级键 |
| 三 · 覆盖与差额 | 四个策略的覆盖率差额是否都在上界内 | `structural` 差额 9 = 上界 9；`fixed` 重复 13.15% 对理论放大量 17.65% |
| 四 · 探针两档深度 | top-3 与 top-8 的差别说明什么 | top-3：2/4 策略 67%；top-8：全部 100% → 差距来自检索深度，不是分块质量 |

## 全量回归（本次运行的护栏结果）

```text
================ 5923 passed, 16 warnings in 593.47s (0:09:53) ================
TOTAL                                                 16956    225   4030    148    98%
Required test coverage of 90.0% reached. Total coverage: 98.11%
```

- 复现：`cd ../../day062/源码/smart-research-agent && pytest -n 8 --dist loadscope`
  （`pytest-xdist` 只是并行工具，不改变用例集合与覆盖率；不加 `-n` 结果相同，
  只是在本机上慢一个数量级）
- 本日**没有任何代码改动**，因此这份结果与 day062 的快照严格对应：
  两次运行的用例数（5923）与覆盖率（98.11%）完全一致，只有耗时随机器负载波动
- `git status` 在快照目录下应当是干净的（运行产生的 `.coverage` / `htmlcov/` /
  `logs/` 都被仓库的 `.gitignore` 覆盖）

## 两天的产物清单（走读入口）

```text
day061  documents/  十个模块  1213 条语句   180 个用例   新包覆盖率 99.6%
        端点 /documents/{loaders,detect,parse,ingest}
        手册 docs/document_loading.md     演示 scripts/documents_demo.py（七节）

day062  chunking/   十个模块  1090 条语句   192 个用例   新包覆盖率 95.85%
        端点 /chunking/{strategies,split,evaluate,records}
        手册 docs/chunking_strategies.md   演示 scripts/chunking_demo.py（八节）
```

## 复习日的走读路线（30 分钟）

```text
1. day062 scripts/chunking_demo.py 第 6 节（四个策略并排）  ——先看结论
2. day062 chunking/__init__.py 的模块清单                  ——十个模块各答什么问题
3. day062 chunking/types.py 的模块 docstring               ——两个身份的取舍
4. day062 chunking/base.py 的 finalize 六步                 ——切完之后发生了什么
5. day061 documents/pipeline.py 的 knowledge_records()      ——接缝的两端摆在一起看
```

## 与下一课的衔接

day064 起进入**向量数据库**：day062 交出去的 `chunk_id` 会成为向量库的主键，
`retrieval_text` 会成为被编码的那份文本，而今天的第 4 步核对（`retrieval_text`
与 `text` 分开）会在那一天第一次以"检索命中率"的形式被验证。
