# RAG 生产化运维手册（`rag_ops`，day072 / M6-D10）

> 本文是 `smart_research_agent/rag_ops/` 的权威说明。教程在
> [`../../教程/教程.md`](../../教程/教程.md)，演示脚本在
> [`../../scripts/rag_ops_demo.py`](../../scripts/rag_ops_demo.py)（输出见
> `outputs/rag_ops_demo.txt`）。

## 1. 这一层回答的三个问题

M6 的前九课交出一条**能回答问题、也能被评估**的链路（day061 ~ day071）。
而"能跑"与"能运营"之间差三个问题，它们**都不是模型问题**：

```text
① 该不该跑？            调度：间隔 + 抖动 + 退避 + 错过窗口
② 该重算什么？          两级增量：文件级（谁变了）+ 块级（哪些向量要重算）
③ 现在能不能服务？      体检 + 监控：库空不空 / 清单对不对 / 语料新不新 / 质量退没退
```

| 模块 | 回答的问题 |
|------|-----------|
| `types.py` | 五副形状 + 六张封闭表（动作 / 模式 / 三态 / 检查项 / 指标 / 运行状态） |
| `snapshot.py` | 语料现在长什么样（一圈扫盘 → 一份可 diff 的账） |
| `sync.py` | 与上一份快照差在哪、这次该走增量还是整库重建 |
| `schedule.py` | 什么时候该跑、为什么、下一次在哪 |
| `health.py` | 这个实例现在能不能服务（四项检查、三态、退出码） |
| `metrics.py` | 这次量到了什么（11 个指标，量不到就不报） |
| `deploy.py` | 三个服务怎么编排（11 条静态判据 + 确定性渲染） |
| `pipeline.py` | 把上面七件事按固定顺序串成一次运行（十步） |
| `worker.py` | 容器里那个"每天凌晨跑一次"的循环 |

## 2. 两级增量

"增量"在 M6 里出现两次，而它们回答的是不同问题：

```text
文件级（rag_ops.sync）    两份快照的差集      → "要不要跑这一趟"
    一份都没变 → 一行都不写（连解析与分块都不做）

块级（indexing.planner）  计划器四条判据      → "要重算哪些向量"
    一份文档改一行 → 只重算变了的块（缓存命中，其余判 unchanged）
```

两者缺一不可，而**差额本身是一条读数**（实测，见演示脚本第 5、6 节）：

```text
改一份 4000 字文档      文件级 sources_changed = 1，块级 写入 1 + 删除 1
改两份（3 份语料里）    文件级 变更比例 66.7% → 整库重建，块级 写入 3 / 删除 2 / 编码 2
```

判据与产物的边界：

```text
Snapshot  来源路径 + 内容指纹 + 字数 + 媒体类型      ← 指纹只算一次，文件只读一遍
SyncPlan  增 / 改 / 删 / 未变 四组 + 两侧快照号       ← 四组必须把两侧来源各解释一遍
```

`SyncPlan` 的两条恒等式在**构造期**校验（漏一份来源 = 库里少一条记录，而没有报错）：

```text
removed ∪ updated ∪ unchanged == 上一份快照的来源
added   ∪ updated ∪ unchanged == 这一份快照的来源
```

**变更比例的分母是"两侧来源的并集"**，不是"当前来源数"：一个"删光了"的目录
用当前数做分母会得到 `5/0`，用并集得到 `1.0`（这一趟全变了）——后者才是要传给
"要不要整库重建"的那个数字。

## 3. 调度：四条规定

```text
① 抖动是**算出来的**   sha256(锚点, 运行序号) → [0, jitter] 秒
                      用 random 会让两次调用给出两个"下一次"（报告与端点对不上）
② 错过窗口只补跑一次   03:00 的计划、12:00 才醒：只跑一次，且下一个计划点按**计划**算
                      若"下一次"从实际运行时间起算，每次迟到都会把调度整体往后推
③ 失败退避             连续失败 n 次的等待 = min(base × 2^(n-1), max)
                      封顶的意义是"永远会重试"，指数增长的意义是"别每 60 秒撞一次提供方"
④ "没到点"是结论       它不是失败：理由具体（"还差 812 分钟"），且进报告与监控
```

五种模式（`SCHEDULE_MODES`）：`first_run` / `due` / `not_due` / `backoff` / `forced`。
其中只有 `due` 与 `forced` 会让 `due=True`；`ScheduleDecision` 在构造期就拒绝
"模式说该跑、判定说不跑"这类自相矛盾。

## 4. 账本：水位只在成功时前进

```text
last_run_at       上次**尝试**的时间（成功或失败都写）
last_success_at   上次**成功**的时间（只有成功才前进）
snapshot          水位本身（上一份**成功**的回快照）
last_error        最近一次失败的原因（连续失败数 > 0 时必然非空）
```

**失败不推进水位**是这一层最关键的一条纪律：一次失败如果推进了水位，
失败的那批来源会被永久跳过（"已经处理过了"），而库里少的记录没有任何地方会提醒你。
代价是"失败之后同一批来源会被再处理一次"，而这笔代价可以承受——
重跑是幂等的（day065 的增量对账会把没变的块判成 `unchanged`），漏更新不是。

**账本刻意不进 git**（见 `.gitignore` 的 `data/ops/`）：

```text
账本进了 git → 一次全新的部署以为"我已经同步过了" → 第一次运行只做增量
而增量在空库上等于什么都没做
```

因此 `SyncLedger.load` 在**文件不存在**时返回空账本（= 首次运行，本来就该全量），
但**读得回来却读不懂**（坏 JSON、格式版本不认识）一律报错——
假账比没有账危险得多。这与 day071 的 `RagBaseline.load`（文件不存在就报错）
刻意相反，理由也正好相反：**没有基线**与**基线全是零**是两回事。

## 5. 健康：四项检查、三态、一个退出码

```text
vector_store         库里有几条记录        0 条 → 它答不了任何知识库问题
index_integrity      清单与库对不对得上     复用 day065 的 verify_index
corpus_freshness     距上次**成功**同步多久  用 last_success_at，不是 last_run_at
quality_regression   质量相对基线退没退化     只读 day071 的结论，不跑评估
```

三条纪律：

1. **探活不许跑评估**。四项检查的输入全部是"别人已经算好的账"，
   因此容器探针可以每 30 秒调一次。把端到端评估挂进探针的后果是：
   提供方一抖动，**全部**实例会被同一条探针一起摘掉。
2. **"跳过"永远不是 `ok`**。缺清单、缺基线、没评估一律 `warn`。
   （与 day071 的"不下结论 ≠ 通过"同源：报告里不许出现看起来没问题的空洞。）
3. **只有"这一秒就在骗人"的事才 `fail`**：库空、清单与库不符、质量判定回归。
   `fail` 会让探针退出码变成 1，编排层据此停止导流量。

`HealthReport` **不允许为空**：一份"一项都没查"的报告与一份"全绿"的报告
在状态上都是 `ok`，而它们的含义相反。

## 6. 监控：11 个指标，量不到就不报

| 指标 | 单位 | 方向 |
|------|------|------|
| `sources_total` / `sources_changed` | count | neutral |
| `chunks_written` / `sync_seconds` | count / seconds | neutral |
| `index_records` | count | neutral |
| `sync_age_minutes` / `run_failures` | minutes / count | lower_better |
| `quality_retrieval_recall` / `quality_grounded_rate` | ratio | higher_better |
| `quality_hallucination_rate` / `quality_bad_case_rate` | ratio | lower_better |

两点必须写清楚：

```text
单位由指标名决定（OPS_METRIC_UNITS）  同一个指标不可能在两次采样里带两个单位
质量那四个是**抄** day071 的基线       不重算：两份实现一定会分家，而分家没有报错
```

**没有评估就没有质量采样**：那四个指标**不出现在样本里**，而不是以 0 出现。
0 会被读成"幻觉率 0，很好"，而真相是"没人量过"。

## 7. 编排：三个服务、四个卷、11 条判据

```text
vector-store   chromadb/chroma:1.5.3     卷 chroma-data:/data        探针 /api/v2/heartbeat
rag-api        本项目镜像:<tag>           卷 rag-index:/app/data/index + rag-ops-state + 语料只读卷
                                        探针 /rag/ops/health
rag-ops-sync   本项目镜像:<tag>           同一个索引卷与账本卷；无 HTTP 端口 → 写明探针豁免理由
```

`validate_spec` 的 11 条静态判据（每条都能在提交时查出来）：

```text
R1  三个服务角色齐全                  R7  卷挂载点不覆盖镜像自带的路径
R2  探针与豁免理由二选一              R8  chroma 后端必须 depends_on 向量库
R3  持久卷被挂载且在顶层声明          R9  depends_on 指向真实服务
R4  镜像 tag 可追溯（不得 latest）     R10 命令不得为空串
R5  宿主端口不重复                    R11 顶层卷名合法
R6  环境变量名全大写、值可安全渲染
```

**那份 yml 由规格渲染出来，并且被测试逐字节守着**：

```bash
python scripts/rag_ops_demo.py --write-compose     # 重新渲染 docker-compose.rag.yml
python -m pytest tests/test_rag_ops_deploy.py -q   # 手改 yml 会在这里变红
```

为什么这条"守得住"：渲染是**确定性**的（键按名排序、没有时间戳、没有随机名），
因此"文件与规格不一致"是一次字符串比较就能回答的问题。
反过来（先手写 yml，再写脚本去猜它合不合规）会得到一个永远在报"可能不一致"的检查，
而那种告警会在三天内被人静音。

## 8. 镜像与探针

- **向量库**：`chromadb/chroma:1.5.3`（官方镜像 + **固定 tag**）。
  Chroma 官方文档明确建议不要依赖 `latest`：它拦住的是"今天与明天拉到两个镜像，
  而配置看起来一模一样"。心跳路径是官方的 `GET /api/v2/heartbeat`。
- **应用镜像的两个入口**：容器里用 **uvicorn 的 factory 模式**
  （`--factory --host 0.0.0.0 --port 8080`），**不用** `python -m ...api.app`——
  那个入口的 `main()` 绑的是 `127.0.0.1:8000`（本地开发的正确默认值），
  在容器里宿主机的端口转发一个包都进不来，而表现是"容器 healthy、端口开着、请求超时"。
- **`Dockerfile.api` 不叫 `Dockerfile`**：本仓库根目录已经有一份 `Dockerfile`
  （day022 的 MCP Server 镜像），共用一个名字会让 `docker build .` 构建出哪一套
  取决于谁最后改的文件。
- **探针路径**：`/rag/ops/health` 而不是 `/health`。前者答"这个实例还能不能
  服务知识库问题"，后者答"进程还活着"。容器编排关心的是前者。

## 9. 端点约定（`/rag/ops/*`）

| 端点 | 状态码 | 说明 |
|------|--------|------|
| `GET /rag/ops/status` | 200 | 四块只读内容（调度 / 索引 / 监控 / 编排），0 次模型调用、0 次写盘 |
| `GET /rag/ops/health` | **200 / 503** | `fail` → 503（探针靠它工作）；响应体里另给三态计数与四项判据 |
| `POST /rag/ops/sync` | 200 / 400 / 503 | 200 = 本次运行**有结论**（含 `failed`）；400 = 请求跑不了（账本坏了、目录不存在）；503 = 没装配流水线 |

```text
POST /rag/ops/sync       body: {"force": false}
  200 → report.status ∈ {not_due, no_change, synced, failed}
```

为什么"运行失败"是 200：四档状态塞进 HTTP 状态码会让"没到点"与"同步失败"
在监控里长得一样，而它们的处置完全不同（前者什么都不用做，后者要有人去看）。
真正的"跑不了"（账本损坏、语料目录没配对）仍然走 400。

## 10. 配置项与接缝

`settings.rag_ops_*`（14 项）分成四类，**它们的可调性不同**：

```text
语料与产物   source_dir / ledger_path / report_path / compose_path
调度节奏     interval_minutes(1440) / jitter_seconds(300) / max_lag_minutes(720)
             backoff_base_seconds(60) / backoff_max_seconds(3600)
判定阈值     freshness_minutes(2160 = 36h) / churn_full_rebuild(0.5)
编排         api_port(8080) / store_port(8000) / image_tag(v0.1.0)
```

与既有包的接缝：

```text
上游  documents(day061) → snapshot  文件 → Document → 内容指纹
      indexing(day065)   → pipeline  构建器 + 清单 + 版本表 + 备份（rag_ops 不重写）
      rag_debug(day071)  → health    质量基线 + 门禁（rag_ops 只读它的结论）
脚下  config 的 rag_ops_*             不改变任何产物的字节，只决定"怎么运维"
下游  worker + docker-compose.rag.yml 两个运行形态（进程内 / 容器内）
```

一处**跨天改动**要记在这里：`indexing.pipeline.IndexingPipeline` 新增了一个
只读属性 `versions`（+21 行，只增不改）。运维层需要"当前生效的是哪一版"，
而**两个 `IndexVersionStore` 实例指向同一目录**会让运维层那份永远是空的——
于是体检永远报"没有清单"，而磁盘上其实躺着一份完整的清单。
把"版本表只有一个来源"变成结构上的必然，比靠两处代码记得同步更可靠。

## 11. 复现实验

```bash
cd day072/源码/smart-research-agent

python scripts/rag_ops_demo.py            # 十节演示，输出 outputs/rag_ops_demo.txt
python -m pytest tests/test_rag_ops_*.py -q --no-cov          # 305 个新用例
python -m pytest -q                                            # 全量回归 + 覆盖率闸门
```

演示脚本全程离线：flat 后端 + char-ngram 编码器，**零网络、零 API Key**。
`docker compose up` 未在本环境执行（本机没有 Docker）；编排层的正确性由
11 条静态判据、渲染一致性测试与 `Dockerfile.api` 的静态比对共同保证。
