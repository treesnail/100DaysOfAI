# Embedding 索引手册（day065 / M6-D4）

> 本文与代码**同源**：每一个数字都由 `scripts/indexing_demo.py` 实测
> （样本是该脚本自带的 8 条记录、编码器是 `CharNgramEmbedding` 的确定性派生类、
> 后端 `flat`、度量 `cosine`、维度 256），每一个函数名都在
> `smart_research_agent/indexing/` 里存在。本机环境：Python 3.10.11、
> numpy 2.2.6、**未安装 faiss 与 chromadb**（本层不依赖它们）。
>
> 运行（cwd 为 `day065/源码/smart-research-agent`）：
> `python scripts/indexing_demo.py` —— 八节、离线、零网络，
> 输出同时打印到 stdout 并写入 `outputs/indexing_demo.txt`；
> 脚本**不 import tests**，可以被复制走单独运行。

## 1. 一条链：从切好的块到"一版可运维的索引"

```text
day061  documents     文件 → Document（全文 + 块 + 内容指纹）
    │
day062  chunking      Document → ChunkSet.knowledge_records()
    │                 {"doc_id": <chunk_id>, "source", "text", "metadata": {…}}
    ▼
day065  indexing      records → planner（差集）→ encoder+缓存（批量编码）
    │                 → manifest + 版本表 + 备份
    ▼
day064  vectorstore   算好的向量 → VectorBackend.upsert/query
```

```text
errors.py     四族失败分家：编码 / 清单 / 版本 / 备份（修复人各不相同）
types.py      账的形状：EmbeddingIdentity / IndexEntry / IndexManifest / IndexPlan
              / EncodeReport / IndexingReport；时间戳为什么不进版本号
cache.py      文本 → 向量的查表；键里为什么必须有编码器身份
encoder.py    批量编码：批大小的代价、批内去重、三项返回值校验
planner.py    差集：四条判据、两个身份分别判、身份变了要单独说出来
manifest.py   清单：可被任何人重算的版本号 + 与库的一致性体检
versioning.py 版本表：血缘（回滚必须沿血缘，不能按时间倒序）
backup.py     备份：为什么不设保留上限是错的、backup_id 里为什么没有时间
builder.py    装配：维度护栏、阈值换挡（增量不是永远更省）、三处降级
pipeline.py   端到端：Document → Chunk → 记录 → 索引；检索复用向量层
```

本层最重要的一句边界（`scripts/indexing_demo.py` 里逐字打印）：

> 本层不回答'这段文本该不该进库'，只回答'它变了没有、要不要重算向量'——
> 前者是分块与过滤的事，后者才是记账的事。

**三条贯穿全包的纪律**（`indexing/__init__.py` 的原文）：

1. **身份进键**：`vector_key` 含 provider / model / dimension，
   `index_version_id` 含编码器身份 / 度量 / 后端 / 内容摘要。
   于是"换配置"会**必然**让旧向量作废，而不是靠人记得重建索引；
2. **能重算的才叫版本号**：时间戳、机器名、耗时一律不进身份——
   同一份内容在任何时间任何机器上必须算出同一个 16 位十六进制数；
3. **"没做成"要分家**：`IndexPlan` 把未变/新增/更新/删除分成四组，
   `IndexingReport` 再把"数据问题"与"环境问题"分开，剩下不认识的异常一律上抛。

## 2. 三个概念的分工表

本层最容易混的是三个"看起来都在说变了没有"的字符串：

| 概念 | 是什么 | 回答的问题 | 变了以后要做什么 |
|------|--------|-----------|-----------------|
| `fingerprint` | `content_id(text)`，即 `sha256(原文)[:16]`（day062 算好放在 `metadata["fingerprint"]`，planner 优先用它） | **内容**变了没有 | 更新这条记录（重写库里的记录） |
| `vector_key` | `sha256(f"{provider}\|{model}\|{dimension}\n{编码文本}")[:16]` | 这个**向量**还能不能用 | 重新编码这一条 |
| `version_id` | `sha256(f"{MANIFEST_VERSION}\|{identity.key}\|{metric}\|{backend}\|{entries_digest}")[:16]` | **这一版是谁** | 登记/采纳/比对/回滚都以它为准 |

**为什么三个都要有**——少一个都会漏掉一整类变更，而且两类漏判都不报错：

```text
只比 fingerprint  → 漏掉"换了编码器"（内容没变、向量作废）
                    → 新旧向量混在一个库里，排序整体错乱
只比 vector_key   → 漏掉"内容变了但编码文本长度不变"（覆盖写场景）
                    → 库里留着旧内容的向量，命中的却是新原文
没有 version_id   → "上一版长什么样"说不清：差集、回滚、备份都失去锚点
```

实测（demo 第 1 节，同一段编码文本
`'索引手册 > 缓存\n缓存键里必须含编码器身份，否则换模型之后会命中旧向量。'`）：

| 身份 | model | dimension | `identity.key` | `vector_key` |
|------|-------|-----------|----------------|--------------|
| 现状（编码器自报） | `'default'` | 256 | `95df00b1373d4b1b` | `b213ef25c724c42c` |
| 只换 model | `'char-ngram-2-3'` | 256 | `33dc151f42b9724a` | `d46048eb2a6d2285` |
| 只换 dimension | `'default'` | 384 | `9486942d12a1e2d3` | `a1d74e61bf7d29ee` |

三个向量键两两不同（`True`）——**换 model 或换 dimension 之后，同一段文本的键全变了**。
这是"身份进键"的兑现：不需要谁记得重建索引，差集会把全部条目判成 `updated`。

`describe_embedding(embedding)` 只用**公开属性**
（`provider = type(embedding).__name__`、`model → model_name → 回落 "default"`、
`dimension = embedding.dimension`），实测本脚本的编码器拿到的是
`DemoEmbedding / 'default' / 256`：n-gram 阶数藏在私有字段里，看不见，
于是身份落到回落值。代价写在明面上——**身份是建议值而不是判决**，
调用方知道真实配置时应当构造 `EmbeddingIdentity` 自己传进
`BatchEncoder(identity=...)`，把"换配置"重新变成一次必然的整库重算。

缓存键与向量键是**同一个函数**（实测 `cache.EmbeddingCache.key(text)
== types.vector_key(...) → True`，`cache.identity_key == identity.key → True`），
所以换编码器之后旧缓存条目**永远不会命中**——
而不是"命中了旧模型的向量，检索结果只是变得不太对"。

## 3. 批量编码的选型：`batch_size`

`batch_size` 是**成本参数，不是正确性参数**：同一批文本，
换批大小不改变"送了几条"，只改变"分几次送"。实测（demo 第 2 节，8 条文本，
每次都用全新编码器 + 全新缓存）：

| `batch_size` | `encoded` | `batches` | `cache_hits` | 提供方调用（脚本自数） | 命中率 |
|--------------|-----------|-----------|--------------|----------------------|--------|
| 32（默认） | 8 | 1 | 0 | 1 | 0% |
| 4 | 8 | 2 | 0 | 2 | 0% |
| 1 | 8 | **8** | 0 | 8 | 0% |

`batches == 提供方调用次数` 逐位相同——报告里的 `batches` 不是估算。
`batch_size=1` 就是 day064 那条"逐条调用"的基线。

两端各有代价（`config.py` / `encoder.py` 的原文）：

```text
太小   退化成逐条调用（batches == encoded），批处理的收益全部消失
       实测 8 条 → 8 批，与 day064 的 embedding_calls 一模一样
太大   一次请求体积大；一次失败要重试**一整批**
       缺省 32 是"批处理收益"与"单次请求体积"的折中，见 settings.indexing_batch_size
<= 0 直接报 IndexingError，**不替你猜成 1**：那会让逐条调用伪装成批量调用
```

"重放同一批时 `encoded=0`"是怎么做到的（实测：第一次
`8 份文本 → 命中 0 / 编码 8 | 2 批`，第二次 `8 份文本 → 命中 8 / 编码 0 | 0 批`，
提供方调用次数 2 → 2，**本次新增 0**）：

```text
1. 缓存键 = vector_key(provider, model, dimension, text)：**身份进了键**，
   所以"同一个编码器 + 同一段文本"必然命中，跨编码器则必然不命中；
2. 顺序是"先查缓存、再决定要不要发调用"。若反过来（先发一批空调用），
   真实 API 上那也是一次真实账单——空批不是免费的。
```

顺带一条容易漏的账：同一批里的**重复文本只编码一次**，
它既不命中缓存（查它时还没写进去），也不该再花一次编码，
`encoder` 把它计成一次 `cache_hits`——它省下的确实是一次调用。

## 4. 增量 vs 全量：`indexing_full_rebuild_threshold`

`IndexBuilder.build(records, mode=...)` 接受 `full` / `incremental`
（缺省 `None` → 取 `settings.indexing_mode`，默认 `"incremental"`）。
两种模式"写出来的库"是一样的，差别在**过程与成本**：

| | 写什么 | `written` | `unchanged` | `removed` | `reuse_ratio` 的口径 |
|---|--------|-----------|-------------|-----------|---------------------|
| `incremental` | 只写 `plan.to_encode` | added + updated | `len(plan.unchanged)` | `delete()` 真正删掉的条数 | `plan.reuse_ratio`（这一版有多少条完全没变） |
| `full` | `clear()` 后整批写回 | added + updated | **恒为 0** | `plan.removed` 的条数（`clear()` 顺带删掉的） | `cache_hits / requested`（这一次请求里多少条靠缓存免掉） |

全量下 `unchanged` 恒为 0 的理由：`clear()` 之后库里没有"没动过"的东西；
此时决定成本的不是"相对上一版变了多少"，而是"这一次有多少条靠缓存免掉了"。

**什么时候谁更省**：

```text
增量更省   变更比例低时：只编码 + 只写那几条，未变的条目一条都不碰库
           实测（demo 第 4 节）：首次 encoded 8 / batches 1 / written 8；
           重放同一批 encoded 0 / batches 0 / written 0 / unchanged 8
全量更省   变更比例很高时：逐条 upsert 加逐条 delete 加维护差集的开销
           高于"clear() + 一次性写回全部"
```

因此装配层有一条自动换挡：增量模式下若
`plan.changed / plan.total > settings.indexing_full_rebuild_threshold`
就从 incremental 切到 full，并把原因写进 `report.note`。

`settings.indexing_full_rebuild_threshold` 的**默认值是 `0.5`**
（含义：一半以上变了就改走全量重建）。它只在四个条件同时成立时才起作用：
模式是 incremental、上一版存在（没有分母）、`plan.total > 0`、
且比例**严格大于**阈值（等于阈值时不换）。

实测（demo 第 5 节，把阈值调到 `0.1`、8 条里改 4 条 = 50%）：

```text
请求模式      mode=None → settings.indexing_mode='incremental'
计划（只读）  新增 0 / 更新 4 / 删除 0 / 未变 4 | 复用率 50.00%
              changed=4 / total=8 → 比例 50.0% > 阈值 0.10
实际构建      mode='full'，version a8d699e98d8a35e1 ← 7a19694afcc814e8
              written=8 unchanged=0 encoded=4 cache_hits=4 batches=1
              reuse_ratio=0.5（全量下报的是缓存命中率）
report.note   变更比例 50.0% 超过阈值 0.10，已从 incremental 切到 full：
              增量不是永远更省——变更比例很高时，逐条 upsert 加逐条 delete
              的开销反而高于清空重写
```

注意 `encoded=4` 而不是 8：**换挡换的是"怎么写库"，不是"要不要编码"**——
没改的那 4 条仍然靠缓存免掉了编码。**全量重建 ≠ 重新编码。**

这句话必须由装配层说：planner 看不到"写库要花多少"，后端也不知道
"这批数据相对上一版变了多少"，两边各缺一半（`builder.py` 的原文）。

## 5. 差集：四条判据与一句原因

`planner.plan_index(previous, records, identity)` 的判据只有四条，
但**顺序不能换**（每一条的"为什么在这个位置"都写在函数 docstring 里）：

```text
previous 为空                → 全部 added（没有上一版，每一条都是新的）
previous 里有、新记录里没有   → removed（先判"谁不在了"）
fingerprint 变了             → updated（内容变了，向量必须重算）
vector_key 变了              → updated（编码器身份或编码文本变了）
两者都没变                    → unchanged（**复用向量**，省钱的来源）
```

实测（demo 第 3 节，同一个 parent 清单
`7a19694afcc814e8`，`DemoEmbedding/default (256d)`，cosine，flat，8 条）：

| 场景 | 新增/更新/删除/未变 | `reuse_ratio` | `to_encode` | `changed` | `plan_reason()`（摘要） |
|------|--------------------|---------------|-------------|-----------|------------------------|
| ① 未变 | 0 / 0 / 0 / 8 | 1.0 | 0 | 0 | 没有变化：8 条全部复用（复用率 100.00%），本次不需要任何编码调用 |
| ② 内容变了（改 1 条） | 0 / 1 / 0 / 7 | 0.875 | 1 | 1 | 增量更新：新增 0 条 / 更新 1 条（内容或编码文本变了）/ 删除 0 条，复用 7 条（复用率 87.50%） |
| ③ 删了一条 | 0 / 0 / 1 / 7 | 0.875 | 0 | 1 | 增量更新：新增 0 条 / 更新 0 条 / 删除 1 条，复用 7 条（复用率 87.50%） |
| ④ 换了身份（只改 model） | 0 / **8** / 0 / 0 | 0.0 | 8 | 8 | 编码器身份变了（旧 `DemoEmbedding/default (256d)` → 新 `DemoEmbedding/char-ngram-2-3 (256d)`）：全部 8 条需重算，这不是数据变更 |

两条读法：

- **② 与 ③ 的计数可以重合**（`updated=1/unchanged=7` 与 `removed=1/unchanged=7`）：
  只比总数看不出差异，差别在"哪些 id 落在哪一组"——这就是 `IndexPlan`
  存 id 列表而不只存计数的理由；
- **④ 是本模块最该记住的地方**：报告上写着"更新 8 条"，但**一条数据都没改**。
  数字相同、下一步动作完全不同（换模型要重新标定检索阈值，改一条原文不用），
  所以 `plan_reason(previous, identity, plan)` 必须把这一句说出来：
  `IndexPlan` 里只剩四组 id，已经看不出原因了。

`IndexPlan` 的两条硬约束（构造时校验，触发一次就说明上游写错了）：
四个 id 列表各自**升序且无重复**、跨动作**不重叠**。
其余是属性算出来的派生量：`to_encode = added + updated`、
`changed = added + updated + removed`、`total` 是四个列表之和、
`reuse_ratio = unchanged / total`（`total == 0` 时是 `0.0`，不是除零错误）。

## 6. 版本与血缘：为什么回滚必须沿血缘

版本表是 `IndexVersionStore`，两个文件：

```text
<indexing_dir>/manifest_history.jsonl   一行一个清单（**追加**语义：加一版 = 写一行）
<indexing_dir>/current.txt              一个版本号（被**采纳**的那一版；空文件表示还没采纳过）
```

`current` 单独放一个文件而不是写进 JSONL，是因为它是**状态**而不是**事件**；
分开之后两者的失败模式互不牵连：历史丢一行只是少一版，指针丢了只是"还没采纳过"。

"登记（`register`）"与"采纳（`adopt`）"刻意分开：
登记的版本可能**从未生效过**（构建失败、验证不过、被人工否决）。
`builder.build` 在成功路径上先 `register` 再 `adopt`，因此"写进版本表的都生效过"
这句话是错的——而回滚正是被这句话带偏的那个动作。

```text
v1 ─► v2（采纳） ─► v3（构建失败，从未采纳）
                    ▲
        按登记顺序倒着数：v3 是"最新的一版" → 回滚会回到 v3
        按血缘从 current 回溯：从 v2 出发          → 回滚回到 v1
```

**回滚到一个从没生效过的版本**在报告里完全看不出来：`current` 从一个合法的
版本号变成另一个合法的版本号，没有异常、没有告警，只有下一次检索开始
给出另一批结果。因此 `rollback` 实现成**沿 `parent_version` 回溯**
（`lineage()` 的最后一版就是 `current`，`steps=1` 取 `chain[-2]`），
并且血缘断裂或成环都直接抛 `VersionError`，而不是"能走多远走多远"。

实测（demo 第 6 节，连续构建三版、每版只改一条原文）：

```text
history（登记顺序，旧 → 新）
  #1 版本 7a19694afcc814e8 ← （首版）            | 摘要 3777a9f5c0ce510e
  #2 版本 fa3967020368edc1 ← 7a19694afcc814e8    | 摘要 35da5854084b4154
  #3 版本 2f60e07f3a3a0170 ← fa3967020368edc1    | 摘要 ad0d9a3e185c6e77
current = 2f60e07f3a3a0170（构建成功即采纳）
v2.lineage(v1) → 紧接 7a19694afcc814e8
v3.lineage(v1) → 接在 fa3967020368edc1 之后（不是 7a19694afcc814e8）
血缘链         7a19694afcc814e8 → fa3967020368edc1 → 2f60e07f3a3a0170

adopt(v2) → current=fa3967020368edc1，血缘链 7a19694afcc814e8 → fa3967020368edc1
rollback(steps=1) → current=7a19694afcc814e8（== v1）
回滚后库仍然 8 条（**回滚只改版本指针，不动库与数据**）
```

最后一行是一条必须写清的边界：**回滚是运维动作，不自动重建索引**；
要真正重建某一版，请再调用一次 `/indexing/build`。

`diff(left, right)` 用同一套判据（`fingerprint` 或 `vector_key` 变即 `updated`）
给出两版之间的差集，`reason` 写成 `"<left> → <right>"`（版本号才是唯一标识，
"第 2 版到第 3 版"会随"哪个表里数"而变化）。

## 7. 备份策略：`keep`、`backup_id` 与"持久化快照"

```text
备份不是日志：日志的价值在"完整"，备份的价值在"能回去"，
而"能回去"只需要最近几份。不设上限的备份目录会一直长到吃满磁盘，
而吃满磁盘的那一刻，正在写库的那次构建会失败——
**备份把主流程搞挂了**，这是备份机制最讽刺的失败方式。
```

因此 `IndexBackupStore.create()` 之后总是自动 `prune()`，`keep <= 0` 直接报
`BackupError`（一个一份都不保留的备份机制是自相矛盾的）。
`keep` 的缺省取 `settings.indexing_backups_keep`，**默认 `3`**
（常量 `backup.DEFAULT_KEEP = 3` 是"配置读不到时"的兜底，也是文档引用的那个数）。

`backup_id` 只由 `(version_id, backend, 序号)` 决定，**`created_at` 不进 id**：

```text
时间戳进 id → 同一份内容在同一台机器上备份两次得到两个 id
            → "这两个备份其实是同一版"这件事再也说不清
            → 而清理与对账正是靠"哪些备份是同一版"来做的
```

代价是"两次创建得到同一个 id 前缀"，于是补一个**四位序号**
（按"这个目录里已有的最大序号 + 1"，不是"份数 + 1"——裁剪之后份数会变小，
用份数会让新备份原地覆盖旧目录）。实测（demo 第 7 节，`keep=2` 建三份，
`created_at` 用注入的固定时钟 `2026-01-02T03:04:05+00:00`）：

```text
backup_id                              版本                文件/字节
7a19694afcc814e8-flat-0001             7a19694afcc814e8    1 个 / 43061 B   ← 被 prune
e0a7df11887477c2-flat-0002             e0a7df11887477c2    1 个 / 42517 B
23659a8f16b66f39-flat-0003             23659a8f16b66f39    1 个 / 41744 B
```

**备份与"持久化快照"是两件事**：

| | `backend.persist()` | `IndexBackupStore.create()` |
|---|---|---|
| 是什么 | **库自己**的落盘 | **额外的**拷贝 |
| 有几份 | 总是"现在这一版"，一份 | 若干份，超过 `keep` 就裁掉最旧的 |
| 带不带账 | 不带 | 库文件 + `manifest.json` **成对**固化，并登记 `BackupRecord` |
| 用来做什么 | 重启后把库读回来 | "重建把库写坏了" → 一次 `restore` |

`prune` 只删**目录**，账本那一行留着——实测的三组数字正好把这三件事分开：

```text
create 是追加写   账本 backups.jsonl 里 3 行（三份各写一行）
内存里的 list()   2 条（被裁掉的那份连内存记录一起清掉）
load() 返回       2 —— **实际可用**份数（目录还在的），不是记录条数
load() 之后       list() 有 3 条（账本里那三行都回来了），
                  被裁掉的那份 get() 仍取得到，但它的目录不在了，restore 会报 BackupError
```

`read_manifest(backup_id)` 是"我想看看那一版是什么，但不想动现在的库"：
实测读它之后库仍是 8 条、账本仍是 3 条。`restore(backup_id, target_dir=...)`
把快照里的文件拷回去，**并且把清单也拷回去**——实测恢复出来的两个文件是
`manifest.json`（1750 B）与 `store_v2.json`（40767 B）：

```text
恢复出来的库如果没有账，下一次增量构建会把它当成"没有上一版"而整库重算。
```

两条纪律：`restore` **不动**目标目录里已有的其它文件（避免"恢复备份顺带清库"）；
`create` 在动手前**校验全部源文件**（缺一个就拒，"少备份一个文件而报告说成功"
是备份机制最不该有的行为）。格式版本（`BACKUP_VERSION = 1`）不符时**拒读/拒恢复**。

## 8. 一致性体检：四种不一致

两个检查点：`manifest.compare_with_store(manifest, backend)` 给原始字段，
`manifest.verify_index(manifest, backend)` 收敛成 `{"ok", "checks", "problems"}`。
`builder.verify(manifest=None)` 是它在装配层的入口（**不做任何修复**——
它的名字是 verify，不是 repair）。

| 不一致 | 表现（`problems` 里的原文口径） | 处置 |
|--------|-------------------------------|------|
| 清单有、库里没有 | "清单里有 N 条记录在库里找不到：[…]——上一次构建可能半途失败，或有人手工删过记录" | 先重建清单（或用这些 id 补回向量），**不要在这份清单上做增量** |
| 库里有、清单没有 | "库里有 N 条记录不在清单里：[…]——它们不会被增量更新覆盖" | 同样先重建清单 |
| 维度不符 | "清单记的是 X 维，库里是 Y 维——这份清单描述的是另一个编码器建的库" | 用当前编码器重建（`build(..., mode='full')`），或换回建库时那个编码器 |
| 度量不符 | "清单记的是 'cosine'，库是 'ip'——同一批向量在两种度量下的最近邻不是同一批" | 用清单的度量**新建实例**，而不是让一个已有实例改度量 |
| 后端不符 | "清单记的是 'flat'，库是 'chroma'——换后端可以重建，但不能与旧版共用一份清单" | 为新后端单独记一份清单 |

维度不符为什么让 `ok` 为假：它意味着这份清单描述的是**另一个编码器建的库**，
拿它算差集会得到一份看起来正常、实际上完全不相干的计划
（所有 `vector_key` 都不同 → 全部重算，"清单与库对不上"被伪装成"数据全变了"）。

实测（demo 第 8 节，构建一版后人为从库里删掉 `'a3e81f5c20d97b46'`，
`delete` 返回真正删掉的条数 = 1）：

```text
missing_in_store  = ['a3e81f5c20d97b46']
orphan_in_store   = []
count_manifest    = 8 / count_store = 7
dimension_match   = True / metric_match = True / backend_match = True
extra_tokens      = {'manifest': 209, 'store': 178, 'diff': -31}
tokens_match      = False
verify_index(...) → ok=False，problems 共 1 条：
  清单里有 1 条记录在库里找不到：['a3e81f5c20d97b46']——上一次构建可能半途失败，
  或有人手工删过记录；请先重建清单（或用这些 id 补回向量），不要在这份清单上做增量。
```

`extra_tokens` 返回三键（`manifest` / `store` / `diff`）而**不是一个合计值**：
"两边的 token 合计"要对账就必须能看出差额是哪一边多出来的。
`tokens_match` 单独给出，但**不参与 `ok` 的判定**——`token_count` 不进内容摘要
（见 `types.entries_digest`），改一个统计字段不需要重建索引，
把它当问题会让检查结果过度敏感。

最后一条对照：`builder.verify()` **无参**时清单是从库现算的
（`manifest_from_store`），实测 `ok=True`、`problems=[]`——**必然对得上**。
所以"库里被人删了一条"这种问题，只有拿**原来那份清单**来对账才看得出来；
反过来说，重建出来的清单只是"从现在起"的基线，不是历史的那一版。

## 9. 六条踩坑清单

1. **时间戳进版本号 → 版本不可复算。** 同一份内容在两台机器上得到两个版本号，
   "这个版本在哪台机器上建的"就变成一个必须回答的问题。`IndexManifest` 带
   `created_at`，但它**不参与** `version_id`（`build_manifest` 里没有它的位置）。
   实测的证据是 demo 第 4 节：重放同一批 → `version_id` 逐位相同
   （`7a19694afcc814e8`），版本表里仍然只有 1 版。若时间戳进了版本号，
   这一节立刻会多出一个"凭空多出来的版本"。
2. **缓存键不含编码器身份 → 换模型后命中旧向量。** 缓存键若取
   `sha256(text)[:16]`，换一次 embedding 模型之后"文本没变 → 键没变 → 命中旧向量"，
   新旧向量混在一个库里，检索结果整体错乱——**且不抛任何异常**。
   本包的键是 `vector_key(provider, model, dimension, text)`；实测换 model 或
   换 dimension 都让同一段文本的键全变（`b213ef25c724c42c` →
   `d46048eb2a6d2285` / `a1d74e61bf7d29ee`）。
3. **逐位比较 float32 → `unchanged` 永不可达。** Chroma 把向量存成 float32，
   读回来必然带量化误差；若按逐位比，重放同一批永远被报成 `updated`，
   而 `unchanged` 正是本课要用的那个数（`vectorstore` 侧对应的容差是
   `VECTOR_EQUAL_TOLERANCE = 1e-5`）。同理，给记录加一个"仅用于展示"的字段
   （例如入库时间）再比较整个 dataclass，也会让整条比较**静默地永远不相等**。
4. **回滚按时间倒序 → 回到没生效过的版本。** 见第 6 节的反例：v3 构建失败、
   从未采纳，按登记顺序倒着数会回到 v3，而 `current` 的变化在报告里
   看不出任何异常。回滚必须沿 `parent_version` 走。
5. **备份源文件缺失若静默跳过 → 假备份。** "少备份一个文件而报告说成功"
   会在恢复后表现为"库少了几条记录"，而那要等到检索结果变差才会被发现。
   `_resolve_sources` 因此**先全部校验再开始拷**，缺一个就拒绝整次备份；
   同名源文件也拒（备份目录是平铺的，同名会互相覆盖）。
6. **批量失败不逐条重试 → 一条坏数据毁掉整批。** 本层三处降级都是
   "先整批、失败再逐条"：空白文本直接进 `failures`（一小段编码都不发起）、
   一批编码抛错则对该批逐条重试、一次 `upsert` 抛错则逐条重试并定位到
   `record_id`。代价只在**出错**时付出，收益是把故障从"这一批 32 条都不见了"
   缩小到"这一条写不进去"；而且**每一条最多被重试一次**，不存在重试风暴。
   注意失败的那条不计入 `encoded` / `batches`，所以出错时的报告里
   编码成本**偏低**——这一条由 `note` 说明，不是可以糊过去的差额。

两条与"报告本身"有关的补充纪律：

- **非 `IndexingError` / `VectorStoreError` 的异常一律上抛**，不记进 `failures`。
  一个宽泛的 `except Exception` 会把后端实现里的 `TypeError` 记成一行
  `failed=1`，于是它永远不会被修——**报告只能收留我们认识的失败**；
- **失败的条目不能进清单**（它没写进去，写进去就会让 verify 报"清单里有、库里没有"）；
  但增量模式下"这一条在上一版清单里"的那些要**保留上一版的条目**，
  并且从待删列表里摘出来：坏数据不该毁掉整批，也不该借它的名义
  把库里已有的东西删掉。

## 10. 配置项与端点

配置项（`config.Settings` 的 `indexing_*` 字段）：

```python
indexing_mode: str = "incremental"                  # incremental / full
indexing_batch_size: int = 32                       # 一次送给提供方的文本条数
indexing_cache_path: str = ""                       # 空串 = 只用内存缓存
indexing_dir: str = "data/index"                    # 清单 / 历史 / 指针 / 备份的目录
indexing_backups_keep: int = 3                      # 备份保留份数（超过就删最旧的）
indexing_full_rebuild_threshold: float = 0.5        # 变更比例超过它 → 改走全量重建
```

这一组与 `vector_*` 的差别：**它们不改变"存什么"，只改变"怎么算出要存的东西"**。
因此换 `batch_size` / `cache_path` 都不需要重建索引（结果逐位相同），
而换 `indexing_dir` 只影响清单与备份放在哪。

`default_indexing_pipeline(*, embedding=None, backend=None, cache=None)`
按 `settings` 装配一条完整流水线（后端 `create_backend()`、编码器
`default_embedding()`、缓存带着编码器身份、版本表 `<indexing_dir>`、
备份表 `<indexing_dir>/backups`）。三处刻意的行为都写在那里的 docstring 里：
缓存的三个身份字段来自**编码器**而不是配置；版本表与备份表**只装配、不读盘**
（要读磁盘上的历史请显式 `load()`）；**一个目录都不建**（问一下状态不该在磁盘上
留下任何东西）。备份目录取 `backups` 子目录（`BACKUP_DIR_NAME`），
因为备份是"若干份快照、每份一个目录"，与单文件的历史/指针混在一起会让
`ls` 的结果读不出"现在生效的是哪一版"。

七个端点（`/indexing/*`，与本层同日的 API 层交付）：

| 端点 | 方法 | 回答什么 |
|------|------|---------|
| `/indexing/status` | GET | `pipeline.stats()`：库 / 身份 / 清单（`include_entries=False`）/ 版本数 / 备份数 / 缓存统计 |
| `/indexing/plan` | POST | 给一批 records → 差集与模式判定（**只读**：不写库、不写盘、不动版本表） |
| `/indexing/build` | POST | 执行构建 → 返回 `IndexingReport`（**唯一会改状态的端点**） |
| `/indexing/versions` | GET | 版本历史（`history()` + `current`） |
| `/indexing/verify` | POST | 清单与库的一致性体检 → `verify_index` 的字典 |
| `/indexing/backup` | POST | 创建一次备份 → 返回 `BackupRecord` |
| `/indexing/rollback` | POST | 回滚到血缘上的上一版 → 返回回滚后的版本摘要 |

两条与部署有关的约定：`create_app(..., indexing_dir="")` 的**缺省是空串**
（与 day064 的 `vector_persist_path = ""` 同一理由：默认行为必须是
"明确不落盘"，否则跑一次测试就会在仓库里留下目录；要落盘请显式传
`indexing_dir=settings.indexing_dir`）；`indexing_dir` 为空串时
`/indexing/backup` **优雅失败成 HTTP 400**（消息说明"本实例没有配置索引目录"），
不 500，也不在请求里偷偷建目录。最近一次构建报告挂在
`app.state.indexing_last_report`（常量 `INDEXING_LAST_REPORT_STATE_KEY`），
换一个 app 实例它必须是 `None`——写操作不跨实例泄漏。

## 11. 与 day064 / day066 的接缝

**脚下 day064（向量库）**——本层**不改它**，只在它之上记账：

```text
VectorBackend.upsert / delete / clear / persist   本层写库的全部手段
VectorRecord / WriteReport                        written = added + updated
vectorstore.pipeline.record_id_of / embedding_text  "id 在哪一层"与"该编码哪段文本"
vectorstore.pipeline.VectorIngestPipeline          pipeline.search 一行转发，不重写
evaluate.index_health                              向量索引与记录表是否同步（另一个体检）
```

两条"一条规则只能有一处定义"的证据：`planner.entry_from_record` 复用
`record_id_of` 与 `embedding_text`（不是各写一份）；`IndexingPipeline.search`
**一行转发** day064 的摄取流水线（同一个后端 + 同一个编码器实例），
所以查询侧与写入侧的口径不可能分叉。`ingest` 留下的
`embedding_calls`（逐条编码的基线）在本层被 `encoded` / `batches` / `cache_hits`
三个数超过——"省下了多少次编码"因此是一个**可比较**的数字。

**上游 day062（分块）**——本层唯一的输入形状，四个键：

```text
{"doc_id": <chunk_id>, "source": ..., "text": <原文片段>, "metadata": {…}}
fingerprint 取 metadata["fingerprint"]（day062 算好的），缺了才用 content_id(原文) 现算
vector_key  取 vector_key(身份, embedding_text(record))，编码文本优先 retrieval_text
```

`fingerprint` 用**原文**、`vector_key` 用**编码文本**（通常是
`retrieval_text` = 面包屑 + 原文）：两者回答的是两个问题，用同一段文本去算它们，
会让"只改标题面包屑"被误判成内容变化（多花一次编码），
或者让"内容变了但检索视图恰好没变"被漏判。

**下游 day066（检索器）**——交给它三样东西：

```text
清单 / 版本表          "这份索引是用哪个编码器的哪个版本建的"（/indexing/status、versions）
IndexingPipeline.search 检索入口（复用 day064 的编码与检索口径，不另建一条路）
verify / compare_with_store  "索引有没有漂移"（day071 的 RAG 评估也会直接复用）
```

day066 要加的是本层**刻意不做**的事：Top-K 之外的过滤与路由
（`vectorstore.filters` 的 `where` 是元数据筛选，不是访问控制）、
多路召回融合（day067 的混合检索）、重排序（day068 的交叉编码器）。
本层不做这些不是因为不需要，而是因为它们与"记账"无关——
一旦混进来，"这次为什么什么都没做"就再也答不出来了。

## 12. 复现

```bash
cd day065/源码/smart-research-agent
python scripts/indexing_demo.py      # 八节演示，全程离线、零网络
python -m pytest tests/test_indexing_*.py -q --no-cov -p no:cacheprovider
```

脚本不需要 `PYTHONPATH`（它自己把仓库根塞进 `sys.path`），结果同时打印到
stdout 并写入 `outputs/indexing_demo.txt`；落盘产物（快照 / 备份 / 恢复）
全部在 `outputs/indexing_demo_work/` 下（`outputs/` 在 `.gitignore` 里），
**不碰 `data/index`**。脚本连 `created_at` 都用注入的固定时钟，
因此两次运行的输出**逐字节相同**（实测 SHA256 一致）。
