# 向量库手册（day064 / M6-D3）

> 本文与代码**同源**：每一张表都能由 `registry.describe_backends()` /
> `metrics.describe_metrics()` / `filters.SUPPORTED_OPERATORS` /
> `evaluate.compare_backends()` 现场打印出来，每一个数字都由
> `scripts/vectorstore_demo.py` 实测（样本是本脚本自带的 6 条 8 维记录，
> 默认后端 `flat`，度量 `cosine`；编码器是脚本自带的查表实现）。
> 本机环境：Python 3.10.11、numpy 2.2.6、**未安装 faiss 与 chromadb**。

## 1. 一条链：从切好的块到"最像的 K 条"

```text
day062  ChunkSet.knowledge_records()   {"doc_id", "source", "text", "metadata"}
    │   pipeline.record_from_knowledge + embedding.embed
    ▼
VectorRecord    record_id(=chunk_id) + vector + text(原文) + metadata(扁平一层)
    │   backend.upsert
    ▼
VectorBackend   flat / faiss / chroma
    │   backend.query(vector, top_k, where=..., min_score=...)
    ▼
SearchResult    hits(SearchHit[]) + candidates + filter_applied
```

```text
errors.py         四族失败分家：环境 / 数据 / 调用 / 查询
metrics.py        唯一一处定义"谁更近"（score 越大越近）
types.py          VectorRecord / SearchHit / SearchResult / WriteReport / StoreInfo
filters.py        where 子句 → 谓词（12 个运算符）
base.py           六个抽象原语 + 全部公共行为（校验 / 过滤 / 阈值 / 稳定排序）
flat.py           纯 Python 暴力检索的参照实现（零依赖、逐位可复现）
faiss_backend.py  id 映射 + -1 填充 + 两文件快照 + 过滤必须预筛
chroma_backend.py 集合命名 + distance 口径 + 默认 EF 会偷偷下载模型
registry.py       后端选型 + 三段式报错（缺什么 / 怎么装 / 还能用什么）
evaluate.py       跨后端对账（ParityRow）+ 索引体检（index_health）
pipeline.py       knowledge_records → 逐条编码 → 写库 → 检索
```

本层最重要的一句边界（`scripts/vectorstore_demo.py` 里逐字打印）：

> 本层只回答'给一个向量、返回最像的 K 条'，再往前一步的问题
> （向量从哪来、命中之后怎么用）都属于别的模块

**一个向量库其实是两个库拼起来的**，这是理解三个后端差异最快的方式：

```text
向量索引   (int64 id, float32 向量)  → "谁离得最近"
记录表     id → (原文, 元数据)        → "命中的那条是什么"
```

Chroma 把两者做在一个集合里（`documents` / `metadatas` 跟着 `embeddings` 存），
所以 `native_metadata = True`；**FAISS 只做前者**（`IndexFlatIP` 连 id 都不存，
要用 `IndexIDMap2` 包一层），所以 `native_metadata = False`，原文与元数据
由 `faiss_backend` 自己扛，并落成一份 `<index>.records.json` 旁挂文件。

## 2. 三个后端的选型决策表

`registry.describe_backends()` 的十个字段里，选型只看前六个：

| 后端 | requires | install_hint | 落盘 | 元数据与向量同库 | 默认度量 |
|------|----------|--------------|------|-----------------|---------|
| `flat` | （无） | — | 支持（JSON 快照） | **是** | `cosine` |
| `faiss` | `faiss`, `numpy` | `pip install faiss-cpu` | 支持（索引文件 + `.records.json`） | 否 | `cosine` |
| `chroma` | `chromadb` | `pip install chromadb` | 支持（一个目录） | **是** | `cosine` |

三者的适用面与代价（`notes` 字段 + 各模块 docstring 的取舍表）：

| 后端 | 什么时候选它 | 代价 |
|------|-------------|------|
| `flat` | 小库、测试与 CI、需要"逐位可复现"的参照；缺依赖机器上的兜底 | 每次查询全量扫描 `O(N·d)`；整库必须进内存；`persist` 每次重写整份 JSON |
| `faiss` | 只要一个**精确**的向量索引，文本/元数据自己管；维度固定 | 构造时**必须**给 `dimension`；带 `where` 的检索要全量 `search` 之后再筛；字符串 id 要哈希成 int64 |
| `chroma` | 想让向量与元数据待在同一个集合、要原生 `where`、快速起步 | 依赖一个外部客户端/目录；自带 HNSW，默认参数会影响结果；集合名有一串硬约束 |

本机实测（demo 第 1 节）：

```text
flat     available=True
faiss    available=False   missing=['faiss']      install_hint='pip install faiss-cpu'
chroma   available=False   missing=['chromadb']   install_hint='pip install chromadb'
```

`available` 与 `missing` **一起给**：只给一个布尔值时，看到 `False`
还得自己去猜缺的是哪一个包。探测走 `importlib.util.find_spec`（只读 `sys.path`，
不真的 import）——**探测的代价与使用的代价是两件事**。

### 什么时候不要用向量库

1. **记录数很少（几百条）且查询简单**：直接线性扫一遍，或沿用 day062 的
   关键词/探针检索就够。向量库在这里只是多一层依赖。
2. **需要精确的结构化查询**：`WHERE x > 10 AND y IN (...)` 加聚合、事务、
   分页排序——那是关系库的活。本包的 `where` 只筛**元数据**，不是一个查询语言。
3. **元数据要做访问控制**：`where` 是筛选，不是权限（权限属于 day015 的工具权限层）。
   把"谁能看哪几条"交给调用方过滤，等于没有过滤。
4. **需要强一致 / 多副本 / 跨进程共享**：本层是单进程视角。
   Chroma 的服务端模式留到 day072 的部署一章。
5. **想要"语义相似"但没有真实 embedding**：`MockEmbedding` 没有语义，
   `CharNgramEmbedding` 只有字面重叠。向量从哪来是 day041/065 的问题，
   换个更贵的后端解决不了它。
6. **指望近似索引给出精确答案**：`IVF` / `HNSW` / `PQ` 都不保证全量排序，
   而本包的"先筛候选、再排序"恰恰依赖它（见第 7 节坑 6）。

## 3. 距离 / 相似度口径对照表

三个库用**三种符号约定**表达同一个概念，这一节是全篇最该背下来的部分：

```text
                 FAISS                                   Chroma（hnsw:space）
cosine   base 先做 L2 归一化，再 IndexFlatIP            d = 1 − cos(a, b)
         D = Σ(a·b)        越大越近                     越小越近
ip       直接 IndexFlatIP                                d = 1 − Σ(a·b)
         D = Σ(a·b)        越大越近                     越小越近
l2       IndexFlatL2                                     d = Σ(ai − bi)²
         D = Σ(ai − bi)²   越小越近                     越小越近
```

结论有两句：**FAISS 的"分数"越大越好，Chroma 的"距离"越小越好，
而两者都不叫 similarity。** 把 Chroma 的 `distances[0][0] = 0.12`
直接当相似度丢进 `if score > 0.8`，会让**最相似的记录被过滤掉**，且不报错。

本包因此只保留两套口径，并用名字钉死：

```text
score      统一口径：**越大越近**（排序 / 阈值 / 报告只用它）
distance   原始口径：**越小越近**（与 Chroma 的 distances 同名同义）
```

`SearchHit` 两个都带（报告需要前者，与外部库对账需要后者），换算只写在
`metrics.similarity_from_distance` **一处**：

```text
cosine / ip   score = 1 − distance      （Chroma 就是这么定义它的距离的）
l2            score = −distance         （平方 L2 越小越近，取负即"越大越近"）
```

FAISS 适配层多一步（`faiss_backend._to_score`）：`IndexFlatIP` 的 `D`
是相似度，先折成 `distance = 1 − D` 再交给上面那个函数；`IndexFlatL2`
的 `D` 已经是距离，直接交进去。"两个库的距离方向相反"这件事因此
只出现在一个 if 里，而不是散落在每一处比较中。

三个度量本身（`metrics.METRICS = ("cosine", "ip", "l2")`）：

| 度量 | 值域 | 需要归一化 | 什么时候用 |
|------|------|-----------|-----------|
| `cosine` | `[-1, 1]` | 是（本包自动做） | **默认**。文本 embedding 的标准用法，阈值可解释 |
| `ip` | 无界 | 否 | 已归一化时与 cosine 等价；未归一化时"长向量占优" |
| `l2` | `[0, ∞)` | 否 | 图像/几何特征；对文本要先确认提供方的训练目标 |

别名表把各处写惯的写法收敛到三个规范名（`cos` / `angular` → `cosine`，
`dot` / `inner_product` / `inner-product` → `ip`，`euclidean` / `sqeuclidean` /
`squared_l2` → `l2`）。未知度量直接报错并列出可选值——**静默当成默认的
cosine 会让"排序不太对"变成一个查不出原因的问题。**

实测（demo 第 3 节，同一个查询 `q_axis`、同一批 6 条记录）：

| 度量 | top-1 | top-2 | top-3 | 说明 |
|------|-------|-------|-------|------|
| `cosine` | 3f5a…（1.0000） | a1f3…（1.0000） | 5c9d…（0.8000） | 前两条**并列** |
| `ip` | a1f3…（**10.0000**） | 3f5a…（1.0000） | 5c9d…（0.8000） | 模长 10 倍的那条被顶到第一 |
| `l2` | 3f5a…（−0.000000） | 5c9d…（−0.400000） | b7e2…（−0.800000） | 又是另一批名次 |

`-0.000000` 是**负零**（`0.0` 取负的结果），不是错误。
这组数字就是"**换度量就是换答案**"的证据：同一份数据、同一个查询、
同一个实现，只改了度量名。

## 4. `where` 语法表（12 个运算符）

本包采用 **Chroma 的语法**作为规范语法（它已经是一份写下来的、
被真实实现的规格），由 `filters.py` 编译成谓词，三个后端共用同一份语义。

| 运算符 | 含义 | 值的要求 |
|--------|------|---------|
| `$eq` | 等于 | 标量（**直接写值等价于 `$eq`**） |
| `$ne` | 不等于 | 标量（字段不存在也算"不等于"） |
| `$gt` / `$gte` | 大于 / 大于等于 | 标量 |
| `$lt` / `$lte` | 小于 / 小于等于 | 标量 |
| `$in` / `$nin` | 在 / 不在集合内 | **非空**列表（空列表会让命中集恒为空，直接拒） |
| `$contains` / `$not_contains` | 数组字段是否含某元素 | 标量；字段必须是数组，否则判为不命中 |
| `$and` / `$or` | 逻辑组合 | **非空**的子句列表，且必须单独占一层 |

四条必须记住的语义（都写在 `filters.py` 的 docstring 里）：

1. **`$nin` 对"字段不存在"也成立**（"不在集合内"对一个不存在的值也成立）；
   要表达"存在且不在内"请写 `$and` 组合。
2. **类型不可比的比较判为不命中，而不是抛错**：一份库里 `page` 既有 `int`
   又有 `"12"` 时，`{"$gt": 10}` 遇到字符串会触发 `TypeError`；抛错会让
   **一次查询因为一条脏记录整体失败**，所以取舍是"查询的可用性优先"。
3. **`True == 1`** 是 Python 的语义：`{"flag": {"$eq": 1}}` 会命中 `flag=True`。
   不要用同一个键混存布尔值与整数——这条只能靠约定。
4. **`$and` / `$or` 不能与字段条件混写在同一个字典里**，编译期直接拒
   （`{"a": 1, "$or": [...]}` 的语义不确定）。

其余会在编译期报 `FilterError` 的写法：未知运算符、`$in` 给空列表、
同一字段写多个运算符、字段条件是空字典 `{"field": {}}`、
`$and` 的值不是非空列表。

实测（demo 第 5 节，6 条记录的 `strategy` 分布为
recursive×2、fixed×1、structural×2、semantic×1）：

| 子句 | `candidates` | `count` | `filter_applied` |
|------|-------------|---------|------------------|
| `{"strategy": {"$in": ["recursive", "fixed"]}}` | 3 | 3 | True |
| `{"$and": [{"strategy": "structural"}, {"oversized": True}]}` | 1 | 1 | True |
| `{"tags": {"$contains": "cost"}}` | 2 | 2 | True |
| `{"strategy": {"$in": ["hybrid"]}}`（库里没有这个值） | 0 | 0 | True |
| 无过滤，但 `min_score=1.5` | 6 | 0 | False |

`candidates` 与 `count` 必须**一起**看，因为三种情形在返回体里长得一样：

```text
candidates=0 且 filter_applied=True    → 过滤器把所有记录都排除了
candidates=0 且 filter_applied=False   → 库是空的
candidates=6 而 count=0                → 阈值把命中切掉了（数据在，只是不够像）
```

## 5. 写入报告：`WriteReport` 与 `IngestReport`

`WriteReport` 把"一次写入"分成五个数——**只把它们分开，
"这次为什么什么都没变"才能被回答**：

```text
added      新插入
updated    同一 id 被覆盖（内容确实变了）
unchanged  同一 id、内容逐位相同 → 没有改动任何东西
skipped    被调用方或护栏拒绝
removed    delete 删掉的条数
```

派生量：`written = added + updated`、`total = added + updated + unchanged + skipped`、
`changed = added + updated + removed`。`unchanged` 的记录**不写回库**
（`flat` / `faiss` / `chroma` 三处都 `continue`），这一条是 day065 的地基：
day065 要靠"库里那个对象没被换过"判断"这一块的向量不需要重算"。

`IngestReport` 在它之上再加两件东西：**编码成本**与**逐条原因台账**：

```text
seen / written / unchanged / skipped / failed / embedding_calls
backend / metric / dimension / failures[(record_id, 原因)]
恒等式：seen == written + unchanged + skipped + failed
```

其中 `skipped` 与 `failed` 是两族：`skipped` 是数据问题（缺 `doc_id`、
`text` 与 `retrieval_text` 都空白），`failed` 是环境与匹配问题（维度不一致、
整批被后端拒绝）。不认识的异常**一律上抛、不记进报告**——
把真 bug 记成一行 `failed=1`，它永远不会被修。

实测（demo 第 2 节，`flat` + 查表编码器）：

```text
第一次 ingest   seen=6 written=6 unchanged=0 skipped=0 failed=0 embedding_calls=6
重放同一批      seen=6 written=0 unchanged=6 skipped=0 failed=0 embedding_calls=6
库状态始终      flat | cosine | 8d | 6 条
```

两个数字合起来才说完一件事：**库一个字节都没变，但 6 次编码真的发生了**
——day064 是**逐条编码、没有向量缓存**（把 `embedding_calls` 用
`len(records)` 顶替，day065 的"缓存省下了多少"就永远答不出来）。
删除侧的对应纪律是：`delete` 返回**真正删掉的条数**（Chroma 靠删除前后
两次 `count()` 求差），请求删 5 条、实际删 0 条是一个必须被看见的事实。

## 6. 持久化与备份

三者都能落盘，但"落盘"的含义完全不同，备份动作也因此不同：

| 后端 | 落盘形态 | 备份什么 | 读出时核对什么 |
|------|---------|---------|---------------|
| `flat` | **一份 JSON 快照** | 一个文件 | `version` / `metric` / `dimension` / `count` 与 `records` 是否自相矛盾 |
| `faiss` | **两个文件**：索引二进制 + `<path>.records.json` | 两个文件必须**成对**、来自同一次 `persist()` | 上面四项 + `index_ntotal` + `id_map` 集合 |
| `chroma` | **一个目录**（`PersistentClient(path=...)` 每写一条就落盘） | 整个目录 | 集合句柄与维度（由数据反推） |

`flat` 的快照长这样（`SNAPSHOT_VERSION = 1`）：

```json
{"version": 1, "backend": "flat", "metric": "cosine", "dimension": 8, "count": 6,
 "records": [{"record_id": "...", "vector": [...], "text": "...", "metadata": {...}}]}
```

三条刻意的决定：**整份重写**（快照要么完整要么不存在）、**自动建目录**、
**记录按 id 升序写出**（同一次状态的两次快照逐字节相同，diff 里不会全是顺序变化）。
JSON 而不是 pickle，是因为快照的价值在于**能被人打开看一眼**。
实测：6 条 8 维记录的这份快照是 **3773 字节**；用 `path=` 新建的实例
自动 `load`，`ids()` 与原实例**逐条相同**。

`faiss` 的两文件是"一个向量库其实是两个库"最直白的写法：`faiss.write_index`
只能写它自己的二进制格式，原文与元数据**根本不在索引里**，
于是它们被写进旁挂文件（后缀 `RECORDS_SUFFIX = ".records.json"`）。
`load` 会强制两者成对存在，并核对 `index_ntotal` 与 `id_map` 集合——
把另一个库的索引文件拷过来会在这里被拦下。

`chroma` 最容易误判：`persist()` **不写快照**，它只是"确认并回显那个目录"
（数据从头就在盘上）；内存客户端（构造时 `path` 为空）调 `persist()`
直接抛 `BackendUnavailable`；目录一旦绑定到客户端就不能改，
想换目录只能用新目录新建实例。

## 7. 六条踩坑清单

1. **Chroma 建集合忘传 `embedding_function=None` → 会触发模型下载。**
   它默认的 embedding function 是 Sentence-Transformers 的
   `all-MiniLM-L6-v2`（本地运行，但**第一次用会去下载权重**）。
   本包永远自己提供向量，因此 `_acquire_collection` 显式传
   `embedding_function=None` 且只传 `configuration`（与 collection metadata
   同时设 `hnsw.space` 会被真库判为冲突）。离线环境里的表现是"卡住"，
   受限网络里是一个与业务无关的超时。
2. **FAISS 的 `-1` 是"此处没有结果"的保留值。** `search` 返回的 `I` 在
   结果不足时用 `-1` 填充；`from_int64_id(-1)` 抛 `VectorError`。
   同理，本包生成的 int64 id 值域取 `[1, 2**63 − 2]`，**避开 `-1` 与 `0`**。
   id 映射用哈希（`1 + sha256(id)[:8] % (2**63−2)`）而不是自增号：
   自增号在进程重启后会重排，而索引文件一旦与映射表对不上，
   检索会**静默返回空结果**；理论上的哈希撞号必须报错而不是静默覆盖。
3. **换 `metric` 必须重新标定阈值。** `score` 的口径随度量变：
   `l2` 的分数是 `≤ 0` 的负数，而两个度量共用一个 `vector_min_score`。
   因此 `settings.vector_min_score` 的默认值是 `None`（不设阈值）而不是 `0`
   ——`0` 在 `l2` 下会把**全部命中**切掉。同时，度量在构造后端时定死、
   **不允许运行期切换**；`flat.load` / `faiss.load` 会核对快照里的 `metric`，
   不一致抛 `VectorError`，出路是"用快照的 metric 新建实例"。
4. **`delete` 的两种方式不能同时给。** `delete(ids=[...], where={...})`
   抛 `RecordError`——同时给会让"到底删了什么"变得不确定；两者都不给
   也不删任何东西（同样报错）。另外 `$nin` 会把"字段不存在"也算命中，
   用它做删除条件会删到你想不到的那些记录。
5. **换了 embedding 提供方就要重建库，不要截断或补零。** 维度在
   `flat` / `chroma` 上是**第一次写入时定下**的，在 `faiss` 上
   **构造时就必须给**（索引的 `d` 建立后不可变）。之后的任何一次不同维度
   写入或查询都抛 `VectorError`；把 768 维截成 384 维不会报错，
   只会让排序**静默失去意义**。
6. **"先取 top-k 再过滤"是错的，而且不报错。** FAISS 没有任何元数据过滤
   能力，唯一正确的做法是"先按条件筛出候选，再在候选里排序取前 K"。
   反过来写会让**返回条数少于 K，尽管库里还有符合条件的记录排在更后面**
   ——症状只是"加了过滤之后结果变少了"。代价是真实的：`IndexFlat` 是精确
   索引，所以可以"全量 `search` + 过滤"；换成 `IVF`/`HNSW` 之后
   "全量"等于放弃索引的意义，那时只剩"过采样 + 承认近似"一条路，
   **"过滤"与"近似"天生互相消耗**。

补充两条与数据形状有关、同样"不报错但结果错"的：

- **元数据的类型比 Python 允许的更严**：只接受 `str` / `int` / `float` /
  `bool` / **同类型且非空**的标量数组；不接受嵌套 `dict`、`None`、空数组、
  混合类型数组。这条约束在**构造记录的那一刻**就报 `RecordError`
  （`assert_metadata`），否则会出现"本地 flat 跑得好好的、换 Chroma 就炸"。
  注意判断顺序：`bool` 必须排在 `int` 之前（`isinstance(True, int)` 为真）。
- **Chroma 的向量存成 float32**，读回来必然带量化误差。因此判定"记录没变"
  用的是相对/绝对容差 `VECTOR_EQUAL_TOLERANCE = 1e-5` 而不是逐位比较；
  若按逐位比，重放同一批永远被报成 `updated`，`unchanged` 这一态
  在 Chroma 上就**永远不可达**——而它正是 day065 要用的那个数。
  分数侧的对应容差是 `SCORE_TOLERANCE = 1e-6`，且**只用于分数、不用于名次**。

## 8. 对账与体检

同一批记录、同一个查询，几个后端给出同一份排名**不是自然事实**，
而是三处独立约定恰好对齐的结果（公式 / 距离方向 / 平局断法）。
对账的产物是 `ParityRow`：

```text
身份   backend / metric / available
规模   count / top_k
结论   agreed（逐位相同的前缀长度）/ overlap（recall@k）/ first_divergence / note
```

`agreed` 是**前缀**长度而不是"相同位置的个数"：一旦第 3 名不同，
第 6 名的"相同"已经不意味着同一件事。`overlap` 是集合口径，
用来把"选错了"与"排错了"分开——`overlap == 1.0` 而 `agreed == 0`
是一种真实且危险的状态。两者各回答一半问题，只用其中一个都会漏。

实测（demo 第 6 节，参照实现是脚本自带的、只用 `math` 的独立实现）：

```text
compare_backends（flat + faiss + chroma × cosine + ip）：
  flat   | cosine | 一致 3/3 | 交叠 100.0% | first_divergence=''
  flat   | ip     | 一致 3/3 | 交叠 100.0% | first_divergence=''
  faiss  | cosine | 不可用 | first_divergence='后端不可用'
  chroma | cosine | 不可用 | first_divergence='后端不可用'
  （ip 下同理各一行）
verify_parity：ok=False checked=6 available=2
```

`ok=False` **不等于对账失败**：不可用的后端也会被记成一条 failure。
这正是"装了 faiss 的机器与没装的机器，报告必须长得一样"的实现方式——
差异只表现为一个 `available` 布尔与一句原因，而不是少了几行报告。
真正把"不可用"折成报告行的是 `compare_backends`/`compare_one`：
只有 `BackendUnavailable` / `ImportError` 被收留，其它异常一律上抛
（把真 bug 记成一行会让对账在没有比过任何东西的情况下变绿）。

只换度量时不比对账，看 `compare_metrics`（基线是 `cosine`）：

```text
flat | cosine | 一致 3/3 | 交叠 100.0% | 逐位一致
flat | ip     | 一致 0/3 | 交叠 100.0% | 第 1 名：期望 '3f5a91c0d27be684'，实际 'a1f34c8b62e0d975'
flat | l2     | 一致 1/3 | 交叠  66.7% | 第 2 名：期望 'a1f34c8b62e0d975'，实际 '5c9d0e13ab7f2468'
```

`ip` 那一行是同一条"换个名字、换掉答案"的结论：**选型记录里最该贴的
就是这一行**。另外 `compare_metrics` 有一条容易踩的前提——
`make_record` 只在 `cosine` 下做归一化，所以用 `cosine` 构造的数据
去比 `ip` 与 `cosine` 会看到**完全相同**的排名；要看见差别，
必须用 `metric="ip"` 构造（demo 与测试都是这么做的）。

体检走 `evaluate.index_health(backend)`，四项各盯一种不同步：
`count` 与 `ids()` 长度是否一致、`ids()` 有无重复、每条记录的维度是否
与库一致、`ids()` 里每一个都能被 `get_many` 取回（**向量索引里有、
记录表里没有**是"两个库"最典型的事故形态，症状只是"结果少了一条"）。

## 9. 配置项与端点

配置项（`config.Settings` 的 `vector_*` 字段，也是 `create_backend` 的基线）：

```python
vector_backend: str = "flat"                 # flat / faiss / chroma
vector_metric: str = "cosine"                # cosine / ip / l2（走别名表）
vector_default_top_k: int = 5                # 默认检索深度
vector_min_score: float | None = None        # 阈值：None = 不设（不是 0）
vector_collection: str = "smart_research_agent"   # 只有 chroma 用得到
vector_persist_path: str = ""                # 空串 = 明确不落盘
```

优先级是三级：**调用点显式参数 > `settings`（项目级基线）> 后端/策略默认值**。
`create_backend(**overrides)` 的短名到配置字段是一张翻译契约
（`backend` / `metric` / `dimension` / `path` / `collection`）；
它**明确拒收** `top_k` 与 `min_score`——它们是**检索**参数，
而 `create_backend` 返回的是一个后端实例，后端没有"默认取几条"这个概念。
那两级优先级属于 `VectorIngestPipeline`（`调用参数 > 构造参数 > settings`）。

`TOP_K` 有两条边界：`< 1` 与 `> MAX_TOP_K`（**1000**）都抛 `FilterError`；
默认深度 `DEFAULT_TOP_K = 5`（不是 3，因为 day062 的探针实验说明
top-3 的真实差距常来自检索深度不够）。`UNKNOWN_DIMENSION = 0`
是"维度未定"的哨兵值，所以 `dimension` 从 1 起计数。

六个端点（`/vectorstore/*`，与本层同日的 API 层交付，形态照 day062 的 `/chunking/*`）：

| 端点 | 方法 | 回答什么 |
|------|------|---------|
| `/vectorstore/backends` | GET | `describe_backends()` + `describe_metrics()` + 当前默认后端 |
| `/vectorstore/upsert` | POST | 收 `knowledge_records` 形状的列表 → 编码 → 写库 → 返回 `IngestReport` |
| `/vectorstore/search` | POST | 查询文本 → 编码 → 检索 → 返回 `SearchResult`（含 `candidates` 与 `filter_applied`） |
| `/vectorstore/stats` | GET | 当前库状态（`StoreInfo`）+ 本实例最近一次写入报告 + 编码器信息 |
| `/vectorstore/records` | DELETE | 按 `ids` 或 `where` 删除 → 返回真正删掉的条数 |
| `/vectorstore/compare` | POST | 一批记录 + 一个查询做跨后端对账 → 返回 `ParityRow` 列表 |

状态挂在应用上：后端通过 `request.app.state.vector_store` 取；缺 faiss/chromadb
的机器上 `/vectorstore/compare` **仍然返回 200**，只是那两个后端的行
`available=False`。`/vectorstore/search` 里非法的 `where` 映射成 **HTTP 400**
（消息里带着支持的运算符列表）；`upsert` / `delete` 会改内存状态，
但**内存后端的写入只活在这个应用实例里**。

## 10. 与 day062 / day065 的接缝

**上游 day062（分块）**——本层的唯一输入形状：

```text
{"doc_id": <chunk_id>, "source": ..., "text": <原文片段>, "metadata": {...}}
record_id  取 doc_id（就是 chunk_id，16 位十六进制），跨模块的契约，不做改写
text       取**原文片段**（命中之后要还给用户看的就是它）
向量化用   metadata["retrieval_text"]（面包屑 + 正文），缺了才回落到 text
```

"存的"与"编码的"不是同一份文本，这是 day062 留下的约定（`EMBEDDING_TEXT_FIELD
= "retrieval_text"` / `FALLBACK_TEXT_FIELD = "text"` / `RECORD_ID_FIELD = "doc_id"`）。
`embedding_input_field(record)` 把这件事变成**可核对的事实**：
报告与调试能直接问出"这一条用的是哪个字段"。代价写在明面上：
`metadata` 原样保留（含 `retrieval_text`），存储约翻一倍——这是刻意保留的，
它是"存的和编码的不一样"的证据。

注意 day062 的 `chunk_id`（谁）与 `fingerprint`（哪段内容）是两个字段：
`chunk_id` 进 `record_id`，`fingerprint` 只留在 `metadata` 里。
把两者混成一个，会出现"要么去重失效、要么块互相覆盖"，而两种失效都不报错。

**下游 day065（增量索引）**——本层交出三件它要用的地基：

```text
IngestReport.embedding_calls   精确到"这次实际编码了几段文本"（逐条编码，是**基线**）
WriteReport.unchanged          "这次写入其实什么都没做"的唯一证据
evaluate.index_health          向量索引与记录表是否同步的四项体检
```

day065 在这三条之上加批量编码、向量缓存与增量对账；它的 `embedding_calls`
会显著小于条数，而"显著小了多少"只有在 day064 把逐条编码的基线钉死之后
才是一个**可比较的数字**。再往后：day066 的检索器在本层之上加
Top-K 之外的过滤与路由，day067 的混合检索另建一层（本层不做倒排/稀疏），
day071 的 RAG 评估会直接复用 `evaluate.py` 的对账与体检结构。

## 11. 已知限制与明确排除的用途

`base.VECTORSTORE_LIMITATIONS`（四条，写在代码里）：

1. 只做暴力/近似最近邻检索，**不提供倒排索引与稀疏检索**（day067 的混合检索另建一层）；
2. **不带重排序**：命中的顺序完全由向量度量决定（day068 加交叉编码器）；
3. **不管批量编码与向量缓存**：本层接收算好的向量（day065 的 indexing 包负责）；
4. **不做权限过滤**：`where` 是元数据筛选，不是访问控制（day015 的工具权限层）。

`base.VECTORSTORE_OUT_OF_SCOPE`（四条，写下来避免"这不算 bug"的争论）：

1. 分布式分片与副本：本层是单进程视角，多副本一致性不在课程范围；
2. 训练型索引（PQ/OPQ 的码本训练）：需要样本量与离线训练，属于生产调优；
3. GPU 索引：FAISS 的 GPU 索引不支持 `write_index`，序列化路径与 CPU 不同；
4. 跨语言/跨进程的实时同步：Chroma 的服务端模式留到 day072 的部署一章。

还有两处"刻意不报错"的取舍值得先知道：**类型不可比的比较判为不命中**
（见第 4 节取舍二），以及 `base.query` 遇到"记录表里取不回 id"时
**跳过并继续**（读路径不该因为一条脏数据整体失败）。后者必须可见，
所以 `evaluate.index_health` 专门查它。

## 12. 复现

```bash
cd day064/源码/smart-research-agent
python scripts/vectorstore_demo.py            # 八节演示，全程离线、零网络
python -m pytest tests/test_vectorstore_*.py -q --no-cov -p no:cacheprovider
```

演示脚本不需要 `PYTHONPATH`（它自己把仓库根塞进 `sys.path`），
结果同时打印到 stdout 并写入 `outputs/vectorstore_demo.txt`
（`logs/` 与 `outputs/` 都在 `.gitignore` 里）。现场 `--collect-only`
收集到 **407** 个用例（`tests/test_vectorstore_*.py`）。
