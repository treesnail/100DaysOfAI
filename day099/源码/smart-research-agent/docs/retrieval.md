# 检索器手册（day066 / M6-D5）

> 本文与代码**同源**：每一条结论都标了落地位置（`retrieval/xxx.py::函数名`），
> 都能在 `smart_research_agent/retrieval/`、`vectorstore/{filters,base,types}.py`、
> `indexing/manifest.py`、`config.py` 的 `retrieval_*` 组里逐字找到。
>
> **本文的数字只有两个来源**，其余一律写"以你实际运行时输出为准"：
>
> ```text
> 代码里写死的常量    DEFAULT_FETCH_MULTIPLIER = 3、MAX_FETCH_K = 1000、
>                     DEFAULT_MIN_HIT_CHARS = 32、retrieval_max_context_chars = 2400…
> 由它们算出的算术    top_k=5 → 深度 ceil(5 × 3) = 15；top_k=500 → 本应 1500、封顶 1000
> ```
>
> 少量标注"本机实测"的行来自本文作者的一次探针运行（`FlatVectorStore` +
> `CharNgramEmbedding(dimension=64)`、5 条记录、离线零网络），**它不是课程产物**，
> 只是用来说明形状；行为相关的正式数字请以本层同日的离线演示
> `scripts/retrieval_demo.py` 与 `python -m pytest tests/test_retrieval_*.py`
> 的实际输出为准（脚本落地后即可复现）。
>
> 本层的落地状态（`retrieval/__init__.py` 的原文）："导出范围（**七个模块全部落地**）：
> `errors` / `types` / `filters` / `retriever` / `routing` / `context` / `pipeline`"。
> 端点层（`api/routes.py` 的 `/retrieval/*`）与本层同日交付，但本文只按
> **占位写法**引用它（"见该文件"），**不复述任何响应字段**。

## 1. 一句话定位：检索器与向量库的分工

```text
向量库    给一个向量，谁最近                    → 排序问题
检索器    给一句话，取回哪几条、                 → 决策问题
          为什么是这几条、为什么只有这几条
```

这四行是 `retrieval/retriever.py` 模块 docstring 的开头，也是本课的主题。
向量库**答不了**后半句，因为它的输入里没有"一句话"：编码器在它之外，
阈值、多样性、深度、路由都在它之外。反过来，检索器**不碰向量**——
它不认识度量公式、不做归一化、不改排序键，一切"谁更近"的判断仍然是
`vectorstore/metrics.py` 的 `score`（统一口径：**越大越近**）。

### 模块地图（每个模块回答一个问题）

```text
errors.py     这一层会怎么失败（按"谁去修"分三族：调用方 / 索引运维 / 打包配置）
types.py      四副形状：查询、命中、索引状态、结果；时间范围的三条约定
filters.py    时间范围怎么变成 where；where 与 time_range 撞车为什么必须报错
retriever.py  九步流水线：Top-K + 深度 + 过滤 + 阈值 + 多样性 + 空结果诊断 + 漂移告警
routing.py    多索引选路：一句查询该去哪个库，写成显式的一层而不是隐式的 if
context.py    上下文打包与引用标记：预算、单条截断、整包丢尾、[n] 编号
pipeline.py   检索 → 打包 → 生成 → 引用（检索为空时绝不调 LLM）
```

### 九步流水线（`Retriever.retrieve` 的固定顺序）

| 步 | 做什么 | 落地位置 | 它留下的可见痕迹 |
|----|--------|---------|-----------------|
| 1 | 规范化查询：`str → RetrievalQuery`，`None` 字段填默认值 | `retriever.py::Retriever._normalize_query` | `result.query`（已解析默认值的那一份） |
| 2 | 索引体检：条数 + 清单漂移 | `retriever.py::Retriever._inspect_index` | `result.index_state`、`notes` |
| 3 | 算召回深度 | `retriever.py::Retriever._fetch_depth` | `result.fetch_k`、封顶时的 `notes` |
| 4 | 编码查询 + 三项校验 | `retriever.py::Retriever._encode_query` | 违规抛 `IndexStateError` |
| 5 | 库侧过滤 | `filters.py::combine_where` → `backend.query` | `result.candidates`、`filter_applied` |
| 6 | 阈值落刀（**在本层**） | `retriever.py::_apply_threshold` | `dropped_below_threshold` |
| 7 | 多样性裁剪 | `retriever.py::_apply_diversity` | `dropped_by_diversity` |
| 8 | 重排名次 + 截断到 `top_k` | `retriever.py::_rerank` | `dropped_by_top_k`、连续的 `rank` |
| 9 | 诊断与装配 | `retriever.py::_diagnose` | `empty_reason`、`latency_ms`、`notes` |

第 3 步刻意排在体检之前：这样"库是空的"这条早退路径也能报告"本来打算取多深"
（一个 `fetch_k=0` 的空结果会让人以为这次没设深度，而实际上深度是算过的，
见 `retriever.py` 第 3 步的注释）。

### 三条贯穿全包的纪律（`retrieval/__init__.py` 原文的意思）

1. **排序与阈值只有一个口径**：`score` 越大越近，阈值用同一个 `>=`、
   在**本层只落一次刀**；
2. **"条数少于期望"必须是可回答的**：三个 `dropped_*` 数字 + `empty_reason`
   只给**一个**原因；
3. **检索器不认识语料，只认识索引状态**：版本号与漂移跟着每条结果走，
   **不静默降级**。

### 已知边界与明确不做的事

`types.RETRIEVAL_LIMITATIONS`（四条，写在代码里）：

| 边界 | 原文要点 |
|------|---------|
| 单路 | 只做单路向量检索：不实现 BM25、不做多路融合与去重（day067 的混合检索另建一层） |
| 无重排 | 不带重排序：命中的顺序完全由向量度量决定（day068 加交叉编码器） |
| 阈值在本层 | 阈值在本层落刀而不是交给库：因为"切掉了几条"这个数字必须能被报告 |
| 无权限 | 不做权限过滤：`where` 是元数据筛选，不是访问控制（那属于 day015 的工具权限层） |

`types.RETRIEVAL_OUT_OF_SCOPE`（四条，写下来避免"这不算 bug"的争论）：
查询改写与扩展、学习型排序（LTR）、跨索引的分布式检索（本层的路由是**业务选路**，
不是分片聚合）、在线索引更新（检索是读路径，写入由 day065 的 `indexing` 包负责）。

## 2. 五个概念：查询、深度、过滤、阈值、多样性

一次检索里能被调的只有五个旋钮，它们各自改变一件可以被单独讨论的事：

| 概念 | 长什么样 | 落地位置 | 默认值 / 口径 |
|------|---------|---------|--------------|
| 查询 | `RetrievalQuery.text` | `types.py::RetrievalQuery` | `strip()` 后必须非空（否则 `QueryError`）；传 `str` 会自动包成 `RetrievalQuery` |
| 深度 | `fetch_k` | `retriever.py::Retriever._fetch_depth` | `max(top_k, ceil(top_k × 3))`，封顶 `MAX_FETCH_K = 1000` |
| 过滤 | `where` + `time_range` | `filters.py::combine_where` | 两者都空 → `None`（`None` 与 `{}` 都表示"没有过滤"） |
| 阈值 | `min_score` | `retriever.py::_apply_threshold` | 默认 `None` = **不设阈值**（不是 0） |
| 多样性 | `max_per_doc` | `retriever.py::_apply_diversity` | 查询里省略 → 取检索器默认；检索器默认来自 `retrieval_max_per_doc = 0` = **不限** |

### 查询：`RetrievalQuery` 的校验全在构造期

`types.py::RetrievalQuery.__post_init__` 当场拒掉五类调用方错误，**不留到运行时**：

```text
空查询（strip 后为空）    → QueryError：空白串会编码成一个固定向量，
                            于是每次查询都返回同一批"最像空白的记录"
top_k < 1 或 > 1000       → QueryError（上限就是 vectorstore.MAX_TOP_K）
fetch_k < top_k           → QueryError：那些"本来能进 top_k"的候选根本不会被取回来
min_score 非有限数        → QueryError：nan 与任何分数比较都返回 False（一条都不留）
max_per_doc < 1           → QueryError：查询里不允许写 0（"最多 0 条"不是一个请求）
```

`where` 的**语法**在构造期就由 `filters.py::validate_where` 调
`vectorstore.compile_filter` 校验（把 `FilterError` 包成 `QueryError`，
并附上 `vectorstore/filters.py::SUPPORTED_OPERATORS` 的 12 个运算符清单）；
而 `where` 与 `time_range` 的**字段冲突**要等到 `combine_where` 才判——
那里同时握着两者，规则只写一份（见第 3 节）。

### 深度：为什么默认要"过取"三倍

```text
没有显式 fetch_k  → ceil(top_k × settings.retrieval_fetch_multiplier)
两条路都封顶      → MAX_FETCH_K = 1000，并在 notes 里说明"已封顶"
```

`DEFAULT_FETCH_MULTIPLIER = 3` 的理由写在 `types.py` 的常量注释里：
三道减法（阈值 / 多样性 / 截断）都要吃名额，而最狠的一道
（`max_per_doc=1` 且同一文档占满 top-k）需要约 3 倍深度才凑得够结果；
再大只是白扫。算一下量级：`retrieval_top_k = 5` → `fetch_k = 15`。

封顶不是沉默的（`retriever.py::Retriever._fetch_depth`）：`top_k=500` 时
本应取 1500 条，实际封顶到 1000，`notes` 里会出现一句
"召回深度按 3 倍本应取 1500 条，已封顶到 MAX_FETCH_K=1000"。
**`MAX_FETCH_K` 直接复用 `vectorstore.base.MAX_TOP_K`**（`= MAX_TOP_K`），
两个数字一旦分开，就会出现"设置里写 2000、实际只取了 1000"这种自相矛盾的配置。

一个边界情形被显式收敛而不是报错（`_normalize_query`）：调用方只给了
`fetch_k=2` 而默认 `top_k=3` 时，把本次 `top_k` **收敛为 2** 并记一条 `notes`
——"深度 ≥ 条数"这条不变量必须成立，而调用方显然想要那个更小的深度。
**一次静默的改写必须可见**，所以它进 `notes`。

### 过滤：`where` + `time_range` 合成一份子句

`filters.py::combine_where` 的四种形态（`none` 与"空字典"在报告里必须能区分）：

```text
两者都空      → None（"没有过滤"）
只有时间范围  → 时间子句
只有 where    → where 原样（不做包装：包装只会给报告加一层噪声）
都有          → {"$and": [where, 时间子句]}
```

`filters.py::unknown_filter_fields` 是**拼写错误的探测器**，它能发现
`{"stratgey": "structural"}`（整个库里都没这个键），但**发现不了**
`{"strategy": "structual"}`（键对、值错）与格式写错的时间字符串。
因此它只出现在 `Retriever.explain()` 的"字段拼写检查"一行里，
作为**提示**而绝不参与 `empty_reason` 的判定
（详见 `filters.py::unknown_filter_fields` 的 docstring）。

### 阈值：为什么必须在**检索器**里落刀

`VectorBackend.query` 也接受 `min_score`，但 `Retriever` **故意不传**它：

```text
交给库过滤   candidates=5、hits=2      → "切掉了几条"这个数字拿不到
在本层落刀   candidates=5、dropped=3   → 数字拿得到（正是 day067 融合要用的）
```

day067 的融合需要按通道判断"**这一路是不是一条都没活下来**"——
一路被阈值清零与一路本来就没召回，对融合是两件完全不同的事
（前者降权、后者可以补其他路的权重）。语义与 `VectorBackend.query(min_score=...)`
完全一致：**同一个 `>=`、同一个分数口径**，因此两种写法的结果集合逐条相同，
差别只在"这个数字有没有被记下来"（`retriever.py::_apply_threshold` 的 docstring）。

用 `>=` 而不是 `>` 不是随手写的：阈值**恰好等于**某条命中的分数这种情形必然出现
（同一个确定性编码器对同一段文本给出逐位相同的向量）。

### 多样性：按文档分组，每组至少留一条

`retriever.py::_apply_diversity` 的两个刻意决定：

```text
key = metadata[doc_id_field]（默认 parent_doc_id，见 types.DEFAULT_DOC_ID_FIELD）
    判据是"同一份文档刷屏"，而 record_id 永远唯一（用 record_id 治不了任何东西）
缺 doc_id 的记录自成一类（键为 ""）  它不属于任何文档，因此不该被别人的上限挤掉；
                                    代价是一批没有 parent 的记录会互相挤，规则对所有人一致
```

裁剪之后 `_rerank` 会**重排名次**：阈值切掉第 2 名之后，后端的 `rank` 会变成
0、1、3、4——一个带空洞的名次列表不能用来索引"第 3 条"，
因此最终结果里的 `rank` 永远从 0 连续。排序规则是
"分数降序 + `record_id` 升序"（与 `vectorstore/types.py::sort_hits` 同一纪律），
**并且在本层再执行一次**：这一层的上限是"任何实现 `VectorBackend` 的后端"，
不确定性往往不是来自算法，而是来自没写下来的排序规则。

### 多索引选路：`StoreRouter`

`routing.py::StoreRouter` 把"这句话该去哪个库"写成一层显式结构，三条规则：

```text
route 为空 + 设了 default   → 用 default（RouteDecision.matched=False，说明"是路由器选的"）
route 为空 + 没设 default   → 只有一个索引就用它；多个则 QueryError 要求显式指定
route 非空                  → 必须存在；不存在时 QueryError 并**列出全部可用名**
路由器为空                  → IndexStateError（"索引还没装配好"是库侧的事，不是参数错）
```

`RouteDecision.matched` 是排查时的关键差别：显式传入的结果不对就是那个库的问题，
而路由器替你选的结果不对，先要确认"该不该是它"。
另外 `StoreRouter.retrieve(query)` **优先用 `query.route`**：一份
`RetrievalQuery` 是可以被序列化、回放（端点就是这么收的）的，那时"该去哪"
是请求的一部分，反过来会让"回放同一份请求去了不同的库"。

### 配置组 `retrieval_*`（`config.py`，8 项）

```python
retrieval_top_k: int = 5                    # 默认返回条数（与 vector_default_top_k 同量级）
retrieval_fetch_multiplier: int = 3          # 过取倍率（见 types.DEFAULT_FETCH_MULTIPLIER）
retrieval_min_score: float | None = None     # None = 不设阈值（标定之前给数字是假的安全感）
retrieval_max_per_doc: int = 0               # 0 = 不限（默认保持"取最像的 K 条"）
retrieval_time_field: str = "created_at"     # 显式入库的时间戳，不用文件 mtime
retrieval_max_context_chars: int = 2400      # 送进提示词的上下文预算（chars）
retrieval_per_hit_chars: int = 800           # 单条命中上限 = 预算的 1/3
retrieval_strict_index: bool = False         # 漂移时是否拒绝服务（见第 5 节）
```

这一组与 `indexing_*` 的差别是它最该被记住的一件事：
**它们不改变任何产物的字节**（不改 `chunk_id`、不改向量、不改清单版本号），
只改变"怎么问"——因此调它们随时生效，**不需要**重建索引；
唯一的例外是 `retrieval_strict_index`，它不改变结果，只改变"漂移时是否拒绝服务"。

## 3. 时间范围的三条约定

时间过滤是本层最容易被误解的地方，因此三条约定被同时写在
`types.py::TimeRange` 的类 docstring 与 `filters.py` 的模块 docstring 里
（**同一份规则只有一处定义**，两处描述互相引用）：

| 约定 | 具体含义 | 为什么 |
|------|---------|--------|
| 闭区间 | `[start, end]`，两端可单独省略 | 单端区间（"这个月之后"）是最常见的问法 |
| 按 ISO 文本比较 | 比较在 `vectorstore` 的 `$gte` / `$lte` 里做 | 同一种 ISO 文本的**字典序 = 时间序**，比较因此是精确的 |
| 字段缺失 → 排除 | `$gte` 对"字段不存在"返回 `False` | 宁可少召回，也不给"这条什么时候建的"编一个答案 |

### 约定一：闭区间，两端可单独省略

`filters.py::time_range_clause` 把 `TimeRange` 翻译成 `where` 子句，
四种形状与三种约定一一对应（下面是本机实测打印出来的真实返回值）：

```python
time_range_clause(TimeRange(start="2026-09-01", end="2026-09-30"))
# {'$and': [{'created_at': {'$gte': '2026-09-01'}}, {'created_at': {'$lte': '2026-09-30'}}]}

time_range_clause(TimeRange(start="2026-09-01"))
# {'created_at': {'$gte': '2026-09-01'}}

time_range_clause(TimeRange())
# {}   ← 两端都没给 = 这次不涉及时间条件（与"区间宽度为 0"是两件事）
```

双端**必须**用 `$and` 包一层而不能写成一个字段的两个运算符：
`vectorstore.filters._compile_field` 明确拒绝"同一字段多个运算符"的条件，
`$and` 是那层留下的唯一合法写法（`time_range_clause` 的 docstring 原文）。

### 约定二：按 ISO 文本比较，日期按当日 0 点解释

本包**不做**"把时间解析成 `datetime` 再比"，而是把 ISO 字符串直接交给
`$gte` / `$lte`。理由：对同一种 ISO 文本（`YYYY-MM-DD` 或
`YYYY-MM-DDTHH:MM:SS`），字典序与时间序一致；而一旦库里混进
`"2026/09/20"` 这种别的写法，字典序就不再等于时间序——
那时更该修的是**入库口径**，而不是让检索器去猜每一条的格式。
**这是本课的显式约定，不是实现细节。**

只看日期时（`"2026-09-20"`）按当日 `00:00:00` 解释（`types.py::_parse_iso`
交给 `datetime.fromisoformat`）；闭区间因此包含当天全天的**起点**，
要包含整天请把上界写成下一天的（半开写法）：

```python
TimeRange(start="2026-09-20", end="2026-09-21")   # 覆盖 9/20 全天
```

`_parse_iso` 还会把 `Z` 结尾（`"2026-09-20T08:30:00Z"`）换成 `+00:00`：
Python 3.10 的 `fromisoformat` 不认 `Z`，而调用方写 `Z` 是完全合理的。
解析失败一律 `QueryError`，消息里列出可接受的三种写法。

一个跨时区的坑在两端都解析出来之后才判（`TimeRange.__post_init__`）：
`start="2026-09-01"`（不带时区）配 `end="2026-09-02T00:00:00+08:00"`
（带时区）在 Python 里无法比较，此时抛 `QueryError` 并提示"请把两端写成同一种形式"；
时间范围倒置（`start > end`）同样**当场拒掉**，而不是让它去库里筛出一个空集
（倒置的区间必然返回空结果，那是一种"少而不报"）。

### 约定三：字段缺失的记录被排除（这不是 bug）

`vectorstore.filters._operator_predicate` 里的比较谓词对
`metadata.get(field) is None` 直接判为不命中，于是没有 `created_at` 的记录
既不在区间内、也不在区间外——**它被排除**。替代方案（把缺失当 0 或当"现在"）
会让"这条什么时候建的"得到一个编出来的答案，而**错误的时间过滤不报错**，
只表现为"过滤之后少了几条"。本机实测（5 条记录里只有 4 条带 `created_at`）：

```text
时间范围 >= 2026-09-05   → 命中 c3、c2；缺 created_at 的 c4 被排除（不在命中里）
```

顺带一个真实的数据形状提醒：`chunking/types.py::ChunkSet.knowledge_records()`
产出的 metadata 是 `parent_doc_id` / `strategy` / `index` / `token_count` /
`token_measurer` / `heading_path` / `fingerprint` / `oversized` / `retrieval_text`，
**其中没有 `created_at`**。也就是说：默认时间字段要真的有值，
需要入库侧显式补上这个键（演示脚本与测试记录就是这么构造的），
否则"时间过滤"会稳定地排除掉全部记录。

### 撞车检测：同一个字段不许出现在两处

```python
# where 与 time_range 都指向 created_at → QueryError（filters.py::combine_where）
R.retrieve(RetrievalQuery(text="x",
                          where={"created_at": {"$gte": "2026-01-01"}},
                          time_range=TimeRange(start="2026-06-01")))
# 过滤条件冲突：字段 'created_at' 同时出现在 where 与 time_range 里。
# 同一个字段的两组条件请自己合并成一个显式区间……
```

合成之后**语义上是可解释的**（两个条件都满足），但它查得出结果、也不报错，
只是比预期少——这类"少而不报"正是本课要消灭的东西，所以选择当场报错，
把两个可能都成立的意图交回调用方决定。冲突只在**时间子句非空**时判：
`TimeRange` 两端都没给时它没有任何条件，此时 `where` 里出现同名字段
只是一次普通的元数据过滤。

两个与字段名有关的事实：
`filters.DEFAULT_TIME_FIELD`、`TimeRange.field` 的默认值、
`config.retrieval_time_field` 三处必须是同一个值（`"created_at"`）；
`Retriever(time_field="created_at")` 这种"传了默认值"的写法会被当成
**"没指定"**去读 `settings.retrieval_time_field`——这条写在
`Retriever.__init__` 的注释里，而不是藏着；要显式覆盖请用
`build_retriever(...)`（装配路径把 settings 的值显式指过去）。

## 4. 空结果四因诊断表

**空结果只有一个原因**，取值来自一份封闭清单 `types.EMPTY_REASONS`
（顺序 = 诊断优先级顺序），判定在 `retriever.py::_diagnose`：

| `empty_reason` | 判据（`_diagnose` 的分支顺序） | 现象 | 处置 |
|----------------|------------------------------|------|------|
| `hits` | 有命中 | 正常 | 无需诊断（它是"没有失败"的哨兵值，让字段永远有值可比） |
| `no_data` | `count_store == 0` | 库是空的 | 先建索引；**这是合法状态，不是错误**（空库不抛异常） |
| `filtered_out` | `filter_applied and candidates == 0`（或过滤开着但没切掉任何东西却仍为空） | 候选被筛成 0 条 | 放宽 `where` 或时间范围；先用 `unknown_filter_fields` 查字段名 |
| `below_threshold` | `dropped_below_threshold > 0` | 命中被阈值切完 | 降低 `min_score` 或改成 `None`（不设阈值） |
| `diversity_trimmed` | `dropped_by_diversity > 0` | 被 `max_per_doc` 挤完 | 放宽每文档条数——**注意它在端到端路径上不可达**（见下） |

`types.EMPTY_REASON_DESCRIPTIONS` 是这几句人话的**唯一来源**，
`RetrievalResult.explain()` 与端点直接引用它，避免同一句话写两遍：

```text
no_data            库是空的（合法状态）：请先建索引
filtered_out       过滤条件把候选筛成了 0 条：请放宽 where 或时间范围
below_threshold    min_score 把命中的全切了：请降低阈值或不设阈值
diversity_trimmed  max_per_doc 把命中的全挤掉了：请放宽每文档条数
```

### 为什么只允许一个值，以及优先级为什么是这个顺序

空结果的四种成因对应的动作完全不同（建索引 / 放宽过滤 / 降阈值 / 放宽多样性），
**报告里写四条可能的猜测等于什么都没说**。优先级把**更靠上游的成因**排在前面：
库是空的，就谈不上"被阈值切掉"（根本没东西可切）。这条纪律的另一半是
`RetrievalResult.__post_init__` 的两道自检：`empty_reason` 必须是五个取值之一，
且**"有命中却报告非 `hits`"直接被拒**——互相矛盾的结果会让诊断失去意义。

### 一个必须写下来的可达性事实

`diversity_trimmed` 这一支在当前算法下**端到端不可达**：第 7 步里每一组
至少会留下一条（`used >= limit` 只对第二、第三条之后的命中成立），
因此 `dropped_by_diversity > 0` 时 `kept` 必然非空。保留这一支的理由是
**诊断规则不该依赖某一步的内部细节**：一旦第 7 步改成"每组至少两条"
或"只保留主文档"，它就会立刻变成活路径，而那时它必须在正确的位置
（阈值之后、兜底之前）。验证它只能直接调用 `retriever.py::_diagnose`
（单元级构造）：

```python
_diagnose(hits=(), count_store=5, candidates=5, filter_applied=False,
          dropped_below_threshold=0, dropped_by_diversity=2)
# → 'diversity_trimmed'      ← 本机实测：直接构造才触发
```

同一张清单在端到端路径上的实测对照（本机实测，5 条记录的小库）：

```text
where={"strategy": "nope"}   → 0 条 filtered_out      （candidates=0，filter_applied=True）
min_score=0.99               → 0 条 below_threshold    （dropped_below_threshold=5，candidates=5）
max_per_doc=1                → 3 条 hits               （dropped_by_diversity=2，但一条都没被挤完）
空库                          → 0 条 no_data           （fetch_k 仍然是算过的 9）
```

最后一行是第 1 节那条"深度先算、体检后做"的直接证据：空库返回的
`fetch_k` 不是 0，而 `notes` 里还有一句"库是空的（count=0）：这是合法状态"。

### `explain()`：把四问一次答完

`RetrievalResult.explain()` 返回逐行的诊断，四行分别回答四件事
（对应关系是固定的，因为读诊断的人要找的就是这四句）：

```text
1) 这次查的是哪一版索引   版本号 + 库/清单条数 + 维度与度量 + 漂移项数
2) 过滤条件是什么         组合后的条件描述 + filter_applied + 过滤后候选数
3) 为什么条数少于 top_k   三个 dropped_* 数字（阈值切掉 / 多样性挤掉 / top_k 截断）
4) 空结果的唯一原因       empty_reason + EMPTY_REASON_DESCRIPTIONS 的人话解释
```

本机实测的一行输出（`top_k=3`，库 5 条）：

```text
条数：3 / top_k=3（召回深度 9） —— 阈值切掉 0 条、每文档上限挤掉 0 条、top_k 截断丢掉 2 条
```

`Retriever.explain(result)` 在它之上补两条**需要读库**才能给出的结论
（`retriever.py::Retriever.explain`）：字段拼写检查（要"库里出现过哪些字段"
这个全集，走 `_metadata_fields()` 的全库遍历）与漂移处置建议
（要说明这次漂在哪、怎么修）。检索主路径上没有任何一次全库遍历。

批量汇总用 `types.py::aggregate_results`，风格对齐
`evaluation/metrics.py::MetricsTracker.report()`，其中 `depth_hit_rate`
的定义值得记牢：它是"取回来的 `fetch_k` 条里最终有多少条活到了结果里"，
衡量的是**深度定得准不准**——接近 1 说明深度几乎没浪费，明显小于 1 说明
深度被三道减法吃掉大半。**它不是召回率**（真正的召回率需要标准答案，
那是 day071 的 RAG 评估）。空批次返回全 0 的一行，而不是 `ZeroDivisionError`。

## 5. 索引漂移：怎么产生、怎么观测、怎么处置

**漂移是"清单与库相互矛盾"的统称**（`types.py::IndexState` 的 docstring），
包括四类，措辞与 `indexing/manifest.py::verify_index` 的 `problems` 对齐——
同一种现象在两处应该长得一样，读日志的人不需要重新学一遍措辞：

| 类别 | 判定来源 | 典型成因 |
|------|---------|---------|
| 清单里有、库里没有 | `compare_with_store()["missing_in_store"]` | 上一次构建半途失败、有人手工删过记录 |
| 库里有、清单里没有 | `compare_with_store()["orphan_in_store"]` | 绕过 indexing 直接写库；增量没走完 |
| 维度不符 | `["dimension_match"]` | 换了编码器之后共用旧清单（清单记的维度与库不符） |
| 口径不符 | `["metric_match"]` / `["backend_match"]` | 换了度量或后端，却还拿旧清单对账 |

`retriever.py::_drift_messages` 把这四项折成几句能直接照做的话
（每条都带具体 id 预览与出路），它与 `verify_index` 的 `problems`
是**同一套判据的两次转述**：`Retriever` 复用
`indexing/manifest.py::compare_with_store`，不做第二套规则——
两份规则一定会分家，而分家之后"清单到底算不算漂"会取决于谁先被调用。

### 怎么观测

| 观测点 | 内容 | 落地位置 |
|--------|------|---------|
| `RetrievalResult.index_state` | `version_id` / `count_store` / `count_manifest` / `dimension` / `metric` / `drift` | `types.py::IndexState` |
| `IndexState.has_drift` | 清单与库是否互相矛盾（`count_manifest` 为 `None` = 没有清单可比） | 同上 |
| `RetrievalResult.notes` | 一句"索引漂移 N 项：<首条明细>" | `retriever.py::Retriever.retrieve` 第 2 步 |
| 日志 | 一条 WARNING（`logger.warning`），不依赖调用方配合 | `retriever.py::Retriever._warn_drift` |
| `Retriever.index_state` | **观测入口**：漂移时照常返回状态 | `retriever.py::Retriever.index_state` |
| `describe()["index"]` | 端点与报告直接返回的结构 | `retriever.py::Retriever.describe` |

用 WARNING 而不是 ERROR：检索仍然会返回结果，它不是失败；但"清单过期"
必须在一个**不依赖调用方配合**的地方留下痕迹——日志是唯一这样的地方
（端点可能被绕过，演示脚本可能不看返回值）。

`Retriever` 与 `Retriever.index_state` 的差别是刻意的：`index_state` 是
**观测**入口（运维想看看"现在漂成什么样了"），`retrieve` 是**取数**入口，
`strict_index=True` 时必须拒绝给出可能不完整的答案。

### 处置建议

```text
1. 重建清单    indexing.manifest.manifest_from_store(backend, identity=...)
               注意：重建出来的是"从现在起"的基线，不是历史的那一版
2. 补回记录    用漂移明细里的 id 把缺的向量补回去（不要在这份清单上做增量）
3. 对齐口径    换过编码器 / 度量 / 后端就要为新口径单独记一份清单
```

第 1 条的"从现在起"是 `manifest.py::manifest_from_store` docstring 的原话：
清单丢了但库还在时，重建只能按库里的现状算（原文不在了、`retrieval_text`
不在 metadata 里，都会让算出来的指纹与当初不同）。
`Retriever.explain()` 会把这段处置建议追加在诊断末尾；
另有另外两个出口用的是同一套判据：`indexing/manifest.py::verify_index`
（逐项结论 + `problems`）与装配阶段的 `indexing/builder.py::IndexBuilder.verify`
（增量更新前的那道门，内部直接转调 `verify_index`）。

顺带一条**只是注记、不是错误**的观测（`retriever.py::Retriever._encoder_notes`）：
当"本次编码器的维度/实现名"与清单的 `identity` 记的不一致时，
`notes` 里会多两句提示。为什么不升级成异常：**它仍然能查出结果**，
而错配的证据本身已经在 `IndexState` 里（维度不符本身就是一项漂移）；
把它升级成异常会让"清单里 provider 名字写得不完全一样"这种无害情形
变成一次服务不可用。

### `strict_index` 的取舍

```text
阻断（strict_index=True）   清单过期 → 检索整体不可用 → 用户什么都拿不到
观测（默认 False）          清单过期 → 检索仍然能跑，只是可能少召回那几条被删掉的记录
```

本层选观测，理由是一个取舍：**"检索仍然能跑、只是可能少召回"比
"因为一份 JSON 过期而拒绝服务"更可取**；而**静默降级**是唯一不可接受的选项
——所以漂移必须进 `RetrievalResult` 与 `notes`，让"这次的结果可能不完整"
成为一个能被读出来的事实。

严格模式下抛的是 `errors.py::IndexStateError`（"改调用点是走不通的"那一族：
要去重建清单或补回记录），消息里给三步出路并提示"若接受'能跑就先跑'，
请关掉 `strict_index`（默认关闭）"。还有一条**只增不减**的护栏：

```python
# retriever.py::Retriever.__init__
self._strict_index = bool(strict_index) or bool(settings.retrieval_strict_index)
```

构造参数只能把它打开，不能覆盖 settings 里已经打开的 `True`——它是
"这台机器一律不接受漂移"的运维开关（典型场景是评估与发布验证）。
`build_retriever(...)` 因此把 `strict_index` 显式指向
`settings.retrieval_strict_index`，而 `top_k` / `fetch_multiplier` /
`min_score` / `max_per_doc` 传 `None` 时由 `Retriever` 自己读 settings——
**抄一遍就会出现两处默认值，而它们迟早会不一致**。

## 6. 上下文打包与引用：预算、截断与"至少保一条"

打包这一半在 `retrieval/context.py`，它把一批 `RetrievalHit` 折成
"**一份带编号、有预算的提示词片段**"（模块 docstring 的原话）：
检索器交出来的是"id + 分数 + 正文 + 元数据"，而提示词只认一段**文本**，
中间隔着的三件事全在这个模块里。

### 预算：单位是 chars，三个数字各有出处

| 常量 | 值 | 落地位置 | 取值理由（原文要点） |
|------|----|---------|--------------------|
| `retrieval_max_context_chars` | `2400` | `config.py` | 按"6 个 ~400 字的片段"标定：够放下 top-5 的全部正文，又不至于把 4k 上下文撑满 |
| `retrieval_per_hit_chars` | `800` | `config.py` | 预算的 1/3：一条过长时先截断它，而不是让它把整包预算吃光 |
| `DEFAULT_MIN_HIT_CHARS` | `32` | `types.py` | 低于它的片段进上下文只是噪声（"阈值设为 0.85"离开上面的标题就毫无意义） |

**预算单位默认是 chars**：`context.py::_char_measure` 就是 `len(text)`，
与 day062 的默认度量同一条理由（确定性、离线、可复现——`len()` 在任何机器上
给出同一个数，而 token 计数要引入分词器与版本）。需要真实 token 口径时
把 `measurer` 注入进来即可（例如接 `llm/tokenizer.py::TokenCounter.count_tokens`
或 `estimate_tokens_chars`）——**本模块不 import 任何分词器**，因此它不认识 token。

### 五步：顺序固定，因为每一步都改变"还剩多少预算"

`context.py::pack_context(hits, *, max_chars=None, per_hit_chars=None,
min_hit_chars=DEFAULT_MIN_HIT_CHARS, measurer=None)` 的 docstring 原文：

```text
1. 解析预算     None → settings.retrieval_max_context_chars / retrieval_per_hit_chars
2. 单条截断     超 per_hit_chars 的块被截断并记 truncated_hits（它仍然在包里）
3. 拼包         用 BLOCK_SEPARATOR 连成一段
4. 丢尾         整包超预算 → 从**最后一条**开始整条丢，记进 dropped_hits
5. 至少保一条   只剩一条还超预算 → 把它截到 max_chars，**返回它而不是返回空**
```

三个刻意的细节写在同一份代码里：

```text
BLOCK_SEPARATOR = "\n\n"    每个块自己就带换行（头部与正文之间），单换行会让"下一条从哪开始"看不出来
TRUNCATION_MARKER = "…"     截断的痕迹要在提示词里看得见：不补标记时模型会顺手把话补完
第 5 步那次截断**不计**进 truncated_hits   truncated_hits 的定义是"被单条上限截断的条数"，
                            第 5 步的证据在 char_count == max_chars 上（包被削到刚好占满）
```

"先截单条、再丢尾部"这个顺序不是随手定的：两种损失的性质不同——
**截断**是"这一条少半句"（内容损失，去调 `retrieval_per_hit_chars`），
**丢尾**是"整条不见了"（召回损失，去调 `retrieval_max_context_chars`）。
反过来或平均切每一段，报告里就只剩一个"包满了"，
而"该调哪个参数"从此无从回答。

`measurer` 注入之后还有一个真实问题：**"切到第几个字符"与"量出来是多少"不再成正比**
（token 计数对中文约 1 字 1 token、对英文约 4 字 1 token）。`context.py::_truncate`
因此是"先按字符粗切，再逐字符回收直到量得过去"：最坏情况是几十次 `len` 级别的调用，
换来的是不会出现"截断之后反而超标"这种只能在运行期才发现的结果。

### 预算过小 → `ContextError`，而且它排在"空 hits"之前

```python
if budget < floor:      # floor = min_hit_chars，缺省 types.DEFAULT_MIN_HIT_CHARS = 32
    raise ContextError("一条命中至少需要 32 字，本预算只给了 20 字：……")
```

消息里写清了理由：**"放不下"不该被静默处理成"空上下文"**——空上下文会让整条
RAG 链路走进"检索为空"的分支（一次 LLM 都不调），于是一次配置错误被伪装成
"库里没有相关内容"。配置检查排在"空 `hits` 返回空上下文"之前是刻意的
（`pack_context` 的 docstring）：预算配错的时候，"这次恰好没有命中"不该让这个错误被跳过
——配错的预算会在下一次有命中时以同样的方式错下去。

`context.py::_resolve_limit` 与 `_resolve_min_hit_chars` 另外挡住两种写法：
`max_chars` / `per_hit_chars` 给 0 或负数 → `ContextError`
（"要'不限'请给一个足够大的正数，不要给 0"）；`min_hit_chars` 与 `per_hit_chars`
冲突时**以硬上限为准**——截断是明确可见的（`truncated_hits`），
而为了下限去放宽上限会静默地违背调用方给的数字。

**空 `hits` → 空上下文**（`text=""`、`citations=()`），不是异常：
"没有命中"在检索层已经是 `empty_reason` 的结论，打包层再抛一次异常
会让同一次空结果在两层各报一次。

### 打包结果自己会交代：`PackedContext`

| 字段 / 属性 | 含义 |
|------------|------|
| `text` | 真正进提示词的文本（`[n]` 编号已就位） |
| `citations` | 编号 → 命中 的对照表（只有进了上下文的那些） |
| `used_hits` / `dropped_hits` | 进了包的命中 / 从尾部被整条丢掉的命中 |
| `truncated_hits` | 被单条上限截断的条数（**只算真的进了上下文的那些**） |
| `char_count` / `max_chars` / `fill_ratio` | 实测占用 / 这次的预算 / 占用率 |
| `marker_for(record_id) -> int \| None` | 某条记录在这份上下文里的编号（不在里面返回 `None`） |

`PackedContext.__post_init__` 还有两道自检：`citations` 与 `used_hits` 必须**一一对应**
（多出来的引用会指向提示词里不存在的片段），且 `marker` 必须是 1 起**连续**的
（中间断号会让模型写出指向空位的 `[n]`）。`char_count` 与 `max_chars` 必须成对给出：
只有一个 `char_count` 时，读的人不知道它是"包刚好这么长"还是"被削到这么长"。

### 引用标记：`RetrievalHit.citation(index)` 是唯一一份回落实现

```python
hit.citation(1)      # → "[1] c2 › 检索器 > 阈值"     （本机实测）
```

`context.py::_render_block` 直接复用它渲染块头，`_source_of` 的三级回落
（`metadata["source"]` → `["doc_id"]` → `record_id`）也与它**逐条相同**：
"来源与标题怎么回落"只该有一份实现，否则"答案里的 `[2]` 指哪份文档"
会取决于读的是哪一份数据。`Citation` 的字段是
`marker` / `record_id` / `source` / `heading_path` / `score`；
`marker < 1` 或 `record_id` 为空都抛 `ContextError`
（引用编号从 1 起是与 day069 引用溯源共用的约定）。

`heading_path` 是 day062 的 `knowledge_records()` 产出的键
（`chunking/types.py`，取 `chunk.heading_text`，形如 `"向量库 > 度量"`）。
这条元数据被用在三处：**引用**（本节的 `[n] 来源 › 路径`）、
**过滤**（`where={"heading_path": ...}` 或 `$contains`）、**多样性分组**
（`doc_id` / `parent_doc_id`）。它与 `retrieval_text`（面包屑 + 正文，
向量化用的那一份）是两件事，别混（见 `chunking/types.py` 的对照表）。

## 7. RAG 生成链路：一条护栏比一段提示词更有用

`retrieval/pipeline.py` 把 `retriever` 与 `context` 串成一条能回答问题的链路，
`RagPipeline.answer` 的四步是**固定**的（docstring 原文）：
**检索 → 判空 → 打包 → 生成**。

### 护栏：检索为空 → 一次 LLM 都不调

```text
命中      打包 → 渲染提示词 → llm.chat([...]) → 答案 + citations + context + retrieval，llm_called=True
未命中    llm_called=False，answer = fallback_answer，citations=()、context=None，**一次调用都不发起**
```

判据是检索层已经给出的结论，不需要"看回答像不像编的"：

```python
retrieval = self._retriever.retrieve(question)      # pipeline.py::RagPipeline.answer
if retrieval.is_empty:                              # = not hits，且 empty_reason != "hits"
    return RagAnswer(..., llm_called=False, notes=tuple(_empty_notes(retrieval)))
```

**判空在打包与渲染之前**——护栏写在调用的上游，而不是调用的下游
（模块 docstring 的原话：空结果 + 调用 LLM 得到的产物"看起来最像成功"：
语法正确、语气确定、引用格式也像模像样，却全部来自预训练记忆）。
`types.py::RetrievalResult.is_empty` 与 `empty_reason` 因此是这条护栏的
**唯一判据**——这也是第 4 节那张诊断表的意义之一：空结果必须能自证，
而不是留给生成层去猜。

`RagAnswer.llm_called` 单独成为一个字段而不是"看 answer 等不等于兜底文案"：
**判定不能依赖字符串比较**——兜底答复将来改一个字，那种判定就静默失效
（`RagAnswer` 的 docstring）。空结果时 `_empty_notes` 还会记三条注记：
为什么没调模型、护栏本身是什么、三条出路。

### 兜底文案：一次失败宣告必须跟一个下一步

`pipeline.py::FALLBACK_NO_CONTEXT` 的原文（它同时是空结果那三条注记的收尾）：

```text
知识库中没有检索到与该问题相关的内容，因此不作回答。
建议：换一种说法、放宽过滤条件（时间范围 / 元数据），或确认索引版本是否包含这批资料。
```

### 提示词：版本化 + 构造期校验占位符

```python
RAG_ANSWER_PROMPT_V1 = """…"""      # v1（day066）：四条约束各自挡一种真实失败
RAG_ANSWER_PROMPT = RAG_ANSWER_PROMPT_V1
RAG_ANSWER_PROMPT_VERSION = "v1"
REQUIRED_PROMPT_FIELDS = ("context", "question")
```

四条约束的注释就写在常量上方（挡的是四类"看起来正常"的失败）：
"只依据片段"挡模型拿预训练知识把答案补圆、"片段不足就说没有"挡编一个通顺的答案、
"引用写 `[n]`"挡答案对但溯源不上、"不许编造"挡把两个相似概念合并成一个。
改模板的规则与 `agent/prompts.py` 的既有约定一致（`REACT_SYSTEM_PROMPT_V2` +
`REACT_SYSTEM_PROMPT_VERSION = "v2"` 就是那个形状）：**历史版本永不删**，
新增 `_V2` 再把 `RAG_ANSWER_PROMPT` 指过去。理由是**答案质量的变化必须能被归因**：
同一份索引、同一批问题，换一版提示词之后变好还是变差，只有固定版本号才回答得了
（day071 的 RAG 评估会拿它做实验分组）。`RagPipeline.prompt_version` 与
`RagAnswer.to_dict()["prompt_version"]` 把这一版号一路带到报告里。

模板占位符缺任何一个都在**构造期**报 `ContextError`（`_validate_prompt`），
而且用的是 `string.Formatter().parse` 而不是"在字符串里搜 `{context}`"：
后者会把 `{{context}}`（转义之后的字面量）也当成占位符。
为什么必须构造期报：缺占位符的模板 `format` **不会失败**（多余的参数被忽略），
模型收到的是一段**没有资料的提示词**——它照着模板回答，看起来一切正常，
实际上一次检索都没用上。那正是这条链路要拦的事。

### 参数与降级：`OVERRIDE_KEYS` 与 `notes`

```python
OVERRIDE_KEYS = ("max_context_chars", "per_hit_chars", "temperature")   # answer(**overrides)
```

`answer(**overrides)` 只认这三个键，**不认识的键直接报 `QueryError` 并列出合法取值**
——静默忽略一个覆盖参数会让调用方以为它生效了（"把 `max_context_chars`
拼成 `max_context_char` 之后，预算仍然是默认值，而答案看起来只是变短了"）。
两个预算参数传 `None` 时读 `settings`（`retrieval_max_context_chars` /
`retrieval_per_hit_chars`），**不重抄检索参数**：同一份默认值抄两遍迟早会不一致
（与 `build_retriever` 的取舍同一条理由）。`temperature` 必须落在 `[0, 2]`，
默认 `0.0`——"RAG 问答是**有依据的复述**，过高的温度会让模型改写片段里的事实"。

降级沿用 `RetrievalResult.notes` 那条通道（第 5 节），三类各记一条可读的话
（`pipeline.py::_degrade_notes`）：

```text
truncated_hits > 0        "有 N 条命中被单块上限 800 字截断……要完整片段请调大 retrieval_per_hit_chars"
dropped_hits 非空         "预算 2400 字放不下，从尾部丢掉了 N 条命中（id 预览）……请调大 retrieval_max_context_chars"
index_state.has_drift     "索引漂移 N 项：本次检索基于可能过期的清单，结果可能少召回……请重建清单"
```

另外一类**不是降级**、但同样容易误判的情况也进 `notes`：模型返回空串时，
链路会写一句"这一次的 answer 是空串，而引用与上下文都在——请检查提供方
（超时、内容过滤、`max_tokens` 太小），**不要把它当成'知识库里没有相关内容'**
（那是 `empty_reason` 的事）"。

### 这一步的测试怎么写

`RagPipeline.__init__` 要求 `llm` 是 `BaseLLM`，而它的 docstring 明说
"测试用假实现是刻意的用法"——未命中路径注一个"一旦被调用就抛异常"的假
`BaseLLM`（`llm/base.py::BaseLLM` 是抽象基类，实现一个 8 行的子类即可），
**它没被调用过，就是护栏生效的证据**。命中路径用
`llm/mock.py::MockLLM(responses=[...])`：它的 `calls` 记着每次调用收到的消息列表，
可以直接断言"传进去的提示词里含 `[1]`"。
`answer_many` 与 `Retriever.retrieve_many` 同样是**刻意的逐条**：
每次问答的 `llm_called` / `notes` / 上下文占用都是独立的证据，
而"这一批总共花了多少"与"其中一次为什么没调模型"是两个问题。

## 8. 端点清单（占位）

`retrieval/__init__.py` 的接缝段原文是："**端点**：`api.routes` 的 `/retrieval/*`；
手册在 `docs/retrieval.md`；离线演示 `scripts/retrieval_demo.py`。"
本轮检索包先落地，API 层随后接上，因此下表只写**入参形状**与
**它落到哪个已落地的入口**；**响应字段一律不在本文复述**——
请以 `api/routes.py` 的 `/retrieval/*` 实现（与 `api/schemas.py` 的模型）为准。

| 端点 | 方法 | 入参（都是已落地的形状） | 落到哪个已落地入口 |
|------|------|------------------------|------------------|
| `/retrieval/status` | GET | — | `retriever.py::Retriever.describe()`（含 `index_state` 的字典） |
| `/retrieval/search` | POST | `RetrievalQuery` 的字段：`query` / `top_k` / `fetch_k` / `where` / `time_range` / `min_score` / `max_per_doc` / `route` | `retriever.py::Retriever.retrieve` → `types.py::RetrievalResult` |
| `/retrieval/explain` | POST | 同上 | `retriever.py::Retriever.explain`（在 `RetrievalResult.explain` 之上补字段拼写检查与漂移处置） |
| `/retrieval/answer` | POST | `question` + 同上（另可按 `pipeline.py::OVERRIDE_KEYS` 覆盖预算与温度） | `pipeline.py::RagPipeline.answer` → `RagAnswer`（未配置 LLM 时应是 400 而不是 500） |
| `/retrieval/routes` | GET | — | `routing.py::StoreRouter.report()`（名字 / 说明 / `describe()`） |

三条与错误映射有关的约定，它们在检索层就已经定死，端点层只需转译：

```text
参数问题        → 400（QueryError：空查询 / top_k 越界 / fetch_k < top_k /
                        where 非法 / 时间范围倒置 / where 与 time_range 冲突 / 路由名不存在）
库侧问题        → 让 IndexStateError 说话（维度不符、编码器坏向量、strict_index 下漂移）
库是空的        → **返回空结果而不是 500**：no_data 是合法状态（Retriever 第 2 步）
```

`/retrieval/search` 与 `/retrieval/explain` 的入参是同一份 `RetrievalQuery`，
因此"这一条查询"在端点、报告、回放、测试里是同一个对象——
`retrieval/types.py::RetrievalQuery.to_dict()` 就是它的可序列化投影，
`route` 字段也在其中（第 2 节：`StoreRouter.retrieve` 优先用 `query.route`）。

## 9. 常见误区（10 条）

| # | 误区 | 事实 | 证据（可核对） |
|---|------|------|--------------|
| 1 | "阈值是库侧切的" | `min_score` **不传给** `backend.query`，在检索器里落刀 | `retriever.py::Retriever.retrieve` 第 5/6 步；`_apply_threshold` |
| 2 | "`dropped_by_top_k` 是异常" | 它只是"取够了"，不是问题；要查的是另外两个 | `types.py::RetrievalResult` 的字段说明 |
| 3 | "把 `filtered_out` 与 `below_threshold` 当同一回事" | 前者改 `where`，后者改 `min_score`；两者都由唯一优先级定序 | `retriever.py::_diagnose` |
| 4 | "`max_per_doc=1` 会让结果为空" | 每组至少留一条，`diversity_trimmed` 端到端**不可达**（防御分支） | `_apply_diversity` / `_diagnose` 的 docstring |
| 5 | "`where` 里再写一遍时间字段没问题" | `combine_where` 直接报 `QueryError`（同一个字段的意图不许猜） | `filters.py::combine_where` |
| 6 | "时间过滤少了没有时间戳的记录是 bug" | 这是**约定三**：字段缺失被排除，宁可少召回也不猜时间 | `types.py::TimeRange` / `vectorstore/filters.py::_operator_predicate` |
| 7 | "`top_k` 就是召回深度" | 深度 = `max(top_k, ceil(top_k × 3))`，`top_k=5` → 15，且封顶 1000 | `retriever.py::_fetch_depth`、`types.MAX_FETCH_K` |
| 8 | "阈值用 `>` 才对" | 用 `>=`，与 `VectorBackend.query(min_score=...)` 逐字一致 | `_apply_threshold` 的 docstring |
| 9 | "漂移会让检索报错" | 默认只观测（`notes` + WARNING 日志），只有 `strict_index=True` 才拒绝服务 | `types.py::IndexState`、`retriever.py::_warn_drift` |
| 10 | "检索器能做权限过滤 / 认识语料" | `where` 是元数据筛选，不是访问控制；检索器只认识索引状态 | `types.py::RETRIEVAL_LIMITATIONS` 第 4 条 |

再补三条同性质的（写下来省一次排查）：

```text
11. "查询里可以写 max_per_doc: 0"        → 不允许（>= 1）；0 只是 Retriever 构造参数里"不限"的写法
12. "RetrievalHit 与 SearchHit 是一回事"  → 前者不带向量、不带 distance，是给打包与展示用的形状
13. "channel 说明已经有混合检索了"        → 单路向量检索时它恒为 "vector"（CHANNEL_VECTOR），
                                           day067 的融合是把这条路打开，不是已经实现
```
