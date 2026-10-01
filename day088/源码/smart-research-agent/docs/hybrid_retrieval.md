# 混合检索手册（day067 / M6-D6）

> 本文与代码**同源**：每一条结论都标了落地位置（`retrieval/xxx.py::函数名`），
> 都能在 `smart_research_agent/retrieval/{lexical,fusion,hybrid}.py`、
> `vectorstore/{filters,base}.py`、`config.py` 的 `retrieval_*` 组里逐字找到。
>
> **本文的数字只有三个来源**，其余一律写"以你实际运行时输出为准"：
>
> ```text
> 代码里写死的常量    DEFAULT_K1 = 1.5、DEFAULT_B = 0.75、DEFAULT_RRF_K = 60、
>                     DEFAULT_ALPHA = 0.5、retrieval_hybrid_enabled = False…
> 由它们算出的算术    1/61 = 0.016393443…；1/61 + 1/68 = 0.031099…；
>                    深度 = max(top_k, ceil(top_k × 3))，top_k=5 → 15
> 本机实测           本文标注"本机实测"的行来自同日的离线演示
>                    `scripts/hybrid_demo.py`（FlatVectorStore + 一张写死的向量表 +
>                    10 条记录，零网络）。它不是课程产物，只是用来说明形状；
>                    正式数字请以 `outputs/hybrid_demo.txt` 与
>                    `python -m pytest tests/test_hybrid_*.py` 的实际输出为准。
> ```
>
> 全层的落地状态（`retrieval/__init__.py` 的原文）："导出范围（**十个模块全部落地**）：
> `errors` / `types` / `filters` / `retriever` / `routing` / `context` / `pipeline` /
> `lexical` / `fusion` / `hybrid`（后三个是 day067 新增的）"。
> 端点层（`api/routes.py` 的 `/retrieval/*`，day066 五个 + day067 两个 = 七个）与
> 本层同日交付，但本文只按**索引式写法**引用它，**不复述任何响应字段**。

## 1. 一句话定位：两路的分工

```text
向量路回答      "意思像不像"        → 擅长同义改写、语义相近但词面完全不同的问法
关键词路回答    "这个词在不在"      → 擅长编号、函数名、报错码、专有名词（字面精确）
```

这是 `retrieval/hybrid.py` 模块 docstring 的主题，也是本课的全部理由。两路各有一个
死穴，而**两个死穴不是同一个**：

```text
向量路的死穴   编码器没见过的东西（编号 / 函数名）→ 它只能"差不多就行"
关键词路的死穴 同义改写（"重算" vs "重新计算"）→ 它只认字面
```

因此把两路合起来**不是为了"更强"**，而是为了**两种失败不再是同一种失败**：

```text
单路检索    错的时候只有一种形状——"返回了一批不太相关的记录"，而报告里看不出它错在哪
两路融合    错误被劈成两种：一路空着、另一路照常给答案，而"哪一路空着、为什么"是可读的
```

### 模块地图（每个模块回答一个问题）

```text
lexical.py   关键词一路：零依赖 BM25 + CJK 2-gram 分词 + "为什么一条都没召回"的证据
fusion.py    融合与去重：RRF（只看名次）与归一化加权（先归一化再加权）；去重 = 合并证据
hybrid.py    混合检索器：两路共用一份过滤子句与深度，融合后仍能逐条解释
```

### 十三步流水线（`HybridRetriever.retrieve` 的固定顺序）

| 步 | 做什么 | 落地位置 | 它留下的可见痕迹 |
|----|--------|---------|-----------------|
| 1 | 规范化查询（`None` 字段沿用向量路默认值） | `hybrid.py::HybridRetriever._normalize_query` | `result.query` |
| 2 | 融合参数解析（ctor 默认 ← `query.extra`） | `hybrid.py::HybridRetriever._resolve_fusion` | `result.fusion`、`notes` |
| 3 | 算召回深度（**复用向量路那唯一一份实现**） | `retriever.py::Retriever._fetch_depth` | `result.fetch_k` |
| 4 | 索引体检（条数与清单漂移） | `retriever.py::Retriever._inspect_index` | `result.index_state`、`notes` |
| 5 | 合成过滤子句（**只调一次**） | `filters.py::combine_where` | 两路拿到同一个对象 |
| 6 | 关键词路取候选 | `lexical.py::LexicalIndex.search` | `channel_candidates["bm25"]` |
| 7 | 向量路取候选（编码 + `backend.query`） | `retriever.py::Retriever._encode_query` | `channel_candidates["vector"]` |
| 8 | 阈值落刀（**只对向量路**） | `retriever.py::_apply_threshold` | `dropped_below_threshold` |
| 9 | 融合 | `fusion.py::fuse` | `result.fusion` |
| 10 | 装配命中（`channel` / `channels`） | `hybrid.py::HybridRetriever._to_hit` | `hits[].channel`、`hits[].channels` |
| 11 | 多样性裁剪（**在融合之后**） | `retriever.py::_apply_diversity` | `dropped_by_diversity` |
| 12 | 截断到 `top_k`（只重编号，不重排） | `hybrid.py::HybridRetriever.retrieve` | `dropped_by_top_k` |
| 13 | 诊断与装配 | `retriever.py::_diagnose` | `empty_reason`、`latency_ms`、`notes` |

第 3、4、8、11、13 步**复用 day066 的函数**（连私有函数都是同几个）：两路的事务
如果各写一份，就会出现"单路检索与混合检索对同一个参数的解释不同"这类分家，
而分家的表现是"同一个库、同一句话、两份报告里的 `dropped_below_threshold` 不一样"。

### 已知边界与明确不做的事

`types.RETRIEVAL_LIMITATIONS`（四条，写在代码里；day067 改写过这一份）：

| 边界 | 原文要点 |
|------|---------|
| 无重排 | 不带重排序：命中的顺序完全由（融合后的）分数决定（day068 加交叉编码器） |
| 阈值只对向量路 | BM25 的分数没有绝对标度，给它一个数字是假的安全感 |
| 2-gram 代替分词器 | 零依赖的 BM25：不做词形还原、不做同义词扩展、不认识未登录词 |
| 无权限 | `where` 是元数据筛选，不是访问控制（那属于 day015 的工具权限层） |

`types.RETRIEVAL_OUT_OF_SCOPE`：查询改写与扩展、学习型排序与学习型融合权重
（本课的权重是显式参数）、跨索引的分布式检索、在线索引更新。

## 2. 关键词一路：零依赖 BM25

### 为什么不用 `jieba` / `rank_bm25`（`lexical.py` 模块 docstring 的原文意思）

```text
1. 不新增依赖   课的全部产物必须能在"只有 Python + 标准库"的机器上复现，
                而 BM25 与分词都是几十行的算术，不值得为之引入依赖
                （jieba 还会带一个约 5 MB 的词典文件）
2. 评估要可复现  分词器是有版本的（jieba 换版本会切出不同的词），
                于是同一天、同一份语料、同一批查询会给出不同的 BM25 分数；
                而 day071 的 RAG 评估要求"分数变了必须能归因到某个显式参数"。
                相邻 2-gram 的分词法**没有版本**：它的定义就是"字符串的相邻两字"
```

代价是真实的，写在 `RETRIEVAL_LIMITATIONS` 里：词表比真实词表大（一条 20 字的句子
会产生 19 个 2-gram）、不认识"未登录词"。

### 分词规则（`lexical.tokenize`，确定性、无参数）

```text
先按 [a-z0-9_]+ | [\u4e00-\u9fff]+ 切段（全角标点、空格、连字符都是分隔符）
拉丁/数字段    原样保留（大小写先折叠）："ERR-2043" → err、2043
汉字段长度 1   保留该单字："熵" → 熵
汉字段长度 ≥2  单字 + **相邻 2-gram**："语义缓存" → 语/义/缓/存/语义/义缓/缓存
```

本机实测（`outputs/hybrid_demo.txt` 第 1 节）：

```text
tokenize('语义缓存') = ['语', '义', '缓', '存', '语义', '义缓', '缓存']
tokenize('ERR-2043') = ['err', '2043']        ← 连字符是分隔符（编号的前缀与序号各成词元）
```

**单字与 2-gram 都要**，各补一件事：单字让"重算"与"重新计算"共享 重/算
（在不引入同义词表的前提下蹭到一点召回），2-gram 让"缓存"与"存缓"区分开。
只留 2-gram 时单字查询（"熵"、"税"）会一个词元都对不上，而单字查询在中文里真实存在。

注意"空白是分隔符"这条：`tokenize("  语义  缓存  ")` **没有**跨空白的 `义缓`
——2-gram 只在同一个汉字段内生成（`tests/test_hybrid_lexical_tokenizer.py`
有逐条对照的用例，期望值由测试侧**独立重写**的切分逻辑算出来）。

### 打分：`idf` 恒正、`k1` 与 `b` 各管一件事

```text
idf(t) = log(1 + (N - df + 0.5) / (df + 0.5))                  ← 恒正
score  = Σ_t idf(t) · tf · (k1 + 1) / (tf + k1 · (1 - b + b · dl / avgdl))
```

**`log(1 + …)` 里的那个 1 是刻意的**（`lexical.py::_idf` 的 docstring）。
经典写法 `log((N - df + 0.5) / (df + 0.5))` 在 `df > N/2` 时**变成负数**——
一个"到处都出现的词"会给文档减分。两种写法的排序多数时候一样，但负 IDF 会让
"分数为负"与"0 分"混在一起，而本模块用 0 分表示"完全没有共同词元"（见下）。
本机实测：`_idf(10, 1) = 1.992430`（`log(1 + 9.5/1.5)`，手算等于实现）。

`k1` 与 `b` 的缺省 `1.5` / `0.75` 出自 **Robertson & Zaragoza（2009）的经典取值**，
与 Lucene / Elasticsearch 的缺省同源。**它们不是本课标定出来的**——引用一个公开取值，
别人才分得清"结果不同"是语料造成的还是参数造成的。两个参数的**方向**
（`outputs/hybrid_demo.txt` 第 6 节，用一份受控梯度语料）：

```text
k1 扫描（b=0.75）   k1=0 → heavy/light 分数比 1.0000（词频完全不参与）
                    k1 越大比值越大（同一个词出现更多次的收益更大）
b  扫描（k1=1.5）   b=0 → 不归一化；b=1 → 完全归一化，比值单调下降
                    （长文档里的同一个词越不值钱）
```

`k1` 与 `b` 的构造期校验在 `lexical.py::BM25Params.__post_init__`：非有限数、
`k1 < 0`、`b ∉ [0, 1]` 都抛 `QueryError`（调用方的问题），消息里写清后果
（`b` 越界会让分母变负，于是"长文档反而得到更高的分数"）。

### 只返回"至少命中一个词元"的文档（第二个关键取舍）

BM25 会给**完全不相关的文档**一个 0 分，而它们在名次上排在所有真实命中之后。
如果把它们也算作"关键词路的命中"，名次靠后的位置上会躺着一批"和查询毫无关系"
的记录——而 RRF **只看名次不看分数**，它会老老实实给"关键词路第 N 名"一个
`1/(k+N+1)` 的贡献。**那条证据是假的。**

因此 `lexical.py::LexicalIndex.search` 只返回 `score > 0` 的文档，并把被排除的
条数写进 `notes`。本机实测（10 条语料、查询 `ERR-2043`）：

```text
关键词路：1 条命中（h-t-01），note 里写着"有 9 篇候选文档与查询没有任何共同词元
（BM25 给 0 分），已排除在名次之外：0 分不是'相关性很低'，而是'这次查询完全没提到它'"
```

### `matched_terms` / `missing_terms`：这一路"为什么空着"的唯一证据

```text
missing_terms 全等于 query_terms  → 一个词都不认识（纯语义改写 / 用了别的语言）
missing_terms 为空                → 词都认识，但过滤之后没有一篇同时包含它们
matched_terms 非空且 hits 非空     → 正常命中
```

本机实测（查询 `午饭吃点什么好`）：

```text
query_terms 13 个（7 个单字 + 6 个 2-gram），matched_terms 为空，
missing_terms 全列，note："词表里有 274 个词元"——"交集为空"因此有了**分母**
```

### 空索引与"中毒索引"（两种极不一样的处境）

```text
0 篇文档        → 空结果 + notes（"这是合法状态"，与向量路 count=0 同一口径），**不抛异常**
有文档但全无词元 → LexicalError（BM25 的分母里有 avgdl，索引没法打分）
                 → 这是索引侧的事：去修入库口径或重建，**不是**"没有命中"
```

第二种由 `lexical.py::LexicalIndex.search` 显式挡住，它属于
`errors.LexicalError`（"改调用点走不通"那一族）。

### `where` 过滤复用同一份语义

`LexicalIndex.search(..., where=...)` 的校验与执行**完全复用**
`vectorstore.filters.compile_filter`（经 `retrieval.filters.validate_where` 包装，
把 `FilterError` 翻成 `QueryError`）。两路各写一份过滤实现的话，
"同一个 `where` 在向量路筛掉 3 条、在关键词路筛掉 2 条"这种分家会永远存在
而且不报错——而 `hybrid` 的"过滤在两路都落刀"要的正是**同一份**语义。

两个刻意的口径（`lexical.py` 的注释里写着）：

```text
idf 与 avgdl 是全库统计    过滤**不重算**它们：它们是语料的性质，不是"这次过滤出来的
                          那几条"的性质。用子集重算会让同一个词在两次不同过滤下
                          有不同的权重，而分数是要跨查询比较的
按 (-score, record_id) 排序  与向量路同一条纪律：分数相同时必须有一个不依赖插入
                          顺序的第二键，否则两次运行会给出不同的名次
```

## 3. 融合：RRF 与 weighted 的取舍

### 第一件事：跨通道的分数量纲不可比

```text
向量路   余弦相似度 ∈ [-1, 1]     "0.82" 有绝对上界
关键词路 BM25 ∈ [0, ∞)            "7.31" **没有上界**
```

两个数字放在一起比较是**类型错误**，而它不会报错：`0.82` 与 `7.31` 都是浮点数，
`sorted` 照单全收。于是"加权求和"这个看起来最自然的融合方式会把
**量纲差当成相关性差**——BM25 只要分数普遍偏大，它就会单方面决定名单顺序，
而 `alpha` 那个权重根本没起作用。

本机实测（第 5 节）：同一份输入下 `[1.0, 0.0, 0.0]` 与 `[4.0006]` 放在一起，
直接相加时后者说了算；归一化之后两边才可比。

### `rrf`：只看名次（缺省策略，免标定）

```text
贡献 = 1 / (k + rank + 1)        rank 从 0 起，k 缺省 60（RRF 原论文取值）
score = Σ_channels 贡献           同一条被两路召回 → 两个贡献相加
```

三个好性质：**免标定**（换编码器、换语料都不用重调）、**免归一化**（不存在除零与
上下界问题）、**抗异常值**（某一路给出一条分数虚高 100 倍的记录，它在 RRF 里
只是"第 1 名"）。代价也要写清楚，否则它会变成一个"永远该选 rrf"的教条：

```text
丢掉"差多少"                 第 1 名与第 2 名差 0.001 还是差 0.5，RRF 一视同仁
丢掉"只有一路有"这个信号        某条被两路同时命中（最强证据）与分别被两路各自
                            命中一次的**两条不同记录**，在 RRF 里得分相近
```

第二条正是 `weighted` 存在的理由之一。`k` 的作用是**压平头部差距**：
本机实测第 4 节手算（k=60，查询 `ERR-2043`，深度 3）：

```text
向量路三名：h-c-01(rank 0)、h-c-02(rank 1)、h-c-03(rank 2)
关键词路命中：h-t-01(rank 0)

手算 = 引擎：
  h-c-01  1/61 = 0.016393443   | 通道 vector
  h-t-01  1/61 = 0.016393443   | 通道 bm25     ← 两条精确并列
  h-c-02  1/62 = 0.016129032   | 通道 vector
  h-c-03  1/63 = 0.015873016   | 通道 vector
逐位一致：True
```

并列的那两条由**第三排序键** `record_id` 决定次序（`h-c-01` < `h-t-01`）。
**这不是"向量路赢了"**，而是"分数并列时必须有确定的第三键"——没有它，
同一份数据两次运行会给出不同的 top-1。排序键是 `(-score, best_rank, record_id)`
（`fusion.py::_assemble`），其中 `best_rank = min(channel_ranks.values())`。

### `weighted`：先归一化，再加权

```text
每一路**内部**做 min-max 归一化到 [0, 1]   → 贡献 = weight × 归一化值
weights 缺省 = {"vector": alpha, "bm25": 1 - alpha}（alpha 缺省 0.5）
```

两个细节都不是可选项（`fusion.py::weighted_score_fusion` 的 docstring）：

```text
1. 归一化必须在**每一路内部**做（而不是对合并后的分数做）
   对合并后的分数做归一化时，"这一路的最大值"里混进了另一路的量纲，
   于是 alpha 调整的是"两路混在一起之后的重心"，仍然不是权重
2. min == max 时定义为 1.0（**不是除零，也不是 0**）
   这一路只有一条命中时必然 min == max；归一化成 0.0 会让"唯一的那条证据"
   在加权和里消失（0 × weight = 0），而它明明是该路的第 1 名
```

实测的 alpha 扫描（第 5 节，两条"各自的第一名"的分数恰好是 `alpha` 与 `1 - alpha`）：

```text
alpha=0.0 → 关键词路那份排第一   alpha=0.5 → 并列（按 id）   alpha=1.0 → 向量路那份排第一
```

`weighted` 的代价（相对量）：min-max 的上下界来自**这次的候选集合**，因此
"同一路换个查询，同一份文档的归一化分不同"。这只要求**同一次融合内部**可比，
而这正是加权求和需要的全部。另一个必然结果：一路里**最差的那条**归一化成 0.0，
它在名单里得 0 分（`tests/test_hybrid_retriever.py` 有用例钉住）——
把它从名单里删掉会让 weighted 的召回比 rrf 少几条，而少了的那几条不会出现在
任何数字里，因此**保留 0 分条目而不是静默丢召回**。

### `fuse` 的三条封闭纪律

```text
strategy 必须落在 FUSION_STRATEGIES          否则 FusionError（列出可用策略）
alpha 必须落在 [0, 1] 且有限                 **任何策略下都校验**（错配置不该被"这次用另一种策略"放过）
k_rrf 必须是 >= 1 的整数                      k=0 会让贡献退化成 1/(rank+1)，头部权重过大
strategy="rrf" 却给了 weights                → FusionError，而不是"忽略它"
通道名必须落在 types.CHANNELS                "bm25x" 这种拼错会凭空多出一路只有一半证据的通道
weighted 缺某一路的权重                      → FusionError，而**不是**默认为 0
```

最后一条最重要：默认 0 会让那一路的证据**静默消失**（结果看起来正常，
只是某一类查询突然变差了——一个查不到病因的现象）。

## 4. 去重：同一条出现在多路，是**最强的证据**

"去重"这个词容易让人以为目标是"把重复的去掉，让名单短一点"。在这里它的含义
恰好相反：**同一条记录被两路独立召回，是融合能给出的最有价值的一条结论**。
因此合并时不是"丢掉一条"，而是：

```text
一条命中          合并成一条 FusedHit，**保留全部通道的证据**
分数              两路贡献之和
名次              两路里最好的那个（best_rank）
channels          按名次好坏排序的通道名（并列时向量优先）
```

`RetrievalResult.candidates`（混合模式下的口径）就是这条语义的算术表达：
它是**两路候选的并集大小**，"两路之和 − 并集"正好等于被合并掉的重复条数
（`fusion.py::contribution_counts` 与 `RetrievalResult.fusion.deduped`）。

本机实测（第 7 节，查询 `retry_budget`）：

```text
h-t-02 的融合分数 0.032787 = 1/61 + 1/61（两路各给了它一个名次 0）
下一名只剩 1/62 = 0.016129（只有一路的名次 1）
```

因此"两路都提到它"比"某一路排第一"更有分量——这是融合给出的**新信息**，
单路检索永远给不出它（它连"被几路提到"这个概念都没有）。

### `FusedHit`：这条凭什么在这里

```text
record_id       哪一条
score           融合后的分数（策略定义的口径，不是一个"相关性"）
rank            融合后的名次（0 起连续）
channels        被哪几路召回（按名次好坏排序）
channel_ranks   每一路给它的名次（{"vector": 3, "bm25": 0}）
channel_scores  每一路的**原始**分数（"两路量纲不同"因此可以对着看）
contributions   每一路对最终分数的贡献（两条一起解释"为什么是 0.0315"）
```

`contributions` 是这份形状存在的核心理由：没有它时，"`alpha=0.3` 的结果为什么与
`alpha=0.7` 一样"只能靠重跑两次来回答；有它时，读一眼
`{"vector": 0.6, "bm25": 0.0}` 就知道**关键词路这次什么都没贡献**
（那时要调的不是 alpha，而是查询说法或语料）。

逐条的数值证据**留在融合层**，不进 `RetrievalHit`：最终命中只带
`channels`（"这一条是哪几路召回的"这个可读结论）。把它们塞进每一条命中，
会让一份报告里出现三张小表 × N 条（与"`RetrievalHit` 不带向量"是同一条理由）。

## 5. 阈值与过滤的落刀位置（本课两个最要紧的位置）

### 阈值：`min_score` **只作用于向量通道**，且必须进 `notes`

```text
向量通道   有绝对标度（余弦 ∈ [-1, 1]）→ 可以设阈值，而且这次落刀的数字要报出来
关键词通道 没有绝对标度（BM25 ∈ [0, ∞)）→ **不设阈值**
```

"假的安全感"是这段话里最要紧的词：BM25 的分数分布取决于语料（词频、文档长度、
词表大小）与查询长度，因此 `min_score=3.0` 在一份 10 篇文档的手册上可能切掉一半，
在一份 100 万篇的语料上可能一条都不切。给一个**看起来能调**的参数，后果是出问题的
人去调它、发现没有稳定效果、于是转去调别的参数——**一个不存在的旋钮比没有旋钮更贵**。
它与 `config.retrieval_min_score = None` 的缺省理由是同一句话：
"标定之前给一个数字是假的安全感"。

被忽略这件事**必须进 notes**（即使这次它一条都没切、即使阈值是 `-1.0`）：
"min_score=X 只作用于向量通道"这句话要出现在每一次带阈值的混合检索里，
否则"我设了阈值为什么关键词路还在返回低分记录"会变成一个反复出现的疑问。

本机实测（第 8 节，查询 `ERR-2043`）：

```text
min_score=None → 向量路候选 10 条、切掉 0 条 → 名单 5 条
min_score=1.0  → 向量路候选  1 条、切掉 9 条 → 名单 2 条（h-c-01、h-t-01）
min_score=1.5  → 向量路候选  0 条、切掉 10 条 → 名单 1 条（h-t-01，只有关键词路召回它）
```

最后一行是本节的要点：**阈值把向量路清空时，关键词路照常给出它那几条**——
那一条的名字、通道证据（`channels=('bm25',)`）与 notes 都完整。

一个随之而来的规则变化（必须写下来，否则会误读单路那条规则）：单路检索里
"`min_score` 切光即是 `below_threshold`"，而在混合模式下 `below_threshold`
**必须同时**满足"关键词路本来就没召回"（阈值不可能清空关键词路）。
本机实测（第 9 节）：`query=QUERY_SEMANTIC` + `min_score=1.5` → `below_threshold`；
换成 `query=ERR-2043` + `min_score=1.5` 则**不是空结果**（关键词路那一条还在）。

### 过滤：`where` 必须在**两路都落刀**，而且是同一份子句

这条看似显然，但它有一个非常安静的失败模式：只在一路过滤时，另一路会把
"已经被排除的东西"重新抬进结果里——**而且不报错**。

```text
只过滤向量路    用户 where={"strategy": "structural"} → 向量路只取 structural 的
                关键词路不过滤 → 它把 fixed/semantic 的记录也捞了回来
                融合之后：结果里出现了过滤条件明确排除掉的记录
                而报告里 filter_applied=True（"过滤生效了"），没有任何异常
```

**"多而不报"比"少而不报"更危险**：少了几条还能靠对比两次运行发现，多了几条在报告里
看起来完全正常（谁也不会去逐条核对每一条命中的 `strategy`）。因此
`hybrid.py::HybridRetriever.retrieve` 第 5 步把子句合成放在**两路取候选之前的一次
调用**里，并把同一个变量交给两个调用点——这是**代码结构**上的保证，
不是注释里的约定。

本机实测（第 9 节，用真实的融合函数算一遍反面对照）：

```text
where = {"parent_doc_id": "doc-concept"}
正确（两路都过滤）：向量 4 条、关键词 0 条 → 名单里全是 doc-concept
错误（只过滤向量路）：融合后的名单里出现 1 条不满足条件的记录
                    [('h-t-01', 'doc-trouble')]，而 filter_applied=True
```

顺带一条 day066 的约定在混合模式下的对照：**缺字段的记录被两路同时排除**。
本机实测：样本里 `h-t-03` 没有 `created_at`，任何时间范围都会把它从
**向量通道与关键词通道**里一起排除（`filters.combine_where` 只合成一次）。

## 6. 结果里的多路证据：读法

| 字段 | 回答 | 落地位置 |
|------|------|---------|
| `RetrievalResult.channels` | 本次结果里出现过哪些通道（按首次出现顺序） | `types.py::RetrievalResult.channels` |
| `RetrievalResult.candidates` | **两路候选的并集**（同一条两路召回只算一次） | `hybrid.py` 第 9 步后 |
| `RetrievalResult.channel_candidates` | 每路各自的候选数（`{"vector": 10, "bm25": 1}`） | 同上 |
| `RetrievalResult.fusion` | 策略 / 参数 / 每路召回条数 / 每路贡献数 / 融合与去重条数 | `hybrid.py::HybridRetriever._fusion_summary` |
| `RetrievalHit.channel` | 哪一路把它顶上来的（贡献最大者，**并列时向量优先**） | `hybrid.py::_dominant_channel` |
| `RetrievalHit.channels` | 这一条被哪些路同时召回（按名次好坏排序） | `fusion.py::_ordered_channels` |
| `RetrievalResult.empty_reason` | 空结果的**唯一**原因（封闭清单，与 day066 同一份） | `retriever.py::_diagnose` |

`channel_candidates` 与 `candidates` 必须**一起**看：并集小于两路之和的差值，
就是"被两路同时命中"的条数（融合最强的那类证据），而这个数字只能从两个数一起读出来。

三条不变量（测试逐条钉住）：

```text
channels[0] == channel            两个问题的两个问法必须给同一个答案
len(set(channels)) == len(channels)  不伪造重复的通道
sum(contributions.values()) == score  证据必须能解释分数（RRF 与 weighted 都成立）
```

### `notes`：被忽略的参数与"某一路空着"都要点名

```text
关键词路一条都没召回：<四种成因之一，各自一句话>—— 这正是 day066 留下的那个钩子
向量通道一条都没活下来：<被阈值切光 / 库侧没有候选>—— 融合要看的信号
min_score=X **只作用于向量通道**（在融合前落刀，本次切掉 N 条）；关键词通道不设阈值……
```

四种"关键词路空着"的成因（`hybrid.py::_lexical_empty_reason`，与 `EMPTY_REASONS`
同一纪律——处置动作不同就不能合成一句）：

```text
索引是空的（0 篇）        → 先建关键词索引（from_backend）
被 where 筛成 0 条        → 放宽过滤条件（这一路的过滤与向量路是同一份）
查询词元与词表交集为空     → 纯语义改写 / 未登录词——**这是最正常的一种**，向量路正好擅长它
词元都认识但没有一篇全中   → 语料里确实没有同时提到这几个词的内容
```

`HybridRetriever.explain(result)` 在 `RetrievalResult.explain()` 的四问之上补五行：
两路明细（召回 vs 贡献）、融合参数、阈值口径、关键词索引现状、分组口径；
若带了 `where` 还会补一条**字段拼写检查**（与 `Retriever.explain` 同一份判据，
需要读库，因此只能由检索器给出）——混合模式下它更要紧：一个拼错的字段名会让
**两路同时空掉**，而"两路都空"很容易被读成"语料里没有相关内容"。

## 7. 融合参数：构造默认 ← `RetrievalQuery.extra`

`HybridRetriever(strategy=..., alpha=..., k_rrf=..., weights=...)` 是**缺省**；
`RetrievalQuery.extra` 可以逐次覆盖（day066 为这一层预留的那个槽）：

```python
query = RetrievalQuery(text="ERR-2043", extra={"strategy": "weighted", "alpha": 0.3})
result = hybrid.retrieve(query)          # result.fusion["params"] == {"alpha": 0.3}
```

键名是**封闭清单**（`fusion.FUSION_OVERRIDE_KEYS = ("strategy", "alpha", "k_rrf", "weights")`）：
多一个键抛 `FusionError` 并列出合法取值。理由与 `RagPipeline.OVERRIDE_KEYS` 完全相同——
静默忽略一个覆盖参数会让调用方以为它生效了（"我把 `alpha` 拼成 `alhpa`，
结果看起来只是没有变化"）。

走 `extra` 而不是给 `retrieve` 多加三个参数，有三个好处（写在
`api/routes.py::hybrid_query_of` 的 docstring 里）：检索器签名不变、
一份查询能被序列化回放（"这次用的是哪组参数"跟着查询走）、
参数校验只有一份实现（`fusion.fuse`）。

**构造期就校验**：`HybridRetriever.__init__` 拿一份空输入先过一遍 `fuse`，
于是参数写错在**装配那一刻**就响，而不是等到第一次检索才响——后者会把一个配置
错误伪装成"这次请求碰巧不对"（那是两种完全不同的动作）。

`config.py` 里新增的六项（`retrieval_hybrid_*` / `retrieval_bm25_*`）与既有的
`retrieval_*` 是同一类（都不改变任何产物的字节，只改变"怎么问"），只多了一条
**"新能力默认不生效"**：`retrieval_hybrid_enabled = False`。

## 8. 端点清单（现在七个）

| 端点 | 方法 | 入参 | 落到哪个入口 |
|------|------|------|-------------|
| `/retrieval/status` | GET | — | `Retriever.describe()`（`defaults` 里含新增六项） |
| `/retrieval/search` | POST | `query` + 七个检索参数 | `Retriever.retrieve` → `RetrievalResult` |
| `/retrieval/explain` | POST | 同上 | `Retriever.explain` |
| `/retrieval/answer` | POST | `question` + 同上 | `RagPipeline.answer` |
| `/retrieval/routes` | GET | — | `StoreRouter.report()` |
| **`/retrieval/hybrid`** | **POST** | 同 `/retrieval/search` 的八个字段 **+ `strategy` / `alpha` / `k_rrf`** | `hybrid.py::HybridRetriever.retrieve` |
| **`/retrieval/lexical/status`** | **GET** | — | `lexical.py::LexicalIndex.describe()` |

三条本组独有的约定（`api/schemas.py` 的段注释）：

```text
1. 融合参数挂在 RetrievalQuery.extra 上（请求体里是三个具名字段）
2. 缺省关闭：retrieval_hybrid_enabled=False 时 /retrieval/hybrid 返回 **400**
   并指出开关名与注入出路——**不静默退回单路**（那会让你以为自己在看
   混合检索的结果，而实际上一次关键词检索都没发生）
3. 两路的证据原样出门：channel_candidates / fusion 直接来自
   RetrievalResult.to_dict()，端点不重新解释它们
```

两条与错误映射有关的细节：

```text
注入了一个类型不对的对象（app.state.retrieval_hybrid 不是 HybridRetriever）
    → **500**：这是装配写错了，不是请求参数写错了。降级成 400 会让一个必然
      复现的配置错误看起来像"这次请求碰巧不对"
关键词索引每次请求现建（from_backend）    → 代价是 O(N) 分词；语料大时请注入
      app.state.retrieval_lexical（与多索引注入 app.state.retrieval_router 同一条出路）
```

一条必须写下来的**边界**：请求体里多余的 JSON 字段会被 pydantic 忽略
（与其余端点一致），因此拼错的字段名**不会**报错（`{"weight": 0.9}` 会得到 200
和一份参数没变的融合摘要）。这与库内 `extra` 的"写错键名报错"不是同一条通道：
`extra` 是库内 API（脚本与评估直接用），HTTP 这一层只有三个具名字段。

## 9. 常见误区（12 条）

| # | 误区 | 事实 | 证据（可核对） |
|---|------|------|--------------|
| 1 | "融合就是把两路分数加起来" | 两路量纲**不可比**（cos ∈ [-1,1]、BM25 ∈ [0,∞)），直接相加会让 BM25 单方面决定名次 | `fusion.py` 模块 docstring；第 5 节实测 |
| 2 | "RRF 也用分数" | RRF 只用名次（`1/(k+rank+1)`）；把某一路的分数放大 100 倍结果逐位不变 | `test_hybrid_fusion.py::test_paper_formula_ignores_the_raw_scores_entirely` |
| 3 | "min == max 时要除零/归一化成 0" | 定义为 **1.0**：一路只有一条命中时它就是该路第一名，归 0 会让唯一证据被权重乘没 | `fusion.py::_min_max_normalize` |
| 4 | "归一化可以对合并后的分数做" | 必须**每一路内部**做，否则 alpha 调的是"混在一起之后的重心"，不是权重 | `weighted_score_fusion` 的 docstring |
| 5 | "min_score 对两路都生效" | 只作用于向量通道、只在融合前落刀；关键词通道不设阈值（假的安全感） | `hybrid.py` 阈值语义一节；第 8 节实测 |
| 6 | "阈值切光就是空结果" | 混合模式下 `below_threshold` 还要"关键词路本来就没召回"——阈值不可能清空关键词路 | 第 9 节实测 |
| 7 | "过滤只要向量路做了就行" | 必须两路都落刀且**同一份子句**，否则会把已排除的记录抬回来且**不报错** | `hybrid.py` 模块 docstring 的坑；第 9 节反面对照 |
| 8 | "关键词路返回的就是它排名里的那些" | 它**只返回至少命中一个词元的文档**：0 分文档会被排除（否则 RRF 会把"毫无关系"当"第 N 名证据"） | `lexical.py::LexicalIndex.search`；`test_hybrid_bm25.py::TestZeroScoreExclusion` |
| 9 | "同一条被两路召回是重复，要去掉" | 那是最强的证据：合并成一条并保留全部通道证据（`channels` 两条、贡献相加） | `fusion.py` 去重语义一节；第 7 节实测 |
| 10 | "关键词索引是空的会报错" | 0 篇是**合法状态**（与向量路 `no_data` 同一口径）；但"有文档却全无词元"抛 `LexicalError` | `lexical.py::search` 的第 4 步与中毒索引分支 |
| 11 | "`channel` 就是唯一的通道" | `channel` 是贡献最大的那一路（并列时向量优先），`channels` 才是全部证据；`() ` 表示"未记录多路证据" | `types.py::RetrievalHit`、`hybrid.py::_dominant_channel` |
| 12 | "混合检索缺省就开着了" | `retrieval_hybrid_enabled` 缺省 **False**，`/retrieval/hybrid` 会返回 400 并说明如何打开 | `config.py`、`api/routes.py::retrieval_hybrid` |

再补三条同性质的（写下来省一次排查）：

```text
13. "关键词索引会跟着向量库自动更新"  → 不会：它是一次性快照（LexicalIndex 刻意
                                        不持久化、不做增量）。库变了要重建或注入
14. "`where` 与 `time_range` 可以同时写同一个字段" → combine_where 直接报 QueryError
                                        （两路会一起被拒，因为子句在取候选之前就合成好了）
15. "阈值写 `>` 也行"                → 用 `>=`，与 VectorBackend.query(min_score=...) 逐字一致；
                                        阈值恰好等于分数时那种差别**必然出现**
```
