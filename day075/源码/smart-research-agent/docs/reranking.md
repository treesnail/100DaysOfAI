# 重排序手册（day068 / M6-D7）

> 本文与代码**同源**：每一条结论都标了落地位置（`retrieval/xxx.py::函数名`），都能在
> `smart_research_agent/retrieval/rerank.py`、`retriever.py`（第 6.5 步）、`hybrid.py`
> （第 10.5 步）、`types.py`、`errors.py` 与 `config.py` 的 `retrieval_rerank_*` 组里逐字找到。
>
> **本文的数字只有三个来源**，其余一律写"以你实际运行时输出为准"：
>
> ```text
> 代码里写死的常量    DEFAULT_RERANK_TOP_N = 20、DEFAULT_RERANK_WEIGHT = 0.5、
>                     DEFAULT_RERANK_BATCH = 16、IDEAL_LEN = 240、
>                     FEATURE_WEIGHTS = (0.5, 0.25, 0.125, 0.125)、
>                     RERANK_MODES = ("replace", "blend")、RERANK_OVERRIDE_KEYS、
>                     retrieval_rerank_enabled = False…
> 由它们算出的算术    0.5 + 0.25 + 0.125 + 0.125 = 1.0（逐位）；proximity = 1/(1+span)；
>                     length_penalty = min(1.0, 240/336) = 0.7142857…；
>                     DCG@5（唯一相关命中在第 5 位）= 1/log2(6) = 0.386853…
> 本机实测           本文标注"本机实测"的行来自同日的离线演示 `scripts/rerank_demo.py`
>                    （FlatVectorStore + 一张写死的向量表 + 12 条记录，零网络、零 Key）；
>                    正式数字请以 `outputs/rerank_demo.txt` 与
>                    `python -m pytest tests/test_rerank_*.py` 的实际输出为准
> ```
>
> 全层的落地状态（`retrieval/__init__.py` 的原文）："导出范围（**十一个模块全部落地**）：
> `errors` / `types` / `filters` / `retriever` / `routing` / `context` / `pipeline` /
> `lexical` / `fusion` / `hybrid` / `rerank`（最后一个是 day068 新增的）"。端点层
> （`smart_research_agent/api/routes.py` 的 `/retrieval/*`）与本层同日交付，但本文只按
> **索引式写法**引用它，**不复述任何响应字段**。

## 1. 一句话定位：重排解决的是"第一阶段排不准"

```text
第一阶段（召回）  双塔 / BM25 / 融合 → 几十~几百条候选    追求"别漏"，允许粗糙
第二阶段（重排）  交叉编码器只对**前 N 条**打分 → 重排       追求"排准"，必须精确
```

day066 的检索器回答"取回哪几条"，day067 的混合检索回答"两路怎么合"，day068 回答第三个
问题：**这一批候选里，哪一条真的最相关**。它的入场条件很具体——第一阶段已经把候选缩小到
一个可接受的范围，否则重排的成本无法承受（见"重排的代价"）。

### 第一阶段为什么排不准（双编码器的结构性缺陷）

```text
双塔（bi-encoder）        查询 → 向量 ┐
                         文档 → 向量 ┘ → 点积        文档侧可预计算，但两两不相见
交叉编码器（cross-encoder） [CLS] 查询 [SEP] 文档 → transformer → 相关性 logit
```

`rerank.py` 模块 docstring 把这件事写成一句话：双塔"查询与文档**从来没有见过面**：两边各自
被压成一个点，相似度只是两个点的夹角"。后果不是"分数不准"，而是**分数与相关性脱钩的方式是
系统性的**：编码器没见过的东西（编号、函数名、报错码）在向量空间里只能"差不多就行"。

本机实测（演示脚本第 1 节，查询 `ERR-2043`、12 条候选）：第一阶段第 1 名是 d-01（余弦
`1.000000`，但它一个编号都不含），真正含这个编号的三条排在第 4、5、6 名（d-11 `0.480000` /
d-12 `0.360000` / d-05 `0.000000`）；重排之后它们变成前三（`0.916667` / `0.916667` /
`0.880952`）。"第一名是 d-01 而它一个编号都不含"就是第一阶段排不准的样子——它不是 bug：
d-01 的余弦是 1.0，在"意思像不像"这个尺度上它确实是第一名，而这次查询要的是"这个编号在不在"。

### 重排的代价：为什么只对前 N 条做

```text
双塔对 N 篇文档是 N 次点积      可批量、可在索引里做、可预计算文档侧
交叉编码器是 N 次前向推理      每一对都要把两个文本拼起来跑一遍模型
```

成本差两个数量级，于是窗口成了本模块的第一条纪律（第 3 节）：`top_n` 是可配的旋钮
（`retrieval_rerank_top_n`，缺省 20），而它的**代价**必须写出来——第 N+1 条永远没有机会证明自己。

### 模块地图（每个模块回答一个问题）

```text
rerank.py             交叉编码器替身（四个可手算特征）+ 窗口纪律 + replace/blend
                      + 三条提升指标（recall@k / RR / nDCG@k）与 LiftReport
retriever.py 第 6.5 步  单路：阈值落刀之后、多样性裁剪之前调 rerank_hits
hybrid.py    第 10.5 步  混合：融合之后、多样性裁剪之前调 rerank_hits
types.py              两个新字段（rerank_score / stage1_rank）+ 第四个数（dropped_by_rerank）
                      + 两条改写过的 RETRIEVAL_LIMITATIONS
errors.py             RerankError（继承 QueryError："这次调用/这次装配写错了参数"）
config.py             六个 retrieval_rerank_*，缺省 retrieval_rerank_enabled = False
```

两处集成分**共用同一对原语**：`rerank_hits`（纯函数：切窗口 / 打分 / 排序）与
`retriever._rebuild_hits`（写回：次序与名次取自重排，其余全部取自原命中）。两份手写的写回
逻辑一定会分家（一份忘了 `channels`、一份忘了 `stage1_rank`），而分家之后"重排把多路证据
弄丢了"这种 bug **不会报错**，只会让报告少一列。

### 已知边界与明确不做的事

`types.RETRIEVAL_LIMITATIONS` 里 day068 改写过的两条：

| 边界 | 原文要点 |
|------|---------|
| 窗口之外 | 重排只作用于前 N 条：窗口之外的命中一条都不打分，名次只由第一阶段的分数决定 |
| 教学级替身 | `cross-encoder-teaching-v1` 没有真实语义，分数只能用于**相对排序**，不能当成"相关性"读 |

`RETRIEVAL_OUT_OF_SCOPE` 里的"学习型排序与学习型融合权重"仍然成立（本课的权重是显式参数，
不从数据里学）。另外三件明确不做的事：**不做模型下载**（本模块不按名字加载模型，`model`
只进报告）、**不做检索**（重排只处理上游已经给的候选）、**不做权限**（`where` 是元数据
筛选，不是访问控制）。

## 2. 交叉编码器：把查询与文档一起读

### 诚实声明：教学版为什么用四维可手算特征

真实交叉编码器（bge-reranker / ms-marco-MiniLM 之类）要下载权重、要推理框架、通常还要 GPU
——**这三件事本课一件都不许有**（离线、确定性、零依赖）。因此 `CrossEncoderReranker` 是一个
**确定性替身**：分数由四个可手算的特征加权而成。docstring 不含糊其辞："它认不出同义改写
（'重算'与'重新计算'在它眼里是两个不同的字串）、它读不懂语序（'甲打败了乙'与'乙打败了甲'
的特征几乎一样）"，并写明"**生产环境应当把它替换为真实重排模型**"（与
`llm.embedding.MockEmbedding` 的诚实写法同源：形状留好、替身说清）。换来的是本课要的东西：
**分数可以手算核对**——一个"必须先下载 2 GB 权重才能复现"的分数，在"这次评估为什么变了"
这个问题前面是哑的。替身还刻意暴露一处差异：`dimension` 返回 **4** 而不是 1（真实模型只
产出一个 logit）。若有人拿 `dimension == 4` 去和真实模型对齐，他会**立刻**发现对不上。

### 四个特征的精确定义（全部落在 `[0, 1]`，因此加权和也落在 `[0, 1]`）

```text
term_coverage    |{t ∈ 去重(query_terms) : t ∈ doc_terms}| / |去重(query_terms)|
                 查询一个词元都没有（纯标点）时定义为 0.0，**不是 1.0**：
                 没有证据不等于证据充分——算成 1.0 会让"什么都匹配"的文档白拿 0.5 分
exact_phrase     query.strip().casefold() 是 text.casefold() 的子串 → 1.0，否则 0.0
                 strip 之后为空串时给 0.0（空串"在"任何文本里）
proximity        1 / (1 + span)，span = 覆盖**全部**查询词元的最短连续窗口宽度；
                 无法覆盖时是 0.0（不是"很远"）
length_penalty   min(1.0, IDEAL_LEN / max(len(text), 1))，IDEAL_LEN = 240
```

`term_coverage` 与 `proximity` 复用 `lexical.tokenize`（**同一套分词**）：重排与关键词路如果
各用一套分词，"同一个查询能不能命中同一篇"就会有两份答案，而这两份答案在报告里长得完全一样
——那是评估最怕的一种不一致（同一条纪律也用在 day067 的"两路共用一份过滤子句"上）。
`RerankFeatures.__post_init__` 逐项校验四个特征（非数字、`nan`/`inf`、越界都报 `RerankError`
——越界会让 `min_score` 的可标定性失去根基）；`_min_span` 是滑动窗口（`O(len(tokens))`），
它返回**宽度**而不是位置（"跨度"是长度概念，位置依赖文档怎么切、跨文档不可比），
而且找不到时给 `None` 而不用 `-1`/`0` 当哨兵（`-1` 会一路漏进 `1/(1+span)` 变成 `1/0`）。

### `FEATURE_WEIGHTS`：取值与"和逐位等于 1.0"

```text
FEATURE_WEIGHTS = (0.5, 0.25, 0.125, 0.125)     顺序 = RERANK_FEATURE_NAMES（封闭清单）
score = 0.5·coverage + 0.25·phrase + 0.125·proximity + 0.125·length_penalty
```

四个数取 1/2、1/4、1/8、1/8——它们都是**二进制精确可表示**的数，因此和**逐位等于 1.0**，
`_check_feature_weights` 可以用**精确比较**校验（`math.fsum(...) != 1.0` 就报 `RerankError`），
不必引入 epsilon。这与 `fusion._min_max_normalize` 的"精确的零跨度"是同一条纪律：一个凭空取出
的 tolerance 会变成第三个需要标定的参数，而它的表现是"某些权重组合明明是 1 却被拒绝了"。
排序含义：覆盖率是主项（替代词汇匹配），整句包含是强证据，邻近度与长度各占 1/8（对冲项）。

本机实测（四特征 + 加权和，手算与引擎逐位一致；完整表见 `outputs/rerank_demo.txt` 第 3 节）：

```text
整句包含、编号紧挨 d-05 → 覆盖 1.0、整句 1.0、邻近 0.3333、长度 1.0000 → 0.916667
整句包含但正文 336 字 d-12 → 同上，但长度 0.7143 → 0.880952      （240/336 = 0.714286）
一个编号都不含 d-01（阶段一第 1 名） → 0.0、0.0、0.0、1.0000 → 0.125000
覆盖率 6/6、邻近度很低 d-06 → 1.0、0.0、0.0370（=1/27，span=26）、1.0000 → 0.629630
并列：d-05 与 d-11 的四项**逐位相同** → 两个分数都 0.9166666666666666
```

### 长度惩罚：它对冲的是"冗长偏置"

交叉编码器的位置偏置里最顽固的一条是**冗长偏好**：长文档里"恰好出现某个词"的概率本来就高，
注意力也更容易找到一段能与查询对上的文字，于是分数系统性地偏高。把长度罚项放进分数，是在
**重排这一侧**对冲它。而"长文档更该被看见吗"没有普适答案，因此它只占 1/8 的权重，不是硬规则；
`IDEAL_LEN = 240` 按本课语料的块长标定（day062 的结构化分块大多落在 150~350 字），并且它是
一个可以显式改的旋钮（`CrossEncoderReranker(ideal_len=...)`），不是藏在实现里的常数。

接真实模型只需要写一个 `BaseReranker` 子类：`score_pairs` 里做真正的
`[CLS] query [SEP] doc` 前向推理，`explain_pairs` 照旧返回空字典（真实模型没有"可手算的
四个特征"，硬凑一份反而是编出来的证据）——**本模块的其余部分不需要改动**：窗口纪律、两种
模式、回填的字段、提升指标都与"分数是怎么来的"无关。

## 3. 窗口纪律：只对前 N 条打分

```text
好消息   成本从 O(候选) 降到 O(N)，N 是一个可配的旋钮（retrieval_rerank_top_n）
坏消息   第 N+1 条**永远没有机会**证明自己——它的名次只由第一阶段的分数决定
```

坏消息那一半**无条件进 notes**（`_build_notes` 的第一个条目，无论窗口是否真的切了东西）：
只报"重排生效了"却不报"有一批条连分都没拿到"，等于把一次**有边界的**改进说成了一次全面的
改进。这条注记是本模块最容易被删掉、也最不该删掉的一行。

### 窗口内 / 窗口外的分界

```text
输入     上游给的命中序列（单路：阈值之后；混合：融合之后）
窗口     candidates[:top_n]      → 每一条都送进重排器打分
尾部     candidates[top_n:]      → **一个都不进重排器**，按原顺序接在后面
名次     整份名单一起重编成 0 起连续（"第 3 条"必须只有一个含义）
```

`RerankResult.scored` 恰好是 `min(candidates, top_n)`，`window` 属性是 `min(scored, top_n)`；
这两个数必须一起读——"窗口有没有生效"只看它们。本机实测（候选 12 条）：

```text
top_n=5    打分了 5 条、没打分 7 条；账：candidates=12、scored=5、window=5
           打过分  d-11、d-01、d-04、d-02、d-03
           没打分  d-12、d-05、d-06、d-07、d-08、d-09、d-10
           重排器计数器差量：calls +1、scored_pairs +5、batches +1
top_n=10   打分了 10 条、没打分 2 条（d-09、d-10）；scored=10、window=10
```

三个计数器（`calls` / `scored_pairs` / `batches`）是"**真的跑了几次**"的账，**只增不减**
（一个实例可以被多次查询复用，"这次调用打了多少分"用调用前后的差值回答）；"top_n 之外的条数
一次都没被打分"这条断言就是靠它们验证的（`scored_pairs` 的差量恰好等于 `window`）。空输入时
**不调用重排器**——一次都不打分，因此计数器不动，而"这条查询没有候选"由 `empty_reason` 说清楚。

### 尾部为什么是 `score=0.0` 而不是"最不相关"

`RerankHit` 的口径：窗口外的条 `score = 0.0`、`features = {}`、`scored = False`、
`stage1_rank` 保持原值。**不是 `None`** 是因为形状上要保住"每条都有一个分数"，而"它其实没被
打分"由 `scored` 标志与 `notes` 里那条无条件注记负责；**不会被误读**是因为窗口外的那批按原
顺序排在窗口结果**之后**——名次与分数在这里是两套证据，而 0.0 **不参与排序**（排序只发生在
窗口内那一段）。

### 为什么先重排、后做多样性裁剪

```text
多样性的动作   按 doc_id 分组，把同一文档的第 2、3… 条**丢掉**
若先裁后重排   那些被丢掉的候选再也回不来——重排器**看不到它们**，
              于是"同一文档里的第 2 段其实更贴题"这个事实永远无法被发现
因此顺序       重排（在完整的候选集上比较）→ 多样性（在重排后的次序上取舍）
```

代价也要写下来：重排**改变了"哪几条会被多样性命中"**——同一文档原本排第 2 的那条被重排提到
第 1 之后，它会占据那个名额，而原先排第 1 的那条反而可能被挤掉。这不是 bug，而是"重排的结论
被尊重"的必然结果。

两处的落刀位置（"重排是第二段"在代码里的样子）：**单路 `retriever.py` 第 6.5 步**是阈值落刀
之后、多样性裁剪之前；**混合 `hybrid.py` 第 10.5 步**是融合之后、多样性裁剪之前。重排放在
融合之后是因为：融合已经把所有通道的证据摆好了，重排看到的每一段文本都是"被某些路真实召回过
的"，而不是某一路的内部抽签结果。

## 4. 两种模式：`replace` 与 `blend`

### 第一件事：跨阶段的分数量纲不可比（与 `fusion` 同一条纪律）

```text
第一阶段   余弦 ∈ [-1, 1] / BM25 ∈ [0, ∞) / RRF 是 1/(k+r) 之和 /
           weighted 融合又是一套归一化分 —— 量纲取决于上游，**没有上界**
重排分     教学替身落在 [0, 1]
```

两个数字放在一起比较是**类型错误**，而它不会报错：`0.48` 与 `0.82` 都是浮点数，`sorted` 照单
全收。于是"直接加权"会把**量纲差当成相关性差**——这正是 day067 在融合层讲过的那件事，在这里
是**第二次**出现，因此这里**复用同一个函数**（`fusion._min_max_normalize`）：同一条纪律只写
一份实现，两份实现一定会分家。

```text
replace   窗口内**只**按重排分重排（第一阶段的分数降级为兜底的排序键）
blend     窗口内把两份分数**各自 min-max 归一化**到 [0, 1]，再按 weight 加权
          blended = weight × 归一化(重排分) + (1 - weight) × 归一化(阶段一分)
```

缺省是 `replace`（`config.retrieval_rerank_mode = "replace"`），理由是多数时候第一阶段的分数
只用来**召回**，而"重排之后还把它混进来"需要额外的理由（它可能就是错的那些条）。
`DEFAULT_RERANK_WEIGHT = 0.5` 是"两边都不占先"的起点：它不代表两个信号一样有用，只代表
**还没有证据**说明该偏向哪一边——先跑一遍 `replace` 看清重排把谁动了，再决定要不要 blend。

### `blend` 的手算演示（`top_n=6`、`weight=0.5`）

```text
窗口内的两份分数：阶段一 min=0.360000 / max=1.000000；重排分 min=0.125000 / max=0.916667

id    |  阶段一  |  重排分  | 归一阶段一 | 归一重排 | blended手算 | blended引擎
d-01  | 1.000000 | 0.125000 |  1.000000 | 0.000000 |    0.500000 |    0.500000
d-04  | 0.960000 | 0.125000 |  0.937500 | 0.000000 |    0.468750 |    0.468750
d-02  | 0.800000 | 0.125000 |  0.687500 | 0.000000 |    0.343750 |    0.343750
d-03  | 0.600000 | 0.125000 |  0.375000 | 0.000000 |    0.187500 |    0.187500
d-11  | 0.480000 | 0.916667 |  0.187500 | 1.000000 |    0.593750 |    0.593750
d-12  | 0.360000 | 0.880952 |  0.000000 | 0.954887 |    0.477444 |    0.477444
```

手算与引擎逐项一致（差 0.0，两边是同一串运算）。次序对照：`replace` 是
`d-11、d-12、d-01、d-04、d-02、d-03`，`blend` 是 `d-11、d-01、d-12、d-04、d-02、d-03`
——差别只在 d-01 与 d-12 互换：blend 保留了"第一阶段也说得过去"的 d-01（它归一化后是 1.0），
代价是把 d-12 压到第 3。

### `blend` 的代价：归一化随窗口变化

归一化只在**这一次窗口内**做，因此同一篇文档的归一化分**随窗口变化**（换一条查询、换一个
`top_n`，它都会变）。本机实测（同一份数据、同一个 weight，只把 `top_n` 从 6 换到 8）：
`d-12` 的 `blended = 0.477444` → `0.657444`，差 **+0.180000**。这是相对量的固有性质，不是
实现缺陷——它只要求"**同一次重排内部可比**"，而那正是加权求和需要的全部。与本条并列的另一条
口径：`min == max` 时归一化定义为 **1.0**（不是除零、也不是 0）——窗口内只有一条时它必然
`min == max`，归一化成 0.0 会让"唯一的那条证据"在加权和里消失，而它明明就是那一段的第一名。

### 第三个兜底键：`record_id`

```text
replace  排序键  (-score,   stage1_rank, record_id)
blend    排序键  (-blended, stage1_rank, record_id)
```

三个键缺一不可：只按分数排序时，"两条分数逐位相同"会让名次取决于 `sorted` 的稳定性（那是
**没写下来的**排序规则），而同一份数据两次运行给出不同 top-1 是最难查的一类 bug。两条必须
说清的事实：**第二键是 `stage1_rank`（输入序列的位置）而不是上游的 `hit.rank`**（阈值落刀
之后 `hit.rank` 会带空洞 0、1、3、4，而排序第二键需要一份稠密的次序；"没有空洞"时两者逐位
相同）；**并列因此实际由"输入位置"决定**——本机实测 d-05 与 d-11 的重排分逐位相同
（0.9166666666666666），d-11 的 `stage1_rank=4` 排在 d-05 的 6 之前（**不是**由 `record_id`
决定：d-05 < d-11 只是碰巧同向）。第三键在这条路径上不会被触发（`stage1_rank` 已经把每一条
区分开了），它写在那里是**防御性兜底**。另一次实测：把库的插入顺序反过来，第一阶段名单逐位不变。

## 5. 阈值与账：第四个 `dropped_*`

### `min_score` 只作用于**被打分过的**命中

```text
比的是重排分         不是第一阶段的分数（这是本层第一个"口径切换"）
作用于被打分过的条    窗口内：min(候选, top_n) 条
窗口外的条           连分数都没有 → **既不会被它切、也不会被它救**
记账                dropped_by_min_score（本次切掉几条）
```

判据是 `>=`（`payload.score < float(min_score)` 才丢），与 `VectorBackend.query(min_score=...)`、
`retriever._apply_threshold` 逐字一致：**阈值恰好等于分数时那种差别必然出现**，两处写法不同
就会变成"同一批数据两份名单"。为什么这里可以设阈值（而 BM25 那边不行）：**可标定性**。
教学替身的重排分落在 `[0, 1]`，因此阈值有绝对含义；而 BM25 没有绝对标度，给它一个数字是
"假的安全感"。缺省仍是 `None`（标定之前给一个数字是假的安全感），差别只是**标定难度低得多**。

### `dropped_by_rerank` 为什么不并进 `dropped_by_top_k`

`RetrievalResult` 上现在有**四道减法**，各自的数字与"该调哪个参数"一一对应：

```text
dropped_below_threshold   第一阶段阈值切掉几条   → 调 retrieval_min_score
dropped_by_diversity      每文档上限挤掉几条     → 调 retrieval_max_per_doc
dropped_by_rerank         **重排阈值**切掉几条    → 调 retrieval_rerank_min_score（day068）
dropped_by_top_k          截断丢掉几条           → 不需要调，它只是"取够了"
```

合成一个 `dropped` 是"看起来无害、实际上致命的简化"：合成之后"该调哪一个参数"这个判据就没了。
重排阈值是**第四个旋钮**，而它切的条数与"top_k 截断"是两件完全不同的事——前者说明阈值定得
太高，后者只是取够了。一条刻意的取舍（`retriever._diagnose` 的 docstring）：重排阈值切出来的
空结果**沿用 `below_threshold` 这个取值**，不新增第六种 `empty_reason`——重排阈值也是"阈值"，
处置动作与第一阶段阈值一样；代价是这一支说不出"是第几阶段的阈值"（要知道请看
`RetrievalResult.rerank` 与 `dropped_by_rerank`）；而 `EMPTY_REASONS` 是一份**封闭清单**
（端点的响应体里整份端出去），为它加第六个取值会改动一处对外协议。

### "只切窗口内的"这个代价，必须进 `notes`

```text
min_score=X **只作用于被打分过的 N 条**（本次切掉 M 条）：窗口外的 tail 条不受它影响
——它们连分数都没有。比的是重排分（教学替身落在 [0, 1]，因此这个阈值可以标定；
与关键词路 BM25 的'没有绝对标度'正好相反）。
```

这条注记**只要写了 `min_score` 就出现**（即使这次一条都没切），理由与 day067 的"`min_score`
只作用于向量通道必须进 notes"完全相同：一个**被静默忽略的作用范围**会让"我设了阈值为什么
尾部还在"变成一个反复出现的疑问。

### 本机实测（12 条候选；分数分布 0.916667 ×2、0.880952 ×1、0.125 ×9）

```text
top_n=5、min_score=0.9   窗口 5 条 → 留 1 条（d-11）、切 4 条；窗口外 7 条（score 全 0.0）
                        **一条都没被切**；命中 8 条
top_n=5、min_score=2.0   窗口 5 条 → 留 0 条、切 5 条；窗口外 7 条照样在 → 命中 7 条、
                        empty_reason 仍是空串
top_n=20、min_score=2.0  窗口 12 条 → 留 0 条、切 12 条；没有尾部 → 命中 0 条，
                        empty_reason = "重排阈值 2.0 把窗口内的 12 条全切掉了…"
```

第二行是本节最要紧的一行：**"切光窗口"与"空结果"是两件事**。窗口被切空时，窗口外那批
（`score=0.0`、`scored=False`）照样在名单里，因此结果**不是空的**；只有窗口覆盖了全部候选、
且没有尾部时，才真的走到空结果那条路。`rerank_hits` 为这条合法路径**显式**写了
`empty_reason` 与一条注记（"这次重排一条都没留下：**不是**重排器失败，而是阈值定得比全部
分数都高"）——否则"切完之后为空"会变成一次构造期异常（`RerankResult` 的不变量要求
"没有命中时必须给出一句话"）。

## 6. 结果里的重排证据：读法

### 逐条命中上的两个新字段（`RetrievalHit`）

| 字段 | 回答 | 落地位置 |
|------|------|---------|
| `RetrievalHit.rerank_score` | 重排分（`None` = 这次没开重排） | `retriever._rebuild_hits` |
| `RetrievalHit.stage1_rank` | 重排**前**它在输入序列里的位置（0 起；`None` = 没开重排） | 同上 |

两条硬事实：**`score` 的口径没有变**（它仍是第一阶段/融合的分数），于是"`score` 越大越靠前"
这条不变量在重排过的名单里**不再成立**（本机实测单路名单的 score 序列 `[0.48, 0.0, 0.36, 1.0,
0.96]` 不再单调）——两个分数各占一个字段：**排序依据看 `rerank_score`，跨版本对比看 `score`**；
**`stage1_rank` 记的是输入序列的位置**，不是上游的 `hit.rank`（阈值落刀之后 `hit.rank` 会带
空洞，"它原来在第几"要一份稠密次序）。`RerankHit.moved = stage1_rank - rank`（**正数 = 上移**）
也把"别人被切掉"算进来：阈值切掉几条之后，后面每条 `rank` 前移，`moved` 也是正的——它是
**名次差**，不是"重排器认为它更好"（要问后者请看 `features`）。

### `RetrievalResult.rerank` 摘要的每个键

形状只定义一次（`RerankResult.to_summary()`），单路与混合两处集成共用它——两份手写的摘要
一定会分家（一个多了 `mode`、一个忘了 `scored`）。

```text
model                这次是谁打的分（缺省 = reranker.name）
mode                 'replace' / 'blend'
params               {"top_n": …, "weight": …}   ← 注意 top_n 在这里，不在顶层
candidates           重排的**输入条数**（阈值/融合之后交给它的那批）
scored               真正被打分的条数（= min(candidates, top_n)）
window               min(scored, top_n)
dropped_by_min_score 重排阈值切掉几条
moved                名次真的变了的条数（len(moved_ids())）
empty_reason         没有命中时的唯一原因（有命中时是空串）
```

**两个键名要看清**：`rerank["candidates"]` 是重排的输入条数，**不等于**
`RetrievalResult.candidates`（库侧过滤之后的候选数）；`window` 是本次真正送进重排器的那一段
长度。沿用形状自己的键名（而不是另起 `incoming` 之类的别名）是为了让
`result.rerank["candidates"]` 与 `RerankResult.candidates` 逐字对得上；代价就是上面这一行
必须写下来——**同名字段两个口径**，是这份代码里最容易读错的一处。摘要**刻意不含正文与特征**：
它进的是每一份检索结果的公共字段，而"20 条 × 4 个特征"放在那里会把它变成第二份 `hits`。

### `explain()` 多出来的那一行

两处集成各补一行（`retriever._rerank_explain_line` 与 `hybrid.explain`）：

```text
单路  重排口径：开启——重排器 'cross-encoder-teaching-v1'（教学级交叉编码器替身，4 维可手算
      特征，批大小 16），模式 'replace'、只对前 8 条打分、权重 0.5、重排阈值 不设阈值；
      它发生在阈值落刀之后、多样性裁剪之前（先让所有候选上桌，再做取舍）
混合  重排口径：重排在**融合之后、多样性之前**（第 10.5 步）——本层默认 enabled=True、
      只对前 8 条打分、模式 'replace'、权重 0.5（可用 RetrievalQuery.extra 逐次覆盖：…）；
      关着的时候一个字段都不会被改写（结果与 day067 逐位相同）
```

`RerankResult.explain()` 自己给出四段（对应关系固定）：窗口纪律（只对多少条打了分、外面还有
多少条）→ 模式与量纲 → 名次变化 → 阈值与注记。没有名次变化时它**不会**沉默，而是写着
"这不代表重排没用，只代表这次它没有改动任何一对相邻次序"。**一处读法上的坑**（本机实测发现，
已经修掉）：`RetrievalResult.explain()` 里那句"只对前 **N** 条打分"必须读摘要的
`params["top_n"]`（回落 `window`）——摘要顶层**没有** `top_n` 这个键，早先按顶层键读时它会
渲染成"只对前 0 条打分（真正打分 8 条）"，一句自相矛盾的诊断。修法是"多一个同义的顶层键"
不如"读已有的那一个"：`params.top_n` 与 `result.rerank["window"]` 是同一件事的**一份**记录。
看真实窗口请读 `params.top_n` 与 `scored` / `window`（`result.rerank` 是权威那一份）。

## 7. 评估提升：`recall@k` / `RR` / `nDCG@k`

"重排到底有没有用"不是靠感觉回答的。三条指标的公式写在 `rerank.py` 的模块 docstring 里，
实现是三个**纯函数**（`recall_at_k` / `reciprocal_rank` / `ndcg_at_k`），因此这张对照表可以
被手算核对——不需要任何真实模型。

```text
recall@k   = |前 k 条 ∩ 金标准| / |金标准|                   （命中面：有没有捞回来）
RR         = 1 / (第一个相关命中的位置 + 1)（无则 0.0）       （头部质量）
nDCG@k     = DCG@k / IDCG@k
             DCG@k  = Σ_{i=0}^{k-1} rel_i / log2(i + 2)       二值增益 rel_i ∈ {0,1}
             IDCG@k = Σ_{i=0}^{m-1} 1 / log2(i + 2)，m = min(|金标准|, k)
```

三个细节都是刻意选的：折损用 `log2(i + 2)`（`i` 从 0 起，因此第一条的折损是 `log2(2)=1`，
不折损）；理想名单用 `min(|金标准|, k)` 条（金标准比 k 多时，理想 DCG 也只数前 k 条）；
**重复 id 只算一次**（第 2 次出现按 0 增益——否则一份"把同一条刷满"的名单会拿到大于 1 的
nDCG，而分母只按去重的金标准算）。两个口径写在明面上，因为它们决定了同一个数字：
`recall@k` 的分母是**去重后**的金标准条数，分子是前 k 条里落在金标准里的**条数**；
`reciprocal_rank` 的位置从 0 起——与 `fusion` 里 RRF 的 `1/(k+rank+1)` 是同一个形状
（那里从名次算、这里从位置算），两处的"第几名"都从 0 起，这一点一致很重要。

三者必须一起看：只看 `recall@k` 时，"把相关的那条从第 4 名提到第 1 名"完全不可见（都在前 k
条里）；只看 `RR` 时，"前 k 条命中了 3 条还是 1 条"不可见。`nDCG@k` 是唯一同时看"命中几条"与
"排得多靠前"的那条。

### 三个形状

```text
LiftProbe(query, relevant)      一句话 + 它的金标准 id。relevant **不能为空**（三条指标的分母
                                都由它给出，空标注会让 recall@k 变成 0/0，而算出来的 0.0 看起来
                                像"一条都没召回到"）；必须是 tuple（frozen 形状里塞一个 list，
                                "这份金标准不曾被改过"就不再成立）
LiftReport(queries, k, before, after, lift, per_query)
                                before/after = 三条均值、lift = after - before、per_query = 逐条
                                明细（两侧名单 + 三条指标 + 谁动了 + 窗口的账）
measure_lift(probes, retrieve, reranker=None, *, k=5, top_n=None, mode=None, weight=None)
```

`lift` 用**差值**而不是比值（比值在 `before == 0` 时是无穷：`0.0 → 1.0` 没有倍数可言）；
空探针列表返回一份"三条指标全 0"的报告而不是报错（与 `aggregate_results` 的空批次同一条
纪律），但 `queries=0` 会如实写在那里——**0 分与"评了 0 条"必须一起被看见**。

### 变量只有一个：有没有重排

`measure_lift` 对每条探针做四件事，顺序固定：`retrieve(probe.query)` 拿一份结果 →
`result.ids()` 算重排前的三条指标 → `rerank_hits(result.hits, …)` 用**同一批命中**重排 →
算重排后的三条指标并记下逐条明细。第三步用**同一批命中**这一点很要紧：如果 after 是"重排之后
重新检索一遍"，差里就混着"两次检索的随机性"，而这份报告的全部价值在于**把变量减到一个**。
`retrieve` 是**可调用对象**而不是"检索器实例"：探针里只有一句话，怎么把它变成一份结果（单路 /
混合 / 带过滤）由调用方决定——因此同一个函数能对两种检索器给出**同一份口径**的提升报告。

一条使用上的真相（演示脚本的第 7 节按它写）：after 只能重排 `retrieve` 交给它的那份名单。
**若那份名单已经被 `top_k` 截到 k 条，"召回提升"在这份报告里不可能出现**（窗口外的条一条都
进不来）。要让 `recall@k` 也能涨，请让 `retrieve` 返回一份**比 k 更深**的名单，并让 `k` 决定
截断位置。

### 本机实测（3 条探针 / k=5）

```text
before = {'recall': 0.5, 'reciprocal_rank': 0.441667, 'ndcg': 0.412399}
after  = {'recall': 1.0, 'reciprocal_rank': 1.0,      'ndcg': 1.0}
lift   = {'recall': 0.5, 'reciprocal_rank': 0.558333, 'ndcg': 0.587601}

'ERR-2043'     before 名单 d-01、d-04、d-02、d-03、d-11、d-12、d-05、…
               after  名单 d-11、d-05、d-12、d-01、d-04、…
               before{recall 0.5, rr 0.2,   ndcg 0.237198} → after{1.0, 1.0, 1.0}
'401 报错码'    before{recall 0.0, rr 0.125, ndcg 0.0}    → after{1.0, 1.0, 1.0}
'retry_budget'  before{recall 1.0, rr 1.0,   ndcg 1.0}    → after 同上（lift 全 0）
```

最后一行要留着：**"提升"不是必然的**，报告要敢写 0（阶段一已经把它排第 1 时，重排能做的
只有"不要弄坏它"）。手算核对（第 1 条探针的 before，纸笔，不从引擎拿数）：

```text
前 k 条里的命中位置 [4]（唯一相关的 d-11 在第 5 位）；m = min(2, 5) = 2
DCG@5  = 1/log2(6) = 0.386853；IDCG@5 = 1/log2(2) + 1/log2(3) = 1.630930
recall@5 = 1/2 = 0.500000      nDCG@5 = 0.386853/1.630930 = 0.237198
RR = 1/(4+1) = 0.200000
```

三个数字与引擎给的逐位一致（`recall_at_k` = 0.500000、`ndcg_at_k` = 0.237198、
`reciprocal_rank` = 0.200000）。这就是"评估可以被手算核对"的落地。

## 8. 参数：构造默认 ← `RetrievalQuery.extra`

优先级是**两级**（`retriever.py` 与 `hybrid.py` 写法一致）：构造参数是这次装配的口径，
`RetrievalQuery.extra` 里的键可以**逐次**覆盖它。

```python
query = RetrievalQuery(text="ERR-2043", top_k=5,
                       extra={"enabled": True, "top_n": 8})
result = hybrid.retrieve(query)      # 这一次打开重排、窗口 8
```

键名是**封闭清单**（`RERANK_OVERRIDE_KEYS = ("enabled", "mode", "model", "top_n", "weight")`）。
**`min_score` 不在这份清单里**：阈值只能由构造参数或 `settings` 决定——"那五个键能逐次改、
第六个不能"是刻意的分工（阈值是"这台机器的标定结果"，不是"这一次请求的旋钮"）。
三条校验纪律：

```text
清单外的键        由第 2 步（融合解析）按**两份清单的并集**判定，抛 FusionError 并列出
                  合法取值。并集判定是 day068 补的：extra 是**一个**字典，只按自己那份判
                  会有一个安静的后果——extra={"mode": "blend"} 在融合层被拒，于是重排的
                  覆盖参数**永远用不上**（"我明明写了 mode"查不出病因）
取值的合法性      由 rerank_hits 的 _resolve_* / _check_* **一份实现**负责，抛 RerankError
                  （top_n < 1、weight 越界、mode 不在 RERANK_MODES、model 是空串…）
enabled 必须是真布尔  "yes" / 1 这类"看起来是真的"写法一律报错——bool("no") 是 True，
                  那会把"这次先关掉重排"静默地变成"打开"
```

一个与 `fusion` **刻意不同**的选择：`mode="replace"` 却显式给了 `weight` 时，本层**记录并说明**
（进 `notes`："mode='replace' 不看权重"）而**不是**报错——那里权重在 RRF 的定义里根本不存在
（给了它说明调用方以为自己在用另一种策略），而这里权重只是另一种模式的参数，报错会让"试一下
blend 的区别"变成一次必须先删参数的实验。两条"必须被说出来"的注记：参数来自 `extra` 时要记下
"本次覆盖了哪几个键"；`enabled=False` 却带了覆盖键时要写出"**没有作用**"（一个真的没生效的
覆盖参数必须被点出来，否则"我写了 top_n 却没变化"会变成一个反复出现的疑问）。**构造期就校验**：
`Retriever.__init__` 与 `HybridRetriever.__init__` 都拿一份**空输入**先过一遍 `rerank_hits`，
于是参数写错在**装配那一刻**就响，而不是等到第一次检索才响——后者会把一个配置错误伪装成
"这次请求碰巧不对"。

### 六个配置项（`config.py` 的 `retrieval_rerank_*` 组）

| 配置项 | 缺省值 | env var | 说明 |
|--------|-------|---------|------|
| `retrieval_rerank_enabled` | `False` | `RETRIEVAL_RERANK_ENABLED` | **新能力默认不生效**：打开它会改变名单顺序 |
| `retrieval_rerank_model` | `"cross-encoder-teaching-v1"` | `RETRIEVAL_RERANK_MODEL` | 只进报告（本层**不按名字加载模型**） |
| `retrieval_rerank_top_n` | `20` | `RETRIEVAL_RERANK_TOP_N` | 窗口大小 N |
| `retrieval_rerank_weight` | `0.5` | `RETRIEVAL_RERANK_WEIGHT` | blend 模式下**重排分**的权重 |
| `retrieval_rerank_mode` | `"replace"` | `RETRIEVAL_RERANK_MODE` | `replace` / `blend` |
| `retrieval_rerank_min_score` | `None` | `RETRIEVAL_RERANK_MIN_SCORE` | 重排阈值（只切被打分过的条） |

（`Settings` 没有设 `env_prefix`，因此环境变量名就是字段名大写——与其余 `retrieval_*` 同规则。）
这一组与既有 `retrieval_*` 是**同一类**：都不改变任何产物的字节（不改 `chunk_id`、不改向量、
不改清单版本号，只改变"怎么问"），因此调它们随时生效、**不需要**重建索引。它也多守了 day067
那条纪律：**新能力默认不生效**——重排会**改变名单顺序**（那正是它存在的意义），因此"换一层
排序"必须是一次显式决定。**关着的时候第 6.5 / 10.5 步什么都不做，day066 与 day067 的结果
逐位不变。** 本机实测（同一个检索器、同一句话）：

```text
关闭（走 settings 缺省）  d-01、d-04、d-02、d-03、d-11
关闭（显式 False）        d-01、d-04、d-02、d-03、d-11     ← 两次逐位相同，分数也逐位相同
打开（rerank_top_n=8）    d-11、d-05、d-12、d-01、d-04
关闭时 rerank == {}：True；dropped_by_rerank = 0；rerank_score / stage1_rank 全是 None
打开时 result.rerank 的键：['candidates', 'dropped_by_min_score', 'empty_reason', 'mode',
                        'model', 'moved', 'params', 'scored', 'window']
```

## 9. 端点清单（现在十个）

| 端点 | 方法 | 入参 | 落到哪个入口 |
|------|------|------|-------------|
| `/retrieval/status` | GET | — | `Retriever.describe()` |
| `/retrieval/search` | POST | `query` + 七个检索参数 | `Retriever.retrieve` → `RetrievalResult` |
| `/retrieval/explain` | POST | 同上 | `Retriever.explain` |
| `/retrieval/answer` | POST | `question` + 同上 | `RagPipeline.answer` |
| `/retrieval/routes` | GET | — | `StoreRouter.report()` |
| `/retrieval/hybrid` | POST | 同 `/retrieval/search` 的八个字段 **+ `strategy` / `alpha` / `k_rrf`** | `hybrid.py::HybridRetriever.retrieve` |
| `/retrieval/lexical/status` | GET | — | `lexical.py::LexicalIndex.describe()` |
| `/retrieval/rerank` | POST | 同 `/retrieval/search` 的八个字段 **+ `enabled` / `mode` / `top_n` / `weight` / `min_score` / `model`** | `rerank.py::rerank_hits`（**强制开启**） |
| `/retrieval/rerank/status` | GET | — | `CrossEncoderReranker.describe()` + settings 回显 |
| `/retrieval/rerank/compare` | POST | `query` + `relevant` + `k` / `top_n` / `mode` / `weight` | `rerank.py::measure_lift` |

这一组的数量与内容**以 `smart_research_agent/api/routes.py` 为准**（响应字段以该文件与
`api/schemas.py` 为准，本文不复述）。day068 的三个重排端点与它们同日在同一个文件里落地，与第 8 节
**同一份**库内口径（重排参数挂在 `RetrievalQuery.extra` 上），而证据原样出门：
`result.rerank` / `dropped_by_rerank` / `hits[].rerank_score` / `hits[].stage1_rank` 直接来自
`RetrievalResult.to_dict()`，端点不重新解释它们。`/retrieval/rerank` 缺省关闭（`False`）时的处置方式与
`/retrieval/hybrid` 同一条先例——**不静默退回不重排**，因为那会让你以为自己在看重排的结果，
而实际上一次重排都没发生（本端点干脆把显式关掉当 400 处理：它与"我要看重排做了什么"是矛盾意图）。

`/retrieval/rerank` 的请求体里那个 `min_score` 是**重排阈值**，不是第一阶段阈值。两个阶段的阈值
共用一个字段名，因此折成检索查询时**必须显式清空**折进去的那一份（`replace(query, min_score=None)`）——
否则同一个数字会落两次刀，而报告上只会显示"结果变少了"。`/retrieval/rerank/compare` 刻意
**不设**重排阈值：阈值改的是名单长度、不是排序质量，混进对照表会让"提升"同时指两件事。

## 10. 常见误区（12 条）

| # | 误区 | 事实 | 证据（可核对） |
|---|------|------|--------------|
| 1 | "重排就是再排一遍，分数越大越靠前" | 重排分与第一阶段分**不同口径**：`score` 仍是第一阶段的分数，重排分在 `rerank_score` 里，因此重排过的名单 `score` **不再单调** | `retriever._rebuild_hits`；`RetrievalHit` 的字段说明；§6 的 `[0.48, 0.0, 0.36, 1.0, 0.96]` |
| 2 | "全库都重排才准" | 窗口是**成本纪律**：只对前 `top_n` 条打分，窗口外的一条都不打分，而这条注记**无条件**进 notes | `rerank._build_notes` 第一条；§3 实测（`scored_pairs` 差量 = 5 = `window`） |
| 3 | "窗口外的条被打了很低的分" | 它是 `score=0.0` + `scored=False` + `features={}`：**不是"最不相关"，而是"没有分"**；尾部按原顺序接在窗口结果之后 | `RerankHit` 的 docstring；§3 实测 |
| 4 | "`min_score` 会切掉不相关的尾部" | 阈值**只作用于被打分过的条**：窗口外连分数都没有，切不到（也救不了）它们 | `rerank_hits` 第 6 步；§5 实测（`top_n=5、min_score=2.0` 时尾部 7 条仍在） |
| 5 | "阈值切光窗口就是空结果" | 窗口被切空但还有窗口外尾部时名单**不空**；只有窗口覆盖全部候选且没有尾部才走到空结果那条路 | §5 实测两行对照 |
| 6 | "`blend` 直接把两份分数相加" | 两份分数**量纲不可比**，必须先各自在**窗口内** min-max 归一化再加权（复用 `fusion._min_max_normalize`） | `rerank.py` 模块 docstring；§4 手算表（差 0.0） |
| 7 | "归一化后的分数是绝对量" | 它随窗口变化：同一篇文档换个 `top_n` 就变（只要求同一次重排内部可比） | §4 实测（d-12 的 blended `0.477444` → `0.657444`） |
| 8 | "`min == max` 时要除零/归一化成 0" | 定义为 **1.0**：窗口内只有一条时它就是那一段的第一名，归 0 会让唯一证据被权重乘没 | `fusion._min_max_normalize`；§4 |
| 9 | "并列时按 `record_id` 决定" | 实际由**第二键 `stage1_rank`**（输入位置）决定；第三键 `record_id` 在这条路径上不会被触发，是防御性兜底 | §4 实测（d-05/d-11 同分，4 < 6） |
| 10 | "`top_n=0` 就是关掉重排" | `top_n` 必须 ≥ 1，`0` 报 `RerankError` 并指出出路（要关掉请把 `rerank_enabled` 设为 `False`——那条路上一个字段都不会被改写） | `rerank._resolve_top_n`；`errors.RerankError` 的模板 |
| 11 | "重排缺省就开着了" | `retrieval_rerank_enabled` 缺省 **False**；关着时 `rerank == {}`、两个新字段是 `None`、结果与 day066/day067 **逐位不变** | `config.py`；§8 实测（两次关闭逐位相同） |
| 12 | "`extra` 里写错键名会静默忽略" | 清单外的键**报错**（由两份清单的并集判定，抛 `FusionError` 并列出合法键）；生效的覆盖参数必须进 `notes` | `hybrid._resolve_fusion` / `hybrid._resolve_rerank` |

再补三条同性质的（写下来省一次排查）：

```text
13. "换 batch_size 会改变分数"    → 不会：分数逐条算，batch 只改变"分成几次算"；空列表连批次
                                  都不产生（"算过 0 批" ≠ "算过 1 批空批"）
14. "explain() 里'只对前 N 条打分'可以随手换成别处那个数字" → 必须读 params["top_n"]（回落
                                  window）：摘要顶层没有 top_n，按顶层键读会渲染成"只对前 0 条
                                  打分"，一句与同一行里 scored 自相矛盾的诊断（本机实测踩过，
                                  已修）
15. "重排分可以当成'相关性'读"    → 教学替身没有真实语义（认不出同义改写、读不懂语序），
                                  重排分只能用于**相对排序**；生产环境请把 BaseReranker
                                  换成真实重排模型
```
