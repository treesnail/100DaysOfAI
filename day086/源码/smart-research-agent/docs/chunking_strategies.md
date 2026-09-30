# 分块策略手册（day062 / M6-D2）

> 本文与代码**同源**：每一张表都能由 `Chunker.describe()` /
> `describe_measurers()` / `DEFAULT_POLICIES` / `separator_histogram()` /
> `chunking_boundaries()` 现场打印出来，每一个数字都由
> `scripts/chunking_demo.py` 实测（样本是一份 614 字符的 Markdown，
> 度量器为 `chars`，embedding 为 `MockEmbedding`）。

## 1. 一条链：从文档到能被检索到的片段

```text
Document（全文 + 块序列 + 内容指纹）
    │  四种策略，同一个形状
    ├─ fixed       每 max_tokens 个单位一刀，不看内容
    ├─ recursive   按分隔符表从强到弱下切，再合并到预算
    ├─ structural  按 day061 抽出的标题层级切小节，保护代码块与表格
    └─ semantic    相邻自然段相似度最低处切（用 embedding）
    ▼
ChunkSet（块序列 + 度量口径 + 覆盖率 / 重复率 / 超预算）
    │
    ▼
knowledge_records()  →  day009 的 VectorStore / day064 的向量库
```

```text
errors.py     分块会怎么失败（参数矛盾 / 未知策略）
tokens.py     预算单位是谁、为什么默认 chars、二分原语
types.py      Chunk / ChunkSet / chunk_id 与 fingerprint 的分工
base.py       收尾六步 + 覆盖率不变量 + 注册表 + 共用原语
fixed.py      窗口算术（参数一确定，成本就算得出来）
recursive.py  分隔符优先级表（中文标点必须在表里）
structural.py 标题栈 + breadcrumb + 原子块保护 + 退化路径
semantic.py   分位数阈值 + 唯一会调用外部能力的策略
pipeline.py   批量与对照（规模 / 重复 / 超预算三列）
evaluate.py   用你自己的探针集打分（hit@k / MRR / 命中块均长）
```

## 2. 四种策略的决策表

| 策略 | 在哪里切 | 依赖 | 重叠 | 确定性来源 |
|------|---------|------|------|-----------|
| `fixed` | 每 `max_tokens` 个单位一刀，**与内容无关** | 无 | 支持（窗口内自带） | 纯字符串运算 |
| `recursive` | 分隔符表从强到弱：`\n\n` → `\n` → `。` → `，` → 空格 → 字符 | 无 | 支持（收尾统一回带） | 纯字符串运算 |
| `structural` | 标题层级处开新小节，小节内按块打包 | day061 的 `blocks` | **不支持**（跨标题会把上一节灌进下一节） | 纯字符串运算 |
| `semantic` | 相邻自然段相似度低于分位数处 | embedding 提供方 | 支持 | **取决于提供方** |

实测（614 字符 / 默认参数）：

| 策略 | 块数 | 平均长度 | 覆盖率 | 重复率 | 超预算 |
|------|------|---------|--------|--------|--------|
| `fixed` | 3 | 235.7 | 115.15% | 15.15% | 0 |
| `recursive` | 3 | 234.3 | 114.50% | 14.50% | 0 |
| `structural` | 5 | 121.0 | 98.53% | 0.00% | 0 |
| `semantic` | 5 | 127.8 | 104.07% | 4.07% | 0 |

三个数字一起看才有意义：**块数决定索引规模与 embedding 花费，重复率决定
重叠白花了多少，超预算决定有多少块注定被模型截断。**

## 3. 默认参数（按策略，不是全局）

```text
fixed       max_tokens=320 overlap_tokens=48  min_tokens=0
recursive   max_tokens=320 overlap_tokens=48  min_tokens=0
structural  max_tokens=320 overlap_tokens=0   min_tokens=0
semantic    max_tokens=400 overlap_tokens=40  min_tokens=60
```

`DEFAULT_POLICIES` 是一张**按策略给出的表**，因为同一个默认值在不同策略下
含义不同：`overlap=48` 对固定长度与递归是必要的（切在句子中间时两侧都要读到
完整句子），对结构策略**是错的**（标题边界就是语义边界）。

参数的三条全局约束（`ChunkPolicy.__post_init__` 强制）：

```text
1 ≤ max_tokens
0 ≤ overlap_tokens < max_tokens       否则步长为 0，窗口原地踏步
0 ≤ min_tokens ≤ max_tokens           否则合并后的块必然超预算
2 × overlap_tokens < max_tokens       仅 recursive / semantic：否则块之间互相复制
```

第三条最容易被忽略，因为它的两个数**各自都合法**：`max_tokens=60` 与
`overlap_tokens=48` 单看都没问题，合起来却让核心窗口只剩 12 个单位。

## 4. 预算单位：`chars` 还是 `tiktoken`

| 度量器 | 单位 | 确定性 | 与计费单位一致 | 默认 |
|--------|------|--------|---------------|------|
| `chars` | 1 个字符 | **是**（只依赖 `len`） | 否 | **是** |
| `tiktoken` | 1 个 token | 否（依赖本机词表缓存） | 是 | 否 |

实测：`"分块预算决定索引规模。"`（11 个字符）在 `cl100k_base` 下是 **13 个
token**。

默认选 `chars` 的理由是**可复现性**：`chunk_id` 是知识库的去重键，而
`tiktoken` 在不同机器上可能给不出同一套计数（缓存缺失时 day034 的
`TokenCounter` 会退化成字符级估算），于是同一份文档切出两套 `chunk_id`，
"这条知识为什么重复了两份"就变成一个查不出来的问题。

要按真实 token 预算时显式切到 `tiktoken`，**并且它会跟着产物走**
（`ChunkSet.token_measurer`）。本机词表不可用时 `resolve_measurer` 直接拒绝，
**不做静默降级**——静默降级会让"我要 token 预算"变成"我拿到字符预算"。

## 5. `chunk_id` 与 `fingerprint` 是两个问题

```text
chunk_id     sha256(doc_id | strategy | index | text)[:16]    ←「这一块是谁」
fingerprint  sha256(text)[:16]                                ←「这是哪段内容」
```

- 只按文本算 id 会坏在"同一段话在同一份文档里出现两次"：两个块 id 相同，
  向量库按 id 覆盖，"命中了 3 次"在库里只剩 1 条；
- 而跨文档去重恰恰要的相反：同一段话在 A、B 文档里各存一份是重复存储。

两个问题、两个字段。混成一个就会出现"要么去重失效、要么块互相覆盖"，
而两种失效都不会报错。`ChunkSet.knowledge_records()` 里 `doc_id` 位置放的是
`chunk_id`，`fingerprint` 放在 `metadata`。

## 6. 不变量：块是原文的连续子串，非空白字符全覆盖

```python
document.text[chunk.start_char : chunk.end_char] == chunk.text
```

它由两处保证：

1. **构造上不可能违反**：`Span` 只带区间、不带文本，块文本只能由
   `document.text[start:end]` 现取；
2. **违反即报错**：`Chunker._assert_spans` 逐个字符核对"原文里每一个非空白
   字符都至少属于一个块"，不过关就抛 `ChunkingError`——
   **不是给一个 `coverage=0.94` 的报告**。

覆盖率可以略低于 1，差额来自"块首尾的空白被修剪"，且有上界：

```text
doc_chars - total_chars ≤ 2 × (块数 - 1) + 1
```

实测：`fixed`（11 块）差 2，`structural`（5 块）差 9（上界 9）。

## 7. 结构分块：breadcrumb 与原子块

**breadcrumb**：每块带 `heading_path`，它进 `retrieval_text`（检索视图）而**不进
`text`**（原文视图）。一句"阈值设为 0.85"单独看没有话题信息，而它上面的标题
才是查询能命中它的原因。

**原子块**（`ATOMIC_KINDS = ("code", "table")`）：超预算时整块保留并标记
`oversized`。两种选择的代价是不对称的：

```text
切开会得到两块「都不对」的文本     ← 更糟：错误很隐蔽
不切会得到一个超出预算的块         ← 更好：超预算是可见、可统计的
```

实测：把预算压到 40，210 字符的代码块成为唯一的超预算块
（`reason=atom:code`），`oversized_count=1`。

**退化路径**：小节的块拼起来若在全文里找不到（规范化改动了某处空白），
该小节按段落切并把 `start_block` / `end_block` 记 `-1`——**不猜一个值**。
退化次数记在 `metadata.unlocated_sections` 里。

## 8. 语义分块：分位数而不是绝对值

```text
1. 按空行切出自然段（最后一段的尾随空白会被剪掉）
2. 每个自然段编码成一个向量
3. 算相邻两段的余弦相似度
4. 相似度低于「相似度分布的分位数」处切一刀
5. 标题处额外断开，再按预算合并
```

阈值用**分位数**：`0.7` 这个绝对值在不同 embedding 提供方上含义不同
（依赖训练目标、维度、是否归一化、语料语言），而"最低的 25%"在任何提供方上
都表示同一件事。

实测（本项目文档，`similarity_percentile` 变化时）：

```text
percentile=0.10 → 5 块 | 阈值 -0.1568 | 断层 4
percentile=0.25 → 5 块 | 阈值 -0.1018 | 断层 6
percentile=0.50 → 5 块 | 阈值  0.0062 | 断层 6
percentile=0.75 → 6 块 | 阈值  0.0492 | 断层 8
```

它也是四种策略里唯一会调用外部能力的一种，因此它的 `describe()` 与
`ChunkSet.metadata` 里都记着 embedding 的类名与维度——**"这份块是怎么切出来的"
必须能被回答。**

## 9. 评估：用你自己的探针集打分

```text
1. 探针集            (question, expect) 列表，expect 是原文里逐字存在的片段
2. 建索引            按某策略切块 → 每块用 retrieval_text 编码 → 入内存向量库
3. 检索              question 编码后取 top_k 块
4. 判定              某个块的**原文**里含 expect 吗？含 → 命中，记下排名
5. 汇总              hit@k、MRR、命中块均长、索引规模
```

判定用的是"块文本里是否含期望片段"，于是它同时测了两件事：检索能不能召回，
以及**期望片段有没有被切碎**——后者用"平均块长"永远看不出来。

实测（`MockEmbedding`，**没有真实语义**）：

| 策略 | hit@3 | MRR | 命中块均长 | hit@8 |
|------|-------|-----|-----------|-------|
| `fixed` | 100.0% | 0.444 | 236 | 100.0% |
| `recursive` | 100.0% | 0.444 | 230 | 100.0% |
| `structural` | 66.7% | 0.500 | 128 | 100.0% |
| `semantic` | 66.7% | 0.444 | 128 | 100.0% |

把 `top_k` 放大到覆盖全部块，四个策略都回到 100%——这说明 top-3 的差距
来自**检索深度**而不是分块质量；要比较语义召回能力，必须注入真实的
embedding 提供方。

两条纪律（比指标公式重要）：

1. 探针必须来自**你自己的查询分布**，抄别人的最优策略只对别人有效；
2. 期望片段在原文里必须**唯一**，出现多次的探针会让分数虚高且毫不显眼
   ——用 `ambiguous_probes()` 自查，端点会把它们单独返回。

## 10. 接口与配置

四个端点（全部只读或纯计算，不写盘、不联网、不遍历目录）：

| 端点 | 方法 | 回答什么 |
|------|------|---------|
| `/chunking/strategies` | GET | 四种策略、度量器、默认参数、分隔符表、覆盖不变量、"不能做什么" |
| `/chunking/split` | POST | 事前估算（块数/放大量）+ 实测统计 + 前 N 块明细（带 `start_char:end_char`） |
| `/chunking/evaluate` | POST | 多策略检索对照（hit@k / MRR / 命中块均长）+ 歧义探针 + markdown |
| `/chunking/records` | POST | 可直接入库的记录（含 `embedding_input` 说明） |

配置项（`settings.chunking_*`）：

```python
chunking_strategy: str = "recursive"          # 默认策略
chunking_max_tokens: int = 320                # 单块预算
chunking_overlap_tokens: int = 48             # 重叠（占预算的 15%）
chunking_min_tokens: int = 40                 # 过短块的合并下限
chunking_measurer: str = "chars"              # 预算单位
chunking_similarity_percentile: float = 0.25  # 语义策略的阈值分位数
chunking_eval_top_k: int = 3                  # 评估的检索深度
```

参数优先级是**三级**：`请求体 > settings（项目级基线）> 策略默认值`。
其中 `overlap_tokens` 与 `min_tokens` **按比例**跟着预算缩放
（`48/320 = 15%`，因此 `max_tokens=64` 配 `overlap=10`）——
把 48 写死会让"预算调成 64"变成"75% 的重叠"，而参数看起来只是变小了。

## 11. 已知限制（每一条都带依据）

1. 覆盖率保证的是"原文每个非空白字符至少属于一个块"，不是"每块都是一句
   完整的话"：固定长度策略必然会切在句子中间；
2. 结构策略的 `start_block` / `end_block` 只在"小节能被精确定位"时有值，
   定位失败的小节退化为按段落切、两个字段记 `-1`；
3. 语义策略只认识自然段，**会把含空行的代码块从中间切开**（与结构策略相反）；
4. `tiktoken` 度量器依赖本机词表缓存，同一份文档在不同机器上可能切出不同的
   `chunk_id`；
5. 重叠只在 `fixed` / `recursive` / `semantic` 下生效，结构策略显式拒绝非零重叠。

明确排除的用途：

1. 指望某一种策略在所有语料上最优——选型必须用自己的查询集跑
   `/chunking/evaluate` 之后再决定；
2. 把分块当作"语义理解"：只有 `semantic` 看内容相似度，而它用的 embedding
   若没有真实语义（如离线的 `char-ngram`），切出来的边界只是"字面差异最大的
   地方"；
3. 用分块结果做原文重建：带重叠的块集合重建会得到重复内容，重建请用
   `Document.text`。

## 12. 复现

```bash
cd day062/源码/smart-research-agent
PYTHONPATH=. python scripts/chunking_demo.py          # 八节演示，全程离线
python -m pytest tests/test_chunking_*.py -q          # 192 个用例
```
