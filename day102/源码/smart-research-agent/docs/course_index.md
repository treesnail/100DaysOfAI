# `course_index` 手册：把 100 天的材料编成一份可检索、可复算的索引

> 对应课程日：**day101（结业之后的第一次扩张）**
> 主线项目：智研 AI 助手（SmartResearch Agent）
> 前置：day099（`capstone` 九段链）、day100（`graduation` 四份交付物）
> 全文离线、确定性：不联网、不读任何环境变量密钥

---

## 一、这一课解决什么问题

走到第 100 天之后，仓库里躺着大量**已经写好但不再被读**的文字：
`docs/` 下 46 份手册、53 个子包的 docstring。它们回答过很多问题，
但"哪一天讲过 XXX"这件事只能靠 `Ctrl+F` 一把一把地翻。

`course_index` 把这件事变成一个**可以被检索、而且可以被复算**的问题：

```text
收语料  →  编索引  →  检索  →  用七条判据证明"这份索引是可被信任的"
```

**今天最值钱的一句话**：

> 把材料编成索引，第一步不是"选一个搜索算法"，而是先说清"哪些材料必须进得来"——
> 一份没进语料的文档，与"这门课没有那份文档"在检索结果里读起来完全一样。

---

## 二、七个模块

```text
errors.py   六个失败族（回来的 TokenError + 继续缺席的 GradientError）
types.py    2 种语料 / 1 套分词口径 / 7 条性质（判据分三类）/ 10 条笔记 / 5 条边界
corpus.py   从 docs/*.md 与子包 docstring 收一份语料（名字带种类，绝不撞车）
index.py    分词（ASCII 词 + 中文二元组）+ 倒排索引 + 复算摘要
query.py    一次确定性检索：score = 命中词数 / 查询词数，并列按名字排序
verify.py   七条性质与三类判据（相等 / 上界 / 下界）
study.py    四张表（语料 / 索引 / 检索 / 性质）
__init__.py 包入口
```

---

## 三、语料：两种材料，名字必须带种类

```text
doc         仓库 docs/ 下的 *.md 全文（课程手册）      当前 46 份
subpackage  每个子包的 __init__.py docstring（包自述）  当前 53 条
```

本仓库当前的读数：

```text
语料 99 份 | 文档 46 / 子包 53 | 合计 535049 字
```

**一条纪律：文档名必须带种类前缀。**

```text
docs/backprop.md              →  doc:backprop
smart_research_agent/backprop →  subpackage:backprop
```

裸名字会撞车——本仓库里 `backprop` / `capstone` / `conv_net` / `graduation`
都同时是"一篇手册"与"一个子包"。撞车之后 `Corpus` 的"名字唯一"检查会抛
`CorpusError`：**宁可当场变红，也不要让两件不同的东西共用一个名字。**

另外 `subpackage_docstrings()` 对"某个子包没有 docstring"是**抛错**而不是跳过：
一个没有自述的包，它的存在永远不会被检索到——那比"报错"更难被发现。

---

## 四、分词：确定性、零依赖，代价写在脸上

```text
ASCII 词    [A-Za-z0-9_]+ 小写化，长度 >= 2
中文串      长度 >= 2 时切成全部相邻二元组；长度 1 的整段作为一个 token
```

为什么不用分词库？因为**换一个版本就会换一份索引**，而"索引可复算"是这份产物的全部价值。

为什么不用空白切词？中文没有空格：按空白切，"整段中文"会变成一个词，索引等于没建。

二元组的代价是"会多召回一些"（比词粗）。本课要的不是最好的检索质量，
而是**一份能被别人重算出同一份结果的索引**——这与 day087 的"位置必须由缓存长度决定"、
day100 的"剧本必须能被重放"是同一条纪律。

一个实现细节值得单独说：分词用**一个**交替正则扫一遍正文，而不是"先扫 ASCII、再扫中文"：

```text
两个独立的正则各扫一遍   ⇒  "先 ASCII 后中文"这个与正文无关的顺序
                            会被写进 query_tokens 与 matched——而它们是会被印出来的
一个交替正则            ⇒  token 顺序跟着正文走
```

---

## 五、索引：倒排表 + 一个可以被别人重算的摘要

```text
token → ((文档, 次数), ...)     按文档名排序
文档 → 词数                      按语料顺序
```

本仓库当前的读数：

```text
索引：99 份文档 | 词典 31278 个词 | 词次 157313 | 摘要 1f60d8bf1a2fcb68
```

`InvertedIndex.__post_init__` 有**一条特别值得说的护栏**：

```text
token → ({doc_a: 2, doc_b: 1})  ⇒  postings 里的文档名必须**排序**
```

正文明明一样，但如果倒排表里的文档名不排序，"索引可复算"这条性质就会永远失败——
因为没有排序的字典序在不同运行里可能不同。**并列的不确定性必须被显式消掉。**

`index_digest` 给出一个 16 位摘要：同一份语料两次构建的摘要相同 ⇔ 逐位相同。

---

## 六、检索：一条公式 + 一条并列规则

```text
score = 该文档命中的查询词数 / 查询词总数        ⇒  数学上落在 [0, 1]
```

```text
① 分数高的在前
② 分数并列时，**文档名**字典序在前     ⇒ 同一个问题两次得到同一份名单
```

本课的金标准是"查询包名、命中该包"：

```text
查询 'course_index' | 词 1 个 | 命中 2 条 | 首条 doc:course_index 1.0000
金标准材料 subpackage:course_index 的名次 = 2
```

注意这里**命中了两条**：同一个词既出现在本课的手册（`doc:course_index`）里，
也出现在本包的 docstring（`subpackage:course_index`）里。两条分数都是 `1.0000`，
因此先后由"并列时按文档名字典序"这条规则决定（`doc:` < `subpackage:`）。
这正是 ② 号并列规则存在的理由：没有它，两份名单都"对"。

因此第 ⑦ 条性质只要求"金标准材料被命中 >= 1 次"，不要求它排在第一——
**名次是排序规则的结果，不是这条性质要管的事**。

为什么金标准要**由构造保证成立**（`course_index` 在自己的 docstring 里必然出现）？
因为一条会因为语料变化而随机变红的性质，最后一定会被人关掉。

`Hit` 里同时有 `score` 与 `matched`（命中的那几个词）：
少了 `matched`，一个 0.5 分就只能被相信、不能被核对。

**"没查"与"查了没命中"是两件事**：

```text
空查询 / 无词查询   ⇒  QueryError（调用点的问题）
查了但名单为空       ⇒  正常返回（语料的问题）
```

因此 `require_hits()` 是"拒绝空名单"的那条可选的路，而不是 `search()` 的默认行为。

---

## 七、七条性质与三类判据

```text
相等（==）   ① corpus_covers_all_docs            手册缺失数 == 0
             ② corpus_covers_all_subpackages     子包缺失数 == 0
             ③ index_is_reproducible             两次构建的差异项数 == 0
             ④ search_is_deterministic           两次检索的差异项数 == 0
上界（<=）   ⑤ scores_within_bounds               最高分数 <= 1.0
下界（>=）   ⑥ every_document_is_indexed          最小词数 >= 1
             ⑦ gold_query_finds_gold_document     金标准材料被命中数 >= 1
```

实测（`verify.check_all()`）：

```text
1. corpus_covers_all_docs          | [equality   ] | 读数 0 ✓ | 手册 46 份 / 期望 46 份
2. corpus_covers_all_subpackages   | [equality   ] | 读数 0 ✓ | 子包 53 条 / 期望 53 条
3. index_is_reproducible           | [equality   ] | 读数 0 ✓ | 摘要 1f60d8bf1a2fcb68 / 1f60d8bf1a2fcb68
4. search_is_deterministic         | [equality   ] | 读数 0 ✓ | 查询 'course_index' 命中 2 条
5. scores_within_bounds            | [upper_bound] | 读数 1 ✓ | 最高分数 1.0000 ≤ 上界 1.0
6. every_document_is_indexed       | [lower_bound] | 读数 2 ✓ | 最小词数 2 ≥ 下界 1
7. gold_query_finds_gold_document  | [lower_bound] | 读数 1 ✓ | 命中 1 次 ≥ 下界 1 | 名次 2
合计：7/7 条通过
```

**为什么 ⑥ ⑦ 必须是下界**："每份文档至少被索引一个词"与"金标准材料必须被命中"
都是**越多越好**的量，只能从下面兜住。写成相等（`== 1`）会在"一份文档切出更多词"
或"金标准同时出现在前排与后排"时误报失败。

**为什么 ⑤ 是上界**："分数不超过 1"是一条越界就不可用的量，因此兜的是上限。

---

## 八、失败族（按「该谁去修」分）

| 族 | 该谁去修 |
|----|----------|
| `CorpusError` | 改语料收集：文档名重复或某份材料读不到时，先把那一份补进语料 |
| `IndexBuildError` | 改索引实现：两次构建不一致时，去掉那个没有固定的量（迭代序 / 集合序） |
| `QueryError` | 改查询或改分词：空查询与无词查询都要在调用点被看见 |
| `ScoreError` | 改打分：命中分数越界时，先回到 `score = 命中词数 / 查询词数` 的定义 |
| `NumericError` | 改数据或改实现：非有限读数与非法上下界都属于"数值不可用" |
| `ParameterError` | 改调用：文档种类 / 性质名 / `top_k` / 序号都是调用点的一次决定 |

两条族常量：

```text
RETURNED_FAMILY = "TokenError"
    day086 的 hf_integration.errors.TokenError 第一次给"分词器配置对不上"命名；
    本课需要**自己决定怎么切词**，于是这个名字回来了——
    但基类从 ParameterError 换成 CourseIndexError(ValueError)。

ABSENT_FAMILY = "GradientError"
    本课是编索引与检索，一个新式子都没有写。
```

---

## 九、四条纪律

1. **不重写任何子系统**：本包只新增代码；`docs/` 下的手册是被**读取**的，
   既有模块、既有测试、`pyproject` 与 docs 下的既有文件**一行未改**。
2. **读数必须现场算出**：`study` 里不存数字；每张表的每行都来自函数调用。
3. **确定性**：分词零依赖、倒排表显式排序、并列命中按名字兜顺序，
   因此"两次构建逐位相同"与"两次检索逐位相同"都能被一条 `==` 判定。
4. **不联网、不读密钥**：本包只读文件系统与 `importlib`。

---

## 十、四张表

```text
① 语料表     全部材料：名字（带种类）/ 种类 / 字符数
② 索引表     词典里最常见的若干 token：出现它的文档数 / 前几个文档与次数
③ 检索表     一次检索的名次 / 文档 / 分数 / 命中的词
④ 性质表     7 条性质：判据类别 / 读数 / 结论 / 一行证据
```

---

## 十一、怎么跑

```bash
cd day101/源码/smart-research-agent
python scripts/course_index_demo.py                                # 十节离线演示
python -m pytest tests/test_course_index.py -q -o addopts=""       # 本包 56 个用例
python -m pytest -q                                                # 全量回归
```

演示脚本的产出：

```text
outputs/course_index_demo.txt          本脚本的完整输出（在 .gitignore 里）
outputs/course_index/search_hits.txt   金标准查询的名单（新的产物，不覆盖任何既有文件）
```

---

## 十二、边界（本课明确不承诺的事）

- 本包**不新增任何第三方依赖**：分词是自写的 bigram + ASCII 词切分。
- 本包**不改动任何既有模块、既有测试、pyproject 与 docs 下的既有文件**——
  它只新增文件；`docs/` 下的手册是被读取的，不是被改写的。
- 本包**不承诺检索质量**：它承诺的是"同一份语料编出同一份索引、
  同一查询给出同一份命中"，而不是"排在最前面的总是最好的"。
- 本包的全部读数**离线、确定性**：只读文件系统与 `importlib`。
- 本包**不含语义检索**：bigram 索引认得的是"字面共现"，不是"同义"——
  把"怎么训练"与"训练方法"联系起来不是它要解决的问题。

---

## 十三、与既有包的接缝

```text
docs       docs/*.md（被读取，不被改写）
packages   importlib（53 个子包的 __init__ docstring）
milestone  day100 的 graduation（它数出"这个仓库里有多少个子包"）
```

本课是结业之后的一次**横向扩张**：主线项目（智研 AI 助手）一行未动，
被索引的是**这门课自己的产出**——也正因为如此，它比任何一次纵向升级都更依赖
"确定性"：一份检索不到的文档，等于一份没写过的文档。
