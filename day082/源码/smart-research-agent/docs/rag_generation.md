# RAG 生成器手册（day069 / M6-D8）

> 本文与代码**同源**：每一条结论都标了落地位置（`retrieval/xxx.py::函数名`），都能在
> `smart_research_agent/retrieval/generation.py`、`pipeline.py`、`context.py`、`errors.py`
> 与 `config.py` 的 `retrieval_prompt_version` / `retrieval_answer_temperature` /
> `retrieval_answer_max_tokens` / `retrieval_require_citation` 四项里逐字找到。
>
> **本文的数字只有三个来源**，其余一律写"以你实际运行时输出为准"：
>
> ```text
> 代码里写死的常量    PROMPT_VERSIONS = ("v1", "v2")、CUSTOM_PROMPT_VERSION = "custom"、
>                     REQUIRED_PROMPT_FIELDS = ("context", "question")、DECLINE_MARKERS（3 条）、
>                     FALLBACK_REASONS（5 个）+ FALLBACK_REASON_NONE（空串）、
>                     GENERATION_OVERRIDE_KEYS（3 个）、_MARKER_PATTERN = r"\[\s*(\d+)\s*\]"、
>                     retrieval_prompt_version = "v2"、retrieval_answer_temperature = 0.0、
>                     retrieval_answer_max_tokens = 1024、retrieval_require_citation = False
> 由它们算出的算术    coverage = len(valid) / len(known)（round(…, 4)）；
>                     grounded = bool(valid) and not invalid；九步的顺序 = 回退判定优先级；
>                     given = len(valid) + len(unused)（= 提示词里真正给出去的片段数）
> 本机实测           标注"本机实测"的行来自同日的离线演示 `scripts/generation_demo.py`
>                     （FlatVectorStore + 一张写死的向量表 + 6 条记录 + MockLLM，零网络、
>                     零 API Key）；正式数字请以 `outputs/generation_demo.txt` 与
>                     `python -m pytest tests/test_retrieval_generation*.py` 的实际输出为准
> ```
>
> 全层的落地状态（`retrieval/__init__.py` 的原文）："导出范围（**十二个模块全部落地**）：…
> `rerank` / `generation`（最后一个是 day069 新增的）"。端点层（`api/routes.py` 的
> `/retrieval/*`）与本层同行交付，但本文只按**索引式写法**引用它，**不复述任何响应字段**。

## 1. 一句话定位：生成器回答的是"这段话能不能被片段核对"

```text
day066 检索器    回答"给一句话，取回哪几条"
day067 混合检索  回答"两路怎么合、谁贡献了哪条"
day068 重排      回答"这一批候选里，哪一条真的最相关"
day069 生成器    回答"**模型给的那段话，能不能被片段核对**"
```

前三天把"取回什么"做到了可解释，day069 面对的是链条末端一直没人管的东西：模型交出的那段话。
它容易骗过人的地方在于三种"看起来没问题"——**调用成功、答案通顺、引用格式像模像样**（超时 /
鉴权 / 额度都不是"答对了"；通顺的句子可以完全来自预训练记忆；一段与库无关的文字后面也能规规
矩矩写上 `[1]`）。因此本模块不把"答案"当成品交出去，而是把它与**那次提示词里的编号表**放在
一起对账，折成三个数字（有效引用 / 幻觉引用 / 未被引用的片段）与一个判定
（`grounded = valid 非空 且 invalid 为空`）。核对的口径必须写在最前面，否则它会被误读成一次
事实核查：

```text
它做的事    答案里的 [n] 与那次提示词里的编号一一对账（**编号层面**）
它不做的事  判断这句话是否忠实于片段（**语义层面**，见 types 的边界声明）
```

### 与 day066 那条护栏的关系（检索为空绝不调 LLM）

day066 在 `pipeline` 里钉死了"检索为空时一次 LLM 都不调"。day069 把它往前加了一道，于是
护栏有了**两个把手**，各自守着自己那一段（`pipeline.py` 的模块 docstring）：

```text
本模块（pipeline）      检索为空（empty_reason 四种）→ 不打包、不渲染、不生成
生成器（generation）    打包为空（PackedContext.is_empty）→ 不渲染、不调用

检索为空 + 调用 LLM    模型手里没有片段，却仍会写出一段通顺的答案 → 读的人看不出区别
检索为空 + 不调用      答案明确写着"知识库中没有检索到相关内容"，并给出三条出路
```

第一行的产物**看起来最像成功**：语法正确、语气确定、引用格式也像模像样——它是 RAG 系统最贵的
一种失败（没有异常、没有告警）。因此生成器里的判空排在**渲染提示词之前**（第 6 节的第 2 步）。

### 模块地图（每个模块回答一个问题）

```text
generation.py          生成器与提示词：多版本模板 + 六条约束 + 引用溯源 + 接地校验 + 回退族
generation.py::_grounding_report   把答案与编号表对账，一次算出三组编号与 checks
generation.py::_declined           固定句式的包含判定（归一化去空白与标点，不做正则）
pipeline.py            第 4 步换成 RAGGenerator；空检索护栏**仍留在**它这里（更上游）
pipeline.py::RagAnswer 尾部新增 check / fallback_reason 两个字段（只增不改）
errors.py              GenerationError（继承 QueryError："这次调用/装配写错了参数"）
config.py              四项：retrieval_prompt_version / retrieval_answer_temperature /
                       retrieval_answer_max_tokens / retrieval_require_citation
```

两处护栏共用**同一套词汇表**：`pipeline` 的空结果报 `generation.FALLBACK_REASON_NO_CONTEXT`，
`RagAnswer.__post_init__` 用 `FALLBACK_REASON_DESCRIPTIONS` 的键集合校验 `fallback_reason`——
一个词汇表两处使用，"这一批里有几次没让模型发挥"才能是一个数字而不是两个键。

### 已知边界与明确不做的事

```text
只做引用编号层面的核对    答案里的 [n] 与那次提示词的编号对账——它不判断"这句话对不对"
不做语义忠实性判断        grounded 说明"有可核对的依据"，不说明"忠实于片段"
                        （所以叫 grounded 而不是 correct）
不做答案质量评估          答案好不好、可核对率有没有变好，属于 **day071 的 RAG 评估**
不做模型加载              这一层不认识任何后端：llm 是**注入**的 BaseLLM，model 只进报告
不做缓存与检索            不缓存答案（缓存在 day046）；只消费 PackedContext 这一种输入
```

`types.RETRIEVAL_LIMITATIONS` 在 day069 第三次改写："不做引用溯源校验"从限制里**删除**，
换成"只做编号层面的核对"这条更精确的边界。

## 2. 提示词：v1 → v2 的四段式

与 `agent.prompts` 的约定一致（day026）：**历史版本永不删、变更原因写进 changelog、
当前版本用 `CURRENT_PROMPT` / `CURRENT_PROMPT_VERSION` 指过去**。

```text
PROMPT_VERSION_V1 = "v1"      PROMPT_VERSION_V2 = "v2"
PROMPT_VERSIONS = ("v1", "v2")      ← **封闭清单**（顺序 = 报告里的顺序）
CURRENT_PROMPT = RAG_PROMPT_V2      CURRENT_PROMPT_VERSION = "v2"
DEFAULT_PROMPT_VERSION = "v2"       ← settings.retrieval_prompt_version 的缺省值就是它
CUSTOM_PROMPT_VERSION = "custom"    ← 注入自定义模板时的版本号（合法，但不是一版受管模板）
```

### V1 的四条约束（每条挡一种真实失败）

v1 是 day066 的首版，四条约束的注释就写在常量上方（`RAG_PROMPT_V1`）：

```text
1) "只依据片段"        挡"模型拿预训练知识把答案补圆"（引用看着正常，内容与库无关）
2) "片段不足就说没有"   挡"编一个通顺的答案"（最贵的一种失败：它看起来最像成功）
3) "引用写 [n]"        挡"答案对但溯源不上"（day069 的引用溯源要拿它当锚点）
4) "不许编造"          挡"把两个相似概念合并成一个"（错得最像常识的那一类）
```

v1 把"资料 / 问题 / 要求"混在同一段里，**顺序即语义**（模型只能靠位置区分"喂进去的"与
"要它做的"）。本机实测：v1 模板 238 字，**0 个小节标题**，4 个编号条目。

### V2 的四段式与六条约束

v2 按"资料片段 / 问题 / 约束 / 输出格式"重写，四件事各自有小节标题；约束从 4 条扩到 6 条，
每条都注明它挡哪一种真实失败（照 v1 的注释写法：**一条挡不住的约束不该占提示词的预算**）：

```text
1) 只使用片段里的信息，不引入片段外的知识、不靠推测补全   → 挡"拿预训练记忆把答案补圆"
2) 资料不足时以固定句式开头（三选一）                    → 挡"编一个通顺的答案"，
                                                        同时让"模型自己说没有"**可检测**
3) 每条资料都标出编号 [1]、[2]，编号必须来自片段          → 挡"溯源不上"，也挡幻觉引用
4) 每条结论后至少一个引用                              → 挡"整段没有引用的总结"（coverage=0）
5) 片段里没有的事实一律不要编造（数字 / 日期 / 代码 / 专名）→ 挡"把两个相似概念合并成一个"
6) 用中文、先结论再依据、不要整段复述原文                → 挡"复述原文"（它把预算挤给无关句子）
```

v2 还新增了 `## 输出格式` 一节：v1 只管"内容该怎样"，没管"这份答案长什么样"，于是
"有没有引用、引用写在哪里"全靠模型自行发挥。本机实测（同一份上下文、6 条片段、417 字）：

```text
        模板长度  渲染后长度  小节标题  编号条目
v1      238 字    649 字      0 个      4 条（无小节标题）
v2      500 字    911 字      4 个      9 条（约束 6 + 输出格式 3）
差      +262 字   +262 字     +4        +5
```

v2 多出来的字符数正好等于渲染后的差值（262 字）——四段式与两条新约束的代价就在这里。

### 固定拒答句式：为什么它是"让拒答可检测"的前提

v1 只说"就直接说`资料中没有相关内容`"，**句式不固定**，于是"模型自己说没有"这件事在代码里
**无法被检测**（任何字符串比较都会静默失效）。v2 把它升级成一条硬要求：

```text
DECLINE_MARKERS（**封闭清单**，3 条）
    "资料中没有相关内容" / "资料未提及" / "无法依据资料回答"
```

检测（`_declined`）与要求（`RAG_PROMPT_V2` 的第 2 条约束）**必须是同一份**：改这里就必须改
那里——两边一旦分家，"模型自己说没有"会变成一件有时能检测、有时检测不到的事，而那种不一致在
报告里看不出来。检测只做**归一化后的包含判定**：先去掉空白与标点（`_DECLINE_NOISE`，模型会
写成 "资料中没有相关内容，" 或 "资料中 没有相关内容"），再看那串字在不在。

```text
可枚举    v2 的约束里写死了这三个句式，检测清单与要求清单是同一份
可测试    每个句式都能被一条用例钉住
可解释    报告里能写出"命中了哪一句"（_declined_marker 返回命中的那一句，而不是布尔）
```

**为什么不用正则、不用模糊匹配**：两者都会引入一个阈值。正则会把"没有相关内容"这半句也算
命中（误判一次拒答等于把一次正常回答丢掉）；编辑距离需要一个没有依据的相似度门限，而同一个
答案在两台机器上还可能得到不同结论。本机实测的反例：`"这段资料里出现了'向量'两个字，但要谈
的是另一件事。"` 含"资料"两个字，**不命中任何固定句式**，因此 `fallback_reason` 不是
`model_declined`——若检测靠"答案里有没有'资料'两个字"，这条会被误判成一次拒答。

### 封闭清单与取值理由

```python
prompt_template_of("v1")   # → RAG_PROMPT_V1
prompt_template_of("v3")   # → GenerationError：受管版本只有 ['v1', 'v2']
```

`prompt_template_of(version)` 只管"这个版本号站不站得住"，不管"缺省是哪一个"——缺省由
`settings.retrieval_prompt_version`（`None` 时）解析。两者分开之后，端点可以直接拿它校验
请求体里的版本号。`_validate_prompt_version` 同时认 `PROMPT_VERSIONS` 与
`CUSTOM_PROMPT_VERSION` **这两份清单**，因此"清单外的版本号一律报错"这条纪律没有被 `custom`
这道特例削弱——特例只有一个名字，且它精确地表示"这一份提示词来自调用方"。

**`CUSTOM_PROMPT_VERSION = "custom"` 的取值理由**：注入模板（构造参数 `prompt=...`）时，那次
生成用的不是任何一版受管模板，因此它的版本号不能借用 `"v1"` / `"v2"`——否则"这一版好不好"
会把一份自定义模板的结论算进 v2 的分组里。它**不在** `RAG_PROMPTS` 里（那不是一版受管模板），
但它是合法取值。**`DEFAULT_PROMPT_VERSION` 与 `CURRENT_PROMPT_VERSION` 今天是同一个值，却服务
两个问题**：前者回答"没指定时用哪一版"，后者回答"现在生效的模板是哪一份"。换版本时只改
`CURRENT_*`；改缺省值是一次显式决定（它会让所有没传 `prompt_version` 的调用方一起换版本）。

### `PROMPT_CHANGELOG`：变更原因写在能被读到的地方

```python
PROMPT_CHANGELOG = {
  "v1": "day066 首版：资料、问题与四条回答要求写在同一段里，顺序即语义（模型只能靠位置区分）",
  "v2": "day069 重写：按资料片段/问题/约束/输出格式四段式拆分；约束扩到 6 条并逐条注明挡哪种
         真实失败；新增固定拒答句式（让'模型自己说没有'可检测）与引用格式约束",
}
```

口吻与 `agent.prompt_library.PromptTemplate.changelog` 一致（一句话说清"改了什么、为什么
改"），它保证"换一版提示词之后答案变好还是变差"这件事**能被归因**——同一份索引、同一批问题，
只有固定的版本号才回答得了（day071 的 RAG 评估拿它分组）。

## 3. 引用溯源：`[n]` 的三个结局

`_grounding_report(answer, citations)` 是唯一一处把答案与编号表对账的地方，产出四组编号：

```text
cited      答案里出现过的编号（升序去重）—— **来自答案**
valid      cited 里真的落在提示词里的那些（升序）—— 有效引用
invalid    cited 里落不到提示词里的那些（升序）—— 幻觉引用
unused     给了它、但答案没引用的编号（升序）—— 覆盖率的分母
```

三条不变量在 `GroundingReport.__post_init__` 里逐条校验，因为它们一旦破了，报告里的数字就会
互相矛盾，而**读的人只会相信那个看起来最合理的**：

```text
valid ∪ invalid == cited 且两者不交   答案里出现的每个编号只可能落在两类之一
unused ∩ valid == ∅                   "给了没引用"与"引用了"不可能同时成立
coverage ∈ [0, 1]                     它是比例，越界说明分母算错了
```

### 编号的解析：`[0]` 不算编号

```text
_MARKER_PATTERN = r"\[\s*(\d+)\s*\]"   读的是**编号语法**：[1] / [ 1 ] / [12] 都认
编号从 1 起                            [0] **不是编号**：不进 cited / valid / invalid
```

`[0]` 为什么不进 `invalid`：编号从 1 起是本层的硬约定（见 `context.Citation.marker`），"0 号
引用"不是一次幻觉引用（它根本没在指任何位置），而是一次格式噪声——把 0 算进 `invalid` 会让
"模型引用了不存在的片段"这句指控多出一批并不成立的样本，而幻觉引用的**条数**是要进报告与评估的
一个数字。噪声只落在报告自己的注记里。本机实测：答案是 `结论 [0]（0 不是编号）。` 时
`cited = []`、`coverage = 0.0000`、`grounded = False`，与"一条引用都没有"的结论**逐位相同**。

### `record_id is None` 的含义（幻觉引用）

```text
marker       答案里出现的那个编号（从 1 起）
record_id    它对应的记录；None = 这个编号**不存在**（即幻觉引用）
used         这条编号在答案里被引到了没有
```

`record_id is None` 与 `used is False` 是**两件完全不同的事**，`checks` 把它们分成两种行：

```text
record_id=None  模型引用了不存在的编号（[9]）→ 幻觉引用，必须报出来；编号来自**答案**
used=False      片段给了它、模型没用          → 覆盖率的分母；编号来自**提示词**（我们给的）
```

第一种换模型或收紧提示词，第二种先看检索给的片段贴不贴题。本机实测：答案是
`…总会追溯到第 9 条片段 [9]。` 时 `invalid = [9]`，`checks` 里多出一行
`[9] **幻觉引用**：这个编号不在那次提示词里`。

### `checks` 的顺序与长度

```text
checks = 编号 1..n 的检查行（每条一行：record_id + used）
       + 越界编号的检查行（**升序**，record_id=None）
```

顺序固定：先按提示词的编号表（1 起连续），再按升序接上越界编号。长度因此是
`给出去的片段数 + 幻觉引用的条数`。本机实测：6 条片段、答案是 `[1][9]` 时 `checks` 有 **7 条**
（6 条来自提示词 + 1 条 `[9]`）；6 条片段、答案是 `[1][2]` 时是 6 条。

## 4. 接地判定：`grounded` 与 `coverage`

两个数字的定义写在这里，而不是靠字符串比较：

```text
coverage = len(valid) / len(known)       保留 **4 位小数**（round(…, 4)，照 PackedContext.fill_ratio）
grounded = valid 非空 且 invalid 为空    "这份答案有可核对的依据"
```

**为什么 `grounded` 不是字符串比较**："valid 非空"挡的是"答案里一个引用都没有"（一次没有依据
的总结），"invalid 为空"挡的是"引用了不存在的编号"（幻觉引用）。两者都不成立时，这份答案
**有可核对的依据**——注意它仍然可能与原意相反，语义判断不在这一层（所以它叫 grounded 而不是
correct）。若换成"看答案里有没有'资料中没有相关内容'"，兜底文案改一个字、模型换个说法，判定
就会静默失效，而它失效的样子恰好是"看起来一切正常"。`GroundingReport.__post_init__` 还会用
`expected = bool(self.valid) and not self.invalid` 反查 `grounded` 字段，口径不一致即报错。

`given` 是**算出来的**而不是存下来的：`given = len(valid) + len(unused)`（= 提示词里真正给出去
的片段数）、`hallucinated = len(invalid)`（报告里最该被看见的那个数）。可以这样算的理由：提示词
里的编号是 1 起连续的（`PackedContext` 校验过），于是"落在提示词里的"（valid）与"给了没引用的"
（unused）恰好把 1..n 分完。多存一个 `given` 就多一处可能与三组编号分家的副本
（`Generation.__post_init__` 会核对 `check.given == len(citations)`）。

### `coverage` 与 `PackedContext.fill_ratio` 的对称性

```text
PackedContext.fill_ratio = char_count / max_chars     两处都是比例，都 round(…, 4)
GroundingReport.coverage = len(valid) / len(known)
```

两者刻意用**同一个四舍五入口径**：报告里的"占用了多少预算"与"用上了多少片段"因此放在一起可比。
`coverage` 的分母是**给出去的片段数**（`known = len(citations)`），不是"检索召回的条数"——被
预算丢掉的尾部不在提示词里、模型看不到它们，把它们算进分母会让覆盖率凭空变低。`known` 为空
（`no_context` 那条路）时 `coverage = 0.0`，报告会额外记一句"没有任何片段可核对"。本机实测
（6 条片段、答案引用 2 条）：`coverage = 0.3333`、`grounded = True`；答案引用 `[1][9]` 时
`coverage = 0.1667`、`grounded = False`。

### 本机实测：编号集合从小到大的扫描表

```text
答案里的编号集合   cited         valid       invalid  unused        coverage  grounded
（无）            []            []          []       [1, 2, 3, 4]    0.0000  False
[1]              [1]           [1]         []       [2, 3, 4]       0.2500  True
[1][2]           [1, 2]        [1, 2]      []       [3, 4]          0.5000  True
[1][2][3]        [1, 2, 3]     [1, 2, 3]   []       [4]             0.7500  True
[1][2][3][4]     [1, 2, 3, 4]  [1, 2, 3, 4] []      []              1.0000  True
[1][9]           [1, 9]        [1]         [9]      [2, 3, 4]       0.2500  False
[0]              []            []          []       [1, 2, 3, 4]    0.0000  False
```

两行要看清：`（无）`与`[0]`的结论逐位相同（都 `0.0000` / 不接地）；`[1][9]` 是唯一"看起来引了
东西、但不接地"的一行（`valid` 非空但 `invalid` 也非空）。

## 5. 回退族：六种原因与判定优先级

"回退"不是"报错"，而是**这次答案不能按正常路径解读**。六种情形各自对应一个处置动作（与
`EMPTY_REASONS` 同一条纪律），因此它们是六条话而不是一个布尔标志。

| 优先级 | 取值 | 触发点 | answer 取谁 | `llm_called` |
|--------|------|--------|-------------|--------------|
| 1 | `no_context` | 检索 / 打包为空（`packed.is_empty`） | 兜底答复 `fallback_answer` | **False** |
| 2 | `llm_error` | 调用抛异常（超时 / 鉴权 / 网络 / 额度） | 兜底答复 | True |
| 3 | `empty_reply` | 模型返回空串（`strip()` 之后为空） | **原样保留**那个空串 | True |
| 4 | `model_declined` | 命中 `DECLINE_MARKERS` 里的固定句式 | 模型给的原文 | True |
| 5 | `unusable_citations` | `require_citation=True` 且 `check.valid` 为空 | 兜底答复（原文被丢弃） | True |
| — | `""`（`NONE`） | 以上都没命中：答案来自模型且核对出了有效引用 | 模型给的原文 | True |

判定顺序即优先级，写死在 `FALLBACK_REASONS` 里（**不含** `NONE`）：

```python
FALLBACK_REASONS = ("no_context", "llm_error", "empty_reply",
                    "model_declined", "unusable_citations")
FALLBACK_REASON_NONE = ""       # 哨兵：让 fallback_reason 永远有值可比，读的人不必先判空
```

### `no_context` 的"一次 LLM 都不调"为什么是硬护栏

它排在最前面，因为它是**最靠上游**的成因：没有片段，就谈不上调用与否。`Generation` 的
`__post_init__` 把这一条钉成一个**可判定的事实**：`llm_called=False`、`citations=()`、
`prompt_text=''` 三条必须同时成立（护栏写在渲染提示词**之前**：没有片段就不该先把它渲染出来，
否则一次"没调模型"会被记成"问过了但没答案"）。本机实测：空上下文时 `llm.calls == 0`（用一个
"记下每次调用"的 `MockLLM` 验到）、`prompt_text == ''`（0 字）、`citations == ()`、
`latency_ms` 接近 0（本次 `0.423ms`）。

### `llm_error` 下 `llm_called=True` 的理由

`llm_called=True` 写在 `_fallback` 这一层（**装配处**）而不是按原因分支：走到这里的路径都已经
真的调过模型了（`llm_error` 是调用抛了异常、`empty_reply` 是调用返回空串……）。护栏管的是
"该不该把片段交给模型"，管不了"给了片段之后调用失败"——把这两件事合成一个值，会让"调了但答案
没引用"与"根本没调"变成同一个标志，而它们的处置动作（换模型 / 调检索）完全不同。同理，
`llm_called` 与 `fallback_reason` **分开**：前者回答"模型有没有机会自由发挥"，后者回答"这份
答案为什么不能按正常路径读"。

### `empty_reply` 与 `no_context` 的区别

```text
empty_reply   调用过了、模型返回空串 → 去看提供方（超时 / 内容过滤 / max_tokens 太小）
no_context    一次都没调（没有片段）  → 去改检索（换说法 / 放宽过滤 / 调大预算）
```

处置动作不同，因此是两条话。"零预算"被刻意排除在这两者之外：`max_tokens < 1` 当场报
`GenerationError`（为 0 会让模型"没有空间说话"，表现是一次空回复——那会被误读成"资料里没有
相关内容"）。本机实测：`MockLLM` 返回 `"   \n  "` 时 `fallback_reason = "empty_reply"`、
`llm_called = True`、`answer.strip() == ""` 为 True，而 `answer` **原样保留**那个空串
（6 个字符），引用的片段仍有 6 条。

### `unusable_citations` 只在开关打开时出现

`require_citation=False`（缺省）时，一条合法引用都没有只会作为报告里的一项被看见（落正常路径）；
`True` 时它变成 `unusable_citations`，答案换兜底答复。它是一道**闸门**而不是一个阈值（"新能力
默认不生效"只对这一个开关生效）：打开之后，模型只是忘了写 `[n]` 就会被整条丢掉，因此先让默认可
核对率跑一段时间（day071 会给出那个数字），有了数据再决定要不要拧上。本机实测的四格矩阵：

```text
require_citation  有合法引用  fallback_reason            is_fallback
False             True        ""（正常路径）              False
False             False       ""（正常路径，coverage=0）  False
True              True        ""（正常路径）              False
True              False       "unusable_citations"        True
```

只有一格走了回退，那一格的 `answer` 换成兜底答复（被丢弃的原文长度记进 `notes`）。

## 6. 九步流水线：顺序本身就是语义

`RAGGenerator.generate(question, context, **overrides)` 的九步是**固定**的（docstring 原文）：

```text
1. 解析本次调用     问题与 overrides 一起校验（清单外的键当场报错）
2. 空上下文早退     is_empty → no_context，**一次 LLM 都不调**
3. 渲染提示词       占位符在构造期校验过，因此这里不会缺参数
4. 调用模型         **宽捕获 Exception** → llm_error（超时/鉴权/网络）
5. 空回复判定       strip 之后是空串 → empty_reply
6. 解析 [n]         折成 GroundingReport（valid / invalid / unused）
7. 拒答判定         命中固定句式 → model_declined
8. 引用开关         require_citation 且没有合法引用 → unusable_citations
9. 正常路径         fallback_reason=NONE（答案来自模型且核对通过）
```

结构图（每一步的输入与产物，以及它与上下两个把手的关系）：

```text
question ──┐  1. 解析（_check_question / _check_context / _resolve_overrides）
context  ──┘     清单外的键 → GenerationError（列出合法取值）
             2. packed.is_empty ? ──是──► _no_context()：llm_called=False、prompt_text=''、
                    │否                    citations=()、check.given=0、answer=兜底答复
                    ▼  3. prompt_text = template.format(context=…, question=…)  ← 原样存进结果
             4. llm.chat([Message(user)]) ──抛异常──► _fallback(llm_error)：answer=兜底答复
             5. reply.strip() == "" ? ──是──► _fallback(empty_reply)：answer=那个空串
             6. check = _grounding_report(reply, packed.citations)     ← 纯字符串，不花钱
             7. _declined(reply) ? ──是──► _fallback(model_declined)：answer=模型原文
             8. require_citation and not check.valid ? ──是──► _fallback(unusable_citations)
             9. Generation(fallback_reason=NONE, llm_called=True, check=check, …)
```

### 为什么第 2 步必须在渲染之前

它是**本课最重要的护栏**（与 `pipeline` 那道同源）：没有片段时模型仍然会写出一段通顺的答案，
而读的人看不出区别。若先渲染再判空，一次"没调模型"会被记成"问过了但没答案"——`prompt_text`
非空、`citations` 为空，报告里读起来像一次失败的提问，而实际上**根本没有提问**。因此
`llm_called=False` / `citations=()` / `prompt_text=""` 是护栏在数据上的形状。

### 为什么第 6 步排在第 7、8 步之前

拒答判定（第 7 步）与"一条合法引用都没有"（第 8 步）都要读核对结果，而核对本身不花钱（纯字符串
对账），先算出来就只写一次。第 4 步的宽捕获是刻意的：超时、鉴权失败、网络抖动都是**可预期的
失败**，它们必须变成一份带原因的结果而不是打断链路（`errors.GenerationError` 因此**不收留**
这三类）。

### `generate_many`：逐条而不是批量

`generate_many(items)` 逐条调用 `generate`（顺序与入参一致），且入参在**调用任何模型之前**逐条
校验——否则一个拼错的问题会在第 7 条才炸出来，而前 6 次调用已经花掉了。与
`RagPipeline.answer_many` / `Retriever.retrieve_many` 同样是刻意的逐条：每一次的 `llm_called` /
`fallback_reason` / 覆盖率都是独立的证据，而"这一批里几次不可交付"与"其中一次为什么不可交付"
是两个问题。

## 7. 参数：构造默认 ← settings ← `generate(**overrides)`

优先级是**三级**（写法与 `retriever` / `rerank` 一致）：

```python
RAGGenerator(llm, prompt_version=None, temperature=None, max_tokens=None,
             require_citation=False, fallback_answer=FALLBACK_NO_CONTEXT, model="")
generate(question, context, temperature=None, max_tokens=None, prompt_version=None)
```

```text
构造默认（字面量）  require_citation / fallback_answer / model / llm
↓ 没指定读 settings  retrieval_prompt_version / retrieval_answer_temperature /
                    retrieval_answer_max_tokens（那三个参数的 None 就是这个意思）
↓ 逐次覆盖          generate(**overrides) 里的三个键（**封闭清单**）
```

`None` 在这里有明确含义：**"没指定，去读 settings"**。于是"项目默认"只有一处定义，端点与演示
脚本不必各抄一遍。

### 四项新配置（`config.py` 的 day069 组）

| 配置项 | 缺省值 | env var | 说明 |
|--------|-------|---------|------|
| `retrieval_prompt_version` | `"v2"` | `RETRIEVAL_PROMPT_VERSION` | 缺省提示词版本；换它**改变答案文本本身** |
| `retrieval_answer_temperature` | `0.0` | `RETRIEVAL_ANSWER_TEMPERATURE` | 上限 `2.0`，越界当场报错 |
| `retrieval_answer_max_tokens` | `1024` | `RETRIEVAL_ANSWER_MAX_TOKENS` | 必须 `>= 1`；调小会得到空回复 |
| `retrieval_require_citation` | `False` | `RETRIEVAL_REQUIRE_CITATION` | 把"没有合法引用"当作不可交付（**闸门**） |

（`Settings` 没有设 `env_prefix`，因此环境变量名就是字段名大写——与其余 `retrieval_*` 同规则。）
这一组与 `retrieval_hybrid_*` / `retrieval_rerank_*` 一样**不改变任何产物的字节**（不改
`chunk_id`、不改向量、不改清单版本号），因此调它们随时生效、**不需要**重建索引。但它比前两组
**重一档**：前两组只改变**名单**，而 `retrieval_prompt_version` 换一版会改变**答案文本本身**
（同一份索引、同一个问题，v1 与 v2 交给用户的是两段不同的话）——因此换版本号是一次**需要留痕
的实验**（报告按它分组，见 `PROMPT_CHANGELOG`）。`temperature` 缺省 `0.0`：RAG 问答是**有依据
的复述**，温度越高模型越会改写片段里的事实（而引用编号仍然对得上，于是"改了事实"这件事在报告
里看不出来）。`max_tokens` 缺省 `1024` 与 `BaseLLM.chat` 的缺省一致：给"先结论后依据"留够空间。

### `GENERATION_OVERRIDE_KEYS` 的封闭清单

```python
GENERATION_OVERRIDE_KEYS = ("temperature", "max_tokens", "prompt_version")
```

多一个键就报 `GenerationError` 并列出合法取值，理由与 `pipeline.OVERRIDE_KEYS` 逐字相同——
**静默忽略一个覆盖参数会让调用方以为它生效了**（"我把 `max_tokens` 拼成 `max_token`，结果只是
答案变短了一点"）。两份清单**各自封闭、各自报错**：`pipeline.OVERRIDE_KEYS` 是
`("max_context_chars", "per_hit_chars", "temperature")`，它**不是**生成器清单的超集；
`temperature` 是唯一"路过"本层的键（`pipeline._PASSTHROUGH_KEYS`，校验只有一份实现）。

`prompt_version` 键有三处例外必须写清（它们就是"覆盖生效必须进 notes"这条纪律的落地）：

```text
注入自定义模板（custom）        逐次覆盖 prompt_version 换不了模板（版本号是标签，换标签不换
                              内容）→ 记一句"覆盖无法生效"进 notes
注入受管模板                   覆盖生效，但会改变答案文本 → 记"本次覆盖了提示词版本"进 notes
构造期同时给 prompt 与 version  以 prompt 为准，**被忽略的版本号**进 notes（而且它也要过清单
                              校验——拼错的版本号被记成"已忽略"会让调用方以为自己指定对了）
```

### 一条端点纪律（由本模块的形状决定）

`generate(**overrides)` 允许逐次覆盖那三个键，但**不允许在请求体里传提示词文本**：提示词是
"这一次模型到底被怎么问的"的唯一证据，一旦它能被请求方改写，`prompt_version` 就不再指着一份
确定的模板，day071 的分组实验也随之失去可复现性。要换模板请在**装配期**注入（`prompt=...`）。

## 8. 结果里的三份证据：读法

### `CitationCheck.to_dict()` 的键（4 个）

```text
marker / record_id / hallucinated / used
marker 必须是从 1 起的整数（0 / 负数 / 布尔都报错）；record_id 必须是非空字符串或 None（空串
顶替 None 会报错）；used 必须是布尔。summary_line()：幻觉引用是 "[9] **幻觉引用**：这个编号不在
那次提示词里"，其余是 "[1] k-02 已引用"
```

### `GroundingReport.to_dict()` 的键（9 个）

```text
given（算出来的）   cited / valid / invalid / unused（四组升序编号）
coverage（4 位小数） grounded（valid 非空 且 invalid 为空）
notes（关于**这份报告本身**的说明：空引用、非编号的 [0]、无从核对）
checks（逐编号检查行：先 1..n，再按升序接上越界编号）
```

`summary_line()` 是演示脚本逐行打印的那一行；`explain()` 给出四段，与四个问题的对应固定：
`1) 给了几条、引用了几条、有效几条 → 核对的口径；2) 覆盖率与接地是什么意思 → 别把它读成一次
事实核查；3) 有没有引用了不存在的编号 → 幻觉引用（编号会被点名）；4) 哪些片段白给了 →
unused`。第 3、4 段**按情况渲染**：没发生就不占一行（"注记多了会被读的人忽略"是这一层最该守的
纪律）。

### `Generation.to_dict()` 的键（14 个，含 `prompt_text`）

```text
question / answer / prompt_version / prompt_chars / llm_called / fallback_reason
is_fallback / grounded / model / latency_ms / citations / check / notes / prompt_text
其中 prompt_chars = len(prompt_text)（**只给长度，不重复那段正文**）；
is_fallback = fallback_reason != ""；grounded 转发 check.grounded；latency_ms round(…, 3)
```

`include_prompt=True` 是默认值（与 `RerankResult.to_dict` 的取向一致，与 `RetrievalResult`
相反）：这里最主要的消费者是**排查一次回答**（"模型到底被怎么问的"），而那段提示词正是它。
只要排序与统计时请显式传 `False`——一段两千字的提示词会把一次 diff 淹掉。

### `to_summary()` 为什么刻意不含 `prompt_text`

```text
prompt_version / model / llm_called / fallback_reason / grounded / given
coverage / valid / invalid / unused / latency_ms          ← 共 11 个键
```

它是**进上游公共字段的那一小段摘要**（端点、演示脚本与 day071 的评估都用它）——形状只在这里
定义一次，因为三份手写的摘要一定会分家。它**刻意不含 `prompt_text`**（也没有 `answer`）：摘要
是公共字段，一段两千字的提示词放在那里会变成第二份正文；要看细节的人手里本来就有完整的
`Generation`（其 `to_dict()` 默认带上提示词）。本机实测：14 个键 / 11 个键。

### `Generation.explain()` 的行结构

```text
1) 这次怎么问的        → 版本 + 提示词长度 + 模型 + 耗时
2) 答案能不能被核对    → 直接接上 check.explain() 的四段
3) 为什么走了回退      → 原因 + 出路（没走回退时这一行不渲染）
4) 还有什么必须被看见  → 注记（含被忽略的参数）
```

### `RagAnswer` 新增的两个字段

`pipeline.RagAnswer` 尾部新增 `check` 与 `fallback_reason`，**只增不改**（照 day068 给
`RetrievalHit` 加 `rerank_score` 的先例）：

```text
check           接地核对报告；**None = 这次没走到生成那一步**（空检索），不是"核对没通过"
fallback_reason 用 generation.FALLBACK_REASON_DESCRIPTIONS 那套封闭清单说明"为什么不算数"
                （空串 = 没有回退，与 RagAnswer 的缺省值一致）
```

空检索那条路上生成器一次都没被调用，因此 `check=None`（没有核对可言），`fallback_reason` 记
`no_context`——它是本层自己的护栏，报出来的名字必须与生成器那套一致（一个词汇表，两处使用）。
`RagAnswer.grounded` 在 `check is None` 时返回 `False` 而不是抛异常："没有核对可言"在报告里的
正确读法是"没有依据"。`to_dict()` 的两个新键加在**末尾**（键集合只增，老消费者读到的键一个都
不少也不改；本机实测共 10 个键）。

## 9. 端点清单（现在十三个）

| 端点 | 方法 | 入参 | 落到哪个入口 |
|------|------|------|-------------|
| `/retrieval/status` | GET | — | `Retriever.describe()` |
| `/retrieval/search` | POST | `query` + 七个检索参数 | `Retriever.retrieve` → `RetrievalResult` |
| `/retrieval/explain` | POST | 同上 | `Retriever.explain` |
| `/retrieval/answer` | POST | `question` + 同上 | `RagPipeline.answer`（day069 起响应尾部**只增** `check` / `fallback_reason`） |
| `/retrieval/routes` | GET | — | `StoreRouter.report()` |
| `/retrieval/hybrid` | POST | 同 `/retrieval/search` **+ `strategy` / `alpha` / `k_rrf`** | `hybrid.py::HybridRetriever.retrieve` |
| `/retrieval/lexical/status` | GET | — | `lexical.py::LexicalIndex.describe()` |
| `/retrieval/rerank` | POST | 同 `/retrieval/search` **+ `enabled` / `mode` / `top_n` / `weight` / `min_score` / `model`** | `rerank.py::rerank_hits`（**强制开启**） |
| `/retrieval/rerank/status` | GET | — | `CrossEncoderReranker.describe()` + settings 回显 |
| `/retrieval/rerank/compare` | POST | `query` + `relevant` + `k` / `top_n` / `mode` / `weight` | `rerank.py::measure_lift` |
| `/retrieval/generate` | POST | 同 `/retrieval/search` **+ `prompt_version` / `temperature` / `max_tokens` / `require_citation`** | `generation.py::RAGGenerator.generate` |
| `/retrieval/prompt/versions` | GET | — | `RAG_PROMPTS` / `PROMPT_CHANGELOG` / `FALLBACK_REASONS` / `DECLINE_MARKERS` |
| `/retrieval/grounding/verify` | POST | `answer` + 编号表 | `generation.py::_grounding_report`（**一次 LLM 都不调**） |

这一组的数量与内容**以 `smart_research_agent/api/routes.py` 为准**（响应字段以该文件与
`api/schemas.py` 为准，本文不复述）。day069 的生成能力有**两个入口**，分工是刻意的：

```text
POST /retrieval/generate        走完整链路：检索 → 打包 → 生成，把 Generation 的全套交代带出门
GET  /retrieval/prompt/versions 纯读：受管版本清单 + 每个版本的元信息 + 回退原因表 + 拒答句式表
POST /retrieval/grounding/verify 只做核对：给一段答案与一张编号表，返回六个数字——**不调模型**
```

`/retrieval/grounding/verify` 那一端点是"接地校验"最纯粹的一层：它连 `app.state.llm` 都不读，
因此可以在"没有配任何模型"的实例上照样答 200——排查"这次答案的引用到底哪里不对"时，
它比重新跑一次生成便宜得多。

三条入口纪律：**请求体里没有提示词文本**（第 7 节那条端点纪律）：要换模板只能给出受管版本名
（`v1` / `v2`）或在装配期注入（`prompt=...`，版本号记为 `custom`）；`/retrieval/generate`
沿用 `retrieval_llm(request)` 的既有判据（未配置 / 仍是 `MockLLM` 占位 → 400）；六个回退的
生成结果**一律 200**，结论落在 `fallback_reason` 与它的人话解释上，而不是状态码上——它们是
**合法的业务结论**，不是错误。生成器一侧的配置由 `RAGGenerator.describe()` 回显
（它顺带列出 `prompt_versions`，因为"合法取值有哪些"只有 `prompt_template_of` 说了算）。

`/retrieval/answer` 的两条**行为**纪律与 day066 相同：检索为空时 `200` 且 `llm_called=False`
（合法状态，不是 400）；未配置 LLM 时 `400` 且给出"注入一个真实 `BaseLLM`"的出路（**不拿
MockLLM 兜底**）。day069 只改了它的**内容**（多两个字段），没有改这两条。

## 10. 常见误区（12 条）

| # | 误区 | 事实 | 证据（可核对） |
|---|------|------|--------------|
| 1 | "生成器会判断这句话说得对不对" | 它只做**编号层面**的对账：`[n]` 与那次提示词的编号一一对上；语义忠实性不在这一层（所以叫 grounded 而不是 correct） | `generation.py` 模块 docstring；§4 |
| 2 | "`grounded` 靠字符串比较（看答案里有没有'资料中没有…'）" | 它由**编号**判定：`valid 非空 且 invalid 为空`；字符串判定会在文案改一个字之后静默失效 | `_grounding_report` 末段；`__post_init__` 的反查；§4 |
| 3 | "空检索时也会调一次模型，让兜底串来自模型" | **一次都不调**：`no_context` 是唯一一条 `llm_called=False` 的路，且必须 `prompt_text=''`、`citations=()` | `generate` 第 2 步；`Generation.__post_init__` 的三条钉住；§5 实测 |
| 4 | "`llm_error` 说明模型没被调用" | 它**确实调用了**（`llm_called=True`）：护栏管的是"该不该给片段"，管不了"给了之后调用失败" | `_fallback` 的装配注释；§5 实测 |
| 5 | "空回复与'知识库里没有相关内容'是一回事" | 前者是"调用过了但返回空串"（查提供方 / 调大 `max_tokens`），后者是"一次都没调"（改检索） | `FALLBACK_REASON_DESCRIPTIONS`；§5 实测两行 |
| 6 | "`[9]` 只是'数字写多了'" | 它是**幻觉引用**（`record_id is None`）：指到一份并不存在的依据；幻觉引用的**条数**要进报告与评估 | `CitationCheck` 的 `record_id=None` 说明；§3 实测 |
| 7 | "`[0]` 也是一次幻觉引用" | 编号从 1 起，`[0]` **不是编号**（是格式噪声）：不进 `cited/valid/invalid`，只进报告注记 | `_grounding_report` 的 `noise` 分支；§4 实测 |
| 8 | "`coverage` 的分母是检索召回的条数" | 分母是**给出去的片段数**（`len(citations)`）：被预算丢掉的尾部不在提示词里，模型看不到它们 | `_grounding_report` 的 `known`；`GroundingReport.given`；§4 |
| 9 | "`unused ∩ valid` 可以同时成立（既引了又没引）" | 不可能：两者互斥，`__post_init__` 直接报 `GenerationError`（账不自洽） | `GroundingReport.__post_init__` 的三条不变量 |
| 10 | "`require_citation` 缺省是开的" | 缺省 **False**（一道闸门，不是阈值）：打开之后模型只是忘了写 `[n]` 就会被整条丢掉 | `config.retrieval_require_citation`；§5 的四格矩阵实测 |
| 11 | "逐次覆盖提示词版本能换模板" | 只有**受管模板**能靠版本号换；注入了自定义模板（`custom`）时覆盖换不了模板，且**必须进 notes** | `_resolve_overrides` 的三条分支；§7 |
| 12 | "请求体里可以传提示词文本" | 不行：提示词是"这次模型被怎么问的"的唯一证据，可被改写会让 `prompt_version` 不再指着确定的模板 | §7 的端点纪律；`RetrievalAnswerRequest` 的"没有 prompt 文本" |

再补三条同性质的（写下来省一次排查）：

```text
13. "to_summary() 里找不到 prompt_text，是不是漏了" → 刻意不含：摘要只进上游公共字段，一段
                                     两千字的提示词放在那里会变成第二份正文（要看细节请用
                                     Generation.to_dict()，它默认带上提示词）
14. "checks 的条数应该恒等于片段数" → 它是「片段数 + 幻觉引用条数」：越界编号会在 1..n 之后
                                     按升序接上（实测 6 条片段 + [9] → 7 行）
15. "'拒答'可以用正则或编辑距离检测得更宽" → 不行：两者都会引入一个阈值。正则会把"没有相关内容"
                                     这半句算命中（等于丢掉一次正常回答），编辑距离的门限没有依据
```
