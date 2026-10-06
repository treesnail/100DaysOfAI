# 结业项目整合（一）手册：把八项能力装成一条可复算的链（day099 / G1-D1）

> 本课不新增任何第三方依赖、不重写任何子系统：它把 day0xx~day098 里已经各自成立的
> 能力**真实装配**成一条端到端调用链，并从清单渲染出「最终版 README + 架构文档」。
>
> 手册里的每一个数都来自本课快照内的可复现读数：
> `python scripts/capstone_demo.py`（十一节，产出 `outputs/capstone_demo.txt`）
> 与 `python -m pytest tests/test_capstone.py`。语料写死、编码器确定性、模型按脚本，
> 因此同一个数可以被重新跑出来。

---

## 一、本课只承诺一件事：**装配 + 渲染**，不承诺新的模型能力

```text
承诺      把八项能力装成一条能一次跑完、读数可复算的链（assembly.SystemAssembly.run）
          并从清单渲染「最终版 README + 架构文档」（document.render_readme / render_architecture）
不承诺    新的模型能力、新的算子、新的性能读数——
          本课一行新算术都不写，只把既有子系统接起来、把结果渲染成文档
```

因此本课的装配链**只调用别人**：`security`、`agent.planner`、`retrieval`（混合检索 /
打包 / 生成）、`evaluation.rag_metrics`、`observability`（成本 / trace）、`tools.calculator`、
`vectorstore` + `llm.embedding`、`llm.mock.MockLLM`。这是刻意的：**自证是不成立的，
真实调用既有子系统才是。**

```text
命令（cwd 为 day099/源码/smart-research-agent）：
    python scripts/capstone_demo.py
    python -m pytest tests/test_capstone.py -q -o addopts=""
```

---

## 二、八项能力总览（用户能看见的八种能力）

| id | 标题 | 承担子包 | 公开符号数 | 覆盖 |
| --- | --- | --- | --- | --- |
| `input_guard` | 输入侧护栏 | security、llm | 8 / 12 | ✓ |
| `task_planning` | 任务规划 | agent、llm | 11 / 12 | ✓ |
| `hybrid_retrieval` | 混合检索 | retrieval、vectorstore | 124 / 95 | ✓ |
| `context_packing` | 上下文打包 | retrieval | 124 | ✓ |
| `grounded_generation` | 接地生成 | retrieval、llm | 124 / 12 | ✓ |
| `offline_evaluation` | 离线评估 | evaluation | 20 | ✓ |
| `cost_and_tracing` | 成本与追踪 | observability、llm | 8 / 12 | ✓ |
| `tool_execution` | 工具调用 | tools | 5 | ✓ |

读数（清单汇总行）：

```text
覆盖 8/8 项能力 | 被认领 8 个包 | 无人认领 4 个
```

"公开符号数"由 `manifest.resolve_subpackage` 用 `importlib` 真的解析出来：
有 `__all__` 的包信 `__all__`，没有的（`agent` / `llm` / `tools`）数它目录下的子模块数
——**这个口径与导入顺序无关**，因此两次调用逐位相同。

---

## 三、九个阶段总览（端到端调用链）

```text
1. guard     护栏：注入检测 + 内容审核，决定这次请求能不能往下走
2. plan      规划：把问题拆成有序子任务（Planner + MockLLM 剧本）
3. retrieve  检索：混合检索器取回 top-k 命中（两路共用一份过滤与深度）
4. pack      打包：把命中折成带编号、有预算的提示词片段（[n] 从 1 起连续）
5. generate  生成：把上下文交给模型，要求它给出可核对的引用
6. ground    接地：把答案里的 [n] 与那次提示词的编号对账（幻觉引用 = 0）
7. evaluate  评估：召回 / 精确 / MRR / NDCG 四条离线指标现场算出
8. account   记账：按模型与接口归因 token 与费用（价格表折算）
9. trace     追踪：把这一整次调用记成一条 trace（root + 每个阶段一个 span）
```

阶段与能力的映射只有一处定义（`types.STAGE_CAPABILITIES`），可以被逐键检查：

```text
guard→input_guard  plan→task_planning  retrieve→hybrid_retrieval  pack→context_packing
generate→grounded_generation  ground→grounded_generation  evaluate→offline_evaluation
account→cost_and_tracing  trace→cost_and_tracing
```

---

## 四、端到端读数（一次真实运行）

```text
 1. [guard   ] ✓ | 读数 0 | 护栏：放行 | 注入 0 / 敏感 0 / PII 0 | 检查 12 字
 2. [plan    ] ✓ | 读数 3 | 规划：3 条子任务 拆解问题、检索语料、核对引用 | 工具自检 2 + 3 * (4 - 1) = 11 ✓
 3. [retrieve] ✓ | 读数 3 | 检索：3 条（k-3=+0.0328、k-2=+0.0320、k-6=+0.0320）| empty_reason=hits
 4. [pack    ] ✓ | 读数 3 | 打包：147/600 字 | 引用 3 条 | 截断 0 / 丢尾 0
 5. [generate] ✓ | 读数 1 | 生成：模型已调用 | 回退 （无） | 32 字
 6. [ground  ] ✓ | 读数 0 | 接地：通过 | 有效 1 / 幻觉 0 / 未引用 2 | 覆盖 33.3%
 7. [evaluate] ✓ | 读数 1 | 评估：召回 1.0000 | 精确 0.3333 | MRR 1.0000 | NDCG 1.0000 | 金标准 ['k-3']
 8. [account ] ✓ | 读数 3.78e-05 | 记账：2 次调用 | token 210（196+14） | 费用 $0.00003780（gpt-4o-mini）
 9. [trace   ] ✓ | 读数 9 | 追踪：9 个 span（root=capstone.run）
 汇总：阶段 9 段全通过=True | 召回 1.0000 | 费用 $0.00003780 | token 210 | span 9 | 摘要 f052e4a7dd78d232
```

四条最值得读的读数：

```text
检索的三条        k-3（含 ERR-2043 的那条）、k-2、k-6 —— 金标准 k-3 排第 1
接地的三组编号    有效 1 / 幻觉 0 / 未引用 2（覆盖率 33.3%）
记账的两个数      2 次调用、210 token（196 + 14）→ $0.00003780
追踪的一行        9 个 span = root(capstone.run) + 8 个真的执行了的阶段
```

第 9 条那个 9 不是写死的：它是 `Tracer` 真的记下来的条数。trace 阶段自己在 root 落盘
之后才算得出来，因此它不占一个 span——`EXPECTED_SPANS = len(ASSEMBLY_STAGES) = 9`。

---

## 五、七条性质与三类判据

```text
 1. capabilities_are_covered                 | [equality   ] | 读数          8 ✓
 2. stages_match_spec                        | [equality   ] | 读数          0 ✓
 3. assembly_is_reproducible                 | [equality   ] | 读数          0 ✓
 4. retrieval_recall_meets_floor             | [lower_bound] | 读数          1 ✓
 5. grounding_has_no_hallucination           | [upper_bound] | 读数          0 ✓
 6. cost_matches_hand_formula                | [upper_bound] | 读数          0 ✓
 7. document_covers_all_capabilities         | [equality   ] | 读数          0 ✓
```

判据分布：

```text
[equality] 4 条：读数与期望逐位 / 整数相同（没有容差空间）
[upper_bound] 2 条：读数不超过某个界（越少越好）
[lower_bound] 1 条：读数不低于某个底（越多越好）
```

第 ④ 条是唯一一条下界，而它**只能**是下界：召回度量的是"金标准命中了没有"。
把召回写成"等于 1.0"的相等判据，会在取回更多条（更好）时误报失败——
召回一旦到 1.0 就封顶，而"等于"这种写法会让人误以为"比 1.0 大才算更好"。
第 ⑤ ⑥ 条是上界：它们定义上应当恰好是 0（幻觉引用）或一个浮点容差（成本相对差），
写成上界能让"为什么不是下界"在判据名上说清楚——它们是**越少越好**的量。

---

## 六、能力清单与覆盖（谁负责哪一项）

```text
覆盖表（8 项能力 → 承担子包）：
  input_guard         | ✓ | security、llm           | 8/12
  task_planning       | ✓ | agent、llm              | 11/12
  hybrid_retrieval    | ✓ | retrieval、vectorstore  | 124/95
  context_packing     | ✓ | retrieval              | 124
  grounded_generation | ✓ | retrieval、llm          | 124/12
  offline_evaluation  | ✓ | evaluation             | 20
  cost_and_tracing    | ✓ | observability、llm      | 8/12
  tool_execution      | ✓ | tools                  | 5

十二个候选子包（在场 / 公开符号数 / 认领）：
  security      | 在场 | 公开   8 个符号 | 认领：input_guard
  agent         | 在场 | 公开  11 个符号 | 认领：task_planning
  retrieval     | 在场 | 公开 124 个符号 | 认领：hybrid_retrieval、context_packing、grounded_generation
  evaluation    | 在场 | 公开  20 个符号 | 认领：offline_evaluation
  observability | 在场 | 公开   8 个符号 | 认领：cost_and_tracing
  tools         | 在场 | 公开   5 个符号 | 认领：tool_execution
  llm           | 在场 | 公开  12 个符号 | 认领：input_guard、task_planning、grounded_generation、cost_and_tracing
  vectorstore   | 在场 | 公开  95 个符号 | 认领：hybrid_retrieval
  indexing      | 在场 | 公开  57 个符号 | 认领：（无人认领）
  mcp_server    | 在场 | 公开   6 个符号 | 认领：（无人认领）
  finetune      | 在场 | 公开  48 个符号 | 认领：（无人认领）
  api           | 在场 | 公开   1 个符号 | 认领：（无人认领）

被认领（8）：security、llm、agent、retrieval、vectorstore、evaluation、observability、tools
无人认领（4）：indexing、mcp_server、finetune、api（已交付，但不属于这 8 项能力清单）
```

"无人认领"这份名单值得单独读：`indexing`（索引清单与版本）、`mcp_server`（MCP 服务端）、
`finetune`（微调数据管线）、`api`（HTTP 端点）都是**真的写出来了**的包，
但它们不在这 8 项能力的清单里——报告必须把这件事说出来，
否则"12 个候选子包里有 4 个没人认领"会被误读成"有 4 个包是坏的"。

---

## 七、复算口径（本课最硬的一条证据）

```text
第一次摘要：f052e4a7dd78d232
第二次摘要：f052e4a7dd78d232
逐位相同：True | 差异项数 0（0 = 逐位相同）
```

`SystemRun.comparable()` 只含**确定性字段**：

```text
问题 / 九段记录（阶段名 / 成没成 / 读数 / 一行证据）/ 答案
四条指标 / 费用 / 输入 token / 输出 token / span 条数
```

**不含**：时间戳、uuid、耗时、日志文本。把它们放进读数会让第 ③ 条性质永远失败——
而"耗时不该进复算口径"是本课刻意留下的一条边界。为了做到这一点，`TraceReading`
只数 span 的**条数与名字**，不碰 `time.time()` 与 `uuid4()`。

两次运行互不干扰的机制在 `SystemAssembly`：`run()` 每次**新建一份底座**
（`build_substrate`），因此 `MockLLM` 的脚本从同一条起跑线出发。
真实的结业链路里，"这一次请求"本来就不该复用一个被上一次请求改过状态的模型对象——
那正是"两次运行读数不同"最常见的病因。

---

## 八、失败族（按「该谁去修」分）

```text
ManifestError    → 改清单或改包：子包名写错 / 包被改没了，先把名字改对再重建清单
CapabilityError  → 改能力表：未知能力或没有承担子包时，先把能力表补完整
StageError       → 改实现或改阶段表：链的形状（次序 / 条数）对不上时，先让两者对齐
AssemblyError    → 改实现：两次运行不一致时，去掉那个没有固定的量（时间 / 哈希序 / 采样）
DocumentError    → 改渲染器：文档漏掉某项能力时，先把那一项渲染进正文
NumericError     → 改数据或改实现：非有限读数与非法上下界都属于'数值不可用'
ParameterError   → 改调用：子包名 / 能力名 / 性质名 / 序号都是调用点的一次决定
```

**回来的族**：`DocumentError`。day0xx 的 `documents.errors.DocumentError` 第一次给
"文档解析失败"命名了，此后各课都把它当工具函数、不把它当一次控制流事件。
day099 把"渲染最终版 README 与架构文档"变成结业链路的最后一个交付物
（`document.render_readme` / `render_architecture`），于是这个名字**回来**了——
但基类从 `Exception` 换成 `CapstoneError(ValueError)`：今天它要能被同一条
`except ValueError` 兜住，与其余六族同源。

**缺席的族**：`GradientError`。本课装配的链里唯一的"梯度事件"仍然是被调用的那一行
（`finetune` 那一侧），本课连一次反向都没有触发；它是一个纯装配与渲染的包——
检索器、生成器、指标、工具全是既有的，本课只把它们接起来、把结果渲染成文档。
因此这一族今天不属于本包——它属于被调用的代码。

---

## 九、文档渲染（最终版 README + 架构文档）

```text
readme         |  2748 字符 | 覆盖 8/8 项能力 | 缺：（无）
architecture   |  3751 字符 | 覆盖 8/8 项能力 | 缺：（无）
```

两份文档**不是手写的**，而是从清单渲染出来的：

```text
render_readme         8 项能力表 + 9 个阶段表 + 7 条性质表 + 被认领/无人认领 + 边界 + 怎么跑
render_architecture   装配总览 + 逐能力落点 + 12 个子包职责 + 接缝 + 复算口径 + 失败族
check_document_covers_all  文档必须覆盖全部 8 项能力，否则抛 DocumentError
```

渲染是**确定性**的：同一清单渲染两次逐字节相同（没有时间戳、没有"生成于"、没有哈希序）。
`write_documents(dir)` 把两份文档写到给定目录（演示脚本写到 `outputs/capstone/`），
**不覆盖仓库根的那份 `README.md`**——"新渲染一份"与"改掉旧的那份"是两件事：
前者可复算，后者会掩盖 diff。

---

## 十、十个模块与它们的职责

```text
errors.py    七个失败族（回来的 DocumentError + 继续缺席的 GradientError）
types.py     8 项能力 / 9 个阶段 / 12 个候选子包 / 7 条性质（判据分三类）/ 10 条笔记 / 5 条边界
manifest.py  用 importlib 把能力真的落到子包上，产出清单与覆盖报告
adapters.py  薄适配层：真实子系统 → 九种统一读数（只带确定性标量）
assembly.py  SystemAssembly + run(question) → SystemRun（九段 + 指标 + 成本 + trace）
document.py  从清单渲染 README 与架构文档，并检查覆盖全部 8 项能力
verify.py    七条性质与三类判据（相等 / 上界 / 下界）
study.py     五张表（能力 / 清单 / 阶段 / 性质 / 文档）
__init__.py  长 docstring + 八个模块 __all__ 的并集（字母序）
```

关键函数签名：

```text
manifest.resolve_subpackage(name: str) -> ModuleInfo
manifest.build_manifest(*, capabilities=None, candidates=None) -> Manifest
adapters.build_substrate(*, question=QUESTION, top_k=TOP_K) -> Substrate
assembly.SystemAssembly(substrate=None, *, top_k, max_chars, per_hit_chars).run(question=None) -> SystemRun
assembly.run(question=None) -> SystemRun
document.render_readme(manifest=None) -> str ; document.render_architecture(manifest=None) -> str
document.check_document_covers_all(documents, *, table=None) -> tuple[str, ...]
verify.check_all(run_result=None, manifest=None, documents=None) -> PropertyReport
study.study_lines(run_result=None, manifest=None, documents=None) -> tuple[str, ...]
```

---

## 十一、十条笔记

```text
 1. 结业项目的整合**不重写任何子系统**：它把已经各自成立的能力装成一条链。
    重写会让你同时面对两处未知——而整合只面对一处：'接缝对不对'。
 2. 一项能力要落到**子包**上，而不是落到某一行上：
    '由哪个包负责'是可以被别人核对的问题，'由哪一行实现'不是。
 3. 护栏排在最前，因为它是唯一一处**能省下后面所有花费**的阶段。
 4. 规划只花一次模型调用，却把'这次要做什么'变成一份可读的清单。
 5. 混合检索的每一条命中都留着'哪几路召回了它、各贡献了多少'。
 6. 打包是唯一一处能回答'上下文为什么这么短'的地方：截断与丢尾各自留数字。
 7. 空上下文时**一次模型都不许调**：最像成功的那种失败，正是一个写得很通顺的
    无依据答案。
 8. 接地的判据是**编号**而不是字符串，因此换一版提示词、改一个字都不会静默失效。
 9. 离线的四条指标只认识'名单 + 金标准'，它量的是**检索**的质量，不是答案的质量。
10. 记账与追踪是同一次调用的两张收据：一张回答'花了多少'，一张回答'经历了几步'。
```

---

## 十二、五条边界与与既有包的接缝

五条边界（**这一课明确不承诺的事**）：

```text
1. 本包**不新增任何第三方依赖**：装配用到的每一个子系统都是快照内既有的
2. 本包**不改动任何既有模块、既有测试、pyproject 与 docs 下的既有文件**——
   它只新增文件，既有子系统一行未改
3. 本包复现的是**装配与读数**，不是新的模型能力：它不重写检索器、生成器或指标
4. 本包的全部读数**离线、确定性**：用 MockLLM、固定语料与固定编码器，
   不联网、不读任何环境变量密钥
5. README 与架构文档是**从清单渲染**出来的文本，本包不承诺它的文笔——
   它承诺的是'8 项能力每一项都出现在正文里'这条可断言的事实
```

与既有包的接缝（本课是它们的使用者，一行未改）：

```text
guard     security.injection_detector / security.content_moderator
plan      agent.planner.Planner（+ tools.calculator.CalculatorTool）
retrieve  retrieval.hybrid.HybridRetriever（build_retriever + LexicalIndex）
pack      retrieval.context.pack_context
generate  retrieval.generation.RAGGenerator（配 llm.mock.MockLLM）
ground    retrieval.generation.GroundingReport
evaluate  evaluation.rag_metrics（recall / precision / mrr / ndcg）
account   observability.cost_tracker.CostTracker
trace     observability.tracing.Tracer
library   vectorstore.FlatVectorStore / llm.embedding.CharNgramEmbedding
```

**下游**：下一课（结业项目整合（二））会在这条链之上接部署与服务，
本课的 `SystemRun` 就是它的"第一份可复算的基线"。

---

## 附：一次跑通的命令与产出

```text
python scripts/capstone_demo.py
    → outputs/capstone_demo.txt               （十一节完整输出）
    → outputs/capstone/README.capstone.md     （渲染出的最终版 README）
    → outputs/capstone/architecture.capstone.md（渲染出的架构文档）

python -m pytest tests/test_capstone.py -q -o addopts=""
    → 77 passed
```
