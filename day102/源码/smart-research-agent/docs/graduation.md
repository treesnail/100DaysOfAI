# `graduation` 手册：把一次能跑的运行交付成四份能被复算的产物

> 对应课程日：**day100（G2 · 结业项目整合（二））**
> 主线项目：智研 AI 助手（SmartResearch Agent）
> 前置：day099 的 `capstone`（九段端到端链 + 12 个候选子包的清单）
> 全文离线、确定性：不联网、不读任何环境变量密钥

---

## 一、这一课解决什么问题

day099 把八项能力装成了一条**端到端可复算的链**，并给了它一条判据：
"同一输入两次运行逐位相同"。但链跑得对，并不等于**交付做得成**。

交付要回答的是一个不同的问题：

```text
别人拿到我们的仓库，能不能不看我们的解释就复核这件事？
```

`graduation` 包把这个问题拆成**四份交付物**，每一份都回答一个独立的问题：

| 交付物 | 它回答的问题 | 产出函数 |
|--------|--------------|----------|
| `demo` 演示剧本 | 这一次端到端运行，能不能被别人原样重放？ | `demo.build_transcript` / `demo.replay` |
| `summary` 课程清单 | 这门课到底交付了哪些子包，各有多少东西？ | `inventory.build_inventory` / `summary.build_summary` |
| `assessment` 能力自评 | 八项能力分别建到了哪一步，各自的证据是什么？ | `assessment.assess` |
| `roadmap` 后续规划 | 按真实缺口排，下一步该补什么？ | `roadmap.plan` |

**今天最值钱的一句话**：

> 交付不是把结果贴出来，而是把结果变成**别人能复算**的产物：
> 剧本要能被重放、清单要能被 `importlib` 核对、自评要能落到证据上、
> 规划要能从缺口推出——四份都如此，交付才算完成。

---

## 二、九个模块

```text
errors.py       七个失败族（回来的 VersionError + 继续缺席的 GradientError）
types.py        4 份交付物 / 4 级自评量程 / 7 条性质（判据分三类）/ 10 条笔记 / 5 条边界
inventory.py    用 pkgutil + importlib 把**全部**一级子包数一遍
demo.py         剧本生成 + 重放（两次逐位比），以及剧本落盘
assessment.py   8 项能力逐项评级：证据 → 量程（一处定义，可被复核）
roadmap.py      从真实缺口推出规划（缺口集合 == 规划项集合）
summary.py      把清单折成一份可核对的课程总结
verify.py       七条性质与三类判据（相等 / 上界 / 下界）
study.py        五张表（交付物 / 清单 / 自评 / 规划 / 性质）
__init__.py     包入口（九个模块 __all__ 的并集 + 两条导入期不变式）
```

---

## 三、四份交付物的口径

### 3.1 演示剧本：展示，而不是宣称

剧本里**没有一个字是形容词**。每一行都是 `capstone.assembly.run()` 真的算出来的读数：

```text
# 演示剧本 · 100 天 AI 学习计划 · 结业里程碑
# 问题：向量维度不一致该怎么处理
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
里程碑：100 天 AI 学习计划 · 结业里程碑 | 第 100 天 | 运行摘要 f052e4a7dd78d232 | 阶段 9 段全通过=True
# 剧本摘要：78c8ba8284c3ba99
```

剧本摘要（`78c8ba8284c3ba99`）= `sha256`（问题 + 九段读数 + 汇总行 + 归档行）前 16 位。
它与 `capstone` 的运行摘要（`f052e4a7dd78d232`）**不是同一个数**：
运行摘要覆盖四条指标、费用与 token，剧本摘要覆盖**剧本的文本**——
两者各管一段，"剧本没变"与"运行没变"是两件事。

### 3.2 重放：一次 `==` 就能证明

```python
first = demo.build_transcript(capstone_assembly.run())
second = demo.build_transcript(capstone_assembly.run())
report = demo.ReplayReport(first=first, second=second)
assert report.identical            # diff_count == 0
```

重放口径（`DemoTranscript.comparable`）**不含**：时间戳、uuid、耗时、日志文本。
把它们放进来，"两次重放逐位相同"这条性质会永远失败——而失败的原因与剧本无关。
这与 day096 把时钟做成可注入、day099 把时间移出复算口径是同一条纪律：
**可复现的前提是先把"不确定的东西"从读数里挪出去。**

### 3.3 课程清单与总结：数出来，不是抄出来

```python
names = inventory.discover_subpackages()   # pkgutil.iter_modules，不 import
report = inventory.build_inventory()       # 对每个子包 importlib 一次，数公开名字与文件
```

本仓库当前的读数（`day100/源码/smart-research-agent`）：

```text
子包 52 个 | 在场 52 | 公开名字 4767 | 内部文件 379
总结：100 天 | 子包 52 | 文件 379 | 名字 4767 | 能力 8 | 阶段 9 | 性质 7 | 交付物 4
```

`InventoryReport.__post_init__` 有一条闭合检查：**发现集合 == 记录集合**。
少一个子包（漏项）与多一个（重复）都会当场变红——
一个被漏掉的包既不会出现在名册里，也不会被任何人负责。

`summary.summary_matches_inventory` 把"总结"与"清单"的两个口径对账：
总结给人读、清单给程序读，**两条独立路径给出同一个数**，才叫"总结被核对过"
（与 day099 第 ⑥ 条"记账 vs 手算"同一套纪律）。

### 3.4 能力自评：评级只由证据推出

每一行的评级来自三条证据：

```text
证据① 承担子包在场数 / 承担数
证据② 公开名字数（对门槛 SYMBOL_FLOOR = 8）
证据③ 端到端链上那一段 ok
```

```text
LEVEL_NOT_BUILT (0)  有承担子包不在场
LEVEL_BUILT     (1)  全部在场，但公开名字数 < 门槛
LEVEL_USABLE    (2)  前两项成立，但链上那一段 ok=False
LEVEL_DELIVERED (3)  三项全过
```

`CapabilityScore.__post_init__` 做两条闭合检查：
① 证据条数至少一条；② 评级**等于**由证据重算出来的那一档。
第二条让"手写一个 3 分"不可能悄悄通过。

**本课真实读数**：

```text
L3 input_guard         | 承担 2/2 | 名字  20 | 阶段 guard ok=True
L3 task_planning       | 承担 2/2 | 名字  23 | 阶段 plan ok=True
L3 hybrid_retrieval    | 承担 2/2 | 名字 219 | 阶段 retrieve ok=True
L3 context_packing     | 承担 1/1 | 名字 124 | 阶段 pack ok=True
L3 grounded_generation | 承担 2/2 | 名字 136 | 阶段 generate ok=True
L3 offline_evaluation  | 承担 1/1 | 名字  20 | 阶段 evaluate ok=True
L3 cost_and_tracing    | 承担 2/2 | 名字  20 | 阶段 account ok=True
L1 tool_execution      | 承担 1/1 | 名字   5 | 阶段 plan ok=True   ← 唯一的 L1
```

`tool_execution` 停在 **L1**：它的承担子包 `tools` 公开面只有 **5** 个名字，
低于门槛 8。这不是缺陷，而是**自评的诚实**——它给出一个 1 分并说明为什么，
并把这件事交给了后续规划（见下一节）。

### 3.5 后续规划：从缺口推出

缺口只有两种，两种都来自一次真实读数：

```text
unclaimed_package   无人认领的子包     已交付、但不属于这 8 项能力清单（来自清单）
capability_below    没到最高级的能力   在场 / 符号 / 阶段三项里有一项没过（来自自评）
```

```text
1. [unclaimed_package] indexing     → 评估是否纳入主线能力清单；若纳入，补一条端到端阶段
2. [unclaimed_package] mcp_server   → 同上
3. [unclaimed_package] finetune     → 同上
4. [unclaimed_package] api          → 同上
5. [capability_below ] tool_execution → 把这一项能力的证据补齐到 L3
```

`Roadmap.missing(gap_keys)` / `Roadmap.ignored(gap_keys)` 检查两个方向：
**有缺口没规划**与**规划指向不存在的缺口**。一份对不上的规划读起来更完整，
但它不可反驳——因此第 ④ 条性质就是"未覆盖的缺口数 + 多余的规划项数 == 0"。

---

## 四、七条性质与三类判据

```text
相等（==）   ① demo_replays_identically        两次重放的差异项数 == 0
             ② deliverables_are_complete       缺失交付物数 == 0
             ③ inventory_accounts_for_all      重复 + 漏项 == 0
             ④ roadmap_covers_every_gap        未覆盖的缺口数 + 多余的规划项数 == 0
             ⑦ summary_matches_inventory       总结与清单的差异项数 == 0
上界（<=）   ⑤ assessment_within_scale         最高评级 <= 3（TOP_LEVEL）
下界（>=）   ⑥ assessment_has_evidence         最少证据条数 >= 1
```

实测（`verify.check_all()`）：

```text
1. demo_replays_identically     | [equality   ] | 读数 0        ✓ | 摘要 78c8ba8284c3ba99 / 78c8ba8284c3ba99
2. deliverables_are_complete    | [equality   ] | 读数 0        ✓ | 交付物 4/4 份非空
3. inventory_accounts_for_all   | [equality   ] | 读数 0        ✓ | 子包 52 / 发现 52 | 重复 0 / 漏项 0
4. roadmap_covers_every_gap     | [equality   ] | 读数 0        ✓ | 缺口 5 / 规划 5 | 未覆盖 0 / 多余 0
5. assessment_within_scale      | [upper_bound] | 读数 3        ✓ | 最高 L3 ≤ 上界 L3
6. assessment_has_evidence      | [lower_bound] | 读数 3        ✓ | 最少证据 3 条 ≥ 下界 1
7. summary_matches_inventory    | [equality   ] | 读数 0        ✓ | 差异项数 0
合计：7/7 条通过
```

**为什么第 ⑥ 条必须是下界**："每一项能力都至少挂一条证据"**只能从下面兜住**——
证据越多越好，到 3 条、4 条都不会"超过"某个上限。写成相等判据会在报告补了一条证据时误报失败。

**为什么第 ⑤ 条是上界**："最高评级不超过量程"是一条**越少越好**的量（越界 = 一个分数被读成它不属于的档）。
它不是相等：8 项能力不可能都恰好是 3 分才是合格的——
本课的真实状态就是"7 项 L3 + 1 项 L1"，而这一项已经被第 ④ 条兜住了。

---

## 五、失败族（按「该谁去修」分）

| 族 | 该谁去修 |
|----|----------|
| `MilestoneError` | 改里程碑定义或改数据：天数与归档对不上时，先把那份可复算的读数补上 |
| `DemoError` | 改剧本生成：重放不一致或剧本缺段时，先让两次重放给出同一份剧本 |
| `InventoryError` | 改扫描：重复发现或漏项时，先让"发现"与"记录"是同一个集合 |
| `AssessmentError` | 改自评规则：未知能力、没有证据或评级越界时，先把证据补齐 |
| `RoadmapError` | 改规划：有缺口没有下一步时，先让缺口集合与规划项逐键对上 |
| `NumericError` | 改数据或改实现：非有限读数与非法上下界都属于"数值不可用" |
| `ParameterError` | 改调用：交付物名 / 缺口类型 / 阈值 / 序号都是调用点的一次决定 |

两条族常量：

```text
RETURNED_FAMILY = "VersionError"
    day065 的 indexing.errors.VersionError 第一次给"索引版本对不上"命名；
    day100 把"100 天里程碑"归档成一个**可复算的版本事件**，于是这个名字回来了——
    但基类从 Exception 换成 GraduationError(ValueError)。

ABSENT_FAMILY = "GradientError"
    本课是交付与归档，一个新式子都没有写；唯一的"梯度事件"仍是被调用的 finetune 那一侧。
```

---

## 六、四条纪律

1. **不重写任何子系统**：本包只新增代码；既有模块、既有测试、`pyproject` 与 `docs/`
   下的既有文件**一行未改**。它消费 day099 的 `capstone`（链与清单）。
2. **读数必须现场算出**：`study` 里不存数字；每张表的每行都来自函数调用。
3. **确定性**：复用 day099 的固定底座（MockLLM + 固定语料 + 固定编码器），
   因此剧本重放两次逐位相同、清单两次相同。
4. **不能联网、不能读密钥**：本包的全部读数都在本地可复算。

---

## 七、五张表

```text
① 交付物表   4 份交付物：id / 标题 / 它回答的问题 / 正文行数
② 清单表     全部一级子包：在场 / 公开名字 / 文件 / __all__
③ 自评表     8 项能力：评级 / 承担子包 / 名字数 / 链上阶段
④ 规划表     每个真实缺口一条：类型 / 理由 / 下一步
⑤ 性质表     7 条性质：判据类别 / 读数 / 结论 / 一行证据
```

每一行都带**两个数**（一个读数与一个参照）——一行只写"通过"的表是没法反驳的。

---

## 八、怎么跑

```bash
cd day100/源码/smart-research-agent
python scripts/graduation_demo.py                                # 十一节离线演示
python -m pytest tests/test_graduation.py -q -o addopts=""       # 本包 74 个用例
python -m pytest -q                                              # 全量回归
```

演示脚本的产出：

```text
outputs/graduation_demo.txt             本脚本的完整输出（在 .gitignore 里）
outputs/graduation/demo.transcript.txt  落盘的演示剧本（新的产物，不覆盖任何既有文件）
```

---

## 九、边界（本课明确不承诺的事）

- 本包**不新增任何第三方依赖**：四份交付物用到的每一个读数都来自快照内的既有子系统。
- 本包**不改动任何既有模块、既有测试、pyproject 与 docs 下的既有文件**——它只新增文件。
- 本包交付的是**产物与判据**，不是新的模型能力：它不重写检索器、生成器或指标。
- 本包的全部读数**离线、确定性**：复用 day099 的固定底座，不联网、不读任何环境变量密钥。
- 能力自评是**基于证据的评级**，不是对真实世界效果的承诺——
  它衡量的是"这一项能力在本仓库里被建到哪一步"，不是"它在生产环境里有多强"。

---

## 十、与既有包的接缝

```text
chain      capstone.assembly.run（九段端到端链，复用 day099）
manifest   capstone.manifest.build_manifest（12 个候选子包的解析结论）
reproduce  capstone.SystemRun.comparable / digest（复算口径与摘要）
scan       pkgutil + importlib（全部一级子包，本课新增）
```

本包是整条 100 天主线的**最后一个交付层**：它不生产新能力，
只把"这条路走完了"变成四份能被别人复核的产物。
