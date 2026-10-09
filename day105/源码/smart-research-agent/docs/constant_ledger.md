# 常量台账（constant_ledger）—— day105 手册

> 本手册是 `smart_research_agent.constant_ledger` 的使用说明与设计说明。
> 它由 day105 交付，对应当天的教程 `day105/教程/教程.md`。

## 1. 这份台账回答什么问题

100 天里，几百份 `.py` 各自把一些名字钉成了值。今天把这些**模块级常量**收成一份**可核对的台账**：

```text
这个模块把哪些名字钉住了？          → ConstantLedger.entry_of(module).names()
它们的取值能不能离线算出来？         → entry_of(module).primary_defs()（literal / opaque）
哪个名字被多个模块同时钉住？         → ConstantLedger.shared_groups()
哪些名字在不同包里**取值不同**？     → ConstantLedger.conflicts()   ← 本课最值钱的读数
整批文件能不能编出同一份台账？       → digest() / diff_count() / require_reproducible()
```

它**不新增任何能力**，只把已经写在源码里的 `UPPER_CASE = 值` 编成一份可复算的表。

## 2. 快速上手

```python
from smart_research_agent.constant_ledger import build_ledger, check_all, study_lines

ledger = build_ledger()
print(ledger.line())          # 台账：N 个模块 | 常量 C | 可取值 L | 读不出 O | 同名 S | 冲突 X | 摘要 xxxx
print(ledger.value_kind_counts())   # (('literal', …), ('opaque', …))
print(len(ledger.conflicts()))      # "同名不同值" 有几组
print("\n".join(study_lines()))     # 四张表
assert check_all().ok
```

## 3. 六个模块

| 模块 | 作用 |
|------|------|
| `errors.py` | 六个失败族（含"回来的" `ShapeError` 与"继续缺席的" `GradientError`） |
| `types.py` | 口径表：1 种扫描目标 / 2 类取值形态 / 4 类同名关系 / 7 条性质 / 10 条笔记 / 5 条边界 |
| `parse.py` | 只用 `ast` + `ast.literal_eval` 读模块级常量与取值指纹（**不做 import、不执行表达式**） |
| `ledger.py` | 编台账：模块行 + 同名表 + 冲突表 + 可复算口径与摘要 |
| `verify.py` | 七条性质与三类判据（相等 / 上界 / 下界） |
| `study.py` | 四张表：模块 / 同名 / 冲突 / 性质 |

## 4. 什么算"被钉死的常量"

```text
ALPHA = 1                ⇒ 收（模块级的单个 Name 目标）
BETA: int = 2            ⇒ 收（带注解）
GAMMA += 3               ⇒ 收，但取值记 opaque（它没有可比较的右端）
DELTA, EPS = 1, 2        ⇒ **不收**（解包赋值：一个右端对两个名字，取值说不清）
lower = 1 / _PRIVATE = 1 ⇒ **不收**（不是全大写 / 以 _ 开头）
def f(): INSIDE = 1      ⇒ **不收**（不在模块级作用域）
```

"模块级"是一条边界：`class C:` 里的 `KAPPA = 9` 与函数体里的 `INSIDE = 1` 都不算——
它们是那一层的实现细节，不是被钉死在外面的数。

## 5. 两类取值形态

```text
literal   EPSILON = 1e-5 / TOLERANCE = 1e-9 / NAMES = ("a", "b")   ⇒ 用 ast.literal_eval 求出值
opaque    ROOT = Path(__file__).resolve().parents[2]               ⇒ 求不出来，只能记"读不出来"
```

两类必须分开记：**把 opaque 当成"空值"去和别的模块比，会让两个根本无关的常量"看起来一致"**。
`literal_eval` **不执行代码、不 import、不联网**，因此"这个值是什么"有一个离线、确定的答案。

## 6. 四类同名关系（以及为什么"冲突"≠"无法比较"）

```text
unique          只在一个模块里出现
consistent      跨模块，且能比较的那些**字面值全都相同**
conflict        跨模块，且能比较的那些字面值**不止一个**
incomparable    跨模块，但至少一侧是 opaque ⇒ 这一课**拒绝下结论**
```

判定顺序写死为 `unique → incomparable → conflict/consistent`。
把 `incomparable` 并进 `conflict`，会让"我算不出来"被读成"它们对不上"。

## 7. "同名不同值"是一个读数，不是一次错误

本仓库里真的有一批同名不同值（如 `FAMILY_OUTCOMES`、`ABSENT_FAMILY`、`DEFAULT_EPSILON` 之类：
同一类东西在不同包里各取了一个数）。本课**不**写一条"同名必须同值"的性质把它处理掉：

```text
它是一处**读数** ⇒ 整张印出来（conflicts()），并对它做两条判据：
  ③ 报出来的每一组冲突都是**真的**（独立复核源码片段确实不同）
  ④ 同名表里每一项的成员集合与独立重数一致
只有调用点**显式**要求"这些必须一致"时，require_consistent() 才把它变成一次拒绝交付。
```

## 8. 可复算口径

```text
comparable() = (modules, ((模块, ((名字, 行号, 注解, 形态, 取值), ...)), ...))
digest()     = sha256(repr(comparable()))[:16]
```

`comparable()` 里**一个集合、一个字典、一个绝对路径都没有**。
这一课还多防了一条很具体的坑：**集合字面量**。

```text
ALPHABET = {"a", "b"}     ⇒ literal_eval 给出的是一份 set
set 的 repr 顺序随进程变化（PYTHONHASHSEED）⇒ 直接 repr 会让摘要跨进程漂移
canonical_value() 把 set / frozenset 的元素**先排序再拼接**，这堵墙才被拆掉
```

测试里有一条**子进程测试**：两个不同的 `PYTHONHASHSEED` 必须给出同一个摘要。

## 9. 七条性质（三类判据）

```text
相等（==）   ① ledger_covers_all_modules             缺失数 == 0
             ② ledger_is_reproducible                两次构建的差异项数 == 0
             ③ conflicts_are_sound                   不成立的冲突组数 == 0
             ④ shared_constants_are_sound            不成立的同名组成员数 == 0
上界（<=）   ⑤ duplicate_assignments_within_module    重复赋值的模块数 <= 0
下界（>=）   ⑥ constants_are_numerous                 常量总数 >= 500
             ⑦ shared_constants_exist                 跨模块同名组数 >= 1
```

第 ⑥ 条尤其要点明：它是**存在性**断言（"扫描口径没写错"），
写成 `== 某个精确数` 会在仓库合理地变化时误报失败。

## 10. 六个失败族与五条边界

| 族 | 该谁去修 |
|----|----------|
| `ParseError` | 改解析：某份 `.py` 读不到或语法错时，先把它补进扫描范围或修好它 |
| `AssignError` | 改赋值：常量右端读不下来时，把它写成字面量，或显式排除在台账之外 |
| `LedgerBuildError` | 改台账实现：两次构建不一致或同一模块同名列两次时，去掉没有固定的量 / 重复项 |
| `ConflictError` | 改数据或改调用：同名常量取值不一致时，要么统一取值，要么别要求它们一致 |
| `NumericError` | 改数据或改实现：非有限读数与非法上下界都属于"数值不可用" |
| `ParameterError` | 改调用：值形态 / 同名关系 / 性质名 / 上限都是调用点的一次决定 |

五条边界：

```text
1. 不新增任何第三方依赖：解析只用标准库 ast 与 ast.literal_eval。
2. 不改动任何既有模块、既有测试、pyproject 与 docs 下的既有文件——只新增文件。
3. 不做 import、不执行被扫描的表达式：Path(__file__) 这类右端被记成 opaque。
4. 全部读数离线、确定性：只读文件系统与 ast，不联网、不读密钥。
5. 不判断常量取值的好坏：只如实记录"谁把哪个名字钉成了什么值"，不评价该不该是这个值。
```

## 11. 与既有包的关系

```text
day101  course_index      把「字」编成索引
day102  failure_ledger    把「错误名字」编成台账
day103  import_graph      把「模块关系」编成图
day104  symbol_catalog    把「对外承诺」编成清单
day105  constant_ledger   把「被钉死的取值」编成台账   ← 本包
```

五者是同一条纪律的五次应用：**先用 ast 收齐材料，再把"没被固定的顺序"显式排掉，
最后用七条性质（三类判据）把"这份中间物能不能被复算"钉住。**
`constant_ledger` 与 `symbol_catalog` 都读**模块级**作用域，
但一个读 `__all__` 与绑定（"承诺落到哪里"），一个读 `UPPER_CASE = 值` 与取值指纹
（"这个数是多少"）——**问题不同，材料相同**。

## 12. 复现与产出

```bash
python scripts/constant_ledger_demo.py     # 现场算一遍，写进 outputs/constant_ledger_demo.txt
pytest tests/test_constant_ledger.py -q    # 本包 54 个用例
```

演示脚本给出的每个数字都是**现场算的**，没有任何一行是手写的——因此它可被复算。
