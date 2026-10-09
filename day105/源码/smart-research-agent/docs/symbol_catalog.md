# 对外承诺清单（symbol_catalog）—— day104 手册

> 本手册是 `smart_research_agent.symbol_catalog` 的使用说明与设计说明。
> 它由 day104 交付，对应当天的教程 `day104/教程/教程.md`。

## 1. 这份清单回答什么问题

100 天里，几百份 `.py` 各自写了 `__all__`。今天把这些**对外承诺**收成一份**可核对的清单**：

```text
这个模块承诺了什么？              → SymbolCatalog.entry_of(module).exports()
这些承诺落到了哪里？              → entry_of(module).bindings()
哪个名字被很多模块同时承诺？       → SymbolCatalog.shared_names()
哪个承诺根本没有落点？            → SymbolCatalog.phantom_pairs()
整批文件能不能编出同一份清单？     → digest() / diff_count() / require_reproducible()
```

它**不新增任何能力**，只把已经写在源码里的 `__all__` 与模块级绑定，编成一份可复算的表。

## 2. 快速上手

```python
from smart_research_agent.symbol_catalog import build_catalog, check_all, study_lines

catalog = build_catalog()
print(catalog.line())            # 清单：N 个模块 | M 份字面量承诺 | 承诺 P | 重名 S | 幽灵 G | 摘要 xxxx
print(catalog.source_counts())   # (('declared', …), ('computed', …), ('implicit', …))
print(catalog.phantom_pairs())   # 承诺了却找不到落点的名字
print("\n".join(study_lines()))  # 四张表
assert check_all().ok
```

## 3. 六个模块

| 模块 | 作用 |
|------|------|
| `errors.py` | 六个失败族（含"回来的" `CoverageError` 与"继续缺席的" `GradientError`） |
| `types.py` | 口径表：1 种扫描目标 / 3 种承诺来源 / 4 类绑定 / 7 条性质 / 10 条笔记 / 5 条边界 |
| `parse.py` | 只用 `ast` 读 `__all__` 与模块级绑定（**不做 import**） |
| `catalog.py` | 编清单：模块行 + 重名表 + 幽灵表 + 可复算口径与摘要 |
| `verify.py` | 七条性质与三类判据（相等 / 上界 / 下界） |
| `study.py` | 四张表：模块 / 幽灵 / 重名 / 性质 |

## 4. 三种承诺来源

```text
declared   __all__ = ["A", "B"]        ⇒ 一份可以逐字核对的清单
computed   __all__ = sorted({...})      ⇒ 它也算承诺，但这一课读不出来
implicit   根本没有写过 __all__         ⇒ 只能退而取顶层公开 def / class
```

**没有 `__all__` 不等于没有承诺**，只等于"这份承诺没有被写下来"。
三种来源必须分开记，因为它们回答的不是同一个问题。

## 5. 四类绑定与"幽灵导出"

判定顺序写死，不能换：

```text
① defined     本模块里有 def / class         ⇒ 落点最硬
② assigned    本模块里被赋过值（模块级）      ⇒ 落点是一个变量
③ imported    是本模块某条 import 的别名      ⇒ 落点在外面
④ unresolved  以上三类都不是                 ⇒ **幽灵导出**，当场点名
```

这里的"模块级"是一条边界：`class Foo:` 里面的 `def bar` 不算本模块的落点，
`def baz(): import json` 里的 import 也不算本模块的导入；而写在 `try:` 里的
`from x import Y` **算**（它就在模块级作用域里）。

## 6. 一个真实的幽灵导出

本仓库有一处**声明了却找不到落点**的承诺：

```text
smart_research_agent/transformer_stack/verify.py 的 __all__ 里写着 "STAGE_ITEMS"
而这个名字在该模块里既没有 def / class，也没有赋值，也没有 import
```

谁写 `from smart_research_agent.transformer_stack.verify import *`，谁就会在那里拿到一个
`AttributeError`。本课把它叫**幽灵导出**，并像 day103 对待"环"一样，把**它印出来**
（`phantom_pairs()`），而不是让一条"必须为 0"的性质把它遮掉。

性质 ③ 只对这些**报出来的**幽灵做一件更硬的事：**独立复核它们是不是真的**——
把该模块里每一段 `__all__` 的源码挖掉之后，这个名字应当**一次都不出现**。

## 7. 重名表：跨包契约的入口

一个名字被几十个模块同时导出（如 `ShapeError` / `NumericError`）：这既是
"同一件事在很多层各写了一遍"的证据，也是跨包契约的入口——
`except ShapeError` 之所以能一路兜住后面几十层，靠的就是这份重名。
清单按"出现模块数降序 × 名字升序"把它整张印出来。

## 8. 可复算口径

```text
comparable() = (modules, ((模块, 来源, 声明, 定义, 赋值, 导入), ...))
digest()     = sha256(repr(comparable()))[:16]
```

`comparable()` 里**一个集合、一个字典、一个绝对路径都没有**。
为什么这条如此重要？因为 Python 的 `dict` **保留插入顺序**，而插入顺序取决于"先遍历了谁"：
只要出现一个集合，`repr` 就会随构建顺序漂移，"两次构建逐位相同"这条性质也就失去意义。
因此本包在任何"集合 → 序列"的地方都显式排序。

## 9. 七条性质（三类判据）

```text
相等（==）   ① catalog_covers_all_modules          缺失数 == 0
             ② catalog_is_reproducible             两次构建的差异项数 == 0
             ③ phantom_exports_are_sound           不成立的幽灵数 == 0
             ④ shared_names_are_sound              不成立的重名条目数 == 0
上界（<=）   ⑤ duplicate_declarations_within_module 重复声明的模块数 <= 0
下界（>=）   ⑥ every_declaring_module_promises       最小承诺数 >= 1
             ⑦ export_names_are_reused              跨模块重名数 >= 1
```

方向要和这个量的方向一致：覆盖数、差异项数、假幽灵数是**相等**；
"重复声明不超过 0"是**上界**；"每个承诺的模块至少承诺一个""至少有一条重名"是**下界**。
方向错了，性质会在系统**变好**的时候误报失败，然后被人关掉——而一条被关掉的性质，等于不存在。

## 10. 六个失败族与五条边界

| 族 | 该谁去修 |
|----|----------|
| `ParseError` | 改解析：某份 `.py` 读不到或语法错时，先把它补进扫描范围或修好它 |
| `DeclareError` | 改声明：字面量 `__all__` 里混进了非字符串时，把它改成一份名字表 |
| `CatalogBuildError` | 改清单实现：两次构建不一致或同一模块重名时，去掉没有固定的量 / 重复项 |
| `BindingError` | 改源码或改声明：承诺没有落点时，要么定义出来，要么从 `__all__` 里删掉 |
| `NumericError` | 改数据或改实现：非有限读数与非法上下界都属于"数值不可用" |
| `ParameterError` | 改调用：承诺来源 / 绑定类别 / 性质名 / 上限都是调用点的一次决定 |

五条边界：

```text
1. 不新增任何第三方依赖：解析只用标准库 ast。
2. 不改动任何既有模块、既有测试、pyproject 与 docs 下的既有文件——只新增文件。
3. 不做 import：读的是 __all__ 与 import **语句**（因此函数体里的延迟 import 也算一次绑定）。
4. 全部读数离线、确定性：只读文件系统与 ast，不联网、不读密钥。
5. 不判断承诺的好坏：只如实记录"谁承诺了什么、落到哪里"，不评价"该不该导出"。
```

## 11. 与既有包的关系

```text
day101  course_index      把「字」编成索引
day102  failure_ledger    把「错误名字」编成台账
day103  import_graph      把「模块关系」编成图
day104  symbol_catalog    把「对外承诺」编成清单   ← 本包
```

四者是同一条纪律的四次应用：**先用 ast 收齐材料，再把"没被固定的顺序"显式排掉，
最后用七条性质（三类判据）把"这份中间物能不能被复算"钉住。**
`symbol_catalog` 与 `import_graph` 都用 `ast` 读同一批 `.py`，
但一个读 `__all__` 与**模块级**绑定，一个读**任意位置**的 import 语句——**问题不同，材料相同**。

## 12. 复现与产出

```bash
python scripts/symbol_catalog_demo.py     # 现场算一遍，写进 outputs/symbol_catalog_demo.txt
pytest tests/test_symbol_catalog.py -q    # 本包 60 个用例
```

演示脚本给出的每个数字都是**现场算的**，没有任何一行是手写的——因此它可被复算。
