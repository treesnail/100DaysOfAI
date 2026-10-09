# 模块依赖图（import_graph）—— day103 手册

> 本手册是 `smart_research_agent.import_graph` 的使用说明与设计说明。
> 它由 day103 交付，对应当天的教程 `day103/教程/教程.md`。

## 1. 这份图回答什么问题

100 天里，450 多个 `.py` 互相 `import`。今天把这些关系收成**一张有向图**：

```text
我依赖谁？           → closure(node, "down")
动了我，谁受影响？    → closure(node, "up")
哪两个包互相咬住？    → rollup_to_packages() 之后看环
线性顺序是什么？      → topological_order()（**它是分量序，不是模块序**）
```

它**不新增任何能力**，只把已经写在 import 语句里的关系，编成一份可复算的图。

## 2. 快速上手

```python
from smart_research_agent.import_graph import build_graph, check_all, study_lines

graph = build_graph()
print(graph.line())              # 图：N 个节点 | M 个包 | K 条边 | 4 个环 | 摘要 xxxx
print(len(graph.cycles()))       # 4
print(graph.topological_order()) # 凝缩后的线性序（长度 = 分量数）
print("\n".join(study_lines()))  # 四张表
assert check_all().ok
```

## 3. 六个模块

| 模块 | 作用 |
|------|------|
| `errors.py` | 六个失败族（含"回来的" `AssemblyError` 与"继续缺席的" `GradientError`） |
| `types.py` | 口径表：1 种扫描目标 / 2 类边 / 2 个方向 / 7 条性质 / 10 条笔记 / 5 条边界 |
| `parse.py` | 只用 `ast` 读节点与 import 目标（**不做 import**） |
| `graph.py` | 建图 + 强连通分量（Kosaraju）+ 凝缩 + 线性序（Kahn）+ 闭包 |
| `verify.py` | 七条性质与三类判据（相等 / 上界 / 下界） |
| `study.py` | 四张表：模块 / 环 / 边 / 性质 |

## 4. 两类边

```text
absolute   from smart_research_agent.tools.base import BaseTool   ⇒ 目标写全了
relative   from .base import BaseTool                            ⇒ 要按 level 减层算
```

相对导入的规则是 `level - 1` 层点：

```text
在 smart_research_agent/foo/bar.py 里：
  from .base import X       level=1 ⇒ smart_research_agent.foo.base
  from ..other import Y     level=2 ⇒ smart_research_agent.other
```

**这是本课唯一"会算错而且不会报错"的地方**：少减一层点，边就指到隔壁模块去了，
而图照样画得出来、照样能排序——只有性质 ⑤（未解析目标数 <= 0）会把它抓住。

## 5. 一个真实的环

```text
api/app.py                  from smart_research_agent.tools.image_analysis import ImageAnalysisTool  ← 顶层
tools/image_analysis.py     （函数体内）from smart_research_agent.api.app import default_llm        ← 延迟
```

两条边合起来是一个环。它被**函数体内的延迟 import** 挡在了运行期之外，
但静态依赖图看得见它——这正是"读 import **语句**、不执行 import"的价值。
本仓库当前读数：模块级 **4 个环**，折成包级 **1 个环**（一个 20 多个包的巨型分量）。

## 6. 强连通分量与凝缩

```text
sccs()                 全部强连通分量（Kosaraju：两遍迭代 DFS）
cycles()               成员数 > 1 的那些（真正的环）
condensation_edges()   分量键 → 分量键（去掉自环、排序去重）
topological_order()    凝缩图的线性序（Kahn + 最小堆，并列取字典序最小）
```

**有环的图没有线性序**：`topological_order()` 的长度是**分量数**，不是模块数。
谁把这两个数弄混，报告里的顺序就会互相矛盾。`require_acyclic()` 是"拒绝交付"的那条路：
图里真有环时抛 `CycleError`，并在消息里把环点名。

## 7. 两个方向

```text
closure(x, "down")   从 x 出发沿边能到谁（我依赖谁）
closure(x, "up")     谁能沿边到 x（动了我，谁受影响）
```

两个方向极不对称。不写方向就取默认值，会让报告里两个完全不同的数长得一模一样。

## 8. 七条性质

| # | 性质 | 判据 | 失败意味着 |
|---|------|------|-----------|
| ① | `graph_covers_all_modules` | 相等 | 某个 `.py` 没进图 |
| ② | `graph_is_reproducible` | 相等 | 图里混进了未固定的迭代序 |
| ③ | `topological_order_is_valid` | 相等 | 线性序里出现向后指的边 |
| ④ | `cycles_are_sound` | 相等 | 报了一个假环 |
| ⑤ | `import_targets_resolve` | 上界 | 有 import 目标对不上（多半是相对导入算错） |
| ⑥ | `closures_include_self` | 下界 | 闭包的起点算错了 |
| ⑦ | `dependency_depth_is_positive` | 下界 | 图退化成了孤点（一条边都没有） |

方向不能写反：⑥⑦ 是"越多越好"，只能从下面兜住；⑤ 是"越少越好"，兜的是上限。

## 9. 一条纪律：并列必须按名字兜顺序

本课真的撞到了这堵墙：`resolve_candidates` 最初写成
`sorted(candidates, key=len, reverse=True)`——长度相同的候选会回到**输入顺序**，
而输入来自一个 `set`，它的迭代序随进程变化。于是
`from . import a, b`（`a` 与 `b` 同长且都是模块）会让边在两次运行之间**指向不同的目标**。

修法是把排序键写成 `(-长度, 名字)`。测试里有一条**子进程测试**
（两个不同的 `PYTHONHASHSEED` 必须给出同一个摘要）专门钉住它。

## 10. 复算口径

```text
comparable() = (nodes, packages, ((源, 目标, 种类), ...))
digest()     = sha256(repr(comparable()))[:16]
```

`comparable()` 里没有任何集合或字典：因此 `repr` 稳定，摘要才有意义。
**两次构建摘要相同 ⇔ 逐位相同。**

## 11. 与既有包的接缝

```text
上游   文件系统（smart_research_agent/**/*.py）——只读，不 import
脚下   config 没有新增配置项：扫描目标、两类边与两个方向都是常量
下游   docs/import_graph.md（本文件）与 scripts/import_graph_demo.py
```

## 12. 边界（这一课明确不承诺的事）

1. 不新增任何第三方依赖：解析只用标准库 `ast`。
2. 不改动任何既有模块、既有测试、`pyproject` 与 `docs/` 下的既有文件——只新增文件。
3. 不做 import：读的是 import **语句**，不是真的导入（所以延迟 import 也是一条边）。
4. 全部读数离线、确定性：只读文件系统与 `ast`，不联网、不读密钥。
5. 不判断分层好坏：只如实记录"谁 import 了谁"，不评价"这个环该不该存在"。
