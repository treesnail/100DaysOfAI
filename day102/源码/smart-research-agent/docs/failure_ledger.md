# 失败族台账（failure_ledger）—— day102 手册

> 本手册是 `smart_research_agent.failure_ledger` 的使用说明与设计说明。
> 它由 day102 交付，对应当天的教程 `day102/教程/教程.md`。

## 1. 这份台账回答什么问题

100 天里，几乎每个子包都写了一份 `errors.py`。今天把这些散落的失败族收成**一张表**：

```text
这个失败该谁去修？    → 每一族都挂着一句"该谁去修"（FAMILY_OUTCOMES）
这一族继承自谁？      → 每一族都有一组归属好的基类（BaseRef）
哪个包没有失败族？    → packages_without_errors() 会给出一份清单
```

它**不新增任何能力**，只把已经写好、但从来没被横向放在一起看过的名字，编成一份可复算的台账。

## 2. 快速上手

```python
from smart_research_agent.failure_ledger import build_ledger, check_all, study_lines

ledger = build_ledger()          # 扫描全部 errors.py 并编成台账
print(ledger.line())             # 台账：N 份模块 | M 个族 | ... | 摘要 xxxx
print("\n".join(study_lines()))  # 四张表
report = check_all()             # 七条性质
assert report.ok
```

## 3. 七个模块

| 模块 | 作用 |
|------|------|
| `errors.py` | 六个失败族（含"回来的" `ShapeError` 与"继续缺席的" `GradientError`） |
| `types.py` | 口径表：1 种扫描目标 / 4 类基类归属 / 7 条性质 / 10 条笔记 / 5 条边界 |
| `scan.py` | 只用 `ast` 读源码，收原始族与 import 别名表（**不做 import**） |
| `resolve.py` | 把基类名归属到 local / builtin / imported / unresolved |
| `ledger.py` | 编台账：定身份（root / sub）+ 算继承深度 + 复算摘要 |
| `verify.py` | 七条性质与三类判据（相等 / 上界 / 下界） |
| `study.py` | 四张表：模块 / 族 / 归属 / 性质 |

## 4. 四类基类归属

```text
local       基类名就在同一个 errors.py 里（class ShapeError(CoreShapeError, StackError) 的 StackError）
builtin     Python 内置异常名（ValueError / Exception / RuntimeError ……）
imported    基类名来自另一个 errors.py，以别名出现（ShapeError as CoreShapeError）
unresolved  以上三类都不是 ⇒ 当场点名，绝不静默地当成类名
```

判定顺序写死为 **local → builtin → imported**。顺序一旦调换，"某个子包自定义了 `ValueError`"
或"本模块里的族与导进来的同名"就会被误判。

## 5. 两种身份与一个不变式

```text
root   没有 local 基类 ⇒ 它是这一族在**本模块内**的根
sub    有 local 基类   ⇒ 它是一个派生族
```

不变式：**每一份 `errors.py` 恰好有一个根**。两个根意味着读者要先猜"我该 except 哪一个"；
零个根只可能来自源码里的继承环（Python 不允许，但静态扫描能把源码里的环看见）。

## 6. 继承深度的口径

```text
depth(root) = 0
depth(sub)  = 1 + max(depth(local 父族))
```

深度只在**同一模块内部**沿 local 基类计算；跨包的 `imported` 基类不参与深度
（它指向另一份 `errors.py`，那是 day075 以来的一份**契约**，不是本模块的层数）。

## 7. 七条性质

| # | 性质 | 判据 | 失败意味着 |
|---|------|------|-----------|
| ① | `ledger_covers_all_error_modules` | 相等 | 某份 `errors.py` 没进台账 |
| ② | `qualified_names_are_unique` | 相等 | 两个族共用了一个 `包.名` |
| ③ | `ledger_is_reproducible` | 相等 | 台账里混进了未固定的迭代序 |
| ④ | `every_module_has_exactly_one_root` | 相等 | 某模块有 0 个或 2 个根 |
| ⑤ | `base_references_resolve` | 上界 | 有基类名无法归属 |
| ⑥ | `every_family_has_a_parent` | 下界 | 某族一个基类都没有 |
| ⑦ | `inheritance_depth_is_positive` | 下界 | 本模块内部根本没有继承链 |

方向不能写反：⑥⑦ 是"越多越好"，只能从下面兜住；⑤ 是"越少越好"，兜的是上限。
把方向写反，性质会在仓库**变好**的时候误报失败，然后被人关掉。

## 8. 四张表

```text
① 模块表   每份 errors.py：包名 / 族数 / 根族
② 族表     每个族：包.名 / 身份 / 基类
③ 归属表   基类按四类归属的计数
④ 性质表   7 条性质：判据类别 / 读数 / 结论
```

每一行都带两个数（读数 + 参照）——一行只写"通过"的表是没法反驳的。

## 9. 复算口径

```text
comparable() = (modules, ((包.名, 身份, ((raw, kind, target), ...)), ...))
digest()     = sha256(repr(comparable()))[:16]
```

`comparable()` 里没有任何集合或字典：因此 `repr` 稳定，摘要才有意义。
**两次构建摘要相同 ⇔ 逐位相同**，这条 `==` 就是"台帐可复算"的全部内容。

## 10. 四个"拒绝交付"的路

```text
require_reproducible(first, second)  两份台账不同 ⇒ LedgerBuildError
require_coverage(ledger, expected=…) 漏了 errors.py ⇒ CoverageError
require_resolved(module, refs)       有未解析基类 ⇒ ResolveError
require_ok(report)                   性质没全过 ⇒ ScanError
```

它们是 `search()` / `check_*()` 之外那条"拒绝坏输入"的路，与 day101 的
`require_reproducible` / `require_hits` 是同一套习惯。

## 11. 与既有包的接缝

```text
上游   文件系统（smart_research_agent/*/errors.py）——只读，不 import
脚下   config 没有新增配置项：扫描目标、白名单与四类归属都是常量
下游   docs/failure_ledger.md（本文件）与 scripts/failure_ledger_demo.py
```

## 12. 边界（这一课明确不承诺的事）

1. 不新增任何第三方依赖：扫描只用标准库 `ast`。
2. 不改动任何既有模块、既有测试、`pyproject` 与 `docs/` 下的既有文件——只新增文件。
3. 不做 import：读源码文本并解析 AST，因此语法错误会被当场点名。
4. 全部读数离线、确定性：只读文件系统与 `ast`，不联网、不读密钥。
5. 不判断设计好坏：只如实记录"谁继承了谁"，不评价"这个继承关系该不该存在"。
