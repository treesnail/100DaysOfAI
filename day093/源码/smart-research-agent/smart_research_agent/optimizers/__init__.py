"""``optimizers``：从"往哪走"到"走多远"（M8-D3 / day092）.

day090 结束时，那句话是被明写的：**"往哪走"已经确定（梯度），剩下的只有"走多远"**。
今天把 day074 的三个更新原语补成一整套**训练期优化器**：新增 Nesterov / RMSProp / AdamW，
把"衰减 + 裁剪 + 调度 + 更新规则"接成一次真正的更新，并把六个变体放在同一个
目标函数上比出读数。

```text
六个更新规则  →  三种衰减  →  一次组装好的更新  →  六个变体的收敛对比
      ↘              ↘                 ↘
   day074 复用三个   耦合 / 解耦       裁剪 + 调度 + 规则（顺序写死）
```

## 一、今天最值钱的一句话

> **一步 = 学习率 × 更新规则；方向由 day090 给，本课只决定这个标量作用在什么上。**

把两者混着调，是这一课要拦住的第一种错误：loss 不降时先怀疑"学习率"，
但真正错的可能是"更新规则用在了错误的地方"（例如把解耦衰减写成了耦合）。

## 二、七条性质（判据分三类）

```text
与 day074 对账①  sgd / momentum / adam 逐步逐位一致
与 day074 对账②  裁剪返回同一个缩放系数（裁剪的数学只有一份）
与 day074 对账③  热身+余弦逐点一致（热身的计步口径只有一份）
新规则的性质①    Nesterov 不是普通动量（**下界**判据：它们必须不同）
新规则的性质②    AdamW 不是 Adam+L2（**下界**判据：解耦不是摆设）
新规则的性质③    RMSProp 把相差 1e4 的梯度归一到几乎相同的步长
几何性质        整体范数裁剪只改长度、不改方向
```

判据有三类：**相等**（逐位）、**不超过上界**（相对误差）、**不低于下界**
（"它们确实不同"）。第三类是本课新增的——没有它，把两个规则写成同一个，
报告会全绿（见 :mod:`optimizers.verify` 的说明）。

## 三、八个模块

```text
errors.py    五个失败族（GradientError **再次缺席**：梯度是上一步交来的输入）
types.py     六个规则 / 三种衰减 / 七条性质 / 十条笔记 / 五条边界 / torch 对照表
rules.py     两个新增更新规则（Nesterov / RMSProp）+ 两种衰减 + 裁剪接线
optimizer.py 训练期优化器：取 lr → 裁剪 → 更新规则 → 衰减（顺序写死）
advisor.py   把六个规则落到本项目真实的六类训练场景上（选型 + 起点超参 + 为什么）
compare.py   三个目标函数上六个变体的收敛对比（复用 day074 的 minimize）
verify.py    七条性质与三类判据（含三条跨天对账）
study.py     六张表（规则 / 一步 / 裁剪 / 调度 / 对比 / 性质）
```

## 四、四条纪律（与前面各天逐字相同）

1. **一个量只写一遍**：clip 的数学只在 day074；热身的计步只在 day074；
   本课只接"什么时候调用它"。
2. **读数必须现场算出**：``study`` 里不存数字；每一张表的每一行都来自函数调用。
3. **跨天对账必须调用别人的包**：三条对账分别调 ``math_foundations``（day073/074）——
   自证是不成立的。
4. **缺席要可断言**：``RETURNED_FAMILY`` / ``ABSENT_FAMILY`` 都是常量，
   因此"GradientError 今天再次缺席"是一件可以被测试逐字钉住的事实。

## 五、与既有包的接缝（M8-D2 → M8-D3 的过渡）

- **上游（本包调用的真实实现）**：
  ``math_foundations``（day073/074：``optim`` 的三个优化器 / 四种调度 / 两个裁剪原语、
  ``calculus.gradient``、``types.validate_vector``）、
  ``transformer_core``（day075：四族错误）、
  ``backprop``（day090：本课用的梯度就是它交出来的那一批）；
- **传承**：day074 给了"一条更新公式 + 四种调度 + 两个裁剪原语"，
  day090 给了"可信的梯度"——**今天把它们接成一次真正的训练期更新**，
  并补上 day074 刻意留白的三个变体；
- **脚下**：``config`` **没有**新增配置项——学习率、衰减、裁剪、调度都是函数参数；
- **下游**：day093（CNN）会第一次需要"参数不是一串数而是若干卷积核"——
  本课把参数压平/还原的契约（day074 的 ``flatten_matrices``）继续沿用。
"""

from __future__ import annotations

from smart_research_agent.optimizers import advisor as _advisor
from smart_research_agent.optimizers import compare as _compare
from smart_research_agent.optimizers import errors as _errors
from smart_research_agent.optimizers import optimizer as _optimizer
from smart_research_agent.optimizers import rules as _rules
from smart_research_agent.optimizers import study as _study
from smart_research_agent.optimizers import types as _types
from smart_research_agent.optimizers import verify as _verify
from smart_research_agent.optimizers.errors import OptimizerError

#: 本包的八个功能模块（``__all__`` 由它们的公开名单合并而来——"一个量只写一遍"）。
_MODULES = (_errors, _types, _rules, _optimizer, _advisor, _compare, _verify, _study)

#: 把八个模块 ``__all__`` 里的名字逐个搬进包命名空间。
#:
#: 故意不手写两份名单（一份 import、一份 ``__all__``）：手写一定会分家，
#: 而"某一个名字在 ``__all__`` 里、却没有人真的导入它"这种失败
#: 在报告里长得和"它不存在"一模一样。下面那条导入期不变式代替人来核对这件事。
for _module in _MODULES:
    for _name in _module.__all__:  # pragma: no cover - 纯搬运
        globals()[_name] = getattr(_module, _name)

#: 本包的公开名单：**八个模块各自 ``__all__`` 的并集**，字母序。
__all__ = sorted({name for module in _MODULES for name in module.__all__})

_MISSING = [name for name in __all__ if name not in globals()]
if _MISSING:  # pragma: no cover - 只在有人改名单时触发
    raise OptimizerError(
        f"__all__ 里的 {_MISSING} 没有被真正导入："
        "``from optimizers import *`` 会静默地少几个名字，而调用方只会看到 NameError。"
    )

_SUBMODULES = {"errors", "types", "rules", "optimizer", "advisor", "compare", "verify", "study"}
if _SUBMODULES & set(__all__):  # pragma: no cover - 只在有人改名单时触发
    raise OptimizerError(
        f"__all__ 不能包含子模块名：{sorted(_SUBMODULES & set(__all__))}——"
        "否则 `from optimizers import compare` 会拿到函数还是模块取决于导入顺序。"
    )
