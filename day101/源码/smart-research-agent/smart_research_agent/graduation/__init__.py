"""``graduation``：把一次能跑的运行交付成**四份能被别人复算的产物**（G2-D1 / day100）.

day099 把八项能力装成了一条端到端可复算的链，并给了它一条判据——
"同一输入两次运行逐位相同"。今天在这条链之上再走完最后一步：

```text
链跑得对  ≠  交付做得成
```

交付要求的是四份"拿起就能用、放下能被反驳"的产物（:data:`types.DELIVERABLE_ORDER`）：

```text
1. 演示剧本    demo.build_transcript / demo.replay     这一次运行能被重放吗
2. 课程清单    inventory.build_inventory / summary     这门课交付了哪些子包、各有多少东西
3. 能力自评    assessment.assess                        八项能力建到哪一步、证据是什么
4. 后续规划    roadmap.plan                             按真实缺口排，下一步补什么
```

## 一、今天最值钱的一句话

> **"交付"不是把结果贴出来，而是把结果变成**别人能复算**的产物：
> 剧本要能被重放、清单要能被 ``importlib`` 核对、自评要能落到证据上、
> 规划要能从缺口推出——四份都如此，交付才算完成。**

## 二、四份交付物各回答一个独立的问题

```text
剧本   "这一次运行长什么样"      给读者一个可复跑的对照物
清单   "这门课有什么"            一门课的家底，数出来而不是抄出来
自评   "我们做到了哪一步"        8 行带证据的评级，不是一句"已完成"
规划   "还差什么"                从真实缺口推出的下一步，不是许愿
```

四份合起来**不重不漏**：任何一份失真，其余三份都补不上它。

## 三、九个模块

```text
errors.py       七个失败族（**回来的 VersionError** + **继续缺席的 GradientError**）
types.py        4 份交付物 / 4 级自评量程 / 7 条性质（判据分三类）/ 10 条笔记 / 5 条边界
inventory.py    用 pkgutil + importlib 把**全部**一级子包数一遍
demo.py         剧本生成 + 重放（两次逐位比），以及剧本落盘
assessment.py   8 项能力逐项评级：证据 → 量程（一处定义，可被复核）
roadmap.py      从真实缺口推出规划（缺口集合 == 规划项集合）
summary.py      把清单折成一份可核对的课程总结
verify.py       七条性质与三类判据（相等 / 上界 / 下界）
study.py        五张表（交付物 / 清单 / 自评 / 规划 / 性质）
__init__.py     本文件
```

## 四、四条纪律

1. **不重写任何子系统**：本包只新增代码，既有模块、既有测试、pyproject 与
   docs 下的既有文件**一行未改**；它消费 day099 的 ``capstone``（链与清单）。
2. **读数必须现场算出**：``study`` 里不存数字；每张表的每行都来自函数调用。
3. **确定性**：复用 day099 的固定底座（MockLLM + 固定语料 + 固定编码器），
   因此剧本重放两次逐位相同、清单两次相同。
4. **不能联网、不能读密钥**：本包的全部读数都在本地可复算。

## 五、与既有包的接缝

- **上游（真实调用的子系统）**：``capstone.assembly``（端到端链）、
  ``capstone.manifest``（候选子包的解析结论）、``capstone.types``（能力与阶段名单）；
  以及被 ``capstone`` 间接调用的 ``security`` / ``agent`` / ``retrieval`` /
  ``evaluation`` / ``observability`` / ``tools`` / ``vectorstore`` / ``llm``；
- **脚下**：``config`` **没有**新增配置项——一切参数都是函数参数 / 常量；
- **下游**：本包是整条 100 天主线的**最后一个交付层**；
  ``docs/graduation.md`` 与 ``scripts/graduation_demo.py`` 是它的两份外围产物。
"""

from __future__ import annotations

from smart_research_agent.graduation import assessment as _assessment
from smart_research_agent.graduation import demo as _demo
from smart_research_agent.graduation import errors as _errors
from smart_research_agent.graduation import inventory as _inventory
from smart_research_agent.graduation import roadmap as _roadmap
from smart_research_agent.graduation import study as _study
from smart_research_agent.graduation import summary as _summary
from smart_research_agent.graduation import types as _types
from smart_research_agent.graduation import verify as _verify
from smart_research_agent.graduation.errors import GraduationError

#: 本包的九个功能模块（``__all__`` 由它们的公开名单合并而来——"一个量只写一遍"）。
_MODULES = (
    _errors,
    _types,
    _inventory,
    _demo,
    _assessment,
    _roadmap,
    _summary,
    _verify,
    _study,
)

#: 把九个模块 ``__all__`` 里的名字逐个搬进包命名空间。
#:
#: 故意不手写两份名单（一份 import、一份 ``__all__``）：手写一定会分家，
#: 而"某一个名字在 ``__all__`` 里、却没有人真的导入它"这种失败
#: 在报告里长得和"它不存在"一模一样。下面那条导入期不变式代替人来核对这件事。
for _module in _MODULES:
    for _name in _module.__all__:  # pragma: no cover - 纯搬运
        globals()[_name] = getattr(_module, _name)

#: 本包的公开名单：**九个模块各自 ``__all__`` 的并集**，字母序。
__all__ = sorted({name for module in _MODULES for name in module.__all__})

_MISSING = [name for name in __all__ if name not in globals()]
if _MISSING:  # pragma: no cover - 只在有人改名单时触发
    raise GraduationError(
        f"__all__ 里的 {_MISSING} 没有被真正导入："
        "``from graduation import *`` 会静默地少几个名字，而调用方只会看到 NameError。"
    )

_SUBMODULES = {
    "errors",
    "types",
    "inventory",
    "demo",
    "assessment",
    "roadmap",
    "summary",
    "verify",
    "study",
}
if _SUBMODULES & set(__all__):  # pragma: no cover - 只在有人改名单时触发
    raise GraduationError(
        f"__all__ 不能包含子模块名：{sorted(_SUBMODULES & set(__all__))}——"
        "否则 `from graduation import demo` 会拿到函数还是模块取决于导入顺序。"
    )
