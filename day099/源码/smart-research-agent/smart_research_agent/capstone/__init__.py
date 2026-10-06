"""``capstone``：把 Agent / MCP / RAG / 微调 / 评估 / 服务的能力装成一条链（G1-D1 / day099）.

day098 结束时，这个项目里已经有了很多各自成立的子系统：

```text
安全     注入检测 / 内容审核 / 工具权限与审计            （M6 之前）
Agent    规划 / ReAct / function calling / 反思 / 多智能体 （M5~M6）
检索     向量路 + 关键词路 + 融合 + 重排 + 打包 + 接地生成 （M6）
评估     召回 / 精确 / MRR / NDCG；Prompt / 输出 / 红队评估 （M6）
观测     成本追踪 / trace / 告警                          （M6）
微调     LoRA / QLoRA / DPO / 数据管线 / 微调评估           （M8 前后）
服务     FastAPI 端点 / 中间件 / MCP 服务端                 （M7~M8）
```

今天是课程 G1 的**结业项目整合（一）**。它不新增任何第三方依赖、不重写任何子系统，
只做两件事：

```text
1) 装配     把八项能力**真实**装成一条端到端可复算的调用链（assembly.SystemAssembly.run）
2) 渲染     从清单把这条链渲染成「最终版 README + 架构文档」（document.render_*）
            并检查文档覆盖全部 8 项能力
```

## 一、今天最值钱的一句话

> **"装上了"与"接上了"是两件事：前者是 import 得到，后者是一次端到端调用里
> 每个阶段都真的跑过、而且读数能被复算——同一输入跑两次，必须逐位相同。**

这句话有一个立刻可用的推论：一条"接上了"的链，可以用**一条元组比较**证明自己是好的
（:meth:`assembly.SystemRun.is_identical_to`）；而一条"装上了"的链，什么也证明不了。

## 二、八项能力与九个阶段

```text
8 项能力    input_guard / task_planning / hybrid_retrieval / context_packing /
            grounded_generation / offline_evaluation / cost_and_tracing / tool_execution
9 个阶段    guard → plan → retrieve → pack → generate → ground → evaluate → account → trace
```

阶段是"这条链怎么走"，能力是"用户能看见什么"；两者用 ``STAGE_CAPABILITIES`` 串起来
（一个阶段承担的能力写在表里，可以被逐键检查）。

## 三、九个模块

```text
errors.py    七个失败族（**回来的 DocumentError** + **继续缺席的 GradientError**）
types.py     8 项能力 / 9 个阶段 / 12 个候选子包 / 7 条性质（判据分三类）/ 10 条笔记 / 5 条边界
manifest.py  用 importlib 把能力**真的**落到子包上，产出清单与覆盖报告
adapters.py  薄适配层：真实子系统 → 九种统一读数（只带确定性标量）
assembly.py  SystemAssembly + run(question) → SystemRun（九段 + 指标 + 成本 + trace）
document.py  从清单渲染 README 与架构文档，并检查覆盖全部 8 项能力
verify.py    七条性质与三类判据（相等 / 上界 / 下界）
study.py     五张表（能力 / 清单 / 阶段 / 性质 / 文档）
__init__.py  本文件
```

## 四、四条纪律

1. **不重写任何子系统**：本包只新增代码，既有模块、既有测试、pyproject 与
   docs 下的既有文件**一行未改**。
2. **读数必须现场算出**：``study`` 里不存数字；每张表的每行都来自函数调用。
3. **确定性**：用 MockLLM、固定语料与固定编码器；复算口径里**不含**时间戳与 uuid，
   因此同一输入两次运行逐位相同。
4. **不能联网、不能读密钥**：本包的全部读数都在本地可复算。

## 五、与既有包的接缝

- **上游（真实调用的子系统）**：``security``（注入检测 / 内容审核）、
  ``agent.planner``（规划）、``retrieval``（混合检索 / 打包 / 生成）、
  ``evaluation.rag_metrics``（四条指标）、``observability``（成本 / trace）、
  ``tools.calculator``（工具）、``vectorstore`` + ``llm.embedding``（库与编码器）、
  ``llm.mock.MockLLM``（确定性模型）；
- **脚下**：``config`` **没有**新增配置项——预算、问题、语料全是函数参数 / 常量；
- **下游**：下一课（结业项目整合（二））会在这条链之上接部署与服务，
  本课的 ``SystemRun`` 就是它的"第一份可复算的基线"。
"""

from __future__ import annotations

from smart_research_agent.capstone import adapters as _adapters
from smart_research_agent.capstone import assembly as _assembly
from smart_research_agent.capstone import document as _document
from smart_research_agent.capstone import errors as _errors
from smart_research_agent.capstone import manifest as _manifest
from smart_research_agent.capstone import study as _study
from smart_research_agent.capstone import types as _types
from smart_research_agent.capstone import verify as _verify
from smart_research_agent.capstone.errors import CapstoneError

#: 本包的八个功能模块（``__all__`` 由它们的公开名单合并而来——"一个量只写一遍"）。
_MODULES = (
    _errors,
    _types,
    _manifest,
    _adapters,
    _assembly,
    _document,
    _verify,
    _study,
)

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
    raise CapstoneError(
        f"__all__ 里的 {_MISSING} 没有被真正导入："
        "``from capstone import *`` 会静默地少几个名字，而调用方只会看到 NameError。"
    )

_SUBMODULES = {
    "errors",
    "types",
    "manifest",
    "adapters",
    "assembly",
    "document",
    "verify",
    "study",
}
if _SUBMODULES & set(__all__):  # pragma: no cover - 只在有人改名单时触发
    raise CapstoneError(
        f"__all__ 不能包含子模块名：{sorted(_SUBMODULES & set(__all__))}——"
        "否则 `from capstone import assembly` 会拿到函数还是模块取决于导入顺序。"
    )
