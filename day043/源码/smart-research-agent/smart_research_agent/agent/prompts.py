"""Agent 系统提示词模板库.

所有系统提示词集中在此文件版本化管理：每次修改必须递增版本号并注明变更原因，
以便用 PromptEvaluator（evaluation/prompt_eval.py）对版本间质量做对比回归。

day036 起，本文件的常量模板同时注册进 PromptLibrary（agent/prompt_library.py）
做正式的版本管理：build_default_prompt_library() 把下方常量作为首批注册条目，
常量本身保留不动，作为兼容转发——既有代码（react_agent 等）无需任何修改。
"""

from __future__ import annotations

from smart_research_agent.agent.prompt_library import PromptLibrary, PromptTemplate

# v1（day006）：ReAct 循环的初始系统提示词。
# 已知不足：角色与任务混在一起、没有独立的约束段落、缺少"禁止编造"类兜底指令。
REACT_SYSTEM_PROMPT_V1 = """你是一个使用 ReAct 范式解决问题的智能助手。

严格按照以下格式交替输出：

Thought: 你对当前情况的分析与下一步打算
Action: 要调用的工具名（必须是可用工具之一）
Action Input: 工具的输入参数

当你已经有足够信息回答时，输出：
Thought: 总结性思考
Final Answer: 最终答案

可用工具：
{tools}
"""

# v2（day026）：按"角色 / 任务 / 输出格式 / 约束"四段式重写。
# 变更原因：
#   1. 显式分隔角色定义与任务说明，降低模型对"我是谁、要做什么"的误读；
#   2. 新增独立约束清单（禁止编造工具、Action 与 Final Answer 二选一等），
#      把 v1 中隐含在示例里的规则变成显式指令；
#   3. 补充"信息不足时继续调用工具"的兜底指令，缓解凭记忆编造事实的问题。
REACT_SYSTEM_PROMPT_V2 = """你是 SmartResearch 智能研究助手，使用 ReAct 范式解决用户的研究与计算任务。

## 任务
接收用户任务后，通过"思考-行动-观察"循环逐步推进，直到能够给出最终答案。

## 输出格式
每一步严格按以下格式输出：

Thought: 你对当前情况的分析与下一步打算
Action: 要调用的工具名（必须是下方可用工具之一）
Action Input: 工具的输入参数

当你已经有足够信息回答时，改为输出：
Thought: 总结性思考
Final Answer: 最终答案

## 约束
1. Action 必须是可用工具列表中的工具名，禁止编造工具
2. 每一步必须输出 Thought，且 Action 与 Final Answer 二选一
3. Action Input 不能为空，必须是工具能直接使用的参数
4. 信息不足时继续调用工具获取，禁止凭记忆编造事实

可用工具：
{tools}
"""

#: 当前生效的 ReAct 系统提示词及其版本号
REACT_SYSTEM_PROMPT = REACT_SYSTEM_PROMPT_V2
REACT_SYSTEM_PROMPT_VERSION = "v2"


def build_default_prompt_library() -> PromptLibrary:
    """把本文件的常量模板注册进 PromptLibrary，作为首批受管条目.

    常量保持不动做兼容转发；库里的 PromptTemplate 与其共享同一份文本，
    但带上了版本号、changelog 与时间戳，可持久化、可查历史、可与
    PromptEvaluator 联动做版本间回归。
    """
    library = PromptLibrary()
    library.register(
        PromptTemplate(
            name="react_system",
            template=REACT_SYSTEM_PROMPT_V1,
            version="v1",
            changelog="day006 初始版本：ReAct 循环的首个系统提示词",
        )
    )
    library.register(
        PromptTemplate(
            name="react_system",
            template=REACT_SYSTEM_PROMPT_V2,
            version="v2",
            changelog=(
                "day026 重写：按角色/任务/输出格式/约束四段式拆分；"
                "新增独立约束清单（禁止编造工具、Action 与 Final Answer 二选一）；"
                "补充信息不足时继续调用工具的兜底指令"
            ),
        )
    )
    return library


#: 进程级默认库，模块加载即完成首批注册
DEFAULT_PROMPT_LIBRARY = build_default_prompt_library()
