"""LangGraph 状态定义：ReAct 流程在图中流转的数据载体."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict):
    """ReAct 状态图的全局状态.

    普通字段（task、thought 等）采用"覆盖式"更新：节点返回新值即替换旧值；
    history 通过 Annotated[..., operator.add] 声明了 reducer，
    节点返回的列表会**追加**到已有轨迹之后，而不是覆盖。
    """

    task: str  # 用户任务，整个运行期间不变
    thought: str  # 当前步的思考
    action: str | None  # 当前步决定调用的工具名
    action_input: str | None  # 工具入参
    observation: str  # 最近一次工具执行结果
    steps: int  # 已执行的工具调用次数
    final_answer: str | None  # 最终答案（done=True 时有值）
    done: bool  # 是否已得出最终答案
    history: Annotated[list[dict[str, Any]], operator.add]  # 已完成的步骤轨迹
