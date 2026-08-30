"""ReAct 流程的 LangGraph 状态图实现.

把 day006 的 while 循环重构为显式状态图：
think 节点调用 LLM 产出 Thought/Action，act 节点执行工具写回 Observation，
条件边负责"继续循环还是终止"的路由判断。
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from smart_research_agent.agent.parser import parse_react_output
from smart_research_agent.agent.react_agent import SYSTEM_PROMPT
from smart_research_agent.graph.state import AgentState
from smart_research_agent.llm.base import BaseLLM, Message
from smart_research_agent.tools.registry import ToolRegistry
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)


def initial_state(task: str) -> AgentState:
    """构造一次新任务的初始状态."""
    return AgentState(
        task=task,
        thought="",
        action=None,
        action_input=None,
        observation="",
        steps=0,
        final_answer=None,
        done=False,
        history=[],
    )


def build_react_graph(
    llm: BaseLLM,
    tool_registry: ToolRegistry,
    max_steps: int = 5,
    interrupt_before: list[str] | None = None,
) -> Any:
    """构建 ReAct 状态图.

    Args:
        llm: LLM 抽象实例（生产用 OpenAICompatibleLLM，测试用 MockLLM）。
        tool_registry: 工具注册表。
        max_steps: 最大工具执行次数，防止无限循环。
        interrupt_before: 在哪些节点执行前暂停（断点调试），如 ["act"]。

    Returns:
        编译完成、带内存检查点的 CompiledStateGraph。
    """

    def think(state: AgentState) -> dict:
        """思考节点：调用 LLM，把输出解析为 Thought/Action 或 Final Answer."""
        messages = [
            Message(role="system", content=SYSTEM_PROMPT.format(tools=tool_registry.describe())),
            Message(role="user", content=f"任务: {state['task']}"),
        ]
        for item in state["history"]:
            messages.append(
                Message(
                    role="assistant",
                    content=(
                        f"Thought: {item['thought']}\n"
                        f"Action: {item['action']}\n"
                        f"Action Input: {item['action_input']}"
                    ),
                )
            )
            messages.append(Message(role="user", content=f"Observation: {item['observation']}"))
        raw = llm.chat(messages)
        logger.info("think 节点 LLM 输出:\n%s", raw)
        step = parse_react_output(raw)
        update: dict[str, Any] = {"thought": step.thought}
        if step.final_answer is not None:
            update["final_answer"] = step.final_answer
            update["done"] = True
        else:
            update["action"] = step.action
            update["action_input"] = step.action_input
        return update

    def act(state: AgentState) -> dict:
        """行动节点：执行工具，把 Observation 与轨迹写回状态."""
        tool = tool_registry.get(state["action"] or "")
        if tool is None:
            observation = f"错误：不存在名为 {state['action']} 的工具"
        else:
            required = tool.parameters.get("required") or []
            kwargs: dict[str, Any] = (
                {required[0]: state["action_input"] or ""} if required else {}
            )
            try:
                observation = tool.execute(**kwargs)
            except Exception as exc:  # noqa: BLE001
                logger.warning("工具执行异常: %s", exc)
                observation = f"工具执行失败: {exc}"
        logger.info("act 节点观测: %s", observation)
        return {
            "observation": observation,
            "steps": state["steps"] + 1,
            "history": [
                {
                    "thought": state["thought"],
                    "action": state["action"],
                    "action_input": state["action_input"],
                    "observation": observation,
                }
            ],
        }

    def route_after_think(state: AgentState) -> str:
        """思考后的路由：已有最终答案则终止，否则去执行工具."""
        return "end" if state["done"] else "act"

    def route_after_act(state: AgentState) -> str:
        """行动后的路由：达到最大步数则终止，否则回去继续思考."""
        return "end" if state["steps"] >= max_steps else "think"

    builder = StateGraph(AgentState)
    builder.add_node("think", think)
    builder.add_node("act", act)
    builder.add_edge(START, "think")
    builder.add_conditional_edges("think", route_after_think, {"act": "act", "end": END})
    builder.add_conditional_edges("act", route_after_act, {"think": "think", "end": END})
    return builder.compile(
        checkpointer=MemorySaver(),
        interrupt_before=interrupt_before or [],
    )


def run_graph(graph: Any, task: str, thread_id: str = "default") -> str:
    """以便捷方式运行状态图，返回最终答案文本.

    thread_id 是会话标识：相同 thread_id 的调用共享同一份检查点状态，
    因此可以断点恢复、事后审计。
    """
    config = {"configurable": {"thread_id": thread_id}}
    final = graph.invoke(initial_state(task), config=config)
    return final["final_answer"] or "达到最大步数限制，未能得出最终答案"
