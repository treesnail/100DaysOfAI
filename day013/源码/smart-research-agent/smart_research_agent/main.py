"""智研 AI 助手入口模块：演示 LangGraph 状态图运行与断点恢复."""

from __future__ import annotations

from smart_research_agent.config import settings
from smart_research_agent.graph import build_react_graph, initial_state, run_graph
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.tools.calculator import CalculatorTool
from smart_research_agent.tools.registry import ToolRegistry
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

# 演示用的两段 LLM 脚本输出：先调计算器，再给最终答案
_DEMO_RESPONSES = [
    "Thought: 需要用计算器\nAction: calculator\nAction Input: 2+3",
    "Thought: 已得到计算结果\nFinal Answer: 2+3 等于 5",
]


def _build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    return registry


def _demo_graph_run() -> None:
    """演示一：状态图完整跑通 ReAct 流程，并从检查点取回状态."""
    llm = MockLLM(responses=list(_DEMO_RESPONSES))
    graph = build_react_graph(llm, _build_registry(), max_steps=5)

    answer = run_graph(graph, "计算 2+3 等于几", thread_id="demo-run")
    logger.info("状态图最终答案: %s", answer)

    snapshot = graph.get_state({"configurable": {"thread_id": "demo-run"}})
    logger.info(
        "检查点状态: steps=%d, done=%s, observation=%s",
        snapshot.values["steps"],
        snapshot.values["done"],
        snapshot.values["observation"],
    )


def _demo_interrupt_resume() -> None:
    """演示二：在 act 节点前断点暂停，人工修改状态后恢复执行."""
    llm = MockLLM(responses=list(_DEMO_RESPONSES))
    graph = build_react_graph(llm, _build_registry(), interrupt_before=["act"])
    config = {"configurable": {"thread_id": "demo-bp"}}

    paused = graph.invoke(initial_state("计算 2+3 等于几"), config=config)
    logger.info("已在 act 前断点暂停，待执行 action: %s", paused["action"])
    logger.info("下一待执行节点: %s", graph.get_state(config).next)

    # 人在回路：把工具入参从 2+3 改成 10*10
    graph.update_state(config, {"action_input": "10*10"})
    logger.info("人工修正 action_input -> 10*10")

    final = graph.invoke(None, config=config)  # None 表示从检查点恢复
    logger.info("断点恢复后 observation: %s", final["observation"])
    logger.info("断点恢复后最终答案: %s", final["final_answer"])


def main() -> int:
    """运行项目入口，输出初始化信息并演示状态图."""
    logger.info("启动项目: %s", settings.project_name)
    logger.info("默认模型: %s", settings.default_model)
    logger.info("调试模式: %s", settings.debug)

    if not settings.openai_api_key:
        logger.warning("未配置 OPENAI_API_KEY，使用 MockLLM 进行离线演示")

    _demo_graph_run()
    _demo_interrupt_resume()

    logger.info("演示完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
