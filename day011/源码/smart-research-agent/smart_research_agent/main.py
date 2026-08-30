"""智研 AI 助手入口模块."""

from __future__ import annotations

from smart_research_agent.agent.react_agent import ReactAgent
from smart_research_agent.agent.reflexion import Reflector
from smart_research_agent.config import settings
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.tools.calculator import CalculatorTool
from smart_research_agent.tools.registry import ToolRegistry
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)


def demo_reflexion() -> None:
    """用 MockLLM 离线演示『失败 → 反思 → 改进 → 成功』的完整流程.

    第一次尝试把中文数字喂给 calculator 必然失败；Reflector 复盘后给出
    「改用阿拉伯数字」的建议，注入第二轮提示词，重试后得出正确答案。
    """
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    agent_llm = MockLLM(
        responses=[
            # 第一次尝试：中文数字不是合法表达式，工具执行失败
            "Thought: 我需要计算 2+3\nAction: calculator\nAction Input: 二加三",
            # 反思后的第二次尝试：按建议改用阿拉伯数字
            "Thought: 上次工具输入格式错误，改用阿拉伯数字\nAction: calculator\nAction Input: 2+3",
            "Thought: 已得到正确结果\nFinal Answer: 2+3 等于 5",
        ]
    )
    reflector_llm = MockLLM(
        responses=[
            '{"reason": "把中文数字作为表达式传给了计算器，无法解析", '
            '"suggestion": "calculator 的输入必须是合法数学表达式，请改用阿拉伯数字，例如 2+3"}'
        ]
    )

    reflector = Reflector(llm=reflector_llm, max_retries=2)
    agent = ReactAgent(llm=agent_llm, registry=registry, reflector=reflector)

    answer = agent.run("2+3 等于几？")

    print("\n========== Reflexion 演示结果 ==========")
    print("最终答案:", answer)
    print("反思记录:")
    for i, r in enumerate(reflector.reflections, start=1):
        print(f"  [{i}] 失败原因: {r.reason}")
        print(f"      改进建议: {r.suggestion}")
    print("========================================\n")


def main() -> int:
    """运行项目入口，输出初始化信息并演示反思流程."""
    logger.info("启动项目: %s", settings.project_name)
    logger.info("默认模型: %s", settings.default_model)
    logger.info("调试模式: %s", settings.debug)

    if not settings.openai_api_key:
        logger.warning("未配置 OPENAI_API_KEY，演示将使用 MockLLM 离线运行")

    demo_reflexion()

    logger.info("项目初始化完成，等待后续模块接入...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
