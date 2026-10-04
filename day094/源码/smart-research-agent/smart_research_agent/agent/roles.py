"""多 Agent 协作的三类角色：研究员、分析师、撰稿人.

每个角色都是 BaseLLM 之上的轻量封装：
- 各自持有独立的 system prompt（职责说明）；
- work() 接收任务与上游产出，调用一次 LLM，返回本角色的产出。

研究员将来可升级为带工具的 ReactAgent（接入搜索工具后具备真实检索能力），
接口保持不变，编排层零改动。
"""

from __future__ import annotations

from smart_research_agent.llm.base import BaseLLM
from smart_research_agent.llm.base import Message as ChatMessage
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)


class BaseRoleAgent:
    """角色基类：独立 system prompt + 单次 LLM 调用的轻量封装."""

    role: str = "role"
    system_prompt: str = ""

    def __init__(self, llm: BaseLLM, name: str | None = None):
        self.llm = llm
        self.name = name or self.role

    def work(self, task: str, context: str = "") -> str:
        """完成本角色的一份工作.

        task 是本角色要做什么，context 是上游角色的产出（可为空）。
        """
        logger.info("[%s] 开始工作: %s", self.name, task)
        prompt = f"任务：{task}" if not context else f"任务：{task}\n\n上游产出：\n{context}"
        result = self.llm.chat(
            [
                ChatMessage(role="system", content=self.system_prompt),
                ChatMessage(role="user", content=prompt),
            ]
        )
        logger.info("[%s] 产出 %d 字", self.name, len(result))
        return result


class ResearcherAgent(BaseRoleAgent):
    """研究员：围绕话题检索、收集资料."""

    role = "researcher"
    system_prompt = """你是一名研究员。你的职责是围绕给定话题收集、整理资料。

要求：
1. 列出与话题直接相关的事实、数据与背景信息
2. 标注每条资料的来源或可信度
3. 只输出资料清单，不要下结论、不要写观点
"""


class AnalystAgent(BaseRoleAgent):
    """分析师：从研究资料中提炼要点与洞察."""

    role = "analyst"
    system_prompt = """你是一名分析师。你的职责是从研究资料中提炼要点。

要求：
1. 归纳出 3~5 个核心要点，每个要点一句话
2. 指出资料之间的矛盾或不确定性
3. 只输出要点列表，不要展开成文章
"""


class WriterAgent(BaseRoleAgent):
    """撰稿人：把分析要点写成结构完整的成稿."""

    role = "writer"
    system_prompt = """你是一名撰稿人。你的职责是把分析要点写成一篇结构完整的短文。

要求：
1. 包含标题、引言、正文、结论四个部分
2. 正文围绕分析要点展开，不引入要点之外的新事实
3. 语言通顺，面向普通读者
"""
