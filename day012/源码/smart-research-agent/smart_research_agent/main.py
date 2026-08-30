"""智研 AI 助手入口模块：演示多 Agent 协作.

离线环境下用 MockLLM 播放预设脚本，展示 研究员 -> 分析师 -> 撰稿人 的
完整协作流水线与消息轨迹；配置 OPENAI_API_KEY 后可替换为真实 LLM。
"""

from __future__ import annotations

from smart_research_agent.agent.orchestrator import MultiAgentOrchestrator
from smart_research_agent.agent.roles import AnalystAgent, ResearcherAgent, WriterAgent
from smart_research_agent.config import settings
from smart_research_agent.llm.base import BaseLLM
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.llm.openai_compatible import OpenAICompatibleLLM
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

# 离线演示脚本：研究 -> 分析 -> 撰稿 三段预设产出
DEMO_SCRIPT = [
    "资料清单：\n1. RAG = 检索增强生成，先检索相关文档再交给 LLM 生成\n"
    "2. 常见框架：LangChain、LlamaIndex、Haystack\n3. 优势：缓解幻觉、知识可更新",
    "核心要点：\n1. RAG 把检索与生成解耦\n2. 检索质量决定生成上限\n3. 适合知识密集型场景",
    "# RAG 技术漫谈\n\n## 引言\n大模型知识存在时效与幻觉问题……\n\n"
    "## 正文\nRAG 把检索与生成解耦，检索质量决定生成上限……\n\n"
    "## 结论\nRAG 是知识密集型场景的首选架构。",
]


def build_llm() -> BaseLLM:
    """有 API Key 时用真实模型，否则回退到 MockLLM 演示脚本."""
    if settings.openai_api_key:
        return OpenAICompatibleLLM()
    logger.warning("未配置 OPENAI_API_KEY，使用 MockLLM 离线演示")
    return MockLLM(responses=DEMO_SCRIPT)


def main() -> int:
    """运行多 Agent 协作演示."""
    logger.info("启动项目: %s", settings.project_name)

    llm = build_llm()
    orchestrator = MultiAgentOrchestrator(
        researcher=ResearcherAgent(llm=llm),
        analyst=AnalystAgent(llm=llm),
        writer=WriterAgent(llm=llm),
    )
    result = orchestrator.run("RAG 检索增强生成技术")

    print("\n===== 消息轨迹 =====")
    for i, msg in enumerate(result["message_log"], start=1):
        print(f"[{i}] {msg['sender']} -> {msg['receiver']} (stage={msg['metadata']['stage']})")
        print(f"    {msg['content'][:60]}")

    print("\n===== 最终成稿 =====")
    print(result["article"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
