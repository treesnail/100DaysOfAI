"""智研 AI 助手入口模块."""

from __future__ import annotations

from smart_research_agent.config import settings
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> int:
    """运行项目入口，输出初始化信息."""
    logger.info("启动项目: %s", settings.project_name)
    logger.info("默认模型: %s", settings.default_model)
    logger.info("调试模式: %s", settings.debug)

    if not settings.openai_api_key:
        logger.warning("未配置 OPENAI_API_KEY，后续 LLM 调用将不可用")

    logger.info("项目初始化完成，等待后续模块接入...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
