"""日志工具：基于标准 logging 模块，支持控制台与滚动文件输出."""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

from smart_research_agent.config import settings

# 日志目录，默认位于项目根目录的 logs/ 下
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """获取一个配置好的 Logger 实例.

    Args:
        name: 通常为 ``__name__``，便于按模块定位日志来源。

    Returns:
        配置好的 ``logging.Logger`` 实例。
    """
    logger = logging.getLogger(name)
    logger.setLevel(settings.log_level.upper())

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 滚动文件输出：单个文件最大 5MB，保留 3 个备份
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "smart_research_agent.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
