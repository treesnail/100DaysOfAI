"""环境验证脚本：检查 Python 版本、依赖导入与项目配置."""

from __future__ import annotations

import sys


def check_python_version() -> bool:
    """检查 Python 版本是否 >= 3.10."""
    if sys.version_info < (3, 10):
        print(f"[失败] Python 版本过低: {sys.version}")
        return False
    print(f"[通过] Python 版本: {sys.version.split()[0]}")
    return True


def check_imports() -> bool:
    """检查核心依赖是否可以导入."""
    try:
        import pydantic  # noqa: F401
        import pydantic_settings  # noqa: F401

        print("[通过] 核心依赖导入正常")
        return True
    except ImportError as exc:
        print(f"[失败] 依赖导入失败: {exc}")
        return False


def check_project() -> bool:
    """检查项目配置与日志模块能否正常加载."""
    try:
        from smart_research_agent.config import settings
        from smart_research_agent.utils.logger import get_logger

        logger = get_logger("verify_env")
        logger.info("配置加载成功: project_name=%s", settings.project_name)
        print("[通过] 项目配置与日志加载正常")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[失败] 项目加载异常: {exc}")
        return False


def main() -> int:
    """主入口."""
    checks = [
        check_python_version(),
        check_imports(),
        check_project(),
    ]
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
