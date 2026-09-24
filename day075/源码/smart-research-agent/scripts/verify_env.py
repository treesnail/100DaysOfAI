#!/usr/bin/env python
"""环境验证脚本：确认开发环境各组件可用.

验证项：
  1. Python 版本 >= 3.10
  2. 关键依赖包已安装且可导入
  3. 项目配置可正常加载
  4. Git 仓库已初始化（可选）
  5. Docker 可用（可选，不阻塞）
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from pathlib import Path


def check_python_version() -> bool:
    v = sys.version_info
    ok = v >= (3, 10)
    status = "OK" if ok else "FAIL"
    print(f"[{status}] Python {v.major}.{v.minor}.{v.micro} (要求 >= 3.10)")
    return ok


def check_packages() -> bool:
    packages = ["pydantic", "pydantic_settings", "dotenv"]
    all_ok = True
    for pkg in packages:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "unknown")
            print(f"[OK] {pkg} {version}")
        except ImportError:
            print(f"[FAIL] {pkg} 未安装，请运行: pip install -r requirements.txt")
            all_ok = False
    return all_ok


def check_project_config() -> bool:
    try:
        from smart_research_agent import __version__
        from smart_research_agent.config import settings

        print(f"[OK] 项目 {settings.project_name} v{__version__} 配置加载成功")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] 项目配置加载失败: {exc}")
        return False


def check_git() -> bool:
    if Path(".git").exists() and shutil.which("git"):
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        branch = result.stdout.strip()
        print(f"[OK] Git 仓库已初始化，当前分支: {branch}")
        return True
    print("[WARN] 当前目录未初始化为 Git 仓库，建议运行: git init")
    return True  # 不阻塞


def check_docker() -> bool:
    if shutil.which("docker"):
        print("[OK] Docker 已安装")
    else:
        print("[SKIP] Docker 未安装（后续 M2 阶段才需要，可暂时跳过）")
    return True  # 不阻塞


def main() -> int:
    print("=" * 50)
    print("智研 AI 助手 - 环境验证")
    print("=" * 50)

    results = [
        check_python_version(),
        check_packages(),
        check_project_config(),
        check_git(),
        check_docker(),
    ]

    print("=" * 50)
    if all(results):
        print("环境验证通过，可以开始开发！")
        return 0
    print("环境验证失败，请根据上面的 FAIL 项修复。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
