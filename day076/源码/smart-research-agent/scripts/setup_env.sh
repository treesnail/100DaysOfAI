#!/usr/bin/env bash
# 一键初始化智研 AI 助手开发环境
#
# 设计原则：
#   1. 幂等性：重复执行不会产生副作用（venv 已存在则跳过创建）
#   2. 失败即停：set -euo pipefail，任何一步失败立即退出
#   3. 可移植：自动检测 python3 / python，兼容 Linux / macOS / Windows(Git Bash)
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/venv"

# 自动检测可用的 Python 命令（优先 python3，退化为 python）
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    echo "错误：未找到 python3 或 python，请先安装 Python 3.10+"
    exit 1
fi

echo "[1/7] 检查 Python 版本"
"$PYTHON_CMD" -c "import sys; assert sys.version_info >= (3, 10), f'需要 Python >= 3.10，当前 {sys.version}'"

echo "[2/7] 创建虚拟环境: $VENV_DIR"
if [ -d "$VENV_DIR" ]; then
    echo "venv 已存在，跳过创建"
else
    "$PYTHON_CMD" -m venv "$VENV_DIR"
fi

# 激活虚拟环境：Windows(Git Bash) 与 Unix 的目录结构不同
if [ -f "$VENV_DIR/Scripts/activate" ]; then
    source "$VENV_DIR/Scripts/activate"   # Windows
else
    source "$VENV_DIR/bin/activate"       # Linux / macOS
fi

echo "[3/7] 升级 pip"
pip install --upgrade pip

echo "[4/7] 安装运行时与开发依赖"
pip install -r "$PROJECT_DIR/requirements.txt"
pip install -e "$PROJECT_DIR[dev]"

echo "[5/7] 安装并注册 pre-commit 钩子"
if [ -d "$PROJECT_DIR/.git" ]; then
    pre-commit install
else
    echo "警告：当前目录不是 Git 仓库，跳过 pre-commit 安装"
fi

echo "[6/7] 准备 .env 文件"
if [ ! -f "$PROJECT_DIR/.env" ]; then
    cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
    echo "已创建 .env，请编辑该文件填入你的 OPENAI_API_KEY"
else
    echo ".env 已存在，跳过复制"
fi

echo "[7/7] 运行环境验证"
python "$PROJECT_DIR/scripts/verify_env.py"

echo ""
echo "初始化完成！激活虚拟环境后即可开始开发："
if [ -f "$VENV_DIR/Scripts/activate" ]; then
    echo "  source venv/Scripts/activate"
else
    echo "  source venv/bin/activate"
fi
