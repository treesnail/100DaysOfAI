#!/usr/bin/env bash
# 一键初始化智研 AI 助手开发环境
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/venv"

echo "[1/6] 创建虚拟环境: $VENV_DIR"
python3 -m venv "$VENV_DIR"

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo "[2/6] 升级 pip"
pip install --upgrade pip

echo "[3/6] 安装运行时与开发依赖"
pip install -r "$PROJECT_DIR/requirements.txt"

echo "[4/6] 以 editable 模式安装本项目"
pip install -e "$PROJECT_DIR"

echo "[5/6] 安装并注册 pre-commit 钩子"
pre-commit install

echo "[6/6] 复制 .env 模板"
if [ ! -f "$PROJECT_DIR/.env" ]; then
    cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
    echo "已创建 .env，请编辑该文件填入你的 OPENAI_API_KEY"
else
    echo ".env 已存在，跳过复制"
fi

echo "运行环境验证..."
python "$PROJECT_DIR/scripts/verify_env.py"

echo "初始化完成！请执行 'source venv/bin/activate' 激活虚拟环境。"
