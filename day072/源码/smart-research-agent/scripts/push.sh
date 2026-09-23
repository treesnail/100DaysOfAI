#!/usr/bin/env bash
# Git 推送脚本：提交并推送当天学习成果
# 用法: ./scripts/push.sh "提交信息"
set -euo pipefail

COMMIT_MSG="${1:-}"

if [ -z "$COMMIT_MSG" ]; then
    echo "错误：请提供提交信息，例如: ./scripts/push.sh 'feat: Day X - 内容摘要'"
    exit 1
fi

echo "[1/3] 暂存所有变更"
git add .

echo "[2/3] 提交: $COMMIT_MSG"
git commit -m "$COMMIT_MSG"

echo "[3/3] 推送到远程 main 分支"
git push origin main

echo "推送完成"
