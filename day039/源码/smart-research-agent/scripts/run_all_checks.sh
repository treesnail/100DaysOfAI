#!/usr/bin/env bash
# 本地一键质量流水线（day032）：单元测试 → Agent 评估 → 红队评估。
# 任一环节失败即以非零退出码结束，与 .github/workflows/ci.yml 的门禁完全一致。
#
# 用法（在项目根目录，先激活 venv）：
#   bash scripts/run_all_checks.sh
# 也可用 PYTHON 环境变量指定解释器：
#   PYTHON=/path/to/.venv/Scripts/python.exe bash scripts/run_all_checks.sh
set -uo pipefail

PYTHON="${PYTHON:-python}"

cd "$(dirname "$0")/.."

step() {
    echo ""
    echo "=================================================================="
    echo "  $1"
    echo "=================================================================="
}

step "[1/3] 单元测试 + 覆盖率门禁（fail_under=80）"
"$PYTHON" -m pytest tests/ -q || { echo "❌ 单元测试失败"; exit 1; }

step "[2/3] Agent 评估门禁（completion_rate>=0.6, step_efficiency>=0.7）"
"$PYTHON" scripts/run_agent_eval.py --min-completion-rate 0.6 --min-step-efficiency 0.7 \
    || { echo "❌ Agent 评估未过门禁"; exit 1; }

step "[3/3] 红队评估门禁（block_rate>=1.0，已知攻击集必须全拦）"
"$PYTHON" scripts/run_redteam.py --min-block-rate 1.0 \
    || { echo "❌ 红队评估未过门禁"; exit 1; }

echo ""
echo "✅ 全部检查通过"
exit 0
