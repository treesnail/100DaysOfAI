# Git 工作流规范

本文档定义「智研 AI 助手」项目的 Git 使用规范。

## 分支模型

本项目采用 **GitHub Flow** 的简化变体（Trunk-Based Development 的入门形态）：

```
main (受保护主分支)
  └── dayXXX-主题   (每日学习功能分支)
```

规则：

1. `main` 始终保持可运行状态，所有测试必须通过。
2. 每天的学习工作在功能分支上进行，完成后合并回 `main`。
3. 分支命名：`dayXXX-简短描述`，例如 `day002-git-workflow`。

### 为什么不直接用 Git Flow

Git Flow（develop / feature / release / hotfix 多分支模型）适合多版本并行的商业软件，
但对单人学习项目来说过重：分支层级多、合并流程长、认知负担大。

GitHub Flow 只有 `main` + 功能分支，配合 Pull Request 评审即可，更适合：

- 单人或小团队
- 持续集成环境
- 频繁交付的项目

## 提交信息规范（Conventional Commits）

格式：

```text
<type>(<scope>): <subject>
```

常用 type：

| type | 含义 | 示例 |
|------|------|------|
| `feat` | 新功能 | `feat(agent): 实现 ReAct 推理循环` |
| `fix` | 修复缺陷 | `fix(logger): 修复 handler 重复添加` |
| `docs` | 文档变更 | `docs(day002): 补充 Git 工作流文档` |
| `test` | 测试相关 | `test(config): 增加环境变量覆盖用例` |
| `chore` | 杂项（配置、工具链） | `chore: 更新 pre-commit 版本` |
| `refactor` | 重构（不改变外部行为） | `refactor(config): 拆分 Settings 字段` |

原则：

- 一个 commit 只做一件事（原子提交）。
- subject 用祈使句、结尾不加句号、不超过 50 字符。
- 需要详细说明时，空一行后写 body。

## 日常工作流

```bash
# 1. 从最新 main 切出当天分支
git checkout main
git pull origin main
git checkout -b dayXXX-主题

# 2. 开发 + 小步提交（完成一个逻辑单元就提交一次）
git add -p            # 交互式选择要提交的改动
git commit -m "feat(scope): 说明"

# 3. 完成后合并回 main
git checkout main
git merge --no-ff dayXXX-主题 -m "merge: dayXXX 主题"
git push origin main
git branch -d dayXXX-主题
```

### 为什么用 `--no-ff` 合并

默认的 fast-forward 合并会让历史变成一条直线，看不出哪些提交属于同一个功能。
`--no-ff` 强制创建一个合并提交（merge commit），在历史图上保留"这一组提交属于某个功能分支"的信息，
日后回顾 100 天学习轨迹时，每个阶段的边界清晰可见。

## 常用命令速查

```bash
git log --oneline --graph     # 图形化查看提交历史
git diff                      # 工作区 vs 暂存区
git diff --staged             # 暂存区 vs 最后一次提交
git restore <file>            # 撤销工作区修改
git restore --staged <file>   # 把文件移出暂存区
git reset --soft HEAD~1       # 撤销上一次提交（保留改动）
```
