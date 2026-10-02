# day091 源码说明

今天是 **M8 前段复盘日（day089 ~ day090 收口）**，**不新增代码**，因此本目录不存放代码快照。

最新完整代码快照请见：[../../day090/源码/smart-research-agent/](../../day090/源码/smart-research-agent/)
（M8-D2「反向传播」完成后的完整项目）。day020 起快照为**累积式**，
包含截至当天全部模块、测试与文档，因此 day090 的快照就是"两天走完之后的最终状态"。

今天的学习材料：

- 教程：[../教程/教程.md](../教程/教程.md) —— 两天地图（day089~090：两个新包 19 个文件
  2092 条语句 196 个新用例）、两个主题按「本质·为什么·误区」复盘（神经网络基础·反向传播）、
  一次前向穿过两个包的全链路走读（含五个提问点与十处接线）、五张对照表（前向↔反向算子对照、
  三种求和指标、三条独立路径、六处必须逐位相等的常量、"恰好 0"与"接近 0"）、
  12 条误区清单（带证据位置）、12 条自查清单对齐到具体某一天、
  三张实验数据表（手算锚点·反向三条路径·一次真的下降）、三个最值钱的推导与 day092 衔接
- 习题：[../习题/习题.md](../习题/习题.md) —— 18 题（10 速答 + 1 连线 + 3 对照 + 2 诊断 + 2 推导）
- 习题答案：[../习题答案/习题答案.md](../习题答案/习题答案.md)

## 建议的代码走读路线

带着一个问题读每一层：**"这一步的输出，依赖前一步的哪一条结论？"**
答不上来的地方，就是这一层你还没有真正读懂。推荐顺序（都是 day090 快照内的相对路径）：

1. `smart_research_agent/neural_basics/types.py` —— 六个激活 / 三个损失 / 五种初始化 /
   七条性质两张表：一天的口径都在这张表里（day089）
2. `smart_research_agent/neural_basics/activations.py` —— 六个激活的前向与
   **三个数值稳定点**（sigmoid 按符号分支、softmax 先减最大值、gelu 用精确 erf 式）（day089）
3. `smart_research_agent/neural_basics/layers.py` —— Dense（`x·Wᵀ + b`）与五种初始化（LCG）：
   权重形状 `(out, in)` 是后面所有反向式子的形状依据（day089）
4. `smart_research_agent/neural_basics/network.py` —— MLP 前向、`forward_trace`、
   恒等塌缩、`ffn_block`：第 7 条性质把前馈接回 day079（day089）
5. `smart_research_agent/backprop/gradients.py` —— 六个局部导数 + softmax 的显式雅可比
   + 那条 O(n) 的 JVP：**先看 `elementwise_backward` 的签名为什么带 `pre_activation`**（day090）
6. `smart_research_agent/backprop/layers.py` —— 一层反向的三块梯度：重点看
   `grad_inputs` 为什么"按 W 的**列**收"，以及累加方式为什么刻意与 day079 相同（day090）
7. `smart_research_agent/backprop/network.py` —— `ForwardCache` 留的三样中间量、
   `activation_backward` 的三个分支、`ffn_backward` 与生产实现的接缝（day090）
8. `smart_research_agent/backprop/graph.py` —— 一张**张量级**计算图：三个约定
   （不改值 / 累加 / 常量不接收梯度）与 `dense` 那个算子的三块局部导数（day090）
9. `smart_research_agent/backprop/errors.py` —— **GradientError 回来了**：
   `RETURNED_FAMILY` 与 `ABSENT_FAMILY = None` 是常量，因此"这一天没有缺席者"可被断言（day090）
10. `smart_research_agent/backprop/verify.py` —— 七条性质与两类判据：重点看
    `GradientCheck.upper_bound` 如何把"相等"与"不超过上界"分开（与 day087~089 同源）（day090）

## 可复现的验证方式

快照内所有离线演示脚本都可以直接跑（不需要 API Key，全部确定性）：

```bash
cd day090/源码/smart-research-agent
python scripts/neural_basics_demo.py      # day089 十一节：激活 / 初始化 / 塌缩 / 损失 / 对账
python scripts/backprop_demo.py           # day090 十一节：导数 / 三块梯度 / 两条路径 / 训练
```

两本权威手册（复习时按需翻）：

```text
docs/neural_basics.md      day089（10 节）
docs/backprop.md           day090（11 节）
```

单文件回归（只跑这两天的新用例，秒级）：

```bash
cd day090/源码/smart-research-agent
python -m pytest tests/test_neural_basics.py tests/test_backprop.py -q --no-cov
# 本机实测：196 passed（96 + 100），用时约 1.2s
# 新包覆盖率（只看新包）：backprop 1188 条语句 / 358 条分支，97.54%
python -m pytest tests/test_backprop.py -q -o addopts='' \
  --cov=smart_research_agent/backprop --cov-report=term-missing --cov-branch
```

全量回归与覆盖率（这就是"测试全绿才 push"那条护栏）：

```bash
cd day090/源码/smart-research-agent
python -m pytest -q
# 本机实测：14372 passed / 1 skipped；总覆盖率 98.08%
#   （比 day089 的 14272 多出的 100 条，就是本日新增的 tests/test_backprop.py）
# 新包 backprop：1188 条语句 / 358 条分支，覆盖率 97.54%
```

## 今天与明天

day091 整理出来的五张对照表是 day092 的起点：

```text
今天的对照表                            →  明天的用法
"三种求和指标"（dW 按样本维、dX 按输出维、softmax 按行）  优化器逐分量更新 ⇒
                                          参数要先按层压平（layers.flatten_parameters）
"一次真的下降"用了 day074 的 sgd / adam   day092 要问的是"往哪走、走多远"：
                                          三个更新公式本身，以及学习率曲线为什么先升后降
"梯度整体范数"（train 里印在每一步）      梯度裁剪用**整体范数**而不是逐分量——
                                          逐分量裁剪会改变方向（day074 的 clip_by_value 已写下这句话）
```

而 day089 留下、day090 接住的那半句问题也在这里被接住：
**"多层凭什么比一层强？"** 有了一条反向的答案——不是"层数多"，而是
**那个激活函数在前向提供非线性、在反向充当唯一的开关**。
没有它，前向塌缩成一次仿射（day089 第 8 章），反向也退化成一次矩阵乘的链（day090 第 4 章）。
