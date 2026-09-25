# day077 源码说明

今天是 **M7 前四课复盘日（day073 ~ day076 收口）**，**不新增代码**，因此本目录不存放代码快照。

最新完整代码快照请见：[../../day076/源码/smart-research-agent/](../../day076/源码/smart-research-agent/)
（M7-D2「多头注意力」完成后的完整项目）。day020 起快照为**累积式**，包含截至当天全部模块、
测试与文档，因此 day076 的快照就是"两层数学 + 两课注意力"走完之后的最终状态。

今天的学习材料：

- 教程：[../教程/教程.md](../教程/教程.md) —— 四天地图、四个主题复盘（本质 → 为什么 → 误区）、
  一次前向的全链路走读、四条接缝与两张对照表、16 条误区清单、22 条自查清单、三张实验数据表、day078 衔接
- 习题：[../习题/习题.md](../习题/习题.md) —— 15 题（6 选择 + 5 简答 + 2 诊断 + 2 推导）
- 习题答案：[../习题答案/习题答案.md](../习题答案/习题答案.md)

## 建议的代码走读路线

带着一个问题读每一层：**"这一层的输出，依赖前一层的哪一条结论？"**
答不上来的地方，就是这一层你还没有真正读懂。推荐顺序（都是 day076 快照内的相对路径）：

1. `smart_research_agent/math_foundations/types.py` —— 形状与四张口径表：
   一天的"判据"都在这张表里（day073/074）
2. `smart_research_agent/math_foundations/attention.py` —— `masked_softmax_rows` 与
   `multi_head_attention`（**给定 Q/K/V 的纯函数版**）、`causal_mask`、`scaling_factor`（day073）
3. `smart_research_agent/math_foundations/calculus.py` —— `gradient`、`difference_resolution`：
   数值侧的"独立参照"与它的分辨率（day074）
4. `smart_research_agent/math_foundations/optim.py` —— `flatten_matrices` 与三种优化器：
   两个注意力包的训练回路都复用它（day074）
5. `smart_research_agent/transformer_core/layers.py` —— 前向七步 + 反向七步，
   以及 `grad_inputs` 是"三条链之和"（day075）
6. `smart_research_agent/transformer_core/verify.py` —— 五项梯度校验与四条性质，
   以及"没开掩码"为什么是**跳过**而不是通过（day075）
7. `smart_research_agent/multi_head/types.py` —— `HeadPartition`（权重按行切、激活按列切）
   与 `MultiHeadShape.scale_ratio`（派生量即断言）（day076）
8. `smart_research_agent/multi_head/layers.py` —— 前向九步 + 反向九步，
   以及"融合梯度 = heads 份块按行拼回"（day076）
9. `smart_research_agent/multi_head/reachability.py` —— 凸包、闵可夫斯基和与手工见证：
   "单头一条线段、双头一个正方形"（day076）
10. `smart_research_agent/multi_head/train.py` —— 四列读数与同参数量对照表（day076）

## 可复现的验证方式

快照内所有离线演示脚本都可以直接跑（不需要 API Key，全部确定性）：

```bash
cd day076/源码/smart-research-agent
python scripts/math_foundations_demo.py   # day073 十一个算子与八项对照
python scripts/calculus_demo.py           # day074 差分/反向/优化器/调度
python scripts/attention_demo.py          # day075 七步前向与七步反向
python scripts/multihead_demo.py          # day076 九步前向、可达集合与同参数量对照
```

四份权威手册（复习时按需翻）：

```text
docs/math_foundations.md          day073（12 节）
docs/calculus_and_optimization.md day074（12 节）
docs/attention_training.md        day075（11 节）
docs/multihead_attention.md       day076（11 节）
```

全量回归与覆盖率（这就是"测试全绿才 push"那条护栏）：

```bash
cd day076/源码/smart-research-agent
python -m pytest -q
```

## 今天与明天

day077 整理出来的三张表（手算锚点、梯度校验、训练读数）是 day078 的起点：

```text
今天的三张表                 →  明天的用法
手算锚点表（0.330238 / 0.107042）   position 编码的值也要能被手算（sin/cos）
梯度校验表（两项近似性质的容差）      新模块的容差继续按"分辨率"来定，而不是拍脑袋
训练读数表（同一批参数的四个读数）    位置编码会再增加一个自变量：位置
```

而 day075 的"置换等变"这条性质已经把 day078 的问题写在纸上了：
**注意力本身不知道顺序**（因果掩码只提供一种很粗的顺序）——
明天要给"位置 3 与位置 5 的距离"一个真正的表示。
