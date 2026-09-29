# day084 源码说明

今天是 **M7 中段复盘日（day078 ~ day083 收口）**，**不新增代码**，因此本目录不存放代码快照。

最新完整代码快照请见：[../../day083/源码/smart-research-agent/](../../day083/源码/smart-research-agent/)
（M7-D8「可解释性与注意力可视化」完成后的完整项目）。day020 起快照为**累积式**，
包含截至当天全部模块、测试与文档，因此 day083 的快照就是"六天中段走完之后的最终状态"。

今天的学习材料：

- 教程：[../教程/教程.md](../教程/教程.md) —— 六天地图（day078~083：六个新包 46 个文件 7298 条语句
  1800 个新用例）、六个主题按「本质·为什么·误区」复盘（位置编码·块与链·训练五个旋钮·三个变体·读表）、
  一次前向穿过六个包的全链路走读（含五个提问点与十处接线）、五张对照表（四对同一件事的两种写法、
  六处必须逐位相等的常量、一处落族差异与一处刻意近似）、24 条误区清单（带证据位置）、
  27 条自查清单对齐到具体某一天、三张实验数据表（手算锚点 10 行·梯度校验 10 行·训练与实验读数 3 组）、
  四个最值钱的推导与 day085 衔接
- 习题：[../习题/习题.md](../习题/习题.md) —— 24 题（8 选择 + 6 简答 + 4 诊断 + 4 推导 + 2 综合）
- 习题答案：[../习题答案/习题答案.md](../习题答案/习题答案.md)

## 建议的代码走读路线

带着一个问题读每一层：**"这一层的输出，依赖前一层的哪一条结论？"**
答不上来的地方，就是这一层你还没有真正读懂。推荐顺序（都是 day083 快照内的相对路径）：

1. `smart_research_agent/positional_encoding/types.py` —— 六个阶段与六项梯度两张口径表、
   七条性质：一天的"判据"都在这张表里（day078）
2. `smart_research_agent/positional_encoding/layers.py` —— 造表 / 注入 / 前向 / 反向：
   重点看反向那两步为什么是"逐位传回 + 按位置累加"（day078）
3. `smart_research_agent/encoder_decoder/layers.py` —— 六个阶段与九项梯度、两处残差：
   前向 `y = x + F(x)`、反向 `dx = dy + dF`；`block_attention` 是"块拥有第一个子层"的实现（day079）
4. `smart_research_agent/encoder_decoder/depth.py` —— 深度实验：
   `pre_residual 9.27e+00`、`post_residual 9.45e-01`、`bare 1.92e-03` 的那张表（day079）
5. `smart_research_agent/transformer_stack/layers.py` —— 链式前向 / 链式反向 / 逐层读数：
   反向的**唯一难点**是 `current = layer_grads.grad_inputs` 那一行（day080）
6. `smart_research_agent/transformer_stack/verify.py` —— 两处梯度校验与六条性质，
   以及"与手写循环逐位一致"为什么用 `==`（day080）
7. `smart_research_agent/transformer_stack/assembly.py` —— 把形状编译成一段 PyTorch 脚本、
   再用 `ast` 解析回结构；参数量差值 `2040 − 1944 = 96 = N × 4d`（day080）
8. `smart_research_agent/training_optim/dropout.py` —— 掩码（确定性 LCG）、两个相、一行反向：
   `mask_seed = seed + step × 1009 + layer × 31`（day081）
9. `smart_research_agent/training_optim/controls.py` —— 调度（转发 day074）、裁剪（转发 day074）、
   早停（新增）：这一课"一条公式都没有重写"（day081）
10. `smart_research_agent/arch_variants/stacks.py` —— 接线：掩码按变体给、两条路接上、
    反向重放前向；`mask_for_streams` 是"掩码该给哪一条流"的答案（day082）
11. `smart_research_agent/arch_variants/probe.py` —— 实测依赖表：
    "因果性是用扰动量出来的"这条判据就落在这里（day082）
12. `smart_research_agent/explainability/extract.py` —— 两条读法：模型**真实**权重与多头读法，
    以及 `heads = 1` 必须逐位相等这条接缝（day083）
13. `smart_research_agent/explainability/analyze.py` —— 六个读数 + 天花板 + 归一化熵 + 头间余弦（day083）
14. `smart_research_agent/explainability/rollout.py` —— `Â = α·A + (1−α)·I` 与
    `R = Â_L ⋯ Â_1`：那一项 `(1−α)·I` 就是残差（day083）

## 可复现的验证方式

快照内所有离线演示脚本都可以直接跑（不需要 API Key，全部确定性）：

```bash
cd day083/源码/smart-research-agent
python scripts/position_demo.py            # day078 十一个算子与八项对照、四个编码刻度
python scripts/encoder_decoder_demo.py     # day079 六个阶段、九项梯度、深度实验
python scripts/transformer_stack_demo.py   # day080 五个阶段、逐层读数、堆叠实验
python scripts/training_optim_demo.py      # day081 五个旋钮、四组实验、181 行输出
python scripts/arch_variants_demo.py       # day082 三张掩码、依赖探针、四张表
python scripts/explainability_demo.py      # day083 两条读法、六个读数、滚动、六条性质
```

六份权威手册（复习时按需翻）：

```text
docs/position_encoding.md          day078（12 节）
docs/encoder_decoder.md            day079（12 节）
docs/transformer_stack.md          day080（12 节）
docs/training_optim.md             day081（10 节）
docs/arch_variants.md              day082（10 节）
docs/explainability.md             day083（10 节）
```

全量回归与覆盖率（这就是"测试全绿才 push"那条护栏）：

```bash
cd day083/源码/smart-research-agent
python -m pytest -q
# 本机实测：13613 passed；总覆盖率 98.13%（六个新包分别 98% / 99% / 98% / 97% / 99% / 99%）
```

## 今天与明天

day084 整理出来的三张表是 day085 的起点：

```text
今天的对照表                       →  明天的用法
"分头放在投影内部"（day083 的接口事实）  HF 的注意力**同时**支持 n_heads 与因果掩码，
                                     靠的正是把分头放进投影里（view(...).transpose(...)）
"多层 = ModuleList + 一个 for"          Hugging Face 的 TransformerEncoder 就是这条
（day080 的伏笔，已写进生成的脚本）         —— 反过来读源码时，那个 for 要认得出来
eps = 1e-5 与 pre/post 的位置           HF 里 LayerNorm 的默认 eps 与本文档同源，
                                     而 pre/post 体现在"norm 放在哪两个 Module 之间"
```

而 day083 留下的那半句问题也在这里被接住：
**"哪些表在说什么"已经有读数了，而"HF 里那些表是怎么造出来的"明天才读。**
day079 与 day083 都记下过一条**接口边界**——"一个块只支持一种掩码"——
而 Hugging Face 的 `T5Attention` 并不接受这条边界：它把"分头"放进投影内部，
于是 `n_heads` 与因果掩码可以在同一个类里共存。这是明天要读的第一处源码。
