# day097 源码说明

今天是 **R3 复习缓冲日（M7 Transformer 架构 + M8 深度学习收口）**，**不新增代码**，
因此本目录不存放代码快照。

最新完整代码快照请见：[../../day096/源码/smart-research-agent/](../../day096/源码/smart-research-agent/)
（M8-D7「PyTorch 高级与综合实践」完成后的完整项目）。day020 起快照为**累积式**，
包含截至当天全部模块、测试与文档，因此 day096 的快照就是"二十余天走完之后的最终状态"——
它里面既有 M7 的注意力 / 多头 / 位置 / 堆叠 / 变体 / 推理优化，也有 M8 的七个新包。

今天的学习材料：

- 教程：[../教程/教程.md](../教程/教程.md) —— 两个阶段的地图（一次点积 → 一次反向 →
  一条训练管线）、M7 按「本质·为什么·误区」复盘（注意力 / 多头与位置与堆叠 / 训练变体与
  可解释 / HF 与推理优化 / 原理串联）、M8 按同样结构复盘（前向 / 反向 / 优化器 / 卷积 /
  循环 / 正则化 / 工程管线）、三条贯穿线（依赖地图 / 判据体系演进 / 失败族演进史）、
  第五章把 7 组底层结论接回主线项目的上层能力、五张对照表（M7↔M8 关键等式 / 三类判据读数 /
  "恰好 0"与"接近 0" / 失败族该谁修 / 七天工程读数）、14 条误区清单（带证据位置）、
  18 条自查清单（对齐到具体某一天）、三个最值钱的推导与走向结业项目
- 习题：[../习题/习题.md](../习题/习题.md) —— 18 题（10 速答 + 1 连线 + 3 对照 + 2 诊断 + 2 推导）
- 习题答案：[../习题答案/习题答案.md](../习题答案/习题答案.md)

## 建议的代码走读路线

带着一个问题读每一层：**"这一步的输出，依赖前一步的哪一条结论？"**
答不上来的地方，就是这一层你还没有真正读懂。推荐顺序（都是 day096 快照内的相对路径）。

M7 部分（"一次点积"这一半，`smart_research_agent/` 下）：

1. `math_foundations/probability.py` —— 那个被 M8 反复复用的 LCG（`uniforms`）：
   **它是怎么让"同一个种子给出同一串数"这件事成立的？**（day073）
2. `transformer_core/attention.py` —— Scaled Dot-Product 的三行：
   **"每一行和为 1"是定义还是学到的？为什么它必须成立？**（day075）
3. `transformer_core/errors.py` —— 失败族按"该谁去修"分：
   **`GradientError` 与 `MathError` 的分界为什么是"控制流 vs 报告"？**（day075）
4. `multi_head/` 与 `positional_encoding/` —— 切头与注入位置：
   **多头是在哪一步切？为什么没有位置编码时输出对顺序不变？**（day076 / day078）
5. `encoder_decoder/layers.py` —— `layer_norm` 与它的 `DEFAULT_EPSILON = 1e-5`：
   **这个常量为什么在 day095 被 day094 的旧账"跨天 import"了一次？**（day079）
6. `inference_optim/` 与 `principle_map/` —— KV Cache 与整张原理图：
   **缓存复用为什么是"逐位精确"、量化误差为什么以"半步长"为上界？**（day087 / day088）

M8 部分（"一次反向 + 一条管线"这一半）：

1. `neural_basics/` —— 六个激活 / 三个损失 / 五种初始化（LCG）：
   **拿掉激活函数，多层为什么塌缩成一层？**（day089）
2. `backprop/gradients.py` —— 六个局部导数 + softmax 的 O(n) JVP：
   **`elementwise_backward` 的签名为什么必须带"激活前的值"？**（day090）
3. `backprop/graph.py` 与 `backprop/verify.py` —— 一张张量级计算图与三类判据：
   **手写反向与图自动微分为什么能逐位一致（0.000e+00）？**（day090）
4. `optimizers/optimizer.py` —— `TrainingOptimizer.step` 写死的四个动作顺序：
   **顺序错了为什么不报错、只算错？**（day092）
5. `conv_net/ops.py` 与 `sequence_models/layers.py` —— 两种权重共享：
   **"同一组权重"在空间与时间上各被复用了多少次？参数量为什么与输入规模无关？**（day093 / day094）
6. `regularization/normalization.py` —— BatchNorm 与它的两相：
   **批大小为 1 时输出为什么会塌成常数 β？训练相与推理相为什么必须不同？**（day095）
7. `torch_pipeline/train.py` 与 `torch_pipeline/checkpoint.py` —— 七阶段与检查点：
   **恢复训练为什么必须逐位相等、而不是"损失差不多"？**（day096）

## 可复现的验证方式

快照内所有离线演示脚本都可以直接跑（不需要 API Key，全部确定性）：

```bash
cd day096/源码/smart-research-agent
# ---- M7 的确定性演示 ----
python scripts/math_foundations_demo.py    # 线性代数 / 概率论（含 LCG）
python scripts/calculus_demo.py            # 微积分 / 数值差分这把尺子 / 三个优化器
python scripts/attention_demo.py           # 注意力
python scripts/multihead_demo.py           # 多头注意力
python scripts/position_demo.py            # 位置编码
python scripts/encoder_decoder_demo.py     # Encoder / Decoder
python scripts/transformer_stack_demo.py   # 可堆叠的块
python scripts/training_optim_demo.py      # 训练回路与旋钮
python scripts/arch_variants_demo.py       # BERT / GPT / T5
python scripts/explainability_demo.py      # 注意力可视化
python scripts/hf_source_demo.py           # HF 源码精读
python scripts/hf_integration_demo.py      # 进程内跑 HF 模型
python scripts/inference_optim_demo.py     # KV Cache / 量化 / 批处理计划
python scripts/principle_map_demo.py       # 原理 → 应用
# ---- M8 的确定性演示 ----
python scripts/neural_basics_demo.py       # day089 十一节：激活 / 初始化 / 塌缩 / 损失 / 对账
python scripts/backprop_demo.py            # day090 十一节：导数 / 三块梯度 / 两条路径 / 训练
python scripts/optimizers_demo.py          # day092 十一节：六规则 / 一步 / 裁剪 / 调度 / 对比
python scripts/conv_net_demo.py            # day093 十一节：层 / 尺寸 / 感受野 / 特征图 / 池化 / 训练
python scripts/sequence_models_demo.py     # day094 十一节：单元 / 门 / 状态 / 反向 / 训练
python scripts/regularization_demo.py      # day095 十一节：两轴 / 两相 / 四旋钮 / 日志 / 消融
python scripts/torch_pipeline_demo.py      # day096 十一节：数据 / 设备 / 检查点 / 推理 / 端到端
```

权威手册（复习时按需翻；文件名与模块同名，均为纯文本）：

```text
docs/math_foundations.md              day073（线性代数 + 概率论）
docs/calculus_and_optimization.md     day074（微积分 + 优化）
docs/attention_training.md            day075（Attention）
docs/multihead_attention.md           day076（多头注意力）
docs/position_encoding.md             day078（位置编码）
docs/encoder_decoder.md               day079（Encoder / Decoder）
docs/transformer_stack.md             day080（从零组装 Transformer Block）
docs/training_optim.md                day081（训练回路与旋钮）
docs/arch_variants.md                 day082（BERT / GPT / T5）
docs/explainability.md                day083（可解释性）
docs/hf_source.md                     day085（HF 源码精读）
docs/hf_integration.md                day086（HF 集成）
docs/inference_optim.md               day087（高效推理与量化）
docs/principle_map.md                 day088（原理串联）
docs/neural_basics.md                 day089（10 节）
docs/backprop.md                      day090（11 节）
docs/optimizers.md                    day092（10 节）
docs/conv_net.md                      day093（10 节）
docs/sequence_models.md               day094（11 节）
docs/regularization.md                day095（12 节）
docs/torch_pipeline.md                day096（12 节）
```

单文件回归（只跑 M8 七天的新用例，秒级）：

```bash
cd day096/源码/smart-research-agent
python -m pytest tests/test_neural_basics.py tests/test_backprop.py tests/test_optimizers.py \
  tests/test_conv_net.py tests/test_sequence_models.py tests/test_regularization.py \
  tests/test_torch_pipeline.py -q --no-cov
# 本机实测：各天新增用例 96 + 100 + 110 + 88 + 80 + 69 + 80
```

新包覆盖率（只看最后一天的新包）：

```bash
cd day096/源码/smart-research-agent
python -m pytest tests/test_torch_pipeline.py -q -o addopts='' \
  --cov=smart_research_agent/torch_pipeline --cov-report=term-missing --cov-branch
# 本机实测：torch_pipeline 1244 条语句 / 306 条分支，覆盖率 100%
```

全量回归与覆盖率（这就是"测试全绿才 push"那条护栏）：

```bash
cd day096/源码/smart-research-agent
python -m pytest -q
# 本机实测（day096）：collected 14800（14799 passed / 1 skipped），总覆盖率 98.20%
#   逐天累计：14372（day090）→ 14482（day092）→ 14570（day093）→
#             14650（day094）→ 14719（day095）→ 14800 collected（day096）
```

## 今天与明天

day097 整理出来的三张线，是结业项目（day099 / day100）的起点：

```text
今天的贯穿线                                   →  结业项目的用法
"依赖地图"（一次点积 → 一次反向 → 一条管线）      挑选模型与训练路径时，先问"这一步依赖哪一条结论"
"判据体系"（相等 / 上界 / 下界三类，各有一个现场读数）部署前的第一张检查单：每条能力都要能被反驳
"失败族"（该谁去修 + 缺席要可断言）              线上排障时，先问"这个症状该谁去修"，
                                                再问"今天这一族是在场还是缺席、理由是什么"
```

而 M7 与 M8 交班的那句话也在这里被接住：

```text
M7  注意力 = 一次打分的点积 + 一次加权的求和
M8  反向传播 = 上游梯度 × 局部导数，然后累加；一条训练管线 = 把流程也变成可断言的算术
→   M7 决定"能不能表达"，M8 决定"能不能学会、能不能复现"；
    两者在下游合流成同一个可运行的系统（day099）。
```
