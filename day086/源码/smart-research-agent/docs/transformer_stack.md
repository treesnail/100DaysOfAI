# `transformer_stack` 手册（day080 / M7-D5）

> 本文件是 day080 的权威参考：把 day079 交出的**一个块**复制成一条链。
> 全部读数都来自 `scripts/transformer_stack_demo.py` 的真实输出（十一节、198 行），
> 样本是 4 层 / d=6 / d_ff=24 / n=4 / seed=7。

## 目录

```
1   一天之内新增的结构：一个 for
2   三个“漏掉不会报错”的地方
3   形状与参数量：342 / 144 / 486 / 1944
4   逐层读数：增益与直通占比
5   链式反向：那一行换手
6   两处梯度校验（1 项 + 8 项）
7   六条性质
8   PyTorch 组装：生成而不是导入
9   参数量的差异：4d 从哪来
10  堆叠实验：每层倍数与深度无关
11  几何平均的那条恒等式
12  边界与踩过的坑
```

---

## 1 一天之内新增的结构：一个 `for`

```text
day079   块 = LN → 注意力 → ⊕ → LN → 前馈 → ⊕        六个阶段，九个形状断言
day080   链 = 块 × N                                  ——今天的全部结构
```

`layers.py` 里真正新增的代码只有三段：

```python
# 前向
attention = block_attention(block_params, current, attention_params, placement=placement)
forward = encoder_block(block_params, current, attention, placement=..., use_residual=..., activation=...)
current = forward.output                      # ← 第一段：把这个值交给下一层

# 反向
layer_grads = encoder_block_backward(layer.block, layer.params, current, activation=...)
grads[index] = layer_grads
current = layer_grads.grad_inputs             # ← 第二段：换手

# 记账
censuses.append(LayerCensus(index=index, input_norm=frobenius(current), ...))   # ← 第三段
```

五张口径表（`types.py`）：

```text
STACK_STAGES             enter / block / census / carry / exit
STAGE_SHAPES             每个阶段两侧的形状（**每一阶段都保形**）
STACK_PROPERTIES         六条性质
STACK_GRADIENT_TARGETS   {stack: (stack_inputs,), layer: 八块参数}
STACK_NOTES              三条边界
```

---

## 2 三个“漏掉不会报错”的地方

| 位置 | 正确写法 | 漏掉/写错的后果 | 谁会抓到 |
|------|---------|----------------|---------|
| 前向传递 | 把 `y_i` 交给下一层 | 重算一遍：数值差 `1e-16` 量级，而“逐位一致”失效 | 性质 2 |
| 反向换手 | `current = grads.grad_inputs` | 每层都从 `dLoss/dy_N` 起步：**形状全对**，第 0 层 `dW_in` 相对差 `2.18e-01` | 第 6 章的八项校验 |
| 范数求和 | `math.fsum` | 跨层比较的两个数口径不同（链长了会累积 `1e-13`） | 读数表本身 |

第二条是本课唯一“值钱”的陷阱，它的实测代价写在第 11 章。

---

## 3 形状与参数量：342 / 144 / 486 / 1944

```text
d = 6，d_ff = 4d = 24，n = 4，N = 4

块参数     4d（两个 LN 的 γ/β）+ 2·d·d_ff（两层线性）+ d_ff（b_in）+ d（b_out）
           = 24 + 288 + 24 + 6 = 342
注意力     4d² = 4 × 36 = 144          （day075 起就在那里，今天不新增）
一层       342 + 144 = 486
整条链     4 × 486 = 1944
```

`StackShape` 把这三个量做成**三个属性**（`block_parameter_count` /
`attention_parameter_count` / `layer_parameter_count`），理由是它们的来源不同：
前两块由不同的包贡献，而“合计”只是层数乘上去。第 7 条性质比较的就是
**解析式**与**逐块数出来的**这两个数——只写一边时，“公式错了”与“构造漏了一块”无法区分。

`flatten()` **只压平块参数**（`4 × 342 = 1368`）：注意力那四个投影由 `unflatten`
原样带回，因为“需要被逐分量更新的”与“只是被携带的”是两件事。

---

## 4 逐层读数：增益与直通占比

样本（0.25 初始幅度、4 层、残差开）的实测表：

```text
  层 |         ‖x‖ |         ‖y‖ |        增益 |      ‖分支一‖ |      ‖分支二‖ |     直通占比
  0 |    0.781025 |    1.747158 |  2.237007 |   1.114224 |   0.764351 |   0.2937
  1 |    1.747158 |    2.078160 |  1.189451 |   0.482152 |   0.860416 |   0.5655
  2 |    2.078160 |    1.687401 |  0.811969 |   0.890196 |   0.450274 |   0.6079
  3 |    1.687401 |    1.932848 |  1.145458 |   0.402814 |   0.561668 |   0.6363
```

三个口径：

```text
gain         ‖y_i‖ / ‖x_i‖                       这一层把整体尺度放大了多少倍
carry_share  ‖x_i‖ / (‖x_i‖ + ‖b1‖ + ‖b2‖)        输出里有多少来自残差那条直通路
（以及）      max_abs_input                        一个**单点**读数（范数是平方和，不是同一个东西）
```

两条必须写下来的约定：

1. `gain` 在 `‖x‖ = 0` 时返回 **`0.0`**，不返回 `nan`（会静默污染整张表）也不返回 `1.0`
   （看起来像“这一层什么都没干”）。`0.0` 在一张增益表里是**显眼的**。
2. `carry_share` 在**残差关**时恰好返回 `0.0`——`use_residual` 被记进读数里。
   否则同一张表里的 `0.30` 既能指“三成来自直通”，也能指“残差其实关着”。

样本上第 0 行的 `‖y_0‖ = 0.781025`，而它恰好等于第 1 行的 `‖x_1‖`——这不是巧合，
而是“链上传递的是同一个值”在读数表上的样子（第 7 条性质逐层核对它）。

---

## 5 链式反向：那一行换手

```python
grads: list[BlockGradients | None] = [None] * forward.depth
current = checked_grad
for index in reversed(range(forward.depth)):
    layer = forward.layer_at(index)
    layer_grads = encoder_block_backward(layer.block, layer.params, current, activation=...)
    grads[index] = layer_grads          # 先存账（按层号）
    current = layer_grads.grad_inputs   # 再换手
```

`StackGradients` 在构造时钉死两件事：

```text
① 每一环的 grad_inputs 与整条链的 grad_inputs **同形**
② grad_inputs 恒等于第 0 环的输入梯度——否则“整条链的输入梯度”与“最底层那一环”
   会指两个不同的东西
```

样本（4 层）的逐层读数：

```text
   层号 |        ‖dx‖ | 与上一层的比
      0 |    0.317052 |            —
      1 |    0.286225 |     0.902771
      2 |    0.257685 |     0.900288
      3 |    0.227080 |     0.881232

   链内首尾之比 ‖dx_0‖ / ‖dx_3‖ = 1.396209
   整体比       ‖dx_0‖ / ‖dy_N‖ = 1.561036（分母 ‖dy_N‖ = 0.203104）
```

这两个量**不是同一个数**，而它们的差别正是第 11 章那条恒等式要讲的事：
`dy_N` 不在 `dx` 序列里。

---

## 6 两处梯度校验（1 项 + 8 项）

```text
kind=stack  名单 (stack_inputs,)              整条链的输入梯度
kind=layer  名单 八块参数（不含 'inputs'）      第 k 层的参数梯度
```

容差 `1e-6`（与 day075~079 同值），实测：

```text
stack 梯度校验 1 项：通过 1、失败 0 | 最大相对误差 8.23e-11
    stack_inputs 8.23e-11

layer 梯度校验 8 项：通过 8、失败 0 | 最大相对误差 7.92e-11
    norm1_gamma 5.36e-11 | norm1_beta 7.89e-11 | ffn_w_in 7.92e-11 | ffn_b_in 3.99e-11
    ffn_w_out 6.20e-11   | ffn_b_out 2.05e-11 | norm2_gamma 4.43e-11 | norm2_beta 4.64e-11
```

**为什么 `layer` 这一份里没有 `inputs`**：中间层的输入不是自变量，它由前一层算出来。
把它当成可扰动对象，会让“这一层的输入”在链上指两个不同的东西。

**为什么必须在第 0 层上查那八块**：如果反向写成“每层都从 `dLoss/dy_N` 起步”，
最后一层的读数**完全正确**（它的 `grad_output` 本来就是 `dLoss/dy_N`），
因此只看最后一层会全绿；而第 0 层会明确地不对。

---

## 7 六条性质

| 性质 | 判据 | 样本实测 |
|------|------|---------|
| `stack_preserves_shape_at_every_layer` | 逐层形状 + 传递 + 出口，全部 `==` | 4 层逐层同形、3 处传递逐位一致、出口逐位一致 |
| `stack_matches_a_hand_written_loop` | 两条路径**逐位**相等 | 4 层手写循环与 `stack_forward` 逐位相等 |
| `stack_is_deterministic` | 两次调用输出与读数逐位相同 | 输出 True、4 行读数 True |
| `stack_is_the_identity_when_every_branch_vanishes` | 每一层两个分支全零 ⇒ 输出 `==` 输入 | 4 层逐位相等 |
| `analytic_parameter_count_matches_construction` | 解析式 `==` 逐块数 | 1944 == 1944、层数 4 |
| `generated_torch_assembly_carries_the_same_shape` | 脚本 `CONFIG` 逐键对齐 | 缺 无、对不上的 无、类 True、模块 True |

三条必须说清的地方：

```text
① 性质函数自己负责关那两个分支（前馈的 w_out/b_out + 注意力的四个投影），
   因此“传一份只关了一半的参数进来”**也会通过**。负例必须另外做——
   把负例挂在性质上会得到一个永远为真的断言。
② 残差那一半**不能关**：关掉之后 y = F(LN(x)) 在分支全零时退化成零矩阵，
   而不是 x。这条反证证明性质测的确实是残差那条 +1 路。
③ 与手写循环一致这一条用 == 而不是容差：两条路径做的是同一串浮点运算。
   把参数错位一层会让它立刻亮红——而那类错误**不会**以形状错误的形式出现。
```

---

## 8 PyTorch 组装：生成而不是导入

```text
包里的代码     纯 Python，**不 import torch**（与 day073~079 同一条纪律）
生成的脚本     assembly_script(shape, placement=, activation=, num_heads=, batch=)
结构核对       assembly_facts(script) → {classes, modules, config, has_main, line_count}
```

`ASSEMBLY_REQUIREMENT = "torch>=2.5"`；脚本头里写着 `pip install "torch>=2.5"`。

脚本的可核对事实（样本）：

```text
类        ('TransformerBlock', 'TransformerStack')
nn 模块   ('GELU', 'LayerNorm', 'Linear', 'Module', 'ModuleList',
           'MultiheadAttention', 'ReLU', 'Sequential')
CONFIG    {'hidden': 6, 'ffn': 24, 'tokens': 4, 'layers': 4, 'num_heads': 2,
           'epsilon': 1e-05, 'placement': 'pre', 'activation': 'relu', 'batch': 2,
           'analytic_layer_parameters': 486, 'analytic_total_parameters': 1944}
行数      83；有 main：True
```

三行最要紧的组装：

```python
self.norm1 = nn.LayerNorm(hidden, eps=float(CONFIG["epsilon"]))
self.attention = nn.MultiheadAttention(hidden, int(CONFIG["num_heads"]), batch_first=True)
self.ffn = nn.Sequential(nn.Linear(hidden, ffn), non_linearity, nn.Linear(ffn, hidden))
```

`eps = 1e-05` 与 day079 的 `DEFAULT_EPSILON` 同源；`nn.ModuleList([...])` 就是本课的“链”。

`num_heads` 必须整除隐藏维——否则 `nn.MultiheadAttention` 会在**前向**时才报错，
而那时离“写错一行”已经很远了。本包把它挡在生成那一刻。

---

## 9 参数量的差异：4d 从哪来

```text
本课解析式                    4d²             （四个 (d, d) 投影，**没有**偏置）
nn.MultiheadAttention(bias=True)  4d² + 4d    （q/k/v 合进 in_proj_weight 且带 in_proj_bias，
                                             out_proj 也带偏置）
样本差值                      96 = 4 层 × 4 × 6 = N × 4d
```

生成的脚本会把两个数都打印出来，**差异必须被看见，而不是被凑平**：
本课的 `AttentionParams` 只有四块矩阵（day075 的决定），
而 PyTorch 的多头注意力把 q/k/v 合成一个 `in_proj_weight` 并加上三份偏置。

---

## 10 堆叠实验：每层倍数与深度无关

两个变体（一次只改一个旋钮）：

```text
residual   y = x + F(LN(x))     现代实现的主流
bare       y = F(LN(x))         把那一项 +x **关掉**（同一条代码路径）
```

实测（2 变体 × 6 档深度，`DEFAULT_STUDY_DEPTHS = (1, 2, 3, 4, 6, 8)`）：

```text
  变体     |   层数 |     整体比 |   每层倍数 |  输出漂移 | 参数量
  residual |    1 | 1.502475 | 1.000000 | 1.666201 |  486
  residual |    2 | 1.531613 | 0.693868 | 2.175905 |  972
  residual |    3 | 1.425915 | 0.884230 | 1.570048 | 1458
  residual |    4 | 1.561036 | 0.894712 | 1.906604 | 1944
  residual |    6 | 1.688102 | 0.902509 | 3.175736 | 2916
  residual |    8 | 1.614977 | 0.948060 | 2.604913 | 3888
  bare     |    1 | 0.209805 |  1.000000 | 1.547405 |  486
  bare     |    2 | 0.084865 | 11.097718 | 1.257904 |  972
  bare     |    3 | 0.132918 |  3.292021 | 1.164696 | 1458
  bare     |    4 | 0.089017 |  1.430753 | 2.228565 | 1944
  bare     |    6 | 0.003398 |  2.060385 | 1.549745 | 2916
  bare     |    8 | 0.000069 |  3.121139 | 1.397810 | 3888
```

三个读法：

```text
residual  整体比在 1.43 ~ 1.69 之间**来回摆**，没有随深度塌下去
          每层倍数在 0.69 ~ 0.95 之间，极差 max/min = 1.441197（< 2 ⇒ 与深度无关）
bare      整体比从 2.10e-01 一路塌到 6.90e-05（8 层）
          每层倍数自己就开始乱跳（极差 11.097718）——**逐层形状是噪声**
```

判决 `verdict_ok`：最深一层上 `bare` 比 `residual` 差一个数量级以上，
**并且**残差开的每层倍数极差 `< 2`。两条一起成立才算过。

三条边界：

```text
① 它回答“每层梯度倍数是多少、与深度有没有关系”，不回答“能训多深”
② 整体比大于 1 也可能是**放大**而不是好事——这条读数只说明那条路没被掐断
③ 这份表是一颗种子的一次观测；要下“残差一定更好”的结论需要多种子与统计检验
```

---

## 11 几何平均的那条恒等式

```text
mean_step_ratio ^ (N − 1)  ==  ‖dx_{N−1}‖ / ‖dx_0‖      ← 链**内部**首尾之比
overall_ratio              ==  ‖dx_0‖   / ‖dy_N‖        ← 链的入口与损失之间
```

4 层链上的实测：

```text
逐层 ‖dx‖ = 0.317052, 0.286225, 0.257685, 0.227080
相邻比值   = 0.902771, 0.900288, 0.881232
几何平均   = 0.894712；0.894712³ = 0.716225
‖dx_last‖ / ‖dx_0‖ = 0.716225 —— 逐位对上
（而 ‖dx_0‖ / ‖dy_N‖ = 1.561036，是另一个量）
```

这条恒等式值得写下来的理由是**第一版文档写错过它**：把 `overall_ratio`
也写成“各层 step 相乘”。而那只对**链内首尾之比**成立——`dx` 序列有 N 个数、
只有 N−1 个比值，而 `dy_N` **不在那个序列里**。

`mean_step_ratio` 用几何平均而不是算术平均，正是因为比值是相乘的：
N−1 个相邻比值的几何平均取 N−1 次方正好回到链内首尾之比。

---

## 12 边界与踩过的坑

### 12.1 一条跨天的守卫：全零行

```text
残差关 + 分支全零（前馈 w_out 与注意力四个投影都置零）
  一层    算得完：输出是全零矩阵（零矩阵是一个合法的结果）
  两层    第二层的注意力拿到**全零行** ⇒ day075 的 NumericError
```

day075 的 `self_attention` 显式拒绝全零行（它的 softmax 会给出均匀分布，
而“均匀分布”与“还没学到”看起来一样）。这不是 bug，而是两天的判据在这里接上了：
**一个把输入压成全零的堆叠，在 day075 的眼里就是一次“不该交给这一层猜”的调用。**

### 12.2 `is` 不是判据

`StackForward` 在构造时会过一遍 `validate_matrix`，而它返回的是一份**重建**的等值元组。
因此“第 i 层的输出与第 i+1 层的输入”只承诺**逐位相等**，不承诺“同一个对象”。
第一版把这条写成 `is`，于是它在 `==` 成立的地方亮红——而那不是 bug。

### 12.3 `carry_share` 在残差关时**不是** 0（第一版）

第一版把它写成纯比例的 `‖x‖/(‖x‖+‖b1‖+‖b2‖)`，与残差开关无关。
于是同一张表里的 `0.30` 既可能指“三成来自直通”，也可能指“残差其实关着”。
修法是把 `use_residual` 记进读数：关时返回 `0.0`。

### 12.4 参数量对不上不是误差（第 9 章）

`nn.MultiheadAttention` 带偏置，本课的解析式不带。差 `N × 4d = 96` 个参数。
**把它凑平**是最坏的处理方式（那样下一次差异出现时就没有参照物了）；
正确的做法是把两个数都打印出来并把差值的来源讲清。

---

## 附：与既有包的接缝

```text
上游   encoder_decoder（day079 的 encoder_block / encoder_block_backward / block_attention）
       transformer_core（day075 的 self_attention / default_parameters / mean_squared_error）
       math_foundations（day073 的 matrix_shape / validate_matrix、day074 的 calculus.gradient）
脚下   config **没有**新增配置项：层数、摆放位置、残差开关与激活都进函数参数
下游   day081（训练优化与正则化）要用这条链去真训，并且要回答 day079 留下的
       “post-LN 需要学习率预热吗”
       day082（变体架构）会看到 BERT/GPT 就是“取这摞块的哪几层”
       day085（源码精读）会看到 Hugging Face 的 TransformerEncoder 就是 ModuleList + 一个 for
```
