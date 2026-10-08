# 神经网络基础手册：从神经元到 FFN（day089 / M8-D1）

> 本课从最小的零件搭起一条链：**一个神经元 → 一层 → 一个网络 → 一个损失**，
> 最后把它接回 Transformer 的 FFN。
>
> 手册里的每一个数都来自本课快照内的可复现读数：
> `python scripts/neural_basics_demo.py`（十一节，产出 `outputs/neural_basics_demo.txt`）
> 与 `python -m pytest`。权重由 LCG 生成、输入写死，因此同一个数可以被重新跑出来。

---

## 一、本课只承诺一件事：**前向与损失**，不承诺反向

```text
承诺      六个激活的前向、一层全连接、一个 MLP 的前向、三个损失
不承诺    反向传播（一行反向都没有）、自动微分（本仓库不依赖 torch）、
          任何"训练效果"或"跨平台位逐位一致"
```

因此本课第 2 节的表格把 PyTorch 的 API 一一列出，**只核对公式与语义**，
不是一次"实测 torch 输出"；具体版本语义以官方文档为准。

还有一条边界要写在最前面：

```text
今天有新式子（六个激活的前向、三个损失），它们每一个都可微，看上去"该有梯度"了；
但本课手写反向一行都没有，也没有调用自动微分。反向传播是 day090 的主题。
因此连续第八天没有 GradientError 那一族（理由见第九节）。
```

---

## 二、纯 Python ↔ PyTorch 对照表（**只对照，不调用**）

| 本课的量 | PyTorch API | 口径要点 |
| --- | --- | --- |
| 单个神经元 | `torch.nn.Linear(1, 1)` | 一次点积 + 偏置（+激活） |
| 一层全连接 | `torch.nn.Linear` | 权重 `(out, in)`、前向 `x·Wᵀ + b` |
| relu | `F.relu` / `nn.ReLU` | `max(0, x)` |
| leaky_relu | `F.leaky_relu` / `nn.LeakyReLU` | 负半轴留一条坡度 |
| sigmoid | `torch.sigmoid` / `nn.Sigmoid` | 本课按符号分支写，两端不上溢 |
| tanh | `torch.tanh` / `nn.Tanh` | 唯一零中心的激活 |
| gelu | `F.gelu` / `nn.GELU` | 本课用**精确 erf 式** |
| softmax | `F.softmax(dim=-1)` / `nn.Softmax` | 先减最大值 |
| log_softmax | `F.log_softmax(dim=-1)` / `nn.LogSoftmax` | 交叉熵的稳定路径 |
| mse | `F.mse_loss` / `nn.MSELoss` | 逐元素平方差的均值 |
| mae | `F.l1_loss` / `nn.L1Loss` | 逐元素绝对差的均值 |
| cross_entropy | `F.cross_entropy` / `nn.CrossEntropyLoss` | 从 **logits** 出发 |
| 初始化 | `nn.init.zeros_ / uniform_ / normal_ / xavier_uniform_ / kaiming_normal_` | LCG 可复现 |

演示读数（对照表里抽出的几项，见第 11 节）：

```text
neuron           → 单个神经元 ≈ torch.nn.Linear(1, 1)（含偏置）
dense_forward    → torch.nn.Linear（x·Wᵀ + b，权重形状 (out, in)）
gelu             → torch.nn.functional.gelu / nn.GELU
xavier_init      → torch.nn.init.xavier_uniform_
```

---

## 三、一条链总览：神经元 → 一层 → 网络 → 损失

```text
神经元   一次点积 + 一个偏置 + 一个非线性        neuron(3→1) 参数 = 3 + 1 = 4
一层     权重 (out, in)、前向 x·Wᵀ + b          dense(3→4)  参数 = 12 + 4 = 16
网络     逐层 Dense → 激活                       MLP 4→8→3   参数 = 67
损失     一个标量 + 一个平均方式                 mse(完美) = 0.0
```

演示读数：

```text
一层账：{'in_features': 3, 'out_features': 4, 'weight_shape': '(4, 3)', 'weights': 12, 'biases': 4, 'total': 16}
网络：MLP 4→8→3 | 2 层 | 参数 67
前向账：4 → 8 → 3 | 层形状 4 -> 8、8 -> 3
第 1 层 dense(4→8) | W (8, 4) | 权重 32 + 偏置 8 = 40
第 2 层 dense(8→3) | W (3, 8) | 权重 24 + 偏置 3 = 27
损失：mse(完美) = 0.0（恰好 0）；mse(样本) = 6.250000e-01
```

参数公式**按定义算**，不四舍五入：一层是 `in×out + out`，一个网络是各层之和。

---

## 四、六个激活：一条口径表 + 三个数值稳定点

```text
relu        max(0, x)                            不饱和、非零中心、[0, +∞)
leaky_relu  x if x > 0 else 0.01x                不饱和、非零中心、(-∞, +∞)
sigmoid     1 / (1 + e^{-x})                     饱和、  非零中心、(0, 1)
tanh        (e^x - e^{-x}) / (e^x + e^{-x})      饱和、  零中心、 (-1, 1)
gelu        0.5x(1 + erf(x/√2))                  不饱和、非零中心、[-0.17, +∞)
softmax     e^{z_i - max} / Σ_j e^{z_j - max}    不饱和、非零中心、(0, 1) 且行和为 1
```

三个数值稳定点（每一个都对着一个可观察的坏结果）：

```text
sigmoid   按符号分支：x>=0 用 1/(1+e^{-x})、x<0 用 e^x/(1+e^x)  ⇒ 两端都只算 e^{≤0}，不上溢
softmax   先减最大值（恒等变形）                                ⇒ z=1000 不会去算 e^{1000}（CPython 会抛 OverflowError）
gelu      采用精确 erf 式                                        ⇒ 与 encoder_decoder / hf_source 同为逐位
```

演示读数（有限性网格有 11 点、含 `x=±1000`，全部有限）：

```text
relu        非有限个数=0  min=+0  max=+1000
sigmoid     非有限个数=0  min=+0  max=+1
tanh        非有限个数=0  min=-1  max=+1
gelu        非有限个数=0  min=-0.158655  max=+1000
softmax     非有限个数=0  min=+0  max=+1
act(1)：sigmoid=+0.731059、tanh=+0.761594、gelu=+0.841345
```

`gelu` 在 `x=1` 处的读数 `+0.841345` 与精确式 `0.5·1·(1+erf(1/√2))` 一致——
它不是 tanh 近似（`gelu_tanh` 另列，两侧各有 O(1e-4) 偏差）。

---

## 五、值域：开区间与闭区间不是一回事

```text
relu       [0, +inf)      闭下界（relu 真的取到 0）
sigmoid    (0, 1)         双开       （x=±1000 会饱和到 0.0 / 1.0——那是定义域代价）
tanh       (-1, 1)        双开
softmax    (0, 1) 且行和 1
```

演示读数（值域网格 9 点，越界个数全为 0）：

```text
relu        值域 [0, +inf]    越界 0
sigmoid     值域 (0, 1)       越界 0
tanh        值域 (-1, 1)      越界 0
softmax     值域 (0, 1)       越界 0
softmax 行和 = 0.9999999999999999（与 1 的偏差 1.110e-16）
softmax 每项落在 (0, 1)：(min=2.046773e-09, max=0.993023)
```

**注意网格的选择**：严格的开区间断言只在**有界**的值域网格上做。
在 `x=±1000` 上，`sigmoid(-1000)` 恰好是 `0.0`——那是饱和，不是 bug；
把两种问题（"还有限吗"与"值域被遵守吗"）用同一张网格去问，就会把饱和误报成越界。

---

## 六、一层全连接与五种初始化

```text
Dense      weight (out, in)、bias (out,)、前向 x·Wᵀ + b（与 nn.Linear 一致）
初始化     LCG + 可注入种子 ⇒ 同一份 spec 两次构造逐位相同
```

演示读数（同一份 `4 → 3` 的 spec、同一个种子）：

```text
zeros     W[0][0]=+0.000000 b[0]=+0.000000
uniform   W[0][0]=+0.069327 b[0]=-0.416808
normal    W[0][0]=-0.026966 b[0]=+0.199829
xavier    W[0][0]=+0.128369 b[0]=-0.771779
he        W[0][0]=-0.038135 b[0]=+0.282601
xavier 半宽 a = sqrt(6/(4+3)) = 0.925820
he 标准差 = sqrt(2/4) = 0.707107
```

`zeros` 的读数全为 0，而它有一个**具体后果**：每个神经元学到的东西完全相同
（对称性不破）——因此它只适合对账，不适合训练。

`xavier` 的分母是 `fan_in + fan_out`：当两者之和为 0 时这个式子没有定义，
本包当场抛 `InitializationError`（而不是让它崩成一次 `ZeroDivisionError`）。

---

## 七、一个网络：MLP 前向与 forward_trace

```text
x → Dense_1 → act_1 → Dense_2 → act_2 → … → Dense_n →（可选 act_n）
```

``forward_trace`` 只把**接口**写下来（不跑数据）：宽度链与逐层形状。
演示读数：

```text
网络 MLP 4→8→3 | 2 层 | 参数 67
前向账：4 → 8 → 3 | 层形状 4 -> 8、8 -> 3
输入 ((0.3, -0.7, 1.1),)
affine 输出 ((-0.9379341873172151, -1.6127885871584553),)
gelu 输出   ((-0.1633310305907053, -0.08611520664135948),)
两步（affine → 逐行 gelu）与一步一致：True
```

最后一行是一条**结构性质**：带激活的一层 = 先 affine 再逐行激活，两步与一步逐位一致。

---

## 八、两条结构事实

### 8.1 恒等激活的多层网络塌缩成一个仿射映射

```text
x·W₁ᵀ + b₁  再  ·W₂ᵀ + b₂   =   x·(W₂·W₁)ᵀ + (b₁·W₂ᵀ + b₂)
```

演示读数（`3 → 4 → 5 → 2`、全恒等）：

```text
多层输出 ((0.35264764058789555, -0.06055667669721776), (3.9362572681318193, 2.689106583018261))
单层输出 ((0.3526476405878956, -0.06055667669721798), (3.936257268131819, 2.68910658301826))
最大偏差 8.882e-16 <= 容差 1e-12：True
```

这是"为什么需要非线性"的**反证**：拿掉激活，深度不增加表达力。
只要有一层带激活，塌缩就抛 `ParameterError`——因为这条等式只在仿射映射上成立。

### 8.2 前馈块 = Dense → 激活 → Dense

演示读数（同一组权重、同一输入，本包 `ffn_block` vs `encoder_decoder.layers.feed_forward`）：

```text
hidden=4、d_ff=16
[relu] 本包第 0 行 (0.7517430751343896, -0.10007198360943836, -1.2085595550078807, -0.36658339458837097)
[relu] 项目第 0 行 (0.7517430751343896, -0.10007198360943836, -1.2085595550078807, -0.36658339458837097)
[relu] 逐位不一致 0 个
[gelu] 逐位不一致 0 个
```

这就是"神经网络基础"接回 M7 的地方：Transformer 的前馈块不是新东西，
它就是一个两层网络加一个激活（先扩张 4 倍再压回）。

---

## 九、三个损失、两条路径、三条跨天对账

### 9.1 三个损失

```text
mse            mean((pred − target)²)     预测等于目标时**恰好**为 0.0
mae            mean(|pred − target|)
cross_entropy  −log softmax(logits)[target]
```

演示读数：

```text
mse            = 6.250000e-01
mae            = 7.500000e-01
cross_entropy  = 2.407606e+00
perplexity(1.0) = 2.718282
```

### 9.2 交叉熵的两条路径

```text
路径 A（稳定）  −log_softmax(logits)[target]
路径 B（直白）  −log(softmax(logits)[target])
```

演示读数：

```text
logits=(1.0, 2.0, 3.0) target=0 | A=2.407605964444 B=2.407605964444
logits=(10.0, -10.0, 0.0) target=1 | A=20.000045400960 B=20.000045400960
两条路径最大偏差 4.441e-16（容差 1e-12：一条走 sum、一条走 fsum）
```

两条路径**不是逐位**：A 的 `log_softmax` 用内置 `sum`（对齐 `sft.loss`），
B 的 `softmax` 用 `math.fsum`（对齐 `math_foundations`）。因此判据是"偏差 <= 1e-12"，
而不是"相等"——把两类判据写清楚，才不会把"恰好为 0"当成通过。

### 9.3 三条跨天对账，分别对的是哪个真实函数

```text
softmax      activations.softmax     ↔ math_foundations.linalg.softmax（day073）     逐位不一致 0 个
交叉熵       losses.cross_entropy    ↔ sft.loss.cross_entropy（生产实现）            逐位不一致 0 个
前馈块       network.ffn_block       ↔ encoder_decoder.layers.feed_forward（day079） 逐位不一致 0 个
附加         activations.gelu        ↔ hf_source.blocks.gelu_exact（day085）         11 点不一致 0 个
```

三条都调用**别的包**来算读数——自证是不成立的，跨包对账才是。

---

## 十、七条性质、失败族与后续接缝

### 10.1 七条性质（判据分四类）

```text
有限性      activations_are_finite（11 点含 ±1000，非有限个数 0）
值域        activation_ranges_are_respected（越界 0；softmax 行和偏差 1.110e-16）
跨天对账①  softmax_rows_are_distributions（与 math_foundations 逐位不一致 0 个）
结构事实    identity_stack_collapses_to_affine（最大偏差 8.882e-16 <= 1e-12）
损失        mse_is_zero_at_perfect（mse(P,P)=0.0；与手算偏差 0.000e+00）
跨天对账②  cross_entropy_paths_agree（与 sft.loss 逐位不一致 0 个；两条路径偏差 4.441e-16）
跨天对账③  ffn_matches_transformer_stack（relu / gelu 逐位不一致 0 个）
全部通过：True
```

`CrossCheck` 带 `upper_bound` 字段：有它时判据是"≤"，没有时才是"=="。
把两类混成一个判据，就会出现"偏差恰好是 0（因为输入全是 0）被当成通过"。

### 10.2 失败族与缺席的 `GradientError`

```text
ShapeError           改调用：张量宽度、权重形状、样本与标签数量先对齐
ParameterError       改调用：激活名 / 初始化名 / 种子 / 层数
NumericError         改数据或改实现：nan / inf / 越界读数
ActivationError      改激活或改定义域
LossError            改数据或改损失：空样本、标签越界、概率为 0 取 log
ForwardError         改结构：层与层宽度接不上
InitializationError  改初始化尺度：xavier 的分母为 0
```

本课**连续第八天**没有 `GradientError`，理由与前面四条**都不同**：

```text
day085  只读别人的推理路径
day086  装进来的层都是别人写好的
day087  量化不可微
day088  本日一个新式子都没有
day089  今天有新式子（前向与损失），但一行反向都没有——反向传播是 day090 的主题
```

### 10.3 五条边界与后续接缝

```text
① 本包实现的是前向与损失，一行反向都没有
② 不依赖也不调用 torch / numpy；对照表只核对公式与语义，不声明版本
③ 六个激活是课程口径下的那一组，不是"所有激活函数"
④ 权重由 LCG 生成、输入写死：复现的是读数，不是统计规律
⑤ 不承诺"跨平台位逐位一致"——跨天对账的容差由各自口径给出
```

接缝：

```text
day073（数学地基）→ day075（可微注意力）→ day079（前馈块）→ day085（HF 源码）
→ day087（推理三笔账）→ day088（十二块拼图）→ day089（今天：从神经元重新搭起）
下游 day090  给这条链补上唯一缺席的那一族 GradientError（反向传播）
```
