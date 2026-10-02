# 微积分与优化手册（`math_foundations` 的 day074 部分，Math-D2）

> 本文是 `smart_research_agent/math_foundations/` 中**微积分与优化**四个模块的权威说明。
> 教程在 [`../../教程/教程.md`](../../教程/教程.md)，演示脚本在
> [`../../scripts/calculus_demo.py`](../../scripts/calculus_demo.py)
> （输出见 `outputs/calculus_demo.txt`，十二节）。
> day073 的部分见 [`math_foundations.md`](math_foundations.md)。

## 1. 这一层为什么存在

day073 把"向量、概率、注意力"的**数学含义**钉死了。但"可微"这个词要落地，
还差三块拼图，它们各自回答一个不同的问题：

```text
往哪走      导数 / 梯度 / 雅可比            calculus.py
怎么算出来  链式法则（自动跑一遍）           autograd.py
走多远     优化器 + 学习率调度 + 梯度裁剪    optim.py
对不对     解析式与数值差分逐点对照          gradcheck.py
```

| 模块 | 回答的问题 | 行数 |
|------|-----------|------|
| `calculus.py` | 三种差分、梯度、方向导数、雅可比、步长误差实验、链式法则 | 710 |
| `autograd.py` | 计算图、拓扑序、反向累积（`+=`）、`value_and_grad` | 535 |
| `optim.py` | SGD / 动量 / Adam（含偏差修正）、四种调度、两种裁剪、训练回路 | 805 |
| `gradcheck.py` | 六项梯度对照（解析式 vs 数值差分） | 662 |

`types.py` 从 663 行长到 910 行（四张口径表 + `FLOAT_EPSILON`），
`__init__.py` 从 331 行长到 564 行（导出从 165 个到 220 个名字）。

**唯一的作者是"公式"**：每个算子的公式都写在 `types.py` 的口径表里
（`CALCULUS_METHOD_FORMULAS` / `OPTIMIZER_UPDATE_FORMULAS` / `SCHEDULE_FORMULAS` /
`GRADIENT_TARGET_FORMULAS`），导入期校验四张表逐键对齐——少一个键**不会让测试变红**，
只会让那一项在报告里失去"它可以被复核"的部分。

## 2. 三种差分方法与误差阶数

```text
前向   f'(x) ≈ (f(x+h) − f(x)) / h            截断误差 O(h)     一次额外求值
后向   f'(x) ≈ (f(x) − f(x−h)) / h            截断误差 O(h)     一次额外求值
中心   f'(x) ≈ (f(x+h) − f(x−h)) / (2h)       截断误差 O(h²)    两次额外求值  ← 缺省
```

中心差分多花一次求值换来的东西可以算出来：把两个泰勒展开相减，
`h f''(x)` 这一项**被减掉了**，剩下的第一项是 `O(h³)`，除以 `2h` 得到 `O(h²)`。

实测（`step_size_study(exp, 1.0, e)` 的前三行）：

```text
h = 1e-1    前向误差 1.406e-01    中心误差 4.533e-03
h = 1e-2    前向误差 1.364e-02    中心误差 4.530e-05
h = 1e-3    前向误差 1.360e-03    中心误差 4.530e-07
```

`observed_order()` 用相邻两条读数算 `log(e₁/e₂)/log(h₁/h₂)`，得到**前向 1.01、中心 2.00**。
阶数只在**截断误差主导区**有意义：拿触底之后的读数去算，得到的数字既不是 1 也不是 2，
而是一个没有意义的中间值。

## 3. 步长：为什么不能一直缩小

两类误差方向相反：

```text
h 变小     截断误差下降（O(h) 或 O(h²)）
           舍入误差上升（f(x+h) − f(x−h) 是两个几乎相等的数相减，有效位被丢掉）
```

交点就是实践最优步长。理论量级：

```text
中心差分     (eps / |f''|)^{1/3} ≈ 6.06e-6    （|f|、|f''| 都是 1 的量级时）
前向差分     eps^{1/2} ≈ 1.5e-8
```

**噪声地板**（`gradient` 与所有对照都必须知道这个数）：

```text
导数上的舍入误差下限 ≈ eps·|f| / (2h)
|f| ≈ 1      h = 1e-6   →  1.1e-10
|f| ≈ 30     h = 1e-6   →  3.3e-09
|f| ≈ 100    h = 1e-6   →  1.1e-08
```

`gradcheck.difference_resolution(magnitude, step)` 就是这条公式的实现。
它把"容差该取多少"从一个拍脑袋的数字变成一次可验算的计算——
`GRADIENT_TOLERANCE = 1e-8` 的理由就是"它比最坏那个案例（|f| ≈ 10）的分辨率高一个数量级"。

> 一个具体的坑：`second_difference` 的分母是 `h²`，舍入误差被放大 `1/h²` 倍。
> 对 `f(x) = 3x + 1`（真 `f'' = 0`）实测得到 `|f''| ≈ 1.8e-3`——**全是噪声**。
> 因此 `best_step_for_central` 不能判断"`|f''| <= eps`"，
> 必须与**噪声地板** `4·eps·|f|/h²` 比较。见 `TestBestStep::test_flat_function_...`。

## 4. 链式法则与反向传播

链式法则只有一条：`(f∘g)'(x) = f'(g(x))·g'(x)`。**容易错的是"在哪个点取值"**：
`f'(g(x))` 而不是 `f'(x)`。`calculus.chain_rule` 把它写成两条并排的路：

```python
overall = central_difference(composed, point)          # 对复合函数整体做差分
product = 1.0
current = point
for function in reversed(functions):
    product *= central_difference(function, current)   # 局部导数在**正确的点**上取
    current = _evaluate(function, current)
```

`autograd.Scalar` 把这件事自动化：每个节点记住"局部导数是什么规则"
（乘法的局部导数是另一个乘数、`exp` 的局部导数是它自己……），
`backward()` 沿**拓扑序从根到叶**走一遍，把 `上游梯度 × 局部导数` 累加给父节点。

三个必须写下来的约定：

```text
① 用 += 累积       一个值被用 k 次，它的梯度是 k 条路径之和（x·x → 2x，不是 x）
② propagate 吃的是**本节点的梯度**，不是左操作数的
                   —— 这一课真的踩到了：写成 self.grad 时，
                   在"只有一条链且左操作数是叶子"的例子上它甚至算出正确结果（叶子梯度初值 0）
③ 复用图前必须 zero_grad()   否则第二次 backward 会把梯度叠上去
```

计算图打印出来是这样（`y = x·x`、`x = 3`）：

```text
节点                运算                 值          梯度
x                 leaf         +3.0000     +6.0000
nodemul           mul          +9.0000     +1.0000
```

读法：`x` 的梯度是 `6 = 2×3`，因为乘法节点把 `3`（另一个乘数的值）沿着**两条边**
各回传了一份。这就是"累积"的全部内容。

## 5. 三个优化器的更新公式

| 优化器 | 公式 | 状态 | 适用场景 |
|--------|------|------|----------|
| `sgd` | `θ ← θ − lr·g` | 无 | 曲率均匀的地形；最诚实的基线 |
| `momentum` | `v ← β·v + g; θ ← θ − lr·v` | `v` | 峡谷地形（压住陡方向的横跳） |
| `adam` | `m ← β₁m + (1−β₁)g; v ← β₂v + (1−β₂)g²; m̂ = m/(1−β₁ᵗ); v̂ = v/(1−β₂ᵗ); θ ← θ − lr·m̂/(√v̂+ε)` | `m`, `v`, `t` | 参数尺度差异大的模型 |

**动量用的是"累加型"写法**（Polyak 原始形式），速度的稳态幅度是 `g/(1−β)`——
`β = 0.9` 时是梯度的 10 倍。它与 EMA 型（`v ← βv + (1−β)g`）可以通过
"把 lr 除以 (1−β)"互相换算，但**不能混着用**：混用的后果是等效学习率差一个因子，
表现为"换了优化器就要重调 lr"，看起来只是"风格不同"。

实测（`scripts/calculus_demo.py` 第 8 节）：

```text
碗形 (x−1)² + 4(y+2)²，100 步           峡谷 x² + 20y²，60 步
SGD    lr=0.05  →  损失 0.000000        SGD    lr=0.01  →  损失 0.796841
动量    lr=0.05  →  损失 0.002793        动量    lr=0.01  →  损失 0.002437
Adam   lr=0.20  →  损失 0.000131        Adam   lr=0.10  →  损失 0.029205
```

结论不是"Adam 最好"：碗形上 SGD 反而最快（它没有额外状态要维护），
峡谷上 SGD 明显最差。**"哪个优化器更好"这个问题离开地形就没有意义。**

## 6. Adam 的偏差修正：第一步恰好是 `lr·sign(g)`

两个滑动平均都从 0 出发，因此前几步它们**偏小**：

```text
m₁ = (1−β₁)g = 0.1g         只有真值的 1/10
v₁ = (1−β₂)g² = 0.001g²     只有真值的 1/1000
无修正：lr·0.1g / (√0.001·|g|) = lr·3.16      ← 比设定值大 3 倍
有修正：lr·g/(|g|+ε) ≈ lr·sign(g)             ← 恰好是 step size
```

实测（第 12 节）：

```text
梯度 g =    +1.0  →  第一步位移 -0.099999999   lr·sign(g) = -0.100000000
梯度 g =   +0.01  →  第一步位移 -0.099999900   lr·sign(g) = -0.100000000
```

**推论**：梯度恒定时每步位移都是 `lr`，与 `|g|` 无关
（`m̂/√v̂ = sign(g)`，两个修正因子恰好抵消）。这既解释了
"Adam 为什么能统一不同尺度参数的学习率"，也解释了它的副作用——
后期会在最优点附近来回抖，因此**后期需要退火**。

## 7. 四种学习率调度

`step` 一律 **1-based**（第 1 步就是第一次更新）。

```text
constant        lr(t) = base_lr
step_decay      lr(t) = base_lr · gamma^{⌊(t−1)/drop_every⌋}
cosine          lr(t) = min_lr + (base_lr−min_lr)·(1 + cos(π(t−1)/(T−1)))/2
warmup_cosine   t ≤ warmup: base_lr·t/warmup；之后按余弦从 base_lr 降到 min_lr
```

两个容易写错的地方都有具体后果：

```text
⌊(t−1)/drop_every⌋ 里的 −1     写掉它会让第 drop_every 步**提前一步**掉档
cosine 的分母用 T−1 而不是 T   用 T 时第 T 步永远差一点点到不了 min_lr
warmup 第 1 步是 base_lr/warmup，**不是 0**   写成 0 时第一次更新完全不发生，
                              日志里表现为"第 1 步 loss 与初始 loss 一模一样"，
                              看起来像"梯度算错了"
```

实测（第 9 节，`base_lr = 0.1`）：

```text
调度              t=1      t=2      t=3      t=4      t=5      t=10
constant      0.100000  0.100000  0.100000  0.100000  0.100000  0.100000
step_decay    0.100000  0.100000  0.100000  0.050000  0.050000  0.012500
cosine        0.100000  0.096985  0.088302  0.075000  0.058682  0.000000
warmup_cosine 0.025000  0.050000  0.075000  0.100000  0.093301  0.000000
```

常数调度不是"被淘汰的旧方案"，而是**基线**：没有它，"调度有没有用"无法回答。

## 8. 梯度裁剪的两个原语

```text
clip_by_global_norm(g, m)   整体缩放：‖g‖ > m 时乘 m/‖g‖，**方向不变**
clip_by_value(g, limit)     逐分量截断：会**改变方向**（(100,1) → (1,1)：从几乎水平变 45°）
```

两者都返回"缩放系数"（前者）或明确说明它做了什么（后者），因为
**"这次裁剪有没有生效"必须可读**：只给一串"看起来正常"的数，
读的人无法分辨它是没裁剪，还是已经被缩了 50 倍。

## 9. 六项梯度对照：解析式 vs 数值差分

对照的意义与 day073 的 `bridge` 相同，但**量换成了变化率**：

| 项 | 解析式 | 数值侧 |
|----|--------|--------|
| `softmax` | `∂p_i/∂z_j = p_i(δ_ij − p_j)` | `calculus.jacobian` |
| `cross_entropy` | `∂(−log p_y)/∂z = p − onehot(y)` | `sft.loss.cross_entropy` |
| `sigmoid` | `σ(1−σ)` | `alignment.objectives.sigmoid` |
| `log_sigmoid` | `σ(−x)` | `alignment.objectives.log_sigmoid` |
| `perplexity` | `d(exp L)/dL = exp L` | `sft.loss.perplexity` |
| `autograd_chain` | 链式法则（自动微分） | `calculus.gradient` |

实测（第 11 节）：六项全部 `agrees`，最大误差 `1.04e-08`。

**为什么 softmax 的雅可比要单独比一次**：它在交叉熵的梯度里**会被抵消**。
把雅可比那一列除以 `−p_y` 恰好得到 `p − onehot(y)`——
也就是说，即使雅可比写错了，"交叉熵的梯度"看起来仍然正确。
只有把雅可比单独拿出来与数值雅可比比，才能拦住它。
测试里把这个等式直接断言下来（`TestSoftmaxJacobianCancellation`）。

**两种结论，而不是三种**：`bridge` 有 `agrees/differs/rejected`，
这里只有前两种——`rejected` 的含义是"一边拒绝一边放行"，
而本模块两侧都是我们自己算的，不存在这种情况。
数值差分被拒（例如在 `perplexity` 的定义域边界 `L = 0` 上取左邻点）
说明"这个点不适合做双向差分"，应当**换掉样本**，而不是记成一条"记录的差异"。

## 10. 三个"不报错但结果错"的陷阱

```text
① 梯度用 = 而不是 +=       一个值被用两次时梯度偏小一半，
                          看起来像"学习率设小了"
② propagate 读错 self     加法把左操作数的旧梯度当上游梯度传下去；
                          在单链 + 叶子左操作数时**恰好正确**，因此极难发现
③ 复用图不 zero_grad       第二次 backward 把梯度叠上去，
                          看起来像"梯度比预期大两倍"
```

三个都有一处共同点：**它们不给异常，只给一个可以解释得通的错结果**。
因此它们的护栏都是"用一个独立的量去核对"——
`gradcheck` 的六项对照、`test_math_autograd.py` 的手算值、以及
`TestMatchesNumericDerivative` 的逐点比较。

## 11. 与既有模块的接缝

```text
上游   无（纯数学，不 import 智能体的任何模块）
脚下   config **没有**新增配置项：容差、步长、学习率曲线、β、γ 都是"这一次计算的判据"，
       它们进的是**函数参数**——写进配置会让人以为"改了容差就等于改了结论"
下游   day075     用 gradcheck 验证注意力权重的梯度、用 optim 跑一个可训练的注意力层
       day080     从零实现 Transformer 时，autograd 的链式法则就是它的全部依据
       day054     DPO 目标的 `∂(−log p)/∂z = p − onehot` 与 cross_entropy 那一项是同一条公式
       day051     LoRA 与 day073 的 low_rank_approximation 是同一件事的两种说法
       day087     高效推理与量化会回到"低秩 + 量化"，而 low_rank 的数学原型在 day073
```

## 12. 一页速查

```text
我要……                          用……
求一个数在一点的变化率            calculus.derivative(f, x)              （中心差分）
求多个变量的梯度                  calculus.gradient(f, point)
知道"沿哪个方向变化最快"          calculus.directional_derivative(...)，最大值 = ‖∇f‖
看二阶导（曲率）                  calculus.second_difference(f, x, step=1e-4)   ← 步长要大
量化"差分有多准"                  calculus.step_size_study / gradcheck.difference_resolution
手算一个复合函数的导数            calculus.chain_rule(functions, x)
让程序自己算复合函数的导数        autograd.value_and_grad(expression, inputs)
看计算图                          node.trace() / autograd.topological_order(node)
清梯度                            node.zero_grad()
走一步                            SGDOptimizer / MomentumOptimizer / AdamOptimizer 的 step()
把参数矩阵交给优化器              optim.flatten_matrices / unflatten_matrices
跑一段训练并留下痕迹              optim.minimize(objective, initial, optimizer=..., steps=...)
要一条学习率曲线                  optim.make_schedule("warmup_cosine", ...)
压住一次过大的更新                optim.clip_by_global_norm(grads, max_norm)
验证一份解析梯度                  gradcheck.check_gradients_all()
```

三条纪律，可以贴在任何一份数值工作的旁边：

```text
① 能算出来的东西就不要只用一句话带过    阶数、最优步长、分辨率都是量出来的
② 低于测量手段分辨率的容差只会产出假警报  它会让人去改一份本来正确的代码
③ 不给异常的错误最难查                  用两套独立的方法互相核对
```
