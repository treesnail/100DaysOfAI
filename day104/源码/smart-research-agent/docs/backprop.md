# 反向传播手册：在一条链上取回每一点的导数（day090 / M8-D2）

> day089 把「一个神经元 → 一层 → 一个网络 → 一个损失」这条链的**前向**搭好，
> 并明写「一行反向都没有」。这一课补上那一行，并第一次让 `GradientError` 被真的抛出。
>
> 手册里的每一个数都来自本课快照内的可复现读数：
> `python scripts/backprop_demo.py`（十一节，产出 `outputs/backprop_demo.txt`）
> 与 `python -m pytest`。权重由 LCG 生成、输入写死，因此同一个数可以被重新跑出来。

---

## 一、本课只承诺一件事：**一阶反向**

```text
承诺      六个激活的局部导数、两个损失的梯度、一层 / 一个网络 / 一个前馈块的逐层回传、
          一张张量级计算图（反向模式自动微分）、一次用解析反奔跑的下降
不承诺    二阶导数（Hessian）、任意计算图的自动微分、GPU / 低精度、
          任何"训练到收敛"的结果或"跨平台位逐位一致"
```

三条边界要写在最前面：

```text
① 自动微分只覆盖一个 MLP 用得到的算子（matmul / 加偏置 / 五个逐元素激活 / softmax / mse / 交叉熵）
② 本仓库不依赖 torch，也不依赖 numpy：PyTorch 的语义只写在对照表里（见第二节），
   核对的是公式与语义，不是"实测 torch 输出"；具体版本以官方文档为准
③ 数值差分在这一课的角色是**尺子**，不是实现：它慢、只到 1e-9，但不依赖任何推导
```

---

## 二、纯 Python ↔ PyTorch 对照表（**只对照，不调用**）

| 本课的量 | PyTorch API | 口径要点 |
| --- | --- | --- |
| 一次反向 | `torch.Tensor.backward()` | 从标量损失出发的反向模式自动微分 |
| 清零 | `torch.optim.Optimizer.zero_grad()` / `param.grad.zero_()` | 图可复用，梯度不会自己清零 |
| 计算图 | `torch.autograd.Function` + `Tensor.grad_fn` | 本课是"张量级"的同一件事 |
| 保留图 | `backward(retain_graph=True)` | 同一张图反两次时必须显式说明 |
| 一层反向 | `nn.Linear` 的 grad | `grad_weight = dYᵀ·X`、`grad_bias = dY 的列求和` |
| relu 反向 | `torch.nn.functional.relu` 的导数 | `x > 0` 原样通过，否则恰好 0 |
| sigmoid 反向 | `torch.sigmoid` 的导数 | `σ(1−σ)`，上界 0.25 |
| tanh 反向 | `torch.tanh` 的导数 | `1 − tanh²`，上界 1 |
| gelu 反向 | `torch.nn.functional.gelu` 的导数 | **需保留输入**（不是只看输出） |
| softmax 反向 | `torch.nn.functional.softmax` 的反向 | 内部同样不建 n×n 雅可比 |
| mse 反向 | `torch.nn.functional.mse_loss` 的梯度 | `2(p − t)/N` |
| 交叉熵反向 | `torch.nn.functional.cross_entropy` 的梯度 | `p − onehot` |
| 梯度校验 | `torch.autograd.gradcheck` | 用数值差分校验解析梯度 |

演示读数（见第 11 节）：

```text
backward           → torch.Tensor.backward()（从标量损失出发做反向模式自动微分）
dense_backward     → nn.Linear 的 grad_weight = dYᵀ·X、grad_bias = dY 的列求和
softmax_backward   → torch.nn.functional.softmax 的反向（内部同样不做 n×n 雅可比）
grad_check         → torch.autograd.gradcheck（用数值差分校验解析梯度）
```

---

## 三、一条规则与一条纪律

```text
规则      上游梯度 × 局部导数
纪律      累加（+=）—— 一个值被用了 k 次，它的梯度是 k 条路径之和
```

前半句是链式法则，后半句才是真正会绊倒人的地方：

```text
y = x·x        两条边都通向 x ⇒ dy/dx = x + x = 2x
写成 = 而不是 +=   不报错，只给出一个**偏小**的梯度
                  而"梯度偏小"看起来像"学习率设小了"——于是有人去调学习率
```

本课把这个失败独立成一族（`ChainError`），并在 :func:`backprop.graph.Node.zero_grad`
与 :func:`backprop.graph._accumulate` 两处钉住它：全模块只有一处写梯度，
且一律走 `add_values`（即 `+=` 的语义）。

第二条纪律是**清零**：

```text
图可以复用，梯度不会自己清零：第二次 backward 会把梯度**加上去**
```

演示读数（`study` 的图那一节）：同一个 `sum_all` 节点连续反两次，
`x` 的梯度从 `1.0` 变成 `2.0`——"这一步的梯度"变成了"这两步之和"。

---

## 四、六个激活的导数

```text
relu        d/dx = 1 if x > 0 else 0                在 x = 0 处取次梯度 0（约定）
leaky_relu  d/dx = 1 if x > 0 else 0.01             负半轴那条坡度也出现在导数里
sigmoid     d/dx = σ(x)(1 − σ(x))                   值域 (0,1) ⇒ 上界 0.25
tanh        d/dx = 1 − tanh²(x)                     值域 (−1,1) ⇒ 上界 1
gelu        d/dx = 0.5(1 + erf(x/√2)) + x·e^{−x²/2}/√(2π)   在 0 处恰好 0.5
softmax     J[i][j] = p_i(δ_ij − p_j)               一整行一起看（唯一逐行的激活）
```

现场读数（`x = 1.0`）：

```text
relu        +1.000000
leaky_relu  +1.000000
sigmoid     +0.196612       （不是 0.25：0.25 只在 x = 0 处取到）
tanh        +0.419974
gelu        +1.083315
softmax     每行和 max|Σ J| = 8.327e-17   （理论上恰好 0）
```

两条工程细节值得单独写下来：

```text
一  σ' 与 tanh' 可以**用前向输出**直接算（σ(1−σ)、1−y²），不必再算一次 exp
   而 gelu' 里有 x·e^{−x²/2} 这一项 —— **它必须知道激活前的值**
   因此逐元素反向的签名里带的是 pre_activation，不是输出
二  relu 在 x = 0 处取 0 是一个**约定**，不是事实
   取 1 也"说得通"（右侧导数），但那样一个恒为负的输入会一直拿到梯度，
   与"不激活就不学习"的直觉相反
```

---

## 五、softmax：局部导数是一张矩阵，但**从不真的建出来**

```text
J[i][j] = p_i(δ_ij − p_j)                 显式：n² 个元素
(Jᵀv)_i = p_i(v_i − ⟨p, v⟩)                反向真正要算的东西：n 个元素
```

第二条的推导只有三行：

```text
(Jᵀv)_i = Σ_j J[j][i]·v_j
        = Σ_j p_j(δ_ji − p_i)·v_j
        = p_i·v_i − p_i·Σ_j p_j·v_j
        = p_i(v_i − ⟨p, v⟩)
```

本课把 `softmax_jacobian` 也实现了，但**它的唯一用途是被对照**（第 3、4 条性质）；
生产路径只走那条 O(n) 的 JVP。演示读数：

```text
概率 (0.2, 0.3, 0.5)、回传梯度 (0.5, -1.0, 2.0)
显式雅可比 ((0.16, -0.06, -0.10), (-0.06, 0.21, -0.15), (-0.10, -0.15, 0.25))
每一行之和 [2.776e-17, 0.0, 0.0]（理论上全为 0）
Jᵀv（显式）     (-0.06, -0.54, 0.6)
p⊙(v − ⟨p,v⟩)  (-0.06000000000000001, -0.54, 0.6)
最大偏差 1.388e-17 <= 容差 1e-12
```

**一个真实的坑**：`softmax_jacobian` 的入参是**概率**，不是打分。把 logits 传进去
它照样返回一张"看起来很像雅可比"的矩阵（对角线上是 `z_i(1−z_i)` 这样的数），
而它描述的是另一个函数。本课真的踩过一次——写 `study` 的导数表时直接传了
`(1.0, 2.0, 3.0)`，表里于是印出"每行和 = 1.500e+01"，而理论上它必须恒为 0。
这个数当时**没有报错**，只是安静地错着（详见第十节的第三次记录）。

---

## 六、两个损失的梯度

```text
mse             ∂L/∂pred  = 2(pred − target)/N      N = **全部元素**的个数
cross_entropy   ∂L/∂logits = p − onehot(target)      p = softmax(logits)
```

演示读数：

```text
mse_grad(((1,2),(3,4)), ((1.5,1.5),(4,3))) = ((-0.25, 0.25), (-0.5, 0.5))    （N = 4）
cross_entropy_grad((1,2,3), 0)  = (-0.909969, 0.244728, 0.665241)
cross_entropy_grad((-1,0,1), 2) = ( 0.090031, 0.244728, -0.334759)
```

两条容易写错的地方，各有一句"不报错但结果错"的说明：

```text
MSE 少一个 2        梯度整体减半 ⇒ 看起来像"学习率需要加倍"
CE 忘记除以行数     梯度大 rows 倍 ⇒ 看起来像"这一批的学习率要除以 batch size"
```

而这一课最想讲清的是**交叉熵的梯度把 softmax 的雅可比消掉了**：

```text
∂(−log p_k)/∂z = p − onehot(k)      ← 中间那一大堆项全部抵消
```

因此「softmax 的雅可比写错」这件事，**在交叉熵的梯度上看不见**——
正确的实现也不直接用它。它必须被第 3 条性质**单独**钉住。
这不是"多验一遍"，而是"只有这一遍能发现它"。

---

## 七、一层与一个网络的反向

一层全连接的三块梯度：

```text
grad_weight  (out, in)  = dYᵀ·X            回传梯度的每一列 · 输入
grad_bias    (out,)     = Σ_i dY[i]        偏置被所有行共用 ⇒ 按**行**求和
grad_inputs  (rows, in) = dY·W             按 W 的**列**收，而不是按行
```

第三条是最容易把下标搞反的地方：

```text
正确   dx_c = Σ_r dY_r · W[r][c]      遍历的是 W 的**列**
写错   dx_c = Σ_r dY_r · W[c][r]      形状仍然对（都是 (rows, in)），数值全错
```

演示读数（`dense(3→4)` + relu）：

```text
输入 ((0.5, -1.0, 2.0), (1.0, 0.25, -0.5))
回传梯度 ((0.5, -1.0, 0.25, 2.0), (-0.5, 0.75, 1.0, -0.25))
dW ((0.0, 0.0, 0.0), (0.75, 0.1875, -0.375), (0.125, -0.25, 0.5), (0.75, -2.0625, 4.125))
db (0.0, 0.75, 0.25, 1.75)
第 1 层  W (4, 3) | ‖dW‖=4.785231e+00 ‖db‖=1.920286e+00 ‖dx‖=2.120846e+00
```

一个网络（`3 → 5 → 2`）的逐层回传：

```text
第 1 层  W (5, 3) | ‖dW‖=4.161506e-01 ‖db‖=2.680254e-01 ‖dx‖=2.298692e-01
第 2 层  W (2, 5) | ‖dW‖=4.597281e-01 ‖db‖=5.809431e-01 ‖dx‖=7.297891e-01
```

**为什么记范数而不是全部元素**：一层 `(5, 3)` 的权重有 15 个偏导数，全印出来
读不出"这一层在学什么"；而三个范数能一眼看出量级差——那是梯度爆炸最早的可读信号。

---

## 八、三条独立路径

```text
路径 A   手写反向（layers.dense_backward + network.mlp_backward）
路径 B   图自动微分（graph 里那张张量级计算图）
路径 C   数值差分（day074 的 calculus.gradient，**不依赖任何推导**）
```

三条路径在同一个点上给出同一个数，是这一课唯一真正的判据：

```text
A vs B   手写反向 32 个偏导数 vs 图自动微分：最大偏差 0.000e+00（**逐位一致**）
A vs C   解析梯度 vs 中心差分：最大相对误差 5.813e-11（容差 1e-5）
```

为什么需要 B：路径 A 与路径 C 各自都有系统偏差——A 可能把某条公式抄错，
C 的分辨率只到 1e-9 且对不可导点无定义。B 是**第三份独立实现**，
它与 A 共享约定（同一组形状、同一批算子）但与 A 不共享任何一行反向代码。

`GradientError` 正是在"三条路径里有两条不一致"时被抛出来的。

---

## 九、七条性质

```text
1  activation_derivatives_match_numerical    五个逐元素激活的解析导数 vs 中心差分（<= 1e-6）
2  relu_subgradient_is_zero_at_origin        x = 0 处取 0：恒负输入的梯度**恰好**是 0
3  softmax_jacobian_matches_numerical        本包雅可比 vs day074 数值雅可比（并读它的结论）
4  softmax_jvp_avoids_matrix                 (Jᵀv)_i = p_i(v_i − ⟨p,v⟩)（<= 1e-12）
5  cross_entropy_gradient_is_p_minus_onehot  p − onehot vs 对 sft.loss 的数值差分
6  mlp_backward_matches_numerical            整条 MLP 的参数梯度 vs 数值差分（<= 1e-5）
7  ffn_backward_matches_encoder_decoder      **逐位**一致（0 个不一致）
```

判据只有两类，因此 `GradientCheck` 带一个 `upper_bound` 字段：

```text
相等（逐位 / 整数 / 计数）   第 2、7 条的读数是**恰好 0**
上界（<=）                   第 1、3、4、5、6 条是"相对误差不超过某个界"
```

把两类混成一个判据，就会出现"偏差恰好是 0（因为输入全是 0）被当成通过"这种事。

容差从哪来（不是拍一个数）：数值差分自身的误差下限是 `eps·|f|/(2h)`：

```text
|f| ≈ 1      h = 1e-6   →  1.110e-10      容差 1e-6 是它的 9000 倍
|f| ≈ 100    h = 1e-6   →  1.110e-08      仍在容差之内
```

低于这个下限的容差只会产出假警报，而假警报比漏报更坏——它会让人去改一份正确的代码。

---

## 十、失败族：**GradientError 回来了**

```text
day075~081   有手写反向            ⇒ 有 GradientError 那一族
day082~089   没有手写反向          ⇒ 这一族连续缺席八天
day090       有手写反向 + 自动微分  ⇒ **它回来了**
```

今天的理由与前八天**都不同**：不是"没有它"，而是"有两条**独立**算出梯度的路径"。
于是"两条路径不一致"第一次成为一个**控制流事件**——它不是报告里的"分歧 1"，
而是必须当场停下的 `GradientError`。

本课第一次让 `ABSENT_FAMILY` 为空：

```text
RETURNED_FAMILY = "GradientError"
ABSENT_FAMILY   = None          （不是"没有理由"，而是"没有缺席者"）
```

八个族各自要改的东西都不同：

```text
ShapeError            改调用：回传梯度与缓存的形状、压平后的长度
ParameterError        改调用：激活名 / 扰动步长 / 容差
NumericError          改数据或改实现：nan / inf 混进梯度
GradientError         改推导：解析式与数值差分对不上
BackwardError         改调用：cache 必须是**同一次前向**的产物
ChainError            改推导：用 += 累加、常量不接收梯度、重复反向前清零
StepError             改调用：梯度合法不代表这一步走完还合法
```

### 同族坑的第三次记录

```text
day088   把 Q 与 K 写成互为转置 ⇒ 报的是形状错误，踩的却是**定义**错误
day089   在中文里写 ASCII 引号 ⇒ 报的是 SyntaxError（少逗号），踩的却是**定界符**错误
day090   把 logits 当概率传给 softmax_jacobian ⇒ **什么都不报**，
         只是"每行和 = 1.5"这个读数安静地不是 0
```

第三次与前两次不同：**它连报错都没有**。前两次至少还有一个栈可以读，
第三次只有一个"看起来合理"的数字。这说明同一族问题的三种表现：

```text
报错的        最好，顺着栈就能找到
报错但报错位置很远的   要问"报错信息的局部症状是不是病灶"（day088 / day089）
不报错的       最难，只能靠**判据**——所以第 3、4 条性质里那句"每行和必须为 0"是有用的
```

---

## 十一、与既有包的接缝

```text
上游（本包调用的真实实现）
  neural_basics（day089）       activations / layers.Dense / losses / network
  math_foundations（day073/074） calculus.gradient / jacobian / gradcheck / optim
  sft（day050 起）              loss.cross_entropy（生产损失）
  encoder_decoder（day079）      layers.feed_forward / feed_forward_backward
  transformer_core（day075）     GradientError 等四族
脚下
  config.py **没有**新增配置项——激活名、步长、容差、种子都是函数参数
下游
  day092（优化器）会沿着同一条链问"走多远"；今天只回答"往哪走"，
  而且要求它可被三条独立路径同时确认
```

一句话总结这条接缝：

```text
day073 给了"数值差分是尺子、自动微分是推导的执行"；day074 把三者对上一次；
day079 写下了项目里真实的前馈反向；day089 把一条链的前向从最小零件搭起来；
day090 **在每个零件上各取一次导数**——再用 day074 的尺子量一遍，
        用 day079 的真实前向反向对一遍，用一张自己的计算图再算一遍。
```

今天最值得带走的一句话，与前几天那几句对着看：

```text
day087  缓存不是"变快"而是"换"；位置必须由**缓存长度**决定
day088  原理只有落到一个**能被解析的函数**上、并且能被**现场量出读数**，才算真的串起来了
day089  深度本身不产生非线性：让"多层"区别于"一层"的，是**那个激活函数**
day090  反向传播 = **上游梯度 × 局部导数，然后累加**——
        前半句是链式法则，后半句是纪律；而纪律那半句写错时，程序不会报错。
```
