# 可训练注意力手册（`transformer_core`，day075 / M7-D1）

> 本文是 `smart_research_agent/transformer_core/` 的权威说明。教程在
> [`../../教程/教程.md`](../../教程/教程.md)，演示脚本在
> [`../../scripts/attention_demo.py`](../../scripts/attention_demo.py)
> （输出见 `outputs/attention_demo.txt`）。
> 上一课（数学地基）见 [`math_foundations.md`](math_foundations.md) 与
> [`calculus_and_optimization.md`](calculus_and_optimization.md)。

## 1. 这一层为什么存在

day073 把 ``softmax(QKᵀ/√d_k)·V`` 算得明明白白，但那份实现里的 Q/K/V 是**输入**。
今天把 Q/K/V 换成参数投影出来的四个矩阵，于是多出一件全新的工作：**反向传播**。

| 模块 | 回答的问题 | 行数 |
|------|-----------|------|
| `errors.py` | 五族失败：形状 / 参数 / 数值 / **梯度**（多出的一族属于控制流） | 131 |
| `types.py` | 三个形状记录 + 三张口径表（七个阶段 / 五项梯度校验 / 四项性质） | 675 |
| `layers.py` | 前向七步、反向七步、逐元素 MSE（含监督行变体） | 482 |
| `verify.py` | 五项梯度校验、四条性质、与检索的类比 | 868 |
| `train.py` | induction 任务、批量读数、训练回路与冻结 | 839 |
| `__init__.py` | 包说明与 83 个导出名 | 249 |

## 2. 乘法口径只有一条

```text
权重 W 的形状是 (d_out, d_in)        与 nn.Linear 一致
投影 = x · Wᵀ                       x 的形状是 (n, d_in)

dW = gradᵀ · x
dx = grad · W
```

两条反向式子各有一条**用手算小矩阵核对**的测试
（``test_attention_layer.py`` 的 ``TestProject`` 与 ``TestBackwardHandComputed``）。

四个投影矩阵的形状契约（``AttentionParams.__post_init__`` 一次校验）：

```text
w_query  (d_k, d_in)      w_key  (d_k, d_in)      行数必须相同（打分是同一个空间里的点积）
w_value  (d_v, d_in)      w_output (d_out, d_v)   列数必须等于 d_v（W_o 消费 context）
四个矩阵的列数必须相同（d_in）——它们作用在同一个输入上
```

四个维度不合成一个 ``d``：``d_k`` 只影响打分尺度、``d_v`` 只影响混合结果宽度、
``d_out`` 只影响最终输出宽度。

## 3. 前向七步与反向七步

```text
① Q/K/V = x·Wᵀ                    ⑦ dW_o = dOutᵀ·context ；dContext = dOut·W_o
② raw = Q·Kᵀ                      ⑥ dWeights = dContext·Vᵀ ；dV = Weightsᵀ·dContext
③ scores = raw/√d_k               ⑤ dScores = softmax 反向（逐行）
④ 掩码：上三角标成"不允许"         ④ 掩码位置的梯度置 0
⑤ weights = 逐行 softmax          ③ dRaw = dScores/√d_k
⑥ context = weights·V             ② dQ = dRaw·K ；dK = dRawᵀ·Q
⑦ output = context·W_oᵀ           ① dW_q = dQᵀ·x 等三块 + dInputs = 三条链之和
```

### 3.1 softmax 的反向：一行公式

```text
dScores_i = p_i(g_i − Σ_j g_j p_j)
```

三条性质：括号里那一项是**整行的加权平均**；被掩码的位置权重为 0，
因此它们的梯度自动是 0；把这一步简写成 ``dScores = dWeights`` 不会报错，
只会让每一步的幅度不对。

### 3.2 三处"漏掉不会报错"的地方

```text
① softmax 的雅可比    dWeights → dScores 不是恒等映射
② 缩放系数            前向除了 √d_k，反向必须乘回 1/√d_k（漏掉差一个 √d_k 倍）
③ 掩码位置的梯度      前向掩码正确，但 dWeights 在被掩码位置一般不为 0——
                     不置 0 会让位置 i 通过一条"它看不到的路径"收到梯度。
                     这一条最隐蔽：**前向看起来完全正确**
```

三处都由 ``check_attention_gradients`` 的五项对照 + 四条性质分别守住。

## 4. 五条纪律

1. **损失取 MSE**：``∂MSE/∂y = 2(y−t)/N`` 一行可手算，
   因此"外部输入的梯度"是平凡的——任何梯度校验失败都只可能来自注意力那七步。
2. **掩码是显式的**：被掩码的位置权重**恰好 0.0**、不计入 softmax 分母
   （沿用 day073 的 ``masked_softmax_rows``）；``causal=True`` 与显式掩码不能同时给。
3. **全零输入被拒绝**：它会让 softmax 给出均匀分布，而"均匀分布"看起来像"还没学到"。
4. **不替调用方猜**：padding 要显式裁剪或用掩码处理，不要交给这一层猜。
5. **三个读数一起看**：损失（混得多准）、峰值权重（注意力多尖）、命中率（argmax 对不对）。

## 5. 五项梯度校验

```text
w_output   反向的第一站（只吃损失对输出的梯度）
w_value    把"输出该混哪些位置"翻译成"value 该怎么调"
w_query    梯度要穿过 softmax
w_key      与 query 共享同一份软打分的梯度
inputs     三条链之和 —— **唯一能发现"漏了一条链"的检查**
```

实测（单位化参数、3 行 3 维样本）：五项最大相对误差 **2.4e-11**，
容差 ``1e-6``（比分辨率约 1e-9 松三个数量级，而"真的写错了"通常差 1e-2 以上）。

### 5.1 一个真实踩过的坑：两侧必须算同一个损失

第一版的数值侧算**全行** MSE，解析侧算**监督行** MSE：
"最大绝对差 7.5e-3"（梯度范数只有 4.8e-2，即差 15%），而**两边都对**。
补上 ``supervised`` 参数之后差值降到 **2.1e-11**。

> 一个"两边都对却对不上"的对照，比"有一边错"的对照更难查——
> 它会让人先去怀疑推导。这就是为什么对照必须把"算的是哪个式子"写进参数名。

## 6. 四种性质

| 性质 | 在说什么 | 失败意味着 |
|------|----------|-----------|
| `row_stochastic` | 每一行权重之和为 1 | 后续加权求和变成缩放错误的组合 |
| `non_negative` | 每个权重 >= 0 | "加权平均"在某处变成减法 |
| `causal_no_leak` | 上三角**恰好 0.0** | 位置 i 看到了未来 |
| `permutation_equivariance` | 无掩码时输入换序、输出按同样方式换序 | 顺序信息被悄悄引入了 |

**最后一条在这一课有第二个答案**：因果掩码下偏差**显著不为 0**，
而这不是 bug——掩码把"谁在谁前面"写进了那张表。
它是 day078（位置编码）的伏笔：

```text
掩码只提供一种很粗的顺序（谁在谁前面）
要表达"位置 3 与位置 5 的距离"，仍然需要位置编码
```

## 7. 与检索的类比

同一次前向里，逐行比较"注意力在混谁"与"余弦检索会取谁"（**只在允许的位置上比**）：

| 读数 | 回答 |
|------|------|
| `peak_agreement` | 两边的 argmax 是不是同一个位置 |
| `overlap` / `overlap_ratio` | top-k 集合的重合 |
| `rank_correlation` | 整体排序的秩相关（Spearman，平均秩处理并列） |

为什么三个都要：top-k 是离散量（``k = 1`` 时只取值 0 或 1），
很多差别会被它吞掉；秩相关是连续量，回答"整体排序像不像"。

**秩相关的两个约定**：并列用**平均秩**（``(1,1,2)`` → ``(1.5,1.5,3)``，
与"字典序名次"给出的答案不同）；任一边方差为 0 时返回 0.0（"没有相关性可言"
与"相关性为 0"在数值上一样，但调用方需要一个数才能继续）。

## 8. induction 任务与三个读数

```text
序列      a b c a b            （3 个不同 token + 2 个重复）
监督位置  第 3、4 位
目标      这一位的输出等于**它自己那个 token 的 one-hot**
```

任务是**可逐位复核**的：一个"找同一个 token"的机制就能做对，
因此"训练有没有效果"不会与"任务的模糊性"混在一起。

实测（``Adam 0.05``、200 步、4 条样本）：

```text
四块全训     损失 0.1634 → 0.0000   命中率 38% → 100%   峰值 0.232 → 0.444
只训 q/k     损失 0.1634 → 0.1554   命中率 38% →  12%   峰值 0.232 → 0.587
```

**两个方向都反直觉**：

```text
只训 q/k      注意力确实变尖了，但损失几乎不动——"看对了地方"只是必要条件，
              还要 value/output 把看到的东西映射成目标
四块全训      损失到 0、命中率 100%，而峰值只有 0.44——
              模型在 value 路径上把不需要的分量抵消掉了
```

**结论：没有任何一个读数能单独说明"学到了什么"。**

## 9. 两种梯度源的对账

```text
SGD  0.05   6 步   损失轨迹最大差 5.09e-14
Adam 0.05   6 步   损失轨迹最大差 4.62e-11
Adam 0.05  30 步   损失轨迹最大差 7.05e-11
Adam 0.10  30 步   损失轨迹最大差 1.41e-10
```

差别来自**数值差分自身**（约 1e-11），而 Adam 的差别比 SGD 大 1000 倍——
它按 ``√v̂`` 归一化，会放大梯度分量的**相对**误差。两支都收敛，
但"验证梯度实现"应当用 SGD（步长与梯度成线性）。

## 10. 与既有模块的接缝

```text
上游   math_foundations（day073 的 masked_softmax_rows 与 softmax、
       day074 的 calculus.gradient / flatten_matrices / 四种优化器）
脚下   config **没有**新增配置项：容差、步长、初始化幅度、学习率曲线都是函数参数
下游   day076    多头：把 d_k 切成若干段（每段一个完整的注意力）
       day078    位置编码：解决"置换等变"那条性质指出的问题
       day079~80 Encoder 与从零实现：把这一层堆起来——而 grad_inputs 就是为
                 那一步准备的（没有它，层没法堆）
       day083    注意力可视化：把 weights 画成热力图
```

## 11. 一页速查

```text
我要……                          用……
构造参数（四个投影）              AttentionParams(w_query=..., ...)
看形状与参数量                    params.shape / params.parameter_count() / describe()
前向                              self_attention(params, inputs, causal=True)
前向的全部中间量                  forward.to_dict() / forward.weights / forward.row_entropies
反向                              attention_backward(forward, grad_output)
把参数交给优化器                  params.flatten() → day074 的 Optimizer
验证反向                          check_attention_gradients(params, inputs, target)
看四条性质                        check_properties(params, inputs, causal=True)
量"掩码破坏了多少置换等变"        permutation_gap(params, inputs, causal=True)
与检索对照                        compare_with_retrieval(forward, top_k=2)
造样本                            make_induction_task / make_induction_batch
训练一层                          train_attention(initial, tasks, optimizer=..., steps=...)
冻结部分参数                      trainable=("w_query", "w_key")
```

三条纪律，可以贴在任何"要训练一层新东西"的旁边：

```text
① 反向传播的每一处"简化"都不会报错——因此每一处都要有独立的检查
② 对照的两侧必须算同一个式子（把式子名写进参数名）
③ 损失降了不等于机制学到了：至少要有一个不看损失的读数
```
