# 多头注意力手册（day076 / M7-D2）

> 这一份是 day076 的**权威手册**：`multi_head` 包里的每一条判据、每一处拒绝、
> 每一个容差都在这里有对应的小节。教程（`../教程/教程.md`）讲"为什么"，
> 本文档讲"是什么、怎么用、出问题看哪里"。

---

## 1. 一页速查

```text
包          smart_research_agent.multi_head
上游        transformer_core（day075）+ math_foundations（day073/074）
下游        day078 位置编码 / day079~080 堆叠 / day083 可视化
新增依赖    无（纯 Python 算术，前向、反向、训练全部不依赖 numpy / PyTorch）
新增配置项  无（头数与容差都是"这一次调用或这一次对照的判据"，进函数参数）
```

一次多头前向的九步：

```text
project → split → score → scale → mask → softmax → mix → merge → output
```

三条与 day075 的差分：

| 维度 | day075（单头） | day076（多头） |
|------|----------------|----------------|
| 每行几个分布 | 1 个 | `heads` 个（各自和为 1） |
| 缩放系数 | `1/√d_k` | `1/√d_h`，`d_h = d_k / heads` |
| 参数与头的关系 | 四个矩阵属于这一层 | 四个矩阵被**所有头共用**（`W_o` 不参与分头） |
| 参数量 | `4·d²`（方阵时） | **完全相同**（heads 不新增参数） |

---

## 2. 形状契约

```python
MultiHeadShape(attention=AttentionShape(inputs, keys, values, outputs), heads)
```

| 派生量 | 公式 | 它决定什么 |
|--------|------|-----------|
| `head_dim` | `keys / heads` | 每一头打分用几维 → 缩放系数 `1/√head_dim` |
| `head_value` | `values / heads` | 每一头混合出几维 → context 分块宽度 |
| `scale` | `1/√head_dim` | 多头**真正**用的缩放 |
| `single_head_scale` | `1/√keys` | day075 用的那个（**只在 `heads=1` 时相等**） |
| `scale_ratio` | `scale / single_head_scale` | 派生量，恒等于 `√heads` |

三条拒绝：

```text
heads < 1                  → ParameterError
keys  % heads != 0         → PartitionError
values % heads != 0        → PartitionError（**不替你挑一个"最接近的合法头数"**）
```

> `AttentionParams`（day075）额外要求四个投影的**列数相同**，因此本包实际满足
> `d_in == d_v`；`head_value = d_v / heads`。

---

## 3. 九步前向（含形状）

```text
① project   (n, d_in) → Q/K/V 各 (n, d_k)/(n, d_k)/(n, d_v)   与 day075 同四个矩阵
② split     Q/K 按 **列** 切 heads 段 → heads × (n, d_h)；V 同理 → heads × (n, d_vh)
③ score     每头 raw_h = Q_h·K_hᵀ → (n, n)
④ scale     scores_h = raw_h / √d_h
⑤ mask      同一张掩码表发给每一头
⑥ softmax   每头按行归一化 → heads 份 (n, n)，**每头每一行**和为 1
⑦ mix       ctx_h = weights_h·V_h → (n, d_vh)
⑧ merge     heads 段按列拼成 (n, d_v)
⑨ output    y = merged·W_oᵀ → (n, d_out)
```

**一处最容易写错的方向**：权重矩阵按**行**切、激活按**列**切。

```text
W_q   形状 (d_k, d_in)      d_k 在它的行上   → split_rows
Q     形状 (n, d_k)         d_k 在它的列上   → split_columns
W_o   形状 (d_out, d_v)     d_v 在它的列上   → split_columns
```

切错不会报错，只会让每一头看到"别人和其他人的混合"——而每一头仍然是一张合法的
`(n, n)` 权重表。

---

## 4. 九步反向（含公式）

```text
⑨ dW_o = dOutᵀ·merged ；dMerged = dOut·W_o                    （W_o 不分头）
⑧ 按列把 dMerged 切回 heads 段（dCtx_h）
⑦ dWeights_h = dCtx_h·V_hᵀ ；dV_h = weights_hᵀ·dCtx_h
⑥ dScores_h ← softmax 反向（逐行）；被掩码位置再显式置 0
⑤④ dRaw_h = dScores_h · scale                                 （scale = 1/√d_h）
③ dQ_h = dRaw_h·K_h ；dK_h = dRaw_hᵀ·Q_h
② 按**行**拼回：dQ = ‖_h dQ_h、dK = ‖_h dK_h、dV = ‖_h dV_h
① dW_q = Σ_h dQ_hᵀ·x ；dW_k = Σ_h dK_hᵀ·x ；dW_v = Σ_h dV_hᵀ·x
  dx   = Σ_h (dQ_h·W_q^h + dK_h·W_k^h + dV_h·W_v^h)
```

两处**刻意的写法**：

```text
"按行拼回"而不是"求和"    每一行只由一头贡献 → 没有求和顺序的歧义 → 可以断言逐位相等
dx 是 heads × 3 条链之和  少一条不报错，只会让下层学得慢（day075 的同一个坑）
```

### 一条**近似**恒等式

```text
matmul(merged, W_oᵀ)  ==  Σ_h matmul(ctx_h, (W_o^h)ᵀ)
```

实数上严格相等；浮点上差 `1e-18` 量级（本样本 `6.94e-18`），因为两种写法的
**求和顺序不同**。判据取 `<= 1e-12`。

---

## 5. 五项梯度校验

```text
w_output   dW_o = dOutᵀ · merged_context       唯一**不被分头**的一块
w_value    dV = Σ_h …；dW_v = Σ_h dV_hᵀ · x    heads 条链在**行维**上相加
w_query    dScores_h ← 每头各自的 softmax 反向；dW_q = Σ_h dQ_hᵀ · x
w_key      dK = Σ_h dK_h；dW_k = Σ_h dK_hᵀ · x
inputs     dx = Σ_h (dQ_h·W_q^h + dK_h·W_k^h + dV_h·W_v^h)
```

| 参数 | 缺省 | 含义 |
|------|------|------|
| `tolerance` | `1e-6` | 与 day075 同值（运行时有一条断言把两者钉成相等） |
| `step` | `1e-6` | 中心差分的步长（day074 的 `calculus.gradient`） |
| `heads` | 1 | **必须与解析侧一致**（见下） |

**三条口径必须两边一致**，否则对照失败看起来像"推导错了"：

```text
损失口径   全行 MSE vs 监督行 MSE（day075 栽过：差 15% 而两边都对）
heads      数值侧忘了传 → 它算的是单头前向（差 2.9e-4）
掩码口径   一个因果、一个全开（差 2.8e-4）
```

后两条各自有一条测试，并把观测到的量级写在断言的注释里。

---

## 6. 六条性质

| 性质 | 判据 | 失败意味着 |
|------|------|-----------|
| `per_head_row_stochastic` | 每头每行和为 1（证据里带 **行数**：`heads × n`） | 后续加权求和变成缩放错误的组合 |
| `head_non_negative` | 每个权重 `>= 0` | 某一头的"加权平均"变成减法 |
| `causal_no_leak_per_head` | 每一头上三角**恰好 0.0** | 位置 i 看到了未来 |
| `merge_inverts_split` | `merge(split(M)) == M`（**恰好**，共 9 组矩阵） | 某一头看到的维度里混进了别的头 |
| `single_head_matches_classic` | `heads=1` 的输出/权重/context 与 day075 **逐位相同** | split/merge 的重构破坏了退化关系 |
| `head_order_is_bookkeeping` | 一致重排头块后输出不变（`<= 1e-12`） | 划分配置或"按列读"的逻辑错了 |

> `causal=False` 时第三条返回的是**"跳过"**而不是"通过"：
> "没开掩码"与"掩码正确"在报告里长得一样，必须分开。

---

## 7. 头间差异（`HeadDisagreement`）

```text
TV(p, q) = ½·Σ|p − q|        （对称、有界 [0,1]，不用 KL：KL 在 q=0 时无穷）

pairs              头对数量 = heads·(heads−1)/2
comparisons        pairs × rows（**只在允许的位置上比**）
mean/max/min TV    三档读数
peak_agreement     两头 argmax 相同的对数
degenerate         平均 TV <= 1e-9 且 pairs >= 1
```

两条必须记住的边界：

```text
heads=1        pairs=0、TV 恒为 0，而 degenerate=False（"没有第二个头"≠"多头退化"）
只在允许位置比  掩码把未来位置的权重写成 0（所有头都一样），不排除它们时 TV 被稀释
```

它**不回答**"各头有没有学到不同机制"——那要看梯度与训练曲线。

---

## 8. 梯度分块：W_o 不属于任何一头

```text
W_q / W_k / W_v   按行分头 → head_gradient_norms 把三块的行范数加起来
W_o               **不分头** → 它的梯度里无法区分"贡献来自哪一头"
```

因此 `head_gradient_norms` / `head_gradient_shares` 只统计前三块，
**一个"某头不学"的结论绝不能只凭 W_o 的梯度下**。

全零梯度返回全 0 占比（一个真实观测：学习率为 0 或已到极小），不抛异常。

---

## 9. 可达集合

```text
单头   可达集合 = conv{ contrib_j }                      contrib_j = W_o·V_j
多头   可达集合 = Σ_h conv{ contrib^h_j } = conv{ Σ_h contrib^h_{j_h} }
```

`conv(A) + conv(B) = conv{a + b}` 让"无限多点的和"变成"有限网格点的凸包"。

**前提**（不满足时本包**拒绝计算**，抛 `ParameterError`）：

```text
head_dim >= |A_i|     否则打分矩阵的秩不足以张出任意 logits，
                      而一个偏大的可达集合会把"到不了"说成"到得了"
d_out == 2            高维可达集合算得出，但"距离"需要一个凸包工具
```

手工见证（`designed_witness()`）：

```text
d_in = d_v = 4、d_k = 8、d_out = 2、heads = 2    → head_dim = 4（**恰好等于**可见位置数）
W_o = [[1,0,0,0],[0,0,0,1]]                     → 第 0 头沿 x 轴、第 1 头沿 y 轴贡献
head 0 的 pull = (1, 0, 0.5, 0.5)               head 1 的 pull = (0, 1, 0.5, 0.5)
```

| 量 | 值 |
|----|-----|
| 单头顶点 | `(1,0) (0,1) (0.5,0.5)`（第三点共线） |
| 单头凸包 | 线段 `(0,1)—(1,0)` |
| 多头凸包 | 正方形 `[0,1]²` |
| 目标 | `(1, 1)` |
| 单头距离 | `1/√2 = 0.7071067811865475`（**公式值**；算法值差最后一位：`…476`） |
| 多头距离 | `0.0` |
| 兑现权重 | `head 0 → (1,0,0,0)`、`head 1 → (0,1,0,0)`（TV = 1.0） |

**两个都对的数可以差最后一位**：距离是"点到线段的算法"，而手算值是一条公式。
把两者写成 `==` 会让某次无关的重排变成假失败。

---

## 10. 训练与同参数量对照

```text
四个读数（长度都等于 steps + 1，含第 0 步）
    损失 / 峰值权重 / 命中率 / **头间差异**
batch_* 的缺省 causal=True
```

| 接口 | 用途 |
|------|------|
| `train_multi_head(initial, tasks, heads=…, optimizer=…, steps=…)` | 训练一层多头注意力 |
| `compare_heads(initial, tasks, heads_list=…, optimizer_factory=…, steps=…)` | 同参数量对照表 |
| `batch_gradients(…, source="analytic" / "numeric")` | 两条梯度来源 |
| `analytic_objective(tasks, heads=…)` | 供 `calculus.gradient` 手工核对的目标函数 |

`optimizer_factory` 是**工厂**而不是实例：优化器带状态（动量/二阶动量），
复用会让第二次训练带着第一次的动量起步——表现为"某一行收敛得莫名其妙地快"。

实测（4 条样本、120 步、Adam 0.05、同一批初始参数、参数量 144）：

```text
heads | 参数量 | 初始损失 | 最终损失 |  降幅 | 命中率 |  峰值 | 头间差异
    1 |   144 | 0.163411 | 0.000005 | 100% |  100% | 0.4432 | 0.0000
    2 |   144 | 0.163424 | 0.000021 | 100% |  100% | 0.4075 | 0.2415
    3 |   144 | 0.163459 | 0.000010 | 100% |  100% | 0.3958 | 0.1663
```

读法：`heads=1` 的头间差异**恒为 0**，而它照样把损失压到 0；
`heads=2/3` 各头给出了不同分布。**这份表是一次观测，不是"heads 越多越好"的结论。**

---

## 11. 常见症状 → 根因

| 症状 | 根因 | 去哪儿看 |
|------|------|---------|
| 训练能降但比预期慢，且 `heads` 越大越慢 | 缩放用了 `1/√d_k` 而不是 `1/√d_h` | 第 2、3 节；`MultiHeadShape.scale` |
| 每一头看起来都"差不多"，报告里 `head_disagreement ≈ 0` | 多头退化（各头学到同一分布） | 第 7 节 |
| `w_query` 那一项梯度校验失败、其余四项都过 | softmax 反向只跑了第一头 | 第 5 节 |
| 五项都过，但堆叠时下层学得慢 | `dx` 少写了某几条链 | 第 4、5 节 |
| `heads=1` 的输出与 day075 对不上 | split/merge 破坏了退化关系 | 第 6 节 `single_head_matches_classic` |
| 可达集合的结论"太好" | `head_dim < |A_i|`（打分矩阵的秩不够） | 第 9 节 |
| 手算值与实测值差最后一位 | 两条不同的计算路径（公式 vs 算法） | 第 9 节 |
| `d_k % heads != 0` | 划分不成立（`PartitionError`） | 第 2 节 |
