# 数学地基手册（`math_foundations`，day073 / Math-D1）

> 本文是 `smart_research_agent/math_foundations/` 的权威说明。教程在
> [`../../教程/教程.md`](../../教程/教程.md)，演示脚本在
> [`../../scripts/math_foundations_demo.py`](../../scripts/math_foundations_demo.py)
> （输出见 `outputs/math_foundations_demo.txt`，164 行）。

## 1. 这一层为什么存在

从 day074 起，整条路线进入**底层**：微积分与优化 → Attention → Transformer →
从零实现 → 深度学习。要读懂那些公式，先把三样东西的**数学含义**钉死：

```text
向量与矩阵   点积、模长、余弦、矩阵乘法、投影        ← 注意力打分的全部零件
概率分布     熵、交叉熵、KL、期望、方差、条件概率、贝叶斯  ← softmax 输出的那"一个分布"
注意力       把上面两样合起来：softmax(QKᵀ/√d_k)·V
```

| 模块 | 回答的问题 |
|------|-----------|
| `errors.py` | 四族失败（形状 / 数值 / 参数 / 概率表），修复人各不相同 |
| `types.py` | 形状与**四张口径表**：算子（含公式）、注意力变体、对照项、分布判据 |
| `linalg.py` | 十个算子：dot / norm / normalize / cosine / matmul / transpose / projection / softmax / power_iteration / low_rank |
| `probability.py` | `Distribution` 与 `JointTable`：熵、交叉熵、KL、期望、方差、采样、条件、贝叶斯、互信息 |
| `attention.py` | 缩放点积注意力、因果掩码、多头、位置编码，以及一个把"为什么要除以 √d_k"**量出来**的实验 |
| `bridge.py` | 与项目既有实现的八项逐点对照 |

## 2. 四个算子，四种"零向量"

同一个零向量，四处的处理**各不相同**，而且都对：

```text
math_foundations.linalg.normalize          拒绝（零向量没有方向）
math_foundations.linalg.cosine             返回 0.0（"不像"是一个合法的答案）
vectorstore.metrics.cosine_similarity       返回 0.0（一条脏数据不该打断整批查询）
llm.embedding.l2_normalize                 原样返回（写入侧不能因为"没有方向"就抛异常）
```

危险的不是"不一样"，而是**"不一样而没人知道"**——因此 `bridge` 把这一条
作为**有意记录的约定差异**写进报告（状态 `rejected`，而不是"分歧"）。

## 3. 十个算子与它们的公式

| 算子 | 公式 | 一句话 |
|------|------|--------|
| `dot` | `a·b = Σ a_i b_i` | 方向上的重合程度（不含长度） |
| `norm` | `‖a‖ = √Σa_i²` | 长度 |
| `normalize` | `â = a/‖a‖` | 缩到单位长度（零向量无方向 → 拒绝） |
| `cosine` | `a·b/(‖a‖‖b‖)` | 落在 [-1, 1] |
| `matmul` | `(AB)_ij = Σ_k A_ik B_kj` | 两次线性组合的复合 |
| `transpose` | `(Aᵀ)_ij = A_ji` | 行列互换 |
| `projection` | `proj_v(u) = u·v̂` | "在这个方向上有多长" |
| `softmax` | `exp(z_i)/Σexp(z_j)` | 打分 → 分布 |
| `power_iteration` | `v ← Av/‖Av‖` | 主特征向量会自己浮现 |
| `low_rank` | `A_r = Σ_{i≤r} σ_i u_i v_iᵀ` | 用少数方向近似矩阵（LoRA 的原型） |

**求和一律用 `math.fsum`**（精确累加，误差与顺序无关），而不是内置 `sum`
（顺序累加，误差随维度线性累积）。这与"检索里两处余弦必须一致"是同一类要求。

## 4. softmax 的三种数值陷阱

```text
① 上溢        z = [1000, 1001] 直接 exp → inf/inf = nan
              解：先减最大值（不改变结果，只把 exp 的自变量压到 <= 0）
② 下溢        z = [0, -1000] 时 softmax 的第二项是 0.0；
              朴素写法 math.log(0.0) 在 Python 里**直接抛 ValueError**（不是返回 -inf）
              解：log_softmax 不经由 softmax，直接算 z − max − log Σexp(z − max)
③ sigmoid 溢出 x = -800 时 exp(800) = inf；解：按符号分支实现
```

三条的共同点：**在常见输入上完全正确**，只在极端输入上炸——因此在测试里
必须显式地喂极端输入（本课的 `LOGIT_SAMPLES` / `SCALAR_SAMPLES` 就是干这个的）。

## 5. 熵、交叉熵、KL（单位一律 nats）

```text
H(p) = −Σ p_i ln p_i        均匀分布最大（ln n），独热为 0
H(p, q) = −Σ p_i ln q_i     交叉熵 ≥ 熵，等号只在 p == q 时成立（Gibbs）
KL(p‖q) = Σ p_i ln(p_i/q_i) 非对称、≥ 0、且等于 交叉熵 − 熵
```

两条实现约定：

```text
p_i = 0 的项贡献 0（不是 0·(−inf) = nan）：一个 nan 会污染整张"平均熵"
q_i = 0 而 p_i > 0 时**报错**（数学上是 +inf）：
    让 inf 流下去会让下游均值变 inf、图表断线，而根因（那个给出 0 概率的实现）查不出来
```

单位为什么必须写下来：`sft.loss` 的交叉熵是 nats、`perplexity = exp(loss)` 也是 nats；
若这里改用 bits，同一个数字在两处报告里会差一个 `ln 2` 倍，而它们各自的说明都"看起来对"。

## 6. 联合分布与贝叶斯

```text
边缘     P(A) = Σ_b P(A, B=b)        把行加起来
条件     P(B|A=a) = P(A=a, B) / P(A=a)  行归一化
贝叶斯   P(A|B=b) = P(B=b|A)·P(A) / P(B=b)
                        似然        先验      证据
互信息   I(A;B) = H(A) + H(B) − H(A,B)    独立时为 0
```

三条实现纪律：

1. **先验缺省取表的边缘** —— 此时后验必然等于列条件分布。这不是巧合，
   而是一条可以断言的自洽性（两者不等说明似然或证据取错了）。
2. **证据为 0 时报错**，不给"后验全是 0"：一个概率为零的观测不会让任何假设变得可能。
3. **似然为 0 而先验不为 0 时报错**：严格按公式会得到"这个原因彻底不可能"，
   而真相是"这份数据不支持它"——应当改用平滑后的似然，而不是接受一个 0。

## 7. 采样与可复现的随机性

```python
sample_index(probabilities, u=0.79)   # 累积区间：p = [0.5, 0.3, 0.2] → 边界 0.5 | 0.8 | 1.0
uniforms(1000, seed=42)               # LCG，**不是密码学安全的**，但确定性
```

`u` 由调用方给（而不是内部调 `random`），因此"按这个分布采 1000 次"这条结论
可以被逐位复现——这与 day065 的"能重算的才叫版本号"是同一条纪律。

边界约定必须写死：`u` 恰好等于某个边界时取**后**一段（`p_i` 覆盖 `[c_{i-1}, c_i)`）。

## 8. 幂迭代与低秩近似

```text
v ← AᵀA v / ‖AᵀA v‖     反复做 → v 收敛到 AᵀA 的主特征向量
σ = ‖A v‖               （= √λ_max(AᵀA)）
u = A v / σ
```

**收敛判据用"两个单位向量是否平行"（|cos| → 1）**，而不是逐分量比大小：
主向量可能在两步之间整体翻符号，逐分量比较会把那次翻转当成"还没收敛"，
于是永远跑到迭代上限（而结果其实已经对了）。

**方向精度是二次的**：余弦差 `1e-12` 对应分量误差 `1e-6`（实测：`d=3` 的对角矩阵
跑 20 轮就到这个量级）。因此"σ 精确到多少"必须与"方向精确到多少"一起读。

低秩近似用"幂迭代 + 逐步正交化"（**教学实现，不是 LAPACK**）：

```text
① 对残差求主奇异三元组   ② 把新方向对已找到的方向做 Gram–Schmidt 正交
③ 从残差里减掉这一层     ④ r 层之后剩下的就是"被丢掉的能量"
```

正交化不能省：没有它时第二轮幂迭代会**再次收敛到同一个主方向**，
于是 A_r 变成 σ₁u₁v₁ᵀ 的重复叠加——误差也没大得离谱，只是"不太准"。

实测（对角矩阵 diag(2, 1, 0.5)）：

```text
秩 1 近似的相对误差  0.487950
秩 2 近似的相对误差  0.218218
秩 3 近似的相对误差  0.000000
```

## 9. 注意力：一行公式，四步熟悉的操作

```text
Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) · V
                     └── matmul ──┘ └ scaling ┘  └ softmax · matmul ┘
```

形状约定：`queries (n_q, d_k)`、`keys (n_k, d_k)`、`values (n_k, d_v)`、
`weights (n_q, n_k)`、`output (n_q, d_v)`。**K 与 V 的行数必须相同**（一一对应）。

### 9.1 因果掩码：用显式掩码，不用 `-inf`

生产实现常见写法是 `scores + (1-mask)·(-inf)`，然后一次 softmax
（`exp(-inf) = 0`，向量化友好）。本包**不这么做**：

```text
① 非有限数不该进入概率层   softmax 会拒绝 -inf（'非有限数一律拒绝'的纪律）
② -inf 会被后续运算静默破坏  -inf·0 = nan、归一化/裁剪/温度缩放都会出问题，
                          而错误出现在离掩码很远的地方
```

因此 `masked_softmax_rows` 只在"允许看"的位置上做 softmax，再把 `0.0` 插回去。
数学上两者完全等价（被屏蔽的位置权重就是 0，且不计入分母），
但显式掩码让"每一行仍然是一个合法的概率分布"**在这一层可校验**。

实测（3×3，因果）：

```text
第 0 行权重 (1.0, 0.0, 0.0)                    ← 只能看自己
第 1 行权重 (0.5, 0.5, 0.0)
第 2 行权重 (0.422319, 0.155362, 0.422319)     ← 能看全部（与不掩码时相同）
```

（不掩码时的平均熵 1.0514 nats / 集中度 4.3%；因果版 0.5702 / 48.1%——
"看不到未来"在数字上就是这个差。）

### 9.2 多头：每个头的 `d_k = d/heads`

```text
缩放系数是 1/√(d/heads)，**不是** 1/√d
头与头之间没有交互（交互发生在拼接之后的线性层）
每个头的分布不同 → "平均熵"与"各头集中度"才有信息量
```

## 10. 为什么要除以 √d_k

若 `q`、`k` 的每个分量独立同分布（均值 0、方差 `σ²`），则
`q·k = Σ q_i k_i` 的方差是 `d σ⁴`——**随维度线性增长**。

`sampled_dot_product_variance` 把这件事量出来（分量取 `[-1,1]` 均匀，`σ² = 1/3`）：

| d | 实测方差 | 理论 d/9 | 标准差 | 缩放后 |
|---|---------|---------|--------|--------|
| 4 | 0.4481 | 0.4444 | 0.6694 | 0.3347 |
| 16 | 1.7436 | 1.7778 | 1.3204 | 0.3301 |
| 64 | 7.5113 | 7.1111 | 2.7407 | 0.3426 |
| 256 | 28.5472 | 28.4444 | 5.3430 | 0.3339 |

**看最后两列**：未缩放的标准差随 `√d` 增长（0.67 → 5.34），
缩放后恒定在 `1/3 = σ²`。这就是 softmax 不会在长向量上饱和（梯度消失）的原因。

## 11. 与既有实现的八项对照

| 对照项 | 来源函数 | 结论 |
|--------|---------|------|
| `cosine` | `vectorstore.metrics.cosine_similarity` | 一致（含零向量约定） |
| `normalize` | `llm.embedding.l2_normalize` | **记录差异**（零向量两条路都对） |
| `softmax` | `sft.loss.softmax` + `llm.sampling.softmax_with_temperature` | 一致 |
| `log_softmax` | `sft.loss.log_softmax` | 一致 |
| `cross_entropy` | `sft.loss.cross_entropy` | 一致（最大误差 5.55e-17） |
| `perplexity` | `sft.loss.perplexity` | 一致 |
| `sigmoid` | `alignment.objectives.sigmoid` | 一致 |
| `log_sigmoid` | `alignment.objectives.log_sigmoid` | 一致 |

三种结论各有含义，**不能合并**：

```text
agrees     逐点在容差内一致
differs    超出容差 —— 这才是需要改代码的信号
rejected   两边的**拒绝行为**不同 —— 记录的约定差异，两条都对
```

对照工作里最容易做错的两件事，本课都踩到了：

1. **拿两串顺序不同的数去比**（softmax 那一项）：得到的"不一致"是假的。
2. **拿定义域不同的函数比数值**（交叉熵那一项）：`sft.loss.cross_entropy` 吃
   **打分（logits）**、真值是 **one-hot 下标**；本包的 `cross_entropy` 吃**概率**、
   真值是一个**分布**。把概率当 logits 传进去，形状对、数值合法、不报错——
   只是又做了一次 softmax：target 是 argmax 时 loss 变大、不是 argmax 时 loss 变小。
   **两个方向都会出现**，所以"loss 不对劲"不能只看方向，必须回到入参形态去查。

## 12. 复现实验

```bash
cd day073/源码/smart-research-agent

python scripts/math_foundations_demo.py                 # 十一节演示
python -m pytest tests/test_math_*.py -q --no-cov       # 219 个新用例
python -m pytest -q                                     # 全量回归 + 覆盖率闸门
```

全部离线：纯 Python 算术（不用 numpy）、零网络、零 API Key。
`math_foundations` 自己的覆盖率 **97.01%**（1004 条语句 / 300 个分支，缺 20 行）。
