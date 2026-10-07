# 训练技巧与正则化：BatchNorm / Dropout / 调度 / 早停（day095 / M8-D6）

> 一本可随手翻的手册，与 `smart_research_agent/regularization/` 一一对应。
> 每一节的每个结论都能在包内找到对应函数；每个读数都能现场重跑。

## 1. 这一课要回答什么

day094 交出一条能跑的序列链，并留下两个旋钮（学习率、展开长度）。今天把它变成**一套**旋钮，
并补上唯一一件新的数学：

```text
归一化      BatchNorm：沿**批**取平均（day079 的 LayerNorm 沿**特征**）——本包新建
随机丢      Dropout：训练置零 + 放大 1/(1−p)，推理恒等               ——转发 day081
学习率衰减  常数 / 阶梯 / 余弦 / 热身+余弦                          ——转发 day074
早停        连续 patience 步没有明显变好就建议停下                   ——转发 day081
可视化      sparkline / ASCII 折线 / 条形图 / 汇总裁剪              ——本包新建
```

一句话本质：**BatchNorm 的统计量来自一批样本——因此"这条链必须分批"、
"训练相与推理相必须配对"，两件事都不是风格问题，而是定义问题。**

## 2. 两条归一化轴

| 轴 | 公式 | 签名 |
|----|------|------|
| `batch`（BatchNorm） | `μ_j = (1/N)Σ_n x_nj；σ²_j = (1/N)Σ_n (x_nj−μ_j)²` | 批大小变成 1 时方差恒为 0 |
| `feature`（LayerNorm，day079） | `μ_n = (1/F)Σ_j x_nj；σ²_n = (1/F)Σ_j (x_nj−μ_n)²` | 与批里有多少条样本完全无关 |

两者是同一件事在两条轴上的做法，因此第 ② 条性质把批**转置**之后对账：

```text
batch_norm_forward(X)  ==  transpose(layer_norm(transpose(X)))      实测 0.000e+00
```

（这只在"没有仿射"时成立：**LayerNorm 的 γ 是逐列的、BatchNorm 的 γ 是逐特征的**，
两套仿射参数不在同一条轴上。）

## 3. 两个相

| 相 | BatchNorm 用哪一套统计量 | Dropout |
|----|------------------------|---------|
| `train` | **这一批自己的** μ/σ（同时把 running 往这批靠一步） | 置零 + 放大 |
| `eval` | **running** 统计量（必须由调用方给出） | 恒等 |

推理相缺少 running 时本包抛 `PhaseError`——因为"拿本批统计量去推理"正是要拦下的那件事：
单条样本时方差为 0，整层输出被压成常数 β。

## 4. BatchNorm 前向

```text
μ_j  = (1/N)Σ_n x_nj；σ²_j = (1/N)Σ_n (x_nj − μ_j)²      ← **有偏**方差（分母是 N）
x̂_nj = (x_nj − μ_j)/√(σ²_j + ε)
y_nj = γ_j·x̂_nj + β_j                                     ← γ/β 是**逐特征**的
running ← (1−m)·running + m·这批                            ← momentum 是"每个批占多少"
```

实现见 `normalization.batch_statistics` / `running_update` / `batch_norm_forward`。
默认 `ε = 1e-5`（**与 day079 同值**，这是第 ② 条能逐位对上的前提）、
`m = 0.1`。

## 5. 批大小为 1 的塌缩（本课真的撞到的第一堵墙）

```text
批大小 1 ⇒ 每个特征的批内方差恒为 0
        ⇒ x̂ = (x − x)/√(0 + ε) = 0
        ⇒ y = γ⊙0 + β = β
```

第一版实现把 BN 插在"单条序列的 h_T"上，于是三条**不同**的配置
（rnn / lstm / 带不带 dropout）给出**同一个 loss**（`0.542374`）。
这不是 bug，而是"BatchNorm 需要批"最直接的证据——
本课的网络因此一次前向处理**一批样本**，而 `normalization.batch_size_one_report`
把这件事变成一个可读的读数（`max_variance = 0.0`、`max_abs_output = 0.0`）。

## 6. BatchNorm 反向：三项 vs 一项

```text
训练相   dβ_j = Σ_n dy_nj；dγ_j = Σ_n dy_nj·x̂_nj；dŷ = γ⊙dy
         dx_nj = (1/σ_j)·( dŷ_nj − mean_n(dŷ_j) − x̂_nj·mean_n(dŷ_j ⊙ x̂_j) )
推理相   dx_nj = dŷ_nj / σ_j
```

训练相那三项存在，是因为 `μ` 与 `σ` **也是 x 的函数**；
推理相的 `μ/σ` 来自 running（常数），那两个减项就不该出现。
把两者互换**形状全对**，只有数值差分能发现——因此第 ④ / ⑤ 条性质各钉一条路。
一个可手算的读数：`dy` 全 1、`γ` 全 2 的 2×2 批上，三项**恰好抵消** ⇒ `dx = 0`。

## 7. 四个旋钮的转发关系

| 技巧 | 由谁实现 |
|------|---------|
| `batchnorm` | 本包新建（`normalization.batch_norm_forward / batch_norm_backward`） |
| `dropout` | day081 `training_optim.dropout`（`dropout_forward / dropout_backward`） |
| `lr_decay` | day074 `math_foundations.optim.make_schedule`（经 day081 转发） |
| `early_stop` | day081 `training_optim.controls.EarlyStopping` |

本课只在 `normalization.py` 里写了一件新的数学，其余全是转发——
`:data:`types.TECHNIQUE_SOURCES`` 把"谁实现了它"写成一张可核对的表。

## 8. 装上旋钮之后的链

```text
一批样本 → day094 的循环单元（**一行不改**） → 每条的 h_T
                                              ↓
                                    BatchNorm(γ, β)      ← 本包新写
                                              ↓
                                    Dropout（训练相）     ← 转发 day081
                                              ↓
                                    dense(H→C)           ← day094 的分类头
                                              ↓
                                            logits
```

反向的路线是 `dense → dropout → BN → 逐条 dh_T → day094 的 BPTT`——
**本课没有重写 BPTT**：它把每条样本的 `dh_T` 放进一条"只在最后一步有梯度"的向量里，
交给 day094，再把 N 条样本的权重梯度加起来（因为损失是批均值）。

参数压平沿用 day090 的契约：`day094 的 flatten_params` + γ + β，
还原时先切出末尾的 2H 个数，其余原样交回 day094。

## 9. 训练回路：四个动作的顺序是写死的

```text
取 lr（调度在优化器内部，day092 的 set_schedule）
  → 算这一步的批均值梯度（day094 的 BPTT）
  → 过 day094 的梯度范数守卫（check_gradient_norm；本课只是调用它）
  → optimizer.step → unflatten
```

每个 epoch 结束用**推理相**评一次（BN 用 running、dropout 恒等）：
训练损失与推理损失是两把尺子，只报其中一把会把 dropout 的代价看反。

## 10. 消融表的读数（本课真的撞到的第二堵墙）

```text
rnn   | 全开     | 训练 0.666587 | 推理 2.240945 | 间隔 +1.574358 | 准确率 68.8% | 早停@2
rnn   | 无归一化   | 训练 0.005928 | 推理 0.003449 | 间隔 -0.002479 | 准确率 100.0% | —
rnn   | 无丢弃    | 训练 0.262312 | 推理 3.892314 | 间隔 +3.630002 | 准确率 50.0% | 早停@1
rnn   | 无调度    | 训练 0.612144 | 推理 1.344223 | 间隔 +0.732078 | 准确率 59.4% | 早停@2
lstm  | 全开     | 训练 0.018691 | 推理 0.000004 | 间隔 -0.018687 | 准确率 100.0% | —
lstm  | 无归一化   | 训练 0.017225 | 推理 0.000201 | 间隔 -0.017024 | 准确率 100.0% | 早停@38
lstm  | 无丢弃    | 训练 0.003022 | 推理 0.000001 | 间隔 -0.003022 | 准确率 100.0% | —
lstm  | 无调度    | 训练 0.024393 | 推理 0.000000 | 间隔 -0.024393 | 准确率 100.0% | —
```

**这张表不证明"正则化有用"。** 它证明的是"这四个旋钮每一个都被真的装上了、
而且每一个的代价都出现在一列可读的数字上"：

```text
lstm 上四个变体都到 100% ⇒ 旋钮没有把它们弄坏
rnn  上带 BN 的三个变体都被**拉开**（间隔 +0.73 ~ +3.63）⇒ 归一化不是免费的
"无丢弃"那一行的间隔最大（+3.630002）——它把"只看训练损失会把代价看反"这句话量了出来
```

## 11. 七条性质与失败族

| # | 性质 | 判据 |
|---|------|------|
| ① | BN 前向 = 手写逐特征标准化 | 逐位相等 |
| ② | BN(训练相) = transpose(LayerNorm(批ᵀ)) | ≤ 1e-12 |
| ③ | 训练相与推理相**必须不同** | **下界** ≥ 0.5 |
| ④ | 训练相反向（三项）= 数值差分 | ≤ 1e-7 |
| ⑤ | 推理相反向（一项）= 数值差分 | ≤ 1e-7 |
| ⑥ | dropout 推理相 = rate=0 训练相 | 逐位相等 |
| ⑦ | sparkline 长度与极值位置 | 整数相等 |

判据三类：**相等**、**不超过上界**、**不超过下界**。见 `regularization/verify.py`。

失败族四个（按"该谁去修"分）：`ShapeError` / `ParameterError` / `NumericError` /
**`PhaseError`**（两相被配错了对）。`GradientError` 今天**再次缺席**——
梯度爆炸的守卫由 day094 装好，本课只是调用它。

## 12. 五条边界

- 只做 BatchNorm 与"转发 day081 的三个旋钮"：不做 GroupNorm / InstanceNorm / 权重标准化。
- 不做分布式训练：running 统计量不跨卡同步。
- 不做 GPU / 向量化：纯 Python 的逐元素实现，为的是可读与可对账，不是速度。
- 只在 day094 的序列分类头上做实验：**不改动 `sequence_models` 的任何一行**。
- 不新增第三方依赖：不 import torch / numpy；PyTorch 只写在对照表里，不安装也不调用。
