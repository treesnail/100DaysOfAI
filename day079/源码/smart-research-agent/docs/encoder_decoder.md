# 编码器/解码器块手册（day079 / M7-D4）

> 这一份是 day079 的**权威手册**：`encoder_decoder` 包里的每一条判据、每一处拒绝、
> 每一个数字都在这里有对应的小节。教程（`../教程/教程.md`）讲"为什么"，
> 本文档讲"是什么、怎么用、出问题看哪里"。
>
> 除特别注明，本文里的每一个数字都来自 `python scripts/encoder_decoder_demo.py`
> 写下的 `outputs/encoder_decoder_demo.txt`（样本见 `tests/encoder_decoder_samples.py`）。

---

## 1. 一页速查

```text
包          smart_research_agent.encoder_decoder
上游        transformer_core（day075 的 self_attention / attention_backward / default_parameters）
            + math_foundations（day073 的 matmul/softmax/transpose 与 uniforms、day074 的 calculus.gradient）
下游        day080（把块组装成可堆叠的一层）/ day082（变体架构）/ day085（读 Hugging Face 的 LayerNorm）
新增依赖    无（纯 Python 算术，前向、反向、深度实验全程不依赖 numpy / PyTorch）
新增配置项  无（eps、d_ff 的倍数、摆放位置都是"这一次调用或这一次对照的判据"，进函数参数）
```

六个阶段一行式（**顺序就是数据流的顺序**，pre-LN）：
`norm1 → branch1 → add1 → norm2 → branch2 → add2`

与 day075 的差分：

| 维度 | day075（自注意力层） | day079（编码器/解码器块） |
|------|----------------------|---------------------------|
| 单位 | 一层注意力 | 一个**块**（三个子层拼起来，块是唯一被复制的单位） |
| 子层 | 一个（注意力） | 三个：注意力 + 前馈 + 两个残差，每个前面还各有 LN |
| 参数 | 四个投影 | 四个投影（day075，给定）**+ 八块**：两个 LN 的 γ/β + 前馈四块 |
| 梯度项 | 五项 | **九项**（八块新增 + 输入）／解码器另有**六项**交叉 + **两项**输入 |
| 摆放位置 | 无 | **pre / post**：只差 LN 的位置，形状完全一样而深堆叠差一个数量级 |
| 失败族 | 四族（形状/参数/数值/梯度） | **五族**（多出 `AssemblyError`：两个各自合法的部件拼不成一个整体） |
| 置换等变 | 无掩码时等变 | LN 与前馈仍然**逐行**（性质 4、5 用 `==` 断言） |

样本形状实测（第 1 节）：`n=4 d=6 d_ff=24（4.0×）| 本块新增参数 342 个`
（前馈 318 + 两个 LN 的 γ/β 各 2 组，注意力那一层的四块**不计**）。

---

## 2. 形状契约

```python
BlockShape(hidden, ffn, tokens)   # 一个块的三个维度
CrossShape(targets, sources, dimension)   # 交叉注意力的两路
```

```text
BlockShape   hidden = d（块内所有张量的宽度）  ffn = 4d      tokens = n（行数）
三条拒绝     任一维度 <= 0（或非整数）→ ParameterError（"- 必须 >= 1"）
派生量       ffn_ratio = ffn/hidden = 4.0   parameter_count = 4d + 2·d·d_ff + d_ff + d
             （= 4×6 + 2×6×24 + 24 + 6 = 342，可手算）

CrossShape   targets = n_tgt   sources = n_src   dimension = d_k
派生量       weights_shape = (n_tgt, n_src)——**长方形**（只要两路长度不同）
```

六类记录的入口校验（每一个 `__post_init__` 都在"拼装那一刻"拒绝坏输入）：

| 记录 | 它守什么 | 典型拒绝 |
|------|---------|---------|
| `NormCache` | 输入/标准化同形、均值方差逐行、γ 逐维 | `ShapeError`（形状不一致 / γ 与列数不符） |
| `FFNWeights` | `w_in (d_ff,d)`、`w_out (d,d_ff)`、两个偏置 | `ShapeError`（前馈是一个往返） |
| `BlockParameters` | 两个 LN 的 γ 长度相同、四块前馈借 `FFNWeights` 的校验 | `ShapeError`（γ 长度必须相同） |
| `BlockForward` | 六个阶段同形、**输入与输出同形** | `ShapeError`（**块不改变形状**，否则没法堆叠） |
| `BlockGradients` | 九块各自是向量/矩阵 | `ShapeError`（复用 day073 的 `validate_*`） |
| `CrossParameters` | `W_q/W_k` 行数相同、`W_v/W_o` 对接、**`W_k`/`W_v` 列数相同** | `ShapeError` / `AssemblyError`（K/V 必须都来自 source） |

---

## 3. 三个子层

```text
LayerNorm   y = γ ⊙ (x − μ)/√(σ² + eps) + β      μ、σ² **逐行**取（不跨行）
前馈         y = act(x·W_inᵀ + b_in)·W_outᵀ + b_out   **逐位置**（每一行各自过一个两层网络）
残差         y = x + F(x)                        **恒等映射加一个分支**
```

手算锚点（第 2 节，`x = (1, 2, 3, 4)`）：

```text
μ = 10/4 = 2.5        σ² = 5/4 = 1.25（**有偏**方差，与 PyTorch 一致）
x̂（eps → 0 的闭式）  = (-1.341641, -0.447214, 0.447214, 1.341641)
x̂（实测，eps=1e-5） = (-1.341635, -0.447212, 0.447212, 1.341635)   ← 只差第 6 位小数
"x̂ 的方差离 1 多远" = 8.00e-06      ← 它是 eps 的定义，**不是误差**
```

后一句话是第 7 节第 1 条性质的全部内容：`eps > 0` 时 `x̂` 的方差是 `σ²/(σ² + eps)`，
偏差 `≈ eps/σ²`。因此判据对每一行**分别**算期望值再比，
而不是拿"方差应当为 1"去比（后者会把数学当成 bug）。

前馈实测（第 4 节）：`(4, 6) → (4, 24) → (4, 24)（relu，激活后零点占比 38.5%）`——
零点占比是 ReLU 的一个便宜读数：关掉的神经元在反向里梯度**恰好是 0**。

两条"逐行"性质（第 3、4 节，判据是 `==`）：

```text
LN(πx)  ==  π·LN(x)       置换 (3, 2, 1, 0) 下逐位相等：True
FFN(πx) ==  π·FFN(x)      同上：True
```

它与 day078 的一句话接上：**打破置换等变性的只有位置编码与掩码**，LN 与前馈都不是。

---

## 4. 六个阶段与形状

| 阶段 | 形状 | 做什么 |
|------|------|--------|
| `norm1` | `(n, d) → (n, d)` | 第一个 LN（逐行标准化） |
| `branch1` | `(n, d) × 四个投影 → (n, d)` | 第一个分支：注意力（day075，**当给定函数**） |
| `add1` | `(n, d) + (n, d) → (n, d)` | 第一个残差（**+1 的路从这里开始**） |
| `norm2` | `(n, d) → (n, d)` | 第二个 LN |
| `branch2` | `(n, d) → (n, d_ff) → (n, d)` | 前馈（先扩张 4 倍再压回） |
| `add2` | `(n, d) + (n, d) → (n, d)` | 第二个残差：块的输出 |

pre 与 post **只差 LN 的位置**（`stage_order` 把它们写成两串阶段名）：

```text
pre-LN   norm1 → branch1 → add1 → norm2 → branch2 → add2      y = x + F(LN(x))
post-LN  branch1 → add1 → norm1 → branch2 → add2 → norm2      y = LN(x + F(x))
```

一处**容易被漏掉**的实现细节：pre 时注意力必须作用在 `LN(x)` 上，而"算好的注意力账"
不会随 `x` 变。因此本包用 `block_attention(params, x, attention_params, placement=…)`
**在同一个函数里**按摆放位置造出注意力——否则数值侧扰动 γ₁ 时那一份账不变，
而解析侧（用注意力自己的 `grad_inputs`）却算进了那条链，两边算的不是同一个函数
（症状见第 11 节的第一行）。

---

## 5. 九项梯度校验

```text
norm1_gamma    dGamma = Σ_rows dOut_norm ⊙ x̂（**按行相加**：γ 被所有行共用）
norm1_beta     dBeta  = Σ_rows dOut_norm（不经过 x̂，是纯求和）
ffn_w_in       dW_in  = (dPre ⊙ act')ᵀ · ln1_out；dPre 由 dOut 经 W_out 回传
ffn_b_in       dB_in  = Σ_rows (dPre ⊙ act')
ffn_w_out      dW_out = dOutᵀ · hidden
ffn_b_out      dB_out = Σ_rows dOut
norm2_gamma    同第一项，但 dOut_norm 来自**块输出**那一路
norm2_beta     同第二项
inputs         dx = dOut + dNorm1_inputs + dNorm2_inputs——**第一项就是残差那条 +1 的路**
```

`tolerance = 1e-6`（**与 day075 同值**，导入期把两者钉成相等）；
`step = 1e-6`（中心差分，day074 的 `calculus.gradient`）。四组对照实测（第 7 节）：

| 摆放 / 残差 | 通过 | 最大相对误差 | 逐项 |
|------|------|--------------|------|
| pre-LN 残差 开 | 9/9 | **8.28e-11** | 各项 3.6e-11 ~ 8.3e-11 |
| pre-LN 残差 关 | 9/9 | **5.09e-11** | 各项 2.6e-11 ~ 5.1e-11 |
| post-LN 残差 开 | 9/9 | **3.61e-10** | 各项 1.9e-10 ~ 3.6e-10 |
| post-LN 残差 关 | 9/9 | **3.92e-10** | 各项 1.1e-10 ~ 3.9e-10 |

容差 `1e-6`、实测 `~1e-10`——隔着**四个数量级**。四组能全部通过的结构性理由：
两侧都走**同一个** `block_attention` 与 `encoder_block`，
而 `placement / use_residual / activation / causal` 由同一组参数一路传给两侧
（"忘了传"在结构上不可能发生）。比较点数也是可手算的：两个偏置与两个 LN 的 γ/β
是**按行相加**的，因此它们的点数等于隐藏维（6）；`ffn_w_in` 是 `24×6 = 144`。

---

## 6. 六项交叉梯度校验（长方形权重）

```text
Q = target·W_qᵀ   K = source·W_kᵀ   V = source·W_vᵀ
scores  = Q·Kᵀ/√d      (n_tgt, n_src)   ← **长方形**
weights = softmax_rows(scores)
output  = (weights·V)·W_oᵀ

w_query        dW_q = dQᵀ · target（Q 只来自 target 那一侧）
w_key          dW_k = dKᵀ · source
w_value        dW_v = dVᵀ · source
w_output       dW_o = dOutᵀ · context
target_inputs  dTarget = dQ · W_q              （**只有一条链**）
source_inputs  dSource = dK·W_k + dV·W_v       （**两条链之和**——少一条不报错）
```

实测（第 8 节，target 4 行、source 6 行，权重 `4×6`，每行和最大偏离 `0.00e+00`）：

| 项 | 最大相对误差 | 逐点数 |
|----|--------------|--------|
| `w_query` | 3.07e-11 | 36 |
| `w_key` | 2.93e-11 | 36 |
| `w_value` | 3.79e-11 | 36 |
| `w_output` | 6.01e-12 | 36 |
| `target_inputs` | 3.70e-11 | 24 |
| `source_inputs` | **5.77e-11** | 36（比 target 多） |

最值钱的一行是 `source_inputs`：它是**两条链之和**，漏掉任何一条都不会报错、
只会让梯度偏小；而 `target_inputs` 只有一条链，因此它的比较点数比 source 少。
形状上的不对称（`dTarget` 与 target 同行数、`dSource` 与 source 同行数）是同一件事的另一种说法。

---

## 7. 八条性质

| 性质 | 判据 | 失败意味着 |
|------|------|-----------|
| `norm_rows_are_standardized` | 每一行均值 0、方差 `σ²/(σ²+eps)` | LN 的分母写错了 |
| `norm_is_invariant_to_a_constant_shift` | `LN(x+c·1) == LN(x)`（精确） | 均值没有减掉（平移泄漏进了输出） |
| `norm_is_equivariant_to_positive_scaling` | `LN(λx) == LN(x)`（`λ>0`，`O(eps/σ²)`） | 分母里的 σ 没有按同一倍数变大 |
| `norm_is_row_independent` | 换行序 → 输出**逐位**跟着换 | 用了 **BatchNorm** 式的跨样本统计量 |
| `feed_forward_is_position_wise` | 同上，**逐位** | 前馈里混进了跨行的求和 |
| `residual_is_the_identity_when_the_branch_vanishes` | 分支为 0 时 `y == x`（**逐位**） | 只零了一个分支（要两支都零） |
| `residual_keeps_a_unit_path_in_the_gradient` | 压小分支后 `dx` 仍有一个下界 | 反向里漏了残差那条 +1 的路 |
| `cross_attention_must_not_be_causal` | `causal=True` 被拒 +**量出加错的代价** | 交叉注意力被偷偷加了掩码 |

实测（第 9 节，饱和的交叉参数）：`性质检查 8 项：通过 8、失败 0`。

```text
[ok] norm_rows_are_standardized   4 行：均值最大偏离 3.15e-16；方差与 σ²/(σ²+eps) 最大偏离 4.44e-16
[ok] norm_is_invariant_to_a_constant_shift   加常数 3.5 前后最大偏差 5.77e-15
[ok] norm_is_equivariant_to_positive_scaling 乘正数 2.5 前后最大偏差 2.11e-11
[ok] norm_is_row_independent / feed_forward_is_position_wise   置换下**逐位**相等：True
[ok] residual_is_the_identity_when_the_branch_vanishes  output == inputs（True）；‖dx‖/‖dy‖ = 1.000000
[ok] residual_keeps_a_unit_path_in_the_gradient 压到 1/16：有残差 1.13e+00、无残差 6.19e-02——相差 18.2 倍
[ok] cross_attention_must_not_be_causal  被拒绝：True；n_tgt == n_src == 4 → **不报错**，权重被改掉 1.00e+00
```

第 8 条是整个包里最值钱的一条——它的失效方式**很坏**：

```text
自注意力（解码器）  Q/K/V 来自同一路 → 必须加因果掩码（不许看未来）
交叉注意力          Q 来自解码器、K/V 来自编码器 → **绝不能加因果掩码**
加错的代价          n_tgt != n_src 时形状对不上（会报错，算运气好）
                    n_tgt == n_src 时形状刚好合适 → **不报错**，只是把源序列后半段删掉
实测（饱和）        代价 = 1.00e+00（每一行的权重几乎被整行重写）
```

---

## 8. 残差：那条 +1 的路，与 pre / post

残差的两条判据（第 5、6 节）：

```text
① 两分支为 0 时 output == inputs（**逐位**）；反向 ‖dx‖/‖dy‖ = 1.000000
② 把前馈输出层整体乘 s（1 → 0.25 → 0.0625）：
      带残差 dx ≈ dy + O(s) → 比值 1.13e+00（几乎不动）
      不带残差 dx = O(s)     → 比值 6.19e-02（随分支一起被压小）
   ⇒ 两者相差 **18.2 倍**——残差是梯度的那条高速路
```

一条必须写下来的坑：**"分支为零"要两支都为零**。
只把前馈的输出层置零时 `residual2 = residual1 = inputs + 注意力输出 ≠ inputs`；
两个分支都置零才会得到 `output == inputs`（判据用 `==`，因为恒等映射没有求和顺序问题）。

pre 与 post 的形状完全一样、参数量完全一样，而第 10 节把差别量了出来——
这就是"一个只差子层顺序的改动"值得单独成章的理由。

---

## 9. 解码器块（九个阶段）

```text
① norm1 = LN(x)          ② branch1 = 因果自注意力(LN(x))  ③ residual1 = x + branch1
④ norm2 = LN(residual1)  ⑤ branch2 = 交叉注意力(·, encoder) ⑥ residual2 = residual1 + branch2
⑦ norm3 = LN(residual2)  ⑧ branch3 = 前馈                 ⑨ output   = residual2 + branch3
```

与编码器块的差别只有一处：**多一个子层，而那个子层的 K/V 来自另一路**。
实测（第 10 节，解码器 3 行、编码器输出 6 行）：

```text
前向   解码器输入 3×6、编码器输出 6×6 → 输出 3×6；交叉注意力权重 (3, 6)（无掩码）
反向   dDecoder 1.591693 | dEncoder 0.094432 —— **dEncoder ≠ 0**
        "解码器只读编码器的输出"这句话在反向里不成立
两路输入梯度校验  2 项：通过 2、失败 0 | 最大相对误差 2.82e-10
```

三处**组装层面**的拒绝都是 `AssemblyError`：

```text
自注意力不是因果的         → 解码器会在训练时直接看到要预测的下一个 token
编码器输出与解码器不同宽   → 交叉注意力要求 K/V 与 Q 落在同一个空间里（两个数各自都合法）
γ/β 不是三个               → pre-LN 的每个子层各有自己的 LN
```

一处契约（第 12 节会再说一次）：`decoder_block` 的自注意力**从外面传进来**，
因此它自己**不是**一个关于 `decoder_inputs` 的纯函数——检查里必须按 pre-LN 契约
把注意力**重建**在 `LN(decoder_inputs)` 上（只用到传入 forward 的 `params` 与 `causal`）。

---

## 10. 深度实验：三个变体的 ‖∂loss/∂x‖

```text
pre_residual    y = x + F(LN(x))    现代实现的主流
post_residual   y = LN(x + F(x))    原论文的写法
bare            y = F(LN(x))        把那一项 +x 直接关掉（同一条代码路径）
```

判据：同一个初始化、同一个目标、同一个深度序列 `(1, 2, 3, 4, 6, 8)`，
只换摆放位置与残差开关。实测（第 11 节）：

| 变体 | 1 层 | 2 层 | 3 层 | 4 层 | 6 层 | 8 层（相对第 1 层） | ‖dx‖ |
|------|------|------|------|------|------|---------------------|------|
| `pre_residual` | 1.00e+00 | 2.15e+00 | 3.43e+00 | 7.54e+00 | 6.70e+00 | **9.27e+00** | 1.353129 |
| `post_residual` | 1.00e+00 | 1.03e+00 | 8.94e-01 | 9.35e-01 | 9.87e-01 | **9.45e-01** | 0.354252 |
| `bare` | 1.00e+00 | 3.54e-01 | 8.98e-01 | 1.54e+00 | 3.31e-02 | **1.92e-03** | 0.000438 |

判决（`verdict_ok`）：最深一层上 `bare < pre / 10` → **True**，
而且 `bare / pre = 2.07e-04`——一个只差"子层顺序 + 一个 +x"的改动，
在 8 层之后让梯度差出**三个数量级**。**这一课的值钱结论就在这里。**

一条边界（必须写下来）：这一条读数回答"梯度能不能传到最底层"（一个**数值**问题），
它**不回答**"这个堆叠能不能训好"——梯度大也可能是爆炸而不是好事。

---

## 11. 常见症状 → 根因

| 症状 | 根因 | 去哪儿看 |
|------|------|---------|
| `norm1_gamma` 那一项梯度差 `1e-1 ~ 1e0`，其余项几乎都对 | 把**算好的**注意力账从外面传进来，而解析侧把 `LN1` 那条链也算进了 `dx`——两边算的不是同一个函数 | 第 4、9 节；`block_attention` / `check_decoder_input_gradients`（第 12 节） |
| 给交叉注意力加了因果掩码却**没有报错** | `n_tgt == n_src`——长方形退化成方阵，掩码"恰好"合适，悄悄删掉源序列后半段 | 第 7 节；`causal_mask_damage` / `check_cross_attention_is_not_causal` |
| `source_inputs` 那一项梯度偏小，其余五项都过 | `dSource` 只写了一条链（漏了 `dK·W_k` 或 `dV·W_v`） | 第 6 节；`cross_attention_backward` |
| `inputs` 那一项偏小、其余八项都过 | `dx` 漏掉残差那条 **+1** 的路（`dOut` 那一项） | 第 5、8 节；`encoder_block_backward` |
| 深堆叠后靠近输入的层"学不动"、曲线平 | 残差被关掉（`bare`）或 post-LN 堆太深——梯度过不了层 | 第 10 节；`depth_study` |
| LayerNorm 之后"方差不是 1"被当成 bug | 那是 `eps` 的定义：方差 = `σ²/(σ²+eps)`，偏差 `≈ eps/σ²` | 第 3、7 节；`check_rows_are_standardized` |
| `residual_is_the_identity...` 标红，但明明把分支置零了 | 只零了**一个**分支（要两支都零）；或用了 post-LN（最后一步是 LN） | 第 7、8 节；`check_residual_identity` |
| 解码器两路里 `encoder_outputs` 的梯度恒为 0 | `W_k/W_v` 的列数取了 target 的宽度（K/V 变成作用在两路上） | 第 2、6 节；`CrossParameters`（会先被 `AssemblyError` 拒绝） |
| 换行序改变了 LayerNorm 的每一行输出 | 用了 **BatchNorm** 式的跨样本统计量；LayerNorm 必须逐行 | 第 3 节；`check_norm_is_row_independent` |
| 给一整行加了常数 / 乘了正数，输出没变，以为是 bug | LN 的**平移不变**与**正尺度不变**——这两样不是信息 | 第 3 节；`check_shift_invariance` / `check_scale_equivariance` |
| 解码器块报 `ShapeError`，但两路各自都合法 | 编码器输出与解码器隐藏维不同宽——**拼起来**才不成立 | 第 9 节；`decoder_block` 的 `AssemblyError` |

---

## 12. 本日测试与一处必须写下来的修复

新增 6 份测试文件（`pytest --collect-only` 计得**共 237 个用例**，实测 `237 passed`，
五份梯度/性质/深度/解码器测试合计 18.31s，远低于 3 分钟）：

| 文件 | 用例数 | 守什么 |
|------|-------|--------|
| `tests/encoder_decoder_samples.py` | —（样本模块，`__all__` + `approx`/`approx_matrix`） | 写死的形状、输入与手算锚点 |
| `tests/test_encoder_decoder_layers.py` | 97 | 形状契约、六类记录、三个子层、块的组装与摆放位置 |
| `tests/test_encoder_decoder_gradients.py` | 46 | 九项/六项/两项梯度校验、报告与三张名单的闭合 |
| `tests/test_encoder_decoder_properties.py` | 31 | 八条性质、逐行判据、`==` 断言与"加错的代价" |
| `tests/test_encoder_decoder_depth.py` | 33 | 三个变体、确定性初始化、深度判决 |
| `tests/test_encoder_decoder_decoder.py` | 30 | 交叉注意力、解码器九阶段与三处组装拒绝 |

实测覆盖率（`--cov=smart_research_agent.encoder_decoder --cov-branch`）：
**语句 99.77%、分支 98.99%、合计 99.63%**（`depth.py` / `verify.py` / `types.py` / `__init__.py` 均 100%）。
三处未覆盖的都是**防御性死分支**，且都有明确理由：

```text
errors.py:146   模块级"两张表不一致就报错"的那句 raise——两张表是同一份代码写死的，永远一致
layers.py:235   layer_norm 里"方差非有限就抛 NumericError"——validate_matrix 已经先拒掉了非有限输入
layers.py:629   块末"块改变了形状"的守卫——块按构造保形，这一条永远不会触发
```

### 一处必须写下来的修复（`check_decoder_input_gradients`）

交付过程中发现新包的一处 bug 并**在报告后做了最小修复**（改动只在 `verify.py` 的一个函数内）：

```text
症状    check_decoder_input_gradients 的 decoder_inputs 一项差 9.93e-01 ~ 1.05e+00
        （正是"最像 bug"的量级），而 encoder_outputs 一项 1.56e-10 正常
根因    decoder_block 的自注意力是**从外面传进来的一份算好的账**，
        因此它自己不是关于 decoder_inputs 的纯函数；而 decoder_block_backward
        按 pre-LN 契约把 LN1 那条链也算进了 grad_decoder_inputs。
        数值侧扰动的是一份**冻结**的注意力 → 两边算的不是同一个函数
修法    与 block_attention 同一条纪律：在检查里按 pre-LN 契约把注意力**重建**在
        LN(decoder_inputs) 上（只用到传入 forward 的 params 与 causal），
        解析侧与数值侧都用重建后的注意力
结果    2/2 通过，最大相对误差 2.82e-10（decoder_inputs 1.64e-10、encoder_outputs 2.82e-10）
```

这条修复与 `verify.py` 模块顶部早就写下的一句话同源：
**"把一份算好的注意力账从外面传进来是不行的"**——编码器块用 `block_attention` 解决了它，
解码器块这一处当时漏了。修复后的契约也写进了第 9 节与函数的 docstring/notes。
