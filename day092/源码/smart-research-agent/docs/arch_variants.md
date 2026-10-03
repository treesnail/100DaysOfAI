# `arch_variants` 手册（day082 / M7-D7）

> 三个变体，只差三处：**自注意力的掩码**、**有没有第二路**、**训练目标**。
>
> 本手册是 day082 的权威参考。所有数字都可以用
> `python scripts/arch_variants_demo.py` 复现（十节、93 行输出）。

---

## 1. 三个变体

| 变体 | 例子 | 自注意力掩码 | 第二路 | 训练目标 |
|------|------|--------------|--------|----------|
| `encoder_only` | BERT | 全开（双向） | 无 | 掩码语言建模（MLM） |
| `decoder_only` | GPT | 因果（只看过去） | 无 | 自回归语言建模（CLM） |
| `encoder_decoder` | T5 | 编码器全开 + 解码器因果 | 有（Q 来自解码器、K/V 来自编码器） | 去噪（文本到文本） |

```python
from smart_research_agent.arch_variants import make_variant_shape, make_variant_parameters, variant_forward

shape = make_variant_shape(tokens=4, hidden=6, layers=3, sources=5)
params = make_variant_parameters(shape, "decoder_only")
forward = variant_forward(params, inputs)          # inputs: (4, 6)
```

**三个变体的每一层都是 day079/080 的同一个块**：`decoder_only` 用的就是
`encoder_block`，只把掩码换成因果的那一张。

---

## 2. 三张掩码

```text
mask_of("full", n)        全开：每一行允许看所有位置        n² 个位置对
mask_of("causal", n)      因果：第 i 行允许 j <= i          n(n+1)/2 个位置对
padding_mask(pads)        填充：(not pads[j]) or (i == j)   对角线永远允许
combine_masks(a, b, ...)  交集（逐位与）
```

```text
n = 4：full 允许 16/16（熵天花板 ln 4 = 1.386294）
      causal 允许 10/16（熵天花板 (1/4)Σln(i+1) = 0.794513）
      因果/全开 的位置对比例 = 0.625（n = 64 时是 0.5078，趋向 1/2）
```

三条边界：

```text
① 填充行的对角线被保留 ⇒ 每一行都有定义（day075 拒绝全 False 行）
② 填充行仍然看得到非填充位置 ⇒ 它的输出没有意义，必须在池化/损失里丢掉
③ 交集可能把某一行清空 ⇒ combine_masks 当场拒绝（而不是让 softmax 在更远处炸）
```

---

## 3. 实测依赖表（本课的核心判据）

```text
扰动第 j 个 token（乘 1.5）→ 重新前向 → 看输出第 i 行变了多少 ⇒ D[i][j]
```

| 变体 | 可达位置（可达 = 读数 > 0） | 挡掉的格子 | 与掩码一致 |
|------|------------------------------|------------|------------|
| `encoder_only`（全开） | `(4, 4, 4, 4)` | 0 个 | 是 |
| `decoder_only`（因果） | `(1, 2, 3, 4)` | 最大读数 `0.000000e+00` | 是 |
| `encoder_decoder`（主流） | `(1, 2, 3, 4)` | 最大读数 `0.000000e+00` | 是 |
| `encoder_decoder`（编码器流） | `(5, 5, 5, 5, 5)` | 0 个 | 是 |
| `encoder_decoder`（交叉那一路） | `(5, 5, 5, 5)`，表是 4×5 长方形 | 0 个 | 是 |

**为什么“恰好 0.0”**：`masked_softmax_rows` 只在允许的位置上做 softmax，
被屏蔽的那些打分**从来没有被读过** ⇒ 改它不会改变任何一个中间量。

反证（判据有分辨力）：

```text
decoder_only 换成全开掩码   越界 6 格（未来被看到了）
encoder_only 换成因果掩码   缺失 6 格（双向能力被砍掉）
```

---

## 4. 家底（参数量）

样本：`tokens=4 sources=5 hidden=6 ffn=24 layers=3`

| 变体 | 块 | 解码器块 | 自注意力层 | 交叉层 | 子层 | 参数量 |
|------|----|----------|------------|--------|------|--------|
| `encoder_only` | 3 | 0 | 3 | 0 | 6 | 1458 |
| `decoder_only` | 3 | 0 | 3 | 0 | 6 | 1458 |
| `encoder_decoder` | 3 | 3 | 6 | 3 | 15 | 3384 |

解析式（与逐个数出来的结果**必须相等**）：

```text
块           2·d·f + f + d + 4·d       = 2·6·24 + 24 + 6 + 24 = 342
自注意力      4·d²                      = 144
交叉注意力    4·d²                      = 144
解码器前馈    2·d·f + f + d             = 318
解码器 LN     6·d                       = 36
```

**因果与否不改变参数量**——它只改变“能看到什么”。

---

## 5. 前缀稳定性

扰动**最后一个** token（位置 3）：

| 变体 | 变了的行 | 第 0 行的变化 |
|------|----------|----------------|
| `encoder_only` | `(0, 1, 2, 3)`（4 行） | `2.869378e-02` |
| `decoder_only` | `(3,)`（1 行） | `0.000000e+00` |
| `encoder_decoder`（解码器流） | `(3,)`（1 行） | `0.000000e+00` |

---

## 6. 六条性质

| 性质 | 判据 | 不适用 |
|------|------|--------|
| `mask_zeroes_are_exact` | 权重表里掩码外的格子 `== 0.0` | 从不 |
| `decoder_reads_only_past` | 实测表 == 因果掩码（逐格） | `encoder_only` |
| `encoder_reads_every_position` | 实测表 == 全开掩码（逐格） | `decoder_only` |
| `cross_spans_all_sources` | 交叉那一路一格 0 都没有 | 没有交叉的变体 |
| `stack_preserves_shape` | `(n, d) → (n, d)` | 从不 |
| `deterministic` | 两次前向逐位相同 | 从不 |

```text
不适用 ≠ 通过：applicable=False 时 passed 也是 False，
报告只对适用的那些下结论（否则“删掉这条检查”与“它通过了”长得一样）
```

---

## 7. 梯度校验

解析（`variant_backward`，重放前向再倒着走）vs 中心差分（`numerical_stream_gradient`）：

| 变体 | 流 | 解析范数 | 数值范数 | 最大差 | 相对差 |
|------|----|----------|----------|--------|--------|
| `encoder_only` | main | 1.355482 | 1.355482 | 1.832e-10 | 1.352e-10 |
| `decoder_only` | main | 1.242010 | 1.242010 | 1.705e-10 | 1.373e-10 |
| `encoder_decoder` | main | 1.418204 | 1.418204 | 2.018e-10 | 1.423e-10 |
| `encoder_decoder` | source | 0.298308 | 0.298308 | 2.562e-10 | 2.562e-10 |

```text
容差 1e-6（与 day075/079 同值），步长 1e-6，激活 gelu（光滑）
把步长放大到 1.0 ⇒ 报告立刻亮红（这条测试证明判据有分辨力）
```

---

## 8. 生成的 PyTorch 脚本

```text
encoder_only      nn.Embedding + nn.TransformerEncoderLayer + nn.TransformerEncoder
decoder_only      nn.Embedding + nn.ModuleList（每层一张上三角掩码 src_mask）
encoder_decoder   nn.Embedding ×2 + TransformerEncoder + TransformerDecoder（tgt_mask）
```

```text
CONFIG 的键（生成侧与解析侧共用）：tokens / sources / hidden / ffn / layers /
                                    heads / vocab / norm_first
norm_first=True ↔ 本包的 placement="pre"（同一条口径的两个名字）
```

本机实测（`torch 2.14.0+cpu`，`heads=1`）：

| 变体 | PyTorch 参数量 | 本包解析式 | 差值 |
|------|----------------|------------|------|
| `encoder_only` | 1722 | 1458 | 264 |
| `decoder_only` | 1722 | 1458 | 264 |
| `encoder_decoder` | 3984 | 3384 | 600 |

差值的来源（**不凑平**）：

```text
词嵌入            32×6 = 192                              encoder_only 那一侧一份
注意力偏置        每层 in_proj_bias(d) + out_proj.bias(d) = 12，×3 层 = 36
⇒ 192 + 36 = 228 ≠ 264
```

更细地拆（`d=6`、`d_ff=24`、`heads=1`）：

```text
每层 = MultiheadAttention(108 + 18 + 36 + 6 = 168) + linear1(24×6+24 = 168)
       + linear2(6×24+6 = 150) + 2×LayerNorm(6+6 = 24)  = 510
三层 = 1530 ，加词嵌入 192 ⇒ 1722 ✓
```

---

## 9. 与既有包的接缝

```text
上游    encoder_decoder（day079 的 encoder_block / decoder_block 与两种反向）
        transformer_core（day075 的 self_attention 与 full_mask）
        math_foundations（day073 的 causal_mask 与 LCG 随机数）
        transformer_stack（day080 的逐层读数口径）
脚下    config 没有新增配置项：变体名、掩码名、层数、倍数都是函数参数
下游    day083（可解释性）要的就是今天留下的三张权重表
        day085（源码精读）会看到 BertEncoder / GPT2Model 与今天的三个变体逐层对应
```

---

## 10. 两个真实的坑

```text
① pre-LN 下注意力作用在 LN(x) 上
   把算好的注意力账从外面传进来时用了原始 x ⇒ 前向形状全对、读数全对，
   而 encoder_block_backward 按“注意力在 LN(x) 上”那一条链分梯度
   ⇒ 两边算的不是同一个函数（误差 1e-1 量级，只有梯度校验能抓到）
   修法：_first_sublayer_input（pre 先过一次 LN）

② decoder_block 要求因果自注意力，而 resolve_mask 拒绝“因果 + 显式掩码”
   ⇒ encoder_decoder + pads 必须**显式拒绝**，而不是静默忽略
   （静默忽略的后果：前向照跑、形状一样、读数只是略有不同）
```
