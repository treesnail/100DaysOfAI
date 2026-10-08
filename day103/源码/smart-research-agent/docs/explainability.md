# `explainability` 手册（day083 / M7-D8）

> 把注意力读出来、画出来、并给出可被断言的读数。
>
> 本手册是 day083 的权威参考。所有数字都可以用
> `python scripts/explainability_demo.py` 复现（十节、90 行输出）。

---

## 1. 四件东西

| 名字 | 是什么 | 从哪来 |
|------|--------|--------|
| `AttentionRecord` | 一层一头的权重 + 它用的掩码 + 标签 | `extract.self_records` / `extract.head_records` |
| `HeadProfile` | 一张表的读数：熵 / 天花板 / 归一化熵 / 峰值 / 支撑 / Frobenius / 对角质量 | `analyze.profile_of` |
| `Attribution` | **谁在被看**（列和，归一化） | `analyze.attribution_of` |
| 文本热力图 | 10 级字符 + 图例，**可解析** | `render.heatmap_block` |

---

## 2. 两条读法（**不许混**）

```text
self_records    模型**真实**看到的权重（头数 1）
                直接取自 day082 的前向记录（block_weights / decoder_self_weights / cross_weights）
head_records    **同一组投影**在多头划分下的读法（用 day076 的 multi_head_attention 重算）
```

两条读法的差别有四条硬事实：

```text
① 缩放不同      单头 1/√d_k；多头 1/√(d_k/heads)  ⇒ 打分尺度差 √heads 倍（day076 的 scale_ratio）
② 作用位置不同   head_records 作用在**某一层的输入**上（单头路径重放到那一层）
③ 块类型不同     day079 的 encoder_block / decoder_block 要求 day075 的 AttentionForward，
                 而多头那份是 MultiHeadForward ⇒ 两处**不能直接拼**（day085 会看到 HF 把分头放进投影内部）
④ 唯一的接缝     head_records(heads=1) 必须**逐位**等于 self_records
```

第 ④ 条是一条可以被抓到的失败：第一次运行时它就抓到了"读错了层"这个 bug
（`heads=1` 在第 0 层相等、在第 2 层不相等）。

```python
from smart_research_agent.explainability import extract, study

shape = study.default_shape()                      # tokens=4 sources=5 hidden=6 layers=3
params = study.study_params("decoder_only", shape)
inputs = study.study_inputs(shape)

records = extract.self_records(params, inputs)     # 3 条（每层一条，头数 1）
heads = extract.head_records(params, inputs, layer=1, heads=2)   # 2 条
assert extract.single_head_matches(params, inputs)  # True：跨口径的接缝
```

---

## 3. 熵要跟天花板比（本课的第一条纪律）

```text
一行的熵 ≤ ln(这一行能看到的位置数)        ← Jensen 不等式
```

| 掩码 | 天花板（n = 4） |
|------|------------------|
| 全开（BERT） | `ln 4 = 1.386294` |
| 因果（GPT） | `(1/4)Σln(i+1) = 0.794513` |

于是 `HeadProfile` 同时给三个数：`entropy` / `ceiling` / `normalized_entropy = entropy/ceiling`。
**未训练模型**的第 0 层读数（演示脚本第九节）：

| 变体 | 熵 | 天花板 | 归一化熵 |
|------|----|--------|----------|
| `encoder_only` | 1.380526 | 1.386294 | 0.995839 |
| `decoder_only` | 0.792550 | 0.794513 | 0.997528 |

```text
熵相差 0.588 —— 但那 0.588 全是**掩码的账**
归一化之后只差 0.0017 —— 两者都几乎均匀（未训练的模型本来就该这样）
```

---

## 4. 六个读数（都能手算）

| 读数 | 定义 | 手算样本（均匀表 n = 4） |
|------|------|---------------------------|
| `entropy` | `−Σ p·ln p`（逐行求平均） | `ln 4 = 1.386294` |
| `peak_weight` / `peak_index` | 最大权重与它的列号 | `0.25 @ 0` |
| `support` | 权重 > 阈值的格子数 | `16`（全部） |
| `frobenius` | `√Σa²`（行随机表 ∈ `[1, √n]`） | `1.0` |
| `diagonal_mass` | 平均每行"看自己"的比例 | `0.25` |
| `sparsity` | `1 − support/格子数` | `0.0` |

对角质量与偏移质量：

```text
diagonal_mass   = offset_mass(record, 0)
offset_mass(k)  = 平均每行在 (i, i−k) 上的质量
k ≥ 列数时返回 0.0（合法读数：这一层不可能看那么远）
```

---

## 5. 头间冗余（逐格余弦）

```text
cosine(A, B) = Σa·b / (|a|·|b|)      逐格展平
手算：均匀表 vs 独热表 = 1/√n（n = 4 时 0.5）
```

未训练模型上三个层的非对角余弦：`0.9947 / 0.9998 / 0.9972`（平均 0.997268）——
**两个头几乎在看同一件事**（随机初始化下这是预期的）。

```text
掩码不同 → 拒绝：比的是"能看到多少"而不是"看了哪里"
形状不同 → 拒绝
```

---

## 6. 偏移质量（"它看的是相邻位置吗"）

| 层 | k=0（自看自） | k=1（前一个位置） | k=2 |
|----|---------------|--------------------|-----|
| 0 | 0.517519 | 0.356631 | 0.321163 |
| 1 | 0.522113 | 0.359689 | 0.290811 |
| 2 | 0.519803 | 0.350054 | 0.317717 |

```text
k = 1 是**归纳头**常出现的那条斜线（day075 的 induction 任务指向它）
因果掩码下 k ≥ i 的格子不存在 ⇒ 第 0 行会拉低平均（那是掩码的账）
```

---

## 7. 滚动

```text
Â = α·A + (1−α)·I        那一项 (1−α)·I 就是**残差**：让"这一层什么都没做"也有一条直通路
R = Â_L · Â_{L−1} · … · Â_1
```

三条性质（都可断言）：

```text
① R 仍然行随机           每个因子行随机 ⇒ 乘积行随机
② 掩码挡掉的格子仍然逐位为 0   下三角 × 下三角 = 下三角（那些 0 是**精确的**）
③ α = 0 时 R = I          把"这个公式在做什么"说清
```

演示脚本的读数（α = 0.5、3 层）：

```text
最后一层的归一化熵   0.998309
滚动之后的归一化熵    0.965952
集中度变化           +0.032358（更尖）
掩码外 (格子数, 最大读数) = (6, 0.0)
```

---

## 8. 六条性质

| 性质 | 判据 | "它会亮红"的反证 |
|------|------|-------------------|
| `rows_are_distributions` | 每一行非负、和为 1、有限 | 构造期：`AttentionRecord` 直接拒绝坏行和 |
| `masked_entries_are_exact_zero` | 掩码外的格子 `== 0.0` | 把被挡的格子填 0.001 ⇒ 亮红 |
| `entropy_within_ceiling` | 每一行熵 ≤ `ln(能看到的位置数)` | 掩码只允许 1 个位置而权重散在两个位置 ⇒ 亮红 |
| `heatmap_round_trip` | 渲染 → 解析逐级相同 | 解析侧：不认识的字符当场拒绝 |
| `rollout_is_stochastic` | 行随机 + 掩码外逐位为 0 | 某一层的被挡格子非 0 ⇒ 滚动之后仍非 0 |
| `deterministic` | 两次渲染/滚动逐位相同 | 没有随机数，因此不可能失败（如实说明） |

演示脚本第七节的证据（6 条记录、24 行）：

```text
[通过] rows_are_distributions | 6 条记录、96 个格子 | 最大的行和误差 1.110e-16（容差 1e-09）
[通过] masked_entries_are_exact_zero | 掩码外共 36 个格子，最大读数 0.000000e+00（逐位为 0）
[通过] entropy_within_ceiling | 24 行 | 最大的'熵−天花板' 0.000e+00
[通过] heatmap_round_trip | 6 条记录、24 行全部逐级相同（10 级是有损的）
[通过] rollout_is_stochastic | α=0.5 | 3 层 | 掩码外 6 个格子最大读数 0.000000e+00 | 上三角全 0：True
[通过] deterministic | 两次渲染逐字符相同：True | 两次滚动逐位相同：True
```

**两处如实说明**（"0 个格子"与"不可能失败"都要写出来）：

```text
全开掩码的表在这一条上被查 0 个格子（它什么都没挡）——那个 0 要印出来
deterministic 在无随机数的实现上不可能失败——它不是没有分辨力，而是"分辨力在别处"
```

---

## 9. 与既有包的接缝

```text
上游   arch_variants（day082 的三张权重表与两种掩码）
       multi_head（day076 的多头前向）
       encoder_decoder（day079 的层与块、layer_norm）
脚下   config 没有新增配置项：头数、阈值、α、级数都是函数参数
下游   day085（源码精读）：BertAttention / GPT2Attention 与 head_records 一一对应
       day088（项目原理串联）：五张表就是"原理 → 应用"分享提纲的骨架
```

依赖方向只有一条（`explainability → arch_variants → encoder_decoder → …`），
因此本包可以被单独导入，也可以在没有 torch 的机器上跑。

---

## 10. 两个真实的坑

```text
① 分头读法必须作用在"那一层的输入"上
   第一版把模型的原始输入喂给第 k 层 ⇒ 形状对、行随机、熵也算得出来，
   只是那些数属于"把输入直接喂给第 k 层"这件事
   判据：heads=1 在第 0 层逐位相等、在第 2 层不相等 ⇒ 读错了层
   修法：_stream_input_at_layer（单头路径重放到那一层）

② 报告里的"0 个格子"与"必然通过"必须写出来
   全开掩码的表没有可查的格子；deterministic 在无随机数时必然通过
   ⇒ 把它们当成"通过"会让判据显得比实际更强（day082 的"不适用 ≠ 通过"同源）
```
