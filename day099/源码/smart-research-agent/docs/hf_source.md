# Hugging Face 源码精读手册（day085 / M7-D9）

> 阅读对象：`huggingface/transformers` **5.x**（本文写作时最新为 v5.17.0，2026-09-09 发布）中的
> `src/transformers/models/gpt2/modeling_gpt2.py` 与
> `src/transformers/models/bert/modeling_bert.py` 的**推理路径**。
>
> 本手册回答一个问题：**那几行代码在做什么，以及怎么把"我读懂了"变成一条能失败的判据。**
>
> 本手册里的每一个数都来自本课快照内的可复现读数：
> `python scripts/hf_source_demo.py`（十一节、约 120 行输出，
> 产出 `outputs/hf_source_demo.txt`）。所有参数由 LCG 生成、输入写死，
> 因此"同一个数"不是一次运气。

---

## 一、本课只承诺两件事，也只承诺两件事

```text
承诺      给定同样的权重与输入，本包复现 HF 那几个类的**计算语义**
           （"ln_1 在自注意力之前""top_p 的核由数据决定"这类**结构**事实）
不承诺    某个符号在第几行、某个内部变量的名字、某个版本的中间张量形状
```

理由与 day080 的"参数量差值必须被逐项解释"同源：HF 的源码会随版本重构，
但结构事实跨版本稳定。因此本课把每一条断言都写成**行为**，
并在测试里给每一条配一个**反证**（"它会失败"也被证明过）。

还有一条边界必须写在最前面：

```text
源码精读读的是**推理**路径。
  modeling_gpt2.py / modeling_bert.py 里，反向不是被"写"出来的——
  它由 autograd 从这张计算图上推出来。
  因此本课**没有** GradientError 那一族：没有手写的反向公式可以错。
```

---

## 二、四个默认值：读源码读出来的第一样东西

配置文件里的默认值不是"一般认为"，而是事实。两个模型的第一处分岔就在这里：

| 项 | GPT-2（`GPT2Config`） | BERT（`BertConfig`） | 在源码里的位置 |
|----|----------------------|---------------------|----------------|
| LayerNorm 的 eps | `layer_norm_epsilon = 1e-5` | `layer_norm_eps = 1e-12` | 每个 `nn.LayerNorm(...)` |
| 激活函数 | `activation_function = "gelu_new"` | `hidden_act = "gelu"` | `MLP` / `intermediate_act_fn` |
| LN 的摆放 | **pre**（`ln_1` / `ln_2` 在子层之前，栈尾有 `ln_f`） | **post**（每个子层之后一次 LayerNorm） | 块内的接线顺序 |
| 注意力投影 | **融合**（一个 `c_attn` 出 `3·hidden` 维） | **三次独立**（`query` / `key` / `value`） | `GPT2Attention` vs `BertSelfAttention` |
| 嵌入后的 LN | 无（只在栈尾 `ln_f`） | 有（`word + token_type + position` 之后） | `GPT2Model` vs `BertEmbeddings` |
| `token_type` | 无 | 有（句子 A / 句子 B） | 同上 |

T5 的 `layer_norm_epsilon = 1e-6` 一并记下，因为"三个默认值"这件事本身就是结论：

```text
同一个 LayerNorm 公式，三个默认 eps：
  eps 越小，"方差离 1 多远"越小 —— 而那不是误差，是**定义**
  gpt2(1e-5)   实测方差 0.99999500   离 1   5.00e-06
  t5  (1e-6)   实测方差 0.99999950   离 1   5.00e-07
  bert(1e-12)  实测方差 1.00000000   离 1   5.00e-13
```

这张表同时复核了 day079 第 3.2 节那条公式：`variance = σ²/(σ²+eps)`，
实测与理论的最大差 `1.110e-16`（浮点求和的有理误差量级）。

---

## 三、「分头发生在投影内部」——本课最值钱的一句话

day079 与 day083 都记下过同一条**接口边界**："一个块只支持一种掩码"。
day079 的 `decoder_block` 要求自注意力是因果的，day082 的 `resolve_mask`
拒绝"因果 + 显式掩码"同时给。

Hugging Face 的注意力并不接受这条边界，而它绕开的办法只有两个形状重排：

```python
qkv   = c_attn(hidden)                                    # (n, 3·hidden)
query, key, value = qkv.split(hidden, dim=-1)             # 三个 (n, hidden)
query = query.view(n, heads, head_dim).transpose(0, 1)    # ← 就在这里
key   = key.view(n, heads, head_dim).transpose(0, 1)
value = value.view(n, heads, head_dim).transpose(0, 1)

scores  = query @ key.transpose(-1, -2)
scores  = scores * scale                                  # scale = 1/√head_dim
scores  = scores + bias                                   # ← 加性掩码
weights = softmax(scores, dim=-1)
context = weights @ value
context = context.transpose(0, 1).reshape(n, hidden)
output  = c_proj(context)
```

**`view` + `transpose` 就是"分头"的全部**——它没有换块类型、没有换掩码口径、
没有换缩放。因此同一个 `GPT2Attention` 既能跑 `heads=12` 又能跑因果掩码。

本包把这件事写成两条**逐位**性质：

```text
split_heads → merge_heads 逐位还原     heads=1 / 2 / 3 全部 True
切回来与三次独立投影逐位相同            True（三条投影路径的最大差 0.000000e+00）
```

---

## 四、加性掩码：`-1e9` 与 `-inf` 在 softmax 这一步**等价**

HF 的掩码不是"把被挡位置的权重写成 0"，而是"给打分**加**一个很大的负数"：

```python
bias = torch.tril(torch.ones(n, n)).view(...)   # 下三角 1、上三角 0
bias = 1.0 - bias                                # 上三角 1
bias = bias * torch.finfo(dtype).min             # 或直接填 -inf
scores = scores + bias
weights = softmax(scores, dim=-1)
```

它为什么与"显式掩码 + 只对允许位置归一化"**逐位相同**？因为 softmax 之前会
先减掉本行的最大值 `m`：

```text
被挡格子  v = s − 1e9 ≤ m − 1e9      ⇒   v − m ≤ −1e9
而        math.exp(-1e9) == 0.0       ← **精确下溢到 0**
```

所以被挡格子的权重**恰好是 0.0**，与显式掩码那一侧逐位一致。实测：

```text
两处权重的最大差：0.000000e+00（逐位相同：True）
被挡格子共 14 个，最大读数 0.000000e+00
```

它是**精确**的，而不是"数值上差不多"——而它成立有两条前提，本包把两条都做成守卫：

```text
① 本行至少有一个允许位置            否则 softmax 的分母是 0  ⇒ AssemblyError
② 打分远小于 floor（本包要求 |score| < floor/2）  ⇒ AssemblyError
```

第 ② 条不是形式主义：如果某个**合法**位置的打分本身就接近 `-1e9`，
被挡格子与合法格子会挤进同一个下溢区间，"恰好 0.0"就不再成立。

---

## 五、因果性用**扰动**量出来

day082 的纪律在这里第二次兑现：**"因果"这件事不读 `causal=True` 四个字**，
而是扰动一个输入位置，看输出怎么变。

```text
把第 n−1 行乘 1.5，重新前向
  第 i < n−1 行：输出**逐位不变**（被掩码挡掉的格子从不参与 exp / sum / 加权）
  第 n−1 行：    确实变了（否则这条检查没有分辨力）
```

实测：

```text
gpt2（因果）：前 3 行逐位不变 3/3；最后一行变了 True
bert（双向）：前 3 行逐位不变 0/3；最后一行变了 True
```

BERT 那一行是这条判据的**反证**：双向前向下"未来"确实会影响过去，
因此 `0/3` 是**正确**的读数，而不是一个失败。

---

## 六、同一件事的两种写法：两条跨天对账

本课从 HF 读出来的东西，与本课程自己写过的两个包必须落在同一个值上。
这正是"两份实现之间要么消灭差异、要么写下差异"的第三次兑现：

| 对账 | 本包 | 另一端 | 判据 | 实测 |
|------|------|--------|------|------|
| `heads=1` 的注意力 | `hf_source.attention` | day076 `multi_head.layers` | 容差 `1e-12` | 最大差 `0.000e+00` |
| pre 摆放的块 | `hf_source.blocks` | day079 `encoder_decoder.layers` | 容差 `1e-12` | 最大差 `0.000e+00` |
| 缩放系数 | `hf_source.attention.hf_scale` | day076 `head_scale` + day075 `softmax_shape_scale` | **逐位** | 三者相等 `0.5773502691896258` |

必须说清的一条：**前两条的判据是容差，不是逐位**——
两份实现各自组织浮点运算的顺序不必相同（一处乘 `1/√head_dim`、一处除 `√head_dim`）。
本课实测恰好是 `0.000e+00`，但**不把"偶然逐位"写成承诺**：判据仍是容差。

---

## 七、两个激活：`gelu` 与 `gelu_new`

```text
BERT   hidden_act = "gelu"       0.5x(1 + erf(x/√2))                      精确式
GPT-2  activation_function = "gelu_new"
                                 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))   tanh 近似
```

两式的差别可以被**量出来**（逐点绝对差，单位是函数值）：

```text
x= -4.0  差 5.644e-05      x=  0.0  差 0.000e+00      x=  4.0  差 5.644e-05
x= -2.0  差 9.796e-05      x= +0.5  差 1.722e-05
x= -1.0  差 1.528e-04      x= +1.0  差 1.528e-04
x= -0.5  差 1.722e-05      x= +2.0  差 9.796e-05
```

`x = 0` 那一行是唯一可被手算断言的一行：两式的**函数值同为 0**（因子 `0.5x`）、
**导数同为 0.5**（`0.5·(1 + tanh(0)) = 0.5`，与 day079 第 4.3 节量过的是同一个数）。
逐点最大绝对差 `1.528e-04`——**它不是"差不多"，它是一个可以被写进表的数**。

---

## 八、生成：四个 warper 与它们的顺序

`GenerationMixin` 把两件事放在同一个循环里，而它们其实是独立的：

```text
① 对 logits 做一连串**纯函数变换**（warper / processor）——不改模型、不碰缓存
② 从变换后的分布里**选一个 token**——贪心 / 采样 / 维护一批 beam
```

本课把 ① 拆成四个按固定顺序执行的纯函数：

```text
RepetitionPenalty   score = score < 0 ? score·penalty : score/penalty   ← **除**，不是减
Temperature         score = score / temperature
TopK                只留最大的 k 个，其余置成"极小值"
TopP                升序排序 + 累积 softmax + 删掉 cumsum <= 1−p 的位置 + 至少留 1 个
```

顺序即语义，而它的两个可读后果：

```text
温度**不改变排名**（乘正数），因此它放在 top_k 前后是安全的
top_k 与 top_p 之间是**串联**：top_p 的核是在**已经截断过的**分布上算的
```

一个会被读错的地方：`top_p = 0.5` **不等于**"留一半的词"。它留的是
"累积概率刚超过 `p` 的最小集合"，因此**保留个数由数据决定**：

```text
分布的 logits = (2, 1, 0.5, 0, -1, -2)      （写死的，可手算）

top_p=0.10 | 保留 1 | 删掉 5 | 留下的最小累积概率 0.5573
top_p=0.30 | 保留 1 | 删掉 5 | 留下的最小累积概率 0.5573
top_p=0.50 | 保留 1 | 删掉 5 | 留下的最小累积概率 0.5573   ← **退化成贪心**
top_p=0.80 | 保留 3 | 删掉 3 | 留下的最小累积概率 0.8866
top_p=0.95 | 保留 4 | 删掉 2 | 留下的最小累积概率 0.9620
```

而 `top_k` 正好相反：

```text
top_k=1 | 保留 1 | 峰值 1.0000      top_k=3 | 保留 3 | 峰值 0.6285
top_k=2 | 保留 2 | 峰值 0.7311      top_k=6 | 保留 6 | 峰值 0.5573
```

**保留个数是常数**（由参数决定），而 `top_p` 的保留个数是**数据决定的**——
把两者混着读，就会得到"我设了 0.5 却退化成贪心"这种看起来像 bug 的读数。

还有一条"不可能失败"的读数：HF 的 `TopPLogitsWarper` 里有一行
`sorted_indices_to_remove[..., -min_tokens_to_keep:] = 0`，它的作用是把
"至少留一个 token"写成一条**恒等式**。本课不把它当判据，而是把
**保留个数的最小值**如实印出来（day083 第 7.2 节的同一条纪律）。

---

## 九、两种搜索：贪心与采样各一条，beam 一条

```text
贪心    每步取 argmax（并列时取最小下标，与 torch.argmax 一致）——**完全确定**
采样    温度缩放后按整条分布逆变换采样；随机性只在"那一串均匀数"里
beam    同时维护 num_beams 条前缀，按长度惩罚后的分数选最终序列
```

同一个玩具模型（确定性）、同一个种子（`seed=7`）：

```text
  greedy  生成 [7,6,6,6,6,6]   逐步保留 [8, 8, 8, 8, 8, 8]
sample   生成 [2,7,5,7,1,6]   逐步保留 [8, 8, 8, 8, 8, 8]
top_k=2  生成 [6,6,7,6,6,7]   逐步保留 [2, 2, 2, 2, 2, 2]
top_p=0.7 生成 [5,7,6,7,5,7]  逐步保留 [4, 4, 3, 3, 3, 3]
```

两条值得指出的读数：

```text
① top_k 的"逐步保留"是常数 2，而 top_p 的是 (4,4,3,3,3,3) —— **随步变化**
② 同一个种子给出同一串 token（采样在"u 的来源"之外是纯函数）：
   换成 seed=14 立刻得到另一串
```

而 beam 与贪心的差别不是"更宽就更准"，是**优化目标不同**：

```text
beams=1 lp=1    生成 [0,7,5,7,6] | 跑满 max_new_tokens
beams=2 lp=1    生成 [6,6,6,6,6] | 归一化分数 -1.350130
beams=2 lp=0.5  生成 [6,6,6,6,6] | 归一化分数 -3.018983
beams=3 lp=2    生成 [6,6,6,6,6] | 归一化分数 -0.270026
```

贪心的第一个 token 是 `0`（那一步的 argmax），而 beam 的第一个 token 是 `6`——
因为 beam 优化的是**整条序列的累积对数概率**。长度惩罚那一项
`final = score / length ** length_penalty` 不是装饰：

```text
每一步都在乘一个小于 1 的概率（对数概率**非正**）⇒ 候选越长、总分越负
lp = 0     不做任何修正      ⇒ 系统性偏向短序列（默认的陷阱）
lp > 0     把"越负"摊薄       ⇒ 偏好更长的序列（lp 越大越强）
lp < 0     反过来放大"越负"   ⇒ 偏好更短的序列
```

这一处最容易读错的地方是**名字与方向不一致**：参数叫"长度惩罚"，
而 HF 的文档口径是 `lp > 0` **promotes longer sequences**。
本包因此明确拒绝负值（`GenerationError`）——它是一次方向反转，
而一个已经名不副实的参数再翻一次方向，读代码的人就只能靠试。

---

## 十、失败族、边界与常见错误对照表

### 10.1 五个失败族（**缺席的是 `GradientError`**）

```text
ShapeError        形状不符      头数不能整除、掩码与打分不一致、位置数超表长    → 改调用
ParameterError    参数越界      temperature / top_k / top_p / penalty 非法      → 改调用
AssemblyError     组装不成立    fused 切错列、加性掩码当权重、布尔掩码被当数      → 改调用
GenerationError   策略不成立    核被清空、beam 宽度 > 词表、长度惩罚为负          → 改调用（换策略）
NumericError      数值不可用    非有限数、行和不是 1、logits 全是 -inf            → 改数据或改实现
```

`AssemblyError` 与 `GenerationError` 都同时是"参数失败"，但**值得有自己的名字**：
它们要求调用方**同时**看两个部件（投影与切法、策略与分布），
而"层号越界"只需要改一个数——**修法不同，族就不同**。

### 10.2 五条边界（不承诺的事）

```text
① 复现的是计算语义，不是 HF 的内部实现细节（变量名、缓存类、中间张量布局）
② 只读推理路径：没有手写反向，因此没有"改推导"那一族
③ 采样用可注入的均匀数：不承诺对 torch.multinomial 逐位复现
④ beam 复现"长度惩罚 + 早停"的打分与选择语义，不承诺并列打破顺序与 HF 逐位一致
⑤ 画像来自**配置默认值**；"哪个更适合你的任务"仍然要回到任务指标
```

### 10.3 常见错误对照表（12 条）

| # | 读错的地方 | 正确的读法 |
|---|-----------|-----------|
| 1 | 以为 GPT-2 与 BERT 的 LayerNorm 是同一个（同名同公式） | 同名同公式、**eps 不同**（1e-5 vs 1e-12），输出不相等 |
| 2 | 以为两个模型的激活都是 `gelu` | BERT 是 `gelu`（erf），GPT-2 是 `gelu_new`（tanh），差 `1.528e-04` |
| 3 | 以为"分头"要换一个块类型 | 它是投影内部的一次 `view` + `transpose` |
| 4 | 以为融合投影与三次独立投影是两种算术 | 它们是**同一个**：切回来逐位相同（本包最值钱的逐位判据） |
| 5 | 以为掩码是"把被挡位置的权重置 0" | 它是**加性**的：先加一个大负数，softmax 的下溢让权重恰好为 0 |
| 6 | 以为 `-1e9` 是"近似 `-inf`" | 在"先减最大值"的 softmax 下两者**逐位等价**（`exp(-1e9) = 0.0`） |
| 7 | 读 `causal=True` 就认为它是因果的 | 因果要**用扰动量**：前 3 行逐位不变才算数 |
| 8 | 以为 `top_p=0.5` 会留一半的词 | 保留个数**由数据决定**；一个尖分布上它会退化成贪心 |
| 9 | 以为 `top_k` 与 `top_p` 互不影响 | 两者**串联**：`top_p` 的核是在截断过的分布上算的 |
| 10 | 以为重复惩罚是"减一个常数" | 它是**除**：正 logits 除以 penalty、负 logits 乘以 penalty |
| 11 | 以为 beam 一定优于贪心 | 它优化的是整条序列的累积对数概率，因此**第一步可能不是 argmax** |
| 12 | 以为"长度惩罚"越大越偏向短序列 | 打分是非正的，因此 `lp > 0` **偏好长序列**、`lp = 0` 才系统性偏短（本包拒绝负值） |

---

## 参考

- 教程：[../day085/教程/教程.md](../day085/教程/教程.md)
- 完整源码：[../day085/源码/smart-research-agent/](../day085/源码/smart-research-agent/)
- 离线演示：[../scripts/hf_source_demo.py](../scripts/hf_source_demo.py)（十一节）
- 上一课：[../day083/教程/教程.md](../day083/教程/教程.md)（可解释性与注意力可视化）
- 相关课程：[../day079/教程/教程.md](../day079/教程/教程.md)（块与两种反向）、
  [../day082/教程/教程.md](../day082/教程/教程.md)（三个变体）、
  [../day076/教程/教程.md](../day076/教程/教程.md)（多头注意力）
- 课程索引：[../docs/curriculum.md](curriculum.md)
