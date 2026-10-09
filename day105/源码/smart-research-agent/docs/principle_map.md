# 项目底层原理串联手册（day088 / M7-D12）

> 本课不新增任何算术：它把 day073~day087 里散落的底层原理**串成一张可校验的图**。
>
> 手册里的每一个数都来自本课快照内的可复现读数：
> `python scripts/principle_map_demo.py`（十一节，产出 `outputs/principle_map_demo.txt`）
> 与 `python -m pytest`。权重由 LCG 生成、输入写死，因此同一个数可以被重新跑出来。

---

## 一、本课只承诺一件事：**图与对账**，不承诺新的模型能力

```text
承诺      把十二块拼图拼成一张可校验的图：
          原理（四层，2/4/3/3）→ 实现（项目里的包与函数）→ 应用（六个能力）
不承诺    新的模型能力、新的算子、新的性能读数——
          本课一行新算术都不写，只把既有读数对齐
```

因此本课的探针**只调用别人**：`math_foundations`（day073）、`transformer_core`（day075）、
`hf_source`（day085）、`positional_encoding`（day078）、`vectorstore`（day064）、
`inference_optim`（day087）。这是刻意的：自证是不成立的，跨包对账才是。

还有一条边界要写在最前面：

```text
本日不写任何新算法：全部动作是跨包对账与串联。
没有手写反向，也没有一个新的前向算术；连一个可微的式子都没有新增。
因此连续第四天**没有** GradientError 那一族（理由见第九节）。
```

---

## 二、十二块拼图总览

| id | 层 | 支撑应用 | 实现落点 | 来源 |
| --- | --- | --- | --- | --- |
| attention_is_differentiable_retrieval | math | agent_reasoning | `math_foundations.attention.scaled_dot_product_attention` | day073 |
| cosine_is_normalized_dot | math | rag_retrieval | `math_foundations.linalg.cosine` | day073 |
| attention_rows_are_distributions | attention | explainability | `transformer_core.types.assert_rows_are_distributions` | day075 |
| causal_mask_blocks_future | attention | generation | `transformer_core.layers.resolve_mask` | day075/079 |
| multi_head_splits_inside_projection | attention | serving | `hf_source.attention.split_heads` | day076/085 |
| position_encoding_breaks_permutation | attention | generation | `positional_encoding.layers.inject` | day078 |
| embedding_similarity_is_direction | representation | rag_retrieval | `vectorstore.metrics.cosine_similarity` | day041/064 |
| rank_ordering_matches_attention_peaks | representation | rag_rerank | `transformer_core.verify.compare_with_retrieval` | day075 |
| cache_reuse_is_bitwise_exact | representation | serving | `inference_optim.cache.compare_with_recompute` | day087 |
| cache_bytes_is_a_formula | inference | serving | `inference_optim.types.cache_bytes` | day087 |
| quantization_error_bounded_by_half_step | inference | serving | `inference_optim.quantize.error_bound` | day087 |
| generation_respects_budget_breakdown | inference | generation | `inference_optim.budget.plan` | day087 |

四层的条数是 **2 / 4 / 3 / 3**（数学 2、注意力 4、表征 3、推理 3）。演示读数：

```text
覆盖 12 条原理 / 4 层 / 6 应用 | 完整 True
```

---

## 三、第一层：数学地基（可微检索 / 余弦 = 归一化点积）

```text
检索（day066~071）   cos(query, doc) 排序 → 取前 k 条 → 拼进提示词   离散的选择，梯度传不回去
注意力（day073）     softmax(q·k) 打分 → 一组权重 → 对 value 加权平均  可微的混合，梯度能一路回传
```

演示读数（探针真的调用 `scaled_dot_product_attention`）：

```text
|output − weights·V| = 0.000e+00；行和与 1 的最大偏差 1.110e-16；负权重个数 0
cos((3,4),(4,3)) = 0.960000（手算 24/25）；cos(2a,b) = 0.960000（同向放大不变）
```

第一条读数（`|output − weights·V|`）就是"可微的混合"的字面含义：
输出**恰好**是 weights 对 V 的凸组合，因此"想要什么"是一条分布，而不是一次取舍。

---

## 四、第二层：注意力与结构（分布 / 掩码 / 分头 / 位置）

```text
行分布     weights 的每一行和为 1（被掩码的位置权重恰好 0.0）
因果掩码   mask[i][j] = (j <= i)；第 i 行看不到任何 j > i
分头       是一次 view + transpose，发生在**投影内部** ⇒ merge(split(x)) == x 逐位
位置       无编码时置换等变；注入位置表之后不再等变 ⇒ 这是模型知道顺序的证据
```

演示读数：

```text
行分布：5 行、最大行和偏差 1.110e-16（容差 1e-09）；最小权重 1.888e-01
因果掩码：严格上三角的最大权重 0.0；掩码表 == (j <= i) 为 True；可见位置总数 15
分头：2 个头，每头 4×2；往返逐位还原 True
位置：无编码的置换缺口 0.000e+00；注入位置表后的缺口 1.822e-03（应 > 0）
```

第 4 条读数里，**"无编码缺口为 0"与"有编码缺口大于 0"两个数必须一起看**：
只印后者，分不清"位置真的起作用了"与"实现里混进了随机性"。

---

## 五、第三层：表征与检索（方向 / 排序 / 缓存复用）

```text
方向相似   cos(5a, a) = 1.0、cos(-a, a) = -1.0 ⇒ 只看方向不看长度 ⇒ 入库前要归一化
排序对照   恒等投影 + 单位行时，注意力打分 q·k 就等于余弦，softmax 单调 ⇒ 排序完全一致
缓存复用   用缓存的 decode 与整段重算算的是同一批数 ⇒ 最后一行 logits 逐位相同
```

演示读数：

```text
cos(5a,a) = 1.000000；cos(-a,a) = -1.000000；与 day073 的差 0.000e+00
峰值一致 5/5；top-2 重合 10/10；秩相关 +1.000000
提示 3 个 token + 新 token 32；两条路径宽度 277（= 词表）；逐位相同 True
```

排序对照那一条有一个**必须写下来的前提**：恒等投影 + 行已归一化。
把这句话省略掉，"注意力排序与检索排序一致"就变成了一句**对所有参数都不成立**的断言——
探针因此把两个前提交代清楚，而不是假装它恒真。

---

## 六、第四层：推理与部署（缓存 / 量化 / 预算）

```text
缓存公式   cache_bytes = 2 · L · T · h · bytes；公式与逐层相加必须整数相等
量化上界   对称量化的实测误差 <= scale/2（可推导，不是经验）
预算拆分   权重 + 缓存 + 激活 = 总量；T_max 来自一次整除，两侧都要查
```

演示读数（与 day087 同源）：

```text
缓存公式：L=2、h=16、每步 256 字节；核对长度 [0, 1, 5, 16, 32]；不等个数 0
量化上界：int8 absmax per_tensor；scale=2.755906e-03；实测最大误差 1.355e-03 <= 上界 1.378e-03；SNR 36.06 dB
预算：权重 46144 + 缓存 8192 + 激活 8192 = 62528；T_max=37、放得下 True、T_max+1 放得下 False
```

量化那一条是**上界判定**（"两个操作数不该相等"），因此它的 `Evidence` 带 `upper_bound` 字段；
缓存公式与预算那两条是**整数相等**，没有容差空间。两类判据的写法必须分开（见第八节）。

---

## 七、覆盖：四层 × 六应用

```text
按层：math=2、attention=4、representation=3、inference=3
按应用：
  agent_reasoning  1 条 ✓ | attention_is_differentiable_retrieval
  rag_retrieval    2 条 ✓ | cosine_is_normalized_dot、embedding_similarity_is_direction
  rag_rerank       1 条 ✓ | rank_ordering_matches_attention_peaks
  generation       3 条 ✓ | causal_mask_blocks_future、position_encoding_breaks_permutation、generation_respects_budget_breakdown
  serving          4 条 ✓ | multi_head_splits_inside_projection、cache_reuse_is_bitwise_exact、cache_bytes_is_a_formula、quantization_error_bounded_by_half_step
  explainability   1 条 ✓ | attention_rows_are_distributions
没有入边的应用：[]
```

"没有入边的应用"这一行是本课最重要的一行：一个应用如果在图上没有任何原理支撑，
它就不是"知识"，而是一句口号。缺支撑时 `coverage_report()` 抛 `CoverageError`，
并且**消息里点名**是哪个应用或哪条原理。

---

## 八、七条性质与两类判据

```text
存在性      every_principle_has_artifact（12/12 可解析）、every_application_is_supported（6/6 有支撑）
可复现性    evidence_is_reproducible（12 条证据两次调用逐位相同）
跨天对账    attention_rows_are_distributions（调 transformer_core，偏差 <= 1e-09）
           cache_formula_matches_day087（调 inference_optim.cache_bytes，整数相等）
次序        outline_respects_dependencies（层次序 ['math','attention','representation','inference']）
完整性      document_covers_all_principles（文档缺失 0 条）
```

**注意判据的写法**：量化那一类性质是"实测 <= 上界"，而缓存公式那一类是"整数相等"。
因此本课的 `CrossCheck` 带 `upper_bound` 字段——有它时判据是"≤"，没有时才是"=="。
把两类混成一个判据，就会出现"实测误差恰好是 0（因为输入全是 0）被当成通过"这种事。

---

## 九、失败族与缺席的 `GradientError`（第四条理由）

```text
ShapeError      改调用：原理字段、应用与原理的数量、覆盖表的行列都要先对齐
ParameterError  改调用：层名 / 应用名 / top_k / 时长都是调用点的一次决定
NumericError    改数据或改实现：非有限读数、负上界、负容差都属于'数值不可用'
ClaimError      改命题或改探针：一条站不住的命题要改写，而不是删掉
ReferenceError  改引用：命题指向的模块或函数不存在时，先把名字改对
CoverageError   改图或改提纲：某个应用没有原理支撑、或某条原理没有落点时补上那一块
OrderError      改提纲：前置依赖（数学 → 注意力 → 表征 → 推理）不能被讲反
```

连续第四天没有 `GradientError`，而理由与前三天的**都不同**：

```text
day085  只读别人的推理路径（HF 的反向由 autograd 推出来）
day086  装进来的层都是别人写好的，全部落在前向之外
day087  量化不可微（round 的导数几乎处处为 0）
day088  本日一个新式子都没有：只做跨包对账与串联
```

一条纪律在这里第五次兑现：**能"改推导"的地方必须是有人真写了推导的地方。**

---

## 十、五条边界与后续接缝

```text
① 本包复现的是图与对账，不是新的模型能力——它不新增任何算术
② 十二块拼图是课程口径下的划分，不是"Transformer 原理的全部"
③ 本包不安装、也不调用 transformers / torch / numpy：跨天对账走本项目自己的实现
④ 探针输入全部写死，因此复现的是读数、不是统计规律
⑤ 本包不承诺分享提纲的"效果"——时长与次序可断言，"听众听懂了没有"不可度量
```

两个悬念，都留给下游：

```text
更多原理   这张图可以继续长（例如 day083 的可解释性还有几条没有落进来）
自动检查   "每条原理都有落点"已经可断言；下一步是"每次改动都重跑这张图"——那是 CI 的事
```

与后续的接缝：

```text
day099  结业项目：这张图会直接变成部署前的第一张检查单——
        任何一条没有落点的能力，都会在那一天以"跑不起来"的形式出现
```
