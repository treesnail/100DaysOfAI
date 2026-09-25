# RAG 评估与调试手册（day071 / M6-D9）

> 对应模块：`smart_research_agent/rag_debug/`（8 个文件）
> 对应端点：`GET /rag/eval/status`、`POST /rag/eval/run`
> 对应演示：`scripts/rag_debug_demo.py`（九节，输出见 `outputs/rag_debug_demo.txt`）
> 对应基线：`data/eval/rag_baseline.json`（随仓库提交）

---

## 1. 这一层在整条链路上的位置

```text
数据侧（离线，一次）                          推理侧（在线，每次）
bytes → Document → Chunk → VectorRecord → 索引清单 → 命中 → 融合 → 重排 → 上下文 → 答案
  day061  day061   day062    day064       day065   day066  day067  day068  day066  day069
                                                    └─────────── 检索的账 ───────────┘
                                                                          └─ 打包的账 ─┘
       ↑                                                                          ↑
       └──────────────────  day071：这一层给上面每一层打分、归因、记基线  ─────────┘
```

day071 **不生产任何被检索的内容，也不改变任何一次回答**。它只做三件事：

```text
评估  在一批带金标准的用例上跑整条链路，算出 11 个质量指标
归因  把"这一条不好"折成一个标签（坏在哪一层）+ 一个动作（下一步动哪个旋钮）
基线  把那 11 个数字存进仓库，于是"这次比上次差吗"有了可判定的答案
```

---

## 2. 核心创新：把"召回"拆成两段

在此之前，"召回"只有一个数字（检索名单里有没有金标准）。但链路上有两个**不同的名单**：

```text
retrieved   检索器交出的那份          recall@k        ← day066~day068 的全部努力
packed      真正进了提示词的那份       packed_recall@k ← day069 的 given 减去丢失的部分
```

金标准"在名单里但不在提示词里"是一门**独立的失败**：

```text
检索抓对了    → 不是检索的错
模型没机会用  → 不是生成的错
是**预算的错** → 处置动作：调 retrieval_max_context_chars / retrieval_per_hit_chars
```

因此本层把 `StageMetrics` 分成两段（`STAGE_RETRIEVED` / `STAGE_PACKED`），
并额外产出 `CaseOutcome.packed_away`（丢了几**条**依据）与 `METRIC_PACKED_RECALL`。

**读报告的姿势**：`retrieval_recall` 与 `packed_recall` 的差额就是"预算吃掉了多少依据"。
差额为 0 说明这次打包没有伤到召回——这本身是一个**结论**，
而它以前只能靠"答案看起来变短了"去猜。

---

## 3. 11 个质量指标与它们的方向

| 指标 | 方向 | 说明 | 归谁管 |
|------|------|------|--------|
| `retrieval_recall` | higher_better | 检索名单的召回率 | 检索 |
| `retrieval_precision` | higher_better | 检索名单的精确率 | 检索 |
| `retrieval_ndcg` | higher_better | 检索名单的 NDCG | 检索 / 排序 |
| `packed_recall` | higher_better | **进了提示词**那一段的召回率 | 打包预算 |
| `packed_ndcg` | higher_better | 进了提示词那一段的 NDCG（模型看到的次序） | 打包 / 排序 |
| `grounded_rate` | higher_better | 接地率（day069 的 `grounded`） | 生成 |
| `mean_coverage` | higher_better | 平均覆盖率（有效引用 / 给出去的片段） | 生成 |
| `hallucination_rate` | lower_better | 幻觉引用率 | 生成 |
| `llm_call_rate` | **neutral** | 模型真的被调用的比例 | 描述性，不判方向 |
| `fallback_rate` | lower_better | 走了回退的比例 | 生成 / 检索 |
| `bad_case_rate` | lower_better | 归因判为坏例的比例 | 全部 |

`llm_call_rate` 为什么没有方向：它下降既可能是护栏生效（检索为空时不调模型，设计如此），
也可能是"该调没调"。给它硬安方向会制造一类"看起来像回归、其实是设计"的误报；
而真正的异常会由 `fallback_rate` / `bad_case_rate` 如实报出来。

---

## 4. 12 类坏例标签与它们的处置动作

判据按**链路的顺序**排列，命中即返回（`BAD_CASE_PRIORITY`）：先出现的先怀疑。

| 顺序 | 标签 | 是什么问题 | 下一步动作 |
|------|------|-----------|-----------|
| ① | `no_data` | 库是空的 | 先灌语料 / 建索引（前置条件，不是模型问题） |
| ② | `retrieval_empty` | 检索为空（其余三种原因） | 按 `empty_reason` 那一个原因处置 |
| ③ | `retrieval_miss` | 金标准一条都没抓到 | 查编码器 / 分块粒度 / 索引版本 |
| ④ | `rerank_cut` | 被重排阈值切掉 | 调 `retrieval_rerank_min_score` |
| ⑤ | `packed_away` | 抓到了但没进提示词 | 调上下文预算（**这一课独有的那门失败**） |
| ⑥ | `rank_bad` | 进了提示词但排太后 | 排序侧：重排开关 / 融合策略 |
| ⑦ | `truncated_chunk` | 片段被单块上限截断且覆盖率低 | 调大 `retrieval_per_hit_chars` |
| ⑧ | `generation_fallback` | 走了六种回退之一 | 按 `fallback_reason` 选动作 |
| ⑨ | `hallucinated_citation` | 引用了不存在的编号 | 收紧提示词引用要求 / 换模型 |
| ⑩ | `ungrounded` | 一个有效引用都没有 | 看提示词引用要求；再考虑 `retrieval_require_citation` |
| ⑪ | `low_coverage` | 覆盖率低于下限 | 先看检索给的片段贴不贴题 |
| ⑫ | `weak_faithfulness` | 与参考答案对比忠实度低 | 逐句对照：生成侧与检索侧各核一次 |

**为什么只给一个标签**：一条坏例往往同时满足多个判据（检索没抓到的时候，答案当然也没有引用），
但它们的因果顺序是固定的——⑩ 是 ③ 的后果。一次给出全部标签会把因果关系摊平成清单，
而读的人只会去修那个最显眼的（最后一个）。

---

## 5. 质量基线：与性能基线方向相反

| | 性能基线（day046） | 质量基线（day071） |
|---|---|---|
| 指标 | 延迟 / token / 费用 | 召回 / 接地 / 覆盖率 |
| 好坏方向 | **越大越差** | **越小越差** |
| 判定口径 | **比值**（`current / baseline > tolerance`） | **差值**（`delta < -tolerance`） |
| 基数为 0 | 跳过该指标（0 的倍数无意义） | 照样可判（`0 → 0.05` 就是 5 个百分点） |
| 第三档方向 | 无 | `neutral`（描述性指标，永不判） |

质量侧必须用差值，三条理由都能独立成立：**方向反转**（否则每次改善都会被判成回归）、
**0 基数是常态**（幻觉率的健康值就是 0，跳过它等于"从零变坏"永远不被发现）、
**量纲可比**（都是 `[0,1]`，差值就是"几个百分点"）。

三档判定的完整规则（`baseline.deltas_between`）：

```text
|delta| <= dead_band（缺省 0.02）        → unchanged（噪声）
方向是 neutral                            → unchanged
变化方向是「好」                          → improved
变化方向是「差」且 |delta| > 容忍度       → degraded
变化方向是「差」但 |delta| <= 容忍度      → unchanged（**允许变差多少**的语义）
```

---

## 6. 三道"能不能比"的检查

`RagBaselineGuard.compare` 先查三件事，**任何一条不过就不给结论**：

```text
k 口径不同        前 3 条与前 5 条不是同一件事        → 口径变了
index_version 不同 换了索引/编码器，历史数字作废        → 条件变了
样本不足          current.samples < min_samples        → 样本不足
```

"不下结论" **不等于"通过"**：`ok = conclusive and not regressions`，
因此样本不足时 `ok=False` 且 `regressions=()`——读报告的人看到的是 `inconclusive`，
而不是一个会让人放松警惕的绿灯。**门禁最危险的行为是在数据不足时装作有结论。**

`prompt_version` 的待遇刻意不同：它是 A/B 的**自变量**（day069 留下的伏笔），
因此两边不同不影响可比性，只会进 `variables` 被记下来。

---

## 7. 变体：一次只改一个旋钮

```text
baseline       当前默认配置（提示词 v2、不强制引用）
prompt-v1      只换提示词版本 v2 → v1（A/B 的经典自变量）
cite-gate      只开 require_citation 闸门（不可核对的答案会被整条丢掉）
tight-budget   只收紧上下文预算（用来量 packed_away）
```

变体只允许声明几个具名旋钮（`prompt_version` / `require_citation` / `top_k` /
`max_context_chars`），并强制写一句 `note` 说明"它改了什么"。
一个变体不允许一次改两件不相关的事：那样跑出来的差值无法归因，
而"无法归因的实验"与"没有实验"的信息量相同（与 `measure_lift` 坚持用同一批命中同源）。

---

## 8. 端点的边界行为

```text
GET  /rag/eval/status   跑 0 条用例、0 次模型调用；基线不存在时给 null（不造全零基线）
POST /rag/eval/run      跑当前 app 注入的那个库（评估对象 = 服务的对象）
```

| 情形 | 行为 |
|------|------|
| 库为空 | 200，每条用例归因 `no_data`（"还没灌语料"≠"检索坏了"） |
| 基线文件不存在 | `comparison=null`（不造一份全零基线） |
| 基线与当前不可比 | `comparison.conclusive=false` + `reasons` 逐条说明，**不报假回归** |
| 变体名不存在 | 400，消息里列出可用变体名 |
| 未配置 LLM | 400（与 `/retrieval/answer` 同一条通道），消息给出出路 |
| 模型是 MockLLM | **允许**（与 `/retrieval/answer` 不同）：本端点的产物是一份**报告**，而报告会如实记着 `llm_called` 与回退分布 |

---

## 9. 已知限制（本次结论只在以下范围内成立）

`RagEvalReport.limitations` 直接搬 `retrieval.RETRIEVAL_LIMITATIONS`——
一份报告不说清"它的结论在什么范围内成立"，就等于让人把一次实验室里的测量当成生产结论：

- 接地校验只做**引用编号**层面的核对（`grounded` 不是 `correct`）；
- 拒答检测靠固定句式（`DECLINE_MARKERS`），换一种说法就检测不到；
- 重排只作用于前 N 条窗口，窗口之外一条都不打分；
- 重排用的是教学级交叉编码器替身，分数只能用于**相对排序**；
- 阈值只对向量通道有效（BM25 没有绝对标度）；
- 关键词一路的 CJK 用 2-gram 代替分词器（不做词形还原与同义词）；
- 不做权限过滤（`where` 是元数据筛选，不是访问控制）。

另外两条属于本层自己：

- **评测集只有 6 条用例**：指标能反映"方向"，不足以做统计显著性判断
  （`min_samples` 缺省 3 就是为了拒绝"用两条用例宣告回归"）；
- **编码器是离线字符 n-gram**（`default_embedding()`）：它只做字面相关，
  因此演示里的召回率不代表真实语义检索的水平——**要真实数字请换真编码器**。

---

## 10. 常见误区

1. **"packed_recall 就是 recall。"** 错。两段名单不同：检索交出的 vs 进了提示词的。
2. **"质量基线可以复用性能基线的判定。"** 错。方向相反（越大越差 vs 越小越差），比值 vs 差值。
3. **"幻觉率的基线是 0，所以不用判。"** 错。差值在 0 基数下照样可判——跳过它等于"从零变坏"永远不被发现。
4. **"样本不足时算是通过。"** 错。`conclusive=False` 与 `ok=True` 是两回事。
5. **"换了索引之后还能和旧基线比。"** 错。`index_version` 变了就不可比（day065：换编码器必须重建索引）。
6. **"换提示词版本也不能比。"** 错。它是 A/B 的自变量，进 `variables` 而不是阻塞。
7. **"一条坏例应该列出所有问题。"** 错。因果有顺序，只报最早的哪一层。
8. **"坏例率低就是系统好。"** 不完全：判据（k / 覆盖率下限 / 是否要求接地）本身就是结论的一部分，换判据要新开基线。
9. **"`/rag/eval/run` 跑的是线上库之外的一份干净副本。"** 错。它跑的就是注入进 app 的那个库——评估对象必须与服务对象是同一套。
10. **"评审和作答用同一个模型没问题。"** 有已知风险（同盲区互相包庇，见 day027）；本层把它做成可注入项，但演示里为了确定性用了同一个替身。
