# MLOps 微调流水线（M5-D10 / day059）

> 本文是 `smart_research_agent/mlops/` 的设计说明，与
> [`scripts/mlops_demo.py`](../scripts/mlops_demo.py) 的实际输出逐项对应。
> 文中所有数字都是**本课程快照上的实测值**，不是经验估计。
> 复现命令（需要先把包放进 `sys.path`，或 `pip install -e .`）：
>
> ```bash
> cd day059/源码/smart-research-agent
> PYTHONPATH=. python scripts/mlops_demo.py
> ```

## 0. 这一课补的是最后一段

M5 走到第十天，散落的零件已经齐了：

```text
数据（day048/057） → 训练（day050/051/052） → 评估（day053）
                  → 对齐（day054/055） → 版本（day058）
```

但它们仍然是**九个可以单独运行的东西**。今天把它们收口成一条可被 CI 驱动、
可被评审读懂的流水线：

```text
ingest → train → evaluate → gate → package → publish
取数据    训练     评估       门禁     打包      发布
    ↓        ↓        ↓         ↓        ↓        ↓
  指纹    适配器    指标     门禁报告  模型卡   版本记录
```

七个模块，每个回答一个问题：

```text
tracking.py   这次实验用了什么参数、结果多少、产出了哪些文件
gates.py      这次的产物够格发布吗（绝对判定）
stages.py     六个阶段分别依赖什么、产出什么（可被程序校验）
packaging.py  它是什么、好不好、凭什么可信、**不能做什么**
ci.py         什么时候自动跑、按什么阈值判定
pipeline.py   把上面五件事串成一次可复盘的动作
```

## 1. 三条贯穿全包的纪律

| 纪律 | 具体表现 | 反例（如果没有它） |
|------|---------|------------------|
| **"出错"与"做不到"分开报** | 门禁拦下发布时 `publish` 是 `blocked`，`failed` 为 `False` | CI 的红灯掩盖真正需要看的那份门禁报告 |
| **缺失就是缺失** | `Run.metric` 返回 `None`；门禁缺指标时判**不通过** | "因为没测延迟所以延迟门禁通过" |
| **文档与实现同源** | 门禁表/阶段表/CI 摘要全部现场读出，且**仓库里的 workflow 由渲染器生成** | YAML 里的阈值悄悄漂移，没人知道谁是权威 |

第二条有一处必须说清的**反直觉**细节：同样是"缺数据"，两类判定的结论**相反**。

```text
相对判定（day058 evaluate_candidate）：缺数据 → 只能说"不知道"（gain = None）
绝对判定（今天 evaluate_gates）：      缺数据 → 必须说"不行"（不通过）
```

理由：相对判定问的是"候选比当前更好吗"，缺一个分数就没有可比的基准；
绝对判定问的是"这份产物够格发布吗"，**没有证据本身就是不发布的理由**。
把两者合并的那一天，就会出现上面那个"没测延迟所以通过"的事故。

## 2. 阶段表：把"顺序即策略"变成可程序校验的结构

六阶段的依赖与产出由代码生成（`stage_table()`），实测：

```text
  1. ingest    依赖 ['（无）'] → 产出 ['dataset_fingerprint']
     确认数据集指纹与基线：没有指纹的产物三个月后说不清来路
  2. train     依赖 ['dataset_fingerprint'] → 产出 ['adapter_sha256']
     调用训练回调，产出适配器（内容哈希是版本三元组的一半）
  3. evaluate  依赖 ['adapter_sha256'] → 产出 ['metrics']
     在领域评估集上评估适配器，产出指标（门禁唯一的判据来源）
  4. gate      依赖 ['metrics', 'adapter_sha256'] → 产出 ['gate_report']
     按发布门禁策略做绝对判定：不通过则 publish 被阻塞
  5. package   依赖 ['metrics', 'gate_report'] → 产出 ['model_card']
     生成模型卡与发布清单（产物要能自描述）
  6. publish   依赖 ['adapter_sha256', 'model_card', 'gate_report'] → 产出 ['version']
     把版本登记进注册表并提升为 stable（**上线是显式动作**）

主链：ingest → train → evaluate → gate → package → publish
```

`validate_stage_order()` 逐条检查"每个 `requires` 都能在它之前的 `produces`
里找到"。实测它能抓住顺序写错：

```text
  依赖写错的表被拒：阶段 publish 依赖了尚未产出的 adapter：顺序即策略——
  把产出它的那一步挪到前面，或者把依赖写对
```

**它抓住的第二类错误更值钱：依赖漏写。** 给 `package` 加一个对
`metrics` 的依赖却忘了写进 `requires`，平时不会报错——只会在某次运行时
出现 `KeyError`，而那时的堆栈指向阶段内部的某一行，不是"依赖表漏了一项"。

### 主链的取法：多个上游时取**最靠后**的那一个

`publish` 同时依赖 `adapter`（train）、`gate_report`（gate）与
`model_card`（package）。若取"第一个上游"，主链会变成
`ingest → train → publish`——**它漏掉了评估与门禁**。

因此 `critical_path` 在多个上游里取序号最大的那个（"离终点最近的上游"），
并有一条断言钉住这个选择（`test_critical_path_uses_the_latest_upstream`）。

### 一处被删掉的检查

第一版还有第五条判据："至少要有一个无依赖阶段，否则流水线永远开不了头"。
覆盖率报告指出它**不可达**——因为第一个阶段的 `requires` 必须落在空集里，
所以它必然为空。于是它被删掉了，理由写进了 docstring：

> **一条恒为真的断言不是保护，而是噪声。** 它会让人以为存在某种未被覆盖的
> 情形，从而在别处写出多余的兜底代码。

## 3. 门禁：六项检查、三类结果

### 3.1 单因素对照（实测，demo 第 3 节）

| 改动 | 结果 | 阻塞失败 | 告警/跳过 |
|------|------|---------|----------|
| 全项达标 | 通过 | 无 | 无 |
| 合格率 0.42 | 不通过 | `pass_rate` | 无 |
| 相对基线 -0.03 | 不通过 | `regression` | 无 |
| 适配器 80 MiB | 不通过 | `adapter_size` | 无 |
| 缺数据集指纹 | 不通过 | `dataset_traceability` | 无 |
| 缺合并产物 | 不通过 | `artifact_completeness` | 无 |
| 加了成本指标 0.09 | 不通过 | `cost` | 无 |

这张表的存在意义是**先看拦在哪一项，再决定改什么**——
把六项合成一个 `passed` 布尔值，运维能做的唯一动作就是重跑一次看日志。

### 3.2 三类结果必须分开命名

| 类别 | 判据 | 含义 |
|------|------|------|
| `blocking_failures` | 阻塞且未通过 | 决定 `passed`——发布被拦下 |
| `warnings` | 非阻塞且未通过 | 有人**显式关掉了**这项检查（一个值得被看见的决定） |
| `skipped_checks` | 非阻塞且没有观测值 | 缺证据（例如首次训练没有基线） |

如果不分开，运维看到"有 2 项没通过"时无法判断该去**补数据**还是
该去**把开关打开**——而这两件事的动作完全不同。

其中"关掉一项检查不是通过"这一点是刻意的：策略里
`require_traceability=False` 会把该项置为 `passed=False, blocking=False`，
因此它落进 `warnings` 而不是静默通过。

### 3.3 缺指标时的两侧（`when_missing` 一列）

六项里有三项"缺指标即不通过"、三项"缺指标则降级"：

| 门禁 | 缺指标时 | 理由 |
|------|---------|------|
| `pass_rate` | **不通过** | 合格率是核心结论，缺了它整次评估没有意义 |
| `adapter_size` | **不通过** | 体积不可知就决定不了部署成本 |
| `artifact_completeness` | **不通过** | 只有适配器、没有合并模型不算交付完成 |
| `regression` | 跳过（告警） | 首次训练没有基线 |
| `traceability` | 不通过（可被策略关闭） | 缺字段就重建不出三元组 |
| `cost` | 跳过（告警） | 离线训练不产生推理成本 |

`gate_table()` 把这一列渲染出来，因为它是最容易被忽略、也最容易出事的一列。

### 3.4 一个写反过的方向

`delta = 候选 − 基线`，**退步是负值**，因此 `max_regression=0.0` 的含义是
`delta >= 0`（一处都不许掉），而**不是** `delta <= 0`。

实现时写反过一次，表现是**所有改进都被拦下**——而"所有候选都被拒绝"
看起来很像"门禁很严格"，不会有人立刻怀疑方向写反。现在它由一条
回归断言钉住：

```python
def test_positive_delta_passes_the_regression_gate():
    report = evaluate({**BASE_METRICS, "eval_pass_rate_delta": 0.05})
    assert report.check(GATE_REGRESSION).passed is True
    assert report.check(GATE_REGRESSION).threshold == ">= -0.0"
```

## 4. 实验追踪：确定性 run_id 与"不能覆盖历史"

### 4.1 与 MLflow / W&B 的对应关系

本模块刻意照着 MLflow 的概念形状设计，换到真框架时是一次机械替换：

| 本模块 | MLflow | Weights & Biases |
|--------|--------|------------------|
| `ExperimentTracker` | `mlflow`（模块级） | `wandb.init()` |
| `Run` | `mlflow.start_run()` | `wandb.Run` |
| `log_params` | `mlflow.log_params` | `run.config` |
| `log_metrics(step=)` | `mlflow.log_metric(key, value, step)` | `wandb.log({...}, step=)` |
| `log_artifact` | `mlflow.log_artifact` | `wandb.Artifact` |
| `finish(status)` | `mlflow.end_run(status)` | `run.finish()` |
| `runs.jsonl` | 后端数据库 | 云端存储 |

**为什么不直接用真框架**：本课程的硬纪律是"离线、可复现、零外部依赖"。
引入 MLflow 需要一个长期运行的追踪服务（测试里就得起容器）；
引入 W&B 需要联网与账号。两者都会让"每天 2 小时"变成"每天先解决环境问题"。

### 4.2 两处与 MLflow 的刻意差异

实测（demo 第 4 节）：

```text
  第一次：97fa8e5a0da1 lora-ablation [running] 3 参数 / 0 指标（无指标）| 0 产物 | 0.00s
  reuse=True  → 同一个 run_id：True
  reuse=False → 新 id（同参数第二次运行）：97fa8e5a0da1-2
  改一个参数 → 全新 id：d6be7bb3c8b0
```

1. **`run_id` 是确定性的，不是随机 UUID**。公式是
   `sha256(name + canonical(params))[:12]`。于是"两次实验的 id 相同
   ⟺ 参数完全相同"这条性质成立——而这正是"这个数字是哪次跑出来的"
   能被直接回答的原因。MLflow 用随机 id，因此重跑永远产生一个新 run；
2. **同一个 `(指标名, step)` 不能重复记录**：

```text
  step 冲突被拒：指标 train_loss 在 step=42 已记录过（旧值 0.3187）：
  覆盖会让曲线变成'最后一次写入的曲线'，要记录第二次请换一个 step
```

第 2 条听起来苛刻，但覆盖写正是"loss 曲线"最容易失效的方式——
它不会报错，只会让历史安静地变成最后一次写入的样子。

无 step 的标量指标落在 `-1` 槽位（`{"-1": 0.72}`）而**不是 0**：
step 0 是一个真实的训练步（day050 的 warmup 就从第 0 步开始）。

### 4.3 三种比较关系与"没测过不算更差"

```text
  compare(mode=max): left=0.5493 right=0.58 delta=0.0307 → better
  compare(mode=min): left=0.5493 right=0.58 delta=0.0307 → worse
  缺指标时：comparable=False relation=None
```

`mode` 必须显式给出（`max` / `min`），**没有一个全局缺省**：同一个数字
（例如 loss）在监督微调里越小越好、在奖励模型里越大越好，给一个全局缺省
会诱导调用方忘记它，而忘记的后果是**结论反向**。

上面第一、二行是同一个 delta 得到相反结论——这正是"方向参数真的被用上了"
的证据。第三行是另一条纪律：**"没测过"不是"更差"，也不是"相等"**。

## 5. 端到端流水线：两条路径的逐阶段状态

实测（demo 第 5 节）：

```text
■ dry_run=True：ecca78533a34 | 全部完成 | 未发布 | 门禁 通过
    [ok     ] ingest       0.00 ms  数据指纹 aa77c31e90f4b258 | 基线 1.0.0 | 数据集已变化
    [ok     ] train        0.02 ms  适配器 sha256:5c6d7e8f9012 | 2 个产物
    [ok     ] evaluate     0.01 ms  指标 {'eval_pass_rate': 0.7222, 'eval_pass_rate_delta': 0.1222}
    [ok     ] gate         0.46 ms  发布门禁 通过 | 6/6 项通过 | 阻塞失败 无
    [ok     ] package      0.16 ms  模型卡 v1.1.0（minor）| 5 个指标 | 4 条已知限制
    [blocked] publish      0.00 ms  dry_run / publish=False：只演练到打包，未写注册表
    版本表规模 1，head=1.0.0
■ dry_run=False：ecca78533a34 | 全部完成 | 已发布 v1.1.0 | 门禁 通过
    …
    [ok     ] publish      0.89 ms  v1.1.0 已发布并置为 stable（键 14dceb829e28ce44）
    版本表规模 2，head=1.1.0
```

两次运行的 `run_id` 相同（`ecca78533a34`）——因为参数一字未改。
**但 `dry_run` 不同**，于是 `publish` 的结论不同。这不是矛盾：
`run_id` 由参数（输入）决定，而发布与否是**本次执行的选择**，
它记录在阶段结果里而不是 run_id 里。

### 5.1 `blocked` ≠ `failed`（本课的核心区分）

实测（demo 第 6 节）：

```text
  4f59b38146fe | 全部完成 | 未发布 | 门禁 不通过/未执行
    [blocked] gate      发布门禁 不通过 | 5/6 项通过 | 阻塞失败 ['artifact_completeness']
    [ok     ] package   模型卡 v1.1.0（patch）| 5 个指标 | 4 条已知限制
    [blocked] publish   门禁不通过，未写注册表：阻塞失败 ['artifact_completeness']
  failed=False（**blocked 不算失败**）
  追踪器里的 run 状态：finished
```

四处一致的表达，缺一不可：

| 位置 | 值 | 含义 |
|------|-----|------|
| `gate` 阶段 | `blocked` | 上游判定不允许继续 |
| `publish` 阶段 | `blocked` | 同上 |
| `outcome.failed` | `False` | **没有任何东西坏了** |
| run 状态 | `finished` | 这次实验正常跑完了 |

把它报成 `failed` 的代价是具体的：CI 会标红，而**那份门禁报告就在同一个
构建产物里**，被一个"失败"的标签挡住了。

### 5.2 门禁不通过**仍然打包**

`package` 阶段在两条路径上都是 `ok`。理由：那份报告与模型卡正是要给人看
的东西——"这一版为什么不能发"必须是一个可下载的结论，而不是一句
"流水线失败了"。

### 5.3 卡片版本号必须等于注册表版本号

同一份产物重跑时，`propose_version` 会算出下一个号（例如 `1.1.1`），
而注册表里那条记录是 `1.1.0`——两者指向**同一条记录**。

因此三元组在**打包前**就算出来，若已存在则卡片用**已有版本号**：

```python
existing = self.registry.find(triple.key)
if existing is not None:
    version_number = existing.version
    bump_kind = existing.tags.get("bump_kind", bump_kind)
```

这类"同一个东西两个编号"的缺陷不会报错，只会让追溯失效，
所以它有一条专门的断言（`test_rerun_card_version_matches_the_registry_record`）。

实测可见（demo 第 6 节）：那条路径上候选的三元组已经登记过，
于是卡片印的是 **v1.1.0（patch）** 而不是"新算出来的 v1.1.1"。

## 6. 模型卡：第四段才是它存在的理由

前三段（身份、指标、门禁）在 day058 的注册表里已经能回答，而"它**不能**
做什么"从来没有人写下来——于是每一次"模型答错了"的讨论都要从头争一遍
"这算不算预期内"。

四条限制，每条都带一个可核对的数字：

| 限制 | 数字依据 |
|------|---------|
| 领域评估集只有 18 条用例，合格率的最小可分辨变化是 `1/18 ≈ 0.0556` | day053 的评估套件 |
| 参考模型是 557×557 的双字符 bigram（约 31 万参数） | day050 的参考模型 |
| 训练数据来自单一来源、单一语言 | day048/day057 的数据画像 |
| 质量门禁只覆盖格式与治理，不覆盖事实正确性 | day057 的五维质量分 |

**把限制写成"可能在某些情况下表现不佳"是没有用的**，因为它无法被用来
否掉任何一个结论。而"小于 0.0556 的差异不可作为提升证据"可以。

有一处刻意的过滤：`gate_` 前缀的指标**不进卡片的指标表**。它们是门禁
自己的结论（`gate_pass_rate=1.0`），而卡片有专门的"发布门禁"一节。
把同一份事实在一份文档里写两遍且措辞不同，是让评审产生不信任的最快方式。

## 7. CI 集成：workflow 是**生成**的

### 7.1 与 day032 的 `ci.yml` 的分工

| | CI（day032） | 持续微调（今天） |
|---|---|---|
| 触发 | push / PR（事件驱动） | 定时（cron）+ 手动 |
| 耗时 | 分钟级 | 十分钟级以上（真实训练更久） |
| 输入 | 代码 | 代码 + 数据 + 上一版模型 |
| 失败意味着 | "这次提交不能合" | "这一版不值得上线" |

最后一行是关键的语义差异，也是为什么两条流水线不能合并。

### 7.2 三个地方只有一份权威

```text
settings.mlops_*  →  ReleaseGates  →  render_github_actions()  →  .github/workflows/finetune-nightly.yml
                                          ↓
                                    workflow_commands()（同时出现在 YAML、教程与终端里）
```

`tests/test_mlops_ci.py::test_committed_workflow_matches_the_renderer` 断言
**仓库里那份文件与渲染器的输出逐字相同**。因此"改门禁阈值"会同时在三个
地方生效，而它们之间只有一份权威；忘了重新生成，测试立刻变红。

实测（demo 第 8 节）：

```text
  文件：.github/workflows/finetune-nightly.yml
  cron：0 20 * * *（GitHub Actions 的 schedule 一律按 UTC 解释）
  超时：30 分钟 | Python 3.11
  门禁命令：python scripts/mlops_demo.py --min-pass-rate 0.5 --max-adapter-mebibytes 64.0
            --max-cost-per-1k-tokens 0.05 --max-regression 0.0
  仓库里的 finetune-nightly.yml 与渲染结果逐字相同：True
  yaml 行数 57，仓库文件存在 True
```

### 7.3 六步与两处细节

```text
checkout → setup-python → 装依赖 → 测试（代码没坏） → 门禁（该不该发） → 上传产物
```

- **产物上传必须是 `if: always()`**：失败时那份报告才是最需要看的；
- **cron 必须按 UTC 写**：GitHub Actions 的 `schedule` 一律按 UTC 解释，
  写成本地时间会出现"配的凌晨 2 点、跑在下午 2 点"。

### 7.4 YAML 1.1 的 `on` 陷阱

`yaml.safe_load` 会把裸 `on:` 解析成**布尔** `True`，因此
`payload["on"]` 会 `KeyError`，`payload[True]` 才对。这不是本课的 bug，
而是所有解析 GitHub Actions workflow 的脚本都会撞上的一件事——
所以它被写成一条可运行的证据（`test_parse_workflow_handles_the_boolean_on_key`）。

## 8. 测试与覆盖率

七个测试文件共 **222** 个用例，全部离线、确定性，一起跑约 **11 秒**：

| 文件 | 用例数 | 守什么 |
|------|--------|--------|
| `test_mlops_tracking.py` | 47 | 确定性 run_id、step 冲突、三种比较关系、落盘与折叠、环检测 |
| `test_mlops_gates.py` | 31 | 方向语义回归、六项单因素、缺指标的两侧、三类结果分离、表与策略同源 |
| `test_mlops_stages.py` | 25 | 依赖自洽、三类坏表、主链取法、状态语义 |
| `test_mlops_packaging.py` | 21 | 四条限制带数字、`gate_` 过滤、三元组必填、清单与卡片一致 |
| `test_mlops_pipeline.py` | 35 | 六阶段顺序、**blocked ≠ failed**、卡片与注册表版本一致、回调契约、失败留档 |
| `test_mlops_ci.py` | 21 | 渲染确定性与结构、**与仓库文件逐字一致**、YAML `on` 陷阱 |
| `test_mlops_api.py` | 42 | 四个端点、对照表与代码同源、400 vs 422、端到端 dry-run |

新包自身的覆盖率（`--cov=smart_research_agent.mlops --cov-branch` 实测）：
**752 条语句、174 个分支，整体 100%**，八个模块全部 100%。

为覆盖率做过的三处**设计调整**，都值得记下来：

1. `test_delta` 的方向写反被发现后，**加了一条专门的回归断言**——
   而不是只改代码。方向类缺陷的可怕之处在于"看起来像严格"；
2. `stages.validate_stage_order` 里"至少要有一个无依赖阶段"那条检查
   **被删掉**：覆盖率报告指出它不可达（第一个阶段的 requires 只能为空）。
   一条恒为真的断言是噪声；
3. `pipeline` 的两条支路（"回调只给最低要求字段"与"失败时渲染 markdown"）
   **补了专门的用例**，而不是把分支标成 `pragma: no cover`——
   它们是真实可走通的路径，只是之前的用例恰好都给了完整输入。

## 9. 三条可以带走的判断

1. **顺序要能被程序校验，而不只是写在注释里**。`requires` / `produces`
   的声明加上一次构造期校验，把"顺序写错"与"依赖漏写"变成启动即失败，
   而不是运行时的一个 `KeyError`；
2. **把"做不到"与"出错了"分开报**。`blocked`、`skipped`、`warnings`
   都是同一个模式：**一个悄悄违反自己约束的护栏，比没有护栏更糟**。
   门禁拦下发布时要说"这一版不该上线"，而不是"流水线失败了"；
3. **文档与实现同源要有"会变红"的机制**。门禁表、阶段表、CI 摘要都是从
   代码现场读出；而 workflow 文件更进一步——它与渲染器的输出被断言逐字
   相同，于是**改一个阈值就是一次可追溯的 diff**。
