"""``torch_pipeline``：把"数据 → 设备 → 训练 → 检查点 → 推理"做成可复算的管线（M8-D7 / day096）.

day095 把训练技巧装成四个旋钮，并留下两个工程问题：**数据怎么喂、状态怎么存**。
今天把这两个问题连同一件"设备算术"讲完，并跑一个端到端的小项目：

```text
数据      不可变样本 + 内容指纹 + 按比例的确定性切分          —— 本包新建
采样      打乱只依赖 (seed, epoch)；worker 分片取模且并集为全集 —— 本包新建
装载      按批切分、可选丢尾批；三个读数（批数 / 丢弃 / 分片） —— 本包新建
设备      参数 + 激活 + 优化器状态 → 字节数，与预算对账       —— 本包新建（纯算术）
训练      DataLoader → 前向 / 反向 → 范数守卫 → 优化器 step    —— 复用 day095 / day094 / day092
检查点    五件套 + 清单哈希；缺文件点名、哈希不符当场拒绝       —— 本包新建（纪律承自 day050）
推理      只走推理相；可注入时钟让延迟读数逐位复现             —— 本包新建
```

## 一、今天最值钱的一句话

> **"中途停下再恢复"与"一口气跑完"必须给出同一批参数——
> 这不是性能问题，而是"检查点到底存没存对"的定义问题。**

本课把它写成一条性质（第 ⑥ 条）：它一次性检验打乱只依赖 ``(seed, epoch)``、
训练相的归一化不依赖 running、前向里没有随机性、以及优化器的跨步状态真的被装回去了。
其中任何一件出错，读数都会从 ``0.0`` 变成一个非零数——而其余六条性质会照常通过。

## 二、七条性质

```text
① shuffle_order_is_deterministic      相等   同 seed 两次 order 的差异项数 = 0
② worker_shards_partition_the_dataset 下界   并集覆盖 = 1.0 且不相交项数 = 0
③ drop_last_matches_batch_formula     相等   批数与 n // b 的差 = 0
④ device_bytes_match_formula          上界   total_bytes 与手算的相对差 <= 1e-9
⑤ checkpoint_round_trip_is_bitwise    相等   save → load 之后压平参数逐位相同
⑥ resume_matches_uninterrupted        相等   中途恢复与一口气训到第 N 轮参数差 = 0  ← 核心
⑦ predict_batch_equals_sequential     相等   批量推理与逐条推理的输出差 = 0
```

判据三类：**相等**、**不超过上界**、**不低于下界**。第 ② 条是唯一的下界——
没有它，把取模分片改写成切块分片时，其余六条会照常通过（漏样本不会报错）。

## 三、十一个模块

```text
errors.py      五个失败族（含 DeviceError / CheckpointError）
types.py       七个阶段 / 四条读数轴 / 三个设备 / 四种精度 / 七条性质 / 十条笔记 / 五条边界 / 十张公式
datasets.py    不可变数据集 + 内容指纹 + 确定性切分
sampler.py     确定性打乱（复用 LCG）+ 取模分片
dataloader.py  按批切分 / 丢尾批 / 先分片再切批的三个读数
device.py      参数量 / 激活 / 优化器状态 → 字节数，与预算对账
checkpoint.py  五件套落盘 + 清单哈希 + 缺文件点名 + 逐位往返
inference.py   推理封装 + 可注入时钟 + 延迟读数
train.py       端到端回路（训练 / 恢复 / 对照），复用 day095 的损失与梯度
verify.py      七条性质与三类判据（第 ⑥ 条真的跑两次训练）
study.py       六张表 + 一条阶段线
```

## 四、四条纪律（与前面各天逐字相同）

1. **一个量只写一遍**：损失与梯度在 day095、范数守卫在 day094、优化器在 day092、
   打乱用的 LCG 在 day073——本课只写"数据 / 设备 / 检查点 / 推理"这四件新的东西。
2. **读数必须现场算出**：``study`` 里不存数字；每张表的每行都来自函数调用。
3. **跨天对账必须调用别人的包**：训练回路**真的**调用 ``regularization.loss_and_grad``
   与 ``sequence_models.train.check_gradient_norm``。
4. **缺席要可断言**：``RETURNED_FAMILY`` / ``ABSENT_FAMILY`` 都是常量。

## 五、与既有包的接缝（M8-D6 → M8-D7 的过渡）

- **上游**：``regularization``（day095：网络、损失、梯度、压平契约）、
  ``sequence_models``（day094：梯度范数守卫、±1 符号数据集）、
  ``optimizers``（day092：更新规则与状态键表）、``math_foundations``（day073：LCG）；
- **不改动任何既有模块**：day095 的 ``regularization`` 一行未改——本课是它的**使用者**；
- **脚下**：``config`` **没有**新增配置项——设备、批大小、检查点目录全部进
  ``train.PipelineConfig``（它是"这一次训练"的判据，而不是服务级默认值）；
- **下游**：day097（R3 复盘）会把 M7～M8 串成一条线，本课的读数就是它的素材。
"""

from __future__ import annotations

from smart_research_agent.torch_pipeline import checkpoint as _checkpoint
from smart_research_agent.torch_pipeline import dataloader as _dataloader
from smart_research_agent.torch_pipeline import datasets as _datasets
from smart_research_agent.torch_pipeline import device as _device
from smart_research_agent.torch_pipeline import errors as _errors
from smart_research_agent.torch_pipeline import inference as _inference
from smart_research_agent.torch_pipeline import sampler as _sampler
from smart_research_agent.torch_pipeline import study as _study
from smart_research_agent.torch_pipeline import train as _train
from smart_research_agent.torch_pipeline import types as _types
from smart_research_agent.torch_pipeline import verify as _verify
from smart_research_agent.torch_pipeline.errors import TorchPipelineError

#: 本包的十一个功能模块（``__all__`` 由它们的公开名单合并而来——"一个量只写一遍"）。
_MODULES = (
    _errors,
    _types,
    _datasets,
    _sampler,
    _dataloader,
    _device,
    _checkpoint,
    _inference,
    _train,
    _verify,
    _study,
)

#: 把十一个模块 ``__all__`` 里的名字逐个搬进包命名空间。
#:
#: 故意不手写两份名单（一份 import、一份 ``__all__``）：手写一定会分家，
#: 而"某一个名字在 ``__all__`` 里、却没有人真的导入它"这种失败
#: 在报告里长得和"它不存在"一模一样。下面那条导入期不变式代替人来核对这件事。
for _module in _MODULES:
    for _name in _module.__all__:  # pragma: no cover - 纯搬运
        globals()[_name] = getattr(_module, _name)

#: 本包的公开名单：**十一个模块各自 ``__all__`` 的并集**，字母序。
__all__ = sorted({name for module in _MODULES for name in module.__all__})

_MISSING = [name for name in __all__ if name not in globals()]
if _MISSING:  # pragma: no cover - 只在有人改名单时触发
    raise TorchPipelineError(
        f"__all__ 里的 {_MISSING} 没有被真正导入："
        "``from torch_pipeline import *`` 会静默地少几个名字，而调用方只会看到 NameError。"
    )

_SUBMODULES = {
    "errors",
    "types",
    "datasets",
    "sampler",
    "dataloader",
    "device",
    "checkpoint",
    "inference",
    "train",
    "verify",
    "study",
}
if _SUBMODULES & set(__all__):  # pragma: no cover - 只在有人改名单时触发
    raise TorchPipelineError(
        f"__all__ 不能包含子模块名：{sorted(_SUBMODULES & set(__all__))}——"
        "否则 `from torch_pipeline import train` 会拿到函数还是模块取决于导入顺序。"
    )
