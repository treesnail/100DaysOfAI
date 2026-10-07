"""``torch_pipeline`` 的失败族：一次"数据 → 设备 → 训练 → 检查点 → 推理"该谁去修（day096 / M8-D7）.

分族的依据与前八天逐字相同——**按"谁的错、该谁去修"分**。

```text
ShapeError       形状不符      索引 / 批里的列数 / 压平长度对不上          → 改调用
ParameterError   参数越界      batch_size / eval_ratio / workers 取值非法   → 改调用
NumericError     数值不可用    非有限读数、空数据集、比例不合法            → 改数据或改实现
DeviceError      设备不成立    未知的设备名 / 精度名 / 预算口径            → 改调用（换设备或换精度）
CheckpointError  检查点不完整  五件套缺一个 / 哈希不符 / JSON 损坏          → 改文件或改流程
TorchPipelineError   本族基类
```

## 一、``DeviceError`` 为什么要独立成族

```text
"这条链跑在哪个设备上、用什么精度算"是**调用点的一次决定**，
而不是"某个形状写错了"或"某个超参越界了"。
把 ``cuda`` 写成 ``CUDA `` / 把 ``float32`` 写成 ``float32x`` 这类失败
既不是形状问题、也不是数值问题——它是"这台机器上没有你要的东西"。
```

它与 :class:`ParameterError` 的处置方向不同：参数越界是"那个数太大 / 太小"，
而这一族是"**这个取值根本不存在**"（设备名 / 精度名是一张白名单）。
混进参数族会让"我写的是 bfloat16 啊"这种诊断被淹没在"数值范围"里。

## 二、``CheckpointError`` 为什么值得独立成族

```text
五件套缺一个文件 ⇒ 那个目录**看着能用**，实际 load 出来的东西是错的
哈希对不上     ⇒ 权重被谁改过 / 上次没写完整，而形状与数值都"正常"
```

它要拦住的第一种错误，正是 day050 ``sft/checkpoint.py`` 写下的那条：
**一个缺了文件的检查点看着能用，比直接报错危险得多**。
因此本族在缺文件时必须**点名是哪一个**（而不是"检查点坏了"）。

## 三、今天没有"回来"的族与新命名

```text
day050   第一次给"检查点读写失败"命名（``sft.checkpoint.CheckpointError``）——此后一直缺席
day090   抛 ``GradientError``：解析梯度与数值差分对不上（离线对账的结论）
day092   缺席：它消费梯度、不计算梯度
day093   缺席：算了梯度，但对账是离线校验
day094   抛 ``GradientError``：梯度真的会爆 ⇒ 做成运行期守卫
day095   缺席：梯度守卫由 day094 装好、本课只是调用它
day096   缺席：训练回路里唯一的梯度控制流事件仍是 day094 的 ``check_gradient_norm``，本课只是调用
```

## 四、跨包的继承关系

```text
ValueError
├── TransformerError（day075）
│   ├── ShapeError / NumericError / ParameterError
│   └── GradientError
└── TorchPipelineError（day096，本模块）
    ├── ShapeError / NumericError / ParameterError   （多继承：既是 day075 族、也是本族）
    ├── DeviceError
    └── CheckpointError
```

本层**继承 day075 的族而不是彼此**：``hf_source``、``inference_optim``、
``principle_map``、``neural_basics``、``backprop``、``optimizers``、``conv_net``、
``sequence_models``、``regularization`` 与本包是十个**兄弟**。
"""

from __future__ import annotations

from smart_research_agent.transformer_core.errors import (
    NumericError as CoreNumericError,
)
from smart_research_agent.transformer_core.errors import (
    ParameterError as CoreParameterError,
)
from smart_research_agent.transformer_core.errors import (
    ShapeError as CoreShapeError,
)


class TorchPipelineError(ValueError):
    """``torch_pipeline`` 这一族错误的基类（判据与 ``RegularizationError`` / ``ConvError`` 同源）."""


class ShapeError(CoreShapeError, TorchPipelineError):
    """形状不符：样本索引越界、批里的列数、压平后的长度对不上.

    出路是**改调用**。同时继承 ``transformer_core.errors.ShapeError``：
    前面各天写的 ``except ShapeError`` 不需要改动就能兜住本层的失败。
    """


class ParameterError(CoreParameterError, TorchPipelineError):
    """参数越界：``batch_size`` / ``eval_ratio`` / ``workers`` / ``drop_last`` 的取值非法.

    出路是**改调用**。本包不接受"越界就取个默认值"这种兜底——
    一个被静默替换的 ``eval_ratio`` 会让"训练集与评估集的重叠度"这件事变成假的。
    """


class NumericError(CoreNumericError, TorchPipelineError):
    """数值不可用：非有限读数、空数据集、切分后的某一侧为空、比例非法.

    读数非有限时，任何比较（相等或上界）都会给出一个**静默为假**的结论，
    因此本包在入口就拒绝。
    """


class DeviceError(TorchPipelineError):
    """设备不成立：未知的设备名 / 精度名，或"这条链放不进预算"的判据口径.

    它与 :class:`ParameterError` 处置的**对象**不同：

    ```text
    ParameterError  调用点的**某一个数**越界了       ⇒ 去改那个数
    DeviceError     "这台机器上没有你要的东西"        ⇒ 去看设备白名单与精度白名单
    ```

    混成一族会让"我明明写的是 bfloat16"这种诊断被误诊成"数值范围没调好"。

    注意：**预算不足不抛异常**——它由 :class:`~torch_pipeline.device.DevicePlan` 的
    ``fits`` 暴露。``DeviceError`` 只在"设备名 / 精度名不在白名单里"时抛。
    """


class CheckpointError(TorchPipelineError):
    """检查点不完整：五件套缺一个、哈希对不上、JSON 损坏.

    它值得独立成族，因为它的处置方向与其余各族都不同：

    ```text
    ShapeError / ParameterError / DeviceError   改**调用**
    CheckpointError                            改**文件或流程**（谁删了 / 谁写坏了）
    ```

    本族在缺文件时**点名是哪一个**：一个缺了 ``params.json`` 的检查点看着能用，
    而"到底缺了什么"是修复它的第一步。
    """


#: 五个族各自"该谁去修"（**这张表要被测试逐键检查**）.
FAMILY_OUTCOMES: dict[str, str] = {
    "ShapeError": "改调用：样本索引 / 批的列数 / 压平长度必须逐维对齐",
    "ParameterError": "改调用：batch_size / eval_ratio / workers 都是调用点的一次决定",
    "NumericError": "改数据或改实现：空数据集或某一侧被切空时，先查更早的那一次切分",
    "DeviceError": "改调用：设备名与精度名是白名单，写错了就换一个已支持的取值",
    "CheckpointError": "改文件或改流程：五件套缺一个时，先查是谁删了 / 谁没写完",
}

#: 本模块真正导出的五个异常类名（与 ``FAMILY_OUTCOMES`` 逐键对齐的闭合检查）.
_FAMILY_CLASSES: dict[str, type[BaseException]] = {
    "ShapeError": ShapeError,
    "ParameterError": ParameterError,
    "NumericError": NumericError,
    "DeviceError": DeviceError,
    "CheckpointError": CheckpointError,
}

if set(FAMILY_OUTCOMES) != set(_FAMILY_CLASSES):  # pragma: no cover - 只在有人改表时触发
    raise TorchPipelineError(
        "失败族的两张表不一致：FAMILY_OUTCOMES 的键必须与 _FAMILY_CLASSES 逐一对应——"
        "少一个键的那一族在报告里只剩类名、没有'该怎么办'。"
    )

#: 今天**回来**的族：``CheckpointError``（day050 命名、此后缺席，本课把它收进本族）.
RETURNED_FAMILY: str | None = "CheckpointError"

#: 回来的理由：**与 day094 请回 ``GradientError`` 的理由不同**.
RETURNED_FAMILY_REASON = (
    "day050 的 ``sft/checkpoint.py`` 第一次给'检查点读写失败'命名了 ``CheckpointError``，"
    "此后 M6 / M7 各课都把它当工具函数、不把它当一次控制流事件。"
    "day096 把'五件套检查点'变成训练回路里的一等公民（每个 epoch 落一次、"
    "缺文件当场点名、哈希对不上当场拒绝），于是这个名字**回来**了——"
    "但基类从 ``RuntimeError`` 换成 ``TorchPipelineError(ValueError)``："
    "今天它要能被同一条 ``except ValueError`` 兜住，与其余四族同源。"
)

#: 今天缺席的那一族：``GradientError``（它由 day090 命名、day094 回来，本课再次让它退场）.
ABSENT_FAMILY: str | None = "GradientError"

#: 缺席理由：**与 day092 / day093 / day095 都不同**.
ABSENT_FAMILY_REASON = (
    "本课的训练回路里唯一与梯度有关的控制流事件，是 day094 的 "
    "``sequence_models.train.check_gradient_norm``（每步进优化器之前看一眼范数）；"
    "本课**调用**它、不重新抛。梯度本身由 day095 的 ``regularization.network.loss_and_grad`` "
    "算好交来，本课连一次解析梯度都没有手写。"
    "因此这一族今天不属于本包——它属于被调用的那一行。"
)

__all__ = [
    "ABSENT_FAMILY",
    "ABSENT_FAMILY_REASON",
    "FAMILY_OUTCOMES",
    "RETURNED_FAMILY",
    "RETURNED_FAMILY_REASON",
    "CheckpointError",
    "DeviceError",
    "NumericError",
    "ParameterError",
    "ShapeError",
    "TorchPipelineError",
]
