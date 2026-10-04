"""``conv_net``：卷积神经网络 —— 同一组权重在输入的不同位置被重复使用（M8-D4 / day093）.

day089 的 ``Dense`` 是"**每个输入位置用一个独立的权重**"；day090 给了"梯度怎么回传"；
day092 给了"拿到梯度后一步走多远"。今天第一次引入**空间上的权重共享**：

```text
权重共享  →  参数量从 O(H·W) 降到 O(k²)          （一张核在全图复用）
局部连接  →  每个输出只看一个 k×k 窗口              （感受野）
多通道    →  在通道维再做一次加权求和               （C_in → C_out）
```

## 一、今天最值钱的一句话

> **卷积的全部秘密是一句话：同一组权重在输入的不同位置被重复使用。**

理解它的最快方式是对比参数个数：6×6 的图直接接一个 18 维全连接头要
``36×18 = 648`` 个权重；一层 3×3、2 输出通道的卷积只要 ``2×9 + 2 = 20`` 个。
**少了 32 倍**，而它还能把"竖线 vs 横线"这类局部纹理区分开——因为核会被
用到全图的每一个位置。

## 二、七条性质

```text
① conv2d = 手写滑窗点积（逐位）            ② 卷积是线性算子（<= 1e-12）
③ same 填充让输出同尺寸（整数）            ④ 尺寸公式与实际形状一致（整数）
⑤ 解析梯度 = 数值差分（<= 1e-9）          ⑥ 池化输出 = 窗口极值 / 均值（<= 1e-12）
⑦ 感受野递推 = 逐层区间传播（整数）
```

判据两类：**相等**（逐位 / 整数）与**不超过上界**（误差）。第 ⑤ 条是这一课的命门：
``d_kernel`` 与 ``d_image`` 互为"转置"，索引写反了形状仍然正确——
只有 day074 那把数值差分的尺子能把它们区分开。

## 三、九个模块

```text
errors.py    五个失败族（含 WindowError：窗口盖不满输入时拒绝兜底）
types.py     两种填充 / 两种池化 / 七条性质 / 十条笔记 / 五条边界 / 三张公式 / torch 对照表
ops.py       算子层：output_size / resolve_padding / pad2d / conv2d / pool2d / receptive_field
layers.py    层：ConvSpec（可复现初始化）+ ConvParams + 多通道前向 + 池化前向
gradients.py 反向：dK / dB / dX 与池化反向（激活导数复用 day090）
network.py   小 CNN 的前向 / 反向 / 参数压平（压平契约沿用 day090）
train.py     合成"竖线 vs 横线"数据集上真训练一次（优化器复用 day092）
verify.py    七条性质与两类判据（第 ⑤ 条跨天调用 day074 的数值差分）
study.py     七张表（层 / 尺寸 / 感受野 / 特征图 / 池化 / 训练 / 性质）
```

## 四、四条纪律（与前面各天逐字相同）

1. **一个量只写一遍**：激活的导数只在 day090；数值差分只在 day074；
   更新规则只在 day092；本课只写"卷积与池化"这一件事。
2. **读数必须现场算出**：``study`` 里不存数字。
3. **跨天对账必须调用别人的包**：数值差分（day074）、激活导数（day090）、
   交叉熵（day089）、优化器（day092）——自证是不成立的。
4. **缺席要可断言**：``RETURNED_FAMILY`` / ``ABSENT_FAMILY`` 都是常量。

## 五、与既有包的接缝（M8-D3 → M8-D4 的过渡）

- **上游**：``neural_basics``（day089：LCG 初始化 / 激活 / 交叉熵）、
  ``backprop``（day090：激活的反向导数 / 参数压平契约）、
  ``optimizers``（day092：训练期更新规则）、``math_foundations``（day073/074：数值差分）；
- **传承**：day092 把"一步走多远"讲透，本课第一次遇到"参数不是一串数，而是若干卷积核"——
  于是 day090 的压平契约被真正用上：**压平成向量给优化器，再还原回核**；
- **脚下**：``config`` **没有**新增配置项——通道数、核、步长、填充、窗口都是函数参数；
- **下游**：day094（RNN / LSTM）会问"序列上的权重共享与空间上的有什么不同"。
"""

from __future__ import annotations

from smart_research_agent.conv_net import errors as _errors
from smart_research_agent.conv_net import gradients as _gradients
from smart_research_agent.conv_net import layers as _layers
from smart_research_agent.conv_net import network as _network
from smart_research_agent.conv_net import ops as _ops
from smart_research_agent.conv_net import study as _study
from smart_research_agent.conv_net import train as _train
from smart_research_agent.conv_net import types as _types
from smart_research_agent.conv_net import verify as _verify
from smart_research_agent.conv_net.errors import ConvError

#: 本包的九个功能模块（``__all__`` 由它们的公开名单合并而来——"一个量只写一遍"）。
_MODULES = (_errors, _types, _ops, _layers, _gradients, _network, _train, _verify, _study)

#: 把九个模块 ``__all__`` 里的名字逐个搬进包命名空间。
for _module in _MODULES:
    for _name in _module.__all__:  # pragma: no cover - 纯搬运
        globals()[_name] = getattr(_module, _name)

#: 本包的公开名单：**九个模块各自 ``__all__`` 的并集**，字母序。
__all__ = sorted({name for module in _MODULES for name in module.__all__})

_MISSING = [name for name in __all__ if name not in globals()]
if _MISSING:  # pragma: no cover - 只在有人改名单时触发
    raise ConvError(
        f"__all__ 里的 {_MISSING} 没有被真正导入："
        "``from conv_net import *`` 会静默地少几个名字，而调用方只会看到 NameError。"
    )

_SUBMODULES = {"errors", "types", "ops", "layers", "gradients", "network", "train", "verify", "study"}
if _SUBMODULES & set(__all__):  # pragma: no cover - 只在有人改名单时触发
    raise ConvError(
        f"__all__ 不能包含子模块名：{sorted(_SUBMODULES & set(__all__))}——"
        "否则 `from conv_net import layers` 会拿到函数还是模块取决于导入顺序。"
    )
