"""``sequence_models``：循环神经网络 —— 同一组权重，用在每一个时刻（M8-D5 / day094）.

day093 的卷积说：**同一组核被用到空间的每一个位置**（权重共享省掉了参数）；
今天把同一句话搬到时间轴上：

```text
时间上的权重共享  →  参数量与序列长度 T **完全无关**        （一个循环体被调用 T 次）
隐状态            →  "到目前为止的全部历史"的有损摘要        （宽度就是记忆预算）
BPTT              →  与 day090 同一条纪律：梯度一律 **累加**   （写成 = 只留下最后一步）
```

```text
① rnn  = 一个状态：h_t = act(W_h·h_{t−1} + W_x·x_t + b)
② lstm = 一个状态 + 一条记忆：c_t = f⊙c_{t−1} + i⊙g，h_t = o⊙tanh(c_t)
③ 步数对齐：T 个输入 ⇒ T 个状态；h₀ 是**初值**，不是输出
```

## 一、今天最值钱的一句话

> **循环的全部秘密是：同一组权重在时间轴的每一个时刻被重复使用。**

它对参数量的效果与卷积一样直接：D=2、H=3 的单元，无论序列是 1 步还是 8 步，
参数量都是 12（RNN）/ 48（LSTM）——**一个也不多**（``study.parameter_rows``）。

## 二、七条性质

```text
① 一步 = 手写递推（逐位）          ② 三门钉死时记忆原样冻结（逐位）
③ T 步与 1 步同一组权重（逐位）     ④ T 个输入 ⇒ T 个状态（整数）
⑤ ∂c_T/∂c_0 = ∏f_t（<= 1e-12）     ⑥ 解析 BPTT = 数值差分（<= 1e-9）
⑦ LSTM 的长程信号**必须**明显强于 tanh-RNN（**下界** 0.5）
```

判据三类：**相等**、**不超过上界**、**不超过下界**。第 ⑦ 条是本课唯一的下界判据——
没有它，把 LSTM 写成"记忆通道不独立"的假 LSTM，前六条会**全绿**。

## 三、九个模块

```text
errors.py    六个失败族（含 TimeStepError：状态数差一个；GradientError：**今天回来了**）
types.py     两种单元 / 四个门 / 七条性质 / 十条笔记 / 五条边界 / 五张公式 / torch 对照表
ops.py       算子层：matvec / matvec_transpose / outer / gate_blocks / parameter_count
layers.py    层：RNNSpec + RNNCell / LSTMSpec + LSTMCell + 确定性初始化 + 一步与整段
gradients.py 反向：BPTT（累加 dW）+ 两条 carry 读数（量梯度衰减）
network.py   序列分类器：整段 → 取 h_T → dense → logits（参数压平沿用 day090 契约）
train.py     合成"±1 序列求和符号"数据上真训练一次（优化器复用 day092）
verify.py    七条性质与三类判据（第 ⑥ 条跨天调用 day074 的数值差分）
study.py     七张表（单元 / 门 / 状态 / 参数量 / 反向 / 训练 / 性质）
```

## 四、四条纪律（与前面各天逐字相同）

1. **一个量只写一遍**：激活的前向只在 day089；导数只在 day090；数值差分只在 day074；
   更新规则只在 day092；本课只写"循环与它的反向"这一件事。
2. **读数必须现场算出**：``study`` 里不存数字。
3. **跨天对账必须调用别人的包**：数值差分（day074）、激活导数（day090）、
   激活前向（day089）、交叉熵（day089）、优化器（day092）——自证是不成立的。
4. **缺席要可断言**：``RETURNED_FAMILY`` / ``ABSENT_FAMILY`` 都是常量。

## 五、与既有包的接缝（M8-D4 → M8-D5 的过渡）

- **上游**：``neural_basics``（day089：LCG 初始化 / 激活 / 交叉熵）、
  ``backprop``（day090：激活的反向导数）、``optimizers``（day092：训练期更新规则）、
  ``math_foundations``（day073/074：数值差分）；
- **传承**：day093 讲"空间上的权重共享"，本课讲"时间上的权重共享"——同一个思想的两个实例；
  区别是卷积的每一步**彼此独立**（可以并行），循环的每一步**依赖上一步**（必须串行）；
- **脚下**：``config`` **没有**新增配置项——输入维、隐藏宽、序列长度都是函数参数；
- **下游**：day095（训练技巧与正则化）会给这条链装上批量归一化、Dropout 与早停；
  day075 的注意力机制与它形成正面比较：注意力把"串行的 T 步"换成"一次算完的两两打分"。
"""

from __future__ import annotations

from smart_research_agent.sequence_models import errors as _errors
from smart_research_agent.sequence_models import gradients as _gradients
from smart_research_agent.sequence_models import layers as _layers
from smart_research_agent.sequence_models import network as _network
from smart_research_agent.sequence_models import ops as _ops
from smart_research_agent.sequence_models import study as _study
from smart_research_agent.sequence_models import train as _train
from smart_research_agent.sequence_models import types as _types
from smart_research_agent.sequence_models import verify as _verify
from smart_research_agent.sequence_models.errors import RecurrentError

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
    raise RecurrentError(
        f"__all__ 里的 {_MISSING} 没有被真正导入："
        "``from sequence_models import *`` 会静默地少几个名字，而调用方只会看到 NameError。"
    )

_SUBMODULES = {
    "errors",
    "types",
    "ops",
    "layers",
    "gradients",
    "network",
    "train",
    "verify",
    "study",
}
if _SUBMODULES & set(__all__):  # pragma: no cover - 只在有人改名单时触发
    raise RecurrentError(
        f"__all__ 不能包含子模块名：{sorted(_SUBMODULES & set(__all__))}——"
        "否则 `from sequence_models import layers` 会拿到函数还是模块取决于导入顺序。"
    )
