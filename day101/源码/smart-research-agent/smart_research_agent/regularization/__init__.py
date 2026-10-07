"""``regularization``：训练技巧与正则化 —— 给 day094 的链装上四个旋钮（M8-D6 / day095）.

day094 把"序列上的权重共享"讲完，并在结语里留下两个旋钮（学习率、展开长度）。
今天把它变成一整套**训练旋钮**，并补上这一课唯一一件新的数学：

```text
归一化      BatchNorm：沿**批**取平均（day079 的 LayerNorm 沿**特征**）——本包新建
随机丢      Dropout：训练置零 + 放大，推理恒等               ——转发 day081
学习率衰减  余弦退火 / 阶梯 / 热身                           ——转发 day074（经优化器执行）
早停        连续 patience 步没有明显变好就建议停下            ——转发 day081
可视化      把上面这些读数画成 sparkline / 折线 / 条形图      ——本包新建
```

## 一、今天最值钱的一句话

> **BatchNorm 的统计量来自一批样本——因此"这条链必须分批"，"
> 训练相与推理相必须配对"，两件事都不是风格问题，而是定义问题。**

本课真的撞到了这两堵墙：

```text
第一堵   把 BN 插在"单条序列的 h_T"上 ⇒ 批大小 1 ⇒ 批内方差恒为 0
         ⇒ 输出被压成常数 β ⇒ 三条不同的配置给出**同一个 loss**（0.542374）
第二堵   训练相用本批统计、推理相用 running 统计，两者在"表示会漂移"时差得很远
         ⇒ 本课的读数：rnn 上 loss_gap = +3.630002（而 lstm 上是 −0.003022）
```

## 二、七条性质

```text
① BN 前向 = 手写逐特征标准化（逐位）        ② BN(训练相) = transpose(LayerNorm(批ᵀ))（<= 1e-12）
③ 训练相与推理相**必须不同**（下界 0.5）      ④ 训练相反向 = 数值差分（<= 1e-7）
⑤ 推理相反向 = 数值差分（<= 1e-7）            ⑥ dropout 推理相 = rate=0 训练相（逐位）
⑦ sparkline 的长度与极值位置都对（整数）
```

判据三类：**相等**、**不超过上界**、**不超过下界**。第 ③ 条是这一课唯一的下界判据——
没有它，把两相写成同一件事时，其余六条里有五条会照常通过。

## 三、十个模块

```text
errors.py        四个失败族（含 PhaseError：两相被配错了对）
types.py         两条轴 / 两个相 / 四个技巧 / 七条性质 / 十条笔记 / 五条边界 / 四张公式
normalization.py 本包唯一的新数学：BatchNorm（统计量 / 两相 / running / 三项与一项反向）
techniques.py    四个旋钮的转发层（dropout / 调度 / 早停）+ RegularizationConfig
visualize.py     训练日志的四张脸（sparkline / 折线 / 条形图 / 汇总裁剪）
network.py       把旋钮装到 day094 的链上（**批版本**，BPTT 仍然调用 day094）
train.py         分批训练 + 调度 + 梯度范数守卫 + 早停 + 消融对比
verify.py        七条性质与三类判据（第 ② 条跨天调用 day079 的 layer_norm）
study.py         七张表（轴 / 相 / 技巧 / 参数 / 消融 / 日志 / 性质）
```

## 四、四条纪律（与前面各天逐字相同）

1. **一个量只写一遍**：BPTT 在 day094、dropout 与早停在 day081、调度在 day074、
   数值差分在 day074——本课只写 BatchNorm 这一件新的数学。
2. **读数必须现场算出**：``study`` 里不存数字；每张表的每行都来自函数调用。
3. **跨天对账必须调用别人的包**：第 ② 条**真的**调用 day079 的 ``layer_norm``。
4. **缺席要可断言**：``RETURNED_FAMILY`` / ``ABSENT_FAMILY`` 都是常量。

## 五、与既有包的接缝（M8-D5 → M8-D6 的过渡）

- **上游**：``sequence_models``（day094：循环单元、BPTT、压平契约）、
  ``training_optim``（day081：dropout / EarlyStopping）、
  ``math_foundations``（day074：调度）、``encoder_decoder``（day079：LayerNorm）、
  ``optimizers``（day092：更新规则）、``neural_basics`` / ``backprop``（day089/090：损失与梯度）；
- **不改动任何既有模块**：day094 的 ``sequence_models`` 一行未改——
  本课是它的**使用者**；
- **脚下**：``config`` **没有**新增配置项——四个旋钮全部进 ``RegularizationConfig``，
  它是"这一次训练"的判据，而不是服务级的默认值；
- **下游**：day096（PyTorch 高级与综合实践）会把这些旋钮搬进 DataLoader 与 GPU 流程；
  day097（R3 复盘）会把 M7～M8 串成一条线。
"""

from __future__ import annotations

from smart_research_agent.regularization import errors as _errors
from smart_research_agent.regularization import network as _network
from smart_research_agent.regularization import normalization as _normalization
from smart_research_agent.regularization import study as _study
from smart_research_agent.regularization import techniques as _techniques
from smart_research_agent.regularization import train as _train
from smart_research_agent.regularization import types as _types
from smart_research_agent.regularization import verify as _verify
from smart_research_agent.regularization import visualize as _visualize
from smart_research_agent.regularization.errors import RegularizationError

#: 本包的九个功能模块（``__all__`` 由它们的公开名单合并而来——"一个量只写一遍"）。
_MODULES = (
    _errors,
    _types,
    _normalization,
    _techniques,
    _visualize,
    _network,
    _train,
    _verify,
    _study,
)

#: 把九个模块 ``__all__`` 里的名字逐个搬进包命名空间。
for _module in _MODULES:
    for _name in _module.__all__:  # pragma: no cover - 纯搬运
        globals()[_name] = getattr(_module, _name)

#: 本包的公开名单：**九个模块各自 ``__all__`` 的并集**，字母序。
__all__ = sorted({name for module in _MODULES for name in module.__all__})

_MISSING = [name for name in __all__ if name not in globals()]
if _MISSING:  # pragma: no cover - 只在有人改名单时触发
    raise RegularizationError(
        f"__all__ 里的 {_MISSING} 没有被真正导入："
        "``from regularization import *`` 会静默地少几个名字，而调用方只会看到 NameError。"
    )

_SUBMODULES = {
    "errors",
    "types",
    "normalization",
    "techniques",
    "visualize",
    "network",
    "train",
    "verify",
    "study",
}
if _SUBMODULES & set(__all__):  # pragma: no cover - 只在有人改名单时触发
    raise RegularizationError(
        f"__all__ 不能包含子模块名：{sorted(_SUBMODULES & set(__all__))}——"
        "否则 `from regularization import network` 会拿到函数还是模块取决于导入顺序。"
    )
