"""DPO 实践层的统一异常（M5-D7）.

与 ``SFTLossError`` / ``PEFTConfigError`` / ``FinetuneEvalError`` 同一条纪律：
**继承 ``ValueError``**，于是路由层只需要 ``except ValueError`` 就能把
"字段非法""取值越界""数据为空"统一映射成 400，而不必逐个捕获。

为什么 day055 新开一个异常类而不是继续复用 ``FinetuneEvalError``：
``alignment`` 复用它是合理的（两者都在"评估/对齐"这条线上），而本课新增的
三个失败面——**三种损失函数的合法取值**、**TRL 训练配置的算术**、
**偏好数据集的构造**——的调用方是训练脚本与路由，它们不需要知道
"评估指标"的存在。异常类归属清晰，才能让 ``except`` 语句的粒度有意义。
"""

from __future__ import annotations


class DPOError(ValueError):
    """DPO 实践层的非法输入（未知损失函数、β 越界、步数算术不成立等）."""
