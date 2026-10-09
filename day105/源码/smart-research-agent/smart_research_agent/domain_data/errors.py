"""领域数据管道的统一异常（M5-D8）.

与 ``finetune.schema.DatasetFormatError``、``dpo.errors.DPOError`` 同一约定：
继承 ``ValueError``，并且**只表达一个意思**——"这批数据/这组参数不能
进流水线"。调用方关心的是"该不该让这批数据继续往下走"，而不是
"第几个字符解析失败"；底层异常（``json.JSONDecodeError``、``KeyError``）
一律留在产生它们的地方，不向上冒泡成领域异常。
"""

from __future__ import annotations


class DomainDataError(ValueError):
    """领域数据准备与增强过程中的参数或数据不合法.

    本包里的校验全部发生在**构造期或进入流水线之前**（沿用 day055 的
    "校验必须在构造期"纪律）：配比阈值、签名长度、权重之和、分组键
    覆盖度这类问题一旦带进流水线，表现出来的是"数据莫名其妙少了"，而
    不是一条报错——静默的错误比响亮的错误危险得多。
    """
