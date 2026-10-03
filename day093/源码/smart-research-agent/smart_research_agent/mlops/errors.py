"""MLOps 微调流水线包的错误类型（M5-D10）.

与 ``domain_data/errors.py``、``registry/errors.py`` 同一条纪律：
**包内只抛自己的错误类型**。调用方（CI 脚本、API 路由、流水线编排）
因此可以一次性 ``except MLOpsError``，既不会漏掉某条分支，
也不会顺手把 ``ValueError`` 这类"到处都是"的异常一起吞掉。

继承 ``RuntimeError`` 而不是 ``Exception``：这类失败全部是
**运行期状态问题**（run 不存在、指标重复记录、阶段依赖缺失、
门禁项无法判定），不是调用方写错了函数签名。
"""

from __future__ import annotations


class MLOpsError(RuntimeError):
    """MLOps 流水线相关的一切失败（追踪冲突、阶段依赖、门禁判定、CI 渲染）."""
