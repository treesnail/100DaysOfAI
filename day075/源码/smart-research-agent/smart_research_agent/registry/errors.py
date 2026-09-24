"""模型版本管理包的错误类型（M5-D9）.

与 ``domain_data/errors.py`` 同一条纪律：**包内只抛自己的错误类型**。
调用方（API 路由、CI 脚本）因此可以一次性 ``except RegistryError``，
既不会漏掉某条分支，也不会顺手把 ``ValueError`` 这类"到处都是"的异常
一起吞掉。

刻意继承 ``RuntimeError`` 而不是 ``Exception``：这类失败全部是
**运行期状态问题**（版本不存在、父版本悬空、不完整的三元组），不是
调用方写错了函数签名。day050 的 ``CheckpointError``、day057 的
``DomainDataError`` 都是同一个选择。
"""

from __future__ import annotations


class RegistryError(RuntimeError):
    """模型版本管理相关的一切失败（注册冲突、版本缺失、三元组非法、回滚不可达）."""
