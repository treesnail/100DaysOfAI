"""serving 子包的异常类型（M5-D11）.

与 day057 的 ``DomainDataError``、day058 的 ``RegistryError``、day059 的
``MLOpsError`` 同一套路：**一个子包一个错误基类**。这样 API 层可以用一行
``except ServingError`` 把"这一层的业务错误"与"程序缺陷"分开——
前者回 400，后者让它冒出去变成 500（因为那是我们自己的 bug）。

刻意不做异常层次（`ServingSpecError` / `ServingBindingError` / ...）：
本包的错误**全部**是"配置或证据不对"，调用方对它们的处理动作是同一个
（打印出可读的说明、拒绝这次部署或切换），多一层继承只会多一层
"该 except 哪一个"的犹豫。
"""

from __future__ import annotations


class ServingError(Exception):
    """部署形态、部署绑定、流量切换与上线验证过程中的业务错误."""


__all__ = ["ServingError"]
