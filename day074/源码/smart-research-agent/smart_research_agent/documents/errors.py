"""documents 子包的异常类型（M6-D1）.

与前面各天的子包同一套路：一个子包一个错误基类，API 层用一行
``except DocumentError`` 把"这一层的业务错误"与"程序缺陷"分开。

本包的错误有一处与别处不同：**"解析不了"经常不是错误，而是一个结论**。
一个加密的 PDF、一个缺 ``word/document.xml`` 的 docx，都不是我们的 bug——
它们是"这份文件不在支持范围内"。因此本包把这类情形分成两类：

```text
DocumentError          我们这边用错了（未知媒体类型、注册表里没有可用的加载器）
UnsupportedDocument    这份文件本身不在支持范围内（加密、缺部件、格式损坏）
```

分开的理由与 day060 把 ``blocked`` 与 ``failed`` 分开完全相同：
**"做不到"要被报成一个可读的结论，否则它会被当成"出错了"重试一遍。**
"""

from __future__ import annotations


class DocumentError(Exception):
    """文档解析与入库过程中的业务错误."""


class UnsupportedDocument(DocumentError):
    """这份文件不在支持范围内（加密 / 缺少必要部件 / 结构损坏）.

    它继承 ``DocumentError``，因此调用方仍然可以只捕一次；但需要区分时
    可以单独捕它——批量入库时最需要的恰恰是这个区分：
    "这一批里有 3 份解析不了"和"这一批里有 3 份触发了我们的 bug"
    对应两种完全不同的后续动作。
    """


__all__ = ["DocumentError", "UnsupportedDocument"]
