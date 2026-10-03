"""分块子包的异常类型（M6-D2）.

与前面各天的子包同一套路：一个子包一个错误基类，API 层用一行
``except ChunkingError`` 把"这一层的业务错误"与"程序缺陷"分开。

本包的错误有一处与 day061 正好相反：**大部分"分块失败"是参数矛盾，不是数据坏了**。
day061 的错误多数来自文件本身（加密 PDF、缺 ``word/document.xml``、二进制），
而今天进来的东西已经是归一化过的 ``Document``——真正会出错的地方在**策略参数**：

```text
ChunkingError           参数矛盾（overlap 不小于 max_tokens、预算装不下一个字符、空文档）
UnsupportedStrategy     策略名不在注册表里
```

把"未知策略"单独一类，理由与 day045 列出候选模型名、day061 列出可用加载器
完全相同：**"你要的是不是其中之一"比 ``KeyError: 'semantic_v2'`` 有用得多。**
"""

from __future__ import annotations


class ChunkingError(ValueError):
    """分块过程中的业务错误.

    继承 ``ValueError`` 而不是 ``Exception``（day061 的 ``DocumentError`` 继承的是
    ``Exception``）：本包的错误几乎全部来自**参数矛盾**，而 ``ValueError`` 正是
    Python 里表达"你给的值不成立"的标准类型。带来的实际好处是调用方
    已经写好的 ``except (DocumentError, ValueError)`` 一行就能兜住它——
    **错误类型的选择也是接口设计的一部分，少一类分支就是少一处漏网的入口。**
    """


class UnsupportedStrategy(ChunkingError):
    """请求的分块策略不在注册表里.

    它继承 ``ChunkingError``，因此调用方仍可只捕一次；需要区分时单独捕它。
    批量分块时最需要的恰恰是这个区分：**"你要的这个策略我们没有"**
    与 **"这个策略跑挂了"** 对应两种完全不同的后续动作。

    名字里的 ``Unsupported`` 而不是 ``...Error``：与 day061 的
    ``UnsupportedDocument`` 保持同一命名习惯（"不在支持范围"是**结论**，
    不是错误）。ruff 的 N818 会对这一族名字报警，这里刻意保留——
    **命名的一致性比一条通用规则更重要**，而且这条规则在别的包里也一样被保留着。
    """


__all__ = ["ChunkingError", "UnsupportedStrategy"]
