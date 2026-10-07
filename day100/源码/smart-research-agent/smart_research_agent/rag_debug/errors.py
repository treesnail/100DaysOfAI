"""``rag_debug`` 的失败族：**这一层会怎么失败，以及每一种该谁去修**（day071）.

分族的依据与 ``retrieval.errors`` 逐字相同——**按"谁的错、该谁去修"分**，
而不是按"哪一行抛的"分。因为报告里最有用的那句话不是"报错了"，
而是"这个错该改什么"：

```text
CaseDataError      评测集写错了（缺字段、金标准为空、等级越界）  → 改数据
DiagnosisError     归因参数或形状用错了（调用方给了不认识的标签） → 改调用
BaselineError      基线与当前**不可比**（指标表不一致、样本不足） → 新开基线或补样本
RagDebugError      本族基类：其余"这一层自己的拒绝"
```

四族都继承 ``ValueError``（与 ``RetrievalError`` 同一句话）：
参数写错是**调用方**的问题，因此端点的"参数一律 400"不必为这一族加分支
（见 ``api.routes`` 里 ``except RagDebugError`` 的那一处）。

**为什么把 BaselineError 单独分一族**：它是这一课最容易被误判的一类失败。
"这次比基线差"不是错误，"这次**没法**和基线比"才是——后者有三种成因
（指标表不同 / 样本量不足 / 条件（索引版本、提示词版本）变了），
三者的处置动作都是"先别比"，但它们各自的下一步不同（重跑 / 补样本 / 新开基线）。
把它们混进一个 ``RagDebugError`` 之后，报告里只会剩下一句"评估失败"。
"""

from __future__ import annotations


class RagDebugError(ValueError):
    """``rag_debug`` 这一族错误的基类（判据与 ``RetrievalError`` 同源）."""


class CaseDataError(RagDebugError):
    """评测集里的某条用例写错了（缺字段、金标准为空、等级越界）.

    出路永远是**改数据**，而不是改代码：一个缺金标准的用例不是"检索没召回"，
    它连分母都没有——放着不管会让 ``recall@k`` 变成 0/0，
    而那个 0.0 看起来像是"一条都没召回到"。
    """


class DiagnosisError(RagDebugError):
    """归因用错了（标签不在封闭清单里、阈值越界、形状不是本层的）.

    出路是**改调用**：归因标签是一张封闭清单（``BAD_CASE_TAGS``），
    自由文本会让"这一批坏例里各类各有多少"这个问题无法统计。
    """


class BaselineError(RagDebugError):
    """基线与当前**不可比**（指标表不一致 / 样本不足 / 条件变了）.

    注意它与"回归"的区别：回归是一个**结论**（进了 `regressions` 列表），
    而本异常表示"这次连结论都给不出来"。把后者伪装成前者，
    会让一份条件变了的评估看起来像一次质量劣化。
    """


__all__ = [
    "BaselineError",
    "CaseDataError",
    "DiagnosisError",
    "RagDebugError",
]
