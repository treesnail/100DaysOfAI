"""``failure_ledger`` 的失败族：把 100 天的失败族编成一份台账时，每一种失败该谁去修（day102）.

分族的依据与前面各天逐字相同——**按"谁的错、该谁去修"分**。
今天这一批都出现在同一个地方：**把 36 份 ``errors.py``（35 份历史 + 本课这一份）
里的 194 个族编成一份跨包的台账**。

```text
ScanError         扫描不成立     某份 errors.py 读不到 / 语法错 / 一个族都没有        → 改扫描
ResolveError      归属不成立     基类名既不是本模块的、也不是内置、也没有导入别名      → 改解析或改源码
LedgerBuildError  台账不成立     同一 (包, 名) 出现两次 / 两次构建给出不同台账        → 改台账实现
CoverageError     覆盖不成立     期望的包没有 errors.py / 某个族没被收进台账          → 改台账范围
NumericError      数值不可用     非有限读数 / 非法上下界                              → 改数据或改实现
ParameterError    参数越界       未知包名 / 未知性质 / 非正上限                       → 改调用
LedgerError       本族基类
```

## 一、``ScanError`` 与 ``ResolveError`` 为什么不是同一族

```text
ScanError     "材料读不进来"     文件读不到 / ast 语法错 / 一份 errors.py 里没有 ClassDef ⇒ 改扫描
ResolveError  "读进来对不上号"   基类名无法归属到 local / builtin / imported 三类中的任何一类 ⇒ 改解析
```

前者管**输入**，后者管**归属**。混成一族会让"少扫了一个包"的人去查别名解析表，
而"别名解析漏了一条"的人去查文件系统。

## 二、``LedgerBuildError`` 与 ``CoverageError`` 为什么要分开

```text
LedgerBuildError  "台账自己不自洽"   同名冲突 / 两次构建不同 ⇒ 改台账实现（去掉未固定的量）
CoverageError     "台账少了一页"     某个子包没有 errors.py / 某个族没进来 ⇒ 改台账范围
```

一个是**自洽性**，一个是**完整性**。day101 把这两件事分别写成
``IndexBuildError`` 与两条覆盖性质；本课沿用同一条切分。

## 三、``GradientError`` 今天继续缺席

```text
day090 / day094   抛 GradientError
day096 / day099   缺席：本日没有一次反向
day100            缺席：本日是交付与归档
day101            缺席：本日是编索引与检索
day102            缺席：本日是**一次纯静态扫描（ast）**，连被调用的那一次反向也没有
```

本课一个张量、一次前向都没有跑。因此这一族今天不属于本包。

## 四、跨包的继承关系

```text
ValueError
├── TransformerError（day075）……（前面那些包）
├── CapstoneError（day099）
├── GraduationError（day100）
├── CourseIndexError（day101）
└── LedgerError（day102，本模块）
    ├── ScanError / ResolveError / LedgerBuildError / CoverageError
    └── NumericError / ParameterError
```

本层**不继承 day101 的七族**：今天要报的是"编台账"这一层的失败，
而它与"编索引"那一层在处置方向上并不同源——``IndexBuildError`` 是
"同一份语料编出两份索引"，而 :class:`LedgerBuildError` 是"同一批文件编出两份台账"。
把它们混成一家，会让 ``except CourseIndexError`` 这种既有写法在台账层悄悄多兜住几族。
"""

from __future__ import annotations


class LedgerError(ValueError):
    """``failure_ledger`` 这一族错误的基类（与 ``CourseIndexError`` / ``GraduationError`` 同源）."""


class ScanError(LedgerError):
    """扫描不成立：文件读不到、ast 语法错、或一份 ``errors.py`` 里一个 ClassDef 都没有.

    它的处置动作最具体：**改扫描**。一份漏掉的 ``errors.py``
    与"这个子包没有失败族"在台账里读起来完全一样。
    """


class ResolveError(LedgerError):
    """归属不成立：某个基类名无法归属到 local / builtin / imported 三类中的任何一类.

    出路是**改解析**（补一条别名规则）或**改源码**（那个名字本来就不该出现）。
    """


class LedgerBuildError(LedgerError):
    """台账不成立：同一 ``(包, 名)`` 出现两次，或同一批文件两次构建给出不同的台账.

    它是本课唯一一族"内部中间物"的失败（见模块 docstring 第二节）。
    """


class CoverageError(LedgerError):
    """覆盖不成立：期望的包在台账里没有 ``errors.py``，或某个族没被收进台账.

    与 :class:`LedgerBuildError` 的分工是"自洽 vs 完整"（见模块 docstring 第二节）。
    """


class NumericError(LedgerError):
    """数值不可用：非有限读数、上界 / 下界非法（负数）.

    读数非有限时任何比较都会给出一个**静默为假**的结论，因此本包在入口就拒绝。
    """


class ParameterError(LedgerError):
    """参数越界：未知包名 / 未知性质 / 未知判据 / 非正上限.

    出路是**改调用**。本包不接受"名字写错就取个默认值"这种兜底——
    一个被兜住的 ``property_id`` 会在报告里长得和"我确实检查了它"完全一样。
    """


#: 六个族各自"该谁去修"（**这张表要被测试逐键检查**）.
FAMILY_OUTCOMES: dict[str, str] = {
    "ScanError": "改扫描：某份 errors.py 读不到或语法错时，先把它补进扫描范围",
    "ResolveError": "改解析或改源码：基类名无法归属时，补一条别名规则或删掉那个名字",
    "LedgerBuildError": "改台账实现：两次构建不一致时，去掉那个没有固定的量（集合序 / 字典序）",
    "CoverageError": "改台账范围：某个子包没有 errors.py 时，要么补一个族、要么显式排除它",
    "NumericError": "改数据或改实现：非有限读数与非法上下界都属于'数值不可用'",
    "ParameterError": "改调用：包名 / 性质名 / 判据 / 上限都是调用点的一次决定",
}

#: 本模块真正导出的六个异常类名（与 ``FAMILY_OUTCOMES`` 逐键对齐的闭合检查）.
_FAMILY_CLASSES: dict[str, type[BaseException]] = {
    "ScanError": ScanError,
    "ResolveError": ResolveError,
    "LedgerBuildError": LedgerBuildError,
    "CoverageError": CoverageError,
    "NumericError": NumericError,
    "ParameterError": ParameterError,
}

if set(FAMILY_OUTCOMES) != set(_FAMILY_CLASSES):  # pragma: no cover - 只在有人改表时触发
    raise LedgerError(
        "失败族的两张表不一致：FAMILY_OUTCOMES 的键必须与 _FAMILY_CLASSES 逐一对应——"
        "少一个键的那一族在报告里只剩类名、没有'该怎么办'。"
    )

#: 今天**回来**的族：``ShapeError``（各包最常见的族名，在 day101 缺席后重新需要一个控制流事件）.
RETURNED_FAMILY: str | None = "ShapeError"

#: 回来的理由：**与前面各课"请回旧名字"的理由同源**.
RETURNED_FAMILY_REASON = (
    "``ShapeError`` 是本台账里出现次数最多的族名（跨包重名之最）："
    "它由 day075 的 ``transformer_core`` 第一次命名，此后被十来个包各写一遍。"
    "day101（编索引）里没有它。day102 要把这些同名族摆在一张表上，"
    "于是'一个族的基类数 / 名字与它该有的对不上'重新需要一次控制流事件——"
    "这个名字**回来**了，但基类从 ``ValueError`` 换成 ``LedgerError(ValueError)``："
    "今天它要能被同一条 ``except ValueError`` 兜住，与其余五族同源。"
)

#: 今天缺席的那一族：``GradientError``（本课是一次纯静态扫描，一行张量运算都没有）.
ABSENT_FAMILY: str | None = "GradientError"

#: 缺席理由：**与 day100 / day101 的理由同源（都是"没有一次反向"），但场景不同**.
ABSENT_FAMILY_REASON = (
    "本课用 ``ast`` 把 36 份 ``errors.py`` 读成一张表：一个**纯静态**的扫描层，"
    "不但没有反向，连一次前向都没有跑（它连 ``import`` 都不做，只读源码）。"
    "因此这一族今天不属于本包——它属于被扫描的代码。"
)

__all__ = [
    "ABSENT_FAMILY",
    "ABSENT_FAMILY_REASON",
    "FAMILY_OUTCOMES",
    "RETURNED_FAMILY",
    "RETURNED_FAMILY_REASON",
    "CoverageError",
    "LedgerBuildError",
    "LedgerError",
    "NumericError",
    "ParameterError",
    "ResolveError",
    "ScanError",
]
