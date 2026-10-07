"""``course_index`` 的失败族：把 100 天的材料编成一份索引时，每一种失败该谁去修（day101）.

分族的依据与前面各天逐字相同——**按"谁的错、该谁去修"分**。
今天这一批都出现在同一个地方：**把散落在 ``docs/`` 与 52 个子包里的文字编成一份可检索的索引**。

```text
CorpusError       语料不成立     文档名重复 / 某一份材料读不到 / 文本为空            → 改语料收集
IndexBuildError        索引不成立     同一个词在同一份文档里的计数丢失 / 索引重复构建不同   → 改索引实现
QueryError        查询不成立     空查询 / 查询里一个可索引的词都没有                  → 改查询或改分词
ScoreError        分数不可用     命中分数越界（不在 [0, 1]）                          → 改打分
NumericError      数值不可用     非有限读数 / 非法上下界                              → 改数据或改实现
ParameterError    参数越界       未知文档种类 / 未知性质 / 非正 top_k                 → 改调用
CourseIndexError  本族基类
```

## 一、``CorpusError`` 与 ``IndexBuildError`` 为什么不是同一族

```text
CorpusError   "材料本身没齐"     某一份 ``docs/*.md`` 没读到 / 某个子包没有 docstring ⇒ 改收集
IndexBuildError    "编出来的东西不对" 同一份语料建两次索引，结果不一样 ⇒ 改索引实现
```

前者管**输入**，后者管**工序**。混成一族会让"补一份漏掉的文档"的人去查索引算法，
而"索引不确定"的人去查文件系统。

## 二、``IndexBuildError`` 为什么必须独立于"结果不对"

索引是**中间物**：它的错会在检索时才显形，而那时现象是"某个查询少了一条命中"——
看起来像查询的问题。因此本包让索引自己可以被**逐位复算**（:func:`index.index_digest`），
把一个"下游看起来像查询故障"的问题，变成一个"上游就能量出来的数"。

## 三、``QueryError`` 为什么要独立成族

```text
空查询        ""              → 没有可检索的词，返回空是**正确地**返回空，但调用点该知道
无词查询      "的的了了"       → 分词后一个 token 都没有（全是停用/单字），
                                与"查了但没命中"是两件事
```

两者的出路都是**改调用或改分词**，而不是"再跑一遍"。

## 四、``GradientError`` 今天继续缺席

```text
day090 / day094   抛 GradientError
day096 / day099   缺席：本日没有一次反向
day100            缺席：本日是交付与归档
day101            缺席：本日是**编索引与检索**，同样一个新式子都没有写
```

本课唯一的"梯度事件"仍是被调用的 ``finetune`` 那一侧。因此这一族今天不属于本包。

## 五、跨包的继承关系

```text
ValueError
├── TransformerError（day075）……（前面那些包）
├── CapstoneError（day099）
├── GraduationError（day100）
└── CourseIndexError（day101，本模块）
    ├── CorpusError / IndexBuildError / QueryError / ScoreError
    └── NumericError / ParameterError
```

本层**不继承 day100 的七族**：今天要报的是"编索引"这一层的失败，
而它们与"交付"那七族在处置方向上并不同源——``DemoError`` 是"剧本重放不出来"，
而 :class:`IndexBuildError` 是"同一份语料编出两份索引"。把它们混成一家，
会让 ``except GraduationError`` 这种既有写法在索引层悄悄多兜住几族。
"""

from __future__ import annotations


class CourseIndexError(ValueError):
    """``course_index`` 这一族错误的基类（与 ``CapstoneError`` / ``GraduationError`` 同源）."""


class CorpusError(CourseIndexError):
    """语料不成立：文档名重复、某一份材料读不到、或文本为空（见模块 docstring 第一节）.

    它的处置动作最具体：**改语料收集**。一份漏掉的 ``docs/*.md``
    与"这门课没有那份材料"在索引里读起来完全一样。
    """


class IndexBuildError(CourseIndexError):
    """索引不成立：同一份语料两次构建给出不同的索引（不确定），或计数丢失.

    它是本课唯一一族"内部中间物"的失败（见模块 docstring 第二节）。
    """


class QueryError(CourseIndexError):
    """查询不成立：空查询，或查询里一个可索引的词都没有（见模块 docstring 第三节）.

    它与"查了但没命中"是两件事：前者是**调用点的问题**，后者是**语料的问题**。
    """


class ScoreError(CourseIndexError):
    """分数不可用：命中分数越界（不在 ``[0, 1]``）.

    ``score = 命中的查询词数 / 查询词总数``，因此它在数学上就落在 ``[0, 1]``；
    越界只可能来自实现里的一个减法或一次错误归一化——它必须当场报，而不是被当成"低分"。
    """


class NumericError(CourseIndexError):
    """数值不可用：非有限读数、上界 / 下界非法（负数）.

    读数非有限时，任何比较都会给出一个**静默为假**的结论，因此本包在入口就拒绝。
    """


class ParameterError(CourseIndexError):
    """参数越界：未知文档种类 / 未知性质 / 非正 ``top_k`` / 序号错位.

    出路是**改调用**。本包不接受"名字写错就取个默认值"这种兜底——
    一个被兜住的 ``kind="docs"`` 会被计进语料覆盖报告，
    而它与"我确实收了 ``docs/*.md``"在报告里**完全一样**。
    """


#: 六个族各自"该谁去修"（**这张表要被测试逐键检查**）.
FAMILY_OUTCOMES: dict[str, str] = {
    "CorpusError": "改语料收集：文档名重复或某份材料读不到时，先把那一份补进语料",
    "IndexBuildError": "改索引实现：两次构建不一致时，去掉那个没有固定的量（迭代序 / 集合序）",
    "QueryError": "改查询或改分词：空查询与无词查询都要在调用点被看见",
    "ScoreError": "改打分：命中分数越界时，先回到 score = 命中词数 / 查询词数 的定义",
    "NumericError": "改数据或改实现：非有限读数与非法上下界都属于'数值不可用'",
    "ParameterError": "改调用：文档种类 / 性质名 / top_k / 序号都是调用点的一次决定",
}

#: 本模块真正导出的六个异常类名（与 ``FAMILY_OUTCOMES`` 逐键对齐的闭合检查）.
_FAMILY_CLASSES: dict[str, type[BaseException]] = {
    "CorpusError": CorpusError,
    "IndexBuildError": IndexBuildError,
    "QueryError": QueryError,
    "ScoreError": ScoreError,
    "NumericError": NumericError,
    "ParameterError": ParameterError,
}

if set(FAMILY_OUTCOMES) != set(_FAMILY_CLASSES):  # pragma: no cover - 只在有人改表时触发
    raise CourseIndexError(
        "失败族的两张表不一致：FAMILY_OUTCOMES 的键必须与 _FAMILY_CLASSES 逐一对应——"
        "少一个键的那一族在报告里只剩类名、没有'该怎么办'。"
    )

#: 今天**回来**的族：``TokenError``（day086 命名、此后缺席，本课要**自己分词**）.
RETURNED_FAMILY: str | None = "TokenError"

#: 回来的理由：**与 day099 / day100 请回旧名字的理由同源**.
RETURNED_FAMILY_REASON = (
    "day086 的 ``hf_integration.errors.TokenError`` 第一次给'分词器配置对不上'命名了，"
    "此后各课都只把它当工具函数。day101 需要**自己决定怎么切词**"
    "（中文二元组 + ASCII 词），于是'切出来的东西不对'重新需要一次控制流事件——"
    "这个名字**回来**了，但基类从 ``ParameterError`` 换成 ``CourseIndexError(ValueError)``："
    "今天它要能被同一条 ``except ValueError`` 兜住，与其余六族同源。"
)

#: 今天缺席的那一族：``GradientError``（本课是编索引与检索，一个新式子都没有写）.
ABSENT_FAMILY: str | None = "GradientError"

#: 缺席理由：**与 day100 的理由同源（都是"没有一次反向"），但场景不同**.
ABSENT_FAMILY_REASON = (
    "本课把已经写在仓库里的文字编成一份索引：一个 forward-only 的检索层，"
    "唯一的'梯度事件'仍然是被调用的那一行（``finetune`` 那一侧）。"
    "因此这一族今天不属于本包——它属于被调用的代码。"
)

__all__ = [
    "ABSENT_FAMILY",
    "ABSENT_FAMILY_REASON",
    "FAMILY_OUTCOMES",
    "RETURNED_FAMILY",
    "RETURNED_FAMILY_REASON",
    "CorpusError",
    "CourseIndexError",
    "IndexBuildError",
    "NumericError",
    "ParameterError",
    "QueryError",
    "ScoreError",
]
