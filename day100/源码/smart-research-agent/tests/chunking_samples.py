"""day062 测试用的分块样本构造器（文档 + 探针 + 度量器）.

放在单独模块里而不是复制到每个测试文件：**同一份样本被七个测试文件用到**
（token、policy、types、四种策略、pipeline/端点、评估），复制七份会让
"样本里加一个代码块"变成要改七处，而漏改的那一处只会表现为
"某个测试不再覆盖原子块保护"——**覆盖率看不出来的那种漏**。

样本本身刻意包含四种"策略差异点"，因此它能同时喂给四个策略：
标题层级（结构策略）、超预算代码块（原子块保护）、中英标点混排
（递归策略的分隔符表）、长短不一的自然段（语义策略的相似度断层）。
"""

from __future__ import annotations

from smart_research_agent.chunking import CharMeasurer, TokenMeasurer
from smart_research_agent.documents import Document, default_registry

#: 四段标题 + 一个 200 余字的代码块 + 六段正文（见模块 docstring）。
GUIDE_MARKDOWN = """# 检索手册

本文给出一套可复现的检索参数，所有数字都在同一台机器上实测。

## 分块

固定长度分块每 320 个字符切一刀，重叠 48 个字符。它不认识标点，因此一定会切在句子中间，
而重叠只能保证被切断的句子在至少一块里是完整的。

递归分块优先在段落边界切开，其次换行，然后是句号与逗号。中文标点必须在分隔符表里，
否则一份中文文档会一路退到按字符硬切，而那种退化不会有任何报错。

```python
def plan(doc_chars: int, max_tokens: int, overlap_tokens: int) -> int:
    step = max_tokens - overlap_tokens
    if doc_chars <= max_tokens:
        return 1
    return -(-(doc_chars - max_tokens) // step) + 1
```

## 检索

默认返回前 5 条，相似度低于 0.35 的直接丢掉。阈值用分位数标定，换 embedding 模型必须重新标定。

混合检索把向量分数与 BM25 分数做归一化后加权，权重 0.6 / 0.4。

## 成本

一次全量重建索引大约 12 万条记录，按每条 320 token 计，embedding 花费约 3.8 美元；
而重叠 15% 意味着其中约 1.8 万条是重复内容，这笔钱换来的只是边界处不丢句子。
"""

#: 没有任何标题的纯段落文档（用来验证"结构策略在无标题文档上退化"）。
PLAIN_TEXT = """第一段讲的是分块预算，它决定了索引的规模与 embedding 的花费。

第二段讲的是重叠，它决定了边界处能不能读到完整的句子。

第三段讲的是检索深度，top-3 之外的内容大概率会被提示词长度稀释掉。

第四段讲的是评估，没有探针集的打分只是一次演示，不是一次测量。
"""

#: 一个超预算的代码块单独成篇（用来验证原子块保护与 oversized 计数）。
BIG_CODE_MARKDOWN = """# 工具脚本

下面是全部实现。

```python
def a():
    return 1

def b():
    return 2

def c():
    return 3
```
"""


def guide_document(source: str = "docs/guide.md") -> Document:
    """把 ``GUIDE_MARKDOWN`` 解析成归一化文档（走 day061 的真实加载器）."""
    return default_registry().load_bytes(
        GUIDE_MARKDOWN.encode("utf-8"), source=source, media_type="text/markdown"
    )


def big_code_document(source: str = "docs/tools.md") -> Document:
    """``BIG_CODE_MARKDOWN`` 解析后的文档."""
    return default_registry().load_bytes(
        BIG_CODE_MARKDOWN.encode("utf-8"), source=source, media_type="text/markdown"
    )


def plain_document(source: str = "docs/plain.txt") -> Document:
    """``PLAIN_TEXT`` 解析后的文档（无标题）."""
    return default_registry().load_bytes(
        PLAIN_TEXT.encode("utf-8"), source=source, media_type="text/plain"
    )


def text_document(
    text: str, source: str = "docs/inline.txt", media_type: str = "text/plain"
) -> Document:
    """任意文本 → 文档（测试里临时构造小样本用）."""
    return default_registry().load_bytes(
        text.encode("utf-8"), source=source, media_type=media_type
    )


def chars() -> TokenMeasurer:
    """字符度量器（**所有测试都用它**：确定性来自"只调用 len"）."""
    return CharMeasurer()
