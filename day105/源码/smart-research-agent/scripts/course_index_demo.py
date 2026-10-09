#!/usr/bin/env python
"""day101 离线演示：把 100 天的材料编成一份可检索、可复算的索引.

十节，全部离线、全部确定性（不需要 API Key，也不联网）：

```text
1   语料：两种材料（docs/*.md 全文 + 53 个子包的 docstring）
2   分词口径：ASCII 词 + 中文二元组（几个手算可核对的例子）
3   索引：词典 / 词次 / 摘要 + 两次构建逐位相同
4   倒排表：几个 token 的"出现在哪些文档里"
5   检索：金标准查询的名单 + 分数是怎么来的（命中的那几个词）
6   检索的确定性：同一查询两次逐位相同
7   七条性质与三类判据（相等 / 上界 / 下界）
8   四张表：语料 / 索引 / 检索 / 性质
9   失败族：六个族各自"该谁去修" + 回来的族与缺席的族
10  十条笔记 / 五条边界与与既有包的接缝
```

运行方式::

    cd day101/源码/smart-research-agent
    python scripts/course_index_demo.py

产出：

```text
outputs/course_index_demo.txt           本脚本的完整输出（在 .gitignore 里）
outputs/course_index/search_hits.txt    金标准查询的名单（新的产物，不覆盖既有文件）
```

**不覆盖仓库里既有的任何文件**：本课只新增文件；``docs/`` 下的手册是被**读取**的。
"""

from __future__ import annotations

import logging
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.course_index import (  # noqa: E402
    ABSENT_FAMILY,
    ABSENT_FAMILY_REASON,
    FAMILY_OUTCOMES,
    RETURNED_FAMILY,
    RETURNED_FAMILY_REASON,
    corpus as corpus_module,
    index as index_module,
    query as query_module,
    study,
    types,
    verify,
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "course_index_demo.txt"
ARTIFACT_DIR = OUTPUT_DIR / "course_index"

#: 演示里手算展示的那几个 token（可逐个人工核对）。
SAMPLE_TOKENS = ("反向", "传播", "course_index", "graduation", "backprop")


class Report:
    """攒行 + 落盘（**不做任何计算**）."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def section(self, index: int, title: str) -> None:
        """开一节并打印标题."""
        self.lines.append("")
        self.lines.append(f"== {index}. {title}")

    def add(self, *texts: str) -> None:
        """往这一节里追加若干行."""
        self.lines.extend(texts)

    def flush(self) -> None:
        """把攒好的文本同时打印到终端并写入 ``outputs/course_index_demo.txt``."""
        text = "\n".join(self.lines)
        print(text)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_FILE.write_text(text + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# 第 1 节：语料
# --------------------------------------------------------------------------- #


def section_1_corpus(report: Report, corpus: corpus_module.Corpus) -> None:
    """第 1 节：两种材料是什么、各有多少、合计多少字."""
    report.section(1, "语料（两种材料：docs/*.md 全文 + 每个子包的 docstring）")
    for kind in types.DOC_KINDS:
        report.add(f"  [{kind:<10}] {types.docs_of_kind(kind)}")
    report.add("")
    report.add("  前 6 份（手册与会话各取前几行）：")
    for row in study.corpus_rows(corpus)[:6]:
        report.add("  " + row.line())
    report.add("  " + corpus.line())
    report.add(
        "  文档名必须带种类前缀：docs/backprop.md 与子包 backprop 都叫 backprop，"
        "裸名字会撞车。"
    )


# --------------------------------------------------------------------------- #
# 第 2 节：分词口径
# --------------------------------------------------------------------------- #


def section_2_tokenizer(report: Report) -> None:
    """第 2 节：分词的三个手算例子."""
    report.section(2, "分词口径（ASCII 词小写化 + 中文二元组；零依赖、按正文顺序）")
    for sample in ("Course_Index", "反向传播", "好", "a of ab"):
        report.add(f"  tokenize({sample!r}) = {index_module.tokenize(sample)}")
    report.add(
        f"  规则：ASCII 长度 >= {types.MIN_WORD_LEN}；中文串长度 >= {types.NGRAM} 时切全部相邻二元组，"
        "更短的整段作为一个 token。"
    )
    report.add("  为什么不用分词库：换一个版本就会换一份索引，而'索引可复算'是这份产物的全部价值。")


# --------------------------------------------------------------------------- #
# 第 3 节：索引
# --------------------------------------------------------------------------- #


def section_3_index(report: Report, first: index_module.InvertedIndex, second: index_module.InvertedIndex) -> None:
    """第 3 节：索引读数 + 两次构建逐位相同."""
    report.section(3, "索引（倒排表：token → 出现它的文档与次数）")
    report.add("  " + first.line())
    report.add(
        f"  两次构建：摘要 {first.digest()} / {second.digest()}"
        f" | 逐位相同={first.diff_count(second) == 0} | 差异项数 {first.diff_count(second)}"
    )
    report.add(
        "  口径（comparable 的字段）：文档名单 / 每份文档的词数 / 倒排表"
        "（倒排表里的文档名**必须排序**——并列的不确定性会让这条性质永远失败）"
    )


# --------------------------------------------------------------------------- #
# 第 4 节：倒排表
# --------------------------------------------------------------------------- #


def section_4_postings(report: Report, built: index_module.InvertedIndex) -> None:
    """第 4 节：几个 token 的倒排表（手算可核对）."""
    report.section(4, "倒排表（几个 token 出现在哪些文档里）")
    for token in SAMPLE_TOKENS:
        hits = built.documents_containing(token)
        preview = "、".join(hits[:4]) + ("…" if len(hits) > 4 else "")
        report.add(f"  {token:<14} | 文档 {len(hits):>3} | {preview or '（无）'}")


# --------------------------------------------------------------------------- #
# 第 5 节：检索
# --------------------------------------------------------------------------- #


def section_5_search(report: Report, result: query_module.SearchResult) -> None:
    """第 5 节：金标准查询的名单与分数来源."""
    report.section(5, f"检索（金标准查询 {types.GOLD_QUERY!r}：查询包名、命中该包）")
    report.add(f"  {result.line()}")
    for row in study.hit_rows(result):
        report.add("  " + row.line())
    report.add(
        "  分数的定义：score = 该文档命中的查询词数 / 查询词总数 —— "
        "`matched` 那一列就是「这条分数是怎么来的」。"
    )
    report.add("  " + query_module.require_hits(result).line())


# --------------------------------------------------------------------------- #
# 第 6 节：检索的确定性
# --------------------------------------------------------------------------- #


def section_6_determinism(
    report: Report, first: query_module.SearchResult, second: query_module.SearchResult
) -> None:
    """第 6 节：同一查询两次逐位相同."""
    report.section(6, "检索的确定性（同一个查询两次给出同一份名单，顺序也相同）")
    report.add(f"  第一次：{first.line()}")
    report.add(f"  第二次：{second.line()}")
    report.add(
        f"  逐位相同={first.is_identical_to(second)} | 差异项数 {first.diff_count(second)}"
        "（0 = 逐位相同）"
    )
    report.add("  排序规则：分数降序；**并列时按文档名字典序**——不这样兜，两份名单都'对'。")


# --------------------------------------------------------------------------- #
# 第 7 节：七条性质
# --------------------------------------------------------------------------- #


def section_7_properties(report: Report, outcome: verify.PropertyReport) -> None:
    """第 7 节：七条性质逐条（判据类别与读数一起印）."""
    report.section(7, "七条性质与三类判据（相等 / 上界 / 下界）")
    for row in study.property_rows(outcome):
        report.add("  " + row.line())
    report.add("")
    report.add(f"  全部通过：{outcome.ok}")
    for criterion, description in types.CRITERION_DESCRIPTIONS.items():
        count = sum(1 for spec in types.property_specs() if spec.criterion == criterion)
        report.add(f"  [{criterion}] {count} 条：{description}")


# --------------------------------------------------------------------------- #
# 第 8 节：四张表
# --------------------------------------------------------------------------- #


def section_8_tables(report: Report, corpus: corpus_module.Corpus, built: index_module.InvertedIndex, result: query_module.SearchResult) -> None:
    """第 8 节：一次跑完四张表."""
    report.section(8, "四张表（语料 / 索引 / 检索 / 性质）")
    for line in study.study_lines(corpus=corpus, index=built, result=result):
        report.add("  " + line)


# --------------------------------------------------------------------------- #
# 第 9 节：失败族
# --------------------------------------------------------------------------- #


def section_9_families(report: Report) -> None:
    """第 9 节：六个失败族 + 回来的族与缺席的族."""
    report.section(9, "失败族（按「该谁去修」分）")
    for name, outcome in FAMILY_OUTCOMES.items():
        report.add(f"  {name:<16} → {outcome}")
    report.add("")
    report.add(f"  回来的族：{RETURNED_FAMILY}")
    report.add(f"    理由：{RETURNED_FAMILY_REASON}")
    report.add(f"  缺席的族：{ABSENT_FAMILY}")
    report.add(f"    理由：{ABSENT_FAMILY_REASON}")


# --------------------------------------------------------------------------- #
# 第 10 节：笔记、边界与接缝
# --------------------------------------------------------------------------- #


def section_10_notes(report: Report, result: query_module.SearchResult) -> None:
    """第 10 节：十条笔记 + 五条边界 + 接缝 + 名单落盘."""
    report.section(10, "十条笔记 / 五条边界与与既有包的接缝")
    report.add("  十条笔记：")
    for line in study.note_lines():
        report.add("    " + line)
    report.add("")
    report.add("  五条边界（本课明确不承诺的事）：")
    for line in study.boundary_lines():
        report.add("    " + line)
    report.add("")
    report.add("  与既有包的接缝（本课不重写任何子系统）：")
    report.add("    docs       docs/*.md（被读取，不被改写）")
    report.add("    packages   importlib（53 个子包的 __init__ docstring）")
    report.add("    milestone  day100 的 graduation（它数出这个仓库里有多少个子包）")
    report.add("")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / "search_hits.txt"
    text = " ".join([result.query, "→", *result.documents]) + "\n"
    path.write_text(text, encoding="utf-8")
    report.add(f"  名单已落盘：{path}")
    report.add(
        "  注：**不覆盖仓库里既有的任何文件**——本课只新增文件；docs/ 下的手册只被读取。"
    )
    report.add("")
    report.add(f"  性质名单（{len(types.COURSE_INDEX_PROPERTIES)} 条）：{'、'.join(types.COURSE_INDEX_PROPERTIES)}")
    report.add(f"  笔记条数：{len(types.COURSE_INDEX_NOTES)} | 边界条数：{len(types.COURSE_INDEX_BOUNDARIES)}")
    report.add(f"  语料种类：{len(types.DOC_KINDS)} | 分词窗口：{types.NGRAM}")


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #


def main() -> int:
    """按节运行演示，并把完整输出写入 ``outputs/course_index_demo.txt``."""
    logging.getLogger("smart_research_agent").setLevel(logging.WARNING)
    logging.getLogger("smart_research_agent.agent.planner").setLevel(logging.WARNING)
    # 单次计算：语料 / 索引 / 检索各算一次（外加一份索引用于"两次构建"的对照）。
    corpus = corpus_module.build_corpus()
    first = index_module.build_index(corpus)
    second = index_module.build_index(corpus)
    result = query_module.search(types.GOLD_QUERY, index=first)
    again = query_module.search(types.GOLD_QUERY, index=first)
    outcome = verify.check_all(corpus=corpus, index=first, result=result)

    report = Report()
    report.add("#" * 78)
    report.add("# day101：把 100 天的材料编成一份可检索、可复算的索引")
    report.add("# 全部离线、全部确定性：不联网、不读任何环境变量密钥")
    report.add("#" * 78)
    section_1_corpus(report, corpus)
    section_2_tokenizer(report)
    section_3_index(report, first, second)
    section_4_postings(report, first)
    section_5_search(report, result)
    section_6_determinism(report, result, again)
    section_7_properties(report, outcome)
    section_8_tables(report, corpus, first, result)
    section_9_families(report)
    section_10_notes(report, result)
    report.add("")
    report.add(f"演示完成（工作目录 {PROJECT_ROOT}）")
    report.add(f"本节输出已同时写入 {OUTPUT_FILE}")
    report.add("十节全部离线：零网络、零新增依赖；语料、索引、名单与判据都是真算出来的。")
    report.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
