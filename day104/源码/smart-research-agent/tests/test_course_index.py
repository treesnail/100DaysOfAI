"""``course_index``：把 100 天的材料编成一份可检索、可复算的索引（day101）.

本文件覆盖新包的七个模块：口径表、失败族、语料、索引、检索、性质与表。
样本有两类：**真实语料**（``docs/*.md`` + 53 个子包的 docstring）与
**注入的小语料**（构造反例时用）。因此这一课考的就是
"同一份语料能不能两次编出同一份索引、同一个查询能不能两次给出同一份命中"。
"""

from __future__ import annotations

import dataclasses
import pathlib
import types as _pytypes

import pytest

from smart_research_agent.course_index import (
    corpus as corpus_module,
    errors,
    index as index_module,
    query as query_module,
    study,
    types,
    verify,
)

# --------------------------------------------------------------------------- 口径表


def test_two_doc_kinds_are_closed() -> None:
    """两种语料：名单与说明逐键对齐。"""
    assert len(types.DOC_KINDS) == 2
    assert set(types.DOC_KIND_DESCRIPTIONS) == set(types.DOC_KINDS)
    assert types.require_doc_kind(types.DOC_KIND_DOC) == types.DOC_KIND_DOC
    with pytest.raises(errors.ParameterError, match="未知的语料种类"):
        types.require_doc_kind("docs")


def test_seven_properties_and_three_criteria() -> None:
    """7 条性质：名单与规格表逐键对齐，且三类判据都有性质。"""
    assert len(types.COURSE_INDEX_PROPERTIES) == 7
    assert set(types.PROPERTY_SPECS) == set(types.COURSE_INDEX_PROPERTIES)
    criteria = {spec.criterion for spec in types.property_specs()}
    assert criteria == set(types.CRITERIA)
    assert types.CRITERION_EQUALITY in criteria
    assert types.CRITERION_UPPER_BOUND in criteria
    assert types.CRITERION_LOWER_BOUND in criteria


def test_notes_are_ten_and_ordered() -> None:
    """十条笔记，且顺序表与键集合一致。"""
    assert len(types.COURSE_INDEX_NOTES) == 10
    assert types.COURSE_INDEX_NOTES_ORDER == tuple(types.COURSE_INDEX_NOTES)
    assert all(types.COURSE_INDEX_NOTES.values())


def test_boundaries_are_five() -> None:
    """五条边界是一份非空清单（这一课明确不承诺的事）。"""
    assert len(types.COURSE_INDEX_BOUNDARIES) == 5
    assert all(types.COURSE_INDEX_BOUNDARIES)


def test_tokenizer_parameters_and_gold_standard() -> None:
    """分词参数与金标准的关系：查询包名、命中该包。"""
    assert types.NGRAM == 2
    assert types.MIN_WORD_LEN == 2
    assert types.TOP_K_DEFAULT >= 1
    assert types.GOLD_DOC == f"{types.DOC_KIND_SUBPACKAGE}:{types.GOLD_QUERY}"
    assert types.docs_of_kind(types.DOC_KIND_DOC)


def test_require_helpers() -> None:
    """两个 require 帮手：合法值原样返回，非法值当场拒绝。"""
    assert types.require_positive_int("top_k", 5) == 5
    assert types.require_property(types.PROPERTY_INDEX_IS_REPRODUCIBLE)
    with pytest.raises(errors.ParameterError, match="必须是 >= 1 的整数"):
        types.require_positive_int("top_k", 0)
    with pytest.raises(errors.ParameterError, match="必须是 >= 1 的整数"):
        types.require_positive_int("top_k", True)
    with pytest.raises(errors.ParameterError, match="未知的性质"):
        types.require_property("nope")


def _property_kwargs(**overrides: object) -> dict[str, object]:
    """一份合法的 PropertySpec 关键字参数."""
    base: dict[str, object] = {
        "id": types.PROPERTY_INDEX_IS_REPRODUCIBLE,
        "description": "说明",
        "criterion": types.CRITERION_EQUALITY,
        "failure": "失败意味着",
    }
    base.update(overrides)
    return base


def test_property_spec_validates_fields() -> None:
    """性质记录的两条护栏与两个渲染方法。"""
    spec = types.PropertySpec(**_property_kwargs())
    assert spec.to_dict()["criterion"] == types.CRITERION_EQUALITY
    assert spec.line().startswith("[equality]")
    with pytest.raises(errors.ParameterError, match="不能为空"):
        types.PropertySpec(**_property_kwargs(description=""))
    with pytest.raises(errors.ParameterError, match="判据"):
        types.PropertySpec(**_property_kwargs(criterion="nope"))


# --------------------------------------------------------------------------- 失败族


def test_error_families_are_closed() -> None:
    """六个失败族：类表与处置表逐键对齐，且都是 ValueError 的子类。"""
    assert len(errors.FAMILY_OUTCOMES) == 6
    assert set(errors.FAMILY_OUTCOMES) == set(errors._FAMILY_CLASSES)
    for name in errors.FAMILY_OUTCOMES:
        cls = errors._FAMILY_CLASSES[name]
        assert issubclass(cls, errors.CourseIndexError)
        assert issubclass(cls, ValueError)


def test_returned_and_absent_families_are_declared() -> None:
    """回来的族与缺席的族都是常量，且各自带理由。"""
    assert errors.RETURNED_FAMILY == "TokenError"
    assert errors.ABSENT_FAMILY == "GradientError"
    assert errors.RETURNED_FAMILY_REASON
    assert errors.ABSENT_FAMILY_REASON


def test_each_family_is_raisable() -> None:
    """每个失败族都能被 raise / except。"""
    for cls in errors._FAMILY_CLASSES.values():
        with pytest.raises(cls):
            raise cls("boom")


# --------------------------------------------------------------------------- 语料


def _small_corpus() -> corpus_module.Corpus:
    """一份注入的小语料（一份手册 + 一个子包）。"""
    return corpus_module.build_corpus(
        documents=(("alpha", "反向传播 与 梯度 下降"),),
        subpackages=(("demo", "course_index demo package"),),
    )


def test_qualify_and_unqualify() -> None:
    """文档名的两种拼法：带种类前缀，可以来回拆。"""
    name = corpus_module.qualify(types.DOC_KIND_DOC, "backprop")
    assert name == "doc:backprop"
    assert corpus_module.unqualify(name) == (types.DOC_KIND_DOC, "backprop")
    with pytest.raises(errors.ParameterError, match="未知的语料种类"):
        corpus_module.qualify("docs", "x")
    with pytest.raises(errors.ParameterError, match="裸名字不能为空"):
        corpus_module.qualify(types.DOC_KIND_DOC, "")
    with pytest.raises(errors.ParameterError, match="必须带种类"):
        corpus_module.unqualify("backprop")


def test_document_validates_fields() -> None:
    """语料记录的三条护栏与两个渲染方法。"""
    document = corpus_module.Document(name="doc:alpha", kind=types.DOC_KIND_DOC, text="内容")
    assert document.raw_name == "alpha"
    assert document.chars == 2
    assert document.to_dict()["kind"] == types.DOC_KIND_DOC
    assert "doc:alpha" in document.line()
    with pytest.raises(errors.ParameterError, match="对不上"):
        corpus_module.Document(
            name="doc:alpha", kind=types.DOC_KIND_SUBPACKAGE, text="内容"
        )
    with pytest.raises(errors.CorpusError, match="正文是空的"):
        corpus_module.Document(name="doc:alpha", kind=types.DOC_KIND_DOC, text="   ")


def test_corpus_closure_and_views() -> None:
    """语料的三条闭合检查与只读视图。"""
    corpus = _small_corpus()
    assert len(corpus.names) == 2
    assert len(corpus.docs) == 1
    assert len(corpus.subpackages) == 1
    assert corpus.document_of("doc:alpha").raw_name == "alpha"
    assert corpus.total_chars > 0
    assert corpus.to_dict()["documents"] == 2
    assert "语料" in corpus.line()
    with pytest.raises(errors.CorpusError, match="不能为空"):
        corpus_module.Corpus(documents=())
    with pytest.raises(errors.CorpusError, match="重复的文档名"):
        dataclasses.replace(corpus, documents=corpus.documents + (corpus.documents[0],))
    with pytest.raises(errors.ParameterError, match="语料里没有文档"):
        corpus.document_of("doc:nope")


def test_build_corpus_default_covers_everything() -> None:
    """默认语料：两种材料都覆盖完整，且名字唯一。"""
    corpus = corpus_module.build_corpus()
    expected = corpus_module.expected_names()
    assert corpus.missing(expected) == ()
    assert len(corpus.names) == len(set(corpus.names))
    assert len(corpus.docs) >= 20
    assert len(corpus.subpackages) >= 50


def test_read_documents_and_subpackages() -> None:
    """两种语料的读取：按名字排序，正文非空。"""
    documents = corpus_module.read_documents()
    assert list(documents) == sorted(documents, key=lambda row: row[0])
    assert all(text.strip() for _, text in documents)
    subpackages = corpus_module.subpackage_docstrings()
    assert list(subpackages) == sorted(subpackages, key=lambda row: row[0])
    assert any(name == "course_index" for name, _ in subpackages)


def test_docs_dir_and_discovery(tmp_path: pathlib.Path) -> None:
    """手册目录：真的存在；不存在时抛；空目录也抛；子包发现是排序的。"""
    directory = corpus_module.docs_dir()
    assert directory.is_dir()
    with pytest.raises(errors.CorpusError, match="手册目录不存在"):
        corpus_module.docs_dir(tmp_path)
    names = corpus_module.discover_subpackages()
    assert list(names) == sorted(names)
    assert "course_index" in names
    empty = tmp_path / corpus_module.DOCS_DIRNAME
    empty.mkdir()
    with pytest.raises(errors.CorpusError, match=r"一份 \.md 都没有"):
        corpus_module.read_documents(tmp_path)


def test_discover_requires_a_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """被扫描的对象不是包（没有 __path__）时抛 CorpusError。"""
    monkeypatch.setattr(
        corpus_module.importlib, "import_module", lambda name: _pytypes.ModuleType(name)
    )
    with pytest.raises(errors.CorpusError, match="不是一个包"):
        corpus_module.discover_subpackages()


def test_subpackage_without_docstring_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """一个没有 docstring 的子包必须被点名，而不是被静默跳过。"""
    blank = _pytypes.ModuleType("blank")
    blank.__doc__ = None
    monkeypatch.setattr(corpus_module, "discover_subpackages", lambda: ("blank",))
    monkeypatch.setattr(corpus_module.importlib, "import_module", lambda name: blank)
    with pytest.raises(errors.CorpusError, match="没有 docstring"):
        corpus_module.subpackage_docstrings()


def test_corpus_missing_and_lines() -> None:
    """语料的缺口清单与逐行打印。"""
    corpus = _small_corpus()
    assert corpus.missing(("doc:alpha", "doc:nope")) == ("doc:nope",)
    lines = corpus_module.corpus_lines(corpus)
    assert lines[0].startswith("语料表")
    assert any("doc:alpha" in line for line in lines)


# --------------------------------------------------------------------------- 分词与索引


def test_tokenize_ascii_and_cjk() -> None:
    """分词：ASCII 词小写化 + 中文二元组；短中文整段变成一个 token。"""
    assert index_module.tokenize("Course_Index") == ("course_index",)
    assert index_module.tokenize("反向传播") == ("反向", "向传", "传播")
    assert index_module.tokenize("好") == ("好",)


def test_tokenize_drops_short_words() -> None:
    """长度为 1 的 ASCII 词被丢掉（长度 2 的保留）。"""
    assert index_module.tokenize("a of ab") == ("of", "ab")
    assert index_module.token_count("a of ab") == 2
    assert index_module.tokenize("a b") == ()


def test_tokenize_rejects_non_string() -> None:
    """分词只接受字符串。"""
    with pytest.raises(errors.ParameterError, match="只接受字符串"):
        index_module.tokenize(123)  # type: ignore[arg-type]


def test_build_index_is_reproducible() -> None:
    """同一份语料两次建索引逐位相同，且索引非空。"""
    corpus = _small_corpus()
    first = index_module.build_index(corpus)
    second = index_module.build_index(corpus)
    assert first.diff_count(second) == 0
    assert first.digest() == second.digest()
    assert len(first.digest()) == index_module.INDEX_DIGEST_LENGTH
    assert first.document_count == 2
    assert first.term_count > 0
    assert first.total_tokens > 0
    assert "索引" in first.line()


def test_inverted_index_lookups() -> None:
    """倒排表的三种查询：token → 文档、文档名 → 词数、未知项被拒。"""
    corpus = _small_corpus()
    built = index_module.build_index(corpus)
    assert built.documents_containing("反向") == ("doc:alpha",)
    assert built.postings_of("不存在的词") == ()
    assert built.tokens_of_document("doc:alpha") >= 3
    assert built.to_dict()["document_count"] == 2
    with pytest.raises(errors.ParameterError, match="索引里没有文档"):
        built.tokens_of_document("doc:nope")


def test_inverted_index_guards() -> None:
    """倒排索引的四条护栏：空、重名、顺序错位、未排序。"""
    good = index_module.build_index(_small_corpus())
    with pytest.raises(errors.IndexBuildError, match="不能没有文档"):
        index_module.InvertedIndex(documents=(), tokens_per_document=(), postings=())
    with pytest.raises(errors.IndexBuildError, match="文档名不唯一"):
        index_module.InvertedIndex(
            documents=("doc:a", "doc:a"),
            tokens_per_document=(("doc:a", 1), ("doc:a", 1)),
            postings=(),
        )
    with pytest.raises(errors.IndexBuildError, match="文档顺序"):
        index_module.InvertedIndex(
            documents=("doc:a",),
            tokens_per_document=(("doc:b", 1),),
            postings=(),
        )
    with pytest.raises(errors.IndexBuildError, match="没有排序"):
        index_module.InvertedIndex(
            documents=("doc:a", "doc:b"),
            tokens_per_document=(("doc:a", 1), ("doc:b", 1)),
            postings=(("x", (("doc:b", 1), ("doc:a", 1))),),
        )
    with pytest.raises(errors.NumericError, match="不能为负"):
        dataclasses.replace(good, tokens_per_document=(("doc:alpha", -1), ("subpackage:demo", 1)))
    with pytest.raises(errors.ParameterError, match="空 token"):
        index_module.InvertedIndex(
            documents=("doc:a",),
            tokens_per_document=(("doc:a", 1),),
            postings=(("", (("doc:a", 1),)),),
        )
    with pytest.raises(errors.NumericError, match="计数必须 >= 1"):
        index_module.InvertedIndex(
            documents=("doc:a",),
            tokens_per_document=(("doc:a", 1),),
            postings=(("x", (("doc:a", 0),)),),
        )


def test_index_digest_and_require_reproducible() -> None:
    """索引摘要与"拒绝不一致"的那条路。"""
    corpus = _small_corpus()
    first = index_module.build_index(corpus)
    second = index_module.build_index(corpus)
    assert index_module.index_digest(first) == first.digest()
    assert index_module.require_reproducible(first, second) is first
    other = index_module.build_index(
        corpus_module.build_corpus(
            documents=(("beta", "另一份语料"),), subpackages=(("demo", "another"),)
        )
    )
    assert first.diff_count(other) > 0
    with pytest.raises(errors.IndexBuildError, match="两份索引差"):
        index_module.require_reproducible(first, other)


def test_index_lines() -> None:
    """索引逐行打印（可截断）。"""
    lines = index_module.index_lines(limit=3)
    assert lines[0].startswith("倒排索引表")
    assert len(lines) == 3 + 2


# --------------------------------------------------------------------------- 检索


def test_unique_tokens_dedupes() -> None:
    """查询分词去重（保留首次出现的顺序）。"""
    assert query_module.unique_tokens("向量 向量 index index") == ("向量", "index")


def test_search_default_gold() -> None:
    """默认检索：金标准查询命中金标准材料，分数为 1.0。

    同一个词可能同时命中"手册"与"子包"两份材料（``doc:course_index`` 与
    ``subpackage:course_index``），因此这里只断言"命中且名次 >= 1"，
    不断言名次恰好是 1——**并列的先后是排序规则的结果，不是这条性质要管的事**。
    """
    result = query_module.search(types.GOLD_QUERY)
    assert types.GOLD_DOC in result.documents
    assert result.rank_of(types.GOLD_DOC) >= 1
    assert result.top_score == 1.0
    assert "查询" in result.line()
    assert result.to_dict()["tokens"] == [types.GOLD_QUERY]
    assert query_module.require_hits(result).is_identical_to(result)


def test_search_is_deterministic() -> None:
    """同一查询两次检索逐位相同。"""
    first = query_module.search(types.GOLD_QUERY)
    second = query_module.search(types.GOLD_QUERY)
    assert first.is_identical_to(second) is True
    assert first.diff_count(second) == 0
    assert query_module.search_matches(first, second) == 0


def test_search_scores_and_order() -> None:
    """分数 = 命中词数 / 查询词数；排序是分数降序、名字升序。"""
    result = query_module.search("反向传播 与 梯度", index=index_module.build_index(_small_corpus()))
    assert result.hits
    assert result.hits[0].score <= 1.0
    scores = [-round(hit.score, query_module.SCORE_DIGITS) for hit in result.hits]
    assert scores == sorted(scores)


def test_search_rejects_bad_input() -> None:
    """空查询 / 无词查询 / 非法 top_k 都被当场拒绝。"""
    with pytest.raises(errors.QueryError, match="一个 token 都没有"):
        query_module.search("   ")
    with pytest.raises(errors.QueryError, match="一个 token 都没有"):
        query_module.search("、、、")
    with pytest.raises(errors.ParameterError, match="top_k"):
        query_module.search(types.GOLD_QUERY, top_k=0)


def test_hit_guards() -> None:
    """命中的三条护栏与两个渲染方法。"""
    hit = query_module.Hit(document="doc:alpha", score=0.5, matched=("反向",))
    assert hit.to_dict()["matched"] == ["反向"]
    assert "doc:alpha" in hit.line()
    with pytest.raises(errors.ParameterError, match="文档名不能为空"):
        query_module.Hit(document="", score=0.5, matched=("x",))
    with pytest.raises(errors.ScoreError, match=r"\[0, 1\]"):
        query_module.Hit(document="doc:a", score=1.5, matched=("x",))
    with pytest.raises(errors.ScoreError, match="一个命中词都没有"):
        query_module.Hit(document="doc:a", score=0.5, matched=())
    with pytest.raises(errors.NumericError, match="必须有限"):
        query_module.Hit(document="doc:a", score=float("nan"), matched=("x",))


def test_search_result_guards() -> None:
    """检索结果的三条护栏：空查询、空词、乱序名单。"""
    good = query_module.search(types.GOLD_QUERY)
    with pytest.raises(errors.QueryError, match="查询不能为空"):
        dataclasses.replace(good, query="")
    with pytest.raises(errors.QueryError, match="一个 token 都没有"):
        query_module.SearchResult(query="q", query_tokens=(), hits=())
    with pytest.raises(errors.ParameterError, match="排序不是"):
        query_module.SearchResult(
            query="q",
            query_tokens=("q",),
            hits=(
                query_module.Hit(document="doc:b", score=0.5, matched=("q",)),
                query_module.Hit(document="doc:a", score=1.0, matched=("q",)),
            ),
        )


def test_require_hits_and_lines() -> None:
    """"拒绝空名单"那条路 + 检索逐行打印。"""
    good = query_module.search(types.GOLD_QUERY)
    assert query_module.require_hits(good) is good
    empty = query_module.SearchResult(query="q", query_tokens=("q",), hits=())
    assert empty.top_score == 0.0
    assert empty.rank_of("doc:a") == 0
    assert "命中 0 条" in empty.line()
    with pytest.raises(errors.QueryError, match="一条命中都没有"):
        query_module.require_hits(empty)
    lines = query_module.search_lines(good)
    assert lines[0].startswith("检索表")


# --------------------------------------------------------------------------- 性质


def test_crosscheck_three_criteria() -> None:
    """三类判据各自的判定与渲染。"""
    equal = verify.CrossCheck(
        name="相等", left="a", right="b", reading=2.0, expected=2.0, exact=True
    )
    assert equal.criterion == types.CRITERION_EQUALITY
    assert equal.passed is True
    assert "读数" in equal.line()
    upper = verify.CrossCheck(
        name="上界", left="a", right="b", reading=1.0, expected=1.0, upper_bound=1.0
    )
    assert upper.criterion == types.CRITERION_UPPER_BOUND
    assert upper.passed is True
    assert "≤ 上界" in upper.line()
    lower = verify.CrossCheck(
        name="下界", left="a", right="b", reading=1.0, expected=1.0, lower_bound=1.0
    )
    assert lower.criterion == types.CRITERION_LOWER_BOUND
    assert lower.passed is True
    assert "≥ 下界" in lower.line()
    assert lower.to_dict()["passed"] is True


def test_crosscheck_failures_and_guards() -> None:
    """判据不满足时的读数 + 五条构造护栏。"""
    failed = verify.CrossCheck(
        name="下界", left="a", right="b", reading=0.0, expected=1.0, lower_bound=1.0
    )
    assert failed.passed is False
    assert "不满足" in failed.line()
    tolerant = verify.CrossCheck(
        name="容差相等", left="a", right="b", reading=1.0, expected=1.0 + 1e-15, exact=False
    )
    assert tolerant.criterion == types.CRITERION_EQUALITY
    assert tolerant.passed is True
    with pytest.raises(errors.ParameterError, match="不能为空"):
        verify.CrossCheck(name="", left="a", right="b", reading=0.0, expected=0.0)
    with pytest.raises(errors.NumericError, match="必须有限"):
        verify.CrossCheck(name="n", left="a", right="b", reading=float("nan"), expected=0.0)
    with pytest.raises(errors.ParameterError, match="同时给了上界与下界"):
        verify.CrossCheck(
            name="n", left="a", right="b", reading=0.0, expected=0.0, upper_bound=0.0, lower_bound=0.0
        )
    with pytest.raises(errors.NumericError, match="上界必须是有限非负数"):
        verify.CrossCheck(
            name="n", left="a", right="b", reading=0.0, expected=0.0, upper_bound=-1.0
        )
    with pytest.raises(errors.NumericError, match="下界必须是有限数"):
        verify.CrossCheck(
            name="n", left="a", right="b", reading=0.0, expected=0.0, lower_bound=float("inf")
        )


def test_property_outcome_guards() -> None:
    """PropertyOutcome 的两条护栏与 criterion / line / to_dict。"""
    ok = verify.PropertyOutcome(
        name="x", applicable=True, passed=True, evidence=("a",),
        cross_check=verify.CrossCheck(name="c", left="l", right="r", reading=1.0, expected=1.0),
    )
    assert ok.criterion == types.CRITERION_EQUALITY
    assert ok.line().startswith("[通过] x")
    assert ok.to_dict()["passed"] is True
    not_applicable = verify.PropertyOutcome(name="x", applicable=False, passed=False)
    assert "不适用" in not_applicable.line()
    with pytest.raises(errors.NumericError, match="不适用"):
        verify.PropertyOutcome(name="x", applicable=False, passed=True)
    with pytest.raises(errors.NumericError, match="不一致"):
        verify.PropertyOutcome(
            name="x", applicable=True, passed=True,
            cross_check=verify.CrossCheck(name="c", left="l", right="r", reading=2.0, expected=1.0),
        )


def test_property_report_views_and_require_ok() -> None:
    """PropertyReport：适用集合 / ok / require_ok / lines 把不适用排在前面。"""
    report = verify.check_all()
    assert report.ok is True
    assert len(report.applicable) == 7
    assert report.require_ok() is None
    assert len(report.lines()) == 7
    assert report.to_dict()["counts"]["applicable"] == 7
    bad = verify.PropertyReport(
        outcomes=(verify.PropertyOutcome(name="x", applicable=True, passed=False),)
    )
    assert bad.ok is False
    with pytest.raises(errors.CorpusError, match="未全部通过"):
        bad.require_ok()


def test_check_functions_default() -> None:
    """七条性质函数在默认输入上全部通过。"""
    assert verify.check_corpus_covers_all_docs().passed is True
    assert verify.check_corpus_covers_all_subpackages().passed is True
    assert verify.check_index_is_reproducible().passed is True
    assert verify.check_search_is_deterministic().passed is True
    assert verify.check_scores_within_bounds().passed is True
    assert verify.check_every_document_is_indexed().passed is True
    assert verify.check_gold_query_finds_gold_document().passed is True


def test_check_corpus_fails_when_missing() -> None:
    """覆盖检查：期望名单里多一项时失败并点名。"""
    corpus = _small_corpus()
    docs = verify.check_corpus_covers_all_docs(corpus, expected=("doc:alpha", "doc:nope"))
    assert docs.passed is False
    assert "doc:nope" in docs.evidence[1]
    subs = verify.check_corpus_covers_all_subpackages(
        corpus, expected=("subpackage:demo", "subpackage:nope")
    )
    assert subs.passed is False


def test_check_index_and_search_fail() -> None:
    """可复算 / 确定性两条相等判据的方向。"""
    first = index_module.build_index(_small_corpus())
    other = index_module.build_index(
        corpus_module.build_corpus(documents=(("beta", "另一份"),), subpackages=(("demo", "x"),))
    )
    assert verify.check_index_is_reproducible(first, other).passed is False
    left = query_module.search("反向", index=first)
    right = query_module.search("梯度", index=first)
    assert verify.check_search_is_deterministic(left, right).passed is False


def test_check_scores_and_tokens_directions() -> None:
    """上界 / 下界判据的方向（用鸭子类型的替身注入越界读数）。"""

    @dataclasses.dataclass(frozen=True)
    class _StubResult:
        top_score: float

        def line(self) -> str:
            return "stub"

    over = verify.check_scores_within_bounds(_StubResult(top_score=1.5))  # type: ignore[arg-type]
    assert over.passed is False
    assert over.criterion == types.CRITERION_UPPER_BOUND
    empty_index = index_module.InvertedIndex(
        documents=("doc:x",), tokens_per_document=(("doc:x", 0),), postings=()
    )
    empty = verify.check_every_document_is_indexed(empty_index)
    assert empty.passed is False
    assert empty.criterion == types.CRITERION_LOWER_BOUND


def test_check_gold_query_fails_on_wrong_query() -> None:
    """金标准判据：换一个不含金标准材料的查询就失败。"""
    result = query_module.search("conv_net")
    outcome = verify.check_gold_query_finds_gold_document(result)
    assert outcome.passed is False
    assert outcome.criterion == types.CRITERION_LOWER_BOUND


def test_check_all_default_and_injected() -> None:
    """check_all：默认全通过；注入输入时给出同一份 7 行报告。"""
    report = verify.check_all()
    assert report.ok is True
    assert [outcome.name for outcome in report.outcomes] == list(types.COURSE_INDEX_PROPERTIES)
    small = _small_corpus()
    injected = verify.check_all(
        corpus=small,
        index=index_module.build_index(small),
        result=query_module.search("反向", index=index_module.build_index(small)),
        expected=small.names,
    )
    assert [outcome.name for outcome in injected.outcomes] == list(types.COURSE_INDEX_PROPERTIES)


def test_require_ok() -> None:
    """require_ok：默认放行，注入一份失败的报告时抛 CorpusError。"""
    assert verify.require_ok() is not None
    bad = verify.PropertyReport(
        outcomes=(verify.PropertyOutcome(name="x", applicable=True, passed=False),)
    )
    with pytest.raises(errors.CorpusError, match="未全部通过"):
        verify.require_ok(bad)


# --------------------------------------------------------------------------- 表


def test_corpus_rows() -> None:
    """语料表：行数与语料一致，每行带字符数。"""
    rows = study.corpus_rows()
    assert len(rows) == len(corpus_module.build_corpus().documents)
    assert all(row.chars > 0 for row in rows)
    assert "doc:" in rows[0].line()


def test_index_rows() -> None:
    """索引表：按出现文档数降序，可限行数。"""
    rows = study.index_rows(limit=6)
    assert len(rows) == 6
    assert rows[0].documents >= rows[-1].documents
    assert "文档" in rows[0].line()


def test_hit_rows() -> None:
    """检索表：名次从 1 起。"""
    rows = study.hit_rows()
    assert rows
    assert rows[0].rank == 1
    assert "分数" in rows[0].line()


def test_property_rows() -> None:
    """性质表：7 行、判据类别齐全、全部通过。"""
    rows = study.property_rows()
    assert len(rows) == 7
    assert all(row.passed for row in rows)
    assert {row.criterion for row in rows} == set(types.CRITERIA)
    assert "读数" in rows[0].line()


def test_note_and_boundary_lines() -> None:
    """笔记 / 边界两类文本行。"""
    assert len(study.note_lines()) == 10
    assert len(study.note_lines(limit=4)) == 4
    assert len(study.boundary_lines()) == len(types.COURSE_INDEX_BOUNDARIES)


def test_study_lines_runs_all_four_tables() -> None:
    """四张表一次跑完：四个小标题。"""
    lines = study.study_lines()
    headings = [line for line in lines if line.startswith("== ")]
    assert len(headings) == 4
    assert any("语料表" in line for line in headings)
    assert any("性质表" in line for line in headings)


def test_study_lines_accepts_injected_inputs() -> None:
    """四张表接收注入的语料与索引。"""
    small = _small_corpus()
    lines = study.study_lines(
        corpus=small,
        index=index_module.build_index(small),
        result=query_module.search("反向", index=index_module.build_index(small)),
    )
    assert len([line for line in lines if line.startswith("== ")]) == 4


def test_to_dict_lines() -> None:
    """把带 to_dict 的行折成 JSON 化字段（四张表都能导出）。"""
    assert study.to_dict_lines(study.corpus_rows())[0]["name"]
    assert study.to_dict_lines(study.index_rows())[0]["token"]
    assert study.to_dict_lines(study.hit_rows())[0]["rank"] == 1
    assert study.to_dict_lines(study.property_rows())[0]["criterion"]


# --------------------------------------------------------------------------- 包


def test_package_all_is_sorted_and_unique() -> None:
    """包的公开名单：字母序、无重复、不含子模块名。"""
    import smart_research_agent.course_index as package

    assert package.__all__ == sorted(package.__all__)
    assert len(package.__all__) == len(set(package.__all__))
    assert set(package.__all__) & {
        "errors",
        "types",
        "corpus",
        "index",
        "query",
        "verify",
        "study",
    } == set()


def test_package_exposes_key_names() -> None:
    """包级命名空间真的导入了关键名字（不是只在 __all__ 里）。"""
    import smart_research_agent.course_index as package

    for name in (
        "build_corpus",
        "build_index",
        "search",
        "check_all",
        "study_lines",
        "CourseIndexError",
    ):
        assert name in package.__all__
        assert hasattr(package, name)
