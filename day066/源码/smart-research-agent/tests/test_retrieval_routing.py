"""day066 ``retrieval.routing`` 的单元测试：显式选路的那一层.

路由是一个**业务决策**（"这句话该去查哪个库"），把它写成一层可读、可测、
可回显的结构，替换掉散在调用点上的 ``if doc_type == ...``。因此本文件的断言
集中在三件事上：

```text
三条规则   默认 / 唯一的那个 / 必须显式指定——每一条都有正例与反例
报错格式   三段式：现象 → 可用清单 → 出路（读报的人要能照着自己改对）
可回显     选路结论进 notes、进 report()（"结果不对"时先确认"该不该是它"）
```

全部离线、确定性：库是 ``FlatVectorStore``，编码器是查表的 ``TableEmbedding``。
"""

from __future__ import annotations

from typing import Any

import pytest

from smart_research_agent.retrieval.errors import IndexStateError, QueryError
from smart_research_agent.retrieval.retriever import Retriever
from smart_research_agent.retrieval.routing import RouteDecision, StoreRouter
from smart_research_agent.retrieval.types import RetrievalQuery
from tests.retrieval_samples import (
    QUERY_TEXTS,
    RECORD_IDS,
    TableEmbedding,
    flat_store,
    sample_manifest,
    sample_records,
    sample_store,
)

AXIS = QUERY_TEXTS["axis"]


def make_retriever(name: str = "alpha", **overrides: Any) -> Retriever:
    """一个名字与库都可控的检索器（默认装在八条样本记录上）."""
    return Retriever(sample_store(), TableEmbedding(), name=name, **overrides)


def small_retriever(name: str = "beta", count: int = 2, **overrides: Any) -> Retriever:
    """只装前 ``count`` 条样本记录的检索器（用来区分"结果来自哪一路"）."""
    store = flat_store(sample_records()[:count], metric="cosine")
    return Retriever(store, TableEmbedding(), name=name, **overrides)


def two_route_router(*, default: str | None = None) -> StoreRouter:
    """一个装了两路的路由器（注册顺序刻意与字典序相反：alpha 先注册、beta 后）."""
    router = StoreRouter(default=default)
    router.register("alpha", make_retriever("alpha"), description="手册库")
    router.register("beta", small_retriever("beta"), description="源码库")
    return router


# --------------------------------------------------------------------------- #
# RouteDecision：选了谁 + 为什么选它
# --------------------------------------------------------------------------- #


class TestRouteDecision:
    """``reason`` 是这个结构存在的理由：没有理由的选路等于不可排查."""

    def test_defaults_to_matched(self) -> None:
        decision = RouteDecision(name="alpha", reason="调用方显式指定的路由")

        assert decision.matched is True
        assert decision.to_dict() == {
            "name": "alpha",
            "reason": "调用方显式指定的路由",
            "matched": True,
        }

    def test_summary_line_marks_explicit_choice(self) -> None:
        line = RouteDecision(name="alpha", reason="调用方显式指定的路由").summary_line()

        assert line == "路由 alpha（显式）：调用方显式指定的路由"

    def test_summary_line_marks_a_choice_made_for_the_caller(self) -> None:
        """``matched=False`` 时先确认"该不该是它"——这是排查时的决定性差别."""
        line = RouteDecision(
            name="alpha", reason="调用方没有指定 route，按路由器的 default 选择", matched=False
        ).summary_line()

        assert line.startswith("路由 alpha（路由器选定）：")
        assert "default" in line

    @pytest.mark.parametrize("name", ["", "   "])
    def test_empty_name_is_rejected(self, name: str) -> None:
        """选路的结论不能是"选了某个没名字的东西"."""
        with pytest.raises(QueryError) as excinfo:
            RouteDecision(name=name, reason="有理由")

        assert "RouteDecision.name 必须是非空字符串" in str(excinfo.value)

    @pytest.mark.parametrize("reason", ["", "   "])
    def test_empty_reason_is_rejected(self, reason: str) -> None:
        with pytest.raises(QueryError) as excinfo:
            RouteDecision(name="alpha", reason=reason)

        assert "RouteDecision.reason 必须是非空字符串" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# 注册与视图
# --------------------------------------------------------------------------- #


class TestRegistration:
    """路由名是"这句话该去哪"的唯一依据，因此重名必须拒绝而不是覆盖."""

    def test_register_and_read_back(self) -> None:
        router = StoreRouter()
        retriever = make_retriever("alpha")

        router.register("alpha", retriever, description="  手册库  ")

        assert router.names() == ["alpha"]
        assert len(router) == 1
        assert "alpha" in router
        assert router.default == ""

    def test_names_are_sorted_not_insertion_ordered(self) -> None:
        """报告要能被逐行 diff，而注册序会随装配顺序变化."""
        router = StoreRouter()
        router.register("zeta", make_retriever("zeta"))
        router.register("alpha", make_retriever("alpha"))
        router.register("mid", make_retriever("mid"))

        assert router.names() == ["alpha", "mid", "zeta"]

    def test_duplicate_name_is_rejected_not_overwritten(self) -> None:
        """覆盖会让"某个路由悄悄指向了另一个库"，而按名字调用的一方毫无察觉."""
        router = StoreRouter()
        first = make_retriever("alpha")
        router.register("alpha", first)

        with pytest.raises(QueryError) as excinfo:
            router.register("alpha", small_retriever("alpha"))

        message = str(excinfo.value)
        assert "已经被注册过" in message
        assert "幂等的拒绝" in message
        assert router.names() == ["alpha"]
        assert router.report()["routes"][0]["retriever"]["index"]["count_store"] == 8

    @pytest.mark.parametrize("name", ["", "   "])
    def test_empty_name_is_rejected(self, name: str) -> None:
        with pytest.raises(QueryError) as excinfo:
            StoreRouter().register(name, make_retriever())

        assert "name 必须是非空字符串" in str(excinfo.value)

    def test_value_must_be_a_retriever(self) -> None:
        """路由表管的是"去哪查"，查询逻辑本身在检索器里."""
        with pytest.raises(QueryError) as excinfo:
            StoreRouter().register("alpha", "不是检索器")  # type: ignore[arg-type]

        assert "必须是 Retriever，收到 str" in str(excinfo.value)

    def test_name_is_stripped(self) -> None:
        router = StoreRouter()

        router.register("  alpha  ", make_retriever("alpha"))

        assert router.names() == ["alpha"]
        assert "alpha" in router

    @pytest.mark.parametrize("name", ["alpha", 1, None])
    def test_membership_needs_a_string(self, name: object) -> None:
        """``in`` 对非字符串返回 False（而不是抛异常）：它常被写成条件判断."""
        router = two_route_router()

        assert (name in router) is (name == "alpha")

    def test_default_is_recorded_without_validation_at_construction(self) -> None:
        """``default`` 在构造期只记名字：装配顺序在动态场景里根本不固定."""
        router = StoreRouter(default="尚未注册")

        assert router.default == "尚未注册"
        assert router.names() == []


# --------------------------------------------------------------------------- #
# 选路的三条规则
# --------------------------------------------------------------------------- #


class TestResolve:
    """三条规则：默认 / 唯一的那个 / 必须显式指定."""

    def test_empty_router_raises_index_state_error(self) -> None:
        """空路由器不是"调用方写错了名字"，而是"索引还没装配好"（库侧的事）."""
        with pytest.raises(IndexStateError) as excinfo:
            StoreRouter().resolve(None)

        message = str(excinfo.value)
        assert "没有注册任何索引" in message
        assert "register(name, retriever)" in message
        assert "库侧的问题" in message

    def test_empty_router_raises_for_an_explicit_name_too(self) -> None:
        """即使给了名字，空路由器也先报"没装配好"（顺序：先看有没有东西可查）."""
        with pytest.raises(IndexStateError):
            StoreRouter().resolve("alpha")

    def test_default_route_is_used_when_route_is_missing(self) -> None:
        router = two_route_router(default="beta")

        retriever, decision = router.resolve(None)

        assert retriever.name == "beta"
        assert decision.name == "beta"
        assert decision.matched is False
        assert "按路由器的 default 选择" in decision.reason

    def test_single_index_does_not_need_a_route_name(self) -> None:
        """只有一个索引时不该强迫调用方写路由名（那是纯粹的样板）."""
        router = StoreRouter()
        retriever = make_retriever("only")
        router.register("only", retriever)

        resolved, decision = router.resolve(None)

        assert resolved is retriever
        assert decision.name == "only"
        assert decision.matched is False
        assert "只有一个索引" in decision.reason

    def test_multiple_indexes_require_an_explicit_route(self) -> None:
        """多个索引时"随便挑一个"会让"问源码库的问题去查了手册库"变成静默的错误答案."""
        router = two_route_router()

        with pytest.raises(QueryError) as excinfo:
            router.resolve(None)

        message = str(excinfo.value)
        assert "有 2 个索引但没有设置默认路由" in message
        assert "必须显式指定 route" in message
        assert "可用路由：alpha、beta" in message
        assert "StoreRouter(default=...)" in message

    def test_explicit_route_matches_the_registered_instance(self) -> None:
        router = StoreRouter()
        alpha = make_retriever("alpha")
        router.register("alpha", alpha)

        resolved, decision = router.resolve("alpha")

        assert resolved is alpha
        assert decision.matched is True
        assert decision.reason == "调用方显式指定的路由"

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_route_means_no_route(self, blank: str) -> None:
        """"没指定"只有一种表示，空串与空白串都归一成 ``None``."""
        router = StoreRouter()
        router.register("alpha", make_retriever("alpha"))

        _, decision = router.resolve(blank)

        assert decision.name == "alpha"
        assert decision.matched is False

    def test_unknown_route_lists_every_available_name(self) -> None:
        """三段式：现象 → 可用清单 → 出路（照着就能改对）."""
        router = two_route_router(default="alpha")

        with pytest.raises(QueryError) as excinfo:
            router.resolve("handbok")

        message = str(excinfo.value)
        assert "未知路由 'handbok'：本路由器没有这个名字的索引。" in message
        assert "可用路由：alpha、beta" in message
        assert "或者改用：不传 route（当前默认路由是 'alpha'）" in message

    def test_unknown_route_without_a_default_says_an_explicit_route_is_needed(self) -> None:
        router = two_route_router()

        with pytest.raises(QueryError) as excinfo:
            router.resolve("handbok")

        message = str(excinfo.value)
        assert "当前有 2 个索引且没有默认路由" in message
        assert "那时必须显式指定" in message

    def test_default_pointing_at_an_unregistered_name_is_reported(self) -> None:
        router = two_route_router(default="gamma")

        with pytest.raises(QueryError) as excinfo:
            router.resolve(None)

        message = str(excinfo.value)
        assert "默认路由 'gamma' 不在已注册的路由里" in message
        assert "可用路由：alpha、beta" in message
        assert "route='alpha'" in message

    @pytest.mark.parametrize("route", [1, 1.5, ["alpha"]])
    def test_route_must_be_a_string_or_none(self, route: Any) -> None:
        with pytest.raises(QueryError) as excinfo:
            two_route_router().resolve(route)

        assert "route 必须是字符串或 None" in str(excinfo.value)

    def test_one_of_many_routes_can_still_be_explicit(self) -> None:
        """有默认路由时显式指定仍然有效（``matched=True`` 与 ``False`` 是可区分的）."""
        router = two_route_router(default="beta")

        _, decision = router.resolve("alpha")

        assert decision.name == "alpha"
        assert decision.matched is True


# --------------------------------------------------------------------------- #
# 选路 + 检索
# --------------------------------------------------------------------------- #


class TestRetrieveThroughRouter:
    """``retrieve`` 把选路结论一路带到 ``notes`` 里（结果自己交代来自哪个路由）."""

    def test_route_note_is_appended_to_notes(self) -> None:
        router = two_route_router(default="alpha")

        result = router.retrieve(AXIS)

        assert result.count == 5
        assert result.notes[-1] == (
            "路由 alpha（路由器选定）：调用方没有指定 route，按路由器的 default 选择"
        )

    def test_result_comes_from_the_routed_index(self) -> None:
        """两路装的是不同规模的库：条数就是"结果来自哪一路"的直接证据."""
        router = two_route_router(default="alpha")

        alpha = router.retrieve(AXIS, route="alpha")
        beta = router.retrieve(AXIS, route="beta")

        assert alpha.count == 5
        assert beta.count == 2
        assert beta.ids() == list(RECORD_IDS[:2])
        assert beta.notes[-1].startswith("路由 beta（显式）")

    def test_route_argument_is_used_when_the_query_has_none(self) -> None:
        router = two_route_router(default="alpha")

        result = router.retrieve(AXIS, route="beta")

        assert result.count == 2

    def test_query_route_wins_over_the_argument(self) -> None:
        """一份 ``RetrievalQuery`` 可以被存盘再回放，那时"该去哪"是请求的一部分.

        反过来（参数赢）会让"回放同一份请求去了不同的库"。
        """
        router = two_route_router(default="alpha")

        result = router.retrieve(RetrievalQuery(text=AXIS, route="beta"), route="alpha")

        assert result.count == 2
        assert result.notes[-1].startswith("路由 beta（显式）")

    def test_existing_notes_survive_the_route_note(self) -> None:
        """漂移注记与路由注记必须**同时**出现（追加而不是替换）."""
        store = sample_store()
        manifest = sample_manifest(store)
        store.delete(ids=[RECORD_IDS[-1]])
        router = StoreRouter(default="alpha")
        router.register("alpha", Retriever(store, TableEmbedding(), manifest=manifest))

        result = router.retrieve(AXIS)

        assert any("索引漂移 1 项" in note for note in result.notes)
        assert result.notes[-1].startswith("路由 alpha")

    def test_unknown_route_through_retrieve_raises(self) -> None:
        with pytest.raises(QueryError):
            two_route_router(default="alpha").retrieve(AXIS, route="gamma")

    def test_router_rejects_when_no_index_is_registered(self) -> None:
        with pytest.raises(IndexStateError):
            StoreRouter().retrieve(AXIS)

    def test_result_stays_frozen_after_the_route_note(self) -> None:
        """``replace`` 而不是就地改：一次检索的交代不该在传递中被悄悄改写."""
        router = two_route_router(default="alpha")

        result = router.retrieve(AXIS)

        assert isinstance(result.notes, tuple)
        with pytest.raises(Exception):
            result.notes = ()  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #


class TestReport:
    """``/retrieval/routes`` 直接返回它：名字 + 说明 + 检索器现状."""

    def test_key_set(self) -> None:
        report = two_route_router(default="alpha").report()

        assert set(report) == {"count", "default", "names", "routes"}
        assert report["count"] == 2
        assert report["default"] == "alpha"
        assert report["names"] == ["alpha", "beta"]

    def test_each_route_carries_name_description_and_status(self) -> None:
        """三者缺一都会让"这一路为什么没结果"变得无法回答（漂移就在 index 里）."""
        report = two_route_router(default="alpha").report()

        first = report["routes"][0]

        assert set(first) == {"name", "description", "retriever"}
        assert first["name"] == "alpha"
        assert first["description"] == "手册库"
        assert set(first["retriever"]) == {
            "name",
            "top_k",
            "fetch_multiplier",
            "min_score",
            "max_per_doc",
            "doc_id_field",
            "time_field",
            "index",
        }
        assert first["retriever"]["index"]["count_store"] == 8
        assert report["routes"][1]["retriever"]["index"]["count_store"] == 2

    def test_routes_are_sorted_by_name(self) -> None:
        router = StoreRouter()
        router.register("zeta", make_retriever("zeta"))
        router.register("alpha", make_retriever("alpha"))

        assert [item["name"] for item in router.report()["routes"]] == ["alpha", "zeta"]

    def test_empty_description_is_an_empty_string(self) -> None:
        router = StoreRouter()
        router.register("alpha", make_retriever("alpha"))

        assert router.report()["routes"][0]["description"] == ""

    def test_empty_router_report(self) -> None:
        """空路由器的报告是"空"而不是"报错"（端点要能照常返回它）."""
        assert StoreRouter().report() == {
            "count": 0,
            "default": "",
            "names": [],
            "routes": [],
        }

    def test_report_shows_the_default_even_when_it_is_not_registered(self) -> None:
        """报告要照实回显配置（哪怕那个名字当时取不到）——它是排查的起点."""
        report = two_route_router(default="gamma").report()

        assert report["default"] == "gamma"
        assert report["names"] == ["alpha", "beta"]
