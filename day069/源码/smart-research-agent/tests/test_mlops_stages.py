"""day059 阶段表测试（M5-D10）：依赖自洽、主链、状态语义.

这个文件守的是"顺序即策略"从注释变成**可被程序校验的结构**这一步：

- ``validate_stage_order`` 逐条检查"每个 ``requires`` 都能在它之前的
  ``produces`` 里找到"。这条校验能抓住两类真实错误——**顺序写错**与
  **依赖漏写**——而后者平时不会报错，只在某次运行时冒出 ``KeyError``，
  堆栈还指在阶段内部而不是依赖表上；
- ``critical_path`` 给出主链。多个上游时的选择规则（取**最靠后**的那一个）
  也被钉住：``publish`` 同时依赖 train / gate / package，
  若取第一个上游，主链会变成 ``ingest → train → publish``，
  而它显然漏掉了评估与门禁。
"""

from __future__ import annotations

import pytest

from smart_research_agent.mlops import (
    ARTIFACT_ADAPTER,
    ARTIFACT_DATASET,
    ARTIFACT_GATE_REPORT,
    ARTIFACT_METRICS,
    ARTIFACT_MODEL_CARD,
    ARTIFACT_VERSION,
    PIPELINE_STAGES,
    STAGE_BLOCKED,
    STAGE_EVALUATE,
    STAGE_FAILED,
    STAGE_GATE,
    STAGE_INGEST,
    STAGE_OK,
    STAGE_PACKAGE,
    STAGE_PUBLISH,
    STAGE_SKIPPED,
    STAGE_SPECS,
    STAGE_STATUSES,
    STAGE_TRAIN,
    MLOpsError,
    StageResult,
    StageSpec,
    critical_path,
    stage_table,
    validate_stage_order,
)


# --------------------------------------------------------------------------- #
# 顺序与依赖
# --------------------------------------------------------------------------- #


def test_pipeline_stages_are_frozen_in_order() -> None:
    """六阶段的名字与顺序是这一课的核心产物，逐字钉住。"""
    assert PIPELINE_STAGES == (
        STAGE_INGEST,
        STAGE_TRAIN,
        STAGE_EVALUATE,
        STAGE_GATE,
        STAGE_PACKAGE,
        STAGE_PUBLISH,
    )


def test_default_specs_validate() -> None:
    """缺省依赖表自洽（不抛异常）。"""
    assert validate_stage_order() is None
    assert validate_stage_order(STAGE_SPECS) is None


def test_every_spec_produces_something() -> None:
    """每个阶段都要产出至少一样东西（一个不产出东西的阶段只是在消耗时间）。"""
    for spec in STAGE_SPECS:
        assert spec.produces, f"{spec.name} 没有产出"


def test_ingest_has_no_dependencies_and_starts_the_chain() -> None:
    """起点必须无依赖，否则流水线永远开不了头。"""
    assert STAGE_SPECS[0].name == STAGE_INGEST
    assert STAGE_SPECS[0].requires == ()


def test_publish_depends_on_adapter_card_and_gate_report() -> None:
    """发布同时依赖三样东西：适配器、模型卡、门禁报告（漏一样就不该发布）。"""
    publish = next(spec for spec in STAGE_SPECS if spec.name == STAGE_PUBLISH)
    assert set(publish.requires) == {
        ARTIFACT_ADAPTER,
        ARTIFACT_MODEL_CARD,
        ARTIFACT_GATE_REPORT,
    }


def test_gate_runs_before_publish_in_the_specs() -> None:
    """``gate`` 的位置在 ``publish`` 之前——顺序即策略，由索引断言。"""
    names = [spec.name for spec in STAGE_SPECS]
    assert names.index(STAGE_GATE) < names.index(STAGE_PUBLISH)
    assert names.index(STAGE_EVALUATE) < names.index(STAGE_GATE)


@pytest.mark.parametrize(
    "specs, match",
    [
        (
            (
                StageSpec(name="ingest", description="", produces=(ARTIFACT_DATASET,)),
                StageSpec(
                    name="publish",
                    description="顺序写错：publish 挪到了 train 之前",
                    requires=(ARTIFACT_ADAPTER,),
                    produces=(ARTIFACT_VERSION,),
                ),
            ),
            "尚未产出",
        ),
        (
            (
                StageSpec(name="ingest", description="", produces=(ARTIFACT_DATASET,)),
                StageSpec(
                    name="train",
                    description="重复产出同一个东西",
                    requires=(ARTIFACT_DATASET,),
                    produces=(ARTIFACT_DATASET,),
                ),
            ),
            "重复产出",
        ),
        (
            (
                StageSpec(name="ingest", description="", produces=(ARTIFACT_DATASET,)),
                StageSpec(
                    name="train",
                    description="依赖了一个没人产出的东西",
                    requires=("nobody_produces_this",),
                    produces=(ARTIFACT_ADAPTER,),
                ),
            ),
            "尚未产出",
        ),
    ],
)
def test_validate_rejects_broken_dependency_tables(specs: tuple, match: str) -> None:
    """三类坏表各自报错：顺序写错、重复产出、依赖没人产出。"""
    with pytest.raises(MLOpsError, match=match):
        validate_stage_order(specs)


def test_validate_rejects_duplicate_stage_names() -> None:
    """阶段名重复 → 报错。"""
    specs = (
        StageSpec(name="ingest", description="", produces=(ARTIFACT_DATASET,)),
        StageSpec(name="ingest", description="", produces=(ARTIFACT_METRICS,)),
    )
    with pytest.raises(MLOpsError, match="阶段名重复"):
        validate_stage_order(specs)


def test_validate_rejects_a_stage_without_production() -> None:
    """不产出东西的阶段被拒（它只是在消耗时间，且下游无从声明依赖）。

    这一条与"至少要有一个无依赖阶段"不同：后者是**恒为真的推论**
    （第一个阶段的 ``requires`` 只能落在空集里），刻意没写——
    一条永远为真的断言不是保护，而是噪声。
    """
    specs = (
        StageSpec(name="ingest", description="", produces=(ARTIFACT_DATASET,)),
        StageSpec(name="train", description="没有产出", requires=(ARTIFACT_DATASET,), produces=()),
    )
    with pytest.raises(MLOpsError, match="没有产出"):
        validate_stage_order(specs)


# --------------------------------------------------------------------------- #
# 主链
# --------------------------------------------------------------------------- #


def test_critical_path_follows_the_longest_dependency_chain() -> None:
    """主链是全部六步——**多上游时取最靠后的那一个**，否则会漏掉中间步骤。"""
    assert critical_path() == list(PIPELINE_STAGES)


def test_critical_path_uses_the_latest_upstream() -> None:
    """显式验证"取最靠后上游"这条规则：``publish`` 的上游里 ``package`` 最靠后."""
    specs = (
        StageSpec(name="ingest", description="", produces=(ARTIFACT_DATASET,)),
        StageSpec(name="train", description="", requires=(ARTIFACT_DATASET,), produces=(ARTIFACT_ADAPTER,)),
        StageSpec(
            name="evaluate",
            description="",
            requires=(ARTIFACT_ADAPTER,),
            produces=(ARTIFACT_METRICS,),
        ),
        StageSpec(
            name="gate",
            description="",
            requires=(ARTIFACT_METRICS, ARTIFACT_ADAPTER),
            produces=(ARTIFACT_GATE_REPORT,),
        ),
        StageSpec(
            name="package",
            description="",
            requires=(ARTIFACT_METRICS, ARTIFACT_GATE_REPORT),
            produces=(ARTIFACT_MODEL_CARD,),
        ),
        StageSpec(
            name="publish",
            description="",
            requires=(ARTIFACT_ADAPTER, ARTIFACT_MODEL_CARD, ARTIFACT_GATE_REPORT),
            produces=(ARTIFACT_VERSION,),
        ),
    )
    path = critical_path(specs)
    assert path == ["ingest", "train", "evaluate", "gate", "package", "publish"]
    assert path != ["ingest", "train", "publish"]  # 取第一个上游会得到这个错误答案


def test_critical_path_on_a_two_stage_table() -> None:
    """最小表：两阶段的主链就是它们自己。"""
    specs = (
        StageSpec(name="ingest", description="", produces=(ARTIFACT_DATASET,)),
        StageSpec(name="train", description="", requires=(ARTIFACT_DATASET,), produces=(ARTIFACT_ADAPTER,)),
    )
    assert critical_path(specs) == ["ingest", "train"]


def test_stage_table_is_ordered_and_complete() -> None:
    """表里的 ``order`` 从 1 开始连续，且携带依赖与产出。"""
    rows = stage_table()
    assert [row["order"] for row in rows] == [1, 2, 3, 4, 5, 6]
    assert [row["name"] for row in rows] == list(PIPELINE_STAGES)
    assert all(row["description"] and isinstance(row["requires"], list) for row in rows)


def test_stage_table_reflects_a_custom_spec_tuple() -> None:
    """``stage_table`` 可以接受自定义表（同一个渲染函数两处复用）。"""
    specs = (StageSpec(name="ingest", description="只有一步", produces=(ARTIFACT_DATASET,)),)
    rows = stage_table(specs)
    assert rows == [
        {
            "order": 1,
            "name": "ingest",
            "description": "只有一步",
            "requires": [],
            "produces": [ARTIFACT_DATASET],
            "blocking": True,
        }
    ]


# --------------------------------------------------------------------------- #
# StageSpec / StageResult
# --------------------------------------------------------------------------- #


def test_stage_spec_to_dict_converts_tuples_to_lists() -> None:
    """投影把元组转成列表（JSON 没有元组类型）。"""
    spec = StageSpec(name="train", description="d", requires=("a",), produces=("b",))
    payload = spec.to_dict()
    assert payload == {
        "name": "train",
        "description": "d",
        "requires": ["a"],
        "produces": ["b"],
        "blocking": True,
    }


@pytest.mark.parametrize("status", STAGE_STATUSES)
def test_stage_result_accepts_every_known_status(status: str) -> None:
    """四个状态都合法（``blocked`` 与 ``skipped`` 不是 ``failed``）。"""
    result = StageResult(name=STAGE_TRAIN, status=status)
    assert result.status == status
    assert result.ok is (status == STAGE_OK)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"name": "train", "status": "done"}, "未知阶段状态"),
        ({"name": "deploy", "status": STAGE_OK}, "未知阶段"),
    ],
)
def test_stage_result_validates_name_and_status(kwargs: dict, match: str) -> None:
    """名字与状态在构造期校验（写错的名字会让失败记在一个"不存在的阶段"上）。"""
    with pytest.raises(MLOpsError, match=match):
        StageResult(**kwargs)


def test_stage_result_status_categories() -> None:
    """``ok`` 只对 ``ok`` 为真；``blocked`` / ``skipped`` / ``failed`` 都不是 ok."""
    assert StageResult(name=STAGE_TRAIN, status=STAGE_OK).ok is True
    assert StageResult(name=STAGE_TRAIN, status=STAGE_BLOCKED).ok is False
    assert StageResult(name=STAGE_TRAIN, status=STAGE_SKIPPED).ok is False
    assert StageResult(name=STAGE_TRAIN, status=STAGE_FAILED).ok is False


def test_stage_result_projection_and_summary() -> None:
    """投影里带 ``ok`` 派生量；摘要含状态、名字与耗时。"""
    result = StageResult(
        name=STAGE_GATE,
        status=STAGE_OK,
        detail="通过",
        duration_ms=1.25,
        produced={"gate_report": {}},
    )
    payload = result.to_dict()
    assert payload["ok"] is True
    assert payload["produced"] == {"gate_report": {}}
    line = result.summary_line()
    assert STAGE_GATE in line and "1.25" in line and "通过" in line
