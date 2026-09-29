"""``training_optim.study``：四组对照实验（day081）."""

from __future__ import annotations

import json

import pytest

from smart_research_agent.training_optim import (
    CONTROL_CLIP,
    CONTROL_DROPOUT,
    CONTROL_GROUPS,
    CONTROL_GROUP_DESCRIPTIONS,
    CONTROL_INIT,
    CONTROL_SCHEDULE,
    INIT_SCHEMES,
    ParameterError,
    StudyRow,
    TrainingStudy,
    compare_clipping,
    compare_dropout,
    compare_initializations,
    compare_schedules,
    gain_profile_of,
    stack_of,
    training_study,
)
from tests.training_samples import (
    HIDDEN,
    STEPS,
    WARMUP_STEPS,
    config,
    inputs,
    params,
    shape,
    target,
)


@pytest.fixture(scope="module")
def study() -> TrainingStudy:
    """整份四组实验（**只跑一次**：它要跑 12 条曲线）。"""
    return training_study(shape(), inputs(), target(), config=config())


class TestStudyTables:
    """两张表的闭合。"""

    def test_four_groups_with_descriptions(self):
        """四组各有一条说明。"""
        assert CONTROL_GROUPS == (CONTROL_INIT, CONTROL_SCHEDULE, CONTROL_CLIP, CONTROL_DROPOUT)
        assert set(CONTROL_GROUP_DESCRIPTIONS) == set(CONTROL_GROUPS)

    def test_study_row_needs_a_known_group(self):
        """不认识的对照组当场报错。"""
        with pytest.raises(ParameterError):
            StudyRow(
                control="something", variant="x", steps=1, first_loss=1.0, best_loss=1.0,
                best_step=1, last_loss=1.0, improvement_ratio=1.0, diverged=False,
            )

    def test_study_row_validates_numbers(self):
        """步数、层号与非负读数都要合法。"""
        with pytest.raises(ParameterError):
            StudyRow(
                control=CONTROL_INIT, variant="x", steps=-1, first_loss=1.0, best_loss=1.0,
                best_step=1, last_loss=1.0, improvement_ratio=1.0, diverged=False,
            )
        with pytest.raises(ParameterError):
            StudyRow(
                control=CONTROL_INIT, variant="x", steps=1, first_loss=1.0, best_loss=1.0,
                best_step=0, last_loss=1.0, improvement_ratio=1.0, diverged=False,
            )


class TestStudyContent:
    """样本上的四组读数（真实数据）。"""

    def test_all_four_groups_are_covered(self, study: TrainingStudy):
        """四组齐备，且每组至少两行（一个旋钮的两种取值）。"""
        assert {row.control for row in study.rows} == set(CONTROL_GROUPS)
        for group in CONTROL_GROUPS:
            assert len(study.rows_of(group)) >= 2

    def test_the_baseline_improves(self, study: TrainingStudy):
        """四组里至少有多行"收敛"（否则这张表在说"什么都没发生"）。"""
        assert sum(1 for row in study.rows if row.settled) >= 4

    def test_kaiming_diverges_while_the_others_do_not(self, study: TrainingStudy):
        """**初始化那一落的结论**：kaiming 是唯一发散的那一组。

        它的初始增益最大（``gain_first ≈ 7.76``），而增益与发散直接相关：
        ``uniform`` 的 2.24、``xavier`` 的 3.50、``normal`` 的 3.56 都活了下来。
        """
        diverged = study.diverged_variants
        assert diverged == ("init:kaiming_uniform",)
        kaiming = study.row(CONTROL_INIT, "kaiming_uniform")
        assert kaiming.diverged
        assert "训练在第" in kaiming.failure
        assert kaiming.extra("gain_first") > 7.0
        for scheme in ("uniform", "xavier_uniform", "normal"):
            assert study.row(CONTROL_INIT, scheme).settled

    def test_schedule_rows_show_the_post_ln_platform(self, study: TrainingStudy):
        """**调度那一落的结论**：post-LN 的起点更差、平台更低、改善倍数更大。"""
        pre = study.row(CONTROL_SCHEDULE, "pre-LN constant")
        post = study.row(CONTROL_SCHEDULE, "post-LN constant")
        assert pre.first_loss == pytest.approx(0.247506, abs=1e-5)
        assert post.first_loss == pytest.approx(0.962255, abs=1e-5)
        assert post.best_loss < pre.best_loss
        assert post.improvement_ratio > pre.improvement_ratio * 5

    def test_warmup_does_not_rescue_anything_here(self, study: TrainingStudy):
        """**day079 的悬案**：在这条小链上热身没有救下任何东西（两组都收敛）。"""
        for placement in ("pre-LN", "post-LN"):
            plain = study.row(CONTROL_SCHEDULE, f"{placement} constant")
            warmed = study.row(CONTROL_SCHEDULE, f"{placement} warmup_cosine")
            assert not plain.diverged and not warmed.diverged
        assert study.row(CONTROL_SCHEDULE, "post-LN warmup_cosine").best_loss > study.row(
            CONTROL_SCHEDULE, "post-LN constant"
        ).best_loss

    def test_clipping_slows_it_down_here(self, study: TrainingStudy):
        """**裁剪那一落**：学习率合适时它只是拖慢（末损失更高、平均缩放 < 1）。"""
        plain = study.row(CONTROL_CLIP, "无裁剪")
        clipped = study.row(CONTROL_CLIP, "裁剪 0.5")
        assert plain.extra("mean_clip_scale") == 1.0
        assert clipped.extra("mean_clip_scale") < 1.0
        assert clipped.best_loss > plain.best_loss

    def test_dropout_raises_train_and_lowers_eval(self, study: TrainingStudy):
        """**Dropout 那一落**：训练损失起点更高、而"之差"为负（推理更损失低）。"""
        plain = study.row(CONTROL_DROPOUT, "rate 0.00")
        dropped = study.row(CONTROL_DROPOUT, "rate 0.20")
        assert dropped.first_loss > plain.first_loss
        assert dropped.extra("mean_gap") < 0.0
        assert dropped.extra("last_eval_loss") < dropped.last_loss

    def test_every_variant_has_the_same_step_budget(self, study: TrainingStudy):
        """没发散的每一行都跑满了计划的步数（否则"改善"没有可比性）。"""
        for row in study.rows:
            if not row.diverged:
                assert row.steps == STEPS

    def test_summary_and_table(self, study: TrainingStudy):
        """两种输出的形状。"""
        assert "四组实验 12 行" in study.summary_line()
        lines = study.table_lines()
        assert len(lines) == len(study.rows) + 2
        assert "取值" in lines[0]

    def test_it_is_serialisable(self, study: TrainingStudy):
        """整份表能 json.dumps。"""
        payload = study.to_dict()
        assert len(payload["rows"]) == len(study.rows)
        assert payload["diverged_variants"] == ["init:kaiming_uniform"]
        assert isinstance(json.dumps(payload), str)


class TestStudyValidation:
    """实验参数与总表的构造检查。"""

    def test_a_study_with_one_group_is_rejected(self):
        """缺组的总表 ⇒ ``ParameterError``（缺一组说不出"这个旋钮有没有用"）。"""
        row = StudyRow(
            control=CONTROL_INIT, variant="uniform", steps=1, first_loss=1.0, best_loss=1.0,
            best_step=1, last_loss=1.0, improvement_ratio=1.0, diverged=False,
        )
        with pytest.raises(ParameterError):
            TrainingStudy(rows=(row,))

    def test_an_empty_study_is_rejected(self):
        """空表没有意义。"""
        with pytest.raises(ParameterError):
            TrainingStudy(rows=())

    def test_unknown_group_lookup_is_rejected(self, study: TrainingStudy):
        """按不认识的组名查行 ⇒ 报错（而不是返回空）。"""
        with pytest.raises(ParameterError):
            study.rows_of("everything")
        with pytest.raises(ParameterError):
            study.row(CONTROL_INIT, "not-a-scheme")

    def test_warmup_must_be_shorter_than_the_budget(self):
        """热身步数必须小于总步数（否则它不是"热身+退火"）。"""
        with pytest.raises(ParameterError):
            compare_schedules(shape(), inputs(), target(), config=config(steps=4), warmup_steps=4)
        with pytest.raises(ParameterError):
            compare_schedules(shape(), inputs(), target(), config=config(), warmup_steps=0)


class TestSingleGroups:
    """四组各自也能单独跑（返回 tuple，不依赖总表）。"""

    def test_initializations_alone(self):
        """初始化组：四行，每行带着第一层 / 最后一层 / 最大增益。"""
        rows = compare_initializations(shape(), inputs(), target(), config=config(steps=3))
        assert [row.variant for row in rows] == list(INIT_SCHEMES)
        for row in rows:
            if not row.diverged:
                assert row.extra("gain_first") > 0.0
                assert row.extra("gain_last") > 0.0
                assert row.extra("gain_max") >= row.extra("gain_first")

    def test_schedules_alone(self):
        """调度组：两条摆放位置 × 两种调度（只改一个旋钮）。"""
        rows = compare_schedules(
            shape(), inputs(), target(), config=config(steps=6), warmup_steps=2
        )
        variants = [row.variant for row in rows]
        assert variants == [
            "pre-LN constant",
            "pre-LN warmup_cosine",
            "post-LN constant",
            "post-LN warmup_cosine",
        ]
        assert all(row.steps == 6 for row in rows)

    def test_clipping_alone(self):
        """裁剪组：无裁剪一行 + 每个阈值一行。"""
        rows = compare_clipping(
            shape(), inputs(), target(), config=config(steps=3), max_norms=(0.5, 0.2)
        )
        assert [row.variant for row in rows] == ["无裁剪", "裁剪 0.5", "裁剪 0.2"]
        assert rows[0].extra("mean_clip_scale") == 1.0

    def test_dropout_alone(self):
        """Dropout 组：两个概率各一行，且**开着的那一行起点更高**.

        注意这里只断言"起点更高"这一条**稳健**的读数：
        "训练/推理之差的符号"需要足够长的预算才稳定（3 步时参数还在初值附近，
        两个相的损失都很乱）——整份实验用的是 40 步，那一组才敢下符号的结论。
        """
        rows = compare_dropout(
            shape(), inputs(), target(), config=config(steps=3), rates=(0.0, 0.3)
        )
        assert [row.variant for row in rows] == ["rate 0.00", "rate 0.30"]
        assert rows[1].first_loss > rows[0].first_loss


class TestStudyHelpers:
    """两个小工具：逐层增益与按方案造参数。"""

    def test_gain_profile_of(self):
        """四种方案的逐层增益都是正数，且 kaiming 的第一层最大。"""
        profiles = {
            scheme: gain_profile_of(shape(), inputs(), scheme=scheme)
            for scheme in INIT_SCHEMES
        }
        for values in profiles.values():
            assert len(values) == 4
            assert all(value > 0.0 for value in values)
        assert profiles["kaiming_uniform"][0] > profiles["xavier_uniform"][0]

    def test_stack_of(self):
        """按方案造参数：与 ``initialize_parameters`` 同一条路径。"""
        assert stack_of(shape(), scheme="xavier_uniform", seed=7, scale=0.25) == params(
            scheme="xavier_uniform"
        )

    def test_stack_of_checks_the_scheme(self):
        """不认识的方案当场报错。"""
        with pytest.raises(ParameterError):
            stack_of(shape(), scheme="lora", seed=7, scale=0.25)

    def test_summary_line_marks_a_diverged_row(self, study: TrainingStudy):
        """发散行的摘要说的是"发散"而不是一串数字。"""
        row = study.row(CONTROL_INIT, "kaiming_uniform")
        assert "发散" in row.summary_line()
        assert row.to_dict()["diverged"] is True

    def test_extra_of_a_missing_key_is_nan(self):
        """取一个不存在的附加读数返回 ``nan``（"没测"与"测到 0"必须分开）。"""
        row = StudyRow(
            control=CONTROL_INIT, variant="uniform", steps=1, first_loss=1.0, best_loss=1.0,
            best_step=1, last_loss=1.0, improvement_ratio=1.0, diverged=False,
        )
        assert row.extra("gain_first") != row.extra("gain_first")  # nan != nan

    def test_shape_helper_of_the_samples(self):
        """样本形状与 day080 的口径一致（4 层 / d=6 / d_ff=24）。"""
        item = shape()
        assert (item.layers, item.hidden, item.ffn) == (4, HIDDEN, 24)
