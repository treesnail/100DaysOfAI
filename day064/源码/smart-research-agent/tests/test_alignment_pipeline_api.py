"""对齐流水线 + 两个对齐端点测试（day054 C 组）.

两块被测对象的性格不同，测试的关注点也不同：

- ``pipeline.py`` 是**顺序**：数据体检 → 按维度分层切分 → 起点自检 → DPO →
  过优化体检。这些顺序错了都不会报错，只会给出一个看起来合理的报告，
  所以本文件钉住的是每一步的**可观测后果**：参考模型哈希（这次对齐偏离的是
  哪一个基准）、偏好数据指纹（用的是哪一份数据）、起点自检（``loss`` 是否为
  ``ln 2``）、以及"当前收集量离统计上可分辨的目标还差多少"（945 − 12 = 933）。
- **起点自检的失败路径必须有**：一个永远不会失败的检查不是检查。这里用
  "先训几步、再拿训练后的 loss 手工构造 ``AlignmentOutcome``"的方式把它
  构造出来（``initial_check_passed is False``）。
- 两个端点走 ``TestClient``（进程内 ASGI 调用，不起真实服务）。
  ``GET /finetune/alignment/dimensions`` 不跑模型；``POST /finetune/alignment/run``
  会**真的跑一遍 DPO**，所以它的断言尽量挂在**一次请求**的响应上（模块内
  共享一次 POST），只有过训练那一档单独付一次算力（140 步，规格要求只写一个）。

实测基线（缺省参数）：偏好数据 12 条（5 维）、切分为训练 7 / 留出 5，
数据指纹 ``5c7cfb2241b63405``、参考模型哈希 ``0da214132a6f``，
起点 loss 恰为 ``ln 2``、留出准确率 ``0.0 → 0.8``（+0.8），
样本量目标 945 / 缺口 933，过优化体检 ``accuracy_saturated=True`` 而
``over_optimized=False``、``kl_exceeded=False``。
"""

from __future__ import annotations

import json
import math

import pytest
from fastapi.testclient import TestClient

from smart_research_agent.alignment import (
    ALIGNMENT_DIMENSIONS,
    DEFAULT_BETA,
    DEFAULT_LEARNING_RATE,
    SEED_PAIRS,
    ZERO_MARGIN_TOLERANCE,
    AlignmentOutcome,
    PreferencePair,
    dpo_step,
    model_hash,
    run_alignment,
    split_preferences,
)
from smart_research_agent.api.app import create_app
from smart_research_agent.finetune_eval.metrics import FinetuneEvalError
from smart_research_agent.sft.encoding import CharTokenizer
from smart_research_agent.sft.reference_model import ModelState, ReferenceSFTModel

PAIRS = list(SEED_PAIRS)
TRAIN, VALID = split_preferences(PAIRS)
TOKENIZER = CharTokenizer.from_texts([p.prompt + p.chosen + p.rejected for p in PAIRS])
REFERENCE = ReferenceSFTModel(TOKENIZER.vocab_size, seed=42)


def fresh_policy() -> ReferenceSFTModel:
    """初始策略 = 参考模型的副本（DPO 的起点）。"""
    return ReferenceSFTModel.from_state(REFERENCE.state_dict())


#: margin 为 0 时的 DPO loss（起点自检的参照值）.
LN2 = math.log(2.0)

#: 流水线与端点的实测读数（写死在断言里，改动实现会立刻显形）.
REFERENCE_HASH = "0da214132a6f"
FINGERPRINT = "5c7cfb2241b63405"
TOTAL_PAIRS = 12
TRAIN_SIZE = 7
VALID_SIZE = 5
BEFORE_ACCURACY = 0.0
AFTER_ACCURACY = 0.8
ACCURACY_GAIN = 0.8
DEFAULT_STEPS = 42
PER_DIMENSION_REQUIRED = 189
TOTAL_REQUIRED = 945
TOTAL_GAP = 933
DEFAULT_KL_FINAL = 0.044928325994291536
DEFAULT_VALID_MARGIN_FINAL = 0.07940482870061487
OVER_STEPS = 140
OVER_KL_FINAL = 2.427636334873461
OVER_VALID_MARGIN_FINAL = 0.908943477970405
DIMENSION_COUNTS = {
    "citation": 3,
    "format": 2,
    "conciseness": 2,
    "refusal": 2,
    "honesty": 3,
}

#: 只有 2 条偏好对（同一个维度）的最小数据集：用来验证 ``pairs`` 注入路径.
TINY_PAIRS = [
    PreferencePair(
        id="tiny-01",
        prompt="检索效果差时先调哪里？",
        chosen="结论：先看召回率。依据：召回不够时排序无从谈起。",
        rejected="先调重排序吧，重排序一般能涨点。",
        dimension="format",
    ),
    PreferencePair(
        id="tiny-02",
        prompt="给 RAG 上线列检查项。",
        chosen="结论：先量基线。依据：没有基线就没有回退判据。",
        rejected="直接上线看看效果，有问题再回滚。",
        dimension="format",
    ),
]


@pytest.fixture(scope="module")
def client() -> TestClient:
    """离线客户端：走 create_app 默认装配，进程内调用 ASGI 应用."""
    return TestClient(create_app(), raise_server_exceptions=False)


@pytest.fixture(scope="module")
def default_outcome() -> AlignmentOutcome:
    """缺省参数下的完整流水线产物（约 14 秒，模块内共用一次）."""
    return run_alignment()


@pytest.fixture(scope="module")
def one_epoch_outcome() -> AlignmentOutcome:
    """``epochs=1`` 的流水线产物（7 步：每轮 7 条训练对）."""
    return run_alignment(epochs=1)


@pytest.fixture(scope="module")
def injected_outcome() -> AlignmentOutcome:
    """注入 ``tokenizer`` / ``reference`` 的流水线产物（``epochs=1``）."""
    return run_alignment(epochs=1, tokenizer=TOKENIZER, reference=REFERENCE)


@pytest.fixture(scope="module")
def tiny_outcome() -> AlignmentOutcome:
    """只有 2 条偏好对的注入数据集（同维度、内容不同）."""
    return run_alignment(pairs=TINY_PAIRS)


@pytest.fixture(scope="module")
def over_trained_outcome() -> AlignmentOutcome:
    """过训练对照：``epochs=20, learning_rate=2.0``（140 步，约 47 秒）."""
    return run_alignment(epochs=20, learning_rate=2.0)


@pytest.fixture(scope="module")
def run_response(client: TestClient) -> dict:
    """``POST /finetune/alignment/run`` 空请求体的响应体（模块内共享一次）."""
    response = client.post("/finetune/alignment/run", json={})
    assert response.status_code == 200
    return response.json()


class TestModelHash:
    """``model_hash``：用**参数本身**做参考模型的标识，而不是用模型名."""

    def test_same_state_hashes_the_same(self):
        """同一个 ``ModelState`` 两次调用得到同一个哈希（可核对）。"""
        state = REFERENCE.state_dict()
        assert model_hash(state) == model_hash(state)

    def test_hash_is_twelve_hex_characters(self):
        """哈希是 ``sha256`` 的前 12 位十六进制（够短能贴进日志，够长不会撞）。"""
        digest = model_hash(REFERENCE.state_dict())
        assert len(digest) == 12
        assert all(character in "0123456789abcdef" for character in digest)

    def test_hash_matches_the_measured_value(self):
        """缺省参考模型（词表 460、seed 42）的实测哈希 ``0da214132a6f``.

        流水线报告里的 ``reference_hash`` 必须与它一致，否则"这次对齐偏离的是
        哪一个基准"就无从核对。
        """
        assert model_hash(REFERENCE.state_dict()) == REFERENCE_HASH

    def test_independently_built_model_hashes_the_same(self):
        """同参数重建的模型哈希相同（哈希只由参数决定，与对象身份无关）."""
        rebuilt = ReferenceSFTModel(TOKENIZER.vocab_size, seed=42)
        assert model_hash(rebuilt.state_dict()) == REFERENCE_HASH

    def test_different_seed_changes_the_hash(self):
        """换 seed 就是换一份参数 ⇒ 哈希必须变（否则这个标识没有信息量）."""
        other = ReferenceSFTModel(TOKENIZER.vocab_size, seed=7)
        assert model_hash(other.state_dict()) != REFERENCE_HASH

    def test_tiny_weight_change_changes_the_hash(self):
        """参数改一点点就不同：改动一个权重的最低有效位也算改."""
        state = REFERENCE.state_dict()
        weights = [list(row) for row in state.weights]
        weights[0][0] += 1e-12
        changed = ModelState(
            vocab_size=state.vocab_size,
            weights=tuple(tuple(row) for row in weights),
            bias=state.bias,
            updates=state.updates,
        )
        assert model_hash(changed) != model_hash(state)

    def test_bias_change_changes_the_hash(self):
        """偏置同样进哈希（只哈希权重会让 bias 漂移悄悄过关）."""
        state = REFERENCE.state_dict()
        bias = list(state.bias)
        bias[0] += 1e-12
        changed = ModelState(
            vocab_size=state.vocab_size,
            weights=state.weights,
            bias=tuple(bias),
            updates=state.updates,
        )
        assert model_hash(changed) != model_hash(state)

    def test_updates_counter_is_ignored(self):
        """``updates`` 不进哈希：**哈希标识的是参数，不是"训了多少步"**."""
        state = REFERENCE.state_dict()
        trained_marker = ModelState(
            vocab_size=state.vocab_size,
            weights=state.weights,
            bias=state.bias,
            updates=state.updates + 99,
        )
        assert model_hash(trained_marker) == model_hash(state)

    def test_vocab_size_is_part_of_the_payload(self):
        """词表大小进哈希：同样的数值、不同的词表是两份不同的模型."""
        flat = ModelState(
            vocab_size=3,
            weights=((0.0, 0.0, 0.0),) * 3,
            bias=(0.0, 0.0, 0.0),
            updates=0,
        )
        wider = ModelState(
            vocab_size=4,
            weights=((0.0,) * 4,) * 4,
            bias=(0.0,) * 4,
            updates=0,
        )
        assert model_hash(flat) != model_hash(wider)
        assert len(model_hash(flat)) == 12


class TestRunAlignmentDefault:
    """``run_alignment()`` 缺省路径：切分 → 起点自检 → DPO → 过优化体检."""

    def test_reference_hash_is_recorded(self, default_outcome):
        """报告里的参考模型哈希就是"参数的 sha256 前 12 位"."""
        assert default_outcome.reference_hash == REFERENCE_HASH
        assert default_outcome.reference_hash == model_hash(REFERENCE.state_dict())

    def test_initial_check_passes(self, default_outcome):
        """起点自检通过：训练之前 ``loss`` 恰好是 ``ln 2``."""
        assert default_outcome.initial_check_passed is True
        assert default_outcome.initial_loss == pytest.approx(LN2, abs=1e-12)
        assert default_outcome.zero_margin_loss == pytest.approx(LN2, abs=1e-12)

    def test_stats_fingerprint_matches_the_measured_value(self, default_outcome):
        """偏好数据指纹实测 ``5c7cfb2241b63405``（对齐结果必须能指回数据）."""
        assert default_outcome.stats["fingerprint"] == FINGERPRINT
        assert default_outcome.stats["total"] == TOTAL_PAIRS == 12

    def test_stats_dimension_counts(self, default_outcome):
        """维度分布与手写种子数据一致（citation 3、format 2、conciseness 2、refusal 2、honesty 3）."""
        assert default_outcome.stats["by_dimension"] == DIMENSION_COUNTS
        assert sum(DIMENSION_COUNTS.values()) == TOTAL_PAIRS

    def test_train_ids_match_the_split(self, default_outcome):
        """``train_ids`` 与 ``split_preferences`` 的结果逐位一致（顺序也一致）."""
        assert default_outcome.train_ids == [pair.id for pair in TRAIN]
        assert len(default_outcome.train_ids) == TRAIN_SIZE == 7

    def test_valid_ids_match_the_split(self, default_outcome):
        """``valid_ids`` 实测 5 条：``pref-cite-02 / fmt-02 / conc-01 / refuse-02 / honest-02``."""
        assert default_outcome.valid_ids == [
            "pref-cite-02",
            "pref-fmt-02",
            "pref-conc-01",
            "pref-refuse-02",
            "pref-honest-02",
        ]
        assert default_outcome.valid_ids == [pair.id for pair in VALID]

    def test_two_sides_partition_the_dataset(self, default_outcome):
        """训练 / 留出两侧互补、不重叠，且每个维度都留出了至少一条."""
        all_ids = [pair.id for pair in PAIRS]
        assert sorted(default_outcome.train_ids + default_outcome.valid_ids) == sorted(all_ids)
        assert not set(default_outcome.train_ids) & set(default_outcome.valid_ids)
        assert len(set(VALID[i].dimension for i in range(len(VALID)))) == len(
            ALIGNMENT_DIMENSIONS
        )

    def test_before_accuracy_is_zero(self, default_outcome):
        """训练前留出准确率 0.0：策略 = 参考模型，隐式奖励之差全是 0.

        判据是"隐式奖励之差 > 0"（``σ(β·margin) > 0.5``），margin 恒为 0 时
        一条都判不对——**这正是"未训练时排序信息为零"的可观测后果**。
        """
        assert default_outcome.before["accuracy"] == BEFORE_ACCURACY
        assert default_outcome.before["correct"] == 0
        assert default_outcome.before["total"] == VALID_SIZE
        assert all(not entry["correct"] for entry in default_outcome.before["pairs"])

    def test_after_accuracy_is_point_eight(self, default_outcome):
        """训练后留出准确率 0.8（5 条里 4 条判对），提升 0.8."""
        assert default_outcome.after["accuracy"] == AFTER_ACCURACY == 0.8
        assert default_outcome.after["correct"] == 4
        assert default_outcome.after["total"] == VALID_SIZE

    def test_accuracy_gain(self, default_outcome):
        """``accuracy_gain`` 就是"后 − 前"（同一份留出集上的两个读数）."""
        assert default_outcome.accuracy_gain == pytest.approx(ACCURACY_GAIN)
        assert default_outcome.accuracy_gain == pytest.approx(
            default_outcome.after["accuracy"] - default_outcome.before["accuracy"]
        )

    def test_plan_sample_budget_gap(self, default_outcome):
        """样本量算术：每维 189 条 → 5 维共 945 条，已收集 12 条，缺口 933.

        这个缺口是"能不能开工"的判据：``ready`` 必须为 False——12 条远小于
        945 条，此时放量等于放大标注噪声。
        """
        summary = default_outcome.plan["summary"]
        assert summary["required"] == TOTAL_REQUIRED == 945
        assert summary["collected"] == TOTAL_PAIRS == 12
        assert summary["gap"] == TOTAL_GAP == 933
        assert summary["dimensions"] == 5
        assert summary["ready_dimensions"] == 0
        assert summary["ready"] is False
        assert default_outcome.plan["per_dimension_required"] == PER_DIMENSION_REQUIRED == 189

    def test_plan_dpo_config_echoes_the_run(self, default_outcome):
        """计划里的 DPO 配置就是这次跑用的 β / 学习率 / KL 预算."""
        config = default_outcome.plan["dpo_config"]
        assert config["beta"] == DEFAULT_BETA == 0.1
        assert config["learning_rate"] == DEFAULT_LEARNING_RATE == 0.5
        assert config["valid_ratio"] == 0.25
        assert config["kl_budget"] == 0.5

    def test_plan_lists_every_dimension_and_risk(self, default_outcome):
        """计划里五个维度与四条风险都在（其中风险互相独立、各自带检测手段）."""
        assert [row["dimension"] for row in default_outcome.plan["dimensions"]] == list(
            ALIGNMENT_DIMENSIONS
        )
        assert [row["name"] for row in default_outcome.plan["risks"]] == [
            "长度偏置",
            "过优化",
            "维度混淆",
            "参考模型漂移",
        ]

    def test_optimization_three_judgements(self, default_outcome):
        """过优化体检的三条判据全部交出来：只有"准确率饱和"触发.

        - ``over_optimized`` False：留出 margin 末步仍是峰值，没有回落；
        - ``kl_exceeded`` False：末步 KL 0.0449 远低于 0.5 的预算；
        - ``accuracy_saturated`` True：留出准确率自第 2 步起不再创新高。
        """
        optimization = default_outcome.optimization
        assert optimization["over_optimized"] is False
        assert optimization["kl_exceeded"] is False
        assert optimization["accuracy_saturated"] is True

    def test_optimization_numbers(self, default_outcome):
        """体检里的数值与训练历史一致（末步 KL、留出 margin、峰值位置）."""
        optimization = default_outcome.optimization
        report = default_outcome.report
        assert optimization["steps"] == DEFAULT_STEPS == 42
        assert optimization["kl_final"] == pytest.approx(DEFAULT_KL_FINAL, rel=1e-6)
        assert optimization["valid_margin_final"] == pytest.approx(
            DEFAULT_VALID_MARGIN_FINAL, rel=1e-6
        )
        assert optimization["valid_margin_peak_step"] == 42
        assert optimization["valid_accuracy_best"] == AFTER_ACCURACY
        assert optimization["valid_accuracy_best_step"] == 2
        assert report is not None
        assert optimization["kl_final"] == pytest.approx(report.records[-1].kl, rel=1e-12)

    def test_optimization_flags_are_explicit(self, default_outcome):
        """告警清单只有一条（准确率饱和），且写清了"从第几步起不再创新高".

        空的清单是有歧义的：它既可能表示"没有过优化"，也可能表示"历史太短、
        什么都没测出来"。所以这里同时断言清单内容与三个布尔值。
        """
        flags = default_outcome.optimization["flags"]
        assert len(flags) == 1
        assert "不再创新高" in flags[0]
        assert "第 2 步" in flags[0]

    def test_report_echoes_the_configuration(self, default_outcome):
        """报告带上了这次训练的完整配置（β / 学习率 / 轮数 / 两个集合的条数）."""
        report = default_outcome.report
        assert report is not None
        assert report.beta == DEFAULT_BETA
        assert report.learning_rate == DEFAULT_LEARNING_RATE
        assert report.epochs == 6
        assert report.train_pairs == TRAIN_SIZE
        assert report.valid_pairs == VALID_SIZE
        assert report.zero_margin_loss == pytest.approx(LN2, abs=1e-12)

    def test_report_history_numbers(self, default_outcome):
        """历史 42 条、步号 1..42、起点 loss 为 ``ln 2``、最佳留出准确率 0.8@step 2."""
        report = default_outcome.report
        assert report is not None
        assert report.steps == DEFAULT_STEPS
        assert [record.step for record in report.records] == list(range(1, DEFAULT_STEPS + 1))
        assert report.initial_loss == pytest.approx(LN2, abs=1e-12)
        assert report.final_loss < report.initial_loss
        assert report.best_valid_accuracy == AFTER_ACCURACY
        assert report.best_step == 2

    def test_to_dict_keys(self, default_outcome):
        """``to_dict`` 十三个键齐全（含拍平的 ``initial_check_passed`` / ``accuracy_gain``）."""
        payload = default_outcome.to_dict()
        assert set(payload) == {
            "stats",
            "train_ids",
            "valid_ids",
            "plan",
            "reference_hash",
            "initial_loss",
            "zero_margin_loss",
            "initial_check_passed",
            "before",
            "after",
            "accuracy_gain",
            "report",
            "optimization",
        }
        assert payload["initial_check_passed"] is True
        assert payload["accuracy_gain"] == pytest.approx(ACCURACY_GAIN)
        assert payload["report"] is not None

    def test_to_dict_is_json_serializable(self, default_outcome):
        """流水线产物必须能直接 ``json.dumps``（它是端点的响应体来源）."""
        payload = default_outcome.to_dict()
        text = json.dumps(payload, ensure_ascii=False)
        assert json.loads(text) == payload
        assert len(payload["report"]["records"]) == DEFAULT_STEPS

    def test_summary_line_is_printable(self, default_outcome):
        """一行摘要里同时有数据指纹、参考模型哈希、两侧条数、起点自检与两个准确率."""
        line = default_outcome.summary_line()
        assert FINGERPRINT in line
        assert REFERENCE_HASH in line
        assert "7/5 对（训练/留出）" in line
        assert "起点自检 通过" in line
        assert "0.0000 → 0.8000（+0.8000）" in line
        assert "KL 0.044928" in line

    def test_before_readings_cover_every_dimension(self, default_outcome):
        """留出集按维度分层：5 条留出对恰好覆盖 5 个维度（每维 1 条）."""
        by_dimension = default_outcome.before["by_dimension"]
        assert sorted(by_dimension) == sorted(ALIGNMENT_DIMENSIONS)
        assert all(stats["total"] == 1 for stats in by_dimension.values())


class TestRunAlignmentVariants:
    """``epochs`` / ``learning_rate`` 的对照：7 步 vs 42 步 vs 140 步."""

    def test_one_epoch_runs_seven_steps(self, one_epoch_outcome):
        """``epochs=1`` ⇒ 7 步（每轮恰好走 7 条训练对）."""
        assert one_epoch_outcome.report is not None
        assert one_epoch_outcome.report.steps == TRAIN_SIZE == 7
        assert one_epoch_outcome.report.epochs == 1

    def test_one_epoch_keeps_the_gain_and_the_plan(self, one_epoch_outcome):
        """单轮也把留出准确率抬到 0.8，且样本量计划与轮数无关."""
        assert one_epoch_outcome.initial_check_passed is True
        assert one_epoch_outcome.before["accuracy"] == BEFORE_ACCURACY
        assert one_epoch_outcome.after["accuracy"] == AFTER_ACCURACY
        assert one_epoch_outcome.accuracy_gain == pytest.approx(ACCURACY_GAIN)
        assert one_epoch_outcome.plan["summary"]["required"] == TOTAL_REQUIRED

    def test_one_epoch_does_not_exceed_the_kl_budget(self, one_epoch_outcome):
        """7 步的 KL 只有 0.0015：远未超预算，也没有过优化."""
        optimization = one_epoch_outcome.optimization
        assert optimization["kl_exceeded"] is False
        assert optimization["over_optimized"] is False
        assert optimization["kl_final"] < 0.01

    def test_over_training_exceeds_the_kl_budget(self, over_trained_outcome):
        """``epochs=20, learning_rate=2.0`` ⇒ 140 步后 KL 超预算（``kl_exceeded``）.

        实测末步 KL 2.4276（预算 0.5 的 4.9 倍，默认配置末步的 54 倍），
        而留出准确率**仍然是 0.8**——"继续训练买到了什么"的答案是"只买到了 KL"。
        """
        assert over_trained_outcome.report is not None
        assert over_trained_outcome.report.steps == OVER_STEPS == 140
        optimization = over_trained_outcome.optimization
        assert optimization["kl_exceeded"] is True
        assert optimization["kl_final"] == pytest.approx(OVER_KL_FINAL, rel=1e-6)
        assert optimization["kl_final"] > 4 * 0.5

    def test_over_training_keeps_the_accuracy_and_the_margin(self, over_trained_outcome):
        """140 步后留出准确率没涨、留出 margin 涨到 0.91（且未回落）."""
        assert over_trained_outcome.after["accuracy"] == AFTER_ACCURACY
        assert over_trained_outcome.accuracy_gain == pytest.approx(ACCURACY_GAIN)
        optimization = over_trained_outcome.optimization
        assert optimization["accuracy_saturated"] is True
        assert optimization["over_optimized"] is False
        assert optimization["valid_margin_final"] == pytest.approx(
            OVER_VALID_MARGIN_FINAL, rel=1e-6
        )


class TestRunAlignmentInjection:
    """``pairs`` / ``tokenizer`` / ``reference`` 三条注入路径."""

    def test_two_pair_dataset_runs(self, tiny_outcome):
        """只有 2 条偏好对也能跑通：切分后训练 1 条 / 留出 1 条.

        逐维度配额是 ``min(n − 1, max(1, round(n·ratio)))``，n = 2 时两种
        ratio 都留出 1 条——所以"能跑通"这件事对 2 条数据也成立。
        """
        assert tiny_outcome.stats["total"] == 2
        assert tiny_outcome.train_ids == ["tiny-01"]
        assert tiny_outcome.valid_ids == ["tiny-02"]
        assert tiny_outcome.report is not None
        assert tiny_outcome.report.train_pairs == 1
        assert tiny_outcome.report.valid_pairs == 1
        assert tiny_outcome.report.steps == 6

    def test_two_pair_dataset_stats(self, tiny_outcome):
        """统计画像按注入的数据算（维度只有一个、指纹随之改变）."""
        assert tiny_outcome.stats["by_dimension"] == {"format": 2}
        assert tiny_outcome.stats["fingerprint"] != FINGERPRINT
        assert tiny_outcome.stats["near_duplicates"] == []
        assert tiny_outcome.stats["length_bias"]["total"] == 2

    def test_two_pair_dataset_reference_is_newly_built(self, tiny_outcome):
        """不注入参考模型时，流水线按注入语料现造词表与参考模型."""
        assert tiny_outcome.reference_hash != REFERENCE_HASH
        assert len(tiny_outcome.reference_hash) == 12

    def test_two_pair_dataset_accuracy_gain(self, tiny_outcome):
        """2 条数据也能看到提升（1 条留出对：0.0 → 1.0）——但**别把它当结论**."""
        assert tiny_outcome.before["accuracy"] == 0.0
        assert tiny_outcome.after["accuracy"] == 1.0
        assert tiny_outcome.accuracy_gain == pytest.approx(1.0)
        assert tiny_outcome.plan["summary"]["collected"] == 2
        assert tiny_outcome.plan["summary"]["gap"] == TOTAL_REQUIRED - 2

    def test_empty_pairs_raises(self):
        """空偏好数据在**统计阶段**就抛错，不等跑到切分或训练才失败."""
        with pytest.raises(FinetuneEvalError, match="不能对空偏好数据集做统计"):
            run_alignment(pairs=[])

    def test_injected_reference_keeps_its_hash(self, injected_outcome):
        """注入的参考模型哈希必须等于它的参数哈希（注入不得被悄悄替换）."""
        assert injected_outcome.reference_hash == REFERENCE_HASH
        assert injected_outcome.reference_hash == model_hash(REFERENCE.state_dict())

    def test_injected_tokenizer_keeps_the_split(self, injected_outcome):
        """注入 tokenizer / reference 不改变切分：``train_ids`` / ``valid_ids`` 与模块级一致."""
        assert injected_outcome.train_ids == [pair.id for pair in TRAIN]
        assert injected_outcome.valid_ids == [pair.id for pair in VALID]
        assert injected_outcome.stats["fingerprint"] == FINGERPRINT
        assert injected_outcome.report is not None
        assert injected_outcome.report.steps == TRAIN_SIZE


class TestInitialCheckFailurePath:
    """起点自检的**失败路径**：一个永远不会失败的检查不是检查."""

    def test_failed_initial_check_is_detected(self):
        """策略已经偏离参考模型时，``initial_loss != ln 2`` 且自检报 False.

        构造方式：先在训练集上走 3 步（``lr=1.0``），再用 ``lr=0`` 量一次
        ——此时的 loss 是"策略已经动过之后"的读数（实测 0.6897，偏离 ``ln 2``
        约 3.4e-3），拿它当 ``initial_loss`` 手工构造 ``AlignmentOutcome``。
        """
        policy = fresh_policy()
        for _ in range(3):
            dpo_step(
                policy,
                REFERENCE,
                TOKENIZER,
                TRAIN,
                beta=DEFAULT_BETA,
                learning_rate=1.0,
            )
        measured = dpo_step(
            policy, REFERENCE, TOKENIZER, TRAIN, beta=DEFAULT_BETA, learning_rate=0.0
        )
        outcome = AlignmentOutcome(
            stats={"fingerprint": FINGERPRINT},
            initial_loss=measured["mean_loss"],
        )
        assert measured["mean_loss"] != pytest.approx(LN2, abs=ZERO_MARGIN_TOLERANCE)
        assert outcome.initial_check_passed is False
        assert outcome.zero_margin_loss == pytest.approx(LN2, abs=1e-12)
        assert outcome.to_dict()["initial_check_passed"] is False

    def test_trained_policy_is_what_breaks_the_check(self):
        """同一个策略：更新前自检通过、更新后不通过（差别只在参数）."""
        policy = fresh_policy()
        before_update = dpo_step(
            policy, REFERENCE, TOKENIZER, TRAIN, beta=DEFAULT_BETA, learning_rate=0.0
        )
        passed = AlignmentOutcome(stats={}, initial_loss=before_update["mean_loss"])
        assert passed.initial_check_passed is True
        for _ in range(3):
            dpo_step(
                policy,
                REFERENCE,
                TOKENIZER,
                TRAIN,
                beta=DEFAULT_BETA,
                learning_rate=1.0,
            )
        after_update = dpo_step(
            policy, REFERENCE, TOKENIZER, TRAIN, beta=DEFAULT_BETA, learning_rate=0.0
        )
        failed = AlignmentOutcome(stats={}, initial_loss=after_update["mean_loss"])
        assert failed.initial_check_passed is False
        assert after_update["mean_loss"] != pytest.approx(LN2, abs=ZERO_MARGIN_TOLERANCE)
        assert after_update["mean_loss"] < before_update["mean_loss"]

    @pytest.mark.parametrize(
        ("deviation", "expected"),
        [
            (0.0, True),
            (1e-10, True),
            (-1e-10, True),
            (1e-8, False),
            (-1e-8, False),
            (1e-6, False),
        ],
    )
    def test_tolerance_boundary(self, deviation, expected):
        """容差是 ``1e-9``：浮点最后一位的差异不算接线错误，1e-8 就算."""
        outcome = AlignmentOutcome(stats={}, initial_loss=LN2 + deviation)
        assert outcome.initial_check_passed is expected

    def test_tolerance_constant(self):
        """容差常量写在 ``pipeline`` 里且很紧（比任何真实的接线错误小十几个数量级）."""
        assert ZERO_MARGIN_TOLERANCE == 1e-9
        assert ZERO_MARGIN_TOLERANCE > 0

    def test_accuracy_gain_reads_the_two_dicts(self):
        """``accuracy_gain`` 只依赖 ``before`` / ``after`` 两个字典（缺字段时按 0 算）."""
        outcome = AlignmentOutcome(
            stats={}, before={"accuracy": 0.2}, after={"accuracy": 0.6}
        )
        assert outcome.accuracy_gain == pytest.approx(0.4)
        assert AlignmentOutcome(stats={}).accuracy_gain == 0.0

    def test_summary_line_without_report_raises(self):
        """流水线没跑完（``report is None``）时摘要直接抛错，而不是打印半行数字."""
        with pytest.raises(FinetuneEvalError, match="流水线尚未完成"):
            AlignmentOutcome(stats={"fingerprint": FINGERPRINT}).summary_line()

    def test_defaults_are_documented(self):
        """``AlignmentOutcome`` 的缺省值自洽：``initial_loss=0.0`` 会被判自检失败."""
        outcome = AlignmentOutcome(stats={})
        assert outcome.initial_loss == 0.0
        assert outcome.initial_check_passed is False
        assert outcome.train_ids == []
        assert outcome.valid_ids == []
        assert outcome.plan == {}
        assert outcome.optimization == {}


class TestAlignmentDimensionsEndpoint:
    """GET /finetune/alignment/dimensions：对齐的"要买什么"（不跑模型）."""

    def test_returns_200_with_every_section(self, client):
        """200，且六个段落齐全（五个维度、样本量表、目标对照、风险、建议、种子画像）."""
        response = client.get("/finetune/alignment/dimensions")
        assert response.status_code == 200
        assert set(response.json()) == {
            "dimensions",
            "sample_budget",
            "objectives",
            "risks",
            "notes",
            "seed_stats",
        }

    def test_dimension_order_and_goals(self, client):
        """五个维度按固定顺序给出，每个都带一句人话目标."""
        dimensions = client.get("/finetune/alignment/dimensions").json()["dimensions"]
        assert [row["dimension"] for row in dimensions] == [
            "citation",
            "format",
            "conciseness",
            "refusal",
            "honesty",
        ]
        assert len(dimensions) == 5
        assert all(row["goal"] for row in dimensions)
        assert all(set(row) == {"dimension", "goal"} for row in dimensions)

    def test_sample_budget_table(self, client):
        """样本量表：55% → 778 条、60% → 189 条、65% → 80 条、70% → 42 条.

        这张表本身就是结论：**"提升 5 个百分点"要比"提升 20 个百分点"多花
        一个数量级的标注量**（778 / 42 ≈ 18.5 倍）。
        """
        budget = client.get("/finetune/alignment/dimensions").json()["sample_budget"]
        assert budget == [
            {"target_accuracy": 0.55, "pairs_per_dimension": 778},
            {"target_accuracy": 0.6, "pairs_per_dimension": 189},
            {"target_accuracy": 0.65, "pairs_per_dimension": 80},
            {"target_accuracy": 0.7, "pairs_per_dimension": 42},
        ]
        assert budget[0]["pairs_per_dimension"] > 10 * budget[-1]["pairs_per_dimension"]

    def test_objectives_rows(self, client):
        """两个目标的对照表：RLHF（PPO）三阶段 + 在线采样，DPO 一阶段 + 离线."""
        objectives = client.get("/finetune/alignment/dimensions").json()["objectives"]
        assert [row["objective"] for row in objectives] == ["RLHF（PPO）", "DPO"]
        assert len(objectives) == 2
        for row in objectives:
            assert set(row) == {
                "objective",
                "stages",
                "stage_detail",
                "models_needed",
                "preference_usage",
                "online_sampling",
                "kl_penalty",
                "main_failure",
                "signal_reusable",
            }
        assert objectives[0]["stages"] == 3
        assert objectives[0]["online_sampling"] is True
        assert objectives[1]["stages"] == 1
        assert objectives[1]["online_sampling"] is False
        assert objectives[1]["models_needed"] == ["policy", "reference"]

    def test_risk_rows(self, client):
        """四条风险：每条都带症状、缓解手段与**用哪个函数检测**."""
        risks = client.get("/finetune/alignment/dimensions").json()["risks"]
        assert [row["name"] for row in risks] == [
            "长度偏置",
            "过优化",
            "维度混淆",
            "参考模型漂移",
        ]
        assert len(risks) == 4
        for row in risks:
            assert set(row) == {"name", "symptom", "mitigation", "detect"}
            assert all(row[key] for key in row)

    def test_notes(self, client):
        """开工前的五条流程建议（每条都是可执行的，不是口号）."""
        notes = client.get("/finetune/alignment/dimensions").json()["notes"]
        assert len(notes) == 5
        assert any("冻结参考模型" in note for note in notes)
        assert any("κ" in note for note in notes)

    def test_seed_stats(self, client):
        """种子偏好数据的统计画像：指纹、条数与维度分布都与模块级一致."""
        stats = client.get("/finetune/alignment/dimensions").json()["seed_stats"]
        assert stats["fingerprint"] == FINGERPRINT
        assert stats["total"] == TOTAL_PAIRS
        assert stats["by_dimension"] == DIMENSION_COUNTS


class TestAlignmentRunEndpoint:
    """POST /finetune/alignment/run：跑一次最小 DPO 对齐."""

    def test_empty_body_returns_200(self, run_response):
        """**空请求体也能 200**：不传参数就能走完整流程（缺省值都是标定过的）."""
        assert run_response["reference_hash"] == REFERENCE_HASH
        assert run_response["initial_check_passed"] is True

    def test_initial_and_zero_margin_loss(self, run_response):
        """响应里的起点 loss 与参照值都是 ``ln 2``（0.693147）."""
        assert run_response["initial_loss"] == pytest.approx(LN2, abs=1e-12)
        assert run_response["zero_margin_loss"] == pytest.approx(LN2, abs=1e-12)

    def test_split_sizes(self, run_response):
        """响应里的切分是 7 / 5，与 ``split_preferences`` 的结果一致."""
        assert len(run_response["train_ids"]) == TRAIN_SIZE == 7
        assert len(run_response["valid_ids"]) == VALID_SIZE == 5
        assert run_response["train_ids"] == [pair.id for pair in TRAIN]
        assert run_response["valid_ids"] == [pair.id for pair in VALID]

    def test_accuracies_and_gain(self, run_response):
        """留出准确率 ``0.0 → 0.8``，提升 0.8（端点的三个数字必须自洽）."""
        assert run_response["before"]["accuracy"] == BEFORE_ACCURACY
        assert run_response["after"]["accuracy"] == AFTER_ACCURACY
        assert run_response["accuracy_gain"] == pytest.approx(ACCURACY_GAIN)

    def test_report_history(self, run_response):
        """响应里的训练报告含 42 步逐步历史（不是只有一个最终 loss）."""
        report = run_response["report"]
        assert report["steps"] == DEFAULT_STEPS == 42
        assert len(report["records"]) == DEFAULT_STEPS
        assert report["best_valid_accuracy"] == AFTER_ACCURACY
        assert report["best_step"] == 2
        assert set(report["records"][0]) == {
            "step",
            "mean_margin",
            "mean_loss",
            "valid_accuracy",
            "valid_margin",
            "kl",
            "learning_rate",
        }

    def test_plan_summary(self, run_response):
        """响应里的计划带样本量算术（945 / 933，``ready`` 为 False）."""
        summary = run_response["plan"]["summary"]
        assert summary["required"] == TOTAL_REQUIRED
        assert summary["gap"] == TOTAL_GAP
        assert summary["ready"] is False

    def test_optimization_judgements(self, run_response):
        """过优化体检的三个布尔值都交出来（空清单的歧义必须被消除）."""
        optimization = run_response["optimization"]
        assert optimization["accuracy_saturated"] is True
        assert optimization["over_optimized"] is False
        assert optimization["kl_exceeded"] is False
        assert optimization["kl_final"] == pytest.approx(DEFAULT_KL_FINAL, rel=1e-6)

    def test_stats_fingerprint(self, run_response):
        """响应里的数据指纹与维度分布（"这次对齐用的是哪一份数据"）."""
        assert run_response["stats"]["fingerprint"] == FINGERPRINT
        assert run_response["stats"]["by_dimension"] == DIMENSION_COUNTS

    def test_response_keys(self, run_response):
        """响应体的十三个键齐全（比 ``to_dict`` 少一个：没有 ``failed_gates``）."""
        assert set(run_response) == {
            "stats",
            "train_ids",
            "valid_ids",
            "plan",
            "reference_hash",
            "initial_loss",
            "zero_margin_loss",
            "initial_check_passed",
            "before",
            "after",
            "accuracy_gain",
            "report",
            "optimization",
        }
        assert "failed_gates" not in run_response

    def test_before_pairs_are_all_wrong(self, run_response):
        """起点上 5 条留出对一条都没判对（隐式奖励之差全为 0）."""
        pairs = run_response["before"]["pairs"]
        assert len(pairs) == VALID_SIZE
        assert all(entry["correct"] is False for entry in pairs)
        assert all(entry["margin"] == 0.0 for entry in pairs)

    @pytest.mark.parametrize(
        "body",
        [
            {"beta": 0.0},
            {"beta": 11.0},
            {"epochs": 0},
            {"epochs": 201},
            {"valid_ratio": 0.04},
            {"valid_ratio": 0.9},
            {"target_accuracy": 0.5},
            {"target_accuracy": 1.0},
            {"learning_rate": 0.0},
            {"learning_rate": 51.0},
            {"kl_budget": 0.0},
            {"seed": -1},
        ],
    )
    def test_out_of_range_fields_return_422(self, client, body):
        """取值越界由 Pydantic 在进路由前拦下 → **422**（不是 400）."""
        assert client.post("/finetune/alignment/run", json=body).status_code == 422

    def test_legal_request_with_small_valid_ratio_returns_200(self, client):
        """``{"epochs": 1, "valid_ratio": 0.05}`` 是**合法请求** → 200 且只走 7 步.

        它是 400 与 422 之间的分界样本：``valid_ratio`` 取下边界（0.05）时
        每个维度仍然留得出 1 条，所以流水线必须跑通。
        """
        response = client.post(
            "/finetune/alignment/run", json={"epochs": 1, "valid_ratio": 0.05}
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["report"]["steps"] == TRAIN_SIZE
        assert payload["initial_check_passed"] is True

    def test_over_training_path(self, client):
        """``{"learning_rate": 2.0, "epochs": 20}`` → 200 但 ``kl_exceeded`` 为 True.

        这一档会真的跑 140 步（约 47 秒），所以只写一条：端点与函数两条路径
        的过训练结论必须一致（另一条在 ``TestRunAlignmentVariants`` 里）。
        """
        response = client.post(
            "/finetune/alignment/run", json={"learning_rate": 2.0, "epochs": 20}
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["report"]["steps"] == OVER_STEPS
        assert payload["optimization"]["kl_exceeded"] is True
        assert payload["optimization"]["kl_final"] == pytest.approx(OVER_KL_FINAL, rel=1e-6)
        assert payload["optimization"]["accuracy_saturated"] is True
        assert payload["after"]["accuracy"] == AFTER_ACCURACY
        assert payload["accuracy_gain"] == pytest.approx(ACCURACY_GAIN)

    def test_response_matches_the_function_pipeline(self, run_response, default_outcome):
        """端点响应与直接调用 ``run_alignment()`` 逐字段同源（端点只是投影）."""
        payload = default_outcome.to_dict()
        assert run_response["reference_hash"] == payload["reference_hash"]
        assert run_response["stats"]["fingerprint"] == payload["stats"]["fingerprint"]
        assert run_response["train_ids"] == payload["train_ids"]
        assert run_response["plan"]["summary"] == payload["plan"]["summary"]
