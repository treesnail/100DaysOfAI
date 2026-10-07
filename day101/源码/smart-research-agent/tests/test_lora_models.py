"""LoRA 参考模型测试（day051）：适配器挂到基座之后的**全部不变式**.

本文件里承担"证明结论"角色的三条用例：

1. :meth:`TestInitMatchesBase.test_logits_are_bitwise_equal_to_base` ——
   ``ΔW = 0`` 时（刚挂上适配器、还没训练）``LoRAReferenceModel.logits(ctx)``
   必须与基座的 ``base.logits(ctx)`` **逐位相等**，且对**全部**词表下标都
   成立。用 ``==`` 逐元素比较而不是 ``approx``：这条断言守的是"行约定与
   列约定没有混用"，而混用之后 loss 曲线完全正常（照样下降、照样收敛），
   只有把两份 logits 摆在一起才发现对不上。
2. :meth:`TestTrainingUpdates.test_base_weights_untouched` —— 一次更新只
   动 ``A`` / ``B``，基座权重**逐位不变**。"参数高效"的全部含义就在这条。
3. :meth:`TestMergeAndState.test_merged_logits_are_bitwise_equal` —— 合并
   之后的普通模型与带适配器的模型在全部词表下标上输出相同，这是 LoRA
   部署路径（合并 → 落盘 → 推理）能被信任的前提。

最后一条用例把 ``LoRAReferenceModel`` 交给 day050 的 ``SFTTrainer`` 跑一遍
``fit``：训练循环一行不改就能接上 LoRA，是本课划定的那条边界。
"""

from __future__ import annotations

import pytest

from smart_research_agent.finetune.schema import TrainingExample
from smart_research_agent.peft.config import LoRAConfig, PEFTConfigError
from smart_research_agent.peft.models import (
    REFERENCE_LORA_ALPHA,
    REFERENCE_LORA_RANK,
    LoRAReferenceModel,
    default_reference_lora_config,
    reference_lora_accounting,
    train_lora_reference,
)
from smart_research_agent.sft import (
    IGNORE_INDEX,
    PLAIN,
    REFERENCE_LEARNING_RATE,
    Batch,
    CharTokenizer,
    ModelState,
    ReferenceSFTModel,
    SFTLossError,
    SFTTrainer,
    SFTTrainingArgs,
    render_supervised,
)

#: 参考模型的词表大小（课程数据集实测值）；r=8 时 8912 / 319718 = 2.787%
REFERENCE_VOCAB = 557

#: 训练用的学习率：与 day050 参考模型同一条标定（557 维 bigram + 纯 SGD）
REFERENCE_LR = REFERENCE_LEARNING_RATE

#: 多轮轨迹用例的学习率。:func:`make_batch` 的数据刻意做得很"集中"（只有 3 个
#: 不同上下文、32 个监督位置），因此同一行的步长是 ``0.375 × lr``——实测
#: ``lr = 8.0`` 会在第二个更新窗口之后发散（loss 6.32 → 3.4e3），而 ``2.0``
#: 稳定收敛（6.32 → 4.3e-3）。**这不是实现缺陷，而是"步长要看每行摊到多少
#: 位置"这条纪律的一个实例**（day050 的学习率标定用的是一千多个监督位置）。
TRAJECTORY_LR = 2.0


def build_base(vocab_size: int = REFERENCE_VOCAB, *, seed: int = 42) -> ReferenceSFTModel:
    """构造基座（``V × V`` 的 bigram 参考模型）."""
    return ReferenceSFTModel(vocab_size, seed=seed)


def build_model(base: ReferenceSFTModel, **overrides) -> LoRAReferenceModel:
    """在基座上挂一个默认的参考适配器（``dropout=0`` / ``targets=bigram``）."""
    return LoRAReferenceModel(base, default_reference_lora_config(**overrides), seed=42)


def make_batch(*, rows: int = 4) -> Batch:
    """构造一个带重复结构的小批次（监督位置多，loss 下降看得见）.

    序列是 ``3,5,7`` 的重复：上下文 3 的下一个 token 恒为 5、5 的下一个恒为
    7、7 的下一个恒为 3——一份**可学**的 bigram 数据，不是随机噪声。
    """
    pattern = (3, 5, 7, 3, 5, 7, 3, 5, 7)
    input_ids = tuple(pattern for _ in range(rows))
    labels = tuple((IGNORE_INDEX, *pattern[1:]) for _ in range(rows))
    masks = tuple(tuple([1] * len(pattern)) for _ in range(rows))
    return Batch(
        input_ids=input_ids,
        labels=labels,
        attention_mask=masks,
        pad_token_id=0,
    )


def make_examples(count: int = 4) -> list[TrainingExample]:
    """合成一批小而重复的样本（供与 day050 训练循环的兼容性用例使用）."""
    return [
        TrainingExample(
            instruction=f"请解释第 {index} 个概念",
            output="它把复杂问题拆成可以逐步核对的步骤，因此更容易验证。",
        )
        for index in range(count)
    ]


def build_tokenizer(examples: list[TrainingExample]) -> CharTokenizer:
    """用样本渲染后的文本建词表（与 day050 测试同一套做法）."""
    texts = [
        render_supervised(example, template=PLAIN, system_prompt=None).text
        for example in examples
    ]
    return CharTokenizer.from_texts(texts)


@pytest.fixture
def base() -> ReferenceSFTModel:
    """一个 ``V=557`` 的参考基座（每个用例新建一份，避免状态泄漏）."""
    return build_base()


@pytest.fixture
def model(base: ReferenceSFTModel) -> LoRAReferenceModel:
    """挂上默认适配器的参考模型（此时 ``ΔW`` 恒为 0）."""
    return build_model(base)


@pytest.fixture
def batches() -> list[Batch]:
    """训练与评估共用的一批小批次（两个 batch / 4 条与 2 条样本）."""
    return [make_batch(rows=4), make_batch(rows=2)]


class TestConstructionConstraints:
    """构造约束：参考模型只接受 ``dropout=0`` 且 ``target_modules`` 命中 bigram."""

    def test_default_config_values(self):
        config = default_reference_lora_config()
        assert REFERENCE_LORA_RANK == 8
        assert REFERENCE_LORA_ALPHA == 16
        assert (config.r, config.lora_alpha, config.lora_dropout) == (8, 16, 0.0)
        assert config.resolved_targets == ("weight",)

    def test_overrides_do_not_touch_defaults(self):
        config = default_reference_lora_config(r=4)
        assert (config.r, config.lora_alpha) == (4, 16)
        assert default_reference_lora_config().r == 8

    def test_non_zero_dropout_rejected(self, base):
        config = LoRAConfig(lora_dropout=0.05, target_modules="bigram")
        with pytest.raises(PEFTConfigError, match="lora_dropout = 0"):
            LoRAReferenceModel(base, config)

    def test_preset_targets_must_hit_bigram(self, base):
        config = LoRAConfig(lora_dropout=0.0, target_modules="attention")
        with pytest.raises(PEFTConfigError, match="target_modules"):
            LoRAReferenceModel(base, config)

    def test_custom_targets_must_be_the_weight_matrix(self, base):
        config = LoRAConfig(lora_dropout=0.0, target_modules=("q_proj",))
        with pytest.raises(PEFTConfigError, match="target_modules"):
            LoRAReferenceModel(base, config)

    def test_from_base_matches_constructor(self, base):
        direct = LoRAReferenceModel(base, default_reference_lora_config(), seed=42)
        factory = LoRAReferenceModel.from_base(base, default_reference_lora_config(), seed=42)
        assert factory.logits(11) == direct.logits(11)
        assert factory.vocab_size == direct.vocab_size


class TestInitMatchesBase:
    """**核心不变式**：``ΔW = 0`` 时适配器模型与基座逐位一致."""

    def test_delta_is_zero_at_init(self, model):
        assert model.delta_is_zero() is True
        assert model.updates == 0

    def test_logits_are_bitwise_equal_to_base(self, model, base):
        """对**全部**词表下标逐元素比较（``==``，不是 ``approx``）."""
        for context in range(base.vocab_size):
            assert model.logits(context) == base.logits(context)

    def test_predict_next_matches_base(self, model, base):
        for context in (0, 1, 2, REFERENCE_VOCAB - 1):
            assert model.predict_next(context) == base.predict_next(context)

    def test_vocab_and_uniform_loss_match_base(self, model, base):
        assert model.vocab_size == base.vocab_size
        assert model.uniform_loss == pytest.approx(base.uniform_loss)

    def test_logits_reject_out_of_range_context(self, model):
        for context in (-1, REFERENCE_VOCAB):
            with pytest.raises(Exception, match="上下文 token"):
                model.logits(context)


class TestEvaluateMatchesBase:
    """评估口径与基座一致：未训练时两条曲线必须完全重合."""

    def test_evaluate_equals_base(self, model, base, batches):
        assert model.evaluate(batches) == base.evaluate(batches)

    def test_evaluate_counts_supervised_tokens(self, model, batches):
        loss, count = model.evaluate(batches)
        assert count == sum(batch.supervised_tokens for batch in batches)
        assert loss == pytest.approx(model.uniform_loss, abs=0.05)


class TestTrainingUpdates:
    """一次真实更新之后的五条事实：loss 降、基座不动、增量非零、计数正确、可复现."""

    def test_loss_decreases(self, model, batches):
        before, _ = model.evaluate(batches)
        for batch in batches:
            model.accumulate(batch)
        model.apply_update(REFERENCE_LR)
        after, _ = model.evaluate(batches)
        assert after < before

    def test_base_weights_untouched(self, base, batches):
        model = build_model(base)
        weights_before = base.state_dict().weights
        for batch in batches:
            model.accumulate(batch)
        model.apply_update(REFERENCE_LR)
        assert base.state_dict().weights == weights_before
        assert base.state_dict().updates == 0
        assert model.delta_is_zero() is False

    def test_delta_becomes_non_zero(self, model, batches):
        assert model.delta_is_zero() is True
        model.accumulate(batches[0])
        model.apply_update(REFERENCE_LR)
        assert model.delta_is_zero() is False
        assert model.layer.describe()["max_abs"] > 0.0

    def test_updates_counter_and_pending(self, model, batches):
        for step, batch in enumerate(batches):
            assert model.layer.pending_positions == 0
            model.accumulate(batch)
            # 待应用计数是**监督位置数**（梯度就是按位置累加的），不是批数
            assert model.layer.pending_positions == batch.supervised_tokens
            model.apply_update(REFERENCE_LR)
            assert model.updates == step + 1
            assert model.layer.pending_positions == 0

    def test_same_seed_gives_identical_loss_sequence(self, base, batches):
        """同一个 seed 两次训练：``loss`` 序列逐位相同（可复现是硬要求）."""

        def run() -> list[float]:
            candidate = build_model(base)
            losses: list[float] = []
            for _ in range(2):
                for batch in batches:
                    loss_sum, count = candidate.accumulate(batch)
                    candidate.apply_update(REFERENCE_LR)
                    losses.append(loss_sum / count)
            return losses

        assert run() == run()


class TestGradientDenominator:
    """分母纪律：``apply_update`` 之前必须有 ``accumulate``；``zero_grad`` 丢弃待应用梯度."""

    def test_apply_update_requires_accumulate(self, model):
        with pytest.raises(PEFTConfigError, match="没有待应用的梯度"):
            model.apply_update(REFERENCE_LR)

    def test_apply_update_rejects_non_positive_lr(self, model, batches):
        model.accumulate(batches[0])
        with pytest.raises(PEFTConfigError, match="learning_rate"):
            model.apply_update(0.0)

    def test_zero_grad_clears_pending(self, model, batches):
        model.accumulate(batches[0])
        assert model.layer.pending_positions == batches[0].supervised_tokens
        model.zero_grad()
        assert model.layer.pending_positions == 0
        assert model.delta_is_zero() is True
        with pytest.raises(PEFTConfigError, match="没有待应用的梯度"):
            model.apply_update(REFERENCE_LR)

    def test_denominator_counts_positions_not_batches(self, model, batches):
        """分母是累积窗口里的**位置数**：两个 batch 一起更新只算**一次**更新."""
        model.accumulate(batches[0])
        model.accumulate(batches[1])
        expected = batches[0].supervised_tokens + batches[1].supervised_tokens
        assert model.layer.pending_positions == expected
        model.apply_update(REFERENCE_LR)
        assert model.updates == 1


class TestEvaluateKeepsGradients:
    """评估只前向：不能污染梯度缓冲（否则"训练 — 评估 — 继续训练"会串味）."""

    def test_gradient_snapshot_is_unchanged(self, model, batches):
        model.accumulate(batches[0])
        snapshot = model.layer.gradient_snapshot()
        model.evaluate(batches)
        assert model.layer.gradient_snapshot() == snapshot

    def test_accumulated_gradients_still_usable(self, model, batches):
        model.accumulate(batches[0])
        model.evaluate(batches)
        model.apply_update(REFERENCE_LR)
        assert model.updates == 1

    def test_weights_are_unchanged_by_evaluation(self, model, batches):
        before = model.layer.snapshot_matrices()
        model.evaluate(batches)
        assert model.layer.snapshot_matrices() == before

    def test_empty_batch_list_rejected(self, model):
        with pytest.raises(Exception, match="评估集没有任何监督位置"):
            model.evaluate([])


class TestMergeAndState:
    """合并：适配器消失，但输出必须与带适配器的模型**逐位一致**."""

    @pytest.fixture
    def trained(self, base, batches) -> LoRAReferenceModel:
        """训练过两个窗口的模型（增量非零，合并才有内容可验证）."""
        candidate = build_model(base)
        for batch in batches:
            candidate.accumulate(batch)
            candidate.apply_update(REFERENCE_LR)
        assert candidate.delta_is_zero() is False
        return candidate

    def test_merge_returns_plain_reference_model(self, trained):
        merged = trained.merge()
        assert isinstance(merged, ReferenceSFTModel)
        assert merged.vocab_size == trained.vocab_size
        # 合并沿用**基座**的更新次数：合并本身不是一次参数更新
        assert merged.updates == trained.base_updates

    def test_merged_logits_are_bitwise_equal(self, trained):
        merged = trained.merge()
        for context in range(trained.vocab_size):
            assert merged.logits(context) == trained.logits(context)

    def test_merged_evaluate_loss_is_identical(self, trained, batches):
        merged = trained.merge()
        assert merged.evaluate(batches) == trained.evaluate(batches)

    def test_state_dict_is_the_merged_state(self, trained):
        state = trained.state_dict()
        assert isinstance(state, ModelState)
        assert state == trained.merge().state_dict()
        assert state.vocab_size == trained.vocab_size

    def test_load_state_accepts_matching_weights(self, model, base):
        """``ΔW = 0`` 时合并权重就是基座权重，因此基座状态与它一致."""
        model.load_state(base.state_dict())
        assert model.updates == 0
        assert model.delta_is_zero() is True

    def test_load_state_rejects_unmatched_weights(self, trained, base):
        with pytest.raises(PEFTConfigError, match="只接受与当前合并权重一致的状态"):
            trained.load_state(base.state_dict())

    def test_load_state_rejects_vocab_mismatch(self, trained):
        wrong_vocab = trained.vocab_size + 1
        state = ModelState(
            vocab_size=wrong_vocab,
            weights=tuple((0.0,) * wrong_vocab for _ in range(wrong_vocab)),
            bias=(0.0,) * wrong_vocab,
            updates=0,
        )
        with pytest.raises(PEFTConfigError, match="词表大小不一致"):
            trained.load_state(state)


class TestParameterAccounting:
    """参数量与手算一致（``V=557`` / ``r=8`` 时的 8912 / 310806 / 319718）."""

    def test_trainable_parameters(self, model):
        assert REFERENCE_VOCAB == 557
        assert model.trainable_parameters == 2 * 8 * REFERENCE_VOCAB == 8912

    def test_frozen_parameters(self, model):
        # V×V 的基座矩阵 + 长度 V 的偏置（参考模型的偏置同样冻结）
        expected = REFERENCE_VOCAB * REFERENCE_VOCAB + REFERENCE_VOCAB
        assert model.frozen_parameters == expected
        assert model.frozen_parameters == 310806

    def test_total_parameters(self, model):
        assert model.total_parameters == 319718
        assert model.total_parameters == model.trainable_parameters + model.frozen_parameters

    def test_trainable_ratio(self, model):
        assert model.trainable_ratio == pytest.approx(8912 / 319718)
        assert model.trainable_ratio < 0.03  # "参数高效"这四个字的量级


class TestDescribe:
    """``describe()``：一份可以直接写进日志的规模画像."""

    def test_fields_and_values(self, model):
        info = model.describe()
        assert info["vocab_size"] == REFERENCE_VOCAB
        assert (info["r"], info["lora_alpha"]) == (8, 16)
        assert info["scaling"] == pytest.approx(2.0)
        assert info["targets"] == ["weight"]
        assert info["trainable_parameters"] == 8912
        assert info["frozen_parameters"] == 310806
        assert info["total_parameters"] == 319718
        assert info["trainable_ratio"] == pytest.approx(8912 / 319718)
        assert info["rank_upper_bound"] == 8
        assert info["delta_max_abs"] == 0.0
        assert info["updates"] == 0

    def test_delta_max_abs_grows_after_update(self, model, batches):
        model.accumulate(batches[0])
        model.apply_update(REFERENCE_LR)
        info = model.describe()
        assert info["delta_max_abs"] > 0.0
        assert info["updates"] == 1


class TestAdapterStatePersistence:
    """适配器状态：真正该落盘的东西（``A`` / ``B``），往返必须逐位一致."""

    def test_roundtrip_restores_matrices(self, base, batches):
        trained = build_model(base)
        for batch in batches:
            trained.accumulate(batch)
            trained.apply_update(REFERENCE_LR)
        state = trained.adapter_state_dict()
        restored = build_model(base)
        restored.load_adapter_state(state)
        assert restored.layer.snapshot_matrices() == trained.layer.snapshot_matrices()
        assert restored.updates == trained.updates
        assert restored.logits(11) == trained.logits(11)

    def test_payload_fields(self, model):
        payload = model.adapter_state_dict()
        assert {"a", "b", "updates", "scaling", "config"} <= set(payload)
        assert payload["config"]["r"] == 8
        assert payload["base_parameters"] == model.frozen_parameters
        assert payload["base_updates"] == model.base_updates

    def test_rank_mismatch_rejected(self, base, model):
        state = model.adapter_state_dict()
        state["config"]["r"] = 4
        with pytest.raises(PEFTConfigError, match="适配器配置与模型不一致"):
            build_model(base).load_adapter_state(state)

    def test_alpha_mismatch_rejected(self, base, model):
        state = model.adapter_state_dict()
        state["config"]["lora_alpha"] = 32
        with pytest.raises(PEFTConfigError, match="适配器配置与模型不一致"):
            build_model(base).load_adapter_state(state)

    def test_matrix_shape_mismatch_rejected(self, base, model):
        state = model.adapter_state_dict()
        state["a"] = [[0.0] * REFERENCE_VOCAB]
        with pytest.raises(PEFTConfigError, match="A 的形状"):
            build_model(base).load_adapter_state(state)


class TestTrainLoraReference:
    """便捷入口：返回三元组，``loss`` 记录长度 = ``epochs × 批次数``."""

    def test_history_length_and_eval_loss(self, base, batches):
        model, history, eval_loss = train_lora_reference(
            base,
            batches,
            config=default_reference_lora_config(),
            learning_rate=TRAJECTORY_LR,
            epochs=3,
            evaluate_batches=batches,
        )
        assert len(history) == 3 * len(batches)
        assert [step for step, _ in history] == list(range(1, 3 * len(batches) + 1))
        assert eval_loss is not None
        assert eval_loss < model.uniform_loss

    def test_without_eval_batches_returns_none(self, base, batches):
        _, history, eval_loss = train_lora_reference(
            base,
            batches,
            config=default_reference_lora_config(),
            learning_rate=TRAJECTORY_LR,
        )
        assert eval_loss is None
        assert len(history) == len(batches)

    def test_loss_sequence_decreases_overall(self, base, batches):
        _, history, _ = train_lora_reference(
            base,
            batches,
            config=default_reference_lora_config(),
            learning_rate=TRAJECTORY_LR,
            epochs=2,
        )
        assert history[-1][1] < history[0][1]

    def test_same_seed_gives_identical_history(self, base, batches):
        _, first, _ = train_lora_reference(
            base,
            batches,
            config=default_reference_lora_config(),
            learning_rate=TRAJECTORY_LR,
            epochs=2,
            seed=5,
        )
        _, second, _ = train_lora_reference(
            base,
            batches,
            config=default_reference_lora_config(),
            learning_rate=TRAJECTORY_LR,
            epochs=2,
            seed=5,
        )
        assert first == second


class TestSFTTrainerCompatibility:
    """与 day050 训练循环的兼容性：``SFTTrainer`` 一行不改就能训 LoRA 模型."""

    def test_fit_returns_report_and_updates_parameters(self, tmp_path):
        train = make_examples(4)
        eval_set = make_examples(2)
        tokenizer = build_tokenizer([*train, *eval_set])
        args = SFTTrainingArgs(
            output_dir=str(tmp_path / "out"),
            learning_rate=REFERENCE_LR,
            num_train_epochs=1.0,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            per_device_eval_batch_size=2,
            max_length=96,
        )
        model = build_model(build_base(tokenizer.vocab_size))
        trainer = SFTTrainer(model, tokenizer, args, template=PLAIN, system_prompt=None)
        report = trainer.fit(train, eval_set, save_at_end=False)

        assert report.optimizer_steps > 0
        assert report.eval_loss is not None
        assert report.initial_loss is not None and report.final_loss is not None
        # 训练循环里的每一次 apply_update 都记在模型自己身上
        assert model.updates == report.optimizer_steps
        assert model.delta_is_zero() is False


class TestReferenceAccounting:
    """``reference_lora_accounting``：参考模型的账**不动模型、纯算术**.

    这个函数存在的理由是一个真实的接口断层：``peft.targets.plan_lora`` 面向
    解码器规格（``q_proj`` 等七个投影），而参考模型只有一个叫 ``weight`` 的
    bigram 矩阵。把两者硬接上会命中"目标模块未匹配"的校验（``GET
    /finetune/lora/defaults`` 因此返回过 500，而单元测试全绿——因为测试
    只覆盖了 helper，没有覆盖端点的组合方式）。

    所以这里额外做一条**交叉核对**：纯算术的结果必须与真的构造一个
    ``LoRAReferenceModel`` 之后的 ``describe()`` 完全一致。两边各算一遍，
    任何口径漂移都会立刻暴露。
    """

    def test_accounting_matches_built_model(self):
        base = build_base()
        config = default_reference_lora_config(r=8, lora_alpha=16)
        accounting = reference_lora_accounting(REFERENCE_VOCAB, config)
        described = build_model(base, r=8, lora_alpha=16).describe()
        for key in (
            "vocab_size",
            "r",
            "lora_alpha",
            "scaling",
            "targets",
            "trainable_parameters",
            "frozen_parameters",
            "total_parameters",
            "trainable_ratio",
            "rank_upper_bound",
        ):
            assert accounting[key] == described[key]

    def test_accounting_numbers(self):
        accounting = reference_lora_accounting(REFERENCE_VOCAB, default_reference_lora_config())
        assert accounting["frozen_parameters"] == 557 * 557 + 557
        assert accounting["trainable_parameters"] == 8 * (557 + 557)
        assert accounting["total_parameters"] == 319718
        assert accounting["trainable_ratio"] == pytest.approx(0.0278746, abs=1e-7)
        assert accounting["rank_upper_bound"] == 8
        assert accounting["targets"] == ["weight"]

    def test_rank_scales_linearly(self):
        """参数量对 r 严格线性：每加一秩，多出 ``2V`` 个参数."""
        first = reference_lora_accounting(REFERENCE_VOCAB, default_reference_lora_config(r=8))
        second = reference_lora_accounting(REFERENCE_VOCAB, default_reference_lora_config(r=16))
        assert second["trainable_parameters"] - first["trainable_parameters"] == 8 * 2 * 557

    def test_rejects_small_vocab(self):
        with pytest.raises(PEFTConfigError, match="vocab_size"):
            reference_lora_accounting(1, default_reference_lora_config())

    def test_rejects_non_reference_targets(self):
        """传解码器预设（q/v）时必须报错——这正是那个接口断层的守卫."""
        with pytest.raises(PEFTConfigError, match="bigram 矩阵"):
            reference_lora_accounting(REFERENCE_VOCAB, LoRAConfig())


class TestErrorPaths:
    """``LoRAReferenceModel`` 的错误路径与边角入口."""

    def test_rejects_non_zero_dropout(self):
        with pytest.raises(PEFTConfigError, match="lora_dropout"):
            build_model(build_base(), lora_dropout=0.1)

    def test_rejects_non_none_bias(self):
        with pytest.raises(PEFTConfigError, match="bias"):
            build_model(build_base(), bias="all")

    def test_rejects_unknown_target_preset(self):
        with pytest.raises(PEFTConfigError, match="bigram 矩阵"):
            build_model(build_base(), target_modules="attention")

    def test_config_property_returns_effective_config(self):
        model = build_model(build_base(), r=4, lora_alpha=8)
        assert model.config.r == 4
        assert model.config.lora_alpha == 8
        assert model.config.resolved_targets == ("weight",)

    def test_step_combines_accumulate_and_update(self):
        model = build_model(build_base())
        loss_sum, count = model.step(make_batch(), learning_rate=TRAJECTORY_LR)
        assert count > 0
        assert loss_sum > 0.0
        assert model.updates == 1
        assert model.delta_is_zero() is False

    def test_accumulate_rejects_out_of_range_label(self):
        model = build_model(build_base())
        bad = Batch(
            input_ids=((0, 3, 5),),
            labels=((IGNORE_INDEX, REFERENCE_VOCAB + 5, 5),),
            attention_mask=((1, 1, 1),),
            pad_token_id=0,
        )
        with pytest.raises(SFTLossError, match="落在词表范围"):
            model.accumulate(bad)

    def test_accumulate_rejects_fully_masked_batch(self):
        model = build_model(build_base())
        masked = Batch(
            input_ids=((0, 3, 5),),
            labels=((IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX),),
            attention_mask=((1, 1, 1),),
            pad_token_id=0,
        )
        with pytest.raises(SFTLossError, match="没有任何监督位置"):
            model.accumulate(masked)

    def test_load_state_rejects_bias_length_mismatch(self):
        """偏置长度不对时必须报错，而不是等到 ``logits`` 抛 IndexError."""
        model = build_model(build_base())
        broken = ModelState(
            vocab_size=REFERENCE_VOCAB,
            weights=tuple(tuple(row) for row in model.merged_weights()),
            bias=(0.0,),
            updates=0,
        )
        with pytest.raises(PEFTConfigError, match="偏置长度"):
            model.load_state(broken)

    def test_load_adapter_state_without_config_block(self):
        """适配器状态里没有 ``config`` 时不校验 r/alpha，只校验形状与缩放."""
        model = build_model(build_base(), r=8, lora_alpha=16)
        state = model.adapter_state_dict()
        state.pop("config")
        restored = build_model(build_base(), r=8, lora_alpha=16)
        restored.load_adapter_state(state)
        assert restored.layer.snapshot_matrices() == model.layer.snapshot_matrices()

    def test_load_adapter_state_rejects_config_drift(self):
        model = build_model(build_base(), r=8, lora_alpha=16)
        state = model.adapter_state_dict()
        other = build_model(build_base(), r=4, lora_alpha=8)
        with pytest.raises(PEFTConfigError, match="适配器配置与模型不一致"):
            other.load_adapter_state(state)

    def test_evaluate_rejects_empty_supervision(self):
        model = build_model(build_base())
        masked = Batch(
            input_ids=((0, 3, 5),),
            labels=((IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX),),
            attention_mask=((1, 1, 1),),
            pad_token_id=0,
        )
        with pytest.raises(SFTLossError, match="没有任何监督位置"):
            model.evaluate([masked])

    def test_logits_rejects_out_of_range_context(self):
        model = build_model(build_base())
        with pytest.raises(SFTLossError, match="上下文 token"):
            model.logits(REFERENCE_VOCAB)
