"""``torch_pipeline``：数据 / 设备 / 训练 / 检查点 / 推理的测试（day096 / M8-D7）.

本文件覆盖新包的十一个模块与包入口：口径表、五个失败族、数据集与内容指纹、
确定性打乱与分片、装载器的三个读数、设备字节算术与预算、五件套检查点与哈希、
可注入时钟的推理读数、端到端训练与恢复、七条性质与六张表。

样本全部来自本包自己的确定性构造；**跨天对账真的调用既有包**：
``regularization.loss_and_grad``（day095）、``sequence_models.train.check_gradient_norm``（day094）、
``optimizers.make_train_optimizer``（day092）、``math_foundations.probability.uniforms``（day073）。
"""

from __future__ import annotations

import json
import math
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from smart_research_agent.optimizers.optimizer import make_train_optimizer
from smart_research_agent.regularization.normalization import initial_running
from smart_research_agent.regularization.network import build_regularized, flatten_params
from smart_research_agent.regularization.train import EpochRecord
from smart_research_agent.sequence_models.train import make_sign_dataset
from smart_research_agent.torch_pipeline import (
    checkpoint,
    dataloader,
    datasets,
    device,
    errors,
    inference,
    sampler,
    study,
    train,
    types,
    verify,
)


def _tiny_model():
    """一个宽度 3 的小网络（检查点与推理用例共用，便宜）。"""
    return build_regularized("lstm", input_size=1, hidden_size=3, classes=2, seed=5)


@pytest.fixture(scope="module")
def sign_dataset() -> datasets.TabularDataset:
    """模块级缓存：day094 的 ±1 符号数据集（32 条，完全确定）。"""
    return datasets.make_dataset(make_sign_dataset(5), name="sign")


@pytest.fixture(scope="module")
def ten_dataset() -> datasets.TabularDataset:
    """模块级缓存：一个 10 条样本的小数据集（用来让 drop_last 的两个分支分家）。"""
    return datasets.make_dataset(make_sign_dataset(5)[:10], name="ten")


@pytest.fixture(scope="module")
def trained(sign_dataset) -> train.PipelineReport:
    """模块级缓存：一次小训练（20 轮 × 批 8），供多个用例共用。"""
    return train.train_pipeline(
        sign_dataset, config=train.PipelineConfig(epochs=20, batch_size=8, seed=100)
    )


@pytest.fixture(scope="module")
def pipeline_model():
    """模块级缓存：与 ``train._build_model`` 同构的一份参数（宽度 4 的 lstm）。"""
    return build_regularized(
        train.DEFAULT_CELL,
        input_size=1,
        hidden_size=train.DEFAULT_HIDDEN_SIZE,
        classes=2,
        seed=train.DEFAULT_SEED,
    )


@pytest.fixture(scope="module")
def ablation(sign_dataset) -> dict[str, train.PipelineReport]:
    """模块级缓存：两个变体各训一次（**只训 4 轮**，够看出读数差异）。"""
    variants = {
        "基准": replace(study.ABLATION_VARIANTS["基准"], epochs=4),
        "丢尾批": replace(study.ABLATION_VARIANTS["丢尾批"], epochs=4),
    }
    return train.compare_configs(sign_dataset, variants)


@pytest.fixture(scope="module")
def property_report() -> verify.PropertyReport:
    """模块级缓存：七条性质的汇总报告（跑一次昂贵的第 ⑥ 条）。"""
    return verify.check_all()


# --------------------------------------------------------------------------- 口径表


def test_stage_tables_are_closed() -> None:
    """七个阶段的三张表逐键对齐。"""
    assert types.STAGES == (
        "dataset",
        "sampler",
        "dataloader",
        "device",
        "train",
        "checkpoint",
        "inference",
    )
    assert set(types.STAGES) == set(types.STAGE_DESCRIPTIONS)
    assert set(types.STAGES) == set(types.STAGE_OWNERS)
    assert "本包新建" in types.STAGE_OWNERS["checkpoint"]
    assert "day095" in types.STAGE_OWNERS["train"]


def test_axis_tables_are_closed() -> None:
    """四条读数轴的三张表逐键对齐。"""
    assert types.PIPELINE_AXES == ("batch", "worker", "device", "time")
    assert set(types.PIPELINE_AXES) == set(types.AXIS_DESCRIPTIONS)
    assert set(types.PIPELINE_AXES) == set(types.AXIS_FORMULAS)
    assert "//" in types.AXIS_FORMULAS["batch"]
    assert "%" in types.AXIS_FORMULAS["worker"]


def test_device_and_dtype_tables_are_closed() -> None:
    """三个设备与四种精度的表逐键对齐。"""
    assert types.DEVICE_KINDS == ("cpu", "cuda", "mps")
    assert set(types.DEVICE_KINDS) == set(types.DEVICE_DESCRIPTIONS)
    assert set(types.DEVICE_KINDS) == set(types.DEVICE_PREFIXES)
    assert types.DTYPE_BYTES == {"float32": 4, "float64": 8, "float16": 2, "bfloat16": 2}
    assert set(types.DTYPE_BYTES) == set(types.DTYPE_DESCRIPTIONS)


def test_seven_properties_have_descriptions_and_failures() -> None:
    """七条性质：名单 / 说明 / "失败意味着什么"三张表逐键对齐。"""
    assert len(types.PIPELINE_PROPERTIES) == 7
    assert set(types.PROPERTY_DESCRIPTIONS) == set(types.PIPELINE_PROPERTIES)
    assert set(types.PROPERTY_FAILURE) == set(types.PIPELINE_PROPERTIES)
    assert all(types.PROPERTY_DESCRIPTIONS.values())
    assert all(types.PROPERTY_FAILURE.values())
    assert types.PROPERTIES == types.PIPELINE_PROPERTIES


def test_notes_are_ten_and_boundaries_are_five() -> None:
    """十条笔记的顺序表与键集合一致；五条边界是一份非空清单。"""
    assert len(types.PIPELINE_NOTES) == 10
    assert types.NOTES_ORDER == tuple(types.PIPELINE_NOTES)
    assert all(types.PIPELINE_NOTES.values())
    assert len(types.PIPELINE_BOUNDARIES) == 5
    assert all(types.PIPELINE_BOUNDARIES)


def test_formulas_and_tolerances() -> None:
    """十张公式都写下了一行可读的式子，四个容差取了写死的值。"""
    assert "n // b" in types.BATCH_COUNT_FORMULA
    assert "ceil" in types.BATCH_COUNT_FORMULA
    assert "dropped" in types.DROPPED_SAMPLES_FORMULA
    assert "optimizer_multiplier" in types.TOTAL_BYTES_FORMULA
    assert "五件套" in types.CHECKPOINT_FILES_FORMULA
    assert "throughput" in types.LATENCY_FORMULA
    assert types.EXACT_TOLERANCE == 0.0
    assert types.PARAMS_TOLERANCE == 1e-12
    assert types.DEVICE_BYTES_TOLERANCE == 1e-9
    assert types.SHARD_COVERAGE_LOWER_BOUND == 1.0


# --------------------------------------------------------------------------- 失败族


def test_family_tables_are_aligned() -> None:
    """五个族的"该怎么办"表与类表逐键对齐。"""
    assert set(errors.FAMILY_OUTCOMES) == {
        "ShapeError",
        "ParameterError",
        "NumericError",
        "DeviceError",
        "CheckpointError",
    }
    assert set(errors.FAMILY_OUTCOMES) == set(errors._FAMILY_CLASSES)
    assert all(errors.FAMILY_OUTCOMES.values())


def test_returned_and_absent_families_are_constants() -> None:
    """本课请回了 ``CheckpointError``，而 ``GradientError`` 再次缺席（可断言的事实）。"""
    assert errors.RETURNED_FAMILY == "CheckpointError"
    assert errors.RETURNED_FAMILY_REASON
    assert errors.ABSENT_FAMILY == "GradientError"
    assert errors.ABSENT_FAMILY_REASON


def test_errors_inherit_both_families() -> None:
    """本族错误既是 ``TorchPipelineError``、也是 day075 的对应族。"""
    from smart_research_agent.transformer_core import errors as core_errors

    assert issubclass(errors.ShapeError, errors.TorchPipelineError)
    assert issubclass(errors.ShapeError, core_errors.ShapeError)
    assert issubclass(errors.ParameterError, core_errors.ParameterError)
    assert issubclass(errors.NumericError, core_errors.NumericError)
    assert issubclass(errors.DeviceError, errors.TorchPipelineError)
    assert issubclass(errors.CheckpointError, errors.TorchPipelineError)
    assert issubclass(errors.TorchPipelineError, ValueError)


def test_four_core_families_can_be_triggered(sign_dataset, tmp_path: Path) -> None:
    """形状 / 参数 / 数值 / 设备 / 检查点五个族各触发一次。"""
    with pytest.raises(errors.ShapeError):
        dataloader.collate(sign_dataset, (999,))
    with pytest.raises(errors.ParameterError):
        dataloader.DataLoader(sign_dataset, 0)
    bad_sample = (sign_dataset[0][0], 2)
    with pytest.raises(errors.NumericError):
        datasets.make_dataset((bad_sample,), name="bad")
    with pytest.raises(errors.DeviceError):
        device.plan_device(4, batch_size=1, width=1, device="tpu")
    with pytest.raises(errors.CheckpointError):
        checkpoint.load_checkpoint(tmp_path / "not-here")


# --------------------------------------------------------------------------- 数据集


def test_as_samples_guards(sign_dataset) -> None:
    """样本校验：空集 / 非可迭代 / 标签越界三种失败各一次。"""
    with pytest.raises(errors.ShapeError):
        datasets.as_samples(())
    with pytest.raises(errors.ShapeError):
        datasets.as_samples(5)
    with pytest.raises(errors.NumericError):
        datasets.as_samples(((sign_dataset[0][0], 7),))
    assert datasets.as_samples(sign_dataset.samples) == sign_dataset.samples


def test_tabular_dataset_validation_and_readings(sign_dataset) -> None:
    """``TabularDataset`` 的校验与读数方法；重建后指纹不变。"""
    assert len(sign_dataset) == 32
    assert sign_dataset[0] == sign_dataset.samples[0]
    assert len(sign_dataset.labels()) == 32
    assert set(sign_dataset.labels()) <= {0, 1}
    assert "指纹" in sign_dataset.line()
    assert "32 条样本" in sign_dataset.line()
    rebuilt = datasets.make_dataset(datasets.as_samples(sign_dataset.samples), name="rebuilt")
    assert rebuilt.fingerprint() == sign_dataset.fingerprint()
    assert rebuilt.name == "rebuilt"
    with pytest.raises(errors.ParameterError):
        datasets.TabularDataset(name="", samples=sign_dataset.samples)


def test_dataset_fingerprint_is_stable_and_sensitive(sign_dataset) -> None:
    """内容指纹：同一份不变、改一个标签就变（16 位十六进制）。"""
    first = datasets.dataset_fingerprint(sign_dataset.samples)
    assert first == datasets.dataset_fingerprint(sign_dataset.samples)
    assert len(first) == 16
    changed = list(sign_dataset.samples)
    changed[0] = (changed[0][0], 1 - changed[0][1])
    assert datasets.dataset_fingerprint(tuple(changed)) != first
    assert sign_dataset.fingerprint() == first


def test_split_dataset_is_deterministic_and_non_empty(sign_dataset) -> None:
    """切分只依赖 ``(seed, n)``，两侧都非空，且并集是全集。"""
    train_a, eval_a = datasets.split_dataset(sign_dataset, eval_ratio=0.25, seed=100)
    train_b, eval_b = datasets.split_dataset(sign_dataset, eval_ratio=0.25, seed=100)
    assert train_a.fingerprint() == train_b.fingerprint()
    assert eval_a.fingerprint() == eval_b.fingerprint()
    assert len(train_a) + len(eval_a) == len(sign_dataset)
    assert len(train_a) > 0 and len(eval_a) > 0
    other, _other_eval = datasets.split_dataset(sign_dataset, eval_ratio=0.25, seed=7)
    assert other.fingerprint() != train_a.fingerprint()


def test_split_dataset_guards() -> None:
    """比例非法 / 样本太少都当场拒绝。"""
    one = datasets.make_dataset((make_sign_dataset(3)[0],), name="one")
    with pytest.raises(errors.NumericError):
        datasets.split_dataset(one, eval_ratio=0.5, seed=0)
    with pytest.raises(errors.ParameterError):
        datasets.split_dataset(one, eval_ratio=0.0, seed=0)
    with pytest.raises(errors.ParameterError):
        datasets.split_dataset(one, eval_ratio=1.0, seed=0)
    with pytest.raises(errors.ParameterError):
        datasets.split_dataset(one, eval_ratio="x", seed=0)


# --------------------------------------------------------------------------- 采样


def test_shuffle_indices_is_a_permutation() -> None:
    """打乱是一个排列，且同种子逐位相同。"""
    first = sampler.shuffle_indices(9, seed=3)
    assert sorted(first) == list(range(9))
    assert sampler.shuffle_indices(9, seed=3) == first
    assert sampler.shuffle_indices(9, seed=4) != first
    with pytest.raises(errors.ParameterError):
        sampler.shuffle_indices(-1, seed=0)


def test_sampler_identity_when_not_shuffled() -> None:
    """``shuffle=False`` 时顺序就是恒等。"""
    plain = sampler.Sampler(size=5, shuffle=False, seed=1)
    assert plain.order(0) == (0, 1, 2, 3, 4)
    assert plain.order(9) == (0, 1, 2, 3, 4)
    assert "shuffle=False" in plain.line()


def test_sampler_order_is_deterministic_and_seed_sensitive() -> None:
    """打乱只依赖 ``(seed, epoch)``：同键相同、改一个就变。"""
    base = sampler.Sampler(size=32, shuffle=True, seed=100)
    assert base.order(3) == base.order(3)
    assert sorted(base.order(3)) == list(range(32))
    assert sampler.Sampler(size=32, shuffle=True, seed=100).order(4) != base.order(3)
    assert sampler.Sampler(size=32, shuffle=True, seed=101).order(3) != base.order(3)


def test_sampler_guards() -> None:
    """非法的 size / shuffle / seed / epoch 各拒绝一次。"""
    with pytest.raises(errors.NumericError):
        sampler.Sampler(size=0)
    with pytest.raises(errors.ParameterError):
        sampler.Sampler(size=4, shuffle="yes")
    with pytest.raises(errors.ParameterError):
        sampler.Sampler(size=4, seed=1.5)
    with pytest.raises(errors.ParameterError):
        sampler.Sampler(size=4).order(-1)
    with pytest.raises(errors.ParameterError):
        sampler.Sampler(size="4")


def test_shard_indices_partition() -> None:
    """分片互不相交、并集为全集。"""
    shards = [sampler.shard_indices(32, workers=4, worker_id=k) for k in range(4)]
    union = set().union(*shards)
    assert union == set(range(32))
    assert sum(len(shard) for shard in shards) == 32
    assert sampler.shard_indices(32, workers=1, worker_id=0) == tuple(range(32))


def test_shard_indices_guards() -> None:
    """非法的 size / workers / worker_id 各拒绝一次。"""
    with pytest.raises(errors.NumericError):
        sampler.shard_indices(0, workers=1, worker_id=0)
    with pytest.raises(errors.ParameterError):
        sampler.shard_indices(8, workers=0, worker_id=0)
    with pytest.raises(errors.ParameterError):
        sampler.shard_indices(8, workers=2, worker_id=2)
    with pytest.raises(errors.ParameterError):
        sampler.shard_indices(8, workers=2, worker_id=-1)


# --------------------------------------------------------------------------- 装载


def test_collate_and_batch_readings(sign_dataset) -> None:
    """``collate`` 取出的三个分量逐位对应，``Batch`` 的读数正确。"""
    batch = dataloader.collate(sign_dataset, (3, 1, 4))
    assert batch.indices == (3, 1, 4)
    assert batch.labels == tuple(sign_dataset[i][1] for i in (3, 1, 4))
    assert batch.size == 3
    assert "3 条" in batch.line()
    empty = dataloader.collate(sign_dataset, ())
    assert empty.size == 0
    assert "空批" in empty.line()


def test_collate_guards(sign_dataset) -> None:
    """非数据集 / 非可迭代下标 / 非整数下标 / 越界下标各拒绝一次。"""
    with pytest.raises(errors.ParameterError):
        dataloader.collate("dataset", (0,))
    with pytest.raises(errors.ShapeError):
        dataloader.collate(sign_dataset, 5)
    with pytest.raises(errors.ShapeError):
        dataloader.collate(sign_dataset, (1.5,))
    with pytest.raises(errors.ShapeError):
        dataloader.collate(sign_dataset, (-1,))


def test_batch_post_init_guards() -> None:
    """批的三个分量必须等长，标签必须是整数（布尔不算）。"""
    row = ((1.0,),)
    with pytest.raises(errors.ShapeError):
        dataloader.Batch(indices=(0,), inputs=(row,), labels=())
    with pytest.raises(errors.ShapeError):
        dataloader.Batch(indices=(0,), inputs=(row,), labels=(True,))


def test_dataloader_batch_counts_drop_last(ten_dataset) -> None:
    """批数公式：drop_last 时是 ``n // b``，且丢弃恰是尾批。"""
    kept = dataloader.DataLoader(ten_dataset, 4, shuffle=False, drop_last=False)
    dropped = dataloader.DataLoader(ten_dataset, 4, shuffle=False, drop_last=True)
    assert len(kept) == 3
    assert len(dropped) == 2
    assert kept.dropped_samples() == 0
    assert dropped.dropped_samples() == 2
    assert len(kept.batches(1)) == 3
    assert all(batch.size <= 4 for batch in kept.batches(1))
    assert kept.steps_per_epoch() == len(kept)
    assert "批数" in kept.line()


def test_dataloader_full_division(sign_dataset) -> None:
    """32 条样本、批 8：两个分支都是 4 批、丢弃 0（整除时没有尾批）。"""
    for drop_last in (False, True):
        loader = dataloader.DataLoader(sign_dataset, 8, shuffle=True, seed=1, drop_last=drop_last)
        assert len(loader) == 4
        assert loader.dropped_samples() == 0


def test_dataloader_worker_shard_batches(sign_dataset) -> None:
    """多进程先分片再切批：两个 worker 的批合起来覆盖全部样本、且互不重叠。"""
    seen: list[int] = []
    for worker_id in range(2):
        loader = dataloader.DataLoader(
            sign_dataset, 4, shuffle=False, workers=2, worker_id=worker_id
        )
        for batch in loader.batches(1):
            seen.extend(batch.indices)
    assert sorted(seen) == list(range(32))


def test_dataloader_guards(sign_dataset) -> None:
    """非数据集 / 非法 batch_size / 非布尔开关各拒绝一次。"""
    with pytest.raises(errors.ParameterError):
        dataloader.DataLoader("dataset", 4)
    with pytest.raises(errors.ParameterError):
        dataloader.DataLoader(sign_dataset, 0)
    with pytest.raises(errors.ParameterError):
        dataloader.DataLoader(sign_dataset, 4, shuffle="yes")
    with pytest.raises(errors.ParameterError):
        dataloader.DataLoader(sign_dataset, 4, drop_last=1)


def test_batch_samples_round_trip(sign_dataset) -> None:
    """``batch_samples`` 把批还原成 ``(输入, 标签)`` 的样本串。"""
    batch = dataloader.collate(sign_dataset, (0, 5))
    samples = dataloader.batch_samples(batch)
    assert len(samples) == 2
    assert samples[0] == sign_dataset[0]


def test_dataloader_is_reproducible(sign_dataset) -> None:
    """同一个 ``(seed, epoch)`` 的两个装载器给出同一串批。"""
    first = dataloader.DataLoader(sign_dataset, 8, shuffle=True, seed=3)
    second = dataloader.DataLoader(sign_dataset, 8, shuffle=True, seed=3)
    assert [batch.indices for batch in first.batches(2)] == [
        batch.indices for batch in second.batches(2)
    ]


# --------------------------------------------------------------------------- 设备


def test_tensor_parameter_activation_bytes() -> None:
    """三张字节公式各算一次（含标量与双精度）。"""
    assert device.tensor_bytes((2, 3)) == 24
    assert device.tensor_bytes((2, 3), dtype="float64") == 48
    assert device.tensor_bytes(()) == 4
    assert device.parameter_bytes(10) == 40
    assert device.activation_bytes(2, 3) == 24


def test_device_plan_readings() -> None:
    """``DevicePlan`` 的总量 / fits / 余量三个派生量。"""
    plan = device.DevicePlan(
        device="cpu",
        dtype="float32",
        params=100,
        activations=40,
        optimizer_bytes=300,
        budget_bytes=1000,
    )
    assert plan.total_bytes == 440
    assert plan.fits is True
    assert plan.headroom_bytes == 560
    assert "fits=True" in plan.line()
    assert "预算充足" in plan.warn_line()
    tight = device.DevicePlan(
        device="cuda",
        dtype="float32",
        params=100,
        activations=40,
        optimizer_bytes=300,
        budget_bytes=400,
    )
    assert tight.fits is False
    assert tight.headroom_bytes == -40
    assert "超出预算 40 字节" in tight.warn_line()


def test_plan_device_formula() -> None:
    """``plan_device`` 的三个分量与手算逐项一致。"""
    plan = device.plan_device(
        44, batch_size=8, width=4, device="cuda", dtype="float32", optimizer_multiplier=3
    )
    assert plan.params == 44 * 4
    assert plan.activations == 8 * 4 * 4
    assert plan.optimizer_bytes == 3 * 44 * 4
    assert plan.total_bytes == 44 * 4 + 8 * 4 * 4 + 3 * 44 * 4
    assert plan.budget_bytes == device.DEFAULT_DEVICE_BUDGETS["cuda"]


def test_plan_device_can_not_fit() -> None:
    """预算不足**不抛异常**：它由 ``fits`` 暴露。"""
    plan = device.plan_device(
        2_000_000_000, batch_size=1024, width=4096, device="cuda", budget_bytes=1024
    )
    assert plan.fits is False
    assert plan.headroom_bytes < 0
    assert "警告" in plan.warn_line()
    assert "fits=False" in plan.line()


def test_move_report_lines() -> None:
    """搬运行：CPU 是 0 字节，其余给出总搬运量与精度。"""
    cpu_plan = device.plan_device(4, batch_size=1, width=1, device="cpu")
    assert "0 字节" in device.move_report(cpu_plan)[0]
    cuda_plan = device.plan_device(4, batch_size=1, width=1, device="cuda")
    lines = device.move_report(cuda_plan)
    assert "搬运" in lines[0] and "float32" in lines[1]


def test_device_guards() -> None:
    """未知设备 / 未知精度是 ``DeviceError``；非法形状是 ``ParameterError``。"""
    with pytest.raises(errors.DeviceError):
        device.plan_device(4, batch_size=1, width=1, device="tpu")
    with pytest.raises(errors.DeviceError):
        device.plan_device(4, batch_size=1, width=1, device="cpu", dtype="float32x")
    with pytest.raises(errors.ParameterError):
        device.tensor_bytes(5)
    with pytest.raises(errors.ParameterError):
        device.tensor_bytes((-1,))
    with pytest.raises(errors.ParameterError):
        device.parameter_bytes(-1)
    with pytest.raises(errors.ParameterError):
        device.parameter_bytes("ten")
    with pytest.raises(errors.ParameterError):
        device.activation_bytes(0, 1)
    with pytest.raises(errors.ParameterError):
        device.plan_device(0, batch_size=1, width=1, device="cpu")
    with pytest.raises(errors.ParameterError):
        device.plan_device(4, batch_size=1, width=1, device="cpu", budget_bytes=0)
    with pytest.raises(errors.ParameterError):
        device.move_report("plan")


# --------------------------------------------------------------------------- 检查点


def _saved_checkpoint(directory, *, optimizer_name: str = "adam", step: int = 4):
    """一个共用的"写一份小检查点"助手（参数用宽度 3 的小网络）。"""
    model = _tiny_model()
    optimizer = make_train_optimizer(optimizer_name, 0.01)
    flat = flatten_params(model)
    optimizer.step(flat, tuple(0.0 for _ in flat))
    return checkpoint.save_checkpoint(
        directory,
        params=model,
        optimizer_state=optimizer.state(),
        step=step,
        metrics={"epoch": 1, "step": step, "initial_loss": 0.5},
        history=(),
    ), model, flat


def test_save_and_load_checkpoint_round_trip(tmp_path: Path) -> None:
    """五件套写盘 → 读回：权重逐位相同，清单字段可读。"""
    manifest, model, flat = _saved_checkpoint(tmp_path)
    assert manifest.step == 4
    assert manifest.params_count == len(flat)
    assert len(manifest.params_sha256) == 64
    assert set(manifest.files) == set(checkpoint.CHECKPOINT_FILES)
    assert "step=4" in manifest.line()
    for name in checkpoint.CHECKPOINT_FILES:
        assert (tmp_path / name).exists()
    loaded = checkpoint.load_checkpoint(tmp_path)
    assert loaded.params == flat
    assert checkpoint.resume_step(loaded) == 4
    assert "检查点" in loaded.line()
    checkpoint.verify_checkpoint(tmp_path, loaded.manifest)


def test_load_checkpoint_names_the_missing_file(tmp_path: Path) -> None:
    """缺文件时错误信息里**点名**是哪一个。"""
    _saved_checkpoint(tmp_path, optimizer_name="sgd")
    (tmp_path / "params.json").unlink()
    with pytest.raises(errors.CheckpointError) as info:
        checkpoint.load_checkpoint(tmp_path)
    assert "params.json" in str(info.value)


def test_verify_checkpoint_detects_hash_mismatch(tmp_path: Path) -> None:
    """优化器状态 / 权重被改过时哈希对不上。"""
    manifest, _model, _flat = _saved_checkpoint(tmp_path)
    (tmp_path / "optimizer.json").write_text(
        json.dumps({"name": "adam", "step_count": 99}), encoding="utf-8"
    )
    with pytest.raises(errors.CheckpointError):
        checkpoint.verify_checkpoint(tmp_path, manifest)
    _saved_checkpoint(tmp_path)
    payload = json.loads((tmp_path / "params.json").read_text(encoding="utf-8"))
    payload["params"][0] = payload["params"][0] + 1.0
    (tmp_path / "params.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(errors.CheckpointError):
        checkpoint.verify_checkpoint(tmp_path, manifest)


def test_verify_checkpoint_guards(tmp_path: Path) -> None:
    """目录不存在 / 文件表不全 / manifest 类型不对 / params 字段缺失各拒绝一次。"""
    manifest, _model, _flat = _saved_checkpoint(tmp_path, optimizer_name="sgd")
    with pytest.raises(errors.CheckpointError):
        checkpoint.verify_checkpoint(tmp_path / "gone", manifest)
    with pytest.raises(errors.ParameterError):
        checkpoint.verify_checkpoint(tmp_path, "manifest")
    (tmp_path / "params.json").write_text('{"count": 3}', encoding="utf-8")
    with pytest.raises(errors.CheckpointError):
        checkpoint.verify_checkpoint(tmp_path, manifest)
    _saved_checkpoint(tmp_path, optimizer_name="sgd")
    (tmp_path / "history.jsonl").unlink()
    with pytest.raises(errors.CheckpointError):
        checkpoint.verify_checkpoint(tmp_path, manifest)


def test_optimizer_payload_uses_state_keys() -> None:
    """优化器状态存哪些键由 ``OPTIMIZER_STATE_KEYS`` 决定。"""
    sgd = make_train_optimizer("sgd", 0.01)
    payload = checkpoint.optimizer_payload(sgd.state())
    assert set(payload) <= set(checkpoint.OPTIMIZER_SCALAR_KEYS) | {"base"}
    nesterov = make_train_optimizer("nesterov", 0.01)
    nesterov.step((1.0, 2.0), (0.1, 0.2))
    assert "velocity" in checkpoint.optimizer_payload(nesterov.state())
    with pytest.raises(errors.ParameterError):
        checkpoint.optimizer_payload("state")
    with pytest.raises(errors.ParameterError):
        checkpoint.optimizer_payload({"name": "nope"})


def test_record_payload_round_trip() -> None:
    """``EpochRecord`` 与 ``history.jsonl`` 的一行可以互相还原。"""
    record = EpochRecord(
        epoch=3, train_loss=0.5, eval_loss=0.25, eval_accuracy=1.0, learning_rate=0.01
    )
    payload = checkpoint.record_payload(record)
    assert checkpoint.record_from_payload(payload) == record
    with pytest.raises(errors.ParameterError):
        checkpoint.record_payload("record")
    with pytest.raises(errors.CheckpointError):
        checkpoint.record_from_payload({"epoch": 1})


def test_checkpoint_guards(tmp_path: Path) -> None:
    """save / manifest / resume_step 的三类护栏。"""
    model = _tiny_model()
    optimizer = make_train_optimizer("sgd", 0.01)
    with pytest.raises(errors.ParameterError):
        checkpoint.save_checkpoint(
            tmp_path, params="model", optimizer_state=optimizer.state(), step=0, metrics={}, history=()
        )
    with pytest.raises(errors.ParameterError):
        checkpoint.save_checkpoint(
            tmp_path, params=model, optimizer_state=optimizer.state(), step=0, metrics="m", history=()
        )
    with pytest.raises(errors.ParameterError):
        checkpoint.save_checkpoint(
            tmp_path, params=model, optimizer_state=optimizer.state(), step=-1, metrics={}, history=()
        )
    with pytest.raises(errors.ParameterError):
        checkpoint.CheckpointManifest(
            step=-1, params_sha256="a", optimizer_sha256="b", files=(), params_count=1
        )
    with pytest.raises(errors.ShapeError):
        checkpoint.CheckpointManifest(
            step=0, params_sha256="a", optimizer_sha256="b", files=(), params_count=0
        )
    with pytest.raises(errors.ParameterError):
        checkpoint.CheckpointManifest(
            step=0, params_sha256="a", optimizer_sha256="b", files=(), params_count="many"
        )
    with pytest.raises(errors.ParameterError):
        checkpoint.resume_step("checkpoint")


def test_load_checkpoint_bad_json(tmp_path: Path) -> None:
    """非法 JSON / 缺字段 / 空行 / 不完整清单四类都抛 ``CheckpointError``（空行只跳过）。"""
    for name in checkpoint.CHECKPOINT_FILES:
        (tmp_path / name).write_text("{not json", encoding="utf-8")
    with pytest.raises(errors.CheckpointError) as info:
        checkpoint.load_checkpoint(tmp_path)
    assert "params.json" in str(info.value) or "meta.json" in str(info.value)
    _saved_checkpoint(tmp_path, optimizer_name="sgd")
    (tmp_path / "params.json").write_text('{"count": 3}', encoding="utf-8")
    with pytest.raises(errors.CheckpointError):
        checkpoint.load_checkpoint(tmp_path)
    _saved_checkpoint(tmp_path, optimizer_name="sgd")
    (tmp_path / "meta.json").write_text("{}", encoding="utf-8")
    with pytest.raises(errors.CheckpointError):
        checkpoint.load_checkpoint(tmp_path)
    _saved_checkpoint(tmp_path, optimizer_name="sgd")
    (tmp_path / "history.jsonl").write_text("\n\n", encoding="utf-8")
    assert checkpoint.load_checkpoint(tmp_path).history == ()


def test_load_checkpoint_bad_history_line(tmp_path: Path) -> None:
    """``history.jsonl`` 的一行坏了也要点名。"""
    _saved_checkpoint(tmp_path, optimizer_name="sgd")
    (tmp_path / "history.jsonl").write_text("{bad\n", encoding="utf-8")
    with pytest.raises(errors.CheckpointError):
        checkpoint.load_checkpoint(tmp_path)


# --------------------------------------------------------------------------- 推理


def test_inference_engine_and_fake_clock() -> None:
    """注入假时钟后，三个计数器逐位可复现。"""
    model = _tiny_model()
    engine = inference.build_inference_engine(model, clock=inference.stepping_clock(step=1e-6))
    assert engine.model is engine.params
    assert engine.features == 3
    logits = inference.predict_one(engine, make_sign_dataset(3)[0])
    assert len(logits) == 2
    assert engine.calls == 1
    assert engine.samples == 1
    assert engine.latency_seconds == pytest.approx(1e-6, abs=1e-18)
    assert "调用 1 次" in engine.line()


def test_predict_one_and_batch_consistent(sign_dataset) -> None:
    """``predict_batch`` 与逐条 ``predict_one`` 的输出逐位相同；也接受一个 ``dataloader.Batch``。"""
    model = _tiny_model()
    samples = tuple(sign_dataset[index] for index in range(4))
    batched = inference.build_inference_engine(model, clock=inference.stepping_clock())
    sequential = inference.build_inference_engine(model, clock=inference.stepping_clock())
    rows = inference.predict_batch(batched, samples)
    one_by_one = tuple(inference.predict_one(sequential, sample) for sample in samples)
    assert rows == one_by_one
    assert batched.calls == 1
    assert batched.samples == 4
    loader_engine = inference.build_inference_engine(model, clock=inference.stepping_clock())
    batch = dataloader.collate(sign_dataset, (0, 1))
    assert len(inference.predict_batch(loader_engine, batch)) == 2
    assert loader_engine.samples == 2


def test_predict_dataset_labels(sign_dataset) -> None:
    """整份数据集的预测标签：长度一致、取值 0 / 1、调用次数等于批数。"""
    model = _tiny_model()
    engine = inference.build_inference_engine(model, clock=inference.stepping_clock())
    labels = inference.predict_dataset(engine, sign_dataset, batch_size=8)
    assert len(labels) == len(sign_dataset)
    assert set(labels) <= {0, 1}
    assert engine.calls == 4
    assert engine.samples == 32


def test_latency_report_derived_readings() -> None:
    """延迟读数：每样本秒数与吞吐逐位可复算。"""
    report = inference.LatencyReport(calls=4, samples=8, seconds=2.0)
    assert report.seconds_per_sample == 0.25
    assert report.throughput == 4.0
    assert "每条" in report.line()
    empty = inference.LatencyReport(calls=0, samples=0, seconds=0.0)
    assert empty.seconds_per_sample == 0.0
    assert empty.throughput == 0.0
    with pytest.raises(errors.NumericError):
        inference.LatencyReport(calls=1, samples=1, seconds=-1.0)
    with pytest.raises(errors.ParameterError):
        inference.LatencyReport(calls=-1, samples=1, seconds=1.0)


def test_latency_report_matches_counter_and_resets(sign_dataset) -> None:
    """``latency_report`` 与注入假 clock 的手算值一致；``reset_counters`` 只清计数器。"""
    model = _tiny_model()
    engine = inference.build_inference_engine(model, clock=inference.stepping_clock(step=1e-6))
    inference.predict_dataset(engine, sign_dataset, batch_size=8)
    report = inference.latency_report(engine)
    assert report.calls == 4
    assert report.samples == 32
    assert report.seconds == pytest.approx(4e-6, abs=1e-18)
    assert report.seconds_per_sample == pytest.approx(1.25e-7, abs=1e-18)
    inference.reset_counters(engine)
    assert (engine.calls, engine.samples, engine.latency_seconds) == (0, 0, 0.0)
    assert engine.running.batches == 0


def test_inference_guards(sign_dataset) -> None:
    """推理层的多类护栏（含 ``running`` 类型不对）。"""
    model = _tiny_model()
    with pytest.raises(errors.ParameterError):
        inference.build_inference_engine("model")
    with pytest.raises(errors.ParameterError):
        inference.InferenceEngine(params="model", running=initial_running(3))
    with pytest.raises(errors.ParameterError):
        inference.InferenceEngine(params=model, running="stats")
    engine = inference.build_inference_engine(model, clock=inference.stepping_clock())
    bad_running = initial_running(model.features + 1)
    with pytest.raises(errors.ShapeError):
        inference.InferenceEngine(params=model, running=bad_running)
    with pytest.raises(errors.ParameterError):
        inference.InferenceEngine(params=model, running=initial_running(3), clock=5)
    with pytest.raises(errors.ParameterError):
        inference.predict_one("engine", sign_dataset[0])
    with pytest.raises(errors.ParameterError):
        inference.predict_batch("engine", ())
    with pytest.raises(errors.ShapeError):
        inference.predict_batch(engine, 5)
    with pytest.raises(errors.ParameterError):
        inference.predict_dataset(engine, sign_dataset, batch_size=0)
    with pytest.raises(errors.ParameterError):
        inference.predict_dataset(engine, "dataset", batch_size=1)
    with pytest.raises(errors.ParameterError):
        inference.predict_dataset("engine", sign_dataset, batch_size=1)
    with pytest.raises(errors.ParameterError):
        inference.latency_report("engine")
    with pytest.raises(errors.ParameterError):
        inference.reset_counters("engine")


def test_clock_guards() -> None:
    """倒退 / 非有限的时钟当场拒绝；``stepping_clock`` 的步长必须为正。"""
    model = _tiny_model()
    values = iter([1.0, 0.0])
    backwards = inference.build_inference_engine(model, clock=lambda: next(values))
    with pytest.raises(errors.NumericError):
        inference.predict_one(backwards, make_sign_dataset(3)[0])
    nan_clock = inference.build_inference_engine(model, clock=lambda: float("nan"))
    with pytest.raises(errors.NumericError):
        inference.predict_one(nan_clock, make_sign_dataset(3)[0])
    with pytest.raises(errors.ParameterError):
        inference.stepping_clock(0.0)


# --------------------------------------------------------------------------- 训练


def test_pipeline_config_validate_and_line() -> None:
    """缺省配置合法，且一行读数包含关键字段。"""
    config = train.PipelineConfig()
    config.validate()
    text = config.line()
    assert "轮" in text and "adam" in text and "cpu/float32" in text


def test_pipeline_config_guards() -> None:
    """非法配置逐项拒绝（数值类 ``ParameterError``、设备类 ``DeviceError``）。"""
    for kwargs in (
        {"epochs": 0},
        {"epochs": 1.5},
        {"batch_size": 0},
        {"workers": 0},
        {"workers": "1"},
        {"early_stop_patience": -1},
        {"learning_rate": 0.0},
        {"optimizer": "nope"},
        {"shuffle": 1},
        {"drop_last": 1},
        {"eval_ratio": 1.0},
        {"eval_ratio": -0.1},
        {"seed": 1.5},
    ):
        with pytest.raises(errors.ParameterError):
            train.PipelineConfig(**kwargs).validate()
    with pytest.raises(errors.DeviceError):
        train.PipelineConfig(device="tpu").validate()
    with pytest.raises(errors.DeviceError):
        train.PipelineConfig(dtype="float32x").validate()


def test_train_pipeline_learns(trained: train.PipelineReport) -> None:
    """端到端训练真的在学：最终训练损失低于初始损失、下降比例为正。"""
    assert trained.final_train_loss < trained.initial_loss
    assert trained.improvement > 0.5
    assert trained.epochs_run == 20
    assert trained.batch_count == 3
    assert trained.steps == trained.batch_count * trained.epochs_run
    assert len(trained.history) == trained.epochs_run
    assert trained.device_plan.fits is True
    assert "训练" in trained.summary_line()


def test_train_pipeline_report_lines(trained: train.PipelineReport) -> None:
    """报告的派生读数：最好评估损失 <= 最终评估损失、逐行输出含设备与指纹。"""
    assert trained.best_eval_loss <= trained.final_eval_loss
    assert trained.dropped_samples == 0
    assert len(trained.dataset_fingerprint) == 16
    lines = trained.lines()
    assert "设备：" in lines[1]
    assert len(lines) == 3 + trained.epochs_run


def test_train_pipeline_is_reproducible(sign_dataset) -> None:
    """同一配置两次训练给出同一批损失。"""
    config = train.PipelineConfig(epochs=3, batch_size=8, seed=5)
    first = train.train_pipeline(sign_dataset, config=config)
    second = train.train_pipeline(sign_dataset, config=config)
    assert [record.train_loss for record in first.history] == [
        record.train_loss for record in second.history
    ]
    assert first.final_eval_loss == second.final_eval_loss


def test_compare_configs_readings_differ(sign_dataset) -> None:
    """``compare_configs`` 两个变体的读数**不同**（丢尾批会少一批、丢 3 条）。"""
    variants = {
        "保留": replace(study.ABLATION_VARIANTS["基准"], epochs=3),
        "丢尾": replace(study.ABLATION_VARIANTS["丢尾批"], epochs=3),
    }
    reports = train.compare_configs(sign_dataset, variants)
    assert set(reports) == {"保留", "丢尾"}
    assert reports["保留"].dropped_samples == 0
    assert reports["丢尾"].dropped_samples == 3
    assert reports["保留"].steps > reports["丢尾"].steps
    with pytest.raises(errors.ParameterError):
        train.compare_configs(sign_dataset, {})


def test_resume_matches_uninterrupted_is_reused() -> None:
    """第 ⑥ 条性质：**直接复用 verify 的函数**（这是本课最值钱的一条）。"""
    outcome = verify.check_resume_matches_uninterrupted()
    assert outcome.name == types.PROPERTY_RESUME_MATCHES_UNINTERRUPTED
    assert outcome.passed is True
    assert outcome.check.reading == 0.0


def test_resume_training_with_sgd(sign_dataset) -> None:
    """用无状态优化器恢复：训到第 4 轮，历史长度与轮数一致。"""
    config = train.PipelineConfig(epochs=4, batch_size=8, optimizer="sgd", seed=7)
    with tempfile.TemporaryDirectory() as directory:
        train.train_pipeline(
            sign_dataset, config=replace(config, epochs=2), checkpoint_dir=directory
        )
        report = train.resume_training(sign_dataset, checkpoint_dir=directory, config=config)
    assert report.epochs_run == 4
    assert len(report.history) == 4
    assert report.steps == report.batch_count * 4


def test_resume_training_guards(sign_dataset, pipeline_model) -> None:
    """恢复训练的多类护栏：优化器不符 / 非 dict / 未知名 / 参数个数不符 / running 宽度不符。"""
    sgd = make_train_optimizer("sgd", 0.05)
    adam = make_train_optimizer("adam", 0.05)
    with pytest.raises(errors.CheckpointError):
        train._restore_optimizer(adam, "payload")
    with pytest.raises(errors.CheckpointError):
        train._restore_optimizer(adam, {"name": "nope"})
    nesterov = make_train_optimizer("nesterov", 0.05)
    train._restore_optimizer(
        nesterov, {"name": "nesterov", "step_count": 2, "velocity": [0.1, 0.2]}
    )
    assert nesterov.velocity == (0.1, 0.2)
    train._restore_optimizer(adam, {"name": "adam", "step_count": 3})
    assert adam.step_count == 3
    with tempfile.TemporaryDirectory() as directory:
        checkpoint.save_checkpoint(
            directory,
            params=pipeline_model,
            optimizer_state=sgd.state(),
            step=0,
            metrics={"epoch": 0, "initial_loss": 0.0},
            history=(),
        )
        with pytest.raises(errors.CheckpointError):
            train.resume_training(
                sign_dataset,
                checkpoint_dir=directory,
                config=train.PipelineConfig(optimizer="adam"),
            )
    tiny = build_regularized("lstm", input_size=1, hidden_size=3, classes=2, seed=5)
    with tempfile.TemporaryDirectory() as directory:
        checkpoint.save_checkpoint(
            directory,
            params=tiny,
            optimizer_state=sgd.state(),
            step=0,
            metrics={"epoch": 0, "initial_loss": 0.0},
            history=(),
        )
        with pytest.raises(errors.ShapeError):
            train.resume_training(
                sign_dataset, checkpoint_dir=directory, config=train.PipelineConfig(epochs=2)
            )
    with tempfile.TemporaryDirectory() as directory:
        checkpoint.save_checkpoint(
            directory,
            params=pipeline_model,
            optimizer_state=sgd.state(),
            step=0,
            metrics={"epoch": 0, "initial_loss": 0.0},
            history=(),
        )
        report = train.resume_training(
            sign_dataset,
            checkpoint_dir=directory,
            config=train.PipelineConfig(epochs=1, optimizer="sgd"),
        )
        assert report.epochs_run == 1
    with tempfile.TemporaryDirectory() as directory:
        checkpoint.save_checkpoint(
            directory,
            params=pipeline_model,
            optimizer_state=sgd.state(),
            step=0,
            metrics={
                "epoch": 0,
                "initial_loss": 0.0,
                "running": {"mean": [0.0], "variance": [1.0], "batches": 1},
            },
            history=(),
        )
        with pytest.raises(errors.ShapeError):
            train.resume_training(
                sign_dataset,
                checkpoint_dir=directory,
                config=train.PipelineConfig(epochs=1, optimizer="sgd"),
            )


def test_train_pipeline_guards(sign_dataset) -> None:
    """非数据集 / 非配置 / 非模型 / 目录不存在 / 恢复参数类型五类护栏。"""
    with pytest.raises(errors.ParameterError):
        train.train_pipeline("dataset")
    with pytest.raises(errors.ParameterError):
        train.train_pipeline(sign_dataset, config="config")
    with pytest.raises(errors.ParameterError):
        train.train_pipeline(sign_dataset, config=train.PipelineConfig(epochs=1), model="model")
    with pytest.raises(errors.ParameterError):
        train.resume_training(
            "dataset", checkpoint_dir="x", config=train.PipelineConfig(epochs=1)
        )
    with pytest.raises(errors.ParameterError):
        train.resume_training(sign_dataset, checkpoint_dir="x", config="config")
    with pytest.raises(errors.CheckpointError):
        train.resume_training(
            sign_dataset,
            checkpoint_dir=str(Path(tempfile.gettempdir()) / "definitely-not-here"),
            config=train.PipelineConfig(epochs=1),
        )


def test_evaluate_readings(sign_dataset, pipeline_model) -> None:
    """``evaluate`` 给出一对读数（推理相）。"""
    loss, accuracy = train.evaluate(
        pipeline_model,
        sign_dataset,
        running=initial_running(pipeline_model.features),
        batch_size=8,
    )
    assert loss > 0.0
    assert 0.0 <= accuracy <= 1.0


def test_pipeline_config_no_split_and_early_stop(sign_dataset) -> None:
    """``eval_ratio=0`` 时不切分；``early_stop_patience`` 到点会**提前结束**。"""
    config = train.PipelineConfig(epochs=2, batch_size=8, eval_ratio=0.0, seed=3)
    report = train.train_pipeline(sign_dataset, config=config)
    assert report.epochs_run == 2
    assert math.isfinite(report.final_eval_loss)
    stopped = train.train_pipeline(
        sign_dataset,
        config=train.PipelineConfig(epochs=40, batch_size=8, seed=100, early_stop_patience=1),
    )
    assert stopped.epochs_run < 40
    assert len(stopped.history) == stopped.epochs_run


# --------------------------------------------------------------------------- 性质校验


def test_check_three_judgement_types() -> None:
    """``Check`` 的三类判据：相等 / 上界 / 下界。"""
    assert verify.Check(reading=0.0, upper_bound=0.0).passed() is True
    assert verify.Check(reading=1e-13, upper_bound=0.0).passed() is False
    assert verify.Check(reading=0.9, lower_bound=0.5).passed() is True
    assert verify.Check(reading=0.1, lower_bound=0.5).passed() is False
    assert verify.Check(reading=0.0).bound_text() == "== 逐位"
    assert verify.Check(reading=0.0, upper_bound=0.0).bound_text() == "== 0"
    assert verify.Check(reading=0.0, upper_bound=1e-7).bound_text() == "<= 1.0e-07"
    assert verify.Check(reading=0.0, lower_bound=0.5).bound_text() == ">= 5.0e-01"
    assert (
        verify.Check(reading=0.0, lower_bound=0.1, upper_bound=0.9).bound_text()
        == "∈ [1.0e-01, 9.0e-01]"
    )
    assert verify._max_abs_gap((1.0,), (1.0, 2.0)) == math.inf
    assert verify._max_abs_gap((1.0, 2.0), (1.0, 2.0)) == 0.0
    assert verify._max_abs_gap((), ()) == 0.0
    assert verify._max_abs_gap(((1.0,),), ((3.0,),), rows=True) == 2.0


def test_check_all_is_seven_and_all_pass(property_report: verify.PropertyReport) -> None:
    """七条性质全部通过（这就是"这条链被真的接上了"）。"""
    assert property_report.total == 7
    assert property_report.passed == 7
    assert property_report.all_passed()
    assert len(property_report.lines()) == 7


def test_each_check_returns_passing_outcome() -> None:
    """逐条跑：每条都返回通过的 ``PropertyOutcome`` 且读数有限。"""
    for name, function in verify.CHECKS.items():
        outcome = function()
        assert outcome.name == name
        assert outcome.passed is True
        assert math.isfinite(outcome.check.reading)
        assert outcome.line().startswith("通过")


def test_check_judgement_types_cover_three_kinds(property_report: verify.PropertyReport) -> None:
    """七条性质里同时存在相等 / 上界 / 下界三类判据。"""
    kinds = set()
    for outcome in property_report.outcomes:
        check = outcome.check
        if check.lower_bound is not None:
            kinds.add("lower")
        elif check.upper_bound == 0.0:
            kinds.add("equal")
        else:
            kinds.add("upper")
    assert kinds == {"equal", "upper", "lower"}


# --------------------------------------------------------------------------- 六张表


def test_dataset_rows_three(sign_dataset) -> None:
    """数据集表三行（全集 / 训练 / 评估），指纹各不相同。"""
    rows = study.dataset_rows(sign_dataset)
    assert [row.name for row in rows] == ["sign", "sign/train", "sign/eval"]
    assert rows[0].size == 32
    assert rows[0].positives + rows[0].negatives == rows[0].size
    assert len({row.fingerprint for row in rows}) == 3
    assert all("指纹" in row.line() for row in rows)


def test_batch_rows_drop_last_differs(sign_dataset) -> None:
    """批次表两行，``drop_last`` 的两个分支给出不同批数与丢弃数。"""
    rows = study.batch_rows(sign_dataset)
    assert [row.drop_last for row in rows] == [False, True]
    kept, dropped = rows
    assert kept.batch_count == math.ceil(32 / study.BATCH_TABLE_SIZE)
    assert dropped.batch_count == 32 // study.BATCH_TABLE_SIZE
    assert dropped.dropped == 32 - study.BATCH_TABLE_SIZE * dropped.batch_count
    assert kept.batch_count != dropped.batch_count


def test_device_rows_three() -> None:
    """设备表三行（cpu / cuda / mps），字节数一致、预算不同。"""
    rows = study.device_rows()
    assert [row.device for row in rows] == list(types.DEVICE_KINDS)
    assert len({row.total_bytes for row in rows}) == 1
    assert len({row.budget_bytes for row in rows}) == 3
    assert all(row.fits for row in rows)


def test_checkpoint_rows_five(tmp_path: Path) -> None:
    """检查点表五行，五个文件都在场；第二次调用走"已存在"分支。"""
    rows = study.checkpoint_rows(tmp_path)
    assert [row.file for row in rows] == list(checkpoint.CHECKPOINT_FILES)
    assert all(row.present for row in rows)
    assert all(row.size_bytes > 0 for row in rows if row.file != "history.jsonl")
    again = study.checkpoint_rows(tmp_path)
    assert [row.size_bytes for row in again] == [row.size_bytes for row in rows]


def test_inference_rows_three(sign_dataset) -> None:
    """推理表三行：样本数都等于数据集大小，批越大调用越少。"""
    rows = study.inference_rows(sign_dataset)
    assert [row.batch_size for row in rows] == list(study.INFERENCE_BATCH_SIZES)
    assert all(row.samples == len(sign_dataset) for row in rows)
    assert rows[0].calls == len(sign_dataset)
    assert rows[-1].calls == len(sign_dataset) // study.INFERENCE_BATCH_SIZES[-1]
    assert all(row.seconds_per_sample > 0.0 for row in rows)


def test_ablation_rows_use_injected_reports(ablation) -> None:
    """消融表的行来自**注入**的报告；默认路径会真的训三个变体。"""
    rows = study.ablation_rows(ablation)
    assert {row.variant for row in rows} == {"基准", "丢尾批"}
    dropped = next(row for row in rows if row.variant == "丢尾批")
    assert dropped.dropped == 3
    assert all("训练" in row.line() and "推理" in row.line() for row in rows)
    computed = study.ablation_reports()
    assert set(computed) == set(study.ABLATION_VARIANTS)
    assert len(study.ablation_rows(computed)) == len(study.ABLATION_VARIANTS)


def test_pipeline_lines_seven_stages() -> None:
    """阶段线：七个阶段、每个阶段两行；``PROPERTY_NAMES`` 就是性质名单。"""
    lines = study.pipeline_lines()
    assert len(lines) == 2 * len(types.STAGES)
    text = "\n".join(lines)
    for stage in types.STAGES:
        assert stage in text
    assert "由谁实现" in text
    assert study.PROPERTY_NAMES == types.PIPELINE_PROPERTIES
    assert study.FRESH_DEVICE == types.DEVICE_CPU


def test_property_rows_are_seven_and_pass() -> None:
    """性质表七行，全部通过。"""
    rows = study.property_rows()
    assert len(rows) == 7
    assert all(row.passed for row in rows)
    assert all("vs" in row.cross_check for row in rows)


def test_study_lines_has_sections(ablation) -> None:
    """一次跑完六张表 + 阶段线：八个小节标题都在。"""
    text = "\n".join(study.study_lines(ablation))
    for index in range(1, 9):
        assert f"== {index}." in text


# --------------------------------------------------------------------------- 包入口


def test_package_all_is_non_empty_and_has_no_submodules() -> None:
    """包入口的 ``__all__`` 非空、且不含子模块名。"""
    import smart_research_agent.torch_pipeline as package

    assert len(package.__all__) > 100
    assert not (
        {
            "errors",
            "types",
            "datasets",
            "sampler",
            "dataloader",
            "device",
            "checkpoint",
            "inference",
            "train",
            "verify",
            "study",
        }
        & set(package.__all__)
    )


def test_package_star_import_resolves_every_name() -> None:
    """每个 ``__all__`` 里的名字都能在包命名空间里取到。"""
    import smart_research_agent.torch_pipeline as package

    assert [name for name in package.__all__ if not hasattr(package, name)] == []
    assert package.__all__ == sorted(package.__all__)


def test_package_exposes_key_objects() -> None:
    """几个关键对象从包入口就能取到。"""
    import smart_research_agent.torch_pipeline as package

    assert package.TabularDataset is datasets.TabularDataset
    assert package.DataLoader is dataloader.DataLoader
    assert package.plan_device is device.plan_device
    assert package.save_checkpoint is checkpoint.save_checkpoint
    assert package.predict_one is inference.predict_one
    assert package.train_pipeline is train.train_pipeline
    assert package.Check is verify.Check
    assert package.tensor_bytes is device.tensor_bytes
