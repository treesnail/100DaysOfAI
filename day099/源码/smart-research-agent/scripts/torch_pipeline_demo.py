"""day096 演示脚本：PyTorch 高级与综合实践 —— 数据 / 设备 / 检查点 / 推理（十一节）.

全部离线、全部确定性：不需要 API Key，不依赖 torch / numpy。
跑法::

    cd day096/源码/smart-research-agent
    python scripts/torch_pipeline_demo.py

十一节对应教程的十一章；打印的读数与 ``tests/test_torch_pipeline.py`` 断言的是同一批。
产出 ``outputs/torch_pipeline_demo.txt``（在 .gitignore 里）。
"""

from __future__ import annotations

import contextlib
import io
import pathlib
import sys
import tempfile

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.optimizers.optimizer import make_train_optimizer  # noqa: E402
from smart_research_agent.regularization.normalization import initial_running  # noqa: E402
from smart_research_agent.regularization.network import build_regularized, flatten_params  # noqa: E402
from smart_research_agent.sequence_models.train import make_sign_dataset  # noqa: E402
from smart_research_agent.torch_pipeline import (  # noqa: E402
    checkpoint,
    dataloader,
    datasets,
    device,
    inference,
    sampler,
    study,
    train,
    types,
    verify,
)

OUTPUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "torch_pipeline_demo.txt"

SEP = "=" * 72

#: 演示用的小网络（宽度 4 的 lstm，与 ``train._build_model`` 同构）.
DEMO_HIDDEN = train.DEFAULT_HIDDEN_SIZE
DEMO_BATCH_SIZE = 8
DEMO_SEED = 100


def section(index: int, title: str) -> None:
    """打印一节的小标题."""
    print()
    print(SEP)
    print(f"[{index}] {title}")
    print(SEP)


def demo_model():
    """造一份演示用的小网络（**不重写网络**：调用 day095 的构造器）."""
    return build_regularized(
        train.DEFAULT_CELL, input_size=1, hidden_size=DEMO_HIDDEN, classes=2, seed=DEMO_SEED
    )


def main() -> None:
    dataset = datasets.make_dataset(make_sign_dataset(5), name="sign")

    section(1, "数据集画像：不可变样本 + 内容指纹")
    for row in study.dataset_rows(dataset):
        print("  " + row.line())
    print(f"  指纹算法：{types.PIPELINE_NOTES['fingerprint']}")

    section(2, "切分：只依赖 (seed, n) 的确定性切分，两侧都非空")
    train_dataset, eval_dataset = datasets.split_dataset(dataset, eval_ratio=0.25, seed=DEMO_SEED)
    print(f"  训练集：{train_dataset.line()}")
    print(f"  评估集：{eval_dataset.line()}")
    print("  口径：切点 = round(n × eval_ratio) 再夹到 [1, n−1]（两侧都非空，打乱用 LCG）")

    section(3, "DataLoader 的三个读数：批数 / 丢弃 / 尾批")
    for drop_last in (False, True):
        loader = dataloader.DataLoader(
            train_dataset, 7, shuffle=True, drop_last=drop_last, seed=DEMO_SEED
        )
        print(f"  {loader.line()}")
    loader = dataloader.DataLoader(train_dataset, 7, shuffle=True, seed=DEMO_SEED)
    print(f"  第 1 轮第 1 批：{loader.batches(1)[0].line()}")
    print(f"  批数公式：{types.BATCH_COUNT_FORMULA}")

    section(4, "工作进程分片：互不相交、并集为全集（下界判据）")
    shards = [sampler.shard_indices(len(train_dataset), workers=4, worker_id=k) for k in range(4)]
    union = set().union(*shards)
    for index, shard in enumerate(shards):
        print(f"  worker {index}：{len(shard)} 条 | 前 6 个下标 {shard[:6]}")
    print(f"  并集覆盖 {len(union) / len(train_dataset):.1f}，重叠项数 {sum(map(len, shards)) - len(union)}")
    for worker_id in range(2):
        worker_loader = dataloader.DataLoader(
            train_dataset, 4, shuffle=False, workers=2, worker_id=worker_id
        )
        print(f"  {worker_loader.line()}")

    section(5, "设备与显存的算术：三块字节数 + fits")
    model = demo_model()
    print(f"  结构：{model.line()}")
    for kind in types.DEVICE_KINDS:
        plan = device.plan_device(
            model.parameter_count,
            batch_size=DEMO_BATCH_SIZE,
            width=model.features,
            device=kind,
        )
        print(f"  {plan.line()}")
        print(f"    {plan.warn_line()}")
    tight = device.plan_device(
        model.parameter_count,
        batch_size=DEMO_BATCH_SIZE,
        width=model.features,
        device="cuda",
        budget_bytes=1024,
    )
    print(f"  人为压小预算：{tight.line()} | {tight.warn_line()}")
    print(f"  {device.move_report(tight)[0]}")
    print(f"  总字节公式：{types.TOTAL_BYTES_FORMULA}")

    section(6, "检查点五件套：落盘 / 校验 / 读回")
    with tempfile.TemporaryDirectory() as directory:
        manifest = checkpoint.save_checkpoint(
            directory,
            params=model,
            optimizer_state=make_train_optimizer("adam", 0.05).state(),
            step=0,
            metrics={"epoch": 0, "step": 0},
            history=(),
        )
        print(f"  清单：{manifest.line()}")
        for row in study.checkpoint_rows(directory):
            print("  " + row.line())
        loaded = checkpoint.load_checkpoint(directory)
        checkpoint.verify_checkpoint(directory, loaded.manifest)
        print(f"  读回：{loaded.line()} | 恢复起点 step = {checkpoint.resume_step(loaded)}")
        print(f"  权重与读回的逐位差：{max(abs(a - b) for a, b in zip(flatten_params(model), loaded.params))}")
    print(f"  五件套口径：{types.CHECKPOINT_FILES_FORMULA}")

    section(7, "恢复训练为什么必须逐位相等")
    config_full = train.PipelineConfig(
        epochs=6, batch_size=DEMO_BATCH_SIZE, seed=11, eval_ratio=0.25
    )
    config_mid = train.PipelineConfig(
        epochs=3, batch_size=DEMO_BATCH_SIZE, seed=11, eval_ratio=0.25
    )
    with tempfile.TemporaryDirectory() as full_dir, tempfile.TemporaryDirectory() as mid_dir:
        train.train_pipeline(dataset, config=config_full, checkpoint_dir=full_dir)
        train.train_pipeline(dataset, config=config_mid, checkpoint_dir=mid_dir)
        train.resume_training(dataset, checkpoint_dir=mid_dir, config=config_full)
        full_params = checkpoint.load_checkpoint(full_dir).params
        resumed_params = checkpoint.load_checkpoint(mid_dir).params
    print(f"  一口气训到第 6 轮：{len(full_params)} 个参数")
    print(f"  第 3 轮落盘后恢复：{len(resumed_params)} 个参数")
    print(f"  最大绝对差：{max(abs(a - b) for a, b in zip(full_params, resumed_params))}")
    print(f"  口径：{types.RESUME_FORMULA}")

    section(8, "推理封装：只走推理相 + 可注入时钟")
    engine = inference.build_inference_engine(
        model, running=initial_running(model.features), clock=inference.stepping_clock()
    )
    labels = inference.predict_dataset(engine, dataset, batch_size=DEMO_BATCH_SIZE)
    print(f"  {engine.line()}")
    print(f"  {inference.latency_report(engine).line()}")
    for row in study.inference_rows(dataset):
        print("  " + row.line())
    print(f"  延迟口径：{types.LATENCY_FORMULA}")

    section(9, "端到端训练：数据 → 设备 → 训练 → 报告")
    report = train.train_pipeline(
        dataset, config=train.PipelineConfig(epochs=10, batch_size=DEMO_BATCH_SIZE, seed=DEMO_SEED)
    )
    for line in report.lines()[:5]:
        print("  " + line)
    print(f"  ...（共 {len(report.lines())} 行：汇总 + 设备 + 指纹 + {report.epochs_run} 轮）")

    section(10, "七条性质：相等 / 上界 / 下界")
    for row in study.property_rows():
        print("  " + row.line())
    outcome = verify.check_all()
    print(f"  合计：{outcome.passed}/{outcome.total} 条通过")

    section(11, "结论表：消融读数、阶段线与五条边界")
    for row in study.ablation_rows():
        print("  " + row.line())
    print("  阶段线：")
    for line in study.pipeline_lines():
        print("    " + line)
    print("  五条边界：")
    for index, boundary in enumerate(types.PIPELINE_BOUNDARIES, start=1):
        print(f"    {index}. {boundary}")


if __name__ == "__main__":
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        main()
    rendered = buffer.getvalue()
    print(rendered)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(rendered, encoding="utf-8")
    print(f"\n（已写入 {OUTPUT_FILE}）")
