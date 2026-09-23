"""离线演示：Transformer 之前的数学地基（day073 / Math-D1）.

跑法：

```bash
python scripts/math_foundations_demo.py      # 十一节，输出写到 outputs/math_foundations_demo.txt
```

全部离线：纯 Python 算术（不用 numpy），零网络、零 API Key。
产物只写在 ``outputs/`` 下（幂等，可随时删）。

十一节里最值得看的是第 4、8、9、10、11 节：

```text
4    softmax：把打分变成概率，且 z = [1000, 1001] 不会炸
8    幂迭代与低秩近似：A_r = Σ σ u vᵀ —— LoRA（day051）的数学原型
9    注意力：softmax(QKᵀ/√d)·V，因果掩码让"看不到未来"变成**权重恰好为 0**
10   为什么要除以 √d_k：把方差随维度增长这件事**量出来**（缩放后回到常数）
11   与项目既有实现的八项对照：七项逐位一致、一项是有意记录的约定差异
```
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.math_foundations import (  # noqa: E402
    ATTENTION_VARIANT_DESCRIPTIONS,
    LINALG_OPS,
    LINALG_OP_FORMULAS,
    Distribution,
    JointTable,
    cosine,
    cross_check_all,
    cross_entropy,
    dot,
    entropy,
    kl_divergence,
    low_rank_approximation,
    multi_head_attention,
    norm,
    normalize,
    positional_encoding,
    power_iteration,
    projection,
    reconstruction_error,
    sample_index,
    sample_outputs,
    sampled_dot_product_variance,
    scaled_dot_product_attention,
    softmax,
    uniforms,
    validate_matrix,
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_NAME = "math_foundations_demo.txt"

#: 演示里用到的向量（都能手算）。
A = (1.0, 2.0, 3.0)
A_SCALED = (2.0, 4.0, 6.0)
ORTHOGONAL = (0.0, 3.0, -2.0)
E1 = (1.0, 0.0, 0.0)

#: 对角矩阵（奇异值就是对角线，手算可验证）。
DIAGONAL = ((2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.5))


class _Tee:
    """把 stdout 同时写终端与文件（与 scripts/indexing_demo.py 同一手法）."""

    def __init__(self, stream, handle) -> None:
        self._stream = stream
        self._handle = handle

    def write(self, text: str) -> int:
        """两边都写，返回写到终端的字符数."""
        self._handle.write(text)
        return self._stream.write(text)

    def flush(self) -> None:
        """两边都刷."""
        self._handle.flush()
        self._stream.flush()


def section(title: str) -> None:
    """打印一节的分隔标题."""
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def show_vector(label: str, vector) -> None:
    """按固定小数位打印一个向量（保留 6 位，便于逐位核对）."""
    print(f"  {label:<22} {tuple(round(value, 6) for value in vector)}")


def main() -> None:
    """十一节演示."""
    # ------------------------------------------------------------------ 第 1 节
    section("第 1 节：一个向量的四种读法（点积 / 模长 / 余弦 / 投影）")
    print(f"a = {A}   b = {A_SCALED}   c = {ORTHOGONAL}（与 a 正交）")
    print(f"  a·b                  {dot(A, A_SCALED):<20.6f}  两个向量逐分量相乘再相加")
    print(f"  ‖a‖                  {norm(A):<20.6f}  长度（√14）")
    print(f"  cos(a, b)            {cosine(A, A_SCALED):<20.6f}  同向 → 1（与长度无关）")
    print(f"  cos(a, c)            {cosine(A, ORTHOGONAL):<20.6f}  正交 → 0")
    show_vector("â = a/‖a‖", normalize(A))
    print(f"  ‖â‖                  {norm(normalize(A)):<20.6f}  归一化之后恒为 1")
    print(f"  proj_e1(a)           {projection(A, E1):<20.6f}  投到 e1 上就是第一个分量")
    print()
    print("  十个算子与它们的公式（口径可读，见 types.LINALG_OP_FORMULAS）：")
    for name in LINALG_OPS:
        print(f"    {name:<18} {LINALG_OP_FORMULAS[name]}")

    # ------------------------------------------------------------------ 第 2 节
    section("第 2 节：零向量上的三种约定（同一样东西，三种正确行为）")
    print("  本包 normalize(零向量)          → 报错（零向量没有方向）")
    try:
        normalize((0.0, 0.0))
    except Exception as exc:
        print(f"                                    {type(exc).__name__}: {str(exc)[:60]}…")
    print("  vectorstore.metrics.cosine      → 0.0（让一条脏数据不打断整批查询）")
    print(
        f"                                    cosine(零向量, a) = "
        f"{cosine((0.0, 0.0, 0.0), A):.6f}"
    )
    print("  llm.embedding.l2_normalize      → 原样返回（写入侧不能因此抛异常）")
    from smart_research_agent.llm.embedding import l2_normalize

    print(f"                                    l2_normalize([0, 0]) = {l2_normalize([0.0, 0.0])}")
    print("  三种都不算错——但**必须各自说清为什么**（day073 教程第二章）")

    # ------------------------------------------------------------------ 第 3 节
    section("第 3 节：矩阵是线性变换（转置 / 乘法 / 单位矩阵）")
    left = ((1.0, 2.0), (3.0, 4.0))
    right = ((5.0, 6.0), (7.0, 8.0))
    print(f"  A·B                  {validate_matrix(left) and _matmul_str(left, right)}")
    print("  手算：第 1 行第 1 列 = 1·5 + 2·7 = 19")
    print(f"  单位矩阵是“什么都不做”： A·I = A → {left}")

    # ------------------------------------------------------------------ 第 4 节
    section("第 4 节：softmax —— 把打分变成概率（且不会炸）")
    print(f"  softmax([0, 0])          {_round(softmax((0.0, 0.0)))}   平分")
    print(f"  softmax([1, 2, 3])       {_round(softmax((1.0, 2.0, 3.0)))}   越大的分拿到越大权重")
    print(f"  softmax([1000, 1001])    {_round(softmax((1000.0, 1001.0)))}   ← 直接 exp 会溢出成 nan")
    print(f"  softmax([1, 2], T=0.01)  {_round(softmax((1.0, 2.0), temperature=0.01))}   低温 → 趋近 one-hot")
    print(f"  softmax([1, 2], T=100)   {_round(softmax((1.0, 2.0), temperature=100.0))}   高温 → 趋近均匀")
    print("  稳定写法：减去最大值（不改变结果，只把 exp 的自变量压到 <= 0）")

    # ------------------------------------------------------------------ 第 5 节
    section("第 5 节：熵、交叉熵与 KL（单位一律 nats）")
    dist = Distribution.from_logits((1.0, 2.0, 3.0), name="next-token")
    flat = Distribution.uniform(3, name="uniform")
    print(f"  预测分布 {dist.summary_line()}")
    print(f"  均匀分布 {flat.summary_line()}")
    print(f"  H(预测)              {entropy(dist.probabilities):.6f}")
    print(f"  H(预测, 均匀)         {cross_entropy(dist.probabilities, flat.probabilities):.6f}  ← 交叉熵 ≥ 熵")
    print(f"  KL(预测‖均匀)         {kl_divergence(dist.probabilities, flat.probabilities):.6f}")
    print(f"  KL(均匀‖预测)         {kl_divergence(flat.probabilities, dist.probabilities):.6f}  ← 非对称")
    print("  H(p, p) = H(p)；KL(p, p) = 0；KL ≥ 0（Gibbs 不等式）")

    # ------------------------------------------------------------------ 第 6 节
    section("第 6 节：联合分布、条件概率与贝叶斯")
    table = JointTable(
        cells=((0.30, 0.10), (0.20, 0.40)),
        names_a=("下雨", "不下雨"),
        names_b=("带伞", "不带伞"),
    )
    print(f"  {table.summary_line()}")
    print(f"  P(A)                 {_round(table.marginal_a().probabilities)}")
    print(f"  P(B)                 {_round(table.marginal_b().probabilities)}")
    print(f"  P(带伞|下雨)          {_round(table.conditional_b_given_a(0).probabilities)}")
    print(f"  P(下雨|带伞)          {_round(table.posterior_a_given_b(0).probabilities)}  ← 贝叶斯后验")
    print(
        f"  与列条件分布一致："
        f"{table.conditional_a_given_b(0).probabilities == table.posterior_a_given_b(0).probabilities}"
    )
    print(f"  互信息 I(A;B)        {table.mutual_information():.6f} nats（独立时为 0）")
    print(f"  与独立的最大偏差      {table.independence_gap():.6f}")

    # ------------------------------------------------------------------ 第 7 节
    section("第 7 节：采样 —— 累积区间与可复现的随机数")
    probabilities = (0.5, 0.3, 0.2)
    print(f"  概率 {probabilities} → 段边界 0.5 | 0.8 | 1.0")
    for u in (0.0, 0.42, 0.5, 0.79, 0.99):
        print(f"    u = {u:<5} → 采到第 {sample_index(probabilities, u=u)} 个事件")
    draws = [sample_index(probabilities, u=value) for value in uniforms(1000, seed=42)]
    frequency = [draws.count(index) / len(draws) for index in range(3)]
    print(f"  用确定性随机数采 1000 次：{_round(tuple(frequency))}（应当接近上面的概率）")
    print("  同一个种子永远同一串数 → 这条结论可以被逐位复现")

    # ------------------------------------------------------------------ 第 8 节
    section("第 8 节：幂迭代与低秩近似（LoRA 的数学原型）")
    print(f"  对角矩阵 diag(2, 1, 0.5) 的奇异值就是对角线：")
    sigma, u, v = power_iteration(DIAGONAL)
    print(f"    σ1 = {sigma:.6f}   u = {_round(u)}   v = {_round(v)}")
    values, _, _, approximate = low_rank_approximation(DIAGONAL, 2)
    print(f"    秩 2 提取到的奇异值 {_round(values)}")
    for rank in (1, 2, 3):
        approximation = low_rank_approximation(DIAGONAL, rank)[3]
        error = reconstruction_error(DIAGONAL, approximation)
        print(f"    秩 {rank} 近似的相对误差  {error:.6f}")
    print("  A_r = Σ_{i≤r} σ_i u_i v_iᵀ：只留前 r 个方向 → 参数从 n² 降到 2nr")
    print("  day051 的 LoRA（r=8 时只训 2·8·d 个参数）就是这件事的工程实现")

    # ------------------------------------------------------------------ 第 9 节
    section("第 9 节：注意力 —— softmax(QKᵀ/√d_k)·V")
    queries, keys, values_matrix = sample_outputs()
    report = scaled_dot_product_attention(queries, keys, values_matrix)
    print(f"  变体：{report.variant} —— {ATTENTION_VARIANT_DESCRIPTIONS[report.variant]}")
    print(f"  {report.summary_line()}")
    for row, weights_row in enumerate(report.weights):
        print(f"    第 {row} 行权重 {_round(weights_row)}  峰值 {report.peak_weights[row]:.4f}"
              f" @ 位置 {report.peak_indices[row]}  H={report.entropies[row]:.4f}")
    causal = scaled_dot_product_attention(queries, keys, values_matrix, causal=True)
    print()
    print(f"  因果版（{causal.summary_line()}）")
    for row, weights_row in enumerate(causal.weights):
        print(f"    第 {row} 行权重 {_round(weights_row)}   ← 上三角恰好是 0.0")
    multi = multi_head_attention(queries, keys, values_matrix, heads=2)
    print()
    print(f"  多头：{multi.summary_line()}")
    print(f"    各头缩放系数 {[report.scale for report in multi.reports]}（每头 d_k = 2 → 1/√2）")
    print("  位置编码（前 4 个位置 × 前 4 维）：")
    for position, row in enumerate(positional_encoding(4, 4)):
        print(f"    pos={position}  {_round(row)}")

    # ------------------------------------------------------------------ 第 10 节
    section("第 10 节：为什么要除以 √d_k（把方差量出来）")
    print("  分量独立同分布（[-1,1] 均匀，方差 1/3）时，q·k = Σ q_i k_i 的方差是 d/9：")
    for dimension in (4, 16, 64, 256):
        study = sampled_dot_product_variance(dimension, trials=2000, seed=42)
        print(f"    {study.summary_line()}")
    print("  看最后两列：**未缩放**的标准差随 √d 增长（0.67 → 5.34），")
    print("  而**缩放后**恒定在 1/3 —— 这就是 softmax 不会在长向量上饱和的原因。")

    # ------------------------------------------------------------------ 第 11 节
    section("第 11 节：与项目既有实现的八项对照")
    bridge = cross_check_all()
    print(f"  {bridge.summary_line()}")
    for outcome in bridge.outcomes:
        print(f"    {outcome.summary_line()}")
        print(f"        {outcome.message}")
        print(f"        对照来源：{outcome.source}")
    print()
    print("  结论：七项逐位一致、一项（零向量归一化）是**有意记录**的约定差异、零项分歧。")
    print("  危险的不是“不一样”，而是“不一样而没人知道”。")


def _round(values) -> tuple:
    """把一串浮点压成 6 位小数（打印与逐位核对都用它）."""
    return tuple(round(value, 6) for value in values)


def _matmul_str(left, right) -> str:
    """打印两个小矩阵相乘的结果（演示用）."""
    from smart_research_agent.math_foundations import matmul

    return str(matmul(left, right))


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / OUTPUT_NAME).open("w", encoding="utf-8") as handle:
        original = sys.stdout
        sys.stdout = _Tee(original, handle)
        try:
            main()
        finally:
            sys.stdout = original
    print(f"\n输出已写入 {OUTPUT_DIR / OUTPUT_NAME}")
