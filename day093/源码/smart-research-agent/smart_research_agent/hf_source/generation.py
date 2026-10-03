"""生成策略：**四个 warper 与两种搜索**（day085 / M7-D9）.

`GenerationMixin` 的那段代码是"读源码"最容易读错的一段，因为它把两件不同的事
放在同一个循环里：

```text
① 对 logits 做**一连串纯函数变换**（warper / processor）——不改模型、不碰缓存
② 从变换后的分布里**选一个 token**——贪心、或采样、或维护一批 beam
```

本模块把这两件事拆开，因此"换策略"与"换模型"可以分别被断言。

## 四个 warper 的顺序（顺序即语义）

```text
RepetitionPenaltyLogitsProcessor   score = score < 0 ? score·penalty : score/penalty
TemperatureLogitsWarper            score = score / temperature
TopKLogitsWarper                   只留最大的 k 个，其余 -inf
TopPLogitsWarper                   升序排序 + 累积 softmax + `cumsum <= 1 - p` 的删除条件 + 至少留 1 个
```

这一串的**每一个**都是纯函数：给定同样的输入，输出逐位相同。
因此"策略"这件事是可复现的——本包的采样只额外需要**一串可注入的均匀数**
（走 day073 的 LCG），于是"同一个种子给出同样的文本"是一条设计目标，而不是运气。

## 三处最容易读错的地方

```text
① 重复惩罚是**除**不是减     正 logits 除以 penalty、负 logits 乘以 penalty
② top-p 的保留个数是**数据决定的**  它留的是"累积概率刚好超过 p 的最小集合"，
                              因此同一个 top_p 在不同分布上留下的候选数不同
③ top_k 与 top_p 之间是**串联**   top_p 的核是在**已经截断过的**分布上算的
```

## 一条"不可能失败"的断言

HF 的 `TopPLogitsWarper` 有一行 `sorted_indices_to_remove[..., -min_tokens_to_keep:] = 0`，
它的作用是把"至少留一个 token"写成一条**恒等式**。本包把这件事写成一条读数：
`kept >= 1` 永远成立，而**保留下来的最小个数**会被如实印出来——
这条纪律来自 day083 第 7.2 节（"不可能失败的性质不删掉，而是把信息量如实印出来"）。
"""

from __future__ import annotations

import math

from smart_research_agent.hf_source.errors import (
    GenerationError,
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.hf_source.types import (
    STRATEGY_BEAM,
    STRATEGY_GREEDY,
    STRATEGY_SAMPLE,
    STRATEGY_TOP_K,
    STRATEGY_TOP_P,
    GenerationResult,
    GenerationSettings,
    GenerationStep,
)
from smart_research_agent.math_foundations.probability import uniforms
from smart_research_agent.math_foundations.types import Vector

#: 被过滤掉的 logits 的取值（一个足够小的有限数，而不是 ``-inf``）.
#:
#: 用 ``-inf`` 也能算对，但 `-inf` 会顺着"平均熵"污染整张表
#: （day073 第七章的同一条）——本包因此用一个有限的极小值，
#: 并在 :func:`softmax_of` 里把 `exp` 的下溢交给浮点自己处理。
FILTERED_LOGIT = -1e30

#: ``top_p`` 允许的最小保留个数（与 HF 的 ``min_tokens_to_keep=1`` 对齐）.
MIN_TOKENS_TO_KEEP = 1

#: 过滤器实验用的固定分布（**写死的 logits**，因此每一步都能手算）.
FILTER_LOGITS: Vector = (2.0, 1.0, 0.5, 0.0, -1.0, -2.0)


def validate_settings(settings: GenerationSettings, vocab: int) -> None:
    """校验一次生成的全部开关（**每一个都指出该改什么**）."""
    if vocab <= 0:
        raise ParameterError(f"词表大小必须为正，收到 {vocab}。")
    if settings.max_new_tokens <= 0:
        raise ParameterError(f"max_new_tokens 必须为正，收到 {settings.max_new_tokens}。")
    if not math.isfinite(settings.temperature) or settings.temperature <= 0.0:
        raise ParameterError(
            f"temperature 必须是正的有限数，收到 {settings.temperature!r}。"
            "HF 在 temperature=0 时会在**除法**里得到 inf/nan——本包挡在入口。"
        )
    if settings.top_k < 0:
        raise ParameterError(f"top_k 不能为负，收到 {settings.top_k}（0 表示不截断）。")
    if not 0.0 < settings.top_p <= 1.0:
        raise ParameterError(f"top_p 必须落在 (0, 1]，收到 {settings.top_p!r}。")
    if not math.isfinite(settings.repetition_penalty) or settings.repetition_penalty <= 0.0:
        raise ParameterError(
            f"repetition_penalty 必须是正的有限数，收到 {settings.repetition_penalty!r}。"
        )
    if settings.num_beams <= 0:
        raise ParameterError(f"num_beams 必须为正，收到 {settings.num_beams}。")
    if settings.eos_token is not None and not 0 <= settings.eos_token < vocab:
        raise ShapeError(f"eos_token={settings.eos_token} 落在词表 [0, {vocab}) 之外。")
    if settings.top_k > vocab:
        raise GenerationError(
            f"top_k={settings.top_k} 比词表大小 {vocab} 还大："
            "这不是'参数不合法'，而是'这次的策略写得太宽'（HF 会把它夹到 vocab，"
            "于是你以为自己在做 top-k，其实在做全词表采样）。"
        )
    if settings.num_beams > vocab:
        raise GenerationError(
            f"num_beams={settings.num_beams} 比词表大小 {vocab} 还大："
            "beam 的宽度不能超过'有多少个不同的下一个 token 可选'。"
        )
    if settings.length_penalty < 0.0:
        raise GenerationError(
            f"length_penalty={settings.length_penalty} 是负的："
            "打分是 `logprob / length**length_penalty`，而 logprob 是**非正**的，"
            "因此 lp > 0 偏好长序列、lp < 0 反过来偏好短序列（HF 文档的口径）。"
            "本包**只接受 lp >= 0**：负指数是一次方向反转，"
            "而它挂在一个叫'长度惩罚'的参数上——名字与方向已经不一致了，"
            "再允许它翻一次方向，读代码的人就只能靠试。"
        )
    if settings.num_beams > 1 and settings.do_sample:
        raise GenerationError(
            "num_beams > 1 与 do_sample=True 同时给：本包只实现**确定性**的 beam 搜索"
            "（HF 的 sampling-beam 是另一个策略，本包明确不做，而不是假装做了）。"
        )


def strategy_of(settings: GenerationSettings) -> str:
    """从配置反推"这一次用的是哪套策略"——**这里刻意只有一个答案**."""
    if settings.num_beams > 1:
        return STRATEGY_BEAM
    if not settings.do_sample:
        return STRATEGY_GREEDY
    if settings.top_k > 0:
        return STRATEGY_TOP_K
    if settings.top_p < 1.0:
        return STRATEGY_TOP_P
    return STRATEGY_SAMPLE


def checked_logits(logits: object) -> Vector:
    """校验一串 logits 的**结构与有限性**，并归入本包自己的失败族.

    它不转发 ``math_foundations.types.validate_vector``，理由是**失败族要一致**：
    那个校验器会把非有限数报成"数据问题"（它的族），而本包希望调用方
    ``except hf_source.errors.NumericError`` 就能兜住——否则同一个错误
    在导入路径不同时会是两个名字（day077 第 4.3 节记过同类的落族差异）。
    """
    if isinstance(logits, (str, bytes)) or not hasattr(logits, "__iter__"):
        raise ShapeError(f"logits 必须是一串实数，收到 {type(logits).__name__}。")
    values = tuple(logits)  # type: ignore[arg-type]
    if not values:
        raise ShapeError("logits 为空：一个空的分布没有'下一个 token'。")
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ShapeError(f"logits 的第 {index} 个分量不是实数（收到 {value!r}）。")
        if not math.isfinite(value):
            raise NumericError(
                f"logits 的第 {index} 个分量不是有限数（收到 {value!r}）："
                "softmax 之后会得到 nan，而一个 nan 会顺着'平均熵'污染整张表。"
                "HF 用加性掩码（一个大负数）而不是 -inf，正是为了避免这一类。"
            )
    return tuple(float(value) for value in values)


def softmax_of(logits: Vector) -> Vector:
    """数值稳定的 softmax（**先减最大值**，与 day073 的 ``linalg.softmax`` 同口径）."""
    checked = checked_logits(logits)
    peak = max(checked)
    shifted = [math.exp(value - peak) for value in checked]
    total = math.fsum(shifted)
    return tuple(value / total for value in shifted)


def apply_temperature(logits: Vector, temperature: float) -> Vector:
    """温度缩放：``score / temperature``（**除**，与 HF 的 ``TemperatureLogitsWarper`` 一致）."""
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ParameterError(f"temperature 必须是正的有限数，收到 {temperature!r}。")
    return tuple(value / temperature for value in logits)


def apply_repetition_penalty(
    logits: Vector,
    generated: tuple[int, ...],
    penalty: float,
) -> Vector:
    """重复惩罚：**正 logits 除以惩罚、负 logits 乘以惩罚**（HF 的原式）.

    写成"减一个常数"在 logits 全为正时看起来一样，一遇到负 logits 就变成**放大**——
    而"放大一个负 logits"正好是"更想选它"，与惩罚的意图相反。
    """
    if not math.isfinite(penalty) or penalty <= 0.0:
        raise ParameterError(f"penalty 必须是正的有限数，收到 {penalty!r}。")
    if penalty == 1.0:
        return logits
    visited = set(generated)
    return tuple(
        (value * penalty if value < 0.0 else value / penalty) if index in visited else value
        for index, value in enumerate(logits)
    )


def top_k_filter(logits: Vector, top_k: int) -> tuple[Vector, int]:
    """top-k：只留最大的 ``k`` 个（并列时**下标小的优先**，与 ``torch.topk`` 的稳定序一致）.

    返回 ``(过滤后的 logits, 保留个数)``——"保留个数"是这一步唯一需要留下的读数：
    它让"我设了 top_k=3，实际只保留了 2 个"（因为词表只有 2 个有效候选）看得见。
    """
    if top_k < 0:
        raise ParameterError(f"top_k 不能为负，收到 {top_k}。")
    if top_k == 0:
        return logits, len(logits)
    order = sorted(range(len(logits)), key=lambda index: (-logits[index], index))
    keep = set(order[:top_k])
    return (
        tuple(value if index in keep else FILTERED_LOGIT for index, value in enumerate(logits)),
        len(keep),
    )


def top_p_filter(logits: Vector, top_p: float) -> tuple[Vector, int]:
    """top-p（核采样）：与 HF 的 ``TopPLogitsWarper`` **同一套算法**.

    ```text
    ① 按 logits **升序**排序（升序是为了让"累积概率"从最小的 token 开始累加）
    ② 对排序后的 logits 求 softmax 再 cumsum
    ③ 删除 `cumulative_probs <= 1 - top_p` 的那些位置
    ④ 至少保留 MIN_TOKENS_TO_KEEP 个（HF 的 min_tokens_to_keep=1）
    ```

    第 ③ 步的意思是"把最小的那些 token 一直删到它们的总质量不超过 1−p 为止"——
    留下的就是"累积概率刚好超过 p 的最小集合"。因此**保留个数由数据决定**：
    同一个 `top_p` 在一个尖分布上留下 1 个、在均匀分布上留下好几个。
    """
    if not 0.0 < top_p <= 1.0:
        raise ParameterError(f"top_p 必须落在 (0, 1]，收到 {top_p!r}。")
    order = sorted(range(len(logits)), key=lambda index: (logits[index], index))
    sorted_logits = tuple(logits[index] for index in order)
    probabilities = softmax_of(sorted_logits)
    threshold = 1.0 - top_p
    remove: set[int] = set()
    running = 0.0
    for position, probability in enumerate(probabilities):
        running += probability
        if running <= threshold:
            remove.add(order[position])
    # ④ 至少保留 min_tokens_to_keep 个：把"最大的那几个"从删除集合里取回来。
    guarded = 0
    for position in reversed(range(len(order))):
        if guarded >= MIN_TOKENS_TO_KEEP:
            break
        remove.discard(order[position])
        guarded += 1
    keep = len(logits) - len(remove)
    if keep < MIN_TOKENS_TO_KEEP:  # pragma: no cover - 上面的循环保证了它不可能发生
        raise GenerationError("核被清空了：top_p 在这一次的分布上留下了 0 个候选。")
    return (
        tuple(value if index not in remove else FILTERED_LOGIT for index, value in enumerate(logits)),
        keep,
    )


def transform_logits(
    logits: Vector,
    settings: GenerationSettings,
    generated: tuple[int, ...],
) -> tuple[Vector, int]:
    """把四个 warper 按**固定顺序**串起来，返回 ``(最终 logits, 保留个数)``.

    顺序见 :data:`types.WARPER_ORDER`。保留个数只在 top-k / top-p 之后才有意义，
    因此没有它们时它等于词表大小（"没有截断"也是一个读数）。
    """
    checked = checked_logits(logits)
    current = apply_repetition_penalty(checked, generated, settings.repetition_penalty)
    current = apply_temperature(current, settings.temperature)
    kept = len(current)
    if settings.top_k > 0:
        current, kept = top_k_filter(current, settings.top_k)
    if settings.top_p < 1.0:
        current, kept = top_p_filter(current, settings.top_p)
    return current, kept


def distribution_entropy(probabilities: Vector) -> float:
    """一个分布的 Shannon 熵（单位 **nats**，与 day073 的口径一致）.

    ``-Σ p·ln p``，零项按 ``0·ln0 = 0`` 处理（**不是**靠 ``-inf`` 的极限行为）。
    """
    total = 0.0
    for probability in probabilities:
        if probability > 0.0:
            total -= probability * math.log(probability)
    return total


def greedy_index(logits: Vector) -> int:
    """贪心：取最大值的**最小下标**（并列时与 ``torch.argmax`` 的取法一致）."""
    best = 0
    for index in range(1, len(logits)):
        if logits[index] > logits[best]:
            best = index
    return best


def sample_index(probabilities: Vector, uniform: float) -> int:
    """逆变换采样：在累积分布上找第一个 ``cumsum > u`` 的位置.

    它只依赖传入的 ``u``，因此"采样"在这里是一个**纯函数**——
    随机性全部集中在 `u` 的来源上（本包用 day073 的 LCG）。
    """
    if not 0.0 <= uniform < 1.0:
        raise ParameterError(f"均匀数必须落在 [0, 1)，收到 {uniform!r}。")
    running = 0.0
    for index, probability in enumerate(probabilities):
        running += probability
        if uniform < running:
            return index
    return len(probabilities) - 1


def select_token(
    logits: Vector,
    settings: GenerationSettings,
    uniform: float,
) -> int:
    """在**已经变换过**的 logits 上选一个 token（贪心或采样）."""
    probabilities = softmax_of(logits)
    if settings.do_sample:
        return sample_index(probabilities, uniform)
    return greedy_index(logits)


def generate(
    logits_fn,
    prompt: tuple[int, ...],
    settings: GenerationSettings,
    *,
    vocab: int,
) -> GenerationResult:
    """自回归生成（贪心 / 采样 / top-k / top-p 共用这一条循环）.

    ``logits_fn`` 是"模型"：给定当前的 token 序列，返回一个长度 ``vocab`` 的向量。
    本课**不跑真模型**——它把生成策略与模型彻底分开，
    因此这里可以只用一个玩具分布把四种策略的差别量清楚。

    ``eos_token`` 命中时提前结束，并把 ``stopped_early`` 记成 ``True``——
    "为什么它少生成了几个 token"必须是可回答的。
    """
    validate_settings(settings, vocab)
    if settings.num_beams > 1:
        return beam_search(logits_fn, prompt, settings, vocab=vocab)
    if not prompt:
        raise ShapeError("prompt 为空：生成需要至少一个起始 token。")
    stream = uniforms(settings.max_new_tokens, seed=settings.seed)
    tokens = list(prompt)
    steps: list[GenerationStep] = []
    stopped = False
    for step in range(settings.max_new_tokens):
        raw = checked_logits(logits_fn(tuple(tokens)))
        if len(raw) != vocab:
            raise ShapeError(f"logits 长度 {len(raw)} 与词表大小 {vocab} 不一致。")
        transformed, kept = transform_logits(raw, settings, tuple(tokens))
        probabilities = softmax_of(transformed)
        chosen = select_token(transformed, settings, stream[step])
        steps.append(
            GenerationStep(
                step=step,
                chosen=chosen,
                kept=kept,
                top_probability=max(probabilities),
                entropy=distribution_entropy(probabilities),
                strategy=strategy_of(settings),
            )
        )
        tokens.append(chosen)
        if settings.eos_token is not None and chosen == settings.eos_token:
            stopped = True
            break
    return GenerationResult(
        token_ids=tuple(tokens),
        prompt_length=len(prompt),
        steps=tuple(steps),
        strategy=strategy_of(settings),
        stopped_early=stopped,
        notes=(
            "eos_token 命中 ⇒ 提前结束" if stopped else "跑满 max_new_tokens",
            f"过滤后至少保留 {min((step.kept for step in steps), default=vocab)} 个候选",
        ),
    )


def log_softmax_of(logits: Vector) -> Vector:
    """逐元素的 log-softmax（beam 的累加分数用它，而不是用连乘）."""
    probabilities = softmax_of(logits)
    return tuple(math.log(value) if value > 0.0 else float("-inf") for value in probabilities)


def beam_search(
    logits_fn,
    prompt: tuple[int, ...],
    settings: GenerationSettings,
    *,
    vocab: int,
) -> GenerationResult:
    """beam search：同时维护 ``num_beams`` 条前缀，按**长度惩罚后的分数**选最终序列.

    ```text
    每一步：对每条 beam 取 log-softmax → 取 top-num_beams 个候选 → 累加分数
            → 在 num_beams² 个候选里挑出最好的 num_beams 条
    结束时：final = score / length ** length_penalty      （length = **新生成**的长度）
    ```

    长度惩罚那一步不是装饰：**每一步都在乘一个小于 1 的概率**（对数概率非正），
    因此候选越长、总分越负——不修正时 beam 会系统性地偏向短序列。
    加分公式是 `final = score / length ** length_penalty`：

    ```text
    lp = 0    不做任何修正 ⇒ 系统性偏向短序列
    lp > 0    把"越负"摊薄       ⇒ 偏好更长的序列（lp 越大越强）
    lp < 0    反过来放大"越负"   ⇒ 偏好更短的序列
    ```

    两个方向都是合法策略，而**负值本包明确拒绝**：
    它是一次方向反转，而参数名是"长度惩罚"——名字与方向已经不一致了。
    """
    validate_settings(settings, vocab)
    if not prompt:
        raise ShapeError("prompt 为空：生成需要至少一个起始 token。")
    beams: list[tuple[tuple[int, ...], float, bool]] = [(tuple(prompt), 0.0, False)]
    finished: list[tuple[tuple[int, ...], float]] = []
    for _step in range(settings.max_new_tokens):
        candidates: list[tuple[tuple[int, ...], float, bool]] = []
        for tokens, score, done in beams:
            if done:
                candidates.append((tokens, score, True))
                continue
            raw = checked_logits(logits_fn(tokens))
            if len(raw) != vocab:
                raise ShapeError(f"logits 长度 {len(raw)} 与词表大小 {vocab} 不一致。")
            current, _kept = transform_logits(raw, settings, tokens)
            log_probs = log_softmax_of(current)
            order = sorted(range(vocab), key=lambda index: (-log_probs[index], index))
            for index in order[: settings.num_beams]:
                if log_probs[index] == float("-inf"):
                    continue
                stopped = settings.eos_token is not None and index == settings.eos_token
                candidates.append((tokens + (index,), score + log_probs[index], stopped))
        if not candidates:  # pragma: no cover - 变换后的分布至少有 1 个有限项
            raise GenerationError("没有可用的 beam 候选：所有 logits 都被过滤掉了。")
        candidates.sort(key=lambda item: (-item[1], item[0]))
        beams = candidates[: settings.num_beams]
        finished.extend((tokens, score) for tokens, score, done in beams if done)
        if settings.early_stopping and finished:
            break
    pool = finished if finished else [(tokens, score) for tokens, score, _done in beams]
    best_tokens, best_score = min(
        pool,
        key=lambda item: (
            -(item[1] / (max(len(item[0]) - len(prompt), 1) ** settings.length_penalty)),
            item[0],
        ),
    )
    generated_length = max(len(best_tokens) - len(prompt), 1)
    normalized = best_score / (generated_length**settings.length_penalty)
    return GenerationResult(
        token_ids=best_tokens,
        prompt_length=len(prompt),
        steps=(
            GenerationStep(
                step=0,
                chosen=best_tokens[-1],
                kept=settings.num_beams,
                top_probability=math.exp(normalized) if math.isfinite(normalized) else 0.0,
                entropy=0.0,
                strategy=STRATEGY_BEAM,
            ),
        ),
        strategy=STRATEGY_BEAM,
        stopped_early=bool(finished),
        notes=(
            f"beam 宽度 {settings.num_beams}",
            f"长度惩罚 {settings.length_penalty:g}（新生成长度 {generated_length}）",
            f"归一化分数 {normalized:.6f}",
        ),
    )


__all__ = [
    "FILTERED_LOGIT",
    "FILTER_LOGITS",
    "MIN_TOKENS_TO_KEEP",
    "apply_repetition_penalty",
    "apply_temperature",
    "beam_search",
    "checked_logits",
    "distribution_entropy",
    "generate",
    "greedy_index",
    "log_softmax_of",
    "sample_index",
    "select_token",
    "softmax_of",
    "strategy_of",
    "top_k_filter",
    "top_p_filter",
    "transform_logits",
    "validate_settings",
]
