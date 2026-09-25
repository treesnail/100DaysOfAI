"""近重复指纹：shingle → MinHash 签名 → Jaccard 估计（M5-D8）.

day048 的去重是**精确**去重：``clean_text(prompt).lower()`` 的 SHA-1。
它能挡住"同一份语料被采集两次"，却挡不住**字面高度重叠**的副本——
同一段内容被加上一句开场白、被截断掉尾巴、被复制到另一份文件里再改两个字，
字符指纹就完全不同，但它们其实是同一份语料。这类样本留在数据集里有三个
具体后果：

1. **频率分布被扭曲**：同一个知识点被反复强化，模型对它的先验被放大；
2. **评估泄漏**：一份进了训练集、另一份进了评估集，评估分数被静默抬高；
3. **配比失真**：按来源/标签统计出来的占比，描述的不能代表真实的语义分布。

## 先划清一条边界：近重复 ≠ 语义重复

本模块抓的是**字面**重复：shingle 集合的 Jaccard 相似度。
"如何评估 RAG 的检索质量？"与"怎么评价 RAG 的检索效果？"意思几乎一样，
但实测**精确** Jaccard 只有 0.3000、MinHash 估计 0.3438——它们
**不该**被本模块判为重复。把阈值一路调到 0.3 去"抓语义重复"是错的：
那会把同一个领域里正常的不同提问全部判死，数据集会被清空。

**语义重复要靠 embedding 与向量检索**（day041 的 ``EmbeddingProvider``、
M6 的向量数据库），那是另一条链路。两件事分开做，各自有各自的阈值与报告，
才说得清"到底丢了什么"。本模块用 ``MINHASH_PRIME`` 那一族线性置换，
做法与面试里常见的"MinHash 去重"完全一致，剩下的只是口径问题。

## 三件套

- ``shingles``：把文本切成字符 n-gram 集合。用**字符级**而非词级，是因为
  中文没有空格分词，词级切分在中英混排上得不到一致的粒度；字符 n-gram
  对中英混排都成立，``k=3`` 是本课程的经验值（短中文句子在 ``k=3`` 时
  仍能产生足量 shingle）。
- ``jaccard_exact``：两个 shingle 集合的**精确** Jaccard 相似度。它是定义，
  不是估计——MinHash 要逼近的正是它，因此测试必须拿它当基准。
- ``minhash_signature`` / ``estimate_jaccard``：MinHash 签名与估计量。
  签名长度固定为 ``num_perm``，估计的期望恰为真实 Jaccard，
  标准差约 ``sqrt(J(1-J)/num_perm)``。``num_perm=64`` 时 ``J=0.5`` 处
  标准差约 0.062——**阈值必须留裕度，不能把 0.70 当成精确判定**。

**为什么不做 LSH 分桶把比较降到近线性**：本课程的数据规模是万级，
朴素两两比较是 ``N²`` 次 64 维整数比较；分桶省下的是常数，代价是
引入一个**不可复现的召回损失**（同一份数据换个分桶参数就得到不同的
重复组）。数据工程里"可复现"比"快一点"重要，所以本课明确选择朴素比较，
并把复杂度写在这里，而不是留给学生自己去猜。
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

from smart_research_agent.domain_data.errors import DomainDataError
from smart_research_agent.finetune.cleaner import DEDUPE_KEY_LENGTH, clean_text
from smart_research_agent.finetune.schema import TrainingExample

#: shingle 的字符窗口长度。3 是本课程在中文短句上的经验值：
#: 取 2 会让"检索增强"与"检索质量"共享大量 shingle（假重复变多），
#: 取 5 会让不足 5 字的提问退化成单 shingle（真重复漏判）。
DEFAULT_SHINGLE_K = 3

#: MinHash 置换个数。64 个置换在 J=0.5 处的标准差约 0.062，
#: 足够支撑 0.8 这一档阈值；调大能降方差，代价是签名与比较都变慢。
DEFAULT_NUM_PERM = 64

#: MinHash 使用的模数：Mersenne 素数 2^61 - 1。
#: 取它是因为 61 位足够容纳 64 位哈希的模运算，且在 Python 里
#: 大整数乘法取模有专门优化——**不要换成 2^32 之类的"小素数"**，
#: 那会让不同 shingle 的哈希值大量碰撞，估计值系统性偏低。
MINHASH_PRIME = (1 << 61) - 1

#: 置换参数的随机种子。固定后同一份数据永远得到同一组签名，
#: 因此"这次重复率 12%"与"上次重复率 12%"是可比较的两个数字。
MINHASH_SEED = 42

#: 空 shingle 集合的签名占位值（见 ``minhash_signature`` 的说明）。
MINHASH_EMPTY = 0

#: 64 位掩码：SplitMix64 的状态与输出都截断到这里。
_MASK64 = (1 << 64) - 1
_SM_GOLDEN = 0x9E3779B97F4A7C15
_SM_MUL1 = 0xBF58476D1CE4E5B9
_SM_MUL2 = 0x94D049BB133111EB


def normalize_for_fingerprint(text: str) -> str:
    """指纹比较前统一的归一化：清洗（NFKC + 折叠空白）后转小写.

    直接复用 ``finetune.cleaner.clean_text``，因此本模块的"看起来一样"
    与 day048 的精确去重**是同一个定义**。若这里另写一份清洗，就会出现
    "精确去重认为两条不同、近去重认为两条相同"的自相矛盾——报告里两个
    数字打架，评审时根本说不清哪一个是错的。
    """
    return clean_text(text).lower()


def shingles(text: str, *, k: int = DEFAULT_SHINGLE_K) -> frozenset[str]:
    """文本的字符 n-gram 集合（归一化之后）.

    两个刻意的边界处理：

    - ``k < 1`` 直接拒绝：``k=0`` 会让每一条文本都得到"空集合"或"无穷集合"
      这种无意义结果；
    - 文本短于 ``k`` 时退化为**整串一个 shingle**。若不这样处理，短文本的
      shingle 集合为空，而空集合之间的 Jaccard 是 1.0——所有短文本会被
      判为"彼此完全相同"，一个"好"字与一个"差"字就成了重复项。
    """
    if k < 1:
        raise DomainDataError(f"shingle 窗口 k 必须 >= 1，收到 {k}")
    normalized = normalize_for_fingerprint(text)
    if not normalized:
        return frozenset()
    if len(normalized) < k:
        return frozenset({normalized})
    return frozenset(normalized[index : index + k] for index in range(len(normalized) - k + 1))


def jaccard_exact(left: frozenset[str], right: frozenset[str]) -> float:
    """两个 shingle 集合的精确 Jaccard 相似度：交集大小 / 并集大小.

    空集合的约定：两边都空 → 1.0（"两条空文本"确实是同一件事），
    只有一边空 → 0.0。这条约定让 ``jaccard_exact`` 与
    ``estimate_jaccard`` 在全空输入上给出同样的答案，两者可以直接对照。
    """
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def hash_shingle(shingle: str) -> int:
    """把 shingle 映射到 64 位无符号整数.

    用 ``blake2b`` 而不是 ``hash()``：Python 内建 ``hash`` 对 ``str``
    默认加盐（``PYTHONHASHSEED``），**跨进程、跨运行都不稳定**——
    基于它算出来的签名每次都不一样，去重结果就不可复现了。
    """
    digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _splitmix64(state: int) -> tuple[int, int]:
    """一步 SplitMix64：返回（新状态, 输出）.

    自己实现 PRNG 而不是用 ``random.Random``：置换参数是**签名的组成部分**，
    一旦某个 Python 版本改了 ``randrange`` 的抽样细节，历史数据集的指纹
    就会变，版本链断掉。SplitMix64 只依赖整数加减乘与移位，
    它的输出可以逐位复现，也不需要把随机数状态一起归档。
    """
    state = (state + _SM_GOLDEN) & _MASK64
    value = state
    value = ((value ^ (value >> 30)) * _SM_MUL1) & _MASK64
    value = ((value ^ (value >> 27)) * _SM_MUL2) & _MASK64
    return state, value ^ (value >> 31)


def permutation_params(
    num_perm: int = DEFAULT_NUM_PERM, *, seed: int = MINHASH_SEED
) -> tuple[tuple[int, int], ...]:
    """生成 ``num_perm`` 组 ``(a, b)`` 置换参数：``h(x) = (a*x + b) mod p``.

    ``a`` 取 ``[1, p-1]``（不能为 0，否则所有输入都映射到 ``b``，签名退化），
    ``b`` 取 ``[0, p-1]``。这一族线性置换在 ``p`` 为素数时是双射的，
    正是 MinHash 需要的性质。
    """
    if num_perm < 1:
        raise DomainDataError(f"置换个数 num_perm 必须 >= 1，收到 {num_perm}")
    params: list[tuple[int, int]] = []
    state = seed & _MASK64
    for _ in range(num_perm):
        state, raw_a = _splitmix64(state)
        state, raw_b = _splitmix64(state)
        params.append((1 + raw_a % (MINHASH_PRIME - 1), raw_b % MINHASH_PRIME))
    return tuple(params)


def minhash_signature(
    shingle_set: Iterable[str],
    *,
    num_perm: int = DEFAULT_NUM_PERM,
    seed: int = MINHASH_SEED,
) -> tuple[int, ...]:
    """MinHash 签名：每个置换下所有 shingle 哈希值的**最小值**（长度恒为 num_perm）.

    签名长度恒定这一点很重要：它让 ``estimate_jaccard`` 可以直接逐位比较，
    也让"两条签名能不能比"这件事在类型层面就成立（长度不同即参数不匹配，
    直接抛 ``DomainDataError`` 而不是给出一个没有意义的分数）。

    **空集合的约定**：没有 shingle 时签名是全零向量。理论上的 MinHash
    对空集无定义，取全零是一个可解释的占位：它保证"空文本 vs 非空文本"
    的估计接近 0（非空签名的取值几乎不可能恰好是 0），而"空文本 vs
    空文本"的估计恰为 1.0。数据侧不用担心这个边界——day048 的
    ``empty_instruction`` 规则已经把空指令挡在流水线之外。
    """
    hashes = sorted(hash_shingle(item) for item in shingle_set)
    params = permutation_params(num_perm, seed=seed)
    if not hashes:
        return tuple(MINHASH_EMPTY for _ in range(num_perm))
    return tuple(min((a * value + b) % MINHASH_PRIME for value in hashes) for a, b in params)


def estimate_jaccard(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    """由两条签名估计 Jaccard 相似度：相同位次的比例（无偏估计）.

    它是**估计**而非精确值，因此流水线里只把它当"要不要人工复核"的
    排序信号；需要精确值的场景（例如把重复率写进报告）请用
    ``jaccard_exact``——两者在本模块里并存，就是为了让"估计"与"事实"
    各自有名字，不会被混着引用。
    """
    if len(left) != len(right):
        raise DomainDataError(
            f"签名长度不一致（{len(left)} vs {len(right)}）："
            "两条签名必须来自同一组 num_perm 与 seed，否则比较没有意义"
        )
    if not left:
        return 1.0
    matches = sum(1 for x, y in zip(left, right) if x == y)
    return matches / len(left)


@dataclass(frozen=True)
class Fingerprint:
    """一条样本的近重复指纹：精确键 + shingle 数 + MinHash 签名.

    ``key`` 与 day048 的 ``dedupe_key`` 使用同一套归一化与同一个长度常量，
    因此"精确去重"与"近去重"在**同一条**样本上永远给出一致的判断基础：
    键相同 → 一定重复；键不同 → 才轮到签名出手。两级判定不是重复劳动，
    而是"便宜且确定的先做，昂贵且近似的后做"。
    """

    key: str
    shingle_count: int
    signature: tuple[int, ...]

    def similarity(self, other: Fingerprint) -> float:
        """与另一条指纹的 Jaccard **估计值**（同参数签名的逐位比较）."""
        return estimate_jaccard(self.signature, other.signature)

    def is_near(self, other: Fingerprint, *, threshold: float) -> bool:
        """是否构成近重复：精确键相同，或估计相似度达到阈值.

        精确键相同时直接返回 True，不依赖估计——一条"完全相同"的样本
        必须是重复项，这一点不能交给概率来决定。
        """
        if self.key == other.key:
            return True
        return self.similarity(other) >= threshold


def exact_key(text: str) -> str:
    """精确指纹：归一化后文本的 SHA-1 前 16 位（与 day048 的 ``dedupe_key`` 同定义）.

    单独抽成函数，是为了让"数据集内容指纹"（``pipeline.dataset_fingerprint``）
    不必为了拿一个键而先算一遍 64 个 MinHash 置换——那会让"给数据集打指纹"
    的成本凭空乘以 64，而它本来只是几十次哈希。
    """
    normalized = normalize_for_fingerprint(text)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:DEDUPE_KEY_LENGTH]


def fingerprint_text(
    text: str,
    *,
    k: int = DEFAULT_SHINGLE_K,
    num_perm: int = DEFAULT_NUM_PERM,
    seed: int = MINHASH_SEED,
) -> Fingerprint:
    """为一段文本生成指纹（比较对象由调用方决定）."""
    normalized = normalize_for_fingerprint(text)
    shingle_set = shingles(normalized, k=k)
    return Fingerprint(
        key=exact_key(normalized),
        shingle_count=len(shingle_set),
        signature=minhash_signature(shingle_set, num_perm=num_perm, seed=seed),
    )


def fingerprint_example(
    example: TrainingExample,
    *,
    k: int = DEFAULT_SHINGLE_K,
    num_perm: int = DEFAULT_NUM_PERM,
    seed: int = MINHASH_SEED,
) -> Fingerprint:
    """为一条训练样本生成指纹——**比较对象是 prompt，不含 output**.

    与 day048 的精确去重口径**必须一致**：那里也只用 prompt。理由是
    "同一个问题配两个不同答案"在监督微调里是**冲突**而不是冗余
    （day048 的纪律），近似判定若换成比较整条样本，"同问不同答"
    就会从"冲突"变成"不同"，报告里的重复数会突然变小——同一份数据
    在两套口径下得到两个结论，是数据治理最忌讳的事。
    """
    return fingerprint_text(example.prompt_text, k=k, num_perm=num_perm, seed=seed)
