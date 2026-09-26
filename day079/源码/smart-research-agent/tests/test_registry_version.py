"""day058 版本三元组测试（M5-D9）：身份、归一化、递增规则.

这个文件只测 ``registry/version.py`` 一个模块，守的是三件"错了不会报错"的事：

1. **内容寻址的稳定性**：同一份产物的不同写法（带 ``sha256:`` 前缀、大写、
   首尾空白）必须得到**同一个键**。这一条破了，注册表里会出现两份指向同一
   产物的记录，而它们的"重复"不会以任何形式暴露；
2. **键的可分辨性**：三个字段各变一次必须得到三个不同的键。这一条破了，
   "换了数据的候选"会与"旧版本"被判成同一份产物，于是版本表少一条，
   而且**回滚目标的选择会悄悄错位**；
3. **递增位的推导**：基座 → ``major``、数据集 → ``minor``、只有适配器 → ``patch``。
   这一条破了，版本号会迅速失去区分度（每次重训都涨一个大版本）。

归一化与校验的边界（空串、非十六进制、过短）也逐条钉住：它们是
"抄哈希时抄错"这类人工事故的唯一防线。
"""

from __future__ import annotations

import pytest

from smart_research_agent.registry import RegistryError
from smart_research_agent.registry.version import (
    BUMP_KINDS,
    BUMP_MAJOR,
    BUMP_MINOR,
    BUMP_PATCH,
    INITIAL_VERSION,
    MIN_ADAPTER_HASH_LENGTH,
    MIN_DATASET_FINGERPRINT_LENGTH,
    VERSION_KEY_LENGTH,
    VersionConflict,
    VersionTriple,
    bump_kind_for,
    bump_version,
    canonical_version_payload,
    comparable,
    format_semver,
    next_version,
    normalize_digest,
    parse_semver,
    semver_sort_key,
    version_key,
)

BASE = "Qwen3-8B"
BIG = "Qwen3-14B"
DATA_A = "3f1b0c9d7e5a2468"
DATA_B = "aa77c31e90f4b258"
ADAPTER_A = "a" * 64
ADAPTER_B = "b" * 64
ADAPTER_C = "c" * 64


# --------------------------------------------------------------------------- #
# normalize_digest：抄哈希时的三道防线
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("  abc123  ", "abc123"),
        ("sha256:abc123", "abc123"),
        ("SHA256:abc123", "abc123"),
        ("sha256: ABC123", "abc123"),
        ("ABC123", "abc123"),
        ("0a1b2c3d", "0a1b2c3d"),
    ],
)
def test_normalize_digest_strips_prefix_and_case(raw: str, expected: str) -> None:
    """前缀、大小写、空白都必须被抹平——否则同一个适配器会有两个键。"""
    assert normalize_digest(raw, field_name="adapter_sha256") == expected


def test_normalize_digest_rejects_empty() -> None:
    """空哈希：与其静默接受，不如立刻报错（空串会与另一个空串"相等"）。"""
    with pytest.raises(RegistryError, match="不能为空"):
        normalize_digest("   ", field_name="adapter_sha256")


@pytest.mark.parametrize("raw", ["xyz", "abc12g", "sha256:zz", "12-34"])
def test_normalize_digest_rejects_non_hex(raw: str) -> None:
    """非十六进制一律拒绝，并在错误信息里点名"抄错了前缀"这个常见原因。"""
    with pytest.raises(RegistryError, match="必须是十六进制字符串"):
        normalize_digest(raw, field_name="dataset_fingerprint")


# --------------------------------------------------------------------------- #
# canonical_version_payload：长度下限与必填
# --------------------------------------------------------------------------- #


def test_canonical_payload_normalizes_all_three_fields() -> None:
    """三个字段都归一化：base 去空白，两个哈希走 normalize_digest。"""
    payload = canonical_version_payload(
        base_model="  Qwen3-8B ",
        adapter_sha256=f"SHA256:{ADAPTER_A.upper()}",
        dataset_fingerprint=DATA_A.upper(),
    )
    assert payload == {
        "adapter_sha256": ADAPTER_A,
        "base_model": BASE,
        "dataset_fingerprint": DATA_A,
    }


def test_canonical_payload_rejects_blank_base_model() -> None:
    """基座名为空：三元组就只剩两项，而"挂在哪个基座上"是它的一半语义。"""
    with pytest.raises(RegistryError, match="base_model 不能为空"):
        canonical_version_payload(
            base_model="   ", adapter_sha256=ADAPTER_A, dataset_fingerprint=DATA_A
        )


def test_canonical_payload_enforces_adapter_length_floor() -> None:
    """适配器哈希长度下限：16 位（截断哈希可用），15 位直接拒绝。"""
    short = "a" * (MIN_ADAPTER_HASH_LENGTH - 1)
    with pytest.raises(RegistryError, match="adapter_sha256 至少"):
        canonical_version_payload(
            base_model=BASE, adapter_sha256=short, dataset_fingerprint=DATA_A
        )
    ok = canonical_version_payload(
        base_model=BASE,
        adapter_sha256="a" * MIN_ADAPTER_HASH_LENGTH,
        dataset_fingerprint=DATA_A,
    )
    assert len(ok["adapter_sha256"]) == MIN_ADAPTER_HASH_LENGTH


def test_canonical_payload_enforces_dataset_length_floor() -> None:
    """数据集指纹长度下限：8 位（``domain_data`` 给的是 16 位）。"""
    short = "b" * (MIN_DATASET_FINGERPRINT_LENGTH - 1)
    with pytest.raises(RegistryError, match="dataset_fingerprint 至少"):
        canonical_version_payload(
            base_model=BASE, adapter_sha256=ADAPTER_A, dataset_fingerprint=short
        )


# --------------------------------------------------------------------------- #
# version_key：内容寻址
# --------------------------------------------------------------------------- #


def test_version_key_is_sixteen_hex_chars() -> None:
    """键长固定 16 位十六进制——短到能在一行报告里写完，长到不会撞。"""
    key = version_key(
        base_model=BASE, adapter_sha256=ADAPTER_A, dataset_fingerprint=DATA_A
    )
    assert len(key) == VERSION_KEY_LENGTH
    assert all(char in "0123456789abcdef" for char in key)


def test_version_key_is_order_independent_of_serialization() -> None:
    """同一个三元组多次调用得到同一个键（规范化序列化的直接结论）。"""
    first = version_key(
        base_model=BASE, adapter_sha256=ADAPTER_A, dataset_fingerprint=DATA_A
    )
    second = version_key(
        base_model=BASE, adapter_sha256=ADAPTER_A, dataset_fingerprint=DATA_A
    )
    assert first == second


@pytest.mark.parametrize(
    "base, adapter, dataset",
    [
        (BASE, ADAPTER_A, DATA_A),
        (BASE, ADAPTER_B, DATA_A),
        (BASE, ADAPTER_A, DATA_B),
        (BIG, ADAPTER_A, DATA_A),
    ],
)
def test_version_keys_differ_per_field(base: str, adapter: str, dataset: str) -> None:
    """三项里任一项不同 → 键必须不同。这条即"可分辨性"。"""
    reference = version_key(
        base_model=BASE, adapter_sha256=ADAPTER_A, dataset_fingerprint=DATA_A
    )
    other = version_key(
        base_model=base, adapter_sha256=adapter, dataset_fingerprint=dataset
    )
    if (base, adapter, dataset) == (BASE, ADAPTER_A, DATA_A):
        assert other == reference
    else:
        assert other != reference


# --------------------------------------------------------------------------- #
# VersionTriple
# --------------------------------------------------------------------------- #


def test_version_triple_normalizes_on_construction() -> None:
    """脏写法与规范写法相等且键相同——这是"同一份产物只有一个身份"的前提。"""
    canonical = VersionTriple(
        base_model=BASE, adapter_sha256=ADAPTER_A, dataset_fingerprint=DATA_A
    )
    sloppy = VersionTriple(
        base_model=f"  {BASE}  ",
        adapter_sha256=f"sha256:{ADAPTER_A.upper()}",
        dataset_fingerprint=DATA_A.upper(),
    )
    assert canonical == sloppy
    assert canonical.key == sloppy.key
    assert sloppy.base_model == BASE
    assert sloppy.adapter_sha256 == ADAPTER_A


def test_version_triple_is_frozen() -> None:
    """三元组不可变：就地修改会让索引里那一行与内存对象指向两件不同的事。"""
    triple = VersionTriple(
        base_model=BASE, adapter_sha256=ADAPTER_A, dataset_fingerprint=DATA_A
    )
    with pytest.raises(Exception):  # noqa: B017 - dataclasses.FrozenInstanceError
        triple.base_model = BIG  # type: ignore[misc]


def test_version_triple_short_adapter_and_describe() -> None:
    """``short_adapter`` 取前 12 位；``describe`` 把三项键都写进一行。"""
    triple = VersionTriple(
        base_model=BASE, adapter_sha256=ADAPTER_A, dataset_fingerprint=DATA_A
    )
    assert triple.short_adapter == ADAPTER_A[:12]
    line = triple.describe()
    assert BASE in line
    assert triple.short_adapter in line
    assert DATA_A in line
    assert triple.key in line


def test_version_triple_roundtrip() -> None:
    """``to_dict`` / ``from_dict`` 往返：派生键 ``key`` 不进构造。"""
    triple = VersionTriple(
        base_model=BASE, adapter_sha256=ADAPTER_A, dataset_fingerprint=DATA_A
    )
    payload = triple.to_dict()
    assert payload["key"] == triple.key
    assert VersionTriple.from_dict(payload) == triple


def test_version_triple_rejects_bad_digest() -> None:
    """构造期校验：非法哈希在进注册表之前就被拦下。"""
    with pytest.raises(RegistryError):
        VersionTriple(base_model=BASE, adapter_sha256="xyz", dataset_fingerprint=DATA_A)


# --------------------------------------------------------------------------- #
# comparable：能不能比
# --------------------------------------------------------------------------- #


def test_comparable_requires_same_base_and_dataset() -> None:
    """同基座同数据 → 可比；任一项不同 → 不可比（不是"差不多"，是"不可比"）。"""
    reference = VersionTriple(
        base_model=BASE, adapter_sha256=ADAPTER_A, dataset_fingerprint=DATA_A
    )
    assert comparable(
        reference,
        VersionTriple(base_model=BASE, adapter_sha256=ADAPTER_B, dataset_fingerprint=DATA_A),
    )
    assert not comparable(
        reference,
        VersionTriple(base_model=BASE, adapter_sha256=ADAPTER_A, dataset_fingerprint=DATA_B),
    )
    assert not comparable(
        reference,
        VersionTriple(base_model=BIG, adapter_sha256=ADAPTER_A, dataset_fingerprint=DATA_A),
    )


# --------------------------------------------------------------------------- #
# semver
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text, expected",
    [("1.0.0", (1, 0, 0)), ("0.1.2", (0, 1, 2)), ("12.34.56", (12, 34, 56))],
)
def test_parse_semver_accepts_three_segments(text: str, expected: tuple[int, int, int]) -> None:
    """只接受 X.Y.Z 三段。"""
    assert parse_semver(text) == expected


@pytest.mark.parametrize("text", ["1.2", "v1.2.0", "1.2.3.4", "", "1.2.x"])
def test_parse_semver_rejects_loose_forms(text: str) -> None:
    """宽松解析会让 '1.2' 与 '1.2.0' 同时存在，排序与去重立刻出问题。"""
    with pytest.raises(RegistryError, match="X.Y.Z"):
        parse_semver(text)


def test_parse_semver_tolerates_surrounding_whitespace() -> None:
    """首尾空白被 strip：从日志里复制版本号时尾部常带换行。"""
    assert parse_semver(" 1.2.3 ") == (1, 2, 3)


def test_format_semver_roundtrip() -> None:
    """格式化与解析互逆。"""
    assert format_semver(parse_semver("3.4.5")) == "3.4.5"
    assert format_semver((0, 0, 1)) == "0.0.1"


def test_semver_sort_key_orders_numerically() -> None:
    """字符串排序会把 1.10.0 排在 1.9.0 前面；整数段排序不会。"""
    assert semver_sort_key("1.10.0") > semver_sort_key("1.9.0")
    assert sorted(["1.10.0", "1.9.0", "2.0.0"], key=semver_sort_key) == [
        "1.9.0",
        "1.10.0",
        "2.0.0",
    ]


@pytest.mark.parametrize(
    "version, kind, expected",
    [
        ("1.2.3", BUMP_PATCH, "1.2.4"),
        ("1.2.3", BUMP_MINOR, "1.3.0"),
        ("1.2.3", BUMP_MAJOR, "2.0.0"),
        ("1.9.3", BUMP_MAJOR, "2.0.0"),
        ("0.0.0", BUMP_PATCH, "0.0.1"),
    ],
)
def test_bump_version_resets_lower_segments(version: str, kind: str, expected: str) -> None:
    """递增某一位时右边的位必须归零——否则会得到 1.9.3 → 2.9.3 这种怪号。"""
    assert bump_version(version, kind=kind) == expected


def test_bump_version_rejects_unknown_kind() -> None:
    """未知递增位报错，而不是当成 patch。"""
    with pytest.raises(RegistryError, match="未知的递增位"):
        bump_version("1.0.0", kind="epoch")


def test_bump_kinds_tuple_matches_module_constants() -> None:
    """``BUMP_KINDS`` 必须恰好含三个常量（布局端点直接消费它）。"""
    assert set(BUMP_KINDS) == {BUMP_MAJOR, BUMP_MINOR, BUMP_PATCH}


# --------------------------------------------------------------------------- #
# bump_kind_for：变化的那一项决定递增位
# --------------------------------------------------------------------------- #


def test_bump_kind_for_first_version_is_minor() -> None:
    """首版返回 minor（占位语义），实际版本号由 ``next_version`` 给 INITIAL。"""
    assert (
        bump_kind_for(None, base_model=BASE, dataset_fingerprint=DATA_A) == BUMP_MINOR
    )


def test_bump_kind_for_base_change_is_major() -> None:
    """换基座 = 整条适配器链作废，必须 major。"""
    previous = VersionTriple(
        base_model=BASE, adapter_sha256=ADAPTER_A, dataset_fingerprint=DATA_A
    )
    assert (
        bump_kind_for(previous, base_model=BIG, dataset_fingerprint=DATA_A)
        == BUMP_MAJOR
    )


def test_bump_kind_for_dataset_change_is_minor() -> None:
    """换数据集 = minor：模型谱系连续，但评估口径变了。"""
    previous = VersionTriple(
        base_model=BASE, adapter_sha256=ADAPTER_A, dataset_fingerprint=DATA_A
    )
    assert (
        bump_kind_for(previous, base_model=BASE, dataset_fingerprint=DATA_B)
        == BUMP_MINOR
    )


def test_bump_kind_for_retrain_is_patch() -> None:
    """同数据同基座的重训 = patch（持续微调里最高频的动作）。"""
    previous = VersionTriple(
        base_model=BASE, adapter_sha256=ADAPTER_A, dataset_fingerprint=DATA_A
    )
    assert (
        bump_kind_for(previous, base_model=BASE, dataset_fingerprint=DATA_A)
        == BUMP_PATCH
    )


def test_bump_kind_for_ignores_adapter_argument() -> None:
    """适配器哈希不在判据里：它变了就是 patch，而 patch 是"默认档"。"""
    previous = VersionTriple(
        base_model=BASE, adapter_sha256=ADAPTER_A, dataset_fingerprint=DATA_A
    )
    assert (
        bump_kind_for(previous, base_model=BASE, dataset_fingerprint=DATA_A)
        == BUMP_PATCH
    )


# --------------------------------------------------------------------------- #
# next_version
# --------------------------------------------------------------------------- #


def test_next_version_empty_history_is_initial() -> None:
    """空历史给 1.0.0。"""
    assert next_version([]) == INITIAL_VERSION
    assert next_version([], kind=BUMP_MAJOR) == INITIAL_VERSION


def test_next_version_uses_max_not_last() -> None:
    """取**最大**版本号而不是最后一个：登记顺序不决定版本序列。"""
    assert next_version(["1.0.0", "1.2.0", "1.1.0"], kind=BUMP_PATCH) == "1.2.1"
    assert next_version(["1.2.0", "1.0.0"], kind=BUMP_MINOR) == "1.3.0"


def test_next_version_handles_double_digit_minor() -> None:
    """两位数小版本号也不能排错（1.9.0 → 1.10.0 而不是 2.0.0）。"""
    assert next_version(["1.9.0"], kind=BUMP_MINOR) == "1.10.0"
    assert next_version(["1.10.0"], kind=BUMP_PATCH) == "1.10.1"


# --------------------------------------------------------------------------- #
# VersionConflict
# --------------------------------------------------------------------------- #


def test_version_conflict_projects_and_prints_fields() -> None:
    """冲突报告必须给出"哪几个字段不同"——只写 conflict 的话，运维只能去翻文件。"""
    conflict = VersionConflict(
        key="abc123",
        existing_version="1.0.0",
        incoming_version="1.0.1",
        fields={"adapter_sha256": ("a" * 8, "b" * 8)},
    )
    payload = conflict.to_dict()
    assert payload["key"] == "abc123"
    assert payload["fields"]["adapter_sha256"] == ["a" * 8, "b" * 8]
    line = conflict.summary_line()
    assert "1.0.0" in line and "1.0.1" in line and "adapter_sha256" in line


def test_version_conflict_without_fields_says_none() -> None:
    """没有字段差异时摘要里写"无"，不留一个空括号让人猜。"""
    conflict = VersionConflict(key="k", existing_version="1.0.0", incoming_version="1.0.0")
    assert "无" in conflict.summary_line()
