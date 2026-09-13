"""day044 PII 扩展测试：银行卡号（Luhn 校验）与 IPv4 地址的检测脱敏（全部离线）.

day031 已覆盖手机号/身份证/邮箱；本文件聚焦 day044 新增的两类 PII，
并重点验证「校验型 PII」的误报防护——银行卡号与订单号/时间戳同形，
Luhn 校验是"确认是卡号"的数学依据。
"""

from __future__ import annotations

from smart_research_agent.security.content_moderator import (
    ContentModerator,
    _luhn_valid,
    _mask_bank_cards,
)


class TestLuhn:
    def test_valid_card_number(self):
        # 4111 1111 1111 1111 是经典 Visa 测试卡号，Luhn 校验和能被 10 整除
        assert _luhn_valid("4111111111111111") is True

    def test_invalid_card_number(self):
        # 改最后一位破坏校验和
        assert _luhn_valid("4111111111111112") is False

    def test_single_digit_and_empty(self):
        assert _luhn_valid("0") is True  # 0 % 10 == 0
        assert _luhn_valid("7") is False


class TestBankCardMasking:
    def test_plain_card_masked(self):
        text, hits = _mask_bank_cards("卡号是 4111111111111111，请查收")
        assert hits is True
        assert "4111111111111111" not in text
        assert "***银行卡号***" in text

    def test_card_with_spaces_masked(self):
        text, hits = _mask_bank_cards("4111 1111 1111 1111")
        assert hits is True
        assert "***银行卡号***" in text

    def test_card_with_hyphens_masked(self):
        text, hits = _mask_bank_cards("4111-1111-1111-1111")
        assert hits is True
        assert "***银行卡号***" in text

    def test_luhn_invalid_long_number_preserved(self):
        """Luhn 不过的长数字串（如订单号）不应被误脱敏."""
        order = "订单号 4111111111111112 请核对"
        text, hits = _mask_bank_cards(order)
        assert hits is False
        assert "4111111111111112" in text  # 原样保留

    def test_multiple_cards_all_masked(self):
        text, hits = _mask_bank_cards("A: 4111111111111111 和 B: 5555555555554444")
        assert hits is True
        assert "4111111111111111" not in text
        assert "5555555555554444" not in text
        assert text.count("***银行卡号***") == 2


class TestIPv4Masking:
    def test_private_ip_masked(self):
        moderator = ContentModerator()
        result = moderator.moderate("服务器地址是 192.168.1.1，请连接")
        assert "IPv4 地址" in result.pii_types
        assert "192.168.1.1" not in result.sanitized_text
        assert "***IPv4地址***" in result.sanitized_text

    def test_out_of_range_not_masked(self):
        """每段超过 255 的伪 IP（999.999.999.999）不是合法地址，不应误判."""
        moderator = ContentModerator()
        result = moderator.moderate("端口 999.999.999.999")
        assert result.is_safe is True
        assert "999.999.999.999" in result.sanitized_text

    def test_loopback_and_edge_octets(self):
        moderator = ContentModerator()
        for ip in ["0.0.0.0", "10.0.0.1", "255.255.255.255"]:
            result = moderator.moderate(f"IP 是 {ip}")
            assert "IPv4 地址" in result.pii_types, ip
            assert ip not in result.sanitized_text


class TestModeratorIntegration:
    def test_bank_card_reported_in_pii_types(self):
        moderator = ContentModerator()
        result = moderator.moderate("我的银行卡 4111111111111111 和手机 13812345678")
        assert "银行卡号" in result.pii_types
        assert "手机号" in result.pii_types
        assert result.is_safe is False

    def test_clean_text_still_safe(self):
        moderator = ContentModerator()
        result = moderator.moderate("今天的天气不错，适合学习。")
        assert result.is_safe is True
        assert result.pii_types == []
        assert result.sanitized_text == "今天的天气不错，适合学习。"
