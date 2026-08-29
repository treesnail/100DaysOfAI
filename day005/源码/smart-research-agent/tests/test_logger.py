"""logger 模块单元测试：验证日志器创建与幂等性."""

from __future__ import annotations

from smart_research_agent.utils.logger import get_logger


class TestGetLogger:
    """get_logger 的行为契约测试."""

    def test_returns_logger_with_name(self):
        logger = get_logger("my_module")
        assert logger.name == "my_module"

    def test_handlers_not_duplicated_on_repeat_calls(self):
        """重复调用不能重复添加 handler，否则日志会输出多遍."""
        logger1 = get_logger("dup_test")
        handler_count = len(logger1.handlers)

        logger2 = get_logger("dup_test")
        assert logger1 is logger2
        assert len(logger2.handlers) == handler_count

    def test_has_console_and_file_handlers(self):
        logger = get_logger("channels_test")
        assert len(logger.handlers) == 2
