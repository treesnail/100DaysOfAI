"""工具权限策略：白名单/黑名单模式的工具调用准入控制."""

from __future__ import annotations


class ToolPermissionError(PermissionError):
    """工具调用被权限策略拒绝."""


class ToolPermissionPolicy:
    """工具权限策略.

    两种互斥模式：
      - 白名单模式（whitelist 非 None）：仅允许名单内的工具；
      - 黑名单模式（blacklist）：禁止名单内的工具，其余放行。
    白名单优先于黑名单。
    """

    def __init__(
        self,
        whitelist: list[str] | None = None,
        blacklist: list[str] | None = None,
    ) -> None:
        self._whitelist = set(whitelist) if whitelist is not None else None
        self._blacklist = set(blacklist or [])

    def check(self, tool_name: str) -> bool:
        """检查工具是否允许调用；拒绝时抛出 ToolPermissionError."""
        if self._whitelist is not None and tool_name not in self._whitelist:
            raise ToolPermissionError(
                f"工具 {tool_name} 不在白名单中，调用被拒绝"
            )
        if tool_name in self._blacklist:
            raise ToolPermissionError(
                f"工具 {tool_name} 在黑名单中，调用被拒绝"
            )
        return True
