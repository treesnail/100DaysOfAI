"""Prompt 版本管理库：注册、取用、渲染与持久化.

设计动机（day036）：
提示词是 Agent 系统的"源代码"——改动它会直接改变线上行为，因此需要与代码同级的
版本管理：每次修改生成新版本、记录变更原因（changelog）、可随时回查演进历史，
并与 evaluation/prompt_eval.py 的 PromptEvaluator 联动做版本间回归。

持久化格式：
- JSONL：一行一个模板（JSON 对象），适合追加写入与版本演进日志式存储；
- YAML：一个模板列表，适合人工阅读与手工编辑。
两者字段完全一致，均为 PromptTemplate 的五个字段。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

#: 模板占位符：{name} 形式，name 必须是合法标识符
_VAR_PATTERN = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


class PromptNotFoundError(KeyError):
    """按名称/版本查找模板失败."""


class PromptRenderError(ValueError):
    """渲染时缺少必填变量."""


@dataclass
class PromptTemplate:
    """一份带版本的提示词模板.

    字段：
      - name: 模板名（同一 name 下按 version 演进）
      - template: 模板文本，变量以 {var} 形式占位
      - version: 版本号，约定 "v1"、"v2"…… 递增
      - changelog: 本版本相对上一版的变更原因（prompt 即代码，必须留痕）
      - created_at: 注册时刻的 ISO 时间戳
    """

    name: str
    template: str
    version: str = "v1"
    changelog: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def variables(self) -> list[str]:
        """模板中出现的占位符变量名（去重、按出现顺序）."""
        seen: dict[str, None] = {}
        for match in _VAR_PATTERN.finditer(self.template):
            seen.setdefault(match.group(1))
        return list(seen)

    def render(self, **variables: str) -> str:
        """填充变量渲染模板；缺少任一变量即抛 PromptRenderError.

        用显式替换而非 str.format：模板中允许出现不属于变量的花括号
        （如 few-shot 示例里的 JSON 片段），只要不构成合法 {标识符} 占位符
        就不参与替换，避免 format 对花括号的全局转义要求。
        """
        missing = [v for v in self.variables if v not in variables]
        if missing:
            raise PromptRenderError(
                f"渲染模板 {self.name}@{self.version} 缺少变量: {', '.join(missing)}"
            )

        def _sub(match: re.Match[str]) -> str:
            key = match.group(1)
            return str(variables[key]) if key in variables else match.group(0)

        return _VAR_PATTERN.sub(_sub, self.template)

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "template": self.template,
            "version": self.version,
            "changelog": self.changelog,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PromptTemplate:
        return cls(
            name=str(data["name"]),
            template=str(data["template"]),
            version=str(data.get("version", "v1")),
            changelog=str(data.get("changelog", "")),
            created_at=str(data.get("created_at", ""))
            or datetime.now(timezone.utc).isoformat(),
        )


class PromptLibrary:
    """提示词模板的集中注册与版本管理.

    同一 name 可注册多个 version，按注册顺序构成演进历史；
    get() 不传版本时返回最新（最后注册）的一版。
    """

    def __init__(self) -> None:
        self._prompts: dict[str, list[PromptTemplate]] = {}

    def register(self, template: PromptTemplate) -> PromptTemplate:
        """注册一个模板版本；同 name+version 重复注册抛 ValueError."""
        versions = self._prompts.setdefault(template.name, [])
        if any(t.version == template.version for t in versions):
            raise ValueError(f"模板重复注册: {template.name}@{template.version}")
        versions.append(template)
        return template

    def get(self, name: str, version: str | None = None) -> PromptTemplate:
        """按名取用模板；version 为 None 时取最新注册的一版."""
        versions = self._prompts.get(name)
        if not versions:
            raise PromptNotFoundError(f"未注册名为 {name} 的模板")
        if version is None:
            return versions[-1]
        for template in versions:
            if template.version == version:
                return template
        raise PromptNotFoundError(f"模板 {name} 不存在版本 {version}")

    def render(self, name: str, version: str | None = None, **variables: str) -> str:
        """取模板并渲染，缺失变量抛 PromptRenderError."""
        return self.get(name, version).render(**variables)

    def history(self, name: str) -> list[PromptTemplate]:
        """某模板的版本演进历史（按注册顺序，旧 -> 新）."""
        versions = self._prompts.get(name)
        if not versions:
            raise PromptNotFoundError(f"未注册名为 {name} 的模板")
        return list(versions)

    def names(self) -> list[str]:
        """所有已注册模板名."""
        return list(self._prompts)

    # ---- 持久化 ----

    def save_jsonl(self, path: str | Path) -> None:
        """全部模板按注册顺序写入 JSONL（一行一个版本）."""
        lines = [
            json.dumps(t.to_dict(), ensure_ascii=False)
            for templates in self._prompts.values()
            for t in templates
        ]
        Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def save_yaml(self, path: str | Path) -> None:
        """全部模板写入 YAML（模板列表，便于人工阅读编辑）."""
        data = [t.to_dict() for templates in self._prompts.values() for t in templates]
        Path(path).write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )

    def load_jsonl(self, path: str | Path) -> int:
        """从 JSONL 加载模板并注册，返回加载的模板数."""
        count = 0
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                self.register(PromptTemplate.from_dict(json.loads(line)))
                count += 1
        return count

    def load_yaml(self, path: str | Path) -> int:
        """从 YAML 加载模板并注册，返回加载的模板数."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
        for item in data:
            self.register(PromptTemplate.from_dict(item))
        return len(data)
