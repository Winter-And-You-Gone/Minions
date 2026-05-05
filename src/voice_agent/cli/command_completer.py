"""Slash 命令补全 — 为 prompt_toolkit 输入框提供命令补全。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from prompt_toolkit.completion import Completer, Completion


@dataclass(frozen=True)
class CommandSpec:
    command: str
    description: str
    usage: str = ""
    aliases: tuple[str, ...] = ()
    subcommands: tuple[str, ...] = ()


COMMAND_SPECS: tuple[CommandSpec, ...] = (
    CommandSpec("/help", "显示帮助和所有命令", "/help", aliases=("/h",)),
    CommandSpec("/status", "查看当前 ASR / Judge / LLM / Wake 状态", "/status"),
    CommandSpec("/debug", "显示最近内部事件和健康检查", "/debug"),
    CommandSpec("/model", "查看当前主 LLM 模型", "/model"),
    CommandSpec("/pause", "暂停 AI 回应，但保留 TUI", "/pause"),
    CommandSpec("/resume", "恢复 AI 回应", "/resume"),
    CommandSpec("/clear", "清空当前聊天显示", "/clear"),
    CommandSpec("/exit", "退出 Minions", "/exit", aliases=("/quit",)),
    CommandSpec(
        "/name",
        "设置或查看 AI 名字和唤醒别名",
        "/name set 琉璃川",
        aliases=("/名字",),
        subcommands=("set", "alias", "save"),
    ),
    CommandSpec(
        "/mic",
        "麦克风管理：list/select/info/monitor/autodetect",
        "/mic list",
        subcommands=("list", "select", "info", "monitor", "autodetect"),
    ),
    CommandSpec("/mode", "查看当前对话状态", "/mode"),
    CommandSpec(
        "/wakeup",
        "叫醒 Minions，启动语音监听和 ASR",
        "/wakeup",
        aliases=("/wake", "/起床", "/叫醒"),
    ),
    CommandSpec(
        "/sleep",
        "让 Minions 进入待机，停止语音监听",
        "/sleep",
        aliases=("/standby", "/睡觉", "/休息"),
    ),
    CommandSpec(
        "/judge",
        "查看或切换判断器：rule/local/llm",
        "/judge rule",
        subcommands=("rule", "local", "llm"),
    ),
)


def iter_command_specs() -> Iterable[CommandSpec]:
    return COMMAND_SPECS


class MinionsCommandCompleter(Completer):
    """Slash command completer with descriptions."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        # 只在 slash 命令中补全
        if not text.startswith("/"):
            return

        parts = text.split()
        current = parts[-1] if parts else text

        # 第一段：补全命令
        if len(parts) <= 1 and not text.endswith(" "):
            for spec in COMMAND_SPECS:
                candidates = (spec.command, *spec.aliases)
                for cmd in candidates:
                    if cmd.startswith(current):
                        yield Completion(
                            cmd,
                            start_position=-len(current),
                            display=cmd,
                            display_meta=spec.description,
                        )
            return

        # 第二段：补全子命令
        command = parts[0]
        spec = None
        for s in COMMAND_SPECS:
            if command == s.command or command in s.aliases:
                spec = s
                break

        if spec is None:
            return

        if spec.subcommands and len(parts) <= 2:
            sub_current = "" if text.endswith(" ") else current
            for sub in spec.subcommands:
                if sub.startswith(sub_current):
                    yield Completion(
                        sub,
                        start_position=-len(sub_current),
                        display=sub,
                        display_meta=f"{spec.command} {sub}",
                    )
