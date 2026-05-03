"""Minions 交互式 CLI 外壳 — prompt_toolkit + rich 风格。"""

from __future__ import annotations

import asyncio
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import Suggestion
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import box

from voice_agent.event_bus import EventBus
from voice_agent.logger import get_logger
from voice_agent.core.agent_core import AgentCore
from voice_agent.core.conversation_state import ConversationState
from voice_agent.core.llm_client import LLMClient

_HISTORY_FILE = Path.home() / ".minions_history"


def _list_devices_str() -> str:
    """返回格式化的音频设备列表。"""
    import sounddevice as sd

    lines = []
    lines.append("可用音频设备（输入设备标记为 🎤 ）")
    lines.append("-" * 60)
    for i, dev in enumerate(sd.query_devices()):
        name = dev["name"]
        inputs = dev["max_input_channels"]
        outputs = dev["max_output_channels"]
        sr = dev["default_samplerate"]
        marker = "🎤" if inputs > 0 else "  "
        io = f"in={inputs} out={outputs}"
        lines.append(f"  {marker} [{i:2d}] {name}  {io}  {sr:.0f} Hz")
    lines.append(f"  默认输入: {sd.default.device[0]}")
    return "\n".join(lines)


def _resolve_device(device_arg: str) -> int | str | None:
    """解析 --device 参数为数字 ID 或字符串。"""
    if device_arg is None:
        return None
    if isinstance(device_arg, str) and device_arg.isdigit():
        return int(device_arg)
    return device_arg


class MinionsShell:
    """交互式 CLI 外壳。

    订阅 EventBus 事件并展示，同时提供 prompt_toolkit 输入。
    用户输入以 asr.final 事件发布，经 AgentCore 处理后显示回复。
    """

    def __init__(
        self,
        bus: EventBus,
        agent: AgentCore,
        state: ConversationState,
        llm: LLMClient,
    ) -> None:
        self._bus = bus
        self._agent = agent
        self._state = state
        self._llm = llm
        self._logger = get_logger()

        # 状态
        self.running = True
        self._paused = False
        self._mic_device: int | str | None = None

        # 事件队列
        self._event_queue: asyncio.Queue[dict] = asyncio.Queue()

        # Rich 控制台
        self._console = Console(highlight=False)

        # 快捷键
        kb = KeyBindings()

        @kb.add("right", filter=~self._session_is_searching)  # type: ignore[arg-type]
        def _accept_suggestion(event: object) -> None:
            """按右键接受 auto-suggest。"""
            buf = self._session.app.current_buffer
            if buf.suggestion:
                buf.insert_text(buf.suggestion.text)

        # prompt_toolkit 会话
        self._session = PromptSession[str](
            history=FileHistory(str(_HISTORY_FILE)),
            enable_history_search=True,
            complete_while_typing=False,
            auto_suggest=self._auto_suggest_command,
            key_bindings=kb,
        )

    @staticmethod
    def _session_is_searching() -> bool:
        """判断是否处于历史搜索模式（此时不禁用右键）。"""
        return False

    # ---- 事件订阅 ----

    async def _on_event(self, event: dict) -> None:
        await self._event_queue.put(event)

    def subscribe(self) -> None:
        self._bus.subscribe(self._on_event)

    def unsubscribe(self) -> None:
        self._bus.unsubscribe(self._on_event)

    # ---- 启动 ----

    async def run(self) -> None:
        """启动 CLI 主循环（双任务：输入 + 事件显示）。"""
        self._print_welcome()

        input_task = asyncio.create_task(self._input_loop())
        event_task = asyncio.create_task(self._event_loop())

        try:
            done, pending = await asyncio.wait(
                [input_task, event_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
        except asyncio.CancelledError:
            pass

        self._console.print()

    # ---- 输入循环 ----

    async def _input_loop(self) -> None:
        while self.running:
            try:
                text = await self._session.prompt_async(
                    "❯ ",
                    bottom_toolbar=self._toolbar,
                )
            except (KeyboardInterrupt, EOFError):
                await self._cmd_exit()
                return

            text = text.strip()
            if not text:
                continue

            if text.startswith("/"):
                await self._dispatch_command(text)
            else:
                await self._publish_user_text(text)

    # ---- 事件显示循环 ----

    async def _event_loop(self) -> None:
        while self.running:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            etype = event.get("type", "")
            self._render_event(etype, event)

    # ---- 事件渲染 ----

    def _render_event(self, etype: str, event: dict) -> None:
        renderers = {
            "agent.reply": self._render_agent_reply,
            "gate.result": self._render_gate_result,
            "state.change": self._render_state_change,
            "user.text": self._render_user_text,
            "bubble": self._render_bubble,
            "system": self._render_system,
        }
        handler = renderers.get(etype)
        if handler:
            handler(event)

    def _render_agent_reply(self, event: dict) -> None:
        text = event.get("text", "")
        panel = Panel(
            Text(text, style="green"),
            title="[bold green]Agent[/]",
            border_style="green",
            box=box.ROUNDED,
            padding=(0, 1),
        )
        self._console.print(panel)

    def _render_gate_result(self, event: dict) -> None:
        action = event.get("action", "")
        score = event.get("score", 0)
        reason = event.get("reason", "")
        style_map = {
            "silent": "dim",
            "bubble": "dim yellow",
            "judge": "yellow",
            "agent": "bold green",
        }
        s = style_map.get(action, "white")
        self._console.print(Text(f"  ┊ Gate: {action} (score={score}) {reason}", style=s))

    def _render_state_change(self, event: dict) -> None:
        s = event.get("state", "")
        self._console.print(Text(f"  ┊ [状态] → {s}", style="magenta"))

    def _render_user_text(self, event: dict) -> None:
        text = event.get("text", "")
        self._console.print(Text(f"  ┊ 你: {text}", style="bold cyan"))

    def _render_bubble(self, event: dict) -> None:
        msg = event.get("message", "")
        self._console.print(Text(f"  ┊ {msg}", style="dim yellow"))

    def _render_system(self, event: dict) -> None:
        msg = event.get("message", "")
        self._console.print(Text(f"  ┊ {msg}", style="dim"))

    # ---- 用户输入发布 ----

    async def _publish_user_text(self, text: str) -> None:
        self._state.mark_user_final_text(text)
        await self._bus.publish({"type": "user.text", "text": text})
        await self._bus.publish({
            "type": "asr.final",
            "text": text,
            "confidence": 1.0,
        })

    # ---- 命令自动猜想 ----

    def _auto_suggest_command(self, buffer: object, document: object) -> Suggestion | None:
        """输入 /h 时淡色显示 /help，右键补全。"""
        from prompt_toolkit.document import Document

        doc: Document = document  # type: ignore[assignment]
        text = doc.text

        if not text.startswith("/"):
            return None

        for cmd in sorted(self.COMMANDS):
            if cmd.startswith(text) and len(cmd) > len(text):
                return Suggestion(cmd[len(text):])

        return None

    # ---- 命令处理 ----

    COMMANDS: dict[str, tuple[str, str]] = {
        "/exit": ("退出程序", "exit"),
        "/quit": ("退出程序", "exit"),
        "/help": ("显示帮助", "help"),
        "/h": ("显示帮助", "help"),
        "/pause": ("暂停 AI 回应", "pause"),
        "/resume": ("恢复 AI 回应", "resume"),
        "/clear": ("清屏", "clear"),
        "/mode": ("查看当前状态", "mode"),
        "/status": ("查看系统状态", "status"),
        "/mic": ("麦克风管理 (list/select/info)", "mic"),
    }

    async def _dispatch_command(self, raw: str) -> None:
        parts = raw.strip().lower().split()
        cmd = parts[0]
        args = parts[1:]

        info = self.COMMANDS.get(cmd)
        if info is None:
            self._console.print(Text(f"  未知命令: {cmd}   输入 /help 查看帮助", style="red"))
            return

        handler_name = f"_cmd_{info[1]}"
        handler = getattr(self, handler_name, None)
        if handler:
            await handler(*args)

    async def _cmd_help(self) -> None:
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        table.add_column("命令", style="bold cyan")
        table.add_column("说明", style="white")

        # 去重显示（/h 和 /help 只显示一次）
        seen = set()
        for cmd, (desc, _) in sorted(self.COMMANDS.items()):
            if desc not in seen:
                seen.add(desc)
                table.add_row(cmd, desc)
            else:
                table.add_row(cmd, "")
        self._console.print(table)

    async def _cmd_exit(self) -> None:
        self.running = False
        await self._bus.publish({"type": "command.exit"})
        self._console.print(Text("  ┊ 再见 👋", style="dim"))

    async def _cmd_pause(self) -> None:
        self._paused = True
        await self._bus.publish({"type": "command.pause"})
        self._console.print(Text("  ┊ 已暂停", style="magenta"))

    async def _cmd_resume(self) -> None:
        self._paused = False
        await self._bus.publish({"type": "command.resume"})
        self._console.print(Text("  ┊ 已恢复", style="magenta"))

    async def _cmd_clear(self) -> None:
        self._console.clear()

    async def _cmd_mode(self) -> None:
        self._console.print(Text(f"  当前状态: {self._state.mode}", style="cyan"))
        self._console.print(Text(f"  active_until: {self._state.active_until:.1f}", style="dim"))
        self._console.print(Text(f"  cooldown_until: {self._state.cooldown_until:.1f}", style="dim"))

    async def _cmd_status(self) -> None:
        self._console.print(Text(f"  模式: {'暂停' if self._paused else '运行'}", style="cyan"))
        self._console.print(Text(f"  LLM 可用: {self._llm.is_available}", style="cyan"))
        self._console.print(Text(f"  状态: {self._state.mode}", style="cyan"))
        self._console.print(Text(f"  麦克风: {self._mic_device or '默认'}", style="cyan"))

    async def _cmd_mic(self, *args: str) -> None:
        """处理 /mic 子命令。"""
        sub = args[0] if args else "help"

        if sub == "list":
            self._console.print(Text(_list_devices_str(), style="cyan"))

        elif sub == "select" and len(args) >= 2:
            device_arg = args[1]
            self._mic_device = _resolve_device(device_arg)
            self._console.print(Text(f"  已选择麦克风设备: {self._mic_device}", style="green"))

        elif sub == "info":
            import sounddevice as sd

            default = sd.default.device[0]
            cur = self._mic_device if self._mic_device is not None else default
            try:
                info = sd.query_devices(cur)
                self._console.print(Text(f"  当前设备 [{cur}]: {info['name']}", style="cyan"))
                self._console.print(Text(f"  采样率: {info['default_samplerate']:.0f} Hz", style="dim"))
                self._console.print(Text(f"  输入通道: {info['max_input_channels']}", style="dim"))
            except Exception as e:
                self._console.print(Text(f"  设备查询失败: {e}", style="red"))

        else:
            table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
            table.add_column("子命令", style="bold cyan")
            table.add_column("说明", style="white")
            table.add_row("/mic list", "列出所有音频设备")
            table.add_row("/mic select <id|名称>", "选择麦克风设备")
            table.add_row("/mic info", "查看当前麦克风信息")
            self._console.print(table)

    # ---- 工具栏 ----

    def _toolbar(self) -> str:
        if self._paused:
            return " [red]已暂停[/red]  输入 /resume 恢复"
        return " [dim]输入 /help 查看命令 | → 补全建议 | ↑↓ 历史[/dim]"

    # ---- 欢迎信息 ----

    def _print_welcome(self) -> None:
        self._console.print()
        header = Panel(
            Text("\n".join([
                "Minions — 常驻语音 Agent  CLI",
                "",
                "直接输入文字与 AI 对话，或输入 /help 查看命令",
            ])),
            title="[bold cyan]Minions[/]",
            border_style="cyan",
            box=box.DOUBLE,
            padding=(1, 2),
        )
        self._console.print(header)
        self._console.print()
