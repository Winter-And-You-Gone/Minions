"""Minions 交互式 CLI 外壳 — prompt_toolkit + rich 风格。"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.style import Style
from rich.table import Table
from rich import box

from voice_agent.event_bus import EventBus
from voice_agent.logger import get_logger
from voice_agent.core.agent_core import AgentCore
from voice_agent.core.conversation_state import ConversationState
from voice_agent.core.llm_client import LLMClient

_HISTORY_FILE = Path.home() / ".minions_history"


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

        # 事件队列：异步任务将事件推入此队列，显示任务消费
        self._event_queue: asyncio.Queue[dict] = asyncio.Queue()

        # Rich 控制台
        self._console = Console(highlight=False)

        # prompt_toolkit 会话
        self._session = PromptSession[str](
            history=FileHistory(str(_HISTORY_FILE)),
            enable_history_search=True,
            complete_while_typing=False,
        )

    # ---- 事件订阅 ----

    async def _on_event(self, event: dict) -> None:
        """EventBus 回调：所有事件入队等待显示。"""
        await self._event_queue.put(event)

    def subscribe(self) -> None:
        """注册到 EventBus。"""
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
        """持续读取用户输入并发布事件。"""
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
        """消费事件队列并实时显示。"""
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
        """根据事件类型渲染到终端。"""
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
        """将用户文本发布到总线（模拟 ASR final）。"""
        self._state.mark_user_final_text(text)
        await self._bus.publish({"type": "user.text", "text": text})
        await self._bus.publish({
            "type": "asr.final",
            "text": text,
            "confidence": 1.0,
        })

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
    }

    async def _dispatch_command(self, raw: str) -> None:
        cmd = raw.strip().lower().split()[0]
        info = self.COMMANDS.get(cmd)
        if info is None:
            self._console.print(Text(f"  未知命令: {cmd}   输入 /help 查看帮助", style="red"))
            return

        handler_name = f"_cmd_{info[1]}"
        handler = getattr(self, handler_name, None)
        if handler:
            await handler()

    async def _cmd_help(self) -> None:
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        table.add_column("命令", style="bold cyan")
        table.add_column("说明", style="white")
        for cmd, desc in sorted(self.COMMANDS.items()):
            table.add_row(cmd, desc[0])
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

    # ---- 工具栏 ----

    def _toolbar(self) -> str:
        """prompt_toolkit 底部工具栏。"""
        if self._paused:
            return " [red]已暂停[/red]  输入 /resume 恢复"
        return " [dim]输入 /help 查看命令 | Tab: 搜索历史[/dim]"

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
