"""Minions 交互式 CLI 外壳 — prompt_toolkit + rich 风格。"""

from __future__ import annotations

import asyncio
import contextlib
import math
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
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
from voice_agent.audio.microphone import Microphone, calculate_rms

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
    if device_arg is None:
        return None
    if isinstance(device_arg, str) and device_arg.isdigit():
        return int(device_arg)
    return device_arg


def _get_mic_info(device_id: int | str | None = None) -> dict:
    import sounddevice as sd

    did = device_id if device_id is not None else sd.default.device[0]
    try:
        info = sd.query_devices(did)
        return {
            "id": did,
            "name": info["name"],
            "sr": info["default_samplerate"],
            "channels": info["max_input_channels"],
            "valid": info["max_input_channels"] > 0,
        }
    except Exception:
        return {"id": did, "name": str(did), "sr": 0, "channels": 0, "valid": False}


def _vu_ascii(rms: float, width: int = 30) -> str:
    """将 RMS 音量转为 ASCII VU 表。"""
    if rms <= 0:
        return "░" * width
    # 用对数刻度更符合听觉：-60dB ~ 0dB 映射到 0 ~ 1
    db = 20 * math.log10(max(rms, 1e-6))
    norm = max(0.0, min(1.0, (db + 60) / 60))
    filled = int(norm * width)
    filled = max(0, min(filled, width))
    bar = "█" * filled + "░" * (width - filled)
    return bar


def _probe_device_rms(device_id: int, sample_rate: int = 16000, duration: float = 0.3) -> float:
    """快速探测指定设备的音频输入音量，返回 RMS，失败返回 -1。"""
    import sounddevice as sd

    try:
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            device=device_id,
            dtype="float32",
        )
        sd.wait()
        return calculate_rms(recording)
    except Exception:
        return -1.0


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
        mic: Microphone | None = None,
        asr_engine: object | None = None,
    ) -> None:
        self._bus = bus
        self._agent = agent
        self._state = state
        self._llm = llm
        self._mic = mic
        self._asr_engine = asr_engine
        self._logger = get_logger()

        # 状态
        self.running = True
        self._paused = False
        self._mic_device: int | str | None = None

        # 麦克风监测
        self._mic_monitoring = False
        self._latest_rms = 0.0
        self._mic_monitor_task: asyncio.Task | None = None

        # 事件队列
        self._event_queue: asyncio.Queue[dict] = asyncio.Queue()

        # Rich 控制台
        self._console = Console(highlight=False)

        # 快捷键
        kb = KeyBindings()

        @kb.add("right")
        def _accept_suggestion(event: object) -> None:
            buf = self._session.app.current_buffer
            if buf.suggestion:
                buf.insert_text(buf.suggestion.text)

        # prompt_toolkit 会话
        self._session = PromptSession[str](
            history=FileHistory(str(_HISTORY_FILE)),
            enable_history_search=True,
            complete_while_typing=False,
            auto_suggest=self._CommandAutoSuggest(self.COMMANDS),
            key_bindings=kb,
        )

    # ---- 事件订阅 ----

    async def _on_event(self, event: dict) -> None:
        await self._event_queue.put(event)

    def subscribe(self) -> None:
        self._bus.subscribe(self._on_event)

    def unsubscribe(self) -> None:
        self._bus.unsubscribe(self._on_event)

    # ---- 启动 ----

    async def run(self) -> None:
        self._print_welcome()

        input_task = asyncio.create_task(self._input_loop())
        event_task = asyncio.create_task(self._event_loop())

        # 收集所有需要并发运行的任务
        tasks = [input_task, event_task]

        # 启动 ASR 引擎（如果传入）
        if self._asr_engine is not None:
            asr_task = asyncio.create_task(self._asr_engine.start())
            tasks.append(asr_task)
        else:
            asr_task = None

        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )

            for task in pending:
                task.cancel()

            for task in pending:
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        finally:
            # 清理麦克风监测
            if self._mic_monitor_task:
                self._mic_monitor_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._mic_monitor_task
                self._mic_monitor_task = None

            if self._mic_monitoring:
                self._mic_monitoring = False
                if self._mic is not None:
                    with contextlib.suppress(Exception):
                        await self._mic.stop()

            # 取消事件订阅，防止重复订阅
            self.unsubscribe()

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
                # /exit 会把 running 设为 False，必须立刻退出
                if not self.running:
                    return
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
        self._logger.debug("[CLI] event: %s", etype)

        if etype == "audio.level":
            self._latest_rms = event.get("rms", 0.0)
            return  # VU 表在 toolbar 显示，不打印到主区域

        renderers = {
            "agent.reply": self._render_agent_reply,
            "gate.result": self._render_gate_result,
            "state.change": self._render_state_change,
            "user.text": self._render_user_text,
            "bubble": self._render_bubble,
            "system": self._render_system,
            "asr.speech_start": self._render_asr_speech_start,
            "asr.speech_end": self._render_asr_speech_end,
            "asr.final": self._render_asr_final,
            "asr.error": self._render_asr_error,
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

    def _render_asr_speech_start(self, event: dict) -> None:
        self._console.print(Text("  ┊ [语音] 检测到语音开始...", style="bold cyan"))

    def _render_asr_speech_end(self, event: dict) -> None:
        dur = event.get("duration_ms", 0)
        forced = event.get("forced", False)
        tag = "（强制截断）" if forced else ""
        self._console.print(Text(f"  ┊ [语音] 结束 ({dur}ms{tag}), 识别中...", style="cyan"))

    def _render_asr_final(self, event: dict) -> None:
        text = event.get("text", "")
        self._console.print(Text(f"  ┊ [ASR] {text}", style="bold cyan"))

    def _render_asr_error(self, event: dict) -> None:
        msg = event.get("message", "")
        self._console.print(Text(f"  ┊ [ASR 错误] {msg}", style="bold red"))

    # ---- 用户输入发布 ----

    async def _publish_user_text(self, text: str) -> None:
        """发布用户输入。

        注意：
        - CLI 只负责把输入发布为事件。
        - 不要在这里调用 state.mark_user_final_text()。
        - ConversationState 必须由 AgentCore.handle_final_text() 统一更新。
        - 否则 Gate 会把当前输入误判为重复文本。
        """
        self._logger.info("[CLI] 用户输入: %s", text)

        await self._bus.publish({
            "type": "user.text",
            "text": text,
            "source": "cli",
        })

        await self._bus.publish({
            "type": "asr.final",
            "text": text,
            "confidence": 1.0,
            "source": "cli",
        })

    # ---- 命令自动猜想 ----

    class _CommandAutoSuggest(AutoSuggest):
        def __init__(self, commands: dict[str, tuple[str, str]]) -> None:
            self._cmd_list = sorted(commands)

        def get_suggestion(
            self,
            buffer: object,
            document: object,
        ) -> Suggestion | None:
            from prompt_toolkit.document import Document

            doc: Document = document  # type: ignore[assignment]
            text = doc.text

            if not text.startswith("/"):
                return None

            for cmd in self._cmd_list:
                if cmd.startswith(text) and len(cmd) > len(text):
                    return Suggestion(cmd[len(text):])

            return None

    # ---- 麦克风监测 ----

    async def _mic_monitor_loop(self) -> None:
        """后台任务：持续采集麦克风并更新 RMS。"""
        if self._mic is None:
            return
        try:
            await self._mic.start()
            while self._mic_monitoring and self.running:
                chunk = await self._mic.read_chunk()
                rms = calculate_rms(chunk)
                self._latest_rms = rms
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self._console.print(Text(f"  ⚠️ 麦克风采集异常: {e}", style="red"))
        finally:
            await self._mic.stop()

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
        "/mic": ("麦克风管理 (monitor/list/select/info)", "mic"),
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

        seen = set()
        for cmd, (desc, _) in sorted(self.COMMANDS.items()):
            if desc not in seen:
                seen.add(desc)
                table.add_row(cmd, desc)
            else:
                table.add_row(cmd, "")
        self._console.print(table)

    async def _cmd_exit(self) -> None:
        """退出 CLI。必须确保输入循环和事件循环都能结束。"""
        self.running = False

        # 停止麦克风监测任务
        if self._mic_monitor_task:
            self._mic_monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._mic_monitor_task
            self._mic_monitor_task = None

        self._mic_monitoring = False

        # 停止麦克风
        if self._mic is not None:
            with contextlib.suppress(Exception):
                await self._mic.stop()

        # 停止 ASR 引擎
        if self._asr_engine is not None:
            await self._asr_engine.stop()

        # 通知主程序关闭 LLM/WebSocket 等资源
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
        mic = _get_mic_info(self._mic_device)
        mic_tag = f"🎤 {mic['name']}" if mic["valid"] else f"⚠️  {mic['name']}（无输入通道）"
        monitoring = "🎤 监测中" if self._mic_monitoring else "已停止"

        self._console.print(Text(f"  模式: {'暂停' if self._paused else '运行'}", style="cyan"))
        self._console.print(Text(f"  LLM 可用: {self._llm.is_available}", style="cyan"))
        self._console.print(Text(f"  状态: {self._state.mode}", style="cyan"))
        self._console.print(Text(f"  麦克风: {mic_tag}", style="cyan"))
        self._console.print(Text(f"  监测: {monitoring}", style="cyan"))

    async def _cmd_mic(self, *args: str) -> None:
        sub = args[0] if args else "help"

        if sub == "list":
            self._console.print(Text(_list_devices_str(), style="cyan"))

        elif sub == "select" and len(args) >= 2:
            device_arg = args[1]
            self._mic_device = _resolve_device(device_arg)
            self._console.print(Text(f"  已选择麦克风设备: {self._mic_device}", style="green"))

        elif sub == "info":
            mic = _get_mic_info(self._mic_device)
            valid_text = "✅ 有效（输入设备）" if mic["valid"] else "❌ 无效（无输入通道）"
            self._console.print(Text(f"  设备: [{mic['id']}] {mic['name']}", style="bold cyan"))
            self._console.print(Text(f"  状态: {valid_text}", style="green" if mic["valid"] else "red"))
            self._console.print(Text(f"  采样率: {mic['sr']:.0f} Hz", style="dim"))
            self._console.print(Text(f"  输入通道: {mic['channels']}", style="dim"))

        elif sub == "monitor":
            if self._mic is None:
                self._console.print(Text("  ❌ 未配置麦克风，启动时未传入 mic 参数", style="red"))
                return

            self._mic_monitoring = not self._mic_monitoring
            if self._mic_monitoring:
                self._mic_monitor_task = asyncio.create_task(self._mic_monitor_loop())
                self._console.print(Text("  🎤 麦克风监测已启动 — 底栏会显示实时音量", style="green"))
            else:
                if self._mic_monitor_task:
                    self._mic_monitor_task.cancel()
                    self._mic_monitor_task = None
                self._latest_rms = 0.0
                self._console.print(Text("  🎤 麦克风监测已停止", style="dim"))

        elif sub == "autodetect":
            await self._cmd_mic_autodetect(*args)

        else:
            table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
            table.add_column("子命令", style="bold cyan")
            table.add_column("说明", style="white")
            table.add_row("/mic monitor", "切换麦克风实时监测（底栏 VU 表）")
            table.add_row("/mic autodetect", "自动检测有音频输入的麦克风")
            table.add_row("/mic autodetect --select", "自动检测并切换到最佳设备")
            table.add_row("/mic list", "列出所有音频设备")
            table.add_row("/mic select <id|名称>", "选择麦克风设备")
            table.add_row("/mic info", "查看当前麦克风信息")
            self._console.print(table)

    async def _cmd_mic_autodetect(self, *args: str) -> None:
        """自动检测所有输入设备的音频音量，选出最佳麦克风。"""
        import sounddevice as sd

        self._console.print(Text("  🔍 正在逐个探测麦克风设备（每设备 300ms）...", style="bold cyan"))

        loop = asyncio.get_running_loop()
        devices = sd.query_devices()
        results: list[tuple[int, str, float]] = []

        for i, dev in enumerate(devices):
            if dev["max_input_channels"] <= 0:
                continue

            self._console.print(Text(f"    探测 [{i}] {dev['name']} ... ", style="dim"), end="")
            rms = await loop.run_in_executor(None, _probe_device_rms, i, 16000, 0.3)
            results.append((i, dev["name"], rms))
            if rms >= 0:
                bar = _vu_ascii(rms, 15)
                db = 20 * math.log10(max(rms, 1e-6))
                self._console.print(Text(f"{bar}  {rms:.6f}  ({db:.0f} dB)", style="green" if rms > 0.005 else "dim"))
            else:
                self._console.print(Text("❌ 打开失败", style="red"))

        results.sort(key=lambda x: x[2], reverse=True)

        self._console.print()
        table = Table(box=box.SIMPLE, show_header=True, padding=(0, 2))
        table.add_column("排名", style="bold", justify="right")
        table.add_column("设备", style="cyan")
        table.add_column("RMS", style="white", justify="right")
        table.add_column("音量条", style="white")
        for rank, (did, name, rms) in enumerate(results, 1):
            bar = _vu_ascii(rms, 20)
            rms_str = f"{rms:.6f}" if rms >= 0 else "❌"
            style = "green" if rank == 1 else "dim"
            table.add_row(f"#{rank}", f"[{did}] {name}", rms_str, bar)
        self._console.print(table)

        # 自动选择
        auto_select = "--select" in args or "-s" in args
        if results and results[0][2] > 0.005:
            best_id, best_name, best_rms = results[0]
            if auto_select:
                self._mic_device = best_id
                self._console.print(Text(f"  ✅ 已选择最佳设备: [{best_id}] {best_name}", style="bold green"))
                # 如果正在监测，切换到新设备
                if self._mic_monitoring and self._mic is not None:
                    self._console.print(Text("  正在重启麦克风监测使用新设备...", style="dim"))
                    if self._mic_monitor_task:
                        self._mic_monitor_task.cancel()
                        try:
                            await self._mic_monitor_task
                        except asyncio.CancelledError:
                            pass
                        self._mic_monitor_task = None
                    await self._mic.stop()
                    self._mic.device = best_id
                    self._mic_monitor_task = asyncio.create_task(self._mic_monitor_loop())
                    self._console.print(Text("  ✅ 监测已切换到新设备", style="green"))
            else:
                self._console.print(Text(f"  提示: 添加 --select 自动选择 [{best_id}] {best_name}", style="dim"))
        elif results and results[0][2] > 0:
            self._console.print(Text("  ⚠️ 所有设备音量极低，未自动选择", style="yellow"))
        else:
            self._console.print(Text("  ⚠️ 未检测到可用的麦克风设备", style="yellow"))

    # ---- 工具栏 ----

    def _toolbar(self) -> str:
        parts = []

        # VU 表（监测中）
        if self._mic_monitoring:
            bar = _vu_ascii(self._latest_rms)
            parts.append(f" 🎤{bar} {self._latest_rms:.4f} ")

        # 状态指示
        if self._paused:
            parts.append("[red]已暂停[/red]  输入 /resume 恢复")
        else:
            parts.append("[dim]输入 /help 查看命令 | → 补全建议 | ↑↓ 历史[/dim]")

        return " |".join(parts)

    # ---- 欢迎信息 ----

    def _print_welcome(self) -> None:
        mic = _get_mic_info()
        mic_status = f"🎤 {mic['name']}" if mic["valid"] else "⚠️  未检测到有效麦克风"

        self._console.print()
        header = Panel(
            Text("\n".join([
                "Minions — 常驻语音 Agent  CLI",
                "",
                f"  {mic_status}",
                "  输入 /mic monitor 启动实时音频监测",
                "直接输入文字与 AI 对话，或输入 /help 查看命令",
            ])),
            title="[bold cyan]Minions[/]",
            border_style="cyan",
            box=box.DOUBLE,
            padding=(1, 2),
        )
        self._console.print(header)
        self._console.print()
