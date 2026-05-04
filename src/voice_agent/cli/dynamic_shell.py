"""状态驱动的动态 CLI 界面 — 使用 prompt_toolkit Application 实现全屏动态刷新。"""

from __future__ import annotations

import asyncio
import math
import time
from pathlib import Path

from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.history import FileHistory
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl

from voice_agent.cli.formatters import format_chat, format_decision, format_header, vu_bar
from voice_agent.cli.shell import (
    _get_mic_info,
    _list_devices_str,
    _probe_device_rms,
    _resolve_device,
)
from voice_agent.cli.ui_state import GateView, LLMView, UIState
from voice_agent.core.agent_core import AgentCore
from voice_agent.core.conversation_state import ConversationState
from voice_agent.core.llm_client import LLMClient
from voice_agent.event_bus import EventBus
from voice_agent.audio.microphone import Microphone, calculate_rms
from voice_agent.logger import get_logger

_HISTORY_FILE = Path.home() / ".minions_history"

_AUDIO_LEVEL_THROTTLE = 0.1  # seconds between audio.level invalidates


class DynamicMinionsShell:
    """状态驱动的动态 CLI 界面。

    使用 prompt_toolkit Application 全屏模式：
    事件 → 更新 UIState → app.invalidate() → 重渲染整个界面。
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
        self.ui = UIState(
            llm=LLMView(
                model=llm.model if llm.is_available and llm.model else "mock",
                available=llm.is_available,
            ),
        )
        self._mic_device: int | str | None = None
        self._mic_monitoring = False
        self._mic_monitor_task: asyncio.Task | None = None
        self._last_invalidate = 0.0

        # 输入缓冲区
        self._input_buffer = Buffer(
            accept_handler=self._on_accept,
            history=FileHistory(str(_HISTORY_FILE)),
        )

        # 先构建快捷键，再构建 UI（UI 需要 _kb）
        self._build_key_bindings()
        self._build_ui()

        # 欢迎信息
        self.ui.add_system_message("Minions — 常驻语音 Agent CLI")
        self.ui.add_system_message("输入 /help 查看命令，直接输入文字与 AI 对话")

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        header_win = Window(
            FormattedTextControl(lambda: format_header(self.ui)),
            height=2,
        )

        self._chat_win = Window(
            FormattedTextControl(lambda: format_chat(self.ui)),
            wrap_lines=True,
            always_hide_cursor=True,
        )

        decision_win = Window(
            FormattedTextControl(lambda: format_decision(self.ui)),
            height=1,
        )

        # 输入行: "你[普通] > "
        input_prompt = Window(
            FormattedTextControl(lambda: self._input_prompt_text()),
            width=12,
            height=1,
            dont_extend_width=True,
        )
        self._input_window = Window(
            BufferControl(buffer=self._input_buffer),
            height=1,
        )
        input_row = VSplit([input_prompt, self._input_window])

        root = HSplit([
            header_win,
            self._chat_win,
            decision_win,
            input_row,
        ])

        self._layout = Layout(root)
        self._layout.focus(self._input_window)

        self._app = Application(
            layout=self._layout,
            full_screen=True,
            key_bindings=self._kb,
            mouse_support=False,
        )
        self._app.bottom_toolbar = lambda: [
            ("ansibrightblack", " /help /status /model /pause /resume /clear /exit /mic "),
        ]

    def _input_prompt_text(self) -> list[tuple[str, str]]:
        mode = "暂停" if self.ui.paused else "普通"
        return [
            ("bold cyan", "你"),
            ("", f"[{mode}] > "),
        ]

    def _build_key_bindings(self) -> None:
        from prompt_toolkit.key_binding import KeyBindings

        self._kb = KeyBindings()

        @self._kb.add(Keys.Enter)
        def _accept(event: object) -> None:
            self._accept_buffer()

        @self._kb.add(Keys.ControlC)
        def _exit(event: object) -> None:
            asyncio.create_task(self._cmd_exit())

        @self._kb.add(Keys.Up)
        def _hist_back(event: object) -> None:
            self._input_buffer.history_backward()

        @self._kb.add(Keys.Down)
        def _hist_forward(event: object) -> None:
            self._input_buffer.history_forward()

    # ------------------------------------------------------------------
    # 事件订阅
    # ------------------------------------------------------------------

    async def on_event(self, event: dict) -> None:
        """EventBus 回调 — 更新 UIState 并刷新界面。"""
        etype = event.get("type", "")

        if etype == "audio.level":
            self.ui.mic.rms = event.get("rms", 0.0)
            now = time.monotonic()
            if now - self._last_invalidate > _AUDIO_LEVEL_THROTTLE:
                self._last_invalidate = now
                self._app.invalidate()
            return

        if etype == "agent.reply":
            self.ui.add_assistant_message(event.get("text", ""))
            self._scroll_to_bottom()

        elif etype == "gate.result":
            self.ui.latest_gate = GateView(
                action=event.get("action", ""),
                score=event.get("score", 0),
                reason=event.get("reason", ""),
            )

        elif etype == "state.change":
            self.ui.conversation_mode = event.get("state", "")

        elif etype == "bubble":
            msg = event.get("message", "")
            if msg:
                self.ui.add_system_message(f"[Bubble] {msg}")

        elif etype == "asr.speech_start":
            self.ui.add_system_message("检测到语音开始...")

        elif etype == "asr.speech_end":
            dur = event.get("duration_ms", 0)
            forced = event.get("forced", False)
            tag = "（强制截断）" if forced else ""
            self.ui.add_system_message(f"语音结束 ({dur}ms{tag})，识别中...")

        elif etype == "asr.final" and event.get("source") != "cli":
            self.ui.add_user_message(event.get("text", ""))
            self._scroll_to_bottom()

        elif etype == "asr.error":
            self.ui.asr.status = "error"
            self.ui.add_system_message(f"ASR 错误: {event.get('message', '')}")

        elif etype == "asr.status":
            self.ui.asr.status = event.get("status", "idle")
            if event.get("model"):
                self.ui.asr.model = event["model"]

        elif etype == "system":
            msg = event.get("message", "")
            if msg:
                self.ui.add_system_message(msg)

        elif etype == "judge.result":
            provider = event.get("provider", "")
            should_reply = event.get("should_reply", False)
            reason = event.get("reason", "")
            confidence = event.get("confidence", 0.0)
            self.ui.status_line = (
                f"Judge[{provider}]: "
                f"{'reply' if should_reply else 'silent'} "
                f"conf={confidence:.2f} {reason}"
            )
            self._app.invalidate()
            return

        # 更新唤醒会话状态栏
        if hasattr(self._state, "is_wake_session_active") and self._state.is_wake_session_active():
            remain = self._state.seconds_until_wake_session_timeout()
            wake_name = self._state.wake_name or "唤醒"
            self.ui.status_line = f"{wake_name} 唤醒会话中，剩余 {remain:.0f}s"
        else:
            self.ui.status_line = ""

        self._app.invalidate()

    def subscribe(self) -> None:
        self._bus.subscribe(self.on_event)

    def unsubscribe(self) -> None:
        self._bus.unsubscribe(self.on_event)

    def _scroll_to_bottom(self) -> None:
        """滚动聊天窗口到底部。"""
        self._chat_win.vertical_scroll = 999999

    # ------------------------------------------------------------------
    # 启动 / 停止
    # ------------------------------------------------------------------

    async def run(self) -> None:
        # 启动 ASR 引擎
        asr_task: asyncio.Task | None = None
        if self._asr_engine is not None:
            asr_task = asyncio.create_task(self._asr_engine.start())

        try:
            try:
                await self._app.run_async()
            except KeyboardInterrupt:
                await self._bus.publish({"type": "command.exit"})
        finally:
            if asr_task is not None:
                asr_task.cancel()
                try:
                    await asr_task
                except (asyncio.CancelledError, Exception):
                    pass
                await self._asr_engine.stop()
            # 停止 mic 监测
            if self._mic_monitor_task:
                self._mic_monitor_task.cancel()
                self._mic_monitor_task = None
            if self._mic_monitoring and self._mic:
                self._mic_monitoring = False
                await self._mic.stop()

    # ------------------------------------------------------------------
    # 输入处理
    # ------------------------------------------------------------------

    def _accept_buffer(self) -> None:
        """提交当前输入框内容。"""
        text = self._input_buffer.text.strip()
        self._input_buffer.text = ""
        if text:
            asyncio.create_task(self._handle_input(text))

    def _on_accept(self, buff: Buffer) -> bool:
        self._accept_buffer()
        return True

    async def _handle_input(self, text: str) -> None:
        if text.startswith("/"):
            await self._dispatch_command(text)
        else:
            self.ui.add_user_message(text)
            self._scroll_to_bottom()
            self._app.invalidate()

            await self._bus.publish({"type": "user.text", "text": text})
            await self._bus.publish({
                "type": "asr.final",
                "text": text,
                "confidence": 1.0,
                "source": "cli",
            })

    # ------------------------------------------------------------------
    # 命令处理
    # ------------------------------------------------------------------

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
        "/model": ("查看当前 LLM 模型", "model"),
        "/mic": ("麦克风管理", "mic"),
        "/name": ("设置或查看 AI 名字", "name"),
        "/名字": ("设置或查看 AI 名字", "name"),
    }

    async def _dispatch_command(self, raw: str) -> None:
        parts = raw.strip().lower().split()
        cmd = parts[0]
        args = parts[1:]

        info = self.COMMANDS.get(cmd)
        if info is None:
            self.ui.add_system_message(f"未知命令: {cmd}    输入 /help 查看帮助")
            self._app.invalidate()
            return

        handler_name = f"_cmd_{info[1]}"
        handler = getattr(self, handler_name, None)
        if handler:
            await handler(*args)

    async def _cmd_help(self) -> None:
        lines = ["可用命令："]
        seen = set()
        for cmd, (desc, _) in sorted(self.COMMANDS.items()):
            if desc not in seen:
                seen.add(desc)
                lines.append(f"  {cmd:<12s} {desc}")
            else:
                lines.append(f"  {cmd:<12s}  ")
        self.ui.add_system_message("\n".join(lines))
        self._app.invalidate()

    async def _cmd_exit(self) -> None:
        self.ui.running = False
        if self._mic_monitor_task:
            self._mic_monitor_task.cancel()
            self._mic_monitor_task = None
        self._mic_monitoring = False
        if self._asr_engine is not None:
            await self._asr_engine.stop()
        await self._bus.publish({"type": "command.exit"})
        self._app.exit()

    async def _cmd_pause(self) -> None:
        self.ui.paused = True
        await self._bus.publish({"type": "command.pause"})
        self._app.invalidate()

    async def _cmd_resume(self) -> None:
        self.ui.paused = False
        await self._bus.publish({"type": "command.resume"})
        self._app.invalidate()

    async def _cmd_clear(self) -> None:
        self.ui.messages.clear()
        self._app.invalidate()

    async def _cmd_mode(self) -> None:
        self.ui.add_system_message(
            f"状态: {self._state.mode}  active_until: {self._state.active_until:.1f}  "
            f"cooldown_until: {self._state.cooldown_until:.1f}"
        )
        self._app.invalidate()

    async def _cmd_status(self) -> None:
        mic = _get_mic_info(self._mic_device)
        mic_tag = f"{mic['name']}" if mic["valid"] else f"{mic['name']}（无输入通道）"
        monitoring = "监测中" if self._mic_monitoring else "已停止"

        self.ui.add_system_message(
            f"模式: {'暂停' if self.ui.paused else '运行'}  "
            f"LLM 可用: {self._llm.is_available}  "
            f"状态: {self._state.mode}  "
            f"麦克风: {mic_tag}  "
            f"监测: {monitoring}"
        )
        self._app.invalidate()

    async def _cmd_model(self) -> None:
        model = self._llm.model if self._llm.model else "mock"
        self.ui.add_system_message(f"当前连接的模型是：{model}")
        self._app.invalidate()

    async def _cmd_name(self, *args: str) -> None:
        matcher = getattr(self._agent._gate, "wake_matcher", None)

        if not args or not matcher:
            if matcher is None:
                self.ui.add_system_message("当前未启用唤醒名功能")
            else:
                cfg = matcher.config
                self.ui.add_system_message(
                    f"当前名字: {cfg.name}\n"
                    f"唤醒别名: {', '.join(cfg.aliases)}\n"
                    "用法:\n"
                    "  /name set 米粒\n"
                    "  /name alias add 迷你\n"
                    "  /name alias remove 迷你\n"
                    "  /name alias list"
                )
            self._app.invalidate()
            return

        if matcher is None:
            self.ui.add_system_message("当前未启用唤醒名功能")
            self._app.invalidate()
            return

        cfg = matcher.config

        if args[0] == "set" and len(args) >= 2:
            new_name = args[1]
            cfg.name = new_name
            if new_name not in cfg.aliases:
                cfg.aliases.insert(0, new_name)
            self.ui.add_system_message(f"AI 名字已设置为：{new_name}")
            self._app.invalidate()
            return

        if args[0] == "alias" and len(args) >= 2:
            action = args[1]

            if action == "add" and len(args) >= 3:
                alias = args[2]
                if alias not in cfg.aliases:
                    cfg.aliases.append(alias)
                self.ui.add_system_message(f"已添加唤醒别名：{alias}")
                self._app.invalidate()
                return

            if action == "remove" and len(args) >= 3:
                alias = args[2]
                cfg.aliases = [x for x in cfg.aliases if x != alias]
                self.ui.add_system_message(f"已移除唤醒别名：{alias}")
                self._app.invalidate()
                return

            if action == "list":
                self.ui.add_system_message(
                    f"当前名字: {cfg.name}\n唤醒别名: {', '.join(cfg.aliases)}"
                )
                self._app.invalidate()
                return

        self.ui.add_system_message(
            "用法:\n"
            "  /name\n"
            "  /name set 米粒\n"
            "  /name alias add 迷你\n"
            "  /name alias remove 迷你\n"
            "  /name alias list"
        )
        self._app.invalidate()

    async def _cmd_mic(self, *args: str) -> None:
        sub = args[0] if args else "help"

        if sub == "list":
            self.ui.add_system_message(_list_devices_str())
            self._app.invalidate()

        elif sub == "select" and len(args) >= 2:
            self._mic_device = _resolve_device(args[1])
            self.ui.mic.device_name = str(self._mic_device)
            self.ui.add_system_message(f"已选择麦克风设备: {self._mic_device}")
            self._app.invalidate()

        elif sub == "info":
            mic = _get_mic_info(self._mic_device)
            valid = "有效（输入设备）" if mic["valid"] else "无效（无输入通道）"
            self.ui.add_system_message(
                f"设备: [{mic['id']}] {mic['name']}  状态: {valid}  "
                f"采样率: {mic['sr']:.0f} Hz  输入通道: {mic['channels']}"
            )
            self._app.invalidate()

        elif sub == "monitor":
            await self._cmd_mic_monitor()

        elif sub == "autodetect":
            await self._cmd_mic_autodetect(*args)

        else:
            help_text = (
                "/mic monitor       切换麦克风实时监测\n"
                "/mic autodetect    自动检测有音频输入的麦克风\n"
                "/mic autodetect --select  自动检测并切换到最佳设备\n"
                "/mic list          列出所有音频设备\n"
                "/mic select <id|名称>  选择麦克风设备\n"
                "/mic info          查看当前麦克风信息"
            )
            self.ui.add_system_message(help_text)
            self._app.invalidate()

    async def _cmd_mic_monitor(self) -> None:
        if self._mic is None:
            self.ui.add_system_message("未配置麦克风，启动时未传入 mic 参数")
            self._app.invalidate()
            return

        self._mic_monitoring = not self._mic_monitoring
        self.ui.mic.monitoring = self._mic_monitoring

        if self._mic_monitoring:
            # 更新设备名称
            mic_info = _get_mic_info(self._mic_device)
            self.ui.mic.device_name = mic_info["name"] if mic_info["valid"] else ""
            self._mic_monitor_task = asyncio.create_task(self._mic_monitor_loop())
            self.ui.add_system_message("麦克风监测已启动")
        else:
            if self._mic_monitor_task:
                self._mic_monitor_task.cancel()
                self._mic_monitor_task = None
            self.ui.mic.rms = 0.0
            self.ui.add_system_message("麦克风监测已停止")

        self._app.invalidate()

    async def _cmd_mic_autodetect(self, *args: str) -> None:
        import sounddevice as sd

        self.ui.add_system_message("正在逐个探测麦克风设备（每设备 300ms）...")

        loop = asyncio.get_running_loop()
        devices = sd.query_devices()
        results: list[tuple[int, str, float]] = []

        for i, dev in enumerate(devices):
            if dev["max_input_channels"] <= 0:
                continue

            self.ui.status_line = f"探测 [{i}] {dev['name']} ..."
            self._app.invalidate()

            rms = await loop.run_in_executor(None, _probe_device_rms, i, 16000, 0.3)
            results.append((i, dev["name"], rms))

        self.ui.status_line = ""
        results.sort(key=lambda x: x[2], reverse=True)

        # 显示结果
        lines = ["麦克风探测结果："]
        for rank, (did, name, rms) in enumerate(results, 1):
            bar = vu_bar(rms, 15)
            tag = " 最佳" if rank == 1 else ""
            status = f"✓ {rms:.6f}" if rms > 0 else "打开失败"
            lines.append(f"  #{rank} [{did}] {name}  {bar}  {status}{tag}")
        self.ui.add_system_message("\n".join(lines))

        # 自动选择
        auto_select = "--select" in args or "-s" in args
        if results and results[0][2] > 0.005:
            best_id, best_name, best_rms = results[0]
            if auto_select:
                self._mic_device = best_id
                self.ui.mic.device_name = best_name
                self.ui.add_system_message(f"已选择最佳设备: [{best_id}] {best_name}")
                # 如果正在监测，重启使用新设备
                if self._mic_monitoring and self._mic is not None:
                    self.ui.add_system_message("正在重启麦克风监测使用新设备...")
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
                    self.ui.add_system_message("监测已切换到新设备")
            else:
                self.ui.add_system_message(
                    f"提示: 添加 --select 自动选择 [{best_id}] {best_name}"
                )
        elif results and results[0][2] > 0:
            self.ui.add_system_message("所有设备音量极低，未自动选择")
        else:
            self.ui.add_system_message("未检测到可用的麦克风设备")

        self._app.invalidate()

    # ------------------------------------------------------------------
    # 麦克风监测
    # ------------------------------------------------------------------

    async def _mic_monitor_loop(self) -> None:
        """后台任务：持续采集麦克风并更新 RMS。"""
        if self._mic is None:
            return
        try:
            await self._mic.start()
            while self._mic_monitoring and self.ui.running:
                chunk = await self._mic.read_chunk()
                rms = calculate_rms(chunk)
                self.ui.mic.rms = rms
                # 限频刷新
                now = time.monotonic()
                if now - self._last_invalidate > _AUDIO_LEVEL_THROTTLE:
                    self._last_invalidate = now
                    self._app.invalidate()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.ui.add_system_message(f"麦克风采集异常: {e}")
            self._app.invalidate()
        finally:
            await self._mic.stop()
