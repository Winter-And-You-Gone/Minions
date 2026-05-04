"""状态驱动的动态 CLI 界面 — 使用 prompt_toolkit Application 实现全屏动态刷新。"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.history import FileHistory
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl

from voice_agent.cli.formatters import format_chat, vu_bar
from voice_agent.cli.shell import (
    _get_mic_info,
    _list_devices_str,
    _probe_device_rms,
    _resolve_device,
)
from voice_agent.cli.tui_renderer import (
    format_chat_panel,
    format_input_prompt,
    format_side_panel,
    format_top_bar,
)
from voice_agent.cli.ui_state import GateView, LLMView, UIState
from voice_agent.cli.command_completer import (
    COMMAND_SPECS as _CMD_SPECS,
    MinionsCommandCompleter,
)
from voice_agent.config import save_config
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
        health_report: object | None = None,
        runtime_info: dict | None = None,
        config: dict | None = None,
        config_path: str = "config.yaml",
    ) -> None:
        self._bus = bus
        self._agent = agent
        self._state = state
        self._llm = llm
        self._mic = mic
        self._asr_engine = asr_engine
        self._health_report = health_report
        self._config = config
        self._config_path = config_path
        self._logger = get_logger()

        # 状态
        self.ui = UIState(
            llm=LLMView(
                model=llm.model if llm.is_available and llm.model else "mock",
                available=llm.is_available,
            ),
        )

        # runtime_info 注入
        if runtime_info:
            self.ui.asr_engine = runtime_info.get("asr_engine", "sherpa-onnx")
            self.ui.judge_provider = runtime_info.get("judge_provider", "local")
            self.ui.judge_model = runtime_info.get("judge_model", "qwen3.5:4b")
            self.ui.llm_model = runtime_info.get("llm_model", "")
            self.ui.assistant_name = runtime_info.get("assistant_name", "琉璃川")

        # 健康检查注入
        if health_report:
            self.ui.health_items = health_report.items if hasattr(health_report, "items") else []

        self._mic_device: int | str | None = None
        self._mic_monitoring = False
        self._mic_monitor_task: asyncio.Task | None = None
        self._last_invalidate = 0.0

        # 输入缓冲区
        self._input_buffer = Buffer(
            accept_handler=self._on_accept,
            history=FileHistory(str(_HISTORY_FILE)),
            completer=MinionsCommandCompleter(),
            complete_while_typing=True,
        )

        # 先构建快捷键，再构建 UI（UI 需要 _kb）
        self._build_key_bindings()
        self._build_ui()

        # 欢迎信息
        self.ui.add_system_message("Minions — 常驻语音 Agent")
        self.ui.add_system_message("输入 /help 查看命令，直接输入文字与 AI 对话")

        # ASR 不可用提示
        if asr_engine is None:
            self.ui.add_notification("ASR 模型文件不存在，语音识别未启动，但可以键盘输入")
            self.ui.add_system_message("ASR 模型文件不存在，语音识别未启动，但可以键盘输入")

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # 顶部状态栏
        top_bar = Window(
            FormattedTextControl(lambda: format_top_bar(self.ui)),
            height=1,
        )

        separator_top = Window(height=1, char="─")

        # 左侧聊天区
        self._chat_win = Window(
            FormattedTextControl(lambda: format_chat_panel(self.ui)),
            wrap_lines=True,
            always_hide_cursor=True,
        )

        # 右侧状态面板
        side_panel = Window(
            FormattedTextControl(lambda: format_side_panel(self.ui)),
            width=40,
            wrap_lines=True,
            always_hide_cursor=True,
        )

        body = VSplit([
            self._chat_win,
            Window(width=1, char="│"),
            side_panel,
        ])

        separator_bottom = Window(height=1, char="─")

        # 底部输入行
        input_prompt = Window(
            FormattedTextControl(lambda: format_input_prompt(self.ui)),
            width=18,
            height=1,
            dont_extend_width=True,
        )
        self._input_window = Window(
            BufferControl(buffer=self._input_buffer),
            height=1,
        )
        input_row = VSplit([input_prompt, self._input_window])

        root = HSplit([
            top_bar,
            separator_top,
            body,
            separator_bottom,
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
            ("ansibrightblack", " / 输入命令 · Tab 补全 · Enter 执行 · Ctrl+C 退出 · /help 查看全部命令 "),
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

        @self._kb.add(Keys.Tab)
        def _complete(event: object) -> None:
            buff = self._input_buffer
            if buff.complete_state:
                buff.complete_next()
            else:
                buff.start_completion(select_first=True)

        @self._kb.add(Keys.BackTab)
        def _complete_prev(event: object) -> None:
            buff = self._input_buffer
            if buff.complete_state:
                buff.complete_previous()
            else:
                buff.start_completion(select_first=True)

        @self._kb.add(Keys.Up)
        def _hist_back(event: object) -> None:
            self._input_buffer.history_backward()

        @self._kb.add(Keys.Down)
        def _hist_forward(event: object) -> None:
            self._input_buffer.history_forward()

    # ------------------------------------------------------------------
    # 事件订阅 — 仅更新 UIState，不污染主聊天区
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

        elif etype == "asr.speech_start":
            self.ui.asr.status = "listening"
            self.ui.add_notification("检测到语音开始")

        elif etype == "asr.speech_end":
            dur = event.get("duration_ms", 0)
            self.ui.asr.status = "recognizing"
            self.ui.add_notification(f"语音结束 ({dur}ms)，识别中...")

        elif etype == "asr.final" and event.get("source") != "cli":
            self.ui.asr.status = "recognized"
            self.ui.add_user_message(event.get("text", ""))
            self.ui.add_notification("识别完成")
            self._scroll_to_bottom()

        elif etype == "asr.error":
            self.ui.asr.status = "error"
            msg = event.get("message", "")
            if msg:
                self.ui.add_notification(f"ASR 错误: {msg}")

        elif etype == "asr.status":
            self.ui.asr.status = event.get("status", "idle")
            if event.get("model"):
                self.ui.asr.model = event["model"]

        elif etype == "judge.result":
            self.ui.latest_judge_provider = event.get("provider", "")
            self.ui.latest_judge_target = event.get("target", "")
            self.ui.latest_judge_should_reply = event.get("should_reply", False)
            self.ui.latest_judge_confidence = event.get("confidence", 0.0)
            self.ui.latest_judge_reason = event.get("reason", "")

        elif etype == "system":
            msg = event.get("message", "")
            if msg:
                self.ui.add_notification(msg)

        # 每次事件后刷新唤醒会话状态
        self._refresh_wake_state()
        self._app.invalidate()

    def _refresh_wake_state(self) -> None:
        if hasattr(self._state, "is_wake_session_active") and self._state.is_wake_session_active():
            self.ui.wake_active = True
            self.ui.wake_remaining_seconds = self._state.seconds_until_wake_session_timeout()
        else:
            self.ui.wake_active = False
            self.ui.wake_remaining_seconds = 0.0

    def _save_assistant_config(self) -> None:
        """把当前 wake matcher 的 assistant 配置写回 config.yaml。"""
        import yaml as _yaml
        from pathlib import Path as _Path

        matcher = getattr(self._agent._gate, "wake_matcher", None)
        if matcher is None:
            self.ui.add_system_message("当前未启用唤醒名功能，无法保存配置")
            self._app.invalidate()
            return

        if self._config is None:
            self.ui.add_system_message("当前没有可写配置对象，无法保存")
            self._app.invalidate()
            return

        cfg = matcher.config

        # 重新读取原始 YAML（保留 ${VAR} 占位符），只 patch assistant 段
        path = _Path(self._config_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                raw = _yaml.safe_load(f) or {}
        else:
            raw = {}

        assistant_cfg = raw.setdefault("assistant", {})
        assistant_cfg["name"] = cfg.name
        assistant_cfg["user_title"] = getattr(cfg, "user_title", "少爷")
        assistant_cfg["wake_aliases"] = list(cfg.aliases)

        wake_cfg = assistant_cfg.setdefault("wake", {})
        wake_cfg["enabled"] = getattr(cfg, "enabled", True)
        wake_cfg["session_seconds"] = getattr(cfg, "session_seconds", 120)
        wake_cfg["silence_timeout_seconds"] = getattr(cfg, "silence_timeout_seconds", 90)
        wake_cfg["strip_wake_name"] = getattr(cfg, "strip_wake_name", True)
        wake_cfg["allow_llm_turn_away_judge"] = getattr(cfg, "allow_llm_turn_away_judge", True)

        with open(path, "w", encoding="utf-8") as f:
            _yaml.safe_dump(
                raw,
                f,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            )

        # 更新内存 config 的 assistant 段
        self._config["assistant"] = assistant_cfg

        self.ui.add_notification(f"配置已保存到 {self._config_path}")

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
        "/debug": ("显示最近内部事件", "debug"),
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
            self.ui.add_system_message(f"未知命令: {cmd}。输入 / 后按 Tab 查看可用命令，或输入 /help。")
            self._app.invalidate()
            return

        handler_name = f"_cmd_{info[1]}"
        handler = getattr(self, handler_name, None)
        if handler:
            await handler(*args)

    async def _cmd_help(self) -> None:
        lines = ["可用命令："]
        for spec in _CMD_SPECS:
            alias = f"  别名: {', '.join(spec.aliases)}" if spec.aliases else ""
            usage = f"  用法: {spec.usage}" if spec.usage else ""
            lines.append(f"  {spec.command:<12s} {spec.description}")
            if alias:
                lines.append(f"                {alias}")
            if usage:
                lines.append(f"                {usage}")
        self.ui.add_system_message("\n".join(lines))
        self._app.invalidate()

    async def _cmd_debug(self) -> None:
        g = self.ui.latest_gate
        lines = ["── Debug ──"]
        lines.append(f"Gate: {g.action} score={g.score} reason={g.reason}")
        lines.append(
            f"Judge: provider={self.ui.latest_judge_provider} "
            f"target={self.ui.latest_judge_target} "
            f"reply={self.ui.latest_judge_should_reply} "
            f"conf={self.ui.latest_judge_confidence:.2f}"
        )
        lines.append(f"ASR: status={self.ui.asr.status} model={self.ui.asr.model}")
        lines.append(f"Mode: {self.ui.conversation_mode}")
        lines.append(f"Wake: active={self.ui.wake_active} remain={self.ui.wake_remaining_seconds:.0f}s")
        lines.append(f"Notices: {len(self.ui.notifications)}")
        if self.ui.health_items:
            for item in self.ui.health_items:
                name = getattr(item, "name", "?")
                ok = getattr(item, "ok", False)
                level = getattr(item, "level", "info")
                status = "✓" if ok else ("✗" if level == "error" else "!")
                lines.append(f"  Health {status} {name}")
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

        lines = ["Minions 状态"]
        lines.append(f"  ASR: {self.ui.asr_engine} / {self.ui.asr.status}")
        jm = self.ui.judge_model or "-"
        lines.append(f"  Judge: {jm} / {self.ui.judge_provider}")
        lines.append(f"  Wake: {'active ' + str(int(self.ui.wake_remaining_seconds)) + 's' if self.ui.wake_active else 'inactive'}")
        llm_label = self.ui.llm_model or self._llm.model or "mock"
        lines.append(f"  LLM: {llm_label}")
        lines.append(f"  Logs: logs/minions.log")
        lines.append(f"  麦克风: {mic_tag}  监测: {monitoring}")
        self.ui.add_system_message("\n".join(lines))
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
                    "  /name\n"
                    "  /name set 琉璃川\n"
                    "  /name alias add 六里川\n"
                    "  /name alias remove 六里川\n"
                    "  /name alias list\n"
                    "  /name save"
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
            self.ui.assistant_name = new_name
            if new_name not in cfg.aliases:
                cfg.aliases.insert(0, new_name)
            self._save_assistant_config()
            self.ui.add_system_message(f"AI 名字已设置并保存为：{new_name}")
            self._app.invalidate()
            return

        if args[0] == "save":
            self._save_assistant_config()
            self.ui.add_system_message("AI 名字和唤醒别名配置已保存")
            self._app.invalidate()
            return

        if args[0] == "alias" and len(args) >= 2:
            action = args[1]

            if action == "add" and len(args) >= 3:
                alias = args[2]
                if alias not in cfg.aliases:
                    cfg.aliases.append(alias)
                self._save_assistant_config()
                self.ui.add_system_message(f"已添加并保存唤醒别名：{alias}")
                self._app.invalidate()
                return

            if action == "remove" and len(args) >= 3:
                alias = args[2]
                cfg.aliases = [x for x in cfg.aliases if x != alias]
                self._save_assistant_config()
                self.ui.add_system_message(f"已移除并保存唤醒别名：{alias}")
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
            "  /name set 琉璃川\n"
            "  /name alias add 六里川\n"
            "  /name alias remove 六里川\n"
            "  /name alias list\n"
            "  /name save"
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

            if self._config is not None:
                # 重新读取原始 YAML，只 patch audio.device
                import yaml as _yaml
                from pathlib import Path as _Path

                path = _Path(self._config_path)
                if path.exists():
                    with open(path, "r", encoding="utf-8") as f:
                        raw = _yaml.safe_load(f) or {}
                else:
                    raw = {}

                audio_cfg = raw.setdefault("audio", {})
                audio_cfg["device"] = self._mic_device

                with open(path, "w", encoding="utf-8") as f:
                    _yaml.safe_dump(
                        raw, f,
                        allow_unicode=True, sort_keys=False, default_flow_style=False,
                    )

                # 更新内存 config
                self._config["audio"] = audio_cfg

                self.ui.add_system_message(f"已选择并保存麦克风设备: {self._mic_device}")
            else:
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
