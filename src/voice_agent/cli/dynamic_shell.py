"""状态驱动的动态 CLI 界面 — 使用 prompt_toolkit Application 实现全屏动态刷新。"""

from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path

from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.history import FileHistory
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import (
    ConditionalContainer,
    HSplit,
    Layout,
    VSplit,
    Window,
)
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl

from voice_agent.cli.shell import (
    _get_mic_info,
    _list_devices_str,
    _probe_device_rms,
    _resolve_device,
)
from voice_agent.cli.tui_renderer import (
    format_command_panel,
    format_footer_bar,
    format_home_panel,
    format_input_prompt,
)
from voice_agent.cli.ui_state import CompletionItem, GateView, LLMView, UIState
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

_AUDIO_LEVEL_THROTTLE = 0.1


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
        runtime_controller: object | None = None,
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
        self._runtime_controller = runtime_controller
        self._config = config
        self._config_path = config_path
        self._logger = get_logger()

        # 退出标记
        self._exit_requested = False

        # 状态
        self.ui = UIState(
            llm=LLMView(
                model=llm.model if llm.is_available and llm.model else "mock",
                available=llm.is_available,
            ),
        )

        # 默认运行时状态为 sleeping（ASR 不会自动启动）
        self.ui.runtime_state = "sleeping"

        # runtime_info 注入
        if runtime_info:
            self.ui.asr_engine = runtime_info.get("asr_engine", "sherpa-onnx")
            self.ui.judge_provider = runtime_info.get("judge_provider", "local")
            self.ui.judge_model = runtime_info.get("judge_model", "qwen3.5:4b")
            self.ui.llm_model = runtime_info.get("llm_model", "")
            self.ui.assistant_name = runtime_info.get("assistant_name", "琉璃川")
            self._completion_enabled = runtime_info.get("completion_enabled", True)
        else:
            self._completion_enabled = True

        # 健康检查注入
        if health_report:
            self.ui.health_items = health_report.items if hasattr(health_report, "items") else []

        self._mic_device: int | str | None = None
        self._mic_monitoring = False
        self._mic_monitor_task: asyncio.Task | None = None
        self._last_invalidate = 0.0

        # 输入历史浏览状态
        self._history_browsing = False
        self._history_index = -1
        self._history_draft = ""

        # 输入缓冲区 — 不使用内置 completer，走自定义补全面板
        self._input_buffer = Buffer(
            accept_handler=self._on_accept,
            history=FileHistory(str(_HISTORY_FILE)),
        )
        # 监听文本变化以刷新补全面板
        self._input_buffer.on_text_changed += self._on_input_changed

        # 先构建快捷键，再构建 UI（UI 需要 _kb）
        self._build_key_bindings()
        self._build_ui()

        # 欢迎信息
        self.ui.add_system_message("Minions — 常驻语音 Agent")
        self.ui.add_system_message("输入 /help 查看命令，直接输入文字与 AI 对话")

        # 默认文字模式已可用，语音监听未启动
        self.ui.add_notification("文字聊天已可用。输入 /listen 启动语音，/wakeup 叫醒 AI")

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # 主面板：LOGO + 欢迎 + 聊天消息 + 提示
        self._home_win = Window(
            FormattedTextControl(lambda: format_home_panel(self.ui)),
            wrap_lines=False,
            always_hide_cursor=True,
        )

        sep_top = Window(height=1, char="─")

        # 输入行
        input_prompt = Window(
            FormattedTextControl(lambda: format_input_prompt(self.ui)),
            width=4,
            height=1,
            dont_extend_width=True,
        )
        self._input_window = Window(
            BufferControl(buffer=self._input_buffer),
            height=1,
        )
        input_row = VSplit([input_prompt, self._input_window])

        # 条件：仅在非 blank 模式显示中间分隔线和命令面板
        show_command_panel = Condition(
            lambda: self.ui.command_panel_mode != "blank"
        )

        command_sep = ConditionalContainer(
            content=Window(height=1, char="─"),
            filter=show_command_panel,
        )

        # 命令面板（固定 14 行，help/completion/output 三种模式）
        self._command_panel_win = Window(
            FormattedTextControl(lambda: format_command_panel(self.ui)),
            height=self.ui.command_panel_reserved_rows,
            wrap_lines=False,
            always_hide_cursor=True,
        )
        command_panel = ConditionalContainer(
            content=self._command_panel_win,
            filter=show_command_panel,
        )

        sep_bot = Window(height=1, char="─")

        # 底部状态栏
        footer = Window(
            FormattedTextControl(lambda: format_footer_bar(self.ui)),
            height=1,
        )

        root = HSplit([
            self._home_win,
            sep_top,
            input_row,
            command_sep,
            command_panel,
            sep_bot,
            footer,
        ])

        self._layout = Layout(root)
        self._layout.focus(self._input_window)

        self._app = Application(
            layout=self._layout,
            full_screen=True,
            key_bindings=self._kb,
            mouse_support=False,
        )

    def _build_key_bindings(self) -> None:
        from prompt_toolkit.key_binding import KeyBindings

        self._kb = KeyBindings()

        # ── 保底退出：同步、无 await ──
        @self._kb.add("c-c", eager=True)
        def _exit_ctrl_c(event: object) -> None:
            self._force_exit(event)

        @self._kb.add("c-d", eager=True)
        def _exit_ctrl_d(event: object) -> None:
            self._force_exit(event)

        @self._kb.add("c-q", eager=True)
        def _exit_ctrl_q(event: object) -> None:
            self._force_exit(event)

        # ── Enter：help 填入 / completion 直接执行 / 提交输入 ──
        @self._kb.add(Keys.Enter)
        def _accept(event: object) -> None:
            try:
                # help 模式：把选中命令填入输入框
                if self.ui.command_panel_mode == "help" and self.ui.help_items:
                    idx = self.ui.command_panel_selected_index
                    if 0 <= idx < len(self.ui.help_items):
                        cmd = self.ui.help_items[idx].get("command", "")
                        self._input_buffer.text = cmd + " "
                        self._input_buffer.cursor_position = len(cmd) + 1
                        self._reset_command_panel_scroll()
                        self.ui.command_panel_mode = "completion"
                        self._refresh_completion_state()
                        self._safe_invalidate()
                        return

                # completion 模式：直接执行选中命令
                if self.ui.command_panel_mode == "completion" and self.ui.completion_items:
                    idx = self.ui.completion_selected_index
                    if 0 <= idx < len(self.ui.completion_items):
                        item = self.ui.completion_items[idx]
                        cmd = (item.execute_text or item.text).strip()
                        self._input_buffer.text = ""
                        self.ui.completion_visible = False
                        self.ui.completion_items = []
                        self.ui.command_panel_mode = "blank"
                        self._reset_command_panel_scroll()
                        self._create_task(self._handle_input(cmd), "completion_execute")
                        self._safe_invalidate()
                        return

                self._accept_buffer()
            except Exception as e:
                self._record_ui_error("enter", e)

        # ── Tab：导航 help / completion（clamp + scroll） ──
        @self._kb.add(Keys.Tab)
        def _complete_next(event: object) -> None:
            try:
                if self.ui.command_panel_mode == "help" and self.ui.help_items:
                    last = len(self.ui.help_items) - 1
                    if self.ui.command_panel_selected_index < last:
                        self.ui.command_panel_selected_index += 1
                        self._ensure_panel_index_visible(
                            self.ui.command_panel_selected_index, len(self.ui.help_items),
                        )
                    self._safe_invalidate()
                    return
                if self.ui.command_panel_mode == "completion" and self.ui.completion_items:
                    last = len(self.ui.completion_items) - 1
                    if self.ui.completion_selected_index < last:
                        self.ui.completion_selected_index += 1
                        self._ensure_panel_index_visible(
                            self.ui.completion_selected_index, len(self.ui.completion_items),
                        )
                    self._safe_invalidate()
                    return
                self._refresh_completion_state()
                if self.ui.completion_items:
                    self.ui.completion_selected_index = 0
                    self.ui.command_panel_mode = "completion"
                    self._reset_command_panel_scroll()
                    self._safe_invalidate()
            except Exception as e:
                self._record_ui_error("tab", e)

        @self._kb.add(Keys.BackTab)
        def _complete_prev(event: object) -> None:
            try:
                if self.ui.command_panel_mode == "help" and self.ui.help_items:
                    if self.ui.command_panel_selected_index > 0:
                        self.ui.command_panel_selected_index -= 1
                        self._ensure_panel_index_visible(
                            self.ui.command_panel_selected_index, len(self.ui.help_items),
                        )
                    self._safe_invalidate()
                    return
                if self.ui.command_panel_mode == "completion" and self.ui.completion_items:
                    if self.ui.completion_selected_index > 0:
                        self.ui.completion_selected_index -= 1
                        self._ensure_panel_index_visible(
                            self.ui.completion_selected_index, len(self.ui.completion_items),
                        )
                    self._safe_invalidate()
                    return
                self._refresh_completion_state()
                if self.ui.completion_items:
                    self.ui.completion_selected_index = 0
                    self.ui.command_panel_mode = "completion"
                    self._reset_command_panel_scroll()
                    self._safe_invalidate()
            except Exception as e:
                self._record_ui_error("backtab", e)

        # ── Escape：关闭面板 ──
        @self._kb.add(Keys.Escape)
        def _escape(event: object) -> None:
            try:
                self.ui.clear_command_panel()
                self._safe_invalidate()
            except Exception as e:
                self._record_ui_error("escape", e)

        # ── Up：选 help/completion / 滚 output / 历史 ──
        @self._kb.add("up", eager=True)
        def _up(event: object) -> None:
            try:
                if self.ui.command_panel_mode == "completion" and self.ui.completion_items:
                    if self.ui.completion_selected_index > 0:
                        self.ui.completion_selected_index -= 1
                        self._ensure_panel_index_visible(
                            self.ui.completion_selected_index, len(self.ui.completion_items),
                        )
                    self._safe_invalidate()
                    return

                if self.ui.command_panel_mode == "help" and self.ui.help_items:
                    if self.ui.command_panel_selected_index > 0:
                        self.ui.command_panel_selected_index -= 1
                        self._ensure_panel_index_visible(
                            self.ui.command_panel_selected_index, len(self.ui.help_items),
                        )
                    self._safe_invalidate()
                    return

                if self.ui.command_panel_mode == "output" and self.ui.command_output_lines:
                    self._scroll_output_lines(-1)
                    self._safe_invalidate()
                    return

                self._history_back()
            except Exception as e:
                self._record_ui_error("up", e)

        # ── Down：选 help/completion / 滚 output / 历史 ──
        @self._kb.add("down", eager=True)
        def _down(event: object) -> None:
            try:
                if self.ui.command_panel_mode == "completion" and self.ui.completion_items:
                    last = len(self.ui.completion_items) - 1
                    if self.ui.completion_selected_index < last:
                        self.ui.completion_selected_index += 1
                        self._ensure_panel_index_visible(
                            self.ui.completion_selected_index, len(self.ui.completion_items),
                        )
                    self._safe_invalidate()
                    return

                if self.ui.command_panel_mode == "help" and self.ui.help_items:
                    last = len(self.ui.help_items) - 1
                    if self.ui.command_panel_selected_index < last:
                        self.ui.command_panel_selected_index += 1
                        self._ensure_panel_index_visible(
                            self.ui.command_panel_selected_index, len(self.ui.help_items),
                        )
                    self._safe_invalidate()
                    return

                if self.ui.command_panel_mode == "output" and self.ui.command_output_lines:
                    self._scroll_output_lines(1)
                    self._safe_invalidate()
                    return

                self._history_forward()
            except Exception as e:
                self._record_ui_error("down", e)

        # ── PageUp ──
        @self._kb.add("pageup", eager=True)
        def _page_up(event: object) -> None:
            try:
                mode = self.ui.command_panel_mode
                if mode in ("help", "completion"):
                    vis = self._get_visible_items_for_command_panel()
                else:
                    vis = self._get_visible_rows_for_command_panel()

                if mode == "help" and self.ui.help_items:
                    self.ui.command_panel_selected_index = max(0, self.ui.command_panel_selected_index - vis)
                    self._ensure_panel_index_visible(
                        self.ui.command_panel_selected_index, len(self.ui.help_items),
                    )
                    self._safe_invalidate()
                    return
                if mode == "completion" and self.ui.completion_items:
                    self.ui.completion_selected_index = max(0, self.ui.completion_selected_index - vis)
                    self._ensure_panel_index_visible(
                        self.ui.completion_selected_index, len(self.ui.completion_items),
                    )
                    self._safe_invalidate()
                    return
                if mode == "output" and self.ui.command_output_lines:
                    self._scroll_output_lines(-vis)
                    self._safe_invalidate()
                    return
            except Exception as e:
                self._record_ui_error("pageup", e)

        # ── PageDown ──
        @self._kb.add("pagedown", eager=True)
        def _page_down(event: object) -> None:
            try:
                mode = self.ui.command_panel_mode
                if mode in ("help", "completion"):
                    vis = self._get_visible_items_for_command_panel()
                else:
                    vis = self._get_visible_rows_for_command_panel()

                if mode == "help" and self.ui.help_items:
                    last = len(self.ui.help_items) - 1
                    self.ui.command_panel_selected_index = min(last, self.ui.command_panel_selected_index + vis)
                    self._ensure_panel_index_visible(
                        self.ui.command_panel_selected_index, len(self.ui.help_items),
                    )
                    self._safe_invalidate()
                    return
                if mode == "completion" and self.ui.completion_items:
                    last = len(self.ui.completion_items) - 1
                    self.ui.completion_selected_index = min(last, self.ui.completion_selected_index + vis)
                    self._ensure_panel_index_visible(
                        self.ui.completion_selected_index, len(self.ui.completion_items),
                    )
                    self._safe_invalidate()
                    return
                if mode == "output" and self.ui.command_output_lines:
                    self._scroll_output_lines(vis)
                    self._safe_invalidate()
                    return
            except Exception as e:
                self._record_ui_error("pagedown", e)

        # ── Home：跳转到第一个 ──
        @self._kb.add("home", eager=True)
        def _home_key(event: object) -> None:
            try:
                if self.ui.command_panel_mode == "help" and self.ui.help_items:
                    self.ui.command_panel_selected_index = 0
                    self._reset_command_panel_scroll()
                    self._safe_invalidate()
                    return
                if self.ui.command_panel_mode == "completion" and self.ui.completion_items:
                    self.ui.completion_selected_index = 0
                    self._reset_command_panel_scroll()
                    self._safe_invalidate()
                    return
                if self.ui.command_panel_mode == "output" and self.ui.command_output_lines:
                    self._reset_command_panel_scroll()
                    self._safe_invalidate()
                    return
            except Exception as e:
                self._record_ui_error("home", e)

        # ── End：跳转到最后一个 ──
        @self._kb.add("end", eager=True)
        def _end_key(event: object) -> None:
            try:
                if self.ui.command_panel_mode == "help" and self.ui.help_items:
                    last = len(self.ui.help_items) - 1
                    self.ui.command_panel_selected_index = last
                    self._ensure_panel_index_visible(last, len(self.ui.help_items))
                    self._safe_invalidate()
                    return
                if self.ui.command_panel_mode == "completion" and self.ui.completion_items:
                    last = len(self.ui.completion_items) - 1
                    self.ui.completion_selected_index = last
                    self._ensure_panel_index_visible(last, len(self.ui.completion_items))
                    self._safe_invalidate()
                    return
                if self.ui.command_panel_mode == "output" and self.ui.command_output_lines:
                    vis = self._get_visible_rows_for_command_panel()
                    total = len(self.ui.command_output_lines)
                    self.ui.command_panel_scroll_offset = max(0, total - vis)
                    self._safe_invalidate()
                    return
            except Exception as e:
                self._record_ui_error("end", e)

    # ------------------------------------------------------------------
    # 事件订阅 — 仅更新 UIState，不污染主聊天区
    # ------------------------------------------------------------------

    async def on_event(self, event: dict) -> None:
        """EventBus 回调 — 外层兜底，不抛异常到事件循环。"""
        try:
            await self._on_event_inner(event)
        except Exception as e:
            self._record_ui_error("on_event", e)

    async def _on_event_inner(self, event: dict) -> None:
        etype = event.get("type", "")

        if etype == "audio.level":
            self.ui.mic.rms = event.get("rms", 0.0)
            now = time.monotonic()
            if now - self._last_invalidate > _AUDIO_LEVEL_THROTTLE:
                self._last_invalidate = now
                self._safe_invalidate()
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

        elif etype == "runtime.status":
            new_state = event.get("state", "text_ready")
            self.ui.runtime_state = new_state
            self.ui.voice_listening = (new_state == "listening")
            msg = event.get("message", "")
            if msg:
                self.ui.add_notification(msg)

        # 每次事件后刷新唤醒会话状态
        self._refresh_wake_state()
        self._safe_invalidate()

    def _refresh_wake_state(self) -> None:
        try:
            if hasattr(self._state, "is_wake_session_active") and self._state.is_wake_session_active():
                self.ui.wake_active = True
                self.ui.wake_remaining_seconds = self._state.seconds_until_wake_session_timeout()
            else:
                self.ui.wake_active = False
                self.ui.wake_remaining_seconds = 0.0
        except Exception:
            pass

    def _save_assistant_config(self) -> None:
        """把当前 wake matcher 的 assistant 配置写回 config.yaml。"""
        import yaml as _yaml
        from pathlib import Path as _Path

        matcher = getattr(self._agent._gate, "wake_matcher", None)
        if matcher is None:
            self.ui.set_command_output("Config", "当前未启用唤醒名功能，无法保存配置")
            self._safe_invalidate()
            return

        if self._config is None:
            self.ui.set_command_output("Config", "当前没有可写配置对象，无法保存")
            self._safe_invalidate()
            return

        cfg = matcher.config

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

        self._config["assistant"] = assistant_cfg
        self.ui.add_notification(f"配置已保存到 {self._config_path}")

    def subscribe(self) -> None:
        self._bus.subscribe(self.on_event)

    def unsubscribe(self) -> None:
        self._bus.unsubscribe(self.on_event)

    # ------------------------------------------------------------------
    # 输入变化 → 补全
    # ------------------------------------------------------------------

    def _completion_to_plain_text(self, value: object) -> str:
        """把 prompt_toolkit 的 formatted text 转成纯文本。"""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            from prompt_toolkit.formatted_text import to_plain_text
            return to_plain_text(value)
        except Exception:
            return str(value)

    def _on_input_changed(self, buf: Buffer) -> None:
        """输入框文本变化时刷新补全面板。"""
        try:
            self._refresh_completion_state()
            self._safe_invalidate()
        except Exception as e:
            self._record_ui_error("input_changed", e)

    def _refresh_completion_state(self) -> None:
        """调用补全器，将结果写入 UIState 并控制命令面板模式。"""
        if not self._completion_enabled:
            self.ui.completion_visible = False
            self.ui.completion_items = []
            if self.ui.command_panel_mode == "completion":
                self.ui.command_panel_mode = "blank"
            return

        # help 模式下不覆盖
        if self.ui.command_panel_mode == "help":
            return

        text = self._input_buffer.text
        if not text.startswith("/"):
            self.ui.completion_visible = False
            self.ui.completion_items = []
            if self.ui.command_panel_mode == "completion":
                self.ui.command_panel_mode = "blank"
            return

        try:
            completer = MinionsCommandCompleter()
            doc = Document(text=text, cursor_position=len(text))
            completions = list(completer.get_completions(doc, None))
        except Exception as e:
            self._logger.exception("[TUI] completions failed: %s", e)
            self.ui.completion_visible = False
            self.ui.completion_items = []
            return

        if not completions:
            self.ui.completion_visible = False
            self.ui.completion_items = []
            self.ui.command_panel_mode = "blank"
            return

        items: list[CompletionItem] = []
        for c in completions:
            try:
                display = self._completion_to_plain_text(getattr(c, "display", None))
                display_meta = self._completion_to_plain_text(getattr(c, "display_meta", None))
                display = getattr(c, "display_text", None) or display
                display_meta = getattr(c, "display_meta_text", None) or display_meta
                display = str(display or c.text)
                display_meta = str(display_meta or "")

                # 计算 execute_text：用 start_position 从原始文本中计算出实际应该执行的完整命令
                try:
                    start_pos = int(getattr(c, "start_position", 0))
                    cut_pos = len(text) + start_pos
                    if cut_pos < 0:
                        cut_pos = 0
                    applied = text[:cut_pos] + str(c.text)
                except Exception:
                    applied = str(c.text)
                execute_text = applied.strip()

                items.append(CompletionItem(
                    text=str(c.text),
                    display=display,
                    display_meta=display_meta,
                    execute_text=execute_text,
                ))
            except Exception as e:
                self._logger.exception("[TUI] skip bad completion: %s", e)

        if not items:
            self.ui.completion_visible = False
            self.ui.completion_items = []
            self.ui.command_panel_mode = "blank"
            return

        self.ui.completion_items = items
        self.ui.completion_selected_index = 0
        self.ui.completion_visible = True
        self.ui.command_panel_mode = "completion"

    def _scroll_to_bottom(self) -> None:
        """滚动主面板到底部。"""
        try:
            self._home_win.vertical_scroll = 999999
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 命令面板滚动辅助
    # ------------------------------------------------------------------

    def _reset_command_panel_scroll(self) -> None:
        self.ui.command_panel_scroll_offset = 0

    def _get_visible_rows_for_command_panel(self) -> int:
        mode = self.ui.command_panel_mode
        rows = self.ui.command_panel_reserved_rows

        if mode == "output":
            return max(1, rows - 3)

        return max(1, rows)

    def _get_visible_items_for_command_panel(self) -> int:
        """返回命令面板中可见的 item 数（而非行数）。

        help 每个 item 占 2 行（命令行 + 描述行），
        completion 每 item 占 1 行，
        output 滚动的是行而非 item。
        """
        rows = self.ui.command_panel_reserved_rows
        mode = self.ui.command_panel_mode

        if mode == "help":
            usable_lines = max(1, rows - 4)  # header 行 + footer 行占用 4
            return max(1, usable_lines // 2)

        if mode == "completion":
            return max(1, rows)

        if mode == "output":
            return max(1, rows - 3)

        return max(1, rows)

    def _ensure_panel_index_visible(self, selected_index: int, total_count: int) -> None:
        visible_items = self._get_visible_items_for_command_panel()
        offset = self.ui.command_panel_scroll_offset

        max_offset = max(0, total_count - visible_items)

        if selected_index < offset:
            offset = selected_index
        elif selected_index >= offset + visible_items:
            offset = selected_index - visible_items + 1

        self.ui.command_panel_scroll_offset = max(0, min(offset, max_offset))

    def _scroll_output_lines(self, delta: int) -> None:
        visible_rows = self._get_visible_rows_for_command_panel()
        total = len(self.ui.command_output_lines)
        max_offset = max(0, total - visible_rows)
        new_offset = self.ui.command_panel_scroll_offset + delta
        self.ui.command_panel_scroll_offset = max(0, min(new_offset, max_offset))

    # ------------------------------------------------------------------
    # 输入历史浏览
    # ------------------------------------------------------------------

    def _get_history_entries(self) -> list[str]:
        try:
            return list(self._input_buffer.history.get_strings())
        except Exception:
            return []

    def _history_back(self) -> None:
        entries = self._get_history_entries()
        if not entries:
            return

        if not self._history_browsing:
            self._history_browsing = True
            self._history_draft = self._input_buffer.text
            self._history_index = len(entries) - 1
        else:
            if self._history_index > 0:
                self._history_index -= 1

        self._input_buffer.text = entries[self._history_index]
        self._input_buffer.cursor_position = len(self._input_buffer.text)

    def _history_forward(self) -> None:
        # 如果没有在浏览历史，但输入框里有内容，Down 应该清空输入框
        if not self._history_browsing:
            if self._input_buffer.text:
                self._input_buffer.text = ""
                self._input_buffer.cursor_position = 0
                self._history_draft = ""
                self._safe_invalidate()
            return

        entries = self._get_history_entries()
        if self._history_index < len(entries) - 1:
            self._history_index += 1
            self._input_buffer.text = entries[self._history_index]
        else:
            # 回到原始草稿；如果原始草稿为空，就显示空输入
            self._history_browsing = False
            self._history_index = -1
            self._input_buffer.text = self._history_draft or ""

        self._input_buffer.cursor_position = len(self._input_buffer.text)

    # ------------------------------------------------------------------
    # 异常 / 错误处理
    # ------------------------------------------------------------------

    def _record_ui_error(self, where: str, exc: BaseException) -> None:
        """记录 UI 错误到日志和状态栏。"""
        msg = f"{where}: {exc}"
        self._logger.exception("[TUI] %s", msg)
        self.ui.error_line = msg
        self.ui.add_notification(f"UI 错误: {where}")
        try:
            self._safe_invalidate()
        except Exception:
            pass

    def _safe_invalidate(self) -> None:
        """安全刷新界面，不抛异常。"""
        try:
            if hasattr(self, "_app"):
                self._app.invalidate()
        except Exception as e:
            self._logger.exception("[TUI] invalidate failed: %s", e)

    def _create_task(self, coro, name: str = "task") -> asyncio.Task:
        """创建异步任务，自动捕获异常显示在 UI 中。"""
        task = asyncio.create_task(coro)
        task.add_done_callback(lambda t: self._on_task_done(t, name))
        return task

    def _on_task_done(self, task: asyncio.Task, name: str) -> None:
        """异步任务完成回调 — 捕获异常。"""
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as e:
            self._record_ui_error(name, e)

    def _force_exit(self, event: object | None = None) -> None:
        """保底退出：不依赖异步命令，立即让 prompt_toolkit 退出。"""
        self._exit_requested = True
        self.ui.running = False
        try:
            self._logger.warning("[TUI] force exit requested")
        except Exception:
            pass
        try:
            if event is not None and hasattr(event, "app"):
                event.app.exit(result=None)
                return
        except Exception:
            pass
        try:
            if hasattr(self, "_app"):
                self._app.exit(result=None)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 启动 / 停止
    # ------------------------------------------------------------------

    async def run(self) -> None:
        try:
            try:
                await self._app.run_async()
            except KeyboardInterrupt:
                self._force_exit()
        finally:
            self._exit_requested = True
            # 通过 RuntimeController 关闭 ASR（如果有）
            if self._runtime_controller is not None:
                await self._runtime_controller.close()
            elif self._asr_engine is not None:
                with contextlib.suppress(Exception):
                    await self._asr_engine.stop()
            if self._mic_monitor_task:
                self._mic_monitor_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await self._mic_monitor_task
                self._mic_monitor_task = None
            self._mic_monitoring = False
            if self._mic is not None:
                with contextlib.suppress(Exception):
                    await self._mic.stop()

    # ------------------------------------------------------------------
    # 输入处理
    # ------------------------------------------------------------------

    def _accept_buffer(self) -> None:
        """提交当前输入框内容，安全包装。"""
        try:
            text = self._input_buffer.text.strip()
            self._input_buffer.text = ""

            # 只清空 completion，不提前把 command_panel_mode 设为 blank
            self.ui.completion_visible = False
            self.ui.completion_items = []

            if text:
                self._create_task(self._handle_input(text), "handle_input")
            else:
                self.ui.clear_command_panel()

            self._safe_invalidate()
        except Exception as e:
            self._record_ui_error("accept_buffer", e)

    def _on_accept(self, buff: Buffer) -> bool:
        self._accept_buffer()
        return True

    async def _handle_input(self, text: str) -> None:
        if text.startswith("/"):
            await self._dispatch_command(text)
        else:
            self.ui.clear_command_panel()
            self.ui.add_user_message(text)
            self._scroll_to_bottom()
            self._safe_invalidate()

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
        "/wakeup": ("叫醒 AI，进入连续对话状态", "wakeup"),
        "/wake": ("叫醒 AI，进入连续对话状态", "wakeup"),
        "/起床": ("叫醒 AI，进入连续对话状态", "wakeup"),
        "/叫醒": ("叫醒 AI，进入连续对话状态", "wakeup"),
        "/sleep": ("让 AI 睡眠，并停止实时监听", "sleep"),
        "/standby": ("让 AI 睡眠，并停止实时监听", "sleep"),
        "/睡觉": ("让 AI 睡眠，并停止实时监听", "sleep"),
        "/休息": ("让 AI 睡眠，并停止实时监听", "sleep"),
        "/judge": ("查看或切换判断器", "judge"),
        "/listen": ("开启实时麦克风倾听 / ASR", "listen"),
        "/倾听": ("开启实时麦克风倾听 / ASR", "listen"),
        "/监听": ("开启实时麦克风倾听 / ASR", "listen"),
    }

    async def _dispatch_command(self, raw: str) -> None:
        try:
            parts = raw.strip().lower().split()
            cmd = parts[0]
            args = parts[1:]

            info = self.COMMANDS.get(cmd)
            if info is None:
                self.ui.set_command_output(
                    "Unknown command",
                    f"未知命令: {cmd}\n输入 / 后按 Tab 查看可用命令，或输入 /help。"
                )
                self._safe_invalidate()
                return

            handler_name = f"_cmd_{info[1]}"
            handler = getattr(self, handler_name, None)
            if handler:
                await handler(*args)
        except Exception as e:
            self._record_ui_error("dispatch_command", e)

    async def _cmd_help(self) -> None:
        self.ui.command_panel_mode = "help"
        self.ui.command_panel_title = "Browse default commands"
        self.ui.help_tab = "commands"
        self.ui.help_items = [
            {
                "command": spec.command,
                "description": spec.description,
                "usage": spec.usage,
                "aliases": list(spec.aliases),
            }
            for spec in _CMD_SPECS
        ]
        self.ui.command_panel_selected_index = 0
        self.ui.completion_visible = False
        self.ui.completion_items = []
        self._safe_invalidate()

    async def _cmd_debug(self) -> None:
        g = self.ui.latest_gate
        lines = []
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
                tag = "✓" if ok else ("✗" if level == "error" else "!")
                lines.append(f"  Health {tag} {name}")
        self.ui.set_command_output("Debug", lines)
        self._safe_invalidate()

    async def _cmd_exit(self) -> None:
        self._force_exit()

    async def _cmd_pause(self) -> None:
        self.ui.paused = True
        await self._bus.publish({"type": "command.pause"})
        self._safe_invalidate()

    async def _cmd_resume(self) -> None:
        self.ui.paused = False
        await self._bus.publish({"type": "command.resume"})
        self._safe_invalidate()

    async def _cmd_clear(self) -> None:
        self.ui.messages.clear()
        self.ui.clear_command_panel()
        self._safe_invalidate()

    async def _cmd_mode(self) -> None:
        self.ui.set_command_output(
            "Conversation Mode",
            [
                f"mode:          {self._state.mode}",
                f"active_until:  {self._state.active_until:.1f}",
                f"cooldown_until: {self._state.cooldown_until:.1f}",
            ],
        )
        self._safe_invalidate()

    async def _cmd_status(self) -> None:
        try:
            mic = _get_mic_info(self._mic_device)
            mic_tag = f"{mic['name']}" if mic["valid"] else f"{mic['name']}（无输入通道）"
            monitoring = "监测中" if self._mic_monitoring else "已停止"

            lines = []
            lines.append(f"ASR:   {self.ui.asr_engine} / {self.ui.asr.status}")
            jm = self.ui.judge_model or "-"
            lines.append(f"Judge: {jm} / {self.ui.judge_provider}")
            lines.append(f"Wake:  {'active ' + str(int(self.ui.wake_remaining_seconds)) + 's' if self.ui.wake_active else 'inactive'}")
            llm_label = self.ui.llm_model or self._llm.model or "mock"
            lines.append(f"LLM:   {llm_label}")
            lines.append(f"Logs:  logs/minions.log")
            lines.append(f"Mic:   {mic_tag}  监测: {monitoring}")
            self.ui.set_command_output("Status", lines)
        except Exception as e:
            self._record_ui_error("cmd_status", e)
        self._safe_invalidate()

    async def _cmd_model(self) -> None:
        model = self._llm.model if self._llm.model else "mock"
        self.ui.set_command_output("Model", f"当前连接的模型是：{model}")
        self._safe_invalidate()

    async def _cmd_name(self, *args: str) -> None:
        try:
            matcher = getattr(self._agent._gate, "wake_matcher", None)

            if not args or not matcher:
                if matcher is None:
                    self.ui.set_command_output("Assistant Name", "当前未启用唤醒名功能")
                else:
                    cfg = matcher.config
                    self.ui.set_command_output(
                        "Assistant Name",
                        [
                            f"当前名字: {cfg.name}",
                            f"唤醒别名: {', '.join(cfg.aliases)}",
                            "",
                            "用法:",
                            "  /name",
                            "  /name set 琉璃川",
                            "  /name alias add 六里川",
                            "  /name alias remove 六里川",
                            "  /name alias list",
                            "  /name save",
                        ],
                    )
                self._safe_invalidate()
                return

            if matcher is None:
                self.ui.set_command_output("Assistant Name", "当前未启用唤醒名功能")
                self._safe_invalidate()
                return

            cfg = matcher.config

            if args[0] == "set" and len(args) >= 2:
                new_name = args[1]
                cfg.name = new_name
                self.ui.assistant_name = new_name
                if new_name not in cfg.aliases:
                    cfg.aliases.insert(0, new_name)
                self._save_assistant_config()
                self.ui.set_command_output("Assistant Name", f"AI 名字已设置并保存为：{new_name}")
                self._safe_invalidate()
                return

            if args[0] == "save":
                self._save_assistant_config()
                self.ui.set_command_output("Assistant Name", "AI 名字和唤醒别名配置已保存")
                self._safe_invalidate()
                return

            if args[0] == "alias" and len(args) >= 2:
                action = args[1]

                if action == "add" and len(args) >= 3:
                    alias = args[2]
                    if alias not in cfg.aliases:
                        cfg.aliases.append(alias)
                    self._save_assistant_config()
                    self.ui.set_command_output("Assistant Name", f"已添加并保存唤醒别名：{alias}")
                    self._safe_invalidate()
                    return

                if action == "remove" and len(args) >= 3:
                    alias = args[2]
                    cfg.aliases = [x for x in cfg.aliases if x != alias]
                    self._save_assistant_config()
                    self.ui.set_command_output("Assistant Name", f"已移除并保存唤醒别名：{alias}")
                    self._safe_invalidate()
                    return

                if action == "list":
                    self.ui.set_command_output(
                        "Assistant Name",
                        [
                            f"当前名字: {cfg.name}",
                            f"唤醒别名: {', '.join(cfg.aliases)}",
                        ],
                    )
                    self._safe_invalidate()
                    return

            self.ui.set_command_output(
                "Assistant Name",
                [
                    "用法:",
                    "  /name",
                    "  /name set 琉璃川",
                    "  /name alias add 六里川",
                    "  /name alias remove 六里川",
                    "  /name alias list",
                    "  /name save",
                ],
            )
        except Exception as e:
            self._record_ui_error("cmd_name", e)
        self._safe_invalidate()

    async def _cmd_mic(self, *args: str) -> None:
        try:
            sub = args[0] if args else "help"

            if sub == "list":
                self.ui.set_command_output("Microphones", _list_devices_str())
                self._safe_invalidate()

            elif sub == "select" and len(args) >= 2:
                self._mic_device = _resolve_device(args[1])
                self.ui.mic.device_name = str(self._mic_device)

                msg = f"已选择麦克风设备: {self._mic_device}"
                if self._config is not None:
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

                    self._config["audio"] = audio_cfg
                    msg = f"已选择并保存麦克风设备: {self._mic_device}"

                self.ui.set_command_output("Microphone", msg)
                self._safe_invalidate()

            elif sub == "info":
                mic = _get_mic_info(self._mic_device)
                valid = "有效（输入设备）" if mic["valid"] else "无效（无输入通道）"
                self.ui.set_command_output(
                    "Microphone Info",
                    [
                        f"设备: [{mic['id']}] {mic['name']}",
                        f"状态: {valid}",
                        f"采样率: {mic['sr']:.0f} Hz",
                        f"输入通道: {mic['channels']}",
                    ],
                )
                self._safe_invalidate()

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
                self.ui.set_command_output("Microphone", help_text)
                self._safe_invalidate()
        except Exception as e:
            self._record_ui_error("cmd_mic", e)

    async def _cmd_mic_monitor(self) -> None:
        try:
            if self._mic is None:
                self.ui.set_command_output("Microphone", "未配置麦克风，启动时未传入 mic 参数")
                self._safe_invalidate()
                return

            self._mic_monitoring = not self._mic_monitoring
            self.ui.mic.monitoring = self._mic_monitoring

            if self._mic_monitoring:
                mic_info = _get_mic_info(self._mic_device)
                self.ui.mic.device_name = mic_info["name"] if mic_info["valid"] else ""
                self._mic_monitor_task = asyncio.create_task(self._mic_monitor_loop())
                self.ui.set_command_output("Microphone", "麦克风监测已启动")
            else:
                if self._mic_monitor_task:
                    self._mic_monitor_task.cancel()
                    self._mic_monitor_task = None
                self.ui.mic.rms = 0.0
                self.ui.set_command_output("Microphone", "麦克风监测已停止")
        except Exception as e:
            self._record_ui_error("mic_monitor", e)
        self._safe_invalidate()

    async def _cmd_mic_autodetect(self, *args: str) -> None:
        try:
            import sounddevice as sd

            self.ui.set_command_output("Microphone", "正在逐个探测麦克风设备（每设备 300ms）...")

            loop = asyncio.get_running_loop()
            devices = sd.query_devices()
            results: list[tuple[int, str, float]] = []

            for i, dev in enumerate(devices):
                if dev["max_input_channels"] <= 0:
                    continue

                self.ui.status_line = f"探测 [{i}] {dev['name']} ..."
                self._safe_invalidate()

                rms = await loop.run_in_executor(None, _probe_device_rms, i, 16000, 0.3)
                results.append((i, dev["name"], rms))

            self.ui.status_line = ""
            results.sort(key=lambda x: x[2], reverse=True)

            out_lines = ["麦克风探测结果："]
            for rank, (did, name, rms) in enumerate(results, 1):
                from voice_agent.cli.formatters import vu_bar
                bar = vu_bar(rms, 15)
                tag = " 最佳" if rank == 1 else ""
                status = f"✓ {rms:.6f}" if rms > 0 else "打开失败"
                out_lines.append(f"  #{rank} [{did}] {name}  {bar}  {status}{tag}")

            auto_select = "--select" in args or "-s" in args
            if results and results[0][2] > 0.005:
                best_id, best_name, best_rms = results[0]
                if auto_select:
                    self._mic_device = best_id
                    self.ui.mic.device_name = best_name
                    out_lines.append(f"已选择最佳设备: [{best_id}] {best_name}")
                    if self._mic_monitoring and self._mic is not None:
                        out_lines.append("正在重启麦克风监测使用新设备...")
                        if self._mic_monitor_task:
                            self._mic_monitor_task.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await self._mic_monitor_task
                            self._mic_monitor_task = None
                        with contextlib.suppress(Exception):
                            await self._mic.stop()
                        self._mic.device = best_id
                        self._mic_monitor_task = asyncio.create_task(self._mic_monitor_loop())
                        out_lines.append("监测已切换到新设备")
                else:
                    out_lines.append(f"提示: 添加 --select 自动选择 [{best_id}] {best_name}")
            elif results and results[0][2] > 0:
                out_lines.append("所有设备音量极低，未自动选择")
            else:
                out_lines.append("未检测到可用的麦克风设备")

            self.ui.set_command_output("Microphone Auto-Detect", out_lines)
        except Exception as e:
            self._record_ui_error("mic_autodetect", e)
        self._safe_invalidate()

    # ------------------------------------------------------------------
    # /wakeup /sleep /judge 命令
    # ------------------------------------------------------------------

    async def _cmd_wakeup(self, *args: str) -> None:
        """叫醒 AI，进入连续对话状态；不启动麦克风。"""
        self.ui.assistant_awake = True
        self.ui.runtime_state = "text_ready"
        self.ui.conversation_mode = "active_chat"

        # 激活 ConversationState wake_session
        try:
            matcher = getattr(self._agent._gate, "wake_matcher", None)
            assistant_name = self.ui.assistant_name or "琉璃川"
            if matcher is not None:
                assistant_name = getattr(matcher.config, "name", assistant_name)
            if hasattr(self._state, "activate_wake_session"):
                self._state.activate_wake_session(120, assistant_name)
        except Exception as e:
            self._logger.warning("[TUI] activate wake session failed: %s", e)

        self.ui.set_command_output(
            "Wakeup",
            [
                f"{self.ui.assistant_name} 已醒来。",
                "文字聊天已可用，你可以直接输入内容。",
                "如果需要实时语音倾听，请输入 /listen。",
                "如果要休息，请输入 /sleep。",
            ],
        )
        self._safe_invalidate()

    async def _cmd_sleep(self, *args: str) -> None:
        """让 AI 进入睡眠状态，并停止实时监听。"""
        self.ui.assistant_awake = False
        self.ui.conversation_mode = "passive_listening"

        try:
            if hasattr(self._state, "end_wake_session"):
                self._state.end_wake_session()
        except Exception:
            pass

        if self._runtime_controller is not None:
            try:
                if hasattr(self._runtime_controller, "stop_listening"):
                    await self._runtime_controller.stop_listening()
                else:
                    await self._runtime_controller.sleep()
            except Exception as e:
                self._logger.exception("[TUI] sleep stop listening failed: %s", e)

        self.ui.voice_listening = False
        self.ui.runtime_state = "text_ready"
        self.ui.asr.status = "idle"

        self.ui.set_command_output(
            "Sleep",
            [
                f"{self.ui.assistant_name} 已进入睡眠状态。",
                "实时语音监听已停止。",
                "文字输入仍然可用。",
                "",
                "输入 /wakeup 可以叫醒 AI。",
                "输入 /listen 可以开启实时倾听。",
            ],
        )
        self._safe_invalidate()

    async def _cmd_judge(self, *args: str) -> None:
        """查看或切换 Judge provider。"""
        gate = getattr(self._agent, "_gate", None)
        if gate is None:
            self.ui.set_command_output("Judge", "Gate 未初始化")
            self._safe_invalidate()
            return

        if not args:
            current = getattr(gate, "judge_provider", "rule")
            self.ui.set_command_output(
                "Judge",
                [
                    f"当前判断器: {current}",
                    "",
                    "用法:",
                    "  /judge            查看当前判断器",
                    "  /judge rule       规则判断（最快）",
                    "  /judge local      本地小模型判断（qwen3.5:4b）",
                    "  /judge llm        主 LLM 判断（最慢）",
                ],
            )
            self._safe_invalidate()
            return

        provider = args[0]
        valid = ("rule", "local", "llm")
        if provider not in valid:
            self.ui.set_command_output(
                "Judge",
                f"无效判断器: {provider}\n可用: {', '.join(valid)}",
            )
            self._safe_invalidate()
            return

        # 更新 Gate 的 judge_provider
        gate.judge_provider = provider
        self.ui.judge_provider = provider
        self.ui.add_notification(f"判断器已切换为: {provider}")
        self.ui.set_command_output("Judge", f"判断器已切换为: {provider}")
        self._safe_invalidate()

    async def _cmd_listen(self, *args: str) -> None:
        """启动实时麦克风倾听 / ASR。"""
        if self._runtime_controller is None:
            self.ui.set_command_output("Listen", "RuntimeController 未初始化，无法启动实时监听。")
            self._safe_invalidate()
            return

        self.ui.set_command_output(
            "Listen",
            [
                "正在启动实时语音监听...",
                "如果首次加载 ASR 模型，可能需要等待几秒。",
            ],
        )
        self._safe_invalidate()

        try:
            if hasattr(self._runtime_controller, "start_listening"):
                ok = await self._runtime_controller.start_listening()
            else:
                ok = await self._runtime_controller.wakeup()

            if ok:
                self.ui.voice_listening = True
                self.ui.runtime_state = "listening"
                self.ui.asr.status = "listening"
                self.ui.set_command_output(
                    "Listen",
                    [
                        "实时语音监听已启动。",
                        "现在可以直接说话。",
                        "",
                        "输入 /sleep 可以停止实时监听并让 AI 休息。",
                    ],
                )
            else:
                self.ui.set_command_output(
                    "Listen",
                    [
                        "实时语音监听启动失败。",
                        "请检查 ASR 模型、麦克风设备和 logs/minions.log。",
                    ],
                )
        except Exception as e:
            self._logger.exception("[TUI] listen failed: %s", e)
            self.ui.set_command_output("Listen", f"启动实时监听失败: {e}")

        self._safe_invalidate()

    # ------------------------------------------------------------------
    # 麦克风监测
    # ------------------------------------------------------------------

    async def _mic_monitor_loop(self) -> None:
        """后台任务：持续采集麦克风并更新 RMS。"""
        if self._mic is None:
            return
        try:
            await self._mic.start()
            while self._mic_monitoring and self.ui.running and not self._exit_requested:
                chunk = await self._mic.read_chunk()
                rms = calculate_rms(chunk)
                self.ui.mic.rms = rms
                now = time.monotonic()
                if now - self._last_invalidate > _AUDIO_LEVEL_THROTTLE:
                    self._last_invalidate = now
                    self._app.invalidate()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            try:
                self.ui.add_system_message(f"麦克风采集异常: {e}")
                self._app.invalidate()
            except Exception:
                pass
        finally:
            with contextlib.suppress(Exception):
                await self._mic.stop()
