"""UI 状态模型 — 驱动动态 CLI 界面的所有状态。"""

import time
from dataclasses import dataclass, field
from enum import Enum


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ChatMessage:
    role: MessageRole
    text: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class CompletionItem:
    text: str = ""
    display: str = ""
    display_meta: str = ""
    execute_text: str = ""


@dataclass
class GateView:
    action: str = ""
    score: int = 0
    reason: str = ""


@dataclass
class ASRView:
    status: str = "idle"  # idle | loading | loaded | listening | recognizing | error
    model: str = ""


@dataclass
class LLMView:
    model: str = ""
    available: bool = False


@dataclass
class MicSnapshot:
    monitoring: bool = False
    rms: float = 0.0
    device_name: str = ""


@dataclass
class UIState:
    running: bool = True
    paused: bool = False

    # 应用信息
    app_name: str = "Minions"
    assistant_name: str = "琉璃川"
    conversation_mode: str = "passive_listening"

    messages: list[ChatMessage] = field(default_factory=list)
    max_visible_messages: int = 8

    latest_gate: GateView = field(default_factory=GateView)
    asr: ASRView = field(default_factory=ASRView)
    llm: LLMView = field(default_factory=LLMView)

    mic: MicSnapshot = field(default_factory=MicSnapshot)

    # 运行时信息
    asr_engine: str = "sherpa-onnx"
    judge_provider: str = "local"
    judge_model: str = "qwen3.5:4b"
    llm_model: str = ""
    runtime_state: str = "text_ready"

    # 三种独立状态
    text_ready: bool = True          # 文字聊天始终可用
    assistant_awake: bool = False    # /wakeup 叫醒后为 True
    voice_listening: bool = False    # /listen 启动实时语音后为 True

    # 唤醒状态
    wake_active: bool = False
    wake_remaining_seconds: float = 0.0

    # 最新 Judge 结果
    latest_judge_provider: str = ""
    latest_judge_target: str = ""
    latest_judge_should_reply: bool = False
    latest_judge_confidence: float = 0.0
    latest_judge_reason: str = ""

    # 右侧通知
    notifications: list[str] = field(default_factory=list)
    max_notifications: int = 6

    # 健康检查
    health_items: list = field(default_factory=list)

    status_line: str = ""
    error_line: str = ""

    # 命令补全
    completion_items: list[CompletionItem] = field(default_factory=list)
    completion_selected_index: int = 0
    completion_visible: bool = False
    completion_reserved_rows: int = 6

    # 命令面板模式
    command_panel_mode: str = "blank"
    # "blank" = 空白 | "completion" = 补全列表 | "help" = 命令浏览器 | "output" = 命令输出
    command_panel_title: str = ""
    command_panel_selected_index: int = 0
    command_panel_reserved_rows: int = 14
    help_tab: str = "commands"
    help_items: list[dict] = field(default_factory=list)

    # 命令输出面板
    command_output_title: str = ""
    command_output_lines: list[str] = field(default_factory=list)

    # 命令面板滚动
    command_panel_scroll_offset: int = 0

    # 底部状态栏
    footer_left: str = ""
    footer_right: str = ""

    # 主页面板
    version_text: str = ""
    tips_lines: list[str] = field(default_factory=list)
    current_path: str = ""

    def add_notification(self, text: str) -> None:
        if not text:
            return
        self.notifications.append(text)
        if len(self.notifications) > self.max_notifications:
            self.notifications = self.notifications[-self.max_notifications:]

    def add_user_message(self, text: str) -> None:
        self.messages.append(ChatMessage(MessageRole.USER, text))

    def add_assistant_message(self, text: str) -> None:
        self.messages.append(ChatMessage(MessageRole.ASSISTANT, text))

    def add_system_message(self, text: str) -> None:
        self.messages.append(ChatMessage(MessageRole.SYSTEM, text))

    def set_command_output(self, title: str, text: str | list[str]) -> None:
        self.command_panel_mode = "output"
        self.command_output_title = title
        if isinstance(text, str):
            self.command_output_lines = text.splitlines() or [text]
        else:
            self.command_output_lines = [str(x) for x in text]
        self.completion_visible = False
        self.completion_items = []
        self.command_panel_scroll_offset = 0

    def clear_command_panel(self) -> None:
        self.command_panel_mode = "blank"
        self.command_output_title = ""
        self.command_output_lines = []
        self.completion_visible = False
        self.completion_items = []
        self.completion_selected_index = 0
        self.command_panel_selected_index = 0
        self.command_panel_scroll_offset = 0

    @property
    def hidden_message_count(self) -> int:
        return max(0, len(self.messages) - self.max_visible_messages)

    @property
    def visible_messages(self) -> list[ChatMessage]:
        return self.messages[-self.max_visible_messages:]


# 向后兼容别名
GateSnapshot = GateView
