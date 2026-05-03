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

    conversation_mode: str = "passive_listening"

    messages: list[ChatMessage] = field(default_factory=list)
    max_visible_messages: int = 8

    latest_gate: GateView = field(default_factory=GateView)
    asr: ASRView = field(default_factory=ASRView)
    llm: LLMView = field(default_factory=LLMView)

    mic: MicSnapshot = field(default_factory=MicSnapshot)

    status_line: str = ""
    error_line: str = ""

    def add_user_message(self, text: str) -> None:
        self.messages.append(ChatMessage(MessageRole.USER, text))

    def add_assistant_message(self, text: str) -> None:
        self.messages.append(ChatMessage(MessageRole.ASSISTANT, text))

    def add_system_message(self, text: str) -> None:
        self.messages.append(ChatMessage(MessageRole.SYSTEM, text))

    @property
    def hidden_message_count(self) -> int:
        return max(0, len(self.messages) - self.max_visible_messages)

    @property
    def visible_messages(self) -> list[ChatMessage]:
        return self.messages[-self.max_visible_messages:]


# 向后兼容别名
GateSnapshot = GateView
