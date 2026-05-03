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
class GateSnapshot:
    action: str = ""
    score: int = 0
    reason: str = ""


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
    llm_model: str = ""
    llm_available: bool = False

    messages: list[ChatMessage] = field(default_factory=list)
    max_visible_messages: int = 8

    latest_gate: GateSnapshot = field(default_factory=GateSnapshot)
    latest_bubble: str = ""

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
