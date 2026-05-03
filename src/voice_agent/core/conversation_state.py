"""对话状态机：管理旁听/对话/冷却/暂停状态。"""

import time
from dataclasses import dataclass


@dataclass
class ConversationState:
    """对话状态。

    mode:
      passive_listening - 旁听状态，默认不回应
      active_chat       - 对话状态，用户正在和 AI 交流
      cooldown          - 冷却状态，避免连续插嘴（active_chat 的子状态）
      paused            - 隐私暂停状态
    """

    mode: str = "passive_listening"
    active_until: float = 0.0
    cooldown_until: float = 0.0
    last_agent_reply_at: float = 0.0
    last_user_text: str = ""
    last_final_text_at: float = 0.0

    _cooldown_seconds: float = 5.0
    _conversation_timeout_seconds: float = 60.0

    def is_active_conversation(self) -> bool:
        """当前是否处于活跃对话中。"""
        now = time.time()
        if self.mode in ("active_chat", "cooldown"):
            if self.active_until > 0 and now > self.active_until:
                self.mode = "passive_listening"
                self.active_until = 0.0
                self.cooldown_until = 0.0
                return False
            return True
        return False

    def is_in_cooldown(self) -> bool:
        """是否在冷却中。"""
        if self.mode == "cooldown":
            if time.time() > self.cooldown_until:
                # 冷却到期，回到 active_chat（如果 active_until 仍有效）或 passive_listening
                if self.active_until > time.time():
                    self.mode = "active_chat"
                else:
                    self.mode = "passive_listening"
                    self.active_until = 0.0
                self.cooldown_until = 0.0
                return False
            return True
        return False

    def mark_user_final_text(self, text: str) -> None:
        """记录用户 final 文本并更新状态。"""
        now = time.time()
        self.last_user_text = text
        self.last_final_text_at = now

        if self.mode == "cooldown":
            if now > self.cooldown_until:
                if self.active_until > now:
                    self.mode = "active_chat"
                else:
                    self.mode = "passive_listening"
                    self.active_until = 0.0
                self.cooldown_until = 0.0

    def mark_agent_replied(self) -> None:
        """标记 AI 已回复，进入活跃对话并延长 active_until。"""
        now = time.time()
        self.last_agent_reply_at = now
        self.active_until = now + self._conversation_timeout_seconds
        self.cooldown_until = 0.0
        if self.mode != "paused":
            self.mode = "active_chat"

    def enter_cooldown(self) -> None:
        """进入冷却状态（不覆盖 active_until）。"""
        now = time.time()
        self.mode = "cooldown"
        self.cooldown_until = now + self._cooldown_seconds

    def seconds_since_last_reply(self) -> float:
        """距离 AI 上次回复的秒数。"""
        if self.last_agent_reply_at == 0:
            return float("inf")
        return time.time() - self.last_agent_reply_at

    def seconds_since_last_final(self) -> float:
        """距离用户最后 final 文本的秒数。"""
        if self.last_final_text_at == 0:
            return float("inf")
        return time.time() - self.last_final_text_at

    def pause(self) -> None:
        self.mode = "paused"
        self.active_until = 0.0
        self.cooldown_until = 0.0

    def resume(self) -> None:
        self.mode = "passive_listening"
        self.active_until = 0.0
        self.cooldown_until = 0.0

    def is_paused(self) -> bool:
        return self.mode == "paused"
