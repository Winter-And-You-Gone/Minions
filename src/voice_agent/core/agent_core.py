"""Agent 核心：编排 ASR → Gate → Judge → Agent 完整链路。"""

from voice_agent.event_bus import EventBus
from voice_agent.core.conversation_state import ConversationState
from voice_agent.core.intervention_gate import InterventionGate, GateAction
from voice_agent.core.llm_client import LLMClient
from voice_agent.logger import get_logger


class AgentCore:
    """Agent 核心处理器。

    流程：
      ASR final text → InterventionGate → (可选 LLM judge) → (可选 LLM agent) → 发布 agent.reply
    """

    def __init__(
        self,
        event_bus: EventBus,
        state: ConversationState,
        gate: InterventionGate,
        llm: LLMClient,
    ) -> None:
        self._bus = event_bus
        self._state = state
        self._gate = gate
        self._llm = llm
        self._logger = get_logger()
        self._recent_context: list[str] = []

    async def handle_final_text(self, text: str, confidence: float = 1.0) -> None:
        """处理 ASR final 文本。"""
        # 1. 先用当前 state 做 Gate 判断（必须在更新 state 之前）
        gate_result = self._gate.evaluate(text, self._state, confidence)
        metadata = gate_result.metadata or {}

        # 2. 检查结束唤醒会话指令
        if metadata.get("end_wake_session"):
            self._state.end_wake_session()
            await self._bus.publish({
                "type": "state.change",
                "state": self._state.mode,
                "reason": "end_wake_session",
            })
            self._logger.info("[Wake] 唤醒会话已结束")
            return

        # 3. 唤醒名检测 → 激活唤醒会话
        if metadata.get("wake_detected"):
            wake_seconds = getattr(self._gate.wake_matcher.config, "session_seconds", 120)
            wake_name = metadata.get("wake_name", "")
            self._state.activate_wake_session(wake_seconds, wake_name)
            self._logger.info("[Wake] 唤醒会话已激活: name=%s seconds=%s", wake_name, wake_seconds)
            # 广播状态变化
            await self._bus.publish({
                "type": "state.change",
                "state": self._state.mode,
                "reason": f"wake:{wake_name}",
            })

        # 4. 更新 state — 使用 gate_result.text（可能已被 strip_wake_name 处理）
        normalized_text = gate_result.text or text
        self._state.mark_user_final_text(normalized_text)

        await self._bus.publish({
            "type": "gate.result",
            "text": text,
            "normalized_text": normalized_text,
            "action": gate_result.action.value,
            "score": gate_result.score,
            "reason": gate_result.reason,
        })
        self._logger.info(
            "[Gate] action=%s score=%d reason=%s",
            gate_result.action.value,
            gate_result.score,
            gate_result.reason,
        )

        # 5. 唤醒会话内每次有效输入刷新超时
        if self._state.is_wake_session_active():
            wake_seconds = getattr(self._gate.wake_matcher.config, "session_seconds", 120)
            self._state.refresh_wake_session(wake_seconds)

        # 6. 根据 action 处理
        if gate_result.action == GateAction.SILENT:
            return

        if gate_result.action == GateAction.BUBBLE:
            await self._bus.publish({
                "type": "bubble",
                "text": text,
                "message": f"[低打扰] 检测到可能意图: {gate_result.reason}",
            })
            return

        # 7. 单独喊名字时本地回复
        if metadata.get("wake_detected") and not metadata.get("text_without_name"):
            reply = "我在。"
            self._state.mark_agent_replied()
            self._state.enter_cooldown()
            self._recent_context.append(f"用户: {text}")
            self._recent_context.append(f"AI: {reply}")
            if len(self._recent_context) > 20:
                self._recent_context = self._recent_context[-20:]
            await self._bus.publish({"type": "agent.reply", "text": reply})
            self._logger.info("[Agent] 唤醒回复: %s", reply)
            return

        # 8. JUDGE → 调用 LLM 二次判断
        if gate_result.action == GateAction.JUDGE:
            should_skip_judge = False

            # 唤醒会话内可让 LLM 判断是否转向别人
            if self._state.is_wake_session_active():
                allow_judge = getattr(
                    self._gate.wake_matcher.config, "allow_llm_turn_away_judge", True
                )
                if allow_judge:
                    turn_away = await self._llm.judge_wake_session_continue(
                        normalized_text,
                        "\n".join(self._recent_context[-6:]),
                    )
                    if not turn_away.get("continue_session", True):
                        self._state.end_wake_session()
                        await self._bus.publish({
                            "type": "state.change",
                            "state": self._state.mode,
                            "reason": "llm_turn_away",
                        })
                        self._logger.info("[Wake] LLM 判断转向别人，结束唤醒会话")
                        return
                    # LLM 确认仍在对话，跳过 JUDGE 直接进入 AGENT
                    should_skip_judge = True
                    self._logger.info("[Wake] LLM 判断仍在对话，跳过 JUDGE 进入 AGENT")

            if not should_skip_judge:
                judge = await self._llm.judge_intervention(
                    text,
                    self._state.mode,
                    self._state.seconds_since_last_reply() < 60,
                )
                self._logger.info("[Judge] result=%s", judge)
            if not judge.get("should_reply", False):
                return

        # 9. AGENT → 调用 LLM 生成回复
        # 系统信息问题由本地回答，不让 LLM 猜
        if self._is_model_info_question(text):
            model_name = self._llm.model or "未配置模型"
            reply = f"我当前连接的模型是 {model_name}。"
            self._state.mark_agent_replied()
            self._state.enter_cooldown()
            self._recent_context.append(f"用户: {text}")
            self._recent_context.append(f"AI: {reply}")
            if len(self._recent_context) > 20:
                self._recent_context = self._recent_context[-20:]
            await self._bus.publish({"type": "agent.reply", "text": reply})
            self._logger.info("[Agent] reply=%s", reply)
            return

        context = "\n".join(self._recent_context[-5:]) if self._recent_context else ""
        reply = await self._llm.generate_reply(text, context)
        self._state.mark_agent_replied()

        # 保存上下文
        self._recent_context.append(f"用户: {text}")
        self._recent_context.append(f"AI: {reply}")
        if len(self._recent_context) > 20:
            self._recent_context = self._recent_context[-20:]

        # 冷却
        self._state.enter_cooldown()

        await self._bus.publish({
            "type": "agent.reply",
            "text": reply,
        })
        self._logger.info("[Agent] reply=%s", reply)

    @staticmethod
    def _is_model_info_question(text: str) -> bool:
        """判断用户是否在询问当前模型信息。

        这类问题由本地配置直接回答，不让 LLM 自己猜。
        """
        keywords = [
            "你是什么模型",
            "你用的什么模型",
            "当前模型",
            "模型名",
            "你是谁家的模型",
        ]
        return any(k in text for k in keywords)

    async def handle_pause(self) -> None:
        self._state.pause()
        await self._bus.publish({
            "type": "state.change",
            "state": "paused",
        })

    async def handle_resume(self) -> None:
        self._state.resume()
        await self._bus.publish({
            "type": "state.change",
            "state": self._state.mode,
        })
