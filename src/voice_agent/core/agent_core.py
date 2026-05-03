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

        # 2. 再更新 state，避免提前更新导致重复文本误判
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

        # 2. 根据 action 处理
        if gate_result.action == GateAction.SILENT:
            return

        if gate_result.action == GateAction.BUBBLE:
            await self._bus.publish({
                "type": "bubble",
                "text": text,
                "message": f"[低打扰] 检测到可能意图: {gate_result.reason}",
            })
            return

        # 3. JUDGE → 调用 LLM 二次判断
        if gate_result.action == GateAction.JUDGE:
            judge = await self._llm.judge_intervention(
                text,
                self._state.mode,
                self._state.seconds_since_last_reply() < 60,
            )
            self._logger.info("[Judge] result=%s", judge)
            if not judge.get("should_reply", False):
                return

        # 4. AGENT → 调用 LLM 生成回复
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
