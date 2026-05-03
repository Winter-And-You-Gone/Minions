"""测试 InterventionGate 唤醒会话行为。"""

from voice_agent.core.conversation_state import ConversationState
from voice_agent.core.intervention_gate import InterventionGate, GateAction
from voice_agent.core.wake_name import WakeNameMatcher, WakeNameConfig


def _gate_with_wake() -> InterventionGate:
    matcher = WakeNameMatcher(WakeNameConfig(name="米粒", aliases=["米粒", "迷你"]))
    return InterventionGate(wake_matcher=matcher)


def _state_with_wake() -> ConversationState:
    state = ConversationState()
    state.activate_wake_session(120, "米粒")
    return state


class TestWakeNameTriggersAgent:
    def test_wake_name_triggers_agent(self) -> None:
        state = ConversationState()
        gate = _gate_with_wake()
        result = gate.evaluate("米粒帮我看看", state, 1.0)
        assert result.action == GateAction.AGENT
        assert result.metadata["wake_detected"] is True

    def test_wake_name_only_name_triggers_agent(self) -> None:
        state = ConversationState()
        gate = _gate_with_wake()
        result = gate.evaluate("米粒", state, 1.0)
        assert result.action == GateAction.AGENT
        assert result.metadata["wake_detected"] is True

    def test_wake_name_alias_triggers_agent(self) -> None:
        state = ConversationState()
        gate = _gate_with_wake()
        result = gate.evaluate("迷你帮我看看", state, 1.0)
        assert result.action == GateAction.AGENT
        assert result.metadata["wake_detected"] is True


class TestWakeSessionFollowup:
    def test_wake_session_followup_triggers_agent(self) -> None:
        state = _state_with_wake()
        gate = InterventionGate()  # no wake_matcher, uses state weighting

        result = gate.evaluate("然后呢", state, 1.0)
        # 55 (followup) + 75 (wake_session) = 130 >= 60 -> AGENT
        assert result.action == GateAction.AGENT

    def test_wake_session_short_question(self) -> None:
        state = _state_with_wake()
        gate = InterventionGate()

        result = gate.evaluate("然后呢", state, 1.0)
        assert result.action == GateAction.AGENT

    def test_wake_session_active_outside_wake(self) -> None:
        """wake_session 加分大于 active_chat 加分。"""
        state = _state_with_wake()
        gate = InterventionGate()

        result = gate.evaluate("然后呢", state, 1.0)
        assert result.score >= 130  # 55 + 75

    def test_no_wake_session_lower_score(self) -> None:
        state = ConversationState()
        state.mode = "active_chat"
        gate = InterventionGate()

        result = gate.evaluate("然后呢", state, 1.0)
        assert result.score < 100  # just followup 55 + active_chat 30 = 85


class TestEndWakeSession:
    def test_end_wake_session_trigger(self) -> None:
        state = _state_with_wake()
        gate = _gate_with_wake()

        result = gate.evaluate("没事了", state, 1.0)
        assert result.action == GateAction.SILENT
        assert result.metadata["end_wake_session"] is True

    def test_end_wake_session_not_related(self) -> None:
        state = ConversationState()
        gate = _gate_with_wake()

        result = gate.evaluate("帮我看一下", state, 1.0)
        assert not result.metadata.get("end_wake_session")

    def test_bi_zui_ends_session(self) -> None:
        state = _state_with_wake()
        gate = _gate_with_wake()

        result = gate.evaluate("闭嘴", state, 1.0)
        assert result.action == GateAction.SILENT
        assert result.metadata["end_wake_session"] is True
