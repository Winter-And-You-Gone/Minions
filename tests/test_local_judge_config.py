"""测试 LocalJudge 配置注入。"""

from voice_agent.main import build_gate
from voice_agent.core.intervention_gate import GateAction
from voice_agent.core.conversation_state import ConversationState


def make_config() -> dict:
    return {
        "intervention": {
            "thresholds": {"bubble": 15, "judge": 30, "agent": 60},
            "uncertain_action": "judge",
        },
        "judge": {
            "provider": "local",
            "thresholds": {
                "local_judge_min": 10,
                "local_judge_max": 74,
            },
        },
        "assistant": {
            "name": "琉璃川",
            "wake_aliases": ["琉璃川"],
            "wake": {"enabled": True},
        },
    }


class TestBuildGateLocalJudge:
    def test_build_gate_local_judge_config(self) -> None:
        gate = build_gate(make_config())

        assert gate.judge_provider == "local"
        assert gate.local_judge_min == 10
        assert gate.local_judge_max == 74

    def test_plain_question_goes_local_judge_when_enabled(self) -> None:
        gate = build_gate(make_config())
        state = ConversationState(_cooldown_seconds=0)

        result = gate.evaluate("这是怎么回事", state, 1.0)

        assert result.action == GateAction.LOCAL_JUDGE

    def test_strong_trigger_still_agent(self) -> None:
        gate = build_gate(make_config())
        state = ConversationState(_cooldown_seconds=0)

        result = gate.evaluate("帮我打开浏览器", state, 1.0)

        assert result.action == GateAction.AGENT

    def test_short_text_still_silent(self) -> None:
        gate = build_gate(make_config())
        state = ConversationState(_cooldown_seconds=0)

        result = gate.evaluate("嗯", state, 1.0)

        assert result.action == GateAction.SILENT

    def test_weak_trigger_goes_local_judge(self) -> None:
        """弱触发词 score=25 在 10-74 范围内 → LOCAL_JUDGE。"""
        gate = build_gate(make_config())
        state = ConversationState(_cooldown_seconds=0)

        result = gate.evaluate("这个代码好奇怪", state, 1.0)
        assert result.action == GateAction.LOCAL_JUDGE
