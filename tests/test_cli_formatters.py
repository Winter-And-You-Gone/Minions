"""测试格式化函数。"""

import math

from voice_agent.cli.formatters import format_chat, format_decision, format_header, vu_bar
from voice_agent.cli.ui_state import UIState, ASRView, LLMView, GateView, GateSnapshot


class TestVuBar:
    def test_zero_rms(self) -> None:
        bar = vu_bar(0.0, width=20)
        assert bar == "░" * 20

    def test_negative_rms(self) -> None:
        bar = vu_bar(-1.0, width=10)
        assert bar == "░" * 10

    def test_positive_rms(self) -> None:
        bar = vu_bar(0.1, width=10)
        assert "█" in bar
        assert bar.count("█") + bar.count("░") == 10

    def test_high_rms_full(self) -> None:
        # rms=1.0 → 0dB → norm=1.0 → full bar
        bar = vu_bar(1.0, width=10)
        assert bar == "█" * 10

    def test_custom_width(self) -> None:
        bar = vu_bar(0.5, width=5)
        assert len(bar) == 5


class TestFormatChat:
    def test_empty_chat(self) -> None:
        s = UIState()
        frags = format_chat(s)
        # 空状态时，不包含任何消息，只有可能的空字符
        texts = [t for _, t in frags]
        combined = "".join(texts)
        assert combined == ""

    def test_user_message_formatted(self) -> None:
        s = UIState()
        s.add_user_message("hello")
        frags = format_chat(s)
        combined = "".join(t for _, t in frags)
        assert "你：" in combined
        assert "hello" in combined

    def test_assistant_message_formatted(self) -> None:
        s = UIState()
        s.add_assistant_message("world")
        frags = format_chat(s)
        combined = "".join(t for _, t in frags)
        assert "AI：" in combined
        assert "world" in combined

    def test_system_message_formatted(self) -> None:
        s = UIState()
        s.add_system_message("system info")
        frags = format_chat(s)
        combined = "".join(t for _, t in frags)
        assert "•" in combined or "system info" in combined

    def test_hidden_count_message(self) -> None:
        s = UIState(max_visible_messages=2)
        for i in range(5):
            s.add_user_message(str(i))
        frags = format_chat(s)
        combined = "".join(t for _, t in frags)
        assert "已折叠" in combined
        assert "3" in combined  # 5 - 2 = 3 hidden

    def test_no_hidden_when_under_max(self) -> None:
        s = UIState(max_visible_messages=10)
        s.add_user_message("hello")
        frags = format_chat(s)
        combined = "".join(t for _, t in frags)
        assert "已折叠" not in combined

    def test_mixed_messages_order(self) -> None:
        s = UIState(max_visible_messages=5)
        s.add_user_message("q1")
        s.add_assistant_message("a1")
        s.add_user_message("q2")
        frags = format_chat(s)
        combined = "".join(t for _, t in frags)
        assert combined.index("q1") < combined.index("a1")
        assert combined.index("a1") < combined.index("q2")

    def test_error_line(self) -> None:
        s = UIState()
        s.error_line = "something broke"
        frags = format_chat(s)
        combined = "".join(t for _, t in frags)
        assert "something broke" in combined


class TestFormatDecision:
    def test_no_gate_result(self) -> None:
        s = UIState()
        frags = format_decision(s)
        combined = "".join(t for _, t in frags)
        assert "等待" in combined

    def test_gate_action_displayed(self) -> None:
        s = UIState()
        s.latest_gate = GateView(action="agent", score=90, reason="test reason")
        frags = format_decision(s)
        combined = "".join(t for _, t in frags)
        assert "agent" in combined
        assert "90" in combined
        assert "test reason" in combined

    def test_gate_silent_short(self) -> None:
        s = UIState()
        s.latest_gate = GateSnapshot(action="silent", score=5)
        frags = format_decision(s)
        combined = "".join(t for _, t in frags)
        assert "silent" in combined
        assert "5" in combined


class TestFormatHeader:
    def test_basic_state(self) -> None:
        s = UIState()
        frags = format_header(s)
        combined = "".join(t for _, t in frags)
        assert "Minions" in combined
        assert "passive_listening" in combined

    def test_shows_llm_model(self) -> None:
        s = UIState(llm=LLMView(model="gpt-4", available=True))
        frags = format_header(s)
        combined = "".join(t for _, t in frags)
        assert "gpt-4" in combined

    def test_shows_asr_status(self) -> None:
        s = UIState(asr=ASRView(status="listening"))
        frags = format_header(s)
        combined = "".join(t for _, t in frags)
        assert "listening" in combined

    def test_shows_paused(self) -> None:
        s = UIState(paused=True)
        frags = format_header(s)
        combined = "".join(t for _, t in frags)
        assert "暂停" in combined

    def test_mic_monitoring(self) -> None:
        s = UIState()
        s.mic.monitoring = True
        s.mic.rms = 0.05
        s.mic.device_name = "Test Mic"
        frags = format_header(s)
        combined = "".join(t for _, t in frags)
        assert "Test Mic" in combined

    def test_mic_not_monitoring(self) -> None:
        s = UIState()
        s.mic.monitoring = False
        frags = format_header(s)
        combined = "".join(t for _, t in frags)
        assert "麦克风已停止" in combined
