"""测试 UIState 数据模型。"""

from voice_agent.cli.ui_state import UIState, ChatMessage, GateSnapshot, MicSnapshot, MessageRole


class TestUIState:
    def test_add_user_message(self) -> None:
        s = UIState()
        s.add_user_message("hello")
        assert len(s.messages) == 1
        assert s.messages[0].role == MessageRole.USER
        assert s.messages[0].text == "hello"

    def test_add_assistant_message(self) -> None:
        s = UIState()
        s.add_assistant_message("world")
        assert len(s.messages) == 1
        assert s.messages[0].role == MessageRole.ASSISTANT
        assert s.messages[0].text == "world"

    def test_add_system_message(self) -> None:
        s = UIState()
        s.add_system_message("info")
        assert len(s.messages) == 1
        assert s.messages[0].role == MessageRole.SYSTEM
        assert s.messages[0].text == "info"

    def test_hidden_count_below_max(self) -> None:
        s = UIState(max_visible_messages=5)
        for i in range(3):
            s.add_user_message(str(i))
        assert s.hidden_message_count == 0

    def test_hidden_count_above_max(self) -> None:
        s = UIState(max_visible_messages=3)
        for i in range(5):
            s.add_user_message(str(i))
        assert s.hidden_message_count == 2

    def test_visible_messages_only_latest(self) -> None:
        s = UIState(max_visible_messages=3)
        for i in range(5):
            s.add_user_message(str(i))
        visible = s.visible_messages
        assert len(visible) == 3
        assert [m.text for m in visible] == ["2", "3", "4"]

    def test_visible_messages_all_if_under_max(self) -> None:
        s = UIState(max_visible_messages=10)
        for i in range(3):
            s.add_user_message(str(i))
        visible = s.visible_messages
        assert len(visible) == 3

    def test_empty_state_no_hidden(self) -> None:
        s = UIState()
        assert s.hidden_message_count == 0
        assert s.visible_messages == []

    def test_gate_snapshot_update(self) -> None:
        s = UIState()
        s.latest_gate = GateSnapshot(action="agent", score=90, reason="test")
        assert s.latest_gate.action == "agent"
        assert s.latest_gate.score == 90
        assert s.latest_gate.reason == "test"

    def test_gate_snapshot_defaults(self) -> None:
        g = GateSnapshot()
        assert g.action == ""
        assert g.score == 0
        assert g.reason == ""

    def test_mic_snapshot_update(self) -> None:
        s = UIState()
        s.mic.monitoring = True
        s.mic.rms = 0.05
        s.mic.device_name = "Realtek"
        assert s.mic.monitoring is True
        assert s.mic.rms == 0.05
        assert s.mic.device_name == "Realtek"

    def test_mic_snapshot_defaults(self) -> None:
        m = MicSnapshot()
        assert m.monitoring is False
        assert m.rms == 0.0
        assert m.device_name == ""

    def test_message_timestamp_set(self) -> None:
        import time
        before = time.time()
        msg = ChatMessage(role=MessageRole.USER, text="hi")
        after = time.time()
        assert before <= msg.timestamp <= after
