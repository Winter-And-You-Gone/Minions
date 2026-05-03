"""测试 render.py — Rich 面板渲染函数。"""

import io
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from voice_agent.cli.render import vu_bar, render_header, render_messages, render_decision, render_app
from voice_agent.cli.ui_state import UIState, GateView


def _render_to_str(renderable) -> str:
    """将 Rich renderable 渲染为纯文本。"""
    console = Console(file=io.StringIO(), width=80)
    console.print(renderable)
    return console.file.getvalue()


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
        bar = vu_bar(1.0, width=10)
        assert bar == "█" * 10

    def test_custom_width(self) -> None:
        bar = vu_bar(0.5, width=5)
        assert len(bar) == 5


class TestRenderHeader:
    def test_returns_panel(self) -> None:
        s = UIState()
        assert isinstance(render_header(s), Panel)

    def test_shows_mode(self) -> None:
        s = UIState(conversation_mode="active_chat")
        text = _render_to_str(render_header(s))
        assert "active_chat" in text


class TestRenderMessages:
    def test_returns_panel(self) -> None:
        assert isinstance(render_messages(UIState()), Panel)

    def test_empty_shows_waiting(self) -> None:
        text = _render_to_str(render_messages(UIState()))
        assert "等待" in text

    def test_user_message_shown(self) -> None:
        s = UIState()
        s.add_user_message("hello")
        text = _render_to_str(render_messages(s))
        assert "hello" in text
        assert "你" in text

    def test_assistant_message_shown(self) -> None:
        s = UIState()
        s.add_assistant_message("world")
        text = _render_to_str(render_messages(s))
        assert "world" in text
        assert "AI" in text

    def test_system_message_shown(self) -> None:
        s = UIState()
        s.add_system_message("info")
        text = _render_to_str(render_messages(s))
        assert "info" in text

    def test_hidden_count(self) -> None:
        s = UIState(max_visible_messages=2)
        for i in range(5):
            s.add_user_message(str(i))
        text = _render_to_str(render_messages(s))
        assert "折叠" in text

    def test_error_line_shown(self) -> None:
        s = UIState(error_line="broken")
        text = _render_to_str(render_messages(s))
        assert "broken" in text


class TestRenderDecision:
    def test_returns_panel(self) -> None:
        assert isinstance(render_decision(UIState()), Panel)

    def test_empty_shows_waiting(self) -> None:
        text = _render_to_str(render_decision(UIState()))
        assert "等待" in text

    def test_gate_action_shown(self) -> None:
        s = UIState()
        s.latest_gate = GateView(action="agent", score=90, reason="test reason")
        text = _render_to_str(render_decision(s))
        assert "agent" in text
        assert "90" in text
        assert "test reason" in text


class TestRenderApp:
    def test_returns_panel(self) -> None:
        assert isinstance(render_app(UIState()), Panel)

    def test_default_state(self) -> None:
        text = _render_to_str(render_app(UIState()))
        assert "Minions" in text
        assert "等待" in text
