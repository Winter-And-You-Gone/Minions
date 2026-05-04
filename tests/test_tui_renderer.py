"""测试 TUI 渲染器基本功能。"""

from voice_agent.cli.ui_state import UIState
from voice_agent.cli.tui_renderer import (
    format_chat_panel,
    format_input_prompt,
    format_side_panel,
    format_top_bar,
)


def test_top_bar_returns_fragments():
    state = UIState()
    result = format_top_bar(state)
    assert result
    assert isinstance(result, list)
    assert all(isinstance(f, tuple) and len(f) == 2 for f in result)


def test_chat_panel_returns_fragments():
    state = UIState()
    result = format_chat_panel(state)
    assert result
    assert isinstance(result, list)


def test_chat_panel_shows_user_and_assistant():
    state = UIState()
    state.assistant_name = "琉璃川"
    state.add_user_message("你好")
    state.add_assistant_message("我在呢")
    result = format_chat_panel(state)
    text = "".join(t for _, t in result)
    assert "你" in text
    assert "琉璃川" in text


def test_side_panel_returns_fragments():
    state = UIState()
    result = format_side_panel(state)
    assert result
    assert isinstance(result, list)


def test_side_panel_shows_status():
    state = UIState()
    state.judge_provider = "local"
    state.judge_model = "qwen3.5:4b"
    state.asr_engine = "sherpa-onnx"
    result = format_side_panel(state)
    text = "".join(t for _, t in result)
    assert "Status" in text
    assert "Runtime" in text


def test_input_prompt_default():
    state = UIState()
    state.assistant_name = "琉璃川"
    result = format_input_prompt(state)
    text = "".join(t for _, t in result)
    assert "琉璃川" in text


def test_input_prompt_paused():
    state = UIState()
    state.paused = True
    result = format_input_prompt(state)
    text = "".join(t for _, t in result)
    assert "暂停" in text
