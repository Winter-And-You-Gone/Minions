"""测试 TUI 渲染器 — home panel / input / completion / footer。"""

from voice_agent.cli.ui_state import UIState, CompletionItem
from voice_agent.cli.tui_renderer import (
    LOGO_LINES,
    format_command_panel,
    format_completion_panel,
    format_footer_bar,
    format_home_panel,
    format_input_prompt,
)


# ── home panel ────────────────────────────────────────────────────────────

def test_home_panel_returns_fragments():
    state = UIState()
    result = format_home_panel(state)
    assert result
    assert isinstance(result, list)
    assert all(isinstance(f, tuple) and len(f) == 2 for f in result)


def test_home_panel_contains_logo():
    state = UIState()
    result = format_home_panel(state)
    text = "".join(t for _, t in result)
    # LOGO lines should contain box-drawing chars
    assert "╭" in text or "◕" in text or "∞" in text


def test_home_panel_contains_welcome():
    state = UIState()
    state.app_name = "Minions"
    state.assistant_name = "西瓜"
    result = format_home_panel(state)
    text = "".join(t for _, t in result)
    assert "Minions" in text
    assert "西瓜" in text
    assert "Welcome back" in text


def test_home_panel_shows_chat_messages():
    state = UIState()
    state.assistant_name = "西瓜"
    state.add_user_message("你好")
    state.add_assistant_message("我在呢")
    result = format_home_panel(state)
    text = "".join(t for _, t in result)
    assert "你" in text
    assert "西瓜" in text
    assert "你好" in text
    assert "我在呢" in text


def test_home_panel_shows_system_messages():
    state = UIState()
    state.add_system_message("系统通知")
    result = format_home_panel(state)
    text = "".join(t for _, t in result)
    assert "系统通知" in text


def test_home_panel_shows_runtime_info():
    state = UIState()
    state.asr_engine = "sherpa-onnx"
    state.judge_model = "qwen3.5:4b"
    state.judge_provider = "local"
    result = format_home_panel(state)
    text = "".join(t for _, t in result)
    assert "sherpa-onnx" in text
    assert "qwen3.5" in text


# ── input prompt ──────────────────────────────────────────────────────────

def test_input_prompt_default():
    state = UIState()
    result = format_input_prompt(state)
    text = "".join(t for _, t in result)
    assert ">" in text


def test_input_prompt_paused():
    state = UIState()
    state.paused = True
    result = format_input_prompt(state)
    text = "".join(t for _, t in result)
    assert "⏸" in text or "pause" in text.lower() or "暂停" in text


# ── completion panel ──────────────────────────────────────────────────────

def test_completion_panel_empty():
    state = UIState()
    result = format_completion_panel(state)
    assert len(result) == state.command_panel_reserved_rows
    # All lines should be blank (whitespace)
    text = "".join(t for _, t in result)
    assert text.strip() == ""


def test_completion_panel_shows_items():
    state = UIState()
    state.command_panel_mode = "completion"
    state.completion_visible = True
    state.completion_items = [
        CompletionItem(text="/help", display="/help", display_meta="显示帮助"),
        CompletionItem(text="/status", display="/status", display_meta="查看状态"),
    ]
    result = format_completion_panel(state)
    text = "".join(t for _, t in result)
    assert "/help" in text
    assert "/status" in text
    assert "显示帮助" in text


def test_completion_panel_highlights_selected():
    state = UIState()
    state.command_panel_mode = "completion"
    state.completion_visible = True
    state.completion_selected_index = 0
    state.completion_items = [
        CompletionItem(text="/help", display="/help", display_meta="帮助"),
        CompletionItem(text="/status", display="/status", display_meta="状态"),
    ]
    result = format_completion_panel(state)
    # First item should have the "▸" marker
    assert any("▸" in t for _, t in result)


# ── command panel (blank / completion / help) ─────────────────────────────

def test_command_panel_blank_renders_rows():
    state = UIState()
    state.command_panel_mode = "blank"
    state.command_panel_reserved_rows = 14
    frags = format_command_panel(state)
    assert frags
    assert len(frags) == state.command_panel_reserved_rows


def test_command_panel_completion_renders_items():
    state = UIState()
    state.command_panel_mode = "completion"
    state.completion_visible = True
    state.completion_items = [
        CompletionItem(text="/exit", display="/exit", display_meta="退出 Minions")
    ]
    frags = format_command_panel(state)
    assert frags


def test_command_panel_help_renders_items():
    state = UIState()
    state.command_panel_mode = "help"
    state.help_items = [
        {"command": "/exit", "description": "退出 Minions", "usage": "/exit", "aliases": []}
    ]
    frags = format_command_panel(state)
    assert frags


# ── footer bar ────────────────────────────────────────────────────────────

def test_footer_bar_returns_fragments():
    state = UIState()
    result = format_footer_bar(state)
    assert result
    assert isinstance(result, list)


def test_footer_bar_shows_app_name():
    state = UIState()
    state.app_name = "Minions"
    result = format_footer_bar(state)
    text = "".join(t for _, t in result)
    assert "Minions" in text


def test_footer_bar_shows_paused():
    state = UIState()
    state.paused = True
    result = format_footer_bar(state)
    text = "".join(t for _, t in result)
    assert "PAUSED" in text or "⏸" in text


# ── LOGO ──────────────────────────────────────────────────────────────────

def test_logo_lines_defined():
    assert LOGO_LINES
    assert len(LOGO_LINES) >= 5
    assert all(isinstance(l, tuple) and len(l) == 2 for l in LOGO_LINES)
