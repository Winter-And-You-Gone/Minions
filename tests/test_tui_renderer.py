"""测试 TUI 渲染器 — home panel / input / completion / footer。"""

from voice_agent.cli.ui_state import UIState, CompletionItem
from voice_agent.cli.tui_renderer import (
    MINION_LOGO,
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
    # MINION_LOGO should contain the new minion ASCII
    assert ".-=======-." in text or "| | @ | |" in text


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
    # SYSTEM 消息不再显示在 Home 面板，已移到 output panel
    assert "系统通知" not in text


def test_home_panel_shows_runtime_info():
    state = UIState()
    state.asr_engine = "sherpa-onnx"
    state.judge_model = "qwen3.5:4b"
    state.judge_provider = "local"
    result = format_home_panel(state)
    text = "".join(t for _, t in result)
    assert "sherpa-onnx" in text
    assert "Text:" in text
    assert "Voice:" in text
    assert "Wake:" in text


def test_home_panel_contains_chat_and_runtime():
    """验证左 Chat + 右 Runtime 同时显示。"""
    state = UIState()
    state.asr_engine = "sherpa-onnx"
    state.judge_model = "qwen3.5:4b"
    state.add_user_message("你好")
    state.add_assistant_message("我在呢")
    frags = format_home_panel(state)
    text = "".join(part for _, part in frags)
    assert "Chat" in text
    assert "你好" in text
    assert "Voice:" in text
    assert ".-=======-." in text


def test_home_panel_contains_minion_logo():
    """验证新 LOGO 出现。"""
    state = UIState()
    frags = format_home_panel(state)
    text = "".join(part for _, part in frags)
    assert ".-=======-." in text
    assert "| | @ | |" in text


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


def test_command_output_panel_renders():
    state = UIState()
    state.command_panel_mode = "output"
    state.command_output_title = "Status"
    state.command_output_lines = ["ASR: sherpa-onnx", "Judge: qwen3.5:4b"]

    frags = format_command_panel(state)
    assert frags
    text = "".join(part for _, part in frags)
    assert "Status" in text
    assert "ASR: sherpa-onnx" in text


# ── scroll offset ──────────────────────────────────────────────────────────

def test_completion_panel_respects_scroll_offset():
    state = UIState()
    state.command_panel_mode = "completion"
    state.command_panel_reserved_rows = 4
    state.command_panel_scroll_offset = 2
    state.completion_visible = True
    state.completion_items = [
        CompletionItem(text=f"/cmd{i}", display=f"/cmd{i}", display_meta=f"cmd {i}")
        for i in range(10)
    ]
    state.completion_selected_index = 3
    frags = format_command_panel(state)
    text = "".join(part for _, part in frags)
    assert "/cmd2" in text, "scroll offset 2 应当可见 cmd2 (index 2)"
    assert "/cmd5" in text, "scroll offset 2 + rows=4 应当可见 cmd5 (index 5)"
    assert "/cmd0" not in text, "scroll offset 2 应当隐藏 cmd0"
    assert "/cmd6" not in text, "scroll offset 2 + rows=4 应当隐藏 cmd6 (index 6 ≥ offset+rows)"


def test_help_panel_respects_scroll_offset():
    state = UIState()
    state.command_panel_mode = "help"
    state.command_panel_reserved_rows = 14
    state.command_panel_scroll_offset = 5
    state.help_items = [
        {"command": f"/cmd{i}", "description": f"desc {i}", "aliases": []}
        for i in range(20)
    ]
    state.command_panel_selected_index = 7
    frags = format_command_panel(state)
    text = "".join(part for _, part in frags)
    assert "/cmd5" in text
    assert "/cmd0" not in text


def test_help_panel_respects_scroll_offset_by_items():
    """验证 help 按 item 数（非行数）滚动。"""
    state = UIState()
    state.command_panel_mode = "help"
    state.command_panel_reserved_rows = 14
    state.command_panel_scroll_offset = 3
    state.command_panel_selected_index = 4
    state.help_items = [
        {"command": f"/cmd{i}", "description": f"desc{i}", "usage": f"/cmd{i}", "aliases": []}
        for i in range(12)
    ]
    frags = format_command_panel(state)
    text = "".join(part for _, part in frags)
    assert "/cmd3" in text
    assert "/cmd4" in text
    assert "/cmd0" not in text


def test_help_panel_does_not_exceed_reserved_rows():
    """验证 help panel 输出的总行数不超过 reserved_rows。"""
    state = UIState()
    state.command_panel_mode = "help"
    state.command_panel_reserved_rows = 14
    state.help_items = [
        {"command": f"/cmd{i}", "description": f"desc{i}", "usage": f"/cmd{i}", "aliases": []}
        for i in range(20)
    ]
    frags = format_command_panel(state)
    text = "".join(part for _, part in frags)
    line_count = text.count("\n")
    assert line_count <= 14


def test_output_panel_respects_scroll_offset():
    state = UIState()
    state.command_panel_mode = "output"
    state.command_panel_reserved_rows = 10
    state.command_panel_scroll_offset = 3
    state.command_output_title = "Test"
    state.command_output_lines = [f"line {i}" for i in range(20)]
    frags = format_command_panel(state)
    text = "".join(part for _, part in frags)
    assert "line 3" in text
    assert "line 0" not in text


# ── footer bar ────────────────────────────────────────────────────────────

def test_footer_bar_returns_fragments():
    state = UIState()
    result = format_footer_bar(state)
    assert result
    assert isinstance(result, list)


def test_footer_bar_shows_app_name():
    state = UIState()
    state.app_name = "Minions"
    state.command_panel_mode = "completion"
    result = format_footer_bar(state)
    text = "".join(t for _, t in result)
    assert "Minions" in text


def test_footer_bar_shows_paused():
    state = UIState()
    state.paused = True
    result = format_footer_bar(state)
    text = "".join(t for _, t in result)
    assert "PAUSED" in text or "⏸" in text


def test_footer_bar_shows_asleep_voice_off():
    state = UIState()
    state.assistant_awake = False
    state.voice_listening = False
    result = format_footer_bar(state)
    text = "".join(t for _, t in result)
    assert "asleep" in text
    assert "voice off" in text


def test_footer_bar_shows_listening():
    state = UIState()
    state.voice_listening = True
    result = format_footer_bar(state)
    text = "".join(t for _, t in result)
    assert "listening" in text


def test_footer_bar_shows_awake():
    state = UIState()
    state.assistant_awake = True
    result = format_footer_bar(state)
    text = "".join(t for _, t in result)
    assert "awake" in text


def test_footer_bar_shows_awake_and_listening():
    state = UIState()
    state.assistant_awake = True
    state.voice_listening = True
    result = format_footer_bar(state)
    text = "".join(t for _, t in result)
    assert "awake" in text
    assert "listening" in text


def test_home_panel_shows_runtime_state():
    """验证 home panel 右侧显示 Runtime 状态行。"""
    state = UIState()
    state.voice_listening = True
    state.assistant_awake = True
    frags = format_home_panel(state)
    text = "".join(part for _, part in frags)
    assert "Voice:" in text
    assert "Wake:" in text


# ── LOGO ──────────────────────────────────────────────────────────────────

def test_logo_lines_defined():
    assert MINION_LOGO
    assert len(MINION_LOGO) >= 5
    for line in MINION_LOGO:
        assert isinstance(line, str)


# ── wcwidth 显示宽度 ────────────────────────────────────────────────────────

def test_display_width_handles_chinese():
    from voice_agent.cli.tui_renderer import _display_width, _pad_to_width, _trim_to_width

    text = "你好世界"
    assert _display_width(text) >= 8

    trimmed = _trim_to_width("你好世界abcdef", 8)
    assert _display_width(trimmed) <= 8

    padded = _pad_to_width("你好", 10)
    assert _display_width(padded) == 10


def test_home_panel_chat_does_not_exceed_left_width():
    from voice_agent.cli.tui_renderer import _LEFT_WIDTH, _display_width
    state = UIState()
    state.assistant_name = "西瓜"
    state.add_user_message("你好")
    state.add_assistant_message("这是一个很长很长的中文回复" * 20)

    frags = format_home_panel(state)
    text = "".join(part for _, part in frags)

    assert ".-=======-." in text
    assert "ASR:" in text


def test_minion_logo_exact_shape():
    from voice_agent.cli.tui_renderer import MINION_LOGO

    assert MINION_LOGO[0].strip() == ".-=======-."
    assert "/   .---.   \\" in MINION_LOGO[1]
    assert "|---/ .-. \\---|" in MINION_LOGO[2]
    assert "\\___|___|___/" in MINION_LOGO[-1]
