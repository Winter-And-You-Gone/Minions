"""测试 TUI 补全稳定性 — 非字符串 display、空补全、异常兜底。"""

from prompt_toolkit.document import Document

from voice_agent.cli.ui_state import UIState, CompletionItem
from voice_agent.cli.tui_renderer import format_completion_panel
from voice_agent.cli.command_completer import MinionsCommandCompleter


# ── CompletionItem 字符串化 ─────────────────────────────────────────────

def test_completion_panel_handles_non_string_display():
    """确保即使 display / display_meta 不是 str，补全面板也不会崩溃。"""
    state = UIState()
    state.completion_visible = True
    state.completion_items = [
        CompletionItem(
            text="/exit",
            display="<object>",
            display_meta="exit minions",
        )
    ]
    frags = format_completion_panel(state)
    assert frags


def test_completion_panel_handles_empty_display():
    """display 为空字符串时用 text 兜底。"""
    state = UIState()
    state.completion_visible = True
    state.completion_items = [
        CompletionItem(
            text="/help",
            display="",
            display_meta="",
        )
    ]
    frags = format_completion_panel(state)
    assert frags
    text = "".join(t for _, t in frags)
    assert "/help" in text


def test_completion_panel_handles_none_text():
    """None text 时 str(None) 不会崩溃。"""
    state = UIState()
    state.completion_visible = True
    state.completion_items = [
        CompletionItem(
            text="None",
            display="None",
            display_meta="",
        )
    ]
    frags = format_completion_panel(state)
    assert frags


# ── MinionsCommandCompleter 不抛异常 ─────────────────────────────────────

def test_completer_slash_no_crash():
    c = MinionsCommandCompleter()
    comps = list(c.get_completions(Document("/", cursor_position=1), None))
    assert comps


def test_completer_normal_text_no_crash():
    c = MinionsCommandCompleter()
    comps = list(c.get_completions(Document("hello", cursor_position=5), None))
    assert comps == []


def test_completer_partial_command_no_crash():
    c = MinionsCommandCompleter()
    comps = list(c.get_completions(Document("/st", cursor_position=3), None))
    assert any("/status" in str(co.text) for co in comps)


def test_completer_multiple_spaces_no_crash():
    c = MinionsCommandCompleter()
    comps = list(c.get_completions(Document("/  ", cursor_position=3), None))
    # should not crash
    assert comps is not None


def test_completer_empty_string_no_crash():
    c = MinionsCommandCompleter()
    comps = list(c.get_completions(Document("", cursor_position=0), None))
    assert comps == []


def test_completer_only_slash_no_crash():
    c = MinionsCommandCompleter()
    comps = list(c.get_completions(Document("/", cursor_position=1), None))
    assert comps
    texts = [co.text for co in comps]
    assert "/help" in texts
    assert "/exit" in texts


def test_completer_subcommand_no_crash():
    c = MinionsCommandCompleter()
    comps = list(c.get_completions(Document("/name ", cursor_position=6), None))
    texts = [co.text for co in comps]
    assert "set" in texts
    assert "alias" in texts
    assert "save" in texts
