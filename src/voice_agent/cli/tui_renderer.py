"""Command-palette 风格 TUI 渲染器 — Home 面板 · 输入行 · 补全区 · 底部状态栏。"""

from __future__ import annotations

from voice_agent.cli.ui_state import UIState, MessageRole

# ── LOGO ──────────────────────────────────────────────────────────────────
# 像素风吉祥物, 15 字符宽, 7 行高, 粉红色

LOGO_LINES: list[tuple[str, str]] = [
    ("bold fg:#ff6eb4", "  ╭─────────╮  "),
    ("bold fg:#ff6eb4", "  │  ╱◕‿◕╲  │  "),
    ("bold fg:#ff6eb4", "  │ ╱     ╲ │  "),
    ("bold fg:#ff6eb4", "  │ │ ∞ ∞ │ │  "),
    ("bold fg:#ff6eb4", "  │ ╲     ╱ │  "),
    ("bold fg:#ff6eb4", "  │  ╲___╱  │  "),
    ("bold fg:#ff6eb4", "  ╰─────────╯  "),
]

_DEFAULT_TIPS = (
    "/help  查看所有命令",
    "Tab / Shift+Tab  浏览补全",
    "直接输入文字与 AI 对话",
    "Ctrl+C  安全退出",
)


# ── Home 面板 ─────────────────────────────────────────────────────────────

def format_home_panel(state: UIState) -> list[tuple[str, str]]:
    """主面板：LOGO → 欢迎 → 运行时 → 聊天消息 → 提示/健康。"""
    try:
        frags: list[tuple[str, str]] = []

        # LOGO
        for style, line in LOGO_LINES:
            frags.append((style, line + "\n"))

        # 欢迎标题
        title = f"  ✦ {state.app_name}"
        if state.version_text:
            title += f"  {state.version_text}"
        frags.append(("bold cyan", title + "\n"))
        frags.append(("bold white", f"  Welcome back, {state.assistant_name}!\n"))
        frags.append(("ansibrightblack", f"  {'─' * 35}\n"))

        # 运行时信息
        llm_label = state.llm_model or state.llm.model or "mock"
        frags.append(("ansibrightblack", f"  ASR:   {state.asr_engine}\n"))
        frags.append(("ansibrightblack", f"  Judge: {state.judge_model} ({state.judge_provider})\n"))
        frags.append(("ansibrightblack", f"  LLM:   {llm_label}\n"))
        frags.append(("ansibrightblack", f"  Mode:  {state.conversation_mode}\n"))

        # 麦克风 VU (仅在监测时显示)
        if state.mic.monitoring:
            try:
                from voice_agent.cli.formatters import vu_bar
                bar = vu_bar(state.mic.rms, width=15)
                frags.append(("ansimagenta", f"  Mic:   {bar}  {state.mic.rms:.4f}\n"))
            except Exception:
                pass

        # 聊天消息
        if state.messages:
            frags.append(("", "\n"))
            frags.append(("bold underline", "  ── Chat ──\n"))
            if state.hidden_message_count > 0:
                frags.append(("ansibrightblack", f"  … 已折叠 {state.hidden_message_count} 条更早消息\n"))
            for msg in state.visible_messages:
                try:
                    if msg.role == MessageRole.USER:
                        frags.append(("bold cyan", f"  你：{msg.text}\n"))
                    elif msg.role == MessageRole.ASSISTANT:
                        prefix = f"  {state.assistant_name}：" if state.assistant_name else "  AI："
                        frags.append(("green", f"{prefix}{msg.text}\n"))
                    elif msg.role == MessageRole.SYSTEM:
                        frags.append(("ansiyellow", f"  • {msg.text}\n"))
                except Exception:
                    frags.append(("ansiyellow", "  • <message error>\n"))

        # 提示（仅在没有消息时显示，节省空间）
        if not state.messages:
            frags.append(("", "\n"))
            frags.append(("bold underline", "  ── Tips ──\n"))
            tips = state.tips_lines if hasattr(state, "tips_lines") and state.tips_lines else _DEFAULT_TIPS
            for tip in tips:
                frags.append(("ansibrightblack", f"  · {tip}\n"))

            # 健康检查
            if state.health_items:
                frags.append(("", "\n"))
                frags.append(("bold underline", "  ── Health ──\n"))
                for item in state.health_items:
                    try:
                        ok = getattr(item, "ok", False)
                        name = getattr(item, "name", "?")
                        level = getattr(item, "level", "info")
                        mark = "✓" if ok else ("✗" if level == "error" else "!")
                        style = "green" if ok else ("red" if level == "error" else "yellow")
                        frags.append((style, f"  {mark} {name}\n"))
                    except Exception:
                        frags.append(("ansibrightblack", "  ? unknown\n"))

        # 错误信息
        if state.error_line:
            frags.append(("red", f"  ✗ {state.error_line}\n"))

        return frags
    except Exception as e:
        return [("red", f"Home panel render error: {e}\n")]


# ── 输入提示 ──────────────────────────────────────────────────────────────

def format_input_prompt(state: UIState) -> list[tuple[str, str]]:
    """底部输入提示符，统一为 ' > '。"""
    try:
        if state.paused:
            return [("red", " ⏸ ")]
        return [("bold cyan", " > ")]
    except Exception:
        return [("", " > ")]


# ── 补全面板 ──────────────────────────────────────────────────────────────

def format_completion_panel(state: UIState) -> list[tuple[str, str]]:
    """固定 6 行补全面板。没有补全时保持空白。"""
    try:
        frags: list[tuple[str, str]] = []
        items = state.completion_items
        selected = state.completion_selected_index
        rows = state.completion_reserved_rows

        if not items or not state.completion_visible:
            for _ in range(rows):
                frags.append(("", " " * 80 + "\n"))
            return frags

        for i in range(rows):
            if i < len(items):
                item = items[i]
                prefix = "▸ " if i == selected else "  "
                style = "bold cyan" if i == selected else "ansibrightblack"
                display = str(item.display or item.text or "")
                meta = str(item.display_meta or "")
                line = f"{prefix}{display:<15s} — {meta}"
                frags.append((style, line.ljust(80) + "\n"))
            else:
                frags.append(("", " " * 80 + "\n"))

        return frags
    except Exception as e:
        return [("red", f"Completion render error: {e}\n")]


# ── 底部状态栏 ────────────────────────────────────────────────────────────

def format_footer_bar(state: UIState) -> list[tuple[str, str]]:
    """底部状态栏：左侧模式/状态，右侧模型信息。"""
    try:
        left = state.footer_left or f"{state.app_name} | {state.conversation_mode}"
        right = state.footer_right or ""

        if state.paused:
            left = "⏸  PAUSED"

        padding = max(0, 80 - len(left) - len(right))
        line = left + " " * padding + right

        return [("reverse", line)]
    except Exception as e:
        return [("reverse", f"Status error: {e}")]
