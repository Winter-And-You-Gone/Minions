"""Command-palette 风格 TUI 渲染器 — Home 面板 · 输入行 · 补全区 · 底部状态栏。"""

from __future__ import annotations

from itertools import zip_longest

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
    """主面板：LOGO（左） + Runtime 信息（右），然后聊天消息，最后提示/健康。"""
    try:
        frags: list[tuple[str, str]] = []

        # ── 两栏布局：LOGO 左侧 + Runtime 右侧 ──
        llm_label = state.llm_model or state.llm.model or "mock"
        wake_text = (
            f"active {int(state.wake_remaining_seconds)}s"
            if state.wake_active
            else "inactive"
        )

        left_lines = [line for _, line in LOGO_LINES]
        right_lines = [
            f"✦ {state.app_name}  {state.version_text or ''}",
            f"Welcome back, {state.assistant_name}!",
            "",
            f"ASR:   {state.asr_engine}",
            f"Judge: {state.judge_model} ({state.judge_provider})",
            f"LLM:   {llm_label}",
            f"Mode:  {state.conversation_mode}",
            f"Wake:  {wake_text}",
        ]

        # 麦克风 VU（仅在监测时显示）
        if state.mic.monitoring:
            try:
                from voice_agent.cli.formatters import vu_bar
                bar = vu_bar(state.mic.rms, width=10)
                right_lines.append(f"Mic:   {bar}  {state.mic.rms:.4f}")
            except Exception:
                pass

        for left, right in zip_longest(left_lines, right_lines, fillvalue=""):
            frags.append(("bold fg:#ff6eb4", left.ljust(22)))
            frags.append(("white", right + "\n"))

        frags.append(("ansibrightblack", f"  {'─' * 52}\n"))

        # ── 聊天消息（仅用户/AI，最多 4 条） ──
        if state.messages:
            visible_chat = [m for m in state.visible_messages if m.role != MessageRole.SYSTEM]
            if visible_chat:
                frags.append(("bold underline", "  Chat\n"))
                if state.hidden_message_count > 0:
                    frags.append(("ansibrightblack", f"  … 已折叠 {state.hidden_message_count} 条更早消息\n"))
                for msg in visible_chat[-4:]:
                    try:
                        if msg.role == MessageRole.USER:
                            frags.append(("bold cyan", f"  你：{msg.text}\n"))
                        elif msg.role == MessageRole.ASSISTANT:
                            prefix = f"  {state.assistant_name}：" if state.assistant_name else "  AI："
                            frags.append(("green", f"{prefix}{msg.text}\n"))
                    except Exception:
                        frags.append(("ansiyellow", "  • <message error>\n"))

        # ── 提示（仅在没有消息时显示） ──
        if not state.messages:
            frags.append(("", "\n"))
            frags.append(("bold underline", "  Tips\n"))
            tips = state.tips_lines if hasattr(state, "tips_lines") and state.tips_lines else _DEFAULT_TIPS
            for tip in tips:
                frags.append(("ansibrightblack", f"  · {tip}\n"))

            # 健康检查
            if state.health_items:
                frags.append(("", "\n"))
                frags.append(("bold underline", "  Health\n"))
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
    """兼容旧引用：转发到新的统一面板。"""
    return format_command_panel(state)


def format_command_panel(state: UIState) -> list[tuple[str, str]]:
    """统一命令面板：根据 command_panel_mode 分发。"""
    if state.command_panel_mode == "help":
        return format_help_panel(state)
    if state.command_panel_mode == "completion":
        return _format_completion_panel_inner(state)
    if state.command_panel_mode == "output":
        return format_output_panel(state)
    # blank
    return _blank_panel(state)


def _blank_panel(state: UIState) -> list[tuple[str, str]]:
    rows = state.command_panel_reserved_rows
    frags: list[tuple[str, str]] = []
    for _ in range(rows):
        frags.append(("", " " * 80 + "\n"))
    return frags


def _format_completion_panel_inner(state: UIState) -> list[tuple[str, str]]:
    """补全模式：筛选后的命令列表。"""
    try:
        frags: list[tuple[str, str]] = []
        items = state.completion_items
        selected = state.completion_selected_index
        rows = state.command_panel_reserved_rows

        if not items or not state.completion_visible:
            return _blank_panel(state)

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


def format_help_panel(state: UIState) -> list[tuple[str, str]]:
    """帮助模式：命令浏览器，类似 Claude Code 风格。"""
    try:
        rows = state.command_panel_reserved_rows
        frags: list[tuple[str, str]] = []

        # ── Tab 栏 ──
        tabs = [
            ("bold underline" if state.help_tab == "Minions" else "bold", " Minions "),
            ("ansibrightblack", "│"),
            ("bold underline" if state.help_tab == "general" else "bold", " general "),
            ("ansibrightblack", "│"),
            ("bold underline" if state.help_tab == "commands" else "bold", " commands "),
            ("ansibrightblack", "│"),
            ("bold underline" if state.help_tab == "custom-commands" else "bold", " custom-commands "),
        ]
        for style, text in tabs:
            frags.append((style, text))
        frags.append(("", "\n"))

        # ── 标题 ──
        title = state.command_panel_title or "Browse default commands"
        frags.append(("bold cyan", f"  {title}\n"))
        frags.append(("", "\n"))

        # ── 命令列表 ──
        items = state.help_items
        selected = state.command_panel_selected_index
        max_display = max(0, rows - 5)  # 留出 tab + title + 空行 + 底部提示

        for i in range(max_display):
            if i < len(items):
                item = items[i]
                cmd = item.get("command", "")
                desc = item.get("description", "")
                aliases = item.get("aliases", [])
                prefix = "↓ " if i == selected else "  "
                cmd_style = "bold cyan" if i == selected else "bold white"
                desc_style = "ansibrightblack" if i == selected else "ansibrightblack"
                alias_text = f"  (别名: {', '.join(aliases)})" if aliases else ""
                frags.append((cmd_style, f"{prefix}{cmd}{alias_text}\n"))
                frags.append((desc_style, f"     {desc}\n"))
            else:
                break

        fill = max(0, rows - max_display - len(items)) if len(items) < max_display else 0
        for _ in range(fill):
            frags.append(("", " " * 80 + "\n"))

        # ── 底部提示 ──
        frags.append(("ansibrightblack", "  Esc to cancel  ·  Enter to select command\n"))

        return frags
    except Exception as e:
        return [("red", f"Help panel render error: {e}\n")]


# ── Output 面板 ──────────────────────────────────────────────────────────

def format_output_panel(state: UIState) -> list[tuple[str, str]]:
    """命令输出面板：显示 /status /debug /name 等命令的结果。"""
    try:
        rows = state.command_panel_reserved_rows
        frags: list[tuple[str, str]] = []

        title = state.command_output_title or "Output"
        frags.append(("bold cyan", f"  {title}\n"))
        frags.append(("ansibrightblack", f"  {'─' * 72}\n"))

        max_lines = max(0, rows - 3)
        lines = state.command_output_lines[:max_lines]

        for line in lines:
            frags.append(("white", f"  {line}\n"))

        if len(state.command_output_lines) > max_lines:
            hidden = len(state.command_output_lines) - max_lines
            frags.append(("ansibrightblack", f"  … 还有 {hidden} 行未显示\n"))

        used = 2 + len(lines)
        while used < rows - 1:
            frags.append(("", " " * 80 + "\n"))
            used += 1

        frags.append(("ansibrightblack", "  Esc to close\n"))
        return frags
    except Exception as e:
        return [("red", f"Output panel render error: {e}\n")]


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
