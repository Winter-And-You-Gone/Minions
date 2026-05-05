"""Command-palette 风格 TUI 渲染器 — Home 面板 · 输入行 · 补全区 · 底部状态栏。"""

from __future__ import annotations

from wcwidth import wcwidth, wcswidth

from voice_agent.cli.ui_state import UIState, MessageRole

# ── LOGO：小黄人风格 ───────────────────────────────────────────────────────

MINION_LOGO: list[str] = [
    "     .-=======-.",
    "    /   .---.   \\",
    "   |---/ .-. \\---|",
    "   |  | | @ | |  |",
    "   |  | |   | |  |",
    "   |---\\ '-' /---|",
    "   | \\    _    / |",
    "   |  \\ .===. /  |",
    "    \\___|___|___/",
]

_DEFAULT_TIPS = (
    "/help  查看所有命令",
    "Tab / Shift+Tab  浏览补全",
    "直接输入文字与 AI 对话",
    "Ctrl+C  安全退出",
)


# ── 终端显示宽度工具 ───────────────────────────────────────────────────────

_LEFT_WIDTH = 64
_GAP = "  │  "


def _display_width(text: str) -> int:
    width = wcswidth(text)
    if width < 0:
        width = sum(max(0, wcwidth(ch)) for ch in text)
    return width


def _trim_to_width(text: str, width: int, ellipsis: str = "…") -> str:
    if width <= 0:
        return ""
    if _display_width(text) <= width:
        return text
    ellipsis_width = max(1, _display_width(ellipsis))
    target = max(0, width - ellipsis_width)
    result: list[str] = []
    current = 0
    for ch in text:
        ch_width = wcwidth(ch)
        if ch_width < 0:
            ch_width = 0
        if current + ch_width > target:
            break
        result.append(ch)
        current += ch_width
    return "".join(result) + ellipsis


def _pad_to_width(text: str, width: int) -> str:
    text = _trim_to_width(text, width)
    pad = max(0, width - _display_width(text))
    return text + (" " * pad)


def _wrap_display_lines(text: str, width: int) -> list[str]:
    """将文本按显示宽度换行，不限制行数。"""
    if width <= 0:
        return []

    lines: list[str] = []
    rest = text

    while rest:
        if _display_width(rest) <= width:
            lines.append(_pad_to_width(rest, width))
            break

        result: list[str] = []
        current = 0
        cut_index = 0
        for idx, ch in enumerate(rest):
            ch_width = wcwidth(ch)
            if ch_width < 0:
                ch_width = 0
            if current + ch_width > width:
                break
            result.append(ch)
            current += ch_width
            cut_index = idx + 1

        if cut_index <= 0:
            cut_index = 1
            result = [rest[0] if rest else ""]

        lines.append(_pad_to_width("".join(result), width))
        rest = rest[cut_index:]

    if not lines:
        lines.append(" " * width)

    return lines


MAX_HOME_ROWS = 18
CHAT_VISIBLE_ROWS = 16


def _wrap_chat_message(
    state: UIState,
    msg_text: str,
    role: MessageRole,
    left_width: int,
) -> list[list[tuple[str, str]]]:
    """将单条消息渲染为带前缀和颜色的行列表。

    第一行有前缀（你：/ 西瓜：），后续行用空格缩进对齐。
    返回每行是一个 (style, text) 片段列表。
    """
    if role == MessageRole.USER:
        prefix = "你："
        prefix_style = "bold cyan"
        content_style = "white"
    elif role == MessageRole.ASSISTANT:
        name = state.assistant_name or "AI"
        prefix = f"{name}："
        prefix_style = "bold yellow"
        content_style = "ansibrightwhite"
    else:
        return []

    prefix_width = _display_width(prefix)
    content_width = max(1, left_width - prefix_width)
    content_lines = _wrap_display_lines(msg_text, content_width)

    result: list[list[tuple[str, str]]] = []
    for i, line in enumerate(content_lines):
        row: list[tuple[str, str]] = []
        if i == 0:
            row.append((prefix_style, prefix))
        else:
            row.append(("", " " * prefix_width))
        # line is already padded to content_width; row total = prefix_width + content_width
        row.append((content_style, line.rstrip()))
        # pad remaining to full left_width
        line_dw = _display_width(line.rstrip())
        pad = left_width - prefix_width - line_dw
        if pad > 0:
            row.append(("", " " * pad))
        result.append(row)
    return result


def format_home_panel(state: UIState) -> list[tuple[str, str]]:
    """主面板：左 Chat，右 LOGO + Runtime 信息 + 提示/健康。"""
    try:
        frags: list[tuple[str, str]] = []
        LEFT = _LEFT_WIDTH

        # ── 左侧：Chat（彩色，按角色区分） ──
        chat_rows: list[list[tuple[str, str]]] = []

        # Header
        chat_rows.append([("bold", "Chat")])
        chat_rows.append([("ansibrightblack", "─" * LEFT)])

        visible_chat = [
            m for m in state.visible_messages
            if m.role in (MessageRole.USER, MessageRole.ASSISTANT)
        ]

        if visible_chat:
            if state.hidden_message_count > 0:
                chat_rows.append([
                    ("ansibrightblack", _pad_to_width(
                        f"… 已折叠 {state.hidden_message_count} 条更早内容", LEFT,
                    )),
                ])
            for msg in visible_chat[-8:]:
                try:
                    msg_rows = _wrap_chat_message(state, msg.text, msg.role, LEFT)
                    chat_rows.extend(msg_rows)
                except Exception:
                    chat_rows.append([("red", _pad_to_width("• <message error>", LEFT))])
        else:
            chat_rows.append([("ansibrightblack", _pad_to_width("暂无对话。直接说话，或输入文字。", LEFT))])
            chat_rows.append([("ansibrightblack", _pad_to_width("输入 /help 查看命令。", LEFT))])
            # Tips
            tips = state.tips_lines if state.tips_lines else _DEFAULT_TIPS
            for tip in tips:
                chat_rows.append([("ansibrightblack", _pad_to_width(f"· {tip}", LEFT))])
            # Health
            if state.health_items:
                chat_rows.append([("", "")])
                for item in state.health_items:
                    try:
                        ok = getattr(item, "ok", False)
                        name = getattr(item, "name", "?")
                        level = getattr(item, "level", "info")
                        mark = "✓" if ok else ("✗" if level == "error" else "!")
                        chat_rows.append([("ansibrightblack", _pad_to_width(f"  {mark} {name}", LEFT))])
                    except Exception:
                        chat_rows.append([("ansibrightblack", _pad_to_width("  ? unknown", LEFT))])

        if state.error_line:
            chat_rows.append([("", "")])
            chat_rows.append([("red", _pad_to_width(f"✗ {state.error_line}", LEFT))])

        # 按行数裁剪，不截断单条消息
        if len(chat_rows) > CHAT_VISIBLE_ROWS:
            hidden = len(chat_rows) - CHAT_VISIBLE_ROWS
            chat_rows = [
                [("ansibrightblack", _pad_to_width(f"… 已折叠 {hidden} 行更早内容", LEFT))],
            ] + chat_rows[-(CHAT_VISIBLE_ROWS - 1):]

        # ── 右侧：LOGO + Runtime ──
        right_rows: list[list[tuple[str, str]]] = []

        llm_label = state.llm_model or state.llm.model or "mock"

        # LOGO
        for line in MINION_LOGO:
            right_rows.append([("yellow", line)])

        right_rows.append([])

        # Runtime 信息 — Text / Voice / Wake 三状态
        right_rows.append([("bold cyan", f"✦ {state.app_name}  {state.version_text or ''}")])
        right_rows.append([("bold white", f"Welcome back, {state.assistant_name}!")])
        right_rows.append([])

        # Text: always ready
        right_rows.append([("cyan", "Text:  "), ("white", "ready")])

        # Voice
        if state.voice_listening:
            voice_text = "listening (/sleep)"
        else:
            voice_text = "off (/listen)"
        right_rows.append([("blue", "Voice: "), ("white", voice_text)])

        # Wake
        if state.assistant_awake:
            wake_text = "awake (/sleep)"
        else:
            wake_text = "asleep (/wakeup)"
        right_rows.append([("yellow", "Wake:  "), ("white", wake_text)])

        right_rows.append([])
        right_rows.append([("cyan", "ASR:   "), ("white", state.asr_engine)])
        right_rows.append([("magenta", "Judge: "), ("white", f"{state.judge_model} ({state.judge_provider})")])
        right_rows.append([("green", "LLM:   "), ("white", llm_label)])
        right_rows.append([("yellow", "Mode:  "), ("white", state.conversation_mode)])

        if state.current_path:
            right_rows.append([("ansibrightblack", "Path:  "), ("white", state.current_path)])

        if state.mic.monitoring:
            try:
                from voice_agent.cli.formatters import vu_bar
                bar = vu_bar(state.mic.rms, width=10)
                right_rows.append([("ansimagenta", "Mic:   "), ("white", f"{bar}  {state.mic.rms:.4f}")])
            except Exception:
                pass

        # ── 合并左右（限制总行数） ──
        max_lines = min(MAX_HOME_ROWS, max(len(chat_rows), len(right_rows)))

        for i in range(max_lines):
            # 左侧
            if i < len(chat_rows):
                for style, text in chat_rows[i]:
                    frags.append((style, text))
            else:
                frags.append(("", " " * LEFT))

            # 间隔
            frags.append(("ansibrightblack", _GAP))

            # 右侧
            if i < len(right_rows):
                row = right_rows[i]
                for style, text in row:
                    frags.append((style, text))
            else:
                frags.append(("", ""))

            frags.append(("", "\n"))

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
    """补全模式：带滚动窗口的筛选后的命令列表。"""
    try:
        frags: list[tuple[str, str]] = []
        items = state.completion_items
        selected = state.completion_selected_index
        rows = state.command_panel_reserved_rows
        offset = state.command_panel_scroll_offset

        if not items or not state.completion_visible:
            return _blank_panel(state)

        visible_items = items[offset: offset + rows]

        for i in range(rows):
            if i < len(visible_items):
                item = visible_items[i]
                absolute_index = offset + i
                prefix = "▸ " if absolute_index == selected else "  "
                style = "bold cyan" if absolute_index == selected else "ansibrightblack"
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
    """帮助模式：带滚动窗口的命令浏览器。"""
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

        # ── 命令列表（每 item 占 2 行） ──
        header_lines = 3  # tab bar + title + 空行
        footer_lines = 1  # 底部提示
        usable_lines = max(1, rows - header_lines - footer_lines)
        visible_item_count = max(1, usable_lines // 2)

        items = state.help_items
        selected = state.command_panel_selected_index
        offset = state.command_panel_scroll_offset

        visible_items = items[offset: offset + visible_item_count]

        for i, item in enumerate(visible_items):
            absolute_index = offset + i
            cmd = item.get("command", "")
            desc = item.get("description", "")
            aliases = item.get("aliases", [])
            prefix = "↓ " if absolute_index == selected else "  "
            cmd_style = "bold cyan" if absolute_index == selected else "bold white"
            alias_text = f"  (别名: {', '.join(aliases)})" if aliases else ""
            frags.append((cmd_style, f"{prefix}{cmd}{alias_text}\n"))
            frags.append(("ansibrightblack", f"     {desc}\n"))

        # 填充剩余行
        used_lines = header_lines + 1 + len(visible_items) * 2 + footer_lines
        fill = max(0, rows - used_lines)
        for _ in range(fill):
            frags.append(("", " " * 80 + "\n"))

        # ── 底部提示 ──
        frags.append(("ansibrightblack", "  ↑↓ move  ·  PgUp/PgDn page  ·  Enter run  ·  Esc close\n"))

        return frags
    except Exception as e:
        return [("red", f"Help panel render error: {e}\n")]


# ── Output 面板 ──────────────────────────────────────────────────────────

def format_output_panel(state: UIState) -> list[tuple[str, str]]:
    """命令输出面板：带滚动窗口的 /status /debug /name 等命令结果。"""
    try:
        rows = state.command_panel_reserved_rows
        frags: list[tuple[str, str]] = []

        title = state.command_output_title or "Output"
        frags.append(("bold cyan", f"  {title}\n"))
        frags.append(("ansibrightblack", f"  {'─' * 72}\n"))

        max_lines = max(0, rows - 3)
        offset = state.command_panel_scroll_offset

        if offset > 0:
            frags.append(("ansibrightblack", "  (可滚动内容)\n"))
            max_lines -= 1

        lines = state.command_output_lines[offset: offset + max_lines]

        for line in lines:
            frags.append(("white", f"  {line}\n"))

        if len(state.command_output_lines) > offset + max_lines:
            hidden = len(state.command_output_lines) - offset - max_lines
            frags.append(("ansibrightblack", f"  … 还有 {hidden} 行未显示\n"))

        used = 2 + len(lines) + (1 if offset > 0 else 0)
        while used < rows - 1:
            frags.append(("", " " * 80 + "\n"))
            used += 1

        frags.append(("ansibrightblack", "  ↑↓ scroll  ·  PgUp/PgDn page  ·  Esc close\n"))
        return frags
    except Exception as e:
        return [("red", f"Output panel render error: {e}\n")]


# ── 底部状态栏 ────────────────────────────────────────────────────────────

def format_footer_bar(state: UIState) -> list[tuple[str, str]]:
    """底部状态栏：左侧 state 提示，右侧快捷键提示。"""
    try:
        mode = state.command_panel_mode
        if mode == "completion":
            left_hint = "↑↓ move  ·  Enter run  ·  Esc close"
        elif mode == "help":
            left_hint = "↑↓ move  ·  PgUp/PgDn page  ·  Enter run  ·  Esc close"
        elif mode == "output":
            left_hint = "↑↓ scroll  ·  PgUp/PgDn page  ·  Esc close"
        else:
            # 三状态提示
            parts = ["text ready"]
            parts.append("listening" if state.voice_listening else "voice off")
            parts.append("awake" if state.assistant_awake else "asleep")
            left_hint = " · ".join(parts)

        if state.paused:
            left_hint = "⏸  PAUSED"
        elif mode != "blank":
            left_hint = f"{state.app_name} | {left_hint}"

        if state.voice_listening or state.assistant_awake:
            right = "/sleep /help"
        else:
            right = "/wakeup /listen /help"
        padding = max(0, 80 - len(left_hint) - len(right))
        line = left_hint + " " * padding + right

        return [("reverse", line)]
    except Exception as e:
        return [("reverse", f"Status error: {e}")]
