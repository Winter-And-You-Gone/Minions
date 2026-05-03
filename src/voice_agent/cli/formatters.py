"""格式化函数 — 将 UIState 渲染为 prompt_toolkit formatted text。"""

import math

from voice_agent.cli.ui_state import UIState, MessageRole


def vu_bar(rms: float, width: int = 20) -> str:
    """将 RMS 音量转为 Unicode VU 条。"""
    if rms <= 0:
        return "░" * width
    db = 20 * math.log10(max(rms, 1e-6))
    norm = max(0.0, min(1.0, (db + 60) / 60))
    filled = int(norm * width)
    filled = max(0, min(filled, width))
    return "█" * filled + "░" * (width - filled)


def _mode_tag(state: UIState) -> str:
    tags = {
        "active_chat": "bold green",
        "passive_listening": "ansibrightblack",
        "cooldown": "yellow",
        "paused": "red",
    }
    return tags.get(state.conversation_mode, "")


def format_header(state: UIState) -> list[tuple[str, str]]:
    """返回顶部状态栏 formatted text。"""
    frags: list[tuple[str, str]] = []

    # Line 1: title + mode + ASR + LLM
    frags.append(("bold cyan", " Minions "))
    mode_style = _mode_tag(state)
    frags.append((mode_style, f"[{state.conversation_mode}]"))
    if state.asr.status != "idle":
        frags.append(("ansibrightblack", f"  ASR: {state.asr.status}"))
    if state.llm.model:
        frags.append(("ansibrightblack", f"  LLM: {state.llm.model}"))
    if state.paused:
        frags.append(("red", "  [暂停]"))
    frags.append(("", "\n"))

    # Line 2: mic VU meter
    if state.mic.monitoring:
        bar = vu_bar(state.mic.rms)
        rms_str = f"{state.mic.rms:.4f}"
        frags.append(("ansimagenta", f" {bar}  {rms_str}"))
    else:
        frags.append(("ansibrightblack", " 麦克风已停止"))
    if state.mic.device_name:
        frags.append(("ansibrightblack", f"  {state.mic.device_name}"))
    frags.append(("", "\n"))

    return frags


def format_chat(state: UIState) -> list[tuple[str, str]]:
    """返回聊天区域 formatted text。"""
    frags: list[tuple[str, str]] = []

    if state.hidden_message_count > 0:
        frags.append(("ansibrightblack", f"已折叠 {state.hidden_message_count} 条更早消息\n"))

    for msg in state.visible_messages:
        if msg.role == MessageRole.USER:
            frags.append(("bold cyan", f"你：{msg.text}\n"))
        elif msg.role == MessageRole.ASSISTANT:
            frags.append(("green", f"AI：{msg.text}\n"))
        elif msg.role == MessageRole.SYSTEM:
            frags.append(("ansiyellow", f"• {msg.text}\n"))

    if state.error_line:
        frags.append(("red", f"! {state.error_line}\n"))

    return frags


def format_decision(state: UIState) -> list[tuple[str, str]]:
    """返回当前 Gate 判断结果 formatted text。"""
    gate = state.latest_gate
    if not gate.action:
        return [("ansibrightblack", "等待用户输入...")]

    style_map = {
        "silent": "ansibrightblack",
        "bubble": "ansiyellow",
        "judge": "yellow",
        "agent": "bold green",
        "tool": "bold blue",
        "confirm": "bold magenta",
    }
    style = style_map.get(gate.action, "")

    line = f"Gate: {gate.action} | score={gate.score}"
    if gate.reason:
        line += f" | {gate.reason}"

    return [(style, line)]


def format_footer(state: UIState) -> list[tuple[str, str]]:
    """返回底部提示行。"""
    if state.status_line:
        return [("ansibrightblack", state.status_line)]
    return []
