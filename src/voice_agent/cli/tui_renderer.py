"""OpenCode 风格 TUI 渲染器 — 将 UIState 渲染为 prompt_toolkit formatted text。"""

from __future__ import annotations

import math

from voice_agent.cli.ui_state import UIState, MessageRole
from voice_agent.cli.formatters import vu_bar


def format_top_bar(state: UIState) -> list[tuple[str, str]]:
    """顶部状态栏：Minions / 琉璃川  ● Listening  ASR:xxx  Judge:xxx  LLM:xxx"""
    frags: list[tuple[str, str]] = []

    # 左侧：应用名 / 助手名
    frags.append(("bold cyan", f" {state.app_name}"))
    frags.append(("bold white", f" {state.assistant_name}  "))

    # 状态指示
    if state.paused:
        frags.append(("red", "● Paused"))
    elif state.asr.status in ("error",):
        frags.append(("red", "● Error"))
    elif state.asr.status in ("recognizing",):
        frags.append(("yellow", "● Recognizing"))
    elif state.asr.status in ("listening",):
        frags.append(("green", "● Listening"))
    else:
        frags.append(("ansibrightblack", "● Idle"))

    # ASR 引擎
    frags.append(("ansibrightblack", f"  ASR:{state.asr_engine}"))

    # Judge
    if state.judge_provider == "local":
        frags.append(("ansibrightblack", f"  Judge:{state.judge_model}"))
    else:
        frags.append(("ansibrightblack", "  Judge:rule"))

    # LLM
    llm_label = state.llm_model or state.llm.model or "mock"
    frags.append(("ansibrightblack", f"  LLM:{llm_label}"))

    return frags


def format_chat_panel(state: UIState) -> list[tuple[str, str]]:
    """左侧聊天区：只显示用户和琉璃川的对话。"""
    frags: list[tuple[str, str]] = []

    if not state.messages:
        frags.append(("ansibrightblack", "等待交互… 输入文字或直接说话\n"))
        return frags

    if state.hidden_message_count > 0:
        frags.append(("ansibrightblack", f"… 已折叠 {state.hidden_message_count} 条更早消息\n"))

    for msg in state.visible_messages:
        if msg.role == MessageRole.USER:
            frags.append(("bold cyan", f"你：{msg.text}\n"))
        elif msg.role == MessageRole.ASSISTANT:
            prefix = f"{state.assistant_name}：" if state.assistant_name else "AI："
            frags.append(("green", f"{prefix}{msg.text}\n"))
        elif msg.role == MessageRole.SYSTEM:
            frags.append(("ansiyellow", f"• {msg.text}\n"))

    if state.error_line:
        frags.append(("red", f"✗ {state.error_line}\n"))

    return frags


def format_side_panel(state: UIState) -> list[tuple[str, str]]:
    """右侧状态面板。"""
    frags: list[tuple[str, str]] = []

    # ── Status ──
    frags.append(("bold underline", "Status"))
    frags.append(("", "\n"))

    # Wake
    if state.wake_active:
        frags.append(("green", f"  Wake: active {state.wake_remaining_seconds:.0f}s"))
    else:
        frags.append(("ansibrightblack", "  Wake: inactive"))
    frags.append(("", "\n"))

    # Mode
    mode_style = {"active_chat": "green", "cooldown": "yellow", "paused": "red"}.get(
        state.conversation_mode, "ansibrightblack"
    )
    frags.append((mode_style, f"  Mode: {state.conversation_mode}"))
    frags.append(("", "\n"))

    # Gate
    g = state.latest_gate
    if g.action:
        gate_style = {
            "agent": "green", "local_judge": "yellow", "judge": "yellow",
            "silent": "ansibrightblack", "bubble": "ansiyellow",
        }.get(g.action, "")
        frags.append((gate_style, f"  Gate: {g.action} score={g.score}"))
    else:
        frags.append(("ansibrightblack", "  Gate: waiting"))
    frags.append(("", "\n"))

    # Judge
    if state.latest_judge_provider:
        tag = "reply" if state.latest_judge_should_reply else "silent"
        jstyle = "green" if state.latest_judge_should_reply else "ansibrightblack"
        frags.append((jstyle, f"  Judge: {tag}"))
        frags.append(("", f" conf={state.latest_judge_confidence:.2f}"))
        if state.latest_judge_target:
            frags.append(("", f" target={state.latest_judge_target}"))
    else:
        frags.append(("ansibrightblack", "  Judge: -"))
    frags.append(("", "\n"))

    # ASR
    asr_status_style = {
        "listening": "green", "recognizing": "yellow",
        "loaded": "green", "loading": "yellow", "error": "red",
    }.get(state.asr.status, "ansibrightblack")
    frags.append((asr_status_style, f"  ASR: {state.asr.status}"))
    frags.append(("", "\n"))

    # Mic
    if state.mic.monitoring:
        bar = vu_bar(state.mic.rms, width=10)
        frags.append(("ansimagenta", f"  Mic: {bar} {state.mic.rms:.4f}"))
    else:
        frags.append(("ansibrightblack", "  Mic: stopped"))
    frags.append(("", "\n"))

    # ── Runtime ──
    frags.append(("", "\n"))
    frags.append(("bold underline", "Runtime"))
    frags.append(("", "\n"))

    llm_label = state.llm_model or state.llm.model or "mock"
    llm_style = "green" if state.llm.available else "yellow"
    frags.append((llm_style, f"  Main LLM: {llm_label}"))
    frags.append(("", "\n"))

    jm = state.judge_model or "-"
    frags.append(("ansibrightblack", f"  Local Judge: {jm}"))
    frags.append(("", "\n"))

    frags.append(("ansibrightblack", "  Logs: logs/minions.log"))
    frags.append(("", "\n"))

    # ── Health ──
    if state.health_items:
        frags.append(("", "\n"))
        frags.append(("bold underline", "Health"))
        frags.append(("", "\n"))
        for item in state.health_items:
            ok = getattr(item, "ok", False)
            name = getattr(item, "name", "?")
            msg = getattr(item, "message", "")
            if ok:
                frags.append(("green", f"  ✓ {name}"))
            elif item.level == "error":
                frags.append(("red", f"  ✗ {name}"))
            else:
                frags.append(("yellow", f"  ! {name}"))
            frags.append(("", "\n"))

    # ── Notices ──
    if state.notifications:
        frags.append(("", "\n"))
        frags.append(("bold underline", "Notices"))
        frags.append(("", "\n"))
        for note in state.notifications[-state.max_notifications:]:
            frags.append(("ansibrightblack", f"  · {note}"))
            frags.append(("", "\n"))

    return frags


def format_input_prompt(state: UIState) -> list[tuple[str, str]]:
    """底部输入提示符。"""
    if state.paused:
        return [("red", "暂停 > ")]
    if state.assistant_name:
        return [("bold cyan", f"{state.assistant_name} > ")]
    return [("bold cyan", "> ")]
