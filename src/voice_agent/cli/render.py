"""Rich 面板渲染 — 将 UIState 渲染为 stable rich Panel 布局。"""

import math

from rich import box
from rich.columns import Columns
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

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


def _mode_style(mode: str) -> str:
    return {
        "active_chat": "green",
        "passive_listening": "bright_black",
        "cooldown": "yellow",
        "paused": "red",
    }.get(mode, "white")


def render_header(state: UIState) -> Panel:
    """顶部状态栏。"""
    lines = []

    # 第一行：标题 + 模式 + LLM
    mode_tag = f"[{state.conversation_mode}]"
    mode_colored = Text.assemble(
        (" Minions ", "bold cyan"),
        (mode_tag, _mode_style(state.conversation_mode)),
    )
    if state.llm.model:
        mode_colored.append(f"  LLM: {state.llm.model}", "bright_black")
    if state.paused:
        mode_colored.append("  [暂停]", "red")
    lines.append(mode_colored)

    # 第二行：麦克风 VU 表
    if state.mic.monitoring:
        bar = vu_bar(state.mic.rms)
        rms_str = f"{state.mic.rms:.4f}"
        mic_line = Text.assemble(
            (f" {bar}  {rms_str}", "magenta"),
        )
        if state.mic.device_name:
            mic_line.append(f"  {state.mic.device_name}", "bright_black")
        lines.append(mic_line)
    else:
        lines.append(Text(" 麦克风已停止", style="bright_black"))

    return Panel(
        Group(*lines),
        border_style="cyan",
        box=box.SQUARE,
        padding=(0, 1),
    )


def render_messages(state: UIState) -> Panel:
    """对话消息面板。"""
    elements: list[RenderableType] = []

    if state.hidden_message_count > 0:
        elements.append(
            Text(f"已折叠 {state.hidden_message_count} 条更早消息", style="bright_black")
        )

    for msg in state.visible_messages:
        if msg.role == MessageRole.USER:
            elements.append(Text(f"你：{msg.text}", style="bold cyan"))
        elif msg.role == MessageRole.ASSISTANT:
            elements.append(Text(f"AI：{msg.text}", style="green"))
        elif msg.role == MessageRole.SYSTEM:
            elements.append(Text(f"• {msg.text}", style="yellow"))

    if state.error_line:
        elements.append(Text(f"! {state.error_line}", style="red"))

    if not elements:
        elements.append(Text("等待输入...", style="bright_black"))

    return Panel(
        Group(*elements),
        title="[bold]对话[/]",
        border_style="bright_black",
        box=box.SQUARE,
        padding=(0, 1),
    )


def render_decision(state: UIState) -> Panel:
    """Gate 判断面板。"""
    gate = state.latest_gate
    if not gate.action:
        body = Text("等待用户输入...", style="bright_black")
    else:
        style_map = {
            "silent": "bright_black",
            "bubble": "yellow",
            "judge": "bold yellow",
            "agent": "bold green",
            "tool": "bold blue",
            "confirm": "bold magenta",
        }
        style = style_map.get(gate.action, "white")
        line = f"Gate: {gate.action}  score={gate.score}"
        if gate.reason:
            line += f"  |  {gate.reason}"
        body = Text(line, style=style)

    return Panel(body, title="[bold]决策[/]", border_style="bright_black", box=box.SQUARE, padding=(0, 1))


def render_footer(state: UIState) -> Panel:
    """底部状态栏。"""
    if state.status_line:
        body = Text(state.status_line, style="bright_black")
    else:
        body = Text("/help 查看命令  |  直接输入文字与 AI 对话", style="bright_black")

    return Panel(body, border_style="bright_black", box=box.SQUARE, padding=(0, 1))


def render_app(state: UIState) -> Panel:
    """组合所有面板为完整界面。"""
    return Panel(
        Group(
            render_header(state),
            render_messages(state),
            render_decision(state),
            render_footer(state),
        ),
        border_style="bright_black",
        box=box.HEAVY,
        padding=(0, 0),
    )
