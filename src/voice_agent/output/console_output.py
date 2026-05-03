"""控制台输出：使用 rich 彩色显示各类事件。"""

from rich.console import Console
from rich.text import Text

_CONSOLE = Console()

STYLE_MAP = {
    "asr.partial": "dim",
    "asr.final": "bold cyan",
    "gate.result": "yellow",
    "agent.reply": "bold green",
    "state.change": "magenta",
    "bubble": "dim yellow",
}


async def handle_console_output(event: dict) -> None:
    """根据事件类型以不同颜色输出。"""
    etype = event.get("type", "")
    style = STYLE_MAP.get(etype, "white")

    if etype == "asr.partial":
        text = event.get("text", "")
        _CONSOLE.print(Text(f"  [partial] {text}", style=style), end="\r")

    elif etype == "asr.speech_start":
        _CONSOLE.print(Text("  [ASR] 检测到语音开始", style="bold cyan"))

    elif etype == "asr.speech_end":
        dur = event.get("duration_ms", 0)
        forced = event.get("forced", False)
        tag = "（强制截断）" if forced else ""
        _CONSOLE.print(Text(f"  [ASR] 语音结束 duration={dur}ms{tag}，开始识别...", style="cyan"))

    elif etype == "asr.error":
        msg = event.get("message", "")
        _CONSOLE.print(Text(f"  [ASR] 错误: {msg}", style="bold red"))

    elif etype == "asr.final":
        text = event.get("text", "")
        conf = event.get("confidence", 1.0)
        _CONSOLE.print(Text(f"[ASR final] {text}  (置信度={conf:.2f})", style=style))

    elif etype == "gate.result":
        action = event.get("action", "")
        score = event.get("score", 0)
        reason = event.get("reason", "")
        _CONSOLE.print(Text(f"[Gate] action={action} score={score} reason={reason}", style=style))

    elif etype == "agent.reply":
        text = event.get("text", "")
        _CONSOLE.print(Text(f"[Agent] {text}", style=style))

    elif etype == "state.change":
        state = event.get("state", "")
        _CONSOLE.print(Text(f"[状态] → {state}", style=style))

    elif etype == "bubble":
        msg = event.get("message", "")
        _CONSOLE.print(Text(f"[Bubble] {msg}", style=style))

    elif etype == "command.exit":
        _CONSOLE.print(Text("[系统] 收到退出指令", style="bold red"))

    elif etype == "command.pause":
        _CONSOLE.print(Text("[系统] 已暂停", style="bold magenta"))

    elif etype == "command.resume":
        _CONSOLE.print(Text("[系统] 已恢复", style="bold magenta"))

    elif etype == "audio.level":
        rms = event.get("rms", 0.0)
        bar_len = min(int(rms * 200), 40)
        bar = "#" * bar_len + "." * (40 - bar_len)
        _CONSOLE.print(Text(f"  [Audio] rms={rms:.4f}  |{bar}|", style="dim"))
