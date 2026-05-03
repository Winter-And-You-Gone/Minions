"""主入口：启动常驻语音 Agent。"""

import argparse
import asyncio
import signal
import sys

from voice_agent.config import get_config
from voice_agent.event_bus import EventBus
from voice_agent.logger import setup_logging, get_logger
from voice_agent.core.conversation_state import ConversationState
from voice_agent.core.intervention_gate import InterventionGate
from voice_agent.core.llm_client import LLMClient
from voice_agent.core.agent_core import AgentCore
from voice_agent.core.wake_name import WakeNameMatcher
from voice_agent.asr.mock_asr import MockASR
from voice_agent.asr.sherpa_onnx_asr import SherpaOnnxASR
from voice_agent.output.console_output import handle_console_output
from voice_agent.output.websocket_server import WebSocketServer
from voice_agent.audio.microphone import Microphone, calculate_rms
from voice_agent.cli.dynamic_shell import DynamicMinionsShell


def build_gate(config: dict) -> InterventionGate:
    ic = config.get("intervention", {})
    thresholds = ic.get("thresholds", {})
    wake_matcher = WakeNameMatcher.from_config(config)
    return InterventionGate(
        min_text_length=ic.get("min_text_length", 4),
        min_asr_confidence=ic.get("min_asr_confidence", 0.55),
        cooldown_seconds=ic.get("cooldown_seconds", 5),
        threshold_bubble=thresholds.get("bubble", 15),
        threshold_judge=thresholds.get("judge", 30),
        threshold_agent=thresholds.get("agent", 60),
        uncertain_action=ic.get("uncertain_action", "judge"),
        wake_matcher=wake_matcher,
    )


def build_llm(config: dict) -> LLMClient:
    lc = config.get("llm", {})
    return LLMClient(
        enabled=lc.get("enabled", True),
        api_base=lc.get("api_base", ""),
        api_key=lc.get("api_key", ""),
        model=lc.get("model", ""),
        timeout_seconds=lc.get("timeout_seconds", 30),
        mock_judge_reply=lc.get("mock_judge_reply", False),
    )


def build_asr(engine_name: str, event_bus: EventBus, config: dict) -> MockASR | SherpaOnnxASR:
    if engine_name == "mock":
        return MockASR(event_bus)
    if engine_name == "sherpa-onnx":
        # 合并 audio 顶层配置 + sherpa_onnx 子配置，使 sherpa_onnx 能继承 sample_rate 等
        asr_section = config.get("asr", {})
        asr_config = asr_section.get("sherpa_onnx", {})
        audio_config = config.get("audio", {})
        merged = {**audio_config, **asr_config}
        # 保留嵌套的 vad 子配置
        if "vad" in asr_config:
            merged["vad"] = asr_config["vad"]
        return SherpaOnnxASR(event_bus, merged)
    raise ValueError(f"不支持的 ASR 引擎: {engine_name}")


def apply_runtime_overrides(config: dict, vad_threshold: float | None = None) -> dict:
    """应用命令行运行时覆盖配置。"""
    if vad_threshold is not None:
        config.setdefault("asr", {})
        config["asr"].setdefault("sherpa_onnx", {})
        config["asr"]["sherpa_onnx"].setdefault("vad", {})
        config["asr"]["sherpa_onnx"]["vad"]["rms_threshold"] = vad_threshold
    return config


async def run(
    config_path: str,
    asr_override: str | None = None,
    vad_threshold: float | None = None,
) -> None:
    config = get_config(config_path)
    config = apply_runtime_overrides(config, vad_threshold)
    debug = config.get("app", {}).get("debug", False)
    logger = setup_logging(debug)

    logger.info("========================================")
    logger.info("  voice-agent 启动中...")
    logger.info("========================================")

    # 初始化各组件
    bus = EventBus()
    state = ConversationState(
        _cooldown_seconds=config.get("intervention", {}).get("cooldown_seconds", 5),
        _conversation_timeout_seconds=config.get("intervention", {}).get("conversation_timeout_seconds", 60),
    )
    gate = build_gate(config)
    llm = build_llm(config)
    agent = AgentCore(bus, state, gate, llm)

    # 注册事件处理器
    bus.subscribe(handle_console_output)

    # WebSocket
    ws_config = config.get("websocket", {})
    ws_server: WebSocketServer | None = None
    if ws_config.get("enabled", True):
        ws_server = WebSocketServer(
            host=ws_config.get("host", "127.0.0.1"),
            port=ws_config.get("port", 8765),
        )
        bus.subscribe(ws_server.on_event)
        await ws_server.start()

    # ASR 引擎
    asr_engine_name = asr_override or config.get("asr", {}).get("engine", "mock")
    asr_engine = build_asr(asr_engine_name, bus, config)

    # 桥接 ASR final → AgentCore
    async def on_asr_final(event: dict) -> None:
        if event.get("type") == "asr.final":
            await agent.handle_final_text(
                event.get("text", ""),
                event.get("confidence", 1.0),
            )

    # 桥接命令
    async def on_command(event: dict) -> None:
        etype = event.get("type", "")
        if etype == "command.pause":
            await agent.handle_pause()
        elif etype == "command.resume":
            await agent.handle_resume()
        elif etype == "command.exit":
            if ws_server:
                await ws_server.stop()
            await asr_engine.stop()
            await llm.close()

    bus.subscribe(on_asr_final)
    bus.subscribe(on_command)

    logger.info("[系统] ASR 引擎: %s", asr_engine_name)
    llm_mode = "真实调用" if llm.is_available else "mock 模式"
    logger.info("[系统] LLM: %s（%s）", config.get("llm", {}).get("model", "mock"), llm_mode)
    logger.info("[系统] 输入文本开始交互...")

    try:
        await asr_engine.start()
    except KeyboardInterrupt:
        logger.info("[系统] 收到中断信号")
    except Exception as e:
        logger.error("[系统] 运行错误: %s", e)
    finally:
        if ws_server:
            await ws_server.stop()
        await llm.close()
        logger.info("[系统] voice-agent 已退出")


def list_devices() -> None:
    """列出所有 audio 设备，标记输入设备。"""
    import sounddevice as sd

    print("=" * 72)
    print("  可用音频设备（输入设备标记为 🎤 ）")
    print("=" * 72)
    for i, dev in enumerate(sd.query_devices()):
        name = dev["name"]
        inputs = dev["max_input_channels"]
        outputs = dev["max_output_channels"]
        sr = dev["default_samplerate"]
        marker = "🎤" if inputs > 0 else "  "
        io = f"in={inputs:<2d} out={outputs:<2d}"
        print(f"  {marker} [{i:2d}] {name:<40s} {io}  {sr:6.0f} Hz")
    print("=" * 72)
    print("使用: python -m voice_agent.main --mic-test --device <id>")


async def run_mic_test(config_path: str, device_override: int | str | None = None) -> None:
    """麦克风测试模式：采集音频 → 计算 RMS → 显示音量。"""
    import sounddevice as sd

    config = get_config(config_path)
    debug = config.get("app", {}).get("debug", False)
    logger = setup_logging(debug)

    ac = config.get("audio", {})
    device = device_override if device_override is not None else ac.get("device")
    mic = Microphone(
        sample_rate=ac.get("sample_rate", 16000),
        channels=ac.get("channels", 1),
        chunk_ms=ac.get("chunk_ms", 100),
        device=device,
    )

    bus = EventBus()
    bus.subscribe(handle_console_output)

    logger.info("========================================")
    logger.info("  麦克风测试模式 — 按 Ctrl+C 退出")
    logger.info("========================================")

    try:
        await mic.start()
        logger.info("[MicTest] 开始采集，每 %dms 输出一次音量...", mic.chunk_ms)
        while True:
            chunk = await mic.read_chunk()
            rms = calculate_rms(chunk)
            await bus.publish({"type": "audio.level", "rms": rms})
    except KeyboardInterrupt:
        logger.info("[MicTest] 用户中断")
    except Exception as e:
        logger.error("[MicTest] 错误: %s", e)
    finally:
        await mic.stop()
        logger.info("[MicTest] 已退出")


async def run_cli(
    config_path: str,
    asr_override: str | None = None,
    vad_threshold: float | None = None,
) -> None:
    """CLI 交互模式：启动 AgentCore 管道 + 交互式外壳，可选 ASR 识别。"""
    config = get_config(config_path)
    config = apply_runtime_overrides(config, vad_threshold)
    debug = config.get("app", {}).get("debug", False)
    logger = setup_logging(
        debug,
        console=False,
        log_file="logs/minions.log",
    )

    # 初始化各组件
    bus = EventBus()
    state = ConversationState(
        _cooldown_seconds=config.get("intervention", {}).get("cooldown_seconds", 5),
        _conversation_timeout_seconds=config.get("intervention", {}).get("conversation_timeout_seconds", 60),
    )
    gate = build_gate(config)
    llm = build_llm(config)
    agent = AgentCore(bus, state, gate, llm)

    # 麦克风（可选，用于 VU 监测）
    ac = config.get("audio", {})
    mic = Microphone(
        sample_rate=ac.get("sample_rate", 16000),
        channels=ac.get("channels", 1),
        chunk_ms=ac.get("chunk_ms", 100),
        device=ac.get("device"),
    )

    # ASR 引擎（可选）
    asr_engine = None
    asr_engine_name = asr_override or config.get("asr", {}).get("engine", "mock")
    if asr_engine_name != "mock":
        asr_engine = build_asr(asr_engine_name, bus, config)
        logger.info("[系统] ASR 引擎: %s", asr_engine_name)

    # 桥接 asr.final → AgentCore
    async def on_asr_final(event: dict) -> None:
        if event.get("type") == "asr.final":
            logger.info(
                "[CLI] 收到 asr.final: text=%s source=%s",
                event.get("text", ""),
                event.get("source", ""),
            )
            await agent.handle_final_text(
                event.get("text", ""),
                event.get("confidence", 1.0),
            )

    # 桥接命令
    async def on_command(event: dict) -> None:
        etype = event.get("type", "")
        if etype == "command.pause":
            await agent.handle_pause()
        elif etype == "command.resume":
            await agent.handle_resume()
        elif etype == "command.exit":
            await llm.close()

    bus.subscribe(on_asr_final)
    bus.subscribe(on_command)

    # 交互式外壳
    shell = DynamicMinionsShell(bus, agent, state, llm, mic=mic, asr_engine=asr_engine)
    shell.subscribe()

    llm_mode = "真实调用" if llm.is_available else "mock 模式"
    logger.info("[系统] LLM: %s（%s）", config.get("llm", {}).get("model", "mock"), llm_mode)

    try:
        await shell.run()
    finally:
        if asr_engine is not None:
            await asr_engine.stop()
        await llm.close()
        logger.info("[系统] voice-agent CLI 已退出")


async def run_asr_test(
    config_path: str,
    engine_name: str,
    vad_threshold: float | None = None,
) -> None:
    """ASR 测试模式：只运行 ASR，不进入 AgentCore/LLM。"""
    config = get_config(config_path)
    config = apply_runtime_overrides(config, vad_threshold)

    debug = config.get("app", {}).get("debug", False)
    logger = setup_logging(debug)

    bus = EventBus()
    bus.subscribe(handle_console_output)

    asr_engine = build_asr(engine_name, bus, config)

    logger.info("========================================")
    logger.info("  ASR 测试模式 — %s", engine_name)
    logger.info("  只测试语音识别，不调用 LLM/Gate")
    logger.info("  按 Ctrl+C 退出")
    logger.info("========================================")

    try:
        await asr_engine.start()
    except KeyboardInterrupt:
        logger.info("[ASRTest] 用户中断")
    finally:
        await asr_engine.stop()
        logger.info("[ASRTest] 已退出")


def main() -> None:
    # Windows 控制台 UTF-8 支持
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="voice-agent: 常驻语音 Agent")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径 (默认: config.yaml)")
    parser.add_argument("--asr", default=None, help="ASR 引擎覆盖 (mock / sherpa-onnx)")
    parser.add_argument("--mic-test", action="store_true", help="麦克风测试模式（不启动 ASR/LLM）")
    parser.add_argument("--device", default=None, help="麦克风设备 ID 或名称子串 (仅 --mic-test 时有效)")
    parser.add_argument("--cli", action="store_true", help="CLI 交互模式（可选 --asr 启用语音）")
    parser.add_argument("--list-devices", action="store_true", help="列出所有音频设备")
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=None,
        help="覆盖 VAD RMS 阈值，例如 0.006",
    )
    parser.add_argument(
        "--asr-test",
        choices=["sherpa-onnx"],
        default=None,
        help="只测试真实 ASR，不调用 LLM/Gate，例如 --asr-test sherpa-onnx",
    )
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    try:
        if args.mic_test:
            device = int(args.device) if args.device and args.device.isdigit() else args.device
            asyncio.run(run_mic_test(args.config, device))
        elif args.asr_test:
            asyncio.run(run_asr_test(args.config, args.asr_test, args.vad_threshold))
        elif args.cli:
            asyncio.run(run_cli(args.config, args.asr, args.vad_threshold))
        else:
            asyncio.run(run(args.config, args.asr, args.vad_threshold))
    except KeyboardInterrupt:
        pass
    except FileNotFoundError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
