"""主入口：启动常驻语音 Agent。"""

import argparse
import asyncio
import contextlib
import os
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
from voice_agent.core.persona_test_cases import PERSONA_TEST_CASES
from voice_agent.core.local_judge_client import LocalJudgeClient
from voice_agent.core.health_check import check_runtime_health
from voice_agent.core.runtime_controller import RuntimeController
from voice_agent.asr.mock_asr import MockASR
from voice_agent.asr.sherpa_onnx_asr import SherpaOnnxASR
from voice_agent.output.console_output import handle_console_output
from voice_agent.output.websocket_server import WebSocketServer
from voice_agent.audio.microphone import Microphone, calculate_rms
from voice_agent.cli.dynamic_shell import DynamicMinionsShell


def build_gate(config: dict) -> InterventionGate:
    ic = config.get("intervention", {})
    thresholds = ic.get("thresholds", {})
    judge_cfg = config.get("judge", {})
    judge_thresholds = judge_cfg.get("thresholds", {})
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
        judge_provider=judge_cfg.get("provider", "rule"),
        local_judge_min=judge_thresholds.get("local_judge_min", 10),
        local_judge_max=judge_thresholds.get("local_judge_max", 74),
    )


def build_local_judge(config: dict) -> LocalJudgeClient | None:
    """根据配置构建 LocalJudgeClient，非 local provider 时返回 None。"""
    jc = config.get("judge", {})
    if jc.get("provider", "rule") != "local":
        return None

    lc = jc.get("local", {})
    return LocalJudgeClient(
        enabled=lc.get("enabled", True),
        api_base=lc.get("api_base", "http://127.0.0.1:11434/v1"),
        api_key=lc.get("api_key", "ollama"),
        model=lc.get("model", "qwen3.5:4b"),
        timeout_seconds=lc.get("timeout_seconds", 6),
        temperature=lc.get("temperature", 0),
        max_tokens=lc.get("max_tokens", 256),
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
    local_judge = build_local_judge(config)
    agent = AgentCore(bus, state, gate, llm, local_judge=local_judge)

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
            if local_judge is not None:
                await local_judge.close()

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
        if local_judge is not None:
            await local_judge.close()
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
    health_report: object | None = None,
    runtime_info: dict | None = None,
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
    local_judge = build_local_judge(config)
    agent = AgentCore(bus, state, gate, llm, local_judge=local_judge)

    # 麦克风（可选，用于 VU 监测）
    ac = config.get("audio", {})
    mic = Microphone(
        sample_rate=ac.get("sample_rate", 16000),
        channels=ac.get("channels", 1),
        chunk_ms=ac.get("chunk_ms", 100),
        device=ac.get("device"),
    )

    # RuntimeController — 管理 ASR 生命周期，默认不启动
    asr_engine_name = asr_override or config.get("asr", {}).get("engine", "mock")
    runtime_controller = RuntimeController(
        bus=bus,
        config=config,
        asr_engine_name=asr_engine_name,
        asr_factory=build_asr,
    )
    logger.info("[系统] ASR 引擎: %s（默认待机，输入 /wakeup 启动）", asr_engine_name)

    # 桥接 asr.final → AgentCore（仅当运行时处于监听状态）
    async def on_asr_final(event: dict) -> None:
        if event.get("type") == "asr.final":
            if not runtime_controller.is_listening:
                logger.debug("[CLI] 跳过 asr.final — 运行时未处于监听状态")
                return
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
            if local_judge is not None:
                await local_judge.close()

    bus.subscribe(on_asr_final)
    bus.subscribe(on_command)

    # 交互式外壳
    shell = DynamicMinionsShell(
        bus, agent, state, llm,
        mic=mic, asr_engine=None,
        runtime_controller=runtime_controller,
        health_report=health_report,
        runtime_info=runtime_info,
        config=config,
        config_path=config_path,
    )
    shell.subscribe()

    llm_mode = "真实调用" if llm.is_available else "mock 模式"
    logger.info("[系统] LLM: %s（%s）", config.get("llm", {}).get("model", "mock"), llm_mode)

    try:
        await shell.run()
    finally:
        await runtime_controller.close()
        await llm.close()
        if local_judge is not None:
            await local_judge.close()
        logger.info("[系统] voice-agent CLI 已退出")


async def run_tui(
    config_path: str,
    asr_override: str | None = None,
    vad_threshold: float | None = None,
    completion_enabled: bool = True,
) -> None:
    """TUI 默认模式：健康检查 + 完整 TUI。"""
    config = get_config(config_path)
    health_report = check_runtime_health(config)

    runtime_info = {
        "asr_engine": asr_override or config.get("asr", {}).get("engine", "mock"),
        "judge_provider": config.get("judge", {}).get("provider", "rule"),
        "judge_model": config.get("judge", {}).get("local", {}).get("model", ""),
        "llm_model": config.get("llm", {}).get("model", ""),
        "assistant_name": config.get("assistant", {}).get("name", "琉璃川"),
        "completion_enabled": completion_enabled,
    }

    # 如果 ASR 模型缺失，运行时仍然可以启动 TUI，/wakeup 时再处理
    await run_cli(
        config_path,
        asr_override=asr_override,
        vad_threshold=vad_threshold,
        health_report=health_report,
        runtime_info=runtime_info,
    )


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


async def run_persona_test(config_path: str) -> None:
    """Persona 测试模式：不经过 ASR，直接用固定文本测试 Gate + LLM + Prompt。"""
    config = get_config(config_path)
    debug = config.get("app", {}).get("debug", False)

    logger = setup_logging(
        debug=debug,
        console=True,
        console_level="WARNING",
        log_file="logs/persona-test.log",
    )
    _ = logger  # used implicitly via logging infra

    bus = EventBus()
    events: list[dict] = []

    async def collect_event(event: dict) -> None:
        events.append(event)

    bus.subscribe(collect_event)

    state = ConversationState(
        _cooldown_seconds=config.get("intervention", {}).get("cooldown_seconds", 5),
        _conversation_timeout_seconds=config.get("intervention", {}).get("conversation_timeout_seconds", 60),
    )
    gate = build_gate(config)
    llm = build_llm(config)
    local_judge = build_local_judge(config)
    agent = AgentCore(bus, state, gate, llm, local_judge=local_judge)

    print("=" * 60)
    print("  琉璃川 Persona Test")
    print("=" * 60)
    print("此模式不经过 ASR，只测试 Gate + LLM + Prompt 效果。")
    print()

    try:
        for i, text in enumerate(PERSONA_TEST_CASES, 1):
            print("-" * 60)
            print(f"[{i}] 用户：{text}")

            before = len(events)
            await agent.handle_final_text(text, 1.0)
            new_events = events[before:]

            gate_events = [e for e in new_events if e.get("type") == "gate.result"]
            replies = [e for e in new_events if e.get("type") == "agent.reply"]
            state_changes = [e for e in new_events if e.get("type") == "state.change"]

            for e in gate_events:
                print(
                    f"    Gate: {e.get('action')} "
                    f"score={e.get('score')} "
                    f"reason={e.get('reason')}"
                )

            for e in state_changes:
                print(
                    f"    State: {e.get('state')} "
                    f"reason={e.get('reason', '')}"
                )

            if replies:
                for e in replies:
                    print(f"    琉璃川：{e.get('text', '')}")
            else:
                print("    琉璃川：<无回复>")

            print()

    finally:
        await llm.close()
        if local_judge is not None:
            await local_judge.close()

    print("=" * 60)
    print("Persona Test 完成")
    print("详细日志见 logs/persona-test.log")


async def run_judge_test(config_path: str, text: str) -> None:
    """Judge 测试模式：只用 Gate + LocalJudge，不经过完整链路。"""
    config = get_config(config_path)
    debug = config.get("app", {}).get("debug", False)
    setup_logging(debug)

    gate = build_gate(config)
    local_judge = build_local_judge(config)

    state = ConversationState(
        _cooldown_seconds=0,
        _conversation_timeout_seconds=config.get("intervention", {}).get("conversation_timeout_seconds", 60),
    )

    result = gate.evaluate(text, state, 1.0)

    print("=" * 72)
    print("  Judge Test")
    print("=" * 72)
    print(f"Text: {text}")
    print(f"Gate: action={result.action.value} score={result.score} reason={result.reason}")
    print()

    if local_judge is None:
        print("LocalJudge 未启用。请设置 judge.provider=local")
        return

    assistant_name = "琉璃川"
    user_title = "少爷"
    if gate.wake_matcher is not None:
        cfg = gate.wake_matcher.config
        assistant_name = cfg.name
        user_title = cfg.user_title

    try:
        judge = await local_judge.judge(
            text=result.text or text,
            state=state.mode,
            wake_session_active=state.is_wake_session_active(),
            recent_context="无",
            gate_action=result.action.value,
            score=result.score,
            reason=result.reason,
            assistant_name=assistant_name,
            user_title=user_title,
        )
        print(f"LocalJudge target: {judge.target}")
        print(f"LocalJudge should_reply: {judge.should_reply}")
        print(f"LocalJudge should_end_wake_session: {judge.should_end_wake_session}")
        print(f"LocalJudge confidence: {judge.confidence}")
        print(f"LocalJudge reason: {judge.reason}")
        print()
        print(f"raw: {judge.raw}")
    finally:
        await local_judge.close()


def _run_with_signal_guard(coro_func, *args, **kwargs):
    try:
        async def _runner():
            loop = asyncio.get_running_loop()
            stop_event = asyncio.Event()

            def _on_sigint():
                stop_event.set()

            try:
                loop.add_signal_handler(signal.SIGINT, _on_sigint)
            except (NotImplementedError, RuntimeError):
                pass

            main_task = asyncio.create_task(coro_func(*args, **kwargs))
            stop_task = asyncio.create_task(stop_event.wait())
            done, _ = await asyncio.wait(
                [main_task, stop_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            try:
                loop.remove_signal_handler(signal.SIGINT)
            except (NotImplementedError, RuntimeError):
                pass

            if stop_task in done:
                main_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await main_task
                raise KeyboardInterrupt()

            return main_task.result()

        asyncio.run(_runner())
    except KeyboardInterrupt:
        pass


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
    parser.add_argument("--cli", action="store_true", help="兼容旧参数：进入 TUI。现在默认就是 TUI，可以省略。")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="无 TUI 运行，直接启动 ASR/Agent 管道",
    )
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
    parser.add_argument(
        "--persona-test",
        action="store_true",
        help="测试琉璃川人格 Prompt 效果，不经过 ASR",
    )
    parser.add_argument(
        "--judge-test",
        default=None,
        help="测试本地 Judge，例如 --judge-test \"这剧情怎么这样\"",
    )
    parser.add_argument(
        "--no-completion",
        action="store_true",
        help="禁用 TUI 命令补全，用于排查输入崩溃",
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
        elif args.persona_test:
            asyncio.run(run_persona_test(args.config))
        elif args.judge_test:
            asyncio.run(run_judge_test(args.config, args.judge_test))
        elif args.cli:
            _run_with_signal_guard(run_tui, args.config, args.asr, args.vad_threshold, not args.no_completion)
        elif args.headless:
            _run_with_signal_guard(run, args.config, args.asr, args.vad_threshold)
        else:
            _run_with_signal_guard(run_tui, args.config, args.asr, args.vad_threshold, not args.no_completion)
    except KeyboardInterrupt:
        pass
    except FileNotFoundError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
