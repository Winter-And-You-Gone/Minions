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
from voice_agent.asr.mock_asr import MockASR
from voice_agent.asr.sherpa_onnx_asr import SherpaOnnxASR
from voice_agent.output.console_output import handle_console_output
from voice_agent.output.websocket_server import WebSocketServer


def build_gate(config: dict) -> InterventionGate:
    ic = config.get("intervention", {})
    thresholds = ic.get("thresholds", {})
    return InterventionGate(
        min_text_length=ic.get("min_text_length", 4),
        min_asr_confidence=ic.get("min_asr_confidence", 0.55),
        cooldown_seconds=ic.get("cooldown_seconds", 5),
        threshold_bubble=thresholds.get("bubble", 15),
        threshold_judge=thresholds.get("judge", 30),
        threshold_agent=thresholds.get("agent", 60),
        uncertain_action=ic.get("uncertain_action", "judge"),
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
        return SherpaOnnxASR(event_bus, config.get("asr", {}).get("sherpa_onnx", {}))
    raise ValueError(f"不支持的 ASR 引擎: {engine_name}")


async def run(config_path: str, asr_override: str | None = None) -> None:
    config = get_config(config_path)
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
    logger.info("[系统] LLM: %s (enabled=%s)", config.get("llm", {}).get("model", "mock"), llm.is_available)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="voice-agent: 常驻语音 Agent")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径 (默认: config.yaml)")
    parser.add_argument("--asr", default=None, help="ASR 引擎覆盖 (mock / sherpa-onnx)")
    args = parser.parse_args()

    try:
        asyncio.run(run(args.config, args.asr))
    except KeyboardInterrupt:
        pass
    except FileNotFoundError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
