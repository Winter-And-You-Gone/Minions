"""MockASR：从控制台读取用户输入模拟语音识别。"""

import asyncio
import sys

from voice_agent.event_bus import EventBus
from voice_agent.logger import get_logger


class MockASR:
    """模拟 ASR 引擎，从控制台逐行读取输入作为 ASR final 文本。

    支持命令:
      /pause  - 暂停
      /resume - 恢复
      /exit   - 退出
    """

    def __init__(self, event_bus: EventBus) -> None:
        self._bus = event_bus
        self._running = False
        self._paused = False
        self._logger = get_logger()

    async def start(self) -> None:
        self._running = True
        self._logger.info("[MockASR] 已启动，输入文本模拟语音（/pause /resume /exit）")
        print("\n========================================")
        print("  MockASR 已就绪，请输入文本：")
        print("  /pause  暂停 | /resume 恢复 | /exit 退出")
        print("========================================\n")

        loop = asyncio.get_event_loop()

        while self._running:
            try:
                line = await loop.run_in_executor(None, sys.stdin.readline)
            except (EOFError, KeyboardInterrupt):
                break

            if not line:
                break

            text = line.strip()
            if not text:
                continue

            if text == "/exit":
                self._running = False
                await self._bus.publish({"type": "command.exit"})
                break

            if text == "/pause":
                self._paused = True
                await self._bus.publish({"type": "command.pause"})
                self._logger.info("[MockASR] 已暂停")
                continue

            if text == "/resume":
                self._paused = False
                await self._bus.publish({"type": "command.resume"})
                self._logger.info("[MockASR] 已恢复")
                continue

            if self._paused:
                self._logger.info("[MockASR] 暂停中，忽略输入: %s", text)
                continue

            await self._bus.publish({
                "type": "asr.final",
                "text": text,
                "confidence": 1.0,
            })

    async def stop(self) -> None:
        self._running = False
        self._logger.info("[MockASR] 已停止")

    async def set_callback(self, cb) -> None:
        pass
