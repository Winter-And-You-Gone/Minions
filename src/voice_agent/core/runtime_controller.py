"""RuntimeController — 管理 ASR 引擎启动/停止和运行时生命周期。

职责：
- 管理 ASR 引擎启动/停止。
- 管理运行状态：sleeping / waking / listening / stopping / error。
- 发布 runtime.status 事件给 TUI。
- 不处理 LLM 回复逻辑，不直接改 Gate / AgentCore。
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any, Callable

from voice_agent.event_bus import EventBus
from voice_agent.logger import get_logger


class RuntimeController:
    def __init__(
        self,
        *,
        bus: EventBus,
        config: dict,
        asr_engine_name: str,
        asr_factory: Callable[[str, EventBus, dict], Any],
    ) -> None:
        self._bus = bus
        self._config = config
        self._asr_engine_name = asr_engine_name
        self._asr_factory = asr_factory
        self._asr_engine: Any | None = None
        self._state = "sleeping"
        self._asr_task: Any | None = None
        self._logger = get_logger()

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_listening(self) -> bool:
        return self._state == "listening"

    @property
    def asr_engine(self) -> Any | None:
        return self._asr_engine

    async def _publish_status(self, message: str = "") -> None:
        await self._bus.publish({
            "type": "runtime.status",
            "state": self._state,
            "message": message,
            "asr_engine": self._asr_engine_name,
        })

    async def start_listening(self) -> bool:
        """启动实时语音监听。"""
        if self._state in ("waking", "listening"):
            await self._publish_status("已经处于监听状态")
            return True

        self._state = "waking"
        await self._publish_status("正在启动语音监听...")

        try:
            if self._asr_engine is None:
                self._asr_engine = self._asr_factory(
                    self._asr_engine_name,
                    self._bus,
                    self._config,
                )

            self._asr_task = asyncio.create_task(self._asr_engine.start())
            self._state = "listening"
            await self._publish_status("语音监听已启动")
            return True

        except Exception as e:
            self._logger.exception("[Runtime] start_listening failed: %s", e)
            self._state = "error"
            await self._publish_status(f"启动失败: {e}")
            return False

    async def stop_listening(self) -> None:
        """停止实时语音监听。"""
        if self._state == "sleeping":
            await self._publish_status("已经处于待机状态")
            return

        self._state = "stopping"
        await self._publish_status("正在停止语音监听...")

        if self._asr_task is not None:
            self._asr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._asr_task
            self._asr_task = None

        if self._asr_engine is not None:
            with contextlib.suppress(Exception):
                await self._asr_engine.stop()

        self._state = "sleeping"
        await self._publish_status("语音监听已停止")

    async def wakeup(self) -> bool:
        """保留兼容：委托给 start_listening。"""
        return await self.start_listening()

    async def sleep(self) -> None:
        """保留兼容：委托给 stop_listening。"""
        await self.stop_listening()

    async def close(self) -> None:
        await self.stop_listening()
