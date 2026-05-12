"""RuntimeController — 管理 ASR 引擎启动/停止和运行时生命周期。

职责：
- 管理 ASR 引擎启动/停止。
- 管理运行状态：sleeping / waking / starting / listening / stopping / error。
- 发布 runtime.status 事件给 TUI。
- 不处理 LLM 回复逻辑，不直接改 Gate / AgentCore。
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any, Callable

from voice_agent.event_bus import EventBus
from voice_agent.logger import get_logger

_ASR_START_TIMEOUT = 30.0      # ASR 引擎启动超时（秒）
_STOP_LISTENING_TIMEOUT = 5.0  # 停止监听操作超时（秒）


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
        self._asr_task: asyncio.Task | None = None
        self._engine_ready: asyncio.Event | None = None
        self._logger = get_logger()
        self._bus.subscribe(self._on_asr_status_event)

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_listening(self) -> bool:
        return self._state == "listening"

    async def _on_asr_status_event(self, event: dict) -> None:
        if event.get("type") != "asr.status":
            return
        status = event.get("status", "")
        if status == "listening" and self._state == "starting":
            self._state = "listening"
            if self._engine_ready is not None:
                self._engine_ready.set()
            await self._publish_status("语音监听已启动")
        elif status == "error" and self._state == "starting":
            self._state = "error"
            if self._engine_ready is not None:
                self._engine_ready.set()
            await self._publish_status(event.get("message", "ASR 启动失败"))

    @property
    def asr_engine(self) -> Any | None:
        return self._asr_engine

    async def _publish_status(self, message: str = "") -> None:
        with contextlib.suppress(Exception):
            await self._bus.publish({
                "type": "runtime.status",
                "state": self._state,
                "message": message,
                "asr_engine": self._asr_engine_name,
            })

    async def start_listening(self) -> bool:
        if self._state in ("waking", "listening", "starting"):
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

            self._engine_ready = asyncio.Event()
            self._asr_task = asyncio.create_task(self._asr_engine.start())
            self._state = "starting"
            await self._publish_status("ASR 引擎初始化中...")

            try:
                await asyncio.wait_for(self._engine_ready.wait(), timeout=_ASR_START_TIMEOUT)
            except asyncio.TimeoutError:
                self._logger.error("[Runtime] ASR 启动超时 (%.0fs)", _ASR_START_TIMEOUT)
                self._state = "error"
                await self._publish_status(f"ASR 启动超时 ({_ASR_START_TIMEOUT:.0f}s)")
                return False

            return self._state == "listening"

        except Exception as e:
            self._logger.exception("[Runtime] start_listening failed: %s", e)
            self._state = "error"
            await self._publish_status(f"启动失败: {e}")
            return False

    async def stop_listening(self) -> None:
        if self._state == "sleeping":
            await self._publish_status("已经处于待机状态")
            return

        self._state = "stopping"
        with contextlib.suppress(asyncio.TimeoutError, Exception):
            await asyncio.wait_for(
                self._publish_status("正在停止语音监听..."),
                timeout=_STOP_LISTENING_TIMEOUT,
            )

        if self._asr_task is not None:
            self._asr_task.cancel()
            try:
                await asyncio.wait_for(self._asr_task, timeout=_STOP_LISTENING_TIMEOUT)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                pass
            self._asr_task = None

        if self._asr_engine is not None:
            try:
                await asyncio.wait_for(self._asr_engine.stop(), timeout=_STOP_LISTENING_TIMEOUT)
            except (asyncio.TimeoutError, Exception):
                pass

        self._state = "sleeping"
        self._engine_ready = None
        with contextlib.suppress(asyncio.TimeoutError, Exception):
            await asyncio.wait_for(
                self._publish_status("语音监听已停止"),
                timeout=_STOP_LISTENING_TIMEOUT,
            )

    async def wakeup(self) -> bool:
        return await self.start_listening()

    async def sleep(self) -> None:
        await self.stop_listening()

    async def close(self) -> None:
        await self.stop_listening()
