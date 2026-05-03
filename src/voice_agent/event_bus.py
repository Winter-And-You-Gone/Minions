"""事件总线：异步发布/订阅模式。"""

import asyncio
import time
from collections.abc import Callable, Coroutine
from typing import Any

from voice_agent.logger import get_logger

Callback = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]


class EventBus:
    """异步事件总线。"""

    def __init__(self) -> None:
        self._subscribers: list[Callback] = []
        self._logger = get_logger()

    def subscribe(self, callback: Callback) -> None:
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callback) -> None:
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    async def publish(self, event: dict[str, Any]) -> None:
        if "timestamp" not in event:
            event["timestamp"] = time.time()
        results = []
        for cb in self._subscribers:
            results.append(cb(event))
        # 收集所有结果并记录异常
        outcomes = await asyncio.gather(
            *[r if asyncio.iscoroutine(r) else _noop(r) for r in results],
            return_exceptions=True,
        )
        for cb, outcome in zip(self._subscribers, outcomes):
            if isinstance(outcome, Exception):
                self._logger.error(
                    "[EventBus] subscriber %s 异常: %s", getattr(cb, "__name__", str(cb)), outcome
                )


async def _noop(_result: Any) -> None:
    pass
