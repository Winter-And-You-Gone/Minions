"""ASR 引擎协议定义。"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class ASREngine(Protocol):
    """ASR 引擎接口协议。"""

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def set_callback(self, cb) -> None: ...
