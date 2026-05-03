"""WebSocket 服务器：广播所有事件给连接的客户端。"""

import asyncio
import json

import websockets
from websockets.asyncio.server import serve

from voice_agent.logger import get_logger


class WebSocketServer:
    """WebSocket 事件广播服务器。"""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self._host = host
        self._port = port
        self._clients: set = set()
        self._server = None
        self._logger = get_logger()
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._broadcast_task: asyncio.Task | None = None

    async def start(self) -> None:
        self._server = await serve(
            self._handler,
            self._host,
            self._port,
        )
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())
        self._logger.info("[WebSocket] 已启动 ws://%s:%s", self._host, self._port)

    async def stop(self) -> None:
        if self._broadcast_task:
            self._broadcast_task.cancel()
            self._broadcast_task = None
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._logger.info("[WebSocket] 已停止")

    async def on_event(self, event: dict) -> None:
        """接收事件并放入广播队列。"""
        await self._event_queue.put(event)

    async def _handler(self, websocket) -> None:
        self._clients.add(websocket)
        self._logger.info("[WebSocket] 客户端连接 (%d 个在线)", len(self._clients))
        try:
            await websocket.wait_closed()
        finally:
            self._clients.discard(websocket)
            self._logger.info("[WebSocket] 客户端断开 (%d 个在线)", len(self._clients))

    async def _broadcast_loop(self) -> None:
        while True:
            try:
                event = await self._event_queue.get()
            except asyncio.CancelledError:
                break
            payload = json.dumps(event, ensure_ascii=False)
            if self._clients:
                disconnected = set()
                for ws in self._clients:
                    try:
                        await ws.send(payload)
                    except websockets.ConnectionClosed:
                        disconnected.add(ws)
                self._clients -= disconnected
