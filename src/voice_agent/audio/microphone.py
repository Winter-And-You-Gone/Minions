"""麦克风采集：使用 sounddevice 实现实时音频流采集。"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from voice_agent.logger import get_logger


def calculate_rms(data: np.ndarray) -> float:
    """计算音频数据的 RMS（均方根）音量。

    Args:
        data: float32 PCM，形状 (frames,) 或 (frames, channels)。

    Returns:
        0~1 范围的 RMS 值。
    """
    if data.size == 0:
        return 0.0
    # 多通道取平均
    return float(np.sqrt(np.mean(np.asarray(data, dtype=np.float64) ** 2)))


class Microphone:
    """麦克风采集器。

    通过 sounddevice.InputStream 回调将音频块送入 asyncio.Queue，
    外部使用 read_chunk() 异步消费。
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_ms: int = 100,
        device: int | str | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_ms = chunk_ms
        self.device = device
        self._stream: Any = None
        self._queue: asyncio.Queue[np.ndarray] | None = None
        self._logger = get_logger()

    async def start(self) -> None:
        """启动麦克风流，开始采集音频块。"""
        import sounddevice as sd

        blocksize = int(self.sample_rate * self.chunk_ms / 1000)
        self._queue = asyncio.Queue()

        device_info = sd.query_devices(self.device, kind="input") if self.device is None else sd.query_devices(self.device)
        actual_device = self.device
        if actual_device is None:
            try:
                default_input = sd.query_devices(kind="input")
                actual_device = default_input.get("name", None)
            except Exception:
                actual_device = None

        # 对于 device=None，sounddevice 会使用默认输入设备
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            device=self.device,  # None = 默认输入设备
            blocksize=blocksize,
            callback=self._audio_callback,
            dtype="float32",
        )
        self._stream.start()

        name = sd.query_devices(self.device, kind="input")["name"] if self.device is not None else sd.default.device[0]
        display = sd.query_devices(name)["name"] if isinstance(name, int) else name
        self._logger.info(
            "[Mic] 已启动 %s Hz %dch %dms blocksize=%d",
            self.sample_rate,
            self.channels,
            self.chunk_ms,
            blocksize,
        )
        if self.device is not None:
            self._logger.info("[Mic] 设备: %s", display)

    def _audio_callback(self, indata: np.ndarray, _frames: int, _time_info: Any, status: Any) -> None:
        """sounddevice 回调：将音频块放入队列。"""
        if status:
            self._logger.warning("[Mic] %s", status)
        if self._queue is not None:
            self._queue.put_nowait(indata.copy())

    async def read_chunk(self) -> np.ndarray:
        """异步读取下一个音频块，返回 float32 PCM (frames, channels)。"""
        if self._queue is None:
            raise RuntimeError("麦克风未启动，请先调用 start()")
        return await self._queue.get()

    async def stop(self) -> None:
        """停止麦克风流。"""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                self._logger.warning("[Mic] 关闭异常: %s", e)
            self._stream = None
            self._queue = None
            self._logger.info("[Mic] 已停止")

    @property
    def blocksize(self) -> int:
        return int(self.sample_rate * self.chunk_ms / 1000)
