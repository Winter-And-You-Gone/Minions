"""麦克风采集（预留实现）。

第一阶段使用 MockASR，不需要真实麦克风。
"""


class Microphone:
    """麦克风采集器（预留）。"""

    def __init__(self, sample_rate: int = 16000, channels: int = 1, device: str | None = None) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self._stream = None

    def start(self) -> None:
        # TODO: 使用 sounddevice.InputStream 采集音频
        pass

    def stop(self) -> None:
        pass

    def read_chunk(self) -> bytes:
        # TODO: 返回音频数据
        return b""
