"""语音活动检测 VAD（预留实现）。

第一阶段使用 MockASR，不需要 VAD。
"""


class VAD:
    """语音活动检测器（预留）。"""

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self._speech_active = False

    def is_speech(self, audio_chunk: bytes) -> bool:
        # TODO: 使用 webrtcvad / silero-vad 检测语音活动
        return True

    @property
    def speech_active(self) -> bool:
        return self._speech_active
