"""基于 RMS 的简单语音分段器（VAD）。

将连续的音频块按音量分为 silence / speech 两段，
检测到语音结束后输出完整音频段，供 offline ASR 识别。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from voice_agent.audio.microphone import calculate_rms


@dataclass
class VADSegmenterConfig:
    sample_rate: int = 16000
    chunk_ms: int = 100
    rms_threshold: float = 0.008
    min_speech_ms: int = 300
    max_speech_ms: int = 15000
    silence_timeout_ms: int = 800
    pre_roll_ms: int = 300

    def __post_init__(self) -> None:
        self._min_speech_samples: int = int(self.min_speech_ms * self.sample_rate / 1000)
        self._max_speech_samples: int = int(self.max_speech_ms * self.sample_rate / 1000)
        self._silence_timeout_samples: int = int(self.silence_timeout_ms * self.sample_rate / 1000)
        self._pre_roll_samples: int = int(self.pre_roll_ms * self.sample_rate / 1000)


_EVENT_SILENCE = "silence"
_EVENT_SPEECH_START = "speech_start"
_EVENT_SPEECHING = "speeching"
_EVENT_SPEECH_END = "speech_end"
_EVENT_SPEECH_FORCED_END = "speech_forced_end"


class SpeechSegmenter:
    """基于 RMS 能量的语音分段器。

    用法:
        seg = SpeechSegmenter(config)
        for chunk in mic_stream:
            event, segment, rms = seg.accept_chunk(chunk)
            if event == "speech_end":
                recognizer.recognize(segment)
    """

    def __init__(self, config: VADSegmenterConfig) -> None:
        self.config = config
        self._state = "silence"
        self._speech_started = False

        # 滚动缓存（silence 时始终保留最近 pre_roll_samples）
        self._pre_roll_buffer: list[np.ndarray] = []
        self._pre_roll_samples = 0

        # 语音缓存
        self._speech_buffer: list[np.ndarray] = []
        self._total_samples = 0  # 当前 utterance 总样本数
        self._speech_samples = 0  # 其中认定为语音的样本数（排除尾部静音）
        self._silence_samples = 0  # 当前尾部连续静音样本数

    def accept_chunk(self, chunk: np.ndarray) -> tuple[str, np.ndarray | None, float]:
        """处理一个音频块。

        Args:
            chunk: float32 PCM，形状 (frames,) 或 (frames, channels)。

        Returns:
            (event_type, segment_audio, rms)
        """
        # 多通道转单声道
        if chunk.ndim > 1 and chunk.shape[1] > 1:
            mono = np.mean(chunk, axis=1).astype(chunk.dtype)
        else:
            mono = chunk.ravel()

        rms = calculate_rms(mono)
        is_speech = rms >= self.config.rms_threshold

        if self._state == "silence":
            return self._handle_silence(mono, rms, is_speech)
        else:
            return self._handle_speech(mono, rms, is_speech)

    def reset(self) -> None:
        """重置分段器状态（外部调用，如强制停止时）。"""
        self._state = "silence"
        self._speech_started = False
        self._pre_roll_buffer.clear()
        self._pre_roll_samples = 0
        self._speech_buffer.clear()
        self._total_samples = 0
        self._speech_samples = 0
        self._silence_samples = 0

    # ---- 内部 ----

    def _handle_silence(
        self, mono: np.ndarray, rms: float, is_speech: bool
    ) -> tuple[str, np.ndarray | None, float]:
        # 维护滚动缓存
        self._pre_roll_buffer.append(mono)
        self._pre_roll_samples += len(mono)
        while self._pre_roll_samples > self.config._pre_roll_samples and len(self._pre_roll_buffer) > 1:
            removed = self._pre_roll_buffer.pop(0)
            self._pre_roll_samples -= len(removed)

        if is_speech:
            # 检测到语音，进入 speech 状态
            self._state = "speech"
            self._speech_started = False
            self._speech_buffer = list(self._pre_roll_buffer)  # 包含预卷
            self._speech_buffer.append(mono)
            self._total_samples = self._pre_roll_samples + len(mono)
            self._speech_samples = len(mono)
            self._silence_samples = 0
            # 预卷缓存清空（已转移到 speech_buffer）
            self._pre_roll_buffer.clear()
            self._pre_roll_samples = 0

        return (_EVENT_SILENCE, None, rms)

    def _handle_speech(
        self, mono: np.ndarray, rms: float, is_speech: bool
    ) -> tuple[str, np.ndarray | None, float]:
        self._speech_buffer.append(mono)
        self._total_samples += len(mono)

        if is_speech:
            self._speech_samples += len(mono)
            self._silence_samples = 0
        else:
            self._silence_samples += len(mono)

        # 1) 超过最大语音长度 → 强制截断
        if self._total_samples >= self.config._max_speech_samples:
            segment = np.concatenate(self._speech_buffer)
            self.reset()
            return (_EVENT_SPEECH_FORCED_END, segment, rms)

        # 2) 尾部静音超时 → 语音结束
        if self._silence_samples >= self.config._silence_timeout_samples:
            speech_len = self._total_samples - self._silence_samples
            if speech_len >= self.config._min_speech_samples:
                # 去掉尾部静音
                segment = np.concatenate(self._speech_buffer)[:speech_len]
                self.reset()
                return (_EVENT_SPEECH_END, segment, rms)
            else:
                # 太短，丢弃
                self.reset()
                return (_EVENT_SILENCE, None, rms)

        # 3) 刚达到最短语音长度 → 触发 speech_start
        if not self._speech_started and self._speech_samples >= self.config._min_speech_samples:
            self._speech_started = True
            return (_EVENT_SPEECH_START, None, rms)

        # 4) 尚未达到 min_speech，仍算 silence
        if not self._speech_started:
            return (_EVENT_SILENCE, None, rms)

        return (_EVENT_SPEECHING, None, rms)
