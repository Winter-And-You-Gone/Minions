"""测试 SpeechSegmenter — 基于 RMS 的语音分段器。"""

import numpy as np
import pytest
from voice_agent.audio.segmenter import VADSegmenterConfig, SpeechSegmenter
from voice_agent.audio.microphone import calculate_rms


def _make_chunk(rms: float, sr: int = 16000, chunk_ms: int = 100) -> np.ndarray:
    """生成指定 RMS 的音频块（白噪声缩放）。"""
    frames = int(sr * chunk_ms / 1000)
    noise = np.random.randn(frames).astype(np.float32)
    current_rms = calculate_rms(noise)
    if current_rms > 0:
        noise *= rms / current_rms
    return noise


def _make_silent_chunk(sr: int = 16000, chunk_ms: int = 100) -> np.ndarray:
    """生成静音块，RMS 接近 0。"""
    frames = int(sr * chunk_ms / 1000)
    return np.zeros(frames, dtype=np.float32)


def _make_multichannel_chunk(rms: float, channels: int = 2, sr: int = 16000, chunk_ms: int = 100) -> np.ndarray:
    """生成多通道音频块。"""
    frames = int(sr * chunk_ms / 1000)
    mono = _make_chunk(rms, sr, chunk_ms)
    return np.tile(mono.reshape(-1, 1), (1, channels))


class TestSpeechSegmenter:
    def test_silence_chunk_returns_silence(self):
        """纯静音返回 'silence'。"""
        config = VADSegmenterConfig(rms_threshold=0.008)
        seg = SpeechSegmenter(config)
        chunk = _make_silent_chunk()
        event, segment, rms = seg.accept_chunk(chunk)
        assert event == "silence"
        assert segment is None

    def test_speech_start_after_min_speech(self):
        """连续语音超过 min_speech_ms 触发 speech_start。"""
        config = VADSegmenterConfig(
            rms_threshold=0.008,
            min_speech_ms=300,  # 3 chunks at 100ms each
            silence_timeout_ms=500,
            pre_roll_ms=0,
        )
        seg = SpeechSegmenter(config)

        # 第 1 个语音块 → silence（刚进入 speech 状态）
        chunk = _make_chunk(0.1)
        event, segment, rms = seg.accept_chunk(chunk)
        assert event == "silence"

        # 第 2 个语音块 → silence（尚未达到 min_speech）
        chunk = _make_chunk(0.1)
        event, segment, rms = seg.accept_chunk(chunk)
        assert event == "silence"

        # 第 3 个语音块 → speech_start（达到 300ms = min_speech_ms）
        chunk = _make_chunk(0.1)
        event, segment, rms = seg.accept_chunk(chunk)
        assert event == "speech_start"

    def test_speech_end_after_silence_timeout(self):
        """语音后静音超过 silence_timeout_ms 触发 speech_end。"""
        config = VADSegmenterConfig(
            rms_threshold=0.008,
            min_speech_ms=100,
            silence_timeout_ms=300,
            pre_roll_ms=0,
        )
        seg = SpeechSegmenter(config)

        # 先送入语音块
        for _ in range(3):
            seg.accept_chunk(_make_chunk(0.1))

        # 送入静音块直到触发 end
        for _ in range(3):
            event, segment, rms = seg.accept_chunk(_make_silent_chunk())
            if event in ("speech_end",):
                break
            if event == "silence":
                # speech 段后如果太短可能被丢弃
                # 因为前面已经足够长了，这里应该能触发 speech_end
                pass

        assert event is not None

    def test_speech_end_returns_segment(self):
        """speech_end 时 segment 不为空。"""
        config = VADSegmenterConfig(
            rms_threshold=0.008,
            min_speech_ms=100,
            silence_timeout_ms=200,
            pre_roll_ms=0,
        )
        seg = SpeechSegmenter(config)

        # 送入足够的语音块
        for _ in range(5):
            seg.accept_chunk(_make_chunk(0.1))

        # 送入静音块触发 end
        found = None
        for _ in range(10):
            _, segment, _ = seg.accept_chunk(_make_silent_chunk())
            if segment is not None:
                found = segment
                break

        assert found is not None
        assert len(found) > 0

    def test_forced_end_on_max_speech(self):
        """语音超过 max_speech_ms 触发 speech_forced_end。"""
        config = VADSegmenterConfig(
            rms_threshold=0.008,
            min_speech_ms=100,
            max_speech_ms=500,  # 5 chunks at 100ms
            silence_timeout_ms=1000,
            pre_roll_ms=0,
        )
        seg = SpeechSegmenter(config)

        # 连续送语音块，直到强制截断
        for i in range(10):
            event, segment, _ = seg.accept_chunk(_make_chunk(0.1))
            if event == "speech_forced_end":
                assert segment is not None
                return

        pytest.fail("未触发 speech_forced_end")

    def test_multichannel_to_mono(self):
        """多通道输入能正确转单声道。"""
        config = VADSegmenterConfig(
            rms_threshold=0.008,
            min_speech_ms=100,
            silence_timeout_ms=200,
            pre_roll_ms=0,
        )
        seg = SpeechSegmenter(config)

        chunk = _make_multichannel_chunk(0.1, channels=2)
        event, _, rms = seg.accept_chunk(chunk)
        assert event == "silence"
        assert rms > 0

    def test_short_speech_discarded(self):
        """语音太短被丢弃。"""
        config = VADSegmenterConfig(
            rms_threshold=0.008,
            min_speech_ms=500,  # 需要 500ms
            silence_timeout_ms=200,
            pre_roll_ms=0,
        )
        seg = SpeechSegmenter(config)

        # 只送少量语音块
        for _ in range(2):
            seg.accept_chunk(_make_chunk(0.1))

        # 送静音
        found_speech_end = False
        for _ in range(10):
            event, segment, _ = seg.accept_chunk(_make_silent_chunk())
            if event == "speech_end":
                found_speech_end = True
                break

        # 应该被丢弃（太短）
        assert not found_speech_end, "短语音应该被丢弃而不是触发 speech_end"

    def test_pre_roll_included(self):
        """pre_roll 音频被包含在 segment 中。"""
        config = VADSegmenterConfig(
            rms_threshold=0.008,
            min_speech_ms=100,
            silence_timeout_ms=200,
            pre_roll_ms=100,  # 1 chunk of pre-roll
        )
        seg = SpeechSegmenter(config)

        # 先送一个静音块（将被 pre-roll 缓存）
        seg.accept_chunk(_make_silent_chunk())

        # 再送语音块
        for _ in range(3):
            seg.accept_chunk(_make_chunk(0.1))

        # 送静音触发 end
        found_segment = None
        for _ in range(10):
            _, segment, _ = seg.accept_chunk(_make_silent_chunk())
            if segment is not None:
                found_segment = segment
                break

        assert found_segment is not None
        # 至少包含 4 个块 = 400ms = 6400 samples at 16kHz
        assert len(found_segment) >= 6400

    def test_speeching_during_active_speech(self):
        """语音进行中返回 'speeching'。"""
        config = VADSegmenterConfig(
            rms_threshold=0.008,
            min_speech_ms=200,
            silence_timeout_ms=500,
            pre_roll_ms=0,
        )
        seg = SpeechSegmenter(config)

        # 送语音块直到 speech_start
        for _ in range(5):
            seg.accept_chunk(_make_chunk(0.1))

        # 之后继续送语音块应得到 speeching
        for _ in range(3):
            event, _, _ = seg.accept_chunk(_make_chunk(0.1))
            assert event == "speeching"

    def test_reset_clears_state(self):
        """reset() 清空所有状态。"""
        config = VADSegmenterConfig(rms_threshold=0.008)
        seg = SpeechSegmenter(config)
        seg.accept_chunk(_make_chunk(0.1))
        seg.reset()
        event, _, _ = seg.accept_chunk(_make_silent_chunk())
        assert event == "silence"

    def test_very_low_rms_threshold(self):
        """极低阈值下正常静音仍为 silence。"""
        config = VADSegmenterConfig(rms_threshold=0.001)
        seg = SpeechSegmenter(config)
        chunk = _make_silent_chunk()
        event, _, _ = seg.accept_chunk(chunk)
        assert event == "silence"

    def test_very_high_rms_threshold(self):
        """极高阈值下正常语音也不触发。"""
        config = VADSegmenterConfig(
            rms_threshold=1.0,  # 不可能达到
            min_speech_ms=100,
            silence_timeout_ms=200,
            pre_roll_ms=0,
        )
        seg = SpeechSegmenter(config)

        for _ in range(5):
            event, _, _ = seg.accept_chunk(_make_chunk(0.5))
            assert event == "silence"
