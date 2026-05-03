"""测试麦克风相关工具函数。

不测试真实硬件，只测试 calculate_rms 等纯函数。
"""

import numpy as np
import pytest
from voice_agent.audio.microphone import calculate_rms


def test_rms_silence():
    """全零数据 RMS = 0。"""
    data = np.zeros((1600, 1), dtype=np.float32)
    assert calculate_rms(data) == 0.0


def test_rms_full_scale():
    """全 1.0 数据 RMS = 1.0。"""
    data = np.ones((1600, 1), dtype=np.float32)
    assert calculate_rms(data) == pytest.approx(1.0)


def test_rms_sine():
    """正弦波 RMS = amplitude / sqrt(2)。"""
    t = np.linspace(0, 1, 1600, endpoint=False)
    amplitude = 0.5
    data = (amplitude * np.sin(2 * np.pi * 440 * t)).reshape(-1, 1).astype(np.float32)
    expected = amplitude / np.sqrt(2)
    assert calculate_rms(data) == pytest.approx(expected, rel=0.01)


def test_rms_empty():
    """空数据 RMS = 0。"""
    data = np.array([]).astype(np.float32)
    assert calculate_rms(data) == 0.0


def test_rms_multichannel():
    """多通道求平均后的 RMS。"""
    data = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    # 每个样本 (1,0)，每个样本平方和=1，mean=0.5，sqrt=0.707
    assert calculate_rms(data) == pytest.approx(np.sqrt(0.5))


def test_rms_negative_values():
    """负数信号 RMS 应与绝对值相同。"""
    data = np.full((100, 1), -0.5, dtype=np.float32)
    assert calculate_rms(data) == pytest.approx(0.5)


def test_rms_mono_flat():
    """单通道无通道维度 (frames,) 也支持。"""
    data = np.full(1600, 0.3, dtype=np.float32)
    assert calculate_rms(data) == pytest.approx(0.3)


def test_rms_very_small():
    """极小声。"""
    data = np.full((1600, 1), 1e-6, dtype=np.float32)
    assert calculate_rms(data) == pytest.approx(1e-6, rel=0.01)
