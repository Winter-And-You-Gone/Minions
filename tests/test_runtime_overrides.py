"""测试运行时配置覆盖。"""

from voice_agent.main import apply_runtime_overrides


def test_apply_vad_threshold_override():
    config = {
        "asr": {
            "sherpa_onnx": {
                "vad": {
                    "rms_threshold": 0.008,
                }
            }
        }
    }
    updated = apply_runtime_overrides(config, vad_threshold=0.006)
    assert updated["asr"]["sherpa_onnx"]["vad"]["rms_threshold"] == 0.006


def test_apply_vad_threshold_override_creates_path():
    config = {}
    updated = apply_runtime_overrides(config, vad_threshold=0.006)
    assert updated["asr"]["sherpa_onnx"]["vad"]["rms_threshold"] == 0.006


def test_no_override_when_none():
    config = {"asr": {"sherpa_onnx": {"vad": {"rms_threshold": 0.008}}}}
    updated = apply_runtime_overrides(config, vad_threshold=None)
    assert updated["asr"]["sherpa_onnx"]["vad"]["rms_threshold"] == 0.008
