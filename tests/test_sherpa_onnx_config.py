"""测试 SherpaOnnxASR 配置校验（无真实模型）。"""

from pathlib import Path
import tempfile

import pytest

# 尝试导入，如果没有 sherpa-onnx 则跳过真实模型测试
sherpa_onnx = pytest.importorskip("sherpa_onnx", reason="需要 sherpa-onnx 包")


def _make_minimal_config() -> dict:
    """生成可通过基本校验的最小配置。"""
    tmp = tempfile.gettempdir()
    # 创建空 tokens 文件
    tokens_path = Path(tmp) / "test_tokens.txt"
    if not tokens_path.exists():
        tokens_path.write_text("a b c\n", encoding="utf-8")
    # 创建空模型文件
    model_path = Path(tmp) / "test_model.int8.onnx"
    if not model_path.exists():
        model_path.write_bytes(b"dummy")
    return {
        "sample_rate": 16000,
        "tokens": str(tokens_path),
        "model": str(model_path),
        "num_threads": 1,
        "decoding_method": "greedy_search",
        "provider": "cpu",
    }


class TestSherpaOnnxConfigValidation:
    """配置校验测试（不加载真实模型，只测试校验逻辑）。"""

    def test_tokens_missing_raises_error(self):
        """tokens 未设置时抛出清晰错误。"""
        from voice_agent.asr.sherpa_onnx_asr import SherpaOnnxASR
        from voice_agent.event_bus import EventBus

        bus = EventBus()
        asr = SherpaOnnxASR(bus, {"sample_rate": 16000, "tokens": ""})
        with pytest.raises(ValueError, match="tokens"):
            asr._validate_config()

    def test_tokens_path_not_exists_raises_error(self):
        """tokens 文件不存在时抛出清晰错误。"""
        from voice_agent.asr.sherpa_onnx_asr import SherpaOnnxASR
        from voice_agent.event_bus import EventBus

        bus = EventBus()
        asr = SherpaOnnxASR(bus, {"sample_rate": 16000, "tokens": "/nonexistent/tokens.txt"})
        with pytest.raises(ValueError, match="tokens 文件不存在"):
            asr._validate_config()

    def test_model_missing_raises_error(self):
        """model 和 encoder/decoder/joiner 都缺失时抛出错误。"""
        from voice_agent.asr.sherpa_onnx_asr import SherpaOnnxASR
        from voice_agent.event_bus import EventBus

        bus = EventBus()
        cfg = _make_minimal_config()
        cfg["model"] = ""
        asr = SherpaOnnxASR(bus, cfg)
        with pytest.raises(ValueError, match="请设置 model"):
            asr._validate_config()

    def test_model_path_not_exists_raises_error(self):
        """模型文件不存在时抛出错误。"""
        from voice_agent.asr.sherpa_onnx_asr import SherpaOnnxASR
        from voice_agent.event_bus import EventBus

        bus = EventBus()
        cfg = _make_minimal_config()
        cfg["model"] = "/nonexistent/model.onnx"
        asr = SherpaOnnxASR(bus, cfg)
        with pytest.raises(ValueError, match="模型文件不存在"):
            asr._validate_config()

    def test_valid_config_passes_validation(self):
        """有效配置通过校验。"""
        from voice_agent.asr.sherpa_onnx_asr import SherpaOnnxASR
        from voice_agent.event_bus import EventBus

        bus = EventBus()
        cfg = _make_minimal_config()
        asr = SherpaOnnxASR(bus, cfg)
        # 不抛出异常即通过
        asr._validate_config()

    def test_transducer_config_valid(self):
        """transducer 模型配置（encoder/decoder/joiner）通过校验。"""
        from voice_agent.asr.sherpa_onnx_asr import SherpaOnnxASR
        from voice_agent.event_bus import EventBus

        tmp = tempfile.gettempdir()
        bus = EventBus()
        cfg = _make_minimal_config()
        cfg["model"] = ""
        cfg["encoder"] = str(Path(tmp) / "test_model.int8.onnx")
        cfg["decoder"] = str(Path(tmp) / "test_model.int8.onnx")
        cfg["joiner"] = str(Path(tmp) / "test_model.int8.onnx")
        asr = SherpaOnnxASR(bus, cfg)
        asr._validate_config()

    def test_sample_rate_zero_raises_error(self):
        """sample_rate 为 0 时抛出错误。"""
        from voice_agent.asr.sherpa_onnx_asr import SherpaOnnxASR
        from voice_agent.event_bus import EventBus

        bus = EventBus()
        cfg = _make_minimal_config()
        cfg["sample_rate"] = 0
        asr = SherpaOnnxASR(bus, cfg)
        with pytest.raises(ValueError, match="sample_rate"):
            asr._validate_config()
