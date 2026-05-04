"""测试默认配置变化。"""

from voice_agent.config import reload_config


def test_default_config_uses_sherpa():
    config = reload_config("config.yaml")
    assert config["asr"]["engine"] == "sherpa-onnx"
    assert config["asr"]["sherpa_onnx"]["enabled"] is True


def test_default_config_uses_local_judge():
    config = reload_config("config.yaml")
    assert config["judge"]["provider"] == "local"
    assert config["judge"]["local"]["model"] == "qwen3.5:4b"


def test_default_config_app_name():
    config = reload_config("config.yaml")
    assert config["app"]["name"] == "Minions"


def test_default_config_has_default_mode():
    config = reload_config("config.yaml")
    assert config["app"]["default_mode"] == "tui"
