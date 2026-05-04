"""测试配置写回。"""

from pathlib import Path

from voice_agent.config import get_config, save_config, reload_config


def test_save_config_roundtrip(tmp_path):
    path = tmp_path / "config.yaml"
    config = {
        "assistant": {
            "name": "测试名",
            "user_title": "少爷",
            "wake_aliases": ["测试名"],
            "wake": {"enabled": True},
        }
    }

    save_config(config, str(path))
    loaded = reload_config(str(path))

    assert loaded["assistant"]["name"] == "测试名"
    assert loaded["assistant"]["wake_aliases"] == ["测试名"]


def test_save_config_preserves_other_fields(tmp_path):
    path = tmp_path / "config.yaml"
    config = {
        "app": {"name": "Minions"},
        "assistant": {"name": "琉璃川"},
        "audio": {"device": 1},
    }

    save_config(config, str(path))
    loaded = reload_config(str(path))

    assert loaded["app"]["name"] == "Minions"
    assert loaded["assistant"]["name"] == "琉璃川"
    assert loaded["audio"]["device"] == 1
