"""测试运行时健康检查。"""

from voice_agent.core.health_check import (
    HealthItem,
    HealthReport,
    check_runtime_health,
)


def test_health_check_handles_missing_asr_model():
    config = {
        "asr": {
            "engine": "sherpa-onnx",
            "sherpa_onnx": {
                "model": "missing.onnx",
                "tokens": "missing.txt",
            },
        },
        "llm": {},
        "judge": {},
        "audio": {},
    }
    report = check_runtime_health(config)
    assert report.items
    assert report.has_error


def test_health_check_mock_asr_ok():
    config = {
        "asr": {"engine": "mock"},
        "llm": {},
        "judge": {},
        "audio": {},
    }
    report = check_runtime_health(config)
    assert report.items
    assert not report.has_error


def test_health_report_has_error():
    items = [
        HealthItem(name="test", ok=False, message="fail", level="error"),
    ]
    report = HealthReport(items=items)
    assert report.has_error


def test_health_report_no_error():
    items = [
        HealthItem(name="test", ok=True, message="ok", level="info"),
        HealthItem(name="warn", ok=False, message="warning", level="warning"),
    ]
    report = HealthReport(items=items)
    assert not report.has_error


def test_health_check_llm_warning_on_missing_config():
    config = {
        "asr": {"engine": "mock"},
        "llm": {"enabled": True, "api_base": "", "api_key": "", "model": ""},
        "judge": {},
        "audio": {},
    }
    report = check_runtime_health(config)
    llm_items = [i for i in report.items if i.name == "Main LLM"]
    assert llm_items
    assert not llm_items[0].ok
    assert llm_items[0].level == "warning"


def test_health_item_dataclass():
    item = HealthItem(name="test", ok=True, message="一切正常")
    assert item.name == "test"
    assert item.ok is True
    assert item.level == "info"
