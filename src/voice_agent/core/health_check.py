"""运行时健康检查 — 启动前检测配置和文件完整性。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class HealthItem:
    name: str
    ok: bool
    message: str
    level: str = "info"  # info / warning / error


@dataclass
class HealthReport:
    items: list[HealthItem] = field(default_factory=list)

    @property
    def has_error(self) -> bool:
        return any((not item.ok) and item.level == "error" for item in self.items)


def check_runtime_health(config: dict) -> HealthReport:
    """检查运行环境是否就绪，只检查配置和本地文件，不发起网络请求。"""
    report = HealthReport()

    _check_asr(report, config)
    _check_llm(report, config)
    _check_judge(report, config)
    _check_audio(report, config)

    return report


def _check_asr(report: HealthReport, config: dict) -> None:
    asr = config.get("asr", {})
    engine = asr.get("engine", "mock")

    if engine == "mock":
        report.items.append(HealthItem(
            name="ASR engine",
            ok=True,
            message="mock（键盘输入）",
            level="info",
        ))
        return

    sherpa = asr.get("sherpa_onnx", {})
    model_path = sherpa.get("model", "")
    tokens_path = sherpa.get("tokens", "")

    if model_path:
        exists = Path(model_path).exists()
        report.items.append(HealthItem(
            name="ASR model",
            ok=exists,
            message=f"{'已找到' if exists else '文件不存在'}: {model_path}",
            level="error" if not exists else "info",
        ))
    else:
        report.items.append(HealthItem(
            name="ASR model",
            ok=False,
            message="未配置 model 路径",
            level="error",
        ))

    if tokens_path:
        exists = Path(tokens_path).exists()
        report.items.append(HealthItem(
            name="ASR tokens",
            ok=exists,
            message=f"{'已找到' if exists else '文件不存在'}: {tokens_path}",
            level="error" if not exists else "info",
        ))
    else:
        report.items.append(HealthItem(
            name="ASR tokens",
            ok=False,
            message="未配置 tokens 路径",
            level="error",
        ))


def _check_llm(report: HealthReport, config: dict) -> None:
    llm = config.get("llm", {})
    if not llm.get("enabled", True):
        report.items.append(HealthItem(
            name="Main LLM",
            ok=True,
            message="已禁用",
            level="info",
        ))
        return

    api_base = llm.get("api_base", "")
    api_key = llm.get("api_key", "")
    model = llm.get("model", "")

    missing = []
    if not api_base:
        missing.append("api_base")
    if not api_key:
        missing.append("api_key")
    if not model:
        missing.append("model")

    if missing:
        report.items.append(HealthItem(
            name="Main LLM",
            ok=False,
            message=f"配置不完整，缺少: {', '.join(missing)}（可通过 .env 设置）",
            level="warning",
        ))
    else:
        report.items.append(HealthItem(
            name="Main LLM",
            ok=True,
            message=f"{model}",
            level="info",
        ))


def _check_judge(report: HealthReport, config: dict) -> None:
    judge = config.get("judge", {})
    provider = judge.get("provider", "rule")

    report.items.append(HealthItem(
        name="Judge provider",
        ok=True,
        message=provider,
        level="info",
    ))

    if provider == "local":
        local = judge.get("local", {})
        api_base = local.get("api_base", "")
        model = local.get("model", "")

        missing = []
        if not api_base:
            missing.append("api_base")
        if not model:
            missing.append("model")

        if missing:
            report.items.append(HealthItem(
                name="Local Judge",
                ok=False,
                message=f"配置不完整，缺少: {', '.join(missing)}",
                level="warning",
            ))
        else:
            report.items.append(HealthItem(
                name="Local Judge",
                ok=True,
                message=f"{model} @ {api_base}",
                level="info",
            ))


def _check_audio(report: HealthReport, config: dict) -> None:
    audio = config.get("audio", {})
    device = audio.get("device")

    if device is not None:
        report.items.append(HealthItem(
            name="Audio device",
            ok=True,
            message=f"ID: {device}",
            level="info",
        ))
    else:
        report.items.append(HealthItem(
            name="Audio device",
            ok=True,
            message="未指定，使用系统默认",
            level="info",
        ))
