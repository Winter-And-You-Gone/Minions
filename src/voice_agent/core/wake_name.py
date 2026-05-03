"""唤醒名检测 — 检测用户是否喊了 AI 的名字。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from voice_agent.utils.text_normalizer import normalize_text, remove_chinese_spaces


@dataclass
class WakeNameConfig:
    name: str = "米粒"
    aliases: list[str] = field(default_factory=list)
    enabled: bool = True
    session_seconds: float = 120.0
    silence_timeout_seconds: float = 90.0
    strip_wake_name: bool = True
    allow_llm_turn_away_judge: bool = True


@dataclass
class WakeNameMatch:
    matched: bool
    name: str = ""
    alias: str = ""
    text_without_name: str = ""
    reason: str = ""


def _normalize_wake_text(text: str) -> str:
    text = remove_chinese_spaces(text)
    text = normalize_text(text)
    text = re.sub(r"[，。！？!?、,.\s]", "", text)
    return text


class WakeNameMatcher:
    def __init__(self, config: WakeNameConfig) -> None:
        self.config = config

    @classmethod
    def from_config(cls, config: dict) -> "WakeNameMatcher":
        assistant_cfg = config.get("assistant", {})
        wake_cfg = assistant_cfg.get("wake", {})

        name = assistant_cfg.get("name", "米粒")
        aliases = assistant_cfg.get("wake_aliases", [])
        all_aliases = [name, *aliases]

        deduped = []
        for item in all_aliases:
            if item and item not in deduped:
                deduped.append(item)

        return cls(WakeNameConfig(
            name=name,
            aliases=deduped,
            enabled=wake_cfg.get("enabled", True),
            session_seconds=float(wake_cfg.get("session_seconds", 120)),
            silence_timeout_seconds=float(wake_cfg.get("silence_timeout_seconds", 90)),
            strip_wake_name=bool(wake_cfg.get("strip_wake_name", True)),
            allow_llm_turn_away_judge=bool(wake_cfg.get("allow_llm_turn_away_judge", True)),
        ))

    def detect(self, raw_text: str) -> WakeNameMatch:
        if not self.config.enabled:
            return WakeNameMatch(False)

        text = _normalize_wake_text(raw_text)

        for alias in self.config.aliases:
            normalized_alias = _normalize_wake_text(alias)
            if not normalized_alias:
                continue

            # 规则 1：句首喊名字
            if text.startswith(normalized_alias):
                rest = text[len(normalized_alias):]
                return WakeNameMatch(
                    matched=True,
                    name=self.config.name,
                    alias=alias,
                    text_without_name=rest,
                    reason=f"句首唤醒名: {alias}",
                )

            # 规则 2：短句只喊名字
            if text == normalized_alias:
                return WakeNameMatch(
                    matched=True,
                    name=self.config.name,
                    alias=alias,
                    text_without_name="",
                    reason=f"单独唤醒名: {alias}",
                )

            # 规则 3：名字在前几个字内，兼容 ASR 前面多出语气词
            prefixes = ["嗯", "那个", "你好", "喂"]
            for p in prefixes:
                pp = _normalize_wake_text(p)
                if text.startswith(pp + normalized_alias):
                    rest = text[len(pp + normalized_alias):]
                    return WakeNameMatch(
                        matched=True,
                        name=self.config.name,
                        alias=alias,
                        text_without_name=rest,
                        reason=f"前缀后唤醒名: {p}+{alias}",
                    )

        return WakeNameMatch(False)
