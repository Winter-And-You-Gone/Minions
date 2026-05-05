"""本地小模型 Judge — 使用 Ollama + qwen3.5:4b 做语义判断。

只做判断，不生成正式回复。正式回复仍然走主 LLM 的琉璃川人格 Prompt。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from voice_agent.logger import get_logger

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```")

LOCAL_JUDGE_PROMPT = """你是一个常驻语音助手的本地判断器。

你的任务不是回答用户，也不是扮演角色。
你的任务是判断：当前这句话是否应该交给 AI 正式回复。

AI 的名字：{assistant_name}
用户称呼：{user_title}

你需要根据：
1. 当前是否处于唤醒会话
2. 用户是否刚喊过 AI 名字
3. 最近对话上下文
4. 当前 ASR 文本
5. 初步规则判断结果

判断当前用户语音的目标对象。

target 只能是：
- ai：用户在对 AI 说话
- other：用户明显在对别人说话
- background：电视、视频、环境对白
- self_talk：自言自语或普通吐槽
- unclear：不确定

判断规则：
- 如果处于唤醒会话，短追问、补充、命令、问题通常是对 AI 说话。
- 如果用户说"继续""然后呢""那怎么修""详细说"，通常是对 AI 说话。
- 如果用户明显说"没事了""不用了""不是跟你说"，应该结束唤醒会话。
- 如果明显是电视对白、别人说话、对另一个人说话，不应该回复。
- 如果不确定，优先不要回复，除非处于唤醒会话且像追问。

请严格输出 JSON，不要输出其他内容。

输入：
当前状态：{state}
是否唤醒会话中：{wake_session_active}
最近上下文：{recent_context}
当前文本：{text}
规则 Gate 初步 action：{gate_action}
规则 Gate 分数：{score}
规则 Gate 原因：{reason}

输出格式：
{{
  "target": "ai | other | background | self_talk | unclear",
  "should_reply": true,
  "should_end_wake_session": false,
  "confidence": 0.0,
  "reason": "简短原因"
}}
"""


@dataclass
class LocalJudgeResult:
    target: str
    should_reply: bool
    should_end_wake_session: bool
    confidence: float
    reason: str
    raw: str = ""
    elapsed_ms: int = 0


class LocalJudgeClient:
    """本地小模型 Judge，只做语义判断不生成回复。"""

    def __init__(
        self,
        enabled: bool = True,
        api_base: str = "http://127.0.0.1:11434/v1",
        api_key: str = "ollama",
        model: str = "qwen3.5:4b",
        timeout_seconds: int = 6,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> None:
        self.enabled = enabled
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: httpx.AsyncClient | None = None
        self._logger = get_logger()

    @property
    def is_available(self) -> bool:
        return self.enabled and bool(self.api_base) and bool(self.model)

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_seconds))
        return self._client

    def _extract_json(self, raw: str) -> str:
        match = _JSON_BLOCK_RE.search(raw)
        if match:
            return match.group(1).strip()
        return raw.strip()

    async def judge(
        self,
        *,
        text: str,
        state: str,
        wake_session_active: bool,
        recent_context: str,
        gate_action: str,
        score: int,
        reason: str,
        assistant_name: str = "琉璃川",
        user_title: str = "少爷",
    ) -> LocalJudgeResult:
        """调用本地模型判断用户语音目标对象。"""
        started = time.perf_counter()

        if not self.is_available:
            return LocalJudgeResult(
                target="unclear",
                should_reply=False,
                should_end_wake_session=False,
                confidence=0.0,
                reason="local judge 不可用",
                elapsed_ms=0,
            )

        prompt = LOCAL_JUDGE_PROMPT.format(
            assistant_name=assistant_name,
            user_title=user_title,
            state=state,
            wake_session_active="是" if wake_session_active else "否",
            recent_context=recent_context if recent_context else "无",
            text=text,
            gate_action=gate_action,
            score=score,
            reason=reason,
        )

        messages = [
            {
                "role": "system",
                "content": "你是一个严格的 JSON 分类器。只输出 JSON，不要解释。",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        raw = await self._chat(messages)
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        try:
            data = json.loads(self._extract_json(raw))
            return LocalJudgeResult(
                target=str(data.get("target", "unclear")),
                should_reply=bool(data.get("should_reply", False)),
                should_end_wake_session=bool(data.get("should_end_wake_session", False)),
                confidence=float(data.get("confidence", 0.0)),
                reason=str(data.get("reason", "")),
                raw=raw,
                elapsed_ms=elapsed_ms,
            )
        except Exception as e:
            self._logger.warning("[LocalJudge] JSON 解析失败: %s raw=%s", e, raw[:200])
            return LocalJudgeResult(
                target="unclear",
                should_reply=False,
                should_end_wake_session=False,
                confidence=0.0,
                reason=f"JSON 解析失败: {e}",
                raw=raw,
                elapsed_ms=elapsed_ms,
            )

    async def _chat(self, messages: list[dict[str, str]]) -> str:
        client = await self._get_client()
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }

        try:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            self._logger.warning("[LocalJudge] 调用失败: %s", e)
            return f"[LocalJudge 调用失败: {e}]"

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
