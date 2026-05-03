"""LLM 客户端：OpenAI-compatible Chat Completions 接口。"""

import json
from typing import Any

import httpx

JUDGE_PROMPT = """你是一个常驻语音助手的介入判断器。

你的任务不是回答用户，而是判断 AI 是否应该主动回应。

用户没有使用唤醒词，所以你要谨慎。
只有当用户明显在提问、请求帮助、表达需要协助、或正在和 AI 连续对话时，才回应。
如果用户只是自言自语、普通聊天、背景对白、情绪感叹，默认不回应。

请只输出 JSON，不要输出其他内容。

输入：
当前状态：{state}
最近 AI 是否刚回应过：{recent_agent_reply}
用户语音识别文本：{text}

输出格式：
{{
  "should_reply": true或false,
  "confidence": 0到1,
  "reason": "简短原因",
  "response_mode": "silent | bubble | text_reply | voice_reply"
}}"""

AGENT_PROMPT = """你是一个常驻语音助手，像一个安静但可靠的管家/宠物。

你平时不会主动打扰用户。
现在用户的语音被判断为需要回应。

要求：
1. 回复要简短自然。
2. 不要解释系统内部判断。
3. 不要说"我检测到你说了"。
4. 如果用户只是问问题，直接回答。
5. 如果用户请求执行工具，但当前系统还没有工具能力，要说明可以先提供建议。
6. 不要自动执行高风险操作。
7. 回复中文。

用户刚才说：
{text}

最近上下文：
{recent_context}"""


class LLMClient:
    """OpenAI-compatible 大模型客户端。"""

    def __init__(
        self,
        enabled: bool = True,
        api_base: str = "",
        api_key: str = "",
        model: str = "",
        timeout_seconds: int = 30,
    ) -> None:
        self.enabled = enabled
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout_seconds
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout))
        return self._client

    @property
    def is_available(self) -> bool:
        return self.enabled and bool(self.api_base) and bool(self.api_key)

    async def chat(self, messages: list[dict[str, str]]) -> str:
        """发送 Chat Completions 请求，返回模型回复文本。"""
        if not self.is_available:
            return self._mock_reply(messages)

        client = await self._get_client()
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
        }

        try:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[LLM 调用失败: {e}]"

    async def judge_intervention(self, text: str, state_mode: str, recent_reply: bool) -> dict[str, Any]:
        """调用轻量判断：是否需要回复用户。"""
        prompt = JUDGE_PROMPT.format(
            state=state_mode,
            recent_agent_reply="是" if recent_reply else "否",
            text=text,
        )
        messages = [
            {"role": "system", "content": "你是一个精准的介入判断器。"},
            {"role": "user", "content": prompt},
        ]

        if not self.is_available:
            # Mock: 默认回复
            return {"should_reply": True, "confidence": 0.7, "reason": "mock judge", "response_mode": "text_reply"}

        raw = await self.chat(messages)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"should_reply": False, "confidence": 0.0, "reason": f"解析失败: {raw[:100]}", "response_mode": "silent"}

    async def generate_reply(self, text: str, recent_context: str = "") -> str:
        """生成正式回复。"""
        prompt = AGENT_PROMPT.format(
            text=text,
            recent_context=recent_context if recent_context else "无",
        )
        messages = [
            {"role": "system", "content": "你是一个安静可靠的语音助手。"},
            {"role": "user", "content": prompt},
        ]

        return await self.chat(messages)

    def _mock_reply(self, messages: list[dict[str, str]]) -> str:
        user_msg = ""
        for m in messages:
            if m.get("role") == "user":
                user_msg = m.get("content", "")
                break

        if "判断" in user_msg or "介入" in user_msg:
            return json.dumps({
                "should_reply": True,
                "confidence": 0.7,
                "reason": "mock judge: 默认介入",
                "response_mode": "text_reply",
            }, ensure_ascii=False)

        return "这是一个模拟回复。当 LLM 配置正确后，我将提供真实回答。"

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
