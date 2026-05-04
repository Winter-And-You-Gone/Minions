"""LLM 客户端：OpenAI-compatible Chat Completions 接口。"""

import json
import re
from typing import Any

import httpx

from voice_agent.logger import get_logger

JUDGE_PROMPT = """你是一个常驻语音助手琉璃川的介入判断器。

你的任务不是回答用户，而是判断 AI（琉璃川）是否应该主动回应。

用户可能正在和琉璃川对话（唤醒会话中），也可能没有喊名字。

判断原则：
1. 如果用户明显在提问、请求帮助、表达需要协助，→ should_reply=true。
2. 如果用户正在和 AI 连续对话（追问、补充），→ should_reply=true。
3. 如果用户只是自言自语、普通聊天、背景对白、情绪感叹，→ should_reply=false。
4. 如果用户明显在和别人说话，→ should_reply=false。
5. 如果不确定，→ should_reply=false（宁可漏判不要误判）。

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

AGENT_PROMPT = """你是{assistant_name}，一位 16 岁的少女女仆。

{user_title}正在用语音和你对话，请用自然、温暖、略带活泼的语气回应。
你是女仆，但不要刻意堆砌刻板的 anime 属性——自然就好。

## 核心设定
- 名字：{assistant_name}（默认「琉璃川」，但用户可能喊你「琉璃」「六里川」「琉璃酱」）
- 年龄：16 岁
- 身份：{user_title} 的专属女仆
- 称呼用户：{user_title}
- 语气：温柔中带一点活泼，有少女感但不幼稚，熟悉后可以偶尔嘴上不饶人（傲娇）
- 背景：对技术有一定了解，能辅助 {user_title} 处理工作事务
- 当前连接模型：{model}

## 对话风格
1. 回复简短自然（语音场景，不要太长）。
2. 不要解释系统内部判断，不要提「语音识别」「我的判断机制」之类。
3. 不要说「我检测到你说了……」，直接回答就好。
4. 如果是问题，直接回答；如果不懂，直接说不知道，不要编。
5. 如果要执行操作（打开/关闭/搜索等），但系统目前没有工具能力，可以说「我暂时还不能做这个，但可以帮你查一下/给你建议」。
6. 如果用户只是随口吐槽（「好烦」「离谱」），可以共情安慰，不用解决问题。
7. 回复中文。
8. 语音输入可能有同音错字，可以结合语境理解。

## 当前状态
- 用户文本：{text}
- 最近对话：{recent_context}
- 系统状态：{state}
- 是否唤醒会话中：{wake_session_active}
- 你对 {user_title} 的称呼：{user_title}"""

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```")


class LLMClient:
    """OpenAI-compatible 大模型客户端。"""

    def __init__(
        self,
        enabled: bool = True,
        api_base: str = "",
        api_key: str = "",
        model: str = "",
        timeout_seconds: int = 30,
        mock_judge_reply: bool = False,
    ) -> None:
        self.enabled = enabled
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout_seconds
        self.mock_judge_reply = mock_judge_reply
        self._client: httpx.AsyncClient | None = None
        self._logger = get_logger()

        if self.enabled and not self.model:
            self._logger.warning(
                "[LLM] enabled=true 但 model 未配置，将使用 mock 模式"
            )
        if self.enabled and not self.api_key:
            self._logger.warning(
                "[LLM] enabled=true 但 api_key 未配置，将使用 mock 模式"
            )

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout))
        return self._client

    @property
    def is_available(self) -> bool:
        return self.enabled and bool(self.api_base) and bool(self.api_key) and bool(self.model)

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
            return {
                "should_reply": self.mock_judge_reply,
                "confidence": 0.5 if self.mock_judge_reply else 0.0,
                "reason": "mock judge (LLM 不可用)",
                "response_mode": "text_reply" if self.mock_judge_reply else "silent",
            }

        raw = await self.chat(messages)
        json_str = self._extract_json(raw)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {"should_reply": False, "confidence": 0.0, "reason": f"解析失败: {raw[:100]}", "response_mode": "silent"}

    def _extract_json(self, raw: str) -> str:
        """从 LLM 回复中提取 JSON 字符串，兼容 ```json 代码块。"""
        match = _JSON_BLOCK_RE.search(raw)
        if match:
            return match.group(1).strip()
        return raw.strip()

    async def generate_reply(
        self,
        text: str,
        recent_context: str = "",
        state: str = "",
        wake_session_active: bool = False,
        assistant_name: str = "琉璃川",
        user_title: str = "少爷",
    ) -> str:
        """生成正式回复。"""
        prompt = AGENT_PROMPT.format(
            text=text,
            recent_context=recent_context if recent_context else "无",
            state=state or "未知",
            wake_session_active="是" if wake_session_active else "否",
            assistant_name=assistant_name,
            user_title=user_title,
            model=self.model or "未配置",
        )
        messages = [
            {"role": "system", "content": f"你是{assistant_name}，一位16岁的少女女仆。"},
            {"role": "user", "content": prompt},
        ]

        return await self.chat(messages)

    async def judge_wake_session_continue(
        self,
        text: str,
        recent_context: str = "",
    ) -> dict:
        """判断用户是否还在对 AI 说话。"""
        if not self.is_available:
            return {
                "continue_session": True,
                "confidence": 0.5,
                "reason": "LLM 不可用，默认继续唤醒会话",
            }

        messages = [
            {
                "role": "system",
                "content": (
                    "你是琉璃川（语音助手的会话判断器）。"
                    "用户之前已经喊过 AI 的名字（琉璃川/琉璃/六里川/琉璃酱），进入了连续对话。"
                    "现在请判断用户这句话是否仍然是在对琉璃川说话。"
                    "如果明显是在和别人说话、看电视对白、或说不用 AI 了（没事了/不用了/先这样/闭嘴），则continue_session=false。"
                    "如果是追问、补充、命令、问题、短句如'然后呢''继续'，则continue_session=true。"
                    "只输出 JSON，不要输出其他内容。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"最近上下文：{recent_context}\n"
                    f"当前用户语音：{text}\n\n"
                    "输出格式："
                    '{"continue_session": true, "confidence": 0.0, "reason": "简短原因"}'
                ),
            },
        ]

        raw = await self.chat(messages)
        json_str = self._extract_json(raw)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {
                "continue_session": True,
                "confidence": 0.3,
                "reason": f"解析失败，默认继续: {raw[:100]}",
            }

    def _mock_reply(self, messages: list[dict[str, str]]) -> str:
        user_msg = ""
        for m in messages:
            if m.get("role") == "user":
                user_msg = m.get("content", "")
                break

        if "判断" in user_msg or "介入" in user_msg:
            return json.dumps({
                "should_reply": self.mock_judge_reply,
                "confidence": 0.5 if self.mock_judge_reply else 0.0,
                "reason": "mock judge (LLM 不可用)",
                "response_mode": "text_reply" if self.mock_judge_reply else "silent",
            }, ensure_ascii=False)

        return "这是一个模拟回复。当 LLM 配置正确后，我将提供真实回答。"

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
