"""测试 CLI shell 的用户输入发布逻辑。

核心验证点：
  - ``_publish_user_text`` 不调用 ``state.mark_user_final_text()``
    （ConversationState 必须由 AgentCore 统一管理）
  - 发布的事件携带 ``source="cli"``
"""

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from voice_agent.event_bus import EventBus
from voice_agent.cli.shell import MinionsShell


@pytest.fixture
def mock_components():
    bus = EventBus()
    state = MagicMock()
    # mark_user_final_text 默认无副作用
    state.mode = "active"
    state.seconds_since_last_reply = MagicMock(return_value=999.0)

    agent = MagicMock()
    llm = MagicMock()
    # LLMClient.is_available is a property
    type(llm).is_available = PropertyMock(return_value=False)
    llm.model = ""

    return bus, state, agent, llm


@pytest.fixture
def shell(mock_components):
    """MinionsShell 在测试环境中无法创建 PromptSession（无真实终端），
    因此 patch 掉 PromptSession，让实例化跳过终端检测。"""
    bus, state, agent, llm = mock_components
    with patch("voice_agent.cli.shell.PromptSession") as mock_ps:
        s = MinionsShell(bus, agent, state, llm)
        yield s


class TestPublishUserText:
    """验证 ``_publish_user_text`` 的事件发布行为。"""

    async def test_publishes_user_text_event(self, shell, mock_components):
        bus, state, agent, llm = mock_components
        events: list[dict] = []

        async def collector(event: dict) -> None:
            events.append(event)

        bus.subscribe(collector)

        await shell._publish_user_text("你好")

        # 应该发布 user.text 和 asr.final
        types = [e["type"] for e in events]
        assert "user.text" in types
        assert "asr.final" in types

    async def test_events_have_cli_source(self, shell, mock_components):
        bus, state, agent, llm = mock_components
        events: list[dict] = []

        async def collector(event: dict) -> None:
            events.append(event)

        bus.subscribe(collector)

        await shell._publish_user_text("test")

        for event in events:
            assert event.get("source") == "cli", (
                f"事件 {event['type']} 缺少 source=cli"
            )

    async def test_does_not_mark_state(self, shell, mock_components):
        """关键的回归测试：_publish_user_text 绝不能调用 mark_user_final_text。"""
        bus, state, agent, llm = mock_components

        await shell._publish_user_text("hello")

        # mark_user_final_text 是 MagicMock，默认不记录调用
        # 但我们主动 assert 它没有被调用过
        state.mark_user_final_text.assert_not_called()

    async def test_text_content_preserved(self, shell, mock_components):
        bus, state, agent, llm = mock_components
        events: list[dict] = []

        async def collector(event: dict) -> None:
            events.append(event)

        bus.subscribe(collector)

        text = "测试中文输入"
        await shell._publish_user_text(text)

        user_text_event = next(e for e in events if e["type"] == "user.text")
        assert user_text_event["text"] == text

        asr_event = next(e for e in events if e["type"] == "asr.final")
        assert asr_event["text"] == text
        assert asr_event["confidence"] == 1.0
