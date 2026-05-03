"""测试对话状态机。"""

import time

import pytest
from voice_agent.core.conversation_state import ConversationState


def test_initial_state_is_passive():
    state = ConversationState()
    assert state.mode == "passive_listening"
    assert not state.is_active_conversation()


def test_mark_agent_replied_enters_active_chat():
    state = ConversationState()
    state.mark_agent_replied()
    assert state.mode == "active_chat"
    assert state.is_active_conversation()


def test_cooldown_entered_and_expires():
    state = ConversationState(_cooldown_seconds=0.01)
    state.enter_cooldown()
    assert state.mode == "cooldown"
    assert state.is_in_cooldown()

    time.sleep(0.02)
    assert not state.is_in_cooldown()
    assert state.mode == "passive_listening"


def test_active_chat_timeout():
    state = ConversationState(_conversation_timeout_seconds=0.01)
    state.mark_agent_replied()
    assert state.is_active_conversation()

    time.sleep(0.02)
    assert not state.is_active_conversation()
    assert state.mode == "passive_listening"


def test_pause_and_resume():
    state = ConversationState()
    state.mark_agent_replied()
    assert state.mode == "active_chat"

    state.pause()
    assert state.is_paused()
    assert not state.is_active_conversation()

    state.resume()
    assert state.mode == "passive_listening"
    assert not state.is_paused()


def test_mark_user_final_text_updates_state():
    state = ConversationState()
    state.mark_user_final_text("你好")
    assert state.last_user_text == "你好"
    assert state.last_final_text_at > 0


def test_seconds_since_last_reply():
    state = ConversationState()
    assert state.seconds_since_last_reply() == float("inf")

    state.mark_agent_replied()
    s = state.seconds_since_last_reply()
    assert 0 <= s < 1


def test_seconds_since_last_final():
    state = ConversationState()
    assert state.seconds_since_last_final() == float("inf")

    state.mark_user_final_text("test")
    s = state.seconds_since_last_final()
    assert 0 <= s < 1
