"""测试介入判断器。"""

import time

import pytest
from voice_agent.core.conversation_state import ConversationState
from voice_agent.core.intervention_gate import InterventionGate, GateAction


def make_state(mode: str = "passive_listening") -> ConversationState:
    state = ConversationState()
    state.mode = mode
    return state


def make_active_state() -> ConversationState:
    state = ConversationState()
    state.mode = "active_chat"
    state.last_agent_reply_at = __import__("time").time()
    state.active_until = __import__("time").time() + 60
    return state


def make_cooldown_state() -> ConversationState:
    """创建处于冷却中的状态（有 active_chat 背景）。"""
    state = ConversationState(_cooldown_seconds=60)
    state.mark_agent_replied()
    state.enter_cooldown()
    return state


def test_silent_for_empty_text():
    gate = InterventionGate()
    result = gate.evaluate("", make_state())
    assert result.action == GateAction.SILENT


def test_silent_for_filler_words():
    gate = InterventionGate()
    for word in ["嗯", "啊", "哦", "好的", "可以"]:
        result = gate.evaluate(word, make_state())
        assert result.action == GateAction.SILENT, f"'{word}' 应为 silent，实际 {result.action.value}"


def test_silent_for_short_text():
    gate = InterventionGate(min_text_length=4)
    result = gate.evaluate("你好", make_state())
    assert result.action == GateAction.SILENT


def test_agent_for_strong_trigger():
    gate = InterventionGate()
    result = gate.evaluate("帮我看一下这个问题", make_state())
    assert result.action == GateAction.AGENT


def test_agent_for_question_trigger():
    gate = InterventionGate()
    result = gate.evaluate("这个报错是什么意思", make_state())
    assert result.action == GateAction.AGENT


def test_agent_for_why_trigger():
    gate = InterventionGate()
    result = gate.evaluate("为什么这个不行", make_state())
    assert result.action == GateAction.AGENT


def test_agent_for_ask_help():
    gate = InterventionGate()
    result = gate.evaluate("帮我查一下这个错误", make_state())
    assert result.action == GateAction.AGENT


def test_agent_for_remind():
    gate = InterventionGate()
    result = gate.evaluate("提醒我等会儿吃饭", make_state())
    assert result.action == GateAction.AGENT


def test_silent_for_statement():
    gate = InterventionGate()
    result = gate.evaluate("等会儿我去吃饭", make_state())
    assert result.action in (GateAction.SILENT, GateAction.BUBBLE)


def test_silent_for_casual_chat():
    gate = InterventionGate()
    result = gate.evaluate("这个好像不太对", make_state())
    assert result.action in (GateAction.BUBBLE, GateAction.JUDGE, GateAction.SILENT)


def test_active_chat_raises_score():
    gate = InterventionGate()
    passive_result = gate.evaluate("这个是什么", make_state("passive_listening"))
    active_result = gate.evaluate("这个是什么", make_active_state())
    assert active_result.score > passive_result.score


def test_silent_for_duplicate_text():
    gate = InterventionGate()
    state = make_state()
    state.last_user_text = "今天天气真好"
    result = gate.evaluate("今天天气真好", state)
    assert result.action == GateAction.SILENT


def test_silent_for_paused():
    gate = InterventionGate()
    state = make_state("paused")
    result = gate.evaluate("帮我查一下", state)
    assert result.action == GateAction.SILENT


def test_silent_for_low_confidence():
    gate = InterventionGate(min_asr_confidence=0.55)
    result = gate.evaluate("帮助我查一下", make_state(), asr_confidence=0.3)
    assert result.action == GateAction.SILENT


def test_open_browser_triggers_agent():
    gate = InterventionGate()
    result = gate.evaluate("打开浏览器", make_state())
    assert result.action == GateAction.AGENT


def test_weird_plot_silent_or_low():
    """TURN_AWAY_TRIGGERS 减分后未唤醒状态下可能 silent。"""
    gate = InterventionGate()
    result = gate.evaluate("这剧情也太离谱了", make_state())
    # 弱触发 +25, 转向 -40 = -15 → < threshold_bubble → SILENT
    # 未唤醒状态下转向内容大概率 silent
    assert result.action in (GateAction.SILENT, GateAction.BUBBLE, GateAction.JUDGE)


def test_question_mark_adds_score():
    gate = InterventionGate()
    with_q = gate.evaluate("这个对吗?", make_state())
    without_q = gate.evaluate("这个对吗", make_state())
    assert with_q.score >= without_q.score


# --- 中文空格归一化 ——

def test_chinese_spaces_strong_trigger():
    """带空格的 '帮 我 看 一 下' 应归一化后匹配强触发词。"""
    gate = InterventionGate()
    result = gate.evaluate("帮 我 看 一 下", make_state())
    assert result.action == GateAction.AGENT


def test_chinese_spaces_question_trigger():
    """带空格的 '什么 意思' 应归一化后匹配问题触发词。"""
    gate = InterventionGate()
    result = gate.evaluate("什么 意思", make_state())
    assert result.action == GateAction.AGENT


def test_chinese_spaces_need_help():
    gate = InterventionGate()
    result = gate.evaluate("帮 我 查 一 下 这 个 错 误", make_state())
    assert result.action == GateAction.AGENT


# --- 冷却绕过 ---

def test_cooldown_silent_for_normal_statement():
    """冷却中普通陈述应 silent。"""
    gate = InterventionGate()
    state = make_cooldown_state()
    result = gate.evaluate("今天天气不错", state)
    assert result.action == GateAction.SILENT


def test_cooldown_bypass_for_strong_trigger():
    """冷却中强触发词应绕过，仍然允许进入 agent。"""
    gate = InterventionGate()
    state = make_cooldown_state()
    result = gate.evaluate("帮我查一下", state)
    assert result.action == GateAction.AGENT


def test_cooldown_bypass_for_question_trigger():
    """冷却中明确问题应绕过。"""
    gate = InterventionGate()
    state = make_cooldown_state()
    result = gate.evaluate("这是什么意思", state)
    assert result.action == GateAction.AGENT


def test_cooldown_bypass_for_question_mark():
    """冷却中以问号结尾可通过。"""
    gate = InterventionGate()
    state = make_cooldown_state()
    result = gate.evaluate("真的吗?", state)
    # 不含强/问题触发词，但问号加分 20，状态加权 30 = 50 → JUDGE（默认 threshold_agent=60）
    assert result.action in (GateAction.JUDGE, GateAction.AGENT)


# --- 参数化主验收用例 ---

@pytest.mark.parametrize("text,expected_action", [
    ("嗯", GateAction.SILENT),
    ("这个报错是什么意思", GateAction.AGENT),
    ("帮我看一下这个问题", GateAction.AGENT),
    ("你觉得这个方案怎么样", GateAction.JUDGE),
    ("提醒我等会儿吃饭", GateAction.AGENT),
    ("打开浏览器", GateAction.AGENT),
    ("帮 我 看 一 下", GateAction.AGENT),
    ("什么 意思", GateAction.AGENT),
])
def test_main_cases_from_spec(text, expected_action):
    """方案文档中的主要验收用例。"""
    gate = InterventionGate()
    result = gate.evaluate(text, make_state())
    assert result.action == expected_action, f"'{text}': 期望 {expected_action.value}，实际 {result.action.value} (score={result.score})"
