"""测试 AgentCore。"""

from voice_agent.event_bus import EventBus
from voice_agent.core.conversation_state import ConversationState
from voice_agent.core.intervention_gate import InterventionGate
from voice_agent.core.llm_client import LLMClient
from voice_agent.core.agent_core import AgentCore


def test_is_model_info_question():
    bus = EventBus()
    state = ConversationState()
    gate = InterventionGate()
    llm = LLMClient(enabled=False, api_base="", api_key="", model="test-model")
    agent = AgentCore(bus, state, gate, llm)

    assert agent._is_model_info_question("你是什么模型")
    assert agent._is_model_info_question("你用的什么模型")
    assert agent._is_model_info_question("当前模型是什么")
    assert agent._is_model_info_question("模型名是什么")
    assert not agent._is_model_info_question("明天天气怎么样")
