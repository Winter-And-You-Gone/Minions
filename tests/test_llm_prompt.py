"""测试 LLM prompt 模板和默认配置。"""

from voice_agent.core.llm_client import AGENT_PROMPT, JUDGE_PROMPT
from voice_agent.core.wake_name import WakeNameConfig


class TestAgentPrompt:
    def test_agent_prompt_formats_with_all_fields(self) -> None:
        """AGENT_PROMPT 能用所有格式化字段正常渲染。"""
        result = AGENT_PROMPT.format(
            text="帮我查一下天气",
            recent_context="用户: 今天好冷\nAI: 多穿点哦",
            state="active_chat",
            wake_session_active="是",
            assistant_name="琉璃川",
            user_title="少爷",
            model="gpt-4",
        )
        assert "琉璃川" in result
        assert "少爷" in result
        assert "帮我查一下天气" in result
        assert "active_chat" in result

    def test_agent_prompt_contains_character_name(self) -> None:
        """AGENT_PROMPT 包含角色名琉璃川。"""
        assert "琉璃川" in AGENT_PROMPT or "{assistant_name}" in AGENT_PROMPT

    def test_agent_prompt_contains_user_title(self) -> None:
        """AGENT_PROMPT 包含用户称呼。"""
        assert "{user_title}" in AGENT_PROMPT


class TestJudgePrompt:
    def test_judge_prompt_formats(self) -> None:
        """JUDGE_PROMPT 能正常渲染。"""
        result = JUDGE_PROMPT.format(
            state="idle",
            recent_agent_reply="否",
            text="今天天气怎么样",
        )
        assert "今天天气怎么样" in result
        assert "idle" in result


class TestWakeNameDefaults:
    def test_default_name_is_liulichuan(self) -> None:
        """WakeNameConfig 默认 name 为琉璃川。"""
        cfg = WakeNameConfig()
        assert cfg.name == "琉璃川"

    def test_default_aliases_include_liuli(self) -> None:
        """默认 aliases 包含琉璃。"""
        cfg = WakeNameConfig()
        assert "琉璃" in cfg.aliases

    def test_default_user_title_is_shaoye(self) -> None:
        """默认 user_title 为少爷。"""
        cfg = WakeNameConfig()
        assert cfg.user_title == "少爷"
