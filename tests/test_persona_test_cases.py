"""测试 PERSONA_TEST_CASES 存在且包含关键条目。"""

from voice_agent.core.persona_test_cases import PERSONA_TEST_CASES


class TestPersonaTestCases:
    def test_not_empty(self) -> None:
        assert PERSONA_TEST_CASES

    def test_contains_wake_name(self) -> None:
        assert "琉璃川" in PERSONA_TEST_CASES

    def test_contains_followup(self) -> None:
        assert "那怎么修" in PERSONA_TEST_CASES

    def test_contains_continue(self) -> None:
        assert "继续" in PERSONA_TEST_CASES

    def test_contains_end_session(self) -> None:
        assert "不用了" in PERSONA_TEST_CASES

    def test_has_reasonable_length(self) -> None:
        assert len(PERSONA_TEST_CASES) >= 8
