"""测试唤醒名检测。"""

from voice_agent.core.wake_name import WakeNameMatcher, WakeNameConfig


def _matcher(aliases: list[str] | None = None) -> WakeNameMatcher:
    return WakeNameMatcher(WakeNameConfig(
        name="米粒",
        aliases=aliases or ["米粒", "迷你"],
    ))


class TestWakeNameExactPrefix:
    def test_wake_name_exact_prefix(self) -> None:
        result = _matcher().detect("米粒帮我看一下")
        assert result.matched
        assert result.alias == "米粒"
        assert result.text_without_name == "帮我看一下"

    def test_wake_name_homophone_alias(self) -> None:
        result = _matcher().detect("迷你这个报错是什么意思")
        assert result.matched
        assert result.alias == "迷你"

    def test_wake_name_with_prefix_word(self) -> None:
        result = _matcher().detect("嗯米粒帮我解释一下")
        assert result.matched

    def test_wake_name_no_match(self) -> None:
        result = _matcher().detect("这个报错是什么意思")
        assert not result.matched


class TestWakeNameOnlyName:
    def test_only_name(self) -> None:
        result = _matcher().detect("米粒")
        assert result.matched
        assert result.text_without_name == ""

    def test_only_alias(self) -> None:
        result = _matcher().detect("迷你")
        assert result.matched
        assert result.text_without_name == ""


class TestWakeNameDisabled:
    def test_disabled_returns_no_match(self) -> None:
        matcher = WakeNameMatcher(WakeNameConfig(
            name="米粒", aliases=["米粒"], enabled=False,
        ))
        result = matcher.detect("米粒帮我看看")
        assert not result.matched


class TestWakeNameWithPrefixes:
    def test_prefix_ni_hao(self) -> None:
        result = _matcher().detect("你好米粒")
        assert result.matched

    def test_prefix_wei(self) -> None:
        result = _matcher().detect("喂米粒")
        assert result.matched

    def test_prefix_na_ge(self) -> None:
        result = _matcher().detect("那个米粒")
        assert result.matched


class TestWakeNameNoMatchCases:
    def test_name_in_middle_not_trigger(self) -> None:
        result = _matcher().detect("帮我看看米粒")
        assert not result.matched

    def test_long_text_without_name(self) -> None:
        result = _matcher().detect("今天天气怎么样")
        assert not result.matched
