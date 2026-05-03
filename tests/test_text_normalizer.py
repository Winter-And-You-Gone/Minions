"""测试文本归一化工具。"""

import pytest
from voice_agent.utils.text_normalizer import normalize_text, remove_chinese_spaces, has_chinese


def test_normalize_strips_whitespace():
    assert normalize_text("  你好  ") == "你好"


def test_normalize_collapses_spaces():
    assert normalize_text("这个  报错  是什么意思") == "这个 报错 是什么意思"


def test_normalize_unifies_question_marks():
    assert normalize_text("这是什么？") == "这是什么?"


def test_normalize_unifies_exclamation_marks():
    assert normalize_text("太好了！") == "太好了!"


def test_normalize_removes_duplicate_punct():
    assert normalize_text("什么？？？", remove_dup_punct=True) == "什么?"


def test_normalize_keeps_dup_punct_if_disabled():
    result = normalize_text("什么？？？", remove_dup_punct=False)
    assert "???" in result


def test_normalize_empty_string():
    assert normalize_text("") == ""


def test_normalize_mixed_whitespace():
    assert normalize_text("  这个   报错   是什么意思？  ") == "这个 报错 是什么意思?"


@pytest.mark.parametrize("input_text,expected_keyword", [
    ("帮我看看这个", "帮我"),
    ("解释一下这个问题", "解释一下"),
    ("这个是什么意思", "什么意思"),
    ("嗯", "嗯"),
])
def test_normalize_preserves_keywords(input_text, expected_keyword):
    result = normalize_text(input_text)
    assert expected_keyword in result


# --- remove_chinese_spaces ---

def test_remove_chinese_spaces_basic():
    assert remove_chinese_spaces("帮 我 看 一 下") == "帮我看一下"


def test_remove_chinese_spaces_partial():
    """中英文混排只移除中文之间的空格。"""
    result = remove_chinese_spaces("帮 我 看 一下 README")
    # "帮 我 看 一下" 中文间空格被移除，但 "一下 README" 中英文间空格保留
    assert result == "帮我看一下 README"


def test_remove_chinese_spaces_no_change():
    assert remove_chinese_spaces("帮我查一下") == "帮我查一下"


def test_remove_chinese_spaces_question():
    assert remove_chinese_spaces("什么 意思") == "什么意思"


def test_remove_chinese_spaces_empty():
    assert remove_chinese_spaces("") == ""


def test_remove_chinese_spaces_mixed():
    assert remove_chinese_spaces("这 个 报 错 是 什么 意思 ?") == "这个报错是什么意思 ?"


# --- has_chinese ---

def test_has_chinese_true():
    assert has_chinese("hello 你好")


def test_has_chinese_false():
    assert not has_chinese("hello world")


def test_has_chinese_empty():
    assert not has_chinese("")
