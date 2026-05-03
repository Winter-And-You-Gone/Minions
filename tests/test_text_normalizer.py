"""测试文本归一化工具。"""

import pytest
from voice_agent.utils.text_normalizer import normalize_text


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
