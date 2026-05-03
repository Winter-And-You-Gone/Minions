"""文本归一化工具：中文文本预处理。"""

import re

_MULTI_SPACE = re.compile(r"\s+")
_DUP_PUNCT = re.compile(r"([!?！？])\1+")
_CHINESE_CHARS = re.compile(r"[一-鿿㐀-䶿豈-﫿]")
_CHINESE_SPACE = re.compile(r"(?<=[一-鿿㐀-䶿])\s+(?=[一-鿿㐀-䶿])")


def normalize_text(text: str, remove_dup_punct: bool = True) -> str:
    """归一化中文语音识别文本。

    - 去除首尾空格
    - 合并多余空白
    - 中文/英文问号统一为半角 ?
    - 中文/英文感叹号统一为半角 !
    - 可选移除重复标点
    """
    text = text.strip()
    text = _MULTI_SPACE.sub(" ", text)
    text = text.replace("？", "?").replace("！", "!")
    if remove_dup_punct:
        text = _DUP_PUNCT.sub(r"\1", text)
    return text


def remove_chinese_spaces(text: str) -> str:
    """移除 ASR 输出中中文字符之间的空格，例如 '帮 我 看 一 下' -> '帮我看一下'。"""
    return _CHINESE_SPACE.sub("", text)


def has_chinese(text: str) -> bool:
    """检查文本是否包含中文字符。"""
    return bool(_CHINESE_CHARS.search(text))
