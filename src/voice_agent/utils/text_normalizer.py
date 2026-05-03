"""文本归一化工具：中文文本预处理。"""

import re

_MULTI_SPACE = re.compile(r"\s+")
_DUP_PUNCT = re.compile(r"([!?！？])\1+")


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
