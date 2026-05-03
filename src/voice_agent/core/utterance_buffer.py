"""话语缓冲区：暂存 partial 文本直到 final。"""

from dataclasses import dataclass, field


@dataclass
class UtteranceBuffer:
    """缓冲当前 utterance 的 partial / final 文本。"""

    current_partial: str = ""
    last_final_text: str = ""
    partials: list[str] = field(default_factory=list)

    def add_partial(self, text: str) -> None:
        self.current_partial = text
        self.partials.append(text)
        if len(self.partials) > 50:
            self.partials = self.partials[-50:]

    def finalize(self, text: str) -> str:
        self.current_partial = ""
        self.last_final_text = text
        self.partials.clear()
        return text

    def reset(self) -> None:
        self.current_partial = ""
        self.partials.clear()
