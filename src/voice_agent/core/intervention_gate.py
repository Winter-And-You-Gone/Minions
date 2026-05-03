"""介入判断器：决定是否需要 AI 回应。

判断流程:
  ASR final text → 硬过滤 → 规则打分 → 状态加权 → 输出 action
"""

import time
from dataclasses import dataclass, field
from enum import Enum

from voice_agent.utils.text_normalizer import normalize_text, remove_chinese_spaces
from voice_agent.core.conversation_state import ConversationState


class GateAction(str, Enum):
    SILENT = "silent"
    BUBBLE = "bubble"
    JUDGE = "judge"
    AGENT = "agent"
    TOOL = "tool"
    CONFIRM = "confirm"


@dataclass
class GateResult:
    action: GateAction
    score: int
    reason: str
    text: str = ""
    metadata: dict = field(default_factory=dict)


# 强触发词：明确请求 AI 帮助
STRONG_TRIGGERS = [
    "帮我", "替我", "给我", "帮忙",
    "解释一下", "分析一下", "总结一下",
    "查一下", "搜索", "打开", "关闭",
    "记一下", "提醒我", "设置提醒",
    "翻译", "写一段", "改一下",
]

# 问题触发词：明显在提问
QUESTION_TRIGGERS = [
    "为什么", "怎么", "怎么办", "什么原因",
    "什么意思", "是什么", "能不能", "可以吗",
    "对不对", "是不是", "合理吗", "你觉得",
]

# 弱触发词：表达困惑/求助
WEAK_TRIGGERS = [
    "好烦", "离谱", "无语", "奇怪", "不对劲", "不太对",
    "有问题", "看不懂", "搞不懂",
]

# 语气词 / 无意义词：直接沉默
FILLER_WORDS = [
    "嗯", "啊", "哦", "额", "呃", "哈哈",
    "好的", "行", "可以",
    "就是说", "就是说呢", "那个", "然后",
    "反正", "随便", "也行", "好吧",
]

# 强触发词加分（直接 agent）
STRONG_TRIGGER_SCORE = 65
# 问题触发词加分（直接 agent）
QUESTION_TRIGGER_SCORE = 60
# 弱触发词加分（bubble / judge）
WEAK_TRIGGER_SCORE = 25
# 句尾问号加分
QUESTION_MARK_SCORE = 20
# 处于 active_chat 加分
ACTIVE_CHAT_SCORE = 30
# AI 最近回复过加分
RECENT_REPLY_SCORE = 20
# 普通陈述减分
STATEMENT_PENALTY = -15
# 短句不明确减分
SHORT_AMBIGUOUS_PENALTY = -15
# 过于口语化减分
COLLOQUIAL_PENALTY = -30

# 唤醒名检测得分（直接 Agent）
WAKE_NAME_SCORE = 100
# 唤醒会话内追问加分
WAKE_SESSION_SCORE = 75

# 连续追问触发词 — 唤醒或 active 会话内短句也能进 Agent
FOLLOWUP_TRIGGERS = [
    "然后呢",
    "继续",
    "接着说",
    "详细说",
    "具体点",
    "为什么",
    "怎么改",
    "怎么修",
    "怎么办",
    "这个呢",
    "那个呢",
    "那它呢",
    "什么意思",
    "重新说",
    "再说一遍",
]

# 结束会话触发词 — 明确表示不再需要 AI
END_SESSION_TRIGGERS = [
    "没事了",
    "不用了",
    "先这样",
    "你先别说",
    "别说话",
    "闭嘴",
    "暂停",
    "不是跟你说",
    "我不是问你",
    "我跟别人说",
]


@dataclass
class InterventionGate:
    """介入判断器。

    硬过滤 → 规则打分 → 状态加权 → 输出 action。
    """

    min_text_length: int = 4
    min_asr_confidence: float = 0.55
    cooldown_seconds: float = 5.0
    threshold_bubble: int = 15
    threshold_judge: int = 30
    threshold_agent: int = 60
    uncertain_action: str = "judge"
    wake_matcher: object | None = None
    _last_result: GateResult | None = field(default=None, init=False)

    def evaluate(
        self,
        raw_text: str,
        state: ConversationState,
        asr_confidence: float = 1.0,
    ) -> GateResult:
        # Gate 专用：移除 ASR 中文字符间空格
        text = remove_chinese_spaces(raw_text)
        text = normalize_text(text)

        # --- 唤醒名检测（在 hard_filter 之前，绕过 min_text_length） ---
        if self.wake_matcher is not None:
            wake_match = self.wake_matcher.detect(text)
            if wake_match.matched:
                cleaned = wake_match.text_without_name if wake_match.text_without_name else text
                result = GateResult(
                    action=GateAction.AGENT,
                    score=WAKE_NAME_SCORE,
                    reason=wake_match.reason,
                    text=cleaned,
                    metadata={
                        "wake_detected": True,
                        "wake_name": wake_match.name,
                        "wake_alias": wake_match.alias,
                        "text_without_name": wake_match.text_without_name,
                    },
                )
                self._last_result = result
                return result

        # --- 硬过滤 ---
        hard_filter_result = self._hard_filter(text, state, asr_confidence)
        if hard_filter_result is not None:
            self._last_result = hard_filter_result
            return hard_filter_result

        # --- 规则打分 ---
        score, reasons = self._rule_score(text, state)

        # --- 状态加权 ---
        score, state_reason = self._state_weight(score, state)
        if state_reason:
            reasons.append(state_reason)

        # --- 分数 → action ---
        action = self._score_to_action(score)
        reason = "; ".join(reasons) if reasons else "无特殊匹配"

        result = GateResult(action=action, score=score, reason=reason, text=text)
        self._last_result = result
        return result

    def _hard_filter(
        self,
        text: str,
        state: ConversationState,
        confidence: float,
    ) -> GateResult | None:
        """硬过滤：满足任一条件直接沉默，返回 GateResult；否则返回 None。"""
        # 文本为空
        if not text:
            return GateResult(GateAction.SILENT, 0, "空文本")

        # 结束唤醒会话指令（在 min_text_length 之前检查）
        for word in END_SESSION_TRIGGERS:
            if word in text:
                return GateResult(
                    GateAction.SILENT, 0, f"结束唤醒会话: {word}",
                    metadata={"end_wake_session": True},
                )

        # 文本太短（唤醒会话内绕过）
        if len(text) < self.min_text_length:
            if not (hasattr(state, "is_wake_session_active") and state.is_wake_session_active()):
                return GateResult(GateAction.SILENT, 0, f"文本太短 ({len(text)} < {self.min_text_length})")

        # ASR 置信度太低
        if confidence < self.min_asr_confidence:
            return GateResult(GateAction.SILENT, 0, f"置信度过低 ({confidence:.2f} < {self.min_asr_confidence})")

        # 只有语气词
        if text in FILLER_WORDS:
            return GateResult(GateAction.SILENT, 0, "语气词")

        # 和上一句完全重复
        if text == state.last_user_text and state.mode != "active_chat":
            return GateResult(GateAction.SILENT, 0, "与上一句重复")

        # 处于 paused 状态
        if state.is_paused():
            return GateResult(GateAction.SILENT, 0, "已暂停")

        # 冷却中 — 强触发词或明确问题可绕过
        if state.is_in_cooldown():
            if not self._has_bypass_trigger(text):
                return GateResult(GateAction.SILENT, 0, "冷却中")

        return None

    def _has_bypass_trigger(self, text: str) -> bool:
        """检查文本是否包含可绕过冷却的强触发词、问题触发词或问号结尾。"""
        for word in STRONG_TRIGGERS:
            if word in text:
                return True
        for word in QUESTION_TRIGGERS:
            if word in text:
                return True
        if text.endswith("?"):
            return True
        return False

    def _rule_score(self, text: str, state: ConversationState) -> tuple[int, list[str]]:
        """规则打分。"""
        score = 0
        reasons = []
        matched_strong = False

        # 强触发词
        for word in STRONG_TRIGGERS:
            if word in text:
                score += STRONG_TRIGGER_SCORE
                reasons.append(f"强触发词: {word}")
                matched_strong = True
                break

        # 问题触发词（只在强触发未命中时检查）
        if not matched_strong:
            for word in QUESTION_TRIGGERS:
                if word in text:
                    score += QUESTION_TRIGGER_SCORE
                    reasons.append(f"问题触发词: {word}")
                    break

        # 弱触发词（只在强/问题未命中时检查）
        if not matched_strong and not reasons:
            for word in WEAK_TRIGGERS:
                if word in text:
                    score += WEAK_TRIGGER_SCORE
                    reasons.append(f"弱触发词: {word}")
                    break

        # 句尾问号
        if text.endswith("?"):
            score += QUESTION_MARK_SCORE
            reasons.append("句尾问号")

        # 连续追问触发词（加分，不阻断其他匹配）
        for word in FOLLOWUP_TRIGGERS:
            if word in text:
                score += 55
                reasons.append(f"连续追问: {word}")
                break

        # 普通陈述句惩罚
        if not reasons and len(text) >= 8:
            score += STATEMENT_PENALTY
            reasons.append("普通陈述句")

        # 短句不明确（唤醒会话内不惩罚）
        if len(text) < 8 and not reasons:
            in_wake = hasattr(state, "is_wake_session_active") and state.is_wake_session_active()
            if not in_wake:
                score += SHORT_AMBIGUOUS_PENALTY
            reasons.append("短句不明确")

        return score, reasons

    def _state_weight(self, score: int, state: ConversationState) -> tuple[int, str | None]:
        """状态加权。"""
        # 唤醒会话加分（优先于 active_chat）
        if hasattr(state, "is_wake_session_active") and state.is_wake_session_active():
            return score + WAKE_SESSION_SCORE, "wake_session 加分"

        # active_chat 加分
        if state.is_active_conversation():
            return score + ACTIVE_CHAT_SCORE, "active_chat 加分"

        # AI 最近回复过（60 秒内）
        if state.seconds_since_last_reply() < 60:
            return score + RECENT_REPLY_SCORE, "AI 最近回复过加分"

        return score, None

    def _score_to_action(self, score: int) -> GateAction:
        if score < self.threshold_bubble:
            return GateAction.SILENT
        if score < self.threshold_judge:
            if self.uncertain_action == "silent":
                return GateAction.SILENT
            return GateAction.BUBBLE
        if score < self.threshold_agent:
            if self.uncertain_action == "silent":
                return GateAction.SILENT
            if self.uncertain_action == "agent":
                return GateAction.AGENT
            return GateAction.JUDGE
        return GateAction.AGENT
