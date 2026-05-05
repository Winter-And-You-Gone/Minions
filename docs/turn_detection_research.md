# Turn Detection 方案调研

## 背景

Minions 常驻语音 Agent 需要判断用户何时完成一句话（Turn End），以触发 ASR final 和后续的 Agent 回复。当前使用 RMS VAD（基于音量的语音活动检测）做简单分段，但在复杂场景（打断、背景噪音、多人对话）下不够鲁棒。

---

## 方案对比

### 1. WebRTC VAD (google/webrtcvad)

| 维度 | 说明 |
|------|------|
| **原理** | 基于高斯混合模型(GMM)的语音/非语音分类，分析频段能量分布 |
| **延迟** | 10-30ms 帧级别 |
| **资源** | 极低，纯 CPU 可运行 |
| **优点** | 轻量、成熟、跨平台、Python 绑定成熟 (`webrtcvad` pip 包) |
| **缺点** | 仅区分"语音/非语音"，不区分"人/噪音"；对音乐、敲键盘等非语音能量敏感 |
| **适配 Minions** | 可直接替换 `segmenter.py` 中的 RMS 阈值判断，或作为 VAD 前级 |
| **参考** | https://github.com/wiseman/py-webrtcvad |

> **结论**: 适合作为 VAD 第一级，快速过滤非语音帧。

### 2. Silero VAD (snakers4/silero-vad)

| 维度 | 说明 |
|------|------|
| **原理** | 基于 LSTM 的深度学习 VAD 模型，直接预测每帧的语音概率 |
| **延迟** | ~30ms |
| **资源** | 中等，ONNX 推理，单核 ~10% CPU |
| **优点** | 最先进的开源 VAD，对噪音、音乐、敲击声鲁棒；支持 ONNX 推理；Python 绑定成熟 |
| **缺点** | 比 WebRTC VAD 重；模型 ~1.7MB；首次加载慢 |
| **适配 Minions** | 可替换 `segmenter.py`，提供比 RMS 更准确的语音端点检测 |
| **参考** | https://github.com/snakers4/silero-vad |

> **结论**: 推荐作为下一版 VAD 升级方案。用 ONNX 运行时推理，延迟可接受。

### 3. LiveKit Turn Detector

| 维度 | 说明 |
|------|------|
| **原理** | 基于 Hyperseg (Wav2Vec2 微调) 的端到端语音活动 + 说话人分割 |
| **延迟** | 100-300ms |
| **资源** | 高，需要 GPU 推理 |
| **优点** | 端到端分段，无需独立 VAD；支持流式；说话人识别；与 LiveKit Agent Framework 深度集成 |
| **缺点** | 依赖 LiveKit 生态；模型 ~150MB；GPU 需求；非通用 Python 包 |
| **适配 Minions** | 需引入 LiveKit Agent Framework，架构改动大 |
| **参考** | https://github.com/livekit/agents |

> **结论**: 如果未来 Minions 迁移到 LiveKit 基础设施，值得采用。当前阶段太重。

### 4. Pipecat Smart Turn v3

| 维度 | 说明 |
|------|------|
| **原理** | 多模态 VAD（音量 + 语义 + 间隔）+ Barge-in 检测 |
| **延迟** | 50-200ms |
| **资源** | 中，综合多种信号 |
| **优点** | 开源 Configurable；支持打断；支持多种 transport 后端；Daily.co / WebRTC |
| **缺点** | Python SDK 尚不成熟（快速迭代中）；文档分散；依赖 Daily.co 或 WebSocket |
| **适配 Minions** | 需引入 Pipecat Pipeline 架构，改动较大 |
| **参考** | https://github.com/pipecat-ai/pipecat |

> **结论**: 适合下一阶段"语音优先"架构重写时借鉴其 Turn 决策逻辑。

---

## Minions 推荐方案

### 短期（当前 sprint）

- **保持 RMS VAD** 作为基础分段 (`segmenter.py`)
- 降低 `rms_threshold` 提高灵敏度，配合 `min_speech_ms` 防误触发
- **新功能**: 添加 `silence_timeout_ms` 参数控制"说完了"判定

### 中期（下个 sprint）

1. **替换 VAD**: 引入 Silero VAD 替代 RMS 阈值
   - 加载 `silero_vad.onnx` 模型
   - 替换 `segmenter.py` 中的 `_rms_vad()` 为 `_silero_vad()`
   - 保持 `min_speech_ms` 和 `silence_timeout_ms` 参数结构不变

2. **添加 Turn End 规则**:
   - 静音超时（已有 `silence_timeout_ms`）
   - 最大话语长度（已有 `max_speech_ms`）
   - 语义完成度（可选，低优先级）

### 长期（未来架构）

- 评估 **Pipecat Smart Turn** 集成到 Minions 管道中
- 用 Pipecat 的 `TurnAnalyzer` 替代自建 VAD + Segmenter
- 评估是否需要 LiveKit 基础设施

---

## 当前实现总结

| 组件 | 实现 | 状态 |
|------|------|------|
| VAD 分段 | `segmenter.py` 基于 RMS + 状态机 | ✅ 已实现 |
| 静音超时 | `silence_timeout_ms` 参数 | ✅ 已实现 |
| 最大话长 | `max_speech_ms` 参数 | ✅ 已实现 |
| 打断支持 | ASR 流式 + Gate 实时评估 | ⚠️ 基础支持 |
| 语义 Turn End | 不适用（ASR 只出 final） | ❌ 未来考虑 |
| Silero VAD | 待引入 | 📋 下个 sprint |
| 说话人识别 | 不适用（单用户场景） | ❌ 不需要 |
