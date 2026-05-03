# Minions — 常驻语音 Agent

常驻后台的语音助手，支持语音活动检测（VAD）、流式语音识别（ASR）、LLM 驱动对话决策，以及可选的 WebSocket 输出。

## 快速开始

```bash
# 安装依赖
pip install -e .

# 启动（使用 mock ASR）
python -m voice_agent.main --asr mock
```

## 配置

`config.yaml` 控制所有行为：ASR 引擎、音频参数、干预策略、LLM 连接等。
敏感信息（API key 等）通过环境变量或 `.env` 文件设置。

## 项目结构

```
src/voice_agent/
├── asr/        # 语音识别引擎（mock / sherpa-onnx）
├── audio/      # 音频采集 & VAD
├── core/       # 核心逻辑（LLM 客户端、干预门控、对话状态）
├── output/     # 输出（控制台 / WebSocket）
├── utils/      # 工具函数
├── config.py   # 配置加载
├── event_bus.py
├── logger.py
└── main.py     # 入口
```
