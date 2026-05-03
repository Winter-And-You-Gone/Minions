# Minions — 常驻语音 Agent

常驻后台的语音助手，支持语音活动检测（VAD）、流式语音识别（ASR）、LLM 驱动对话决策，以及可选的 WebSocket 输出。

## 快速开始

```bash
# 安装基础依赖
pip install -e .

# 启动（使用 mock ASR，从控制台输入文本模拟语音）
python -m voice_agent.main --asr mock
```

## 配置

`config.yaml` 控制所有行为：ASR 引擎、音频参数、干预策略、LLM 连接等。
敏感信息（API key 等）通过环境变量或 `.env` 文件设置。

## 命令行选项

| 选项 | 说明 |
|------|------|
| `--asr mock` | Mock ASR（键盘输入模拟语音） |
| `--asr sherpa-onnx` | 真实语音识别（需安装可选依赖） |
| `--cli` | CLI 交互模式（MinionsShell，含 VU 监测） |
| `--mic-test` | 麦克风测试模式 |
| `--list-devices` | 列出所有音频设备 |
| `--device <id>` | 指定麦克风设备 ID |

## 真实语音识别 sherpa-onnx

第一版使用 RMS VAD 分段识别，不是逐字 streaming。

工作流程：麦克风采集 → 基于能量的语音分段 → 静音检测到结束 → 整段送入 sherpa-onnx offline recognizer → 输出文本。

### 安装

```bash
pip install -e ".[asr]"
```

### 准备模型

从 https://github.com/k2-fsa/sherpa-onnx/releases 下载离线模型。

### 配置

修改 `config.yaml`：

```yaml
asr:
  engine: "sherpa-onnx"
  sherpa_onnx:
    tokens: "./models/sherpa-onnx/tokens.txt"
    model: "./models/sherpa-onnx/model.int8.onnx"
    # 或者 transducer 模型：
    # encoder: "./models/encoder.onnx"
    # decoder: "./models/decoder.onnx"
    # joiner: "./models/joiner.onnx"
```

### 启动

```bash
python -m voice_agent.main --asr sherpa-onnx
```

## CLI 交互模式

```bash
python -m voice_agent.main --cli
```

启动后可用命令：

| 命令 | 说明 |
|------|------|
| `/mic monitor` | 实时麦克风 VU 音量监测 |
| `/mic list` | 列出音频设备 |
| `/mic select <id>` | 选择麦克风 |
| `/mic autodetect` | 自动检测有音频输入的设备 |
| `/mic autodetect --select` | 自动检测并切换到最佳设备 |
| `/mic info` | 查看当前麦克风信息 |
| `/pause` / `/resume` | 暂停/恢复 AI 回应 |
| `/status` | 系统状态 |
| `/help` | 帮助 |

## 项目结构

```
src/voice_agent/
├── asr/                    # 语音识别引擎
│   ├── base.py             # ASR 引擎协议
│   ├── mock_asr.py         # 模拟 ASR（键盘输入）
│   └── sherpa_onnx_asr.py  # sherpa-onnx 真实 ASR
├── audio/                  # 音频采集 & VAD
│   ├── microphone.py       # 麦克风采集（sounddevice）
│   └── segmenter.py        # RMS 语音分段器
├── cli/                    # 交互式 CLI
│   └── shell.py            # MinionsShell（prompt_toolkit + rich）
├── core/                   # 核心逻辑
│   ├── agent_core.py       # Agent 核心
│   ├── conversation_state.py
│   ├── intervention_gate.py
│   └── llm_client.py       # LLM 客户端（OpenAI 兼容）
├── output/                 # 输出
│   ├── console_output.py   # 控制台彩色输出
│   └── websocket_server.py # WebSocket 输出
├── utils/
│   └── text_normalizer.py
├── config.py
├── event_bus.py
├── logger.py
└── main.py                 # 入口
```
