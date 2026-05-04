# Minions — 常驻语音 Agent TUI

常驻语音 Agent，支持语音活动检测（VAD）、流式语音识别（ASR）、LLM 驱动对话决策、本地小模型语义 Judge，以及可选的 WebSocket 输出。

采用 OpenCode 风格的 TUI 界面，顶部状态栏 + 左侧对话 + 右侧状态面板 + 底部输入框。

## 日常启动

```bash
conda run -n Minions python -m voice_agent.main
```

默认启动：
- TUI 全屏界面
- sherpa-onnx ASR（语音识别）
- Local Judge qwen3.5:4b（语义判断）
- 主 LLM（琉璃川人格）
- WebSocket 输出

无需记忆任何参数。

## 依赖安装

```bash
# 基础依赖
pip install -e .

# ASR 可选依赖（sherpa-onnx）
pip install -e ".[asr]"

# 开发依赖（pytest 等）
pip install -e ".[dev]"

# Local Judge 模型
ollama pull qwen3.5:4b
```

## 配置

`config.yaml` 控制所有行为：ASR 引擎、音频参数、干预策略、LLM 连接、Judge 设置等。
敏感信息（API key 等）通过环境变量或 `.env` 文件设置。

## 调试命令

```bash
# 麦克风测试
python -m voice_agent.main --mic-test

# ASR 测试（只测试语音识别，不调 LLM）
python -m voice_agent.main --asr-test sherpa-onnx

# 测试本地 Judge 判断
python -m voice_agent.main --judge-test "这剧情怎么这样"

# 测试琉璃川人格 Prompt 效果
python -m voice_agent.main --persona-test

# 列出音频设备
python -m voice_agent.main --list-devices

# 无 TUI 后台模式
python -m voice_agent.main --headless
```

## TUI 命令

程序启动后，在底部输入框可输入命令：

| 命令 | 说明 |
|------|------|
| `/help` | 显示帮助 |
| `/status` | 系统状态 |
| `/debug` | 显示 Gate/Judge/ASR 内部状态 |
| `/mic monitor` | 实时麦克风 VU 音量监测 |
| `/mic list` | 列出音频设备 |
| `/mic select <id>` | 选择麦克风 |
| `/mic autodetect` | 自动检测有音频输入的设备 |
| `/pause` | 暂停 AI 回应 |
| `/resume` | 恢复 AI 回应 |
| `/clear` | 清屏 |
| `/exit` | 退出 |

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
├── cli/                    # 交互式 TUI
│   ├── dynamic_shell.py    # 全屏 prompt_toolkit TUI
│   ├── tui_renderer.py     # OpenCode 风格渲染器
│   ├── ui_state.py         # UI 状态模型
│   ├── formatters.py       # 格式化函数
│   └── shell.py            # 基础 shell
├── core/                   # 核心逻辑
│   ├── agent_core.py       # Agent 核心
│   ├── intervention_gate.py# 介入判断器
│   ├── local_judge_client.py # 本地语义 Judge
│   ├── health_check.py     # 运行时健康检查
│   ├── conversation_state.py
│   └── llm_client.py       # LLM 客户端
├── output/                 # 输出
│   ├── console_output.py   # 控制台输出
│   └── websocket_server.py # WebSocket 输出
├── utils/
│   └── text_normalizer.py
├── config.py
├── event_bus.py
├── logger.py
└── main.py                 # 入口
```
