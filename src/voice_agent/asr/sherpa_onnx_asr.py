"""SherpaOnnx ASR 适配器（预留实现）。

需要安装 sherpa-onnx 后才能使用：
  pip install sherpa-onnx
"""

from voice_agent.event_bus import EventBus
from voice_agent.logger import get_logger


class SherpaOnnxASR:
    """sherpa-onnx 流式 ASR 引擎（预留）。"""

    def __init__(self, event_bus: EventBus, config: dict) -> None:
        self._bus = event_bus
        self._config = config
        self._running = False
        self._logger = get_logger()

    async def start(self) -> None:
        try:
            import sherpa_onnx  # noqa: F401
        except ImportError:
            self._logger.error(
                "[SherpaOnnxASR] sherpa-onnx 未安装。请运行: pip install sherpa-onnx"
            )
            self._logger.error(
                "[SherpaOnnxASR] 模型下载: https://github.com/k2-fsa/sherpa-onnx/releases"
            )
            raise RuntimeError("sherpa-onnx 未安装")

        self._running = True
        self._logger.info("[SherpaOnnxASR] 已启动（TODO: 实现流式识别）")
        # TODO: 实现 sherpa-onnx 流式识别
        # 1. 加载模型配置
        # 2. 打开麦克风
        # 3. 流式传入音频
        # 4. 检测 VAD 端点
        # 5. 发送 asr.partial / asr.final 事件

    async def stop(self) -> None:
        self._running = False
        self._logger.info("[SherpaOnnxASR] 已停止")

    async def set_callback(self, cb) -> None:
        pass
