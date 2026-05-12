"""SherpaOnnx 流式 ASR — OnlineRecognizer 实时逐字识别。

工作流程:
  麦克风持续采集
  → RMS VAD 检测语音
  → 说话时创建 OnlineStream，边采边识别
  → 发布 asr.partial（中间结果）
  → 说话结束发布 asr.final（最终结果）
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import numpy as np

from voice_agent.audio.microphone import Microphone, calculate_rms
from voice_agent.audio.segmenter import (
    VADSegmenterConfig,
    SpeechSegmenter,
    _EVENT_SPEECH_START,
    _EVENT_SPEECH_END,
    _EVENT_SPEECH_FORCED_END,
    _EVENT_SPEECHING,
)
from voice_agent.event_bus import EventBus
from voice_agent.logger import get_logger


_STREAMING_MAX_SEGMENT_S = 5.0       # 最长连续识别窗口（秒），超时结束当前 stream
_STREAMING_SILENCE_DELAY_S = 0.6     # 尾部静音后等待秒数再结束识别
_PARTIAL_INTERVAL_S = 0.3            # 发布 partial 结果的间隔


class SherpaOnnxStreamingASR:
    """sherpa-onnx OnlineRecognizer 流式识别引擎。"""

    def __init__(self, event_bus: EventBus, config: dict) -> None:
        self._bus = event_bus
        self._config = config
        self._running = False
        self._logger = get_logger()
        self._recognizer: Any = None
        self._mic: Microphone | None = None
        self._segmenter: SpeechSegmenter | None = None
        self._result_config = config.get("result", {})
        self._asr_cfg = config.get("sherpa_onnx", config)

    async def start(self) -> None:
        try:
            try:
                import sherpa_onnx  # noqa: F401
            except ImportError:
                msg = (
                    "sherpa-onnx 未安装，请运行:\n"
                    "  pip install -e \".[asr]\"\n"
                    "或:\n"
                    "  pip install sherpa-onnx"
                )
                self._logger.error("[StreamingASR] %s", msg)
                raise RuntimeError("sherpa-onnx 未安装")

            self._validate_config()
            sample_rate = self._asr_cfg.get("sample_rate", 16000)
            chunk_ms = self._asr_cfg.get("chunk_ms", 100)

            await self._bus.publish({
                "type": "asr.status",
                "status": "loading_model",
                "message": "正在加载流式 sherpa-onnx 模型...",
            })
            self._recognizer = self._create_recognizer()
            await self._bus.publish({
                "type": "asr.status",
                "status": "model_loaded",
                "message": "流式语音识别模型加载完成",
            })

            await self._bus.publish({
                "type": "asr.status",
                "status": "starting_microphone",
                "message": "正在启动麦克风...",
            })
            self._mic = Microphone(
                sample_rate=sample_rate,
                channels=self._asr_cfg.get("channels", 1),
                chunk_ms=chunk_ms,
                device=self._asr_cfg.get("device"),
            )

            vad_cfg = self._asr_cfg.get("vad", {})
            segmenter_cfg = VADSegmenterConfig(
                sample_rate=sample_rate,
                chunk_ms=chunk_ms,
                rms_threshold=vad_cfg.get("rms_threshold", 0.008),
                min_speech_ms=vad_cfg.get("min_speech_ms", 300),
                max_speech_ms=vad_cfg.get("max_speech_ms", 15000),
                silence_timeout_ms=vad_cfg.get("silence_timeout_ms", 800),
                pre_roll_ms=vad_cfg.get("pre_roll_ms", 300),
            )
            self._segmenter = SpeechSegmenter(segmenter_cfg)

            self._running = True
            await self._bus.publish({
                "type": "asr.status",
                "status": "listening",
                "message": "正在监听麦克风（流式），请说话",
            })
            self._logger.info("[StreamingASR] 启动成功")

            try:
                await self._mic.start()
                await self._run_loop(sample_rate)
            except asyncio.CancelledError:
                self._logger.info("[StreamingASR] 被取消")
            except Exception as e:
                self._logger.error("[StreamingASR] 运行错误: %s", e)
        finally:
            await self._cleanup()

    async def stop(self) -> None:
        self._running = False
        await self._cleanup()
        self._logger.info("[StreamingASR] 已停止")

    async def _cleanup(self) -> None:
        self._running = False
        if self._mic is not None:
            try:
                await self._mic.stop()
            except Exception as e:
                self._logger.warning("[StreamingASR] mic 清理异常: %s", e)

    def _validate_config(self) -> None:
        cfg = self._asr_cfg
        tokens = cfg.get("tokens", "")
        if not tokens or not Path(tokens).exists():
            raise ValueError(f"Streaming ASR 配置错误: tokens 文件不存在: {tokens}")
        for key in ("encoder", "decoder", "joiner"):
            val = cfg.get(key, "")
            if not val or not Path(val).exists():
                raise ValueError(f"Streaming ASR 配置错误: {key} 文件不存在: {val}")

    def _create_recognizer(self) -> Any:
        import sherpa_onnx

        cfg = self._asr_cfg
        model_dir = Path(cfg.get("model_dir", cfg.get("encoder", ""))).parent
        tokens = str(Path(cfg["tokens"]).resolve())
        encoder = str(Path(cfg["encoder"]).resolve())
        decoder = str(Path(cfg["decoder"]).resolve())
        joiner = str(Path(cfg["joiner"]).resolve())
        num_threads = cfg.get("num_threads", 2)
        provider = cfg.get("provider", "cpu")

        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens,
            num_threads=num_threads,
            sample_rate=cfg.get("sample_rate", 16000),
            feature_dim=80,
            decoding_method=cfg.get("decoding_method", "greedy_search"),
            provider=provider,
        )
        self._logger.info("[StreamingASR] 创建 OnlineRecognizer: encoder=%s, threads=%d",
                          Path(encoder).name, num_threads)
        return recognizer

    async def _run_loop(self, sample_rate: int) -> None:
        import sherpa_onnx

        loop = asyncio.get_running_loop()
        assert self._segmenter is not None
        assert self._mic is not None

        streaming: Any = None
        streaming_start_time: float = 0.0
        last_speech_time: float = 0.0
        last_partial_time: float = 0.0
        segment_samples: list[np.ndarray] = []

        async def _finalize_stream() -> str | None:
            nonlocal streaming, segment_samples
            if streaming is None:
                return None
            try:
                streaming.input_finished()
                while self._recognizer.is_ready(streaming):
                    self._recognizer.decode_stream(streaming)
                result = self._recognizer.get_result(streaming)
                text = result.text.strip() if result else ""
                self._recognizer.reset(streaming)
                streaming = None
                segment_samples = []
                return text
            except Exception as exc:
                self._logger.error("[StreamingASR] finalize 失败: %s", exc)
                streaming = None
                segment_samples = []
                return None

        while self._running:
            try:
                chunk = await asyncio.wait_for(self._mic.read_chunk(), timeout=0.3)
            except asyncio.TimeoutError:
                now = time.monotonic()
                if (streaming is not None
                        and segment_samples
                        and now - last_speech_time > _STREAMING_SILENCE_DELAY_S):
                    text = await _finalize_stream()
                    if text and self._should_publish_text(text):
                        self._logger.info("[StreamingASR] 静音超时 final: %s", text)
                        await self._bus.publish({
                            "type": "asr.final",
                            "text": text,
                            "confidence": 1.0,
                            "engine": "sherpa-onnx-streaming",
                        })
                continue

            rms = calculate_rms(chunk)
            await self._bus.publish({"type": "audio.level", "rms": rms})

            event_type, segment, seg_rms = self._segmenter.accept_chunk(chunk)

            if event_type == _EVENT_SPEECH_START:
                if streaming is None:
                    streaming = self._recognizer.create_stream()
                streaming_start_time = time.monotonic()
                last_speech_time = streaming_start_time
                last_partial_time = 0.0
                segment_samples = []

            if streaming is not None and event_type in (
                _EVENT_SPEECH_START, _EVENT_SPEECHING,
                _EVENT_SPEECH_END, _EVENT_SPEECH_FORCED_END,
            ):
                segment_samples.append(chunk.copy())
                streaming.accept_waveform(sample_rate, chunk)
                while self._recognizer.is_ready(streaming):
                    self._recognizer.decode_stream(streaming)
                last_speech_time = time.monotonic()

                now = time.monotonic()
                if now - last_partial_time >= _PARTIAL_INTERVAL_S:
                    result = self._recognizer.get_result(streaming)
                    partial = result.text.strip() if result else ""
                    if partial:
                        self._logger.debug("[StreamingASR] partial: %s", partial)
                        await self._bus.publish({
                            "type": "asr.partial",
                            "text": partial,
                            "engine": "sherpa-onnx-streaming",
                        })
                    last_partial_time = now

            if event_type in (_EVENT_SPEECH_END, _EVENT_SPEECH_FORCED_END):
                forced = event_type == _EVENT_SPEECH_FORCED_END
                duration_ms = int(len(np.concatenate(segment_samples)) / sample_rate * 1000) if segment_samples else 0
                await self._bus.publish({
                    "type": "asr.speech_end",
                    "duration_ms": duration_ms,
                    "forced": forced,
                })

                if streaming is not None:
                    text = await _finalize_stream()
                    if text and self._should_publish_text(text):
                        self._logger.info("[StreamingASR] final: %s", text)
                        await self._bus.publish({
                            "type": "asr.final",
                            "text": text,
                            "confidence": 1.0,
                            "engine": "sherpa-onnx-streaming",
                        })
                        await self._bus.publish({
                            "type": "asr.status",
                            "status": "recognized",
                            "message": "语音识别完成",
                        })
                    else:
                        self._logger.debug("[StreamingASR] 识别结果被过滤或为空")
                segment_samples = []
                streaming = None

            if (streaming is not None
                    and segment_samples
                    and time.monotonic() - streaming_start_time > _STREAMING_MAX_SEGMENT_S):
                self._logger.warning("[StreamingASR] 强制截断 (>%.0fs)", _STREAMING_MAX_SEGMENT_S)
                text = await _finalize_stream()
                if text and self._should_publish_text(text):
                    await self._bus.publish({
                        "type": "asr.final",
                        "text": text,
                        "confidence": 1.0,
                        "engine": "sherpa-onnx-streaming",
                    })
                segment_samples = []

    def _should_publish_text(self, text: str) -> bool:
        min_text_length = int(self._result_config.get("min_text_length", 1))
        normalized = text.strip()
        if not normalized:
            return False
        if len(normalized) < min_text_length:
            return False
        return True
