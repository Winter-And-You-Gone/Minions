"""SherpaOnnx ASR 适配器 — 非流式 VAD 分段识别。

工作流程:
  麦克风持续采集
  → RMS VAD 分段
  → 语音结束触发 sherpa-onnx offline recognizer
  → 发布 asr.final
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np

from voice_agent.audio.microphone import Microphone, calculate_rms
from voice_agent.audio.segmenter import (
    VADSegmenterConfig,
    SpeechSegmenter,
    _EVENT_SILENCE,
    _EVENT_SPEECH_START,
    _EVENT_SPEECHING,
    _EVENT_SPEECH_END,
    _EVENT_SPEECH_FORCED_END,
)
from voice_agent.event_bus import EventBus
from voice_agent.logger import get_logger


class SherpaOnnxASR:
    """sherpa-onnx 非流式 VAD 分段识别引擎。"""

    def __init__(self, event_bus: EventBus, config: dict) -> None:
        self._bus = event_bus
        self._config = config
        self._running = False
        self._logger = get_logger()
        self._recognizer = None
        self._mic: Microphone | None = None
        self._segmenter: SpeechSegmenter | None = None
        self._result_config = config.get("result", {})

    async def start(self) -> None:
        try:
            # 1) 检查依赖安装
            try:
                import sherpa_onnx  # noqa: F401
            except ImportError:
                msg = (
                    "sherpa-onnx 未安装，请运行:\n"
                    "  pip install -e \".[asr]\"\n"
                    "或:\n"
                    "  pip install sherpa-onnx"
                )
                self._logger.error("[SherpaOnnxASR] %s", msg)
                print(f"  [ASR] 错误: {msg}", flush=True)
                raise RuntimeError("sherpa-onnx 未安装")

            # 2) 校验配置
            self._validate_config()
            sample_rate = self._config.get("sample_rate", 16000)
            chunk_ms = self._config.get("chunk_ms", 100)

            # 3) 创建 recognizer
            await self._bus.publish({
                "type": "asr.status",
                "status": "loading_model",
                "message": "正在加载 sherpa-onnx 模型...",
            })
            self._recognizer = self._create_recognizer()
            await self._bus.publish({
                "type": "asr.status",
                "status": "model_loaded",
                "message": "语音识别模型加载完成",
            })

            # 4) 创建麦克风
            await self._bus.publish({
                "type": "asr.status",
                "status": "starting_microphone",
                "message": "正在启动麦克风...",
            })
            self._mic = Microphone(
                sample_rate=sample_rate,
                channels=self._config.get("channels", 1),
                chunk_ms=chunk_ms,
                device=self._config.get("device"),
            )

            # 5) 创建分段器
            vad_cfg = self._config.get("vad", {})
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

            self._logger.info(
                "[SherpaOnnxASR] VAD 参数: threshold=%.4f min=%dms silence=%dms max=%dms pre_roll=%dms",
                segmenter_cfg.rms_threshold,
                segmenter_cfg.min_speech_ms,
                segmenter_cfg.silence_timeout_ms,
                segmenter_cfg.max_speech_ms,
                segmenter_cfg.pre_roll_ms,
            )
            await self._bus.publish({
                "type": "asr.status",
                "status": "vad_config",
                "message": (
                    f"VAD: threshold={segmenter_cfg.rms_threshold:.4f}, "
                    f"min={segmenter_cfg.min_speech_ms}ms, "
                    f"silence={segmenter_cfg.silence_timeout_ms}ms"
                ),
            })

            # 6) 启动主循环
            self._running = True
            await self._bus.publish({
                "type": "asr.status",
                "status": "listening",
                "message": "正在监听麦克风，请说话",
            })
            self._logger.info("[SherpaOnnxASR] 启动成功，等待语音输入...")

            try:
                await self._mic.start()
                await self._run_loop(sample_rate)
            except asyncio.CancelledError:
                self._logger.info("[SherpaOnnxASR] 被取消")
            except Exception as e:
                self._logger.error("[SherpaOnnxASR] 运行错误: %s", e)
                print(f"  [ASR] 运行错误: {e}", flush=True)
        finally:
            await self._cleanup()

    async def stop(self) -> None:
        self._running = False
        await self._cleanup()
        self._logger.info("[SherpaOnnxASR] 已停止")

    async def _cleanup(self) -> None:
        self._running = False
        if self._mic is not None:
            try:
                await self._mic.stop()
            except Exception as e:
                self._logger.warning("[SherpaOnnxASR] mic 清理异常: %s", e)

    # ---- 内部实现 ----

    def _validate_config(self) -> None:
        """校验配置完整性，缺失项抛出清晰错误。"""
        tokens = self._config.get("tokens", "")
        model = self._config.get("model", "")
        encoder = self._config.get("encoder", "")
        decoder = self._config.get("decoder", "")
        joiner = self._config.get("joiner", "")
        sr = self._config.get("sample_rate", 0)

        if not sr or sr <= 0:
            raise ValueError("SherpaOnnxASR 配置错误: sample_rate 必须为正数")

        if not tokens:
            raise ValueError("SherpaOnnxASR 配置错误: tokens 未设置，请指定 tokens 文件路径")

        tokens_path = Path(tokens)
        if not tokens_path.exists():
            raise ValueError(f"SherpaOnnxASR 配置错误: tokens 文件不存在: {tokens}")

        has_single_model = bool(model)
        has_transducer = bool(encoder) and bool(decoder) and bool(joiner)

        if not has_single_model and not has_transducer:
            raise ValueError(
                "SherpaOnnxASR 配置错误: 请设置 model（单模型）或 encoder+decoder+joiner（transducer）"
            )

        if has_single_model:
            model_path = Path(model)
            if not model_path.exists():
                raise ValueError(f"SherpaOnnxASR 配置错误: 模型文件不存在: {model}")

        if has_transducer:
            for key, val in [("encoder", encoder), ("decoder", decoder), ("joiner", joiner)]:
                if not Path(val).exists():
                    raise ValueError(f"SherpaOnnxASR 配置错误: {key} 文件不存在: {val}")

    def _create_recognizer(self):
        """创建 sherpa-onnx OfflineRecognizer，适配不同模型配置。

        模型类型选择优先级：
          1. config 中显式设置的 ``type`` 字段
          2. 根据 model 文件名自动检测（sense_voice / paraformer / whisper）
          3. encoder+decoder+joiner transducer 配置
        """
        import sherpa_onnx

        tokens = self._config.get("tokens", "")
        model = self._config.get("model", "")
        encoder = self._config.get("encoder", "")
        decoder = self._config.get("decoder", "")
        joiner = self._config.get("joiner", "")
        num_threads = self._config.get("num_threads", 2)
        decoding_method = self._config.get("decoding_method", "greedy_search")
        provider = self._config.get("provider", "cpu")

        # 从配置读取模型类型、语言、ITN 开关
        model_type = self._config.get("type", "").strip().lower()
        language = self._config.get("language", "zh")
        use_itn = self._config.get("use_itn", True)

        model_path = str(Path(model).resolve()) if model else ""

        # 确定模型类型：显式配置 > 文件名推断
        if not model_type:
            if "sensevoice" in model.lower() or "sense_voice" in model.lower() or "sense-voice" in model.lower():
                model_type = "sense_voice"
            elif "paraformer" in model.lower():
                model_type = "paraformer"
            elif "whisper" in model.lower():
                model_type = "whisper"

        if model_type == "sense_voice":
            recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=model_path,
                tokens=str(Path(tokens).resolve()),
                num_threads=num_threads,
                provider=provider,
                language=language,
                use_itn=use_itn,
            )
        elif model_type == "paraformer":
            recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
                paraformer=model_path,
                tokens=str(Path(tokens).resolve()),
                num_threads=num_threads,
                provider=provider,
            )
        elif model_type == "whisper":
            recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
                encoder=model_path,
                decoder="",
                tokens=str(Path(tokens).resolve()),
                num_threads=num_threads,
                provider=provider,
            )
        elif encoder and decoder and joiner:
            recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=str(Path(encoder).resolve()),
                decoder=str(Path(decoder).resolve()),
                joiner=str(Path(joiner).resolve()),
                tokens=str(Path(tokens).resolve()),
                num_threads=num_threads,
                provider=provider,
            )
        else:
            raise ValueError("无法创建 recognizer: 不支持的模型配置")

        self._logger.info(
            "[SherpaOnnxASR] 创建 recognizer: model=%s tokens=%s threads=%d method=%s",
            model or f"encoder={encoder}",
            tokens,
            num_threads,
            decoding_method,
        )
        return recognizer

    def _should_publish_text(self, text: str) -> bool:
        """根据配置过滤识别结果。"""
        min_text_length = int(self._result_config.get("min_text_length", 1))
        ignore_empty = bool(self._result_config.get("ignore_empty", True))

        normalized = text.strip()

        if ignore_empty and not normalized:
            return False

        if len(normalized) < min_text_length:
            return False

        return True

    async def _run_loop(self, sample_rate: int) -> None:
        """主循环: 读麦克风 → 分段 → 识别。"""
        loop = asyncio.get_running_loop()
        assert self._segmenter is not None

        while self._running:
            try:
                chunk = await asyncio.wait_for(self._mic.read_chunk(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            rms = calculate_rms(chunk)
            await self._bus.publish({"type": "audio.level", "rms": rms})

            event_type, segment, seg_rms = self._segmenter.accept_chunk(chunk)

            if event_type == _EVENT_SPEECH_START:
                await self._bus.publish({"type": "asr.speech_start"})

            elif event_type in (_EVENT_SPEECH_END, _EVENT_SPEECH_FORCED_END):
                duration_ms = int(len(segment) / sample_rate * 1000) if segment is not None else 0
                await self._bus.publish({
                    "type": "asr.speech_end",
                    "duration_ms": duration_ms,
                    "forced": event_type == _EVENT_SPEECH_FORCED_END,
                })

                # 在 executor 中运行识别（CPU 密集）
                try:
                    await self._bus.publish({
                        "type": "asr.status",
                        "status": "recognizing",
                        "message": "正在识别语音...",
                    })
                    text = await loop.run_in_executor(
                        None, self._recognize_audio, segment, sample_rate
                    )
                except Exception as e:
                    self._logger.error("[SherpaOnnxASR] 识别错误: %s", e)
                    await self._bus.publish({
                        "type": "asr.status",
                        "status": "error",
                        "message": str(e),
                    })
                    await self._bus.publish({"type": "asr.error", "message": str(e)})
                    continue

                if self._should_publish_text(text):
                    await self._bus.publish({
                        "type": "asr.status",
                        "status": "recognized",
                        "message": "语音识别完成",
                    })
                    await self._bus.publish({
                        "type": "asr.final",
                        "text": text.strip(),
                        "confidence": 1.0,
                        "engine": "sherpa-onnx",
                    })
                else:
                    self._logger.debug("[SherpaOnnxASR] 识别结果被过滤: %r", text)

    def _recognize_audio(self, samples: np.ndarray, sample_rate: int) -> str:
        """在 executor 中执行离线识别。"""
        import sherpa_onnx  # noqa: F401

        # 确保 float32 且为一维
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.ravel()

        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        self._recognizer.decode_stream(stream)
        result = stream.result
        return result.text.strip()
