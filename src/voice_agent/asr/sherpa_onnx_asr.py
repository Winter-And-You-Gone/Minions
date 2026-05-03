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

    async def start(self) -> None:
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
            print(f"  ❌ {msg}", flush=True)
            raise RuntimeError("sherpa-onnx 未安装")

        # 2) 校验配置
        self._validate_config()
        sample_rate = self._config.get("sample_rate", 16000)
        chunk_ms = self._config.get("chunk_ms", 100)

        # 3) 创建 recognizer
        self._recognizer = self._create_recognizer()

        # 4) 创建麦克风
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

        # 6) 启动主循环
        self._running = True
        self._logger.info("[SherpaOnnxASR] 启动成功，等待语音输入...")
        print("  🎤 [ASR] 已启动，请说话...", flush=True)

        try:
            await self._mic.start()
            await self._run_loop(sample_rate)
        except asyncio.CancelledError:
            self._logger.info("[SherpaOnnxASR] 被取消")
        except Exception as e:
            self._logger.error("[SherpaOnnxASR] 运行错误: %s", e)
            print(f"  ❌ [ASR] 运行错误: {e}", flush=True)
        finally:
            await self._mic.stop()
            self._running = False
            self._logger.info("[SherpaOnnxASR] 已退出")

    async def stop(self) -> None:
        self._running = False
        if self._mic:
            await self._mic.stop()
        self._logger.info("[SherpaOnnxASR] 已停止")

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
        """创建 sherpa-onnx OfflineRecognizer，适配不同模型配置。"""
        import sherpa_onnx

        tokens = self._config.get("tokens", "")
        model = self._config.get("model", "")
        encoder = self._config.get("encoder", "")
        decoder = self._config.get("decoder", "")
        joiner = self._config.get("joiner", "")
        num_threads = self._config.get("num_threads", 2)
        decoding_method = self._config.get("decoding_method", "greedy_search")
        provider = self._config.get("provider", "cpu")

        # 构建离线模型配置
        if model:
            # SenseVoice / Paraformer 等单模型
            feat_config = sherpa_onnx.OfflineModelConfig(
                tokens=tokens,
                num_threads=num_threads,
                provider=provider,
            )
            # 尝试多种模型类型
            model_path = str(Path(model).resolve())
            if "sensevoice" in model.lower() or "sensvoice" in model.lower():
                feat_config.sense_voice = sherpa_onnx.OfflineSenseVoiceModelConfig(
                    model=model_path,
                )
            elif "paraformer" in model.lower():
                feat_config.paraformer = sherpa_onnx.OfflineParaformerModelConfig(
                    model=model_path,
                )
            elif "whisper" in model.lower():
                feat_config.whisper = sherpa_onnx.OfflineWhisperModelConfig(
                    model=model_path,
                )
            elif "nemo" in model.lower() or "neMo" in model:
                feat_config.nemo_ctc = sherpa_onnx.OfflineNemoEncDecModelConfig(
                    model=model_path,
                )
            else:
                # 默认尝试 Paraformer
                feat_config.paraformer = sherpa_onnx.OfflineParaformerModelConfig(
                    model=model_path,
                )
        else:
            # Transducer 模型
            feat_config = sherpa_onnx.OfflineModelConfig(
                transducer=sherpa_onnx.OfflineTransducerModelConfig(
                    encoder=str(Path(encoder).resolve()),
                    decoder=str(Path(decoder).resolve()),
                    joiner=str(Path(joiner).resolve()),
                ),
                tokens=tokens,
                num_threads=num_threads,
                provider=provider,
            )

        recognizer_config = sherpa_onnx.OfflineRecognizerConfig(
            model=feat_config,
            decoding_method=decoding_method,
        )

        self._logger.info(
            "[SherpaOnnxASR] 创建 recognizer: model=%s tokens=%s threads=%d method=%s",
            model or f"encoder={encoder}",
            tokens,
            num_threads,
            decoding_method,
        )
        return sherpa_onnx.OfflineRecognizer(recognizer_config)

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
                    text = await loop.run_in_executor(
                        None, self._recognize_audio, segment, sample_rate
                    )
                except Exception as e:
                    self._logger.error("[SherpaOnnxASR] 识别错误: %s", e)
                    await self._bus.publish({"type": "asr.error", "message": str(e)})
                    continue

                if text:
                    await self._bus.publish({
                        "type": "asr.final",
                        "text": text,
                        "confidence": 1.0,
                        "engine": "sherpa-onnx",
                    })
                else:
                    self._logger.debug("[SherpaOnnxASR] 识别结果为空，跳过")

    def _recognize_audio(self, samples: np.ndarray, sample_rate: int) -> str:
        """在 executor 中执行离线识别。"""
        import numpy as np
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
