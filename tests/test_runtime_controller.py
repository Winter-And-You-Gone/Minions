"""测试 RuntimeController — ASR 生命周期管理。"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from voice_agent.core.runtime_controller import RuntimeController
from voice_agent.event_bus import EventBus


class FakeASREngine:
    """模拟 ASR 引擎，不依赖于实际模型文件。"""

    def __init__(self, engine_name: str, bus: EventBus, config: dict):
        self.engine_name = engine_name
        self.bus = bus
        self.config = config
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True
        await self.bus.publish({
            "type": "asr.status",
            "status": "listening",
            "message": "语音监听已启动",
        })

    async def stop(self) -> None:
        self.stopped = True


def make_runtime_controller(
    engine_name: str = "mock",
    asr_factory=FakeASREngine,
) -> RuntimeController:
    bus = EventBus()
    config = {"runtime": {"autostart": False}}
    return RuntimeController(
        bus=bus,
        config=config,
        asr_engine_name=engine_name,
        asr_factory=asr_factory,
    )


@pytest.mark.asyncio
async def test_default_state_is_sleeping():
    ctrl = make_runtime_controller()
    assert ctrl.state == "sleeping"
    assert not ctrl.is_listening


@pytest.mark.asyncio
async def test_wakeup_transitions_to_listening():
    ctrl = make_runtime_controller()
    assert ctrl.state == "sleeping"
    ok = await ctrl.wakeup()
    assert ok
    assert ctrl.state == "listening"
    assert ctrl.is_listening


@pytest.mark.asyncio
async def test_wakeup_creates_asr_engine():
    ctrl = make_runtime_controller()
    assert ctrl.asr_engine is None
    await ctrl.wakeup()
    assert ctrl.asr_engine is not None
    assert ctrl.asr_engine.started


@pytest.mark.asyncio
async def test_wakeup_idempotent():
    ctrl = make_runtime_controller()
    ok1 = await ctrl.wakeup()
    ok2 = await ctrl.wakeup()
    assert ok1 and ok2
    assert ctrl.state == "listening"


@pytest.mark.asyncio
async def test_sleep_transitions_to_sleeping():
    ctrl = make_runtime_controller()
    await ctrl.wakeup()
    assert ctrl.state == "listening"
    await ctrl.sleep()
    assert ctrl.state == "sleeping"
    assert not ctrl.is_listening


@pytest.mark.asyncio
async def test_sleep_stops_asr_engine():
    ctrl = make_runtime_controller()
    await ctrl.wakeup()
    engine = ctrl.asr_engine
    await ctrl.sleep()
    assert engine.stopped


@pytest.mark.asyncio
async def test_sleep_idempotent():
    ctrl = make_runtime_controller()
    await ctrl.sleep()  # already sleeping
    assert ctrl.state == "sleeping"


@pytest.mark.asyncio
async def test_close_stops_engine():
    ctrl = make_runtime_controller()
    await ctrl.wakeup()
    engine = ctrl.asr_engine
    await ctrl.close()
    assert ctrl.state == "sleeping"
    assert engine.stopped


@pytest.mark.asyncio
async def test_publishes_runtime_status_on_wakeup():
    events: list[dict] = []
    bus = EventBus()
    bus.subscribe(lambda e: events.append(e) if e.get("type") == "runtime.status" else None)

    config = {"runtime": {"autostart": False}}
    ctrl = RuntimeController(
        bus=bus, config=config,
        asr_engine_name="mock", asr_factory=FakeASREngine,
    )
    await ctrl.wakeup()

    status_events = [e for e in events if e.get("type") == "runtime.status"]
    assert len(status_events) >= 1
    assert status_events[-1]["state"] == "listening"


@pytest.mark.asyncio
async def test_factory_failure_returns_false():
    def failing_factory(engine_name, bus, config):
        raise RuntimeError("模型加载失败")

    ctrl = make_runtime_controller(asr_factory=failing_factory)
    ok = await ctrl.wakeup()
    assert not ok
    assert ctrl.state == "error"


@pytest.mark.asyncio
async def test_asr_engine_name_property():
    ctrl = make_runtime_controller(engine_name="sherpa-onnx")
    assert ctrl._asr_engine_name == "sherpa-onnx"
