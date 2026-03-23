"""tests/test_streaming.py — Streamer 기본 테스트."""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from src.streaming.streamer import StreamFrame, Streamer
from tests.conftest import blank_frame, make_threat_result
from src.threat_detection.detector import ThreatLevel


def make_stream_frame(index: int = 0) -> StreamFrame:
    return StreamFrame(frame_index=index, raw_frame=blank_frame())


# ---------------------------------------------------------------------------
# StreamFrame
# ---------------------------------------------------------------------------

class TestStreamFrame:
    def test_fields(self) -> None:
        frame = make_stream_frame(5)
        assert frame.frame_index == 5
        assert frame.raw_frame.shape == (64, 64, 3)
        assert frame.protection_response is None
        assert frame.metadata == {}

    def test_metadata_stored(self) -> None:
        frame = StreamFrame(
            frame_index=0,
            raw_frame=blank_frame(),
            metadata={"source": "camera"},
        )
        assert frame.metadata["source"] == "camera"


# ---------------------------------------------------------------------------
# Streamer — start / stop
# ---------------------------------------------------------------------------

class TestStreamerStartStop:
    @pytest.mark.asyncio
    async def test_initial_state_not_running(self) -> None:
        streamer = Streamer()
        assert streamer.is_running is False

    @pytest.mark.asyncio
    async def test_start_sets_running(self) -> None:
        streamer = Streamer()
        await streamer.start()
        assert streamer.is_running is True

    @pytest.mark.asyncio
    async def test_stop_clears_running(self) -> None:
        streamer = Streamer()
        await streamer.start()
        await streamer.stop()
        assert streamer.is_running is False


# ---------------------------------------------------------------------------
# Streamer — push
# ---------------------------------------------------------------------------

class TestStreamerPush:
    @pytest.mark.asyncio
    async def test_push_noop_when_not_running(self) -> None:
        streamer = Streamer(queue_size=4)
        await streamer.push(make_stream_frame(0))  # 예외 없이 무시
        assert streamer._queue.empty()

    @pytest.mark.asyncio
    async def test_push_enqueues_frame(self) -> None:
        streamer = Streamer(queue_size=4)
        await streamer.start()
        await streamer.push(make_stream_frame(1))
        assert streamer._queue.qsize() == 1
        await streamer.stop()

    @pytest.mark.asyncio
    async def test_push_drops_frame_when_queue_full(self) -> None:
        streamer = Streamer(queue_size=2)
        await streamer.start()
        await streamer.push(make_stream_frame(0))
        await streamer.push(make_stream_frame(1))
        # 큐가 가득 찬 상태에서 추가 push → 예외 없이 드롭
        await streamer.push(make_stream_frame(2))
        # 큐 크기는 2를 초과하지 않음
        assert streamer._queue.qsize() <= 2
        await streamer.stop()


# ---------------------------------------------------------------------------
# Streamer — stream (async generator)
# ---------------------------------------------------------------------------

class TestStreamerStream:
    @pytest.mark.asyncio
    async def test_stream_yields_pushed_frames(self) -> None:
        streamer = Streamer(queue_size=8)
        await streamer.start()
        await streamer.push(make_stream_frame(10))
        await streamer.push(make_stream_frame(11))
        await streamer.stop()

        received: list[StreamFrame] = []
        async for frame in streamer.stream():
            received.append(frame)

        assert len(received) == 2
        assert received[0].frame_index == 10
        assert received[1].frame_index == 11

    @pytest.mark.asyncio
    async def test_stream_stops_on_sentinel(self) -> None:
        streamer = Streamer(queue_size=4)
        await streamer.start()
        await streamer.stop()  # sentinel 삽입

        received: list[StreamFrame] = []
        async for frame in streamer.stream():
            received.append(frame)

        assert received == []


# ---------------------------------------------------------------------------
# Streamer — run_pipeline
# ---------------------------------------------------------------------------

class TestStreamerRunPipeline:
    @pytest.mark.asyncio
    async def test_run_pipeline_processes_all_frames(self) -> None:
        async def fake_source():
            for _ in range(3):
                yield blank_frame()

        processed: list[int] = []

        def process_fn(frame: np.ndarray, index: int):
            processed.append(index)
            return None  # ProtectionResponse 없음

        streamer = Streamer(fps=1000.0, queue_size=16)
        await streamer.run_pipeline(fake_source(), process_fn)

        assert processed == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_run_pipeline_pushes_stream_frames(self) -> None:
        async def fake_source():
            for _ in range(2):
                yield blank_frame()

        streamer = Streamer(fps=1000.0, queue_size=16)

        collected: list[StreamFrame] = []

        async def collect() -> None:
            async for frame in streamer.stream():
                collected.append(frame)

        collector_task = asyncio.create_task(collect())
        await streamer.run_pipeline(fake_source(), lambda f, i: None)
        await collector_task

        assert len(collected) == 2
        assert collected[0].frame_index == 0
        assert collected[1].frame_index == 1
