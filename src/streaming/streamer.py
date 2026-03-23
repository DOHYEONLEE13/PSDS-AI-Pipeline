from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.protection.protector import ProtectionResponse

logger = logging.getLogger(__name__)


@dataclass
class StreamFrame:
    """스트리밍 파이프라인의 단일 출력 프레임."""

    frame_index: int
    raw_frame: np.ndarray
    protection_response: ProtectionResponse | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Streamer:
    """비동기 프레임 스트리밍 및 파이프라인 오케스트레이션."""

    def __init__(self, fps: float = 30.0, queue_size: int = 64) -> None:
        self._fps = fps
        self._interval = 1.0 / fps
        self._queue: asyncio.Queue[StreamFrame | None] = asyncio.Queue(maxsize=queue_size)
        self._running = False
        self._subscribers: list[asyncio.Queue[StreamFrame]] = []

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """스트리머를 시작합니다."""
        self._running = True
        logger.info("Streamer 시작 (%.1f fps)", self._fps)

    async def stop(self) -> None:
        """스트리머를 중지하고 모든 구독자에게 종료를 알립니다."""
        self._running = False
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        await self._queue.put(None)  # sentinel
        logger.info("Streamer 중지")

    async def push(self, frame: StreamFrame) -> None:
        """처리된 프레임을 큐에 넣습니다."""
        if not self._running:
            return
        try:
            self._queue.put_nowait(frame)
        except asyncio.QueueFull:
            logger.warning("스트림 큐 가득 참, 프레임 드롭 (index=%d)", frame.frame_index)

    async def stream(self) -> AsyncGenerator[StreamFrame, None]:
        """프레임을 비동기 제너레이터로 소비합니다."""
        while True:
            frame = await self._queue.get()
            if frame is None:
                break
            yield frame

    async def run_pipeline(
        self,
        frame_source: AsyncGenerator[np.ndarray, None],
        process_fn: Any,  # Callable[[np.ndarray, int], ProtectionResponse]
    ) -> None:
        """프레임 소스를 읽어 처리 함수를 거쳐 스트림에 푸시합니다."""
        await self.start()
        index = 0
        try:
            async for raw in frame_source:
                if not self._running:
                    break
                response = await asyncio.to_thread(process_fn, raw, index)
                await self.push(StreamFrame(
                    frame_index=index,
                    raw_frame=raw,
                    protection_response=response,
                ))
                index += 1
                await asyncio.sleep(self._interval)
        finally:
            await self.stop()
