"""Bridge between the synchronous progress callbacks emitted by
``stem_separate_impl`` / ``analyze_structure`` and the async SSE response.

The work runs in an anyio thread pool worker. We can't ``await`` from there,
so the worker drops progress events into a bounded memory stream and the
SSE handler async-iterates over the receiver to emit them.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream


class ProgressChannel:
    """Sync-write, async-read progress event bridge.

    Worker thread calls :py:meth:`sync_emit`; the SSE handler iterates with
    :py:meth:`stream`. The buffer is sized so a typical run can fit all
    expected events without dropping; ``send_nowait`` silently drops on
    overflow so progress never blocks the worker.
    """

    def __init__(self, buffer: int = 256) -> None:
        send, recv = anyio.create_memory_object_stream[dict[str, Any]](buffer)
        self._send: MemoryObjectSendStream[dict[str, Any]] = send
        self._recv: MemoryObjectReceiveStream[dict[str, Any]] = recv
        self._closed = False

    def sync_emit(self, stage: str, fraction: float, detail: str | None = None) -> None:
        """Push a progress event from the worker thread. Never blocks."""
        if self._closed:
            return
        try:
            self._send.send_nowait({"stage": stage, "fraction": fraction, "detail": detail})
        except anyio.WouldBlock:
            # Drop progress under back-pressure rather than slowing the worker.
            pass
        except anyio.ClosedResourceError:
            # Channel already closed (job finished or errored) — nothing to do.
            pass

    async def stream(self) -> AsyncIterator[dict[str, Any]]:
        """Async-iterate over emitted progress events until the channel closes."""
        async with self._recv:
            async for evt in self._recv:
                yield evt

    def close(self) -> None:
        """Close the sender so ``stream()`` exits. Idempotent."""
        if not self._closed:
            self._closed = True
            self._send.close()
