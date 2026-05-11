"""Thread-local Demucs progress routing.

Demucs's ``apply_model(..., progress=True)`` writes a tqdm bar to stderr. The
only documented way to intercept it is to replace ``demucs.apply.tqdm.tqdm``
with a subclass — but that's a process-global side effect. Earlier code in
``cli/stem_separate.py`` did this with a ``nonlocal counter`` captured in a
closure, which is fine for single-threaded CLI use but incompatible with
concurrent stem jobs (two threads would interleave counter increments).

This module installs the global tqdm replacement ONCE at import time. Each
request thread sets its own progress sink via ``install_sink(...)`` before
calling ``apply_model``; the patched tqdm reads that thread-local sink, so two
concurrent stem jobs each see only their own counter sequence.
"""
from __future__ import annotations

import threading
from typing import Callable

import demucs.apply
from tqdm import tqdm as _tqdm_base

_progress_sink = threading.local()
"""Per-thread sink. Callers set ``_progress_sink.sink = callback`` before
running Demucs; ``_RoutingTqdm.update`` reads it via ``getattr(...)``."""


class _RoutingTqdm(_tqdm_base):  # type: ignore[misc]
    """tqdm subclass that mirrors every ``update`` call to the calling
    thread's progress sink. Stderr rendering is preserved for the CLI case."""

    def update(self, n: float = 1) -> None:
        super().update(n)
        sink = getattr(_progress_sink, "sink", None)
        if sink is not None:
            try:
                sink(int(self.n), int(self.total) if self.total else 0)
            except Exception:
                # Never let a buggy sink take down Demucs.
                pass


# Install once at module import. Idempotent — re-importing this module from
# the same process is a no-op because demucs.apply.tqdm.tqdm is already
# replaced with our subclass (or with a re-subclassed version of it).
demucs.apply.tqdm.tqdm = _RoutingTqdm  # type: ignore[attr-defined]


def install_sink(sink: Callable[[int, int], None]) -> None:
    """Register a progress sink for the calling thread.

    ``sink(current, total)`` is called on every tqdm ``update``. Both
    arguments are integers — typically Demucs reports ``current`` ticking
    from 1 to ``model.shifts * len(model.models)``.

    Must be paired with ``clear_sink()`` in a ``finally`` block so the sink
    doesn't leak to a future call that happens to land on the same worker
    thread.
    """
    _progress_sink.sink = sink


def clear_sink() -> None:
    """Drop the calling thread's progress sink."""
    _progress_sink.sink = None
