"""Two threads each register their own progress sink and run a tqdm-update
sequence. Confirms the routing layer in ``_demucs_progress`` is truly
thread-local — sinks must NEVER see counts that belong to the other thread.

This test exercises the threading mechanism without invoking Demucs, so it
runs in milliseconds.
"""
import io
import threading
import time

from audio_analysis_mcp._demucs_progress import (
    _RoutingTqdm,
    clear_sink,
    install_sink,
)


def _silent_tqdm(total: int) -> _RoutingTqdm:
    """Construct a routing tqdm whose stderr output is swallowed but whose
    ``update`` method still runs (``disable=True`` short-circuits ``update``
    entirely, which would defeat the routing layer)."""
    return _RoutingTqdm(total=total, file=io.StringIO(), mininterval=0)


def test_concurrent_threads_have_isolated_progress_sinks() -> None:
    captured_a: list[tuple[int, int]] = []
    captured_b: list[tuple[int, int]] = []

    barrier = threading.Barrier(2)

    def thread_a() -> None:
        install_sink(lambda current, total: captured_a.append((current, total)))
        try:
            barrier.wait()
            bar = _silent_tqdm(total=5)
            for _ in range(5):
                bar.update(1)
                time.sleep(0.001)  # tiny yield to let the other thread interleave
            bar.close()
        finally:
            clear_sink()

    def thread_b() -> None:
        install_sink(lambda current, total: captured_b.append((current, total)))
        try:
            barrier.wait()
            bar = _silent_tqdm(total=10)
            for _ in range(10):
                bar.update(1)
                time.sleep(0.001)
            bar.close()
        finally:
            clear_sink()

    ta = threading.Thread(target=thread_a)
    tb = threading.Thread(target=thread_b)
    ta.start(); tb.start()
    ta.join(); tb.join()

    # Each thread's sink must only see updates from ITS OWN bar.
    assert len(captured_a) == 5
    assert all(total == 5 for _, total in captured_a)
    assert [current for current, _ in captured_a] == [1, 2, 3, 4, 5]

    assert len(captured_b) == 10
    assert all(total == 10 for _, total in captured_b)
    assert [current for current, _ in captured_b] == list(range(1, 11))


def test_cleared_sink_drops_subsequent_updates() -> None:
    captured: list[tuple[int, int]] = []
    install_sink(lambda c, t: captured.append((c, t)))
    bar = _silent_tqdm(total=3)
    bar.update(1)
    clear_sink()
    bar.update(1)
    bar.update(1)
    bar.close()
    # Only the first update happened while the sink was installed.
    assert captured == [(1, 3)]
