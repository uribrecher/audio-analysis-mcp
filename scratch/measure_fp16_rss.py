"""Spike: measure SongFormer fp16 vs fp32 memory + accuracy.

Per /Users/uribrecher/.claude/plans/nifty-discovering-toast.md step 1.0.

Usage:
    uv run python scratch/measure_fp16_rss.py          # full A/B + table
    uv run python scratch/measure_fp16_rss.py fp16     # single mode (worker)
    uv run python scratch/measure_fp16_rss.py fp32     # single mode (worker)

The full mode spawns a fresh subprocess per dtype so RSS isn't polluted by
the previous run's freed-but-cached allocator state. Each subprocess prints
a single JSON line to stdout with its measurements; the orchestrator
collects both and prints a comparison table to stderr.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import threading
from pathlib import Path


# 4-min track, normalized 44.1k mono — fits a "realistic" analyze run.
DEFAULT_AUDIO = Path.home() / ".audio-analysis-mcp" / "workspace" / "jobs" / "smooth-criminal-radio-edit" / "source.wav"


def _current_rss_bytes(pid: int) -> int:
    """Cheapest cross-platform RSS read — macOS `ps` returns RSS in KB."""
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)]).strip()
    return int(out) * 1024


def _sample_rss_until(pid: int, stop: threading.Event, interval: float = 1.0) -> list[int]:
    samples: list[int] = []
    while not stop.is_set():
        try:
            samples.append(_current_rss_bytes(pid))
        except Exception:
            break
        stop.wait(interval)
    return samples


def _worker(dtype: str, audio_path: Path) -> None:
    """Run one analyze with the given dtype, print one JSON line of results."""
    import gc

    import torch
    from songformer import SongFormerPipeline

    pid = os.getpid()
    stop = threading.Event()
    rss_samples: list[int] = []

    def sampler() -> None:
        rss_samples.extend(_sample_rss_until(pid, stop, interval=1.0))

    t = threading.Thread(target=sampler, daemon=True)
    t.start()

    pipeline = SongFormerPipeline.from_pretrained()
    if dtype == "fp16":
        pipeline.muq.half()
        pipeline.musicfm.half()
        pipeline.msa.half()
    weights_rss = _current_rss_bytes(pid)

    t_start = time.monotonic()
    result = pipeline.analyze(str(audio_path))
    analyze_seconds = time.monotonic() - t_start
    peak_during = max(rss_samples) if rss_samples else weights_rss

    # Surface only the bits we need to compare. Segments are dumped as
    # (start, end, label) tuples so the orchestrator can diff boundaries.
    segments = [
        {"start": float(s.start), "end": float(s.end), "label": str(s.label)}
        for s in result.segments
    ]

    del pipeline, result
    gc.collect()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(3.0)  # let the allocator settle
    final_idle_rss = _current_rss_bytes(pid)

    stop.set()
    t.join(timeout=2.0)

    payload = {
        "dtype": dtype,
        "weights_rss": weights_rss,
        "peak_inference_rss": peak_during,
        "final_idle_rss": final_idle_rss,
        "analyze_seconds": analyze_seconds,
        "num_segments": len(segments),
        "segments": segments,
    }
    print(json.dumps(payload))


def _orchestrate(audio_path: Path) -> None:
    if not audio_path.exists():
        print(f"audio not found: {audio_path}", file=sys.stderr)
        sys.exit(2)

    print(f"[orchestrator] audio: {audio_path}", file=sys.stderr)
    runs: dict[str, dict] = {}
    for dtype in ("fp32", "fp16"):
        print(f"[orchestrator] spawning {dtype} worker…", file=sys.stderr)
        proc = subprocess.run(
            [sys.executable, __file__, dtype, str(audio_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            print(f"[orchestrator] {dtype} worker FAILED rc={proc.returncode}", file=sys.stderr)
            print(proc.stderr, file=sys.stderr)
            sys.exit(proc.returncode)
        # Worker prints one JSON line — pick the last one in case of stderr leakage.
        line = next(ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith("{"))
        runs[dtype] = json.loads(line)
        print(
            f"[orchestrator]   weights={runs[dtype]['weights_rss']/2**30:.2f}GB "
            f"peak={runs[dtype]['peak_inference_rss']/2**30:.2f}GB "
            f"idle_after={runs[dtype]['final_idle_rss']/2**30:.2f}GB "
            f"seconds={runs[dtype]['analyze_seconds']:.1f} "
            f"segments={runs[dtype]['num_segments']}",
            file=sys.stderr,
        )

    # Accuracy drift — pair boundaries by index, report max abs delta.
    drift = _boundary_drift(runs["fp32"]["segments"], runs["fp16"]["segments"])

    # Final table to stdout for easy copy-paste.
    print("\n=== fp16 vs fp32 comparison ===")
    print(f"{'metric':30} {'fp32':>14} {'fp16':>14} {'delta':>14}")
    for key, fmt in [
        ("weights_rss",       "{:.2f}GB"),
        ("peak_inference_rss","{:.2f}GB"),
        ("final_idle_rss",    "{:.2f}GB"),
        ("analyze_seconds",   "{:.1f}s"),
        ("num_segments",      "{:d}"),
    ]:
        a, b = runs["fp32"][key], runs["fp16"][key]
        if key.endswith("_rss"):
            a_disp = fmt.format(a / 2**30)
            b_disp = fmt.format(b / 2**30)
            delta_disp = f"{(b - a) / 2**30:+.2f}GB ({(b - a) / a * 100:+.0f}%)"
        elif key == "analyze_seconds":
            a_disp = fmt.format(a)
            b_disp = fmt.format(b)
            delta_disp = f"{b - a:+.1f}s"
        else:
            a_disp, b_disp = fmt.format(a), fmt.format(b)
            delta_disp = f"{b - a:+d}"
        print(f"{key:30} {a_disp:>14} {b_disp:>14} {delta_disp:>14}")
    print(f"{'boundary_drift_max':30} {'-':>14} {'-':>14} {drift['max']:>13.2f}s")
    print(f"{'boundary_drift_p95':30} {'-':>14} {'-':>14} {drift['p95']:>13.2f}s")
    print(f"{'boundaries_drifted_>0.5s':30} {'-':>14} {'-':>14} {drift['gt_half_sec']:>13d}")


def _boundary_drift(fp32_segs: list[dict], fp16_segs: list[dict]) -> dict:
    """Per-boundary absolute time delta between the two runs.

    Pairs by index — if fp16 produced a different number of segments, the
    drift naturally goes high and that itself is the signal.
    """
    n = min(len(fp32_segs), len(fp16_segs))
    deltas: list[float] = []
    for i in range(n):
        deltas.append(abs(fp32_segs[i]["start"] - fp16_segs[i]["start"]))
        deltas.append(abs(fp32_segs[i]["end"] - fp16_segs[i]["end"]))
    if abs(len(fp32_segs) - len(fp16_segs)) > 0:
        deltas.append(float("inf"))
    if not deltas:
        return {"max": 0.0, "p95": 0.0, "gt_half_sec": 0}
    deltas.sort()
    p95_idx = max(0, int(len(deltas) * 0.95) - 1)
    return {
        "max": deltas[-1],
        "p95": deltas[p95_idx],
        "gt_half_sec": sum(1 for d in deltas if d > 0.5),
    }


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] in {"fp16", "fp32"}:
        dtype = sys.argv[1]
        audio = Path(sys.argv[2]) if len(sys.argv) >= 3 else DEFAULT_AUDIO
        _worker(dtype, audio)
    else:
        audio = Path(sys.argv[1]) if len(sys.argv) >= 2 else DEFAULT_AUDIO
        _orchestrate(audio)
