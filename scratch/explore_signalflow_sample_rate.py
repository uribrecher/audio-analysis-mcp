"""Explore whether SignalFlow's AudioGraph respects a configured sample_rate.

Question: does setting AudioGraphConfig.sample_rate before constructing
AudioGraph actually change the rendered output rate? Or does it silently
ignore the config and render at SignalFlow's default (typically 44_100)?

Approach: render a 1-second 1kHz sine at multiple sample rates, then count
zero crossings in the output buffer to verify the actual render rate.

If the sample_rate is honored, we should see:
- 1s @ 44100 Hz → buffer length 44100, ~2000 zero crossings (1kHz × 2 × 1s)
- 1s @ 22050 Hz → buffer length 22050, ~2000 zero crossings (still 1kHz tone,
  but the *tone frequency* in Hz is independent of rate as long as the
  oscillator interprets its frequency arg as Hz at the actual graph rate).

Failure mode A: setting sample_rate has no effect → we render at 44100 even
though we asked for 22050. We'd see correct zero-crossing count BUT the
buffer length would be wrong relative to wall-clock duration. (Or if num_frames
is exact, we'd see 22050 samples of audio that *physically* represent 0.5s of
sound, with 1000 zero crossings instead of 2000.)
"""

from __future__ import annotations

import numpy as np
import signalflow as sf_lib


def count_zero_crossings(audio: np.ndarray) -> int:
    """Count sign changes in the signal."""
    return int(np.sum(np.diff(np.sign(audio)) != 0))


def render_sine_at(sample_rate: int, freq_hz: float, duration_s: float) -> np.ndarray:
    """Render a sine wave at the given sample rate and return raw float32 array."""
    cfg = sf_lib.AudioGraphConfig()
    cfg.backend_name = "null"
    # The thing we're testing: does setting sample_rate here actually take effect?
    cfg.sample_rate = sample_rate
    graph = sf_lib.AudioGraph(config=cfg, start=False)
    try:
        # Read back what the graph thinks its sample_rate is.
        actual_sr = getattr(graph, "sample_rate", None)
        print(f"  requested sample_rate={sample_rate}, graph.sample_rate={actual_sr}")
        osc = sf_lib.SineOscillator(frequency=freq_hz)
        # Need an envelope so we get sustained output (mirrors renderer.py).
        env = sf_lib.ADSREnvelope(attack=0.001, decay=0.0, sustain=1.0, release=0.0, gate=1)
        node = osc * env
        node.play()
        n_samples = int(round(sample_rate * duration_s))
        buf = graph.render_to_new_buffer(num_frames=n_samples)
        arr = np.asarray(buf.data, dtype=np.float32)
        return arr[0] if arr.ndim == 2 else arr
    finally:
        graph.destroy()


def main() -> None:
    print("Test 1: render 1s of 1kHz sine @ 44100 Hz")
    a44 = render_sine_at(sample_rate=44_100, freq_hz=1000.0, duration_s=1.0)
    zc44 = count_zero_crossings(a44)
    print(f"  buffer length = {a44.size}, zero crossings = {zc44}")
    print(f"  expected: length=44100, zc≈2000 if 1s of 1kHz tone\n")

    print("Test 2: render 1s of 1kHz sine @ 22050 Hz")
    a22 = render_sine_at(sample_rate=22_050, freq_hz=1000.0, duration_s=1.0)
    zc22 = count_zero_crossings(a22)
    print(f"  buffer length = {a22.size}, zero crossings = {zc22}")
    print(f"  if sample_rate honored: length=22050, zc≈2000 (1s of 1kHz)")
    print(f"  if sample_rate ignored: length=22050, zc≈1000 (0.5s of 1kHz @ 44100)\n")

    print("Test 3: render 1s of 1kHz sine @ 88200 Hz")
    a88 = render_sine_at(sample_rate=88_200, freq_hz=1000.0, duration_s=1.0)
    zc88 = count_zero_crossings(a88)
    print(f"  buffer length = {a88.size}, zero crossings = {zc88}")
    print(f"  if sample_rate honored: length=88200, zc≈2000 (1s of 1kHz)")
    print(f"  if sample_rate ignored: length=88200, zc≈4000 (2s of 1kHz @ 44100)\n")

    # Also explore what AudioGraphConfig actually accepts.
    print("AudioGraphConfig attributes:")
    cfg = sf_lib.AudioGraphConfig()
    for attr in sorted(dir(cfg)):
        if not attr.startswith("_"):
            try:
                v = getattr(cfg, attr)
                if not callable(v):
                    print(f"  {attr} = {v!r}")
            except Exception as e:
                print(f"  {attr}: <error: {e}>")


if __name__ == "__main__":
    main()
