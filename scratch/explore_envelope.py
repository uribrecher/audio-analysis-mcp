"""Sanity-check librosa.feature.rms before implementing the amplitude expert.

Synthesizes the same ADSR-shaped test signal the Task 3 unit tests will use
(20ms attack, 100ms decay to 0.6, 500ms sustain, 150ms release at 220Hz),
runs RMS over a 20ms window with 5ms hop, and prints the envelope shape +
peak location so we can verify our test tolerances are realistic.

Run:  uv run python scratch/explore_envelope.py
"""

import numpy as np
import librosa

SR = 22050
FRAME_LENGTH_MS = 20.0
HOP_LENGTH_MS = 5.0


def adsr_test_signal(
    attack_s: float = 0.02,
    decay_s: float = 0.10,
    sustain_level: float = 0.6,
    sustain_s: float = 0.5,
    release_s: float = 0.15,
    sr: int = SR,
    freq: float = 220.0,
) -> np.ndarray:
    n_attack = int(attack_s * sr)
    n_decay = int(decay_s * sr)
    n_sustain = int(sustain_s * sr)
    n_release = int(release_s * sr)
    env = np.concatenate([
        np.linspace(0.0, 1.0, n_attack, endpoint=False),
        np.linspace(1.0, sustain_level, n_decay, endpoint=False),
        np.full(n_sustain, sustain_level),
        np.linspace(sustain_level, 0.0, n_release, endpoint=True),
    ])
    t = np.arange(env.size) / sr
    return (env * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def main() -> None:
    audio = adsr_test_signal()
    frame_length = int(round(FRAME_LENGTH_MS * SR / 1000.0))
    hop_length = int(round(HOP_LENGTH_MS * SR / 1000.0))

    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length,
        center=False,
    )[0]

    env_sr = SR / hop_length
    peak_idx = int(np.argmax(rms))
    peak_time_s = peak_idx / env_sr

    print(f"audio:           {audio.size} samples ({audio.size / SR:.3f}s @ {SR}Hz)")
    print(f"frame_length:    {frame_length} samples ({FRAME_LENGTH_MS}ms)")
    print(f"hop_length:      {hop_length} samples ({HOP_LENGTH_MS}ms)")
    print(f"envelope size:   {rms.size} frames")
    print(f"envelope sr:     {env_sr:.1f} Hz")
    print(f"envelope peak:   value={rms.max():.4f} at frame {peak_idx} = {peak_time_s:.4f}s")
    print(f"first frame:     {rms[0]:.4f}")
    print(f"last frame:      {rms[-1]:.4f}")
    print(f"min in last 50ms (release tail): {rms[-int(0.05 * env_sr):].min():.4f}")

    # Compare to the unit-test assertions in the plan
    print()
    print("--- Unit test sanity ---")
    print(f"shape test:    abs({rms.size} - {audio.size // hop_length}) <= 1  →  {abs(rms.size - audio.size // hop_length) <= 1}")
    print(f"peak in 0.01–0.05s window:                                     →  {0.01 < peak_time_s < 0.05}")
    print(f"final frame < 0.05:                                            →  {rms[-1] < 0.05}")


if __name__ == "__main__":
    main()