"""Sanity-check sustain isolation for Task 5.

Verifies the envelope-frame → audio-sample index conversion and the 100ms
minimum gate. End-to-end check: build an audio buffer, run a real envelope
extraction (Task 3), feed the result through a real ADSR fit (Task 4), then
check that isolating the reported sustain region gives audio that's clearly
the steady-state portion (low variance compared to the full clip).

Run: uv run python scratch/explore_sustain_isolation.py
"""

import numpy as np

from audio_analysis_mcp.analysis.envelope import extract_rms_envelope
from audio_analysis_mcp.analysis.adsr_fit import fit_adsr

SR = 22050
_MIN_SUSTAIN_MS = 100.0


def _isolate_sustain(audio, sample_rate, sustain_start_idx, sustain_end_idx, envelope_hop_length):
    if sustain_end_idx <= sustain_start_idx:
        return None
    duration_ms = 1000.0 * (sustain_end_idx - sustain_start_idx) * envelope_hop_length / sample_rate
    if duration_ms < _MIN_SUSTAIN_MS:
        return None
    start = sustain_start_idx * envelope_hop_length
    end = sustain_end_idx * envelope_hop_length
    start = max(0, start)
    end = min(audio.size, end)
    if end <= start:
        return None
    return audio[start:end].astype(np.float32)


def main():
    # Synth signal with known structure: 20ms attack, 100ms decay to 0.6, 500ms sustain, 150ms release.
    n_a = int(0.02 * SR)
    n_d = int(0.10 * SR)
    n_s = int(0.50 * SR)
    n_r = int(0.15 * SR)
    env = np.concatenate([
        np.linspace(0.0, 1.0, n_a, endpoint=False),
        np.linspace(1.0, 0.6, n_d, endpoint=False),
        np.full(n_s, 0.6),
        np.linspace(0.6, 0.0, n_r, endpoint=True),
    ])
    t = np.arange(env.size) / SR
    audio = (env * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)

    env_result = extract_rms_envelope(audio, sample_rate=SR)
    fit = fit_adsr(env_result.envelope, env_result.envelope_sample_rate, peak_velocity=1.0)

    print(f"audio:           {audio.size} samples ({audio.size / SR:.3f}s)")
    print(f"envelope:        {env_result.envelope.size} frames; hop_length={env_result.hop_length} samples ({env_result.hop_length * 1000.0 / SR:.1f}ms)")
    print(f"sustain frames:  [{fit.sustain_start_idx}, {fit.sustain_end_idx})")
    print(f"sustain ms:      [{fit.sustain_start_idx * env_result.hop_length * 1000.0 / SR:.1f}, "
          f"{fit.sustain_end_idx * env_result.hop_length * 1000.0 / SR:.1f}) ms")

    sustain = _isolate_sustain(
        audio, SR, fit.sustain_start_idx, fit.sustain_end_idx, env_result.hop_length,
    )
    assert sustain is not None, "Expected sustain isolation to succeed"

    expected_samples = (fit.sustain_end_idx - fit.sustain_start_idx) * env_result.hop_length
    print(f"sustain audio:   {sustain.size} samples (expected {expected_samples})")
    print(f"sustain RMS:     {np.sqrt((sustain ** 2).mean()):.4f}")
    print(f"full audio RMS:  {np.sqrt((audio ** 2).mean()):.4f}")

    # The sustain slice should sit inside the true plateau (samples ~120ms..620ms).
    start_sample = fit.sustain_start_idx * env_result.hop_length
    end_sample = fit.sustain_end_idx * env_result.hop_length
    plateau_start = n_a + n_d
    plateau_end = n_a + n_d + n_s
    print(f"plateau (samples): [{plateau_start}, {plateau_end})")
    print(f"slice   (samples): [{start_sample}, {end_sample})")
    print(f"  start inside plateau? {plateau_start <= start_sample <= plateau_end}")
    print(f"  end inside plateau?   {plateau_start <= end_sample <= plateau_end}")

    # Boundary tests for the gate
    print()
    print("--- gate boundary tests ---")
    hop = env_result.hop_length
    # 50ms request: 10 frames * 5ms = 50ms → below minimum, should reject
    too_short = _isolate_sustain(audio, SR, 100, 110, hop)
    print(f"50ms request:     {'rejected' if too_short is None else 'accepted (UNEXPECTED)'}")
    # 100ms request: 20 frames * 5ms = 100ms → at the threshold; rule is "< 100ms rejects" so equal accepts
    at_threshold = _isolate_sustain(audio, SR, 100, 120, hop)
    print(f"100ms request:    {'accepted' if at_threshold is not None else 'rejected (UNEXPECTED)'}")
    # Slice past audio end
    past_end = _isolate_sustain(audio, SR, 100, 100_000, hop)
    print(f"past-end request: returns size={past_end.size if past_end is not None else None} (audio size {audio.size})")


if __name__ == "__main__":
    main()