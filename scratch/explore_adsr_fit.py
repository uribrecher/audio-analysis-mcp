"""Validate the heuristic ADSR fitter against the Task 4 test envelope.

Implements the algorithm exactly as specified in the plan and runs it on
the four-segment piecewise test envelope (20ms attack, 100ms decay to 0.6,
500ms sustain, 150ms release; envelope sample rate = 200 Hz). Prints the
recovered ADSR values and whether each falls within the plan's stated
tolerance.

Also runs the pluck-fallback case (attack + immediate decay to zero, no
sustain) to verify the no-sustain branch.

Run: uv run python scratch/explore_adsr_fit.py
"""

import numpy as np

# Constants from the plan's adsr_fit.py
_ATTACK_THRESHOLD = 0.05
_RELEASE_THRESHOLD = 0.05
_SUSTAIN_DROP_THRESHOLD = 0.10
_SUSTAIN_STDDEV_THRESHOLD = 0.02
_SUSTAIN_WINDOW_MS = 50.0
_MIN_SUSTAIN_MS = 30.0
_PLUCK_FALLBACK_FRACTION = 0.5


def _frame_to_ms(n_frames, env_sr):
    return 1000.0 * n_frames / env_sr


def fit_adsr(envelope, env_sr, peak_velocity):
    peak = float(envelope.max())
    if peak <= 0:
        return dict(attack_ms=0.0, decay_ms=0.0, sustain_level=0.0, release_ms=0.0,
                    sustain_start_idx=0, sustain_end_idx=0)

    peak_idx = int(np.argmax(envelope))

    attack_thresh = _ATTACK_THRESHOLD * peak
    above = np.where(envelope[:peak_idx + 1] >= attack_thresh)[0]
    attack_start_idx = int(above[0]) if above.size > 0 else 0
    attack_frames = peak_idx - attack_start_idx
    attack_ms = _frame_to_ms(attack_frames, env_sr)

    window_frames = max(1, int(round(_SUSTAIN_WINDOW_MS * env_sr / 1000.0)))
    stddev_thresh = _SUSTAIN_STDDEV_THRESHOLD * peak
    drop_floor = _SUSTAIN_DROP_THRESHOLD * peak

    sustain_start_idx = peak_idx
    found_start = False
    for i in range(peak_idx, envelope.size - window_frames):
        window = envelope[i:i + window_frames]
        if window.std() < stddev_thresh and window.mean() >= drop_floor:
            sustain_start_idx = i
            found_start = True
            break

    sustain_end_idx = sustain_start_idx
    if found_start:
        for i in range(sustain_start_idx, envelope.size - window_frames):
            window = envelope[i:i + window_frames]
            if window.std() >= stddev_thresh or window.mean() < drop_floor:
                sustain_end_idx = i
                break
        else:
            sustain_end_idx = envelope.size - window_frames

    sustain_duration_ms = _frame_to_ms(sustain_end_idx - sustain_start_idx, env_sr)

    if not found_start or sustain_duration_ms < _MIN_SUSTAIN_MS:
        below = np.where(envelope[peak_idx:] < _PLUCK_FALLBACK_FRACTION * peak)[0]
        marker = peak_idx + int(below[0]) if below.size > 0 else envelope.size - 1
        sustain_start_idx = marker
        sustain_end_idx = marker
        sustain_level_raw = 0.0
    else:
        sustain_level_raw = float(envelope[sustain_start_idx:sustain_end_idx].mean())

    sustain_level = float(np.clip(sustain_level_raw / peak_velocity, 0.0, 1.0))
    decay_ms = _frame_to_ms(sustain_start_idx - peak_idx, env_sr)

    release_thresh = _RELEASE_THRESHOLD * peak
    tail = envelope[sustain_end_idx:]
    drops = np.where(tail < release_thresh)[0]
    release_frames = int(drops[0]) if drops.size > 0 else tail.size
    release_ms = _frame_to_ms(release_frames, env_sr)

    return dict(
        attack_ms=attack_ms, decay_ms=decay_ms,
        sustain_level=sustain_level, release_ms=release_ms,
        sustain_start_idx=sustain_start_idx, sustain_end_idx=sustain_end_idx,
    )


def piecewise_envelope(attack_s, decay_s, sustain_level, sustain_s, release_s, env_sr=200.0):
    n_a = int(attack_s * env_sr)
    n_d = int(decay_s * env_sr)
    n_s = int(sustain_s * env_sr)
    n_r = int(release_s * env_sr)
    return np.concatenate([
        np.linspace(0.0, 1.0, n_a, endpoint=False),
        np.linspace(1.0, sustain_level, n_d, endpoint=False),
        np.full(n_s, sustain_level),
        np.linspace(sustain_level, 0.0, n_r, endpoint=True),
    ]).astype(np.float32)


def report(label, recovered, expected, tolerances):
    print(f"\n--- {label} ---")
    print(f"recovered: {recovered}")
    if expected is None:
        return
    for k, tol in tolerances.items():
        rec = recovered[k]
        exp = expected[k]
        within = abs(rec - exp) < tol
        flag = "OK " if within else "FAIL"
        print(f"  {flag}  {k}: recovered={rec:.3f}  expected={exp:.3f}  |Δ|={abs(rec-exp):.3f} (tol {tol})")


def main():
    env_sr = 200.0

    # Case 1: full ADSR with sustain
    env1 = piecewise_envelope(0.02, 0.10, 0.6, 0.5, 0.15, env_sr)
    print(f"env1 size = {env1.size} frames @ {env_sr} Hz = {env1.size / env_sr * 1000:.0f}ms total")
    fit1 = fit_adsr(env1, env_sr, peak_velocity=1.0)
    report("Case 1 — full ADSR (peak_velocity=1.0)",
           fit1,
           dict(attack_ms=20.0, decay_ms=100.0, sustain_level=0.6, release_ms=150.0),
           dict(attack_ms=10.0, decay_ms=25.0, sustain_level=0.05, release_ms=25.0))

    # Case 2: same shape, half velocity → sustain_level should still recover ~0.6
    env2 = env1 * 0.5
    fit2 = fit_adsr(env2, env_sr, peak_velocity=0.5)
    report("Case 2 — half velocity",
           fit2,
           dict(attack_ms=20.0, decay_ms=100.0, sustain_level=0.6, release_ms=150.0),
           dict(attack_ms=10.0, decay_ms=25.0, sustain_level=0.05, release_ms=25.0))

    # Case 3: pluck (attack + immediate decay to zero, no sustain plateau)
    env3 = np.concatenate([
        np.linspace(0.0, 1.0, 5),
        np.linspace(1.0, 0.0, 40),
    ]).astype(np.float32)
    fit3 = fit_adsr(env3, env_sr, peak_velocity=1.0)
    print(f"\n--- Case 3 — pluck (no sustain) ---")
    print(f"recovered: {fit3}")
    print(f"  expected: sustain_level < 0.1, sustain_start_idx == sustain_end_idx")
    print(f"  flag: sustain_level={fit3['sustain_level']:.3f} < 0.1 ? {fit3['sustain_level'] < 0.1}")
    print(f"  flag: start==end? {fit3['sustain_start_idx'] == fit3['sustain_end_idx']}")


    # Case 4: exponential decay during the "decay" segment instead of linear
    # Simulates a real synth: attack, exponential decay to sustain, flat sustain, release
    n_a = int(0.02 * env_sr)
    n_d = int(0.10 * env_sr)
    n_s = int(0.50 * env_sr)
    n_r = int(0.15 * env_sr)
    sus = 0.6
    tau = 0.05  # 50ms time constant — typical synth decay
    t_decay = np.arange(n_d) / env_sr
    decay_curve = sus + (1.0 - sus) * np.exp(-t_decay / tau)
    env4 = np.concatenate([
        np.linspace(0.0, 1.0, n_a, endpoint=False),
        decay_curve,
        np.full(n_s, sus),
        np.linspace(sus, 0.0, n_r, endpoint=True),
    ]).astype(np.float32)
    fit4 = fit_adsr(env4, env_sr, peak_velocity=1.0)
    report("Case 4 — exponential decay segment",
           fit4,
           dict(attack_ms=20.0, decay_ms=100.0, sustain_level=0.6, release_ms=150.0),
           dict(attack_ms=10.0, decay_ms=25.0, sustain_level=0.05, release_ms=25.0))

    # Case 5: long sustain that runs to the very end (no release segment).
    # Exercises the for-else fallback where sustain_end_idx semantics matter.
    env5 = np.concatenate([
        np.linspace(0.0, 1.0, n_a, endpoint=False),
        np.linspace(1.0, sus, n_d, endpoint=False),
        np.full(int(0.6 * env_sr), sus),  # sustain runs to end, no release
    ]).astype(np.float32)
    fit5 = fit_adsr(env5, env_sr, peak_velocity=1.0)
    print(f"\n--- Case 5 — sustain runs to envelope end (no release) ---")
    print(f"env5 size = {env5.size} frames; recovered: {fit5}")
    sus_len = fit5['sustain_end_idx'] - fit5['sustain_start_idx']
    print(f"  sustain region length: {sus_len} frames = {sus_len * 1000.0 / env_sr:.0f}ms")
    print(f"  (true sustain plateau in env: ~{int(0.6 * env_sr)} frames = 600ms; under-count would flag the off-by-window_frames bug)")


if __name__ == "__main__":
    main()
