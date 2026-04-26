import numpy as np

from audio_analysis_mcp.analysis.adsr_fit import fit_adsr, ADSRFit


def _piecewise_envelope(
    attack_s: float,
    decay_s: float,
    sustain_level: float,
    sustain_s: float,
    release_s: float,
    sr: float = 200.0,  # 5ms hop → 200 Hz envelope rate
) -> np.ndarray:
    n_a = int(attack_s * sr)
    n_d = int(decay_s * sr)
    n_s = int(sustain_s * sr)
    n_r = int(release_s * sr)
    return np.concatenate([
        np.linspace(0.0, 1.0, n_a, endpoint=False),
        np.linspace(1.0, sustain_level, n_d, endpoint=False),
        np.full(n_s, sustain_level),
        np.linspace(sustain_level, 0.0, n_r, endpoint=True),
    ]).astype(np.float32)


def test_fit_returns_canonical_struct():
    env = _piecewise_envelope(0.02, 0.10, 0.6, 0.5, 0.15)
    fit = fit_adsr(env, envelope_sample_rate=200.0, peak_velocity=1.0)
    assert isinstance(fit, ADSRFit)
    assert fit.attack_ms > 0
    assert fit.decay_ms > 0
    assert fit.release_ms > 0
    assert 0.0 <= fit.sustain_level <= 1.0


def test_fit_recovers_known_adsr_within_tolerance():
    env = _piecewise_envelope(0.02, 0.10, 0.6, 0.5, 0.15)
    fit = fit_adsr(env, envelope_sample_rate=200.0, peak_velocity=1.0)
    # ±10 ms attack, ±25 ms for decay/release (envelope rate is 5ms)
    assert abs(fit.attack_ms - 20.0) < 10.0
    assert abs(fit.decay_ms - 100.0) < 25.0
    assert abs(fit.sustain_level - 0.6) < 0.05
    assert abs(fit.release_ms - 150.0) < 25.0


def test_fit_returns_sustain_window_indices():
    env = _piecewise_envelope(0.02, 0.10, 0.6, 0.5, 0.15)
    fit = fit_adsr(env, envelope_sample_rate=200.0, peak_velocity=1.0)
    assert fit.sustain_start_idx < fit.sustain_end_idx
    assert fit.sustain_end_idx <= env.size


def test_fit_normalizes_by_velocity():
    # Same shape, struck at half velocity → sustain_level should match the original
    env = _piecewise_envelope(0.02, 0.10, 0.6, 0.5, 0.15) * 0.5
    fit = fit_adsr(env, envelope_sample_rate=200.0, peak_velocity=0.5)
    assert abs(fit.sustain_level - 0.6) < 0.05


def test_fit_pluck_with_no_sustain():
    # Attack + immediate decay to zero, no sustain
    env = np.concatenate([
        np.linspace(0.0, 1.0, 5),
        np.linspace(1.0, 0.0, 40),
    ]).astype(np.float32)
    fit = fit_adsr(env, envelope_sample_rate=200.0, peak_velocity=1.0)
    assert fit.sustain_level < 0.1
    assert fit.sustain_start_idx == fit.sustain_end_idx