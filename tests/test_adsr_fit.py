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
    fit = fit_adsr(env, envelope_sample_rate=200.0)
    assert isinstance(fit, ADSRFit)
    assert fit.attack_ms > 0
    assert fit.decay_ms > 0
    assert fit.release_ms > 0
    assert 0.0 <= fit.sustain_level <= 1.0


def test_fit_recovers_known_adsr_within_tolerance():
    env = _piecewise_envelope(0.02, 0.10, 0.6, 0.5, 0.15)
    fit = fit_adsr(env, envelope_sample_rate=200.0)
    # ±10 ms attack, ±25 ms for decay/release (envelope rate is 5ms)
    assert abs(fit.attack_ms - 20.0) < 10.0
    assert abs(fit.decay_ms - 100.0) < 25.0
    assert abs(fit.sustain_level - 0.6) < 0.05
    assert abs(fit.release_ms - 150.0) < 25.0


def test_fit_returns_sustain_window_indices():
    env = _piecewise_envelope(0.02, 0.10, 0.6, 0.5, 0.15)
    fit = fit_adsr(env, envelope_sample_rate=200.0)
    assert fit.sustain_start_idx < fit.sustain_end_idx
    assert fit.sustain_end_idx <= env.size


def test_fit_independent_of_envelope_scale():
    # Same shape at half scale → sustain_level (a ratio) should be unchanged.
    env = _piecewise_envelope(0.02, 0.10, 0.6, 0.5, 0.15) * 0.5
    fit = fit_adsr(env, envelope_sample_rate=200.0)
    assert abs(fit.sustain_level - 0.6) < 0.05


def test_fit_pluck_with_no_sustain():
    # Attack + immediate decay to zero, no sustain
    env = np.concatenate([
        np.linspace(0.0, 1.0, 5),
        np.linspace(1.0, 0.0, 40),
    ]).astype(np.float32)
    fit = fit_adsr(env, envelope_sample_rate=200.0)
    assert fit.sustain_level < 0.1
    assert fit.sustain_start_idx == fit.sustain_end_idx


def test_fit_handles_exponential_decay():
    # Real synth decays are exponential, not linear. Validate the fitter still recovers ADSR.
    env_sr = 200.0
    n_a, n_d, n_s, n_r = 4, 20, 100, 30
    sus = 0.6
    tau = 0.05  # 50ms time constant
    t_decay = np.arange(n_d) / env_sr
    decay_curve = sus + (1.0 - sus) * np.exp(-t_decay / tau)
    env = np.concatenate([
        np.linspace(0.0, 1.0, n_a, endpoint=False),
        decay_curve,
        np.full(n_s, sus),
        np.linspace(sus, 0.0, n_r, endpoint=True),
    ]).astype(np.float32)
    fit = fit_adsr(env, envelope_sample_rate=env_sr)
    assert abs(fit.attack_ms - 20.0) < 10.0
    assert abs(fit.decay_ms - 100.0) < 25.0
    assert abs(fit.sustain_level - 0.6) < 0.05
    assert abs(fit.release_ms - 150.0) < 25.0


def test_fit_sustain_runs_to_envelope_end():
    # Sustain runs to the end of the envelope (no release segment).
    # Exercises the for-else fallback. Exclusive end semantics → sustain_end_idx == envelope.size.
    env = np.concatenate([
        np.linspace(0.0, 1.0, 4, endpoint=False),
        np.linspace(1.0, 0.6, 20, endpoint=False),
        np.full(120, 0.6),
    ]).astype(np.float32)
    fit = fit_adsr(env, envelope_sample_rate=200.0)
    assert fit.sustain_end_idx == env.size
    # No release segment → release_ms should be 0
    assert fit.release_ms == 0.0
    # Sustain length should reflect the full plateau (allow attack+decay slack)
    sustain_frames = fit.sustain_end_idx - fit.sustain_start_idx
    assert sustain_frames >= 100  # at least 500ms of the 600ms plateau
