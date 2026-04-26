import numpy as np

from audio_analysis_mcp.analysis.envelope import (
    extract_rms_envelope,
    EnvelopeResult,
)


SR = 22050


def _adsr_test_signal(
    attack_s: float = 0.02,
    decay_s: float = 0.10,
    sustain_level: float = 0.6,
    sustain_s: float = 0.5,
    release_s: float = 0.15,
    sr: int = SR,
    freq: float = 220.0,
) -> np.ndarray:
    """Sine carrier multiplied by a four-segment piecewise envelope. Peak amplitude = 1.0."""
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


def test_envelope_result_shape():
    audio = _adsr_test_signal()
    result = extract_rms_envelope(audio, sample_rate=SR)
    assert isinstance(result, EnvelopeResult)
    assert result.envelope.ndim == 1
    assert result.envelope_sample_rate > 0
    # librosa.feature.rms with center=False produces this exact frame count
    expected_len = (audio.size - result.frame_length) // result.hop_length + 1
    assert result.envelope.size == expected_len


def test_envelope_tracks_amplitude_shape():
    audio = _adsr_test_signal()
    result = extract_rms_envelope(audio, sample_rate=SR)
    env = result.envelope
    # Peak should be near the attack→decay boundary, not at the very end (release)
    peak_idx = int(np.argmax(env))
    peak_time = peak_idx / result.envelope_sample_rate
    assert 0.01 < peak_time < 0.05, f"peak at {peak_time}s outside attack region"
    # Final sample should be near zero (release ended)
    assert env[-1] < 0.05


def test_envelope_silence_has_low_rms():
    silence = np.zeros(SR, dtype=np.float32)
    result = extract_rms_envelope(silence, sample_rate=SR)
    assert float(result.envelope.max()) < 1e-6