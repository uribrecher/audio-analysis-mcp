import numpy as np
import librosa
from pathlib import Path
from audio_analysis_mcp.analysis.spectral import (
    compute_mel_spectrogram,
    extract_spectral_features,
    estimate_adsr,
    detect_modulation,
)


def test_mel_spectrogram_shape(sine_440_wav: Path):
    y, sr = librosa.load(str(sine_440_wav), sr=44100)
    mel = compute_mel_spectrogram(y, sr, n_mels=128, hop_length=512, n_fft=2048)
    assert mel.shape[0] == 128
    assert mel.shape[1] > 0


def test_sine_fundamental(sine_440_wav: Path):
    y, sr = librosa.load(str(sine_440_wav), sr=44100)
    features = extract_spectral_features(y, sr)
    assert features.fundamental_hz is not None
    assert abs(features.fundamental_hz - 440) < 10


def test_sine_weak_harmonics(sine_440_wav: Path):
    y, sr = librosa.load(str(sine_440_wav), sr=44100)
    features = extract_spectral_features(y, sr)
    if len(features.harmonic_ratios) > 1:
        assert features.harmonic_ratios[1] < 0.15


def test_square_has_odd_harmonics(square_440_wav: Path):
    y, sr = librosa.load(str(square_440_wav), sr=44100)
    features = extract_spectral_features(y, sr)
    assert features.fundamental_hz is not None
    assert abs(features.fundamental_hz - 440) < 10
    # Square wave has strong 3rd harmonic
    assert len(features.harmonic_ratios) >= 3
    assert features.harmonic_ratios[1] > 0.2


def test_silence_fundamental_is_none(silence_wav: Path):
    y, sr = librosa.load(str(silence_wav), sr=44100)
    features = extract_spectral_features(y, sr)
    assert features.fundamental_hz is None


def test_adsr_attack_near_zero_for_sine(sine_440_wav: Path):
    y, sr = librosa.load(str(sine_440_wav), sr=44100)
    adsr = estimate_adsr(y, sr)
    assert adsr.attack_ms < 50


def test_modulation_none_for_pure_sine(sine_440_wav: Path):
    y, sr = librosa.load(str(sine_440_wav), sr=44100)
    mod = detect_modulation(y, sr)
    assert mod.vibrato_hz is None
    assert mod.tremolo_hz is None
    assert mod.chorus_detected is False
