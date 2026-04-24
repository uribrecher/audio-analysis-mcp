import numpy as np
import soundfile as sf
import pytest
from pathlib import Path


def _write_wav(path: Path, y: np.ndarray, sr: int = 44100) -> Path:
    sf.write(str(path), y, sr, subtype="PCM_16")
    return path


@pytest.fixture
def sine_440_wav(tmp_path: Path) -> Path:
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return _write_wav(tmp_path / "sine_440.wav", y, sr)


@pytest.fixture
def sine_880_wav(tmp_path: Path) -> Path:
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
    return _write_wav(tmp_path / "sine_880.wav", y, sr)


@pytest.fixture
def square_440_wav(tmp_path: Path) -> Path:
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y = (0.5 * np.sign(np.sin(2 * np.pi * 440 * t))).astype(np.float32)
    return _write_wav(tmp_path / "square_440.wav", y, sr)


@pytest.fixture
def silence_wav(tmp_path: Path) -> Path:
    sr = 44100
    y = np.zeros(sr, dtype=np.float32)
    return _write_wav(tmp_path / "silence.wav", y, sr)


@pytest.fixture
def high_sr_wav(tmp_path: Path) -> Path:
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return _write_wav(tmp_path / "sine_48k.wav", y, sr)


@pytest.fixture
def stereo_wav(tmp_path: Path) -> Path:
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.5 * np.sin(2 * np.pi * 880 * t)
    y = np.column_stack([left, right]).astype(np.float32)
    return _write_wav(tmp_path / "stereo.wav", y, sr)


@pytest.fixture
def two_note_wav(tmp_path: Path) -> Path:
    """C4 (261.63 Hz) + E4 (329.63 Hz) played simultaneously for 1 second."""
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    c4 = 0.4 * np.sin(2 * np.pi * 261.63 * t)
    e4 = 0.4 * np.sin(2 * np.pi * 329.63 * t)
    y = (c4 + e4).astype(np.float32)
    return _write_wav(tmp_path / "two_notes.wav", y, sr)


@pytest.fixture
def sequential_notes_wav(tmp_path: Path) -> Path:
    """Three 1-second notes played sequentially: C4, E4, G4 (3 seconds total)."""
    sr = 44100
    duration = 1.0
    samples_per_note = int(sr * duration)
    c4 = 0.5 * np.sin(2 * np.pi * 261.63 * np.linspace(0, duration, samples_per_note, endpoint=False))
    e4 = 0.5 * np.sin(2 * np.pi * 329.63 * np.linspace(0, duration, samples_per_note, endpoint=False))
    g4 = 0.5 * np.sin(2 * np.pi * 392.00 * np.linspace(0, duration, samples_per_note, endpoint=False))
    y = np.concatenate([c4, e4, g4]).astype(np.float32)
    return _write_wav(tmp_path / "sequential_notes.wav", y, sr)


@pytest.fixture
def two_freq_wav(tmp_path: Path) -> Path:
    """440 Hz + 1000 Hz simultaneously for 1 second. For isolation tests."""
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y = (0.4 * np.sin(2 * np.pi * 440 * t) + 0.4 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    return _write_wav(tmp_path / "two_freq.wav", y, sr)
