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
