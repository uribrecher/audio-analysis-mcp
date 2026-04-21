import soundfile as sf
import pytest
from pathlib import Path
from audio_analysis_mcp.audio.normalize import normalize_audio


def test_normalize_44100_16bit(sine_440_wav: Path, tmp_path: Path):
    out = tmp_path / "out.wav"
    dur, ch = normalize_audio(str(sine_440_wav), str(out))
    info = sf.info(str(out))
    assert info.samplerate == 44100
    assert info.subtype == "PCM_16"
    assert ch == 1
    assert abs(dur - 1.0) < 0.01


def test_normalize_resamples_48k(high_sr_wav: Path, tmp_path: Path):
    out = tmp_path / "out.wav"
    dur, ch = normalize_audio(str(high_sr_wav), str(out))
    info = sf.info(str(out))
    assert info.samplerate == 44100


def test_normalize_downmixes_stereo(stereo_wav: Path, tmp_path: Path):
    out = tmp_path / "out.wav"
    dur, ch = normalize_audio(str(stereo_wav), str(out))
    info = sf.info(str(out))
    assert info.channels == 1
    assert ch == 1


def test_normalize_trims(sine_440_wav: Path, tmp_path: Path):
    out = tmp_path / "out.wav"
    dur, ch = normalize_audio(str(sine_440_wav), str(out), start_time=0.2, duration=0.5)
    assert abs(dur - 0.5) < 0.02


def test_normalize_nonexistent_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        normalize_audio("/nonexistent.wav", str(tmp_path / "out.wav"))
