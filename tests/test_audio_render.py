import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from audio_analysis_mcp.audio.capture import list_audio_devices, capture_audio


def _mock_sd(**overrides):
    """Create a mock sounddevice module."""
    sd = MagicMock()
    for k, v in overrides.items():
        setattr(sd, k, v)
    return sd


def test_list_devices():
    fake_devices = [
        {"name": "BlackHole 2ch", "max_input_channels": 2, "index": 0},
        {"name": "MacBook Pro Microphone", "max_input_channels": 1, "index": 1},
    ]
    sd = _mock_sd(query_devices=MagicMock(return_value=fake_devices))
    with patch("audio_analysis_mcp.audio.capture._get_sd", return_value=sd):
        devices = list_audio_devices()
    assert len(devices) == 2
    assert devices[0]["name"] == "BlackHole 2ch"


def test_list_devices_filters_output_only():
    fake_devices = [
        {"name": "BlackHole 2ch", "max_input_channels": 2, "index": 0},
        {"name": "Speaker", "max_input_channels": 0, "index": 1},
    ]
    sd = _mock_sd(query_devices=MagicMock(return_value=fake_devices))
    with patch("audio_analysis_mcp.audio.capture._get_sd", return_value=sd):
        devices = list_audio_devices()
    assert len(devices) == 1
    assert devices[0]["name"] == "BlackHole 2ch"


def test_capture_records_audio(tmp_path: Path):
    fake_audio = np.random.randn(44100, 1).astype(np.float32)
    sd = _mock_sd(rec=MagicMock(return_value=fake_audio))
    with patch("audio_analysis_mcp.audio.capture._get_sd", return_value=sd):
        path = capture_audio(duration=1.0, output_dir=tmp_path, device="BlackHole 2ch")
    assert Path(path).exists()


def test_capture_passes_device(tmp_path: Path):
    fake_audio = np.random.randn(44100, 1).astype(np.float32)
    sd = _mock_sd(rec=MagicMock(return_value=fake_audio))
    with patch("audio_analysis_mcp.audio.capture._get_sd", return_value=sd):
        capture_audio(duration=1.0, output_dir=tmp_path, device=0)
    assert sd.rec.call_args[1]["device"] == 0
