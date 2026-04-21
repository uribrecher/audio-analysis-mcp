import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from audio_analysis_mcp.audio.capture import list_audio_devices, capture_audio


def test_list_devices():
    fake_devices = [
        {"name": "BlackHole 2ch", "max_input_channels": 2, "index": 0},
        {"name": "MacBook Pro Microphone", "max_input_channels": 1, "index": 1},
    ]
    with patch("sounddevice.query_devices", return_value=fake_devices):
        devices = list_audio_devices()
    assert len(devices) == 2
    assert devices[0]["name"] == "BlackHole 2ch"


def test_list_devices_filters_output_only():
    fake_devices = [
        {"name": "BlackHole 2ch", "max_input_channels": 2, "index": 0},
        {"name": "Speaker", "max_input_channels": 0, "index": 1},
    ]
    with patch("sounddevice.query_devices", return_value=fake_devices):
        devices = list_audio_devices()
    assert len(devices) == 1
    assert devices[0]["name"] == "BlackHole 2ch"


def test_capture_records_audio(tmp_path: Path):
    fake_audio = np.random.randn(44100, 1).astype(np.float32)
    with patch("sounddevice.rec", return_value=fake_audio), \
         patch("sounddevice.wait"):
        path = capture_audio(duration=1.0, output_dir=tmp_path, device="BlackHole 2ch")
    assert Path(path).exists()


def test_capture_passes_device(tmp_path: Path):
    fake_audio = np.random.randn(44100, 1).astype(np.float32)
    with patch("sounddevice.rec", return_value=fake_audio) as mock_rec, \
         patch("sounddevice.wait"):
        capture_audio(duration=1.0, output_dir=tmp_path, device=0)
    assert mock_rec.call_args[1]["device"] == 0
