import uuid
from pathlib import Path
from typing import Any
import numpy as np
import sounddevice as sd
import soundfile as sf


def list_audio_devices() -> list[dict[str, Any]]:
    """List available audio input devices."""
    devices = sd.query_devices()
    if isinstance(devices, dict):
        devices = [devices]
    return [d for d in devices if d.get("max_input_channels", 0) > 0]


def capture_audio(
    duration: float,
    output_dir: Path,
    device: str | int | None = None,
    sample_rate: int = 44100,
) -> str:
    """Capture audio from an input device. Returns path to WAV file."""
    frames = int(duration * sample_rate)
    recording = sd.rec(
        frames=frames,
        samplerate=sample_rate,
        channels=1,
        dtype=np.float32,
        device=device,
    )
    sd.wait()

    output_path = output_dir / f"render_{uuid.uuid4().hex[:8]}.wav"
    sf.write(str(output_path), recording, sample_rate, subtype="PCM_16")
    return str(output_path)
