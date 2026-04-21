from pathlib import Path
import librosa
import soundfile as sf

TARGET_SR = 44100


def normalize_audio(
    input_path: str,
    output_path: str,
    start_time: float | None = None,
    duration: float | None = None,
) -> tuple[float, int]:
    """Normalize audio to 44.1kHz 16-bit mono WAV.

    Returns (duration_seconds, channels).
    Raises FileNotFoundError if input doesn't exist.
    """
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Audio file not found: {input_path}")
    if start_time is not None and start_time < 0:
        raise ValueError(f"start_time must be >= 0, got {start_time}")
    if duration is not None and duration <= 0:
        raise ValueError(f"duration must be > 0, got {duration}")

    y, sr = librosa.load(
        input_path,
        sr=TARGET_SR,
        mono=True,
        offset=0.0 if start_time is None else start_time,
        duration=duration,
    )

    sf.write(output_path, y, TARGET_SR, subtype="PCM_16")
    return len(y) / TARGET_SR, 1
