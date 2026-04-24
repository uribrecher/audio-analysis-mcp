from typing import Any

import numpy as np
import numpy.typing as npt
import librosa

NDArray = npt.NDArray[Any]


def isolate_note(
    audio_path: str,
    start_time: float,
    end_time: float,
    start_freq: float,
    end_freq: float,
    sr: int = 44100,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> tuple[NDArray, int]:
    """Apply time-frequency box mask via STFT to isolate a note.

    Returns (isolated_audio_array, sample_rate).
    """
    if start_time >= end_time:
        raise ValueError(f"start_time ({start_time}) must be < end_time ({end_time})")
    if start_freq >= end_freq:
        raise ValueError(f"start_freq ({start_freq}) must be < end_freq ({end_freq})")

    # Load and slice to time window
    y, sr_loaded = librosa.load(audio_path, sr=sr, mono=True, offset=start_time, duration=end_time - start_time)
    sr = int(sr_loaded)

    # STFT
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

    # Build frequency mask
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    freq_mask = ((freqs >= start_freq) & (freqs <= end_freq)).astype(np.float32)

    # Apply mask (broadcast over time axis)
    S_masked = S * freq_mask[:, np.newaxis]

    # Inverse STFT
    y_isolated: NDArray = librosa.istft(S_masked, hop_length=hop_length)
    return y_isolated, sr
