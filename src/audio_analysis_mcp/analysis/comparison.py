from typing import Any

import numpy as np
import numpy.typing as npt
import librosa

NDArray = npt.NDArray[Any]
from audio_analysis_mcp.schemas import AudioCompareResult, BandDiff
from audio_analysis_mcp.analysis.spectral import compute_mel_spectrogram

BANDS = {
    "low (0-300Hz)": (0, 300),
    "mid (300-2kHz)": (300, 2000),
    "high (2k-8kHz)": (2000, 8000),
    "presence (8k-16kHz)": (8000, 16000),
}


def _band_energy_db(
    S: NDArray, freqs: NDArray, lo: float, hi: float
) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    if not mask.any():
        return -100.0
    energy = float(np.mean(S[mask, :] ** 2))
    return float(10 * np.log10(energy + 1e-10))


def compare_audio(target_path: str, rendered_path: str) -> AudioCompareResult:
    """Compare two audio files using mel spectrogram distance and per-band energy."""
    y_target, sr_val = librosa.load(target_path, sr=44100, mono=True)
    y_rendered, _ = librosa.load(rendered_path, sr=44100, mono=True)
    sr = int(sr_val)

    # Pad shorter signal to match lengths
    max_len = max(len(y_target), len(y_rendered))
    y_target = np.pad(y_target, (0, max_len - len(y_target)))
    y_rendered = np.pad(y_rendered, (0, max_len - len(y_rendered)))

    # Mel spectrogram L2 distance (normalized to 0-1 range)
    mel_target = compute_mel_spectrogram(y_target, sr)
    mel_rendered = compute_mel_spectrogram(y_rendered, sr)
    t_range = mel_target.max() - mel_target.min()
    r_range = mel_rendered.max() - mel_rendered.min()
    mel_target_norm = (mel_target - mel_target.min()) / (t_range + 1e-10)
    mel_rendered_norm = (mel_rendered - mel_rendered.min()) / (r_range + 1e-10)
    mel_distance = float(np.sqrt(np.mean((mel_target_norm - mel_rendered_norm) ** 2)))

    # Per-band energy comparison
    S_target = np.abs(librosa.stft(y_target, n_fft=2048))
    S_rendered = np.abs(librosa.stft(y_rendered, n_fft=2048))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    band_diffs: list[BandDiff] = []
    for name, (lo, hi) in BANDS.items():
        target_db = _band_energy_db(S_target, freqs, lo, hi)
        rendered_db = _band_energy_db(S_rendered, freqs, lo, hi)
        band_diffs.append(
            BandDiff(
                band=name,
                target_energy_db=target_db,
                rendered_energy_db=rendered_db,
                diff_db=rendered_db - target_db,
            )
        )

    # CLAP: placeholder for Phase 2
    clap_similarity = None

    return AudioCompareResult(
        mel_spectrogram_distance=mel_distance,
        clap_cosine_similarity=clap_similarity,
        band_diffs=band_diffs,
    )
