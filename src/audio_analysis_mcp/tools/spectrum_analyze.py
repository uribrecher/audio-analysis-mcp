import uuid
import numpy as np
import librosa
from audio_analysis_mcp.server import mcp, get_workspace
from audio_analysis_mcp.analysis.spectral import (
    compute_mel_spectrogram,
    extract_spectral_features,
    estimate_adsr,
    detect_modulation,
)
from audio_analysis_mcp.schemas import MelSpectrogramData, SpectrumAnalyzeResult


@mcp.tool()
def spectrum_analyze(
    audio_path: str,
    start_time: float | None = None,
    duration: float | None = None,
    n_mels: int = 128,
    hop_length: int = 512,
) -> str:
    """Extract mel spectrogram and spectral features from audio."""
    ws = get_workspace()
    y, sr_val = librosa.load(
        audio_path,
        sr=44100,
        mono=True,
        offset=start_time or 0.0,
        duration=duration or 5.0,
    )
    sr = int(sr_val)

    # Mel spectrogram -> save as .npy
    mel = compute_mel_spectrogram(y, sr, n_mels=n_mels, hop_length=hop_length)
    npy_path = ws.spectrograms / f"mel_{uuid.uuid4().hex[:8]}.npy"
    np.save(str(npy_path), mel)

    features = extract_spectral_features(y, sr)
    adsr = estimate_adsr(y, sr)
    modulation = detect_modulation(y, sr)

    return SpectrumAnalyzeResult(
        mel_spectrogram=MelSpectrogramData(
            array_path=str(npy_path),
            n_mels=n_mels,
            hop_length=hop_length,
            n_fft=2048,
            sample_rate=44100,
        ),
        spectral_features=features,
        adsr=adsr,
        modulation=modulation,
    ).model_dump_json(indent=2)
