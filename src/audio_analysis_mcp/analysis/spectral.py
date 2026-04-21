import numpy as np
import librosa
import scipy.signal
from audio_analysis_mcp.schemas import (
    SpectralFeatures,
    ADSREstimate,
    ModulationDetection,
)


def compute_mel_spectrogram(
    y: np.ndarray,
    sr: int,
    n_mels: int = 128,
    hop_length: int = 512,
    n_fft: int = 2048,
) -> np.ndarray:
    """Compute log-power mel spectrogram. Returns (n_mels, time_frames) array."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
    )
    result: np.ndarray = librosa.power_to_db(mel, ref=np.max)
    return result


def extract_spectral_features(y: np.ndarray, sr: int) -> SpectralFeatures:
    """Extract fundamental frequency, harmonic ratios, and spectral shape."""
    rms_energy = float(np.sqrt(np.mean(y**2)))
    if rms_energy < 1e-6:
        return SpectralFeatures(
            fundamental_hz=None,
            harmonic_ratios=[],
            spectral_centroid_hz=0.0,
            spectral_rolloff_hz=0.0,
            spectral_bandwidth_hz=0.0,
        )

    # Fundamental via pyin
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=4000, sr=sr)
    f0_valid = f0[~np.isnan(f0)]
    fundamental_hz = float(np.median(f0_valid)) if len(f0_valid) > 0 else None

    # Harmonic ratios from magnitude spectrum
    S = np.abs(librosa.stft(y, n_fft=4096))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    spectral_mean = np.mean(S, axis=1)
    peaks, _ = scipy.signal.find_peaks(
        spectral_mean, height=spectral_mean.max() * 0.05,
    )
    harmonic_ratios: list[float] = []
    if len(peaks) > 0:
        peak_amps = spectral_mean[peaks]
        sorted_idx = np.argsort(peak_amps)[::-1]
        top_amps = peak_amps[sorted_idx[:8]]
        max_amp = top_amps[0]
        harmonic_ratios = [float(a / max_amp) for a in top_amps]

    # Spectral shape
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)

    return SpectralFeatures(
        fundamental_hz=fundamental_hz,
        harmonic_ratios=harmonic_ratios,
        spectral_centroid_hz=float(np.mean(centroid)),
        spectral_rolloff_hz=float(np.mean(rolloff)),
        spectral_bandwidth_hz=float(np.mean(bandwidth)),
    )


def estimate_adsr(y: np.ndarray, sr: int) -> ADSREstimate:
    """Estimate ADSR envelope from amplitude envelope."""
    frame_length = max(int(0.01 * sr), 1)
    hop = frame_length // 2 or 1
    envelope = np.array([
        np.sqrt(np.mean(y[i : i + frame_length] ** 2))
        for i in range(0, max(len(y) - frame_length, 1), hop)
    ])

    if len(envelope) == 0 or envelope.max() < 1e-6:
        return ADSREstimate(attack_ms=0, decay_ms=0, sustain_level=0, release_ms=0)

    envelope = envelope / envelope.max()
    frame_ms = (hop / sr) * 1000

    # If envelope is nearly flat (std < 5% of mean), there's no real attack
    if np.std(envelope) < 0.05:
        return ADSREstimate(
            attack_ms=0.0,
            decay_ms=0.0,
            sustain_level=float(np.mean(envelope)),
            release_ms=0.0,
        )

    peak_idx = int(np.argmax(envelope))
    attack_ms = peak_idx * frame_ms

    # Sustain: mean amplitude in middle 50%
    mid_start = len(envelope) // 4
    mid_end = 3 * len(envelope) // 4
    sustain_level = (
        float(np.mean(envelope[mid_start:mid_end]))
        if mid_end > mid_start
        else 0.0
    )

    # Decay: peak to sustain level
    decay_end = peak_idx
    for i in range(peak_idx, len(envelope)):
        if envelope[i] <= sustain_level * 1.05:
            decay_end = i
            break
    decay_ms = (decay_end - peak_idx) * frame_ms

    # Release: last point above threshold to end
    release_start = len(envelope) - 1
    for i in range(len(envelope) - 1, 0, -1):
        if envelope[i] > 0.05:
            release_start = i
            break
    release_ms = (len(envelope) - 1 - release_start) * frame_ms

    return ADSREstimate(
        attack_ms=float(attack_ms),
        decay_ms=float(decay_ms),
        sustain_level=float(sustain_level),
        release_ms=float(release_ms),
    )


def detect_modulation(y: np.ndarray, sr: int) -> ModulationDetection:
    """Detect vibrato, tremolo, and chorus. Simple heuristic approach."""
    rms_energy = float(np.sqrt(np.mean(y**2)))
    if rms_energy < 1e-6:
        return ModulationDetection(
            vibrato_hz=None, tremolo_hz=None, chorus_detected=False
        )

    # Vibrato: pitch track variation
    vibrato_hz = None
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=4000, sr=sr)
    f0_valid = f0[~np.isnan(f0)]
    if len(f0_valid) > 20:
        f0_mean = np.mean(f0_valid)
        f0_std = np.std(f0_valid)
        if f0_mean > 0 and f0_std / f0_mean > 0.01:
            f0_centered = f0_valid - f0_mean
            autocorr = np.correlate(f0_centered, f0_centered, mode="full")
            autocorr = autocorr[len(autocorr) // 2 :]
            if autocorr[0] > 0:
                autocorr = autocorr / autocorr[0]
                hop_time = 512 / sr
                for i in range(1, min(len(autocorr) - 1, 200)):
                    if (
                        autocorr[i] > autocorr[i - 1]
                        and autocorr[i] > autocorr[i + 1]
                        and autocorr[i] > 0.3
                    ):
                        rate = 1.0 / (i * hop_time)
                        if 3 <= rate <= 15:
                            vibrato_hz = rate
                        break

    # Tremolo: amplitude envelope variation
    tremolo_hz = None
    rms_frames = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    if len(rms_frames) > 20 and np.mean(rms_frames) > 0.01:
        rms_mean = np.mean(rms_frames)
        rms_std = np.std(rms_frames)
        if rms_std / rms_mean > 0.1:
            rms_c = rms_frames - rms_mean
            autocorr = np.correlate(rms_c, rms_c, mode="full")
            autocorr = autocorr[len(autocorr) // 2 :]
            if autocorr[0] > 0:
                autocorr = autocorr / autocorr[0]
                hop_time = 512 / sr
                for i in range(1, min(len(autocorr) - 1, 200)):
                    if (
                        autocorr[i] > autocorr[i - 1]
                        and autocorr[i] > autocorr[i + 1]
                        and autocorr[i] > 0.3
                    ):
                        rate = 1.0 / (i * hop_time)
                        if 1 <= rate <= 15:
                            tremolo_hz = rate
                        break

    # Chorus: stub — reliable detection needs more sophisticated analysis
    chorus_detected = False

    return ModulationDetection(
        vibrato_hz=float(vibrato_hz) if vibrato_hz else None,
        tremolo_hz=float(tremolo_hz) if tremolo_hz else None,
        chorus_detected=chorus_detected,
    )
