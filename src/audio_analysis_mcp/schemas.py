from pydantic import BaseModel


class ImportAudioResult(BaseModel):
    audio_path: str
    sample_rate: int
    duration_seconds: float
    channels: int


class StemFile(BaseModel):
    stem: str
    path: str


class StemSeparateResult(BaseModel):
    stems: list[StemFile]
    model: str
    cached: bool


class AudioRenderResult(BaseModel):
    audio_path: str
    duration_seconds: float
    device: str
    sample_rate: int


class MelSpectrogramData(BaseModel):
    array_path: str
    n_mels: int
    hop_length: int
    n_fft: int
    sample_rate: int


class SpectralFeatures(BaseModel):
    fundamental_hz: float | None
    harmonic_ratios: list[float]
    spectral_centroid_hz: float
    spectral_rolloff_hz: float
    spectral_bandwidth_hz: float


class ADSREstimate(BaseModel):
    attack_ms: float
    decay_ms: float
    sustain_level: float
    release_ms: float


class ModulationDetection(BaseModel):
    vibrato_hz: float | None
    tremolo_hz: float | None
    chorus_detected: bool


class SpectrumAnalyzeResult(BaseModel):
    mel_spectrogram: MelSpectrogramData
    spectral_features: SpectralFeatures
    adsr: ADSREstimate
    modulation: ModulationDetection


class BandDiff(BaseModel):
    band: str
    target_energy_db: float
    rendered_energy_db: float
    diff_db: float


class AudioCompareResult(BaseModel):
    mel_spectrogram_distance: float
    clap_cosine_similarity: float | None
    band_diffs: list[BandDiff]
