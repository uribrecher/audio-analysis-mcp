from pydantic import BaseModel


class ImportAudioResult(BaseModel):
    audio_path: str
    job_name: str
    sample_rate: int
    duration_seconds: float
    channels: int


class StemFile(BaseModel):
    stem: str
    path: str


class StemSeparateResult(BaseModel):
    stems: list[StemFile]
    model: str
    preset: str
    cached: bool


class AudioDevice(BaseModel):
    name: str
    index: int
    max_input_channels: int


class ListAudioDevicesResult(BaseModel):
    devices: list[AudioDevice]


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


class NoteEvent(BaseModel):
    start_time: float
    end_time: float
    pitch_midi: int
    amplitude: float
    pitch_bends: list[int] | None


class NoteTranscribeResult(BaseModel):
    midi_path: str
    notes_path: str
    note_count: int


class PolyphonyWindow(BaseModel):
    start_time: float
    end_time: float
    note_count: int


class CandidateNote(BaseModel):
    note: NoteEvent
    score: float
    start_time: float
    end_time: float
    start_freq: float
    end_freq: float


class NoteTriageFileData(BaseModel):
    """Full triage data written to the JSON file."""
    polyphony_profile: list[PolyphonyWindow]
    candidates: list[CandidateNote]


class NoteTriageResult(BaseModel):
    """Lightweight result returned by the MCP tool."""
    triage_path: str
    candidate_count: int
    top_candidates: list[CandidateNote]


class NoteIsolateResult(BaseModel):
    audio_path: str
    duration_seconds: float
