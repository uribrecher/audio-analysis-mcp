# Audio Pipeline MCP Server — Implementation Plan

> Extracted from the original monolithic plan (`archive/7-audio-analysis-mcp.md`). Covers only the engineering layer — audio pipeline tools with no ML. The inverse synthesis and engine detection tools are handled by separate research projects under `research/`.

## Context

A Python MCP server (stdio transport) providing audio processing tools that the sound-recreation agent uses to import, separate, analyze, transcribe, and isolate audio. These tools produce clean audio segments that feed into the two research sub-projects (engine detection, inverse synthesis).

**Tech stack:** Python 3.12+, `mcp` (Python MCP SDK), `uv` (package management), `pytest`, `mypy`

**Design spec:** `docs/superpowers/specs/2026-04-20-research-decomposition-design.md`

## Project Structure

```
audio-analysis-mcp/
  pyproject.toml
  .github/
    workflows/
      ci.yml                             # pytest (non-slow) + mypy
  CLAUDE.md
  src/
    audio_analysis_mcp/
      __init__.py
      server.py                        # MCP server, tool registration
      workspace.py                     # Temp/workspace directory management
      schemas.py                       # Pydantic output schemas for all tools
      tools/
        __init__.py
        import_audio.py                # Local file import + normalization
        stem_separate.py               # Demucs stem separation
        spectrum_analyze.py            # Mel spectrogram + spectral feature extraction
        audio_compare.py               # Mel spectrogram + CLAP embedding comparison
        audio_render.py                # Capture audio from system device
        note_transcribe.py             # Polyphonic transcription via Basic Pitch
        note_triage.py                 # Candidate selection from transcription
        note_isolate.py                # Time-frequency box isolation
      analysis/
        __init__.py
        spectral.py                    # Librosa-based feature extraction + mel spectrogram
        comparison.py                  # Mel spectrogram diff + CLAP embedding similarity
        transcription.py               # Basic Pitch integration
        note_triage.py                 # Polyphony profiling + candidate selection
        note_isolation.py              # Time-frequency masking
      audio/
        __init__.py
        capture.py                     # sounddevice recording
        normalize.py                   # WAV normalization
  tests/
    test_spectral.py
    test_comparison.py
    test_transcription.py
    test_note_triage.py
    test_note_isolation.py
    test_pipeline_integration.py
    test_mcp_tools.py
```

## Dependencies

```toml
[project]
name = "audio-analysis-mcp"
requires-python = ">=3.12"
dependencies = [
  "mcp>=1.0.0",
  "demucs>=4.0.0",         # Stem separation
  "librosa>=0.10.0",       # Spectral analysis + mel spectrograms
  "torch>=2.0",            # Already pulled by Demucs
  "torchaudio>=2.0",       # Audio transforms
  "numpy>=1.24",
  "scipy>=1.10",
  "soundfile>=0.12",       # WAV I/O
  "sounddevice>=0.4",      # Audio capture
  "basic-pitch>=0.3.0",    # Polyphonic transcription (Spotify, Apache 2.0)
  "laion-clap>=1.0",       # Audio embedding for perceptual similarity
  "pydantic>=2.0",         # Structured output schemas
]

[project.optional-dependencies]
dev = [
  "mypy>=1.10",
  "pytest>=8.0",
]
```

**Removed:** `yt-dlp` (legal concerns — see Decision Log), `nussl` (replaced by simpler time-frequency box isolation).

## Output Schemas

All tools return structured JSON matching these Pydantic models.

### ImportAudioResult

```python
class ImportAudioResult(BaseModel):
    audio_path: str            # Path to normalized WAV
    sample_rate: int           # Always 44100
    duration_seconds: float
    channels: int
```

### StemSeparateResult

```python
class StemFile(BaseModel):
    stem: str                  # "vocals" | "drums" | "bass" | "other"
    path: str

class StemSeparateResult(BaseModel):
    stems: list[StemFile]
    model: str                 # Demucs model used
    cached: bool               # Whether result was from cache
```

### AudioRenderResult

```python
class AudioRenderResult(BaseModel):
    audio_path: str
    duration_seconds: float
    device: str
    sample_rate: int
```

### SpectrumAnalyzeResult

```python
class MelSpectrogramData(BaseModel):
    array_path: str            # Path to saved .npy file (n_mels x time_frames)
    n_mels: int
    hop_length: int
    n_fft: int
    sample_rate: int

class SpectralFeatures(BaseModel):
    fundamental_hz: float | None
    harmonic_ratios: list[float]       # Amplitude ratios of first N harmonics
    spectral_centroid_hz: float
    spectral_rolloff_hz: float
    spectral_bandwidth_hz: float

class ADSREstimate(BaseModel):
    attack_ms: float
    decay_ms: float
    sustain_level: float               # 0.0 - 1.0
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
```

### AudioCompareResult

```python
class BandDiff(BaseModel):
    band: str                  # e.g. "low (0-300Hz)", "mid (300-2kHz)", "high (2k-8kHz)"
    target_energy_db: float
    rendered_energy_db: float
    diff_db: float

class AudioCompareResult(BaseModel):
    mel_spectrogram_distance: float    # L2 distance on normalized mel spectrograms
    clap_cosine_similarity: float      # Cosine similarity of CLAP embeddings (0-1)
    band_diffs: list[BandDiff]
```

### NoteTranscribeResult

```python
class NoteEvent(BaseModel):
    index: int
    pitch_midi: int                    # MIDI note number (0-127)
    pitch_name: str                    # e.g. "C4", "F#5"
    start_time: float                  # Seconds
    end_time: float                    # Seconds
    duration: float                    # Seconds
    amplitude: float                   # 0.0 - 1.0
    pitch_bend: list[int] | None       # MIDI pitch bend values per frame

class NoteTranscribeResult(BaseModel):
    midi_path: str                     # Path to .mid file
    note_events: list[NoteEvent]
    total_notes: int
    duration_seconds: float
```

### NoteTriageResult

```python
class TriageWindow(BaseModel):
    start_time: float
    end_time: float
    polyphony_count: int               # Number of simultaneous notes

class CandidateNote(BaseModel):
    note_index: int                    # Index into NoteTranscribeResult.note_events
    pitch_midi: int
    pitch_name: str
    start_time: float
    end_time: float
    start_freq: float                  # Hz — lower bound of isolation box
    end_freq: float                    # Hz — upper bound of isolation box
    selection_reason: str              # e.g. "monophonic, long duration, good isolation"

class NoteTriageResult(BaseModel):
    polyphony_profile: list[TriageWindow]
    candidates: list[CandidateNote]
```

### NoteIsolateResult

```python
class IsolatedNote(BaseModel):
    audio_path: str                    # Path to isolated WAV
    start_time: float
    end_time: float
    start_freq: float
    end_freq: float
    duration: float

class NoteIsolateResult(BaseModel):
    isolated_notes: list[IsolatedNote]
```

## Tools

### 1. `import_audio`

Import a local audio file. Normalize to 44.1kHz 16-bit WAV.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| file_path | string | yes | Path to local audio file (WAV, FLAC, MP3, etc.) |
| start_time | float | no | Trim start (seconds) |
| duration | float | no | Trim duration (seconds) |

**Returns:** `ImportAudioResult` — path to normalized WAV in `{workspace}/imported/`.

**Note:** YouTube/streaming download was deliberately excluded due to legal concerns (YouTube ToS violation, active DMCA Section 1201 litigation against download tools). Users must provide audio files they have rights to.

### 2. `stem_separate`

Demucs stem separation into vocals, drums, bass, other (keyboards/synths).

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| audio_path | string | yes | Path to audio file |
| model | string | no | Demucs model (default: `htdemucs`) |

**Returns:** `StemSeparateResult` — paths to all stem WAV files. Cached by input hash.

**Long-running:** 1-5 min. Runs as async subprocess with 10 min timeout.

### 3. `audio_render`

Capture audio from a system audio device (BlackHole, USB audio).

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| duration | float | yes | Recording duration (seconds) |
| device | string | no | Audio input device name/index |
| list_devices | bool | no | Just list available devices |

**Returns:** `AudioRenderResult` — path to recorded WAV, or device list.

**macOS permissions required:**
- The host process (Terminal, packaged .app) must have **Microphone** permission granted via System Settings > Privacy & Security > Microphone. This applies to virtual devices (BlackHole) identically to physical microphones — macOS TCC makes no distinction.
- For a packaged .app: requires `com.apple.security.device.audio-input` entitlement and `NSMicrophoneUsageDescription` in Info.plist.
- BlackHole itself requires one-time System Extension approval in System Settings.
- The tool should detect permission denial and return a clear error message guiding the user to enable the required permission.

### 4. `spectrum_analyze`

Extract mel spectrogram and spectral features for diagnostics and iterative matching.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| audio_path | string | yes | Path to audio file |
| start_time | float | no | Analysis window start |
| duration | float | no | Analysis window (default: 5s) |
| n_mels | int | no | Mel frequency bins (default: 128) |
| hop_length | int | no | Samples between frames (default: 512) |

**Returns:** `SpectrumAnalyzeResult` — mel spectrogram saved as .npy array, plus spectral features, ADSR estimate, and modulation detection.

The mel spectrogram is the primary output for ML pipelines. It is saved as a numpy `.npy` file that downstream models can load directly. Generation uses librosa:

```python
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
np.save(output_path, mel_spec_db)
```

### 5. `audio_compare`

Compare target audio vs. synthesized attempt using mel spectrogram distance and CLAP embedding similarity.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| target_path | string | yes | Reference audio |
| rendered_path | string | yes | Synthesized attempt |

**Returns:** `AudioCompareResult` — mel spectrogram L2 distance, CLAP cosine similarity (0-1), per-band energy diffs.

**Approach:**
1. **Mel spectrogram L2 distance** — compute mel spectrograms of both signals, normalize, take L2 distance. Fast, interpretable, good for low-level fidelity.
2. **CLAP embedding cosine similarity** — pass both through LAION-CLAP encoder, compute cosine similarity. Captures perceptual/semantic similarity (is this the "same kind of sound"?).
3. **Per-band energy comparison** — split into low/mid/high frequency bands, report energy differences in dB.

### 6. `note_transcribe`

Polyphonic transcription using Spotify Basic Pitch. Extracts MIDI note events.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| audio_path | string | yes | Path to audio file (typically the "other" keyboard stem) |

**Returns:** `NoteTranscribeResult` — MIDI file path + structured note events array.

Basic Pitch (Apache 2.0, Spotify) outputs per-note: `(start_time, end_time, pitch_midi, amplitude, pitch_bend)`. This tool wraps that into the `NoteEvent` schema above.

### 7. `note_triage`

Analyze a transcription and select the best candidate notes for isolation, based on polyphony profile.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| audio_path | string | yes | Path to audio file |
| transcription | NoteTranscribeResult | yes | Output from `note_transcribe` |

**Returns:** `NoteTriageResult` — polyphony profile (per-window note counts) and ranked candidate notes with recommended time/frequency isolation boxes.

**Candidate selection criteria:** monophonic or low-polyphony windows, duration > 0.5s, temporal isolation from neighboring notes, pitch range spread for diverse sampling.

### 8. `note_isolate`

Isolate a sound from audio within a time-frequency box.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| audio_path | string | yes | Path to audio file |
| start_time | float | yes | Start of isolation window (seconds) |
| end_time | float | yes | End of isolation window (seconds) |
| start_freq | float | yes | Lower frequency bound (Hz) |
| end_freq | float | yes | Upper frequency bound (Hz) |

**Returns:** `NoteIsolateResult` — path to isolated WAV file.

The algorithm applies a time-frequency mask (STFT → zero out bins outside the box → inverse STFT) to extract the target sound. The time/frequency window selection is handled upstream by `note_triage`.

## Workspace

```
~/.audio-analysis-mcp/
  workspace/
    imported/         # Imported/normalized audio
    stems/            # Demucs output
    spectrograms/     # Mel spectrogram .npy files
    transcriptions/   # Basic Pitch MIDI + note event JSON
    isolated_notes/   # Per-note WAV files from TF masking
    rendered/         # Captured synth recordings
```

## Agent Workflow (fallback — no trained models)

```
1. import_audio → stem_separate → other.wav
2. note_transcribe(other.wav) → transcription + note events
3. note_triage(transcription) → select best candidates with time/freq windows
4. note_isolate(other.wav, candidate windows) → clean isolated notes
5. STOP — user listens to isolated notes and evaluates quality manually
```

Once the research projects deliver trained models, the workflow extends with `engine_detect` + `inverse_synth` to go from isolated notes to keyboard parameter settings automatically.

## Implementation Sequence

### Phase 1: Scaffold + Core Pipeline

1. Scaffold: `pyproject.toml` (managed by `uv`), project structure, `server.py` with stdio transport, `schemas.py` with all Pydantic output models
2. `workspace.py`: directory management
3. `import_audio`: local file import + WAV normalization (44.1kHz 16-bit)
4. `stem_separate`: Demucs subprocess with caching by input hash
5. `audio_render`: sounddevice listing + capture + macOS permission detection
6. `spectrum_analyze`: mel spectrogram generation (librosa) + spectral features (harmonics, envelope, ADSR, modulation)
7. `audio_compare`: mel spectrogram L2 distance + CLAP embedding cosine similarity + per-band diffs

### Phase 2: Note-Level Extraction

8. `analysis/transcription.py`: Basic Pitch integration → NoteTranscribeResult
9. `note_transcribe` tool: wire transcription module to MCP
10. `analysis/note_triage.py`: polyphony profiling + candidate selection → NoteTriageResult
11. `note_triage` tool: wire triage module to MCP
12. `analysis/note_isolation.py`: STFT time-frequency box masking
13. `note_isolate` tool: wire isolation module to MCP

## CI / GitHub Workflow

`.github/workflows/ci.yml`:

```yaml
name: CI
on: [push, pull_request]

jobs:
  test-and-lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync --dev
      - run: uv run mypy src/
      - run: uv run pytest -m "not slow"
```

## Test Coverage

### Unit tests

**`tests/test_spectral.py`:**
- Pure sine at 440Hz → fundamental ~440Hz, no significant harmonics
- Square wave → harmonics at 3x, 5x, 7x fundamental
- Clear ADSR envelope → detected values within tolerance
- Silence → graceful handling
- Mel spectrogram shape matches expected (n_mels, time_frames)

**`tests/test_comparison.py`:**
- Identical inputs → mel distance ~0, CLAP similarity ~1.0
- 440Hz vs 880Hz → significant mel distance, low CLAP similarity
- Sine vs square at same fundamental → moderate mel distance, band diffs highlight harmonics

**`tests/test_transcription.py`:**
- Single-note sine → 1 note event with correct pitch
- Two simultaneous sines (C4 + E4) → 2 events with correct pitches
- Note event schema: all fields populated (pitch_midi, start_time, end_time, amplitude)

**`tests/test_note_triage.py`:**
- Polyphony profile → monophonic/polyphonic windows detected correctly
- Candidates → prefer monophonic, duration > 0.5s
- Candidate notes include time/freq isolation boxes

**`tests/test_note_isolation.py`:**
- Single sine in TF box → isolated cleanly
- Two sines, isolate one by freq → other attenuated
- Time-only window → correct segment extracted

### Integration tests (`@pytest.mark.slow`)

- import + analyze: local WAV → spectrum_analyze → structured output with mel spectrogram
- import + separate: short file → stem_separate → 4 stems
- transcribe + triage + isolate: polyphonic file → note_transcribe → note_triage → note_isolate → isolated WAVs

### E2E tests

- MCP tool listing: all 8 tools present
- spectrum_analyze round-trip via MCP
- audio_compare round-trip via MCP

CI runs `pytest -m "not slow"`.

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-20 | Dropped YouTube download (yt-dlp) | YouTube ToS violation, active DMCA Section 1201 litigation. Users provide local files. |
| 2026-04-20 | Replaced nussl with STFT TF-box masking | Simpler approach: note_isolate receives explicit time/freq bounds. Selection logic moved to separate note_triage tool. |
| 2026-04-20 | Added CLAP embeddings to audio_compare | Mel spectrogram L2 alone lacks perceptual grounding. CLAP provides semantic similarity score. |
| 2026-04-20 | Fallback workflow stops at isolated notes | Spectral-features-to-params loop is speculative without trained models. Manual evaluation first. |
| 2026-04-20 | Basic Pitch license corrected to Apache 2.0 | Was incorrectly listed as MIT. |

## MCP Configuration

```json
{
  "mcpServers": {
    "audio-analysis-mcp": {
      "command": "/path/to/audio-analysis-mcp/.venv/bin/python",
      "args": ["-m", "audio_analysis_mcp"]
    }
  }
}
```