# Phase 2: Note-Level Extraction — Design Spec

## Goal

Add three MCP tools (`note_transcribe`, `note_triage`, `note_isolate`) that take a separated keyboard stem and produce clean, isolated single-note WAV files suitable for downstream spectral analysis or future ML-based engine classification.

## Pipeline Position

```
stem_separate → other.wav
                   ↓
            note_transcribe  (Basic Pitch → MIDI + note events)
                   ↓
            note_triage      (select best candidate notes)
                   ↓
            note_isolate     (STFT TF-box masking → clean WAV per note)
```

## Decision: Basic Pitch with ONNX Backend

Use `basic-pitch[onnx]` (Apache 2.0, Spotify) instead of the default TensorFlow backend. Rationale: PyTorch is already in the stack for Demucs — adding TensorFlow would bloat the install. ONNX Runtime is lightweight and sufficient.

**New dependency:** `basic-pitch[onnx]` in `pyproject.toml`.

## Decision: Frequency Bounds from MIDI Pitch

`note_triage` outputs time bounds + frequency bounds derived mathematically from the MIDI pitch number (fundamental from `librosa.midi_to_hz` + overtones up to the Nth harmonic). No separate harmonic analysis needed — Basic Pitch already provides the pitch.

Default: include up to the 8th harmonic or 10kHz, whichever is lower.

---

## Tool 1: `note_transcribe`

### Interface

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| audio_path | string | yes | Path to audio file (typically the "other" keyboard stem) |

### Returns: `NoteTranscribeResult`

| Field | Type | Description |
|-------|------|-------------|
| midi_path | str | Path to saved MIDI file in `workspace/transcriptions/` |
| notes | list[NoteEvent] | Structured note events |

### `NoteEvent` Schema

| Field | Type | Description |
|-------|------|-------------|
| start_time | float | Note onset (seconds) |
| end_time | float | Note offset (seconds) |
| pitch_midi | int | MIDI note number (0-127) |
| amplitude | float | Velocity/amplitude (0.0-1.0) |
| pitch_bends | list[int] \| None | Raw MIDI pitch bend values per frame (centered at 8192), None if absent |

### Logic Module: `analysis/transcription.py`

```python
def transcribe_audio(audio_path: str) -> tuple[str, list[NoteEvent]]:
    """Run Basic Pitch on audio, return (midi_bytes_or_path, note_events)."""
```

- Load audio via `basic_pitch.inference.predict`
- Convert Basic Pitch output (`model_output` → `note_events`) to `NoteEvent` schema
- Save MIDI file to `workspace/transcriptions/`
- Return MIDI path and structured note list

### MCP Wrapper: `tools/note_transcribe.py`

Thin wrapper: load audio path, call `transcribe_audio`, serialize `NoteTranscribeResult` to JSON.

---

## Tool 2: `note_triage`

### Interface

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| audio_path | string | yes | Path to audio file |
| notes_json | string | yes | JSON string of NoteEvent array (from `note_transcribe` output) |
| min_duration | float | no | Minimum note duration in seconds (default: 0.5) |
| max_candidates | int | no | Maximum candidates to return (default: 10) |

### Returns: `NoteTriageResult`

| Field | Type | Description |
|-------|------|-------------|
| polyphony_profile | list[PolyphonyWindow] | Per-window note counts |
| candidates | list[CandidateNote] | Ranked candidate notes with isolation boxes |

### `PolyphonyWindow` Schema

| Field | Type | Description |
|-------|------|-------------|
| start_time | float | Window start (seconds) |
| end_time | float | Window end (seconds) |
| note_count | int | Number of simultaneous notes |

### `CandidateNote` Schema

| Field | Type | Description |
|-------|------|-------------|
| note | NoteEvent | The candidate note event |
| score | float | Ranking score (higher = better candidate) |
| start_time | float | Isolation window start (seconds, may include padding) |
| end_time | float | Isolation window end (seconds, may include padding) |
| start_freq | float | Lower frequency bound (Hz) |
| end_freq | float | Upper frequency bound (Hz) |

### Logic Module: `analysis/note_triage.py`

```python
def triage_notes(
    notes: list[NoteEvent],
    sr: int = 44100,
    min_duration: float = 0.5,
    max_candidates: int = 10,
) -> NoteTriageResult:
    """Profile polyphony and rank candidate notes for isolation."""
```

**Polyphony profiling:**
- Divide the audio timeline into 0.5s windows
- Count overlapping notes per window

**Candidate scoring criteria (higher = better):**
1. **Low polyphony** — notes in windows with fewer simultaneous notes score higher
2. **Duration** — longer notes (above `min_duration`) score higher, capped at diminishing returns
3. **Temporal isolation** — notes with more gap before/after neighboring notes score higher
4. **Pitch diversity** — when selecting final candidates, prefer spread across pitch range

**Frequency bounds derivation:**
- Fundamental: `librosa.midi_to_hz(pitch_midi)`
- Lower bound: `fundamental * 0.9` (allow slight detuning margin)
- Upper bound: `min(fundamental * 8, 10000.0)` (8th harmonic or 10kHz cap)

**Time bounds:**
- `start_time`: `note.start_time - 0.05` (50ms pre-padding for attack transient)
- `end_time`: `note.end_time + 0.05` (50ms post-padding for release)

### MCP Wrapper: `tools/note_triage.py`

Parse `notes_json` string into `list[NoteEvent]` via Pydantic, call `triage_notes`, serialize result.

---

## Tool 3: `note_isolate`

### Interface

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| audio_path | string | yes | Path to audio file |
| start_time | float | yes | Start of isolation window (seconds) |
| end_time | float | yes | End of isolation window (seconds) |
| start_freq | float | yes | Lower frequency bound (Hz) |
| end_freq | float | yes | Upper frequency bound (Hz) |

### Returns: `NoteIsolateResult`

| Field | Type | Description |
|-------|------|-------------|
| audio_path | str | Path to isolated WAV in `workspace/isolated_notes/` |
| duration_seconds | float | Duration of isolated clip |

### Logic Module: `analysis/note_isolation.py`

```python
def isolate_note(
    audio_path: str,
    start_time: float,
    end_time: float,
    start_freq: float,
    end_freq: float,
    sr: int = 44100,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> tuple[np.ndarray, int]:
    """Apply time-frequency box mask via STFT to isolate a note."""
```

**Algorithm:**
1. Load audio, slice to `[start_time, end_time]`
2. Compute STFT (`librosa.stft`)
3. Build frequency mask: compute frequency per bin (`librosa.fft_frequencies`), zero out bins outside `[start_freq, end_freq]`
4. Apply mask to STFT magnitude
5. Inverse STFT (`librosa.istft`) to reconstruct the isolated waveform
6. Save to `workspace/isolated_notes/` as WAV

### MCP Wrapper: `tools/note_isolate.py`

Validate inputs (times > 0, freqs > 0, start < end), call `isolate_note`, save WAV, return `NoteIsolateResult`.

---

## Schemas (additions to `schemas.py`)

```python
class NoteEvent(BaseModel):
    start_time: float
    end_time: float
    pitch_midi: int
    amplitude: float
    pitch_bends: list[int] | None

class NoteTranscribeResult(BaseModel):
    midi_path: str
    notes: list[NoteEvent]

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

class NoteTriageResult(BaseModel):
    polyphony_profile: list[PolyphonyWindow]
    candidates: list[CandidateNote]

class NoteIsolateResult(BaseModel):
    audio_path: str
    duration_seconds: float
```

---

## New Files

| File | Purpose |
|------|---------|
| `src/audio_analysis_mcp/analysis/transcription.py` | Basic Pitch integration logic |
| `src/audio_analysis_mcp/analysis/note_triage.py` | Polyphony profiling + candidate ranking |
| `src/audio_analysis_mcp/analysis/note_isolation.py` | STFT time-frequency box masking |
| `src/audio_analysis_mcp/tools/note_transcribe.py` | MCP wrapper for transcription |
| `src/audio_analysis_mcp/tools/note_triage.py` | MCP wrapper for triage |
| `src/audio_analysis_mcp/tools/note_isolate.py` | MCP wrapper for isolation |
| `tests/test_transcription.py` | Transcription unit tests |
| `tests/test_note_triage.py` | Triage unit tests |
| `tests/test_note_isolation.py` | Note isolation unit tests |

## Modified Files

| File | Change |
|------|--------|
| `pyproject.toml` | Add `basic-pitch[onnx]` dependency |
| `src/audio_analysis_mcp/schemas.py` | Add 6 new Pydantic models |
| `src/audio_analysis_mcp/__main__.py` | Add 3 tool imports |
| `tests/conftest.py` | Add fixtures for two-note audio, chord audio |
| `tests/test_mcp_tools.py` | Add E2E tests for new tools, update tool count assertion |

---

## Test Plan

### Unit: `tests/test_transcription.py`

- Single-note sine (A4/440Hz, 1s) → 1 NoteEvent, pitch_midi == 69, times within tolerance
- Two simultaneous sines (C4 + E4) → 2 NoteEvents with correct pitches (60, 64)
- All NoteEvent fields populated (start_time, end_time, pitch_midi, amplitude, pitch_bends)
- MIDI file written to expected path
- **Mock `basic_pitch.inference.predict`** in unit tests to avoid model download

### Unit: `tests/test_note_triage.py`

- Polyphony profile: 3 sequential monophonic notes → all windows have count 1
- Polyphony profile: overlapping notes → correct window counts
- Candidates prefer monophonic windows over polyphonic
- Candidates respect `min_duration` filter (short notes excluded)
- Candidates include correct frequency bounds (fundamental * 0.9 to min(fundamental * 8, 10kHz))
- `max_candidates` limits output length
- Pitch diversity: candidates spread across pitch range when possible

### Unit: `tests/test_note_isolation.py`

- Single sine in TF box → isolated cleanly (high energy in output)
- Two sines at different frequencies, isolate one → other attenuated significantly
- Time-only window (full freq range) → correct time segment extracted
- Output WAV duration matches requested time window
- Edge case: zero-length window → error

### E2E: `tests/test_mcp_tools.py` (additions)

- MCP tool listing includes all 9 tools (6 Phase 1 + 3 Phase 2)
- `note_transcribe` round-trip via MCP (with mocked Basic Pitch)
- `note_isolate` round-trip via MCP

### Integration (`@pytest.mark.slow`)

- Full pipeline: transcribe → triage → isolate on a synthetic multi-note audio file