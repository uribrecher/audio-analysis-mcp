# Audio Pipeline MCP Server — Implementation Plan

> Extracted from the original monolithic plan (`archive/7-audio-analysis-mcp.md`). Covers only the engineering layer — audio pipeline tools with no ML. The inverse synthesis and engine detection tools are handled by separate research projects under `research/`.

## Context

A Python MCP server (stdio transport) providing audio processing tools that the sound-recreation agent uses to fetch, separate, analyze, transcribe, and isolate audio. These tools produce clean audio segments that feed into the two research sub-projects (engine detection, inverse synthesis).

**Tech stack:** Python 3.12+, `mcp` (Python MCP SDK), `uv`, `pytest`, `mypy`

**Design spec:** `docs/superpowers/specs/2026-04-20-research-decomposition-design.md`

## Project Structure

```
audio-analysis-mcp/
  pyproject.toml
  CLAUDE.md
  src/
    audio_analysis_mcp/
      __init__.py
      server.py                        # MCP server, tool registration
      workspace.py                     # Temp/workspace directory management
      tools/
        __init__.py
        fetch_audio.py                 # YouTube download / local file import
        stem_separate.py               # Demucs stem separation
        spectrum_analyze.py            # Spectral feature extraction
        audio_compare.py               # A/B spectral diff
        audio_render.py                # Capture audio from system device
        note_transcribe.py             # Polyphonic transcription via Basic Pitch
        note_isolate.py                # Score-informed source separation via nussl
      analysis/
        __init__.py
        spectral.py                    # Librosa-based feature extraction
        comparison.py                  # A/B spectral diff logic
        transcription.py               # Basic Pitch integration + polyphony profiling
        note_isolation.py              # nussl time-frequency masking + quality assessment
      audio/
        __init__.py
        capture.py                     # sounddevice recording
        normalize.py                   # WAV normalization
  tests/
    test_spectral.py
    test_comparison.py
    test_transcription.py
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
  "librosa>=0.10.0",       # Spectral analysis
  "torch>=2.0",            # Already pulled by Demucs
  "torchaudio>=2.0",       # Audio transforms, mel spectrograms
  "numpy>=1.24",
  "scipy>=1.10",
  "soundfile>=0.12",       # WAV I/O
  "sounddevice>=0.4",      # Audio capture
  "yt-dlp>=2024.0",        # YouTube download
  "basic-pitch>=0.3.0",    # Polyphonic transcription (Spotify, MIT license)
  "nussl>=1.1.0",          # Score-informed source separation
  "pydantic>=2.0",         # Structured output schemas
]

[project.optional-dependencies]
dev = [
  "mypy>=1.10",
  "pytest>=8.0",
]
```

## Tools

### 1. `fetch_audio`

Download from YouTube or import a local file. Normalize to 44.1kHz 16-bit WAV.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| source | string | yes | YouTube URL or local file path |
| start_time | float | no | Trim start (seconds) |
| duration | float | no | Trim duration (seconds) |

**Returns:** Path to normalized WAV in `{workspace}/fetched/`.

### 2. `stem_separate`

Demucs stem separation into vocals, drums, bass, other (keyboards/synths).

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| audio_path | string | yes | Path to audio file |
| model | string | no | Demucs model (default: `htdemucs`) |

**Returns:** Paths to all stem WAV files. Cached by input hash.

**Long-running:** 1-5 min. Runs as async subprocess with 10 min timeout.

### 3. `audio_render`

Capture audio from a system audio device (BlackHole, USB audio).

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| duration | float | yes | Recording duration (seconds) |
| device | string | no | Audio input device name/index |
| list_devices | bool | no | Just list available devices |

**Returns:** Path to recorded WAV, or device list.

### 4. `spectrum_analyze`

Extract spectral features for diagnostics and iterative matching.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| audio_path | string | yes | Path to audio file |
| start_time | float | no | Analysis window start |
| duration | float | no | Analysis window (default: 5s) |

**Returns:** Harmonic profile, spectral envelope, ADSR, modulation detection (JSON).

Note: The original plan included "synth hints" in this tool's output. That functionality is now the domain of the engine detection research project. This tool returns raw spectral features only.

### 5. `audio_compare`

A/B spectral diff — used for iterative matching (render -> compare -> tweak).

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| target_path | string | yes | Reference audio |
| rendered_path | string | yes | Synthesized attempt |

**Returns:** Similarity scores, frequency band diffs, prioritized action items (JSON).

### 6. `note_transcribe`

Polyphonic transcription using Spotify Basic Pitch. Extracts MIDI note events with polyphony profiling and candidate note selection.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| audio_path | string | yes | Path to audio file (typically the "other" keyboard stem) |

**Returns:** MIDI file path, note events array (pitch, onset, offset, velocity, polyphony count), polyphony profile (monophonic/low/high windows), pre-selected candidate notes for isolation.

Candidate selection criteria: monophonic or low-polyphony windows, duration > 0.5s, temporal isolation, pitch range spread.

### 7. `note_isolate`

Score-informed source separation using nussl time-frequency masking. For monophonic windows, uses simple time-slice extraction.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| audio_path | string | yes | Path to audio file |
| transcription_path | string | yes | Path to MIDI transcription from `note_transcribe` |
| note_indices | int[] | yes | Indices of notes to isolate (from `note_events`) |
| assess_quality | bool | no | Run effects/distortion triage (default: true) |

**Returns:** Per-note: WAV path, pitch, duration, isolation method (time_slice or nussl_tf_mask), quality score, detected effects, usability flag. Plus a `recommended_for_analysis` list of the cleanest notes.

Quality triage flags: clean, reverb/delay (usable), chorus/modulation (usable), heavy distortion (unusable), masking artifacts (discard).

## Workspace

```
~/.audio-analysis-mcp/
  workspace/
    fetched/          # Downloaded/imported audio
    stems/            # Demucs output
    transcriptions/   # Basic Pitch MIDI + note event JSON
    isolated_notes/   # Per-note WAV files from nussl masking
    rendered/         # Captured synth recordings
```

## Agent Workflow (fallback — no trained models)

```
1. fetch_audio → stem_separate → other.wav
2. note_transcribe(other.wav) → transcription.mid + candidates
3. note_isolate(other.wav, transcription.mid, [...]) → clean isolated notes
4. spectrum_analyze(isolated_note.wav) → spectral features
5. Agent uses spectral features to set initial params via keyboards-mcp
6. audio_render → audio_compare → spectral diff + action items
7. Agent adjusts params based on action_items, repeat 6-7
```

Once the research projects deliver trained models, steps 4-5 will be replaced by `engine_detect` + `inverse_synth`.

## Implementation Sequence

### Phase 1: Scaffold + Core Pipeline

1. Scaffold: `pyproject.toml`, project structure, `server.py` with stdio transport
2. `workspace.py`: directory management
3. `fetch_audio`: yt-dlp download + local file import + WAV normalization
4. `stem_separate`: Demucs subprocess with caching by input hash
5. `audio_render`: sounddevice listing + capture
6. `spectrum_analyze`: librosa-based spectral features (harmonics, envelope, ADSR, modulation)
7. `audio_compare`: A/B spectral diff + similarity scores + action items

### Phase 2: Note-Level Extraction

8. `analysis/transcription.py`: Basic Pitch integration + polyphony profiling + candidate selection
9. `note_transcribe` tool: wire transcription module to MCP
10. `analysis/note_isolation.py`: nussl time-frequency masking + time-slice extraction + quality assessment
11. `note_isolate` tool: wire isolation module to MCP

## Test Coverage

### Unit tests

**`tests/test_spectral.py`:**
- Pure sine at 440Hz → fundamental ~440Hz, no significant harmonics
- Square wave → harmonics at 3x, 5x, 7x fundamental
- Clear ADSR envelope → detected values within tolerance
- Silence → graceful handling

**`tests/test_comparison.py`:**
- Identical inputs → similarity ~1.0, no action items
- 440Hz vs 880Hz → frequency band diff highlights shift
- Sine vs square at same fundamental → spectral envelope diff flags harmonics

**`tests/test_transcription.py`:**
- Single-note sine → 1 note event with correct pitch
- Two simultaneous sines (C4 + E4) → 2 events with correct pitches
- Polyphony profile → monophonic/polyphonic windows detected correctly
- Candidates → prefer monophonic, duration > 0.5s

**`tests/test_note_isolation.py`:**
- Monophonic window → `time_slice` method
- Polyphonic window → `nussl_tf_mask` method
- Clean note → quality_score > 0.8, usable=true
- Distorted note → usable=false, detected_effects includes heavy_distortion
- Reverb note → usable=true, detected_effects includes reverb

### Integration tests (`@pytest.mark.slow`)

- fetch + analyze: local WAV → spectrum_analyze → structured output
- fetch + separate: short file → stem_separate → 4 stems
- transcribe + isolate: polyphonic file → note_transcribe → note_isolate → quality-scored WAVs

### E2E tests

- MCP tool listing: all 7 tools present
- spectrum_analyze round-trip via MCP
- audio_compare round-trip via MCP

CI runs `pytest -m "not slow"`.

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