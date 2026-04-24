# Phase 2: Note-Level Extraction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three MCP tools (`note_transcribe`, `note_triage`, `note_isolate`) that extract clean, isolated single-note WAV files from keyboard stems.

**Architecture:** Each tool follows the existing pattern: pure logic in `analysis/` module, thin MCP wrapper in `tools/`, Pydantic schemas in `schemas.py`. Basic Pitch (ONNX backend) handles polyphonic transcription. STFT time-frequency masking handles note isolation.

**Tech Stack:** Python 3.11, FastMCP, basic-pitch (CoreML backend on macOS), librosa, numpy, scipy, pydantic, pytest

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `pyproject.toml` | Modify | Add `basic-pitch>=0.4.0` dependency |
| `src/audio_analysis_mcp/schemas.py` | Modify | Add 6 Pydantic models |
| `src/audio_analysis_mcp/analysis/transcription.py` | Create | Basic Pitch wrapper → NoteEvent list |
| `src/audio_analysis_mcp/analysis/note_triage.py` | Create | Polyphony profiling + candidate ranking |
| `src/audio_analysis_mcp/analysis/note_isolation.py` | Create | STFT TF-box masking |
| `src/audio_analysis_mcp/tools/note_transcribe.py` | Create | MCP tool for transcription |
| `src/audio_analysis_mcp/tools/note_triage.py` | Create | MCP tool for triage |
| `src/audio_analysis_mcp/tools/note_isolate.py` | Create | MCP tool for isolation |
| `src/audio_analysis_mcp/__main__.py` | Modify | Register 3 new tools |
| `tests/conftest.py` | Modify | Add multi-note fixtures |
| `tests/test_transcription.py` | Create | Transcription unit tests (mocked Basic Pitch) |
| `tests/test_note_triage.py` | Create | Triage unit tests (pure logic, no mocks) |
| `tests/test_note_isolation.py` | Create | Isolation unit tests (synthetic audio) |
| `tests/test_mcp_tools.py` | Modify | E2E tests for new tools |

---

### Task 1: Add dependency and schemas

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/audio_analysis_mcp/schemas.py`

- [ ] **Step 1: Add `basic-pitch` dependency to `pyproject.toml`**

In `pyproject.toml`, add to the `dependencies` list:

```toml
  "basic-pitch>=0.4.0",
```

- [ ] **Step 2: Install dependencies**

Run: `cd /Users/uribrecher/test/sounds-and-recreation/audio-analysis-mcp && uv sync --dev`
Expected: resolves and installs basic-pitch with onnx extras, no conflicts.

- [ ] **Step 3: Add Phase 2 schemas to `schemas.py`**

Append to `src/audio_analysis_mcp/schemas.py`:

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

- [ ] **Step 4: Type-check**

Run: `uv run mypy src/audio_analysis_mcp/schemas.py`
Expected: Success, no errors.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock src/audio_analysis_mcp/schemas.py
git commit -m "feat: add Phase 2 schemas and basic-pitch dependency"
```

---

### Task 2: Add test fixtures for multi-note audio

**Files:**
- Modify: `tests/conftest.py`

- [ ] **Step 1: Add two-note and sequential-note fixtures to `conftest.py`**

Append to `tests/conftest.py`:

```python
@pytest.fixture
def two_note_wav(tmp_path: Path) -> Path:
    """C4 (261.63 Hz) + E4 (329.63 Hz) played simultaneously for 1 second."""
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    c4 = 0.4 * np.sin(2 * np.pi * 261.63 * t)
    e4 = 0.4 * np.sin(2 * np.pi * 329.63 * t)
    y = (c4 + e4).astype(np.float32)
    return _write_wav(tmp_path / "two_notes.wav", y, sr)


@pytest.fixture
def sequential_notes_wav(tmp_path: Path) -> Path:
    """Three 1-second notes played sequentially: C4, E4, G4 (3 seconds total)."""
    sr = 44100
    duration = 1.0
    samples_per_note = int(sr * duration)
    c4 = 0.5 * np.sin(2 * np.pi * 261.63 * np.linspace(0, duration, samples_per_note, endpoint=False))
    e4 = 0.5 * np.sin(2 * np.pi * 329.63 * np.linspace(0, duration, samples_per_note, endpoint=False))
    g4 = 0.5 * np.sin(2 * np.pi * 392.00 * np.linspace(0, duration, samples_per_note, endpoint=False))
    y = np.concatenate([c4, e4, g4]).astype(np.float32)
    return _write_wav(tmp_path / "sequential_notes.wav", y, sr)


@pytest.fixture
def two_freq_wav(tmp_path: Path) -> Path:
    """440 Hz + 1000 Hz simultaneously for 1 second. For isolation tests."""
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y = (0.4 * np.sin(2 * np.pi * 440 * t) + 0.4 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    return _write_wav(tmp_path / "two_freq.wav", y, sr)
```

- [ ] **Step 2: Run existing tests to verify no breakage**

Run: `uv run pytest tests/conftest.py --collect-only`
Expected: fixtures collected, no errors.

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add multi-note audio fixtures for Phase 2"
```

---

### Task 3: Transcription logic + tests (mocked Basic Pitch)

**Files:**
- Create: `src/audio_analysis_mcp/analysis/transcription.py`
- Create: `tests/test_transcription.py`

- [ ] **Step 1: Write failing tests for `transcribe_audio`**

Create `tests/test_transcription.py`:

```python
"""Tests for analysis.transcription — Basic Pitch is mocked."""
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pretty_midi
import pytest

from audio_analysis_mcp.analysis.transcription import transcribe_audio
from audio_analysis_mcp.schemas import NoteEvent


def _mock_predict_result(notes: list[tuple[float, float, int, float]]):
    """Build a mock return value matching basic_pitch.inference.predict signature.

    predict returns: (model_output_dict, pretty_midi.PrettyMIDI, note_events_list)
    Each note event: (start_s, end_s, pitch_midi, velocity, pitch_bends)
    """
    model_output = {"note": np.zeros((1, 1)), "onset": np.zeros((1, 1))}
    midi_data = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for start, end, pitch, vel in notes:
        note = pretty_midi.Note(
            velocity=int(vel * 127), pitch=pitch, start=start, end=end,
        )
        inst.notes.append(note)
    midi_data.instruments.append(inst)
    note_events = [(s, e, p, v, None) for s, e, p, v in notes]
    return model_output, midi_data, note_events


@patch("audio_analysis_mcp.analysis.transcription.predict")
def test_single_note(mock_predict: MagicMock, sine_440_wav: Path, tmp_path: Path):
    mock_predict.return_value = _mock_predict_result([
        (0.05, 0.95, 69, 0.8),  # A4 = MIDI 69
    ])
    midi_path, notes = transcribe_audio(str(sine_440_wav), output_dir=str(tmp_path))
    assert Path(midi_path).exists()
    assert Path(midi_path).suffix == ".mid"
    assert len(notes) == 1
    assert notes[0].pitch_midi == 69
    assert notes[0].start_time == pytest.approx(0.05, abs=0.01)
    assert notes[0].end_time == pytest.approx(0.95, abs=0.01)
    assert 0.0 <= notes[0].amplitude <= 1.0


@patch("audio_analysis_mcp.analysis.transcription.predict")
def test_two_simultaneous_notes(mock_predict: MagicMock, two_note_wav: Path, tmp_path: Path):
    mock_predict.return_value = _mock_predict_result([
        (0.05, 0.95, 60, 0.7),  # C4
        (0.05, 0.95, 64, 0.7),  # E4
    ])
    midi_path, notes = transcribe_audio(str(two_note_wav), output_dir=str(tmp_path))
    assert len(notes) == 2
    pitches = {n.pitch_midi for n in notes}
    assert pitches == {60, 64}


@patch("audio_analysis_mcp.analysis.transcription.predict")
def test_note_event_fields_populated(mock_predict: MagicMock, sine_440_wav: Path, tmp_path: Path):
    mock_predict.return_value = _mock_predict_result([
        (0.1, 0.9, 69, 0.6),
    ])
    _, notes = transcribe_audio(str(sine_440_wav), output_dir=str(tmp_path))
    note = notes[0]
    assert isinstance(note.start_time, float)
    assert isinstance(note.end_time, float)
    assert isinstance(note.pitch_midi, int)
    assert isinstance(note.amplitude, float)
    assert note.pitch_bends is None  # mock returns None for pitch_bends


@patch("audio_analysis_mcp.analysis.transcription.predict")
def test_empty_transcription(mock_predict: MagicMock, silence_wav: Path, tmp_path: Path):
    mock_predict.return_value = _mock_predict_result([])
    midi_path, notes = transcribe_audio(str(silence_wav), output_dir=str(tmp_path))
    assert Path(midi_path).exists()
    assert notes == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_transcription.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'audio_analysis_mcp.analysis.transcription'`

- [ ] **Step 3: Implement `analysis/transcription.py`**

Create `src/audio_analysis_mcp/analysis/transcription.py`:

```python
from pathlib import Path
import uuid

from basic_pitch.inference import predict

from audio_analysis_mcp.schemas import NoteEvent


def transcribe_audio(
    audio_path: str,
    output_dir: str,
) -> tuple[str, list[NoteEvent]]:
    """Run Basic Pitch on audio file.

    Returns (midi_path, note_events).
    """
    model_output, midi_data, note_events = predict(audio_path)

    # Save MIDI file
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    midi_path = out / f"transcription_{uuid.uuid4().hex[:8]}.mid"
    midi_data.write(str(midi_path))

    # Convert to NoteEvent schema
    notes: list[NoteEvent] = []
    for start_s, end_s, pitch_midi, velocity, pitch_bends in note_events:
        notes.append(
            NoteEvent(
                start_time=float(start_s),
                end_time=float(end_s),
                pitch_midi=int(pitch_midi),
                amplitude=float(velocity),
                pitch_bends=list(pitch_bends) if pitch_bends is not None else None,
            )
        )
    return str(midi_path), notes
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_transcription.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Type-check**

Run: `uv run mypy src/audio_analysis_mcp/analysis/transcription.py`
Expected: Success. If basic_pitch has no stubs, add `"basic_pitch.*"` to the `ignore_missing_imports` list in `pyproject.toml` under `[[tool.mypy.overrides]]`.

- [ ] **Step 6: Commit**

```bash
git add src/audio_analysis_mcp/analysis/transcription.py tests/test_transcription.py
git commit -m "feat: add transcription logic with Basic Pitch integration"
```

---

### Task 4: Note triage logic + tests

**Files:**
- Create: `src/audio_analysis_mcp/analysis/note_triage.py`
- Create: `tests/test_note_triage.py`

- [ ] **Step 1: Write failing tests for `triage_notes`**

Create `tests/test_note_triage.py`:

```python
"""Tests for analysis.note_triage — pure logic, no mocks needed."""
import pytest
import librosa

from audio_analysis_mcp.analysis.note_triage import triage_notes
from audio_analysis_mcp.schemas import NoteEvent


def _note(start: float, end: float, pitch: int, amp: float = 0.8) -> NoteEvent:
    return NoteEvent(
        start_time=start, end_time=end,
        pitch_midi=pitch, amplitude=amp, pitch_bends=None,
    )


class TestPolyphonyProfile:
    def test_sequential_monophonic(self):
        """Three non-overlapping notes → all windows show count <= 1."""
        notes = [_note(0.0, 0.8, 60), _note(1.0, 1.8, 64), _note(2.0, 2.8, 67)]
        result = triage_notes(notes)
        for window in result.polyphony_profile:
            assert window.note_count <= 1

    def test_overlapping_notes(self):
        """Two notes overlapping in time → at least one window with count 2."""
        notes = [_note(0.0, 1.0, 60), _note(0.5, 1.5, 64)]
        result = triage_notes(notes)
        max_count = max(w.note_count for w in result.polyphony_profile)
        assert max_count == 2

    def test_windows_cover_full_range(self):
        """Profile windows span from 0 to past the last note offset."""
        notes = [_note(0.0, 0.8, 60), _note(2.0, 3.0, 67)]
        result = triage_notes(notes)
        assert result.polyphony_profile[0].start_time == 0.0
        assert result.polyphony_profile[-1].end_time >= 3.0


class TestCandidateSelection:
    def test_prefer_monophonic_over_polyphonic(self):
        """A note in a monophonic window scores higher than one in polyphonic."""
        mono_note = _note(0.0, 1.0, 60)
        poly_note1 = _note(2.0, 3.0, 64)
        poly_note2 = _note(2.2, 3.0, 67)  # overlaps with poly_note1
        result = triage_notes([mono_note, poly_note1, poly_note2])
        # The monophonic note (pitch 60) should be ranked first
        assert result.candidates[0].note.pitch_midi == 60

    def test_min_duration_filter(self):
        """Notes shorter than min_duration are excluded."""
        short = _note(0.0, 0.3, 60)  # 0.3s — below default 0.5s threshold
        long = _note(1.0, 2.0, 64)   # 1.0s — above threshold
        result = triage_notes([short, long], min_duration=0.5)
        pitches = [c.note.pitch_midi for c in result.candidates]
        assert 64 in pitches
        assert 60 not in pitches

    def test_max_candidates_limits_output(self):
        """No more than max_candidates returned."""
        notes = [_note(float(i), float(i) + 0.8, 60 + i) for i in range(20)]
        result = triage_notes(notes, max_candidates=5)
        assert len(result.candidates) <= 5

    def test_frequency_bounds_from_midi(self):
        """Candidate freq bounds: lower = fundamental*0.9, upper = min(fundamental*8, 10kHz)."""
        note = _note(0.0, 1.0, 69)  # A4 = 440 Hz
        result = triage_notes([note])
        assert len(result.candidates) == 1
        c = result.candidates[0]
        fundamental = librosa.midi_to_hz(69)  # 440.0
        assert c.start_freq == pytest.approx(fundamental * 0.9, rel=0.01)
        assert c.end_freq == pytest.approx(min(fundamental * 8, 10000.0), rel=0.01)

    def test_time_bounds_include_padding(self):
        """Candidate time bounds include 50ms padding before and after."""
        note = _note(1.0, 2.0, 60)
        result = triage_notes([note])
        c = result.candidates[0]
        assert c.start_time == pytest.approx(0.95, abs=0.01)
        assert c.end_time == pytest.approx(2.05, abs=0.01)

    def test_time_padding_clamps_to_zero(self):
        """Start time padding doesn't go below 0."""
        note = _note(0.01, 1.0, 60)
        result = triage_notes([note])
        assert result.candidates[0].start_time >= 0.0

    def test_empty_notes(self):
        """Empty input → empty profile and candidates."""
        result = triage_notes([])
        assert result.polyphony_profile == []
        assert result.candidates == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_note_triage.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'audio_analysis_mcp.analysis.note_triage'`

- [ ] **Step 3: Implement `analysis/note_triage.py`**

Create `src/audio_analysis_mcp/analysis/note_triage.py`:

```python
import librosa
import numpy as np

from audio_analysis_mcp.schemas import (
    NoteEvent,
    PolyphonyWindow,
    CandidateNote,
    NoteTriageResult,
)

WINDOW_SIZE = 0.5  # seconds
TIME_PADDING = 0.05  # 50ms before/after note
MAX_FREQ_HZ = 10000.0
NUM_HARMONICS = 8


def _build_polyphony_profile(
    notes: list[NoteEvent],
) -> list[PolyphonyWindow]:
    """Divide timeline into fixed windows and count simultaneous notes per window."""
    if not notes:
        return []
    end_time = max(n.end_time for n in notes)
    windows: list[PolyphonyWindow] = []
    t = 0.0
    while t < end_time:
        w_end = t + WINDOW_SIZE
        count = sum(
            1 for n in notes if n.start_time < w_end and n.end_time > t
        )
        windows.append(PolyphonyWindow(start_time=t, end_time=w_end, note_count=count))
        t = w_end
    return windows


def _polyphony_at(note: NoteEvent, profile: list[PolyphonyWindow]) -> float:
    """Average polyphony count across windows that overlap with this note."""
    overlapping = [
        w for w in profile if w.start_time < note.end_time and w.end_time > note.start_time
    ]
    if not overlapping:
        return 1.0
    return sum(w.note_count for w in overlapping) / len(overlapping)


def _temporal_gap(note: NoteEvent, all_notes: list[NoteEvent]) -> float:
    """Minimum time gap to the nearest neighboring note (seconds)."""
    min_gap = float("inf")
    for other in all_notes:
        if other is note:
            continue
        gap = max(0.0, max(other.start_time - note.end_time, note.start_time - other.end_time))
        min_gap = min(min_gap, gap)
    return min_gap if min_gap != float("inf") else 1.0


def _freq_bounds(pitch_midi: int) -> tuple[float, float]:
    """Compute frequency isolation bounds from MIDI pitch."""
    fundamental: float = float(librosa.midi_to_hz(pitch_midi))
    lower = fundamental * 0.9
    upper = min(fundamental * NUM_HARMONICS, MAX_FREQ_HZ)
    return lower, upper


def _score_note(
    note: NoteEvent,
    profile: list[PolyphonyWindow],
    all_notes: list[NoteEvent],
) -> float:
    """Score a note for isolation suitability. Higher = better."""
    duration = note.end_time - note.start_time
    poly = _polyphony_at(note, profile)
    gap = _temporal_gap(note, all_notes)

    # Low polyphony is best (invert: 1/poly)
    poly_score = 1.0 / max(poly, 1.0)
    # Duration: log scale, capped at diminishing returns past 2s
    dur_score = float(np.log1p(min(duration, 2.0)))
    # Temporal gap: more gap is better, log scale
    gap_score = float(np.log1p(gap))

    return poly_score * 2.0 + dur_score + gap_score * 0.5


def triage_notes(
    notes: list[NoteEvent],
    min_duration: float = 0.5,
    max_candidates: int = 10,
) -> NoteTriageResult:
    """Profile polyphony and rank candidate notes for isolation."""
    profile = _build_polyphony_profile(notes)

    if not notes:
        return NoteTriageResult(polyphony_profile=[], candidates=[])

    # Filter by duration
    eligible = [n for n in notes if (n.end_time - n.start_time) >= min_duration]

    # Score and rank
    scored = [(n, _score_note(n, profile, notes)) for n in eligible]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Pitch diversity: greedily select from scored list, penalizing nearby pitches
    selected: list[tuple[NoteEvent, float]] = []
    selected_pitches: list[int] = []
    for note, score in scored:
        if len(selected) >= max_candidates:
            break
        # Penalize if a very similar pitch already selected (within 2 semitones)
        if any(abs(note.pitch_midi - p) <= 2 for p in selected_pitches):
            score *= 0.5
        selected.append((note, score))
        selected_pitches.append(note.pitch_midi)

    # Re-sort after diversity adjustment
    selected.sort(key=lambda x: x[1], reverse=True)
    selected = selected[:max_candidates]

    # Build candidates with time/freq bounds
    candidates: list[CandidateNote] = []
    for note, score in selected:
        start_freq, end_freq = _freq_bounds(note.pitch_midi)
        padded_start = max(0.0, note.start_time - TIME_PADDING)
        padded_end = note.end_time + TIME_PADDING
        candidates.append(
            CandidateNote(
                note=note,
                score=round(score, 4),
                start_time=padded_start,
                end_time=padded_end,
                start_freq=round(start_freq, 2),
                end_freq=round(end_freq, 2),
            )
        )

    return NoteTriageResult(polyphony_profile=profile, candidates=candidates)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_note_triage.py -v`
Expected: all 8 tests PASS.

- [ ] **Step 5: Type-check**

Run: `uv run mypy src/audio_analysis_mcp/analysis/note_triage.py`
Expected: Success.

- [ ] **Step 6: Commit**

```bash
git add src/audio_analysis_mcp/analysis/note_triage.py tests/test_note_triage.py
git commit -m "feat: add note triage logic with polyphony profiling and candidate ranking"
```

---

### Task 5: Note isolation logic + tests

**Files:**
- Create: `src/audio_analysis_mcp/analysis/note_isolation.py`
- Create: `tests/test_note_isolation.py`

- [ ] **Step 1: Write failing tests for `isolate_note`**

Create `tests/test_note_isolation.py`:

```python
"""Tests for analysis.note_isolation — STFT time-frequency masking."""
from pathlib import Path

import numpy as np
import librosa
import pytest

from audio_analysis_mcp.analysis.note_isolation import isolate_note


def test_single_sine_in_box(sine_440_wav: Path):
    """A 440Hz sine isolated with a box that includes 440Hz → high energy output."""
    y_isolated, sr = isolate_note(
        audio_path=str(sine_440_wav),
        start_time=0.0,
        end_time=1.0,
        start_freq=400.0,
        end_freq=500.0,
    )
    rms = float(np.sqrt(np.mean(y_isolated**2)))
    assert rms > 0.1  # significant energy preserved


def test_isolate_one_of_two_freqs(two_freq_wav: Path):
    """440Hz + 1000Hz signal, isolate 440Hz → 1000Hz attenuated."""
    y_isolated, sr = isolate_note(
        audio_path=str(two_freq_wav),
        start_time=0.0,
        end_time=1.0,
        start_freq=400.0,
        end_freq=500.0,
    )
    # Check that 1000Hz is strongly attenuated
    S = np.abs(librosa.stft(y_isolated))
    freqs = librosa.fft_frequencies(sr=sr)
    bin_440 = int(np.argmin(np.abs(freqs - 440)))
    bin_1000 = int(np.argmin(np.abs(freqs - 1000)))
    energy_440 = float(np.mean(S[bin_440, :]))
    energy_1000 = float(np.mean(S[bin_1000, :]))
    assert energy_440 > energy_1000 * 5  # 440Hz should dominate


def test_time_window_slicing(sequential_notes_wav: Path):
    """3-second audio (3 notes), isolate second 1-2s → duration ~1s."""
    y_isolated, sr = isolate_note(
        audio_path=str(sequential_notes_wav),
        start_time=1.0,
        end_time=2.0,
        start_freq=20.0,
        end_freq=10000.0,
    )
    duration = len(y_isolated) / sr
    assert duration == pytest.approx(1.0, abs=0.05)


def test_output_duration_matches_window(sine_440_wav: Path):
    """Requested window of 0.5s → output is ~0.5s."""
    y_isolated, sr = isolate_note(
        audio_path=str(sine_440_wav),
        start_time=0.2,
        end_time=0.7,
        start_freq=20.0,
        end_freq=10000.0,
    )
    duration = len(y_isolated) / sr
    assert duration == pytest.approx(0.5, abs=0.05)


def test_invalid_time_range(sine_440_wav: Path):
    """start_time >= end_time → ValueError."""
    with pytest.raises(ValueError):
        isolate_note(
            audio_path=str(sine_440_wav),
            start_time=0.5,
            end_time=0.5,
            start_freq=100.0,
            end_freq=1000.0,
        )


def test_invalid_freq_range(sine_440_wav: Path):
    """start_freq >= end_freq → ValueError."""
    with pytest.raises(ValueError):
        isolate_note(
            audio_path=str(sine_440_wav),
            start_time=0.0,
            end_time=1.0,
            start_freq=1000.0,
            end_freq=100.0,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_note_isolation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'audio_analysis_mcp.analysis.note_isolation'`

- [ ] **Step 3: Implement `analysis/note_isolation.py`**

Create `src/audio_analysis_mcp/analysis/note_isolation.py`:

```python
import numpy as np
import librosa


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
    y_isolated: np.ndarray = librosa.istft(S_masked, hop_length=hop_length)
    return y_isolated, sr
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_note_isolation.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Type-check**

Run: `uv run mypy src/audio_analysis_mcp/analysis/note_isolation.py`
Expected: Success.

- [ ] **Step 6: Commit**

```bash
git add src/audio_analysis_mcp/analysis/note_isolation.py tests/test_note_isolation.py
git commit -m "feat: add note isolation via STFT time-frequency masking"
```

---

### Task 6: MCP tool wrappers + registration

**Files:**
- Create: `src/audio_analysis_mcp/tools/note_transcribe.py`
- Create: `src/audio_analysis_mcp/tools/note_triage.py`
- Create: `src/audio_analysis_mcp/tools/note_isolate.py`
- Modify: `src/audio_analysis_mcp/__main__.py`

- [ ] **Step 1: Create `tools/note_transcribe.py`**

```python
from audio_analysis_mcp.server import mcp, get_workspace
from audio_analysis_mcp.analysis.transcription import transcribe_audio
from audio_analysis_mcp.schemas import NoteTranscribeResult


@mcp.tool()
def note_transcribe(audio_path: str) -> str:
    """Transcribe polyphonic audio to MIDI note events using Basic Pitch."""
    ws = get_workspace()
    midi_path, notes = transcribe_audio(audio_path, output_dir=str(ws.transcriptions))
    return NoteTranscribeResult(midi_path=midi_path, notes=notes).model_dump_json(indent=2)
```

- [ ] **Step 2: Create `tools/note_triage.py`**

```python
from pydantic import TypeAdapter

from audio_analysis_mcp.server import mcp
from audio_analysis_mcp.analysis.note_triage import triage_notes
from audio_analysis_mcp.schemas import NoteEvent


@mcp.tool()
def note_triage(
    audio_path: str,
    notes_json: str,
    min_duration: float = 0.5,
    max_candidates: int = 10,
) -> str:
    """Analyze transcription and select best candidate notes for isolation."""
    adapter = TypeAdapter(list[NoteEvent])
    notes = adapter.validate_json(notes_json)
    result = triage_notes(
        notes=notes,
        min_duration=min_duration,
        max_candidates=max_candidates,
    )
    return result.model_dump_json(indent=2)
```

- [ ] **Step 3: Create `tools/note_isolate.py`**

```python
import uuid
import soundfile as sf

from audio_analysis_mcp.server import mcp, get_workspace
from audio_analysis_mcp.analysis.note_isolation import isolate_note
from audio_analysis_mcp.schemas import NoteIsolateResult


@mcp.tool()
def note_isolate(
    audio_path: str,
    start_time: float,
    end_time: float,
    start_freq: float,
    end_freq: float,
) -> str:
    """Isolate a sound from audio within a time-frequency box using STFT masking."""
    if start_time < 0:
        raise ValueError(f"start_time must be >= 0, got {start_time}")
    if end_time <= start_time:
        raise ValueError(f"end_time ({end_time}) must be > start_time ({start_time})")
    if start_freq < 0:
        raise ValueError(f"start_freq must be >= 0, got {start_freq}")
    if end_freq <= start_freq:
        raise ValueError(f"end_freq ({end_freq}) must be > start_freq ({start_freq})")

    ws = get_workspace()
    y_isolated, sr = isolate_note(
        audio_path=audio_path,
        start_time=start_time,
        end_time=end_time,
        start_freq=start_freq,
        end_freq=end_freq,
    )
    out_path = ws.isolated_notes / f"isolated_{uuid.uuid4().hex[:8]}.wav"
    sf.write(str(out_path), y_isolated, sr, subtype="PCM_16")
    duration = len(y_isolated) / sr

    return NoteIsolateResult(
        audio_path=str(out_path),
        duration_seconds=round(duration, 3),
    ).model_dump_json(indent=2)
```

- [ ] **Step 4: Register tools in `__main__.py`**

Add these three lines after the existing tool imports in `src/audio_analysis_mcp/__main__.py`:

```python
import audio_analysis_mcp.tools.note_transcribe  # noqa: F401
import audio_analysis_mcp.tools.note_triage  # noqa: F401
import audio_analysis_mcp.tools.note_isolate  # noqa: F401
```

- [ ] **Step 5: Type-check all new tools**

Run: `uv run mypy src/audio_analysis_mcp/tools/note_transcribe.py src/audio_analysis_mcp/tools/note_triage.py src/audio_analysis_mcp/tools/note_isolate.py`
Expected: Success. If `pretty_midi` has no stubs, add `"pretty_midi.*"` to the mypy overrides in `pyproject.toml`.

- [ ] **Step 6: Commit**

```bash
git add src/audio_analysis_mcp/tools/note_transcribe.py src/audio_analysis_mcp/tools/note_triage.py src/audio_analysis_mcp/tools/note_isolate.py src/audio_analysis_mcp/__main__.py
git commit -m "feat: add MCP tool wrappers for note_transcribe, note_triage, note_isolate"
```

---

### Task 7: E2E tests

**Files:**
- Modify: `tests/test_mcp_tools.py`

- [ ] **Step 1: Add E2E tests for Phase 2 tools**

Add these imports at the top of `tests/test_mcp_tools.py`, alongside the existing tool imports:

```python
import audio_analysis_mcp.tools.note_transcribe  # noqa: F401
import audio_analysis_mcp.tools.note_triage  # noqa: F401
import audio_analysis_mcp.tools.note_isolate  # noqa: F401
```

Then add these test functions at the end of the file:

```python
from unittest.mock import patch, MagicMock
import numpy as np
import pretty_midi


def _mock_predict_result(notes: list[tuple[float, float, int, float]]):
    model_output = {"note": np.zeros((1, 1)), "onset": np.zeros((1, 1))}
    midi_data = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for start, end, pitch, vel in notes:
        note = pretty_midi.Note(
            velocity=int(vel * 127), pitch=pitch, start=start, end=end,
        )
        inst.notes.append(note)
    midi_data.instruments.append(inst)
    note_events = [(s, e, p, v, None) for s, e, p, v in notes]
    return model_output, midi_data, note_events


@patch("audio_analysis_mcp.analysis.transcription.predict")
def test_note_transcribe_e2e(mock_predict: MagicMock, sine_440_wav: Path):
    from audio_analysis_mcp.tools.note_transcribe import note_transcribe

    mock_predict.return_value = _mock_predict_result([
        (0.05, 0.95, 69, 0.8),
    ])
    result = json.loads(note_transcribe(audio_path=str(sine_440_wav)))
    assert Path(result["midi_path"]).exists()
    assert len(result["notes"]) == 1
    assert result["notes"][0]["pitch_midi"] == 69


def test_note_isolate_e2e(sine_440_wav: Path):
    from audio_analysis_mcp.tools.note_isolate import note_isolate

    result = json.loads(
        note_isolate(
            audio_path=str(sine_440_wav),
            start_time=0.0,
            end_time=0.5,
            start_freq=400.0,
            end_freq=500.0,
        )
    )
    assert Path(result["audio_path"]).exists()
    assert result["duration_seconds"] > 0
```

- [ ] **Step 2: Run all E2E tests**

Run: `uv run pytest tests/test_mcp_tools.py -v`
Expected: all tests PASS (existing + 2 new).

- [ ] **Step 3: Commit**

```bash
git add tests/test_mcp_tools.py
git commit -m "test: add E2E tests for Phase 2 MCP tools"
```

---

### Task 8: Full test suite and type-check

**Files:** None (verification only)

- [ ] **Step 1: Run entire test suite**

Run: `uv run pytest -v -m "not slow"`
Expected: all tests PASS — both Phase 1 and Phase 2.

- [ ] **Step 2: Run mypy on entire source**

Run: `uv run mypy src/`
Expected: Success, no errors.

- [ ] **Step 3: Final commit if any mypy fixups were needed**

If any type errors required fixes (e.g., adding to mypy overrides), commit them:

```bash
git add -u
git commit -m "fix: resolve mypy type errors for Phase 2"
```

---

### Task 9: Update pipeline plan

**Files:**
- Modify: `audio-pipeline-plan.md`

- [ ] **Step 1: Mark Phase 2 as completed in the plan**

In `audio-pipeline-plan.md`, change line 398:

```markdown
### Phase 2: Note-Level Extraction ✅
```

- [ ] **Step 2: Commit**

```bash
git add audio-pipeline-plan.md
git commit -m "docs: mark Phase 2 as completed in pipeline plan"
```