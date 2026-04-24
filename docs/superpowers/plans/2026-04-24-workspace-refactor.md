# Workspace Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the flat workspace layout with a job-centric folder structure where all pipeline outputs for a source audio file live together, and fix `note_triage` to write results to a file instead of returning 64K+ JSON.

**Architecture:** A new `JobContext` dataclass and path-resolution functions in `workspace.py` provide the folder structure. Each tool resolves its output paths via the workspace module. Old flat directories remain for backward compatibility.

**Tech Stack:** Python 3.11, pathlib, re, pydantic, pytest

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/audio_analysis_mcp/workspace.py` | Modify | Add `sanitize_job_name`, `JobContext`, `resolve_job_context`, job-path methods |
| `src/audio_analysis_mcp/schemas.py` | Modify | Add `job_name` to `ImportAudioResult`, replace `NoteTriageResult` with file-based version |
| `src/audio_analysis_mcp/tools/import_audio.py` | Modify | Write to `jobs/<name>/source.wav`, return `job_name` |
| `src/audio_analysis_mcp/tools/stem_separate.py` | Modify | Write to `jobs/<job>/stems/<preset>/`, resolve job from input path |
| `src/audio_analysis_mcp/tools/note_transcribe.py` | Modify | Write to `jobs/<job>/transcriptions/<stem>_<preset>/` |
| `src/audio_analysis_mcp/tools/note_triage.py` | Modify | Write triage.json to `jobs/<job>/triage/<stem>_<preset>/`, return path |
| `src/audio_analysis_mcp/tools/note_isolate.py` | Modify | Write to `jobs/<job>/isolated_notes/<stem>_<preset>/` with human-readable names |
| `src/audio_analysis_mcp/analysis/note_triage.py` | Modify | Return full data model (unchanged), add `NoteTriageFileData` schema |
| `tests/test_workspace.py` | Modify | Add tests for `sanitize_job_name`, `resolve_job_context` |
| `tests/test_mcp_tools.py` | Modify | Update E2E tests for new output paths and schemas |
| `tests/test_import_audio.py` | No change | Tests normalize_audio directly, not affected |
| `tests/test_note_triage.py` | No change | Tests `triage_notes` logic directly, not the tool wrapper |

---

### Task 1: Workspace module — sanitize and resolve

**Files:**
- Modify: `src/audio_analysis_mcp/workspace.py`
- Modify: `tests/test_workspace.py`

- [ ] **Step 1: Write failing tests for `sanitize_job_name`**

Add to `tests/test_workspace.py`:

```python
from audio_analysis_mcp.workspace import sanitize_job_name


def test_sanitize_basic():
    assert sanitize_job_name("My Song.wav") == "my-song"


def test_sanitize_parentheses():
    assert sanitize_job_name("Smooth Criminal (Radio Edit).wav") == "smooth-criminal-radio-edit"


def test_sanitize_special_chars():
    assert sanitize_job_name("song #1 [feat. Artist] & More!.mp3") == "song-1-feat-artist-more"


def test_sanitize_multiple_spaces():
    assert sanitize_job_name("  too   many   spaces  .wav") == "too-many-spaces"


def test_sanitize_unicode():
    assert sanitize_job_name("café naïve.wav") == "caf-nave"


def test_sanitize_dots_and_underscores():
    assert sanitize_job_name("my_song.v2.final.wav") == "my-song-v2-final"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_workspace.py::test_sanitize_basic -v`
Expected: FAIL — `ImportError: cannot import name 'sanitize_job_name'`

- [ ] **Step 3: Write failing tests for `resolve_job_context`**

Add to `tests/test_workspace.py`:

```python
from audio_analysis_mcp.workspace import resolve_job_context, Workspace


def test_resolve_source(tmp_path: Path):
    ws = Workspace(tmp_path)
    source = tmp_path / "jobs" / "my-song" / "source.wav"
    source.parent.mkdir(parents=True)
    source.touch()
    ctx = resolve_job_context(str(source), ws)
    assert ctx.job_name == "my-song"
    assert ctx.stem is None
    assert ctx.preset is None


def test_resolve_stem(tmp_path: Path):
    ws = Workspace(tmp_path)
    stem = tmp_path / "jobs" / "my-song" / "stems" / "medium" / "bass.wav"
    stem.parent.mkdir(parents=True)
    stem.touch()
    ctx = resolve_job_context(str(stem), ws)
    assert ctx.job_name == "my-song"
    assert ctx.stem == "bass"
    assert ctx.preset == "medium"


def test_resolve_outside_workspace(tmp_path: Path):
    ws = Workspace(tmp_path)
    import pytest
    with pytest.raises(ValueError, match="not inside"):
        resolve_job_context("/some/other/path.wav", ws)
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/test_workspace.py::test_resolve_source -v`
Expected: FAIL — `ImportError: cannot import name 'resolve_job_context'`

- [ ] **Step 5: Implement `sanitize_job_name`, `JobContext`, `resolve_job_context`, and job-path methods**

Replace the full content of `src/audio_analysis_mcp/workspace.py`:

```python
from dataclasses import dataclass
from pathlib import Path
import re


DEFAULT_ROOT = Path.home() / ".audio-analysis-mcp" / "workspace"


def sanitize_job_name(filename: str) -> str:
    """Convert a filename to a filesystem-safe, human-readable slug."""
    # Remove file extension
    name = Path(filename).stem
    # Lowercase
    name = name.lower()
    # Replace non-alphanumeric chars with hyphens
    name = re.sub(r"[^a-z0-9]+", "-", name)
    # Collapse multiple hyphens
    name = re.sub(r"-+", "-", name)
    # Trim leading/trailing hyphens
    name = name.strip("-")
    return name


@dataclass
class JobContext:
    job_name: str
    stem: str | None = None
    preset: str | None = None


def resolve_job_context(path: str, workspace: "Workspace") -> JobContext:
    """Parse job name, stem, and preset from a path within the workspace.

    Expects paths like:
      jobs/<job>/source.wav
      jobs/<job>/stems/<preset>/<stem>.wav
      jobs/<job>/transcriptions/<stem>_<preset>/transcription.json
    """
    p = Path(path).resolve()
    root = workspace.root.resolve()
    try:
        rel = p.relative_to(root / "jobs")
    except ValueError:
        raise ValueError(f"Path is not inside the workspace jobs directory: {path}")

    parts = rel.parts
    if len(parts) < 2:
        raise ValueError(f"Path is not inside a job folder: {path}")

    job_name = parts[0]

    # jobs/<job>/stems/<preset>/<stem>.wav
    if len(parts) >= 4 and parts[1] == "stems":
        preset = parts[2]
        stem = Path(parts[3]).stem
        return JobContext(job_name=job_name, stem=stem, preset=preset)

    # jobs/<job>/source.wav or other direct children
    return JobContext(job_name=job_name)


class Workspace:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or DEFAULT_ROOT

    def _ensure(self, name: str) -> Path:
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    # --- Old flat directories (backward compat) ---

    @property
    def imported(self) -> Path:
        return self._ensure("imported")

    @property
    def stems(self) -> Path:
        return self._ensure("stems")

    @property
    def spectrograms(self) -> Path:
        return self._ensure("spectrograms")

    @property
    def transcriptions(self) -> Path:
        return self._ensure("transcriptions")

    @property
    def isolated_notes(self) -> Path:
        return self._ensure("isolated_notes")

    @property
    def rendered(self) -> Path:
        return self._ensure("rendered")

    # --- Job-centric directories ---

    def job_dir(self, job_name: str) -> Path:
        return self._ensure(f"jobs/{job_name}")

    def job_stems_dir(self, job_name: str, preset: str) -> Path:
        return self._ensure(f"jobs/{job_name}/stems/{preset}")

    def job_transcriptions_dir(self, job_name: str, stem: str, preset: str) -> Path:
        return self._ensure(f"jobs/{job_name}/transcriptions/{stem}_{preset}")

    def job_triage_dir(self, job_name: str, stem: str, preset: str) -> Path:
        return self._ensure(f"jobs/{job_name}/triage/{stem}_{preset}")

    def job_isolated_notes_dir(self, job_name: str, stem: str, preset: str) -> Path:
        return self._ensure(f"jobs/{job_name}/isolated_notes/{stem}_{preset}")
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_workspace.py -v`
Expected: all tests PASS (existing + new).

- [ ] **Step 7: Type-check**

Run: `uv run mypy src/audio_analysis_mcp/workspace.py`

- [ ] **Step 8: Commit**

```bash
git add src/audio_analysis_mcp/workspace.py tests/test_workspace.py
git commit -m "feat: add job-centric workspace paths, sanitize_job_name, resolve_job_context"
```

---

### Task 2: Update schemas

**Files:**
- Modify: `src/audio_analysis_mcp/schemas.py`

- [ ] **Step 1: Add `job_name` to `ImportAudioResult`**

Change `ImportAudioResult` to:

```python
class ImportAudioResult(BaseModel):
    audio_path: str
    job_name: str
    sample_rate: int
    duration_seconds: float
    channels: int
```

- [ ] **Step 2: Replace `NoteTriageResult` with file-based version and add `NoteTriageFileData`**

Replace the current `NoteTriageResult` and add `NoteTriageFileData`:

```python
class NoteTriageFileData(BaseModel):
    """Full triage data written to the JSON file."""
    polyphony_profile: list[PolyphonyWindow]
    candidates: list[CandidateNote]


class NoteTriageResult(BaseModel):
    """Lightweight result returned by the MCP tool."""
    triage_path: str
    candidate_count: int
    top_candidate_summary: str
```

- [ ] **Step 3: Type-check**

Run: `uv run mypy src/audio_analysis_mcp/schemas.py`

- [ ] **Step 4: Commit**

```bash
git add src/audio_analysis_mcp/schemas.py
git commit -m "feat: update schemas for job-centric workspace"
```

---

### Task 3: Update `import_audio` tool

**Files:**
- Modify: `src/audio_analysis_mcp/tools/import_audio.py`
- Modify: `tests/test_mcp_tools.py`

- [ ] **Step 1: Update `import_audio` tool**

Replace the full content of `src/audio_analysis_mcp/tools/import_audio.py`:

```python
from pathlib import Path
from audio_analysis_mcp.server import mcp, get_workspace
from audio_analysis_mcp.workspace import sanitize_job_name
from audio_analysis_mcp.audio.normalize import normalize_audio
from audio_analysis_mcp.schemas import ImportAudioResult


@mcp.tool()
def import_audio(
    file_path: str,
    start_time: float | None = None,
    duration: float | None = None,
) -> str:
    """Import a local audio file. Normalize to 44.1kHz 16-bit mono WAV."""
    ws = get_workspace()
    job_name = sanitize_job_name(Path(file_path).name)
    job_dir = ws.job_dir(job_name)
    output_path = job_dir / "source.wav"

    dur, ch = normalize_audio(file_path, str(output_path), start_time, duration)

    return ImportAudioResult(
        audio_path=str(output_path),
        job_name=job_name,
        sample_rate=44100,
        duration_seconds=dur,
        channels=ch,
    ).model_dump_json(indent=2)
```

- [ ] **Step 2: Update `test_import_audio_e2e` in `tests/test_mcp_tools.py`**

Replace the existing `test_import_audio_e2e`:

```python
def test_import_audio_e2e(sine_440_wav: Path):
    from audio_analysis_mcp.tools.import_audio import import_audio

    result = json.loads(import_audio(file_path=str(sine_440_wav)))
    assert result["sample_rate"] == 44100
    assert result["channels"] == 1
    assert result["job_name"] == "sine-440"
    assert Path(result["audio_path"]).exists()
    assert "jobs/sine-440/source.wav" in result["audio_path"]
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_mcp_tools.py::test_import_audio_e2e -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/audio_analysis_mcp/tools/import_audio.py tests/test_mcp_tools.py
git commit -m "feat: import_audio writes to job-centric folder"
```

---

### Task 4: Update `stem_separate` tool

**Files:**
- Modify: `src/audio_analysis_mcp/tools/stem_separate.py`

- [ ] **Step 1: Update `stem_separate_impl` to use job-centric paths**

In `src/audio_analysis_mcp/tools/stem_separate.py`, make these changes:

1. Replace the `_file_hash` function and the cache directory logic in `stem_separate_impl`. The cache key changes from content hash to the directory itself (if stems exist, they're cached).

2. Replace the `stem_separate` MCP tool function to resolve the job context from the input path.

Replace the full content of `src/audio_analysis_mcp/tools/stem_separate.py`:

```python
import json
from dataclasses import dataclass
from pathlib import Path
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import AudioFile, save_audio
from audio_analysis_mcp.server import mcp, get_workspace
from audio_analysis_mcp.workspace import resolve_job_context
from audio_analysis_mcp.schemas import StemFile, StemSeparateResult

MANIFEST_FILE = "sources.json"


@dataclass(frozen=True)
class SeparationPreset:
    model: str
    shifts: int
    overlap: float


PRESETS: dict[str, SeparationPreset] = {
    "fast": SeparationPreset(model="htdemucs_6s", shifts=1, overlap=0.1),
    "medium": SeparationPreset(model="htdemucs_6s", shifts=3, overlap=0.25),
    "accurate": SeparationPreset(model="htdemucs_6s", shifts=7, overlap=0.25),
}


def _read_cached(cache_dir: Path) -> list[str] | None:
    """Read source names from cache manifest. Returns None on cache miss."""
    manifest = cache_dir / MANIFEST_FILE
    if not manifest.exists():
        return None
    try:
        parsed = json.loads(manifest.read_text())
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list) or not all(isinstance(s, str) for s in parsed):
        return None
    source_names: list[str] = parsed
    if all((cache_dir / f"{s}.wav").exists() for s in source_names):
        return source_names
    return None


def _best_device() -> str:
    """Auto-detect the best available compute device."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_preset(preset_name: str) -> SeparationPreset:
    """Validate and return a separation preset."""
    if preset_name not in PRESETS:
        raise ValueError(
            f"Unknown preset: {preset_name}. Allowed: {', '.join(sorted(PRESETS))}"
        )
    return PRESETS[preset_name]


def stem_separate_impl(
    audio_path: str, stems_dir: Path, preset_name: str = "medium"
) -> StemSeparateResult:
    """Run Demucs stem separation via Python API. Returns cached result if available."""
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    preset = _resolve_preset(preset_name)
    cache_dir = stems_dir / f"{preset.model}_{preset_name}"

    # Check cache
    cached_sources = _read_cached(cache_dir)
    if cached_sources is not None:
        return StemSeparateResult(
            stems=[StemFile(stem=s, path=str(cache_dir / f"{s}.wav")) for s in cached_sources],
            model=preset.model,
            preset=preset_name,
            cached=True,
        )

    # Cache miss — load model and run separation
    model = get_model(preset.model)
    model.eval()
    source_names = list(model.sources)

    wav = AudioFile(Path(audio_path)).read(streams=0, samplerate=model.samplerate, channels=model.audio_channels)  # type: ignore[no-untyped-call]
    with torch.no_grad():
        sources = apply_model(
            model, wav[None], device=_best_device(),
            shifts=preset.shifts, overlap=preset.overlap, progress=True,
        )[0]

    cache_dir.mkdir(parents=True, exist_ok=True)
    for i, source_name in enumerate(source_names):
        save_audio(sources[i], cache_dir / f"{source_name}.wav", samplerate=model.samplerate)
    (cache_dir / MANIFEST_FILE).write_text(json.dumps(source_names))

    return StemSeparateResult(
        stems=[StemFile(stem=s, path=str(cache_dir / f"{s}.wav")) for s in source_names],
        model=preset.model,
        preset=preset_name,
        cached=False,
    )


@mcp.tool()
def stem_separate(audio_path: str, preset: str = "fast") -> str:
    """Separate audio into stems using Demucs.

    Input must be inside a job folder (use import_audio first).
    Returns 6 stems: vocals, drums, bass, other, guitar, piano.
    """
    ws = get_workspace()
    ctx = resolve_job_context(audio_path, ws)
    stems_dir = ws.job_stems_dir(ctx.job_name, preset)
    result = stem_separate_impl(audio_path, stems_dir, preset_name=preset)
    return result.model_dump_json(indent=2)
```

Key changes:
- Removed `_file_hash` — no longer needed since cache is by job+preset directory
- `stem_separate_impl` receives `stems_dir` directly (the preset-specific directory), cache is flat within it
- `stem_separate` MCP tool resolves job context from input path

- [ ] **Step 2: Update stem_separate tests**

The tests call `stem_separate_impl` directly with a `tmp_path`, so they still work — the function signature is unchanged. But the cache directory structure changed: previously `stems_dir / <hash> / <model>_<preset>`, now `stems_dir / <model>_<preset>` (the hash level is removed).

Run: `uv run pytest tests/test_stem_separate.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add src/audio_analysis_mcp/tools/stem_separate.py
git commit -m "feat: stem_separate uses job-centric paths, resolves job from input"
```

---

### Task 5: Update `note_transcribe` tool

**Files:**
- Modify: `src/audio_analysis_mcp/tools/note_transcribe.py`

- [ ] **Step 1: Update `note_transcribe` to write to job folder**

Replace the full content of `src/audio_analysis_mcp/tools/note_transcribe.py`:

```python
from audio_analysis_mcp.server import mcp, get_workspace
from audio_analysis_mcp.workspace import resolve_job_context
from audio_analysis_mcp.analysis.transcription import transcribe_audio
from audio_analysis_mcp.schemas import NoteTranscribeResult


@mcp.tool()
def note_transcribe(audio_path: str) -> str:
    """Transcribe polyphonic audio to MIDI note events using Basic Pitch.

    Input must be a stem file inside a job folder.
    """
    ws = get_workspace()
    ctx = resolve_job_context(audio_path, ws)
    if ctx.stem is None or ctx.preset is None:
        raise ValueError(
            f"Expected a stem file (jobs/<job>/stems/<preset>/<stem>.wav), got: {audio_path}"
        )
    output_dir = str(ws.job_transcriptions_dir(ctx.job_name, ctx.stem, ctx.preset))
    midi_path, notes_path, notes = transcribe_audio(audio_path, output_dir=output_dir)
    return NoteTranscribeResult(
        midi_path=midi_path,
        notes_path=notes_path,
        note_count=len(notes),
    ).model_dump_json(indent=2)
```

- [ ] **Step 2: Update E2E test**

In `tests/test_mcp_tools.py`, update `test_note_transcribe_e2e` to set up a job folder structure:

```python
@patch("audio_analysis_mcp.analysis.transcription.predict")
def test_note_transcribe_e2e(mock_predict: MagicMock, sine_440_wav: Path, tmp_path: Path):
    from audio_analysis_mcp.tools.note_transcribe import note_transcribe

    # Set up a job folder with a stem file
    stem_dir = tmp_path / "workspace" / "jobs" / "test-song" / "stems" / "fast"
    stem_dir.mkdir(parents=True)
    stem_file = stem_dir / "bass.wav"
    import shutil
    shutil.copy(sine_440_wav, stem_file)

    mock_predict.return_value = _mock_predict_result([
        (0.05, 0.95, 69, 0.8),
    ])
    result = json.loads(note_transcribe(audio_path=str(stem_file)))
    assert Path(result["midi_path"]).exists()
    assert Path(result["notes_path"]).exists()
    assert result["note_count"] == 1
    assert "test-song/transcriptions/bass_fast" in result["midi_path"]
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_mcp_tools.py::test_note_transcribe_e2e -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/audio_analysis_mcp/tools/note_transcribe.py tests/test_mcp_tools.py
git commit -m "feat: note_transcribe writes to job-centric transcriptions folder"
```

---

### Task 6: Update `note_triage` tool — file-based output

**Files:**
- Modify: `src/audio_analysis_mcp/tools/note_triage.py`
- Modify: `src/audio_analysis_mcp/analysis/note_triage.py`

- [ ] **Step 1: Update `analysis/note_triage.py` return type**

The `triage_notes` function currently returns `NoteTriageResult` which we've renamed to hold MCP-level fields. Change it to return `NoteTriageFileData`:

At the top of `src/audio_analysis_mcp/analysis/note_triage.py`, update the import:

```python
from audio_analysis_mcp.schemas import (
    NoteEvent,
    PolyphonyWindow,
    CandidateNote,
    NoteTriageFileData,
)
```

Change the return type annotation and the return statement of `triage_notes`:

```python
def triage_notes(
    notes: list[NoteEvent],
    min_duration: float = 0.5,
    max_candidates: int = 10,
) -> NoteTriageFileData:
```

And both return statements to use `NoteTriageFileData` instead of `NoteTriageResult`:

```python
    return NoteTriageFileData(polyphony_profile=[], candidates=[])
```

```python
    return NoteTriageFileData(polyphony_profile=profile, candidates=candidates)
```

- [ ] **Step 2: Update `tools/note_triage.py` to write file and return summary**

Replace the full content of `src/audio_analysis_mcp/tools/note_triage.py`:

```python
from pathlib import Path

from pydantic import TypeAdapter

from audio_analysis_mcp.server import mcp, get_workspace
from audio_analysis_mcp.workspace import resolve_job_context
from audio_analysis_mcp.analysis.note_triage import triage_notes
from audio_analysis_mcp.schemas import NoteEvent, NoteTriageResult


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _midi_to_name(midi: int) -> str:
    return NOTE_NAMES[midi % 12] + str(midi // 12 - 1)


@mcp.tool()
def note_triage(
    audio_path: str,
    notes_path: str,
    min_duration: float = 0.5,
    max_candidates: int = 10,
) -> str:
    """Analyze transcription and select best candidate notes for isolation.

    notes_path must be the JSON file from note_transcribe.
    """
    ws = get_workspace()
    ctx = resolve_job_context(audio_path, ws)
    if ctx.stem is None or ctx.preset is None:
        raise ValueError(
            f"Expected a stem file (jobs/<job>/stems/<preset>/<stem>.wav), got: {audio_path}"
        )

    adapter = TypeAdapter(list[NoteEvent])
    notes_json = Path(notes_path).read_text()
    notes = adapter.validate_json(notes_json)

    file_data = triage_notes(
        notes=notes,
        min_duration=min_duration,
        max_candidates=max_candidates,
    )

    # Write full triage data to file
    triage_dir = ws.job_triage_dir(ctx.job_name, ctx.stem, ctx.preset)
    triage_path = triage_dir / "triage.json"
    triage_path.write_text(file_data.model_dump_json(indent=2))

    # Build summary
    top_summary = "no candidates"
    if file_data.candidates:
        top = file_data.candidates[0]
        name = _midi_to_name(top.note.pitch_midi)
        top_summary = f"{name} at {top.start_time:.1f}s (score {top.score:.2f})"

    return NoteTriageResult(
        triage_path=str(triage_path),
        candidate_count=len(file_data.candidates),
        top_candidate_summary=top_summary,
    ).model_dump_json(indent=2)
```

- [ ] **Step 3: Update triage unit tests**

In `tests/test_note_triage.py`, update the import and return type references. The `triage_notes` function now returns `NoteTriageFileData`:

Change the import at the top:

```python
from audio_analysis_mcp.analysis.note_triage import triage_notes
from audio_analysis_mcp.schemas import NoteEvent
```

No other changes needed — the tests access `.polyphony_profile` and `.candidates` which exist on both old `NoteTriageResult` and new `NoteTriageFileData`.

- [ ] **Step 4: Run all tests**

Run: `uv run pytest tests/test_note_triage.py -v`
Expected: all 8+ tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/audio_analysis_mcp/analysis/note_triage.py src/audio_analysis_mcp/tools/note_triage.py
git commit -m "feat: note_triage writes results to file, returns path and summary"
```

---

### Task 7: Update `note_isolate` tool

**Files:**
- Modify: `src/audio_analysis_mcp/tools/note_isolate.py`

- [ ] **Step 1: Update `note_isolate` to use job-centric paths with human-readable names**

Replace the full content of `src/audio_analysis_mcp/tools/note_isolate.py`:

```python
import soundfile as sf

from audio_analysis_mcp.server import mcp, get_workspace
from audio_analysis_mcp.workspace import resolve_job_context
from audio_analysis_mcp.analysis.note_isolation import isolate_note
from audio_analysis_mcp.schemas import NoteIsolateResult


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _midi_to_name(midi: int) -> str:
    return NOTE_NAMES[midi % 12] + str(midi // 12 - 1)


@mcp.tool()
def note_isolate(
    audio_path: str,
    start_time: float,
    end_time: float,
    start_freq: float,
    end_freq: float,
    pitch_midi: int | None = None,
) -> str:
    """Isolate a sound from audio within a time-frequency box using STFT masking.

    Input must be a stem file inside a job folder.
    pitch_midi is optional — used for human-readable output filenames.
    """
    if start_time < 0:
        raise ValueError(f"start_time must be >= 0, got {start_time}")
    if end_time <= start_time:
        raise ValueError(f"end_time ({end_time}) must be > start_time ({start_time})")
    if start_freq < 0:
        raise ValueError(f"start_freq must be >= 0, got {start_freq}")
    if end_freq <= start_freq:
        raise ValueError(f"end_freq ({end_freq}) must be > start_freq ({start_freq})")

    ws = get_workspace()
    ctx = resolve_job_context(audio_path, ws)
    if ctx.stem is None or ctx.preset is None:
        raise ValueError(
            f"Expected a stem file (jobs/<job>/stems/<preset>/<stem>.wav), got: {audio_path}"
        )

    y_isolated, sr = isolate_note(
        audio_path=audio_path,
        start_time=start_time,
        end_time=end_time,
        start_freq=start_freq,
        end_freq=end_freq,
    )

    out_dir = ws.job_isolated_notes_dir(ctx.job_name, ctx.stem, ctx.preset)
    # Count existing files to get next index
    existing = list(out_dir.glob("note_*.wav"))
    idx = len(existing) + 1
    # Build human-readable filename
    note_label = _midi_to_name(pitch_midi) if pitch_midi is not None else "unk"
    out_path = out_dir / f"note_{idx:03d}_{note_label}_{start_time:.1f}s.wav"

    sf.write(str(out_path), y_isolated, sr, subtype="PCM_16")
    duration = len(y_isolated) / sr

    return NoteIsolateResult(
        audio_path=str(out_path),
        duration_seconds=round(duration, 3),
    ).model_dump_json(indent=2)
```

- [ ] **Step 2: Update `test_note_isolate_e2e` in `tests/test_mcp_tools.py`**

Replace the existing `test_note_isolate_e2e`:

```python
def test_note_isolate_e2e(sine_440_wav: Path, tmp_path: Path):
    from audio_analysis_mcp.tools.note_isolate import note_isolate

    # Set up a job folder with a stem file
    stem_dir = tmp_path / "workspace" / "jobs" / "test-song" / "stems" / "fast"
    stem_dir.mkdir(parents=True)
    stem_file = stem_dir / "bass.wav"
    import shutil
    shutil.copy(sine_440_wav, stem_file)

    result = json.loads(
        note_isolate(
            audio_path=str(stem_file),
            start_time=0.0,
            end_time=0.5,
            start_freq=400.0,
            end_freq=500.0,
            pitch_midi=69,
        )
    )
    assert Path(result["audio_path"]).exists()
    assert result["duration_seconds"] > 0
    assert "note_001_A4_0.0s.wav" in result["audio_path"]
    assert "test-song/isolated_notes/bass_fast" in result["audio_path"]
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_mcp_tools.py::test_note_isolate_e2e -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/audio_analysis_mcp/tools/note_isolate.py tests/test_mcp_tools.py
git commit -m "feat: note_isolate writes to job folder with human-readable filenames"
```

---

### Task 8: Full test suite and type-check

**Files:** None (verification only)

- [ ] **Step 1: Run entire test suite**

Run: `uv run pytest -v -m "not slow"`
Expected: all tests PASS.

- [ ] **Step 2: Run mypy on entire source**

Run: `uv run mypy src/`
Expected: Success, no errors.

- [ ] **Step 3: Commit if any fixups needed**

```bash
git add -u
git commit -m "fix: resolve type errors from workspace refactor"
```

---

### Task 9: Update pipeline plan

**Files:**
- Modify: `audio-pipeline-plan.md`

- [ ] **Step 1: Update the workspace section in `audio-pipeline-plan.md`**

Replace the workspace section (around line 361) with:

```markdown
## Workspace

```
~/.audio-analysis-mcp/
  workspace/
    jobs/
      <sanitized-song-name>/
        source.wav                              # Imported/normalized audio
        stems/
          <preset>/                             # e.g. htdemucs_6s_accurate
            bass.wav, drums.wav, vocals.wav, other.wav, ...
        transcriptions/
          <stem>_<preset>/                      # e.g. bass_medium
            transcription.mid                   # MIDI for DAW
            transcription.json                  # Note events for triage
        triage/
          <stem>_<preset>/
            triage.json                         # Polyphony profile + candidates
        isolated_notes/
          <stem>_<preset>/
            note_001_F1_94.3s.wav               # Human-readable filenames
    spectrograms/          # Flat (used by spectrum_analyze)
    rendered/              # Flat (used by audio_render)
```
```

- [ ] **Step 2: Commit**

```bash
git add audio-pipeline-plan.md
git commit -m "docs: update workspace section in pipeline plan for job-centric structure"
```
