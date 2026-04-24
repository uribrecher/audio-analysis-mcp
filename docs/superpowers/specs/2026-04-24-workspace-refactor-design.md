# Workspace Refactor — Job-Centric Folder Structure

## Goal

Replace the flat, disconnected workspace layout with a job-centric structure where all pipeline outputs for a source audio file live together under one human-readable folder. Also fix `note_triage` to write results to a file and return a path, matching the pattern established by `note_transcribe`.

## Current Problem

- Stems live under a content hash (`46f99f29a338afad/`) — no human can tell what song this is
- Transcriptions use random UUIDs — no link back to which stem produced them
- Triage returns full JSON payload (64K+) instead of a file path
- A human browsing the workspace cannot follow the pipeline from source → stems → transcription → triage → isolated notes

## New Folder Structure

```
workspace/
  jobs/
    <sanitized-song-name>/
      source.wav
      stems/
        <preset>/
          bass.wav
          drums.wav
          vocals.wav
          other.wav
          guitar.wav
          piano.wav
      transcriptions/
        <stem>_<preset>/
          transcription.mid
          transcription.json
      triage/
        <stem>_<preset>/
          triage.json
      isolated_notes/
        <stem>_<preset>/
          note_001_<NoteName>_<time>s.wav
          note_002_<NoteName>_<time>s.wav
  # Old flat directories (imported/, stems/, spectrograms/, etc.) remain untouched
```

### Job Name Derivation

Source filename is sanitized to a filesystem-safe, human-readable slug:
- Lowercase
- Replace spaces and special characters with hyphens
- Strip parentheses, brackets, quotes
- Collapse multiple consecutive hyphens
- Trim leading/trailing hyphens

Example: `Smooth Criminal (Radio Edit).wav` → `smooth-criminal-radio-edit`

### Job Context Resolution

Given any file path within a job folder, the workspace can resolve:
- **job name** — the folder name under `jobs/`
- **stem name** — e.g. `bass`, `other`
- **preset name** — e.g. `htdemucs_6s_accurate`, `medium`

This is done by parsing the path structure: `jobs/<job>/stems/<preset>/<stem>.wav`

## Changes Per Tool

### `import_audio`

**Before:** Copies to `workspace/imported/<name>_<hash>.wav`
**After:** Copies to `workspace/jobs/<sanitized-name>/source.wav`

Returns: `ImportAudioResult` with added `job_name: str` field.

If a job folder already exists for this name, reuse it (idempotent).

### `stem_separate`

**Before:** Writes to `workspace/stems/<content-hash>/<preset>/`
**After:** Writes to `workspace/jobs/<job>/stems/<preset>/`

The job name is resolved from the input `audio_path`:
- If the input is inside a job folder (`jobs/<job>/source.wav`), use that job name
- If the input is outside the workspace, raise an error — user must import first

Caching: keyed by `(job_name, preset)` — if `jobs/<job>/stems/<preset>/` exists and is non-empty, return cached result.

### `note_transcribe`

**Before:** Writes to `workspace/transcriptions/transcription_<uuid>.*`
**After:** Writes to `workspace/jobs/<job>/transcriptions/<stem>_<preset>/transcription.{mid,json}`

The job, stem, and preset are resolved from the input `audio_path` (a stem file inside a job folder).

No change to return schema (`NoteTranscribeResult` with `midi_path`, `notes_path`, `note_count`).

### `note_triage`

**Before:** Returns full JSON (64K+ polyphony profile + candidates)
**After:** Writes to `workspace/jobs/<job>/triage/<stem>_<preset>/triage.json`, returns path and summary

Return schema changes to:

```python
class NoteTriageResult(BaseModel):
    triage_path: str
    candidate_count: int
    top_candidates: list[CandidateNote]  # top 5 candidates with full isolation params
```

The full triage data (polyphony profile + all candidates) is in the JSON file. The `top_candidates` field returns the top 5 candidates inline for direct use by `note_isolate`.

### `note_isolate`

**Before:** Writes to `workspace/isolated_notes/isolated_<uuid>.wav`
**After:** Writes to `workspace/jobs/<job>/isolated_notes/<stem>_<preset>/note_<NNN>_<NoteName>_<time>s.wav`

The job, stem, and preset are resolved from the input `audio_path`.

Note naming: `note_001_F1_94.3s.wav` — sequential index, MIDI note name, start time. The index is based on how many isolated notes already exist in the folder.

No change to return schema (`NoteIsolateResult` with `audio_path`, `duration_seconds`).

## Workspace Module Changes

### `workspace.py`

Add:
- `sanitize_job_name(filename: str) -> str` — filename to slug
- `resolve_job_context(path: str) -> JobContext` — parse job/stem/preset from a path within the workspace
- `job_dir(job_name: str) -> Path` — `workspace/jobs/<job>/`
- `stems_dir(job_name: str, preset: str) -> Path`
- `transcriptions_dir(job_name: str, stem: str, preset: str) -> Path`
- `triage_dir(job_name: str, stem: str, preset: str) -> Path`
- `isolated_notes_dir(job_name: str, stem: str, preset: str) -> Path`

```python
@dataclass
class JobContext:
    job_name: str
    stem: str | None
    preset: str | None
```

The old flat directory properties (`imported`, `stems`, `spectrograms`, etc.) remain for backward compatibility — they are not removed.

## Schema Changes

### `ImportAudioResult` — add field

```python
class ImportAudioResult(BaseModel):
    audio_path: str
    job_name: str      # NEW
    sample_rate: int
    duration_seconds: float
    channels: int
```

### `NoteTriageResult` — replace with file-based output

```python
class NoteTriageResult(BaseModel):
    triage_path: str
    candidate_count: int
    top_candidates: list[CandidateNote]
```

The full triage data (polyphony profile + all candidates) is written to the JSON file. The `top_candidates` field returns the top 5 inline for direct use by `note_isolate`. The file contains:

```python
class NoteTriageFileData(BaseModel):
    polyphony_profile: list[PolyphonyWindow]
    candidates: list[CandidateNote]
```

### No changes to

- `NoteEvent`, `NoteTranscribeResult`, `NoteIsolateResult`, `CandidateNote`, `PolyphonyWindow` — these remain as-is

## Backward Compatibility

- Old flat workspace directories are not removed or migrated
- `spectrum_analyze` and `audio_compare` continue using their existing flat directories (`spectrograms/`) — they are not part of the job pipeline
- Tools that accept `audio_path` from outside the workspace will fail with a clear error message pointing the user to `import_audio` first

## Test Changes

- Update `conftest.py` workspace fixture to create the `jobs/` directory
- Update E2E tests for `import_audio` to check `job_name` in result
- Update E2E tests for `note_transcribe` to use a job-structured path
- Add tests for `sanitize_job_name` (edge cases: unicode, dots, multiple spaces)
- Add tests for `resolve_job_context` (valid paths, paths outside workspace)
- Update `note_triage` tests to check file output instead of inline JSON
