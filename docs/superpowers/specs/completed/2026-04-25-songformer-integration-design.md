# Structure Analysis MCP Tool Design

Add a `structure_analyze` tool to audio-analysis-mcp that detects song sections (intro, verse, chorus, bridge, etc.) with timestamps.

## Context

The audio-analysis-mcp pipeline needs song structure awareness. This feeds into downstream tools — `note_triage` can focus on specific sections, and the agent can reason about song structure when recreating sounds.

This tool uses the `songformer` library (see `2026-04-25-songformer-library-design.md` for the library design). The library is assumed to be installed as a path dependency and provides:

```python
from songformer import SongFormerPipeline, Segment, AnalysisResult

pipeline = SongFormerPipeline.from_pretrained(device="mps")
result = pipeline.analyze("song.mp3")
# result.segments: list[Segment]  — each has .start, .end, .label
# result.duration: float
```

## Tool: `structure_analyze`

New file: `src/audio_analysis_mcp/tools/structure_analyze.py`

```python
@mcp.tool()
def structure_analyze(audio_path: str) -> str:
    """Detect song structure — identifies sections like intro, verse,
    chorus, bridge, pre-chorus, outro with timestamps."""
    ctx = resolve_job_context(audio_path, ws)
    structure_dir = ws.job_song_structure_dir(ctx.job_name)

    # Cache: return if structure.json already exists
    # Otherwise: run pipeline.analyze(), save to structure.json

    return StructureAnalyzeResult(...).model_dump_json(indent=2)
```

## Schema

Added to `schemas.py`:

```python
class StructureSegment(BaseModel):
    start: float
    end: float
    label: str

class StructureAnalyzeResult(BaseModel):
    structure_path: str
    segments: list[StructureSegment]
    duration: float
    cached: bool
```

## Workspace

Add `job_song_structure_dir(job_name)` to the `Workspace` class, returning `jobs/<job>/song_structure/`.

Results cached at `jobs/<job>/song_structure/structure.json`.

## Model loading

The `SongFormerPipeline` is loaded at MCP server startup (in `__main__.py` or a dedicated init module). Since weights are pre-downloaded via `songformer-download`, `from_pretrained()` only loads from disk to memory — this takes seconds, not minutes.

## Dependency

In `audio-analysis-mcp/pyproject.toml`:

```toml
dependencies = [
    "songformer @ file:///${PROJECT_ROOT}/../SongFormer",
    ...existing deps...
]
```

## Directory layout

```
jobs/<job>/
├── source.wav
├── song_structure/
│   └── structure.json          ← NEW
├── stems/<preset>/
├── transcriptions/<stem>_<preset>/
├── triage/<stem>_<preset>/
└── isolated_notes/<stem>_<preset>/
```