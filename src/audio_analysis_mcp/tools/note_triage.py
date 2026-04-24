from pathlib import Path

from pydantic import TypeAdapter

from audio_analysis_mcp.server import mcp
from audio_analysis_mcp.analysis.note_triage import triage_notes
from audio_analysis_mcp.schemas import NoteEvent


@mcp.tool()
def note_triage(
    audio_path: str,
    notes_path: str,
    min_duration: float = 0.5,
    max_candidates: int = 10,
) -> str:
    """Analyze transcription and select best candidate notes for isolation."""
    adapter = TypeAdapter(list[NoteEvent])
    notes_json = Path(notes_path).read_text()
    notes = adapter.validate_json(notes_json)
    result = triage_notes(
        notes=notes,
        min_duration=min_duration,
        max_candidates=max_candidates,
    )
    return result.model_dump_json(indent=2)
