from pathlib import Path

from pydantic import TypeAdapter

from audio_analysis_mcp.server import mcp, get_workspace
from audio_analysis_mcp.workspace import resolve_job_context
from audio_analysis_mcp.analysis.note_triage import triage_notes
from audio_analysis_mcp.schemas import NoteEvent, NoteTriageResult


@mcp.tool()
def note_triage(
    audio_path: str,
    notes_path: str,
    min_duration: float = 0.5,
    max_candidates: int = 10,
    start_time: float | None = None,
    end_time: float | None = None,
) -> str:
    """Triage notes into ranked clusters (single / chord) for downstream analysis.

    notes_path must be the JSON file from note_transcribe.
    Optional start_time/end_time filter notes to a song region.
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
        start_time=start_time,
        end_time=end_time,
    )

    triage_dir = ws.job_triage_dir(ctx.job_name, ctx.stem, ctx.preset)
    triage_path = triage_dir / "triage.json"
    triage_path.write_text(file_data.model_dump_json(indent=2))

    return NoteTriageResult(
        triage_path=str(triage_path),
        candidate_count=len(file_data.candidates),
        top_candidates=file_data.candidates[:5],
    ).model_dump_json(indent=2)
