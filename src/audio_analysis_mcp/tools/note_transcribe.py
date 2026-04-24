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
