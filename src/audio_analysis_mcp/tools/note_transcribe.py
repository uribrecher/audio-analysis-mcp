from audio_analysis_mcp.server import mcp, get_workspace
from audio_analysis_mcp.analysis.transcription import transcribe_audio
from audio_analysis_mcp.schemas import NoteTranscribeResult


@mcp.tool()
def note_transcribe(audio_path: str) -> str:
    """Transcribe polyphonic audio to MIDI note events using Basic Pitch."""
    ws = get_workspace()
    midi_path, notes_path, notes = transcribe_audio(audio_path, output_dir=str(ws.transcriptions))
    return NoteTranscribeResult(
        midi_path=midi_path,
        notes_path=notes_path,
        note_count=len(notes),
    ).model_dump_json(indent=2)
