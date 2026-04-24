from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import uuid

from basic_pitch.inference import predict

from audio_analysis_mcp.schemas import NoteEvent


def transcribe_audio(
    audio_path: str,
    output_dir: str,
) -> tuple[str, str, list[NoteEvent]]:
    """Run Basic Pitch on audio file.

    Returns (midi_path, notes_json_path, note_events).
    """
    # basic_pitch.predict writes debug output to stdout which corrupts
    # the MCP stdio transport — suppress it
    with redirect_stdout(io.StringIO()):
        model_output, midi_data, note_events = predict(audio_path)

    # Save files
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    file_id = uuid.uuid4().hex[:8]
    midi_path = out / f"transcription_{file_id}.mid"
    notes_path = out / f"transcription_{file_id}.json"
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

    # Save notes JSON sidecar
    notes_path.write_text(
        json.dumps([n.model_dump() for n in notes], indent=2)
    )

    return str(midi_path), str(notes_path), notes
