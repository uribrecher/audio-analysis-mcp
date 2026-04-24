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
