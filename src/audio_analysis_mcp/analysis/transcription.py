from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from typing import Callable, TypeAlias

from basic_pitch.inference import predict

from audio_analysis_mcp.schemas import NoteEvent


ProgressFn: TypeAlias = Callable[[str, float, "str | None"], None]
"""Caller-side progress callback. Stages: ``cache_hit | load_audio | predict |
write | done``. ``detail`` is currently always ``None``; the parameter exists
so the type matches the other tool ProgressFn signatures."""


def transcribe_audio(
    audio_path: str,
    output_dir: str,
    progress: ProgressFn | None = None,
) -> tuple[str, str, list[NoteEvent], bool]:
    """Run Basic Pitch on audio file.

    Returns (midi_path, notes_json_path, note_events, cached). A cache hit
    is detected when ``<output_dir>/transcription.mid`` already exists —
    the sidecar JSON is re-loaded from disk so the contract stays the
    same regardless of whether predict() ran.
    """
    def emit(stage: str, fraction: float, detail: "str | None" = None) -> None:
        if progress is None:
            return
        try:
            progress(stage, max(0.0, min(1.0, fraction)), detail)
        except Exception:
            # Don't let a buggy sink kill the run.
            pass

    out = Path(output_dir)
    midi_path = out / "transcription.mid"
    notes_path = out / "transcription.json"

    notes: list[NoteEvent]

    # Cache hit: MIDI already on disk. Re-hydrate the sidecar JSON so the
    # return shape is identical to a fresh run.
    if midi_path.exists() and notes_path.exists():
        emit("cache_hit", 1.0)
        try:
            raw = json.loads(notes_path.read_text())
            notes = [NoteEvent(**n) for n in raw]
        except Exception:
            notes = []
        return str(midi_path), str(notes_path), notes, True

    out.mkdir(parents=True, exist_ok=True)
    emit("load_audio", 0.05)

    # basic_pitch.predict writes debug output to stdout which corrupts
    # the MCP stdio transport — suppress it. predict() is opaque (no
    # internal progress hook), so we emit at coarse boundaries around it.
    emit("predict", 0.10)
    with redirect_stdout(io.StringIO()):
        model_output, midi_data, note_events = predict(audio_path)
    emit("predict", 0.90)

    emit("write", 0.95)
    midi_data.write(str(midi_path))

    notes = []
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

    notes_path.write_text(
        json.dumps([n.model_dump() for n in notes], indent=2)
    )

    emit("done", 1.0)
    return str(midi_path), str(notes_path), notes, False
