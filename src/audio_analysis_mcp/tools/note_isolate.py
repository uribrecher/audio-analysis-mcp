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
