import uuid
import soundfile as sf

from audio_analysis_mcp.server import mcp, get_workspace
from audio_analysis_mcp.analysis.note_isolation import isolate_note
from audio_analysis_mcp.schemas import NoteIsolateResult


@mcp.tool()
def note_isolate(
    audio_path: str,
    start_time: float,
    end_time: float,
    start_freq: float,
    end_freq: float,
) -> str:
    """Isolate a sound from audio within a time-frequency box using STFT masking."""
    if start_time < 0:
        raise ValueError(f"start_time must be >= 0, got {start_time}")
    if end_time <= start_time:
        raise ValueError(f"end_time ({end_time}) must be > start_time ({start_time})")
    if start_freq < 0:
        raise ValueError(f"start_freq must be >= 0, got {start_freq}")
    if end_freq <= start_freq:
        raise ValueError(f"end_freq ({end_freq}) must be > start_freq ({start_freq})")

    ws = get_workspace()
    y_isolated, sr = isolate_note(
        audio_path=audio_path,
        start_time=start_time,
        end_time=end_time,
        start_freq=start_freq,
        end_freq=end_freq,
    )
    out_path = ws.isolated_notes / f"isolated_{uuid.uuid4().hex[:8]}.wav"
    sf.write(str(out_path), y_isolated, sr, subtype="PCM_16")
    duration = len(y_isolated) / sr

    return NoteIsolateResult(
        audio_path=str(out_path),
        duration_seconds=round(duration, 3),
    ).model_dump_json(indent=2)
