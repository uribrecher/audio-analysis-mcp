import uuid
from pathlib import Path
from audio_analysis_mcp.server import mcp, get_workspace
from audio_analysis_mcp.audio.normalize import normalize_audio
from audio_analysis_mcp.schemas import ImportAudioResult


@mcp.tool()
def import_audio(
    file_path: str,
    start_time: float | None = None,
    duration: float | None = None,
) -> str:
    """Import a local audio file. Normalize to 44.1kHz 16-bit mono WAV."""
    ws = get_workspace()
    stem = Path(file_path).stem
    output_path = ws.imported / f"{stem}_{uuid.uuid4().hex[:8]}.wav"

    dur, ch = normalize_audio(file_path, str(output_path), start_time, duration)

    return ImportAudioResult(
        audio_path=str(output_path),
        sample_rate=44100,
        duration_seconds=dur,
        channels=ch,
    ).model_dump_json(indent=2)
