from pathlib import Path
from audio_analysis_mcp.server import mcp, get_workspace
from audio_analysis_mcp.workspace import sanitize_job_name
from audio_analysis_mcp.audio.normalize import normalize_audio
from audio_analysis_mcp.schemas import ImportAudioResult


def import_audio_impl(
    file_path: str,
    start_time: float | None = None,
    duration: float | None = None,
) -> ImportAudioResult:
    """Import a local audio file. Normalize to 44.1kHz 16-bit mono WAV.

    Shared between the MCP tool wrapper (returns JSON) and the HTTP service
    (returns the model directly).
    """
    ws = get_workspace()
    job_name = sanitize_job_name(Path(file_path).name)
    job_dir = ws.job_dir(job_name)
    output_path = job_dir / "source.wav"

    dur, ch = normalize_audio(file_path, str(output_path), start_time, duration)

    return ImportAudioResult(
        audio_path=str(output_path),
        job_name=job_name,
        sample_rate=44100,
        duration_seconds=dur,
        channels=ch,
    )


@mcp.tool()
def import_audio(
    file_path: str,
    start_time: float | None = None,
    duration: float | None = None,
) -> str:
    """Import a local audio file. Normalize to 44.1kHz 16-bit mono WAV."""
    return import_audio_impl(file_path, start_time, duration).model_dump_json(indent=2)
