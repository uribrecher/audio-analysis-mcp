from audio_analysis_mcp.server import mcp
from audio_analysis_mcp.analysis.comparison import compare_audio


@mcp.tool()
def audio_compare(target_path: str, rendered_path: str) -> str:
    """Compare target audio vs. synthesized attempt.

    Returns mel spectrogram distance, CLAP similarity (when available),
    and per-band energy diffs.
    """
    result = compare_audio(target_path, rendered_path)
    return result.model_dump_json(indent=2)
