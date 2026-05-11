from audio_analysis_mcp.analysis.structure_analysis import analyze_structure
from audio_analysis_mcp.server import get_structure_pipeline, get_workspace, mcp
from audio_analysis_mcp.workspace import resolve_job_context


@mcp.tool()
def structure_analyze(audio_path: str) -> str:
    """Detect song structure — identifies sections like intro, verse, chorus,
    bridge, pre-chorus, outro with timestamps. Input must be a source audio
    file inside a job folder.
    """
    ws = get_workspace()
    ctx = resolve_job_context(audio_path, ws)
    output_dir = str(ws.job_song_structure_dir(ctx.job_name))
    pipeline = get_structure_pipeline()
    result = analyze_structure(
        audio_path=audio_path,
        output_dir=output_dir,
        pipeline=pipeline,
    )
    return result.model_dump_json(indent=2)
