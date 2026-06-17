from audio_analysis_mcp.server import mcp


def _register_tools() -> None:
    """Import every tool module so its ``@mcp.tool()`` decorator registers it."""
    import audio_analysis_mcp.tools.import_audio  # noqa: F401
    import audio_analysis_mcp.tools.stem_separate  # noqa: F401
    import audio_analysis_mcp.tools.audio_render  # noqa: F401
    import audio_analysis_mcp.tools.spectrum_analyze  # noqa: F401
    import audio_analysis_mcp.tools.audio_compare  # noqa: F401
    import audio_analysis_mcp.tools.note_transcribe  # noqa: F401
    import audio_analysis_mcp.tools.note_triage  # noqa: F401
    import audio_analysis_mcp.tools.note_isolate  # noqa: F401
    import audio_analysis_mcp.tools.amplitude_analyze  # noqa: F401
    import audio_analysis_mcp.tools.structure_analyze  # noqa: F401


def main() -> None:
    """Console-script + ``python -m`` entry point: register tools, run stdio server."""
    _register_tools()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
