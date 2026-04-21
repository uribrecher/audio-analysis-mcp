from audio_analysis_mcp.server import mcp

import audio_analysis_mcp.tools.import_audio  # noqa: F401

mcp.run(transport="stdio")
