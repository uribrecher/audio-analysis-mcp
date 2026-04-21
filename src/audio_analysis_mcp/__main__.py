from audio_analysis_mcp.server import mcp

import audio_analysis_mcp.tools.import_audio  # noqa: F401
import audio_analysis_mcp.tools.stem_separate  # noqa: F401

mcp.run(transport="stdio")
