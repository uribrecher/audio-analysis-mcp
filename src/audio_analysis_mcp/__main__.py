from audio_analysis_mcp.server import mcp

# Tool imports are added as tools are implemented (Tasks 3-7).
# Each import triggers @mcp.tool() registration.

mcp.run(transport="stdio")
