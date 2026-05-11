"""HTTP+SSE service exposing the slow audio-analysis operations.

Mirrors the MCP tools but streams progress events while running, so clients
like the keyboards-mcp Electron mock-runner can show real progress instead
of staring at an opaque tool call.

Run via:
    uv run python -m audio_analysis_mcp.service
"""

from audio_analysis_mcp.service.app import app

__all__ = ["app"]
