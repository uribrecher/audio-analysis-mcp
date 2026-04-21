from mcp.server.fastmcp import FastMCP
from audio_analysis_mcp.workspace import Workspace

mcp = FastMCP("audio-analysis-mcp")

_workspace: Workspace | None = None


def get_workspace() -> Workspace:
    global _workspace
    if _workspace is None:
        _workspace = Workspace()
    return _workspace
