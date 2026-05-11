from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP
from audio_analysis_mcp.workspace import Workspace

if TYPE_CHECKING:
    from songformer import SongFormerPipeline

mcp = FastMCP("audio-analysis-mcp")

_workspace: Workspace | None = None
_structure_pipeline: "SongFormerPipeline | None" = None


def get_workspace() -> Workspace:
    global _workspace
    if _workspace is None:
        _workspace = Workspace()
    return _workspace


def get_structure_pipeline() -> "SongFormerPipeline":
    global _structure_pipeline
    if _structure_pipeline is None:
        from songformer import SongFormerPipeline

        _structure_pipeline = SongFormerPipeline.from_pretrained()
    return _structure_pipeline
