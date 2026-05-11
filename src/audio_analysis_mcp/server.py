from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP
from audio_analysis_mcp.workspace import Workspace

if TYPE_CHECKING:
    from songformer import SongFormerPipeline

mcp = FastMCP("audio-analysis-mcp")

_workspace: Workspace | None = None
_structure_pipeline: "SongFormerPipeline | None" = None
_demucs_models: dict[str, Any] = {}


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


def get_demucs_model(model_name: str) -> Any:
    """Cached Demucs model lookup keyed by model name (e.g. ``htdemucs_6s``).

    First call loads the weights (~10-20s); subsequent calls return the
    in-memory model. The HTTP service uses this so repeated stem requests
    don't pay the load cost every time. The MCP tool consumes the same
    singleton — no behavior change there beyond faster second runs.
    """
    if model_name not in _demucs_models:
        from demucs.pretrained import get_model

        model = get_model(model_name)
        model.eval()
        _demucs_models[model_name] = model
    return _demucs_models[model_name]
