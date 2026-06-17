import gc
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
        try:
            from songformer import SongFormerPipeline
        except ImportError as exc:
            raise RuntimeError(
                "structure_analyze requires SongFormer, which is not installed. "
                "Enable it with: uvx --with "
                "'songformer @ git+https://github.com/uribrecher/SongFormer.git@v0.2.0' "
                "audio-analysis-mcp . Note: this pulls MuQ model weights licensed "
                "CC-BY-NC-4.0 (non-commercial use only)."
            ) from exc

        _structure_pipeline = SongFormerPipeline.from_pretrained()
    return _structure_pipeline


def get_demucs_model(model_name: str) -> Any:
    """Cached Demucs model lookup keyed by model name (e.g. ``htdemucs_6s``).

    First call loads the weights (~10-20s); subsequent calls return the
    in-memory model. The HTTP service drops the singleton after each
    request (see ``release_demucs_model``) to keep idle RSS at zero — the
    reload cost is acceptable against the 5-15GB the model holds when
    parked. The MCP stdio tool path keeps the cache across calls.
    """
    if model_name not in _demucs_models:
        from demucs.pretrained import get_model

        model = get_model(model_name)
        model.eval()
        _demucs_models[model_name] = model
    return _demucs_models[model_name]


def _drain_torch_allocator_caches() -> None:
    """Best-effort cleanup of PyTorch's per-device allocator caches.

    ``gc.collect`` runs first to drop any lingering Python refs; without it
    the empty_cache calls only release tensors whose Python wrappers are
    already gone. Wrapped in try/except so importing torch on a barebones
    env that doesn't ship the MPS/CUDA modules doesn't crash the service.
    """
    gc.collect()
    try:
        import torch

        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        # Cache-drain is opportunistic — never block a request on it.
        pass


def release_structure_pipeline() -> None:
    """Drop the cached SongFormer pipeline and reclaim its RSS.

    Used by the HTTP service's ``/jobs/structure`` handler after each
    request. Pipeline weights alone are ~7-8GB at fp32; combined with
    intermediate activations PyTorch's allocator holds, the steady-state
    cost is unacceptable for a Mac-class machine. Callers MUST hold
    ``service.app._structure_lock`` so a concurrent request doesn't see
    the singleton mid-teardown.
    """
    global _structure_pipeline
    _structure_pipeline = None
    _drain_torch_allocator_caches()


def release_demucs_model(model_name: str) -> None:
    """Drop the cached Demucs model for ``model_name``. Service-only.

    Callers MUST hold ``service.app._demucs_lock`` for the same reason
    as ``release_structure_pipeline`` above. No-op if the model wasn't
    cached (e.g. a cache-hit path that never touched ``get_demucs_model``).
    """
    _demucs_models.pop(model_name, None)
    _drain_torch_allocator_caches()
