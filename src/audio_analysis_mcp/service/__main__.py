"""Run the service:

    uv run python -m audio_analysis_mcp.service

Host/port via ``AUDIO_ANALYSIS_SERVICE_HOST`` (default ``127.0.0.1``) and
``AUDIO_ANALYSIS_SERVICE_PORT`` (default ``8765``).
"""
from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.environ.get("AUDIO_ANALYSIS_SERVICE_HOST", "127.0.0.1")
    port = int(os.environ.get("AUDIO_ANALYSIS_SERVICE_PORT", "8765"))
    # Single worker is mandatory — model singletons (SongFormer, Demucs) live
    # in process memory and we want one shared copy, not N replicas.
    uvicorn.run(
        "audio_analysis_mcp.service.app:app",
        host=host,
        port=port,
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
