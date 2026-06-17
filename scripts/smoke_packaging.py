# scripts/smoke_packaging.py
"""End-to-end packaging smoke test.

Builds nothing itself — expects a pre-built wheel in dist/. Spawns the
installed server in an isolated uvx env and drives an MCP stdio session:
the server must start (proving the turnkey install + Python 3.11 + no
import-time failure from missing optional deps), list all tools, run a
light tool, and degrade gracefully for SongFormer (and PortAudio, on a
headless runner via --expect-no-portaudio).

Usage:
    python scripts/smoke_packaging.py [--expect-no-portaudio]
"""
from __future__ import annotations

import asyncio
import glob
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

EXPECTED_TOOLS = {
    "import_audio", "stem_separate", "audio_list_devices", "audio_render",
    "spectrum_analyze", "audio_compare", "note_transcribe", "note_triage",
    "note_isolate", "amplitude_analyze", "structure_analyze",
}


def _make_sine_wav(path: Path) -> None:
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sf.write(str(path), 0.5 * np.sin(2 * np.pi * 440 * t), sr, subtype="PCM_16")


def _text(result) -> str:
    return " ".join(getattr(c, "text", "") for c in result.content)


async def run(wheel: str, expect_no_portaudio: bool) -> None:
    params = StdioServerParameters(command="uvx", args=["--from", wheel, "audio-analysis-mcp"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            assert EXPECTED_TOOLS <= names, f"missing tools: {EXPECTED_TOOLS - names}"

            with tempfile.TemporaryDirectory() as d:
                wav = Path(d) / "sine.wav"
                _make_sine_wav(wav)

                imported = await session.call_tool("import_audio", {"file_path": str(wav)})
                src = json.loads(_text(imported))["audio_path"]

                spec = await session.call_tool(
                    "spectrum_analyze", {"audio_path": src, "duration": 1.0}
                )
                f0 = json.loads(_text(spec))["spectral_features"]["fundamental_hz"]
                assert f0 is not None and abs(f0 - 440) < 10, f"bad fundamental: {f0}"

                structure = await session.call_tool("structure_analyze", {"audio_path": src})
                assert structure.isError and "SongFormer" in _text(structure), _text(structure)

                if expect_no_portaudio:
                    render = await session.call_tool("audio_render", {"duration": 0.1})
                    assert render.isError and "PortAudio" in _text(render), _text(render)

    print("packaging smoke: OK")


def main() -> int:
    expect_no_portaudio = "--expect-no-portaudio" in sys.argv[1:]
    matches = sorted(glob.glob("dist/*.whl"))
    if not matches:
        print("no wheel in dist/ — run `uv build --wheel` first", file=sys.stderr)
        return 1
    asyncio.run(run(matches[-1], expect_no_portaudio))
    return 0


if __name__ == "__main__":
    sys.exit(main())
