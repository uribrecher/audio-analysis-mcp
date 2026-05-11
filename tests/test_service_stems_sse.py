"""SSE smoke for POST /jobs/stems. Stubs `stem_separate_impl` so the test
runs in ~0.5s without loading Demucs weights — we only care that the SSE
wiring correctly serializes progress events emitted from a worker thread
and closes with a final `result` event.
"""
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

import audio_analysis_mcp.server as srv
from audio_analysis_mcp.schemas import StemFile, StemSeparateResult
from audio_analysis_mcp.service.app import app
from audio_analysis_mcp.workspace import Workspace


@pytest.fixture(autouse=True)
def _isolated_workspace(tmp_path: Path):
    old = srv._workspace
    srv._workspace = Workspace(tmp_path / "ws")
    yield
    srv._workspace = old


def _fake_stem_impl(audio_path, stems_dir, preset_name="medium", progress=None):
    """Stand-in for the real Demucs run. Emits a few progress ticks then
    returns a fabricated StemSeparateResult."""
    if progress is not None:
        progress("load_model", 0.05, None)
        progress("run", 0.30, "run 1/2")
        progress("run", 0.65, "run 2/2")
        progress("write", 0.97, None)
        progress("done", 1.0, None)
    return StemSeparateResult(
        stems=[StemFile(stem="vocals", path=str(stems_dir / "vocals.wav"))],
        model="htdemucs_6s",
        preset=preset_name,
        cached=False,
    )


def _parse_sse(body: str) -> list[dict]:
    """Tiny SSE parser: extracts {event, data-as-dict} pairs in order."""
    events: list[dict] = []
    current: dict = {}
    for line in body.splitlines():
        if line.startswith("event:"):
            current["event"] = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            current["data"] = json.loads(line.split(":", 1)[1].strip())
        elif line.strip() == "":
            if current:
                events.append(current)
                current = {}
    if current:
        events.append(current)
    return events


@pytest.mark.asyncio
async def test_jobs_stems_streams_progress_then_result(sine_440_wav: Path) -> None:
    # Stage a source.wav under jobs/<job>/ so resolve_job_context accepts it.
    ws = srv.get_workspace()
    source = ws.job_dir("test-song") / "source.wav"
    shutil.copy(sine_440_wav, source)

    transport = ASGITransport(app=app)
    with patch(
        "audio_analysis_mcp.service.app.stem_separate_impl",
        side_effect=_fake_stem_impl,
    ):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post(
                "/jobs/stems",
                json={"audio_path": str(source), "preset": "fast"},
            )
            body = r.text

    assert r.status_code == 200
    events = _parse_sse(body)
    # Expect: progress... progress... progress... progress... progress... result
    progresses = [e for e in events if e["event"] == "progress"]
    results = [e for e in events if e["event"] == "result"]
    assert len(progresses) == 5
    assert progresses[0]["data"]["stage"] == "load_model"
    assert progresses[-1]["data"]["stage"] == "done"
    # Monotonic
    fractions = [p["data"]["fraction"] for p in progresses]
    assert fractions == sorted(fractions)
    assert len(results) == 1
    assert results[0]["data"]["model"] == "htdemucs_6s"
    assert results[0]["data"]["preset"] == "fast"


@pytest.mark.asyncio
async def test_jobs_stems_rejects_invalid_preset() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/jobs/stems", json={"audio_path": "/tmp/whatever.wav", "preset": "nuclear"}
        )
    # Pydantic Literal rejects with 422; the manual check raises 400 — either is fine.
    assert r.status_code in (400, 422)
