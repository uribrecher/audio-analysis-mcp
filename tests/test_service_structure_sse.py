"""SSE smoke for POST /jobs/structure. Mocks the SongFormer pipeline so
we don't pay the 4s+ cold load."""
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

import audio_analysis_mcp.server as srv
from audio_analysis_mcp.service.app import app
from audio_analysis_mcp.workspace import Workspace


@pytest.fixture(autouse=True)
def _isolated_workspace(tmp_path: Path):
    old = srv._workspace
    srv._workspace = Workspace(tmp_path / "ws")
    yield
    srv._workspace = old


def _parse_sse(body: str) -> list[dict]:
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


def _fake_pipeline_factory():
    """Builds a MagicMock pipeline whose .analyze() emits progress and returns
    a fabricated AnalysisResult."""
    pipeline = MagicMock()

    def analyze(audio_path, progress=None):
        if progress is not None:
            progress("load_audio", 0.05)
            progress("encode", 0.25)
            progress("encode", 0.50)
            progress("infer", 0.85)
            progress("postprocess", 0.97)
            progress("done", 1.0)
        result = MagicMock()
        result.duration = 1.0
        seg = MagicMock()
        seg.start = 0.0
        seg.end = 1.0
        seg.label = "intro"
        result.segments = [seg]
        return result

    pipeline.analyze.side_effect = analyze
    return pipeline


@pytest.mark.asyncio
async def test_jobs_structure_streams_progress_then_result(sine_440_wav: Path) -> None:
    ws = srv.get_workspace()
    source = ws.job_dir("test-song") / "source.wav"
    shutil.copy(sine_440_wav, source)

    pipeline = _fake_pipeline_factory()
    saved = srv._structure_pipeline
    srv._structure_pipeline = pipeline
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post(
                "/jobs/structure", json={"audio_path": str(source)}
            )
            body = r.text
    finally:
        srv._structure_pipeline = saved

    assert r.status_code == 200
    events = _parse_sse(body)
    progresses = [e for e in events if e["event"] == "progress"]
    results = [e for e in events if e["event"] == "result"]
    assert len(progresses) >= 5
    fractions = [p["data"]["fraction"] for p in progresses]
    assert fractions == sorted(fractions)
    assert len(results) == 1
    payload = results[0]["data"]
    assert payload["duration"] == 1.0
    assert len(payload["segments"]) == 1
    assert payload["segments"][0]["label"] == "intro"
    assert "test-song/song_structure/structure.json" in payload["structure_path"]


@pytest.mark.asyncio
async def test_jobs_structure_cache_hit_emits_one_progress(sine_440_wav: Path) -> None:
    ws = srv.get_workspace()
    source = ws.job_dir("test-song") / "source.wav"
    shutil.copy(sine_440_wav, source)

    pipeline = _fake_pipeline_factory()
    saved = srv._structure_pipeline
    srv._structure_pipeline = pipeline
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # First call populates cache.
            await client.post("/jobs/structure", json={"audio_path": str(source)})
            # Second call should hit the cache and emit `cache_hit` only.
            r = await client.post("/jobs/structure", json={"audio_path": str(source)})
            body = r.text
    finally:
        srv._structure_pipeline = saved

    events = _parse_sse(body)
    progresses = [e for e in events if e["event"] == "progress"]
    assert len(progresses) == 1
    assert progresses[0]["data"]["stage"] == "cache_hit"
    assert progresses[0]["data"]["fraction"] == 1.0
    results = [e for e in events if e["event"] == "result"]
    assert results[0]["data"]["cached"] is True
