"""SSE smoke for POST /jobs/triage. Stubs ``triage_notes_by_sections`` so
the test runs without exercising the clustering logic (covered by the
analysis-layer tests); we only care that the SSE wiring serializes
per-section progress and closes with the lean ``{triage_path, cached}``
result event."""
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

import audio_analysis_mcp.server as srv
from audio_analysis_mcp.schemas import NoteTriageBySectionsFileData
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


def _fake_triage_impl(notes, sections, min_duration, max_candidates, jitter_tolerance, progress):
    """Stand-in for triage_notes_by_sections — emits one progress tick per
    section, then a terminal done, and returns an empty per-section
    container (the on-disk file is what the test cares about)."""
    for i, s in enumerate(sections):
        progress("section", (i + 1) / max(1, len(sections)), s.label)
    progress("done", 1.0, None)
    return NoteTriageBySectionsFileData(sections=[])


def _stage_stem_and_notes(ws: Workspace, sine_440_wav: Path) -> Path:
    """Plant both prerequisites: stems/medium/other.wav and the
    transcription.json that /jobs/triage reads as input."""
    stem_path = ws.job_stems_dir("test-song", "medium") / "other.wav"
    shutil.copy(sine_440_wav, stem_path)
    notes_dir = ws.job_transcriptions_dir("test-song", "other", "medium")
    (notes_dir / "transcription.json").write_text("[]")
    return stem_path


@pytest.mark.asyncio
async def test_jobs_triage_streams_progress_then_result(sine_440_wav: Path) -> None:
    ws = srv.get_workspace()
    stem_path = _stage_stem_and_notes(ws, sine_440_wav)
    sections = [
        {"start_time": 0.0, "end_time": 1.0, "label": "intro"},
        {"start_time": 1.0, "end_time": 2.0, "label": "verse"},
    ]

    transport = ASGITransport(app=app)
    with patch(
        "audio_analysis_mcp.service.app.triage_notes_by_sections",
        side_effect=_fake_triage_impl,
    ):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post(
                "/jobs/triage",
                json={"audio_path": str(stem_path), "sections": sections},
            )
            body = r.text

    assert r.status_code == 200
    events = _parse_sse(body)
    progresses = [e for e in events if e["event"] == "progress"]
    results = [e for e in events if e["event"] == "result"]
    # 1 priming "triage 0.05" tick + 2 section ticks + 1 terminal "done"
    assert len(progresses) == 4
    fractions = [p["data"]["fraction"] for p in progresses]
    assert fractions == sorted(fractions)
    # Wire payload is the lean shape: triage_path + cached only.
    assert len(results) == 1
    payload = results[0]["data"]
    assert set(payload.keys()) == {"triage_path", "cached"}
    assert payload["cached"] is False
    assert "triage/other_medium/triage_by_sections.json" in payload["triage_path"]


@pytest.mark.asyncio
async def test_jobs_triage_cache_hit_emits_one_progress(sine_440_wav: Path) -> None:
    """If triage_by_sections.json already exists in the per-stem triage dir,
    skip the work and return cached=True after a single cache_hit event."""
    ws = srv.get_workspace()
    stem_path = _stage_stem_and_notes(ws, sine_440_wav)
    # Pre-seed the cache file so the handler short-circuits.
    triage_dir = ws.job_triage_dir("test-song", "other", "medium")
    (triage_dir / "triage_by_sections.json").write_text('{"sections":[]}')

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/jobs/triage",
            json={"audio_path": str(stem_path), "sections": []},
        )
        body = r.text

    assert r.status_code == 200
    events = _parse_sse(body)
    progresses = [e for e in events if e["event"] == "progress"]
    results = [e for e in events if e["event"] == "result"]
    assert len(progresses) == 1
    assert progresses[0]["data"]["stage"] == "cache_hit"
    assert results[0]["data"]["cached"] is True


@pytest.mark.asyncio
async def test_jobs_triage_rejects_non_stem_path(sine_440_wav: Path) -> None:
    ws = srv.get_workspace()
    source = ws.job_dir("test-song") / "source.wav"
    shutil.copy(sine_440_wav, source)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/jobs/triage", json={"audio_path": str(source), "sections": []},
        )
    assert r.status_code == 400
    assert "stem file" in r.json()["detail"]


@pytest.mark.asyncio
async def test_jobs_triage_missing_transcription_returns_409(sine_440_wav: Path) -> None:
    """Without a prior /jobs/transcribe write, triage has no notes to chew
    on — fail fast with 409 instead of letting the read explode in the
    worker thread."""
    ws = srv.get_workspace()
    stem_path = ws.job_stems_dir("test-song", "medium") / "other.wav"
    shutil.copy(sine_440_wav, stem_path)
    # Note: did NOT stage transcription.json — that's the test.

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/jobs/triage", json={"audio_path": str(stem_path), "sections": []},
        )
    assert r.status_code == 409
    assert "Transcription JSON not found" in r.json()["detail"]
