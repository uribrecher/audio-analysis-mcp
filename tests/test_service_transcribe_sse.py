"""SSE smoke for POST /jobs/transcribe. Stubs ``transcribe_audio`` so the
test runs in ~0.5s without loading Basic Pitch — we only care that the
SSE wiring serializes progress events from a worker thread and closes
with a final ``result`` event carrying the minimal NoteTranscribeService
payload (midi_path + cached only)."""
import json
import shutil
from pathlib import Path
from unittest.mock import patch

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


def _fake_transcribe_impl(audio_path, output_dir, progress=None):
    """Stand-in for ``transcribe_audio`` — emits a few progress ticks and
    writes a fake transcription.mid so the result midi_path points at
    something that exists on disk."""
    if progress is not None:
        progress("load_audio", 0.05, None)
        progress("predict", 0.10, None)
        progress("predict", 0.90, None)
        progress("write", 0.95, None)
        progress("done", 1.0, None)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    midi = out / "transcription.mid"
    midi.write_bytes(b"MThd")  # not a real MIDI; just needs to exist
    notes = out / "transcription.json"
    notes.write_text("[]")
    return str(midi), str(notes), [], False


@pytest.mark.asyncio
async def test_jobs_transcribe_streams_progress_then_result(sine_440_wav: Path) -> None:
    # Stage a stem at jobs/<job>/stems/medium/other.wav so resolve_job_context
    # gives back stem="other" and preset="medium" and the output dir lands
    # under the expected per-stem-per-preset directory.
    ws = srv.get_workspace()
    stem_path = ws.job_stems_dir("test-song", "medium") / "other.wav"
    shutil.copy(sine_440_wav, stem_path)

    transport = ASGITransport(app=app)
    with patch(
        "audio_analysis_mcp.service.app.transcribe_audio",
        side_effect=_fake_transcribe_impl,
    ):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post(
                "/jobs/transcribe", json={"audio_path": str(stem_path)}
            )
            body = r.text

    assert r.status_code == 200
    events = _parse_sse(body)
    progresses = [e for e in events if e["event"] == "progress"]
    results = [e for e in events if e["event"] == "result"]
    assert len(progresses) == 5
    assert progresses[0]["data"]["stage"] == "load_audio"
    assert progresses[-1]["data"]["stage"] == "done"
    # Monotonic
    fractions = [p["data"]["fraction"] for p in progresses]
    assert fractions == sorted(fractions)
    # Result payload is the lean shape: midi_path + cached only.
    assert len(results) == 1
    payload = results[0]["data"]
    assert set(payload.keys()) == {"midi_path", "cached"}
    assert payload["cached"] is False
    assert "transcriptions/other_medium/transcription.mid" in payload["midi_path"]


@pytest.mark.asyncio
async def test_jobs_transcribe_rejects_non_stem_path(sine_440_wav: Path) -> None:
    """source.wav (or any path not living under stems/<preset>/) is rejected
    with 400 so callers don't accidentally trigger the noisier full-mix
    transcription path."""
    ws = srv.get_workspace()
    source = ws.job_dir("test-song") / "source.wav"
    shutil.copy(sine_440_wav, source)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/jobs/transcribe", json={"audio_path": str(source)}
        )
    assert r.status_code == 400
    assert "stem file" in r.json()["detail"]
