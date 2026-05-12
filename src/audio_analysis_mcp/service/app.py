"""FastAPI app exposing the audio-analysis operations over HTTP + SSE.

Routes:

    GET  /healthz             — liveness; clients hit this after spawning the
                                worker process to know it's ready.
    POST /jobs/import         — sync; normalize an external audio file into
                                the workspace as ``jobs/<job>/source.wav``.
    POST /jobs/stems          — SSE; stream Demucs progress, final result.
    POST /jobs/structure      — SSE; stream SongFormer progress, final result.

Concurrency: per-pipeline locks. Stems and structure jobs run **in parallel
with each other** — different models, different state, GPU is the only shared
resource. Two stem jobs or two structure jobs serialize. Why per-pipeline,
not none:

    - SongFormer: empirically NOT thread-safe under PyTorch eval mode.
      Concurrent ``analyze()`` calls on the same pipeline instance crash
      with tensor-shape mismatches (verified by firing two simultaneous
      /jobs/structure cache-misses against the same singleton).
    - Demucs: not verified thread-safe — locking conservatively. Even with
      the thread-local Demucs progress refactor (which removes the
      *progress-stream* contamination), the underlying model forward pass
      may share state. Can be revisited once we have an empirical test.

The Demucs progress refactor (:py:mod:`audio_analysis_mcp._demucs_progress`)
still earns its keep: it cleans up a process-global tqdm monkey-patch and
makes progress streams unambiguously per-thread, so the stem lock is the
*only* serialization point on the stems path — not also a stderr-progress
collision waiting to happen.

An optional ``AUDIO_ANALYSIS_SERVICE_MAX_GPU_JOBS`` env var caps total GPU
concurrency on top of the per-pipeline locks (default unbounded).

Single-worker uvicorn is required so the model singletons in
:py:mod:`audio_analysis_mcp.server` are shared across requests.
"""
from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, Literal

import anyio
from anyio import to_thread
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from audio_analysis_mcp.analysis.structure_analysis import analyze_structure
from audio_analysis_mcp.analysis.transcription import transcribe_audio
from audio_analysis_mcp.schemas import ImportAudioResult, NoteTranscribeServiceResult
from audio_analysis_mcp.server import (
    get_structure_pipeline,
    get_workspace,
    release_demucs_model,
    release_structure_pipeline,
)
from audio_analysis_mcp.service.progress import ProgressChannel
from audio_analysis_mcp.service.sse import error_event, progress_event, result_event
from audio_analysis_mcp.tools.import_audio import import_audio_impl
from audio_analysis_mcp.tools.stem_separate import PRESETS, stem_separate_impl
from audio_analysis_mcp.workspace import resolve_job_context

app = FastAPI(title="audio-analysis-mcp", version="0.1.0")

# Per-pipeline locks: stems and structure can run in parallel with each
# other (different models), but two of the same kind serialize because the
# underlying model forward passes aren't thread-safe.
_demucs_lock = anyio.Lock()
_structure_lock = anyio.Lock()
# Basic Pitch's predict() is not documented as thread-safe; serialize
# concurrent transcribe requests for the same reason we serialize stems
# and structure.
_transcribe_lock = anyio.Lock()

# Optional cap on simultaneous GPU jobs. ``0`` or unset is "unbounded".
# Parse defensively so an invalid env value (typo, empty string) downgrades
# to "unbounded" with a warning instead of crashing the app at import time.
def _parse_max_gpu_jobs(raw: str | None) -> int:
    if not raw:
        return 0
    try:
        v = int(raw)
        return max(0, v)
    except ValueError:
        import warnings

        warnings.warn(
            f"AUDIO_ANALYSIS_SERVICE_MAX_GPU_JOBS={raw!r} is not an integer; "
            "ignoring and running unbounded.",
            stacklevel=2,
        )
        return 0


_max_gpu_jobs = _parse_max_gpu_jobs(os.environ.get("AUDIO_ANALYSIS_SERVICE_MAX_GPU_JOBS"))
_gpu_slot: anyio.Semaphore | None = (
    anyio.Semaphore(_max_gpu_jobs) if _max_gpu_jobs > 0 else None
)


# -------- request models --------


class ImportRequest(BaseModel):
    file_path: str
    start_time: float | None = None
    duration: float | None = None


class StemsRequest(BaseModel):
    audio_path: str
    preset: Literal["fast", "medium", "accurate"] = "medium"


class StructureRequest(BaseModel):
    audio_path: str


class TranscribeRequest(BaseModel):
    audio_path: str


# -------- shared helpers --------


ProgressSink = Callable[[str, float, "str | None"], None]
BlockingCall = Callable[[ProgressSink], Any]


@asynccontextmanager
async def _optional_gpu_slot() -> AsyncIterator[None]:
    """Acquire the GPU concurrency slot if one is configured; otherwise a no-op.

    Keeps the route handlers free of conditional ``async with`` ladders.
    """
    if _gpu_slot is None:
        yield
        return
    async with _gpu_slot:
        yield


async def _run_with_progress_stream(
    blocking_call: BlockingCall,
) -> AsyncIterator[dict[str, Any]]:
    """Run a blocking callable that takes a ``progress`` kwarg in a worker
    thread, streaming the progress events it emits as SSE messages and
    yielding a final ``result`` (or ``error``) event.

    The blocking call is given ``ch.sync_emit`` as its progress sink.
    """
    ch = ProgressChannel()
    state: dict[str, object] = {}

    async def worker() -> None:
        try:
            state["result"] = await to_thread.run_sync(
                lambda: blocking_call(ch.sync_emit),
            )
        except BaseException as exc:  # noqa: BLE001 — we need to surface anything
            state["error"] = exc
        finally:
            ch.close()

    async with anyio.create_task_group() as tg:
        tg.start_soon(worker)
        async for evt in ch.stream():
            yield progress_event(evt["stage"], evt["fraction"], evt.get("detail"))

    if "error" in state:
        yield error_event(state["error"])  # type: ignore[arg-type]
    else:
        yield result_event(state["result"])


# -------- routes --------


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/jobs/import")
async def jobs_import(req: ImportRequest) -> ImportAudioResult:
    """Stage an audio file into the workspace. Sync, fast."""
    try:
        # Run in thread to avoid blocking the event loop on file I/O.
        return await to_thread.run_sync(
            lambda: import_audio_impl(req.file_path, req.start_time, req.duration),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/jobs/stems")
async def jobs_stems(req: StemsRequest) -> EventSourceResponse:
    """Demucs stem separation with streamed progress.

    Preset is validated by the ``Literal[...]`` annotation on ``StemsRequest``
    — Pydantic rejects unknown values with 422 before this handler runs, so
    no manual whitelist check is needed.
    """
    ws = get_workspace()
    try:
        ctx = resolve_job_context(req.audio_path, ws)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    stems_dir = ws.job_stems_dir(ctx.job_name, req.preset)

    async def gen() -> AsyncIterator[dict[str, Any]]:
        async with _demucs_lock, _optional_gpu_slot():
            try:
                async for evt in _run_with_progress_stream(
                    lambda sink: stem_separate_impl(
                        req.audio_path, stems_dir, preset_name=req.preset, progress=sink
                    )
                ):
                    yield evt
            finally:
                # Drop the Demucs model's RSS while we still hold the
                # lock — a concurrent /jobs/stems queued behind us will
                # pay the reload cost rather than hit a half-torn-down
                # singleton. The cap on idle memory is worth the ~10-20s
                # reload on the next request.
                release_demucs_model(PRESETS[req.preset].model)

    return EventSourceResponse(gen())


@app.post("/jobs/structure")
async def jobs_structure(req: StructureRequest) -> EventSourceResponse:
    """SongFormer song-structure analysis with streamed progress."""
    ws = get_workspace()
    try:
        ctx = resolve_job_context(req.audio_path, ws)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    output_dir = str(ws.job_song_structure_dir(ctx.job_name))

    async def gen() -> AsyncIterator[dict[str, Any]]:
        async with _structure_lock, _optional_gpu_slot():
            try:
                async for evt in _run_with_progress_stream(
                    lambda sink: analyze_structure(
                        req.audio_path, output_dir, get_structure_pipeline(), progress=sink
                    )
                ):
                    yield evt
            finally:
                # Drop the SongFormer pipeline immediately to release its
                # ~1-7GB RSS footprint (varies by song length and device).
                # See release_structure_pipeline docstring re: lock ordering.
                release_structure_pipeline()

    return EventSourceResponse(gen())


@app.post("/jobs/transcribe")
async def jobs_transcribe(req: TranscribeRequest) -> EventSourceResponse:
    """Basic Pitch transcription of a stem to MIDI, with streamed progress.

    The endpoint accepts any path inside a job, but to keep output paths
    deterministic we require the path to resolve to a stem (i.e. live at
    ``jobs/<job>/stems/<preset>/<stem>.wav``) — that's what the SONG
    ANALYSIS panel sends today (``other.wav``). Source-level paths are
    rejected with 400 so callers don't accidentally hit the noisier
    full-mix code path.
    """
    ws = get_workspace()
    try:
        ctx = resolve_job_context(req.audio_path, ws)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if ctx.stem is None or ctx.preset is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected a stem file (jobs/<job>/stems/<preset>/<stem>.wav), got: "
                f"{req.audio_path}"
            ),
        )
    # `resolve_job_context` recognises any file under `stems/<preset>/` as a
    # stem regardless of extension, so an explicit suffix check is the only
    # thing keeping `.../stems/medium/other.mp3` from sneaking through and
    # tripping the renderer downstream (the panel only ever sends .wav).
    if not req.audio_path.lower().endswith(".wav"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected a .wav stem file, got: {req.audio_path}"
            ),
        )
    output_dir = str(ws.job_transcriptions_dir(ctx.job_name, ctx.stem, ctx.preset))

    def run(sink: ProgressSink) -> NoteTranscribeServiceResult:
        midi_path, _notes_path, _notes, cached = transcribe_audio(
            req.audio_path, output_dir=output_dir, progress=sink,
        )
        return NoteTranscribeServiceResult(midi_path=midi_path, cached=cached)

    async def gen() -> AsyncIterator[dict[str, Any]]:
        async with _transcribe_lock, _optional_gpu_slot():
            async for evt in _run_with_progress_stream(run):
                yield evt

    return EventSourceResponse(gen())
