import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeAlias

import torch
from demucs.apply import apply_model
from demucs.audio import AudioFile, save_audio

from audio_analysis_mcp import _demucs_progress
from audio_analysis_mcp.server import get_demucs_model, get_workspace, mcp
from audio_analysis_mcp.workspace import resolve_job_context
from audio_analysis_mcp.schemas import StemFile, StemSeparateResult

MANIFEST_FILE = "sources.json"

ProgressFn: TypeAlias = Callable[[str, float, "str | None"], None]
"""Progress callback used by ``stem_separate_impl``. See its docstring."""


@dataclass(frozen=True)
class SeparationPreset:
    model: str
    shifts: int
    overlap: float


PRESETS: dict[str, SeparationPreset] = {
    "fast": SeparationPreset(model="htdemucs_6s", shifts=1, overlap=0.1),
    "medium": SeparationPreset(model="htdemucs_6s", shifts=3, overlap=0.25),
    "accurate": SeparationPreset(model="htdemucs_6s", shifts=7, overlap=0.25),
}


def _read_cached(cache_dir: Path) -> list[str] | None:
    """Read source names from cache manifest. Returns None on cache miss."""
    manifest = cache_dir / MANIFEST_FILE
    if not manifest.exists():
        return None
    try:
        parsed = json.loads(manifest.read_text())
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list) or not all(isinstance(s, str) for s in parsed):
        return None
    source_names: list[str] = parsed
    if all((cache_dir / f"{s}.wav").exists() for s in source_names):
        return source_names
    return None


def _best_device() -> str:
    """Auto-detect the best available compute device."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_preset(preset_name: str) -> SeparationPreset:
    """Validate and return a separation preset."""
    if preset_name not in PRESETS:
        raise ValueError(
            f"Unknown preset: {preset_name}. Allowed: {', '.join(sorted(PRESETS))}"
        )
    return PRESETS[preset_name]


def stem_separate_impl(
    audio_path: str,
    stems_dir: Path,
    preset_name: str = "medium",
    progress: ProgressFn | None = None,
) -> StemSeparateResult:
    """Run Demucs stem separation via Python API. Returns cached result if available.

    Args:
        audio_path: Source audio file.
        stems_dir: Output directory for stems + ``sources.json`` manifest.
        preset_name: One of ``fast | medium | accurate``.
        progress: Optional callback ``(stage, fraction, detail) -> None``.
            Stages: ``cache_hit | load_model | run | write | done``.
            Fraction is monotonically non-decreasing in ``[0, 1]``. Thread-safe:
            two concurrent calls install independent sinks via
            ``_demucs_progress.install_sink`` so progress streams don't cross.
    """
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    def emit(stage: str, fraction: float, detail: str | None = None) -> None:
        if progress is None:
            return
        try:
            progress(stage, max(0.0, min(1.0, fraction)), detail)
        except Exception:
            # Don't let a buggy progress sink kill the job.
            pass

    preset = _resolve_preset(preset_name)
    cache_dir = stems_dir

    # Check cache
    cached_sources = _read_cached(cache_dir)
    if cached_sources is not None:
        emit("cache_hit", 1.0)
        return StemSeparateResult(
            stems=[StemFile(stem=s, path=str(cache_dir / f"{s}.wav")) for s in cached_sources],
            model=preset.model,
            preset=preset_name,
            cached=True,
        )

    # Cache miss — load model and run separation
    emit("load_model", 0.02)
    model = get_demucs_model(preset.model)
    source_names = list(model.sources)
    num_sub_models = len(model.models) if hasattr(model, "models") else 1
    total_runs = max(1, num_sub_models * preset.shifts)
    emit("load_model", 0.05)

    wav = AudioFile(Path(audio_path)).read(streams=0, samplerate=model.samplerate, channels=model.audio_channels)  # type: ignore[no-untyped-call]

    # Demucs's `progress=True` creates one tqdm bar per run (= one per shift
    # per sub-model). The routing layer (audio_analysis_mcp._demucs_progress)
    # forwards each tqdm `update` to the sink we install here. We detect a
    # new run by watching `current` reset to a lower value than the previous
    # update — `total` is identical across runs on the same audio (e.g. all
    # three bars at 403.65 for htdemucs_6s medium), so the previous
    # total-change heuristic missed every boundary and the UI bar jumped
    # backward at each new shift.
    completed_runs = 0
    last_current = 0

    def _on_tqdm_update(current: int, total: int) -> None:
        nonlocal completed_runs, last_current
        if current < last_current and last_current > 0:
            completed_runs += 1
        last_current = current
        run_progress = (current / total) if total else 0.0
        overall = (completed_runs + run_progress) / total_runs
        emit("run", 0.05 + 0.9 * overall, f"run {min(completed_runs + 1, total_runs)}/{total_runs}")

    _demucs_progress.install_sink(_on_tqdm_update)
    try:
        with torch.no_grad():
            sources = apply_model(
                model, wav[None], device=_best_device(),
                shifts=preset.shifts, overlap=preset.overlap, progress=True,
            )[0]
    finally:
        _demucs_progress.clear_sink()

    emit("write", 0.97)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for i, source_name in enumerate(source_names):
        save_audio(sources[i], cache_dir / f"{source_name}.wav", samplerate=model.samplerate)
    (cache_dir / MANIFEST_FILE).write_text(json.dumps(source_names))

    emit("done", 1.0)
    return StemSeparateResult(
        stems=[StemFile(stem=s, path=str(cache_dir / f"{s}.wav")) for s in source_names],
        model=preset.model,
        preset=preset_name,
        cached=False,
    )


@mcp.tool()
def stem_separate(audio_path: str, preset: str = "fast") -> str:
    """Separate audio into stems using Demucs.

    Input must be inside a job folder (use import_audio first).
    Returns 6 stems: vocals, drums, bass, other, guitar, piano.
    """
    _resolve_preset(preset)
    ws = get_workspace()
    ctx = resolve_job_context(audio_path, ws)
    stems_dir = ws.job_stems_dir(ctx.job_name, preset)
    result = stem_separate_impl(audio_path, stems_dir, preset_name=preset)
    return result.model_dump_json(indent=2)
