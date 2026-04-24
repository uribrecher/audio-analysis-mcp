import json
from dataclasses import dataclass
from pathlib import Path
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import AudioFile, save_audio
from audio_analysis_mcp.server import mcp, get_workspace
from audio_analysis_mcp.workspace import resolve_job_context
from audio_analysis_mcp.schemas import StemFile, StemSeparateResult

MANIFEST_FILE = "sources.json"


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
    audio_path: str, stems_dir: Path, preset_name: str = "medium"
) -> StemSeparateResult:
    """Run Demucs stem separation via Python API. Returns cached result if available."""
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    preset = _resolve_preset(preset_name)
    cache_dir = stems_dir

    # Check cache
    cached_sources = _read_cached(cache_dir)
    if cached_sources is not None:
        return StemSeparateResult(
            stems=[StemFile(stem=s, path=str(cache_dir / f"{s}.wav")) for s in cached_sources],
            model=preset.model,
            preset=preset_name,
            cached=True,
        )

    # Cache miss — load model and run separation
    model = get_model(preset.model)
    model.eval()
    source_names = list(model.sources)

    wav = AudioFile(Path(audio_path)).read(streams=0, samplerate=model.samplerate, channels=model.audio_channels)  # type: ignore[no-untyped-call]
    with torch.no_grad():
        sources = apply_model(
            model, wav[None], device=_best_device(),
            shifts=preset.shifts, overlap=preset.overlap, progress=True,
        )[0]

    cache_dir.mkdir(parents=True, exist_ok=True)
    for i, source_name in enumerate(source_names):
        save_audio(sources[i], cache_dir / f"{source_name}.wav", samplerate=model.samplerate)
    (cache_dir / MANIFEST_FILE).write_text(json.dumps(source_names))

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
    ws = get_workspace()
    ctx = resolve_job_context(audio_path, ws)
    stems_dir = ws.job_stems_dir(ctx.job_name, preset)
    result = stem_separate_impl(audio_path, stems_dir, preset_name=preset)
    return result.model_dump_json(indent=2)
