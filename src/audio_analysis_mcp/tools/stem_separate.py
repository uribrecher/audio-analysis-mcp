import hashlib
import json
from pathlib import Path
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import AudioFile, save_audio
from audio_analysis_mcp.server import mcp, get_workspace
from audio_analysis_mcp.schemas import StemFile, StemSeparateResult

MANIFEST_FILE = "sources.json"


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


ALLOWED_MODELS = {"htdemucs", "htdemucs_ft", "htdemucs_6s", "hdemucs_mmi", "mdx", "mdx_extra"}


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


def _sanitize_model_name(model_name: str) -> str:
    """Validate model name to prevent path traversal."""
    if model_name not in ALLOWED_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. Allowed: {', '.join(sorted(ALLOWED_MODELS))}"
        )
    return model_name


def stem_separate_impl(
    audio_path: str, stems_dir: Path, model_name: str = "htdemucs"
) -> StemSeparateResult:
    """Run Demucs stem separation via Python API. Returns cached result if available."""
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    safe_model = _sanitize_model_name(model_name)
    fhash = _file_hash(audio_path)
    cache_dir = stems_dir / fhash / safe_model

    # Check cache without loading the model
    cached_sources = _read_cached(cache_dir)
    if cached_sources is not None:
        return StemSeparateResult(
            stems=[StemFile(stem=s, path=str(cache_dir / f"{s}.wav")) for s in cached_sources],
            model=model_name,
            cached=True,
        )

    # Cache miss — load model and run separation
    model = get_model(model_name)
    model.eval()
    source_names = list(model.sources)

    wav = AudioFile(Path(audio_path)).read(streams=0, samplerate=model.samplerate, channels=model.audio_channels)  # type: ignore[no-untyped-call]
    with torch.no_grad():
        sources = apply_model(model, wav[None], device="cpu")[0]

    cache_dir.mkdir(parents=True, exist_ok=True)
    for i, source_name in enumerate(source_names):
        save_audio(sources[i], cache_dir / f"{source_name}.wav", samplerate=model.samplerate)
    (cache_dir / MANIFEST_FILE).write_text(json.dumps(source_names))

    return StemSeparateResult(
        stems=[StemFile(stem=s, path=str(cache_dir / f"{s}.wav")) for s in source_names],
        model=model_name,
        cached=False,
    )


@mcp.tool()
def stem_separate(audio_path: str, model: str = "htdemucs") -> str:
    """Separate audio into stems (vocals, drums, bass, other) using Demucs."""
    ws = get_workspace()
    result = stem_separate_impl(audio_path, ws.stems, model)
    return result.model_dump_json(indent=2)
