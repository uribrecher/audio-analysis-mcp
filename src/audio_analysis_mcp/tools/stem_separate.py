import hashlib
from pathlib import Path
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import AudioFile, save_audio
from audio_analysis_mcp.server import mcp, get_workspace
from audio_analysis_mcp.schemas import StemFile, StemSeparateResult


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def stem_separate_impl(
    audio_path: str, stems_dir: Path, model_name: str = "htdemucs"
) -> StemSeparateResult:
    """Run Demucs stem separation via Python API. Returns cached result if available."""
    fhash = _file_hash(audio_path)
    model = get_model(model_name)
    source_names = list(model.sources)
    cache_dir = stems_dir / fhash / model_name

    if cache_dir.exists() and all((cache_dir / f"{s}.wav").exists() for s in source_names):
        return StemSeparateResult(
            stems=[StemFile(stem=s, path=str(cache_dir / f"{s}.wav")) for s in source_names],
            model=model_name,
            cached=True,
        )

    model.eval()
    wav = AudioFile(Path(audio_path)).read(streams=0, samplerate=model.samplerate, channels=model.audio_channels)  # type: ignore[no-untyped-call]
    with torch.no_grad():
        sources = apply_model(model, wav[None], device="cpu")[0]

    cache_dir.mkdir(parents=True, exist_ok=True)
    for i, source_name in enumerate(source_names):
        save_audio(sources[i], cache_dir / f"{source_name}.wav", samplerate=model.samplerate)

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
