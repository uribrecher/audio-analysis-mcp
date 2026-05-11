"""CLI for stem separation with live progress on stderr.

Usage:
    uv run python -m audio_analysis_mcp.cli.stem_separate <audio_path> [--preset fast|medium|accurate]
"""
import argparse
import sys

from demucs.pretrained import get_model

# Import for side effect: installs the routing tqdm class once at startup.
from audio_analysis_mcp import _demucs_progress  # noqa: F401
from audio_analysis_mcp.tools.stem_separate import PRESETS, stem_separate_impl
from audio_analysis_mcp.workspace import Workspace


def main() -> None:
    parser = argparse.ArgumentParser(description="Separate audio into stems using Demucs")
    parser.add_argument("audio_path", help="Path to the audio file")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="medium",
        help="Quality preset (default: medium)",
    )
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    model = get_model(preset.model)
    num_models = len(model.models) if hasattr(model, "models") else 1
    total_runs = num_models * preset.shifts
    del model

    print(
        f"Running {num_models} model(s) x {preset.shifts} shift(s) = {total_runs} total runs",
        file=sys.stderr,
    )

    def progress(stage: str, fraction: float, detail: str | None = None) -> None:
        if detail:
            print(f"  [{fraction * 100:5.1f}%] {stage}: {detail}", file=sys.stderr)
        else:
            print(f"  [{fraction * 100:5.1f}%] {stage}", file=sys.stderr)

    ws = Workspace()
    result = stem_separate_impl(
        args.audio_path,
        ws.stems,
        preset_name=args.preset,
        progress=progress,
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
