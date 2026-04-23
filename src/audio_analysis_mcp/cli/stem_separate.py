"""CLI for stem separation with live tqdm progress.

Usage:
    uv run python -m audio_analysis_mcp.cli.stem_separate <audio_path> [--preset fast|medium|accurate]
"""
import argparse
import sys

import demucs.apply
from demucs.pretrained import get_model

from audio_analysis_mcp.tools.stem_separate import stem_separate_impl, PRESETS
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

    # Count sub-models to calculate total runs
    model = get_model(preset.model)
    num_models = len(model.models) if hasattr(model, "models") else 1
    total_runs = num_models * preset.shifts
    del model

    print(
        f"Running {num_models} model(s) x {preset.shifts} shift(s) = {total_runs} total runs",
        file=sys.stderr,
    )

    # Patch tqdm to show run counter
    counter = [0]
    original_tqdm = demucs.apply.tqdm.tqdm  # type: ignore[attr-defined]

    class _CountingTqdm(original_tqdm):  # type: ignore[misc,valid-type]
        def __init__(self, *a, **kw):  # type: ignore[no-untyped-def]
            counter[0] += 1
            kw["desc"] = f"run {counter[0]}/{total_runs}"
            super().__init__(*a, **kw)

    demucs.apply.tqdm.tqdm = _CountingTqdm  # type: ignore[attr-defined]
    try:
        ws = Workspace()
        result = stem_separate_impl(args.audio_path, ws.stems, preset_name=args.preset)
        print(result.model_dump_json(indent=2))
    finally:
        demucs.apply.tqdm.tqdm = original_tqdm  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
