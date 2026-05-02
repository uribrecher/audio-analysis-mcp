"""Re-run eval against a saved checkpoint without retraining.

Usage:
    uv run python scripts/eval_tone_generation.py \\
        --checkpoint scratch/tone_gen_checkpoints/checkpoint.pt \\
        --dataset-dir scratch/tone_gen_dataset

Writes the same eval_report.json shape as `train_tone_generation.py`'s final
test eval (per-param metrics + 4x4 confusion + round-trip mel cosine + schema
validation).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Sibling-script import (see train_tone_generation.py for the same comment):
# `_eval_helpers.py` lives in scripts/ and is not packaged, so we put scripts/
# on sys.path. The `audio_analysis_mcp` package is editable-installed.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _eval_helpers import compute_full_eval  # noqa: E402

from audio_analysis_mcp.research.tone_generation.dataset import (  # noqa: E402
    ToneGenerationDataset,
)
from audio_analysis_mcp.research.tone_generation.model import (  # noqa: E402
    ToneGenerationCNN,
)


def _split_indices(n: int) -> tuple[list[int], list[int], list[int]]:
    """Modulo-10 deterministic split — must match train_tone_generation.py."""
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for i in range(n):
        bucket = i % 10
        if bucket < 8:
            train.append(i)
        elif bucket == 8:
            val.append(i)
        else:
            test.append(i)
    return train, val, test


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: <checkpoint dir>/eval_report.json).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = _select_device()
    print(f"device: {device}")

    full = ToneGenerationDataset(args.dataset_dir)
    _, _, test_idx = _split_indices(len(full))
    print(f"dataset: total={len(full)} test={len(test_idx)}")

    model = ToneGenerationCNN().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    metrics = compute_full_eval(
        model, full, test_idx, device, batch_size=args.batch_size
    )
    out = args.out or args.checkpoint.parent / "eval_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"eval report -> {out}")


if __name__ == "__main__":
    main()
