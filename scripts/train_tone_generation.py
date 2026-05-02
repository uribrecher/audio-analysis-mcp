"""Train the subtractive tone-generation CNN.

Usage:
    uv run python scripts/train_tone_generation.py \\
        --dataset-dir scratch/tone_gen_dataset \\
        --checkpoint-out scratch/tone_gen_checkpoints/checkpoint.pt \\
        --epochs 50 --batch-size 64 --seed 0

Splits the dataset by `idx % 10` (train if <8, val if ==8, test if ==9), trains
with summed CE-shape + MSE-cutoff + MSE-resonance loss, early-stops on val-loss
plateau, reloads the best checkpoint, and writes eval_report.json on the test
set with `schema_validation_failures` measured by round-tripping each
prediction through `denormalize_predictions` + `validate_canonical`.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# Sibling-script import: `_eval_helpers.py` lives in the same scripts/ dir but
# isn't a package member, so we have to put scripts/ on sys.path. The package
# imports below DON'T need a sys.path hack — `audio_analysis_mcp` is installed
# editable via `uv sync --dev` and resolved through site-packages.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _eval_helpers import _collate, _split_indices, compute_full_eval  # noqa: E402

from audio_analysis_mcp.research.tone_generation.dataset import (  # noqa: E402
    ToneGenerationDataset,
)
from audio_analysis_mcp.research.tone_generation.model import (  # noqa: E402
    ToneGenerationCNN,
)


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    device = _select_device()
    print(f"device: {device}")

    full = ToneGenerationDataset(args.dataset_dir)
    train_idx, val_idx, test_idx = _split_indices(len(full))
    print(
        f"dataset: total={len(full)} train={len(train_idx)} "
        f"val={len(val_idx)} test={len(test_idx)}"
    )
    train_loader: DataLoader[Any] = DataLoader(
        Subset(full, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate,
    )
    val_loader: DataLoader[Any] = DataLoader(
        Subset(full, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_collate,
    )
    # No test_loader: the final eval uses `compute_full_eval(model, full,
    # test_idx, ...)` which builds its own loader internally.

    model = ToneGenerationCNN().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val = math.inf
    epochs_without_improve = 0
    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0
        for mels, pitches, shape_label, cutoff, res in train_loader:
            mels = mels.to(device)
            pitches = pitches.to(device)
            shape_label = shape_label.to(device)
            cutoff = cutoff.to(device)
            res = res.to(device)
            out = model(mels, pitches)
            loss = (
                F.cross_entropy(out["shape_logits"], shape_label)
                + F.mse_loss(out["cutoff_norm"], cutoff)
                + F.mse_loss(out["resonance"], res)
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss += float(loss.item()) * int(mels.size(0))
            n_train += int(mels.size(0))
        train_loss /= max(n_train, 1)

        # Lightweight val-loss forward — same loss the train step minimizes.
        # Skipping the full eval (round-trip render, schema validation) keeps
        # epochs fast; full metrics run once at the end on the test set.
        val_loss_total = 0.0
        n_val = 0
        model.eval()
        with torch.no_grad():
            for mels, pitches, shape_label, cutoff, res in val_loader:
                mels = mels.to(device)
                pitches = pitches.to(device)
                shape_label = shape_label.to(device)
                cutoff = cutoff.to(device)
                res = res.to(device)
                out = model(mels, pitches)
                loss = (
                    F.cross_entropy(out["shape_logits"], shape_label)
                    + F.mse_loss(out["cutoff_norm"], cutoff)
                    + F.mse_loss(out["resonance"], res)
                )
                val_loss_total += float(loss.item()) * int(mels.size(0))
                n_val += int(mels.size(0))
        val_loss = val_loss_total / max(n_val, 1)
        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
        )

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            epochs_without_improve = 0
            torch.save(model.state_dict(), args.checkpoint_out)
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= args.patience:
                print(f"early stop after {epoch} epochs.")
                break

    # If training never improved (e.g. degenerate run), still save the current
    # weights so downstream eval can load *something* and the smoke test sees
    # a checkpoint file.
    if not args.checkpoint_out.exists():
        torch.save(model.state_dict(), args.checkpoint_out)

    # Final eval on test set with the best checkpoint. Uses the shared helper,
    # which adds round-trip mel cosine + 4x4 confusion matrix on top of the
    # per-param metrics + schema validation that the per-epoch loop tracked.
    model.load_state_dict(torch.load(args.checkpoint_out, map_location=device))
    test_metrics = compute_full_eval(
        model, full, test_idx, device, batch_size=args.batch_size
    )
    eval_report_path = args.checkpoint_out.parent / "eval_report.json"
    eval_report_path.write_text(json.dumps(test_metrics, indent=2) + "\n")
    print(f"\ntest metrics:\n{json.dumps(test_metrics, indent=2)}")
    print(f"eval report -> {eval_report_path}")


if __name__ == "__main__":
    main()
