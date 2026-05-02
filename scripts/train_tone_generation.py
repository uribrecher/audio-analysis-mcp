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
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from audio_analysis_mcp.research.tone_generation.dataset import ToneGenerationDataset
from audio_analysis_mcp.research.tone_generation.model import ToneGenerationCNN
from audio_analysis_mcp.research.tone_generation.schema_io import (
    denormalize_predictions,
    validate_canonical,
)


# Pitch multihot index 0 → MIDI 21 (A0), per dataset._PITCH_MULTIHOT_LO.
_PITCH_MULTIHOT_LO = 21
_PITCH_FALLBACK = 60  # middle C — used only if multihot somehow has zero pitches.


def _split_indices(n: int) -> tuple[list[int], list[int], list[int]]:
    """Modulo-10 deterministic split: train if i%10 < 8, val if ==8, test if ==9."""
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


def _collate(
    batch: list[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten Dataset's (mel, pitch, target_dict) tuples into a 5-tensor batch."""
    mels = torch.stack([item[0] for item in batch])
    pitches = torch.stack([item[1] for item in batch])
    shape_label = torch.tensor(
        [item[2]["shape_label"] for item in batch], dtype=torch.long
    )
    cutoff = torch.tensor(
        [item[2]["cutoff_norm"] for item in batch], dtype=torch.float32
    )
    res = torch.tensor(
        [item[2]["resonance"] for item in batch], dtype=torch.float32
    )
    return mels, pitches, shape_label, cutoff, res


def _compute_eval(
    model: ToneGenerationCNN,
    loader: DataLoader[Any],
    device: torch.device,
    report_canonical_failures: bool = True,
) -> dict[str, Any]:
    """Per-param metrics; optionally validates each prediction's canonical instance."""
    model.eval()
    n = 0
    correct_shape = 0
    cutoff_se = 0.0
    res_se = 0.0
    failures = 0
    with torch.no_grad():
        for mels, pitches, shape_label, cutoff, res in loader:
            mels = mels.to(device)
            pitches = pitches.to(device)
            shape_label = shape_label.to(device)
            cutoff = cutoff.to(device)
            res = res.to(device)
            out = model(mels, pitches)
            preds_shape = out["shape_logits"].argmax(dim=1)
            correct_shape += int((preds_shape == shape_label).sum().item())
            cutoff_se += float(
                F.mse_loss(out["cutoff_norm"], cutoff, reduction="sum").item()
            )
            res_se += float(
                F.mse_loss(out["resonance"], res, reduction="sum").item()
            )
            n += int(mels.size(0))
            if report_canonical_failures:
                for i in range(mels.size(0)):
                    pitch_idxs = (
                        (pitches[i] > 0.5)
                        .nonzero(as_tuple=False)
                        .squeeze(-1)
                        .tolist()
                    )
                    midi_pitches = [_PITCH_MULTIHOT_LO + j for j in pitch_idxs] or [
                        _PITCH_FALLBACK
                    ]
                    inst = denormalize_predictions(
                        shape_label=int(preds_shape[i].item()),
                        cutoff_norm=float(out["cutoff_norm"][i].item()),
                        resonance=float(out["resonance"][i].item()),
                        midi_pitches=midi_pitches,
                    )
                    try:
                        validate_canonical(inst)
                    except Exception:
                        failures += 1
    denom = max(n, 1)
    return {
        "shape_accuracy": correct_shape / denom,
        "cutoff_norm_mse": cutoff_se / denom,
        "resonance_mse": res_se / denom,
        "schema_validation_failures": failures,
        "n_samples": n,
    }


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
    test_loader: DataLoader[Any] = DataLoader(
        Subset(full, test_idx),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_collate,
    )

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

        val_metrics = _compute_eval(
            model, val_loader, device, report_canonical_failures=False
        )
        val_loss = (
            (1.0 - val_metrics["shape_accuracy"])
            + val_metrics["cutoff_norm_mse"]
            + val_metrics["resonance_mse"]
        )
        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"shape_acc={val_metrics['shape_accuracy']:.3f} "
            f"cutoff_mse={val_metrics['cutoff_norm_mse']:.4f} "
            f"res_mse={val_metrics['resonance_mse']:.4f}"
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

    # Final eval on test set with the best checkpoint.
    model.load_state_dict(torch.load(args.checkpoint_out, map_location=device))
    test_metrics = _compute_eval(model, test_loader, device)
    eval_report_path = args.checkpoint_out.parent / "eval_report.json"
    eval_report_path.write_text(json.dumps(test_metrics, indent=2) + "\n")
    print(f"\ntest metrics:\n{json.dumps(test_metrics, indent=2)}")
    print(f"eval report -> {eval_report_path}")


if __name__ == "__main__":
    main()
