"""Shared eval logic for tone-generation training + standalone eval scripts.

Both `train_tone_generation.py` (final test eval) and `eval_tone_generation.py`
(re-run eval against a saved checkpoint) consume `compute_full_eval`. The
helper computes per-param metrics, schema-validation pass count, a 4x4 shape
confusion matrix, AND a round-trip mel-cosine — each predicted canonical
instance is rendered back through SignalFlow + the dataset's mel pipeline and
compared by cosine similarity against the original mel.

`_audio_to_mel` is imported as a private symbol from the dataset module on
purpose: round-tripping requires the exact same mel-spec preprocessing as the
training samples, and duplicating those constants here would silently drift.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from audio_analysis_mcp.research.tone_generation.dataset import (
    ToneGenerationDataset,
    _audio_to_mel,  # private on purpose: see module docstring.
)
from audio_analysis_mcp.research.tone_generation.renderer import render_chord
from audio_analysis_mcp.research.tone_generation.schema_io import (
    denormalize_predictions,
    validate_canonical,
)


# Render parameters for round-trip — match the dataset generator's defaults so
# the rendered slice has the same characteristics as training audio.
_SAMPLE_RATE = 44_100
_TOTAL_DURATION_S = 1.2

# Pitch multihot index 0 -> MIDI 21 (A0). Mirrors dataset._PITCH_MULTIHOT_LO.
_PITCH_MULTIHOT_LO = 21
_PITCH_FALLBACK = 60  # middle C — used only if multihot somehow has zero pitches.


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


def _round_trip_mel_cosine(
    instance: dict[str, Any],
    midi_pitches: list[int],
    original_mel: torch.Tensor,
) -> float:
    """Render the predicted instance, mel-spec it, return cosine vs original.

    Mels are flattened to 1-D, length-aligned to the shorter of the two (the
    dataset slice and freshly-rendered slice may differ by a frame due to
    librosa's frame-count rounding), L2-normalized, and dotted. Result is in
    [-1, 1].
    """
    audio = render_chord(
        params_canonical=instance,
        midi_pitches=midi_pitches,
        sample_rate=_SAMPLE_RATE,
        total_duration_s=_TOTAL_DURATION_S,
    )
    rendered_mel = _audio_to_mel(audio, _SAMPLE_RATE)  # (1, 128, T)
    a = rendered_mel.flatten()
    b = original_mel.detach().cpu().flatten()
    n = min(a.size(0), b.size(0))
    a, b = a[:n], b[:n]
    eps = 1e-8
    return float(
        torch.dot(a, b)
        / (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b) + eps)
    )


def _split_indices(n: int) -> tuple[list[int], list[int], list[int]]:
    """Modulo-10 deterministic split: train if i%10 < 8, val if ==8, test if ==9.

    Shared by `train_tone_generation.py` (split for train/val/test) and
    `eval_tone_generation.py` (recover the test bucket from a saved checkpoint).
    Keep the two consumers in lockstep — if this changes, both shift together.
    """
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


def compute_full_eval(
    model: torch.nn.Module,
    dataset: ToneGenerationDataset,
    indices: list[int],
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Run model on `indices` subset; return metrics dict.

    Includes per-param metrics (shape accuracy, cutoff/resonance MSE), schema
    validation failure count, 4x4 shape confusion matrix, and round-trip mel
    cosine (mean + median). Each test-time prediction triggers a SignalFlow
    render — keep eval sets bounded.
    """
    # Guard against manifest-rate drift: the round-trip render and mel
    # extraction below use the module-level _SAMPLE_RATE / _TOTAL_DURATION_S,
    # so a regenerated dataset with different rate/duration would silently
    # produce a meaningless cosine. Fail loud instead.
    assert dataset.sample_rate == _SAMPLE_RATE, (
        f"dataset.sample_rate={dataset.sample_rate} != _SAMPLE_RATE={_SAMPLE_RATE}; "
        "regenerate dataset or update _eval_helpers constants"
    )
    assert dataset.total_duration_s == _TOTAL_DURATION_S, (
        f"dataset.total_duration_s={dataset.total_duration_s} != "
        f"_TOTAL_DURATION_S={_TOTAL_DURATION_S}; "
        "regenerate dataset or update _eval_helpers constants"
    )
    model.eval()
    loader: DataLoader[Any] = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
    )

    n = 0
    correct_shape = 0
    cutoff_se = 0.0
    res_se = 0.0
    failures = 0
    cosines: list[float] = []
    confusion = np.zeros((4, 4), dtype=np.int64)

    with torch.no_grad():
        for mels, pitches, shape_label, cutoff, res in loader:
            mels_d = mels.to(device)
            pitches_d = pitches.to(device)
            shape_label_d = shape_label.to(device)
            cutoff_d = cutoff.to(device)
            res_d = res.to(device)
            out = model(mels_d, pitches_d)
            preds_shape = out["shape_logits"].argmax(dim=1)
            correct_shape += int((preds_shape == shape_label_d).sum().item())
            cutoff_se += float(
                F.mse_loss(out["cutoff_norm"], cutoff_d, reduction="sum").item()
            )
            res_se += float(
                F.mse_loss(out["resonance"], res_d, reduction="sum").item()
            )
            n += int(mels.size(0))
            for i in range(mels.size(0)):
                true_label = int(shape_label[i].item())
                pred_label = int(preds_shape[i].item())
                confusion[true_label, pred_label] += 1
                pitch_idxs = (
                    (pitches[i] > 0.5).nonzero(as_tuple=False).squeeze(-1).tolist()
                )
                midi_pitches = [_PITCH_MULTIHOT_LO + j for j in pitch_idxs] or [
                    _PITCH_FALLBACK
                ]
                inst = denormalize_predictions(
                    shape_label=pred_label,
                    cutoff_norm=float(out["cutoff_norm"][i].item()),
                    resonance=float(out["resonance"][i].item()),
                    midi_pitches=midi_pitches,
                )
                try:
                    validate_canonical(inst)
                except Exception:
                    failures += 1
                    continue
                cos = _round_trip_mel_cosine(inst, midi_pitches, mels[i])
                cosines.append(cos)

    denom = max(n, 1)
    return {
        "shape_accuracy": correct_shape / denom,
        "shape_confusion": confusion.tolist(),
        "cutoff_norm_mse": cutoff_se / denom,
        "resonance_mse": res_se / denom,
        "schema_validation_failures": failures,
        "round_trip_mel_cosine_mean": float(np.mean(cosines)) if cosines else 0.0,
        "round_trip_mel_cosine_median": (
            float(np.median(cosines)) if cosines else 0.0
        ),
        "n_samples": n,
    }
