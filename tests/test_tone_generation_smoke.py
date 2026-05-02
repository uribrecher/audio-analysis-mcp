"""End-to-end smoke test for the subtractive tone-generation training pipeline.

Generates a tiny dataset, runs the train script for a few epochs, and asserts
that the checkpoint + eval_report.json are produced and look structurally sane.
Marked `slow` because dataset generation + a few CNN epochs run for tens of
seconds even at this scale.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
GEN_SCRIPT = REPO_ROOT / "scripts" / "generate_subtractive_dataset.py"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_tone_generation.py"


@pytest.mark.slow
def test_train_smoke(tmp_path: Path) -> None:
    ds_dir = tmp_path / "ds"
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    # Tiny dataset.
    subprocess.run(
        [
            sys.executable,
            str(GEN_SCRIPT),
            "--n-samples",
            "40",
            "--seed",
            "0",
            "--out-dir",
            str(ds_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    # Train for a few epochs.
    res = subprocess.run(
        [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--dataset-dir",
            str(ds_dir),
            "--checkpoint-out",
            str(ckpt_dir / "checkpoint.pt"),
            "--epochs",
            "3",
            "--batch-size",
            "8",
            "--seed",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    print(res.stdout)
    assert (ckpt_dir / "checkpoint.pt").exists()
    eval_path = ckpt_dir / "eval_report.json"
    assert eval_path.exists()
    report = json.loads(eval_path.read_text())
    assert "shape_accuracy" in report
    assert "cutoff_norm_mse" in report
    assert "resonance_mse" in report
    assert report["schema_validation_failures"] == 0
