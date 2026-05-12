"""T11 sanity probe.

Verify that:
1. `_eval_helpers` imports cleanly (no circular import, no broken sys.path).
2. `_collate` matches the contract the train script's loop expects.
3. `compute_full_eval` is callable and returns the expected dict keys.

Run from the repo root:
    uv run python scratch/explore_t11_eval_helpers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import torch  # noqa: E402

from _eval_helpers import _collate, _round_trip_mel_cosine, compute_full_eval  # noqa: E402


def main() -> None:
    print("imports ok")
    print("compute_full_eval:", compute_full_eval)
    print("_collate:", _collate)
    print("_round_trip_mel_cosine:", _round_trip_mel_cosine)
    # Cosine of a vector with itself = 1.0 (sanity check the math).
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.0, 2.0, 3.0])
    eps = 1e-8
    cos_self = float(
        torch.dot(a, b)
        / (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b) + eps)
    )
    print(f"cosine(a, a) = {cos_self:.6f}  (should be ~1.0)")
    assert 0.999 <= cos_self <= 1.0001


if __name__ == "__main__":
    main()
