"""Validate the divergence threshold (0.15) before locking it into the orchestrator.

Each candidate ADSR is normalized to a 4-D vector:
  (attack_ms / 1000, decay_ms / 1000, sustain_level, release_ms / 1000)

Tests:
  1. Three near-identical ADSRs (small jitter, ±5ms attack/decay/release, ±0.02 sustain)
     → max pairwise distance should be well under 0.15.
  2. Three identical-ish + one outlier (different shape)
     → max pairwise distance should clearly exceed 0.15.
  3. Two ADSRs that differ by a "musically meaningful" amount
     (e.g. attack 20ms vs 50ms, sustain 0.6 vs 0.4) — confirm threshold sits
     between "definitely same" and "definitely different."

Run: uv run python scratch/explore_consensus.py
"""

import numpy as np

_DIVERGENCE_THRESHOLD = 0.15


def to_vec(attack_ms: float, decay_ms: float, sustain: float, release_ms: float) -> np.ndarray:
    return np.array([attack_ms / 1000.0, decay_ms / 1000.0, sustain, release_ms / 1000.0])


def pairwise_max_distance(vectors: np.ndarray) -> float:
    if len(vectors) <= 1:
        return 0.0
    diffs = vectors[:, None, :] - vectors[None, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=-1))
    return float(dists.max())


def report(label: str, vectors: list[np.ndarray]) -> None:
    arr = np.array(vectors)
    max_d = pairwise_max_distance(arr)
    is_consistent = max_d < _DIVERGENCE_THRESHOLD
    mean_vec = arr.mean(axis=0)
    print(f"\n--- {label} ---")
    for i, v in enumerate(vectors):
        print(f"  v{i}: attack={v[0]*1000:.0f}ms decay={v[1]*1000:.0f}ms sustain={v[2]:.2f} release={v[3]*1000:.0f}ms")
    print(f"  max pairwise distance: {max_d:.4f}  →  is_consistent ({max_d} < {_DIVERGENCE_THRESHOLD})? {is_consistent}")
    print(f"  mean: attack={mean_vec[0]*1000:.0f}ms decay={mean_vec[1]*1000:.0f}ms sustain={mean_vec[2]:.2f} release={mean_vec[3]*1000:.0f}ms")


def main() -> None:
    # Case 1: three near-identical ADSRs (small jitter)
    base = to_vec(20, 100, 0.6, 150)
    jitter = [
        base + np.array([0.005, 0.005, 0.02, 0.005]),    # +5ms a/d/r, +0.02 sustain
        base + np.array([-0.005, -0.005, -0.02, 0.010]), # -5ms a/d, -0.02 sustain, +10ms release
        base,
    ]
    report("Case 1 — three near-identical (should be consistent)", jitter)

    # Case 2: three similar + one outlier
    similar = [base, base + np.array([0.005, 0.005, 0.02, 0.005]), base - np.array([0.005, 0.005, 0.02, 0.005])]
    outlier = to_vec(80, 40, 0.2, 40)  # short envelope, low sustain — clearly different shape
    report("Case 2 — three similar + one outlier (should be divergent)", similar + [outlier])

    # Case 3: knife-edge — moderately different shapes
    a = to_vec(20, 100, 0.6, 150)
    b = to_vec(50, 100, 0.4, 150)  # attack +30ms, sustain -0.2
    report("Case 3 — moderately different (should likely be divergent)", [a, b])

    # Case 4: musically same shape but absolute amplitude / velocity scaling
    # Should be consistent because we already normalize sustain_level by peak (per Task 4 design)
    a = to_vec(20, 100, 0.6, 150)
    b = to_vec(20, 100, 0.6, 150)  # identical (velocity normalization handled upstream)
    report("Case 4 — identical (should be consistent, distance 0)", [a, b])

    # Case 5: two identical mono envelopes vs slightly different release
    a = to_vec(20, 100, 0.6, 150)
    b = to_vec(20, 100, 0.6, 250)  # release +100ms
    report("Case 5 — release differs by 100ms (likely just over threshold)", [a, b])


if __name__ == "__main__":
    main()