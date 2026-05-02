"""Dataset sampling + on-disk Dataset class for the tone-generation MVP.

This module has two layers:

1. `sample_dataset_config(n_samples, seed)` — pure-Python iterable of
   DatasetItem dataclasses. No I/O; uses scipy Sobol + numpy RNG.
2. `ToneGenerationDataset` — torch.utils.data.Dataset that reads samples
   from disk produced by `scripts/generate_subtractive_dataset.py`.

The generator script in scripts/ uses (1) to produce labels and audio,
then training/eval code uses (2) to read them back.

Layer 2 is added in Task 8.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Iterator

from scipy.stats import qmc

from audio_analysis_mcp.research.tone_generation.schema_io import (
    CUTOFF_HZ_MAX,
    CUTOFF_HZ_MIN,
    SHAPE_LABELS,
    build_canonical_instance,
)

_PITCH_LO = 36
_PITCH_HI = 84
_PITCH_WINDOW = 12  # max semitone span across voices in one render


@dataclass
class DatasetItem:
    """One synthetic training sample's full label.

    Treat as immutable after construction. Not hashable (contains dict/list fields).
    """

    params_canonical: dict[str, Any]
    midi_pitches: list[int]
    n_voices: int


def _next_pow2(n: int) -> int:
    """Next power of 2 >= n. Returns 1 for n <= 1."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _sample_chord_pitches(rng: random.Random, n_voices: int) -> list[int]:
    if n_voices == 1:
        return [rng.randint(_PITCH_LO, _PITCH_HI)]
    # Pick a base pitch; ensure the 12-semitone window stays inside [_PITCH_LO, _PITCH_HI].
    base_lo = _PITCH_LO
    base_hi = _PITCH_HI - _PITCH_WINDOW
    base = rng.randint(base_lo, base_hi)
    candidates = list(range(base, base + _PITCH_WINDOW + 1))
    rng.shuffle(candidates)
    return sorted(candidates[:n_voices])


def sample_dataset_config(*, n_samples: int, seed: int) -> Iterator[DatasetItem]:
    """Yield n_samples DatasetItem instances. Same-process deterministic given seed.

    Cross-version determinism (across Python/scipy versions) is not guaranteed.
    For published-dataset reproducibility, store a fixture mapping seed -> expected items.
    """
    n_pow2 = _next_pow2(n_samples)
    sobol = qmc.Sobol(d=2, scramble=True, seed=seed)
    cont_samples = sobol.random(n_pow2)[:n_samples]  # slice to requested n
    rng = random.Random(seed)
    for i in range(n_samples):
        cutoff_norm, resonance = cont_samples[i]
        cutoff_hz = math.exp(
            math.log(CUTOFF_HZ_MIN)
            + float(cutoff_norm) * (math.log(CUTOFF_HZ_MAX) - math.log(CUTOFF_HZ_MIN))
        )
        shape = SHAPE_LABELS[rng.randrange(len(SHAPE_LABELS))]
        n_voices = rng.randint(1, 3)
        pitches = _sample_chord_pitches(rng, n_voices)
        params = build_canonical_instance(
            shape=shape, cutoff_hz=cutoff_hz, resonance=float(resonance)
        )
        yield DatasetItem(params_canonical=params, midi_pitches=pitches, n_voices=n_voices)
