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

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import librosa
import numpy as np
import soundfile as sf
import torch
from scipy.stats import qmc
from torch.utils.data import Dataset

from audio_analysis_mcp.research.tone_generation.schema_io import (
    CUTOFF_HZ_MAX,
    CUTOFF_HZ_MIN,
    SHAPE_LABELS,
    build_canonical_instance,
    normalize_params,
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


# ---- Layer 2: on-disk torch Dataset ----------------------------------------

# 88-key piano range: A0 (MIDI 21) → C8 (MIDI 108).
_PITCH_MULTIHOT_LO = 21
_PITCH_MULTIHOT_HI = 108

# Sustain-region slice: skip the first 100ms (attack), take 300ms.
_SLICE_OFFSET_S = 0.10
_SLICE_DURATION_S = 0.30

# Mel-spectrogram params. At sr=44.1k, hop=441 → 10ms frames → 30 frames per
# 300ms slice. n_fft=2048 (~46ms window), win_length=1102 (~25ms).
_MEL_N_MELS = 128
_MEL_HOP_LENGTH = 441
_MEL_WIN_LENGTH = 1102
_MEL_N_FFT = 2048


def _pitch_multihot(midi_pitches: list[int]) -> torch.Tensor:
    """88-dim float32 multi-hot: 1.0 at index (pitch - 21) for each pitch."""
    vec = torch.zeros(_PITCH_MULTIHOT_HI - _PITCH_MULTIHOT_LO + 1, dtype=torch.float32)
    for p in midi_pitches:
        if _PITCH_MULTIHOT_LO <= p <= _PITCH_MULTIHOT_HI:
            vec[p - _PITCH_MULTIHOT_LO] = 1.0
    return vec


def _audio_to_mel(audio: np.ndarray[Any, Any], sample_rate: int) -> torch.Tensor:
    """Slice a 300ms sustain-region window, compute log-mel, return (1, 128, T)."""
    start = int(sample_rate * _SLICE_OFFSET_S)
    stop = start + int(sample_rate * _SLICE_DURATION_S)
    if stop > audio.shape[0]:
        # Pad with zeros if audio is too short for the slice.
        pad = np.zeros(stop - audio.shape[0], dtype=audio.dtype)
        slice_audio = np.concatenate([audio[start:], pad])
    else:
        slice_audio = audio[start:stop]
    mel = librosa.feature.melspectrogram(
        y=slice_audio.astype(np.float32),
        sr=sample_rate,
        n_fft=_MEL_N_FFT,
        hop_length=_MEL_HOP_LENGTH,
        win_length=_MEL_WIN_LENGTH,
        n_mels=_MEL_N_MELS,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    tensor = torch.from_numpy(log_mel.astype(np.float32)).unsqueeze(0)
    return tensor


class ToneGenerationDataset(Dataset[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]):
    """Reads a disk-backed dataset produced by `generate_subtractive_dataset.py`.

    Each item is a `(mel, pitch_multihot, target)` tuple:

    - `mel`: log-mel spectrogram of a 300ms sustain-region slice, shape (1, 128, T).
    - `pitch_multihot`: 88-dim float32 multi-hot over MIDI pitches 21..108.
    - `target`: dict with normalized regression targets + shape class index.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        labels_path = self.root / "labels.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found at {manifest_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"labels.jsonl not found at {labels_path}")
        with manifest_path.open() as f:
            self.manifest: dict[str, Any] = json.load(f)
        self.sample_rate: int = int(self.manifest["sample_rate"])
        self.samples_dir = self.root / "samples"
        with labels_path.open() as f:
            self.labels: list[dict[str, Any]] = [json.loads(line) for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        label = self.labels[idx]
        wav_path = self.samples_dir / f"{int(label['idx']):06d}.wav"
        audio, sr = sf.read(str(wav_path))
        if sr != self.sample_rate:
            raise ValueError(
                f"sample rate mismatch for {wav_path}: file={sr}, manifest={self.sample_rate}"
            )
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = np.asarray(audio, dtype=np.float32)

        mel = _audio_to_mel(audio, self.sample_rate)
        pitch_multihot = _pitch_multihot(list(label["midi_pitches"]))

        params = label["params_canonical"]["params"]
        shape = params["osc"]["1"]["shape"]
        cutoff_hz = float(params["filter"]["lp"]["cutoff_hz"])
        resonance = float(params["filter"]["lp"]["resonance"])
        norm = normalize_params(shape=shape, cutoff_hz=cutoff_hz, resonance=resonance)
        target: dict[str, Any] = {
            "shape_label": int(norm["shape_label"]),
            "cutoff_norm": float(norm["cutoff_norm"]),
            "resonance": float(norm["resonance"]),
        }
        return mel, pitch_multihot, target
