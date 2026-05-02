import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from audio_analysis_mcp.research.tone_generation.dataset import (
    DatasetItem,
    ToneGenerationDataset,
    sample_dataset_config,
)
from audio_analysis_mcp.research.tone_generation.schema_io import (
    SHAPE_LABELS,
    build_canonical_instance,
    validate_canonical,
)


def test_sample_dataset_config_count_and_types():
    items = list(sample_dataset_config(n_samples=20, seed=42))
    assert len(items) == 20
    for it in items:
        assert isinstance(it, DatasetItem)
        validate_canonical(it.params_canonical)
        assert 1 <= it.n_voices <= 3
        assert len(it.midi_pitches) == it.n_voices
        assert all(36 <= p <= 84 for p in it.midi_pitches)


def test_sample_dataset_config_pitches_unique_and_in_window():
    items = list(sample_dataset_config(n_samples=200, seed=42))
    for it in items:
        if it.n_voices > 1:
            assert len(set(it.midi_pitches)) == it.n_voices, "pitches must be distinct"
            assert max(it.midi_pitches) - min(it.midi_pitches) <= 12, (
                f"pitches outside 12-semitone window: {it.midi_pitches}"
            )


def test_sample_dataset_config_param_ranges():
    items = list(sample_dataset_config(n_samples=200, seed=42))
    for it in items:
        p = it.params_canonical["params"]
        assert p["osc"]["1"]["shape"] in SHAPE_LABELS
        cutoff = p["filter"]["lp"]["cutoff_hz"]
        assert 50.0 <= cutoff <= 10_000.0
        res = p["filter"]["lp"]["resonance"]
        assert 0.0 <= res <= 1.0


def test_sample_dataset_config_deterministic():
    a = list(sample_dataset_config(n_samples=50, seed=7))
    b = list(sample_dataset_config(n_samples=50, seed=7))
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x.midi_pitches == y.midi_pitches
        assert x.n_voices == y.n_voices
        assert x.params_canonical == y.params_canonical


def test_sample_dataset_config_different_seeds_differ():
    a = list(sample_dataset_config(n_samples=20, seed=1))
    b = list(sample_dataset_config(n_samples=20, seed=2))
    diffs = sum(1 for x, y in zip(a, b) if x.params_canonical != y.params_canonical)
    assert diffs > 5  # most should differ


def test_sample_dataset_config_cutoff_log_distributed():
    items = list(sample_dataset_config(n_samples=2000, seed=42))
    cutoffs = [it.params_canonical["params"]["filter"]["lp"]["cutoff_hz"] for it in items]
    # Mean of log(cutoff) should be near the midpoint of [log 50, log 10000] for uniform-on-log.
    log_mean = sum(math.log(c) for c in cutoffs) / len(cutoffs)
    log_midpoint = (math.log(50.0) + math.log(10_000.0)) / 2.0
    assert abs(log_mean - log_midpoint) < 0.5, "cutoff is not log-distributed"


@pytest.mark.slow
def test_generate_dataset_script_smoke(tmp_path: Path):
    out_dir = tmp_path / "ds"
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "generate_subtractive_dataset.py"
    res = subprocess.run(
        [sys.executable, str(script),
         "--n-samples", "5", "--seed", "0", "--out-dir", str(out_dir)],
        check=True, capture_output=True, text=True,
    )
    print(res.stdout)
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "labels.jsonl").exists()
    samples = sorted((out_dir / "samples").glob("*.wav"))
    assert len(samples) == 5
    # Confirm one wav decodes.
    audio, sr = sf.read(str(samples[0]))
    assert sr == 44100
    assert audio.size > 0
    # Confirm labels.jsonl line count matches.
    lines = (out_dir / "labels.jsonl").read_text().strip().splitlines()
    assert len(lines) == 5
    # Manifest shape.
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["n_samples"] == 5
    assert manifest["schema_version"] == "0.1"
    assert manifest["sample_rate"] == 44100


def _build_mini_dataset(tmp_path: Path, n: int = 4) -> Path:
    """Build a tiny on-disk dataset for ToneGenerationDataset tests."""
    sample_rate = 44_100
    total_duration_s = 1.2
    n_samples_audio = int(sample_rate * total_duration_s)

    out_dir = tmp_path / "mini_ds"
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Crude audio: silence except a flat 0.5 in the middle (covers the
    # 100-400ms slice that ToneGenerationDataset extracts).
    audio = np.zeros(n_samples_audio, dtype=np.float32)
    start = int(sample_rate * 0.05)
    stop = int(sample_rate * 0.6)
    audio[start:stop] = 0.5

    pitches_per_item = [
        [60],
        [60, 64],
        [60, 64, 67],
        [48, 72],
    ]

    labels_path = out_dir / "labels.jsonl"
    with labels_path.open("w") as labels_f:
        for idx in range(n):
            pitches = pitches_per_item[idx % len(pitches_per_item)]
            shape = SHAPE_LABELS[idx % len(SHAPE_LABELS)]
            params_canonical = build_canonical_instance(
                shape=shape, cutoff_hz=1000.0, resonance=0.3
            )
            wav_path = samples_dir / f"{idx:06d}.wav"
            sf.write(str(wav_path), audio, sample_rate, subtype="PCM_16")
            labels_f.write(
                json.dumps(
                    {
                        "idx": idx,
                        "params_canonical": params_canonical,
                        "midi_pitches": pitches,
                        "n_voices": len(pitches),
                    }
                )
                + "\n"
            )

    manifest = {
        "n_samples": n,
        "n_samples_produced": n,
        "seed": 0,
        "schema_version": "0.1",
        "sample_rate": sample_rate,
        "total_duration_s": total_duration_s,
        "renderer_git_sha": "test",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out_dir


def test_tone_generation_dataset_shapes(tmp_path: Path):
    out_dir = _build_mini_dataset(tmp_path, n=4)
    ds = ToneGenerationDataset(out_dir)
    assert len(ds) == 4
    mel, pitch_multihot, target = ds[0]

    # mel shape (1, 128, ≥25)
    assert isinstance(mel, torch.Tensor)
    assert mel.dtype == torch.float32
    assert mel.ndim == 3
    assert mel.shape[0] == 1
    assert mel.shape[1] == 128
    assert mel.shape[2] >= 25

    # pitch_multihot shape (88,), dtype float32, sum >= 1
    assert isinstance(pitch_multihot, torch.Tensor)
    assert pitch_multihot.dtype == torch.float32
    assert pitch_multihot.shape == (88,)
    assert float(pitch_multihot.sum()) >= 1.0

    # target dict
    assert isinstance(target, dict)
    assert "shape_label" in target
    assert "cutoff_norm" in target
    assert "resonance" in target
    assert isinstance(target["shape_label"], int)
    assert isinstance(target["cutoff_norm"], float)
    assert isinstance(target["resonance"], float)
    assert 0.0 <= target["cutoff_norm"] <= 1.0
    assert 0.0 <= target["resonance"] <= 1.0


def test_tone_generation_dataset_pitch_multihot_correct(tmp_path: Path):
    out_dir = _build_mini_dataset(tmp_path, n=4)
    ds = ToneGenerationDataset(out_dir)
    # _build_mini_dataset cycles through pitches_per_item:
    expected = [
        [60],
        [60, 64],
        [60, 64, 67],
        [48, 72],
    ]
    for idx, pitches in enumerate(expected):
        _mel, pitch_multihot, _target = ds[idx]
        assert pitch_multihot.shape == (88,)
        # Indices set should correspond to MIDI pitches - 21 (A0 == 21).
        for p in pitches:
            assert pitch_multihot[p - 21] == 1.0, (
                f"expected pitch {p} (idx {p - 21}) to be set"
            )
        # Total ones equals number of distinct pitches.
        assert int(pitch_multihot.sum().item()) == len(set(pitches))


def test_tone_generation_dataset_caches_mel(tmp_path: Path):
    ds_dir = _build_mini_dataset(tmp_path, n=2)
    ds = ToneGenerationDataset(ds_dir)
    # First access populates cache.
    mel1, pitch1, _ = ds[0]
    assert 0 in ds._cache  # type: ignore[attr-defined]
    # Second access returns same tensor (cache hit).
    mel2, _, _ = ds[0]
    assert mel1 is mel2  # cache returns the same object


def test_tone_generation_dataset_init_fails_on_missing_wav(tmp_path: Path):
    ds_dir = _build_mini_dataset(tmp_path, n=2)
    # Delete one WAV to simulate corrupt dataset.
    (ds_dir / "samples" / "000001.wav").unlink()
    with pytest.raises(FileNotFoundError):
        ToneGenerationDataset(ds_dir)
