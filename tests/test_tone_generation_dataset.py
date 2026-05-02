import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import soundfile as sf

from audio_analysis_mcp.research.tone_generation.dataset import (
    DatasetItem,
    sample_dataset_config,
)
from audio_analysis_mcp.research.tone_generation.schema_io import (
    SHAPE_LABELS,
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
