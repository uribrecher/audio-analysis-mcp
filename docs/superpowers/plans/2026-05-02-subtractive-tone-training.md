# Subtractive Tone-Generation Training Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the MVP slice of the subtractive tone-generation training pipeline — SignalFlow renderer → Sobol-sampled synthetic dataset → conditioned CNN → eval with round-trip mel-cosine + schema-validation gate. Predicts three free synth params (`osc.1.shape`, `filter.lp.cutoff_hz`, `filter.lp.resonance`) from a sustain-region log-mel spectrogram, conditioned on ground-truth played pitches.

**Architecture:** Three offline scripts (`generate_subtractive_dataset.py`, `train_tone_generation.py`, `eval_tone_generation.py`) plus a shared module package at `src/audio_analysis_mcp/research/tone_generation/`. The package contains pure-logic modules: `renderer.py` (SignalFlow chord renderer), `dataset.py` (Sobol sampler + torch Dataset with on-the-fly mel-spec), `model.py` (small custom CNN with pitch-conditioning concat), `schema_io.py` (canonical-schema constants, normalization, validation). The plan follows TDD where unit-testable, scratch-first verification for SignalFlow / PyTorch-MPS assumptions, and integration smoke tests for the full loop.

**Tech Stack:** Python 3.11, `torch` (already a dep, MPS via standard macOS wheel), `numpy`, `scipy.stats.qmc` (Sobol), `librosa` (mel-spec — already a dep), `soundfile`, `signalflow` (NEW dep — used at runtime for dataset gen + round-trip eval, not just dev), `jsonschema` (NEW dep — schema validation). No `keyboards-mcp` integration in MVP.

**Reference spec:** `reverse-synth-research/docs/superpowers/specs/2026-05-02-subtractive-tone-training-mvp.md`. **DO NOT plan from the backlog file.**

**Reference schema:** `reverse-synth-research/parameter-mapping/subtractive.schema.json` (extends `synth-base.schema.json`). The schema is the source of truth — do not duplicate it in this repo.

---

## Pre-flight

Before starting Task 1, verify:

1. **Branch state:** `audio-analysis-mcp` should already be on the feature branch `subtractive-tone-training-mvp` (created when this plan file was committed). Confirm:
   ```bash
   cd ~/test/sounds-and-recreation/audio-analysis-mcp
   git branch --show-current   # expect: subtractive-tone-training-mvp
   git log --oneline -1        # expect: the commit that added this plan file
   ```
   If you are on `main` instead, check the branch out: `git checkout subtractive-tone-training-mvp`.
2. **Schema availability:** `reverse-synth-research/parameter-mapping/subtractive.schema.json` exists and is on `main`. Verify:
   ```bash
   ls ~/test/sounds-and-recreation/reverse-synth-research/parameter-mapping/subtractive.schema.json
   ls ~/test/sounds-and-recreation/reverse-synth-research/parameter-mapping/synth-base.schema.json
   ```
3. **Working directory:** all subsequent commands assume cwd = `audio-analysis-mcp/` unless stated otherwise.

---

## File Map

**Create:**
- `scratch/explore_subtractive_renderer.py` (T1 — scratch verification)
- `scratch/explore_pytorch_mps.py` (T2 — scratch verification)
- `src/audio_analysis_mcp/research/__init__.py`
- `src/audio_analysis_mcp/research/tone_generation/__init__.py`
- `src/audio_analysis_mcp/research/tone_generation/schema_io.py`
- `src/audio_analysis_mcp/research/tone_generation/renderer.py`
- `src/audio_analysis_mcp/research/tone_generation/dataset.py`
- `src/audio_analysis_mcp/research/tone_generation/model.py`
- `src/audio_analysis_mcp/research/tone_generation/README.md`
- `scripts/generate_subtractive_dataset.py`
- `scripts/train_tone_generation.py`
- `scripts/eval_tone_generation.py`
- `tests/test_tone_generation_schema_io.py`
- `tests/test_tone_generation_renderer.py`
- `tests/test_tone_generation_dataset.py`
- `tests/test_tone_generation_model.py`
- `tests/test_tone_generation_smoke.py`

**Modify:**
- `pyproject.toml` — add `signalflow`, `jsonschema` to `dependencies`
- `.gitignore` — ignore `scratch/tone_gen_dataset/`, `scratch/tone_gen_checkpoints/` (default output paths)

---

## Conventions

- Repo uses `mypy strict = true`. Use `numpy.typing.NDArray[np.float32]` for ndarray annotations. PyTorch tensor types: `torch.Tensor` only — do not annotate dtype/shape in types.
- All modules use `from audio_analysis_mcp.research.tone_generation.<module> import ...`.
- Pure-logic functions take/return numpy arrays + plain dicts / dataclasses. Disk I/O lives in the scripts (`scripts/*.py`) and in `ToneGenerationDataset.__getitem__` only.
- Scratch scripts live in `scratch/` per repo convention. **No inline `python -c "..."` ever** — always run scratch files. This rule extends to debugging during plan execution.
- Run `uv run pytest -m "not slow" -q` and `uv run mypy src/` clean before each commit.
- All emitted canonical instances must validate against `subtractive.schema.json` — assert this in tests.

---

## Task 1: Scratch — verify polyphonic SignalFlow

> **Why scratch first:** the MVP renderer needs `SquareOscillator` / `SawOscillator` / `TriangleOscillator`, a low-pass filter node, and parallel-summed voices in one graph. The existing `scratch/explore_signalflow.py` only verified `SineOscillator` + `ADSREnvelope`. We must confirm the rest of the API behaves as the spec assumes BEFORE writing `renderer.py`.

**Files:**
- Create: `scratch/explore_subtractive_renderer.py`

- [ ] **Step 1: Write the scratch script**

Create `scratch/explore_subtractive_renderer.py`:

```python
"""Probe SignalFlow's API for the subtractive renderer:

1. Confirm SquareOscillator / SawOscillator / TriangleOscillator exist (or find the right names).
2. Confirm a low-pass filter node exists (SVFilter, OnePoleLowPassFilter, ...).
3. Render a single 440 Hz saw through an LP at cutoff=2 kHz, resonance=0.5,
   confirm the audio looks reasonable (peak ~0.5-0.95, no NaN, harmonics
   below cutoff present, harmonics above cutoff attenuated).
4. Sum 3 voices (440, 554, 659 Hz — a major triad) in one graph, render,
   confirm peak after-sum stays in [0.5, 1.0] after manual normalization.

If any of these fail, FIX the spec / plan before dispatching Task 5 (renderer).

Run: uv run python scratch/explore_subtractive_renderer.py
"""

from __future__ import annotations

import numpy as np
import signalflow as sf_lib

print("signalflow module:", sf_lib.__file__)
osc_attrs = sorted(a for a in dir(sf_lib) if "scillator" in a.lower())
filter_attrs = sorted(
    a for a in dir(sf_lib) if "ilter" in a.lower() or "passfilter" in a.lower()
)
print("Oscillator-like attrs:", osc_attrs)
print("Filter-like attrs:", filter_attrs[:20])


def render_voice(
    graph: sf_lib.AudioGraph,
    osc_class: type,
    frequency: float,
    cutoff_hz: float,
    resonance: float,
    duration_s: float,
    sr: int,
) -> np.ndarray:
    osc = osc_class(frequency=frequency)
    # gate=1 required: SignalFlow 0.5.3 ADSREnvelope defaults gate=0 → silence.
    env = sf_lib.ADSREnvelope(attack=0.05, decay=0.10, sustain=0.7, release=0.10, gate=1)
    # Try common LP names; fall back as needed.
    lp_class_name = None
    for candidate in ("SVFilter", "OnePoleLowPassFilter", "LowPassFilter"):
        if hasattr(sf_lib, candidate):
            lp_class_name = candidate
            break
    if lp_class_name is None:
        raise RuntimeError("No known LP filter class on signalflow module")
    lp_class = getattr(sf_lib, lp_class_name)
    print(f"  using LP: {lp_class_name}")
    # Try lp(input, cutoff, resonance) — exact ctor varies; print help.
    print(f"  {lp_class_name}.__init__ signature:")
    help(lp_class.__init__)
    # Best-effort instantiation; tweak per actual signature once help() above prints.
    lp = lp_class(osc, cutoff_hz, resonance)
    out = lp * env
    out.play()
    return None  # filled in after we render the graph below


def main() -> None:
    sr = 44100
    duration_s = 1.0
    n_samples = int(sr * duration_s)

    # Discover oscillator class names — print attrs above tells us what's there.
    # Typical signalflow names in 0.4.x: SineOscillator, SquareOscillator,
    # SawOscillator, TriangleOscillator. Fail if missing.
    needed = ["SineOscillator", "SquareOscillator", "SawOscillator", "TriangleOscillator"]
    for name in needed:
        if not hasattr(sf_lib, name):
            print(f"  WARNING: missing {name} on signalflow — find the right name")

    # 1) Single voice through LP
    graph = sf_lib.AudioGraph(output_device=None, start=False)
    SawOsc = getattr(sf_lib, "SawOscillator", None) or getattr(sf_lib, "SineOscillator")
    print("\n=== Single saw voice through LP ===")
    render_voice(graph, SawOsc, 440.0, 2000.0, 0.5, duration_s, sr)
    buf = graph.render_to_new_buffer(num_frames=n_samples)
    audio = np.asarray(buf.data, dtype=np.float32)
    audio = audio[0] if audio.ndim == 2 else audio
    print(f"  audio shape: {audio.shape}, peak: {np.abs(audio).max():.3f}, RMS: {np.sqrt((audio ** 2).mean()):.3f}")
    assert audio.size == n_samples, f"unexpected sample count: {audio.size} vs {n_samples}"
    assert not np.any(np.isnan(audio)), "audio contains NaN"
    assert not np.any(np.isinf(audio)), "audio contains Inf"

    # 2) Sum 3 voices (parallel in one graph)
    print("\n=== Sum 3 saw voices (major triad) ===")
    graph2 = sf_lib.AudioGraph(output_device=None, start=False)
    voices = []
    for f in (440.0, 554.37, 659.26):
        osc = SawOsc(frequency=f)
        env = sf_lib.ADSREnvelope(attack=0.05, decay=0.10, sustain=0.7, release=0.10, gate=1)
        # Skip filter for the sum test to isolate the chord-summation behavior.
        voices.append(osc * env)
    summed = voices[0] + voices[1] + voices[2]
    summed.play()
    buf2 = graph2.render_to_new_buffer(num_frames=n_samples)
    audio2 = np.asarray(buf2.data, dtype=np.float32)
    audio2 = audio2[0] if audio2.ndim == 2 else audio2
    peak2 = np.abs(audio2).max()
    print(f"  summed shape: {audio2.shape}, peak: {peak2:.3f}")
    assert audio2.size == n_samples
    # Peak before normalization can exceed 1.0 — that's expected. We normalize after.
    audio2_norm = audio2 / max(peak2, 1e-6) * 0.95
    print(f"  normalized peak: {np.abs(audio2_norm).max():.3f}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the scratch**

```bash
uv run python scratch/explore_subtractive_renderer.py
```

Expected: prints oscillator/filter attr lists, prints LP class help signature, then "All checks passed." If any oscillator name is wrong or the LP ctor signature differs, **stop and fix the script** before continuing — do not proceed to Task 5 with broken assumptions.

- [ ] **Step 3: Commit**

```bash
git add scratch/explore_subtractive_renderer.py
git commit -m "scratch: probe SignalFlow polyphonic + LP filter for subtractive renderer"
```

> **Human gate:** review the script's output. If oscillator names differ from `SquareOscillator`/`SawOscillator`/`TriangleOscillator`, or if the LP ctor differs from `lp_class(input, cutoff, resonance)`, update the renderer plan (Task 5) accordingly.

---

## Task 2: Scratch — verify PyTorch MPS on M3 Pro

> **Why scratch first:** the MVP spec specifies `device = "mps" if available else "cpu"` (Apple Silicon GPU). MPS has known gotchas with deterministic-mode flags, certain ops (older PyTorch versions), and tensor dtypes. Confirm a tiny CNN forward + backward runs on MPS without errors before writing `model.py` and the train loop.

**Files:**
- Create: `scratch/explore_pytorch_mps.py`

- [ ] **Step 1: Write the scratch script**

Create `scratch/explore_pytorch_mps.py`:

```python
"""Probe PyTorch MPS on Apple Silicon (M3 Pro):

1. Confirm torch.backends.mps.is_available() and is_built().
2. Run a tiny Conv2d → BN → ReLU → MaxPool → AdaptiveAvgPool1d → Linear
   forward + backward on mps; check for runtime errors.
3. Time a 50-iteration training loop on a synthetic batch — sanity
   check (should be sub-second on M3 Pro).

Run: uv run python scratch/explore_pytorch_mps.py
"""

from __future__ import annotations

import time
import torch
import torch.nn as nn

print("torch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())

if not torch.backends.mps.is_available():
    print("MPS unavailable on this machine — cpu-only fallback.")
    device = torch.device("cpu")
else:
    device = torch.device("mps")
print("device:", device)


class TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(32 * 64 * 15, 4)  # for 128x30 input after one pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pool(torch.relu(self.bn1(self.conv1(x))))
        h = h.flatten(1)
        return self.fc(h)


def main() -> None:
    model = TinyCNN().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(8, 1, 128, 30, device=device)
    y = torch.randint(0, 4, (8,), device=device)
    crit = nn.CrossEntropyLoss()

    # Single fwd/bwd
    out = model(x)
    print("forward output shape:", out.shape, "dtype:", out.dtype)
    loss = crit(out, y)
    print("loss before:", loss.item())
    loss.backward()
    optim.step()

    # 50-step loop, time it
    t0 = time.time()
    for _ in range(50):
        optim.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        optim.step()
    elapsed = time.time() - t0
    print(f"50 steps on {device}: {elapsed:.3f}s ({1000 * elapsed / 50:.1f} ms/step)")
    print("loss after:", loss.item())
    print("\nMPS smoke test passed.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the scratch**

```bash
uv run python scratch/explore_pytorch_mps.py
```

Expected on M3 Pro: `MPS available: True`, `device: mps`, forward output shape `torch.Size([8, 4])`, 50 steps under 1 s. Loss should decrease from initial → after.

- [ ] **Step 3: Commit**

```bash
git add scratch/explore_pytorch_mps.py
git commit -m "scratch: verify PyTorch MPS forward+backward on M3 Pro"
```

---

## Task 3: Add deps + scaffold the research subpackage

**Files:**
- Modify: `pyproject.toml`
- Create: `src/audio_analysis_mcp/research/__init__.py`
- Create: `src/audio_analysis_mcp/research/tone_generation/__init__.py`
- Modify: `.gitignore`

- [ ] **Step 1: Add `signalflow` and `jsonschema` to dependencies**

Edit `pyproject.toml`. In the `[project]` `dependencies` list, add:

```toml
  "signalflow>=0.4.0",
  "jsonschema>=4.21",
```

Also add the new modules to the existing `[[tool.mypy.overrides]]` `ignore_missing_imports` block:

```toml
[[tool.mypy.overrides]]
module = [
  "librosa.*",
  "sounddevice.*",
  "soundfile.*",
  "scipy.*",
  "demucs.*",
  "tqdm.*",
  "basic_pitch.*",
  "pretty_midi.*",
  "signalflow.*",
  "jsonschema.*",
]
ignore_missing_imports = true
```

- [ ] **Step 2: Sync dependencies**

```bash
uv sync --dev
```

Expected: resolves and installs `signalflow` and `jsonschema`. No errors.

- [ ] **Step 3: Create the package scaffold**

```bash
mkdir -p src/audio_analysis_mcp/research/tone_generation
```

Create `src/audio_analysis_mcp/research/__init__.py`:

```python
"""Research subpackage. Houses experimental ML research code that is not part of the live MCP server. Each subpackage is independent."""
```

Create `src/audio_analysis_mcp/research/tone_generation/__init__.py`:

```python
"""Subtractive tone-generation training pipeline (MVP).

See:
- spec: reverse-synth-research/docs/superpowers/specs/2026-05-02-subtractive-tone-training-mvp.md
- plan: audio-analysis-mcp/docs/superpowers/plans/2026-05-02-subtractive-tone-training.md
"""
```

- [ ] **Step 4: Update .gitignore**

Append to `.gitignore`:

```
# Tone-generation training (default output paths — datasets and checkpoints are large)
scratch/tone_gen_dataset/
scratch/tone_gen_checkpoints/
```

- [ ] **Step 5: Verify mypy still clean**

```bash
uv run mypy src/
```

Expected: clean (the new empty `__init__.py` files compile fine; new ignored modules aren't imported yet).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock .gitignore src/audio_analysis_mcp/research/
git commit -m "tone-generation: add deps (signalflow, jsonschema) + scaffold research subpackage"
```

---

## Task 4: schema_io — constants, builder, normalize/denormalize, validate

**Files:**
- Create: `src/audio_analysis_mcp/research/tone_generation/schema_io.py`
- Create: `tests/test_tone_generation_schema_io.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tone_generation_schema_io.py`:

```python
import math

import pytest

from audio_analysis_mcp.research.tone_generation.schema_io import (
    BASELINE_AMP_ADSR,
    SHAPE_LABELS,
    build_canonical_instance,
    denormalize_predictions,
    normalize_params,
    validate_canonical,
)


def test_baseline_amp_adsr_is_valid_adsr_dict():
    keys = {"attack_ms", "decay_ms", "sustain", "release_ms"}
    assert set(BASELINE_AMP_ADSR.keys()) == keys
    assert BASELINE_AMP_ADSR["attack_ms"] >= 0
    assert 0.0 <= BASELINE_AMP_ADSR["sustain"] <= 1.0


def test_shape_labels_match_schema():
    assert SHAPE_LABELS == ["sine", "saw", "square", "triangle"]


def test_build_canonical_instance_shape():
    inst = build_canonical_instance(shape="saw", cutoff_hz=2000.0, resonance=0.5)
    assert inst["schema_version"] == "0.1"
    assert inst["engine"] == "subtractive"
    p = inst["params"]
    assert p["osc"]["1"]["shape"] == "saw"
    assert p["osc"]["1"]["level"] == 1.0
    assert p["filter"]["lp"]["cutoff_hz"] == 2000.0
    assert p["filter"]["lp"]["resonance"] == 0.5
    assert p["filter"]["lp"]["envelope_amount"] == 0.0
    assert p["voice"]["mode"] == "poly"
    assert p["envelope"]["amp"] == BASELINE_AMP_ADSR


def test_build_canonical_instance_validates():
    inst = build_canonical_instance(shape="square", cutoff_hz=440.0, resonance=0.0)
    validate_canonical(inst)  # raises on invalid


@pytest.mark.parametrize("shape", SHAPE_LABELS)
@pytest.mark.parametrize("cutoff_hz", [50.0, 100.0, 1000.0, 5000.0, 10000.0])
@pytest.mark.parametrize("resonance", [0.0, 0.5, 1.0])
def test_build_canonical_instance_parametric(shape, cutoff_hz, resonance):
    inst = build_canonical_instance(shape=shape, cutoff_hz=cutoff_hz, resonance=resonance)
    validate_canonical(inst)


def test_validate_canonical_rejects_invalid():
    bad = {"schema_version": "0.1", "engine": "subtractive", "params": {}}
    with pytest.raises(Exception):
        validate_canonical(bad)


def test_normalize_denormalize_roundtrip_cutoff():
    cutoff_hz = 2000.0
    norm = math.log(cutoff_hz / 50.0) / math.log(10000.0 / 50.0)
    out = normalize_params(shape="saw", cutoff_hz=cutoff_hz, resonance=0.5)
    assert math.isclose(out["cutoff_norm"], norm, rel_tol=1e-6)
    assert out["resonance"] == 0.5
    assert out["shape_label"] == 1  # saw == 1


def test_denormalize_returns_canonical_instance():
    inst = denormalize_predictions(
        shape_label=1, cutoff_norm=0.5, resonance=0.42, midi_pitches=[60, 64, 67]
    )
    validate_canonical(inst)
    assert inst["params"]["osc"]["1"]["shape"] == "saw"
    expected_cutoff = math.exp(
        math.log(50.0) + 0.5 * (math.log(10000.0) - math.log(50.0))
    )
    assert math.isclose(
        inst["params"]["filter"]["lp"]["cutoff_hz"], expected_cutoff, rel_tol=1e-4
    )


def test_normalize_then_denormalize_roundtrip_continuous():
    out = normalize_params(shape="square", cutoff_hz=2000.0, resonance=0.6)
    inst = denormalize_predictions(
        shape_label=out["shape_label"],
        cutoff_norm=out["cutoff_norm"],
        resonance=out["resonance"],
        midi_pitches=[60],
    )
    p = inst["params"]
    assert p["osc"]["1"]["shape"] == "square"
    assert math.isclose(p["filter"]["lp"]["cutoff_hz"], 2000.0, rel_tol=1e-4)
    assert math.isclose(p["filter"]["lp"]["resonance"], 0.6, abs_tol=1e-9)
```

- [ ] **Step 2: Run tests, expect ImportError**

```bash
uv run pytest tests/test_tone_generation_schema_io.py -v
```

Expected: FAIL with `ImportError: cannot import name 'BASELINE_AMP_ADSR'`.

- [ ] **Step 3: Implement schema_io.py**

Create `src/audio_analysis_mcp/research/tone_generation/schema_io.py`:

```python
"""Canonical-schema I/O for subtractive tone-generation MVP.

The schema source of truth is `subtractive.schema.json` in the sibling
`reverse-synth-research/parameter-mapping/` directory. We resolve the path
once at import; an env var override (TONE_GEN_SCHEMA_DIR) handles non-monorepo
layouts.
"""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import jsonschema

# ---- Schema-locked constants ------------------------------------------------

SHAPE_LABELS: list[str] = ["sine", "saw", "square", "triangle"]
"""Free shapes the MVP predicts. `pulse` deliberately excluded — see backlog."""

CUTOFF_HZ_MIN = 50.0
CUTOFF_HZ_MAX = 10_000.0
"""Free-cutoff sampling bounds. Schema's hard bounds are 20 / 20_000."""

BASELINE_AMP_ADSR: dict[str, float] = {
    "attack_ms": 10.0,
    "decay_ms": 200.0,
    "sustain": 0.7,
    "release_ms": 200.0,
}
"""Frozen amp ADSR for MVP. Pad-style preset; identical across all renders."""

# ---- Schema loading ---------------------------------------------------------

_DEFAULT_SCHEMA_DIR = (
    Path(__file__).resolve().parents[5]
    / "reverse-synth-research"
    / "parameter-mapping"
)


def _schema_dir() -> Path:
    override = os.environ.get("TONE_GEN_SCHEMA_DIR")
    if override:
        return Path(override)
    return _DEFAULT_SCHEMA_DIR


@lru_cache(maxsize=1)
def _load_validator() -> jsonschema.Validator:
    schema_dir = _schema_dir()
    subtractive_path = schema_dir / "subtractive.schema.json"
    base_path = schema_dir / "synth-base.schema.json"
    if not subtractive_path.exists():
        raise FileNotFoundError(
            f"subtractive.schema.json not found at {subtractive_path} — "
            "set TONE_GEN_SCHEMA_DIR if your monorepo layout differs."
        )
    with subtractive_path.open() as f:
        subtractive_schema = json.load(f)
    with base_path.open() as f:
        base_schema = json.load(f)
    # Local resolver so $ref to synth-base.schema.json resolves to the loaded base.
    store = {
        subtractive_schema["$id"]: subtractive_schema,
        base_schema["$id"]: base_schema,
        "synth-base.schema.json": base_schema,
    }
    resolver = jsonschema.RefResolver.from_schema(subtractive_schema, store=store)
    cls = jsonschema.validators.validator_for(subtractive_schema)
    cls.check_schema(subtractive_schema)
    return cls(subtractive_schema, resolver=resolver)


def validate_canonical(instance: dict[str, Any]) -> None:
    """Validate instance against subtractive.schema.json. Raises on invalid."""
    _load_validator().validate(instance)


# ---- Canonical instance construction ---------------------------------------


def build_canonical_instance(
    *, shape: str, cutoff_hz: float, resonance: float
) -> dict[str, Any]:
    """Build a schema-conformant subtractive instance from the 3 free MVP params."""
    if shape not in SHAPE_LABELS:
        raise ValueError(f"shape must be one of {SHAPE_LABELS}, got {shape!r}")
    return {
        "schema_version": "0.1",
        "engine": "subtractive",
        "params": {
            "osc": {
                "1": {
                    "shape": shape,
                    "level": 1.0,
                    "octave": 0,
                    "detune_cents": 0,
                }
            },
            "filter": {
                "lp": {
                    "cutoff_hz": float(cutoff_hz),
                    "resonance": float(resonance),
                    "envelope_amount": 0.0,
                    "key_tracking": 0.0,
                    "drive": 0.0,
                }
            },
            "envelope": {"amp": dict(BASELINE_AMP_ADSR)},
            "voice": {"mode": "poly"},
            # No lfo block — schema makes lfo required for subtractive, so we MUST
            # supply one. MVP renders without modulation; lfo.depth = 0 makes it
            # acoustically inert. Modulation expert will own this slot for real.
            "lfo": _inert_lfo(),
        },
    }


def _inert_lfo() -> dict[str, Any]:
    """LFO with depth=0 — schema-required slot, acoustically silent."""
    return {
        "1": {
            "rate_hz": 1.0,
            "shape": "sine",
            "depth": 0.0,
            "target": "filter.cutoff",
        }
    }


# ---- Normalize / denormalize ------------------------------------------------


def _cutoff_to_norm(cutoff_hz: float) -> float:
    return (math.log(cutoff_hz) - math.log(CUTOFF_HZ_MIN)) / (
        math.log(CUTOFF_HZ_MAX) - math.log(CUTOFF_HZ_MIN)
    )


def _norm_to_cutoff(cutoff_norm: float) -> float:
    log_lo = math.log(CUTOFF_HZ_MIN)
    log_hi = math.log(CUTOFF_HZ_MAX)
    return math.exp(log_lo + cutoff_norm * (log_hi - log_lo))


def normalize_params(
    *, shape: str, cutoff_hz: float, resonance: float
) -> dict[str, float | int]:
    """Canonical params → training-time normalized scalars."""
    return {
        "shape_label": SHAPE_LABELS.index(shape),
        "cutoff_norm": _cutoff_to_norm(cutoff_hz),
        "resonance": float(resonance),
    }


def denormalize_predictions(
    *,
    shape_label: int,
    cutoff_norm: float,
    resonance: float,
    midi_pitches: list[int],
) -> dict[str, Any]:
    """Predictions → schema-conformant canonical instance."""
    cutoff_norm = max(0.0, min(1.0, cutoff_norm))
    resonance = max(0.0, min(1.0, resonance))
    shape = SHAPE_LABELS[shape_label]
    cutoff_hz = _norm_to_cutoff(cutoff_norm)
    return build_canonical_instance(
        shape=shape, cutoff_hz=cutoff_hz, resonance=resonance
    )
```

> Note: `_inert_lfo()` is required because the canonical schema makes `lfo` a required field for subtractive engines (`"required": ["osc", "filter", "lfo"]`). The MVP renders without modulation, so we emit a depth-0 LFO that is acoustically inert. The modulation expert will own this slot for real predictions later.

- [ ] **Step 4: Run tests, expect pass**

```bash
uv run pytest tests/test_tone_generation_schema_io.py -v && uv run mypy src/
```

Expected: all tests pass, mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/audio_analysis_mcp/research/tone_generation/schema_io.py tests/test_tone_generation_schema_io.py
git commit -m "tone-generation: schema_io — constants, builder, normalize/denormalize, validate"
```

---

## Task 5: renderer — SubtractiveRenderer + render_chord

> **Pre-validation:** Task 1 scratch must have passed. The exact oscillator class names and LP filter ctor signature found in the scratch determine the implementation here. If Task 1's scratch revealed different names, adjust the imports below.

**Files:**
- Create: `src/audio_analysis_mcp/research/tone_generation/renderer.py`
- Create: `tests/test_tone_generation_renderer.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tone_generation_renderer.py`:

```python
import numpy as np
import pytest

from audio_analysis_mcp.research.tone_generation.renderer import render_chord
from audio_analysis_mcp.research.tone_generation.schema_io import (
    build_canonical_instance,
)


SR = 44100
TOTAL_DURATION_S = 1.2  # 100 ms attack + 1000 ms hold + 100 ms release


def _params(shape: str = "saw", cutoff_hz: float = 2000.0, resonance: float = 0.5):
    return build_canonical_instance(
        shape=shape, cutoff_hz=cutoff_hz, resonance=resonance
    )


def test_render_chord_single_voice_shape_and_dtype():
    audio = render_chord(
        params_canonical=_params(),
        midi_pitches=[69],  # A4
        sample_rate=SR,
        total_duration_s=TOTAL_DURATION_S,
    )
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    expected_len = int(SR * TOTAL_DURATION_S)
    assert audio.size == expected_len
    assert not np.any(np.isnan(audio))
    assert not np.any(np.isinf(audio))


def test_render_chord_normalized_peak():
    audio = render_chord(
        params_canonical=_params(), midi_pitches=[69], sample_rate=SR, total_duration_s=TOTAL_DURATION_S
    )
    peak = np.abs(audio).max()
    assert 0.85 <= peak <= 1.0, f"peak {peak} out of [0.85, 1.0]"


@pytest.mark.parametrize("n_voices", [1, 2, 3])
def test_render_chord_polyphony(n_voices: int):
    pitches = [60, 64, 67][:n_voices]
    audio = render_chord(
        params_canonical=_params(),
        midi_pitches=pitches,
        sample_rate=SR,
        total_duration_s=TOTAL_DURATION_S,
    )
    assert audio.size == int(SR * TOTAL_DURATION_S)
    assert not np.any(np.isnan(audio))
    assert np.abs(audio).max() <= 1.0


def test_render_chord_different_shapes_produce_different_audio():
    a = render_chord(
        params_canonical=_params(shape="sine"),
        midi_pitches=[69], sample_rate=SR, total_duration_s=TOTAL_DURATION_S,
    )
    b = render_chord(
        params_canonical=_params(shape="saw"),
        midi_pitches=[69], sample_rate=SR, total_duration_s=TOTAL_DURATION_S,
    )
    assert not np.allclose(a, b, atol=0.05)


def test_render_chord_rejects_empty_pitches():
    with pytest.raises(ValueError):
        render_chord(
            params_canonical=_params(), midi_pitches=[], sample_rate=SR, total_duration_s=TOTAL_DURATION_S
        )
```

- [ ] **Step 2: Run tests, expect ImportError**

```bash
uv run pytest tests/test_tone_generation_renderer.py -v
```

Expected: FAIL with `ImportError: cannot import name 'render_chord'`.

- [ ] **Step 3: Implement renderer.py**

Create `src/audio_analysis_mcp/research/tone_generation/renderer.py`. **Reference the Task 1 scratch script for the exact SignalFlow API** — adjust ctor signatures if the scratch found different names:

```python
"""SignalFlow-based subtractive renderer for the tone-generation training MVP.

Single voice topology: osc → LP filter → fixed amp ADSR → output.
Polyphonic chord = sum of N parallel voices in one AudioGraph, normalized.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import signalflow as sf_lib

from audio_analysis_mcp.research.tone_generation.schema_io import BASELINE_AMP_ADSR

_SHAPE_TO_OSC_CLASS: dict[str, str] = {
    "sine": "SineOscillator",
    "saw": "SawOscillator",
    "square": "SquareOscillator",
    "triangle": "TriangleOscillator",
}


def _midi_to_hz(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def _build_voice(
    *,
    shape: str,
    cutoff_hz: float,
    resonance: float,
    midi_pitch: int,
    amp_adsr: dict[str, float],
) -> Any:
    """Build a single voice: osc → LP → amp env. Returns an unrooted SignalFlow node."""
    osc_class_name = _SHAPE_TO_OSC_CLASS[shape]
    if not hasattr(sf_lib, osc_class_name):
        raise RuntimeError(
            f"signalflow has no class {osc_class_name}; "
            "scratch/explore_subtractive_renderer.py needs re-running"
        )
    osc_class = getattr(sf_lib, osc_class_name)
    osc = osc_class(frequency=_midi_to_hz(midi_pitch))
    lp_class = getattr(sf_lib, "SVFilter", None) or getattr(sf_lib, "OnePoleLowPassFilter")
    # SignalFlow's SVFilter ctor: SVFilter(input, filter_type, cutoff, resonance).
    # Fall back to OnePoleLowPassFilter(input, cutoff) if SVFilter is unavailable.
    if hasattr(sf_lib, "SVFilter"):
        filtered = lp_class(osc, "low_pass", cutoff_hz, resonance)
    else:
        filtered = lp_class(osc, cutoff_hz)
    # gate=1 is REQUIRED on every ADSREnvelope: SignalFlow 0.5.3's default is
    # gate=0, which produces silence. Verified empirically by
    # scratch/explore_subtractive_renderer.py test case 3.
    env = sf_lib.ADSREnvelope(
        attack=amp_adsr["attack_ms"] / 1000.0,
        decay=amp_adsr["decay_ms"] / 1000.0,
        sustain=amp_adsr["sustain"],
        release=amp_adsr["release_ms"] / 1000.0,
        gate=1,
    )
    return filtered * env


def render_chord(
    *,
    params_canonical: dict[str, Any],
    midi_pitches: list[int],
    sample_rate: int,
    total_duration_s: float,
) -> npt.NDArray[np.float32]:
    """Render N voices in parallel, summed, peak-normalized to ~0.95.

    Parameters
    ----------
    params_canonical : schema-conformant subtractive instance.
    midi_pitches : one MIDI pitch per voice. Length determines polyphony.
    sample_rate : in Hz.
    total_duration_s : total render length (covers attack + hold + release).
    """
    if not midi_pitches:
        raise ValueError("midi_pitches must be non-empty")
    p = params_canonical["params"]
    shape = p["osc"]["1"]["shape"]
    cutoff_hz = float(p["filter"]["lp"]["cutoff_hz"])
    resonance = float(p["filter"]["lp"]["resonance"])
    amp_adsr = p["envelope"]["amp"]

    graph = sf_lib.AudioGraph(output_device=None, start=False)
    voices = [
        _build_voice(
            shape=shape,
            cutoff_hz=cutoff_hz,
            resonance=resonance,
            midi_pitch=pitch,
            amp_adsr=amp_adsr,
        )
        for pitch in midi_pitches
    ]
    summed = voices[0]
    for v in voices[1:]:
        summed = summed + v
    summed.play()
    n_samples = int(round(sample_rate * total_duration_s))
    buf = graph.render_to_new_buffer(num_frames=n_samples)
    arr = np.asarray(buf.data, dtype=np.float32)
    audio = arr[0] if arr.ndim == 2 else arr
    if audio.size != n_samples:
        # SignalFlow rounding edge case — pad / truncate to expected length.
        audio = np.pad(audio, (0, max(0, n_samples - audio.size)))[:n_samples]
    peak = float(np.abs(audio).max())
    if peak > 1e-6:
        audio = audio * (0.95 / peak)
    return audio.astype(np.float32, copy=False)


# Touch BASELINE_AMP_ADSR so import isn't dead — also provides a quick reference
# for callers that want to use the canonical baseline in a custom render.
_ = BASELINE_AMP_ADSR
```

- [ ] **Step 4: Run tests, expect pass**

```bash
uv run pytest tests/test_tone_generation_renderer.py -v && uv run mypy src/
```

Expected: 6 tests pass, mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/audio_analysis_mcp/research/tone_generation/renderer.py tests/test_tone_generation_renderer.py
git commit -m "tone-generation: renderer — SubtractiveRenderer + render_chord"
```

---

## Task 6: dataset — config sampler

**Files:**
- Create: `src/audio_analysis_mcp/research/tone_generation/dataset.py` (sampler portion)
- Create: `tests/test_tone_generation_dataset.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tone_generation_dataset.py`:

```python
import math

import pytest

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
```

- [ ] **Step 2: Run tests, expect ImportError**

```bash
uv run pytest tests/test_tone_generation_dataset.py -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement the sampler portion of dataset.py**

Create `src/audio_analysis_mcp/research/tone_generation/dataset.py`:

```python
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

import numpy as np
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


@dataclass(frozen=True)
class DatasetItem:
    """One synthetic training sample's full label."""

    params_canonical: dict[str, Any]
    midi_pitches: list[int]
    n_voices: int


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
    """Yield n_samples DatasetItem instances. Deterministic given seed."""
    sobol = qmc.Sobol(d=2, scramble=True, seed=seed)
    cont_samples = sobol.random(n_samples)  # shape (n_samples, 2) — uniform [0, 1)
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
```

- [ ] **Step 4: Run tests, expect pass**

```bash
uv run pytest tests/test_tone_generation_dataset.py -v && uv run mypy src/
```

Expected: 6 tests pass, mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/audio_analysis_mcp/research/tone_generation/dataset.py tests/test_tone_generation_dataset.py
git commit -m "tone-generation: dataset — Sobol+uniform sampler with chord-pitch window"
```

---

## Task 7: Dataset generator script

**Files:**
- Create: `scripts/generate_subtractive_dataset.py`
- Modify: `tests/test_tone_generation_dataset.py` (add a smoke test)

- [ ] **Step 1: Add a smoke test**

Append to `tests/test_tone_generation_dataset.py`:

```python
import json
import subprocess
import sys
from pathlib import Path

import soundfile as sf


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
```

- [ ] **Step 2: Run the smoke test, expect failure**

```bash
uv run pytest tests/test_tone_generation_dataset.py::test_generate_dataset_script_smoke -v -m slow
```

Expected: FAIL — script doesn't exist.

- [ ] **Step 3: Implement the script**

Create `scripts/generate_subtractive_dataset.py`:

```python
"""Generate a synthetic subtractive dataset for tone-generation training.

Usage:
    uv run python scripts/generate_subtractive_dataset.py \\
        --n-samples 10000 --seed 0 --out-dir scratch/tone_gen_dataset

Outputs (under --out-dir):
    samples/{idx:06d}.wav   16-bit PCM, 44.1 kHz mono
    labels.jsonl            one JSON object per sample
    manifest.json           dataset-level metadata
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import soundfile as sf

# Local repo imports — assume cwd = audio-analysis-mcp/ when running.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_analysis_mcp.research.tone_generation.dataset import sample_dataset_config
from audio_analysis_mcp.research.tone_generation.renderer import render_chord


SAMPLE_RATE = 44_100
TOTAL_DURATION_S = 1.2


def _git_sha() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "labels.jsonl"
    manifest_path = out_dir / "manifest.json"

    t0 = time.time()
    with labels_path.open("w") as labels_f:
        for idx, item in enumerate(
            sample_dataset_config(n_samples=args.n_samples, seed=args.seed)
        ):
            audio = render_chord(
                params_canonical=item.params_canonical,
                midi_pitches=item.midi_pitches,
                sample_rate=SAMPLE_RATE,
                total_duration_s=TOTAL_DURATION_S,
            )
            wav_path = samples_dir / f"{idx:06d}.wav"
            sf.write(str(wav_path), audio, SAMPLE_RATE, subtype="PCM_16")
            labels_f.write(
                json.dumps(
                    {
                        "idx": idx,
                        "params_canonical": item.params_canonical,
                        "midi_pitches": item.midi_pitches,
                        "n_voices": item.n_voices,
                    }
                )
                + "\n"
            )
            if (idx + 1) % 500 == 0:
                rate = (idx + 1) / (time.time() - t0)
                print(f"  rendered {idx + 1}/{args.n_samples} ({rate:.1f} samples/s)")

    manifest = {
        "n_samples": args.n_samples,
        "seed": args.seed,
        "schema_version": "0.1",
        "sample_rate": SAMPLE_RATE,
        "total_duration_s": TOTAL_DURATION_S,
        "renderer_git_sha": _git_sha(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Done. {args.n_samples} samples → {out_dir} in {time.time() - t0:.1f}s."
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the smoke test, expect pass**

```bash
uv run pytest tests/test_tone_generation_dataset.py::test_generate_dataset_script_smoke -v -m slow
```

Expected: passes. 5 wavs + 5 labels.jsonl lines + manifest.json.

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_subtractive_dataset.py tests/test_tone_generation_dataset.py
git commit -m "tone-generation: dataset generator script + smoke test"
```

---

## Task 8: ToneGenerationDataset (torch Dataset)

**Files:**
- Modify: `src/audio_analysis_mcp/research/tone_generation/dataset.py`
- Modify: `tests/test_tone_generation_dataset.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tone_generation_dataset.py`:

```python
import json as _json
import numpy as np
import soundfile as _sf
import torch

from audio_analysis_mcp.research.tone_generation.dataset import ToneGenerationDataset


def _build_mini_dataset(tmp_dir: Path, n: int = 4) -> Path:
    items = list(sample_dataset_config(n_samples=n, seed=0))
    samples_dir = tmp_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    labels_path = tmp_dir / "labels.jsonl"
    sr = 44100
    duration_s = 1.2
    audio = np.zeros(int(sr * duration_s), dtype=np.float32)
    audio[5000:50_000] = 0.5  # crude attack+sustain region so slice has signal
    with labels_path.open("w") as f:
        for idx, item in enumerate(items):
            wav_path = samples_dir / f"{idx:06d}.wav"
            _sf.write(str(wav_path), audio, sr, subtype="PCM_16")
            f.write(_json.dumps({
                "idx": idx,
                "params_canonical": item.params_canonical,
                "midi_pitches": item.midi_pitches,
                "n_voices": item.n_voices,
            }) + "\n")
    (tmp_dir / "manifest.json").write_text(_json.dumps({
        "n_samples": n, "seed": 0, "schema_version": "0.1",
        "sample_rate": sr, "total_duration_s": duration_s, "renderer_git_sha": "test",
    }))
    return tmp_dir


def test_tone_generation_dataset_shapes(tmp_path: Path):
    ds_dir = _build_mini_dataset(tmp_path, n=4)
    ds = ToneGenerationDataset(ds_dir)
    assert len(ds) == 4
    mel, pitch_mh, target = ds[0]
    assert isinstance(mel, torch.Tensor)
    assert mel.shape[0] == 1  # channel dim
    assert mel.shape[1] == 128  # mel bins
    assert mel.shape[2] >= 25  # ~30 frames for 300 ms / 10 ms hop
    assert pitch_mh.shape == (88,)
    assert pitch_mh.dtype == torch.float32
    assert pitch_mh.sum() >= 1
    assert "shape_label" in target
    assert "cutoff_norm" in target
    assert "resonance" in target
    assert isinstance(target["shape_label"], int)
    assert 0.0 <= float(target["cutoff_norm"]) <= 1.0
    assert 0.0 <= float(target["resonance"]) <= 1.0


def test_tone_generation_dataset_pitch_multihot_correct(tmp_path: Path):
    ds_dir = _build_mini_dataset(tmp_path, n=4)
    ds = ToneGenerationDataset(ds_dir)
    _, pitch_mh, _ = ds[0]
    label_line = (ds_dir / "labels.jsonl").read_text().splitlines()[0]
    expected = _json.loads(label_line)["midi_pitches"]
    indices = [p - 21 for p in expected]
    for i, v in enumerate(pitch_mh.tolist()):
        if i in indices:
            assert v == 1.0
        else:
            assert v == 0.0
```

- [ ] **Step 2: Run tests, expect ImportError**

```bash
uv run pytest tests/test_tone_generation_dataset.py::test_tone_generation_dataset_shapes -v
```

Expected: FAIL with `ImportError: cannot import name 'ToneGenerationDataset'`.

- [ ] **Step 3: Append ToneGenerationDataset to dataset.py**

Append to `src/audio_analysis_mcp/research/tone_generation/dataset.py`:

```python
import json
from pathlib import Path

import librosa
import soundfile as sf
import torch
from torch.utils.data import Dataset

from audio_analysis_mcp.research.tone_generation.schema_io import normalize_params

_PITCH_MULTIHOT_LO = 21
_PITCH_MULTIHOT_HI = 108  # inclusive — total dim 88
_SLICE_OFFSET_S = 0.10
_SLICE_DURATION_S = 0.30
_MEL_N_MELS = 128
_MEL_HOP_LENGTH = 441   # 10 ms at 44.1 kHz
_MEL_WIN_LENGTH = 1102  # 25 ms at 44.1 kHz
_MEL_N_FFT = 2048


def _pitch_multihot(midi_pitches: list[int]) -> torch.Tensor:
    vec = torch.zeros(_PITCH_MULTIHOT_HI - _PITCH_MULTIHOT_LO + 1, dtype=torch.float32)
    for p in midi_pitches:
        if _PITCH_MULTIHOT_LO <= p <= _PITCH_MULTIHOT_HI:
            vec[p - _PITCH_MULTIHOT_LO] = 1.0
    return vec


def _audio_to_mel(audio: np.ndarray, sample_rate: int) -> torch.Tensor:
    start = int(_SLICE_OFFSET_S * sample_rate)
    end = start + int(_SLICE_DURATION_S * sample_rate)
    slice_ = audio[start:end]
    if slice_.shape[0] < int(_SLICE_DURATION_S * sample_rate):
        # pad with zeros if too short
        pad = int(_SLICE_DURATION_S * sample_rate) - slice_.shape[0]
        slice_ = np.pad(slice_, (0, pad))
    mel = librosa.feature.melspectrogram(
        y=slice_,
        sr=sample_rate,
        n_fft=_MEL_N_FFT,
        hop_length=_MEL_HOP_LENGTH,
        win_length=_MEL_WIN_LENGTH,
        n_mels=_MEL_N_MELS,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # Channel dim first.
    return torch.from_numpy(log_mel.astype(np.float32)).unsqueeze(0)


class ToneGenerationDataset(Dataset):
    """Reads a generated subtractive dataset directory.

    Returns per item: (mel_spec, pitch_multihot, target_dict).
    target_dict keys: shape_label (int), cutoff_norm (float), resonance (float).
    """

    def __init__(self, dataset_dir: Path) -> None:
        self.dataset_dir = Path(dataset_dir)
        manifest = json.loads((self.dataset_dir / "manifest.json").read_text())
        self.sample_rate: int = int(manifest["sample_rate"])
        labels_path = self.dataset_dir / "labels.jsonl"
        self.labels: list[dict[str, Any]] = [
            json.loads(line) for line in labels_path.read_text().splitlines()
        ]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
        label = self.labels[idx]
        wav_path = self.dataset_dir / "samples" / f"{label['idx']:06d}.wav"
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != self.sample_rate:
            raise ValueError(f"unexpected sample rate {sr} != {self.sample_rate}")
        mel = _audio_to_mel(audio, sr)
        pitch_mh = _pitch_multihot(label["midi_pitches"])
        p = label["params_canonical"]["params"]
        target = normalize_params(
            shape=p["osc"]["1"]["shape"],
            cutoff_hz=float(p["filter"]["lp"]["cutoff_hz"]),
            resonance=float(p["filter"]["lp"]["resonance"]),
        )
        return mel, pitch_mh, target
```

- [ ] **Step 4: Run tests, expect pass**

```bash
uv run pytest tests/test_tone_generation_dataset.py -v && uv run mypy src/
```

Expected: all dataset tests pass, mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/audio_analysis_mcp/research/tone_generation/dataset.py tests/test_tone_generation_dataset.py
git commit -m "tone-generation: ToneGenerationDataset (torch Dataset) with mel-spec + pitch multihot"
```

---

## Task 9: model — ToneGenerationCNN

> **Pre-validation:** Task 2 scratch must have passed. The model architecture relies on standard ops (Conv2d, BatchNorm2d, MaxPool2d, AdaptiveAvgPool2d, Linear) all confirmed working on MPS in the scratch.

**Files:**
- Create: `src/audio_analysis_mcp/research/tone_generation/model.py`
- Create: `tests/test_tone_generation_model.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tone_generation_model.py`:

```python
import torch

from audio_analysis_mcp.research.tone_generation.model import ToneGenerationCNN


def test_forward_shapes_cpu():
    model = ToneGenerationCNN()
    mel = torch.randn(8, 1, 128, 30)
    pitch_mh = torch.zeros(8, 88)
    pitch_mh[:, 39] = 1.0  # MIDI 60
    out = model(mel, pitch_mh)
    assert out["shape_logits"].shape == (8, 4)
    assert out["cutoff_norm"].shape == (8,)
    assert out["resonance"].shape == (8,)


def test_forward_outputs_in_range():
    model = ToneGenerationCNN().eval()
    mel = torch.randn(2, 1, 128, 30)
    pitch_mh = torch.zeros(2, 88)
    pitch_mh[:, 39] = 1.0
    with torch.no_grad():
        out = model(mel, pitch_mh)
    assert torch.all(out["cutoff_norm"] >= 0.0) and torch.all(out["cutoff_norm"] <= 1.0)
    assert torch.all(out["resonance"] >= 0.0) and torch.all(out["resonance"] <= 1.0)


def test_backward_runs():
    model = ToneGenerationCNN()
    mel = torch.randn(4, 1, 128, 30, requires_grad=False)
    pitch_mh = torch.zeros(4, 88)
    pitch_mh[:, 40] = 1.0
    out = model(mel, pitch_mh)
    loss = (
        torch.nn.functional.cross_entropy(out["shape_logits"], torch.tensor([0, 1, 2, 3]))
        + torch.nn.functional.mse_loss(out["cutoff_norm"], torch.tensor([0.5, 0.5, 0.5, 0.5]))
        + torch.nn.functional.mse_loss(out["resonance"], torch.tensor([0.5, 0.5, 0.5, 0.5]))
    )
    loss.backward()
    # at least one param should have a non-None gradient
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
```

- [ ] **Step 2: Run tests, expect ImportError**

```bash
uv run pytest tests/test_tone_generation_model.py -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement model.py**

Create `src/audio_analysis_mcp/research/tone_generation/model.py`:

```python
"""Conditioned CNN for subtractive tone-generation MVP.

Architecture:
  mel-spec (1, 128, ~30) → 3 conv blocks → AdaptiveAvgPool2d((-, 1))  # pool over time
  → flatten → concat with 88-dim pitch multihot → 2-layer MLP → 3 heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_PITCH_DIM = 88


class _ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ToneGenerationCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block1 = _ConvBlock(1, 32)
        self.block2 = _ConvBlock(32, 64)
        self.block3 = _ConvBlock(64, 128)
        # After 3 MaxPool2d(2): mel dim 128 -> 16; time dim ~30 -> ~3.
        # AdaptiveAvgPool2d((16, 1)) collapses time but preserves freq.
        self.time_pool = nn.AdaptiveAvgPool2d((16, 1))
        bottleneck_dim = 128 * 16  # channels * mel after pool
        self.mlp = nn.Sequential(
            nn.Linear(bottleneck_dim + _PITCH_DIM, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        )
        self.head_shape = nn.Linear(256, 4)
        self.head_cutoff = nn.Linear(256, 1)
        self.head_resonance = nn.Linear(256, 1)

    def forward(
        self, mel: torch.Tensor, pitch_multihot: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        h = self.block1(mel)
        h = self.block2(h)
        h = self.block3(h)
        h = self.time_pool(h)
        h = h.flatten(1)
        h = torch.cat([h, pitch_multihot], dim=1)
        h = self.mlp(h)
        return {
            "shape_logits": self.head_shape(h),
            "cutoff_norm": torch.sigmoid(self.head_cutoff(h)).squeeze(-1),
            "resonance": torch.sigmoid(self.head_resonance(h)).squeeze(-1),
        }
```

- [ ] **Step 4: Run tests, expect pass**

```bash
uv run pytest tests/test_tone_generation_model.py -v && uv run mypy src/
```

Expected: 3 tests pass, mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/audio_analysis_mcp/research/tone_generation/model.py tests/test_tone_generation_model.py
git commit -m "tone-generation: ToneGenerationCNN — conditioned CNN with pitch multihot concat"
```

---

## Task 10: Training script

**Files:**
- Create: `scripts/train_tone_generation.py`
- Create: `tests/test_tone_generation_smoke.py`

- [ ] **Step 1: Write the failing smoke test**

Create `tests/test_tone_generation_smoke.py`:

```python
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
GEN_SCRIPT = REPO_ROOT / "scripts" / "generate_subtractive_dataset.py"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_tone_generation.py"


@pytest.mark.slow
def test_train_smoke(tmp_path: Path):
    ds_dir = tmp_path / "ds"
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    # Tiny dataset.
    subprocess.run(
        [sys.executable, str(GEN_SCRIPT),
         "--n-samples", "40", "--seed", "0", "--out-dir", str(ds_dir)],
        check=True, capture_output=True, text=True,
    )
    # Train for a few epochs.
    res = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT),
         "--dataset-dir", str(ds_dir),
         "--checkpoint-out", str(ckpt_dir / "checkpoint.pt"),
         "--epochs", "3",
         "--batch-size", "8",
         "--seed", "0"],
        check=True, capture_output=True, text=True,
    )
    print(res.stdout)
    assert (ckpt_dir / "checkpoint.pt").exists()
    eval_path = ckpt_dir / "eval_report.json"
    assert eval_path.exists()
    report = json.loads(eval_path.read_text())
    assert "shape_accuracy" in report
    assert "cutoff_log_mse" in report
    assert "resonance_mse" in report
    assert report["schema_validation_failures"] == 0
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/test_tone_generation_smoke.py::test_train_smoke -v -m slow
```

Expected: FAIL — train script doesn't exist.

- [ ] **Step 3: Implement training script**

Create `scripts/train_tone_generation.py`:

```python
"""Train the subtractive tone-generation CNN.

Usage:
    uv run python scripts/train_tone_generation.py \\
        --dataset-dir scratch/tone_gen_dataset \\
        --checkpoint-out scratch/tone_gen_checkpoints/checkpoint.pt \\
        --epochs 50 --batch-size 64 --seed 0
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_analysis_mcp.research.tone_generation.dataset import ToneGenerationDataset
from audio_analysis_mcp.research.tone_generation.model import ToneGenerationCNN
from audio_analysis_mcp.research.tone_generation.schema_io import (
    SHAPE_LABELS,
    denormalize_predictions,
    validate_canonical,
)


def _split_indices(n: int) -> tuple[list[int], list[int], list[int]]:
    train, val, test = [], [], []
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


def _collate(batch):
    mels = torch.stack([item[0] for item in batch])
    pitches = torch.stack([item[1] for item in batch])
    shape_label = torch.tensor([item[2]["shape_label"] for item in batch], dtype=torch.long)
    cutoff = torch.tensor([item[2]["cutoff_norm"] for item in batch], dtype=torch.float32)
    res = torch.tensor([item[2]["resonance"] for item in batch], dtype=torch.float32)
    return mels, pitches, shape_label, cutoff, res


def _compute_eval(model, loader, device, report_canonical_failures: bool = True) -> dict:
    model.eval()
    n = 0
    correct_shape = 0
    cutoff_se = 0.0
    res_se = 0.0
    failures = 0
    with torch.no_grad():
        for mels, pitches, shape_label, cutoff, res in loader:
            mels, pitches = mels.to(device), pitches.to(device)
            shape_label = shape_label.to(device)
            cutoff = cutoff.to(device)
            res = res.to(device)
            out = model(mels, pitches)
            preds_shape = out["shape_logits"].argmax(dim=1)
            correct_shape += (preds_shape == shape_label).sum().item()
            cutoff_se += F.mse_loss(out["cutoff_norm"], cutoff, reduction="sum").item()
            res_se += F.mse_loss(out["resonance"], res, reduction="sum").item()
            n += mels.size(0)
            if report_canonical_failures:
                # Build canonical instance for each prediction; validate.
                for i in range(mels.size(0)):
                    pitch_idxs = (pitches[i] > 0.5).nonzero(as_tuple=False).squeeze(-1).tolist()
                    midi_pitches = [21 + j for j in pitch_idxs] or [60]
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
    return {
        "shape_accuracy": correct_shape / max(n, 1),
        "cutoff_log_mse": cutoff_se / max(n, 1),
        "resonance_mse": res_se / max(n, 1),
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
    train_loader = DataLoader(Subset(full, train_idx), batch_size=args.batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(Subset(full, val_idx), batch_size=args.batch_size, shuffle=False, collate_fn=_collate)
    test_loader = DataLoader(Subset(full, test_idx), batch_size=args.batch_size, shuffle=False, collate_fn=_collate)

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
            mels, pitches = mels.to(device), pitches.to(device)
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
            train_loss += loss.item() * mels.size(0)
            n_train += mels.size(0)
        train_loss /= max(n_train, 1)

        val_metrics = _compute_eval(model, val_loader, device, report_canonical_failures=False)
        val_loss = (
            (1.0 - val_metrics["shape_accuracy"])
            + val_metrics["cutoff_log_mse"]
            + val_metrics["resonance_mse"]
        )
        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"shape_acc={val_metrics['shape_accuracy']:.3f} "
            f"cutoff_mse={val_metrics['cutoff_log_mse']:.4f} "
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

    # Final eval on test set with the best checkpoint.
    model.load_state_dict(torch.load(args.checkpoint_out, map_location=device))
    test_metrics = _compute_eval(model, test_loader, device)
    eval_report_path = args.checkpoint_out.parent / "eval_report.json"
    eval_report_path.write_text(json.dumps(test_metrics, indent=2) + "\n")
    print(f"\ntest metrics:\n{json.dumps(test_metrics, indent=2)}")
    print(f"eval report → {eval_report_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run smoke test, expect pass**

```bash
uv run pytest tests/test_tone_generation_smoke.py::test_train_smoke -v -m slow
```

Expected: 40-sample dataset, 3 epochs, finishes in < 60 s on M3 Pro, eval_report.json with `schema_validation_failures: 0`.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_tone_generation.py tests/test_tone_generation_smoke.py
git commit -m "tone-generation: training script + slow smoke test"
```

---

## Task 11: Eval-only script + round-trip mel-cosine

**Files:**
- Create: `scripts/eval_tone_generation.py`
- Modify: `scripts/train_tone_generation.py` (factor eval into a helper used by both scripts)
- Modify: `tests/test_tone_generation_smoke.py` (add eval-only smoke test)

- [ ] **Step 1: Refactor — extract eval into a shared helper**

Move the round-trip mel-cosine logic + `_compute_eval` out of `train_tone_generation.py` into a new file `scripts/_eval_helpers.py` so both train and eval scripts can import it. (Scripts import via the same `sys.path` insertion at the top.)

Create `scripts/_eval_helpers.py`:

```python
"""Shared eval logic for tone-generation training + standalone eval scripts."""

from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_analysis_mcp.research.tone_generation.dataset import (
    ToneGenerationDataset,
    _audio_to_mel,  # type: ignore[attr-defined]
)
from audio_analysis_mcp.research.tone_generation.renderer import render_chord
from audio_analysis_mcp.research.tone_generation.schema_io import (
    denormalize_predictions,
    validate_canonical,
)


_SAMPLE_RATE = 44_100
_TOTAL_DURATION_S = 1.2


def _round_trip_mel_cosine(
    instance: dict, midi_pitches: list[int], original_mel: torch.Tensor
) -> float:
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


def compute_full_eval(
    model: torch.nn.Module,
    dataset: ToneGenerationDataset,
    indices: list[int],
    device: torch.device,
    batch_size: int = 64,
) -> dict:
    model.eval()
    from torch.utils.data import Subset

    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, collate_fn=_collate)

    n = 0
    correct_shape = 0
    cutoff_se = 0.0
    res_se = 0.0
    failures = 0
    cosines: list[float] = []
    confusion = np.zeros((4, 4), dtype=np.int64)

    with torch.no_grad():
        for batch_idx, (mels, pitches, shape_label, cutoff, res) in enumerate(loader):
            mels_d, pitches_d = mels.to(device), pitches.to(device)
            shape_label_d = shape_label.to(device)
            cutoff_d = cutoff.to(device)
            res_d = res.to(device)
            out = model(mels_d, pitches_d)
            preds_shape = out["shape_logits"].argmax(dim=1)
            correct_shape += (preds_shape == shape_label_d).sum().item()
            cutoff_se += F.mse_loss(out["cutoff_norm"], cutoff_d, reduction="sum").item()
            res_se += F.mse_loss(out["resonance"], res_d, reduction="sum").item()
            n += mels.size(0)
            for i in range(mels.size(0)):
                confusion[int(shape_label[i].item()), int(preds_shape[i].item())] += 1
                pitch_idxs = (pitches[i] > 0.5).nonzero(as_tuple=False).squeeze(-1).tolist()
                midi_pitches = [21 + j for j in pitch_idxs] or [60]
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
                    continue
                cos = _round_trip_mel_cosine(inst, midi_pitches, mels[i])
                cosines.append(cos)

    return {
        "shape_accuracy": correct_shape / max(n, 1),
        "shape_confusion": confusion.tolist(),
        "cutoff_log_mse": cutoff_se / max(n, 1),
        "resonance_mse": res_se / max(n, 1),
        "schema_validation_failures": failures,
        "round_trip_mel_cosine_mean": float(np.mean(cosines)) if cosines else 0.0,
        "round_trip_mel_cosine_median": float(np.median(cosines)) if cosines else 0.0,
        "n_samples": n,
    }


def _collate(batch):
    mels = torch.stack([item[0] for item in batch])
    pitches = torch.stack([item[1] for item in batch])
    shape_label = torch.tensor([item[2]["shape_label"] for item in batch], dtype=torch.long)
    cutoff = torch.tensor([item[2]["cutoff_norm"] for item in batch], dtype=torch.float32)
    res = torch.tensor([item[2]["resonance"] for item in batch], dtype=torch.float32)
    return mels, pitches, shape_label, cutoff, res
```

- [ ] **Step 2: Update train script to use the shared helper**

Make these specific edits to `scripts/train_tone_generation.py`:

(a) Add `_eval_helpers` to the importable path. Right after the existing `sys.path.insert(...)` block (which adds `src/`), insert:

```python
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _eval_helpers import _collate, compute_full_eval  # noqa: E402
```

(b) **Delete** the local `_collate` function and the local `_compute_eval` function.

(c) Replace the per-epoch val loop (the block that previously called `_compute_eval(model, val_loader, device, report_canonical_failures=False)`) with a lightweight val-loss forward — no round-trip during training so epochs stay fast:

```python
val_loss_total = 0.0
n_val = 0
model.eval()
with torch.no_grad():
    for mels, pitches, shape_label, cutoff, res in val_loader:
        mels = mels.to(device); pitches = pitches.to(device)
        shape_label = shape_label.to(device); cutoff = cutoff.to(device); res = res.to(device)
        out = model(mels, pitches)
        loss = (F.cross_entropy(out["shape_logits"], shape_label)
                + F.mse_loss(out["cutoff_norm"], cutoff)
                + F.mse_loss(out["resonance"], res))
        val_loss_total += loss.item() * mels.size(0)
        n_val += mels.size(0)
val_loss = val_loss_total / max(n_val, 1)
print(
    f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
)
```

(d) Replace the final test eval (the previous call to `_compute_eval(model, test_loader, device)`) with the shared full-eval that includes round-trip mel cosine + confusion matrix:

```python
test_metrics = compute_full_eval(model, full, test_idx, device, batch_size=args.batch_size)
```

- [ ] **Step 3: Implement standalone eval script**

Create `scripts/eval_tone_generation.py`:

```python
"""Re-run eval against a saved checkpoint without retraining.

Usage:
    uv run python scripts/eval_tone_generation.py \\
        --checkpoint scratch/tone_gen_checkpoints/checkpoint.pt \\
        --dataset-dir scratch/tone_gen_dataset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _eval_helpers import compute_full_eval

from audio_analysis_mcp.research.tone_generation.dataset import ToneGenerationDataset
from audio_analysis_mcp.research.tone_generation.model import ToneGenerationCNN


def _split_indices(n: int) -> tuple[list[int], list[int], list[int]]:
    train, val, test = [], [], []
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = _select_device()
    full = ToneGenerationDataset(args.dataset_dir)
    _, _, test_idx = _split_indices(len(full))

    model = ToneGenerationCNN().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    metrics = compute_full_eval(model, full, test_idx, device, batch_size=args.batch_size)
    out = args.out or args.checkpoint.parent / "eval_report.json"
    out.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"eval report → {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add eval-only smoke test**

Append to `tests/test_tone_generation_smoke.py`:

```python
EVAL_SCRIPT = REPO_ROOT / "scripts" / "eval_tone_generation.py"


@pytest.mark.slow
def test_eval_smoke(tmp_path: Path):
    ds_dir = tmp_path / "ds"
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    subprocess.run(
        [sys.executable, str(GEN_SCRIPT),
         "--n-samples", "40", "--seed", "0", "--out-dir", str(ds_dir)],
        check=True, capture_output=True, text=True,
    )
    subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT),
         "--dataset-dir", str(ds_dir),
         "--checkpoint-out", str(ckpt_dir / "checkpoint.pt"),
         "--epochs", "2", "--batch-size", "8", "--seed", "0"],
        check=True, capture_output=True, text=True,
    )
    res = subprocess.run(
        [sys.executable, str(EVAL_SCRIPT),
         "--checkpoint", str(ckpt_dir / "checkpoint.pt"),
         "--dataset-dir", str(ds_dir),
         "--out", str(tmp_path / "eval2.json")],
        check=True, capture_output=True, text=True,
    )
    print(res.stdout)
    report = json.loads((tmp_path / "eval2.json").read_text())
    assert report["schema_validation_failures"] == 0
    assert "round_trip_mel_cosine_mean" in report
    assert -1.0 <= report["round_trip_mel_cosine_mean"] <= 1.0
    assert "shape_confusion" in report
```

- [ ] **Step 5: Run all smoke tests, expect pass**

```bash
uv run pytest tests/test_tone_generation_smoke.py -v -m slow
```

Expected: both smoke tests pass. Round-trip mel-cosine in `[-1, 1]`, schema validation failures = 0.

- [ ] **Step 6: Commit**

```bash
git add scripts/_eval_helpers.py scripts/eval_tone_generation.py scripts/train_tone_generation.py tests/test_tone_generation_smoke.py
git commit -m "tone-generation: shared eval helper + standalone eval script with round-trip mel cosine"
```

---

## Task 12: README + final verification

**Files:**
- Create: `src/audio_analysis_mcp/research/tone_generation/README.md`

- [ ] **Step 1: Write the README**

Create `src/audio_analysis_mcp/research/tone_generation/README.md`:

```markdown
# Subtractive Tone-Generation Training (MVP)

Implementation of the MVP slice of the subtractive tone-generation training pipeline. Predicts three free synth params (`osc.1.shape`, `filter.lp.cutoff_hz`, `filter.lp.resonance`) from a sustain-region log-mel spectrogram, conditioned on ground-truth played MIDI pitches.

**Spec:** `reverse-synth-research/docs/superpowers/specs/2026-05-02-subtractive-tone-training-mvp.md`

**Schema:** `reverse-synth-research/parameter-mapping/subtractive.schema.json` (v0.1)

## End-to-end workflow

From the `audio-analysis-mcp/` directory:

```bash
# 1) Generate a 10K-sample synthetic dataset (~few minutes on M3 Pro).
uv run python scripts/generate_subtractive_dataset.py \
    --n-samples 10000 --seed 0 \
    --out-dir scratch/tone_gen_dataset

# 2) Train (≤ 30 min CPU, ≤ 5 min MPS on M3 Pro).
uv run python scripts/train_tone_generation.py \
    --dataset-dir scratch/tone_gen_dataset \
    --checkpoint-out scratch/tone_gen_checkpoints/checkpoint.pt \
    --epochs 50 --batch-size 64 --seed 0

# 3) Re-run eval at any time.
uv run python scripts/eval_tone_generation.py \
    --checkpoint scratch/tone_gen_checkpoints/checkpoint.pt \
    --dataset-dir scratch/tone_gen_dataset
```

## Output of the training run

- `checkpoint.pt` — model weights, the best-val-loss checkpoint kept during training
- `eval_report.json` — per-param accuracy/MSE + round-trip mel-cosine + schema validation pass count

## Module layout

| File | Responsibility |
|---|---|
| `schema_io.py` | Schema constants, canonical-instance builder, normalize/denormalize, validation |
| `renderer.py` | SignalFlow chord renderer (osc → LP → fixed amp ADSR → sum of N voices) |
| `dataset.py` | Sobol+uniform sampler + `ToneGenerationDataset` (mel-spec on the fly) |
| `model.py` | Small custom CNN with 88-dim pitch multihot conditioning |

## What this MVP deliberately does NOT do

See the [backlog](../../../../../reverse-synth-research/docs/superpowers/specs/2026-05-02-subtractive-tone-training-backlog.md) for the deferred-features list. Big ones: `osc.2`/`osc.sub`/`noise`, modulation invariance, attack-region multi-slice, real-keyboard recordings, `note_transcribe` integration at dataset time, larger backbones, top-K predictions.
```

- [ ] **Step 2: Final sanity — run the full test suite**

```bash
uv run pytest -v -m "not slow"
uv run pytest -v -m slow
uv run mypy src/
```

Expected: everything passes, mypy clean.

- [ ] **Step 3: Commit**

```bash
git add src/audio_analysis_mcp/research/tone_generation/README.md
git commit -m "tone-generation: workflow README"
```

---

## Self-Review

The following checks were run after writing this plan; fixes are inline.

**Spec coverage** (every In-scope item in the MVP spec maps to a task):

| Spec item | Task |
|---|---|
| Renderer: minimal subtractive engine | T5 |
| Free synth params (3) | T4 (schema_io builder), T6 (sampler), T9 (model heads) |
| Frozen synth params + baseline ADSR | T4 (`BASELINE_AMP_ADSR`, `build_canonical_instance`) |
| Polyphonic chord render (1–3 voices, 12-semitone window) | T1 (verified), T5 (`render_chord`), T6 (`_sample_chord_pitches`) |
| Sobol + log-distributed cutoff | T6 (`sample_dataset_config`) |
| Dataset output layout (samples/, labels.jsonl, manifest.json) | T7 |
| 10K samples, deterministic, modulo-10 split | T6 (deterministic), T10 (`_split_indices`) |
| Sustain slice 300 ms / 100 ms offset / 128-mel / 25 ms win / 10 ms hop | T8 (`_audio_to_mel` constants) |
| 88-dim pitch multi-hot | T8 (`_pitch_multihot`), T9 (concat at bottleneck) |
| Param normalization (cutoff log-norm, resonance passthrough, shape int) | T4 (`normalize_params`) |
| 3-block CNN + global temporal pool + concat + MLP heads | T9 |
| Adam, lr=1e-3, weight_decay=1e-4, batch=64, epochs=50, patience=5 | T10 |
| MPS-first device autodetect | T10 (`_select_device`) |
| Per-param metrics + confusion matrix | T11 (`compute_full_eval`) |
| Round-trip mel-cosine | T11 (`_round_trip_mel_cosine`) |
| Schema-validation gate | T4 (`validate_canonical`), T11 (per-prediction validation) |
| Three deliverable scripts | T7, T10, T11 |
| Module package + README | T3, T4–T9, T12 |
| Pre-implementation scratch for SignalFlow polyphony | T1 |
| `torch` MPS dependency note | T2 (verified), T3 (no special build needed) |
| Eval report JSON | T10 + T11 |

**Spec → backlog cross-check:** every Out-of-scope item in the MVP spec is named in the backlog file (verified during the spec self-review prior to handoff).

**Type consistency check:** `_collate` is defined identically in `_eval_helpers.py` and was originally in `train_tone_generation.py` — Task 11 Step 1 + 2 explicitly remove the duplicate from the train script after factoring. `denormalize_predictions` signature matches between `schema_io.py` (Task 4) and its callers in Tasks 10 + 11. `ToneGenerationCNN.forward` signature `(mel, pitch_multihot)` matches across model tests, train script, and eval helper.

**Placeholder scan:** no TBD / TODO / "implement appropriately" / "similar to Task N" patterns. Every code block is complete.

**Inert LFO note:** Task 4 includes `_inert_lfo()` because the canonical schema requires `params.lfo` for subtractive engines. The MVP renders without modulation; depth=0 makes it acoustically silent. This is documented in the schema_io.py docstring and called out in the task description.

---

## Execution Handoff

Plan complete and saved to `audio-analysis-mcp/docs/superpowers/plans/2026-05-02-subtractive-tone-training.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration. Good fit here because tasks are independent and TDD-shaped.

**2. Inline Execution** — run tasks in this session via `superpowers:executing-plans`, batched with checkpoints for review.

**Which approach?**
