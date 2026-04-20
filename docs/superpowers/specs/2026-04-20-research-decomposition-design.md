# Audio Analysis MCP: Research Decomposition Design

**Date:** 2026-04-20
**Status:** Approved
**Supersedes:** `archive/7-audio-analysis-mcp.md` (monolithic plan)

## Problem

The original audio-analysis-mcp plan (`7-audio-analysis-mcp.md`) combines three fundamentally different efforts in one document:

1. **Audio pipeline tools** — straightforward engineering (fetch, stems, spectrum, capture, compare, transcribe, isolate)
2. **Synth engine detection** — open research question (how to classify what synthesis type produced a sound)
3. **Inverse synthesis** — heavy ML research (predicting synth parameters from audio)

Mixing engineering and research in one plan creates a false sense of sequential progress and makes it impossible to scope or schedule either research effort.

## Decision

Decompose into three independent layers within the same repo:

| Layer | Location | Nature |
|-------|----------|--------|
| Pipeline (MCP tools) | `src/audio_analysis_mcp/` | Engineering |
| Engine Detection | `research/engine-detection/` | Exploratory research |
| Inverse Synthesis | `research/inverse-synth/` | Heavy ML research |

Each layer gets its own design doc, plan, and implementation cycle.

## Layer 1: Audio Pipeline (Engineering)

The MCP server with audio processing tools. No ML, no research — known tools, known APIs.

**Tools:**

| Tool | Purpose |
|------|---------|
| `fetch_audio` | Download + normalize audio from YouTube or local file |
| `stem_separate` | Demucs 4-way stem separation |
| `spectrum_analyze` | Spectral feature extraction (harmonics, ADSR, modulation) |
| `audio_compare` | A/B spectral diff with action items |
| `audio_render` | System audio device capture |
| `note_transcribe` | Polyphonic transcription via Basic Pitch |
| `note_isolate` | Score-informed source separation via nussl |

**Status:** Scaffold plan exists (`archive/8c-audio-analysis-mcp.md`) but has not been implemented yet. Scaffolding + tool implementations are the next engineering effort.

**Can start:** Now. No dependencies on research.

## Layer 2: Engine Detection Research

**Goal:** Given an audio segment (isolated note or stem), classify which synthesis engine category produced it.

**Target categories (5 classes):** subtractive, FM, organ/additive, wavetable, sample-based.

### Research Phases

1. **Data collection** — Generate labeled audio samples by rendering known patches through software synths (one per category). For sample-based, use recordings of real pianos, Rhodes, Wurlitzer, etc.

2. **Feature exploration** — Extract candidate spectral features (harmonic ratios, spectral centroid, spectral flux, ADSR envelope shape, inharmonicity, modulation depth/rate, formant structure). Use notebooks to visualize which features discriminate between engine types.

3. **Handcrafted-features path** — Build a feature vector from the best discriminators. Train LightGBM / random forest / decision tree. Fast iteration, interpretable, easy to debug.

4. **End-to-end path** — Train a small CNN on mel spectrograms with engine type as the label. Compare accuracy against the handcrafted approach.

5. **Evaluation** — Confusion matrix across the 5 categories. Identify which pairs are hardest to distinguish. Decide if the taxonomy needs adjustment based on data.

### Deliverable

A trained classifier packaged so the MCP server can call it from an `engine_detect` tool. Replaces the hand-wavy "synth hints" in the original plan's `spectrum_analyze`.

### Directory Structure

```
research/engine-detection/
  notebooks/           # Jupyter exploration
  data/                # Generated/collected labeled samples
  features/            # Feature extraction code
  models/              # Trained classifiers
  README.md            # Research log / findings
```

**Can start:** In parallel with pipeline. Needs audio samples, not pipeline tools.

## Layer 3: Inverse Synthesis Research

**Goal:** Given audio + a known synth type, predict the parameter vector that would reproduce that sound. One model per synthesis type.

### Research Phases

1. **Rendering infrastructure** — Build or integrate synth renderers that take a parameter vector and produce audio programmatically. Explore: pure Python DSP, headless VST plugins (SurgeXT, Dexed, setBfree), or MIDI-to-audio via keyboards-mcp.

2. **Dataset generation pipeline** — Random parameter vectors -> render -> augmentation (effects, noise, polyphony variations, stem bleed). Training data is free because the synth is the ground truth. Research determines: how many samples, what augmentation strategy, what musical content variation is sufficient.

3. **Model architecture exploration** — Start with subtractive (simplest parameter space). Explore architectures: CNN backbone + per-param MLP heads, simpler baselines, existing approaches from literature (InverSynth, DiffMoog, DDSP). Address the many-to-one problem (multiple param combos produce the same sound).

4. **Training & evaluation** — Define metrics: per-parameter accuracy, round-trip audio similarity (predict params -> render -> compare to input), perceptual similarity. Establish baselines.

5. **Generalization** — Once subtractive works, expand to FM and organ. Assess how much architecture and pipeline transfers between synth types.

### Deliverable

Trained model(s) + inference code that the MCP server can call from `inverse_synth`, `train_model`, and `list_models` tools. Documentation of what worked, what didn't, and recommended next steps.

### Directory Structure

```
research/inverse-synth/
  notebooks/           # Architecture experiments, evaluation
  data/                # Generated datasets (potentially large, gitignored)
  renderers/           # Synth rendering backends
  models/              # Training code + checkpoints
  evaluation/          # Metrics, comparison scripts
  README.md            # Research log / findings
```

**Can start:** Data generation in parallel with other layers. Full evaluation benefits from engine detection being usable.

## Integration Points

1. **Engine Detection -> MCP:** Trained classifier loaded by a new `engine_detect` tool on the MCP server.

2. **Inverse Synth -> MCP:** Trained models loaded by the existing `inverse_synth` tool stub. `train_model` and `list_models` wire into whatever the research settles on.

3. **Pipeline -> Research:** Both research projects consume pipeline outputs — isolated notes from `note_isolate` are the primary input for both detection and inverse synthesis.

## Build Order

1. **Pipeline tools** — can start now (engineering, no unknowns)
2. **Engine detection research** — can start in parallel with pipeline
3. **Inverse synth research** — can start data generation in parallel, full evaluation benefits from engine detection

## What Changes

- Archive `7-audio-analysis-mcp.md` and `8c-audio-analysis-mcp.md` to `archive/`
- Pipeline gets its own trimmed plan (tools 1-7 from the original phase 1/1.5)
- Each research project gets its own design doc and plan on its own timeline