# Research Sub-Projects Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the directory structure and project scaffolding for two research sub-projects (engine detection, inverse synthesis) inside audio-analysis-mcp, plus a .gitignore and updated CLAUDE.md that reflects the new three-layer decomposition.

**Architecture:** Two research directories under `research/` with notebooks, data, models, and README files. A repo-level .gitignore excludes large generated artifacts (datasets, model checkpoints, venv). CLAUDE.md documents the decomposition and how the three layers connect.

**Tech Stack:** Git, Markdown, Jupyter notebooks (placeholder `.gitkeep` files for empty directories)

**Spec:** `docs/superpowers/specs/2026-04-20-research-decomposition-design.md`

---

### File Map

```
audio-analysis-mcp/
  .gitignore                                    # CREATE — repo-level ignores
  CLAUDE.md                                     # CREATE — project guidance
  research/
    engine-detection/
      README.md                                 # CREATE — research project overview + log
      notebooks/.gitkeep                        # CREATE — placeholder
      data/.gitkeep                             # CREATE — placeholder
      features/.gitkeep                         # CREATE — placeholder
      models/.gitkeep                           # CREATE — placeholder
    inverse-synth/
      README.md                                 # CREATE — research project overview + log
      notebooks/.gitkeep                        # CREATE — placeholder
      data/.gitkeep                             # CREATE — placeholder
      renderers/.gitkeep                        # CREATE — placeholder
      models/.gitkeep                           # CREATE — placeholder
      evaluation/.gitkeep                       # CREATE — placeholder
```

---

### Task 1: Create .gitignore

**Files:**
- Create: `.gitignore`

- [ ] **Step 1: Create .gitignore**

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
.venv/
*.egg-info/
dist/

# IDE
.mypy_cache/
.pytest_cache/
.vscode/
.idea/

# Research artifacts (large, generated)
research/*/data/*.wav
research/*/data/*.mp3
research/*/data/*.npy
research/*/data/*.npz
research/*/data/*.h5
research/*/models/*.pt
research/*/models/*.onnx
research/*/models/*.pkl
research/*/models/*.joblib

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store

# Environment
.env
```

- [ ] **Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: add .gitignore for research artifacts and Python"
```

---

### Task 2: Scaffold engine detection research directory

**Files:**
- Create: `research/engine-detection/README.md`
- Create: `research/engine-detection/notebooks/.gitkeep`
- Create: `research/engine-detection/data/.gitkeep`
- Create: `research/engine-detection/features/.gitkeep`
- Create: `research/engine-detection/models/.gitkeep`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p research/engine-detection/{notebooks,data,features,models}
touch research/engine-detection/{notebooks,data,features,models}/.gitkeep
```

- [ ] **Step 2: Create README.md**

```markdown
# Engine Detection Research

Classify audio segments by synthesis engine type.

## Goal

Given an isolated note or audio stem, predict which synthesis engine category produced it.

## Target Categories (5 classes)

| Category | Examples |
|----------|----------|
| Subtractive | Prophet-6, Moog, Juno-106 |
| FM | DX7, FM8 |
| Organ / Additive | Hammond B3, drawbar organs |
| Wavetable | PPG Wave, Waldorf |
| Sample-based | Acoustic piano, Rhodes, Wurlitzer, Clavinet |

## Research Phases

1. **Data collection** — Generate labeled audio samples by rendering known patches through software synths. For sample-based, use recordings of real instruments.
2. **Feature exploration** — Extract candidate spectral features. Visualize which features discriminate between engine types.
3. **Handcrafted-features path** — Feature vector + LightGBM / random forest / decision tree.
4. **End-to-end path** — Small CNN on mel spectrograms. Compare against handcrafted approach.
5. **Evaluation** — Confusion matrix, identify hard-to-distinguish pairs, adjust taxonomy if needed.

## Deliverable

Trained classifier callable from an `engine_detect` MCP tool.

## Directory Layout

| Directory | Contents |
|-----------|----------|
| `notebooks/` | Jupyter exploration and experiments |
| `data/` | Generated/collected labeled audio samples (large files gitignored) |
| `features/` | Feature extraction code |
| `models/` | Trained classifier checkpoints (large files gitignored) |

## Research Log

_Record findings, experiment results, and decisions here as the research progresses._
```

- [ ] **Step 3: Commit**

```bash
git add research/engine-detection/
git commit -m "feat: scaffold engine detection research sub-project"
```

---

### Task 3: Scaffold inverse synthesis research directory

**Files:**
- Create: `research/inverse-synth/README.md`
- Create: `research/inverse-synth/notebooks/.gitkeep`
- Create: `research/inverse-synth/data/.gitkeep`
- Create: `research/inverse-synth/renderers/.gitkeep`
- Create: `research/inverse-synth/models/.gitkeep`
- Create: `research/inverse-synth/evaluation/.gitkeep`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p research/inverse-synth/{notebooks,data,renderers,models,evaluation}
touch research/inverse-synth/{notebooks,data,renderers,models,evaluation}/.gitkeep
```

- [ ] **Step 2: Create README.md**

```markdown
# Inverse Synthesis Research

Predict synthesizer parameters from audio.

## Goal

Given audio + a known synthesis type, predict the parameter vector that would reproduce that sound. One model per synthesis type.

## The Problem

```
Forward (the synth):     param_vector  ->  synth_engine  ->  audio
Inverse (what we need):  audio         ->  trained_model ->  param_vector
```

## Research Phases

1. **Rendering infrastructure** — Build or integrate synth renderers that take a parameter vector and produce audio. Explore: pure Python DSP, headless VST plugins (SurgeXT, Dexed, setBfree), MIDI-to-audio via keyboards-mcp.
2. **Dataset generation pipeline** — Random params -> render -> augment (effects, noise, polyphony, stem bleed). Training data is free because the synth is the ground truth.
3. **Model architecture exploration** — Start with subtractive synthesis. Explore: CNN + per-param MLP heads, simpler baselines, literature approaches (InverSynth, DiffMoog, DDSP). Address the many-to-one problem.
4. **Training & evaluation** — Metrics: per-parameter accuracy, round-trip audio similarity, perceptual similarity.
5. **Generalization** — Expand from subtractive to FM and organ. Assess cross-type transfer.

## Deliverable

Trained model(s) + inference code callable from `inverse_synth`, `train_model`, and `list_models` MCP tools.

## Directory Layout

| Directory | Contents |
|-----------|----------|
| `notebooks/` | Architecture experiments, evaluation analysis |
| `data/` | Generated datasets (large files gitignored) |
| `renderers/` | Synth rendering backends for data generation |
| `models/` | Training code + model checkpoints (large files gitignored) |
| `evaluation/` | Metrics computation, comparison scripts |

## Key References

| Paper | Approach |
|-------|----------|
| [InverSynth](https://arxiv.org/abs/1812.06349) (2018) | CNN on spectrograms -> quantized params |
| [DiffMoog](https://arxiv.org/abs/2401.12570) (2024) | Differentiable synth + encoder network |
| [DDSP](https://magenta.tensorflow.org/ddsp-vst-blog) (Google) | Neural net predicts controls for classical DSP |

## Research Log

_Record findings, experiment results, and decisions here as the research progresses._
```

- [ ] **Step 3: Commit**

```bash
git add research/inverse-synth/
git commit -m "feat: scaffold inverse synthesis research sub-project"
```

---

### Task 4: Create CLAUDE.md

**Files:**
- Create: `CLAUDE.md`

- [ ] **Step 1: Create CLAUDE.md**

```markdown
# CLAUDE.md

## Project Structure

This repo has three layers:

| Layer | Location | Nature |
|-------|----------|--------|
| Audio pipeline (MCP server) | `src/audio_analysis_mcp/` | Engineering (not yet scaffolded) |
| Engine detection research | `research/engine-detection/` | Exploratory ML research |
| Inverse synthesis research | `research/inverse-synth/` | Heavy ML research |

Design spec: `docs/superpowers/specs/2026-04-20-research-decomposition-design.md`

Archived original monolithic plan: `archive/7-audio-analysis-mcp.md`

## Dependencies

The pipeline (once built) produces clean isolated notes that feed both research projects:

```
Pipeline tools -> clean audio
Engine Detection -> classifies synth type
Inverse Synthesis -> predicts parameter vector
MCP server exposes both as tools
```

## Workspace Context

Part of `~/test/sounds-and-recreation/`:
- `../keyboards-mcp/` — Keyboard MIDI control (MCP server, TypeScript)
- `../sound-recreation-agent/` — AI agent orchestrator (TypeScript)
- `../macos-packager/` — macOS app packaging

## Research Directories

Large generated files (audio samples, datasets, model checkpoints) are gitignored. See `.gitignore` for patterns.

Each research sub-project has a `README.md` with its goals, phases, and a research log section for recording findings.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add CLAUDE.md reflecting three-layer decomposition"
```