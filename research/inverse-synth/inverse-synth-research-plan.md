# Inverse Synthesis — Research Plan

> Predict synthesizer parameters from audio. This is a heavy ML research project — the plan defines the research pipeline phases, evaluation methodology, and milestones. Specific tool and architecture choices will be made during the research based on experimental results.

## Goal

Given audio + a known synthesis type, predict the parameter vector that would reproduce that sound. One model per synthesis type.

```
Forward (the synth):     param_vector  →  synth_engine  →  audio
Inverse (what we need):  audio         →  trained_model →  param_vector
```

## Why This Is Research

This is an active area of academic research (InverSynth 2018, DiffMoog 2024, DDSP). Key challenges:

1. **The many-to-one problem:** Multiple parameter combinations produce perceptually identical sounds. There isn't a unique correct answer.
2. **Input variability:** Real-world audio has effects, noise, polyphony — the model must see through all of that to the dry timbre.
3. **Evaluation is hard:** Per-parameter MSE doesn't capture whether the sound is "right." Perceptual similarity metrics are needed.
4. **Rendering infrastructure:** Generating training data requires programmatic control of synth engines, which is non-trivial.

## Prior Art

| Paper | Year | Approach | Key insight |
|-------|------|----------|-------------|
| [InverSynth](https://arxiv.org/abs/1812.06349) | 2018 | CNN on spectrograms → quantized params (16 levels) | Classification > regression for synth params |
| [DiffMoog](https://arxiv.org/abs/2401.12570) | 2024 | Differentiable synth + encoder network, signal-chain loss | Self-generated data; loss at every signal chain stage |
| [DDSP](https://magenta.tensorflow.org/ddsp-vst-blog) | Google | Neural net → controls for classical DSP | Interpretable: outputs map to oscillators/filters |

The critical insight we share with all of these: **training data is free** because we generate it by randomizing parameters and rendering through the synth.

## Phase 1: Rendering Infrastructure

**Objective:** Build or integrate at least one synth renderer that can take a parameter vector and produce audio programmatically.

### The rendering problem

To generate training data, we need:
1. A definition of the parameter space (which params, what ranges, which are continuous vs discrete)
2. A renderer that takes a parameter vector + musical content (MIDI notes) and outputs audio
3. The ability to generate thousands of samples efficiently

### Renderer options to explore

| Option | Pros | Cons |
|--------|------|------|
| **Pure Python DSP** | No external deps, full control, easy to debug | Limited realism, have to implement everything |
| **Headless VST plugins** (SurgeXT, Dexed, setBfree) | Realistic sound, real synth engines | Complex setup, platform-specific, harder to parameterize |
| **keyboards-mcp MIDI → audio_render** | Uses real hardware, most realistic | Very slow (real-time rendering), needs physical keyboard |
| **DDSP-style differentiable synth** | Enables gradient-based optimization, not just prediction | Complex to implement, may not match real synth behavior |

### Recommended starting point

Start with the simplest option that produces usable data. Upgrade later if the data quality is a bottleneck.

### Parameter space definition (subtractive — first target)

Define a normalized parameter vector for subtractive synthesis:

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| osc1_shape | discrete | {sine, saw, square, triangle, pulse} | Oscillator 1 waveform |
| osc1_pulse_width | continuous | 0.0-1.0 | Pulse width (if pulse) |
| osc2_shape | discrete | {sine, saw, square, triangle, pulse} | Oscillator 2 waveform |
| osc2_detune | continuous | 0.0-1.0 | Detune from osc1 |
| osc2_level | continuous | 0.0-1.0 | Osc2 mix level |
| lp_cutoff | continuous | 0.0-1.0 | Low-pass filter cutoff (normalized) |
| lp_resonance | continuous | 0.0-1.0 | Filter resonance |
| lp_env_amount | continuous | 0.0-1.0 | Filter envelope depth |
| amp_attack | continuous | 0.0-1.0 | Amplifier attack time |
| amp_decay | continuous | 0.0-1.0 | Amplifier decay time |
| amp_sustain | continuous | 0.0-1.0 | Amplifier sustain level |
| amp_release | continuous | 0.0-1.0 | Amplifier release time |
| filter_attack | continuous | 0.0-1.0 | Filter envelope attack |
| filter_decay | continuous | 0.0-1.0 | Filter envelope decay |
| filter_sustain | continuous | 0.0-1.0 | Filter envelope sustain |
| filter_release | continuous | 0.0-1.0 | Filter envelope release |
| lfo_rate | continuous | 0.0-1.0 | LFO frequency (normalized) |
| lfo_depth | continuous | 0.0-1.0 | LFO modulation depth |
| lfo_target | discrete | {pitch, filter, amplitude} | LFO modulation target |
| noise_level | continuous | 0.0-1.0 | White noise mix level |

This is a starting point — refine based on what the renderer supports and what features actually matter for perceptual similarity.

### Milestone

Can generate 100 (audio, param_vector) pairs for subtractive synthesis. Audio is WAV, params are stored alongside as JSON.

## Phase 2: Dataset Generation Pipeline

**Objective:** Generate large labeled datasets with realistic variation.

### Pipeline

```
for each sample:
  1. Generate random param_vector within synth's valid ranges
  2. Generate random musical content:
     - Single note at random pitch (C2-C6), velocity, duration
     - OR two-note interval at random root
     - OR 3-4 note chord at random root
     - OR short melodic phrase (3-5 notes)
  3. Render dry audio through synth engine
  4. Apply random effects augmentation:
     - Reverb (convolution with random IR, wet/dry 10-60%)
     - Delay (100-500ms, feedback 10-40%)
     - Chorus (rate 0.5-3Hz, depth 20-60%)
     - Compression (ratio 2:1-8:1)
     - EQ (random 2-band boost/cut, +/-6dB)
  5. Apply random noise/degradation:
     - Gaussian noise (SNR 20-50dB)
     - Pink noise / hum (SNR 30-60dB)
     - Stem bleed from other instruments (0-10% level)
  6. Compute mel spectrogram of augmented audio
  7. Store (spectrogram, param_vector) pair
  8. Also store clean dry render (for contrastive loss / audio validation)
```

### Dataset size

- Target: 50K unique patches x ~8 augmented variants = ~400K training pairs
- Start smaller (1K-5K patches) to validate the pipeline and model before scaling

### Storage format

- Audio: WAV files or memory-mapped numpy arrays
- Labels: JSON sidecar per sample with param_vector + metadata
- Spectrograms: precomputed and cached as .npy for training speed

### Milestone

Can generate 5,000 (spectrogram, param_vector) pairs with augmentation. Pipeline is automated and reproducible (seeded RNG).

## Phase 3: Model Architecture Exploration

**Objective:** Find an architecture that predicts parameter vectors from mel spectrograms, starting with subtractive synthesis.

### Baseline approach: CNN + per-param heads

```
Input: Mel spectrogram (1 x 128 x T)
       │
  ┌────┴─────────────────┐
  │  Timbre Encoder       │
  │  CNN backbone         │
  │  + temporal pooling   │  ← Pooling across time: invariant to
  └────┬─────────────────┘    duration and note content
       │
  Timbre Embedding (512-dim)
       │
  ┌────┼─────┬─────┬── ... ──┐
  │    │     │     │         │
 MLP₁ MLP₂ MLP₃  MLP₄    MLPₙ    Per-parameter heads
  │    │     │     │         │
 osc1  osc1  lp    amp     lfo     Parameter predictions
 shape pw    cutoff attack  rate
```

- **Temporal pooling** makes the embedding invariant to duration and note content
- **Per-param heads** allow different loss functions per parameter (MSE for continuous, cross-entropy for discrete)
- Backbone options: ResNet-18 (pretrained), lightweight custom CNN (3-5 layers), or MobileNet

### Alternative approaches to explore

| Approach | When to try | Expected benefit |
|----------|------------|------------------|
| **Simpler baselines first** | Before deep learning | Maybe a small MLP on handcrafted features works well enough? |
| **DiffMoog-style signal-chain loss** | If MSE on params isn't working | Loss at each synthesis stage constrains solution space |
| **Contrastive/triplet learning** | If embedding quality is poor | Forces the embedding to capture timbre identity |
| **Variational (VAE)** | If many-to-one problem is severe | Generates a distribution over plausible param vectors |
| **Quantized prediction (InverSynth)** | If regression is unstable | Discretize params to 16/32 levels, treat as classification |

### Training recipe

- **Loss:** MSE for continuous params + cross-entropy for discrete params + optional contrastive loss on embedding
- **Optimizer:** Adam, lr=1e-3 with cosine annealing
- **Batch size:** 32-128 depending on memory
- **Epochs:** 50-200 with early stopping on validation loss
- **Regularization:** dropout in MLP heads, weight decay

### The many-to-one problem

Multiple parameter combinations produce perceptually identical sounds. Mitigation strategies to explore:
- **Signal-chain loss:** Compute loss at each synthesis stage, not just final audio
- **Contrastive embedding:** Triplet loss clusters perceptually-similar patches
- **Top-K predictions:** Return multiple plausible param vectors ranked by confidence
- **Audio-domain validation:** Render predicted params, measure spectral similarity to input

### Milestone

A trained model for subtractive synthesis that, given a mel spectrogram, outputs a parameter vector. Loss decreases over training. Predictions are in valid ranges.

## Phase 4: Training and Evaluation

**Objective:** Define evaluation metrics, establish baselines, and train models to a useful level of accuracy.

### Evaluation metrics

| Metric | What it measures | How to compute |
|--------|------------------|----------------|
| **Per-param MSE** | How close predicted values are to ground truth | MSE between predicted and actual param vectors |
| **Per-param accuracy** (discrete) | How often discrete params are correct | Exact match rate for waveform type, LFO target, etc. |
| **Round-trip audio similarity** | Does the predicted sound match the input sound? | Render predicted params → compute mel spectrogram → cosine similarity with input spectrogram |
| **Perceptual similarity** | Does it *sound* right to a human? | PESQ, ViSQOL, or learned perceptual metric |
| **Top-K hit rate** | Is the correct answer in the top K predictions? | Compare each of top-K predictions to ground truth |

### Evaluation protocol

1. **Synthetic test set:** Held-out synthetic data from the same distribution as training. This measures raw model performance.
2. **Real-world test set:** Audio from actual hardware recordings (via keyboards-mcp or sample packs). This measures generalization.
3. **Ablation studies:** Remove augmentation components one at a time to measure their contribution.
4. **Failure analysis:** Examine worst-performing samples. Are they pathological parameter combos? Effects artifacts? Dataset issues?

### Baselines to establish

- **Random prediction:** Random param vector → round-trip similarity score (lower bound)
- **Nearest-neighbor in feature space:** Given input spectrogram, find closest training spectrogram, return its params
- **Handcrafted heuristics:** Rule-based param estimation from spectral features (e.g., detect filter cutoff from spectral rolloff)

### Milestone

Quantitative results on all metrics for subtractive synthesis. Clear comparison against baselines. Identification of failure modes and what to improve.

## Phase 5: Generalization to Other Synthesis Types

**Objective:** Expand beyond subtractive to FM and organ synthesis. Assess how much transfers.

### For each new synthesis type:

1. Define the parameter space (different for each type)
2. Build or configure a renderer
3. Generate a dataset
4. Train a model (can the subtractive encoder transfer? or train from scratch?)
5. Evaluate

### FM synthesis specifics

- Parameter space: operator ratios, modulation indices, algorithms, envelopes per operator
- Much larger parameter space than subtractive
- Highly non-linear relationship between params and sound
- May need more training data

### Organ synthesis specifics

- Parameter space: 9 drawbar levels, percussion on/off, vibrato/chorus, rotary speed
- Smaller, more constrained parameter space
- Should be easier than subtractive — more direct mapping between drawbars and harmonics

### Transfer learning question

Does a model trained on subtractive synthesis learn useful audio representations that transfer to FM/organ? Or is each synthesis type so different that we need to train from scratch?

### Milestone

At least 2 synthesis types with trained models and evaluation results. Understanding of cross-type transfer potential.

## Deliverable

Trained model(s) + inference code packaged as a Python module that:
1. Loads a model checkpoint for a given synthesis type
2. Takes an audio file (or mel spectrogram) as input
3. Returns a predicted parameter vector (normalized 0-1) with confidence
4. Can be called by the MCP server's `inverse_synth`, `train_model`, and `list_models` tools

```python
# Target interface
from inverse_synth import InverseSynthModel

model = InverseSynthModel.load("subtractive", "path/to/checkpoint.pt")
result = model.predict("path/to/audio.wav", top_k=3)
# result = {
#   "synth_type": "subtractive",
#   "predictions": [
#     {"confidence": 0.87, "vector": [0.99, 0.50, ...], "labels": ["osc1_shape", "osc1_pw", ...]},
#     {"confidence": 0.72, "vector": [0.85, 0.45, ...], "labels": [...]},
#     ...
#   ]
# }
```

## Dependencies

```
torch>=2.0               # Model training
torchaudio>=2.0          # Mel spectrogram transforms
numpy>=1.24
scipy>=1.10
librosa>=0.10.0          # Audio feature extraction
soundfile>=0.12          # WAV I/O
matplotlib>=3.7          # Visualization
jupyter>=1.0             # Notebooks
tensorboard>=2.14        # Training monitoring (optional)
```

Rendering backends will add their own dependencies depending on what approach wins in Phase 1.

## Open Questions (to resolve during research)

- **Rendering backend:** What's the most practical way to generate training data at scale?
- **Parameter vector → device params:** How to map a normalized vector to specific hardware params? This is a shared problem with the broader system — vector DB lookup? Per-device mapping model? Direct name matching?
- **Minimum data:** How many training samples are actually needed for useful predictions?
- **Audio duration:** What's the minimum clip length for reliable prediction? Can the model work on 0.5s clips? 2s? 5s?
- **Real-world gap:** How well do models trained on synthetic data perform on real-world recordings?
- **Multi-note input:** Should the model process isolated single notes (cleaner) or can it handle polyphonic input directly?
- **Incremental learning:** Can a deployed model be fine-tuned on new synth types without forgetting old ones?