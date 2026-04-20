# Synth Engine Detection — Research Plan

> Classify audio segments by synthesis engine type. This is an open research problem — the plan defines the exploration phases, not a fixed implementation path.

## Goal

Given an isolated note or audio stem, predict which synthesis engine category produced it:

| Category | Examples | Sound generation |
|----------|----------|-----------------|
| **Subtractive** | Prophet-6, Moog, Juno-106/60 | Oscillators → filter → amplifier |
| **FM** | DX7, FM8 | Carrier/modulator operators, algorithms |
| **Organ / Additive** | Hammond B3, drawbar organs | Drawbar harmonics + percussion + rotary |
| **Wavetable** | PPG Wave, Waldorf | Wavetable position scanning + filter |
| **Sample-based** | Acoustic piano, Rhodes, Wurlitzer, Clavinet | Sample playback + effects modeling |

This feeds into the inverse synthesis pipeline: you need to know the engine type before you can predict parameters.

## Why This Is Research

There is no established method for synthesis type classification from audio. The question is fundamentally: **what acoustic features distinguish these engine types?**

Some hypotheses to test:
- Subtractive synths have characteristic filter resonance signatures
- FM sounds have inharmonic partials at non-integer ratios
- Organ sounds have strict harmonic-series partials (drawbar levels)
- Wavetable synths show spectral morphing over time
- Sample-based instruments have natural microvariation and noise

These may or may not hold across the full range of patches. That's what the research needs to determine.

## Phase 1: Data Collection

**Objective:** Build a labeled dataset of audio samples with ground-truth engine type labels.

### Approach A: Software synth rendering (synthetic data)

For each engine type, use a software synth to render patches with known parameters:

| Engine type | Software synth candidates |
|-------------|--------------------------|
| Subtractive | SurgeXT, TAL-NoiseMaker, Dexed (in subtractive mode) |
| FM | Dexed (DX7 emulator), FM8 |
| Organ | setBfree (Hammond emulator), HammondEggs |
| Wavetable | Vital, WaveEdit |
| Sample-based | FluidSynth with SoundFont, sfizz |

For each synth:
- Render 200-500 patches (use factory presets + random parameter variations)
- Each patch rendered at multiple pitches (C2, C3, C4, C5), velocities, and durations
- Render dry (no effects) for clean baseline, plus with-effects variants

### Approach B: Real-world recordings

- Record or source samples from actual hardware (keyboards-mcp can capture from connected synths)
- Use publicly available sample packs with known provenance
- These serve as a validation set — train on synthetic, test on real

### Dataset size target

- ~500 unique patches per category (2,500 total)
- ~10 variants per patch (pitch, velocity, duration) = ~25,000 audio clips
- Split: 70% train, 15% validation, 15% test

## Phase 2: Feature Exploration

**Objective:** Understand which acoustic features discriminate between engine types.

### Candidate features to extract (using librosa + custom code)

**Spectral features:**
- Spectral centroid, bandwidth, rolloff
- Spectral flatness (noisy vs tonal)
- Spectral contrast (valley-to-peak ratio per frequency band)
- Mel-frequency cepstral coefficients (MFCCs, 13-20 coefficients)

**Harmonic features:**
- Harmonic-to-noise ratio (HNR)
- Inharmonicity (deviation of partials from integer harmonics)
- Number of significant harmonics
- Harmonic amplitude profile (relative levels of first N harmonics)
- Harmonic spacing regularity

**Temporal features:**
- Attack time, decay time, sustain level, release time (ADSR)
- Spectral flux (rate of spectral change over time)
- Onset strength
- Temporal centroid (where the energy is concentrated in time)

**Modulation features:**
- Amplitude modulation depth and rate (tremolo)
- Frequency modulation depth and rate (vibrato)
- Spectral modulation (chorus, phaser signatures)
- Spectral morphing rate (wavetable scanning detection)

### Exploration methodology

Use Jupyter notebooks for:
1. Extract all candidate features for every sample in the dataset
2. Visualize feature distributions per engine type (box plots, violin plots)
3. Compute feature importance (mutual information, correlation with labels)
4. Dimensionality reduction (t-SNE, UMAP) to see if engine types cluster
5. Identify which feature pairs/combinations create the cleanest separation
6. Document which features are useful and which are not (in README research log)

## Phase 3: Handcrafted-Features Classifier

**Objective:** Build a classical ML classifier using the most discriminative features.

### Pipeline

1. Feature extraction: select top N features from Phase 2 exploration
2. Feature vector: concatenate selected features into a fixed-size vector per sample
3. Train classifiers:
   - **LightGBM** — gradient-boosted decision trees, handles mixed feature types well
   - **Random Forest** — good baseline, less prone to overfitting
   - **Decision Tree** — most interpretable, useful for understanding what the model learned
4. Hyperparameter tuning: grid search or Optuna on validation set
5. Evaluate on held-out test set

### Evaluation metrics

- **Accuracy** (overall)
- **Per-class precision, recall, F1** — which categories are hardest?
- **Confusion matrix** — which pairs get confused most often?
- **Feature importance** from the tree model — which features drive decisions?

### Expected challenges

- **Subtractive vs wavetable:** Many wavetable patches sound like filtered oscillators
- **Sample-based vs subtractive:** Modern sample engines can sound very synthetic
- **Effects confounding:** Heavy effects (distortion, heavy reverb) may obscure engine-type cues

### Deliverable

A trained LightGBM/RF model + feature extraction pipeline that can classify a new audio clip.

## Phase 4: End-to-End CNN Classifier

**Objective:** Compare a deep learning approach against the handcrafted-features baseline.

### Architecture

- Input: mel spectrogram (128 bins x T frames, standardized length via padding/truncation)
- Backbone: small CNN (3-5 conv layers) or ResNet-18 (pretrained on ImageNet, fine-tuned)
- Global average pooling → 5-class softmax
- Training: cross-entropy loss, Adam optimizer, learning rate schedule

### Training

- Same train/val/test split as Phase 3
- Data augmentation: random pitch shift, time stretch, noise injection, gain variation
- Early stopping on validation loss
- Compare accuracy against handcrafted-features classifier

### When to prefer CNN over handcrafted

- CNN wins: if feature engineering hits a ceiling and more data helps
- Handcrafted wins: if interpretability matters, dataset is small, or CNN overfits

Both approaches may coexist — handcrafted for explainability, CNN for accuracy.

## Phase 5: Evaluation and Taxonomy Refinement

**Objective:** Assess overall classifier quality and decide if the 5-class taxonomy is right.

### Questions to answer

1. What is the overall accuracy on real-world audio (not just synthetic)?
2. Which category pairs are most confused? Does the confusion suggest merging categories?
3. Does wavetable need to be its own category, or does it cluster with subtractive?
4. Are there sub-categories within sample-based that behave very differently (piano vs Rhodes)?
5. How robust is classification when effects are present (reverb, distortion, compression)?
6. What's the minimum audio duration needed for reliable classification?

### Taxonomy decisions

Based on evaluation, decide:
- Keep 5 classes as-is
- Merge classes that can't be reliably distinguished
- Add classes if data shows clear sub-clusters
- Define a "confidence threshold" below which the classifier says "unsure"

## Deliverable

A trained classifier (whichever approach wins) packaged as a Python module that:
1. Takes an audio file path as input
2. Returns predicted engine type + confidence score
3. Can be loaded and called by the MCP server's `engine_detect` tool

```python
# Target interface
from engine_detection import EngineDetector

detector = EngineDetector.load("path/to/model")
result = detector.predict("path/to/audio.wav")
# result = {"engine_type": "subtractive", "confidence": 0.87, "scores": {"subtractive": 0.87, "fm": 0.05, ...}}
```

## Dependencies

```
librosa>=0.10.0          # Feature extraction
numpy>=1.24
scipy>=1.10
scikit-learn>=1.3        # Random forest, metrics, preprocessing
lightgbm>=4.0            # Gradient boosted trees
torch>=2.0               # CNN experiments
torchaudio>=2.0          # Mel spectrogram transforms
matplotlib>=3.7          # Visualization
seaborn>=0.12            # Statistical plots
jupyter>=1.0             # Notebooks
optuna>=3.0              # Hyperparameter tuning (optional)
```

## Open Questions (to resolve during research)

- What rendering backends are most practical for dataset generation?
- How much data is enough? Start with 2,500 patches, scale up if needed.
- Should we use transfer learning from audio classification models (e.g., PANNs, AudioSet-pretrained)?
- Can the classifier handle audio with multiple overlapping synth types (e.g., organ + strings)?
- How well does a classifier trained on software synths transfer to real hardware recordings?