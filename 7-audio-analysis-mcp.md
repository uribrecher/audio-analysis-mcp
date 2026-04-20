# Audio Analysis MCP Server

> **Execution order: 7 of 7** — Separate Python repo. Phase 1 (audio pipeline) has no dependencies on keyboards-mcp plans. Phase 2+ (inverse synthesis) needs the architecture plan for the parameter contract. Can be developed in parallel with plans 3-6.

## Context

A synthesizer is a mathematical function: `f(params) → audio`. Recreating a song's keyboard sound means solving the inverse: `f⁻¹(audio) → params`. Today the AI agent can research gear and set parameters, but cannot listen — requiring human ears in the loop.

This plan adds a standalone MCP server with two layers:
1. **Audio pipeline tools** — stem separation, audio capture, spectral analysis
2. **Inverse synthesis models** — ML models trained per synth type that directly predict parameter vectors from audio, replacing the iterative guess-and-check loop

## The Inverse Synthesis Problem

```
Forward (the synth):     param_vector  →  synth_engine  →  audio
Inverse (what we need):  audio         →  trained_model →  param_vector
```

### Why ML, not iterative A/B

An iterative loop (render → compare → tweak → repeat) is slow and fragile — it requires an agent to interpret spectral diffs and decide which knob to turn. A trained inverse model produces the parameter vector in one forward pass.

### Prior art

| Paper | Approach | Key insight |
|-------|----------|-------------|
| [InverSynth](https://arxiv.org/abs/1812.06349) (2018) | CNN on spectrograms → quantized params (16 levels per param) | Classification > regression for synth params |
| [DiffMoog](https://arxiv.org/abs/2401.12570) (2024) | Differentiable synth + encoder network, signal-chain loss | Self-generated training data; loss at every stage of signal chain |
| [DDSP](https://magenta.tensorflow.org/ddsp-vst-blog) (Google) | Neural net predicts controls for classical DSP elements | Interpretable: network outputs map directly to oscillators/filters |

### Training data is free

The critical insight: we don't need labeled real-world audio. We generate unlimited training data by randomizing the parameter vector and rendering through the synth:

```
for i in 1..N:
    params = random_param_vector(synth_definition)
    audio  = render(synth_engine, params, note=C4, duration=2s)
    dataset.append((audio, params))
```

This is fully self-supervised — the synth itself is the ground truth.

### One model per synth type

Different synthesis types have fundamentally different parameter spaces and audio characteristics. A model trained on subtractive synthesis won't generalize to FM or organ. We train inverse models only for **synthesized** sound categories:

#### Inverse synthesis candidates (trainable)

| Synthesis type | Example hardware | Parameter space character |
|---------------|-----------------|--------------------------|
| Subtractive | Prophet-6, Moog, JUNO-106/60 | Oscillator shapes, filter cutoff/resonance, envelopes |
| FM | DX7, FM8 | Operator ratios, modulation indices, algorithms |
| Organ (drawbar/additive) | Hammond B3, Nord Organ engine | Drawbar levels, percussion, vibrato/chorus, rotary speed |
| Wavetable | PPG Wave, Waldorf | Wavetable position, modulation sources, filter |

Each model learns the specific `f⁻¹` for its synthesis type. Could potentially fine-tune per specific synth model (e.g., Prophet-6 vs Moog Sub 37).

#### NOT inverse synthesis candidates (preset/sample matching only)

Acoustic and electro-mechanical keyboard instruments are reproduced via **sample playback engines** — recordings of real instruments at multiple velocities/pitches, plus effects modeling. There is no synthesizable parameter vector to predict.

| Instrument category | Sound generation mechanism | Reproduction strategy |
|-------------------|--------------------------|----------------------|
| **Acoustic piano** | Felt hammers strike metal strings; resonance box + soundboard | Dedicated piano engine (Nord Piano, Roland RD Piano); match sample set + adjust EQ/resonance |
| **Harpsichord** | Strings plucked by quills; no velocity control | Piano/sample engine; select harpsichord sample set |
| **Clavinet** | Rubber hammers strike strings; piezo pickups per string group | Piano/EP engine; select clavinet sample + dial in pickup model and wah/phaser effects |
| **Fender Rhodes** | Metal tines + tonebars; piezo pickup per tine | Piano/EP engine (Nord Piano, Roland RD Piano); select Rhodes sample + amp/tremolo/chorus |
| **Wurlitzer** | Metal reeds; piezo pickup per reed | Piano/EP engine; select Wurlitzer sample + overdrive/tremolo |

**Key insight:** For electro-mechanical instruments (Rhodes, Wurlitzer, Clavinet), the **pickup/amp modeling and effects chain** are often more important to the final sound than the base sample. Modern keyboard engines (Nord Piano, Roland RD Piano) include dedicated controls for these.

Different keyboard manufacturers have their own piano/EP/organ engine implementations — Yamaha, Roland, Korg, Nord all approach sample playback and modeling differently. Use `list_synth_engines` (keyboards-mcp) to discover which engines a connected device supports.

## Decision: Pure Python

The audio/ML ecosystem lives in Python (PyTorch, Demucs, librosa). Use the Python MCP SDK (`mcp` package) with `stdio_server` transport.

## Project Structure

```
audio-analysis-mcp/
  pyproject.toml
  README.md
  CLAUDE.md
  .mcp.json
  src/
    audio_analysis_mcp/
      __init__.py
      server.py                        # MCP server, tool registration
      workspace.py                     # Temp/workspace directory management
      tools/
        __init__.py
        fetch_audio.py                 # YouTube download / local file import
        stem_separate.py               # Demucs stem separation
        spectrum_analyze.py            # Spectral feature extraction
        audio_compare.py               # A/B spectral diff (fallback for untrained synths)
        audio_render.py                # Capture audio from system device
        note_transcribe.py             # Polyphonic transcription via Basic Pitch
        note_isolate.py                # Score-informed source separation via nussl
        inverse_synth.py               # ML-based param prediction
        train_model.py                 # Generate dataset + train inverse model
        list_models.py                 # List available trained models
      analysis/
        __init__.py
        spectral.py                    # Librosa-based feature extraction
        comparison.py                  # A/B spectral diff
        transcription.py               # Basic Pitch polyphonic transcription + polyphony profiling
        note_isolation.py              # nussl time-frequency masking + quality assessment
      models/
        __init__.py
        architecture.py               # Shared encoder architecture (ResNet/CNN backbone + MLP heads)
        dataset.py                     # Dataset generation: random params → render → spectrogram pairs
        augmentation.py                # Effects chain, noise injection, stem bleed simulation
        trainer.py                     # Training loop (param loss + contrastive loss), checkpointing
        inference.py                   # Load trained model, predict params from audio
        synth_renderers/
          __init__.py
          base.py                      # Abstract renderer interface
          subtractive.py               # Renders subtractive synth patches (e.g., via SurgeXT, Dexed)
          fm.py                        # FM synthesis renderer
          organ.py                     # Additive/organ renderer
      audio/
        __init__.py
        capture.py                     # sounddevice recording
        normalize.py                   # WAV normalization
  trained_models/                      # Stored model checkpoints
    subtractive_prophet6/
    fm_dx7/
    organ_b3/
  tests/
    test_spectral.py
    test_dataset_generation.py
    test_inference.py
```

## Dependencies

```toml
[project]
name = "audio-analysis-mcp"
requires-python = ">=3.10"
dependencies = [
  "mcp>=1.0.0",
  "demucs>=4.0.0",         # Stem separation
  "librosa>=0.10.0",       # Spectral analysis
  "torch>=2.0",            # ML models (already pulled by Demucs)
  "torchaudio>=2.0",       # Audio transforms, mel spectrograms
  "numpy>=1.24",
  "scipy>=1.10",
  "soundfile>=0.12",       # WAV I/O
  "sounddevice>=0.4",      # Audio capture
  "yt-dlp>=2024.0",        # YouTube download
  "basic-pitch>=0.3.0",    # Polyphonic transcription (Spotify, MIT license)
  "nussl>=1.1.0",          # Score-informed source separation (time-frequency masking)
  "pydantic>=2.0",         # Structured output schemas
]
```

## Tools

### Audio Pipeline Tools

#### 1. `fetch_audio`

Download from YouTube or import a local file. Normalize to 44.1kHz 16-bit WAV.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| source | string | yes | YouTube URL or local file path |
| start_time | float | no | Trim start (seconds) |
| duration | float | no | Trim duration (seconds) |

**Returns:** Path to normalized WAV in `{workspace}/fetched/`.

#### 2. `stem_separate`

Demucs stem separation → vocals, drums, bass, other (keyboards/synths).

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| audio_path | string | yes | Path to audio file |
| model | string | no | Demucs model (default: `htdemucs`) |

**Returns:** Paths to all stem WAV files. Cached by input hash.

**Long-running:** 1-5 min. Runs as async subprocess with 10 min timeout.

#### 3. `audio_render`

Capture audio from a system audio device (BlackHole, USB audio).

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| duration | float | yes | Recording duration (seconds) |
| device | string | no | Audio input device name/index |
| list_devices | bool | no | Just list available devices |

**Returns:** Path to recorded WAV, or device list.

#### 4. `spectrum_analyze`

Extract spectral features — useful for diagnostics and for synths without a trained model.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| audio_path | string | yes | Path to audio file |
| start_time | float | no | Analysis window start |
| duration | float | no | Analysis window (default: 5s) |

**Returns:** Harmonic profile, spectral envelope, ADSR, modulation detection, synth hints (JSON).

#### 5. `audio_compare`

A/B spectral diff — fallback for iterative matching when no trained model exists.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| target_path | string | yes | Reference audio |
| rendered_path | string | yes | Synthesized attempt |

**Returns:** Similarity scores, frequency band diffs, prioritized action items (JSON).

### Note-Level Extraction Tools (Score-Informed Source Separation)

These tools extract clean, individual notes from polyphonic keyboard stems. They sit between stem separation and inverse synthesis — producing single-note audio that is far more reliable input for synthesis detection and parameter prediction.

#### 6. `note_transcribe`

Polyphonic transcription using **Spotify Basic Pitch** (open source, MIT). Extracts MIDI note events (pitch, onset, offset, velocity) from a polyphonic audio stem. Also computes a polyphony profile — how many notes overlap at each point in time.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| audio_path | string | yes | Path to audio file (typically the "other" keyboard stem) |

**Returns:**

```json
{
  "midi_path": "{workspace}/transcriptions/other_transcription.mid",
  "note_events": [
    {"index": 0, "pitch_midi": 60, "onset_sec": 0.12, "offset_sec": 0.85, "velocity": 92, "polyphony_at_onset": 1},
    {"index": 1, "pitch_midi": 64, "onset_sec": 0.50, "offset_sec": 1.20, "velocity": 87, "polyphony_at_onset": 2},
    ...
  ],
  "polyphony_profile": {
    "max_polyphony": 6,
    "monophonic_windows": [{"start": 0.0, "end": 0.49}, ...],
    "low_polyphony_windows": [{"start": 2.1, "end": 3.4, "max_voices": 2}, ...],
    "high_polyphony_windows": [{"start": 5.0, "end": 8.2, "max_voices": 6}, ...]
  },
  "candidate_notes": [3, 7, 12, 18, 25],
  "candidate_selection_criteria": "monophonic or low-polyphony, duration > 0.5s, spread across pitch range"
}
```

The `candidate_notes` field pre-selects the best notes for isolation based on:
- Monophonic or low-polyphony windows (cleanest signal)
- Sufficient duration (> 0.5s) to capture full ADSR envelope
- Temporal isolation (minimal overlap with adjacent notes)
- Pitch range spread (capture timbre at different registers)

#### 7. `note_isolate`

Score-informed source separation using **nussl** time-frequency masking. Uses the MIDI transcription to guide a soft mask in the STFT domain, isolating individual notes from the polyphonic mix. For monophonic windows, uses simple time-slice extraction (no masking needed).

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| audio_path | string | yes | Path to audio file |
| transcription_path | string | yes | Path to MIDI transcription from `note_transcribe` |
| note_indices | int[] | yes | Indices of notes to isolate (from `note_events`) |
| assess_quality | bool | no | Run effects/distortion triage on each note (default: true) |

**Returns:**

```json
{
  "isolated_notes": [
    {
      "index": 3,
      "path": "{workspace}/isolated_notes/note_003.wav",
      "pitch_midi": 67,
      "duration_sec": 1.2,
      "isolation_method": "time_slice",
      "quality_score": 0.92,
      "detected_effects": [],
      "usable": true
    },
    {
      "index": 7,
      "path": "{workspace}/isolated_notes/note_007.wav",
      "pitch_midi": 72,
      "duration_sec": 0.8,
      "isolation_method": "nussl_tf_mask",
      "quality_score": 0.78,
      "detected_effects": ["reverb"],
      "usable": true
    },
    {
      "index": 12,
      "path": "{workspace}/isolated_notes/note_012.wav",
      "pitch_midi": 55,
      "duration_sec": 0.6,
      "isolation_method": "nussl_tf_mask",
      "quality_score": 0.31,
      "detected_effects": ["heavy_distortion"],
      "usable": false,
      "reason": "Heavy distortion — non-invertible; skip for inverse synthesis"
    }
  ],
  "recommended_for_analysis": [3, 7]
}
```

**Quality assessment** (when `assess_quality=true`):

| Condition | Detection | Action |
|-----------|----------|--------|
| Clean | Clear harmonics, low spectral spread | Best candidates — use directly |
| Reverb/delay | Energy after note-off, comb-filter signatures | Usable if attack transient is clean |
| Chorus/modulation | Spectral smearing, beating | Usable — note the modulation rate |
| Heavy distortion | Dense inharmonic partials, intermodulation | **Flag unusable** — distortion is non-invertible |
| Masking artifacts | Phase cancellation, hollow sound | Discard — try a different note |

### Inverse Synthesis Tools

#### 8. `inverse_synth`

**The core tool.** Given audio and a target synthesis type, predict a raw parameter vector.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| audio_path | string | yes | Path to audio file (stem or recording) |
| synth_type | string | yes | Synthesis type: `subtractive`, `fm`, `organ` |
| top_k | int | no | Return top K predictions ranked by confidence (default: 1) |

**Returns:** A raw numeric vector — not device-specific parameter names. The vector represents the synthesis type's parameter space (e.g., subtractive: oscillator shapes, filter cutoff, envelopes).

```json
{
  "synth_type": "subtractive",
  "predictions": [
    {
      "confidence": 0.87,
      "vector": [0.99, 0.50, 0.99, 0.59, 0.72, 0.09, 0.02, 0.99, ...],
      "vector_labels": ["osc1_shape", "osc1_pulse_width", "osc2_shape", "osc2_freq", "lp_freq", "lp_resonance", "vca_env_attack", "vca_env_sustain", ...],
      "notes": "Pulse wave oscillators with moderate LP filter — consistent with Prophet-style organ pad"
    }
  ],
  "model_version": "v1.2",
  "training_samples": 50000
}
```

**Design decisions:**
- **One model per synthesis type**, not per hardware device. A `subtractive` model covers Prophet-6, Moog, Juno-X, etc.
- **Output is a normalized vector (0.0-1.0)**, not device-specific 0-127 values. The vector represents abstract synthesis parameters.
- **Mapping vector → device params is an open research problem.** Options under investigation include: vector DB lookup against known patches, per-device mapping models, or direct parameter-name matching when the device's params align with the vector labels.
- **Constrained pairing:** Each synthesis type model should only be paired with devices of the matching type. A sample-based keyboard (e.g., Nord piano/sample engine) is NOT a valid target for `inverse_synth` — use the fallback research workflow (Step 4b) instead.

**Implementation:**
1. Convert audio to mel spectrogram (standardized input representation)
2. Load trained model checkpoint for the specified synthesis type
3. Forward pass → raw parameter predictions (0-1 normalized)
4. Return as-is — scaling to device-specific ranges happens downstream
5. Optionally return multiple predictions if `top_k > 1` (useful when the many-to-one problem means multiple param combos could match)

#### 9. `train_model`

Generate training data and train (or fine-tune) an inverse model for a synthesis type.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| synth_type | string | yes | Synthesis type: `subtractive`, `fm`, `organ` |
| num_samples | int | no | Training samples to generate (default: 50000) |
| epochs | int | no | Training epochs (default: 100) |
| resume_from | string | no | Checkpoint path to resume/fine-tune from |

**Returns:** Training metrics (loss curve, validation accuracy), model checkpoint path.

**Dataset generation pipeline:**
```
for each sample:
  1. Generate random param_vector within synth's valid ranges
  2. Generate random musical content (pitch, polyphony, duration, velocity)
  3. Render dry audio through synth engine
  4. Apply random effects augmentation (reverb, chorus, compression, EQ)
  5. Apply random noise/degradation (hiss, hum, stem bleed)
  6. Compute mel spectrogram of augmented audio
  7. Store (spectrogram, param_vector) pair
  8. Also store clean dry render (for contrastive loss triplet mining)
```

Each param vector should produce ~5-10 augmented variants (different notes, effects, noise levels) to train invariance. 50K unique patches × 8 variants = 400K training pairs.

**Rendering backends** (in `models/synth_renderers/`):
- Software synths via headless plugins: SurgeXT (subtractive), Dexed (FM), setBfree (organ)
- Or: keyboards-mcp itself — send params via MIDI, capture via `audio_render`, but much slower
- Or: pure Python DSP (simplest for v1 — implement basic oscillators + filters in numpy)

**Training approach:**
- Input: mel spectrogram (128 bins x T frames)
- Backbone: ResNet-18 or lightweight CNN with temporal pooling
- Per-parameter MLP heads (one per param)
- Continuous params: sigmoid output + MSE loss
- Discrete params (waveform type, filter type): classification + cross-entropy loss
- Contrastive/triplet loss on the embedding layer (same patch = close, different patch = far)
- Validation: render predicted params, compare audio via spectral similarity

#### 10. `list_models`

List available trained inverse models with their metadata.

**Returns:**
```json
{
  "models": [
    {
      "synth_type": "subtractive",
      "version": "v1.2",
      "vector_size": 31,
      "vector_labels": ["osc1_shape", "osc1_pulse_width", ...],
      "training_samples": 50000,
      "validation_accuracy": 0.84,
      "checkpoint_path": "trained_models/subtractive/v1.2.pt"
    }
  ]
}
```

## Model Architecture Detail

### The Timbre Embedding Problem

A synth patch defines a **timbre** — the tonal character independent of which notes are played, how many notes sound simultaneously, how long they sustain, or what effects are applied downstream. The model must learn an embedding (latent vector) that captures this timbre identity while being invariant to everything else.

**Variables the embedding must be invariant to:**

| Variable | Real-world reality | Training data strategy |
|----------|-------------------|----------------------|
| **Pitch** | Stems contain melodies and bass lines across the full range | Render each patch at multiple random pitches (C2-C6), not just C4 |
| **Polyphony** | Chord progressions, not single notes | Render monophonic, 2-note intervals, 3-4 note chords, full voicings |
| **Duration** | Notes from staccato 100ms to held pads | Vary note lengths: 0.1s, 0.5s, 1s, 2s, 4s |
| **Velocity** | Players hit keys at varying intensity | Randomize MIDI velocity (40-127) per note |
| **Noise** | Audio interface hiss, cable noise, preamp coloring | Add Gaussian noise, pink noise, hum (50/60Hz) at varying SNR levels |
| **Effects (wetness)** | Reverb, delay, chorus, compression from mixing | Apply random effect chains to rendered audio (see below) |
| **Stem bleed** | Demucs isn't perfect — other instruments leak into "other" stem | Mix in small amounts of drums/bass/vocals stems as contamination |

### Data Augmentation Pipeline

Training on clean monophonic C4 renders produces a model that only works on clean monophonic C4 input. The training data must span the full variability the model will encounter at inference time.

**Level 1 — Musical variation (during rendering):**
```
for each sample:
  params = random_param_vector()
  notes  = random_musical_content()   # See below
  audio  = render(synth, params, notes)
```

Where `random_musical_content()` produces one of:
- Single note at random pitch (C2-C6), random velocity, random duration
- Two-note interval (3rd, 5th, octave) at random root
- 3-4 note chord (major, minor, 7th voicings) at random root
- Short melodic phrase (3-5 notes, random rhythm)

**Level 2 — Effects augmentation (post-render):**

Apply random subsets of these to the dry render:
- Reverb (convolution with random IR, wet/dry 10-60%)
- Delay (100-500ms, feedback 10-40%)
- Chorus (rate 0.5-3Hz, depth 20-60%)
- Compression (ratio 2:1-8:1, threshold -20 to -6dB)
- EQ (random 2-band boost/cut, ±6dB)
- Saturation/overdrive (mild, 5-20%)

This teaches the model to "hear through" the effects to the dry timbre underneath.

**Level 3 — Noise and degradation (post-effects):**
- Additive Gaussian noise (SNR 20-50dB)
- Pink noise / hum (SNR 30-60dB)
- Bandpass filtering (simulating lo-fi recording)
- Mix in small stem bleed from other instruments (0-10% level)

### Architecture

The model has two stages: **timbre encoder** (produces the embedding) and **parameter decoder** (maps embedding to param vector).

```
Input: Mel spectrogram (1 x 128 x T)
        │
  ┌─────┴──────────────────┐
  │  Timbre Encoder         │
  │  CNN/ResNet backbone    │
  │  + temporal pooling     │  ← Pooling across time makes it
  └─────┬──────────────────┘    duration/note-count invariant
        │
  Timbre Embedding (512-dim)   ← This vector IS the patch identity
        │                        Same patch → similar vectors regardless
        │                        of notes, polyphony, effects, noise
  ┌─────┼──────┬──────┬─── ... ──┐
  │     │      │      │          │
 MLP₁  MLP₂  MLP₃  MLP₄      MLPₙ     Per-parameter heads
  │     │      │      │          │
 osc1   osc1   osc2   lp       vca      Parameter predictions
 shape  pw     freq   freq     attack
```

**Key design: temporal pooling.** The backbone produces per-frame features. A global pooling layer (average + attention-weighted) collapses the time dimension into a fixed-size embedding. This is what makes the embedding invariant to duration and note content — it extracts "what kind of sound is this" regardless of "what is being played."

**Training loss — multi-component:**
- **Parameter loss** (primary): MSE for continuous params, cross-entropy for discrete
- **Triplet/contrastive loss** (embedding quality): same patch with different notes/effects should have closer embeddings than different patches
  - Anchor: patch A, chord voicing, dry
  - Positive: patch A, single note, with reverb
  - Negative: patch B, chord voicing, dry
  - This explicitly trains invariance into the embedding space
- **Audio reconstruction loss** (validation): render predicted params, measure spectral similarity to clean dry version

### The Many-to-One Problem

Multiple parameter combinations can produce perceptually identical sounds (e.g., two detuned saws vs. one saw + chorus). Mitigation strategies:
- **Signal-chain loss** (from DiffMoog): loss computed at each stage of the synth, not just final output — constrains the solution space
- **Contrastive embedding**: the triplet loss naturally clusters perceptually-similar patches together, accepting that they may have different param vectors
- **Top-K predictions**: return multiple plausible param vectors, ranked by confidence
- **Audio-domain validation**: after prediction, render the predicted params and verify via `audio_compare`

### Inference on Real Audio

At inference time, the model receives **clean isolated notes** from the note-level extraction pipeline (tools 6-7), not raw polyphonic stems. This dramatically improves input quality:

**Before note-level extraction** (raw stem input):
- Polyphonic content (chord progressions) — model must disentangle
- Effects from mixing (reverb, compression, EQ)
- Stem separation artifacts (bleed from other instruments)
- Recording noise

**After note-level extraction** (isolated note input):
- Single-note audio (monophonic or cleanly masked)
- Effects may still be present but are easier to see through on a single note
- Quality-assessed — heavily distorted or artifact-laden notes are excluded

The model's augmentation-trained invariance still helps with residual effects and noise, but the cleaner input means higher confidence predictions. Running inference on multiple isolated notes and comparing results provides an additional consistency check.

## Workspace

```
~/.audio-analysis-mcp/
  workspace/
    fetched/          # Downloaded/imported audio
    stems/            # Demucs output
    transcriptions/   # Basic Pitch MIDI + note event JSON
    isolated_notes/   # Per-note WAV files from nussl masking
    rendered/         # Captured synth recordings
  trained_models/     # Model checkpoints
  training_data/      # Generated datasets (can be large)
```

## MCP Configuration

```json
{
  "mcpServers": {
    "audio-analysis-mcp": {
      "command": "/path/to/audio-analysis-mcp/.venv/bin/python",
      "args": ["-m", "audio_analysis_mcp"]
    }
  }
}
```

## Agent Workflow (with trained model)

```
1. fetch_audio("youtube.com/watch?v=...")         → full_mix.wav
2. stem_separate(full_mix.wav)                     → other.wav (keyboards)
3. note_transcribe(other.wav)                      → transcription.mid + polyphony profile
4. note_isolate(other.wav, transcription.mid, [candidates]) → clean isolated notes
5. inverse_synth(isolated_note.wav, "subtractive") → raw parameter vector (0-1)
   (run on multiple clean notes, compare predictions for consistency)
6. Agent maps vector to target device params       → (open research problem)
7. Agent calls keyboards-mcp set_parameters(...)   → synth is configured
8. (Optional) audio_render + audio_compare         → validate & fine-tune
```

Steps 1-5 replace the research + manual patch design process. Steps 3-4 (note-level extraction) are critical for feeding clean input to the inverse model. Step 6 (vector → device params) is under active research.

## Agent Workflow (without trained model — fallback)

```
1. fetch_audio → stem_separate → other.wav
2. note_transcribe(other.wav)                      → transcription.mid + candidates
3. note_isolate(other.wav, transcription.mid, [...]) → clean isolated notes
4. spectrum_analyze(isolated_note.wav)              → spectral features + synth hints
5. Agent uses synth hints to set initial params via keyboards-mcp
6. audio_render → audio_compare                    → spectral diff + action items
7. Agent adjusts params based on action_items, repeat 6-7
```

## Implementation Sequence

### Phase 1: Audio Pipeline (MCP tools, no ML)
1. Scaffold: `pyproject.toml`, project structure, `server.py`
2. `workspace.py`: directory management
3. `fetch_audio`: YouTube + local file
4. `stem_separate`: Demucs subprocess with caching
5. `audio_render`: device listing + capture
6. `spectrum_analyze`: spectral features + synth hints
7. `audio_compare`: A/B diff + action items

### Phase 1.5: Note-Level Extraction (Score-Informed Source Separation)
8. `analysis/transcription.py`: Basic Pitch integration + polyphony profiling + candidate selection
9. `note_transcribe` tool: wire transcription to MCP
10. `analysis/note_isolation.py`: nussl time-frequency masking + time-slice extraction + quality assessment
11. `note_isolate` tool: wire isolation to MCP

### Phase 2: Inverse Synthesis Framework
12. `models/dataset.py`: dataset generation pipeline
13. `models/synth_renderers/base.py`: abstract renderer interface
14. `models/synth_renderers/subtractive.py`: first renderer (pure Python DSP or SurgeXT)
15. `models/architecture.py`: CNN backbone + per-param MLP heads
16. `models/trainer.py`: training loop with validation
17. `models/inference.py`: load checkpoint, predict

### Phase 3: First Trained Model (subtractive)
18. Define the subtractive synthesis parameter vector (oscillators, filter, envelopes, LFO)
19. Generate 50K samples using subtractive renderer
20. Train inverse model, validate on held-out set
21. `inverse_synth` tool: wire up inference to MCP
22. `train_model` tool: expose training pipeline to agent
23. `list_models` tool: model inventory

### Phase 4: Expand synthesis types
24. Additional renderers (FM, organ)
25. Train models per synthesis type
26. Research vector → device param mapping (vector DB, name matching, learned mapping)

## Verification

1. **Audio pipeline:** `fetch_audio` → `stem_separate` → verify 4 stems produced
2. **Spectrum:** `spectrum_analyze` on pure sine at 440Hz → fundamental ~440Hz, no harmonics
3. **Transcription:** `note_transcribe` on a known polyphonic audio file → verify MIDI notes match expected pitches and timings
4. **Note isolation:** `note_isolate` on a polyphonic stem → verify isolated notes are monophonic and quality-scored
5. **Monophonic shortcut:** `note_isolate` on a monophonic window → verify time-slice extraction (no masking) produces clean output
6. **Distortion triage:** `note_isolate` with `assess_quality=true` on distorted audio → verify distorted notes flagged as unusable
7. **Dataset gen:** Generate 100 samples for Prophet-6, verify (spectrogram, param_vector) pairs
8. **Training:** Train on 1K samples, verify loss decreases, predictions are reasonable
9. **Round-trip:** Generate random params → render → predict via model → compare predicted vs original params
10. **End-to-end:** Separate a song → `note_transcribe` → `note_isolate` → `inverse_synth` → apply params to keyboard → `audio_render` → `audio_compare` → similarity score

## Test Coverage

> This is a separate Python repo. Tests use `pytest`. Structure mirrors the `src/` layout under `tests/`.

### Unit tests

**`tests/test_spectral.py`** — spectral analysis module:
- **Pure sine:** Generate a 440Hz sine wave in numpy. Run `spectrum_analyze`. Assert fundamental is ~440Hz, no significant harmonics.
- **Harmonics:** Generate a square wave. Assert harmonics at 3x, 5x, 7x fundamental.
- **ADSR detection:** Generate a signal with clear attack/decay/sustain/release envelope. Assert detected ADSR values are within tolerance.
- **Empty/silence input:** Assert graceful handling (no crash, returns zeros or "silence" indicator).

**`tests/test_comparison.py`** — A/B spectral diff:
- **Identical inputs:** Compare a file to itself. Assert similarity score ~1.0, no action items.
- **Known difference:** Compare a sine at 440Hz to a sine at 880Hz. Assert frequency band diff highlights the shift.
- **Different timbres:** Compare a sine to a square wave at same fundamental. Assert spectral envelope diff flags the harmonic content.

**`tests/test_transcription.py`** — polyphonic transcription:
- **Known monophonic input:** Generate a single-note sine wave. Run `note_transcribe`. Assert exactly 1 note event with correct pitch.
- **Known polyphonic input:** Generate two simultaneous sine waves (C4 + E4). Assert 2 note events with correct pitches and overlapping time windows.
- **Polyphony profile:** Assert monophonic windows detected where only 1 note sounds, polyphonic windows where 2+ overlap.
- **Candidate selection:** Assert candidates prefer monophonic windows and notes with duration > 0.5s.

**`tests/test_note_isolation.py`** — score-informed source separation:
- **Monophonic extraction:** Provide a monophonic window. Assert `isolation_method` is `time_slice` (no masking needed).
- **Polyphonic extraction:** Provide a 2-note polyphonic window. Assert `isolation_method` is `nussl_tf_mask`.
- **Quality assessment — clean:** Isolate a clean synthesized note. Assert `quality_score > 0.8` and `usable=true`.
- **Quality assessment — distorted:** Isolate a heavily distorted note. Assert `usable=false` and `detected_effects` includes `heavy_distortion`.
- **Quality assessment — reverb:** Isolate a note with reverb. Assert `usable=true` and `detected_effects` includes `reverb`.

**`tests/test_dataset_generation.py`** — dataset pipeline:
- **Output shape:** Generate 10 samples for subtractive synth. Assert each has (spectrogram, param_vector) with correct dimensions.
- **Param ranges:** Assert all param values in generated vectors are within [0, 1].
- **Augmentation variation:** Generate two augmented variants from same params. Assert spectrograms differ (effects/noise applied).
- **Reproducibility:** Same seed produces same dataset.

**`tests/test_inference.py`** — model inference:
- **Output shape:** Mock a trained model checkpoint. Run inference on a test spectrogram. Assert output vector length matches expected param count.
- **Value range:** Assert all predicted values are in [0, 1] (sigmoid output).
- **top_k:** Request top_k=3. Assert 3 predictions returned, sorted by confidence descending.
- **Unknown synth type:** Request inference for unsupported synth type. Assert clear error.

### Integration tests

**`tests/test_pipeline_integration.py`**:
- **fetch + analyze:** Fetch a local test WAV file, run `spectrum_analyze`. Assert structured output (no crash, expected keys present).
- **fetch + separate:** Fetch a short test file, run stem separation (with `htdemucs`). Assert 4 stem files produced in workspace. (Slow — mark with `@pytest.mark.slow`.)
- **transcribe + isolate:** Transcribe a short polyphonic test file, then isolate candidate notes. Assert isolated WAV files are produced and quality-scored. (Slow — mark with `@pytest.mark.slow`.)
- **dataset + train:** Generate 100 samples, train for 2 epochs. Assert loss decreased between epoch 1 and 2. Assert checkpoint file written. (Slow — mark with `@pytest.mark.slow`.)

### E2E tests

**`tests/test_mcp_tools.py`** — MCP server tool invocations:
- **Tool listing:** Start the MCP server, list tools. Assert `fetch_audio`, `stem_separate`, `spectrum_analyze`, `audio_compare`, `note_transcribe`, `note_isolate`, `inverse_synth` are all present.
- **spectrum_analyze round-trip:** Call the MCP tool with a test WAV path. Assert JSON response contains expected fields (`harmonics`, `spectral_envelope`, `synth_hints`).
- **audio_compare round-trip:** Call with two identical file paths. Assert high similarity score.
- **inverse_synth without model:** Call `inverse_synth` for a synth type with no trained model. Assert user-friendly error message (not a stack trace).

> **Note:** Tests that require GPU (full training runs, large dataset generation) are excluded from CI. Mark with `@pytest.mark.gpu`. CI runs `pytest -m "not slow and not gpu"`.
