# Subtractive tone-generation training pipeline (MVP)

Synthetic dataset generation, supervised CNN training, and re-eval for the
subtractive synth engine. Predicts three per-sample parameters from audio:

- `osc.1.shape` (5-class softmax: saw / square / triangle / sine / pulse)
- `filter.lp.cutoff_hz` (regression in log-Hz, denormalized to canonical Hz)
- `filter.lp.resonance` (regression in [0, 1])

All outputs are normalized to the canonical `subtractive.schema.json` instance.

## References

- **Spec:** [`../../../../../reverse-synth-research/docs/superpowers/specs/2026-05-02-subtractive-tone-training-mvp.md`](../../../../../reverse-synth-research/docs/superpowers/specs/2026-05-02-subtractive-tone-training-mvp.md)
- **Backlog (out-of-scope items):** [`../../../../../reverse-synth-research/docs/superpowers/specs/2026-05-02-subtractive-tone-training-backlog.md`](../../../../../reverse-synth-research/docs/superpowers/specs/2026-05-02-subtractive-tone-training-backlog.md)
- **Canonical schema:** [`../../../../../reverse-synth-research/parameter-mapping/subtractive.schema.json`](../../../../../reverse-synth-research/parameter-mapping/subtractive.schema.json)

## End-to-end workflow

All commands run from the `audio-analysis-mcp/` repo root.

### 1. Generate dataset

```bash
uv run python scripts/generate_subtractive_dataset.py \
    --n-samples 10000 --seed 0 --out-dir scratch/tone_gen_dataset
```

Writes:

- `scratch/tone_gen_dataset/samples/{idx:06d}.wav` — 16-bit PCM, 44.1 kHz mono, 1.2 s.
- `scratch/tone_gen_dataset/labels.jsonl` — one JSON object per sample with
  `params_canonical`, `midi_pitches`, `n_voices`, `audio_path`.
- `scratch/tone_gen_dataset/manifest.json` — dataset-level metadata
  (n_samples, seed, sample_rate, duration, git SHA, timestamps).

### 2. Train

```bash
uv run python scripts/train_tone_generation.py \
    --dataset-dir scratch/tone_gen_dataset \
    --checkpoint-out scratch/tone_gen_dataset/checkpoint.pt \
    --seed 0
```

Optional flags: `--epochs` (default 50), `--batch-size` (default 64),
`--patience` (default 5, early stopping).

Writes `checkpoint.pt` next to the dataset (or wherever `--checkpoint-out`
points). Training logs per-epoch loss + per-head metrics to stdout.

### 3. Eval

```bash
uv run python scripts/eval_tone_generation.py \
    --checkpoint scratch/tone_gen_dataset/checkpoint.pt \
    --dataset-dir scratch/tone_gen_dataset
```

Writes `eval_report.json` next to the checkpoint by default (override with
`--out`). The report contains:

- Per-parameter metrics (shape accuracy + confusion, cutoff MAE in log-Hz +
  Hz, resonance MAE).
- Mel-cosine similarity between input audio and re-rendered audio (uses the
  same SignalFlow renderer + same pitches + same `n_voices`).
- Schema-validation pass count over the test set.

## Outputs at a glance

| Artifact | Producer | Consumer |
|----------|----------|----------|
| `samples/*.wav` + `labels.jsonl` + `manifest.json` | `generate_subtractive_dataset.py` | `train_tone_generation.py`, `eval_tone_generation.py` |
| `checkpoint.pt` | `train_tone_generation.py` | `eval_tone_generation.py` |
| `eval_report.json` | `eval_tone_generation.py` | manual review |

## Module layout

| File | Responsibility |
|------|----------------|
| `schema_io.py` | Canonical schema loading + caching, `BASELINE_AMP_ADSR` preset, `build_canonical_instance`, `normalize_params` / `denormalize_predictions`, `validate_canonical`. |
| `renderer.py` | `render_chord(params_canonical, midi_pitches, sample_rate, total_duration_s)` — SignalFlow polyphonic graph (osc -> LP -> fixed amp ADSR, summed across voices, peak-normalized). |
| `dataset.py` | `sample_dataset_config` (Sobol over `(cutoff_hz_log, resonance)` + uniform RNG over shape / `n_voices` / pitches) and `ToneGenerationDataset` (mel-spec cache + init-time corruption check, returns `(mel_spec, pitch_multihot, target_dict)`). |
| `model.py` | `ToneGenerationCNN` — ~707K-param 3-block CNN with `AdaptiveAvgPool2d` over time, concat with 88-dim MIDI multihot, MLP heads for shape / cutoff / resonance. |

## Out of scope

Items deliberately deferred from the MVP — engine classifier head, modulation
invariance training, additional `osc` / `master` / filter parameters,
real-keyboard recordings, transcription-noise augmentation, hyperparameter
sweeps, etc. — live in
[`../../../../../reverse-synth-research/docs/superpowers/specs/2026-05-02-subtractive-tone-training-backlog.md`](../../../../../reverse-synth-research/docs/superpowers/specs/2026-05-02-subtractive-tone-training-backlog.md).
