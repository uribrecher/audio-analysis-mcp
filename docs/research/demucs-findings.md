# Demucs Stem Separation — Research Findings

## Models tested

| Model | Stems | Sub-models | Notes |
|-------|-------|------------|-------|
| `htdemucs` | 4 (drums, bass, other, vocals) | 1 | Default. Significant bass bleed into "other" stem |
| `htdemucs_ft` | 4 | 4 (bag) | Fine-tuned. 4x slower per shift. Similar bleed issues to htdemucs |
| `htdemucs_6s` | 6 (+ guitar, piano) | 1 | **Best for keyboard extraction.** Dedicated piano/guitar stems. Much less bass bleed in "other" |
| `hdemucs_mmi` | 4 | 1 | Not tested yet |
| `mdx` | 4 | 4 (bag) | Older architecture, not tested |
| `mdx_extra` | 4 | 4 (bag) | mdx + extra training data, not tested |

## Key findings

### htdemucs_6s is the best model for our use case
- Tested on "Smooth Criminal" (Michael Jackson) — a song with heavy bass line + synth brass + vocals
- The 4-stem models (`htdemucs`, `htdemucs_ft`) dump everything non-vocal/drum/bass into "other", causing severe bass bleed
- `htdemucs_6s` separates guitar and piano into dedicated stems, leaving "other" much cleaner
- The piano stem was near-silent for Smooth Criminal (correct — no piano in this song)
- Guitar stem captures some content but is "a bit weird" — acceptable
- Single sub-model, so fast to run (1 run per shift)

### shifts parameter has minimal audible impact
- Compared `htdemucs` with shifts=1 (fast) vs shifts=5 (medium)
- Numerically different (correlation 0.94), but **no meaningful audible improvement**
- shifts averages out random artifacts at segment boundaries, not fundamental bleed
- The model choice matters far more than shifts/overlap tuning

### GPU acceleration (MPS) gives ~5x speedup on Apple Silicon
- CPU: ~1.04 seconds/s per run
- MPS: ~5.0 seconds/s per run
- `apply_model` keeps full tracks on CPU, moves only active segments to GPU — memory-efficient
- Auto-detect: MPS > CUDA > CPU

### htdemucs_ft (fine-tuned) is slow with marginal benefit
- Bag of 4 sub-models, each running independently
- With shifts=10: 4 models x 10 shifts = 40 progress bar runs (~40 min on CPU)
- For our keyboard extraction use case, `htdemucs_6s` with fewer shifts is both faster and better

## Current preset configuration

All presets use `htdemucs_6s`:

| Preset | Shifts | Overlap | Typical time (MPS, 4-min song) |
|--------|--------|---------|-------------------------------|
| fast | 1 | 0.25 | ~1 min |
| medium | 5 | 0.5 | ~5 min |
| accurate | 10 | 0.75 | ~10 min |

## MCP vs CLI

- MCP tool defaults to `fast` — suitable for quick exploration
- CLI (`uv run python -m audio_analysis_mcp.cli.stem_separate`) for medium/accurate — shows live tqdm progress bar
- Results are cached by `{file_hash}/{model}_{preset}` — re-runs are instant

## Open questions

- How does `hdemucs_mmi` compare to `htdemucs_6s`?
- Would a two-stage approach work? (htdemucs_6s for separation, then spectrum analysis on piano/other stems)
- Is there value in combining stems post-separation (e.g. piano + other for full keyboard content)?
