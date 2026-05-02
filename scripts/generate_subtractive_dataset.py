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
import os
import subprocess
import time
from pathlib import Path

import soundfile as sf

from audio_analysis_mcp.research.tone_generation.dataset import sample_dataset_config
from audio_analysis_mcp.research.tone_generation.renderer import render_chord


SAMPLE_RATE = 44_100
TOTAL_DURATION_S = 1.2


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
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
    n_produced = 0
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
            labels_f.flush()
            n_produced = idx + 1
            if (idx + 1) % 500 == 0:
                rate = (idx + 1) / (time.time() - t0)
                print(f"  rendered {idx + 1}/{args.n_samples} ({rate:.1f} samples/s)")

    manifest = {
        "n_samples": args.n_samples,
        "n_samples_produced": n_produced,
        "seed": args.seed,
        "schema_version": "0.1",
        "sample_rate": SAMPLE_RATE,
        "total_duration_s": TOTAL_DURATION_S,
        "renderer_git_sha": _git_sha(),
    }
    manifest_path_partial = manifest_path.with_suffix(".json.partial")
    manifest_path_partial.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(manifest_path_partial, manifest_path)
    print(
        f"Done. {args.n_samples} samples → {out_dir} in {time.time() - t0:.1f}s."
    )


if __name__ == "__main__":
    main()
