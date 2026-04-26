"""Extract each amplitude-analyzed cluster as a standalone WAV with 0.3s
margins on either side so you can audibly verify what the analyzer saw.

Run: uv run python scratch/extract_cluster_audio.py
"""

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_analysis_mcp.schemas import NoteTriageFileData

JOB = Path(
    "/Users/uribrecher/.audio-analysis-mcp/workspace/jobs/"
    "van-halen-jump-lyrics-bugg-lyrics"
)
TRIAGE = JOB / "triage/other_fast/triage.json"
STEM = JOB / "stems/fast/other.wav"
OUT_DIR = Path("/tmp/van-halen-clusters")
MARGIN_S = 0.3
TARGET_INDICES = [0, 1, 3]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    audio, sr = sf.read(STEM, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype("float32")

    file_data = NoteTriageFileData.model_validate_json(TRIAGE.read_text())

    for idx in TARGET_INDICES:
        c = file_data.candidates[idx]
        start_t = max(0.0, c.start_time - MARGIN_S)
        end_t = min(audio.size / sr, c.end_time + MARGIN_S)
        start_sample = int(start_t * sr)
        end_sample = int(end_t * sr)
        clip = audio[start_sample:end_sample]
        out_path = OUT_DIR / f"cluster_{idx:02d}_{c.kind}_{c.start_time:.2f}s.wav"
        sf.write(out_path, clip, sr)
        print(
            f"cluster {idx:>2} {c.kind:6s}  "
            f"core=[{c.start_time:.3f}, {c.end_time:.3f}]s "
            f"clip=[{start_t:.3f}, {end_t:.3f}]s ({clip.size / sr:.2f}s)  "
            f"→ {out_path}"
        )


if __name__ == "__main__":
    main()