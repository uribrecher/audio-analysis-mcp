"""Check what notes overlap in time with the top 'single' candidate at [4.345, 4.554]s.

Run: uv run python scratch/check_single_overlap.py
"""

import json
from pathlib import Path

NOTES_PATH = Path(
    "/Users/uribrecher/.audio-analysis-mcp/workspace/jobs/"
    "van-halen-jump-lyrics-bugg-lyrics/transcriptions/other_fast/"
    "transcription_06019111.json"
)


def midi_to_name(p: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[p % 12]}{p // 12 - 1}"


def main() -> None:
    notes = json.loads(NOTES_PATH.read_text())

    # Top "single" candidate: B4 (pitch 71) at start=4.345, end=4.554
    target_start = 4.345
    target_end = 4.554
    target_pitch = 71

    print(f"Target 'single': B4 (pitch {target_pitch}) at [{target_start:.3f}, {target_end:.3f}]s\n")

    print("Notes that overlap (time-intersect) with the target:")
    print(f"{'pitch':>5}  {'name':>4}  {'start':>6}  {'end':>6}  {'dur':>5}  {'vel':>4}  overlap_with_target")
    print("-" * 75)
    for n in sorted(notes, key=lambda x: x["start_time"]):
        # overlap check: ranges intersect if start < target_end and end > target_start
        if n["start_time"] < target_end and n["end_time"] > target_start:
            same = " ← THE TARGET ITSELF" if (
                abs(n["start_time"] - target_start) < 0.01
                and abs(n["end_time"] - target_end) < 0.01
                and n["pitch_midi"] == target_pitch
            ) else ""
            ov_start = max(n["start_time"], target_start)
            ov_end = min(n["end_time"], target_end)
            ov = ov_end - ov_start
            dur = n["end_time"] - n["start_time"]
            print(f"{n['pitch_midi']:>5}  {midi_to_name(n['pitch_midi']):>4}  {n['start_time']:6.3f}  {n['end_time']:6.3f}  {dur:5.3f}  {n['amplitude']:.2f}  overlap={ov:.3f}s{same}")


if __name__ == "__main__":
    main()