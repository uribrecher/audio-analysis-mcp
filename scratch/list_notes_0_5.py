"""List every transcribed note in the [0s, 5s] window for visual verification.

Run: uv run python scratch/list_notes_0_5.py
"""

import json
from pathlib import Path

NOTES_PATH = Path(
    "/Users/uribrecher/.audio-analysis-mcp/workspace/jobs/"
    "van-halen-jump-lyrics-bugg-lyrics/transcriptions/other_fast/"
    "transcription_06019111.json"
)
WINDOW_START = 0.0
WINDOW_END = 5.0


def midi_to_name(p: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[p % 12]}{p // 12 - 1}"


def main() -> None:
    notes = json.loads(NOTES_PATH.read_text())
    in_window = [n for n in notes if WINDOW_START < n["end_time"] and n["start_time"] < WINDOW_END]
    in_window.sort(key=lambda n: (n["start_time"], n["pitch_midi"]))

    print(f"Window [{WINDOW_START}s, {WINDOW_END}s] — {len(in_window)} notes total\n")
    print(f"{'#':>3}  {'start':>6}  {'end':>6}  {'dur':>5}  {'pitch':>5}  {'name':>4}  {'vel':>4}")
    print("-" * 50)
    for i, n in enumerate(in_window):
        dur = n["end_time"] - n["start_time"]
        print(f"{i:>3}  {n['start_time']:6.3f}  {n['end_time']:6.3f}  {dur:5.3f}  {n['pitch_midi']:>5}  {midi_to_name(n['pitch_midi']):>4}  {n['amplitude']:.2f}")


if __name__ == "__main__":
    main()