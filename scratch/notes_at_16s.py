"""List all notes that are sounding at t=16s in the van-halen-jump MIDI.

Run: uv run python scratch/notes_at_16s.py
"""

import json
from pathlib import Path

NOTES_PATH = Path(
    "~/.audio-analysis-mcp/workspace/jobs/"
    "van-halen-jump-lyrics-bugg-lyrics/transcriptions/other_fast/"
    "transcription_06019111.json"
)
T = 16.0
WINDOW = 0.5  # show notes overlapping [T - WINDOW, T + WINDOW]


def midi_to_name(p: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[p % 12]}{p // 12 - 1}"


def main() -> None:
    notes = json.loads(NOTES_PATH.read_text())

    print(f"Notes whose [start, end] intersects [{T - WINDOW:.2f}, {T + WINDOW:.2f}]s:\n")
    print(f"{'pitch':>5}  {'name':>4}  {'start':>6}  {'end':>6}  {'dur':>5}  {'vel':>4}")
    print("-" * 50)
    matches = []
    for n in sorted(notes, key=lambda x: x["start_time"]):
        if n["start_time"] < T + WINDOW and n["end_time"] > T - WINDOW:
            matches.append(n)
            dur = n["end_time"] - n["start_time"]
            print(f"{n['pitch_midi']:>5}  {midi_to_name(n['pitch_midi']):>4}  "
                  f"{n['start_time']:6.3f}  {n['end_time']:6.3f}  {dur:5.3f}  {n['amplitude']:.2f}")

    # Highlight which ones touch t=16s exactly
    print(f"\nSounding exactly at t={T}s (start_time ≤ {T} ≤ end_time):")
    for n in matches:
        if n["start_time"] <= T <= n["end_time"]:
            print(f"  pitch={midi_to_name(n['pitch_midi'])} ({n['pitch_midi']}) at [{n['start_time']:.3f}, {n['end_time']:.3f}]")


if __name__ == "__main__":
    main()