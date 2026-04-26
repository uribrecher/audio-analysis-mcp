"""Validate the 3-pass clustering algorithm before locking in the implementation.

Synthesizes several note-event sequences (single, chord, arpeggio, mixed)
and walks the chord-first / arpeggio-second greedy pass to confirm the
groupings match the plan's intended semantics. Verifies edge cases that
the plan's unit tests will assert against.

Run: uv run python scratch/explore_clustering.py
"""

from dataclasses import dataclass

_CHORD_TOLERANCE_S = 0.030
_ARPEGGIO_GAP_S = 0.150
_ARPEGGIO_MIN_SIZE = 3


@dataclass
class Note:
    start: float
    end: float
    pitch: int

    def __repr__(self) -> str:
        return f"({self.start:.2f}-{self.end:.2f},{self.pitch})"


def cluster_notes(notes: list[Note]) -> list[tuple[str, list[int]]]:
    """Returns list of (kind, indices_into_sorted_notes)."""
    if not notes:
        return []
    notes_sorted = sorted(range(len(notes)), key=lambda i: notes[i].start)
    sn = [notes[i] for i in notes_sorted]

    used: set[int] = set()
    chord_groups: list[list[int]] = []
    for i, n in enumerate(sn):
        if i in used:
            continue
        group = [i]
        for j in range(i + 1, len(sn)):
            if j in used:
                continue
            o = sn[j]
            if (abs(o.start - n.start) <= _CHORD_TOLERANCE_S
                    and abs(o.end - n.end) <= _CHORD_TOLERANCE_S):
                group.append(j)
        if len(group) >= 2:
            for k in group:
                used.add(k)
            chord_groups.append(group)

    remaining = [i for i in range(len(sn)) if i not in used]
    arpeggio_groups: list[list[int]] = []
    run: list[int] = []
    for idx in remaining:
        if not run:
            run = [idx]
            continue
        prev = run[-1]
        gap = sn[idx].start - sn[prev].start
        if 0.0 <= gap <= _ARPEGGIO_GAP_S:
            run.append(idx)
        else:
            if len(run) >= _ARPEGGIO_MIN_SIZE:
                arpeggio_groups.append(run)
                for k in run:
                    used.add(k)
            run = [idx]
    if len(run) >= _ARPEGGIO_MIN_SIZE:
        arpeggio_groups.append(run)
        for k in run:
            used.add(k)

    specs: list[tuple[str, list[int]]] = []
    specs.extend(("chord", g) for g in chord_groups)
    specs.extend(("arpeggio", g) for g in arpeggio_groups)
    for i in range(len(sn)):
        if i not in used:
            specs.append(("single", [i]))
    specs.sort(key=lambda spec: sn[spec[1][0]].start)
    return specs


def show(label: str, notes: list[Note]) -> None:
    print(f"\n--- {label} ---")
    print(f"input: {notes}")
    clusters = cluster_notes(notes)
    print(f"clusters: {len(clusters)}")
    sorted_idx = sorted(range(len(notes)), key=lambda i: notes[i].start)
    for kind, indices in clusters:
        members = [notes[sorted_idx[i]] for i in indices]
        print(f"  {kind}: {members}")


def main() -> None:
    show("A: single mono note", [Note(0.0, 1.0, 60)])

    show("B: simple chord (3 simultaneous)", [
        Note(0.0, 1.0, 60), Note(0.005, 1.0, 64), Note(0.02, 0.99, 67),
    ])

    show("C: arpeggio of 6 notes 100ms apart", [
        Note(i * 0.1, i * 0.1 + 0.4, 60 + i) for i in range(6)
    ])

    show("D: 2 sequential notes — too short for arpeggio (≥3 required)", [
        Note(0.0, 0.4, 60), Note(0.45, 0.85, 62),
    ])

    show("E: chord + sequential single (mixed)", [
        Note(0.0, 1.0, 60), Note(0.0, 1.0, 64), Note(2.0, 3.0, 72),
    ])

    show("F: arpeggio breaks on long gap (4 notes, 400ms gap in middle)", [
        Note(0.0, 0.1, 60), Note(0.1, 0.2, 62),
        Note(0.5, 0.6, 64), Note(0.6, 0.7, 65),
    ])

    show("G: chord-first wins — overlapping chord at start of arpeggio", [
        Note(0.0, 1.0, 60), Note(0.01, 1.0, 64),  # chord
        Note(0.5, 0.7, 67), Note(0.6, 0.8, 69), Note(0.7, 0.9, 71),  # arpeggio attempt
    ])
    # Expected: chord [60,64], then 3 arpeggio notes (gap 0.1, 0.1) → arpeggio.

    show("H: chord rejects on end-time mismatch", [
        Note(0.0, 1.0, 60), Note(0.01, 1.5, 64),  # different end times
    ])
    # Expected: 2 singles (chord rule requires ends within 30ms).

    show("I: 30ms boundary — exactly at threshold should still form chord", [
        Note(0.0, 1.0, 60), Note(0.030, 1.030, 64),  # both starts and ends offset by exactly 30ms
    ])
    # Expected: chord (≤ tolerance is inclusive).

    show("J: 31ms — just over threshold, should be 2 singles", [
        Note(0.0, 1.0, 60), Note(0.031, 1.0, 64),
    ])
    # Expected: 2 singles.

    show("K: arpeggio of exactly 3 (the minimum)", [
        Note(0.0, 0.4, 60), Note(0.1, 0.5, 62), Note(0.2, 0.6, 64),
    ])
    # Expected: 1 arpeggio with 3 members.


if __name__ == "__main__":
    main()