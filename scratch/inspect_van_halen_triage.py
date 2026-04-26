"""Inspect van-halen-jump triage and simulate the proposed fixes.

Two bugs found:
  A) Chord detection requires both start AND end within 30ms — but transcribed
     chord members have end-times drifting by 100-500ms. Fix: drop end rule.
  B) min_duration filter strips chord members before clustering. Fix: drop
     the filter entirely; let cluster-level duration scoring sort it out.

This script simulates the new triage on the [10s, 30s] window and shows the
predicted top-10 candidates so we can validate before touching the real code.

Run: uv run python scratch/inspect_van_halen_triage.py
"""

import json
from pathlib import Path

NOTES_PATH = Path(
    "~/.audio-analysis-mcp/workspace/jobs/"
    "van-halen-jump-lyrics-bugg-lyrics/transcriptions/other_fast/"
    "transcription_06019111.json"
)


WINDOW_START = 10.0
WINDOW_END = 15.0


def main() -> None:
    notes = json.loads(NOTES_PATH.read_text())
    in_window_all = [n for n in notes if WINDOW_START < n["end_time"] and n["start_time"] < WINDOW_END]
    in_window = [n for n in in_window_all if (n["end_time"] - n["start_time"]) >= 0.5]
    in_window_all.sort(key=lambda n: n["start_time"])
    in_window.sort(key=lambda n: n["start_time"])
    print(f"Window: [{WINDOW_START}, {WINDOW_END}]s")

    print(f"Notes in [10s, 30s] window:")
    print(f"  total transcribed: {len(in_window_all)}")
    print(f"  surviving min_duration=0.5: {len(in_window)} ← what triage sees")
    print(f"  filtered out (duration < 500ms): {len(in_window_all) - len(in_window)}")

    # Spotlight: notes the triage classified as 'single' — show their short-duration companions.
    targets = [(28.30, 60), (9.54, 65), (22.70, 81), (25.30, 76), (12.84, 60)]
    print("\n--- Each top 'single' — were there simultaneous notes filtered out? ---")
    for t, p in targets:
        cohort = [n for n in in_window_all if abs(n["start_time"] - t) < 0.030]
        survived = [n for n in in_window if abs(n["start_time"] - t) < 0.030]
        print(f"  t≈{t}s pitch={p}:  {len(cohort)} notes onset-aligned (transcribed),  {len(survived)} survived min_duration")
        for n in cohort:
            dur = n["end_time"] - n["start_time"]
            kept = "KEPT" if dur >= 0.5 else "DROPPED"
            print(f"    pitch={n['pitch_midi']:3d}  start={n['start_time']:.3f}  dur={dur:.3f}s  {kept}")

    # Group by onset proximity (start times within 30 ms) — current chord rule.
    print("\n--- Onset clusters (start within 30ms) — what triage's chord rule sees ---")
    groups: list[list[dict]] = []
    for n in in_window:
        if not groups or n["start_time"] - groups[-1][-1]["start_time"] > 0.030:
            groups.append([n])
        else:
            groups[-1].append(n)

    for i, g in enumerate(groups):
        starts = [round(x["start_time"], 3) for x in g]
        ends = [round(x["end_time"], 3) for x in g]
        pitches = [x["pitch_midi"] for x in g]
        start_spread = max(starts) - min(starts)
        end_spread = max(ends) - min(ends)
        chord_under_current_rule = len(g) >= 2 and start_spread <= 0.030 and end_spread <= 0.030
        chord_under_loose_rule = len(g) >= 2 and start_spread <= 0.030
        flag = "CHORD✓" if chord_under_current_rule else ("would-chord-if-no-end-rule" if chord_under_loose_rule else "single/seq")
        print(f"  group {i}: n={len(g)}  starts={starts}  end_spread={end_spread:.3f}s  pitches={pitches}  → {flag}")

    print()
    simulate_new_triage(in_window_all)


def simulate_new_triage(notes: list[dict]) -> None:
    """Apply the proposed fixes (drop end-rule, drop min_duration filter) and show top 10."""
    import math

    _CHORD_TOLERANCE_S = 0.030
    _ARPEGGIO_GAP_S = 0.150
    _ARPEGGIO_MIN_SIZE = 3
    _KIND_BONUS = {"single": 3.0, "chord": 2.0, "arpeggio": 0.0}

    notes = sorted(notes, key=lambda n: n["start_time"])

    # Pass 1: cluster (NEW: only start-time check, no end-time gate)
    used: set[int] = set()
    chord_groups: list[list[int]] = []
    for i, n in enumerate(notes):
        if i in used:
            continue
        group = [i]
        for j in range(i + 1, len(notes)):
            if j in used:
                continue
            other = notes[j]
            if abs(other["start_time"] - n["start_time"]) <= _CHORD_TOLERANCE_S:
                group.append(j)
        if len(group) >= 2:
            for k in group:
                used.add(k)
            chord_groups.append(group)

    remaining = [i for i in range(len(notes)) if i not in used]
    arpeggio_groups: list[list[int]] = []
    run: list[int] = []
    for idx in remaining:
        if not run:
            run = [idx]
            continue
        if 0.0 <= notes[idx]["start_time"] - notes[run[-1]]["start_time"] <= _ARPEGGIO_GAP_S:
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
    for i in range(len(notes)):
        if i not in used:
            specs.append(("single", [i]))

    # Pass 2: score each cluster
    def score_cluster(kind: str, members: list[dict], all_specs: list[tuple[str, list[int]]]) -> float:
        start_t = min(m["start_time"] for m in members)
        end_t = max(m["end_time"] for m in members)
        dur = end_t - start_t
        vel = sum(m["amplitude"] for m in members) / len(members)
        # gap to nearest cluster (rough)
        min_gap = float("inf")
        for k2, idxs in all_specs:
            other_members = [notes[i] for i in idxs]
            if other_members is members:
                continue
            os = min(m["start_time"] for m in other_members)
            oe = max(m["end_time"] for m in other_members)
            if os == start_t and oe == end_t:
                continue
            gap = max(0.0, max(os - end_t, start_t - oe))
            min_gap = min(min_gap, gap)
        gap = min_gap if min_gap != float("inf") else 1.0
        # poly = constant 1 here (simplified)
        return _KIND_BONUS[kind] + 1.0 * 2.0 + math.log1p(min(dur, 2.0)) * 1.0 + math.log1p(gap) * 0.5 + vel * 1.0

    scored = []
    for kind, idxs in specs:
        members = [notes[i] for i in idxs]
        s = score_cluster(kind, members, specs)
        scored.append((kind, idxs, s, members))

    # Pass 3: drop arpeggios, sort, take top 10
    scored = [t for t in scored if t[0] != "arpeggio"]
    scored.sort(key=lambda x: x[2], reverse=True)

    print(f"--- ALL clusters with fixes (drop end-rule, drop min_duration), in start-time order ---")
    print(f"  totals: chords={len(chord_groups)}  arpeggios={len(arpeggio_groups)}  remaining-singles={len(specs) - len(chord_groups) - len(arpeggio_groups)}")
    print()
    # Sort by start_time so we see the song in order, with kind labels.
    by_start = sorted(specs, key=lambda spec: notes[spec[1][0]]["start_time"])
    for kind, idxs in by_start:
        members = [notes[i] for i in idxs]
        start_t = min(m["start_time"] for m in members)
        end_t = max(m["end_time"] for m in members)
        dur = end_t - start_t
        pitches = [m["pitch_midi"] for m in members]
        vel = sum(m["amplitude"] for m in members) / len(members)
        durs = [round(m["end_time"] - m["start_time"], 2) for m in members]
        print(f"  {kind:8s}  t=[{start_t:.3f}, {end_t:.3f}]s  cluster_dur={dur:.2f}s  vel={vel:.2f}  pitches={pitches}  member_durs={durs}")


if __name__ == "__main__":
    main()