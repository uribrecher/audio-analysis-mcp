"""Validate the redesigned triage on [0s, 5s] of van-halen-jump.

Two design changes from the current algorithm:

1. **Cluster by time-overlap, not onset proximity.**
   - Build a graph: link two notes if their [start, end] intervals intersect.
   - Connected components = clusters.
   - For each component:
       - 1 note → single
       - 2+ notes that share a common sounding interval (intersection of all
         member intervals is non-empty) → chord
       - 2+ notes with no common interval (overlap pairwise but not all) → arpeggio

2. **Polyphony profile on whole song.**
   - Build the polyphony profile from ALL notes, not just the windowed subset.
   - For clustering, keep only notes whose start_time falls inside the window.
   - This avoids underestimating polyphony at window edges.

Run: uv run python scratch/explore_overlap_clustering.py
"""

import json
import math
from pathlib import Path

NOTES_PATH = Path(
    "/Users/uribrecher/.audio-analysis-mcp/workspace/jobs/"
    "van-halen-jump-lyrics-bugg-lyrics/transcriptions/other_fast/"
    "transcription_06019111.json"
)
WINDOW_START = 0.0
WINDOW_END = 5.0
PROFILE_BUCKET_S = 0.5
KIND_BONUS = {"single": 3.0, "chord": 2.0, "arpeggio": 0.0}


def midi_to_name(p: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[p % 12]}{p // 12 - 1}"


def build_polyphony_profile(all_notes):
    if not all_notes:
        return []
    end = max(n["end_time"] for n in all_notes)
    profile = []
    t = 0.0
    while t < end:
        w_end = t + PROFILE_BUCKET_S
        count = sum(1 for n in all_notes if n["start_time"] < w_end and n["end_time"] > t)
        profile.append((t, w_end, count))
        t = w_end
    return profile


def cluster_by_overlap(notes):
    """Connected components in the interval-overlap graph."""
    n = len(notes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if notes[i]["start_time"] < notes[j]["end_time"] and notes[i]["end_time"] > notes[j]["start_time"]:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def classify(member_indices, notes):
    if len(member_indices) == 1:
        return "single"
    members = [notes[i] for i in member_indices]
    common_start = max(m["start_time"] for m in members)
    common_end = min(m["end_time"] for m in members)
    if common_start < common_end:
        return "chord"
    return "arpeggio"


def cluster_polyphony(start_t, end_t, profile):
    overlapping = [c for s, e, c in profile if s < end_t and e > start_t]
    if not overlapping:
        return 1.0
    return sum(overlapping) / len(overlapping)


def cluster_temporal_gap(start_t, end_t, all_clusters):
    min_gap = float("inf")
    for c in all_clusters:
        if c["start_t"] == start_t and c["end_t"] == end_t:
            continue
        gap = max(0.0, max(c["start_t"] - end_t, start_t - c["end_t"]))
        min_gap = min(min_gap, gap)
    return min_gap if min_gap != float("inf") else 1.0


def main() -> None:
    all_notes = json.loads(NOTES_PATH.read_text())

    # Profile on whole song
    profile = build_polyphony_profile(all_notes)
    print(f"Polyphony profile: {len(profile)} buckets covering whole song")

    # Filter for clustering: start_time in window
    in_window = [n for n in all_notes if WINDOW_START <= n["start_time"] < WINDOW_END]
    in_window.sort(key=lambda n: n["start_time"])
    print(f"Notes whose onset is in [{WINDOW_START}, {WINDOW_END}]s: {len(in_window)}\n")

    # Cluster by overlap
    components = cluster_by_overlap(in_window)
    print(f"Connected components from overlap graph: {len(components)}\n")

    # Classify and pre-compute bounds
    clusters = []
    for comp in components:
        members = [in_window[i] for i in comp]
        start_t = min(m["start_time"] for m in members)
        end_t = max(m["end_time"] for m in members)
        kind = classify(comp, in_window)
        clusters.append({
            "kind": kind, "start_t": start_t, "end_t": end_t,
            "members": members,
        })

    # Score
    for c in clusters:
        dur = c["end_t"] - c["start_t"]
        vel = sum(m["amplitude"] for m in c["members"]) / len(c["members"])
        poly = cluster_polyphony(c["start_t"], c["end_t"], profile)
        gap = cluster_temporal_gap(c["start_t"], c["end_t"], clusters)
        c["score"] = (KIND_BONUS[c["kind"]] + (1.0 / max(poly, 1.0)) * 2.0
                      + math.log1p(min(dur, 2.0)) * 1.0
                      + math.log1p(gap) * 0.5
                      + vel * 1.0)

    # Drop arpeggios, sort
    chosen = [c for c in clusters if c["kind"] != "arpeggio"]
    chosen.sort(key=lambda c: c["score"], reverse=True)

    print(f"--- All clusters in time order (kind, time, members) ---")
    for c in sorted(clusters, key=lambda c: c["start_t"]):
        pitches = [m["pitch_midi"] for m in c["members"]]
        names = [midi_to_name(p) for p in pitches]
        print(f"  {c['kind']:8s}  t=[{c['start_t']:.3f}, {c['end_t']:.3f}]s  dur={c['end_t']-c['start_t']:.2f}s  pitches={names}  score={c['score']:.2f}")

    print()
    print(f"--- Top 10 candidates after dropping arpeggios ---")
    for rank, c in enumerate(chosen[:10]):
        pitches = [m["pitch_midi"] for m in c["members"]]
        names = [midi_to_name(p) for p in pitches]
        vel = sum(m["amplitude"] for m in c["members"]) / len(c["members"])
        print(f"  #{rank+1}  score={c['score']:.2f}  {c['kind']:6s}  t=[{c['start_t']:.3f}, {c['end_t']:.3f}]s  vel={vel:.2f}  pitches={names}")


if __name__ == "__main__":
    main()