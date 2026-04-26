import librosa
import numpy as np

from audio_analysis_mcp.schemas import (
    NoteEvent,
    PolyphonyWindow,
    CandidateNote,
    CandidateCluster,
    NoteTriageFileData,
)

WINDOW_SIZE = 0.5  # seconds
TIME_PADDING = 0.05  # 50ms before/after note
MAX_FREQ_HZ = 10000.0
NUM_HARMONICS = 8


def _build_polyphony_profile(
    notes: list[NoteEvent],
) -> list[PolyphonyWindow]:
    """Divide timeline into fixed windows and count simultaneous notes per window."""
    if not notes:
        return []
    end_time = max(n.end_time for n in notes)
    windows: list[PolyphonyWindow] = []
    t = 0.0
    while t < end_time:
        w_end = t + WINDOW_SIZE
        count = sum(
            1 for n in notes if n.start_time < w_end and n.end_time > t
        )
        windows.append(PolyphonyWindow(start_time=t, end_time=w_end, note_count=count))
        t = w_end
    return windows


def _polyphony_at(note: NoteEvent, profile: list[PolyphonyWindow]) -> float:
    """Average polyphony count across windows that overlap with this note."""
    overlapping = [
        w for w in profile if w.start_time < note.end_time and w.end_time > note.start_time
    ]
    if not overlapping:
        return 1.0
    return sum(w.note_count for w in overlapping) / len(overlapping)


def _temporal_gap(note: NoteEvent, all_notes: list[NoteEvent]) -> float:
    """Minimum time gap to the nearest neighboring note (seconds)."""
    min_gap = float("inf")
    for other in all_notes:
        if other is note:
            continue
        gap = max(0.0, max(other.start_time - note.end_time, note.start_time - other.end_time))
        min_gap = min(min_gap, gap)
    return min_gap if min_gap != float("inf") else 1.0


def _freq_bounds(pitch_midi: int) -> tuple[float, float]:
    """Compute frequency isolation bounds from MIDI pitch."""
    fundamental: float = float(librosa.midi_to_hz(pitch_midi))
    lower = fundamental * 0.9
    upper = min(fundamental * NUM_HARMONICS, MAX_FREQ_HZ)
    return lower, upper


def _score_note(
    note: NoteEvent,
    profile: list[PolyphonyWindow],
    all_notes: list[NoteEvent],
) -> float:
    """Score a note for isolation suitability. Higher = better."""
    duration = note.end_time - note.start_time
    poly = _polyphony_at(note, profile)
    gap = _temporal_gap(note, all_notes)

    # Low polyphony is best (invert: 1/poly)
    poly_score = 1.0 / max(poly, 1.0)
    # Duration: log scale, capped at diminishing returns past 2s
    dur_score = float(np.log1p(min(duration, 2.0)))
    # Temporal gap: more gap is better, log scale
    gap_score = float(np.log1p(gap))

    return poly_score * 2.0 + dur_score + gap_score * 0.5


_CHORD_TOLERANCE_S = 0.030
_ARPEGGIO_GAP_S = 0.150
_ARPEGGIO_MIN_SIZE = 3

_KIND_BONUS = {"single": 3.0, "chord": 2.0, "arpeggio": 0.0}


def _cluster_polyphony(cluster: CandidateCluster, profile: list[PolyphonyWindow]) -> float:
    """Average polyphony count across windows that overlap with the cluster's time range."""
    overlapping = [
        w for w in profile if w.start_time < cluster.end_time and w.end_time > cluster.start_time
    ]
    if not overlapping:
        return 1.0
    return sum(w.note_count for w in overlapping) / len(overlapping)


def _cluster_temporal_gap(cluster: CandidateCluster, all_clusters: list[CandidateCluster]) -> float:
    """Minimum time gap to the nearest neighboring cluster (seconds)."""
    min_gap = float("inf")
    for other in all_clusters:
        if other is cluster:
            continue
        gap = max(0.0, max(other.start_time - cluster.end_time, cluster.start_time - other.end_time))
        min_gap = min(min_gap, gap)
    return min_gap if min_gap != float("inf") else 1.0


def _score_cluster(
    cluster: CandidateCluster,
    profile: list[PolyphonyWindow],
    all_clusters: list[CandidateCluster],
) -> float:
    """Score a cluster for ADSR-analysis suitability. Higher = better."""
    duration = cluster.end_time - cluster.start_time
    poly = _cluster_polyphony(cluster, profile)
    gap = _cluster_temporal_gap(cluster, all_clusters)
    velocity = sum(m.note.amplitude for m in cluster.members) / max(len(cluster.members), 1)

    poly_score = 1.0 / max(poly, 1.0)
    dur_score = float(np.log1p(min(duration, 2.0)))
    gap_score = float(np.log1p(gap))
    kind_bonus = _KIND_BONUS[cluster.kind]

    return kind_bonus + poly_score * 2.0 + dur_score + gap_score * 0.5 + velocity * 1.0


def _build_candidate_note(note: NoteEvent) -> CandidateNote:
    start_freq, end_freq = _freq_bounds(note.pitch_midi)
    padded_start = max(0.0, note.start_time - TIME_PADDING)
    padded_end = note.end_time + TIME_PADDING
    return CandidateNote(
        note=note,
        score=0.0,                       # filled in by Task 4
        start_time=padded_start,
        end_time=padded_end,
        start_freq=round(start_freq, 2),
        end_freq=round(end_freq, 2),
    )


def _cluster_notes(notes: list[NoteEvent]) -> list[CandidateCluster]:
    """Pass 1: group notes into single / chord / arpeggio clusters."""
    if not notes:
        return []

    notes_sorted = sorted(notes, key=lambda n: n.start_time)

    # Step 1: collect chord groups (greedy left-to-right).
    used: set[int] = set()
    chord_groups: list[list[int]] = []
    for i, n in enumerate(notes_sorted):
        if i in used:
            continue
        group = [i]
        for j in range(i + 1, len(notes_sorted)):
            if j in used:
                continue
            other = notes_sorted[j]
            if (abs(other.start_time - n.start_time) <= _CHORD_TOLERANCE_S
                    and abs(other.end_time - n.end_time) <= _CHORD_TOLERANCE_S):
                group.append(j)
        if len(group) >= 2:
            for k in group:
                used.add(k)
            chord_groups.append(group)

    # Step 2: from the remaining notes, find arpeggio runs.
    remaining_indices = [i for i in range(len(notes_sorted)) if i not in used]
    arpeggio_groups: list[list[int]] = []
    run: list[int] = []
    for idx in remaining_indices:
        if not run:
            run = [idx]
            continue
        prev_idx = run[-1]
        gap = notes_sorted[idx].start_time - notes_sorted[prev_idx].start_time
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

    # Step 3: emit clusters in start-time order.
    cluster_specs: list[tuple[str, list[int]]] = []
    cluster_specs.extend(("chord", g) for g in chord_groups)
    cluster_specs.extend(("arpeggio", g) for g in arpeggio_groups)
    for i in range(len(notes_sorted)):
        if i not in used:
            cluster_specs.append(("single", [i]))

    cluster_specs.sort(key=lambda spec: notes_sorted[spec[1][0]].start_time)

    clusters: list[CandidateCluster] = []
    for kind, indices in cluster_specs:
        members = [_build_candidate_note(notes_sorted[i]) for i in indices]
        start_time = min(m.start_time for m in members)
        end_time = max(m.end_time for m in members)
        start_freq = min(m.start_freq for m in members)
        end_freq = max(m.end_freq for m in members)
        clusters.append(CandidateCluster(
            kind=kind,                              # type: ignore[arg-type]
            score=0.0,
            start_time=start_time,
            end_time=end_time,
            start_freq=start_freq,
            end_freq=end_freq,
            members=members,
        ))

    return clusters


def triage_notes(
    notes: list[NoteEvent],
    min_duration: float = 0.5,
    max_candidates: int = 10,
) -> NoteTriageFileData:
    """Profile polyphony and rank candidate notes for isolation."""
    profile = _build_polyphony_profile(notes)

    if not notes:
        return NoteTriageFileData(polyphony_profile=[], candidates=[])

    # Filter by duration
    eligible = [n for n in notes if (n.end_time - n.start_time) >= min_duration]

    # Score and rank
    scored = [(n, _score_note(n, profile, notes)) for n in eligible]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Pitch diversity: greedily select from scored list, penalizing nearby pitches
    selected: list[tuple[NoteEvent, float]] = []
    selected_pitches: list[int] = []
    for note, score in scored:
        if len(selected) >= max_candidates:
            break
        # Penalize if a very similar pitch already selected (within 2 semitones)
        if any(abs(note.pitch_midi - p) <= 2 for p in selected_pitches):
            score *= 0.5
        selected.append((note, score))
        selected_pitches.append(note.pitch_midi)

    # Re-sort after diversity adjustment
    selected.sort(key=lambda x: x[1], reverse=True)
    selected = selected[:max_candidates]

    # Build candidates with time/freq bounds
    candidates: list[CandidateNote] = []
    for note, score in selected:
        start_freq, end_freq = _freq_bounds(note.pitch_midi)
        padded_start = max(0.0, note.start_time - TIME_PADDING)
        padded_end = note.end_time + TIME_PADDING
        candidates.append(
            CandidateNote(
                note=note,
                score=round(score, 4),
                start_time=padded_start,
                end_time=padded_end,
                start_freq=round(start_freq, 2),
                end_freq=round(end_freq, 2),
            )
        )

    return NoteTriageFileData(polyphony_profile=profile, candidates=candidates)
