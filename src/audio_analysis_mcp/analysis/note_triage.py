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


def _freq_bounds(pitch_midi: int) -> tuple[float, float]:
    """Compute frequency isolation bounds from MIDI pitch."""
    fundamental: float = float(librosa.midi_to_hz(pitch_midi))
    lower = fundamental * 0.9
    upper = min(fundamental * NUM_HARMONICS, MAX_FREQ_HZ)
    return lower, upper


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


def _intervals_overlap(a: NoteEvent, b: NoteEvent) -> bool:
    """True iff a and b are sounding simultaneously at some moment."""
    return a.start_time < b.end_time and a.end_time > b.start_time


def _connected_components(notes: list[NoteEvent]) -> list[list[int]]:
    """Group note indices into connected components of the time-overlap graph."""
    n = len(notes)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if _intervals_overlap(notes[i], notes[j]):
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _classify(members: list[NoteEvent]) -> str:
    """Decide single / chord / arpeggio for a connected component.

    - 1 note → single.
    - 2+ notes that share a non-empty common sounding interval → chord.
    - 2+ notes connected by pairwise overlap but with no all-members
      overlap → arpeggio.
    """
    if len(members) == 1:
        return "single"
    common_start = max(m.start_time for m in members)
    common_end = min(m.end_time for m in members)
    return "chord" if common_start < common_end else "arpeggio"


def _cluster_notes(notes: list[NoteEvent]) -> list[CandidateCluster]:
    """Pass 1: group notes into single / chord / arpeggio clusters by time-overlap."""
    if not notes:
        return []

    notes_sorted = sorted(notes, key=lambda n: n.start_time)
    components = _connected_components(notes_sorted)

    cluster_specs: list[tuple[str, list[int]]] = []
    for comp in components:
        kind = _classify([notes_sorted[i] for i in comp])
        cluster_specs.append((kind, comp))

    cluster_specs.sort(key=lambda spec: notes_sorted[spec[1][0]].start_time)

    clusters: list[CandidateCluster] = []
    for kind, indices in cluster_specs:
        candidate_members = [_build_candidate_note(notes_sorted[i]) for i in indices]
        start_time = min(m.start_time for m in candidate_members)
        end_time = max(m.end_time for m in candidate_members)
        start_freq = min(m.start_freq for m in candidate_members)
        end_freq = max(m.end_freq for m in candidate_members)
        clusters.append(CandidateCluster(
            kind=kind,                              # type: ignore[arg-type]
            score=0.0,
            start_time=start_time,
            end_time=end_time,
            start_freq=start_freq,
            end_freq=end_freq,
            members=candidate_members,
        ))

    return clusters


def triage_notes(
    notes: list[NoteEvent],
    min_duration: float = 0.5,
    max_candidates: int = 10,
    start_time: float | None = None,
    end_time: float | None = None,
) -> NoteTriageFileData:
    """Three-pass triage: cluster → score → select.

    Optional `start_time`/`end_time` restrict which notes are considered as
    candidates (a note is included iff its start_time is within the window).
    The polyphony profile is built from ALL notes regardless of window so
    the polyphony at window boundaries reflects notes that started outside
    but were still sounding inside.
    """
    # Polyphony profile uses all notes — including ones that started outside
    # the analysis window but still affect polyphony at the window edges.
    profile = _build_polyphony_profile(notes)

    # Filter for clustering: only notes whose onset falls inside the window
    # become candidates. Notes that started outside contribute to the profile
    # but cannot themselves be candidates.
    if start_time is not None:
        notes = [n for n in notes if n.start_time >= start_time]
    if end_time is not None:
        notes = [n for n in notes if n.start_time < end_time]

    # Pluck filter
    notes = [n for n in notes if (n.end_time - n.start_time) >= min_duration]

    if not notes:
        return NoteTriageFileData(polyphony_profile=profile, candidates=[])

    # Pass 1: clustering
    clusters = _cluster_notes(notes)

    # Pass 2: score each cluster
    scored: list[tuple[CandidateCluster, float]] = [
        (c, _score_cluster(c, profile, clusters)) for c in clusters
    ]

    # Pass 3: drop arpeggios, sort, apply pitch-diversity penalty greedily, take top N
    scored = [(c, s) for c, s in scored if c.kind != "arpeggio"]
    scored.sort(key=lambda x: x[1], reverse=True)

    selected: list[tuple[CandidateCluster, float]] = []
    selected_pitches: list[int] = []
    for cluster, score in scored:
        if len(selected) >= max_candidates:
            break
        rep_pitch = cluster.members[0].note.pitch_midi  # highest-scoring member by sort order
        if any(abs(rep_pitch - p) <= 2 for p in selected_pitches):
            score *= 0.5
        selected.append((cluster, score))
        selected_pitches.append(rep_pitch)

    selected.sort(key=lambda x: x[1], reverse=True)
    selected = selected[:max_candidates]

    candidates = [
        c.model_copy(update={"score": round(s, 4)}) for c, s in selected
    ]
    return NoteTriageFileData(polyphony_profile=profile, candidates=candidates)
