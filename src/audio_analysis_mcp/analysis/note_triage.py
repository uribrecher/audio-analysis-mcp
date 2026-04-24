import librosa
import numpy as np

from audio_analysis_mcp.schemas import (
    NoteEvent,
    PolyphonyWindow,
    CandidateNote,
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
