"""Tests for analysis.note_triage — pure logic, no mocks needed."""
import pytest
import librosa

from audio_analysis_mcp.analysis.note_triage import triage_notes
from audio_analysis_mcp.schemas import NoteEvent


def _note(start: float, end: float, pitch: int, amp: float = 0.8) -> NoteEvent:
    return NoteEvent(
        start_time=start, end_time=end,
        pitch_midi=pitch, amplitude=amp, pitch_bends=None,
    )


class TestPolyphonyProfile:
    def test_sequential_monophonic(self):
        """Three non-overlapping notes → all windows show count <= 1."""
        notes = [_note(0.0, 0.8, 60), _note(1.0, 1.8, 64), _note(2.0, 2.8, 67)]
        result = triage_notes(notes)
        for window in result.polyphony_profile:
            assert window.note_count <= 1

    def test_overlapping_notes(self):
        """Two notes overlapping in time → at least one window with count 2."""
        notes = [_note(0.0, 1.0, 60), _note(0.5, 1.5, 64)]
        result = triage_notes(notes)
        max_count = max(w.note_count for w in result.polyphony_profile)
        assert max_count == 2

    def test_windows_cover_full_range(self):
        """Profile windows span from 0 to past the last note offset."""
        notes = [_note(0.0, 0.8, 60), _note(2.0, 3.0, 67)]
        result = triage_notes(notes)
        assert result.polyphony_profile[0].start_time == 0.0
        assert result.polyphony_profile[-1].end_time >= 3.0


from audio_analysis_mcp.analysis.note_triage import _cluster_notes


def _ev(start: float, end: float, pitch: int = 60, amp: float = 0.8) -> NoteEvent:
    return NoteEvent(
        start_time=start, end_time=end, pitch_midi=pitch, amplitude=amp, pitch_bends=None
    )


def test_cluster_empty():
    assert _cluster_notes([]) == []


def test_cluster_single_note():
    clusters = _cluster_notes([_ev(0.0, 1.0)])
    assert len(clusters) == 1
    assert clusters[0].kind == "single"
    assert len(clusters[0].members) == 1


def test_cluster_chord_three_simultaneous():
    notes = [_ev(0.0, 1.0, 60), _ev(0.01, 1.0, 64), _ev(0.02, 0.99, 67)]
    clusters = _cluster_notes(notes)
    assert len(clusters) == 1
    assert clusters[0].kind == "chord"
    assert len(clusters[0].members) == 3


def test_cluster_arpeggio_six_notes():
    # 6 notes spaced 100 ms apart: each onset gap is 100 ms ≤ 150 ms threshold
    notes = [_ev(i * 0.1, i * 0.1 + 0.4, 60 + i) for i in range(6)]
    clusters = _cluster_notes(notes)
    assert len(clusters) == 1
    assert clusters[0].kind == "arpeggio"
    assert len(clusters[0].members) == 6


def test_cluster_arpeggio_minimum_size_3():
    # Two sequential notes are NOT an arpeggio → two singles
    notes = [_ev(0.0, 0.4, 60), _ev(0.45, 0.85, 62)]
    clusters = _cluster_notes(notes)
    assert {c.kind for c in clusters} == {"single"}
    assert len(clusters) == 2


def test_cluster_mixed_chord_and_single():
    # A chord at t=0..1 + a sequential single note at t=2..3
    notes = [_ev(0.0, 1.0, 60), _ev(0.0, 1.0, 64), _ev(2.0, 3.0, 72)]
    clusters = _cluster_notes(notes)
    assert len(clusters) == 2
    assert {c.kind for c in clusters} == {"chord", "single"}


def test_cluster_arpeggio_breaks_on_long_gap():
    # 4 notes with onsets at 0.0, 0.1, 0.5 (gap 400ms > 150ms), 0.6 → first 2 too short for arpeggio,
    # last 2 too short → 4 singles
    notes = [_ev(0.0, 0.1), _ev(0.1, 0.2), _ev(0.5, 0.6), _ev(0.6, 0.7)]
    clusters = _cluster_notes(notes)
    assert all(c.kind == "single" for c in clusters)
    assert len(clusters) == 4


from audio_analysis_mcp.analysis.note_triage import _score_cluster, _build_polyphony_profile
from audio_analysis_mcp.schemas import CandidateNote, CandidateCluster


def _build_candidate_note_for_test(start: float, end: float, pitch: int, amp: float = 0.8) -> CandidateNote:
    return CandidateNote(
        note=NoteEvent(start_time=start, end_time=end, pitch_midi=pitch, amplitude=amp, pitch_bends=None),
        score=0.0, start_time=start, end_time=end, start_freq=200.0, end_freq=2000.0,
    )


def test_score_single_outscores_chord_outscores_arpeggio():
    # Three identical-shape clusters differing only in kind.
    clusters = []
    for kind in ("single", "chord", "arpeggio"):
        members = [_build_candidate_note_for_test(0.0, 1.0, 60)]
        clusters.append(CandidateCluster(
            kind=kind, score=0.0,
            start_time=0.0, end_time=1.0, start_freq=200.0, end_freq=2000.0,
            members=members,
        ))
    profile = _build_polyphony_profile([_ev(0.0, 1.0)])
    s_single = _score_cluster(clusters[0], profile, clusters)
    s_chord = _score_cluster(clusters[1], profile, clusters)
    s_arp = _score_cluster(clusters[2], profile, clusters)
    assert s_single > s_chord > s_arp


def test_score_velocity_helps():
    soft = CandidateCluster(
        kind="single", score=0.0, start_time=0.0, end_time=1.0,
        start_freq=200.0, end_freq=2000.0,
        members=[_build_candidate_note_for_test(0.0, 1.0, 60, amp=0.1)],
    )
    loud = CandidateCluster(
        kind="single", score=0.0, start_time=2.0, end_time=3.0,
        start_freq=200.0, end_freq=2000.0,
        members=[_build_candidate_note_for_test(2.0, 3.0, 60, amp=0.9)],
    )
    profile = _build_polyphony_profile([_ev(0.0, 1.0), _ev(2.0, 3.0)])
    assert _score_cluster(loud, profile, [soft, loud]) > _score_cluster(soft, profile, [soft, loud])


def test_score_longer_duration_helps():
    short = CandidateCluster(
        kind="single", score=0.0, start_time=0.0, end_time=0.6,
        start_freq=200.0, end_freq=2000.0,
        members=[_build_candidate_note_for_test(0.0, 0.6, 60)],
    )
    long_c = CandidateCluster(
        kind="single", score=0.0, start_time=2.0, end_time=4.0,
        start_freq=200.0, end_freq=2000.0,
        members=[_build_candidate_note_for_test(2.0, 4.0, 60)],
    )
    profile = _build_polyphony_profile([_ev(0.0, 0.6), _ev(2.0, 4.0)])
    assert _score_cluster(long_c, profile, [short, long_c]) > _score_cluster(short, profile, [short, long_c])


def test_triage_returns_clusters_sorted_by_score():
    notes = [
        _ev(0.0, 0.3, 60, amp=0.2),       # short low-velocity single → low score
        _ev(2.0, 4.0, 64, amp=0.9),       # long high-velocity single → high score
        _ev(6.0, 7.0, 67, amp=0.6),       # medium single
    ]
    data = triage_notes(notes, min_duration=0.0, max_candidates=10)
    assert len(data.candidates) == 3
    # Highest score first
    assert data.candidates[0].kind == "single"
    assert data.candidates[0].members[0].note.pitch_midi == 64


def test_triage_filters_arpeggios():
    # Two singles + one arpeggio of 4 notes
    notes = [
        _ev(0.0, 1.0, 60),                       # single
        _ev(2.0, 3.0, 64),                       # single
        _ev(5.0, 5.2, 60), _ev(5.15, 5.35, 62),  # part of arpeggio
        _ev(5.30, 5.50, 64), _ev(5.45, 5.65, 65),
    ]
    data = triage_notes(notes, min_duration=0.0, max_candidates=10)
    assert all(c.kind != "arpeggio" for c in data.candidates)


def test_triage_min_duration_filters_short_notes():
    notes = [_ev(0.0, 0.3, 60), _ev(1.0, 2.0, 64)]
    data = triage_notes(notes, min_duration=0.5)
    assert len(data.candidates) == 1
    assert data.candidates[0].members[0].note.pitch_midi == 64


def test_triage_respects_time_window():
    notes = [_ev(0.0, 1.0, 60), _ev(5.0, 6.0, 64), _ev(10.0, 11.0, 67)]
    data = triage_notes(notes, min_duration=0.0, start_time=4.0, end_time=8.0, max_candidates=10)
    assert len(data.candidates) == 1
    assert data.candidates[0].members[0].note.pitch_midi == 64


def test_triage_respects_max_candidates():
    notes = [_ev(i * 2.0, i * 2.0 + 1.0, 60 + i) for i in range(15)]
    data = triage_notes(notes, min_duration=0.0, max_candidates=5)
    assert len(data.candidates) == 5


def test_triage_pitch_diversity_penalty():
    # Two equal-shape singles at the same pitch → second one penalized.
    # Plus one cluster at a different pitch → should win over the second same-pitch one.
    notes = [
        _ev(0.0, 1.0, 60, amp=0.9),
        _ev(2.0, 3.0, 60, amp=0.9),
        _ev(4.0, 5.0, 72, amp=0.5),
    ]
    data = triage_notes(notes, min_duration=0.0, max_candidates=3)
    pitches = [c.members[0].note.pitch_midi for c in data.candidates]
    # The first same-pitch instance ranks above the diverse-pitch cluster (because amp=0.9 vs 0.5),
    # but the second same-pitch instance is penalized below the diverse one.
    assert pitches.index(60) < pitches.index(72) < pitches.index(60, pitches.index(60) + 1)


def test_triage_empty_notes_returns_empty():
    data = triage_notes([])
    assert data.candidates == []
    assert data.polyphony_profile == []