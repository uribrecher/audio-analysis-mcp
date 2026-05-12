"""Tests for analysis.note_triage — pure logic, no mocks needed."""
import pytest
import librosa

from audio_analysis_mcp.analysis.note_triage import triage_notes, triage_notes_by_sections
from audio_analysis_mcp.schemas import NoteEvent, TriageSection


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


def test_cluster_jittered_chord_still_chord():
    # Real-world chord with onset jitter beyond the old 30ms tolerance.
    # All three notes overlap during a common interval [1.10, 1.40] → chord.
    notes = [
        _ev(1.00, 1.50, 60),  # C4
        _ev(1.10, 1.55, 64),  # E4 (onset 100ms after C4)
        _ev(1.05, 1.40, 67),  # G4 (onset 50ms after C4)
    ]
    data = triage_notes(notes, min_duration=0.0)
    assert len(data.candidates) == 1
    assert data.candidates[0].kind == "chord"
    assert len(data.candidates[0].members) == 3


def test_cluster_overlap_with_no_common_moment_is_arpeggio():
    # Three notes connected by pairwise overlap but no instant when all three
    # are sounding together. A ends before C starts; only B bridges them.
    notes = [
        _ev(0.0, 0.5, 60),    # A
        _ev(0.3, 1.0, 64),    # B  (overlaps A in [0.3, 0.5] and C in [0.6, 1.0])
        _ev(0.6, 1.2, 67),    # C
    ]
    data = triage_notes(notes, min_duration=0.0, max_candidates=10)
    # All three connected → one component. No common interval → arpeggio.
    # Arpeggios are filtered, so no candidates.
    assert all(c.kind != "arpeggio" for c in data.candidates)
    # And no other clusters exist either, so output should be empty.
    assert data.candidates == []


def test_cluster_overlapping_note_is_not_a_single():
    # The bug the user found: a note that overlaps with another note must
    # never be classified as 'single'.
    notes = [
        _ev(0.0, 1.0, 60),    # held note
        _ev(0.3, 0.5, 64),    # short note inside it
    ]
    data = triage_notes(notes, min_duration=0.0)
    assert len(data.candidates) == 1
    assert data.candidates[0].kind == "chord"  # both members share [0.3, 0.5]
    assert len(data.candidates[0].members) == 2


def test_jitter_tolerance_default_is_strict():
    # Two notes that don't quite overlap (5ms gap between A's end and B's start).
    # Default jitter_tolerance=0.0 → they remain separate singles.
    notes = [_ev(0.0, 1.0, 60), _ev(1.005, 2.0, 64)]
    data = triage_notes(notes, min_duration=0.0, max_candidates=10)
    assert len(data.candidates) == 2
    assert all(c.kind == "single" for c in data.candidates)


def test_jitter_tolerance_groups_near_overlap():
    # Same two notes — with a 30ms jitter tolerance they get linked.
    # The two notes' effective ranges overlap so they form a 2-note cluster.
    # Common interval check: max(starts) - min(ends) = 1.005 - 1.0 = 0.005s
    # < jitter (0.030) → still classified as a chord.
    notes = [_ev(0.0, 1.0, 60), _ev(1.005, 2.0, 64)]
    data = triage_notes(notes, min_duration=0.0, max_candidates=10, jitter_tolerance=0.030)
    assert len(data.candidates) == 1
    assert data.candidates[0].kind == "chord"
    assert len(data.candidates[0].members) == 2


def test_jitter_tolerance_jittered_chord_onsets():
    # A chord struck simultaneously but transcribed with onset jitter beyond
    # the strict overlap window. With jitter_tolerance=0.0 the algorithm can
    # still recover it (members all overlap). With jitter_tolerance=0.1 it
    # also recovers chords with broader detection slack.
    notes = [
        _ev(1.0, 1.05, 60),   # very short — ends before B/C even start
        _ev(1.10, 1.50, 64),  # starts AFTER A ends — strict gap 50ms
        _ev(1.15, 1.50, 67),  # starts AFTER A ends — strict gap 100ms
    ]
    # Strict: A doesn't overlap B or C; B and C share [1.15, 1.50] → chord of B+C, plus a single A.
    strict = triage_notes(notes, min_duration=0.0, max_candidates=10, jitter_tolerance=0.0)
    assert len(strict.candidates) == 2
    # With 100ms jitter, A connects to B (50ms gap), and (B-C share an interval directly):
    # All three end up in one cluster.
    relaxed = triage_notes(notes, min_duration=0.0, max_candidates=10, jitter_tolerance=0.100)
    assert len(relaxed.candidates) == 1


def test_polyphony_profile_includes_notes_outside_window():
    # A long note from before the window contributes to polyphony at the window edge.
    notes = [
        _ev(0.0, 5.0, 60, amp=0.8),    # spans whole song
        _ev(2.0, 2.5, 64, amp=0.8),    # candidate inside the window
    ]
    data = triage_notes(notes, min_duration=0.0, start_time=2.0, end_time=3.0)
    # Only the second note's onset is inside [2.0, 3.0] → only it can be a candidate.
    pitches = [c.members[0].note.pitch_midi for c in data.candidates]
    assert pitches == [64]
    # But the polyphony profile should reflect the long note too, so polyphony
    # at t≈2.0 should be 2, not 1.
    bucket_at_2 = next(w for w in data.polyphony_profile if w.start_time <= 2.0 < w.end_time)
    assert bucket_at_2.note_count == 2


class TestTriageBySections:
    """Per-section triage shares the underlying single-window logic; these
    tests pin the array-shape, index alignment, label preservation, and
    polyphony-profile slicing rather than re-testing the scoring math."""

    def test_empty_sections(self):
        notes = [_ev(0.0, 1.0, 60)]
        out = triage_notes_by_sections(notes, sections=[], min_duration=0.0)
        assert out.sections == []

    def test_section_with_no_notes_kept_in_output(self):
        """An empty section must NOT be skipped — array index has to stay
        aligned with the input section list so a consumer can pair them
        positionally with the original SongFormer segments."""
        notes = [_ev(0.0, 0.8, 60)]  # all activity in [0, 0.8]
        sections = [
            TriageSection(start_time=0.0, end_time=1.0, label="intro"),
            TriageSection(start_time=10.0, end_time=12.0, label="outro"),  # empty
        ]
        out = triage_notes_by_sections(notes, sections=sections, min_duration=0.0)
        assert len(out.sections) == 2
        assert out.sections[1].label == "outro"
        assert out.sections[1].candidates == []

    def test_label_and_index_preserved(self):
        notes = [
            _ev(0.5, 0.9, 60), _ev(1.5, 1.9, 62),
            _ev(2.5, 2.9, 64), _ev(3.5, 3.9, 67),
        ]
        sections = [
            TriageSection(start_time=0.0, end_time=2.0, label="verse"),
            TriageSection(start_time=2.0, end_time=4.0, label="chorus"),
        ]
        out = triage_notes_by_sections(notes, sections=sections, min_duration=0.0)
        assert [s.index for s in out.sections] == [0, 1]
        assert [s.label for s in out.sections] == ["verse", "chorus"]
        assert (out.sections[0].start_time, out.sections[0].end_time) == (0.0, 2.0)
        assert (out.sections[1].start_time, out.sections[1].end_time) == (2.0, 4.0)

    def test_candidates_filtered_by_section_window(self):
        """A note's ONSET determines which section's candidate list it can
        belong to — same semantics as the single-window `start_time`/`end_time`
        filter in triage_notes."""
        notes = [
            _ev(0.5, 0.9, 60),    # in verse
            _ev(2.5, 2.9, 64),    # in chorus
            _ev(5.5, 5.9, 67),    # outside both
        ]
        sections = [
            TriageSection(start_time=0.0, end_time=2.0, label="verse"),
            TriageSection(start_time=2.0, end_time=4.0, label="chorus"),
        ]
        out = triage_notes_by_sections(notes, sections=sections, min_duration=0.0)
        verse_pitches = {c.members[0].note.pitch_midi for c in out.sections[0].candidates}
        chorus_pitches = {c.members[0].note.pitch_midi for c in out.sections[1].candidates}
        assert verse_pitches == {60}
        assert chorus_pitches == {64}

    def test_polyphony_profile_trimmed_to_section(self):
        """Each section's persisted polyphony_profile must be the slice of
        the whole-song profile that overlaps the section — not the full
        profile duplicated across every entry."""
        notes = [
            _ev(0.0, 10.0, 60),   # spans whole song; contributes to polyphony everywhere
            _ev(1.0, 1.2, 64),
            _ev(5.0, 5.2, 67),
        ]
        sections = [
            TriageSection(start_time=0.0, end_time=2.0, label="a"),
            TriageSection(start_time=4.0, end_time=6.0, label="b"),
        ]
        out = triage_notes_by_sections(notes, sections=sections, min_duration=0.0)
        # Each section's profile lives within (or straddling) its own window only.
        for entry in out.sections:
            for w in entry.polyphony_profile:
                assert w.end_time > entry.start_time
                assert w.start_time < entry.end_time