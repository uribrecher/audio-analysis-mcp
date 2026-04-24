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


class TestCandidateSelection:
    def test_prefer_monophonic_over_polyphonic(self):
        """A note in a monophonic window scores higher than one in polyphonic."""
        mono_note = _note(0.0, 1.0, 60)
        poly_note1 = _note(2.0, 3.0, 64)
        poly_note2 = _note(2.2, 3.0, 67)  # overlaps with poly_note1
        result = triage_notes([mono_note, poly_note1, poly_note2])
        # The monophonic note (pitch 60) should be ranked first
        assert result.candidates[0].note.pitch_midi == 60

    def test_min_duration_filter(self):
        """Notes shorter than min_duration are excluded."""
        short = _note(0.0, 0.3, 60)  # 0.3s — below default 0.5s threshold
        long = _note(1.0, 2.0, 64)   # 1.0s — above threshold
        result = triage_notes([short, long], min_duration=0.5)
        pitches = [c.note.pitch_midi for c in result.candidates]
        assert 64 in pitches
        assert 60 not in pitches

    def test_max_candidates_limits_output(self):
        """No more than max_candidates returned."""
        notes = [_note(float(i), float(i) + 0.8, 60 + i) for i in range(20)]
        result = triage_notes(notes, max_candidates=5)
        assert len(result.candidates) <= 5

    def test_frequency_bounds_from_midi(self):
        """Candidate freq bounds: lower = fundamental*0.9, upper = min(fundamental*8, 10kHz)."""
        note = _note(0.0, 1.0, 69)  # A4 = 440 Hz
        result = triage_notes([note])
        assert len(result.candidates) == 1
        c = result.candidates[0]
        fundamental = librosa.midi_to_hz(69)  # 440.0
        assert c.start_freq == pytest.approx(fundamental * 0.9, rel=0.01)
        assert c.end_freq == pytest.approx(min(fundamental * 8, 10000.0), rel=0.01)

    def test_time_bounds_include_padding(self):
        """Candidate time bounds include 50ms padding before and after."""
        note = _note(1.0, 2.0, 60)
        result = triage_notes([note])
        c = result.candidates[0]
        assert c.start_time == pytest.approx(0.95, abs=0.01)
        assert c.end_time == pytest.approx(2.05, abs=0.01)

    def test_time_padding_clamps_to_zero(self):
        """Start time padding doesn't go below 0."""
        note = _note(0.01, 1.0, 60)
        result = triage_notes([note])
        assert result.candidates[0].start_time >= 0.0

    def test_empty_notes(self):
        """Empty input → empty profile and candidates."""
        result = triage_notes([])
        assert result.polyphony_profile == []
        assert result.candidates == []
