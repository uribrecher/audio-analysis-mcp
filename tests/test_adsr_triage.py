from audio_analysis_mcp.analysis.adsr_triage import classify_adsr_triage
from audio_analysis_mcp.schemas import AmplitudeTriage, NoteEvent


def _note(start: float, end: float, pitch: int = 60, amp: float = 0.8) -> NoteEvent:
    return NoteEvent(
        start_time=start,
        end_time=end,
        pitch_midi=pitch,
        amplitude=amp,
        pitch_bends=None,
    )


def test_empty_notes_rejected():
    assert classify_adsr_triage([]) == AmplitudeTriage.REJECTED


def test_single_note_is_monophonic():
    assert classify_adsr_triage([_note(0.0, 1.0)]) == AmplitudeTriage.MONOPHONIC


def test_simultaneous_chord_is_block_chord():
    notes = [_note(0.0, 1.0, 60), _note(0.01, 1.0, 64), _note(0.02, 0.99, 67)]
    assert classify_adsr_triage(notes) == AmplitudeTriage.BLOCK_CHORD


def test_dense_arpeggio_is_rejected():
    # 8 notes in 1 second = 8 onsets/sec
    notes = [_note(i * 0.125, i * 0.125 + 0.1) for i in range(8)]
    assert classify_adsr_triage(notes) == AmplitudeTriage.REJECTED


def test_sequential_two_notes_is_monophonic():
    notes = [_note(0.0, 0.5), _note(0.6, 1.1)]
    assert classify_adsr_triage(notes) == AmplitudeTriage.MONOPHONIC
